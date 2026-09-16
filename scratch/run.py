"""Experiment driver for the reproduction of arXiv:2608.08770 section 4.1 and the Proteina
steering phases. Configured from YAML: see DRST/configs/.
"""
import json
import math
import os
import sys

import torch
import yaml

from steering.ips import normalise
from steering.sde import Config, MMD, Pointwise, multibatch, steer
from steering.toy import (GaussianMixtureFlow, TrainedFlow, grid_mu_star,
                          objective_paper, sample_grid, w1_to_grid)


def out_path(cfg, part, tag=""):
    os.makedirs(cfg["out_dir"], exist_ok=True)
    return os.path.join(cfg["out_dir"], f"{part}_sd{cfg['sd']:g}{tag}.json")


# ---------------------------------------------------------------- toy setup
def base_model(cfg, sched="const"):
    """p1: what the flow model is trained to sample, and the exact denoiser for reference."""
    return GaussianMixtureFlow(eps_mode=sched, eps_clamp=cfg["eps_clamp"], means=tuple(cfg["means"]),
                               stds=(cfg["sd"], cfg["sd"]), weights=tuple(cfg["p1_weights"]))


def trained(cfg, sched="const"):
    """**[paper]** C.1 trains a flow-matching model; the exact denoiser is a different function,
    so the reproduction uses the trained one."""
    m = TrainedFlow(base_model(cfg, sched), width=cfg["width"], eps_mode=sched,
                    eps_clamp=cfg["eps_clamp"])
    m.net.load_state_dict(torch.load(cfg["ckpt"], weights_only=True))
    for p in m.net.parameters():
        p.requires_grad_(False)
    return m


def target_sample(cfg):
    """nu: the tilting measure. Same means, weights reversed."""
    tgt = GaussianMixtureFlow(means=tuple(cfg["means"]), stds=(cfg["sd"], cfg["sd"]),
                              weights=tuple(cfg["nu_weights"]))
    return tgt.sample_p1(cfg["nu_samples"],
                         torch.Generator().manual_seed(cfg["nu_seed"])).reshape(-1, 1)


def _series(cfg):
    """Every curve in Figure 3, as Config kwargs from the axis value x."""
    return {
        "kfp1": lambda x: dict(lam=x, k_fp=1, damp=1.0),
        "kfp3": lambda x: dict(lam=x, k_fp=3, damp=1.0),
        "kfp5": lambda x: dict(lam=x, k_fp=5, damp=1.0),
        "kfp50": lambda x: dict(lam=x, k_fp=50, damp=cfg["converged_damp"]),
        "implicit": lambda x: dict(lam=x, psidot_solver="implicit"),
        "gradient": lambda x: dict(lam=x, reweight=False),
        "gradient_p04": lambda x: dict(lam=x, reweight=False,
                                       steer_coeff=cfg["p04_range"] / cfg["lams"][-1]
                                       * cfg["lam_star"]),
    }


# ---------------------------------------------------------------- stages
def train(cfg):
    if os.path.exists(cfg["ckpt"]):
        print(f"{cfg['ckpt']} exists; skipping", flush=True)
        return
    m = TrainedFlow(base_model(cfg), width=cfg["width"], eps_mode="const",
                    eps_clamp=cfg["eps_clamp"]).fit(
        steps=cfg["train_steps"], batch=cfg["train_batch"], lr=cfg["train_lr"], seed=0,
        verbose=True)
    torch.save(m.net.state_dict(), cfg["ckpt"])
    ref, g = base_model(cfg), torch.Generator().manual_seed(1)
    print(f"\nvelocity error vs the exact denoiser (sd={cfg['sd']:g})", flush=True)
    for t in (0.1, 0.3, 0.5, 0.7, 0.9, 0.99):
        x = torch.randn(4000, 1, generator=g)
        tt = torch.full((4000,), t)
        _, ve = ref.endpoint(x, tt)
        _, vt = m.endpoint(x, tt)
        e, s = float((vt - ve).abs().mean()), float(ve.abs().mean())
        print(f"  t={t:4.2f}  mean|dv| {e:.4f}  rel {e / s:.4f}", flush=True)
    print(f"wrote {cfg['ckpt']}", flush=True)


def fig2(cfg):
    """Trajectories and final measures at lam*, sigma_t = 1 (the paper's Figure 2)."""
    n, lam_star = cfg["n"], cfg["lam_star"]
    m, y = trained(cfg, "const"), target_sample(cfg)
    grid, q = grid_mu_star(base_model(cfg), y, lam_star, cfg["h"])
    pot = MMD(y, cfg["h"])
    unif = torch.full((n,), 1.0 / n)
    nu = GaussianMixtureFlow(means=tuple(cfg["means"]), stds=(cfg["sd"], cfg["sd"]),
                             weights=tuple(cfg["nu_weights"]))
    out = {"grid": grid.tolist(), "mu_star": q.tolist(), "sd": cfg["sd"], "h": cfg["h"], "N": n,
           "p1": torch.softmax(base_model(cfg).log_p1(grid), 0).tolist(),
           "nu": torch.softmax(nu.log_p1(grid), 0).tolist()}
    out["w1_floor"] = sum(
        w1_to_grid(sample_grid(grid, q, n, torch.Generator().manual_seed(s)), unif, grid, q)
        for s in range(8)) / 8.0
    runs = {"base": dict(lam=0.0, resample_every=0),
            # **[paper]** C.1: "3 fixed-point iterations for the visualisation"
            "meanfield": dict(lam=lam_star, k_fp=3, damp=1.0,
                              resample_every=cfg["resample_fig2"]),
            "gradient": dict(lam=lam_star, reweight=False, resample_every=0)}
    for k, kw in runs.items():
        o = steer(m, pot, Config(nsteps=cfg["steps"], seed=cfg["fig2_seed"],
                                 record_traj=cfg["record_traj"], **kw), n)
        w = o["w"] if kw.get("reweight", True) else unif
        out[f"x_{k}"], out[f"w_{k}"] = o["x"].reshape(-1).tolist(), w.reshape(-1).tolist()
        out[f"traj_{k}"], out["ts"] = o["traj"].tolist(), o["ts"].tolist()
        out[f"w1_{k}"] = w1_to_grid(o["x"], w, grid, q)
        print(f"  {k:10s} W1 to mu* = {out[f'w1_{k}']:.4f}", flush=True)
    json.dump(out, open(out_path(cfg, "fig2"), "w"))
    print(f"wrote {out_path(cfg, 'fig2')}  (W1 floor {out['w1_floor']:.4f})", flush=True)


def fig3(cfg, sched, only=None):
    """Objective gap vs steering strength, one noise schedule (the paper's Figure 3)."""
    n, lam_star = cfg["n"], cfg["lam_star"]
    m, y = trained(cfg, sched), target_sample(cfg)
    grid, q = grid_mu_star(base_model(cfg), y, lam_star, cfg["h"])
    pot = MMD(y, cfg["h"])
    unif = torch.full((n,), 1.0 / n)
    fn = out_path(cfg, "fig3", f"_{sched}")
    out = json.load(open(fn)) if (only and os.path.exists(fn)) else {}
    out.update(sd=cfg["sd"], h=cfg["h"], N=n, steps=cfg["steps"], sched=sched,
               lams=cfg["lams"], lam_star=lam_star)
    out["J_star"] = sum(
        objective_paper(sample_grid(grid, q, n, torch.Generator().manual_seed(s)), unif,
                        base_model(cfg), y, lam_star, cfg["h"], cfg["kl_bw"])
        for s in range(cfg["jstar_seeds"])) / cfg["jstar_seeds"]
    print(f"[{sched}] J(mu*) = {out['J_star']:.5f}", flush=True)
    series = _series(cfg)
    for name in ([only] if only else series):
        spec = series[name]
        J, W1 = [], []
        for lam in cfg["lams"]:
            js, ws = [], []
            for sd in cfg["seeds"]:
                o = steer(m, pot, Config(nsteps=cfg["steps"], seed=sd,
                                         resample_every=(0 if name.startswith("gradient")
                                                         else cfg["resample_fig3"]),
                                         **spec(lam)), n)
                w = o["w"] if not name.startswith("gradient") else unif
                js.append(objective_paper(o["x"], w, base_model(cfg), y, lam_star, cfg["h"],
                                          cfg["kl_bw"]))
                ws.append(w1_to_grid(o["x"], w, grid, q))
            J.append(sum(js) / len(js)); W1.append(sum(ws) / len(ws))
            print(f"  {name:13s} lam={lam:5.1f} gap={J[-1] - out['J_star']:+9.5f} "
                  f"W1={W1[-1]:.4f}", flush=True)
        out[f"J_{name}"], out[f"W1_{name}"] = J, W1
    json.dump(out, open(fn, "w"))
    print(f"wrote {fn}", flush=True)


def pooling(cfg):
    """Algorithm 2: does pooling the measure across batches beat the same number of INDEPENDENT
    samples?
    """
    from dataclasses import replace
    lam = cfg["lam_star"]
    m, y = trained(cfg, "const"), target_sample(cfg)
    grid, q = grid_mu_star(base_model(cfg), y, lam, cfg["h"])
    pot = MMD(y, cfg["h"])
    out = {"sd": cfg["sd"], "h": cfg["h"], "lam": lam, "nseeds": cfg["pool_seeds"], "rows": []}
    print(f"{cfg['pool_seeds']} seeds per cell; mean +- sd of W1 to mu*", flush=True)
    print(f"{'N':>5} {'M':>3} {'total':>6}  {'pooled':>18} {'independent':>18}  {'ratio':>6}",
          flush=True)
    for n in cfg["pool_n"]:
        for M in cfg["pool_M"]:
            wp, wi = [], []
            for s in range(cfg["pool_seeds"]):
                c = Config(nsteps=cfg["steps"], lam=lam, k_fp=50, damp=cfg["converged_damp"],
                           resample_every=cfg["resample_fig3"], seed=100 * s)
                wp.append(w1_to_grid(*(lambda o: (o["x"], o["w"]))(
                    multibatch(m, pot, c, n, M)), grid, q))
                xs, as_ = [], []
                for b in range(M):
                    ob = steer(m, pot, replace(c, seed=c.seed + b), n)
                    xs.append(ob["x"]); as_.append(ob["a"])
                wi.append(w1_to_grid(torch.cat(xs), normalise(torch.cat(as_)), grid, q))
            P, I = torch.tensor(wp), torch.tensor(wi)
            row = dict(n=n, M=M, pooled=float(P.mean()), pooled_sd=float(P.std()),
                       indep=float(I.mean()), indep_sd=float(I.std()),
                       ratio=float(P.mean() / I.mean()), wins=int((P < I).sum()))
            out["rows"].append(row)
            print(f"{n:5d} {M:3d} {n * M:6d}  {row['pooled']:8.4f}+-{row['pooled_sd']:.4f}"
                  f"{'':<4} {row['indep']:8.4f}+-{row['indep_sd']:.4f}{'':<4} "
                  f"{row['ratio']:6.3f}   pooled wins {row['wins']}/{cfg['pool_seeds']}",
                  flush=True)
    unif = torch.full((cfg["n"],), 1.0 / cfg["n"])
    out["floor"] = sum(
        w1_to_grid(sample_grid(grid, q, cfg["n"], torch.Generator().manual_seed(s)), unif,
                   grid, q) for s in range(6)) / 6.0
    print(f"\nW1 Monte-Carlo floor at N={cfg['n']}: {out['floor']:.4f}", flush=True)
    json.dump(out, open(out_path(cfg, "pooling"), "w"), indent=1)


def plot(cfg):
    figures(lambda part, tag="": out_path(cfg, part, tag), cfg["sd"])


# ---------------------------------------------------------------- Proteina stages
def _proteina(cfg):
    from steering.proteina import ProteinaModel
    return ProteinaModel(ckpt=cfg["ckpt"], data_path=cfg["data_path"], n_res=cfg["n_res"],
                         n_particles=cfg["n"], gt_mode=cfg["gt_mode"], gt_p=cfg["gt_p"],
                         gt_clamp_val=cfg["gt_clamp_val"], nsteps=cfg["steps"])


def proteina_pointwise(cfg):
    """Phase 2: pointwise steering on the frozen checkpoint, against an absolute reference."""
    from steering.proteina import radius_of_gyration, ca_ca_spacing
    n, n_res = cfg["n"], cfg["n_res"]
    mdl = _proteina(cfg)
    dev = mdl.device

    def mask_for(k):
        """mdl.mask is sized for n_particles; the pooled reference set is larger, and slicing
        past the end silently returns the short tensor instead of raising."""
        return torch.ones(k, n_res, dtype=torch.bool, device=dev)

    reward = lambda th: radius_of_gyration(th, mask_for(th.shape[0]))
    pot = Pointwise(reward)

    def stats(x, w):
        """Rg is the reward; e2e and span are NOT controlled by it -- those are the ones that
        discriminate a sampler exploiting a gauge freedom from a correct one."""
        rg = reward(x)
        cc = ca_ca_spacing(x)
        e2e = (x[:, -1] - x[:, 0]).norm(dim=-1)
        span = torch.cdist(x, x).amax(dim=(-1, -2))
        w = (w / w.sum()).to(device=rg.device, dtype=rg.dtype)
        m = lambda v: float((w.to(v.dtype) * v).sum())
        return dict(rg=m(rg), rg2=m(rg ** 2), ca_ca=m(cc), e2e=m(e2e), span=m(span))

    print(f"N={n} n_res={n_res} steps={cfg['steps']} pool={cfg['pool_batches']} "
          f"steer={cfg['steer']}", flush=True)
    print(f"eps on the grid: [{float(mdl._eps.min()):.4g}, {float(mdl._eps.max()):.4g}]",
          flush=True)
    xs, rs = [], []
    for b in range(cfg["pool_batches"]):
        torch.manual_seed(1000 + b)
        o = steer(mdl, pot, Config(ts=mdl.ts, lam=0.0, resample_every=0,
                                   vf_tail_t=cfg["vf_tail_t"], global_rng=True), n)
        xs.append(o["x"].detach()); rs.append(reward(o["x"].detach()))
    X0, R0 = torch.cat(xs), torch.cat(rs)
    unif0 = torch.full((X0.shape[0],), 1.0 / X0.shape[0], device=X0.device)
    out = {"N": n, "n_res": n_res, "steer": cfg["steer"], "pool": int(X0.shape[0]),
           "base": stats(X0, unif0), "rg_sd": float(R0.std()), "runs": {},
           # Which sampler produced this. The published Table-1 config is sc_scale_noise 0.45,
           # gt_mode "1/t", gt_clamp_val null, dt 0.0025 (400 steps); mean-field steering needs
           # temperature 1, so a steered run is NOT the published sampler and must say so.
           "sampler": {k: cfg[k] for k in ("gt_mode", "gt_p", "gt_clamp_val", "steps",
                                           "vf_tail_t")}}
    print(f"unsteered pool: {X0.shape[0]} samples, Rg {float(R0.mean()):.4f}+-"
          f"{float(R0.std()):.4f} nm, CA-CA {out['base']['ca_ca']:.3f} A (native ~3.80)",
          flush=True)
    print(f"{'lam':>8} {'ESS/N':>7} {'|st|/|b|':>9} | {'Rg steered':>11} {'Rg SNIS':>9} "
          f"{'rel':>7} | {'CA-CA':>7} {'e2e st/IS':>14} {'span st/IS':>14} {'IS ESS':>7}",
          flush=True)
    for lam in cfg["lams"]:
        torch.manual_seed(7)
        o = steer(mdl, pot, Config(ts=mdl.ts, lam=lam, k_fp=1, damp=1.0, resample_every=0,
                                   vf_tail_t=cfg["vf_tail_t"],
                                   steer_coeff=(None if cfg["steer"] else 0.0),
                                   check_drift=True, global_rng=True), n)
        st = stats(o["x"].detach(), o["w"])
        a = lam * R0.double(); a = a - a.max()
        wi = torch.softmax(a, 0)
        ref = stats(X0, wi)
        is_ess = float(1.0 / (wi ** 2).sum() / wi.numel())
        ess = float(o["diag"][-1]["ess"]) / n
        sn = [d["steer_norm"] for d in o["diag"]]; bn = [d["base_norm"] for d in o["diag"]]
        ratio = max(s_ / b_ for s_, b_ in zip(sn, bn) if b_ > 0)
        rel = abs(st["rg"] - ref["rg"]) / max(1e-12, abs(ref["rg"]))
        out["runs"][f"{lam:g}"] = dict(steered=st, snis=ref, ess=ess, is_ess=is_ess,
                                       steer_over_base=ratio)
        print(f"{lam:8.0f} {ess:7.3f} {ratio:9.1e} | {st['rg']:11.4f} {ref['rg']:9.4f} "
              f"{rel:7.2%} | {st['ca_ca']:7.3f} {st['e2e']:6.3f}/{ref['e2e']:6.3f} "
              f"{st['span']:6.3f}/{ref['span']:6.3f} {is_ess:7.4f}", flush=True)
    tag = "" if cfg["steer"] else "_nosteer"
    json.dump(out, open(out_path(cfg, f"proteina_pointwise_r{n_res}", tag), "w"), indent=1)


def binning_gradient(cfg):
    """How much of grad_x Psi does the binned pair representation withhold?"""
    from proteinfoundation.nn import feature_factory as ff
    from steering.proteina import radius_of_gyration, soft_bin_pairwise_distances
    n, n_res = cfg["n"], cfg["n_res"]
    mdl = _proteina(cfg)
    reward = lambda th: radius_of_gyration(
        th, torch.ones(th.shape[0], n_res, dtype=torch.bool, device=mdl.device))
    hard = ff.bin_pairwise_distances

    def grad_psi(t_val):
        x = mdl.prior(n, None)
        xg = x.detach().requires_grad_(True)
        tg = torch.full((n,), t_val, device=x.device, dtype=x.dtype)
        theta, _ = mdl.endpoint(xg, tg)
        (cfg["lam"] * t_val * reward(theta)).sum().backward()
        return xg.grad.detach().clone()

    print(f"N={n} n_res={n_res} lam={cfg['lam']}  reward = radius of gyration", flush=True)
    print(f"{'t':>6} {'||g_hard||':>11} {'||g_soft||':>11} {'ratio':>7} {'cosine':>7}", flush=True)
    out = {"n_res": n_res, "N": n, "rows": [],
           "sampler": {k: cfg[k] for k in ("gt_mode", "gt_p", "gt_clamp_val", "steps")}}
    for t in cfg["ts"]:
        torch.manual_seed(0); ff.bin_pairwise_distances = hard
        gh = grad_psi(t)
        torch.manual_seed(0); ff.bin_pairwise_distances = soft_bin_pairwise_distances
        gs = grad_psi(t)
        ff.bin_pairwise_distances = hard
        nh, ns = float(gh.norm()), float(gs.norm())
        cos = float((gh.flatten() @ gs.flatten()) / (gh.norm() * gs.norm() + 1e-30))
        out["rows"].append(dict(t=t, hard=nh, soft=ns, ratio=ns / max(nh, 1e-30), cos=cos))
        print(f"{t:6.2f} {nh:11.4f} {ns:11.4f} {ns / max(nh, 1e-30):7.3f} {cos:7.4f}", flush=True)
    json.dump(out, open(out_path(cfg, f"binning_gradient_r{n_res}"), "w"), indent=1)


STAGES = {"train": train, "fig2": fig2, "fig3": fig3, "pooling": pooling, "plot": plot,
          "proteina_pointwise": proteina_pointwise, "binning_gradient": binning_gradient}


# ---------------------------------------------------------------------------- figures


SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983"
BLUE, GREEN, ORANGE, BROWN = "#2a78d6", "#008300", "#eb6834", "#9c6644"
RAMP = ["#86b6ef", "#2a78d6", "#104281"]
# Figure 3's curves. Both gradient-steering parameterisations are shown as peers: only the
# product coeff*lam enters, so gradient_p04 is `gradient` compressed 4x, and the 4 is the paper's
# display choice rather than a derived constant.
F3 = [("kfp1", RAMP[0], "o", (0, (4, 2)), r"Mean-field, $K_{fp}=1$"),
      ("kfp3", RAMP[1], "^", (0, (5, 1.5, 1, 1.5)), r"Mean-field, $K_{fp}=3$"),
      ("kfp5", RAMP[2], "s", (0, (1, 1.6)), r"Mean-field, $K_{fp}=5$"),
      ("kfp50", GREEN, "D", "-", r"Mean-field, converged ($K_{fp}=50$, damp .5)"),
      ("implicit", "#7a3fa0", "P", (0, (6, 1.5)), r"Mean-field, Algorithm 3 (implicit)"),
      ("gradient", ORANGE, "o", "-", r"Gradient steering, eq (11): $\varepsilon_t$"),
      ("gradient_p04", BROWN, "v", "-", r"Gradient steering, strength $[0,4]$ (paper's Fig 3)")]


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": .8,
                         "figure.facecolor": SURF, "axes.facecolor": SURF,
                         "legend.frameon": False, "axes.spines.top": False,
                         "axes.spines.right": False})
    return plt


def figures(out_path, sd):
    import numpy as np
    plt = _style()
    from matplotlib.lines import Line2D

    # ---- Figure 2: histogram strip over trajectories, t on the vertical axis
    f2 = json.load(open(out_path("fig2")))
    grid = np.array(f2["grid"]); ts = np.array(f2["ts"]); dx = grid[1] - grid[0]
    fig = plt.figure(figsize=(12.6, 4.6), facecolor=SURF)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 2.4], hspace=.04, wspace=.12,
                          left=.06, right=.985, top=.86, bottom=.17)
    bins = np.linspace(-4, 4, 61)
    for c, (key, title) in enumerate([("base", "Base Dynamics"),
                                      ("gradient", "Gradient Steering"),
                                      ("meanfield", "Mean-Field Steering")]):
        ah = fig.add_subplot(gs[0, c]); at = fig.add_subplot(gs[1, c], sharex=ah)
        for dens, col in ((f2["p1"], BLUE), (f2["nu"], GREEN)):
            ah.hist(np.random.default_rng(0).choice(grid, size=20000, p=np.array(dens)),
                    bins=bins, density=True, color=col, alpha=.55, lw=0)
        x, w = np.array(f2[f"x_{key}"]), np.array(f2[f"w_{key}"])
        ah.hist(x, bins=bins, weights=w / w.sum() / np.diff(bins)[0], color=ORANGE, alpha=.65, lw=0)
        ah.plot(grid, np.array(f2["mu_star"]) / dx, color=INK, lw=1.8)
        ah.set_title(title, color=INK, fontsize=10)
        ah.set_yticks([]); ah.tick_params(labelbottom=False)
        ah.spines["left"].set_visible(False)
        ah.text(.02, .86, f"$W_1$ to $\\mu^*$ = {f2[f'w1_{key}']:.3f}", transform=ah.transAxes,
                fontsize=7.6, color=INK2, va="top")
        at.plot(np.array(f2[f"traj_{key}"]), ts[:, None], color=ORANGE, lw=.4, alpha=.28)
        at.set_xlim(-4, 4); at.set_ylim(0, 1); at.set_xlabel("x")
        (at.set_ylabel("t") if c == 0 else at.tick_params(labelleft=False))
    fig.legend(handles=[Line2D([], [], color=BLUE, lw=7, alpha=.55),
                        Line2D([], [], color=GREEN, lw=7, alpha=.55),
                        Line2D([], [], color=ORANGE, lw=7, alpha=.65),
                        Line2D([], [], color=INK, lw=1.8)],
               labels=[r"Base Measure $p_1$", r"Tilting Measure $\nu$", r"Generated $\mu$",
                       r"True $\mu^*$"],
               loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(.52, .015))
    fig.suptitle(
        r"Figure 2 reproduced: $\lambda^*=10$, $\sigma_t=1$, $K_{FP}=3$ undamped, trained flow "
        r"model, mixture component sd " + f"{sd:g}." + "\n"
        r"Gradient steering is eq (11): $\varepsilon_t\nabla_x\Psi_t$ at the same $\lambda$, "
        r"with no (12b) reweighting.  $W_1$ Monte-Carlo floor "
        + f"{f2['w1_floor']:.3f}.", fontsize=8.5, color=INK2)
    fig.savefig("fig2_repro.png", dpi=170, facecolor=SURF)
    print("wrote fig2_repro.png", flush=True)

    # ---- Figure 3: objective gap vs steering strength, one panel per noise schedule
    scheds = [("const", r"$\sigma_t = 1$"), ("sqrt1mt", r"$\sigma_t = \sqrt{1-t}$"),
              ("memoryless", r"$\sigma_t = \sqrt{2(1-t)/t}$")]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9), constrained_layout=True, facecolor=SURF)
    for ax, (sch, lab) in zip(axes, scheds):
        d = json.load(open(out_path("fig3", f"_{sch}")))
        lams = np.array(d["lams"])
        ax.axvline(d["lam_star"], color=INK, lw=1.1, ls=(0, (3, 2)))
        for key, col, mk, ls, nm in F3:
            if f"J_{key}" not in d:
                continue
            raw = np.array(d[f"J_{key}"]) - d["J_star"]
            g = np.maximum(raw, 1e-4)
            ax.plot(lams, g, color=col, lw=1.6, ls=ls, marker=mk, ms=4, label=nm)
            neg = raw < 1e-4
            if neg.any():
                ax.plot(lams[neg], g[neg], ls="none", marker=mk, ms=6, mfc="none", mec=col, mew=1.2)
        ax.set_yscale("log"); ax.set_title(lab, color=INK, fontsize=10)
        ax.set_xlabel(r"Steering strength $\lambda$")
        if ax is axes[0]:
            ax.set_ylabel(r"$J(\mu)-J(\mu^*)$")
    h, l = axes[0].get_legend_handles_labels()
    h += [Line2D([], [], color=INK, lw=1.1, ls=(0, (3, 2)))]; l += [r"True $\lambda^*$"]
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(.5, -.28))
    fig.suptitle(
        r"Figure 3 reproduced. $\lambda^*=10$, reward bandwidth $\sigma=5$ ($h=5\sqrt{2}$), "
        r"mixture component sd " + f"{sd:g}" + r", trained flow model, $N=" + f"{f2['N']}" + r"$."
        "\n"
        r"Gradient steering is eq (11): the SAME steering drift as the mean-field runs at every "
        r"$\lambda$, differing only by (12b) reweighting.  Brown is the same family at the "
        r"paper's display range $[0,4]$."
        "\n"
        r"Hollow markers sit below the KL-estimator floor.  $J(\mu^*)$ uses the same estimator "
        r"on $N$ draws from a grid-solved $\mu^*$, so its plug-in bias cancels.",
        fontsize=8.5, color=INK2, y=1.30)
    fig.savefig("fig3_repro.png", dpi=170, facecolor=SURF, bbox_inches="tight")
    print("wrote fig3_repro.png", flush=True)

    # ---- Algorithm 2: does pooling the measure across batches help?
    if os.path.exists(out_path("pooling_mb")):
        d = json.load(open(out_path("pooling_mb")))
        fig, ax = plt.subplots(figsize=(5.4, 3.6), constrained_layout=True, facecolor=SURF)
        for (n, row), col, mk in zip(sorted(d["w1"].items(), key=lambda kv: int(kv[0])),
                                     (RAMP[0], RAMP[2]), ("o", "s")):
            Ms = sorted(int(k) for k in row)
            ax.plot(Ms, [row[str(M)] for M in Ms], color=col, marker=mk, lw=1.6,
                    label=f"$N={n}$ per batch")
        ax.axhline(d["w1_floor"], color=MUTED, lw=1.0, ls=(0, (3, 2)),
                   label=f"MC floor at $N=4000$ ({d['w1_floor']:.3f})")
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlabel("batches $M$"); ax.set_ylabel(r"$W_1$ to $\mu^*$")
        ax.set_title("Algorithm 2: pooling the measure across batches", color=INK, fontsize=10)
        ax.legend(fontsize=8)
        fig.savefig("figmb_repro.png", dpi=170, facecolor=SURF)
        print("wrote figmb_repro.png", flush=True)


def main(argv):
    if len(argv) < 2:
        raise SystemExit(__doc__)
    cfg = yaml.safe_load(open(argv[0]))
    # The toy stages require float64: the old scripts each set it at module level, and omitting it
    # ran the time grid, the prior and the noise in float32, which moved the Figure-2 positions by
    # 1.15e-5 against the stored reference. Set explicitly from config rather than inherited from
    # whatever imported torch first.
    torch.set_default_dtype(getattr(torch, cfg.get("torch_dtype", "float64")))
    stage, rest = argv[1], argv[2:]
    pos = [a for a in rest if not a.startswith("--set")]
    for a in rest:
        if a.startswith("--set"):
            k, v = a.split("=", 1)
            cfg[k.removeprefix("--set").lstrip(".")] = yaml.safe_load(v)
    if stage not in STAGES:
        raise SystemExit(f"stage {stage!r}; choose from {sorted(STAGES)}")
    STAGES[stage](cfg, *pos)


if __name__ == "__main__":
    main(sys.argv[1:])
