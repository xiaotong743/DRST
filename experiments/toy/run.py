"""
Meanfiled the 1-D toy
"""
import json
import os
import sys
from dataclasses import dataclass

import torch
import yaml

from steering.smc import normalise
from steering.rewards import MMD
from steering.steerer import Config, FKConfig, FKSteerer, Steerer, multibatch, steer

from .toy import (GaussianMixtureFlow, TrainedFlow, Visual, grid_mu_star, objective_paper,
                  sample_grid, w1_to_grid)


@dataclass
class VisualConfig(Config):
    record_traj: int = 0        # watched initial particles; fig2 only


@dataclass
class VisualFKConfig(FKConfig):
    record_traj: int = 0


class VisualSteerer(Visual, Steerer):
    """Steerer that also returns the genealogy fig2 draws."""


class VisualFKSteerer(Visual, FKSteerer):
    """FKSteerer that also returns the genealogy fig2 draws."""


def out_path(cfg, part, tag=""):
    """`sd` is the mixture component width, so it belongs to the toy's filenames and nothing
    else: no Proteina config defines it."""
    os.makedirs(cfg["out_dir"], exist_ok=True)
    sd = f"_sd{cfg['sd']:g}" if "sd" in cfg else ""
    return os.path.join(cfg["out_dir"], f"{part}{sd}{tag}.json")


def _gen(seed):
    """Generator on the default device; `torch.Generator()` is always CPU."""
    return torch.Generator(device=torch.get_default_device()).manual_seed(seed)


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
    """nu: the tilting measure. Same means and reversed weights unless `nu_means` moves it."""
    tgt = GaussianMixtureFlow(means=tuple(cfg.get("nu_means") or cfg["means"]),
                              stds=(cfg["sd"], cfg["sd"]),
                              weights=tuple(cfg["nu_weights"]))
    return tgt.sample_p1(cfg["nu_samples"],
                         _gen(cfg["nu_seed"])).reshape(-1, 1)

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


def _fk_series(cfg):
    """Gradient-free FK potential steering on the SAME reward. Deliberately NOT in `_series`:
    verify.py re-runs the full sweep and compares key-for-key against the snapshot."""
    return {"fk": lambda x: dict(lam=x, potential_type="diff")}


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
    ref, g = base_model(cfg), _gen(1)
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
    nu = GaussianMixtureFlow(means=tuple(cfg.get("nu_means") or cfg["means"]),
                             stds=(cfg["sd"], cfg["sd"]),
                             weights=tuple(cfg["nu_weights"]))
    out = {"grid": grid.tolist(), "mu_star": q.tolist(), "sd": cfg["sd"], "h": cfg["h"], "N": n,
           "fig_tag": cfg.get("fig_tag", ""), "nu_means": cfg.get("nu_means") or cfg["means"],
           "xlim": cfg.get("xlim", 4.0), "lam_star": lam_star,
           "resample_every": cfg["resample_fig2"], "record_traj": cfg["record_traj"],
           "resample_start": cfg.get("resample_start", 0),
           "resample_end": cfg.get("resample_end"),
           "nu_weights": cfg["nu_weights"], "p1_weights": cfg["p1_weights"],
           "means": cfg["means"],
           "p1": torch.softmax(base_model(cfg).log_p1(grid), 0).tolist(),
           "nu": torch.softmax(nu.log_p1(grid), 0).tolist()}
    out["w1_floor"] = sum(
        w1_to_grid(sample_grid(grid, q, n, _gen(s)), unif, grid, q)
        for s in range(8)) / 8.0
    runs = {"base": dict(lam=0.0, resample_every=0),
            # **[paper]** C.1: "3 fixed-point iterations for the visualisation"
            "meanfield": dict(lam=lam_star, k_fp=3, damp=1.0,
                              resample_every=cfg["resample_fig2"]),
            "gradient": dict(lam=lam_star, reweight=False, resample_every=0),
            # same lam and the same resampling stride as mean-field, so the panels differ only
            # in how the log-weights are made: eq (12b)'s rate against a potential of the VALUE.
            # reward_min_value stays None, so `max` seeds at -inf: the paper's Max has no floor.
            **{f"fk_{k}": dict(lam=lam_star, potential_type=k,
                               resample_every=cfg["resample_fig2"])
               for k in ("add", "diff", "max")}}
    for k, kw in runs.items():
        base = dict(nsteps=cfg["steps"], seed=cfg["fig2_seed"],
                    record_traj=cfg["record_traj"],
                    resample_start=cfg.get("resample_start", 0),
                    resample_end=cfg.get("resample_end"),
                    **kw)
        o = (VisualFKSteerer(m, pot, VisualFKConfig(**base), n).run() if "potential_type" in kw
             else VisualSteerer(m, pot, VisualConfig(**base), n).run())
        w = o["w"] if kw.get("reweight", True) else unif
        out[f"x_{k}"], out[f"w_{k}"] = o["x"].reshape(-1).tolist(), w.reshape(-1).tolist()
        out[f"traj_{k}"], out["ts"] = o["traj"].tolist(), o["ts"].tolist()
        out[f"jump_{k}"] = [[j, h.tolist()] for j, h in o["jumps"]]
        out[f"ghost_{k}"] = [[j, h.tolist()] for j, h in o["ghosts"]]
        # what resampling did to the WHOLE population: the drawn 120 of 4000 are too sparse to
        # show coalescence even when the population is heavily duplicated.
        out[f"res_{k}"] = [[d["step"], d["ess"], d["n_resampled"]]
                           for d in o["diag"] if d["n_resampled"]]
        out[f"w1_{k}"] = w1_to_grid(o["x"], w, grid, q)
        print(f"  {k:10s} W1 to mu* = {out[f'w1_{k}']:.4f}", flush=True)
    ft = cfg.get("fig_tag", "")
    json.dump(out, open(out_path(cfg, "fig2", ft), "w"))
    print(f"wrote {out_path(cfg, 'fig2', ft)}  (W1 floor {out['w1_floor']:.4f})",
          flush=True)

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
        objective_paper(sample_grid(grid, q, n, _gen(s)), unif,
                        base_model(cfg), y, lam_star, cfg["h"], cfg["kl_bw"])
        for s in range(cfg["jstar_seeds"])) / cfg["jstar_seeds"]
    print(f"[{sched}] J(mu*) = {out['J_star']:.5f}", flush=True)
    series = {**_series(cfg), **_fk_series(cfg)}
    for name in ([only] if only else _series(cfg)):
        spec = series[name]
        J, W1 = [], []
        for lam in cfg["lams"]:
            js, ws = [], []
            for sd in cfg["seeds"]:
                kw = dict(nsteps=cfg["steps"], seed=sd,
                          resample_every=(0 if name.startswith("gradient")
                                          else cfg["resample_fig3"]),
                          **spec(lam))
                dev = torch.get_default_device()
                o = (FKSteerer(m, pot, FKConfig(**kw), n, device=dev).run()
                     if "potential_type" in kw
                     else steer(m, pot, Config(**kw), n, device=dev))
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
        w1_to_grid(sample_grid(grid, q, cfg["n"], _gen(s)), unif,
                   grid, q) for s in range(6)) / 6.0
    print(f"\nW1 Monte-Carlo floor at N={cfg['n']}: {out['floor']:.4f}", flush=True)
    json.dump(out, open(out_path(cfg, "pooling"), "w"), indent=1)

def plot(cfg):
    from .toy import figures
    figures(lambda part, tag="": out_path(cfg, part, tag), cfg["sd"],
            cfg.get("fig_tag", ""))

STAGES = {"train": train, "fig2": fig2, "fig3": fig3, "pooling": pooling, "plot": plot}


def main(argv):
    if len(argv) < 2:
        raise SystemExit(__doc__)
    cfg = yaml.safe_load(open(argv[0]))
    # float64: float32 moved the Figure-2 positions by 1.15e-5 against the stored reference.
    torch.set_default_dtype(getattr(torch, cfg.get("torch_dtype", "float64")))
    torch.set_default_device(cfg.get("device", "cpu"))   # cpu keeps every result bitwise
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
