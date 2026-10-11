"""Shared helpers for the Proteina-side experiments.

Experiment drivers live in experiments/: steer_a.py (reward steering) and run.py (the 1-D
reproduction and the gamma-sweep reproduction). Algorithms live in src/steering/. Nothing here
belongs in src: it is all experiment scaffolding.

A branch supplies two things to everything here: `mk_cfg(mdl, **overrides)`, its own Config
factory, and its `Steerer` subclass. This module never names ConfigA or SteererA, because the
temperature lives on the branch's config and not on the published one.
"""
import json
import os

import torch
import yaml

from steering.rewards import Pointwise


PAPER_MFS = {
    0.35: dict(des=98.2, pdb_fpsd=411.2, afdb_fpsd=392.1, fs=(1.93, 5.16, 16.79),
               pdb_fjsd=1.96, afdb_fjsd=1.53, diversity=0.49),
    0.45: dict(des=96.4, pdb_fpsd=388.0, afdb_fpsd=368.2, fs=(2.06, 5.32, 19.05),
               pdb_fjsd=1.65, afdb_fjsd=1.23, diversity=0.63),
    0.50: dict(des=91.4, pdb_fpsd=380.1, afdb_fpsd=359.8, fs=(2.10, 5.18, 19.07),
               pdb_fjsd=1.55, afdb_fjsd=1.13, diversity=0.71),
}
# Table 1 uses TWO generation protocols, so a comparison is only as good as the one it matches:
# designability/diversity/novelty/SS from 500 samples (App. I p47), FPSD/fS/fJSD from a separate
# unfiltered 5,000 (App. G.2 p31).

PAPER_PROTOCOL = dict(
    # designability / diversity / novelty / secondary structure
    lengths=(50, 100, 150, 200, 250), per_length=100, n_samples=500,
    tm_threshold=0.5, seqs_per_struct=8, scrmsd_threshold=2.0,
    # FPSD / fS / fJSD -- a DIFFERENT sample set, 5x larger, and unfiltered
    fid_lengths=tuple(range(60, 256, 5)), fid_per_length=125, fid_samples=5000,
    fid_designability_filter=False,
    # reference sets for FPSD/fJSD
    pdb_reference_n=15357, afdb_reference="D_FS Foldseek clusters",
)
# Ours: designability/diversity at 5 x 100 = 500, exactly their protocol. FPSD/fS/fJSD on their
# grid but 25 per length = 1000, ONE FIFTH their samples -- Frechet distances are biased upward
# at smaller N, the likely source of our +3.5..+5.8 FPSD offset. Not agreement without this.

OUR_PROTOCOL = dict(per_length=100, n_samples=500, fid_per_length=25, fid_samples=1000)
# A single-length pool is NOT comparable to the paper's diversity: their clusters come from a
# 5-length pool, and Foldseek's coverage requirement means different lengths rarely cluster
# together, so the pooled count is structurally larger.

def out_path(cfg, part, tag=""):
    os.makedirs(cfg["out_dir"], exist_ok=True)
    return os.path.join(cfg["out_dir"], f"{part}{tag}.json")


# ---------------------------------------------------------------- toy setup

def _proteina(cfg):
    from steering.models.proteina import ProteinaModel
    return ProteinaModel(ckpt=cfg["ckpt"], data_path=cfg["data_path"], n_res=cfg["n_res"],
                         n_particles=cfg["n"], gt_mode=cfg["gt_mode"], gt_p=cfg["gt_p"],
                         gt_clamp_val=cfg["gt_clamp_val"], nsteps=cfg["steps"])

ZERO_POT = Pointwise(lambda th: th.reshape(th.shape[0], -1).sum(1) * 0.0)

def _binning(cfg):
    """Soft binning is installed for the WHOLE run, not just the gradient: inside the block the
    frozen model IS the soft-binned model, so the drift and the gradient are one function."""
    from contextlib import nullcontext
    from steering.models.proteina import soft_binning
    return soft_binning() if cfg["soft_binning"] else nullcontext()

def _ab_features(cfg):
    from steering.rewards import SoftContactFeatures
    return SoftContactFeatures(torch.ones(cfg["n_res"], dtype=torch.bool),
                               min_sep=cfg["min_sep"], cutoff_nm=cfg["cutoff_nm"],
                               width_nm=cfg["width_nm"])

def _native_pool(cfg, mdl, n_batches, mk_cfg, steerer_cls, seed0=1000):
    """Unsteered draws at the run's own temperature: the A/B comparison's reference sampler.

    Built from the cfg PASSED IN, which is how `calibrate` pins the pilot to h_calib_temperature.
    """
    mk, xs = mk_cfg(cfg), []
    for b in range(n_batches):
        torch.manual_seed(seed0 + b)
        o = steerer_cls(mdl, ZERO_POT, mk(mdl, lam=0.0, resample_every=0), cfg["n"]).run()
        xs.append(o["x"].detach())
        print(f"  pilot batch {b + 1}/{n_batches}", flush=True)
    return torch.cat(xs)


# ---------------------------------------------------------------- reward bandwidth

def bandwidth_path(cfg):
    """One bandwidth per (length, calibration temperature, feature geometry).

    The name carries every field that defines the reward, so two runs that disagree on any of
    them cannot read each other's calibration. width_nm matters most: **[measured]** the shipped
    0.05 nm sits near a hard contact indicator and its agreement with TM-score decays with length
    (Spearman 0.53 -> 0.25 from L=50 to L=250), while retuning per length (0.2/0.05/0.4/0.8/1.6)
    holds it FLAT at 0.53/0.57/0.48/0.53/0.52.
    """
    return out_path(cfg, f"ab_bandwidth_r{cfg['n_res']}"
                         f"_g{round(cfg['h_calib_temperature'] * 100):03d}"
                         f"_s{cfg['min_sep']}"
                         f"_c{round(cfg['cutoff_nm'] * 100):03d}"
                         f"_w{round(cfg['width_nm'] * 1000):04d}")


def calibrate(cfg, mk_cfg, steerer_cls):
    """Freeze the reward's bandwidth on a native pilot pool, before any steering.

    Calibrated at `h_calib_temperature`, NOT the run's own, so every gamma at a length shares one
    reward; a gamma-dependent bandwidth would be a gamma-dependent objective. Recomputing h from
    the current particles would make it measure-dependent and add a term to the reward's first
    variation that nothing here accounts for.
    """
    from steering.rewards import calibrate_bandwidth
    cfg = dict(cfg, temperature=cfg["h_calib_temperature"])
    path = bandwidth_path(cfg)
    if os.path.exists(path) and not cfg.get("force_calib"):
        # Concurrent gamma jobs at one length calibrate the same thing: first to finish wins.
        print(f"bandwidth already at {path}; reusing (--set.force_calib=true to redo)",
              flush=True)
        return
    mdl = _proteina(cfg)
    feats = _ab_features(cfg)
    print(f"pilot: {cfg['pilot_batches']} x {cfg['n']} at temperature "
          f"{cfg['h_calib_temperature']} (fixed, not the run's), "
          f"soft_binning={cfg['soft_binning']}", flush=True)
    with _binning(cfg):
        pool = _native_pool(cfg, mdl, cfg["pilot_batches"], mk_cfg, steerer_cls)
    h, info = calibrate_bandwidth(feats, pool)
    info.update(calib_temperature=cfg["h_calib_temperature"],
                soft_binning=cfg["soft_binning"], n_res=cfg["n_res"],
                min_sep=cfg["min_sep"], cutoff_nm=cfg["cutoff_nm"], width_nm=cfg["width_nm"])
    print(f"h = {h:.6g}  over {info['n_pilot']} structures, {info['n_features']} features; "
          f"d2 in [{info['d2_min']:.3g}, {info['d2_max']:.3g}], "
          f"zero-distance fraction {info['frac_zero']:.3g}", flush=True)
    tmp = f"{path}.{os.getpid()}.tmp"        # atomic: concurrent writers must not interleave
    json.dump(info, open(tmp, "w"), indent=1)
    os.replace(tmp, path)


def resolve_h(cfg):
    """The reward bandwidth, from the calibration cache unless `h` is set explicitly.

    Resolved here rather than by the caller, so the filename has one source of truth.
    """
    if cfg.get("h") is not None:
        return cfg["h"]
    path = bandwidth_path(cfg)
    if not os.path.exists(path):
        raise SystemExit(f"no bandwidth at {path}: run the `calibrate` stage first, or pass "
                         f"--set.h=<value>")
    info = json.load(open(path))
    print(f"bandwidth h={info['h']:.6g} from {path} (calibrated at temperature "
          f"{info['calib_temperature']}, {info['n_features']} features)", flush=True)
    return info["h"]

def run_variants(cfg, table, which, mk_cfg, steerer_cls, pot_fn=None):
    """Run the named variants of `table` at one temperature and save each particle cloud.

    `table` maps name -> a dict of config overrides for that variant, `mk_cfg(cfg)` is the
    branch's config factory and `steerer_cls` its Steerer subclass. `pot_fn(cfg)` builds the
    potential; the default is the smooth contact reward every branch-A run used.
    """
    import numpy as np
    from steering.rewards import ContactRepulsion
    from steering.models.proteina import radius_of_gyration, ca_ca_spacing
    mdl = _proteina(cfg)
    feats = _ab_features(cfg)
    pot = pot_fn(cfg) if pot_fn else ContactRepulsion(feats, cfg["h"])
    n, tau, eta = cfg["n"], cfg["temperature"], cfg["eta"]
    mask = torch.ones(n, cfg["n_res"], dtype=torch.bool, device=mdl.device)

    def report(o):
        x, w = o["x"].detach(), o["w"]
        d, unif = o["diag"], torch.full((n,), 1.0 / n, dtype=torch.float64, device=x.device)
        wm = lambda v: float((w.to(v.dtype) * v).sum())
        pair = torch.cdist(x.reshape(n, -1), x.reshape(n, -1))
        return dict(
            reward_weighted=float(pot.value(x, w)),
            reward_uniform=float(pot.value(x, unif)),
            # the V-statistic's diagonal is exactly sum w_i^2 = 1/ESS, so at low ESS the plain
            # form reports weight concentration rather than population diversity
            reward_weighted_offdiag=float(pot.value_offdiag(x, w)),
            reward_uniform_offdiag=float(pot.value_offdiag(x, unif)),
            weight_sq_sum=float((w / w.sum()).pow(2).sum()),
            ess=float(d[-1]["ess"]) / n,
            picard_resid_max=max(r["picard_resid"] for r in d),
            # scale-free: ~0 = the last iteration moved nothing; >=1 = k_fp stopped the loop
            picard_contract_max=max(r.get("picard_contract", 0.0) for r in d),
            picard_contract_median=float(np.median([r.get("picard_contract", 0.0)
                                                    for r in d])),
            picard_iters=max(r["picard_iters"] for r in d),
            steer_over_base=max((r["steer_norm"] / r["base_norm"]
                                 for r in d if r["base_norm"] > 0), default=0.0),
            logw_spread=float(o["a"].max() - o["a"].min()),
            rg=wm(radius_of_gyration(x, mask)), ca_ca=wm(ca_ca_spacing(x)),
            min_pair_dist=float(pair[~torch.eye(n, dtype=torch.bool, device=x.device)].min()))

    out = dict(n=n, n_res=cfg["n_res"], steps=cfg["steps"], temperature=tau, eta=eta,
               resample_every=cfg["resample_every"],
               h=cfg["h"], soft_binning=cfg["soft_binning"], k_fp=cfg["k_fp"],
               damp=cfg["damp"], endpoint_lag=cfg["endpoint_lag"],
               runs={},
               sampler={k: cfg[k] for k in ("gt_mode", "gt_p", "gt_clamp_val", "steps",
                                            "vf_tail_t")})
    print(f"N={n} n_res={cfg['n_res']} tau={tau} eta={eta} h={cfg['h']:.6g} "
          f"soft_binning={cfg['soft_binning']} grad_chunk={cfg.get('grad_chunk')} "
          f"resample_every={cfg['resample_every']}", flush=True)
    with _binning(cfg):
        print(f"{'variant':>8} {'R_w':>10} {'R_unif':>10} {'ESS/N':>7} {'|st|/|b|':>9} "
              f"{'logw':>8} {'Rg':>7} {'CA-CA':>7}", flush=True)
        mk = mk_cfg(cfg)
        for name in (which or tuple(table)):
            over = table[name]
            e = over.get("lam", 0.0)
            torch.manual_seed(cfg["seed"])
            o = steerer_cls(mdl, pot, mk(mdl, **over), n).run()
            r = report(o)
            r.update(eta=e)
            out["runs"][name] = r
            cloud = os.path.join(cfg["out_dir"],
                                 f"ab_{name}{cfg.get('tag', '')}_r{cfg['n_res']}.pt")
            re_ = over.get("resample_every", cfg["resample_every"])
            prov = dict(eta=e, temperature=tau,
                        resample_every=re_, soft_binning=cfg["soft_binning"],
                        steps=cfg["steps"], seed=cfg["seed"])
            if os.path.exists(cloud):
                old = {k: torch.load(cloud, map_location="cpu").get(k) for k in prov}
                if old != prov and any(v is not None for v in old.values()):
                    raise FileExistsError(
                        f"{cloud} holds a cloud from different settings and steered_eval reads "
                        f"it by name, so overwriting would silently mix provenance.\n  on disk: "
                        f"{old}\n  this run: {prov}\nPass --set.tag=_something to separate them.")
            torch.save(dict(x=o["x"].detach().cpu(), a=o["a"].cpu(), w=o["w"].cpu(),
                            eta=e, temperature=tau,
                            resample_every=re_, h=cfg["h"],
                            soft_binning=cfg["soft_binning"], seed=cfg["seed"],
                            potential_type=over.get("potential_type"),
                            steps=cfg["steps"], gt_mode=cfg["gt_mode"]),
                       cloud)
            print(f"{name:>8} {r['reward_weighted']:10.5f} {r['reward_uniform']:10.5f} "
                  f"{r['ess']:7.3f} {r['steer_over_base']:9.2e} "
                  f"{r['logw_spread']:8.2f} {r['rg']:7.4f} {r['ca_ca']:7.3f}", flush=True)
    json.dump(out, open(out_path(cfg, f"ab_run{cfg.get('tag', '')}_r{cfg['n_res']}"), "w"),
              indent=1)
    return out


def mem_probe(cfg, *spec, mk_cfg, steerer_cls):
    """Peak memory of REAL steered steps, swept over (length, N), with the actual reward.

    **[measured]** extrapolation fails: per-particle cost is not constant in N, and a proxy loss
    is not the reward -- scaling 1.14 GB/particle at L=128 by (100/128)^2 gave 0.70 against a
    true ~1.6, and a bare endpoint backward at N=4 gave 0.844 against the real job's 1.59. So
    this runs `steer` itself for two steps and reports the peak.

    Usage: mem_probe [L:N,N,... [L:N,... ]]   default 100:16,32,48 250:4,8,16
    """
    import torch as _t
    mk = mk_cfg(cfg)
    from steering.rewards import SoftContactFeatures, ContactRepulsion
    from steering.models.proteina import ProteinaModel

    plan = []
    for item in (spec or ("100:16,32,48", "250:4,8,16")):
        L, ns = item.split(":")
        plan.append((int(L), [int(x) for x in ns.split(",")]))
    out = {}
    print(f"{'L':>5} {'N':>4} {'chunk':>6} {'peak GB':>9} {'GB/part':>9} {'s/step':>8} "
          f"{'result':>8}", flush=True)
    for L, ns in plan:
        feats = SoftContactFeatures(_t.ones(L, dtype=_t.bool), min_sep=cfg["min_sep"],
                                    cutoff_nm=cfg["cutoff_nm"], width_nm=cfg["width_nm"])
        pot = ContactRepulsion(feats, cfg["h"] or 0.25)
        for n in ns:
            chunk = cfg.get("grad_chunk")
            key = f"{L}_{n}_{chunk}"
            mdl = None
            try:
                mdl = ProteinaModel(ckpt=cfg["ckpt"], data_path=cfg["data_path"], n_res=L,
                                    n_particles=n, gt_mode=cfg["gt_mode"], gt_p=cfg["gt_p"],
                                    gt_clamp_val=cfg["gt_clamp_val"], nsteps=cfg["steps"])
                _t.cuda.empty_cache(); _t.cuda.reset_peak_memory_stats()
                import time
                t0 = time.time()
                with _binning(cfg):
                    steerer_cls(mdl, pot, mk(mdl, ts=mdl.ts[:3], lam=cfg["eta"],
                                             resample_every=0, grad_chunk=chunk), n).run()
                _t.cuda.synchronize()
                dt = (time.time() - t0) / 2
                peak = _t.cuda.max_memory_allocated() / 2**30
                out[key] = dict(length=L, n=n, grad_chunk=chunk, peak_gb=peak,
                                per_particle_gb=(peak - 0.78) / n, sec_per_step=dt, ok=True)
                print(f"{L:5d} {n:4d} {str(chunk):>6} {peak:9.2f} "
                      f"{(peak - 0.78) / n:9.3f} {dt:8.2f} {'ok':>8}", flush=True)
            except _t.OutOfMemoryError:
                out[key] = dict(length=L, n=n, grad_chunk=chunk, ok=False)
                print(f"{L:5d} {n:4d} {str(chunk):>6} {'-':>9} {'-':>9} {'-':>8} {'OOM':>8}",
                      flush=True)
            finally:
                del mdl
                _t.cuda.empty_cache()
    json.dump(out, open(out_path(cfg, "mem_probe"), "w"), indent=1)


def timing_probe(cfg, *spec):
    """Seconds per step and peak memory against nograd_chunk, plus the unsteered baseline.

    The three no_grad endpoint passes were inheriting grad_chunk, which is sized for the
    BACKWARD's activations. **[measured]** L=250 N=100 at grad_chunk=4 peaks at 19.72 GB of 80
    and takes 44.46 s/step, splitting 3 of 4 forwards into 25 launches of 4 for nothing. Batch
    size can select a different cuBLAS kernel, so the result is verified here, not assumed.

    Usage: timing_probe [L [nsteps [chunks]]]   default 250 6 4,8,16,32,100
    """
    import time
    import torch as _t
    from steer_a import ConfigA, SteererA
    from steering.rewards import ContactRepulsion

    L = int(spec[0]) if spec else cfg["n_res"]
    nsteps = int(spec[1]) if len(spec) > 1 else 6
    chunks = [int(c) for c in spec[2].split(",")] if len(spec) > 2 else [4, 8, 16, 32, 100]
    cfg = dict(cfg, n_res=L)
    cfg["h"] = resolve_h(cfg)
    mdl = _proteina(cfg)
    pot = ContactRepulsion(_ab_features(cfg), cfg["h"])
    n, gc = cfg["n"], cfg.get("grad_chunk")
    ts = mdl.ts[:nsteps + 1]

    def run(lam, ngc, reweight=True, k_fp=None, lag=None):
        base = dict(ts=ts, lam=lam, temperature=cfg["temperature"],
                    k_fp=cfg["k_fp"] if k_fp is None else k_fp,
                    damp=cfg["damp"], endpoint_lag=lag or cfg["endpoint_lag"],
                    reweight=reweight, resample_every=0,
                    vf_tail_t=cfg["vf_tail_t"], grad_chunk=gc, nograd_chunk=ngc,
                    global_rng=True, seed=cfg["seed"])
        _t.cuda.empty_cache(); _t.cuda.reset_peak_memory_stats()
        _t.manual_seed(cfg["seed"])
        with _binning(cfg):
            _t.cuda.synchronize(); t0 = time.time()
            out = SteererA(mdl, pot, ConfigA(**base), n).run()
            _t.cuda.synchronize()
        return (time.time() - t0) / nsteps, _t.cuda.max_memory_allocated() / 2**30, out

    print(f"L={L} N={n} grad_chunk={gc} steps={nsteps} soft_binning={cfg['soft_binning']}",
          flush=True)
    print(f"{'config':>22}{'s/step':>9}{'peak GB':>9}{'vs base':>9}{'max|dx| vs gc':>15}",
          flush=True)
    out = {}
    s0, m0, _ = run(0.0, gc)                      # unsteered, the thing steering is measured against
    print(f"{'unsteered lam=0':>22}{s0:>9.2f}{m0:>9.2f}{1.0:>8.2f}x{'-':>15}", flush=True)
    out["unsteered"] = dict(sec_per_step=s0, peak_gb=m0)
    ref = None
    for c in chunks:
        try:
            sec, gb, o = run(cfg["eta"], c)
        except _t.OutOfMemoryError:
            print(f"{('nograd_chunk=' + str(c)):>22}{'OOM':>9}{'-':>9}{'-':>9}{'-':>15}",
                  flush=True)
            out[f"nograd_{c}"] = dict(ok=False)
            _t.cuda.empty_cache()
            continue
        if ref is None:
            ref = o["x"].detach().clone()
            dx = 0.0
        else:
            dx = float((o["x"].detach() - ref).abs().max())
        print(f"{('nograd_chunk=' + str(c)):>22}{sec:>9.2f}{gb:>9.2f}{sec/s0:>8.2f}x{dx:>15.3e}",
              flush=True)
        out[f"nograd_{c}"] = dict(sec_per_step=sec, peak_gb=gb, ratio_to_unsteered=sec/s0,
                                  max_dx_vs_first=dx, ok=True)
        del o
        _t.cuda.empty_cache()

    # Where the time goes. reweight=False keeps grad_x Psi and drops the ENTIRE weight path, so
    # that difference isolates it; varying k_fp then separates the Picard iterations from the two
    # forwards they sit on -- flat from 1 to 10 means the solve is free.
    print("", flush=True)
    print(f"{'decomposition':>24}{'s/step':>9}{'vs base':>9}{'contract':>10}"
          f"{'max|dx| vs kfp3':>17}{'max|da| vs kfp3':>17}", flush=True)
    variants = [("grad only (reweight=F)", dict(reweight=False)),
                ("+weights k_fp=1", dict(k_fp=1)),
                ("+weights k_fp=3", dict(k_fp=3)),
                ("+weights k_fp=10", dict(k_fp=10)),
                ("+weights k_fp=30", dict(k_fp=30)),
                ("k_fp=3 lag=advance", dict(k_fp=3, lag="advance"))]
    res = {}
    for name, kw in variants:
        try:
            sec, gb, o = run(cfg["eta"], gc, **kw)
        except _t.OutOfMemoryError:
            print(f"{name:>24}{'OOM':>9}", flush=True)
            _t.cuda.empty_cache()
            continue
        res[name] = (sec, gb, o["x"].detach().clone(), o["a"].clone(),
                     max(r.get("picard_contract", 0.0) for r in o["diag"]))
        _t.cuda.empty_cache()
    # k_fp=3 is production. If raising it does not move x or a the fixed point was already
    # solved; if it does, k_fp was truncating and every result at that setting inherits the bias.
    ref = res.get("+weights k_fp=3")
    for name, (sec, gb, x, a, ctr) in res.items():
        dx = float((x - ref[2]).abs().max()) if ref is not None else float("nan")
        da = float((a - ref[3]).abs().max()) if ref is not None else float("nan")
        print(f"{name:>24}{sec:>9.2f}{sec/s0:>8.2f}x{ctr:>10.2e}{dx:>17.3e}{da:>17.3e}",
              flush=True)
        out[name] = dict(sec_per_step=sec, peak_gb=gb, ratio_to_unsteered=sec / s0,
                         picard_contract_max=ctr, max_dx_vs_kfp3=dx, max_da_vs_kfp3=da)

    # --- FK, on the same node and model, so the two are directly comparable. FK skips the
    # backward AND the look-ahead forwards, keeping only the Picard matvecs; its one real cost
    # is the external reward call, which fires on checkpoints. nsteps is short, so the stride is
    # set to 2 to make checkpoints fire at all -- the per-checkpoint cost is the difference from
    # the no-checkpoint run divided by their number.
    from steer_f import ConfigF, SteererF
    def runfk(kind, k_fp, every, p=None):
        base = dict(ts=ts, lam=cfg["eta"], temperature=cfg["temperature"], k_fp=k_fp,
                    damp=cfg["damp"], endpoint_lag=cfg["endpoint_lag"], potential_type=kind,
                    resample_every=every, vf_tail_t=cfg["vf_tail_t"], grad_chunk=gc,
                    nograd_chunk=gc, global_rng=True, seed=cfg["seed"])
        _t.cuda.empty_cache(); _t.cuda.reset_peak_memory_stats()
        _t.manual_seed(cfg["seed"])
        with _binning(cfg):
            _t.cuda.synchronize(); t0 = time.time()
            SteererF(mdl, p or pot, ConfigF(**base), n).run()
            _t.cuda.synchronize()
        return (time.time() - t0) / nsteps, _t.cuda.max_memory_allocated() / 2**30

    print("", flush=True)
    print(f"{'FK':>24}{'s/step':>9}{'vs base':>9}{'peak GB':>10}", flush=True)
    nck = max(nsteps // 2, 1)
    fkv = [("diff, no checkpoints", "diff", 3, 0, None),
           ("diff k_fp=1", "diff", 1, 2, None),
           ("diff k_fp=3", "diff", 3, 2, None),
           ("diff k_fp=10", "diff", 10, 2, None),
           ("diff k_fp=30", "diff", 30, 2, None),
           ("add k_fp=3", "add", 3, 2, None)]
    work = os.path.join(cfg["out_dir"], f"timing_fk_r{L}")
    os.makedirs(work, exist_ok=True)
    from steering.rewards import ClusterDiversity
    fkv.append(("diff + foldseek reward", "diff", 3, 2,
                ClusterDiversity(foldseek_labeller(work))))
    for name, kind, kfp, every, pp in fkv:
        try:
            sec, gb = runfk(kind, kfp, every, pp)
        except Exception as e:
            print(f"{name:>24}  failed: {str(e)[:50]}", flush=True); continue
        print(f"{name:>24}{sec:>9.2f}{sec/s0:>8.2f}x{gb:>10.2f}", flush=True)
        out[f"fk {name}"] = dict(sec_per_step=sec, peak_gb=gb, ratio_to_unsteered=sec / s0,
                                 checkpoints=0 if every == 0 else nck)
    json.dump(out, open(out_path(cfg, f"timing_probe_r{L}"), "w"), indent=1)


def _tm_labels(tm, thresh=0.5):
    """Single-linkage cluster labels of a pairwise TM-score matrix at `thresh`, by union-find."""
    n = tm.shape[0]
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if tm[i, j] > thresh:
                a, b = find(i), find(j)
                if a != b:
                    parent[a] = b
    return [find(i) for i in range(n)]


def _clusters_at(tm, thresh=0.5):
    return len(set(_tm_labels(tm, thresh)))

def _pairwise_tm(ca_ang):
    """[N, L, 3] CA coordinates in ANGSTROM -> [N, N] TM-score matrix.

    tmtools wraps the original TM-align C++, so these are real TM-scores, not a re-derivation.
    Backbones carry no sequence (Proteina writes every residue as ALA), so a constant sequence is
    passed and the alignment is purely structural.
    """
    import numpy as np
    from tmtools import tm_align
    n, L = ca_ang.shape[0], ca_ang.shape[1]
    seq = "A" * L
    tm = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            r = tm_align(np.ascontiguousarray(ca_ang[i]), np.ascontiguousarray(ca_ang[j]),
                         seq, seq)
            tm[i, j] = tm[j, i] = max(r.tm_norm_chain1, r.tm_norm_chain2)
    return tm

def steered_eval(cfg, *variants):
    """Designability and STRUCTURAL diversity of the saved steered clouds.

    The reward's own value (-w^T K w) is what the sampler optimises, so it cannot serve as
    evidence that steering improved diversity. This stage computes the paper's quantities
    instead: designability from ProteinMPNN -> ESMFold -> scRMSD (min over 8 sequences, < 2 A),
    and diversity as single-linkage TM-score clusters at TM > 0.5 divided by the designable
    count (paper Appendix F/F.1).

    The sampler returns a WEIGHTED cloud, so both views are reported: sum_i w_i 1[designable_i],
    the fraction of the TARGET measure that is designable, and an equal-weight set from residual
    resampling, what a downstream user holds. At low ESS these differ sharply, and that is the
    finding.
    """
    import sys
    import numpy as np
    import proteinfoundation
    import torch as _t
    from proteinfoundation.metrics.designability import scRMSD
    from proteinfoundation.utils.coors_utils import trans_nm_to_atom37
    from proteinfoundation.utils.ff_utils.pdb_utils import write_prot_to_pdb
    from steering.smc import ess, residual_resample

    # `run_proteinmpnn` shells out to the RELATIVE path ./ProteinMPNN/protein_mpnn_run.py and
    # sends stderr to /dev/null, so from any other cwd it fails silently and scRMSD then dies on
    # a missing .fa. Run from the Proteina repo root with absolute paths everywhere else, and
    # pin PYTHON_EXEC rather than trusting PATH.
    prot_root = os.path.dirname(os.path.dirname(os.path.abspath(proteinfoundation.__file__)))
    mpnn = os.path.join(prot_root, "ProteinMPNN", "protein_mpnn_run.py")
    if not os.path.exists(mpnn):
        raise FileNotFoundError(f"ProteinMPNN not found at {mpnn}")
    os.environ.setdefault("PYTHON_EXEC", sys.executable)

    prev_dtype = _t.get_default_dtype()
    _t.set_default_dtype(_t.float32)
    variants = variants or ("native", "A")
    tmp_root = os.path.abspath(os.path.join(cfg["out_dir"],
                                            f"steered_eval{cfg.get('tag', '')}"))
    os.makedirs(tmp_root, exist_ok=True)
    out = {}
    print(f"{'variant':>8} {'N':>4} {'ESS/N':>7} {'des% (w)':>9} {'des% (rs)':>10} "
          f"{'scRMSD med':>11} {'meanTM':>7} {'clust':>6} {'clust/des':>10}", flush=True)
    for name in variants:
        f = os.path.abspath(os.path.join(
            cfg["out_dir"], f"ab_{name}{cfg.get('tag', '')}_r{cfg['n_res']}.pt"))
        if not os.path.exists(f):
            print(f"  {name}: {f} missing, skipped", flush=True)
            continue
        d = _t.load(f, map_location="cpu")
        x, w = d["x"].float(), d["w"].double()
        n = x.shape[0]
        e = ess(d["a"]) / n
        # residual resampling gives the equal-weight set; a fixed seed keeps it reproducible
        idx = residual_resample(w / w.sum(), _t.Generator().manual_seed(0))

        vdir = os.path.join(tmp_root, name)
        os.makedirs(vdir, exist_ok=True)
        atom37 = trans_nm_to_atom37(x)                      # -> Angstrom, CA slot only
        scr = []
        cwd = os.getcwd()
        try:
            os.chdir(prot_root)
            for i in range(n):
                pdb = os.path.join(vdir, f"s{i}.pdb")
                write_prot_to_pdb(atom37[i].numpy(), pdb, overwrite=True, no_indexing=True)
                scr.append(scRMSD(pdb, tmp_path=os.path.join(vdir, f"tmp{i}"),
                                  num_seq_per_target=8))
                print(f"    {name} {i + 1}/{n}: scRMSD {scr[-1]:.3f}", flush=True)
        finally:
            os.chdir(cwd)
        scr = np.array(scr)
        des = scr < 2.0
        wn = (w / w.sum()).numpy()
        des_w = float((wn * des).sum()) * 100.0
        des_rs = float(des[idx.numpy()].mean()) * 100.0

        # Diversity over the DESIGNABLE members of the equal-weight set. Foldseek with the
        # paper's exact flags is the primary number, so it is the same metric as PAPER_MFS;
        # mean TM comes from TM-align and is reported alongside as a continuous summary.
        # NOTE the absolute value is still NOT comparable to the paper's: these pools are one
        # sequence length, theirs spans five, and cluster counts depend on both.
        keep = np.unique(idx.numpy()[des[idx.numpy()]])
        tm = _pairwise_tm((x[keep] * 10.0).numpy()) if len(keep) > 1 else np.eye(max(len(keep), 1))
        off = tm[~np.eye(tm.shape[0], dtype=bool)] if tm.shape[0] > 1 else np.array([1.0])
        nc_tm = _clusters_at(tm) if len(keep) > 1 else len(keep)
        fs_pool = os.path.join(vdir, "fs_pool")
        import shutil as _sh
        _sh.rmtree(fs_pool, ignore_errors=True)
        os.makedirs(fs_pool)
        for i in keep:
            _sh.copy(os.path.join(vdir, f"s{i}.pdb"), os.path.join(fs_pool, f"s{i}.pdb"))
        nc = (foldseek_clusters(fs_pool, os.path.join(vdir, "fs_tmp"))[0]
              if len(keep) > 1 else len(keep))
        out[name] = dict(n=n, ess=e, designable_weighted_pct=des_w, designable_resampled_pct=des_rs,
                         scrmsd_median=float(np.median(scr)), n_designable_unique=int(len(keep)),
                         scrmsd_per_struct={str(i): float(v) for i, v in enumerate(scr)},
                         mean_tm=float(off.mean()), max_tm=float(off.max()) if off.size else 1.0,
                         n_clusters=int(nc), n_clusters_tmalign=int(nc_tm),
                         clusters_per_designable=(nc / len(keep)) if len(keep) else 0.0,
                         eta=d.get("eta"), temperature=d.get("temperature"))
        r = out[name]
        print(f"{name:>8} {n:4d} {e:7.3f} {des_w:9.1f} {des_rs:10.1f} "
              f"{r['scrmsd_median']:11.3f} {r['mean_tm']:7.3f} {nc:6d} "
              f"{r['clusters_per_designable']:10.3f}", flush=True)
    _t.set_default_dtype(prev_dtype)
    # One sidecar per variant, then rebuild the aggregate from all of them. A plain dump of
    # `out` would erase every variant this process did not evaluate, so the 13 eval jobs of a
    # sweep -- which run concurrently and each hold one variant -- would clobber each other.
    import glob as _g
    path = out_path(cfg, f"steered_eval{cfg.get('tag', '')}_r{cfg['n_res']}")
    stem = path[:-5]
    for k, v in out.items():
        json.dump(v, open(f"{stem}__{k}.json", "w"), indent=1)
    merged = {os.path.basename(f)[:-5].split("__")[-1]: json.load(open(f))
              for f in sorted(_g.glob(f"{stem}__*.json"))}
    json.dump(merged, open(path, "w"), indent=1)

def tm_matrix():
    """theta [N,L,3] nm -> [N,N] TM-score matrix. Proteina Table 1's TM-Sc. is the MEAN of the
    off-diagonal entries, so this is the metric itself with no clustering step."""
    return lambda theta: _pairwise_tm(theta.detach().cpu().numpy() * 10.0)    # nm -> Angstrom


def tm_labeller(thresh=0.5):
    """theta [N,L,3] nm -> [N] single-linkage TM cluster labels, the rule `_clusters_at` counts.
    All particles share a length, so _pairwise_tm's shorter-chain caveat does not apply."""
    return lambda theta: _tm_labels(_pairwise_tm(theta.detach().cpu().numpy() * 10.0), thresh)


def foldseek_labeller(work_dir):
    """theta [N,L,3] nm -> [N] labels, by the paper's own easy-cluster command.

    foldseek_clusters is reused unchanged -- published numbers come from it -- and only its TSV
    is re-read here, for the representative column.
    """
    import shutil as _sh
    from proteinfoundation.utils.coors_utils import trans_nm_to_atom37
    from proteinfoundation.utils.ff_utils.pdb_utils import write_prot_to_pdb
    pool, tmp = os.path.join(work_dir, "pool"), os.path.join(work_dir, "tmp")

    def label(theta):
        _sh.rmtree(pool, ignore_errors=True)
        os.makedirs(pool)
        atom37 = trans_nm_to_atom37(theta.detach().cpu().float())
        for i in range(atom37.shape[0]):
            write_prot_to_pdb(atom37[i].numpy(), os.path.join(pool, f"s{i}.pdb"),
                              overwrite=True, no_indexing=True)
        foldseek_clusters(pool, tmp)
        rep = {}
        for line in open(os.path.join(tmp, "res_cluster.tsv")):
            a, b = line.split()[:2]
            rep[b] = a
        ids, out = {}, []
        for i in range(atom37.shape[0]):
            r = rep.get(f"s{i}", f"s{i}")          # a singleton may be absent from the tsv
            out.append(ids.setdefault(r, len(ids)))
        return out
    return label


FOLDSEEK = "/lustre/groups/ml01/workspace/xiaotong.fu/playground/proteina_data/tools/foldseek/bin/foldseek"

def foldseek_clusters(pdb_dir, tmp_dir):
    """Cluster a directory of PDBs with the paper's EXACT command (App. F.1, verbatim):

        foldseek easy-cluster <path_samples> <path_tmp>/res <path_tmp>
            --alignment-type 1 --cov-mode 0 --min-seq-id 0 --tmscore-threshold 0.5

    `<path_samples>` holds all DESIGNABLE samples, pooled across lengths. Use this for any
    comparison against published numbers: our `_clusters_at` is NOT the paper's metric, since it
    normalises by the shorter chain and imposes no coverage requirement, letting a short backbone
    match a nested region of a long one (**[measured]** TM 0.769 for a 50-mer against a 250-mer).
    """
    import shutil
    import subprocess
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)
    cmd = [FOLDSEEK, "easy-cluster", pdb_dir, os.path.join(tmp_dir, "res"), tmp_dir,
           "--alignment-type", "1", "--cov-mode", "0", "--min-seq-id", "0",
           "--tmscore-threshold", "0.5"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    tsv = os.path.join(tmp_dir, "res_cluster.tsv")
    if r.returncode != 0 or not os.path.exists(tsv):
        raise RuntimeError(f"foldseek failed (rc={r.returncode})\n{r.stdout[-2000:]}\n"
                           f"{r.stderr[-2000:]}")
    reps, members = set(), set()
    for line in open(tsv):
        a, b = line.split()[:2]
        reps.add(a)
        members.add(b)
    return len(reps), len(members)


# ---------------------------------------------------------------- driver CLI

def main(stages, argv):
    """The stage dispatcher every experiment driver shares.

    Usage: python experiments/<driver>.py <config.yaml> <stage> [args] [--set.key=value ...]
    """
    if len(argv) < 2:
        raise SystemExit(f"give a config and a stage; stages: {sorted(stages)}")
    cfg = yaml.safe_load(open(argv[0]))
    torch.set_default_dtype(getattr(torch, cfg.get("torch_dtype", "float32")))
    stage, rest = argv[1], argv[2:]
    pos = [a for a in rest if not a.startswith("--set")]
    for a in rest:
        if a.startswith("--set"):
            k, v = a.split("=", 1)
            cfg[k.removeprefix("--set").lstrip(".")] = yaml.safe_load(v)
    if stage not in stages:
        raise SystemExit(f"stage {stage!r}; choose from {sorted(stages)}")
    stages[stage](cfg, *pos)
