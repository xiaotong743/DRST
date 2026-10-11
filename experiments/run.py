"""Proteina driver: the Table-1 gamma sweep, designability and the pointwise/binning probes.

Driven by configs/proteina*.yaml. The 1-D reproduction is experiments/run_toy.py; steering
branches are steer_a.py and steer_f.py; shared scaffolding is proteina.py; algorithms src/steering/.
"""
import json
import os
import sys

import torch
import yaml

from proteina import (OUR_PROTOCOL, PAPER_MFS, PAPER_PROTOCOL, _pairwise_tm, _proteina,
                    foldseek_clusters, out_path)
from steering.rewards import Pointwise
from steering.steerer import Config, steer


# ---------------------------------------------------------------- Proteina stages

def proteina_pointwise(cfg):
    """Phase 2: pointwise steering on the frozen checkpoint, against an absolute reference."""
    from steering.models.proteina import radius_of_gyration, ca_ca_spacing
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
    from steering.models.proteina import radius_of_gyration, soft_bin_pairwise_distances
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


# ---------------------------------------------------------------- A/B temperature experiments

def gamma_diversity(cfg, length="all", inference_dir="proteina/inference"):
    """The paper's diversity metric for the gamma sweep, computed with the paper's own tool.

    Foldseek `easy-cluster` over the DESIGNABLE samples pooled across all five lengths, at
    TM > 0.5, with the cluster count divided by the designable count -- App. F.1 verbatim. The
    gamma=0.45 row is the calibration: the paper reports 0.63, i.e. 305 clusters over ~482
    designable.

    Pooling across lengths is the paper's choice and is safe with THEIR command but not with our
    TM-align one: `_pairwise_tm` normalises by the shorter chain and applies no coverage
    requirement, so a 50-mer scores TM 0.769 against a nested region of a 250-mer (measured).
    That is why this stage does not reuse it.
    """
    import glob
    import shutil

    dpath = out_path(cfg, "designability")
    if not os.path.exists(dpath):
        raise FileNotFoundError(f"{dpath} missing: run the `designability` stage first")
    des_all = json.load(open(dpath))
    work = os.path.abspath(os.path.join(cfg["out_dir"], "foldseek"))
    os.makedirs(work, exist_ok=True)
    out = {}
    if length == "all":
        print(f"{'gamma':>6} {'nDes':>5} {'clusters':>9} {'ours':>7} {'paper':>7} {'diff':>7}",
              flush=True)
    else:
        print(f"pooled length restricted to L={length}; the paper's diversity is pooled over "
              f"all of {PAPER_PROTOCOL['lengths']}, so it is NOT comparable here and is omitted",
              flush=True)
        print(f"{'gamma':>6} {'nTot':>5} {'nDes':>5} {'des%':>6} {'clusters':>9} {'div':>7}",
              flush=True)
    for gkey, rec in sorted(des_all.items(), key=lambda kv: float(kv[0])):
        gam = rec["gamma"]
        per = rec.get("per_struct") or {}
        if not per:
            print(f"gamma={gam}: no per-structure scRMSD; re-run `designability`", flush=True)
            continue
        want = f"gam{round(gam * 100):03d}"                        # 0.45 -> "gam045"
        run_dir = next((d for d in glob.glob(os.path.join(inference_dir, f"*{want}_des*"))
                        if os.path.isdir(d)), None)
        if run_dir is None:
            print(f"gamma={gam}: no {want}_des directory", flush=True)
            continue
        designable = sorted(n for n, v in per.items()
                            if v < PAPER_PROTOCOL["scrmsd_threshold"]
                            and (length == "all" or n.startswith(f"n_{length}_")))
        n_total = sum(1 for n in per
                      if length == "all" or n.startswith(f"n_{length}_"))
        pool = os.path.join(work, f"{want}_L{length}_pool")
        shutil.rmtree(pool, ignore_errors=True)
        os.makedirs(pool)
        for n in designable:                                      # pooled across ALL lengths
            src = os.path.join(run_dir, n, n + ".pdb")
            if os.path.exists(src):
                os.symlink(os.path.abspath(src), os.path.join(pool, n + ".pdb"))
        nc, nm = foldseek_clusters(pool, os.path.join(work, f"{want}_L{length}_tmp"))
        nd = len(designable)
        ours = nc / nd if nd else 0.0
        # PAPER_MFS's diversity is pooled over all FIVE lengths, so it is only a valid
        # comparison for the pooled computation. Printing a diff against it for a single length
        # would compare cluster counts drawn from different pools.
        p = PAPER_MFS.get(round(gam, 2)) if length == "all" else None
        pd = (p or {}).get("diversity")
        out[f"{gam:g}"] = dict(gamma=gam, length=length, n_total=n_total, n_designable=nd,
                               designable_pct=100.0 * nd / n_total if n_total else 0.0,
                               n_clustered=nm, n_clusters=int(nc),
                               clusters_per_designable=ours, paper_diversity=pd,
                               command="foldseek easy-cluster --alignment-type 1 --cov-mode 0 "
                                       "--min-seq-id 0 --tmscore-threshold 0.5")
        if length == "all":
            print(f"{gam:6.2f} {nd:5d} {nc:9d} {ours:7.3f} "
                  f"{(f'{pd:.2f}' if pd is not None else '-'):>7} "
                  f"{(f'{ours - pd:+.3f}' if pd is not None else '-'):>7}", flush=True)
        else:
            print(f"{gam:6.2f} {n_total:5d} {nd:5d} "
                  f"{100.0 * nd / n_total if n_total else 0:6.1f} {nc:9d} {ours:7.3f}",
                  flush=True)
    tag = "" if length == "all" else f"_L{length}"
    json.dump(out, open(out_path(cfg, f"gamma_diversity{tag}"), "w"), indent=1)

def surrogate_check(cfg, run="inference_gam045_des.yaml", length="100", limit="100",
                   inference_dir="proteina/inference"):
    """Does contact-feature similarity predict TM-score? The handoff's precondition for any
    diversity claim, and it needs no steering -- only a native pool at ONE fixed length.

    The reward is -E k(theta, zeta) on smooth contact features. If k is unrelated to structural
    similarity then driving it up changes the contact statistics and nothing a structural
    biologist would call diversity. Reports the rank correlation between k and TM, and how well
    a k-threshold separates the TM > 0.5 pairs the paper's cluster metric uses.
    """
    import glob
    import numpy as np
    import torch as _t
    from proteinfoundation.metrics.designability import load_pdb
    from steering.rewards import rbf_gram, SoftContactFeatures, calibrate_bandwidth

    prev = _t.get_default_dtype()
    _t.set_default_dtype(_t.float32)
    L, lim = int(length), int(limit)
    dirs = sorted(glob.glob(os.path.join(inference_dir, run, f"n_{L}_id_*")))[:lim]
    if not dirs:
        raise FileNotFoundError(f"no n_{L}_id_* under {os.path.join(inference_dir, run)}")
    ca = []
    for d in dirs:
        pdb = os.path.join(d, os.path.basename(d) + ".pdb")
        if os.path.exists(pdb):
            ca.append(_t.tensor(load_pdb(pdb).atom_positions)[:, 1, :])     # CA, Angstrom
    x_ang = _t.stack(ca)
    print(f"{x_ang.shape[0]} native structures at length {L} from {run}", flush=True)

    feats = SoftContactFeatures(_t.ones(L, dtype=_t.bool), min_sep=cfg["min_sep"],
                                cutoff_nm=cfg["cutoff_nm"], width_nm=cfg["width_nm"])
    x_nm = x_ang / 10.0
    h, info = calibrate_bandwidth(feats, x_nm)
    k = rbf_gram(feats(x_nm), feats(x_nm), h).double().numpy()
    tm = _pairwise_tm(x_ang.numpy())
    n = tm.shape[0]
    off = ~np.eye(n, dtype=bool)
    kv, tv = k[off], tm[off]

    rank = lambda v: np.argsort(np.argsort(v)).astype(float)
    spearman = float(np.corrcoef(rank(kv), rank(tv))[0, 1])
    pearson = float(np.corrcoef(kv, tv)[0, 1])
    same = tv > 0.5
    # AUC of k as a classifier of "same fold" (TM > 0.5), by the rank-sum identity
    auc = (float((rank(kv)[same].mean() - (same.sum() - 1) / 2.0) / (~same).sum())
           if same.any() and (~same).any() else float("nan"))
    out = dict(run=run, length=L, n=n, h=h, bandwidth_info=info,
               spearman_k_vs_tm=spearman, pearson_k_vs_tm=pearson,
               frac_pairs_same_fold=float(same.mean()), auc_k_predicts_same_fold=auc,
               mean_k_same_fold=float(kv[same].mean()) if same.any() else float("nan"),
               mean_k_diff_fold=float(kv[~same].mean()) if (~same).any() else float("nan"),
               mean_tm=float(tv.mean()))
    print(f"h = {h:.6g}; {n * (n - 1)} ordered pairs, {100 * out['frac_pairs_same_fold']:.1f}% "
          f"with TM > 0.5", flush=True)
    print(f"  Spearman(k, TM) = {spearman:+.3f}   Pearson = {pearson:+.3f}", flush=True)
    print(f"  mean k: same-fold {out['mean_k_same_fold']:.4f} vs different-fold "
          f"{out['mean_k_diff_fold']:.4f}", flush=True)
    print(f"  AUC of k predicting TM > 0.5 = {auc:.3f}  (0.5 = no signal)", flush=True)
    _t.set_default_dtype(prev)
    json.dump(out, open(out_path(cfg, f"surrogate_check_L{L}"), "w"), indent=1)

def designability(cfg, inference_dir="proteina/inference"):
    """Recompute scRMSD per structure from the designability artifacts already on disk.

    The results CSV only records the LAST structure's 8 sequences -- `_res_scRMSD` there is the
    min over those 8, not a population statistic -- so the 500-sample number has to come from
    the per-structure directories. Uses Proteina's own `rmsd_metric` (CA only, Kabsch-aligned)
    and `load_pdb`, so the definition is theirs and not a re-derivation: scRMSD is the MIN over
    the 8 ProteinMPNN/ESMFold sequences, and designable means scRMSD < 2 A.
    """
    import ast
    import csv
    import glob
    import torch as _t
    from proteinfoundation.metrics.designability import load_pdb, rmsd_metric

    # Proteina's Kabsch helper builds intermediates at the DEFAULT dtype while `torch.Tensor`
    # is always float32, so float64 here raises "expected scalar type Float but found Double".
    prev_dtype = _t.get_default_dtype()
    _t.set_default_dtype(_t.float32)
    out = {}
    for d in sorted(glob.glob(os.path.join(inference_dir, "*gam*_des*"))):
        if not os.path.isdir(d):
            continue
        # gamma comes from the run's own CSV, which records the value actually sampled with;
        # the directory name is only a fallback ("gam055" -> 0.55).
        tag = "gam" + d.split("gam")[1][:3]
        found = glob.glob(os.path.join(inference_dir, f"results_*{tag}_des*.csv"))
        gam = (float(list(csv.DictReader(open(found[0])))[-1]["sampling_caflow_sc_scale_noise"])
               if found else int(tag[3:]) / 100.0)
        row = list(csv.DictReader(open(found[0])))[-1] if found else {}
        # Per LENGTH, because the jobs emit structures in length order and designability falls
        # steeply with length: averaging over whatever has finished is biased UPWARD, since the
        # long hard cases come last. A partial gamma is reported as partial, never as a number.
        # scRMSD is a MIN over sequences, so a structure with fewer than the configured number
        # of ESMFold predictions gets a min over a partial set -- biased upward (less chance of
        # a good one) and silently counted as a finished structure.
        want_seqs = int(row.get("designability_seqs_per_struct") or 8)
        by_len, partial_seqs, per_struct = {}, 0, {}
        for sd in sorted(glob.glob(os.path.join(d, "n_*_id_*"))):
            gen = os.path.join(sd, os.path.basename(sd) + ".pdb")
            esms = sorted(glob.glob(os.path.join(sd, "esm_*.pdb_esm")))
            if not os.path.exists(gen) or len(esms) < want_seqs:
                partial_seqs += 1 if esms else 0
                continue
            g = _t.Tensor(load_pdb(gen).atom_positions)
            r = [rmsd_metric(g, _t.Tensor(load_pdb(e).atom_positions)) for e in esms]
            by_len.setdefault(int(os.path.basename(sd).split("_")[1]), []).append(min(r))
            per_struct[os.path.basename(sd)] = min(r)
        vals = [x for v in by_len.values() for x in v]
        if not vals:
            continue
        quota = int(row.get("nsamples_per_len") or 0)
        want = ast.literal_eval(row["nres_lens"]) if row.get("nres_lens", "None") != "None" \
            else sorted(by_len)
        missing = {L: quota - len(by_len.get(L, [])) for L in want
                   if len(by_len.get(L, [])) < quota}
        v = sorted(vals)
        rec = dict(gamma=gam, n=len(v), complete=(not missing and partial_seqs == 0),
                   missing=missing, seqs_per_struct=want_seqs,
                   structures_with_too_few_seqs=partial_seqs,
                   designable_pct=100.0 * sum(x < 2.0 for x in v) / len(v),
                   scrmsd_median=v[len(v) // 2], scrmsd_mean=sum(v) / len(v),
                   per_struct=per_struct,
                   per_len={str(L): dict(n=len(w),
                                         designable_pct=100.0 * sum(x < 2.0 for x in w) / len(w),
                                         scrmsd_median=sorted(w)[len(w) // 2])
                            for L, w in sorted(by_len.items())})
        out[f"{gam:g}"] = rec
        flag = ("" if rec["complete"] else
                f"  PARTIAL, missing {missing}" + (f", {partial_seqs} structures with <"
                                                   f"{want_seqs} seqs" if partial_seqs else ""))
        print(f"gamma={gam:.2f}: {len(v):4d} structures, designable "
              f"{rec['designable_pct']:5.1f}%, median scRMSD {v[len(v) // 2]:.3f} A{flag}",
              flush=True)
        print("    by length: " + "  ".join(
            f"{L}:{p['n']}/{quota or p['n']}={p['designable_pct']:.0f}%"
            for L, p in rec["per_len"].items()), flush=True)
    _t.set_default_dtype(prev_dtype)
    json.dump(out, open(out_path(cfg, "designability"), "w"), indent=1)

def gamma_table(cfg, inference_dir="proteina/inference"):
    """Proteina's Table-1 M_FS row across the sc_scale_noise sweep, read back from the
    inference CSVs. `fJSD` is the MEAN over the C/A/T levels and `fS` is reported per level;
    that mapping is pinned by the gamma=0.45 reproduction, not guessed from the column names.
    """
    import csv, glob
    paper = PAPER_MFS
    rows = []
    for f in sorted(glob.glob(os.path.join(inference_dir, "results_*gam*_fid*.csv"))):
        r = list(csv.DictReader(open(f)))[-1]
        mean = lambda pre: sum(float(r[f"_res_{pre}fJSD_{k}"]) for k in "CAT") / 3.0
        rows.append(dict(gamma=float(r["sampling_caflow_sc_scale_noise"]),
                         pdb_fid=float(r["_res_PDB_FID"]), afdb_fid=float(r["_res_AFDB_FID"]),
                         pdb_fjsd=mean("PDB_"), afdb_fjsd=mean("AFDB_"),
                         fs=[float(r[f"_res_fS_{k}"]) for k in "CAT"],
                         nsamples=int(r["nsamples_per_len"]) * 40))
    # from the `designability` stage: the CSVs hold only the last structure's 8 sequences
    dpath = out_path(cfg, "designability")
    des = {}
    if os.path.exists(dpath):
        des = {float(k): (v["designable_pct"], v.get("complete", False))
               for k, v in json.load(open(dpath)).items()}
    # Every row is printed as ours / paper's AT THE SAME GAMMA, across all nine metrics the
    # paper reports, so a missing comparison is visible rather than merely absent. A metric is
    # only marked comparable when OUR protocol matches theirs as well:
    #   designability, diversity -> 5 lengths x 100 = 500, same as the paper           VALID
    #   FPSD / fS / fJSD         -> the paper uses 5000; this sweep uses 1000          NOT VALID
    # A Frechet distance is biased upward at small N (measured: the same gamma=0.45 samples give
    # 396.8 averaged over 1250-sample shards against 387.0 pooled over 5000), so the reduced-N
    # FID columns are an internal gamma-gradient, not a comparison. Where a 5000-sample pool
    # exists it is loaded from gamma_fid5000.json and used for the comparison instead.
    div = {}
    dpath2 = out_path(cfg, "gamma_diversity")
    if os.path.exists(dpath2):
        div = {float(k): v for k, v in json.load(open(dpath2)).items()}
    pool5k = {}
    ppath = out_path(cfg, "gamma_fid5000")
    if os.path.exists(ppath):
        pool5k = {float(k): v for k, v in json.load(open(ppath)).items()}
    VALID = {"des%", "div"}                       # protocol-matched without a 5000-sample pool
    hdr = ("gamma", "des%", "PDBfpsd", "AFDBfpsd", "fS_C", "fS_A", "fS_T", "pJSD", "aJSD", "div")
    print("  ours / paper at the same gamma ('-' = paper does not report this gamma)\n")
    print("".join(f"{h:>18}" if h != "gamma" else f"{h:>6}" for h in hdr))
    for r in rows:
        d = des.get(r["gamma"])
        p = paper.get(r["gamma"])
        dv = div.get(r["gamma"])
        ours = [None if not d else d[0], r["pdb_fid"], r["afdb_fid"], r["fs"][0], r["fs"][1],
                r["fs"][2], r["pdb_fjsd"], r["afdb_fjsd"],
                (dv or {}).get("clusters_per_designable")]
        theirs = ([p["des"], p["pdb_fpsd"], p["afdb_fpsd"], p["fs"][0], p["fs"][1], p["fs"][2],
                   p["pdb_fjsd"], p["afdb_fjsd"], p["diversity"]] if p else [None] * 9)
        # prefer a 5000-sample pool for the FID-family metrics when one exists
        pk = pool5k.get(round(r["gamma"], 2))
        if pk:
            for j, key in enumerate(("pdb_fpsd", "afdb_fpsd", "fs_c", "fs_a", "fs_t",
                                     "pdb_fjsd", "afdb_fjsd"), start=1):
                if pk.get(key) is not None:
                    ours[j] = pk[key]
        cells = []
        for name, o, t in zip(hdr[1:], ours, theirs):
            os_ = "-" if o is None else f"{o:.2f}"
            ts = "-" if t is None else f"{t:.2f}"
            # "~" marks a value whose protocol does not match the paper's, so the pair is not a
            # comparison even though both numbers are present
            comparable = name in VALID or (pk is not None and name != "div")
            mark = "" if (comparable or t is None) else "~"
            cells.append(f"{os_ + ' / ' + ts + mark:>18}")
        star = "" if (d and d[1]) else "*"
        print(f"{r['gamma']:6.2f}" + "".join(cells) + star)
        r["fid_samples"] = (pk or {}).get("n_samples", OUR_PROTOCOL["fid_samples"])
        r["designability"] = d[0] if d else None
        r["designability_complete"] = bool(d[1]) if d else None
        r["diversity"] = (dv or {}).get("clusters_per_designable")
        r["paper"] = p
    print(f"\n  ~ = our protocol does not match the paper's for that metric, so the pair is not a "
          f"comparison.\n      designability and diversity use the paper's 500-sample, 5-length "
          f"protocol exactly.\n      FPSD/fS/fJSD need {PAPER_PROTOCOL['fid_samples']} samples; "
          f"this sweep has {OUR_PROTOCOL['fid_samples']}"
          + (f", with 5000-sample pools at {sorted(pool5k)}" if pool5k else "") + ".")
    if any(d and not d[1] for d in (des.get(r["gamma"]) for r in rows)):
        print("\n* designability from an INCOMPLETE run: structures are emitted in length order "
              "and designability falls with length, so a partial subset is biased upward.")
    if rows:
        print(f"\nFID path: {rows[0]['nsamples']} samples per gamma (paper 5000), so absolute "
              f"FPSD sits above Table 1 while the gamma gradient at fixed N is preserved.")
    json.dump(rows, open(out_path(cfg, "gamma_table"), "w"), indent=1)

import steer_a as _a                                                          # noqa: E402
from proteina import calibrate, steered_eval                                    # noqa: E402
from proteina import mem_probe                                                  # noqa: E402

STAGES = {"proteina_pointwise": proteina_pointwise, "binning_gradient": binning_gradient,
          "designability": designability, "gamma_diversity": gamma_diversity,
          "gamma_table": gamma_table, "surrogate_check": surrogate_check,
          "mem_probe": lambda cfg, *s: mem_probe(cfg, *s, mk_cfg=_a.mk_cfg,
                                                 steerer_cls=_a.SteererA),
          "ab_calibrate": lambda cfg: calibrate(cfg, _a.mk_cfg, _a.SteererA),
          "steered_eval": steered_eval,
          "ab_run": lambda cfg, *w: _a.run(cfg, *w),
          "steered_pool_diversity": _a.steered_pool_diversity}

def main(argv):
    if len(argv) < 2:
        raise SystemExit(__doc__)
    cfg = yaml.safe_load(open(argv[0]))
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
