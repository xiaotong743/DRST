"""Reporting for the Proteina experiments: the stages that read results back and present them.

Split from common.py by what they touch, not by size. common.py is machinery a run DEPENDS on --
the sampler driver, the evaluation pipeline, the verified Table 1 reference. Nothing here can
change a number a run produces: these read finished outputs off disk and pool, compare or plot
them. That is the line to keep when adding to either file.
"""
import json
import os

from proteina import (PAPER_MFS, PAPER_PROTOCOL, _ab_features, foldseek_clusters,
                      out_path)

def steered_pool_diversity(cfg, variant="A", *tags):
    """Table-1-comparable designability and diversity for a steered run.

    The paper's protocol is 5 lengths x 100 = 500 samples, with diversity = foldseek clusters at
    TM > 0.5 over the DESIGNABLE samples POOLED across lengths, divided by the designable count
    (App. F.1 / I). A per-length number is not comparable to it, so this gathers the per-length
    `steered_eval` outputs for one gamma, pools them, and compares against PAPER_MFS at that
    gamma.

    Usage: steered_pool_diversity A _g045L50 _g045L100 ...
    Reads experiments/out/steered_eval<tag>_r<L>.json and the PDBs steered_eval wrote beside them,
    so it needs no GPU and no re-folding.
    """
    import glob
    import shutil

    if not tags:
        raise SystemExit("give the per-length tags, e.g. A _g045L50 _g045L100 ...")
    work = os.path.abspath(os.path.join(cfg["out_dir"], "steered_pool"))
    os.makedirs(work, exist_ok=True)
    pool = os.path.join(work, f"{variant}{tags[0]}_pooled")
    shutil.rmtree(pool, ignore_errors=True)
    os.makedirs(pool)

    n_tot, n_des, per_len, gammas = 0, 0, {}, set()
    for tag in tags:
        hits = glob.glob(os.path.join(cfg["out_dir"], f"steered_eval{tag}_r*.json"))
        if not hits:
            print(f"  {tag}: no steered_eval output, skipped", flush=True)
            continue
        rec = json.load(open(hits[0]))[variant]
        L = int(hits[0].rsplit("_r", 1)[1].split(".")[0])
        scr = rec.get("scrmsd_per_struct") or {}
        if not scr:
            raise SystemExit(f"{hits[0]} has no per-structure scRMSD; re-run steered_eval")
        gammas.add(rec.get("temperature"))
        src_dir = os.path.join(cfg["out_dir"], f"steered_eval{tag}", variant)
        des = [i for i, v in scr.items() if v < PAPER_PROTOCOL["scrmsd_threshold"]]
        for i in des:
            f = os.path.join(src_dir, f"s{i}.pdb")
            if os.path.exists(f):
                os.symlink(os.path.abspath(f), os.path.join(pool, f"L{L}_s{i}.pdb"))
        per_len[str(L)] = dict(n=len(scr), n_designable=len(des),
                               designable_pct=100.0 * len(des) / len(scr))
        n_tot += len(scr)
        n_des += len(des)
        print(f"  L={L}: {len(des)}/{len(scr)} designable", flush=True)

    if len(gammas) > 1:
        raise SystemExit(f"tags mix temperatures {gammas}; pool one gamma at a time")
    gam = gammas.pop() if gammas else None
    nc, _ = foldseek_clusters(pool, os.path.join(work, "tmp"))
    div = nc / n_des if n_des else 0.0
    p = PAPER_MFS.get(round(gam, 2)) if gam is not None else None
    out = dict(variant=variant, gamma=gam, tags=list(tags), n_total=n_tot,
               n_designable=n_des, designable_pct=100.0 * n_des / n_tot if n_tot else 0.0,
               n_clusters=int(nc), clusters_per_designable=div, per_len=per_len,
               paper=p, protocol_matched=(n_tot == PAPER_PROTOCOL["n_samples"]))
    print(f"\n{variant} at gamma={gam}: {n_des}/{n_tot} designable "
          f"({out['designable_pct']:.1f}%), {nc} clusters, diversity {div:.3f}", flush=True)
    if p:
        print(f"  paper at the same gamma (UNSTEERED): designability {p['des']:.1f}%, "
              f"diversity {p['diversity']:.2f}", flush=True)
    if n_tot != PAPER_PROTOCOL["n_samples"]:
        print(f"  NOTE {n_tot} samples, not the paper's {PAPER_PROTOCOL['n_samples']}: "
              f"cluster counts depend on pool size, so this is not yet comparable", flush=True)
    json.dump(out, open(out_path(cfg, f"steered_pool_{variant}_g{round((gam or 0) * 100):03d}"),
                        "w"), indent=1)


def pareto(cfg, *lengths):
    """Designability against diversity per length: steering's PAIRED effect on the front.

    The arrow is the result. Each arrow runs from the unsteered control to the steered run at
    the same gamma, and the two share a seed -- `global_rng=True` plus run_variants' per-variant
    manual_seed mean they consume the same prior draw and the same Brownian stream, so the pair
    differs only by the steering term. A down-left arrow is steering making the front worse.

    The control exists because this plot used to compare steered runs (soft-binned, through
    `steer`) against the gamma sweep from proteina/inference (hard-binned, the released
    full_simulation). Two different models, so nothing in it separated "steering moved the
    front" from "soft binning moved the front". The released sweep is still drawn, in muted
    ink, as the absolute reference it is -- not as the thing steering is measured against.

    Both axes are the paper's own metrics: scRMSD < 2 A, and foldseek clusters at TM > 0.5 over
    the designable set. Every panel is ONE sequence length, because cluster counts are not
    comparable across pool sizes.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Palette: the repo's house hues, checked against the five computable data-viz tests on the
    # #fcfcfb surface -- CVD dE 24.7 (target 8), normal-vision dE 33.6 (floor 15), contrast
    # 4.30:1 and 3.12:1 (min 3). Identity is carried by MARKER SHAPE as well as hue, so the
    # panels survive being printed in grey.
    SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983"
    BLUE, ORANGE = "#2a78d6", "#eb6834"
    plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": .8,
                         "figure.facecolor": SURF, "axes.facecolor": SURF,
                         "legend.frameon": False, "axes.spines.top": False,
                         "axes.spines.right": False})
    GAMMAS = ((0.35, "035"), (0.45, "045"), (0.50, "050"))
    Ls = [int(x) for x in (lengths or (50, 100, 150, 200, 250))]
    fig, axes = plt.subplots(1, len(Ls), figsize=(2.62 * len(Ls), 3.05), sharey=False)
    axes = [axes] if len(Ls) == 1 else list(axes)
    dd, dv = [], []

    for ax, L in zip(axes, Ls):
        ref = out_path(cfg, f"gamma_diversity_L{L}")
        if os.path.exists(ref):
            un = json.load(open(ref))
            gs = sorted(un, key=float)
            ax.plot([un[g]["designable_pct"] for g in gs],
                    [un[g]["clusters_per_designable"] for g in gs],
                    "-o", color=MUTED, lw=1.0, ms=3.2, mfc=SURF, mew=1.0, zorder=1, alpha=.85,
                    label=r"released sampler, hard binning ($\gamma$ .30-.60)")
        cx, cy, sx, sy = [], [], [], []
        for gam, gg in GAMMAS:
            fc = out_path(cfg, f"steered_eval_c{gg}L{L}_r{L}")
            fs = out_path(cfg, f"steered_eval_g{gg}L{L}_r{L}")
            if not (os.path.exists(fc) and os.path.exists(fs)):
                continue
            c = json.load(open(fc))["native"]
            a = json.load(open(fs))["A"]
            x0, y0 = c["designable_resampled_pct"], div(c)
            x1, y1 = a["designable_resampled_pct"], a["clusters_per_designable"]
            cx.append(x0); cy.append(y0); sx.append(x1); sy.append(y1)
            dd.append(x1 - x0); dv.append(y1 - y0)
            # the paired delta, drawn as the thing it is
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=2,
                        arrowprops=dict(arrowstyle="-|>,head_width=.17,head_length=.38",
                                        color=INK2, lw=1.0, alpha=.55,
                                        shrinkA=5.0, shrinkB=5.0))
            # gamma labelled once per pair, at the arrow's MIDPOINT. Anchoring it to either
            # endpoint put it on top of a neighbouring mark whenever two pairs ran close
            # together (L=150's 0.35 landed under a diamond, L=200/250's 0.50 and 0.45 on each
            # other). A midpoint is empty by construction -- the arrow spans the gap between
            # the only two marks that could be there -- and the surface-coloured box keeps it
            # legible where it crosses the reference curve.
            ax.annotate(f"{gam:.2f}", ((x0 + x1) / 2, (y0 + y1) / 2), zorder=5,
                        textcoords="offset points", xytext=(0, 5 if y1 < y0 else -9),
                        color=INK2, fontsize=6.5, ha="center",
                        bbox=dict(boxstyle="round,pad=.12", fc=SURF, ec="none", alpha=.8))
        if cx:
            # 2px surface ring so an arrow passing under a mark does not touch its fill
            ax.scatter(cx, cy, s=46, marker="o", color=BLUE, zorder=4,
                       edgecolor=SURF, linewidth=1.2, label=r"unsteered control, soft binning")
            ax.scatter(sx, sy, s=54, marker="D", color=ORANGE, zorder=4,
                       edgecolor=SURF, linewidth=1.2,
                       label=rf"steered A, soft binning ($\eta$={cfg['eta']})")
        ax.set_title(f"L = {L}", color=INK, fontsize=9)
        ax.set_xlabel("designable (%)")
        if ax is axes[0]:
            ax.set_ylabel(ylab)
        ax.grid(alpha=.18, lw=.5)

    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=7.5, bbox_to_anchor=(.5, -.055))
    n = len(dd)
    gain = sum(1 for v in dv if v > 0)
    fig.suptitle(
        "Each arrow is one seed-matched pair: unsteered control to steered, same $\gamma$. "
        "Up and to the right is better" + (
            f"\nsteering moves the pair by {sum(dd)/n:+.1f} pp designability and "
            f"{sum(dv)/n:+.3f} diversity on average; diversity improves in {gain} of {n}"
            if n else ""),
        fontsize=8.5, color=INK2, y=1.04)
    fig.tight_layout()
    out = os.path.join(cfg["out_dir"], "pareto_by_length.png")
    fig.savefig(out, dpi=190, bbox_inches="tight")
    print(f"wrote {out}", flush=True)
    if n:
        print(f"  {n} seed-matched pairs: mean d_designability {sum(dd)/n:+.2f} pp, "
              f"mean d_diversity {sum(dv)/n:+.4f}, diversity improved in {gain}/{n}", flush=True)


def pareto_reward(cfg, *lengths):
    """Designability against the REWARD ITSELF, per length, with the same paired arrows.

    The sibling of `pareto`, on the axis steering actually optimises. Both plots show the same
    fifteen seed-matched pairs; only the y quantity differs, and that is the whole finding:
    **[measured]** the reward rises in 15/15 cells (mean +0.090, ~24%, 21.5 sigma at L=250)
    while Foldseek diversity falls (-0.200, 5.5 sigma). Algorithm 1 optimises what it is given;
    what it was given stopped tracking the target.

    y is the OFF-DIAGONAL reward -w'Kw. The plain V-statistic's diagonal is exactly sum w_i^2 =
    1/ESS, so it would rise from weight concentration alone; here ESS/N is 1.000 everywhere, but
    the off-diagonal form is the one that compares populations rather than weights.

    The control points are joined because gamma is itself a knob on this plane -- more sampling
    noise buys reward at the cost of designability -- so the question is the same as in `pareto`:
    does steering reach a point the gamma curve could not? Reward is NOT comparable across
    panels: h is calibrated per length, so each panel is a different functional, and eta is one
    number across all five while **[measured]** steering authority follows ~1/L (1.000 / 0.538 /
    0.313 / 0.263 / 0.225 relative to L=50). Normalising the feature map instead cannot fix that:
    rescaling features by c rescales h by the same c, since calibrate_bandwidth sets
    h = sqrt(median d2), leaving exp(-d^2/h^2) unchanged (h'/h 2.238010 vs c 2.238010 at L=50).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # the repo palette, checked against the five computable data-viz tests on this surface:
    # CVD dE 24.7 (target 8), normal-vision dE 33.6 (floor 15), contrast 4.30:1 and 3.12:1.
    SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983"
    BLUE, ORANGE = "#2a78d6", "#eb6834"
    plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": .8,
                         "figure.facecolor": SURF, "axes.facecolor": SURF,
                         "legend.frameon": False, "axes.spines.top": False,
                         "axes.spines.right": False})
    GAMMAS = ((0.35, "035"), (0.45, "045"), (0.50, "050"))
    Ls = [int(x) for x in (lengths or (50, 100, 150, 200, 250))]
    fig, axes = plt.subplots(1, len(Ls), figsize=(2.62 * len(Ls), 3.05))
    axes = [axes] if len(Ls) == 1 else list(axes)
    n_up = n_tot = 0

    def cell(tag, L, var):
        """(designability %, off-diagonal reward) for one run, or None."""
        ev = out_path(cfg, f"steered_eval{tag}_r{L}")
        rn = out_path(cfg, f"ab_run{tag}_r{L}")
        if not (os.path.exists(ev) and os.path.exists(rn)):
            return None
        e = json.load(open(ev)).get(var)
        r = json.load(open(rn))["runs"].get(var)
        if not (e and r):
            return None
        return e["designable_resampled_pct"], r["reward_uniform_offdiag"]

    for ax, L in zip(axes, Ls):
        cx, cy, sx, sy = [], [], [], []
        for gam, gg in GAMMAS:
            c = cell(f"_c{gg}L{L}", L, "native")
            a = cell(f"_g{gg}L{L}", L, "A")
            if not (c and a):
                continue
            cx.append(c[0]); cy.append(c[1]); sx.append(a[0]); sy.append(a[1])
            n_tot += 1; n_up += a[1] > c[1]
            ax.annotate("", xy=a, xytext=c, zorder=2,
                        arrowprops=dict(arrowstyle="-|>,head_width=.17,head_length=.38",
                                        color=INK2, lw=1.0, alpha=.55, shrinkA=5.0, shrinkB=5.0))
            # label at the arrow MIDPOINT: anchored to an endpoint it lands on a neighbouring mark
            ax.annotate(f"{gam:.2f}", ((c[0] + a[0]) / 2, (c[1] + a[1]) / 2), zorder=5,
                        textcoords="offset points", xytext=(0, 5 if a[1] < c[1] else -9),
                        color=INK2, fontsize=6.5, ha="center",
                        bbox=dict(boxstyle="round,pad=.12", fc=SURF, ec="none", alpha=.8))
        if cx:
            order = sorted(range(len(cx)), key=lambda i: cx[i])
            ax.plot([cx[i] for i in order], [cy[i] for i in order], "-", color=MUTED, lw=1.0,
                    zorder=1, alpha=.8)
            ax.scatter(cx, cy, s=46, marker="o", color=BLUE, zorder=4, edgecolor=SURF,
                       linewidth=1.2, label="unsteered control")
            ax.scatter(sx, sy, s=54, marker="D", color=ORANGE, zorder=4, edgecolor=SURF,
                       linewidth=1.2, label=rf"steered A ($\eta$={cfg['eta']})")
        ax.set_title(f"L = {L}", color=INK, fontsize=9)
        ax.set_xlabel("designable (%)")
        if ax is axes[0]:
            ax.set_ylabel(r"reward  $-w^{\top}Kw$  (off-diagonal)")
        ax.grid(alpha=.18, lw=.5)

    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, fontsize=7.5, bbox_to_anchor=(.5, -.05))
    fig.suptitle("The objective steering was GIVEN, on the same fifteen seed-matched pairs as "
                 "pareto_by_length.\n"
                 f"Up is the reward improving: it does so in {n_up} of {n_tot}. "
                 "Foldseek diversity, on the same runs, falls in 13 of 15.",
                 fontsize=8.5, color=INK2, y=1.04)
    fig.tight_layout()
    out = os.path.join(cfg["out_dir"], "pareto_reward_by_length.png")
    fig.savefig(out, dpi=190, bbox_inches="tight")
    print(f"wrote {out}", flush=True)
    print(f"  reward improved in {n_up}/{n_tot} seed-matched pairs", flush=True)


_STEPS = 400   # the published sampler's step count; s/step -> GPU-h per 100-sample run


def gpu_hours(cfg, *lengths):
    """Sampler-only GPU-hours: steered against unsteered, and where the extra goes.

    NO eval. The earlier version plotted job wall-clock, which folds in ProteinMPNN/ESMFold --
    a fixed cost paid identically by both arms, and **[measured]** 63% of the unsteered job at
    L=50 against 23% at L=250. That made steering look nearly free at short lengths (job 1.34x
    against a true sampler 2.56x) and is not the quantity anyone means by "what does steering
    cost".

    Reads experiments/out/timing_probe_r<L>.json, written by common.timing_probe on a PINNED
    a100_80gb with no eval, so every length is comparable. Run the `timing` stage for a length
    before expecting it here.

    Per step the sampler does 1 no_grad forward unsteered, and 4 forwards + 1 backward steered.
    That predicts ~6x, and it is only reached once the model's FLOPs dominate the fixed
    per-launch overhead: **[measured]** 5.48x at L=250 but 2.56x at L=50, where grad_chunk=4
    makes the UNSTEERED step 25 micro-batches of 4 particles and so overhead-bound. Raising
    nograd_chunk there saves 13%, against 0.7% at L=250 -- the same regime split, measured
    independently.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983"
    ORANGE = "#eb6834"
    RAMP = ["#86b6ef", "#2a78d6", "#104281"]   # ordinal: dL 0.189 / 0.190, lightest 2.06:1
    plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": .8,
                         "figure.facecolor": SURF, "axes.facecolor": SURF,
                         "legend.frameon": False, "axes.spines.top": False,
                         "axes.spines.right": False})

    want = [int(x) for x in (lengths or (50, 100, 150, 200, 250))]
    Ls, D = [], {}
    for L in want:
        f = out_path(cfg, f"timing_probe_r{L}")
        if not os.path.exists(f):
            print(f"  L={L}: no timing_probe_r{L}.json, skipped "
                  f"(run: steer_a.py <cfg> timing {L} 6 4)", flush=True)
            continue
        d = json.load(open(f))
        g = lambda k: d[k]["sec_per_step"] * _STEPS / 3600
        Ls.append(L)
        D[L] = dict(base=g("unsteered"), grad=g("grad only (reweight=F)"),
                    full=g("+weights k_fp=3"), adv=g("k_fp=3 lag=advance"))
    if not Ls:
        raise SystemExit("no timing_probe outputs found; run the `timing` stage first")

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.8, 3.6),
                                 gridspec_kw=dict(width_ratios=[1.5, 1]))
    xs = np.arange(len(Ls)); w = 0.38
    bse = np.array([D[L]["base"] for L in Ls])
    grd = np.array([D[L]["grad"] for L in Ls])
    ful = np.array([D[L]["full"] for L in Ls])

    # unsteered drawn in the SAME colour as the steered stack's base segment: they are the
    # same quantity, and the eye should read the extra as sitting on top of it.
    ax.bar(xs - w / 2, bse, w * .92, color=RAMP[0], edgecolor=SURF, linewidth=1.0, zorder=3,
           label=r"unsteered  $\eta$=0  (1 fwd)")
    ax.bar(xs + w / 2, bse, w * .92, color=RAMP[0], edgecolor=SURF, linewidth=1.4, zorder=3)
    ax.bar(xs + w / 2, grd - bse, w * .92, bottom=bse, color=RAMP[1], edgecolor=SURF,
           linewidth=1.4, zorder=3, label=r"+ $\nabla_x\Psi$   (fwd + bwd)")
    ax.bar(xs + w / 2, ful - grd, w * .92, bottom=grd, color=RAMP[2], edgecolor=SURF,
           linewidth=1.4, zorder=3, label=r"+ $\partial_t\Psi$ look-ahead   (2 fwd)")
    for x, b, f_ in zip(xs, bse, ful):
        ax.annotate(f"{b:.2f}", (x - w / 2, b), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=6.8, color=INK2)
        ax.annotate(f"{f_:.2f}", (x + w / 2, f_), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=6.8, color=INK2)
    ax.set_xticks(xs); ax.set_xticklabels(Ls)
    ax.set_xlabel("sequence length"); ax.set_ylabel("GPU-hours per 100 samples")
    ax.set_title("sampler only, no eval  (400 steps, N=100, pinned A100)", color=INK, fontsize=9)
    ax.legend(loc="upper left", fontsize=7.2); ax.grid(axis="y", alpha=.18, lw=.5, zorder=0)

    r = ful / bse
    bx.plot(Ls, r, "-o", color=RAMP[2], lw=1.6, ms=6, mfc=SURF, mew=1.6, zorder=3,
            label="steered / unsteered")
    bx.plot(Ls, np.array([D[L]["adv"] for L in Ls]) / bse, "--s", color=ORANGE, lw=1.3, ms=5,
            mfc=SURF, mew=1.3, zorder=3, label='with lag="advance"')
    bx.axhline(6.0, color=MUTED, lw=1.0, ls=":", zorder=1)
    bx.annotate(r"4 fwd + 1 bwd $\approx$ 6x", (Ls[-1], 6.0), textcoords="offset points",
                xytext=(-4, 5), ha="right", fontsize=6.8, color=MUTED)
    for L, v in zip(Ls, r):
        bx.annotate(f"{v:.2f}x", (L, v), textcoords="offset points", xytext=(0, -12),
                    ha="center", fontsize=6.8, color=INK2)
    bx.set_xlabel("sequence length"); bx.set_ylabel(r"cost of steering  ($\times$ unsteered)")
    bx.set_title("the ratio is not a constant", color=INK, fontsize=9)
    bx.set_ylim(0, 7); bx.legend(loc="lower right", fontsize=7.2)
    bx.grid(alpha=.18, lw=.5, zorder=0)

    fig.suptitle("Per step the sampler does 1 forward unsteered and 4 forwards + 1 "
                 "backward steered.\nThe " r"$\approx$" "6x that predicts is only "
                 "reached once the model's FLOPs dominate the per-launch overhead.",
                 fontsize=8.5, color=INK2, y=1.04)
    fig.tight_layout()
    out = os.path.join(cfg["out_dir"], "gpu_hours.png")
    fig.savefig(out, dpi=190, bbox_inches="tight")
    print(f"wrote {out}", flush=True)
    for L in Ls:
        d = D[L]
        print(f"  L={L:<4} unsteered {d['base']:6.2f}  steered {d['full']:6.2f} GPU-h "
              f"({d['full']/d['base']:.2f}x)   advance {d['adv']/d['base']:.2f}x", flush=True)


def pareto_fk(cfg, *modes):
    """Designability against the reward, for FK's three potentials -- the gradient-free analogue
    of `pareto_reward`, with resample stride on the arrows where that plot has gamma.

    One panel per potential type, one arrow per stride, all at one gamma and one length because
    that is the controlled comparison: branch A and every FK variant share the reward, the
    bandwidth, the schedule and the seed, so the ONLY difference is how the log-weights are made.
    A's point is drawn in every panel as the gradient reference FK is being measured against.

    y is recomputed from the saved clouds rather than read from ab_run's json, because the
    sweep's jobs run concurrently and each rewrites that file with only its own variant.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch as _t
    from steering.rewards import ContactRepulsion

    SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983"
    BLUE, GREEN, ORANGE = "#2a78d6", "#008300", "#eb6834"
    plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": .8,
                         "figure.facecolor": SURF, "axes.facecolor": SURF,
                         "legend.frameon": False, "axes.spines.top": False,
                         "axes.spines.right": False})
    L, tag = cfg["n_res"], cfg.get("tag", "")
    ev = json.load(open(out_path(cfg, f"steered_eval{tag}_r{L}")))
    pot = None

    def cell(var, y):
        """(designable %, y) for one variant, or None if either half is missing."""
        nonlocal pot
        f = os.path.join(cfg["out_dir"], f"ab_{var}{tag}_r{L}.pt")
        if var not in ev or not os.path.exists(f):
            return None
        if y == "diversity":
            return (ev[var]["designable_resampled_pct"], ev[var]["clusters_per_designable"],
                    ev[var]["n_designable_unique"])
        d = _t.load(f, map_location="cpu", weights_only=False)
        if pot is None:
            pot = _steered_reward(cfg, d["h"])
        # under the cloud's OWN weights: for FK the weights ARE the steering, so uniform
        # weights would score a population the sampler does not claim to have produced
        return (ev[var]["designable_resampled_pct"],
                float(pot.value_offdiag(d["x"].double(), d["w"].double())),
                ev[var]["n_designable_unique"])

    from steer_f import FREQS, MODES          # the sweep is defined by the driver, not here
    modes = modes or MODES
    # Both axes, always: the reward is what the sampler OPTIMISES, so showing it alone would
    # report the objective as though it were the result. Foldseek clusters are the held-out one.
    rlab = {"contact": r"reward  $-w^{\top}Kw$  (off-diagonal)"}.get(
        cfg.get("reward", "contact"), "steered reward  $-\sum_c m_c^2$  (off-diagonal)")
    for y, ylab, fsuf in (("reward", rlab, ""),
                          ("diversity", "diversity  (clusters / designable)", "_div")):
        _draw_fk(cfg, cell, modes, y, ylab, fsuf, L, tag, plt, FREQS,
                 (SURF, INK, INK2, MUTED, BLUE, GREEN, ORANGE))


def _steered_reward(cfg, h):
    """The functional the run actually optimised, so the reward axis is that run's objective."""
    kind = cfg.get("reward", "contact")
    if kind == "contact":
        return ContactRepulsion(_ab_features(cfg), h=h)
    from steering.rewards import ClusterDiversity
    from proteina import foldseek_labeller, tm_labeller
    work = os.path.join(cfg["out_dir"], f"paretoscore_{kind}")
    os.makedirs(work, exist_ok=True)
    return ClusterDiversity(foldseek_labeller(work) if kind == "foldseek" else tm_labeller())


def _draw_fk(cfg, cell, modes, y, ylab, fsuf, L, tag, plt, FREQS, pal):
    import os
    SURF, INK, INK2, MUTED, BLUE, GREEN, ORANGE = pal
    nat, A = cell("native", y), cell("A", y)
    fig, axes = plt.subplots(1, len(modes), figsize=(3.05 * len(modes), 3.5))
    axes = [axes] if len(modes) == 1 else list(axes)
    n_up = n_tot = 0
    for ax, m in zip(axes, modes):
        sx, sy, su = [], [], []
        for q, fq in enumerate(FREQS):
            s = cell(f"{m}{fq}", y)
            if not (s and nat):
                continue
            sx.append(s[0]); sy.append(s[1]); su.append(s[2])
            n_tot += 1; n_up += s[1] > nat[1]
            ax.annotate("", xy=s[:2], xytext=nat[:2], zorder=2,
                        arrowprops=dict(arrowstyle="-|>,head_width=.17,head_length=.38",
                                        color=INK2, lw=1.0, alpha=.55, shrinkA=5.0, shrinkB=5.0))
            f = 0.26 + 0.20 * q                       # stagger along the arrow, not across it
            ax.annotate(f"{fq}", (nat[0] + f * (s[0] - nat[0]), nat[1] + f * (s[1] - nat[1])),
                        zorder=5, color=INK2, fontsize=6.5, ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=.14", fc=SURF, ec="none", alpha=.85))
        area = lambda u: 14.0 + 0.52 * u          # marker area tracks surviving structures
        if nat:
            ax.scatter([nat[0]], [nat[1]], s=area(nat[2]), marker="o", color=BLUE, zorder=4,
                       edgecolor=SURF, linewidth=1.2, label="unsteered control")
        if A:
            ax.scatter([A[0]], [A[1]], s=area(A[2]), marker="s", color=GREEN, zorder=4,
                       edgecolor=SURF, linewidth=1.2,
                       label=rf"mean-field A ($\eta$={cfg['eta']}, gradient)")
        if sx:
            ax.scatter(sx, sy, s=[area(u) for u in su], marker="D", color=ORANGE, zorder=4,
                       edgecolor=SURF, linewidth=1.2,
                       label=rf"FK potential ($\eta$={cfg['eta']}, gradient-free)")
        ax.set_title(f"'{m}'", color=INK, fontsize=9)
        ax.set_xlabel("designable (%)")
        if ax is axes[0]:
            ax.set_ylabel(ylab)
        ax.grid(alpha=.18, lw=.5)

    h, lab = [], []
    for a in axes:                                  # a panel with no FK point lacks that handle
        for hh, ll in zip(*a.get_legend_handles_labels()):
            if ll not in lab:
                h.append(hh); lab.append(ll)
    fig.tight_layout(rect=(0, 0.10, 1, 0.86))
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=7.5, bbox_to_anchor=(.5, 0.005))
    fig.suptitle(f"FK potential steering at gamma={cfg['temperature']}, L={L}: same reward, "
                 f"bandwidth, schedule and seed as mean-field A.\nArrows are the resample "
                 f"stride, marker area the unique designable structures left after resampling. "
                 f"Up is {y} improving: {n_up} of {n_tot}.",
                 fontsize=8.5, color=INK2, y=0.99)
    fname = os.path.join(cfg["out_dir"], f"pareto_fk{fsuf}{tag}.png")
    fig.savefig(fname, dpi=170, facecolor=SURF, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fname}", flush=True)


def _rarefied(tag, L, n, cache="experiments/out/rarefy_cache.json", reps=4):
    """Native's own designable pool cut to n. clusters/designable rises as n falls, so a steered
    point with fewer survivors must be compared against this, not against native's full pool."""
    import glob
    import random
    import shutil
    from proteina import foldseek_clusters
    key = f"{tag}:{n}"
    c = json.load(open(cache)) if os.path.exists(cache) else {}
    if key in c:
        return c[key]
    pool = sorted(glob.glob(f"experiments/out/steered_eval{tag}/native/fs_pool/*.pdb"))
    n = min(n, len(pool))
    tmp = f"/tmp/claude-1000372/rf_{os.getpid()}"
    vals = []
    for r in range(reps):
        d = f"{tmp}/pool"
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
        for f in random.Random(r).sample(pool, n):
            shutil.copy(f, d)
        vals.append(foldseek_clusters(d, f"{tmp}/t")[0] / n)
    shutil.rmtree(tmp, ignore_errors=True)
    c[key] = sum(vals) / len(vals)
    json.dump(c, open(cache, "w"), indent=1)
    return c[key]


def pareto_cases(cfg, which="length", *lengths):
    """`pareto`'s axes for the FK campaigns: one panel per length, or per seed at L=100."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983"
    BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#008300"
    plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": .8,
                         "figure.facecolor": SURF, "axes.facecolor": SURF,
                         "legend.frameon": False, "axes.spines.top": False,
                         "axes.spines.right": False})
    # which cluster count feeds y: the run is scored on the metric it STEERED on, so a
    # tm-steered campaign is read with n_clusters_tmalign and a foldseek one with n_clusters
    tm = cfg.get("reward") == "tm"
    # tmsc steers on the paper's OWN TM-Sc. column (mean pairwise TM, down) rather than on a
    # cluster count, so it reads mean_tm and the y axis is inverted to keep "up is better".
    tmsc = cfg.get("reward") == "tmsc"
    pre, base = (("_tmscL", "_tmscL100") if tmsc else
                 ("_tmL", "_tmL100") if tm else ("_fsL", "_foldseekL100"))
    Ls = [int(x) for x in lengths] or [50, 100, 150, 200, 250]
    CASES = ([(f"L = {L}", base if L == 100 else f"{pre}{L}", L)
              for L in Ls] if which == "length" else
             [(f"seed {s}", base if s == 7 else f"{pre}100s{s}", 100)
              for s in (7, 8, 9, 10)])
    VAR = ((("max20", ORANGE, "D"), ("add20", GREEN, "s")) if (tm or tmsc)
           else (("diff80", ORANGE, "D"), ("add20", GREEN, "s")))
    div = ((lambda e: e["mean_tm"]) if tmsc
           else (lambda e: e["n_clusters_tmalign"] / e["n_designable_unique"]) if tm
           else (lambda e: e["clusters_per_designable"]))
    ylab = ("TM-Sc.  (mean pairwise TM, axis inverted)" if tmsc
            else "diversity  (TM clusters / designable)" if tm
            else "diversity  (clusters / designable)")
    metric = "TM-Sc." if tmsc else "TM-score" if tm else "Foldseek"
    better = (lambda a, b: a < b) if tmsc else (lambda a, b: a > b)
    fig, axes = plt.subplots(1, len(CASES), figsize=(2.62 * len(CASES), 3.2))
    axes = [axes] if len(CASES) == 1 else list(axes)
    up = tot = 0
    for ax, (lab, tag, L) in zip(axes, CASES):
        p_ = out_path(cfg, f"steered_eval{tag}_r{L}")
        if not os.path.exists(p_):
            ax.set_title(f"{lab} (missing)", color=MUTED, fontsize=9)
            continue
        ev = json.load(open(p_))
        c = ev["native"]
        x0, y0 = c["designable_resampled_pct"], div(c)
        for v, col, mk in VAR:
            if v not in ev:
                continue
            e = ev[v]
            x1, y1 = e["designable_resampled_pct"], div(e)
            tot += 1; up += better(y1, y0)
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=2,
                        arrowprops=dict(arrowstyle="-|>,head_width=.17,head_length=.38",
                                        color=INK2, lw=1.0, alpha=.55, shrinkA=5.0, shrinkB=5.0))
            ax.scatter([x1], [y1], s=54, marker=mk, color=col, zorder=4, edgecolor=SURF,
                       linewidth=1.2, label=f"FK {v}")
            ax.annotate(f"n={e['n_designable_unique']}", (x1, y1), textcoords="offset points",
                        xytext=(6, 2), color=INK2, fontsize=6.0, ha="left")
        ax.scatter([x0], [y0], s=46, marker="o", color=BLUE, zorder=4, edgecolor=SURF,
                   linewidth=1.2, label="unsteered control")
        ax.annotate(f"n={c['n_designable_unique']}", (x0, y0), textcoords="offset points",
                    xytext=(6, 2), color=INK2, fontsize=6.0, ha="left")
        ax.set_title(lab, color=INK, fontsize=9)
        ax.set_xlabel("designable (%)")
        if tmsc:
            ax.invert_yaxis()
        if ax is axes[0]:
            ax.set_ylabel(ylab)
        ax.grid(alpha=.18, lw=.5)
    h, lb = [], []
    for a in axes:
        for hh, ll in zip(*a.get_legend_handles_labels()):
            if ll not in lb:
                h.append(hh); lb.append(ll)
    fig.tight_layout(rect=(0, 0.11, 1, 0.86))
    fig.legend(h, lb, loc="lower center", ncol=3, fontsize=7.5, bbox_to_anchor=(.5, 0.005))
    fig.suptitle(
        f"FK potential steering on {metric}{'' if tmsc else ' cluster'} diversity, gamma=0.45, "
        f"{'L=100' if which == 'seed' else 'seed 7'}."
        f"  Up is diversity improving: {up} of {tot}.\n" + (
            "mean pairwise TM is a U-statistic, so unlike clusters/designable it does not rise "
            "as n falls; `n=` is annotated as context, not as a correction."
            if tmsc else
            "clusters/designable rises as n falls, so "
            "`n=` is annotated: a variant holding fewer structures scores higher for that alone."),
        fontsize=8.5, color=INK2, y=0.99)
    fname = os.path.join(cfg["out_dir"],
                         f"pareto_fk{'_tmsc' if tmsc else '_tm' if tm else ''}_by_{which}.png")
    fig.savefig(fname, dpi=170, facecolor=SURF, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fname}", flush=True)
