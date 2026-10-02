"""1:1 reproduction check: does the current code still produce the stored numbers?

Comment trims, the A/B split, the no_grad path at lam=0 and the microbatching all rewrote parts
of `steer`, so a passing test suite is not the same as reproducing stored results. Each stage
compares against a snapshot taken before those changes.

Stages:
    steered <tag> ...   compare experiments/out/prefix_snapshot/ab_run_<tag>.json with the current one
    toy [sched ...]     re-run fig2/fig3 into a temp dir and compare against the snapshot
Usage: python experiments/verify.py configs/simulation.yaml <stage> [args]
"""
import json
import os
import subprocess
import sys

import yaml

SNAP = "experiments/out/prefix_snapshot"


def _leaves(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _leaves(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _leaves(v, f"{path}[{i}]")
    else:
        yield path, obj


def _compare(a, b, label, atol=0.0, skip=()):
    """Element-wise compare two nested structures. atol=0 demands bitwise equality."""
    la, lb = dict(_leaves(a)), dict(_leaves(b))
    only = (set(la) ^ set(lb)) - set(skip)
    worst, worst_at, n, bad = 0.0, None, 0, 0
    for k in sorted(set(la) & set(lb)):
        if any(s in k for s in skip):
            continue
        x, y = la[k], lb[k]
        if isinstance(x, bool) or isinstance(y, bool) or not isinstance(x, (int, float)):
            if x != y:
                bad += 1
                if worst_at is None:
                    worst_at = f"{k}: {x!r} vs {y!r}"
            continue
        n += 1
        d = abs(float(x) - float(y))
        if d > worst:
            worst, worst_at = d, f"{k}: {x!r} vs {y!r}"
        if d > atol:
            bad += 1
    verdict = "IDENTICAL" if (bad == 0 and not only) else f"{bad} MISMATCH"
    print(f"  {label:<44} {n:5d} numbers  worst |d| = {worst:.3e}  -> {verdict}", flush=True)
    if only:
        print(f"      keys on one side only: {sorted(only)[:6]}", flush=True)
    if bad:
        print(f"      first difference: {worst_at}", flush=True)
    return bad == 0 and not only


def steered(cfg, *tags):
    """Compare stored vs current ab_run output. Tolerance 0: these must be bitwise equal unless
    grad_chunk differs, in which case chunk-invariance allows float-level reordering only."""
    ok = True
    for tag in tags:
        cur = f"experiments/out/ab_run_{tag}.json"
        snap = os.path.join(SNAP, os.path.basename(cur))
        if not (os.path.exists(cur) and os.path.exists(snap)):
            print(f"  {tag}: missing ({'current' if not os.path.exists(cur) else 'snapshot'})")
            ok = False
            continue
        # Refuse to "pass" by comparing a file with its own copy. If the current output has not
        # been regenerated since the snapshot was taken, there is nothing being tested.
        import hashlib
        h = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
        if h(cur) == h(snap) and os.path.getmtime(cur) <= os.path.getmtime(snap) + 1:
            print(f"  {tag:<44} NOT RE-RUN since the snapshot -- nothing compared", flush=True)
            ok = False
            continue
        a, b = json.load(open(snap)), json.load(open(cur))
        chunk_changed = a.get("grad_chunk") != b.get("grad_chunk")
        atol = 1e-6 if chunk_changed else 0.0
        note = " (grad_chunk differs: invariance, not bitwise)" if chunk_changed else ""
        ok &= _compare(a, b, tag + note, atol=atol, skip=("grad_chunk",))
    return ok


def _subset(snap, lams):
    """The snapshot restricted to `lams`. Every per-lambda list is indexed by position in
    snap["lams"], so a reduced re-run must be compared against the SAME positions -- otherwise
    J_kfp1[0..2] would line up lam 0,10,20 against lam 0,2,4 and report a spurious mismatch."""
    idx = [snap["lams"].index(l) for l in lams]      # raises if a lam is not in the snapshot
    out = dict(snap, lams=list(lams))
    for k, v in snap.items():
        if isinstance(v, list) and k != "lams" and len(v) == len(snap["lams"]):
            out[k] = [v[i] for i in idx]
    return out


def toy(cfg, *scheds):
    """Re-run the 1-D reproduction into a temp dir and compare with the snapshot.

    The temp dir is PRIVATE to this process. It used to be the fixed "/tmp/verify_toy", so two
    concurrent checks -- or one left running from a dead shell -- wrote the same
    fig3_sd0.5_<sched>.json and each then compared whatever the other had just written.
    """
    import tempfile
    tmp = tempfile.mkdtemp(prefix=f"verify_toy_{os.getpid()}_")
    py, ok = sys.executable, True
    sub = cfg.get("verify_lams")          # subset of lams; None re-runs the full sweep
    extra = [f"--set.lams={json.dumps(sub)}"] if sub else []
    for sched in (scheds or ("const", "sqrt1mt", "memoryless")):
        name = f"fig3_sd0.5_{sched}.json"
        r = subprocess.run([py, "experiments/run.py", "configs/simulation.yaml", "fig3", sched,
                            f"--set.out_dir={tmp}"] + extra, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  fig3 {sched}: FAILED\n{r.stdout[-1500:]}{r.stderr[-1500:]}")
            ok = False
            continue
        snap = json.load(open(os.path.join(SNAP, name)))
        if sub:
            snap = _subset(snap, sub)
        ok &= _compare(snap, json.load(open(os.path.join(tmp, name))), f"fig3 {sched}", atol=0.0)
    return ok


STAGES = {"steered": steered, "toy": toy}


def main(argv):
    if len(argv) < 2:
        raise SystemExit(__doc__)
    cfg = yaml.safe_load(open(argv[0]))
    import torch
    torch.set_default_dtype(getattr(torch, cfg.get("torch_dtype", "float64")))
    rest = argv[2:]
    for a in rest:
        if a.startswith("--set"):
            k, v = a.split("=", 1)
            cfg[k.removeprefix("--set").lstrip(".")] = yaml.safe_load(v)
    argv = [argv[0], argv[1]] + [a for a in rest if not a.startswith("--set")]
    if argv[1] not in STAGES:
        raise SystemExit(f"stage {argv[1]!r}; choose from {sorted(STAGES)}")
    print(f"=== 1:1 reproduction check: {argv[1]} ===", flush=True)
    ok = STAGES[argv[1]](cfg, *argv[2:])
    print(f"=== {'ALL IDENTICAL' if ok else 'DIFFERENCES FOUND'} ===", flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main(sys.argv[1:])
