"""The fixed FK potentials against fkd_class.py:111-131 and Eq. 5, on hand-made rewards."""
import torch
from steering.steerer import log_potential

def run(kind, rs, r_star, lam=1.0, seed=None):
    """Mimic run(): a enters each checkpoint at 0, reindex moves it into logP, then the terminal."""
    if seed is None:
        seed = float("-inf") if kind == "max" else 0.0
    a, logP, r_prev, per = torch.zeros(1), torch.zeros(1), torch.full((1,), seed), []
    for r in rs:
        a, r_prev = log_potential(kind, lam, torch.tensor([r]), torch.zeros(1), r_prev)
        per.append(round(float(a), 4)); logP = logP + a
    term = lam * r_star - logP                        # finalise's rule, unchanged
    return per, round(float(term), 4), round(float(logP + term), 4)

rs, r_star = [0.1, 0.5, 0.3, 0.8], 0.6
REF = {"max": [0.1, 0.5, 0.5, 0.8], "add": [0.1, 0.6, 0.9, 1.7],
       "diff": [0.1, 0.4, -0.2, 0.5], "rt": [0.1, 0.5, 0.3, 0.8]}   # fkd_class.py, by hand
print(f"r = {rs}, r* = {r_star}\n{'kind':5s} {'per-checkpoint (ours)':28s} {'reference':24s} match  total=lam*r*?")
bad = 0
for k in ("diff", "max", "add", "rt"):
    per, term, tot = run(k, rs, r_star)
    ok = per == REF[k]; eq5 = abs(tot - r_star) < 1e-9
    bad += (not ok) + (not eq5)
    print(f"{k:5s} {per!s:28s} {REF[k]!s:24s} {'YES' if ok else 'NO ':5s}  {tot} {'YES' if eq5 else 'NO'}")

print("\nnegative rewards, as ours are (psi = -2(Sw) <= 0):")
neg = [-0.3, -0.1, -0.2, -0.05]
p_new, _, _ = run("max", neg, -0.15)                  # seed None -> -inf
p_old, _, _ = run("max", neg, -0.15, seed=0.0)        # the old floor
print(f"  max, seed -inf (new default): {p_new}")
print(f"  max, seed 0    (old default): {p_old}   <- inert")
bad += p_new != [-0.3, -0.1, -0.1, -0.05]
print("\nALL CHECKS PASS" if not bad else f"\n{bad} CHECK(S) FAILED")
