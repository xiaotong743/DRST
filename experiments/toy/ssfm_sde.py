"""Replication of SSFM's non-linear SDE experiment (arXiv:2606.01086 sec 4.1).

Their experiments/sde/sde.py, in torch. No teacher: the SDE is known, so the matching target is
one exact Euler-Maruyama step of it. The number to reproduce is the fitted STRONG order, which
they take over the FINE regime only (sde.py:684-686, printed at :695); the coarse counts are
reported, not fitted.

Usage: PYTHONPATH=experiments python -m toy.ssfm_sde configs/ssfm_sde.yaml <stage>
"""
import json
import sys

import torch
import yaml

from .run import out_path
from steering.flowmap import (BrownianPath, EulerMaruyamaFlowMap, MLPStep, _sq,
                              chen_combine, sample_levy)

BETA_MIN, BETA_MAX = 0.1, 20.0            # sde.py:16-17; _beta at :20-21, so 0.1 + 19.9 t


def beta(t):
    return BETA_MIN + (BETA_MAX - BETA_MIN) * t


def b_drift(t, x):
    """sde.py:24-25 `b_drift` -- the double well."""
    return x - x ** 3


def sigma(t):
    """sde.py:28-29 `sigma_diffusion`."""
    return beta(t).sqrt() if torch.is_tensor(t) else beta(torch.as_tensor(t)).sqrt()


def fine_em_to(x0, s, n_sub):
    """sde.py:130-141 `fine_em_to_xs`: reach time s by n_sub Euler-Maruyama substeps. Fresh
    noise, since only the
    interval [s, s+h] has to share a path with the target."""
    x, t = x0, torch.zeros_like(s)
    h = s / n_sub
    for _ in range(n_sub):
        z = torch.randn(x.shape, dtype=x.dtype, device=x.device)
        x = x + b_drift(t, x) * h + sigma(t) * h.sqrt() * z
        t = t + h
    return x


def _step_target(ys, s, h, I0):
    return ys + b_drift(s, ys) * h + sigma(s) * I0


def train(cfg):
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(cfg["seed"])
    dev = cfg["device"]
    mk = lambda: EulerMaruyamaFlowMap(MLPStep(dim=1, n_levy=cfg["n_coeffs"], width=cfg["width"],
                                              depth=cfg["depth"], dt_feat=False)).to(dev)
    fm, ema = mk(), mk()
    ema.load_state_dict(fm.state_dict())
    opt = torch.optim.Adam(fm.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["iters"])
    n, dt, eps = cfg["batch"], cfg["dt"], cfg["t_eps"]
    nm = int(n * cfg["eta"])
    hist = []
    for it in range(cfg["iters"]):
        # matching: short steps, h ~ U(dt/2, dt), s ~ U(t_eps, 1-h)      sde.py:179-184
        h = dt / 2 + (dt / 2) * torch.rand(nm, 1, device=dev)
        s = eps + (1.0 - h - eps) * torch.rand(nm, 1, device=dev)
        ys = fine_em_to(torch.randn(nm, 1, device=dev), s, cfg["n_sub"])
        I = sample_levy(h, cfg["n_coeffs"])
        lm = _sq(fm(ys, s, s + h, I), _step_target(ys, s, h, I[0]).detach(), h)

        # semigroup: long steps, h ~ U(dt, h_max). The halves are sampled and Chen-combined, so
        # both sides of the loss are driven by one path without needing a Brownian tree.
        nd = n - nm
        h2 = dt + (cfg["h_max"] - dt) * torch.rand(nd, 1, device=dev)
        s2 = eps + torch.clamp(1.0 - h2 - eps, min=1e-6) * torch.rand(nd, 1, device=dev)
        y2 = fine_em_to(torch.randn(nd, 1, device=dev), s2, cfg["n_sub"])
        L, R = sample_levy(h2 / 2, cfg["n_coeffs"]), sample_levy(h2 / 2, cfg["n_coeffs"])
        with torch.no_grad():
            mid = s2 + h2 / 2
            tgt = ema(ema(y2, s2, mid, L), mid, s2 + h2, R)
        ld = _sq(fm(y2, s2, s2 + h2, chen_combine(L, R)), tgt, h2)

        loss = cfg["eta"] * lm + (1 - cfg["eta"]) * ld
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(fm.parameters(), cfg["clip"])
        opt.step(); sched.step()
        with torch.no_grad():
            for q, r in zip(ema.parameters(), fm.parameters()):
                q.mul_(cfg["ema"]).add_(r, alpha=1 - cfg["ema"])
        if it % cfg["log_every"] == 0:
            hist.append(dict(it=it, loss=float(loss), matching=float(lm), semigroup=float(ld)))
            print(f"{it:7d}  loss {float(loss):11.6f}  match {float(lm):11.6f}  "
                  f"semi {float(ld):11.6f}", flush=True)
    torch.save(ema.state_dict(), cfg["out"])   # sde.py:651 serialises the EMA, not the student
    json.dump(hist, open(out_path(cfg, "ssfm_sde_train"), "w"), indent=1)


def _reference(path, x0):
    """Euler-Maruyama on the finest level, returned as the WHOLE trajectory with x0 prepended."""
    x, traj = x0, [x0]
    for k in range(2 ** path.levels):
        a, b = path.span(0, k)
        ta = torch.full_like(x, a)
        x = x + b_drift(ta, x) * (b - a) + sigma(ta) * path.levy(0, k)[0]
        traj.append(x)
    return torch.stack(traj)


def _rollout(fm, path, x0, level):
    x, traj = x0, [x0]
    for idx in range(2 ** (path.levels - level)):
        a, b = path.span(level, idx)
        x = fm(x, torch.full_like(x, a), torch.full_like(x, b), path.levy(level, idx))
        traj.append(x)
    return torch.stack(traj)


def _regime(cfg, fm, levels, steps, label):
    torch.manual_seed(cfg["seed"] + 7)
    path = BrownianPath((cfg["eval_n"], 1), levels, n_levy=cfg["n_coeffs"],
                        device=cfg["device"])
    x0 = torch.randn(cfg["eval_n"], 1, device=cfg["device"], dtype=torch.float64)
    out = {}
    with torch.no_grad():
        ref = _reference(path, x0)
        print(f"\n{label}: reference {2 ** levels} steps", flush=True)
        print(f"{'steps':>6} {'strong RMSE':>13} {'h':>10}", flush=True)
        for k in steps:
            level = levels - int(k).bit_length() + 1
            # sde.py:379-381 compares the whole coarse trajectory against truth[::stride],
            # not just t=1; their strong order is defined under that metric. We report RMSE;
            # their Figure 2 plots the MSE, i.e. THIS VALUE SQUARED, and only their fine-regime
            # order fit roots it (sde.py:684-686), which is the regime ours matches directly.
            coarse = _rollout(fm, path, x0, level)
            err = float(((coarse - ref[::2 ** level]) ** 2).mean().sqrt())
            out[str(k)] = err
            print(f"{k:6d} {err:13.6f} {1.0 / k:10.6f}", flush=True)
    return out


def evaluate(cfg):
    import numpy as np
    torch.set_default_dtype(torch.float64)
    fm = EulerMaruyamaFlowMap(MLPStep(dim=1, n_levy=cfg["n_coeffs"], width=cfg["width"],
                                      depth=cfg["depth"], dt_feat=False)).to(cfg["device"])
    fm.load_state_dict(torch.load(cfg["out"], map_location=cfg["device"]))
    fm.eval().requires_grad_(False)
    coarse = _regime(cfg, fm, cfg["levels_coarse"], cfg["steps_coarse"], "coarse regime")
    fine = _regime(cfg, fm, cfg["levels_fine"], cfg["steps_fine"], "fine regime")
    ks = sorted(int(k) for k in fine)
    order = float(np.polyfit(np.log([1.0 / k for k in ks]), np.log([fine[str(k)] for k in ks]), 1)[0])
    print(f"\nfitted strong order (fine regime {ks}): {order:.3f}", flush=True)
    json.dump(dict(coarse=coarse, fine=fine, strong_order=order),
              open(out_path(cfg, "ssfm_sde_eval"), "w"), indent=1)


STAGES = {"train": train, "eval": evaluate}


def main(argv):
    STAGES[argv[1]](yaml.safe_load(open(argv[0])))


if __name__ == "__main__":
    main(sys.argv[1:])
