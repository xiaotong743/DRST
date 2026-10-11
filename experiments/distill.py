"""Distil a pretrained flow model into a strong stochastic flow map (arXiv:2606.01086).

SSFM trains from scratch against the analytic score of a VP kernel. Our teacher is a trained
flow model, so the matching term regresses one Euler-Maruyama step of ITS reverse SDE instead --
the same step `Steerer._move` takes. The semigroup term is unchanged.

Usage: PYTHONPATH=experiments python -m distill configs/distill.yaml <stage>
"""
import json
import sys

import torch
import yaml

from proteina import out_path
from steering.flowmap import (BrownianPath, EulerMaruyamaFlowMap, MLPStep, _sq,
                              chen_combine, distill_loss, sample_levy)
from toy.run import base_model, trained as _teacher


def forward_sample(ref, n, s):
    """(x_t, x_0, x_1) on the interpolant the teacher was trained on, for a per-sample s [n,1]."""
    x1 = ref.sample_p1(n).reshape(-1, 1)
    x0 = torch.randn(n, 1, dtype=x1.dtype)
    return (1.0 - s) * x0 + s * x1, x0, x1


def simulate_to(teacher, n, s, n_sub):
    """Query points by SIMULATING the teacher from noise -- MFM's teacher mode, no data.

    One rollout passes through every intermediate state, so at scale this is cached rather than
    re-simulated; at 1-D toy cost it is cheaper to redo it.
    """
    x = teacher.prior(n)
    h = s / n_sub
    t = torch.zeros_like(s)
    for _ in range(n_sub):
        tv = t.reshape(-1)
        _, v = teacher.endpoint(x, tv)
        g = torch.as_tensor([teacher.eps(a) for a in tv.tolist()], dtype=x.dtype).reshape_as(s)
        x = x + (v + g * teacher.score(x, v, t)) * h + (2.0 * g * h).sqrt() * torch.randn_like(x)
        t = t + h
    return x


def _target(cfg, teacher, ref, n, s, h, I0):
    """(ys, target). The three arms differ ONLY here -- in where ys comes from and whose drift
    builds the target. The semigroup term is identical in all of them."""
    mode = cfg["mode"]
    g = lambda tv: torch.as_tensor([teacher.eps(a) for a in tv.flatten().tolist()],
                                   dtype=torch.float64).reshape_as(tv)
    if mode == "ssfm":                      # data + analytic conditional score, no teacher
        ys, x0, x1 = forward_sample(ref, n, s)
        gs = g(s)
        v = (x1 - ys) / (1.0 - s)
        score = -x0 / (1.0 - s)
    else:
        ys = (forward_sample(ref, n, s)[0] if mode == "teacher_interp"
              else simulate_to(teacher, n, s, cfg["n_sub"]))
        with torch.no_grad():
            _, v = teacher.endpoint(ys, s.reshape(-1))
        gs = g(s)
        score = teacher.score(ys, v, s)
    return ys, (ys + (v + gs * score) * h + (2.0 * gs).sqrt() * I0).detach()


def reference(teacher, x0, path):
    """The teacher's own SDE on the finest level, as the WHOLE trajectory with x0 prepended."""
    x, traj = x0, [x0]
    for k in range(2 ** path.levels):
        a, b = path.span(0, k)
        tv = torch.full((x.shape[0],), a, dtype=x.dtype)
        _, v = teacher.endpoint(x, tv)
        g = float(teacher.eps(a))
        x = x + (v + g * teacher.score(x, v, a)) * (b - a) + (2.0 * g) ** 0.5 * path.levy(0, k)[0]
        traj.append(x)
    return torch.stack(traj)


def rollout(fm, x0, path, level):
    x, traj = x0, [x0]
    col = lambda a: torch.full((x0.shape[0], 1), a, dtype=x0.dtype)
    for idx in range(2 ** (path.levels - level)):
        a, b = path.span(level, idx)
        x = fm(x, col(a), col(b), path.levy(level, idx))
        traj.append(x)
    return torch.stack(traj)


def train(cfg):
    torch.set_default_dtype(torch.float64)
    torch.set_default_device(cfg.get("device", "cpu"))
    torch.manual_seed(cfg["seed"])
    teacher, ref = _teacher(cfg), base_model(cfg)
    mk = lambda: EulerMaruyamaFlowMap(MLPStep(dim=1, n_levy=cfg["n_levy"], width=cfg["fm_width"]))
    fm, ema = mk(), mk()
    ema.load_state_dict(fm.state_dict())            # the teacher IS the student, as in SSFM
    opt = torch.optim.Adam(fm.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["iters"])
    n, dt, eps = cfg["batch"], cfg["dt"], cfg["t_eps"]
    nm, hist = int(n * cfg["eta"]), []
    for it in range(cfg["iters"]):
        # continuous (s, h) PER SAMPLE, as sde.py:179-184 and :238-244. The dyadic ladder this
        # replaced gave one interval to the whole batch and oversampled small h.
        h = dt / 2 + (dt / 2) * torch.rand(nm, 1)
        s = eps + (1.0 - h - eps) * torch.rand(nm, 1)
        I = sample_levy(h, cfg["n_levy"])
        ys, tgt = _target(cfg, teacher, ref, nm, s, h, I[0])
        lm = _sq(fm(ys, s, s + h, I), tgt, h)

        nd = n - nm
        h2 = dt + (cfg["h_max"] - dt) * torch.rand(nd, 1)
        s2 = eps + torch.clamp(1.0 - h2 - eps, min=1e-6) * torch.rand(nd, 1)
        y2 = (forward_sample(ref, nd, s2)[0] if cfg["mode"] != "teacher_sim"
              else simulate_to(teacher, nd, s2, cfg["n_sub"]))
        L, R = sample_levy(h2 / 2, cfg["n_levy"]), sample_levy(h2 / 2, cfg["n_levy"])
        ld = distill_loss(fm, ema, y2, s2, s2 + h2 / 2, s2 + h2, L, R, chen_combine(L, R))

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
    torch.save(ema.state_dict(), cfg["out"].replace(".pt", f"_{cfg['mode']}.pt"))   # sde.py:651 serialises the EMA, not the student
    json.dump(hist, open(out_path(cfg, f"distill_train_{cfg['mode']}"), "w"), indent=1)
    print(f"wrote {cfg['out']}", flush=True)


def _regime(cfg, fm, teacher, levels, steps, label):
    torch.manual_seed(cfg["seed"] + 7)
    path = BrownianPath((cfg["eval_n"], 1), levels, n_levy=cfg["n_levy"])
    x0 = teacher.prior(cfg["eval_n"])
    out = {}
    with torch.no_grad():
        ref = reference(teacher, x0, path)
        print(f"\n{label}: reference {2 ** levels} steps", flush=True)
        print(f"{'steps':>6} {'strong RMSE':>13} {'h':>10}", flush=True)
        for k in steps:
            level = levels - int(k).bit_length() + 1
            err = float(((rollout(fm, x0, path, level) - ref[::2 ** level]) ** 2).mean().sqrt())
            out[str(k)] = err
            print(f"{k:6d} {err:13.6f} {1.0 / k:10.6f}", flush=True)
    return out


def evaluate(cfg):
    """Strong (pathwise) error against the teacher's fine solve, over both regimes."""
    import numpy as np
    torch.set_default_dtype(torch.float64)
    torch.set_default_device(cfg.get("device", "cpu"))
    teacher = _teacher(cfg)
    fm = EulerMaruyamaFlowMap(MLPStep(dim=1, n_levy=cfg["n_levy"], width=cfg["fm_width"]))
    fm.load_state_dict(torch.load(cfg["out"].replace(".pt", f"_{cfg['mode']}.pt"),
                                  map_location=cfg.get("device", "cpu")))
    fm.eval().requires_grad_(False)
    coarse = _regime(cfg, fm, teacher, cfg["levels_coarse"], cfg["steps_coarse"], "coarse regime")
    fine = _regime(cfg, fm, teacher, cfg["levels_fine"], cfg["steps_fine"], "fine regime")
    ks = sorted(int(k) for k in fine)
    order = float(np.polyfit(np.log([1.0 / k for k in ks]),
                             np.log([fine[str(k)] for k in ks]), 1)[0])
    print(f"\nfitted strong order (fine regime {ks}): {order:.3f}", flush=True)
    json.dump(dict(coarse=coarse, fine=fine, strong_order=order),
              open(out_path(cfg, f"distill_eval_{cfg['mode']}"), "w"), indent=1)


STAGES = {"train": train, "eval": evaluate}


def main(argv):
    cfg = yaml.safe_load(open(argv[0]))
    for a in argv[2:]:
        if a.startswith("--mode="):
            cfg["mode"] = a.split("=", 1)[1]
    STAGES[argv[1]](cfg)


if __name__ == "__main__":
    main(sys.argv[1:])
