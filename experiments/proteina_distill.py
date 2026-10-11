"""Distil Proteina's own reverse SDE into a strong stochastic flow map, teacher-only, no cache.

The target is the PUBLISHED sampler at gamma = 1 -- Proteina itself, not the tempered one the
steering campaigns use -- so `temperature` here is 1.0 and the ODE tail at t > 0.99 is kept
because r3n_fm.py:523 is part of that sampler.

Query points come from ONE PERSISTENT CHAIN per particle, advanced by the very half-step the
semigroup term already takes: the forward at (y_m, m) that the term's second leg needs IS the next
step's forward at (y, u). Every teacher call is used twice, so query generation is free and a
training step costs 1 forward. Simulating to a random s instead would cost ~200 (45 GPU-days), and
distill.py's `simulate_to` does not transfer: 32 uniform-u steps give h*B(0) = 14.5, which diverges.

Usage: PYTHONPATH=experiments python experiments/proteina_distill.py configs/proteina_distill.yaml <stage>
"""
import json
import sys
import time

import torch

from proteina import _binning, _proteina, main, out_path
from steering.expmap import DTAU, GT, TAU, ExpStep, VField, abs_u, coeffs_em, coeffs_exp
from steering.flowmap import (BrownianPath, EulerMaruyamaFlowMap, _sq, chen_combine,
                              distill_loss, sample_levy)


def em_step(v, y, s, h, I0, gam):
    """Eq 15 in u: one EM step of the sampler's SDE on the SAME Brownian increment.

    Written from vf_to_score's own expression rather than the a/b reduction ExpStep uses, so the
    gate compares the two forms against each other instead of a form against itself.
    """
    t, d = TAU(s), DTAU(s)
    g = GT(t)
    return y + (v + g * (t * v - y) / (1.0 - t)) * (d * h) + (2.0 * g * gam * d).sqrt() * I0


def _mk(cfg, vf, dev):
    return EulerMaruyamaFlowMap(ExpStep(vf, cfg["init"], cfg["n_levy"], cfg["fm_width"],
                                        gamma=cfg["temperature"])).to(dev)


def reference(vf, x0, path, gam):
    """The teacher's own SDE on the finest dyadic level, as the whole trajectory with x0 first."""
    x, traj = x0, [x0]
    for k in range(2 ** path.levels):
        a, b = path.span(0, k)
        s = torch.full((x.shape[0], 1, 1), a, dtype=x.dtype, device=x.device)
        x = em_step(vf.fresh(x, s), x, s, torch.full_like(s, b - a), path.levy(0, k)[0], gam)
        traj.append(x)
    return torch.stack(traj)


def rollout(fm, x0, path, level):
    """Each step clears the memo, so the student gets v at ITS OWN left endpoint, not the last."""
    x, traj = x0, [x0]
    col = lambda a: torch.full((x0.shape[0], 1, 1), a, dtype=x0.dtype, device=x0.device)
    for idx in range(2 ** (path.levels - level)):
        a, b = path.span(level, idx)
        fm.step_model.vf.memo = {}
        x = fm(x, col(a), col(b), path.levy(level, idx))
        traj.append(x)
    return torch.stack(traj)


def train(cfg):
    """One chain per particle, advanced by the half-step the semigroup term already took."""
    mdl = _proteina(cfg)
    vf = VField(mdl)
    torch.manual_seed(cfg["seed"])
    fm, ema = _mk(cfg, vf, mdl.device), _mk(cfg, vf, mdl.device)
    ema.load_state_dict(fm.state_dict())
    opt = torch.optim.Adam(fm.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg["iters"], 1))
    n, dt, nl, gam = cfg["n"], cfg["dt"], cfg["n_levy"], cfg["temperature"]
    hist, t0 = [], time.time()
    with _binning(cfg):
        y = mdl.prior(n)
        u = (1.0 - cfg["h_max"]) * torch.rand(n, 1, 1, dtype=y.dtype, device=y.device)
        v = None
        lv = lambda hh: sample_levy(hh.expand_as(y), nl)
        for it in range(cfg["iters"] + 1):
            vf.memo = {}
            if v is not None:
                vf.seed(u, v)
            h = dt + (cfg["h_max"] - dt) * torch.rand_like(u)
            m, t = u + h / 2, u + h
            L, R = lv(h / 2), lv(h / 2)
            with torch.no_grad():
                ym = ema(y, u, m, L)                 # memo hit on u, no forward
            v = vf(ym, m)                            # THE one forward of this step
            ld = distill_loss(fm, ema, y, u, m, t, L, R, chen_combine(L, R))

            hm = dt / 2 + (dt / 2) * torch.rand_like(u)
            Im = lv(hm)
            lm = _sq(fm(y, u, u + hm, Im), em_step(vf(y, u), y, u, hm, Im[0], gam), hm)

            loss = cfg["eta"] * lm + (1 - cfg["eta"]) * ld
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fm.parameters(), cfg["clip"])
            opt.step()
            sched.step()
            with torch.no_grad():
                for q, r in zip(ema.parameters(), fm.parameters()):
                    q.mul_(cfg["ema"]).add_(r, alpha=1 - cfg["ema"])
            y, u = ym, m                             # the chain IS the semigroup half-step
            done = (u + cfg["h_max"] > 1.0 - cfg["t_eps"]).flatten()
            if bool(done.any()):
                y, u, v = y.clone(), u.clone(), v.clone()
                y[done], u[done] = mdl.prior(int(done.sum())), 0.0
                v[done] = vf.fresh(y[done], u[done])
            if it % cfg["log_every"] == 0:
                # saved every log interval, not just at the end: the loop is hours long and a
                # wall-clock kill would otherwise discard all of it. The head is ~50k params.
                torch.save(ema.state_dict(), cfg["out"].replace(".pt", f"_{cfg['init']}.pt"))
                f = mdl.n_forward / (it + 1)
                hist.append(dict(it=it, loss=float(loss), matching=float(lm),
                                 semigroup=float(ld), fwd_per_it=f,
                                 sec_per_it=(time.time() - t0) / (it + 1)))
                print(f"{it:7d}  loss {float(loss):11.6f}  match {float(lm):11.6f}  "
                      f"semi {float(ld):11.6f}  fwd/it {f:5.2f}", flush=True)
    torch.save(ema.state_dict(), cfg["out"].replace(".pt", f"_{cfg['init']}.pt"))
    json.dump(hist, open(out_path(cfg, f"fmtrain_{cfg['init']}"), "w"), indent=1)
    print(f"wrote {cfg['out']}", flush=True)


def _regime(cfg, fm, vf, levels, steps, label):
    torch.manual_seed(cfg["seed"] + 7)
    out = {}
    with torch.no_grad():
        x0 = vf.mdl.prior(cfg["eval_n"])
        path = BrownianPath(x0.shape, levels, n_levy=cfg["n_levy"], device=x0.device,
                            dtype=x0.dtype)
        ref = reference(vf, x0, path, cfg["temperature"])
        print(f"\n{label}: reference {2 ** levels} steps", flush=True)
        print(f"{'steps':>6} {'strong RMSE':>13} {'sd at u=1':>11}", flush=True)
        for k in steps:
            level = levels - int(k).bit_length() + 1
            r = rollout(fm, x0, path, level)
            err = float(((r - ref[::2 ** level]) ** 2).mean().sqrt())
            out[str(k)] = err
            print(f"{k:6d} {err:13.6f} {float(r[-1].std()):11.4f}", flush=True)
    return out


def evaluate(cfg):
    """Strong (pathwise) error against the teacher's own fine solve, on one shared path."""
    mdl = _proteina(cfg)
    vf = VField(mdl)
    fm = _mk(cfg, vf, mdl.device)
    fm.load_state_dict(torch.load(cfg["out"].replace(".pt", f"_{cfg['init']}.pt"),
                                  map_location=mdl.device))
    fm.eval().requires_grad_(False)
    with _binning(cfg):
        r = _regime(cfg, fm, vf, cfg["levels_coarse"], cfg["steps_coarse"], "coarse regime")
    json.dump(r, open(out_path(cfg, f"fmeval_{cfg['init']}"), "w"), indent=1)


def baseline(cfg):
    """The UNTRAINED student, which at zero-init is the analytic integrator `init` names.

    This is the curve training has to beat, from the identical code path with no weights loaded --
    a real scheme rather than a strawman.
    """
    mdl = _proteina(cfg)
    vf = VField(mdl)
    fm = _mk(cfg, vf, mdl.device).eval().requires_grad_(False)
    with _binning(cfg):
        r = _regime(cfg, fm, vf, cfg["levels_coarse"], cfg["steps_coarse"],
                    f"untrained {cfg['init']}")
    json.dump(r, open(out_path(cfg, f"fmeval_untrained_{cfg['init']}"), "w"), indent=1)


def gate(cfg):
    """The two checks that cost nothing and would invalidate the design if they failed."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gam = cfg["temperature"]
    col = lambda a, n=1: torch.full((n, 1, 1), float(a), dtype=torch.float32, device=dev)

    print("GATE 2a: stiffness and the analytic coefficients (no forwards)", flush=True)
    B0 = float(abs_u(col(0.0), gam)[1])
    print(f"  B(0) = {B0:.4f}   plain EM needs K > {B0 / 2:.1f} steps", flush=True)
    print(f"  {'K':>5} {'E_em(step 0)':>14} {'E_exp(step 0)':>14} {'S0_exp':>10}", flush=True)
    for K in (1, 16, 64, 128, 256):
        h = col(1.0 / K)
        e_em = float(coeffs_em(col(0.0), h, gam)[0])
        ce = coeffs_exp(col(0.0), h, gam)
        print(f"  {K:5d} {e_em:14.4f} {float(ce[0]):14.6f} {float(ce[2]):10.4f}", flush=True)
    assert abs(B0 - 465.1687) < 1e-2, f"B(0) = {B0}, so TAU/DTAU/GT disagree with the schedule"

    print("\nGATE 1: at init=em the untrained student IS em_step, so match must be ~0",
          flush=True)
    mdl = _proteina(cfg)
    vf = VField(mdl)
    torch.manual_seed(cfg["seed"])
    with _binning(cfg), torch.no_grad():
        y = mdl.prior(cfg["n"])
        u = (1.0 - cfg["h_max"]) * torch.rand(cfg["n"], 1, 1, dtype=y.dtype, device=y.device)
        h = cfg["dt"] / 2 + (cfg["dt"] / 2) * torch.rand_like(u)
        I = sample_levy(h.expand_as(y), cfg["n_levy"])
        fm = _mk(dict(cfg, init="em"), vf, mdl.device)
        lm = float(_sq(fm(y, u, u + h, I), em_step(vf(y, u), y, u, h, I[0], gam), h))
    print(f"  matching loss at init=em: {lm:.3e}   (must be < 1e-6)", flush=True)
    print(f"  teacher forwards for the whole gate: {mdl.n_forward}", flush=True)
    assert lm < 1e-6, f"matching loss {lm:.3e}: the u-conversion or the a/b reduction is wrong"
    print("\nboth gates pass", flush=True)


STAGES = {"gate": gate, "baseline": baseline, "train": train, "eval": evaluate}


if __name__ == "__main__":
    main(STAGES, sys.argv[1:])
