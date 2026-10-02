"""Branch A's two expressions of one sampler must agree, bitwise.

`SteererA` in experiments/steer_a.py subclasses the published `Steerer` and overrides three hooks.
`sample_a` below writes the same dynamics out flat, end to end, with no Config indirection. Two
implementations only help if they are pinned to each other, so that is what this file does.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from steer_a import ConfigA, SteererA                                       # noqa: E402
from steering.rewards import ContactRepulsion, SoftContactFeatures  # noqa: E402

torch.set_default_dtype(torch.float32)
L, N = 24, 12


class _Walk3D:
    """[L, 3] stand-in with the endpoint contract, plus Proteina's zero-COM projection and an
    `eps` schedule, so both samplers exercise the same paths without a GPU or a checkpoint."""

    def __init__(self, length=L, k=0.4):
        self.length, self.k = length, k
        self.ts = torch.linspace(0, 1, 9)

    def prior(self, n, generator=None):
        x = torch.randn(n, self.length, 3, generator=generator) * 0.10
        x[:, :, 0] += torch.arange(self.length) * 0.38
        return self.project(x)

    def endpoint(self, x, t):
        v = self.k * x
        return x + (1 - t.reshape(-1, 1, 1)) * v, v

    def score(self, x, v, t):
        return -x

    def eps(self, t):
        return 0.3 * (1.0 - t)

    def project(self, x):
        return x - x.mean(dim=1, keepdim=True)


def _pot():
    return ContactRepulsion(SoftContactFeatures(torch.ones(L, dtype=torch.bool)), h=0.3)


# ------------------------------------------------------------------ the reference oracle

def sample_a(model, potential, *, eta, tau, n, ts, k_fp=3, damp=1.0, resample_every=25,
             vf_tail_t=0.99, grad_chunk=4, seed=7, global_rng=True, endpoint_lag="literal"):
    """Branch A's dynamics written out flat, as the oracle SteererA is pinned against.

    Algorithm 1 (positions, then weights) with Algorithm 4 -- steering.meanfield.picard -- for the
    log-weight fixed point. Algorithm 3 is not used here.

        dX = [v + g_t s + eps_t grad_x Psi] dt + sqrt(2 g_t tau) dB      eps_t = tau * g_t
        dA = [b . grad_x Psi + dPsi/dt] dt                               b = v alone

    with Psi_t = eta * t * psi(xhat1(x,t), pihat_t). The temperature enters ONLY as tau on the
    noise and on grad_x Psi; the score keeps the raw g_t. That is Proteina's eq (3), and A adds
    no Feynman-Kac temperature term.

    Everything Proteina-specific is explicit: the ODE tail at vf_tail_t, the noise built in
    their operand order, the zero-COM projection, and the microbatched backward. It lives here
    rather than in experiments/steer_a.py because nothing ships it -- its only job is to be a second,
    independently written expression of the same equations, so that a mistake in the hook
    plumbing of `Steerer`/`SteererA` shows up as a bitwise disagreement rather than as a plausible
    number. It earned that keep on 2026-09-19, when it caught nothing but confirmed a rebuild of
    `Steerer.run()` was faithful.
    """
    from steering.meanfield import picard
    from steering.smc import ess, normalise, residual_resample
    from steering.steerer import _chunked, _dot, _reward_endpoint     # mechanical helpers

    g = None if global_rng else torch.Generator(device="cpu").manual_seed(seed)
    x = model.prior(n, g)
    a = torch.zeros(n, dtype=torch.float64, device=x.device)
    diag = []

    for k in range(len(ts) - 1):
        t_val, dt_t = float(ts[k]), (ts[k + 1] - ts[k])
        dt = float(dt_t)
        g_t = model.eps(t_val)
        if t_val > vf_tail_t:
            g_t = 0.0                      # Proteina flips to the ODE here: no noise, no score
        eps_t = tau * g_t
        w = normalise(a)
        t_vec = torch.full((n,), t_val, dtype=x.dtype, device=x.device)

        # (1) the measure: detached endpoints and the velocity, no graph
        with torch.no_grad():
            theta_d, v_d = _chunked(model.endpoint, x, t_vec, grad_chunk)
            theta_d, v_d = theta_d.detach(), v_d.detach()
            psi_d = potential.psi(theta_d, theta_d, w).detach()

        # (2) grad_x Psi and dPsi/dt, microbatched over queries; the measure stays full
        gx = torch.zeros_like(x)
        gt = torch.zeros(n, dtype=x.dtype, device=x.device)
        for lo in range(0, n, grad_chunk or n):
            hi = min(lo + (grad_chunk or n), n)
            xg = x[lo:hi].detach().requires_grad_(True)
            tg = torch.full((hi - lo,), t_val, dtype=x.dtype,
                            device=x.device).requires_grad_(True)
            th_c, _ = model.endpoint(xg, tg)
            (eta * tg * potential.psi(th_c, theta_d, w)).sum().backward()
            gx[lo:hi], gt[lo:hi] = xg.grad, tg.grad

        # (3) positions first, so X_{t+dt} is available to the weight solve
        base = v_d + g_t * model.score(x, v_d, t_val)      # RAW g_t on the score
        drift = base + eps_t * gx
        noise = (torch.randn(x.shape, generator=g, dtype=x.dtype, device=x.device)
                 if eps_t > 0 else torch.zeros_like(x))
        g_tt = torch.as_tensor(g_t, dtype=dt_t.dtype)
        std = (torch.sqrt(2 * g_tt * tau * dt_t) if eps_t > 0
               else torch.zeros((), dtype=dt_t.dtype))
        x_next = model.project(x + (drift * dt_t + std * noise))

        # (4) log-weights: Algorithm 4. "literal" advances BOTH endpoints to t+dt, so the finite
        #     difference already carries the model-time term and time_term must be zero.
        t_nx = torch.full((n,), min(t_val + dt, 1.0 - 1e-9), dtype=x.dtype, device=x.device)
        query = _reward_endpoint(model, x, t_nx, grad_chunk)
        support = _reward_endpoint(model, x_next, t_nx, grad_chunk)
        if endpoint_lag != "literal":
            raise NotImplementedError(f"sample_a only implements the literal lag, not "
                                      f"{endpoint_lag!r}; use steering.steerer.steer for the others")
        fn = potential.weights_fn(query, support)
        a, w, k_used, resid, _, _ = picard(
            a0=a, psi_t=psi_d.double(), psi_of_weights=lambda ww: fn(ww).double(),
            time_term=torch.zeros(n, dtype=torch.float64, device=x.device),
            bgrad=_dot(v_d, gx).double(), t=t_val, dt=dt, lam=eta, k_fp=k_fp, damp=damp)

        x = x_next
        ess_pre = ess(a)
        if resample_every and eps_t > 0 and (k + 1) % resample_every == 0:
            idx = residual_resample(normalise(a), g)
            if not torch.equal(idx, torch.arange(n, device=idx.device)):
                x = x[idx]
            a = torch.zeros_like(a)          # residual resampling returns equal weights
        diag.append(dict(step=k, t=t_val, eps=eps_t, ess=ess_pre, picard_iters=k_used,
                         picard_resid=resid))
    return dict(x=x, a=a, w=normalise(a), diag=diag, ts=ts)

# ------------------------------------------------------------------ the pin

@pytest.mark.parametrize("eta,tau,chunk,resample", [(5.0, 0.45, 4, 25), (2.0, 1.0, None, 0),
                                                    (5.0, 0.45, 3, 4)])
def test_sample_a_matches_the_generic_sampler_bitwise(eta, tau, chunk, resample):
    m = _Walk3D()
    common = dict(k_fp=3, damp=1.0, resample_every=resample, vf_tail_t=0.99,
                  grad_chunk=chunk, seed=7, endpoint_lag="literal")

    torch.manual_seed(1234)
    a_out = sample_a(m, _pot(), eta=eta, tau=tau, n=N, ts=m.ts, global_rng=True, **common)

    torch.manual_seed(1234)
    g_out = SteererA(m, _pot(), ConfigA(ts=m.ts, lam=eta, temperature=tau, global_rng=True,
                                        **common), N).run()

    dx = float((a_out["x"] - g_out["x"]).abs().max())
    da = float((a_out["a"] - g_out["a"]).abs().max())
    assert dx == 0.0 and da == 0.0, (
        f"sample_a and SteererA diverged: max|dx|={dx:.3e}, max|da|={da:.3e}")
    # and the run must be non-trivial, or this compares two do-nothing samplers
    torch.manual_seed(1234)
    base = SteererA(m, _pot(), ConfigA(ts=m.ts, lam=0.0, temperature=tau, global_rng=True,
                                       **common), N).run()
    assert float((g_out["x"] - base["x"]).abs().max()) > 1e-8, "steering moved nothing"


def test_sample_a_refuses_the_lags_it_does_not_implement():
    m = _Walk3D()
    with pytest.raises(NotImplementedError, match="literal"):
        sample_a(m, _pot(), eta=1.0, tau=0.45, n=4, ts=m.ts, endpoint_lag="advance")


# ------------------------------------------------------------------ the variant table

def test_each_variant_reaches_the_config_with_the_settings_its_name_claims():
    """The table is the only definition of a variant, and A must not carry B's fields.

    This has bitten: `run_variants` used to take positional tuples, so a wrong slot was silent.
    A wrong key here is a TypeError.
    """
    import steer_a

    class _M:
        ts = torch.linspace(0, 1, 5)

    cfg = dict(temperature=0.45, k_fp=3, damp=1.0, endpoint_lag="literal", vf_tail_t=0.99,
               resample_every=25, grad_chunk=4, eta=3.0)
    a = {k: steer_a.mk_cfg(cfg)(_M(), **o) for k, o in steer_a._table(cfg).items()}

    assert (a["native"].lam, a["A"].lam) == (0.0, 3.0)
    assert all(c.temperature == 0.45 and c.grad_chunk == 4 for c in a.values())


# ------------------------------------------------------------------ chunking is a free knob

@pytest.mark.parametrize("nograd", [1, 3, 5, None])
def test_nograd_chunk_changes_no_number(nograd):
    """The three no_grad endpoint passes may be batched any way at all.

    They store no activations, so the chunk that fits the BACKWARD is far smaller than they
    need -- and until this was split out, all three ran at grad_chunk. At L=250, N=100,
    grad_chunk=4 that is 25 launches of 4 particles for each of 3 of the 4 forward passes per
    step, at a measured peak of 19.72 GB on an 80 GB card. Raising it is pure wall-clock.

    The model is per-particle independent, so splitting the batch and concatenating is bitwise
    identical however it is split -- ON CPU, which is what this asserts. If that ever stops
    being true the sampler has grown a cross-particle coupling it is not supposed to have.

    It is NOT bitwise on GPU, and this test cannot see that. **[measured]** on an A100, chunk 8
    against chunk 4 differs by 1.4e-5 nm at L=250 and 1.5e-6 at L=50, because a different batch
    size selects a different cuBLAS kernel. And the speedup is inverted: raising it buys 13% at
    L=50 (7.59 -> 6.58 s/step) but 0.7% at L=250 (107.87 -> 107.10), where the cost is the
    trunk's FLOPs rather than launch overhead. So this pins the ALGEBRA, not a licence to turn
    the knob on a run that has to match a bitwise gate.
    """
    m = _Walk3D()
    common = dict(k_fp=3, damp=1.0, resample_every=4, vf_tail_t=0.99, grad_chunk=4,
                  seed=7, endpoint_lag="literal", lam=5.0, temperature=0.45, global_rng=True)
    torch.manual_seed(99)
    ref = SteererA(m, _pot(), ConfigA(ts=m.ts, **common), N).run()
    torch.manual_seed(99)
    got = SteererA(m, _pot(), ConfigA(ts=m.ts, nograd_chunk=nograd, **common), N).run()
    dx = float((ref["x"] - got["x"]).abs().max())
    da = float((ref["a"] - got["a"]).abs().max())
    assert dx == 0.0 and da == 0.0, (
        f"nograd_chunk={nograd} changed the result: max|dx|={dx:.3e}, max|da|={da:.3e}")
