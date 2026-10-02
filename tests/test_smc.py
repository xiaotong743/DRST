"""smc.py: ESS, resampling identities/unbiasedness, and the Picard collapse property."""

import torch
import pytest

from steering.meanfield import picard
from steering.smc import ess, residual_resample, systematic_resample

torch.set_default_dtype(torch.float64)


def test_ess_uniform_and_point_mass():
    n = 32
    assert abs(ess(torch.zeros(n)) - n) < 1e-10
    spike = torch.full((n,), -80.0)
    spike[3] = 0.0
    assert abs(ess(spike) - 1.0) < 1e-6


def test_ess_invariant_to_shift():
    a = torch.randn(16, generator=torch.Generator().manual_seed(0))
    assert abs(ess(a) - ess(a + 7.5)) < 1e-9


def test_residual_resample_uniform_is_identity_with_no_draws():
    """The lambda=0 bypass condition: uniform weights must give arange(N) and consume
    no randomness, so an unsteered run stays bit-identical with resampling enabled."""
    n = 24
    w = torch.full((n,), 1.0 / n)
    g = torch.Generator().manual_seed(123)
    before = g.get_state()
    idx = residual_resample(w, g)
    assert torch.equal(idx, torch.arange(n))
    assert torch.equal(g.get_state(), before)


def test_resample_preserves_population_size():
    g = torch.Generator().manual_seed(1)
    for n in (7, 16, 64):
        w = torch.rand(n, generator=g).add(1e-3)
        w = w / w.sum()
        assert residual_resample(w, g).shape == (n,)
        assert systematic_resample(w, g).shape == (n,)


@pytest.mark.parametrize("fn", [residual_resample, systematic_resample])
def test_resample_unbiased(fn):
    """E[count_i / N] == w_i."""
    n = 8
    w = torch.tensor([0.30, 0.25, 0.20, 0.10, 0.07, 0.05, 0.02, 0.01])
    g = torch.Generator().manual_seed(7)
    trials = 4000
    counts = torch.zeros(n)
    for _ in range(trials):
        counts += torch.bincount(fn(w, g), minlength=n).double()
    emp = counts / (trials * n)
    assert torch.allclose(emp, w, atol=4e-3)


def test_resample_deterministic_under_fixed_generator():
    w = torch.rand(20, generator=torch.Generator().manual_seed(5)).add(0.01)
    w = w / w.sum()
    a = residual_resample(w, torch.Generator().manual_seed(99))
    b = residual_resample(w, torch.Generator().manual_seed(99))
    assert torch.equal(a, b)


# --- Picard ---------------------------------------------------------------

def _pointwise_setup(n=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    psi_t = torch.randn(n, generator=g)
    return dict(
        a0=torch.zeros(n),
        psi_t=psi_t,
        psi_of_weights=lambda w: psi_t,          # no measure dependence
        time_term=torch.randn(n, generator=g) * 0.1,
        bgrad=torch.randn(n, generator=g) * 0.1,
        t=0.4, dt=0.01, lam=1.0,
    )


def test_picard_pointwise_collapses_in_one_step_at_damp_one():
    """A pointwise reward makes the map constant in k, so damp=1 lands on the fixed point
    immediately. If this fails, the psi-dot assembly is wrong."""
    s = _pointwise_setup()
    a, w, k, resid, _, psidot = picard(**s, k_fp=50, damp=1.0, tol=1e-12)
    assert k == 2                      # one step to move, one to confirm resid==0
    assert resid < 1e-12
    # closed form: a = a0 + dt*(lam*psi_t + time_term + bgrad)
    want = s["a0"] + s["dt"] * (s["lam"] * s["psi_t"] + s["time_term"] + s["bgrad"])
    assert torch.allclose(a, want, atol=1e-12)
    # the returned psidot (paper line 11) must be the same quantity the increment used
    want_psidot = s["lam"] * s["psi_t"] + s["time_term"]
    assert torch.allclose(psidot, want_psidot, atol=1e-12)


@pytest.mark.parametrize("damp", [0.3, 0.5, 1.0])
def test_picard_fixed_point_is_damping_invariant(damp):
    """Damping changes the number of iterations, never the fixed point."""
    s = _pointwise_setup()
    a, _, _, _, _, _ = picard(**s, k_fp=500, damp=damp, tol=1e-14)
    want = s["a0"] + s["dt"] * (s["lam"] * s["psi_t"] + s["time_term"] + s["bgrad"])
    assert torch.allclose(a, want, atol=1e-10)


def test_picard_measure_dependent_iterates_and_contracts():
    """With a genuine measure dependence the loop must actually iterate and the residual
    must decay monotonically."""
    n = 16
    g = torch.Generator().manual_seed(3)
    z = torch.randn(n, generator=g)
    psi_t = z.clone()

    def psi_of_weights(w):
        return z - 3.0 * (w.to(z.dtype) * z).sum()      # mean-field coupling

    a, w, k, resid, hist, _ = picard(
        a0=torch.zeros(n), psi_t=psi_t, psi_of_weights=psi_of_weights,
        time_term=torch.zeros(n), bgrad=torch.zeros(n),
        t=0.5, dt=0.02, lam=5.0, k_fp=200, damp=0.5, tol=1e-12)
    assert k > 2
    assert resid < 1e-12
    tail = hist[1:8]
    assert all(b <= a_ for a_, b in zip(tail, tail[1:]))


def test_picard_increment_clip_binds():
    """The clip must SATURATE, and the unclipped run must exceed it. Asserting only
    |a| <= clip passes on a stub that zeroes the increment entirely -- measured, replacing
    `inc` with zeros leaves every test in this file green."""
    s = _pointwise_setup()
    s["lam"] = 1e4
    a_free, *_ = picard(**s, k_fp=5, damp=1.0)
    a, *_ = picard(**s, k_fp=5, damp=1.0, increment_clip=0.5)
    assert float(a_free.abs().max()) > 0.5, "unclipped run never reaches the clip"
    assert float(a.abs().max()) <= 0.5 + 1e-12
    # saturated, not zeroed: the clip is active on at least one particle
    assert float(a.abs().max()) > 0.5 - 1e-9, f"increment vanished instead of clipping: {a}"
    assert float((a - s["a0"]).abs().max()) > 1e-9, "log-weights did not move at all"


