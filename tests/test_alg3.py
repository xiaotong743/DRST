"""alg3.py: Algorithm 3, the implicit Psidot solver, against the second variation it claims."""

import math

import pytest
import torch

from steering.alg3 import implicit_psidot, psidot_terms
from steering.sde import steer, Config, Pointwise, MMD
from steering.toy import GaussianMixtureFlow, grid_mu_star, sample_grid, w1_to_grid

torch.set_default_dtype(torch.float64)

SD = 0.5                       # the paper's mixture: unit variance OF THE MIXTURE
H = 5.0 * math.sqrt(2.0)


def _setup(n=400, eps_mode="const"):
    m = GaussianMixtureFlow(eps_mode=eps_mode, eps_clamp=20.0, means=(-1.0, 1.0),
                            stds=(SD, SD), weights=(1.0, 3.0))
    tgt = GaussianMixtureFlow(means=(-1.0, 1.0), stds=(SD, SD), weights=(3.0, 1.0))
    y = tgt.sample_p1(n, torch.Generator().manual_seed(1)).reshape(-1, 1)
    return m, MMD(y, H), y


def test_phi_tilde_is_the_second_variation():
    """Phitilde must equal delta^2 R / delta mu^2 mean-centred, i.e. -2(k - E_pi k) for
    R = -MMD^2. Pinned three ways, because the Gateaux finite difference alone is an exact
    identity here (MMD^2 is quadratic, so the O(e) remainder it claims to bound is never present)
    """
    _, pot, _ = _setup()
    g = torch.Generator().manual_seed(3)
    q = torch.randn(7, 1, generator=g)
    z = torch.randn(50, 1, generator=g)
    w = torch.full((50,), 1.0 / 50)

    # (i) against an independent construction from the kernel: sign and factor of 2
    from steering.mmd import rbf_gram
    k = rbf_gram(q, z, pot.h)
    want = -2.0 * (k - (k @ w)[:, None])
    assert torch.allclose(pot.phi_tilde(q, z, w), want, atol=1e-12)
    # Phitilde is mean-centred, so its w-weighted row mean is ZERO by construction; the sign has
    # to be pinned on the uncentred Phi, where Phi = -2k < 0 because the kernel is positive.
    assert float((pot.phi_tilde(q, z, w) * w).sum(1).abs().max()) < 1e-12
    assert float(pot.phi_sum(q, z, w).max()) < 0

    # (ii) Gateaux: d/de psi(q, (1-e)mu + e delta_y) = Phitilde(q, y), for several directions
    for j in (0, 17, 49):
        y = z[j:j + 1]
        # Phitilde(q, z_j, mu) is column j of the matrix; building it from
        # phi_tilde(q, y, ones(1)) instead gives identically zero, because centring over a
        # single-point measure subtracts that very point.
        exact = pot.phi_tilde(q, z, w)[:, j]
        e = 1e-6
        we = (1 - e) * w
        fd = (pot.psi(q, torch.cat([z, y]), torch.cat([we, torch.tensor([e])]))
              - pot.psi(q, z, w)) / e
        assert torch.allclose(fd, exact, atol=1e-5, rtol=1e-4), j

    # (iii) phi_sum must be the w-contraction of the UNCENTRED Phi, with no centring term
    assert torch.allclose(pot.phi_sum(q, z, w), -2.0 * (k @ w), atol=1e-12)


def test_pointwise_second_variation_vanishes_and_alg3_reduces_to_dt_psi():
    """R(mu) = int r dmu is linear, so Phi == 0, A == I, and psidot == d_t Psi exactly."""
    r = Pointwise(lambda th: th.squeeze(-1) ** 2)
    q = torch.randn(9, 1)
    w = torch.full((9,), 1.0 / 9)
    assert float(r.phi_tilde(q, q, w).abs().max()) == 0.0
    assert float(r.phi_sum(q, q, w).abs().max()) == 0.0
    dpsi = torch.randn(9)
    psidot, resid = implicit_psidot(torch.zeros(9, 9), w, dpsi, torch.zeros(9), torch.randn(9))
    assert torch.allclose(psidot, dpsi)
    assert resid < 1e-12


def test_pointwise_implicit_matches_picard_end_to_end():
    """With Phi == 0 both solvers compute the same object, so the samplers must agree."""
    m, _, _ = _setup()
    r = Pointwise(lambda th: th.squeeze(-1))
    kw = dict(nsteps=60, lam=0.5, resample_every=0, seed=5)
    a = steer(m, r, Config(psidot_solver="picard", k_fp=1, damp=1.0, **kw), 500)
    b = steer(m, r, Config(psidot_solver="implicit", **kw), 500)
    assert torch.equal(a["x"], b["x"])                       # same drift, same noise stream
    d = float((a["a"] - b["a"]).abs().max())
    assert d < 1e-9, d
    # and the trap itself, pinned: damped-and-truncated Picard under-applies the tilt
    half = steer(m, r, Config(psidot_solver="picard", k_fp=1, damp=0.5, **kw), 500)
    assert abs(float((half["a"] / a["a"]).median()) - 0.5) < 1e-6


def test_implicit_psidot_matches_a_converged_picard_at_the_same_state():
    """The sharp check: Algorithm 3 and a CONVERGED, damped Algorithm 4 must compute the same
    Psidot at the same cloud. Measured in scratch/alg34_nscale.py, the residual disagreement is
    finite-N and falls with N (7.2e-3 -> 1.0e-3 relative over 16x N at lam=2), so a few percent
    """
    from steering.alg3 import implicit_psidot, psidot_terms
    from steering.ips import picard, normalise
    from steering.sde import _dot

    m, pot, _ = _setup()
    n, lam, t, dt = 800, 6.0, 0.6, 1e-3
    g = torch.Generator().manual_seed(17)
    x = m.prior(n, g)
    a = 0.3 * torch.randn(n, generator=g)
    w = normalise(a)

    xg = x.detach().requires_grad_(True)
    tg = torch.full((n,), t).requires_grad_(True)
    theta, v = m.endpoint(xg, tg)
    psi_q = pot.psi(theta, theta.detach(), w)
    (lam * tg * psi_q).sum().backward()
    gx, gt = xg.grad.clone(), tg.grad.clone()
    psi_d, v_d = psi_q.detach(), v.detach()
    bgrad = _dot(v_d, gx).double()

    phi, dpsi_dt, pgb = psidot_terms(m, pot, x, t, dt, w, lam, v_d, gx)
    pd_a3, resid = implicit_psidot(phi.double(), w.double(), dpsi_dt.double(), pgb.double(), bgrad)
    assert resid < 1e-10

    with torch.no_grad():
        support, _ = m.endpoint(x + v_d * dt, torch.full((n,), t + dt))
    _, _, _, _, _, pd_a4 = picard(
        a0=a.double(), psi_t=psi_d.double(),
        psi_of_weights=lambda ww: pot.weights_fn(theta.detach(), support)(ww).double(),
        time_term=(gt - lam * psi_d).double(), bgrad=bgrad,
        t=t, dt=dt, lam=lam, k_fp=4000, damp=min(0.5, 1.5 / (1.0 + lam)), tol=1e-14)

    rel = float((pd_a3 - pd_a4).abs().max() / pd_a3.abs().max())
    assert rel < 0.05, f"Algorithm 3 vs converged Algorithm 4 disagree by {rel:.3e}"
    # and neither is trivially zero, or the agreement would be vacuous
    assert float(pd_a3.abs().max()) > 1e-3


def test_implicit_solver_needs_no_contraction_where_picard_diverges():
    """Algorithm 3 is a direct solve, so it carries no rho < 1 requirement."""
    m, pot, y = _setup()
    n, lam = 400, 30.0                       # rho ~ 0.077 * lam > 1
    grid, q = grid_mu_star(m, y, lam, H)
    unif = torch.full((n,), 1.0 / n)
    kw = dict(nsteps=60, lam=lam, resample_every=0, seed=5)
    p = steer(m, pot, Config(psidot_solver="picard", k_fp=3, damp=1.0, **kw), n)
    i = steer(m, pot, Config(psidot_solver="implicit", **kw), n)
    w1_p = w1_to_grid(p["x"], p["w"], grid, q)
    w1_i = w1_to_grid(i["x"], i["w"], grid, q)
    unw = w1_to_grid(i["x"], unif, grid, q)
    assert w1_i < w1_p, f"implicit W1 {w1_i:.4f} not better than diverging picard {w1_p:.4f}"
    # a zeroed or sign-flipped psidot leaves the weights near-uniform: require the reweighting to
    # earn its place against simply ignoring it.
    assert w1_i < 0.8 * unw, f"implicit W1 {w1_i:.4f} vs its own unweighted {unw:.4f}"


@pytest.mark.slow
def test_implicit_solver_targets_mu_star():
    """The end-to-end claim: Algorithm 1 driven by Algorithm 3 lands on mu*(lam)."""
    m, pot, y = _setup()
    n = 3000
    grid, q = grid_mu_star(m, y, 10.0, H)
    unif = torch.full((n,), 1.0 / n)
    floor = float(torch.tensor(
        [w1_to_grid(sample_grid(grid, q, n, torch.Generator().manual_seed(s)), unif, grid, q)
         for s in range(4)]).mean())
    ws = []
    for lam in (8.0, 10.0, 12.0):
        o = steer(m, pot, Config(psidot_solver="implicit", nsteps=100, lam=lam,
                                 resample_every=10, seed=4), n)
        ws.append(w1_to_grid(o["x"], o["w"], grid, q))
    assert ws[1] == min(ws), f"argmin not at lam*: {ws}"
    assert ws[1] < 2.5 * floor, f"W1 at lam* {ws[1]:.4f} vs floor {floor:.4f}"
