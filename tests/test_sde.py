"""sde.py: lambda=0 identity, drift bookkeeping, and the gauge freedom in psi."""

import torch

from steering.sde import steer, Config, Pointwise, _dot
from steering.toy import GaussianMixtureFlow

torch.set_default_dtype(torch.float64)


def _model(**kw):
    return GaussianMixtureFlow(eps_mode="memoryless", eps_clamp=20.0, **kw)


def _euler_reference(model, n, cfg, potential=None):
    """Euler-Maruyama written independently of steer(), with the same RNG consumption order."""
    g = torch.Generator().manual_seed(cfg.seed)
    ts = torch.linspace(0, 1, cfg.nsteps + 1)
    x = model.prior(n, g)
    for k in range(cfg.nsteps):
        t, dt_t = float(ts[k]), (ts[k + 1] - ts[k])
        e = model.eps(t)
        tt = torch.full((n,), t)
        steer_term = torch.zeros_like(x)
        if potential is not None and cfg.lam != 0.0:
            xg = x.detach().requires_grad_(True)
            tg = torch.full((n,), t, requires_grad=True)
            th_g, _ = model.endpoint(xg, tg)
            (cfg.lam * tg * potential.psi(th_g, th_g.detach(),
                                          torch.full((n,), 1.0 / n))).sum().backward()
            steer_term = (cfg.steer_coeff if cfg.steer_coeff is not None else e) * xg.grad
        theta, v = model.endpoint(x, tt)
        s = model.score(x, v, t)
        noise = torch.randn(x.shape, generator=g) if e > 0 else torch.zeros_like(x)
        # same association as steer() / Proteina's step_euler: delta first, then one add
        std = torch.sqrt(2 * torch.as_tensor(e, dtype=dt_t.dtype) * dt_t)
        x = x + ((v + e * s + steer_term) * dt_t + std * noise)
    return x


def test_lambda_zero_is_bitwise_euler_maruyama():
    m = _model()
    cfg = Config(nsteps=60, lam=0.0, resample_every=0, seed=11)
    out = steer(m, Pointwise(lambda th: th.squeeze(-1)), cfg, 64)
    assert torch.equal(out["x"], _euler_reference(m, 64, cfg))
    assert float(out["a"].abs().max()) == 0.0


def test_lambda_zero_bitwise_with_resampling_enabled():
    """Uniform weights make residual resampling the identity with no draws, so enabling it
    must not perturb the trajectory."""
    m = _model()
    base = Config(nsteps=60, lam=0.0, resample_every=0, seed=11)
    withrs = Config(nsteps=60, lam=0.0, resample_every=10, seed=11)
    r = Pointwise(lambda th: th.squeeze(-1))
    assert torch.equal(steer(m, r, base, 64)["x"], steer(m, r, withrs, 64)["x"])


def test_steered_positions_match_an_independent_integrator():
    """The positional update (12a) at lam != 0, against an integrator that rebuilds grad_x Psi
    from the model itself.
    """
    m = _model()
    r = Pointwise(lambda th: -(th.squeeze(-1) ** 2))
    cfg = Config(nsteps=40, lam=5.0, resample_every=0, seed=3, check_drift=True)
    out = steer(m, r, cfg, 32)
    assert torch.equal(out["x"], _euler_reference(m, 32, cfg, potential=r))
    # and the steering must actually be doing something at this lambda
    plain = Config(nsteps=40, lam=0.0, resample_every=0, seed=3)
    assert float((out["x"] - steer(m, r, plain, 32)["x"]).abs().max()) > 1e-3


def test_constant_psi_is_a_gauge_freedom():
    """An additive constant in psi shifts A uniformly and leaves x and w untouched."""
    m = _model()
    cfg0 = Config(nsteps=40, lam=0.0, resample_every=0, seed=5)
    cfgc = Config(nsteps=40, lam=1.0, resample_every=0, seed=5)
    ref = steer(m, Pointwise(lambda th: th.squeeze(-1)), cfg0, 32)
    con = steer(m, Pointwise(lambda th: torch.full_like(th.squeeze(-1), 2.5)), cfgc, 32)
    assert torch.equal(ref["x"], con["x"])                 # zero gradient -> same trajectory
    assert torch.allclose(con["w"], torch.full((32,), 1 / 32), atol=1e-12)
    assert float(con["a"].std()) < 1e-12                   # shifted uniformly
    assert float(con["a"].abs().max()) > 1e-6              # but actually shifted


def test_eps_zero_kills_noise_and_steering_together():
    m = GaussianMixtureFlow(eps_mode="zero")
    cfg = Config(nsteps=30, lam=3.0, resample_every=0, seed=1)
    a = steer(m, Pointwise(lambda th: th.squeeze(-1)), cfg, 16)
    b = steer(m, Pointwise(lambda th: -th.squeeze(-1)), cfg, 16)
    assert torch.equal(a["x"], b["x"])   # eps=0 -> no steering influence on positions
    assert not torch.allclose(a["a"], b["a"])   # weights still respond


def test_reweight_flag_changes_weights_not_positions():
    m = _model()
    r = Pointwise(lambda th: -(th.squeeze(-1) ** 2))
    on = steer(m, r, Config(nsteps=40, lam=4.0, resample_every=0, seed=9, reweight=True), 32)
    off = steer(m, r, Config(nsteps=40, lam=4.0, resample_every=0, seed=9, reweight=False), 32)
    assert torch.equal(on["x"], off["x"])
    assert float(off["a"].abs().max()) == 0.0
    assert float(on["a"].std()) > 1e-6


def test_weights_fn_matches_psi_for_both_potentials():
    """weights_fn caches the Gram so the Picard loop is matvecs. It must agree with psi()."""
    from steering.sde import MMD
    g = torch.Generator().manual_seed(2)
    theta = torch.randn(24, 1, generator=g)
    w = torch.rand(24, generator=g).add(0.05)
    w = (w / w.sum()).double()

    r = Pointwise(lambda th: th.squeeze(-1) ** 2)
    assert torch.allclose(r.weights_fn(theta, theta)(w), r.psi(theta, theta, w))

    target = torch.randn(40, 1, generator=g) + 1.0
    m = MMD(target, h=0.5)
    assert torch.allclose(m.weights_fn(theta, theta)(w), m.psi(theta, theta, w), atol=1e-12)


def test_literal_fd_converges_to_eq20():
    """Appendix B gives two Psi-dot implementations: the plain finite difference (used in the
    paper's low-dimensional experiments) and eq (20) (used in the protein experiments). They
    agree only to O(dt), so the gap must shrink with the step size."""
    import math
    from steering.toy import GaussianMixtureFlow
    from steering.sde import MMD
    m = GaussianMixtureFlow(eps_mode="const", means=(-1.0, 1.0), stds=(1.0, 1.0),
                            weights=(1.0, 3.0))
    tgt = GaussianMixtureFlow(means=(-1.0, 1.0), stds=(1.0, 1.0), weights=(3.0, 1.0))
    y = tgt.sample_p1(400, torch.Generator().manual_seed(1)).reshape(-1, 1)
    pot = MMD(y, 5.0 * math.sqrt(2.0))
    gaps = []
    for nst in (50, 200, 800):
        o = {lag: steer(m, pot, Config(nsteps=nst, lam=10.0, k_fp=50, damp=0.5, seed=0,
                                       resample_every=0, endpoint_lag=lag), 400)
             for lag in ("advance", "literal")}
        gaps.append(float((o["advance"]["x"] - o["literal"]["x"]).abs().max()))
    assert gaps[0] > gaps[1] > gaps[2], gaps
    assert gaps[-1] < 0.25 * gaps[0], gaps
