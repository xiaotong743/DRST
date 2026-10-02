"""Branch A's temperature split, and the microbatch invariance of grad_x Psi."""

import torch

from steer_a import ConfigA, SteererA
from steering.rewards import Pointwise
from toy import GaussianMixtureFlow

torch.set_default_dtype(torch.float64)


def steer(model, potential, cfg, n, device="cpu"):
    return SteererA(model, potential, cfg, n, device).run()


def _reference(model, n, cfg, potential=None, score_coeff="g"):
    """Euler-Maruyama written independently of steer(), same RNG consumption order.

    `score_coeff` selects which coefficient multiplies the score, so a test can assert that the
    WRONG convention fails to reproduce the run.
    """
    tau = cfg.temperature
    g = torch.Generator().manual_seed(cfg.seed)
    ts = torch.linspace(0, 1, cfg.nsteps + 1)
    x = model.prior(n, g)
    for k in range(cfg.nsteps):
        t, dt_t = float(ts[k]), (ts[k + 1] - ts[k])
        g_t = model.eps(t) if t <= cfg.vf_tail_t else 0.0
        e = tau * g_t
        tt = torch.full((n,), t)
        steer_term = torch.zeros_like(x)
        if potential is not None and cfg.lam != 0.0:
            xg = x.detach().requires_grad_(True)
            tg = torch.full((n,), t, requires_grad=True)
            th_g, _ = model.endpoint(xg, tg)
            (cfg.lam * tg * potential.psi(th_g, th_g.detach(),
                                          torch.full((n,), 1.0 / n))).sum().backward()
            steer_term = e * xg.grad
        _, v = model.endpoint(x, tt)
        s = model.score(x, v, t)
        noise = torch.randn(x.shape, generator=g) if e > 0 else torch.zeros_like(x)
        # same order as steer() and as Proteina's step_euler: 2 * g * tau * dt
        std = torch.sqrt(2 * torch.as_tensor(g_t, dtype=dt_t.dtype) * tau * dt_t)
        coeff = g_t if score_coeff == "g" else e
        x = x + ((v + coeff * s + steer_term) * dt_t + std * noise)
    return x


def test_the_score_keeps_raw_g_while_noise_and_steering_carry_tau_g():
    """The split is the whole positional content of the temperature patch: score gets g_t, the
    noise and grad_x Psi get eps_t = tau*g_t. The second assertion is what makes this sharp --
    putting eps_t on the score instead must NOT reproduce the run."""
    m = GaussianMixtureFlow(stds=(0.5, 0.5), eps_mode="sqrt1mt")
    pot = Pointwise(lambda q: q.reshape(-1))
    cfg = ConfigA(nsteps=40, lam=2.0, temperature=0.4, k_fp=1, damp=1.0,
                  resample_every=0, seed=5)
    out = steer(m, pot, cfg, 64)
    assert torch.equal(out["x"], _reference(m, 64, cfg, pot, score_coeff="g"))
    wrong = _reference(m, 64, cfg, pot, score_coeff="eps")
    assert float((out["x"] - wrong).abs().max()) > 1e-6, (
        "putting eps_t on the score reproduces the run bitwise, so this test cannot tell the two "
        "conventions apart and proves nothing")


def test_grad_x_psi_is_microbatch_size_invariant():
    """Microbatching the grad_x Psi backward must split the VJP and NOT the measure.

    The support and weights are detached and are always the full population, so every chunk size
    has to give the same trajectory. This is what lets the reward keep the paper's 100 samples
    per length at n_res=250, where a single-chunk backward fits only a handful of particles; if it
    failed, the objective would quietly differ between sequence lengths.

    Every chunk size is compared against the unchunked run INSIDE one test. Parametrising over
    chunk sizes cannot express this: each case would only compare its own result with itself.
    """
    from steering.rewards import SoftContactFeatures, ContactRepulsion

    n, L = 16, 24
    pot = ContactRepulsion(SoftContactFeatures(torch.ones(L, dtype=torch.bool)), h=0.3)
    cfg = dict(nsteps=6, lam=2.0, k_fp=3, damp=1.0, resample_every=0, seed=4,
               endpoint_lag="literal")
    runs = {c: steer(_Walk3D(L), pot, ConfigA(**cfg, grad_chunk=c), n)
            for c in (None, 1, 3, 7, 16)}
    ref = runs[None]
    for c, out in runs.items():
        if c is None:
            continue
        dx = float((out["x"] - ref["x"]).abs().max())
        da = float((out["a"] - ref["a"]).abs().max())
        assert dx < 1e-10 and da < 1e-10, (
            f"grad_chunk={c} diverges from the unchunked run: max|dx|={dx:.3e}, max|da|={da:.3e}")
    # and the run must not be trivial -- steering has to have moved something
    base = steer(_Walk3D(L), pot, ConfigA(**dict(cfg, lam=0.0), grad_chunk=None), n)
    assert float((ref["x"] - base["x"]).abs().max()) > 1e-8, (
        "the steered and unsteered runs coincide, so this test compares zeros")


class _Walk3D:
    """Minimal [L, 3] model for the contact reward, mirroring test_protein_rewards._Walk."""

    def __init__(self, length, k=0.5):
        self.length, self.k = length, k

    def prior(self, n, generator=None):
        x = torch.randn(n, self.length, 3, generator=generator) * 0.10
        x[:, :, 0] += torch.arange(self.length) * 0.38
        return x

    def endpoint(self, x, t):
        v = self.k * x
        return x + (1 - t.reshape(-1, 1, 1)) * v, v

    def score(self, x, v, t):
        return -x

    def eps(self, t):
        return 0.2
