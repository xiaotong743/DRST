"""fk: the potential weight mode (arXiv:2501.06848) inside Algorithm 1's sampler.

The target is p1 * exp(lam*r), with potentials constrained so prod_t G_t = exp(lam*r(x1)). That
constraint is what every test here pins, directly or through its consequences.
"""

import math

import pytest
import torch

from steering.smc import ess, normalise
from steering.rewards import Pointwise
from steering.steerer import (POTENTIALS, Config, FKConfig, FKSteerer, Steerer,
                              log_potential)
from toy import GaussianMixtureFlow, grid_tempered, sample_grid, w1_to_grid

torch.set_default_dtype(torch.float64)

LAM = 1.5


def _model(eps_mode="const"):
    return GaussianMixtureFlow(eps_mode=eps_mode, eps_clamp=20.0, means=(-1.0, 1.0),
                               stds=(0.5, 0.5), weights=(1.0, 3.0))


def _reward(x):
    """Smooth, bounded, and not symmetric about either mode, so a wrong tilt shows up."""
    return torch.tanh(x.reshape(-1))


def _cfg(**kw):
    """FKConfig when a potential is named, else the plain mean-field Config."""
    base = dict(nsteps=60, lam=LAM, resample_every=0, seed=3)
    base.update(kw)
    return (FKConfig if "potential_type" in base else Config)(**base)


def steer(model, potential, cfg, n):                      # noqa: F811
    cls = FKSteerer if isinstance(cfg, FKConfig) else Steerer
    return cls(model, potential, cfg, n).run()


@pytest.mark.parametrize("kind", POTENTIALS)
def test_terminal_weight_is_exactly_lam_r_without_resampling(kind):
    """THE master invariant. prod_t G_t = exp(lam*r(x1)) means that with no resampling every
    potential type lands on the same weights, and they are SNIS weights.

    This one assertion pins the terminal correction, the logP bookkeeping, and the claim that the
    potential type only matters through resampling.

    The terminal reward is read at t = 1 - 1e-9, where the toy's denoiser is the identity only to
    O(1e-9) -- hence the tolerance. It is exact for ProteinaModel, which returns x unchanged at
    t >= 1 - 1e-6.
    """
    m, pot = _model(), Pointwise(_reward)
    out = steer(m, pot, _cfg(potential_type=kind), 256)
    want = LAM * _reward(out["x"]).double()
    assert torch.allclose(out["a"], want, atol=1e-7, rtol=0), \
        f"{kind}: max|a - lam*r| = {float((out['a'] - want).abs().max()):.3e}"


@pytest.mark.parametrize("kind", POTENTIALS)
def test_fk_does_not_move_the_particles(kind):
    """FK reweights; it must not touch the dynamics. Bitwise against the unsteered sampler, so
    a steered run stays seed-matched to its own control.
    """
    m, pot = _model(), Pointwise(_reward)
    base = steer(m, pot, _cfg(lam=0.0), 128)
    fk = steer(m, pot, _cfg(potential_type=kind), 128)
    assert torch.equal(base["x"], fk["x"])
    # and the weights really did something, or the test above is vacuous
    assert float(fk["a"].std()) > 0.1


def test_lam_zero_is_bitwise_unsteered_with_resampling_on():
    """The same identity every other branch has: at lam=0 nothing is weighted, uniform weights
    make residual resampling the identity, and no randomness is drawn."""
    m, pot = _model(), Pointwise(_reward)
    base = steer(m, pot, _cfg(lam=0.0, resample_every=7), 64)
    fk = steer(m, pot, _cfg(lam=0.0, resample_every=7, potential_type="diff"), 64)
    assert torch.equal(base["x"], fk["x"])
    assert float(fk["a"].abs().max()) == 0.0


@pytest.mark.parametrize("kind", ["diff", "max"])
def test_resampling_preserves_the_target(kind):
    """Resampling changes which particles survive and the weights they carry, but not what the
    weighted cloud estimates. Both runs share a seed, so this compares the estimator and not the
    draw. `add` is excluded: its spread grows with the checkpoint count. `is` is vacuous (a
    stays zero, nothing resamples) and `rt` is bit-identical to `diff`, since _resample zeros
    `a` at every checkpoint."""
    m, pot = _model(), Pointwise(_reward)
    n = 4000
    plain = steer(m, pot, _cfg(potential_type=kind), n)
    res = steer(m, pot, _cfg(potential_type=kind, resample_every=7), n)
    mean = lambda o: float((o["w"] * _reward(o["x"]).double()).sum())
    assert abs(mean(plain) - mean(res)) < 0.02, f"{kind}: {mean(plain):.4f} vs {mean(res):.4f}"


def test_resampling_moves_the_population_toward_high_reward():
    """The in-loop potential exists to make resampling favour high-reward lineages, and the
    terminal rule overwrites `a` -- so this is the ONLY place its SIGN is observable. Measured on
    the unweighted particles, which is what a downstream user holds.
    """
    m, pot = _model(), Pointwise(_reward)
    n = 2000
    base = steer(m, pot, _cfg(lam=0.0, resample_every=5), n)
    fk = steer(m, pot, _cfg(lam=4.0, potential_type="diff", resample_every=5), n)
    assert float(_reward(fk["x"]).mean()) > float(_reward(base["x"]).mean()) + 0.2


@pytest.mark.parametrize("kind", ["diff", "max", "add", "rt"])
def test_fk_weights_is_the_potential_it_claims(kind):
    """The arithmetic of one step, against the definitions written out by hand."""
    r = torch.tensor([0.5, -1.0, 2.0], dtype=torch.float64)
    prev = torch.tensor([1.0, -2.0, 0.0], dtype=torch.float64)
    a = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    logP = torch.tensor([0.7, 0.0, -0.4], dtype=torch.float64)
    got, new_prev = log_potential(kind, 2.0, r, a, prev, logP)
    if kind == "diff":
        assert torch.allclose(got, a + 2.0 * (r - prev)) and torch.equal(new_prev, r)
    else:
        g = {"max": torch.maximum(r, prev), "add": r + prev, "rt": r}[kind]
        assert torch.allclose(got, 2.0 * g - logP) and torch.allclose(new_prev, g)


def test_the_reward_is_never_differentiated():
    """The whole point: r may be a black box. A reward that raises on any grad path still runs."""
    class NoGrad:
        def psi(self, query, support, w):
            assert not query.requires_grad, "FK differentiated the reward"
            return torch.tanh(query.detach().reshape(-1))

        def weights_fn(self, query, support):
            val = self.psi(query, support, None)
            return lambda w: val

    out = steer(_model(), NoGrad(), _cfg(potential_type="diff"), 64)
    assert torch.isfinite(out["a"]).all()


@pytest.mark.slow
def test_fk_reaches_the_closed_form_tilted_target():
    """Absolute ground truth: grid_tempered at tau=1 IS p1*exp(lam*r), solved on a grid."""
    m, pot = _model(), Pointwise(_reward)
    n = 8000
    grid, q = grid_tempered(m, tau=1.0, reward=_reward, eta=LAM)
    out = steer(m, pot, _cfg(nsteps=200, potential_type="diff", resample_every=10), n)
    got = w1_to_grid(out["x"], out["w"], grid, q)
    floor = sum(w1_to_grid(sample_grid(grid, q, n, torch.Generator().manual_seed(s)),
                           torch.full((n,), 1.0 / n), grid, q) for s in range(4)) / 4.0
    assert got < max(4.0 * floor, 0.05), f"W1 to the tilted target {got:.4f}, MC floor {floor:.4f}"
    # sharpness: the UNSTEERED cloud must be far outside that tolerance
    base = steer(m, pot, _cfg(nsteps=200, lam=0.0), n)
    assert w1_to_grid(base["x"], base["w"], grid, q) > 3.0 * got
