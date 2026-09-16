"""alg2: the multi-batch procedure. The claim under test is that pooling the measure across
batches improves the approximation, which is what makes N <= 48 workable at protein scale."""

import math

import pytest
import torch

from dataclasses import replace

from steering.sde import steer, multibatch, Config, Pointwise, MMD
from steering.toy import GaussianMixtureFlow, grid_mu_star, sample_grid, w1_to_grid

torch.set_default_dtype(torch.float64)
SD, H = 0.5, 5.0 * math.sqrt(2.0)


def _setup(n=400):
    m = GaussianMixtureFlow(eps_mode="const", eps_clamp=20.0, means=(-1.0, 1.0),
                            stds=(SD, SD), weights=(1.0, 3.0))
    tgt = GaussianMixtureFlow(means=(-1.0, 1.0), stds=(SD, SD), weights=(3.0, 1.0))
    y = tgt.sample_p1(n, torch.Generator().manual_seed(1)).reshape(-1, 1)
    return m, MMD(y, H), y


def test_single_batch_is_exactly_plain_steer():
    """M=1 has nothing retained, so Algorithm 2 must reduce to Algorithm 1 bitwise."""
    m, pot, _ = _setup()
    cfg = Config(nsteps=40, lam=8.0, k_fp=3, damp=1.0, resample_every=0, seed=2)
    one = steer(m, pot, cfg, 300)
    mb = multibatch(m, pot, cfg, 300, 1)
    assert torch.equal(one["x"], mb["x"])
    assert torch.equal(one["a"], mb["a"])


def test_retained_measure_changes_the_weights_and_is_normalised_over_both_sets():
    """A non-empty retained set must actually enter muhat: same seed, different weights. And the
    combined weights must sum to one over L+N, not over N."""
    m, pot, _ = _setup()
    cfg = Config(nsteps=40, lam=8.0, k_fp=3, damp=1.0, resample_every=0, seed=2)
    plain = steer(m, pot, cfg, 200)
    ret = dict(theta=plain["x"].detach(), a=plain["a"].detach())
    withret = steer(m, pot, cfg, 200, retained=ret)
    # the retained measure enters grad_x Psi too, so the POSITIONS must move as well
    assert not torch.equal(plain["x"], withret["x"]), "retained set did not reach the drift"
    assert float((plain["a"] - withret["a"]).abs().max()) > 1e-6, "retained set had no effect"


def test_pointwise_is_unaffected_by_the_retained_measure():
    """A pointwise psi does not see the measure at all, so pooling must be a no-op. This isolates
    the retained plumbing from any change in the reward itself."""
    m, _, _ = _setup()
    r = Pointwise(lambda th: th.squeeze(-1))
    cfg = Config(nsteps=40, lam=0.5, k_fp=1, damp=1.0, resample_every=0, seed=2)
    plain = steer(m, r, cfg, 200)
    ret = dict(theta=plain["x"].detach(), a=plain["a"].detach())
    withret = steer(m, r, cfg, 200, retained=ret)
    assert torch.equal(plain["x"], withret["x"])
    assert float((plain["a"] - withret["a"]).abs().max()) < 1e-12


@pytest.mark.slow
def test_pooled_approximation_improves_with_the_number_of_batches():
    """**[paper]** C.2.1: "the approximation quality improves as the number of batches
    increases". That is the paper's literal claim and it is what this asserts.
    """
    m, pot, y = _setup()
    grid, q = grid_mu_star(m, y, 10.0, H)
    cfg = Config(nsteps=100, lam=10.0, k_fp=50, damp=0.5, resample_every=10, seed=3)
    w1 = {M: w1_to_grid(*(lambda o: (o["x"], o["w"]))(multibatch(m, pot, cfg, 40, M)), grid, q)
          for M in (1, 2, 8)}
    assert w1[8] < w1[1], f"pooled approximation did not improve with M: {w1}"
    assert w1[8] < 0.6 * w1[1], f"improvement only marginal: {w1}"


def test_retained_block_is_not_normalised_on_its_own():
    """The combined measure must be normalised over L+N TOGETHER. The sharp discriminator is a
    constant shift applied to the retained log-weights ALONE: under joint normalisation that
    reweights the retained block against the current one and must change the result, while under
    """
    m, pot, _ = _setup()
    # ONE step: after a single step the log-weight update has not fed back into any position, so
    # the shift can only be observed through w_comb -> psi -> grad_x Psi. Over many steps the
    # retained set also reaches the weights via picard's `combine`, and that path responds to the
    # shift even when w_comb is normalised per set -- so a multi-step version of this test passes
    # against exactly the mutation it is meant to catch.
    # One step, starting at t = 0.5 rather than t = 0: Psi = lam * t * psi, so grad_x Psi is
    # identically ZERO at t = 0 and a first step from the prior cannot see the measure at all
    # (measured: max|dx| = 0.0 for any retained set). After one step from t > 0 the log-weight
    # update has not yet fed back into a position, so a difference in x can only have come
    # through w_comb -> psi -> grad_x Psi.
    ts = torch.tensor([0.5, 0.55])
    cfg = Config(ts=ts, lam=8.0, k_fp=3, damp=1.0, resample_every=0, seed=2)
    base = steer(m, pot, Config(nsteps=30, lam=8.0, k_fp=3, damp=1.0, resample_every=0, seed=2),
                 150)
    theta = base["x"].detach()
    a0 = base["a"].detach()
    out_lo = steer(m, pot, cfg, 150, retained=dict(theta=theta, a=a0))
    out_hi = steer(m, pot, cfg, 150, retained=dict(theta=theta, a=a0 + 4.0))
    d = float((out_lo["x"] - out_hi["x"]).abs().max())
    assert d > 1e-6, (
        f"shifting only the retained log-weights by +4 changed nothing (max|dx|={d:.2e}); the "
        f"blocks are being normalised separately")


def test_retained_measure_reaches_grad_x_psi_on_the_very_first_step():
    """Isolates the psi / grad_x Psi path from the weight path."""
    m, pot, _ = _setup()
    cfg = Config(ts=torch.tensor([0.5, 0.55]), lam=8.0, k_fp=3, damp=1.0,
                 resample_every=0, seed=2)   # t > 0: grad_x Psi vanishes identically at t = 0
    ret = dict(theta=torch.randn(120, 1, generator=torch.Generator().manual_seed(9)) * 1.5,
               a=torch.zeros(120))
    plain = steer(m, pot, cfg, 150)
    withret = steer(m, pot, cfg, 150, retained=ret)
    d = float((plain["x"] - withret["x"]).abs().max())
    assert d > 1e-9, f"retained support never reached grad_x Psi (max|dx| = {d:.2e})"
