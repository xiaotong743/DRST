"""toy.py: reproduce the paper's section 4.1 claims against closed-form ground truth."""

import math

import pytest
import torch
from scipy import stats

from steering.rewards import Pointwise, MMD
from steering.steerer import steer, Config
from toy import GaussianMixtureFlow, grid_mu_star

torch.set_default_dtype(torch.float64)

# paper section 4.1: base weights (1,3), target the same means with weights (3,1)
BASE = dict(means=(-1.0, 1.0), stds=(1.0, 1.0), weights=(1.0, 3.0))
TARGET_W = (3.0, 1.0)
LAM_STAR = 10.0
H = 0.2 * math.sqrt(2.0)          # bandwidth 0.2 in the exp(-d^2/2s^2) convention


def _model(eps_mode="const"):
    return GaussianMixtureFlow(eps_mode=eps_mode, eps_clamp=20.0, **BASE)


def test_lambda_zero_recovers_the_base_distribution():
    m = _model()
    out = steer(m, Pointwise(lambda th: th.squeeze(-1)),
                Config(nsteps=200, lam=0.0, resample_every=0, seed=0), 4000)
    ref = m.sample_p1(4000, torch.Generator().manual_seed(77))
    ks = stats.ks_2samp(out["x"].squeeze(-1).numpy(), ref.numpy())
    assert ks.pvalue > 0.01, f"lambda=0 does not reproduce p1 (KS p={ks.pvalue:.3g})"


def test_lambda_to_zero_slope_matches_covariance():
    """For a pointwise r, mu* ∝ e^{lam r} p1 exactly, so
    d/dlam E_{mu*}[f] |_{lam=0} = Cov_{p1}(f, r).
    With f = r = x and p1 = 0.25 N(-1,1) + 0.75 N(1,1), that is Var_{p1}(x) = 1.75.
    """
    m = _model()
    r = Pointwise(lambda th: th.squeeze(-1))
    n, lams = 8000, [0.0, 0.05, 0.10, 0.20]
    means = []
    for lam in lams:
        o = steer(m, r, Config(nsteps=200, lam=lam, resample_every=0, seed=4), n)
        means.append(float((o["w"] * o["x"].squeeze(-1)).sum()))
    lam_t = torch.tensor(lams)
    d = torch.tensor(means) - means[0]
    slope = float((lam_t * d).sum() / (lam_t ** 2).sum())      # least squares through origin
    # Measured convergence toward 1.75 (N=8000, seed 4), both eps schedules agreeing:
    #   steps    200     800    3200
    #   const   1.6515  1.6838  1.7050
    #   sqrt1mt 1.6602  1.6678  1.7068
    # The residual is terminal-stiffness discretisation from the 1/(1-t) score, shrinking
    # ~1.48x per 4x steps. 200 steps is the fast setting used here.
    assert abs(slope - 1.75) < 0.18, f"slope {slope:.4f} vs analytic 1.75"


def test_weighting_is_what_makes_the_slope_right():
    """The same trajectories without the corrective weights must miss the analytic slope.
    This isolates the contribution of the weight ODE (12b)."""
    m = _model()
    r = Pointwise(lambda th: th.squeeze(-1))
    n, lams = 8000, [0.0, 0.05, 0.10, 0.20]
    unw = []
    for lam in lams:
        o = steer(m, r, Config(nsteps=200, lam=lam, resample_every=0, seed=4,
                               reweight=False), n)
        unw.append(float(o["x"].squeeze(-1).mean()))
    lam_t = torch.tensor(lams)
    d = torch.tensor(unw) - unw[0]
    slope = float((lam_t * d).sum() / (lam_t ** 2).sum())
    assert abs(slope - 1.75) > 0.25, f"unweighted slope {slope:.4f} unexpectedly matches 1.75"


@pytest.mark.slow
def test_lambda_star_claim():
    """The paper's central claim (section 4.1 / Figure 3): the steering strength minimising the
    distance to mu* coincides with lam* for the mean-field sampler, and does not for
    gradient-only.
    """
    from toy import grid_mu_star, sample_grid, w1_to_grid
    SD, H, LAM_STAR, N, STEPS = 0.5, 5.0 * math.sqrt(2.0), 10.0, 3000, 100
    paper = dict(means=(-1.0, 1.0), stds=(SD, SD), weights=(1.0, 3.0))
    m = GaussianMixtureFlow(eps_mode="const", eps_clamp=20.0, **paper)
    tgt = GaussianMixtureFlow(means=(-1.0, 1.0), stds=(SD, SD), weights=TARGET_W)
    target = tgt.sample_p1(1000, torch.Generator().manual_seed(1)).reshape(-1, 1)
    grid, q = grid_mu_star(m, target, LAM_STAR, H)
    pot = MMD(target, H)
    unif = torch.full((N,), 1.0 / N)

    def w1(lam, **kw):
        v = []
        for sd in (4, 5):
            o = steer(m, pot, Config(nsteps=STEPS, lam=lam, resample_every=10, seed=sd, **kw), N)
            v.append(w1_to_grid(o["x"], o["w"] if kw.get("reweight", True) else unif, grid, q))
        return sum(v) / len(v)

    lams = (6.0, 8.0, 10.0, 12.0, 14.0)
    argmin = lambda c: lams[min(range(len(c)), key=lambda i: c[i])]

    # the paper's Figure-3 setting: K_FP = 3, undamped
    mf3 = [w1(l, k_fp=3, damp=1.0) for l in lams]
    assert argmin(mf3) == LAM_STAR, f"K_FP=3 argmin {argmin(mf3)} != lam* {LAM_STAR}; {mf3}"

    floor = float(torch.tensor(
        [w1_to_grid(sample_grid(grid, q, N, torch.Generator().manual_seed(s)), unif, grid, q)
         for s in range(6)]).mean())
    at_star = mf3[lams.index(LAM_STAR)]
    assert at_star < 2.5 * floor, f"W1 at lam* is {at_star:.4f}, floor {floor:.4f}"

    # K_FP = 1 is biased low, exactly as the paper reports and as their own curve shows
    mf1 = [w1(l, k_fp=1, damp=1.0) for l in lams]
    assert argmin(mf1) < LAM_STAR, f"K_FP=1 argmin {argmin(mf1)} should be below lam*; {mf1}"

    # gradient-only, eq (11) as written, must be far from mu* at lam*
    assert w1(LAM_STAR, reweight=False) > 5 * at_star
