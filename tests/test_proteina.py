"""Phase 1: the Proteina adapter. Requires a GPU and the 2.5 GB checkpoint.

The gate is the lambda=0 bitwise identity against Proteina's own fm.full_simulation.
"""
from functools import partial

import pytest
import torch

pytestmark = pytest.mark.gpu

CKPT = "/lustre/groups/ml01/workspace/xiaotong.fu/playground/proteina_data/ckpts/proteina_v1.1_DFS_200M_tri.ckpt"
DATA = "/lustre/groups/ml01/workspace/xiaotong.fu/playground/proteina_data"
N, NRES, STEPS = 4, 64, 100


@pytest.fixture(scope="module")
def mdl():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    from steering.models.proteina import ProteinaModel
    return ProteinaModel(CKPT, DATA, n_res=NRES, n_particles=N, nsteps=STEPS,
                         gt_mode="us", gt_p=1.0, gt_clamp_val=10.0,
                         schedule_mode="log", schedule_p=2.0)


def test_flow_matcher_conventions(mdl):
    assert mdl.fm.zero_com is True and mdl.fm.scale_ref == 1.0
    assert sum(p.requires_grad for p in mdl.model.parameters()) == 0


def test_endpoint_restores_the_velocity_identity(mdl):
    """predict_clean_n_v_w_guidance returns an uncentred x_pred, so we recentre and recompute v.
    v == (theta - x)/(1-t) is what the weight term b.grad Psi depends on."""
    x = mdl.prior(N)
    t = torch.full((N,), 0.4, dtype=x.dtype, device=x.device)
    theta, v = mdl.endpoint(x, t)
    assert torch.allclose(v, (theta - x) / (1 - 0.4), atol=1e-5)
    com = theta.mean(dim=1).norm(dim=-1)
    assert float(com.max()) < 1e-5, f"theta not centred: max ||COM|| = {float(com.max()):.2e}"


def test_units_are_nanometres(mdl):
    """Rg in nm must be O(1); a 10x Angstrom error trips this immediately."""
    from steering.models.proteina import radius_of_gyration, ca_ca_spacing
    x = mdl.prior(N)
    t = torch.full((N,), 0.99, dtype=x.dtype, device=x.device)
    theta, _ = mdl.endpoint(x, t)
    rg = radius_of_gyration(theta, mdl.mask[:N])
    assert (0.05 < rg).all() and (rg < 5.0).all(), f"Rg out of nm range: {rg.tolist()}"
    assert ca_ca_spacing(theta).shape == (N,)


def test_reward_is_se3_invariant(mdl):
    from steering.models.proteina import radius_of_gyration
    x = mdl.prior(N)
    g = torch.Generator(device="cpu").manual_seed(0)
    r0 = radius_of_gyration(x, mdl.mask[:N])
    for _ in range(8):
        q, _ = torch.linalg.qr(torch.randn(3, 3, generator=g, dtype=torch.float64))
        q = q.to(x.dtype).to(x.device)
        shift = torch.randn(3, generator=g, dtype=torch.float64).to(x.dtype).to(x.device)
        r1 = radius_of_gyration(x @ q + shift, mdl.mask[:N])
        assert torch.allclose(r0, r1, atol=1e-4)


def test_lambda_zero_matches_full_simulation_bitwise(mdl):
    """THE GATE. Both paths must consume the global torch RNG in the same order: one randn in
    sample_reference, then one per step that actually adds noise.
    """
    from steering.rewards import Pointwise
    from steering.steerer import steer, Config
    from steering.models.proteina import radius_of_gyration

    torch.manual_seed(1234)
    ref = mdl.fm.full_simulation(
        partial(mdl.model.predict_clean_n_v_w_guidance,
                guidance_weight=1.0, autoguidance_ratio=0.0),
        dt=1.0 / STEPS, nsamples=N, n=NRES, self_cond=False, cath_code=None,
        device=mdl.device, mask=mdl.mask[:N],
        schedule_mode="log", schedule_p=2.0, sampling_mode="sc",
        sc_scale_noise=1.0, sc_scale_score=1.0,
        gt_mode="us", gt_p=1.0, gt_clamp_val=10.0, dtype=mdl.dtype)

    torch.manual_seed(1234)
    out = steer(mdl, Pointwise(lambda th: radius_of_gyration(th, mdl.mask[:th.shape[0]])),
                Config(nsteps=STEPS, lam=0.0, resample_every=0, global_rng=True,
                       vf_tail_t=0.99), N, device=mdl.device)

    d = (out["x"] - ref).abs().max()
    assert float(out["a"].abs().max()) == 0.0
    assert float(d) == 0.0, f"lambda=0 not bitwise: max|dx| = {float(d):.3e}"


@pytest.mark.parametrize("gamma", [0.45, 0.3])
def test_low_temperature_matches_full_simulation_bitwise(mdl, gamma):
    """THE GATE FOR THE TEMPERATURE SPLIT. Proteina scales the noise VARIANCE by
    sc_scale_noise (std_eps = sqrt(2*gt*sc_scale_noise*dt), r3n_fm.py:332) and leaves the RAW gt
    on the score, so `temperature=gamma` must reproduce their sampler at that gamma exactly.

    Putting eps_t = gamma*gt on the score instead would still look like a plausible
    low-temperature sampler and would silently be a different process; this is what rules it out.
    """
    from steering.rewards import Pointwise
    from steering.steerer import steer, Config
    from steering.models.proteina import radius_of_gyration

    torch.manual_seed(1234)
    ref = mdl.fm.full_simulation(
        partial(mdl.model.predict_clean_n_v_w_guidance,
                guidance_weight=1.0, autoguidance_ratio=0.0),
        dt=1.0 / STEPS, nsamples=N, n=NRES, self_cond=False, cath_code=None,
        device=mdl.device, mask=mdl.mask[:N],
        schedule_mode="log", schedule_p=2.0, sampling_mode="sc",
        sc_scale_noise=gamma, sc_scale_score=1.0,
        gt_mode="us", gt_p=1.0, gt_clamp_val=10.0, dtype=mdl.dtype)

    torch.manual_seed(1234)
    from steer_a import ConfigA, SteererA
    out = SteererA(mdl, Pointwise(lambda th: radius_of_gyration(th, mdl.mask[:th.shape[0]])),
                   ConfigA(nsteps=STEPS, lam=0.0, resample_every=0, global_rng=True,
                           vf_tail_t=0.99, temperature=gamma), N, device=mdl.device).run()
    d = float((out["x"] - ref).abs().max())
    assert d == 0.0, f"temperature={gamma} not bitwise against full_simulation: max|dx| = {d:.3e}"
    assert float(out["a"].abs().max()) == 0.0, "lam=0 without the FK must not touch the weights"




def test_reward_endpoint_is_finite_at_t_equal_one(mdl):
    """The weight update evaluates the endpoint at min(t+dt, 1-1e-9), and 1-1e-9 is exactly 1.0
    in float32, so the final step of every run hits this. `endpoint` there divides by 1-t = 0."""
    x = mdl.prior(N)
    # The PREMISE, tested on the value steer() actually computes: min(t+dt, 1 - 1e-9) cast to
    # the model dtype. Asserting torch.ones() == 1.0 instead was a tautology that pinned nothing.
    clamped = torch.full((N,), min(1.0, 1.0 - 1e-9), dtype=x.dtype, device=x.device)
    assert float(clamped[0]) == 1.0, (
        f"1 - 1e-9 no longer rounds to 1.0 in {x.dtype}; the guard's premise has changed")
    th = mdl.reward_endpoint(x, clamped)
    assert torch.isfinite(th).all() and torch.equal(th, x)
    _, v_bad = mdl.endpoint(x, clamped)
    assert not torch.isfinite(v_bad).all(), (
        "endpoint at t=1 is finite, so reward_endpoint's guard is protecting against nothing")


@pytest.fixture(scope="module")
def mdl_pub():
    """The PUBLISHED Table-1 sampler: gt_mode '1/t', no clamp, dt = 0.0025. The A/B experiments
    run here, and it is NOT the configuration the other gates cover (those use 'us', clamp 10,
    100 steps), so its correctness was never actually established."""
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    from steering.models.proteina import ProteinaModel
    return ProteinaModel(CKPT, DATA, n_res=NRES, n_particles=N, nsteps=400,
                         gt_mode="1/t", gt_p=1.0, gt_clamp_val=None,
                         schedule_mode="log", schedule_p=2.0)


def test_published_sampler_matches_full_simulation_bitwise(mdl_pub):
    from steering.rewards import Pointwise
    from steering.steerer import steer, Config
    from steering.models.proteina import radius_of_gyration

    torch.manual_seed(1234)
    ref = mdl_pub.fm.full_simulation(
        partial(mdl_pub.model.predict_clean_n_v_w_guidance,
                guidance_weight=1.0, autoguidance_ratio=0.0),
        dt=0.0025, nsamples=N, n=NRES, self_cond=False, cath_code=None,
        device=mdl_pub.device, mask=mdl_pub.mask[:N],
        schedule_mode="log", schedule_p=2.0, sampling_mode="sc",
        sc_scale_noise=0.45, sc_scale_score=1.0,
        gt_mode="1/t", gt_p=1.0, gt_clamp_val=None, dtype=mdl_pub.dtype)

    torch.manual_seed(1234)
    from steer_a import ConfigA, SteererA
    out = SteererA(mdl_pub,
                   Pointwise(lambda th: radius_of_gyration(th, mdl_pub.mask[:th.shape[0]])),
                   ConfigA(ts=mdl_pub.ts, lam=0.0, resample_every=0, global_rng=True,
                           vf_tail_t=0.99, temperature=0.45), N).run()
    d = float((out["x"] - ref).abs().max())
    assert d == 0.0, f"published sampler not bitwise: max|dx| = {d:.3e} nm"


def test_soft_binning_does_not_wreck_the_published_sampler(mdl_pub):
    """Soft binning redefines the frozen model. Rg and CA-CA were measured as unchanged, but
    those are insensitive summaries -- this compares the actual coordinates and the endpoint
    velocity, which is what the sampler integrates."""
    from steering.models.proteina import soft_binning

    x = mdl_pub.prior(N)
    t = torch.full((N,), 0.5, dtype=x.dtype, device=x.device)
    with torch.no_grad():
        th_hard, v_hard = mdl_pub.endpoint(x, t)
        with soft_binning():
            th_soft, v_soft = mdl_pub.endpoint(x, t)
    rel_th = float((th_soft - th_hard).norm() / th_hard.norm())
    rel_v = float((v_soft - v_hard).norm() / v_hard.norm())
    print(f"\nsoft vs hard at t=0.5: relative |dtheta| = {rel_th:.4f}, |dv| = {rel_v:.4f}")
    assert rel_th > 0, "soft binning changed nothing, so it cannot be supplying a gradient"
    # A single step's endpoint should not move by order 1: that would be a different model.
    assert rel_th < 0.25, (
        f"soft binning moves the endpoint by {rel_th:.1%} at t=0.5, so the soft-binned model is "
        f"not a perturbation of the frozen one and every absolute number it produces belongs to "
        f"a different model")




