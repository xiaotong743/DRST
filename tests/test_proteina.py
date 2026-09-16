"""Phase 1: the Proteina adapter. Requires a GPU and the 2.5 GB checkpoint.

The gate is the lambda=0 bitwise identity against Proteina's own fm.full_simulation.
"""
import os
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
    from steering.proteina import ProteinaModel
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
    from steering.proteina import radius_of_gyration, ca_ca_spacing
    x = mdl.prior(N)
    t = torch.full((N,), 0.99, dtype=x.dtype, device=x.device)
    theta, _ = mdl.endpoint(x, t)
    rg = radius_of_gyration(theta, mdl.mask[:N])
    assert (0.05 < rg).all() and (rg < 5.0).all(), f"Rg out of nm range: {rg.tolist()}"
    assert ca_ca_spacing(theta).shape == (N,)


def test_reward_is_se3_invariant(mdl):
    from steering.proteina import radius_of_gyration
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
    from steering.sde import steer, Config, Pointwise
    from steering.proteina import radius_of_gyration

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
