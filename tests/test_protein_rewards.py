"""rewards.py: the contact features and the repulsion functional's first variation.

These check the reward's algebra and interface. They say nothing about the model's endpoint
gradient, which is where the binning question lives.
"""

import math

import pytest
import torch

from steering.rewards import rbf_gram
from steering.steerer import steer
from steering.rewards import (SoftContactFeatures, ContactRepulsion,
                                      calibrate_bandwidth)

torch.set_default_dtype(torch.float64)
L, N = 24, 6


def _pool(n=N, length=L, seed=0):
    """Helix-like chains with REAL contacts, and varying pitch so the kernel has structure.

    The previous fixture was an extended walk (0.38 nm per residue along x). Every pair is then
    at least 1.5 nm apart, so c_ij = sigmoid((0.8 - d)/0.05) ~ 5e-7 for all of them: the feature
    vectors are essentially zero, the Gram is all-ones, and psi is the constant -2, which makes
    the first variation identically zero. Several checks here passed in that degenerate regime.
    An alpha-helix puts CA(i) and CA(i+4) about 0.6 nm apart, inside the 0.8 nm cutoff.
    """
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(length, dtype=torch.get_default_dtype())
    out = []
    for _ in range(n):
        rad = 0.23 + 0.05 * torch.rand((), generator=g)
        rise = 0.14 + 0.06 * torch.rand((), generator=g)
        turn = 1.75 + 0.30 * torch.rand((), generator=g)
        out.append(torch.stack([rad * torch.cos(t * turn), rad * torch.sin(t * turn),
                                rise * t], -1) + 0.015 * torch.randn(length, 3, generator=g))
    return torch.stack(out)


def _feat():
    return SoftContactFeatures(torch.ones(L, dtype=torch.bool), min_sep=4,
                              cutoff_nm=0.8, width_nm=0.05)


def _rot(seed=0):
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=g))
    return q * torch.sign(torch.linalg.det(q))


def test_features_are_invariant_to_rotation_and_translation():
    f, x = _feat(), _pool()
    moved = x @ _rot(1).T + torch.tensor([1.5, -3.0, 0.25])
    assert float((f(x) - f(moved)).abs().max()) < 1e-12


def test_features_are_reflection_invariant_which_is_a_documented_limitation():
    """Pinned as a test, not only in prose: this reward cannot see chirality, so it must never be
    used alone to argue that steered backbones are structurally diverse."""
    f, x = _feat(), _pool()
    assert float((f(x) - f(x * torch.tensor([1.0, 1.0, -1.0]))).abs().max()) < 1e-12


def test_squared_feature_distance_is_the_mean_squared_contact_difference():
    f, x = _feat(), _pool()
    fx = f(x)
    raw = torch.sigmoid((f.cutoff - (x[:, f.i, :] - x[:, f.j, :]).norm(dim=-1)) / f.width)
    assert torch.allclose((fx[0] - fx[1]).pow(2).sum(), (raw[0] - raw[1]).pow(2).mean())


def test_psi_is_minus_two_times_the_weighted_kernel_row_with_the_diagonal_kept():
    """The diagonal contributes -2 w_i. It has no spatial gradient, but it does move the weights,
    so a leave-one-out variant would be a different finite-population update."""
    f, x = _feat(), _pool()
    pot = ContactRepulsion(f, h=0.3)
    w = torch.tensor([0.30, 0.25, 0.20, 0.13, 0.07, 0.05])
    k = rbf_gram(f(x), f(x), 0.3)
    assert torch.allclose(pot.psi(x, x, w), -2.0 * (k @ w))
    loo = k.clone().fill_diagonal_(0.0)
    assert float((pot.psi(x, x, w) - (-2.0 * (loo @ w))).abs().max()) > 1e-6


def test_the_spatial_query_gradient_matches_a_central_finite_difference():
    f, x = _feat(), _pool()
    pot = ContactRepulsion(f, h=0.3)
    w = torch.softmax(torch.linspace(-1, 1, N), 0)
    xg = x.clone().requires_grad_(True)
    pot.psi(xg, x, w).sum().backward()
    g = torch.Generator().manual_seed(7)
    d = torch.randn(x.shape, generator=g)
    eps = 1e-6
    fd = float((pot.psi(x + eps * d, x, w).sum() - pot.psi(x - eps * d, x, w).sum()) / (2 * eps))
    an = float((xg.grad * d).sum())
    assert abs(fd - an) <= 1e-6 * max(1.0, abs(an)), f"fd {fd:.10f} analytic {an:.10f}"


def test_psi_is_the_first_variation_of_the_functional():
    """d/ds R((1-s)nu + s delta_z)|_0 = psi(z; nu) - E_nu[psi]. This is the identity that makes
    psi the right thing to differentiate, independently of how it is coded."""
    f = _feat()
    # z must be genuinely unlike the pool, or psi(z) - E[psi] is near zero and an absolute
    # tolerance passes on noise. The earlier version compared |fd - exact| <= 1e-6 against a
    # quantity of magnitude 3e-6, i.e. it pinned psi only to about +/-30%.
    x, z = _pool(seed=0), _pool(n=1, seed=3)
    pot = ContactRepulsion(f, h=0.3)
    w = torch.softmax(torch.linspace(-0.5, 1.5, N), 0)

    def R(s):
        pts = torch.cat([x, z])
        ww = torch.cat([(1 - s) * w, torch.tensor([s])])
        return float(pot.value(pts, ww))

    h = 1e-6
    fd = (R(h) - R(-h)) / (2 * h)
    psi = pot.psi(x, x, w)
    exact = float(pot.psi(z, x, w)[0] - (w * psi).sum())
    assert abs(exact) > 1e-3, (
        f"the reference quantity is {exact:.3e}; too small for this check to constrain psi")
    assert abs(fd - exact) <= 1e-5 * abs(exact), (
        f"fd {fd:.10f} exact {exact:.10f}, relative gap {abs(fd - exact) / abs(exact):.2e}")


def test_value_is_minus_w_K_w():
    f, x = _feat(), _pool()
    pot = ContactRepulsion(f, h=0.3)
    w = torch.softmax(torch.linspace(0, 2, N), 0)
    k = rbf_gram(f(x), f(x), 0.3)
    assert torch.allclose(pot.value(x, w), -(w @ k @ w))


@pytest.mark.parametrize("w", ["uniform", "skewed"])
def test_weights_fn_agrees_with_psi_on_the_same_query_support_and_weights(w):
    """weights_fn is the Picard loop's only view of the reward, so a divergence between the two
    would make the fixed point solve a different functional than the drift follows."""
    f, x = _feat(), _pool()
    pot = ContactRepulsion(f, h=0.3)
    ww = (torch.full((N,), 1.0 / N) if w == "uniform"
          else torch.softmax(torch.linspace(-2, 3, N), 0))
    assert torch.allclose(pot.weights_fn(x, x)(ww), pot.psi(x, x, ww))


def test_weights_fn_reacts_to_the_candidate_weights():
    f, x = _feat(), _pool()
    fn = ContactRepulsion(f, h=0.3).weights_fn(x, x)
    a = fn(torch.full((N,), 1.0 / N))
    b = fn(torch.softmax(torch.linspace(-3, 3, N), 0))
    assert float((a - b).abs().max()) > 1e-6, (
        "the closure ignores its argument, so Picard would iterate a constant map")


def test_unnormalised_candidate_weights_are_renormalised():
    f, x = _feat(), _pool()
    pot = ContactRepulsion(f, h=0.3)
    w = torch.softmax(torch.linspace(-1, 2, N), 0)
    assert torch.allclose(pot.psi(x, x, 7.5 * w), pot.psi(x, x, w))


def test_bandwidth_calibration_puts_the_median_similarity_at_exp_minus_one():
    f, x = _feat(), _pool(n=24, seed=9)
    h, info = calibrate_bandwidth(f, x)
    # h^2 IS the median squared distance, so the kernel evaluated there is exactly exp(-1).
    assert abs(h ** 2 - info["d2_median"]) < 1e-12
    k = rbf_gram(f(x), f(x), h)
    off = k[~torch.eye(24, dtype=torch.bool)]
    # State the property without depending on a median convention: half the pairs must sit above
    # exp(-1) and half below. Comparing torch.median(off) to exp(-1) directly is off by one
    # element (torch.median takes the lower middle, and the kernel decreases in the distance),
    # and the size of that offset grows with the spread of the distances.
    # Counted, not as a fraction: at exactly 2 pairs off, the float form of the same comparison
    # is a tie that rounding decides (0.0036231884057971175 against 0.0036231884057971015).
    n_above = int((off > math.exp(-1.0)).sum())
    assert abs(2 * n_above - off.numel()) <= 4, (
        f"{n_above} of {off.numel()} pairs are above exp(-1), not half")
    assert info["n_features"] == f.i.numel() and info["h"] == h


def test_degenerate_features_raise_rather_than_returning_a_zero_bandwidth():
    f = _feat()
    same = _pool(n=1).repeat(5, 1, 1)
    with pytest.raises(ValueError, match="constant"):
        calibrate_bandwidth(f, same)


@pytest.mark.parametrize("bad,match", [
    (dict(mask=torch.ones(4, 5, dtype=torch.bool)), "1-D mask"),
    (dict(mask=torch.ones(L, dtype=torch.bool), width_nm=0.0), "positive smoothing"),
    (dict(mask=torch.ones(6, dtype=torch.bool), min_sep=9), "no valid pairs"),
])
def test_bad_feature_settings_raise(bad, match):
    with pytest.raises(ValueError, match=match):
        SoftContactFeatures(**bad)


def test_a_wrong_coordinate_shape_raises_instead_of_broadcasting():
    f = _feat()
    with pytest.raises(ValueError, match="coordinates in nm"):
        f(_pool(length=L + 1))
    with pytest.raises(ValueError, match="coordinates in nm"):
        f(_pool()[0])


def test_a_non_positive_bandwidth_is_rejected():
    for h in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="finite and positive"):
            ContactRepulsion(_feat(), h=h)


def test_algorithm_three_is_refused_rather_than_assumed():
    pot = ContactRepulsion(_feat(), h=0.3)
    x = _pool()
    for fn in (pot.phi_tilde, pot.phi_sum):
        with pytest.raises(NotImplementedError, match="Algorithm 3"):
            fn(x, x, torch.full((N,), 1.0 / N))


def test_masked_residues_are_excluded_from_every_pair():
    mask = torch.ones(L, dtype=torch.bool)
    mask[3] = mask[11] = False
    f = SoftContactFeatures(mask, min_sep=4)
    assert not bool(((f.i == 3) | (f.j == 3) | (f.i == 11) | (f.j == 11)).any())
    x = _pool()
    moved = x.clone()
    moved[:, 3] += 5.0
    assert float((f(x) - f(moved)).abs().max()) < 1e-12


class _Walk:
    """A 3-D toy meeting the endpoint contract, so `steer` can drive the REAL reward on CPU.

    This exercises the reward-to-sampler interface -- psi's signature, weights_fn under candidate
    weights, dtypes and shapes -- which no other test covers without a GPU and the checkpoint.
    v = k x, so div b = k * L * 3 exactly.
    """

    def __init__(self, length=L, k=0.5):
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


def _steered(eta, steps=12):
    from steer_a import ConfigA, SteererA
    cfg = ConfigA(nsteps=steps, lam=eta, k_fp=3, damp=1.0, resample_every=0,
                  endpoint_lag="literal", temperature=0.45, seed=0)
    return SteererA(_Walk(), ContactRepulsion(_feat(), h=0.3), cfg, N).run()


def test_steer_drives_the_contact_reward_end_to_end_on_cpu():
    out = _steered(1.0)
    assert torch.isfinite(out["x"]).all() and torch.isfinite(out["a"]).all()
    assert abs(float(out["w"].sum()) - 1.0) < 1e-12
    base = _steered(0.0)
    assert float((out["x"] - base["x"]).abs().max()) > 1e-8, (
        "the reward gradient never reached the positions, so grad_x Psi is zero everywhere")
    assert float((out["a"] - base["a"]).abs().max()) > 1e-8, (
        "the reward never reached the log-weights")


def test_a_stronger_reward_raises_the_weighted_diversity():
    """The functional's own direction: more reward strength, less average population similarity.
    Measured on the weighted terminal cloud, which is what the sampler returns."""
    pot = ContactRepulsion(_feat(), h=0.3)
    vals = [float(pot.value(o["x"].detach(), o["w"]))
            for o in (_steered(0.0), _steered(2.0), _steered(8.0))]
    assert vals[0] < vals[1] < vals[2], f"reward did not increase with eta: {vals}"


# ---- ClusterDiversity: the non-differentiable reward FK makes reachable --------------------

def _cd(labels):
    from steering.rewards import ClusterDiversity
    return ClusterDiversity(lambda theta: labels)


LABELS = [0, 0, 0, 1, 1, 2]          # masses 3/6, 2/6, 1/6 -> R = -14/36, Hill-2 = 18/7


def test_cluster_diversity_value_is_minus_sum_of_squared_masses():
    pot, w = _cd(LABELS), torch.full((N,), 1.0 / N)
    assert abs(float(pot.value(_pool(), w)) + 14.0 / 36.0) < 1e-12


def test_psi_is_the_first_variation_of_value():
    """R(mu) = -int int s dmu dmu is quadratic, so int psi dmu = 2R exactly. This is the identity
    the mean-field weights and FK's self-consistent solve both rely on."""
    pot, theta = _cd(LABELS), _pool()
    w = torch.rand(N).double(); w /= w.sum()
    assert abs(float((w * pot.psi(theta, theta, w)).sum())
               - 2.0 * float(pot.value(theta, w))) < 1e-12


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_psi_lands_on_the_particles_device(device):
    """labels() returns CPU ints whatever the particles are on -- a Python list from foldseek or
    tmtools. The co-cluster matrix must follow the particles, or every downstream weight is a
    silent CPU round-trip that then fails against a CUDA r_prev."""
    pot, theta = _cd(LABELS), _pool().to(device)
    w = torch.full((N,), 1.0 / N, device=device, dtype=theta.dtype)
    assert pot.psi(theta, theta, w).device.type == device
    assert pot.weights_fn(theta, theta)(w).device.type == device


def test_psi_is_bounded_and_ranks_crowded_particles_lowest():
    """psi in [-2, 0] is what makes the fixed point stable; the ordering is what steering uses."""
    pot, theta = _cd(LABELS), _pool()
    psi = pot.psi(theta, theta, torch.full((N,), 1.0 / N))
    assert float(psi.max()) <= 0.0 and float(psi.min()) >= -2.0
    assert float(psi[0]) < float(psi[3]) < float(psi[5])    # cluster sizes 3 > 2 > 1
