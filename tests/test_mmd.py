"""rewards.py: kernel sanity, and the Gateaux identity that certifies psi = dR/dmu."""

import math
import torch
import pytest

from steering.rewards import rbf_gram, mmd2, psi

torch.set_default_dtype(torch.float64)


def _pts(n, d, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g)


def test_gram_symmetric_and_psd():
    x = _pts(12, 3, 0)
    k = rbf_gram(x, x, h=0.7)
    assert torch.allclose(k, k.T)
    assert torch.allclose(k.diagonal(), torch.ones(12))
    assert torch.linalg.eigvalsh(k).min() > -1e-10


def test_bandwidth_convention_relation():
    """Our h and the exp(-d^2/2s^2) convention are related by h = sqrt(2) s."""
    x, y = _pts(5, 2, 1), _pts(6, 2, 2)
    s = 0.4
    ours = rbf_gram(x, y, h=(2.0 ** 0.5) * s)
    other = torch.exp(-torch.cdist(x, y).pow(2) / (2 * s * s))
    assert torch.allclose(ours, other)


def test_mmd2_zero_for_identical_measures_and_positive_otherwise():
    x = _pts(20, 2, 3)
    assert mmd2(x, x, h=0.5).abs() < 1e-12
    y = _pts(20, 2, 4) + 3.0
    assert mmd2(x, y, h=0.5) > 0


def test_mmd2_invariant_to_weight_renormalisation():
    """mmd2 must normalise its weights internally, so an unnormalised vector and its normalised
    form agree. The two arguments have to be genuinely different objects: passing the same values
    twice is an identity that any weight handling satisfies."""
    g = torch.Generator().manual_seed(5)
    x = torch.randn(30, 2, generator=g)
    y = torch.randn(22, 2, generator=g)
    w = torch.rand(30, generator=g) + 0.05
    for scale in (1.0, 3.7, 1e-3):
        got = mmd2(x, y, 1.5, wx=w * scale)
        want = mmd2(x, y, 1.5, wx=w / w.sum())
        assert abs(got - want) < 1e-12, scale
    # and it must actually depend on the weights, or the test above is vacuous
    skew = torch.zeros(30); skew[0] = 1.0
    assert abs(mmd2(x, y, 1.5, wx=skew) - mmd2(x, y, 1.5)) > 1e-3


@pytest.mark.parametrize("h", [0.5, 1.0, 5.0 * math.sqrt(2.0)])
def test_gateaux_identity(h):
    """(R(mu + e(delta_q - mu)) - R(mu)) / e  ->  psi(q, mu) - E_mu[psi]."""
    z, y = _pts(15, 2, 7), _pts(11, 2, 8)
    w = torch.rand(15).add(0.1)
    w = w / w.sum()
    q = _pts(1, 2, 9)

    def R(weights, support):
        return -mmd2(support, y, h, wx=weights)

    # mu + e(delta_q - mu) as a weighted measure on [z; q]
    support = torch.cat([z, q], dim=0)
    base = torch.cat([w, torch.zeros(1)])
    delta = torch.cat([torch.zeros(15), torch.ones(1)])

    p = psi(q, z, w, y, h)                      # [1]
    expect = (p - (psi(z, z, w, y, h) * w).sum()).item()

    prev = None
    for e in (1e-3, 1e-4, 1e-5):
        fd = ((R(base + e * (delta - base), support) - R(base, support)) / e).item()
        err = abs(fd - expect)
        if prev is not None:
            assert err < prev            # error shrinks with e
        prev = err
    assert abs(fd - expect) < 1e-4 * max(1.0, abs(expect))


def test_rbf_gram_is_accurate_where_the_expansion_is_not():
    """rbf_gram forms the squared distance as a direct difference, so forward-mode AD works
    (torch.cdist has no jvp rule) AND the d^2 << |a|^2 regime stays accurate.
    """
    def exact(a, b, h):
        d2 = (a.double()[:, None, :] - b.double()[None, :, :]).pow(2).sum(-1)
        return torch.exp(-d2 / (h * h))

    g = torch.Generator().manual_seed(11)
    for dtype, tol in ((torch.float64, 1e-12), (torch.float32, 1e-6)):
        for scale in (1.0, 100.0, 1e4):
            base = (torch.randn(40, 3, generator=g) * scale).to(dtype)
            near = base[:25] + torch.full((25, 3), 1e-3, dtype=dtype)   # d^2 << |a|^2
            for a, b in ((base, near), (base, base[:25])):
                for h in (0.5, 2.0, 7.071):
                    err = float((rbf_gram(a, b, h).double() - exact(a, b, h)).abs().max())
                    assert err < tol, (dtype, scale, h, err)


def test_rbf_gram_supports_forward_mode_ad():
    """Algorithm 3 takes a jvp through the kernel (implt.psidot_terms), which torch.cdist cannot
    do. Pin it so a future switch back to cdist fails here rather than inside the solver."""
    a = torch.randn(6, 2, dtype=torch.float64)
    b = torch.randn(4, 2, dtype=torch.float64)
    _, tang = torch.func.jvp(lambda z: rbf_gram(a, z, 2.0), (b,), (torch.ones_like(b),))
    assert tang.shape == (6, 4)
    assert float(tang.abs().max()) > 0, "jvp returned an all-zero tangent"
