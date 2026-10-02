"""The soft pair-distance binning must be a drop-in for Proteina's hard one.

`soft_binning()` claims the frozen model IS the soft-binned model. That only holds if the
substitution is invisible to the network everywhere except in the gradient it restores, so these
check the contract rather than the physics: same shape, same dtype, same normalisation, and the
patch reaching exactly the intended call sites.
"""

import pytest
import torch

pytest.importorskip("proteinfoundation")

from proteinfoundation.nn import feature_factory as ff          # noqa: E402
from steering.models.proteina import soft_bin_pairwise_distances, soft_binning   # noqa: E402

MIN_D, MAX_D, DIM = 0.1, 3.0, 64        # ca_af3_200M_yes_tri.yaml


def _x(dtype=torch.float32):
    g = torch.Generator().manual_seed(0)
    return torch.randn(3, 8, 3, generator=g).to(dtype) * 0.5


@pytest.mark.parametrize("default", [torch.float32, torch.float64])
@pytest.mark.parametrize("xdtype", [torch.float32, torch.float64])
def test_soft_matches_hard_in_shape_and_dtype(default, xdtype):
    """The dtype half of this failed in a real run: hard follows the DEFAULT dtype via
    `one_hot(...) * 1.0`, so returning x.dtype crashed the first Linear in the pair path."""
    old = torch.get_default_dtype()
    try:
        torch.set_default_dtype(default)
        x = _x(xdtype)
        hard = ff.bin_pairwise_distances(x, MIN_D, MAX_D, DIM)
        soft = soft_bin_pairwise_distances(x, MIN_D, MAX_D, DIM)
        assert soft.shape == hard.shape, f"{tuple(soft.shape)} vs {tuple(hard.shape)}"
        assert soft.dtype == hard.dtype, f"{soft.dtype} vs {hard.dtype}"
    finally:
        torch.set_default_dtype(old)


@pytest.mark.parametrize("scale,shift", [(0.5, 1.0), (1.0, 0.0), (3.0, 0.0)])
def test_both_normalisations_partition_unity_everywhere_including_out_of_range(scale, shift):
    """Hard is one-hot; soft is a hat interpolation. Both must partition unity on EVERY entry,
    so the pair representation keeps its scale.

    Every entry, not just the interior off-diagonal ones. The earlier version of this test used
    one fixture whose distances all lay inside [min_dist, max_dist] and then excluded the
    diagonal, which is precisely where the defect lived: soft binning had no open-ended first
    and last bin, so 21.9% of pairs at prior scale -- and every d=0 diagonal entry -- came back
    as an all-zero row while hard binning returned the catch-all one-hot. The `scale` values
    here straddle the bin range in both directions, and d=0 covers the low catch-all.
    """
    x = _x() * scale + shift
    for fn in (ff.bin_pairwise_distances, soft_bin_pairwise_distances):
        out = fn(x, MIN_D, MAX_D, DIM)
        sums = out.sum(-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), (
            f"{fn.__name__} at scale {scale}: row sums span "
            f"[{float(sums.min()):.4f}, {float(sums.max()):.4f}]; "
            f"{int((sums == 0).sum())} of {sums.numel()} rows are all zero")


@pytest.mark.parametrize("scale", [0.5, 1.0, 3.0])
def test_soft_binning_puts_its_mass_in_the_same_bin_hard_binning_chooses(scale):
    """The strongest drop-in property available without the network: the hat's dominant bin must
    be the bin bucketize selects, including in the two catch-all bins."""
    x = _x() * scale
    hard = ff.bin_pairwise_distances(x, MIN_D, MAX_D, DIM)
    soft = soft_bin_pairwise_distances(x, MIN_D, MAX_D, DIM)
    agree = (soft.argmax(-1) == hard.argmax(-1)).double().mean()
    assert float(agree) == 1.0, f"argmax agrees on only {float(agree):.1%} of pairs"


def test_only_soft_carries_a_gradient():
    """The reason for the whole exercise: bucketize + one_hot is piecewise constant, so the
    pair-distance path contributes exactly zero to grad_x Psi.

    The bin axis is contracted with weights rather than summed. A plain sum is the one
    functional that CANNOT have a gradient here: both binnings partition unity over bins, so
    out.sum() is the pair count, identically constant, and the test would report no gradient for
    soft binning as well -- which is what it did on the first run.
    """
    wts = torch.arange(DIM, dtype=torch.float64)
    for fn, want in ((ff.bin_pairwise_distances, False), (soft_bin_pairwise_distances, True)):
        x = _x(torch.float64).requires_grad_(True)
        out = fn(x, MIN_D, MAX_D, DIM)
        if not out.requires_grad:
            assert not want, f"{fn.__name__} produced no graph at all"
            continue
        (out * wts).sum().backward()
        got = x.grad is not None and float(x.grad.abs().max()) > 0
        assert got == want, f"{fn.__name__}: gradient present={got}, expected {want}"
        assert x.grad is None or torch.isfinite(x.grad).all(), "non-finite gradient"


def test_soft_binning_patches_only_the_pairwise_distance_entry_point():
    """`rel_seq_sep` also bins, through `bin_and_one_hot`, with half-integer limits on integer
    separations -- a hat there would split every separation 50/50 and destroy the model's main
    inductive bias. The context manager must not touch it, and must restore on exit."""
    hard_fn, hard_boh = ff.bin_pairwise_distances, ff.bin_and_one_hot
    with soft_binning():
        assert ff.bin_pairwise_distances is soft_bin_pairwise_distances
        assert ff.bin_and_one_hot is hard_boh
    assert ff.bin_pairwise_distances is hard_fn
    assert ff.bin_and_one_hot is hard_boh


def test_soft_binning_restores_the_original_after_an_exception():
    hard_fn = ff.bin_pairwise_distances
    with pytest.raises(RuntimeError, match="boom"):
        with soft_binning():
            raise RuntimeError("boom")
    assert ff.bin_pairwise_distances is hard_fn
