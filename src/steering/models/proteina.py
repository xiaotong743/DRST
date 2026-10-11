"""Proteina adapter
"""

import contextlib
import math
import os

import torch

from .base_model import BaseModel


class ProteinaModel(BaseModel):
    def __init__(self, ckpt, data_path, n_res, n_particles, device="cuda",
                 gt_mode="us", gt_p=1.0, gt_clamp_val=10.0,
                 schedule_mode="log", schedule_p=2.0, nsteps=100,
                 store_dir=None, dtype=torch.float32):
        os.environ.setdefault("DATA_PATH", str(data_path))
        from proteinfoundation.proteinflow.proteina import Proteina

        self.model = Proteina.load_from_checkpoint(
            ckpt, map_location="cpu",
            store_dir=store_dir or os.path.join(os.environ.get("TMPDIR", "/tmp"), "proteina_tmp"))
        self.model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.fm = self.model.fm
        assert self.fm.zero_com is True and self.fm.scale_ref == 1.0, \
            f"unexpected flow matcher: zero_com={self.fm.zero_com} scale_ref={self.fm.scale_ref}"
        assert self.model.cfg_exp.model.target_pred == "v"

        self.device, self.dtype, self.n_res = device, dtype, n_res
        self.mask = torch.ones(n_particles, n_res, dtype=torch.bool, device=device)
        self.ts = self.fm.get_schedule(schedule_mode, nsteps, p1=schedule_p)
        # eps precomputed on the grid: get_gt is not pointwise in t unless gt_p == 1
        self._eps = self.fm.get_gt(self.ts[:-1], gt_mode, gt_p, gt_clamp_val).clone()
        self.n_forward = 0

    def prior(self, n, generator=None):
        """Uses the GLOBAL torch RNG when generator is None, matching sample_reference, so a
        lambda=0 run can be compared bitwise against fm.full_simulation."""
        return self.fm.sample_reference(self.n_res, shape=(n,), device=self.device,
                                        dtype=self.dtype, mask=self.mask[:n])

    def endpoint(self, x, t):
        """
        one step forward of E[x1|x_t] 
        """
        mask = self.mask[:x.shape[0]]
        batch = {"x_t": x, "t": t, "mask": mask}          # exactly these three keys
        self.n_forward += 1
        x_pred, _ = self.model.predict_clean_n_v_w_guidance(
            batch, guidance_weight=1.0, autoguidance_ratio=0.0)
        v = self.fm.xt_dot(x_pred, x, t, mask)
        theta = self.fm._mask_and_zero_com(x_pred, mask)
        return theta, v

    def score(self, x, v, t):
        n = x.shape[1]
        t_ext = torch.full((x.shape[0], n), float(t), dtype=x.dtype, device=x.device)
        return self.fm.vf_to_score(x, v, t_ext)

    def eps(self, t):
        """Nearest grid point. get_gt is only pointwise in t when gt_p == 1, so eps must be read
        off the precomputed step grid rather than recomputed at an arbitrary t."""
        i = int((self.ts[:-1] - float(t)).abs().argmin())
        assert abs(float(self.ts[i]) - float(t)) < 1e-6, (
            f"t={t} is not on the model's schedule; pass Config(ts=model.ts) or let steer() "
            f"use it (nearest grid point {float(self.ts[i])})")
        return float(self._eps[i])

    def project(self, x):
        return self.fm._mask_and_zero_com(x, self.mask[:x.shape[0]])

    def to_pdb(self, x, out_dir, prefix="sample"):
        from proteinfoundation.utils.ff_utils.pdb_utils import write_prot_to_pdb
        os.makedirs(out_dir, exist_ok=True)
        atom37 = self.model.samples_to_atom37(x.detach().cpu())     # -> Angstrom, CA only
        return [write_prot_to_pdb(atom37[i].numpy(),
                                  os.path.join(out_dir, f"{prefix}_{i}.pdb"),
                                  overwrite=True, no_indexing=True)
                for i in range(atom37.shape[0])]



## Helper functions for proteina
def radius_of_gyration(theta, mask):
    """[N,n,3] nm, [N,n] bool -> [N] nm. Differentiable, SE(3) invariant."""
    from proteinfoundation.utils.align_utils.align_utils import mean_w_mask
    c = theta - mean_w_mask(theta, mask, keepdim=True)
    m = mask[..., None].to(theta.dtype)
    return ((c ** 2 * m).sum(dim=(-1, -2)) / m.sum(dim=(-1, -2))).sqrt()


def ca_ca_spacing(x):
    """[N,n,3] nm -> [N] mean consecutive CA-CA distance in ANGSTROM (the geometry guard)."""
    return (x[:, 1:] - x[:, :-1]).norm(dim=-1).mean(dim=-1) * 10.0


def soft_bin_pairwise_distances(x, min_dist, max_dist, dim):
    d = torch.cdist(x, x)
    limits = torch.linspace(min_dist, max_dist, dim - 1, device=x.device, dtype=x.dtype)
    w = limits[1] - limits[0]
    centers = torch.cat([limits[:1] - w / 2, (limits[:-1] + limits[1:]) / 2, limits[-1:] + w / 2])
    u = (d.clamp(float(centers[0]), float(centers[-1]))[..., None] - centers) / w
    hat = (1.0 - u.abs()).clamp_min(0.0)
    return hat.to(torch.get_default_dtype())


@contextlib.contextmanager
def soft_binning():
    """Context manager to monkey-patch Proteina's hard binning with a differentiable version.
    This is needed for the reward to be differentiable, and is only used in the toy experiments.
    """
    from proteinfoundation.nn import feature_factory as ff
    hard = ff.bin_pairwise_distances
    ff.bin_pairwise_distances = soft_bin_pairwise_distances
    try:
        yield
    finally:
        ff.bin_pairwise_distances = hard
