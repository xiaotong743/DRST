"""Proteina adapter: the frozen model behind steering's endpoint(x, t) contract.

Everything Proteina-specific lives here. Measured constraints it encodes:

- Coordinates are NANOMETRES inside the flow matcher and network (ang_to_nm at proteina.py:112).
- `predict_clean_n_v_w_guidance` returns an UNCENTRED x_pred while computing v from the centred
  one, so v == (theta - x)/(1-t) does not hold as returned (measured ||COM(x_pred)|| = 0.005-0.019
  nm). We centre theta and recompute v.
- `torch.autograd.grad` and `backward(inputs=[...])` both raise RuntimeError against the reentrant
  checkpoint at protein_transformer.py:381/385/389. Plain `.backward()` works and, with frozen
  parameters, allocates no parameter gradients.
- get_gt is only pointwise in t when gt_p == 1 (transform_gt normalises over the whole vector), so
  eps is precomputed on the step grid and looked up.
- The batch dict is exactly {x_t, t, mask}: t must be 1-D [N] and mask must be torch.bool.
  "strict_feats" must not appear.
- DATA_PATH must be set before load: FoldEmbeddingSeqFeat reads
  $DATA_PATH/pdb_raw/cath_label_mapping.pt at construction.
"""

import os

import torch


class ProteinaModel:
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

    # --- model contract ---------------------------------------------------
    def prior(self, n, generator=None):
        """Uses the GLOBAL torch RNG when generator is None, matching sample_reference, so a
        lambda=0 run can be compared bitwise against fm.full_simulation."""
        return self.fm.sample_reference(self.n_res, shape=(n,), device=self.device,
                                        dtype=self.dtype, mask=self.mask[:n])

    def endpoint(self, x, t):
        """x [N,n,3] nm, t [N] -> (theta [N,n,3] centred+masked, v [N,n,3]); differentiable."""
        mask = self.mask[:x.shape[0]]
        batch = {"x_t": x, "t": t, "mask": mask}          # exactly these three keys
        self.n_forward += 1
        x_pred, _ = self.model.predict_clean_n_v_w_guidance(
            batch, guidance_weight=1.0, autoguidance_ratio=0.0)
        # v from the RAW x_pred, exactly as Proteina does it (xt_dot centres both arguments
        # internally). Centring x_pred first and letting xt_dot centre again is equal in exact
        # arithmetic but not in float32, and the difference accumulates over the trajectory.
        v = self.fm.xt_dot(x_pred, x, t, mask)
        # theta is centred separately for the reward; since x is already zero-COM, this keeps
        # v == (theta - x)/(1-t) to float precision.
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

    # --- output -----------------------------------------------------------
    def to_pdb(self, x, out_dir, prefix="sample"):
        from proteinfoundation.utils.ff_utils.pdb_utils import write_prot_to_pdb
        os.makedirs(out_dir, exist_ok=True)
        atom37 = self.model.samples_to_atom37(x.detach().cpu())     # -> Angstrom, CA only
        return [write_prot_to_pdb(atom37[i].numpy(),
                                  os.path.join(out_dir, f"{prefix}_{i}.pdb"),
                                  overwrite=True, no_indexing=True)
                for i in range(atom37.shape[0])]


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
    """Differentiable stand-in for Proteina's `bin_pairwise_distances`.

    **[src]** the shipped version is `bucketize` + `one_hot` (feature_factory.py), so the
    pairwise-distance pair feature contributes EXACTLY zero gradient and Psi_t(x) is discontinuous
    in x. The coordinate path into the trunk (`linear_3d_embed(x_t)`,
    protein_transformer.py:665-670) is differentiable, so steering still works; what the binning
    withholds is the pair-distance contribution.

    This replaces the one-hot with a hat function (linear interpolation onto bin centres): weights
    sum to one for interior values and the map is C^0 in x, so the pair path carries a gradient.

    Patch ONLY bin_pairwise_distances. `rel_seq_sep` also calls `bin_and_one_hot`, with
    half-integer limits on INTEGER separations, so a hat there would split every separation 50/50
    and destroy the model's main inductive bias.

    Adopting this means the frozen model IS the soft-binned model: using hard binning for the
    drift and soft for the gradient makes them different functions.
    """
    d = torch.cdist(x, x)
    limits = torch.linspace(min_dist, max_dist, dim - 1, device=x.device, dtype=x.dtype)
    w = limits[1] - limits[0]
    centers = torch.cat([limits[:1] - w / 2, (limits[:-1] + limits[1:]) / 2, limits[-1:] + w / 2])
    u = (d[..., None] - centers) / w
    return (1.0 - u.abs()).clamp_min(0.0)
