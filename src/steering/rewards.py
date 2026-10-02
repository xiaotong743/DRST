"""
Rewards and first variations
"""

import math

import torch


def rbf_gram(a, b, h):
    """[n, d], [m, d] -> [n, m] Gaussian kernel matrix.

    The squared distance is formed as a direct difference rather than expanded as
    |a|^2 + |b|^2 - 2a.b, and not taken from torch.cdist. Two reasons:

    * torch.cdist has no forward-mode AD rule, and Algorithm 3 needs a jvp through this kernel
      (implt.psidot_terms), so cdist is unusable here.
    * the expansion cancels catastrophically when d^2 << |a|^2. Measured against a float64 cdist
      reference at |a| ~ 1e2 / 1e4 with nearby points: float64 error 2.9e-11 / 4.8e-7, and
      float32 error 6.1e-2 / 1.0 -- at float32 and large magnitudes the kernel is simply wrong.
      Our features are O(1) nm, so the expansion would have been adequate, but a caller switching
      to Angstrom or to a distance histogram would have crossed into that regime silently.

    Cost: n*m*d intermediate instead of n*m. For the shapes here (d = 1 in the toy, a few hundred
    at protein scale with n <= 48) that is not the binding constraint; the Gram itself is.
    """
    d2 = (a[:, None, :] - b[None, :, :]).pow(2).sum(-1)
    return torch.exp(-d2 / (h * h))


def mmd2(x, y, h, wx=None, wy=None):
    """Squared MMD between weighted empirical measures.

    x: [n, d], y: [m, d]; wx/wy are normalised weight vectors (uniform if None).
    Biased (V-statistic) estimator: the diagonal terms are retained, which is what
    makes `psi` below its exact first variation.
    """
    wx = _uniform(x, wx)
    wy = _uniform(y, wy)
    kxx = rbf_gram(x, x, h)
    kyy = rbf_gram(y, y, h)
    kxy = rbf_gram(x, y, h)
    return wx @ kxx @ wx + wy @ kyy @ wy - 2.0 * (wx @ kxy @ wy)


def psi(q, z, w, y, h):
    """First variation of R(mu) = -MMD^2(mu, nu) at mu = sum_j w_j delta_{z_j}, nu = unif(y).

    q: [n, d] query points, z: [m, d] measure support, w: [m] weights, y: [r, d] target sample.
    Returns [n].
    """
    w = _uniform(z, w)
    cross = rbf_gram(q, y, h).mean(dim=1)     # E_{y~nu} k(q, y)
    inner = rbf_gram(q, z, h) @ w             # E_{z~mu} k(q, z)
    return 2.0 * (cross - inner)


def _uniform(x, w):
    """Uniform weights when none are given, and NORMALISE otherwise.

    Nothing previously enforced the normalisation the docstrings assume, so passing an
    unnormalised vector returned a wrong MMD with no error (measured: 76.2 against 0.017).
    Callers already pass softmax output, so this is a no-op for them and a guard for everyone
    else."""
    if w is None:
        return x.new_full((x.shape[0],), 1.0 / x.shape[0])
    return w / w.sum()


_mmd_psi = psi          # MMD.psi would otherwise read as self-recursion


class Pointwise:
    """psi(theta, mu) = r(theta). No measure dependence, so Algorithm 4 collapses."""

    def __init__(self, r):
        self.r = r

    def psi(self, query, support, w):
        return self.r(query)

    def weights_fn(self, query, support):
        """Constant in w and in support here."""
        val = self.r(query)
        return lambda w: val

    def phi_tilde(self, query, support, w):
        return torch.zeros(query.shape[0], support.shape[0],
                           dtype=query.dtype, device=query.device)

    def phi_sum(self, query, support, w):
        return torch.zeros(query.shape[0], dtype=query.dtype, device=query.device)


class MMD:
    """psi = first variation of R(mu) = -MMD^2(features#mu, nu)."""

    def __init__(self, target, h, features=lambda x: x.reshape(x.shape[0], -1)):
        self.target, self.h, self.features = target, h, features

    def psi(self, query, support, w):
        return _mmd_psi(self.features(query), self.features(support), w.to(query.dtype),
                       self.target, self.h)

    def weights_fn(self, query, support):
        """w -> psi(query, pihat(support, w)), reusing one Gram so Picard is matvecs."""
        fq = self.features(query)
        fs = self.features(support)
        cross = rbf_gram(fq, self.target, self.h).mean(dim=1)     # [N], fixed
        gram = rbf_gram(fq, fs, self.h)                           # [N, M], fixed
        return lambda w: 2.0 * (cross - gram @ w.to(fq.dtype))

    # Algorithm 3's second variation; omits lam * alpha(t), as `psi` does.
    def phi_tilde(self, query, support, w):
        """Mean-centred: Phi(x,y) - int Phi(x,z) dmu(z).  [N, M]"""
        k = rbf_gram(self.features(query), self.features(support), self.h)
        return -2.0 * (k - (k @ w.to(k.dtype))[:, None])

    def phi_sum(self, query, support, w):
        """sum_j w_j Phi(x_i, y_j). Phitilde's centring term has no y-dependence, so it is
        absent here by design."""
        k = rbf_gram(self.features(query), self.features(support), self.h)
        return -2.0 * (k @ w.to(k.dtype))



def _norm(w, ref):
    """Weights onto the Gram's dtype/device, renormalised."""
    w = w.to(ref)
    return w / w.sum()


class SoftContactFeatures:
    """theta [N, L, 3] nm -> [N, |J|] smooth contact map over pairs with j - i >= min_sep.

    d_ij = sqrt(|theta_i - theta_j|^2 + delta^2) stays smooth for coincident points;
    c_ij = sigmoid((r_c - d_ij)/a) is a soft indicator. The sqrt(|J|) divisor makes a squared
    feature distance the MEAN squared contact difference. Translation- and rotation-invariant.
    One fixed mask for the population: a per-particle mask would give different feature spaces.
    """

    def __init__(self, mask, min_sep=4, cutoff_nm=0.8, width_nm=0.05, delta_nm=1e-6):
        mask = torch.as_tensor(mask, dtype=torch.bool).cpu()
        if mask.ndim != 1 or min_sep < 1 or width_nm <= 0 or delta_nm <= 0:
            raise ValueError(
                f"expected one 1-D mask and positive smoothing scales, got mask.ndim="
                f"{mask.ndim}, min_sep={min_sep}, width_nm={width_nm}, delta_nm={delta_nm}")
        self.length = mask.numel()
        i, j = torch.triu_indices(self.length, self.length, offset=min_sep)
        valid = mask[i] & mask[j]
        self.i, self.j = i[valid], j[valid]
        if self.i.numel() == 0:
            raise ValueError(f"no valid pairs at length {self.length} with min_sep={min_sep}")
        self.cutoff, self.width, self.delta = cutoff_nm, width_nm, delta_nm
        self.min_sep = min_sep

    def __call__(self, theta):
        if theta.ndim != 3 or theta.shape[1:] != (self.length, 3):
            raise ValueError(
                f"expected [N, {self.length}, 3] coordinates in nm, got {tuple(theta.shape)}")
        i, j = self.i.to(theta.device), self.j.to(theta.device)
        diff = theta[:, i, :] - theta[:, j, :]
        dist = (diff.square().sum(-1) + self.delta ** 2).sqrt()
        return torch.sigmoid((self.cutoff - dist) / self.width) / math.sqrt(i.numel())


class ContactRepulsion:
    """psi = first variation of R(nu) = -int int k dnu dnu. No reference population needed.

    Returns UNSCALED psi: `steer` applies lam * alpha(t) exactly once.
    """

    def __init__(self, features, h):
        if not math.isfinite(h) or h <= 0:
            raise ValueError(f"bandwidth must be finite and positive, got {h}")
        self.features, self.h = features, float(h)

    def psi(self, query, support, w):
        """[N_q, L, 3], [M, L, 3], [M] -> [N_q]. Only `query` carries the spatial gradient."""
        fq = self.features(query)
        with torch.no_grad():
            fs = self.features(support.detach())
        k = rbf_gram(fq, fs, self.h)
        return -2.0 * (k @ _norm(w.detach(), k))

    def weights_fn(self, query, support):
        """w -> psi(query, pihat(support, w)), reusing one Gram so Picard is matvecs.

        Rebuilt for each step's (query, support) pair; the Gram is only reused WITHIN that
        step's fixed-point solve.
        """
        with torch.no_grad():
            k = rbf_gram(self.features(query.detach()), self.features(support.detach()), self.h)
        return lambda w: -2.0 * (k @ _norm(w, k))

    def value(self, theta, w):
        """R(nu_N) = -w^T K w, the functional the sampler optimises. For logging only: backward
        on it gives w_i times the intended first variation, since it differentiates the support.

        NOT comparable across runs with different ESS. The diagonal's share is exactly
        sum_i w_i^2 = 1/ESS, so a degenerate weighting drives this to -1 whatever the population
        looks like. Use `value_offdiag` to compare.
        """
        k = rbf_gram(self.features(theta), self.features(theta), self.h)
        w = _norm(w, k)
        return -(w @ k @ w)

    def value_offdiag(self, theta, w):
        """`value` with the self-similarity removed and renormalised: the U-statistic analogue,
        comparable across runs with different ESS. Not the objective itself, so report both."""
        k = rbf_gram(self.features(theta), self.features(theta), self.h)
        w = _norm(w, k)
        diag = (w * w).sum()
        return -((w @ k @ w) - diag) / (1.0 - diag)

    # Phi = -2 k(theta, zeta) by the same algebra as MMD's, but left unimplemented rather than
    # assumed: Algorithm 3 also needs a jvp through `features`, which is untested here.
    def phi_tilde(self, query, support, w):
        raise NotImplementedError(
            "ContactRepulsion has no verified second variation, so Algorithm 3 "
            "(psidot_solver='implicit') is not available; use 'picard'.")

    phi_sum = phi_tilde


def calibrate_bandwidth(features, theta):
    """h^2 = median positive squared feature distance over a pilot pool, so median similarity is
    exp(-1). Calibrate ONCE on a native pool and freeze: recomputing h from the current particles
    makes the bandwidth measure-dependent, adding a term to the first variation nothing accounts
    for. Returns (h, diagnostics)."""
    with torch.no_grad():
        f = features(theta).double()
        p, d = f.shape
        # Chunked over rows. The dense [P, P, |J|] difference tensor this used to build costs
        # P^2*|J|*8 bytes to produce a single median: 9.1 GB at P=200 for n_res=250 (|J| = 30381)
        # and 145 GB at the default 8 pilot batches of 100. The row blocks are exact -- verified
        # identical to the dense form -- and bounded by block*P*|J|.
        block = max(1, int(2 ** 27 // max(p * d, 1)))          # ~1 GB of float64 per block
        vals = []
        eye = torch.eye(p, dtype=torch.bool, device=f.device)
        for lo in range(0, p, block):
            hi = min(lo + block, p)
            d2 = (f[lo:hi, None, :] - f[None, :, :]).pow(2).sum(-1)
            vals.append(d2[~eye[lo:hi]])
        off = torch.cat(vals)
        # Relative floor, not `> 0`. Identical structures do NOT give exactly zero distances --
        # **[measured]** 5 copies of one chain leave 40% of the pairs at d2 ~ 1e-35 -- so an
        # exact test let those through and returned h = 3.5e-18, a kernel that is zero on every
        # pair and a reward that is identically flat.
        scale = float(f.pow(2).sum(-1).mean())
        pos = off[off > 1e-12 * max(scale, 1e-300)]
    if pos.numel() == 0:
        raise ValueError(
            f"all {off.numel()} off-diagonal feature distances are negligible against the "
            f"feature scale {scale:.3e} over {theta.shape[0]} pilot structures: the features "
            f"are constant, so the reward has no gradient")
    h = float(pos.median().sqrt())
    return h, dict(n_pilot=int(theta.shape[0]), n_features=int(f.shape[1]), h=h,
                   d2_min=float(pos.min()), d2_median=float(pos.median()),
                   d2_max=float(pos.max()), frac_zero=float((off == 0).double().mean()))
