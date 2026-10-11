"""
Rewards and first variations
"""

import math

import torch


def rbf_gram(a, b, h):
    d2 = (a[:, None, :] - b[None, :, :]).pow(2).sum(-1)
    return torch.exp(-d2 / (h * h))


def mmd2(x, y, h, wx=None, wy=None):
    wx = _uniform(x, wx)
    wy = _uniform(y, wy)
    kxx = rbf_gram(x, x, h)
    kyy = rbf_gram(y, y, h)
    kxy = rbf_gram(x, y, h)
    return wx @ kxx @ wx + wy @ kyy @ wy - 2.0 * (wx @ kxy @ wy)


def psi(q, z, w, y, h):
    """
    First variation of R(mu) = -MMD^2(mu, nu)
    """
    w = _uniform(z, w)
    cross = rbf_gram(q, y, h).mean(dim=1)     # E_{y~nu} k(q, y)
    inner = rbf_gram(q, z, h) @ w             # E_{z~mu} k(q, z)
    return 2.0 * (cross - inner)


def _uniform(x, w):
    if w is None:
        return x.new_full((x.shape[0],), 1.0 / x.shape[0])
    return w / w.sum()


_mmd_psi = psi          # MMD.psi would otherwise read as self-recursion


class Pointwise:
    """psi(theta, mu) = r(theta)"""

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
        fq = self.features(query)
        fs = self.features(support)
        cross = rbf_gram(fq, self.target, self.h).mean(dim=1)     # [N], fixed
        gram = rbf_gram(fq, fs, self.h)                           # [N, M], fixed
        return lambda w: 2.0 * (cross - gram @ w.to(fq.dtype))

    def phi_tilde(self, query, support, w):
        k = rbf_gram(self.features(query), self.features(support), self.h)
        return -2.0 * (k - (k @ w.to(k.dtype))[:, None])

    def phi_sum(self, query, support, w):
        k = rbf_gram(self.features(query), self.features(support), self.h)
        return -2.0 * (k @ w.to(k.dtype))



def _norm(w, ref):
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
    with torch.no_grad():
        f = features(theta).double()
        p, d = f.shape
        block = max(1, int(2 ** 27 // max(p * d, 1)))          # ~1 GB of float64 per block
        vals = []
        eye = torch.eye(p, dtype=torch.bool, device=f.device)
        for lo in range(0, p, block):
            hi = min(lo + block, p)
            d2 = (f[lo:hi, None, :] - f[None, :, :]).pow(2).sum(-1)
            vals.append(d2[~eye[lo:hi]])
        off = torch.cat(vals)
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


class ClusterDiversity:

    def __init__(self, labels):
        self.labels = labels

    def _co(self, theta):
        lab = torch.as_tensor(self.labels(theta))
        return (lab[:, None] == lab[None, :]).to(theta)   # labels come back on the CPU

    def psi(self, query, support, w):
        s = self._co(support.detach())
        return -2.0 * (s @ _norm(w.detach(), s))

    def weights_fn(self, query, support):
        """One labels() call; the closure is a matvec, so _solve iterating costs nothing."""
        s = self._co(support.detach())
        return lambda w: -2.0 * (s @ _norm(w, s))

    def value(self, theta, w):
        s = self._co(theta)
        w = _norm(w, s)
        return -(w @ s @ w)

    def value_offdiag(self, theta, w):
        s = self._co(theta)
        w = _norm(w, s)
        diag = (w * w).sum()
        return -((w @ s @ w) - diag) / (1.0 - diag)

    def phi_tilde(self, query, support, w):
        raise NotImplementedError("no verified second variation; use psidot_solver='picard'")

    phi_sum = phi_tilde




class TMDiversity(ClusterDiversity):
    """Proteina Table 1's TM-Sc. as a steerable functional: R(mu) = -int int TM(x,y) dmu dmu.

    TM-Sc. is the MEAN PAIRWISE TM-score and is reported lower-is-better, so maximising R
    minimises it. No clustering and no threshold, unlike ClusterDiversity, whose single-linkage
    partition at TM > 0.5 is our own construction rather than the paper's metric.
    """

    def __init__(self, tm):
        self.labels = tm          # theta -> [N, N] TM matrix

    def _co(self, theta):
        return torch.as_tensor(self.labels(theta)).to(theta)