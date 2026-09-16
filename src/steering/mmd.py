"""Squared MMD and its first variation.

Kernel convention: k(a, b) = exp(-|a - b|^2 / h^2).  A stated "bandwidth" is h.
The other convention in this workspace, exp(-d^2 / (2 s^2)), corresponds to h = sqrt(2) * s.

For R(mu) = -MMD^2(mu, nu) the first variation is

    psi(q, mu) = 2 * [ E_{y~nu} k(q, y) - E_{z~mu} k(q, z) ],

which satisfies the Gateaux identity  d/dh R(mu + h(delta_q - mu))|_{h=0} = psi(q, mu) - E_mu[psi].
"""

import torch


def rbf_gram(a, b, h):
    """[n, d], [m, d] -> [n, m] Gaussian kernel matrix.

    The squared distance is formed as a direct difference rather than expanded as
    |a|^2 + |b|^2 - 2a.b, and not taken from torch.cdist. Two reasons:

    * torch.cdist has no forward-mode AD rule, and Algorithm 3 needs a jvp through this kernel
      (alg3.psidot_terms), so cdist is unusable here.
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
