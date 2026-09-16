"""Algorithm 3: the implicit linear solver for Psidot.

Solves (I - w*Phitilde) psidot = r from Proposition 3.3's eq (13). A direct solve, so unlike
Algorithm 4 it carries no contraction requirement.

Two facts make it tractable: `y` in line 4 is a free point, so the mean-centring term drops out
of grad_y Phitilde; and line 6 needs only the contraction sum_j w_j b_j . g_ij, which is one jvp.
Potentials supply Phi without lam or alpha(t); the caller applies lam * alpha(t).
"""

import torch


def implicit_psidot(phi, w, dpsi_dt, phi_grad_b, bgrad):
    """Lines 5-7. Returns (psidot, residual).

        phi [N,N] Phitilde including lam*alpha(t); dpsi_dt [N] with the pushforward moving with t;
        phi_grad_b [N] the line-6 contraction; bgrad [N] = b_j . grad_y Psi_t, as Algorithm 4 uses.
        """
    n = phi.shape[0]
    r = dpsi_dt + phi_grad_b + phi @ (w * bgrad)
    a_mat = torch.eye(n, dtype=phi.dtype, device=phi.device) - w[None, :] * phi
    psidot = torch.linalg.solve(a_mat, r)
    resid = float((a_mat @ psidot - r).abs().max())
    return psidot, resid


def psidot_terms(model, potential, x, t_val, dt, w, lam, b, bgrad, alpha=None):
    """Assemble line 4's ingredients for an arbitrary model and potential.

    Returns (phi, dpsi_dt, phi_grad_b) ready for `implicit_psidot`.

    alpha(t) defaults to t, the paper's R_t = t R(xhat1(.,t) # mu).
    """
    alpha = (lambda s: s) if alpha is None else alpha
    n = x.shape[0]
    dev, dt_ = x.device, x.dtype

    # --- d_t Psi_t, with BOTH the query endpoint and the measure's pushforward moving with t.
    # A single shared scalar t is differentiated, and the output is an N-vector, so this is a
    # forward-mode derivative: reverse mode would sum over i and mix the particles.
    def psi_of_t(s):
        th, _ = model.endpoint(x, s.expand(n))
        return lam * alpha(s) * potential.psi(th, th, w)

    t_s = torch.as_tensor(float(t_val), dtype=dt_, device=dev)
    _, dpsi_dt = torch.func.jvp(psi_of_t, (t_s,), (torch.ones_like(t_s),))

    with torch.no_grad():
        theta, _ = model.endpoint(x, torch.full((n,), float(t_val), dtype=dt_, device=dev))
    la = lam * alpha(torch.as_tensor(float(t_val), dtype=dt_, device=dev))

    # --- phi = lam alpha(t) * Phitilde at the current cloud
    phi = la * potential.phi_tilde(theta, theta, w)

    # --- sum_j w_j b_j . grad_y Phitilde(X^i, X^j): one jvp along b, query held fixed
    def phi_sum_of_support(y):
        th_s, _ = model.endpoint(y, torch.full((n,), float(t_val), dtype=dt_, device=dev))
        return potential.phi_sum(theta, th_s, w)

    _, phi_grad_b = torch.func.jvp(phi_sum_of_support, (x,), (b,))
    return phi, dpsi_dt.to(phi.dtype), (la * phi_grad_b).to(phi.dtype)
