"""The mean-field log-weight update, and the two ways to solve it.

Algorithm 4 (`picard`) is a damped fixed point; Algorithm 3 (`implicit_psidot`) is the direct
solve of eq (13) and carries no contraction requirement. They target the same Psidot, so a run
can be scored against the other solver. Both need the reward's first variation -- FK needs
neither.
"""

import torch

from .smc import normalise

def picard(a0, psi_t, psi_of_weights, time_term, bgrad, t, dt, lam,
           k_fp=50, damp=1.0, tol=1e-9, increment_clip=None, combine=None):
    """Algorithm 4: damped fixed point on the log-weights.

    Returns (a, w, k_used, resid, history, psidot). `psidot` is one iteration behind `w`, as the
    paper's line 11 has it. For a pointwise reward `psi_of_weights` ignores w, so the fixed point
    is reached in one step at damp=1.
    """
    a = a0.clone()
    w = normalise(a)
    history = []
    resid = float("inf")
    k_used = 0
    psidot = torch.zeros_like(a0)
    for k in range(k_fp):
        psi_next = psi_of_weights(combine(a) if combine is not None else w)
        psidot = lam * ((t + dt) * psi_next - t * psi_t) / dt + time_term
        inc = dt * (psidot + bgrad)
        if increment_clip is not None:
            inc = inc.clamp(-increment_clip, increment_clip)
        a_tilde = a0 + inc
        a_new = (1.0 - damp) * a + damp * a_tilde
        resid = float((a_new - a).abs().max())
        history.append(resid)
        a, w = a_new, normalise(a_new)
        k_used = k + 1
        if resid < tol:
            break
    return a, w, k_used, resid, history, psidot


# --- Algorithm 3: the direct solve of (I - w*Phitilde) psidot = r, eq (13)


def implicit_psidot(phi, w, dpsi_dt, phi_grad_b, bgrad):
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
