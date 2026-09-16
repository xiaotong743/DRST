"""Interacting-particle bookkeeping: ESS, resampling, and the Algorithm-4 weight solver.

Log-weights `a` are the paper's A_t (unnormalised). Everything here is pure tensor algebra:
`picard` must never evaluate the model.
"""

import torch


def normalise(a):
    """log-weights [N] -> normalised weights [N]."""
    return torch.softmax(a.double(), dim=0)


def ess(a):
    """Effective sample size from log-weights, exp(2*lse(a) - lse(2a)). Returns a float."""
    a = a.double()
    return float(torch.exp(2 * torch.logsumexp(a, 0) - torch.logsumexp(2 * a, 0)))


def systematic_resample(w, generator=None, num=None):
    """w [N] normalised -> ancestor indices [num] int64 (num defaults to N). One uniform draw.

    `num` must be the number of draws wanted: taking `num` strata out of N and truncating
    would bias the result toward low indices.
    """
    n = w.shape[0]
    num = n if num is None else num
    u = torch.rand((), generator=generator, dtype=torch.float64, device=w.device)
    positions = (torch.arange(num, dtype=torch.float64, device=w.device) + u) / num
    cdf = w.double().cumsum(0).contiguous()
    return torch.searchsorted(cdf, positions.contiguous()).clamp_(max=n - 1)


def residual_resample(w, generator=None):
    """Residual resampling: floor(N*w) deterministic copies, remainder drawn systematically.

    With exactly uniform weights this returns arange(N) and draws no randomness, which is what
    makes a lambda=0 run bit-identical to the unsteered sampler even with resampling enabled.
    """
    n = w.shape[0]
    w = w.double()
    counts = torch.floor(n * w)
    n_det = int(counts.sum().item())
    idx = torch.repeat_interleave(torch.arange(n, device=w.device), counts.long())
    n_res = n - n_det
    if n_res > 0:
        resid = n * w - counts
        resid = resid / resid.sum()
        idx = torch.cat([idx, systematic_resample(resid, generator, num=n_res)])
    return idx[:n].long()


def picard(a0, psi_t, psi_of_weights, time_term, bgrad, t, dt, lam,
           k_fp=50, damp=1.0, tol=1e-9, increment_clip=None, combine=None):
    """Algorithm 4: damped fixed point on the log-weights.

    a0          [N]  log-weights A_t
    psi_t       [N]  psi(theta_i, pihat_t)          (no lambda, no alpha)
    psi_of_weights(w) -> [N]  psi(theta_i, pihat(w)) for candidate weights w.
                              For a pointwise reward this ignores w, so the map is constant
                              in k and the fixed point is reached in one step at damp=1.
    time_term   [N]  alpha(t) * dpsi/ds through the model's time argument, i.e.
                     (dPsi/dt from autograd) - lam*psi_t.  The finite difference below
                     supplies the alpha'(t)*psi and measure-advance terms.
    bgrad       [N]  b_t . grad_x Psi_t, with b = v alone (no eps*score term)
    combine     optional a -> weights handed to psi_of_weights. Default normalise(a), i.e. the
                current batch alone. Algorithm 2 passes a version that concatenates the retained
                terminal log-weights and normalises over BOTH sets; that has to happen on the
                LOG-weights, because normalising the current block first would absorb its mass and
                silently change the two sets' relative weighting.
    Returns (a, w, k_used, resid, history, psidot). The paper's line 11 sets
    psidot = psidot^(K_FP - 1) while w = w^(K_FP), i.e. the returned psidot is one iteration
    behind the returned weights; that is reproduced here. Nothing in Algorithm 1 consumes
    psidot -- it is returned so that the two solvers can be compared inside a real run rather
    than only in a scratch script.
    """
    if k_fp < 1:
        raise ValueError(
            f"k_fp={k_fp}: the loop would not execute, so the weights would stay at A_t and the "
            f"run would silently become the paper's gradient-steering baseline (eq 11) while "
            f"still reporting reweight=True. Pass k_fp >= 1, or reweight=False to mean it.")
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


def oscillation_sign(history_vecs):
    """Given successive increments d^k = a^{k+1}-a^k, report the fraction of steps where
    d^{k+1}.d^k < 0. Persistent negativity means the undamped map is not contractive."""
    if len(history_vecs) < 2:
        return 0.0
    flips = sum(1 for p, q in zip(history_vecs, history_vecs[1:]) if float(p @ q) < 0)
    return flips / (len(history_vecs) - 1)
