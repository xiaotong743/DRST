"""Closed-form analytic testbed: a reward whose tilted target is known exactly.

Library code, not an experiment. Three scratch scripts imported this from scratch/, which is why
it now lives here.

The model is the 1-D linear-Gaussian flow x_t = t x_1 + (1-t) x_0 with x_0, x_1 ~ N(0,1), for
which the denoiser, the velocity and the score are all closed form. The reward is mean-matching,
R(pi) = -(lam/2)(mean(pi) - c)^2, chosen because every Algorithm-3 and Algorithm-4 ingredient is
then analytic AND the tilted target has the closed form

    m* = lam c / (1 + lam),

so Algorithm 1 can be scored against an exact number rather than against another implementation.
It is also the reward for which the Picard contraction factor is largest -- rho = lam t^3 / s2(t)
-> lam as t -> 1 -- which makes it the hard case for Algorithm 4 and the clean case for
Algorithm 3, whose direct solve carries no contraction requirement.
"""
import numpy as np

def s2(t):    return (1 - t) ** 2 + t ** 2
def T(x, t):  return t * x / s2(t)
def dTdx(t):  return t / s2(t)
def dTdt(x, t): return x * (1 - 2 * t ** 2) / s2(t) ** 2
def b_drift(x, t): return (2 * t - 1) * x / s2(t)
def score(x, t):   return -x / s2(t)
def softmax(a):
    a = a - a.max(); e = np.exp(a); return e / e.sum()
def ess(w): return 1.0 / np.sum(w ** 2)
def mstar(lam, c=1.0): return lam * c / (1.0 + lam)


def algorithm3(X, w, t, dt, lam, c, dense=True):
    """Returns (psidot_dense, psidot_sherman). Direct solve, no iteration.

    `dense=False` skips the O(N^3) solve and returns (None, psidot_sherman); the two are
    algebraically identical because A = I + u v^T is a rank-one update of the identity.
    """
    q = T(X, t)
    mbar = float(w @ q)
    dq_dx = dTdx(t)
    dq_dt = dTdt(X, t)
    dmbar_dt = float(w @ dq_dt)
    bj = b_drift(X, t)
    # line 4. Both phi and g are RANK ONE in (i,j):
    #   phi[i,j] = -lam t q_i (q_j - mbar),   g[i,j] = -lam t q_i dq_dx   (constant in j)
    # so line 6's two sums collapse to O(N) and no N x N array is needed. Building them densely
    # costs 8 N^2 bytes -- 8 GB at N = 32000, which is what made an earlier sweep time out.
    h   = -lam * t * (mbar - c) * dq_dx             # scalar (same for all j)
    dPsi_dt = -lam * ((mbar - c) * q + t * dmbar_dt * q + t * (mbar - c) * dq_dt)
    # line 6, rank-one form
    s_g   = float(np.dot(w, bj))                    # sum_j w_j b_j
    s_phi = float(np.dot(w * bj, q - mbar))         # sum_j w_j b_j (q_j - mbar)
    r = dPsi_dt + (-lam * t * dq_dx * s_g) * q + (-lam * t * h * s_phi) * q
    # line 5 + 7, dense -- only for the small-N cross-check against the rank-one form
    psidot_dense = None
    if dense:
        phi = -lam * t * np.outer(q, q - mbar)      # [N,N]
        A = np.eye(len(X)) - w[None, :] * phi
        psidot_dense = np.linalg.solve(A, r)
    # line 5 + 7 via Sherman-Morrison: A = I + u v^T, u = lam t q, v = w*(q-mbar)
    u = lam * t * q
    v = w * (q - mbar)
    # 1 + v.u = 1 + lam t Var_w(T) > 0 always, so A is never singular -- the paper's claim that
    # the implicit solver "worked reliably in all cases" is structural for this reward.
    psidot_sm = r - u * (v @ r) / (1.0 + v @ u)
    return psidot_dense, psidot_sm


def algorithm4(X, A_t, Xn, t, dt, lam, c, K_FP, damp):
    """Algorithm 4, eq (20) form, for the same reward. Returns (psidot, a_next)."""
    q  = T(X, t)                       # query endpoints, held at time t
    qn = T(Xn, t + dt)                 # next-step support endpoints (line 5)
    w_t = softmax(A_t)
    mbar_t = float(w_t @ q)
    psi_t = -lam * (mbar_t - c) * q                       # psi(theta_i, pihat_t), no alpha
    gradPsi = -lam * t * (mbar_t - c) * dTdx(t)
    bgrad = b_drift(X, t) * gradPsi
    # the model-time term alpha(t) d/ds psi(xhat1(s,x), pihat_t), measure detached
    time_term = -lam * t * (mbar_t - c) * dTdt(X, t)
    a = A_t.copy(); psidot = np.zeros_like(X)
    for _ in range(K_FP):
        w_k = softmax(a)
        mbar_next = float(w_k @ qn)
        psi_next = -lam * (mbar_next - c) * q             # query fixed, measure advanced
        # psi already carries lam (it is delta R/delta pi and lam sits inside R), so the
        # alpha(t) finite difference must NOT multiply by lam again.
        psidot = ((t + dt) * psi_next - t * psi_t) / dt + time_term
        a_tilde = A_t + dt * (psidot + bgrad)
        a = (1 - damp) * a + damp * a_tilde
    return psidot, a


def algorithm1(N, nsteps, lam, c=1.0, eps=0.5, solver="alg3", K_FP=3, damp=1.0, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(N)
    A_t = np.zeros(N)
    ts = np.linspace(0, 1, nsteps + 1)
    for k in range(nsteps):
        t, dt = ts[k], ts[k + 1] - ts[k]
        w = softmax(A_t)
        q = T(X, t); mbar = float(w @ q)
        gradPsi = -lam * t * (mbar - c) * dTdx(t)
        drift = b_drift(X, t) + eps * score(X, t) + eps * gradPsi
        Xn = X + drift * dt + np.sqrt(2 * eps * dt) * rng.standard_normal(N)
        if solver == "alg3":
            # Sherman-Morrison is exact here (A is I plus a rank-one term) and is O(N); check it
            # against the O(N^3) dense solve on a few steps only.
            pd, psm = algorithm3(X, w, t, dt, lam, c, dense=(k % 100 == 0))
            if pd is not None:
                assert np.abs(pd - psm).max() < 1e-8 * max(1.0, np.abs(pd).max()), "SM != dense"
            A_t = A_t + dt * (psm + b_drift(X, t) * gradPsi)
        else:
            _, A_t = algorithm4(X, A_t, Xn, t, dt, lam, c, K_FP, damp)
        X = Xn
    w = softmax(A_t)
    return dict(wmean=float(w @ X), ess=float(ess(w)), w=w, X=X)


