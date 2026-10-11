"""The flow-map student for Proteina: a stochastic exponential integrator in schedule time u.

Proteina's reverse SDE is LINEAR in x once v is given. `vf_to_score` is
score = (t*v - x)/(1-t) (r3n_fm.py:355-364, scale_ref == 1 as models/proteina.py asserts), so

    dx = [a(t) v(x,t) - b(t) x] dt + sqrt(2 g(t) gamma) dW,
    a = 1 + g t/(1-t),   b = g/(1-t),   g = 1/(t + 1e-2)

and freezing v over a step leaves an Ornstein-Uhlenbeck step with a closed form. Pushing into u,
where the model's own 400-point `log` grid is uniform, the map is

    Psi(y, s, t, I) = E y + P vhat + S0 I0 + S1 I1 + S2 I2

-- five SCALARS multiplying equivariant tensors, so the map is exactly O(3)-equivariant, exactly
zero-CoM preserving and L-agnostic without a transformer. The learned part is a zero-initialised
correction to those five, so an untrained student IS the analytic integrator `init` names.

Why not SSFM's MLPStep flattened to L*3: b(0)*TAU'(0) = 465.17, so an ADDITIVE-drift student is
linearly unstable for any step count below 233, and the whole point is 1-64 steps.
"""
import math

import torch
from torch import nn

from .flowmap import AbstractEMStepModel

LN10 = math.log(10.0)
# get_schedule(mode="log", p1=2.0) in closed form (r3n_fm.py:646-653): matched to the model's own
# 400-point grid to 7.5e-08 in float32, so training in u needs no interpolation.
TAU = lambda u: (1.0 - 10.0 ** (-2.0 * u)) / (1.0 - 1e-2)
DTAU = lambda u: 2.0 * LN10 * 10.0 ** (-2.0 * u) / (1.0 - 1e-2)
# get_gt at gt_mode "1/t", gt_p 1.0, clamp None, AND Proteina's own ODE tail: r3n_fm.py:523 sets
# sampling_mode = "vf" for t > 0.99, which is g = 0.
GT = lambda t: torch.where(t > 0.99, torch.zeros_like(t),
                           1.0 / (t.clamp(0.0, 1.0 - 1e-5) + 1e-2))


def abs_u(u, gam):
    """(A, B, Sigma) of dy/du = A vhat - B y + Sigma dW_u: Proteina's SDE pushed into u."""
    t, d = TAU(u), DTAU(u)
    g = GT(t)
    return (1.0 + g * t / (1.0 - t)) * d, g / (1.0 - t) * d, (2.0 * g * gam * d).sqrt()


def coeffs_em(s, h, gam):
    """Plain Euler-Maruyama. B(0) = 465.17, so this is the baseline that must blow up at low K."""
    A, B, S = abs_u(s, gam)
    return 1.0 - h * B, h * A, S, torch.zeros_like(S), torch.zeros_like(S)


def coeffs_exp(s, h, gam, q=256):
    """The exact OU step with vhat frozen, by midpoint quadrature.

    S_n projects the diffusion onto shifted Legendre polynomials, matching sample_levy's
    I^(n) ~ N(0, h/(2n+1)) (flowmap.py:21-24). E = exp(-int B) is in (0, 1], so unlike `em` this
    is unconditionally stable at every step count, down to one step.

    q is 256 because S2 sets it, not E: **[measured]** at (s, h) = (0.1, 0.05) the coefficients
    reach their q=4096 values by q=64 except S2, which is still 2.2x off there and only settles
    by 256. S2 is the smallest of the five, so the rule's absolute error is large relative to it
    while negligible for the rest. The cost is a [n,1,1,q] tensor beside a 200M-param forward.

    Steps crossing u = 0.8506 straddle the ODE tail's jump in g, where midpoint is O(1/q) rather
    than O(1/q^2); the baseline is mildly wrong there by construction and the head learns it.
    """
    z = (torch.arange(q, dtype=s.dtype, device=s.device) + 0.5) / q
    A, B, S = abs_u(s[..., None] + h[..., None] * z, gam)
    w = torch.exp(-h[..., None] * (B.flip(-1).cumsum(-1).flip(-1) - 0.5 * B) / q)
    return (torch.exp(-h * B.mean(-1)), h * (w * A).mean(-1), (w * S).mean(-1),
            3.0 * (w * S * (2 * z - 1)).mean(-1),
            5.0 * (w * S * (1.5 * (2 * z - 1) ** 2 - 0.5)).mean(-1))


BASE = {"em": coeffs_em, "exp": coeffs_exp}


class ExpStep(AbstractEMStepModel):
    """Five scalar corrections on a FROZEN trunk; zero-init, so untrained == the `init` baseline."""

    def __init__(self, vfield, init="exp", n_levy=3, width=128, depth=3, gamma=1.0):
        super().__init__()
        self.vf, self.base, self.gamma = vfield, BASE[init], gamma
        self.head = nn.Sequential(*sum(
            ([nn.Linear(3 if k == 0 else width, width), nn.SiLU()] for k in range(depth)),
            []), nn.Linear(width, 5))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def coeffs(self, s, t):
        """The five scalars: analytic baseline plus a learned correction in the baseline's units."""
        h = t - s
        base = self.base(s, h, self.gamma)
        c = self.head(torch.cat([s, t, h], -1).flatten(1)).view(-1, 5, 1, 1).unbind(1)
        # corrections carried in units of (1, h, Sigma, Sigma, Sigma): Sigma spans 0.07 to 20, so
        # an unscaled additive head would be badly conditioned across the schedule
        unit = (torch.ones_like(h), h, base[2], base[2], base[2])
        return [b + d * w for b, d, w in zip(base, c, unit)]

    def drift(self, ys, s, t, I):
        E, P, _, S1, S2 = self.coeffs(s, t)
        return ((E - 1.0) * ys + P * self.vf(ys, s) + S1 * I[1] + S2 * I[2]) / (t - s)

    def diffusion(self, s, t, I):
        return self.coeffs(s, t)[2]


class VField:
    """The frozen trunk's v, memoised on the QUERY TIME tensor's identity.

    A training step queries exactly two (state, time) pairs, (y,u) and (y_m,m), and hands the same
    two time tensors to every map call, so the key suffices and the step costs ONE new forward.

    Each entry KEEPS a reference to its time tensor. Without that the key is unsound rather than
    merely imprecise: **[measured]** 199 of 200 freshly built per-step tensors land on the same
    address, so a freed key silently matches a later, unrelated tensor and returns the wrong v.
    The caller still clears the memo each step -- that is what caps the cost at one forward.
    """

    def __init__(self, mdl):
        self.mdl = mdl
        self.memo = {}

    def fresh(self, ys, s):
        with torch.no_grad():
            return self.mdl.endpoint(ys, TAU(s).reshape(-1))[1]

    def seed(self, s, v):
        """Start a step with the chain's carried v. Clearing first is what caps the cost at one."""
        self.memo = {s.data_ptr(): (s, v)}

    def __call__(self, ys, s):
        hit = self.memo.get(s.data_ptr())
        if hit is None:
            self.memo[s.data_ptr()] = (s, self.fresh(ys, s))    # the entry HOLDS s alive, so its
            hit = self.memo[s.data_ptr()]                       # address cannot be recycled under
        return hit[1]                                           # a later tensor while it is cached
