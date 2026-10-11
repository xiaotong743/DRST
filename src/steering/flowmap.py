"""Strong stochastic flow maps (arXiv:2606.01086), in torch.

The map is an Euler-Maruyama step over an arbitrary interval that takes the REALISED path as
input, so it reproduces the SDE's trajectory for a given Brownian motion rather than only its
marginal. Structure follows their SDE experiment, experiments/sde/sde.py:85-127 (the step model
and map) and :144-276 (both losses). NOT ssfm/losses.py, which wraps every loss in a learned
uncertainty weighting exp(-u)*raw + u and draws its coefficients from diffrax -- we have neither.
The Brownian path is ours because diffrax is not installed.
"""
import torch
from torch import nn

# Chen relations for combining two h/2 Levy vectors into one h vector, from
# ssfm experiments/sde/sde.py:32-40. Exact, so the coefficients carry no discretisation error.
_C = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                   [-1 / 2, 1 / 2, 0.0, 0.0],
                   [0.0, -3 / 4, 1 / 4, 0.0],
                   [1 / 8, 3 / 8, -5 / 8, 1 / 8]], dtype=torch.float64)
_SIGN = torch.tensor([[(-1.0) ** (n + m) for m in range(4)] for n in range(4)],
                     dtype=torch.float64)


def sample_levy(h, n_levy=3, gen=None):
    """I^(n) ~ N(0, h/(2n+1)) for a per-sample h [N,1]. ssfm sde.py:43-49 `sample_L`."""
    return [(h / (2 * n + 1)).sqrt() * torch.randn(h.shape, generator=gen, dtype=h.dtype,
                                                   device=h.device) for n in range(n_levy)]


def chen_combine(left, right):
    """Two h/2 Levy vectors -> the h vector on their union. ssfm sde.py:52-57.

    Lets training sample ARBITRARY (s, h): draw the two halves, combine for the parent. Only the
    evaluation, which needs one path across many intervals, requires the full pyramid.
    """
    n = len(left)
    c = _C[:n, :n].to(left[0])
    cs = (_C[:n, :n] * _SIGN[:n, :n]).to(left[0])      # already C*SIGN; do not multiply by C again
    return [sum(c[a, b] * left[b] + cs[a, b] * right[b] for b in range(n)) for a in range(n)]


class BrownianPath:
    """A dyadic pyramid of Levy coefficients for one Brownian path.

    Thm 3.4 gives I^(n)_{s,t} ~ N(0, (t-s)/(2n+1)), so the finest level is sampled directly and
    coarser levels follow by Chen relations -- exactly as `sample_L` and `aggregate_to` do.
    Integrating Legendre polynomials against fine increments instead would leave an O(1/M) error
    in the coefficients that does not shrink with the evaluation step, flooring the strong error.
    That reasoning is OURS -- their repo never does it, so there is nothing to cite for it.

    Level 0 is the finest (2**levels intervals); level `levels` is the whole span.
    """

    def __init__(self, shape, levels, n_levy=3, T=1.0, gen=None, device=None,
                 dtype=torch.float64):
        self.levels, self.T, self.n_levy = levels, T, n_levy
        n = 2 ** levels
        h = T / n
        std = (h / (2 * torch.arange(n_levy, dtype=dtype, device=device) + 1)).sqrt()
        L = (torch.randn((n, n_levy) + tuple(shape), generator=gen, device=device, dtype=dtype)
             * std.reshape(1, -1, *([1] * len(shape))))
        self.L = [L]
        while L.shape[0] > 1:
            L = self._chen(L)
            self.L.append(L)

    def _chen(self, L):
        n = self.n_levy
        c = _C[:n, :n].to(L)
        cs = (_C[:n, :n] * _SIGN[:n, :n]).to(L)
        left, right = L[0::2], L[1::2]
        return (torch.einsum("nm,im...->in...", c, left)
                + torch.einsum("nm,im...->in...", cs, right))

    def span(self, level, idx):
        """(s, t) of dyadic interval `idx` at `level`."""
        w = self.T * 2 ** level / 2 ** self.levels
        return idx * w, (idx + 1) * w

    def levy(self, level, idx):
        """[I^(0), ..., I^(n_levy-1)] on that interval."""
        return list(self.L[level][idx])


class AbstractEMStepModel(nn.Module):
    def drift(self, ys, s, t, I):
        raise NotImplementedError

    def diffusion(self, s, t, I):
        raise NotImplementedError


class EulerMaruyamaFlowMap(nn.Module):
    """Psi_{s,t}(X_s, I) = X_s + f(X_s, s, t, I)(t-s) + g(s, t, I) I^(0)."""

    def __init__(self, step_model):
        super().__init__()
        self.step_model = step_model

    def forward(self, ys, s, t, I):
        return (ys
                + (t - s) * self.step_model.drift(ys, s, t, I)
                + I[0] * self.step_model.diffusion(s, t, I))


class MLPStep(AbstractEMStepModel):
    """Both heads see the path, as in Def 3.1's Psi; only the drift head sees the state."""

    def __init__(self, dim, n_levy=3, width=256, depth=3, dt_feat=True):
        super().__init__()
        self.n_levy, self.dt_feat = n_levy, dt_feat
        cond = (3 if dt_feat else 2) + n_levy * dim
        mlp = lambda i, o: nn.Sequential(*sum(
            ([nn.Linear(i if k == 0 else width, width), nn.SiLU()] for k in range(depth)),
            []), nn.Linear(width, o))
        self.f = mlp(dim + cond, dim)
        self.g = mlp(cond, dim)

    def _cond(self, s, t, I):
        base = [s, t, t - s] if self.dt_feat else [s, t]
        return torch.cat(base + [i.flatten(1) for i in I], -1)

    def drift(self, ys, s, t, I):
        return self.f(torch.cat([ys.flatten(1), self._cond(s, t, I)], -1)).view_as(ys)

    def diffusion(self, s, t, I):
        return self.g(self._cond(s, t, I)).view_as(I[0])


def _sq(pred, target, h):
    """Per-sample sum over data dims, divided by THAT sample's h, then batch mean.

    sde.py:168 returns `jnp.sum((pred - sg(target))**2) / h` inside a vmap and :190 takes
    `jnp.mean(losses)`. Summing the whole batch instead scales each term by its own sub-batch
    size, which turns the eta=0.75 split into a realised 9:1 rather than 3:1.
    """
    return (((pred - target) ** 2).flatten(1).sum(1) / h.flatten()).mean()


def matching_loss(fm, teacher, ys, s, t, I):
    """Eq 15: one exact Euler-Maruyama step of the teacher's reverse SDE, on the same path.

    Pathwise, not distributional: the target uses I^(0) rather than a fresh draw, so the map is
    asked to land where THIS Brownian increment lands.
    """
    with torch.no_grad():
        _, v = teacher.endpoint(ys, s.reshape(-1))
        # eps returns a python float, so this must loop; .tolist() is ONE device sync rather
        # than one per sample, which on cuda is the difference between usable and unusable.
        g = torch.as_tensor([teacher.eps(a) for a in s.flatten().tolist()],
                            dtype=ys.dtype, device=ys.device).reshape_as(s)
        drift = v + g * teacher.score(ys, v, s)      # per-sample t; s varies down the batch
        target = ys + drift * (t - s) + (2.0 * g).sqrt() * I[0]
    return _sq(fm(ys, s, t, I), target, t - s)


def distill_loss(fm, ema, ys, s, m, t, I_sm, I_mt, I_st):
    """Eq 16: two EMA half-steps teach one student step, all three driven by one path."""
    with torch.no_grad():
        target = ema(ema(ys, s, m, I_sm), m, t, I_mt)
    return _sq(fm(ys, s, t, I_st), target, t - s)
