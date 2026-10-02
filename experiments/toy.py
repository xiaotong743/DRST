"""1-D Gaussian-mixture flow model with an exact denoiser, and the paper's section 4.1 setup.

Interpolant x_t = (1-t) x0 + t x1 with x0 ~ N(0,1) and x1 ~ sum_k pi_k N(m_k, s_k^2). Then

    x_t | x1 ~ N(t x1, (1-t)^2)
    p_t      = sum_k pi_k N(t m_k, t^2 s_k^2 + (1-t)^2)
    xhat1(x,t) = sum_k w_k(x,t) [ m_k + t s_k^2 / var_k(t) * (x - t m_k) ],
                 w_k ∝ pi_k N(x; t m_k, var_k(t)),  var_k(t) = t^2 s_k^2 + (1-t)^2

so nothing is trained and mu* can be obtained on a grid to numerical precision.

Paper section 4.1 / C.1: base weights (1,3) at means (-1,1), unit variances; target nu the same
means with weights (3,1); mu* = argmin_mu {KL(mu||p1) + lam* MMD^2(mu,nu)}, lam* = 10.
"""

import math

import torch


class GaussianMixtureFlow:
    """Exact flow model for a 1-D Gaussian mixture target. Event shape is [1]."""

    def __init__(self, means=(-1.0, 1.0), stds=(1.0, 1.0), weights=(1.0, 3.0),
                 eps_mode="memoryless", eps_clamp=None, dtype=torch.float64):
        self.m = torch.tensor(means, dtype=dtype)
        self.s = torch.tensor(stds, dtype=dtype)
        p = torch.tensor(weights, dtype=dtype)
        self.pi = p / p.sum()
        self.eps_mode, self.eps_clamp, self.dtype = eps_mode, eps_clamp, dtype

    # --- model contract ---------------------------------------------------
    def prior(self, n, generator=None):
        return torch.randn(n, 1, generator=generator, dtype=self.dtype)

    def endpoint(self, x, t):
        """x [N,1], t [N] -> (xhat1 [N,1], v [N,1]); differentiable in both."""
        tt = t.unsqueeze(-1)                                    # [N,1]
        var = tt ** 2 * self.s ** 2 + (1 - tt) ** 2             # [N,K]
        logw = (torch.log(self.pi) - 0.5 * torch.log(2 * math.pi * var)
                - (x - tt * self.m) ** 2 / (2 * var))           # [N,K]
        wk = torch.softmax(logw, dim=-1)
        post = self.m + (tt * self.s ** 2 / var) * (x - tt * self.m)
        theta = (wk * post).sum(-1, keepdim=True)               # [N,1]
        v = (theta - x) / (1 - tt)
        return theta, v

    def score(self, x, v, t):
        return (t * v - x) / (1 - t)

    def eps(self, t):
        """eps_t, related to the paper's quoted sigma_t by sigma_t^2 = 2 eps_t.

        The paper lists sigma_t in {1, sqrt(1-t), sqrt(2(1-t)/t)}; the third gives
        eps_t = (1-t)/t, the canonical memoryless value, which fixes the convention.
        """
        if self.eps_mode == "const":            # sigma_t = 1
            e = 0.5
        elif self.eps_mode == "sqrt1mt":        # sigma_t = sqrt(1-t)
            e = 0.5 * (1.0 - t)
        elif self.eps_mode == "memoryless":     # sigma_t = sqrt(2(1-t)/t)
            e = (1.0 - t) / t if t > 0 else float("inf")
        elif self.eps_mode == "zero":
            return 0.0
        else:
            raise ValueError(self.eps_mode)
        cap = self.eps_clamp if self.eps_clamp is not None else 50.0
        return float(min(e, cap))

    # --- analytic references ----------------------------------------------
    def log_p1(self, x):
        """x [...] -> log p1(x)."""
        x = x.unsqueeze(-1)
        comp = (torch.log(self.pi) - 0.5 * torch.log(2 * math.pi * self.s ** 2)
                - (x - self.m) ** 2 / (2 * self.s ** 2))
        return torch.logsumexp(comp, dim=-1)

    def sample_p1(self, n, generator=None):
        k = torch.multinomial(self.pi, n, replacement=True, generator=generator)
        return (self.m[k] + self.s[k] * torch.randn(n, generator=generator, dtype=self.dtype))


def grid_tempered(model, tau, grid=None, reward=None, eta=0.0):
    """mu ∝ p1^(1/tau) exp(eta * r) on a grid: experiment B's terminal target for a POINTWISE
    reward, exact up to the grid spacing. Returns (grid, density).

    Only p1 appears, not p_t, because alpha(1) = 1 and the tilt is evaluated at the endpoint --
    so this reference needs no time-marginal and no fixed point, unlike `grid_mu_star`.
    """
    if grid is None:
        grid = torch.linspace(-8, 8, 3201, dtype=model.dtype)
    log_q = model.log_p1(grid) / tau
    if reward is not None and eta != 0.0:
        log_q = log_q + eta * reward(grid.reshape(-1, 1)).reshape(-1)
    return grid, torch.softmax(log_q, dim=0)


def grid_mu_star(model, target, lam, h, grid=None, iters=2000, damp=0.5):
    """Solve mu ∝ p1 exp(Psi(., mu)) on a grid, where Psi = lam * psi and psi is the MMD first
    variation. Returns (grid, density). Provides absolute ground truth for the toy tests."""
    from steering.rewards import rbf_gram
    if grid is None:
        grid = torch.linspace(-6, 6, 1201, dtype=model.dtype)
    logp1 = model.log_p1(grid)
    gg = rbf_gram(grid.unsqueeze(-1), grid.unsqueeze(-1), h)
    gy = rbf_gram(grid.unsqueeze(-1), target, h).mean(dim=1)
    q = torch.softmax(logp1, dim=0)
    for _ in range(iters):
        psi = 2.0 * (gy - gg @ q)
        q_new = torch.softmax(logp1 + lam * psi, dim=0)
        q = (1 - damp) * q + damp * q_new
    return grid, q


def sample_grid(grid, q, n, generator=None):
    """Draw n samples from a grid density (grid [G], q [G] normalised)."""
    idx = torch.multinomial(q, n, replacement=True, generator=generator)
    return grid[idx].reshape(-1, 1)


def objective(samples, weights, model, target, lam, h, kl_bw=0.2):
    """J(mu) = lam * MMD^2(mu, nu) + KL(mu || p1), the paper's objective.

    KL is estimated by a kernel-density plug-in on the weighted particle cloud, matching the
    leave-one-out style estimator the paper used for the low-dimensional experiments.
    """
    from steering.rewards import mmd2, rbf_gram
    s = samples.reshape(-1, 1)
    m2 = mmd2(s, target, h, wx=weights.to(s.dtype))
    k = rbf_gram(s, s, kl_bw * (2 ** 0.5))
    dens = (k * weights.to(s.dtype)).sum(1) / (kl_bw * math.sqrt(2 * math.pi))
    kl = float((weights.to(s.dtype) * (torch.log(dens + 1e-30) - model.log_p1(s.squeeze(-1)))).sum())
    return float(lam * m2) + kl


def w1_to_grid(samples, weights, grid, q):
    """Exact 1-D Wasserstein-1 between a weighted particle set and a grid density.

    W1 = integral |F_mu(x) - F_nu(x)| dx, with both CDFs evaluated on `grid`. Free of any
    bandwidth or density-estimation choice, so it does not bias the argmin over lambda the way
    a KDE plug-in KL does.
    """
    s = samples.reshape(-1)
    order = torch.argsort(s)
    s, w = s[order], weights.reshape(-1)[order].to(s.dtype)
    w = w / w.sum()
    # F_mu on the grid: cumulative weight of samples <= grid point
    cw = torch.cat([torch.zeros(1, dtype=w.dtype), w.cumsum(0)])
    idx = torch.searchsorted(s.contiguous(), grid.contiguous(), right=True)
    F_mu = cw[idx]
    F_nu = q.to(s.dtype).cumsum(0)
    dx = float(grid[1] - grid[0])
    return float((F_mu - F_nu).abs().sum() * dx)


def kl_loo(samples, weights, model, bw=0.2):
    """Leave-one-out KDE estimate of KL(mu || p1), as the paper's low-dim experiments use.

    p_hat(x_i) = sum_{j != i} w_j K_bw(x_i - x_j) / (1 - w_i), then
    KL ~= sum_i w_i [log p_hat(x_i) - log p1(x_i)].
    Bandwidth 0.2 is the paper's value (Appendix C.1), and is the KL ESTIMATOR's bandwidth --
    distinct from the reward kernel's 5.0.
    """
    x = samples.reshape(-1)
    w = weights.reshape(-1).to(x.dtype)
    w = w / w.sum()
    d2 = (x[:, None] - x[None, :]) ** 2
    K = torch.exp(-d2 / (2 * bw * bw)) / (bw * math.sqrt(2 * math.pi))
    num = (K * w[None, :]).sum(1) - K.diagonal() * w      # drop the self term
    dens = num / (1.0 - w).clamp_min(1e-12)
    return float((w * (torch.log(dens.clamp_min(1e-300)) - model.log_p1(x))).sum())


def objective_paper(samples, weights, model, target, lam_star, h_reward, kl_bw=0.2):
    """J(mu) = lam* MMD^2(mu, nu) + KL(mu || p1)  (paper C.1.1)."""
    from steering.rewards import mmd2
    s = samples.reshape(-1, 1)
    w = weights.reshape(-1).to(s.dtype)
    w = w / w.sum()
    return lam_star * float(mmd2(s, target, h_reward, wx=w)) + kl_loo(s, w, model, kl_bw)


class TrainedFlow(torch.nn.Module):
    """A small trained flow-matching model for the 1-D mixture, as the paper uses.

    Same `endpoint(x, t)` contract as GaussianMixtureFlow. Trained on the independent coupling
    x_t = (1-t) x_0 + t x_1 with x_0 ~ N(0,1), x_1 ~ p1, regressing the velocity x_1 - x_0.
    Note (x_1 - x_0) equals (xhat1 - x_t)/(1-t), so the flow-matching velocity IS the xt_dot
    velocity the sampler wants, and xhat1 = x_t + (1-t) v -- the same v-parameterisation Proteina
    uses (target_pred = "v").
    """

    def __init__(self, ref: "GaussianMixtureFlow", width=128, eps_mode="const", eps_clamp=None,
                 dtype=torch.float64):
        super().__init__()
        self.ref, self.eps_mode, self.eps_clamp, self.dtype = ref, eps_mode, eps_clamp, dtype
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, width), torch.nn.SiLU(),
            torch.nn.Linear(width, width), torch.nn.SiLU(),
            torch.nn.Linear(width, width), torch.nn.SiLU(),
            torch.nn.Linear(width, 1)).to(dtype)

    def _feat(self, x, t):
        tt = t.reshape(-1, 1).to(x.dtype)
        return torch.cat([x, tt, torch.cos(math.pi * tt)], dim=-1)

    def fit(self, steps=6000, batch=4096, lr=2e-3, seed=0, verbose=False):
        g = torch.Generator(device=torch.get_default_device()).manual_seed(seed)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        for i in range(steps):
            x1 = self.ref.sample_p1(batch, g).reshape(-1, 1)
            x0 = torch.randn(batch, 1, generator=g, dtype=self.dtype)
            t = torch.rand(batch, generator=g, dtype=self.dtype)
            xt = (1 - t.reshape(-1, 1)) * x0 + t.reshape(-1, 1) * x1
            loss = ((self.net(self._feat(xt, t)) - (x1 - x0)) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if verbose and (i + 1) % 1000 == 0:
                print(f"  step {i+1:5d}  loss {float(loss):.5f}", flush=True)
        for p in self.net.parameters():
            p.requires_grad_(False)
        return self

    # --- model contract (identical to GaussianMixtureFlow) ---------------
    def prior(self, n, generator=None):
        return torch.randn(n, 1, generator=generator, dtype=self.dtype)

    def endpoint(self, x, t):
        with torch.enable_grad():
            v = self.net(self._feat(x, t))
        theta = x + (1 - t.reshape(-1, 1)) * v
        return theta, v

    def score(self, x, v, t):
        return (t * v - x) / (1 - t)

    def eps(self, t):
        return GaussianMixtureFlow.eps(self, t)

    def log_p1(self, x):
        return self.ref.log_p1(x)

    def sample_p1(self, n, generator=None):
        return self.ref.sample_p1(n, generator)
