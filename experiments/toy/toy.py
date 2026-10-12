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


import json
import os


class Visual:
    """Records the particle genealogy for fig2's trajectory panels, through Steerer's hooks.

    The drawn set is chosen at t=0: `watch` marks record_traj INITIAL particles and `root` tracks
    which of them each slot descends from, so every branch of a watched ancestor is drawn -- the
    survivors and the ones pruned on the way. Drawing a sample of t=1 winners instead biases the
    t=0 cloud toward the survivors. Nothing here touches x, a or the Brownian stream.
    """

    def begin(self, x, a):
        super().begin(x, a)
        n = self.n
        self._k, self._jumps, self._ghosts = 0, [], []
        self._traj = [x.reshape(n, -1).clone()]
        self._root = torch.arange(n, device=x.device)
        self._watch = torch.zeros(n, dtype=torch.bool, device=x.device)
        g = torch.Generator(device=x.device).manual_seed(int(self.cfg.seed) + 90210)
        self._watch[torch.randperm(n, generator=g)[:self.cfg.record_traj]] = True

    def reindex(self, idx, a, x):
        super().reindex(idx, a, x)
        n = self.n
        if torch.equal(idx, torch.arange(n, device=idx.device)):
            return
        dead = torch.ones(n, dtype=torch.bool, device=idx.device)
        dead[idx] = False                                # no new slot inherits these
        sel = (dead & self._watch[self._root]).nonzero().flatten()
        if sel.numel():                                  # branches that end here, kept whole
            self._ghosts.append((self._k, torch.stack([h[sel] for h in self._traj]
                                                      + [x.reshape(n, -1)[sel]])))
        self._traj = [h[idx] for h in self._traj]        # histories follow their ancestors
        self._root = self._root[idx]
        # the lineage that died in each slot: recorded pre-permutation, then permuted by every
        # later event so it stays aligned with the surviving line it attaches to
        self._jumps = [(j, h[idx]) for j, h in self._jumps]
        self._jumps.append((self._k, x.reshape(n, -1).clone()))

    def on_step(self, x):
        super().on_step(x)
        self._traj.append(x.reshape(self.n, -1).clone())
        self._k += 1

    def extra_output(self):
        keep = self._watch[self._root].nonzero().flatten()   # every surviving watched branch
        return dict(super().extra_output(),
                    traj=torch.stack(self._traj)[:, keep, :].squeeze(-1),
                    jumps=[(j, h[keep].squeeze(-1)) for j, h in self._jumps],
                    ghosts=[(j, h.squeeze(-1)) for j, h in self._ghosts])


SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983"
BLUE, GREEN, ORANGE, BROWN = "#2a78d6", "#008300", "#eb6834", "#9c6644"

RAMP = ["#86b6ef", "#2a78d6", "#104281"]


F3 = [("kfp1", RAMP[0], "o", (0, (4, 2)), r"Mean-field, $K_{fp}=1$"),
      ("kfp3", RAMP[1], "^", (0, (5, 1.5, 1, 1.5)), r"Mean-field, $K_{fp}=3$"),
      ("kfp5", RAMP[2], "s", (0, (1, 1.6)), r"Mean-field, $K_{fp}=5$"),
      ("kfp50", GREEN, "D", "-", r"Mean-field, converged ($K_{fp}=50$, damp .5)"),
      ("implicit", "#7a3fa0", "P", (0, (6, 1.5)), r"Mean-field, Algorithm 3 (implicit)"),
      ("gradient", ORANGE, "o", "-", r"Gradient steering, eq (11): $\varepsilon_t$"),
      ("gradient_p04", BROWN, "v", "-", r"Gradient steering, strength $[0,4]$ (paper's Fig 3)")]

F3_FK = ("fk", "#c81e78", "X", (0, (2, 1)), r"FK potential steering, gradient-free ('diff')")

def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": .8,
                         "figure.facecolor": SURF, "axes.facecolor": SURF,
                         "legend.frameon": False, "axes.spines.top": False,
                         "axes.spines.right": False})
    return plt

def _mix(w, m):
    """`0.95 N(-1) + 0.05 N(+1)`, so the caption says which mixture a variant actually changed."""
    if not w or not m:
        return "?"
    tot = sum(w)
    return " + ".join(f"{x/tot:g} N({mu:+g})" for x, mu in zip(w, m))


def figures(out_path, sd, tag=""):
    import numpy as np
    plt = _style()
    from matplotlib.lines import Line2D

    # ---- Figure 2: histogram strip over trajectories, t on the vertical axis
    f2 = json.load(open(out_path("fig2", tag)))
    grid = np.array(f2["grid"]); ts = np.array(f2["ts"]); dx = grid[1] - grid[0]
    fig = plt.figure(figsize=(12.6, 8.8), facecolor=SURF)
    # two panel-rows, each its own gridspec so the gap BETWEEN rows is wider than the gap
    # between a panel's histogram and its trajectories
    ROWS = [fig.add_gridspec(2, 3, height_ratios=[1, 2.4], hspace=.04, wspace=.12,
                             left=.06, right=.985, top=.92, bottom=.56),
            fig.add_gridspec(2, 3, height_ratios=[1, 2.4], hspace=.04, wspace=.12,
                             left=.06, right=.985, top=.46, bottom=.10)]
    PANELS = [("base", "Base Dynamics"), ("gradient", "Gradient Steering"),
              ("meanfield", "Mean-Field Steering"),
              ("fk_add", "FK potential: 'add'"), ("fk_diff", "FK potential: 'diff'"),
              ("fk_max", "FK potential: 'max'")]
    xl = f2.get("xlim", 4.0)                 # a tilt far from p1 needs a wider frame than +-4
    bins = np.linspace(-xl, xl, int(15 * xl) + 1)
    for i, (key, title) in enumerate(PANELS):
        if f"x_{key}" not in f2:
            continue
        r, c = divmod(i, 3)
        gs = ROWS[r]
        ah = fig.add_subplot(gs[0, c]); at = fig.add_subplot(gs[1, c], sharex=ah)
        for dens, col in ((f2["p1"], BLUE), (f2["nu"], GREEN)):
            ah.hist(np.random.default_rng(0).choice(grid, size=20000, p=np.array(dens)),
                    bins=bins, density=True, color=col, alpha=.55, lw=0)
        x, w = np.array(f2[f"x_{key}"]), np.array(f2[f"w_{key}"])
        ah.hist(x, bins=bins, weights=w / w.sum() / np.diff(bins)[0], color=ORANGE, alpha=.65, lw=0)
        ah.plot(grid, np.array(f2["mu_star"]) / dx, color=INK, lw=1.8)
        ah.set_title(title, color=INK, fontsize=10)
        ah.set_yticks([]); ah.tick_params(labelbottom=False)
        ah.spines["left"].set_visible(False)
        ah.text(.02, .86, f"$W_1$ to $\\mu^*$ = {f2[f'w1_{key}']:.3f}", transform=ah.transAxes,
                fontsize=7.6, color=INK2, va="top")
        tr = np.array(f2[f"traj_{key}"])
        # alpha is set by the watched-root count, so every panel starts with the same ink at t=0.
        # Coalesced lineages overlay exactly, so a duplicated ancestor reads as a darker stroke.
        al = .28 * 120 / f2["record_traj"]
        at.plot(tr, ts[:, None], color=ORANGE, lw=.4, alpha=al)
        # pruned branches, drawn whole in the SAME stroke and stopping dead at the event that
        # killed them. Without these the panel shows only survivors, so its t=0 cloud would be
        # the prior AFTER selection; undistinguished, the cloud is just the particle system.
        for j, gh in f2.get(f"ghost_{key}", []):
            g = np.array(gh)
            at.plot(g, ts[:g.shape[0], None], color=ORANGE, lw=.4, alpha=al)
        # resampling: the horizontal hop from the lineage that died in a slot to the one that
        # replaced it. Same stroke as the paths, so an event reads as the kink it used to be.
        for j, frm in f2.get(f"jump_{key}", []):
            frm, to = np.array(frm), tr[j + 1]
            m = frm != to
            at.hlines(np.full(int(m.sum()), ts[j + 1]), frm[m], to[m],
                      color=ORANGE, lw=.4, alpha=al)
        at.set_xlim(-xl, xl); at.set_ylim(0, 1); at.set_xlabel("x")
        (at.set_ylabel("t") if c == 0 else at.tick_params(labelleft=False))
    fig.legend(handles=[Line2D([], [], color=BLUE, lw=7, alpha=.55),
                        Line2D([], [], color=GREEN, lw=7, alpha=.55),
                        Line2D([], [], color=ORANGE, lw=7, alpha=.65),
                        Line2D([], [], color=INK, lw=1.8)],
               labels=[r"Base Measure $p_1$", r"Tilting Measure $\nu$", r"Generated $\mu$",
                       r"True $\mu^*$"],
               loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(.52, .004))
    fig.suptitle(
        r"Figure 2 reproduced: $\lambda^*=" + f"{f2.get('lam_star', 10):g}"
        + r"$, $\sigma_t=1$, $K_{FP}=3$ undamped, resample every "
        + f"{f2.get('resample_every', 34)} steps"
        + (f" within [{f2['resample_start']}, {f2['resample_end']}]"
           if f2.get("resample_end") is not None else "")
        + ", trained flow "
        r"model, mixture component sd " + f"{sd:g}." + "\n"
        r"Gradient steering is eq (11): $\varepsilon_t\nabla_x\Psi_t$ at the same $\lambda$, "
        r"with no (12b) reweighting.  $W_1$ Monte-Carlo floor "
        + f"{f2['w1_floor']:.3f}."
        + (r"  Variant: $p_1$ = " + _mix(f2.get("p1_weights"), f2.get("means"))
           + r",  $\nu$ = " + _mix(f2.get("nu_weights"), f2["nu_means"])
           if tag else ""), fontsize=8.5, color=INK2)
    fig.savefig(f"fig2_repro{tag}.png", dpi=170, facecolor=SURF)
    print(f"wrote fig2_repro{tag}.png", flush=True)
    if tag:                              # a variant moves nu only, so fig3 is not its to redraw
        return

    # ---- Figure 3: objective gap vs steering strength, one panel per noise schedule
    scheds = [("const", r"$\sigma_t = 1$"), ("sqrt1mt", r"$\sigma_t = \sqrt{1-t}$"),
              ("memoryless", r"$\sigma_t = \sqrt{2(1-t)/t}$")]

    def draw_fig3(series, fname, extra_note=""):
        fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9), constrained_layout=True,
                                 facecolor=SURF)
        for ax, (sch, lab) in zip(axes, scheds):
            d = json.load(open(out_path("fig3", f"_{sch}")))
            lams = np.array(d["lams"])
            ax.axvline(d["lam_star"], color=INK, lw=1.1, ls=(0, (3, 2)))
            for key, col, mk, ls, nm in series:
                if f"J_{key}" not in d:
                    continue
                raw = np.array(d[f"J_{key}"]) - d["J_star"]
                g = np.maximum(raw, 1e-4)
                ax.plot(lams, g, color=col, lw=1.6, ls=ls, marker=mk, ms=4, label=nm)
                neg = raw < 1e-4
                if neg.any():
                    ax.plot(lams[neg], g[neg], ls="none", marker=mk, ms=6, mfc="none",
                            mec=col, mew=1.2)
            ax.set_yscale("log"); ax.set_title(lab, color=INK, fontsize=10)
            ax.set_xlabel(r"Steering strength $\lambda$")
            if ax is axes[0]:
                ax.set_ylabel(r"$J(\mu)-J(\mu^*)$")
        h, l = axes[0].get_legend_handles_labels()
        h += [Line2D([], [], color=INK, lw=1.1, ls=(0, (3, 2)))]; l += [r"True $\lambda^*$"]
        fig.legend(h, l, loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(.5, -.28))
        fig.suptitle(
            r"Figure 3 reproduced. $\lambda^*=10$, reward bandwidth $\sigma=5$ ($h=5\sqrt{2}$), "
            r"mixture component sd " + f"{sd:g}" + r", trained flow model, $N=" + f"{f2['N']}"
            + r"$." "\n"
            r"Gradient steering is eq (11): the SAME steering drift as the mean-field runs at "
            r"every $\lambda$, differing only by (12b) reweighting.  Brown is the same family at "
            r"the paper's display range $[0,4]$." "\n"
            r"Hollow markers sit below the KL-estimator floor.  $J(\mu^*)$ uses the same "
            r"estimator on $N$ draws from a grid-solved $\mu^*$, so its plug-in bias cancels."
            + extra_note,
            fontsize=8.5, color=INK2, y=1.30)
        fig.savefig(fname, dpi=170, facecolor=SURF, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {fname}", flush=True)

    draw_fig3(F3, "fig3_repro.png")
    if any("J_fk" in json.load(open(out_path("fig3", f"_{s_}"))) for s_, _ in scheds):
        draw_fig3(F3 + [F3_FK], "fig3_newfk.png",
                  "\nPink: FK potential steering -- gradient-free, weights from a potential of "
                  "the reward VALUE, resampled every 10 steps.  This objective's $\\psi$ is "
                  "measure-dependent, so FK solves it against its own normalised weights at every\n"
                  "checkpoint: 'diff' telescopes to "
                  r"$\lambda\psi(x_1,\mu_T)$, so its weights solve the SAME fixed point "
                  r"$\mu \propto p_1 e^{\lambda\psi(\cdot,\mu)}$ as the mean-field runs.")

    # ---- Algorithm 2: does pooling the measure across batches help?
    if os.path.exists(out_path("pooling_mb")):
        d = json.load(open(out_path("pooling_mb")))
        fig, ax = plt.subplots(figsize=(5.4, 3.6), constrained_layout=True, facecolor=SURF)
        for (n, row), col, mk in zip(sorted(d["w1"].items(), key=lambda kv: int(kv[0])),
                                     (RAMP[0], RAMP[2]), ("o", "s")):
            Ms = sorted(int(k) for k in row)
            ax.plot(Ms, [row[str(M)] for M in Ms], color=col, marker=mk, lw=1.6,
                    label=f"$N={n}$ per batch")
        ax.axhline(d["w1_floor"], color=MUTED, lw=1.0, ls=(0, (3, 2)),
                   label=f"MC floor at $N=4000$ ({d['w1_floor']:.3f})")
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlabel("batches $M$"); ax.set_ylabel(r"$W_1$ to $\mu^*$")
        ax.set_title("Algorithm 2: pooling the measure across batches", color=INK, fontsize=10)
        ax.legend(fontsize=8)
        fig.savefig("figmb_repro.png", dpi=170, facecolor=SURF)
        print("wrote figmb_repro.png", flush=True)
