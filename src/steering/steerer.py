"""Mean-field steered sampler (Algorithm 1), and the FK potential variant.

    dX = [b + eps_t*s + eps_t*grad_x Psi] dt + sqrt(2 eps_t) dB      (12a)
    dA = [b . grad_x Psi + dPsi/dt] dt                               (12b)

Psi_t(x, mu) = lam * t * psi(xhat1(x,t), pihat_t); b in (12b) is v alone.

This module is the published method and nothing else. Variants subclass `Steerer` and override
its hooks: `FKSteerer` below, and experiments/steer_a.py.

Model contract: prior(n, gen), endpoint(x, t) -> (theta, v), score(x, v, t), eps(t); optionally
project(x) and reward_endpoint(x, t).
"""

from dataclasses import dataclass, field

import torch

from .meanfield import picard
from .smc import normalise, ess, residual_resample


def _as_double(fn):
    return lambda w: fn(w).double()


def _retained_combine(retained):
    """Algorithm 2's candidate measure. The two log-weight sets normalise TOGETHER."""
    if retained is None:
        return None
    a_ret = retained["a"].double()
    return lambda a_cur: torch.softmax(torch.cat([a_ret, a_cur.double()]), 0)


def _chunked(fn, x, t_vec, chunk):
    """Per-particle-independent model call, in chunks. Exact: nothing couples the batch."""
    n = x.shape[0]
    if not chunk or chunk >= n:
        return fn(x, t_vec)
    parts = [fn(x[lo:lo + chunk], t_vec[lo:lo + chunk]) for lo in range(0, n, chunk)]
    if isinstance(parts[0], tuple):
        return tuple(torch.cat([p[k] for p in parts]) for k in range(len(parts[0])))
    return torch.cat(parts)


def _reward_endpoint(model, x, t_vec, chunk=None):
    """theta = D_t(x) for the reward only. ProteinaModel overrides it to return x at t = 1."""
    fn = getattr(model, "reward_endpoint", None)
    with torch.no_grad():
        if fn is not None:
            return _chunked(fn, x, t_vec, chunk)
        theta, _ = _chunked(model.endpoint, x, t_vec, chunk)
        return theta


def _dot(a, b):
    """Per-particle inner product over all event dims."""
    n = a.shape[0]
    return (a.reshape(n, -1) * b.reshape(n, -1)).sum(1)


@dataclass
class Config:
    nsteps: int = 100
    lam: float = 0.0
    psidot_solver: str = "picard"  # "picard" = Algorithm 4, "implicit" = Algorithm 3
    k_fp: int = 50
    damp: float = 0.5                 # k_fp=1 needs damp=1.0
    resample_every: int = 10          # 0 disables
    increment_clip: float | None = None
    reweight: bool = True             # False -> gradient-only steering (the paper's baseline)
    steer_coeff: float | None = None  # replaces eps_t on grad_x Psi. None = eq (11)/(12a).
                                      # With reweight=True and c_t != the DIFFUSION coefficient,
                                      # (12b) is short by (c_t - D)*[lap Psi + s.grad Psi + |grad Psi|^2].
    endpoint_lag: str = "advance"     # "advance" = eq (20); "literal" = plain finite difference
    seed: int = 0
    ts: torch.Tensor | None = None    # explicit time grid; uniform if None
    record_traj: int = 0              # record this many particles' positions each step
    check_drift: bool = False         # assert the realised drift matches the intended one
    global_rng: bool = False          # GLOBAL torch RNG, so lam=0 is bitwise fm.full_simulation
    vf_tail_t: float = 1.0            # t > this: g_t = 0. Proteina flips to the ODE at 0.99.
    grad_chunk: int | None = None     # microbatch for the grad_x Psi backward. Exact at any
                                      # value: the measure is detached.
    nograd_chunk: int | None = None   # microbatch for the no_grad endpoint passes; None follows
                                      # grad_chunk. KEEP IT NONE: **[measured]** it buys <1% at
                                      # L=250 and costs ~1e-5 in x, because Proteina checkpoints
                                      # internally so the no_grad and grad paths differ.


class Steerer:
    """Algorithm 1. Each step is four phases: reward -> move -> weights -> resample."""

    def __init__(self, model, potential, cfg, n_particles, device="cpu", retained=None):
        self.model, self.potential, self.cfg = model, potential, cfg
        self.n, self.device, self.retained = n_particles, device, retained
        self._diag_extra = {}
        self.check()

    # ------------------------------------------------------------------ hooks
    def check(self):
        pass

    def initial_particles(self, gen):
        return self.model.prior(self.n, gen)

    def coeffs(self, t_val):
        """(score coefficient, steering coefficient). Both are eps_t in eq (12a)."""
        g = self.model.eps(t_val)
        if t_val > self.cfg.vf_tail_t:
            g = 0.0
        return g, g

    def begin(self, x, a):
        """Per-run state. Algorithm 1 has none."""

    def reindex(self, idx, a):
        """Permute per-particle state onto the resampled ancestors. Algorithm 1 has none."""

    def resample_gen(self, gen):
        """RNG for the resampling draw. A subclass can use a private stream."""
        return gen

    def finalise(self, x, a):
        """Last word on (x, a) after the loop. Algorithm 1 has none."""
        return x, a

    def noise_std(self, t_val, dt_t):
        g, _ = self.coeffs(t_val)
        if g <= 0:
            return torch.zeros((), dtype=dt_t.dtype)
        return torch.sqrt(2 * torch.as_tensor(g, dtype=dt_t.dtype) * dt_t)

    def _nograd_chunk(self):
        return self.cfg.nograd_chunk or self.cfg.grad_chunk

    def _reward(self, x, t_val, w_comb):
        """PHASE 1. No-grad pass: the detached endpoints (which ARE the measure), v, and psi."""
        model, cfg, retained = self.model, self.cfg, self.retained
        t_vec = torch.full((self.n,), t_val, dtype=x.dtype, device=x.device)
        with torch.no_grad():
            theta_d, v_d = _chunked(model.endpoint, x, t_vec, self._nograd_chunk())
            theta_d, v_d = theta_d.detach(), v_d.detach()
            sup = (torch.cat([retained["theta"].to(theta_d.dtype), theta_d])
                   if retained is not None else theta_d)
            psi_d = self.potential.psi(theta_d, sup, w_comb).detach()
        return theta_d, v_d, sup, psi_d

    def _cal_sample_gradient(self, x, t_val, sup, w_comb):
        """PHASE 1b. grad_x Psi and dPsi/dt, microbatched over the QUERY particles only.

        `sup` and `w_comb` stay detached and FULL, so chunking splits the VJP, not the measure.
        """
        cfg, n_particles = self.cfg, self.n
        gx = torch.zeros_like(x)
        gt = torch.zeros(n_particles, dtype=x.dtype, device=x.device)
        if cfg.lam == 0.0:
            return gx, gt
        step = cfg.grad_chunk or n_particles
        for lo in range(0, n_particles, step):
            hi = min(lo + step, n_particles)
            xg = x[lo:hi].detach().requires_grad_(True)
            tg = torch.full((hi - lo,), t_val, dtype=x.dtype,
                            device=x.device).requires_grad_(True)
            th_c, _ = self.model.endpoint(xg, tg)
            psi_c = self.potential.psi(th_c, sup, w_comb)
            (cfg.lam * tg * psi_c).sum().backward()
            if xg.grad is not None:
                gx[lo:hi] = xg.grad
            if tg.grad is not None:
                gt[lo:hi] = tg.grad
        return gx, gt

    def _move(self, x, t_val, dt_t, v_d, gx, gen):
        """PHASE 2. One Euler-Maruyama step of eq (12a). FK leaves it untouched: gx is zero."""
        cfg, model = self.cfg, self.model
        c_score, c_steer = self.coeffs(t_val)
        std = self.noise_std(t_val, dt_t)
        base = v_d + c_score * model.score(x, v_d, t_val)
        c_t = cfg.steer_coeff(t_val) if callable(cfg.steer_coeff) else cfg.steer_coeff
        steer_term = (c_t if c_t is not None else c_steer) * gx
        drift = base + steer_term
        noise = (torch.randn(x.shape, generator=gen, dtype=x.dtype, device=x.device)
                 if float(std) > 0 else torch.zeros_like(x))
        x_next = x + (drift * dt_t + std * noise)
        dev = 0.0
        if cfg.check_drift:                 # round-off and NaN tripwire only
            dt = float(dt_t)
            realised = (x_next - x - std * noise) / dt_t
            tol = (8.0 * torch.finfo(x.dtype).eps * float(x.abs().max()) / dt
                   + 1e-6 * float(drift.abs().max()))
            dev = float((realised - drift).abs().max())
            assert dev <= tol, (
                f"realised drift deviates by {dev:.3e} at t={t_val:.4f}, above the round-off "
                f"bound {tol:.3e} -- the drift applied is not the one assembled")
        x_next = model.project(x_next) if hasattr(model, "project") else x_next
        return x_next, base, steer_term, dev

    def _weights(self, a, w, w_comb, x, x_next, t_val, dt, theta_d, v_d, psi_d, gx, gt):
        """PHASE 3. The log-weight update, and the whole difference between the two methods:
        an FK potential of the reward VALUE, or the mean-field rate of eq (12b)."""
        cfg, model, potential, n = self.cfg, self.model, self.potential, self.n
        retained = self.retained
        d = dict(picard_iters=0, picard_resid=0.0, picard_contract=0.0, psidot_mean=0.0)
        if not (cfg.reweight and cfg.lam != 0.0):
            return a, w, d

        if cfg.endpoint_lag not in ("advance", "literal"):
            raise ValueError(f"endpoint_lag={cfg.endpoint_lag!r}; use 'advance' (eq 20) or "
                             f"'literal' (the plain finite difference)")
        t_nx_v = torch.full((n,), min(t_val + dt, 1.0 - 1e-9), dtype=x.dtype, device=x.device)
        query = theta_d
        time_term = (gt - cfg.lam * psi_d).double()
        support = _reward_endpoint(model, x_next, t_nx_v, self._nograd_chunk())
        if cfg.endpoint_lag == "literal":   # query moves too, so it carries the time term
            query = _reward_endpoint(model, x, t_nx_v, self._nograd_chunk())
            time_term = torch.zeros_like(time_term)
        bgrad = _dot(v_d, gx).double()
        if cfg.psidot_solver == "implicit":
            if retained is not None:
                raise NotImplementedError(
                    "Algorithm 3 with Algorithm 2's retained measure is not supported: (13) "
                    "integrates Phitilde(x,y) Psidot(y) over the WHOLE measure, so the retained "
                    "terminal points would need a Psidot of their own, and frozen at t=1 they "
                    "have none. Use psidot_solver='picard'.")
            from .meanfield import implicit_psidot, psidot_terms
            phi, dpsi_dt, phi_grad_b = psidot_terms(model, potential, x, t_val, dt, w, cfg.lam,
                                                    v_d, gx)
            psidot, resid = implicit_psidot(phi.double(), w.double(), dpsi_dt.double(),
                                            phi_grad_b.double(), bgrad)
            inc = dt * (psidot + bgrad)
            if cfg.increment_clip is not None:
                inc = inc.clamp(-cfg.increment_clip, cfg.increment_clip)
            a = a + inc
            d.update(picard_iters=1, picard_resid=resid, psidot_mean=float(psidot.mean()))
            return a, normalise(a), d
        if cfg.psidot_solver != "picard":
            raise ValueError(f"psidot_solver={cfg.psidot_solver!r}; use 'picard' or 'implicit'")
        a, w, k_used, resid, hist, psidot = picard(
            a0=a, psi_t=psi_d.double(),
            psi_of_weights=_as_double(potential.weights_fn(
                query, torch.cat([retained["theta"].to(support.dtype), support])
                if retained is not None else support)),
            time_term=time_term, bgrad=bgrad, t=t_val, dt=dt, lam=cfg.lam,
            k_fp=cfg.k_fp, damp=cfg.damp, increment_clip=cfg.increment_clip,
            combine=_retained_combine(retained))
        # picard_contract = |da| last over |da| first: ~0 converged, >=1 means k_fp truncated.
        d.update(picard_iters=k_used, picard_resid=resid,
                 picard_contract=(hist[-1] / hist[0]) if hist and hist[0] > 0 else 0.0,
                 psidot_mean=float(psidot.mean()))
        return a, w, d

    def _resample(self, x, a, k, diffusing, gen):
        """PHASE 4. Residual resampling, skipped in the noiseless tail where duplicates would
        follow identical paths. ESS is read BEFORE the reset."""
        cfg, n_particles = self.cfg, self.n
        ess_pre = ess(a)
        n_res = 0
        if cfg.resample_every and diffusing and (k + 1) % cfg.resample_every == 0:
            idx = residual_resample(normalise(a), self.resample_gen(gen))
            if not torch.equal(idx, torch.arange(n_particles, device=idx.device)):
                x = x[idx]
                n_res = int(n_particles - torch.unique(idx).numel())
            self.reindex(idx, a)
            a = torch.zeros_like(a)
        return x, a, ess_pre, n_res

    def run(self):
        model, potential, cfg, n_particles = self.model, self.potential, self.cfg, self.n
        retained = self.retained
        g = None if cfg.global_rng else torch.Generator(
            device=self.device).manual_seed(cfg.seed)
        if cfg.ts is not None:
            ts = cfg.ts
        elif hasattr(model, "ts"):
            ts = model.ts        # authoritative: eps is tied to it
        else:
            ts = torch.linspace(0, 1, cfg.nsteps + 1)
        x = self.initial_particles(g)
        a = torch.zeros(n_particles, dtype=torch.float64, device=x.device)
        self.begin(x, a)
        diag = []
        traj = [x[:cfg.record_traj].reshape(-1).clone()] if cfg.record_traj else None

        for k in range(len(ts) - 1):
            # dt stays a tensor: fp32 addition is not associative and 1/(1-t) amplifies it
            t_val, dt_t = float(ts[k]), (ts[k + 1] - ts[k])
            dt = float(dt_t)
            c_score, c_steer = self.coeffs(t_val)
            diffusing = float(self.noise_std(t_val, dt_t)) > 0
            w = normalise(a)
            if retained is not None:
                w_comb = normalise(torch.cat([retained["a"].to(a.dtype), a]))
            else:
                w_comb = w

            theta_d, v_d, sup, psi_d = self._reward(x, t_val, w_comb)
            gx, gt = self._cal_sample_gradient(x, t_val, sup, w_comb)
            # positions first (Algorithm 1 order), so X_{t+dt} reaches Algorithm 4
            x_next, base, steer_term, drift_dev = self._move(x, t_val, dt_t, v_d, gx, g)
            a, w, wdiag = self._weights(a, w, w_comb, x, x_next, t_val, dt,
                                        theta_d, v_d, psi_d, gx, gt)
            x = x_next
            x, a, ess_pre, n_res = self._resample(x, a, k, diffusing, g)

            if traj is not None:
                traj.append(x[:cfg.record_traj].reshape(-1).clone())
            diag.append(dict(
                step=k, t=t_val, dt=dt, eps=c_steer, g=c_score, ess=ess_pre, ess_post=ess(a),
                n_resampled=n_res, drift_dev=drift_dev,
                psi_mean=float(psi_d.mean()),
                steer_norm=float(steer_term.reshape(n_particles, -1).norm(dim=1).median())
                if cfg.lam != 0 else 0.0,
                base_norm=float(base.reshape(n_particles, -1).norm(dim=1).median()),
                **wdiag, **self._diag_extra))

        x, a = self.finalise(x, a)

        return dict(x=x, a=a, w=normalise(a), diag=diag,
                    traj=(torch.stack(traj) if traj is not None else None), ts=ts)




def steer(model, potential, cfg, n_particles, device="cpu", retained=None):
    """Algorithm 1. Thin wrapper so callers need not name the class."""
    return Steerer(model, potential, cfg, n_particles, device, retained).run()


def multibatch(model, potential, cfg, n_particles, n_batches, device="cpu"):
    """Algorithm 2: mean-field steering across M sequential batches.

    Each batch runs Algorithm 1, but the reward sees the combined terminal measure: retained
    terminal samples from earlier batches plus the current batch's Tweedie estimates, normalised
    over both. Only the current batch is updated. Returns dict(x, a, w, batches).
    """
    from dataclasses import replace

    xs, as_, batches, retained = [], [], [], None
    for m in range(n_batches):
        out = steer(model, potential, replace(cfg, seed=cfg.seed + m), n_particles,
                    device=device, retained=retained)
        theta_end = out["x"].detach()          # terminal samples: real endpoints, not estimates
        xs.append(theta_end)
        as_.append(out["a"].detach())
        batches.append(out["diag"])
        retained = dict(theta=torch.cat(xs), a=torch.cat(as_))
    x = torch.cat(xs)
    a = torch.cat(as_)
    return dict(x=x, a=a, w=normalise(a), batches=batches, n_batches=n_batches)


# --------------------------------------------------------------------------------------------
# Feynman-Kac potential steering (arXiv:2501.06848). The dynamics are untouched: FK acts only
# through the weights and resampling, and the weight comes from a potential of the reward VALUE
# rather than the mean-field rate of eq (12b), so no gradient is taken and r may be any black box.
#
#     target     p(x1) prop. p_theta(x1) * exp(lam * r(x1))
#     constraint prod_t G_t = exp(lam * r(x1))
#
# r is read off psi, so a measure-dependent potential works unchanged, and `_solve` makes it
# SELF-CONSISTENT: psi is read at the measure the weights themselves define, both per step and at
# the terminal rule, so the weights solve mu prop. p1 exp(lam psi(.,mu)). A per-particle reward
# ignores its weight argument, so the solve exits on the first pass and is bitwise unaffected.
# **[measured]** on the toy MMD at lam=20, one pass collapses to ESS/N 0.003 where the solve holds
# 0.993. The reference implementation has no such loop; this is the distributional extension.
# --------------------------------------------------------------------------------------------


POTENTIALS = ("diff", "max", "add", "rt", "is")


def log_potential(kind, lam, r, a, r_prev, logP):
    """One FK step. Returns (a, r_prev).

    `diff` is an increment; the others are cumulative and subtract `logP`, what resampling paid.
    """
    if kind == "diff":
        return a + lam * (r - r_prev), r
    if kind == "is":
        return a, r_prev
    if kind == "max":
        r_prev = torch.maximum(r, r_prev)
    elif kind == "add":
        r_prev = r + r_prev
    elif kind == "rt":
        r_prev = r
    else:
        raise ValueError(f"potential_type={kind!r}; use one of {POTENTIALS}")
    return lam * r_prev - logP, r_prev


@dataclass
class FKConfig(Config):
    potential_type: str = "diff"      # one of POTENTIALS
    reward_min_value: float = 0.0     # r_prev seed; only "max" is sensitive to it


class FKSteerer(Steerer):
    """Algorithm 1's dynamics with the log-weights driven by an FK potential.

    The potential type matters only through the resampling: `finalise` ends every run at
    a = lam*r(x1) - logP, so at resample_every=0 they all collapse to SNIS.
    """

    def check(self):
        super().check()
        if self.cfg.potential_type not in POTENTIALS:
            raise ValueError(f"potential_type={self.cfg.potential_type!r}; use one of {POTENTIALS}")

    def begin(self, x, a):
        self._logP = torch.zeros_like(a)
        self._r_prev = torch.full_like(a, self.cfg.reward_min_value)
        # private stream, so turning FK on does not shift the Brownian sequence
        self._gen = torch.Generator(device=x.device).manual_seed(self.cfg.seed + 20_000_003)

    def resample_gen(self, gen):
        return self._gen

    def reindex(self, idx, a):
        self._logP = (self._logP + a)[idx]      # ancestral, never reset
        self._r_prev = self._r_prev[idx]

    def _cal_sample_gradient(self, x, t_val, sup, w_comb):
        """FK weights the reward VALUE and never differentiates it."""
        return torch.zeros_like(x), torch.zeros(self.n, dtype=x.dtype, device=x.device)

    def _solve(self, a0, r_prev, f):
        """Self-consistent weights: psi is read at the measure the weights themselves define.

        Iteration 0 takes the FULL step, so a per-particle reward -- where `f` ignores its
        argument -- lands on the fixed point in one pass and is bitwise unaffected by damping.
        `r_prev` stays frozen; max/add would otherwise compound it once per iteration.
        """
        cfg, a, resid, new_prev = self.cfg, a0, 0.0, r_prev
        for k in range(max(cfg.k_fp, 1)):
            r = f(normalise(a)).double()
            cand, new_prev = log_potential(cfg.potential_type, cfg.lam, r, a0, r_prev, self._logP)
            cand = cand if k == 0 else (1 - cfg.damp) * a + cfg.damp * cand
            resid = float((cand - a).abs().max())
            a = cand
            if resid < 1e-9:
                break
        return a, new_prev, k + 1, resid

    def _weights(self, a, w, w_comb, x, x_next, t_val, dt, theta_d, v_d, psi_d, gx, gt):
        d = dict(picard_iters=0, picard_resid=0.0, picard_contract=0.0, psidot_mean=0.0)
        if not (self.cfg.reweight and self.cfg.lam != 0.0):
            return a, w, d
        a, self._r_prev, k, resid = self._solve(
            a, self._r_prev, self.potential.weights_fn(theta_d, theta_d))
        d.update(picard_iters=k, picard_resid=resid)
        return a, normalise(a), d

    def finalise(self, x, a):
        """The loop scores theta at the PRE-update positions, so the final sample is never seen.
        This is the whole terminal correction, and it is the same for every potential type."""
        cfg = self.cfg
        if not (cfg.reweight and cfg.lam != 0.0):
            return x, a
        t_end = torch.full((self.n,), 1.0 - 1e-9, dtype=x.dtype, device=x.device)
        theta_end = _reward_endpoint(self.model, x, t_end, self._nograd_chunk())
        f = self.potential.weights_fn(theta_end, theta_end)
        for k in range(max(cfg.k_fp, 1)):          # terminal rule is lam*r - logP for every type
            cand = cfg.lam * f(normalise(a)).double() - self._logP
            cand = cand if k == 0 else (1 - cfg.damp) * a + cfg.damp * cand
            done = float((cand - a).abs().max()) < 1e-9
            a = cand
            if done:
                break
        return x, a
