"""Mean-field steered sampler (Algorithm 1).

    dX = [b + eps_t*s + eps_t*grad_x Psi] dt + sqrt(2 eps_t) dB      (12a)
    dA = [b . grad_x Psi + dPsi/dt] dt                               (12b)

Psi_t(x, mu) = lam * t * psi(xhat1(x,t), pihat_t), and b in (12b) is v alone: every eps term
cancels in the tilted Fokker-Planck equation, which forces temperature 1.

Model contract: prior(n, gen), endpoint(x, t) -> (theta, v), score(x, v, t), eps(t).
"""

from dataclasses import dataclass, field

import torch

from .ips import normalise, ess, residual_resample, picard


def _as_double(fn):
    return lambda w: fn(w).double()


def _retained_combine(retained):
    """Algorithm 2's candidate measure: normalise over the retained terminal log-weights and the
    current batch's candidate log-weights TOGETHER. Operating on log-weights is essential -- the
    current block's normalising constant is what sets the two sets' relative mass."""
    if retained is None:
        return None
    a_ret = retained["a"].double()
    return lambda a_cur: torch.softmax(torch.cat([a_ret, a_cur.double()]), 0)


def _dot(a, b):
    """Per-particle inner product over all event dims."""
    n = a.shape[0]
    return (a.reshape(n, -1) * b.reshape(n, -1)).sum(1)


class Pointwise:
    """psi(theta, mu) = r(theta). No measure dependence, so Algorithm 4 collapses."""

    def __init__(self, r):
        self.r = r

    def psi(self, query, support, w):
        return self.r(query)

    def weights_fn(self, query, support):
        """w -> psi(query, pihat(support, w)). Constant in w and in support here."""
        val = self.r(query)
        return lambda w: val

    # R(mu) = int r dmu is linear in mu, so delta^2 R / delta mu^2 == 0. Algorithm 3 then reduces
    # to psidot = d_t Psi with A = I, and Algorithm 4 converges in one iteration. Both are exact.
    def phi_tilde(self, query, support, w):
        return torch.zeros(query.shape[0], support.shape[0],
                           dtype=query.dtype, device=query.device)

    def phi_sum(self, query, support, w):
        return torch.zeros(query.shape[0], dtype=query.dtype, device=query.device)


class MMD:
    """psi = first variation of R(mu) = -MMD^2(features#mu, nu)."""

    def __init__(self, target, h, features=lambda x: x.reshape(x.shape[0], -1)):
        self.target, self.h, self.features = target, h, features

    def psi(self, query, support, w):
        from .mmd import psi as mmd_psi
        return mmd_psi(self.features(query), self.features(support), w.to(query.dtype),
                       self.target, self.h)

    def weights_fn(self, query, support):
        """w -> psi(query, pihat(support, w)), reusing one Gram so Picard is matvecs.

        eq (20) holds the QUERY at t and advances only the support, so the two differ.
        """
        from .mmd import rbf_gram
        fq = self.features(query)
        fs = self.features(support)
        cross = rbf_gram(fq, self.target, self.h).mean(dim=1)     # [N], fixed
        gram = rbf_gram(fq, fs, self.h)                           # [N, M], fixed
        return lambda w: 2.0 * (cross - gram @ w.to(fq.dtype))

    # --- Algorithm 3's second variation. MMD^2 is quadratic in mu, so
    #     Phi(x,y) = delta^2 R / delta mu^2 = -2 k(theta_x, theta_y),
    # independent of the measure. Both methods omit lam * alpha(t), as `psi` does.
    def phi_tilde(self, query, support, w):
        """Mean-centred: Phi(x,y) - int Phi(x,z) dmu(z).  [N, M]"""
        from .mmd import rbf_gram
        k = rbf_gram(self.features(query), self.features(support), self.h)
        return -2.0 * (k - (k @ w.to(k.dtype))[:, None])

    def phi_sum(self, query, support, w):
        """sum_j w_j Phi(x_i, y_j), the scalar whose y-derivative line 6 contracts with b.
        The centring term of Phitilde has no y-dependence, so it is absent here by design."""
        from .mmd import rbf_gram
        k = rbf_gram(self.features(query), self.features(support), self.h)
        return -2.0 * (k @ w.to(k.dtype))


@dataclass
class Config:
    nsteps: int = 100
    lam: float = 0.0
    psidot_solver: str = "picard"  # "picard" = Algorithm 4, "implicit" = Algorithm 3
    k_fp: int = 50
    damp: float = 0.5                  # the undamped Picard map is expansive at lam ~ rho > 1;
                                       # one damped iteration applies a + damp*inc, not the fixed
                                       # point, so k_fp=1 needs damp=1.0
    resample_every: int = 10          # 0 disables
    increment_clip: float | None = None
    reweight: bool = True             # False -> gradient-only steering (the paper's baseline)
    steer_coeff: float | None = None  # replaces eps_t on grad_x Psi. None = eq (11)/(12a).
                                      # Only for probing the baseline's unstated coefficient;
                                      # NOT eq (23a), which is additive and was not used.
    endpoint_lag: str = "advance"      # "advance": eq (20), query frozen at theta_i(t), support
                                       #   advanced, model-time term from autograd. Used by the
                                       #   paper in the protein experiments.
                                       # "literal": the plain finite difference, both endpoints at
                                       #   t+dt. The paper's low-dimensional setting. O(dt) apart.
                                       # "frozen": support held at theta_j(t). Drops an O(1) term.
    seed: int = 0
    ts: torch.Tensor | None = None    # explicit time grid; uniform if None
    record_traj: int = 0              # if > 0, record this many particles' positions each step
                                      # (post-resampling, so resampling shows as a jump)
    check_drift: bool = False         # assert the realised drift matches the intended one
    global_rng: bool = False          # use the GLOBAL torch RNG instead of a private generator,
                                      # so a lambda=0 run consumes the same stream as Proteina's
                                      # fm.full_simulation and can be compared bitwise
    vf_tail_t: float = 1.0            # for t > this, eps_eff = 0: no noise AND no steering.
                                      # 1.0 disables it (the paper's low-dim setup has no tail);
                                      # Proteina's sampler flips to ODE at 0.99.


def steer(model, potential, cfg, n_particles, device="cpu", retained=None):
    """Algorithm 1. `retained`, when given, is Algorithm 2's combined terminal measure --
    dict(theta=[L, *event], a=[L]) -- normalised jointly with the current batch.
    """
    g = None if cfg.global_rng else torch.Generator(device=device).manual_seed(cfg.seed)
    if cfg.ts is not None:
        ts = cfg.ts
    elif hasattr(model, "ts"):
        ts = model.ts            # the model's schedule is authoritative (eps is tied to it)
    else:
        ts = torch.linspace(0, 1, cfg.nsteps + 1)
    x = model.prior(n_particles, g)
    a = torch.zeros(n_particles, dtype=torch.float64, device=x.device)
    diag = []
    traj = [x[:cfg.record_traj].reshape(-1).clone()] if cfg.record_traj else None

    for k in range(len(ts) - 1):
        # keep dt as a tensor of the schedule's dtype and build the update in the same
        # association as Proteina's step_euler, so a lambda=0 run is bitwise comparable:
        # float32 addition is not associative and the 1/(1-t) score amplifies any difference.
        t_val, dt_t = float(ts[k]), (ts[k + 1] - ts[k])
        dt = float(dt_t)
        eps_t = model.eps(t_val)
        if t_val > cfg.vf_tail_t:
            eps_t = 0.0                    # one source of truth: no noise and no steering
        w = normalise(a)
        if retained is not None:
            # normalise over the combined set, then split: w_comb[:L] weights the retained
            # terminal samples, w_comb[L:] the current batch's endpoints.
            w_comb = normalise(torch.cat([retained["a"].to(a.dtype), a]))
        else:
            w_comb = w

        # --- one forward + one backward: grad_x Psi and dPsi/dt together
        xg = x.detach().requires_grad_(True)
        tg = torch.full((n_particles,), t_val, dtype=x.dtype, device=x.device).requires_grad_(True)
        theta, v = model.endpoint(xg, tg)
        sup = (torch.cat([retained["theta"].to(theta.dtype), theta.detach()])
               if retained is not None else theta.detach())
        psi_q = potential.psi(theta, sup, w_comb)                # measure held fixed
        Psi = cfg.lam * tg * psi_q
        if cfg.lam != 0.0:
            Psi.sum().backward()
            gx = xg.grad if xg.grad is not None else torch.zeros_like(x)
            gt = tg.grad if tg.grad is not None else torch.zeros_like(tg)
        else:
            gx = torch.zeros_like(x)
            gt = torch.zeros(n_particles, dtype=x.dtype, device=x.device)
        theta_d, v_d, psi_d = theta.detach(), v.detach(), psi_q.detach()

        # --- POSITIONAL UPDATE FIRST (Algorithm 1 order), so X_{t+dt} is available to Alg 4
        s_score = model.score(x, v_d, t_val)
        base = v_d + eps_t * s_score
        c_t = cfg.steer_coeff(t_val) if callable(cfg.steer_coeff) else cfg.steer_coeff
        steer_term = (c_t if c_t is not None else eps_t) * gx
        drift = base + steer_term
        noise = (torch.randn(x.shape, generator=g, dtype=x.dtype, device=x.device)
                 if eps_t > 0 else torch.zeros_like(x))
        eps_tt = torch.as_tensor(eps_t, dtype=dt_t.dtype)
        std = torch.sqrt(2 * eps_tt * dt_t) if eps_t > 0 else torch.zeros((), dtype=dt_t.dtype)
        delta = drift * dt_t + std * noise
        x_next = x + delta
        drift_dev = 0.0
        if cfg.check_drift:
            # Round-off self-consistency and a NaN tripwire only: `realised` is rebuilt from the
            # expression three lines up, so it cannot fail on a structural error. The independent
            # check is tests/test_sde.py's reference integrator.
            realised = (x_next - x - std * noise) / dt_t
            tol = (8.0 * torch.finfo(x.dtype).eps * float(x.abs().max()) / dt
                   + 1e-6 * float(drift.abs().max()))
            drift_dev = float((realised - drift).abs().max())
            assert drift_dev <= tol, (
                f"realised drift deviates by {drift_dev:.3e} at t={t_val:.4f}, above the "
                f"round-off bound {tol:.3e} -- the assembled drift is not what was applied")
        x_next = model.project(x_next) if hasattr(model, "project") else x_next

        # --- LOG-WEIGHT UPDATE, using the next-step endpoints as the candidate support
        k_used, resid = 0, 0.0
        psidot = torch.zeros(n_particles, dtype=torch.float64, device=x.device)
        if cfg.reweight and cfg.lam != 0.0:
            if cfg.endpoint_lag not in ("advance", "literal", "frozen"):
                raise ValueError(
                    f"endpoint_lag={cfg.endpoint_lag!r} is not recognised; use 'advance' "
                    f"(eq 20), 'literal' (the plain finite difference) or 'frozen'. The old name "
                    f"'exact' silently behaved as 'frozen', which made two scratch experiments "
                    f"compare frozen to frozen.")
            t_nx = min(t_val + dt, 1.0 - 1e-9)
            t_nx_v = torch.full((n_particles,), t_nx, dtype=x.dtype, device=x.device)
            query = theta_d
            time_term = (gt - cfg.lam * psi_d).double()
            if cfg.endpoint_lag == "frozen":
                support = theta_d
            else:
                with torch.no_grad():
                    support, _ = model.endpoint(x_next, t_nx_v)
                if cfg.endpoint_lag == "literal":
                    # Psi_{t+dt}(X_t, .) evaluates the network on the OLD positions at the NEW
                    # time, so the query endpoint moves too and no separate model-time term is
                    # needed: the FD already carries it.
                    with torch.no_grad():
                        query, _ = model.endpoint(x, t_nx_v)
                    time_term = torch.zeros_like(time_term)
            bgrad = _dot(v_d, gx).double()
            if cfg.psidot_solver == "implicit":
                if retained is not None:
                    raise NotImplementedError(
                        "Algorithm 3 with Algorithm 2's retained measure is not supported. "
                        "Naively passing the combined weights is wrong: (13) contains "
                        "integral Phitilde(x,y) Psidot(y) dmu(y) over the WHOLE measure, so the "
                        "retained terminal points would need a Psidot of their own -- but they "
                        "are frozen at t=1 and have none, and the paper does not say what to do "
                        "there. Use psidot_solver='picard' for multi-batch, which is what the "
                        "paper used in its multi-batch experiments.")
                # Algorithm 3. Assembled at the CURRENT cloud, which is where Proposition 3.3's
                # relation (13) holds ("under the target flow"); the advance/literal endpoint
                # question is a property of the finite difference and does not arise here.
                from .alg3 import implicit_psidot, psidot_terms
                phi, dpsi_dt, phi_grad_b = psidot_terms(
                    model, potential, x, t_val, dt, w, cfg.lam, v_d, gx)
                psidot, resid = implicit_psidot(phi.double(), w.double(), dpsi_dt.double(),
                                                phi_grad_b.double(), bgrad)
                inc = dt * (psidot + bgrad)
                if cfg.increment_clip is not None:
                    inc = inc.clamp(-cfg.increment_clip, cfg.increment_clip)
                a = a + inc
                w, k_used = normalise(a), 1
            elif cfg.psidot_solver == "picard":
                a, w, k_used, resid, _, psidot = picard(
                    a0=a, psi_t=psi_d.double(),
                    psi_of_weights=_as_double(potential.weights_fn(
                        query, torch.cat([retained["theta"].to(support.dtype), support])
                        if retained is not None else support)),
                    time_term=time_term,
                    bgrad=bgrad,
                    t=t_val, dt=dt, lam=cfg.lam,
                    k_fp=cfg.k_fp, damp=cfg.damp, increment_clip=cfg.increment_clip,
                    combine=_retained_combine(retained))
            else:
                raise ValueError(f"psidot_solver={cfg.psidot_solver!r}; use 'picard' or "
                                 f"'implicit'")
        x = x_next

        ess_pre = ess(a)          # before any reset, so the diagnostic is meaningful
        # resampling: skip in the noiseless tail, where duplicates would be bitwise identical
        n_res = 0
        if cfg.resample_every and eps_t > 0 and (k + 1) % cfg.resample_every == 0:
            idx = residual_resample(normalise(a), g)
            if not torch.equal(idx, torch.arange(n_particles, device=idx.device)):
                x = x[idx]
                n_res = int(n_particles - torch.unique(idx).numel())
            a = torch.zeros_like(a)

        if traj is not None:
            traj.append(x[:cfg.record_traj].reshape(-1).clone())
        diag.append(dict(step=k, t=t_val, dt=dt, eps=eps_t, ess=ess_pre, ess_post=ess(a),
                         n_resampled=n_res, drift_dev=drift_dev,
                         picard_iters=k_used, picard_resid=resid,
                         psidot_mean=float(psidot.mean()) if cfg.lam != 0 else 0.0,
                         psi_mean=float(psi_d.mean()),
                         steer_norm=float(steer_term.reshape(n_particles, -1).norm(dim=1).median())
                         if cfg.lam != 0 else 0.0,
                         base_norm=float(base.reshape(n_particles, -1).norm(dim=1).median())))

    return dict(x=x, a=a, w=normalise(a), diag=diag,
                traj=(torch.stack(traj) if traj is not None else None),
                ts=ts)


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
