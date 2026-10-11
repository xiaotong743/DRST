# ===== SSFM: distillation loss, verbatim from experiments/toy/ssfm_sde.py =====
# the SDE is KNOWN: b_drift and sigma are analytic, there is no teacher and no data.

BETA_MIN, BETA_MAX = 0.1, 20.0            # sde.py:16-17; _beta at :20-21, so 0.1 + 19.9 t


def beta(t):
    return BETA_MIN + (BETA_MAX - BETA_MIN) * t


def b_drift(t, x):
    """sde.py:24-25 `b_drift` -- the double well."""
    return x - x ** 3


def sigma(t):
    """sde.py:28-29 `sigma_diffusion`."""
    return beta(t).sqrt() if torch.is_tensor(t) else beta(torch.as_tensor(t)).sqrt()


def fine_em_to(x0, s, n_sub):
    """sde.py:130-141 `fine_em_to_xs`: reach time s by n_sub Euler-Maruyama substeps. Fresh
    noise, since only the
    interval [s, s+h] has to share a path with the target."""
    x, t = x0, torch.zeros_like(s)
    h = s / n_sub
    for _ in range(n_sub):
        z = torch.randn(x.shape, dtype=x.dtype, device=x.device)
        x = x + b_drift(t, x) * h + sigma(t) * h.sqrt() * z
        t = t + h
    return x


def _step_target(ys, s, h, I0):
    return ys + b_drift(s, ys) * h + sigma(s) * I0


# ---------------- training step ----------------
def train(cfg):
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(cfg["seed"])
    dev = cfg["device"]
    mk = lambda: EulerMaruyamaFlowMap(MLPStep(dim=1, n_levy=cfg["n_coeffs"], width=cfg["width"],
                                              depth=cfg["depth"], dt_feat=False)).to(dev)
    fm, ema = mk(), mk()
    ema.load_state_dict(fm.state_dict())
    opt = torch.optim.Adam(fm.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["iters"])
    n, dt, eps = cfg["batch"], cfg["dt"], cfg["t_eps"]
    nm = int(n * cfg["eta"])
    hist = []
    for it in range(cfg["iters"]):
        # matching: short steps, h ~ U(dt/2, dt), s ~ U(t_eps, 1-h)      sde.py:179-184
        h = dt / 2 + (dt / 2) * torch.rand(nm, 1, device=dev)
        s = eps + (1.0 - h - eps) * torch.rand(nm, 1, device=dev)
        ys = fine_em_to(torch.randn(nm, 1, device=dev), s, cfg["n_sub"])
        I = sample_levy(h, cfg["n_coeffs"])
        lm = _sq(fm(ys, s, s + h, I), _step_target(ys, s, h, I[0]).detach(), h)

        # semigroup: long steps, h ~ U(dt, h_max). The halves are sampled and Chen-combined, so
        # both sides of the loss are driven by one path without needing a Brownian tree.
        nd = n - nm
        h2 = dt + (cfg["h_max"] - dt) * torch.rand(nd, 1, device=dev)
        s2 = eps + torch.clamp(1.0 - h2 - eps, min=1e-6) * torch.rand(nd, 1, device=dev)
        y2 = fine_em_to(torch.randn(nd, 1, device=dev), s2, cfg["n_sub"])
        L, R = sample_levy(h2 / 2, cfg["n_coeffs"]), sample_levy(h2 / 2, cfg["n_coeffs"])
        with torch.no_grad():
            mid = s2 + h2 / 2
            tgt = ema(ema(y2, s2, mid, L), mid, s2 + h2, R)
        ld = _sq(fm(y2, s2, s2 + h2, chen_combine(L, R)), tgt, h2)

        loss = cfg["eta"] * lm + (1 - cfg["eta"]) * ld
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(fm.parameters(), cfg["clip"])
        opt.step(); sched.step()
        with torch.no_grad():
            for q, r in zip(ema.parameters(), fm.parameters()):
                q.mul_(cfg["ema"]).add_(r, alpha=1 - cfg["ema"])
        if it % cfg["log_every"] == 0:
            hist.append(dict(it=it, loss=float(loss), matching=float(lm), semigroup=float(ld)))
            print(f"{it:7d}  loss {float(loss):11.6f}  match {float(lm):11.6f}  "
