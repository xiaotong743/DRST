# ===== PROTEINA: distillation loss, verbatim from experiments/proteina_distill.py =====
# the SDE's COEFFICIENTS are analytic (g, TAU, DTAU) but the drift v comes from the
# frozen teacher. Query points come from a persistent on-policy chain, not simulation.

def em_step(v, y, s, h, I0, gam):
    """Eq 15 in u: one EM step of the sampler's SDE on the SAME Brownian increment.

    Written from vf_to_score's own expression rather than the a/b reduction ExpStep uses, so the
    gate compares the two forms against each other instead of a form against itself.
    """
    t, d = TAU(s), DTAU(s)
    g = GT(t)
    return y + (v + g * (t * v - y) / (1.0 - t)) * (d * h) + (2.0 * g * gam * d).sqrt() * I0



# ---------------- training step ----------------
def train(cfg):
    """One chain per particle, advanced by the half-step the semigroup term already took."""
    mdl = _proteina(cfg)
    vf = VField(mdl)
    torch.manual_seed(cfg["seed"])
    fm, ema = _mk(cfg, vf, mdl.device), _mk(cfg, vf, mdl.device)
    ema.load_state_dict(fm.state_dict())
    opt = torch.optim.Adam(fm.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg["iters"], 1))
    n, dt, nl, gam = cfg["n"], cfg["dt"], cfg["n_levy"], cfg["temperature"]
    hist, t0 = [], time.time()
    with _binning(cfg):
        y = mdl.prior(n)
        u = (1.0 - cfg["h_max"]) * torch.rand(n, 1, 1, dtype=y.dtype, device=y.device)
        v = None
        lv = lambda hh: sample_levy(hh.expand_as(y), nl)
        for it in range(cfg["iters"] + 1):
            vf.memo = {}
            if v is not None:
                vf.seed(u, v)
            h = dt + (cfg["h_max"] - dt) * torch.rand_like(u)
            m, t = u + h / 2, u + h
            L, R = lv(h / 2), lv(h / 2)
            with torch.no_grad():
                ym = ema(y, u, m, L)                 # memo hit on u, no forward
            v = vf(ym, m)                            # THE one forward of this step
            ld = distill_loss(fm, ema, y, u, m, t, L, R, chen_combine(L, R))

            hm = dt / 2 + (dt / 2) * torch.rand_like(u)
            Im = lv(hm)
            lm = _sq(fm(y, u, u + hm, Im), em_step(vf(y, u), y, u, hm, Im[0], gam), hm)

            loss = cfg["eta"] * lm + (1 - cfg["eta"]) * ld
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fm.parameters(), cfg["clip"])
            opt.step()
            sched.step()
            with torch.no_grad():
                for q, r in zip(ema.parameters(), fm.parameters()):
                    q.mul_(cfg["ema"]).add_(r, alpha=1 - cfg["ema"])
            y, u = ym, m                             # the chain IS the semigroup half-step
            done = (u + cfg["h_max"] > 1.0 - cfg["t_eps"]).flatten()
            if bool(done.any()):
                y, u, v = y.clone(), u.clone(), v.clone()
                y[done], u[done] = mdl.prior(int(done.sum())), 0.0
                v[done] = vf.fresh(y[done], u[done])
            if it % cfg["log_every"] == 0:
                # saved every log interval, not just at the end: the loop is hours long and a
                # wall-clock kill would otherwise discard all of it. The head is ~50k params.
                torch.save(ema.state_dict(), cfg["out"].replace(".pt", f"_{cfg['init']}.pt"))
                f = mdl.n_forward / (it + 1)
                hist.append(dict(it=it, loss=float(loss), matching=float(lm),
                                 semigroup=float(ld), fwd_per_it=f,
                                 sec_per_it=(time.time() - t0) / (it + 1)))
                print(f"{it:7d}  loss {float(loss):11.6f}  match {float(lm):11.6f}  "
                      f"semi {float(ld):11.6f}  fwd/it {f:5.2f}", flush=True)
    torch.save(ema.state_dict(), cfg["out"].replace(".pt", f"_{cfg['init']}.pt"))
    json.dump(hist, open(out_path(cfg, f"fmtrain_{cfg['init']}"), "w"), indent=1)
    print(f"wrote {cfg['out']}", flush=True)


