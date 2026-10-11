"""The drawn genealogy must be an unbiased view of the particle system, not of its survivors."""
import sys, torch, yaml
sys.path[:0] = ["src", "experiments"]
from toy.run import trained, target_sample
from steering.rewards import MMD
from steering.steerer import Config, FKConfig, FKSteerer, steer

cfg = yaml.safe_load(open("configs/simulation_farther.yaml"))
torch.set_default_dtype(torch.float64)
m, y = trained(cfg, "const"), target_sample(cfg)
pot, N, lam = MMD(y, cfg["h"]), cfg["n"], cfg["lam_star"]
M = cfg["record_traj"]

def run(rt, kw):
    b = dict(nsteps=cfg["steps"], seed=cfg["fig2_seed"], record_traj=rt, **kw)
    return (FKSteerer(m, pot, FKConfig(**b), N).run() if "potential_type" in kw
            else steer(m, pot, Config(**b), N))

prior = run(M, dict(lam=0.0, resample_every=0))["traj"][0]
print(f"unsteered prior at t=0:  mean {prior.mean():+.4f}  sd {prior.std():.4f}  n {len(prior)}")

for name, kw in (("meanfield", dict(lam=lam, k_fp=3, damp=1.0,
                                    resample_every=cfg["resample_fig2"])),
                 ("fk_add", dict(lam=lam, potential_type="add",
                                 resample_every=cfg["resample_fig2"]))):
    o = run(M, kw)
    surv = o["traj"][0]
    roots = [surv] + [g[0] for _, g in o["ghosts"]]
    allt0 = torch.cat(roots)
    uniq = torch.unique(allt0)
    print(f"\n{name}:")
    print(f"  survivors only      mean {surv.mean():+.4f}  sd {surv.std():.4f}  n {len(surv)}")
    print(f"  + pruned branches   mean {allt0.mean():+.4f}  sd {allt0.std():.4f}  n {len(allt0)}")
    print(f"  DISTINCT watched roots drawn: {len(uniq)} (asked for {M})")
    print(f"  distinct-root cloud mean {uniq.mean():+.4f}  sd {uniq.std():.4f}")

a, b = run(0, dict(lam=lam, k_fp=3, damp=1.0, resample_every=cfg["resample_fig2"])), \
       run(M, dict(lam=lam, k_fp=3, damp=1.0, resample_every=cfg["resample_fig2"]))
for k in ("x", "a", "w"):
    assert torch.equal(a[k], b[k]), f"{k} moved when recording was switched on"
print("\nx/a/w bitwise identical with and without recording")
