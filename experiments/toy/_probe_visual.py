"""Visual must not perturb the sampler, and its t=0 cloud must be the prior."""
import sys, torch, yaml
sys.path[:0] = ["src", "experiments"]
from toy.run import (trained, target_sample, VisualConfig, VisualFKConfig,
                     VisualSteerer, VisualFKSteerer)
from steering.rewards import MMD
from steering.steerer import Config, FKConfig, FKSteerer, Steerer

cfg = yaml.safe_load(open(sys.argv[1]))
torch.set_default_dtype(torch.float64)
m, y = trained(cfg, "const"), target_sample(cfg)
pot, N, M = MMD(y, cfg["h"]), cfg["n"], cfg["record_traj"]
kw = dict(nsteps=cfg["steps"], seed=cfg["fig2_seed"], lam=cfg["lam_star"],
          resample_every=cfg["resample_fig2"])
bad = 0
for name, plain, vis in (
        ("meanfield", lambda: Steerer(m, pot, Config(k_fp=3, damp=1.0, **kw), N).run(),
                      lambda: VisualSteerer(m, pot, VisualConfig(k_fp=3, damp=1.0, record_traj=M, **kw), N).run()),
        ("fk_max",    lambda: FKSteerer(m, pot, FKConfig(potential_type="max", **kw), N).run(),
                      lambda: VisualFKSteerer(m, pot, VisualFKConfig(potential_type="max", record_traj=M, **kw), N).run())):
    p, v = plain(), vis()
    same = all(torch.equal(p[k], v[k]) for k in ("x", "a", "w"))
    roots = torch.unique(torch.cat([v["traj"][0]] + [g[0] for _, g in v["ghosts"]]))
    bad += (not same) + (len(roots) != M)
    print(f"{name:10s} x/a/w bitwise vs plain: {same}   distinct roots drawn: {len(roots)}/{M}   "
          f"root mean {roots.mean():+.4f} sd {roots.std():.4f}   survivors {v['traj'].shape[1]}")
prior = torch.randn(N, 1, generator=torch.Generator().manual_seed(0))
print(f"prior N(0,1) reference: mean ~0, sd ~1")
print("ALL CHECKS PASS" if not bad else f"{bad} CHECK(S) FAILED")
