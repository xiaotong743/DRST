"""Where mu* sits as lambda grows, for the configs passed in."""
import sys, torch, yaml
sys.path[:0] = ["src", "experiments"]
from toy.run import base_model, target_sample
from toy.toy import grid_mu_star

torch.set_default_dtype(torch.float64)
for cf in sys.argv[1:]:
    cfg = yaml.safe_load(open(cf))
    m, y = base_model(cfg), target_sample(cfg)
    grid = torch.linspace(-6, 6, 1201)
    p1 = torch.softmax(m.log_p1(grid), 0)
    left = float(p1[grid < 0].sum())
    print(f"\n=== {cf}  p1 mean {float((grid*p1).sum()):+.3f}  "
          f"p1 mass left of 0 = {left:.3f}   nu mean {float(y.mean()):+.3f}")
    print(f"{'lam':>5} {'mu* mean':>9} {'mu* left mass':>14} {'residual':>10}")
    for lam in (5.0, 10.0, 20.0, 30.0, 50.0):
        g, q = grid_mu_star(m, y, lam, cfg["h"], grid=grid)
        g2, q2 = grid_mu_star(m, y, lam, cfg["h"], grid=grid, iters=4000)
        print(f"{lam:5.0f} {float((g*q).sum()):+9.3f} {float(q[g<0].sum()):14.3f} "
              f"{float((q-q2).abs().max()):10.2e}")
