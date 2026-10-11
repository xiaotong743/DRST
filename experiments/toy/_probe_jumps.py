"""Jump recording: alignment with the drawn line, and no perturbation of the sampler."""
import sys, torch, yaml
sys.path[:0] = ["src", "experiments"]
from toy.run import base_model, trained
from steering.steerer import Config, steer
from steering.rewards import MMD
from toy.run import target_sample

cfg = yaml.safe_load(open("configs/simulation.yaml"))
torch.set_default_dtype(torch.float64)
m, y = trained(cfg, "const"), target_sample(cfg)
pot = MMD(y, cfg["h"])
N = 200

def run(record, every):
    return steer(m, pot, Config(nsteps=10, seed=4, lam=10.0, k_fp=3, damp=1.0,
                                resample_every=every, record_traj=record), N)

# 1. exactly one event (k=5 of 0..9): frm is the full pre-resample population in slot order,
#    so every drawn endpoint must come from it.
o = run(N, 6)
assert len(o["jumps"]) == 1, [j for j, _ in o["jumps"]]
j, frm = o["jumps"][0]
to = o["traj"][j + 1]
assert set(to.tolist()) <= set(frm.tolist()), "hop endpoint is not a pre-resample particle"
moved = int((frm != to).sum())
print(f"one event at k={j}: {moved}/{N} slots hopped, "
      f"{len(set(to.tolist()))} distinct lineages survive of {len(set(frm.tolist()))}")

# 2. two events: shapes line up with the traj they annotate
o2 = run(N, 4)
print(f"two+ events at k={[j for j, _ in o2['jumps']]}")
for j, frm in o2["jumps"]:
    assert frm.shape == o2["traj"][j + 1].shape, (frm.shape, o2["traj"][j + 1].shape)

# 3. recording must not touch the sampler
a = run(0, 4); b = run(120, 4)
for k in ("x", "a", "w"):
    assert torch.equal(a[k], b[k]), f"{k} changed when record_traj was switched on"
print("x/a/w bitwise identical with and without trajectory+jump recording")
print("OK")
