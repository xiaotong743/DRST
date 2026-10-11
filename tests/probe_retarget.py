"""Can test_steer_a.py catch the three mutations that currently point at test_temperature.py?

If it can, test_temperature.py is a second independent reimplementation of dynamics `sample_a`
already pins, and the file can go with its mutations retargeted. Run on a node, not the login
host. Prints CAUGHT / MISSED per mutation; exit 0 if all three are caught.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = "tests/test_steer_a.py"

PROBES = [
    ("temperature-moved-onto-the-score", "steer_a.py",
     "        return g, float(self.cfg.temperature) * g",
     "        return float(self.cfg.temperature) * g, float(self.cfg.temperature) * g"),
    ("noise-std-ignores-temperature", "steer_a.py",
     "        return torch.sqrt(2 * torch.as_tensor(g, dtype=dt_t.dtype)\n"
     "                          * float(self.cfg.temperature) * dt_t)",
     "        return torch.sqrt(2 * torch.as_tensor(g, dtype=dt_t.dtype)\n"
     "                          * dt_t)"),
    ("microbatch-splits-the-measure", "steerer.py",
     "            psi_c = self.potential.psi(th_c, sup, w_comb)",
     "            psi_c = self.potential.psi(th_c, sup[lo:hi], "
     "w_comb[lo:hi] / w_comb[lo:hi].sum())"),
]

ok = True
for name, fname, old, new in PROBES:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for d in ("src", "tests"):
            shutil.copytree(ROOT / d, tmp / d, ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(ROOT / "experiments", tmp / "experiments",
                        ignore=shutil.ignore_patterns("__pycache__", "out"))
        f = next(c for c in (tmp / "src" / "steering" / fname, tmp / "experiments" / fname)
                 if c.exists())
        src = f.read_text()
        assert src.count(old) == 1, f"{name}: anchor matched {src.count(old)}x in {fname}"
        f.write_text(src.replace(old, new))
        r = subprocess.run([sys.executable, "-m", "pytest", TARGET, "-q", "-x", "--no-header"],
                           cwd=tmp, capture_output=True, text=True,
                           env={"PYTHONPATH": str(tmp / "src"), "PATH": "/usr/bin:/bin"})
        caught = r.returncode == 1
        ok &= caught
        print(f"{name:38s} -> {'CAUGHT' if caught else f'MISSED (exit {r.returncode})'}",
              flush=True)
        if not caught:
            print(r.stdout[-1500:], flush=True)

print(f"\nall three caught by {TARGET}: {ok}")
sys.exit(0 if ok else 1)
