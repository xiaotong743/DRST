"""Branch F: FK potential steering on the frozen Proteina model, at its own temperature.

Same reward, same gamma and same seed as branch A -- the only difference is that the log-weights
come from a potential of the reward VALUE instead of eq (12b)'s gradient rate. So A vs F is a
controlled gradient / no-gradient comparison, and F costs ~1x the unsteered sampler against A's
2.56-5.48x.

`resample_every` is the one stride governing both resampling and reward evaluation, as in the
reference implementation; the variant table sweeps it.

Usage: python experiments/steer_f.py configs/steer_f.yaml <stage> [args] [--set.key=value ...]
"""
import os
import sys
from dataclasses import dataclass

from steer_a import ConfigA, SteererA
from steer_a import mk_cfg as mk_cfg_a
from proteina import (calibrate, foldseek_labeller, main, resolve_h, run_variants,
                    steered_eval, tm_labeller)
from steering.steerer import FKConfig, FKSteerer

MODES = ("diff", "max", "add")
FREQS = (1, 5, 20, 80)


@dataclass
class ConfigF(FKConfig, ConfigA):
    """FK's potential settings plus A's temperature. The decorator is load-bearing: without it
    the class inherits FKConfig's frozen __init__, which has no `temperature`."""


class SteererF(FKSteerer, SteererA):
    """MRO: F -> FKSteerer -> SteererA -> Steerer. The parents override disjoint hooks, so FK's
    weights compose with A's raw-g/eps split; FKSteerer.check calls super() to reach A's."""


def _table(cfg):
    e = cfg["eta"]
    t = {"native": dict(lam=0.0)}
    for kind in MODES:
        for f in FREQS:
            t[f"{kind}{f}"] = dict(lam=e, potential_type=kind, resample_every=f)
    return t


def mk_cfg(cfg):
    return mk_cfg_a(cfg, ConfigF)


def _pot_fn(cfg, which):
    """Which reward this run steers on. `contact` is branch A's differentiable one; the other
    two are the paper's own diversity metrics, usable only because FK needs no gradient."""
    kind = cfg.get("reward", "contact")
    if kind == "contact":
        return None                       # run_variants' default
    from steering.rewards import ClusterDiversity, TMDiversity
    if kind == "tmsc":                    # Table 1's TM-Sc.: mean pairwise TM, no clustering
        from proteina import tm_matrix
        return lambda c: TMDiversity(tm_matrix())
    # foldseek_clusters rmtree's its tmp dir on entry, so concurrent jobs sharing one work dir
    # delete each other's intermediates -- or, worse, cluster each other's PDBs.
    work = os.path.join(cfg["out_dir"], f"fk_{kind}{cfg.get('tag', '')}_r{cfg['n_res']}",
                        "_".join(which) or "all")
    os.makedirs(work, exist_ok=True)
    lab = foldseek_labeller(work) if kind == "foldseek" else tm_labeller()
    return lambda c: ClusterDiversity(lab)


def run(cfg, *which):
    cfg = dict(cfg, h=resolve_h(cfg))
    return run_variants(cfg, _table(cfg), which, mk_cfg, SteererF, _pot_fn(cfg, which))


def evaluate(cfg, *variants):
    return steered_eval(cfg, *(variants or ("native",) + tuple(_table(cfg))[1:]))


def plot(cfg, *modes):
    from utils import pareto_fk
    return pareto_fk(cfg, *modes)


def plot_cases(cfg, which="length", *lengths):
    from utils import pareto_cases
    return pareto_cases(cfg, which, *lengths)


STAGES = {"calibrate": lambda cfg: calibrate(cfg, mk_cfg, SteererF), "run": run,
          "eval": evaluate, "plot": plot,
          "plot_cases": plot_cases}


if __name__ == "__main__":
    main(STAGES, sys.argv[1:])
