"""Experiment A: reward steering on the frozen Proteina model at its own temperature.

    dX = [b + g_t*s + eps_t*grad_x Psi] dt + sqrt(2 eps_t) dB,   eps_t = tau*g_t

No temperature Feynman-Kac term, so the weights carry only the reward: A does NOT claim to tilt
the low-temperature marginal.

Branch A IS `SteererA` below: three hook overrides on the published Algorithm 1 in
src/steering/steerer.py, which stays the reference. What is left in this file defines the branch and
wires its stages; the machinery they stand on is in common.py and the reporting in utils.py.

Stages, in order:
    calibrate   freeze the reward bandwidth on a native pilot pool at this length
    run         steer N particles at one (gamma, length) and save the cloud
    eval        designability (ProteinMPNN -> ESMFold -> scRMSD) + diversity for that cloud
    pool        pool the 5 lengths into the paper's 500-sample set and compare with Table 1
    pareto      designability against diversity per length, steered on the unsteered gamma curve

Usage: python experiments/steer_a.py configs/steer_a.yaml <stage> [args] [--set.key=value ...]
"""
import sys
from dataclasses import dataclass

import torch

from steering.steerer import Config, Steerer
from common import calibrate, main, resolve_h, run_variants, steered_eval, timing_probe
from utils import gpu_hours, pareto, pareto_reward, steered_pool_diversity


@dataclass
class ConfigA(Config):
    """Algorithm 1's config plus the one thing branch A adds: the sampling temperature."""
    temperature: float = 1.0          # tau == Proteina's sc_scale_noise, a noise-VARIANCE
                                      # multiplier (std = sqrt(2*gt*gamma*dt), r3n_fm.py:332)


class SteererA(Steerer):
    """Branch A: reward steering at the model's own sampling temperature.

    The ONLY change from Algorithm 1 is where tau lands. Proteina's eq (3) is

        dx = [v + g(t) s] dt + sqrt(2 g(t) gamma) dW

    so the score keeps the RAW g_t while the noise carries gamma*g_t; the reward gradient goes
    with the noise, since eq (12a)'s coefficient is the diffusion's. At tau = 1 both hooks
    return exactly what Algorithm 1 returns, bitwise.

    A adds NO weight term: its log-weights carry the reward alone, so A does not claim to tilt
    the low-temperature marginal. Concretely, at tau != 1 the score keeps g while the diffusion is
    tau*g, so eq (12b) is short by (g - tau*g)*[lap(p)/p + s.grad Psi].
    """

    def check(self):
        tau = float(self.cfg.temperature)
        if tau <= 0.0:
            raise ValueError(f"temperature={tau}: must be positive, it multiplies a variance")
        if tau != 1.0 and self.cfg.steer_coeff is not None:
            raise ValueError(
                f"steer_coeff={self.cfg.steer_coeff} at temperature={tau}: steer_coeff replaces "
                f"the steering coefficient and would discard the temperature.")

    def coeffs(self, t_val):
        g = self.model.eps(t_val)
        if t_val > self.cfg.vf_tail_t:
            g = 0.0
        return g, float(self.cfg.temperature) * g

    def noise_std(self, t_val, dt_t):
        g, _ = self.coeffs(t_val)
        if g <= 0:
            return torch.zeros((), dtype=dt_t.dtype)
        # Proteina's own operand order, 2*g*tau*dt. Forming tau*g first differs by 1 ulp on 12
        # of 100 steps at tau=0.45 and 1/(1-t) amplifies it; at tau=1 the orders agree exactly.
        return torch.sqrt(2 * torch.as_tensor(g, dtype=dt_t.dtype)
                          * float(self.cfg.temperature) * dt_t)


def effective_eta(cfg):
    """eta, optionally rescaled so steering authority is comparable ACROSS LENGTHS.

    **[measured]** at fixed eta the steering-to-base drift ratio follows 1/L almost exactly --
    relative to L=50 it is 1.000 / 0.538 / 0.313 / 0.263 / 0.225 at L = 50/100/150/200/250,
    against the 1/L law's 1.000 / 0.500 / 0.333 / 0.250 / 0.200. So one eta is five different
    effective strengths, and the gamma campaign's cross-length comparison was confounded.

    `eta_ref_len` is the length at which `eta` means what it says; eta scales linearly in L
    away from it. null keeps the raw eta, which is what every result on disk used.

    NOT done by renormalising the feature map instead. The obvious fix -- dividing
    SoftContactFeatures by sqrt(active pairs) rather than sqrt(|J|), since contacts are O(L)
    while |J| is O(L^2) -- is provably inert: rescaling the features by c rescales every
    distance by c, and `calibrate_bandwidth` sets h = sqrt(median d2), so h rescales by the same
    c and exp(-d^2/h^2) is unchanged. **[measured]** h'/h = 2.238010 against a predicted c of
    2.238010 at L=50 and 4.300300 / 4.300300 at L=250, with psi identical to 1e-7. The divisor
    cancels; only eta is left to carry the length correction.
    """
    ref = cfg.get("eta_ref_len")
    if not ref:
        return cfg["eta"]
    return cfg["eta"] * cfg["n_res"] / float(ref)


def _table(cfg):
    """Variant name -> the config overrides that define it."""
    return {"native": dict(lam=0.0), "A": dict(lam=effective_eta(cfg))}


def mk_cfg(cfg, cls=ConfigA, **extra):
    """A's Config factory, as `run_variants` and `calibrate` want it: cfg -> (mdl, **over) -> cfg.

    `cls` and `extra` are how a derived branch extends it without this file naming its fields.
    """
    def mk(mdl, **over):
        kw = dict(ts=mdl.ts, temperature=cfg["temperature"], k_fp=cfg["k_fp"],
                  damp=cfg["damp"], endpoint_lag=cfg["endpoint_lag"],
                  vf_tail_t=cfg["vf_tail_t"], resample_every=cfg["resample_every"],
                  grad_chunk=cfg.get("grad_chunk"), check_drift=True, global_rng=True)
        kw.update(extra)
        kw.update(over)          # the variant's own settings win
        return cls(**kw)
    return mk


def run(cfg, *which):
    """Steer at one (gamma, length). `which` selects from the variant table; default both."""
    cfg = dict(cfg, h=resolve_h(cfg))
    return run_variants(cfg, _table(cfg), which, mk_cfg, SteererA)


def evaluate(cfg, *variants):
    return steered_eval(cfg, *(variants or ("native", "A")))


STAGES = {"calibrate": lambda cfg: calibrate(cfg, mk_cfg, SteererA), "run": run,
          "eval": evaluate, "pool": steered_pool_diversity, "pareto": pareto, "pareto_reward": pareto_reward, "gpu_hours": gpu_hours,
          "timing": timing_probe}


if __name__ == "__main__":
    main(STAGES, sys.argv[1:])
