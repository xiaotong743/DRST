"""Mutation tests: do the other tests actually discriminate?

Each entry breaks one line of source and names a test that MUST then fail. Two rules keep the
file honest, both learned the hard way when the A/B split moved code out from under it:

  * the anchor must match its file exactly once -- asserted below, so a stale entry errors
    rather than quietly testing nothing;
  * the REPLACEMENT must still be valid code. `steering-folded-into-v` once substituted in a
    variable (`eps_t`) that a refactor had renamed, so the mutant died of NameError and the
    target "failed" without ever discriminating. A mutation that crashes the module proves
    nothing about the test pointed at it.

Branch code in experiments/ is mutated too: the temperature split lives on SteererA, so experiments/ is
copied into the sandbox alongside src/ and tests/conftest.py puts the COPY on the path.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# (id, file in src/steering/ or experiments/, exact text to replace, replacement, failing test)
MUTATIONS = [
    ("steering-term-deleted", "steerer.py",
     "        drift = base + steer_term", "        drift = base",
     "tests/test_sde.py::test_steered_positions_match_an_independent_integrator"),
    ("steering-folded-into-v", "steerer.py",
     "        drift = base + steer_term",
     "        drift = v_d + steer_term + c_score * model.score(x, v_d + steer_term, t_val)",
     "tests/test_sde.py::test_steered_positions_match_an_independent_integrator"),
    ("score-doubled", "steerer.py",
     "        base = v_d + c_score * model.score(x, v_d, t_val)",
     "        base = v_d + 2.0 * c_score * model.score(x, v_d, t_val)",
     "tests/test_sde.py::test_lambda_zero_is_bitwise_euler_maruyama"),
    ("picard-increment-zeroed", "meanfield.py",
     "        inc = dt * (psidot + bgrad)", "        inc = torch.zeros_like(a0)",
     "tests/test_smc.py::test_picard_increment_clip_binds"),
    ("picard-uses-a-k-not-A-t", "meanfield.py",
     "        a_tilde = a0 + inc", "        a_tilde = a + inc",
     "tests/test_smc.py::test_picard_fixed_point_is_damping_invariant"),
    ("implicit-psidot-zeroed", "meanfield.py",
     "    psidot = torch.linalg.solve(a_mat, r)",
     "    psidot = torch.zeros_like(r)",
     "tests/test_meanfield.py::test_implicit_psidot_matches_a_converged_picard_at_the_same_state"),
    ("implicit-sign-of-A-flipped", "meanfield.py",
     "    a_mat = torch.eye(n, dtype=phi.dtype, device=phi.device) - w[None, :] * phi",
     "    a_mat = torch.eye(n, dtype=phi.dtype, device=phi.device) + w[None, :] * phi",
     "tests/test_meanfield.py::test_implicit_psidot_matches_a_converged_picard_at_the_same_state"),
    # phi_grad_b is identically zero for a POINTWISE reward (Phi == 0), so only a target with a
    # genuine measure dependence can see this term go missing.
    ("implicit-phi-grad-b-dropped", "meanfield.py",
     "    r = dpsi_dt + phi_grad_b + phi @ (w * bgrad)", "    r = dpsi_dt + phi @ (w * bgrad)",
     "tests/test_meanfield.py::test_implicit_psidot_matches_a_converged_picard_at_the_same_state"),
    ("phi-tilde-sign-flipped", "rewards.py",
     "        return -2.0 * (k - (k @ w.to(k.dtype))[:, None])",
     "        return 2.0 * (k - (k @ w.to(k.dtype))[:, None])",
     "tests/test_meanfield.py::test_phi_tilde_is_the_second_variation"),
    ("retained-normalised-per-set", "steerer.py",
     '            w_comb = normalise(torch.cat([retained["a"].to(a.dtype), a]))',
     '            w_comb = 0.5 * torch.cat([normalise(retained["a"].to(a.dtype)), normalise(a)])',
     "tests/test_alg2.py::test_retained_block_is_not_normalised_on_its_own"),
    # The retained measure has to reach grad_x Psi, which now happens in the microbatched pass.
    # Without a retained measure `sup` IS the current block, so this is a no-op except under
    # Algorithm 2 -- which is exactly the test it is pointed at.
    ("retained-dropped-from-psi", "steerer.py",
     "            psi_c = self.potential.psi(th_c, sup, w_comb)",
     "            psi_c = self.potential.psi(th_c, sup[-self.n:], "
     "w_comb[-self.n:] / w_comb[-self.n:].sum())",
     "tests/test_alg2.py::test_retained_measure_reaches_grad_x_psi_on_the_very_first_step"),
    ("rbf-gram-expanded-again", "rewards.py",
     "    d2 = (a[:, None, :] - b[None, :, :]).pow(2).sum(-1)",
     "    d2 = ((a * a).sum(-1)[:, None] + (b * b).sum(-1)[None, :]\n"
     "          - 2.0 * (a @ b.transpose(-1, -2))).clamp_min(0.0)",
     "tests/test_mmd.py::test_rbf_gram_is_accurate_where_the_expansion_is_not"),
    # Anchored on _uniform's two-line tail: the bare `return w / w.sum()` is also _norm's
    # last line, so after the merge it matches twice and the entry errors on its own guard.
    ("mmd2-weights-unnormalised", "rewards.py",
     "        return x.new_full((x.shape[0],), 1.0 / x.shape[0])\n    return w / w.sum()",
     "        return x.new_full((x.shape[0],), 1.0 / x.shape[0])\n    return w",
     "tests/test_mmd.py::test_mmd2_invariant_to_weight_renormalisation"),
    # --- temperature. At tau = 1 both are invisible, which is the point.
    ("temperature-moved-onto-the-score", "steer_a.py",
     "        return g, float(self.cfg.temperature) * g",
     "        return float(self.cfg.temperature) * g, float(self.cfg.temperature) * g",
     "tests/test_temperature.py::"
     "test_the_score_keeps_raw_g_while_noise_and_steering_carry_tau_g"),
    ("noise-std-ignores-temperature", "steer_a.py",
     "        return torch.sqrt(2 * torch.as_tensor(g, dtype=dt_t.dtype)\n"
     "                          * float(self.cfg.temperature) * dt_t)",
     "        return torch.sqrt(2 * torch.as_tensor(g, dtype=dt_t.dtype)\n"
     "                          * dt_t)",
     "tests/test_temperature.py::"
     "test_the_score_keeps_raw_g_while_noise_and_steering_carry_tau_g"),
    # --- FK potential weights
    ("fk-terminal-correction-dropped", "steerer.py",
     "            cand = cfg.lam * f(normalise(a)).double() - self._logP",
     "            cand = cfg.lam * f(normalise(a)).double()",
     "tests/test_fk.py::test_resampling_preserves_the_target"),
    ("fk-logP-reset-on-resampling", "steerer.py",
     "        self._logP = (self._logP + a)[idx]      # ancestral, never reset",
     "        self._logP = torch.zeros_like(a)",
     "tests/test_fk.py::test_resampling_preserves_the_target"),
    ("fk-max-is-not-a-running-max", "steerer.py",
     "        r_prev = torch.maximum(r, r_prev)", "        r_prev = r",
     "tests/test_fk.py::test_fk_weights_is_the_potential_it_claims"),
    ("fk-diff-sign-flipped", "steerer.py",
     "        return a + lam * (r - r_prev), r", "        return a - lam * (r - r_prev), r",
     "tests/test_fk.py::test_resampling_moves_the_population_toward_high_reward"),
    # --- contact reward
    ("contact-psi-sign-flipped", "rewards.py",
     "        return -2.0 * (k @ _norm(w.detach(), k))",
     "        return 2.0 * (k @ _norm(w.detach(), k))",
     "tests/test_protein_rewards.py::test_psi_is_the_first_variation_of_the_functional"),
    ("contact-features-unnormalised", "rewards.py",
     "        return torch.sigmoid((self.cutoff - dist) / self.width) / math.sqrt(i.numel())",
     "        return torch.sigmoid((self.cutoff - dist) / self.width)",
     "tests/test_protein_rewards.py::"
     "test_squared_feature_distance_is_the_mean_squared_contact_difference"),
    # Microbatching must split the VJP, not the measure. Chunking the support/weights alongside
    # the query is the natural slip and it makes grad_x Psi depend on grad_chunk.
    ("microbatch-splits-the-measure", "steerer.py",
     "            psi_c = self.potential.psi(th_c, sup, w_comb)",
     "            psi_c = self.potential.psi(th_c, sup[lo:hi], "
     "w_comb[lo:hi] / w_comb[lo:hi].sum())",
     "tests/test_temperature.py::test_grad_x_psi_is_microbatch_size_invariant"),
    ("contact-weights-fn-ignores-its-argument", "rewards.py",
     "        return lambda w: -2.0 * (k @ _norm(w, k))",
     "        return lambda w: -2.0 * (k @ _norm(torch.ones(k.shape[1]), k))",
     "tests/test_protein_rewards.py::test_weights_fn_reacts_to_the_candidate_weights"),
]


@pytest.mark.slow
@pytest.mark.parametrize("name,fname,old,new,target", MUTATIONS,
                         ids=[m[0] for m in MUTATIONS])
def test_mutation_is_caught(name, fname, old, new, target):
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        shutil.copytree(ROOT / "src", tmp / "src",
                        ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(ROOT / "experiments", tmp / "experiments",
                        ignore=shutil.ignore_patterns("__pycache__", "out"))
        shutil.copytree(ROOT / "tests", tmp / "tests",
                        ignore=shutil.ignore_patterns("__pycache__", "test_mutations.py"))
        # conftest.py adds <rootdir>/experiments to sys.path, so the COPY is what gets imported
        f = next(c for c in (tmp / "src" / "steering" / fname, tmp / "experiments" / fname)
                 if c.exists())
        src = f.read_text()
        assert src.count(old) == 1, (
            f"{name}: anchor text appears {src.count(old)} times in {fname}; the mutation list is "
            f"stale relative to the source")
        f.write_text(src.replace(old, new))
        r = subprocess.run([sys.executable, "-m", "pytest", target, "-q", "-x", "--no-header"],
                           cwd=tmp, capture_output=True, text=True,
                           env={"PYTHONPATH": str(tmp / "src"), "PATH": "/usr/bin:/bin"})
        assert r.returncode == 1, (
            f"{name}: expected {target} to FAIL (exit 1) against the mutated source; got exit "
            f"{r.returncode}. Exit 4 means the target could not even be imported, which would "
            f"self-certify as 'caught'.\n{r.stdout[-2500:]}")
