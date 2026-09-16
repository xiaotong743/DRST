"""Mutation tests: do the other tests actually discriminate?"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# (id, file under src/steering, exact text to replace, replacement, test that must fail)
MUTATIONS = [
    ("steering-term-deleted", "sde.py",
     "        drift = base + steer_term", "        drift = base",
     "tests/test_sde.py::test_steered_positions_match_an_independent_integrator"),
    ("steering-folded-into-v", "sde.py",
     "        drift = base + steer_term",
     "        drift = v_d + steer_term + eps_t * model.score(x, v_d + steer_term, t_val)",
     "tests/test_sde.py::test_steered_positions_match_an_independent_integrator"),
    ("score-doubled", "sde.py",
     "        base = v_d + eps_t * s_score", "        base = v_d + 2.0 * eps_t * s_score",
     "tests/test_sde.py::test_lambda_zero_is_bitwise_euler_maruyama"),
    ("picard-increment-zeroed", "ips.py",
     "        inc = dt * (psidot + bgrad)", "        inc = torch.zeros_like(a0)",
     "tests/test_ips.py::test_picard_increment_clip_binds"),
    ("picard-uses-a-k-not-A-t", "ips.py",
     "        a_tilde = a0 + inc", "        a_tilde = a + inc",
     "tests/test_ips.py::test_picard_fixed_point_is_damping_invariant"),
    ("alg3-psidot-zeroed", "alg3.py",
     "    psidot = torch.linalg.solve(a_mat, r)",
     "    psidot = torch.zeros_like(r)",
     "tests/test_alg3.py::test_implicit_psidot_matches_a_converged_picard_at_the_same_state"),
    ("alg3-sign-of-A-flipped", "alg3.py",
     "    a_mat = torch.eye(n, dtype=phi.dtype, device=phi.device) - w[None, :] * phi",
     "    a_mat = torch.eye(n, dtype=phi.dtype, device=phi.device) + w[None, :] * phi",
     "tests/test_alg3.py::test_implicit_psidot_matches_a_converged_picard_at_the_same_state"),
    # phi_grad_b is identically zero for a POINTWISE reward (Phi == 0), so only a target with a
    # genuine measure dependence can see this term go missing.
    ("alg3-phi-grad-b-dropped", "alg3.py",
     "    r = dpsi_dt + phi_grad_b + phi @ (w * bgrad)", "    r = dpsi_dt + phi @ (w * bgrad)",
     "tests/test_alg3.py::test_implicit_psidot_matches_a_converged_picard_at_the_same_state"),
    ("phi-tilde-sign-flipped", "sde.py",
     "        return -2.0 * (k - (k @ w.to(k.dtype))[:, None])",
     "        return 2.0 * (k - (k @ w.to(k.dtype))[:, None])",
     "tests/test_alg3.py::test_phi_tilde_is_the_second_variation"),
    ("retained-normalised-per-set", "sde.py",
     '            w_comb = normalise(torch.cat([retained["a"].to(a.dtype), a]))',
     '            w_comb = 0.5 * torch.cat([normalise(retained["a"].to(a.dtype)), normalise(a)])',
     "tests/test_alg2.py::test_retained_block_is_not_normalised_on_its_own"),
    ("retained-dropped-from-psi", "sde.py",
     "        psi_q = potential.psi(theta, sup, w_comb)                # measure held fixed",
     "        psi_q = potential.psi(theta, theta.detach(), w)          # measure held fixed",
     "tests/test_alg2.py::test_retained_measure_reaches_grad_x_psi_on_the_very_first_step"),
    ("rbf-gram-expanded-again", "mmd.py",
     "    d2 = (a[:, None, :] - b[None, :, :]).pow(2).sum(-1)",
     "    d2 = ((a * a).sum(-1)[:, None] + (b * b).sum(-1)[None, :]\n"
     "          - 2.0 * (a @ b.transpose(-1, -2))).clamp_min(0.0)",
     "tests/test_mmd.py::test_rbf_gram_is_accurate_where_the_expansion_is_not"),
    ("mmd2-weights-unnormalised", "mmd.py",
     "    return w / w.sum()", "    return w",
     "tests/test_mmd.py::test_mmd2_invariant_to_weight_renormalisation"),
]


@pytest.mark.slow
@pytest.mark.parametrize("name,fname,old,new,target", MUTATIONS,
                         ids=[m[0] for m in MUTATIONS])
def test_mutation_is_caught(name, fname, old, new, target):
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        shutil.copytree(ROOT / "src", tmp / "src")
        shutil.copytree(ROOT / "tests", tmp / "tests",
                        ignore=shutil.ignore_patterns("__pycache__", "test_mutations.py"))
        f = tmp / "src" / "steering" / fname
        src = f.read_text()
        assert src.count(old) == 1, (
            f"{name}: anchor text appears {src.count(old)} times in {fname}; the mutation list is "
            f"stale relative to the source")
        f.write_text(src.replace(old, new))
        r = subprocess.run([sys.executable, "-m", "pytest", target, "-q", "-x", "--no-header"],
                           cwd=tmp, capture_output=True, text=True,
                           env={"PYTHONPATH": str(tmp / "src"), "PATH": "/usr/bin:/bin"})
        assert r.returncode != 0, (
            f"{name}: {target} PASSED against the mutated source, so it does not test what it "
            f"claims.\n{r.stdout[-2500:]}")
