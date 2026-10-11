"""The flow-map student's schedule algebra, on CPU and without the Proteina checkpoint."""
import math

import pytest
import torch

from steering.expmap import DTAU, GT, TAU, ExpStep, abs_u, coeffs_em, coeffs_exp

B0_EXPECTED = 465.1687          # (1/0.01) * 2 ln10 / 0.99, the stiffness the design rests on


def _col(a, n=1):
    return torch.full((n, 1, 1), float(a), dtype=torch.float64)


def test_tau_is_proteinas_log_schedule():
    """TAU is get_schedule(mode='log', p1=2) in closed form (r3n_fm.py:648-654)."""
    t = 1.0 - torch.logspace(-2.0, 0.0, 401, dtype=torch.float64).flip(0)
    t = (t - t.min()) / t.max()
    u = torch.arange(401, dtype=torch.float64) / 400.0
    assert torch.allclose(TAU(u), t, atol=1e-15)


def test_dtau_is_taus_derivative():
    u = torch.linspace(0.0, 0.99, 50, dtype=torch.float64)
    e = 1e-7
    assert torch.allclose((TAU(u + e) - TAU(u - e)) / (2 * e), DTAU(u), rtol=1e-6)


def test_gt_is_one_over_t_with_proteinas_ode_tail():
    """g = 1/(t+1e-2) at gt_p=1, and zero past 0.99 where r3n_fm.py:523 switches to the ODE."""
    t = torch.tensor([0.0, 0.5, 0.989, 0.991], dtype=torch.float64)
    assert torch.allclose(GT(t)[:3], 1.0 / (t[:3] + 1e-2))
    assert float(GT(t)[3]) == 0.0


def test_stiffness_at_u_zero():
    """B(0) = 465.17 is what rules out an additive-drift student below K = 233."""
    assert abs(float(abs_u(_col(0.0), 1.0)[1]) - B0_EXPECTED) < 1e-3


@pytest.mark.parametrize("K", [1, 16, 64, 128, 256])
def test_em_multiplier_blows_up_below_the_threshold(K):
    """|E| > 1 is the instability; it must appear exactly where B(0)/2 says it does."""
    e = float(coeffs_em(_col(0.0), _col(1.0 / K), 1.0)[0])
    assert e == pytest.approx(1.0 - B0_EXPECTED / K, abs=1e-3)
    assert (abs(e) > 1.0) == (K < B0_EXPECTED / 2)


@pytest.mark.parametrize("K", [1, 16, 64, 128, 256])
def test_exp_multiplier_is_contractive_at_every_step_count(K):
    """exp(-int B) lies in (0, 1], so the exponential student is stable even at one step."""
    e = float(coeffs_exp(_col(0.0), _col(1.0 / K), 1.0)[0])
    assert 0.0 < e <= 1.0


def test_exp_coefficients_converge_in_the_quadrature():
    """S2 is what sets q: it is the last of the five to settle, by a factor of 4 in q."""
    s, h = _col(0.1), _col(0.05)
    a = [float(x) for x in coeffs_exp(s, h, 1.0)]            # the default q
    b = [float(x) for x in coeffs_exp(s, h, 1.0, q=1024)]
    assert all(abs(x - y) < 1e-3 * max(1.0, abs(y)) for x, y in zip(a, b))


def test_untrained_em_student_reproduces_one_euler_maruyama_step():
    """Gate 1 on CPU: at init='em' the zero-initialised student IS the EM step it regresses on."""
    from steering.flowmap import EulerMaruyamaFlowMap, sample_levy
    torch.manual_seed(0)
    n, L = 4, 6
    y = torch.randn(n, L, 3, dtype=torch.float64)
    vhat = torch.randn(n, L, 3, dtype=torch.float64)
    s = _col(0.3, n)
    h = _col(0.01, n)
    I = sample_levy(h.expand_as(y), 3)
    fm = EulerMaruyamaFlowMap(ExpStep(lambda ys, ss: vhat, init="em", gamma=1.0)).double()
    t_, d = TAU(s), DTAU(s)
    g = GT(t_)
    want = (y + (vhat + g * (t_ * vhat - y) / (1.0 - t_)) * (d * h)
            + (2.0 * g * 1.0 * d).sqrt() * I[0])
    assert torch.allclose(fm(y, s, s + h, I), want, atol=1e-12)
