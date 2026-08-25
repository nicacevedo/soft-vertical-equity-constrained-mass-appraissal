"""Focused theory tests for the EXPERIMENTAL / TOY hybrid corrections."""

from __future__ import annotations

import numpy as np

from soft_constrained_models.boosting_models import (
    canonical_direct_exact_scaled_hessian,
    canonical_direct_scaled_grad_hess,
    canonical_surrogate_scaled_grad_hess,
)
from soft_constrained_models.toy_hybrid_objectives import (
    QD_HESS_EXACT,
    QNL_CURVATURE,
    guardrail_d23,
    majorizer_gap,
    quadratic_direct_cap_exact_hessian,
    quadratic_direct_cap_scaled_grad_hess,
    quadratic_direct_cap_scaled_objective,
    quadratic_nl_guardrail_exact_hessian,
    quadratic_nl_guardrail_scaled_grad_hess,
    quadratic_nl_guardrail_scaled_objective,
)
from soft_constrained_models.toy_mechanism_objectives import build_training_moment_basis


def _synth(n: int = 24, seed: int = 3):
    rng = np.random.default_rng(seed)
    y = rng.normal(12.0, 0.4, size=n)
    basis = build_training_moment_basis(y)
    e = rng.normal(0.0, 0.2, size=n)
    return basis, e, rng


def test_qd_lambda0_matches_quadratic():
    basis, e, _ = _synth()
    c = basis["c_train"]
    alpha = 1.7
    g, h, _ = quadratic_direct_cap_scaled_grad_hess(e, c, alpha=alpha, lam=0.0)
    y_true = c.copy()
    y_pred = e + c
    g_q, h_q, _ = canonical_surrogate_scaled_grad_hess(y_true, y_pred, y_mean=0.0, rho=alpha)
    np.testing.assert_allclose(g, g_q, atol=1e-12)
    np.testing.assert_allclose(h, h_q, atol=1e-12)


def test_qd_alpha0_matches_canonical_direct():
    basis, e, _ = _synth()
    c = basis["c_train"]
    lam = 2.4
    g, h, extra = quadratic_direct_cap_scaled_grad_hess(e, c, alpha=0.0, lam=lam)
    y_true = c.copy()
    y_pred = e + c
    g_d, h_d, extra_d = canonical_direct_scaled_grad_hess(y_true, y_pred, y_mean=0.0, rho=lam)
    np.testing.assert_allclose(g, g_d, atol=1e-12)
    np.testing.assert_allclose(h, h_d, atol=1e-12)
    np.testing.assert_allclose(extra["C"], extra_d["C"], atol=1e-12)
    H = quadratic_direct_cap_exact_hessian(c, alpha=0.0, lam=lam)
    np.testing.assert_allclose(H, canonical_direct_exact_scaled_hessian(c, lam), atol=1e-12)


def test_qd_combined_finite_differences_and_exact_hessian():
    basis, e, _ = _synth(n=12, seed=8)
    c = basis["c_train"]
    alpha, lam = 1.3, 2.1
    g, h, _ = quadratic_direct_cap_scaled_grad_hess(e, c, alpha=alpha, lam=lam)
    step = 1e-6
    fd = np.zeros_like(e)
    for i in range(e.size):
        up = e.copy(); up[i] += step
        dn = e.copy(); dn[i] -= step
        fd[i] = (
            quadratic_direct_cap_scaled_objective(up, c, alpha=alpha, lam=lam)
            - quadratic_direct_cap_scaled_objective(dn, c, alpha=alpha, lam=lam)
        ) / (2 * step)
    np.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-6)
    H = quadratic_direct_cap_exact_hessian(c, alpha=alpha, lam=lam)
    expected_diag = 1.0 + alpha * np.square(c) + (lam / (2.0 * e.size)) * np.square(c)
    np.testing.assert_allclose(np.diag(H), expected_diag, atol=1e-12)
    np.testing.assert_allclose(h, expected_diag, atol=1e-12)
    assert QD_HESS_EXACT.startswith("H = I")
    # Hessian-vector: finite difference of gradient
    rng = np.random.default_rng(8)
    v = rng.normal(size=e.size)
    g_up, _, _ = quadratic_direct_cap_scaled_grad_hess(e + 1e-6 * v, c, alpha=alpha, lam=lam)
    g_dn, _, _ = quadratic_direct_cap_scaled_grad_hess(e - 1e-6 * v, c, alpha=alpha, lam=lam)
    hv = (g_up - g_dn) / (2e-6)
    np.testing.assert_allclose(hv, H @ v, rtol=1e-4, atol=1e-5)


def test_qnl_gamma0_matches_quadratic():
    basis, e, _ = _synth()
    c = basis["c_train"]
    phi = basis["phi_train"]
    d23 = guardrail_d23(phi)
    rho = 2.075
    g, h, extra = quadratic_nl_guardrail_scaled_grad_hess(e, c, phi, d23, rho=rho, gamma=0.0)
    y_true = c.copy()
    y_pred = e + c
    g_q, h_q, _ = canonical_surrogate_scaled_grad_hess(y_true, y_pred, y_mean=0.0, rho=rho)
    np.testing.assert_allclose(g, g_q, atol=1e-12)
    np.testing.assert_allclose(h, h_q, atol=1e-12)
    assert extra["curvature"] == QNL_CURVATURE


def test_qnl_finite_differences_and_exact_hessian():
    basis, e, _ = _synth(n=14, seed=11)
    c = basis["c_train"]
    phi = basis["phi_train"]
    d23 = guardrail_d23(phi)
    rho, gamma = 1.4, 2.2
    g, h, extra = quadratic_nl_guardrail_scaled_grad_hess(e, c, phi, d23, rho=rho, gamma=gamma)
    step = 1e-6
    fd = np.zeros_like(e)
    for i in range(e.size):
        up = e.copy(); up[i] += step
        dn = e.copy(); dn[i] -= step
        fd[i] = (
            quadratic_nl_guardrail_scaled_objective(up, c, phi, rho=rho, gamma=gamma)
            - quadratic_nl_guardrail_scaled_objective(dn, c, phi, rho=rho, gamma=gamma)
        ) / (2 * step)
    np.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-6)
    H = quadratic_nl_guardrail_exact_hessian(c, phi, rho=rho, gamma=gamma)
    rng = np.random.default_rng(11)
    v = rng.normal(size=e.size)
    g_up, _, _ = quadratic_nl_guardrail_scaled_grad_hess(e + 1e-6 * v, c, phi, d23, rho=rho, gamma=gamma)
    g_dn, _, _ = quadratic_nl_guardrail_scaled_grad_hess(e - 1e-6 * v, c, phi, d23, rho=rho, gamma=gamma)
    hv = (g_up - g_dn) / (2e-6)
    np.testing.assert_allclose(hv, H @ v, rtol=1e-4, atol=1e-5)
    assert np.all(np.isfinite(h))
    assert np.all(h > 0.0)
    assert np.all(d23 >= -1e-15)
    assert extra["M23"] >= 0.0


def test_d23_majorizer_psd_many_directions():
    basis, _e, rng = _synth(n=80, seed=21)
    phi = basis["phi_train"]
    d23 = guardrail_d23(phi)
    assert np.all(np.isfinite(d23))
    assert np.all(d23 >= -1e-15)
    worst = 0.0
    for seed in range(80):
        x = rng.normal(size=phi.shape[0])
        gap = majorizer_gap(phi, d23, x)
        worst = min(worst, gap) if gap < 0 else worst
        assert gap <= 1e-8, gap
    # Tightness: x = sign(phi2) can be near equality for the k=2 term alone.
    x_sign = np.sign(phi[:, 1])
    x_sign[x_sign == 0] = 1.0
    gap_sign = majorizer_gap(phi, d23, x_sign)
    assert gap_sign <= 1e-8
