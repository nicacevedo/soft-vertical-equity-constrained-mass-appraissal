"""Finite-difference and invariance tests for the paper v6 canonical objectives."""

from __future__ import annotations

import numpy as np

from soft_constrained_models.boosting_models import (
    LGBCovPenalty,
    LGBSmoothPenalty,
    canonical_direct_exact_scaled_hessian,
    canonical_direct_scaled_grad_hess,
    canonical_surrogate_scaled_grad_hess,
)


def _scaled_direct_objective(e, c, rho):
    n = float(e.size)
    C = float(np.mean(e * c))
    return 0.5 * float(np.sum(e ** 2)) + (n * float(rho) / 4.0) * (C ** 2)


def _scaled_surrogate_objective(e, c, rho):
    return 0.5 * float(np.sum((e ** 2) * (1.0 + float(rho) * (c ** 2))))


def test_direct_gradient_matches_finite_differences():
    rng = np.random.default_rng(7)
    y_true = rng.normal(12.0, 0.4, size=11)
    y_pred = y_true + rng.normal(0.0, 0.2, size=11)
    y_mean = float(np.mean(y_true))
    rho = 3.7
    grad, hess, extras = canonical_direct_scaled_grad_hess(y_true, y_pred, y_mean=y_mean, rho=rho)
    c = y_true - y_mean
    e = y_pred - y_true
    eps = 1e-6
    fd = np.zeros_like(e)
    for i in range(e.size):
        e_up = e.copy(); e_up[i] += eps
        e_dn = e.copy(); e_dn[i] -= eps
        fd[i] = (_scaled_direct_objective(e_up, c, rho) - _scaled_direct_objective(e_dn, c, rho)) / (2 * eps)
    np.testing.assert_allclose(grad, fd, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(extras["C"], float(np.mean(e * c)))


def test_direct_exact_hessian_is_diagonal_plus_rank_one():
    rng = np.random.default_rng(3)
    c = rng.normal(0.0, 1.0, size=8)
    c = c - float(np.mean(c))
    rho = 5.0
    H = canonical_direct_exact_scaled_hessian(c, rho)
    n = c.size
    expected = np.eye(n) + (rho / (2.0 * n)) * np.outer(c, c)
    np.testing.assert_allclose(H, expected, rtol=1e-12, atol=1e-12)
    # Squared-error block is I; covariance curvature is rank-one.
    cov_curvature = H - np.eye(n)
    assert np.linalg.matrix_rank(cov_curvature, tol=1e-10) == 1


def test_supplied_lightgbm_hessian_is_exact_diagonal():
    rng = np.random.default_rng(4)
    y_true = rng.normal(11.5, 0.3, size=9)
    y_pred = y_true + rng.normal(0.0, 0.15, size=9)
    y_mean = float(np.mean(y_true))
    rho = 2.5
    _grad, hess, _ = canonical_direct_scaled_grad_hess(y_true, y_pred, y_mean=y_mean, rho=rho)
    H = canonical_direct_exact_scaled_hessian(y_true - y_mean, rho)
    np.testing.assert_allclose(hess, np.diag(H), rtol=1e-12, atol=1e-12)


def test_surrogate_gradient_and_hessian_match_finite_differences():
    rng = np.random.default_rng(11)
    y_true = rng.normal(12.2, 0.5, size=10)
    y_pred = y_true + rng.normal(0.0, 0.25, size=10)
    y_mean = float(np.mean(y_true))
    rho = 4.2
    grad, hess, _ = canonical_surrogate_scaled_grad_hess(y_true, y_pred, y_mean=y_mean, rho=rho)
    c = y_true - y_mean
    e = y_pred - y_true
    eps = 1e-6
    fd_g = np.zeros_like(e)
    fd_h = np.zeros_like(e)
    for i in range(e.size):
        e_up = e.copy(); e_up[i] += eps
        e_dn = e.copy(); e_dn[i] -= eps
        fd_g[i] = (_scaled_surrogate_objective(e_up, c, rho) - _scaled_surrogate_objective(e_dn, c, rho)) / (2 * eps)
        fd_h[i] = (
            _scaled_surrogate_objective(e_up, c, rho)
            - 2.0 * _scaled_surrogate_objective(e, c, rho)
            + _scaled_surrogate_objective(e_dn, c, rho)
        ) / (eps ** 2)
    np.testing.assert_allclose(grad, fd_g, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(hess, fd_h, rtol=1e-4, atol=1e-5)


def test_rho_zero_invariants_and_identical_families():
    rng = np.random.default_rng(1)
    y_true = rng.normal(12.0, 0.4, size=13)
    y_pred = y_true + rng.normal(0.0, 0.3, size=13)
    y_mean = float(np.mean(y_true))
    e = y_pred - y_true
    g_d, h_d, _ = canonical_direct_scaled_grad_hess(y_true, y_pred, y_mean=y_mean, rho=0.0)
    g_s, h_s, _ = canonical_surrogate_scaled_grad_hess(y_true, y_pred, y_mean=y_mean, rho=0.0)
    np.testing.assert_allclose(g_d, e, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(h_d, np.ones_like(e), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(g_s, e, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(h_s, np.ones_like(e), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(g_d, g_s, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(h_d, h_s, rtol=1e-12, atol=1e-12)


def test_covariance_invariant_to_constant_residual_shift():
    rng = np.random.default_rng(2)
    y_true = rng.normal(12.0, 0.4, size=15)
    y_pred = y_true + rng.normal(0.0, 0.2, size=15)
    y_mean = float(np.mean(y_true))
    e = y_pred - y_true
    c = y_true - y_mean
    C0 = float(np.mean(e * c))
    C1 = float(np.mean((e + 0.75) * c))
    np.testing.assert_allclose(C1, C0, rtol=1e-12, atol=1e-12)


def test_residual_is_log_price_ratio():
    P = np.array([100.0, 250.0, 800.0])
    Phat = np.array([110.0, 200.0, 960.0])
    y_true = np.log(P)
    y_pred = np.log(Phat)
    e = y_pred - y_true
    np.testing.assert_allclose(e, np.log(Phat / P), rtol=1e-12, atol=1e-12)


def test_class_fobj_uses_canonical_formulas_and_does_not_floor_zero_grad():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = y_true.copy()  # exact fit: residual 0, C=0, gradient must remain 0
    cov = LGBCovPenalty(rho=1.5, ratio_mode="diff", early_stopping_rounds=None, verbose=False)
    cov.y_mean_ = float(np.mean(y_true))
    g, h = cov.fobj(y_true, y_pred)
    np.testing.assert_allclose(g, np.zeros(3), atol=1e-12)
    assert np.all(h > 0)

    surr = LGBSmoothPenalty(
        rho=1.5,
        ratio_mode="diff",
        weighting_proxy_mode="identity",
        early_stopping_rounds=None,
        verbose=False,
    )
    surr.y_mean_ = float(np.mean(y_true))
    g2, h2 = surr.fobj(y_true, y_pred)
    np.testing.assert_allclose(g2, np.zeros(3), atol=1e-12)
    assert np.all(h2 > 0)
