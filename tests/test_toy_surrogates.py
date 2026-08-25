"""Focused mathematical tests for the EXPERIMENTAL / TOY surrogate ablation.

These tests do not alter canonical paper objectives.
"""

from __future__ import annotations

import numpy as np

from soft_constrained_models.boosting_models import canonical_surrogate_scaled_grad_hess
from soft_constrained_models.toy_surrogate_models import (
    CAP_QUANTILE,
    VARIANT_SPECS,
    capped_weights,
    covariance_C,
    exact_profiled_hessian,
    fairness_penalty_gradient_unit_rho_from_ec,
    k_tau_constant,
    penalty_value,
    scaled_objective,
    toy_surrogate_scaled_grad_hess,
)


def _make_ec(seed: int = 11, n: int = 12, shift: float = 0.0):
    rng = np.random.default_rng(seed)
    y_true = rng.normal(12.0, 0.55, size=n)
    y_pred = y_true + rng.normal(0.15, 0.28, size=n) + float(shift)
    y_mean = float(np.mean(y_true))
    e = y_pred - y_true
    c = y_true - y_mean
    return y_true, y_pred, y_mean, e, c


def _fd_grad(e, c, rho, penalty_shape, level_invariant, tau=None, eps=1e-6, mask=None):
    fd = np.zeros_like(e)
    for i in range(e.size):
        if mask is not None and not mask[i]:
            continue
        e_up = e.copy(); e_up[i] += eps
        e_dn = e.copy(); e_dn[i] -= eps
        j_up = scaled_objective(
            e_up, c, rho, penalty_shape=penalty_shape, level_invariant=level_invariant, tau=tau, profile=True
        )
        j_dn = scaled_objective(
            e_dn, c, rho, penalty_shape=penalty_shape, level_invariant=level_invariant, tau=tau, profile=True
        )
        fd[i] = (j_up - j_dn) / (2.0 * eps)
    return fd


def test_all_six_reduce_to_native_l2_at_rho_zero():
    y_true, y_pred, y_mean, e, _c = _make_ec()
    for shape, li in VARIANT_SPECS:
        g, h, _ = toy_surrogate_scaled_grad_hess(
            y_true, y_pred, y_mean=y_mean, rho=0.0, penalty_shape=shape, level_invariant=li
        )
        np.testing.assert_allclose(g, e, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(h, np.ones_like(e), rtol=1e-12, atol=1e-12)


def test_quadratic_fixed_matches_canonical_surrogate():
    y_true, y_pred, y_mean, _e, _c = _make_ec(seed=4, n=15)
    rho = 4.2
    g, h, _ = toy_surrogate_scaled_grad_hess(
        y_true, y_pred, y_mean=y_mean, rho=rho, penalty_shape="quadratic", level_invariant=False
    )
    g0, h0, _ = canonical_surrogate_scaled_grad_hess(y_true, y_pred, y_mean=y_mean, rho=rho)
    np.testing.assert_allclose(g, g0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(h, h0, rtol=1e-12, atol=1e-12)


def test_fixed_quadratic_and_capped_match_finite_differences():
    y_true, y_pred, y_mean, e, c = _make_ec(seed=9, n=10)
    rho = 3.1
    for shape in ("quadratic", "capped_quadratic"):
        tau = None
        if shape == "capped_quadratic":
            _w, tau = capped_weights(c)
        g, h, _ = toy_surrogate_scaled_grad_hess(
            y_true, y_pred, y_mean=y_mean, rho=rho, penalty_shape=shape, level_invariant=False, tau=tau
        )
        fd_g = _fd_grad(e, c, rho, shape, False, tau=tau)
        np.testing.assert_allclose(g, fd_g, rtol=1e-5, atol=1e-6)
        eps = 1e-6
        fd_h = np.zeros_like(e)
        for i in range(e.size):
            e_up = e.copy(); e_up[i] += eps
            e_dn = e.copy(); e_dn[i] -= eps
            j0 = scaled_objective(e, c, rho, penalty_shape=shape, level_invariant=False, tau=tau)
            j_up = scaled_objective(e_up, c, rho, penalty_shape=shape, level_invariant=False, tau=tau)
            j_dn = scaled_objective(e_dn, c, rho, penalty_shape=shape, level_invariant=False, tau=tau)
            fd_h[i] = (j_up - 2.0 * j0 + j_dn) / (eps ** 2)
        np.testing.assert_allclose(h, fd_h, rtol=1e-3, atol=1e-4)


def test_level_invariant_quadratic_and_capped_profiled_gradient_and_hessian_diag():
    y_true, y_pred, y_mean, e, c = _make_ec(seed=21, n=11)
    rho = 2.7
    for shape in ("quadratic", "capped_quadratic"):
        w, tau = (c ** 2, None) if shape == "quadratic" else capped_weights(c)
        g, h, extras = toy_surrogate_scaled_grad_hess(
            y_true, y_pred, y_mean=y_mean, rho=rho, penalty_shape=shape, level_invariant=True, tau=tau
        )
        fd_g = _fd_grad(e, c, rho, shape, True, tau=tau)
        np.testing.assert_allclose(g, fd_g, rtol=1e-5, atol=1e-6)
        H = exact_profiled_hessian(w, rho)
        np.testing.assert_allclose(h, np.diag(H), rtol=1e-12, atol=1e-12)
        assert extras["a_star"] == extras["a_star"]


def test_absolute_subgradients_match_fd_away_from_kinks():
    rng = np.random.default_rng(13)
    y_true = np.linspace(10.5, 13.8, 9)
    e = np.array([-0.41, 0.22, -0.17, 0.33, -0.28, 0.19, -0.36, 0.44, -0.12])
    y_pred = y_true + e
    y_mean = float(np.mean(y_true))
    c = y_true - y_mean
    assert np.all(np.abs(e) > 1e-3)
    rho = 5.0
    g, h, _ = toy_surrogate_scaled_grad_hess(
        y_true, y_pred, y_mean=y_mean, rho=rho, penalty_shape="absolute", level_invariant=False
    )
    fd_g = _fd_grad(e, c, rho, "absolute", False, eps=1e-7)
    np.testing.assert_allclose(g, fd_g, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(h, np.ones_like(e))

    g_li, h_li, extras = toy_surrogate_scaled_grad_hess(
        y_true, y_pred, y_mean=y_mean, rho=rho, penalty_shape="absolute", level_invariant=True
    )
    a_star = float(extras["a_star"])
    away = np.abs(e - a_star) > 1e-3
    assert int(np.sum(away)) >= 6
    fd_li = _fd_grad(e, c, rho, "absolute", True, eps=1e-7, mask=away)
    np.testing.assert_allclose(g_li[away], fd_li[away], rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(h_li, np.ones_like(e))
    assert rng is not None


def test_fixed_level_not_shift_invariant_li_is_invariant():
    _y_true, _y_pred, _y_mean, e, c = _make_ec(seed=8, n=14)
    k = 0.85
    for shape in ("quadratic", "absolute", "capped_quadratic"):
        tau = None if shape != "capped_quadratic" else capped_weights(c)[1]
        psi0 = penalty_value(e, c, penalty_shape=shape, level_invariant=False, tau=tau)["Psi"]
        psi_k = penalty_value(e + k, c, penalty_shape=shape, level_invariant=False, tau=tau)["Psi"]
        assert abs(psi_k - psi0) > 1e-6
        psi_li0 = penalty_value(e, c, penalty_shape=shape, level_invariant=True, tau=tau)["Psi"]
        psi_lik = penalty_value(e + k, c, penalty_shape=shape, level_invariant=True, tau=tau)["Psi"]
        np.testing.assert_allclose(psi_lik, psi_li0, rtol=1e-10, atol=1e-12)


def test_covariance_bounds_quadratic_and_absolute():
    _y_true, _y_pred, _y_mean, e, c = _make_ec(seed=3, n=16)
    C = covariance_C(e, c)
    for li in (False, True):
        psi_q = penalty_value(e, c, penalty_shape="quadratic", level_invariant=li)["Psi"]
        psi_a = penalty_value(e, c, penalty_shape="absolute", level_invariant=li)["Psi"]
        assert (C ** 2) <= psi_q + 1e-12
        assert abs(C) <= psi_a + 1e-12


def test_capped_generalized_bound_and_weight_range():
    _y_true, _y_pred, _y_mean, e, c = _make_ec(seed=5, n=20)
    w, tau = capped_weights(c, quantile=CAP_QUANTILE)
    assert np.all(w >= -1e-15)
    assert np.all(w <= tau ** 2 + 1e-12)
    K = k_tau_constant(c, w)
    C = covariance_C(e, c)
    for li in (False, True):
        psi_c = penalty_value(e, c, penalty_shape="capped_quadratic", level_invariant=li, tau=tau)["Psi"]
        assert (C ** 2) <= K * psi_c + 1e-10


def test_gradients_hessians_finite_and_total_hessians_strictly_positive():
    y_true, y_pred, y_mean, _e, _c = _make_ec(seed=19, n=13)
    for rho in (0.0, 1.0, 30.0, 100.0):
        for shape, li in VARIANT_SPECS:
            g, h, _ = toy_surrogate_scaled_grad_hess(
                y_true, y_pred, y_mean=y_mean, rho=rho, penalty_shape=shape, level_invariant=li
            )
            assert np.all(np.isfinite(g))
            assert np.all(np.isfinite(h))
            assert np.all(h > 0.0)


def test_unit_rho_penalty_gradient_helper_matches_fobj_minus_residual():
    y_true, y_pred, y_mean, e, c = _make_ec(seed=6, n=9)
    for shape, li in VARIANT_SPECS:
        g, _h, _ = toy_surrogate_scaled_grad_hess(
            y_true, y_pred, y_mean=y_mean, rho=1.0, penalty_shape=shape, level_invariant=li
        )
        g_pen = fairness_penalty_gradient_unit_rho_from_ec(e, c, penalty_shape=shape, level_invariant=li)
        np.testing.assert_allclose(g_pen, g - e, rtol=1e-12, atol=1e-12)
