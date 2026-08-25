"""Huber-specific tests for the EXPERIMENTAL / TOY V2 surrogate comparison.

Does not alter canonical paper objectives. Existing six-variant tests remain in
test_toy_surrogates.py.
"""

from __future__ import annotations

import numpy as np

from soft_constrained_models.boosting_models import canonical_surrogate_scaled_grad_hess
from soft_constrained_models.toy_surrogate_models import (
    covariance_C,
    fairness_penalty_gradient_unit_rho_from_ec,
    huber_delta_from_q,
    huber_phi,
    penalty_value,
    scaled_objective,
    toy_surrogate_scaled_grad_hess,
)

V2_FAMILIES = ("quadratic", "absolute", "huber")


def _make_ec(seed: int = 11, n: int = 12):
    rng = np.random.default_rng(seed)
    y_true = rng.normal(12.0, 0.55, size=n)
    y_pred = y_true + rng.normal(0.15, 0.28, size=n)
    y_mean = float(np.mean(y_true))
    e = y_pred - y_true
    c = y_true - y_mean
    return y_true, y_pred, y_mean, e, c


def _huber_delta_for(e, c, quantile=0.80):
    return huber_delta_from_q(e * c, quantile=quantile)


def _fd_grad(e, c, rho, penalty_shape, huber_delta=None, eps=1e-6, mask=None):
    fd = np.zeros_like(e)
    for i in range(e.size):
        if mask is not None and not mask[i]:
            continue
        e_up = e.copy(); e_up[i] += eps
        e_dn = e.copy(); e_dn[i] -= eps
        j_up = scaled_objective(
            e_up, c, rho, penalty_shape=penalty_shape, level_invariant=False, huber_delta=huber_delta
        )
        j_dn = scaled_objective(
            e_dn, c, rho, penalty_shape=penalty_shape, level_invariant=False, huber_delta=huber_delta
        )
        fd[i] = (j_up - j_dn) / (2.0 * eps)
    return fd


def test_v2_three_families_rho_zero_are_native_l2():
    y_true, y_pred, y_mean, e, c = _make_ec()
    delta = _huber_delta_for(e, c)
    for shape in V2_FAMILIES:
        g, h, _ = toy_surrogate_scaled_grad_hess(
            y_true, y_pred, y_mean=y_mean, rho=0.0, penalty_shape=shape, level_invariant=False, huber_delta=delta
        )
        np.testing.assert_allclose(g, e, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(h, np.ones_like(e), rtol=1e-12, atol=1e-12)


def test_v2_quadratic_matches_canonical():
    y_true, y_pred, y_mean, _e, _c = _make_ec(seed=4, n=15)
    rho = 4.2
    g, h, _ = toy_surrogate_scaled_grad_hess(
        y_true, y_pred, y_mean=y_mean, rho=rho, penalty_shape="quadratic", level_invariant=False
    )
    g0, h0, _ = canonical_surrogate_scaled_grad_hess(y_true, y_pred, y_mean=y_mean, rho=rho)
    np.testing.assert_allclose(g, g0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(h, h0, rtol=1e-12, atol=1e-12)


def test_v2_quadratic_finite_differences():
    y_true, y_pred, y_mean, e, c = _make_ec(seed=9, n=10)
    rho = 3.1
    g, h, _ = toy_surrogate_scaled_grad_hess(
        y_true, y_pred, y_mean=y_mean, rho=rho, penalty_shape="quadratic", level_invariant=False
    )
    fd = _fd_grad(e, c, rho, "quadratic")
    np.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-6)
    eps = 1e-6
    fd_h = np.zeros_like(e)
    for i in range(e.size):
        e_up = e.copy(); e_up[i] += eps
        e_dn = e.copy(); e_dn[i] -= eps
        j0 = scaled_objective(e, c, rho, penalty_shape="quadratic", level_invariant=False)
        j_up = scaled_objective(e_up, c, rho, penalty_shape="quadratic", level_invariant=False)
        j_dn = scaled_objective(e_dn, c, rho, penalty_shape="quadratic", level_invariant=False)
        fd_h[i] = (j_up - 2.0 * j0 + j_dn) / (eps ** 2)
    np.testing.assert_allclose(h, fd_h, rtol=1e-3, atol=1e-4)


def test_v2_absolute_fd_away_from_q_zero():
    c = np.array([-1.15, -0.82, -0.54, -0.28, 0.31, 0.52, 0.76, 1.05, 1.27])
    c = c - float(np.mean(c))
    e = np.array([-0.41, 0.22, -0.17, 0.33, -0.28, 0.19, -0.36, 0.44, -0.12])
    y_true = c.copy()
    y_pred = y_true + e
    y_mean = 0.0
    q = e * c
    assert np.all(np.abs(c) > 0.05)
    assert np.all(np.abs(q) > 1e-4)
    rho = 5.0
    g, h, _ = toy_surrogate_scaled_grad_hess(
        y_true, y_pred, y_mean=y_mean, rho=rho, penalty_shape="absolute", level_invariant=False
    )
    fd = _fd_grad(e, c, rho, "absolute", eps=1e-7)
    np.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(h, np.ones_like(e))


def test_huber_fd_inside_outside_and_near_threshold():
    rng = np.random.default_rng(21)
    c = rng.normal(0.0, 0.6, size=12)
    c = c - float(np.mean(c))
    e = rng.normal(0.0, 0.25, size=12)
    q = e * c
    delta = float(np.quantile(np.abs(q), 0.55))
    assert delta > 1e-6
    idx_c = np.flatnonzero(np.abs(c) > 0.05)
    e[idx_c[0]] = (0.7 * delta) / c[idx_c[0]]
    e[idx_c[1]] = (1.4 * delta) / c[idx_c[1]]
    e[idx_c[2]] = (delta - 5e-4) / c[idx_c[2]]
    e[idx_c[3]] = (delta + 5e-4) / c[idx_c[3]]
    y_true = c.copy()
    y_pred = y_true + e
    y_mean = 0.0
    rho = 2.8
    q = e * c
    inside = np.abs(q) < 0.85 * delta
    outside = np.abs(q) > 1.15 * delta
    near = np.abs(np.abs(q) - delta) < 2e-3
    g, h, extras = toy_surrogate_scaled_grad_hess(
        y_true, y_pred, y_mean=y_mean, rho=rho, penalty_shape="huber", level_invariant=False, huber_delta=delta
    )
    fd = _fd_grad(e, c, rho, "huber", huber_delta=delta, eps=1e-7)
    if np.any(inside):
        np.testing.assert_allclose(g[inside], fd[inside], rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(h[inside], 1.0 + rho * (c[inside] ** 2), rtol=1e-12, atol=1e-12)
    if np.any(outside):
        np.testing.assert_allclose(g[outside], fd[outside], rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(h[outside], np.ones(int(np.sum(outside))), rtol=1e-12, atol=1e-12)
    if np.any(near):
        np.testing.assert_allclose(g[near], fd[near], rtol=2e-3, atol=2e-4)
    np.testing.assert_allclose(h[np.abs(q) <= delta], 1.0 + rho * (c[np.abs(q) <= delta] ** 2), rtol=1e-12)
    np.testing.assert_allclose(h[np.abs(q) > delta], 1.0, rtol=1e-12)
    assert extras["huber_delta"] == delta


def test_huber_objective_and_first_derivative_continuous_at_threshold():
    delta = 0.35
    rho = 1.7
    c = 0.8
    e_at = delta / c
    for sign in (1.0, -1.0):
        e0 = sign * e_at
        q0 = np.array([e0 * c])
        np.testing.assert_allclose(huber_phi(q0, delta), q0 ** 2, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(huber_phi(q0, delta), 2.0 * delta * np.abs(q0) - delta ** 2, rtol=1e-12, atol=1e-12)
        g_at, h_at, _ = toy_surrogate_scaled_grad_hess(
            np.array([c]), np.array([c + e0]), y_mean=0.0, rho=rho,
            penalty_shape="huber", level_invariant=False, huber_delta=delta,
        )
        g_in, h_in, _ = toy_surrogate_scaled_grad_hess(
            np.array([c]), np.array([c + e0 * (1 - 1e-8)]), y_mean=0.0, rho=rho,
            penalty_shape="huber", level_invariant=False, huber_delta=delta,
        )
        g_out, h_out, _ = toy_surrogate_scaled_grad_hess(
            np.array([c]), np.array([c + e0 * (1 + 1e-8)]), y_mean=0.0, rho=rho,
            penalty_shape="huber", level_invariant=False, huber_delta=delta,
        )
        np.testing.assert_allclose(g_in, g_at, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(g_out, g_at, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(g_at, np.array([e0 + rho * e0 * (c ** 2)]), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(g_at, np.array([e0 + rho * delta * abs(c) * np.sign(e0)]), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(h_in, 1.0 + rho * (c ** 2), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(h_out, 1.0, rtol=1e-12, atol=1e-12)
        j0 = scaled_objective(np.array([e0]), np.array([c]), rho, penalty_shape="huber", level_invariant=False, huber_delta=delta)
        j_in = scaled_objective(np.array([e0 * (1 - 1e-6)]), np.array([c]), rho, penalty_shape="huber", level_invariant=False, huber_delta=delta)
        j_out = scaled_objective(np.array([e0 * (1 + 1e-6)]), np.array([c]), rho, penalty_shape="huber", level_invariant=False, huber_delta=delta)
        np.testing.assert_allclose(j_in, j0, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(j_out, j0, rtol=1e-5, atol=1e-6)


def test_huber_central_region_matches_quadratic():
    y_true, y_pred, y_mean, e, c = _make_ec(seed=2, n=10)
    q = e * c
    delta = 10.0 * float(np.max(np.abs(q)))
    rho = 3.4
    g_h, h_h, _ = toy_surrogate_scaled_grad_hess(
        y_true, y_pred, y_mean=y_mean, rho=rho, penalty_shape="huber", level_invariant=False, huber_delta=delta
    )
    g_q, h_q, _ = toy_surrogate_scaled_grad_hess(
        y_true, y_pred, y_mean=y_mean, rho=rho, penalty_shape="quadratic", level_invariant=False
    )
    np.testing.assert_allclose(g_h, g_q, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(h_h, h_q, rtol=1e-12, atol=1e-12)
    psi_h = penalty_value(e, c, penalty_shape="huber", level_invariant=False, huber_delta=delta)["Psi"]
    psi_q = penalty_value(e, c, penalty_shape="quadratic", level_invariant=False)["Psi"]
    np.testing.assert_allclose(psi_h, psi_q, rtol=1e-12, atol=1e-12)


def test_v2_covariance_bounds():
    _yt, _yp, _ym, e, c = _make_ec(seed=3, n=20)
    C = covariance_C(e, c)
    q = e * c
    psi_q = penalty_value(e, c, penalty_shape="quadratic", level_invariant=False)["Psi"]
    psi_a = penalty_value(e, c, penalty_shape="absolute", level_invariant=False)["Psi"]
    assert (C ** 2) <= psi_q + 1e-12
    assert abs(C) <= psi_a + 1e-12
    delta = _huber_delta_for(e, c)
    psi_h = penalty_value(e, c, penalty_shape="huber", level_invariant=False, huber_delta=delta)["Psi"]
    assert abs(C) <= (psi_h + delta ** 2) / (2.0 * delta) + 1e-10
    assert np.all(huber_phi(q, delta) + delta ** 2 >= 2.0 * delta * np.abs(q) - 1e-12)


def test_v2_hessians_finite_strictly_positive():
    y_true, y_pred, y_mean, e, c = _make_ec(seed=19, n=13)
    delta = _huber_delta_for(e, c)
    for rho in (0.0, 1.0, 8.0):
        for shape in V2_FAMILIES:
            g, h, _ = toy_surrogate_scaled_grad_hess(
                y_true, y_pred, y_mean=y_mean, rho=rho, penalty_shape=shape, level_invariant=False, huber_delta=delta
            )
            assert np.all(np.isfinite(g)) and np.all(np.isfinite(h))
            assert np.all(h > 0.0)


def test_huber_unit_rho_penalty_gradient_helper():
    y_true, y_pred, y_mean, e, c = _make_ec(seed=6, n=9)
    delta = _huber_delta_for(e, c)
    g, _h, _ = toy_surrogate_scaled_grad_hess(
        y_true, y_pred, y_mean=y_mean, rho=1.0, penalty_shape="huber", level_invariant=False, huber_delta=delta
    )
    g_pen = fairness_penalty_gradient_unit_rho_from_ec(
        e, c, penalty_shape="huber", level_invariant=False, huber_delta=delta
    )
    np.testing.assert_allclose(g_pen, g - e, rtol=1e-12, atol=1e-12)


def test_huber_phi_matches_quadratic_at_threshold():
    delta = 0.42
    q = np.array([-delta, delta], dtype=float)
    np.testing.assert_allclose(huber_phi(q, delta), q ** 2, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(huber_phi(q, delta), 2.0 * delta * np.abs(q) - delta ** 2, rtol=1e-12, atol=1e-12)
