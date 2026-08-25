"""Focused theory tests for the EXPERIMENTAL / TOY six-path mechanism experiment."""

from __future__ import annotations

import inspect

import numpy as np

from soft_constrained_models.boosting_models import (
    LGBCovPenalty,
    canonical_direct_exact_scaled_hessian,
    canonical_direct_scaled_grad_hess,
    canonical_surrogate_scaled_grad_hess,
)
from soft_constrained_models.toy_mechanism_objectives import (
    MM_CURVATURE_DESCRIPTION,
    apply_moment_basis,
    build_training_moment_basis,
    eval_nonlinear_subspace,
    inspect_current_direct,
    k1_squared_covariance_identity,
    local_slope_fairness_grad_hess,
    local_slope_scaled_grad_hess,
    local_slope_scaled_objective,
    make_current_direct,
    moment_mm_scaled_grad_hess,
    moment_prox,
    projector_apply,
    quadratic_pointwise_prox,
    scaled_moment_objective,
    smooth_nl_metrics,
    unit_fairness_gradient,
)


def _basis_from_rng(n: int = 80, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = rng.normal(12.0, 0.45, size=n)
    return build_training_moment_basis(y), y


def test_rho0_reduces_to_native_l2():
    rng = np.random.default_rng(4)
    y_true = rng.normal(12.0, 0.3, size=17)
    y_pred = y_true + rng.normal(0.0, 0.2, size=17)
    y_mean = float(np.mean(y_true))
    e = y_pred - y_true
    g_d, h_d, _ = canonical_direct_scaled_grad_hess(y_true, y_pred, y_mean=y_mean, rho=0.0)
    g_q, h_q, _ = canonical_surrogate_scaled_grad_hess(y_true, y_pred, y_mean=y_mean, rho=0.0)
    basis, _y = _basis_from_rng(17, seed=4)
    for k in (1, 2, 3):
        g_m, h_m, _ = moment_mm_scaled_grad_hess(e, basis["phi_train"][:, :k], 0.0)
        np.testing.assert_allclose(g_m, e, atol=1e-12)
        np.testing.assert_allclose(h_m, np.ones_like(e), atol=1e-12)
    c = y_true - y_mean
    g_ls, h_ls, _ = local_slope_scaled_grad_hess(e, c, rho=0.0, epsilon=0.08)
    np.testing.assert_allclose(g_d, e, atol=1e-12)
    np.testing.assert_allclose(h_d, np.ones_like(e), atol=1e-12)
    np.testing.assert_allclose(g_q, e, atol=1e-12)
    np.testing.assert_allclose(h_q, np.ones_like(e), atol=1e-12)
    np.testing.assert_allclose(g_ls, e, atol=1e-12)
    np.testing.assert_allclose(h_ls, np.ones_like(e), atol=1e-12)


def test_current_direct_imported_unchanged_formulas():
    info = inspect_current_direct()
    assert info["class"] == "LGBCovPenalty"
    assert info["imported_unchanged"] is True
    assert "canonical_direct_scaled_grad_hess" == canonical_direct_scaled_grad_hess.__name__
    assert info["canonical_grad"].startswith("grad_i = e_i + (rho/2)")
    assert "c c^T" in info["canonical_exact_hessian"]
    src = inspect.getsource(LGBCovPenalty.fobj)
    assert "canonical_direct_scaled_grad_hess" in src
    model = make_current_direct(1.2, lgbm_params={"n_estimators": 2, "verbosity": -1}, verbose=False)
    assert isinstance(model, LGBCovPenalty)
    assert model.ratio_mode == "diff"


def test_basis_means_gram_phi1_nesting_and_heldout_map():
    basis, y = _basis_from_rng(120, seed=11)
    phi = basis["phi_train"]
    z = basis["z_train"]
    np.testing.assert_allclose(np.mean(phi, axis=0), np.zeros(3), atol=1e-10)
    gram = phi.T @ phi / phi.shape[0]
    np.testing.assert_allclose(gram, np.eye(3), atol=1e-10)
    np.testing.assert_allclose(phi[:, 0], z, atol=1e-10)
    # Nested prefixes remain orthonormal.
    g2 = phi[:, :2].T @ phi[:, :2] / phi.shape[0]
    np.testing.assert_allclose(g2, np.eye(2), atol=1e-10)
    for k in range(3):
        raw = z ** (k + 1)
        assert float(np.dot(phi[:, k], raw)) > 0.0
    y_hold = y[::-1] + 0.01
    phi_hold = apply_moment_basis(y_hold, basis)
    phi_hold2 = apply_moment_basis(y_hold, {"y_mean": basis["y_mean"], "sigma_c": basis["sigma_c"], "rinv": basis["rinv"]})
    np.testing.assert_allclose(phi_hold, phi_hold2, atol=1e-12)
    # Held-out map must not depend on held-out QR.
    assert phi_hold.shape == (y_hold.size, 3)


def test_projector_and_mm_hessian_geometry():
    basis, y = _basis_from_rng(36, seed=3)
    rng = np.random.default_rng(3)
    e = rng.normal(size=36)
    rho = 2.4
    for k in (1, 2, 3):
        phi = basis["phi_train"][:, :k]
        pv = projector_apply(e, phi)
        p2 = projector_apply(pv, phi)
        np.testing.assert_allclose(p2, pv, atol=1e-10)
        # Rayleigh quotient in [0, 1]
        rq = float(np.dot(e, pv) / np.dot(e, e))
        assert -1e-10 <= rq <= 1.0 + 1e-10
        g, h, extras = moment_mm_scaled_grad_hess(e, phi, rho)
        np.testing.assert_allclose(g, e + rho * pv, atol=1e-12)
        np.testing.assert_allclose(h, np.full_like(e, 1.0 + rho), atol=1e-12)
        assert extras["curvature"] == MM_CURVATURE_DESCRIPTION
        # H_exact v = v + rho P v, never claimed as supplied Hessian.
        hv = e + rho * pv
        np.testing.assert_allclose(g, hv, atol=1e-12)


def test_moment_finite_difference_gradient():
    basis, _y = _basis_from_rng(18, seed=8)
    rng = np.random.default_rng(8)
    e = rng.normal(size=18)
    rho = 1.7
    eps = 1e-6
    for k in (1, 2, 3):
        phi = basis["phi_train"][:, :k]
        g, _h, _ = moment_mm_scaled_grad_hess(e, phi, rho)
        fd = np.zeros_like(e)
        for i in range(e.size):
            up = e.copy(); up[i] += eps
            dn = e.copy(); dn[i] -= eps
            fd[i] = (scaled_moment_objective(up, phi, rho) - scaled_moment_objective(dn, phi, rho)) / (2 * eps)
        np.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-6)


def test_mm_majorization_inequality():
    basis, _y = _basis_from_rng(24, seed=9)
    rng = np.random.default_rng(9)
    e = rng.normal(size=24)
    rho = 3.1
    for k in (1, 2, 3):
        phi = basis["phi_train"][:, :k]
        g, _h, _ = moment_mm_scaled_grad_hess(e, phi, rho)
        f0 = scaled_moment_objective(e, phi, rho)
        for _ in range(40):
            d = rng.normal(size=24)
            f1 = scaled_moment_objective(e + d, phi, rho)
            major = f0 + float(np.dot(g, d)) + 0.5 * (1.0 + rho) * float(np.dot(d, d))
            assert f1 <= major + 1e-8


def test_k1_squared_covariance_equivalence():
    basis, y = _basis_from_rng(90, seed=2)
    rng = np.random.default_rng(2)
    e = rng.normal(size=90)
    ident = k1_squared_covariance_identity(e, basis["c_train"], basis["phi_train"][:, 0], basis["sigma_c"])
    np.testing.assert_allclose(ident["eT_P1_e"], ident["n_cov_over_sigma_sq"], rtol=1e-10, atol=1e-10)


def test_shift_invariance_direct_and_moments_not_quadratic_or_local_slope():
    basis, y = _basis_from_rng(40, seed=5)
    rng = np.random.default_rng(5)
    e = rng.normal(size=40)
    c = basis["c_train"]
    a = 0.63
    cov0 = float(np.mean(e * c))
    cov1 = float(np.mean((e + a) * c))
    np.testing.assert_allclose(cov0, cov1, atol=1e-12)
    for k in (1, 2, 3):
        m0 = (basis["phi_train"][:, :k].T @ e) / e.size
        m1 = (basis["phi_train"][:, :k].T @ (e + a)) / e.size
        np.testing.assert_allclose(m0, m1, atol=1e-10)
    q0 = float(np.mean(np.square(e * c)))
    q1 = float(np.mean(np.square((e + a) * c)))
    assert abs(q1 - q0) > 1e-6
    ls0 = local_slope_scaled_objective(e, c, 1.0, 0.1)
    ls1 = local_slope_scaled_objective(e + a, c, 1.0, 0.1)
    assert abs(ls1 - ls0) > 1e-6


def test_local_slope_finite_differences_and_linear_residual_invariants():
    rng = np.random.default_rng(12)
    c = rng.normal(0.0, 0.4, size=14)
    c = c - float(np.mean(c))
    e = rng.normal(0.0, 0.2, size=14)
    rho = 2.2
    eps0 = 0.09
    g, h, _ = local_slope_scaled_grad_hess(e, c, rho, eps0)
    fd_g = np.zeros_like(e)
    fd_h = np.zeros_like(e)
    step = 1e-6
    f0 = local_slope_scaled_objective(e, c, rho, eps0)
    for i in range(e.size):
        up = e.copy(); up[i] += step
        dn = e.copy(); dn[i] -= step
        fd_g[i] = (local_slope_scaled_objective(up, c, rho, eps0) - local_slope_scaled_objective(dn, c, rho, eps0)) / (2 * step)
        fd_h[i] = (
            local_slope_scaled_objective(up, c, rho, eps0) - 2.0 * f0 + local_slope_scaled_objective(dn, c, rho, eps0)
        ) / (step ** 2)
    np.testing.assert_allclose(g, fd_g, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(h, fd_h, rtol=2e-3, atol=2e-4)
    assert np.all(h > 0.0)
    # Covariance upper bound.
    cov_abs = abs(float(np.mean(e * c)))
    mean_abs = float(np.mean(np.abs(e * c)))
    mean_smooth = float(np.mean(np.abs(c) * np.sqrt(np.square(e) + np.square(eps0 * c))))
    assert cov_abs <= mean_abs + 1e-12
    assert mean_abs <= mean_smooth + 1e-12
    # e = beta c => fairness_grad / c constant, hessian constant off c=0.
    beta = -0.17
    e_lin = beta * c
    fg, fh = local_slope_fairness_grad_hess(e_lin, c, eps0)
    nz = np.abs(c) > 1e-12
    ratio = fg[nz] / c[nz]
    np.testing.assert_allclose(ratio, np.full(ratio.size, ratio[0]), atol=1e-10)
    np.testing.assert_allclose(fh[nz], np.full(np.sum(nz), fh[nz][0]), atol=1e-10)


def test_synthetic_geometry_moment_shrinkage_and_quadratic_higher_order():
    basis, _y = _basis_from_rng(100, seed=21)
    phi = basis["phi_train"]
    rng = np.random.default_rng(21)
    a1, a2, a3 = 1.4, -0.8, 0.55
    noise = 0.05 * rng.normal(size=100)
    e0 = a1 * phi[:, 0] + a2 * phi[:, 1] + a3 * phi[:, 2] + noise
    rho = 2.0
    for k, expect in ((1, (True, False, False)), (2, (True, True, False)), (3, (True, True, True))):
        e1 = moment_prox(e0, phi[:, :k], rho)
        m0 = (phi.T @ e0) / e0.size
        m1 = (phi.T @ e1) / e1.size
        shrink = 1.0 / (1.0 + rho)
        for j, touched in enumerate(expect):
            if touched:
                np.testing.assert_allclose(m1[j], shrink * m0[j], atol=1e-10)
            else:
                np.testing.assert_allclose(m1[j], m0[j], atol=1e-10)
    e_lin = 0.9 * phi[:, 0]
    for k in (1, 2, 3):
        e1 = moment_prox(e_lin, phi[:, :k], rho)
        m1 = (phi.T @ e1) / e1.size
        np.testing.assert_allclose(m1[1:], np.zeros(2), atol=1e-10)
        assert abs(m1[0]) < abs(float((phi[:, 0] @ e_lin) / e_lin.size))
    # Local-slope linear residual stays in span(c).
    c = basis["c_train"]
    e_beta = -0.2 * c
    fg, _fh = local_slope_fairness_grad_hess(e_beta, c, 0.11)
    # gradient proportional to c
    coef = np.linalg.lstsq(c.reshape(-1, 1), fg, rcond=None)[0][0]
    np.testing.assert_allclose(fg, coef * c, atol=1e-10)
    # Quadratic prox from pure linear residual creates higher-order projection.
    e_prox = quadratic_pointwise_prox(0.25 * c, c, rho=3.5)
    m_q = (phi.T @ e_prox) / e_prox.size
    assert abs(m_q[1]) + abs(m_q[2]) > 1e-6


def test_eval_nl_subspace_is_orthogonal_to_affine_c():
    rng = np.random.default_rng(6)
    c = rng.normal(size=60)
    c = c - float(np.mean(c))
    phi_nl = eval_nonlinear_subspace(c)
    ones = np.ones(c.size)
    np.testing.assert_allclose(phi_nl.T @ ones / c.size, np.zeros(2), atol=1e-10)
    np.testing.assert_allclose(phi_nl.T @ c / c.size, np.zeros(2), atol=1e-10)
    gram = phi_nl.T @ phi_nl / c.size
    np.testing.assert_allclose(gram, np.eye(2), atol=1e-10)
    e = 0.3 * c + 0.1 * (c ** 2 - np.mean(c ** 2))
    mets = smooth_nl_metrics(e, phi_nl)
    assert mets["L_NL"] >= -1e-15
    assert mets["NL_share"] >= -1e-15


def test_unit_fairness_gradients_match_definitions():
    basis, _y = _basis_from_rng(30, seed=7)
    rng = np.random.default_rng(7)
    e = rng.normal(size=30)
    c = basis["c_train"]
    phi = basis["phi_train"]
    g_dir = unit_fairness_gradient(method="current_direct", e=e, c=c, phi=phi, epsilon=0.1)
    np.testing.assert_allclose(g_dir, 0.5 * float(np.mean(e * c)) * c, atol=1e-12)
    g_q = unit_fairness_gradient(method="quadratic", e=e, c=c, phi=phi, epsilon=0.1)
    np.testing.assert_allclose(g_q, e * np.square(c), atol=1e-12)


def test_direct_exact_hessian_formula_recorded():
    rng = np.random.default_rng(1)
    c = rng.normal(size=9)
    c = c - float(np.mean(c))
    rho = 4.0
    H = canonical_direct_exact_scaled_hessian(c, rho)
    n = c.size
    expected = np.eye(n) + (rho / (2.0 * n)) * np.outer(c, c)
    np.testing.assert_allclose(H, expected, atol=1e-12)
