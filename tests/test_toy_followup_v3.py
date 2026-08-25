"""Focused tests for the EXPERIMENTAL / TOY follow-up V3 refinements."""

from __future__ import annotations

import numpy as np

from soft_constrained_models.toy_followup_metrics import (
    n3_orth_from_phi3col,
    orthonormality_diagnostics,
    reconstruct_phi_full,
)
from soft_constrained_models.toy_hybrid_objectives import (
    quadratic_direct_cap_scaled_grad_hess,
    quadratic_nl_guardrail_scaled_grad_hess,
    quadratic_nl_guardrail_scaled_objective,
    guardrail_d23,
)
from soft_constrained_models.toy_mechanism_objectives import build_training_moment_basis
from soft_constrained_models.boosting_models import canonical_surrogate_scaled_grad_hess


def test_four_column_basis_is_orthonormal_and_nonlinear_is_orthogonal_to_linear():
    rng = np.random.default_rng(2025)
    y = rng.normal(12.0, 0.45, size=400)
    basis = build_training_moment_basis(y)
    phi_full = reconstruct_phi_full(basis["c_train"], float(basis["sigma_c"]), basis["rinv"])
    diag = orthonormality_diagnostics(phi_full)
    assert diag["gram_offdiag_max"] < 1e-8
    assert diag["gram_diag_max_abs_err"] < 1e-8
    assert abs(diag["phi2_mean"]) < 1e-8
    assert abs(diag["phi3_mean"]) < 1e-8
    assert abs(diag["phi2_dot_z"]) < 1e-8
    assert abs(diag["phi3_dot_z"]) < 1e-8
    assert abs(diag["phi2_dot_phi3"]) < 1e-8
    # Dropped intercept column: remaining 3-col matrix matches six-path phi_train.
    np.testing.assert_allclose(phi_full[:, 1:4], basis["phi_train"], atol=1e-12)


def test_n3_matches_first_qnl_m2_m3_indexing():
    rng = np.random.default_rng(7)
    y = rng.normal(12.0, 0.4, size=80)
    e = rng.normal(0.0, 0.2, size=80)
    basis = build_training_moment_basis(y)
    m2, m3, n3, n3_rel = n3_orth_from_phi3col(e, basis["phi_train"])
    n = float(e.size)
    m2_ref = float(np.dot(basis["phi_train"][:, 1], e) / n)
    m3_ref = float(np.dot(basis["phi_train"][:, 2], e) / n)
    np.testing.assert_allclose(m2, m2_ref)
    np.testing.assert_allclose(m3, m3_ref)
    np.testing.assert_allclose(n3, m2_ref ** 2 + m3_ref ** 2)
    var_e = float(np.mean(np.square(e - np.mean(e))))
    np.testing.assert_allclose(n3_rel, n3 / var_e)


def test_qd_lambda0_and_qnl_gamma0_still_match_quadratic():
    rng = np.random.default_rng(3)
    y = rng.normal(12.0, 0.4, size=24)
    basis = build_training_moment_basis(y)
    e = rng.normal(0.0, 0.2, size=24)
    c = basis["c_train"]
    alpha = 1.6601530432925289
    g, h, _ = quadratic_direct_cap_scaled_grad_hess(e, c, alpha=alpha, lam=0.0)
    gq, hq, _ = canonical_surrogate_scaled_grad_hess(c, e + c, y_mean=0.0, rho=alpha)
    np.testing.assert_allclose(g, gq, atol=1e-12)
    np.testing.assert_allclose(h, hq, atol=1e-12)
    d23 = guardrail_d23(basis["phi_train"])
    gn, hn, _ = quadratic_nl_guardrail_scaled_grad_hess(e, c, basis["phi_train"], d23, rho=alpha, gamma=0.0)
    np.testing.assert_allclose(gn, gq, atol=1e-12)
    np.testing.assert_allclose(hn, hq, atol=1e-12)


def test_qnl_finite_difference_on_orthogonal_penalty():
    rng = np.random.default_rng(11)
    y = rng.normal(12.0, 0.4, size=14)
    basis = build_training_moment_basis(y)
    e = rng.normal(0.0, 0.2, size=14)
    d23 = guardrail_d23(basis["phi_train"])
    rho, gamma = 1.66, 8.0
    g, _, _ = quadratic_nl_guardrail_scaled_grad_hess(e, basis["c_train"], basis["phi_train"], d23, rho=rho, gamma=gamma)
    step = 1e-6
    fd = np.zeros_like(e)
    for i in range(e.size):
        up = e.copy(); up[i] += step
        dn = e.copy(); dn[i] -= step
        fd[i] = (
            quadratic_nl_guardrail_scaled_objective(up, basis["c_train"], basis["phi_train"], rho=rho, gamma=gamma)
            - quadratic_nl_guardrail_scaled_objective(dn, basis["c_train"], basis["phi_train"], rho=rho, gamma=gamma)
        ) / (2 * step)
    np.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-6)


def test_interpolate_lambda_brackets_with_zero():
    import sys
    from pathlib import Path

    scripts = str(Path(__file__).resolve().parents[1] / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import toy_followup_v3 as fu

    lam, how = fu.interpolate_lambda([(0.0, 0.301), (2.85, 0.202)], 0.225)
    assert how == "log_lambda_bracket"
    assert 0.0 < lam < 2.85
    lam0, how0 = fu.interpolate_lambda([(0.0, 0.301)], 0.25)
    assert how0 == "nearest_is_lambda0"
    assert lam0 is None
