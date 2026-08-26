"""Focused tests for the frozen paper-v6 Delta_NL estimator."""

from __future__ import annotations

import numpy as np

from utils.delta_nl import (
    estimate_delta_nl,
    estimator_spec_hash,
    identifier_fold_assignment,
)


def _ids(n: int):
    return [f"sale-{i:05d}" for i in range(n)]


def test_affine_conditional_mean_near_zero():
    rng = np.random.default_rng(7)
    n = 4000
    y = rng.normal(12.4, 0.55, size=n)
    z = (y - y.mean()) / y.std()
    e = -0.12 + 0.35 * z
    yhat = y + e
    out = estimate_delta_nl(y, yhat, _ids(n))
    assert out["Delta_NL"] < 5e-3
    assert abs(out["Delta_NL_raw"]) < 5e-3


def test_u_shaped_conditional_mean_clearly_positive():
    rng = np.random.default_rng(11)
    n = 4000
    y = rng.normal(12.4, 0.55, size=n)
    z = (y - y.mean()) / y.std()
    e = 0.45 * (z**2 - 1.0)
    yhat = y + e
    out = estimate_delta_nl(y, yhat, _ids(n))
    assert out["Delta_NL"] > 0.05
    assert out["Delta_NL_raw"] > 0.05


def test_s_shaped_conditional_mean_clearly_positive():
    rng = np.random.default_rng(19)
    n = 4000
    y = rng.normal(12.4, 0.55, size=n)
    z = (y - y.mean()) / y.std()
    e = 0.35 * (z**3 - 1.2 * z)
    yhat = y + e
    out = estimate_delta_nl(y, yhat, _ids(n))
    assert out["Delta_NL"] > 0.02
    assert out["Delta_NL_raw"] > 0.02


def test_deterministic_repeated_call():
    rng = np.random.default_rng(3)
    n = 800
    y = rng.normal(12.0, 0.4, size=n)
    yhat = y + 0.2 * ((y - y.mean()) / y.std()) ** 2
    ids = _ids(n)
    a = estimate_delta_nl(y, yhat, ids)
    b = estimate_delta_nl(y, yhat, ids)
    assert a["Delta_NL"] == b["Delta_NL"]
    assert a["Delta_NL_raw"] == b["Delta_NL_raw"]
    assert a["fold_assignment_hash"] == b["fold_assignment_hash"]
    assert a["estimator_spec_hash"] == estimator_spec_hash()


def test_constant_shift_does_not_materially_change():
    rng = np.random.default_rng(23)
    n = 2500
    y = rng.normal(12.4, 0.5, size=n)
    z = (y - y.mean()) / y.std()
    e = 0.3 * (z**2 - 1.0) + 0.05 * z
    ids = _ids(n)
    base = estimate_delta_nl(y, y + e, ids)
    shifted = estimate_delta_nl(y, y + e + 0.75, ids)
    assert abs(base["Delta_NL"] - shifted["Delta_NL"]) < 1e-6


def test_nonzero_scale_invariance():
    rng = np.random.default_rng(29)
    n = 2500
    y = rng.normal(12.4, 0.5, size=n)
    z = (y - y.mean()) / y.std()
    e = 0.25 * (z**3 - z)
    ids = _ids(n)
    base = estimate_delta_nl(y, y + e, ids)
    scaled = estimate_delta_nl(y, y + 2.5 * e, ids)
    assert abs(base["Delta_NL"] - scaled["Delta_NL"]) < 1e-6
    assert abs(base["Delta_NL_raw"] - scaled["Delta_NL_raw"]) < 1e-6


def test_identical_prediction_vector_ignores_label():
    rng = np.random.default_rng(31)
    n = 1200
    y = rng.normal(12.1, 0.45, size=n)
    yhat = y + 0.2 * np.sin(3.0 * (y - y.mean()) / y.std())
    ids = _ids(n)
    a = estimate_delta_nl(y, yhat, ids)
    b = estimate_delta_nl(y, np.array(yhat, copy=True), ids)
    assert a["Delta_NL"] == b["Delta_NL"]
    folds = identifier_fold_assignment(ids)
    assert folds.min() == 0
    assert folds.max() == 4
    assert set(folds.tolist()) == {0, 1, 2, 3, 4}
