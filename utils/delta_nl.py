"""Frozen paper-v6 nonlinear conditional-mean diagnostic Delta_NL.

Population target
    Delta_NL = E[(m(Z) - Pi_aff m(Z))^2] / Var(e)
where e = yhat_log - y, Z is standardized log sale price, and m(Z)=E[e|Z].

Finite-sample estimator (frozen; not a model-selection object):
    5 identifier-based OOF folds; affine vs cubic B-spline; truncation at 0.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import SplineTransformer

ESTIMATOR_NAME = "paper_v6_delta_nl_oof_spline"
ESTIMATOR_VERSION = "1.0"
N_FOLDS = 5
SPLINE_DEGREE = 3
SPLINE_N_KNOTS = 8
SPLINE_KNOTS = "quantile"
SPLINE_INCLUDE_BIAS = False
FOLD_SALT = "paper_v6_delta_nl_v1"


def estimator_spec() -> Dict[str, Any]:
    return {
        "name": ESTIMATOR_NAME,
        "version": ESTIMATOR_VERSION,
        "n_folds": N_FOLDS,
        "e": "yhat_log - y_log",
        "Z": "(y_log - mean(y_log)) / sd(y_log) on the evaluation sample",
        "fold_assignment": {
            "rule": "int(sha256(salt + '|' + str(row_id))[:16], 16) % n_folds",
            "depends_on": "observation identifier only; independent of model/predictions",
            "n_folds": N_FOLDS,
            "salt": FOLD_SALT,
        },
        "affine": "LinearRegression of e on Z with intercept",
        "spline": {
            "transformer": "sklearn.preprocessing.SplineTransformer",
            "degree": SPLINE_DEGREE,
            "n_knots": SPLINE_N_KNOTS,
            "knots": SPLINE_KNOTS,
            "include_bias": SPLINE_INCLUDE_BIAS,
            "fit_on": "fold training portion only",
            "head": "LinearRegression with intercept",
        },
        "mse": "mean of squared OOF residuals",
        "var_e": "population variance of e on the evaluation sample (ddof=0)",
        "Delta_NL_raw": "(MSE_aff - MSE_spl) / Var_e",
        "Delta_NL": "max(0, Delta_NL_raw)",
        "cross_fitting": True,
        "no_heldout_or_2025_model_selection": True,
    }


def estimator_spec_hash() -> str:
    blob = json.dumps(estimator_spec(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _as_1d(values) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def _as_ids(values: Sequence[Any]) -> np.ndarray:
    return np.asarray([str(v) for v in values], dtype=object).reshape(-1)


def identifier_fold_assignment(
    row_ids: Sequence[Any],
    *,
    n_folds: int = N_FOLDS,
    salt: str = FOLD_SALT,
) -> np.ndarray:
    """Deterministic 0..n_folds-1 assignment from identifiers only."""
    ids = _as_ids(row_ids)
    folds = np.empty(ids.size, dtype=np.int32)
    prefix = f"{salt}|".encode("utf-8")
    for i, rid in enumerate(ids):
        digest = hashlib.sha256(prefix + str(rid).encode("utf-8")).hexdigest()
        folds[i] = int(digest[:16], 16) % int(n_folds)
    return folds


def fold_assignment_hash(row_ids: Sequence[Any], folds: np.ndarray) -> str:
    ids = _as_ids(row_ids)
    order = np.argsort(ids, kind="mergesort")
    payload = {
        "salt": FOLD_SALT,
        "n_folds": N_FOLDS,
        "n": int(ids.size),
        "counts": [int(np.sum(folds == k)) for k in range(N_FOLDS)],
        "id_sha256": hashlib.sha256("|".join(ids[order].tolist()).encode("utf-8")).hexdigest(),
        "fold_sha256": hashlib.sha256(folds[order].tobytes()).hexdigest(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _fit_predict_affine(z_train: np.ndarray, e_train: np.ndarray, z_test: np.ndarray) -> np.ndarray:
    model = LinearRegression(fit_intercept=True)
    model.fit(z_train.reshape(-1, 1), e_train)
    return np.asarray(model.predict(z_test.reshape(-1, 1)), dtype=float).reshape(-1)


def _fit_predict_spline(z_train: np.ndarray, e_train: np.ndarray, z_test: np.ndarray) -> np.ndarray:
    transformer = SplineTransformer(
        n_knots=SPLINE_N_KNOTS,
        degree=SPLINE_DEGREE,
        knots=SPLINE_KNOTS,
        include_bias=SPLINE_INCLUDE_BIAS,
    )
    basis_train = transformer.fit_transform(z_train.reshape(-1, 1))
    basis_test = transformer.transform(z_test.reshape(-1, 1))
    model = LinearRegression(fit_intercept=True)
    model.fit(basis_train, e_train)
    return np.asarray(model.predict(basis_test), dtype=float).reshape(-1)


def estimate_delta_nl(
    y_log,
    yhat_log,
    row_ids: Sequence[Any],
    *,
    folds: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """OOF affine-vs-spline estimator of Delta_NL on one evaluation sample."""
    y = _as_1d(y_log)
    yhat = _as_1d(yhat_log)
    ids = _as_ids(row_ids)
    if y.size != yhat.size or y.size != ids.size:
        raise ValueError("y_log, yhat_log, and row_ids must have the same length.")
    if y.size == 0:
        raise ValueError("evaluation sample is empty.")
    if np.unique(ids).size != ids.size:
        raise ValueError("row_ids must be unique within a split.")

    e = yhat - y
    y_mean = float(np.mean(y))
    y_sd = float(np.std(y, ddof=0))
    if not np.isfinite(y_sd) or y_sd <= 0.0:
        raise ValueError("log-price standard deviation must be positive.")
    z = (y - y_mean) / y_sd
    var_e = float(np.var(e, ddof=0))
    if not np.isfinite(var_e) or var_e <= 0.0:
        raise ValueError("population variance of e must be positive.")

    if folds is None:
        folds = identifier_fold_assignment(ids)
    else:
        folds = np.asarray(folds, dtype=np.int32).reshape(-1)
        if folds.size != y.size:
            raise ValueError("folds must have the same length as the evaluation sample.")

    ehat_aff = np.empty(y.size, dtype=float)
    ehat_spl = np.empty(y.size, dtype=float)
    fold_counts = []
    for k in range(N_FOLDS):
        test_mask = folds == k
        train_mask = ~test_mask
        n_test = int(test_mask.sum())
        n_train = int(train_mask.sum())
        if n_test == 0 or n_train == 0:
            raise RuntimeError(f"fold {k} is empty: n_train={n_train} n_test={n_test}")
        ehat_aff[test_mask] = _fit_predict_affine(z[train_mask], e[train_mask], z[test_mask])
        ehat_spl[test_mask] = _fit_predict_spline(z[train_mask], e[train_mask], z[test_mask])
        fold_counts.append({"fold": int(k), "n_train": n_train, "n_test": n_test})

    mse_aff = float(np.mean((e - ehat_aff) ** 2))
    mse_spl = float(np.mean((e - ehat_spl) ** 2))
    raw = (mse_aff - mse_spl) / var_e
    truncated = float(max(0.0, raw))
    return {
        "n": int(y.size),
        "n_folds": N_FOLDS,
        "fold_counts": fold_counts,
        "fold_assignment_hash": fold_assignment_hash(ids, folds),
        "estimator_spec_hash": estimator_spec_hash(),
        "y_mean": y_mean,
        "y_sd": y_sd,
        "var_e": var_e,
        "MSE_aff": mse_aff,
        "MSE_spl": mse_spl,
        "Delta_NL_raw": float(raw),
        "Delta_NL": truncated,
    }


def estimate_delta_nl_from_frame(
    df,
    *,
    y_col: str = "y_true_log",
    pred_col: str = "y_pred_log",
    id_col: str = "row_id",
    folds: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    return estimate_delta_nl(
        df[y_col].to_numpy(dtype=float),
        df[pred_col].to_numpy(dtype=float),
        df[id_col].to_numpy(),
        folds=folds,
    )
