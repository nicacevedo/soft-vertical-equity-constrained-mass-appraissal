#!/usr/bin/env python
"""
Theory-informed rho-range analysis for LGBCovPenalty[diff].

Purpose
-------
Fit only the unpenalized LGBM baseline, then compute theory-implied rho values
for the covariance-penalty objective used by LGBCovPenalty[diff]:

    empirical LightGBM objective scale:  MSE_log + 0.5 * rho * Cov(e, Y)^2

where
    Y = log(price),
    f0(X) = baseline LGBM log-price prediction,
    e = f0(X) - Y,
    Cov(e, Y) is the same log-residual/log-price covariance proxy as the
    diff-mode covariance penalty.

The code implements the 5-step method:
  1. Fit baseline LGBM and compute log-space baseline quantities.
  2. Compute rho values for target covariance shrinkage levels.
  3. Convert PRD targets into price-ratio covariance targets.
  4. Calibrate price-ratio covariance targets back to log-covariance scale.
  5. Aggregate per-split/per-config recommendations into a robust theory range.

Run from the project root, for example:

    python scripts/theory_informed_rho_range.py \
        --data-path ./data/CCAO/2025/training_data.parquet \
        --assessment-years 2022,2023,2024,2025 \
        --lgbm-config-keys cv_top1_r2,test_best_r2,cv_top2_r2 \
        --lgbm-hyperparameter-file best_lgbm_baseline_configs.yaml \
        --out-dir output/theory_rho_range

If your hyperparameter file does not contain all three keys, pass the keys that
exist, e.g. --lgbm-config-keys test_best_r2.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

try:
    import lightgbm as lgb
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install lightgbm before running this script.") from exc


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

_RUN_T0 = time.perf_counter()
_TIMING_ROWS: List[Dict[str, Any]] = []


def _format_seconds(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 1.0:
        return f"{1000.0 * seconds:.0f}ms"
    if seconds < 60.0:
        return f"{seconds:.2f}s"
    minutes, rem = divmod(seconds, 60.0)
    return f"{int(minutes)}m{rem:04.1f}s"


def _log(msg: str, **fields: Any) -> None:
    elapsed = _format_seconds(time.perf_counter() - _RUN_T0)
    suffix = ""
    if fields:
        suffix = " | " + " | ".join(f"{k}={v}" for k, v in fields.items())
    print(f"[theory-rho +{elapsed}] {msg}{suffix}", flush=True)


@contextmanager
def _timed_step(step: str, **fields: Any):
    start = time.perf_counter()
    _log(f"start: {step}", **fields)
    status = "ok"
    try:
        yield
    except Exception:
        status = "failed"
        raise
    finally:
        duration = time.perf_counter() - start
        row = {"step": step, "seconds": duration, "status": status, **fields}
        _TIMING_ROWS.append(row)
        _log(f"end: {step}", duration=_format_seconds(duration), status=status, **fields)


def _timing_summary_df() -> pd.DataFrame:
    if not _TIMING_ROWS:
        return pd.DataFrame(columns=["step", "calls", "total_seconds", "mean_seconds", "max_seconds"])
    df = pd.DataFrame(_TIMING_ROWS)
    return (
        df.groupby("step", dropna=False)["seconds"]
        .agg(calls="count", total_seconds="sum", mean_seconds="mean", max_seconds="max")
        .reset_index()
        .sort_values("total_seconds", ascending=False)
    )


def _parse_csv_list(raw: str, *, cast=str) -> List[Any]:
    out = []
    for token in str(raw).split(","):
        token = token.strip()
        if token:
            out.append(cast(token))
    return out


def _safe_mean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else float("nan")


def _safe_var(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean((x - np.mean(x)) ** 2)) if x.size else float("nan")


def _safe_cov(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        return float("nan")
    return float(np.mean((x - np.mean(x)) * (y - np.mean(y))))


def _prd_from_arrays(y_pred_price: np.ndarray, y_true_price: np.ndarray) -> float:
    y_pred_price = np.asarray(y_pred_price, dtype=float).reshape(-1)
    y_true_price = np.asarray(y_true_price, dtype=float).reshape(-1)
    mask = np.isfinite(y_pred_price) & np.isfinite(y_true_price) & (y_true_price > 0.0)
    pred = y_pred_price[mask]
    actual = y_true_price[mask]
    if actual.size == 0 or float(np.sum(actual)) <= 0.0:
        return float("nan")
    ratio = pred / actual
    mean_ratio = float(np.mean(ratio))
    weighted_mean_ratio = float(np.sum(pred) / np.sum(actual))
    if weighted_mean_ratio == 0.0 or not np.isfinite(weighted_mean_ratio):
        return float("nan")
    return mean_ratio / weighted_mean_ratio


def _rho_from_shrinkage(q: float, A: float) -> float:
    """rho(q) = 2(1-q)/(q A), where q is remaining covariance fraction."""
    q = float(q)
    A = float(A)
    if not (np.isfinite(q) and np.isfinite(A)) or q <= 0.0 or q >= 1.0 or A <= 0.0:
        return float("nan")
    return float(2.0 * (1.0 - q) / (q * A))


def _q_from_rho(rho: float, A: float) -> float:
    rho = float(rho)
    A = float(A)
    if not (np.isfinite(rho) and np.isfinite(A)) or rho < 0.0 or A <= 0.0:
        return float("nan")
    return float(1.0 / (1.0 + rho * A / 2.0))


def _rho_from_accuracy_budget(alpha: float, *, B: float, C0: float, A: float) -> float:
    """
    Max rho whose approximate MSE increase is within alpha * baseline MSE.

    Rank-one theory gives:
        DeltaMSE(q) ~= (C0^2 / A) * (1 - q)^2.
    Requiring DeltaMSE <= alpha * B gives
        q >= 1 - sqrt(alpha * B * A / C0^2).
    The largest rho under this budget is rho(q_min).
    """
    alpha = float(alpha)
    B = float(B)
    C0 = float(C0)
    A = float(A)
    if not all(np.isfinite(v) for v in [alpha, B, C0, A]):
        return float("nan")
    if alpha <= 0.0 or B <= 0.0 or A <= 0.0 or C0 == 0.0:
        return float("nan")
    q_min = 1.0 - math.sqrt(max(0.0, alpha * B * A / (C0 ** 2)))
    q_min = float(np.clip(q_min, 1e-9, 1.0 - 1e-9))
    return _rho_from_shrinkage(q_min, A)


# ---------------------------------------------------------------------------
# Data loading, inspired by quick_test_models.py
# ---------------------------------------------------------------------------

def _build_lgbm_params_from_files(model_params: dict, ccao_params: dict, seed: int) -> dict:
    """Match quick_test_models.py fallback logic."""
    model_default = dict(model_params.get("LGBMRegressor", {}))
    hp_default = dict(ccao_params["model"]["hyperparameter"]["default"])

    num_leaves = int(model_default.get("num_leaves", hp_default["num_leaves"]))
    if "max_depth" in model_default and model_default["max_depth"] is not None:
        max_depth = int(model_default["max_depth"])
    else:
        add_to_linked_depth = int(hp_default.get("add_to_linked_depth", 4))
        max_depth = int(np.floor(np.log2(max(num_leaves, 2))) + add_to_linked_depth)

    return {
        "boosting_type": str(model_default.get("boosting_type", "gbdt")),
        "objective": str(model_default.get("objective", "mse")),
        "n_estimators": int(model_default.get("n_estimators", hp_default["num_iterations"])),
        "learning_rate": float(model_default.get("learning_rate", hp_default["learning_rate"])),
        "num_leaves": num_leaves,
        "max_depth": max_depth,
        "max_bin": int(model_default.get("max_bin", hp_default["max_bin"])),
        "min_child_samples": int(model_default.get("min_child_samples", hp_default["min_data_in_leaf"])),
        "min_split_gain": float(model_default.get("min_split_gain", hp_default["min_gain_to_split"])),
        "colsample_bytree": float(model_default.get("colsample_bytree", hp_default["feature_fraction"])),
        "reg_alpha": float(model_default.get("reg_alpha", hp_default["lambda_l1"])),
        "reg_lambda": float(model_default.get("reg_lambda", hp_default["lambda_l2"])),
        "max_cat_threshold": int(model_default.get("max_cat_threshold", hp_default["max_cat_threshold"])),
        "min_data_per_group": int(model_default.get("min_data_per_group", hp_default["min_data_per_group"])),
        "cat_smooth": float(model_default.get("cat_smooth", hp_default["cat_smooth"])),
        "cat_l2": float(model_default.get("cat_l2", hp_default["cat_l2"])),
        "random_state": int(model_default.get("random_state", seed)),
        "n_jobs": int(model_default.get("n_jobs", 1)),
        "verbosity": int(model_default.get("verbosity", -1)),
        "importance_type": str(model_default.get("importance_type", "split")),
    }


def _load_lgbm_params_from_hyperparameter_file(path: str, config_key: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    baselines = cfg.get("lgbm_baselines", {})
    if config_key not in baselines:
        available = ", ".join(sorted(str(k) for k in baselines.keys()))
        raise KeyError(f"LGBM config key '{config_key}' not found in {path}. Available: {available}")
    params = baselines[config_key].get("lgbm_params", {})
    if not isinstance(params, dict) or not params:
        raise ValueError(f"LGBM config key '{config_key}' in {path} lacks lgbm_params.")
    return dict(params)


def _load_and_split_data(
    *,
    data_path: str,
    params: dict,
    target_column: str,
    date_column: str,
    sample_frac: Optional[float],
    sample_seed: int,
    assessment_year: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    predictor_cols = list(params["model"]["predictor"]["all"])
    categorical_cols = list(params["model"]["predictor"]["categorical"])
    filter_cols = ["ind_pin_is_multicard", "sv_is_outlier"]
    required_cols = list(dict.fromkeys(predictor_cols + [target_column, date_column] + filter_cols))
    row_filters = [
        ("ind_pin_is_multicard", "==", False),
        ("sv_is_outlier", "==", False),
    ]

    _log(
        "loading parquet",
        data_path=data_path,
        assessment_year=assessment_year,
        selected_cols=len(required_cols),
    )
    read_engine = "fastparquet"
    pushdown_enabled = False
    pushdown_reason = "pyarrow_unavailable"
    try:
        import pyarrow.dataset as ds
        import pyarrow.types as patypes

        # PyArrow's native reader is multi-threaded; use it whenever available.
        read_engine = "pyarrow"
        schema = ds.dataset(data_path, format="parquet").schema
        if all(name in schema.names and patypes.is_boolean(schema.field(name).type) for name in filter_cols):
            read_engine = "pyarrow"
            pushdown_enabled = True
            pushdown_reason = "bool_filter_schema"
        else:
            pushdown_reason = "non_boolean_filter_schema"
    except Exception as exc:
        pushdown_reason = f"pushdown_probe_failed:{type(exc).__name__}"

    with _timed_step("data: read parquet", data_path=data_path, assessment_year=assessment_year):
        if pushdown_enabled:
            df = pd.read_parquet(
                data_path,
                engine=read_engine,
                columns=required_cols,
                filters=row_filters,
            )
        else:
            df = pd.read_parquet(data_path, engine=read_engine, columns=required_cols)
    _log(
        "parquet loaded",
        rows=int(df.shape[0]),
        cols=int(df.shape[1]),
        engine=read_engine,
        row_pushdown=pushdown_enabled,
        pushdown_reason=pushdown_reason,
    )

    with _timed_step("data: filter rows", assessment_year=assessment_year, rows_in=int(df.shape[0])):
        df = df[
            (~df["ind_pin_is_multicard"].astype("bool").fillna(True))
            & (~df["sv_is_outlier"].astype("bool").fillna(True))
        ].copy()
    _log("row filters applied", rows=int(df.shape[0]))

    keep_cols = predictor_cols + [target_column, date_column]
    with _timed_step("data: project columns", assessment_year=assessment_year, kept_cols=len(keep_cols)):
        df = df.loc[:, keep_cols].copy()

    if sample_frac is not None and float(sample_frac) < 1.0:
        if not (0.0 < float(sample_frac) <= 1.0):
            raise ValueError("sample_frac must be in (0, 1].")
        with _timed_step("data: sample rows", assessment_year=assessment_year, sample_frac=float(sample_frac)):
            df = df.sample(frac=float(sample_frac), random_state=int(sample_seed)).copy()
        _log("sampling applied", sample_frac=float(sample_frac), rows=int(df.shape[0]))

    with _timed_step("data: parse and sort dates", assessment_year=assessment_year, rows=int(df.shape[0])):
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(date_column).reset_index(drop=True)

    with _timed_step("data: split by assessment year", assessment_year=assessment_year, rows=int(df.shape[0])):
        df_assess = df.loc[df[date_column].dt.year == int(assessment_year), :].copy()
        df_train_all = df.loc[df[date_column].dt.year < int(assessment_year), :].copy()

    train_prop = float(params["cv"]["split_prop"])
    split_idx = int(train_prop * df_train_all.shape[0])
    df_test = df_train_all.iloc[split_idx:, :].copy()
    df_train_validate = df_train_all.iloc[:split_idx, :].copy()
    _log(
        "data split completed",
        assessment_year=assessment_year,
        train_validate_rows=int(df_train_validate.shape[0]),
        test_rows=int(df_test.shape[0]),
        assess_rows=int(df_assess.shape[0]),
    )
    return df_train_validate, df_test, df_assess, predictor_cols, categorical_cols


def _cast_categoricals(X: pd.DataFrame, categorical_cols: Sequence[str]) -> pd.DataFrame:
    X = X.copy()
    for col in categorical_cols:
        if col in X.columns:
            X[col] = X[col].astype("category")
    return X


# ---------------------------------------------------------------------------
# Theory quantities
# ---------------------------------------------------------------------------

def _compute_prd_identity_quantities(y_log: np.ndarray, f_log: np.ndarray) -> Dict[str, float]:
    price = np.exp(y_log)
    pred_price = np.exp(f_log)
    ratio = pred_price / price
    mu_r = _safe_mean(ratio)
    mu_p = _safe_mean(price)
    prd_value = _prd_from_arrays(pred_price, price)
    cov_r_p_emp = _safe_cov(ratio, price)
    if np.isfinite(mu_r) and np.isfinite(mu_p) and np.isfinite(prd_value) and prd_value != 0.0:
        cov_r_p_from_prd = float(mu_r * mu_p * (1.0 / prd_value - 1.0))
    else:
        cov_r_p_from_prd = float("nan")
    return {
        "mu_ratio_price": mu_r,
        "mu_price": mu_p,
        "prd": prd_value,
        "cov_ratio_price_empirical": cov_r_p_emp,
        "cov_ratio_price_from_prd": cov_r_p_from_prd,
        "cov_ratio_price_identity_abs_error": abs(cov_r_p_emp - cov_r_p_from_prd)
        if np.isfinite(cov_r_p_emp) and np.isfinite(cov_r_p_from_prd) else float("nan"),
        "cov_ratio_price_identity_rel_error": abs(cov_r_p_emp - cov_r_p_from_prd) / max(abs(cov_r_p_emp), 1e-12)
        if np.isfinite(cov_r_p_emp) and np.isfinite(cov_r_p_from_prd) else float("nan"),
    }


def _valid_prd_target(prd0: float, target: float) -> bool:
    """Use targets that lie between the baseline PRD and perfect PRD=1."""
    if not (np.isfinite(prd0) and np.isfinite(target)):
        return False
    if prd0 > 1.0:
        return 1.0 <= target <= prd0
    if prd0 < 1.0:
        return prd0 <= target <= 1.0
    return False


def compute_theory_for_predictions(
    *,
    y_log: np.ndarray,
    f0_log: np.ndarray,
    split_label: str,
    data_source: str,
    config_key: str,
    assessment_year: int,
    prd_targets: Sequence[float],
    shrinkage_q_values: Sequence[float],
    accuracy_budgets: Sequence[float],
    empirical_rho_low: Optional[float] = None,
    empirical_rho_high: Optional[float] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return one summary row plus detailed shrinkage/PRD/budget tables."""
    y_log = np.asarray(y_log, dtype=float).reshape(-1)
    f0_log = np.asarray(f0_log, dtype=float).reshape(-1)
    mask = np.isfinite(y_log) & np.isfinite(f0_log)
    y_log = y_log[mask]
    f0_log = f0_log[mask]

    e = f0_log - y_log
    price = np.exp(y_log)

    # Baseline log-space quantities.
    A = _safe_var(f0_log)                         # G ~= Var(E[Y|X]) ~= Var(f0)
    B = float(np.mean(e ** 2)) if e.size else float("nan")
    C_log = _safe_cov(e, y_log)                   # metric-centered log covariance
    C_log_code_centered = float(np.mean(e * (y_log - np.mean(y_log)))) if e.size else float("nan")
    bayes_ratio = C_log / (-B) if B > 0 and np.isfinite(C_log) else float("nan")

    # Log-to-price bridge.
    cov_e_price = _safe_cov(e, price)
    kappa_first_order = cov_e_price / C_log if np.isfinite(cov_e_price) and np.isfinite(C_log) and C_log != 0.0 else float("nan")
    kappa_lognormal = _safe_mean(price)

    prd_info = _compute_prd_identity_quantities(y_log, f0_log)
    prd0 = prd_info["prd"]
    cov_rp0 = prd_info["cov_ratio_price_from_prd"]
    kappa_ratio_price = cov_rp0 / C_log if np.isfinite(cov_rp0) and np.isfinite(C_log) and C_log != 0.0 else float("nan")
    first_order_bridge_rel_error = (
        abs(cov_e_price - cov_rp0) / max(abs(cov_rp0), 1e-12)
        if np.isfinite(cov_e_price) and np.isfinite(cov_rp0) else float("nan")
    )

    # Step 2: rho values for generic covariance shrinkage.
    shrink_rows: List[Dict[str, Any]] = []
    for q in shrinkage_q_values:
        rho_q = _rho_from_shrinkage(q, A)
        d_mse = (C_log ** 2 / A) * ((1.0 - q) ** 2) if A > 0 and np.isfinite(C_log) else float("nan")
        shrink_rows.append({
            "data_source": data_source,
            "assessment_year": assessment_year,
            "config_key": config_key,
            "split": split_label,
            "q_remaining_covariance": float(q),
            "covariance_reduction": float(1.0 - q),
            "rho_theory": rho_q,
            "delta_mse_log_theory": d_mse,
            "delta_mse_log_frac_of_baseline": d_mse / B if B > 0 and np.isfinite(d_mse) else float("nan"),
        })
    shrink_df = pd.DataFrame(shrink_rows)

    # Step 3 + 4: PRD targets -> covariance targets -> rho.
    prd_rows: List[Dict[str, Any]] = []
    for target in prd_targets:
        if not _valid_prd_target(prd0, float(target)):
            continue
        if not all(np.isfinite(v) for v in [prd_info["mu_ratio_price"], prd_info["mu_price"], cov_rp0]):
            continue
        cov_rp_target = float(prd_info["mu_ratio_price"] * prd_info["mu_price"] * (1.0 / float(target) - 1.0))
        q_prd_direct = abs(cov_rp_target) / abs(cov_rp0) if cov_rp0 != 0.0 else float("nan")
        rho_prd_direct = _rho_from_shrinkage(q_prd_direct, A)

        # Exact baseline ratio-price bridge. This assumes the ratio-price covariance
        # shrinks proportionally with the penalized log-residual/log-price covariance.
        if np.isfinite(kappa_ratio_price) and kappa_ratio_price != 0.0 and np.isfinite(C_log) and C_log != 0.0:
            C_log_target_ratio = cov_rp_target / kappa_ratio_price
            q_log_ratio = abs(C_log_target_ratio) / abs(C_log)
            rho_log_ratio = _rho_from_shrinkage(q_log_ratio, A)
        else:
            C_log_target_ratio = float("nan")
            q_log_ratio = float("nan")
            rho_log_ratio = float("nan")

        # First-order sensitivity: exp(e) ~= 1 + e gives Cov(r, price) ~= Cov(e, price).
        # This is useful as a diagnostic, but it can be poor when log residuals are not small.
        if np.isfinite(kappa_first_order) and kappa_first_order != 0.0 and np.isfinite(C_log) and C_log != 0.0:
            C_log_target_emp = cov_rp_target / kappa_first_order
            q_log_emp = abs(C_log_target_emp) / abs(C_log)
            rho_log_emp = _rho_from_shrinkage(q_log_emp, A)
        else:
            C_log_target_emp = float("nan")
            q_log_emp = float("nan")
            rho_log_emp = float("nan")

        # Lognormal bridge sanity check: Cov(e, price) ~= E[price] Cov(e, log price).
        if np.isfinite(kappa_lognormal) and kappa_lognormal != 0.0 and np.isfinite(C_log) and C_log != 0.0:
            C_log_target_logn = cov_rp_target / kappa_lognormal
            q_log_logn = abs(C_log_target_logn) / abs(C_log)
            rho_log_logn = _rho_from_shrinkage(q_log_logn, A)
        else:
            C_log_target_logn = float("nan")
            q_log_logn = float("nan")
            rho_log_logn = float("nan")

        prd_rows.append({
            "data_source": data_source,
            "assessment_year": assessment_year,
            "config_key": config_key,
            "split": split_label,
            "prd_baseline": prd0,
            "prd_target": float(target),
            "cov_ratio_price_baseline_from_prd": cov_rp0,
            "cov_ratio_price_target_from_prd": cov_rp_target,
            "q_prd_direct_price_cov": q_prd_direct,
            "rho_prd_direct_shrinkage": rho_prd_direct,
            "kappa_ratio_price_over_cov_e_logprice": kappa_ratio_price,
            "q_log_ratio_price_bridge": q_log_ratio,
            "rho_prd_ratio_price_bridge": rho_log_ratio,
            "kappa_first_order_cov_e_price_over_cov_e_logprice": kappa_first_order,
            "first_order_bridge_rel_error_at_baseline": first_order_bridge_rel_error,
            "q_log_first_order_bridge": q_log_emp,
            "rho_prd_first_order_bridge": rho_log_emp,
            "q_log_lognormal_bridge": q_log_logn,
            "rho_prd_lognormal_bridge": rho_log_logn,
        })
    prd_df = pd.DataFrame(prd_rows)

    # Step 5a: accuracy budgets.
    budget_rows: List[Dict[str, Any]] = []
    for alpha in accuracy_budgets:
        rho_alpha = _rho_from_accuracy_budget(alpha, B=B, C0=C_log, A=A)
        q_alpha = _q_from_rho(rho_alpha, A)
        budget_rows.append({
            "data_source": data_source,
            "assessment_year": assessment_year,
            "config_key": config_key,
            "split": split_label,
            "accuracy_budget_frac_of_baseline_mse": float(alpha),
            "rho_max_under_budget": rho_alpha,
            "q_remaining_covariance_at_budget": q_alpha,
            "covariance_reduction_at_budget": 1.0 - q_alpha if np.isfinite(q_alpha) else float("nan"),
        })
    budget_df = pd.DataFrame(budget_rows)

    # Per-run decision numbers.
    rho_25 = _rho_from_shrinkage(0.75, A)
    rho_50 = _rho_from_shrinkage(0.50, A)
    rho_67 = _rho_from_shrinkage(0.33, A)

    # PRD boundary: if outside [0.98,1.03], target the nearest guidance boundary.
    guidance_target = float("nan")
    rho_prd_guidance = float("nan")
    if np.isfinite(prd0):
        if prd0 > 1.03:
            guidance_target = 1.03
        elif prd0 < 0.98:
            guidance_target = 0.98
    if np.isfinite(guidance_target) and not prd_df.empty:
        row = prd_df.loc[np.isclose(prd_df["prd_target"].astype(float), guidance_target)]
        if not row.empty:
            # Prefer the exact baseline ratio-price bridge; fall back to direct shrinkage.
            cand = float(row.iloc[0].get("rho_prd_ratio_price_bridge", np.nan))
            if not np.isfinite(cand):
                cand = float(row.iloc[0].get("rho_prd_direct_shrinkage", np.nan))
            rho_prd_guidance = cand

    rho_budget_1pct = float("nan")
    if not budget_df.empty:
        one_pct = budget_df.loc[np.isclose(budget_df["accuracy_budget_frac_of_baseline_mse"].astype(float), 0.01)]
        if not one_pct.empty:
            rho_budget_1pct = float(one_pct.iloc[0]["rho_max_under_budget"])

    lower_candidates = [rho_25]
    if np.isfinite(rho_prd_guidance):
        lower_candidates.append(rho_prd_guidance)
    theory_lower = float(np.nanmax(lower_candidates)) if lower_candidates else float("nan")

    upper_candidates = [rho_67]
    if np.isfinite(rho_budget_1pct):
        upper_candidates.append(rho_budget_1pct)
    theory_upper = float(np.nanmin(upper_candidates)) if upper_candidates else float("nan")

    # Confident rho: 50% shrinkage, clipped into [lower, upper] if that interval is valid.
    confident_rho = rho_50
    if np.isfinite(theory_lower) and np.isfinite(theory_upper) and theory_lower <= theory_upper:
        confident_rho = float(np.clip(confident_rho, theory_lower, theory_upper))

    summary = {
        "data_source": data_source,
        "assessment_year": assessment_year,
        "config_key": config_key,
        "split": split_label,
        "n": int(y_log.size),
        "A_var_f0_log": A,
        "B_mse_log": B,
        "C0_cov_log_residual_logprice": C_log,
        "bayes_optimality_diagnostic_C0_over_minus_B": bayes_ratio,
        "cov_e_price": cov_e_price,
        "kappa_first_order": kappa_first_order,
        "kappa_ratio_price": kappa_ratio_price,
        "kappa_lognormal_mean_price": kappa_lognormal,
        "first_order_bridge_rel_error_at_baseline": first_order_bridge_rel_error,
        **prd_info,
        "rho_shrink_25pct": rho_25,
        "rho_shrink_50pct": rho_50,
        "rho_shrink_67pct": rho_67,
        "prd_guidance_target": guidance_target,
        "rho_prd_guidance": rho_prd_guidance,
        "rho_budget_1pct_mse": rho_budget_1pct,
        "theory_range_low": theory_lower,
        "theory_confident_rho": confident_rho,
        "theory_range_high": theory_upper,
    }

    if empirical_rho_low is not None and empirical_rho_high is not None:
        summary["empirical_rho_low"] = float(empirical_rho_low)
        summary["empirical_rho_high"] = float(empirical_rho_high)
        summary["empirical_range_overlaps_theory"] = bool(
            np.isfinite(theory_lower)
            and np.isfinite(theory_upper)
            and max(float(empirical_rho_low), theory_lower) <= min(float(empirical_rho_high), theory_upper)
        )
        summary["empirical_low_q_implied"] = _q_from_rho(float(empirical_rho_low), A)
        summary["empirical_high_q_implied"] = _q_from_rho(float(empirical_rho_high), A)
        summary["empirical_low_cov_reduction_implied"] = 1.0 - summary["empirical_low_q_implied"]
        summary["empirical_high_cov_reduction_implied"] = 1.0 - summary["empirical_high_q_implied"]

    return summary, shrink_df, prd_df, budget_df


# ---------------------------------------------------------------------------
# Fitting/evaluation loop
# ---------------------------------------------------------------------------

RHO_SWEEP_FOLDER_RE = re.compile(
    r"^(?P<src>.+?)_assess(?P<year>\d{4})__(?P<cfg>.+)_(?P<cid>[0-9a-f]{8})$"
)


def _parse_data_source_specs(raw: str) -> List[Tuple[str, int, str]]:
    """
    Parse semicolon-separated specs of the form:
        source_label:assessment_year:path/to/file.parquet
    """
    specs: List[Tuple[str, int, str]] = []
    for chunk in str(raw or "").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(":", 2)
        if len(parts) != 3:
            raise ValueError(
                "Each --data-source-specs entry must be source:assessment_year:path; "
                f"got {chunk!r}"
            )
        src, year_raw, path = parts
        specs.append((src.strip(), int(year_raw), path.strip()))
    return specs


def _default_data_source_specs() -> str:
    return ";".join(
        [
            "ccao2025:2025:./data/CCAO/2025/training_data.parquet",
            "ccao_old:2024:./data/CCAO/2025/training_data_old.parquet",
            "ccao_sim2024:2023:./data/CCAO/2025/training_data_sim2024.parquet",
            "ccao_sim2023:2022:./data/CCAO/2025/training_data_sim2023.parquet",
        ]
    )


def _analysis_specs_from_args(args: argparse.Namespace) -> List[Tuple[str, int, str]]:
    source_specs = _parse_data_source_specs(args.data_source_specs)
    if source_specs:
        return source_specs
    data_source = str(args.data_source_label or Path(args.data_path).stem)
    return [(data_source, year, args.data_path) for year in _parse_csv_list(args.assessment_years, cast=int)]

# Theory split label -> baseline-prediction file suffix written by quick_test_models.py.
_SPLIT_FILE_SUFFIX = {"test": "test", "assessment": "assess"}


def _resolve_fit_n_jobs(requested: Optional[int]) -> int:
    """Use all CPUs available to this process for the (rare) fallback baseline fits."""
    if requested is not None:
        return int(requested)
    try:
        return max(1, len(os.sched_getaffinity(0)))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return max(1, int(os.cpu_count() or 1))


def _find_sweep_folder(sweep_root: Optional[str], data_source: str, year: int, config_key: str) -> Optional[Path]:
    if not sweep_root:
        return None
    root = Path(sweep_root)
    if not root.exists():
        return None
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        m = RHO_SWEEP_FOLDER_RE.match(folder.name)
        if m and m["src"] == data_source and int(m["year"]) == int(year) and m["cfg"] == config_key:
            return folder
    return None


def _read_prediction_parquet(path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    try:
        df = pd.read_parquet(path)
    except Exception:  # pragma: no cover - missing/corrupt cache must not break the run
        return None
    if "y_log" not in df.columns or "f0_log" not in df.columns:
        return None
    y = pd.to_numeric(df["y_log"], errors="coerce").to_numpy(dtype=float)
    f0 = pd.to_numeric(df["f0_log"], errors="coerce").to_numpy(dtype=float)
    if y.size == 0 or y.size != f0.size:
        return None
    return y, f0


def _cache_path(cache_dir: Path, data_source: str, year: int, config_key: str, split: str) -> Path:
    return cache_dir / f"{data_source}__{int(year)}__{config_key}__{split}.parquet"


def _recycle_baseline_predictions(
    *,
    data_source: str,
    year: int,
    config_key: str,
    split: str,
    sweep_root: Optional[str],
    cache_dir: Path,
) -> Tuple[Optional[Tuple[np.ndarray, np.ndarray]], str]:
    """Recycle baseline log predictions from the rho-sweep outputs, then a local cache.

    Returns ((y_log, f0_log), source_label) where source_label is one of
    {"rho_sweep", "cache", "fit"}; ("fit" means nothing recyclable was found).
    """
    folder = _find_sweep_folder(sweep_root, data_source, year, config_key)
    if folder is not None:
        suffix = _SPLIT_FILE_SUFFIX.get(split, split)
        pred = _read_prediction_parquet(folder / f"baseline_predictions_{suffix}.parquet")
        if pred is not None:
            return pred, "rho_sweep"
    pred = _read_prediction_parquet(_cache_path(cache_dir, data_source, year, config_key, split))
    if pred is not None:
        return pred, "cache"
    return None, "fit"


def fit_lgbm_and_predict(
    *,
    X_train: pd.DataFrame,
    y_train_log: np.ndarray,
    X_eval: pd.DataFrame,
    lgbm_params: dict,
    timing_label: str,
) -> np.ndarray:
    model = lgb.LGBMRegressor(**lgbm_params)
    with _timed_step(
        "lgbm: fit baseline",
        label=timing_label,
        train_rows=int(len(y_train_log)),
        n_features=int(X_train.shape[1]),
        n_estimators=int(lgbm_params.get("n_estimators", -1)),
        num_leaves=int(lgbm_params.get("num_leaves", -1)),
    ):
        model.fit(X_train, y_train_log)
    with _timed_step("lgbm: predict baseline", label=timing_label, eval_rows=int(X_eval.shape[0])):
        pred = np.asarray(model.predict(X_eval), dtype=float).reshape(-1)
    return pred


def run_analysis(args: argparse.Namespace) -> Dict[str, Path]:
    start = time.perf_counter()
    status = "ok"
    try:
        return _run_analysis_impl(args)
    except Exception:
        status = "failed"
        raise
    finally:
        duration = time.perf_counter() - start
        _TIMING_ROWS.append({"step": "run: total", "seconds": duration, "status": status})
        _log("end: run: total", duration=_format_seconds(duration), status=status)
        try:
            out_dir = Path(args.out_dir)
            if out_dir.exists():
                pd.DataFrame(_TIMING_ROWS).to_csv(out_dir / "theory_rho_timing_detail.csv", index=False)
                _timing_summary_df().to_csv(out_dir / "theory_rho_timing_summary.csv", index=False)
        except Exception as exc:  # pragma: no cover - diagnostics should not mask the real error
            _log("failed to write timing CSVs", error=repr(exc))


def _run_analysis_impl(args: argparse.Namespace) -> Dict[str, Path]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with _timed_step("config: load yaml"):
        with open(args.params_path, "r", encoding="utf-8") as f:
            params = yaml.safe_load(f)
        with open(args.model_params_path, "r", encoding="utf-8") as f:
            model_params = yaml.safe_load(f)

    with _timed_step("config: parse arguments"):
        analysis_specs = _analysis_specs_from_args(args)
        config_keys = _parse_csv_list(args.lgbm_config_keys, cast=str)
        prd_targets = _parse_csv_list(args.prd_targets, cast=float)
        shrinkage_q_values = _parse_csv_list(args.shrinkage_q_values, cast=float)
        accuracy_budgets = _parse_csv_list(args.accuracy_budgets, cast=float)
    _log(
        "analysis matrix",
        data_sources=len(analysis_specs),
        configs=len(config_keys),
        max_baseline_fits=len(analysis_specs) * len(config_keys) * 2,
    )

    empirical_low = None
    empirical_high = None
    empirical_parts = _parse_csv_list(args.empirical_rho_range, cast=float) if args.empirical_rho_range else []
    if len(empirical_parts) == 2:
        empirical_low, empirical_high = float(empirical_parts[0]), float(empirical_parts[1])

    cache_dir = Path(args.baseline_cache_dir) if args.baseline_cache_dir else (out_dir / "baseline_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    fit_n_jobs = _resolve_fit_n_jobs(args.lgbm_n_jobs)
    provenance = {"rho_sweep": 0, "cache": 0, "fit": 0}

    all_summary: List[Dict[str, Any]] = []
    all_shrink: List[pd.DataFrame] = []
    all_prd: List[pd.DataFrame] = []
    all_budget: List[pd.DataFrame] = []

    def _build_config_params(config_key: str) -> dict:
        if args.lgbm_hyperparameter_file:
            p = _load_lgbm_params_from_hyperparameter_file(args.lgbm_hyperparameter_file, config_key)
        else:
            p = _build_lgbm_params_from_files(model_params, params, args.seed)
        if args.lgbm_n_estimators is not None:
            p["n_estimators"] = int(args.lgbm_n_estimators)
        p["n_jobs"] = int(fit_n_jobs)
        p["random_state"] = int(p.get("random_state", args.seed))
        return p

    for data_source, year, data_path in analysis_specs:
        # Plan: decide per (config, split) whether the baseline log predictions can be
        # recycled (from the rho-sweep outputs or a local cache) or must be fit here.
        plan: Dict[Tuple[str, str], Tuple[Optional[Tuple[np.ndarray, np.ndarray]], str]] = {}
        need_fit = False
        for config_key in config_keys:
            for split in ("test", "assessment"):
                pred, source = _recycle_baseline_predictions(
                    data_source=data_source,
                    year=year,
                    config_key=config_key,
                    split=split,
                    sweep_root=args.rho_sweep_root,
                    cache_dir=cache_dir,
                )
                plan[(config_key, split)] = (pred, source)
                if pred is None:
                    need_fit = True

        # Load and split the dataset only if at least one baseline fit is actually needed.
        data_ctx: Optional[Dict[str, Any]] = None
        if need_fit:
            with _timed_step("dataset: load and split", data_source=data_source, assessment_year=year):
                df_tv, df_test, df_assess, predictor_cols, categorical_cols = _load_and_split_data(
                    data_path=data_path,
                    params=params,
                    target_column=args.target_column,
                    date_column=args.date_column,
                    sample_frac=args.sample_frac,
                    sample_seed=args.seed,
                    assessment_year=year,
                )
            if df_tv.empty or df_test.empty:
                _log("skipping year due to empty train/test", assessment_year=year)
                continue
            with _timed_step("dataset: build matrices", data_source=data_source, assessment_year=year):
                df_pre = pd.concat([df_tv, df_test], ignore_index=True)
                data_ctx = {
                    "X_tv": _cast_categoricals(df_tv[predictor_cols], categorical_cols),
                    "y_tv_log": np.log(df_tv[args.target_column].to_numpy(dtype=float)),
                    "X_test": _cast_categoricals(df_test[predictor_cols], categorical_cols),
                    "y_test_log": np.log(df_test[args.target_column].to_numpy(dtype=float)),
                    "X_pre": _cast_categoricals(df_pre[predictor_cols], categorical_cols),
                    "y_pre_log": np.log(df_pre[args.target_column].to_numpy(dtype=float)),
                    "X_assess": _cast_categoricals(df_assess[predictor_cols], categorical_cols) if not df_assess.empty else None,
                    "y_assess_log": np.log(df_assess[args.target_column].to_numpy(dtype=float)) if not df_assess.empty else None,
                }
        else:
            _log("recycling all baselines for dataset (no fit, no data load)", data_source=data_source, assessment_year=year)

        for config_key in config_keys:
            lgbm_params: Optional[dict] = None
            for split in ("test", "assessment"):
                pred, source = plan[(config_key, split)]

                if source == "fit":
                    if data_ctx is None:
                        continue
                    if split == "assessment" and (
                        data_ctx["X_assess"] is None
                        or data_ctx["y_assess_log"] is None
                        or len(data_ctx["y_assess_log"]) == 0
                    ):
                        continue
                    if lgbm_params is None:
                        with _timed_step("config: build lgbm params", data_source=data_source, assessment_year=year, config_key=config_key):
                            lgbm_params = _build_config_params(config_key)
                    if split == "test":
                        X_tr, y_tr, X_ev, y_ev = data_ctx["X_tv"], data_ctx["y_tv_log"], data_ctx["X_test"], data_ctx["y_test_log"]
                    else:
                        X_tr, y_tr, X_ev, y_ev = data_ctx["X_pre"], data_ctx["y_pre_log"], data_ctx["X_assess"], data_ctx["y_assess_log"]
                    f0_log = fit_lgbm_and_predict(
                        X_train=X_tr,
                        y_train_log=y_tr,
                        X_eval=X_ev,
                        lgbm_params=lgbm_params,
                        timing_label=f"{data_source}/{year}/{config_key}/{split}",
                    )
                    y_log = np.asarray(y_ev, dtype=float).reshape(-1)
                    try:
                        pd.DataFrame({"y_log": y_log, "f0_log": f0_log}).to_parquet(
                            _cache_path(cache_dir, data_source, year, config_key, split), index=False
                        )
                    except Exception as exc:  # pragma: no cover
                        _log("failed to cache baseline predictions", error=repr(exc))
                else:
                    y_log, f0_log = pred  # type: ignore[misc]

                provenance[source] = provenance.get(source, 0) + 1
                with _timed_step(
                    "theory: compute split quantities",
                    data_source=data_source, assessment_year=year, config_key=config_key, split=split, baseline_source=source,
                ):
                    summary, shrink_df, prd_df, budget_df = compute_theory_for_predictions(
                        y_log=y_log,
                        f0_log=f0_log,
                        split_label=split,
                        data_source=data_source,
                        config_key=config_key,
                        assessment_year=year,
                        prd_targets=prd_targets,
                        shrinkage_q_values=shrinkage_q_values,
                        accuracy_budgets=accuracy_budgets,
                        empirical_rho_low=empirical_low,
                        empirical_rho_high=empirical_high,
                    )
                summary["baseline_source"] = source
                all_summary.append(summary)
                all_shrink.append(shrink_df)
                all_prd.append(prd_df)
                all_budget.append(budget_df)

                with _timed_step(
                    "output: write intermediate split artifacts",
                    data_source=data_source, assessment_year=year, config_key=config_key, split=split,
                ):
                    written = _write_intermediate_artifacts(
                        out_dir=out_dir,
                        summary=summary,
                        shrink_df=shrink_df,
                        prd_df=prd_df,
                        budget_df=budget_df,
                        args=args,
                    )
                    if written:
                        _log("intermediate artifacts written", count=len(written), first=str(written[0]))

    _log("baseline provenance", recycled_rho_sweep=provenance["rho_sweep"], recycled_cache=provenance["cache"], fitted=provenance["fit"])

    with _timed_step("output: assemble dataframes"):
        summary_df = pd.DataFrame(all_summary)
        shrink_df = pd.concat(all_shrink, ignore_index=True) if all_shrink else pd.DataFrame()
        prd_df = pd.concat(all_prd, ignore_index=True) if all_prd else pd.DataFrame()
        budget_df = pd.concat(all_budget, ignore_index=True) if all_budget else pd.DataFrame()

    summary_path = out_dir / "theory_rho_summary_by_run.csv"
    shrink_path = out_dir / "theory_rho_shrinkage_targets.csv"
    prd_path = out_dir / "theory_rho_prd_targets.csv"
    budget_path = out_dir / "theory_rho_accuracy_budgets.csv"
    aggregate_path = out_dir / "theory_rho_aggregate_recommendation.csv"
    report_path = out_dir / "theory_rho_report.md"

    timing_detail_path = out_dir / "theory_rho_timing_detail.csv"
    timing_summary_path = out_dir / "theory_rho_timing_summary.csv"

    with _timed_step("output: write theory csvs"):
        summary_df.to_csv(summary_path, index=False)
        shrink_df.to_csv(shrink_path, index=False)
        prd_df.to_csv(prd_path, index=False)
        budget_df.to_csv(budget_path, index=False)

    with _timed_step("output: aggregate theory ranges"):
        aggregate_df = aggregate_theory_ranges(summary_df)
        aggregate_df.to_csv(aggregate_path, index=False)

    with _timed_step("empirical: load sweep metrics", rho_sweep_root=args.rho_sweep_root):
        empirical_df = _load_empirical_sweep_metrics(args.rho_sweep_root) if args.rho_sweep_root else pd.DataFrame()
    comparison_df, ops_df, plot_paths = make_theory_empirical_plots(summary_df, empirical_df, out_dir)

    with _timed_step("output: render report"):
        report_text = render_markdown_report(
            summary_df,
            shrink_df,
            prd_df,
            budget_df,
            aggregate_df,
            args,
            comparison_df=comparison_df,
            ops_df=ops_df,
            plot_paths=plot_paths,
        )
        report_path.write_text(report_text, encoding="utf-8")

    with _timed_step("output: write timing csvs"):
        pd.DataFrame(_TIMING_ROWS).to_csv(timing_detail_path, index=False)
        _timing_summary_df().to_csv(timing_summary_path, index=False)

    _log("done", report=str(report_path))
    result = {
        "summary": summary_path,
        "shrinkage": shrink_path,
        "prd": prd_path,
        "budget": budget_path,
        "aggregate": aggregate_path,
        "report": report_path,
        "timing_detail": timing_detail_path,
        "timing_summary": timing_summary_path,
    }
    for path in plot_paths:
        result[path.stem] = path
    return result


def aggregate_theory_ranges(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for split_name, df in [("all", summary_df), *list(summary_df.groupby("split"))]:
        if isinstance(split_name, tuple):
            # defensive; pandas groupby with a single key returns scalar keys
            split_name = str(split_name[0])
        d = df.copy()
        low = pd.to_numeric(d["theory_range_low"], errors="coerce")
        high = pd.to_numeric(d["theory_range_high"], errors="coerce")
        conf = pd.to_numeric(d["theory_confident_rho"], errors="coerce")
        rho50 = pd.to_numeric(d["rho_shrink_50pct"], errors="coerce")

        # Orange plot bands used in rho-evolution diagnostics.
        # Log-MSE: raw-rho inflection rho = 1/A = rho50/2, band [0.5*rho*, 2*rho*].
        # Covariance: log-rho inflection rho = 2/A = rho50, transition band
        # [rho*/3, 3*rho*] (~25-75% covariance reduction).
        mse_band_low = 0.25 * rho50
        mse_ref = 0.50 * rho50
        mse_band_high = rho50
        cov_band_low = rho50 / 3.0
        cov_ref = rho50
        cov_band_high = 3.0 * rho50
        overlap_low = pd.concat([mse_band_low, cov_band_low], axis=1).max(axis=1)
        overlap_high = pd.concat([mse_band_high, cov_band_high], axis=1).min(axis=1)
        overlap_ref = np.sqrt(overlap_low * overlap_high).where(
            (overlap_low > 0.0) & (overlap_high >= overlap_low)
        )

        # Robust intersection-like range: 75th percentile lower bound and 25th percentile upper bound.
        # If empty/reversed, fall back to median lower/upper.
        robust_low = float(np.nanquantile(low, 0.75)) if low.notna().any() else float("nan")
        robust_high = float(np.nanquantile(high, 0.25)) if high.notna().any() else float("nan")
        if np.isfinite(robust_low) and np.isfinite(robust_high) and robust_low > robust_high:
            robust_low = float(np.nanmedian(low)) if low.notna().any() else float("nan")
            robust_high = float(np.nanmedian(high)) if high.notna().any() else float("nan")

        row = {
            "split_group": str(split_name),
            "n_runs": int(d.shape[0]),
            "robust_theory_range_low": robust_low,
            "robust_theory_range_high": robust_high,
            "median_theory_range_low": float(np.nanmedian(low)) if low.notna().any() else float("nan"),
            "median_confident_rho": float(np.nanmedian(conf)) if conf.notna().any() else float("nan"),
            "median_theory_range_high": float(np.nanmedian(high)) if high.notna().any() else float("nan"),
            "median_rho_shrink_25pct": float(np.nanmedian(pd.to_numeric(d["rho_shrink_25pct"], errors="coerce"))),
            "median_rho_shrink_50pct": float(np.nanmedian(rho50)),
            "median_rho_shrink_67pct": float(np.nanmedian(pd.to_numeric(d["rho_shrink_67pct"], errors="coerce"))),
            "median_orange_mse_band_low": float(np.nanmedian(mse_band_low)),
            "median_orange_mse_ref_rho": float(np.nanmedian(mse_ref)),
            "median_orange_mse_band_high": float(np.nanmedian(mse_band_high)),
            "median_orange_cov_band_low": float(np.nanmedian(cov_band_low)),
            "median_orange_cov_ref_rho": float(np.nanmedian(cov_ref)),
            "median_orange_cov_band_high": float(np.nanmedian(cov_band_high)),
            "median_orange_overlap_low": float(np.nanmedian(overlap_low)),
            "median_orange_overlap_ref_rho": float(np.nanmedian(overlap_ref)),
            "median_orange_overlap_high": float(np.nanmedian(overlap_high)),
            "median_bayes_diagnostic_C0_over_minus_B": float(np.nanmedian(pd.to_numeric(d["bayes_optimality_diagnostic_C0_over_minus_B"], errors="coerce"))),
        }
        if "empirical_range_overlaps_theory" in d.columns:
            row["empirical_overlap_rate"] = float(pd.Series(d["empirical_range_overlaps_theory"]).mean())
            row["median_empirical_low_cov_reduction_implied"] = float(np.nanmedian(pd.to_numeric(d.get("empirical_low_cov_reduction_implied"), errors="coerce")))
            row["median_empirical_high_cov_reduction_implied"] = float(np.nanmedian(pd.to_numeric(d.get("empirical_high_cov_reduction_implied"), errors="coerce")))
        rows.append(row)
    return pd.DataFrame(rows)


def _safe_abs_ratio(value: float, baseline: float, target: float) -> float:
    value = float(value)
    baseline = float(baseline)
    denom = abs(baseline - target)
    if not (np.isfinite(value) and np.isfinite(baseline)) or denom <= 1e-12:
        return float("nan")
    return abs(value - target) / denom


def _first_existing_column(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    colset = set(columns)
    for col in candidates:
        if col in colset:
            return col
    return None


def _load_empirical_sweep_metrics(root: str) -> pd.DataFrame:
    root_path = Path(root)
    if not root_path.exists():
        _log("empirical sweep root not found", root=str(root_path))
        return pd.DataFrame()

    rows: List[pd.DataFrame] = []
    usecols = {
        "model_name", "model_family", "ratio_mode", "rho", "rho_group", "eta", "keep",
        "R2", "R2 (log)", "RMSE", "MAE", "MAPE", "MdAPE", "COD", "PRD", "PRB", "VEI",
        "MSE_log", "log_MSE", "Log MSE", "log_mse", "mse_log",
        "RMSE_log", "log_RMSE", "Log RMSE", "log_rmse", "rmse_log",
        "C_log_resid_logprice", "Slope_log_resid_logprice", "Corr_log_resid_logprice",
        "C_log_resid_price", "C_ratio_price", "C_ratio_logprice",
        "PRB_proxy_ratio_logprice", "VEI_log_proxy_fixed_deciles",
        "C_ratio_price_taylor1", "C_ratio_price_taylor2",
        "C_ratio_price_taylor1_rel_error", "C_ratio_price_taylor2_rel_error",
        "PRD_taylor1_abs_error", "PRD_taylor2_abs_error",
        "Median ratio", "Mean ratio", "W. Mean ratio", "MKI",
    }
    split_files = {
        "assessment": "quick_test_metrics_assess.csv",
        "test": "quick_test_metrics_test.csv",
        "validation": "quick_test_metrics_validation_bootstrap_avg.csv",
    }
    files_read = 0
    for folder in sorted(p for p in root_path.iterdir() if p.is_dir()):
        m = RHO_SWEEP_FOLDER_RE.match(folder.name)
        if not m:
            continue
        for split, fname in split_files.items():
            path = folder / fname
            if not path.exists():
                continue
            with _timed_step("empirical: read metric csv", split=split, folder=folder.name):
                df = pd.read_csv(path, usecols=lambda c: c in usecols)
            df["data_source"] = m["src"]
            df["assessment_year"] = int(m["year"])
            df["config_key"] = m["cfg"]
            df["config_id"] = m["cid"]
            df["split"] = split
            df["empirical_folder"] = str(folder)
            rows.append(df)
            files_read += 1
    if not rows:
        _log("no empirical sweep metric CSVs matched", root=str(root_path))
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    _log("empirical metrics loaded", files=files_read, rows=int(out.shape[0]), cols=int(out.shape[1]))
    return out


def build_theory_empirical_comparison(summary_df: pd.DataFrame, empirical_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty or empirical_df.empty:
        return pd.DataFrame()

    theory = summary_df.copy()
    theory["split"] = theory["split"].replace({"assess": "assessment"})
    key = ["data_source", "assessment_year", "config_key", "split"]
    theory_cols = key + [
        "A_var_f0_log", "B_mse_log", "C0_cov_log_residual_logprice",
        "prd", "theory_range_low", "theory_confident_rho", "theory_range_high",
        "rho_shrink_25pct", "rho_shrink_50pct", "rho_shrink_67pct",
        "rho_prd_guidance", "rho_budget_1pct_mse",
    ]
    theory_cols = [c for c in theory_cols if c in theory.columns]
    theory = theory.loc[:, theory_cols].rename(columns={"prd": "theory_baseline_prd"})

    emp = empirical_df[empirical_df["model_family"].astype(str).eq("LGBCovPenalty[diff]")].copy()
    emp = emp[emp["split"].isin(["assessment", "test"])].copy()
    if emp.empty:
        return pd.DataFrame()

    baseline = empirical_df[empirical_df["model_name"].astype(str).eq("LGBMRegressor")].copy()
    baseline = baseline[baseline["split"].isin(["assessment", "test"])].copy()
    baseline_cols = key + [
        "R2", "R2 (log)", "RMSE", "MdAPE", "COD", "PRD", "PRB", "VEI",
        "MSE_log", "log_MSE", "Log MSE", "log_mse", "mse_log",
        "RMSE_log", "log_RMSE", "Log RMSE", "log_rmse", "rmse_log",
        "C_log_resid_logprice", "Slope_log_resid_logprice", "Corr_log_resid_logprice",
        "C_log_resid_price", "C_ratio_price", "C_ratio_logprice",
        "PRB_proxy_ratio_logprice", "VEI_log_proxy_fixed_deciles",
        "C_ratio_price_taylor1", "C_ratio_price_taylor2",
        "C_ratio_price_taylor1_rel_error", "C_ratio_price_taylor2_rel_error",
        "PRD_taylor1_abs_error", "PRD_taylor2_abs_error",
    ]
    baseline = baseline[[c for c in baseline_cols if c in baseline.columns]].rename(
        columns={c: f"baseline_{c}" for c in baseline_cols if c not in key}
    )

    out = emp.merge(theory, on=key, how="inner").merge(baseline, on=key, how="left")
    if out.empty:
        return out

    out["rho"] = pd.to_numeric(out["rho"], errors="coerce")
    for col in ["A_var_f0_log", "B_mse_log", "C0_cov_log_residual_logprice"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["q_theory_remaining_covariance"] = [
        _q_from_rho(r, a) for r, a in zip(out["rho"], out["A_var_f0_log"])
    ]
    out["covariance_reduction_theory"] = 1.0 - out["q_theory_remaining_covariance"]
    out["delta_mse_log_frac_theory"] = (
        (out["C0_cov_log_residual_logprice"] ** 2 / out["A_var_f0_log"])
        * ((1.0 - out["q_theory_remaining_covariance"]) ** 2)
        / out["B_mse_log"]
    )
    if "C_log_resid_logprice" in out.columns and "baseline_C_log_resid_logprice" in out.columns:
        denom = out["baseline_C_log_resid_logprice"].replace(0.0, np.nan)
        out["q_empirical_signed"] = out["C_log_resid_logprice"] / denom
        out["q_empirical_abs"] = out["C_log_resid_logprice"].abs() / denom.abs()
        out["covariance_reduction_empirical"] = 1.0 - out["q_empirical_signed"]
        out["covariance_abs_reduction_empirical"] = 1.0 - out["q_empirical_abs"]
        out["q_error_empirical_minus_theory"] = out["q_empirical_signed"] - out["q_theory_remaining_covariance"]
        out["C_log_resid_logprice_theory"] = out["q_theory_remaining_covariance"] * out["baseline_C_log_resid_logprice"]
        out["C_log_resid_logprice_error"] = out["C_log_resid_logprice"] - out["C_log_resid_logprice_theory"]
    if "Slope_log_resid_logprice" in out.columns and "baseline_Slope_log_resid_logprice" in out.columns:
        out["Slope_log_resid_logprice_theory"] = (
            out["q_theory_remaining_covariance"] * out["baseline_Slope_log_resid_logprice"]
        )
        out["Slope_log_resid_logprice_error"] = (
            out["Slope_log_resid_logprice"] - out["Slope_log_resid_logprice_theory"]
        )
    if "PRB_proxy_ratio_logprice" in out.columns and "baseline_PRB_proxy_ratio_logprice" in out.columns:
        out["PRB_proxy_ratio_logprice_theory"] = (
            out["q_theory_remaining_covariance"] * out["baseline_PRB_proxy_ratio_logprice"]
        )
        out["PRB_proxy_ratio_logprice_error"] = (
            out["PRB_proxy_ratio_logprice"] - out["PRB_proxy_ratio_logprice_theory"]
        )
    if "VEI_log_proxy_fixed_deciles" in out.columns and "baseline_VEI_log_proxy_fixed_deciles" in out.columns:
        out["VEI_log_proxy_fixed_deciles_theory"] = (
            out["q_theory_remaining_covariance"] * out["baseline_VEI_log_proxy_fixed_deciles"]
        )
        out["VEI_log_proxy_fixed_deciles_error"] = (
            out["VEI_log_proxy_fixed_deciles"] - out["VEI_log_proxy_fixed_deciles_theory"]
        )

    for metric in [
        "R2", "R2 (log)", "RMSE", "MdAPE", "COD", "PRD", "PRB", "VEI",
        "MSE_log", "log_MSE", "Log MSE", "log_mse", "mse_log",
        "RMSE_log", "log_RMSE", "Log RMSE", "log_rmse", "rmse_log",
        "C_log_resid_logprice", "Slope_log_resid_logprice", "Corr_log_resid_logprice",
        "C_log_resid_price", "C_ratio_price", "C_ratio_logprice",
        "PRB_proxy_ratio_logprice", "VEI_log_proxy_fixed_deciles",
        "C_ratio_price_taylor1", "C_ratio_price_taylor2",
        "C_ratio_price_taylor1_rel_error", "C_ratio_price_taylor2_rel_error",
        "PRD_taylor1_abs_error", "PRD_taylor2_abs_error",
    ]:
        if metric in out.columns:
            out[metric] = pd.to_numeric(out[metric], errors="coerce")
        bcol = f"baseline_{metric}"
        if bcol in out.columns:
            out[bcol] = pd.to_numeric(out[bcol], errors="coerce")

    out["empirical_R2_delta"] = out["R2"] - out["baseline_R2"]
    out["empirical_RMSE_frac_delta"] = out["RMSE"] / out["baseline_RMSE"] - 1.0
    out["empirical_MdAPE_frac_delta"] = out["MdAPE"] / out["baseline_MdAPE"] - 1.0
    out["empirical_COD_frac_delta"] = out["COD"] / out["baseline_COD"] - 1.0

    log_mse_col = _first_existing_column(
        out.columns, ["MSE_log", "log_MSE", "Log MSE", "log_mse", "mse_log"]
    )
    if log_mse_col is not None and f"baseline_{log_mse_col}" in out.columns:
        out["empirical_MSE_log_frac_delta"] = (
            out[log_mse_col] / out[f"baseline_{log_mse_col}"] - 1.0
        )
        out["empirical_delta_MSE_log"] = out[log_mse_col] - out[f"baseline_{log_mse_col}"]
        out["delta_MSE_log_theory"] = (
            (out["C0_cov_log_residual_logprice"] ** 2 / out["A_var_f0_log"])
            * ((1.0 - out["q_theory_remaining_covariance"]) ** 2)
        )
        out["delta_MSE_log_error"] = out["empirical_delta_MSE_log"] - out["delta_MSE_log_theory"]
    else:
        log_rmse_col = _first_existing_column(
            out.columns, ["RMSE_log", "log_RMSE", "Log RMSE", "log_rmse", "rmse_log"]
        )
        if log_rmse_col is not None and f"baseline_{log_rmse_col}" in out.columns:
            out["empirical_MSE_log_frac_delta"] = (
                (out[log_rmse_col] ** 2) / (out[f"baseline_{log_rmse_col}"] ** 2) - 1.0
            )
        elif "R2 (log)" in out.columns and "baseline_R2 (log)" in out.columns:
            baseline_log_sse_frac = 1.0 - out["baseline_R2 (log)"]
            out["empirical_MSE_log_frac_delta"] = np.where(
                np.isfinite(baseline_log_sse_frac) & (np.abs(baseline_log_sse_frac) > 1e-12),
                (1.0 - out["R2 (log)"]) / baseline_log_sse_frac - 1.0,
                np.nan,
            )
    out["empirical_abs_PRD_minus_1_ratio"] = [
        _safe_abs_ratio(v, b, 1.0) for v, b in zip(out["PRD"], out["baseline_PRD"])
    ]
    out["empirical_abs_PRB_ratio"] = [
        _safe_abs_ratio(v, b, 0.0) for v, b in zip(out["PRB"], out["baseline_PRB"])
    ]
    out["empirical_abs_VEI_ratio"] = [
        _safe_abs_ratio(v, b, 0.0) for v, b in zip(out["VEI"], out["baseline_VEI"])
    ]
    out["empirical_PRD_error_reduction"] = 1.0 - out["empirical_abs_PRD_minus_1_ratio"]
    out["empirical_PRB_error_reduction"] = 1.0 - out["empirical_abs_PRB_ratio"]
    out["empirical_VEI_error_reduction"] = 1.0 - out["empirical_abs_VEI_ratio"]
    out["inside_theory_range"] = (
        (out["rho"] >= out["theory_range_low"])
        & (out["rho"] <= out["theory_range_high"])
    )
    return out


def empirical_operating_points(empirical_df: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
    if empirical_df.empty or summary_df.empty:
        return pd.DataFrame()
    key = ["data_source", "assessment_year", "config_key"]
    theory = summary_df[summary_df["split"].isin(["assessment", "assess"])].copy()
    if theory.empty:
        return pd.DataFrame()
    theory = theory.loc[:, key + [
        "theory_range_low", "theory_confident_rho", "theory_range_high",
        "rho_shrink_25pct", "rho_shrink_50pct", "rho_shrink_67pct",
    ]]

    val = empirical_df[
        empirical_df["split"].eq("validation")
        & empirical_df["model_family"].astype(str).eq("LGBCovPenalty[diff]")
    ].copy()
    assess = empirical_df[
        empirical_df["split"].eq("assessment")
        & empirical_df["model_family"].astype(str).eq("LGBCovPenalty[diff]")
    ].copy()
    if val.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for group_key, d in val.groupby(key):
        d = d.copy()
        choices = {
            "validation min COD": d["COD"].idxmin() if "COD" in d else None,
            "validation min MdAPE": d["MdAPE"].idxmin() if "MdAPE" in d else None,
            "validation PRD closest": (d["PRD"] - 1.0).abs().idxmin() if "PRD" in d else None,
            "validation max R2": d["R2"].idxmax() if "R2" in d else None,
        }
        for criterion, idx in choices.items():
            if idx is None:
                continue
            selected = d.loc[idx]
            row = {
                "data_source": group_key[0],
                "assessment_year": int(group_key[1]),
                "config_key": group_key[2],
                "criterion": criterion,
                "selected_rho": float(selected["rho"]),
                "validation_R2": float(selected.get("R2", np.nan)),
                "validation_COD": float(selected.get("COD", np.nan)),
                "validation_PRD": float(selected.get("PRD", np.nan)),
                "validation_PRB": float(selected.get("PRB", np.nan)),
            }
            ad = assess[
                (assess["data_source"].eq(group_key[0]))
                & (assess["assessment_year"].eq(group_key[1]))
                & (assess["config_key"].eq(group_key[2]))
            ].copy()
            if not ad.empty:
                nearest_idx = (pd.to_numeric(ad["rho"], errors="coerce") - float(selected["rho"])).abs().idxmin()
                ar = ad.loc[nearest_idx]
                row.update({
                    "assessment_R2": float(ar.get("R2", np.nan)),
                    "assessment_COD": float(ar.get("COD", np.nan)),
                    "assessment_PRD": float(ar.get("PRD", np.nan)),
                    "assessment_PRB": float(ar.get("PRB", np.nan)),
                })
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.merge(theory, on=key, how="left")
    out["inside_theory_range"] = (
        (out["selected_rho"] >= out["theory_range_low"])
        & (out["selected_rho"] <= out["theory_range_high"])
    )
    return out


def _plot_save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def _safe_slug(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_") or "na"


def _atomic_to_csv(df: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    """Write a CSV via a temporary file so partial files are not left after crashes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp_{os.getpid()}")
    df.to_csv(tmp, index=index)
    os.replace(tmp, path)


def _append_to_csv(df: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    """Append to a CSV, writing a header only when the file does not exist."""
    if df is None or df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, mode="a", header=not path.exists(), index=index)


def _single_run_stem(summary: Dict[str, Any]) -> str:
    return "__".join(
        [
            _safe_slug(summary.get("data_source", "source")),
            f"assess{int(summary.get('assessment_year', 0))}",
            _safe_slug(summary.get("config_key", "config")),
            _safe_slug(summary.get("split", "split")),
        ]
    )


def _plot_single_run_theory_curves(
    *,
    summary: Dict[str, Any],
    shrink_df: pd.DataFrame,
    prd_df: pd.DataFrame,
    budget_df: pd.DataFrame,
    out_path: Path,
) -> Optional[Path]:
    """Save a compact per-run plot immediately after a split is computed."""
    if not summary:
        return None

    A = float(summary.get("A_var_f0_log", np.nan))
    C0 = float(summary.get("C0_cov_log_residual_logprice", np.nan))
    B = float(summary.get("B_mse_log", np.nan))
    if not (np.isfinite(A) and A > 0.0 and np.isfinite(C0) and np.isfinite(B) and B > 0.0):
        return None

    lo = float(summary.get("theory_range_low", np.nan))
    hi = float(summary.get("theory_range_high", np.nan))
    conf = float(summary.get("theory_confident_rho", np.nan))
    rho_grid = np.geomspace(1e-3, 100.0, 400)
    q_grid = np.array([_q_from_rho(r, A) for r in rho_grid], dtype=float)
    cov_red = 1.0 - q_grid
    delta_mse_frac = ((C0 ** 2 / A) * (cov_red ** 2)) / B

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))

    ax = axes[0]
    ax.plot(rho_grid, cov_red, color="#2563EB", lw=2.0)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("rho")
    ax.set_ylabel("theory covariance reduction")
    ax.set_title("Shrinkage curve")
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(rho_grid, delta_mse_frac, color="#DC2626", lw=2.0)
    for alpha in pd.to_numeric(budget_df.get("accuracy_budget_frac_of_baseline_mse", pd.Series(dtype=float)), errors="coerce").dropna().unique():
        ax.axhline(float(alpha), color="#6B7280", ls=":", lw=0.8, alpha=0.55)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("rho")
    ax.set_ylabel("predicted ΔMSE_log / baseline MSE_log")
    ax.set_title("Accuracy budget")
    ax.grid(alpha=0.25, which="both")

    ax = axes[2]
    if prd_df is not None and not prd_df.empty and "prd_target" in prd_df.columns:
        x = pd.to_numeric(prd_df["prd_target"], errors="coerce")
        y = pd.to_numeric(prd_df.get("rho_prd_ratio_price_bridge", prd_df.get("rho_prd_direct_shrinkage")), errors="coerce")
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.any():
            ax.scatter(x[mask], y[mask], color="#16A34A", s=45)
            for xx, yy in zip(x[mask], y[mask]):
                ax.annotate(f"{xx:.3g}", (xx, yy), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.axhline(conf, color="#111827", ls="--", lw=1.2, alpha=0.8, label="confident rho") if np.isfinite(conf) else None
    ax.set_xlabel("PRD target")
    ax.set_ylabel("rho implied by PRD target")
    ax.set_title("PRD target map")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")

    for ax in axes:
        if np.isfinite(lo) and np.isfinite(hi) and lo > 0 and hi > lo:
            ax.axvspan(lo, hi, color="#2563EB", alpha=0.08, label="theory range") if ax is not axes[2] else None
        if np.isfinite(conf) and conf > 0:
            ax.axvline(conf, color="#111827", ls="--", lw=1.1, alpha=0.85) if ax is not axes[2] else None

    title = (
        f"{summary.get('data_source')} | assess{summary.get('assessment_year')} | "
        f"{summary.get('config_key')} | {summary.get('split')}"
    )
    fig.suptitle(title, fontweight="bold")
    return _plot_save(fig, out_path)


def _write_intermediate_artifacts(
    *,
    out_dir: Path,
    summary: Dict[str, Any],
    shrink_df: pd.DataFrame,
    prd_df: pd.DataFrame,
    budget_df: pd.DataFrame,
    args: argparse.Namespace,
) -> List[Path]:
    """Persist per-split results immediately so a later crash does not lose them."""
    if not getattr(args, "write_intermediate_results", True):
        return []

    stem = _single_run_stem(summary)
    checkpoint_dir = out_dir / "checkpoints"
    plots_dir = out_dir / "plots" / "per_run"
    paths: List[Path] = []

    summary_df = pd.DataFrame([summary])
    file_map = [
        (summary_df, checkpoint_dir / f"{stem}__summary.csv"),
        (shrink_df, checkpoint_dir / f"{stem}__shrinkage.csv"),
        (prd_df, checkpoint_dir / f"{stem}__prd_targets.csv"),
        (budget_df, checkpoint_dir / f"{stem}__accuracy_budgets.csv"),
    ]
    for df, path in file_map:
        _atomic_to_csv(df, path, index=False)
        paths.append(path)

    # Append-only journal CSVs are convenient for tailing progress while a job runs.
    _append_to_csv(summary_df, out_dir / "theory_rho_summary_by_run_incremental.csv", index=False)
    _append_to_csv(shrink_df, out_dir / "theory_rho_shrinkage_targets_incremental.csv", index=False)
    _append_to_csv(prd_df, out_dir / "theory_rho_prd_targets_incremental.csv", index=False)
    _append_to_csv(budget_df, out_dir / "theory_rho_accuracy_budgets_incremental.csv", index=False)

    if getattr(args, "write_intermediate_plots", True):
        plot_path = plots_dir / f"{stem}__theory_tradeoff.{getattr(args, 'plot_format', 'png')}"
        maybe = _plot_single_run_theory_curves(
            summary=summary, shrink_df=shrink_df, prd_df=prd_df, budget_df=budget_df, out_path=plot_path
        )
        if maybe is not None:
            paths.append(maybe)
    return paths


def plot_theory_empirical_ranges(ops_df: pd.DataFrame, path: Path) -> Optional[Path]:
    if ops_df.empty:
        return None
    base = ops_df.drop_duplicates(["data_source", "assessment_year", "config_key"]).copy()
    base = base.sort_values(["assessment_year", "config_key", "data_source"]).reset_index(drop=True)
    labels = [
        f"{int(r.assessment_year)} {r.config_key}\n{r.data_source}"
        for r in base.itertuples(index=False)
    ]
    y = np.arange(len(base))
    fig, ax = plt.subplots(figsize=(11, max(4.5, 0.38 * len(base) + 1.2)))
    low = pd.to_numeric(base["theory_range_low"], errors="coerce").to_numpy()
    high = pd.to_numeric(base["theory_range_high"], errors="coerce").to_numpy()
    conf = pd.to_numeric(base["theory_confident_rho"], errors="coerce").to_numpy()
    for i, (lo, hi, cf) in enumerate(zip(low, high, conf)):
        if np.isfinite(lo) and np.isfinite(hi):
            ax.hlines(i, lo, hi, color="#2563EB", lw=5, alpha=0.55, label="theory range" if i == 0 else None)
        if np.isfinite(cf):
            ax.scatter(cf, i, marker="D", s=48, color="#111827", zorder=4, label="theory confident rho" if i == 0 else None)
    markers = {
        "validation min COD": ("s", "#7C3AED"),
        "validation min MdAPE": ("^", "#F97316"),
        "validation PRD closest": ("o", "#16A34A"),
        "validation max R2": ("X", "#111827"),
    }
    for criterion, d in ops_df.groupby("criterion"):
        mk, col = markers.get(criterion, ("o", "#6B7280"))
        x = []
        yy = []
        for r in d.itertuples(index=False):
            idx = base.index[
                (base["data_source"].eq(r.data_source))
                & (base["assessment_year"].eq(r.assessment_year))
                & (base["config_key"].eq(r.config_key))
            ]
            if len(idx):
                x.append(float(r.selected_rho))
                yy.append(int(idx[0]))
        ax.scatter(x, yy, marker=mk, s=44, color=col, edgecolor="white", linewidth=0.5, label=criterion, zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("rho")
    ax.set_title("Theory rho ranges vs validation-selected empirical operating points", fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    return _plot_save(fig, path)


def plot_metric_response_vs_theory(comparison_df: pd.DataFrame, path: Path) -> Optional[Path]:
    if comparison_df.empty:
        return None
    d = comparison_df[comparison_df["split"].eq("assessment")].copy()
    if d.empty:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), sharex=True)
    specs = [
        ("empirical_PRD_error_reduction", "PRD error reduction\n1 - |PRD-1| / baseline"),
        ("empirical_PRB_error_reduction", "PRB error reduction\n1 - |PRB| / baseline"),
        ("empirical_COD_frac_delta", "COD fractional change\nnegative is better"),
        ("empirical_R2_delta", "R2 change vs baseline"),
    ]
    sc = None
    for ax, (col, ylabel) in zip(axes.ravel(), specs):
        finite = np.isfinite(d["covariance_reduction_theory"]) & np.isfinite(d[col])
        if not finite.any():
            ax.set_axis_off()
            continue
        sc = ax.scatter(
            d.loc[finite, "covariance_reduction_theory"],
            d.loc[finite, col],
            c=d.loc[finite, "rho"],
            cmap="viridis",
            s=22,
            alpha=0.8,
        )
        if col in ("empirical_PRD_error_reduction", "empirical_PRB_error_reduction"):
            ax.axline((0, 0), slope=1, color="#6B7280", lw=1, ls="--", alpha=0.65)
        ax.axhline(0, color="#111827", lw=0.8, alpha=0.45)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("theory-implied log-covariance reduction")
    axes[-1, 1].set_xlabel("theory-implied log-covariance reduction")
    if sc is not None:
        cb = fig.colorbar(sc, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
        cb.set_label("rho")
    fig.suptitle("Do empirical metric changes track theory-implied covariance shrinkage?", fontweight="bold")
    return _plot_save(fig, path)


def plot_direct_theory_checks(comparison_df: pd.DataFrame, path: Path) -> Optional[Path]:
    if comparison_df.empty:
        return None
    d = comparison_df[comparison_df["split"].eq("assessment")].copy()
    if d.empty:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 9.0))
    specs = [
        ("q_theory_remaining_covariance", "q_empirical_signed", "theory q", "empirical q = C/C0"),
        ("C_log_resid_logprice_theory", "C_log_resid_logprice", "theory C(f)", "empirical C(f)"),
        ("Slope_log_resid_logprice_theory", "Slope_log_resid_logprice", "theory log-residual slope", "empirical log-residual slope"),
        ("delta_MSE_log_theory", "empirical_delta_MSE_log", "theory delta MSE_log", "empirical delta MSE_log"),
    ]
    plotted = False
    for ax, (xcol, ycol, xlabel, ylabel) in zip(axes.ravel(), specs):
        if xcol not in d.columns or ycol not in d.columns:
            ax.set_axis_off()
            continue
        x = pd.to_numeric(d[xcol], errors="coerce")
        y = pd.to_numeric(d[ycol], errors="coerce")
        finite = np.isfinite(x) & np.isfinite(y)
        if not finite.any():
            ax.set_axis_off()
            continue
        plotted = True
        sc = ax.scatter(x[finite], y[finite], c=pd.to_numeric(d.loc[finite, "rho"], errors="coerce"), cmap="viridis", s=22, alpha=0.85)
        lo = float(np.nanmin([x[finite].min(), y[finite].min()]))
        hi = float(np.nanmax([x[finite].max(), y[finite].max()]))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            ax.plot([lo, hi], [lo, hi], color="#111827", ls="--", lw=1.0, alpha=0.75)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    if not plotted:
        plt.close(fig)
        return None
    cb = fig.colorbar(sc, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cb.set_label("rho")
    fig.suptitle("Direct mechanism checks: projection theory vs retrained LGBM", fontweight="bold")
    return _plot_save(fig, path)


def plot_accuracy_budget_response(comparison_df: pd.DataFrame, path: Path) -> Optional[Path]:
    if comparison_df.empty:
        return None
    d = comparison_df[comparison_df["split"].eq("assessment")].copy()
    if d.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    specs = [
        ("empirical_RMSE_frac_delta", "empirical RMSE fractional change"),
        ("empirical_MdAPE_frac_delta", "empirical MdAPE fractional change"),
    ]
    sc = None
    for ax, (col, ylabel) in zip(axes, specs):
        finite = np.isfinite(d["delta_mse_log_frac_theory"]) & np.isfinite(d[col])
        if finite.any():
            sc = ax.scatter(
                d.loc[finite, "delta_mse_log_frac_theory"],
                d.loc[finite, col],
                c=d.loc[finite, "rho"],
                cmap="viridis",
                s=24,
                alpha=0.85,
            )
        ax.axhline(0, color="#111827", lw=0.8, alpha=0.45)
        ax.axvline(0.01, color="#DC2626", lw=1.1, ls="--", alpha=0.8, label="1% theory log-MSE budget")
        ax.set_xlabel("theory-predicted log-MSE increase / baseline log-MSE")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="best")
    if sc is not None:
        cb = fig.colorbar(sc, ax=axes, fraction=0.035, pad=0.02)
        cb.set_label("rho")
    fig.suptitle("Theory accuracy budget vs empirical price-scale accuracy movement", fontweight="bold")
    return _plot_save(fig, path)


def plot_prd_rho_overlay(summary_df: pd.DataFrame, empirical_df: pd.DataFrame, path: Path) -> Optional[Path]:
    if summary_df.empty or empirical_df.empty:
        return None
    assess = empirical_df[
        empirical_df["split"].eq("assessment")
        & empirical_df["model_family"].astype(str).eq("LGBCovPenalty[diff]")
    ].copy()
    if assess.empty:
        return None
    groups = sorted(assess.groupby(["data_source", "assessment_year", "config_key"]).groups.keys(), key=lambda x: (x[1], x[2], x[0]))
    n = len(groups)
    ncols = 3
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.5, max(3.0, 2.6 * nrows)), squeeze=False, sharex=True)
    theory = summary_df[summary_df["split"].isin(["assessment", "assess"])].copy()
    for ax, group in zip(axes.ravel(), groups):
        src, year, cfg = group
        d = assess[
            assess["data_source"].eq(src)
            & assess["assessment_year"].eq(year)
            & assess["config_key"].eq(cfg)
        ].sort_values("rho")
        ax.plot(d["rho"], d["PRD"], "-o", ms=2.4, lw=1.1, color="#16A34A")
        ax.axhline(1.0, color="#2F9E44", lw=1.0, ls=":", alpha=0.9)
        tr = theory[
            theory["data_source"].eq(src)
            & theory["assessment_year"].eq(year)
            & theory["config_key"].eq(cfg)
        ]
        if not tr.empty:
            r = tr.iloc[0]
            lo, hi = float(r["theory_range_low"]), float(r["theory_range_high"])
            if np.isfinite(lo) and np.isfinite(hi):
                ax.axvspan(lo, hi, color="#2563EB", alpha=0.12)
            for col, color in [("rho_shrink_50pct", "#111827"), ("rho_prd_guidance", "#DC2626")]:
                v = float(r.get(col, np.nan))
                if np.isfinite(v):
                    ax.axvline(v, color=color, lw=1.0, ls="--", alpha=0.85)
        ax.set_title(f"{year} {cfg}\n{src}", fontsize=8.5)
        ax.grid(alpha=0.22)
    for ax in axes.ravel()[len(groups):]:
        ax.set_axis_off()
    for ax in axes[-1, :]:
        ax.set_xlabel("rho")
    for ax in axes[:, 0]:
        ax.set_ylabel("assessment PRD")
    fig.suptitle("Assessment PRD vs rho with theory range overlays", fontweight="bold")
    return _plot_save(fig, path)


def _metric_rho_overlay_specs() -> List[Dict[str, Any]]:
    """
    Empirical-vs-theory rho-evolution plots.

    same_y_scale=True is used only when the empirical and theory quantities are
    comparable unitless quantities. Otherwise the empirical metric keeps the left
    axis and the theory quantity keeps the right axis with its own natural scale.
    """
    return [
        {
            "empirical_col": "empirical_PRD_error_reduction",
            "empirical_label": "Empirical PRD error reduction",
            "theory_col": "covariance_reduction_theory",
            "theory_label": "Theory log-covariance reduction",
            "filename": "prd_error_reduction",
            "same_y_scale": True,
            "interpretation": "Higher empirical PRD error reduction should generally track higher theory covariance shrinkage.",
        },
        {
            "empirical_col": "empirical_PRB_error_reduction",
            "empirical_label": "Empirical PRB error reduction",
            "theory_col": "covariance_reduction_theory",
            "theory_label": "Theory log-covariance reduction",
            "filename": "prb_error_reduction",
            "same_y_scale": True,
            "interpretation": "Higher empirical PRB error reduction should generally track higher theory covariance shrinkage.",
        },
        {
            "empirical_col": "empirical_VEI_error_reduction",
            "empirical_label": "Empirical VEI error reduction",
            "theory_col": "covariance_reduction_theory",
            "theory_label": "Theory log-covariance reduction",
            "filename": "vei_error_reduction",
            "same_y_scale": True,
            "interpretation": "Higher empirical VEI error reduction should generally track higher theory covariance shrinkage.",
        },
        {
            "empirical_col": "empirical_R2_delta",
            "empirical_label": "Empirical R2 change vs LGBM",
            "theory_col": "delta_mse_log_frac_theory",
            "theory_label": "Theory log-MSE cost / baseline",
            "filename": "r2_delta",
            "same_y_scale": False,
            "interpretation": "Empirical R2 should stay near zero or positive while theory log-MSE cost remains small.",
        },
        {
            "empirical_col": "empirical_MSE_log_frac_delta",
            "empirical_label": "Empirical log-MSE fractional change",
            "theory_col": "delta_mse_log_frac_theory",
            "theory_label": "Theory log-MSE cost / baseline",
            "filename": "mse_log_frac_delta",
            "same_y_scale": True,
            "interpretation": "Negative empirical log-MSE change is good; this directly compares the log-scale empirical MSE movement to the theory log-MSE cost.",
        },
        {
            "empirical_col": "empirical_RMSE_frac_delta",
            "empirical_label": "Empirical RMSE fractional change",
            "theory_col": "delta_mse_log_frac_theory",
            "theory_label": "Theory log-MSE cost / baseline",
            "filename": "rmse_frac_delta",
            "same_y_scale": False,
            "interpretation": "Negative empirical RMSE change is good; compare where it starts worsening against theory MSE cost.",
        },
        {
            "empirical_col": "empirical_MdAPE_frac_delta",
            "empirical_label": "Empirical MdAPE fractional change",
            "theory_col": "delta_mse_log_frac_theory",
            "theory_label": "Theory log-MSE cost / baseline",
            "filename": "mdape_frac_delta",
            "same_y_scale": False,
            "interpretation": "Negative empirical MdAPE change is good; compare where it starts worsening against theory MSE cost.",
        },
        {
            "empirical_col": "empirical_COD_frac_delta",
            "empirical_label": "Empirical COD fractional change",
            "theory_col": "covariance_reduction_theory",
            "theory_label": "Theory log-covariance reduction",
            "filename": "cod_frac_delta",
            "same_y_scale": False,
            "interpretation": "Theory does not directly predict COD; this plot shows COD as an empirical side-effect of increasing covariance shrinkage.",
        },
    ]


def _format_rho_tick(value: float) -> str:
    if not np.isfinite(value):
        return ""
    value = float(value)
    if abs(value) >= 10.0:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    if abs(value) >= 1.0:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.3g}"


def _set_actual_rho_axis(ax: plt.Axes, x: np.ndarray, *, max_ticks: int = 22) -> None:
    """Use a linear rho axis with equidistant, human-readable ticks.

    The sweep rho grid is geometric, so labelling every sample point crowds the
    axis. Instead, place a small number of evenly spaced 'nice' ticks (like the
    normalized-evolution plots) spanning the observed rho range.
    """
    from matplotlib.ticker import MaxNLocator

    finite_x = np.asarray(x, dtype=float)
    finite_x = finite_x[np.isfinite(finite_x)]
    if finite_x.size == 0:
        return
    x_min, x_max = float(finite_x.min()), float(finite_x.max())
    pad = max(0.02 * (x_max - x_min), 0.05)
    ax.set_xscale("linear")
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, steps=[1, 2, 2.5, 5, 10]))
    ax.tick_params(axis="x", labelrotation=0, labelsize=7.5)


def _theory_inflection_band(
    x: np.ndarray,
    y: np.ndarray,
) -> Dict[str, float]:
    """Return the local theory inflection band in rho-space.

    The reference rho is the raw-rho convexity change of the theory curve.
    Around that point, flag a wider transition region rather than only the
    adjacent grid interval. By default, use a symmetric multiplicative window
    [0.5 * rho_star, 2 * rho_star], clipped to the plotted rho range.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 5:
        return {"low": np.nan, "high": np.nan, "rho_star": np.nan, "curvature_star": np.nan}

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # Drop duplicated rho values; numerical derivatives require a strictly
    # increasing x-grid.
    keep = np.concatenate([[True], np.diff(x) > 0.0])
    x = x[keep]
    y = y[keep]
    if x.size < 5:
        return {"low": np.nan, "high": np.nan, "rho_star": np.nan, "curvature_star": np.nan}

    dy = np.gradient(y, x)
    d2y = np.gradient(dy, x)
    finite = np.isfinite(d2y)
    if finite.sum() < 2:
        return {"low": np.nan, "high": np.nan, "rho_star": np.nan, "curvature_star": np.nan}

    # Find adjacent points where curvature changes sign. Treat exact zeros as
    # inflection candidates too, but avoid producing a band when curvature only
    # approaches zero without crossing it.
    candidates: List[Tuple[float, float]] = []
    for i in range(x.size - 1):
        if not (finite[i] and finite[i + 1]):
            continue
        c0 = float(d2y[i])
        c1 = float(d2y[i + 1])
        if c0 == 0.0:
            rho_star = float(x[i])
        elif c1 == 0.0:
            rho_star = float(x[i + 1])
        elif c0 * c1 < 0.0:
            # Linear interpolation of d2y(rho)=0 between x[i] and x[i+1].
            rho_star = float(x[i] - c0 * (x[i + 1] - x[i]) / (c1 - c0))
        else:
            continue
        candidates.append((rho_star, min(abs(c0), abs(c1))))

    if not candidates:
        return {"low": np.nan, "high": np.nan, "rho_star": np.nan, "curvature_star": np.nan}

    # Prefer the clearest local sign change, measured by the smallest nearby
    # absolute curvature on the plotted rho grid.
    rho_star, curv = min(candidates, key=lambda t: t[1])
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    low = max(x_min, 0.5 * rho_star)
    high = min(x_max, 2.0 * rho_star)
    return {
        "low": low,
        "high": high,
        "rho_star": rho_star,
        "curvature_star": curv,
    }



def _theory_covariance_linear_band(
    x: np.ndarray,
    y: np.ndarray,
) -> Dict[str, float]:
    """Return the covariance curve's log-rho inflection band.

    In raw rho units, covariance reduction is concave from the start. The
    meaningful "knee" for the log-like covariance curve is the inflection point
    after reparameterizing by z = log(rho), i.e. the point where multiplicative
    increases in rho stop having increasing returns and start having diminishing
    returns. Around that point, flag the transition region [rho_star / 3,
    3 * rho_star] (~25-75% covariance reduction), clipped to the plotted rho range.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0.0)
    x = x[mask]
    y = y[mask]
    if x.size < 5:
        return {"low": np.nan, "high": np.nan, "rho_star": np.nan, "curvature_star": np.nan}

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    keep = np.concatenate([[True], np.diff(x) > 0.0])
    x = x[keep]
    y = y[keep]
    if x.size < 5:
        return {"low": np.nan, "high": np.nan, "rho_star": np.nan, "curvature_star": np.nan}

    z = np.log(x)
    dy_dz = np.gradient(y, z)
    d2y_dz2 = np.gradient(dy_dz, z)
    finite = np.isfinite(d2y_dz2)
    if finite.sum() < 2:
        return {"low": np.nan, "high": np.nan, "rho_star": np.nan, "curvature_star": np.nan}

    candidates: List[Tuple[float, float]] = []
    for i in range(x.size - 1):
        if not (finite[i] and finite[i + 1]):
            continue
        c0 = float(d2y_dz2[i])
        c1 = float(d2y_dz2[i + 1])
        if c0 == 0.0:
            rho_star = float(x[i])
        elif c1 == 0.0:
            rho_star = float(x[i + 1])
        elif c0 * c1 < 0.0:
            # Linear interpolation in z = log(rho), then map back to rho.
            z_star = float(z[i] - c0 * (z[i + 1] - z[i]) / (c1 - c0))
            rho_star = float(np.exp(z_star))
        else:
            continue
        candidates.append((rho_star, min(abs(c0), abs(c1))))

    if candidates:
        rho_star, curv = min(candidates, key=lambda t: t[1])
    else:
        # Fallback: if the plotted grid misses the sign crossing, use the grid point
        # whose log-rho curvature is closest to zero.
        idx = int(np.nanargmin(np.abs(d2y_dz2)))
        rho_star = float(x[idx])
        curv = float(abs(d2y_dz2[idx]))

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    low = max(x_min, rho_star / 3.0)
    high = min(x_max, 3.0 * rho_star)
    return {
        "low": low,
        "high": high,
        "rho_star": rho_star,
        "curvature_star": curv,
    }


def plot_rho_evolution_theory_empirical_overlays(
    comparison_df: pd.DataFrame,
    out_dir: Path,
    *,
    splits: Sequence[str] = ("assessment", "test"),
) -> List[Path]:
    """
    For every metric, plot empirical evolution vs rho and theory evolution vs rho
    in the same panel grid.

    Output:
      out_dir / rho_evolution_<split>_<metric>.png

    Each small panel is one (data_source, assessment_year, config_key).
    Left y-axis: empirical metric change from the rho sweep.
    Right y-axis: theory-implied curve at the same rho.
    """
    paths: List[Path] = []
    if comparison_df.empty:
        return paths

    out_dir.mkdir(parents=True, exist_ok=True)
    df = comparison_df.copy()
    df["split"] = df["split"].replace({"assess": "assessment"})
    df["rho"] = pd.to_numeric(df["rho"], errors="coerce")
    df = df.loc[np.isfinite(df["rho"]), :].copy()
    if df.empty:
        return paths

    group_cols = ["data_source", "assessment_year", "config_key"]

    for split in splits:
        split_df = df.loc[df["split"].eq(split), :].copy()
        if split_df.empty:
            continue

        groups = sorted(
            split_df.groupby(group_cols).groups.keys(),
            key=lambda x: (int(x[1]), str(x[2]), str(x[0])),
        )
        if not groups:
            continue

        ncols = 3
        nrows = int(math.ceil(len(groups) / ncols))

        for spec in _metric_rho_overlay_specs():
            empirical_col = spec["empirical_col"]
            theory_col = spec["theory_col"]
            if empirical_col not in split_df.columns or theory_col not in split_df.columns:
                continue

            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(14.5, max(3.2, 3.0 * nrows)),
                squeeze=False,
                sharex=True,
            )

            legend_handles: List[Any] = []
            legend_labels: List[str] = []

            def _add_legend(handle: Any, label: str) -> None:
                if handle is not None and label not in legend_labels:
                    legend_handles.append(handle)
                    legend_labels.append(label)

            for ax, group in zip(axes.ravel(), groups):
                src, year, cfg = group
                d = split_df[
                    split_df["data_source"].eq(src)
                    & split_df["assessment_year"].eq(year)
                    & split_df["config_key"].eq(cfg)
                ].copy()
                d = d.sort_values("rho")

                x = pd.to_numeric(d["rho"], errors="coerce").to_numpy(dtype=float)
                y_emp = pd.to_numeric(d[empirical_col], errors="coerce").to_numpy(dtype=float)
                y_theory = pd.to_numeric(d[theory_col], errors="coerce").to_numpy(dtype=float)

                mask_emp = np.isfinite(x) & np.isfinite(y_emp)
                mask_theory = np.isfinite(x) & np.isfinite(y_theory)

                # Key theory rho (confident rho). The per-run theory_range_low/high
                # interval is degenerate (low > high) on this data, so it is not drawn.
                conf = float(pd.to_numeric(d.get("theory_confident_rho"), errors="coerce").dropna().iloc[0]) if "theory_confident_rho" in d and d["theory_confident_rho"].notna().any() else np.nan

                if theory_col == "covariance_reduction_theory":
                    shape_info = _theory_covariance_linear_band(x, y_theory)
                    shape_band_label = "theory covariance log-rho inflection band"
                    shape_rho_label = "theory covariance log-rho inflection rho"
                else:
                    shape_info = _theory_inflection_band(x, y_theory)
                    shape_band_label = "theory inflection band"
                    shape_rho_label = "theory inflection rho"

                if np.isfinite(shape_info["low"]) and np.isfinite(shape_info["high"]):
                    h = ax.axvspan(
                        shape_info["low"],
                        shape_info["high"],
                        color="#FDBA74",
                        alpha=0.28,
                        label=shape_band_label,
                        zorder=1,
                    )
                    _add_legend(h, shape_band_label)
                if np.isfinite(shape_info["rho_star"]):
                    h = ax.axvline(
                        shape_info["rho_star"],
                        color="#7C2D12",
                        linestyle=":",
                        linewidth=1.3,
                        alpha=0.95,
                        label=shape_rho_label,
                    )
                    _add_legend(h, shape_rho_label)

                if np.isfinite(conf):
                    h = ax.axvline(
                        conf,
                        color="#111827",
                        linestyle="--",
                        linewidth=1.1,
                        alpha=0.9,
                        label="theory confident rho",
                    )
                    _add_legend(h, "theory confident rho")

                # Empirical metric on left axis.
                line_emp = None
                if mask_emp.any():
                    (line_emp,) = ax.plot(
                        x[mask_emp],
                        y_emp[mask_emp],
                        "-o",
                        ms=3.0,
                        lw=1.5,
                        color="#16A34A",
                        label=spec["empirical_label"],
                    )
                    ax.axhline(0.0, color="#6B7280", lw=0.8, alpha=0.8)
                    _add_legend(line_emp, spec["empirical_label"])

                # Theory metric on right axis.
                ax2 = ax.twinx()
                line_theory = None
                if mask_theory.any():
                    (line_theory,) = ax2.plot(
                        x[mask_theory],
                        y_theory[mask_theory],
                        "--",
                        lw=1.5,
                        color="#DC2626",
                        label=spec["theory_label"],
                    )
                    _add_legend(line_theory, spec["theory_label"])

                if bool(spec.get("same_y_scale", False)):
                    y_scale_vals = np.concatenate([y_emp[mask_emp], y_theory[mask_theory]])
                    y_scale_vals = y_scale_vals[np.isfinite(y_scale_vals)]
                    if y_scale_vals.size:
                        y_min = min(float(np.nanmin(y_scale_vals)), 0.0)
                        y_max = max(float(np.nanmax(y_scale_vals)), 0.0)
                        pad = 0.05 * (y_max - y_min) if y_max > y_min else 0.05 * max(abs(y_max), 1.0)
                        ax.set_ylim(y_min - pad, y_max + pad)
                        ax2.set_ylim(y_min - pad, y_max + pad)

                ax.set_title(f"{int(year)} {cfg}\n{src}", fontsize=8.5)
                ax.grid(alpha=0.22)
                _set_actual_rho_axis(ax, x)
                ax.set_xlabel("rho")
                ax.set_ylabel(spec["empirical_label"], color="#166534", fontsize=8)
                ax2.set_ylabel(spec["theory_label"], color="#991B1B", fontsize=8)
                ax.tick_params(axis="y", labelcolor="#166534")
                ax2.tick_params(axis="y", labelcolor="#991B1B")

                ax.text(
                    0.02,
                    0.98,
                    "green = empirical\nred dashed = theory\norange = theory knee/inflection",
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=6.8,
                    bbox=dict(facecolor="white", alpha=0.72, edgecolor="none"),
                )

            for ax in axes.ravel()[len(groups):]:
                ax.set_axis_off()

            if legend_handles:
                fig.legend(legend_handles, legend_labels, loc="lower center", ncol=3, frameon=False)

            fig.suptitle(
                f"{split.capitalize()}: empirical rho evolution vs theory-implied curve\n"
                f"{spec['interpretation']}",
                fontsize=13,
                fontweight="bold",
            )
            fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.93))

            out_path = out_dir / f"rho_evolution_{split}_{spec['filename']}.png"
            paths.append(_plot_save(fig, out_path))

    return paths

def make_theory_empirical_plots(
    summary_df: pd.DataFrame,
    empirical_df: pd.DataFrame,
    out_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Path]]:
    if empirical_df.empty or summary_df.empty:
        return pd.DataFrame(), pd.DataFrame(), []
    with _timed_step("empirical: build comparison table"):
        comparison_df = build_theory_empirical_comparison(summary_df, empirical_df)
    with _timed_step("empirical: select operating points"):
        ops_df = empirical_operating_points(empirical_df, summary_df)
    paths: List[Path] = []
    comp_path = out_dir / "theory_empirical_comparison.csv"
    ops_path = out_dir / "theory_empirical_operating_points.csv"
    with _timed_step("output: write empirical comparison csvs"):
        comparison_df.to_csv(comp_path, index=False)
        ops_df.to_csv(ops_path, index=False)
    paths.extend([comp_path, ops_path])
    plot_specs = [
        (
            "plot: theory ranges vs empirical operating points",
            lambda: plot_theory_empirical_ranges(ops_df, out_dir / "plots" / "theory_ranges_vs_empirical_operating_points.png"),
        ),
        (
            "plot: empirical metric response vs theory shrinkage",
            lambda: plot_metric_response_vs_theory(comparison_df, out_dir / "plots" / "empirical_metric_response_vs_theory_shrinkage.png"),
        ),
        (
            "plot: direct projection-theory mechanism checks",
            lambda: plot_direct_theory_checks(comparison_df, out_dir / "plots" / "direct_projection_theory_checks.png"),
        ),
        (
            "plot: accuracy response vs theory budget",
            lambda: plot_accuracy_budget_response(comparison_df, out_dir / "plots" / "accuracy_response_vs_theory_budget.png"),
        ),
        (
            "plot: assessment PRD vs rho theory overlay",
            lambda: plot_prd_rho_overlay(summary_df, empirical_df, out_dir / "plots" / "assessment_prd_vs_rho_theory_overlay.png"),
        ),
    ]
    for step, plotter in plot_specs:
        with _timed_step(step):
            maybe = plotter()
        if maybe is not None:
            paths.append(maybe)

    with _timed_step("plot: empirical metric evolution vs theory rho evolution overlays"):
        overlay_paths = plot_rho_evolution_theory_empirical_overlays(
            comparison_df,
            out_dir / "plots" / "rho_evolution_theory_empirical",
        )
    paths.extend(overlay_paths)

    return comparison_df, ops_df, paths


def _fmt(x: Any, digits: int = 4) -> str:
    try:
        v = float(x)
    except Exception:
        return str(x)
    if not np.isfinite(v):
        return "NA"
    return f"{v:.{digits}f}"


def render_markdown_report(
    summary_df: pd.DataFrame,
    shrink_df: pd.DataFrame,
    prd_df: pd.DataFrame,
    budget_df: pd.DataFrame,
    aggregate_df: pd.DataFrame,
    args: argparse.Namespace,
    *,
    comparison_df: Optional[pd.DataFrame] = None,
    ops_df: Optional[pd.DataFrame] = None,
    plot_paths: Optional[Sequence[Path]] = None,
) -> str:
    lines: List[str] = []
    lines.append("# Theory-informed rho-range report")
    lines.append("")
    lines.append("This report fits the unpenalized LGBM baseline and computes rho values implied by the rank-one covariance-shrinkage theory for `LGBCovPenalty[diff]`.")
    lines.append("")
    lines.append("## Main aggregate recommendation")
    lines.append("")
    if aggregate_df.empty:
        lines.append("No valid runs were produced.")
    else:
        agg = aggregate_df.loc[aggregate_df["split_group"] == "all"].iloc[0] if (aggregate_df["split_group"] == "all").any() else aggregate_df.iloc[0]
        lines.append(f"- Robust theory range: **[{_fmt(agg['robust_theory_range_low'], 3)}, {_fmt(agg['robust_theory_range_high'], 3)}]**")
        lines.append(f"- Median confident rho: **{_fmt(agg['median_confident_rho'], 3)}**")
        lines.append(f"- Median rho for 25% covariance reduction: {_fmt(agg['median_rho_shrink_25pct'], 3)}")
        lines.append(f"- Median rho for 50% covariance reduction: {_fmt(agg['median_rho_shrink_50pct'], 3)}")
        lines.append(f"- Median rho for 67% covariance reduction: {_fmt(agg['median_rho_shrink_67pct'], 3)}")
        if "median_orange_overlap_low" in agg.index:
            lines.append(
                f"- Orange-band overlap range: **[{_fmt(agg['median_orange_overlap_low'], 3)}, "
                f"{_fmt(agg['median_orange_overlap_high'], 3)}]**, recommended rho "
                f"**{_fmt(agg['median_orange_overlap_ref_rho'], 3)}**"
            )
            lines.append(
                f"  - Log-MSE orange band: [{_fmt(agg['median_orange_mse_band_low'], 3)}, "
                f"{_fmt(agg['median_orange_mse_band_high'], 3)}], reference rho "
                f"{_fmt(agg['median_orange_mse_ref_rho'], 3)}"
            )
            lines.append(
                f"  - Covariance orange band: [{_fmt(agg['median_orange_cov_band_low'], 3)}, "
                f"{_fmt(agg['median_orange_cov_band_high'], 3)}], reference rho "
                f"{_fmt(agg['median_orange_cov_ref_rho'], 3)}"
            )
        if "empirical_overlap_rate" in agg.index:
            lines.append(f"- Empirical range overlap rate: {_fmt(agg['empirical_overlap_rate'], 3)}")
            lines.append(f"- Empirical low/high implied covariance reduction: {_fmt(agg['median_empirical_low_cov_reduction_implied'], 3)} to {_fmt(agg['median_empirical_high_cov_reduction_implied'], 3)}")
    lines.append("")
    lines.append("## Aggregate table")
    lines.append("")
    if not aggregate_df.empty:
        lines.append(aggregate_df.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## Per-run summary")
    lines.append("")
    display_cols = [
        "data_source", "assessment_year", "config_key", "split", "n", "prd", "A_var_f0_log", "B_mse_log",
        "C0_cov_log_residual_logprice", "bayes_optimality_diagnostic_C0_over_minus_B",
        "first_order_bridge_rel_error_at_baseline",
        "rho_shrink_25pct", "rho_shrink_50pct", "rho_shrink_67pct",
        "rho_prd_guidance", "rho_budget_1pct_mse", "theory_range_low", "theory_confident_rho", "theory_range_high",
    ]
    if not summary_df.empty:
        cols = [c for c in display_cols if c in summary_df.columns]
        lines.append(summary_df.loc[:, cols].to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## Interpretation notes")
    lines.append("")
    lines.append("- `A_var_f0_log = Var(f0(X))` sets the rho scale. Larger signal variance means smaller rho is needed for the same covariance shrinkage.")
    lines.append("- `B_mse_log` is the baseline log-MSE. `C0/(-B)` close to 1 supports the Bayes-optimal residual story: the baseline residual covariance is mostly irreducible regression-to-the-mean covariance.")
    lines.append("- `rho_shrink_50pct` is the rho that should approximately halve the baseline log-residual covariance under the quadratic theory.")
    lines.append("- `rho_prd_guidance` is the rho implied by moving PRD to the nearest IAAO-style boundary, using the exact baseline ratio-price covariance identity and assuming proportional covariance shrinkage.")
    lines.append("- `rho_budget_1pct_mse` is an upper-bound rho under an approximate 1% baseline log-MSE budget.")
    lines.append("- Orange-band overlap is the overlap between the log-MSE transition band and the covariance log-rho transition band; the reported reference rho is the geometric midpoint of that overlap.")
    lines.append("")
    if args.empirical_rho_range:
        lines.append(f"Empirical range supplied for comparison: **[{args.empirical_rho_range}]**.")
    lines.append("")
    lines.append("## Theory vs empirical sweep checks")
    lines.append("")
    if comparison_df is None or comparison_df.empty:
        lines.append(f"No empirical sweep rows were matched from `{args.rho_sweep_root}`.")
    else:
        assess = comparison_df.loc[comparison_df["split"].eq("assessment")].copy()
        inside_rate = float(assess["inside_theory_range"].mean()) if "inside_theory_range" in assess and not assess.empty else float("nan")
        lines.append(f"- Matched empirical covariance-penalty rows: **{comparison_df.shape[0]}**.")
        lines.append(f"- Share of assessment sweep rho values inside the per-run theory range: **{_fmt(inside_rate, 3)}**.")
        for y_col, x_col, label in [
            ("q_empirical_signed", "q_theory_remaining_covariance", "empirical q vs theory q"),
            ("empirical_delta_MSE_log", "delta_MSE_log_theory", "empirical vs theory log-MSE delta"),
            ("Slope_log_resid_logprice", "Slope_log_resid_logprice_theory", "empirical vs theory log-residual slope"),
        ]:
            if not assess.empty and y_col in assess and x_col in assess and assess[[x_col, y_col]].dropna().shape[0] >= 2:
                corr = assess[[x_col, y_col]].corr().iloc[0, 1]
                lines.append(f"- Corr({label}): **{_fmt(corr, 3)}**.")
        for col, label in [
            ("empirical_PRD_error_reduction", "PRD error reduction"),
            ("empirical_PRB_error_reduction", "PRB error reduction"),
            ("empirical_VEI_error_reduction", "VEI error reduction"),
            ("empirical_COD_frac_delta", "COD fractional change"),
            ("empirical_R2_delta", "R2 change"),
        ]:
            if not assess.empty and col in assess and assess[["covariance_reduction_theory", col]].dropna().shape[0] >= 2:
                corr = assess[["covariance_reduction_theory", col]].corr().iloc[0, 1]
                lines.append(f"- Corr(theory covariance reduction, {label}): **{_fmt(corr, 3)}**.")
        if not assess.empty and "q_error_empirical_minus_theory" in assess:
            qerr = pd.to_numeric(assess["q_error_empirical_minus_theory"], errors="coerce").dropna()
            if not qerr.empty:
                lines.append(f"- Median |empirical q - theory q|: **{_fmt(float(qerr.abs().median()), 4)}**.")
        if not assess.empty and "delta_MSE_log_error" in assess:
            merr = pd.to_numeric(assess["delta_MSE_log_error"], errors="coerce").dropna()
            if not merr.empty:
                lines.append(f"- Median |empirical delta-MSE_log - theory delta-MSE_log|: **{_fmt(float(merr.abs().median()), 6)}**.")
    if ops_df is not None and not ops_df.empty:
        op_inside = float(ops_df["inside_theory_range"].mean()) if "inside_theory_range" in ops_df else float("nan")
        lines.append(f"- Validation-selected empirical operating points inside theory range: **{_fmt(op_inside, 3)}**.")
        cols = [
            "data_source", "assessment_year", "config_key", "criterion", "selected_rho",
            "theory_range_low", "theory_confident_rho", "theory_range_high", "inside_theory_range",
        ]
        cols = [c for c in cols if c in ops_df.columns]
        lines.append("")
        lines.append(ops_df.loc[:, cols].to_markdown(index=False, floatfmt=".4f"))
    if plot_paths:
        lines.append("")
        lines.append("Generated comparison artifacts:")
        for path in plot_paths:
            rel = os.path.relpath(path, args.out_dir)
            lines.append(f"- `{rel}`")
    timing_summary = _timing_summary_df()
    if not timing_summary.empty:
        lines.append("")
        lines.append("## Timing summary")
        lines.append("")
        lines.append("Slowest instrumented steps:")
        display = timing_summary.head(12).copy()
        for col in ["total_seconds", "mean_seconds", "max_seconds"]:
            display[col] = display[col].map(lambda x: round(float(x), 3))
        lines.append(display.to_markdown(index=False))
    lines.append("")
    lines.append("Adversarial caveats:")
    lines.append("- The closed-form rho curve is a local rank-one functional approximation; tree refitting, early stopping, and finite boosting capacity can move empirical optima.")
    lines.append("- The PRD identity uses exact ratio-price covariance at the baseline, but translating PRD movement into log-covariance shrinkage assumes the two covariances move proportionally after penalization.")
    lines.append("- `first_order_bridge_rel_error_at_baseline` should be inspected before trusting the first-order `exp(e) ~= 1 + e` bridge.")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compute theory-informed rho ranges for LGBCovPenalty[diff].")
    p.add_argument("--data-path", type=str, default="./data/CCAO/2025/training_data.parquet")
    p.add_argument(
        "--data-source-label",
        type=str,
        default="",
        help="Label used when --data-source-specs is empty; useful for matching empirical folders.",
    )
    p.add_argument(
        "--data-source-specs",
        type=str,
        default=_default_data_source_specs(),
        help=(
            "Semicolon-separated source:assessment_year:parquet specs. The default matches "
            "scripts/rho_sweep_experiments.sh and therefore the dashboard output folders. "
            "Pass an empty string to use --data-path with --assessment-years instead."
        ),
    )
    p.add_argument("--params-path", type=str, default="params.yaml")
    p.add_argument("--model-params-path", type=str, default="model_params.yaml")
    p.add_argument("--lgbm-hyperparameter-file", type=str, default="best_lgbm_baseline_configs.yaml")
    p.add_argument("--lgbm-config-keys", type=str, default="cv_top1_r2,test_best_r2,cv_top2_r2")
    p.add_argument("--lgbm-n-jobs", type=int, default=None)
    p.add_argument("--lgbm-n-estimators", type=int, default=None)
    p.add_argument("--assessment-years", type=str, default="2022,2023,2024,2025")
    p.add_argument("--target-column", type=str, default="meta_sale_price")
    p.add_argument("--date-column", type=str, default="meta_sale_date")
    p.add_argument("--sample-frac", type=float, default=None)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out-dir", type=str, default="output/theory_rho_range")
    p.add_argument(
        "--baseline-cache-dir",
        type=str,
        default="",
        help="Where to cache fitted baseline log predictions for reuse across runs. Defaults to <out-dir>/baseline_cache.",
    )
    p.add_argument("--rho-sweep-root", type=str, default="output/rho_sweep_500_estimators")
    p.add_argument("--shrinkage-q-values", type=str, default="0.75,0.50,0.33,0.25")
    p.add_argument("--prd-targets", type=str, default="1.03,1.02,1.01,0.99,0.98")
    p.add_argument("--accuracy-budgets", type=str, default="0.001,0.005,0.01,0.02")
    p.add_argument("--empirical-rho-range", type=str, default="2.56,3.54")
    p.add_argument(
        "--write-intermediate-results",
        dest="write_intermediate_results",
        action="store_true",
        default=True,
        help="Write per-split checkpoint CSVs and incremental journal CSVs as soon as each split finishes.",
    )
    p.add_argument(
        "--no-write-intermediate-results",
        dest="write_intermediate_results",
        action="store_false",
        help="Disable per-split checkpoint/incremental writes.",
    )
    p.add_argument(
        "--write-intermediate-plots",
        dest="write_intermediate_plots",
        action="store_true",
        default=True,
        help="Write per-split theory tradeoff plots as soon as each split finishes.",
    )
    p.add_argument(
        "--no-write-intermediate-plots",
        dest="write_intermediate_plots",
        action="store_false",
        help="Disable per-split theory tradeoff plots.",
    )
    p.add_argument(
        "--plot-format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Format for intermediate and summary plots.",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    paths = run_analysis(args)
    for key, path in paths.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
