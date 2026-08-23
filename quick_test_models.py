"""
Quick test runner.

Goal
----
Fit and evaluate selected fairness-regularized models on:
  - held-out test split (most recent pre-2024 sales; ~2023 by CCAO-style split)
  - assessment split (2024 sales)

This script intentionally mirrors the preprocessing + split logic in `main.py`,
but avoids CV and bootstrapping to stay fast and easy to read.

Models
------
1) LinearRegression baseline
2) LGBMRegressor baseline
3) LGBCovPenalty in `diff` mode
4) LGBSmoothPenalty in `diff` mode

Outputs
-------
Writes CSV tables and text reports under `--out-dir`:
  - quick_test_metrics_test.csv
  - quick_test_metrics_assess.csv
  - quick_test_metrics_validation_bootstrap_avg.csv
  - quick_test_report_test.txt
  - quick_test_report_assess.txt

Each table contains accuracy + vertical equity metrics computed with the same
metric routine used elsewhere in this repo (`_compute_extended_metrics`).

Usage
-----
From the `soft-vertical-equity-constrained-mass-appraissal/` directory:

  python quick_test_models.py --rho 1.0
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd
import yaml
import lightgbm as lgb
from sklearn.linear_model import LinearRegression

from preprocessing.recipes_pipelined import build_model_pipeline
from soft_constrained_models.linear_models import LinearProjectionCovariancePath
from soft_constrained_models.boosting_models import LGBCovPenalty, LGBSmoothPenalty
from utils.projection_theory_utils import compute_projection_theory_metrics
from utils.plotting_utils import (
    plot_ratio_vs_logprice,
    plot_residual_vs_logprice,
    plot_ratio_vs_logprediction,
    plot_residual_vs_logprediction,
)
from utils.motivation_utils import (
    IAAO_COD_RANGES,
    IAAO_LEVEL_RANGE,
    IAAO_PRB_RANGE,
    IAAO_PRD_RANGE,
    IAAO_VEI_RANGE,
    _build_time_block_bootstrap_indices,
    _compute_extended_metrics,
    prb,
    prd,
)


_PAIRWISE_DEPENDENCE_SAMPLE_N = 1024
_QUANTILE_ANALYSIS_COUNTS = (3, 4, 5, 10)
_RATIO_MODES = ("diff",)
_MAIN_SWEEP_FAMILIES = (
    "LGBCovPenalty[diff]",
    "LGBSmoothPenalty[diff]",
)
_GROUPED_SWEEP_FAMILY = "LGBSmoothPenaltyGrouped"
_COMPUTE_DEPENDENCE_METRICS = False
_RHO_PLOT_METRICS = ("R2", "MdAPE", "COD", "PRD", "PRB", "VEI")
_RHO_METRIC_COLORS = {
    "R2": "#111827",
    "MdAPE": "#F97316",
    "COD": "#7C3AED",
    "PRD": "#16A34A",
    "PRB": "#2563EB",
    "VEI": "#DC2626",
}
_RHO_HIGHER_IS_BETTER = {"R2"}
_RHO_LOWER_IS_BETTER = {"MdAPE", "COD"}
_RHO_TARGET_IS_BETTER = {"PRD": 1.0, "PRB": 0.0, "VEI": 0.0}
_RHO_DECIMALS = 3
_TRADEOFF_FAIRNESS_METRICS = ("PRD", "PRB", "VEI")
_TRADEOFF_TARGET_METRICS = ("R2", "MdAPE", "COD")
_TRADEOFF_BASELINE_FAMILIES = ("LinearRegression", "LGBMRegressor")
_TRADEOFF_FAMILY_COLORS = {
    "LinearRegression": "#4B5563",
    "LGBMRegressor": "#111827",
    "LGBCovPenalty[diff]": "#2563EB",
    "LGBSmoothPenalty[diff]": "#DC2626",
}
_DEFAULT_LGBM_HYPERPARAMETER_FILE = "best_lgbm_baseline_configs.yaml"
_DEFAULT_LGBM_CONFIG_KEY = "test_best_r2"
_META_TOWNSHIP_TRIAD_COL = "meta_township_triad"
_CHAR_CLASS_BUCKET_COL = "char_class_bucket"
_TOWNSHIP_TRIAD_MAP = {
    "10": "north_northwest_suburbs",
    "16": "north_northwest_suburbs",
    "17": "north_northwest_suburbs",
    "18": "north_northwest_suburbs",
    "20": "north_northwest_suburbs",
    "22": "north_northwest_suburbs",
    "23": "north_northwest_suburbs",
    "24": "north_northwest_suburbs",
    "25": "north_northwest_suburbs",
    "26": "north_northwest_suburbs",
    "29": "north_northwest_suburbs",
    "35": "north_northwest_suburbs",
    "38": "north_northwest_suburbs",
    "11": "south_southwest_suburbs",
    "12": "south_southwest_suburbs",
    "13": "south_southwest_suburbs",
    "14": "south_southwest_suburbs",
    "15": "south_southwest_suburbs",
    "19": "south_southwest_suburbs",
    "21": "south_southwest_suburbs",
    "27": "south_southwest_suburbs",
    "28": "south_southwest_suburbs",
    "30": "south_southwest_suburbs",
    "31": "south_southwest_suburbs",
    "32": "south_southwest_suburbs",
    "33": "south_southwest_suburbs",
    "34": "south_southwest_suburbs",
    "36": "south_southwest_suburbs",
    "37": "south_southwest_suburbs",
    "39": "south_southwest_suburbs",
    "70": "city_of_chicago",
    "71": "city_of_chicago",
    "72": "city_of_chicago",
    "73": "city_of_chicago",
    "74": "city_of_chicago",
    "75": "city_of_chicago",
    "76": "city_of_chicago",
    "77": "city_of_chicago",
}
_CHAR_CLASS_BUCKET_MAP = {
    "202": "one_story_residences",
    "203": "one_story_residences",
    "204": "one_story_residences",
    "205": "two_story_older_than_62_years",
    "206": "two_story_older_than_62_years",
    "207": "two_story_newer_than_62_years",
    "208": "two_story_newer_than_62_years",
    "209": "two_story_newer_than_62_years",
    "278": "two_story_newer_than_62_years",
    "210": "townhomes_rowhouses",
    "295": "townhomes_rowhouses",
    "211": "multi_family_mixed_use_2_to_6_units",
    "212": "multi_family_mixed_use_2_to_6_units",
    "234": "split_level_residences",
    "299": "condominiums",
    "218": "char_class_anomaly",
    "219": "char_class_anomaly",
    "297": "char_class_anomaly",
    "NA": "NA",
}

_LOG_T0 = time.perf_counter()


def _log(message: str, **fields: Any) -> None:
    dt = time.perf_counter() - _LOG_T0
    suffix = ""
    if fields:
        suffix = " | " + " | ".join(f"{k}={v}" for k, v in fields.items())
    print(f"[quick_test_models +{dt:7.1f}s] {message}{suffix}", flush=True)


def _build_lgbm_params_from_files(model_params: dict, ccao_params: dict, seed: int) -> dict:
    """
    Match `main.py`: use `model_params.yaml` as primary defaults and fall back to
    `params.yaml` for any missing keys.
    """
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
        raise ValueError(f"LGBM config key '{config_key}' in {path} does not contain a non-empty lgbm_params block.")
    return dict(params)


def _series_to_string_with_na(series: pd.Series) -> pd.Series:
    return series.astype("object").where(series.notna(), "NA").astype(str)


def _map_meta_township_triad(series: pd.Series) -> pd.Series:
    return _series_to_string_with_na(series).map(_TOWNSHIP_TRIAD_MAP).fillna("other_or_unmapped_township")


def _map_char_class_bucket(series: pd.Series) -> pd.Series:
    return _series_to_string_with_na(series).map(_CHAR_CLASS_BUCKET_MAP).fillna("other_or_unmapped_char_class")


def _add_quick_test_grouped_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "meta_township_code" in df.columns:
        df[_META_TOWNSHIP_TRIAD_COL] = _map_meta_township_triad(df["meta_township_code"])
    if "char_class" in df.columns:
        df[_CHAR_CLASS_BUCKET_COL] = _map_char_class_bucket(df["char_class"])
    return df


def _load_and_split_data(
    *,
    data_path: str,
    params: dict,
    target_column: str,
    date_column: str,
    sample_frac: float | None,
    sample_seed: int,
    assessment_year: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """
    Mirrors `main.py`:
      - load parquet
      - filter out multicard and outliers
      - keep only predictor + target + date
      - sort by date
      - split into assess (== assessment_year), and pre-assess (< assessment_year)
        then train/validate + test (last 1-split_prop fraction of pre-assess).
    """
    _log("loading parquet", data_path=data_path)
    df = pd.read_parquet(data_path, engine="fastparquet")
    _log("parquet loaded", rows=int(df.shape[0]), cols=int(df.shape[1]))
    df = df[
        (~df["ind_pin_is_multicard"].astype("bool").fillna(True))
        & (~df["sv_is_outlier"].astype("bool").fillna(True))
    ].copy()
    _log("row filters applied", rows=int(df.shape[0]))

    predictor_cols = list(params["model"]["predictor"]["all"])
    categorical_cols = list(params["model"]["predictor"]["categorical"])
    keep_cols = predictor_cols + [target_column, date_column]
    df = df.loc[:, keep_cols].copy()
    _log("projected columns", kept_cols=int(len(keep_cols)))

    if sample_frac is not None:
        if not (0.0 < float(sample_frac) <= 1.0):
            raise ValueError("sample_frac must be in (0, 1]. Use None to disable sampling.")
        if float(sample_frac) < 1.0:
            df = df.sample(frac=float(sample_frac), random_state=int(sample_seed)).copy()
            _log("sampling applied", sample_frac=float(sample_frac), rows=int(df.shape[0]))

    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(date_column).reset_index(drop=True)
    _log("date sort completed", rows=int(df.shape[0]))

    df_assess = df.loc[df[date_column].dt.year == int(assessment_year), :].copy()
    df_train_all = df.loc[df[date_column].dt.year < int(assessment_year), :].copy()
    _log(
        "assessment-year split",
        assessment_year=int(assessment_year),
        assess_year_rows=int(df_assess.shape[0]),
        pre_assess_rows=int(df_train_all.shape[0]),
    )

    train_prop = float(params["cv"]["split_prop"])
    split_idx = int(train_prop * df_train_all.shape[0])
    df_test = df_train_all.iloc[split_idx:, :].copy()
    df_train_validate = df_train_all.iloc[:split_idx, :].copy()
    _log(
        "data split completed",
        train_validate_rows=int(df_train_validate.shape[0]),
        test_rows=int(df_test.shape[0]),
        assess_rows=int(df_assess.shape[0]),
    )

    return df_train_validate, df_test, df_assess, predictor_cols, categorical_cols


def _build_rho_sweep(
    rho: float,
    rho_range_raw: str,
    rho_count: int,
    rho_scale: str,
    rho_extra_raw: str = "",
) -> List[float]:
    extra = [float(token.strip()) for token in str(rho_extra_raw).split(",") if token.strip()]

    if str(rho_range_raw).strip() == "":
        return _round_rho_values([float(rho), *extra])

    bounds = [float(token.strip()) for token in str(rho_range_raw).split(",") if token.strip()]
    if len(bounds) != 2:
        raise ValueError("rho_range must contain exactly two comma-separated values: min,max.")

    count = int(rho_count)
    if count < 1:
        raise ValueError("rho_count must be >= 1.")

    lo, hi = float(bounds[0]), float(bounds[1])
    if count == 1:
        return _round_rho_values([lo, *extra])

    scale = str(rho_scale).strip().lower()
    if scale == "linear":
        values = np.linspace(lo, hi, count, dtype=float)
    elif scale in {"log", "geom"}:
        if lo <= 0.0 or hi <= 0.0:
            raise ValueError("rho_range bounds must be > 0 for rho_scale=log/geom.")
        values = np.geomspace(lo, hi, count, dtype=float)
    else:
        raise ValueError("rho_scale must be one of: linear, log, geom.")

    # Append any explicit extra rho values (e.g. a recommended operating point) and
    # return a sorted, de-duplicated grid so downstream code can rely on monotonicity.
    return _round_rho_values(sorted(values.tolist() + extra))


def _round_rho_value(value: float) -> float:
    rounded = float(np.round(float(value), _RHO_DECIMALS))
    return 0.0 if rounded == 0.0 else rounded


def _round_rho_values(values: List[float]) -> List[float]:
    out: List[float] = []
    seen: set[float] = set()
    for value in values:
        rounded = _round_rho_value(float(value))
        if rounded not in seen:
            out.append(rounded)
            seen.add(rounded)
    return out


def _parse_float_list(values_raw: str) -> List[float]:
    values = [float(token.strip()) for token in str(values_raw).split(",") if token.strip()]
    if not values:
        raise ValueError("Expected at least one numeric value.")
    return values


def _parse_model_families(values_raw: str) -> List[str]:
    aliases = {
        "linear": "linear",
        "linearregression": "linear",
        "linear_cov": "linear_cov",
        "linearcov": "linear_cov",
        "linearcovariance": "linear_cov",
        "linearprojectioncovariancepath": "linear_cov",
        "lgbm": "lgbm",
        "lgbmregressor": "lgbm",
        "cov": "cov",
        "lgbcovpenalty": "cov",
        "lgbcovpenalty[diff]": "cov",
        "smooth": "smooth",
        "surr": "smooth",
        "surrogate": "smooth",
        "lgbsurrpenalty": "smooth",
        "lgbsmoothpenalty": "smooth",
        "lgbsmoothpenalty[diff]": "smooth",
    }
    order = ["linear", "linear_cov", "lgbm", "cov", "smooth"]
    tokens = [token.strip().lower() for token in str(values_raw).split(",") if token.strip()]
    if not tokens or tokens == ["all"]:
        return order

    out: List[str] = []
    for token in tokens:
        if token not in aliases:
            raise ValueError("--models entries must be one of: linear,linear_cov,lgbm,cov,smooth,surr,all.")
        value = aliases[token]
        if value not in out:
            out.append(value)
    return out


def _pairwise_metric_subsample(x: np.ndarray, y: np.ndarray, max_n: int) -> Tuple[np.ndarray, np.ndarray]:
    n = int(x.size)
    if n <= max_n:
        return x, y
    order = np.argsort(x, kind="mergesort")
    take = (np.arange(max_n, dtype=int) * n) // max_n
    idx = order[take]
    return x[idx], y[idx]


def _double_center(matrix: np.ndarray) -> np.ndarray:
    row_mean = matrix.mean(axis=1, keepdims=True)
    col_mean = matrix.mean(axis=0, keepdims=True)
    grand_mean = float(matrix.mean())
    return matrix - row_mean - col_mean + grand_mean


def _distance_correlation_sampled(x: np.ndarray, y: np.ndarray, max_n: int) -> Tuple[float, int]:
    if x.size < 2 or y.size < 2:
        return np.nan, int(min(x.size, y.size))
    xs, ys = _pairwise_metric_subsample(x, y, max_n=max_n)
    a = np.abs(xs[:, None] - xs[None, :])
    b = np.abs(ys[:, None] - ys[None, :])
    a = _double_center(a)
    b = _double_center(b)
    dcov2 = float(np.mean(a * b))
    dvar_x = float(np.mean(a * a))
    dvar_y = float(np.mean(b * b))
    if dvar_x <= 0.0 or dvar_y <= 0.0:
        return 0.0, int(xs.size)
    dcor2 = float(np.clip(dcov2 / np.sqrt(dvar_x * dvar_y), 0.0, 1.0))
    return float(np.sqrt(dcor2)), int(xs.size)


def _rbf_kernel_from_1d(values: np.ndarray) -> np.ndarray:
    diffs = values[:, None] - values[None, :]
    sqdist = diffs * diffs
    upper = sqdist[np.triu_indices(values.size, k=1)]
    positive = upper[upper > 0.0]
    if positive.size == 0:
        return np.ones_like(sqdist)
    sigma2 = float(np.median(positive))
    gamma = 1.0 / max(2.0 * sigma2, 1e-12)
    return np.exp(-gamma * sqdist)


def _normalized_hsic_sampled(x: np.ndarray, y: np.ndarray, max_n: int) -> Tuple[float, int]:
    if x.size < 2 or y.size < 2:
        return np.nan, int(min(x.size, y.size))
    xs, ys = _pairwise_metric_subsample(x, y, max_n=max_n)
    kx = _double_center(_rbf_kernel_from_1d(xs))
    ky = _double_center(_rbf_kernel_from_1d(ys))
    denom = float(np.sqrt(np.sum(kx * kx) * np.sum(ky * ky)))
    if denom <= 0.0:
        return 0.0, int(xs.size)
    nhsic = float(np.sum(kx * ky) / denom)
    return float(np.clip(nhsic, 0.0, 1.0)), int(xs.size)


def _chatterjee_xi(x: np.ndarray, y: np.ndarray) -> float:
    n = int(min(x.size, y.size))
    if n < 2:
        return np.nan
    if np.all(x == x[0]) or np.all(y == y[0]):
        return 0.0

    order = np.argsort(x, kind="mergesort")
    y_ord = y[order]
    y_sorted = np.sort(y)
    r = np.searchsorted(y_sorted, y_ord, side="right").astype(float)
    l = (n - np.searchsorted(y_sorted, y_ord, side="left")).astype(float)

    denom = float(2.0 * np.sum(l * (n - l)))
    if denom <= 0.0:
        return 0.0

    xi = 1.0 - (float(n) * float(np.sum(np.abs(np.diff(r))))) / denom
    return float(np.clip(xi, -1.0, 1.0))


def _compute_logprice_dependence_metrics(
    *,
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    ratio_mode: str,
    eps_y: float = 1e-12,
) -> Dict[str, Any]:
    y_true_log = np.asarray(y_true_log, dtype=float).reshape(-1)
    y_pred_log = np.asarray(y_pred_log, dtype=float).reshape(-1)

    if ratio_mode == "diff":
        r = y_pred_log - y_true_log
    elif ratio_mode == "div":
        r = y_pred_log / np.maximum(np.abs(y_true_log), eps_y)
    elif ratio_mode == "ratio":
        log_ratio = np.clip(y_pred_log - y_true_log, -50.0, 50.0)
        r = np.exp(log_ratio)
    else:
        raise ValueError("ratio_mode must be 'div', 'diff', or 'ratio'.")

    mask = np.isfinite(y_true_log) & np.isfinite(r)
    x = y_true_log[mask]
    y = r[mask]

    dcor, sample_n = _distance_correlation_sampled(x, y, max_n=_PAIRWISE_DEPENDENCE_SAMPLE_N)
    nhsic, _ = _normalized_hsic_sampled(x, y, max_n=_PAIRWISE_DEPENDENCE_SAMPLE_N)
    xi = _chatterjee_xi(x, y)

    return {
        "dCor(r,logprice)_sampled": dcor,
        "ChatterjeeXi(r,logprice)": xi,
        "nHSIC(r,logprice)_sampled": nhsic,
        "pairwise_dependence_sample_n": int(sample_n),
    }


def _in_range(value: float, bounds: Tuple[float, float]) -> bool:
    return bool(np.isfinite(value) and (float(bounds[0]) <= float(value) <= float(bounds[1])))


def _interp_prd(value: float) -> str:
    if not np.isfinite(value):
        return "—"
    if value > float(IAAO_PRD_RANGE[1]):
        return "Regressive tendency"
    if value < float(IAAO_PRD_RANGE[0]):
        return "Progressive tendency"
    return "Within guidance"


def _interp_prb(value: float) -> str:
    if not np.isfinite(value):
        return "—"
    if value < 0.0:
        return "Regressive tendency"
    if value > 0.0:
        return "Progressive tendency"
    return "No bias"


def _interp_vei(value: float) -> str:
    if not np.isfinite(value):
        return "—"
    if _in_range(value, IAAO_VEI_RANGE):
        return "Within guidance"
    if value < float(IAAO_VEI_RANGE[0]):
        return "Regressive beyond guidance"
    return "Progressive beyond guidance"


def _interp_level(value: float) -> str:
    if not np.isfinite(value):
        return "—"
    if _in_range(value, IAAO_LEVEL_RANGE):
        return "Within 0.90–1.10 level"
    if value < float(IAAO_LEVEL_RANGE[0]):
        return "Below 0.90 level"
    return "Above 1.10 level"


def _interp_cod(value: float, *, property_class: str = "Residential Improved") -> str:
    bounds = IAAO_COD_RANGES.get(str(property_class))
    if not np.isfinite(value):
        return "—"
    if bounds is None:
        return "Lower is better"
    if _in_range(value, bounds):
        return "Within Table 7 range"
    if value < float(bounds[0]):
        return "Below Table 7 range"
    return "Above Table 7 range"


def _fmt_num(value: Any, digits: int = 4, *, comma: bool = False) -> str:
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if value is None:
        return "—"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}" if comma else f"{int(value)}"
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(val):
        return "—"
    fmt = f",.{digits}f" if comma else f".{digits}f"
    return format(val, fmt)


def _fmt_range(bounds: Tuple[float, float] | None, digits: int = 2, *, pct: bool = False) -> str:
    if bounds is None:
        return "—"
    lo, hi = float(bounds[0]), float(bounds[1])
    if pct:
        return f"[{lo:.{digits}f}, {hi:.{digits}f}]%"
    return f"[{lo:.{digits}f}, {hi:.{digits}f}]"


def _render_metric_table(rows: List[Tuple[str, str, str, str]], *, title: str) -> str:
    col_names = ("Metric", "Value", "IAAO expected", "Interpretation")
    cols = list(zip(*([col_names] + rows)))
    widths = [max(len(str(v)) for v in col) for col in cols]

    def _line(ch: str = "-") -> str:
        return ch * (sum(widths) + 3 * (len(widths) - 1))

    lines = [_line("="), title, _line("=")]
    header = "  ".join(
        [
            f"{col_names[i]:<{widths[i]}}" if i == 0 else f"{col_names[i]:>{widths[i]}}"
            for i in range(len(widths))
        ]
    )
    lines.append(header)
    lines.append(_line("-"))
    for metric, value, expected, interp in rows:
        lines.append(
            "  ".join(
                [
                    f"{metric:<{widths[0]}}",
                    f"{value:>{widths[1]}}",
                    f"{expected:>{widths[2]}}",
                    f"{interp:<{widths[3]}}",
                ]
            )
        )
    lines.append(_line("="))
    return "\n".join(lines)


def _compute_quantile_diagnostic_tables(
    *,
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    quantile_counts: Tuple[int, ...] = _QUANTILE_ANALYSIS_COUNTS,
) -> Dict[int, pd.DataFrame]:
    y_true = np.exp(np.asarray(y_true_log, dtype=float).reshape(-1))
    y_pred = np.exp(np.asarray(y_pred_log, dtype=float).reshape(-1))

    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0.0) & (y_pred > 0.0)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    tables: Dict[int, pd.DataFrame] = {}
    if y_true.size == 0:
        for q in quantile_counts:
            tables[int(q)] = pd.DataFrame()
        return tables

    df = pd.DataFrame(
        {
            "Actual": y_true,
            "Predicted": y_pred,
        }
    )
    df["Ratio"] = df["Predicted"] / df["Actual"]
    df["Error"] = df["Predicted"] - df["Actual"]
    df["AbsError"] = df["Error"].abs()
    df["PEFrac"] = df["Error"] / df["Actual"]
    df["APEFrac"] = df["AbsError"] / df["Actual"]
    df["DAPERepoPct"] = 100.0 * np.abs((df["Actual"] / df["Predicted"]) - 1.0)
    total_actual = float(df["Actual"].sum())
    for q in quantile_counts:
        q_int = int(q)
        try:
            bins = pd.qcut(df["Actual"], q=q_int, labels=False, duplicates="drop")
        except ValueError:
            bins = pd.Series(np.zeros(df.shape[0], dtype=int), index=df.index)

        bin_codes = pd.to_numeric(pd.Series(bins, index=df.index), errors="coerce")
        valid_codes = sorted(int(code) for code in bin_codes.dropna().unique().tolist())
        rows: List[Dict[str, Any]] = []
        for code in valid_codes:
            group = df.loc[bin_codes == code, :].copy()
            if group.empty:
                continue
            ratios = group["Ratio"].to_numpy(dtype=float)
            actual = group["Actual"].to_numpy(dtype=float)
            predicted = group["Predicted"].to_numpy(dtype=float)
            rows.append(
                {
                    "Quantile": int(code + 1),
                    "Count": int(group.shape[0]),
                    "Actual Share (%)": float(100.0 * group["Actual"].sum() / total_actual) if total_actual > 0.0 else np.nan,
                    "Min ($)": float(group["Actual"].min()),
                    "Max ($)": float(group["Actual"].max()),
                    "Mean Actual ($)": float(group["Actual"].mean()),
                    "Mean Pred ($)": float(group["Predicted"].mean()),
                    "Median Pred ($)": float(group["Predicted"].median()),
                    "Mean Error ($)": float(group["Error"].mean()),
                    "Mean Error (%)": float(100.0 * group["PEFrac"].mean()),
                    "MAE ($)": float(group["AbsError"].mean()),
                    "MAPE (%)": float(100.0 * group["APEFrac"].mean()),
                    "MdAPE (%)": float(group["DAPERepoPct"].median()),
                    "Mean Ratio": float(np.mean(ratios)),
                    "Median Ratio": float(np.median(ratios)),
                    "Weighted Mean Ratio": float(predicted.sum() / actual.sum()) if float(actual.sum()) > 0.0 else np.nan,
                    "PRD": float(prd(predicted, actual, na_rm=True)),
                    "PRB": float(prb(predicted, actual, na_rm=True)),
                }
            )

        table = pd.DataFrame(rows)
        if not table.empty:
            table = table.sort_values("Quantile").set_index("Quantile")
        tables[q_int] = table

    return tables


def _quantile_table_to_metric_dict(
    quantile_tables: Dict[int, pd.DataFrame],
    *,
    quantile_counts: Tuple[int, ...] = _QUANTILE_ANALYSIS_COUNTS,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    for q in quantile_counts:
        q_int = int(q)
        table = quantile_tables.get(q_int, pd.DataFrame())
        effective_bins = int(table.shape[0]) if table is not None else 0
        metrics[f"effective_bins_q{q_int}"] = effective_bins
        for bin_idx in range(1, q_int + 1):
            col_suffix = f"q{q_int}_bin{bin_idx}"
            if table is None or table.empty or bin_idx not in table.index:
                metrics[f"Count_{col_suffix}"] = np.nan
                metrics[f"ActualSharePct_{col_suffix}"] = np.nan
                metrics[f"MeanActual_{col_suffix}"] = np.nan
                metrics[f"MeanError_{col_suffix}"] = np.nan
                metrics[f"MPE_{col_suffix}"] = np.nan
                metrics[f"MeanRatio_{col_suffix}"] = np.nan
                metrics[f"MedianRatio_{col_suffix}"] = np.nan
                metrics[f"WeightedMeanRatio_{col_suffix}"] = np.nan
                metrics[f"PRD_{col_suffix}"] = np.nan
                metrics[f"PRB_{col_suffix}"] = np.nan
                metrics[f"MAE_{col_suffix}"] = np.nan
                metrics[f"MAPE_{col_suffix}"] = np.nan
                metrics[f"MdAPE_{col_suffix}"] = np.nan
                metrics[f"MeanPred_{col_suffix}"] = np.nan
                metrics[f"MedianPred_{col_suffix}"] = np.nan
                continue

            row = table.loc[bin_idx]
            metrics[f"Count_{col_suffix}"] = int(row["Count"])
            metrics[f"ActualSharePct_{col_suffix}"] = float(row["Actual Share (%)"])
            metrics[f"MeanActual_{col_suffix}"] = float(row["Mean Actual ($)"])
            metrics[f"MeanError_{col_suffix}"] = float(row["Mean Error ($)"])
            metrics[f"MPE_{col_suffix}"] = float(row["Mean Error (%)"]) / 100.0
            metrics[f"MeanRatio_{col_suffix}"] = float(row["Mean Ratio"])
            metrics[f"MedianRatio_{col_suffix}"] = float(row["Median Ratio"])
            metrics[f"WeightedMeanRatio_{col_suffix}"] = float(row["Weighted Mean Ratio"])
            metrics[f"PRD_{col_suffix}"] = float(row["PRD"])
            metrics[f"PRB_{col_suffix}"] = float(row["PRB"])
            metrics[f"MAE_{col_suffix}"] = float(row["MAE ($)"])
            metrics[f"MAPE_{col_suffix}"] = float(row["MAPE (%)"]) / 100.0
            metrics[f"MdAPE_{col_suffix}"] = float(row["MdAPE (%)"])
            metrics[f"MeanPred_{col_suffix}"] = float(row["Mean Pred ($)"])
            metrics[f"MedianPred_{col_suffix}"] = float(row["Median Pred ($)"])
    return metrics


def _render_quantile_contrast(table: pd.DataFrame) -> str:
    if table is None or table.empty or table.shape[0] < 2:
        return "Low-vs-high quantile contrast unavailable."

    low = table.iloc[0]
    high = table.iloc[-1]
    median_ratio_drop = float(low["Median Ratio"] - high["Median Ratio"])
    weighted_ratio_drop = float(low["Weighted Mean Ratio"] - high["Weighted Mean Ratio"])
    mean_error_gap = float(high["Mean Error ($)"] - low["Mean Error ($)"])
    mpe_gap = float(high["Mean Error (%)"] - low["Mean Error (%)"])
    mape_gap = float(high["MAPE (%)"] - low["MAPE (%)"])
    mean_actual_multiple = (
        float(high["Mean Actual ($)"] / low["Mean Actual ($)"])
        if np.isfinite(low["Mean Actual ($)"]) and float(low["Mean Actual ($)"]) > 0.0
        else np.nan
    )
    direction = "regressive" if median_ratio_drop > 0.0 else ("progressive" if median_ratio_drop < 0.0 else "flat")
    return (
        "Low-vs-high strata contrast: "
        f"median_ratio_drop={_fmt_num(median_ratio_drop, 4)}, "
        f"weighted_ratio_drop={_fmt_num(weighted_ratio_drop, 4)}, "
        f"mean_error_gap={_fmt_num(mean_error_gap, 2, comma=True)}, "
        f"MPE_gap={_fmt_num(mpe_gap, 2)} pts, "
        f"MAPE_gap={_fmt_num(mape_gap, 2)} pts, "
        f"mean_actual_multiple={_fmt_num(mean_actual_multiple, 2)}x | "
        f"Interpretation: {direction} tilt if ratio drop stays positive."
    )


def _render_quantile_table(table: pd.DataFrame, *, n_quantiles: int) -> str:
    title = f"DIAGNOSTICS BY PRICE QUANTILE (by Actual; n_quantiles={int(n_quantiles)})"
    if table is None or table.empty:
        return "\n".join(["=" * 75, title, "=" * 75, "(empty)", "=" * 75])

    display_df = table.loc[
        :,
        [
            "Count",
            "Actual Share (%)",
            "Min ($)",
            "Max ($)",
            "Mean Actual ($)",
            "Mean Error ($)",
            "Mean Error (%)",
            "MAE ($)",
            "MAPE (%)",
            "Mean Ratio",
            "Median Ratio",
            "Weighted Mean Ratio",
            "PRD",
            "PRB",
        ],
    ].copy()

    formatters = {
        "Count": lambda x: f"{int(x):,}",
        "Actual Share (%)": lambda x: _fmt_num(x, 1),
        "Min ($)": lambda x: _fmt_num(x, 0, comma=True),
        "Max ($)": lambda x: _fmt_num(x, 0, comma=True),
        "Mean Actual ($)": lambda x: _fmt_num(x, 0, comma=True),
        "Mean Error ($)": lambda x: _fmt_num(x, 0, comma=True),
        "Mean Error (%)": lambda x: _fmt_num(x, 2),
        "MAE ($)": lambda x: _fmt_num(x, 0, comma=True),
        "MAPE (%)": lambda x: _fmt_num(x, 2),
        "Mean Ratio": lambda x: _fmt_num(x, 4),
        "Median Ratio": lambda x: _fmt_num(x, 4),
        "Weighted Mean Ratio": lambda x: _fmt_num(x, 4),
        "PRD": lambda x: _fmt_num(x, 4),
        "PRB": lambda x: _fmt_num(x, 4),
    }

    lines = ["=" * 75, title, "=" * 75, display_df.to_string(formatters=formatters), "=" * 75]
    lines.append(_render_quantile_contrast(display_df))
    return "\n".join(lines)


def _render_price_dependence_table(metrics_row: Dict[str, Any]) -> str:
    corr_price = float(metrics_row.get("Corr(r,price)", np.nan))
    corr_logprice = float(metrics_row.get("Corr(r,logprice)", np.nan))
    dcor = float(metrics_row.get("dCor(r,logprice)_sampled", np.nan))
    xi = float(metrics_row.get("ChatterjeeXi(r,logprice)", np.nan))
    nhsic = float(metrics_row.get("nHSIC(r,logprice)_sampled", np.nan))

    rows = [
        (
            "Corr(r,price)",
            _fmt_num(corr_price, 4),
            "Near 0",
            "Negative suggests underprediction at higher prices" if np.isfinite(corr_price) and corr_price < -0.02 else "Closer to 0 is better",
        ),
        (
            "Corr(r,logprice)",
            _fmt_num(corr_logprice, 4),
            "Near 0",
            "Negative suggests underprediction at higher prices" if np.isfinite(corr_logprice) and corr_logprice < -0.02 else "Closer to 0 is better",
        ),
        (
            "dCor(r,logprice)",
            _fmt_num(dcor, 4),
            "Near 0",
            "Lower is better",
        ),
        (
            "ChatterjeeXi(r,logprice)",
            _fmt_num(xi, 4),
            "Near 0",
            "Lower is better",
        ),
        (
            "nHSIC(r,logprice)",
            _fmt_num(nhsic, 4),
            "Near 0",
            "Lower is better",
        ),
    ]
    return _render_metric_table(rows, title="PRICE-DEPENDENCE DIAGNOSTICS")


def _build_model_report_text(
    *,
    split_label: str,
    model_name: str,
    estimator_repr: str,
    metrics_row: Dict[str, Any],
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
) -> str:
    property_class = "Residential Improved"
    cod_range = IAAO_COD_RANGES.get(property_class)
    median_ratio = float(metrics_row.get("Median ratio", np.nan))
    weighted_mean_ratio = float(metrics_row.get("W. Mean ratio", np.nan))
    cod_value = float(metrics_row.get("COD", np.nan))
    prd_value = float(metrics_row.get("PRD", np.nan))
    prb_value = float(metrics_row.get("PRB", np.nan))
    vei_value = float(metrics_row.get("VEI", np.nan))

    accuracy_rows = [
        ("Count", _fmt_num(metrics_row.get("val_rows", len(y_true_log)), 0, comma=True), "—", "Valid observations"),
        ("Mean Price ($)", _fmt_num(np.mean(np.exp(y_true_log)), 2, comma=True), "—", "Scale of target"),
        ("R2", _fmt_num(metrics_row.get("R2", np.nan), 4), "—", "Closer to 1 is better"),
        ("OOS R2", _fmt_num(metrics_row.get("OOS R2", np.nan), 4), "—", "Closer to 1 is better"),
        ("R2 (log)", _fmt_num(metrics_row.get("R2 (log)", np.nan), 4), "—", "Closer to 1 is better"),
        ("MAE ($)", _fmt_num(metrics_row.get("MAE", np.nan), 2, comma=True), "—", "Lower is better"),
        ("RMSE ($)", _fmt_num(metrics_row.get("RMSE", np.nan), 2, comma=True), "—", "Lower is better"),
        ("MPE (%)", _fmt_num(100.0 * float(metrics_row.get("MPE", np.nan)), 2), "—", "Signed average percent bias"),
        ("MAPE (%)", _fmt_num(100.0 * float(metrics_row.get("MAPE", np.nan)), 2), "—", "Lower is better"),
        ("MdAPE (%)", _fmt_num(metrics_row.get("MdAPE", np.nan), 2), "—", "Robust to outliers"),
    ]
    fairness_rows = [
        ("Median Ratio", _fmt_num(median_ratio, 4), _fmt_range(IAAO_LEVEL_RANGE, 2), _interp_level(median_ratio)),
        ("Weighted Mean Ratio", _fmt_num(weighted_mean_ratio, 4), _fmt_range(IAAO_LEVEL_RANGE, 2), _interp_level(weighted_mean_ratio)),
        (
            f"COD (%) [{property_class}]",
            _fmt_num(cod_value, 2),
            _fmt_range(cod_range, 1) if cod_range is not None else "—",
            _interp_cod(cod_value, property_class=property_class),
        ),
        ("PRD", _fmt_num(prd_value, 4), _fmt_range(IAAO_PRD_RANGE, 2), _interp_prd(prd_value)),
        ("PRB", _fmt_num(prb_value, 4), _fmt_range(IAAO_PRB_RANGE, 2), _interp_prb(prb_value)),
        ("VEI (%)", _fmt_num(vei_value, 2), _fmt_range(IAAO_VEI_RANGE, 0, pct=True), _interp_vei(vei_value)),
    ]

    quantile_tables = _compute_quantile_diagnostic_tables(
        y_true_log=y_true_log,
        y_pred_log=y_pred_log,
        quantile_counts=_QUANTILE_ANALYSIS_COUNTS,
    )

    parts = [
        "=" * 100,
        f"Split: {split_label}",
        f"Model: {model_name}",
        f"Fitting:  {estimator_repr}",
        "=" * 100,
        _render_metric_table(accuracy_rows, title="MODEL ACCURACY (PRICE SCALE + LOG SCALE)"),
        _render_metric_table(fairness_rows, title="IAAO-STYLE RATIO STUDY METRICS (Pred/Actual as AV/SP)"),
        _render_price_dependence_table(metrics_row),
    ]
    for q in _QUANTILE_ANALYSIS_COUNTS:
        parts.append(_render_quantile_table(quantile_tables.get(int(q), pd.DataFrame()), n_quantiles=int(q)))
    return "\n\n".join(parts)


def _write_split_report(
    *,
    split_label: str,
    results_df: pd.DataFrame,
    y_true_log: np.ndarray,
    pred_logs: Dict[str, np.ndarray],
    model_specs_by_name: Dict[str, Dict[str, Any]],
    out_path: Path,
) -> None:
    if results_df.empty or y_true_log.size == 0 or not pred_logs:
        return

    parts: List[str] = []
    for _, row in results_df.iterrows():
        model_name = str(row.get("model_name", "model"))
        y_pred_log = pred_logs.get(model_name)
        if y_pred_log is None:
            continue
        spec = model_specs_by_name.get(model_name, {})
        estimator_repr = repr(spec.get("estimator", model_name))
        parts.append(
            _build_model_report_text(
                split_label=split_label,
                model_name=model_name,
                estimator_repr=estimator_repr,
                metrics_row=row.to_dict(),
                y_true_log=np.asarray(y_true_log, dtype=float),
                y_pred_log=np.asarray(y_pred_log, dtype=float),
            )
        )

    if not parts:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n\n".join(parts) + "\n", encoding="utf-8")


def _compute_quantile_block_error_metrics(
    *,
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    quantile_counts: Tuple[int, ...] = _QUANTILE_ANALYSIS_COUNTS,
) -> Dict[str, Any]:
    quantile_tables = _compute_quantile_diagnostic_tables(
        y_true_log=y_true_log,
        y_pred_log=y_pred_log,
        quantile_counts=quantile_counts,
    )
    return _quantile_table_to_metric_dict(
        quantile_tables,
        quantile_counts=quantile_counts,
    )


def _compute_quick_test_metrics(
    *,
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    y_train_log: np.ndarray,
    ratio_mode: str,
) -> Dict[str, Any]:
    metrics = _compute_extended_metrics(
        y_true_log=y_true_log,
        y_pred_log=y_pred_log,
        y_train_log=y_train_log,
        ratio_mode=ratio_mode,
    )
    metrics.update(
        compute_projection_theory_metrics(
            y_true_log=y_true_log,
            y_pred_log=y_pred_log,
        )
    )
    y_true = np.exp(np.asarray(y_true_log, dtype=float).reshape(-1))
    y_pred = np.exp(np.asarray(y_pred_log, dtype=float).reshape(-1))
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0.0)
    if np.any(mask):
        metrics["MPE"] = float(np.mean((y_pred[mask] - y_true[mask]) / y_true[mask]))
    else:
        metrics["MPE"] = np.nan
    if _COMPUTE_DEPENDENCE_METRICS:
        metrics.update(
            _compute_logprice_dependence_metrics(
                y_true_log=y_true_log,
                y_pred_log=y_pred_log,
                ratio_mode=ratio_mode,
            )
        )
    metrics.update(
        _compute_quantile_block_error_metrics(
            y_true_log=y_true_log,
            y_pred_log=y_pred_log,
        )
    )
    return metrics


def _build_quick_test_models(
    *,
    rho_values: List[float],
    eta_values: List[float],
    keep_values: List[float],
    lgbm_params: dict,
    early_stopping_rounds: int | None,
    model_families: List[str] | None = None,
) -> List[Dict[str, Any]]:
    models: List[Dict[str, Any]] = []

    rho_list = [float(r) for r in rho_values]
    enabled = set(model_families or ["linear", "lgbm", "cov", "smooth"])

    if "linear" in enabled:
        models.append(
            {
                "model_name": "LinearRegression",
                "model_family": "LinearRegression",
                "ratio_mode": "diff",
                "rho": np.nan,
                "rho_group": np.nan,
                "estimator": LinearRegression(fit_intercept=True),
                "requires_linear_pipeline": True,
            }
        )
    if "linear_cov" in enabled:
        for rho_value in rho_list:
            models.append(
                {
                    "model_name": f"LinearProjectionCovariancePath_rho_{rho_value}",
                    "model_family": "LinearProjectionCovariancePath",
                    "ratio_mode": "diff",
                    "rho": float(rho_value),
                    "rho_group": np.nan,
                    "estimator": LinearProjectionCovariancePath(rho=float(rho_value), fit_intercept=True),
                    "requires_linear_pipeline": True,
                }
            )
    if "lgbm" in enabled:
        models.append(
            {
                "model_name": "LGBMRegressor",
                "model_family": "LGBMRegressor",
                "ratio_mode": "diff",
                "rho": np.nan,
                "rho_group": np.nan,
                "estimator": lgb.LGBMRegressor(**lgbm_params),
                "requires_linear_pipeline": False,
            }
        )

    for ratio_mode in _RATIO_MODES:
        for rho_value in rho_list:
            if "cov" in enabled:
                models.append(
                    {
                        "model_name": f"LGBCovPenalty_mode_{ratio_mode}_rho_{rho_value}",
                        "model_family": f"LGBCovPenalty[{ratio_mode}]",
                        "ratio_mode": ratio_mode,
                        "rho": float(rho_value),
                        "rho_group": np.nan,
                        "estimator": LGBCovPenalty(
                            rho=float(rho_value),
                            ratio_mode=ratio_mode,
                            early_stopping_rounds=early_stopping_rounds,
                            zero_grad_tol=1e-12,
                            lgbm_params=lgbm_params,
                            verbose=True,
                        ),
                        "requires_linear_pipeline": False,
                    }
                )
            if "smooth" in enabled:
                models.append(
                    {
                        "model_name": f"LGBSmoothPenalty_mode_{ratio_mode}_rho_{rho_value}",
                        "model_family": f"LGBSmoothPenalty[{ratio_mode}]",
                        "ratio_mode": ratio_mode,
                        "rho": float(rho_value),
                        "rho_group": np.nan,
                        "estimator": LGBSmoothPenalty(
                            rho=float(rho_value),
                            ratio_mode=ratio_mode,
                            early_stopping_rounds=early_stopping_rounds,
                            zero_grad_tol=1e-12,
                            lgbm_params=lgbm_params,
                            verbose=True,
                        ),
                        "requires_linear_pipeline": False,
                    }
                )
    if not models:
        raise ValueError("No models selected. Use --models linear,linear_cov,lgbm,cov,smooth or --models all.")
    return models


def _write_rho_evolution_plot(
    df: pd.DataFrame,
    *,
    split_label: str,
    out_path: Path,
) -> None:
    if df.empty or "model_family" not in df.columns or "rho" not in df.columns:
        return

    plot_df = df.loc[df["model_family"].isin(_MAIN_SWEEP_FAMILIES), :].copy()
    if plot_df.empty:
        return

    plot_df["rho"] = pd.to_numeric(plot_df["rho"], errors="coerce")
    plot_df = plot_df.loc[np.isfinite(plot_df["rho"]), :]
    if plot_df.empty:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    metric_names = [metric_name for metric_name in _RHO_PLOT_METRICS if metric_name in plot_df.columns]
    if not metric_names:
        return
    for metric_name in metric_names:
        plot_df[metric_name] = pd.to_numeric(plot_df[metric_name], errors="coerce")

    for metric_name in metric_names:
        metric_out_path = out_path.with_name(
            f"{out_path.stem}_{_sanitize_plot_filename(metric_name.lower())}{out_path.suffix}"
        )
        _write_rho_evolution_original_metric_plot(
            plot_df,
            family_names=_MAIN_SWEEP_FAMILIES,
            metric_name=metric_name,
            split_label=split_label,
            out_path=metric_out_path,
        )

    _write_rho_evolution_normalized_plot(
        plot_df,
        family_names=_MAIN_SWEEP_FAMILIES,
        metric_names=metric_names,
        split_label=split_label,
        out_path=out_path,
    )


def _write_grouped_rho_evolution_plots(
    df: pd.DataFrame,
    *,
    split_label: str,
    out_dir: Path,
) -> None:
    if df.empty or "model_family" not in df.columns or "rho" not in df.columns or "rho_group" not in df.columns:
        return

    plot_df = df.loc[df["model_family"] == _GROUPED_SWEEP_FAMILY, :].copy()
    if plot_df.empty:
        return

    plot_df["rho"] = pd.to_numeric(plot_df["rho"], errors="coerce")
    plot_df["rho_group"] = pd.to_numeric(plot_df["rho_group"], errors="coerce")
    plot_df = plot_df.loc[np.isfinite(plot_df["rho"]) & np.isfinite(plot_df["rho_group"]), :]
    if plot_df.empty:
        return

    metric_names = [metric_name for metric_name in _RHO_PLOT_METRICS if metric_name in plot_df.columns]
    if not metric_names:
        return

    split_slug = _sanitize_plot_filename(split_label.lower())
    out_dir.mkdir(parents=True, exist_ok=True)
    rho_group_values = sorted(float(v) for v in plot_df["rho_group"].dropna().unique().tolist())

    for rho_group_value in rho_group_values:
        family_df = plot_df.loc[plot_df["rho_group"] == rho_group_value, :].copy()
        if family_df.empty:
            continue
        rho_group_slug = _sanitize_plot_filename(f"{rho_group_value}")
        base_out_path = out_dir / f"quick_test_rho_evolution_grouped_rho_group_{rho_group_slug}_{split_slug}.pdf"

        for metric_name in metric_names:
            metric_out_path = base_out_path.with_name(
                f"{base_out_path.stem}_{_sanitize_plot_filename(metric_name.lower())}{base_out_path.suffix}"
            )
            _write_rho_evolution_original_metric_plot(
                family_df,
                family_names=(_GROUPED_SWEEP_FAMILY,),
                metric_name=metric_name,
                split_label=f"{split_label}, rho_group={rho_group_value}",
                out_path=metric_out_path,
            )

        _write_rho_evolution_normalized_plot(
            family_df,
            family_names=(_GROUPED_SWEEP_FAMILY,),
            metric_names=metric_names,
            split_label=f"{split_label}, rho_group={rho_group_value}",
            out_path=base_out_path,
        )


def _rho_family_metric_frame(
    plot_df: pd.DataFrame,
    *,
    family: str,
    metric_names: Tuple[str, ...] | List[str],
) -> pd.DataFrame:
    family_df = plot_df.loc[plot_df["model_family"] == family, :].copy()
    if family_df.empty:
        return family_df
    cols = ["rho"] + [metric_name for metric_name in metric_names if metric_name in family_df.columns]
    if family_df["rho"].duplicated().any():
        return (
            family_df.groupby("rho", as_index=False)[cols[1:]]
            .mean(numeric_only=True)
            .sort_values("rho")
        )
    return family_df.loc[:, cols].sort_values("rho")


def _normalize_rho_metric_values(
    metric_name: str,
    values: np.ndarray,
    *,
    reference_values: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    value_mask = np.isfinite(values)
    if not value_mask.any():
        return out

    reference = values if reference_values is None else np.asarray(reference_values, dtype=float)
    reference = reference[np.isfinite(reference)]
    if reference.size == 0:
        return out

    if metric_name in _RHO_HIGHER_IS_BETTER:
        lo = float(np.nanmin(reference))
        hi = float(np.nanmax(reference))
        denom = hi - lo
        if denom <= 0.0:
            out[value_mask] = 1.0
        else:
            out[value_mask] = (values[value_mask] - lo) / denom
    elif metric_name in _RHO_LOWER_IS_BETTER:
        lo = float(np.nanmin(reference))
        hi = float(np.nanmax(reference))
        denom = hi - lo
        if denom <= 0.0:
            out[value_mask] = 1.0
        else:
            out[value_mask] = (hi - values[value_mask]) / denom
    elif metric_name in _RHO_TARGET_IS_BETTER:
        target = float(_RHO_TARGET_IS_BETTER[metric_name])
        reference_deviation = np.abs(reference - target)
        best = float(np.nanmin(reference_deviation))
        worst = float(np.nanmax(reference_deviation))
        denom = worst - best
        if denom <= 0.0:
            out[value_mask] = 1.0
        else:
            out[value_mask] = (worst - np.abs(values[value_mask] - target)) / denom
    else:
        lo = float(np.nanmin(reference))
        hi = float(np.nanmax(reference))
        denom = hi - lo
        if denom <= 0.0:
            out[value_mask] = 1.0
        else:
            out[value_mask] = (values[value_mask] - lo) / denom

    out[value_mask] = np.clip(out[value_mask], 0.0, 1.0)
    return out


def _write_rho_evolution_original_metric_plot(
    plot_df: pd.DataFrame,
    *,
    family_names: Tuple[str, ...],
    metric_name: str,
    split_label: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, len(family_names), figsize=(6 * len(family_names), 5), sharey=True)
    if len(family_names) == 1:
        axes = [axes]

    any_plotted = False
    color = _RHO_METRIC_COLORS.get(metric_name, "C0")
    std_col = f"{metric_name}_std"
    for ax, family in zip(axes, family_names):
        family_df = _rho_family_metric_frame(plot_df, family=family, metric_names=(metric_name, std_col))
        if family_df.empty or metric_name not in family_df.columns:
            ax.set_visible(False)
            continue

        y = pd.to_numeric(family_df[metric_name], errors="coerce").to_numpy(dtype=float)
        x = family_df["rho"].to_numpy(dtype=float)
        if not np.isfinite(y).any():
            ax.set_visible(False)
            continue

        ax.plot(x, y, marker="o", linewidth=1.8, linestyle="--", color=color, label=metric_name)
        if std_col in family_df.columns:
            y_std = np.abs(pd.to_numeric(family_df[std_col], errors="coerce").to_numpy(dtype=float))
            band_mask = np.isfinite(y) & np.isfinite(y_std)
            if band_mask.any():
                ax.fill_between(
                    x[band_mask],
                    y[band_mask] - y_std[band_mask],
                    y[band_mask] + y_std[band_mask],
                    color=color,
                    alpha=0.14,
                    linewidth=0.0,
                )
        ax.set_title(family)
        ax.set_xlabel("rho")
        ax.grid(True, linestyle=":", alpha=0.4)
        any_plotted = True

    if not any_plotted:
        plt.close(fig)
        return

    axes[0].set_ylabel(metric_name)
    fig.suptitle(f"{metric_name} Evolution vs rho ({split_label})")
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.92))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_rho_evolution_normalized_plot(
    plot_df: pd.DataFrame,
    *,
    family_names: Tuple[str, ...],
    metric_names: List[str] | Tuple[str, ...],
    split_label: str,
    out_path: Path,
) -> None:
    references = {
        metric_name: pd.to_numeric(plot_df[metric_name], errors="coerce").to_numpy(dtype=float)
        for metric_name in metric_names
        if metric_name in plot_df.columns
    }
    if not references:
        return

    fig, axes = plt.subplots(1, len(family_names), figsize=(6 * len(family_names), 5), sharey=True)
    if len(family_names) == 1:
        axes = [axes]

    any_plotted = False
    first_handles = None
    first_labels = None
    for ax, family in zip(axes, family_names):
        family_df = _rho_family_metric_frame(plot_df, family=family, metric_names=metric_names)
        if family_df.empty:
            ax.set_visible(False)
            continue

        for metric_name in metric_names:
            if metric_name not in family_df.columns or metric_name not in references:
                continue
            y = pd.to_numeric(family_df[metric_name], errors="coerce").to_numpy(dtype=float)
            y_norm = _normalize_rho_metric_values(
                metric_name,
                y,
                reference_values=references[metric_name],
            )
            if not np.isfinite(y_norm).any():
                continue
            ax.plot(
                family_df["rho"].to_numpy(dtype=float),
                y_norm,
                marker="o",
                linewidth=1.8,
                linestyle="--",
                color=_RHO_METRIC_COLORS.get(metric_name, None),
                label=metric_name,
            )
            any_plotted = True

        ax.set_ylim(-0.03, 1.03)
        ax.set_title(family)
        ax.set_xlabel("rho")
        ax.grid(True, linestyle=":", alpha=0.4)
        if first_handles is None:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                first_handles = handles
                first_labels = labels

    if not any_plotted:
        plt.close(fig)
        return

    axes[0].set_ylabel("normalized score (0=worst, 1=best)")
    if first_handles:
        fig.legend(first_handles, first_labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Normalized Metric Evolution vs rho ({split_label})")
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.92))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _blend_with_white(color: Any, intensity: float) -> Tuple[float, float, float]:
    intensity = float(np.clip(intensity, 0.0, 1.0))
    rgb = np.asarray(mcolors.to_rgb(color), dtype=float)
    return tuple((1.0 - intensity) + intensity * rgb)


def _tradeoff_band(metric_name: str) -> Tuple[float, float, float] | None:
    if metric_name == "PRD":
        return float(IAAO_PRD_RANGE[0]), float(IAAO_PRD_RANGE[1]), 1.0
    if metric_name == "PRB":
        return float(IAAO_PRB_RANGE[0]), float(IAAO_PRB_RANGE[1]), 0.0
    if metric_name == "VEI":
        return float(IAAO_VEI_RANGE[0]), float(IAAO_VEI_RANGE[1]), 0.0
    return None


def _format_rho_label(value: float) -> str:
    if not np.isfinite(value):
        return "rho=nan"
    if value == 0.0:
        return "rho=0"
    abs_val = abs(float(value))
    if abs_val >= 1000.0 or abs_val < 0.01:
        return f"rho={value:.1e}"
    if abs_val >= 10.0:
        return f"rho={value:.1f}"
    return f"rho={value:.3g}"


def _write_tradeoff_plot(
    df: pd.DataFrame,
    *,
    split_label: str,
    out_path: Path,
    y_metric: str,
) -> None:
    if df.empty or y_metric not in df.columns or "model_family" not in df.columns:
        return

    allowed_families = set(_TRADEOFF_BASELINE_FAMILIES).union(_MAIN_SWEEP_FAMILIES)
    plot_df = df.loc[df["model_family"].isin(allowed_families), :].copy()
    if plot_df.empty:
        return

    numeric_cols = list(_TRADEOFF_FAIRNESS_METRICS) + [y_metric, "rho"]
    for col in numeric_cols:
        if col in plot_df.columns:
            plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

    fairness_metrics = [metric for metric in _TRADEOFF_FAIRNESS_METRICS if metric in plot_df.columns]
    if not fairness_metrics:
        return

    fig, axes = plt.subplots(1, len(fairness_metrics), figsize=(6.2 * len(fairness_metrics), 5.6), sharey=True)
    if len(fairness_metrics) == 1:
        axes = [axes]

    for ax, fairness_metric in zip(axes, fairness_metrics):
        ax.grid(True, linestyle=":", alpha=0.4)
        band = _tradeoff_band(fairness_metric)
        if band is not None:
            lo, hi, target = band
            ax.axvspan(min(lo, hi), max(lo, hi), color="limegreen", alpha=0.18, zorder=0)
            ax.axvline(target, color="forestgreen", linestyle="--", linewidth=1.2, alpha=0.9, zorder=1)

        for family in _TRADEOFF_BASELINE_FAMILIES:
            family_df = plot_df.loc[plot_df["model_family"] == family, :].copy()
            if family_df.empty:
                continue
            family_df = family_df.loc[np.isfinite(family_df[fairness_metric]) & np.isfinite(family_df[y_metric]), :]
            if family_df.empty:
                continue
            color = _TRADEOFF_FAMILY_COLORS.get(family, "C0")
            x_vals = family_df[fairness_metric].to_numpy(dtype=float)
            y_vals = family_df[y_metric].to_numpy(dtype=float)
            ax.scatter(
                x_vals,
                y_vals,
                s=150,
                marker="*",
                color=color,
                edgecolors="black",
                linewidths=0.6,
                zorder=4,
            )

        for family in _MAIN_SWEEP_FAMILIES:
            family_df = plot_df.loc[plot_df["model_family"] == family, :].copy()
            if family_df.empty or "rho" not in family_df.columns:
                continue
            family_df = family_df.loc[
                np.isfinite(family_df["rho"])
                & np.isfinite(family_df[fairness_metric])
                & np.isfinite(family_df[y_metric]),
                :,
            ].copy()
            if family_df.empty:
                continue

            metric_cols = [fairness_metric, y_metric]
            if family_df["rho"].duplicated().any():
                family_df = (
                    family_df.groupby("rho", as_index=False)[metric_cols]
                    .mean(numeric_only=True)
                    .sort_values("rho")
                )
            else:
                family_df = family_df.sort_values("rho")

            x_vals = family_df[fairness_metric].to_numpy(dtype=float)
            y_vals = family_df[y_metric].to_numpy(dtype=float)
            rho_vals = family_df["rho"].to_numpy(dtype=float)
            if x_vals.size == 0:
                continue

            base_color = _TRADEOFF_FAMILY_COLORS.get(family, "C0")
            ax.plot(x_vals, y_vals, color=base_color, linewidth=1.8, alpha=0.8, zorder=2)

            if rho_vals.size > 1:
                log_rho = np.log10(np.maximum(rho_vals, 1e-12))
                lo_rho = float(np.nanmin(log_rho))
                hi_rho = float(np.nanmax(log_rho))
                if hi_rho > lo_rho:
                    intensities = 0.35 + 0.65 * ((log_rho - lo_rho) / (hi_rho - lo_rho))
                else:
                    intensities = np.full_like(log_rho, 0.85, dtype=float)
            else:
                intensities = np.array([0.85], dtype=float)

            colors = [_blend_with_white(base_color, val) for val in intensities.tolist()]
            ax.scatter(
                x_vals,
                y_vals,
                s=48,
                color=colors,
                edgecolors=base_color,
                linewidths=0.8,
                zorder=3,
            )

            for idx in range(len(x_vals) - 1):
                ax.annotate(
                    "",
                    xy=(x_vals[idx + 1], y_vals[idx + 1]),
                    xytext=(x_vals[idx], y_vals[idx]),
                    arrowprops=dict(
                        arrowstyle="-|>",      # A filled arrow head usually looks cleaner than the open "->"
                        color=base_color, 
                        lw=0.8,                # Thinner line weight 
                        alpha=0.5,             # Lower opacity to let the underlying line/points show
                        shrinkA=3,             # Pulls the tail away from the starting point (in points)
                        shrinkB=3,             # Pulls the head away from the ending point (in points)
                        mutation_scale=10      # Controls the overall size of the arrow head (default is often too big)
                    ),
                    # arrowprops=dict(arrowstyle="->", color=base_color, lw=1.2, alpha=0.75),
                    zorder=2,
                )

            ax.annotate(
                _format_rho_label(float(rho_vals[0])),
                xy=(x_vals[0], y_vals[0]),
                xytext=(4, 6),
                textcoords="offset points",
                fontsize=7,
                color=base_color,
            )
            if len(x_vals) > 1:
                ax.annotate(
                    _format_rho_label(float(rho_vals[-1])),
                    xy=(x_vals[-1], y_vals[-1]),
                    xytext=(4, -10),
                    textcoords="offset points",
                    fontsize=7,
                    color=base_color,
                )

        ax.set_xlabel(fairness_metric)
        ax.set_title(f"{fairness_metric} vs {y_metric}")

    axes[0].set_ylabel(y_metric)
    legend_handles = [
        Patch(facecolor="limegreen", edgecolor="none", alpha=0.18, label="IAAO band"),
        Line2D([0], [0], color="forestgreen", linestyle="--", linewidth=1.2, label="IAAO target"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor=_TRADEOFF_FAMILY_COLORS["LinearRegression"], markeredgecolor="black", markersize=12, label="LinearRegression"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor=_TRADEOFF_FAMILY_COLORS["LGBMRegressor"], markeredgecolor="black", markersize=12, label="LGBMRegressor"),
        Line2D([0], [0], marker="o", color=_TRADEOFF_FAMILY_COLORS["LGBCovPenalty[diff]"], markerfacecolor=_TRADEOFF_FAMILY_COLORS["LGBCovPenalty[diff]"], markersize=6, linewidth=1.8, label="LGBCovPenalty"),
        Line2D([0], [0], marker="o", color=_TRADEOFF_FAMILY_COLORS["LGBSmoothPenalty[diff]"], markerfacecolor=_TRADEOFF_FAMILY_COLORS["LGBSmoothPenalty[diff]"], markersize=6, linewidth=1.8, label="LGBSmoothPenalty"),
    ]
    fig.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(f"{y_metric} Tradeoff vs PRD / PRB / VEI ({split_label})\nArrows indicate increasing rho")
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.92))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_tradeoff_plots(
    df: pd.DataFrame,
    *,
    split_label: str,
    out_dir: Path,
) -> None:
    if df.empty:
        return
    split_slug = _sanitize_plot_filename(split_label.lower())
    for y_metric in _TRADEOFF_TARGET_METRICS:
        if y_metric not in df.columns:
            continue
        out_path = out_dir / f"quick_test_tradeoff_{y_metric.lower()}_{split_slug}.pdf"
        _write_tradeoff_plot(
            df,
            split_label=split_label,
            out_path=out_path,
            y_metric=y_metric,
        )


def _sanitize_plot_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name).strip())
    return safe or "plot"


def _cap_scatter_plot_samples(
    *,
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    group_labels: np.ndarray | None,
    max_samples: int | None,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    y_true_log = np.asarray(y_true_log)
    y_pred_log = np.asarray(y_pred_log)
    if max_samples is None or int(max_samples) <= 0 or y_true_log.size <= int(max_samples):
        return y_true_log, y_pred_log, group_labels
    idx = np.random.default_rng(int(seed)).choice(y_true_log.size, size=int(max_samples), replace=False)
    labels = None if group_labels is None else np.asarray(group_labels)[idx]
    return y_true_log[idx], y_pred_log[idx], labels


def _write_ratio_vs_logprice_plots(
    *,
    split_label: str,
    results_df: pd.DataFrame,
    y_true_log: np.ndarray,
    pred_logs: Dict[str, np.ndarray],
    out_dir: Path,
    grouped_feature_labels: np.ndarray | None = None,
    grouped_feature_name: str | None = None,
    scatter_plot_max_samples: int | None = None,
    scatter_plot_sample_seed: int = 0,
) -> None:
    if results_df.empty or not pred_logs:
        return

    split_dir = out_dir / split_label.lower()
    split_dir.mkdir(parents=True, exist_ok=True)

    for _, row in results_df.iterrows():
        model_name = str(row.get("model_name", "model"))
        y_pred_log = pred_logs.get(model_name)
        if y_pred_log is None:
            continue
        model_family = str(row.get("model_family", ""))
        color_labels = grouped_feature_labels if model_family == _GROUPED_SWEEP_FAMILY else None
        y_true_plot, y_pred_plot, color_labels = _cap_scatter_plot_samples(
            y_true_log=y_true_log,
            y_pred_log=y_pred_log,
            group_labels=color_labels,
            max_samples=scatter_plot_max_samples,
            seed=scatter_plot_sample_seed,
        )
        out_path = split_dir / f"{_sanitize_plot_filename(model_name)}.pdf"
        plot_ratio_vs_logprice(
            y_true_log=y_true_plot,
            y_pred_log=y_pred_plot,
            out_path=out_path,
            model_label=model_name,
            split_label=split_label,
            metrics=row.to_dict(),
            group_labels=color_labels,
            group_label_name=(grouped_feature_name if color_labels is not None else None),
            y_limits=(0.0, 4.0),
        )


def _write_residual_vs_logprice_plots(
    *,
    split_label: str,
    results_df: pd.DataFrame,
    y_true_log: np.ndarray,
    pred_logs: Dict[str, np.ndarray],
    out_dir: Path,
    grouped_feature_labels: np.ndarray | None = None,
    grouped_feature_name: str | None = None,
    scatter_plot_max_samples: int | None = None,
    scatter_plot_sample_seed: int = 0,
) -> None:
    if results_df.empty or not pred_logs:
        return

    split_dir = out_dir / split_label.lower()
    split_dir.mkdir(parents=True, exist_ok=True)

    for _, row in results_df.iterrows():
        model_name = str(row.get("model_name", "model"))
        y_pred_log = pred_logs.get(model_name)
        if y_pred_log is None:
            continue
        model_family = str(row.get("model_family", ""))
        color_labels = grouped_feature_labels if model_family == _GROUPED_SWEEP_FAMILY else None
        y_true_plot, y_pred_plot, color_labels = _cap_scatter_plot_samples(
            y_true_log=y_true_log,
            y_pred_log=y_pred_log,
            group_labels=color_labels,
            max_samples=scatter_plot_max_samples,
            seed=scatter_plot_sample_seed,
        )
        out_path = split_dir / f"{_sanitize_plot_filename(model_name)}.pdf"
        plot_residual_vs_logprice(
            y_true_log=y_true_plot,
            y_pred_log=y_pred_plot,
            out_path=out_path,
            model_label=model_name,
            split_label=split_label,
            metrics=row.to_dict(),
            group_labels=color_labels,
            group_label_name=(grouped_feature_name if color_labels is not None else None),
            y_limits=(-1.5, 1.5),
        )


def _write_ratio_vs_logprediction_plots(
    *,
    split_label: str,
    results_df: pd.DataFrame,
    y_true_log: np.ndarray,
    pred_logs: Dict[str, np.ndarray],
    out_dir: Path,
    grouped_feature_labels: np.ndarray | None = None,
    grouped_feature_name: str | None = None,
    scatter_plot_max_samples: int | None = None,
    scatter_plot_sample_seed: int = 0,
) -> None:
    if results_df.empty or not pred_logs:
        return

    split_dir = out_dir / split_label.lower()
    split_dir.mkdir(parents=True, exist_ok=True)

    for _, row in results_df.iterrows():
        model_name = str(row.get("model_name", "model"))
        y_pred_log = pred_logs.get(model_name)
        if y_pred_log is None:
            continue
        model_family = str(row.get("model_family", ""))
        color_labels = grouped_feature_labels if model_family == _GROUPED_SWEEP_FAMILY else None
        y_true_plot, y_pred_plot, color_labels = _cap_scatter_plot_samples(
            y_true_log=y_true_log,
            y_pred_log=y_pred_log,
            group_labels=color_labels,
            max_samples=scatter_plot_max_samples,
            seed=scatter_plot_sample_seed,
        )
        out_path = split_dir / f"{_sanitize_plot_filename(model_name)}.pdf"
        plot_ratio_vs_logprediction(
            y_true_log=y_true_plot,
            y_pred_log=y_pred_plot,
            out_path=out_path,
            model_label=model_name,
            split_label=split_label,
            metrics=row.to_dict(),
            group_labels=color_labels,
            group_label_name=(grouped_feature_name if color_labels is not None else None),
            y_limits=(0.0, 4.0),
        )


def _write_residual_vs_logprediction_plots(
    *,
    split_label: str,
    results_df: pd.DataFrame,
    y_true_log: np.ndarray,
    pred_logs: Dict[str, np.ndarray],
    out_dir: Path,
    grouped_feature_labels: np.ndarray | None = None,
    grouped_feature_name: str | None = None,
    scatter_plot_max_samples: int | None = None,
    scatter_plot_sample_seed: int = 0,
) -> None:
    if results_df.empty or not pred_logs:
        return

    split_dir = out_dir / split_label.lower()
    split_dir.mkdir(parents=True, exist_ok=True)

    for _, row in results_df.iterrows():
        model_name = str(row.get("model_name", "model"))
        y_pred_log = pred_logs.get(model_name)
        if y_pred_log is None:
            continue
        model_family = str(row.get("model_family", ""))
        color_labels = grouped_feature_labels if model_family == _GROUPED_SWEEP_FAMILY else None
        y_true_plot, y_pred_plot, color_labels = _cap_scatter_plot_samples(
            y_true_log=y_true_log,
            y_pred_log=y_pred_log,
            group_labels=color_labels,
            max_samples=scatter_plot_max_samples,
            seed=scatter_plot_sample_seed,
        )
        out_path = split_dir / f"{_sanitize_plot_filename(model_name)}.pdf"
        plot_residual_vs_logprediction(
            y_true_log=y_true_plot,
            y_pred_log=y_pred_plot,
            out_path=out_path,
            model_label=model_name,
            split_label=split_label,
            metrics=row.to_dict(),
            group_labels=color_labels,
            group_label_name=(grouped_feature_name if color_labels is not None else None),
            y_limits=(-1.5, 1.5),
        )


def _fit_predict_and_score(
    *,
    model_name: str,
    estimator: Any,
    requires_linear_pipeline: bool,
    linear_pipeline_builder,
    X_train: pd.DataFrame,
    y_train_log: np.ndarray,
    X_eval: pd.DataFrame,
    y_eval_log: np.ndarray,
    fairness_ratio_mode: str,
    return_prediction_log: bool = False,
    X_in_sample: pd.DataFrame | None = None,
    y_in_sample_log: np.ndarray | None = None,
    return_in_sample_prediction_log: bool = False,
) -> Dict[str, Any]:
    _log(
        "model evaluation start",
        model_name=model_name,
        split_rows=int(len(y_eval_log)),
        requires_linear_pipeline=bool(requires_linear_pipeline),
    )
    if requires_linear_pipeline:
        _log("building linear pipeline", model_name=model_name)
        pipe = linear_pipeline_builder()
        X_train_m = pipe.fit_transform(X_train, y_train_log)
        X_eval_m = pipe.transform(X_eval)
        X_in_sample_m = None
        if X_in_sample is not None:
            X_in_sample_m = pipe.transform(X_in_sample)
        _log(
            "linear pipeline ready",
            model_name=model_name,
            train_matrix_shape=str(getattr(X_train_m, "shape", "")),
            eval_matrix_shape=str(getattr(X_eval_m, "shape", "")),
        )
    else:
        X_train_m = X_train
        X_eval_m = X_eval
        X_in_sample_m = X_in_sample

    _log("fitting model", model_name=model_name)
    estimator.fit(X_train_m, y_train_log)
    _log("fit completed", model_name=model_name)
    y_pred_eval_log = np.asarray(estimator.predict(X_eval_m), dtype=float).reshape(-1)
    _log("prediction completed", model_name=model_name)
    metrics = _compute_quick_test_metrics(
        y_true_log=y_eval_log,
        y_pred_log=y_pred_eval_log,
        y_train_log=y_train_log,
        ratio_mode=fairness_ratio_mode,
    )
    out = {"model_name": model_name, **metrics}
    if bool(return_prediction_log):
        out["_y_pred_eval_log"] = y_pred_eval_log
    if X_in_sample_m is not None and y_in_sample_log is not None:
        y_pred_in_sample_log = np.asarray(estimator.predict(X_in_sample_m), dtype=float).reshape(-1)
        in_sample_metrics = _compute_quick_test_metrics(
            y_true_log=np.asarray(y_in_sample_log, dtype=float),
            y_pred_log=y_pred_in_sample_log,
            y_train_log=y_train_log,
            ratio_mode=fairness_ratio_mode,
        )
        out["_in_sample_metrics"] = in_sample_metrics
        if bool(return_in_sample_prediction_log):
            out["_y_pred_in_sample_log"] = y_pred_in_sample_log
    _log("model evaluation completed", model_name=model_name)
    return out


def _evaluate_single_model_spec(
    *,
    spec: Dict[str, Any],
    linear_pipeline_builder,
    X_train: pd.DataFrame,
    y_train_log: np.ndarray,
    X_eval: pd.DataFrame,
    y_eval_log: np.ndarray,
    X_in_sample: pd.DataFrame | None,
    y_in_sample_log: np.ndarray | None,
) -> Dict[str, Any]:
    return _fit_predict_and_score(
        model_name=str(spec["model_name"]),
        estimator=spec["estimator"],
        requires_linear_pipeline=bool(spec["requires_linear_pipeline"]),
        linear_pipeline_builder=linear_pipeline_builder,
        X_train=X_train,
        y_train_log=y_train_log,
        X_eval=X_eval,
        y_eval_log=y_eval_log,
        fairness_ratio_mode=str(spec["ratio_mode"]),
        return_prediction_log=True,
        X_in_sample=X_in_sample,
        y_in_sample_log=y_in_sample_log,
        return_in_sample_prediction_log=True,
    )


def _evaluate_model_specs(
    *,
    model_specs: List[Dict[str, Any]],
    linear_pipeline_builder,
    X_train: pd.DataFrame,
    y_train_log: np.ndarray,
    X_eval: pd.DataFrame,
    y_eval_log: np.ndarray,
    X_in_sample: pd.DataFrame | None,
    y_in_sample_log: np.ndarray | None,
    parallel_models: bool,
) -> List[Dict[str, Any]]:
    if not parallel_models or len(model_specs) <= 1:
        return [
            _evaluate_single_model_spec(
                spec=spec,
                linear_pipeline_builder=linear_pipeline_builder,
                X_train=X_train,
                y_train_log=y_train_log,
                X_eval=X_eval,
                y_eval_log=y_eval_log,
                X_in_sample=X_in_sample,
                y_in_sample_log=y_in_sample_log,
            )
            for spec in model_specs
        ]

    # Detect the cores actually available to this process. os.cpu_count() reports
    # the full machine and ignores SLURM/cgroup affinity, which can oversubscribe
    # threads and trigger OpenMP allocation failures. Prefer the CPU affinity mask
    # (honours --cpus-per-task) and allow an explicit override via env var.
    try:
        available_cpus = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        available_cpus = int(os.cpu_count() or 1)
    env_cap = os.environ.get("QUICK_TEST_MAX_WORKERS", "").strip()
    if env_cap:
        try:
            available_cpus = max(1, int(env_cap))
        except ValueError:
            pass
    max_workers = min(len(model_specs), max(1, int(available_cpus)))
    _log(
        "parallel model evaluation enabled",
        n_models=int(len(model_specs)),
        max_workers=int(max_workers),
        available_cpus=int(available_cpus),
    )
    rows: List[Dict[str, Any] | None] = [None] * len(model_specs)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                _evaluate_single_model_spec,
                spec=spec,
                linear_pipeline_builder=linear_pipeline_builder,
                X_train=X_train,
                y_train_log=y_train_log,
                X_eval=X_eval,
                y_eval_log=y_eval_log,
                X_in_sample=X_in_sample,
                y_in_sample_log=y_in_sample_log,
            ): idx
            for idx, spec in enumerate(model_specs)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            rows[idx] = future.result()
    return [row for row in rows if row is not None]


def run_quick_test(
    *,
    rho: float,
    rho_values: List[float] | None,
    eta: float | None,
    eta_values: List[float] | None,
    keep_values: List[float],
    rho_group: float,
    rho_group_values: List[float] | None,
    early_stopping_rounds: int | None,
    out_dir: str,
    data_path: str,
    sample_frac: float | None,
    seed: int,
    scatter_plot_max_samples: int | None = None,
    lgbm_hyperparameter_file: str | None = _DEFAULT_LGBM_HYPERPARAMETER_FILE,
    lgbm_config_key: str = _DEFAULT_LGBM_CONFIG_KEY,
    lgbm_n_jobs: int | None = None,
    lgbm_n_estimators: int | None = None,
    model_families: List[str] | None = None,
    skip_delete_analysis: bool = True,
    n_bootstrap_validation: int = 0,
    bootstrap_block_freq: str = "M",
    parallel_models: bool = False,
    assessment_year: int = 2024,
) -> Dict[str, str]:
    """Runs the quick test and writes the output CSV tables."""
    target_column = "meta_sale_price"
    date_column = "meta_sale_date"
    scatter_plot_max_samples = (
        None
        if scatter_plot_max_samples is None or int(scatter_plot_max_samples) <= 0
        else int(scatter_plot_max_samples)
    )
    _log(
        "quick test start",
        out_dir=out_dir,
        data_path=data_path,
        rho=rho,
        sample_frac=sample_frac,
        scatter_plot_max_samples=scatter_plot_max_samples,
    )

    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    with open("model_params.yaml", "r", encoding="utf-8") as f:
        model_params = yaml.safe_load(f)
    _log("configuration loaded")

    df_train_validate, df_test, df_assess, predictor_cols, categorical_cols = _load_and_split_data(
        data_path=data_path,
        params=params,
        target_column=target_column,
        date_column=date_column,
        sample_frac=sample_frac,
        sample_seed=seed,
        assessment_year=int(assessment_year),
    )
    _log("data load/split finished")
    predictor_cols = list(predictor_cols)
    categorical_cols = list(categorical_cols)

    # Pipeline builder for linear models (matches `main.py`).
    linear_pipeline_builder = lambda: build_model_pipeline(
        pred_vars=predictor_cols,
        cat_vars=categorical_cols,
        id_vars=params["model"]["predictor"]["id"],
    )

    # Train/eval matrices.
    X_tv = df_train_validate[predictor_cols].copy()
    y_tv_log = np.log(df_train_validate[target_column].to_numpy())

    if not skip_delete_analysis:
        # DELETE
        from sklearn.cluster import AgglomerativeClustering, KMeans
        from sklearn.metrics import normalized_mutual_info_score, silhouette_score
        from sklearn.preprocessing import StandardScaler

        def _cramers_v_from_crosstab(contingency: pd.DataFrame) -> float:
            observed = contingency.to_numpy(dtype=float)
            n_obs = float(observed.sum())
            if n_obs <= 1.0 or observed.shape[0] < 2 or observed.shape[1] < 2:
                return 0.0
            row_sums = observed.sum(axis=1, keepdims=True)
            col_sums = observed.sum(axis=0, keepdims=True)
            expected = (row_sums @ col_sums) / n_obs
            valid = expected > 0.0
            if not np.any(valid):
                return 0.0
            chi2 = float(np.sum(((observed - expected) ** 2 / np.where(valid, expected, 1.0))[valid]))
            phi2 = chi2 / n_obs
            n_rows, n_cols = observed.shape
            phi2_corr = max(0.0, phi2 - ((n_cols - 1.0) * (n_rows - 1.0)) / max(n_obs - 1.0, 1.0))
            n_rows_corr = n_rows - ((n_rows - 1.0) ** 2) / max(n_obs - 1.0, 1.0)
            n_cols_corr = n_cols - ((n_cols - 1.0) ** 2) / max(n_obs - 1.0, 1.0)
            denom = min(n_cols_corr - 1.0, n_rows_corr - 1.0)
            if denom <= 0.0:
                return 0.0
            return float(np.sqrt(phi2_corr / denom))

        def _eta_squared_by_cluster(values: np.ndarray, labels: np.ndarray) -> float:
            if values.size == 0 or np.all(labels == labels[0]):
                return 0.0
            overall_mean = float(np.mean(values))
            ss_total = float(np.sum((values - overall_mean) ** 2))
            if ss_total <= 0.0:
                return 0.0
            ss_between = 0.0
            for label in np.unique(labels):
                cluster_values = values[labels == label]
                if cluster_values.size == 0:
                    continue
                ss_between += float(cluster_values.size) * (float(np.mean(cluster_values)) - overall_mean) ** 2
            return float(np.clip(ss_between / ss_total, 0.0, 1.0))

        analysis_out_dir = Path(out_dir)
        analysis_out_dir.mkdir(parents=True, exist_ok=True)

        max_categorical_levels = 50
        cat_cols = [c for c in categorical_cols if c in X_tv.columns]
        low_card_cat_data: Dict[str, pd.Series] = {}
        cat_cardinality_rows: List[Dict[str, Any]] = []
        for col in cat_cols:
            series = _series_to_string_with_na(X_tv[col])
            n_levels = int(series.nunique(dropna=False))
            cat_cardinality_rows.append({"feature": col, "n_levels": n_levels})
            if 2 <= n_levels <= max_categorical_levels:
                low_card_cat_data[col] = series

        for raw_feature in ("meta_township_code", "char_class"):
            low_card_cat_data.pop(raw_feature, None)
            cat_cardinality_rows = [row for row in cat_cardinality_rows if row["feature"] != raw_feature]

        low_card_cat_cols = list(low_card_cat_data.keys())
        X_cat_raw = pd.DataFrame(low_card_cat_data, index=X_tv.index)
        cat_cardinality_df = pd.DataFrame(cat_cardinality_rows)
        if not cat_cardinality_df.empty:
            cat_cardinality_df = cat_cardinality_df.sort_values(["n_levels", "feature"]).reset_index(drop=True)
        else:
            cat_cardinality_df = pd.DataFrame(columns=["feature", "n_levels"])

        numeric_cols = [
            c for c in X_tv.columns
            if c not in cat_cols and pd.api.types.is_numeric_dtype(X_tv[c])
        ]
        X_num_raw = X_tv[numeric_cols].apply(pd.to_numeric, errors="coerce")
        X_num_imputed = X_num_raw.copy()
        if not X_num_imputed.empty:
            X_num_imputed = X_num_imputed.fillna(X_num_imputed.median())
            X_num_cluster = pd.DataFrame(
                StandardScaler().fit_transform(X_num_imputed),
                columns=numeric_cols,
                index=X_tv.index,
            )
        else:
            X_num_cluster = pd.DataFrame(index=X_tv.index)

        cat_blocks: List[pd.DataFrame] = []
        for col in low_card_cat_cols:
            block = pd.get_dummies(X_cat_raw[col], prefix=col, drop_first=False, dtype=float)
            block = block / np.sqrt(max(block.shape[1], 1))
            cat_blocks.append(block)
        X_cat_cluster = pd.concat(cat_blocks, axis=1) if cat_blocks else pd.DataFrame(index=X_tv.index)

        X_cluster = pd.concat([X_num_cluster, X_cat_cluster], axis=1)
        X_cluster = X_cluster.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
        valid_rows = X_cluster.notna().all(axis=1)
        X_cluster = X_cluster.loc[valid_rows].copy()
        X_num_imputed = X_num_imputed.loc[valid_rows].copy()
        X_num_cluster = X_num_cluster.loc[valid_rows].copy()
        X_cat_raw = X_cat_raw.loc[valid_rows].copy()

        if X_cluster.empty:
            print("Temporary cluster separability analysis could not run: no valid clustering features remained.")
            exit()

        analysis_sample_n = min(10000, X_cluster.shape[0])
        sampled_index = (
            X_cluster.sample(n=analysis_sample_n, random_state=seed).index
            if X_cluster.shape[0] > analysis_sample_n
            else X_cluster.index
        )
        X_cluster_sample = X_cluster.loc[sampled_index].copy()
        X_num_cluster_sample = X_num_cluster.loc[sampled_index].copy()
        X_cat_sample = X_cat_raw.loc[sampled_index].copy()

        print("=" * 90)
        print("TEMPORARY CLUSTER SEPARABILITY ANALYSIS")
        print("=" * 90)
        print(f"rows_available={X_cluster.shape[0]}")
        print(f"rows_sampled={X_cluster_sample.shape[0]}")
        print(f"numeric_features_used={len(numeric_cols)}")
        print(f"low_cardinality_categorical_features_used={len(low_card_cat_cols)}")
        print("Categorical cardinalities:")
        print(cat_cardinality_df.to_string(index=False))

        solution_rows: List[Dict[str, Any]] = []
        categorical_rows: List[Dict[str, Any]] = []
        category_level_rows: List[Dict[str, Any]] = []
        numeric_rows: List[Dict[str, Any]] = []
        cluster_settings = [
            ("kmeans", lambda k: KMeans(n_clusters=k, n_init=20, random_state=seed)),
            ("agglomerative_ward", lambda k: AgglomerativeClustering(n_clusters=k, linkage="ward")),
        ]

        for method_name, cluster_factory in cluster_settings:
            for n_clusters in (2, 3, 4, 5):
                if X_cluster_sample.shape[0] <= n_clusters:
                    continue
                clusterer = cluster_factory(n_clusters)
                labels = clusterer.fit_predict(X_cluster_sample)
                unique_labels, cluster_sizes = np.unique(labels, return_counts=True)
                if unique_labels.size < 2:
                    continue

                silhouette = np.nan
                if X_cluster_sample.shape[0] > unique_labels.size:
                    silhouette = float(
                        silhouette_score(
                            X_cluster_sample,
                            labels,
                            metric="euclidean",
                            sample_size=min(1000, X_cluster_sample.shape[0]),
                            random_state=seed,
                        )
                    )

                solution_rows.append(
                    {
                        "method": method_name,
                        "n_clusters": int(n_clusters),
                        "silhouette": silhouette,
                        "cluster_sizes": ",".join(str(int(v)) for v in cluster_sizes.tolist()),
                    }
                )

                if not X_cat_sample.empty:
                    for col in low_card_cat_cols:
                        feature_values = X_cat_sample[col]
                        contingency = pd.crosstab(feature_values, labels)
                        nmi = float(normalized_mutual_info_score(feature_values, labels))
                        cramers_v = _cramers_v_from_crosstab(contingency)
                        feature_purity = float(contingency.max(axis=1).sum() / contingency.to_numpy().sum())
                        purity_baseline = 1.0 / float(n_clusters)
                        purity_gain = float(
                            np.clip(
                                (feature_purity - purity_baseline) / max(1.0 - purity_baseline, 1e-12),
                                0.0,
                                1.0,
                            )
                        )
                        separation_score = float(np.mean([nmi, cramers_v, purity_gain]))
                        categorical_rows.append(
                            {
                                "method": method_name,
                                "n_clusters": int(n_clusters),
                                "feature": col,
                                "n_levels": int(feature_values.nunique()),
                                "nmi": nmi,
                                "cramers_v": cramers_v,
                                "purity": feature_purity,
                                "purity_gain": purity_gain,
                                "separation_score": separation_score,
                            }
                        )

                        for category_value, counts in contingency.iterrows():
                            counts = counts.astype(float)
                            category_n = float(counts.sum())
                            dominant_cluster = int(counts.idxmax())
                            dominant_share = float(counts.max() / max(category_n, 1.0))
                            dominance_gain = float(
                                np.clip(
                                    (dominant_share - purity_baseline) / max(1.0 - purity_baseline, 1e-12),
                                    0.0,
                                    1.0,
                                )
                            )
                            support_share = float(category_n / X_cluster_sample.shape[0])
                            category_level_rows.append(
                                {
                                    "method": method_name,
                                    "n_clusters": int(n_clusters),
                                    "feature": col,
                                    "category": str(category_value),
                                    "category_n": int(category_n),
                                    "support_share": support_share,
                                    "dominant_cluster": dominant_cluster,
                                    "dominant_share": dominant_share,
                                    "dominance_gain": dominance_gain,
                                    "support_weighted_split_score": float(dominance_gain * np.sqrt(support_share)),
                                }
                            )

                if not X_num_cluster_sample.empty:
                    for col in numeric_cols:
                        eta2 = _eta_squared_by_cluster(X_num_cluster_sample[col].to_numpy(), labels)
                        numeric_rows.append(
                            {
                                "method": method_name,
                                "n_clusters": int(n_clusters),
                                "feature": col,
                                "eta_squared": eta2,
                            }
                        )

        solution_df = pd.DataFrame(solution_rows)
        if not solution_df.empty:
            solution_df = solution_df.sort_values(["method", "n_clusters"]).reset_index(drop=True)
        else:
            solution_df = pd.DataFrame(columns=["method", "n_clusters", "silhouette", "cluster_sizes"])
        categorical_detail_df = pd.DataFrame(categorical_rows)
        category_detail_df = pd.DataFrame(category_level_rows)
        numeric_detail_df = pd.DataFrame(numeric_rows)

        if categorical_detail_df.empty:
            categorical_rank_df = pd.DataFrame(
                columns=["feature", "n_levels", "avg_nmi", "avg_cramers_v", "avg_purity_gain", "avg_separation_score"]
            )
        else:
            categorical_rank_df = (
                categorical_detail_df.groupby("feature", as_index=False)
                .agg(
                    n_levels=("n_levels", "first"),
                    avg_nmi=("nmi", "mean"),
                    avg_cramers_v=("cramers_v", "mean"),
                    avg_purity_gain=("purity_gain", "mean"),
                    avg_separation_score=("separation_score", "mean"),
                    best_separation_score=("separation_score", "max"),
                )
                .sort_values(["avg_separation_score", "best_separation_score", "avg_nmi"], ascending=False)
                .reset_index(drop=True)
            )

        if category_detail_df.empty:
            category_rank_df = pd.DataFrame(
                columns=[
                    "feature",
                    "category",
                    "avg_dominance_gain",
                    "avg_dominant_share",
                    "support_share",
                    "avg_support_weighted_split_score",
                ]
            )
        else:
            category_rank_df = (
                category_detail_df.groupby(["feature", "category"], as_index=False)
                .agg(
                    support_share=("support_share", "mean"),
                    avg_dominant_share=("dominant_share", "mean"),
                    avg_dominance_gain=("dominance_gain", "mean"),
                    avg_support_weighted_split_score=("support_weighted_split_score", "mean"),
                    best_support_weighted_split_score=("support_weighted_split_score", "max"),
                    avg_category_n=("category_n", "mean"),
                )
                .sort_values(
                    ["avg_support_weighted_split_score", "avg_dominance_gain", "support_share"],
                    ascending=False,
                )
                .reset_index(drop=True)
            )

        if numeric_detail_df.empty:
            numeric_rank_df = pd.DataFrame(columns=["feature", "avg_eta_squared", "best_eta_squared"])
        else:
            numeric_rank_df = (
                numeric_detail_df.groupby("feature", as_index=False)
                .agg(
                    avg_eta_squared=("eta_squared", "mean"),
                    best_eta_squared=("eta_squared", "max"),
                )
                .sort_values(["avg_eta_squared", "best_eta_squared"], ascending=False)
                .reset_index(drop=True)
            )

        solution_path = analysis_out_dir / "temp_cluster_solution_summary.csv"
        categorical_path = analysis_out_dir / "temp_cluster_categorical_feature_ranking.csv"
        categorical_detail_path = analysis_out_dir / "temp_cluster_categorical_feature_detail.csv"
        category_path = analysis_out_dir / "temp_cluster_category_ranking.csv"
        category_detail_path = analysis_out_dir / "temp_cluster_category_detail.csv"
        numeric_path = analysis_out_dir / "temp_cluster_numeric_feature_ranking.csv"
        numeric_detail_path = analysis_out_dir / "temp_cluster_numeric_feature_detail.csv"

        solution_df.to_csv(solution_path, index=False)
        categorical_rank_df.to_csv(categorical_path, index=False)
        categorical_detail_df.to_csv(categorical_detail_path, index=False)
        category_rank_df.to_csv(category_path, index=False)
        category_detail_df.to_csv(category_detail_path, index=False)
        numeric_rank_df.to_csv(numeric_path, index=False)
        numeric_detail_df.to_csv(numeric_detail_path, index=False)

        print("\nClustering solutions:")
        print(solution_df.to_string(index=False))
        print("\nCategorical feature ranking:")
        print(categorical_rank_df.head(25).to_string(index=False))
        print("\nMost separating category values:")
        print(category_rank_df.head(25).to_string(index=False))
        print("\nNumeric feature separation context:")
        print(numeric_rank_df.head(25).to_string(index=False))
        print("\nTemporary analysis files written to:")
        print(solution_path)
        print(categorical_path)
        print(categorical_detail_path)
        print(category_path)
        print(category_detail_path)
        print(numeric_path)
        print(numeric_detail_path)

        exit()
        # DELETE

    X_test = df_test[predictor_cols].copy()
    y_test_log = np.log(df_test[target_column].to_numpy())

    X_assess = df_assess[predictor_cols].copy()
    y_assess_log = np.log(df_assess[target_column].to_numpy()) if not df_assess.empty else np.array([], dtype=float)

    # Categorical handling (matches `main.py`).
    cat_cols = [c for c in categorical_cols if c in X_tv.columns]
    for c in cat_cols:
        X_tv[c] = X_tv[c].astype("category")
        X_test[c] = X_test[c].astype("category")
        if not df_assess.empty:
            X_assess[c] = X_assess[c].astype("category")

    # Model parameterization (baseline LGBM defaults).
    lgbm_params = _build_lgbm_params_from_files(model_params=model_params, ccao_params=params, seed=seed)
    if lgbm_hyperparameter_file is not None and str(lgbm_hyperparameter_file).strip():
        lgbm_params = _load_lgbm_params_from_hyperparameter_file(
            str(lgbm_hyperparameter_file),
            str(lgbm_config_key),
        )
        _log(
            "LGBM parameters loaded from hyperparameter file",
            hyperparameter_file=str(lgbm_hyperparameter_file),
            config_key=str(lgbm_config_key),
        )
    if lgbm_n_jobs is not None:
        lgbm_params["n_jobs"] = int(lgbm_n_jobs)
        _log("LGBM n_jobs override applied", n_jobs=int(lgbm_n_jobs))
    if lgbm_n_estimators is not None:
        lgbm_params["n_estimators"] = int(lgbm_n_estimators)
        _log("LGBM n_estimators override applied", n_estimators=int(lgbm_n_estimators))
    rho_sweep = _round_rho_values([float(r) for r in (rho_values if rho_values is not None else [rho])])
    eta_sweep = [float(v) for v in (eta_values if eta_values is not None else [rho if eta is None else eta])]
    models = _build_quick_test_models(
        rho_values=rho_sweep,
        eta_values=eta_sweep,
        keep_values=keep_values,
        lgbm_params=lgbm_params,
        early_stopping_rounds=early_stopping_rounds,
        model_families=model_families,
    )
    _log("model specs built", n_models=int(len(models)), models=",".join(str(m) for m in (model_families or [])))
    model_ratio_modes = {str(spec["model_name"]): str(spec["ratio_mode"]) for spec in models}
    model_specs_by_name = {str(spec["model_name"]): spec for spec in models}

    # --- Evaluate on TEST (train on df_train_validate only; strict out-of-time).
    _log("starting test evaluation", n_models=int(len(models)))
    test_rows = []
    test_pred_logs: Dict[str, np.ndarray] = {}
    train_test_rows = []
    train_test_pred_logs: Dict[str, np.ndarray] = {}
    test_eval_rows = _evaluate_model_specs(
        model_specs=models,
        linear_pipeline_builder=linear_pipeline_builder,
        X_train=X_tv,
        y_train_log=y_tv_log,
        X_eval=X_test,
        y_eval_log=y_test_log,
        X_in_sample=X_tv,
        y_in_sample_log=y_tv_log,
        parallel_models=parallel_models,
    )
    for spec, row in zip(models, test_eval_rows):
        row_meta = {
            "model_family": str(spec["model_family"]),
            "ratio_mode": str(spec["ratio_mode"]),
            "rho": spec["rho"],
            "rho_group": spec.get("rho_group", np.nan),
            "eta": spec.get("eta", np.nan),
            "keep": spec.get("keep", np.nan),
        }
        row.update(row_meta)
        test_pred_logs[str(spec["model_name"])] = np.asarray(row.pop("_y_pred_eval_log"), dtype=float).reshape(-1)
        in_sample_metrics = dict(row.pop("_in_sample_metrics", {}))
        if "_y_pred_in_sample_log" in row:
            train_test_pred_logs[str(spec["model_name"])] = np.asarray(
                row.pop("_y_pred_in_sample_log"),
                dtype=float,
            ).reshape(-1)
        if in_sample_metrics:
            train_row = {"model_name": str(spec["model_name"]), **in_sample_metrics, **row_meta}
            train_test_rows.append(train_row)
        test_rows.append(row)
    test_df = pd.DataFrame(test_rows)
    train_test_df = pd.DataFrame(train_test_rows)
    _log("test evaluation finished", rows=int(test_df.shape[0]))

    # --- Evaluate on ASSESS (train on ALL pre-2024 sales, i.e., train_validate + test).
    assess_df = pd.DataFrame()
    assess_pred_logs: Dict[str, np.ndarray] = {}
    train_assess_df = pd.DataFrame()
    train_assess_pred_logs: Dict[str, np.ndarray] = {}
    if not df_assess.empty:
        _log("starting assessment evaluation", n_models=int(len(models)))
        df_pre2024 = pd.concat([df_train_validate, df_test], ignore_index=True)
        X_pre = df_pre2024[predictor_cols].copy()
        y_pre_log = np.log(df_pre2024[target_column].to_numpy())
        for c in cat_cols:
            X_pre[c] = X_pre[c].astype("category")

        assess_rows = []
        assess_train_rows = []
        assess_eval_rows = _evaluate_model_specs(
            model_specs=models,
            linear_pipeline_builder=linear_pipeline_builder,
            X_train=X_pre,
            y_train_log=y_pre_log,
            X_eval=X_assess,
            y_eval_log=y_assess_log,
            X_in_sample=X_pre,
            y_in_sample_log=y_pre_log,
            parallel_models=parallel_models,
        )
        for spec, row in zip(models, assess_eval_rows):
            row_meta = {
                "model_family": str(spec["model_family"]),
                "ratio_mode": str(spec["ratio_mode"]),
                "rho": spec["rho"],
                "rho_group": spec.get("rho_group", np.nan),
                "eta": spec.get("eta", np.nan),
                "keep": spec.get("keep", np.nan),
            }
            row.update(row_meta)
            assess_pred_logs[str(spec["model_name"])] = np.asarray(row.pop("_y_pred_eval_log"), dtype=float).reshape(-1)
            in_sample_metrics = dict(row.pop("_in_sample_metrics", {}))
            if "_y_pred_in_sample_log" in row:
                train_assess_pred_logs[str(spec["model_name"])] = np.asarray(
                    row.pop("_y_pred_in_sample_log"),
                    dtype=float,
                ).reshape(-1)
            if in_sample_metrics:
                train_row = {"model_name": str(spec["model_name"]), **in_sample_metrics, **row_meta}
                assess_train_rows.append(train_row)
            assess_rows.append(row)
        assess_df = pd.DataFrame(assess_rows)
        train_assess_df = pd.DataFrame(assess_train_rows)
        _log("assessment evaluation finished", rows=int(assess_df.shape[0]))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    test_path = out / "quick_test_metrics_test.csv"
    assess_path = out / "quick_test_metrics_assess.csv"
    train_test_path = out / "quick_test_metrics_train_for_test.csv"
    train_assess_path = out / "quick_test_metrics_train_for_assess.csv"
    bootstrap_val_path = out / "quick_test_metrics_validation_bootstrap_avg.csv"
    test_report_path = out / "quick_test_report_test.txt"
    assess_report_path = out / "quick_test_report_assess.txt"
    train_test_report_path = out / "quick_test_report_train_for_test.txt"
    train_assess_report_path = out / "quick_test_report_train_for_assess.txt"
    test_df.to_csv(test_path, index=False)
    if not train_test_df.empty:
        train_test_df.to_csv(train_test_path, index=False)
    if not assess_df.empty:
        assess_df.to_csv(assess_path, index=False)
    if not train_assess_df.empty:
        train_assess_df.to_csv(train_assess_path, index=False)

    # Persist the unpenalized baseline log-space predictions so downstream theory
    # tooling (e.g. scripts/theory_informed_rho_range.py) can recycle them instead
    # of refitting the baseline LGBM. This is cheap (two small parquet files of
    # already-computed arrays) and adds no model fitting.
    def _save_baseline_predictions(pred_logs, y_true_log, split_label):
        if y_true_log is None:
            return
        y_arr = np.asarray(y_true_log, dtype=float).reshape(-1)
        cols = {"y_log": y_arr}
        for model_name, key in (("LGBMRegressor", "f0_log"), ("LinearRegression", "linear_log")):
            if model_name in pred_logs:
                p = np.asarray(pred_logs[model_name], dtype=float).reshape(-1)
                if p.shape[0] == y_arr.shape[0]:
                    cols[key] = p
        if "f0_log" not in cols:
            return
        path = out / f"baseline_predictions_{split_label}.parquet"
        try:
            pd.DataFrame(cols).to_parquet(path, index=False)
            _log("baseline predictions saved for theory recycling", path=str(path), rows=int(y_arr.shape[0]))
        except Exception as exc:  # pragma: no cover - persistence must not break the run
            _log("failed to save baseline predictions", split=split_label, error=repr(exc))

    _save_baseline_predictions(test_pred_logs, y_test_log, "test")
    if not assess_df.empty:
        _save_baseline_predictions(assess_pred_logs, y_assess_log, "assess")

    _write_split_report(
        split_label="Test",
        results_df=test_df,
        y_true_log=y_test_log,
        pred_logs=test_pred_logs,
        model_specs_by_name=model_specs_by_name,
        out_path=test_report_path,
    )
    if not train_test_df.empty:
        _write_split_report(
            split_label="TrainInSample-TestFit",
            results_df=train_test_df,
            y_true_log=y_tv_log,
            pred_logs=train_test_pred_logs,
            model_specs_by_name=model_specs_by_name,
            out_path=train_test_report_path,
        )
    if not assess_df.empty:
        _write_split_report(
            split_label="Assessment",
            results_df=assess_df,
            y_true_log=y_assess_log,
            pred_logs=assess_pred_logs,
            model_specs_by_name=model_specs_by_name,
            out_path=assess_report_path,
        )
    if not train_assess_df.empty:
        _write_split_report(
            split_label="TrainInSample-AssessFit",
            results_df=train_assess_df,
            y_true_log=y_pre_log,
            pred_logs=train_assess_pred_logs,
            model_specs_by_name=model_specs_by_name,
            out_path=train_assess_report_path,
        )

    # --- Small bootstrap summary over the quick-test split (validation-like diagnostic).
    bootstrap_rows: List[Dict[str, Any]] = []
    n_bs = max(0, int(n_bootstrap_validation))
    if n_bs > 0 and y_test_log.size > 1 and test_pred_logs:
        _log("starting bootstrap summary", n_bootstrap=int(n_bs), n_models=int(len(test_pred_logs)))
        bs_indices = _build_time_block_bootstrap_indices(
            val_dates=pd.to_datetime(df_test[date_column]),
            n_bootstrap=n_bs,
            block_freq=str(bootstrap_block_freq),
            rng_seed=int(seed),
        )
        for model_name, y_pred_log in test_pred_logs.items():
            per_bs: List[Dict[str, Any]] = []
            metric_ratio_mode = str(model_ratio_modes.get(str(model_name), "diff"))
            for sample_idx in bs_indices:
                idx = np.asarray(sample_idx, dtype=int)
                if idx.size < 2:
                    continue
                m = _compute_quick_test_metrics(
                    y_true_log=y_test_log[idx],
                    y_pred_log=y_pred_log[idx],
                    y_train_log=y_tv_log,
                    ratio_mode=metric_ratio_mode,
                )
                per_bs.append(m)
            if not per_bs:
                continue
            bs_df = pd.DataFrame(per_bs)
            row: Dict[str, Any] = {
                "model_name": str(model_name),
                "n_bootstrap": int(len(per_bs)),
                "bootstrap_block_freq": str(bootstrap_block_freq),
            }
            if not test_df.empty:
                match = test_df.loc[test_df["model_name"] == str(model_name), ["model_family", "ratio_mode", "rho", "rho_group", "eta", "keep"]]
                if not match.empty:
                    row["model_family"] = str(match.iloc[0]["model_family"])
                    row["ratio_mode"] = str(match.iloc[0]["ratio_mode"])
                    row["rho"] = match.iloc[0]["rho"]
                    row["rho_group"] = match.iloc[0]["rho_group"]
                    row["eta"] = match.iloc[0]["eta"]
                    row["keep"] = match.iloc[0]["keep"]
            for c in bs_df.columns:
                s = pd.to_numeric(bs_df[c], errors="coerce")
                v = s.to_numpy(dtype=float)
                finite_v = v[np.isfinite(v)]
                if finite_v.size > 0:
                    row[c] = float(np.nanmean(finite_v))
                    if finite_v.size > 1:
                        row[f"{c}_std"] = float(np.nanstd(finite_v, ddof=1))
            bootstrap_rows.append(row)
    bootstrap_df = pd.DataFrame(bootstrap_rows)
    bootstrap_df.to_csv(bootstrap_val_path, index=False)
    _log("bootstrap summary written", rows=int(bootstrap_df.shape[0]))

    plots_dir = out / "plots"
    rho_plots_dir = plots_dir / "rho_evolution"
    grouped_rho_plots_dir = rho_plots_dir / "grouped"
    ratio_plots_dir = plots_dir / "ratio_vs_logprice"
    residual_plots_dir = plots_dir / "residual_vs_logprice"
    ratio_pred_plots_dir = plots_dir / "ratio_vs_logprediction"
    residual_pred_plots_dir = plots_dir / "residual_vs_logprediction"
    tradeoff_plots_dir = plots_dir / "tradeoff"
    _log("writing plots", plots_dir=str(plots_dir))
    _write_rho_evolution_plot(
        bootstrap_df,
        split_label="Validation bootstrap average",
        out_path=rho_plots_dir / "quick_test_rho_evolution_validation.pdf",
    )
    _write_grouped_rho_evolution_plots(
        bootstrap_df,
        split_label="Validation bootstrap average",
        out_dir=grouped_rho_plots_dir,
    )
    _write_tradeoff_plots(
        bootstrap_df,
        split_label="Validation bootstrap average",
        out_dir=tradeoff_plots_dir,
    )
    _write_rho_evolution_plot(
        test_df,
        split_label="Test",
        out_path=rho_plots_dir / "quick_test_rho_evolution_test.pdf",
    )
    _write_grouped_rho_evolution_plots(
        test_df,
        split_label="Test",
        out_dir=grouped_rho_plots_dir,
    )
    _write_tradeoff_plots(
        test_df,
        split_label="Test",
        out_dir=tradeoff_plots_dir,
    )
    if not train_test_df.empty:
        _write_rho_evolution_plot(
            train_test_df,
            split_label="TrainInSample-TestFit",
            out_path=rho_plots_dir / "quick_test_rho_evolution_train_test_fit.pdf",
        )
        _write_tradeoff_plots(
            train_test_df,
            split_label="TrainInSample-TestFit",
            out_dir=tradeoff_plots_dir,
        )
    if not assess_df.empty:
        _write_rho_evolution_plot(
            assess_df,
            split_label="Assessment",
            out_path=rho_plots_dir / "quick_test_rho_evolution_assess.pdf",
        )
        _write_grouped_rho_evolution_plots(
            assess_df,
            split_label="Assessment",
            out_dir=grouped_rho_plots_dir,
        )
        _write_tradeoff_plots(
            assess_df,
            split_label="Assessment",
            out_dir=tradeoff_plots_dir,
        )
    if not train_assess_df.empty:
        _write_rho_evolution_plot(
            train_assess_df,
            split_label="TrainInSample-AssessFit",
            out_path=rho_plots_dir / "quick_test_rho_evolution_train_assess_fit.pdf",
        )
        _write_tradeoff_plots(
            train_assess_df,
            split_label="TrainInSample-AssessFit",
            out_dir=tradeoff_plots_dir,
        )

    scatter_plot_kwargs = {
        "scatter_plot_max_samples": scatter_plot_max_samples,
        "scatter_plot_sample_seed": int(seed),
    }
    _write_ratio_vs_logprice_plots(
        split_label="Test",
        results_df=test_df,
        y_true_log=y_test_log,
        pred_logs=test_pred_logs,
        out_dir=ratio_plots_dir,
        grouped_feature_labels=df_test[_META_TOWNSHIP_TRIAD_COL].astype(str).to_numpy()
        if _META_TOWNSHIP_TRIAD_COL in df_test.columns
        else None,
        grouped_feature_name=_META_TOWNSHIP_TRIAD_COL,
        **scatter_plot_kwargs,
    )
    _write_ratio_vs_logprediction_plots(
        split_label="Test",
        results_df=test_df,
        y_true_log=y_test_log,
        pred_logs=test_pred_logs,
        out_dir=ratio_pred_plots_dir,
        grouped_feature_labels=df_test[_META_TOWNSHIP_TRIAD_COL].astype(str).to_numpy()
        if _META_TOWNSHIP_TRIAD_COL in df_test.columns
        else None,
        grouped_feature_name=_META_TOWNSHIP_TRIAD_COL,
        **scatter_plot_kwargs,
    )
    _write_residual_vs_logprice_plots(
        split_label="Test",
        results_df=test_df,
        y_true_log=y_test_log,
        pred_logs=test_pred_logs,
        out_dir=residual_plots_dir,
        grouped_feature_labels=df_test[_META_TOWNSHIP_TRIAD_COL].astype(str).to_numpy()
        if _META_TOWNSHIP_TRIAD_COL in df_test.columns
        else None,
        grouped_feature_name=_META_TOWNSHIP_TRIAD_COL,
        **scatter_plot_kwargs,
    )
    _write_residual_vs_logprediction_plots(
        split_label="Test",
        results_df=test_df,
        y_true_log=y_test_log,
        pred_logs=test_pred_logs,
        out_dir=residual_pred_plots_dir,
        grouped_feature_labels=df_test[_META_TOWNSHIP_TRIAD_COL].astype(str).to_numpy()
        if _META_TOWNSHIP_TRIAD_COL in df_test.columns
        else None,
        grouped_feature_name=_META_TOWNSHIP_TRIAD_COL,
        **scatter_plot_kwargs,
    )
    if not train_test_df.empty:
        _write_ratio_vs_logprice_plots(
            split_label="TrainInSample-TestFit",
            results_df=train_test_df,
            y_true_log=y_tv_log,
            pred_logs=train_test_pred_logs,
            out_dir=ratio_plots_dir,
            grouped_feature_labels=df_train_validate[_META_TOWNSHIP_TRIAD_COL].astype(str).to_numpy()
            if _META_TOWNSHIP_TRIAD_COL in df_train_validate.columns
            else None,
            grouped_feature_name=_META_TOWNSHIP_TRIAD_COL,
        **scatter_plot_kwargs,
        )
        _write_ratio_vs_logprediction_plots(
            split_label="TrainInSample-TestFit",
            results_df=train_test_df,
            y_true_log=y_tv_log,
            pred_logs=train_test_pred_logs,
            out_dir=ratio_pred_plots_dir,
            grouped_feature_labels=df_train_validate[_META_TOWNSHIP_TRIAD_COL].astype(str).to_numpy()
            if _META_TOWNSHIP_TRIAD_COL in df_train_validate.columns
            else None,
            grouped_feature_name=_META_TOWNSHIP_TRIAD_COL,
        **scatter_plot_kwargs,
        )
        _write_residual_vs_logprice_plots(
            split_label="TrainInSample-TestFit",
            results_df=train_test_df,
            y_true_log=y_tv_log,
            pred_logs=train_test_pred_logs,
            out_dir=residual_plots_dir,
            grouped_feature_labels=df_train_validate[_META_TOWNSHIP_TRIAD_COL].astype(str).to_numpy()
            if _META_TOWNSHIP_TRIAD_COL in df_train_validate.columns
            else None,
            grouped_feature_name=_META_TOWNSHIP_TRIAD_COL,
        **scatter_plot_kwargs,
        )
        _write_residual_vs_logprediction_plots(
            split_label="TrainInSample-TestFit",
            results_df=train_test_df,
            y_true_log=y_tv_log,
            pred_logs=train_test_pred_logs,
            out_dir=residual_pred_plots_dir,
            grouped_feature_labels=df_train_validate[_META_TOWNSHIP_TRIAD_COL].astype(str).to_numpy()
            if _META_TOWNSHIP_TRIAD_COL in df_train_validate.columns
            else None,
            grouped_feature_name=_META_TOWNSHIP_TRIAD_COL,
        **scatter_plot_kwargs,
        )
    if not assess_df.empty:
        _write_ratio_vs_logprice_plots(
            split_label="Assessment",
            results_df=assess_df,
            y_true_log=y_assess_log,
            pred_logs=assess_pred_logs,
            out_dir=ratio_plots_dir,
            grouped_feature_labels=df_assess[_META_TOWNSHIP_TRIAD_COL].astype(str).to_numpy()
            if _META_TOWNSHIP_TRIAD_COL in df_assess.columns
            else None,
            grouped_feature_name=_META_TOWNSHIP_TRIAD_COL,
        **scatter_plot_kwargs,
        )
        _write_ratio_vs_logprediction_plots(
            split_label="Assessment",
            results_df=assess_df,
            y_true_log=y_assess_log,
            pred_logs=assess_pred_logs,
            out_dir=ratio_pred_plots_dir,
            grouped_feature_labels=df_assess[_META_TOWNSHIP_TRIAD_COL].astype(str).to_numpy()
            if _META_TOWNSHIP_TRIAD_COL in df_assess.columns
            else None,
            grouped_feature_name=_META_TOWNSHIP_TRIAD_COL,
        **scatter_plot_kwargs,
        )
        _write_residual_vs_logprice_plots(
            split_label="Assessment",
            results_df=assess_df,
            y_true_log=y_assess_log,
            pred_logs=assess_pred_logs,
            out_dir=residual_plots_dir,
            grouped_feature_labels=df_assess[_META_TOWNSHIP_TRIAD_COL].astype(str).to_numpy()
            if _META_TOWNSHIP_TRIAD_COL in df_assess.columns
            else None,
            grouped_feature_name=_META_TOWNSHIP_TRIAD_COL,
        **scatter_plot_kwargs,
        )
        _write_residual_vs_logprediction_plots(
            split_label="Assessment",
            results_df=assess_df,
            y_true_log=y_assess_log,
            pred_logs=assess_pred_logs,
            out_dir=residual_pred_plots_dir,
            grouped_feature_labels=df_assess[_META_TOWNSHIP_TRIAD_COL].astype(str).to_numpy()
            if _META_TOWNSHIP_TRIAD_COL in df_assess.columns
            else None,
            grouped_feature_name=_META_TOWNSHIP_TRIAD_COL,
        **scatter_plot_kwargs,
        )
    if not train_assess_df.empty:
        _write_ratio_vs_logprice_plots(
            split_label="TrainInSample-AssessFit",
            results_df=train_assess_df,
            y_true_log=y_pre_log,
            pred_logs=train_assess_pred_logs,
            out_dir=ratio_plots_dir,
            grouped_feature_labels=df_pre2024[_META_TOWNSHIP_TRIAD_COL].astype(str).to_numpy()
            if _META_TOWNSHIP_TRIAD_COL in df_pre2024.columns
            else None,
            grouped_feature_name=_META_TOWNSHIP_TRIAD_COL,
        **scatter_plot_kwargs,
        )
        _write_ratio_vs_logprediction_plots(
            split_label="TrainInSample-AssessFit",
            results_df=train_assess_df,
            y_true_log=y_pre_log,
            pred_logs=train_assess_pred_logs,
            out_dir=ratio_pred_plots_dir,
            grouped_feature_labels=df_pre2024[_META_TOWNSHIP_TRIAD_COL].astype(str).to_numpy()
            if _META_TOWNSHIP_TRIAD_COL in df_pre2024.columns
            else None,
            grouped_feature_name=_META_TOWNSHIP_TRIAD_COL,
        **scatter_plot_kwargs,
        )
        _write_residual_vs_logprice_plots(
            split_label="TrainInSample-AssessFit",
            results_df=train_assess_df,
            y_true_log=y_pre_log,
            pred_logs=train_assess_pred_logs,
            out_dir=residual_plots_dir,
            grouped_feature_labels=df_pre2024[_META_TOWNSHIP_TRIAD_COL].astype(str).to_numpy()
            if _META_TOWNSHIP_TRIAD_COL in df_pre2024.columns
            else None,
            grouped_feature_name=_META_TOWNSHIP_TRIAD_COL,
        **scatter_plot_kwargs,
        )
        _write_residual_vs_logprediction_plots(
            split_label="TrainInSample-AssessFit",
            results_df=train_assess_df,
            y_true_log=y_pre_log,
            pred_logs=train_assess_pred_logs,
            out_dir=residual_pred_plots_dir,
            grouped_feature_labels=df_pre2024[_META_TOWNSHIP_TRIAD_COL].astype(str).to_numpy()
            if _META_TOWNSHIP_TRIAD_COL in df_pre2024.columns
            else None,
            grouped_feature_name=_META_TOWNSHIP_TRIAD_COL,
        **scatter_plot_kwargs,
        )
    _log("quick test finished", plots_dir=str(plots_dir))

    return {
        "test_csv": str(test_path),
        "assess_csv": str(assess_path),
        "train_test_csv": str(train_test_path),
        "train_assess_csv": str(train_assess_path),
        "bootstrap_validation_avg_csv": str(bootstrap_val_path),
        "test_report_txt": str(test_report_path),
        "assess_report_txt": str(assess_report_path),
        "train_test_report_txt": str(train_test_report_path),
        "train_assess_report_txt": str(train_assess_report_path),
        "plots_dir": str(plots_dir),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Quick test: covariance-regularized LightGBM models in diff mode on test (~2023) + assessment (2024).")
    p.add_argument("--rho", type=float, default=1.0, help="Rho used for LGBCovPenalty and LGBSmoothPenalty.")
    p.add_argument(
        "--eta",
        type=float,
        default=None,
        help="Eta used for LGBSmoothPenaltyGroupCVaR when --eta-range is omitted. Defaults to --rho.",
    )
    p.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=10,
        help="Training-loss patience for the LightGBM fairness models. Use 0 to disable.",
    )
    p.add_argument(
        "--rho-range",
        type=str,
        default="",
        help="Optional comma-separated rho range for LGBCovPenalty and LGBSmoothPenalty in the form min,max. If omitted, uses --rho.",
    )
    p.add_argument("--rho-count", type=int, default=5, help="Number of rho values to generate when --rho-range is provided.")
    p.add_argument(
        "--rho-extra",
        type=str,
        default="",
        help="Optional comma-separated rho values appended to the generated sweep (e.g. a recommended operating point like 3.01). Merged, de-duplicated and sorted into the grid.",
    )
    p.add_argument(
        "--rho-scale",
        type=str,
        default="log",
        help="Scale for rho sweep when --rho-range is provided. Allowed: linear, log, geom.",
    )
    p.add_argument(
        "--eta-range",
        type=str,
        default="",
        help="Optional comma-separated eta range for LGBSmoothPenaltyGroupCVaR in the form min,max. If omitted, uses --eta or --rho.",
    )
    p.add_argument("--eta-count", type=int, default=5, help="Number of eta values to generate when --eta-range is provided.")
    p.add_argument(
        "--eta-scale",
        type=str,
        default="log",
        help="Scale for eta sweep when --eta-range is provided. Allowed: linear, log, geom.",
    )
    p.add_argument(
        "--keep-values",
        type=str,
        default="0.5,0.7,0.9",
        help="Comma-separated keep values for LGBSmoothPenaltyGroupCVaR, e.g. 0.5,0.7,0.9.",
    )
    p.add_argument(
        "--lgbm-hyperparameter-file",
        type=str,
        default=_DEFAULT_LGBM_HYPERPARAMETER_FILE,
        help="YAML file containing reusable LGBM params. Use an empty string to fall back to model_params.yaml/params.yaml.",
    )
    p.add_argument(
        "--lgbm-config-key",
        type=str,
        default=_DEFAULT_LGBM_CONFIG_KEY,
        help="Key under lgbm_baselines in --lgbm-hyperparameter-file, e.g. test_best_r2 or cv_best_r2.",
    )
    p.add_argument(
        "--lgbm-n-jobs",
        type=int,
        default=None,
        help="Override n_jobs for each LGBM-based estimator. Default keeps the value from the loaded config.",
    )
    p.add_argument(
        "--lgbm-n-estimators",
        type=int,
        default=None,
        help="Override n_estimators for each LGBM-based estimator. Default keeps the value from the loaded config.",
    )
    p.add_argument(
        "--models",
        type=str,
        default="linear,lgbm,cov,smooth",
        help="Comma-separated models to run: linear,linear_cov,lgbm,cov,smooth. Alias surr maps to smooth. Use all for the default set.",
    )
    p.add_argument("--out-dir", type=str, default="./output/quick_test", help="Directory to write CSV outputs.")
    p.add_argument(
        "--data-path",
        type=str,
        default="./data/CCAO/2025/training_data.parquet",
        help="Path to training_data.parquet (same file used by main.py).",
    )
    p.add_argument(
        "--skip-delete-analysis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip the temporary analysis block between the DELETE flags. Use --no-skip-delete-analysis to run it.",
    )
    p.add_argument(
        "--parallel-models",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run the independent model fits in parallel. Disabled by default to preserve the current workflow.",
    )
    p.add_argument("--sample-frac", type=float, default=None, help="Optional down-sampling fraction in (0,1].")
    p.add_argument("--scatter-plot-max-samples", type=int, default=None, help="Maximum randomly sampled points per scatter plot. Use 0 or omit to disable the cap.")
    p.add_argument("--seed", type=int, default=4050, help="Random seed (mirrors main.py default).")
    p.add_argument("--n-bootstrap-validation", type=int, default=0, help="Small number of bootstrap resamples for quick validation-like summary.")
    p.add_argument("--bootstrap-block-freq", type=str, default="M", help="Time block frequency for bootstrap resampling (e.g., M, W, Q).")
    p.add_argument(
        "--assessment-year",
        type=int,
        default=2024,
        help="Calendar year used as the held-out assessment block. Sales in this year form the assessment set; sales before it form train/validate + test (the last 1-split_prop fraction is the test split).",
    )
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    out = run_quick_test(
        rho=float(args.rho),
        rho_values=_build_rho_sweep(
            float(args.rho),
            str(args.rho_range),
            int(args.rho_count),
            str(args.rho_scale),
            str(args.rho_extra),
        ),
        eta=(None if args.eta is None else float(args.eta)),
        eta_values=(
            _build_rho_sweep(
                float(args.eta if args.eta is not None else args.rho),
                str(args.eta_range),
                int(args.eta_count),
                str(args.eta_scale),
            )
            if str(args.eta_range).strip()
            else None
        ),
        keep_values=_parse_float_list(str(args.keep_values)),
        rho_group=1.0,
        rho_group_values=None,
        early_stopping_rounds=(None if int(args.early_stopping_rounds) <= 0 else int(args.early_stopping_rounds)),
        out_dir=str(args.out_dir),
        data_path=str(args.data_path),
        sample_frac=(None if args.sample_frac is None else float(args.sample_frac)),
        seed=int(args.seed),
        scatter_plot_max_samples=(
            None if args.scatter_plot_max_samples is None else int(args.scatter_plot_max_samples)
        ),
        lgbm_hyperparameter_file=(
            None
            if str(args.lgbm_hyperparameter_file).strip() == ""
            else str(args.lgbm_hyperparameter_file)
        ),
        lgbm_config_key=str(args.lgbm_config_key),
        lgbm_n_jobs=(None if args.lgbm_n_jobs is None else int(args.lgbm_n_jobs)),
        lgbm_n_estimators=(None if args.lgbm_n_estimators is None else int(args.lgbm_n_estimators)),
        model_families=_parse_model_families(str(args.models)),
        skip_delete_analysis=bool(args.skip_delete_analysis),
        n_bootstrap_validation=int(args.n_bootstrap_validation),
        bootstrap_block_freq=str(args.bootstrap_block_freq),
        parallel_models=bool(args.parallel_models),
        assessment_year=int(args.assessment_year),
    )
    print("=" * 90)
    print("QUICK TEST COMPLETED")
    print("=" * 90)
    print(f"test_csv={out['test_csv']}")
    print(f"assess_csv={out['assess_csv']}")
    print(f"bootstrap_validation_avg_csv={out['bootstrap_validation_avg_csv']}")
    print(f"plots_dir={out['plots_dir']}")
