"""
Full-X baseline + simplified original-style spatial lag experiments.

Purpose
-------
This script keeps the repo/new-code preprocessing and feature set:
  - X = params.yaml::model.predictor.all
  - categorical/id preprocessing via preprocessing.recipes_pipelined.build_model_pipeline
  - quick-test split style: early pre-2024 train, later pre-2024 test, 2024 assess

But it replaces the more elaborate spatial-time lag with the simpler original idea:
  - for each target year, use only prior years as history
  - compute a normalized spatial distance and normalized time distance
  - combine them as: D = (1 - w) * D_space_norm + w * D_time_norm
  - enforce strict same-class AND same-modeling-group filters
  - take top K valid historical neighbors
  - use the unweighted mean log sale price as st_neighborhood_avg

This is intentionally simpler than the KDTree/bandwidth version.  To avoid giant
memory spikes, the original year-by-year distance computation is implemented in
chunks, while preserving the same normalization constants per target year.

Example
-------
python spatial_lag_full_x_simple_original.py \
  --data-path ./data/CCAO/2025/training_data.parquet \
  --out-dir ./outputs/spatial_full_x_simple \
  --sample-frac 0.20 \
  --Ks 1,3,5,10,14 \
  --ws 0,0.001,0.01,0.1,1
"""

from __future__ import annotations

import argparse
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics.pairwise import haversine_distances

from preprocessing.recipes_pipelined import build_model_pipeline

try:
    from utils.motivation_utils import _compute_extended_metrics
except Exception:  # lets the script run outside the full repo
    _compute_extended_metrics = None


TARGET_COL = "meta_sale_price"
DATE_COL = "meta_sale_date"
LAT_COL = "loc_latitude"
LON_COL = "loc_longitude"
X_FT_COL = "loc_x_3435"
Y_FT_COL = "loc_y_3435"
DEFAULT_GROUP_FILTER_COL = "meta_modeling_group"
DEFAULT_CLASS_FILTER_COL = "meta_class"
DEED_COL = "meta_sale_deed_type"
EARTH_RADIUS_MILES = 3958.8
LOG_T0 = time.perf_counter()


def log(message: str, **fields: Any) -> None:
    dt = time.perf_counter() - LOG_T0
    suffix = ""
    if fields:
        suffix = " | " + " | ".join(f"{k}={v}" for k, v in fields.items())
    print(f"[spatial_simple +{dt:8.1f}s] {message}{suffix}", flush=True)


def parse_int_list(raw: str) -> Tuple[int, ...]:
    vals = tuple(int(x.strip()) for x in str(raw).split(",") if x.strip())
    if not vals:
        raise ValueError("Expected at least one integer.")
    return vals


def parse_float_list(raw: str) -> Tuple[float, ...]:
    vals = tuple(float(x.strip()) for x in str(raw).split(",") if x.strip())
    if not vals:
        raise ValueError("Expected at least one float.")
    return vals


def parse_str_list(raw: str) -> Tuple[str, ...]:
    vals = tuple(x.strip() for x in str(raw).split(",") if x.strip())
    if not vals:
        raise ValueError("Expected at least one string token.")
    return vals


def safe_log_price(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if np.any(~np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError("Target contains non-finite or non-positive prices after filtering.")
    return np.log(arr)


@dataclass
class PreprocessedLinearModel:
    """Wrapper around the repo preprocessing recipe + a linear estimator."""

    pred_vars: List[str]
    cat_vars: List[str]
    id_vars: List[str]
    estimator_name: str = "linear"
    ridge_alpha: float = 1.0

    def __post_init__(self) -> None:
        self.preprocessor_: Any = None
        self.estimator_: Optional[RegressorMixin] = None

    def _new_estimator(self) -> RegressorMixin:
        name = str(self.estimator_name).lower().strip()
        if name == "linear":
            return LinearRegression(fit_intercept=True)
        if name == "ridge":
            return Ridge(alpha=float(self.ridge_alpha), fit_intercept=True)
        raise ValueError("estimator_name must be 'linear' or 'ridge'.")

    def fit(self, X: pd.DataFrame, y_log: np.ndarray) -> "PreprocessedLinearModel":
        self.preprocessor_ = build_model_pipeline(
            pred_vars=list(self.pred_vars),
            cat_vars=list(self.cat_vars),
            id_vars=[c for c in self.id_vars if c in self.pred_vars],
        )
        X_m = self.preprocessor_.fit_transform(X[list(self.pred_vars)], y_log)
        self.estimator_ = self._new_estimator()
        self.estimator_.fit(X_m, np.asarray(y_log, dtype=float).reshape(-1))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.preprocessor_ is None or self.estimator_ is None:
            raise RuntimeError("Model is not fitted yet.")
        X_m = self.preprocessor_.transform(X[list(self.pred_vars)])
        return np.asarray(self.estimator_.predict(X_m), dtype=float).reshape(-1)


def cast_categoricals(df: pd.DataFrame, categorical_cols: Sequence[str]) -> pd.DataFrame:
    df = df.copy()
    for c in categorical_cols:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df


def load_params(params_path: str) -> dict:
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def add_spatial_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df["year"] = df[DATE_COL].dt.year.astype(int)
    df["y_log"] = safe_log_price(df[TARGET_COL].to_numpy())
    df["sale_day"] = df[DATE_COL].astype("int64") / 1e9 / 86400.0
    return df


def values_as_str_with_na(series: pd.Series) -> np.ndarray:
    return series.astype("object").where(series.notna(), "NA").astype(str).to_numpy()


def load_and_split_data(
    *,
    data_path: str,
    params: dict,
    sample_frac: Optional[float],
    sample_seed: int,
    parquet_engine: str,
    group_filter_col: str,
    class_filter_col: str,
    include_deed_col: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], List[str], List[str]]:
    predictor_cols = list(params["model"]["predictor"]["all"])
    categorical_cols = list(params["model"]["predictor"].get("categorical", []))
    id_vars = list(params["model"]["predictor"].get("id", []))

    filter_cols = ["ind_pin_is_multicard", "sv_is_outlier"]
    extra_cols = [
        TARGET_COL,
        DATE_COL,
        LAT_COL,
        LON_COL,
        X_FT_COL,
        Y_FT_COL,
        group_filter_col,
        class_filter_col,
    ]
    if include_deed_col:
        extra_cols.append(DEED_COL)

    cols_to_load = sorted(set(predictor_cols + filter_cols + extra_cols))

    log("loading parquet", data_path=data_path, engine=parquet_engine, requested_cols=len(cols_to_load))
    try:
        df = pd.read_parquet(data_path, engine=parquet_engine, columns=cols_to_load)
    except Exception as exc:
        log("column-pruned parquet read failed; retrying full read", error=repr(exc))
        df = pd.read_parquet(data_path, engine=parquet_engine)
        missing = [c for c in cols_to_load if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        df = df.loc[:, cols_to_load].copy()

    log("parquet loaded", rows=int(df.shape[0]), cols=int(df.shape[1]))

    df = df[
        (~df["ind_pin_is_multicard"].astype("bool").fillna(True))
        & (~df["sv_is_outlier"].astype("bool").fillna(True))
    ].copy()
    log("repo row filters applied", rows=int(df.shape[0]))

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL, TARGET_COL, LAT_COL, LON_COL]).copy()
    df = df.loc[pd.to_numeric(df[TARGET_COL], errors="coerce") > 0.0, :].copy()

    if sample_frac is not None:
        if not (0.0 < float(sample_frac) <= 1.0):
            raise ValueError("sample_frac must be in (0, 1].")
        if float(sample_frac) < 1.0:
            df = df.sample(frac=float(sample_frac), random_state=int(sample_seed)).copy()
            log("sampling applied", sample_frac=float(sample_frac), rows=int(df.shape[0]))

    df = df.sort_values(DATE_COL).reset_index(drop=True)
    log("date sort completed", rows=int(df.shape[0]))

    df_assess = df.loc[df[DATE_COL].dt.year == 2024, :].copy()
    df_train_all = df.loc[df[DATE_COL].dt.year < 2024, :].copy()

    train_prop = float(params["cv"]["split_prop"])
    split_idx = int(train_prop * df_train_all.shape[0])
    df_train_validate = df_train_all.iloc[:split_idx, :].copy()
    df_test = df_train_all.iloc[split_idx:, :].copy()

    log(
        "quick-test split completed",
        train_validate_rows=int(df_train_validate.shape[0]),
        test_rows=int(df_test.shape[0]),
        assess_rows=int(df_assess.shape[0]),
    )

    df_train_validate = cast_categoricals(df_train_validate, categorical_cols)
    df_test = cast_categoricals(df_test, categorical_cols)
    df_assess = cast_categoricals(df_assess, categorical_cols)

    return df_train_validate, df_test, df_assess, predictor_cols, categorical_cols, id_vars


def fit_full_x_model(
    *,
    train_df: pd.DataFrame,
    pred_vars: List[str],
    cat_vars: List[str],
    id_vars: List[str],
    estimator_name: str,
    ridge_alpha: float,
) -> PreprocessedLinearModel:
    model = PreprocessedLinearModel(
        pred_vars=list(pred_vars),
        cat_vars=[c for c in cat_vars if c in pred_vars],
        id_vars=[c for c in id_vars if c in pred_vars],
        estimator_name=estimator_name,
        ridge_alpha=ridge_alpha,
    )
    model.fit(train_df[pred_vars].copy(), train_df["y_log"].to_numpy(dtype=float))
    return model


def fit_predict_full_x(
    *,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    pred_vars: List[str],
    cat_vars: List[str],
    id_vars: List[str],
    estimator_name: str,
    ridge_alpha: float,
) -> Tuple[np.ndarray, PreprocessedLinearModel]:
    model = fit_full_x_model(
        train_df=train_df,
        pred_vars=pred_vars,
        cat_vars=cat_vars,
        id_vars=id_vars,
        estimator_name=estimator_name,
        ridge_alpha=ridge_alpha,
    )
    pred = model.predict(eval_df[pred_vars].copy())
    return pred, model


def evaluate_predictions(
    *,
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    y_train_log: np.ndarray,
    label: str,
) -> Dict[str, Any]:
    y_true_log = np.asarray(y_true_log, dtype=float).reshape(-1)
    y_pred_log = np.asarray(y_pred_log, dtype=float).reshape(-1)
    log_resid = y_pred_log - y_true_log
    ratio = np.exp(np.clip(log_resid, -50.0, 50.0))

    out: Dict[str, Any] = {
        "model": label,
        "n": int(y_true_log.size),
        "r2_log": float(r2_score(y_true_log, y_pred_log)),
        "rmse_log": float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
        "mae_log": float(mean_absolute_error(y_true_log, y_pred_log)),
        "median_ratio": float(np.median(ratio)),
        "mean_ratio": float(np.mean(ratio)),
    }

    if y_true_log.size >= 2 and np.nanstd(y_true_log) > 0:
        slope_model = LinearRegression().fit(y_true_log.reshape(-1, 1), log_resid.reshape(-1, 1))
        out["log_ratio_slope_vs_log_price"] = float(slope_model.coef_[0, 0])
    else:
        out["log_ratio_slope_vs_log_price"] = np.nan

    if _compute_extended_metrics is not None:
        try:
            ext = _compute_extended_metrics(
                y_true_log=y_true_log,
                y_pred_log=y_pred_log,
                y_train_log=np.asarray(y_train_log, dtype=float).reshape(-1),
                ratio_mode="diff",
            )
            for k, v in ext.items():
                if k not in out:
                    out[k] = v
        except Exception as exc:
            out["extended_metrics_error"] = repr(exc)

    return out


def _distance_matrix_miles(
    df: pd.DataFrame,
    curr_idx: np.ndarray,
    past_idx: np.ndarray,
    *,
    distance_method: str,
) -> np.ndarray:
    if distance_method == "projected" and X_FT_COL in df.columns and Y_FT_COL in df.columns:
        x = df[X_FT_COL].to_numpy(dtype=float)
        y = df[Y_FT_COL].to_numpy(dtype=float)
        if np.isfinite(x[curr_idx]).all() and np.isfinite(y[curr_idx]).all() and np.isfinite(x[past_idx]).all() and np.isfinite(y[past_idx]).all():
            dx = x[curr_idx, None] - x[past_idx][None, :]
            dy = y[curr_idx, None] - y[past_idx][None, :]
            return np.sqrt(dx * dx + dy * dy) / 5280.0

    coords_rad = np.radians(df[[LAT_COL, LON_COL]].to_numpy(dtype=float))
    return haversine_distances(coords_rad[curr_idx], coords_rad[past_idx]) * EARTH_RADIUS_MILES


def _year_max_space_distance(
    df: pd.DataFrame,
    curr_idx: np.ndarray,
    past_idx: np.ndarray,
    *,
    distance_method: str,
    chunk_size: int,
) -> float:
    max_space = 0.0
    for start in range(0, len(curr_idx), int(chunk_size)):
        chunk_idx = curr_idx[start:start + int(chunk_size)]
        d_space = _distance_matrix_miles(df, chunk_idx, past_idx, distance_method=distance_method)
        if d_space.size:
            max_space = max(max_space, float(np.nanmax(d_space)))
    return max_space


def compute_original_style_spatial_lag(
    df: pd.DataFrame,
    target_indices: Sequence[int],
    history_indices: Sequence[int],
    *,
    K: int,
    w: float,
    class_filter_col: str,
    group_filter_col: str,
    chunk_size: int,
    distance_method: str,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Original-style lag, but restricted to explicit target/history splits.

    The attached prototype used target-year loops, prior years only, normalized
    spatial/time matrices, strict class AND group filters, top-K valid neighbors,
    and a simple unweighted average of historical log prices.  This function keeps
    that behavior but computes current-year rows in chunks to avoid huge matrices.
    """
    n = len(df)
    out = np.full(n, np.nan, dtype=float)
    n_neighbors_used = np.zeros(n, dtype=int)
    mean_space_miles = np.full(n, np.nan, dtype=float)
    mean_time_days = np.full(n, np.nan, dtype=float)

    target_indices = np.asarray(target_indices, dtype=int)
    history_indices = np.asarray(history_indices, dtype=int)

    years = df["year"].to_numpy(dtype=int)
    time_days = df["sale_day"].to_numpy(dtype=float)
    y_log = df["y_log"].to_numpy(dtype=float)
    classes = values_as_str_with_na(df[class_filter_col])
    groups = values_as_str_with_na(df[group_filter_col])

    for target_year in sorted(np.unique(years[target_indices])):
        curr_idx = target_indices[years[target_indices] == target_year]
        past_idx = history_indices[years[history_indices] < target_year]

        if len(curr_idx) == 0 or len(past_idx) <= int(K):
            continue

        max_time = float(np.nanmax(time_days[curr_idx]) - np.nanmin(time_days[past_idx]))
        max_space = _year_max_space_distance(
            df,
            curr_idx,
            past_idx,
            distance_method=distance_method,
            chunk_size=chunk_size,
        )

        if max_time <= 0.0 or not np.isfinite(max_time):
            max_time = 1.0
        if max_space <= 0.0 or not np.isfinite(max_space):
            max_space = 1.0

        for start in range(0, len(curr_idx), int(chunk_size)):
            chunk_idx = curr_idx[start:start + int(chunk_size)]
            d_space = _distance_matrix_miles(df, chunk_idx, past_idx, distance_method=distance_method)
            d_time = np.abs(time_days[chunk_idx, None] - time_days[past_idx][None, :])

            d_combined = (1.0 - float(w)) * (d_space / max_space) + float(w) * (d_time / max_time)

            # Strict original filters: same class AND same modeling group.
            d_combined[classes[chunk_idx, None] != classes[past_idx]] = np.inf
            d_combined[groups[chunk_idx, None] != groups[past_idx]] = np.inf

            k_eff = min(int(K), d_combined.shape[1])
            if k_eff <= 0:
                continue

            top_k_local = np.argpartition(d_combined, kth=k_eff - 1, axis=1)[:, :k_eff]
            top_k_dist = np.take_along_axis(d_combined, top_k_local, axis=1)
            valid_mask = np.isfinite(top_k_dist)

            y_past = y_log[past_idx[top_k_local]]
            y_past = np.where(valid_mask, y_past, np.nan)
            selected_space = np.take_along_axis(d_space, top_k_local, axis=1)
            selected_time = np.take_along_axis(d_time, top_k_local, axis=1)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                chunk_lag = np.nanmean(y_past, axis=1)
                chunk_space = np.nanmean(np.where(valid_mask, selected_space, np.nan), axis=1)
                chunk_time = np.nanmean(np.where(valid_mask, selected_time, np.nan), axis=1)

            out[chunk_idx] = chunk_lag
            n_neighbors_used[chunk_idx] = np.sum(valid_mask, axis=1).astype(int)
            mean_space_miles[chunk_idx] = chunk_space
            mean_time_days[chunk_idx] = chunk_time

    diagnostics = {
        "n_neighbors_used": n_neighbors_used,
        "mean_neighbor_space_miles": mean_space_miles,
        "mean_neighbor_time_days": mean_time_days,
    }
    return out, diagnostics


def run_one_split(
    *,
    split_label: str,
    train_df_raw: pd.DataFrame,
    eval_df_raw: pd.DataFrame,
    predictor_cols: List[str],
    categorical_cols: List[str],
    id_vars: List[str],
    estimator_name: str,
    ridge_alpha: float,
    Ks: Sequence[int],
    ws: Sequence[float],
    group_filter_col: str,
    class_filter_col: str,
    allow_sequential_eval_history: bool,
    chunk_size: int,
    distance_method: str,
) -> pd.DataFrame:
    if eval_df_raw.empty:
        log("split skipped: empty eval data", split=split_label)
        return pd.DataFrame()

    train_df = add_spatial_time_columns(train_df_raw).copy()
    eval_df = add_spatial_time_columns(eval_df_raw).copy()

    train_df["__split_row_id"] = [f"train_{i}" for i in range(len(train_df))]
    eval_df["__split_row_id"] = [f"eval_{i}" for i in range(len(eval_df))]
    combined = pd.concat([train_df, eval_df], ignore_index=True).sort_values(DATE_COL).reset_index(drop=True)
    combined = cast_categoricals(combined, categorical_cols)

    train_idx = np.where(combined["__split_row_id"].astype(str).str.startswith("train_"))[0]
    eval_idx = np.where(combined["__split_row_id"].astype(str).str.startswith("eval_"))[0]

    train_for_model = combined.iloc[train_idx].copy()
    eval_for_model = combined.iloc[eval_idx].copy()
    y_train_log = train_for_model["y_log"].to_numpy(dtype=float)
    y_eval_log = eval_for_model["y_log"].to_numpy(dtype=float)

    log(
        "split start",
        split=split_label,
        train_rows=len(train_idx),
        eval_rows=len(eval_idx),
        train_window=f"{train_for_model[DATE_COL].min().date()}..{train_for_model[DATE_COL].max().date()}",
        eval_window=f"{eval_for_model[DATE_COL].min().date()}..{eval_for_model[DATE_COL].max().date()}",
    )

    base_pred_eval, _ = fit_predict_full_x(
        train_df=train_for_model,
        eval_df=eval_for_model,
        pred_vars=predictor_cols,
        cat_vars=categorical_cols,
        id_vars=id_vars,
        estimator_name=estimator_name,
        ridge_alpha=ridge_alpha,
    )

    rows: List[Dict[str, Any]] = []
    base_label = f"{split_label}_baseline_full_X_{estimator_name}"
    base_metrics = evaluate_predictions(
        y_true_log=y_eval_log,
        y_pred_log=base_pred_eval,
        y_train_log=y_train_log,
        label=base_label,
    )
    base_metrics.update(
        {
            "split": split_label,
            "model_family": "baseline_full_X",
            "K": np.nan,
            "w": np.nan,
            "valid_train_rows": int(len(train_idx)),
            "valid_eval_rows": int(len(eval_idx)),
            "mean_eval_neighbors": np.nan,
            "mean_eval_neighbor_space_miles": np.nan,
            "mean_eval_neighbor_time_days": np.nan,
            "distance_method": distance_method,
            "allow_sequential_eval_history": bool(allow_sequential_eval_history),
        }
    )
    rows.append(base_metrics)
    log(
        "baseline finished",
        split=split_label,
        r2_log=f"{base_metrics['r2_log']:.4f}",
        rmse_log=f"{base_metrics['rmse_log']:.4f}",
        median_ratio=f"{base_metrics['median_ratio']:.4f}",
    )

    train_history_idx = train_idx
    eval_history_idx = np.arange(len(combined), dtype=int) if allow_sequential_eval_history else train_idx

    for K in Ks:
        for w in ws:
            log("simple spatial feature start", split=split_label, K=K, w=w)
            train_feat, train_diag = compute_original_style_spatial_lag(
                combined,
                target_indices=train_idx,
                history_indices=train_history_idx,
                K=int(K),
                w=float(w),
                class_filter_col=class_filter_col,
                group_filter_col=group_filter_col,
                chunk_size=int(chunk_size),
                distance_method=distance_method,
            )
            eval_feat, eval_diag = compute_original_style_spatial_lag(
                combined,
                target_indices=eval_idx,
                history_indices=eval_history_idx,
                K=int(K),
                w=float(w),
                class_filter_col=class_filter_col,
                group_filter_col=group_filter_col,
                chunk_size=int(chunk_size),
                distance_method=distance_method,
            )

            feature_col = f"st_original_avg_K{K}_w{w}"
            n_col = f"{feature_col}_n_neighbors"
            space_col = f"{feature_col}_mean_space_miles"
            time_col = f"{feature_col}_mean_time_days"
            combined[feature_col] = np.nan
            combined.loc[train_idx, feature_col] = train_feat[train_idx]
            combined.loc[eval_idx, feature_col] = eval_feat[eval_idx]
            combined[n_col] = 0
            combined.loc[train_idx, n_col] = train_diag["n_neighbors_used"][train_idx]
            combined.loc[eval_idx, n_col] = eval_diag["n_neighbors_used"][eval_idx]
            combined[space_col] = np.nan
            combined.loc[train_idx, space_col] = train_diag["mean_neighbor_space_miles"][train_idx]
            combined.loc[eval_idx, space_col] = eval_diag["mean_neighbor_space_miles"][eval_idx]
            combined[time_col] = np.nan
            combined.loc[train_idx, time_col] = train_diag["mean_neighbor_time_days"][train_idx]
            combined.loc[eval_idx, time_col] = eval_diag["mean_neighbor_time_days"][eval_idx]

            train_valid = train_idx[np.isfinite(combined.loc[train_idx, feature_col].to_numpy(dtype=float))]
            eval_valid = eval_idx[np.isfinite(combined.loc[eval_idx, feature_col].to_numpy(dtype=float))]

            if len(train_valid) < 50 or len(eval_valid) < 50:
                log("simple spatial model skipped: too few valid rows", split=split_label, K=K, w=w, train_valid=len(train_valid), eval_valid=len(eval_valid))
                continue

            pred_vars_spatial = list(predictor_cols) + [feature_col]
            cat_vars_spatial = [c for c in categorical_cols if c in pred_vars_spatial]
            label = f"{split_label}_full_X_plus_original_spatial_K{K}_w{w}"

            train_model_df = combined.iloc[train_valid].copy()
            eval_model_df = combined.iloc[eval_valid].copy()
            pred_eval, _ = fit_predict_full_x(
                train_df=train_model_df,
                eval_df=eval_model_df,
                pred_vars=pred_vars_spatial,
                cat_vars=cat_vars_spatial,
                id_vars=id_vars,
                estimator_name=estimator_name,
                ridge_alpha=ridge_alpha,
            )

            metrics = evaluate_predictions(
                y_true_log=eval_model_df["y_log"].to_numpy(dtype=float),
                y_pred_log=pred_eval,
                y_train_log=train_model_df["y_log"].to_numpy(dtype=float),
                label=label,
            )
            metrics.update(
                {
                    "split": split_label,
                    "model_family": "full_X_plus_original_spatial",
                    "K": int(K),
                    "w": float(w),
                    "valid_train_rows": int(len(train_valid)),
                    "valid_eval_rows": int(len(eval_valid)),
                    "mean_eval_neighbors": float(np.nanmean(combined.loc[eval_valid, n_col].to_numpy(dtype=float))),
                    "mean_eval_neighbor_space_miles": float(np.nanmean(combined.loc[eval_valid, space_col].to_numpy(dtype=float))),
                    "mean_eval_neighbor_time_days": float(np.nanmean(combined.loc[eval_valid, time_col].to_numpy(dtype=float))),
                    "class_filter_col": class_filter_col,
                    "group_filter_col": group_filter_col,
                    "distance_method": distance_method,
                    "allow_sequential_eval_history": bool(allow_sequential_eval_history),
                }
            )
            rows.append(metrics)
            log(
                "simple spatial model finished",
                split=split_label,
                model=label,
                r2_log=f"{metrics['r2_log']:.4f}",
                rmse_log=f"{metrics['rmse_log']:.4f}",
                median_ratio=f"{metrics['median_ratio']:.4f}",
                slope=f"{metrics['log_ratio_slope_vs_log_price']:.5f}",
            )

    result_df = pd.DataFrame(rows)
    if not result_df.empty:
        result_df = result_df.sort_values(["rmse_log", "log_ratio_slope_vs_log_price"], ascending=[True, False]).reset_index(drop=True)
    return result_df


def sanitize_filename(name: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(name))
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "plot"


def write_summary_plots(df: pd.DataFrame, *, split_label: str, out_dir: Path) -> None:
    if df.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_df = df.sort_values("rmse_log").head(15).copy()
    labels = plot_df["model"].astype(str).str.replace(f"{split_label}_", "", regex=False).to_list()
    labels = [s if len(s) <= 64 else s[:61] + "..." for s in labels]

    fig, ax = plt.subplots(figsize=(11, max(5, 0.35 * len(plot_df) + 1.5)))
    y_pos = np.arange(len(plot_df))
    ax.barh(y_pos, plot_df["rmse_log"].to_numpy(dtype=float))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("RMSE_log (lower is better)")
    ax.set_title(f"Top full-X/simple-spatial models by RMSE_log — {split_label}")
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / f"{sanitize_filename(split_label.lower())}_top_rmse.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    grid_df = df.loc[df["model_family"].astype(str) != "baseline_full_X", :].copy()
    if not grid_df.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(
            grid_df["rmse_log"].to_numpy(dtype=float),
            grid_df["log_ratio_slope_vs_log_price"].to_numpy(dtype=float),
            alpha=0.7,
        )
        ax.axhline(0.0, linestyle="--", linewidth=1.0)
        ax.set_xlabel("RMSE_log (lower is better)")
        ax.set_ylabel("log-ratio slope vs log price (closer to 0 is better)")
        ax.set_title(f"Accuracy vs vertical-equity slope — {split_label}")
        ax.grid(True, linestyle=":", alpha=0.4)
        fig.tight_layout()
        fig.savefig(out_dir / f"{sanitize_filename(split_label.lower())}_rmse_scatter.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def write_outputs(*, out_dir: str, test_df: pd.DataFrame, assess_df: pd.DataFrame) -> Dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, str] = {}
    test_path = out / "spatial_simple_full_x_metrics_test.csv"
    assess_path = out / "spatial_simple_full_x_metrics_assess.csv"
    all_path = out / "spatial_simple_full_x_metrics_all.csv"

    test_df.to_csv(test_path, index=False)
    paths["test_metrics"] = str(test_path)
    if not assess_df.empty:
        assess_df.to_csv(assess_path, index=False)
        paths["assess_metrics"] = str(assess_path)

    all_df = pd.concat([test_df, assess_df], ignore_index=True)
    all_df.to_csv(all_path, index=False)
    paths["all_metrics"] = str(all_path)

    write_summary_plots(test_df, split_label="test", out_dir=plots)
    if not assess_df.empty:
        write_summary_plots(assess_df, split_label="assess", out_dir=plots)
    paths["plots_dir"] = str(plots)
    return paths


def apply_deed_scenario(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    scenario = str(scenario).lower().strip()
    if scenario in {"all", "unfiltered"}:
        return df.copy()
    if scenario in {"no_05", "drop_05", "filtered"}:
        if DEED_COL not in df.columns:
            raise ValueError(f"Scenario {scenario} requires {DEED_COL}; rerun with deed column available.")
        return df.loc[df[DEED_COL].astype(str) != "05", :].copy()
    raise ValueError("deed scenario must be one of: all, no_05")


def run_experiments(args: argparse.Namespace) -> Dict[str, str]:
    params = load_params(args.params_path)
    deed_scenarios = parse_str_list(args.deed_scenarios)
    include_deed_col = any(s.lower().strip() not in {"all", "unfiltered"} for s in deed_scenarios)

    df_train_validate, df_test, df_assess, predictor_cols, categorical_cols, id_vars = load_and_split_data(
        data_path=args.data_path,
        params=params,
        sample_frac=args.sample_frac,
        sample_seed=args.seed,
        parquet_engine=args.parquet_engine,
        group_filter_col=args.group_filter_col,
        class_filter_col=args.class_filter_col,
        include_deed_col=include_deed_col,
    )

    Ks = parse_int_list(args.Ks)
    ws = parse_float_list(args.ws)

    all_test_results: List[pd.DataFrame] = []
    all_assess_results: List[pd.DataFrame] = []

    for scenario in deed_scenarios:
        log("deed scenario start", scenario=scenario)
        train_s = apply_deed_scenario(df_train_validate, scenario)
        test_s = apply_deed_scenario(df_test, scenario)
        assess_s = apply_deed_scenario(df_assess, scenario) if not df_assess.empty else df_assess.copy()

        test_results = run_one_split(
            split_label="test",
            train_df_raw=train_s,
            eval_df_raw=test_s,
            predictor_cols=predictor_cols,
            categorical_cols=categorical_cols,
            id_vars=id_vars,
            estimator_name=args.estimator,
            ridge_alpha=args.ridge_alpha,
            Ks=Ks,
            ws=ws,
            group_filter_col=args.group_filter_col,
            class_filter_col=args.class_filter_col,
            allow_sequential_eval_history=args.allow_sequential_eval_history,
            chunk_size=args.chunk_size,
            distance_method=args.distance_method,
        )
        if not test_results.empty:
            test_results.insert(0, "deed_scenario", scenario)
            all_test_results.append(test_results)

        assess_results = pd.DataFrame()
        if not assess_s.empty:
            pre2024_s = pd.concat([train_s, test_s], ignore_index=True)
            assess_results = run_one_split(
                split_label="assess",
                train_df_raw=pre2024_s,
                eval_df_raw=assess_s,
                predictor_cols=predictor_cols,
                categorical_cols=categorical_cols,
                id_vars=id_vars,
                estimator_name=args.estimator,
                ridge_alpha=args.ridge_alpha,
                Ks=Ks,
                ws=ws,
                group_filter_col=args.group_filter_col,
                class_filter_col=args.class_filter_col,
                allow_sequential_eval_history=args.allow_sequential_eval_history,
                chunk_size=args.chunk_size,
                distance_method=args.distance_method,
            )
            if not assess_results.empty:
                assess_results.insert(0, "deed_scenario", scenario)
                all_assess_results.append(assess_results)

    test_df = pd.concat(all_test_results, ignore_index=True) if all_test_results else pd.DataFrame()
    assess_df = pd.concat(all_assess_results, ignore_index=True) if all_assess_results else pd.DataFrame()
    paths = write_outputs(out_dir=args.out_dir, test_df=test_df, assess_df=assess_df)
    log("experiments finished", **paths)
    return paths


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Full-X baseline + original-style simple spatial lag experiments.")
    p.add_argument("--data-path", default="./data/CCAO/2025/training_data.parquet")
    p.add_argument("--params-path", default="params.yaml")
    p.add_argument("--out-dir", default="./outputs/spatial_full_x_simple")
    p.add_argument("--parquet-engine", default="pyarrow", choices=["pyarrow", "fastparquet", "auto"])
    p.add_argument("--sample-frac", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--estimator", default="linear", choices=["linear", "ridge"])
    p.add_argument("--ridge-alpha", type=float, default=1.0)

    p.add_argument("--Ks", default="1,3,5,10,14")
    p.add_argument("--ws", default="0,0.001,0.01,0.1,1")

    p.add_argument("--group-filter-col", default=DEFAULT_GROUP_FILTER_COL)
    p.add_argument("--class-filter-col", default=DEFAULT_CLASS_FILTER_COL)
    p.add_argument("--deed-scenarios", default="all", help="Comma list: all,no_05. Default only all.")
    p.add_argument("--allow-sequential-eval-history", action="store_true", help="Allow earlier eval-period years as history. Default is strict holdout history.")
    p.add_argument("--chunk-size", type=int, default=250, help="Rows per current-year distance block. Lower uses less memory.")
    p.add_argument("--distance-method", default="haversine", choices=["haversine", "projected"], help="haversine matches the attached prototype; projected is faster/local.")
    return p


if __name__ == "__main__":
    parser = build_arg_parser()
    run_experiments(parser.parse_args())
