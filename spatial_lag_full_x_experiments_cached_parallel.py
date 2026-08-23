"""
Spatial-time lag experiments using the repo/main-model preprocessing.

Optimized variant: KDTree candidate caches are precomputed once per K/split and
then reused across bandwidths and feature types; cache building can use threads.

What this script changes relative to the simple spatial notebook/script
--------------------------------------------------------------------
1) The baseline model is no longer a small hand-written control model.
   It uses every feature in params.yaml::model.predictor.all and the same
   preprocessing recipe used by the repo's linear model:

       preprocessing.recipes_pipelined.build_model_pipeline(...)

2) Spatial lag features are added on top of that setting.  The default grid
   follows the findings from the previous spatial-time experiment:
   K around 10, spatial bandwidth around 0.5-1.0 miles, and a longer time
   bandwidth around 365 days.

3) Evaluation mirrors the quick-test split logic:
   - TEST split: train on early pre-2024 sales, evaluate on held-out later
     pre-2024 sales.
   - ASSESS split: train on all pre-2024 sales, evaluate on 2024 sales.

Outputs
-------
Writes under --out-dir:
  - spatial_full_x_metrics_test.csv
  - spatial_full_x_metrics_assess.csv
  - spatial_full_x_metrics_all.csv
  - plots/test_top_rmse.png
  - plots/assess_top_rmse.png
  - plots/test_top_rmse_price.png
  - plots/assess_top_rmse_price.png
  - plots/test_rmse_scatter.png
  - plots/assess_rmse_scatter.png
  - plots/test_price_rmse_vs_r2.png
  - plots/assess_price_rmse_vs_r2.png

Usage
-----
From the project root, for a quick run:

  python spatial_lag_full_x_experiments.py \
    --data-path ./data/CCAO/2025/training_data.parquet \
    --out-dir ./outputs/spatial_full_x \
    --sample-frac 0.20

For the full run:

  python spatial_lag_full_x_experiments.py \
    --data-path ./data/CCAO/2025/training_data.parquet \
    --out-dir ./outputs/spatial_full_x
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KDTree

try:
    from joblib import Parallel, delayed
except Exception:  # pragma: no cover
    Parallel = None
    delayed = None

from preprocessing.recipes_pipelined import build_model_pipeline

try:
    from utils.motivation_utils import _compute_extended_metrics
except Exception:  # pragma: no cover - lets the script run outside the full repo
    _compute_extended_metrics = None


# -----------------------------------------------------------------------------
# Constants / defaults
# -----------------------------------------------------------------------------

TARGET_COL = "meta_sale_price"
DATE_COL = "meta_sale_date"
LAT_COL = "loc_latitude"
LON_COL = "loc_longitude"
X_FT_COL = "loc_x_3435"
Y_FT_COL = "loc_y_3435"
DEFAULT_GROUP_FILTER_COL = "meta_modeling_group"
DEFAULT_CLASS_FILTER_COL = "char_class"  # fallback to meta_class if missing
META_CLASS_COL = "meta_class"
EARTH_RADIUS_MILES = 3958.8
LOG_T0 = time.perf_counter()


# -----------------------------------------------------------------------------
# Logging / parsing helpers
# -----------------------------------------------------------------------------

def log(message: str, **fields: Any) -> None:
    dt = time.perf_counter() - LOG_T0
    suffix = ""
    if fields:
        suffix = " | " + " | ".join(f"{k}={v}" for k, v in fields.items())
    print(f"[spatial_full_x +{dt:8.1f}s] {message}{suffix}", flush=True)


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


def safe_log_price(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if np.any(~np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError("Target contains non-finite or non-positive prices after filtering.")
    return np.log(arr)


# -----------------------------------------------------------------------------
# Model wrapper: same repo preprocessing + sklearn linear estimator
# -----------------------------------------------------------------------------

@dataclass
class PreprocessedLinearModel:
    """A small wrapper around build_model_pipeline + a linear estimator."""

    pred_vars: List[str]
    cat_vars: List[str]
    id_vars: List[str]
    estimator_name: str = "linear"
    ridge_alpha: float = 1.0

    def __post_init__(self) -> None:
        self.preprocessor_: Any = None
        self.estimator_: RegressorMixin | None = None

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


# -----------------------------------------------------------------------------
# Data loading: mirrors quick_test_models.py split and preprocessing inputs
# -----------------------------------------------------------------------------

def load_params(params_path: str) -> dict:
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_and_split_data(
    *,
    data_path: str,
    params: dict,
    sample_frac: Optional[float],
    sample_seed: int,
    parquet_engine: str,
    group_filter_col: str,
    class_filter_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], List[str], List[str]]:
    """Mirrors the quick-test loader, but also keeps spatial/filter columns.

    The model X itself remains exactly params["model"]["predictor"]["all"].
    Extra columns are kept only for filtering, target/date, and spatial lag building.
    """

    predictor_cols = list(params["model"]["predictor"]["all"])
    categorical_cols = list(params["model"]["predictor"].get("categorical", []))
    id_vars = list(params["model"]["predictor"].get("id", []))

    filter_cols = ["ind_pin_is_multicard", "sv_is_outlier"]
    spatial_extra_cols = [
        TARGET_COL,
        DATE_COL,
        LAT_COL,
        LON_COL,
        X_FT_COL,
        Y_FT_COL,
        META_CLASS_COL,
        group_filter_col,
        class_filter_col,
    ]
    cols_to_load = sorted(set(predictor_cols + filter_cols + spatial_extra_cols))

    log("loading parquet", data_path=data_path, engine=parquet_engine, requested_cols=len(cols_to_load))
    try:
        df = pd.read_parquet(data_path, engine=parquet_engine, columns=cols_to_load)
    except Exception as exc:
        log("column-pruned parquet read failed; retrying full read", error=repr(exc))
        df = pd.read_parquet(data_path, engine=parquet_engine)
        missing_after_full = [c for c in cols_to_load if c not in df.columns]
        if missing_after_full:
            raise ValueError(f"Missing required columns: {missing_after_full}")
        df = df.loc[:, cols_to_load].copy()

    log("parquet loaded", rows=int(df.shape[0]), cols=int(df.shape[1]))

    # Same row filters as quick_test_models.py: drop multicards and sales-val outliers.
    if "ind_pin_is_multicard" not in df.columns or "sv_is_outlier" not in df.columns:
        raise ValueError("Data must include ind_pin_is_multicard and sv_is_outlier for the repo-style filter.")
    df = df[
        (~df["ind_pin_is_multicard"].astype("bool").fillna(True))
        & (~df["sv_is_outlier"].astype("bool").fillna(True))
    ].copy()
    log("repo row filters applied", rows=int(df.shape[0]))

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL, TARGET_COL]).copy()
    df = df.loc[pd.to_numeric(df[TARGET_COL], errors="coerce") > 0.0, :].copy()

    # Need coordinates for the spatial lag. Prefer projected feet; if missing, fallback
    # code uses lat/lon, so require one complete coordinate system.
    has_xy = X_FT_COL in df.columns and Y_FT_COL in df.columns and df[[X_FT_COL, Y_FT_COL]].notna().all(axis=1).any()
    has_latlon = LAT_COL in df.columns and LON_COL in df.columns and df[[LAT_COL, LON_COL]].notna().all(axis=1).any()
    if not has_xy and not has_latlon:
        raise ValueError("Need either loc_x_3435/loc_y_3435 or loc_latitude/loc_longitude for spatial lags.")

    # Drop rows missing both usable spatial coordinate systems.  This affects only
    # the spatial experiment universe; the X columns are otherwise untouched.
    if has_xy:
        df = df.dropna(subset=[X_FT_COL, Y_FT_COL]).copy()
    else:
        df = df.dropna(subset=[LAT_COL, LON_COL]).copy()

    if sample_frac is not None:
        if not (0.0 < float(sample_frac) <= 1.0):
            raise ValueError("sample_frac must be in (0, 1]. Use None to disable sampling.")
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

    # Cast categoricals to match the quick-test behavior before model fitting.
    df_train_validate = cast_categoricals(df_train_validate, categorical_cols)
    df_test = cast_categoricals(df_test, categorical_cols)
    df_assess = cast_categoricals(df_assess, categorical_cols)

    return df_train_validate, df_test, df_assess, predictor_cols, categorical_cols, id_vars


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def evaluate_predictions(
    *,
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    y_train_log: np.ndarray,
    label: str,
    ratio_mode: str = "diff",
) -> Dict[str, Any]:
    y_true_log = np.asarray(y_true_log, dtype=float).reshape(-1)
    y_pred_log = np.asarray(y_pred_log, dtype=float).reshape(-1)

    # Log-scale residuals/ratios.  Ratios are clipped only to avoid numerical
    # overflow in pathological cases; the log predictions themselves are not
    # modified for log-scale metrics.
    log_resid = y_pred_log - y_true_log
    ratio = np.exp(np.clip(log_resid, -50.0, 50.0))

    # Original price-scale predictions.  These are the metrics requested for the
    # exponentiated objective: exp(y_log) and exp(y_hat_log).  Since the training
    # target is log(price), these metrics are not the optimized loss, but they
    # are the interpretable sale-price-scale RMSE/R²/MAE.
    y_true_price = np.exp(np.clip(y_true_log, -50.0, 50.0))
    y_pred_price = np.exp(np.clip(y_pred_log, -50.0, 50.0))

    out: Dict[str, Any] = {
        "model": label,
        "n": int(y_true_log.size),
        "r2_log": float(r2_score(y_true_log, y_pred_log)),
        "rmse_log": float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
        "mae_log": float(mean_absolute_error(y_true_log, y_pred_log)),
        "r2_price": float(r2_score(y_true_price, y_pred_price)),
        "rmse_price": float(np.sqrt(mean_squared_error(y_true_price, y_pred_price))),
        "mae_price": float(mean_absolute_error(y_true_price, y_pred_price)),
        "median_ratio": float(np.median(ratio)),
        "mean_ratio": float(np.mean(ratio)),
    }

    # Negative slope means lower ratios for higher-price homes = regressivity.
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
                ratio_mode=ratio_mode,
            )
            # Keep existing simple metric names, append repo metrics as-is.
            for k, v in ext.items():
                if k not in out:
                    out[k] = v
        except Exception as exc:
            out["extended_metrics_error"] = repr(exc)

    return out


# -----------------------------------------------------------------------------
# Time slope and spatial lag feature generation
# -----------------------------------------------------------------------------

def add_spatial_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df["y_log"] = safe_log_price(df[TARGET_COL].to_numpy())
    # pandas datetime64[ns] as days since epoch
    df["sale_day"] = df[DATE_COL].astype("int64") / 1e9 / 86400.0
    df["sale_year_float"] = df["sale_day"] / 365.25
    return df


def estimate_global_time_slope(train_df: pd.DataFrame) -> float:
    """Simple train-only annual log-price slope for time-adjusted neighbor price.

    This intentionally stays independent of the full X preprocessing because it is
    used only to move a neighbor sale from its sale date to the target sale date.
    """
    x = train_df["sale_year_float"].to_numpy(dtype=float)
    y = train_df["y_log"].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2 or np.nanstd(x[mask]) <= 0.0:
        return 0.0
    return float(np.polyfit(x[mask], y[mask], deg=1)[0])


def coordinate_distance_miles(df: pd.DataFrame, target_idx: np.ndarray, history_idx: np.ndarray) -> np.ndarray:
    """Return target x history distance matrix in miles.

    Prefer Cook County projected feet coordinates when available.  They avoid the
    extra haversine dependency and are better for local Euclidean distances.
    """
    if X_FT_COL in df.columns and Y_FT_COL in df.columns and df[[X_FT_COL, Y_FT_COL]].notna().all(axis=1).all():
        x = df[X_FT_COL].to_numpy(dtype=float)
        y = df[Y_FT_COL].to_numpy(dtype=float)
        dx = x[target_idx, None] - x[history_idx][None, :]
        dy = y[target_idx, None] - y[history_idx][None, :]
        return np.sqrt(dx * dx + dy * dy) / 5280.0

    lat = np.radians(df[LAT_COL].to_numpy(dtype=float))
    lon = np.radians(df[LON_COL].to_numpy(dtype=float))
    dlat = lat[target_idx, None] - lat[history_idx][None, :]
    dlon = lon[target_idx, None] - lon[history_idx][None, :]
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat[target_idx, None]) * np.cos(lat[history_idx][None, :]) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_MILES * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def values_as_str_with_na(series: pd.Series) -> np.ndarray:
    return series.astype("object").where(series.notna(), "NA").astype(str).to_numpy()



def _spatial_xy_miles(df: pd.DataFrame) -> np.ndarray:
    """Return 2D coordinates in miles, preferring projected Cook County feet."""
    if X_FT_COL in df.columns and Y_FT_COL in df.columns and df[[X_FT_COL, Y_FT_COL]].notna().all(axis=1).all():
        return np.column_stack([
            df[X_FT_COL].to_numpy(dtype=float) / 5280.0,
            df[Y_FT_COL].to_numpy(dtype=float) / 5280.0,
        ])

    # Local equirectangular approximation in miles. This is only a fallback;
    # loc_x_3435 / loc_y_3435 should normally be present for CCAO data.
    lat_rad = np.radians(df[LAT_COL].to_numpy(dtype=float))
    lon_rad = np.radians(df[LON_COL].to_numpy(dtype=float))
    lat0 = float(np.nanmean(lat_rad))
    return np.column_stack([
        EARTH_RADIUS_MILES * lon_rad * np.cos(lat0),
        EARTH_RADIUS_MILES * lat_rad,
    ])


class _SpatialPoolIndex:
    """KDTree cache for fixed history rows, grouped by group/class filters.

    The old implementation formed a dense target x history distance matrix.  This
    cache instead looks up only a fixed number of spatially nearby candidates,
    then applies the same temporal, group/class, and combined score logic within
    that candidate set.
    """

    def __init__(
        self,
        *,
        df: pd.DataFrame,
        history_indices: Sequence[int],
        group_filter_col: Optional[str],
        class_filter_col: Optional[str],
    ) -> None:
        self.df = df
        self.history_indices = np.asarray(history_indices, dtype=int)
        self.coords = _spatial_xy_miles(df)
        self.sale_day = df["sale_day"].to_numpy(dtype=float)

        self.group_vals = None
        if group_filter_col and group_filter_col in df.columns:
            self.group_vals = values_as_str_with_na(df[group_filter_col])

        self.class_vals = None
        if class_filter_col and class_filter_col in df.columns:
            self.class_vals = values_as_str_with_na(df[class_filter_col])

        self._pools: Dict[Tuple[str, str], Tuple[np.ndarray, KDTree]] = {}
        self._build_pools()

    def _group_key(self, idx: int) -> str:
        if self.group_vals is None:
            return "__all_groups__"
        return str(self.group_vals[int(idx)])

    def _class_key(self, idx: int) -> str:
        if self.class_vals is None:
            return "__all_classes__"
        return str(self.class_vals[int(idx)])

    def _build_pools(self) -> None:
        if self.history_indices.size == 0:
            return

        group_keys = (
            np.full(self.history_indices.size, "__all_groups__", dtype=object)
            if self.group_vals is None
            else self.group_vals[self.history_indices].astype(object)
        )
        class_keys = (
            np.full(self.history_indices.size, "__all_classes__", dtype=object)
            if self.class_vals is None
            else self.class_vals[self.history_indices].astype(object)
        )

        # Group-only pools are used for class fallback. Class-specific pools are
        # used first when enough same-class prior candidates exist.
        pool_members: Dict[Tuple[str, str], List[int]] = {}
        for hist_idx, g, c in zip(self.history_indices, group_keys, class_keys):
            gi = str(g)
            ci = str(c)
            pool_members.setdefault((gi, "__all_classes__"), []).append(int(hist_idx))
            pool_members.setdefault((gi, ci), []).append(int(hist_idx))

        if self.group_vals is None:
            # Also make a universal pool.  This is useful if users disable group
            # filtering by passing a missing group column.
            pool_members.setdefault(("__all_groups__", "__all_classes__"), [int(i) for i in self.history_indices])

        for key, members in pool_members.items():
            idx = np.asarray(members, dtype=int)
            if idx.size == 0:
                continue
            self._pools[key] = (idx, KDTree(self.coords[idx], leaf_size=40, metric="euclidean"))

    def _query_spatial_candidates(
        self,
        *,
        target_i: int,
        key: Tuple[str, str],
        n_candidates: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        pool = self._pools.get(key)
        if pool is None:
            return np.empty(0, dtype=int), np.empty(0, dtype=float)
        idx, tree = pool
        if idx.size == 0:
            return np.empty(0, dtype=int), np.empty(0, dtype=float)
        k = min(int(n_candidates), int(idx.size))
        if k <= 0:
            return np.empty(0, dtype=int), np.empty(0, dtype=float)
        dist, pos = tree.query(self.coords[int(target_i)].reshape(1, -1), k=k)
        pos = np.asarray(pos[0], dtype=int)
        dist = np.asarray(dist[0], dtype=float)
        return idx[pos], dist

    def candidate_set(
        self,
        *,
        target_i: int,
        K: int,
        min_same_class_pool: int,
        allow_class_fallback: bool,
        max_neighbor_age_days: Optional[float],
        max_spatial_candidates: int,
        candidate_multiplier: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """Return candidate history indices, distances, ages, and fallback flag.

        Adaptive KDTree expansion:
        - Start with roughly K * candidate_multiplier spatial candidates.
        - Apply the strict temporal filter (prior sales only) and optional max-age
          filter *after* each query.
        - If too few valid prior candidates remain, double the spatial query size
          until enough valid candidates are found or max_spatial_candidates / pool
          size is reached.

        This fixes the bounded-candidate failure mode where a valid prior comp is
        just outside the initial spatial cutoff because many closer properties are
        future sales or age-filtered away.
        """
        g = self._group_key(target_i)
        c = self._class_key(target_i)
        threshold = max(1, min(int(K), int(min_same_class_pool)))

        max_query = int(max_spatial_candidates) if max_spatial_candidates is not None and int(max_spatial_candidates) > 0 else 0
        if max_query <= 0:
            max_query = 10**9

        start_query = max(int(K) * int(candidate_multiplier), int(K), threshold)
        start_query = max(1, min(start_query, max_query))

        def valid_from_key(key: Tuple[str, str], required_valid: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            pool = self._pools.get(key)
            if pool is None:
                return np.empty(0, dtype=int), np.empty(0, dtype=float), np.empty(0, dtype=float)
            pool_idx, _tree = pool
            pool_size = int(pool_idx.size)
            if pool_size == 0:
                return np.empty(0, dtype=int), np.empty(0, dtype=float), np.empty(0, dtype=float)

            query_n = min(start_query, pool_size, max_query)
            best_idx = np.empty(0, dtype=int)
            best_space = np.empty(0, dtype=float)
            best_time = np.empty(0, dtype=float)

            while query_n > 0:
                cand_idx, d_space = self._query_spatial_candidates(target_i=target_i, key=key, n_candidates=query_n)
                if cand_idx.size == 0:
                    return cand_idx, d_space, np.empty(0, dtype=float)

                d_time = self.sale_day[int(target_i)] - self.sale_day[cand_idx]
                valid = d_time > 0.0
                if max_neighbor_age_days is not None and float(max_neighbor_age_days) > 0.0:
                    valid &= d_time <= float(max_neighbor_age_days)

                best_idx = cand_idx[valid]
                best_space = d_space[valid]
                best_time = d_time[valid]

                if best_idx.size >= int(required_valid):
                    break

                next_query_n = min(query_n * 2, pool_size, max_query)
                if next_query_n <= query_n:
                    break
                query_n = next_query_n

            return best_idx, best_space, best_time

        # Try same class first.  The class-specific tree is expanded before
        # falling back, so fallback is not triggered merely because the first
        # fixed-size spatial query had too many future or age-filtered sales.
        used_fallback = False
        if self.class_vals is not None:
            same_idx, same_space, same_time = valid_from_key((g, c), required_valid=threshold)
            if same_idx.size >= threshold or not allow_class_fallback:
                return same_idx, same_space, same_time, False
            used_fallback = True

        group_idx, group_space, group_time = valid_from_key((g, "__all_classes__"), required_valid=threshold)
        return group_idx, group_space, group_time, used_fallback


@dataclass
class _SpatialCandidateCache:
    """Variable-length KDTree candidate lists for a fixed K/history/target set.

    Candidate lookup is independent of spatial/time bandwidth and feature type.
    Reusing this cache is the main acceleration over calling KDTree once per
    (feature_type, spatial_bw, time_bw) grid cell.
    """

    target_indices: np.ndarray
    candidate_indices: List[np.ndarray]
    d_space_miles: List[np.ndarray]
    d_time_days: List[np.ndarray]
    used_class_fallback_local: np.ndarray
    candidate_count_local: np.ndarray


def precompute_spatial_candidate_cache(
    *,
    df: pd.DataFrame,
    target_indices: Sequence[int],
    history_indices: Sequence[int],
    K: int,
    group_filter_col: Optional[str],
    class_filter_col: Optional[str],
    min_same_class_pool: int,
    allow_class_fallback: bool,
    max_neighbor_age_days: Optional[float],
    max_spatial_candidates: int,
    candidate_multiplier: int,
    n_jobs: int = 1,
    progress_label: str = "candidate_cache",
) -> _SpatialCandidateCache:
    """Precompute bounded spatial candidate sets for one K and one split.

    The expensive KDTree work is done here once. Later, each bandwidth/feature
    configuration just rescoring these same candidates, which is much cheaper.
    """

    target_indices_arr = np.asarray(target_indices, dtype=int)
    pool_index = _SpatialPoolIndex(
        df=df,
        history_indices=history_indices,
        group_filter_col=group_filter_col,
        class_filter_col=class_filter_col,
    )

    def _one(target_i: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        return pool_index.candidate_set(
            target_i=int(target_i),
            K=int(K),
            min_same_class_pool=int(min_same_class_pool),
            allow_class_fallback=bool(allow_class_fallback),
            max_neighbor_age_days=max_neighbor_age_days,
            max_spatial_candidates=int(max_spatial_candidates),
            candidate_multiplier=int(candidate_multiplier),
        )

    n_jobs_eff = int(n_jobs) if n_jobs is not None else 1
    if n_jobs_eff == 0:
        n_jobs_eff = max(1, int(os.cpu_count() or 1))

    if n_jobs_eff == 1 or Parallel is None or delayed is None:
        results = []
        for pos, i in enumerate(target_indices_arr):
            results.append(_one(int(i)))
            if pos > 0 and pos % 25000 == 0:
                log("candidate cache progress", label=progress_label, K=K, done=pos, total=len(target_indices_arr))
    else:
        # KDTree queries are read-only; threading avoids copying the full frame/tree
        # to child processes.  Keep n_jobs modest on shared nodes.
        results = Parallel(n_jobs=n_jobs_eff, prefer="threads", batch_size=256)(
            delayed(_one)(int(i)) for i in target_indices_arr
        )

    cand_indices: List[np.ndarray] = []
    d_spaces: List[np.ndarray] = []
    d_times: List[np.ndarray] = []
    fallback = np.zeros(target_indices_arr.size, dtype=bool)
    cand_count = np.zeros(target_indices_arr.size, dtype=int)

    for pos, (cand_idx, d_space, d_time, used_fallback) in enumerate(results):
        cand_indices.append(np.asarray(cand_idx, dtype=int))
        d_spaces.append(np.asarray(d_space, dtype=float))
        d_times.append(np.asarray(d_time, dtype=float))
        fallback[pos] = bool(used_fallback)
        cand_count[pos] = int(len(cand_idx))

    return _SpatialCandidateCache(
        target_indices=target_indices_arr,
        candidate_indices=cand_indices,
        d_space_miles=d_spaces,
        d_time_days=d_times,
        used_class_fallback_local=fallback,
        candidate_count_local=cand_count,
    )


def compute_spatial_time_lag_from_cache(
    df: pd.DataFrame,
    cache: _SpatialCandidateCache,
    *,
    K: int,
    spatial_bw_miles: float,
    time_bw_days: float,
    feature_type: str,
    beta_time: float,
    base_residuals: Optional[np.ndarray],
    base_pred: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Compute one lag feature by rescoring a precomputed candidate cache."""

    valid_feature_types = {"raw_price", "time_adjusted_price", "residual", "scaled_price", "scaled_time_adjusted_price"}
    if feature_type not in valid_feature_types:
        raise ValueError(f"feature_type must be one of {sorted(valid_feature_types)}.")
    if feature_type == "residual" and base_residuals is None:
        raise ValueError("base_residuals are required for residual spatial lag.")
    if feature_type in {"scaled_price", "scaled_time_adjusted_price"} and base_pred is None:
        raise ValueError("base_pred is required for baseline-scaled spatial price lags.")

    n = len(df)
    out = np.full(n, np.nan, dtype=float)
    n_neighbors_used = np.zeros(n, dtype=int)
    mean_space_miles = np.full(n, np.nan, dtype=float)
    mean_time_days = np.full(n, np.nan, dtype=float)
    used_class_fallback = np.zeros(n, dtype=bool)
    candidate_count = np.zeros(n, dtype=int)

    sale_year_float = df["sale_year_float"].to_numpy(dtype=float)
    y_log = df["y_log"].to_numpy(dtype=float)
    base_residuals_arr = None if base_residuals is None else np.asarray(base_residuals, dtype=float)
    base_pred_arr = None if base_pred is None else np.asarray(base_pred, dtype=float)

    for local_pos, i in enumerate(cache.target_indices):
        i_int = int(i)
        cand_idx = cache.candidate_indices[local_pos]
        d_space = cache.d_space_miles[local_pos]
        d_time = cache.d_time_days[local_pos]
        used_class_fallback[i_int] = bool(cache.used_class_fallback_local[local_pos])
        candidate_count[i_int] = int(cache.candidate_count_local[local_pos])

        if cand_idx.size == 0:
            continue

        score = d_space / float(spatial_bw_miles) + d_time / float(time_bw_days)
        finite = np.isfinite(score)
        if not finite.any():
            continue
        cand_idx = cand_idx[finite]
        d_space = d_space[finite]
        d_time = d_time[finite]
        score = score[finite]

        k_eff = min(int(K), int(cand_idx.size))
        selected_local = np.argpartition(score, k_eff - 1)[:k_eff]
        selected_global = cand_idx[selected_local]
        selected_score = score[selected_local]

        weights = np.exp(-selected_score)
        weights_sum = float(weights.sum())
        if weights_sum <= 0.0 or not np.isfinite(weights_sum):
            continue

        if feature_type == "raw_price":
            vals = y_log[selected_global]
        elif feature_type == "time_adjusted_price":
            delta_years = sale_year_float[i_int] - sale_year_float[selected_global]
            vals = y_log[selected_global] + float(beta_time) * delta_years
        elif feature_type == "scaled_price":
            # Structure-adjust the neighbor sale into the target's baseline-value currency.
            # log(price_neighbor_scaled_to_target) = log(price_neighbor) + base_pred_target - base_pred_neighbor
            vals = y_log[selected_global] + base_pred_arr[i_int] - base_pred_arr[selected_global]
        elif feature_type == "scaled_time_adjusted_price":
            delta_years = sale_year_float[i_int] - sale_year_float[selected_global]
            vals = (
                y_log[selected_global]
                + float(beta_time) * delta_years
                + base_pred_arr[i_int]
                - base_pred_arr[selected_global]
            )
        else:
            vals = base_residuals_arr[selected_global]

        out[i_int] = float(np.sum(weights * vals) / weights_sum)
        n_neighbors_used[i_int] = k_eff
        mean_space_miles[i_int] = float(np.sum(weights * d_space[selected_local]) / weights_sum)
        mean_time_days[i_int] = float(np.sum(weights * d_time[selected_local]) / weights_sum)

    diagnostics = {
        "n_neighbors_used": n_neighbors_used,
        "mean_neighbor_space_miles": mean_space_miles,
        "mean_neighbor_time_days": mean_time_days,
        "used_class_fallback": used_class_fallback,
        "candidate_count": candidate_count,
    }
    return out, diagnostics


def compute_spatial_time_lag(
    df: pd.DataFrame,
    target_indices: Sequence[int],
    history_indices: Sequence[int],
    *,
    K: int,
    spatial_bw_miles: float,
    time_bw_days: float,
    feature_type: str,
    beta_time: float,
    base_residuals: Optional[np.ndarray],
    base_pred: Optional[np.ndarray] = None,
    group_filter_col: Optional[str] = None,
    class_filter_col: Optional[str] = None,
    min_same_class_pool: int = 10,
    allow_class_fallback: bool = True,
    max_neighbor_age_days: Optional[float] = None,
    chunk_size: int = 500,
    max_spatial_candidates: int = 512,
    candidate_multiplier: int = 64,
    search_method: str = "kdtree",
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Compute temporally valid spatial-time lag features.

    search_method="dense" reproduces the original target x history matrix logic
    exactly, but can be extremely slow.  search_method="kdtree" keeps the same
    experimental behavior but limits each target to a fixed number of spatially
    close candidates before applying the temporal/class filters and final
    combined score.
    """

    if search_method not in {"kdtree", "dense"}:
        raise ValueError("search_method must be 'kdtree' or 'dense'.")

    if search_method == "dense":
        return _compute_spatial_time_lag_dense(
            df,
            target_indices,
            history_indices,
            K=K,
            spatial_bw_miles=spatial_bw_miles,
            time_bw_days=time_bw_days,
            feature_type=feature_type,
            beta_time=beta_time,
            base_residuals=base_residuals,
            base_pred=base_pred,
            group_filter_col=group_filter_col,
            class_filter_col=class_filter_col,
            min_same_class_pool=min_same_class_pool,
            allow_class_fallback=allow_class_fallback,
            max_neighbor_age_days=max_neighbor_age_days,
            chunk_size=chunk_size,
        )

    valid_feature_types = {"raw_price", "time_adjusted_price", "residual", "scaled_price", "scaled_time_adjusted_price"}
    if feature_type not in valid_feature_types:
        raise ValueError(f"feature_type must be one of {sorted(valid_feature_types)}.")
    if feature_type == "residual" and base_residuals is None:
        raise ValueError("base_residuals are required for residual spatial lag.")
    if feature_type in {"scaled_price", "scaled_time_adjusted_price"} and base_pred is None:
        raise ValueError("base_pred is required for baseline-scaled spatial price lags.")

    n = len(df)
    out = np.full(n, np.nan, dtype=float)
    n_neighbors_used = np.zeros(n, dtype=int)
    mean_space_miles = np.full(n, np.nan, dtype=float)
    mean_time_days = np.full(n, np.nan, dtype=float)
    used_class_fallback = np.zeros(n, dtype=bool)
    candidate_count = np.zeros(n, dtype=int)

    target_indices = np.asarray(target_indices, dtype=int)
    history_indices = np.asarray(history_indices, dtype=int)

    sale_year_float = df["sale_year_float"].to_numpy(dtype=float)
    y_log = df["y_log"].to_numpy(dtype=float)
    base_residuals_arr = None if base_residuals is None else np.asarray(base_residuals, dtype=float)
    base_pred_arr = None if base_pred is None else np.asarray(base_pred, dtype=float)

    pool_index = _SpatialPoolIndex(
        df=df,
        history_indices=history_indices,
        group_filter_col=group_filter_col,
        class_filter_col=class_filter_col,
    )

    for row_counter, i in enumerate(target_indices):
        cand_idx, d_space, d_time, fallback = pool_index.candidate_set(
            target_i=int(i),
            K=int(K),
            min_same_class_pool=int(min_same_class_pool),
            allow_class_fallback=bool(allow_class_fallback),
            max_neighbor_age_days=max_neighbor_age_days,
            max_spatial_candidates=int(max_spatial_candidates),
            candidate_multiplier=int(candidate_multiplier),
        )
        candidate_count[int(i)] = int(cand_idx.size)
        used_class_fallback[int(i)] = bool(fallback)
        if cand_idx.size == 0:
            continue

        score = d_space / float(spatial_bw_miles) + d_time / float(time_bw_days)
        finite = np.isfinite(score)
        if not finite.any():
            continue
        cand_idx = cand_idx[finite]
        d_space = d_space[finite]
        d_time = d_time[finite]
        score = score[finite]

        k_eff = min(int(K), int(cand_idx.size))
        selected_local = np.argpartition(score, k_eff - 1)[:k_eff]
        selected_global = cand_idx[selected_local]
        selected_score = score[selected_local]

        weights = np.exp(-selected_score)
        weights_sum = float(weights.sum())
        if weights_sum <= 0.0 or not np.isfinite(weights_sum):
            continue

        if feature_type == "raw_price":
            vals = y_log[selected_global]
        elif feature_type == "time_adjusted_price":
            delta_years = sale_year_float[int(i)] - sale_year_float[selected_global]
            vals = y_log[selected_global] + float(beta_time) * delta_years
        elif feature_type == "scaled_price":
            vals = y_log[selected_global] + base_pred_arr[int(i)] - base_pred_arr[selected_global]
        elif feature_type == "scaled_time_adjusted_price":
            delta_years = sale_year_float[int(i)] - sale_year_float[selected_global]
            vals = (
                y_log[selected_global]
                + float(beta_time) * delta_years
                + base_pred_arr[int(i)]
                - base_pred_arr[selected_global]
            )
        else:
            vals = base_residuals_arr[selected_global]

        out[int(i)] = float(np.sum(weights * vals) / weights_sum)
        n_neighbors_used[int(i)] = k_eff
        mean_space_miles[int(i)] = float(np.sum(weights * d_space[selected_local]) / weights_sum)
        mean_time_days[int(i)] = float(np.sum(weights * d_time[selected_local]) / weights_sum)

        if row_counter > 0 and row_counter % 10000 == 0:
            log(
                "spatial feature progress",
                targets_done=row_counter,
                targets_total=len(target_indices),
                feature_type=feature_type,
                K=K,
            )

    diagnostics = {
        "n_neighbors_used": n_neighbors_used,
        "mean_neighbor_space_miles": mean_space_miles,
        "mean_neighbor_time_days": mean_time_days,
        "used_class_fallback": used_class_fallback,
        "candidate_count": candidate_count,
    }
    return out, diagnostics


def _compute_spatial_time_lag_dense(
    df: pd.DataFrame,
    target_indices: Sequence[int],
    history_indices: Sequence[int],
    *,
    K: int,
    spatial_bw_miles: float,
    time_bw_days: float,
    feature_type: str,
    beta_time: float,
    base_residuals: Optional[np.ndarray],
    base_pred: Optional[np.ndarray] = None,
    group_filter_col: Optional[str] = None,
    class_filter_col: Optional[str] = None,
    min_same_class_pool: int = 10,
    allow_class_fallback: bool = True,
    max_neighbor_age_days: Optional[float] = None,
    chunk_size: int = 500,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Original dense implementation, kept for validation/debugging."""

    valid_feature_types = {"raw_price", "time_adjusted_price", "residual", "scaled_price", "scaled_time_adjusted_price"}
    if feature_type not in valid_feature_types:
        raise ValueError(f"feature_type must be one of {sorted(valid_feature_types)}.")
    if feature_type == "residual" and base_residuals is None:
        raise ValueError("base_residuals are required for residual spatial lag.")
    if feature_type in {"scaled_price", "scaled_time_adjusted_price"} and base_pred is None:
        raise ValueError("base_pred is required for baseline-scaled spatial price lags.")

    n = len(df)
    out = np.full(n, np.nan, dtype=float)
    n_neighbors_used = np.zeros(n, dtype=int)
    mean_space_miles = np.full(n, np.nan, dtype=float)
    mean_time_days = np.full(n, np.nan, dtype=float)
    used_class_fallback = np.zeros(n, dtype=bool)

    target_indices = np.asarray(target_indices, dtype=int)
    history_indices = np.asarray(history_indices, dtype=int)

    coords = _spatial_xy_miles(df)
    sale_day = df["sale_day"].to_numpy(dtype=float)
    sale_year_float = df["sale_year_float"].to_numpy(dtype=float)
    y_log = df["y_log"].to_numpy(dtype=float)
    base_pred_arr = None if base_pred is None else np.asarray(base_pred, dtype=float)

    group_vals = None
    if group_filter_col and group_filter_col in df.columns:
        group_vals = values_as_str_with_na(df[group_filter_col])

    class_vals = None
    if class_filter_col and class_filter_col in df.columns:
        class_vals = values_as_str_with_na(df[class_filter_col])

    for start in range(0, len(target_indices), int(chunk_size)):
        chunk_idx = target_indices[start:start + int(chunk_size)]
        dx = coords[chunk_idx, 0][:, None] - coords[history_indices, 0][None, :]
        dy = coords[chunk_idx, 1][:, None] - coords[history_indices, 1][None, :]
        d_space = np.sqrt(dx * dx + dy * dy)
        d_time = sale_day[chunk_idx, None] - sale_day[history_indices][None, :]

        base_valid = d_time > 0.0
        if max_neighbor_age_days is not None and float(max_neighbor_age_days) > 0.0:
            base_valid &= d_time <= float(max_neighbor_age_days)
        if group_vals is not None:
            base_valid &= group_vals[chunk_idx, None] == group_vals[history_indices][None, :]

        score = d_space / float(spatial_bw_miles) + d_time / float(time_bw_days)

        for row_pos, i in enumerate(chunk_idx):
            valid = base_valid[row_pos].copy()
            if class_vals is not None:
                same_class_valid = valid & (class_vals[int(i)] == class_vals[history_indices])
                if same_class_valid.sum() >= min(int(K), int(min_same_class_pool)):
                    valid = same_class_valid
                elif not allow_class_fallback:
                    valid = same_class_valid
                else:
                    used_class_fallback[int(i)] = True

            row_score = np.where(valid, score[row_pos], np.inf)
            valid_positions = np.where(np.isfinite(row_score))[0]
            if valid_positions.size == 0:
                continue

            k_eff = min(int(K), int(valid_positions.size))
            selected_local = valid_positions[np.argpartition(row_score[valid_positions], k_eff - 1)[:k_eff]]
            selected_global = history_indices[selected_local]
            selected_score = row_score[selected_local]

            weights = np.exp(-selected_score)
            weights_sum = float(weights.sum())
            if weights_sum <= 0.0 or not np.isfinite(weights_sum):
                continue

            if feature_type == "raw_price":
                vals = y_log[selected_global]
            elif feature_type == "time_adjusted_price":
                delta_years = sale_year_float[int(i)] - sale_year_float[selected_global]
                vals = y_log[selected_global] + float(beta_time) * delta_years
            elif feature_type == "scaled_price":
                vals = y_log[selected_global] + base_pred_arr[int(i)] - base_pred_arr[selected_global]
            elif feature_type == "scaled_time_adjusted_price":
                delta_years = sale_year_float[int(i)] - sale_year_float[selected_global]
                vals = (
                    y_log[selected_global]
                    + float(beta_time) * delta_years
                    + base_pred_arr[int(i)]
                    - base_pred_arr[selected_global]
                )
            else:
                vals = np.asarray(base_residuals, dtype=float)[selected_global]

            out[int(i)] = float(np.sum(weights * vals) / weights_sum)
            n_neighbors_used[int(i)] = k_eff
            mean_space_miles[int(i)] = float(np.sum(weights * d_space[row_pos, selected_local]) / weights_sum)
            mean_time_days[int(i)] = float(np.sum(weights * d_time[row_pos, selected_local]) / weights_sum)

    diagnostics = {
        "n_neighbors_used": n_neighbors_used,
        "mean_neighbor_space_miles": mean_space_miles,
        "mean_neighbor_time_days": mean_time_days,
        "used_class_fallback": used_class_fallback,
        "candidate_count": np.full(n, np.nan),
    }
    return out, diagnostics


# -----------------------------------------------------------------------------
# Experiment runner for one split
# -----------------------------------------------------------------------------

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
    y_train_log = train_df["y_log"].to_numpy(dtype=float)
    model.fit(train_df[pred_vars].copy(), y_train_log)
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
    spatial_bandwidths_miles: Sequence[float],
    time_bandwidths_days: Sequence[float],
    feature_types: Sequence[str],
    group_filter_col: str,
    class_filter_col: str,
    min_same_class_pool: int,
    allow_class_fallback: bool,
    max_neighbor_age_days: Optional[float],
    allow_sequential_eval_history: bool,
    chunk_size: int,
    include_full_x_residual: bool,
    search_method: str,
    max_spatial_candidates: int,
    candidate_multiplier: int,
    n_jobs: int,
    include_spatial_first: bool,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], pd.DataFrame]:
    """Run baseline and all spatial experiments for a single train/eval split."""

    if eval_df_raw.empty:
        log("split skipped: empty eval data", split=split_label)
        return pd.DataFrame(), {}, pd.DataFrame()

    train_df = add_spatial_time_columns(train_df_raw).copy()
    eval_df = add_spatial_time_columns(eval_df_raw).copy()
    combined = pd.concat([train_df, eval_df], ignore_index=True)
    combined = combined.sort_values(DATE_COL).reset_index(drop=True)
    combined = cast_categoricals(combined, categorical_cols)

    # Reconstruct train/eval indices after sorting.  Use a temporary stable row id.
    train_df_tmp = train_df.copy()
    eval_df_tmp = eval_df.copy()
    train_df_tmp["__split_row_id"] = [f"train_{i}" for i in range(len(train_df_tmp))]
    eval_df_tmp["__split_row_id"] = [f"eval_{i}" for i in range(len(eval_df_tmp))]
    combined = pd.concat([train_df_tmp, eval_df_tmp], ignore_index=True).sort_values(DATE_COL).reset_index(drop=True)
    combined = cast_categoricals(combined, categorical_cols)
    train_idx = np.where(combined["__split_row_id"].astype(str).str.startswith("train_"))[0]
    eval_idx = np.where(combined["__split_row_id"].astype(str).str.startswith("eval_"))[0]

    # For model fitting, use the original train/eval row sets in sorted combined order.
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

    # ------------------------------------------------------------------
    # Full-X baseline
    # ------------------------------------------------------------------
    base_pred_eval, base_model = fit_predict_full_x(
        train_df=train_for_model,
        eval_df=eval_for_model,
        pred_vars=predictor_cols,
        cat_vars=categorical_cols,
        id_vars=id_vars,
        estimator_name=estimator_name,
        ridge_alpha=ridge_alpha,
    )

    # Predict combined rows for base residuals.  Residuals are only consumed from
    # history rows unless allow_sequential_eval_history is True.
    combined["base_pred"] = base_model.predict(combined[predictor_cols].copy())
    combined["base_residual"] = combined["y_log"].to_numpy(dtype=float) - combined["base_pred"].to_numpy(dtype=float)

    rows: List[Dict[str, Any]] = []
    pred_logs: Dict[str, np.ndarray] = {}

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
            "feature_type": "none",
            "K": np.nan,
            "spatial_bw_miles": np.nan,
            "time_bw_days": np.nan,
            "valid_train_rows": int(len(train_idx)),
            "valid_eval_rows": int(len(eval_idx)),
            "mean_eval_neighbors": np.nan,
            "mean_eval_neighbor_space_miles": np.nan,
            "mean_eval_neighbor_time_days": np.nan,
            "class_fallback_share_eval": np.nan,
            "allow_sequential_eval_history": bool(allow_sequential_eval_history),
            "stage_order": "baseline",
        }
    )
    rows.append(base_metrics)
    pred_logs[base_label] = base_pred_eval
    log(
        "baseline finished",
        split=split_label,
        r2_log=f"{base_metrics['r2_log']:.4f}",
        rmse_log=f"{base_metrics['rmse_log']:.4f}",
        r2_price=f"{base_metrics['r2_price']:.4f}",
        rmse_price=f"{base_metrics['rmse_price']:.0f}",
        median_ratio=f"{base_metrics['median_ratio']:.4f}",
    )

    beta_time = estimate_global_time_slope(train_for_model)
    log("train-only time slope estimated", split=split_label, beta=f"{beta_time:.5f}", growth=f"{math.exp(beta_time)-1.0:.2%}")

    if allow_sequential_eval_history:
        eval_history_idx = np.arange(len(combined), dtype=int)
    else:
        eval_history_idx = train_idx
    train_history_idx = train_idx

    # ------------------------------------------------------------------
    # Spatial candidate caches
    # ------------------------------------------------------------------
    candidate_caches: Dict[Tuple[str, int], _SpatialCandidateCache] = {}
    if search_method == "kdtree":
        for K_cache in Ks:
            K_cache_int = int(K_cache)
            log(
                "candidate cache start",
                split=split_label,
                side="train",
                K=K_cache_int,
                targets=len(train_idx),
                history=len(train_history_idx),
                n_jobs=n_jobs,
            )
            candidate_caches[("train", K_cache_int)] = precompute_spatial_candidate_cache(
                df=combined,
                target_indices=train_idx,
                history_indices=train_history_idx,
                K=K_cache_int,
                group_filter_col=group_filter_col,
                class_filter_col=class_filter_col,
                min_same_class_pool=int(min_same_class_pool),
                allow_class_fallback=bool(allow_class_fallback),
                max_neighbor_age_days=max_neighbor_age_days,
                max_spatial_candidates=int(max_spatial_candidates),
                candidate_multiplier=int(candidate_multiplier),
                n_jobs=int(n_jobs),
                progress_label=f"{split_label}:train:K{K_cache_int}",
            )

            log(
                "candidate cache finished",
                split=split_label,
                side="train",
                K=K_cache_int,
                # Using an f-string here removes the need for multiple nested parentheses 
                mean_candidates=f"{np.mean(candidate_caches[('train', K_cache_int)].candidate_count_local):.1f}",
                # mean_candidates=f"{np.mean(candidate_caches[('eval', K_cache_int)].candidate_count_local):.1f}",
            )

            log(
                "candidate cache start",
                split=split_label,
                side="eval",
                K=K_cache_int,
                targets=len(eval_idx),
                history=len(eval_history_idx),
                n_jobs=n_jobs,
            )

            candidate_caches[("eval", K_cache_int)] = precompute_spatial_candidate_cache(
                df=combined,
                target_indices=eval_idx,
                history_indices=eval_history_idx,
                K=K_cache_int,
                group_filter_col=group_filter_col,
                class_filter_col=class_filter_col,
                min_same_class_pool=int(min_same_class_pool),
                allow_class_fallback=bool(allow_class_fallback),
                max_neighbor_age_days=max_neighbor_age_days,
                max_spatial_candidates=int(max_spatial_candidates),
                candidate_multiplier=int(candidate_multiplier),
                n_jobs=int(n_jobs),
                progress_label=f"{split_label}:eval:K{K_cache_int}",
            )
            log(
                "candidate cache finished",
                split=split_label,
                side="eval",
                K=K_cache_int,
                # mean_candidates=f"{np.mean(candidate_caches[("eval", K_cache_int)].candidate_count_local):.1f}",
                mean_candidates=f"{np.mean(candidate_caches[('eval', K_cache_int)].candidate_count_local):.1f}",
            )

    # ------------------------------------------------------------------
    # Spatial grids
    # ------------------------------------------------------------------
    for K in Ks:
        for h_space in spatial_bandwidths_miles:
            for h_time in time_bandwidths_days:
                for feature_type in feature_types:
                    feature_type = str(feature_type)
                    log(
                        "spatial feature start",
                        split=split_label,
                        feature_type=feature_type,
                        K=K,
                        h_space=h_space,
                        h_time=h_time,
                    )

                    if search_method == "kdtree":
                        train_feat, train_diag = compute_spatial_time_lag_from_cache(
                            combined,
                            candidate_caches[("train", int(K))],
                            K=int(K),
                            spatial_bw_miles=float(h_space),
                            time_bw_days=float(h_time),
                            feature_type=feature_type,
                            beta_time=beta_time,
                            base_residuals=combined["base_residual"].to_numpy(dtype=float),
                            base_pred=combined["base_pred"].to_numpy(dtype=float),
                        )
                        eval_feat, eval_diag = compute_spatial_time_lag_from_cache(
                            combined,
                            candidate_caches[("eval", int(K))],
                            K=int(K),
                            spatial_bw_miles=float(h_space),
                            time_bw_days=float(h_time),
                            feature_type=feature_type,
                            beta_time=beta_time,
                            base_residuals=combined["base_residual"].to_numpy(dtype=float),
                            base_pred=combined["base_pred"].to_numpy(dtype=float),
                        )
                    else:
                        train_feat, train_diag = compute_spatial_time_lag(
                            combined,
                            target_indices=train_idx,
                            history_indices=train_history_idx,
                            K=int(K),
                            spatial_bw_miles=float(h_space),
                            time_bw_days=float(h_time),
                            feature_type=feature_type,
                            beta_time=beta_time,
                            base_residuals=combined["base_residual"].to_numpy(dtype=float),
                            base_pred=combined["base_pred"].to_numpy(dtype=float),
                            group_filter_col=group_filter_col,
                            class_filter_col=class_filter_col,
                            min_same_class_pool=int(min_same_class_pool),
                            allow_class_fallback=bool(allow_class_fallback),
                            max_neighbor_age_days=max_neighbor_age_days,
                            chunk_size=int(chunk_size),
                            search_method=search_method,
                            max_spatial_candidates=int(max_spatial_candidates),
                            candidate_multiplier=int(candidate_multiplier),
                        )
                        eval_feat, eval_diag = compute_spatial_time_lag(
                            combined,
                            target_indices=eval_idx,
                            history_indices=eval_history_idx,
                            K=int(K),
                            spatial_bw_miles=float(h_space),
                            time_bw_days=float(h_time),
                            feature_type=feature_type,
                            beta_time=beta_time,
                            base_residuals=combined["base_residual"].to_numpy(dtype=float),
                            base_pred=combined["base_pred"].to_numpy(dtype=float),
                            group_filter_col=group_filter_col,
                            class_filter_col=class_filter_col,
                            min_same_class_pool=int(min_same_class_pool),
                            allow_class_fallback=bool(allow_class_fallback),
                            max_neighbor_age_days=max_neighbor_age_days,
                            chunk_size=int(chunk_size),
                            search_method=search_method,
                            max_spatial_candidates=int(max_spatial_candidates),
                            candidate_multiplier=int(candidate_multiplier),
                        )

                    feature_col = f"st_{feature_type}_K{K}_hs{h_space}_ht{h_time}"

                    # Keep spatial features/diagnostics as NumPy arrays instead of
                    # repeatedly inserting columns into `combined`. Repeated
                    # DataFrame column insertion causes pandas fragmentation and
                    # triggers PerformanceWarning on large grids.
                    feature_values = np.full(len(combined), np.nan, dtype=float)
                    feature_values[train_idx] = train_feat[train_idx]
                    feature_values[eval_idx] = eval_feat[eval_idx]

                    n_neighbors_values = np.zeros(len(combined), dtype=float)
                    n_neighbors_values[train_idx] = train_diag["n_neighbors_used"][train_idx]
                    n_neighbors_values[eval_idx] = eval_diag["n_neighbors_used"][eval_idx]

                    mean_space_values = np.full(len(combined), np.nan, dtype=float)
                    mean_space_values[train_idx] = train_diag["mean_neighbor_space_miles"][train_idx]
                    mean_space_values[eval_idx] = eval_diag["mean_neighbor_space_miles"][eval_idx]

                    mean_time_values = np.full(len(combined), np.nan, dtype=float)
                    mean_time_values[train_idx] = train_diag["mean_neighbor_time_days"][train_idx]
                    mean_time_values[eval_idx] = eval_diag["mean_neighbor_time_days"][eval_idx]

                    fallback_values = np.zeros(len(combined), dtype=bool)
                    fallback_values[train_idx] = train_diag["used_class_fallback"][train_idx]
                    fallback_values[eval_idx] = eval_diag["used_class_fallback"][eval_idx]

                    train_valid = train_idx[np.isfinite(feature_values[train_idx])]
                    eval_valid = eval_idx[np.isfinite(feature_values[eval_idx])]

                    if len(train_valid) < 50 or len(eval_valid) < 50:
                        log("spatial model skipped: too few valid rows", split=split_label, train_valid=len(train_valid), eval_valid=len(eval_valid))
                        continue

                    def _model_frame_with_feature(indices: np.ndarray) -> pd.DataFrame:
                        """Fresh, defragmented frame with one spatial feature column."""
                        frame = combined.iloc[indices].copy()
                        frame[feature_col] = feature_values[indices]
                        return frame

                    # --------------------------------------------------
                    # Current experiment form, adapted to full-X preprocessing:
                    #   raw/time-adjusted: full X + local price signal
                    #   residual: base_pred + local residual signal
                    # --------------------------------------------------
                    candidate_specs: List[Tuple[str, List[str], List[str], pd.DataFrame, pd.DataFrame]] = []

                    if feature_type in {"raw_price", "time_adjusted_price", "scaled_price", "scaled_time_adjusted_price"}:
                        pred_vars_spatial = list(predictor_cols) + [feature_col]
                        cat_vars_spatial = [c for c in categorical_cols if c in pred_vars_spatial]
                        label = f"{split_label}_full_X_plus_{feature_type}_K{K}_hs{h_space}_ht{h_time}"
                        candidate_specs.append((
                            label,
                            pred_vars_spatial,
                            cat_vars_spatial,
                            _model_frame_with_feature(train_valid),
                            _model_frame_with_feature(eval_valid),
                        ))

                    elif feature_type == "residual":
                        # Preserve the current residual-correction idea:
                        # y_hat is learned from base_pred + local residual avg.
                        label = f"{split_label}_base_plus_local_residual_K{K}_hs{h_space}_ht{h_time}"
                        residual_train = _model_frame_with_feature(train_valid)
                        residual_eval = _model_frame_with_feature(eval_valid)
                        residual_pred_vars = ["base_pred", feature_col]
                        candidate_specs.append((label, residual_pred_vars, [], residual_train, residual_eval))

                        # Optional diagnostic: also test full X + residual lag.
                        if include_full_x_residual:
                            label2 = f"{split_label}_full_X_plus_residual_K{K}_hs{h_space}_ht{h_time}"
                            pred_vars_spatial = list(predictor_cols) + [feature_col]
                            cat_vars_spatial = [c for c in categorical_cols if c in pred_vars_spatial]
                            candidate_specs.append((label2, pred_vars_spatial, cat_vars_spatial, residual_train, residual_eval))

                    def _common_metric_fields(label: str, stage_order: str) -> Dict[str, Any]:
                        return {
                            "split": split_label,
                            "model_family": label.replace(f"{split_label}_", "").rsplit("_K", 1)[0],
                            "feature_type": feature_type,
                            "K": int(K),
                            "spatial_bw_miles": float(h_space),
                            "time_bw_days": float(h_time),
                            "valid_train_rows": int(len(train_valid)),
                            "valid_eval_rows": int(len(eval_valid)),
                            "mean_eval_neighbors": float(np.nanmean(n_neighbors_values[eval_valid])),
                            "mean_eval_neighbor_space_miles": float(np.nanmean(mean_space_values[eval_valid])),
                            "mean_eval_neighbor_time_days": float(np.nanmean(mean_time_values[eval_valid])),
                            "class_fallback_share_eval": float(np.mean(fallback_values[eval_valid])),
                            "search_method": search_method,
                            "max_spatial_candidates": int(max_spatial_candidates),
                            "candidate_multiplier": int(candidate_multiplier),
                            "allow_sequential_eval_history": bool(allow_sequential_eval_history),
                            "beta_time_train": float(beta_time),
                            "annual_growth_implied": float(np.exp(beta_time) - 1.0),
                            "group_filter_col": group_filter_col,
                            "class_filter_col": class_filter_col,
                            "min_same_class_pool": int(min_same_class_pool),
                            "allow_class_fallback": bool(allow_class_fallback),
                            "max_neighbor_age_days": max_neighbor_age_days if max_neighbor_age_days is not None else np.nan,
                            "stage_order": stage_order,
                        }

                    for label, pred_vars_model, cat_vars_model, train_model_df, eval_model_df in candidate_specs:
                        y_train_model = train_model_df["y_log"].to_numpy(dtype=float)
                        y_eval_model = eval_model_df["y_log"].to_numpy(dtype=float)
                        pred_eval, _ = fit_predict_full_x(
                            train_df=train_model_df,
                            eval_df=eval_model_df,
                            pred_vars=pred_vars_model,
                            cat_vars=cat_vars_model,
                            id_vars=id_vars,
                            estimator_name=estimator_name,
                            ridge_alpha=ridge_alpha,
                        )

                        metrics = evaluate_predictions(
                            y_true_log=y_eval_model,
                            y_pred_log=pred_eval,
                            y_train_log=y_train_model,
                            label=label,
                        )
                        if feature_type in {"raw_price", "time_adjusted_price", "scaled_price", "scaled_time_adjusted_price"}:
                            stage_order = "full_X_plus_spatial_lag"
                        elif feature_type == "residual" and label.startswith(f"{split_label}_base_plus_local_residual"):
                            stage_order = "full_X_first_spatial_residual"
                        else:
                            stage_order = "full_X_plus_spatial_residual"
                        metrics.update(_common_metric_fields(label, stage_order))
                        rows.append(metrics)
                        pred_logs[label] = pred_eval
                        log(
                            "spatial model finished",
                            split=split_label,
                            model=label,
                            r2_log=f"{metrics['r2_log']:.4f}",
                            rmse_log=f"{metrics['rmse_log']:.4f}",
                            r2_price=f"{metrics['r2_price']:.4f}",
                            rmse_price=f"{metrics['rmse_price']:.0f}",
                            median_ratio=f"{metrics['median_ratio']:.4f}",
                            slope=f"{metrics['log_ratio_slope_vs_log_price']:.5f}",
                        )

                    # Alternative direction:
                    # Stage 1: spatial lag only captures nearby trend/comps.
                    # Stage 2: full-X model predicts the remaining residual.
                    # Final prediction = calibrated spatial prediction + X residual correction.
                    if include_spatial_first and feature_type in {"raw_price", "time_adjusted_price"}:
                        sf_label = f"{split_label}_spatial_first_{feature_type}_plus_X_residual_K{K}_hs{h_space}_ht{h_time}"
                        sf_train = _model_frame_with_feature(train_valid)
                        sf_eval = _model_frame_with_feature(eval_valid)

                        y_train_sf = sf_train["y_log"].to_numpy(dtype=float)
                        y_eval_sf = sf_eval["y_log"].to_numpy(dtype=float)

                        spatial_stage = LinearRegression(fit_intercept=True)
                        spatial_stage.fit(sf_train[[feature_col]].to_numpy(dtype=float), y_train_sf)
                        spatial_pred_train = np.asarray(
                            spatial_stage.predict(sf_train[[feature_col]].to_numpy(dtype=float)),
                            dtype=float,
                        ).reshape(-1)
                        spatial_pred_eval = np.asarray(
                            spatial_stage.predict(sf_eval[[feature_col]].to_numpy(dtype=float)),
                            dtype=float,
                        ).reshape(-1)

                        residual_target_train = y_train_sf - spatial_pred_train
                        x_residual_model = PreprocessedLinearModel(
                            pred_vars=list(predictor_cols),
                            cat_vars=[c for c in categorical_cols if c in predictor_cols],
                            id_vars=[c for c in id_vars if c in predictor_cols],
                            estimator_name=estimator_name,
                            ridge_alpha=ridge_alpha,
                        )
                        x_residual_model.fit(sf_train[predictor_cols].copy(), residual_target_train)
                        residual_pred_eval = x_residual_model.predict(sf_eval[predictor_cols].copy())

                        pred_eval_sf = spatial_pred_eval + residual_pred_eval
                        sf_metrics = evaluate_predictions(
                            y_true_log=y_eval_sf,
                            y_pred_log=pred_eval_sf,
                            y_train_log=y_train_sf,
                            label=sf_label,
                        )
                        sf_metrics.update(_common_metric_fields(sf_label, "spatial_first_X_residual"))
                        rows.append(sf_metrics)
                        pred_logs[sf_label] = pred_eval_sf
                        log(
                            "spatial model finished",
                            split=split_label,
                            model=sf_label,
                            r2_log=f"{sf_metrics['r2_log']:.4f}",
                            rmse_log=f"{sf_metrics['rmse_log']:.4f}",
                            r2_price=f"{sf_metrics['r2_price']:.4f}",
                            rmse_price=f"{sf_metrics['rmse_price']:.0f}",
                            median_ratio=f"{sf_metrics['median_ratio']:.4f}",
                            slope=f"{sf_metrics['log_ratio_slope_vs_log_price']:.5f}",
                        )

    result_df = pd.DataFrame(rows)
    if not result_df.empty:
        result_df = result_df.sort_values(["rmse_log", "log_ratio_slope_vs_log_price"], ascending=[True, False]).reset_index(drop=True)
    return result_df, pred_logs, combined


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------

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
    ax.set_title(f"Top spatial/full-X models by RMSE_log — {split_label}")
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / f"{sanitize_filename(split_label.lower())}_top_rmse.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    if "rmse_price" in df.columns:
        price_plot_df = df.sort_values("rmse_price").head(15).copy()
        price_labels = price_plot_df["model"].astype(str).str.replace(f"{split_label}_", "", regex=False).to_list()
        price_labels = [s if len(s) <= 64 else s[:61] + "..." for s in price_labels]

        fig, ax = plt.subplots(figsize=(11, max(5, 0.35 * len(price_plot_df) + 1.5)))
        y_pos = np.arange(len(price_plot_df))
        ax.barh(y_pos, price_plot_df["rmse_price"].to_numpy(dtype=float))
        ax.set_yticks(y_pos)
        ax.set_yticklabels(price_labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("RMSE on original sale-price scale (lower is better)")
        ax.set_title(f"Top spatial/full-X models by price-scale RMSE — {split_label}")
        ax.grid(axis="x", linestyle=":", alpha=0.4)
        fig.tight_layout()
        fig.savefig(out_dir / f"{sanitize_filename(split_label.lower())}_top_rmse_price.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    grid_df = df.loc[df["feature_type"].astype(str) != "none", :].copy()
    if not grid_df.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        x = grid_df["rmse_log"].to_numpy(dtype=float)
        y = grid_df["log_ratio_slope_vs_log_price"].to_numpy(dtype=float)
        ax.scatter(x, y, alpha=0.7)
        ax.axhline(0.0, linestyle="--", linewidth=1.0)
        ax.set_xlabel("RMSE_log (lower is better)")
        ax.set_ylabel("log-ratio slope vs log price (closer to 0 is better)")
        ax.set_title(f"Accuracy vs vertical-equity slope — {split_label}")
        ax.grid(True, linestyle=":", alpha=0.4)
        fig.tight_layout()
        fig.savefig(out_dir / f"{sanitize_filename(split_label.lower())}_rmse_scatter.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

        if "rmse_price" in grid_df.columns and "r2_price" in grid_df.columns:
            fig, ax = plt.subplots(figsize=(8, 6))
            x = grid_df["rmse_price"].to_numpy(dtype=float)
            y = grid_df["r2_price"].to_numpy(dtype=float)
            ax.scatter(x, y, alpha=0.7)
            ax.set_xlabel("RMSE on original sale-price scale (lower is better)")
            ax.set_ylabel("R² on original sale-price scale (higher is better)")
            ax.set_title(f"Original-scale accuracy — {split_label}")
            ax.grid(True, linestyle=":", alpha=0.4)
            fig.tight_layout()
            fig.savefig(out_dir / f"{sanitize_filename(split_label.lower())}_price_rmse_vs_r2.png", dpi=180, bbox_inches="tight")
            plt.close(fig)


def write_outputs(
    *,
    out_dir: str,
    test_df: pd.DataFrame,
    assess_df: pd.DataFrame,
) -> Dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, str] = {}
    test_path = out / "spatial_full_x_metrics_test.csv"
    assess_path = out / "spatial_full_x_metrics_assess.csv"
    all_path = out / "spatial_full_x_metrics_all.csv"

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


# -----------------------------------------------------------------------------
# Main orchestration
# -----------------------------------------------------------------------------

def run_experiments(args: argparse.Namespace) -> Dict[str, str]:
    params = load_params(args.params_path)

    # Prefer char_class for class-similarity.  If absent, fallback to meta_class.
    requested_class_filter_col = args.class_filter_col
    if requested_class_filter_col.lower() == "auto":
        requested_class_filter_col = DEFAULT_CLASS_FILTER_COL

    df_train_validate, df_test, df_assess, predictor_cols, categorical_cols, id_vars = load_and_split_data(
        data_path=args.data_path,
        params=params,
        sample_frac=args.sample_frac,
        sample_seed=args.seed,
        parquet_engine=args.parquet_engine,
        group_filter_col=args.group_filter_col,
        class_filter_col=requested_class_filter_col,
    )

    # Fallback if char_class was requested but not available in the loaded data.
    class_filter_col = requested_class_filter_col
    if class_filter_col not in df_train_validate.columns and META_CLASS_COL in df_train_validate.columns:
        log("class filter fallback", requested=requested_class_filter_col, using=META_CLASS_COL)
        class_filter_col = META_CLASS_COL

    feature_types = tuple(x.strip() for x in args.feature_types.split(",") if x.strip())
    Ks = parse_int_list(args.Ks)
    spatial_bandwidths = parse_float_list(args.spatial_bandwidths_miles)
    time_bandwidths = parse_float_list(args.time_bandwidths_days)

    # TEST: train on early pre-2024, evaluate on held-out later pre-2024.
    test_results, _, _ = run_one_split(
        split_label="test",
        train_df_raw=df_train_validate,
        eval_df_raw=df_test,
        predictor_cols=predictor_cols,
        categorical_cols=categorical_cols,
        id_vars=id_vars,
        estimator_name=args.estimator,
        ridge_alpha=args.ridge_alpha,
        Ks=Ks,
        spatial_bandwidths_miles=spatial_bandwidths,
        time_bandwidths_days=time_bandwidths,
        feature_types=feature_types,
        group_filter_col=args.group_filter_col,
        class_filter_col=class_filter_col,
        min_same_class_pool=args.min_same_class_pool,
        allow_class_fallback=not args.no_class_fallback,
        max_neighbor_age_days=args.max_neighbor_age_days,
        allow_sequential_eval_history=args.allow_sequential_eval_history,
        chunk_size=args.chunk_size,
        include_full_x_residual=args.include_full_x_residual,
        search_method=args.search_method,
        max_spatial_candidates=args.max_spatial_candidates,
        candidate_multiplier=args.candidate_multiplier,
        n_jobs=args.n_jobs,
        include_spatial_first=not args.no_spatial_first,
    )

    # ASSESS: train on all pre-2024 sales, evaluate on 2024 sales.
    assess_results = pd.DataFrame()
    if not df_assess.empty:
        df_pre2024 = pd.concat([df_train_validate, df_test], ignore_index=True)
        assess_results, _, _ = run_one_split(
            split_label="assess",
            train_df_raw=df_pre2024,
            eval_df_raw=df_assess,
            predictor_cols=predictor_cols,
            categorical_cols=categorical_cols,
            id_vars=id_vars,
            estimator_name=args.estimator,
            ridge_alpha=args.ridge_alpha,
            Ks=Ks,
            spatial_bandwidths_miles=spatial_bandwidths,
            time_bandwidths_days=time_bandwidths,
            feature_types=feature_types,
            group_filter_col=args.group_filter_col,
            class_filter_col=class_filter_col,
            min_same_class_pool=args.min_same_class_pool,
            allow_class_fallback=not args.no_class_fallback,
            max_neighbor_age_days=args.max_neighbor_age_days,
            allow_sequential_eval_history=args.allow_sequential_eval_history,
            chunk_size=args.chunk_size,
            include_full_x_residual=args.include_full_x_residual,
            search_method=args.search_method,
            max_spatial_candidates=args.max_spatial_candidates,
            candidate_multiplier=args.candidate_multiplier,
            n_jobs=args.n_jobs,
            include_spatial_first=not args.no_spatial_first,
        )

    paths = write_outputs(out_dir=args.out_dir, test_df=test_results, assess_df=assess_results)
    log("experiments finished", **paths)
    return paths


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Full-X baseline + spatial-time lag experiments.")
    p.add_argument("--data-path", default="./data/CCAO/2025/training_data.parquet")
    p.add_argument("--params-path", default="params.yaml")
    p.add_argument("--out-dir", default="./outputs/spatial_full_x")
    p.add_argument("--parquet-engine", default="pyarrow", choices=["pyarrow", "fastparquet", "auto"])
    p.add_argument("--sample-frac", type=float, default=None, help="Optional row sample fraction for fast tests.")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--estimator", default="linear", choices=["linear", "ridge"])
    p.add_argument("--ridge-alpha", type=float, default=1.0)

    # Defaults are concentrated around the best region from the previous run,
    # but still include enough grid to verify the trend.
    p.add_argument("--Ks", default="5,10")
    p.add_argument("--spatial-bandwidths-miles", default="0.5,1.0,2.0")
    p.add_argument("--time-bandwidths-days", default="180,365,730")
    p.add_argument("--feature-types", default="time_adjusted_price,residual,scaled_time_adjusted_price,raw_price")

    p.add_argument("--group-filter-col", default=DEFAULT_GROUP_FILTER_COL)
    p.add_argument("--class-filter-col", default="auto", help="Use 'auto' for char_class with fallback to meta_class.")
    p.add_argument("--min-same-class-pool", type=int, default=10)
    p.add_argument("--no-class-fallback", action="store_true", help="If set, require same class even when sparse.")
    p.add_argument("--max-neighbor-age-days", type=float, default=None, help="Optional hard age cutoff in addition to time bandwidth.")
    p.add_argument("--allow-sequential-eval-history", action="store_true", help="Allow earlier eval-period sales as neighbors. Default is strict holdout.")
    p.add_argument("--chunk-size", type=int, default=500, help="Used only by --search-method dense.")
    p.add_argument("--search-method", default="kdtree", choices=["kdtree", "dense"], help="kdtree is fast bounded-candidate search; dense reproduces the original exact matrix search.")
    p.add_argument("--max-spatial-candidates", type=int, default=4096, help="Adaptive upper cap for KDTree spatial candidates per target. The query expands after time/class filtering until enough valid prior comps are found or this cap is reached.")
    p.add_argument("--candidate-multiplier", type=int, default=64, help="Initial KDTree candidates ~= K * multiplier. Adaptive expansion doubles from there up to --max-spatial-candidates when temporal filtering leaves too few comps.")
    p.add_argument("--n-jobs", type=int, default=1, help="Threads for KDTree candidate-cache precomputation. Use 0 for all available CPUs; 4-16 is usually enough on shared nodes.")
    p.add_argument("--include-full-x-residual", action="store_true", help="Also test X + residual lag, in addition to base_pred + residual lag.")
    p.add_argument("--no-spatial-first", action="store_true", help="Disable the alternative spatial-first model: spatial lag baseline + full-X residual correction.")
    return p


if __name__ == "__main__":
    parser = build_arg_parser()
    run_experiments(parser.parse_args())
