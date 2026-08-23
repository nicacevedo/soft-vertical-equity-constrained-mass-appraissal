"""
Final market-value-proxy / spatial-comparable / AVM experiment runner.

Purpose
-------
This script implements a leak-safe experiment grid with three explicit layers:

1) Target layer: construct alternative training labels from raw sale prices.
   - raw sale log price
   - global time-slope adjusted label
   - area-specific shrinkage time-slope adjusted label
   - hedonic-partial global/area time-slope adjusted label

2) Spatial comparable layer: construct deployment-safe spatial-time features from
   prior comparable sales only.
   - raw prior price lag
   - time-adjusted prior price lag
   - target-label lag
   - scaled target-label lag
   - scaled time-adjusted prior price lag
   - local residual lag from cross-fitted base-model residuals

3) AVM/calibration layer: train base and spatial-augmented models, then optionally
   calibrate predictions using a strictly held-out calibration slice from the
   training window.

This file is designed for the CCAO-style dataset/params.yaml described in the
conversation, but it is intentionally self-contained: it does not require the
repo preprocessing code. It uses sklearn preprocessing and optionally LightGBM.

Recommended first quick run
---------------------------
python final_market_value_spatial_experiments.py \
  --data-path ./data/CCAO/2025/training_data.parquet \
  --params-path ./params.yaml \
  --out-dir ./outputs/final_mv_spatial \
  --sample-frac 0.10 \
  --target-variants raw,global_slope,area_slope:meta_township_code,hedonic_global_slope,hedonic_area_slope:meta_township_code \
  --learners ridge,lgbm_l2,lgbm_l1 \
  --spatial-feature-types scaled_time_adjusted_price,residual,time_adjusted_price,scaled_target_label \
  --Ks 10,20,30 \
  --spatial-bandwidths-miles 0.75,1.0,1.25 \
  --time-bandwidths-days 365,500,730 \
  --calibration-modes none,median_center,affine \
  --calib-frac 0.20 \
  --n-jobs 8

Important leakage rules implemented
-----------------------------------
- Target adjustment is fit on the model-training core only.
- Calibration uses a trailing slice of the training window, not the eval window.
- Spatial features for eval/calibration use only prior sales from the model-training core by default.
- Spatial features for training rows use prior rows only, and residual features use cross-fitted residuals.
- Group filtering is off by default; class filtering is on by default.
- Precomputed spatial_lag_* columns and sale-specific validation metadata are removed from the final AVM feature list.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KDTree
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

try:
    from joblib import Parallel, delayed
except Exception:  # pragma: no cover
    Parallel = None
    delayed = None

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
    category=UserWarning,
)


# =============================================================================
# Constants
# =============================================================================

TARGET_COL = "meta_sale_price"
DATE_COL = "meta_sale_date"
LAT_COL = "loc_latitude"
LON_COL = "loc_longitude"
X_FT_COL = "loc_x_3435"
Y_FT_COL = "loc_y_3435"
META_CLASS_COL = "meta_class"
CHAR_CLASS_COL = "char_class"
TRIAD_COL = "meta_triad_name"
TOWNSHIP_COL = "meta_township_code"
NBHD_COL = "meta_nbhd_code"
EARTH_RADIUS_MILES = 3958.8
LOG_T0 = time.perf_counter()

SALE_ONLY_PREFIXES = (
    "sv_",
    "meta_mailed_",
    "meta_certified_",
    "meta_board_",
    "meta_1yr_pri_board_",
    "meta_2yr_pri_board_",
    "spatial_lag_",
)

SALE_ONLY_EXACT = {
    TARGET_COL,
    DATE_COL,
    "meta_sale_document_num",
    "meta_sale_deed_type",
    "meta_sale_buyer_name",
    "meta_sale_seller_name",
    "sv_review_json",
    "sv_run_id",
    "loc_property_address",
    "meta_pin",
    "meta_pin10",
}

# Core structural/location controls for hedonic time adjustment.  Only columns
# present in the dataset and in the predictor set will be used.
HEDONIC_CONTROL_CANDIDATES = [
    "char_class", "meta_class", "meta_township_code", "meta_nbhd_code", "meta_triad_name",
    "char_yrblt", "char_bldg_sf", "char_land_sf", "char_rooms", "char_beds",
    "char_fbath", "char_hbath", "char_frpl", "char_gar1_size", "char_bsmt",
    "char_air", "char_heat", "char_ext_wall", "char_recent_renovation",
    "loc_latitude", "loc_longitude", "loc_census_tract_geoid",
]


# =============================================================================
# Logging and parsing
# =============================================================================

def log(message: str, **fields: Any) -> None:
    dt = time.perf_counter() - LOG_T0
    suffix = ""
    if fields:
        suffix = " | " + " | ".join(f"{k}={v}" for k, v in fields.items())
    print(f"[mv_spatial +{dt:8.1f}s] {message}{suffix}", flush=True)


def parse_csv(raw: str, cast=str) -> Tuple[Any, ...]:
    if raw is None or str(raw).strip() == "":
        return tuple()
    return tuple(cast(x.strip()) for x in str(raw).split(",") if x.strip())


def parse_float_csv(raw: str) -> Tuple[float, ...]:
    return parse_csv(raw, float)


def parse_int_csv(raw: str) -> Tuple[int, ...]:
    return parse_csv(raw, int)


def safe_log_price(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if np.any(~np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError("Target contains non-finite or non-positive prices after filtering.")
    return np.log(arr)


def parse_bool_series(s: pd.Series, *, default: bool = False) -> pd.Series:
    """Parse mixed bool/numeric/string flags without treating missing values as True."""
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(default).astype(bool)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(float(default)).ne(0.0)

    true_values = {"1", "true", "t", "yes", "y"}
    false_values = {"0", "false", "f", "no", "n", ""}
    raw = s.astype("object").where(s.notna(), default).astype(str).str.strip().str.lower()
    parsed = raw.map(lambda x: True if x in true_values else False if x in false_values else bool(default))
    return parsed.astype(bool)


def year_float_from_datetime(s: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(s, errors="coerce")
    day = dt.astype("int64") / 1e9 / 86400.0
    return np.asarray(day / 365.25, dtype=float)


def timestamp_to_year_float(ts: pd.Timestamp) -> float:
    return float(ts.value / 1e9 / 86400.0 / 365.25)


def infer_eval_valuation_date(eval_df: pd.DataFrame, mode: str, assessment_date: str) -> pd.Timestamp:
    mode = str(mode).lower().strip()
    if mode == "assessment_date":
        return pd.Timestamp(assessment_date)
    dates = pd.to_datetime(eval_df[DATE_COL], errors="coerce").dropna().sort_values()
    if dates.empty:
        return pd.Timestamp(assessment_date)
    if mode == "eval_max":
        return pd.Timestamp(dates.iloc[-1])
    if mode == "eval_min":
        return pd.Timestamp(dates.iloc[0])
    if mode == "eval_median":
        return pd.Timestamp(dates.iloc[len(dates) // 2])
    raise ValueError("--valuation-date-mode must be one of assessment_date, eval_max, eval_min, eval_median")


# =============================================================================
# Params and feature handling
# =============================================================================

def load_params(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def is_forbidden_predictor(col: str) -> bool:
    if col in SALE_ONLY_EXACT:
        return True
    if any(col.startswith(p) for p in SALE_ONLY_PREFIXES):
        return True
    return False


def clean_predictor_list(predictors: Sequence[str], *, strict_feature_screen: bool = True) -> List[str]:
    out = []
    for c in predictors:
        if strict_feature_screen and is_forbidden_predictor(c):
            continue
        out.append(c)
    # Deduplicate while preserving order.
    seen = set()
    final = []
    for c in out:
        if c not in seen:
            final.append(c)
            seen.add(c)
    return final


def add_engineered_features(df: pd.DataFrame, assessment_year: int) -> pd.DataFrame:
    """Small, transparent feature additions.  Safe for assessment stage."""
    df = df.copy()
    if "char_bldg_sf" in df.columns:
        df["eng_log_bldg_sf"] = np.log1p(pd.to_numeric(df["char_bldg_sf"], errors="coerce"))
    if "char_land_sf" in df.columns:
        df["eng_log_land_sf"] = np.log1p(pd.to_numeric(df["char_land_sf"], errors="coerce"))
    if "char_yrblt" in df.columns:
        yr = pd.to_numeric(df["char_yrblt"], errors="coerce")
        age = np.maximum(0.0, float(assessment_year) - yr)
        df["eng_age_at_assessment"] = age
        df["eng_log_age1p"] = np.log1p(age)
    if "char_fbath" in df.columns or "char_hbath" in df.columns:
        fb = pd.to_numeric(df.get("char_fbath", 0.0), errors="coerce")
        hb = pd.to_numeric(df.get("char_hbath", 0.0), errors="coerce")
        df["eng_bath_total"] = fb.fillna(0.0) + 0.5 * hb.fillna(0.0)
    if "char_bldg_sf" in df.columns and "char_land_sf" in df.columns:
        b = pd.to_numeric(df["char_bldg_sf"], errors="coerce")
        l = pd.to_numeric(df["char_land_sf"], errors="coerce")
        df["eng_bldg_land_ratio"] = b / np.maximum(l, 1.0)
    for c in list(df.columns):
        if c.startswith("prox_") and c.endswith("_dist_ft"):
            df[f"eng_log1p_{c}"] = np.log1p(pd.to_numeric(df[c], errors="coerce"))
    return df


def infer_categoricals(df: pd.DataFrame, predictors: Sequence[str], params_cat: Sequence[str]) -> List[str]:
    cats = set(c for c in params_cat if c in predictors)
    for c in predictors:
        if c not in df.columns:
            continue
        dtype = df[c].dtype
        if (
            pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_string_dtype(dtype)
            or isinstance(dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(dtype)
            or not pd.api.types.is_numeric_dtype(dtype)
        ):
            cats.add(c)
    return [c for c in predictors if c in cats]


def load_model_frame(
    *,
    data_path: str,
    params: dict,
    sample_frac: Optional[float],
    sample_seed: int,
    parquet_engine: str,
    strict_feature_screen: bool,
    add_engineered: bool,
) -> Tuple[pd.DataFrame, List[str], List[str], List[str], str, int]:
    predictor_cols_raw = list(params["model"]["predictor"]["all"])
    categorical_cols_raw = list(params["model"]["predictor"].get("categorical", []))
    id_vars_raw = list(params["model"]["predictor"].get("id", []))

    assessment_year = int(params.get("assessment", {}).get("year", 2025))
    assessment_date = str(params.get("assessment", {}).get("date", f"{assessment_year}-01-01"))

    predictor_cols = clean_predictor_list(predictor_cols_raw, strict_feature_screen=strict_feature_screen)
    categorical_cols = [c for c in categorical_cols_raw if c in predictor_cols]
    id_vars = [c for c in id_vars_raw if c in predictor_cols]

    extra_cols = [
        TARGET_COL, DATE_COL, LAT_COL, LON_COL, X_FT_COL, Y_FT_COL,
        META_CLASS_COL, CHAR_CLASS_COL, TRIAD_COL, TOWNSHIP_COL, NBHD_COL,
        "ind_pin_is_multicard", "sv_is_outlier",
        "sv_outlier_reason", "sv_outlier_reason1", "sv_outlier_reason2", "sv_outlier_reason3",
        "meta_sale_deed_type", "meta_sale_count_past_n_years",
    ]
    cols_to_load = sorted(set(predictor_cols + categorical_cols + id_vars + extra_cols + HEDONIC_CONTROL_CANDIDATES))

    log("loading parquet", data_path=data_path, requested_cols=len(cols_to_load))
    try:
        df = pd.read_parquet(data_path, engine=parquet_engine, columns=cols_to_load)
    except Exception as exc:
        log("column-pruned read failed; retrying full read", error=repr(exc))
        df = pd.read_parquet(data_path, engine=parquet_engine)
        missing = [c for c in cols_to_load if c not in df.columns]
        if missing:
            log("columns missing after full read", missing=missing[:20], n_missing=len(missing))
        keep = [c for c in cols_to_load if c in df.columns]
        df = df.loc[:, keep].copy()
    log("parquet loaded", rows=df.shape[0], cols=df.shape[1])

    # Basic target/date filters.
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL, TARGET_COL]).copy()
    df = df.loc[df[TARGET_COL] > 0.0].copy()

    # Repo-style sale filters when present.
    if "ind_pin_is_multicard" in df.columns:
        df = df.loc[~parse_bool_series(df["ind_pin_is_multicard"], default=False)].copy()
    if "sv_is_outlier" in df.columns:
        df = df.loc[~parse_bool_series(df["sv_is_outlier"], default=False)].copy()
    log("basic/repo filters applied", rows=df.shape[0])

    # Coordinates are required for spatial features.  Prefer projected feet.
    has_xy = X_FT_COL in df.columns and Y_FT_COL in df.columns
    has_latlon = LAT_COL in df.columns and LON_COL in df.columns
    if has_xy and df[[X_FT_COL, Y_FT_COL]].notna().all(axis=1).any():
        df = df.dropna(subset=[X_FT_COL, Y_FT_COL]).copy()
    elif has_latlon and df[[LAT_COL, LON_COL]].notna().all(axis=1).any():
        df = df.dropna(subset=[LAT_COL, LON_COL]).copy()
    else:
        raise ValueError("Need either projected x/y or lat/lon coordinates for spatial features.")

    if add_engineered:
        df = add_engineered_features(df, assessment_year=assessment_year)
        engineered_cols = [c for c in df.columns if c.startswith("eng_")]
        predictor_cols = predictor_cols + [c for c in engineered_cols if c not in predictor_cols]

    # Keep predictors actually present.
    predictor_cols = [c for c in predictor_cols if c in df.columns]
    categorical_cols = infer_categoricals(df, predictor_cols, categorical_cols)
    id_vars = [c for c in id_vars if c in predictor_cols]

    if sample_frac is not None and float(sample_frac) < 1.0:
        df = df.sample(frac=float(sample_frac), random_state=int(sample_seed)).copy()
        log("sampling applied", sample_frac=sample_frac, rows=df.shape[0])

    df = df.sort_values(DATE_COL).reset_index(drop=True)
    df["y_raw_log"] = safe_log_price(df[TARGET_COL].to_numpy())
    df["sale_year_float"] = year_float_from_datetime(df[DATE_COL])
    df["sale_day"] = df[DATE_COL].astype("int64") / 1e9 / 86400.0

    for c in categorical_cols:
        if c in df.columns:
            df[c] = df[c].astype("category")

    log("model frame ready", rows=df.shape[0], predictors=len(predictor_cols), categoricals=len(categorical_cols))
    return df, predictor_cols, categorical_cols, id_vars, assessment_date, assessment_year


def make_quick_splits(df: pd.DataFrame, params: dict, assess_eval_year: int) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    split_prop = float(params.get("cv", {}).get("split_prop", 0.9))
    assess_df = df.loc[df[DATE_COL].dt.year == int(assess_eval_year)].copy()
    pre_assess = df.loc[df[DATE_COL].dt.year < int(assess_eval_year)].copy()
    split_idx = int(split_prop * len(pre_assess))
    train_validate = pre_assess.iloc[:split_idx].copy()
    test_eval = pre_assess.iloc[split_idx:].copy()
    return {
        "test": (train_validate, test_eval),
        "assess": (pre_assess, assess_df),
    }


# =============================================================================
# Preprocessing and models
# =============================================================================

def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:  # sklearn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def as_object_frame(x: Any) -> Any:
    if hasattr(x, "astype"):
        return x.astype("object")
    return np.asarray(x, dtype=object)


def make_preprocessor(df: pd.DataFrame, predictors: Sequence[str], categorical_cols: Sequence[str]) -> ColumnTransformer:
    predictors = [c for c in predictors if c in df.columns]
    categorical_cols = [c for c in categorical_cols if c in predictors]
    numeric_cols = [c for c in predictors if c not in categorical_cols]
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline([
                    ("to_object", FunctionTransformer(as_object_frame, validate=False)),
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", make_one_hot_encoder()),
                ]),
                categorical_cols,
            ),
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                ]),
                numeric_cols,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def make_estimator(
    name: str,
    seed: int,
    ridge_alpha: float = 1.0,
    n_estimators: int = 600,
    learning_rate: float = 0.05,
    lgbm_n_jobs: int = 1,
) -> RegressorMixin:
    name = str(name).lower().strip()
    if name == "ridge":
        return Ridge(alpha=float(ridge_alpha), fit_intercept=True, random_state=None)
    if name == "linear":
        return LinearRegression(fit_intercept=True)
    if name == "huber":
        # Scaling is handled by the preprocessed numeric/categorical representation only indirectly.
        # Huber can be slow with high-cardinality one-hot features; use for small/diagnostic runs.
        return HuberRegressor(alpha=0.0001, epsilon=1.35, max_iter=500)
    if name.startswith("lgbm"):
        if lgb is None:
            raise ImportError("lightgbm is not installed. Use ridge/linear/huber or install lightgbm.")
        objective = "regression"
        alpha = None
        if name in {"lgbm_l1", "lgbm_mae"}:
            objective = "regression_l1"
        elif name in {"lgbm_l2", "lgbm_rmse", "lgbm_mse"}:
            objective = "regression"
        elif name in {"lgbm_quantile", "lgbm_median"}:
            objective = "quantile"
            alpha = 0.5
        else:
            raise ValueError(f"Unknown learner: {name}")
        kwargs = dict(
            objective=objective,
            n_estimators=int(n_estimators),
            learning_rate=float(learning_rate),
            num_leaves=63,
            min_child_samples=80,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=int(seed),
            n_jobs=max(1, int(lgbm_n_jobs)),
            verbose=-1,
            force_row_wise=True,
        )
        if alpha is not None:
            kwargs["alpha"] = alpha
        return lgb.LGBMRegressor(**kwargs)
    raise ValueError(f"Unknown learner: {name}")


@dataclass
class ModelSpec:
    learner: str
    predictors: List[str]
    categorical_cols: List[str]
    seed: int
    ridge_alpha: float
    n_estimators: int
    learning_rate: float
    lgbm_n_jobs: int = 1

    def new_pipeline(self, fit_df: pd.DataFrame) -> Pipeline:
        pre = make_preprocessor(fit_df, self.predictors, self.categorical_cols)
        est = make_estimator(
            self.learner,
            seed=self.seed,
            ridge_alpha=self.ridge_alpha,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            lgbm_n_jobs=self.lgbm_n_jobs,
        )
        return Pipeline([("prep", pre), ("est", est)])


def fit_predict_model(spec: ModelSpec, train_df: pd.DataFrame, eval_df: pd.DataFrame, y_train: np.ndarray) -> Tuple[np.ndarray, Pipeline]:
    model = spec.new_pipeline(train_df)
    model.fit(train_df[spec.predictors], y_train)
    pred = np.asarray(model.predict(eval_df[spec.predictors]), dtype=float).reshape(-1)
    return pred, model


def temporal_oof_predictions(
    spec: ModelSpec,
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    *,
    n_folds: int,
    mode: str = "rolling",
) -> np.ndarray:
    n = len(train_df)
    oof = np.full(n, np.nan, dtype=float)
    if n_folds <= 1 or n < max(50, n_folds * 10):
        # Fallback: in-sample only when too small.  The caller logs this.
        pred, _ = fit_predict_model(spec, train_df, train_df, y_train)
        return pred

    mode = str(mode).lower().strip()
    if mode not in {"rolling", "blocked"}:
        raise ValueError("--oof-mode must be rolling or blocked")

    # Time-ordered contiguous folds.  Rolling mode predicts each block from
    # prior rows only, so residual-lag training features do not use future core
    # outcomes.  Blocked mode preserves the older all-other-rows OOF behavior.
    fold_ids = np.array_split(np.arange(n), int(n_folds))
    for k, val_idx in enumerate(fold_ids):
        if mode == "rolling":
            tr_idx = np.arange(0, int(val_idx[0]), dtype=int)
            if len(tr_idx) < max(50, len(val_idx) // 2):
                continue
        else:
            tr_idx = np.setdiff1d(np.arange(n), val_idx)
        if len(tr_idx) == 0 or len(val_idx) == 0:
            continue
        pred, _ = fit_predict_model(spec, train_df.iloc[tr_idx].copy(), train_df.iloc[val_idx].copy(), y_train[tr_idx])
        oof[val_idx] = pred
    if mode == "blocked" and np.any(~np.isfinite(oof)):
        fill_pred, _ = fit_predict_model(spec, train_df, train_df, y_train)
        oof[~np.isfinite(oof)] = fill_pred[~np.isfinite(oof)]
    return oof


# =============================================================================
# Target adjustment layer
# =============================================================================

@dataclass
class TargetAdjuster:
    variant: str
    target_date: pd.Timestamp
    area_col: Optional[str] = None
    shrink_n: float = 100.0
    beta_global: float = 0.0
    beta_by_area: Optional[Dict[str, float]] = None
    residual_model: Optional[Pipeline] = None
    residual_control_cols: Optional[List[str]] = None
    residual_control_cats: Optional[List[str]] = None

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        y = df["y_raw_log"].to_numpy(dtype=float).copy()
        if self.variant == "raw":
            return y
        t_target = timestamp_to_year_float(self.target_date)
        delta = t_target - df["sale_year_float"].to_numpy(dtype=float)
        if self.beta_by_area is None or self.area_col is None or self.area_col not in df.columns:
            beta = np.full(len(df), float(self.beta_global), dtype=float)
        else:
            vals = df[self.area_col].astype("object").where(df[self.area_col].notna(), "NA").astype(str).to_numpy()
            beta = np.asarray([self.beta_by_area.get(v, self.beta_global) for v in vals], dtype=float)
        return y + beta * delta


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2 or np.nanstd(x[mask]) <= 0:
        return 0.0
    return float(np.polyfit(x[mask], y[mask], deg=1)[0])


def _fit_hedonic_residuals(
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    predictor_cols: Sequence[str],
    categorical_cols: Sequence[str],
    seed: int,
) -> Tuple[np.ndarray, Pipeline, List[str], List[str]]:
    controls = [c for c in HEDONIC_CONTROL_CANDIDATES if c in train_df.columns and c in predictor_cols]
    if len(controls) < 3:
        controls = [c for c in predictor_cols if c in train_df.columns][: min(30, len(predictor_cols))]
    cats = [c for c in categorical_cols if c in controls]
    spec = ModelSpec(
        learner="ridge",
        predictors=list(controls),
        categorical_cols=cats,
        seed=seed,
        ridge_alpha=10.0,
        n_estimators=100,
        learning_rate=0.05,
    )
    pred, model = fit_predict_model(spec, train_df, train_df, y_train)
    return y_train - pred, model, list(controls), list(cats)


def fit_target_adjuster(
    variant_raw: str,
    train_df: pd.DataFrame,
    *,
    target_date: pd.Timestamp,
    predictor_cols: Sequence[str],
    categorical_cols: Sequence[str],
    seed: int,
    shrink_n: float,
) -> TargetAdjuster:
    variant_raw = str(variant_raw).strip()
    if variant_raw == "raw":
        return TargetAdjuster(variant="raw", target_date=target_date)

    y = train_df["y_raw_log"].to_numpy(dtype=float)
    t = train_df["sale_year_float"].to_numpy(dtype=float)

    use_hedonic = variant_raw.startswith("hedonic_")
    base_variant = variant_raw.replace("hedonic_", "") if use_hedonic else variant_raw
    residuals = y
    residual_model = None
    controls = None
    cats = None
    if use_hedonic:
        residuals, residual_model, controls, cats = _fit_hedonic_residuals(train_df, y, predictor_cols, categorical_cols, seed=seed)

    if base_variant == "global_slope":
        beta = _ols_slope(t, residuals)
        log("target adjuster fit", variant=variant_raw, beta=f"{beta:.5f}", growth=f"{np.exp(beta)-1:.2%}")
        return TargetAdjuster(
            variant=variant_raw,
            target_date=target_date,
            beta_global=beta,
            residual_model=residual_model,
            residual_control_cols=controls,
            residual_control_cats=cats,
        )

    if base_variant.startswith("area_slope"):
        parts = base_variant.split(":", 1)
        area_col = parts[1] if len(parts) == 2 and parts[1] else TOWNSHIP_COL
        if area_col not in train_df.columns:
            raise ValueError(f"Area column for target variant {variant_raw} not found: {area_col}")
        beta_global = _ols_slope(t, residuals)
        vals = train_df[area_col].astype("object").where(train_df[area_col].notna(), "NA").astype(str)
        beta_by_area = {}
        for g, idx in vals.groupby(vals).groups.items():
            idx_arr = np.asarray(list(idx), dtype=int)
            beta_g_raw = _ols_slope(t[idx_arr], residuals[idx_arr])
            n_g = float(len(idx_arr))
            w = n_g / (n_g + float(shrink_n))
            beta_by_area[str(g)] = float(w * beta_g_raw + (1.0 - w) * beta_global)
        log("target adjuster fit", variant=variant_raw, beta_global=f"{beta_global:.5f}", n_areas=len(beta_by_area))
        return TargetAdjuster(
            variant=variant_raw,
            target_date=target_date,
            area_col=area_col,
            shrink_n=shrink_n,
            beta_global=beta_global,
            beta_by_area=beta_by_area,
            residual_model=residual_model,
            residual_control_cols=controls,
            residual_control_cats=cats,
        )

    raise ValueError(f"Unknown target variant: {variant_raw}")


# =============================================================================
# Spatial comparable layer
# =============================================================================

def spatial_xy_miles(df: pd.DataFrame) -> np.ndarray:
    if X_FT_COL in df.columns and Y_FT_COL in df.columns and df[[X_FT_COL, Y_FT_COL]].notna().all(axis=1).all():
        return np.column_stack([
            pd.to_numeric(df[X_FT_COL], errors="coerce").to_numpy(dtype=float) / 5280.0,
            pd.to_numeric(df[Y_FT_COL], errors="coerce").to_numpy(dtype=float) / 5280.0,
        ])
    lat = np.radians(pd.to_numeric(df[LAT_COL], errors="coerce").to_numpy(dtype=float))
    lon = np.radians(pd.to_numeric(df[LON_COL], errors="coerce").to_numpy(dtype=float))
    lat0 = float(np.nanmean(lat))
    return np.column_stack([EARTH_RADIUS_MILES * lon * np.cos(lat0), EARTH_RADIUS_MILES * lat])


def str_values_with_na(df: pd.DataFrame, col: Optional[str]) -> Optional[np.ndarray]:
    if col is None or col not in df.columns:
        return None
    return df[col].astype("object").where(df[col].notna(), "NA").astype(str).to_numpy()


class SpatialIndex:
    """KDTree candidate index with optional same-class filtering and no group filtering by default."""

    def __init__(self, df: pd.DataFrame, history_indices: Sequence[int], class_col: Optional[str]) -> None:
        self.df = df
        self.history_indices = np.asarray(history_indices, dtype=int)
        self.coords = spatial_xy_miles(df)
        self.sale_day = df["sale_day"].to_numpy(dtype=float)
        self.class_vals = str_values_with_na(df, class_col)
        self.pools: Dict[str, Tuple[np.ndarray, KDTree]] = {}
        self._build_pools()

    def _build_pools(self) -> None:
        if self.history_indices.size == 0:
            return
        # Universal pool.
        idx = self.history_indices.copy()
        self.pools["__all__"] = (idx, KDTree(self.coords[idx], leaf_size=40, metric="euclidean"))
        if self.class_vals is not None:
            class_keys = self.class_vals[idx]
            for cls in pd.unique(class_keys):
                members = idx[class_keys == cls]
                if members.size:
                    self.pools[str(cls)] = (members, KDTree(self.coords[members], leaf_size=40, metric="euclidean"))

    def _query_pool(self, target_i: int, key: str, n_query: int) -> Tuple[np.ndarray, np.ndarray]:
        if key not in self.pools:
            return np.empty(0, dtype=int), np.empty(0, dtype=float)
        idx, tree = self.pools[key]
        k = min(int(n_query), len(idx))
        if k <= 0:
            return np.empty(0, dtype=int), np.empty(0, dtype=float)
        dist, pos = tree.query(self.coords[int(target_i)].reshape(1, -1), k=k)
        return idx[np.asarray(pos[0], dtype=int)], np.asarray(dist[0], dtype=float)

    def candidate_set(
        self,
        target_i: int,
        *,
        K: int,
        min_same_class_pool: int,
        allow_class_fallback: bool,
        max_neighbor_age_days: Optional[float],
        max_spatial_candidates: int,
        candidate_multiplier: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        target_i = int(target_i)
        class_key = "__all__"
        if self.class_vals is not None:
            class_key = str(self.class_vals[target_i])
        threshold = max(1, min(int(K), int(min_same_class_pool)))

        def valid_candidates(pool_key: str, required: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            if pool_key not in self.pools:
                return np.empty(0, dtype=int), np.empty(0, dtype=float), np.empty(0, dtype=float)
            pool_idx, _ = self.pools[pool_key]
            pool_size = len(pool_idx)
            n_query = min(max(int(K) * int(candidate_multiplier), required), int(max_spatial_candidates), pool_size)
            n_query = max(1, n_query)
            best_i = np.empty(0, dtype=int)
            best_d = np.empty(0, dtype=float)
            best_t = np.empty(0, dtype=float)
            while True:
                cand_idx, d_space = self._query_pool(target_i, pool_key, n_query)
                d_time = self.sale_day[target_i] - self.sale_day[cand_idx]
                valid = d_time > 0.0
                if max_neighbor_age_days is not None and float(max_neighbor_age_days) > 0.0:
                    valid &= d_time <= float(max_neighbor_age_days)
                best_i, best_d, best_t = cand_idx[valid], d_space[valid], d_time[valid]
                if best_i.size >= required:
                    break
                next_n = min(n_query * 2, pool_size, int(max_spatial_candidates))
                if next_n <= n_query:
                    break
                n_query = next_n
            return best_i, best_d, best_t

        if self.class_vals is not None and class_key in self.pools:
            idx, d, t = valid_candidates(class_key, threshold)
            if idx.size >= threshold or not allow_class_fallback:
                return idx, d, t, False
        else:
            idx, d, t = np.empty(0, dtype=int), np.empty(0, dtype=float), np.empty(0, dtype=float)

        idx2, d2, t2 = valid_candidates("__all__", threshold)
        return idx2, d2, t2, bool(self.class_vals is not None and idx.size < threshold)


@dataclass
class CandidateCache:
    target_indices: np.ndarray
    candidate_indices: List[np.ndarray]
    d_space_miles: List[np.ndarray]
    d_time_days: List[np.ndarray]
    fallback: np.ndarray
    candidate_count: np.ndarray


def build_candidate_cache(
    df: pd.DataFrame,
    target_indices: Sequence[int],
    history_indices: Sequence[int],
    *,
    K: int,
    class_col: Optional[str],
    min_same_class_pool: int,
    allow_class_fallback: bool,
    max_neighbor_age_days: Optional[float],
    max_spatial_candidates: int,
    candidate_multiplier: int,
    n_jobs: int,
) -> CandidateCache:
    target_indices = np.asarray(target_indices, dtype=int)
    sidx = SpatialIndex(df, history_indices, class_col)

    def one(i: int):
        return sidx.candidate_set(
            int(i),
            K=int(K),
            min_same_class_pool=int(min_same_class_pool),
            allow_class_fallback=bool(allow_class_fallback),
            max_neighbor_age_days=max_neighbor_age_days,
            max_spatial_candidates=int(max_spatial_candidates),
            candidate_multiplier=int(candidate_multiplier),
        )

    if n_jobs and n_jobs != 1 and Parallel is not None and delayed is not None:
        out = Parallel(n_jobs=int(n_jobs), prefer="threads", batch_size=256)(delayed(one)(int(i)) for i in target_indices)
    else:
        out = [one(int(i)) for i in target_indices]
    cand, ds, dt, fb = [], [], [], []
    for idx, d, t, f in out:
        cand.append(np.asarray(idx, dtype=int))
        ds.append(np.asarray(d, dtype=float))
        dt.append(np.asarray(t, dtype=float))
        fb.append(bool(f))
    return CandidateCache(
        target_indices=target_indices,
        candidate_indices=cand,
        d_space_miles=ds,
        d_time_days=dt,
        fallback=np.asarray(fb, dtype=bool),
        candidate_count=np.asarray([len(x) for x in cand], dtype=int),
    )


def compute_spatial_feature(
    df: pd.DataFrame,
    cache: CandidateCache,
    *,
    K: int,
    spatial_bw_miles: float,
    time_bw_days: float,
    feature_type: str,
    y_raw: np.ndarray,
    y_target: np.ndarray,
    beta_time: float,
    base_pred: np.ndarray,
    base_residual: np.ndarray,
    spatial_target_year_float: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    n = len(df)
    out = np.full(n, np.nan, dtype=float)
    n_used = np.zeros(n, dtype=int)
    mean_space = np.full(n, np.nan, dtype=float)
    mean_time = np.full(n, np.nan, dtype=float)
    fallback = np.zeros(n, dtype=bool)
    cand_count = np.zeros(n, dtype=int)
    sale_year = df["sale_year_float"].to_numpy(dtype=float)

    for pos, i in enumerate(cache.target_indices):
        i = int(i)
        idx = cache.candidate_indices[pos]
        d_space = cache.d_space_miles[pos]
        d_time = cache.d_time_days[pos]
        fallback[i] = bool(cache.fallback[pos])
        cand_count[i] = int(cache.candidate_count[pos])
        if idx.size == 0:
            continue
        score = d_space / float(spatial_bw_miles) + d_time / float(time_bw_days)
        finite = np.isfinite(score)
        if not finite.any():
            continue
        idx = idx[finite]
        d_space = d_space[finite]
        d_time = d_time[finite]
        score = score[finite]
        k_eff = min(int(K), idx.size)
        chosen = np.argpartition(score, k_eff - 1)[:k_eff]
        j = idx[chosen]
        weights = np.exp(-score[chosen])
        sw = float(weights.sum())
        if sw <= 0 or not np.isfinite(sw):
            continue

        if feature_type == "raw_price":
            vals = y_raw[j]
        elif feature_type == "time_adjusted_price":
            vals = y_raw[j] + float(beta_time) * (spatial_target_year_float[i] - sale_year[j])
        elif feature_type == "target_label":
            vals = y_target[j]
        elif feature_type == "scaled_target_label":
            vals = y_target[j] + base_pred[i] - base_pred[j]
        elif feature_type == "scaled_price":
            vals = y_raw[j] + base_pred[i] - base_pred[j]
        elif feature_type == "scaled_time_adjusted_price":
            vals = y_raw[j] + float(beta_time) * (spatial_target_year_float[i] - sale_year[j]) + base_pred[i] - base_pred[j]
        elif feature_type == "residual":
            vals = base_residual[j]
        else:
            raise ValueError(f"Unknown spatial feature type: {feature_type}")

        finite_vals = np.isfinite(vals) & np.isfinite(weights)
        if not finite_vals.any():
            continue
        vals = vals[finite_vals]
        weights = weights[finite_vals]
        j = j[finite_vals]
        chosen_space = d_space[chosen][finite_vals]
        chosen_time = d_time[chosen][finite_vals]
        sw = float(weights.sum())
        if sw <= 0 or not np.isfinite(sw):
            continue

        out[i] = float(np.sum(weights * vals) / sw)
        n_used[i] = int(len(vals))
        mean_space[i] = float(np.sum(weights * chosen_space) / sw)
        mean_time[i] = float(np.sum(weights * chosen_time) / sw)

    return out, {
        "n_neighbors_used": n_used,
        "mean_neighbor_space_miles": mean_space,
        "mean_neighbor_time_days": mean_time,
        "used_class_fallback": fallback,
        "candidate_count": cand_count,
    }


# =============================================================================
# Calibration and metrics
# =============================================================================

@dataclass
class Calibrator:
    mode: str
    intercept: float = 0.0
    slope: float = 1.0

    def predict(self, yhat: np.ndarray) -> np.ndarray:
        yhat = np.asarray(yhat, dtype=float)
        if self.mode == "none":
            return yhat
        if self.mode == "median_center":
            return yhat + self.intercept
        if self.mode == "affine":
            return self.intercept + self.slope * yhat
        raise ValueError(f"Unknown calibrator mode: {self.mode}")


def fit_calibrator(mode: str, yhat_calib: np.ndarray, y_calib: np.ndarray) -> Calibrator:
    mode = str(mode).strip()
    if mode == "none":
        return Calibrator(mode="none")
    resid = np.asarray(y_calib, dtype=float) - np.asarray(yhat_calib, dtype=float)
    if mode == "median_center":
        return Calibrator(mode=mode, intercept=float(np.nanmedian(resid)), slope=1.0)
    if mode == "affine":
        mask = np.isfinite(yhat_calib) & np.isfinite(y_calib)
        if mask.sum() < 10 or np.nanstd(yhat_calib[mask]) <= 0:
            return Calibrator(mode="median_center", intercept=float(np.nanmedian(resid)), slope=1.0)
        lr = LinearRegression().fit(yhat_calib[mask].reshape(-1, 1), y_calib[mask])
        return Calibrator(mode=mode, intercept=float(lr.intercept_), slope=float(lr.coef_[0]))
    raise ValueError(f"Unknown calibration mode: {mode}")


def decile_curve_stats(y_true_log: np.ndarray, y_pred_log: np.ndarray, n_deciles: int = 10) -> Dict[str, float]:
    y_true_log = np.asarray(y_true_log, dtype=float)
    y_pred_log = np.asarray(y_pred_log, dtype=float)
    e = y_pred_log - y_true_log
    mask = np.isfinite(y_true_log) & np.isfinite(e)
    if mask.sum() < n_deciles * 5:
        return {"curve_level": np.nan, "curve_trend": np.nan, "curve_shape": np.nan, "curve_max_gap": np.nan}
    y = y_true_log[mask]
    e = e[mask]
    try:
        bins = pd.qcut(y, q=n_deciles, labels=False, duplicates="drop")
    except Exception:
        return {"curve_level": np.nan, "curve_trend": np.nan, "curve_shape": np.nan, "curve_max_gap": np.nan}
    tmp = pd.DataFrame({"y": y, "e": e, "bin": bins})
    grp = tmp.groupby("bin", observed=True).agg(y_med=("y", "median"), e_med=("e", "median"), n=("e", "size")).reset_index(drop=True)
    if len(grp) < 3:
        return {"curve_level": np.nan, "curve_trend": np.nan, "curve_shape": np.nan, "curve_max_gap": np.nan}
    z = grp["y_med"].to_numpy(dtype=float)
    c = grp["e_med"].to_numpy(dtype=float)
    w = grp["n"].to_numpy(dtype=float)
    zc = z - np.average(z, weights=w)
    level = float(np.average(c, weights=w))
    if np.sum(w * zc * zc) <= 0:
        trend = np.nan
        shape = np.nan
    else:
        trend = float(np.sum(w * zc * (c - level)) / np.sum(w * zc * zc))
        residual = c - level - trend * zc
        shape = float(np.sqrt(np.average(residual ** 2, weights=w)))
    return {
        "curve_level": level,
        "curve_trend": trend,
        "curve_shape": shape,
        "curve_max_gap": float(np.nanmax(c) - np.nanmin(c)),
    }


def evaluate(
    *,
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    y_true_raw_log: np.ndarray,
    y_train_log: np.ndarray,
    label: str,
    n_deciles: int,
) -> Dict[str, Any]:
    y_true_log = np.asarray(y_true_log, dtype=float)
    y_pred_log = np.asarray(y_pred_log, dtype=float)
    y_true_raw_log = np.asarray(y_true_raw_log, dtype=float)
    e_target = y_pred_log - y_true_log
    e_raw = y_pred_log - y_true_raw_log
    ratio_raw = np.exp(np.clip(e_raw, -50, 50))
    ratio_target = np.exp(np.clip(e_target, -50, 50))
    out: Dict[str, Any] = {
        "model": label,
        "n": int(len(y_true_log)),
        "r2_log_target": float(r2_score(y_true_log, y_pred_log)),
        "rmse_log_target": float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
        "mae_log_target": float(mean_absolute_error(y_true_log, y_pred_log)),
        "median_ratio_target": float(np.nanmedian(ratio_target)),
        "r2_log_raw": float(r2_score(y_true_raw_log, y_pred_log)),
        "rmse_log_raw": float(np.sqrt(mean_squared_error(y_true_raw_log, y_pred_log))),
        "mae_log_raw": float(mean_absolute_error(y_true_raw_log, y_pred_log)),
        "median_ratio_raw": float(np.nanmedian(ratio_raw)),
    }
    y_true_price = np.exp(np.clip(y_true_raw_log, -50, 50))
    y_pred_price = np.exp(np.clip(y_pred_log, -50, 50))
    out.update({
        "r2_price_raw": float(r2_score(y_true_price, y_pred_price)),
        "rmse_price_raw": float(np.sqrt(mean_squared_error(y_true_price, y_pred_price))),
        "mae_price_raw": float(mean_absolute_error(y_true_price, y_pred_price)),
    })
    if len(y_true_raw_log) >= 2 and np.nanstd(y_true_raw_log) > 0:
        lr = LinearRegression().fit(y_true_raw_log.reshape(-1, 1), e_raw.reshape(-1, 1))
        out["log_ratio_slope_vs_raw_log_price"] = float(lr.coef_[0, 0])
    else:
        out["log_ratio_slope_vs_raw_log_price"] = np.nan
    out.update({f"raw_{k}": v for k, v in decile_curve_stats(y_true_raw_log, y_pred_log, n_deciles=n_deciles).items()})
    out.update({f"target_{k}": v for k, v in decile_curve_stats(y_true_log, y_pred_log, n_deciles=n_deciles).items()})
    return out


# =============================================================================
# Experiment runner
# =============================================================================

def split_core_calibration(train_df: pd.DataFrame, calib_frac: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if calib_frac <= 0:
        return train_df.copy(), train_df.iloc[0:0].copy()
    train_df = train_df.sort_values(DATE_COL).reset_index(drop=True)
    n = len(train_df)
    cut = int((1.0 - calib_frac) * n)
    cut = min(max(cut, 100), n)
    core = train_df.iloc[:cut].copy()
    calib = train_df.iloc[cut:].copy()
    return core, calib


def build_combined(core_df: pd.DataFrame, calib_df: pd.DataFrame, eval_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    parts = []
    for name, d in [("core", core_df), ("calib", calib_df), ("eval", eval_df)]:
        x = d.copy()
        x["__split_part"] = name
        x["__orig_pos"] = np.arange(len(x))
        parts.append(x)
    combined = pd.concat(parts, axis=0, ignore_index=True).sort_values(DATE_COL).reset_index(drop=True)
    core_idx = np.where(combined["__split_part"].to_numpy() == "core")[0]
    calib_idx = np.where(combined["__split_part"].to_numpy() == "calib")[0]
    eval_idx = np.where(combined["__split_part"].to_numpy() == "eval")[0]
    return combined, core_idx, calib_idx, eval_idx


def prepare_augmented_predictors(
    base_predictors: List[str],
    base_cats: List[str],
    spatial_col: str,
    diag_cols: Sequence[str],
) -> Tuple[List[str], List[str]]:
    pred = list(base_predictors)
    for c in [spatial_col] + list(diag_cols):
        if c not in pred:
            pred.append(c)
    cats = [c for c in base_cats if c in pred]
    return pred, cats


METRIC_KEY_COLS = [
    "split",
    "target_variant",
    "learner",
    "stage",
    "spatial_feature_type",
    "K",
    "spatial_bw_miles",
    "time_bw_days",
    "calibration_mode",
]

METRIC_STREAM_COLUMNS = [
    "model",
    "n",
    "r2_log_target",
    "rmse_log_target",
    "mae_log_target",
    "median_ratio_target",
    "r2_log_raw",
    "rmse_log_raw",
    "mae_log_raw",
    "median_ratio_raw",
    "r2_price_raw",
    "rmse_price_raw",
    "mae_price_raw",
    "log_ratio_slope_vs_raw_log_price",
    "raw_curve_level",
    "raw_curve_trend",
    "raw_curve_shape",
    "raw_curve_max_gap",
    "target_curve_level",
    "target_curve_trend",
    "target_curve_shape",
    "target_curve_max_gap",
    "split",
    "target_variant",
    "learner",
    "stage",
    "spatial_feature_type",
    "K",
    "spatial_bw_miles",
    "time_bw_days",
    "calibration_mode",
    "cal_intercept",
    "cal_slope",
    "beta_time",
    "valuation_date",
    "valid_train_rows",
    "valid_eval_rows",
    "eval_coverage_share",
    "mean_eval_neighbors",
    "mean_eval_neighbor_space_miles",
    "mean_eval_neighbor_time_days",
    "class_fallback_share_eval",
    "strict_eval_history",
]


def _metric_key_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    if isinstance(value, (float, np.floating, int, np.integer)) and not isinstance(value, bool):
        return f"{float(value):.12g}"
    return str(value)


def metric_key(row: Dict[str, Any]) -> Tuple[str, ...]:
    return tuple(_metric_key_value(row.get(c)) for c in METRIC_KEY_COLS)


def metric_keys_from_frame(df: pd.DataFrame) -> set:
    if df is None or df.empty:
        return set()
    return {metric_key(row) for row in df.to_dict(orient="records")}


def append_metric_row(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists() or path.stat().st_size == 0
    pd.DataFrame([row]).reindex(columns=METRIC_STREAM_COLUMNS).to_csv(
        path,
        mode="a",
        header=header,
        index=False,
    )


def load_metric_stream(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def dedupe_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    df = metrics.copy()
    df["__metric_key"] = [metric_key(row) for row in df.to_dict(orient="records")]
    df = df.drop_duplicates("__metric_key", keep="last").drop(columns="__metric_key")
    return df.reset_index(drop=True)


def run_one_split(
    *,
    split_name: str,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    params: dict,
    predictor_cols: List[str],
    categorical_cols: List[str],
    target_variants: Sequence[str],
    learners: Sequence[str],
    spatial_feature_types: Sequence[str],
    spatial_stages: Sequence[str],
    Ks: Sequence[int],
    spatial_bws: Sequence[float],
    time_bws: Sequence[float],
    calibration_modes: Sequence[str],
    calib_frac: float,
    valuation_date_mode: str,
    seed: int,
    ridge_alpha: float,
    n_estimators: int,
    learning_rate: float,
    lgbm_n_jobs: int,
    oof_mode: str,
    target_shrink_n: float,
    class_filter_col: Optional[str],
    min_same_class_pool: int,
    allow_class_fallback: bool,
    max_neighbor_age_days: Optional[float],
    max_spatial_candidates: int,
    candidate_multiplier: int,
    n_jobs: int,
    n_oof_folds: int,
    n_deciles: int,
    strict_eval_history: bool,
    stream_path: Optional[Path] = None,
    completed_metric_keys: Optional[set] = None,
) -> pd.DataFrame:
    if eval_df.empty or train_df.empty:
        return pd.DataFrame()
    assessment_date = str(params.get("assessment", {}).get("date", "2025-01-01"))
    valuation_date = infer_eval_valuation_date(eval_df, valuation_date_mode, assessment_date)
    core_df, calib_df = split_core_calibration(train_df, calib_frac)
    log("split start", split=split_name, train=len(train_df), core=len(core_df), calib=len(calib_df), eval=len(eval_df), valuation_date=str(valuation_date.date()))

    rows: List[Dict[str, Any]] = []
    completed_metric_keys = set() if completed_metric_keys is None else set(completed_metric_keys)

    def emit_metric(metrics: Dict[str, Any]) -> None:
        key = metric_key(metrics)
        if key in completed_metric_keys:
            return
        rows.append(metrics)
        if stream_path is not None:
            append_metric_row(stream_path, metrics)
        completed_metric_keys.add(key)

    def spatial_stage_complete(
        *,
        target_variant_: str,
        learner_: str,
        stage_: str,
        feat_type_: str,
        K_: int,
        h_s_: float,
        h_t_: float,
    ) -> bool:
        for mode in calibration_modes:
            key = metric_key({
                "split": split_name,
                "target_variant": target_variant_,
                "learner": learner_,
                "stage": stage_,
                "spatial_feature_type": feat_type_,
                "K": int(K_),
                "spatial_bw_miles": float(h_s_),
                "time_bw_days": float(h_t_),
                "calibration_mode": mode,
            })
            if key not in completed_metric_keys:
                return False
        return True

    stage_set = {str(s).strip() for s in spatial_stages if str(s).strip()}
    if "none" in stage_set or "base_only" in stage_set:
        stage_set = set()
    if not stage_set:
        run_full_x_spatial = False
        run_base_plus_residual = False
    else:
        allowed_stages = {"full_X_plus_spatial", "base_plus_local_residual"}
        unknown_stages = sorted(stage_set - allowed_stages)
        if unknown_stages:
            raise ValueError(f"Unknown spatial stage(s): {unknown_stages}. Allowed: {sorted(allowed_stages)}")
        run_full_x_spatial = "full_X_plus_spatial" in stage_set
        run_base_plus_residual = "base_plus_local_residual" in stage_set

    for target_variant in target_variants:
        adjuster = fit_target_adjuster(
            target_variant,
            core_df,
            target_date=valuation_date,
            predictor_cols=predictor_cols,
            categorical_cols=categorical_cols,
            seed=seed,
            shrink_n=target_shrink_n,
        )
        combined, core_idx, calib_idx, eval_idx = build_combined(core_df, calib_df, eval_df)
        for c in categorical_cols:
            if c in combined.columns:
                combined[c] = combined[c].astype("category")
        combined["y_target_log"] = adjuster.transform(combined)
        combined["y_raw_log"] = safe_log_price(combined[TARGET_COL].to_numpy())
        combined["spatial_target_year_float"] = combined["sale_year_float"].to_numpy(dtype=float)
        if target_variant != "raw":
            combined["spatial_target_year_float"] = timestamp_to_year_float(valuation_date)

        # Beta for spatial time adjustment.  Use the target adjuster global beta if available; otherwise raw global slope.
        if adjuster.variant != "raw":
            beta_time = float(adjuster.beta_global)
        else:
            beta_time = _ols_slope(combined.loc[core_idx, "sale_year_float"].to_numpy(), combined.loc[core_idx, "y_raw_log"].to_numpy())

        for learner in learners:
            base_spec = ModelSpec(
                learner=learner,
                predictors=list(predictor_cols),
                categorical_cols=list(categorical_cols),
                seed=seed,
                ridge_alpha=ridge_alpha,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                lgbm_n_jobs=lgbm_n_jobs,
            )

            y_core = combined.loc[core_idx, "y_target_log"].to_numpy(dtype=float)
            y_calib = combined.loc[calib_idx, "y_target_log"].to_numpy(dtype=float) if len(calib_idx) else np.array([])
            y_eval = combined.loc[eval_idx, "y_target_log"].to_numpy(dtype=float)
            y_eval_raw = combined.loc[eval_idx, "y_raw_log"].to_numpy(dtype=float)

            log("base fit", split=split_name, target=target_variant, learner=learner)
            base_oof_core = temporal_oof_predictions(
                base_spec,
                combined.iloc[core_idx].copy(),
                y_core,
                n_folds=n_oof_folds,
                mode=oof_mode,
            )
            pred_eval_base, base_model = fit_predict_model(base_spec, combined.iloc[core_idx].copy(), combined.iloc[eval_idx].copy(), y_core)
            pred_calib_base = np.asarray(base_model.predict(combined.iloc[calib_idx][predictor_cols]), dtype=float) if len(calib_idx) else np.array([])

            base_pred_all = np.full(len(combined), np.nan, dtype=float)
            base_pred_all[core_idx] = base_oof_core
            if len(calib_idx):
                base_pred_all[calib_idx] = pred_calib_base
            base_pred_all[eval_idx] = pred_eval_base
            base_resid_all = np.full(len(combined), np.nan, dtype=float)
            base_resid_all[core_idx] = combined.loc[core_idx, "y_target_log"].to_numpy(dtype=float) - base_oof_core
            # Residuals for calib/eval are never used as history in strict mode, but fill for diagnostics.
            if len(calib_idx):
                base_resid_all[calib_idx] = combined.loc[calib_idx, "y_target_log"].to_numpy(dtype=float) - pred_calib_base
            base_resid_all[eval_idx] = y_eval - pred_eval_base

            for calib_mode in calibration_modes:
                if len(calib_idx) and calib_mode != "none":
                    cal = fit_calibrator(calib_mode, pred_calib_base, y_calib)
                else:
                    cal = Calibrator(mode="none")
                pred_eval_cal = cal.predict(pred_eval_base)
                metrics = evaluate(
                    y_true_log=y_eval,
                    y_pred_log=pred_eval_cal,
                    y_true_raw_log=y_eval_raw,
                    y_train_log=y_core,
                    label=f"{split_name}__target={target_variant}__learner={learner}__base__cal={cal.mode}",
                    n_deciles=n_deciles,
                )
                metrics.update({
                    "split": split_name,
                    "target_variant": target_variant,
                    "learner": learner,
                    "stage": "base_full_X",
                    "spatial_feature_type": "none",
                    "K": np.nan,
                    "spatial_bw_miles": np.nan,
                    "time_bw_days": np.nan,
                    "calibration_mode": cal.mode,
                    "cal_intercept": cal.intercept,
                    "cal_slope": cal.slope,
                    "beta_time": beta_time,
                    "valuation_date": str(valuation_date.date()),
                    "valid_train_rows": int(len(core_idx)),
                    "valid_eval_rows": int(len(eval_idx)),
                    "eval_coverage_share": 1.0,
                })
                emit_metric(metrics)

            effective_spatial_feature_types = tuple(
                ft for ft in spatial_feature_types
                if run_full_x_spatial or (run_base_plus_residual and ft == "residual")
            )
            if not effective_spatial_feature_types:
                continue

            # Spatial experiments.
            # History for calibration/eval is core only by default.  Strictly avoid eval outcomes.
            history_for_eval = core_idx
            if not strict_eval_history:
                # Sequential option: still no future sale for each target due to candidate d_time > 0,
                # but earlier calibration/eval sales can become history.  Use only for operational simulation.
                history_for_eval = np.concatenate([core_idx, calib_idx, eval_idx])

            for K in Ks:
                log("candidate cache", split=split_name, target=target_variant, learner=learner, K=K)
                cache_core = build_candidate_cache(
                    combined, core_idx, core_idx,
                    K=K,
                    class_col=class_filter_col,
                    min_same_class_pool=min_same_class_pool,
                    allow_class_fallback=allow_class_fallback,
                    max_neighbor_age_days=max_neighbor_age_days,
                    max_spatial_candidates=max_spatial_candidates,
                    candidate_multiplier=candidate_multiplier,
                    n_jobs=n_jobs,
                )
                cache_calib = build_candidate_cache(
                    combined, calib_idx, core_idx,
                    K=K,
                    class_col=class_filter_col,
                    min_same_class_pool=min_same_class_pool,
                    allow_class_fallback=allow_class_fallback,
                    max_neighbor_age_days=max_neighbor_age_days,
                    max_spatial_candidates=max_spatial_candidates,
                    candidate_multiplier=candidate_multiplier,
                    n_jobs=n_jobs,
                ) if len(calib_idx) else None
                cache_eval = build_candidate_cache(
                    combined, eval_idx, history_for_eval,
                    K=K,
                    class_col=class_filter_col,
                    min_same_class_pool=min_same_class_pool,
                    allow_class_fallback=allow_class_fallback,
                    max_neighbor_age_days=max_neighbor_age_days,
                    max_spatial_candidates=max_spatial_candidates,
                    candidate_multiplier=candidate_multiplier,
                    n_jobs=n_jobs,
                )

                for h_s in spatial_bws:
                    for h_t in time_bws:
                        for feat_type in effective_spatial_feature_types:
                            do_full_x_spatial = run_full_x_spatial
                            do_base_plus_residual = run_base_plus_residual and feat_type == "residual"
                            full_x_done = do_full_x_spatial and spatial_stage_complete(
                                target_variant_=target_variant,
                                learner_=learner,
                                stage_="full_X_plus_spatial",
                                feat_type_=feat_type,
                                K_=K,
                                h_s_=h_s,
                                h_t_=h_t,
                            )
                            residual_done = do_base_plus_residual and spatial_stage_complete(
                                target_variant_=target_variant,
                                learner_=learner,
                                stage_="base_plus_local_residual",
                                feat_type_=feat_type,
                                K_=K,
                                h_s_=h_s,
                                h_t_=h_t,
                            )
                            if (not do_full_x_spatial or full_x_done) and (not do_base_plus_residual or residual_done):
                                continue

                            spatial_col = f"sp_{feat_type}_K{K}_hs{h_s}_ht{h_t}"
                            diag_cols = [
                                f"{spatial_col}_n",
                                f"{spatial_col}_space",
                                f"{spatial_col}_age",
                                f"{spatial_col}_fallback",
                            ]
                            y_raw_all = combined["y_raw_log"].to_numpy(dtype=float)
                            y_targ_all = combined["y_target_log"].to_numpy(dtype=float)
                            sp_target_year = combined["spatial_target_year_float"].to_numpy(dtype=float)

                            feat_core, diag_core = compute_spatial_feature(
                                combined, cache_core,
                                K=K,
                                spatial_bw_miles=h_s,
                                time_bw_days=h_t,
                                feature_type=feat_type,
                                y_raw=y_raw_all,
                                y_target=y_targ_all,
                                beta_time=beta_time,
                                base_pred=base_pred_all,
                                base_residual=base_resid_all,
                                spatial_target_year_float=sp_target_year,
                            )
                            feat_eval, diag_eval = compute_spatial_feature(
                                combined, cache_eval,
                                K=K,
                                spatial_bw_miles=h_s,
                                time_bw_days=h_t,
                                feature_type=feat_type,
                                y_raw=y_raw_all,
                                y_target=y_targ_all,
                                beta_time=beta_time,
                                base_pred=base_pred_all,
                                base_residual=base_resid_all,
                                spatial_target_year_float=sp_target_year,
                            )
                            if len(calib_idx):
                                feat_calib, diag_calib = compute_spatial_feature(
                                    combined, cache_calib,
                                    K=K,
                                    spatial_bw_miles=h_s,
                                    time_bw_days=h_t,
                                    feature_type=feat_type,
                                    y_raw=y_raw_all,
                                    y_target=y_targ_all,
                                    beta_time=beta_time,
                                    base_pred=base_pred_all,
                                    base_residual=base_resid_all,
                                    spatial_target_year_float=sp_target_year,
                                )
                            else:
                                feat_calib = np.full(len(combined), np.nan)
                                diag_calib = {k: np.full(len(combined), np.nan) for k in diag_core.keys()}

                            # Merge feature/diagnostics into combined copy for this candidate.
                            cdf = combined.copy()
                            cdf[spatial_col] = np.nan
                            feature_sources = [
                                (core_idx, feat_core),
                                (calib_idx, feat_calib),
                                (eval_idx, feat_eval),
                            ]
                            for target_rows, arr in feature_sources:
                                if len(target_rows) == 0:
                                    continue
                                target_vals = np.asarray(arr, dtype=float)[target_rows]
                                mask = np.isfinite(target_vals)
                                if mask.any():
                                    cdf.loc[target_rows[mask], spatial_col] = target_vals[mask]
                            # Diagnostics.
                            diag_sources = [
                                (core_idx, diag_core),
                                (calib_idx, diag_calib),
                                (eval_idx, diag_eval),
                            ]
                            diag_arrays: Dict[str, np.ndarray] = {}
                            for key, name in [
                                ("n_neighbors_used", diag_cols[0]),
                                ("mean_neighbor_space_miles", diag_cols[1]),
                                ("mean_neighbor_time_days", diag_cols[2]),
                                ("used_class_fallback", diag_cols[3]),
                            ]:
                                vals = np.full(len(combined), np.nan, dtype=float)
                                for target_rows, dd in diag_sources:
                                    if len(target_rows) == 0:
                                        continue
                                    if key in dd:
                                        arr = np.asarray(dd[key], dtype=float)
                                        vals[target_rows] = arr[target_rows]
                                cdf[name] = vals
                                diag_arrays[name] = vals

                            train_valid = core_idx[np.isfinite(cdf.loc[core_idx, spatial_col].to_numpy(dtype=float))]
                            eval_valid = eval_idx[np.isfinite(cdf.loc[eval_idx, spatial_col].to_numpy(dtype=float))]
                            calib_valid = calib_idx[np.isfinite(cdf.loc[calib_idx, spatial_col].to_numpy(dtype=float))] if len(calib_idx) else np.array([], dtype=int)
                            if len(train_valid) < 100 or len(eval_valid) < 50:
                                continue

                            # Stage 1: full-X plus spatial lag feature.
                            if do_full_x_spatial and not full_x_done:
                                aug_pred, aug_cats = prepare_augmented_predictors(predictor_cols, categorical_cols, spatial_col, diag_cols)
                                aug_spec = ModelSpec(
                                    learner=learner,
                                    predictors=aug_pred,
                                    categorical_cols=aug_cats,
                                    seed=seed,
                                    ridge_alpha=ridge_alpha,
                                    n_estimators=n_estimators,
                                    learning_rate=learning_rate,
                                    lgbm_n_jobs=lgbm_n_jobs,
                                )
                                pred_eval_sp, model_sp = fit_predict_model(
                                    aug_spec,
                                    cdf.iloc[train_valid].copy(),
                                    cdf.iloc[eval_valid].copy(),
                                    cdf.loc[train_valid, "y_target_log"].to_numpy(dtype=float),
                                )
                                pred_calib_sp = np.asarray(model_sp.predict(cdf.iloc[calib_valid][aug_pred]), dtype=float) if len(calib_valid) else np.array([])

                                for calib_mode in calibration_modes:
                                    if len(calib_valid) and calib_mode != "none":
                                        cal = fit_calibrator(calib_mode, pred_calib_sp, cdf.loc[calib_valid, "y_target_log"].to_numpy(dtype=float))
                                    else:
                                        cal = Calibrator(mode="none")
                                    pred_eval_cal = cal.predict(pred_eval_sp)
                                    metrics = evaluate(
                                        y_true_log=cdf.loc[eval_valid, "y_target_log"].to_numpy(dtype=float),
                                        y_pred_log=pred_eval_cal,
                                        y_true_raw_log=cdf.loc[eval_valid, "y_raw_log"].to_numpy(dtype=float),
                                        y_train_log=cdf.loc[train_valid, "y_target_log"].to_numpy(dtype=float),
                                        label=f"{split_name}__target={target_variant}__learner={learner}__fullX_plus_{feat_type}__K={K}__hs={h_s}__ht={h_t}__cal={cal.mode}",
                                        n_deciles=n_deciles,
                                    )
                                    metrics.update({
                                        "split": split_name,
                                        "target_variant": target_variant,
                                        "learner": learner,
                                        "stage": "full_X_plus_spatial",
                                        "spatial_feature_type": feat_type,
                                        "K": int(K),
                                        "spatial_bw_miles": float(h_s),
                                        "time_bw_days": float(h_t),
                                        "calibration_mode": cal.mode,
                                        "cal_intercept": cal.intercept,
                                        "cal_slope": cal.slope,
                                        "beta_time": beta_time,
                                        "valuation_date": str(valuation_date.date()),
                                        "valid_train_rows": int(len(train_valid)),
                                        "valid_eval_rows": int(len(eval_valid)),
                                        "eval_coverage_share": float(len(eval_valid) / max(1, len(eval_idx))),
                                        "mean_eval_neighbors": float(np.nanmean(cdf.loc[eval_valid, diag_cols[0]])),
                                        "mean_eval_neighbor_space_miles": float(np.nanmean(cdf.loc[eval_valid, diag_cols[1]])),
                                        "mean_eval_neighbor_time_days": float(np.nanmean(cdf.loc[eval_valid, diag_cols[2]])),
                                        "class_fallback_share_eval": float(np.nanmean(cdf.loc[eval_valid, diag_cols[3]])),
                                        "strict_eval_history": bool(strict_eval_history),
                                    })
                                    emit_metric(metrics)

                            # Residual-specific correction model: base_pred + local residual only.
                            if do_base_plus_residual and not residual_done:
                                x_train = pd.DataFrame({
                                    "base_pred": base_pred_all[train_valid],
                                    "local_resid": cdf.loc[train_valid, spatial_col].to_numpy(dtype=float),
                                })
                                x_eval = pd.DataFrame({
                                    "base_pred": base_pred_all[eval_valid],
                                    "local_resid": cdf.loc[eval_valid, spatial_col].to_numpy(dtype=float),
                                })
                                ridge = Ridge(alpha=1.0).fit(x_train, cdf.loc[train_valid, "y_target_log"].to_numpy(dtype=float))
                                pred_eval_resid = ridge.predict(x_eval)
                                if len(calib_valid):
                                    x_cal = pd.DataFrame({
                                        "base_pred": base_pred_all[calib_valid],
                                        "local_resid": cdf.loc[calib_valid, spatial_col].to_numpy(dtype=float),
                                    })
                                    pred_calib_resid = ridge.predict(x_cal)
                                else:
                                    pred_calib_resid = np.array([])
                                for calib_mode in calibration_modes:
                                    if len(calib_valid) and calib_mode != "none":
                                        cal = fit_calibrator(calib_mode, pred_calib_resid, cdf.loc[calib_valid, "y_target_log"].to_numpy(dtype=float))
                                    else:
                                        cal = Calibrator(mode="none")
                                    pred_eval_cal = cal.predict(pred_eval_resid)
                                    metrics = evaluate(
                                        y_true_log=cdf.loc[eval_valid, "y_target_log"].to_numpy(dtype=float),
                                        y_pred_log=pred_eval_cal,
                                        y_true_raw_log=cdf.loc[eval_valid, "y_raw_log"].to_numpy(dtype=float),
                                        y_train_log=cdf.loc[train_valid, "y_target_log"].to_numpy(dtype=float),
                                        label=f"{split_name}__target={target_variant}__learner={learner}__base_plus_residual__K={K}__hs={h_s}__ht={h_t}__cal={cal.mode}",
                                        n_deciles=n_deciles,
                                    )
                                    metrics.update({
                                        "split": split_name,
                                        "target_variant": target_variant,
                                        "learner": learner,
                                        "stage": "base_plus_local_residual",
                                        "spatial_feature_type": feat_type,
                                        "K": int(K),
                                        "spatial_bw_miles": float(h_s),
                                        "time_bw_days": float(h_t),
                                        "calibration_mode": cal.mode,
                                        "cal_intercept": cal.intercept,
                                        "cal_slope": cal.slope,
                                        "beta_time": beta_time,
                                        "valuation_date": str(valuation_date.date()),
                                        "valid_train_rows": int(len(train_valid)),
                                        "valid_eval_rows": int(len(eval_valid)),
                                        "eval_coverage_share": float(len(eval_valid) / max(1, len(eval_idx))),
                                        "mean_eval_neighbors": float(np.nanmean(cdf.loc[eval_valid, diag_cols[0]])),
                                        "mean_eval_neighbor_space_miles": float(np.nanmean(cdf.loc[eval_valid, diag_cols[1]])),
                                        "mean_eval_neighbor_time_days": float(np.nanmean(cdf.loc[eval_valid, diag_cols[2]])),
                                        "class_fallback_share_eval": float(np.nanmean(cdf.loc[eval_valid, diag_cols[3]])),
                                        "strict_eval_history": bool(strict_eval_history),
                                    })
                                    emit_metric(metrics)

    return pd.DataFrame(rows)


def rank_and_select(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    df = metrics.copy()
    # A simple utility for screening: accuracy with penalties for level/trend/shape.
    df["abs_slope_raw"] = df["log_ratio_slope_vs_raw_log_price"].abs()
    df["abs_median_ratio_raw_gap"] = (df["median_ratio_raw"] - 1.0).abs()
    df["abs_curve_level_raw"] = df["raw_curve_level"].abs()
    df["abs_curve_trend_raw"] = df["raw_curve_trend"].abs()
    df["selection_score"] = (
        df["rmse_log_raw"]
        + 0.50 * df["abs_slope_raw"].fillna(0.0)
        + 0.50 * df["abs_median_ratio_raw_gap"].fillna(0.0)
        + 0.50 * df["raw_curve_shape"].fillna(0.0)
    )
    return df.sort_values(["selection_score", "rmse_log_raw"], ascending=True).reset_index(drop=True)


def save_outputs(metrics: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_dir / "metrics_all.csv", index=False)
    if not metrics.empty:
        ranked = rank_and_select(metrics)
        ranked.to_csv(out_dir / "metrics_ranked.csv", index=False)
        for split, sub in ranked.groupby("split"):
            sub.to_csv(out_dir / f"metrics_{split}.csv", index=False)
        # Compact top summaries.
        top_cols = [
            "split", "model", "stage", "target_variant", "learner", "spatial_feature_type",
            "K", "spatial_bw_miles", "time_bw_days", "calibration_mode",
            "rmse_log_raw", "r2_log_raw", "rmse_price_raw", "median_ratio_raw",
            "log_ratio_slope_vs_raw_log_price", "raw_curve_shape", "selection_score",
            "valid_eval_rows", "eval_coverage_share", "mean_eval_neighbors",
            "mean_eval_neighbor_space_miles", "mean_eval_neighbor_time_days",
            "class_fallback_share_eval",
        ]
        cols = [c for c in top_cols if c in ranked.columns]
        ranked[cols].head(100).to_csv(out_dir / "top100_selection.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Market-value target + spatial comparable + AVM experiment runner")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--params-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--parquet-engine", default="pyarrow")
    parser.add_argument("--sample-frac", type=float, default=None)
    parser.add_argument("--sample-seed", type=int, default=2025)
    parser.add_argument("--assess-eval-year", type=int, default=2024)
    parser.add_argument("--valuation-date-mode", default="eval_max", choices=["assessment_date", "eval_max", "eval_min", "eval_median"])
    parser.add_argument("--target-variants", default="raw,global_slope,area_slope:meta_township_code,hedonic_global_slope,hedonic_area_slope:meta_township_code")
    parser.add_argument("--learners", default="ridge,lgbm_l2,lgbm_l1")
    parser.add_argument("--spatial-feature-types", default="scaled_time_adjusted_price,residual,time_adjusted_price,scaled_target_label")
    parser.add_argument("--spatial-stages", default="full_X_plus_spatial,base_plus_local_residual")
    parser.add_argument("--Ks", default="10,20,30")
    parser.add_argument("--spatial-bandwidths-miles", default="0.75,1.0,1.25")
    parser.add_argument("--time-bandwidths-days", default="365,500,730")
    parser.add_argument("--calibration-modes", default="none,median_center,affine")
    parser.add_argument("--calib-frac", type=float, default=0.20)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--lgbm-n-jobs", type=int, default=1)
    parser.add_argument("--target-shrink-n", type=float, default=100.0)
    parser.add_argument("--class-filter-col", default="char_class")
    parser.add_argument("--no-class-filter", action="store_true")
    parser.add_argument("--min-same-class-pool", type=int, default=10)
    parser.add_argument("--no-class-fallback", action="store_true")
    parser.add_argument("--max-neighbor-age-days", type=float, default=None)
    parser.add_argument("--max-spatial-candidates", type=int, default=2048)
    parser.add_argument("--candidate-multiplier", type=int, default=64)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--n-oof-folds", type=int, default=5)
    parser.add_argument("--oof-mode", default="rolling", choices=["rolling", "blocked"])
    parser.add_argument("--n-deciles", type=int, default=10)
    parser.add_argument("--allow-sequential-eval-history", action="store_true")
    parser.add_argument("--no-engineered-features", action="store_true")
    parser.add_argument("--no-strict-feature-screen", action="store_true")
    parser.add_argument("--no-stream-metrics", action="store_true", help="Disable per-candidate metrics_<split>_stream.csv checkpointing.")
    parser.add_argument("--resume", action="store_true", help="Reuse existing metrics_<split>_raw.csv files and run only missing splits.")
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    params = load_params(args.params_path)
    df, predictors, cats, id_vars, assessment_date, assessment_year = load_model_frame(
        data_path=args.data_path,
        params=params,
        sample_frac=args.sample_frac,
        sample_seed=args.sample_seed,
        parquet_engine=args.parquet_engine,
        strict_feature_screen=not args.no_strict_feature_screen,
        add_engineered=not args.no_engineered_features,
    )
    splits = make_quick_splits(df, params, assess_eval_year=args.assess_eval_year)

    target_variants = parse_csv(args.target_variants, str)
    learners = parse_csv(args.learners, str)
    spatial_feature_types = parse_csv(args.spatial_feature_types, str)
    spatial_stages = parse_csv(args.spatial_stages, str)
    Ks = parse_int_csv(args.Ks)
    spatial_bws = parse_float_csv(args.spatial_bandwidths_miles)
    time_bws = parse_float_csv(args.time_bandwidths_days)
    calibration_modes = parse_csv(args.calibration_modes, str)
    class_filter_col = None if args.no_class_filter else args.class_filter_col
    if class_filter_col is not None and class_filter_col not in df.columns:
        log("requested class filter missing; disabling", class_filter_col=class_filter_col)
        class_filter_col = None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "experiment_config.json"
    if args.resume and config_path.exists():
        config_path = out_dir / f"experiment_config_resume_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)
    pd.Series(predictors, name="predictor").to_csv(out_dir / "predictors_used.csv", index=False)
    pd.Series(cats, name="categorical").to_csv(out_dir / "categoricals_used.csv", index=False)

    all_metrics = []
    for split_name, (train_df, eval_df) in splits.items():
        if train_df.empty or eval_df.empty:
            log("empty split skipped", split=split_name, train=len(train_df), eval=len(eval_df))
            continue
        split_metrics_path = out_dir / f"metrics_{split_name}_raw.csv"
        if args.resume and split_metrics_path.exists():
            log("existing split metrics found; skipping split", split=split_name, path=str(split_metrics_path))
            all_metrics.append(pd.read_csv(split_metrics_path))
            continue
        stream_path = None if args.no_stream_metrics else out_dir / f"metrics_{split_name}_stream.csv"
        existing_stream = pd.DataFrame()
        completed_metric_keys = set()
        if stream_path is not None:
            if args.resume:
                existing_stream = load_metric_stream(stream_path)
                completed_metric_keys = metric_keys_from_frame(existing_stream)
                if completed_metric_keys:
                    log("existing streamed metrics found; resuming split", split=split_name, rows=len(existing_stream))
            elif stream_path.exists():
                stream_path.unlink()
        metrics = run_one_split(
            split_name=split_name,
            train_df=train_df,
            eval_df=eval_df,
            params=params,
            predictor_cols=predictors,
            categorical_cols=cats,
            target_variants=target_variants,
            learners=learners,
            spatial_feature_types=spatial_feature_types,
            spatial_stages=spatial_stages,
            Ks=Ks,
            spatial_bws=spatial_bws,
            time_bws=time_bws,
            calibration_modes=calibration_modes,
            calib_frac=float(args.calib_frac),
            valuation_date_mode=args.valuation_date_mode,
            seed=int(args.seed),
            ridge_alpha=float(args.ridge_alpha),
            n_estimators=int(args.n_estimators),
            learning_rate=float(args.learning_rate),
            lgbm_n_jobs=int(args.lgbm_n_jobs),
            oof_mode=args.oof_mode,
            target_shrink_n=float(args.target_shrink_n),
            class_filter_col=class_filter_col,
            min_same_class_pool=int(args.min_same_class_pool),
            allow_class_fallback=not args.no_class_fallback,
            max_neighbor_age_days=args.max_neighbor_age_days,
            max_spatial_candidates=int(args.max_spatial_candidates),
            candidate_multiplier=int(args.candidate_multiplier),
            n_jobs=int(args.n_jobs),
            n_oof_folds=int(args.n_oof_folds),
            n_deciles=int(args.n_deciles),
            strict_eval_history=not args.allow_sequential_eval_history,
            stream_path=stream_path,
            completed_metric_keys=completed_metric_keys,
        )
        if not existing_stream.empty:
            metrics = dedupe_metrics(pd.concat([existing_stream, metrics], axis=0, ignore_index=True))
        all_metrics.append(metrics)
        if not metrics.empty:
            metrics.to_csv(out_dir / f"metrics_{split_name}_raw.csv", index=False)

    if all_metrics:
        metrics_all = pd.concat(all_metrics, axis=0, ignore_index=True)
    else:
        metrics_all = pd.DataFrame()
    save_outputs(metrics_all, out_dir)
    log("finished", out_dir=str(out_dir), rows=len(metrics_all))


if __name__ == "__main__":
    main()
