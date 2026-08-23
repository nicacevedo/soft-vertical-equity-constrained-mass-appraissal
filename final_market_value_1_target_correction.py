"""
Target-correction study for market-value AVMs.

This runner is intentionally narrower than final_market_value_spatial_experiments.py:
it compares linear regression, ridge, and lasso under the target-correction ladder
described in final_market_promt, without adding the local-market prior as a final
X feature.  The local comparable signal is used only for target shrinkage variants.

Leakage controls implemented here:
- chronological train/test and train/assessment splits;
- a trailing calibration slice is cut out of each training window;
- target/time corrections are fit only on the model-training core;
- local comparable targets use prior core sales only, never calibration/eval outcomes;
- sale-validation metadata is used only for row filtering/weights, not as predictors;
- time_sale_* predictors are overwritten per evaluation phase:
  fixed_assessment_date uses one fixed date T for all train/calib/eval rows, while
  actual_sale_date uses each row's real sale date.

Typical smoke run:
python final_market_value_1_target_correction.py \
  --data-path data/CCAO/2025/training_data.parquet \
  --params-path params.yaml \
  --out-dir output/final_mv_target_correction/smoke \
  --sample-frac 0.01 \
  --target-variants raw,strict_raw,weighted_raw,robust_raw,time_global,local_shrink:0.75,robust_local_shrink:0.75 \
  --learners linear,ridge,lasso \
  --calibration-modes none,median_center \
  --preprocess-mode repo \
  --no-tune-alphas \
  --n-jobs 2
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

try:
    from preprocessing.recipes_pipelined import build_model_pipeline as build_repo_model_pipeline
except Exception:  # pragma: no cover - keeps the simple sklearn path usable in stripped environments
    build_repo_model_pipeline = None

try:
    from joblib import Parallel, delayed
except Exception:  # pragma: no cover
    Parallel = None
    delayed = None

from final_market_value_spatial_experiments import (
    CHAR_CLASS_COL,
    DATE_COL,
    HEDONIC_CONTROL_CANDIDATES,
    META_CLASS_COL,
    NBHD_COL,
    TARGET_COL,
    TOWNSHIP_COL,
    TRIAD_COL,
    X_FT_COL,
    Y_FT_COL,
    _fit_hedonic_residuals,
    _ols_slope,
    build_candidate_cache,
    load_model_frame,
    make_quick_splits,
    parse_bool_series,
    safe_log_price,
    timestamp_to_year_float,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)


LOG_T0 = time.perf_counter()

TIME_FEATURE_COLS = [
    "time_sale_year",
    "time_sale_day",
    "time_sale_quarter_of_year",
    "time_sale_month_of_year",
    "time_sale_day_of_year",
    "time_sale_day_of_month",
    "time_sale_day_of_week",
    "time_sale_post_covid",
]

SALE_METADATA_FOR_TARGETING = {
    "meta_sale_count_past_n_years",
}

DEFAULT_TARGET_VARIANTS = (
    "raw",
    "strict_raw",
    "weighted_raw",
    "robust_raw",
    "time_global",
    "time_hedonic_global",
    "time_area:meta_township_code",
    "time_hedonic_area:meta_township_code",
    "local_shrink:0.90",
    "local_shrink:0.75",
    "local_shrink:0.50",
    "local_shrink_adaptive",
    "robust_local_shrink:0.75",
)


def log(message: str, **fields: Any) -> None:
    dt = time.perf_counter() - LOG_T0
    suffix = ""
    if fields:
        suffix = " | " + " | ".join(f"{k}={v}" for k, v in fields.items())
    print(f"[target_correction +{dt:8.1f}s] {message}{suffix}", flush=True)


def parse_csv(raw: Optional[str], cast=str) -> Tuple[Any, ...]:
    if raw is None or str(raw).strip() == "":
        return tuple()
    return tuple(cast(x.strip()) for x in str(raw).split(",") if x.strip())


def parse_float_csv(raw: Optional[str]) -> Tuple[float, ...]:
    return parse_csv(raw, float)


def parse_int_csv(raw: Optional[str]) -> Tuple[int, ...]:
    return parse_csv(raw, int)


def load_params(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:  # sklearn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def as_object_frame(x: Any) -> Any:
    if hasattr(x, "astype"):
        return x.astype("object")
    return np.asarray(x, dtype=object)


def make_simple_preprocessor(df: pd.DataFrame, predictors: Sequence[str], categorical_cols: Sequence[str]) -> ColumnTransformer:
    predictors = [c for c in predictors if c in df.columns]
    categorical_cols = [c for c in categorical_cols if c in predictors]
    numeric_cols = [c for c in predictors if c not in categorical_cols]
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    [
                        ("to_object", FunctionTransformer(as_object_frame, validate=False)),
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_one_hot_encoder()),
                    ]
                ),
                categorical_cols,
            ),
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_cols,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def make_repo_preprocessor(
    predictors: Sequence[str],
    categorical_cols: Sequence[str],
    id_vars: Sequence[str],
) -> Pipeline:
    if build_repo_model_pipeline is None:
        raise RuntimeError(
            "preprocess_mode='repo' requires preprocessing.recipes_pipelined.build_model_pipeline. "
            "Use --preprocess-mode simple to run the fallback sklearn preprocessing."
        )
    return build_repo_model_pipeline(
        pred_vars=list(predictors),
        cat_vars=list(categorical_cols),
        id_vars=list(id_vars),
    )


def make_linear_estimator(name: str, *, alpha: float, seed: int, lasso_max_iter: int) -> Any:
    name = str(name).lower().strip()
    if name == "linear":
        return LinearRegression(fit_intercept=True)
    if name == "ridge":
        return Ridge(alpha=float(alpha), fit_intercept=True, random_state=None)
    if name == "lasso":
        return Lasso(
            alpha=float(alpha),
            fit_intercept=True,
            max_iter=int(lasso_max_iter),
            tol=1e-4,
            selection="cyclic",
            random_state=int(seed),
        )
    raise ValueError(f"Unknown learner: {name}")


@dataclass
class LinearModelSpec:
    learner: str
    predictors: List[str]
    categorical_cols: List[str]
    id_vars: List[str]
    alpha: float
    seed: int
    lasso_max_iter: int = 5000
    preprocess_mode: str = "repo"

    def input_columns(self, df: pd.DataFrame) -> List[str]:
        # Keep this explicit so repo/id-aware preprocessing can be used without
        # accidentally exposing non-predictive sale validation metadata.
        cols = list(dict.fromkeys(list(self.predictors) + list(self.id_vars)))
        return [c for c in cols if c in df.columns]

    def new_pipeline(self, fit_df: pd.DataFrame) -> Pipeline:
        est = make_linear_estimator(
            self.learner,
            alpha=self.alpha,
            seed=self.seed,
            lasso_max_iter=self.lasso_max_iter,
        )
        mode = str(self.preprocess_mode).lower().strip()
        if mode == "repo":
            pre = make_repo_preprocessor(
                predictors=[c for c in self.predictors if c in fit_df.columns],
                categorical_cols=[c for c in self.categorical_cols if c in fit_df.columns],
                id_vars=[c for c in self.id_vars if c in fit_df.columns],
            )
            return Pipeline(
                [
                    ("prep", pre),
                    ("est", est),
                ]
            )
        if mode == "simple":
            pre = make_simple_preprocessor(fit_df, self.predictors, self.categorical_cols)
            return Pipeline(
                [
                    ("prep", pre),
                    ("scale", StandardScaler(with_mean=False)),
                    ("est", est),
                ]
            )
        raise ValueError("--preprocess-mode must be repo or simple")


def fit_model(
    spec: LinearModelSpec,
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    *,
    sample_weight: Optional[np.ndarray] = None,
) -> Pipeline:
    model = spec.new_pipeline(train_df)
    fit_kwargs = {}
    if sample_weight is not None:
        fit_sig = inspect.signature(model.named_steps["est"].fit)
        if "sample_weight" in fit_sig.parameters:
            fit_kwargs["est__sample_weight"] = np.asarray(sample_weight, dtype=float)
    model.fit(train_df[spec.input_columns(train_df)], np.asarray(y_train, dtype=float), **fit_kwargs)
    return model


def predict_model(model: Pipeline, df: pd.DataFrame, spec: LinearModelSpec) -> np.ndarray:
    if len(df) == 0:
        return np.array([], dtype=float)
    return np.asarray(model.predict(df[spec.input_columns(df)]), dtype=float).reshape(-1)


def fit_predict_model(
    spec: LinearModelSpec,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    y_train: np.ndarray,
    *,
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Pipeline]:
    model = fit_model(spec, train_df, y_train, sample_weight=sample_weight)
    return predict_model(model, eval_df, spec), model


def temporal_oof_predictions(
    spec: LinearModelSpec,
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    *,
    sample_weight: Optional[np.ndarray],
    n_folds: int,
    mode: str,
) -> np.ndarray:
    n = len(train_df)
    oof = np.full(n, np.nan, dtype=float)
    if n_folds <= 1 or n < max(50, n_folds * 10):
        pred, _ = fit_predict_model(spec, train_df, train_df, y_train, sample_weight=sample_weight)
        return pred

    mode = str(mode).lower().strip()
    if mode not in {"rolling", "blocked"}:
        raise ValueError("--oof-mode must be rolling or blocked")

    fold_ids = np.array_split(np.arange(n), int(n_folds))
    for val_idx in fold_ids:
        if mode == "rolling":
            tr_idx = np.arange(0, int(val_idx[0]), dtype=int)
            if len(tr_idx) < max(50, len(val_idx) // 2):
                continue
        else:
            tr_idx = np.setdiff1d(np.arange(n), val_idx)
        if len(tr_idx) == 0 or len(val_idx) == 0:
            continue
        sw = None if sample_weight is None else np.asarray(sample_weight, dtype=float)[tr_idx]
        pred, _ = fit_predict_model(
            spec,
            train_df.iloc[tr_idx].copy(),
            train_df.iloc[val_idx].copy(),
            np.asarray(y_train, dtype=float)[tr_idx],
            sample_weight=sw,
        )
        oof[val_idx] = pred

    if np.any(~np.isfinite(oof)):
        if mode == "rolling":
            finite = np.isfinite(oof)
            fill = float(np.nanmedian(oof[finite])) if finite.any() else float(np.nanmedian(y_train))
            oof[~finite] = fill
        else:
            fill_pred, _ = fit_predict_model(spec, train_df, train_df, y_train, sample_weight=sample_weight)
            oof[~np.isfinite(oof)] = fill_pred[~np.isfinite(oof)]
    return oof


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
        raise ValueError(f"Unknown calibration mode: {self.mode}")


def fit_calibrator(mode: str, yhat_calib: np.ndarray, y_calib_reference: np.ndarray) -> Calibrator:
    mode = str(mode).strip()
    if mode == "none" or len(yhat_calib) == 0:
        return Calibrator(mode="none")
    yhat = np.asarray(yhat_calib, dtype=float)
    y = np.asarray(y_calib_reference, dtype=float)
    mask = np.isfinite(yhat) & np.isfinite(y)
    if mask.sum() == 0:
        return Calibrator(mode="none")
    resid = y[mask] - yhat[mask]
    if mode == "median_center":
        return Calibrator(mode=mode, intercept=float(np.nanmedian(resid)), slope=1.0)
    if mode == "affine":
        if mask.sum() < 20 or np.nanstd(yhat[mask]) <= 0:
            return Calibrator(mode="median_center", intercept=float(np.nanmedian(resid)), slope=1.0)
        lr = LinearRegression().fit(yhat[mask].reshape(-1, 1), y[mask])
        return Calibrator(mode=mode, intercept=float(lr.intercept_), slope=float(lr.coef_[0]))
    raise ValueError(f"Unknown calibration mode: {mode}")


def split_core_calibration(train_df: pd.DataFrame, calib_frac: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if calib_frac <= 0:
        return train_df.copy(), train_df.iloc[0:0].copy()
    train_df = train_df.sort_values(DATE_COL).reset_index(drop=True)
    n = len(train_df)
    cut = int((1.0 - float(calib_frac)) * n)
    cut = min(max(cut, 100), n)
    return train_df.iloc[:cut].copy(), train_df.iloc[cut:].copy()


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


def infer_time_sale_day_origin(df: pd.DataFrame) -> Optional[pd.Timestamp]:
    if "time_sale_day" not in df.columns or DATE_COL not in df.columns:
        return None
    day = pd.to_numeric(df["time_sale_day"], errors="coerce")
    date = pd.to_datetime(df[DATE_COL], errors="coerce")
    mask = day.notna() & date.notna()
    if mask.sum() < 10:
        return None
    origins = date.loc[mask] - pd.to_timedelta(day.loc[mask].astype(float) - 1.0, unit="D")
    try:
        return pd.Timestamp(origins.mode(dropna=True).iloc[0]).normalize()
    except Exception:
        return pd.Timestamp(origins.median()).normalize()


def _date_feature_frame(dates: pd.Series, origin: Optional[pd.Timestamp]) -> pd.DataFrame:
    dt = pd.to_datetime(dates, errors="coerce")
    out = pd.DataFrame(index=dt.index)
    out["time_sale_year"] = dt.dt.year.astype(float)
    if origin is not None:
        out["time_sale_day"] = (dt.dt.normalize() - origin).dt.days.astype(float) + 1.0
    else:
        out["time_sale_day"] = dt.astype("int64") / 1e9 / 86400.0
    out["time_sale_quarter_of_year"] = "Q" + dt.dt.quarter.astype("Int64").astype(str)
    out["time_sale_month_of_year"] = dt.dt.month.astype(float)
    out["time_sale_day_of_year"] = dt.dt.dayofyear.astype(float)
    out["time_sale_day_of_month"] = dt.dt.day.astype(float)
    # Existing CCAO feature appears to use 1=Sunday, ..., 7=Saturday.
    out["time_sale_day_of_week"] = ((dt.dt.dayofweek + 1) % 7 + 1).astype(float)
    out["time_sale_post_covid"] = dt >= pd.Timestamp("2020-03-01")
    return out


def apply_temporal_feature_policy(
    df: pd.DataFrame,
    *,
    predictors: Sequence[str],
    eval_phase: str,
    fixed_date: pd.Timestamp,
    time_day_origin: Optional[pd.Timestamp],
) -> pd.DataFrame:
    out = df.copy()
    available = [c for c in TIME_FEATURE_COLS if c in out.columns and c in predictors]
    if not available:
        return out
    if eval_phase == "fixed_assessment_date":
        dates = pd.Series(pd.Timestamp(fixed_date), index=out.index)
    elif eval_phase == "actual_sale_date":
        dates = pd.to_datetime(out[DATE_COL], errors="coerce")
    else:
        raise ValueError(f"Unknown eval_phase: {eval_phase}")
    feats = _date_feature_frame(dates, time_day_origin)
    for c in available:
        out[c] = feats[c]
    return out


def target_year_float_for_phase(df: pd.DataFrame, eval_phase: str, fixed_date: pd.Timestamp) -> np.ndarray:
    if eval_phase == "fixed_assessment_date":
        return np.full(len(df), timestamp_to_year_float(pd.Timestamp(fixed_date)), dtype=float)
    if eval_phase == "actual_sale_date":
        return df["sale_year_float"].to_numpy(dtype=float)
    raise ValueError(f"Unknown eval_phase: {eval_phase}")


def _json_get_bool(raw: Any, key: str) -> Optional[bool]:
    if raw is None or pd.isna(raw):
        return None
    if isinstance(raw, str):
        text = raw.strip()
        if not text or text.lower() == "none":
            return None
        try:
            data = json.loads(text)
        except Exception:
            return None
    elif isinstance(raw, dict):
        data = raw
    else:
        return None
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    return None


def _json_get_str(raw: Any, key: str) -> Optional[str]:
    if raw is None or pd.isna(raw):
        return None
    if isinstance(raw, str):
        text = raw.strip()
        if not text or text.lower() == "none":
            return None
        try:
            data = json.loads(text)
        except Exception:
            return None
    elif isinstance(raw, dict):
        data = raw
    else:
        return None
    value = data.get(key)
    if value is None:
        return None
    return str(value)


def strict_validation_mask(df: pd.DataFrame, strict_deed_types: Sequence[str]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    if "sv_is_outlier" in df.columns:
        mask &= ~parse_bool_series(df["sv_is_outlier"], default=False)
    if "ind_pin_is_multicard" in df.columns:
        mask &= ~parse_bool_series(df["ind_pin_is_multicard"], default=False)
    if "meta_sale_deed_type" in df.columns and strict_deed_types:
        allowed = {str(x).strip() for x in strict_deed_types}
        mask &= df["meta_sale_deed_type"].astype(str).str.strip().isin(allowed)
    if "sv_review_json" in df.columns:
        arms = df["sv_review_json"].map(lambda x: _json_get_bool(x, "is_arms_length"))
        flip = df["sv_review_json"].map(lambda x: _json_get_bool(x, "is_flip"))
        class_change = df["sv_review_json"].map(lambda x: _json_get_bool(x, "has_class_change"))
        char_change = df["sv_review_json"].map(lambda x: _json_get_str(x, "has_characteristic_change"))
        mask &= arms.map(lambda x: True if x is None else bool(x))
        mask &= ~flip.fillna(False).astype(bool)
        mask &= ~class_change.fillna(False).astype(bool)
        mask &= ~char_change.fillna("").astype(str).str.lower().str.contains("major", na=False)
    return mask.fillna(False).astype(bool)


def reliability_weights(df: pd.DataFrame, deed_weights: Dict[str, float]) -> np.ndarray:
    w = np.ones(len(df), dtype=float)
    if "meta_sale_deed_type" in df.columns:
        deed = df["meta_sale_deed_type"].astype(str).str.strip().to_numpy()
        w *= np.asarray([float(deed_weights.get(x, 0.70)) for x in deed], dtype=float)
    if "meta_sale_count_past_n_years" in df.columns:
        count = pd.to_numeric(df["meta_sale_count_past_n_years"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        w *= np.power(0.90, np.minimum(np.maximum(count, 0.0), 5.0))
    if "sv_review_json" in df.columns:
        arms = df["sv_review_json"].map(lambda x: _json_get_bool(x, "is_arms_length"))
        flip = df["sv_review_json"].map(lambda x: _json_get_bool(x, "is_flip"))
        class_change = df["sv_review_json"].map(lambda x: _json_get_bool(x, "has_class_change"))
        char_change = df["sv_review_json"].map(lambda x: _json_get_str(x, "has_characteristic_change"))
        w *= np.where(arms.map(lambda x: False if x is None else not bool(x)).to_numpy(dtype=bool), 0.35, 1.0)
        w *= np.where(flip.fillna(False).to_numpy(dtype=bool), 0.60, 1.0)
        w *= np.where(class_change.fillna(False).to_numpy(dtype=bool), 0.60, 1.0)
        char = char_change.fillna("").astype(str).str.lower()
        w *= np.where(char.str.contains("major", na=False).to_numpy(dtype=bool), 0.65, 1.0)
        w *= np.where(char.str.contains("minor", na=False).to_numpy(dtype=bool), 0.85, 1.0)
    if "sv_is_outlier" in df.columns:
        w *= np.where(parse_bool_series(df["sv_is_outlier"], default=False).to_numpy(dtype=bool), 0.10, 1.0)
    return np.clip(w, 0.05, 1.0)


def parse_deed_weights(raw: str) -> Dict[str, float]:
    out: Dict[str, float] = {"01": 1.00, "02": 0.85, "05": 0.75}
    if raw is None or str(raw).strip() == "":
        return out
    for item in str(raw).split(","):
        if not item.strip():
            continue
        if ":" not in item:
            raise ValueError(f"Invalid deed weight entry: {item!r}. Use code:weight.")
        k, v = item.split(":", 1)
        out[str(k).strip()] = float(v)
    return out


def robust_huber_weights(residual: np.ndarray, reliability: np.ndarray, c: float) -> np.ndarray:
    r = np.asarray(residual, dtype=float)
    q = np.asarray(reliability, dtype=float)
    finite = np.isfinite(r)
    if finite.sum() < 10:
        return np.clip(q, 0.05, 1.0)
    med = float(np.nanmedian(r[finite]))
    mad = float(np.nanmedian(np.abs(r[finite] - med)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = float(np.nanstd(r[finite]))
    if not np.isfinite(scale) or scale <= 1e-8:
        return np.clip(q, 0.05, 1.0)
    cutoff = float(c) * scale
    abs_r = np.abs(r)
    huber = np.ones_like(abs_r)
    big = abs_r > cutoff
    huber[big] = cutoff / np.maximum(abs_r[big], 1e-12)
    return np.clip(q * huber, 0.02, 1.0)


@dataclass
class TimeAdjuster:
    variant: str
    beta_global: float
    area_col: Optional[str] = None
    beta_by_area: Optional[Dict[str, float]] = None
    residual_model: Optional[Pipeline] = None

    def beta_for(self, df: pd.DataFrame) -> np.ndarray:
        if self.area_col is None or self.beta_by_area is None or self.area_col not in df.columns:
            return np.full(len(df), float(self.beta_global), dtype=float)
        vals = df[self.area_col].astype("object").where(df[self.area_col].notna(), "NA").astype(str).to_numpy()
        return np.asarray([self.beta_by_area.get(v, self.beta_global) for v in vals], dtype=float)

    def transform(self, df: pd.DataFrame, target_year_float: np.ndarray) -> np.ndarray:
        y = df["y_raw_log"].to_numpy(dtype=float)
        delta = np.asarray(target_year_float, dtype=float) - df["sale_year_float"].to_numpy(dtype=float)
        return y + self.beta_for(df) * delta


def normalize_time_variant(name: str) -> str:
    v = str(name).strip()
    aliases = {
        "global_slope": "time_global",
        "hedonic_global_slope": "time_hedonic_global",
    }
    if v in aliases:
        return aliases[v]
    if v.startswith("area_slope"):
        return "time_" + v.replace("area_slope", "area", 1)
    if v.startswith("hedonic_area_slope"):
        return "time_" + v.replace("hedonic_area_slope", "hedonic_area", 1)
    return v


def fit_time_adjuster(
    variant: str,
    core_df: pd.DataFrame,
    *,
    predictor_cols: Sequence[str],
    categorical_cols: Sequence[str],
    seed: int,
    shrink_n: float,
) -> TimeAdjuster:
    variant = normalize_time_variant(variant)
    y = core_df["y_raw_log"].to_numpy(dtype=float)
    t = core_df["sale_year_float"].to_numpy(dtype=float)
    residuals = y
    residual_model = None
    if variant.startswith("time_hedonic_"):
        residuals, residual_model, _, _ = _fit_hedonic_residuals(
            core_df,
            y,
            predictor_cols,
            categorical_cols,
            seed=seed,
        )
    if variant in {"time_global", "time_hedonic_global"}:
        beta = _ols_slope(t, residuals)
        return TimeAdjuster(variant=variant, beta_global=beta, residual_model=residual_model)
    if variant.startswith("time_area") or variant.startswith("time_hedonic_area"):
        parts = variant.split(":", 1)
        area_col = parts[1] if len(parts) == 2 and parts[1] else TOWNSHIP_COL
        if area_col not in core_df.columns:
            raise ValueError(f"Area column for target variant {variant} not found: {area_col}")
        beta_global = _ols_slope(t, residuals)
        vals = core_df[area_col].astype("object").where(core_df[area_col].notna(), "NA").astype(str)
        beta_by_area = {}
        for g, idx in vals.groupby(vals).groups.items():
            idx_arr = np.asarray(list(idx), dtype=int)
            beta_g_raw = _ols_slope(t[idx_arr], residuals[idx_arr])
            n_g = float(len(idx_arr))
            w = n_g / (n_g + float(shrink_n))
            beta_by_area[str(g)] = float(w * beta_g_raw + (1.0 - w) * beta_global)
        return TimeAdjuster(
            variant=variant,
            beta_global=beta_global,
            area_col=area_col,
            beta_by_area=beta_by_area,
            residual_model=residual_model,
        )
    raise ValueError(f"Unknown time target variant: {variant}")


@dataclass
class LocalPrior:
    mean: np.ndarray
    var: np.ndarray
    n_used: np.ndarray
    mean_space: np.ndarray
    mean_time: np.ndarray
    fallback: np.ndarray
    candidate_count: np.ndarray


def _local_prior_from_cache(
    df: pd.DataFrame,
    cache: Any,
    *,
    K: int,
    spatial_bw_miles: float,
    time_bw_days: float,
    y_raw: np.ndarray,
    beta_time: float,
    target_year_float: np.ndarray,
    base_pred: np.ndarray,
) -> LocalPrior:
    n = len(df)
    mean = np.full(n, np.nan, dtype=float)
    var = np.full(n, np.nan, dtype=float)
    n_used = np.zeros(n, dtype=int)
    mean_space = np.full(n, np.nan, dtype=float)
    mean_time = np.full(n, np.nan, dtype=float)
    fallback = np.zeros(n, dtype=bool)
    candidate_count = np.zeros(n, dtype=int)
    sale_year = df["sale_year_float"].to_numpy(dtype=float)

    for pos, i_raw in enumerate(cache.target_indices):
        i = int(i_raw)
        idx = cache.candidate_indices[pos]
        d_space = cache.d_space_miles[pos]
        d_time = cache.d_time_days[pos]
        fallback[i] = bool(cache.fallback[pos])
        candidate_count[i] = int(cache.candidate_count[pos])
        if idx.size == 0 or not np.isfinite(base_pred[i]):
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
        vals = (
            y_raw[j]
            + float(beta_time) * (float(target_year_float[i]) - sale_year[j])
            + float(base_pred[i])
            - base_pred[j]
        )
        chosen_space = d_space[chosen]
        chosen_time = d_time[chosen]
        finite_vals = np.isfinite(vals) & np.isfinite(weights) & np.isfinite(base_pred[j])
        if not finite_vals.any():
            continue
        vals = vals[finite_vals]
        weights = weights[finite_vals]
        chosen_space = chosen_space[finite_vals]
        chosen_time = chosen_time[finite_vals]
        sw = float(weights.sum())
        if sw <= 0.0 or not np.isfinite(sw):
            continue
        mu = float(np.sum(weights * vals) / sw)
        mean[i] = mu
        var[i] = float(np.sum(weights * (vals - mu) ** 2) / sw)
        n_used[i] = int(len(vals))
        mean_space[i] = float(np.sum(weights * chosen_space) / sw)
        mean_time[i] = float(np.sum(weights * chosen_time) / sw)

    return LocalPrior(
        mean=mean,
        var=var,
        n_used=n_used,
        mean_space=mean_space,
        mean_time=mean_time,
        fallback=fallback,
        candidate_count=candidate_count,
    )


def compute_local_prior(
    combined: pd.DataFrame,
    *,
    core_idx: np.ndarray,
    calib_idx: np.ndarray,
    eval_idx: np.ndarray,
    target_year_float: np.ndarray,
    y_time_all: np.ndarray,
    predictors: Sequence[str],
    categorical_cols: Sequence[str],
    id_vars: Sequence[str],
    preprocess_mode: str,
    seed: int,
    base_ridge_alpha: float,
    n_oof_folds: int,
    oof_mode: str,
    K: int,
    spatial_bw_miles: float,
    time_bw_days: float,
    class_filter_col: Optional[str],
    min_same_class_pool: int,
    allow_class_fallback: bool,
    max_neighbor_age_days: Optional[float],
    max_spatial_candidates: int,
    candidate_multiplier: int,
    n_jobs: int,
) -> Tuple[LocalPrior, Dict[str, Any]]:
    base_spec = LinearModelSpec(
        learner="ridge",
        predictors=list(predictors),
        categorical_cols=list(categorical_cols),
        id_vars=list(id_vars),
        alpha=float(base_ridge_alpha),
        seed=int(seed),
        preprocess_mode=str(preprocess_mode),
    )
    y_core = np.asarray(y_time_all, dtype=float)[core_idx]
    core_df = combined.iloc[core_idx].copy()
    log("local-prior baseline fit", core_rows=len(core_idx), K=K, hs=spatial_bw_miles, ht=time_bw_days)
    oof_core = temporal_oof_predictions(
        base_spec,
        core_df,
        y_core,
        sample_weight=None,
        n_folds=int(n_oof_folds),
        mode=oof_mode,
    )
    pred_eval, base_model = fit_predict_model(base_spec, core_df, combined.iloc[eval_idx].copy(), y_core)
    pred_calib = predict_model(base_model, combined.iloc[calib_idx].copy(), base_spec) if len(calib_idx) else np.array([])

    base_pred = np.full(len(combined), np.nan, dtype=float)
    base_pred[core_idx] = oof_core
    if len(calib_idx):
        base_pred[calib_idx] = pred_calib
    base_pred[eval_idx] = pred_eval
    if np.any(~np.isfinite(base_pred[core_idx])):
        fill = float(np.nanmedian(y_core))
        base_pred[core_idx[~np.isfinite(base_pred[core_idx])]] = fill

    beta_time = _ols_slope(
        combined.loc[core_idx, "sale_year_float"].to_numpy(dtype=float),
        combined.loc[core_idx, "y_raw_log"].to_numpy(dtype=float),
    )
    history_for_eval = core_idx
    cache_core = build_candidate_cache(
        combined,
        core_idx,
        core_idx,
        K=K,
        class_col=class_filter_col,
        min_same_class_pool=min_same_class_pool,
        allow_class_fallback=allow_class_fallback,
        max_neighbor_age_days=max_neighbor_age_days,
        max_spatial_candidates=max_spatial_candidates,
        candidate_multiplier=candidate_multiplier,
        n_jobs=n_jobs,
    )
    cache_calib = (
        build_candidate_cache(
            combined,
            calib_idx,
            core_idx,
            K=K,
            class_col=class_filter_col,
            min_same_class_pool=min_same_class_pool,
            allow_class_fallback=allow_class_fallback,
            max_neighbor_age_days=max_neighbor_age_days,
            max_spatial_candidates=max_spatial_candidates,
            candidate_multiplier=candidate_multiplier,
            n_jobs=n_jobs,
        )
        if len(calib_idx)
        else None
    )
    cache_eval = build_candidate_cache(
        combined,
        eval_idx,
        history_for_eval,
        K=K,
        class_col=class_filter_col,
        min_same_class_pool=min_same_class_pool,
        allow_class_fallback=allow_class_fallback,
        max_neighbor_age_days=max_neighbor_age_days,
        max_spatial_candidates=max_spatial_candidates,
        candidate_multiplier=candidate_multiplier,
        n_jobs=n_jobs,
    )

    y_raw_all = combined["y_raw_log"].to_numpy(dtype=float)
    prior_core = _local_prior_from_cache(
        combined,
        cache_core,
        K=K,
        spatial_bw_miles=spatial_bw_miles,
        time_bw_days=time_bw_days,
        y_raw=y_raw_all,
        beta_time=beta_time,
        target_year_float=target_year_float,
        base_pred=base_pred,
    )
    prior_eval = _local_prior_from_cache(
        combined,
        cache_eval,
        K=K,
        spatial_bw_miles=spatial_bw_miles,
        time_bw_days=time_bw_days,
        y_raw=y_raw_all,
        beta_time=beta_time,
        target_year_float=target_year_float,
        base_pred=base_pred,
    )
    if cache_calib is not None:
        prior_calib = _local_prior_from_cache(
            combined,
            cache_calib,
            K=K,
            spatial_bw_miles=spatial_bw_miles,
            time_bw_days=time_bw_days,
            y_raw=y_raw_all,
            beta_time=beta_time,
            target_year_float=target_year_float,
            base_pred=base_pred,
        )
    else:
        prior_calib = LocalPrior(
            mean=np.full(len(combined), np.nan),
            var=np.full(len(combined), np.nan),
            n_used=np.zeros(len(combined), dtype=int),
            mean_space=np.full(len(combined), np.nan),
            mean_time=np.full(len(combined), np.nan),
            fallback=np.zeros(len(combined), dtype=bool),
            candidate_count=np.zeros(len(combined), dtype=int),
        )

    def merge(name: str) -> np.ndarray:
        arr = np.full(len(combined), np.nan, dtype=float)
        arr[core_idx] = getattr(prior_core, name)[core_idx]
        if len(calib_idx):
            arr[calib_idx] = getattr(prior_calib, name)[calib_idx]
        arr[eval_idx] = getattr(prior_eval, name)[eval_idx]
        return arr

    prior = LocalPrior(
        mean=merge("mean"),
        var=merge("var"),
        n_used=np.nan_to_num(merge("n_used"), nan=0.0).astype(int),
        mean_space=merge("mean_space"),
        mean_time=merge("mean_time"),
        fallback=np.nan_to_num(merge("fallback"), nan=0.0).astype(bool),
        candidate_count=np.nan_to_num(merge("candidate_count"), nan=0.0).astype(int),
    )
    diag = {
        "local_K": int(K),
        "local_spatial_bw_miles": float(spatial_bw_miles),
        "local_time_bw_days": float(time_bw_days),
        "local_beta_time": float(beta_time),
        "local_core_coverage": float(np.isfinite(prior.mean[core_idx]).mean()) if len(core_idx) else np.nan,
        "local_calib_coverage": float(np.isfinite(prior.mean[calib_idx]).mean()) if len(calib_idx) else np.nan,
        "local_eval_coverage": float(np.isfinite(prior.mean[eval_idx]).mean()) if len(eval_idx) else np.nan,
        "local_eval_mean_neighbors": float(np.nanmean(prior.n_used[eval_idx])) if len(eval_idx) else np.nan,
        "local_eval_mean_space_miles": float(np.nanmean(prior.mean_space[eval_idx])) if len(eval_idx) else np.nan,
        "local_eval_mean_time_days": float(np.nanmean(prior.mean_time[eval_idx])) if len(eval_idx) else np.nan,
    }
    return prior, diag


@dataclass
class PreparedTarget:
    variant: str
    family: str
    y_all: np.ndarray
    sample_weight_all: np.ndarray
    train_mask_all: np.ndarray
    diagnostics: Dict[str, Any]


def parse_alpha_from_variant(variant: str, default: float) -> float:
    if ":" not in variant:
        return float(default)
    return float(variant.split(":", 1)[1])


def prepare_target_variant(
    variant_raw: str,
    *,
    combined: pd.DataFrame,
    core_idx: np.ndarray,
    target_year_float: np.ndarray,
    strict_mask: np.ndarray,
    reliability: np.ndarray,
    global_time_adjuster: TimeAdjuster,
    time_adjusters: Dict[str, TimeAdjuster],
    local_prior: Optional[LocalPrior],
    local_diag: Dict[str, Any],
    robust_weight_base_pred: np.ndarray,
    robust_c: float,
    adaptive_alpha_min: float,
    adaptive_alpha_max: float,
) -> PreparedTarget:
    variant = normalize_time_variant(str(variant_raw).strip())
    y_raw = combined["y_raw_log"].to_numpy(dtype=float)
    y_time_global = global_time_adjuster.transform(combined, target_year_float)
    train_mask = np.ones(len(combined), dtype=bool)
    weights = np.ones(len(combined), dtype=float)
    diag: Dict[str, Any] = {}

    if variant == "raw":
        y = y_raw.copy()
        family = "raw"
    elif variant == "strict_raw":
        y = y_raw.copy()
        train_mask &= strict_mask
        family = "strict_validation"
    elif variant == "weighted_raw":
        y = y_raw.copy()
        weights *= reliability
        family = "soft_validation"
    elif variant == "robust_raw":
        y = y_raw.copy()
        residual = y_raw - robust_weight_base_pred
        weights *= robust_huber_weights(residual, reliability, robust_c)
        family = "robust_noisy_label"
        diag["robust_weight_mean"] = float(np.nanmean(weights[core_idx]))
        diag["robust_weight_p10"] = float(np.nanquantile(weights[core_idx], 0.10))
    elif variant.startswith("time_"):
        adj = time_adjusters.get(variant)
        if adj is None:
            raise ValueError(f"Time adjuster not fit for {variant}")
        y = adj.transform(combined, target_year_float)
        family = "time_adjusted"
        diag["beta_global"] = float(adj.beta_global)
        diag["area_col"] = adj.area_col
    elif variant.startswith("local_shrink") or variant.startswith("robust_local_shrink"):
        if local_prior is None:
            raise ValueError(f"Local prior requested by {variant} but was not computed.")
        finite_local = np.isfinite(local_prior.mean)
        mu = np.where(finite_local, local_prior.mean, y_time_global)
        if variant == "local_shrink_adaptive":
            residual = y_time_global[core_idx] - robust_weight_base_pred[core_idx]
            sale_sigma2 = float(np.nanvar(residual[np.isfinite(residual)]))
            if not np.isfinite(sale_sigma2) or sale_sigma2 <= 1e-8:
                sale_sigma2 = 0.05 ** 2
            tau2 = np.asarray(local_prior.var, dtype=float) / np.maximum(local_prior.n_used.astype(float), 1.0)
            tau2 = np.where(np.isfinite(tau2), tau2, sale_sigma2)
            sigma2 = sale_sigma2 / np.maximum(reliability, 0.05)
            alpha = tau2 / np.maximum(tau2 + sigma2, 1e-12)
            alpha = np.clip(alpha, float(adaptive_alpha_min), float(adaptive_alpha_max))
            y = alpha * y_time_global + (1.0 - alpha) * mu
            family = "local_market_belief_shrinkage"
            diag["target_alpha_mode"] = "adaptive"
            diag["target_alpha_mean_core"] = float(np.nanmean(alpha[core_idx]))
            diag["target_alpha_p10_core"] = float(np.nanquantile(alpha[core_idx], 0.10))
            diag["target_alpha_p90_core"] = float(np.nanquantile(alpha[core_idx], 0.90))
        else:
            alpha = parse_alpha_from_variant(variant, default=0.75)
            y = float(alpha) * y_time_global + (1.0 - float(alpha)) * mu
            family = "local_market_belief_shrinkage"
            diag["target_alpha_mode"] = "global"
            diag["target_alpha"] = float(alpha)
        if variant.startswith("robust_local_shrink"):
            residual = y_time_global - robust_weight_base_pred
            weights *= robust_huber_weights(residual, reliability, robust_c)
            family = "robust_shrinkage"
            diag["robust_weight_mean"] = float(np.nanmean(weights[core_idx]))
        diag.update(local_diag)
        diag["local_missing_core"] = float((~finite_local[core_idx]).mean()) if len(core_idx) else np.nan
    else:
        raise ValueError(f"Unknown target variant: {variant_raw}")

    diag_idx = core_idx[train_mask[core_idx]]
    diagnostics = {
        **diag,
        "target_train_mean": float(np.nanmean(y[diag_idx])) if len(diag_idx) else np.nan,
        "target_train_std": float(np.nanstd(y[diag_idx])) if len(diag_idx) else np.nan,
        "sample_weight_mean_core": float(np.nanmean(weights[diag_idx])) if len(diag_idx) else np.nan,
        "train_mask_share_core": float(np.mean(train_mask[core_idx])) if len(core_idx) else np.nan,
    }
    return PreparedTarget(
        variant=variant_raw,
        family=family,
        y_all=np.asarray(y, dtype=float),
        sample_weight_all=np.asarray(weights, dtype=float),
        train_mask_all=np.asarray(train_mask, dtype=bool),
        diagnostics=diagnostics,
    )


def cod_from_ratio(ratio: np.ndarray) -> float:
    r = np.asarray(ratio, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return np.nan
    med = float(np.nanmedian(r))
    if not np.isfinite(med) or med == 0.0:
        return np.nan
    return float(100.0 * np.nanmean(np.abs(r - med)) / med)


def prd_from_values(predicted: np.ndarray, actual: np.ndarray) -> float:
    p = np.asarray(predicted, dtype=float)
    a = np.asarray(actual, dtype=float)
    mask = np.isfinite(p) & np.isfinite(a) & (a > 0.0)
    if mask.sum() == 0:
        return np.nan
    p = p[mask]
    a = a[mask]
    ratio = p / a
    mean_ratio = float(np.nanmean(ratio))
    weighted_mean = float(np.nansum(p) / np.nansum(a)) if np.nansum(a) > 0 else np.nan
    if not np.isfinite(weighted_mean) or weighted_mean == 0.0:
        return np.nan
    return float(mean_ratio / weighted_mean)


def prb_from_values(predicted: np.ndarray, actual: np.ndarray) -> float:
    p = np.asarray(predicted, dtype=float)
    a = np.asarray(actual, dtype=float)
    mask = np.isfinite(p) & np.isfinite(a) & (p > 0.0) & (a > 0.0)
    if mask.sum() < 2:
        return np.nan
    p = p[mask]
    a = a[mask]
    ratio = p / a
    med = float(np.nanmedian(ratio))
    if not np.isfinite(med) or med == 0.0:
        return np.nan
    lhs = (ratio - med) / med
    proxy = 0.5 * (p / med + a)
    valid = np.isfinite(lhs) & np.isfinite(proxy) & (proxy > 0.0)
    if valid.sum() < 2 or np.nanstd(np.log2(proxy[valid])) <= 0:
        return np.nan
    x = np.log2(proxy[valid])
    y = lhs[valid]
    return float(np.polyfit(x, y, 1)[0])


def vei_from_values(predicted: np.ndarray, actual: np.ndarray) -> float:
    p = np.asarray(predicted, dtype=float)
    a = np.asarray(actual, dtype=float)
    mask = np.isfinite(p) & np.isfinite(a) & (p > 0.0) & (a > 0.0)
    p = p[mask]
    a = a[mask]
    n = len(p)
    if n < 20:
        return np.nan
    if n <= 50:
        k = 2
    elif n <= 500:
        k = 4
    else:
        k = 10
    ratio = p / a
    med = float(np.nanmedian(ratio))
    if not np.isfinite(med) or med == 0.0:
        return np.nan
    proxy = 0.5 * a + 0.5 * (p / med)
    order = np.argsort(proxy, kind="mergesort")
    chunks = np.array_split(np.arange(n), k)
    first = order[chunks[0]]
    last = order[chunks[-1]]
    if first.size < 10 or last.size < 10:
        return np.nan
    m_first = float(np.nanmedian(ratio[first]))
    m_last = float(np.nanmedian(ratio[last]))
    if not (np.isfinite(m_first) and np.isfinite(m_last)):
        return np.nan
    return float(100.0 * (m_last - m_first) / med)


def oos_r2_log(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    tr = np.asarray(y_train, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    if mask.sum() == 0:
        return np.nan
    denom = float(np.sum((y[mask] - np.nanmean(tr)) ** 2))
    if denom <= 0.0 or not np.isfinite(denom):
        return np.nan
    return float(1.0 - np.sum((y[mask] - p[mask]) ** 2) / denom)


def decile_curve(
    *,
    y_basis_log: np.ndarray,
    y_actual_log: np.ndarray,
    y_pred_log: np.ndarray,
    n_deciles: int,
) -> pd.DataFrame:
    basis = np.asarray(y_basis_log, dtype=float)
    actual = np.asarray(y_actual_log, dtype=float)
    pred = np.asarray(y_pred_log, dtype=float)
    mask = np.isfinite(basis) & np.isfinite(actual) & np.isfinite(pred)
    if mask.sum() < max(20, n_deciles * 5):
        return pd.DataFrame()
    tmp = pd.DataFrame(
        {
            "basis_log": basis[mask],
            "actual_log": actual[mask],
            "pred_log": pred[mask],
        }
    )
    try:
        tmp["decile"] = pd.qcut(tmp["basis_log"], q=int(n_deciles), labels=False, duplicates="drop")
    except Exception:
        return pd.DataFrame()
    tmp["ratio"] = np.exp(np.clip(tmp["pred_log"] - tmp["actual_log"], -50, 50))
    tmp["actual_price"] = np.exp(np.clip(tmp["actual_log"], -50, 50))
    tmp["pred_price"] = np.exp(np.clip(tmp["pred_log"], -50, 50))
    grp = (
        tmp.groupby("decile", observed=True)
        .agg(
            n=("ratio", "size"),
            basis_log_min=("basis_log", "min"),
            basis_log_max=("basis_log", "max"),
            basis_log_median=("basis_log", "median"),
            actual_price_median=("actual_price", "median"),
            pred_price_median=("pred_price", "median"),
            median_ratio=("ratio", "median"),
            mean_ratio=("ratio", "mean"),
            weighted_mean_ratio=("pred_price", "sum"),
            actual_sum=("actual_price", "sum"),
        )
        .reset_index()
    )
    grp["decile"] = grp["decile"].astype(int) + 1
    grp["weighted_mean_ratio"] = grp["weighted_mean_ratio"] / grp["actual_sum"].replace(0.0, np.nan)
    return grp.drop(columns=["actual_sum"])


def curve_shape_stats(curve: pd.DataFrame) -> Dict[str, float]:
    if curve.empty or curve.shape[0] < 3:
        return {"curve_level": np.nan, "curve_trend": np.nan, "curve_shape": np.nan, "curve_max_gap": np.nan}
    z = curve["basis_log_median"].to_numpy(dtype=float)
    c = np.log(np.clip(curve["median_ratio"].to_numpy(dtype=float), 1e-12, np.inf))
    w = curve["n"].to_numpy(dtype=float)
    zc = z - np.average(z, weights=w)
    level = float(np.average(c, weights=w))
    denom = float(np.sum(w * zc * zc))
    if denom <= 0.0:
        trend = np.nan
        shape = np.nan
    else:
        trend = float(np.sum(w * zc * (c - level)) / denom)
        residual = c - level - trend * zc
        shape = float(np.sqrt(np.average(residual ** 2, weights=w)))
    return {
        "curve_level": level,
        "curve_trend": trend,
        "curve_shape": shape,
        "curve_max_gap": float(np.nanmax(c) - np.nanmin(c)),
    }


def evaluate_predictions(
    *,
    y_raw_log: np.ndarray,
    y_reference_log: np.ndarray,
    y_pred_log: np.ndarray,
    y_train_reference_log: np.ndarray,
    n_deciles: int,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    y_raw = np.asarray(y_raw_log, dtype=float)
    y_ref = np.asarray(y_reference_log, dtype=float)
    y_pred = np.asarray(y_pred_log, dtype=float)
    raw_price = np.exp(np.clip(y_raw, -50, 50))
    ref_price = np.exp(np.clip(y_ref, -50, 50))
    pred_price = np.exp(np.clip(y_pred, -50, 50))
    ratio_raw = pred_price / np.maximum(raw_price, 1e-12)
    ratio_ref = pred_price / np.maximum(ref_price, 1e-12)

    out: Dict[str, Any] = {
        "n": int(len(y_raw)),
        "r2_log_raw": float(r2_score(y_raw, y_pred)) if len(y_raw) else np.nan,
        "rmse_log_raw": float(np.sqrt(mean_squared_error(y_raw, y_pred))) if len(y_raw) else np.nan,
        "mae_log_raw": float(mean_absolute_error(y_raw, y_pred)) if len(y_raw) else np.nan,
        "r2_log_reference": float(r2_score(y_ref, y_pred)) if len(y_ref) else np.nan,
        "oos_r2_log_reference": oos_r2_log(y_ref, y_pred, y_train_reference_log),
        "rmse_log_reference": float(np.sqrt(mean_squared_error(y_ref, y_pred))) if len(y_ref) else np.nan,
        "mae_log_reference": float(mean_absolute_error(y_ref, y_pred)) if len(y_ref) else np.nan,
        "r2_price_raw": float(r2_score(raw_price, pred_price)) if len(raw_price) else np.nan,
        "rmse_price_raw": float(np.sqrt(mean_squared_error(raw_price, pred_price))) if len(raw_price) else np.nan,
        "mae_price_raw": float(mean_absolute_error(raw_price, pred_price)) if len(raw_price) else np.nan,
        "r2_price_reference": float(r2_score(ref_price, pred_price)) if len(ref_price) else np.nan,
        "rmse_price_reference": float(np.sqrt(mean_squared_error(ref_price, pred_price))) if len(ref_price) else np.nan,
        "mae_price_reference": float(mean_absolute_error(ref_price, pred_price)) if len(ref_price) else np.nan,
        "median_ratio_raw": float(np.nanmedian(ratio_raw)),
        "mean_ratio_raw": float(np.nanmean(ratio_raw)),
        "weighted_mean_ratio_raw": float(np.nansum(pred_price) / np.nansum(raw_price)) if np.nansum(raw_price) > 0 else np.nan,
        "COD_raw": cod_from_ratio(ratio_raw),
        "PRD_raw": prd_from_values(pred_price, raw_price),
        "PRB_raw": prb_from_values(pred_price, raw_price),
        "VEI_raw": vei_from_values(pred_price, raw_price),
        "median_ratio_reference": float(np.nanmedian(ratio_ref)),
        "mean_ratio_reference": float(np.nanmean(ratio_ref)),
        "weighted_mean_ratio_reference": float(np.nansum(pred_price) / np.nansum(ref_price)) if np.nansum(ref_price) > 0 else np.nan,
        "COD_reference": cod_from_ratio(ratio_ref),
        "PRD_reference": prd_from_values(pred_price, ref_price),
        "PRB_reference": prb_from_values(pred_price, ref_price),
        "VEI_reference": vei_from_values(pred_price, ref_price),
    }
    if len(y_raw) >= 2 and np.nanstd(y_raw) > 0:
        log_ratio_raw = np.log(np.clip(ratio_raw, 1e-12, np.inf))
        lr = LinearRegression().fit(y_raw.reshape(-1, 1), log_ratio_raw.reshape(-1, 1))
        out["log_ratio_slope_vs_raw_log_price"] = float(lr.coef_[0, 0])
        out["corr_log_ratio_vs_raw_log_price"] = float(np.corrcoef(log_ratio_raw, y_raw)[0, 1])
    else:
        out["log_ratio_slope_vs_raw_log_price"] = np.nan
        out["corr_log_ratio_vs_raw_log_price"] = np.nan
    if len(y_ref) >= 2 and np.nanstd(y_ref) > 0:
        log_ratio_ref = np.log(np.clip(ratio_ref, 1e-12, np.inf))
        lr = LinearRegression().fit(y_ref.reshape(-1, 1), log_ratio_ref.reshape(-1, 1))
        out["log_ratio_slope_vs_reference_log_price"] = float(lr.coef_[0, 0])
        out["corr_log_ratio_vs_reference_log_price"] = float(np.corrcoef(log_ratio_ref, y_ref)[0, 1])
    else:
        out["log_ratio_slope_vs_reference_log_price"] = np.nan
        out["corr_log_ratio_vs_reference_log_price"] = np.nan

    raw_curve = decile_curve(y_basis_log=y_raw, y_actual_log=y_raw, y_pred_log=y_pred, n_deciles=n_deciles)
    ref_curve = decile_curve(y_basis_log=y_ref, y_actual_log=y_ref, y_pred_log=y_pred, n_deciles=n_deciles)
    out.update({f"raw_{k}": v for k, v in curve_shape_stats(raw_curve).items()})
    out.update({f"reference_{k}": v for k, v in curve_shape_stats(ref_curve).items()})
    return out, raw_curve, ref_curve


def select_alpha(
    learner: str,
    *,
    train_df: pd.DataFrame,
    y: np.ndarray,
    sample_weight: np.ndarray,
    predictors: Sequence[str],
    categorical_cols: Sequence[str],
    id_vars: Sequence[str],
    preprocess_mode: str,
    alpha_grid: Sequence[float],
    seed: int,
    lasso_max_iter: int,
    inner_val_frac: float,
) -> Tuple[float, Dict[str, Any]]:
    learner = str(learner).lower().strip()
    if learner == "linear":
        return 0.0, {"alpha_selected": 0.0, "alpha_selection_rmse": np.nan}
    grid = [float(a) for a in alpha_grid]
    if not grid:
        raise ValueError(f"Empty alpha grid for learner {learner}")
    n = len(train_df)
    if n < 200 or inner_val_frac <= 0.0:
        return grid[0], {"alpha_selected": grid[0], "alpha_selection_rmse": np.nan}
    cut = int((1.0 - float(inner_val_frac)) * n)
    cut = min(max(cut, 100), n - 20)
    tr = np.arange(0, cut, dtype=int)
    va = np.arange(cut, n, dtype=int)
    if tr.size < 50 or va.size < 20:
        return grid[0], {"alpha_selected": grid[0], "alpha_selection_rmse": np.nan}
    scores = []
    for alpha in grid:
        spec = LinearModelSpec(
            learner=learner,
            predictors=list(predictors),
            categorical_cols=list(categorical_cols),
            id_vars=list(id_vars),
            alpha=float(alpha),
            seed=int(seed),
            lasso_max_iter=int(lasso_max_iter),
            preprocess_mode=str(preprocess_mode),
        )
        pred, _ = fit_predict_model(
            spec,
            train_df.iloc[tr].copy(),
            train_df.iloc[va].copy(),
            np.asarray(y, dtype=float)[tr],
            sample_weight=np.asarray(sample_weight, dtype=float)[tr],
        )
        rmse = float(np.sqrt(mean_squared_error(np.asarray(y, dtype=float)[va], pred)))
        scores.append((rmse, alpha))
    scores.sort(key=lambda x: (x[0], x[1]))
    return float(scores[0][1]), {"alpha_selected": float(scores[0][1]), "alpha_selection_rmse": float(scores[0][0])}


def rank_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    df = metrics.copy()
    df["abs_reference_median_ratio_gap"] = (df["median_ratio_reference"] - 1.0).abs()
    df["abs_reference_prb"] = df["PRB_reference"].abs()
    df["abs_reference_slope"] = df["log_ratio_slope_vs_reference_log_price"].abs()
    df["selection_score"] = (
        df["rmse_log_reference"]
        + 0.50 * df["abs_reference_median_ratio_gap"].fillna(0.0)
        + 0.50 * df["reference_curve_shape"].fillna(0.0)
        + 0.25 * df["abs_reference_slope"].fillna(0.0)
        + 0.10 * df["abs_reference_prb"].fillna(0.0)
    )
    return df.sort_values(["split", "eval_phase", "eval_subset", "selection_score", "rmse_log_reference"]).reset_index(drop=True)


def add_model_id_columns(frame: pd.DataFrame, fields: Dict[str, Any]) -> pd.DataFrame:
    out = frame.copy()
    for k, v in fields.items():
        out[k] = v
    return out


def run_one_split_phase(
    *,
    split_name: str,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    params: dict,
    predictors: List[str],
    categorical_cols: List[str],
    id_vars: List[str],
    preprocess_mode: str,
    target_variants: Sequence[str],
    learners: Sequence[str],
    calibration_modes: Sequence[str],
    eval_phase: str,
    fixed_date_mode: str,
    time_day_origin: Optional[pd.Timestamp],
    seed: int,
    ridge_alpha: float,
    lasso_alpha: float,
    ridge_alphas: Sequence[float],
    lasso_alphas: Sequence[float],
    tune_alphas: bool,
    lasso_max_iter: int,
    inner_val_frac: float,
    calib_frac: float,
    target_shrink_n: float,
    strict_deed_types: Sequence[str],
    deed_weights: Dict[str, float],
    robust_c: float,
    local_K: int,
    local_spatial_bw_miles: float,
    local_time_bw_days: float,
    base_ridge_alpha: float,
    adaptive_alpha_min: float,
    adaptive_alpha_max: float,
    class_filter_col: Optional[str],
    min_same_class_pool: int,
    allow_class_fallback: bool,
    max_neighbor_age_days: Optional[float],
    max_spatial_candidates: int,
    candidate_multiplier: int,
    n_jobs: int,
    n_oof_folds: int,
    oof_mode: str,
    n_deciles: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if train_df.empty or eval_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if fixed_date_mode == "eval_min":
        fixed_date = pd.Timestamp(eval_df[DATE_COL].min()).normalize()
    elif fixed_date_mode == "assessment_date":
        fixed_date = pd.Timestamp(params.get("assessment", {}).get("date", "2025-01-01")).normalize()
    else:
        raise ValueError("--fixed-date-mode must be eval_min or assessment_date")

    core_df, calib_df = split_core_calibration(train_df, calib_frac)
    combined, core_idx, calib_idx, eval_idx = build_combined(core_df, calib_df, eval_df)
    combined = apply_temporal_feature_policy(
        combined,
        predictors=predictors,
        eval_phase=eval_phase,
        fixed_date=fixed_date,
        time_day_origin=time_day_origin,
    )
    for c in categorical_cols:
        if c in combined.columns:
            combined[c] = combined[c].astype("category")
    target_year = target_year_float_for_phase(combined, eval_phase, fixed_date)

    log(
        "split/phase start",
        split=split_name,
        phase=eval_phase,
        core=len(core_idx),
        calib=len(calib_idx),
        eval=len(eval_idx),
        fixed_date=str(fixed_date.date()),
    )

    strict_mask = strict_validation_mask(combined, strict_deed_types).to_numpy(dtype=bool)
    reliability = reliability_weights(combined, deed_weights)
    global_time_adjuster = fit_time_adjuster(
        "time_global",
        combined.iloc[core_idx].copy(),
        predictor_cols=predictors,
        categorical_cols=categorical_cols,
        seed=seed,
        shrink_n=target_shrink_n,
    )
    y_reference_all = global_time_adjuster.transform(combined, target_year)
    if eval_phase == "actual_sale_date":
        y_reference_all = combined["y_raw_log"].to_numpy(dtype=float)

    needed_time_variants = {"time_global"}
    for variant in target_variants:
        v = normalize_time_variant(variant)
        if v.startswith("time_"):
            needed_time_variants.add(v)
    time_adjusters = {}
    for v in sorted(needed_time_variants):
        time_adjusters[v] = fit_time_adjuster(
            v,
            combined.iloc[core_idx].copy(),
            predictor_cols=predictors,
            categorical_cols=categorical_cols,
            seed=seed,
            shrink_n=target_shrink_n,
        )

    needs_robust = any("robust" in str(v).lower() for v in target_variants)
    needs_local = any("local_shrink" in str(v).lower() for v in target_variants)

    # Robust target variants use a common ridge baseline fit on the
    # phase-reference labels. This is fit only when a requested target needs it.
    robust_pred_all = np.full(len(combined), np.nan, dtype=float)
    if needs_robust:
        robust_spec = LinearModelSpec(
            learner="ridge",
            predictors=list(predictors),
            categorical_cols=list(categorical_cols),
            id_vars=list(id_vars),
            alpha=float(base_ridge_alpha),
            seed=int(seed),
            preprocess_mode=str(preprocess_mode),
        )
        robust_oof = temporal_oof_predictions(
            robust_spec,
            combined.iloc[core_idx].copy(),
            y_reference_all[core_idx],
            sample_weight=None,
            n_folds=int(n_oof_folds),
            mode=oof_mode,
        )
        robust_pred_all[core_idx] = robust_oof
        robust_model = fit_model(robust_spec, combined.iloc[core_idx].copy(), y_reference_all[core_idx])
        if len(calib_idx):
            robust_pred_all[calib_idx] = predict_model(robust_model, combined.iloc[calib_idx].copy(), robust_spec)
        robust_pred_all[eval_idx] = predict_model(robust_model, combined.iloc[eval_idx].copy(), robust_spec)
        if np.any(~np.isfinite(robust_pred_all)):
            robust_pred_all[~np.isfinite(robust_pred_all)] = float(np.nanmedian(y_reference_all[core_idx]))

    local_prior = None
    local_diag: Dict[str, Any] = {}
    if needs_local:
        y_time_global = global_time_adjuster.transform(combined, target_year)
        local_prior, local_diag = compute_local_prior(
            combined,
            core_idx=core_idx,
            calib_idx=calib_idx,
            eval_idx=eval_idx,
            target_year_float=target_year,
            y_time_all=y_time_global,
            predictors=predictors,
            categorical_cols=categorical_cols,
            id_vars=id_vars,
            preprocess_mode=preprocess_mode,
            seed=seed,
            base_ridge_alpha=base_ridge_alpha,
            n_oof_folds=n_oof_folds,
            oof_mode=oof_mode,
            K=local_K,
            spatial_bw_miles=local_spatial_bw_miles,
            time_bw_days=local_time_bw_days,
            class_filter_col=class_filter_col,
            min_same_class_pool=min_same_class_pool,
            allow_class_fallback=allow_class_fallback,
            max_neighbor_age_days=max_neighbor_age_days,
            max_spatial_candidates=max_spatial_candidates,
            candidate_multiplier=candidate_multiplier,
            n_jobs=n_jobs,
        )

    metrics_rows: List[Dict[str, Any]] = []
    decile_rows: List[pd.DataFrame] = []
    target_diag_rows: List[Dict[str, Any]] = []

    eval_subsets = {
        "all_repo_valid": np.ones(len(combined), dtype=bool),
        "strict_valid": strict_mask,
    }

    for target_variant in target_variants:
        prepared = prepare_target_variant(
            target_variant,
            combined=combined,
            core_idx=core_idx,
            target_year_float=target_year,
            strict_mask=strict_mask,
            reliability=reliability,
            global_time_adjuster=global_time_adjuster,
            time_adjusters=time_adjusters,
            local_prior=local_prior,
            local_diag=local_diag,
            robust_weight_base_pred=robust_pred_all,
            robust_c=robust_c,
            adaptive_alpha_min=adaptive_alpha_min,
            adaptive_alpha_max=adaptive_alpha_max,
        )
        target_diag_rows.append(
            {
                "split": split_name,
                "eval_phase": eval_phase,
                "target_variant": target_variant,
                "target_family": prepared.family,
                "core_rows": int(len(core_idx)),
                "calib_rows": int(len(calib_idx)),
                "eval_rows": int(len(eval_idx)),
                "strict_share_core": float(strict_mask[core_idx].mean()) if len(core_idx) else np.nan,
                "reliability_mean_core": float(np.nanmean(reliability[core_idx])),
                "reference_beta_global": float(global_time_adjuster.beta_global),
                "fixed_date": str(fixed_date.date()),
                **prepared.diagnostics,
            }
        )

        train_idx = core_idx[prepared.train_mask_all[core_idx] & np.isfinite(prepared.y_all[core_idx])]
        if len(train_idx) < 100:
            log("target skipped: too few training rows", target=target_variant, train_rows=len(train_idx))
            continue
        calib_valid = calib_idx[np.isfinite(y_reference_all[calib_idx])] if len(calib_idx) else np.array([], dtype=int)

        for learner in learners:
            learner = str(learner).lower().strip()
            default_alpha = 0.0 if learner == "linear" else float(ridge_alpha if learner == "ridge" else lasso_alpha)
            if tune_alphas and learner in {"ridge", "lasso"}:
                grid = ridge_alphas if learner == "ridge" else lasso_alphas
                alpha, alpha_diag = select_alpha(
                    learner,
                    train_df=combined.iloc[train_idx].copy(),
                    y=prepared.y_all[train_idx],
                    sample_weight=prepared.sample_weight_all[train_idx],
                    predictors=predictors,
                    categorical_cols=categorical_cols,
                    id_vars=id_vars,
                    preprocess_mode=preprocess_mode,
                    alpha_grid=grid,
                    seed=seed,
                    lasso_max_iter=lasso_max_iter,
                    inner_val_frac=inner_val_frac,
                )
            else:
                alpha, alpha_diag = default_alpha, {"alpha_selected": default_alpha, "alpha_selection_rmse": np.nan}

            spec = LinearModelSpec(
                learner=learner,
                predictors=list(predictors),
                categorical_cols=list(categorical_cols),
                id_vars=list(id_vars),
                alpha=float(alpha),
                seed=int(seed),
                lasso_max_iter=int(lasso_max_iter),
                preprocess_mode=str(preprocess_mode),
            )
            log("model fit", split=split_name, phase=eval_phase, target=target_variant, learner=learner, alpha=f"{alpha:.4g}", n=len(train_idx))
            pred_eval, model = fit_predict_model(
                spec,
                combined.iloc[train_idx].copy(),
                combined.iloc[eval_idx].copy(),
                prepared.y_all[train_idx],
                sample_weight=prepared.sample_weight_all[train_idx],
            )
            pred_calib = predict_model(model, combined.iloc[calib_valid].copy(), spec) if len(calib_valid) else np.array([])

            for calib_mode in calibration_modes:
                cal = fit_calibrator(calib_mode, pred_calib, y_reference_all[calib_valid])
                pred_eval_cal = cal.predict(pred_eval)
                for subset_name, subset_mask_all in eval_subsets.items():
                    keep_eval_pos = subset_mask_all[eval_idx]
                    if keep_eval_pos.sum() < 20:
                        continue
                    eval_rows = eval_idx[keep_eval_pos]
                    pred_subset = pred_eval_cal[keep_eval_pos]
                    metrics, raw_curve, ref_curve = evaluate_predictions(
                        y_raw_log=combined.loc[eval_rows, "y_raw_log"].to_numpy(dtype=float),
                        y_reference_log=y_reference_all[eval_rows],
                        y_pred_log=pred_subset,
                        y_train_reference_log=y_reference_all[train_idx],
                        n_deciles=n_deciles,
                    )
                    common = {
                        "split": split_name,
                        "eval_phase": eval_phase,
                        "fixed_date": str(fixed_date.date()),
                        "eval_subset": subset_name,
                        "target_variant": target_variant,
                        "target_family": prepared.family,
                        "learner": learner,
                        "learner_alpha": float(alpha),
                        "calibration_mode": cal.mode,
                        "cal_intercept": float(cal.intercept),
                        "cal_slope": float(cal.slope),
                        "train_rows": int(len(train_idx)),
                        "calib_rows": int(len(calib_valid)),
                        "eval_rows_total": int(len(eval_idx)),
                        "reference_kind": "fixed_date_time_adjusted_sale" if eval_phase == "fixed_assessment_date" else "actual_sale_price",
                        **alpha_diag,
                        **{k: v for k, v in prepared.diagnostics.items() if k not in metrics},
                    }
                    metrics_rows.append({**common, **metrics})
                    for basis_name, curve in [("raw_sale_price", raw_curve), ("reference_price", ref_curve)]:
                        if curve.empty:
                            continue
                        fields = {
                            "split": split_name,
                            "eval_phase": eval_phase,
                            "fixed_date": str(fixed_date.date()),
                            "eval_subset": subset_name,
                            "target_variant": target_variant,
                            "target_family": prepared.family,
                            "learner": learner,
                            "learner_alpha": float(alpha),
                            "calibration_mode": cal.mode,
                            "decile_basis": basis_name,
                        }
                        decile_rows.append(add_model_id_columns(curve, fields))

    metrics_df = pd.DataFrame(metrics_rows)
    deciles_df = pd.concat(decile_rows, axis=0, ignore_index=True) if decile_rows else pd.DataFrame()
    target_diag_df = pd.DataFrame(target_diag_rows)
    return metrics_df, deciles_df, target_diag_df


def save_summary_tables(metrics: pd.DataFrame, out_dir: Path) -> None:
    if metrics.empty:
        return
    ranked = rank_metrics(metrics)
    ranked.to_csv(out_dir / "metrics_ranked.csv", index=False)
    top_cols = [
        "split",
        "eval_phase",
        "eval_subset",
        "target_variant",
        "target_family",
        "learner",
        "calibration_mode",
        "learner_alpha",
        "target_alpha",
        "rmse_log_reference",
        "mae_log_reference",
        "r2_log_reference",
        "median_ratio_reference",
        "COD_reference",
        "PRD_reference",
        "PRB_reference",
        "VEI_reference",
        "log_ratio_slope_vs_reference_log_price",
        "reference_curve_shape",
        "rmse_log_raw",
        "median_ratio_raw",
        "COD_raw",
        "PRD_raw",
        "PRB_raw",
        "selection_score",
        "train_rows",
        "n",
    ]
    cols = [c for c in top_cols if c in ranked.columns]
    ranked[cols].head(200).to_csv(out_dir / "top200_selection.csv", index=False)

    group_cols = ["split", "eval_phase", "eval_subset", "target_variant", "target_family", "learner"]
    best = (
        ranked.sort_values("selection_score")
        .groupby(group_cols, observed=True, as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    best[cols].to_csv(out_dir / "best_by_target_learner.csv", index=False)

    pivot_metric = "rmse_log_reference"
    pivot = best.pivot_table(
        index=["split", "eval_phase", "eval_subset", "target_variant"],
        columns="learner",
        values=pivot_metric,
        aggfunc="min",
    )
    pivot.to_csv(out_dir / f"pivot_{pivot_metric}.csv")


def plot_decile_curves(deciles: pd.DataFrame, metrics: pd.DataFrame, out_dir: Path, *, max_targets: int) -> None:
    if deciles.empty:
        return
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    ranked = rank_metrics(metrics)
    if not ranked.empty and "selection_score" in ranked.columns:
        keep_targets = (
            ranked.loc[ranked["eval_subset"].eq("all_repo_valid"), ["target_variant", "selection_score"]]
            .groupby("target_variant", observed=True)["selection_score"]
            .min()
            .sort_values()
            .head(int(max_targets))
            .index.tolist()
        )
    else:
        keep_targets = list(pd.unique(deciles["target_variant"]))[: int(max_targets)]

    dd = deciles.loc[
        deciles["target_variant"].isin(keep_targets)
        & deciles["eval_subset"].eq("all_repo_valid")
        & deciles["decile_basis"].eq("reference_price")
    ].copy()
    if dd.empty:
        return
    for (split, phase), sub0 in dd.groupby(["split", "eval_phase"], observed=True):
        learners = list(pd.unique(sub0["learner"]))
        if not learners:
            continue
        fig, axes = plt.subplots(1, len(learners), figsize=(6.0 * len(learners), 4.6), sharey=True)
        if len(learners) == 1:
            axes = [axes]
        legend_handles: Dict[str, Any] = {}
        for ax, learner in zip(axes, learners):
            sub = sub0.loc[sub0["learner"].eq(learner)].copy()
            for (target, calib), line in sub.groupby(["target_variant", "calibration_mode"], observed=True):
                # Plot only each target's best calibration mode by reference RMSE.
                m = metrics.loc[
                    metrics["split"].eq(split)
                    & metrics["eval_phase"].eq(phase)
                    & metrics["eval_subset"].eq("all_repo_valid")
                    & metrics["target_variant"].eq(target)
                    & metrics["learner"].eq(learner)
                    & metrics["calibration_mode"].eq(calib)
                ]
                if m.empty:
                    continue
                best_cal = (
                    metrics.loc[
                        metrics["split"].eq(split)
                        & metrics["eval_phase"].eq(phase)
                        & metrics["eval_subset"].eq("all_repo_valid")
                        & metrics["target_variant"].eq(target)
                        & metrics["learner"].eq(learner)
                    ]
                    .sort_values("rmse_log_reference")
                    .head(1)["calibration_mode"]
                    .iloc[0]
                )
                if calib != best_cal:
                    continue
                line = line.sort_values("decile")
                label = str(target) if str(calib) == "none" else f"{target} ({calib})"
                handle, = ax.plot(line["decile"], line["median_ratio"], marker="o", linewidth=1.6, label=label)
                legend_handles.setdefault(label, handle)
            ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
            ax.set_title(str(learner))
            ax.set_xlabel("Reference price decile")
            ax.set_xticks(range(1, int(deciles["decile"].max()) + 1))
            ax.grid(True, alpha=0.25)
        axes[0].set_ylabel("Median predicted / reference sale proxy")
        if legend_handles:
            fig.legend(
                list(legend_handles.values()),
                list(legend_handles.keys()),
                title="Target correction",
                loc="center left",
                bbox_to_anchor=(0.86, 0.5),
                bbox_transform=fig.transFigure,
                fontsize=8,
                title_fontsize=9,
            )
        fig.text(0.01, 0.01, "Each line uses that target/model's best calibration mode by reference RMSE.", fontsize=8)
        fig.suptitle(f"Decile Ratio Curve: {split} / {phase}")
        fig.tight_layout(rect=[0, 0.03, 0.82, 0.94])
        fig.savefig(plot_dir / f"decile_ratio_curve_{split}_{phase}.png", dpi=180, bbox_inches="tight")
        fig.savefig(plot_dir / f"decile_ratio_curve_{split}_{phase}.pdf", bbox_inches="tight")
        plt.close(fig)


def write_analysis_notes(out_dir: Path, config: Dict[str, Any]) -> None:
    text = f"""# Target Correction Analysis

This output was generated by `final_market_value_1_target_correction.py`.

Core design:
- chronological train/test and train/assessment splits;
- target corrections fit only on the training core;
- calibration uses the trailing training slice, not the eval rows;
- local shrinkage targets use prior core comparable sales only;
- `time_sale_*` features are overwritten according to the evaluation phase;
- sale-validation metadata is used for target filters/weights only, not as X.

Evaluation phases:
- `fixed_assessment_date`: every row is represented as of one fixed date for the split (`eval_min` by default).
- `actual_sale_date`: every row uses its actual sale date, including eval rows, as an optimistic diagnostic.

Primary metric ranking uses `rmse_log_reference` plus penalties for median-ratio level, decile-curve shape, and vertical slope. `raw_*` metrics remain in the tables so target corrections can be checked against unadjusted sale prices.

Key files:
- `metrics_all.csv`: all model/target/phase/calibration metrics.
- `metrics_ranked.csv`: ranked version of the metrics table.
- `best_by_target_learner.csv`: best calibration mode per target/model.
- `decile_curves.csv`: long 10-decile ratio curves.
- `target_diagnostics.csv`: fitted target-correction diagnostics.
- `plots/decile_ratio_curve_*.png`: model comparison ratio curves.

Run config:

```json
{json.dumps(config, indent=2, default=str)}
```
"""
    (out_dir / "analysis_notes.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Linear-model target-correction experiment runner")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--params-path", required=True)
    parser.add_argument("--out-dir", default="output/final_mv_target_correction/manual")
    parser.add_argument("--parquet-engine", default="pyarrow")
    parser.add_argument("--sample-frac", type=float, default=None)
    parser.add_argument("--sample-seed", type=int, default=2025)
    parser.add_argument("--assess-eval-year", type=int, default=2024)
    parser.add_argument("--fixed-date-mode", default="eval_min", choices=["eval_min", "assessment_date"])
    parser.add_argument("--eval-phases", default="fixed_assessment_date,actual_sale_date")
    parser.add_argument("--target-variants", default=",".join(DEFAULT_TARGET_VARIANTS))
    parser.add_argument("--learners", default="linear,ridge,lasso")
    parser.add_argument(
        "--preprocess-mode",
        default="repo",
        choices=["repo", "simple"],
        help="repo matches quick_test_models.py preprocessing; simple is a sparse OHE fallback.",
    )
    parser.add_argument("--calibration-modes", default="none,median_center,affine")
    parser.add_argument("--calib-frac", type=float, default=0.20)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--lasso-alpha", type=float, default=0.001)
    parser.add_argument("--ridge-alphas", default="0.1,1,10,100")
    parser.add_argument("--lasso-alphas", default="0.0003,0.001,0.003,0.01")
    parser.add_argument("--no-tune-alphas", action="store_true")
    parser.add_argument("--inner-val-frac", type=float, default=0.20)
    parser.add_argument("--lasso-max-iter", type=int, default=5000)
    parser.add_argument("--target-shrink-n", type=float, default=100.0)
    parser.add_argument("--strict-deed-types", default="01")
    parser.add_argument("--soft-deed-weights", default="01:1.0,02:0.85,05:0.75")
    parser.add_argument("--robust-c", type=float, default=1.5)
    parser.add_argument("--local-K", type=int, default=20)
    parser.add_argument("--local-spatial-bw-miles", type=float, default=1.0)
    parser.add_argument("--local-time-bw-days", type=float, default=500.0)
    parser.add_argument("--base-ridge-alpha", type=float, default=10.0)
    parser.add_argument("--adaptive-alpha-min", type=float, default=0.10)
    parser.add_argument("--adaptive-alpha-max", type=float, default=0.98)
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
    parser.add_argument("--plot-top-targets", type=int, default=10)
    parser.add_argument("--no-engineered-features", action="store_true")
    parser.add_argument("--no-strict-feature-screen", action="store_true")
    parser.add_argument("--allow-sale-count-feature", action="store_true")
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

    if not args.allow_sale_count_feature:
        predictors = [c for c in predictors if c not in SALE_METADATA_FOR_TARGETING]
        cats = [c for c in cats if c in predictors]
    predictors = [c for c in predictors if c in df.columns]
    cats = [c for c in cats if c in predictors]
    id_vars = [c for c in id_vars if c in df.columns]

    class_filter_col = None if args.no_class_filter else args.class_filter_col
    if class_filter_col is not None and class_filter_col not in df.columns:
        log("requested class filter missing; disabling", class_filter_col=class_filter_col)
        class_filter_col = None

    splits = make_quick_splits(df, params, assess_eval_year=int(args.assess_eval_year))
    target_variants = parse_csv(args.target_variants, str)
    learners = parse_csv(args.learners, str)
    eval_phases = parse_csv(args.eval_phases, str)
    calibration_modes = parse_csv(args.calibration_modes, str)
    ridge_alphas = parse_float_csv(args.ridge_alphas)
    lasso_alphas = parse_float_csv(args.lasso_alphas)
    strict_deed_types = parse_csv(args.strict_deed_types, str)
    deed_weights = parse_deed_weights(args.soft_deed_weights)
    time_day_origin = infer_time_sale_day_origin(df)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update(
        {
            "assessment_date_from_params": assessment_date,
            "assessment_year_from_params": assessment_year,
            "predictor_count": len(predictors),
            "categorical_count": len(cats),
            "id_var_count": len(id_vars),
            "time_sale_day_origin": str(time_day_origin.date()) if time_day_origin is not None else None,
        }
    )
    with open(out_dir / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=str)
    pd.Series(predictors, name="predictor").to_csv(out_dir / "predictors_used.csv", index=False)
    pd.Series(cats, name="categorical").to_csv(out_dir / "categoricals_used.csv", index=False)
    pd.Series(id_vars, name="id_var").to_csv(out_dir / "id_vars_used.csv", index=False)

    all_metrics = []
    all_deciles = []
    all_target_diag = []
    for split_name, (train_df, eval_df) in splits.items():
        if train_df.empty or eval_df.empty:
            log("empty split skipped", split=split_name, train=len(train_df), eval=len(eval_df))
            continue
        for eval_phase in eval_phases:
            metrics, deciles, target_diag = run_one_split_phase(
                split_name=split_name,
                train_df=train_df,
                eval_df=eval_df,
                params=params,
                predictors=predictors,
                categorical_cols=cats,
                id_vars=id_vars,
                preprocess_mode=args.preprocess_mode,
                target_variants=target_variants,
                learners=learners,
                calibration_modes=calibration_modes,
                eval_phase=eval_phase,
                fixed_date_mode=args.fixed_date_mode,
                time_day_origin=time_day_origin,
                seed=int(args.seed),
                ridge_alpha=float(args.ridge_alpha),
                lasso_alpha=float(args.lasso_alpha),
                ridge_alphas=ridge_alphas,
                lasso_alphas=lasso_alphas,
                tune_alphas=not args.no_tune_alphas,
                lasso_max_iter=int(args.lasso_max_iter),
                inner_val_frac=float(args.inner_val_frac),
                calib_frac=float(args.calib_frac),
                target_shrink_n=float(args.target_shrink_n),
                strict_deed_types=strict_deed_types,
                deed_weights=deed_weights,
                robust_c=float(args.robust_c),
                local_K=int(args.local_K),
                local_spatial_bw_miles=float(args.local_spatial_bw_miles),
                local_time_bw_days=float(args.local_time_bw_days),
                base_ridge_alpha=float(args.base_ridge_alpha),
                adaptive_alpha_min=float(args.adaptive_alpha_min),
                adaptive_alpha_max=float(args.adaptive_alpha_max),
                class_filter_col=class_filter_col,
                min_same_class_pool=int(args.min_same_class_pool),
                allow_class_fallback=not args.no_class_fallback,
                max_neighbor_age_days=args.max_neighbor_age_days,
                max_spatial_candidates=int(args.max_spatial_candidates),
                candidate_multiplier=int(args.candidate_multiplier),
                n_jobs=int(args.n_jobs),
                n_oof_folds=int(args.n_oof_folds),
                oof_mode=args.oof_mode,
                n_deciles=int(args.n_deciles),
            )
            if not metrics.empty:
                metrics.to_csv(out_dir / f"metrics_{split_name}_{eval_phase}.csv", index=False)
                all_metrics.append(metrics)
            if not deciles.empty:
                deciles.to_csv(out_dir / f"decile_curves_{split_name}_{eval_phase}.csv", index=False)
                all_deciles.append(deciles)
            if not target_diag.empty:
                target_diag.to_csv(out_dir / f"target_diagnostics_{split_name}_{eval_phase}.csv", index=False)
                all_target_diag.append(target_diag)

    metrics_all = pd.concat(all_metrics, axis=0, ignore_index=True) if all_metrics else pd.DataFrame()
    deciles_all = pd.concat(all_deciles, axis=0, ignore_index=True) if all_deciles else pd.DataFrame()
    target_diag_all = pd.concat(all_target_diag, axis=0, ignore_index=True) if all_target_diag else pd.DataFrame()
    metrics_all.to_csv(out_dir / "metrics_all.csv", index=False)
    deciles_all.to_csv(out_dir / "decile_curves.csv", index=False)
    target_diag_all.to_csv(out_dir / "target_diagnostics.csv", index=False)
    save_summary_tables(metrics_all, out_dir)
    plot_decile_curves(deciles_all, metrics_all, out_dir, max_targets=int(args.plot_top_targets))
    write_analysis_notes(out_dir, config)
    log("finished", out_dir=str(out_dir), metrics_rows=len(metrics_all), decile_rows=len(deciles_all))


if __name__ == "__main__":
    main()
