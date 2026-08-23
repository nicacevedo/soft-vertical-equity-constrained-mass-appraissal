"""Polished neighbor-feature experiment runner.

This script keeps the experiment lightweight and notebook-friendly:
  1. optional sample construction from the CCAO parquet,
  2. chronological train/test split,
  3. train-only preprocessing with the existing CCAO pipeline,
  4. optional Option-A kernel neighbor target features,
  5. train and OOS metric tables for each experiment.

The neighbor experiments are parallelized with ThreadPoolExecutor. This keeps the
same preprocessed train/test data shared in memory; it does not spawn separate
processes or copy the full dataset to workers.

Expected project imports:
  - preprocessing.recipes_pipelined.build_model_pipeline
  - preprocessing/spatiotemporal_neighbors.SpatioTemporalKernelTargetNeighbors

Typical use, if you already created `datasets`, `params`, `predictor_cols`, and
`categorical_cols` in the notebook:

    results_df, train_df, oos_df, overview_train, overview_oos = run_all_experiments(
        datasets=datasets,
        params=params,
        predictor_cols=predictor_cols,
        categorical_cols=categorical_cols,
        train_frac=0.80,
        n_jobs=8,
        save_dir="./output/neighbor_experiments",
    )
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Optional
import argparse
import importlib.util
import inspect
import json
import os
import time

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover - LGBM experiments are skipped if unavailable.
    lgb = None
try:
    from soft_constrained_models.boosting_models import LGBCovPenalty
except ImportError:  # pragma: no cover - the cov model is skipped if unavailable.
    LGBCovPenalty = None

from preprocessing.recipes_pipelined import build_model_pipeline
from utils.motivation_utils import compute_taxation_metrics, _build_time_block_bootstrap_indices


LOG_T0 = time.perf_counter()


def log(message: str, **fields: Any) -> None:
    dt = time.perf_counter() - LOG_T0
    fields = {"node": os.uname().nodename, **fields}
    suffix = " | " + " | ".join(f"{k}={v}" for k, v in fields.items())
    print(f"[spatial_analysis +{dt:8.1f}s] {message}{suffix}", flush=True)


# ---------------------------------------------------------------------
# Optional data construction helpers
# ---------------------------------------------------------------------
DEFAULT_EXTRA_COLS = [
    "meta_class",
    "meta_triad_name",
    "meta_sale_deed_type",
    "sv_review_json",
]

SINGLE_FAMILY_META_CLASSES = {
    "202", "203", "204", "205", "206", "207", "208", "209",
    "210", "234", "278", "295",
}


def load_params(path: str | Path = "params.yaml") -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def safe_parse_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if pd.isna(value) or value in {"None", None}:
        return {}
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {}
    return {}


def load_ccao_sales_data(
    *,
    data_path: str | Path,
    params: dict[str, Any],
    target_col: str = "meta_sale_price",
    date_col: str = "meta_sale_date",
    sample_size: Optional[int] = None,
    random_state: int = 42,
    parquet_engine: str = "pyarrow",
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load only needed columns and apply the common sale-validity filters."""
    predictor_cols = list(params["model"]["predictor"]["all"])
    categorical_cols = list(params["model"]["predictor"]["categorical"])
    filter_cols = ["ind_pin_is_multicard", "sv_is_outlier"]

    cols_to_load = list(dict.fromkeys(
        predictor_cols + [target_col, date_col] + filter_cols + DEFAULT_EXTRA_COLS
    ))
    df = pd.read_parquet(data_path, engine=parquet_engine, columns=cols_to_load)

    df = df[(df["ind_pin_is_multicard"] != True) & (df["sv_is_outlier"] != True)].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[date_col, target_col])
    df = df[df[target_col] > 0].copy()

    # The spatiotemporal neighbor transformer requires complete coordinates: a tiny
    # number of rows (<=5 in ~365k) have missing lat/lon and otherwise raise and
    # poison every neighbor task for whichever sampled dataset happens to include
    # them. Drop them uniformly so all datasets are clean and comparable.
    geo_cols = [c for c in ("loc_latitude", "loc_longitude") if c in df.columns]
    if geo_cols:
        df = df.dropna(subset=geo_cols)

    if sample_size is not None and len(df) > int(sample_size):
        df = df.sample(int(sample_size), random_state=int(random_state)).copy()

    return df.sort_values(date_col).reset_index(drop=True), predictor_cols, categorical_cols


def build_modeling_datasets(
    df: pd.DataFrame,
    *,
    target_col: str = "meta_sale_price",
    date_col: str = "meta_sale_date",
) -> dict[str, pd.DataFrame]:
    """Create the same filtered samples as the notebook, without EDA plotting."""
    data = df.copy()

    if "sv_review_json" in data.columns:
        data["is_arms_length"] = data["sv_review_json"].apply(
            lambda x: safe_parse_json(x).get("is_arms_length")
        )
    else:
        data["is_arms_length"] = np.nan

    if "meta_sale_count_past_n_years" in data.columns:
        data["past_sale_indicator"] = (
            pd.to_numeric(data["meta_sale_count_past_n_years"], errors="coerce").fillna(0) > 0
        ).astype(int)

    data["log_price"] = np.log(pd.to_numeric(data[target_col], errors="coerce"))

    arms = data.loc[data["is_arms_length"] != False].copy()
    deed_flag = arms.get("meta_sale_deed_type", pd.Series(index=arms.index)).isin(["01", "02"])
    deed = arms.loc[deed_flag].copy()

    if "meta_class" in deed.columns:
        is_sf = deed["meta_class"].astype(str).isin(SINGLE_FAMILY_META_CLASSES)
        single_family = deed.loc[is_sf].copy()
    else:
        single_family = deed.copy()

    return {
        "all_filtered": data,
        "arms_length_or_missing": arms,
        "deed_01_02": deed,
        "single_family": single_family,
    }


# Filter labels are now applied to the TRAINING set only; the OOS test set is the
# common most-recent slice of the full universe, identical across all experiments.
FILTER_LABELS = ["all_filtered", "arms_length_or_missing", "deed_01_02", "single_family"]


def add_filter_columns(
    df: pd.DataFrame,
    *,
    target_col: str = "meta_sale_price",
) -> pd.DataFrame:
    """Attach the derived columns needed by the train-only filters, without subsetting."""
    data = df.copy()
    if "sv_review_json" in data.columns:
        data["is_arms_length"] = data["sv_review_json"].apply(
            lambda x: safe_parse_json(x).get("is_arms_length")
        )
    else:
        data["is_arms_length"] = np.nan
    if "meta_sale_count_past_n_years" in data.columns:
        data["past_sale_indicator"] = (
            pd.to_numeric(data["meta_sale_count_past_n_years"], errors="coerce").fillna(0) > 0
        ).astype(int)
    data["log_price"] = np.log(pd.to_numeric(data[target_col], errors="coerce"))
    return data


def train_filter_mask(df: pd.DataFrame, label: str) -> pd.Series:
    """Boolean mask selecting the TRAIN rows that belong to ``label`` (cumulative)."""
    if label == "all_filtered":
        return pd.Series(True, index=df.index)

    arms = df["is_arms_length"] if "is_arms_length" in df.columns else pd.Series(np.nan, index=df.index)
    arms_mask = arms != False  # arms-length True or missing/unknown
    if label == "arms_length_or_missing":
        return arms_mask

    deed = df["meta_sale_deed_type"] if "meta_sale_deed_type" in df.columns else pd.Series(index=df.index, dtype=object)
    deed_mask = arms_mask & deed.isin(["01", "02"])
    if label == "deed_01_02":
        return deed_mask

    if label == "single_family":
        if "meta_class" in df.columns:
            sf = df["meta_class"].astype(str).isin(SINGLE_FAMILY_META_CLASSES)
        else:
            sf = pd.Series(False, index=df.index)
        return deed_mask & sf

    raise ValueError(f"Unknown filter label '{label}'. Available: {FILTER_LABELS}")


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
# Full accuracy + vertical-equity/fairness metric set, computed on the log scale
# via the shared repo implementation (utils.motivation_utils.compute_taxation_metrics).
# Keys are reused verbatim so downstream tables/plots stay consistent with the rest
# of the project.
METRIC_KEYS = [
    "R2", "OOS R2", "R2 (log)", "RMSE", "MAE", "MAPE", "MdAPE",
    "Corr(r,price)", "Corr(r,logprice)", "Slope(r~logy)",
    "Std ratio", "Median ratio", "Mean ratio", "W. Mean ratio",
    "COD", "COV_IAAO", "VEI", "PRD", "PRB", "MKI",
]
RATIO_DECILE_KEYS = [f"MedianRatio_q10_bin{i}" for i in range(1, 11)]
METRIC_KEYS.extend(RATIO_DECILE_KEYS)


def compute_ratio_decile_metrics(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    *,
    n_deciles: int = 10,
) -> dict[str, float]:
    """Median predicted/actual ratio within actual-price deciles."""
    keys = [f"MedianRatio_q{int(n_deciles)}_bin{i}" for i in range(1, int(n_deciles) + 1)]
    out = {k: np.nan for k in keys}
    if y_true_log.size < int(n_deciles):
        return out

    tmp = pd.DataFrame({
        "actual_log": y_true_log,
        "ratio": np.exp(np.clip(y_pred_log - y_true_log, -50.0, 50.0)),
    })
    try:
        tmp["decile"] = pd.qcut(
            tmp["actual_log"],
            q=int(n_deciles),
            labels=False,
            duplicates="drop",
        )
    except Exception:
        return out

    for decile, value in tmp.groupby("decile", observed=True)["ratio"].median().items():
        if pd.notna(decile):
            out[f"MedianRatio_q{int(n_deciles)}_bin{int(decile) + 1}"] = float(value)
    return out


def compute_metrics(
    y_true_log: Any,
    y_pred_log: Any,
    y_train_log: Any,
) -> dict[str, float]:
    """Accuracy + ratio-study/vertical-equity diagnostics on log-price predictions.

    Inputs are log(price). Predictions and targets are exponentiated inside
    `compute_taxation_metrics`. Non-finite (true, pred) pairs are dropped so a few
    overflowing predictions cannot crash the whole metric table.
    """
    y_true_log = np.asarray(pd.Series(y_true_log).astype(float), dtype=float)
    y_pred_log = np.asarray(pd.Series(y_pred_log).astype(float), dtype=float)
    y_train_log = np.asarray(pd.Series(y_train_log).astype(float), dtype=float)

    valid = np.isfinite(y_true_log) & np.isfinite(y_pred_log)
    y_true_log = y_true_log[valid]
    y_pred_log = y_pred_log[valid]
    if y_true_log.size == 0:
        return {**{m: np.nan for m in METRIC_KEYS}, "N": 0}

    try:
        metrics = compute_taxation_metrics(
            y_true_log, y_pred_log, scale="log", y_train=y_train_log
        )
    except Exception:
        metrics = {m: np.nan for m in METRIC_KEYS}
    metrics = {k: metrics.get(k, np.nan) for k in METRIC_KEYS}
    metrics.update(compute_ratio_decile_metrics(y_true_log, y_pred_log))
    metrics["N"] = int(y_true_log.size)
    return metrics


def compute_oos_metrics_boot(
    y_true_log: Any,
    y_pred_log: Any,
    y_train_log: Any,
    boot_indices: Optional[list[np.ndarray]],
) -> dict[str, float]:
    """OOS metrics averaged over month-block bootstrap resamples of the test set.

    ``boot_indices`` are positional indices into the (fixed-order) test arrays and
    are shared across every experiment, so metrics are comparable on the same test
    subsamples. Returns the per-metric mean plus a ``<metric>_std`` column; falls
    back to a single point estimate when no bootstrap is requested.
    """
    if not boot_indices:
        return compute_metrics(y_true_log, y_pred_log, y_train_log)

    y_true_log = np.asarray(pd.Series(y_true_log).astype(float), dtype=float)
    y_pred_log = np.asarray(pd.Series(y_pred_log).astype(float), dtype=float)
    per_sample = [
        compute_metrics(y_true_log[idx], y_pred_log[idx], y_train_log)
        for idx in boot_indices
    ]
    table = pd.DataFrame(per_sample)
    out: dict[str, float] = {}
    for m in METRIC_KEYS:
        col = pd.to_numeric(table[m], errors="coerce")
        out[m] = float(col.mean())
        out[f"{m}_std"] = float(col.std(ddof=1)) if len(col) > 1 else 0.0
    out["N"] = float(pd.to_numeric(table["N"], errors="coerce").mean())
    out["n_boot"] = int(len(boot_indices))
    return out


# ---------------------------------------------------------------------
# Experiment specification
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class NeighborExperiment:
    name: str
    k_values: Iterable[int]
    group: str = "spatial"  # coarse family tag, used for SLURM-array sharding
    categorical_filter_roots: Optional[list[str]] = None

    # Option-A composite kernel controls.
    kernel: str = "gaussian"
    bandwidth: Any = "adaptive"
    bandwidth_scale: float = 1.0
    geo_weight: float = 1.0

    use_feature_distance: bool = False
    numeric_feature_cols: Optional[list[str]] = None
    feature_alpha: float = 0.0       # feature-distance weight
    feature_bandwidth: float = 1.0   # feature-distance normalization

    use_time_trend: bool = False
    use_time_decay: bool = False     # if True, time enters the same composite kernel
    time_weight: float = 0.0
    time_bandwidth_days: Optional[float] = 365.25

    neighbor_time_rule: str = "past"  # conservative default for train-row feature construction
    min_candidates: Optional[int] = None
    candidate_multiplier: int = 10
    include_diagnostics: bool = False


NUMERIC_SIMILARITY_COLS = [
    "char_beds",
    "char_yrblt",
    "char_bldg_sf",
    "char_land_sf",
    "char_fbath",
    "char_hbath",
]

# ---------------------------------------------------------------------
# Experiment grid configuration
# ---------------------------------------------------------------------
# Defaults are intentionally moderate; the CLI / SLURM launcher widens them.
K_VALUES = sorted({int(round(v)) for v in np.linspace(1, 80, 40)})
KERNELS = ["gaussian"]
GEO_WEIGHT_VALUES = [1.0]
FEATURE_WEIGHT_VALUES = [0.25, 0.5, 1.0]
TIME_WEIGHT_VALUES = [0.25, 0.5, 1.0]
BANDWIDTH_SCALE_VALUES = [1.0]

DEFAULT_FEATURE_BANDWIDTH = 1.0
DEFAULT_TIME_BANDWIDTH_DAYS = 365.25
DEFAULT_BANDWIDTH = "adaptive"
DEFAULT_FILTER_ROOTS = ["char_type_resd"]

# Coarse families used both for readability and SLURM-array sharding. Each entry
# toggles which kernel-distance components are active in the composite kernel.
EXPERIMENT_GROUPS = ["spatial", "spatial_nofilter", "feature", "trend", "time"]


def _tag(value: Any) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def build_experiments(
    *,
    groups: Iterable[str] = EXPERIMENT_GROUPS,
    k_values: Iterable[int] = K_VALUES,
    kernels: Iterable[str] = KERNELS,
    geo_weights: Iterable[float] = GEO_WEIGHT_VALUES,
    feature_weights: Iterable[float] = FEATURE_WEIGHT_VALUES,
    time_weights: Iterable[float] = TIME_WEIGHT_VALUES,
    bandwidth_scales: Iterable[float] = BANDWIDTH_SCALE_VALUES,
    feature_bandwidth: float = DEFAULT_FEATURE_BANDWIDTH,
    time_bandwidth_days: float = DEFAULT_TIME_BANDWIDTH_DAYS,
    bandwidth: Any = DEFAULT_BANDWIDTH,
    filter_roots: Optional[list[str]] = None,
    numeric_feature_cols: Optional[list[str]] = None,
    min_candidates: int = 50,
) -> list[NeighborExperiment]:
    """Build the neighbor-experiment grid.

    Groups
    ------
    spatial          : geography-only composite kernel, with categorical filtering.
    spatial_nofilter : geography-only kernel, no categorical pooling (filtering ablation).
    feature          : geography + standardized hedonic feature distance.
    trend            : feature group + global log-price time-trend adjustment.
    time             : trend group + time-decay component in the composite kernel.
    """
    groups = list(groups)
    k_values = [int(k) for k in k_values]
    kernels = list(kernels)
    bandwidth_scales = [float(b) for b in bandwidth_scales]
    filter_roots = list(DEFAULT_FILTER_ROOTS if filter_roots is None else filter_roots)
    numeric_feature_cols = list(
        NUMERIC_SIMILARITY_COLS if numeric_feature_cols is None else numeric_feature_cols
    )
    experiments: list[NeighborExperiment] = []

    def kbtag(kern: str, bs: float) -> str:
        return f"{kern}_bs{_tag(bs)}"

    for kern, bs in product(kernels, bandwidth_scales):
        if "spatial" in groups:
            for gw in geo_weights:
                experiments.append(NeighborExperiment(
                    name=f"spatial_{kbtag(kern, bs)}_g{_tag(gw)}",
                    group="spatial",
                    k_values=k_values,
                    categorical_filter_roots=filter_roots,
                    kernel=kern,
                    bandwidth=bandwidth,
                    bandwidth_scale=bs,
                    geo_weight=float(gw),
                ))

        if "spatial_nofilter" in groups:
            for gw in geo_weights:
                experiments.append(NeighborExperiment(
                    name=f"spatialnf_{kbtag(kern, bs)}_g{_tag(gw)}",
                    group="spatial_nofilter",
                    k_values=k_values,
                    categorical_filter_roots=None,
                    kernel=kern,
                    bandwidth=bandwidth,
                    bandwidth_scale=bs,
                    geo_weight=float(gw),
                ))

        if "feature" in groups:
            for gw, fw in product(geo_weights, feature_weights):
                experiments.append(NeighborExperiment(
                    name=f"feature_{kbtag(kern, bs)}_g{_tag(gw)}_f{_tag(fw)}",
                    group="feature",
                    k_values=k_values,
                    categorical_filter_roots=filter_roots,
                    kernel=kern,
                    bandwidth=bandwidth,
                    bandwidth_scale=bs,
                    geo_weight=float(gw),
                    use_feature_distance=True,
                    numeric_feature_cols=numeric_feature_cols,
                    feature_alpha=float(fw),
                    feature_bandwidth=float(feature_bandwidth),
                    min_candidates=min_candidates,
                ))

        if "trend" in groups:
            for gw, fw in product(geo_weights, feature_weights):
                experiments.append(NeighborExperiment(
                    name=f"trend_{kbtag(kern, bs)}_g{_tag(gw)}_f{_tag(fw)}",
                    group="trend",
                    k_values=k_values,
                    categorical_filter_roots=filter_roots,
                    kernel=kern,
                    bandwidth=bandwidth,
                    bandwidth_scale=bs,
                    geo_weight=float(gw),
                    use_feature_distance=True,
                    numeric_feature_cols=numeric_feature_cols,
                    feature_alpha=float(fw),
                    feature_bandwidth=float(feature_bandwidth),
                    use_time_trend=True,
                    min_candidates=min_candidates,
                ))

        if "time" in groups:
            for gw, fw, tw in product(geo_weights, feature_weights, time_weights):
                experiments.append(NeighborExperiment(
                    name=f"time_{kbtag(kern, bs)}_g{_tag(gw)}_f{_tag(fw)}_t{_tag(tw)}",
                    group="time",
                    k_values=k_values,
                    categorical_filter_roots=filter_roots,
                    kernel=kern,
                    bandwidth=bandwidth,
                    bandwidth_scale=bs,
                    geo_weight=float(gw),
                    use_feature_distance=True,
                    numeric_feature_cols=numeric_feature_cols,
                    feature_alpha=float(fw),
                    feature_bandwidth=float(feature_bandwidth),
                    use_time_trend=True,
                    use_time_decay=True,
                    time_weight=float(tw),
                    time_bandwidth_days=float(time_bandwidth_days),
                    min_candidates=min_candidates,
                ))

    return experiments


DEFAULT_NEIGHBOR_EXPERIMENTS = build_experiments()


# ---------------------------------------------------------------------
# Data split and preprocessing
# ---------------------------------------------------------------------
def normalize_datasets(datasets: Any) -> list[tuple[str, pd.DataFrame]]:
    """Accept either a list of dataframes or a {label: dataframe} dictionary."""
    if isinstance(datasets, dict):
        return [(str(k), v.copy()) for k, v in datasets.items()]
    return [(f"dataset_{i + 1}", d.copy()) for i, d in enumerate(datasets)]


def chronological_train_test_split(
    dataset: pd.DataFrame,
    *,
    date_col: str = "meta_sale_date",
    train_frac: float = 0.80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = dataset.copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data = data.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    split_idx = int(float(train_frac) * len(data))
    train = data.iloc[:split_idx].copy()
    test = data.iloc[split_idx:].copy()

    # Stable, non-overlapping indices help self-exclusion in the neighbor transformer.
    train.index = pd.Index([f"train_{i}" for i in range(len(train))])
    test.index = pd.Index([f"test_{i}" for i in range(len(test))])
    return train, test


def fit_transform_ccao_pipeline(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    *,
    predictor_cols: list[str],
    categorical_cols: list[str],
    params: dict[str, Any],
    target_col: str = "meta_sale_price",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pipeline = build_model_pipeline(
        pred_vars=predictor_cols,
        cat_vars=categorical_cols,
        id_vars=params["model"]["predictor"].get("id", []),
    )
    train_t = pipeline.fit_transform(train_data, train_data[target_col])
    test_t = pipeline.transform(test_data)

    if len(train_t) == len(train_data):
        train_t.index = train_data.index
    if len(test_t) == len(test_data):
        test_t.index = test_data.index

    for col in ["past_sale_indicator"]:
        if col not in train_t.columns and col in train_data.columns:
            train_t[col] = train_data[col]
        if col not in test_t.columns and col in test_data.columns:
            test_t[col] = test_data[col]

    return train_t, test_t


def split_X_y(
    train_t: pd.DataFrame,
    test_t: pd.DataFrame,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    *,
    params: dict[str, Any],
    target_col: str = "meta_sale_price",
    date_col: str = "meta_sale_date",
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    id_cols = list(params["model"]["predictor"].get("id", []))
    drop_cols = [target_col, date_col] + id_cols

    X_train = train_t.drop(columns=drop_cols, errors="ignore").copy()
    X_test = test_t.drop(columns=drop_cols, errors="ignore").copy()

    y_train = train_t[target_col].copy() if target_col in train_t else train_data[target_col].copy()
    y_test = test_t[target_col].copy() if target_col in test_t else test_data[target_col].copy()
    y_train.index = X_train.index
    y_test.index = X_test.index
    return X_train, y_train.astype(float), X_test, y_test.astype(float)


# ---------------------------------------------------------------------
# Neighbor feature construction
# ---------------------------------------------------------------------
def _load_neighbor_class_from_path(path: Path):
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("preprocessing/spatiotemporal_neighbors", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "SpatioTemporalKernelTargetNeighbors", None)


def resolve_neighbor_class():
    """Resolve and validate the current Option-A neighbor transformer.

    The explicit signature check prevents accidentally using the older class that
    still applies time decay as a separate post-kernel multiplier.
    """
    candidates = []

    # Prefer a normal import from the working directory / Python path.
    try:
        from preprocessing.spatiotemporal_neighbors import SpatioTemporalKernelTargetNeighbors
        candidates.append(SpatioTemporalKernelTargetNeighbors)
    except Exception:
        pass

    # Keep this fallback for repo layouts that place the transformer under preprocessing/.
    try:
        from preprocessing.spatiotemporal_neighbors import SpatioTemporalKernelTargetNeighbors
        candidates.append(SpatioTemporalKernelTargetNeighbors)
    except Exception:
        pass

    # Notebook fallback: class already defined in a previous cell.
    cls = globals().get("SpatioTemporalKernelTargetNeighbors")
    if callable(cls):
        candidates.append(cls)

    # Sandbox/notebook path fallback. Harmless in normal use; useful when the
    # transformer file is next to this runner or in /mnt/data.
    for path in [
        Path(__file__).resolve().parent / "preprocessing" / "spatiotemporal_neighbors.py",
        Path.cwd() / "preprocessing/spatiotemporal_neighbors.py",
        Path("/mnt/data/preprocessing/spatiotemporal_neighbors.py"),
    ]:
        try:
            cls_from_path = _load_neighbor_class_from_path(path)
        except Exception:
            cls_from_path = None
        if callable(cls_from_path):
            candidates.append(cls_from_path)

    required_params = {"geo_weight", "feature_bandwidth", "time_weight", "time_bandwidth_days"}
    for candidate in candidates:
        if not callable(candidate):
            continue
        params = set(inspect.signature(candidate.__init__).parameters)
        if required_params.issubset(params):
            return candidate

    raise ImportError(
        "Could not resolve the current Option-A SpatioTemporalKernelTargetNeighbors. "
        "Make sure preprocessing/spatiotemporal_neighbors.py is on the Python path "
        "and includes geo_weight, feature_bandwidth, time_weight, and time_bandwidth_days. "
        "If you defined the class in a notebook, re-run the updated class cell first."
    )


def required_neighbor_columns(exp: NeighborExperiment) -> list[str]:
    cols = ["loc_latitude", "loc_longitude"]
    if exp.use_time_trend or exp.use_time_decay or exp.neighbor_time_rule != "none":
        cols.append("meta_sale_date")
    if exp.use_feature_distance and exp.numeric_feature_cols:
        cols.extend(exp.numeric_feature_cols)
    return list(dict.fromkeys(cols))


def add_raw_columns_for_neighbors(
    X: pd.DataFrame,
    raw_data: pd.DataFrame,
    cols: Iterable[str],
) -> pd.DataFrame:
    X = X.copy()
    for col in dict.fromkeys(c for c in cols if c):
        if col not in X.columns and col in raw_data.columns:
            X[col] = raw_data[col]
    return X


def build_neighbor_transformer(exp: NeighborExperiment, k: int):
    NeighborClass = resolve_neighbor_class()
    needed = required_neighbor_columns(exp)
    return NeighborClass(
        k=int(k),
        lat_col="loc_latitude",
        lon_col="loc_longitude",
        date_col="meta_sale_date" if "meta_sale_date" in needed else None,
        kernel=exp.kernel,
        bandwidth=exp.bandwidth,
        bandwidth_scale=float(exp.bandwidth_scale),
        geo_weight=float(exp.geo_weight),
        target_transform="log",
        include_aggregate=True,
        include_diagnostics=bool(exp.include_diagnostics),
        categorical_filter_roots=exp.categorical_filter_roots,
        filter_fallback="global",
        use_feature_distance=bool(exp.use_feature_distance),
        numeric_feature_cols=exp.numeric_feature_cols,
        feature_scaler="standard",
        feature_alpha=float(exp.feature_alpha),
        feature_bandwidth=float(exp.feature_bandwidth),
        candidate_multiplier=int(exp.candidate_multiplier),
        min_candidates=exp.min_candidates,
        use_time_trend=bool(exp.use_time_trend),
        time_trend="linear",
        use_time_decay=bool(exp.use_time_decay),
        time_weight=float(exp.time_weight),
        time_bandwidth_days=exp.time_bandwidth_days,
        neighbor_time_rule=str(exp.neighbor_time_rule),
        insufficient_neighbors="nan",
        exclude_self=True,
        feature_prefix="geo",
    )


def _attach_raw_columns(frame: pd.DataFrame, raw: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Force-set raw geo/date/numeric columns, overwriting any transformed copy.

    `frame` and `raw` share the same train/test row order, so values are assigned
    positionally; this guarantees the neighbor transformer sees true coordinates
    and feature values rather than pipeline-scaled ones.
    """
    frame = frame.copy()
    for col in dict.fromkeys(c for c in cols if c):
        if col in raw.columns:
            frame[col] = raw[col].to_numpy()
    return frame


def neighbor_input_frame(X_transformed: pd.DataFrame, raw: pd.DataFrame, exp: NeighborExperiment) -> pd.DataFrame:
    """Slim input for the neighbor transformer: the categorical-filter one-hot
    columns produced by the CCAO pipeline plus the true raw geo/date/numeric columns."""
    filter_cols: list[str] = []
    for root in (exp.categorical_filter_roots or []):
        prefix = f"{root}_"
        for col in X_transformed.columns:
            semantic = str(col).split("__")[-1]
            if semantic == root or semantic.startswith(prefix):
                filter_cols.append(col)
    frame = X_transformed[filter_cols].copy() if filter_cols else pd.DataFrame(index=X_transformed.index)
    return _attach_raw_columns(frame, raw, required_neighbor_columns(exp))


def compute_neighbor_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_price: pd.Series,
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    exp: NeighborExperiment,
    k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the neighbor transformer on train and emit kernel target features for both
    splits. This is model-agnostic, so it is computed once per (experiment, k) and
    reused by every model (linear and LGBM)."""
    Xn_train = neighbor_input_frame(X_train, train_raw, exp)
    Xn_test = neighbor_input_frame(X_test, test_raw, exp)
    neighbor = build_neighbor_transformer(exp, k)
    G_train, G_test = neighbor.fit_transform_train_test(Xn_train, Xn_test, y_train_price)
    G_train.index = X_train.index
    G_test.index = X_test.index
    return G_train, G_test


# ---------------------------------------------------------------------
# Model fitting and tables
# ---------------------------------------------------------------------
MODEL_CHOICES = ("linear", "lgbm", "cov")
LGBM_LIKE_MODELS = {"lgbm", "cov"}
DEFAULT_COV_RHO = 2.397
DEFAULT_COV_RATIO_MODE = "diff"
DEFAULT_COV_EARLY_STOPPING_ROUNDS = 10


def _needs_lgbm_base(models: Iterable[str]) -> bool:
    return any(str(model) in LGBM_LIKE_MODELS for model in models)


def numeric_model_matrices(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """LinearRegression needs finite numeric inputs; fit imputers on train only."""
    X_train = X_train.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X_test = X_test.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    medians = X_train.median(axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X_train.fillna(medians).fillna(0.0), X_test.fillna(medians).fillna(0.0)


def load_lgbm_params(path: str | Path = "model_params.yaml", *, n_jobs: int = 1) -> dict[str, Any]:
    """Load the tuned LGBMRegressor config (model_params.yaml). Keys map directly to
    sklearn LGBMRegressor kwargs."""
    with open(path, "r", encoding="utf-8") as f:
        model_params = yaml.safe_load(f) or {}
    lgbm_params = dict(model_params.get("LGBMRegressor", {}))
    lgbm_params.setdefault("objective", "mse")
    lgbm_params.setdefault("verbosity", -1)
    lgbm_params["n_jobs"] = int(n_jobs)
    return lgbm_params


def build_lgbm_base(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    predictor_cols: list[str],
    categorical_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raw-predictor matrices with native pandas categorical dtype (LGBM best practice,
    matching run_temporal_cv.py). NaNs are preserved for LGBM to handle internally.

    LGBM only accepts int/float/bool or pandas 'category' columns, so any declared
    categorical column and any remaining object/string predictor is cast to category.
    """
    cols = [c for c in predictor_cols if c in train_raw.columns]
    X_train = train_raw[cols].copy()
    X_test = test_raw[cols].copy()
    cat_set = set(categorical_cols)
    for col in cols:
        is_object = X_train[col].dtype == object or str(X_train[col].dtype) == "string"
        if col in cat_set or is_object:
            X_train[col] = X_train[col].astype("category")
            X_test[col] = X_test[col].astype("category")
    return X_train, X_test


def _predict_linear(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train_log: np.ndarray):
    Xtr, Xte = numeric_model_matrices(X_train, X_test)
    model = LinearRegression(fit_intercept=True)
    model.fit(Xtr, np.asarray(y_train_log, dtype=float))
    return model.predict(Xtr), model.predict(Xte)


def _predict_lgbm(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train_log: np.ndarray, lgbm_params: dict[str, Any]):
    if lgb is None:
        raise ImportError("lightgbm is not installed; cannot run the 'lgbm' model.")
    model = lgb.LGBMRegressor(**dict(lgbm_params))
    model.fit(X_train, np.asarray(y_train_log, dtype=float))
    return model.predict(X_train), model.predict(X_test)


def _predict_cov(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_log: np.ndarray,
    lgbm_params: dict[str, Any],
    *,
    cov_rho: float,
):
    if LGBCovPenalty is None:
        raise ImportError("LGBCovPenalty is unavailable; cannot run the 'cov' model.")
    model = LGBCovPenalty(
        rho=float(cov_rho),
        ratio_mode=DEFAULT_COV_RATIO_MODE,
        early_stopping_rounds=DEFAULT_COV_EARLY_STOPPING_ROUNDS,
        zero_grad_tol=1e-12,
        lgbm_params=dict(lgbm_params),
        verbose=False,
    )
    model.fit(X_train, np.asarray(y_train_log, dtype=float))
    return model.predict(X_train), model.predict(X_test)


def fit_score_model(
    model_name: str,
    *,
    linear_train: Optional[pd.DataFrame],
    linear_test: Optional[pd.DataFrame],
    lgbm_train: Optional[pd.DataFrame],
    lgbm_test: Optional[pd.DataFrame],
    y_train_log: np.ndarray,
    y_test_log: np.ndarray,
    lgbm_params: dict[str, Any],
    cov_rho: float = DEFAULT_COV_RHO,
    boot_indices: Optional[list[np.ndarray]] = None,
) -> dict[str, dict[str, float]]:
    """Fit one model on log(price) and return train/OOS metric dicts on the price scale.

    OOS metrics are averaged over the shared bootstrap resamples when ``boot_indices``
    is provided (otherwise a single point estimate is used)."""
    if model_name == "linear":
        train_pred_log, test_pred_log = _predict_linear(linear_train, linear_test, y_train_log)
    elif model_name == "lgbm":
        train_pred_log, test_pred_log = _predict_lgbm(lgbm_train, lgbm_test, y_train_log, lgbm_params)
    elif model_name == "cov":
        train_pred_log, test_pred_log = _predict_cov(lgbm_train, lgbm_test, y_train_log, lgbm_params, cov_rho=cov_rho)
    else:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {MODEL_CHOICES}.")
    return {
        "train": compute_metrics(y_train_log, train_pred_log, y_train_log),
        "oos": compute_oos_metrics_boot(y_test_log, test_pred_log, y_train_log, boot_indices),
    }


def metric_row(meta: dict[str, Any], split: str, metrics: dict[str, float]) -> dict[str, Any]:
    return {**meta, "split": split, **metrics}


def print_metrics(name: str, metrics_by_split: dict[str, dict[str, float]]) -> None:
    tr = metrics_by_split["train"]
    oo = metrics_by_split["oos"]
    print(f"{name}", flush=True)
    print(f"  Train: R2={tr['R2']:.4f} | MAPE={tr['MAPE']:.3f} | COD={tr['COD']:.2f} | PRD={tr['PRD']:.3f} | PRB={tr['PRB']:.3f} | VEI={tr['VEI']:.2f}", flush=True)
    print(f"  OOS:   R2={oo['R2']:.4f} | MAPE={oo['MAPE']:.3f} | COD={oo['COD']:.2f} | PRD={oo['PRD']:.3f} | PRB={oo['PRB']:.3f} | VEI={oo['VEI']:.2f}", flush=True)


def resolve_n_jobs(n_jobs: Optional[int], n_tasks: int, *, parallel: bool) -> int:
    if not parallel or int(n_tasks) <= 1:
        return 1
    if n_jobs is None:
        return max(1, min(8, int(os.cpu_count() or 1), int(n_tasks)))
    return max(1, min(int(n_jobs), int(n_tasks)))


def _baseline_meta(
    model_name: str,
    dataset_idx: int,
    dataset_label: str,
    *,
    cov_rho: float = DEFAULT_COV_RHO,
) -> dict[str, Any]:
    return {
        "dataset_id": dataset_idx,
        "dataset_label": dataset_label,
        "experiment": "baseline_no_neighbors",
        "group": "baseline",
        "model": model_name,
        "rho": float(cov_rho) if model_name == "cov" else np.nan,
        "ratio_mode": DEFAULT_COV_RATIO_MODE if model_name == "cov" else "",
        "k": np.nan,
        "cat_filters": "",
        "numeric_features": "",
        "kernel": "",
        "bandwidth": "",
        "bandwidth_scale": np.nan,
        "geo_weight": np.nan,
        "feature_weight": np.nan,
        "feature_bandwidth": np.nan,
        "time_trend": False,
        "time_decay": False,
        "time_weight": np.nan,
        "time_bandwidth_days": np.nan,
        "neighbor_time_rule": "none",
    }


def _experiment_meta(
    model_name: str,
    *,
    dataset_idx: int,
    dataset_label: str,
    exp: NeighborExperiment,
    k: int,
    cov_rho: float = DEFAULT_COV_RHO,
) -> dict[str, Any]:
    return {
        "dataset_id": dataset_idx,
        "dataset_label": dataset_label,
        "experiment": exp.name,
        "group": exp.group,
        "model": model_name,
        "rho": float(cov_rho) if model_name == "cov" else np.nan,
        "ratio_mode": DEFAULT_COV_RATIO_MODE if model_name == "cov" else "",
        "k": int(k),
        "cat_filters": ",".join(exp.categorical_filter_roots or []),
        "numeric_features": ",".join(exp.numeric_feature_cols or []),
        "kernel": exp.kernel,
        "bandwidth": exp.bandwidth,
        "bandwidth_scale": float(exp.bandwidth_scale),
        "geo_weight": float(exp.geo_weight),
        "feature_weight": float(exp.feature_alpha) if exp.use_feature_distance else 0.0,
        "feature_bandwidth": float(exp.feature_bandwidth) if exp.use_feature_distance else np.nan,
        "time_trend": bool(exp.use_time_trend),
        "time_decay": bool(exp.use_time_decay),
        "time_weight": float(exp.time_weight) if exp.use_time_decay else 0.0,
        "time_bandwidth_days": exp.time_bandwidth_days if exp.use_time_decay else np.nan,
        "neighbor_time_rule": str(exp.neighbor_time_rule),
    }


def run_neighbor_task(
    *,
    dataset_idx: int,
    dataset_label: str,
    task_order: int,
    n_tasks: int,
    exp: NeighborExperiment,
    k: int,
    models: list[str],
    linear_train_base: pd.DataFrame,
    linear_test_base: pd.DataFrame,
    lgbm_train_base: Optional[pd.DataFrame],
    lgbm_test_base: Optional[pd.DataFrame],
    y_train_price: pd.Series,
    y_train_log: np.ndarray,
    y_test_log: np.ndarray,
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    lgbm_params: dict[str, Any],
    cov_rho: float = DEFAULT_COV_RHO,
    boot_indices: Optional[list[np.ndarray]] = None,
) -> tuple[str, dict[str, dict[str, dict[str, float]]], list[dict[str, Any]]]:
    """Compute neighbor features once for (exp, k), then fit every requested model."""
    task_start = time.perf_counter()
    log(
        "neighbor task start",
        dataset=dataset_label,
        dataset_id=dataset_idx,
        task=f"{int(task_order) + 1}/{int(n_tasks)}",
        experiment=exp.name,
        models=",".join(models),
        k=int(k),
    )
    G_train, G_test = compute_neighbor_features(
        linear_train_base, linear_test_base, y_train_price, train_raw, test_raw, exp, int(k)
    )

    rows: list[dict[str, Any]] = []
    metrics_by_model: dict[str, dict[str, dict[str, float]]] = {}
    for model_name in models:
        if model_name == "linear":
            lin_train = pd.concat([linear_train_base, G_train], axis=1)
            lin_test = pd.concat([linear_test_base, G_test], axis=1)
            metrics = fit_score_model(
                "linear",
                linear_train=lin_train, linear_test=lin_test,
                lgbm_train=None, lgbm_test=None,
                y_train_log=y_train_log, y_test_log=y_test_log, lgbm_params=lgbm_params,
                cov_rho=cov_rho,
                boot_indices=boot_indices,
            )
        else:
            gbm_train = pd.concat([lgbm_train_base, G_train], axis=1)
            gbm_test = pd.concat([lgbm_test_base, G_test], axis=1)
            metrics = fit_score_model(
                model_name,
                linear_train=None, linear_test=None,
                lgbm_train=gbm_train, lgbm_test=gbm_test,
                y_train_log=y_train_log, y_test_log=y_test_log, lgbm_params=lgbm_params,
                cov_rho=cov_rho,
                boot_indices=boot_indices,
            )
        meta = _experiment_meta(
            model_name,
            dataset_idx=dataset_idx,
            dataset_label=dataset_label,
            exp=exp,
            k=int(k),
            cov_rho=cov_rho,
        )
        rows.extend(metric_row(meta, split, met) for split, met in metrics.items())
        metrics_by_model[model_name] = metrics

    label = f"{exp.name} | k={int(k)}"
    log(
        "neighbor task done",
        dataset=dataset_label,
        dataset_id=dataset_idx,
        task=f"{int(task_order) + 1}/{int(n_tasks)}",
        experiment=exp.name,
        k=int(k),
        elapsed_sec=f"{time.perf_counter() - task_start:.1f}",
    )
    return label, metrics_by_model, rows


def _write_metric_csvs(results: pd.DataFrame, save_path: Path, suffix: str) -> None:
    """Write the all/train/oos metric CSVs under the standard filenames.

    Reused for both incremental checkpoints and the final save, so a crashed or
    timed-out shard still leaves usable, aggregatable outputs (the plot/aggregate
    step globs these same filenames)."""
    save_path.mkdir(parents=True, exist_ok=True)
    results.to_csv(save_path / f"neighbor_experiment_all_results_train_oos{suffix}.csv", index=False)
    results.loc[results["split"] == "train"].to_csv(
        save_path / f"neighbor_experiment_train_metrics{suffix}.csv", index=False)
    results.loc[results["split"] == "oos"].to_csv(
        save_path / f"neighbor_experiment_oos_metrics{suffix}.csv", index=False)


def run_one_dataset(
    *,
    dataset: pd.DataFrame,
    dataset_label: str,
    dataset_idx: int,
    params: dict[str, Any],
    predictor_cols: list[str],
    categorical_cols: list[str],
    experiments: Iterable[NeighborExperiment],
    models: list[str],
    train_frac: float,
    parallel: bool,
    n_jobs: Optional[int],
    lgbm_params: dict[str, Any],
    train_size: Optional[int] = 80_000,
    test_size: Optional[int] = 20_000,
    n_bootstrap: int = 5,
    block_freq: str = "M",
    cov_rho: float = DEFAULT_COV_RHO,
    seed: int = 42,
    target_col: str = "meta_sale_price",
    date_col: str = "meta_sale_date",
    flush_fn: Optional[Any] = None,
) -> list[dict[str, Any]]:
    dataset_start = time.perf_counter()
    print(f"\n--- {dataset_label} ---", flush=True)

    # Split the FULL universe once (chronological). The most-recent slice becomes the
    # COMMON OOS test set for every filter/experiment; the dataset filter is applied
    # to TRAIN ONLY. This keeps the measured test set identical (and unfiltered)
    # across all experiments, so results are directly comparable.
    train_full, test_full = chronological_train_test_split(dataset, date_col=date_col, train_frac=train_frac)

    if test_size and len(test_full) > int(test_size):
        test_raw = test_full.sample(int(test_size), random_state=int(seed)).sort_values(date_col).copy()
    else:
        test_raw = test_full.copy()

    mask = train_filter_mask(train_full, dataset_label)
    train_raw = train_full.loc[mask]
    if train_size and len(train_raw) > int(train_size):
        train_raw = train_raw.sample(int(train_size), random_state=int(seed))
    train_raw = train_raw.copy()

    log("dataset start", dataset=dataset_label, dataset_id=dataset_idx,
        train_rows=len(train_raw), test_rows=len(test_raw),
        train_filter=dataset_label, models=",".join(models))

    # Shared month-block bootstrap of the common test set: identical positional
    # indices for every experiment, so OOS metrics are averaged on the same subsamples.
    boot_indices: Optional[list[np.ndarray]] = None
    if n_bootstrap and int(n_bootstrap) > 0:
        boot_indices = _build_time_block_bootstrap_indices(
            pd.to_datetime(test_raw[date_col]).reset_index(drop=True),
            n_bootstrap=int(n_bootstrap),
            block_freq=str(block_freq),
            rng_seed=int(seed),
        )
    log(
        "dataset split complete",
        dataset=dataset_label,
        dataset_id=dataset_idx,
        train_rows=len(train_raw),
        oos_rows=len(test_raw),
        n_bootstrap=int(n_bootstrap),
    )
    train_t, test_t = fit_transform_ccao_pipeline(
        train_raw,
        test_raw,
        predictor_cols=predictor_cols,
        categorical_cols=categorical_cols,
        params=params,
        target_col=target_col,
    )
    log(
        "preprocessing complete",
        dataset=dataset_label,
        dataset_id=dataset_idx,
        train_rows=len(train_t),
        oos_rows=len(test_t),
    )
    X_train, y_train, X_test, y_test = split_X_y(
        train_t,
        test_t,
        train_raw,
        test_raw,
        params=params,
        target_col=target_col,
        date_col=date_col,
    )
    # Models are fit on log(price); metrics are reported back on the price scale.
    y_train_log = np.log(np.asarray(y_train, dtype=float))
    y_test_log = np.log(np.asarray(y_test, dtype=float))

    linear_train_base, linear_test_base = X_train, X_test
    lgbm_train_base = lgbm_test_base = None
    if _needs_lgbm_base(models):
        lgbm_train_base, lgbm_test_base = build_lgbm_base(train_raw, test_raw, predictor_cols, categorical_cols)

    rows: list[dict[str, Any]] = []
    for model_name in models:
        base_metrics = fit_score_model(
            model_name,
            linear_train=linear_train_base, linear_test=linear_test_base,
            lgbm_train=lgbm_train_base, lgbm_test=lgbm_test_base,
            y_train_log=y_train_log, y_test_log=y_test_log, lgbm_params=lgbm_params,
            cov_rho=cov_rho,
            boot_indices=boot_indices,
        )
        rows.extend(
            metric_row(_baseline_meta(model_name, dataset_idx, dataset_label, cov_rho=cov_rho), split, met)
            for split, met in base_metrics.items()
        )
        log("baseline model complete", dataset=dataset_label, dataset_id=dataset_idx, model=model_name)
        print_metrics(f"baseline_no_neighbors [{model_name}]", base_metrics)

    # Checkpoint baselines immediately so they survive any later neighbor-task crash.
    if flush_fn is not None:
        flush_fn(rows)

    tasks = [
        (i, exp, int(k))
        for i, (exp, k) in enumerate((exp, k) for exp in experiments for k in exp.k_values)
    ]
    if not tasks:
        log("dataset done", dataset=dataset_label, dataset_id=dataset_idx, elapsed_sec=f"{time.perf_counter() - dataset_start:.1f}")
        return rows

    workers = resolve_n_jobs(n_jobs, len(tasks), parallel=parallel)
    log(
        "neighbor fits start",
        dataset=dataset_label,
        dataset_id=dataset_idx,
        tasks=len(tasks),
        n_jobs=workers,
        parallel=bool(workers > 1),
    )

    def _submit(order: int, exp: NeighborExperiment, k: int):
        return run_neighbor_task(
            dataset_idx=dataset_idx,
            dataset_label=dataset_label,
            task_order=order,
            n_tasks=len(tasks),
            exp=exp,
            k=k,
            models=models,
            linear_train_base=linear_train_base,
            linear_test_base=linear_test_base,
            lgbm_train_base=lgbm_train_base,
            lgbm_test_base=lgbm_test_base,
            y_train_price=y_train,
            y_train_log=y_train_log,
            y_test_log=y_test_log,
            boot_indices=boot_indices,
            train_raw=train_raw,
            test_raw=test_raw,
            lgbm_params=lgbm_params,
            cov_rho=cov_rho,
        )

    # Tasks are collected as they finish; rows are appended and checkpointed per
    # task. A single failing task is logged and skipped (non-fatal) so a crash in
    # one configuration never discards the rest of the shard's completed work.
    task_results: dict[int, tuple[str, dict[str, dict[str, dict[str, float]]], list[dict[str, Any]]]] = {}
    failures: list[tuple[int, str, int]] = []

    def _record(order: int, result) -> None:
        task_results[order] = result
        rows.extend(result[2])
        if flush_fn is not None:
            flush_fn(rows)

    if workers == 1:
        for order, exp, k in tasks:
            try:
                _record(order, _submit(order, exp, k))
            except Exception as e:
                failures.append((order, exp.name, int(k)))
                log("neighbor task failed", dataset=dataset_label, dataset_id=dataset_idx,
                    task=f"{order + 1}/{len(tasks)}", experiment=exp.name, k=int(k), error=repr(e))
                continue
            log("neighbor progress", dataset=dataset_label, dataset_id=dataset_idx,
                completed=len(task_results), total=len(tasks), last_experiment=exp.name, last_k=int(k))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_order = {executor.submit(_submit, order, exp, k): order for order, exp, k in tasks}
            for future in as_completed(future_to_order):
                order = future_to_order[future]
                _, exp, k = tasks[order]
                try:
                    _record(order, future.result())
                except Exception as e:
                    failures.append((order, exp.name, int(k)))
                    log("neighbor task failed", dataset=dataset_label, dataset_id=dataset_idx,
                        task=f"{order + 1}/{len(tasks)}", experiment=exp.name, k=int(k), error=repr(e))
                    continue
                log("neighbor progress", dataset=dataset_label, dataset_id=dataset_idx,
                    completed=len(task_results), total=len(tasks), last_experiment=exp.name, last_k=int(k))

    for order in sorted(task_results):
        label, metrics_by_model, _ = task_results[order]
        for model_name, metrics in metrics_by_model.items():
            print_metrics(f"{label} [{model_name}]", metrics)

    if failures:
        log("dataset task failures", dataset=dataset_label, dataset_id=dataset_idx,
            failed=len(failures), total=len(tasks))

    log(
        "dataset done",
        dataset=dataset_label,
        dataset_id=dataset_idx,
        elapsed_sec=f"{time.perf_counter() - dataset_start:.1f}",
    )
    return rows


# Selection objective for the "best configuration" overview: maximize OOS accuracy
# while preferring assessments closer to vertical equity (PRD~1, PRB~0, low COD).
_OVERVIEW_KEEP = [
    "dataset_label", "experiment", "group", "model", "k",
    "rho", "ratio_mode",
    "R2", "OOS R2", "MAPE", "MdAPE", "COD", "COV_IAAO", "VEI", "PRD", "PRB", "MKI",
    "Median ratio", "Mean ratio", "W. Mean ratio", "N",
    "cat_filters", "kernel", "bandwidth", "bandwidth_scale",
    "geo_weight", "feature_weight", "feature_bandwidth", "time_trend",
    "time_decay", "time_weight", "time_bandwidth_days", "neighbor_time_rule",
]


def build_overview_table(results_df: pd.DataFrame, *, split: str = "oos") -> pd.DataFrame:
    """Pick the best k per (dataset, experiment, model) using split-specific R2 first,
    then vertical-equity tie-breakers."""
    df = results_df.loc[results_df["split"] == split].copy()
    if df.empty:
        return df

    df["abs_PRD_dev"] = (df["PRD"] - 1.0).abs()
    df["abs_PRB_dev"] = df["PRB"].abs()

    picked = []
    for _, group in df.groupby(["dataset_id", "experiment", "model"], dropna=False):
        group = group.sort_values(
            by=["R2", "abs_PRD_dev", "abs_PRB_dev", "COD", "MAPE"],
            ascending=[False, True, True, True, True],
        )
        picked.append(group.iloc[0])

    out = pd.DataFrame(picked)
    keep = [c for c in _OVERVIEW_KEEP if c in out.columns]
    return out[keep].sort_values(["dataset_label", "model", "experiment"]).reset_index(drop=True)


def run_all_experiments(
    *,
    datasets: Any,
    params: dict[str, Any],
    predictor_cols: list[str],
    categorical_cols: list[str],
    experiments: Optional[Iterable[NeighborExperiment]] = None,
    models: Optional[list[str]] = None,
    train_frac: float = 0.80,
    parallel: bool = True,
    n_jobs: Optional[int] = None,
    lgbm_params: Optional[dict[str, Any]] = None,
    train_size: Optional[int] = 80_000,
    test_size: Optional[int] = 20_000,
    n_bootstrap: int = 5,
    block_freq: str = "M",
    cov_rho: float = DEFAULT_COV_RHO,
    seed: int = 42,
    save_dir: Optional[str | Path] = None,
    tag: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_start = time.perf_counter()
    experiments = list(DEFAULT_NEIGHBOR_EXPERIMENTS if experiments is None else experiments)
    models = list(MODEL_CHOICES) if models is None else list(models)
    lgbm_params = load_lgbm_params() if lgbm_params is None else lgbm_params
    normalized_datasets = normalize_datasets(datasets)
    log(
        "run start",
        datasets=len(normalized_datasets),
        experiments=len(experiments),
        models=",".join(models),
        parallel=parallel,
        n_jobs=n_jobs,
        cov_rho=float(cov_rho),
    )

    checkpoint_path = Path(save_dir) if save_dir is not None else None
    checkpoint_suffix = f"_{tag}" if tag else ""

    all_rows: list[dict[str, Any]] = []
    for i, (label, data) in enumerate(normalized_datasets):
        log("run position", dataset=label, dataset_id=i, dataset_number=f"{i + 1}/{len(normalized_datasets)}")

        def _flush(current_rows: list[dict[str, Any]], _prior: list[dict[str, Any]] = list(all_rows)) -> None:
            if checkpoint_path is None:
                return
            _write_metric_csvs(pd.DataFrame(_prior + current_rows), checkpoint_path, checkpoint_suffix)

        all_rows.extend(run_one_dataset(
            flush_fn=_flush,
            dataset=data,
            dataset_label=label,
            dataset_idx=i,
            params=params,
            predictor_cols=predictor_cols,
            categorical_cols=categorical_cols,
            experiments=experiments,
            models=models,
            train_frac=train_frac,
            parallel=parallel,
            n_jobs=n_jobs,
            lgbm_params=lgbm_params,
            train_size=train_size,
            test_size=test_size,
            n_bootstrap=n_bootstrap,
            block_freq=block_freq,
            cov_rho=float(cov_rho),
            seed=seed,
        ))

    results = pd.DataFrame(all_rows)
    for col in METRIC_KEYS + ["N", "rho"]:
        if col in results:
            results[col] = pd.to_numeric(results[col], errors="coerce")

    train_df = results.loc[results["split"] == "train"].reset_index(drop=True)
    oos_df = results.loc[results["split"] == "oos"].reset_index(drop=True)
    overview_train = build_overview_table(results, split="train")
    overview_oos = build_overview_table(results, split="oos")

    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        suffix = f"_{tag}" if tag else ""
        results.to_csv(save_path / f"neighbor_experiment_all_results_train_oos{suffix}.csv", index=False)
        train_df.to_csv(save_path / f"neighbor_experiment_train_metrics{suffix}.csv", index=False)
        oos_df.to_csv(save_path / f"neighbor_experiment_oos_metrics{suffix}.csv", index=False)
        overview_train.to_csv(save_path / f"neighbor_experiment_overview_train{suffix}.csv", index=False)
        overview_oos.to_csv(save_path / f"neighbor_experiment_overview_oos{suffix}.csv", index=False)
        log("results saved", save_dir=str(save_path), tag=tag or "(none)")

    log("run done", datasets=len(normalized_datasets), result_rows=len(results), elapsed_sec=f"{time.perf_counter() - run_start:.1f}")
    return results, train_df, oos_df, overview_train, overview_oos


# ---------------------------------------------------------------------
# Command-line entry point (shardable for SLURM arrays)
# ---------------------------------------------------------------------
def _parse_list(value: str, cast=str) -> list:
    return [cast(x.strip()) for x in str(value).split(",") if str(x).strip() != ""]


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Spatiotemporal neighbor-feature experiments (LinearRegression, LGBM, and cov-penalized LGBM).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-path", default="./data/CCAO/2025/training_data.parquet")
    p.add_argument("--params-path", default="params.yaml")
    p.add_argument("--model-params-path", default="model_params.yaml")
    p.add_argument("--train-size", type=int, default=80_000,
                   help="Cap on the (filtered) training rows. Use 0 for no cap.")
    p.add_argument("--test-size", type=int, default=20_000,
                   help="Common OOS test subsample size (identical across all filters). Use 0 for full OOS.")
    p.add_argument("--n-bootstrap", type=int, default=5,
                   help="Month-block bootstrap resamples of the common test set (0 disables).")
    p.add_argument("--bootstrap-block-freq", default="M", help="Pandas period freq for bootstrap blocks (e.g. M, W, Q).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-frac", type=float, default=0.80)

    p.add_argument("--datasets", default="all",
                   help="Comma list of dataset labels, or 'all'. Labels: all_filtered,arms_length_or_missing,deed_01_02,single_family")
    p.add_argument("--models", default="linear,lgbm,cov", help="Comma list from: linear,lgbm,cov")
    p.add_argument("--cov-rho", type=float, default=DEFAULT_COV_RHO,
                   help="rho for the squared covariance penalized LGBM model.")
    p.add_argument("--groups", default=",".join(EXPERIMENT_GROUPS),
                   help=f"Comma list from: {','.join(EXPERIMENT_GROUPS)}")

    p.add_argument("--k-values", default=",".join(str(k) for k in K_VALUES))
    p.add_argument("--kernels", default=",".join(KERNELS), help="Comma list: gaussian,exponential,epanechnikov,triangular")
    p.add_argument("--geo-weights", default=",".join(str(x) for x in GEO_WEIGHT_VALUES))
    p.add_argument("--feature-weights", default=",".join(str(x) for x in FEATURE_WEIGHT_VALUES))
    p.add_argument("--time-weights", default=",".join(str(x) for x in TIME_WEIGHT_VALUES))
    p.add_argument("--bandwidth-scales", default=",".join(str(x) for x in BANDWIDTH_SCALE_VALUES))
    p.add_argument("--feature-bandwidth", type=float, default=DEFAULT_FEATURE_BANDWIDTH)
    p.add_argument("--time-bandwidth-days", type=float, default=DEFAULT_TIME_BANDWIDTH_DAYS)
    p.add_argument("--min-candidates", type=int, default=50)

    p.add_argument("--n-jobs", type=int, default=8, help="Thread workers over (experiment, k) tasks.")
    p.add_argument("--lgbm-n-jobs", type=int, default=1, help="Threads per LGBM fit.")
    p.add_argument("--no-parallel", action="store_true")
    p.add_argument("--save-dir", default="./output/neighbor_experiments")
    p.add_argument("--tag", default="", help="Suffix appended to output CSVs (use for SLURM array shards).")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    params = load_params(args.params_path)

    # Load the full universe once and attach the derived filter columns. The filters
    # are applied to TRAIN only inside run_one_dataset; the OOS test is the common
    # most-recent slice of this universe, so every label shares the same test set.
    data, predictor_cols, categorical_cols = load_ccao_sales_data(
        data_path=args.data_path,
        params=params,
        sample_size=None,
        random_state=int(args.seed),
    )
    universe = add_filter_columns(data)

    if args.datasets.strip().lower() == "all":
        wanted = list(FILTER_LABELS)
    else:
        wanted = _parse_list(args.datasets)
        missing = [d for d in wanted if d not in FILTER_LABELS]
        if missing:
            raise ValueError(f"Unknown dataset label(s) {missing}. Available: {FILTER_LABELS}")
    # Each label runs on the same full universe; the filter is applied to train only.
    selected = {label: universe for label in wanted}

    experiments = build_experiments(
        groups=_parse_list(args.groups),
        k_values=_parse_list(args.k_values, int),
        kernels=_parse_list(args.kernels),
        geo_weights=_parse_list(args.geo_weights, float),
        feature_weights=_parse_list(args.feature_weights, float),
        time_weights=_parse_list(args.time_weights, float),
        bandwidth_scales=_parse_list(args.bandwidth_scales, float),
        feature_bandwidth=float(args.feature_bandwidth),
        time_bandwidth_days=float(args.time_bandwidth_days),
        min_candidates=int(args.min_candidates),
    )
    lgbm_params = load_lgbm_params(args.model_params_path, n_jobs=int(args.lgbm_n_jobs))
    train_size = None if int(args.train_size) <= 0 else int(args.train_size)
    test_size = None if int(args.test_size) <= 0 else int(args.test_size)

    log(
        "cli configured",
        datasets=",".join(wanted),
        models=args.models,
        cov_rho=float(args.cov_rho),
        groups=args.groups,
        kernels=args.kernels,
        n_experiments=len(experiments),
        train_size=train_size,
        test_size=test_size,
        n_bootstrap=int(args.n_bootstrap),
        tag=args.tag or "(none)",
    )

    run_all_experiments(
        datasets=selected,
        params=params,
        predictor_cols=predictor_cols,
        categorical_cols=categorical_cols,
        experiments=experiments,
        models=_parse_list(args.models),
        train_frac=float(args.train_frac),
        parallel=not args.no_parallel,
        n_jobs=int(args.n_jobs),
        lgbm_params=lgbm_params,
        cov_rho=float(args.cov_rho),
        train_size=train_size,
        test_size=test_size,
        n_bootstrap=int(args.n_bootstrap),
        block_freq=str(args.bootstrap_block_freq),
        seed=int(args.seed),
        save_dir=args.save_dir,
        tag=args.tag,
    )


if __name__ == "__main__":
    main()
