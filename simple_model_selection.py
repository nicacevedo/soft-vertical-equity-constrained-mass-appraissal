"""
Simple multi-objective model selection over CV run artifacts.

This script mirrors the input layout of `optimize_stacking_weights.py`, but keeps
the selection logic intentionally simple:

1. Load fold-level CV metrics for a `(result_root, data_id, split_id)` run.
2. Optionally skip the first `k` folds.
3. Only keep configurations whose held-out test metrics were already written.
4. For each penalized model family of interest, filter configs that do not satisfy
   the requested fairness/ratio-study constraints under a chosen fold aggregation.
5. Rank the surviving configs by a chosen accuracy/error metric under a chosen
   fold aggregation.
6. Break near-ties using the fold-to-fold standard deviation of the ranking metric.
7. Write a single CSV containing:
   - the selected summary rows for the penalized families,
   - fold rows for the selected penalized configs,
   - baseline summary rows,
   - baseline fold rows.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from utils.motivation_utils import (
    IAAO_COD_RANGES,
    IAAO_LEVEL_RANGE,
    IAAO_PRB_RANGE,
    IAAO_PRD_RANGE,
    IAAO_VEI_RANGE,
    _compute_extended_metrics,
)


_PENALIZED_FAMILIES: Tuple[str, ...] = ("LGBCovPenalty", "LGBSmoothPenalty")
_BASELINE_FAMILIES: Tuple[str, ...] = ("LinearRegression", "LGBMRegressor")
_COMBINED_SELECTION_GROUP = "ALL_PENALIZED"
_COMBINED_STACKING_GROUP = "ALL_MODELS"


def _candidate_pool_runs_df(runs_df: pd.DataFrame, *, group_name: str) -> pd.DataFrame:
    group_key = str(group_name)
    if group_key == _COMBINED_SELECTION_GROUP:
        families = set(_PENALIZED_FAMILIES)
    elif group_key == _COMBINED_STACKING_GROUP:
        families = set(runs_df["model_family"].dropna().astype(str).unique().tolist())
    else:
        families = {group_key}
    return runs_df.loc[runs_df["model_family"].astype(str).isin(families), :].copy()

# Project defaults for the simple selector. Edit here if policy changes.
_DEFAULT_COD_GUIDANCE_CLASS = "Residential Improved"
_DEFAULT_COD_MAX = float(IAAO_COD_RANGES[_DEFAULT_COD_GUIDANCE_CLASS][1])
_DEFAULT_COV_MAX = 0.15

_DEFAULT_CONSTRAINT_METRICS: Tuple[str, ...] = ("PRD", "PRB", "VEI")
_DEFAULT_SELECTION_METRIC = "R2"
_DEFAULT_SELECTION_METHOD = "both"
_DEFAULT_UTOPIA_AGGREGATION = "average_fold"
_TIE_TOL = 1e-4
_POSITIVE_EPS = 1e-12
_CONVEX_STACKING_SELECTION_METRICS: Tuple[str, ...] = ("MSE", "RMSE", "MAE", "MAPE")
_CONVEX_STACKING_CONSTRAINT_METRICS: Tuple[str, ...] = ("PRD", "MEAN_RATIO", "WEIGHTED_MEAN_RATIO")
_LEVEL_CONSTRAINT_METRICS: Tuple[str, ...] = ("MEAN_RATIO", "MEDIAN_RATIO", "WEIGHTED_MEAN_RATIO")
_DISPERSION_CONSTRAINT_METRICS: Tuple[str, ...] = ("COD", "COV")


@dataclass(frozen=True)
class ConstraintSpec:
    column: str
    lower: Optional[float]
    upper: Optional[float]


@dataclass(frozen=True)
class SelectionMetricSpec:
    column: str
    higher_is_better: bool
    transform: str = "identity"


_CONSTRAINT_SPECS: Dict[str, ConstraintSpec] = {
    "PRD": ConstraintSpec(column="PRD", lower=float(IAAO_PRD_RANGE[0]), upper=float(IAAO_PRD_RANGE[1])),
    "PRB": ConstraintSpec(column="PRB", lower=float(IAAO_PRB_RANGE[0]), upper=float(IAAO_PRB_RANGE[1])),
    "VEI": ConstraintSpec(column="VEI", lower=float(IAAO_VEI_RANGE[0]), upper=float(IAAO_VEI_RANGE[1])),
    "COD": ConstraintSpec(column="COD", lower=None, upper=float(_DEFAULT_COD_MAX)),
    "MEAN_RATIO": ConstraintSpec(column="Mean ratio", lower=float(IAAO_LEVEL_RANGE[0]), upper=float(IAAO_LEVEL_RANGE[1])),
    "MEDIAN_RATIO": ConstraintSpec(column="Median ratio", lower=float(IAAO_LEVEL_RANGE[0]), upper=float(IAAO_LEVEL_RANGE[1])),
    "WEIGHTED_MEAN_RATIO": ConstraintSpec(
        column="W. Mean ratio",
        lower=float(IAAO_LEVEL_RANGE[0]),
        upper=float(IAAO_LEVEL_RANGE[1]),
    ),
    "COV": ConstraintSpec(column="COV_IAAO", lower=None, upper=float(_DEFAULT_COV_MAX)),
}

_SELECTION_METRIC_SPECS: Dict[str, SelectionMetricSpec] = {
    "R2": SelectionMetricSpec(column="R2", higher_is_better=True),
    "R2_LOG": SelectionMetricSpec(column="R2 (log)", higher_is_better=True),
    "MSE": SelectionMetricSpec(column="RMSE", higher_is_better=False, transform="square"),
    "RMSE": SelectionMetricSpec(column="RMSE", higher_is_better=False),
    "RMSE_LOG": SelectionMetricSpec(column="RMSE(LogRatio)", higher_is_better=False),
    "MAE": SelectionMetricSpec(column="MAE", higher_is_better=False),
    "MAPE": SelectionMetricSpec(column="MAPE", higher_is_better=False),
}

_SUMMARY_METRIC_ALIASES: Dict[str, str] = {
    "R2": "R2",
    "OOS R2": "OOS_R2",
    "R2 (log)": "R2_LOG",
    "MSE": "MSE",
    "RMSE": "RMSE",
    "RMSE(LogRatio)": "RMSE_LOG",
    "MAE": "MAE",
    "MAPE": "MAPE",
    "COD": "COD",
    "PRD": "PRD",
    "PRB": "PRB",
    "VEI": "VEI",
    "Median ratio": "MEDIAN_RATIO",
    "Mean ratio": "MEAN_RATIO",
    "W. Mean ratio": "WEIGHTED_MEAN_RATIO",
    "COV_IAAO": "COV",
}


def _parse_identifier_list(raw: str | None, *, default: Sequence[str]) -> List[str]:
    if raw is None:
        return [str(x) for x in default]
    vals = [str(x).strip().upper() for x in str(raw).split(",")]
    vals = [x for x in vals if x]
    return vals if vals else [str(x) for x in default]


def _parse_constraint_metric_subsets(raw: str | None, *, default: Sequence[str]) -> List[List[str]]:
    if raw is None or not str(raw).strip():
        return [list(default)]
    subsets: List[List[str]] = []
    for chunk in str(raw).split(";"):
        vals = [str(x).strip().upper() for x in str(chunk).split(",")]
        vals = [x for x in vals if x]
        if vals:
            subsets.append(vals)
    return subsets if subsets else [list(default)]


def _normalize_constraint_metric_ids(metric_ids: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for raw_metric in metric_ids:
        metric_id = str(raw_metric).strip().upper()
        if not metric_id or metric_id in seen:
            continue
        if metric_id not in _CONSTRAINT_SPECS:
            raise ValueError(
                f"Unknown constraint metric '{metric_id}'. "
                f"Valid options: {sorted(_CONSTRAINT_SPECS.keys())}"
            )
        out.append(metric_id)
        seen.add(metric_id)
    if not out:
        raise ValueError("At least one constraint metric must be specified.")
    return out


def _constraint_metrics_slug(metric_ids: Sequence[str]) -> str:
    return "_".join(str(metric_id).strip().lower() for metric_id in metric_ids if str(metric_id).strip())


def _format_constraint_metrics_label(metric_ids: Sequence[str]) -> str:
    labels = [str(metric_id).strip().upper() for metric_id in metric_ids if str(metric_id).strip()]
    if not labels:
        return ""
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return f"{', '.join(labels[:-1])}, and {labels[-1]}"


def _attach_constraint_subset_metadata(
    result: Dict[str, Any],
    *,
    constraint_metrics: Sequence[str],
) -> Dict[str, Any]:
    metric_ids = _normalize_constraint_metric_ids(constraint_metrics)
    label = _format_constraint_metrics_label(metric_ids)
    slug = _constraint_metrics_slug(metric_ids)
    metric_text = ",".join(metric_ids)

    summary_row = dict(result.get("selected_summary_row", {}))
    if summary_row:
        summary_row["constraint_metrics"] = metric_text
        summary_row["constraint_metrics_label"] = label
        summary_row["constraint_metrics_slug"] = slug
        result["selected_summary_row"] = summary_row

    fold_rows: List[Dict[str, Any]] = []
    for row in result.get("selected_fold_rows", []):
        out = dict(row)
        out["constraint_metrics"] = metric_text
        out["constraint_metrics_label"] = label
        out["constraint_metrics_slug"] = slug
        fold_rows.append(out)
    result["selected_fold_rows"] = fold_rows
    return result


def _validate_fold_aggregation(name: str, value: str) -> None:
    if value not in {"average_fold", "worst_fold"}:
        raise ValueError(f"{name} must be one of: average_fold, worst_fold")


def _resolve_constraint_aggregation(
    metric_id: str,
    *,
    constraint_aggregation: str,
    level_constraint_aggregation: Optional[str] = None,
    dispersion_constraint_aggregation: Optional[str] = None,
) -> str:
    metric_key = str(metric_id).upper()
    if metric_key in _LEVEL_CONSTRAINT_METRICS and level_constraint_aggregation is not None:
        return str(level_constraint_aggregation)
    if metric_key in _DISPERSION_CONSTRAINT_METRICS and dispersion_constraint_aggregation is not None:
        return str(dispersion_constraint_aggregation)
    return str(constraint_aggregation)


def _constraint_aggregation_fields(
    *,
    constraint_aggregation: str,
    level_constraint_aggregation: Optional[str] = None,
    dispersion_constraint_aggregation: Optional[str] = None,
) -> Dict[str, str]:
    return {
        "constraint_aggregation": str(constraint_aggregation),
        "level_constraint_aggregation": (
            str(level_constraint_aggregation) if level_constraint_aggregation is not None else str(constraint_aggregation)
        ),
        "dispersion_constraint_aggregation": (
            str(dispersion_constraint_aggregation)
            if dispersion_constraint_aggregation is not None
            else str(constraint_aggregation)
        ),
    }


def _base_model_name(model_name: Any) -> str:
    return str(model_name).split("(", 1)[0].strip()


def _load_runs_df(
    *,
    result_root: str,
    data_id: str,
    split_id: str,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    runs_dir = Path(result_root) / "runs" / f"data_id={data_id}" / f"split_id={split_id}"
    if not runs_dir.exists():
        raise FileNotFoundError(f"CV runs directory not found: {runs_dir}")
    paths = sorted(runs_dir.rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No run parquet files found under: {runs_dir}")
    read_columns = None if columns is None else list(dict.fromkeys(columns))
    dfs = [pd.read_parquet(path, columns=read_columns) for path in paths]
    return pd.concat(dfs, ignore_index=True)


def _load_tested_config_ids(*, result_root: str, data_id: str, split_id: str) -> Set[str]:
    test_metrics_path = (
        Path(result_root) / "analysis" / f"data_id={data_id}" / f"split_id={split_id}" / "test_metrics.csv"
    )
    if not test_metrics_path.exists():
        return set()
    try:
        test_df = pd.read_csv(test_metrics_path, usecols=["config_id"])
    except pd.errors.EmptyDataError:
        return set()
    if "config_id" not in test_df.columns:
        raise KeyError(f"Missing required column in held-out test metrics: config_id ({test_metrics_path})")
    return set(test_df["config_id"].dropna().astype(str).tolist())


def _align_complete_grid(
    df: pd.DataFrame,
    *,
    required_metrics: List[str],
    fold_col: str = "fold_id",
    model_col: str = "config_id",
) -> Tuple[pd.DataFrame, List[int], List[str]]:
    """
    Keep the largest fold x model grid that is fully observed for the requested metrics.
    """
    base_cols = [fold_col, model_col]
    for col in base_cols + required_metrics:
        if col not in df.columns:
            raise KeyError(f"Missing required column in runs_df: {col}")

    keep_cols = base_cols + required_metrics + ["model_name"]
    if "run_id" in df.columns:
        keep_cols.append("run_id")
    if "model_family" in df.columns:
        keep_cols.append("model_family")
    if "model_config_json" in df.columns:
        keep_cols.append("model_config_json")
    dfx = df.loc[:, list(dict.fromkeys(keep_cols))].copy()
    dfx = dfx.dropna(subset=required_metrics)
    dfx = dfx.drop_duplicates(subset=base_cols, keep="first")

    present = (
        dfx.assign(_present=1)
        .pivot_table(index=fold_col, columns=model_col, values="_present", aggfunc="max", fill_value=0)
    )
    folds_keep = present.index[present.sum(axis=1) == present.shape[1]].tolist()
    models_keep = present.columns[present.sum(axis=0) == present.shape[0]].tolist()

    if not folds_keep or not models_keep:
        folds_keep = sorted(dfx[fold_col].unique().tolist())
        models_keep = sorted(dfx[model_col].astype(str).unique().tolist())

    dfx = dfx[dfx[fold_col].isin(folds_keep) & dfx[model_col].astype(str).isin([str(m) for m in models_keep])].copy()
    dfx[model_col] = dfx[model_col].astype(str)
    dfx["model_name"] = dfx["model_name"].astype(str)
    if "model_family" in dfx.columns:
        dfx["model_family"] = dfx["model_family"].astype(str)
    return dfx, [int(x) for x in sorted(folds_keep)], [str(x) for x in sorted(models_keep)]


def _normalize_runs_df(runs_df: pd.DataFrame) -> pd.DataFrame:
    dfx = runs_df.copy()
    if "fold_id" not in dfx.columns:
        raise KeyError("Missing required column: fold_id")
    if "config_id" not in dfx.columns:
        raise KeyError("Missing required column: config_id")
    if "model_name" not in dfx.columns:
        raise KeyError("Missing required column: model_name")

    dfx["fold_id"] = pd.to_numeric(dfx["fold_id"], errors="coerce")
    dfx = dfx.dropna(subset=["fold_id"]).copy()
    dfx["fold_id"] = dfx["fold_id"].astype(int)
    dfx["config_id"] = dfx["config_id"].astype(str)
    dfx["model_name"] = dfx["model_name"].astype(str)
    dfx["model_family"] = dfx["model_name"].map(_base_model_name)
    return dfx


def _bounds_label(spec: ConstraintSpec) -> str:
    lo = spec.lower
    hi = spec.upper
    if lo is not None and hi is not None:
        return f"[{lo}, {hi}]"
    if lo is not None:
        return f"[{lo}, +inf)"
    if hi is not None:
        return f"(-inf, {hi}]"
    return "unbounded"


def _is_within_bounds(value: float, *, lower: Optional[float], upper: Optional[float]) -> bool:
    if not np.isfinite(value):
        return False
    if lower is not None and value < lower:
        return False
    if upper is not None and value > upper:
        return False
    return True


def _constraint_violation(value: float, *, lower: Optional[float], upper: Optional[float]) -> float:
    if not np.isfinite(value):
        return float("inf")
    violation = 0.0
    if lower is not None and value < lower:
        violation = max(violation, float(lower - value))
    if upper is not None and value > upper:
        violation = max(violation, float(value - upper))
    return float(violation)


def _constraint_margin(value: float, *, lower: Optional[float], upper: Optional[float]) -> float:
    if not np.isfinite(value):
        return -float("inf")
    margins: List[float] = []
    if lower is not None:
        margins.append(float(value - lower))
    if upper is not None:
        margins.append(float(upper - value))
    if not margins:
        return float("inf")
    return float(min(margins))


def _evaluate_constraint(values: np.ndarray, spec: ConstraintSpec, aggregation: str) -> Dict[str, Any]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {
            "passes": False,
            "aggregation_value": np.nan,
            "worst_violation": np.nan,
        }

    if aggregation == "average_fold":
        agg_val = float(np.mean(vals))
        passes = _is_within_bounds(agg_val, lower=spec.lower, upper=spec.upper)
        worst_violation = float(
            np.max([_constraint_violation(v, lower=spec.lower, upper=spec.upper) for v in vals])
        )
        return {
            "passes": bool(passes),
            "aggregation_value": agg_val,
            "worst_violation": worst_violation,
        }

    if aggregation != "worst_fold":
        raise ValueError("aggregation must be one of: average_fold, worst_fold")

    violations = np.asarray(
        [_constraint_violation(v, lower=spec.lower, upper=spec.upper) for v in vals],
        dtype=float,
    )
    margins = np.asarray(
        [_constraint_margin(v, lower=spec.lower, upper=spec.upper) for v in vals],
        dtype=float,
    )
    if np.any(violations > 0.0):
        idx = int(np.argmax(violations))
    else:
        idx = int(np.argmin(margins))
    return {
        "passes": bool(np.max(violations) <= 0.0),
        "aggregation_value": float(vals[idx]),
        "worst_violation": float(np.max(violations)),
    }


def _aggregate_selection_metric(values: np.ndarray, spec: SelectionMetricSpec, aggregation: str) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    if aggregation == "average_fold":
        return float(np.mean(vals))
    if aggregation != "worst_fold":
        raise ValueError("aggregation must be one of: average_fold, worst_fold")
    if spec.higher_is_better:
        return float(np.min(vals))
    return float(np.max(vals))


def _transform_selection_metric_values(values: np.ndarray, spec: SelectionMetricSpec) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    if spec.transform == "identity":
        return vals
    if spec.transform == "square":
        return np.square(vals)
    raise ValueError(f"Unsupported selection metric transform: {spec.transform}")


def _selection_metric_values_from_df(row_df: pd.DataFrame, spec: SelectionMetricSpec) -> np.ndarray:
    vals = pd.to_numeric(row_df[spec.column], errors="coerce").to_numpy(dtype=float)
    return _transform_selection_metric_values(vals, spec)


def _metric_std(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    return float(np.std(vals, ddof=0))


def _constraint_target(spec: ConstraintSpec) -> float:
    if spec.lower is not None and spec.upper is not None:
        if float(spec.lower) <= 0.0 <= float(spec.upper):
            return 0.0
        if float(spec.lower) <= 1.0 <= float(spec.upper):
            return 1.0
        return float(0.5 * (float(spec.lower) + float(spec.upper)))
    if spec.lower is not None:
        return float(spec.lower)
    if spec.upper is not None:
        return float(spec.upper)
    return np.nan


def _constraint_preference_value(value: float, spec: ConstraintSpec) -> float:
    if not np.isfinite(value):
        return np.nan
    if spec.lower is not None and spec.upper is not None:
        target = _constraint_target(spec)
        return float(-abs(float(value) - float(target)))
    if spec.upper is not None:
        return float(-float(value))
    if spec.lower is not None:
        return float(value)
    return float(value)


def _positive_selection_utility(value: float, spec: SelectionMetricSpec) -> float:
    if not np.isfinite(value):
        return np.nan
    val = float(value)
    if spec.higher_is_better:
        if spec.column in {"R2", "R2 (log)"}:
            return float(max(1.0 + val, _POSITIVE_EPS))
        return float(max(val, _POSITIVE_EPS))
    return float(1.0 / max(val, _POSITIVE_EPS))


def _positive_constraint_utility(value: float, spec: ConstraintSpec) -> float:
    if not np.isfinite(value):
        return np.nan
    val = float(value)
    target = _constraint_target(spec)

    if spec.lower is not None and spec.upper is not None:
        if np.isfinite(target) and abs(float(target)) > _POSITIVE_EPS and val > _POSITIVE_EPS:
            return float(min(val, float(target)) / max(val, float(target)))
        span = max(abs(float(spec.lower)), abs(float(spec.upper)), 1.0)
        return float(1.0 / (1.0 + abs(val - float(target)) / span))

    if spec.upper is not None:
        scale = max(abs(float(spec.upper)), 1.0)
        return float(1.0 / (1.0 + max(val, 0.0) / scale))

    if spec.lower is not None:
        scale = max(abs(float(spec.lower)), 1.0)
        if val <= 0.0:
            return _POSITIVE_EPS
        return float(1.0 / (1.0 + max(float(spec.lower) - val, 0.0) / scale))

    return float(max(val, _POSITIVE_EPS))


def _normalize_preference(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=float).reshape(-1)
    out = np.full(vals.shape, np.nan, dtype=float)
    finite_mask = np.isfinite(vals)
    if not np.any(finite_mask):
        return out
    finite_vals = vals[finite_mask]
    best = float(np.max(finite_vals))
    worst = float(np.min(finite_vals))
    denom = float(best - worst)
    if abs(denom) <= 1e-12:
        out[finite_mask] = 1.0
        return out
    out[finite_mask] = (finite_vals - worst) / denom
    return np.clip(out, 0.0, 1.0)


def _best_index(values: pd.Series, *, higher_is_better: bool) -> Optional[int]:
    vals = pd.to_numeric(values, errors="coerce")
    if vals.notna().sum() == 0:
        return None
    return int(vals.idxmax()) if higher_is_better else int(vals.idxmin())


def _select_best_candidate(
    stats_df: pd.DataFrame,
    *,
    selection_spec: SelectionMetricSpec,
) -> pd.Series:
    if stats_df.empty:
        raise RuntimeError("No candidate rows available for selection.")

    primary = pd.to_numeric(stats_df["selection_value"], errors="coerce")
    if primary.notna().sum() == 0:
        raise RuntimeError("No finite selection values available.")

    best_primary = float(primary.max()) if selection_spec.higher_is_better else float(primary.min())
    if selection_spec.higher_is_better:
        tie_mask = primary >= (best_primary - _TIE_TOL)
    else:
        tie_mask = primary <= (best_primary + _TIE_TOL)

    tied = stats_df.loc[tie_mask, :].copy()
    tied["selection_std"] = pd.to_numeric(tied["selection_std"], errors="coerce")
    tied = tied.sort_values(
        by=["selection_std", "config_id"],
        ascending=[True, True],
        na_position="last",
        ignore_index=True,
    )
    return tied.iloc[0]


def _select_utopia_candidate(
    stats_df: pd.DataFrame,
    *,
    constraint_metrics: Sequence[str],
    selection_spec: SelectionMetricSpec,
) -> pd.Series:
    if stats_df.empty:
        raise RuntimeError("No candidate rows available for utopia selection.")

    ranked = stats_df.copy()
    selection_vals = pd.to_numeric(ranked["selection_value"], errors="coerce").to_numpy(dtype=float)
    selection_pref = selection_vals if selection_spec.higher_is_better else -selection_vals
    ranked["utopia_selection_score"] = _normalize_preference(selection_pref)

    score_cols = ["utopia_selection_score"]
    for metric_id in constraint_metrics:
        spec = _CONSTRAINT_SPECS[metric_id]
        value_col = f"constraint_{metric_id}_value"
        metric_vals = pd.to_numeric(ranked.get(value_col), errors="coerce").to_numpy(dtype=float)
        metric_pref = np.asarray([_constraint_preference_value(v, spec) for v in metric_vals], dtype=float)
        score_col = f"utopia_{metric_id}_score"
        ranked[score_col] = _normalize_preference(metric_pref)
        score_cols.append(score_col)

    score_mat = ranked.loc[:, score_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if score_mat.size == 0:
        raise RuntimeError("No finite utopia scores available.")
    ranked["utopia_distance"] = np.sqrt(np.nanmean((1.0 - score_mat) ** 2, axis=1))
    ranked["selection_std"] = pd.to_numeric(ranked["selection_std"], errors="coerce")
    ranked = ranked.sort_values(
        by=["utopia_distance", "selection_std", "config_id"],
        ascending=[True, True, True],
        na_position="last",
        ignore_index=True,
    )
    if ranked.empty or not np.isfinite(pd.to_numeric(ranked["utopia_distance"], errors="coerce")).any():
        raise RuntimeError("Could not compute a finite utopia distance for any candidate.")
    return ranked.iloc[0]


def _select_nash_candidate(
    stats_df: pd.DataFrame,
    *,
    constraint_metrics: Sequence[str],
    selection_spec: SelectionMetricSpec,
) -> pd.Series:
    if stats_df.empty:
        raise RuntimeError("No candidate rows available for nash selection.")

    ranked = stats_df.copy()
    selection_vals = pd.to_numeric(ranked["selection_value"], errors="coerce").to_numpy(dtype=float)
    selection_utils = np.asarray(
        [_positive_selection_utility(v, selection_spec) for v in selection_vals],
        dtype=float,
    )
    ranked["nash_selection_utility"] = selection_utils

    utility_cols = ["nash_selection_utility"]
    for metric_id in constraint_metrics:
        spec = _CONSTRAINT_SPECS[metric_id]
        value_col = f"constraint_{metric_id}_value"
        metric_vals = pd.to_numeric(ranked.get(value_col), errors="coerce").to_numpy(dtype=float)
        utility_col = f"nash_{metric_id}_utility"
        ranked[utility_col] = np.asarray(
            [_positive_constraint_utility(v, spec) for v in metric_vals],
            dtype=float,
        )
        utility_cols.append(utility_col)

    utility_mat = ranked.loc[:, utility_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if utility_mat.size == 0:
        raise RuntimeError("No finite nash utilities available.")
    positive_mask = np.all(np.isfinite(utility_mat) & (utility_mat > 0.0), axis=1)
    ranked["nash_log_utility"] = np.nan
    if np.any(positive_mask):
        ranked.loc[positive_mask, "nash_log_utility"] = np.sum(np.log(utility_mat[positive_mask]), axis=1)

    ranked["selection_std"] = pd.to_numeric(ranked["selection_std"], errors="coerce")
    ranked = ranked.sort_values(
        by=["nash_log_utility", "selection_std", "config_id"],
        ascending=[False, True, True],
        na_position="last",
        ignore_index=True,
    )
    if ranked.empty or not np.isfinite(pd.to_numeric(ranked["nash_log_utility"], errors="coerce")).any():
        raise RuntimeError("Could not compute a finite nash log-utility for any candidate.")
    return ranked.iloc[0]


def _build_metric_summary(row_df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for col, alias in _SUMMARY_METRIC_ALIASES.items():
        if col not in row_df.columns:
            continue
        vals = pd.to_numeric(row_df[col], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            out[f"{alias}_mean"] = np.nan
            out[f"{alias}_worst"] = np.nan
            out[f"{alias}_std"] = np.nan
            continue
        out[f"{alias}_mean"] = float(np.mean(vals))
        out[f"{alias}_std"] = float(np.std(vals, ddof=0))
        if col in {"R2", "OOS R2", "R2 (log)"}:
            out[f"{alias}_worst"] = float(np.min(vals))
        else:
            out[f"{alias}_worst"] = float(np.max(vals))
    if "RMSE" in row_df.columns and "MSE_mean" not in out:
        mse_vals = np.square(pd.to_numeric(row_df["RMSE"], errors="coerce").to_numpy(dtype=float))
        mse_vals = mse_vals[np.isfinite(mse_vals)]
        if mse_vals.size == 0:
            out["MSE_mean"] = np.nan
            out["MSE_worst"] = np.nan
            out["MSE_std"] = np.nan
        else:
            out["MSE_mean"] = float(np.mean(mse_vals))
            out["MSE_std"] = float(np.std(mse_vals, ddof=0))
            out["MSE_worst"] = float(np.max(mse_vals))
    return out


def _extract_model_config_json(row_df: pd.DataFrame) -> str:
    if "model_config_json" not in row_df.columns:
        return ""
    vals = row_df["model_config_json"].dropna().astype(str)
    return "" if vals.empty else str(vals.iloc[0])


def _parse_model_config_json(cfg_raw: Any) -> Dict[str, Any]:
    if not isinstance(cfg_raw, str) or not cfg_raw.strip():
        return {}
    try:
        payload = json.loads(cfg_raw)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _extract_ratio_mode_from_model_config_json(cfg_raw: Any) -> str:
    payload = _parse_model_config_json(cfg_raw)
    mode = str(payload.get("ratio_mode", "")).strip().lower()
    return mode if mode in {"div", "diff"} else ""


def _resolve_prediction_paths(
    *,
    preds_dir: Path,
    selected_runs_by_fold: Dict[int, List[str]],
) -> List[Path]:
    selected_paths: List[Path] = []
    for fold_id, run_ids in selected_runs_by_fold.items():
        for run_id in run_ids:
            path = preds_dir / f"fold_id={int(fold_id)}" / f"{str(run_id)}.parquet"
            if path.exists():
                selected_paths.append(path)
    return sorted(set(selected_paths))


def _load_predictions_df(
    *,
    result_root: str,
    data_id: str,
    split_id: str,
    selected_runs_by_fold: Dict[int, List[str]],
) -> pd.DataFrame:
    preds_dir = Path(result_root) / "predictions" / f"data_id={data_id}" / f"split_id={split_id}"
    if not preds_dir.exists():
        raise FileNotFoundError(f"Predictions directory not found: {preds_dir}")
    paths = _resolve_prediction_paths(preds_dir=preds_dir, selected_runs_by_fold=selected_runs_by_fold)
    if not paths:
        raise FileNotFoundError(f"No prediction parquet files found for requested runs under: {preds_dir}")
    dfs = [pd.read_parquet(path, columns=["run_id", "row_id", "y_true", "y_pred"]) for path in paths]
    out = pd.concat(dfs, ignore_index=True)
    out["run_id"] = out["run_id"].astype(str)
    out["row_id"] = pd.to_numeric(out["row_id"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["row_id", "y_true", "y_pred"]).copy()
    out["row_id"] = out["row_id"].astype(np.int64)
    return out


def _validate_convex_stacking_request(
    *,
    selection_metric_id: str,
    constraint_metrics: Sequence[str],
) -> None:
    if selection_metric_id not in _CONVEX_STACKING_SELECTION_METRICS:
        raise ValueError(
            "Convex stacking only supports selection metrics "
            f"{sorted(_CONVEX_STACKING_SELECTION_METRICS)}; got '{selection_metric_id}'."
        )
    unsupported = [str(metric_id) for metric_id in constraint_metrics if metric_id not in _CONVEX_STACKING_CONSTRAINT_METRICS]
    if unsupported:
        raise ValueError(
            "Convex stacking only supports constraint metrics "
            f"{sorted(_CONVEX_STACKING_CONSTRAINT_METRICS)}; got unsupported {unsupported}."
        )


def _stacking_fold_loss_value(y_true: np.ndarray, y_pred: np.ndarray, *, selection_metric_id: str) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    y_pos = np.maximum(y_true, 1e-9)
    resid = y_pred - y_true
    if selection_metric_id == "MSE":
        return float(np.mean(np.square(resid)))
    if selection_metric_id == "RMSE":
        return float(np.sqrt(np.mean(np.square(resid))))
    if selection_metric_id == "MAE":
        return float(np.mean(np.abs(resid)))
    if selection_metric_id == "MAPE":
        return float(np.mean(np.abs(resid) / y_pos))
    raise ValueError(f"Unsupported convex stacking selection metric: {selection_metric_id}")


def _choose_convex_stacking_solver(cp_mod: Any, solver: Optional[str]) -> str:
    installed = set(cp_mod.installed_solvers())
    if solver is not None:
        solver_name = str(solver).strip().upper()
        if solver_name not in installed:
            raise ValueError(f"Requested cvxpy solver '{solver_name}' is not installed. Installed: {sorted(installed)}")
        return solver_name
    for candidate in ["ECOS", "OSQP", "SCS", "CLARABEL"]:
        if candidate in installed:
            return candidate
    raise RuntimeError("No supported cvxpy solver installed. Install one of: ecos, osqp, scs, clarabel.")


def _apply_stacking_weight_threshold(weights: np.ndarray, *, min_weight: float) -> np.ndarray:
    weights_arr = np.asarray(weights, dtype=float).reshape(-1)
    if weights_arr.size == 0:
        raise ValueError("Stacking weights cannot be empty.")
    if not np.all(np.isfinite(weights_arr)):
        raise ValueError("Stacking weights must be finite.")
    weights_arr = np.maximum(weights_arr, 0.0)
    total = float(np.sum(weights_arr))
    if total <= 0.0:
        raise ValueError("Stacking weights must sum to a positive value before thresholding.")
    weights_arr = weights_arr / total

    floor = float(min_weight)
    if floor <= 0.0:
        return weights_arr

    keep_mask = weights_arr >= floor
    if not np.any(keep_mask):
        keep_mask[int(np.argmax(weights_arr))] = True
    pruned = np.where(keep_mask, weights_arr, 0.0)
    pruned_total = float(np.sum(pruned))
    if pruned_total <= 0.0:
        raise ValueError("Stacking weights sum to zero after thresholding.")
    return pruned / pruned_total


def _pivot_metric_matrix(
    df: pd.DataFrame,
    *,
    metric: str,
    folds: Sequence[int],
    models: Sequence[str],
) -> np.ndarray:
    pvt = df.pivot_table(index="fold_id", columns="config_id", values=str(metric), aggfunc="first")
    pvt = pvt.reindex(index=[int(f) for f in folds], columns=[str(m) for m in models])
    if pvt.isna().any().any():
        missing = int(pvt.isna().sum().sum())
        raise ValueError(f"Missing values after alignment for metric '{metric}': {missing}")
    return pvt.to_numpy(dtype=float)


def _build_linearized_metric_matrices(
    family_df: pd.DataFrame,
    *,
    folds: Sequence[int],
    models: Sequence[str],
) -> Dict[str, np.ndarray]:
    matrices: Dict[str, np.ndarray] = {}
    metric_cols = [str(col) for col in _SUMMARY_METRIC_ALIASES.keys() if str(col) != "MSE" and str(col) in family_df.columns]
    for col in metric_cols:
        matrices[col] = _pivot_metric_matrix(family_df, metric=col, folds=folds, models=models)
    if "RMSE" in matrices:
        matrices["MSE"] = np.square(np.asarray(matrices["RMSE"], dtype=float))
    return matrices


def _empty_linearized_stacking_result(
    *,
    family_name: str,
    status: str,
    selection_metric_id: str,
    selection_aggregation: str,
    constraint_metrics: Sequence[str],
    constraint_aggregation: str,
    level_constraint_aggregation: Optional[str] = None,
    dispersion_constraint_aggregation: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "status": str(status),
        "family_name": str(family_name),
        "selected_summary_row": {
            "row_kind": "stacking_summary",
            "comparison_group": str(family_name),
            "selection_method": "linearized_stacking",
            "status": str(status),
            "config_id": "",
            "model_name": f"{family_name}_LINEARIZED_STACKING",
            "model_family": str(family_name),
            "fold_id": np.nan,
            "n_folds_used": 0,
            "selection_metric": str(selection_metric_id),
            "selection_aggregation": str(selection_aggregation),
            "selection_value": np.nan,
            "selection_std": np.nan,
            "constraint_metrics": ",".join(constraint_metrics),
            **_constraint_aggregation_fields(
                constraint_aggregation=constraint_aggregation,
                level_constraint_aggregation=level_constraint_aggregation,
                dispersion_constraint_aggregation=dispersion_constraint_aggregation,
            ),
            "utopia_metric_aggregation": "",
            "utopia_distance": np.nan,
            "nash_log_utility": np.nan,
            "n_candidate_models": 0,
            "n_feasible_models": np.nan,
            "stacking_weights_csv": "",
            "stacking_solution_name": "",
            "stacking_problem_status": "",
            "stacking_solver": "",
            "stacking_n_active_models": 0,
            "stacking_total_gap": np.nan,
        },
        "selected_fold_rows": [],
    }


def _select_linearized_stacking_for_family(
    *,
    runs_df: pd.DataFrame,
    output_dir: Path,
    family_name: str,
    eligible_config_ids: Optional[Set[str]],
    constraint_metrics: Sequence[str],
    constraint_aggregation: str,
    selection_metric_id: str,
    selection_aggregation: str,
    solver: Optional[str],
    weight_min_threshold: float,
    artifact_suffix: str = "",
    level_constraint_aggregation: Optional[str] = None,
    dispersion_constraint_aggregation: Optional[str] = None,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_family_df = _candidate_pool_runs_df(runs_df, group_name=family_name)
    if raw_family_df.empty:
        return _empty_linearized_stacking_result(
            family_name=family_name,
            status="family_not_found",
            selection_metric_id=selection_metric_id,
            selection_aggregation=selection_aggregation,
            constraint_metrics=constraint_metrics,
            constraint_aggregation=constraint_aggregation,
            level_constraint_aggregation=level_constraint_aggregation,
            dispersion_constraint_aggregation=dispersion_constraint_aggregation,
        )
    if eligible_config_ids is not None:
        raw_family_df = raw_family_df.loc[raw_family_df["config_id"].isin(eligible_config_ids), :].copy()
        if raw_family_df.empty:
            return _empty_linearized_stacking_result(
                family_name=family_name,
                status="no_test_metrics",
                selection_metric_id=selection_metric_id,
                selection_aggregation=selection_aggregation,
                constraint_metrics=constraint_metrics,
                constraint_aggregation=constraint_aggregation,
                level_constraint_aggregation=level_constraint_aggregation,
                dispersion_constraint_aggregation=dispersion_constraint_aggregation,
            )

    selection_spec = _SELECTION_METRIC_SPECS[selection_metric_id]
    needed_cols = [str(selection_spec.column)]
    needed_cols.extend(str(_CONSTRAINT_SPECS[metric_id].column) for metric_id in constraint_metrics)
    grid_df, folds, models = _align_complete_grid(raw_family_df, required_metrics=list(needed_cols))
    if grid_df.empty:
        return _empty_linearized_stacking_result(
            family_name=family_name,
            status="no_complete_candidates",
            selection_metric_id=selection_metric_id,
            selection_aggregation=selection_aggregation,
            constraint_metrics=constraint_metrics,
            constraint_aggregation=constraint_aggregation,
            level_constraint_aggregation=level_constraint_aggregation,
            dispersion_constraint_aggregation=dispersion_constraint_aggregation,
        )

    family_df = raw_family_df.loc[
        raw_family_df["fold_id"].isin(folds) & raw_family_df["config_id"].astype(str).isin([str(m) for m in models]),
        :,
    ].copy()
    family_df = family_df.drop_duplicates(subset=["fold_id", "config_id"], keep="first")
    metric_mats = _build_linearized_metric_matrices(family_df, folds=folds, models=models)
    if selection_metric_id not in metric_mats:
        raise KeyError(f"Missing linearized stacking metric matrix for selection metric '{selection_metric_id}'.")

    try:
        import cvxpy as cp
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Linearized stacking in simple_model_selection.py requires cvxpy.") from exc

    n_models = len(models)
    n_folds = len(folds)
    w = cp.Variable(n_models, nonneg=True)
    base_constraints = [cp.sum(w) == 1]
    gap_vars = {str(metric_id): cp.Variable(nonneg=True) for metric_id in constraint_metrics}

    sel_mat = np.asarray(metric_mats[selection_metric_id], dtype=float)
    fold_selection_exprs = [sel_mat[i, :] @ w for i in range(n_folds)]
    if selection_aggregation == "average_fold":
        selection_objective_expr = cp.sum(cp.hstack(fold_selection_exprs)) / float(max(1, n_folds))
    elif selection_aggregation == "worst_fold":
        t = cp.Variable()
        if selection_spec.higher_is_better:
            base_constraints.extend(expr >= t for expr in fold_selection_exprs)
        else:
            base_constraints.extend(expr <= t for expr in fold_selection_exprs)
        selection_objective_expr = t
    else:
        raise ValueError("selection_aggregation must be one of: average_fold, worst_fold")

    for metric_id in constraint_metrics:
        metric_col = str(_CONSTRAINT_SPECS[metric_id].column)
        metric_mat = metric_mats.get(metric_id)
        if metric_mat is None:
            metric_mat = metric_mats.get(metric_col)
        if metric_mat is None:
            raise KeyError(f"Missing linearized stacking metric matrix for constraint metric '{metric_id}'.")
        spec = _CONSTRAINT_SPECS[metric_id]
        fold_exprs = [np.asarray(metric_mat, dtype=float)[i, :] @ w for i in range(n_folds)]
        metric_constraint_aggregation = _resolve_constraint_aggregation(
            metric_id,
            constraint_aggregation=constraint_aggregation,
            level_constraint_aggregation=level_constraint_aggregation,
            dispersion_constraint_aggregation=dispersion_constraint_aggregation,
        )
        if metric_constraint_aggregation == "worst_fold":
            for expr in fold_exprs:
                if spec.lower is not None:
                    base_constraints.append(expr >= float(spec.lower) - gap_vars[metric_id])
                if spec.upper is not None:
                    base_constraints.append(expr <= float(spec.upper) + gap_vars[metric_id])
        else:
            avg_expr = cp.sum(cp.hstack(fold_exprs)) / float(max(1, len(fold_exprs)))
            if spec.lower is not None:
                base_constraints.append(avg_expr >= float(spec.lower) - gap_vars[metric_id])
            if spec.upper is not None:
                base_constraints.append(avg_expr <= float(spec.upper) + gap_vars[metric_id])

    solver_name = _choose_convex_stacking_solver(cp, solver)
    total_gap_expr = cp.sum(cp.hstack([gap_vars[mid] for mid in constraint_metrics])) if gap_vars else 0.0
    gap_prob = cp.Problem(cp.Minimize(total_gap_expr), list(base_constraints))
    gap_prob.solve(solver=solver_name, verbose=False)
    if gap_prob.status not in {"optimal", "optimal_inaccurate"} or w.value is None:
        raise RuntimeError(f"Linearized stacking feasibility optimization failed for {family_name}. status={gap_prob.status}")
    gap_values = {metric_id: float(gap_vars[metric_id].value or 0.0) for metric_id in constraint_metrics}
    weights_raw = np.asarray(w.value, dtype=float).reshape(-1)
    gap_tol = 1e-6
    objective_constraints = list(base_constraints)
    for metric_id in constraint_metrics:
        objective_constraints.append(
            gap_vars[metric_id] <= gap_values[metric_id] + max(gap_tol, 1e-4 * max(1.0, gap_values[metric_id]))
        )
    if selection_spec.higher_is_better:
        prob = cp.Problem(cp.Maximize(selection_objective_expr), objective_constraints)
    else:
        prob = cp.Problem(cp.Minimize(selection_objective_expr), objective_constraints)
    prob.solve(solver=solver_name, verbose=False)
    if prob.status in {"optimal", "optimal_inaccurate"} and w.value is not None:
        weights_raw = np.asarray(w.value, dtype=float).reshape(-1)
    else:
        prob = gap_prob

    selected_status = (
        "selected"
        if max([float(val) for val in gap_values.values()], default=0.0) <= gap_tol
        else "selected_closest_infeasible"
    )
    weights = np.asarray(weights_raw, dtype=float).reshape(-1)
    if not np.all(np.isfinite(weights)):
        raise RuntimeError(f"Linearized stacking produced non-finite weights for {family_name}.")
    weights = _apply_stacking_weight_threshold(weights, min_weight=weight_min_threshold)

    family_configs = family_df.loc[
        :,
        [c for c in ["config_id", "model_name", "model_config_json", "model_family"] if c in family_df.columns],
    ]
    family_configs = family_configs.drop_duplicates("config_id").copy()
    weights_df = pd.DataFrame({"config_id": [str(m) for m in models], "weight": weights}).merge(
        family_configs,
        on="config_id",
        how="left",
    )
    weights_df = weights_df.sort_values("weight", ascending=False, ignore_index=True)

    fold_metric_rows: List[Dict[str, Any]] = []
    for idx, fold_id in enumerate(folds):
        row: Dict[str, Any] = {"fold_id": int(fold_id)}
        for metric_name, metric_mat in metric_mats.items():
            row[str(metric_name)] = float(np.asarray(metric_mat, dtype=float)[idx, :] @ weights)
        if "MSE" not in row and "RMSE" in row and np.isfinite(float(row["RMSE"])):
            row["MSE"] = float(float(row["RMSE"]) ** 2)
        row["selection_value"] = float(np.asarray(sel_mat, dtype=float)[idx, :] @ weights)
        fold_metric_rows.append(row)
    fold_metrics_df = pd.DataFrame(fold_metric_rows).sort_values("fold_id", kind="mergesort").reset_index(drop=True)

    selection_vals = pd.to_numeric(fold_metrics_df["selection_value"], errors="coerce").to_numpy(dtype=float)
    selected_row = pd.Series(
        {
            "selection_value": _aggregate_selection_metric(selection_vals, selection_spec, selection_aggregation),
            "selection_std": _metric_std(selection_vals),
        }
    )

    suffix = f"_{str(artifact_suffix).strip()}" if str(artifact_suffix).strip() else ""
    solution_name = f"{family_name}_LINEARIZED_STACKING{suffix.upper()}"
    weights_csv = output_dir / f"weights_stacking_linearized_{family_name}{suffix}.csv"
    weights_df.to_csv(weights_csv, index=False)

    summary_row: Dict[str, Any] = {
        "row_kind": "stacking_summary",
        "comparison_group": str(family_name),
        "selection_method": "linearized_stacking",
        "status": selected_status,
        "config_id": "",
        "model_name": solution_name,
        "model_family": str(family_name),
        "fold_id": np.nan,
        "n_folds_used": int(len(folds)),
        "selection_metric": str(selection_metric_id),
        "selection_aggregation": str(selection_aggregation),
        "selection_value": float(selected_row["selection_value"]),
        "selection_std": float(selected_row["selection_std"]),
        "constraint_metrics": ",".join(constraint_metrics),
        **_constraint_aggregation_fields(
            constraint_aggregation=constraint_aggregation,
            level_constraint_aggregation=level_constraint_aggregation,
            dispersion_constraint_aggregation=dispersion_constraint_aggregation,
        ),
        "utopia_metric_aggregation": "",
        "utopia_distance": np.nan,
        "nash_log_utility": np.nan,
        "model_config_json": "",
        "n_candidate_models": int(len(models)),
        "n_feasible_models": np.nan,
        "stacking_weights_csv": str(weights_csv),
        "stacking_solution_name": solution_name,
        "stacking_problem_status": str(prob.status),
        "stacking_solver": str(solver_name),
        "stacking_n_active_models": int((weights_df["weight"] > 1e-8).sum()),
        "stacking_total_gap": float(sum(gap_values.values())),
        "stacking_weight_min_threshold": float(weight_min_threshold),
    }
    summary_row.update(_build_metric_summary(fold_metrics_df))
    if constraint_metrics:
        summary_row.update(
            _build_constraint_columns(
                fold_metrics_df,
                constraint_metrics=constraint_metrics,
                constraint_aggregation=constraint_aggregation,
                level_constraint_aggregation=level_constraint_aggregation,
                dispersion_constraint_aggregation=dispersion_constraint_aggregation,
            )
        )

    fold_rows: List[Dict[str, Any]] = []
    for _, row in fold_metrics_df.iterrows():
        out: Dict[str, Any] = {
            "row_kind": "stacking_fold",
            "comparison_group": str(family_name),
            "selection_method": "linearized_stacking",
            "status": selected_status,
            "config_id": "",
            "model_name": solution_name,
            "model_family": str(family_name),
            "fold_id": int(row["fold_id"]),
            "n_folds_used": int(len(folds)),
            "selection_metric": str(selection_metric_id),
            "selection_aggregation": str(selection_aggregation),
            "selection_value": float(pd.to_numeric(row.get("selection_value"), errors="coerce")),
            "selection_std": np.nan,
            "constraint_metrics": ",".join(constraint_metrics),
            **_constraint_aggregation_fields(
                constraint_aggregation=constraint_aggregation,
                level_constraint_aggregation=level_constraint_aggregation,
                dispersion_constraint_aggregation=dispersion_constraint_aggregation,
            ),
            "utopia_metric_aggregation": "",
            "utopia_distance": np.nan,
            "nash_log_utility": np.nan,
            "model_config_json": "",
            "stacking_weights_csv": str(weights_csv),
            "stacking_solution_name": solution_name,
            "stacking_problem_status": str(prob.status),
            "stacking_solver": str(solver_name),
            "stacking_n_active_models": int((weights_df["weight"] > 1e-8).sum()),
            "stacking_total_gap": float(sum(gap_values.values())),
            "stacking_weight_min_threshold": float(weight_min_threshold),
        }
        for col, alias in _SUMMARY_METRIC_ALIASES.items():
            out[col] = float(pd.to_numeric(row.get(col), errors="coerce")) if col in row.index else np.nan
            if col in row.index:
                out[alias] = out[col]
        fold_rows.append(out)

    return {
        "status": selected_status,
        "family_name": str(family_name),
        "selected_summary_row": summary_row,
        "selected_fold_rows": fold_rows,
    }


def _build_stacking_prediction_panels(
    *,
    preds_df: pd.DataFrame,
    family_df: pd.DataFrame,
    models: Sequence[str],
) -> Tuple[List[int], List[str], Dict[int, Tuple[np.ndarray, np.ndarray]], pd.DataFrame]:
    if "run_id" not in family_df.columns:
        raise KeyError("Convex stacking requires run_id in runs_df.")
    run_map = family_df.loc[:, ["fold_id", "config_id", "run_id", "model_name"]].drop_duplicates().copy()
    run_map["config_id"] = run_map["config_id"].astype(str)
    run_map["run_id"] = run_map["run_id"].astype(str)
    meta_df = run_map.loc[:, ["config_id", "model_name"]].drop_duplicates("config_id").copy()

    preds_min = preds_df.loc[:, ["run_id", "row_id", "y_true", "y_pred"]].copy()
    preds_min["run_id"] = preds_min["run_id"].astype(str)
    preds_min["row_id"] = preds_min["row_id"].astype(np.int64, copy=False)

    pred_by_run: Dict[str, Dict[str, np.ndarray]] = {}
    for run_id, g in preds_min.groupby("run_id", sort=False):
        gg = g.drop_duplicates(subset=["row_id"], keep="first").sort_values("row_id", kind="mergesort")
        pred_by_run[str(run_id)] = {
            "row_id": gg["row_id"].to_numpy(dtype=np.int64, copy=False),
            "y_true": gg["y_true"].to_numpy(dtype=float, copy=False),
            "y_pred": gg["y_pred"].to_numpy(dtype=float, copy=False),
        }

    fold_panels: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    keep_models = [str(m) for m in models]
    for fold_id in sorted(run_map["fold_id"].astype(int).unique().tolist()):
        sub = run_map.loc[(run_map["fold_id"].astype(int) == int(fold_id)) & (run_map["config_id"].isin(keep_models)), :].copy()
        if sub.empty:
            continue
        cfg_to_run = dict(zip(sub["config_id"].astype(str), sub["run_id"].astype(str)))
        if any(str(cfg) not in cfg_to_run for cfg in keep_models):
            continue
        run_ids = [cfg_to_run[str(cfg)] for cfg in keep_models]
        if any(run_id not in pred_by_run for run_id in run_ids):
            continue

        common_rows = pred_by_run[run_ids[0]]["row_id"]
        for run_id in run_ids[1:]:
            common_rows = np.intersect1d(common_rows, pred_by_run[run_id]["row_id"], assume_unique=False)
            if common_rows.size < 2:
                break
        if common_rows.size < 2:
            continue

        ref_run = pred_by_run[run_ids[0]]
        ref_idx = np.searchsorted(ref_run["row_id"], common_rows)
        y_true = np.maximum(ref_run["y_true"][ref_idx], 1e-9)
        cols: List[np.ndarray] = []
        ok = True
        for cfg in keep_models:
            pred_run = pred_by_run[cfg_to_run[str(cfg)]]
            pred_idx = np.searchsorted(pred_run["row_id"], common_rows)
            y_pred = np.maximum(pred_run["y_pred"][pred_idx], 1e-9)
            if not np.all(np.isfinite(y_pred)):
                ok = False
                break
            cols.append(y_pred)
        if not ok:
            continue
        fold_panels[int(fold_id)] = (np.column_stack(cols), y_true)

    folds_final = sorted(fold_panels.keys())
    if not folds_final:
        raise RuntimeError("Could not build any complete fold prediction panels for convex stacking.")
    return folds_final, keep_models, fold_panels, meta_df


def _empty_convex_stacking_result(
    *,
    family_name: str,
    status: str,
    selection_metric_id: str,
    selection_aggregation: str,
    constraint_metrics: Sequence[str],
    constraint_aggregation: str,
    level_constraint_aggregation: Optional[str] = None,
    dispersion_constraint_aggregation: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "status": str(status),
        "family_name": str(family_name),
        "selected_summary_row": {
            "row_kind": "stacking_summary",
            "comparison_group": str(family_name),
            "selection_method": "convex_stacking",
            "status": str(status),
            "config_id": "",
            "model_name": f"{family_name}_stacking",
            "model_family": str(family_name),
            "fold_id": np.nan,
            "n_folds_used": 0,
            "selection_metric": str(selection_metric_id),
            "selection_aggregation": str(selection_aggregation),
            "selection_value": np.nan,
            "selection_std": np.nan,
            "constraint_metrics": ",".join(constraint_metrics),
            **_constraint_aggregation_fields(
                constraint_aggregation=constraint_aggregation,
                level_constraint_aggregation=level_constraint_aggregation,
                dispersion_constraint_aggregation=dispersion_constraint_aggregation,
            ),
            "utopia_metric_aggregation": "",
            "utopia_distance": np.nan,
            "nash_log_utility": np.nan,
            "n_candidate_models": 0,
            "n_feasible_models": np.nan,
            "stacking_weights_csv": "",
            "stacking_solution_name": "",
            "stacking_problem_status": "",
            "stacking_solver": "",
            "stacking_n_active_models": 0,
        },
        "selected_fold_rows": [],
    }


def _select_convex_stacking_for_family(
    *,
    runs_df: pd.DataFrame,
    result_root: str,
    data_id: str,
    split_id: str,
    output_dir: Path,
    family_name: str,
    eligible_config_ids: Optional[Set[str]],
    constraint_metrics: Sequence[str],
    constraint_aggregation: str,
    selection_metric_id: str,
    selection_aggregation: str,
    solver: Optional[str],
    weight_min_threshold: float,
    artifact_suffix: str = "",
    level_constraint_aggregation: Optional[str] = None,
    dispersion_constraint_aggregation: Optional[str] = None,
) -> Dict[str, Any]:
    _validate_convex_stacking_request(selection_metric_id=selection_metric_id, constraint_metrics=constraint_metrics)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_family_df = _candidate_pool_runs_df(runs_df, group_name=family_name)
    if raw_family_df.empty:
        return _empty_convex_stacking_result(
            family_name=family_name,
            status="family_not_found",
            selection_metric_id=selection_metric_id,
            selection_aggregation=selection_aggregation,
            constraint_metrics=constraint_metrics,
            constraint_aggregation=constraint_aggregation,
            level_constraint_aggregation=level_constraint_aggregation,
            dispersion_constraint_aggregation=dispersion_constraint_aggregation,
        )
    if eligible_config_ids is not None:
        raw_family_df = raw_family_df.loc[raw_family_df["config_id"].isin(eligible_config_ids), :].copy()
        if raw_family_df.empty:
            return _empty_convex_stacking_result(
                family_name=family_name,
                status="no_test_metrics",
                selection_metric_id=selection_metric_id,
                selection_aggregation=selection_aggregation,
                constraint_metrics=constraint_metrics,
                constraint_aggregation=constraint_aggregation,
                level_constraint_aggregation=level_constraint_aggregation,
                dispersion_constraint_aggregation=dispersion_constraint_aggregation,
            )

    needed_cols = [str(_SELECTION_METRIC_SPECS[selection_metric_id].column)]
    needed_cols.extend(str(_CONSTRAINT_SPECS[metric_id].column) for metric_id in constraint_metrics)
    family_df, _, models = _align_complete_grid(raw_family_df, required_metrics=list(needed_cols))
    if family_df.empty:
        return _empty_convex_stacking_result(
            family_name=family_name,
            status="no_complete_candidates",
            selection_metric_id=selection_metric_id,
            selection_aggregation=selection_aggregation,
            constraint_metrics=constraint_metrics,
            constraint_aggregation=constraint_aggregation,
            level_constraint_aggregation=level_constraint_aggregation,
            dispersion_constraint_aggregation=dispersion_constraint_aggregation,
        )
    if "run_id" not in family_df.columns:
        raise KeyError("Convex stacking requires run_id in the CV runs artifact.")

    selected_runs_by_fold = {
        int(fold_id): [str(run_id) for run_id in fold_df["run_id"].dropna().astype(str).tolist()]
        for fold_id, fold_df in family_df.groupby("fold_id", sort=False)
    }
    preds_df = _load_predictions_df(
        result_root=result_root,
        data_id=data_id,
        split_id=split_id,
        selected_runs_by_fold=selected_runs_by_fold,
    )
    folds, models, fold_panels, meta_df = _build_stacking_prediction_panels(
        preds_df=preds_df,
        family_df=family_df,
        models=models,
    )

    try:
        import cvxpy as cp
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Convex stacking in simple_model_selection.py requires cvxpy.") from exc

    n_models = len(models)
    w = cp.Variable(n_models, nonneg=True)
    constraints = [cp.sum(w) == 1]
    fold_losses: List[Any] = []
    prd_low_terms: List[Any] = []
    prd_high_terms: List[Any] = []
    mean_ratio_terms: List[Any] = []
    weighted_mean_terms: List[Any] = []
    gap_vars = {str(metric_id): cp.Variable(nonneg=True) for metric_id in constraint_metrics}

    for fold_id in folds:
        P, y_true = fold_panels[int(fold_id)]
        y_pos = np.maximum(np.asarray(y_true, dtype=float).reshape(-1), 1e-9)
        pred_expr = P @ w
        resid = pred_expr - y_pos
        ratio_expr = cp.multiply(1.0 / y_pos, pred_expr)
        mean_ratio_expr = cp.sum(ratio_expr) / float(y_pos.shape[0])
        weighted_mean_expr = cp.sum(pred_expr) / float(np.sum(y_pos))

        if selection_metric_id == "MSE":
            fold_loss = cp.sum_squares(resid) / float(y_pos.shape[0])
        elif selection_metric_id == "RMSE":
            fold_loss = cp.norm(resid, 2) / float(np.sqrt(y_pos.shape[0]))
        elif selection_metric_id == "MAE":
            fold_loss = cp.norm1(resid) / float(y_pos.shape[0])
        elif selection_metric_id == "MAPE":
            fold_loss = cp.norm1(cp.multiply(1.0 / y_pos, resid)) / float(y_pos.shape[0])
        else:  # pragma: no cover
            raise ValueError(f"Unsupported convex stacking selection metric: {selection_metric_id}")
        fold_losses.append(fold_loss)
        mean_ratio_terms.append(mean_ratio_expr)
        weighted_mean_terms.append(weighted_mean_expr)
        prd_low_terms.append(float(IAAO_PRD_RANGE[0]) * weighted_mean_expr - mean_ratio_expr)
        prd_high_terms.append(mean_ratio_expr - float(IAAO_PRD_RANGE[1]) * weighted_mean_expr)

    if selection_aggregation == "average_fold":
        objective_expr = cp.sum(cp.hstack(fold_losses)) / float(len(fold_losses))
    elif selection_aggregation == "worst_fold":
        t = cp.Variable(nonneg=True)
        for fold_loss in fold_losses:
            constraints.append(fold_loss <= t)
        objective_expr = t
    else:
        raise ValueError("selection_aggregation must be one of: average_fold, worst_fold")

    for metric_id in constraint_metrics:
        spec = _CONSTRAINT_SPECS[metric_id]
        metric_constraint_aggregation = _resolve_constraint_aggregation(
            metric_id,
            constraint_aggregation=constraint_aggregation,
            level_constraint_aggregation=level_constraint_aggregation,
            dispersion_constraint_aggregation=dispersion_constraint_aggregation,
        )
        if metric_id == "MEAN_RATIO":
            exprs = mean_ratio_terms
            if metric_constraint_aggregation == "worst_fold":
                for expr in exprs:
                    if spec.lower is not None:
                        constraints.append(expr >= float(spec.lower) - gap_vars[metric_id])
                    if spec.upper is not None:
                        constraints.append(expr <= float(spec.upper) + gap_vars[metric_id])
            else:
                avg_expr = cp.sum(cp.hstack(exprs)) / float(len(exprs))
                if spec.lower is not None:
                    constraints.append(avg_expr >= float(spec.lower) - gap_vars[metric_id])
                if spec.upper is not None:
                    constraints.append(avg_expr <= float(spec.upper) + gap_vars[metric_id])
        elif metric_id == "WEIGHTED_MEAN_RATIO":
            exprs = weighted_mean_terms
            if metric_constraint_aggregation == "worst_fold":
                for expr in exprs:
                    if spec.lower is not None:
                        constraints.append(expr >= float(spec.lower) - gap_vars[metric_id])
                    if spec.upper is not None:
                        constraints.append(expr <= float(spec.upper) + gap_vars[metric_id])
            else:
                avg_expr = cp.sum(cp.hstack(exprs)) / float(len(exprs))
                if spec.lower is not None:
                    constraints.append(avg_expr >= float(spec.lower) - gap_vars[metric_id])
                if spec.upper is not None:
                    constraints.append(avg_expr <= float(spec.upper) + gap_vars[metric_id])
        elif metric_id == "PRD":
            if metric_constraint_aggregation == "worst_fold":
                for low_aff, high_aff in zip(prd_low_terms, prd_high_terms):
                    constraints.append(low_aff <= gap_vars[metric_id])
                    constraints.append(high_aff <= gap_vars[metric_id])
            else:
                constraints.append(cp.sum(cp.hstack(prd_low_terms)) / float(len(prd_low_terms)) <= gap_vars[metric_id])
                constraints.append(cp.sum(cp.hstack(prd_high_terms)) / float(len(prd_high_terms)) <= gap_vars[metric_id])
        else:  # pragma: no cover
            raise ValueError(f"Unsupported convex stacking constraint metric: {metric_id}")

    solver_name = _choose_convex_stacking_solver(cp, solver)
    total_gap_expr = cp.sum(cp.hstack([gap_vars[metric_id] for metric_id in constraint_metrics])) if gap_vars else 0.0
    gap_prob = cp.Problem(cp.Minimize(total_gap_expr), constraints)
    gap_prob.solve(solver=solver_name, verbose=False)
    if gap_prob.status not in {"optimal", "optimal_inaccurate"} or w.value is None:
        raise RuntimeError(f"Convex stacking feasibility optimization failed for {family_name}. status={gap_prob.status}")
    gap_values = {metric_id: float(gap_vars[metric_id].value or 0.0) for metric_id in constraint_metrics}
    weights_raw = np.asarray(w.value, dtype=float).reshape(-1)
    gap_tol = 1e-6
    objective_constraints = list(constraints)
    for metric_id in constraint_metrics:
        objective_constraints.append(
            gap_vars[metric_id] <= gap_values[metric_id] + max(gap_tol, 1e-4 * max(1.0, gap_values[metric_id]))
        )
    prob = cp.Problem(cp.Minimize(objective_expr), objective_constraints)
    prob.solve(solver=solver_name, verbose=False)
    if prob.status in {"optimal", "optimal_inaccurate"} and w.value is not None:
        weights_raw = np.asarray(w.value, dtype=float).reshape(-1)
    else:
        prob = gap_prob
    selected_status = (
        "selected"
        if max([float(val) for val in gap_values.values()], default=0.0) <= gap_tol
        else "selected_closest_infeasible"
    )

    weights = np.asarray(weights_raw, dtype=float).reshape(-1)
    if not np.all(np.isfinite(weights)):
        raise RuntimeError(f"Convex stacking produced non-finite weights for {family_name}.")
    weights = _apply_stacking_weight_threshold(weights, min_weight=weight_min_threshold)

    family_configs = (
        family_df.loc[:, ["config_id", "model_name", "model_config_json", "model_family"]]
        .drop_duplicates("config_id")
        .copy()
    )
    weights_df = pd.DataFrame({"config_id": models, "weight": weights}).merge(
        family_configs.loc[:, ["config_id", "model_name", "model_config_json", "model_family"]],
        on="config_id",
        how="left",
    )
    weights_df = weights_df.sort_values("weight", ascending=False, ignore_index=True)

    positive_cfg = weights_df.loc[weights_df["weight"] > 1e-8, :].copy()
    ratio_modes = [
        _extract_ratio_mode_from_model_config_json(val)
        for val in positive_cfg.get("model_config_json", pd.Series(dtype=str)).astype(str).tolist()
    ]
    ratio_modes = [mode for mode in ratio_modes if mode]
    ratio_mode = ratio_modes[0] if ratio_modes else "diff"

    fold_metric_rows: List[Dict[str, Any]] = []
    for fold_id in folds:
        P, y_true = fold_panels[int(fold_id)]
        y_pred = np.maximum(P @ weights, 1e-9)
        y_true_log = np.log(np.maximum(y_true, 1e-9))
        y_pred_log = np.log(y_pred)
        metrics = _compute_extended_metrics(
            y_true_log=y_true_log,
            y_pred_log=y_pred_log,
            y_train_log=y_true_log,
            ratio_mode=ratio_mode,
        )
        metrics["MSE"] = float(np.mean(np.square(y_pred - y_true)))
        metrics["selection_value"] = _stacking_fold_loss_value(y_true, y_pred, selection_metric_id=selection_metric_id)
        metrics["fold_id"] = int(fold_id)
        fold_metric_rows.append(metrics)

    fold_metrics_df = pd.DataFrame(fold_metric_rows).sort_values("fold_id", kind="mergesort").reset_index(drop=True)
    selection_vals = pd.to_numeric(fold_metrics_df["selection_value"], errors="coerce").to_numpy(dtype=float)
    selected_row = pd.Series(
        {
            "selection_value": _aggregate_selection_metric(selection_vals, _SELECTION_METRIC_SPECS[selection_metric_id], selection_aggregation),
            "selection_std": _metric_std(selection_vals),
        }
    )

    suffix = f"_{str(artifact_suffix).strip()}" if str(artifact_suffix).strip() else ""
    solution_name = f"{family_name}_CONVEX_STACKING{suffix.upper()}"
    weights_csv = output_dir / f"weights_stacking_{family_name}{suffix}.csv"
    weights_df.to_csv(weights_csv, index=False)

    summary_row: Dict[str, Any] = {
        "row_kind": "stacking_summary",
        "comparison_group": str(family_name),
        "selection_method": "convex_stacking",
        "status": selected_status,
        "config_id": "",
        "model_name": solution_name,
        "model_family": str(family_name),
        "fold_id": np.nan,
        "n_folds_used": int(len(folds)),
        "selection_metric": str(selection_metric_id),
        "selection_aggregation": str(selection_aggregation),
        "selection_value": float(selected_row["selection_value"]),
        "selection_std": float(selected_row["selection_std"]),
        "constraint_metrics": ",".join(constraint_metrics),
        **_constraint_aggregation_fields(
            constraint_aggregation=constraint_aggregation,
            level_constraint_aggregation=level_constraint_aggregation,
            dispersion_constraint_aggregation=dispersion_constraint_aggregation,
        ),
        "utopia_metric_aggregation": "",
        "utopia_distance": np.nan,
        "nash_log_utility": np.nan,
        "model_config_json": "",
        "n_candidate_models": int(len(models)),
        "n_feasible_models": np.nan,
        "stacking_weights_csv": str(weights_csv),
        "stacking_solution_name": solution_name,
        "stacking_problem_status": str(prob.status),
        "stacking_solver": str(solver_name),
        "stacking_n_active_models": int((weights_df["weight"] > 1e-8).sum()),
        "stacking_total_gap": float(sum(gap_values.values())),
        "stacking_weight_min_threshold": float(weight_min_threshold),
    }
    summary_row.update(_build_metric_summary(fold_metrics_df))
    if constraint_metrics:
        summary_row.update(
            _build_constraint_columns(
                fold_metrics_df,
                constraint_metrics=constraint_metrics,
                constraint_aggregation=constraint_aggregation,
                level_constraint_aggregation=level_constraint_aggregation,
                dispersion_constraint_aggregation=dispersion_constraint_aggregation,
            )
        )

    fold_rows: List[Dict[str, Any]] = []
    for _, row in fold_metrics_df.iterrows():
        out: Dict[str, Any] = {
            "row_kind": "stacking_fold",
            "comparison_group": str(family_name),
            "selection_method": "convex_stacking",
            "status": selected_status,
            "config_id": "",
            "model_name": solution_name,
            "model_family": str(family_name),
            "fold_id": int(row["fold_id"]),
            "n_folds_used": int(len(folds)),
            "selection_metric": str(selection_metric_id),
            "selection_aggregation": str(selection_aggregation),
            "selection_value": float(pd.to_numeric(row.get("selection_value"), errors="coerce")),
            "selection_std": np.nan,
            "constraint_metrics": ",".join(constraint_metrics),
            **_constraint_aggregation_fields(
                constraint_aggregation=constraint_aggregation,
                level_constraint_aggregation=level_constraint_aggregation,
                dispersion_constraint_aggregation=dispersion_constraint_aggregation,
            ),
            "utopia_metric_aggregation": "",
            "utopia_distance": np.nan,
            "nash_log_utility": np.nan,
            "model_config_json": "",
            "stacking_weights_csv": str(weights_csv),
            "stacking_solution_name": solution_name,
            "stacking_problem_status": str(prob.status),
            "stacking_solver": str(solver_name),
            "stacking_n_active_models": int((weights_df["weight"] > 1e-8).sum()),
            "stacking_total_gap": float(sum(gap_values.values())),
            "stacking_weight_min_threshold": float(weight_min_threshold),
        }
        for col, alias in _SUMMARY_METRIC_ALIASES.items():
            out[col] = float(pd.to_numeric(row.get(col), errors="coerce")) if col in row.index else np.nan
            if col in row.index:
                out[alias] = out[col]
        fold_rows.append(out)

    return {
        "status": selected_status,
        "family_name": str(family_name),
        "selected_summary_row": summary_row,
        "selected_fold_rows": fold_rows,
    }


def _build_constraint_columns(
    row_df: pd.DataFrame,
    *,
    constraint_metrics: Sequence[str],
    constraint_aggregation: str,
    level_constraint_aggregation: Optional[str] = None,
    dispersion_constraint_aggregation: Optional[str] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for metric_id in constraint_metrics:
        spec = _CONSTRAINT_SPECS[metric_id]
        vals = pd.to_numeric(row_df[spec.column], errors="coerce").to_numpy(dtype=float)
        metric_constraint_aggregation = _resolve_constraint_aggregation(
            metric_id,
            constraint_aggregation=constraint_aggregation,
            level_constraint_aggregation=level_constraint_aggregation,
            dispersion_constraint_aggregation=dispersion_constraint_aggregation,
        )
        eval_out = _evaluate_constraint(vals, spec, metric_constraint_aggregation)
        out[f"constraint_{metric_id}_bounds"] = _bounds_label(spec)
        out[f"constraint_{metric_id}_value"] = eval_out["aggregation_value"]
        out[f"constraint_{metric_id}_pass"] = bool(eval_out["passes"])
        out[f"constraint_{metric_id}_violation"] = _constraint_violation(
            float(eval_out["aggregation_value"]),
            lower=spec.lower,
            upper=spec.upper,
        )
        out[f"constraint_{metric_id}_worst_violation"] = eval_out["worst_violation"]
    return out


def _select_closest_infeasible_candidate(
    stats_df: pd.DataFrame,
    *,
    constraint_metrics: Sequence[str],
    selection_spec: SelectionMetricSpec,
) -> pd.Series:
    if stats_df.empty:
        raise RuntimeError("No candidate rows available for infeasible fallback selection.")

    ranked = stats_df.copy()
    sort_cols: List[str] = []
    ascending: List[bool] = []

    for metric_id in constraint_metrics:
        violation_col = f"constraint_{metric_id}_violation"
        if violation_col not in ranked.columns:
            raise KeyError(f"Missing fallback violation column: {violation_col}")
        ranked[violation_col] = pd.to_numeric(ranked[violation_col], errors="coerce")
        sort_cols.append(violation_col)
        ascending.append(True)

    ranked["selection_value"] = pd.to_numeric(ranked["selection_value"], errors="coerce")
    ranked["selection_std"] = pd.to_numeric(ranked["selection_std"], errors="coerce")
    sort_cols.extend(["selection_value", "selection_std", "config_id"])
    ascending.extend([not selection_spec.higher_is_better, True, True])

    ranked = ranked.sort_values(
        by=sort_cols,
        ascending=ascending,
        na_position="last",
        ignore_index=True,
    )
    return ranked.iloc[0]


def _collect_candidate_stats(
    family_df: pd.DataFrame,
    *,
    constraint_metrics: Sequence[str],
    constraint_aggregation: str,
    level_constraint_aggregation: Optional[str] = None,
    dispersion_constraint_aggregation: Optional[str] = None,
    selection_metric_id: str,
    selection_aggregation: str,
    apply_constraints: bool,
) -> pd.DataFrame:
    selection_spec = _SELECTION_METRIC_SPECS[selection_metric_id]
    rows: List[Dict[str, Any]] = []
    for config_id, config_df in family_df.groupby("config_id", sort=False):
        model_name = str(config_df["model_name"].iloc[0])
        selection_vals = _selection_metric_values_from_df(config_df, selection_spec)
        row: Dict[str, Any] = {
            "config_id": str(config_id),
            "model_name": model_name,
            "model_family": (
                str(config_df["model_family"].iloc[0])
                if "model_family" in config_df.columns
                else _base_model_name(model_name)
            ),
            "n_folds_used": int(config_df["fold_id"].nunique()),
            "selection_metric": str(selection_metric_id),
            "selection_aggregation": str(selection_aggregation),
            "selection_value": _aggregate_selection_metric(selection_vals, selection_spec, selection_aggregation),
            "selection_std": _metric_std(selection_vals),
            "model_config_json": _extract_model_config_json(config_df),
        }
        if apply_constraints:
            row.update(
                _constraint_aggregation_fields(
                    constraint_aggregation=constraint_aggregation,
                    level_constraint_aggregation=level_constraint_aggregation,
                    dispersion_constraint_aggregation=dispersion_constraint_aggregation,
                )
            )
            row["constraint_metrics"] = ",".join(constraint_metrics)
            row.update(
                _build_constraint_columns(
                    config_df,
                    constraint_metrics=constraint_metrics,
                    constraint_aggregation=constraint_aggregation,
                    level_constraint_aggregation=level_constraint_aggregation,
                    dispersion_constraint_aggregation=dispersion_constraint_aggregation,
                )
            )
            row["passes_constraints"] = bool(
                all(bool(row.get(f"constraint_{metric_id}_pass", False)) for metric_id in constraint_metrics)
            )
        else:
            row["constraint_aggregation"] = ""
            row["level_constraint_aggregation"] = ""
            row["dispersion_constraint_aggregation"] = ""
            row["constraint_metrics"] = ""
            row["passes_constraints"] = True
        rows.append(row)
    return pd.DataFrame(rows)


def _build_summary_row(
    *,
    row_kind: str,
    comparison_group: str,
    selected_row: pd.Series,
    selected_df: pd.DataFrame,
    selection_metric_id: str,
    selection_aggregation: str,
    constraint_metrics: Sequence[str],
    constraint_aggregation: str,
    level_constraint_aggregation: Optional[str] = None,
    dispersion_constraint_aggregation: Optional[str] = None,
    status: str,
    selection_method: str,
    utopia_metric_aggregation: str,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "row_kind": str(row_kind),
        "comparison_group": str(comparison_group),
        "selection_method": str(selection_method),
        "status": str(status),
        "config_id": str(selected_row["config_id"]),
        "model_name": str(selected_row["model_name"]),
        "model_family": str(selected_row["model_family"]),
        "fold_id": np.nan,
        "n_folds_used": int(selected_df["fold_id"].nunique()),
        "selection_metric": str(selection_metric_id),
        "selection_aggregation": str(selection_aggregation),
        "selection_value": float(selected_row["selection_value"]),
        "selection_std": float(selected_row["selection_std"]),
        "constraint_metrics": ",".join(constraint_metrics),
        **_constraint_aggregation_fields(
            constraint_aggregation=constraint_aggregation,
            level_constraint_aggregation=level_constraint_aggregation,
            dispersion_constraint_aggregation=dispersion_constraint_aggregation,
        ),
        "utopia_metric_aggregation": str(utopia_metric_aggregation),
        "utopia_distance": float(pd.to_numeric(selected_row.get("utopia_distance"), errors="coerce")),
        "nash_log_utility": float(pd.to_numeric(selected_row.get("nash_log_utility"), errors="coerce")),
        "model_config_json": _extract_model_config_json(selected_df),
    }
    out.update(_build_metric_summary(selected_df))
    if constraint_metrics:
        out.update(
            _build_constraint_columns(
                selected_df,
                constraint_metrics=constraint_metrics,
                constraint_aggregation=constraint_aggregation,
                level_constraint_aggregation=level_constraint_aggregation,
                dispersion_constraint_aggregation=dispersion_constraint_aggregation,
            )
        )
    return out


def _build_fold_rows(
    *,
    row_kind: str,
    comparison_group: str,
    selected_df: pd.DataFrame,
    selection_metric_id: str,
    selection_aggregation: str,
    constraint_metrics: Sequence[str],
    constraint_aggregation: str,
    level_constraint_aggregation: Optional[str] = None,
    dispersion_constraint_aggregation: Optional[str] = None,
    selection_method: str,
    utopia_metric_aggregation: str,
    status: str,
    utopia_distance: float,
    nash_log_utility: float,
) -> List[Dict[str, Any]]:
    selection_spec = _SELECTION_METRIC_SPECS[selection_metric_id]
    rows: List[Dict[str, Any]] = []
    for _, row in selected_df.sort_values("fold_id", kind="mergesort").iterrows():
        selection_val = _selection_metric_values_from_df(pd.DataFrame([row]), selection_spec)
        out: Dict[str, Any] = {
            "row_kind": str(row_kind),
            "comparison_group": str(comparison_group),
            "selection_method": str(selection_method),
            "status": str(status),
            "config_id": str(row["config_id"]),
            "model_name": str(row["model_name"]),
            "model_family": str(row["model_family"]),
            "fold_id": int(row["fold_id"]),
            "n_folds_used": int(selected_df["fold_id"].nunique()),
            "selection_metric": str(selection_metric_id),
            "selection_aggregation": str(selection_aggregation),
            "selection_value": float(selection_val[0]) if selection_val.size > 0 and np.isfinite(selection_val[0]) else np.nan,
            "selection_std": np.nan,
            "constraint_metrics": ",".join(constraint_metrics),
            **_constraint_aggregation_fields(
                constraint_aggregation=constraint_aggregation,
                level_constraint_aggregation=level_constraint_aggregation,
                dispersion_constraint_aggregation=dispersion_constraint_aggregation,
            ),
            "utopia_metric_aggregation": str(utopia_metric_aggregation),
            "utopia_distance": float(utopia_distance),
            "nash_log_utility": float(nash_log_utility),
            "model_config_json": str(row.get("model_config_json", "")) if pd.notna(row.get("model_config_json", "")) else "",
        }
        for col, alias in _SUMMARY_METRIC_ALIASES.items():
            out[col] = float(pd.to_numeric(row.get(col), errors="coerce")) if col in row.index else np.nan
            if col in row.index:
                out[alias] = out[col]
        rows.append(out)
    return rows


def _empty_selection_result(
    *,
    family_name: str,
    selection_method: str,
    status: str,
    selection_metric_id: str,
    selection_aggregation: str,
    constraint_metrics: Sequence[str],
    constraint_aggregation: str,
    level_constraint_aggregation: Optional[str] = None,
    dispersion_constraint_aggregation: Optional[str] = None,
    utopia_metric_aggregation: str,
) -> Dict[str, Any]:
    return {
        "status": str(status),
        "family_name": str(family_name),
        "selected_config_id": None,
        "selected_summary_row": {
            "row_kind": "summary",
            "comparison_group": str(family_name),
            "selection_method": str(selection_method),
            "status": str(status),
            "config_id": "",
            "model_name": "",
            "model_family": str(family_name),
            "fold_id": np.nan,
            "n_folds_used": 0,
            "selection_metric": str(selection_metric_id),
            "selection_aggregation": str(selection_aggregation),
            "selection_value": np.nan,
            "selection_std": np.nan,
            "constraint_metrics": ",".join(constraint_metrics),
            **_constraint_aggregation_fields(
                constraint_aggregation=constraint_aggregation,
                level_constraint_aggregation=level_constraint_aggregation,
                dispersion_constraint_aggregation=dispersion_constraint_aggregation,
            ),
            "utopia_metric_aggregation": str(utopia_metric_aggregation),
            "utopia_distance": np.nan,
            "nash_log_utility": np.nan,
            "n_candidate_models": 0,
            "n_feasible_models": np.nan,
        },
        "selected_fold_rows": [],
    }


def _prepare_family_panel(
    runs_df: pd.DataFrame,
    *,
    family_name: str,
    required_columns: Sequence[str],
) -> Tuple[pd.DataFrame, List[int], List[str]]:
    family_df = runs_df.loc[runs_df["model_family"] == str(family_name), :].copy()
    if family_df.empty:
        return pd.DataFrame(), [], []
    return _align_complete_grid(family_df, required_metrics=list(required_columns))


def _select_for_family(
    runs_df: pd.DataFrame,
    *,
    family_name: str,
    eligible_config_ids: Optional[Set[str]] = None,
    constraint_metrics: Sequence[str],
    constraint_aggregation: str,
    level_constraint_aggregation: Optional[str] = None,
    dispersion_constraint_aggregation: Optional[str] = None,
    selection_metric_id: str,
    selection_aggregation: str,
    apply_constraints: bool,
    selection_method: str = "constrained",
    utopia_metric_aggregation: str = "",
) -> Dict[str, Any]:
    raw_family_df = _candidate_pool_runs_df(runs_df, group_name=family_name)
    if raw_family_df.empty:
        return _empty_selection_result(
            family_name=family_name,
            selection_method=selection_method,
            status="family_not_found",
            selection_metric_id=selection_metric_id,
            selection_aggregation=selection_aggregation,
            constraint_metrics=constraint_metrics,
            constraint_aggregation=constraint_aggregation,
            level_constraint_aggregation=level_constraint_aggregation,
            dispersion_constraint_aggregation=dispersion_constraint_aggregation,
            utopia_metric_aggregation=utopia_metric_aggregation,
        )
    if eligible_config_ids is not None:
        raw_family_df = raw_family_df.loc[raw_family_df["config_id"].isin(eligible_config_ids), :].copy()
        if raw_family_df.empty:
            return _empty_selection_result(
                family_name=family_name,
                selection_method=selection_method,
                status="no_test_metrics",
                selection_metric_id=selection_metric_id,
                selection_aggregation=selection_aggregation,
                constraint_metrics=constraint_metrics,
                constraint_aggregation=constraint_aggregation,
                level_constraint_aggregation=level_constraint_aggregation,
                dispersion_constraint_aggregation=dispersion_constraint_aggregation,
                utopia_metric_aggregation=utopia_metric_aggregation,
            )

    needed_cols = [str(_SELECTION_METRIC_SPECS[selection_metric_id].column)]
    needed_cols.extend(str(_CONSTRAINT_SPECS[metric_id].column) for metric_id in constraint_metrics)
    family_df, folds, models = _align_complete_grid(raw_family_df, required_metrics=list(needed_cols))
    if family_df.empty:
        return _empty_selection_result(
            family_name=family_name,
            selection_method=selection_method,
            status="no_complete_candidates",
            selection_metric_id=selection_metric_id,
            selection_aggregation=selection_aggregation,
            constraint_metrics=constraint_metrics,
            constraint_aggregation=constraint_aggregation,
            level_constraint_aggregation=level_constraint_aggregation,
            dispersion_constraint_aggregation=dispersion_constraint_aggregation,
            utopia_metric_aggregation=utopia_metric_aggregation,
        )

    stats_df = _collect_candidate_stats(
        family_df,
        constraint_metrics=constraint_metrics,
        constraint_aggregation=constraint_aggregation,
        level_constraint_aggregation=level_constraint_aggregation,
        dispersion_constraint_aggregation=dispersion_constraint_aggregation,
        selection_metric_id=selection_metric_id,
        selection_aggregation=selection_aggregation,
        apply_constraints=apply_constraints,
    )
    if apply_constraints:
        feasible_df = stats_df.loc[stats_df["passes_constraints"].fillna(False), :].copy()
    else:
        feasible_df = stats_df.copy()

    selection_spec = _SELECTION_METRIC_SPECS[selection_metric_id]
    selected_status = "selected"
    if feasible_df.empty:
        selected_row = _select_closest_infeasible_candidate(
            stats_df,
            constraint_metrics=constraint_metrics,
            selection_spec=selection_spec,
        )
        selected_status = "selected_closest_infeasible"
    else:
        selected_row = _select_best_candidate(
            feasible_df,
            selection_spec=selection_spec,
        )
    selected_config_id = str(selected_row["config_id"])
    selected_df = family_df.loc[family_df["config_id"] == selected_config_id, :].copy()
    summary_row = _build_summary_row(
        row_kind="summary",
        comparison_group=family_name,
        selected_row=selected_row,
        selected_df=selected_df,
        selection_metric_id=selection_metric_id,
        selection_aggregation=selection_aggregation,
        constraint_metrics=constraint_metrics if apply_constraints else (),
        constraint_aggregation=constraint_aggregation if apply_constraints else "",
        level_constraint_aggregation=(level_constraint_aggregation if apply_constraints else ""),
        dispersion_constraint_aggregation=(dispersion_constraint_aggregation if apply_constraints else ""),
        status=selected_status,
        selection_method=selection_method,
        utopia_metric_aggregation=utopia_metric_aggregation,
    )
    summary_row["n_candidate_models"] = int(len(models))
    summary_row["n_feasible_models"] = int(feasible_df.shape[0])
    fold_rows = _build_fold_rows(
        row_kind="fold",
        comparison_group=family_name,
        selected_df=selected_df,
        selection_metric_id=selection_metric_id,
        selection_aggregation=selection_aggregation,
        constraint_metrics=constraint_metrics if apply_constraints else (),
        constraint_aggregation=constraint_aggregation if apply_constraints else "",
        level_constraint_aggregation=(level_constraint_aggregation if apply_constraints else ""),
        dispersion_constraint_aggregation=(dispersion_constraint_aggregation if apply_constraints else ""),
        selection_method=selection_method,
        utopia_metric_aggregation=utopia_metric_aggregation,
        status=selected_status,
        utopia_distance=float(pd.to_numeric(selected_row.get("utopia_distance"), errors="coerce")),
        nash_log_utility=float(pd.to_numeric(selected_row.get("nash_log_utility"), errors="coerce")),
    )
    return {
        "status": selected_status,
        "family_name": str(family_name),
        "selected_config_id": str(selected_config_id),
        "selected_summary_row": summary_row,
        "selected_fold_rows": fold_rows,
    }


def _select_for_family_utopia(
    runs_df: pd.DataFrame,
    *,
    family_name: str,
    eligible_config_ids: Optional[Set[str]] = None,
    constraint_metrics: Sequence[str],
    selection_metric_id: str,
    utopia_metric_aggregation: str,
    level_constraint_aggregation: Optional[str] = None,
    dispersion_constraint_aggregation: Optional[str] = None,
) -> Dict[str, Any]:
    raw_family_df = _candidate_pool_runs_df(runs_df, group_name=family_name)
    if raw_family_df.empty:
        return _empty_selection_result(
            family_name=family_name,
            selection_method="utopia",
            status="family_not_found",
            selection_metric_id=selection_metric_id,
            selection_aggregation=utopia_metric_aggregation,
            constraint_metrics=constraint_metrics,
            constraint_aggregation=utopia_metric_aggregation,
            level_constraint_aggregation=level_constraint_aggregation,
            dispersion_constraint_aggregation=dispersion_constraint_aggregation,
            utopia_metric_aggregation=utopia_metric_aggregation,
        )
    if eligible_config_ids is not None:
        raw_family_df = raw_family_df.loc[raw_family_df["config_id"].isin(eligible_config_ids), :].copy()
        if raw_family_df.empty:
            return _empty_selection_result(
                family_name=family_name,
                selection_method="utopia",
                status="no_test_metrics",
                selection_metric_id=selection_metric_id,
                selection_aggregation=utopia_metric_aggregation,
                constraint_metrics=constraint_metrics,
                constraint_aggregation=utopia_metric_aggregation,
                level_constraint_aggregation=level_constraint_aggregation,
                dispersion_constraint_aggregation=dispersion_constraint_aggregation,
                utopia_metric_aggregation=utopia_metric_aggregation,
            )

    needed_cols = [str(_SELECTION_METRIC_SPECS[selection_metric_id].column)]
    needed_cols.extend(str(_CONSTRAINT_SPECS[metric_id].column) for metric_id in constraint_metrics)
    family_df, folds, models = _align_complete_grid(raw_family_df, required_metrics=list(needed_cols))
    if family_df.empty:
        return _empty_selection_result(
            family_name=family_name,
            selection_method="utopia",
            status="no_complete_candidates",
            selection_metric_id=selection_metric_id,
            selection_aggregation=utopia_metric_aggregation,
            constraint_metrics=constraint_metrics,
            constraint_aggregation=utopia_metric_aggregation,
            level_constraint_aggregation=level_constraint_aggregation,
            dispersion_constraint_aggregation=dispersion_constraint_aggregation,
            utopia_metric_aggregation=utopia_metric_aggregation,
        )

    stats_df = _collect_candidate_stats(
        family_df,
        constraint_metrics=constraint_metrics,
        constraint_aggregation=utopia_metric_aggregation,
        level_constraint_aggregation=level_constraint_aggregation,
        dispersion_constraint_aggregation=dispersion_constraint_aggregation,
        selection_metric_id=selection_metric_id,
        selection_aggregation=utopia_metric_aggregation,
        apply_constraints=bool(constraint_metrics),
    )
    selection_spec = _SELECTION_METRIC_SPECS[selection_metric_id]
    selected_row = _select_utopia_candidate(
        stats_df,
        constraint_metrics=constraint_metrics,
        selection_spec=selection_spec,
    )
    selected_config_id = str(selected_row["config_id"])
    selected_df = family_df.loc[family_df["config_id"] == selected_config_id, :].copy()
    summary_row = _build_summary_row(
        row_kind="summary",
        comparison_group=family_name,
        selected_row=selected_row,
        selected_df=selected_df,
        selection_metric_id=selection_metric_id,
        selection_aggregation=utopia_metric_aggregation,
        constraint_metrics=constraint_metrics,
        constraint_aggregation=utopia_metric_aggregation,
        level_constraint_aggregation=level_constraint_aggregation,
        dispersion_constraint_aggregation=dispersion_constraint_aggregation,
        status="selected",
        selection_method="utopia",
        utopia_metric_aggregation=utopia_metric_aggregation,
    )
    summary_row["n_candidate_models"] = int(len(models))
    summary_row["n_feasible_models"] = np.nan
    fold_rows = _build_fold_rows(
        row_kind="fold",
        comparison_group=family_name,
        selected_df=selected_df,
        selection_metric_id=selection_metric_id,
        selection_aggregation=utopia_metric_aggregation,
        constraint_metrics=constraint_metrics,
        constraint_aggregation=utopia_metric_aggregation,
        level_constraint_aggregation=level_constraint_aggregation,
        dispersion_constraint_aggregation=dispersion_constraint_aggregation,
        selection_method="utopia",
        utopia_metric_aggregation=utopia_metric_aggregation,
        status="selected",
        utopia_distance=float(pd.to_numeric(selected_row.get("utopia_distance"), errors="coerce")),
        nash_log_utility=float(pd.to_numeric(selected_row.get("nash_log_utility"), errors="coerce")),
    )
    return {
        "status": "selected",
        "family_name": str(family_name),
        "selected_config_id": str(selected_config_id),
        "selected_summary_row": summary_row,
        "selected_fold_rows": fold_rows,
    }


def _select_for_family_nash(
    runs_df: pd.DataFrame,
    *,
    family_name: str,
    eligible_config_ids: Optional[Set[str]] = None,
    constraint_metrics: Sequence[str],
    selection_metric_id: str,
    nash_metric_aggregation: str,
    level_constraint_aggregation: Optional[str] = None,
    dispersion_constraint_aggregation: Optional[str] = None,
) -> Dict[str, Any]:
    raw_family_df = _candidate_pool_runs_df(runs_df, group_name=family_name)
    if raw_family_df.empty:
        return _empty_selection_result(
            family_name=family_name,
            selection_method="nash",
            status="family_not_found",
            selection_metric_id=selection_metric_id,
            selection_aggregation=nash_metric_aggregation,
            constraint_metrics=constraint_metrics,
            constraint_aggregation=nash_metric_aggregation,
            level_constraint_aggregation=level_constraint_aggregation,
            dispersion_constraint_aggregation=dispersion_constraint_aggregation,
            utopia_metric_aggregation=nash_metric_aggregation,
        )
    if eligible_config_ids is not None:
        raw_family_df = raw_family_df.loc[raw_family_df["config_id"].isin(eligible_config_ids), :].copy()
        if raw_family_df.empty:
            return _empty_selection_result(
                family_name=family_name,
                selection_method="nash",
                status="no_test_metrics",
                selection_metric_id=selection_metric_id,
                selection_aggregation=nash_metric_aggregation,
                constraint_metrics=constraint_metrics,
                constraint_aggregation=nash_metric_aggregation,
                level_constraint_aggregation=level_constraint_aggregation,
                dispersion_constraint_aggregation=dispersion_constraint_aggregation,
                utopia_metric_aggregation=nash_metric_aggregation,
            )

    needed_cols = [str(_SELECTION_METRIC_SPECS[selection_metric_id].column)]
    needed_cols.extend(str(_CONSTRAINT_SPECS[metric_id].column) for metric_id in constraint_metrics)
    family_df, folds, models = _align_complete_grid(raw_family_df, required_metrics=list(needed_cols))
    if family_df.empty:
        return _empty_selection_result(
            family_name=family_name,
            selection_method="nash",
            status="no_complete_candidates",
            selection_metric_id=selection_metric_id,
            selection_aggregation=nash_metric_aggregation,
            constraint_metrics=constraint_metrics,
            constraint_aggregation=nash_metric_aggregation,
            level_constraint_aggregation=level_constraint_aggregation,
            dispersion_constraint_aggregation=dispersion_constraint_aggregation,
            utopia_metric_aggregation=nash_metric_aggregation,
        )

    stats_df = _collect_candidate_stats(
        family_df,
        constraint_metrics=constraint_metrics,
        constraint_aggregation=nash_metric_aggregation,
        level_constraint_aggregation=level_constraint_aggregation,
        dispersion_constraint_aggregation=dispersion_constraint_aggregation,
        selection_metric_id=selection_metric_id,
        selection_aggregation=nash_metric_aggregation,
        apply_constraints=bool(constraint_metrics),
    )
    selection_spec = _SELECTION_METRIC_SPECS[selection_metric_id]
    selected_row = _select_nash_candidate(
        stats_df,
        constraint_metrics=constraint_metrics,
        selection_spec=selection_spec,
    )
    selected_config_id = str(selected_row["config_id"])
    selected_df = family_df.loc[family_df["config_id"] == selected_config_id, :].copy()
    summary_row = _build_summary_row(
        row_kind="summary",
        comparison_group=family_name,
        selected_row=selected_row,
        selected_df=selected_df,
        selection_metric_id=selection_metric_id,
        selection_aggregation=nash_metric_aggregation,
        constraint_metrics=constraint_metrics,
        constraint_aggregation=nash_metric_aggregation,
        level_constraint_aggregation=level_constraint_aggregation,
        dispersion_constraint_aggregation=dispersion_constraint_aggregation,
        status="selected",
        selection_method="nash",
        utopia_metric_aggregation=nash_metric_aggregation,
    )
    summary_row["n_candidate_models"] = int(len(models))
    summary_row["n_feasible_models"] = np.nan
    fold_rows = _build_fold_rows(
        row_kind="fold",
        comparison_group=family_name,
        selected_df=selected_df,
        selection_metric_id=selection_metric_id,
        selection_aggregation=nash_metric_aggregation,
        constraint_metrics=constraint_metrics,
        constraint_aggregation=nash_metric_aggregation,
        level_constraint_aggregation=level_constraint_aggregation,
        dispersion_constraint_aggregation=dispersion_constraint_aggregation,
        selection_method="nash",
        utopia_metric_aggregation=nash_metric_aggregation,
        status="selected",
        utopia_distance=float(pd.to_numeric(selected_row.get("utopia_distance"), errors="coerce")),
        nash_log_utility=float(pd.to_numeric(selected_row.get("nash_log_utility"), errors="coerce")),
    )
    return {
        "status": "selected",
        "family_name": str(family_name),
        "selected_config_id": str(selected_config_id),
        "selected_summary_row": summary_row,
        "selected_fold_rows": fold_rows,
    }


def run_simple_model_selection(
    *,
    result_root: str,
    data_id: str,
    split_id: str,
    skip_first_folds: int = 0,
    constraint_metrics: Optional[Sequence[str]] = None,
    constraint_metric_subsets: Optional[Sequence[Sequence[str]]] = None,
    constraint_aggregation: str = "average_fold",
    level_constraint_aggregation: Optional[str] = None,
    dispersion_constraint_aggregation: Optional[str] = None,
    selection_metric: str = _DEFAULT_SELECTION_METRIC,
    selection_aggregation: str = "average_fold",
    selection_method: str = _DEFAULT_SELECTION_METHOD,
    utopia_metric_aggregation: str = _DEFAULT_UTOPIA_AGGREGATION,
    run_convex_stacking: bool = False,
    convex_stacking_solver: Optional[str] = None,
    run_linearized_stacking: bool = False,
    linearized_stacking_solver: Optional[str] = None,
    stacking_weight_min_threshold: float = 0.0,
) -> Dict[str, str]:
    base_constraint_metric_ids = _normalize_constraint_metric_ids(
        constraint_metrics or _DEFAULT_CONSTRAINT_METRICS
    )
    requested_constraint_sets = (
        list(constraint_metric_subsets)
        if constraint_metric_subsets is not None
        else [base_constraint_metric_ids]
    )
    constraint_metric_sets: List[List[str]] = []
    seen_constraint_sets: Set[Tuple[str, ...]] = set()
    for subset in requested_constraint_sets:
        subset_ids = _normalize_constraint_metric_ids(subset or base_constraint_metric_ids)
        subset_key = tuple(subset_ids)
        if subset_key in seen_constraint_sets:
            continue
        constraint_metric_sets.append(subset_ids)
        seen_constraint_sets.add(subset_key)
    if not constraint_metric_sets:
        constraint_metric_sets = [base_constraint_metric_ids]

    selection_metric_id = str(selection_metric).upper()
    if selection_metric_id not in _SELECTION_METRIC_SPECS:
        raise ValueError(
            f"Unknown selection metric '{selection_metric}'. "
            f"Valid options: {sorted(_SELECTION_METRIC_SPECS.keys())}"
        )

    _validate_fold_aggregation("constraint_aggregation", constraint_aggregation)
    if level_constraint_aggregation is not None:
        _validate_fold_aggregation("level_constraint_aggregation", str(level_constraint_aggregation))
    if dispersion_constraint_aggregation is not None:
        _validate_fold_aggregation("dispersion_constraint_aggregation", str(dispersion_constraint_aggregation))
    _validate_fold_aggregation("selection_aggregation", selection_aggregation)
    _validate_fold_aggregation("utopia_metric_aggregation", utopia_metric_aggregation)
    if selection_method not in {"constrained", "utopia", "nash", "both"}:
        raise ValueError("selection_method must be one of: constrained, utopia, nash, both")
    if float(stacking_weight_min_threshold) < 0.0 or float(stacking_weight_min_threshold) >= 1.0:
        raise ValueError("stacking_weight_min_threshold must be in [0, 1).")

    runs_df = _normalize_runs_df(
        _load_runs_df(result_root=result_root, data_id=data_id, split_id=split_id, columns=None)
    )
    tested_config_ids = _load_tested_config_ids(result_root=result_root, data_id=data_id, split_id=split_id)
    if int(skip_first_folds) > 0:
        runs_df = runs_df.loc[runs_df["fold_id"] >= int(skip_first_folds), :].copy()
    if runs_df.empty:
        raise RuntimeError("No CV rows remain after applying skip_first_folds.")

    summary_rows: List[Dict[str, Any]] = []
    fold_rows: List[Dict[str, Any]] = []
    selected_meta: Dict[str, Any] = {}
    out_dir = Path(result_root) / "analysis" / f"data_id={data_id}" / f"split_id={split_id}" / "simple_model_selection"
    out_dir.mkdir(parents=True, exist_ok=True)

    penalized_methods = ["constrained", "utopia", "nash"] if selection_method == "both" else [selection_method]
    use_artifact_suffix = len(constraint_metric_sets) > 1
    for constraint_metric_ids in constraint_metric_sets:
        constraint_metrics_label = _format_constraint_metrics_label(constraint_metric_ids)
        constraint_metrics_slug = _constraint_metrics_slug(constraint_metric_ids)
        artifact_suffix = constraint_metrics_slug if use_artifact_suffix else ""

        for method in penalized_methods:
            if method == "constrained":
                selected = _select_for_family(
                    runs_df,
                    family_name=_COMBINED_SELECTION_GROUP,
                    eligible_config_ids=tested_config_ids,
                    constraint_metrics=constraint_metric_ids,
                    constraint_aggregation=constraint_aggregation,
                    level_constraint_aggregation=level_constraint_aggregation,
                    dispersion_constraint_aggregation=dispersion_constraint_aggregation,
                    selection_metric_id=selection_metric_id,
                    selection_aggregation=selection_aggregation,
                    apply_constraints=True,
                    selection_method="constrained",
                    utopia_metric_aggregation="",
                )
            elif method == "utopia":
                selected = _select_for_family_utopia(
                    runs_df,
                    family_name=_COMBINED_SELECTION_GROUP,
                    eligible_config_ids=tested_config_ids,
                    constraint_metrics=constraint_metric_ids,
                    selection_metric_id=selection_metric_id,
                    utopia_metric_aggregation=utopia_metric_aggregation,
                    level_constraint_aggregation=level_constraint_aggregation,
                    dispersion_constraint_aggregation=dispersion_constraint_aggregation,
                )
            else:
                selected = _select_for_family_nash(
                    runs_df,
                    family_name=_COMBINED_SELECTION_GROUP,
                    eligible_config_ids=tested_config_ids,
                    constraint_metrics=constraint_metric_ids,
                    selection_metric_id=selection_metric_id,
                    nash_metric_aggregation=utopia_metric_aggregation,
                    level_constraint_aggregation=level_constraint_aggregation,
                    dispersion_constraint_aggregation=dispersion_constraint_aggregation,
                )
            selected = _attach_constraint_subset_metadata(
                selected,
                constraint_metrics=constraint_metric_ids,
            )
            summary_rows.append(selected["selected_summary_row"])
            fold_rows.extend(selected["selected_fold_rows"])
            selected_meta[f"{constraint_metrics_slug}:{_COMBINED_SELECTION_GROUP}:{method}"] = {
                "status": selected["status"],
                "selected_config_id": selected["selected_config_id"],
                "constraint_metrics": list(constraint_metric_ids),
                "constraint_metrics_label": constraint_metrics_label,
                "constraint_metrics_slug": constraint_metrics_slug,
            }

        if bool(run_convex_stacking):
            stacked = _select_convex_stacking_for_family(
                runs_df=runs_df,
                result_root=result_root,
                data_id=data_id,
                split_id=split_id,
                output_dir=out_dir,
                family_name=_COMBINED_STACKING_GROUP,
                eligible_config_ids=tested_config_ids,
                constraint_metrics=constraint_metric_ids,
                constraint_aggregation=constraint_aggregation,
                level_constraint_aggregation=level_constraint_aggregation,
                dispersion_constraint_aggregation=dispersion_constraint_aggregation,
                selection_metric_id=selection_metric_id,
                selection_aggregation=selection_aggregation,
                solver=convex_stacking_solver,
                weight_min_threshold=float(stacking_weight_min_threshold),
                artifact_suffix=artifact_suffix,
            )
            stacked = _attach_constraint_subset_metadata(
                stacked,
                constraint_metrics=constraint_metric_ids,
            )
            summary_rows.append(stacked["selected_summary_row"])
            fold_rows.extend(stacked["selected_fold_rows"])
            selected_meta[f"{constraint_metrics_slug}:{_COMBINED_STACKING_GROUP}:convex_stacking"] = {
                "status": stacked["status"],
                "selected_config_id": None,
                "stacking_solution_name": str(stacked["selected_summary_row"].get("stacking_solution_name", "")),
                "stacking_weights_csv": str(stacked["selected_summary_row"].get("stacking_weights_csv", "")),
                "constraint_metrics": list(constraint_metric_ids),
                "constraint_metrics_label": constraint_metrics_label,
                "constraint_metrics_slug": constraint_metrics_slug,
            }

        if bool(run_linearized_stacking):
            stacked = _select_linearized_stacking_for_family(
                runs_df=runs_df,
                output_dir=out_dir,
                family_name=_COMBINED_STACKING_GROUP,
                eligible_config_ids=tested_config_ids,
                constraint_metrics=constraint_metric_ids,
                constraint_aggregation=constraint_aggregation,
                level_constraint_aggregation=level_constraint_aggregation,
                dispersion_constraint_aggregation=dispersion_constraint_aggregation,
                selection_metric_id=selection_metric_id,
                selection_aggregation=selection_aggregation,
                solver=linearized_stacking_solver,
                weight_min_threshold=float(stacking_weight_min_threshold),
                artifact_suffix=artifact_suffix,
            )
            stacked = _attach_constraint_subset_metadata(
                stacked,
                constraint_metrics=constraint_metric_ids,
            )
            summary_rows.append(stacked["selected_summary_row"])
            fold_rows.extend(stacked["selected_fold_rows"])
            selected_meta[f"{constraint_metrics_slug}:{_COMBINED_STACKING_GROUP}:linearized_stacking"] = {
                "status": stacked["status"],
                "selected_config_id": None,
                "stacking_solution_name": str(stacked["selected_summary_row"].get("stacking_solution_name", "")),
                "stacking_weights_csv": str(stacked["selected_summary_row"].get("stacking_weights_csv", "")),
                "constraint_metrics": list(constraint_metric_ids),
                "constraint_metrics_label": constraint_metrics_label,
                "constraint_metrics_slug": constraint_metrics_slug,
            }

    summary_df = pd.DataFrame(summary_rows + fold_rows)
    summary_csv = out_dir / "selection_summary.csv"
    config_json = out_dir / "selection_config.json"
    summary_df.to_csv(summary_csv, index=False)
    config_json.write_text(
        json.dumps(
            {
                "result_root": str(result_root),
                "data_id": str(data_id),
                "split_id": str(split_id),
                "skip_first_folds": int(skip_first_folds),
                "constraint_metrics": list(base_constraint_metric_ids),
                "constraint_metric_subsets": [
                    {
                        "constraint_metrics": list(metric_ids),
                        "constraint_metrics_label": _format_constraint_metrics_label(metric_ids),
                        "constraint_metrics_slug": _constraint_metrics_slug(metric_ids),
                    }
                    for metric_ids in constraint_metric_sets
                ],
                "constraint_aggregation": str(constraint_aggregation),
                "level_constraint_aggregation": (
                    str(level_constraint_aggregation) if level_constraint_aggregation is not None else None
                ),
                "dispersion_constraint_aggregation": (
                    str(dispersion_constraint_aggregation) if dispersion_constraint_aggregation is not None else None
                ),
                "selection_metric": str(selection_metric_id),
                "selection_aggregation": str(selection_aggregation),
                "selection_method": str(selection_method),
                "utopia_metric_aggregation": str(utopia_metric_aggregation),
                "nash_metric_aggregation": str(utopia_metric_aggregation),
                "run_convex_stacking": bool(run_convex_stacking),
                "convex_stacking_solver": (
                    str(convex_stacking_solver).upper().strip() if convex_stacking_solver is not None else None
                ),
                "run_linearized_stacking": bool(run_linearized_stacking),
                "linearized_stacking_solver": (
                    str(linearized_stacking_solver).upper().strip()
                    if linearized_stacking_solver is not None
                    else None
                ),
                "stacking_weight_min_threshold": float(stacking_weight_min_threshold),
                "convex_stacking_supported_selection_metrics": list(_CONVEX_STACKING_SELECTION_METRICS),
                "convex_stacking_supported_constraint_metrics": list(_CONVEX_STACKING_CONSTRAINT_METRICS),
                "selection_requires_held_out_test_metrics": True,
                "tested_config_ids_count": int(len(tested_config_ids)),
                "cod_guidance_class": str(_DEFAULT_COD_GUIDANCE_CLASS),
                "cod_upper_bound": float(_DEFAULT_COD_MAX),
                "cov_upper_bound": float(_DEFAULT_COV_MAX),
                "selected": selected_meta,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "summary_csv": str(summary_csv),
        "config_json": str(config_json),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simple constrained model selection over CV fold metrics.")
    parser.add_argument("--result-root", type=str, default="./output/robust_rolling_origin_cv")
    parser.add_argument("--data-id", type=str, required=True)
    parser.add_argument("--split-id", type=str, required=True)
    parser.add_argument(
        "--skip-first-folds",
        type=int,
        default=0,
        help="Ignore the first k folds before constraint checking and ranking.",
    )
    parser.add_argument(
        "--constraint-metrics",
        type=str,
        default=",".join(_DEFAULT_CONSTRAINT_METRICS),
        help=(
            "Comma-separated list of constraint identifiers. "
            "Valid options: PRD, PRB, VEI, COD, MEAN_RATIO, MEDIAN_RATIO, WEIGHTED_MEAN_RATIO, COV."
        ),
    )
    parser.add_argument(
        "--constraint-metric-subsets",
        type=str,
        default=None,
        help=(
            "Optional semicolon-separated list of constraint-metric subsets to run in one invocation. "
            "Example: 'PRD;PRD,PRB;PRD,PRB,VEI'. If omitted, the single subset from --constraint-metrics is used."
        ),
    )
    parser.add_argument(
        "--constraint-aggregation",
        type=str,
        default="average_fold",
        choices=["average_fold", "worst_fold"],
        help="How to aggregate constraints across folds before filtering.",
    )
    parser.add_argument(
        "--level-constraint-aggregation",
        type=str,
        default=None,
        help=(
            "Optional fold aggregation override for ratio assessment level constraints "
            "(MEAN_RATIO, MEDIAN_RATIO, WEIGHTED_MEAN_RATIO). "
            "Valid options: average_fold, worst_fold."
        ),
    )
    parser.add_argument(
        "--dispersion-constraint-aggregation",
        type=str,
        default=None,
        help=(
            "Optional fold aggregation override for dispersion constraints (COD, COV). "
            "Valid options: average_fold, worst_fold."
        ),
    )
    parser.add_argument(
        "--selection-metric",
        type=str,
        default=_DEFAULT_SELECTION_METRIC,
        choices=sorted(_SELECTION_METRIC_SPECS.keys()),
        help="Metric used to rank feasible configurations.",
    )
    parser.add_argument(
        "--selection-aggregation",
        type=str,
        default="average_fold",
        choices=["average_fold", "worst_fold"],
        help="How to aggregate the ranking metric across folds.",
    )
    parser.add_argument(
        "--selection-method",
        type=str,
        default=_DEFAULT_SELECTION_METHOD,
        choices=["constrained", "utopia", "nash", "both"],
        help="Penalized-model selection rule to run. 'both' writes constrained, utopia, and nash selections.",
    )
    parser.add_argument(
        "--utopia-metric-aggregation",
        type=str,
        default=_DEFAULT_UTOPIA_AGGREGATION,
        choices=["average_fold", "worst_fold"],
        help="Aggregation used to compute the utopia and nash scores over the selected accuracy metric and requested fairness metrics.",
    )
    parser.add_argument(
        "--run-convex-stacking",
        action="store_true",
        help=(
            "Also solve an exact convex stacking alternative over the simplex for each penalized family. "
            f"Supported selection metrics: {', '.join(_CONVEX_STACKING_SELECTION_METRICS)}. "
            f"Supported constraint metrics: {', '.join(_CONVEX_STACKING_CONSTRAINT_METRICS)}."
        ),
    )
    parser.add_argument(
        "--convex-stacking-solver",
        type=str,
        default=None,
        help="Optional cvxpy solver name for convex stacking, e.g. ECOS, OSQP, SCS, CLARABEL.",
    )
    parser.add_argument(
        "--run-linearized-stacking",
        action="store_true",
        help=(
            "Also solve a metric-space linearized stacking alternative over the simplex for each penalized family. "
            "This treats fold metrics as linear in the model weights and therefore supports the same metric identifiers "
            "as the single-model selector."
        ),
    )
    parser.add_argument(
        "--linearized-stacking-solver",
        type=str,
        default=None,
        help="Optional cvxpy solver name for linearized stacking, e.g. ECOS, OSQP, SCS, CLARABEL.",
    )
    parser.add_argument(
        "--stacking-weight-min-threshold",
        type=float,
        default=0.0,
        help=(
            "Post-solve minimum weight retained in either stacking solution. "
            "Weights below this threshold are zeroed out and the remaining weights are renormalized."
        ),
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    constraint_metrics = _parse_identifier_list(args.constraint_metrics, default=_DEFAULT_CONSTRAINT_METRICS)
    constraint_metric_subsets = (
        _parse_constraint_metric_subsets(args.constraint_metric_subsets, default=constraint_metrics)
        if args.constraint_metric_subsets is not None and str(args.constraint_metric_subsets).strip()
        else None
    )
    out = run_simple_model_selection(
        result_root=str(args.result_root),
        data_id=str(args.data_id),
        split_id=str(args.split_id),
        skip_first_folds=int(args.skip_first_folds),
        constraint_metrics=constraint_metrics,
        constraint_metric_subsets=constraint_metric_subsets,
        constraint_aggregation=str(args.constraint_aggregation),
        level_constraint_aggregation=(
            None if args.level_constraint_aggregation is None else str(args.level_constraint_aggregation)
        ),
        dispersion_constraint_aggregation=(
            None if args.dispersion_constraint_aggregation is None else str(args.dispersion_constraint_aggregation)
        ),
        selection_metric=str(args.selection_metric),
        selection_aggregation=str(args.selection_aggregation),
        selection_method=str(args.selection_method),
        utopia_metric_aggregation=str(args.utopia_metric_aggregation),
        run_convex_stacking=bool(args.run_convex_stacking),
        convex_stacking_solver=args.convex_stacking_solver,
        run_linearized_stacking=bool(args.run_linearized_stacking),
        linearized_stacking_solver=args.linearized_stacking_solver,
        stacking_weight_min_threshold=float(args.stacking_weight_min_threshold),
    )
    print("=" * 90)
    print("SIMPLE MODEL SELECTION COMPLETED")
    print("=" * 90)
    print(f"summary_csv={out['summary_csv']}")
    print(f"config_json={out['config_json']}")
