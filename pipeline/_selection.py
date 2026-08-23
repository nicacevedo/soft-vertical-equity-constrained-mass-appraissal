"""
Simplified model selection for the pipeline stage ``02-assess``.

Selections are produced from a finished rolling-origin CV run
(``runs/`` parquet + ``analysis/.../test_metrics.csv``):

1. ``linear_regression`` — the untuned ``LinearRegression`` baseline. It is
   included for reporting but has no hyperparameter / rho selection.

2. ``lgbm_min_rmse`` — pick the ``LGBMRegressor`` configuration with the lowest
   mean fold RMSE (or MSE). This mirrors the Cook County AVM
   ``select_best(lgbm_search, metric = params$cv$best_metric)`` logic: pure
   validation-error minimization.

3. ``cov_penalty_min_mse`` — pick the best ``LGBCovPenalty`` rho within family.

4. ``smooth_identity_min_mse`` — pick the best ``LGBSmoothPenalty`` rho with
   ``weighting_proxy_mode="identity"``.

5. ``smooth_logistic_min_mse`` — pick the best ``LGBSmoothPenalty`` rho with
   ``weighting_proxy_mode="logistic_quantile"``
   (``LGBSmoothPenaltyLogisticProxy`` in the raw run artifacts).

Each selection starts from configs that appear in both CV ``runs/`` and
``test_metrics.csv``, then applies its family filter.

Outputs (under ``analysis/data_id=…/split_id=…/selected/``):

- ``selected_models.json`` — winners + per-fold metrics + held-out test metrics
- ``selected_models.csv``  — flat one-row-per-selection summary table
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from utils.motivation_utils import IAAO_PRB_RANGE, IAAO_PRD_RANGE, IAAO_VEI_RANGE


# ----------------------------------------------------------------------------
# Specs (mirrors the small subset of `simple_model_selection.py` we still need)
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class ConstraintSpec:
    column: str
    lower: Optional[float]
    upper: Optional[float]


@dataclass(frozen=True)
class AccuracyMetricSpec:
    metric_id: str
    column: str
    higher_is_better: bool
    transform: str = "identity"  # "identity" or "square" (RMSE -> MSE)


CONSTRAINT_SPECS: Dict[str, ConstraintSpec] = {
    "PRD": ConstraintSpec(column="PRD", lower=float(IAAO_PRD_RANGE[0]), upper=float(IAAO_PRD_RANGE[1])),
    "PRB": ConstraintSpec(column="PRB", lower=float(IAAO_PRB_RANGE[0]), upper=float(IAAO_PRB_RANGE[1])),
    "VEI": ConstraintSpec(column="VEI", lower=float(IAAO_VEI_RANGE[0]), upper=float(IAAO_VEI_RANGE[1])),
}

ACCURACY_SPECS: Dict[str, AccuracyMetricSpec] = {
    "RMSE": AccuracyMetricSpec("RMSE", "RMSE", higher_is_better=False, transform="identity"),
    "MSE": AccuracyMetricSpec("MSE", "RMSE", higher_is_better=False, transform="square"),
    "MAE": AccuracyMetricSpec("MAE", "MAE", higher_is_better=False),
    "R2": AccuracyMetricSpec("R2", "R2", higher_is_better=True),
}

DEFAULT_CONSTRAINT_METRICS: Tuple[str, ...] = ("PRD",)
DEFAULT_ACCURACY_METRIC: str = "RMSE"
DEFAULT_PENALIZED_SELECTION_MODE: str = "mse_only"
DEFAULT_NASH_ACCURACY_METRIC: str = "RMSE"
DEFAULT_NASH_CONSTRAINT_METRICS: Tuple[str, ...] = ("PRD",)
_POSITIVE_EPS: float = 1e-12

# Default candidate pools per selection rule. CCAO picks the best LightGBM
# tuning trial regardless of fairness; here that pool is strictly the baseline
# LightGBM family, while the penalized families are selected independently.
DEFAULT_CCAO_FAMILIES: Tuple[str, ...] = ("LGBMRegressor",)
DEFAULT_UTOPIA_FAMILIES: Tuple[str, ...] = ("LGBCovPenalty", "LGBSmoothPenalty", "LGBSmoothPenaltyLogisticProxy")
DEFAULT_NASH_FAMILIES: Tuple[str, ...] = ("LGBCovPenalty",)
SMOOTH_IDENTITY_FAMILIES: Tuple[str, ...] = ("LGBSmoothPenalty",)
SMOOTH_LOGISTIC_FAMILIES: Tuple[str, ...] = ("LGBSmoothPenaltyLogisticProxy",)
SMOOTH_NASH_FAMILIES: Tuple[str, ...] = SMOOTH_LOGISTIC_FAMILIES


# ----------------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------------


def _runs_dir(result_root: Path, data_id: str, split_id: str) -> Path:
    return result_root / "runs" / f"data_id={data_id}" / f"split_id={split_id}"


def _analysis_dir(result_root: Path, data_id: str, split_id: str) -> Path:
    return result_root / "analysis" / f"data_id={data_id}" / f"split_id={split_id}"


def load_runs_df(result_root: Path, data_id: str, split_id: str) -> pd.DataFrame:
    runs_dir = _runs_dir(result_root, data_id, split_id)
    paths = sorted(runs_dir.rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No CV run parquet files found under {runs_dir}")
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def load_test_metrics(result_root: Path, data_id: str, split_id: str) -> pd.DataFrame:
    path = _analysis_dir(result_root, data_id, split_id) / "test_metrics.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Held-out test metrics not found: {path}")
    return pd.read_csv(path)


# ----------------------------------------------------------------------------
# Aggregation helpers
# ----------------------------------------------------------------------------


def _model_family(model_name: str) -> str:
    if not isinstance(model_name, str):
        return ""
    return model_name.split("[", 1)[0]


def _per_config_fold_stats(
    runs_df: pd.DataFrame,
    *,
    accuracy: AccuracyMetricSpec,
    constraint_ids: Sequence[str],
) -> pd.DataFrame:
    needed = {"config_id", "fold_id", "model_name", accuracy.column}
    needed |= {CONSTRAINT_SPECS[c].column for c in constraint_ids}
    missing = needed - set(runs_df.columns)
    if missing:
        raise KeyError(f"Missing required columns in runs dataframe: {sorted(missing)}")

    df = runs_df.copy()
    df["config_id"] = df["config_id"].astype(str)
    df["model_name"] = df["model_name"].astype(str)
    df["model_family"] = df["model_name"].map(_model_family)

    acc_vals = pd.to_numeric(df[accuracy.column], errors="coerce")
    if accuracy.transform == "square":
        acc_vals = acc_vals.pow(2)
    df["_acc"] = acc_vals

    rows: List[Dict[str, Any]] = []
    for cfg, sub in df.groupby("config_id", sort=False):
        row: Dict[str, Any] = {
            "config_id": cfg,
            "model_name": str(sub["model_name"].iloc[0]),
            "model_family": str(sub["model_family"].iloc[0]),
            "n_folds": int(sub["fold_id"].nunique()),
            "model_config_json": _first_str(sub.get("model_config_json")),
        }
        acc_arr = sub["_acc"].to_numpy(dtype=float)
        acc_arr = acc_arr[np.isfinite(acc_arr)]
        if acc_arr.size == 0:
            row[f"{accuracy.metric_id}_mean"] = np.nan
            row[f"{accuracy.metric_id}_std"] = np.nan
            row[f"{accuracy.metric_id}_max"] = np.nan
            row[f"{accuracy.metric_id}_min"] = np.nan
        else:
            row[f"{accuracy.metric_id}_mean"] = float(np.mean(acc_arr))
            row[f"{accuracy.metric_id}_std"] = float(np.std(acc_arr, ddof=0))
            row[f"{accuracy.metric_id}_max"] = float(np.max(acc_arr))
            row[f"{accuracy.metric_id}_min"] = float(np.min(acc_arr))
        for cid in constraint_ids:
            spec = CONSTRAINT_SPECS[cid]
            vals = pd.to_numeric(sub[spec.column], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            row[f"{cid}_mean"] = float(np.mean(vals)) if vals.size else np.nan
            row[f"{cid}_std"] = float(np.std(vals, ddof=0)) if vals.size else np.nan
            row[f"{cid}_in_bounds_mean"] = bool(
                vals.size > 0
                and (spec.lower is None or float(np.mean(vals)) >= spec.lower)
                and (spec.upper is None or float(np.mean(vals)) <= spec.upper)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _first_str(series: Any) -> str:
    if series is None:
        return ""
    try:
        s = pd.Series(series).dropna().astype(str)
    except Exception:
        return ""
    return "" if s.empty else str(s.iloc[0])


# ----------------------------------------------------------------------------
# Selection rules
# ----------------------------------------------------------------------------


def _normalize_preference(values: np.ndarray) -> np.ndarray:
    """Map a vector of "preference" values to [0, 1] (higher = better)."""
    out = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return out
    finite_vals = values[finite]
    lo = float(np.min(finite_vals))
    hi = float(np.max(finite_vals))
    if abs(hi - lo) <= 1e-12:
        out[finite] = 1.0
        return out
    out[finite] = (finite_vals - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def _constraint_target(spec: ConstraintSpec) -> float:
    if spec.lower is not None and spec.upper is not None:
        if spec.lower <= 0.0 <= spec.upper:
            return 0.0
        if spec.lower <= 1.0 <= spec.upper:
            return 1.0
        return float(0.5 * (spec.lower + spec.upper))
    if spec.lower is not None:
        return float(spec.lower)
    if spec.upper is not None:
        return float(spec.upper)
    return float("nan")


def _constraint_preference(value: float, spec: ConstraintSpec) -> float:
    if not np.isfinite(value):
        return float("nan")
    target = _constraint_target(spec)
    if spec.lower is not None and spec.upper is not None:
        return float(-abs(value - target))
    if spec.upper is not None:
        return float(-value)
    if spec.lower is not None:
        return float(value)
    return float(value)


def _positive_accuracy_utility(value: float, spec: AccuracyMetricSpec) -> float:
    if not np.isfinite(value):
        return float("nan")
    val = float(value)
    if spec.higher_is_better:
        if spec.column == "R2":
            return float(max(1.0 + val, _POSITIVE_EPS))
        return float(max(val, _POSITIVE_EPS))
    return float(1.0 / max(val, _POSITIVE_EPS))


def _positive_constraint_utility(value: float, spec: ConstraintSpec) -> float:
    if not np.isfinite(value):
        return float("nan")
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


def select_min_rmse(stats_df: pd.DataFrame, *, accuracy: AccuracyMetricSpec) -> Dict[str, Any]:
    col = f"{accuracy.metric_id}_mean"
    primary = pd.to_numeric(stats_df[col], errors="coerce")
    if primary.notna().sum() == 0:
        raise RuntimeError(f"No finite values for {col} in candidate pool.")
    best_val = float(primary.min()) if not accuracy.higher_is_better else float(primary.max())
    tol = 1e-6 * max(abs(best_val), 1.0)
    if accuracy.higher_is_better:
        tied_mask = primary >= (best_val - tol)
    else:
        tied_mask = primary <= (best_val + tol)
    tied = stats_df.loc[tied_mask].copy()
    tied = tied.sort_values(by=[f"{accuracy.metric_id}_std", "config_id"], ascending=[True, True])
    return tied.iloc[0].to_dict()


def select_utopia(
    stats_df: pd.DataFrame,
    *,
    accuracy: AccuracyMetricSpec,
    constraint_ids: Sequence[str],
) -> Dict[str, Any]:
    df = stats_df.copy()
    acc_col = f"{accuracy.metric_id}_mean"
    acc_vals = pd.to_numeric(df[acc_col], errors="coerce").to_numpy(dtype=float)
    acc_pref = acc_vals if accuracy.higher_is_better else -acc_vals
    df["utopia_acc_score"] = _normalize_preference(acc_pref)

    score_cols = ["utopia_acc_score"]
    for cid in constraint_ids:
        spec = CONSTRAINT_SPECS[cid]
        raw = pd.to_numeric(df[f"{cid}_mean"], errors="coerce").to_numpy(dtype=float)
        pref = np.asarray([_constraint_preference(v, spec) for v in raw], dtype=float)
        df[f"utopia_{cid}_score"] = _normalize_preference(pref)
        score_cols.append(f"utopia_{cid}_score")

    score_mat = df[score_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    df["utopia_distance"] = np.sqrt(np.nanmean((1.0 - score_mat) ** 2, axis=1))
    df = df.sort_values(by=["utopia_distance", f"{accuracy.metric_id}_std", "config_id"], ascending=True)
    return df.iloc[0].to_dict()


def select_nash(
    stats_df: pd.DataFrame,
    *,
    accuracy: AccuracyMetricSpec,
    constraint_ids: Sequence[str],
) -> Dict[str, Any]:
    if stats_df.empty:
        raise RuntimeError("No candidate rows available for Nash selection.")

    df = stats_df.copy()
    acc_col = f"{accuracy.metric_id}_mean"
    acc_vals = pd.to_numeric(df[acc_col], errors="coerce").to_numpy(dtype=float)
    utility_cols = ["nash_accuracy_utility"]
    df["nash_accuracy_utility"] = np.asarray(
        [_positive_accuracy_utility(v, accuracy) for v in acc_vals],
        dtype=float,
    )

    for cid in constraint_ids:
        spec = CONSTRAINT_SPECS[cid]
        raw = pd.to_numeric(df[f"{cid}_mean"], errors="coerce").to_numpy(dtype=float)
        utility_col = f"nash_{cid}_utility"
        df[utility_col] = np.asarray(
            [_positive_constraint_utility(v, spec) for v in raw],
            dtype=float,
        )
        utility_cols.append(utility_col)

    utility_mat = df[utility_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    positive_mask = np.all(np.isfinite(utility_mat) & (utility_mat > 0.0), axis=1)
    df["nash_log_utility"] = np.nan
    if np.any(positive_mask):
        df.loc[positive_mask, "nash_log_utility"] = np.sum(np.log(utility_mat[positive_mask]), axis=1)

    df[f"{accuracy.metric_id}_std"] = pd.to_numeric(df[f"{accuracy.metric_id}_std"], errors="coerce")
    df = df.sort_values(
        by=["nash_log_utility", f"{accuracy.metric_id}_std", "config_id"],
        ascending=[False, True, True],
        na_position="last",
        ignore_index=True,
    )
    if df.empty or not np.isfinite(pd.to_numeric(df["nash_log_utility"], errors="coerce")).any():
        raise RuntimeError("Could not compute a finite Nash log-utility for any candidate.")
    return df.iloc[0].to_dict()


# ----------------------------------------------------------------------------
# Public entrypoint
# ----------------------------------------------------------------------------


def _filter_pool(
    runs_df: pd.DataFrame,
    *,
    eligible_ids: set,
    families: Optional[Sequence[str]],
) -> pd.DataFrame:
    df = runs_df.loc[runs_df["config_id"].astype(str).isin(eligible_ids)].copy()
    if families:
        keep = {f for f in families if f}
        df["model_family"] = df["model_name"].astype(str).map(_model_family)
        df = df.loc[df["model_family"].isin(keep)].copy()
    return df


def _pool_summary(pool: pd.DataFrame, stats: pd.DataFrame) -> Dict[str, Any]:
    return {
        "n_configs": int(stats.shape[0]),
        "families": sorted(stats["model_family"].dropna().astype(str).unique().tolist()) if not stats.empty else [],
        "n_folds": int(pool["fold_id"].nunique()) if "fold_id" in pool.columns and not pool.empty else 0,
    }


def _family_stats(
    runs_df: pd.DataFrame,
    *,
    eligible_ids: set,
    families: Sequence[str],
    label: str,
    accuracy: AccuracyMetricSpec,
    constraint_ids: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pool = _filter_pool(runs_df, eligible_ids=eligible_ids, families=families)
    if pool.empty:
        raise RuntimeError(
            f"{label} candidate pool is empty after filtering by families={list(families)}."
        )
    stats = _per_config_fold_stats(pool, accuracy=accuracy, constraint_ids=constraint_ids)
    return pool, stats


def run_selection(
    *,
    result_root: Path,
    data_id: str,
    split_id: str,
    accuracy_metric: str = DEFAULT_ACCURACY_METRIC,
    constraint_metrics: Sequence[str] = DEFAULT_CONSTRAINT_METRICS,
    ccao_families: Optional[Sequence[str]] = DEFAULT_CCAO_FAMILIES,
    utopia_families: Optional[Sequence[str]] = DEFAULT_UTOPIA_FAMILIES,
    nash_families: Optional[Sequence[str]] = DEFAULT_NASH_FAMILIES,
    penalized_selection_mode: str = DEFAULT_PENALIZED_SELECTION_MODE,
) -> Dict[str, Any]:
    metric_id = accuracy_metric.upper()
    if metric_id not in ACCURACY_SPECS:
        raise ValueError(f"Unknown accuracy metric '{accuracy_metric}'. Choose from {sorted(ACCURACY_SPECS)}.")
    accuracy = ACCURACY_SPECS[metric_id]
    penalized_mode = str(penalized_selection_mode or DEFAULT_PENALIZED_SELECTION_MODE).strip().lower()
    if penalized_mode not in {"mse_only", "nash_only"}:
        raise ValueError("penalized_selection_mode must be one of ['mse_only', 'nash_only'].")
    penalized_accuracy = ACCURACY_SPECS["MSE"] if penalized_mode == "mse_only" else ACCURACY_SPECS[DEFAULT_NASH_ACCURACY_METRIC]

    constraint_ids = [c.upper() for c in constraint_metrics]
    unknown = [c for c in constraint_ids if c not in CONSTRAINT_SPECS]
    if unknown:
        raise ValueError(f"Unknown constraint metric(s) {unknown}. Choose from {sorted(CONSTRAINT_SPECS)}.")
    penalized_constraint_ids = list(constraint_ids) if penalized_mode == "mse_only" else list(DEFAULT_NASH_CONSTRAINT_METRICS)

    runs_df = load_runs_df(result_root, data_id, split_id)
    test_df = load_test_metrics(result_root, data_id, split_id)

    eligible_ids = set(test_df["config_id"].astype(str).tolist())
    runs_df["config_id"] = runs_df["config_id"].astype(str)
    runs_df["model_family"] = runs_df["model_name"].astype(str).map(_model_family)

    linear_pool, linear_stats = _family_stats(
        runs_df,
        eligible_ids=eligible_ids,
        families=("LinearRegression",),
        label="LinearRegression",
        accuracy=accuracy,
        constraint_ids=constraint_ids,
    )
    lgbm_pool, lgbm_stats = _family_stats(
        runs_df,
        eligible_ids=eligible_ids,
        families=tuple(ccao_families or DEFAULT_CCAO_FAMILIES),
        label="LGBMRegressor",
        accuracy=accuracy,
        constraint_ids=constraint_ids,
    )
    cov_pool, cov_stats = _family_stats(
        runs_df,
        eligible_ids=eligible_ids,
        families=tuple(nash_families or DEFAULT_NASH_FAMILIES),
        label="LGBCovPenalty",
        accuracy=penalized_accuracy,
        constraint_ids=penalized_constraint_ids,
    )
    smooth_identity_pool, smooth_identity_stats = _family_stats(
        runs_df,
        eligible_ids=eligible_ids,
        families=SMOOTH_IDENTITY_FAMILIES,
        label="LGBSmoothPenalty identity",
        accuracy=penalized_accuracy,
        constraint_ids=penalized_constraint_ids,
    )
    smooth_logistic_pool, smooth_logistic_stats = _family_stats(
        runs_df,
        eligible_ids=eligible_ids,
        families=SMOOTH_LOGISTIC_FAMILIES,
        label="LGBSmoothPenalty logistic_quantile",
        accuracy=penalized_accuracy,
        constraint_ids=penalized_constraint_ids,
    )

    # LinearRegression has no tuned hyperparameters. If multiple linear rows ever
    # appear, keep the deterministic lowest config_id rather than selecting on
    # the held-out test set.
    pick_linear = linear_stats.sort_values("config_id", ascending=True).iloc[0].to_dict()
    pick_lgbm = select_min_rmse(lgbm_stats, accuracy=accuracy)
    if penalized_mode == "mse_only":
        pick_cov = select_min_rmse(cov_stats, accuracy=penalized_accuracy)
        pick_smooth_identity = select_min_rmse(smooth_identity_stats, accuracy=penalized_accuracy)
        pick_smooth_logistic = select_min_rmse(smooth_logistic_stats, accuracy=penalized_accuracy)
    else:
        pick_cov = select_nash(cov_stats, accuracy=penalized_accuracy, constraint_ids=penalized_constraint_ids)
        pick_smooth_identity = select_nash(
            smooth_identity_stats,
            accuracy=penalized_accuracy,
            constraint_ids=penalized_constraint_ids,
        )
        pick_smooth_logistic = select_nash(
            smooth_logistic_stats,
            accuracy=penalized_accuracy,
            constraint_ids=penalized_constraint_ids,
        )

    test_lookup = test_df.set_index(test_df["config_id"].astype(str)).to_dict(orient="index")

    out_dir = _analysis_dir(result_root, data_id, split_id) / "selected"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "result_root": str(result_root),
        "data_id": data_id,
        "split_id": split_id,
        "accuracy_metric": metric_id,
        "constraint_metrics": constraint_ids,
        "penalized_selection_mode": penalized_mode,
        "penalized_accuracy_metric": penalized_accuracy.metric_id,
        "penalized_constraint_metrics": penalized_constraint_ids,
        "nash_accuracy_metric": penalized_accuracy.metric_id,
        "nash_constraint_metrics": penalized_constraint_ids,
        "candidate_pools": {
            "linear_regression": _pool_summary(linear_pool, linear_stats),
            "lgbm_min_rmse": _pool_summary(lgbm_pool, lgbm_stats),
            "cov_penalty_min_mse": _pool_summary(cov_pool, cov_stats),
            "smooth_identity_min_mse": _pool_summary(smooth_identity_pool, smooth_identity_stats),
            "smooth_logistic_min_mse": _pool_summary(smooth_logistic_pool, smooth_logistic_stats),
        },
        "selections": {
            "linear_regression": _selection_record(
                rule="linear_regression",
                row=pick_linear,
                test_metrics=test_lookup.get(str(pick_linear["config_id"]), {}),
                accuracy=accuracy,
                constraint_ids=constraint_ids,
                extra={
                    "selector_rule": "untuned_baseline",
                    "selector_label": "Untuned LinearRegression",
                    "selector_description": (
                        "LinearRegression has no hyperparameter or rho sweep; it is included as the fixed "
                        "linear baseline and is not selected using held-out test performance."
                    ),
                },
            ),
            "lgbm_min_rmse": _selection_record(
                rule="lgbm_min_rmse",
                row=pick_lgbm,
                test_metrics=test_lookup.get(str(pick_lgbm["config_id"]), {}),
                accuracy=accuracy,
                constraint_ids=constraint_ids,
                extra={
                    "selector_rule": "lgbm_min_validation_error",
                    "selector_label": f"LGBM min {accuracy.metric_id}",
                    "selector_description": (
                        f"Configuration with the minimum mean fold {accuracy.metric_id} within the "
                        "baseline LGBMRegressor family."
                    ),
                },
            ),
            "cov_penalty_min_mse": _selection_record(
                rule="cov_penalty_min_mse",
                row=pick_cov,
                test_metrics=test_lookup.get(str(pick_cov["config_id"]), {}),
                accuracy=penalized_accuracy,
                constraint_ids=penalized_constraint_ids,
                extra=_penalized_selector_extra(
                    base_rule="cov_penalty_min_mse",
                    penalized_mode=penalized_mode,
                    row=pick_cov,
                ),
            ),
            "smooth_identity_min_mse": _selection_record(
                rule="smooth_identity_min_mse",
                row=pick_smooth_identity,
                test_metrics=test_lookup.get(str(pick_smooth_identity["config_id"]), {}),
                accuracy=penalized_accuracy,
                constraint_ids=penalized_constraint_ids,
                extra=_penalized_selector_extra(
                    base_rule="smooth_identity_min_mse",
                    penalized_mode=penalized_mode,
                    row=pick_smooth_identity,
                ),
            ),
            "smooth_logistic_min_mse": _selection_record(
                rule="smooth_logistic_min_mse",
                row=pick_smooth_logistic,
                test_metrics=test_lookup.get(str(pick_smooth_logistic["config_id"]), {}),
                accuracy=penalized_accuracy,
                constraint_ids=penalized_constraint_ids,
                extra=_penalized_selector_extra(
                    base_rule="smooth_logistic_min_mse",
                    penalized_mode=penalized_mode,
                    row=pick_smooth_logistic,
                ),
            ),
        },
    }

    json_path = out_dir / "selected_models.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")

    flat = pd.DataFrame(list(payload["selections"].values()))
    csv_path = out_dir / "selected_models.csv"
    flat.to_csv(csv_path, index=False)

    payload["json_path"] = str(json_path)
    payload["csv_path"] = str(csv_path)
    return payload


def _penalized_selector_extra(
    *,
    base_rule: str,
    penalized_mode: str,
    row: Dict[str, Any],
) -> Dict[str, Any]:
    family = str(row.get("model_family", ""))
    family_meta = {
        "cov_penalty_min_mse": ("CovPenalty", "cov_penalty_min_mse"),
        "smooth_identity_min_mse": ("SmoothPenalty identity", "smooth_identity_min_mse"),
        "smooth_logistic_min_mse": ("SmoothPenalty logistic-quantile", "smooth_logistic_min_mse"),
        "nash": ("CovPenalty", "cov_penalty_min_mse"),
        "smooth_penalty_nash": ("SmoothPenalty logistic-quantile", "smooth_logistic_min_mse"),
    }.get(base_rule)
    if family_meta is None:
        family_meta = {
            "LGBCovPenalty": ("CovPenalty", "cov_penalty_min_mse"),
            "LGBSmoothPenalty": ("SmoothPenalty identity", "smooth_identity_min_mse"),
            "LGBSmoothPenaltyLogisticProxy": ("SmoothPenalty logistic-quantile", "smooth_logistic_min_mse"),
        }.get(family, (family or base_rule, f"{family or base_rule}_min_mse"))
    family_label, mse_rule = family_meta
    if penalized_mode == "mse_only":
        return {
            "selector_rule": mse_rule,
            "selector_label": f"{family_label} min MSE",
            "selector_description": (
                f"Configuration with the minimum mean fold MSE within the {family_label} family. "
                "This selector uses only predictive error and does not apply fairness-utility aggregation."
            ),
        }
    extra = {
        "selector_rule": f"{base_rule}_nash",
        "selector_label": (
            f"{family_label} Nash"
        ),
        "selector_description": (
            "Penalized-model winner via Nash-style product of utilities: transform fold-mean "
            "RMSE to 1/RMSE and map PRD through an IAAO-band positive utility, then maximize "
            "the sum of log-utilities with no across-candidate normalization."
        ),
        "nash_log_utility": float(row.get("nash_log_utility", np.nan)),
    }
    return extra


def _selection_record(
    *,
    rule: str,
    row: Dict[str, Any],
    test_metrics: Dict[str, Any],
    accuracy: AccuracyMetricSpec,
    constraint_ids: Sequence[str],
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "rule": rule,
        "config_id": str(row.get("config_id", "")),
        "model_name": str(row.get("model_name", "")),
        "model_family": str(row.get("model_family", "")),
        "n_folds": int(row.get("n_folds", 0)),
        f"cv_{accuracy.metric_id}_mean": _to_float(row.get(f"{accuracy.metric_id}_mean")),
        f"cv_{accuracy.metric_id}_std": _to_float(row.get(f"{accuracy.metric_id}_std")),
        f"cv_{accuracy.metric_id}_max": _to_float(row.get(f"{accuracy.metric_id}_max")),
    }
    for cid in constraint_ids:
        record[f"cv_{cid}_mean"] = _to_float(row.get(f"{cid}_mean"))
        record[f"cv_{cid}_in_bounds_mean"] = bool(row.get(f"{cid}_in_bounds_mean", False))
    for col in ("R2", "RMSE", "MAE", "MAPE", "PRD", "PRB", "VEI", "COD", "Mean ratio", "Median ratio"):
        if col in test_metrics:
            record[f"test_{col}"] = _to_float(test_metrics.get(col))
    record["model_config_json"] = str(row.get("model_config_json", ""))
    record.update(extra)
    return record


def _to_float(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v if np.isfinite(v) else float("nan")
