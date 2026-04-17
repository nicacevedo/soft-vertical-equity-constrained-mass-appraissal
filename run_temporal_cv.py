"""
Run the robust rolling-origin CV pipeline + held-out test evaluation.

This is the main entry point for generating CV artifacts under:
  `output/robust_rolling_origin_cv/`

It also writes held-out test artifacts under:
  `output/robust_rolling_origin_cv/analysis/data_id=.../split_id=.../`
    - `test_metrics.csv`
    - `test_predictions.parquet`
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression

try:
    import lightgbm as lgb
except ImportError as e:  # pragma: no cover
    raise ImportError("`run_temporal_cv.py` requires lightgbm. Install via `pip install lightgbm`.") from e

from preprocessing.recipes_pipelined import build_model_pipeline
from soft_constrained_models.boosting_models import (
    LGBCovPenalty,
    # LGBCovPenaltyCVaR,
    LGBCovPenaltyCVaRTotal,
    LGBSmoothPenalty,
    LGBSmoothPenaltyCVaR,
    LGBSmoothPenaltyCVaRTotal,
)
from utils.motivation_utils import _compute_extended_metrics, _stable_hash, run_robust_rolling_origin_cv


_ASSESSMENT_YEAR: int = 2024  # Calendar year used as the held-out assessment block.
_LOG_T0 = time.perf_counter()


def _log(message: str, **fields: Any) -> None:
    dt = time.perf_counter() - _LOG_T0
    suffix = ""
    if fields:
        suffix = " | " + " | ".join(f"{k}={v}" for k, v in fields.items())
    print(f"[run_temporal_cv +{dt:8.1f}s] {message}{suffix}", flush=True)


def _write_json_atomic(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=str)
    tmp_path.replace(path)


def _write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def _write_parquet_atomic(df: pd.DataFrame, path: Path, *, engine: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp_path, index=False, engine=engine)
    tmp_path.replace(path)


def _read_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _estimate_dataframe_bytes(df: pd.DataFrame) -> int:
    try:
        return int(df.memory_usage(index=True, deep=True).sum())
    except Exception:
        try:
            return int(df.memory_usage(index=True).sum())
        except Exception:
            return 0


def _available_memory_bytes() -> Optional[int]:
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        avail_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None
    if page_size <= 0 or avail_pages <= 0:
        return None
    return int(page_size * avail_pages)


def _resolve_held_out_worker_count(
    *,
    pending_models: int,
    parallel_enabled: bool,
    parallel_cpu_fraction: float,
    parallel_max_workers: Optional[int],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_log: np.ndarray,
    y_test_log: np.ndarray,
) -> Dict[str, Any]:
    if (not parallel_enabled) or int(pending_models) <= 1:
        return {
            "workers": 1,
            "cpu_limit": 1,
            "memory_limit": 1,
            "safe_cap": 1,
            "estimated_bytes_per_worker": 0,
            "available_memory_bytes": _available_memory_bytes(),
        }

    cpu_count = max(1, int(os.cpu_count() or 1))
    cpu_limit = max(1, int(np.floor(cpu_count * max(float(parallel_cpu_fraction), 1e-9))))
    if parallel_max_workers is not None:
        cpu_limit = min(cpu_limit, max(1, int(parallel_max_workers)))
    cpu_limit = min(cpu_limit, int(pending_models))

    shared_bytes = (
        _estimate_dataframe_bytes(X_train)
        + _estimate_dataframe_bytes(X_test)
        + int(np.asarray(y_train_log).nbytes)
        + int(np.asarray(y_test_log).nbytes)
    )
    # Conservative per-worker estimate: one extra working copy plus model/prediction buffers.
    estimated_bytes_per_worker = max(512 * 1024 * 1024, int(max(shared_bytes, 1) * 2.0))

    available_bytes = _available_memory_bytes()
    if available_bytes is None:
        memory_limit = min(cpu_limit, 2)
    else:
        usable_bytes = max(0, int(available_bytes * 0.5))
        memory_limit = max(1, usable_bytes // estimated_bytes_per_worker)
        memory_limit = min(memory_limit, cpu_limit)

    # Additional hard cap to avoid oversubscribing large machines in the held-out stage.
    safe_cap = 8
    workers = max(1, min(cpu_limit, memory_limit, safe_cap))
    return {
        "workers": int(workers),
        "cpu_limit": int(cpu_limit),
        "memory_limit": int(max(1, memory_limit)),
        "safe_cap": int(safe_cap),
        "estimated_bytes_per_worker": int(estimated_bytes_per_worker),
        "available_memory_bytes": available_bytes,
    }


def _first_bad_numeric_value(payload: Dict[str, Any], *, abs_cap: float) -> Optional[Dict[str, Any]]:
    cap = float(abs_cap)
    if not np.isfinite(cap) or cap <= 0.0:
        return None
    for k, v in dict(payload).items():
        if isinstance(v, (bool, np.bool_)):
            continue
        if isinstance(v, (int, float, np.integer, np.floating)):
            fv = float(v)
            if not np.isfinite(fv):
                return {"field": str(k), "value": fv, "reason": "non_finite"}
            if abs(fv) > cap:
                return {"field": str(k), "value": fv, "reason": "abs_gt_cap"}
    return None


def _numeric_guard_fields(
    *,
    bad: Optional[Dict[str, Any]],
    stage: str = "",
    cv_flagged: bool = False,
) -> Dict[str, Any]:
    flagged = bool(cv_flagged) or (bad is not None)
    guard_stage = ""
    if bad is not None:
        guard_stage = str(stage)
    elif cv_flagged:
        guard_stage = "cv_prior_warning"
    return {
        "numeric_stability_status": "flagged" if flagged else "stable",
        "numeric_guard_flagged": bool(flagged),
        "numeric_guard_stage": guard_stage,
        "numeric_guard_field": str((bad or {}).get("field", (bad or {}).get("metric", ""))),
        "numeric_guard_value": (bad or {}).get("value", np.nan),
        "numeric_guard_reason": str((bad or {}).get("reason", "")),
        "cv_numeric_warning_flagged": bool(cv_flagged),
    }


def _parse_float_list(values: str) -> List[float]:
    if values.strip() == "":
        return []
    return [float(x) for x in values.split(",")]


def _parse_ratio_mode_list(values: str) -> List[str]:
    modes = [str(x).strip().lower() for x in str(values).split(",") if str(x).strip()]
    if not modes:
        return ["diff"]
    out: List[str] = []
    seen = set()
    for mode in modes:
        if mode not in {"div", "diff"}:
            raise ValueError("ratio_modes must contain only 'div' and/or 'diff'.")
        if mode not in seen:
            out.append(mode)
            seen.add(mode)
    return out


def _build_rho_values(
    rho_values_raw: List[float],
    *,
    rho_count: int,
    rho_scale: str,
) -> List[float]:
    """
    Build rho sweep values from either:
      - range form [rho_min, rho_max] + (count, scale), or
      - explicit list (backward-compatible fallback).
    """
    vals = [float(x) for x in rho_values_raw]
    if not vals:
        return []

    count = int(rho_count)
    if count < 1:
        raise ValueError("rho_count must be >= 1.")

    scale = str(rho_scale).strip().lower()
    if scale not in {"linear", "log", "geom"}:
        raise ValueError("rho_scale must be one of: linear, log, geom.")

    # Preferred new behavior: 2-point range [min, max].
    if len(vals) == 2:
        lo, hi = float(vals[0]), float(vals[1])
        if count == 1:
            return [lo]
        if scale == "linear":
            out = np.linspace(lo, hi, count, dtype=float)
        else:
            if lo <= 0.0 or hi <= 0.0:
                raise ValueError("For rho_scale=log/geom, rho range bounds must be > 0.")
            out = np.geomspace(lo, hi, count, dtype=float)
        return [float(x) for x in out.tolist()]

    # Backward-compatible fallback: explicit list passthrough.
    return vals


# LightGBM's own built-in defaults for keys that model_params.yaml may not specify.
# Used when use_ccao_fallback=False (the default) so that only model_params.yaml
# drives behaviour and there is no silent override from params.yaml.
_LGBM_NATIVE_DEFAULTS: Dict[str, Any] = {
    "num_iterations": 100,
    "learning_rate": 0.1,
    "num_leaves": 31,
    "max_bin": 255,
    "min_gain_to_split": 0.0,    # LightGBM native default; params.yaml uses 75.5 (strongly regularising)
    "min_data_in_leaf": 20,
    "feature_fraction": 1.0,
    "lambda_l1": 0.0,
    "lambda_l2": 0.0,
    "max_cat_threshold": 32,
    "min_data_per_group": 100,
    "cat_smooth": 10.0,
    "cat_l2": 10.0,
}


def _build_lgbm_params_from_files(
    model_params: dict,
    ccao_params: dict,
    seed: int,
    use_ccao_fallback: bool = False,
) -> dict:
    """
    Build LightGBM params dict.

    When use_ccao_fallback=False (default):
      model_params.yaml is the only source; any missing key falls back to
      LightGBM's own native defaults (_LGBM_NATIVE_DEFAULTS).

    When use_ccao_fallback=True (opt-in):
      model_params.yaml is primary; missing keys fall back to params.yaml's
      hyperparameter.default section (the original CCAO behaviour).
    """
    model_default = dict(model_params.get("LGBMRegressor", {}))
    if use_ccao_fallback:
        hp_default = dict(ccao_params["model"]["hyperparameter"]["default"])
    else:
        hp_default = dict(_LGBM_NATIVE_DEFAULTS)

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


def _sample_uniform_value(bounds: List[float], rng: np.random.Generator) -> float:
    lo, hi = float(bounds[0]), float(bounds[1])
    if hi < lo:
        raise ValueError(f"Invalid bounds: {bounds}")
    return float(rng.uniform(lo, hi))


def _sample_int_value(bounds: List[float], rng: np.random.Generator) -> int:
    lo = int(np.ceil(float(bounds[0])))
    hi = int(np.floor(float(bounds[1])))
    if hi < lo:
        raise ValueError(f"Invalid integer bounds: {bounds}")
    return int(rng.integers(lo, hi + 1))


def _sample_log10_value(bounds: List[float], rng: np.random.Generator) -> float:
    return float(10.0 ** _sample_uniform_value(bounds, rng))


def _sample_baseline_lgbm_candidate(
    *,
    hp_range: Dict[str, List[float]],
    base_lgbm_params: Dict[str, Any],
    rng: np.random.Generator,
) -> Dict[str, Any]:
    num_leaves = _sample_int_value(hp_range["num_leaves"], rng)
    add_to_linked_depth = _sample_int_value(hp_range["add_to_linked_depth"], rng)
    max_depth = int(np.floor(np.log2(max(num_leaves, 2))) + add_to_linked_depth)

    candidate = dict(base_lgbm_params)
    candidate.update(
        {
            "n_estimators": _sample_int_value(hp_range["num_iterations"], rng),
            "learning_rate": _sample_log10_value(hp_range["learning_rate"], rng),
            "max_bin": _sample_int_value(hp_range["max_bin"], rng),
            "num_leaves": int(num_leaves),
            "max_depth": int(max_depth),
            "colsample_bytree": _sample_uniform_value(hp_range["feature_fraction"], rng),
            "min_split_gain": _sample_log10_value(hp_range["min_gain_to_split"], rng),
            "min_child_samples": _sample_int_value(hp_range["min_data_in_leaf"], rng),
            "max_cat_threshold": _sample_int_value(hp_range["max_cat_threshold"], rng),
            "min_data_per_group": _sample_int_value(hp_range["min_data_per_group"], rng),
            "cat_smooth": _sample_uniform_value(hp_range["cat_smooth"], rng),
            "cat_l2": _sample_log10_value(hp_range["cat_l2"], rng),
            "reg_alpha": _sample_log10_value(hp_range["lambda_l1"], rng),
            "reg_lambda": _sample_log10_value(hp_range["lambda_l2"], rng),
        }
    )
    return candidate


def _build_baseline_search_candidates(
    *,
    hp_range: Dict[str, List[float]],
    base_lgbm_params: Dict[str, Any],
    n_random_trials: int,
    seed: int,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = [dict(base_lgbm_params)]
    seen = {json.dumps(base_lgbm_params, sort_keys=True)}
    rng = np.random.default_rng(int(seed))

    while len(candidates) < (1 + max(0, int(n_random_trials))):
        candidate = _sample_baseline_lgbm_candidate(
            hp_range=hp_range,
            base_lgbm_params=base_lgbm_params,
            rng=rng,
        )
        key = json.dumps(candidate, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(candidate)

    return candidates


def _run_baseline_lgbm_search(
    *,
    result_root: str,
    params: dict,
    df_train_validate: pd.DataFrame,
    predictor_cols: List[str],
    categorical_cols: List[str],
    linear_pipeline_builder,
    split_protocol: Dict[str, Any],
    parquet_engine: str,
    parallel_enabled: bool,
    parallel_cpu_fraction: float,
    parallel_max_workers: Optional[int],
    numeric_sanity_abs_cap: float,
    base_lgbm_params: Dict[str, Any],
    data_signature: Dict[str, Any],
    seed: int,
    n_random_trials: Optional[int],
    date_col: str,
    target_col: str,
    fairness_ratio_mode: str,
) -> Dict[str, Any]:
    hp_range = dict(params["model"]["hyperparameter"]["range"])
    random_trials = params.get("cv", {}).get("initial_set", 20) if n_random_trials is None else int(n_random_trials)
    if int(random_trials) < 0:
        raise ValueError("baseline_search_trials must be >= 0.")

    candidates = _build_baseline_search_candidates(
        hp_range=hp_range,
        base_lgbm_params=base_lgbm_params,
        n_random_trials=int(random_trials),
        seed=int(seed),
    )

    search_specs: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    for trial_idx, candidate_params in enumerate(candidates):
        candidate_params = dict(candidate_params)
        config_id = _stable_hash({"model_name": "LGBMRegressor", "config": candidate_params})
        search_specs.append(
            {
                "name": "LGBMRegressor",
                "config": candidate_params,
                "requires_linear_pipeline": False,
                "factory": (lambda candidate_params=candidate_params: lgb.LGBMRegressor(**dict(candidate_params))),
            }
        )
        candidate_rows.append(
            {
                "search_trial_idx": int(trial_idx),
                "is_default_baseline": bool(trial_idx == 0),
                "config_id": str(config_id),
                **candidate_params,
            }
        )

    search_root = str(Path(result_root) / "baseline_lgbm_search")
    _log(
        "starting baseline-only LGBM search",
        search_root=search_root,
        candidates=int(len(search_specs)),
        random_trials=int(random_trials),
    )
    search_out = run_robust_rolling_origin_cv(
        df_train_validate=df_train_validate,
        date_col=date_col,
        target_col=target_col,
        predictor_cols=predictor_cols,
        categorical_cols=categorical_cols,
        model_specs=search_specs,
        linear_pipeline_builder=linear_pipeline_builder,
        result_root=search_root,
        data_signature={**data_signature, "mode": "baseline_lgbm_search"},
        split_protocol=split_protocol,
        bootstrap_protocol={"n_bootstrap": 0, "block_freq": "M", "seed": int(seed)},
        fairness_ratio_mode=fairness_ratio_mode,
        predict_store=False,
        parquet_engine=parquet_engine,
        log_progress=True,
        parallel_enabled=parallel_enabled,
        parallel_cpu_fraction=parallel_cpu_fraction,
        parallel_max_workers=parallel_max_workers,
        parallel_backend="loky",
        numeric_sanity_abs_cap=float(numeric_sanity_abs_cap),
    )

    run_records = search_out.get("run_records", pd.DataFrame())
    failed_records = search_out.get("failed_records", pd.DataFrame())
    flagged_config_ids = {str(x) for x in search_out.get("flagged_config_ids", [])}
    fold_count = int(search_out["fold_count"])
    failed_counts: Dict[str, int] = {}
    if isinstance(failed_records, pd.DataFrame) and (not failed_records.empty) and ("config_id" in failed_records.columns):
        failed_counts = failed_records["config_id"].astype(str).value_counts().to_dict()

    summary_rows: List[Dict[str, Any]] = []
    for candidate_row in candidate_rows:
        config_id = str(candidate_row["config_id"])
        candidate_runs = run_records.loc[run_records["config_id"].astype(str) == config_id, :].copy() if isinstance(run_records, pd.DataFrame) and (not run_records.empty) else pd.DataFrame()
        completed_folds = int(candidate_runs.shape[0])
        mean_rmse = float(candidate_runs["RMSE"].mean()) if ("RMSE" in candidate_runs.columns and completed_folds > 0) else np.nan
        std_rmse = float(candidate_runs["RMSE"].std(ddof=0)) if ("RMSE" in candidate_runs.columns and completed_folds > 0) else np.nan
        flagged = bool(config_id in flagged_config_ids)
        failed_fold_count = int(failed_counts.get(config_id, 0))
        eligible = bool((completed_folds == fold_count) and (failed_fold_count == 0) and (not flagged) and np.isfinite(mean_rmse))
        summary_rows.append(
            {
                **candidate_row,
                "completed_folds": int(completed_folds),
                "expected_folds": int(fold_count),
                "failed_folds": int(failed_fold_count),
                "numeric_guard_flagged": bool(flagged),
                "mean_validation_rmse": mean_rmse,
                "std_validation_rmse": std_rmse,
                "eligible_for_selection": bool(eligible),
                "selection_objective": mean_rmse if eligible else np.inf,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["selection_objective", "search_trial_idx"],
        ascending=[True, True],
        na_position="last",
    ).reset_index(drop=True)

    search_analysis_dir = Path(search_root) / "analysis" / f"data_id={search_out['data_id']}" / f"split_id={search_out['split_id']}"
    summary_path = search_analysis_dir / "baseline_lgbm_search_summary.csv"
    best_params_path = search_analysis_dir / "baseline_lgbm_best_params.json"
    metadata_path = search_analysis_dir / "baseline_lgbm_search_metadata.json"

    _write_csv_atomic(summary_df, summary_path)

    best_row = summary_df.loc[summary_df["eligible_for_selection"].fillna(False), :].head(1)
    fallback_used = bool(best_row.empty)
    if fallback_used:
        best_params = dict(base_lgbm_params)
        best_config_id = str(_stable_hash({"model_name": "LGBMRegressor", "config": best_params}))
        best_mean_validation_rmse = np.nan
        _log("baseline-only LGBM search fallback", reason="no_eligible_candidate", config_id=best_config_id)
    else:
        best_record = best_row.iloc[0].to_dict()
        best_config_id = str(best_record["config_id"])
        best_mean_validation_rmse = float(best_record["mean_validation_rmse"])
        best_params = {
            key: best_record[key]
            for key in base_lgbm_params.keys()
            if key in best_record
        }
        _log(
            "baseline-only LGBM search selected best config",
            config_id=best_config_id,
            mean_validation_rmse=f"{best_mean_validation_rmse:.6f}",
        )

    _write_json_atomic(
        best_params_path,
        {
            "search_enabled": True,
            "fallback_used": bool(fallback_used),
            "best_config_id": str(best_config_id),
            "best_mean_validation_rmse": best_mean_validation_rmse,
            "best_lgbm_params": best_params,
        },
    )
    _write_json_atomic(
        metadata_path,
        {
            "search_root": str(search_root),
            "data_id": str(search_out["data_id"]),
            "split_id": str(search_out["split_id"]),
            "fold_count": int(fold_count),
            "n_candidates": int(len(candidates)),
            "n_random_trials": int(random_trials),
            "best_config_id": str(best_config_id),
            "best_mean_validation_rmse": best_mean_validation_rmse,
            "summary_csv": str(summary_path),
            "best_params_json": str(best_params_path),
            "fallback_used": bool(fallback_used),
        },
    )

    return {
        "best_lgbm_params": best_params,
        "search_root": str(search_root),
        "search_summary_csv": str(summary_path),
        "best_params_json": str(best_params_path),
        "search_metadata_json": str(metadata_path),
        "search_data_id": str(search_out["data_id"]),
        "search_split_id": str(search_out["split_id"]),
        "search_n_candidates": int(len(candidates)),
        "search_best_config_id": str(best_config_id),
        "search_best_mean_validation_rmse": best_mean_validation_rmse,
        "search_fallback_used": bool(fallback_used),
    }


def _load_and_split_data(
    *,
    data_path: str,
    params: dict,
    target_column: str,
    date_column: str,
    sample_frac: float | None,
    sample_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """
    Mirrors the split protocol used elsewhere in this repo:
      - assessment: year == 2024
      - pre-2024: split by time into train/validate universe vs held-out test block
    """
    predictor_cols = list(params["model"]["predictor"]["all"])
    categorical_cols = list(params["model"]["predictor"]["categorical"])
    filter_cols = ["ind_pin_is_multicard", "sv_is_outlier"]
    required_cols = list(dict.fromkeys(predictor_cols + [target_column, date_column] + filter_cols))
    row_filters = [
        ("ind_pin_is_multicard", "==", False),
        ("sv_is_outlier", "==", False),
    ]

    load_start = time.perf_counter()
    _log("loading parquet", data_path=data_path, selected_cols=int(len(required_cols)))
    read_engine = "fastparquet"
    pushdown_enabled = False
    pushdown_reason = "pyarrow_unavailable"
    try:
        import pyarrow.dataset as ds
        import pyarrow.types as patypes

        schema = ds.dataset(data_path, format="parquet").schema
        if all(name in schema.names and patypes.is_boolean(schema.field(name).type) for name in filter_cols):
            read_engine = "pyarrow"
            pushdown_enabled = True
            pushdown_reason = "bool_filter_schema"
        else:
            pushdown_reason = "non_boolean_filter_schema"
    except Exception as exc:
        pushdown_reason = f"pushdown_probe_failed:{type(exc).__name__}"

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
        elapsed_sec=f"{time.perf_counter() - load_start:.2f}",
    )

    filter_start = time.perf_counter()
    df = df[(~df["ind_pin_is_multicard"].astype("bool").fillna(True)) & (~df["sv_is_outlier"].astype("bool").fillna(True))]
    _log(
        "row filters applied",
        rows=int(df.shape[0]),
        elapsed_sec=f"{time.perf_counter() - filter_start:.2f}",
    )

    keep_cols = predictor_cols + [target_column, date_column]
    keep_start = time.perf_counter()
    drop_cols = [c for c in filter_cols if c not in keep_cols and c in df.columns]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)
    _log(
        "columns projected",
        kept_cols=int(len(keep_cols)),
        dropped_cols=int(len(drop_cols)),
        elapsed_sec=f"{time.perf_counter() - keep_start:.2f}",
    )

    if sample_frac is not None:
        if not (0.0 < float(sample_frac) <= 1.0):
            raise ValueError("sample_frac must be in (0, 1]. Use None to disable sampling.")
        if float(sample_frac) < 1.0:
            sample_start = time.perf_counter()
            df = df.sample(frac=float(sample_frac), random_state=int(sample_seed))
            _log(
                "sampling applied",
                sample_frac=float(sample_frac),
                rows=int(df.shape[0]),
                elapsed_sec=f"{time.perf_counter() - sample_start:.2f}",
            )

    split_start = time.perf_counter()
    date_parse_start = time.perf_counter()
    date_values = pd.to_datetime(df[date_column])
    df[date_column] = date_values
    date_years = date_values.dt.year.to_numpy(copy=False)
    _log(
        "date column normalized",
        elapsed_sec=f"{time.perf_counter() - date_parse_start:.2f}",
    )

    order_start = time.perf_counter()
    sorted_idx = np.argsort(date_values.to_numpy(copy=False), kind="quicksort")
    sorted_years = date_years[sorted_idx]
    pre2024_sorted_idx = sorted_idx[sorted_years < _ASSESSMENT_YEAR]
    assess_sorted_idx = sorted_idx[sorted_years == _ASSESSMENT_YEAR]
    _log(
        "date ordering prepared",
        pre2024_rows=int(pre2024_sorted_idx.size),
        assess_rows=int(assess_sorted_idx.size),
        elapsed_sec=f"{time.perf_counter() - order_start:.2f}",
    )

    train_prop = float(params["cv"]["split_prop"])
    split_idx = int(train_prop * pre2024_sorted_idx.size)
    df_train_validate = df.iloc[pre2024_sorted_idx[:split_idx], :].copy().reset_index(drop=True)
    df_test = df.iloc[pre2024_sorted_idx[split_idx:], :].copy().reset_index(drop=True)
    df_assess = df.iloc[assess_sorted_idx, :].copy().reset_index(drop=True)
    _log(
        "data split completed",
        train_validate_rows=int(df_train_validate.shape[0]),
        test_rows=int(df_test.shape[0]),
        assess_rows=int(df_assess.shape[0]),
        elapsed_sec=f"{time.perf_counter() - split_start:.2f}",
    )

    return df_train_validate, df_test, df_assess, predictor_cols, categorical_cols


def _build_model_specs(
    *,
    lgbm_params: dict,
    rho_values_smooth: List[float],
    rho_values_cov: List[float],
    keep_values: List[float],
    ratio_modes: List[str],
    fairness_ratio_mode: str,
    include_cvar_models: bool = False,
) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    lgbm_base_config_id = _stable_hash({"lgbm_params": lgbm_params})

    # Baselines
    specs.append(
        {
            "name": "LinearRegression",
            "config": {},
            "requires_linear_pipeline": True,
            "factory": lambda: LinearRegression(fit_intercept=True),
        }
    )
    specs.append(
        {
            "name": "LGBMRegressor",
            "config": {"lgbm_base_config_id": lgbm_base_config_id},
            "requires_linear_pipeline": False,
            "factory": lambda: lgb.LGBMRegressor(**dict(lgbm_params)),
        }
    )

    # Soft penalty variants (rho sweep)
    ratio_mode_sweep = [str(m) for m in ratio_modes] if ratio_modes else [str(fairness_ratio_mode)]
    for ratio_mode in ratio_mode_sweep:
        for rho in rho_values_smooth:
            r = float(rho)
            specs.append(
                {
                    "name": "LGBSmoothPenalty",
                    "config": {"rho": r, "ratio_mode": ratio_mode, "lgbm_base_config_id": lgbm_base_config_id},
                    "metric_ratio_mode": ratio_mode,
                    "requires_linear_pipeline": False,
                    "factory": (
                        lambda rho=r, ratio_mode=ratio_mode: LGBSmoothPenalty(
                            rho=rho,
                            ratio_mode=ratio_mode,
                            zero_grad_tol=1e-12,
                            lgbm_params=dict(lgbm_params),
                            verbose=False,
                        )
                    ),
                }
            )
            if bool(include_cvar_models):
                keep_sweep = [float(k) for k in keep_values] if keep_values else [1.0]
                for keep in keep_sweep:
                    k = float(keep)
                    specs.append(
                        {
                            "name": "LGBSmoothPenaltyCVaR",
                            "config": {"rho": r, "keep": k, "ratio_mode": ratio_mode, "lgbm_base_config_id": lgbm_base_config_id},
                            "metric_ratio_mode": ratio_mode,
                            "requires_linear_pipeline": False,
                            "factory": (
                                lambda rho=r, keep=k, ratio_mode=ratio_mode: LGBSmoothPenaltyCVaR(
                                    rho=rho,
                                    mse_keep=keep,
                                    ratio_mode=ratio_mode,
                                    zero_grad_tol=1e-12,
                                    lgbm_params=dict(lgbm_params),
                                    verbose=False,
                                )
                            ),
                        }
                    )
                    specs.append(
                        {
                            "name": "LGBSmoothPenaltyCVaRTotal",
                            "config": {"rho": r, "keep": k, "ratio_mode": ratio_mode, "lgbm_base_config_id": lgbm_base_config_id},
                            "metric_ratio_mode": ratio_mode,
                            "requires_linear_pipeline": False,
                            "factory": (
                                lambda rho=r, keep=k, ratio_mode=ratio_mode: LGBSmoothPenaltyCVaRTotal(
                                    rho=rho,
                                    keep=keep,
                                    ratio_mode=ratio_mode,
                                    zero_grad_tol=1e-12,
                                    lgbm_params=dict(lgbm_params),
                                    verbose=False,
                                )
                            ),
                        }
                    )

        for rho in rho_values_cov:
            r = float(rho)
            specs.append(
                {
                    "name": "LGBCovPenalty",
                    "config": {"rho": r, "ratio_mode": ratio_mode, "lgbm_base_config_id": lgbm_base_config_id},
                    "metric_ratio_mode": ratio_mode,
                    "requires_linear_pipeline": False,
                    "factory": (
                        lambda rho=r, ratio_mode=ratio_mode: LGBCovPenalty(
                            rho=rho,
                            ratio_mode=ratio_mode,
                            zero_grad_tol=1e-12,
                            lgbm_params=dict(lgbm_params),
                            verbose=False,
                        )
                    ),
                }
            )
            if bool(include_cvar_models):
                keep_sweep = [float(k) for k in keep_values] if keep_values else [1.0]
                for keep in keep_sweep:
                    k = float(keep)
                    # specs.append( # NOTE: This variant is not stable, since its just one side of the loss. It has overflow issues.
                    #     {
                    #         "name": "LGBCovPenaltyCVaR",
                    #         "config": {"rho": r, "keep": k, "ratio_mode": ratio_mode},
                    #         "metric_ratio_mode": ratio_mode,
                    #         "requires_linear_pipeline": False,
                    #         "factory": (
                    #             lambda rho=r, keep=k, ratio_mode=ratio_mode: LGBCovPenaltyCVaR(
                    #                 rho=rho,
                    #                 mse_keep=keep,
                    #                 ratio_mode=ratio_mode,
                    #                 zero_grad_tol=1e-12,
                    #                 lgbm_params=dict(lgbm_params),
                    #                 verbose=False,
                    #             )
                    #         ),
                    #     }
                    # )
                    specs.append(
                        {
                            "name": "LGBCovPenaltyCVaRTotal",
                            "config": {"rho": r, "keep": k, "ratio_mode": ratio_mode, "lgbm_base_config_id": lgbm_base_config_id},
                            "metric_ratio_mode": ratio_mode,
                            "requires_linear_pipeline": False,
                            "factory": (
                                lambda rho=r, keep=k, ratio_mode=ratio_mode: LGBCovPenaltyCVaRTotal(
                                    rho=rho,
                                    mse_keep=keep,
                                    ratio_mode=ratio_mode,
                                    zero_grad_tol=1e-12,
                                    lgbm_params=dict(lgbm_params),
                                    verbose=False,
                                )
                            ),
                        }
                    )

    # # Primal-dual (CVaR-like) variants (rho × keep sweep)
    # for rho in rho_values:
    #     for keep in keep_values:
    #         r = float(rho)
    #         k = float(keep)
    #         specs.append(
    #             {
    #                 "name": "LGBPrimalDual",
    #                 "config": {"rho": r, "keep": k},
    #                 "requires_linear_pipeline": False,
    #                 "factory": (lambda rho=r, keep=k: LGBPrimalDual(rho=rho, keep=keep, adversary_type="overall", eta_adv=0.1, zero_grad_tol=1e-12, lgbm_params=dict(lgbm_params))),
    #             }
    #         )

    return specs


def _evaluate_models_on_test_set(
    *,
    df_train_validate: pd.DataFrame,
    df_test: pd.DataFrame,
    predictor_cols: List[str],
    categorical_cols: List[str],
    date_col: str,
    target_col: str,
    model_specs: List[Dict[str, Any]],
    linear_pipeline_builder,
    fairness_ratio_mode: str,
    analysis_dir: Path,
    parquet_engine: str,
    numeric_sanity_abs_cap: float,
    parallel_enabled: bool,
    parallel_cpu_fraction: float,
    parallel_max_workers: Optional[int],
    invalid_config_ids: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Fit each config on the full train/validate universe and evaluate once on held-out test.
    """
    analysis_dir.mkdir(parents=True, exist_ok=True)
    test_metrics_path = analysis_dir / "test_metrics.csv"
    test_predictions_path = analysis_dir / "test_predictions.parquet"
    test_meta_path = analysis_dir / "test_eval_metadata.json"
    flagged_path = analysis_dir / "test_flagged_configs.csv"
    legacy_flagged_path = analysis_dir / "test_rejected_configs.csv"
    status_path = analysis_dir / "test_eval_status.json"
    shard_metrics_root = analysis_dir / "test_run_metrics"
    shard_preds_root = analysis_dir / "test_run_predictions"
    shard_status_root = analysis_dir / "test_run_status"
    eval_start = time.perf_counter()

    spec_jobs: List[Dict[str, Any]] = []
    for spec in model_specs:
        model_name = str(spec["name"])
        model_config = dict(spec.get("config", {}))
        metric_ratio_mode = str(spec.get("metric_ratio_mode", model_config.get("ratio_mode", fairness_ratio_mode)))
        config_id = _stable_hash({"model_name": model_name, "config": model_config})
        spec_jobs.append(
            {
                "spec": spec,
                "model_name": model_name,
                "model_config": model_config,
                "model_config_json": json.dumps(model_config, sort_keys=True),
                "metric_ratio_mode": metric_ratio_mode,
                "config_id": config_id,
                "metrics_file": shard_metrics_root / f"{config_id}.parquet",
                "predictions_file": shard_preds_root / f"{config_id}.parquet",
                "status_file": shard_status_root / f"{config_id}.json",
            }
        )

    eval_manifest = {
        "fairness_ratio_mode": fairness_ratio_mode,
        "n_models": int(len(spec_jobs)),
        "config_ids": [str(job["config_id"]) for job in spec_jobs],
        "models": [
            {
                "config_id": str(job["config_id"]),
                "model_name": str(job["model_name"]),
                "ratio_mode": str(job["metric_ratio_mode"]),
                "model_config_json": str(job["model_config_json"]),
            }
            for job in spec_jobs
        ],
    }
    eval_signature = _stable_hash({"stage": "held_out_test", **eval_manifest})

    existing_status = _read_json_if_exists(status_path)
    if (
        existing_status.get("status") == "completed"
        and str(existing_status.get("eval_signature", "")) == str(eval_signature)
        and test_metrics_path.exists()
        and test_predictions_path.exists()
        and test_meta_path.exists()
    ):
        _log(
            "held-out test evaluation already complete; reusing aggregate artifacts",
            n_models=int(len(spec_jobs)),
            analysis_dir=str(analysis_dir),
        )
        out = {"test_metrics_csv": str(test_metrics_path), "test_predictions_parquet": str(test_predictions_path)}
        if flagged_path.exists():
            out["test_flagged_configs_csv"] = str(flagged_path)
        if legacy_flagged_path.exists():
            out["test_rejected_configs_csv"] = str(legacy_flagged_path)
        return out

    _log(
        "starting held-out test evaluation",
        analysis_dir=str(analysis_dir),
        n_models=int(len(model_specs)),
        n_train_validate=int(df_train_validate.shape[0]),
        n_test=int(df_test.shape[0]),
    )

    X_tv = df_train_validate[predictor_cols].copy()
    y_tv_log = np.log(df_train_validate[target_col].to_numpy())
    X_test = df_test[predictor_cols].copy()
    y_test_log = np.log(df_test[target_col].to_numpy())

    cat_cols = [c for c in categorical_cols if c in X_tv.columns]
    for c in cat_cols:
        X_tv[c] = X_tv[c].astype("category")
        X_test[c] = X_test[c].astype("category")

    invalid_set = {str(x) for x in (invalid_config_ids or [])}
    pred_columns = [
        "config_id",
        "model_name",
        "ratio_mode",
        "row_id",
        "sale_date",
        "numeric_stability_status",
        "numeric_guard_flagged",
        "numeric_guard_stage",
        "numeric_guard_field",
        "numeric_guard_value",
        "numeric_guard_reason",
        "cv_numeric_warning_flagged",
        "y_true_log",
        "y_pred_log",
        "y_true",
        "y_pred",
    ]

    reusable_jobs = 0
    pending_jobs: List[Dict[str, Any]] = []
    for job in spec_jobs:
        status_payload = _read_json_if_exists(job["status_file"])
        completed = status_payload.get("status") in {"completed", "completed_with_numeric_warning"}
        artifacts_ok = job["metrics_file"].exists() and job["predictions_file"].exists()
        same_signature = str(status_payload.get("eval_signature", "")) == str(eval_signature)
        if completed and artifacts_ok and same_signature:
            reusable_jobs += 1
            _log(
                "held-out test model skip",
                model_name=str(job["model_name"]),
                config_id=str(job["config_id"]),
                reason="existing_completed_artifacts",
            )
            continue
        pending_jobs.append(job)

    _write_json_atomic(
        status_path,
        {
            "status": "started",
            "eval_signature": eval_signature,
            "analysis_dir": str(analysis_dir),
            "n_models": int(len(spec_jobs)),
            "reused_models": int(reusable_jobs),
            "pending_models": int(len(pending_jobs)),
            "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
            "manifest": eval_manifest,
        },
    )
    _log(
        "held-out test execution plan",
        pending_models=int(len(pending_jobs)),
        reused_models=int(reusable_jobs),
        total_models=int(len(spec_jobs)),
    )

    worker_plan = _resolve_held_out_worker_count(
        pending_models=int(len(pending_jobs)),
        parallel_enabled=parallel_enabled,
        parallel_cpu_fraction=parallel_cpu_fraction,
        parallel_max_workers=parallel_max_workers,
        X_train=X_tv,
        X_test=X_test,
        y_train_log=y_tv_log,
        y_test_log=y_test_log,
    )
    _log(
        "held-out parallel plan",
        parallel_enabled=bool(parallel_enabled),
        workers=int(worker_plan["workers"]),
        cpu_limit=int(worker_plan["cpu_limit"]),
        memory_limit=int(worker_plan["memory_limit"]),
        safe_cap=int(worker_plan["safe_cap"]),
        estimated_bytes_per_worker=int(worker_plan["estimated_bytes_per_worker"]),
        available_memory_bytes=worker_plan["available_memory_bytes"],
        bootstrap_applied=False,
    )

    def _run_single_held_out_job(job: Dict[str, Any]) -> None:
        model_start = time.perf_counter()
        spec = job["spec"]
        model_name = str(job["model_name"])
        model_config_json = str(job["model_config_json"])
        metric_ratio_mode = str(job["metric_ratio_mode"])
        config_id = str(job["config_id"])
        metrics_file = job["metrics_file"]
        predictions_file = job["predictions_file"]
        model_status_file = job["status_file"]
        _log("held-out test model start", model_name=model_name, config_id=config_id)
        _write_json_atomic(
            model_status_file,
            {
                "status": "started",
                "eval_signature": eval_signature,
                "config_id": config_id,
                "model_name": model_name,
                "ratio_mode": metric_ratio_mode,
                "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
            },
        )
        try:
            estimator = spec["factory"]()
            prep_elapsed = 0.0
            if bool(spec.get("requires_linear_pipeline", False)):
                prep_start = time.perf_counter()
                pipe = linear_pipeline_builder()
                X_train_m = pipe.fit_transform(X_tv, y_tv_log)
                X_test_m = pipe.transform(X_test)
                prep_elapsed = time.perf_counter() - prep_start
            else:
                X_train_m = X_tv
                X_test_m = X_test

            fit_start = time.perf_counter()
            estimator.fit(X_train_m, y_tv_log)
            y_pred_test_log = np.asarray(estimator.predict(X_test_m), dtype=float).reshape(-1)
            fit_elapsed = time.perf_counter() - fit_start

            metric_start = time.perf_counter()
            metrics = _compute_extended_metrics(
                y_true_log=y_test_log,
                y_pred_log=y_pred_test_log,
                y_train_log=y_tv_log,
                ratio_mode=metric_ratio_mode,
            )
            metric_elapsed = time.perf_counter() - metric_start
            bad_metric = _first_bad_numeric_value(metrics, abs_cap=float(numeric_sanity_abs_cap))
            bad_pred = _first_bad_numeric_value(
                {
                    "y_pred_log_min": float(np.nanmin(y_pred_test_log)),
                    "y_pred_log_max": float(np.nanmax(y_pred_test_log)),
                },
                abs_cap=float(numeric_sanity_abs_cap),
            )
            bad = bad_metric if bad_metric is not None else bad_pred
            numeric_fields = _numeric_guard_fields(
                bad=bad,
                stage=("test_metrics" if bad_metric is not None else "test_predictions") if bad is not None else "",
                cv_flagged=(config_id in invalid_set),
            )
            if bad is not None:
                _log(
                    "held-out test model flagged for invalid numeric output",
                    model_name=model_name,
                    config_id=config_id,
                    offending_field=str(bad.get("field", bad.get("metric", ""))),
                    offending_reason=str(bad.get("reason", "")),
                    total_elapsed_sec=f"{time.perf_counter() - model_start:.2f}",
                )

            metrics_row = {
                "config_id": config_id,
                "model_name": model_name,
                "ratio_mode": metric_ratio_mode,
                "model_config_json": model_config_json,
                **numeric_fields,
                **metrics,
            }
            pred_df = pd.DataFrame(
                {
                    "config_id": config_id,
                    "model_name": model_name,
                    "ratio_mode": metric_ratio_mode,
                    "row_id": df_test.index.to_numpy(),
                    "sale_date": df_test[date_col].to_numpy(),
                    "numeric_stability_status": numeric_fields["numeric_stability_status"],
                    "numeric_guard_flagged": numeric_fields["numeric_guard_flagged"],
                    "numeric_guard_stage": numeric_fields["numeric_guard_stage"],
                    "numeric_guard_field": numeric_fields["numeric_guard_field"],
                    "numeric_guard_value": numeric_fields["numeric_guard_value"],
                    "numeric_guard_reason": numeric_fields["numeric_guard_reason"],
                    "cv_numeric_warning_flagged": numeric_fields["cv_numeric_warning_flagged"],
                    "y_true_log": y_test_log,
                    "y_pred_log": y_pred_test_log,
                    "y_true": np.exp(y_test_log),
                    "y_pred": np.exp(y_pred_test_log),
                }
            )
            _write_parquet_atomic(pd.DataFrame([metrics_row]), metrics_file, engine=parquet_engine)
            _write_parquet_atomic(pred_df, predictions_file, engine=parquet_engine)
            _write_json_atomic(
                model_status_file,
                {
                    "status": "completed_with_numeric_warning" if bool(numeric_fields["numeric_guard_flagged"]) else "completed",
                    "eval_signature": eval_signature,
                    "config_id": config_id,
                    "model_name": model_name,
                    "ratio_mode": metric_ratio_mode,
                    "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
                    "artifacts": {
                        "metrics_file": str(metrics_file),
                        "predictions_file": str(predictions_file),
                    },
                    "numeric_stability_status": numeric_fields["numeric_stability_status"],
                    "numeric_guard_flagged": bool(numeric_fields["numeric_guard_flagged"]),
                },
            )
            _log(
                "held-out test model completed",
                model_name=model_name,
                config_id=config_id,
                prep_sec=f"{prep_elapsed:.2f}",
                fit_predict_sec=f"{fit_elapsed:.2f}",
                metrics_sec=f"{metric_elapsed:.2f}",
                total_sec=f"{time.perf_counter() - model_start:.2f}",
            )
        except Exception as exc:
            _write_json_atomic(
                model_status_file,
                {
                    "status": "failed",
                    "eval_signature": eval_signature,
                    "config_id": config_id,
                    "model_name": model_name,
                    "ratio_mode": metric_ratio_mode,
                    "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
                    "error_type": exc.__class__.__name__,
                    "error_message": str(exc),
                }
            )
            raise

    if int(worker_plan["workers"]) <= 1 or len(pending_jobs) <= 1:
        for job in pending_jobs:
            _run_single_held_out_job(job)
    else:
        with ThreadPoolExecutor(max_workers=int(worker_plan["workers"])) as executor:
            futures = [executor.submit(_run_single_held_out_job, job) for job in pending_jobs]
            for future in as_completed(futures):
                future.result()

    write_start = time.perf_counter()
    test_metric_frames: List[pd.DataFrame] = []
    pred_rows: List[pd.DataFrame] = []
    for job in spec_jobs:
        metrics_file = job["metrics_file"]
        predictions_file = job["predictions_file"]
        if not metrics_file.exists():
            raise FileNotFoundError(f"Missing held-out metrics shard for config_id={job['config_id']}: {metrics_file}")
        if not predictions_file.exists():
            raise FileNotFoundError(f"Missing held-out predictions shard for config_id={job['config_id']}: {predictions_file}")
        test_metric_frames.append(pd.read_parquet(metrics_file))
        pred_rows.append(pd.read_parquet(predictions_file))

    test_metrics_df = pd.concat(test_metric_frames, ignore_index=True) if test_metric_frames else pd.DataFrame()
    if "config_id" in test_metrics_df.columns:
        test_metrics_df["config_id"] = test_metrics_df["config_id"].astype(str)
    _write_csv_atomic(test_metrics_df, test_metrics_path)

    if pred_rows:
        test_predictions_df = pd.concat(pred_rows, ignore_index=True)
        _write_parquet_atomic(test_predictions_df, test_predictions_path, engine=parquet_engine)
    else:
        test_predictions_df = pd.DataFrame(columns=pred_columns)
        _write_parquet_atomic(test_predictions_df, test_predictions_path, engine=parquet_engine)

    flagged_columns = [
        "config_id",
        "model_name",
        "ratio_mode",
        "model_config_json",
        "numeric_stability_status",
        "numeric_guard_flagged",
        "numeric_guard_stage",
        "numeric_guard_field",
        "numeric_guard_value",
        "numeric_guard_reason",
        "cv_numeric_warning_flagged",
    ]
    flagged_df = pd.DataFrame(columns=flagged_columns)
    if (not test_metrics_df.empty) and ("numeric_guard_flagged" in test_metrics_df.columns):
        flagged_df = test_metrics_df.loc[test_metrics_df["numeric_guard_flagged"].fillna(False), flagged_columns].copy()
    if not flagged_df.empty:
        _write_csv_atomic(flagged_df, flagged_path)
        _write_csv_atomic(flagged_df, legacy_flagged_path)
    else:
        if flagged_path.exists():
            flagged_path.unlink()
        if legacy_flagged_path.exists():
            legacy_flagged_path.unlink()

    _write_json_atomic(
        test_meta_path,
        {
            "fairness_ratio_mode": fairness_ratio_mode,
            "swept_ratio_modes": sorted(
                {
                    str(dict(spec.get("config", {})).get("ratio_mode", spec.get("metric_ratio_mode", fairness_ratio_mode)))
                    for spec in model_specs
                }
            ),
            # For reproducibility of `OOS R2` in downstream stacked test overlays.
            # Note: in this repo's current metric implementation, `OOS R2` uses
            # the mean of the provided y_train array (whatever scale it is in).
            "y_train_log_mean": float(np.mean(y_tv_log)),
            "n_train_validate": int(df_train_validate.shape[0]),
            "n_test": int(df_test.shape[0]),
            "eval_signature": eval_signature,
            "config_ids": [str(job["config_id"]) for job in spec_jobs],
        },
    )
    _write_json_atomic(
        status_path,
        {
            "status": "completed",
            "eval_signature": eval_signature,
            "analysis_dir": str(analysis_dir),
            "n_models": int(len(spec_jobs)),
            "reused_models": int(reusable_jobs),
            "computed_models": int(len(pending_jobs)),
            "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
            "manifest": eval_manifest,
            "artifacts": {
                "test_metrics_csv": str(test_metrics_path),
                "test_predictions_parquet": str(test_predictions_path),
                "test_eval_metadata_json": str(test_meta_path),
                "test_flagged_configs_csv": str(flagged_path) if flagged_df.shape[0] > 0 else None,
                "test_rejected_configs_csv": str(legacy_flagged_path) if flagged_df.shape[0] > 0 else None,
            },
        },
    )
    _log(
        "held-out test artifacts written",
        metrics_rows=int(test_metrics_df.shape[0]),
        prediction_frames=int(len(pred_rows)),
        flagged_configs=int(flagged_df.shape[0]),
        reused_models=int(reusable_jobs),
        computed_models=int(len(pending_jobs)),
        write_sec=f"{time.perf_counter() - write_start:.2f}",
        total_sec=f"{time.perf_counter() - eval_start:.2f}",
    )

    out = {"test_metrics_csv": str(test_metrics_path), "test_predictions_parquet": str(test_predictions_path)}
    if not flagged_df.empty:
        out["test_flagged_configs_csv"] = str(flagged_path)
        out["test_rejected_configs_csv"] = str(legacy_flagged_path)
    return out


def run_full_pipeline(
    *,
    result_root: str,
    data_path: str,
    sample_frac: float | None,
    seed: int,
    rho_values: List[float],
    rho_values_smooth: Optional[List[float]],
    rho_values_cov: Optional[List[float]],
    keep_values: List[float],
    ratio_modes: List[str],
    split_protocol: Dict[str, Any],
    bootstrap_protocol: Dict[str, Any],
    parallel_enabled: bool,
    parallel_cpu_fraction: float,
    parallel_max_workers: Optional[int],
    parquet_engine: str,
    use_ccao_fallback: bool = False,
    numeric_sanity_abs_cap: float = 1e6,
    baseline_search: bool = False,
    baseline_search_trials: Optional[int] = None,
    include_cvar_models: bool = False,
) -> Dict[str, Any]:
    """
    Run the full pipeline end-to-end:
      1. Load and split data into train/validate and held-out test sets.
      2. Build model specs (baseline + penalty sweeps) from YAML configs.
      3. Run robust rolling-origin CV with bootstrap resampling.
      4. Evaluate each model on the held-out test set (single run).

    Returns a summary dict with data_id, split_id, artifact paths, and row counts.
    """
    target_col = "meta_sale_price"
    date_col = "meta_sale_date"
    fairness_ratio_mode = "diff"
    pipeline_start = time.perf_counter()
    _log(
        "pipeline start",
        result_root=result_root,
        data_path=data_path,
        sample_frac=sample_frac,
        seed=int(seed),
        parallel=bool(parallel_enabled),
    )

    config_start = time.perf_counter()
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    with open("model_params.yaml", "r", encoding="utf-8") as f:
        model_params = yaml.safe_load(f)
    _log("configuration loaded", elapsed_sec=f"{time.perf_counter() - config_start:.2f}")

    data_start = time.perf_counter()
    df_train_validate, df_test, df_assess, predictor_cols, categorical_cols = _load_and_split_data(
        data_path=data_path,
        params=params,
        target_column=target_col,
        date_column=date_col,
        sample_frac=sample_frac,
        sample_seed=seed,
    )
    _log("data load/split finished", elapsed_sec=f"{time.perf_counter() - data_start:.2f}")

    linear_pipeline_builder = lambda: build_model_pipeline(
        pred_vars=predictor_cols,
        cat_vars=categorical_cols,
        id_vars=params["model"]["predictor"]["id"],
    )

    data_signature = {
        "data_path": str(data_path),
        "target_col": target_col,
        "date_col": date_col,
        "predictor_cols": predictor_cols,
        "categorical_cols": categorical_cols,
        "filters": {"drop_multicard": True, "drop_outliers": True},
        "sample_frac": sample_frac,
        "sample_seed": int(seed),
        "split_prop_pre2024": float(params["cv"]["split_prop"]),
    }

    model_setup_start = time.perf_counter()
    lgbm_params = _build_lgbm_params_from_files(model_params=model_params, ccao_params=params, seed=seed, use_ccao_fallback=use_ccao_fallback)
    baseline_search_artifacts: Dict[str, Any] = {}
    if bool(baseline_search):
        baseline_search_artifacts = _run_baseline_lgbm_search(
            result_root=result_root,
            params=params,
            df_train_validate=df_train_validate,
            predictor_cols=predictor_cols,
            categorical_cols=categorical_cols,
            linear_pipeline_builder=linear_pipeline_builder,
            split_protocol=split_protocol,
            parquet_engine=parquet_engine,
            parallel_enabled=parallel_enabled,
            parallel_cpu_fraction=parallel_cpu_fraction,
            parallel_max_workers=parallel_max_workers,
            numeric_sanity_abs_cap=float(numeric_sanity_abs_cap),
            base_lgbm_params=lgbm_params,
            data_signature=data_signature,
            seed=int(seed),
            n_random_trials=baseline_search_trials,
            date_col=date_col,
            target_col=target_col,
            fairness_ratio_mode=fairness_ratio_mode,
        )
        lgbm_params = dict(baseline_search_artifacts["best_lgbm_params"])
    smooth_rhos = [float(x) for x in (rho_values if rho_values_smooth is None else rho_values_smooth)]
    cov_rhos = [float(x) for x in (rho_values if rho_values_cov is None else rho_values_cov)]
    model_specs = _build_model_specs(
        lgbm_params=lgbm_params,
        rho_values_smooth=smooth_rhos,
        rho_values_cov=cov_rhos,
        keep_values=keep_values,
        ratio_modes=ratio_modes,
        fairness_ratio_mode=fairness_ratio_mode,
        include_cvar_models=bool(include_cvar_models),
    )
    _log(
        "model specs built",
        n_models=int(len(model_specs)),
        n_smooth_rhos=int(len(smooth_rhos)),
        n_cov_rhos=int(len(cov_rhos)),
        n_ratio_modes=int(len(ratio_modes)),
        elapsed_sec=f"{time.perf_counter() - model_setup_start:.2f}",
    )

    cv_start = time.perf_counter()
    _log(
        "starting rolling-origin CV",
        split_protocol=json.dumps(split_protocol, sort_keys=True),
        bootstrap_protocol=json.dumps(bootstrap_protocol, sort_keys=True),
    )
    cv_out = run_robust_rolling_origin_cv(
        df_train_validate=df_train_validate,
        date_col=date_col,
        target_col=target_col,
        predictor_cols=predictor_cols,
        categorical_cols=categorical_cols,
        model_specs=model_specs,
        linear_pipeline_builder=linear_pipeline_builder,
        result_root=result_root,
        data_signature=data_signature,
        split_protocol=split_protocol,
        bootstrap_protocol=bootstrap_protocol,
        fairness_ratio_mode=fairness_ratio_mode,
        predict_store=True,
        parquet_engine=parquet_engine,
        log_progress=True,
        parallel_enabled=parallel_enabled,
        parallel_cpu_fraction=parallel_cpu_fraction,
        parallel_max_workers=parallel_max_workers,
        parallel_backend="loky",
        numeric_sanity_abs_cap=float(numeric_sanity_abs_cap),
    )
    _log(
        "rolling-origin CV finished",
        data_id=str(cv_out["data_id"]),
        split_id=str(cv_out["split_id"]),
        fold_count=int(cv_out["fold_count"]),
        flagged_configs=int(len(cv_out.get("flagged_config_ids", cv_out.get("invalid_config_ids", [])))),
        elapsed_sec=f"{time.perf_counter() - cv_start:.2f}",
    )

    data_id = str(cv_out["data_id"])
    split_id = str(cv_out["split_id"])

    analysis_dir = Path(result_root) / "analysis" / f"data_id={data_id}" / f"split_id={split_id}"
    test_eval_start = time.perf_counter()
    test_artifacts = _evaluate_models_on_test_set(
        df_train_validate=df_train_validate,
        df_test=df_test,
        predictor_cols=predictor_cols,
        categorical_cols=categorical_cols,
        date_col=date_col,
        target_col=target_col,
        model_specs=model_specs,
        linear_pipeline_builder=linear_pipeline_builder,
        fairness_ratio_mode=fairness_ratio_mode,
        analysis_dir=analysis_dir,
        parquet_engine=parquet_engine,
        numeric_sanity_abs_cap=float(numeric_sanity_abs_cap),
        parallel_enabled=parallel_enabled,
        parallel_cpu_fraction=parallel_cpu_fraction,
        parallel_max_workers=parallel_max_workers,
        invalid_config_ids=[str(x) for x in cv_out.get("flagged_config_ids", cv_out.get("invalid_config_ids", []))],
    )
    _log("held-out test evaluation finished", elapsed_sec=f"{time.perf_counter() - test_eval_start:.2f}")

    _log(
        "pipeline finished",
        total_sec=f"{time.perf_counter() - pipeline_start:.2f}",
        n_models=int(len(model_specs)),
        n_folds=int(cv_out["fold_count"]),
    )

    return {
        "data_id": data_id,
        "split_id": split_id,
        "result_root": str(Path(result_root).resolve()),
        "analysis_dir": str(analysis_dir),
        **test_artifacts,
        "n_train_validate": int(df_train_validate.shape[0]),
        "n_test": int(df_test.shape[0]),
        "n_assess": int(df_assess.shape[0]),
        "n_models": int(len(model_specs)),
        "n_folds": int(cv_out["fold_count"]),
        "n_flagged_configs": int(len(cv_out.get("flagged_config_ids", cv_out.get("invalid_config_ids", [])))),
        "n_invalid_configs": int(len(cv_out.get("flagged_config_ids", cv_out.get("invalid_config_ids", [])))),
        "baseline_search_enabled": bool(baseline_search),
        "include_cvar_models": bool(include_cvar_models),
        **baseline_search_artifacts,
    }


_CV_CONFIG_PATH = "cv_config.yaml"


def _load_cv_config(config_path: str = _CV_CONFIG_PATH) -> dict:
    """Load cv_config.yaml, falling back to an empty dict if the file is absent."""
    p = Path(config_path)
    if p.is_file():
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _build_arg_parser(cfg: dict) -> argparse.ArgumentParser:
    """
    Build the CLI parser.  Defaults come from cv_config.yaml (via `cfg`); any
    flag passed on the command line overrides the YAML value.
    """
    sp = cfg.get("split_protocol", {})
    bp = cfg.get("bootstrap_protocol", {})
    pp = cfg.get("parallel", {})

    p = argparse.ArgumentParser(
        description="Run rolling-origin CV and held-out test evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- I/O ---
    p.add_argument("--config", type=str, default=_CV_CONFIG_PATH, help="Path to cv_config.yaml.")
    p.add_argument("--result-root", type=str, default=cfg.get("result_root", "./output/robust_rolling_origin_cv"))
    p.add_argument("--data-path", type=str, default=cfg.get("data_path", "./data/CCAO/2025/training_data.parquet"))
    p.add_argument("--sample-frac", type=float, default=cfg.get("sample_frac", None))
    p.add_argument("--seed", type=int, default=cfg.get("seed", 2025))

    # --- Sweep grids ---
    default_rho = ",".join(str(v) for v in cfg.get("rho_values", [0.0, 10.0]))
    default_keep = ",".join(str(v) for v in cfg.get("keep_values", [0.5, 0.7, 0.9]))
    ratio_modes_cfg = cfg.get("ratio_modes", ["diff"])
    if isinstance(ratio_modes_cfg, str):
        default_ratio_modes = ratio_modes_cfg
    else:
        default_ratio_modes = ",".join(str(v) for v in ratio_modes_cfg)
    p.add_argument(
        "--rho-values",
        type=str,
        default=default_rho,
        help="Preferred: two comma-separated bounds 'rho_min,rho_max'. Backward-compatible: explicit comma-separated rho list.",
    )
    p.add_argument("--rho-count", type=int, default=int(cfg.get("rho_count", 6)),
                   help="Number of rho values generated between rho min/max (inclusive).")
    p.add_argument("--rho-scale", type=str, default=str(cfg.get("rho_scale", "linear")),
                   choices=["linear", "log", "geom"],
                   help="Spacing scale used when --rho-values provides two bounds.")
    # Optional family-specific rho sweeps (fallback to common rho sweep when omitted).
    default_rho_smooth = cfg.get("rho_values_smooth", None)
    default_rho_cov = cfg.get("rho_values_cov", None)
    p.add_argument(
        "--rho-values-smooth",
        type=str,
        default=(None if default_rho_smooth is None else ",".join(str(v) for v in default_rho_smooth)),
        help="Optional smooth-family rho sweep override (LGBSmooth*). Same format as --rho-values.",
    )
    p.add_argument(
        "--rho-values-cov",
        type=str,
        default=(None if default_rho_cov is None else ",".join(str(v) for v in default_rho_cov)),
        help="Optional cov-family rho sweep override (LGBCov*). Same format as --rho-values.",
    )
    p.add_argument(
        "--rho-count-smooth",
        type=int,
        default=(None if cfg.get("rho_count_smooth", None) is None else int(cfg.get("rho_count_smooth"))),
        help="Optional rho_count override for --rho-values-smooth.",
    )
    p.add_argument(
        "--rho-count-cov",
        type=int,
        default=(None if cfg.get("rho_count_cov", None) is None else int(cfg.get("rho_count_cov"))),
        help="Optional rho_count override for --rho-values-cov.",
    )
    p.add_argument(
        "--rho-scale-smooth",
        type=str,
        default=cfg.get("rho_scale_smooth", None),
        choices=["linear", "log", "geom"],
        help="Optional rho_scale override for --rho-values-smooth.",
    )
    p.add_argument(
        "--rho-scale-cov",
        type=str,
        default=cfg.get("rho_scale_cov", None),
        choices=["linear", "log", "geom"],
        help="Optional rho_scale override for --rho-values-cov.",
    )
    p.add_argument("--keep-values", type=str, default=default_keep, help="Comma-separated keep values for CVaR-style variants when enabled.")
    p.add_argument(
        "--include-cvar-models",
        action=argparse.BooleanOptionalAction,
        default=bool(cfg.get("include_cvar_models", False)),
        help=(
            "When enabled, also evaluate CVaR-style penalty variants. "
            "By default, only the four basic families are run: LinearRegression, "
            "LGBMRegressor, LGBSmoothPenalty, and LGBCovPenalty."
        ),
    )
    p.add_argument(
        "--ratio-modes",
        type=str,
        default=default_ratio_modes,
        help="Comma-separated ratio modes for rho-weighted models. Allowed values: div,diff.",
    )

    # --- Split protocol ---
    p.add_argument("--train-mode", type=str, default=sp.get("train_mode", "expanding"), choices=["expanding", "sliding"])
    p.add_argument("--initial-train-months", type=int, default=sp.get("initial_train_months", 9))
    p.add_argument("--val-fraction", type=float, default=sp.get("val_fraction", None),
                   help="Fraction of rows used as validation each fold (Mode A). Set to 0 or omit to use fixed-time-window mode.")
    p.add_argument("--val-window-months", type=int, default=sp.get("val_window_months", 9),
                   help="Calendar length of each validation block (Mode B, used when --val-fraction is not set).")
    p.add_argument("--step-months", type=int, default=sp.get("step_months", 9),
                   help="Months the origin advances between folds (Mode B).")
    p.add_argument("--min-train-rows", type=int, default=sp.get("min_train_rows", 200))
    p.add_argument("--min-val-rows", type=int, default=sp.get("min_val_rows", 100))

    # --- Bootstrap protocol ---
    p.add_argument("--n-bootstrap", type=int, default=bp.get("n_bootstrap", 200))
    p.add_argument("--bootstrap-block-freq", type=str, default=bp.get("bootstrap_block_freq", "M"),
                   help="Pandas Period freq for time blocks (e.g. 'M', 'W', 'Q').")

    # --- Parallelism ---
    p.add_argument("--parallel", action="store_true", default=bool(pp.get("enabled", False)),
                   help="Enable joblib parallel CV execution.")
    p.add_argument("--no-parallel", dest="parallel", action="store_false",
                   help="Disable parallel execution (overrides --parallel / YAML).")
    p.add_argument("--parallel-cpu-fraction", type=float, default=float(pp.get("cpu_fraction", 0.9)))
    p.add_argument("--parallel-max-workers", type=int, default=pp.get("max_workers", 32))

    # --- Storage ---
    p.add_argument("--parquet-engine", type=str, default=cfg.get("parquet_engine", "fastparquet"),
                   choices=["fastparquet", "pyarrow"])

    # --- LGBM param sourcing ---
    p.add_argument(
        "--use-ccao-params-fallback",
        action="store_true",
        default=bool(cfg.get("use_ccao_params_fallback", False)),
        help=(
            "When set, missing keys in model_params.yaml fall back to params.yaml's "
            "hyperparameter.default (original CCAO behaviour). "
            "By default, missing keys fall back to LightGBM's own native defaults."
        ),
    )
    p.add_argument(
        "--numeric-sanity-abs-cap",
        type=float,
        default=float(cfg.get("numeric_sanity_abs_cap", 1e6)),
        help=(
            "Absolute-value cap for numeric sanity checks in CV/test metrics. "
            "If any metric exceeds this cap or is non-finite, the corresponding results are saved but flagged."
        ),
    )
    p.add_argument(
        "--baseline-search",
        action=argparse.BooleanOptionalAction,
        default=bool(cfg.get("baseline_search", False)),
        help=(
            "Opt-in baseline-only LightGBM hyperparameter search using rolling-origin CV. "
            "When enabled, the selected best baseline parameters are reused as the base configuration "
            "for the later penalized-model sweeps."
        ),
    )
    p.add_argument(
        "--baseline-search-trials",
        type=int,
        default=cfg.get("baseline_search_trials", None),
        help=(
            "Number of random baseline-LGBM candidates to evaluate when --baseline-search is enabled. "
            "The current baseline configuration is always included as an additional candidate. "
            "Defaults to params.yaml cv.initial_set when omitted."
        ),
    )
    return p


if __name__ == "__main__":
    # Two-pass parse: first resolve --config so we load the right YAML, then
    # re-parse with full defaults derived from that YAML.
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--config", type=str, default=_CV_CONFIG_PATH)
    _known, _ = _pre.parse_known_args()

    cfg = _load_cv_config(_known.config)
    args = _build_arg_parser(cfg).parse_args()

    # Re-load config if --config was overridden explicitly on the CLI so the
    # YAML path recorded in output matches what was actually used.
    if args.config != _CV_CONFIG_PATH:
        cfg = _load_cv_config(args.config)
    _log("cli arguments parsed", config_path=str(args.config))

    val_fraction = float(args.val_fraction) if (args.val_fraction is not None and float(args.val_fraction) > 0) else None

    rho_values_raw = _parse_float_list(str(args.rho_values))
    rho_values = _build_rho_values(
        rho_values_raw,
        rho_count=int(args.rho_count),
        rho_scale=str(args.rho_scale),
    )
    rho_values_smooth: Optional[List[float]] = None
    rho_values_cov: Optional[List[float]] = None
    if args.rho_values_smooth is not None and str(args.rho_values_smooth).strip() != "":
        rho_values_smooth = _build_rho_values(
            _parse_float_list(str(args.rho_values_smooth)),
            rho_count=int(args.rho_count if args.rho_count_smooth is None else args.rho_count_smooth),
            rho_scale=str(args.rho_scale if args.rho_scale_smooth is None else args.rho_scale_smooth),
        )
    if args.rho_values_cov is not None and str(args.rho_values_cov).strip() != "":
        rho_values_cov = _build_rho_values(
            _parse_float_list(str(args.rho_values_cov)),
            rho_count=int(args.rho_count if args.rho_count_cov is None else args.rho_count_cov),
            rho_scale=str(args.rho_scale if args.rho_scale_cov is None else args.rho_scale_cov),
        )

    out = run_full_pipeline(
        result_root=str(args.result_root),
        data_path=str(args.data_path),
        sample_frac=(None if args.sample_frac is None else float(args.sample_frac)),
        seed=int(args.seed),
        rho_values=rho_values,
        rho_values_smooth=rho_values_smooth,
        rho_values_cov=rho_values_cov,
        keep_values=[float(x) for x in _parse_float_list(str(args.keep_values))],
        ratio_modes=_parse_ratio_mode_list(str(args.ratio_modes)),
        split_protocol={
            "train_mode": str(args.train_mode),
            "initial_train_months": int(args.initial_train_months),
            "val_fraction": val_fraction,
            "val_window_months": int(args.val_window_months),
            "step_months": int(args.step_months),
            "min_train_rows": int(args.min_train_rows),
            "min_val_rows": int(args.min_val_rows),
        },
        bootstrap_protocol={
            "n_bootstrap": int(args.n_bootstrap),
            "block_freq": str(args.bootstrap_block_freq),
            "seed": int(args.seed),
        },
        parallel_enabled=bool(args.parallel),
        parallel_cpu_fraction=float(args.parallel_cpu_fraction),
        parallel_max_workers=(None if args.parallel_max_workers is None else int(args.parallel_max_workers)),
        parquet_engine=str(args.parquet_engine),
        use_ccao_fallback=bool(args.use_ccao_params_fallback),
        numeric_sanity_abs_cap=float(args.numeric_sanity_abs_cap),
        baseline_search=bool(args.baseline_search),
        baseline_search_trials=(None if args.baseline_search_trials is None else int(args.baseline_search_trials)),
        include_cvar_models=bool(args.include_cvar_models),
    )
    print("=" * 90)
    print("TEMPORAL CV COMPLETED")
    print("=" * 90)
    print(f"data_id={out['data_id']} | split_id={out['split_id']}")
    print(f"analysis_dir={out['analysis_dir']}")
    print(f"test_metrics_csv={out['test_metrics_csv']}")
    print(f"test_predictions_parquet={out['test_predictions_parquet']}")
    if bool(out.get("baseline_search_enabled", False)):
        print(f"baseline_search_summary_csv={out['search_summary_csv']}")
        print(f"baseline_search_best_params_json={out['best_params_json']}")
