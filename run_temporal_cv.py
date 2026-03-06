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
import json
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
    LGBCovPenaltyCVaR,
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
    fairness_ratio_mode: str,
) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []

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
            "config": {},
            "requires_linear_pipeline": False,
            "factory": lambda: lgb.LGBMRegressor(**dict(lgbm_params)),
        }
    )

    # Soft penalty variants (rho sweep)
    keep_sweep = [float(k) for k in keep_values] if keep_values else [1.0]
    for rho in rho_values_smooth:
        r = float(rho)
        specs.append(
            {
                "name": "LGBSmoothPenalty",
                "config": {"rho": r},
                "requires_linear_pipeline": False,
                "factory": (lambda rho=r: LGBSmoothPenalty(rho=rho, ratio_mode=fairness_ratio_mode, zero_grad_tol=1e-12, lgbm_params=dict(lgbm_params), verbose=False)),
            }
        )
        # for keep in keep_sweep:
        #     k = float(keep)
        #     specs.append(
        #         {
        #             "name": "LGBSmoothPenaltyCVaR",
        #             "config": {"rho": r, "keep": k},
        #             "requires_linear_pipeline": False,
        #             "factory": (
        #                 lambda rho=r, keep=k: LGBSmoothPenaltyCVaR(
        #                     rho=rho,
        #                     mse_keep=keep,
        #                     ratio_mode=fairness_ratio_mode,
        #                     zero_grad_tol=1e-12,
        #                     lgbm_params=dict(lgbm_params),
        #                     verbose=False,
        #                 )
        #             ),
        #         }
        #     )
        #     specs.append(
        #         {
        #             "name": "LGBSmoothPenaltyCVaRTotal",
        #             "config": {"rho": r, "keep": k},
        #             "requires_linear_pipeline": False,
        #             "factory": (
        #                 lambda rho=r, keep=k: LGBSmoothPenaltyCVaRTotal(
        #                     rho=rho,
        #                     keep=keep,
        #                     ratio_mode=fairness_ratio_mode,
        #                     zero_grad_tol=1e-12,
        #                     lgbm_params=dict(lgbm_params),
        #                     verbose=False,
        #                 )
        #             ),
        #         }
        #     )

    for rho in rho_values_cov:
        r = float(rho)
        specs.append(
            {
                "name": "LGBCovPenalty",
                "config": {"rho": r},
                "requires_linear_pipeline": False,
                "factory": (lambda rho=r: LGBCovPenalty(rho=rho, ratio_mode=fairness_ratio_mode, zero_grad_tol=1e-12, lgbm_params=dict(lgbm_params), verbose=False)),
            }
        )
        # for keep in keep_sweep:
        #     k = float(keep)
        #     # specs.append( # NOTE: This variant is not stable, since its just one side of the loss. It has overflow issues.
        #     #     {
        #     #         "name": "LGBCovPenaltyCVaR",
        #     #         "config": {"rho": r, "keep": k},
        #     #         "requires_linear_pipeline": False,
        #     #         "factory": (
        #     #             lambda rho=r, keep=k: LGBCovPenaltyCVaR(
        #     #                 rho=rho,
        #     #                 mse_keep=keep,
        #     #                 ratio_mode=fairness_ratio_mode,
        #     #                 zero_grad_tol=1e-12,
        #     #                 lgbm_params=dict(lgbm_params),
        #     #                 verbose=False,
        #     #             )
        #     #         ),
        #     #     }
        #     # )
        #     specs.append(
        #         {
        #             "name": "LGBCovPenaltyCVaRTotal",
        #             "config": {"rho": r, "keep": k},
        #             "requires_linear_pipeline": False,
        #             "factory": (
        #                 lambda rho=r, keep=k: LGBCovPenaltyCVaRTotal(
        #                     rho=rho,
        #                     mse_keep=keep,
        #                     ratio_mode=fairness_ratio_mode,
        #                     zero_grad_tol=1e-12,
        #                     lgbm_params=dict(lgbm_params),
        #                     verbose=False,
        #                 )
        #             ),
        #         }
        #     )

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
    invalid_config_ids: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Fit each config on the full train/validate universe and evaluate once on held-out test.
    """
    analysis_dir.mkdir(parents=True, exist_ok=True)
    eval_start = time.perf_counter()
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

    test_rows: List[Dict[str, Any]] = []
    pred_rows: List[pd.DataFrame] = []
    flagged_rows: List[Dict[str, Any]] = []
    invalid_set = {str(x) for x in (invalid_config_ids or [])}

    for spec in model_specs:
        model_start = time.perf_counter()
        model_name = str(spec["name"])
        model_config = dict(spec.get("config", {}))
        config_id = _stable_hash({"model_name": model_name, "config": model_config})
        _log("held-out test model start", model_name=model_name, config_id=config_id)
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
            ratio_mode=fairness_ratio_mode,
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
        test_rows.append(
            {
                "config_id": config_id,
                "model_name": model_name,
                "model_config_json": json.dumps(model_config, sort_keys=True),
                **numeric_fields,
                **metrics,
            }
        )
        if bool(numeric_fields["numeric_guard_flagged"]):
            flagged_rows.append(
                {
                    "config_id": config_id,
                    "model_name": model_name,
                    "model_config_json": json.dumps(model_config, sort_keys=True),
                    **numeric_fields,
                }
            )

        pred_rows.append(
            pd.DataFrame(
                {
                    "config_id": config_id,
                    "model_name": model_name,
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

    test_metrics_path = analysis_dir / "test_metrics.csv"
    test_predictions_path = analysis_dir / "test_predictions.parquet"
    test_meta_path = analysis_dir / "test_eval_metadata.json"
    flagged_path = analysis_dir / "test_flagged_configs.csv"
    legacy_flagged_path = analysis_dir / "test_rejected_configs.csv"

    write_start = time.perf_counter()
    pd.DataFrame(test_rows).to_csv(test_metrics_path, index=False)
    if pred_rows:
        pd.concat(pred_rows, ignore_index=True).to_parquet(test_predictions_path, index=False, engine=parquet_engine)
    else:
        pd.DataFrame(
            columns=[
                "config_id",
                "model_name",
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
        ).to_parquet(test_predictions_path, index=False, engine=parquet_engine)
    if flagged_rows:
        pd.DataFrame(flagged_rows).to_csv(flagged_path, index=False)
        pd.DataFrame(flagged_rows).to_csv(legacy_flagged_path, index=False)
    test_meta_path.write_text(
        json.dumps(
            {
                "fairness_ratio_mode": fairness_ratio_mode,
                # For reproducibility of `OOS R2` in downstream stacked test overlays.
                # Note: in this repo's current metric implementation, `OOS R2` uses
                # the mean of the provided y_train array (whatever scale it is in).
                "y_train_log_mean": float(np.mean(y_tv_log)),
                "n_train_validate": int(df_train_validate.shape[0]),
                "n_test": int(df_test.shape[0]),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _log(
        "held-out test artifacts written",
        metrics_rows=int(len(test_rows)),
        prediction_frames=int(len(pred_rows)),
        flagged_configs=int(len(flagged_rows)),
        write_sec=f"{time.perf_counter() - write_start:.2f}",
        total_sec=f"{time.perf_counter() - eval_start:.2f}",
    )

    out = {"test_metrics_csv": str(test_metrics_path), "test_predictions_parquet": str(test_predictions_path)}
    if flagged_rows:
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
    split_protocol: Dict[str, Any],
    bootstrap_protocol: Dict[str, Any],
    parallel_enabled: bool,
    parallel_cpu_fraction: float,
    parallel_max_workers: Optional[int],
    parquet_engine: str,
    use_ccao_fallback: bool = False,
    numeric_sanity_abs_cap: float = 1e6,
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

    model_setup_start = time.perf_counter()
    lgbm_params = _build_lgbm_params_from_files(model_params=model_params, ccao_params=params, seed=seed, use_ccao_fallback=use_ccao_fallback)
    smooth_rhos = [float(x) for x in (rho_values if rho_values_smooth is None else rho_values_smooth)]
    cov_rhos = [float(x) for x in (rho_values if rho_values_cov is None else rho_values_cov)]
    model_specs = _build_model_specs(
        lgbm_params=lgbm_params,
        rho_values_smooth=smooth_rhos,
        rho_values_cov=cov_rhos,
        keep_values=keep_values,
        fairness_ratio_mode=fairness_ratio_mode,
    )
    _log(
        "model specs built",
        n_models=int(len(model_specs)),
        n_smooth_rhos=int(len(smooth_rhos)),
        n_cov_rhos=int(len(cov_rhos)),
        elapsed_sec=f"{time.perf_counter() - model_setup_start:.2f}",
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
    p.add_argument("--keep-values", type=str, default=default_keep, help="Comma-separated keep values for primal-dual models.")

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
    )
    print("=" * 90)
    print("TEMPORAL CV COMPLETED")
    print("=" * 90)
    print(f"data_id={out['data_id']} | split_id={out['split_id']}")
    print(f"analysis_dir={out['analysis_dir']}")
    print(f"test_metrics_csv={out['test_metrics_csv']}")
    print(f"test_predictions_parquet={out['test_predictions_parquet']}")
