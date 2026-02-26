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
from soft_constrained_models.boosting_models import LGBCovPenalty, LGBSmoothPenalty, LGBSmoothPenaltyCVaR, LGBSmoothPenaltyCVaRTotal
from utils.motivation_utils import _compute_extended_metrics, _stable_hash, run_robust_rolling_origin_cv


_ASSESSMENT_YEAR: int = 2024  # Calendar year used as the held-out assessment block.


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
    df = pd.read_parquet(data_path, engine="fastparquet")
    df = df[(~df["ind_pin_is_multicard"].astype("bool").fillna(True)) & (~df["sv_is_outlier"].astype("bool").fillna(True))].copy()

    predictor_cols = list(params["model"]["predictor"]["all"])
    categorical_cols = list(params["model"]["predictor"]["categorical"])
    keep_cols = predictor_cols + [target_column, date_column]
    df = df.loc[:, keep_cols].copy()

    if sample_frac is not None:
        if not (0.0 < float(sample_frac) <= 1.0):
            raise ValueError("sample_frac must be in (0, 1]. Use None to disable sampling.")
        if float(sample_frac) < 1.0:
            df = df.sample(frac=float(sample_frac), random_state=int(sample_seed)).copy()

    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(date_column).reset_index(drop=True)

    df_assess = df.loc[df[date_column].dt.year == _ASSESSMENT_YEAR, :].copy()
    df_pre2024 = df.loc[df[date_column].dt.year < _ASSESSMENT_YEAR, :].copy()

    train_prop = float(params["cv"]["split_prop"])
    split_idx = int(train_prop * df_pre2024.shape[0])
    df_test = df_pre2024.iloc[split_idx:, :].copy()
    df_train_validate = df_pre2024.iloc[:split_idx, :].copy()

    return df_train_validate, df_test, df_assess, predictor_cols, categorical_cols


def _build_model_specs(
    *,
    lgbm_params: dict,
    rho_values: List[float],
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
    for rho in rho_values:
        r = float(rho)
        specs.append(
            {
                "name": "LGBSmoothPenalty",
                "config": {"rho": r},
                "requires_linear_pipeline": False,
                "factory": (lambda rho=r: LGBSmoothPenalty(rho=rho, ratio_mode=fairness_ratio_mode, zero_grad_tol=1e-12, lgbm_params=dict(lgbm_params), verbose=False)),
            }
        )
        specs.append(
            {
                "name": "LGBCovPenalty",
                "config": {"rho": r},
                "requires_linear_pipeline": False,
                "factory": (lambda rho=r: LGBCovPenalty(rho=rho, ratio_mode=fairness_ratio_mode, zero_grad_tol=1e-12, lgbm_params=dict(lgbm_params), verbose=False)),
            }
        )
        for keep in keep_sweep:
            k = float(keep)
            specs.append(
                {
                    "name": "LGBSmoothPenaltyCVaR",
                    "config": {"rho": r, "keep": k},
                    "requires_linear_pipeline": False,
                    "factory": (
                        lambda rho=r, keep=k: LGBSmoothPenaltyCVaR(
                            rho=rho,
                            mse_keep=keep,
                            ratio_mode=fairness_ratio_mode,
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
                    "config": {"rho": r, "keep": k},
                    "requires_linear_pipeline": False,
                    "factory": (
                        lambda rho=r, keep=k: LGBSmoothPenaltyCVaRTotal(
                            rho=rho,
                            keep=keep,
                            ratio_mode=fairness_ratio_mode,
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
) -> Dict[str, str]:
    """
    Fit each config on the full train/validate universe and evaluate once on held-out test.
    """
    analysis_dir.mkdir(parents=True, exist_ok=True)

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

    for spec in model_specs:
        model_name = str(spec["name"])
        model_config = dict(spec.get("config", {}))
        config_id = _stable_hash({"model_name": model_name, "config": model_config})

        estimator = spec["factory"]()
        if bool(spec.get("requires_linear_pipeline", False)):
            pipe = linear_pipeline_builder()
            X_train_m = pipe.fit_transform(X_tv, y_tv_log)
            X_test_m = pipe.transform(X_test)
        else:
            X_train_m = X_tv
            X_test_m = X_test

        estimator.fit(X_train_m, y_tv_log)
        y_pred_test_log = np.asarray(estimator.predict(X_test_m), dtype=float).reshape(-1)

        metrics = _compute_extended_metrics(
            y_true_log=y_test_log,
            y_pred_log=y_pred_test_log,
            y_train_log=y_tv_log,
            ratio_mode=fairness_ratio_mode,
        )
        test_rows.append(
            {
                "config_id": config_id,
                "model_name": model_name,
                "model_config_json": json.dumps(model_config, sort_keys=True),
                **metrics,
            }
        )

        pred_rows.append(
            pd.DataFrame(
                {
                    "config_id": config_id,
                    "model_name": model_name,
                    "row_id": df_test.index.to_numpy(),
                    "sale_date": df_test[date_col].to_numpy(),
                    "y_true_log": y_test_log,
                    "y_pred_log": y_pred_test_log,
                    "y_true": np.exp(y_test_log),
                    "y_pred": np.exp(y_pred_test_log),
                }
            )
        )

    test_metrics_path = analysis_dir / "test_metrics.csv"
    test_predictions_path = analysis_dir / "test_predictions.parquet"
    test_meta_path = analysis_dir / "test_eval_metadata.json"

    pd.DataFrame(test_rows).to_csv(test_metrics_path, index=False)
    pd.concat(pred_rows, ignore_index=True).to_parquet(test_predictions_path, index=False, engine=parquet_engine)
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

    return {"test_metrics_csv": str(test_metrics_path), "test_predictions_parquet": str(test_predictions_path)}


def run_full_pipeline(
    *,
    result_root: str,
    data_path: str,
    sample_frac: float | None,
    seed: int,
    rho_values: List[float],
    keep_values: List[float],
    split_protocol: Dict[str, Any],
    bootstrap_protocol: Dict[str, Any],
    parallel_enabled: bool,
    parallel_cpu_fraction: float,
    parallel_max_workers: Optional[int],
    parquet_engine: str,
    use_ccao_fallback: bool = False,
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

    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    with open("model_params.yaml", "r", encoding="utf-8") as f:
        model_params = yaml.safe_load(f)

    df_train_validate, df_test, df_assess, predictor_cols, categorical_cols = _load_and_split_data(
        data_path=data_path,
        params=params,
        target_column=target_col,
        date_column=date_col,
        sample_frac=sample_frac,
        sample_seed=seed,
    )

    linear_pipeline_builder = lambda: build_model_pipeline(
        pred_vars=predictor_cols,
        cat_vars=categorical_cols,
        id_vars=params["model"]["predictor"]["id"],
    )

    lgbm_params = _build_lgbm_params_from_files(model_params=model_params, ccao_params=params, seed=seed, use_ccao_fallback=use_ccao_fallback)
    model_specs = _build_model_specs(
        lgbm_params=lgbm_params,
        rho_values=rho_values,
        keep_values=keep_values,
        fairness_ratio_mode=fairness_ratio_mode,
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
    )

    data_id = str(cv_out["data_id"])
    split_id = str(cv_out["split_id"])

    analysis_dir = Path(result_root) / "analysis" / f"data_id={data_id}" / f"split_id={split_id}"
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

    val_fraction = float(args.val_fraction) if (args.val_fraction is not None and float(args.val_fraction) > 0) else None

    rho_values_raw = _parse_float_list(str(args.rho_values))
    rho_values = _build_rho_values(
        rho_values_raw,
        rho_count=int(args.rho_count),
        rho_scale=str(args.rho_scale),
    )

    out = run_full_pipeline(
        result_root=str(args.result_root),
        data_path=str(args.data_path),
        sample_frac=(None if args.sample_frac is None else float(args.sample_frac)),
        seed=int(args.seed),
        rho_values=rho_values,
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
    )
    print("=" * 90)
    print("TEMPORAL CV COMPLETED")
    print("=" * 90)
    print(f"data_id={out['data_id']} | split_id={out['split_id']}")
    print(f"analysis_dir={out['analysis_dir']}")
    print(f"test_metrics_csv={out['test_metrics_csv']}")
    print(f"test_predictions_parquet={out['test_predictions_parquet']}")

