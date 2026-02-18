"""
Quick test runner.

Goal
----
Fit and evaluate the 4 core models on:
  - held-out test split (most recent pre-2024 sales; ~2023 by CCAO-style split)
  - assessment split (2024 sales)

This script intentionally mirrors the preprocessing + split logic in `main.py`,
but avoids CV and bootstrapping to stay fast and easy to read.

Models
------
1) LinearRegression (baseline)
2) LGBMRegressor (baseline; defaults from `model_params.yaml` + fallback `params.yaml`)
3) LGBSmoothPenalty (fairness-regularized; uses `rho`)
4) LGBCovPenalty (fairness-regularized; uses `rho`)

Outputs
-------
Writes 2 CSV tables under `--out-dir`:
  - quick_test_metrics_test.csv
  - quick_test_metrics_assess.csv

Each table contains accuracy + vertical equity metrics computed with the same
metric routine used elsewhere in this repo (`_compute_extended_metrics`).

Usage
-----
From the `soft-vertical-equity-constrained-mass-appraissal/` directory:

  python quick_test_models.py --rho 1.0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression
import lightgbm as lgb

from preprocessing.recipes_pipelined import build_model_pipeline
from soft_constrained_models.boosting_models import LGBCovPenalty, LGBSmoothPenalty
from utils.motivation_utils import _compute_extended_metrics


def _build_lgbm_params_from_files(model_params: dict, ccao_params: dict, seed: int) -> dict:
    """
    Match `main.py`: use `model_params.yaml` as primary defaults and fall back to
    `params.yaml` for any missing keys.
    """
    model_default = dict(model_params.get("LGBMRegressor", {}))
    hp_default = dict(ccao_params["model"]["hyperparameter"]["default"])

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
    Mirrors `main.py`:
      - load parquet
      - filter out multicard and outliers
      - keep only predictor + target + date
      - sort by date
      - split into assess (2024), and pre-assess (<2024) then train/validate + test
    """
    df = pd.read_parquet(data_path, engine="fastparquet")
    df = df[
        (~df["ind_pin_is_multicard"].astype("bool").fillna(True))
        & (~df["sv_is_outlier"].astype("bool").fillna(True))
    ].copy()

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

    df_assess = df.loc[df[date_column].dt.year == 2024, :].copy()
    df_train_all = df.loc[df[date_column].dt.year < 2024, :].copy()

    train_prop = float(params["cv"]["split_prop"])
    split_idx = int(train_prop * df_train_all.shape[0])
    df_test = df_train_all.iloc[split_idx:, :].copy()
    df_train_validate = df_train_all.iloc[:split_idx, :].copy()

    return df_train_validate, df_test, df_assess, predictor_cols, categorical_cols


def _fit_predict_and_score(
    *,
    model_name: str,
    estimator: Any,
    requires_linear_pipeline: bool,
    linear_pipeline_builder,
    X_train: pd.DataFrame,
    y_train_log: np.ndarray,
    X_eval: pd.DataFrame,
    y_eval_log: np.ndarray,
    fairness_ratio_mode: str,
) -> Dict[str, Any]:
    if requires_linear_pipeline:
        pipe = linear_pipeline_builder()
        X_train_m = pipe.fit_transform(X_train, y_train_log)
        X_eval_m = pipe.transform(X_eval)
    else:
        X_train_m = X_train
        X_eval_m = X_eval

    estimator.fit(X_train_m, y_train_log)
    y_pred_eval_log = np.asarray(estimator.predict(X_eval_m), dtype=float).reshape(-1)
    metrics = _compute_extended_metrics(
        y_true_log=y_eval_log,
        y_pred_log=y_pred_eval_log,
        y_train_log=y_train_log,
        ratio_mode=fairness_ratio_mode,
    )
    return {"model_name": model_name, **metrics}


def run_quick_test(
    *,
    rho: float,
    out_dir: str,
    data_path: str,
    sample_frac: float | None,
    seed: int,
) -> Dict[str, str]:
    """
    Runs the 4-model quick test and writes the output CSV tables.
    """
    target_column = "meta_sale_price"
    date_column = "meta_sale_date"

    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    with open("model_params.yaml", "r", encoding="utf-8") as f:
        model_params = yaml.safe_load(f)

    df_train_validate, df_test, df_assess, predictor_cols, categorical_cols = _load_and_split_data(
        data_path=data_path,
        params=params,
        target_column=target_column,
        date_column=date_column,
        sample_frac=sample_frac,
        sample_seed=seed,
    )

    # Pipeline builder for linear models (matches `main.py`).
    linear_pipeline_builder = lambda: build_model_pipeline(
        pred_vars=predictor_cols,
        cat_vars=categorical_cols,
        id_vars=params["model"]["predictor"]["id"],
    )

    # Train/eval matrices.
    X_tv = df_train_validate[predictor_cols].copy()
    y_tv_log = np.log(df_train_validate[target_column].to_numpy())

    X_test = df_test[predictor_cols].copy()
    y_test_log = np.log(df_test[target_column].to_numpy())

    X_assess = df_assess[predictor_cols].copy()
    y_assess_log = np.log(df_assess[target_column].to_numpy()) if not df_assess.empty else np.array([], dtype=float)

    # Categorical handling (matches `main.py`).
    cat_cols = [c for c in categorical_cols if c in X_tv.columns]
    for c in cat_cols:
        X_tv[c] = X_tv[c].astype("category")
        X_test[c] = X_test[c].astype("category")
        if not df_assess.empty:
            X_assess[c] = X_assess[c].astype("category")

    # Model parameterization (baseline LGBM defaults).
    lgbm_params = _build_lgbm_params_from_files(model_params=model_params, ccao_params=params, seed=seed)

    models = [
        ("LinearRegression", LinearRegression(fit_intercept=True), True),
        ("LGBMRegressor", lgb.LGBMRegressor(**lgbm_params), False),
        (
            f"LGBSmoothPenalty_rho_{rho}",
            LGBSmoothPenalty(
                rho=float(rho),
                ratio_mode="diff",
                zero_grad_tol=1e-12,
                lgbm_params=lgbm_params,
                verbose=True,
            ),
            False,
        ),
        (
            f"LGBCovPenalty_rho_{rho}",
            LGBCovPenalty(
                rho=float(rho),
                ratio_mode="diff",
                zero_grad_tol=1e-12,
                lgbm_params=lgbm_params,
                verbose=True,
            ),
            False,
        ),
    ]

    fairness_ratio_mode = "diff"

    # --- Evaluate on TEST (train on df_train_validate only; strict out-of-time).
    test_rows = []
    for name, est, needs_pipe in models:
        test_rows.append(
            _fit_predict_and_score(
                model_name=name,
                estimator=est,
                requires_linear_pipeline=needs_pipe,
                linear_pipeline_builder=linear_pipeline_builder,
                X_train=X_tv,
                y_train_log=y_tv_log,
                X_eval=X_test,
                y_eval_log=y_test_log,
                fairness_ratio_mode=fairness_ratio_mode,
            )
        )
    test_df = pd.DataFrame(test_rows)

    # --- Evaluate on ASSESS (train on ALL pre-2024 sales, i.e., train_validate + test).
    assess_df = pd.DataFrame()
    if not df_assess.empty:
        df_pre2024 = pd.concat([df_train_validate, df_test], ignore_index=True)
        X_pre = df_pre2024[predictor_cols].copy()
        y_pre_log = np.log(df_pre2024[target_column].to_numpy())
        for c in cat_cols:
            X_pre[c] = X_pre[c].astype("category")

        assess_rows = []
        for name, est, needs_pipe in models:
            assess_rows.append(
                _fit_predict_and_score(
                    model_name=name,
                    estimator=est,
                    requires_linear_pipeline=needs_pipe,
                    linear_pipeline_builder=linear_pipeline_builder,
                    X_train=X_pre,
                    y_train_log=y_pre_log,
                    X_eval=X_assess,
                    y_eval_log=y_assess_log,
                    fairness_ratio_mode=fairness_ratio_mode,
                )
            )
        assess_df = pd.DataFrame(assess_rows)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    test_path = out / "quick_test_metrics_test.csv"
    assess_path = out / "quick_test_metrics_assess.csv"
    test_df.to_csv(test_path, index=False)
    if not assess_df.empty:
        assess_df.to_csv(assess_path, index=False)

    return {"test_csv": str(test_path), "assess_csv": str(assess_path)}


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Quick test: 4 core models on test (~2023) + assessment (2024).")
    p.add_argument("--rho", type=float, default=1.0, help="Rho used for the two regularized models.")
    p.add_argument("--out-dir", type=str, default="./output/quick_test", help="Directory to write CSV outputs.")
    p.add_argument(
        "--data-path",
        type=str,
        default="./data/CCAO/2025/training_data.parquet",
        help="Path to training_data.parquet (same file used by main.py).",
    )
    p.add_argument("--sample-frac", type=float, default=None, help="Optional down-sampling fraction in (0,1].")
    p.add_argument("--seed", type=int, default=4050, help="Random seed (mirrors main.py default).")
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    out = run_quick_test(
        rho=float(args.rho),
        out_dir=str(args.out_dir),
        data_path=str(args.data_path),
        sample_frac=(None if args.sample_frac is None else float(args.sample_frac)),
        seed=int(args.seed),
    )
    print("=" * 90)
    print("QUICK TEST COMPLETED")
    print("=" * 90)
    print(f"test_csv={out['test_csv']}")
    print(f"assess_csv={out['assess_csv']}")

