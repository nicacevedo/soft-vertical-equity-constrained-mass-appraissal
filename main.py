import pandas as pd
import numpy as np
import yaml
from pathlib import Path

from sklearn.linear_model import LinearRegression
import lightgbm as lgb

from preprocessing.recipes_pipelined import build_model_pipeline
from utils.motivation_utils import (
    run_robust_rolling_origin_cv,
    _compute_extended_metrics,
    _stable_hash,
)
from soft_constrained_models.boosting_models import LGBCovPenalty, LGBSmoothPenalty


# -----------------------------------------------------------------------------
# Helper builders
# -----------------------------------------------------------------------------
def _build_lgbm_params_from_files(model_params: dict, ccao_params: dict, seed: int) -> dict:
    """
    Build sklearn-style LGBMRegressor kwargs using model_params.yaml as the
    primary default source, with fallback values from params.yaml defaults.
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


def _build_model_specs(
    lgbm_params: dict,
    rho_values: np.ndarray,
    cvar_keep_values: list[float],
) -> list:
    # Define all model configurations evaluated in rolling-origin CV.
    # Each item includes:
    # - a human-readable name
    # - a factory() that instantiates the model
    # - whether preprocessing pipeline is required
    # - a config payload used to build deterministic run IDs
    specs = [
        {
            "name": "LinearRegression",
            "factory": lambda: LinearRegression(fit_intercept=True),
            "requires_linear_pipeline": True,
            "config": {"fit_intercept": True},
        },
        {
            "name": "LGBMRegressor",
            "factory": lambda: lgb.LGBMRegressor(**lgbm_params),
            "requires_linear_pipeline": False,
            "config": {"model": "LGBMRegressor", **lgbm_params},
        },
    ]

    # Smooth penalty (rho sweep).
    for rho in rho_values:
        specs.append(
            {
                "name": f"LGBSmoothPenalty_rho_{rho}",
                "factory": lambda rho=rho: LGBSmoothPenalty(
                    rho=rho,
                    ratio_mode="diff",
                    zero_grad_tol=1e-12,
                    lgbm_params=lgbm_params,
                    verbose=False,
                ),
                "requires_linear_pipeline": False,
                "config": {"model": "LGBSmoothPenalty", "rho": rho, "ratio_mode": "diff", "lgbm_params": lgbm_params},
            }
        )

    # Direct covariance penalty (rho sweep).
    for rho in rho_values:
        specs.append(
            {
                "name": f"LGBCovPenalty_rho_{rho}",
                "factory": lambda rho=rho: LGBCovPenalty(
                    rho=rho,
                    ratio_mode="diff",
                    zero_grad_tol=1e-12,
                    lgbm_params=lgbm_params,
                    verbose=False,
                ),
                "requires_linear_pipeline": False,
                "config": {"model": "LGBCovPenalty", "rho": rho, "ratio_mode": "diff", "lgbm_params": lgbm_params},
            }
        )

    # # CVaR-style smooth variants using primal-dual capped-simplex weighting.
    # # - "CVaR" maps to individual adversaries (MSE and fairness terms weighted separately)
    # # - "CVaRTotal" maps to overall adversary (single joint robust objective)
    # for keep in cvar_keep_values:
    #     for rho in rho_values:
    #         specs.append(
    #             {
    #                 "name": f"LGBSmoothPenaltyCVaR_keep_{keep}_rho_{rho}",
    #                 "factory": lambda keep=keep, rho=rho: LGBPrimalDual(
    #                     rho=rho,
    #                     keep=keep,
    #                     adversary_type="individual",
    #                     eta_adv=0.1,
    #                     zero_grad_tol=1e-12,
    #                     lgbm_params=lgbm_params,
    #                 ),
    #                 "requires_linear_pipeline": False,
    #                 "config": {
    #                     "model": "LGBSmoothPenaltyCVaR",
    #                     "rho": rho,
    #                     "keep": keep,
    #                     "adversary_type": "individual",
    #                     "eta_adv": 0.1,
    #                     "lgbm_params": lgbm_params,
    #                 },
    #             }
    #         )
    #         specs.append(
    #             {
    #                 "name": f"LGBSmoothPenaltyCVaRTotal_keep_{keep}_rho_{rho}",
    #                 "factory": lambda keep=keep, rho=rho: LGBPrimalDual(
    #                     rho=rho,
    #                     keep=keep,
    #                     adversary_type="overall",
    #                     eta_adv=0.1,
    #                     zero_grad_tol=1e-12,
    #                     lgbm_params=lgbm_params,
    #                 ),
    #                 "requires_linear_pipeline": False,
    #                 "config": {
    #                     "model": "LGBSmoothPenaltyCVaRTotal",
    #                     "rho": rho,
    #                     "keep": keep,
    #                     "adversary_type": "overall",
    #                     "eta_adv": 0.1,
    #                     "lgbm_params": lgbm_params,
    #                 },
    #             }
    #         )

    return specs


def _evaluate_models_on_test_set(
    *,
    df_train_validate: pd.DataFrame,
    df_test: pd.DataFrame,
    predictor_cols: list[str],
    categorical_cols: list[str],
    target_col: str,
    date_col: str,
    model_specs: list,
    linear_pipeline_builder,
    fairness_ratio_mode: str,
    result_root: str,
    data_id: str,
    split_id: str,
) -> dict:
    """
    Fit each model on train_validate and evaluate on the held-out test split.
    Saves:
      - analysis/.../test_metrics.csv
      - analysis/.../test_predictions.parquet
    """
    analysis_dir = Path(result_root) / "analysis" / f"data_id={data_id}" / f"split_id={split_id}"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    X_train = df_train_validate[predictor_cols].copy()
    y_train_log = np.log(df_train_validate[target_col].to_numpy())
    X_test = df_test[predictor_cols].copy()
    y_test_log = np.log(df_test[target_col].to_numpy())

    cat_cols = [c for c in categorical_cols if c in X_train.columns]
    for c in cat_cols:
        X_train[c] = X_train[c].astype("category")
        X_test[c] = X_test[c].astype("category")

    test_rows = []
    pred_chunks = []
    test_min_sale_date = str(pd.to_datetime(df_test[date_col]).min().date())
    test_max_sale_date = str(pd.to_datetime(df_test[date_col]).max().date())

    # Stable row id for joins in downstream stacking.
    test_row_id = np.arange(df_test.shape[0], dtype=int)

    for spec in model_specs:
        model_name = str(spec["name"])
        model_config = dict(spec.get("config", {}))
        config_id = _stable_hash({"model_name": model_name, "config": model_config})

        estimator = spec["factory"]()
        if bool(spec.get("requires_linear_pipeline", False)):
            linear_pipeline = linear_pipeline_builder()
            X_train_model = linear_pipeline.fit_transform(X_train, y_train_log)
            X_test_model = linear_pipeline.transform(X_test)
        else:
            X_train_model = X_train
            X_test_model = X_test

        estimator.fit(X_train_model, y_train_log)
        y_pred_test_log = np.asarray(estimator.predict(X_test_model), dtype=float).reshape(-1)
        metrics = _compute_extended_metrics(
            y_true_log=y_test_log,
            y_pred_log=y_pred_test_log,
            y_train_log=y_train_log,
            ratio_mode=fairness_ratio_mode,
        )

        rho_val = model_config.get("rho", np.nan)
        try:
            rho_val = float(rho_val)
        except Exception:
            rho_val = np.nan

        test_rows.append(
            {
                "split_name": "test",
                "data_id": str(data_id),
                "split_id": str(split_id),
                "config_id": str(config_id),
                "model_name": model_name,
                "model_repr": model_name,
                "rho": rho_val,
                "test_rows": int(df_test.shape[0]),
                "test_min_sale_date": test_min_sale_date,
                "test_max_sale_date": test_max_sale_date,
                **metrics,
            }
        )
        pred_chunks.append(
            pd.DataFrame(
                {
                    "split_name": "test",
                    "data_id": str(data_id),
                    "split_id": str(split_id),
                    "config_id": str(config_id),
                    "model_name": model_name,
                    "row_id": test_row_id,
                    "sale_date": pd.to_datetime(df_test[date_col]).to_numpy(),
                    "y_true_log": y_test_log,
                    "y_pred_log": y_pred_test_log,
                    "y_true": np.exp(y_test_log),
                    "y_pred": np.exp(y_pred_test_log),
                    "y_train_log_mean": float(np.mean(y_train_log)),
                }
            )
        )

    metrics_df = pd.DataFrame(test_rows)
    preds_df = pd.concat(pred_chunks, ignore_index=True) if pred_chunks else pd.DataFrame()

    metrics_path = analysis_dir / "test_metrics.csv"
    preds_path = analysis_dir / "test_predictions.parquet"
    if not metrics_df.empty:
        metrics_df.to_csv(metrics_path, index=False)
    if not preds_df.empty:
        preds_df.to_parquet(preds_path, index=False, engine="fastparquet")

    return {
        "test_metrics_csv": str(metrics_path),
        "test_predictions_parquet": str(preds_path),
        "n_test_models": int(metrics_df.shape[0]),
        "n_test_prediction_rows": int(preds_df.shape[0]),
    }


def run_full_pipeline() -> None:
    # -------------------------------------------------------------------------
    # 0) Global experiment inputs
    # -------------------------------------------------------------------------
    seed = 2025 * 2
    target_column = "meta_sale_price"
    date_column = "meta_sale_date"
    # Optional random down-sampling for faster experiments on huge datasets.
    # Set to a float in (0, 1), e.g. 0.10 for 10%; keep as None for full data.
    sample_frac = 1#0.4
    sample_seed = seed #2025 
    # Parallel CV controls (minimal, opt-in)
    cv_parallel_enabled = True
    cv_parallel_cpu_fraction = 0.90  # percentage of detected CPUs to use
    cv_parallel_max_workers = 32   # set int to hard-cap workers (e.g., 6)
    cv_parallel_backend = "loky"     # "loky" (processes) or "threading"

    with open("params.yaml", "r", encoding="utf-8") as file:
        params = yaml.safe_load(file)
    with open("model_params.yaml", "r", encoding="utf-8") as file:
        model_params = yaml.safe_load(file)

    # -------------------------------------------------------------------------
    # 1) Load, clean, and (optionally) sample the training data
    # -------------------------------------------------------------------------
    df = pd.read_parquet(f"./data/CCAO/2025/training_data.parquet", engine="fastparquet")
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
            print(
                f"[sampling] using random sample with frac={float(sample_frac):.4f}, "
                f"seed={int(sample_seed)}, n_rows={df.shape[0]}"
            )
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(date_column).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # 2) Out-of-time split protocol
    #    - assessment: future year used only for final prediction stage
    #    - test: most recent 10% of pre-assessment sales
    #    - train/validate: remaining older sales used in rolling-origin CV
    # -------------------------------------------------------------------------
    df_assess = df.loc[df[date_column].dt.year == 2024, :].copy()
    df_train_all = df.loc[df[date_column].dt.year < 2024, :].copy()
    train_prop = float(params["cv"]["split_prop"])
    split_idx = int(train_prop * df_train_all.shape[0])
    df_test = df_train_all.iloc[split_idx:, :].copy()
    df_train_validate = df_train_all.iloc[:split_idx, :].copy()

    # -------------------------------------------------------------------------
    # 3) Build model parameterization and model list
    # -------------------------------------------------------------------------
    lgbm_params = _build_lgbm_params_from_files(model_params=model_params, ccao_params=params, seed=seed)
    rho_values = np.logspace(start=-3, stop=1, num=10)
    cvar_keep_values = [0.9, 0.8, 0.7]
    model_specs = _build_model_specs(
        lgbm_params=lgbm_params,
        rho_values=rho_values,
        cvar_keep_values=cvar_keep_values,
    )

    # -------------------------------------------------------------------------
    # 4) Configure rolling-origin and bootstrap protocols
    #    Rolling-origin here follows CCAO-style logic:
    #    train expands over time, and each validation block is the next fixed
    #    10% chunk of the train/validate universe (val_fraction = 0.10).
    # -------------------------------------------------------------------------
    split_protocol = {
        "train_mode": "expanding",
        "initial_train_months": 9,
        # Match CCAO rolling-origin figure: validation is the next fixed 10% block
        # of the train/validate universe, immediately after each training window.
        "val_fraction": 0.10,
        # "val_window_months": 1, # Only for fixed-time validation windows.
        "step_months": 9,
        "min_train_rows": 300,
        "min_val_rows": 100,
    }
    bootstrap_protocol = {
        "type": "time_block",
        "block_freq": "M",
        "n_bootstrap": 100,
        "seed": seed,
    }
    data_signature = {
        "source": "CCAO",
        "seed": seed,
        "target_col": target_column,
        "date_col": date_column,
        "predictors": predictor_cols,
        "categoricals": categorical_cols,
        "n_rows_train_validate": int(df_train_validate.shape[0]),
        "n_rows_test": int(df_test.shape[0]),
        "n_rows_assess": int(df_assess.shape[0]),
        "date_min_train_validate": str(df_train_validate[date_column].min().date()),
        "date_max_train_validate": str(df_train_validate[date_column].max().date()),
        "preprocess_version": "build_model_pipeline_v1",
        "sample_frac": (None if sample_frac is None else float(sample_frac)),
        "sample_seed": int(sample_seed),
    }

    # Build preprocessing pipeline factory used by linear models.
    linear_pipeline_builder = lambda: build_model_pipeline(
        pred_vars=predictor_cols,
        cat_vars=categorical_cols,
        id_vars=params["model"]["predictor"]["id"],
    )

    # -------------------------------------------------------------------------
    # 5) Run robust rolling-origin CV and persist artifacts
    # -------------------------------------------------------------------------
    robust_cv = run_robust_rolling_origin_cv(
        df_train_validate=df_train_validate,
        date_col=date_column,
        target_col=target_column,
        predictor_cols=predictor_cols,
        categorical_cols=categorical_cols,
        model_specs=model_specs,
        linear_pipeline_builder=linear_pipeline_builder,
        result_root="./output/robust_rolling_origin_cv",
        data_signature=data_signature,
        split_protocol=split_protocol,
        bootstrap_protocol=bootstrap_protocol,
        fairness_ratio_mode="diff",
        predict_store=True,
        parquet_engine="fastparquet",
        log_progress=True,
        parallel_enabled=cv_parallel_enabled,
        parallel_cpu_fraction=cv_parallel_cpu_fraction,
        parallel_max_workers=cv_parallel_max_workers,
        parallel_backend=cv_parallel_backend,
    )

    test_eval = _evaluate_models_on_test_set(
        df_train_validate=df_train_validate,
        df_test=df_test,
        predictor_cols=predictor_cols,
        categorical_cols=categorical_cols,
        target_col=target_column,
        date_col=date_column,
        model_specs=model_specs,
        linear_pipeline_builder=linear_pipeline_builder,
        fairness_ratio_mode="diff",
        result_root="./output/robust_rolling_origin_cv",
        data_id=str(robust_cv["data_id"]),
        split_id=str(robust_cv["split_id"]),
    )

    # -------------------------------------------------------------------------
    # 6) Print high-level run summary
    # -------------------------------------------------------------------------
    print("=" * 90)
    print("ROBUST TEMPORAL CV COMPLETED")
    print("=" * 90)
    print(f"data_id={robust_cv['data_id']} | split_id={robust_cv['split_id']} | folds={robust_cv['fold_count']}")
    print(f"new runs={robust_cv['run_records'].shape[0]} | new bootstrap_rows={robust_cv['bootstrap_records'].shape[0]}")
    print(
        f"test_metrics_csv={test_eval['test_metrics_csv']} | "
        f"test_predictions_parquet={test_eval['test_predictions_parquet']}"
    )


if __name__ == "__main__":
    run_full_pipeline()

