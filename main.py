import pandas as pd
import numpy as np
import yaml

from sklearn.linear_model import LinearRegression
import lightgbm as lgb

from preprocessing.recipes_pipelined import build_model_pipeline
from utils.motivation_utils import run_robust_rolling_origin_cv
from soft_constrained_models.boosting_models import (
    LGBCovPenalty,
    LGBSmoothPenalty,
    # LGBPrimalDual,
    # LGBVarCondMeanPenalty,
    # LGBCovDispPenalty,
    # LGBBinIndepSurrogatePenalty,
)


# -----------------------------------------------------------------------------
# Helper builders
# -----------------------------------------------------------------------------
def _build_lgbm_params_from_ccao_params(ccao_params: dict, seed: int) -> dict:
    # Map the CCAO-style params.yaml defaults into sklearn's LGBMRegressor args.
    hp_default = ccao_params["model"]["hyperparameter"]["default"]

    num_leaves = int(hp_default["num_leaves"])
    add_to_linked_depth = int(hp_default.get("add_to_linked_depth", 4))
    max_depth = int(np.floor(np.log2(max(num_leaves, 2))) + add_to_linked_depth)

    return {
        "boosting_type": "gbdt",
        "objective": "mse",
        "n_estimators": int(hp_default["num_iterations"]),
        "learning_rate": float(hp_default["learning_rate"]),
        "num_leaves": num_leaves,
        "max_depth": max_depth,
        "max_bin": int(hp_default["max_bin"]),
        "min_child_samples": int(hp_default["min_data_in_leaf"]),
        "min_split_gain": float(hp_default["min_gain_to_split"]),
        "colsample_bytree": float(hp_default["feature_fraction"]),
        "reg_alpha": float(hp_default["lambda_l1"]),
        "reg_lambda": float(hp_default["lambda_l2"]),
        "max_cat_threshold": int(hp_default["max_cat_threshold"]),
        "min_data_per_group": int(hp_default["min_data_per_group"]),
        "cat_smooth": float(hp_default["cat_smooth"]),
        "cat_l2": float(hp_default["cat_l2"]),
        "random_state": seed,
        "n_jobs": 1,
        "verbosity": -1,
        "importance_type": "split",
    }


def _build_model_specs(lgbm_params: dict) -> list:
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

    # Smooth penalty (rho range around mild and moderate).
    for rho in np.logspace(start=-2, stop=10, num=10):
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

    # Direct covariance penalty.
    for rho in np.logspace(start=-2, stop=10, num=10):
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

    return specs


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # 0) Global experiment inputs
    # -------------------------------------------------------------------------
    seed = 2025
    target_column = "meta_sale_price"
    date_column = "meta_sale_date"
    # Optional random down-sampling for faster experiments on huge datasets.
    # Set to a float in (0, 1), e.g. 0.10 for 10%; keep as None for full data.
    sample_frac = 0.40
    sample_seed = 2025

    with open("params.yaml", "r", encoding="utf-8") as file:
        params = yaml.safe_load(file)

    # -------------------------------------------------------------------------
    # 1) Load, clean, and (optionally) sample the training data
    # -------------------------------------------------------------------------
    df = pd.read_parquet(f"./data/CCAO/{seed}/training_data.parquet", engine="fastparquet")
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
    lgbm_params = _build_lgbm_params_from_ccao_params(params, seed=seed)
    model_specs = _build_model_specs(lgbm_params=lgbm_params)

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
        "step_months": 9, # Only for fixed-time validation windows.
        "min_train_rows": 300,
        "min_val_rows": 100,
    }
    bootstrap_protocol = {
        "type": "time_block",
        "block_freq": "M",
        "n_bootstrap": 1,
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
    )

    # -------------------------------------------------------------------------
    # 6) Print high-level run summary
    # -------------------------------------------------------------------------
    print("=" * 90)
    print("ROBUST TEMPORAL CV COMPLETED")
    print("=" * 90)
    print(f"data_id={robust_cv['data_id']} | split_id={robust_cv['split_id']} | folds={robust_cv['fold_count']}")
    print(f"new runs={robust_cv['run_records'].shape[0]} | new bootstrap_rows={robust_cv['bootstrap_records'].shape[0]}")

