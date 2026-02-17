# ======================== Clean and Simplified Version ========================= 
import pandas as pd
import numpy as np 
import yaml

# Baseline models
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge
import lightgbm as lgb

# Preprocessing Pipelines
from preprocessing.recipes_pipelined import build_model_pipeline, build_model_pipeline_supress_onehot

# Utils & Metrics
from utils.motivation_utils import (
    get_rolling_origin_indices,
    compute_taxation_metrics,
    run_rho_sweep_and_tradeoff,
)  # , get_sliding_window_indices,

# Custom Linear Models
# # from soft_constrained_models.linear_models import LeastAbsoluteDeviationRegression, MaxDeviationConstrainedLinearRegression, LeastMaxDeviationRegression, GroupDeviationConstrainedLinearRegression, StableRegression, LeastProportionalDeviationRegression#LeastMSEConstrainedRegression, LeastProportionalDeviationRegression
# # from soft_constrained_models.linear_models import MyGLMRegression, GroupDeviationConstrainedLogisticRegression, RobustStableLADPRDCODRegressor, StableAdversarialSurrogateRegressor, StableAdversarialSurrogateRegressor2

# Custom Boosting Models
from soft_constrained_models.boosting_models import LGBSmoothPenalty, LGBCovPenalty, LGBSmoothPenaltyCVaR, LGBSmoothPenaltyCVaRTotal


# NOTE: Basic pipeline to test the current "best" models.
# The "compute_taxation_metrics" measures all the relevant metrics for this specific application.
# This includes


if __name__ == "__main__":
    # Inputs
    seed = 2025
    source = "CCAO"
    target_column = "meta_sale_price"
    date_column = "meta_sale_date"
    # Optional fast-run controls (set sample_enabled=False for full data)
    sample_enabled = True
    sample_frac = .5   # use None to disable fraction-based sampling
    sample_n = None      # use int (e.g., 120_000) to sample fixed size; cannot be used with sample_frac

    # 1. Data source: It is supposed to be after "ingest" step
    df = pd.read_parquet(f"./data/CCAO/{seed}/training_data.parquet", engine="fastparquet")

    # 2. Filter of columns
    # 2.1. Direct ones
    df = df[
        (~df['ind_pin_is_multicard'].astype('bool').fillna(True)) &
        (~df['sv_is_outlier'].astype('bool').fillna(True))
    ]
    # 2.2. Filter by desired columns in params file
    with open('params.yaml', 'r') as file:
        params = yaml.safe_load(file)
    df = df.loc[:, params['model']['predictor']['all'] +  [target_column, date_column] ]

    # 2.3 Optional random sample to speed up model comparisons
    if sample_enabled:
        if (sample_frac is not None) and (sample_n is not None):
            raise ValueError("Use either sample_frac or sample_n, not both.")
        if sample_frac is not None:
            if not (0 < sample_frac <= 1):
                raise ValueError("sample_frac must be in (0, 1].")
            df = df.sample(frac=sample_frac, random_state=seed)
            print(f"Sampling enabled: using {sample_frac:.1%} of rows -> n={df.shape[0]:,}")
        elif sample_n is not None:
            if sample_n <= 0:
                raise ValueError("sample_n must be > 0.")
            sample_n_eff = min(int(sample_n), int(df.shape[0]))
            df = df.sample(n=sample_n_eff, random_state=seed)
            print(f"Sampling enabled: using fixed n={sample_n_eff:,} rows")
        else:
            print("Sampling enabled, but no sample_frac/sample_n provided. Using full data.")

    # 3. Data splits: training, validation, testing, assessment
    df.sort_values(by=date_column, ascending=True, inplace=True) # Sort by sale date
    # 3.1. Assessment data: 2024
    df_assess = df.loc[df[date_column].dt.year == 2024, :]
    df_train = df.loc[~df.index.isin(df_assess.index) ] # Data for training process

    # 3.2. Testing data: ~2023
    train_prop = 0.9
    df_test = df_train.iloc[int(train_prop*df_train.shape[0]):,:]
    df_train = df_train.loc[~df_train.index.isin(df_test.index), :] # Train/validate data

    # 3.3. Rolling-Origin sample map of indices: 2016-2022
    folds = get_rolling_origin_indices(df_train, date_col=date_column, growth_window='9MS', val_pct=0.10)
    print("# of generated folds: ", len(folds))

    # 4. List of models to compare: Baseline models v/s covariance-penalized LGBM
    with open('model_params.yaml', 'r') as file:
        model_params = yaml.safe_load(file)
    models = [
        LinearRegression(fit_intercept=True),                   # Linear Regression
        lgb.LGBMRegressor(**model_params["LGBMRegressor"]),     # LightGBM
        LGBCovPenalty(
            rho=6.58,               # rho: Cov-Penalty weight
            ratio_mode="diff",      # ratio proxy: [v1]="diff" -> Cov(log-residuals, log-price) || [v2]="div" -> Cov(log-ratio, log-price)
            zero_grad_tol=1e-12,    # Safeguard for 0-valued coordinates in the gradient computation for LGBM
            lgbm_params=model_params["LGBMRegressor"]   # Same hyperparameters as the baseline LGBM
        ),
        LGBSmoothPenalty(                                       # LGBMSurrPenalty
            rho=2.043,              # rho: Cov-Penalty weight
            ratio_mode="diff",      # ratio proxy: [v1]="diff" -> Cov(log-residuals, log-price) || [v2]="div" -> Cov(log-ratio, log-price)
            zero_grad_tol=1e-12,    # Safeguard for 0-valued coordinates in the gradient computation for LGBM
            lgbm_params=model_params["LGBMRegressor"]   # Same hyperparameters as the baseline LGBM
        ),
        LGBSmoothPenaltyCVaR(
            rho=2.043,              # rho: Cov-Penalty weight
            ratio_mode="diff",      # ratio proxy: [v1]="diff" -> Cov(log-residuals, log-price) || [v2]="div" -> Cov(log-ratio, log-price)
            zero_grad_tol=1e-12,    # Safeguard for 0-valued coordinates in the gradient computation for LGBM
            # CVaR robustness knobs
            mse_keep=.9,            # keep fraction for CVaR on squared error; 1.0 => original mean MSE
            mse_mix_uniform=0.0,     # optional mixing with uniform weights in [0,1); default 0.0
            lgbm_params=model_params["LGBMRegressor"]   # Same hyperparameters as the baseline LGBM
        ),
        LGBSmoothPenaltyCVaR(
            rho=2.043,              # rho: Cov-Penalty weight
            ratio_mode="diff",      # ratio proxy: [v1]="diff" -> Cov(log-residuals, log-price) || [v2]="div" -> Cov(log-ratio, log-price)
            zero_grad_tol=1e-12,    # Safeguard for 0-valued coordinates in the gradient computation for LGBM
            # CVaR robustness knobs
            mse_keep=.8,            # keep fraction for CVaR on squared error; 1.0 => original mean MSE
            mse_mix_uniform=0.0,     # optional mixing with uniform weights in [0,1); default 0.0
            lgbm_params=model_params["LGBMRegressor"]   # Same hyperparameters as the baseline LGBM
        ),
        LGBSmoothPenaltyCVaR(
            rho=2.043,              # rho: Cov-Penalty weight
            ratio_mode="diff",      # ratio proxy: [v1]="diff" -> Cov(log-residuals, log-price) || [v2]="div" -> Cov(log-ratio, log-price)
            zero_grad_tol=1e-12,    # Safeguard for 0-valued coordinates in the gradient computation for LGBM
            # CVaR robustness knobs
            mse_keep=.7,            # keep fraction for CVaR on squared error; 1.0 => original mean MSE
            mse_mix_uniform=0.0,     # optional mixing with uniform weights in [0,1); default 0.0
            lgbm_params=model_params["LGBMRegressor"]   # Same hyperparameters as the baseline LGBM
        ),
        LGBSmoothPenaltyCVaRTotal(
            rho=2.043,              # rho: Cov-Penalty weight
            ratio_mode="diff",      # ratio proxy: [v1]="diff" -> Cov(log-residuals, log-price) || [v2]="div" -> Cov(log-ratio, log-price)
            zero_grad_tol=1e-12,    # Safeguard for 0-valued coordinates in the gradient computation for LGBM
            # CVaR robustness knobs
            keep=.9,            # keep fraction for CVaR on squared error; 1.0 => original mean MSE
            mix_uniform=0.0,     # optional mixing with uniform weights in [0,1); default 0.0
            lgbm_params=model_params["LGBMRegressor"]   # Same hyperparameters as the baseline LGBM
        ),
        LGBSmoothPenaltyCVaRTotal(
            rho=2.043,              # rho: Cov-Penalty weight
            ratio_mode="diff",      # ratio proxy: [v1]="diff" -> Cov(log-residuals, log-price) || [v2]="div" -> Cov(log-ratio, log-price)
            zero_grad_tol=1e-12,    # Safeguard for 0-valued coordinates in the gradient computation for LGBM
            # CVaR robustness knobs
            keep=.8,            # keep fraction for CVaR on squared error; 1.0 => original mean MSE
            mix_uniform=0.0,     # optional mixing with uniform weights in [0,1); default 0.0
            lgbm_params=model_params["LGBMRegressor"]   # Same hyperparameters as the baseline LGBM
        ),
        LGBSmoothPenaltyCVaRTotal(
            rho=2.043,              # rho: Cov-Penalty weight
            ratio_mode="diff",      # ratio proxy: [v1]="diff" -> Cov(log-residuals, log-price) || [v2]="div" -> Cov(log-ratio, log-price)
            zero_grad_tol=1e-12,    # Safeguard for 0-valued coordinates in the gradient computation for LGBM
            # CVaR robustness knobs
            keep=.7,            # keep fraction for CVaR on squared error; 1.0 => original mean MSE
            mix_uniform=0.0,     # optional mixing with uniform weights in [0,1); default 0.0
            lgbm_params=model_params["LGBMRegressor"]   # Same hyperparameters as the baseline LGBM
        ),
    ]

    # 5. Pipeline for preprocessing linear models
    linear_pipeline = build_model_pipeline(                             # X to use linear models directly
        pred_vars=params['model']['predictor']['all'],
        cat_vars=params['model']['predictor']['categorical'],
        id_vars=params['model']['predictor']['id'],
    )

    # # ----- Rolling-Origin: train/validation split, fit, and validation metrics -----
    # global_results = []
    # for fold_idx, fold_pair in enumerate(folds):
    #     train_idx, val_idx = fold_pair
    #     print("-"*10)
    #     df_train_fold = df_train.iloc[train_idx,:]
    #     df_val_fold = df_train.iloc[val_idx,:]

    #     # 4. Preprocess the data
    #     # 4.1. Define X and y for train and test on the fold
    #     X_train, y_train = df_train_fold.drop(columns=['meta_sale_date', 'meta_sale_price']), df_train_fold['meta_sale_price']
    #     X_val, y_val = df_val_fold.drop(columns=['meta_sale_date', 'meta_sale_price']), df_val_fold['meta_sale_price']
        
    #     # 4.2. Log-prices as objective
    #     y_train_log = np.log(y_train)
    #     y_val_log = np.log(y_val)

    #     # 4.2. Fit pipeline on training set and transform the validation set
    #     X_train_prep = linear_pipeline.fit_transform(X_train, y_train_log)
    #     X_val_prep = linear_pipeline.transform(X_val)

    #     # 4.3. Category type-correction
    #     cat_cols = [col for col in set(X_train.columns) & set(params['model']['predictor']['categorical']) ]
    #     X_train[cat_cols] = X_train[cat_cols].astype("category")
    #     X_val[cat_cols] = X_val[cat_cols].astype("category")

    #     # 5. Evolution of the metrics on different folds train/val
    #     fold_results = []
    #     for model in models:
    #         print(f"Fitting model={str(model)} in fold={fold_idx}...")
    #         if "LGB" in str(model): # LGBM training
    #             model.fit(X_train, y_train_log)
    #             y_pred_log = model.predict(X_val)
    #         else:
    #             model.fit(X_train_prep, y_train_log) # Linear training
    #             y_pred_log = model.predict(X_val_prep)

    #         model_result = {"model":str(model), "fold":fold_idx} | compute_taxation_metrics(y_val_log, y_pred_log, scale="log", y_train=y_train_log)
    #         fold_results.append(model_result)
    #         global_results.append(model_result)

    #         # Resulting dataframes
    #         df_results_fold = pd.DataFrame(fold_results)
    #         print(f"Rolling-Origin Results for model={str(model)} in fold={fold_idx}:")
    #         print(df_results_fold)

    #         df_results = pd.DataFrame(global_results)
    #         print(f"Rolling-Origin Results for model={str(model)} in fold={fold_idx}:")
    #         print(df_results)

    

    # Check on the testing and assessment sets

    # ---- Test set results (~2023) ----

    # Preprocess the data
    X_train, y_train = df_train.drop(columns=['meta_sale_date', 'meta_sale_price']), df_train['meta_sale_price']
    X_test, y_test = df_test.drop(columns=['meta_sale_date', 'meta_sale_price']), df_test['meta_sale_price']
    
   # Log-prices as objective
    y_train_log = np.log(y_train)
    y_test_log = np.log(y_test)

    # Fit pipeline on training set and transform the testing set
    X_train_prep = linear_pipeline.fit_transform(X_train, y_train_log)
    X_test_prep = linear_pipeline.transform(X_test)

    # Category type-correction
    cat_cols = [col for col in set(X_train.columns) & set(params['model']['predictor']['categorical']) ]
    X_train[cat_cols] = X_train[cat_cols].astype("category")
    X_test[cat_cols] = X_test[cat_cols].astype("category")

    # Evolution of the metrics on different folds train/test
    run_rho_sweep = True
    if run_rho_sweep:
        rho_grid = np.linspace(0, 7, 10)#np.array([0.5, 1.0, 2.0, 5.0, 10.0], dtype=float)
        metrics_df = run_rho_sweep_and_tradeoff(
            models=models,
            rho_values=rho_grid,
            X_train=X_train,
            y_train=y_train_log,
            X_test=X_test,
            y_test=y_test_log,
            scale="log",
            predictions_cache_path="./tmp/rho_sweep_predictions_test.pkl",
            metrics_csv_path="./tmp/rho_sweep_metrics_test.csv",
            plots_dir="./img/rho_tradeoff_test",
            load_predictions_if_exists=True,
            save_predictions=True,
            penalty_attrs=("rho",),
            include_models_without_rho=False,
            verbose=True,
        )
        print("Rho-sweep metrics (test split):")
        print(metrics_df)
        exit()

    # Evolution of the metrics on different folds train/test
    fold_results = []
    for model in models:
        print(f"Fitting model={str(model)} in fold=test...")
        if "LGB" in str(model): # LGBM training
            model.fit(X_train, y_train_log)
            y_pred_log = model.predict(X_test)
        else:
            model.fit(X_train_prep, y_train_log) # Linear training
            y_pred_log = model.predict(X_test_prep)

        model_result = {"model":str(model), "fold":"test"} | compute_taxation_metrics(y_test_log, y_pred_log, scale="log", y_train=y_train_log)
        fold_results.append(model_result)

        # DELETE: check the mean of assessment ratio by bin of log y
        print("-"*100)
        y_test_log_bins = pd.cut(y_test_log, bins=5)
        bin_dataframe = pd.DataFrame({'y_test_log': y_test_log, 'y_test_log_bins': y_test_log_bins, 'y_pred_log': y_pred_log})
        bin_dataframe['ratio'] = np.exp(bin_dataframe['y_pred_log']) / np.exp(bin_dataframe['y_test_log'])
        print("Mean of assessment ratio by bin of log y:")
        print(bin_dataframe.groupby('y_test_log_bins')['ratio'].mean())
        print("Std of assessment ratio by bin of log y:")
        print(bin_dataframe.groupby('y_test_log_bins')['ratio'].std())
        print("Correlation between y_test_log and ratio:")
        for bin in bin_dataframe['y_test_log_bins'].unique():
            print(f"Bin {bin}:")
            print(np.corrcoef( bin_dataframe[bin_dataframe['y_test_log_bins'] == bin]['ratio'],  bin_dataframe[bin_dataframe['y_test_log_bins'] == bin]['y_test_log'])[0, 1])
        print("-"*100)
        # END DELETE

        # Resulting dataframes
        df_results_fold = pd.DataFrame(fold_results)
        print(f"Testing Results for model={str(model)} in fold=test:")
        print(df_results_fold)


    # ---- Assessment Set (2024) -----

    # Preprocess the data
    df_train = pd.concat([df_train, df_test])
    X_train, y_train = df_train.drop(columns=['meta_sale_date', 'meta_sale_price']), df_train['meta_sale_price']
    X_test, y_test = df_assess.drop(columns=['meta_sale_date', 'meta_sale_price']), df_assess['meta_sale_price']
    
   # Log-prices as objective
    y_train_log = np.log(y_train)
    y_test_log = np.log(y_test)

    # Fit pipeline on training set and transform the testing set
    X_train_prep = linear_pipeline.fit_transform(X_train, y_train_log)
    X_test_prep = linear_pipeline.transform(X_test)

    # Category type-correction
    cat_cols = [col for col in set(X_train.columns) & set(params['model']['predictor']['categorical']) ]
    X_train[cat_cols] = X_train[cat_cols].astype("category")
    X_test[cat_cols] = X_test[cat_cols].astype("category")

    # Evolution of the metrics on different folds train/test
    fold_results = []
    for model in models:
        print(f"Fitting model={str(model)} in fold=test...")
        if "LGB" in str(model): # LGBM training
            model.fit(X_train, y_train_log)
            y_pred_log = model.predict(X_test)
        else:
            model.fit(X_train_prep, y_train_log) # Linear training
            y_pred_log = model.predict(X_test_prep)

        model_result = {"model":str(model), "fold":"test"} | compute_taxation_metrics(y_test_log, y_pred_log, scale="log", y_train=y_train_log)
        fold_results.append(model_result)

        # Resulting dataframes
        df_results_fold = pd.DataFrame(fold_results)
        print(f"Assessment Results for model={str(model)} in fold=test:")
        print(df_results_fold)


    exit()

