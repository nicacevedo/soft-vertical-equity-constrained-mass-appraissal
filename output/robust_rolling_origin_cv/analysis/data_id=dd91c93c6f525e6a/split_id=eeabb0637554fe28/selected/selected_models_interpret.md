# Selected models — interpret stage

- result_root: `/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/output/robust_rolling_origin_cv`
- data_id: `dd91c93c6f525e6a`  
- split_id: `eeabb0637554fe28`  
- accuracy metric: `RMSE`  
- constraint metrics: `['PRD', 'PRB', 'VEI']`
- candidate pools:
  - `ccao_min_rmse`: 51 configs across 8 folds (families: ['LGBCovPenalty', 'LGBMRegressor'])
  - `nash`: 50 configs across 8 folds (families: ['LGBCovPenalty'])

## Selection rule: `ccao_min_rmse`
- **config_id:** `3e68c37f6db0d0dc`
- **model_name:** `LGBMRegressor`
- **model_family:** `LGBMRegressor`
- **n_folds:** 8
- **CV RMSE (mean ± std):** 101201 ± 11102.5
- **CV PRD mean:** 1.06701  (above [0.98, 1.03])
- **CV PRB mean:** -0.0752087  (below [-0.05, 0.05])
- **CV VEI mean:** -21.6686  (below [-10.0, 10.0])

**Held-out test metrics:**
  - R2: 0.885082
  - RMSE: 122395
  - MAE: 71912.1
  - PRD: 1.08514
  - PRB: -0.117753
  - VEI: -37.637
  - COD: 23.0711

**Hyperparameters (`model_config_json`):**

```
  - lgbm_base_config_id: 7ce5a0a0e6e4f38b
  - (LightGBM base hyperparameters from model_params.yaml::LGBMRegressor)
      · boosting_type: gbdt
      · class_weight: None
      · colsample_bytree: 1.0
      · importance_type: split
      · learning_rate: 0.012806
      · max_bin: 322
      · max_depth: 15
      · min_child_samples: 30
      · min_split_gain: 0.023317
      · n_estimators: 500
      · n_jobs: 1
      · num_leaves: 1209
      · objective: mse
      · random_state: 2025
      · reg_alpha: 1e-3
      · reg_lambda: 1e-2
      · subsample_for_bin: 200000
```

## Selection rule: `nash`
- **config_id:** `27c5ba8768c4d7e3`
- **model_name:** `LGBCovPenalty`
- **model_family:** `LGBCovPenalty`
- **n_folds:** 8
- **nash_log_utility:** -11.6582
- **CV RMSE (mean ± std):** 109757 ± 9454.5
- **CV PRD mean:** 1.02656  (inside [0.98, 1.03])
- **CV PRB mean:** -0.0149428  (inside [-0.05, 0.05])
- **CV VEI mean:** -0.112269  (inside [-10.0, 10.0])

**Held-out test metrics:**
  - R2: 0.869146
  - RMSE: 130606
  - MAE: 74634.1
  - PRD: 1.04195
  - PRB: -0.0476438
  - VEI: -13.1978
  - COD: 23.2873

**Hyperparameters (`model_config_json`):**

```
  - lgbm_base_config_id: 7ce5a0a0e6e4f38b
  - ratio_mode: diff
  - rho: 24.420530945486497
  - (LightGBM base hyperparameters from model_params.yaml::LGBMRegressor)
      · boosting_type: gbdt
      · class_weight: None
      · colsample_bytree: 1.0
      · importance_type: split
      · learning_rate: 0.012806
      · max_bin: 322
      · max_depth: 15
      · min_child_samples: 30
      · min_split_gain: 0.023317
      · n_estimators: 500
      · n_jobs: 1
      · num_leaves: 1209
      · objective: mse
      · random_state: 2025
      · reg_alpha: 1e-3
      · reg_lambda: 1e-2
      · subsample_for_bin: 200000
```
