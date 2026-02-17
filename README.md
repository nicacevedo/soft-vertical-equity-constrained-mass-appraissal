# Soft Vertical Equity Constrained Mass Appraisal

Minimal experiment runner for comparing baseline and fairness-constrained models
with robust rolling-origin validation.

## Installation

```bash
pip install -r requirements.txt
```

## Quick start

```bash
python main.py
```

## What `main.py` does (high level)

1. Loads `training_data.parquet` and keeps only selected predictors + target/date.
2. Optionally applies random sampling (`sample_frac`, `sample_seed`) for smaller experiments.
3. Splits data by time into:
   - assessment block (future period),
   - out-of-time test block (most recent 10%),
   - train/validate universe (remaining older sales).
4. Builds model configs (Linear Regression, baseline LightGBM, and constrained variants).
5. Runs robust rolling-origin CV on the train/validate universe:
   - expanding training window,
   - fixed-size validation block (`val_fraction=0.10`) immediately after training,
   - block bootstrap over validation folds for uncertainty estimates.
6. Stores fold-level metrics and artifacts to disk with deterministic IDs.

## Key settings to edit in `main.py`

- `sample_frac` and `sample_seed`: optional dataset down-sampling with reproducibility.
- `split_protocol`: rolling-origin settings (`initial_train_months`, `val_fraction`, minimum rows).
- `bootstrap_protocol`: bootstrap frequency and number of resamples.
- `model_specs`: list of models and hyperparameter grids to evaluate.

## Outputs

By default, outputs are written under:

- `output/robust_rolling_origin_cv/runs/` -> fold/config point metrics
- `output/robust_rolling_origin_cv/bootstrap_metrics/` -> bootstrap metrics
- `output/robust_rolling_origin_cv/predictions/` -> validation predictions
- `output/robust_rolling_origin_cv/run_status/` -> started/completed/failed status per run
- `output/robust_rolling_origin_cv/protocol/` -> saved fold and bootstrap definitions

This allows you to resume experiments without recomputing finished runs.

