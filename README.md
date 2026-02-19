# Soft Vertical Equity Constrained Mass Appraisal

Minimal pipeline for:
- robust rolling-origin CV on time-split sales data
- mirrored test-set evaluation (single holdout)
- post-hoc analysis + plotting (validation avg + validation single-fold + test)
- optional convex stacking optimization with IAAO fairness constraints

## Installation

```bash
pip install -r requirements.txt
```

## Quick start (recommended for first-time users)

Run these from the `soft-vertical-equity-constrained-mass-appraissal/` directory.

Start with the **quick test** (4 core models, no CV). This is the easiest way to
verify your environment + data path and see the metric tables on:
- held-out test split (~2023)
- assessment split (2024)

```bash
python quick_test_models.py --rho 1.0 --out-dir ./output/quick_test
```

This writes:
- `output/quick_test/quick_test_metrics_test.csv`
- `output/quick_test/quick_test_metrics_assess.csv`

## Full pipeline workflow (CV → optional stacking → plots)

Run the full rolling-origin CV + test evaluation:

```bash
python run_temporal_cv.py
```

Then (optional) optimize stacking weights from CV artifacts:

```bash
python optimize_stacking_weights.py \
  --result-root ./output/robust_rolling_origin_cv \
  --data-id <DATA_ID> \
  --split-id <SPLIT_ID> \
  --accuracy-metric "OOS R2" \
  --objective-mode worst_fold
```

Finally generate analysis tables + plots (includes stacking overlays if present):

```bash
python analyze_results.py \
  --result-root ./output/robust_rolling_origin_cv \
  --data-id <DATA_ID> \
  --split-id <SPLIT_ID>
```

## What each script does

### `quick_test_models.py` (start here)

Runs 4 core models (2 baselines + 2 fairness-regularized with `rho`) and writes
two CSV tables with `R2`, `OOS R2`, PRD/PRB/VEI, etc.:
- test split (most recent pre-2024 sales)
- assessment split (2024)

It mirrors the preprocessing and split logic used in the full pipeline so it’s
a reliable “smoke test” before running CV.

### `run_temporal_cv.py`

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
7. Fits each model/config on train/validate and evaluates on the held-out test split, writing:
   - `analysis/.../test_metrics.csv`
   - `analysis/.../test_predictions.parquet` (per-row predictions; used for true stacking-on-test metrics)

When `run_temporal_cv.py` finishes it prints the `data_id` and `split_id` you should reuse.

### `optimize_stacking_weights.py`

Reads CV fold-level metrics and solves a convex weight optimization:
- decision variables: nonnegative weights \(w\) that sum to 1
- objective: maximize worst-fold or mean-fold accuracy (e.g. `OOS R2`)
- constraints: PRD/PRB/VEI inside IAAO ranges **on every fold**

Writes under:
`output/robust_rolling_origin_cv/analysis/data_id=.../split_id=.../stacking_pf_opt/`
- `weights.csv`
- `fold_ensemble_metrics.csv`
- `optimization_summary.json`

### `analyze_results.py`

Loads CV artifacts and produces:
- summary tables (`summary_by_config.*`, `pareto_front.*`, `shortlist.json`)
- plots under `analysis/.../plots/`:
  - `validation/`: cross-fold averages (one point per config)
  - `validation_single_run/`: a single chosen fold (hardest fold by mean accuracy)
  - `test/`: held-out test metrics (single run)

If `stacking_pf_opt/weights.csv` exists *and* `test_predictions.parquet` exists, the test plots also include:
- **stacking optimum (avg)**: computed from blended test predictions
- **stacking optimum (worst block)**: worst quarterly block on test (robustness analogue)

## Key settings to edit in `run_temporal_cv.py`

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
- `output/robust_rolling_origin_cv/analysis/` -> summaries, plots, test artifacts, and stacking outputs

This allows you to resume experiments without recomputing finished runs.

