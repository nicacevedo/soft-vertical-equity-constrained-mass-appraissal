#!/bin/bash
#SBATCH --job-name=mv_spatial_smoke
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=0:45:00
#SBATCH --output=temp/logs/mv_spatial_smoke_%j.out
#SBATCH --error=temp/logs/mv_spatial_smoke_%j.err

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal}"
PY="${PY:-/home/nacevedo/.conda/envs/fairness_env/bin/python}"
cd "${REPO_ROOT}"
mkdir -p temp/logs output/smoke_mv_spatial_slurm

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYARROW_NUM_THREADS=4
export PYTHONUNBUFFERED=1

"${PY}" final_market_value_spatial_experiments.py \
  --data-path data/CCAO/2025/training_data.parquet \
  --params-path params.yaml \
  --out-dir "output/smoke_mv_spatial_slurm/${SLURM_JOB_ID}" \
  --sample-frac 0.05 \
  --sample-seed 42 \
  --target-variants raw,global_slope,hedonic_area_slope:meta_township_code \
  --learners ridge,lgbm_l2 \
  --spatial-feature-types scaled_time_adjusted_price,residual,time_adjusted_price \
  --Ks 10,20 \
  --spatial-bandwidths-miles 0.75,1.0 \
  --time-bandwidths-days 500,730 \
  --calibration-modes none,median_center \
  --calib-frac 0.15 \
  --n-jobs 12 \
  --n-oof-folds 3 \
  --n-estimators 200 \
  --learning-rate 0.06

echo "[mv_spatial_smoke] done job=${SLURM_JOB_ID}"
