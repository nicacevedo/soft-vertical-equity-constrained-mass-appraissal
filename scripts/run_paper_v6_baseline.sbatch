#!/bin/bash
#SBATCH --job-name=paper_v6_baseline
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/output/robust_rolling_origin_cv_v2/logs/baseline_%j.out
#SBATCH --error=/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/output/robust_rolling_origin_cv_v2/logs/baseline_%j.err

set -euo pipefail
cd /home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal
mkdir -p output/robust_rolling_origin_cv_v2/logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

# CPU LightGBM only. Do not request GPUs.
# Stage is baseline-search only; penalty CV/test/forward are not started here.
conda run -n fairness_env --no-capture-output python run_temporal_cv.py \
  --stage baseline-search \
  --result-root ./output/robust_rolling_origin_cv_v2 \
  --parallel \
  --parallel-cpu-fraction 0.90 \
  --parallel-max-workers 16 \
  --no-include-cvar-models
