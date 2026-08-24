#!/bin/bash
#SBATCH --job-name=p6_pre_cv
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=36:00:00
#SBATCH --output=/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/output/paper_v6_preselection/logs/cv_%j.out
#SBATCH --error=/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/output/paper_v6_preselection/logs/cv_%j.err

set -euo pipefail
cd /home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal
mkdir -p output/paper_v6_preselection/logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export NUMBA_CACHE_DIR="/tmp/numba_cache_${SLURM_JOB_ID:-$$}"
mkdir -p "$NUMBA_CACHE_DIR"
# LightGBM C++ fits can exceed loky's default 300s worker heartbeat.
export LOKY_WORKER_TIMEOUT=14400

# 104 configs x 7 folds = 728 fits. Do not invoke model selection.
conda run -n fairness_env --no-capture-output python run_temporal_cv.py \
  --stage cv \
  --result-root ./output/paper_v6_preselection \
  --lgbm-config-json ./output/paper_v6_preselection/section2_lgbm_config.json \
  --no-include-cvar-models \
  --parallel \
  --parallel-cpu-fraction 0.90 \
  --parallel-max-workers 16 \
  --n-bootstrap 0
