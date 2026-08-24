#!/bin/bash
#SBATCH --job-name=p6_pre_oos
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/output/paper_v6_preselection/logs/oos_%j.out
#SBATCH --error=/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/output/paper_v6_preselection/logs/oos_%j.err

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
export LOKY_WORKER_TIMEOUT=14400

# Held-out and 2025 paths. Do not invoke model selection. Requires complete CV.
conda run -n fairness_env --no-capture-output python run_temporal_cv.py \
  --stage test \
  --result-root ./output/paper_v6_preselection \
  --lgbm-config-json ./output/paper_v6_preselection/section2_lgbm_config.json \
  --no-include-cvar-models \
  --parallel \
  --parallel-cpu-fraction 0.90 \
  --parallel-max-workers 16 \
  --n-bootstrap 0

conda run -n fairness_env --no-capture-output python run_temporal_cv.py \
  --stage forward \
  --result-root ./output/paper_v6_preselection \
  --lgbm-config-json ./output/paper_v6_preselection/section2_lgbm_config.json \
  --no-include-cvar-models \
  --parallel \
  --parallel-cpu-fraction 0.90 \
  --parallel-max-workers 16 \
  --n-bootstrap 0
