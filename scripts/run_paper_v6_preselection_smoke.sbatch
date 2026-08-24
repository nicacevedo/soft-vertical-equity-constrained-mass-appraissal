#!/bin/bash
#SBATCH --job-name=p6_pre_smoke
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --output=/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/output/paper_v6_preselection/logs/smoke_%j.out
#SBATCH --error=/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/output/paper_v6_preselection/logs/smoke_%j.err

set -euo pipefail
cd /home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal
mkdir -p output/paper_v6_preselection/logs output/paper_v6_preselection/smoke

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export NUMBA_CACHE_DIR="/tmp/numba_cache_${SLURM_JOB_ID:-$$}"
mkdir -p "$NUMBA_CACHE_DIR"

conda run -n fairness_env --no-capture-output python -c "import dcor; print('dcor', dcor.__version__)"

conda run -n fairness_env --no-capture-output python run_temporal_cv.py \
  --stage cv \
  --result-root ./output/paper_v6_preselection/smoke \
  --lgbm-config-json ./output/paper_v6_preselection/section2_lgbm_config.json \
  --rho-values 0.1,1,10,100 \
  --rho-count 4 \
  --rho-scale linear \
  --no-include-cvar-models \
  --sample-frac 0.05 \
  --max-folds 2 \
  --allow-incomplete-cv \
  --parallel \
  --parallel-cpu-fraction 0.90 \
  --parallel-max-workers 8 \
  --n-bootstrap 0

conda run -n fairness_env --no-capture-output python scripts/qa_preselection_smoke.py \
  --result-root ./output/paper_v6_preselection/smoke \
  --section2-config ./output/paper_v6_preselection/section2_lgbm_config.json
