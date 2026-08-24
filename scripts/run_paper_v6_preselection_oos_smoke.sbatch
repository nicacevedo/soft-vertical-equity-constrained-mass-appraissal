#!/bin/bash
#SBATCH --job-name=p6_oos_smoke
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/output/paper_v6_preselection/logs/oos_smoke_%j.out
#SBATCH --error=/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/output/paper_v6_preselection/logs/oos_smoke_%j.err

set -euo pipefail
cd /home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal
ROOT=output/paper_v6_preselection/smoke/oos_heldout
mkdir -p "$ROOT" output/paper_v6_preselection/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-$PWD/output/paper_v6_preselection/cache}/numba_${SLURM_JOB_ID}"
mkdir -p "$NUMBA_CACHE_DIR"
export LOKY_WORKER_TIMEOUT=14400

echo "SMOKE start job=$SLURM_JOB_ID node=${SLURMD_NODENAME:-}"
conda run -n fairness_env --no-capture-output python -c "import dcor, lightgbm; print('dcor', dcor.__version__, 'lgbm', lightgbm.__version__)"

run_once() {
  conda run -n fairness_env --no-capture-output python run_temporal_cv.py \
    --stage test \
    --result-root "$ROOT" \
    --lgbm-config-json ./output/paper_v6_preselection/section2_lgbm_config.json \
    --rho-values 0.1,1,10,100 \
    --rho-count 4 \
    --rho-scale linear \
    --no-include-cvar-models \
    --sample-frac 0.05 \
    --allow-incomplete-cv \
    --no-parallel \
    --n-bootstrap 0
}

run_once
echo "SMOKE first pass done; rerunning for resume"
run_once

conda run -n fairness_env --no-capture-output python scripts/qa_paper_v6_oos_smoke.py \
  --result-root "$ROOT" \
  --section2-config ./output/paper_v6_preselection/section2_lgbm_config.json
echo "SMOKE_PASS"
