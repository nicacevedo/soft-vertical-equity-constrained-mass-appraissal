#!/bin/bash
#SBATCH --job-name=p6_oos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --array=0-7
#SBATCH --output=/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/output/paper_v6_preselection/logs/%x_%A_%a.out
#SBATCH --error=/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/output/paper_v6_preselection/logs/%x_%A_%a.err

set -euo pipefail
cd /home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal

FAMILY="${P6_FAMILY:?}"
STAGE="${P6_STAGE:?}"
EVAL="${P6_EVAL:?}"
N_CHUNKS="${P6_N_CHUNKS:-8}"
CHUNK="$SLURM_ARRAY_TASK_ID"
CHUNK_LAB=$(printf "%02d" "$CHUNK")
ROOT="output/paper_v6_preselection/reporting_preview/${EVAL}/${FAMILY}/chunk_${CHUNK_LAB}"
mkdir -p "$ROOT" output/paper_v6_preselection/logs output/paper_v6_preselection/cache

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-$PWD/output/paper_v6_preselection/cache}/numba_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$NUMBA_CACHE_DIR"
export LOKY_WORKER_TIMEOUT=14400

echo "OOS family=$FAMILY stage=$STAGE eval=$EVAL chunk=$CHUNK/$N_CHUNKS root=$ROOT node=${SLURMD_NODENAME:-}"

conda run -n fairness_env --no-capture-output python run_temporal_cv.py \
  --stage "$STAGE" \
  --result-root "$ROOT" \
  --lgbm-config-json ./output/paper_v6_preselection/section2_lgbm_config.json \
  --no-include-cvar-models \
  --allow-incomplete-cv \
  --only-model-names "$FAMILY" \
  --rho-chunk "${CHUNK}/${N_CHUNKS}" \
  --no-parallel \
  --n-bootstrap 0 \
  --skip-aggregate-write

echo "OOS_CHUNK_DONE family=$FAMILY eval=$EVAL chunk=$CHUNK"
