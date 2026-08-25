#!/bin/bash
# EXPERIMENTAL / TOY / NON-CANONICAL — one hybrid target (QD or QNL).
# Requires HYBRID_METHOD in the environment. Array task 0/1/2 -> s=0.20/0.15/0.10.
#SBATCH --job-name=toy_hyb_tgt
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=03:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/output/toy_surrogate_ablation_v2/logs/%x_%A_%a.out
#SBATCH --error=/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/output/toy_surrogate_ablation_v2/logs/%x_%A_%a.err

set -euo pipefail
cd /home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal
mkdir -p output/toy_surrogate_ablation_v2/logs output/toy_surrogate_ablation_v2/cache

if [[ -z "${HYBRID_METHOD:-}" ]]; then
  echo "HYBRID_METHOD is required" >&2
  exit 1
fi

S_LIST=(0.20 0.15 0.10)
TARGET_S=${S_LIST[${SLURM_ARRAY_TASK_ID}]}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export NUMBA_CACHE_DIR="${HOME}/.cache/numba"
mkdir -p "$NUMBA_CACHE_DIR"

echo "TOY_HYBRID_TARGET method=${HYBRID_METHOD} target_s=${TARGET_S} partition=${SLURM_JOB_PARTITION:-} node=${SLURMD_NODENAME:-} cpus=${SLURM_CPUS_PER_TASK:-} job=${SLURM_JOB_ID:-} array=${SLURM_ARRAY_JOB_ID:-}:${SLURM_ARRAY_TASK_ID:-}"

/home/nacevedo/.conda/envs/fairness_env/bin/python scripts/toy_hybrid_selection.py \
  --mode fit-target \
  --method "${HYBRID_METHOD}" \
  --target-s "${TARGET_S}" \
  --output-root ./output/toy_surrogate_ablation_v2

echo "TOY_HYBRID_TARGET_DONE method=${HYBRID_METHOD} target_s=${TARGET_S}"
