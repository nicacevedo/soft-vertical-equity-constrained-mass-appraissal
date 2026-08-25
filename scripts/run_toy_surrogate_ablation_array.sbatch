#!/bin/bash
# EXPERIMENTAL / TOY / NON-CANONICAL — one surrogate variant, positive lambdas sequentially.
# Array task 0-5 maps onto the six formulations. Requires probe/calibrate artifacts.
#SBATCH --job-name=toy_surr_var
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --array=0-5
#SBATCH --output=/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/output/toy_surrogate_ablation/logs/%x_%A_%a.out
#SBATCH --error=/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/output/toy_surrogate_ablation/logs/%x_%A_%a.err

set -euo pipefail
cd /home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal
mkdir -p output/toy_surrogate_ablation/logs output/toy_surrogate_ablation/cache

VARIANTS=(quadratic_fixed quadratic_li absolute_fixed absolute_li capped_quadratic_fixed capped_quadratic_li)
VARIANT="${VARIANTS[$SLURM_ARRAY_TASK_ID]}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-$PWD/output/toy_surrogate_ablation/cache}/numba_${SLURM_JOB_ID:-$$}_${SLURM_ARRAY_TASK_ID:-}"
mkdir -p "$NUMBA_CACHE_DIR"
export LOKY_WORKER_TIMEOUT=14400

echo "TOY_VARIANT variant=$VARIANT task=$SLURM_ARRAY_TASK_ID partition=${SLURM_JOB_PARTITION:-} node=${SLURMD_NODENAME:-} cpus=${SLURM_CPUS_PER_TASK:-}"

/home/nacevedo/.conda/envs/fairness_env/bin/python scripts/toy_surrogate_ablation.py \
  --mode fit-variant \
  --variant "$VARIANT" \
  --output-root ./output/toy_surrogate_ablation

echo "TOY_VARIANT_DONE variant=$VARIANT"
