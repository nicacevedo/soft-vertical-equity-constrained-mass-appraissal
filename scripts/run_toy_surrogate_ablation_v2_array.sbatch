#!/bin/bash
# EXPERIMENTAL / TOY / NON-CANONICAL V2 — one family dense path + at most four training-s refinements.
# Array 0=quadratic, 1=huber, 2=absolute. Requires calibrate artifacts.
#SBATCH --job-name=toy_v2_fam
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/output/toy_surrogate_ablation_v2/logs/%x_%A_%a.out
#SBATCH --error=/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/output/toy_surrogate_ablation_v2/logs/%x_%A_%a.err

set -euo pipefail
cd /home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal
mkdir -p output/toy_surrogate_ablation_v2/logs output/toy_surrogate_ablation_v2/cache

FAMILIES=(quadratic huber absolute)
FAMILY="${FAMILIES[$SLURM_ARRAY_TASK_ID]}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-$PWD/output/toy_surrogate_ablation_v2/cache}/numba_${SLURM_JOB_ID:-$$}_${SLURM_ARRAY_TASK_ID:-}"
mkdir -p "$NUMBA_CACHE_DIR"
export LOKY_WORKER_TIMEOUT=14400

echo "TOY_V2_FAMILY family=$FAMILY task=$SLURM_ARRAY_TASK_ID partition=${SLURM_JOB_PARTITION:-} node=${SLURMD_NODENAME:-} cpus=${SLURM_CPUS_PER_TASK:-}"

/home/nacevedo/.conda/envs/fairness_env/bin/python scripts/toy_surrogate_ablation_v2.py \
  --mode fit-family \
  --family "$FAMILY" \
  --output-root ./output/toy_surrogate_ablation_v2

echo "TOY_V2_FAMILY_DONE family=$FAMILY"
