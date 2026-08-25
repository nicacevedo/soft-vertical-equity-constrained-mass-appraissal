#!/bin/bash
# EXPERIMENTAL / TOY smoke: 3% sample, 30 trees, development->held-out only.
# Not a scientific output. Verifies all six objectives execute. No CV.
#SBATCH --job-name=toy_surr_smoke
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/output/toy_surrogate_ablation/logs/%x_%j.out
#SBATCH --error=/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/output/toy_surrogate_ablation/logs/%x_%j.err

set -euo pipefail
cd /home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal
mkdir -p output/toy_surrogate_ablation/logs output/toy_surrogate_ablation/cache

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-$PWD/output/toy_surrogate_ablation/cache}/numba_smoke_${SLURM_JOB_ID:-$$}"
mkdir -p "$NUMBA_CACHE_DIR"

echo "TOY_SMOKE partition=${SLURM_JOB_PARTITION:-} node=${SLURMD_NODENAME:-} cpus=${SLURM_CPUS_PER_TASK:-}"

/home/nacevedo/.conda/envs/fairness_env/bin/python scripts/toy_surrogate_ablation.py \
  --mode smoke \
  --output-root ./output/toy_surrogate_ablation

echo "TOY_SMOKE_DONE"
