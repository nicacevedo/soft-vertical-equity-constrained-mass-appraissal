#!/bin/bash
# EXPERIMENTAL / TOY follow-up V3 precheck. Isolated output root.
#SBATCH --job-name=toy_fu_pre
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:50:00
#SBATCH --output=/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/output/toy_surrogate_followup_v3/logs/%x_%j.out
#SBATCH --error=/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/output/toy_surrogate_followup_v3/logs/%x_%j.err

set -euo pipefail
cd /home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal
mkdir -p output/toy_surrogate_followup_v3/logs output/toy_surrogate_followup_v3/manifests

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export NUMBA_CACHE_DIR="${HOME}/.cache/numba"
mkdir -p "$NUMBA_CACHE_DIR"

echo "TOY_FOLLOWUP_PRECHECK partition=${SLURM_JOB_PARTITION:-} node=${SLURMD_NODENAME:-} cpus=${SLURM_CPUS_PER_TASK:-} job=${SLURM_JOB_ID:-}"

/home/nacevedo/.conda/envs/fairness_env/bin/python scripts/toy_followup_v3.py \
  --mode preflight \
  --output-root ./output/toy_surrogate_followup_v3

echo "TOY_FOLLOWUP_PRECHECK_DONE"
