#!/bin/bash
# EXPERIMENTAL / TOY follow-up V3 QD anchor job.
# Array 0 -> anchor 0.30; array 1 -> anchor 0.25.
#SBATCH --job-name=toy_fu_qd
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --array=0-1
#SBATCH --output=/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/output/toy_surrogate_followup_v3/logs/%x_%A_%a.out
#SBATCH --error=/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/output/toy_surrogate_followup_v3/logs/%x_%A_%a.err

set -euo pipefail
cd /home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal
mkdir -p output/toy_surrogate_followup_v3/logs

ANCHORS=(0.30 0.25)
ANCHOR_S=${ANCHORS[${SLURM_ARRAY_TASK_ID}]}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export NUMBA_CACHE_DIR="${HOME}/.cache/numba"
mkdir -p "$NUMBA_CACHE_DIR"

echo "TOY_FOLLOWUP_QD anchor=${ANCHOR_S} partition=${SLURM_JOB_PARTITION:-} node=${SLURMD_NODENAME:-} job=${SLURM_JOB_ID:-}"

/home/nacevedo/.conda/envs/fairness_env/bin/python scripts/toy_followup_v3.py \
  --mode fit-qd-anchor \
  --anchor-s "${ANCHOR_S}" \
  --output-root ./output/toy_surrogate_followup_v3

echo "TOY_FOLLOWUP_QD_DONE anchor=${ANCHOR_S}"
