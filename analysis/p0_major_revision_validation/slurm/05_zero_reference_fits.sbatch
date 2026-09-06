#!/bin/bash
#SBATCH --job-name=p0_zref
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=110G
#SBATCH --array=0-8
#SBATCH --output=/orcd/home/002/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/analysis/p0_major_revision_validation/logs/05_zero_reference_fits_%A_%a.out
#SBATCH --error=/orcd/home/002/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/analysis/p0_major_revision_validation/logs/05_zero_reference_fits_%A_%a.err
set -euo pipefail
cd /orcd/home/002/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal
# HISTORICAL canonical execution settings. No Track-P deterministic pins anywhere.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
echo "host=$(hostname) job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} block=${SLURM_ARRAY_TASK_ID} start=$(date -Is)"
/home/nacevedo/.conda/envs/fairness_env/bin/python \
  analysis/p0_major_revision_validation/code/p0_zero_reference_fits.py \
  --block-index "${SLURM_ARRAY_TASK_ID}" --cells A,B,C
echo "end=$(date -Is)"
