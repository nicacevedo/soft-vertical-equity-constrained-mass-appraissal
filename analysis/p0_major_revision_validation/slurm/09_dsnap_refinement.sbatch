#!/bin/bash
#SBATCH --job-name=p0_dsnap_refine
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=110G
#SBATCH --array=0-44
#SBATCH --output=/orcd/home/002/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/analysis/p0_major_revision_validation/logs/09_refine_%A_%a.out
#SBATCH --error=/orcd/home/002/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/analysis/p0_major_revision_validation/logs/09_refine_%A_%a.err
set -euo pipefail
cd /orcd/home/002/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal
# Gate G5a targeted local refinement. HISTORICAL model settings; no Track-P pins.
# This is NOT a full 82-point regeneration; it fills only the skipped original-grid
# rho values inside the two affected regions.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
echo "host=$(hostname) job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} start=$(date -Is)"
/home/nacevedo/.conda/envs/fairness_env/bin/python \
  analysis/p0_major_revision_validation/code/p0_6_temporal_designs.py \
  --mode refine --shard "${SLURM_ARRAY_TASK_ID}"
echo "end=$(date -Is)"
