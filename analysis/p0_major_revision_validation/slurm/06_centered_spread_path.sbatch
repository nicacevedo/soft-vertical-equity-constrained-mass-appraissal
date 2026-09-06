#!/bin/bash
#SBATCH --job-name=p0_cspath
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=90G
#SBATCH --array=0-19
#SBATCH --output=/orcd/home/002/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/analysis/p0_major_revision_validation/logs/06_centered_spread_path_%A_%a.out
#SBATCH --error=/orcd/home/002/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/analysis/p0_major_revision_validation/logs/06_centered_spread_path_%A_%a.err
set -euo pipefail
cd /orcd/home/002/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
echo "host=$(hostname) job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} shard=${SLURM_ARRAY_TASK_ID} start=$(date -Is)"
/home/nacevedo/.conda/envs/fairness_env/bin/python \
  analysis/p0_major_revision_validation/code/p0_4_centered_spread.py \
  --mode path --shard "${SLURM_ARRAY_TASK_ID}"
echo "end=$(date -Is)"
