#!/bin/bash
#SBATCH --job-name=p0_race_t12
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --exclude=node2621
#SBATCH --array=0-10
#SBATCH --output=/orcd/home/002/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/analysis/p0_major_revision_validation/race/task12_parallel/logs/race_%A_%a.out
#SBATCH --error=/orcd/home/002/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/analysis/p0_major_revision_validation/race/task12_parallel/logs/race_%A_%a.err
set -euo pipefail
cd /orcd/home/002/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal
# Redundant parallel re-execution of refinement array task 12 (region A / fold_7 / Direct /
# chunk 1-of-2). ONE rho per array element. HISTORICAL model settings; no Track-P pins.
# This is NOT a new experiment and NOT part of the 82-point regeneration.
#
# Identical to slurm/09_dsnap_refinement.sbatch except, all scheduler-only:
#   --mem      110G -> 16G   (measured MaxRSS of every completed refinement task: 2.4-2.7 GB;
#                             110G blocked concurrency on a near-saturated partition)
#   --exclude  node2621      (the oversubscribed node the canonical task is stuck on)
#   --array    0-10          (one rho per element instead of 11 serial rho in one task)
# Partition, wall clock, cpus-per-task, environment and ALL thread settings are unchanged.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
# Cache LOCATION only -- avoids the stale-NFS numba cache read that killed 22151066_1.
# Numerics, the dCor estimator and every metric implementation are untouched.
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_cache_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$NUMBA_CACHE_DIR"
echo "host=$(hostname) job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} start=$(date -Is)"
echo "numba_cache=$NUMBA_CACHE_DIR"
/home/nacevedo/.conda/envs/fairness_env/bin/python \
  analysis/p0_major_revision_validation/code/p0_6_race_task12.py \
  --rho-index "${SLURM_ARRAY_TASK_ID}"
echo "end=$(date -Is)"
