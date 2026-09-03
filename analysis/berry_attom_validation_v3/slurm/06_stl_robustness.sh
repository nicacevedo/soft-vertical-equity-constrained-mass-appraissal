#!/bin/bash
#SBATCH --job-name=v3_stl_rob
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=analysis/berry_attom_validation_v3/logs/stl_rob_%j.out
#SBATCH --error=analysis/berry_attom_validation_v3/logs/stl_rob_%j.err
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal}"
PY="${PY:-/home/nacevedo/.conda/envs/fairness_env/bin/python}"
cd "${REPO_ROOT}"
mkdir -p analysis/berry_attom_validation_v3/logs output/berry_attom_validation_v3/common_cohorts
CPUS="${SLURM_CPUS_PER_TASK:-16}"
# dcor/numba JIT cache on node-local disk. Job 21882241_0 (Wayne surrogate)
# died with OSError errno 116 "Stale file handle" reading the numba index off
# the shared filesystem, which cancelled the downstream afterok chain.
NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_cache_${SLURM_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "${NUMBA_CACHE_DIR}"
export NUMBA_CACHE_DIR
trap 'rm -rf "${NUMBA_CACHE_DIR}"' EXIT
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${CPUS}"
echo "stl_rob host=$(hostname) partition=${SLURM_JOB_PARTITION:-} start=$(date -Is)"
"${PY}" analysis/berry_attom_validation_v3/scripts/stl_source_robustness.py
echo "done $(date -Is)"
