#!/bin/bash
#SBATCH --job-name=v3_final
#SBATCH --array=0-2
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=8:00:00
#SBATCH --output=analysis/berry_attom_validation_v3/logs/final_%A_%a.out
#SBATCH --error=analysis/berry_attom_validation_v3/logs/final_%A_%a.err
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal}"
PY="${PY:-/home/nacevedo/.conda/envs/fairness_env/bin/python}"
cd "${REPO_ROOT}"
mkdir -p analysis/berry_attom_validation_v3/logs analysis/berry_attom_validation_v3/final_baselines output/berry_attom_validation_v3/final_models
KEYS=(wayne philadelphia st_louis_county)
KEY="${KEYS[${SLURM_ARRAY_TASK_ID:-0}]}"
CPUS="${SLURM_CPUS_PER_TASK:-16}"
# dcor/numba JIT cache on node-local disk. Job 21882241_0 (Wayne surrogate)
# died with OSError errno 116 "Stale file handle" reading the numba index off
# the shared filesystem, which cancelled the downstream afterok chain.
NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_cache_${SLURM_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "${NUMBA_CACHE_DIR}"
export NUMBA_CACHE_DIR
trap 'rm -rf "${NUMBA_CACHE_DIR}"' EXIT
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${CPUS}"
echo "final baselines county=${KEY} cpus=${CPUS} host=$(hostname) partition=${SLURM_JOB_PARTITION:-} start=$(date -Is)"
"${PY}" analysis/berry_attom_validation_v3/scripts/run_final_baselines.py --county-key "${KEY}" --lgbm-threads "${CPUS}"
echo "done ${KEY} $(date -Is)"
