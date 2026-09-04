#!/bin/bash
#SBATCH --job-name=v1_basecv2
#SBATCH --array=0-4
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --output=analysis/external_jurisdiction_benchmark_v1/logs/basecv2_%A_%a.out
#SBATCH --error=analysis/external_jurisdiction_benchmark_v1/logs/basecv2_%A_%a.err
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal}"
PY="${PY:-/home/nacevedo/.conda/envs/fairness_env/bin/python}"
cd "${REPO_ROOT}"
KEYS=(st_louis_county maricopa king miami_dade middlesex)
KEY="${KEYS[${SLURM_ARRAY_TASK_ID:-0}]}"
CPUS="${SLURM_CPUS_PER_TASK:-16}"
NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_cache_${SLURM_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "${NUMBA_CACHE_DIR}"
export NUMBA_CACHE_DIR
trap 'rm -rf "${NUMBA_CACHE_DIR}"' EXIT
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${CPUS}"
echo "basecv2 county=${KEY} cpus=${CPUS} host=$(hostname) partition=${SLURM_JOB_PARTITION:-} start=$(date -Is)"
"${PY}" analysis/external_jurisdiction_benchmark_v1/scripts/run_baseline_cv.py --county-key "${KEY}" --lgbm-threads "${CPUS}"
echo "done ${KEY} $(date -Is)"
