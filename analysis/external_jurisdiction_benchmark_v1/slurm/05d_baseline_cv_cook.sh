#!/bin/bash
#SBATCH --job-name=v1_basecv_cook
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --output=analysis/external_jurisdiction_benchmark_v1/logs/basecv_cook_%j.out
#SBATCH --error=analysis/external_jurisdiction_benchmark_v1/logs/basecv_cook_%j.err
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal}"
PY="${PY:-/home/nacevedo/.conda/envs/fairness_env/bin/python}"
cd "${REPO_ROOT}"
CPUS="${SLURM_CPUS_PER_TASK:-16}"
NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_cache_${SLURM_JOB_ID:-0}"
mkdir -p "${NUMBA_CACHE_DIR}"
export NUMBA_CACHE_DIR
trap 'rm -rf "${NUMBA_CACHE_DIR}"' EXIT
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${CPUS}"
echo "basecv_cook cpus=${CPUS} host=$(hostname) partition=${SLURM_JOB_PARTITION:-} start=$(date -Is)"
"${PY}" analysis/external_jurisdiction_benchmark_v1/scripts/run_baseline_cv.py --county-key cook --lgbm-threads "${CPUS}"
echo "done $(date -Is)"
