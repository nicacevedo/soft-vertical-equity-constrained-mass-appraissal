#!/bin/bash
# Frozen 2025 path: one task per (jurisdiction, family). Loops rho internally.
#SBATCH --job-name=v1_fwd_path
#SBATCH --array=0-17
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=16:00:00
#SBATCH --output=analysis/external_jurisdiction_benchmark_v1/logs/fwd_path_%A_%a.out
#SBATCH --error=analysis/external_jurisdiction_benchmark_v1/logs/fwd_path_%A_%a.err
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal}"
PY="${PY:-/home/nacevedo/.conda/envs/fairness_env/bin/python}"
cd "${REPO_ROOT}"
PAIRS=(
  wayne:direct wayne:surrogate
  philadelphia:direct philadelphia:surrogate
  st_louis_county:direct st_louis_county:surrogate
  allegheny:direct allegheny:surrogate
  maricopa:direct maricopa:surrogate
  king:direct king:surrogate
  miami_dade:direct miami_dade:surrogate
  middlesex:direct middlesex:surrogate
  cook:direct cook:surrogate
)
PAIR="${PAIRS[${SLURM_ARRAY_TASK_ID:-0}]}"
KEY="${PAIR%%:*}"
FAMILY="${PAIR##*:}"
CPUS="${SLURM_CPUS_PER_TASK:-16}"
NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_cache_${SLURM_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "${NUMBA_CACHE_DIR}"
export NUMBA_CACHE_DIR
trap 'rm -rf "${NUMBA_CACHE_DIR}"' EXIT
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${CPUS}"
echo "forward_path county=${KEY} family=${FAMILY} cpus=${CPUS} host=$(hostname) partition=${SLURM_JOB_PARTITION:-} start=$(date -Is)"
"${PY}" analysis/external_jurisdiction_benchmark_v1/scripts/run_forward_path.py --county-key "${KEY}" --family "${FAMILY}" --lgbm-threads "${CPUS}"
echo "done ${KEY} ${FAMILY} $(date -Is)"
