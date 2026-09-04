#!/bin/bash
# Full-panel launch: submitted only after pilot QA passes (Step 6), covering
# every remaining primary jurisdiction. KEYS is passed via --export=ALL,KEYS="..."
# (space-separated county keys) at submission time; array bounds must match.
#SBATCH --job-name=v1_direct_panel
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=16:00:00
#SBATCH --output=analysis/external_jurisdiction_benchmark_v1/logs/directpanel_%A_%a.out
#SBATCH --error=analysis/external_jurisdiction_benchmark_v1/logs/directpanel_%A_%a.err
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal}"
PY="${PY:-/home/nacevedo/.conda/envs/fairness_env/bin/python}"
cd "${REPO_ROOT}"
read -ra KEYS_ARR <<< "${KEYS:?KEYS env var (space-separated county keys) required}"
KEY="${KEYS_ARR[${SLURM_ARRAY_TASK_ID:-0}]}"
CPUS="${SLURM_CPUS_PER_TASK:-16}"
NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_cache_${SLURM_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "${NUMBA_CACHE_DIR}"
export NUMBA_CACHE_DIR
trap 'rm -rf "${NUMBA_CACHE_DIR}"' EXIT
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${CPUS}"
echo "direct_panel county=${KEY} cpus=${CPUS} host=$(hostname) partition=${SLURM_JOB_PARTITION:-} start=$(date -Is)"
"${PY}" analysis/external_jurisdiction_benchmark_v1/scripts/run_normalized_direct_cv.py --county-key "${KEY}" --lgbm-threads "${CPUS}"
echo "done ${KEY} $(date -Is)"
