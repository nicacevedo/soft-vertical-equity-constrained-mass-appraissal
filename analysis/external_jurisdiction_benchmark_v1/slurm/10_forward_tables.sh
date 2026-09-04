#!/bin/bash
# Frozen 2025 full modeling tables. Writes history_market_core_full.parquet only.
# Does not overwrite history_market_core_dev.parquet.
#SBATCH --job-name=v1_fwd_tables
#SBATCH --array=0-8
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=8:00:00
#SBATCH --output=analysis/external_jurisdiction_benchmark_v1/logs/fwd_tables_%A_%a.out
#SBATCH --error=analysis/external_jurisdiction_benchmark_v1/logs/fwd_tables_%A_%a.err
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal}"
PY="${PY:-/home/nacevedo/.conda/envs/fairness_env/bin/python}"
cd "${REPO_ROOT}"
KEYS=(wayne philadelphia st_louis_county allegheny maricopa king miami_dade middlesex cook)
KEY="${KEYS[${SLURM_ARRAY_TASK_ID:-0}]}"
CPUS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${CPUS}"
echo "forward_tables county=${KEY} cpus=${CPUS} host=$(hostname) partition=${SLURM_JOB_PARTITION:-} start=$(date -Is)"
"${PY}" analysis/external_jurisdiction_benchmark_v1/scripts/build_modeling_tables.py --county-key "${KEY}" --mode forward --end-date 2025-12-31
echo "done ${KEY} $(date -Is)"
