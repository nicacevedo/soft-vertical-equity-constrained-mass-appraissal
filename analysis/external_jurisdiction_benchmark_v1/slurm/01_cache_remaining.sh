#!/bin/bash
#SBATCH --job-name=v1_cache6
#SBATCH --array=0-5
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=3:00:00
#SBATCH --output=analysis/external_jurisdiction_benchmark_v1/logs/cache_%A_%a.out
#SBATCH --error=analysis/external_jurisdiction_benchmark_v1/logs/cache_%A_%a.err
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal}"
PY="${PY:-/home/nacevedo/.conda/envs/fairness_env/bin/python}"
cd "${REPO_ROOT}"
KEYS=(st_louis_county allegheny maricopa king miami_dade middlesex)
KEY="${KEYS[${SLURM_ARRAY_TASK_ID:-0}]}"
CPUS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${CPUS}"
echo "cache county=${KEY} cpus=${CPUS} host=$(hostname) partition=${SLURM_JOB_PARTITION:-} start=$(date -Is)"
"${PY}" analysis/external_jurisdiction_benchmark_v1/scripts/build_county_caches.py --county-key "${KEY}"
echo "done ${KEY} $(date -Is)"
