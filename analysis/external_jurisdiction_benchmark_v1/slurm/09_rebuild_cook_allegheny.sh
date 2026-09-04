#!/bin/bash
# Rebuild ONLY Cook and Allegheny: cache -> modeling table, using the
# resolved canonical (broad) History source. Does not touch Wayne,
# Philadelphia, or any other jurisdiction's cache/table/baseline.
#SBATCH --job-name=v1_rebuild_ca
#SBATCH --array=0-1
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=4:00:00
#SBATCH --output=analysis/external_jurisdiction_benchmark_v1/logs/rebuildca_%A_%a.out
#SBATCH --error=analysis/external_jurisdiction_benchmark_v1/logs/rebuildca_%A_%a.err
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal}"
PY="${PY:-/home/nacevedo/.conda/envs/fairness_env/bin/python}"
cd "${REPO_ROOT}"
KEYS=(cook allegheny)
KEY="${KEYS[${SLURM_ARRAY_TASK_ID:-0}]}"
CPUS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${CPUS}"
echo "rebuild county=${KEY} cpus=${CPUS} host=$(hostname) partition=${SLURM_JOB_PARTITION:-} start=$(date -Is)"
"${PY}" analysis/external_jurisdiction_benchmark_v1/scripts/build_county_caches.py --county-key "${KEY}"
"${PY}" analysis/external_jurisdiction_benchmark_v1/scripts/build_modeling_tables.py --county-key "${KEY}" --end-date 2024-12-31
echo "done ${KEY} $(date -Is)"
