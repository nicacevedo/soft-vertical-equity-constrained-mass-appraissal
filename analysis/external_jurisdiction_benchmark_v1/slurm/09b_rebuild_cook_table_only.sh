#!/bin/bash
# Cook's broad History cache (32.7M rows) already built successfully in
# 21935442_0; only the modeling-table step OOM'd at 100G. Rerun that step
# alone with more memory. Does not rebuild the cache (already correct, hash
# recorded) and does not touch any other jurisdiction.
#SBATCH --job-name=v1_cooktable
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=4:00:00
#SBATCH --output=analysis/external_jurisdiction_benchmark_v1/logs/cooktable_%j.out
#SBATCH --error=analysis/external_jurisdiction_benchmark_v1/logs/cooktable_%j.err
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal}"
PY="${PY:-/home/nacevedo/.conda/envs/fairness_env/bin/python}"
cd "${REPO_ROOT}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
echo "cooktable host=$(hostname) partition=${SLURM_JOB_PARTITION:-} start=$(date -Is)"
"${PY}" analysis/external_jurisdiction_benchmark_v1/scripts/build_modeling_tables.py --county-key cook --end-date 2024-12-31
echo "done $(date -Is)"
