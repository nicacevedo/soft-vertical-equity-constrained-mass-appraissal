#!/bin/bash
#SBATCH --job-name=v3_cache
#SBATCH --array=0-2
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --output=analysis/berry_attom_validation_v3/logs/cache_%A_%a.out
#SBATCH --error=analysis/berry_attom_validation_v3/logs/cache_%A_%a.err
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal}"
PY="${PY:-/home/nacevedo/.conda/envs/fairness_env/bin/python}"
cd "${REPO_ROOT}"
mkdir -p analysis/berry_attom_validation_v3/logs output/berry_attom_validation_v3/cache
KEYS=(wayne philadelphia st_louis_county)
KEY="${KEYS[${SLURM_ARRAY_TASK_ID:-0}]}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
echo "cache county=${KEY} host=$(hostname) partition=${SLURM_JOB_PARTITION:-} start=$(date -Is)"
"${PY}" analysis/berry_attom_validation_v3/scripts/build_county_caches.py --county-key "${KEY}"
echo "done ${KEY} $(date -Is)"
