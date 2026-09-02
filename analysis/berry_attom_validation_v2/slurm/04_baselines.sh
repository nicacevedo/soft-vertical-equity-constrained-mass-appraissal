#!/bin/bash
#SBATCH --job-name=v2_base
#SBATCH --array=0-2
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=8:00:00
#SBATCH --output=analysis/berry_attom_validation_v2/logs/baseline_%A_%a.out
#SBATCH --error=analysis/berry_attom_validation_v2/logs/baseline_%A_%a.err
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal}"
PY="${PY:-/home/nacevedo/.conda/envs/fairness_env/bin/python}"
cd "${REPO_ROOT}"
mkdir -p analysis/berry_attom_validation_v2/logs analysis/berry_attom_validation_v2/baselines output/berry_attom_validation_v2/baselines
KEYS=(wayne philadelphia st_louis_county)
KEY="${KEYS[${SLURM_ARRAY_TASK_ID:-0}]}"
CPUS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${CPUS}"
echo "baseline county=${KEY} cpus=${CPUS} host=$(hostname) start=$(date -Is)"
"${PY}" analysis/berry_attom_validation_v2/scripts/run_baselines.py --county-key "${KEY}" --lgbm-threads "${CPUS}"
echo "done ${KEY} $(date -Is)"
