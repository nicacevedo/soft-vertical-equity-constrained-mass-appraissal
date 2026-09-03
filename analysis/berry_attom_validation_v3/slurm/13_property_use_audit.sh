#!/bin/bash
# Per-property-use-code structural profile and keep-rate-by-price-decile for all
# three counties. --audit-only never rewrites a modeling table: the freeze,
# held-out baselines and Direct path were computed against their recorded sha256.
#SBATCH --job-name=v3_useaudit
#SBATCH --array=0-2
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=2:00:00
#SBATCH --output=analysis/berry_attom_validation_v3/logs/useaudit_%A_%a.out
#SBATCH --error=analysis/berry_attom_validation_v3/logs/useaudit_%A_%a.err
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal}"
PY="${PY:-/home/nacevedo/.conda/envs/fairness_env/bin/python}"
cd "${REPO_ROOT}"
KEYS=(wayne philadelphia st_louis_county)
KEY="${KEYS[${SLURM_ARRAY_TASK_ID:-0}]}"
CPUS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${CPUS}"
echo "useaudit county=${KEY} cpus=${CPUS} host=$(hostname) partition=${SLURM_JOB_PARTITION:-} start=$(date -Is)"
"${PY}" analysis/berry_attom_validation_v3/scripts/build_modeling_tables.py \
  --county-key "${KEY}" --audit-only
echo "done ${KEY} $(date -Is)"
