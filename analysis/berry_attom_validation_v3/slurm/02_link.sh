#!/bin/bash
#SBATCH --job-name=v3_link
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=8:00:00
#SBATCH --output=analysis/berry_attom_validation_v3/logs/link_%j.out
#SBATCH --error=analysis/berry_attom_validation_v3/logs/link_%j.err
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal}"
PY="${PY:-/home/nacevedo/.conda/envs/fairness_env/bin/python}"
cd "${REPO_ROOT}"
mkdir -p analysis/berry_attom_validation_v3/logs analysis/berry_attom_validation_v3/linkage analysis/berry_attom_validation_v3/figures
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
echo "link host=$(hostname) partition=${SLURM_JOB_PARTITION:-} start=$(date -Is)"
"${PY}" analysis/berry_attom_validation_v3/scripts/link_berry_attom.py
echo "done $(date -Is)"
