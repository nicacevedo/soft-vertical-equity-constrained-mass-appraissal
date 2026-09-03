#!/bin/bash
#SBATCH --job-name=v3_report
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --output=analysis/berry_attom_validation_v3/logs/report_%j.out
#SBATCH --error=analysis/berry_attom_validation_v3/logs/report_%j.err
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal}"
PY="${PY:-/home/nacevedo/.conda/envs/fairness_env/bin/python}"
cd "${REPO_ROOT}"
export PYTHONUNBUFFERED=1
"${PY}" analysis/berry_attom_validation_v3/scripts/write_final_report.py
echo "wrote FINAL_V3_REPORT.md $(date -Is)"
