#!/bin/bash
#SBATCH --job-name=v3_freeze
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --output=analysis/berry_attom_validation_v3/logs/freeze_%j.out
#SBATCH --error=analysis/berry_attom_validation_v3/logs/freeze_%j.err
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal}"
PY="${PY:-/home/nacevedo/.conda/envs/fairness_env/bin/python}"
cd "${REPO_ROOT}"
mkdir -p analysis/berry_attom_validation_v3/logs analysis/berry_attom_validation_v3/panel_freeze analysis/berry_attom_validation_v3/reports
export PYTHONUNBUFFERED=1
echo "freeze host=$(hostname) partition=${SLURM_JOB_PARTITION:-} start=$(date -Is)"
"${PY}" analysis/berry_attom_validation_v3/scripts/freeze_panel.py
"${PY}" analysis/berry_attom_validation_v3/scripts/write_final_report.py
echo "Direct/Surrogate are NOT submitted from this job. Dispatch inspects the freeze file."
echo "done $(date -Is)"
