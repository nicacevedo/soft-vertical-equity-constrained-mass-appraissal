#!/bin/bash
#SBATCH --job-name=v2_freeze
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --output=analysis/berry_attom_validation_v2/logs/freeze_%j.out
#SBATCH --error=analysis/berry_attom_validation_v2/logs/freeze_%j.err
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal}"
PY="${PY:-/home/nacevedo/.conda/envs/fairness_env/bin/python}"
cd "${REPO_ROOT}"
mkdir -p analysis/berry_attom_validation_v2/logs analysis/berry_attom_validation_v2/panel_freeze analysis/berry_attom_validation_v2/reports
export PYTHONUNBUFFERED=1
"${PY}" analysis/berry_attom_validation_v2/scripts/freeze_panel.py
"${PY}" analysis/berry_attom_validation_v2/scripts/write_final_report.py
# Direct/Surrogate are NOT submitted here. Inspect
# panel_freeze/final_panel_freeze_v2.yaml and only then sbatch 08_direct_surrogate.sh.
