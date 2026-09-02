#!/bin/bash
#SBATCH --job-name=v2_inventory
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=analysis/berry_attom_validation_v2/logs/inventory_%j.out
#SBATCH --error=analysis/berry_attom_validation_v2/logs/inventory_%j.err
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal}"
PY="${PY:-/home/nacevedo/.conda/envs/fairness_env/bin/python}"
cd "${REPO_ROOT}"
mkdir -p analysis/berry_attom_validation_v2/logs
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
"${PY}" analysis/berry_attom_validation_v2/scripts/inventory_new_dewey.py
