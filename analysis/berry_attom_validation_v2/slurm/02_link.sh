#!/bin/bash
#SBATCH --job-name=v2_link
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=6:00:00
#SBATCH --output=analysis/berry_attom_validation_v2/logs/link_%j.out
#SBATCH --error=analysis/berry_attom_validation_v2/logs/link_%j.err
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal}"
PY="${PY:-/home/nacevedo/.conda/envs/fairness_env/bin/python}"
cd "${REPO_ROOT}"
mkdir -p analysis/berry_attom_validation_v2/logs analysis/berry_attom_validation_v2/linkage analysis/berry_attom_validation_v2/figures
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
"${PY}" analysis/berry_attom_validation_v2/scripts/link_berry_attom.py
