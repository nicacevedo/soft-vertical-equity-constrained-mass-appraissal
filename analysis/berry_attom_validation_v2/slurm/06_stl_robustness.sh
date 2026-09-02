#!/bin/bash
#SBATCH --job-name=v2_stl_rob
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=analysis/berry_attom_validation_v2/logs/stl_rob_%j.out
#SBATCH --error=analysis/berry_attom_validation_v2/logs/stl_rob_%j.err
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal}"
PY="${PY:-/home/nacevedo/.conda/envs/fairness_env/bin/python}"
cd "${REPO_ROOT}"
mkdir -p analysis/berry_attom_validation_v2/logs
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
"${PY}" analysis/berry_attom_validation_v2/scripts/stl_source_robustness.py
