#!/bin/bash
#SBATCH --job-name=v2_paths
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=90G
#SBATCH --time=12:00:00
#SBATCH --output=analysis/berry_attom_validation_v2/logs/paths_%j.out
#SBATCH --error=analysis/berry_attom_validation_v2/logs/paths_%j.err
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal}"
PY="${PY:-/home/nacevedo/.conda/envs/fairness_env/bin/python}"
cd "${REPO_ROOT}"
mkdir -p analysis/berry_attom_validation_v2/logs output/berry_attom_validation_v2/method_transfer
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
# Aborts immediately if the freeze file is missing or unauthorized.
"${PY}" analysis/berry_attom_validation_v2/scripts/run_direct_surrogate.py --lgbm-threads "${SLURM_CPUS_PER_TASK:-16}"
"${PY}" analysis/berry_attom_validation_v2/scripts/write_final_report.py
