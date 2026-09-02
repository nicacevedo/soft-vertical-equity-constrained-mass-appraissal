#!/bin/bash
#SBATCH --job-name=v2_stl_local
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=8:00:00
#SBATCH --output=analysis/berry_attom_validation_v2/logs/stl_local_%j.out
#SBATCH --error=analysis/berry_attom_validation_v2/logs/stl_local_%j.err
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal}"
PY="${PY:-/home/nacevedo/.conda/envs/fairness_env/bin/python}"
cd "${REPO_ROOT}"
mkdir -p analysis/berry_attom_validation_v2/logs analysis/berry_attom_validation_v2/st_louis_source_robustness output/berry_attom_validation_v2/st_louis_source_robustness
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
"${PY}" analysis/berry_attom_validation_v2/scripts/stl_local_avm.py
