#!/bin/bash
#SBATCH --job-name=v3_stl_loc
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=8:00:00
#SBATCH --output=analysis/berry_attom_validation_v3/logs/stl_local_%j.out
#SBATCH --error=analysis/berry_attom_validation_v3/logs/stl_local_%j.err
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal}"
PY="${PY:-/home/nacevedo/.conda/envs/fairness_env/bin/python}"
cd "${REPO_ROOT}"
mkdir -p analysis/berry_attom_validation_v3/logs analysis/berry_attom_validation_v3/st_louis_source_robustness output/berry_attom_validation_v3/st_louis_source_robustness
CPUS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${CPUS}"
echo "stl_local host=$(hostname) partition=${SLURM_JOB_PARTITION:-} start=$(date -Is)"
"${PY}" analysis/berry_attom_validation_v3/scripts/stl_local_avm.py
echo "done $(date -Is)"
