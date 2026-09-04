#!/bin/bash
#SBATCH --job-name=v1_codefreq
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=analysis/external_jurisdiction_benchmark_v1/logs/codefreq_%j.out
#SBATCH --error=analysis/external_jurisdiction_benchmark_v1/logs/codefreq_%j.err
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal}"
PY="${PY:-/home/nacevedo/.conda/envs/fairness_env/bin/python}"
cd "${REPO_ROOT}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
echo "codefreq host=$(hostname) partition=${SLURM_JOB_PARTITION:-} start=$(date -Is)"
"${PY}" analysis/external_jurisdiction_benchmark_v1/scripts/property_use_frequency_table.py
echo "done $(date -Is)"
