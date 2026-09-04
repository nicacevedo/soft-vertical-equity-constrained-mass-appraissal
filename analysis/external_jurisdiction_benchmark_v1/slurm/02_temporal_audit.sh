#!/bin/bash
#SBATCH --job-name=v1_temporal
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=analysis/external_jurisdiction_benchmark_v1/logs/temporal_%j.out
#SBATCH --error=analysis/external_jurisdiction_benchmark_v1/logs/temporal_%j.err
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal}"
PY="${PY:-/home/nacevedo/.conda/envs/fairness_env/bin/python}"
cd "${REPO_ROOT}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
echo "temporal_audit host=$(hostname) partition=${SLURM_JOB_PARTITION:-} start=$(date -Is)"
"${PY}" analysis/external_jurisdiction_benchmark_v1/scripts/temporal_completeness_audit.py
echo "done $(date -Is)"
