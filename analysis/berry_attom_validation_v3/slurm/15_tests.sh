#!/bin/bash
# Full v3 assertion suite against the finished artifacts.
#SBATCH --job-name=v3_tests
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=analysis/berry_attom_validation_v3/logs/tests_v3_%j.out
#SBATCH --error=analysis/berry_attom_validation_v3/logs/tests_v3_%j.err
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal}"
PY="${PY:-/home/nacevedo/.conda/envs/fairness_env/bin/python}"
cd "${REPO_ROOT}"
export PYTHONUNBUFFERED=1
echo "tests host=$(hostname) partition=${SLURM_JOB_PARTITION:-} start=$(date -Is)"
"${PY}" analysis/berry_attom_validation_v3/scripts/run_v3_tests.py
echo "done $(date -Is)"
