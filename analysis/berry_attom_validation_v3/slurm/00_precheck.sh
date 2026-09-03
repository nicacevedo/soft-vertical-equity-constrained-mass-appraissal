#!/bin/bash
#SBATCH --job-name=v3_precheck
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --output=analysis/berry_attom_validation_v3/logs/precheck_%j.out
#SBATCH --error=analysis/berry_attom_validation_v3/logs/precheck_%j.err
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal}"
PY="${PY:-/home/nacevedo/.conda/envs/fairness_env/bin/python}"
cd "${REPO_ROOT}"
mkdir -p analysis/berry_attom_validation_v3/logs
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
echo "precheck host=$(hostname) start=$(date -Is) partition=${SLURM_JOB_PARTITION:-}"
"${PY}" analysis/berry_attom_validation_v3/scripts/copy_v2_provenance.py
"${PY}" analysis/berry_attom_validation_v3/scripts/write_six_county_metadata.py
"${PY}" analysis/berry_attom_validation_v3/scripts/run_v3_tests.py
"${PY}" analysis/berry_attom_validation_v3/scripts/write_final_report.py
echo "precheck done $(date -Is)"
