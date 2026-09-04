#!/bin/bash
# Postprocess after all 18 forward path tasks succeed.
#SBATCH --job-name=v1_fwd_post
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=analysis/external_jurisdiction_benchmark_v1/logs/fwd_post_%j.out
#SBATCH --error=analysis/external_jurisdiction_benchmark_v1/logs/fwd_post_%j.err
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal}"
PY="${PY:-/home/nacevedo/.conda/envs/fairness_env/bin/python}"
cd "${REPO_ROOT}"
CPUS="${SLURM_CPUS_PER_TASK:-8}"
NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_cache_${SLURM_JOB_ID:-0}"
mkdir -p "${NUMBA_CACHE_DIR}"
export NUMBA_CACHE_DIR
trap 'rm -rf "${NUMBA_CACHE_DIR}"' EXIT
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${CPUS}"
echo "forward_postprocess host=$(hostname) partition=${SLURM_JOB_PARTITION:-} start=$(date -Is)"
"${PY}" analysis/external_jurisdiction_benchmark_v1/scripts/aggregate_forward_results.py
"${PY}" analysis/external_jurisdiction_benchmark_v1/scripts/run_forward_bootstrap.py
"${PY}" analysis/external_jurisdiction_benchmark_v1/scripts/make_forward_figures.py
"${PY}" analysis/external_jurisdiction_benchmark_v1/scripts/write_forward_reports.py
"${PY}" analysis/external_jurisdiction_benchmark_v1/scripts/run_v1_tests.py
echo "done $(date -Is)"
