#!/bin/bash
# Cook full 2016-2025 table: 80G OOM'd on 21976297_8.
#SBATCH --job-name=v1_fwd_cook_tbl
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=8:00:00
#SBATCH --output=analysis/external_jurisdiction_benchmark_v1/logs/fwd_tables_cook_%j.out
#SBATCH --error=analysis/external_jurisdiction_benchmark_v1/logs/fwd_tables_cook_%j.err
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal}"
PY="${PY:-/home/nacevedo/.conda/envs/fairness_env/bin/python}"
cd "${REPO_ROOT}"
CPUS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${CPUS}"
echo "forward_tables county=cook cpus=${CPUS} host=$(hostname) partition=${SLURM_JOB_PARTITION:-} start=$(date -Is)"
"${PY}" analysis/external_jurisdiction_benchmark_v1/scripts/build_modeling_tables.py --county-key cook --mode forward --end-date 2025-12-31
echo "done cook $(date -Is)"
