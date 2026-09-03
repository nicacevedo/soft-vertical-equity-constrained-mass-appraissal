#!/bin/bash
#SBATCH --job-name=v3_lgbm_probe
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0:05:00
#SBATCH --output=analysis/berry_attom_validation_v3/logs/lgbm_probe_r8_%j.out
#SBATCH --error=analysis/berry_attom_validation_v3/logs/lgbm_probe_r8_%j.err
set -euo pipefail
PY="${PY:-/home/nacevedo/.conda/envs/fairness_env/bin/python}"
echo "host=$(hostname) partition=${SLURM_JOB_PARTITION:-}"
ldd --version | head -1
"${PY}" -c "import lightgbm, pyarrow, pandas, sklearn; print('lgbm_ok', lightgbm.__version__)"
