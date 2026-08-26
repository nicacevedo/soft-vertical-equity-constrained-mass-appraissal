#!/bin/bash
#SBATCH --job-name=p6_dnl
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --array=0-15%12
#SBATCH --requeue
#SBATCH --output=/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/output/paper_v6_preselection_994/logs/%x_%A_%a.out
#SBATCH --error=/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/output/paper_v6_preselection_994/logs/%x_%A_%a.err

set -euo pipefail
cd /home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal
mkdir -p output/paper_v6_preselection_994/logs output/paper_v6_preselection_994/final_local_results/delta_nl_shards
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg

set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fairness_env
set -u

python scripts/final_local_results_994.py delta-nl-shard \
  --shard "${SLURM_ARRAY_TASK_ID}" \
  --n-shards 16

echo "DELTA_NL_SHARD_DONE ${SLURM_ARRAY_TASK_ID}"
