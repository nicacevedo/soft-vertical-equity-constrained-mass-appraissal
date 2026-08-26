#!/bin/bash
#SBATCH --job-name=p6_agg
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/output/paper_v6_preselection_994/logs/%x_%j.out
#SBATCH --error=/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/output/paper_v6_preselection_994/logs/%x_%j.err

set -euo pipefail
cd /home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1

set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fairness_env
set -u

python scripts/final_local_results_994.py aggregate-delta-nl --n-shards 16
echo "AGGREGATE_DONE"
