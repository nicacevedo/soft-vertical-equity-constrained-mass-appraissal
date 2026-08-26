#!/bin/bash
#SBATCH --job-name=p6_recal
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/output/paper_v6_preselection_994/logs/%x_%j.out
#SBATCH --error=/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/output/paper_v6_preselection_994/logs/%x_%j.err

set -euo pipefail
cd /home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal
mkdir -p output/paper_v6_preselection_994/logs output/paper_v6_preselection_994/final_local_results
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg

set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fairness_env
set -u

python scripts/final_local_results_994.py recalibrate
echo "RECALIBRATE_DONE"
