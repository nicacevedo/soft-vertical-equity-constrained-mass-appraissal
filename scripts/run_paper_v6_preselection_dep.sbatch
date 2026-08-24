#!/bin/bash
#SBATCH --job-name=p6_dep
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/output/paper_v6_preselection/logs/%x_%j.out
#SBATCH --error=/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/output/paper_v6_preselection/logs/%x_%j.err

set -euo pipefail
cd /home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal
mkdir -p output/paper_v6_preselection/logs
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-$PWD/output/paper_v6_preselection/cache}/numba_${SLURM_JOB_ID}"
mkdir -p "$NUMBA_CACHE_DIR"

CMD="${P6_CMD:?}"
echo "DEP command=$CMD job=$SLURM_JOB_ID"
conda run -n fairness_env --no-capture-output python scripts/paper_v6_preselection_pipeline.py "$CMD"
echo "DEP_DONE $CMD"
