#!/bin/bash
# EXPERIMENTAL / TOY / NON-CANONICAL — merge metrics and write the three required figures.
#SBATCH --job-name=toy_surr_asm
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/output/toy_surrogate_ablation/logs/%x_%j.out
#SBATCH --error=/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/output/toy_surrogate_ablation/logs/%x_%j.err

set -euo pipefail
cd /home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal
mkdir -p output/toy_surrogate_ablation/logs

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg

/home/nacevedo/.conda/envs/fairness_env/bin/python scripts/toy_surrogate_ablation.py \
  --mode assemble \
  --output-root ./output/toy_surrogate_ablation

echo "TOY_ASSEMBLE_DONE"
