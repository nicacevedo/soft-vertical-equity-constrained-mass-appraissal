#!/bin/bash
# EXPERIMENTAL / TOY / NON-CANONICAL — assemble, audit, optional cleanup.
#SBATCH --job-name=toy_mech_asm
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/output/toy_surrogate_ablation_v2/logs/%x_%j.out
#SBATCH --error=/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/output/toy_surrogate_ablation_v2/logs/%x_%j.err

set -euo pipefail
cd /home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal
mkdir -p output/toy_surrogate_ablation_v2/logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export NUMBA_CACHE_DIR="${HOME}/.cache/numba"
mkdir -p "$NUMBA_CACHE_DIR"

echo "TOY_MECH_ASSEMBLE partition=${SLURM_JOB_PARTITION:-} node=${SLURMD_NODENAME:-} job=${SLURM_JOB_ID:-}"

/home/nacevedo/.conda/envs/fairness_env/bin/python scripts/toy_mechanism_selection.py \
  --mode assemble \
  --output-root ./output/toy_surrogate_ablation_v2

echo "TOY_MECH_ASSEMBLE_DONE"
