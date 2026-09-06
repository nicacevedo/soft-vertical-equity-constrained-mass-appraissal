#!/bin/bash
#SBATCH --job-name=p0_01_parity_track_h_ladder
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --output=/orcd/home/002/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/analysis/p0_major_revision_validation/logs/01_parity_track_h_ladder_%j.out
#SBATCH --error=/orcd/home/002/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/analysis/p0_major_revision_validation/logs/01_parity_track_h_ladder_%j.err
set -euo pipefail
cd /orcd/home/002/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal
# Historical thread environment. Determinism pins are applied by the Python layer for
# Track P only (plan F.0); never exported here.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
echo "host=$(hostname) job=${SLURM_JOB_ID} start=$(date -Is)"
/home/nacevedo/.conda/envs/fairness_env/bin/python \
  analysis/p0_major_revision_validation/code/p0_2_parity_tracks.py --mode ladder --track historical --capacities T,M,F --splits heldout,forward_2025
echo "end=$(date -Is)"
