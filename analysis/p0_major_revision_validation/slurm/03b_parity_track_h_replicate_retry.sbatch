#!/bin/bash
#SBATCH --job-name=p0_03b_parity_track_h_replicate_retry
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --output=/orcd/home/002/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/analysis/p0_major_revision_validation/logs/03b_parity_track_h_replicate_retry_%j.out
#SBATCH --error=/orcd/home/002/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/analysis/p0_major_revision_validation/logs/03b_parity_track_h_replicate_retry_%j.err
set -euo pipefail
cd /orcd/home/002/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal
# Historical thread environment. Determinism pins are applied by the Python layer for
# Track P only (plan F.0); never exported here.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
echo "host=$(hostname) job=${SLURM_JOB_ID} start=$(date -Is)"
/home/nacevedo/.conda/envs/fairness_env/bin/python \
  analysis/p0_major_revision_validation/code/p0_2_parity_tracks.py --mode replicate --out-suffix _retry
echo "end=$(date -Is)"
