#!/bin/bash
# EXPERIMENTAL / TOY / NON-CANONICAL surrogate ablation — one full-data lambda=0 fit
# plus rho-scale calibration. Used as the timing probe and shared lambda=0 artifact.
#SBATCH --job-name=toy_surr_probe
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/output/toy_surrogate_ablation/logs/%x_%j.out
#SBATCH --error=/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/output/toy_surrogate_ablation/logs/%x_%j.err

set -euo pipefail
cd /home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal
mkdir -p output/toy_surrogate_ablation/logs output/toy_surrogate_ablation/cache

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-$PWD/output/toy_surrogate_ablation/cache}/numba_${SLURM_JOB_ID:-$$}"
mkdir -p "$NUMBA_CACHE_DIR"
export LOKY_WORKER_TIMEOUT=14400

echo "TOY_PROBE partition=${SLURM_JOB_PARTITION:-} node=${SLURMD_NODENAME:-} cpus=${SLURM_CPUS_PER_TASK:-} job=${SLURM_JOB_ID:-}"

/home/nacevedo/.conda/envs/fairness_env/bin/python scripts/toy_surrogate_ablation.py \
  --mode probe \
  --output-root ./output/toy_surrogate_ablation

echo "TOY_PROBE_DONE"
