#!/bin/bash
# EXPERIMENTAL / TOY / NON-CANONICAL V2 extension: missing training-s refinements only.
# Does not rerun calibration, rho=0, or dense lambda paths.
# Array 0=quadratic, 1=huber, 2=absolute.
#SBATCH --job-name=toy_v2_ext
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/output/toy_surrogate_ablation_v2/logs/%x_%A_%a.out
#SBATCH --error=/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/output/toy_surrogate_ablation_v2/logs/%x_%A_%a.err

set -euo pipefail
cd /home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal
mkdir -p output/toy_surrogate_ablation_v2/logs output/toy_surrogate_ablation_v2/cache

FAMILIES=(quadratic huber absolute)
TARGETS=(
  "0.30,0.25,0.20,0.15,0.10"
  "0.25,0.20,0.15,0.10"
  "0.40,0.30"
)
FAMILY="${FAMILIES[$SLURM_ARRAY_TASK_ID]}"
TARG="${TARGETS[$SLURM_ARRAY_TASK_ID]}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-$PWD/output/toy_surrogate_ablation_v2/cache}/numba_${SLURM_JOB_ID:-$$}_${SLURM_ARRAY_TASK_ID:-}"
mkdir -p "$NUMBA_CACHE_DIR"
export LOKY_WORKER_TIMEOUT=14400

echo "TOY_V2_EXTEND family=$FAMILY targets=$TARG task=$SLURM_ARRAY_TASK_ID partition=${SLURM_JOB_PARTITION:-} node=${SLURMD_NODENAME:-} cpus=${SLURM_CPUS_PER_TASK:-}"

/home/nacevedo/.conda/envs/fairness_env/bin/python scripts/toy_surrogate_ablation_v2.py \
  --mode extend-targets \
  --family "$FAMILY" \
  --targets "$TARG" \
  --output-root ./output/toy_surrogate_ablation_v2

echo "TOY_V2_EXTEND_DONE family=$FAMILY"
