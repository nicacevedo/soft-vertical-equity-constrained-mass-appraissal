#!/bin/bash
#SBATCH --job-name=v12lr_stage
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/output/paper_v12_lower_rho_extension_994_v2/logs/%x_%j.out
#SBATCH --error=/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/output/paper_v12_lower_rho_extension_994_v2/logs/%x_%j.err

set -euo pipefail
cd /home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/numba_cache_${SLURM_JOB_ID:-$$}"
mkdir -p "$NUMBA_CACHE_DIR" output/paper_v12_lower_rho_extension_994_v2/logs

STAGE="${V12_STAGE:?}"
PY="/home/nacevedo/.conda/envs/fairness_env/bin/python"
echo "V12LR stage=$STAGE job=${SLURM_JOB_ID:-} node=${SLURMD_NODENAME:-}"
"$PY" scripts/paper_v12_lower_rho_v2.py "$STAGE"
echo "V12LR_DONE $STAGE"
