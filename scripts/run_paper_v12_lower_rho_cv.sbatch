#!/bin/bash
#SBATCH --job-name=v12lr_cv
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=36:00:00
#SBATCH --output=/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/output/paper_v12_lower_rho_extension_994_v2/logs/cv_%j.out
#SBATCH --error=/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/output/paper_v12_lower_rho_extension_994_v2/logs/cv_%j.err

set -euo pipefail
cd /home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal
mkdir -p output/paper_v12_lower_rho_extension_994_v2/logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/numba_cache_${SLURM_JOB_ID:-$$}"
mkdir -p "$NUMBA_CACHE_DIR"
export LOKY_WORKER_TIMEOUT=14400

PY="/home/nacevedo/.conda/envs/fairness_env/bin/python"
RHOS=$("$PY" scripts/paper_v12_lower_rho_v2.py print-rho-cli)
echo "CV32 n_rhos=$(echo "$RHOS" | awk -F, '{print NF}') node=${SLURMD_NODENAME:-}"

"$PY" run_temporal_cv.py \
  --stage cv \
  --result-root ./output/paper_v12_lower_rho_extension_994_v2 \
  --lgbm-config-json ./output/paper_v6_preselection_994/lgbm_config.json \
  --no-include-cvar-models \
  --only-model-names LGBCovPenalty,LGBSmoothPenalty \
  --rho-values "$RHOS" \
  --no-explicit-zero \
  --parallel \
  --parallel-cpu-fraction 0.90 \
  --parallel-max-workers 16 \
  --n-bootstrap 0

"$PY" scripts/paper_v12_lower_rho_v2.py cv-qa
echo "CV32_DONE"
