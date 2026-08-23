#!/bin/bash
# =============================================================================
# projection_linear_experiments.sh
#
# SLURM job-array driver for the exact linear projection-path experiment in the
# rho-projection manuscript. Each task runs one held-out assessment-year dataset.
#
# Outputs:
#   ${OUT_ROOT}/${src}_assess${year}/
#     linear_projection_metrics.csv
#     linear_projection_theory_empirical_comparison.csv
#     linear_projection_verification_summary.csv
#     plots/*.png
#
# Usage:
#   sbatch scripts/projection_linear_experiments.sh
#   sbatch --array=0-1 scripts/projection_linear_experiments.sh
#   bash scripts/projection_linear_experiments.sh list
#   PROJECTION_LINEAR_TASK_ID=0 bash scripts/projection_linear_experiments.sh
# =============================================================================
#SBATCH --job-name=proj_linear
#SBATCH --partition=mit_normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96GB
#SBATCH --array=0-3
#SBATCH --output=temp/logs/projection_linear_%A_%a.out
#SBATCH --error=temp/logs/projection_linear_%A_%a.err
#SBATCH -t 0-04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nacevedo@mit.edu

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal}"
PY="${PY:-/home/nacevedo/.conda/envs/fairness_env/bin/python}"
cd "${REPO_ROOT}"

Q_GRID="${Q_GRID:-1.00,0.90,0.80,0.70,0.60,0.50,0.40,0.30,0.20}"
SAMPLE_FRAC="${SAMPLE_FRAC:-1}"
SEED="${SEED:-4050}"
OUT_ROOT="${OUT_ROOT:-output/projection_linear}"

SRC_LABELS=(ccao2025 ccao_old ccao_sim2024 ccao_sim2023)
SRC_PATHS=(
  "data/CCAO/2025/training_data.parquet"
  "data/CCAO/2025/training_data_old.parquet"
  "data/CCAO/2025/training_data_sim2024.parquet"
  "data/CCAO/2025/training_data_sim2023.parquet"
)
SRC_ASSESS_YEARS=(2025 2024 2023 2022)
N_TOTAL=${#SRC_LABELS[@]}

derive() {
  local tid="$1"
  EXP_SRC="${SRC_LABELS[$tid]}"
  EXP_PATH="${SRC_PATHS[$tid]}"
  EXP_YEAR="${SRC_ASSESS_YEARS[$tid]}"
  EXP_OUT="${OUT_ROOT}/${EXP_SRC}_assess${EXP_YEAR}"
}

if [[ "${1:-}" == "list" ]]; then
  echo "linear projection experiment matrix (${N_TOTAL} tasks):"
  for ((t=0; t<N_TOTAL; t++)); do
    derive "$t"
    printf "  [%2d] src=%-13s assess=%s -> %s\n" "$t" "$EXP_SRC" "$EXP_YEAR" "$EXP_OUT"
  done
  exit 0
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-${PROJECTION_LINEAR_TASK_ID:-${1:-}}}"
if [[ -z "${TASK_ID}" ]]; then
  echo "ERROR: no task id. Run under sbatch (--array) or pass PROJECTION_LINEAR_TASK_ID / a CLI index." >&2
  exit 2
fi
if (( TASK_ID < 0 || TASK_ID >= N_TOTAL )); then
  echo "ERROR: task id ${TASK_ID} out of range [0, $((N_TOTAL-1))]." >&2
  exit 2
fi

CPUS="${SLURM_CPUS_PER_TASK:-$(nproc)}"
export OMP_NUM_THREADS="${CPUS}"
export OPENBLAS_NUM_THREADS="${CPUS}"
export MKL_NUM_THREADS="${CPUS}"
export NUMEXPR_NUM_THREADS="${CPUS}"

mkdir -p temp/logs "${OUT_ROOT}"
derive "${TASK_ID}"

echo "==============================================================="
echo "[projection_linear] task=${TASK_ID}/${N_TOTAL} started $(date)"
echo "[projection_linear] host=$(hostname) cpus=${CPUS} python=${PY}"
echo "[projection_linear] source=${EXP_SRC} data=${EXP_PATH} assess_year=${EXP_YEAR}"
echo "[projection_linear] q_grid=${Q_GRID}"
echo "[projection_linear] out_dir=${EXP_OUT}"
echo "==============================================================="

if [[ ! -x "${PY}" ]]; then
  echo "ERROR: Python executable not available: ${PY}" >&2
  exit 127
fi
if [[ ! -f "${EXP_PATH}" ]]; then
  echo "ERROR: data file not found: ${EXP_PATH}" >&2
  exit 2
fi

set +e
"${PY}" scripts/projection_linear_experiments.py \
  --data-source-label "${EXP_SRC}" \
  --data-path "${EXP_PATH}" \
  --assessment-year "${EXP_YEAR}" \
  --q-grid "${Q_GRID}" \
  --out-dir "${EXP_OUT}" \
  --sample-frac "${SAMPLE_FRAC}" \
  --seed "${SEED}"
ec=$?
echo "[projection_linear] task=${TASK_ID} (${EXP_SRC}) finished $(date) exit=${ec}"
exit "${ec}"
