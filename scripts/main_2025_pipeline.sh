#!/bin/bash
#SBATCH --job-name=svm_2025_pipeline
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=164GB
#SBATCH --output=temp/logs/svm_2025_pipeline_%j.out
#SBATCH --error=temp/logs/svm_2025_pipeline_%j.err
#SBATCH -t 1-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nacevedo@mit.edu

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal}"
PY="${PY:-/home/nacevedo/.conda/envs/fairness_env/bin/python}"

DATA_PATH="${DATA_PATH:-data/CCAO/2025/training_data.parquet}"
RESULT_ROOT="${RESULT_ROOT:-output/robust_rolling_origin_cv_2025}"
ASSESSMENT_YEAR="${ASSESSMENT_YEAR:-2025}"

BASELINE_SEARCH_TRIALS="${BASELINE_SEARCH_TRIALS:-180}"
RHO_VALUES="${RHO_VALUES:-0.1,100.0}"
RHO_COUNT="${RHO_COUNT:-50}"
RHO_SCALE="${RHO_SCALE:-geom}"
SAMPLE_FRAC="${SAMPLE_FRAC:-1.0}"
N_BOOTSTRAP="${N_BOOTSTRAP:-0}"
CPU_FRACTION="${CPU_FRACTION:-0.90}"

CPUS_ON_NODE="${SLURM_CPUS_PER_TASK:-64}"
DEFAULT_WORKERS=$(( CPUS_ON_NODE > 8 ? CPUS_ON_NODE - 4 : CPUS_ON_NODE ))
PARALLEL_MAX_WORKERS="${PARALLEL_MAX_WORKERS:-${DEFAULT_WORKERS}}"

DRY_RUN="${DRY_RUN:-0}"

run_cmd() {
  echo
  echo "[pipeline-2025] $*"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  "$@"
}

cd "${REPO_ROOT}"
mkdir -p temp/logs

if [[ ! -x "${PY}" ]]; then
  echo "ERROR: Python executable is not available: ${PY}" >&2
  exit 127
fi

echo "[pipeline-2025] started at $(date)"
echo "[pipeline-2025] host=$(hostname) cpus=${CPUS_ON_NODE} workers=${PARALLEL_MAX_WORKERS}"
echo "[pipeline-2025] python=${PY}"
echo "[pipeline-2025] data=${DATA_PATH}"
echo "[pipeline-2025] result_root=${RESULT_ROOT}"

run_cmd "${PY}" pipeline/00_ingest.py \
  --data-path "${DATA_PATH}" \
  --sample-rows 500000

run_cmd "${PY}" pipeline/01_train.py \
  --config cv_config.yaml \
  --data-path "${DATA_PATH}" \
  --result-root "${RESULT_ROOT}" \
  --assessment-year "${ASSESSMENT_YEAR}" \
  --heldout-test-mode assessment_year \
  --sample-frac "${SAMPLE_FRAC}" \
  --baseline-search \
  --baseline-search-trials "${BASELINE_SEARCH_TRIALS}" \
  --rho-values "${RHO_VALUES}" \
  --rho-count "${RHO_COUNT}" \
  --rho-scale "${RHO_SCALE}" \
  --ratio-modes diff \
  --no-include-cvar-models \
  --n-bootstrap "${N_BOOTSTRAP}" \
  --parallel \
  --parallel-cpu-fraction "${CPU_FRACTION}" \
  --parallel-max-workers "${PARALLEL_MAX_WORKERS}" \
  --parquet-engine fastparquet

run_cmd "${PY}" pipeline/02_assess.py \
  --result-root "${RESULT_ROOT}" \
  --accuracy-metric RMSE \
  --penalized-selection-mode mse_only

run_cmd "${PY}" pipeline/03_evaluate.py \
  --result-root "${RESULT_ROOT}"

run_cmd "${PY}" pipeline/04_interpret.py \
  --result-root "${RESULT_ROOT}"

run_cmd "${PY}" pipeline/05_finalize.py \
  --result-root "${RESULT_ROOT}"

run_cmd "${PY}" pipeline/06_report.py \
  --result-root "${RESULT_ROOT}" \
  --training-data "${DATA_PATH}"

echo
echo "[pipeline-2025] finished at $(date)"
