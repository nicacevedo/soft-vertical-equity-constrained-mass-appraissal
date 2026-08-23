#!/bin/bash
#SBATCH --job-name=mv_target_corr
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --array=0-32%8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB
#SBATCH --time=12:00:00
#SBATCH --output=temp/logs/mv_target_corr_%A_%a.out
#SBATCH --error=temp/logs/mv_target_corr_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nacevedo@mit.edu

set -euo pipefail

# Target-correction linear-family experiment launcher.
#
# Submit smoke:
#   SMOKE=1 sbatch -p sched_mit_sloan_batch_r8 --array=0-5%3 final_mv_target_correction_experiments.sh
#   SMOKE=1 sbatch -p mit_normal --array=0-5%3 final_mv_target_correction_experiments.sh
#
# Submit full default grid:
#   sbatch -p sched_mit_sloan_batch_r8 -t 2-00:00:00 --cpus-per-task=16 --mem=64GB --array=0-38%4 final_mv_target_correction_experiments.sh
#
# Aggregate completed task folders:
#   AGGREGATE_ONLY=1 RUN_TAG=<job_or_run_tag> bash final_mv_target_correction_experiments.sh

REPO_ROOT="${REPO_ROOT:-/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal}"
PY="${PY:-/home/nacevedo/.conda/envs/fairness_env/bin/python}"
SCRIPT_PATH="${SCRIPT_PATH:-final_market_value_1_target_correction.py}"

DATA_PATH="${DATA_PATH:-data/CCAO/2025/training_data.parquet}"
PARAMS_PATH="${PARAMS_PATH:-params.yaml}"
SMOKE="${SMOKE:-0}"
GRID_PRESET="${GRID_PRESET:-$([[ "${SMOKE}" == "1" ]] && echo smoke || echo full)}"
RUN_TAG="${RUN_TAG:-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}}}"
RESULT_ROOT="${RESULT_ROOT:-output/final_mv_target_correction/${RUN_TAG}}"

DEFAULT_TARGET_VARIANTS="raw,strict_raw,weighted_raw,robust_raw,time_global,time_hedonic_global,time_area:meta_township_code,time_hedonic_area:meta_township_code,local_shrink:0.90,local_shrink:0.75,local_shrink:0.50,local_shrink_adaptive,robust_local_shrink:0.75"
TARGET_VARIANTS="${TARGET_VARIANTS:-}"
LEARNERS="${LEARNERS:-}"
EVAL_PHASES="${EVAL_PHASES:-}"
CALIBRATION_MODES="${CALIBRATION_MODES:-}"
PREPROCESS_MODE="${PREPROCESS_MODE:-repo}"

SAMPLE_FRAC="${SAMPLE_FRAC:-}"
SAMPLE_SEED="${SAMPLE_SEED:-2025}"
ASSESS_EVAL_YEAR="${ASSESS_EVAL_YEAR:-2024}"
FIXED_DATE_MODE="${FIXED_DATE_MODE:-eval_min}"
PARQUET_ENGINE="${PARQUET_ENGINE:-pyarrow}"

CALIB_FRAC="${CALIB_FRAC:-0.20}"
RIDGE_ALPHA="${RIDGE_ALPHA:-10.0}"
LASSO_ALPHA="${LASSO_ALPHA:-0.001}"
RIDGE_ALPHAS="${RIDGE_ALPHAS:-0.01,0.03,0.1,0.3,1,3,10,30,100,300}"
LASSO_ALPHAS="${LASSO_ALPHAS:-0.00005,0.0001,0.0003,0.001,0.003,0.01,0.03}"
INNER_VAL_FRAC="${INNER_VAL_FRAC:-0.20}"
LASSO_MAX_ITER="${LASSO_MAX_ITER:-}"

TARGET_SHRINK_N="${TARGET_SHRINK_N:-100.0}"
STRICT_DEED_TYPES="${STRICT_DEED_TYPES:-01}"
SOFT_DEED_WEIGHTS="${SOFT_DEED_WEIGHTS:-01:1.0,02:0.85,05:0.75}"
ROBUST_C="${ROBUST_C:-1.5}"

LOCAL_K="${LOCAL_K:-20}"
LOCAL_SPATIAL_BW_MILES="${LOCAL_SPATIAL_BW_MILES:-1.0}"
LOCAL_TIME_BW_DAYS="${LOCAL_TIME_BW_DAYS:-500.0}"
BASE_RIDGE_ALPHA="${BASE_RIDGE_ALPHA:-10.0}"
ADAPTIVE_ALPHA_MIN="${ADAPTIVE_ALPHA_MIN:-0.10}"
ADAPTIVE_ALPHA_MAX="${ADAPTIVE_ALPHA_MAX:-0.98}"
CLASS_FILTER_COL="${CLASS_FILTER_COL:-char_class}"
MIN_SAME_CLASS_POOL="${MIN_SAME_CLASS_POOL:-10}"
MAX_NEIGHBOR_AGE_DAYS="${MAX_NEIGHBOR_AGE_DAYS:-}"
MAX_SPATIAL_CANDIDATES="${MAX_SPATIAL_CANDIDATES:-2048}"
CANDIDATE_MULTIPLIER="${CANDIDATE_MULTIPLIER:-64}"

CPUS_ON_NODE="${SLURM_CPUS_PER_TASK:-32}"
DEFAULT_WORKERS=$(( CPUS_ON_NODE > 6 ? CPUS_ON_NODE - 4 : CPUS_ON_NODE ))
N_JOBS="${N_JOBS:-${DEFAULT_WORKERS}}"
N_OOF_FOLDS="${N_OOF_FOLDS:-}"
OOF_MODE="${OOF_MODE:-rolling}"
N_DECILES="${N_DECILES:-10}"
PLOT_TOP_TARGETS="${PLOT_TOP_TARGETS:-10}"

DRY_RUN="${DRY_RUN:-0}"
AGGREGATE_ONLY="${AGGREGATE_ONLY:-0}"
NO_TUNE_ALPHAS="${NO_TUNE_ALPHAS:-}"
NO_ENGINEERED_FEATURES="${NO_ENGINEERED_FEATURES:-0}"
NO_STRICT_FEATURE_SCREEN="${NO_STRICT_FEATURE_SCREEN:-0}"
ALLOW_SALE_COUNT_FEATURE="${ALLOW_SALE_COUNT_FEATURE:-0}"
NO_CLASS_FILTER="${NO_CLASS_FILTER:-0}"
NO_CLASS_FALLBACK="${NO_CLASS_FALLBACK:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYARROW_NUM_THREADS="${PYARROW_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1

sanitize() {
  echo "$1" | sed -E 's/[^A-Za-z0-9_.-]+/_/g'
}

split_csv_to_array() {
  local raw="$1"
  local -n out_arr=$2
  IFS=',' read -r -a out_arr <<< "${raw}"
}

run_cmd() {
  echo
  echo "[mv-target-correction] $*"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  "$@"
}

aggregate_results() {
  RESULT_ROOT="${RESULT_ROOT}" "${PY}" - <<'PY'
import os
from pathlib import Path

import pandas as pd

from final_market_value_1_target_correction import plot_decile_curves, save_summary_tables

root = Path(os.environ["RESULT_ROOT"])
root.mkdir(parents=True, exist_ok=True)

def combine(name: str) -> pd.DataFrame:
    frames = []
    for path in sorted(root.glob(f"task_*/{name}")):
        if path.stat().st_size > 0:
            frames.append(pd.read_csv(path))
    out = pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()
    out.to_csv(root / name, index=False)
    print(f"[mv-target-correction] combined {name}: rows={len(out)}")
    return out

metrics = combine("metrics_all.csv")
deciles = combine("decile_curves.csv")
combine("target_diagnostics.csv")
if not metrics.empty:
    save_summary_tables(metrics, root)
    plot_decile_curves(deciles, metrics, root, max_targets=int(os.environ.get("PLOT_TOP_TARGETS", "10")))
PY
}

cd "${REPO_ROOT}"
mkdir -p temp/logs "${RESULT_ROOT}"

if [[ ! -x "${PY}" ]]; then
  echo "ERROR: Python executable is not available: ${PY}" >&2
  exit 127
fi
if [[ ! -f "${SCRIPT_PATH}" ]]; then
  echo "ERROR: target-correction script not found: ${SCRIPT_PATH}" >&2
  exit 2
fi
if [[ "${AGGREGATE_ONLY}" == "1" ]]; then
  aggregate_results
  exit 0
fi
if [[ ! -f "${DATA_PATH}" ]]; then
  echo "ERROR: data file not found: ${DATA_PATH}" >&2
  exit 2
fi
if [[ ! -f "${PARAMS_PATH}" ]]; then
  echo "ERROR: params file not found: ${PARAMS_PATH}" >&2
  exit 2
fi

case "${GRID_PRESET}" in
  smoke)
    TARGET_VARIANTS="${TARGET_VARIANTS:-raw,time_global}"
    LEARNERS="${LEARNERS:-linear,ridge,lasso}"
    SAMPLE_FRAC="${SAMPLE_FRAC:-0.005}"
    CALIBRATION_MODES="${CALIBRATION_MODES:-none,median_center}"
    EVAL_PHASES="${EVAL_PHASES:-fixed_assessment_date,actual_sale_date}"
    NO_TUNE_ALPHAS="${NO_TUNE_ALPHAS:-1}"
    N_OOF_FOLDS="${N_OOF_FOLDS:-2}"
    LASSO_MAX_ITER="${LASSO_MAX_ITER:-1500}"
    ;;
  focused)
    TARGET_VARIANTS="${TARGET_VARIANTS:-raw,weighted_raw,robust_raw,time_global,time_hedonic_global,time_area:meta_township_code,local_shrink:0.75,robust_local_shrink:0.75}"
    ;;
  full)
    ;;
  *)
    echo "ERROR: GRID_PRESET must be smoke, focused, or full; got ${GRID_PRESET}" >&2
    exit 2
    ;;
esac

TARGET_VARIANTS="${TARGET_VARIANTS:-${DEFAULT_TARGET_VARIANTS}}"
LEARNERS="${LEARNERS:-linear,ridge,lasso}"
EVAL_PHASES="${EVAL_PHASES:-fixed_assessment_date,actual_sale_date}"
CALIBRATION_MODES="${CALIBRATION_MODES:-none,median_center,affine}"
SAMPLE_FRAC="${SAMPLE_FRAC:-1.0}"
NO_TUNE_ALPHAS="${NO_TUNE_ALPHAS:-0}"
N_OOF_FOLDS="${N_OOF_FOLDS:-5}"
LASSO_MAX_ITER="${LASSO_MAX_ITER:-12000}"

split_csv_to_array "${TARGET_VARIANTS}" TARGET_ARR
split_csv_to_array "${LEARNERS}" LEARNER_ARR
N_TARGETS=${#TARGET_ARR[@]}
N_LEARNERS=${#LEARNER_ARR[@]}
TOTAL_TASKS=$(( N_TARGETS * N_LEARNERS ))
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

if (( TASK_ID >= TOTAL_TASKS )); then
  echo "[mv-target-correction] task_id=${TASK_ID} >= total_tasks=${TOTAL_TASKS}; exiting cleanly."
  exit 0
fi

TARGET_IDX=$(( TASK_ID / N_LEARNERS ))
LEARNER_IDX=$(( TASK_ID % N_LEARNERS ))
TARGET_VARIANT="${TARGET_ARR[$TARGET_IDX]}"
LEARNER="${LEARNER_ARR[$LEARNER_IDX]}"
TARGET_SAFE="$(sanitize "${TARGET_VARIANT}")"
LEARNER_SAFE="$(sanitize "${LEARNER}")"
TASK_OUT_DIR="${RESULT_ROOT}/task_${TASK_ID}__target_${TARGET_SAFE}__learner_${LEARNER_SAFE}"
mkdir -p "${TASK_OUT_DIR}"

CMD=(
  "${PY}" "${SCRIPT_PATH}"
  --data-path "${DATA_PATH}"
  --params-path "${PARAMS_PATH}"
  --out-dir "${TASK_OUT_DIR}"
  --parquet-engine "${PARQUET_ENGINE}"
  --sample-frac "${SAMPLE_FRAC}"
  --sample-seed "${SAMPLE_SEED}"
  --assess-eval-year "${ASSESS_EVAL_YEAR}"
  --fixed-date-mode "${FIXED_DATE_MODE}"
  --eval-phases "${EVAL_PHASES}"
  --target-variants "${TARGET_VARIANT}"
  --learners "${LEARNER}"
  --preprocess-mode "${PREPROCESS_MODE}"
  --calibration-modes "${CALIBRATION_MODES}"
  --calib-frac "${CALIB_FRAC}"
  --ridge-alpha "${RIDGE_ALPHA}"
  --lasso-alpha "${LASSO_ALPHA}"
  --ridge-alphas "${RIDGE_ALPHAS}"
  --lasso-alphas "${LASSO_ALPHAS}"
  --inner-val-frac "${INNER_VAL_FRAC}"
  --lasso-max-iter "${LASSO_MAX_ITER}"
  --target-shrink-n "${TARGET_SHRINK_N}"
  --strict-deed-types "${STRICT_DEED_TYPES}"
  --soft-deed-weights "${SOFT_DEED_WEIGHTS}"
  --robust-c "${ROBUST_C}"
  --local-K "${LOCAL_K}"
  --local-spatial-bw-miles "${LOCAL_SPATIAL_BW_MILES}"
  --local-time-bw-days "${LOCAL_TIME_BW_DAYS}"
  --base-ridge-alpha "${BASE_RIDGE_ALPHA}"
  --adaptive-alpha-min "${ADAPTIVE_ALPHA_MIN}"
  --adaptive-alpha-max "${ADAPTIVE_ALPHA_MAX}"
  --class-filter-col "${CLASS_FILTER_COL}"
  --min-same-class-pool "${MIN_SAME_CLASS_POOL}"
  --max-spatial-candidates "${MAX_SPATIAL_CANDIDATES}"
  --candidate-multiplier "${CANDIDATE_MULTIPLIER}"
  --n-jobs "${N_JOBS}"
  --n-oof-folds "${N_OOF_FOLDS}"
  --oof-mode "${OOF_MODE}"
  --n-deciles "${N_DECILES}"
  --plot-top-targets "${PLOT_TOP_TARGETS}"
)

if [[ "${NO_TUNE_ALPHAS}" == "1" ]]; then
  CMD+=(--no-tune-alphas)
fi
if [[ "${NO_ENGINEERED_FEATURES}" == "1" ]]; then
  CMD+=(--no-engineered-features)
fi
if [[ "${NO_STRICT_FEATURE_SCREEN}" == "1" ]]; then
  CMD+=(--no-strict-feature-screen)
fi
if [[ "${ALLOW_SALE_COUNT_FEATURE}" == "1" ]]; then
  CMD+=(--allow-sale-count-feature)
fi
if [[ "${NO_CLASS_FILTER}" == "1" ]]; then
  CMD+=(--no-class-filter)
fi
if [[ "${NO_CLASS_FALLBACK}" == "1" ]]; then
  CMD+=(--no-class-fallback)
fi
if [[ -n "${MAX_NEIGHBOR_AGE_DAYS}" ]]; then
  CMD+=(--max-neighbor-age-days "${MAX_NEIGHBOR_AGE_DAYS}")
fi
if [[ -n "${EXTRA_ARGS}" ]]; then
  read -r -a EXTRA_ARR <<< "${EXTRA_ARGS}"
  CMD+=("${EXTRA_ARR[@]}")
fi

echo "[mv-target-correction] started at $(date)"
echo "[mv-target-correction] host=$(hostname) job=${SLURM_JOB_ID:-NA} array_task=${TASK_ID}/${TOTAL_TASKS}"
echo "[mv-target-correction] partition=${SLURM_JOB_PARTITION:-NA} cpus=${CPUS_ON_NODE} n_jobs=${N_JOBS}"
echo "[mv-target-correction] grid=${GRID_PRESET} target=${TARGET_VARIANT} learner=${LEARNER}"
echo "[mv-target-correction] output=${TASK_OUT_DIR}"

run_cmd "${CMD[@]}"

echo "[mv-target-correction] finished at $(date)"
