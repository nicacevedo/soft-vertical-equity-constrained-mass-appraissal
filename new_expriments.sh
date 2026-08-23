#!/bin/bash
#SBATCH --job-name=mv_spatial_2025
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --array=0-19%8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB
#SBATCH --output=temp/logs/mv_spatial_2025_%A_%a.out
#SBATCH --error=temp/logs/mv_spatial_2025_%A_%a.err
#SBATCH -t 1-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nacevedo@mit.edu

set -euo pipefail

# -----------------------------------------------------------------------------
# Market-value target + spatial-comparable AVM experiments
# -----------------------------------------------------------------------------
# Efficient design:
#   - Use a SLURM array to split the grid by (target_variant, learner).
#   - By default, use a focused learner-specific grid based on the first completed
#     results: Ridge runs only the local-residual correction; LightGBM runs only
#     full-X time-adjusted comparable lags. Set GRID_PRESET=full to recover the
#     broad exploratory grid.
#   - Each task requests half of an r8 node by default, so the scheduler can pack
#     two tasks per node. Joblib threads are used for KDTree candidate queries;
#     LightGBM uses a small explicit thread pool.
#
# Default array size above is intentionally an upper bound. The focused default
# uses 3 target variants x 3 learners = 9 tasks, and extra array tasks exit
# cleanly. Override TARGET_VARIANTS/LEARNERS when you want a larger pass.
#
# Work per task under GRID_PRESET=full is approximately two splits (test +
# assess) × |K|×|h_space|×|h_time|×|spatial_feature_types| augmented-model fits,
# plus residual-correction fits when residual is included. The focused default
# removes the dominated stages from that product.
#
# Optional denser/full grid (export before sbatch):
#   export GRID_PRESET=full
#   export KS=10,20,30,40
#   export SPATIAL_BWS=0.75,1.0,1.25,1.5
#   export TIME_BWS=184,365,500,730
#
# To ease scheduler/memory pressure, reduce concurrent array tasks, e.g.:
#   #SBATCH --array=0-19%2
# -----------------------------------------------------------------------------

REPO_ROOT="${REPO_ROOT:-/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal}"
PY="${PY:-/home/nacevedo/.conda/envs/fairness_env/bin/python}"
SCRIPT_PATH="${SCRIPT_PATH:-final_market_value_spatial_experiments.py}"

DATA_PATH="${DATA_PATH:-data/CCAO/2025/training_data.parquet}"
PARAMS_PATH="${PARAMS_PATH:-params.yaml}"
RUN_TAG="${RUN_TAG:-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}}}"
RESULT_ROOT="${RESULT_ROOT:-output/final_mv_spatial_2025/${RUN_TAG}}"

# Main experiment grid. Keep commas; they are parsed below and also passed to Python.
TARGET_VARIANTS="${TARGET_VARIANTS:-raw,global_slope,hedonic_area_slope:meta_township_code}"
LEARNERS="${LEARNERS:-ridge,lgbm_l2,lgbm_l1}"
GRID_PRESET="${GRID_PRESET:-focused}"  # focused or full
SPATIAL_FEATURE_TYPES="${SPATIAL_FEATURE_TYPES:-}"
SPATIAL_STAGES="${SPATIAL_STAGES:-}"
KS="${KS:-}"
SPATIAL_BWS="${SPATIAL_BWS:-}"
TIME_BWS="${TIME_BWS:-}"
CALIBRATION_MODES="${CALIBRATION_MODES:-none,median_center,affine}"

# Data/evaluation settings.
SAMPLE_FRAC="${SAMPLE_FRAC:-1.0}"
SAMPLE_SEED="${SAMPLE_SEED:-2025}"
ASSESS_EVAL_YEAR="${ASSESS_EVAL_YEAR:-2024}"
VALUATION_DATE_MODE="${VALUATION_DATE_MODE:-eval_max}"  # alternatives: assessment_date, eval_min, eval_median
PARQUET_ENGINE="${PARQUET_ENGINE:-pyarrow}"

# Model settings.
CALIB_FRAC="${CALIB_FRAC:-0.20}"
RIDGE_ALPHA="${RIDGE_ALPHA:-1.0}"
N_ESTIMATORS="${N_ESTIMATORS:-600}"
LEARNING_RATE="${LEARNING_RATE:-0.05}"
LGBM_N_JOBS="${LGBM_N_JOBS:-8}"
TARGET_SHRINK_N="${TARGET_SHRINK_N:-100.0}"
N_OOF_FOLDS="${N_OOF_FOLDS:-5}"
OOF_MODE="${OOF_MODE:-rolling}"  # rolling is leak-safer; blocked reproduces older runs.
N_DECILES="${N_DECILES:-10}"

# Spatial settings.
CLASS_FILTER_COL="${CLASS_FILTER_COL:-char_class}"
MIN_SAME_CLASS_POOL="${MIN_SAME_CLASS_POOL:-10}"
MAX_NEIGHBOR_AGE_DAYS="${MAX_NEIGHBOR_AGE_DAYS:-}"  # empty means no max-age filter
MAX_SPATIAL_CANDIDATES="${MAX_SPATIAL_CANDIDATES:-2048}"
CANDIDATE_MULTIPLIER="${CANDIDATE_MULTIPLIER:-64}"

# Resource settings.
CPUS_ON_NODE="${SLURM_CPUS_PER_TASK:-32}"
DEFAULT_WORKERS=$(( CPUS_ON_NODE > 8 ? CPUS_ON_NODE - 4 : CPUS_ON_NODE ))
N_JOBS="${N_JOBS:-${DEFAULT_WORKERS}}"

# Keep BLAS/OpenMP from oversubscribing. The Python runner parallelizes explicitly
# through joblib for spatial candidate queries and passes LGBM_N_JOBS to LightGBM.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${LGBM_N_JOBS}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
# Cap Arrow decode threads so login nodes / shared jobs avoid pthread stampedes.
export PYARROW_NUM_THREADS="${PYARROW_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1

# Toggles.
DRY_RUN="${DRY_RUN:-0}"
ALLOW_SEQUENTIAL_EVAL_HISTORY="${ALLOW_SEQUENTIAL_EVAL_HISTORY:-0}"
NO_ENGINEERED_FEATURES="${NO_ENGINEERED_FEATURES:-0}"
NO_STRICT_FEATURE_SCREEN="${NO_STRICT_FEATURE_SCREEN:-0}"
NO_CLASS_FILTER="${NO_CLASS_FILTER:-0}"
NO_CLASS_FALLBACK="${NO_CLASS_FALLBACK:-0}"
RESUME="${RESUME:-1}"
NO_STREAM_METRICS="${NO_STREAM_METRICS:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

run_cmd() {
  echo
  echo "[mv-spatial] $*"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  "$@"
}

sanitize() {
  echo "$1" | sed -E 's/[^A-Za-z0-9_.-]+/_/g'
}

split_csv_to_array() {
  local raw="$1"
  local -n out_arr=$2
  IFS=',' read -r -a out_arr <<< "${raw}"
}

cd "${REPO_ROOT}"
mkdir -p temp/logs "${RESULT_ROOT}"

if [[ ! -x "${PY}" ]]; then
  echo "ERROR: Python executable is not available: ${PY}" >&2
  exit 127
fi
if [[ ! -f "${SCRIPT_PATH}" ]]; then
  echo "ERROR: experiment script not found: ${SCRIPT_PATH}" >&2
  echo "Place final_market_value_spatial_experiments.py in REPO_ROOT or set SCRIPT_PATH." >&2
  exit 2
fi
if [[ ! -f "${DATA_PATH}" ]]; then
  echo "ERROR: data file not found: ${DATA_PATH}" >&2
  exit 2
fi
if [[ ! -f "${PARAMS_PATH}" ]]; then
  echo "ERROR: params file not found: ${PARAMS_PATH}" >&2
  exit 2
fi

split_csv_to_array "${TARGET_VARIANTS}" TARGET_ARR
split_csv_to_array "${LEARNERS}" LEARNER_ARR
N_TARGETS=${#TARGET_ARR[@]}
N_LEARNERS=${#LEARNER_ARR[@]}
TOTAL_TASKS=$(( N_TARGETS * N_LEARNERS ))
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

if (( TASK_ID >= TOTAL_TASKS )); then
  echo "[mv-spatial] task_id=${TASK_ID} >= total_tasks=${TOTAL_TASKS}; exiting cleanly."
  exit 0
fi

TARGET_IDX=$(( TASK_ID / N_LEARNERS ))
LEARNER_IDX=$(( TASK_ID % N_LEARNERS ))
TARGET_VARIANT="${TARGET_ARR[$TARGET_IDX]}"
LEARNER="${LEARNER_ARR[$LEARNER_IDX]}"

case "${GRID_PRESET}" in
  focused)
    if [[ "${LEARNER}" == "ridge" || "${LEARNER}" == "linear" || "${LEARNER}" == "huber" ]]; then
      SPATIAL_FEATURE_TYPES="${SPATIAL_FEATURE_TYPES:-residual}"
      SPATIAL_STAGES="${SPATIAL_STAGES:-base_plus_local_residual}"
      KS="${KS:-20,30,40}"
      SPATIAL_BWS="${SPATIAL_BWS:-0.75,1.0}"
      TIME_BWS="${TIME_BWS:-500,730}"
    elif [[ "${LEARNER}" == lgbm* ]]; then
      SPATIAL_FEATURE_TYPES="${SPATIAL_FEATURE_TYPES:-time_adjusted_price}"
      SPATIAL_STAGES="${SPATIAL_STAGES:-full_X_plus_spatial}"
      KS="${KS:-10,20,30}"
      SPATIAL_BWS="${SPATIAL_BWS:-0.75,1.0,1.25}"
      TIME_BWS="${TIME_BWS:-365,500}"
    else
      SPATIAL_FEATURE_TYPES="${SPATIAL_FEATURE_TYPES:-scaled_time_adjusted_price,residual,time_adjusted_price,scaled_target_label}"
      SPATIAL_STAGES="${SPATIAL_STAGES:-full_X_plus_spatial,base_plus_local_residual}"
      KS="${KS:-10,20,30}"
      SPATIAL_BWS="${SPATIAL_BWS:-0.75,1.0,1.25}"
      TIME_BWS="${TIME_BWS:-365,500,730}"
    fi
    ;;
  full)
    SPATIAL_FEATURE_TYPES="${SPATIAL_FEATURE_TYPES:-scaled_time_adjusted_price,residual,time_adjusted_price,scaled_target_label,raw_price}"
    SPATIAL_STAGES="${SPATIAL_STAGES:-full_X_plus_spatial,base_plus_local_residual}"
    KS="${KS:-10,20,30}"
    SPATIAL_BWS="${SPATIAL_BWS:-0.75,1.0,1.25}"
    TIME_BWS="${TIME_BWS:-365,500,730}"
    ;;
  *)
    echo "ERROR: GRID_PRESET must be focused or full, got: ${GRID_PRESET}" >&2
    exit 2
    ;;
esac

TARGET_SAFE="$(sanitize "${TARGET_VARIANT}")"
LEARNER_SAFE="$(sanitize "${LEARNER}")"
TASK_OUT_DIR="${RESULT_ROOT}/task_${TASK_ID}__target_${TARGET_SAFE}__learner_${LEARNER_SAFE}"
mkdir -p "${TASK_OUT_DIR}"

echo "[mv-spatial] started at $(date)"
echo "[mv-spatial] host=$(hostname) job=${SLURM_JOB_ID:-NA} array_task=${TASK_ID}/${TOTAL_TASKS}"
echo "[mv-spatial] cpus=${CPUS_ON_NODE} n_jobs=${N_JOBS} mem=${SLURM_MEM_PER_NODE:-NA}"
echo "[mv-spatial] python=${PY}"
echo "[mv-spatial] script=${SCRIPT_PATH}"
echo "[mv-spatial] data=${DATA_PATH}"
echo "[mv-spatial] params=${PARAMS_PATH}"
echo "[mv-spatial] run_tag=${RUN_TAG}"
echo "[mv-spatial] out=${TASK_OUT_DIR}"
echo "[mv-spatial] target_variant=${TARGET_VARIANT} learner=${LEARNER}"
echo "[mv-spatial] grid_preset=${GRID_PRESET} resume=${RESUME}"
echo "[mv-spatial] spatial_feature_types=${SPATIAL_FEATURE_TYPES}"
echo "[mv-spatial] spatial_stages=${SPATIAL_STAGES}"
echo "[mv-spatial] Ks=${KS} spatial_bws=${SPATIAL_BWS} time_bws=${TIME_BWS} calibration=${CALIBRATION_MODES}"
echo "[mv-spatial] lgbm_n_jobs=${LGBM_N_JOBS} oof_mode=${OOF_MODE}"

ARGS=(
  "${SCRIPT_PATH}"
  --data-path "${DATA_PATH}"
  --params-path "${PARAMS_PATH}"
  --out-dir "${TASK_OUT_DIR}"
  --parquet-engine "${PARQUET_ENGINE}"
  --sample-frac "${SAMPLE_FRAC}"
  --sample-seed "${SAMPLE_SEED}"
  --assess-eval-year "${ASSESS_EVAL_YEAR}"
  --valuation-date-mode "${VALUATION_DATE_MODE}"
  --target-variants "${TARGET_VARIANT}"
  --learners "${LEARNER}"
  --spatial-feature-types "${SPATIAL_FEATURE_TYPES}"
  --spatial-stages "${SPATIAL_STAGES}"
  --Ks "${KS}"
  --spatial-bandwidths-miles "${SPATIAL_BWS}"
  --time-bandwidths-days "${TIME_BWS}"
  --calibration-modes "${CALIBRATION_MODES}"
  --calib-frac "${CALIB_FRAC}"
  --ridge-alpha "${RIDGE_ALPHA}"
  --n-estimators "${N_ESTIMATORS}"
  --learning-rate "${LEARNING_RATE}"
  --lgbm-n-jobs "${LGBM_N_JOBS}"
  --target-shrink-n "${TARGET_SHRINK_N}"
  --class-filter-col "${CLASS_FILTER_COL}"
  --min-same-class-pool "${MIN_SAME_CLASS_POOL}"
  --max-spatial-candidates "${MAX_SPATIAL_CANDIDATES}"
  --candidate-multiplier "${CANDIDATE_MULTIPLIER}"
  --n-jobs "${N_JOBS}"
  --n-oof-folds "${N_OOF_FOLDS}"
  --oof-mode "${OOF_MODE}"
  --n-deciles "${N_DECILES}"
  --seed "${SAMPLE_SEED}"
)

if [[ -n "${MAX_NEIGHBOR_AGE_DAYS}" ]]; then
  ARGS+=(--max-neighbor-age-days "${MAX_NEIGHBOR_AGE_DAYS}")
fi
if [[ "${ALLOW_SEQUENTIAL_EVAL_HISTORY}" == "1" ]]; then
  ARGS+=(--allow-sequential-eval-history)
fi
if [[ "${NO_ENGINEERED_FEATURES}" == "1" ]]; then
  ARGS+=(--no-engineered-features)
fi
if [[ "${NO_STRICT_FEATURE_SCREEN}" == "1" ]]; then
  ARGS+=(--no-strict-feature-screen)
fi
if [[ "${NO_CLASS_FILTER}" == "1" ]]; then
  ARGS+=(--no-class-filter)
fi
if [[ "${NO_CLASS_FALLBACK}" == "1" ]]; then
  ARGS+=(--no-class-fallback)
fi
if [[ "${RESUME}" == "1" ]]; then
  ARGS+=(--resume)
fi
if [[ "${NO_STREAM_METRICS}" == "1" ]]; then
  ARGS+=(--no-stream-metrics)
fi

# EXTRA_ARGS is intentionally appended last, so you can override/add flags from sbatch:
#   sbatch --export=ALL,EXTRA_ARGS='--n-estimators 300' run_mv_spatial_2025_array.sbatch
if [[ -n "${EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARRAY=( ${EXTRA_ARGS} )
  ARGS+=("${EXTRA_ARRAY[@]}")
fi

run_cmd "${PY}" "${ARGS[@]}"

echo
echo "[mv-spatial] finished at $(date)"
echo "[mv-spatial] results in ${TASK_OUT_DIR}"
