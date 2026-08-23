#!/bin/bash
# =============================================================================
# rho_sweep_experiments.sh
#
# SLURM job-array driver for the covariance-penalty rho-sweep experiments.
# Each array task runs ONE independent experiment = (dataset, LGBM baseline
# config) on its own node, fully in parallel with the others.
#
# For every experiment the script runs quick_test_models.py with:
#   - models = linear, lgbm (baseline), cov (rho sweep)
#   - a log-spaced rho sweep over [RHO_MIN, RHO_MAX] with RHO_COUNT points
# and writes a self-contained result folder (its own plots/ + metric tables)
# whose name encodes the data source, the held-out assessment year, the LGBM
# config key, and an 8-char config id, e.g.:
#   output/rho_sweep/ccao2025_assess2025__test_best_r2_dee08fa9
#
# Experiment matrix (DATASETS x CONFIGS):
#   datasets (4): held-out assessment years 2025 / 2024 / 2023 / 2022, all on the
#                 same 2025 feature schema (the real 2022/2023 vintages use a
#                 smaller, incompatible schema and are intentionally excluded).
#   configs  (3): CV-validated LGBM baselines (distinct hyperparameter sets).
#   => 12 experiments, array indices 0..11.
#
# Usage:
#   sbatch scripts/rho_sweep_experiments.sh            # submit all 12 (array 0-11)
#   sbatch --array=0-2 scripts/rho_sweep_experiments.sh  # submit a subset
#   bash   scripts/rho_sweep_experiments.sh list       # print the matrix, no run
#   QUICK_TEST_TASK_ID=0 bash scripts/rho_sweep_experiments.sh  # run one locally
# =============================================================================
#SBATCH --job-name=rho_sweep
#SBATCH --partition=mit_normal # ou_sloan_batch # 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96GB
#SBATCH --array=0-11
#SBATCH --output=temp/logs/rho_sweep_%A_%a.out
#SBATCH --error=temp/logs/rho_sweep_%A_%a.err
#SBATCH -t 0-08:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nacevedo@mit.edu

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal}"
PY="${PY:-/home/nacevedo/.conda/envs/fairness_env/bin/python}"
cd "${REPO_ROOT}"

# --- Sweep / model knobs (match the reference command; override via env) -----
RHO_MIN="${RHO_MIN:-1e-1}"
RHO_MAX="${RHO_MAX:-20}"
RHO_COUNT="${RHO_COUNT:-50}"
RHO_SCALE="${RHO_SCALE:-log}"
# Extra explicit rho points merged into the geometric grid (e.g. the recommended
# operating point) so they are evaluated exactly rather than via the nearest grid node.
# Include the current theory-knee/reference region exactly. Override this env var
# after re-estimating the theory bands if a new run recommends different nodes.
RHO_EXTRA="${RHO_EXTRA:-1.62,2.397,2.807,3.01,4.861}"
LGBM_N_ESTIMATORS="${LGBM_N_ESTIMATORS:-500}"
SAMPLE_FRAC="${SAMPLE_FRAC:-1}"
SEED="${SEED:-4050}"
N_BOOTSTRAP_VALIDATION="${N_BOOTSTRAP_VALIDATION:-5}"
BOOTSTRAP_BLOCK_FREQ="${BOOTSTRAP_BLOCK_FREQ:-M}"
SCATTER_MAX="${SCATTER_MAX:-10000}"
OUT_ROOT="${OUT_ROOT:-output/rho_sweep_500_estimators}"
HYPERPARAM_FILE="${HYPERPARAM_FILE:-best_lgbm_baseline_configs.yaml}"

# --- Experiment matrix -------------------------------------------------------
# Datasets (index d): same feature schema, different held-out assessment year.
SRC_LABELS=(ccao2025 ccao_old ccao_sim2024 ccao_sim2023)
SRC_PATHS=(
  "data/CCAO/2025/training_data.parquet"
  "data/CCAO/2025/training_data_old.parquet"
  "data/CCAO/2025/training_data_sim2024.parquet"
  "data/CCAO/2025/training_data_sim2023.parquet"
)
SRC_ASSESS_YEARS=(2025 2024 2023 2022)

# LGBM baseline configs (index c): distinct CV-validated hyperparameter sets.
CFG_KEYS=(test_best_r2 cv_top1_r2 cv_top2_r2)
CFG_IDS=(dee08fa9 c6fc2c3b a1c87203)

N_DATASETS=${#SRC_LABELS[@]}
N_CONFIGS=${#CFG_KEYS[@]}
N_TOTAL=$(( N_DATASETS * N_CONFIGS ))

derive() {  # $1 = task id -> sets global EXP_* variables
  local tid="$1"
  local d=$(( tid / N_CONFIGS ))
  local c=$(( tid % N_CONFIGS ))
  EXP_SRC="${SRC_LABELS[$d]}"
  EXP_PATH="${SRC_PATHS[$d]}"
  EXP_YEAR="${SRC_ASSESS_YEARS[$d]}"
  EXP_KEY="${CFG_KEYS[$c]}"
  EXP_CID="${CFG_IDS[$c]}"
  EXP_OUT="${OUT_ROOT}/${EXP_SRC}_assess${EXP_YEAR}__${EXP_KEY}_${EXP_CID}"
}

if [[ "${1:-}" == "list" ]]; then
  echo "rho-sweep experiment matrix (${N_TOTAL} tasks):"
  for ((t=0; t<N_TOTAL; t++)); do
    derive "$t"
    printf "  [%2d] src=%-13s assess=%s  cfg=%-13s -> %s\n" "$t" "$EXP_SRC" "$EXP_YEAR" "$EXP_KEY" "$EXP_OUT"
  done
  exit 0
fi

# Resolve which task to run: SLURM array id, explicit env, or first CLI arg.
TASK_ID="${SLURM_ARRAY_TASK_ID:-${QUICK_TEST_TASK_ID:-${1:-}}}"
if [[ -z "${TASK_ID}" ]]; then
  echo "ERROR: no task id. Run under sbatch (--array) or pass QUICK_TEST_TASK_ID / a CLI index." >&2
  exit 2
fi
if (( TASK_ID < 0 || TASK_ID >= N_TOTAL )); then
  echo "ERROR: task id ${TASK_ID} out of range [0, $((N_TOTAL-1))]." >&2
  exit 2
fi

# --- Thread hygiene: one OpenMP/BLAS thread per process worker ---------------
# quick_test_models.py runs the model fits concurrently with a thread pool sized
# to the CPU affinity mask; pinning the math libs to 1 thread each prevents
# OpenMP oversubscription / allocation failures on the allocated cores.
CPUS="${SLURM_CPUS_PER_TASK:-$(nproc)}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export LGBM_NUM_THREADS=1
export QUICK_TEST_MAX_WORKERS="${CPUS}"

mkdir -p temp/logs "${OUT_ROOT}"
derive "${TASK_ID}"

echo "==============================================================="
echo "[rho_sweep] task=${TASK_ID}/${N_TOTAL} started $(date)"
echo "[rho_sweep] host=$(hostname) cpus=${CPUS} python=${PY}"
echo "[rho_sweep] source=${EXP_SRC} data=${EXP_PATH} assess_year=${EXP_YEAR}"
echo "[rho_sweep] lgbm_config=${EXP_KEY} (id=${EXP_CID}) n_estimators=${LGBM_N_ESTIMATORS}"
echo "[rho_sweep] rho=[${RHO_MIN},${RHO_MAX}] count=${RHO_COUNT} scale=${RHO_SCALE}"
echo "[rho_sweep] out_dir=${EXP_OUT}"
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
"${PY}" quick_test_models.py \
  --rho-range "${RHO_MIN},${RHO_MAX}" \
  --rho-count "${RHO_COUNT}" \
  --rho-scale "${RHO_SCALE}" \
  --rho-extra "${RHO_EXTRA}" \
  --lgbm-hyperparameter-file "${HYPERPARAM_FILE}" \
  --lgbm-config-key "${EXP_KEY}" \
  --lgbm-n-jobs 1 \
  --lgbm-n-estimators "${LGBM_N_ESTIMATORS}" \
  --models linear,lgbm,cov \
  --out-dir "${EXP_OUT}" \
  --data-path "${EXP_PATH}" \
  --assessment-year "${EXP_YEAR}" \
  --parallel-models \
  --sample-frac "${SAMPLE_FRAC}" \
  --seed "${SEED}" \
  --n-bootstrap-validation "${N_BOOTSTRAP_VALIDATION}" \
  --bootstrap-block-freq "${BOOTSTRAP_BLOCK_FREQ}" \
  --scatter-plot-max-samples "${SCATTER_MAX}"

ec=$?
echo "[rho_sweep] task=${TASK_ID} (${EXP_SRC}/${EXP_KEY}) finished $(date) exit=${ec}"
exit ${ec}
