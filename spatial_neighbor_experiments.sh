#!/bin/bash
# -----------------------------------------------------------------------------
# Spatiotemporal neighbor-feature experiments (LinearRegression + LightGBM + CovPenalty)
# Submission DRIVER: fans the grid out into many small single-task jobs.
# -----------------------------------------------------------------------------
# Design of this run set:
#   * Big neighbor grid: k = unique(round(linspace(1, 80, 40))) (~40 values), to
#     trace the evolution of metrics vs. number of neighbors.
#   * Common OOS test set: the most-recent slice of the FULL universe, identical
#     across every filter/experiment (no filtering on the test side). The
#     single_family / deed / arms-length filters are applied to TRAIN ONLY.
#   * Test metrics are averaged over a shared month-block bootstrap (mean + std).
#
# Sharding (keeps every shard <= ~80 (exp,k) tasks, ~2-3h wall):
#   spatial / spatial_nofilter : per (dataset, kernel)            -> 40 tasks
#   feature / trend            : per (dataset, kernel)            -> 80 tasks
#   time                       : per (dataset, kernel, feat-wt)   -> 80 tasks
# Each shard writes a tag-suffixed CSV into RESULT_ROOT, so shards never collide.
#
# LightGBM in fairness_env needs glibc >= 2.27 -> only newer-OS partitions:
#   mit_normal, sched_mit_sloan_batch_r8, ou_sloan_batch (round-robined below).
#
# Usage:
#   bash spatial_neighbor_experiments.sh                 # submit the full run
#   SMOKE=1 bash spatial_neighbor_experiments.sh         # tiny local smoke (no sbatch)
#   AGGREGATE_ONLY=1 RESULT_ROOT=output/neighbor_experiments/<tag> \
#       bash spatial_neighbor_experiments.sh             # aggregate + plots
# -----------------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal}"
PY="${PY:-/home/nacevedo/.conda/envs/fairness_env/bin/python}"
cd "${REPO_ROOT}"
mkdir -p temp/logs

DATA_PATH="${DATA_PATH:-data/CCAO/2025/training_data.parquet}"
PARAMS_PATH="${PARAMS_PATH:-params.yaml}"
MODEL_PARAMS_PATH="${MODEL_PARAMS_PATH:-model_params.yaml}"

RUN_TAG="${RUN_TAG:-spatial_$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-output/neighbor_experiments/${RUN_TAG}}"

# --- Grid knobs (override via env) -------------------------------------------
SMOKE="${SMOKE:-0}"
DATASETS_ALL=(all_filtered arms_length_or_missing deed_01_02 single_family)
KERNELS_ALL=("gaussian") #(gaussian epanechnikov triangular)

MODELS="${MODELS:-linear,lgbm,cov}"
COV_RHO="${COV_RHO:-2.397}"
NJOBS="${NJOBS:-8}"
LGBM_NJOBS="${LGBM_NJOBS:-1}"

# k = unique(round(linspace(1, 80, 40)))
K_VALUES="${K_VALUES:-$(${PY} -c 'import numpy as np;print(",".join(map(str,sorted({int(round(v)) for v in np.linspace(1,80,40)}))))')}"
FEATURE_WEIGHTS_ALL="${FEATURE_WEIGHTS_ALL:-1.0}"   # used by feature/trend; split per-shard for time
TIME_WEIGHTS="${TIME_WEIGHTS:-1.0}"
GEO_WEIGHTS="${GEO_WEIGHTS:-1.0}"
BANDWIDTH_SCALES="${BANDWIDTH_SCALES:-1.0}"

TRAIN_SIZE="${TRAIN_SIZE:-240000}"
TEST_SIZE="${TEST_SIZE:-60000}"
N_BOOTSTRAP="${N_BOOTSTRAP:-5}"
BOOTSTRAP_FREQ="${BOOTSTRAP_FREQ:-M}"

# SLURM resources per shard
PARTITIONS=(mit_normal sched_mit_sloan_batch_r8 ou_sloan_batch)
CPUS="${CPUS:-8}"
MEM="${MEM:-24G}"
WALL="${WALL:-6:00:00}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export PYARROW_NUM_THREADS="${PYARROW_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1

# --- Aggregate-only mode -----------------------------------------------------
if [[ "${AGGREGATE_ONLY:-0}" == "1" ]]; then
  echo "[aggregate] RESULT_ROOT=${RESULT_ROOT}"
  "${PY}" spatial_analysis_plots.py --input-dir "${RESULT_ROOT}"
  exit 0
fi

# Shared python args (everything except dataset/group/kernel/weights/tag).
COMMON_ARGS=(
  --data-path "${DATA_PATH}"
  --params-path "${PARAMS_PATH}"
  --model-params-path "${MODEL_PARAMS_PATH}"
  --models "${MODELS}"
  --cov-rho "${COV_RHO}"
  --k-values "${K_VALUES}"
  --geo-weights "${GEO_WEIGHTS}"
  --time-weights "${TIME_WEIGHTS}"
  --bandwidth-scales "${BANDWIDTH_SCALES}"
  --train-size "${TRAIN_SIZE}"
  --test-size "${TEST_SIZE}"
  --n-bootstrap "${N_BOOTSTRAP}"
  --bootstrap-block-freq "${BOOTSTRAP_FREQ}"
  --n-jobs "${NJOBS}"
  --lgbm-n-jobs "${LGBM_NJOBS}"
  --save-dir "${RESULT_ROOT}"
)

# --- Smoke mode: run ONE tiny shard locally (no sbatch) ----------------------
if [[ "${SMOKE}" == "1" ]]; then
  echo "[smoke] local tiny run -> ${RESULT_ROOT}"
  "${PY}" spatial_analysis.py \
    --data-path "${DATA_PATH}" --params-path "${PARAMS_PATH}" --model-params-path "${MODEL_PARAMS_PATH}" \
    --datasets single_family --groups time --models "${MODELS}" --cov-rho "${COV_RHO}" \
    --k-values "5,20" --kernels gaussian --geo-weights 1.0 \
    --feature-weights 1.0 --time-weights 1.0 --bandwidth-scales 1.0 \
    --train-size 6000 --test-size 3000 --n-bootstrap 3 --bootstrap-block-freq "${BOOTSTRAP_FREQ}" \
    --n-jobs 2 --lgbm-n-jobs 1 --save-dir "${RESULT_ROOT}" --tag "smoke__single_family__time__gaussian"
  echo "[smoke] done -> ${RESULT_ROOT}"
  exit 0
fi

# --- Build the shard list ----------------------------------------------------
# Each entry: "GROUP|KERNEL|FEATURE_WEIGHTS|TAG"
declare -a SHARDS=()
build_shards_for_dataset() {
  local ds="$1"
  local kern
  for kern in "${KERNELS_ALL[@]}"; do
    SHARDS+=("${ds}|spatial|${FEATURE_WEIGHTS_ALL}|${ds}__spatial__${kern}|${kern}")
    SHARDS+=("${ds}|spatial_nofilter|${FEATURE_WEIGHTS_ALL}|${ds}__spatial_nofilter__${kern}|${kern}")
    SHARDS+=("${ds}|feature|${FEATURE_WEIGHTS_ALL}|${ds}__feature__${kern}|${kern}")
    SHARDS+=("${ds}|trend|${FEATURE_WEIGHTS_ALL}|${ds}__trend__${kern}|${kern}")
    local fw
    for fw in ${FEATURE_WEIGHTS_ALL//,/ }; do
      SHARDS+=("${ds}|time|${fw}|${ds}__time__${kern}__f${fw}|${kern}")
    done
  done
}

DATASETS_RUN=("${DATASETS_ALL[@]}")
if [[ -n "${DATASETS:-}" ]]; then IFS=',' read -r -a DATASETS_RUN <<< "${DATASETS}"; fi
for ds in "${DATASETS_RUN[@]}"; do build_shards_for_dataset "${ds}"; done

echo "[submit] RESULT_ROOT=${RESULT_ROOT}"
echo "[submit] shards=${#SHARDS[@]} k=${K_VALUES}"

# --- Submit each shard as its own single-task job, round-robin partitions -----
JOB_IDS=()
i=0
for entry in "${SHARDS[@]}"; do
  IFS='|' read -r DS GROUP FW TAG KERN <<< "${entry}"
  PART="${PARTITIONS[$(( i % ${#PARTITIONS[@]} ))]}"
  i=$(( i + 1 ))

  CMD="${PY} spatial_analysis.py ${COMMON_ARGS[*]} --datasets ${DS} --groups ${GROUP} --kernels ${KERN} --feature-weights ${FW} --tag ${TAG}"
  JID=$(sbatch --parsable \
    -p "${PART}" -t "${WALL}" --nodes=1 --ntasks=1 --cpus-per-task="${CPUS}" --mem="${MEM}" \
    --job-name="snbr_${TAG}" \
    --output="temp/logs/snbr_%j_${TAG}.out" --error="temp/logs/snbr_%j_${TAG}.err" \
    --wrap "cd ${REPO_ROOT} && ${CMD}")
  JOB_IDS+=("${JID}")
  echo "[submit] ${TAG} -> job ${JID} on ${PART}"
done

# --- Aggregation job: runs after all shards finish (success or fail) ----------
DEP=$(IFS=:; echo "${JOB_IDS[*]}")
AGG_JID=$(sbatch --parsable \
  -p mit_normal -t 30:00 --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G \
  --dependency="afterany:${DEP}" --job-name="snbr_agg" \
  --output="temp/logs/snbr_agg_%j.out" --error="temp/logs/snbr_agg_%j.err" \
  --wrap "cd ${REPO_ROOT} && ${PY} spatial_analysis_plots.py --input-dir ${RESULT_ROOT}")
echo "[submit] aggregation -> job ${AGG_JID} (afterany ${#JOB_IDS[@]} shards)"
echo "[submit] RESULT_ROOT=${RESULT_ROOT}"
