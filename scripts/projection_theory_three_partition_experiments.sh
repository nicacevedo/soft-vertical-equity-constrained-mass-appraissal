#!/bin/bash
# =============================================================================
# projection_theory_three_partition_experiments.sh
#
# Three-partition submission driver for the rho-projection manuscript experiments.
# This mirrors the spatial neighbor driver: independent shards are submitted as
# single-task jobs and round-robined across:
#   mit_normal, sched_mit_sloan_batch_r8, ou_sloan_batch
#
# Submitted DAG:
#   1. exact linear projection-path tasks       (4 jobs, independent)
#   2. retrained LGBCovPenalty rho sweeps       (12 jobs, independent)
#   3. theory-vs-empirical LGBM tasks           (12 jobs, each after matching rho)
#   4. theory merge                             (after all theory tasks)
#   5. combined linear + LGBM manuscript report (after linear + theory merge)
#
# Usage:
#   bash scripts/projection_theory_three_partition_experiments.sh list
#   bash scripts/projection_theory_three_partition_experiments.sh
#
# Useful overrides:
#   RUN_TAG=projection_theory_YYYYMMDD_HHMMSS
#   Q_GRID=1.00,0.90,0.80,0.70,0.60,0.50,0.40,0.30,0.20
#   RHO_MIN=1e-1 RHO_MAX=20 RHO_COUNT=50 RHO_SCALE=log
# =============================================================================

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal}"
PY="${PY:-/home/nacevedo/.conda/envs/fairness_env/bin/python}"
cd "${REPO_ROOT}"
mkdir -p temp/logs

PARTITIONS=(mit_normal sched_mit_sloan_batch_r8 ou_sloan_batch)
CPUS="${CPUS:-32}"
MEM="${MEM:-96GB}"
LINEAR_WALL="${LINEAR_WALL:-0-04:00:00}"
RHO_WALL="${RHO_WALL:-0-08:00:00}"
THEORY_WALL="${THEORY_WALL:-0-04:00:00}"
MERGE_WALL="${MERGE_WALL:-0-01:00:00}"

RUN_TAG="${RUN_TAG:-projection_theory_$(date +%Y%m%d_%H%M%S)}"
LINEAR_OUT_ROOT="${LINEAR_OUT_ROOT:-output/projection_linear/${RUN_TAG}}"
RHO_OUT_ROOT="${RHO_OUT_ROOT:-output/rho_sweep_500_estimators/${RUN_TAG}}"
THEORY_OUT_ROOT="${THEORY_OUT_ROOT:-output/theory_rho_range_500_estimators/${RUN_TAG}}"
COLLECT_OUT_DIR="${COLLECT_OUT_DIR:-output/projection_theory_comparison/${RUN_TAG}}"

N_LINEAR=4
N_RHO=12

partition_for_index() {
  local idx="$1"
  echo "${PARTITIONS[$(( idx % ${#PARTITIONS[@]} ))]}"
}

print_matrix() {
  echo "projection-theory run tag: ${RUN_TAG}"
  echo "partitions: ${PARTITIONS[*]}"
  echo "linear out root: ${LINEAR_OUT_ROOT}"
  echo "rho out root: ${RHO_OUT_ROOT}"
  echo "theory out root: ${THEORY_OUT_ROOT}"
  echo "combined out dir: ${COLLECT_OUT_DIR}"
  echo
  echo "linear tasks:"
  OUT_ROOT="${LINEAR_OUT_ROOT}" bash scripts/projection_linear_experiments.sh list
  echo
  echo "rho-sweep tasks:"
  OUT_ROOT="${RHO_OUT_ROOT}" bash scripts/rho_sweep_experiments.sh list
  echo
  echo "theory tasks:"
  OUT_ROOT="${THEORY_OUT_ROOT}" RHO_SWEEP_ROOT="${RHO_OUT_ROOT}" bash scripts/theory_rho_range_experiments.sh list
}

if [[ "${1:-}" == "list" ]]; then
  print_matrix
  exit 0
fi

if [[ ! -x "${PY}" ]]; then
  echo "ERROR: Python executable not available: ${PY}" >&2
  exit 127
fi
if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found; run this on the cluster login node." >&2
  exit 127
fi

echo "[submit] projection-theory run tag=${RUN_TAG}"
echo "[submit] outputs:"
echo "  linear:  ${LINEAR_OUT_ROOT}"
echo "  rho:     ${RHO_OUT_ROOT}"
echo "  theory:  ${THEORY_OUT_ROOT}"
echo "  collect: ${COLLECT_OUT_DIR}"

LINEAR_JOB_IDS=()
RHO_JOB_IDS=()
THEORY_JOB_IDS=()

for tid in $(seq 0 $((N_LINEAR - 1))); do
  part="$(partition_for_index "${tid}")"
  jid="$(sbatch --parsable \
    -p "${part}" -t "${LINEAR_WALL}" --nodes=1 --ntasks=1 --cpus-per-task="${CPUS}" --mem="${MEM}" \
    --job-name="projlin_${tid}_${RUN_TAG}" \
    --output="temp/logs/projlin_%j_${tid}_${RUN_TAG}.out" \
    --error="temp/logs/projlin_%j_${tid}_${RUN_TAG}.err" \
    --wrap "cd ${REPO_ROOT} && OUT_ROOT='${LINEAR_OUT_ROOT}' PROJECTION_LINEAR_TASK_ID='${tid}' PY='${PY}' bash scripts/projection_linear_experiments.sh")"
  LINEAR_JOB_IDS+=("${jid}")
  echo "[submit] linear task ${tid} -> job ${jid} on ${part}"
done

for tid in $(seq 0 $((N_RHO - 1))); do
  part="$(partition_for_index "${tid}")"
  jid="$(sbatch --parsable \
    -p "${part}" -t "${RHO_WALL}" --nodes=1 --ntasks=1 --cpus-per-task="${CPUS}" --mem="${MEM}" \
    --job-name="rhosweep_${tid}_${RUN_TAG}" \
    --output="temp/logs/rhosweep_%j_${tid}_${RUN_TAG}.out" \
    --error="temp/logs/rhosweep_%j_${tid}_${RUN_TAG}.err" \
    --wrap "cd ${REPO_ROOT} && OUT_ROOT='${RHO_OUT_ROOT}' QUICK_TEST_TASK_ID='${tid}' PY='${PY}' bash scripts/rho_sweep_experiments.sh")"
  RHO_JOB_IDS+=("${jid}")
  echo "[submit] rho task ${tid} -> job ${jid} on ${part}"
done

for tid in $(seq 0 $((N_RHO - 1))); do
  part="$(partition_for_index "$((tid + N_RHO))")"
  dep="${RHO_JOB_IDS[$tid]}"
  jid="$(sbatch --parsable \
    -p "${part}" -t "${THEORY_WALL}" --nodes=1 --ntasks=1 --cpus-per-task="${CPUS}" --mem="${MEM}" \
    --dependency="afterok:${dep}" \
    --job-name="theoryrho_${tid}_${RUN_TAG}" \
    --output="temp/logs/theoryrho_%j_${tid}_${RUN_TAG}.out" \
    --error="temp/logs/theoryrho_%j_${tid}_${RUN_TAG}.err" \
    --wrap "cd ${REPO_ROOT} && OUT_ROOT='${THEORY_OUT_ROOT}' RHO_SWEEP_ROOT='${RHO_OUT_ROOT}' THEORY_TASK_ID='${tid}' PY='${PY}' bash scripts/theory_rho_range_experiments.sh")"
  THEORY_JOB_IDS+=("${jid}")
  echo "[submit] theory task ${tid} -> job ${jid} on ${part} afterok:${dep}"
done

THEORY_DEP="$(IFS=:; echo "${THEORY_JOB_IDS[*]}")"
MERGE_JOB_ID="$(sbatch --parsable \
  -p "${PARTITIONS[0]}" -t "${MERGE_WALL}" --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32GB \
  --dependency="afterok:${THEORY_DEP}" \
  --job-name="theorymerge_${RUN_TAG}" \
  --output="temp/logs/theorymerge_%j_${RUN_TAG}.out" \
  --error="temp/logs/theorymerge_%j_${RUN_TAG}.err" \
  --wrap "cd ${REPO_ROOT} && OUT_ROOT='${THEORY_OUT_ROOT}' RHO_SWEEP_ROOT='${RHO_OUT_ROOT}' PY='${PY}' bash scripts/theory_rho_range_experiments.sh merge")"
echo "[submit] theory merge -> job ${MERGE_JOB_ID} afterok ${#THEORY_JOB_IDS[@]} theory jobs"

LINEAR_DEP="$(IFS=:; echo "${LINEAR_JOB_IDS[*]}")"
COLLECT_DEP="${MERGE_JOB_ID}:${LINEAR_DEP}"
COLLECT_JOB_ID="$(sbatch --parsable \
  -p "${PARTITIONS[0]}" -t "${MERGE_WALL}" --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32GB \
  --dependency="afterok:${COLLECT_DEP}" \
  --job-name="projcollect_${RUN_TAG}" \
  --output="temp/logs/projcollect_%j_${RUN_TAG}.out" \
  --error="temp/logs/projcollect_%j_${RUN_TAG}.err" \
  --wrap "cd ${REPO_ROOT} && '${PY}' scripts/projection_theory_collect.py --linear-root '${LINEAR_OUT_ROOT}' --lgbm-theory-root '${THEORY_OUT_ROOT}' --out-dir '${COLLECT_OUT_DIR}'")"
echo "[submit] combined collector -> job ${COLLECT_JOB_ID} afterok merge+linear"

MANIFEST="output/projection_theory_comparison/${RUN_TAG}_job_manifest.txt"
mkdir -p "$(dirname "${MANIFEST}")"
{
  echo "run_tag=${RUN_TAG}"
  echo "linear_out_root=${LINEAR_OUT_ROOT}"
  echo "rho_out_root=${RHO_OUT_ROOT}"
  echo "theory_out_root=${THEORY_OUT_ROOT}"
  echo "collect_out_dir=${COLLECT_OUT_DIR}"
  echo "linear_jobs=${LINEAR_JOB_IDS[*]}"
  echo "rho_jobs=${RHO_JOB_IDS[*]}"
  echo "theory_jobs=${THEORY_JOB_IDS[*]}"
  echo "merge_job=${MERGE_JOB_ID}"
  echo "collector_job=${COLLECT_JOB_ID}"
} > "${MANIFEST}"
echo "[submit] manifest -> ${MANIFEST}"
