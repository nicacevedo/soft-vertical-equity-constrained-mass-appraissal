#!/bin/bash
# Finish the v3 DAG. Run on a login node, not under sbatch.
#
# Completed and NOT resubmitted: caches, linkage, modeling tables, pre-freeze
# baselines, panel freeze, held-out baselines, Direct. Only the stages that
# failed, were cancelled, or are newly required run here.
#
#   USE_AUDIT (array 0-2) ─┐
#   SURROGATE (array 0-2) ─┼─→ BOOTSTRAP ─→ REPORT
#   PHL_SENSITIVITY ───────┘        └─────→ TESTS
#
# Partition: sched_mit_sloan_batch_r8 (Sloan). Never mit_normal.
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal}"
cd "${REPO_ROOT}"
SLURM_DIR="analysis/berry_attom_validation_v3/slurm"
LOGS="analysis/berry_attom_validation_v3/logs"
mkdir -p "${LOGS}"

FREEZE="analysis/berry_attom_validation_v3/panel_freeze/final_panel_freeze_v3.yaml"
[[ -f "${FREEZE}" ]] || { echo "missing freeze file; refusing to run penalty stages" >&2; exit 1; }

# Roots run in parallel; none depends on another.
AUDIT=$(sbatch --parsable "${SLURM_DIR}/13_property_use_audit.sh")
SURR=$(sbatch --parsable "${SLURM_DIR}/10_surrogate.sh")
SENS=$(sbatch --parsable "${SLURM_DIR}/14_phl_sensitivity.sh")
echo "submitted use_audit ${AUDIT} surrogate_pass2 ${SURR} phl_sensitivity ${SENS}"

# Bootstrap needs the surrogate held-out parquets, and nothing else.
BOOT=$(sbatch --parsable --dependency=afterok:"${SURR}" "${SLURM_DIR}/11_bootstrap.sh")

# Tests and the report must not be held hostage to the two probe jobs, so those
# are afterany. Pass 1 lost the bootstrap and report to a single afterok edge on
# a transient node fault; do not repeat that coupling.
TESTS=$(sbatch --parsable --dependency=afterany:"${BOOT}":"${AUDIT}":"${SENS}" "${SLURM_DIR}/15_tests.sh")
REPORT=$(sbatch --parsable \
  --dependency=afterok:"${BOOT}",afterany:"${AUDIT}":"${SENS}" "${SLURM_DIR}/12_report.sh")
echo "submitted bootstrap ${BOOT} tests ${TESTS} report ${REPORT}"

echo "${AUDIT} ${SURR} ${SENS} ${BOOT} ${TESTS} ${REPORT}" >> "${LOGS}/submitted_job_ids.txt"
{
  echo "# finish-v3 DAG submitted $(date -Is) on sched_mit_sloan_batch_r8"
  echo "use_audit=${AUDIT} array=0-2 cpus=8 mem=80G deps=none"
  echo "surrogate_pass2=${SURR} array=0-2 cpus=16 mem=80G deps=none"
  echo "phl_sensitivity=${SENS} cpus=16 mem=80G deps=none"
  echo "bootstrap=${BOOT} cpus=8 mem=32G deps=afterok:${SURR}"
  echo "tests=${TESTS} cpus=2 mem=16G deps=afterany:${BOOT}:${AUDIT}:${SENS}"
  echo "report=${REPORT} cpus=2 mem=8G deps=afterok:${BOOT},afterany:${AUDIT}:${SENS}"
} >> "${LOGS}/finish_v3_dag.txt"
