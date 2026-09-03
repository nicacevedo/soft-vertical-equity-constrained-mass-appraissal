#!/bin/bash
# After freeze exists: submit final baselines always; Direct/Surrogate only if authorized.
#SBATCH --job-name=v3_dispatch
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=0:20:00
#SBATCH --output=analysis/berry_attom_validation_v3/logs/dispatch_%j.out
#SBATCH --error=analysis/berry_attom_validation_v3/logs/dispatch_%j.err
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal}"
PY="${PY:-/home/nacevedo/.conda/envs/fairness_env/bin/python}"
cd "${REPO_ROOT}"
FINAL_ID="${FINAL_ID:-}"
if [[ -z "${FINAL_ID}" ]]; then
  echo "FINAL_ID env var required" >&2
  exit 1
fi
FREEZE="analysis/berry_attom_validation_v3/panel_freeze/final_panel_freeze_v3.yaml"
if [[ ! -f "${FREEZE}" ]]; then
  echo "missing freeze file" >&2
  exit 1
fi
AUTH=$("${PY}" -c "import yaml; print(bool(yaml.safe_load(open('${FREEZE}'))['direct_surrogate_authorized']))")
echo "authorized=${AUTH} final_baselines_job=${FINAL_ID}"
SLURM_DIR="analysis/berry_attom_validation_v3/slurm"
if [[ "${AUTH}" == "True" ]]; then
  D=$(sbatch --parsable --dependency=afterok:"${FINAL_ID}" "${SLURM_DIR}/09_direct.sh")
  S=$(sbatch --parsable --dependency=afterok:"${FINAL_ID}" "${SLURM_DIR}/10_surrogate.sh")
  echo "submitted direct ${D} surrogate ${S}"
  B=$(sbatch --parsable --dependency=afterok:"${D}":"${S}" "${SLURM_DIR}/11_bootstrap.sh")
else
  echo "Direct/Surrogate NOT submitted."
  B=$(sbatch --parsable --dependency=afterok:"${FINAL_ID}" "${SLURM_DIR}/11_bootstrap.sh")
fi
R=$(sbatch --parsable --dependency=afterok:"${B}" "${SLURM_DIR}/12_report.sh")
echo "submitted bootstrap ${B} report ${R}"
echo "${D:-na} ${S:-na} ${B} ${R}" >> analysis/berry_attom_validation_v3/logs/submitted_job_ids.txt
