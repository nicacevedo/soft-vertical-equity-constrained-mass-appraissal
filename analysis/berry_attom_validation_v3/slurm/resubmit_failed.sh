#!/bin/bash
# Resubmit only the failed v3 stages. Completed caches/tables/Philly prefreeze/STL local are reused.
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal}"
cd "${REPO_ROOT}"
mkdir -p analysis/berry_attom_validation_v3/logs
SLURM_DIR="analysis/berry_attom_validation_v3/slurm"
LINK=$(sbatch --parsable "${SLURM_DIR}/02_link.sh")
echo "submitted link ${LINK}"
# array 0=wayne, 2=st_louis_county; philadelphia prefreeze already completed
PRE=$(sbatch --parsable --array=0,2 "${SLURM_DIR}/04_prefreeze_baselines.sh")
echo "submitted prefreeze ${PRE}"
ROB=$(sbatch --parsable "${SLURM_DIR}/06_stl_robustness.sh")
echo "submitted stl_robustness ${ROB}"
FRZ=$(sbatch --parsable --dependency=afterok:"${LINK}":"${PRE}":"${ROB}" "${SLURM_DIR}/07_freeze.sh")
echo "submitted freeze ${FRZ}"
FIN=$(sbatch --parsable --dependency=afterok:"${FRZ}" "${SLURM_DIR}/08_final_baselines.sh")
echo "submitted final_baselines ${FIN}"
DISP=$(sbatch --parsable --dependency=afterok:"${FRZ}" --export=ALL,FINAL_ID="${FIN}" "${SLURM_DIR}/07b_dispatch.sh")
echo "submitted dispatch ${DISP}"
echo "${LINK} ${PRE} ${ROB} ${FRZ} ${FIN} ${DISP}" | tee -a analysis/berry_attom_validation_v3/logs/submitted_job_ids.txt
