#!/bin/bash
# Submit the v3 Sloan DAG with afterok dependencies.
# Direct/Surrogate are submitted only after freeze authorizes them (07b_dispatch).
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal}"
cd "${REPO_ROOT}"
mkdir -p analysis/berry_attom_validation_v3/logs
SLURM_DIR="analysis/berry_attom_validation_v3/slurm"
PRE=$(sbatch --parsable "${SLURM_DIR}/00_precheck.sh")
echo "submitted precheck ${PRE}"
CACHE=$(sbatch --parsable --dependency=afterok:"${PRE}" "${SLURM_DIR}/01_cache.sh")
echo "submitted cache ${CACHE}"
LINK=$(sbatch --parsable --dependency=afterok:"${CACHE}" "${SLURM_DIR}/02_link.sh")
echo "submitted link ${LINK}"
TAB=$(sbatch --parsable --dependency=afterok:"${CACHE}" "${SLURM_DIR}/03_tables.sh")
echo "submitted tables ${TAB}"
BASE=$(sbatch --parsable --dependency=afterok:"${TAB}" "${SLURM_DIR}/04_prefreeze_baselines.sh")
echo "submitted prefreeze ${BASE}"
STL=$(sbatch --parsable --dependency=afterok:"${PRE}" "${SLURM_DIR}/05_stl_local.sh")
echo "submitted stl_local ${STL}"
ROB=$(sbatch --parsable --dependency=afterok:"${STL}":"${TAB}" "${SLURM_DIR}/06_stl_robustness.sh")
echo "submitted stl_robustness ${ROB}"
FRZ=$(sbatch --parsable --dependency=afterok:"${LINK}":"${BASE}":"${ROB}" "${SLURM_DIR}/07_freeze.sh")
echo "submitted freeze ${FRZ}"
FIN=$(sbatch --parsable --dependency=afterok:"${FRZ}" "${SLURM_DIR}/08_final_baselines.sh")
echo "submitted final_baselines ${FIN}"
DISP=$(sbatch --parsable --dependency=afterok:"${FRZ}" --export=ALL,FINAL_ID="${FIN}" "${SLURM_DIR}/07b_dispatch.sh")
echo "submitted dispatch ${DISP}"
echo "Direct/Surrogate NOT submitted here. Dispatch inspects panel_freeze/final_panel_freeze_v3.yaml."
echo "${PRE} ${CACHE} ${LINK} ${TAB} ${BASE} ${STL} ${ROB} ${FRZ} ${FIN} ${DISP}" > analysis/berry_attom_validation_v3/logs/submitted_job_ids.txt
cat analysis/berry_attom_validation_v3/logs/submitted_job_ids.txt
