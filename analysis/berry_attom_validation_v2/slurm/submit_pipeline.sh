#!/bin/bash
# Submit the v2 pipeline with afterok dependencies.
# Does NOT submit Direct/Surrogate. Inspect the freeze file first.
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal}"
cd "${REPO_ROOT}"
mkdir -p analysis/berry_attom_validation_v2/logs
CACHE_ID=$(sbatch --parsable analysis/berry_attom_validation_v2/slurm/01_cache.sh)
echo "submitted cache ${CACHE_ID}"
LINK_ID=$(sbatch --parsable --dependency=afterok:${CACHE_ID} analysis/berry_attom_validation_v2/slurm/02_link.sh)
echo "submitted link ${LINK_ID}"
TAB_ID=$(sbatch --parsable --dependency=afterok:${CACHE_ID} analysis/berry_attom_validation_v2/slurm/03_tables.sh)
echo "submitted tables ${TAB_ID}"
BASE_ID=$(sbatch --parsable --dependency=afterok:${TAB_ID} analysis/berry_attom_validation_v2/slurm/04_baselines.sh)
echo "submitted baselines ${BASE_ID}"
STL_ID=$(sbatch --parsable analysis/berry_attom_validation_v2/slurm/05_stl_local.sh)
echo "submitted stl_local ${STL_ID}"
ROB_ID=$(sbatch --parsable --dependency=afterok:${STL_ID}:${TAB_ID} analysis/berry_attom_validation_v2/slurm/06_stl_robustness.sh)
echo "submitted stl_robustness ${ROB_ID}"
FRZ_ID=$(sbatch --parsable --dependency=afterok:${LINK_ID}:${BASE_ID}:${ROB_ID} analysis/berry_attom_validation_v2/slurm/07_freeze.sh)
echo "submitted freeze ${FRZ_ID}"
echo "Direct/Surrogate NOT submitted. After freeze, inspect panel_freeze/final_panel_freeze_v2.yaml."
echo "${CACHE_ID} ${LINK_ID} ${TAB_ID} ${BASE_ID} ${STL_ID} ${ROB_ID} ${FRZ_ID}" > analysis/berry_attom_validation_v2/logs/submitted_job_ids.txt
