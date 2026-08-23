#!/bin/bash
# -----------------------------------------------------------------------------
# Broad vs. strict sale-validation cohort on Cook County.
#
# The two cohorts differ only in how undocumented / unmapped validation codes
# are treated. Broad fails open on codes the dictionary does not decide; strict
# additionally demands the documented transfer-accuracy and deed whitelists.
# Strict is not portable: the `135` transfer-accuracy code covers 0.3% of Cook
# but 49% of Allegheny, and the two-code deed whitelist ranges from 0.8%
# (Middlesex MA) to 65% (Mecklenburg NC) -- hence broad for cross-county work.
#
# No --property-use-codes here on purpose: this exercises the default, which
# now keeps every observed property-use code.
#
# Usage:
#   sbatch attom_cohort_benchmark.sh
# -----------------------------------------------------------------------------
#SBATCH --job-name=attom_cohort
#SBATCH --array=0-1
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=8:00:00
#SBATCH --output=temp/logs/attom_cohort_%A_%a.out
#SBATCH --error=temp/logs/attom_cohort_%A_%a.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal}"
PY="${PY:-/home/nacevedo/.conda/envs/fairness_env/bin/python}"
cd "${REPO_ROOT}"
mkdir -p temp/logs

COHORTS=(broad strict)
IDX="${SLURM_ARRAY_TASK_ID:-0}"
COHORT="${COHORTS[${IDX}]}"

CPUS="${SLURM_CPUS_PER_TASK:-16}"
OUT_DIR="output/attom_recorder_sample_${COHORT}"

export OMP_NUM_THREADS="${CPUS}"
export PYTHONUNBUFFERED=1

echo "host=$(hostname) cohort=${COHORT} cpus=${CPUS} start=$(date -Is)"

"${PY}" scripts/other_counties_benchmars.py \
  --sale-cohort "${COHORT}" \
  --lgbm-threads "${CPUS}" \
  --output-dir "${OUT_DIR}"

echo "done cohort=${COHORT} -> ${OUT_DIR} end=$(date -Is)"
