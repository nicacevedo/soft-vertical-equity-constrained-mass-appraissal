#!/bin/bash
# -----------------------------------------------------------------------------
# ATTOM cross-county benchmark: Cook (17031), Allegheny (42003), Maricopa
# (04013), King (53033), Miami-Dade (12086), and Middlesex (25017), two price
# floors each. Verifies that
# scripts/other_counties_benchmars.py is portable across counties with no code
# changes -- only --county-fips and --assessor-dir.
#
# Array layout (12 tasks): task % 6 selects the county, task / 6 selects the floor.
#   0 -> 17031 floor 10000     6 -> 17031 floor 50000
#   1 -> 42003 floor 10000     7 -> 42003 floor 50000
#   2 -> 04013 floor 10000     8 -> 04013 floor 50000
#   3 -> 53033 floor 10000     9 -> 53033 floor 50000
#   4 -> 12086 floor 10000    10 -> 12086 floor 50000
#   5 -> 25017 floor 10000    11 -> 25017 floor 50000
#
# The $50k floor is the recommended setting: the bottom price decile carries
# almost all of the error, and dropping it also stabilizes target-scale
# selection (Allegheny picks `raw` at $10k and `log` at $50k).
#
# Usage:
#   sbatch attom_county_benchmark.sh
# -----------------------------------------------------------------------------
#SBATCH --job-name=attom_county
#SBATCH --array=0-11
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=90G
#SBATCH --time=8:00:00
#SBATCH --output=temp/logs/attom_county_%A_%a.out
#SBATCH --error=temp/logs/attom_county_%A_%a.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal}"
PY="${PY:-/home/nacevedo/.conda/envs/fairness_env/bin/python}"
cd "${REPO_ROOT}"
mkdir -p temp/logs

FIPS_LIST=(17031 42003 04013 53033 12086 25017)
FLOORS=(10000 50000)
IDX="${SLURM_ARRAY_TASK_ID:-0}"
FIPS="${FIPS_LIST[$(( IDX % 6 ))]}"
FLOOR="${FLOORS[$(( IDX / 6 ))]}"

# Every county but Cook lives in its own extract; Cook is the script default.
# Point Allegheny at biggest-10-counties-in-the-us-20216-2025 instead to avoid
# the one corrupt shard in the county-specific folder.
case "${FIPS}" in
  17031) ASSESSOR_DIR="data/dewey-downloads/cookcounty-2016-2025-all-features" ;;
  42003) ASSESSOR_DIR="data/dewey-downloads/allegheny-county-2016-2025-all-features" ;;
  04013) ASSESSOR_DIR="data/dewey-downloads/maricopa-county-2016-2025-all-features" ;;
  53033) ASSESSOR_DIR="data/dewey-downloads/king-county-2015-2025-all-features" ;;
  12086) ASSESSOR_DIR="data/dewey-downloads/miami-dade-county-2006-2025-all-features" ;;
  25017) ASSESSOR_DIR="data/dewey-downloads/middlesex-county-2006-2025-all-features" ;;
  *) echo "unknown fips ${FIPS}" >&2; exit 1 ;;
esac

CPUS="${SLURM_CPUS_PER_TASK:-32}"
OUT_DIR="${OUT_DIR:-output/county_bench_${FIPS}_floor${FLOOR}}"

export OMP_NUM_THREADS="${CPUS}"
export PYTHONUNBUFFERED=1

echo "host=$(hostname) fips=${FIPS} floor=${FLOOR} cpus=${CPUS} start=$(date -Is)"

"${PY}" scripts/other_counties_benchmars.py \
  --county-fips "${FIPS}" \
  --assessor-dir "${ASSESSOR_DIR}" \
  --property-use-codes 385 \
  --sale-cohort broad \
  --minimum-sale-price "${FLOOR}" \
  --lgbm-threads "${CPUS}" \
  --output-dir "${OUT_DIR}"

echo "done fips=${FIPS} floor=${FLOOR} -> ${OUT_DIR} end=$(date -Is)"
