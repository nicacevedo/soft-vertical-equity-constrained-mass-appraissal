#!/bin/bash
# Build the six-county penalty dashboard only from completed schema-v3 county runs.
# Submit after the county array with, for example:
#   sbatch --dependency=afterok:<array_job_id> county_penalty_dashboard.sh
#SBATCH --job-name=county_dashboard
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=temp/logs/county_dashboard_%j.out
#SBATCH --error=temp/logs/county_dashboard_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal}"
PY="${PY:-/home/nacevedo/.conda/envs/fairness_env/bin/python}"
cd "${REPO_ROOT}"
mkdir -p temp/logs

RUNS=(
  output/county_bench_17031_floor50000
  output/county_bench_42003_floor50000
  output/county_bench_04013_floor50000
  output/county_bench_53033_floor50000
  output/county_bench_12086_floor50000
  output/county_bench_25017_floor50000
)

# The dashboard's schema-v3 loader is the single authoritative preflight.  It
# verifies the complete manifest, transformer provenance, paired bootstrap
# endpoints, and local-equity files before reading/rendering any result.

args=()
for run in "${RUNS[@]}"; do
  args+=(--run "${run}")
done
"${PY}" scripts/build_county_penalty_dashboard.py "${args[@]}" \
  --out output/county_penalty/county_penalty_report.html
