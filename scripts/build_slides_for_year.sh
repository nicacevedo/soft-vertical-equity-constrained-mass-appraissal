#!/bin/bash
# End-to-end driver: given a year tag (2023 or 2024), produce figures + summary
# JSON from the rolling-origin CV output and apply them to the slides file.
#
# Example:
#   bash scripts/build_slides_for_year.sh 2023
#   bash scripts/build_slides_for_year.sh 2024
set -euo pipefail

YEAR="${1:?usage: $0 <2023|2024>}"
case "${YEAR}" in
  2023|2024) ;;
  *) echo "ERROR: year must be 2023 or 2024, got '${YEAR}'" >&2; exit 2;;
esac

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

PY="/home/nacevedo/.conda/envs/fairness_env/bin/python"
RESULT_ROOT="output/rolling_origin_cv_sim${YEAR}"
DATA_PATH="data/CCAO/2025/training_data_sim${YEAR}.parquet"
IMG_DIR="Paper/GE slides/img${YEAR}"
BUILD_DIR="slide_build"
TEX_FILE="Paper/GE slides/slides_v1_5 ${YEAR}.tex"
SUMMARY_JSON="${BUILD_DIR}/slide_summary_${YEAR}.json"

mkdir -p "${IMG_DIR}" "${BUILD_DIR}"

echo "[build_slides ${YEAR}] step 1/2 reproduce pipeline"
${PY} scripts/reproduce_slides_pipeline.py \
  --year-tag "${YEAR}" \
  --result-root "${RESULT_ROOT}" \
  --data-path "${DATA_PATH}" \
  --img-out-dir "${IMG_DIR}" \
  --build-dir "${BUILD_DIR}"

echo "[build_slides ${YEAR}] step 2/2 apply edits to ${TEX_FILE}"
${PY} scripts/update_slides_tex.py \
  --year "${YEAR}" \
  --summary-json "${SUMMARY_JSON}" \
  --tex-file "${TEX_FILE}" \
  --img-rel "img${YEAR}"

echo "[build_slides ${YEAR}] done"
