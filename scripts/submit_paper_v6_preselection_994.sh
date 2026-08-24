#!/bin/bash
# Submit the 994-tree paper-v6 preselection chain. No penalized-model selection.
set -euo pipefail
cd /home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal

ROOT=output/paper_v6_preselection_994
mkdir -p "$ROOT/logs" "$ROOT/manifests"
PART_LONG="sched_mit_sloan_batch_r8"
PART_MIX="sched_mit_sloan_batch_r8,ou_sloan_batch,mit_normal"
OOS=scripts/run_paper_v6_preselection_994_oos_array.sbatch
DEP=scripts/run_paper_v6_preselection_994_dep.sbatch

python scripts/setup_paper_v6_preselection_994.py

CV_ID=$(sbatch --parsable -p "$PART_LONG" scripts/run_paper_v6_preselection_994_cv.sbatch)
BASE_ID=$(sbatch --parsable -p "$PART_MIX" scripts/run_paper_v6_preselection_994_baseline_report.sbatch)

HO_DIR=$(sbatch --parsable -p "$PART_MIX" --job-name=p6_994_ho_dir \
  --export=ALL,P6_FAMILY=LGBCovPenalty,P6_STAGE=test,P6_EVAL=heldout "$OOS")
HO_SUR=$(sbatch --parsable -p "$PART_MIX" --job-name=p6_994_ho_sur \
  --export=ALL,P6_FAMILY=LGBSmoothPenalty,P6_STAGE=test,P6_EVAL=heldout "$OOS")
FW_DIR=$(sbatch --parsable -p "$PART_MIX" --job-name=p6_994_fw_dir \
  --export=ALL,P6_FAMILY=LGBCovPenalty,P6_STAGE=forward,P6_EVAL=forward_2025 "$OOS")
FW_SUR=$(sbatch --parsable -p "$PART_MIX" --job-name=p6_994_fw_sur \
  --export=ALL,P6_FAMILY=LGBSmoothPenalty,P6_STAGE=forward,P6_EVAL=forward_2025 "$OOS")

CVQA=$(sbatch --parsable -p "$PART_MIX" --job-name=p6_994_cvqa \
  --dependency=afterok:"$CV_ID" --export=ALL,P6_CMD=cv-qa "$DEP")
PREV=$(sbatch --parsable -p "$PART_MIX" --job-name=p6_994_preview \
  --dependency=afterok:"${HO_DIR}:${HO_SUR}:${FW_DIR}:${FW_SUR}:${BASE_ID}" \
  --export=ALL,P6_CMD=preview "$DEP")
MERGE=$(sbatch --parsable -p "$PART_MIX" --job-name=p6_994_merge \
  --dependency=afterok:"${CVQA}:${PREV}" --export=ALL,P6_CMD=merge "$DEP")
POP=$(sbatch --parsable -p "$PART_MIX" --job-name=p6_994_pop \
  --dependency=afterok:"$MERGE" --export=ALL,P6_CMD=populate "$DEP")
QA=$(sbatch --parsable -p "$PART_MIX" --job-name=p6_994_finalqa \
  --dependency=afterok:"$POP" --export=ALL,P6_CMD=final-qa "$DEP")

python - <<PY
import json
from pathlib import Path
payload = {
    "selection_performed": False,
    "baseline_decision": "ADOPT_994",
    "cv": "$CV_ID",
    "baseline_report": "$BASE_ID",
    "heldout_direct": "$HO_DIR",
    "heldout_surrogate": "$HO_SUR",
    "forward_direct": "$FW_DIR",
    "forward_surrogate": "$FW_SUR",
    "cv_qa": "$CVQA",
    "preview": "$PREV",
    "merge": "$MERGE",
    "populate": "$POP",
    "final_qa": "$QA",
    "partitions_long": "$PART_LONG",
    "partitions_mix": "$PART_MIX",
    "result_root": "$ROOT",
    "no_selection_confirmation": "No rho, penalty family, or penalized configuration was selected or ranked in this analysis.",
}
Path("$ROOT/manifests/slurm_graph.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY
