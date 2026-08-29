#!/bin/bash
# Submit v2 lower-rho DAG. OOS is afterok on CV freeze. No TeX compilation.
set -euo pipefail
cd /home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal

EXT=output/paper_v12_lower_rho_extension_994_v2
mkdir -p "$EXT/logs" "$EXT/cluster"
PART_LONG="sched_mit_sloan_batch_r8"
PART_MIX="sched_mit_sloan_batch_r8,ou_sloan_batch,mit_normal"
OOS=scripts/run_paper_v12_lower_rho_oos_array.sbatch
ST=scripts/run_paper_v12_lower_rho_stage.sbatch
CV=scripts/run_paper_v12_lower_rho_cv.sbatch

CV_ID=$(sbatch --parsable -p "$PART_LONG" "$CV")
FREEZE=$(sbatch --parsable -p "$PART_MIX" --job-name=v12lr_freeze \
  --dependency=afterok:"$CV_ID" --export=ALL,V12_STAGE=freeze-cv "$ST")

HO_DIR=$(sbatch --parsable -p "$PART_MIX" --job-name=v12lr_ho_dir \
  --dependency=afterok:"$FREEZE" \
  --export=ALL,P6_FAMILY=LGBCovPenalty,P6_STAGE=test,P6_EVAL=heldout "$OOS")
HO_SUR=$(sbatch --parsable -p "$PART_MIX" --job-name=v12lr_ho_sur \
  --dependency=afterok:"$FREEZE" \
  --export=ALL,P6_FAMILY=LGBSmoothPenalty,P6_STAGE=test,P6_EVAL=heldout "$OOS")
FW_DIR=$(sbatch --parsable -p "$PART_MIX" --job-name=v12lr_fw_dir \
  --dependency=afterok:"$FREEZE" \
  --export=ALL,P6_FAMILY=LGBCovPenalty,P6_STAGE=forward,P6_EVAL=forward_2025 "$OOS")
FW_SUR=$(sbatch --parsable -p "$PART_MIX" --job-name=v12lr_fw_sur \
  --dependency=afterok:"$FREEZE" \
  --export=ALL,P6_FAMILY=LGBSmoothPenalty,P6_STAGE=forward,P6_EVAL=forward_2025 "$OOS")

MERGE=$(sbatch --parsable -p "$PART_MIX" --job-name=v12lr_merge \
  --dependency=afterok:"${HO_DIR}:${HO_SUR}:${FW_DIR}:${FW_SUR}" \
  --mem=32G --time=03:00:00 --export=ALL,V12_STAGE=merge-complete "$ST")
FIGS=$(sbatch --parsable -p "$PART_MIX" --job-name=v12lr_figs \
  --dependency=afterok:"$MERGE" --mem=16G --time=01:30:00 --export=ALL,V12_STAGE=figures "$ST")
TABS=$(sbatch --parsable -p "$PART_MIX" --job-name=v12lr_tabs \
  --dependency=afterok:"$FIGS" --export=ALL,V12_STAGE=table-sources "$ST")
POP=$(sbatch --parsable -p "$PART_MIX" --job-name=v12lr_pop \
  --dependency=afterok:"$TABS" --export=ALL,V12_STAGE=populate-tex "$ST")
QA=$(sbatch --parsable -p "$PART_MIX" --job-name=v12lr_qa \
  --dependency=afterok:"$POP" --export=ALL,V12_STAGE=audit "$ST")

python3 - <<PY
import json
from pathlib import Path
from datetime import datetime, timezone
payload = {
    "submitted_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "cv": "$CV_ID",
    "freeze": "$FREEZE",
    "heldout_direct": "$HO_DIR",
    "heldout_surrogate": "$HO_SUR",
    "forward_direct": "$FW_DIR",
    "forward_surrogate": "$FW_SUR",
    "merge": "$MERGE",
    "figures": "$FIGS",
    "table_sources": "$TABS",
    "populate_tex": "$POP",
    "audit": "$QA",
    "oos_afterok_freeze": True,
    "no_tex_compilation": True,
}
Path("$EXT/cluster/SUBMISSION_STATUS.json").write_text(json.dumps(payload, indent=2)+"\n")
print(json.dumps(payload, indent=2))
PY
