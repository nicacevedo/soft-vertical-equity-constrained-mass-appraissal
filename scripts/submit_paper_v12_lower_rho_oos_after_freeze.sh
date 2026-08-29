#!/bin/bash
# Submit OOS + downstream stages after CV freeze PASS. Do not run before freeze.
set -euo pipefail
cd /home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal
EXT=output/paper_v12_lower_rho_extension_994_v2
FREEZE_STATUS="$EXT/analysis/data_id=d4929d43ec19badf/split_id=3d464d4a611b131b/penalty_path_analysis/transition_regions_v2_lower_rho/qa/CV_TRANSITION_FREEZE_STATUS.json"
if [[ ! -f "$FREEZE_STATUS" ]]; then
  echo "FAIL: freeze status missing: $FREEZE_STATUS" >&2
  exit 1
fi
python3 - <<PY
import json,sys
from pathlib import Path
p=Path("$FREEZE_STATUS")
d=json.loads(p.read_text())
if d.get("status") != "PASS":
    raise SystemExit("FAIL: freeze is not PASS")
if not d.get("oos_information_was_not_read"):
    raise SystemExit("FAIL: freeze did not certify OOS unread")
print("FREEZE_PASS")
PY

PART_MIX="sched_mit_sloan_batch_r8,ou_sloan_batch,mit_normal"
OOS=scripts/run_paper_v12_lower_rho_oos_array.sbatch
ST=scripts/run_paper_v12_lower_rho_stage.sbatch

HO_DIR=$(sbatch --parsable -p "$PART_MIX" --job-name=v12lr_ho_dir --export=ALL,P6_FAMILY=LGBCovPenalty,P6_STAGE=test,P6_EVAL=heldout "$OOS")
HO_SUR=$(sbatch --parsable -p "$PART_MIX" --job-name=v12lr_ho_sur --export=ALL,P6_FAMILY=LGBSmoothPenalty,P6_STAGE=test,P6_EVAL=heldout "$OOS")
FW_DIR=$(sbatch --parsable -p "$PART_MIX" --job-name=v12lr_fw_dir --export=ALL,P6_FAMILY=LGBCovPenalty,P6_STAGE=forward,P6_EVAL=forward_2025 "$OOS")
FW_SUR=$(sbatch --parsable -p "$PART_MIX" --job-name=v12lr_fw_sur --export=ALL,P6_FAMILY=LGBSmoothPenalty,P6_STAGE=forward,P6_EVAL=forward_2025 "$OOS")
MERGE=$(sbatch --parsable -p "$PART_MIX" --job-name=v12lr_merge --dependency=afterok:"${HO_DIR}:${HO_SUR}:${FW_DIR}:${FW_SUR}" --mem=32G --time=03:00:00 --export=ALL,V12_STAGE=merge-complete "$ST")
FIGS=$(sbatch --parsable -p "$PART_MIX" --job-name=v12lr_figs --dependency=afterok:"$MERGE" --mem=16G --time=01:30:00 --export=ALL,V12_STAGE=figures "$ST")
TABS=$(sbatch --parsable -p "$PART_MIX" --job-name=v12lr_tabs --dependency=afterok:"$FIGS" --export=ALL,V12_STAGE=table-sources "$ST")
POP=$(sbatch --parsable -p "$PART_MIX" --job-name=v12lr_pop --dependency=afterok:"$TABS" --export=ALL,V12_STAGE=populate-tex "$ST")
QA=$(sbatch --parsable -p "$PART_MIX" --job-name=v12lr_qa --dependency=afterok:"$POP" --export=ALL,V12_STAGE=audit "$ST")

python3 - <<PY
import json
from pathlib import Path
from datetime import datetime, timezone
p=Path("$EXT/cluster/SUBMISSION_STATUS.json")
d=json.loads(p.read_text()) if p.exists() else {}
d.update({
    "oos_submitted_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "heldout_direct": "$HO_DIR",
    "heldout_surrogate": "$HO_SUR",
    "forward_direct": "$FW_DIR",
    "forward_surrogate": "$FW_SUR",
    "merge": "$MERGE",
    "figures": "$FIGS",
    "table_sources": "$TABS",
    "populate_tex": "$POP",
    "audit": "$QA",
    "oos_submitted": True,
    "oos_after_freeze_pass": True,
})
p.write_text(json.dumps(d, indent=2)+"\n")
print(json.dumps(d, indent=2))
PY
