#!/bin/bash
# Submit the 994-tree paper-asset follow-up DAG. No model fitting.
# J0 PREFLIGHT -> J1 ANALYSIS -> J2 RENDER -> J3 AUDIT
set -euo pipefail
cd /home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal

RESULT_ROOT="output/paper_v6_preselection_994"
DATA_ID="d4929d43ec19badf"
SPLIT_ID="3d464d4a611b131b"
V1_ROOT="${RESULT_ROOT}/analysis/data_id=${DATA_ID}/split_id=${SPLIT_ID}/penalty_path_analysis/transition_regions_v1"
OUTPUT_ROOT="${RESULT_ROOT}/analysis/data_id=${DATA_ID}/split_id=${SPLIT_ID}/penalty_path_analysis/transition_regions_paper_assets_v1"
SBATCH_SRC="scripts/run_transition_paper_assets_stage.sbatch"

mkdir -p \
  "$OUTPUT_ROOT/tables" \
  "$OUTPUT_ROOT/figures/main" \
  "$OUTPUT_ROOT/figures/appendix" \
  "$OUTPUT_ROOT/figures/diagnostic" \
  "$OUTPUT_ROOT/report" \
  "$OUTPUT_ROOT/qa" \
  "$OUTPUT_ROOT/provenance" \
  "$OUTPUT_ROOT/logs" \
  "$OUTPUT_ROOT/cluster" \
  "$OUTPUT_ROOT/code_snapshot"

cp -f "$SBATCH_SRC" "$OUTPUT_ROOT/cluster/run_transition_paper_assets_stage.sbatch"
cp -f "$0" "$OUTPUT_ROOT/cluster/submit_transition_paper_assets_v1.sh" || true
chmod +x "$OUTPUT_ROOT/cluster/submit_transition_paper_assets_v1.sh" "$OUTPUT_ROOT/cluster/run_transition_paper_assets_stage.sbatch"

PARTS=(
  "sched_mit_sloan_batch_r8"
  "sched_mit_sloan_batch"
  "ou_sloan_batch"
  "mit_normal"
)
PART=""
for cand in "${PARTS[@]}"; do
  echo "Trying sbatch --test-only -p $cand"
  if sbatch --test-only -p "$cand" --cpus-per-task=1 --mem=4G --time=00:15:00 \
      --export=ALL,TR_STAGE=preflight,TR_RESULT_ROOT="$RESULT_ROOT",TR_OUTPUT_ROOT="$OUTPUT_ROOT",TR_V1_ROOT="$V1_ROOT" \
      "$OUTPUT_ROOT/cluster/run_transition_paper_assets_stage.sbatch" >/tmp/tr994pa_testonly.out 2>/tmp/tr994pa_testonly.err; then
    PART="$cand"
    echo "Selected partition $PART"
    cat /tmp/tr994pa_testonly.out || true
    break
  else
    echo "Partition $cand not usable:"
    cat /tmp/tr994pa_testonly.err || true
  fi
done
if [[ -z "$PART" ]]; then
  echo "No usable CPU partition found" >&2
  exit 1
fi

EXPORT_BASE="ALL,TR_RESULT_ROOT=${RESULT_ROOT},TR_OUTPUT_ROOT=${OUTPUT_ROOT},TR_V1_ROOT=${V1_ROOT}"
SB="$OUTPUT_ROOT/cluster/run_transition_paper_assets_stage.sbatch"

J0=$(sbatch --parsable -p "$PART" --job-name=tr994pa_j0_preflight \
  --cpus-per-task=1 --mem=4G --time=00:20:00 \
  --export="${EXPORT_BASE},TR_STAGE=preflight" "$SB")
J1=$(sbatch --parsable -p "$PART" --job-name=tr994pa_j1_analysis \
  --cpus-per-task=2 --mem=8G --time=00:30:00 \
  --dependency=afterok:"$J0" \
  --export="${EXPORT_BASE},TR_STAGE=analysis" "$SB")
J2=$(sbatch --parsable -p "$PART" --job-name=tr994pa_j2_render \
  --cpus-per-task=2 --mem=12G --time=00:45:00 \
  --dependency=afterok:"$J1" \
  --export="${EXPORT_BASE},TR_STAGE=render" "$SB")
J3=$(sbatch --parsable -p "$PART" --job-name=tr994pa_j3_audit \
  --cpus-per-task=1 --mem=4G --time=00:25:00 \
  --dependency=afterok:"$J2" \
  --export="${EXPORT_BASE},TR_STAGE=audit" "$SB")

{
  echo "J0_PAPER_ASSET_PREFLIGHT $J0"
  echo "J1_FOLLOWUP_ANALYSIS $J1 afterok:$J0"
  echo "J2_PAPER_RENDER $J2 afterok:$J1"
  echo "J3_FINAL_PAPER_ASSET_AUDIT $J3 afterok:$J2"
  echo "partition $PART"
  echo "output_root $OUTPUT_ROOT"
  echo "log_root $OUTPUT_ROOT/logs"
} | tee "$OUTPUT_ROOT/cluster/job_ids.txt"

squeue -u "$USER" -o "%.18i %.12P %.22j %.8u %.2t %.10M %.6D %R" | tee "$OUTPUT_ROOT/cluster/squeue_after_submit.txt"

python3 - <<PY
import json
from pathlib import Path
from datetime import datetime, timezone
payload = {
    "submitted_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "partition": "$PART",
    "result_root": str(Path("$RESULT_ROOT").resolve()),
    "v1_root": str(Path("$V1_ROOT").resolve()),
    "output_root": str(Path("$OUTPUT_ROOT").resolve()),
    "log_root": str(Path("$OUTPUT_ROOT/logs").resolve()),
    "jobs": {
        "J0_PAPER_ASSET_PREFLIGHT": {"id": "$J0", "dependency": None, "cpus": 1, "mem": "4G", "time": "00:20:00", "stage": "preflight"},
        "J1_FOLLOWUP_ANALYSIS": {"id": "$J1", "dependency": "afterok:$J0", "cpus": 2, "mem": "8G", "time": "00:30:00", "stage": "analysis"},
        "J2_PAPER_RENDER": {"id": "$J2", "dependency": "afterok:$J1", "cpus": 2, "mem": "12G", "time": "00:45:00", "stage": "render"},
        "J3_FINAL_PAPER_ASSET_AUDIT": {"id": "$J3", "dependency": "afterok:$J2", "cpus": 1, "mem": "4G", "time": "00:25:00", "stage": "audit"},
    },
    "graph": "PREFLIGHT -> FOLLOWUP_ANALYSIS -> PAPER_RENDER -> FINAL_PAPER_ASSET_AUDIT",
    "gpu": False,
    "model_fitting": False,
    "session_independent": True,
    "no_tex": True,
    "no_paper_writes": True,
    "no_v1_mutation": True,
}
Path("$OUTPUT_ROOT/cluster/SUBMISSION_STATUS.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY

echo "All four paper-asset jobs submitted."
