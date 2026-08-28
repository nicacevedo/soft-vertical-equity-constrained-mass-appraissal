#!/bin/bash
# Submit the 994-tree transition-region analysis DAG. No model fitting.
# J0 PRECHECK -> (J1 TESTS || J2 CORE) -> J3 RENDER -> J4 FINAL_AUDIT
set -euo pipefail
cd /home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal

RESULT_ROOT="output/paper_v6_preselection_994"
DATA_ID="d4929d43ec19badf"
SPLIT_ID="3d464d4a611b131b"
OUTPUT_ROOT="${RESULT_ROOT}/analysis/data_id=${DATA_ID}/split_id=${SPLIT_ID}/penalty_path_analysis/transition_regions_v1"
SBATCH_SRC="scripts/run_transition_regions_stage.sbatch"
REPO="$(pwd)"

mkdir -p \
  "$OUTPUT_ROOT/protocol" \
  "$OUTPUT_ROOT/tables" \
  "$OUTPUT_ROOT/figures" \
  "$OUTPUT_ROOT/provenance" \
  "$OUTPUT_ROOT/logs" \
  "$OUTPUT_ROOT/cluster" \
  "$OUTPUT_ROOT/report" \
  "$OUTPUT_ROOT/code_snapshot" \
  "$OUTPUT_ROOT/qa"

cp -f "$SBATCH_SRC" "$OUTPUT_ROOT/cluster/run_transition_regions_stage.sbatch"
cp -f "$0" "$OUTPUT_ROOT/cluster/submit_transition_regions_v1.sh" || true
chmod +x "$OUTPUT_ROOT/cluster/submit_transition_regions_v1.sh" "$OUTPUT_ROOT/cluster/run_transition_regions_stage.sbatch"

# Freeze protocol to disk before any event extraction (no metric paths are read here).
conda run -n fairness_env --no-capture-output python - <<PY
import sys
from pathlib import Path
sys.path.insert(0, ".")
from utils.transition_regions import protocol_canonical_json, protocol_sha256
out = Path("$OUTPUT_ROOT") / "protocol" / "transition_analysis_protocol.json"
out.parent.mkdir(parents=True, exist_ok=True)
text = protocol_canonical_json()
if out.is_file() and out.read_text(encoding="utf-8") != text:
    raise SystemExit("existing protocol does not match frozen code protocol")
out.write_text(text, encoding="utf-8")
print("protocol_sha256", protocol_sha256())
print("wrote", out)
PY

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
      --export=ALL,TR_STAGE=precheck,TR_RESULT_ROOT="$RESULT_ROOT",TR_OUTPUT_ROOT="$OUTPUT_ROOT" \
      "$OUTPUT_ROOT/cluster/run_transition_regions_stage.sbatch" >/tmp/tr994_testonly.out 2>/tmp/tr994_testonly.err; then
    PART="$cand"
    echo "Selected partition $PART"
    cat /tmp/tr994_testonly.out || true
    break
  else
    echo "Partition $cand not usable:"
    cat /tmp/tr994_testonly.err || true
  fi
done
if [[ -z "$PART" ]]; then
  echo "No usable CPU partition found" >&2
  exit 1
fi

EXPORT_BASE="ALL,TR_RESULT_ROOT=${RESULT_ROOT},TR_OUTPUT_ROOT=${OUTPUT_ROOT}"
SB="$OUTPUT_ROOT/cluster/run_transition_regions_stage.sbatch"

J0=$(sbatch --parsable -p "$PART" --job-name=tr994_j0_precheck \
  --cpus-per-task=1 --mem=4G --time=00:20:00 \
  --export="${EXPORT_BASE},TR_STAGE=precheck" "$SB")
J1=$(sbatch --parsable -p "$PART" --job-name=tr994_j1_tests \
  --cpus-per-task=2 --mem=8G --time=00:30:00 \
  --dependency=afterok:"$J0" \
  --export="${EXPORT_BASE},TR_STAGE=tests" "$SB")
J2=$(sbatch --parsable -p "$PART" --job-name=tr994_j2_core \
  --cpus-per-task=2 --mem=8G --time=00:30:00 \
  --dependency=afterok:"$J0" \
  --export="${EXPORT_BASE},TR_STAGE=core" "$SB")
J3=$(sbatch --parsable -p "$PART" --job-name=tr994_j3_render \
  --cpus-per-task=2 --mem=8G --time=00:30:00 \
  --dependency=afterok:"${J1}:${J2}" \
  --export="${EXPORT_BASE},TR_STAGE=render" "$SB")
J4=$(sbatch --parsable -p "$PART" --job-name=tr994_j4_audit \
  --cpus-per-task=1 --mem=4G --time=00:20:00 \
  --dependency=afterok:"$J3" \
  --export="${EXPORT_BASE},TR_STAGE=audit" "$SB")

{
  echo "J0_PRECHECK $J0"
  echo "J1_TESTS $J1 afterok:$J0"
  echo "J2_CORE_ANALYSIS $J2 afterok:$J0"
  echo "J3_RENDER $J3 afterok:${J1}:${J2}"
  echo "J4_FINAL_AUDIT $J4 afterok:$J3"
  echo "partition $PART"
  echo "output_root $OUTPUT_ROOT"
  echo "log_root $OUTPUT_ROOT/logs"
} | tee "$OUTPUT_ROOT/cluster/job_ids.txt"

squeue -u "$USER" -o "%.18i %.12P %.22j %.8u %.2t %.10M %.6D %R" | tee "$OUTPUT_ROOT/cluster/squeue_after_submit.txt"

python - <<PY
import json
from pathlib import Path
from datetime import datetime, timezone
payload = {
    "submitted_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "partition": "$PART",
    "result_root": str(Path("$RESULT_ROOT").resolve()),
    "output_root": str(Path("$OUTPUT_ROOT").resolve()),
    "log_root": str(Path("$OUTPUT_ROOT/logs").resolve()),
    "jobs": {
        "J0_PRECHECK": {"id": "$J0", "dependency": None, "cpus": 1, "mem": "4G", "time": "00:20:00", "stage": "precheck"},
        "J1_TESTS": {"id": "$J1", "dependency": "afterok:$J0", "cpus": 2, "mem": "8G", "time": "00:30:00", "stage": "tests"},
        "J2_CORE_ANALYSIS": {"id": "$J2", "dependency": "afterok:$J0", "cpus": 2, "mem": "8G", "time": "00:30:00", "stage": "core"},
        "J3_RENDER": {"id": "$J3", "dependency": "afterok:${J1}:${J2}", "cpus": 2, "mem": "8G", "time": "00:30:00", "stage": "render"},
        "J4_FINAL_AUDIT_AND_BUNDLE": {"id": "$J4", "dependency": "afterok:$J3", "cpus": 1, "mem": "4G", "time": "00:20:00", "stage": "audit"},
    },
    "graph": "PRECHECK -> (TESTS || CORE_ANALYSIS) -> RENDER -> FINAL_AUDIT",
    "gpu": False,
    "model_fitting": False,
    "session_independent": True,
    "no_tex": True,
    "no_paper_writes": True,
}
Path("$OUTPUT_ROOT/cluster/SUBMISSION_STATUS.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY

echo "All five jobs submitted."
