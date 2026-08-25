#!/bin/bash
# Submit the EXPERIMENTAL / TOY 3x2 surrogate ablation.
# Default: probe/calibrate first. After it finishes, rerun with --follow-up
# to launch the six-variant array and assemble job using measured resources.
set -euo pipefail
cd /home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal

ROOT=output/toy_surrogate_ablation
mkdir -p "$ROOT/logs" "$ROOT/manifests"
PART="${TOY_PARTITION:-sched_mit_sloan_batch_r8}"
PROBE=scripts/run_toy_surrogate_ablation_probe.sbatch
ARRAY=scripts/run_toy_surrogate_ablation_array.sbatch
ASM=scripts/run_toy_surrogate_ablation_assemble.sbatch

if [[ "${1:-}" == "--follow-up" ]]; then
  TIMING="$ROOT/probe_timing.json"
  if [[ ! -f "$TIMING" ]]; then
    echo "Missing $TIMING; probe/calibrate has not finished." >&2
    exit 1
  fi
  CPUS=$(/home/nacevedo/.conda/envs/fairness_env/bin/python - <<'PY'
import json
from pathlib import Path
t=json.loads(Path("output/toy_surrogate_ablation/probe_timing.json").read_text())
cpus=int(t.get("recommended_cpus") or 8)
mem=int(t.get("recommended_mem_gb") or 80)
hours=float(t.get("recommended_time_hours_per_variant") or 12)
hours=min(36.0, max(4.0, hours))
hh=int(hours); mm=int(round((hours-hh)*60))
print(cpus)
print(max(48, mem))
print(f"{hh:02d}:{mm:02d}:00")
print(t.get("runtime_sec"))
print(t.get("peak_rss_gb"))
PY
)
  REC_CPUS=$(echo "$CPUS" | sed -n '1p')
  REC_MEM=$(echo "$CPUS" | sed -n '2p')
  REC_TIME=$(echo "$CPUS" | sed -n '3p')
  echo "follow-up cpus=$REC_CPUS mem=${REC_MEM}G time=$REC_TIME partition=$PART"
  ARR_ID=$(sbatch --parsable -p "$PART" --cpus-per-task="$REC_CPUS" --mem="${REC_MEM}G" --time="$REC_TIME" "$ARRAY")
  ASM_ID=$(sbatch --parsable -p "$PART" --dependency=afterok:"$ARR_ID" "$ASM")
  /home/nacevedo/.conda/envs/fairness_env/bin/python - <<PY
import json
from pathlib import Path
payload = {
    "experiment_label": "EXPERIMENTAL / TOY / NON-CANONICAL",
    "partition": "$PART",
    "array_job": "$ARR_ID",
    "assemble_job": "$ASM_ID",
    "cpus_per_task": int("$REC_CPUS"),
    "mem_gb": int("$REC_MEM"),
    "time": "$REC_TIME",
}
Path("$ROOT/manifests/slurm_followup.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY
  exit 0
fi

PROBE_ID=$(sbatch --parsable -p "$PART" "$PROBE")
/home/nacevedo/.conda/envs/fairness_env/bin/python - <<PY
import json
from pathlib import Path
payload = {
    "experiment_label": "EXPERIMENTAL / TOY / NON-CANONICAL",
    "partition": "$PART",
    "probe_job": "$PROBE_ID",
    "note": "After probe finishes, run scripts/submit_toy_surrogate_ablation.sh --follow-up",
}
Path("$ROOT/manifests/slurm_probe.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY
