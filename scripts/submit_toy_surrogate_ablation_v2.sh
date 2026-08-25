#!/bin/bash
# Submit EXPERIMENTAL / TOY V2: calibrate -> 3 family tasks -> assemble.
set -euo pipefail
cd /home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal

ROOT=output/toy_surrogate_ablation_v2
mkdir -p "$ROOT/logs" "$ROOT/manifests"
PART="${TOY_PARTITION:-sched_mit_sloan_batch_r8}"
CAL=scripts/run_toy_surrogate_ablation_v2_calibrate.sbatch
ARR=scripts/run_toy_surrogate_ablation_v2_array.sbatch
ASM=scripts/run_toy_surrogate_ablation_v2_assemble.sbatch

CAL_ID=$(sbatch --parsable -p "$PART" "$CAL")
ARR_ID=$(sbatch --parsable -p "$PART" --dependency=afterok:"$CAL_ID" "$ARR")
ASM_ID=$(sbatch --parsable -p "$PART" --dependency=afterok:"$ARR_ID" "$ASM")

/home/nacevedo/.conda/envs/fairness_env/bin/python - <<PY
import json
from pathlib import Path
payload = {
    "experiment_label": "EXPERIMENTAL / TOY / NON-CANONICAL",
    "experiment": "toy_surrogate_ablation_v2",
    "partition": "$PART",
    "calibrate_job": "$CAL_ID",
    "array_job": "$ARR_ID",
    "assemble_job": "$ASM_ID",
    "cpus_per_task_family": 8,
    "mem_gb_family": 24,
    "time_family": "04:00:00",
    "gpus": 0,
    "note": "No CV. Three parallel family tasks after shared calibration. Refinements run inside each family job.",
}
Path("$ROOT/manifests/slurm_submit.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY
