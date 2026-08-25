#!/bin/bash
# Submit the COMPLETE cluster-resident follow-up V3 graph.
set -euo pipefail
cd /home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal

ROOT=output/toy_surrogate_followup_v3
mkdir -p "$ROOT/logs" "$ROOT/manifests"
PART="${TOY_PARTITION:-sched_mit_sloan_batch_r8}"
PRE=scripts/run_toy_followup_preflight.sbatch
QD=scripts/run_toy_followup_qd.sbatch
QNL=scripts/run_toy_followup_qnl.sbatch
ASM=scripts/run_toy_followup_assemble.sbatch

PRE_ID=$(sbatch --parsable -p "$PART" "$PRE")
echo "Submitted follow-up precheck ${PRE_ID}"

QD_ID=$(sbatch --parsable -p "$PART" --dependency=afterok:"$PRE_ID" "$QD")
echo "Submitted QD array ${QD_ID} (afterok:${PRE_ID})"

QNL_ID=$(sbatch --parsable -p "$PART" --dependency=afterok:"$PRE_ID" "$QNL")
echo "Submitted QNL array ${QNL_ID} (afterok:${PRE_ID})"

ASM_ID=$(sbatch --parsable -p "$PART" --dependency=afterok:"${QD_ID}":"${QNL_ID}" "$ASM")
echo "Submitted follow-up assemble ${ASM_ID} (afterok:${QD_ID}:${QNL_ID})"

export PRE_ID QD_ID QNL_ID ASM_ID PART ROOT
/home/nacevedo/.conda/envs/fairness_env/bin/python - <<'PY'
import json, os
from pathlib import Path
payload = {
    "experiment_label": "EXPERIMENTAL / TOY / NON-CANONICAL",
    "experiment": "toy_surrogate_followup_v3",
    "partition": os.environ["PART"],
    "precheck_job_id": os.environ["PRE_ID"],
    "qd_array_job_id": os.environ["QD_ID"],
    "qnl_array_job_id": os.environ["QNL_ID"],
    "assemble_job_id": os.environ["ASM_ID"],
    "dependency": {
        "qd_array": "afterok:" + os.environ["PRE_ID"],
        "qnl_array": "afterok:" + os.environ["PRE_ID"],
        "assemble": "afterok:" + os.environ["QD_ID"] + ":" + os.environ["QNL_ID"],
    },
    "cpus_per_task": 8,
    "mem_gb": 24,
    "gpus": 0,
    "cluster_resident": True,
    "does_not_modify_v2": True,
    "note": "No local scientific fallback. Login session may disconnect.",
}
path = Path(os.environ["ROOT"]) / "manifests" / "RUN_MANIFEST.json"
path.write_text(json.dumps(payload, indent=2) + "\n")
Path(os.environ["ROOT"]).joinpath("RUN_MANIFEST.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY
