#!/bin/bash
# Submit the COMPLETE cluster-resident hybrid continuation graph.
# Local terminal is used only to submit; scientific work does not depend on it.
set -euo pipefail
cd /home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal

ROOT=output/toy_surrogate_ablation_v2
mkdir -p "$ROOT/logs" "$ROOT/manifests" "$ROOT/cache" \
  "$ROOT/families/quadratic_direct_cap" "$ROOT/families/quadratic_nl_guardrail"
PART="${TOY_PARTITION:-sched_mit_sloan_batch_r8}"
TIME_TGT="${TOY_HYBRID_TARGET_TIME:-03:00:00}"
PRE=scripts/run_toy_hybrid_preflight.sbatch
TGT=scripts/run_toy_hybrid_target.sbatch
ASM=scripts/run_toy_hybrid_assemble.sbatch

PRE_ID=$(sbatch --parsable -p "$PART" "$PRE")
echo "Submitted hybrid preflight ${PRE_ID}"

QD_ID=$(sbatch --parsable -p "$PART" --time="$TIME_TGT" --dependency=afterok:"$PRE_ID" \
  --job-name="toy_hyb_qd" \
  --export=ALL,HYBRID_METHOD=quadratic_direct_cap \
  "$TGT")
echo "Submitted QD array ${QD_ID} (afterok:${PRE_ID})"

QNL_ID=$(sbatch --parsable -p "$PART" --time="$TIME_TGT" --dependency=afterok:"$PRE_ID" \
  --job-name="toy_hyb_qnl" \
  --export=ALL,HYBRID_METHOD=quadratic_nl_guardrail \
  "$TGT")
echo "Submitted QNL array ${QNL_ID} (afterok:${PRE_ID})"

ASM_ID=$(sbatch --parsable -p "$PART" --dependency=afterany:"${QD_ID}":"${QNL_ID}" "$ASM")
echo "Submitted hybrid assemble ${ASM_ID} (afterany:${QD_ID}:${QNL_ID})"

export PRE_ID QD_ID QNL_ID ASM_ID PART ROOT TIME_TGT
/home/nacevedo/.conda/envs/fairness_env/bin/python - <<'PY'
import json, os
from pathlib import Path
payload = {
    "experiment_label": "EXPERIMENTAL / TOY / NON-CANONICAL",
    "experiment": "toy_hybrid_selection_v2",
    "partition": os.environ["PART"],
    "preflight_job_id": os.environ["PRE_ID"],
    "qd_array_job_id": os.environ["QD_ID"],
    "qnl_array_job_id": os.environ["QNL_ID"],
    "assemble_job_id": os.environ["ASM_ID"],
    "target_s": [0.20, 0.15, 0.10],
    "methods": ["quadratic_direct_cap", "quadratic_nl_guardrail"],
    "dependency": {
        "qd_array": "afterok:" + os.environ["PRE_ID"],
        "qnl_array": "afterok:" + os.environ["PRE_ID"],
        "assemble": "afterany:" + os.environ["QD_ID"] + ":" + os.environ["QNL_ID"],
    },
    "cpus_per_task_target": 8,
    "mem_gb_target": 12,
    "time_target": os.environ["TIME_TGT"],
    "gpus": 0,
    "cluster_resident": True,
    "reuses_locked_benchmarks": ["current_direct", "direct_mm_k1", "quadratic"],
    "note": "No local scientific fallback. Login session may disconnect. Assemble runs even if a target fails.",
}
path = Path(os.environ["ROOT"]) / "manifests" / "hybrid_job_graph.json"
path.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY
