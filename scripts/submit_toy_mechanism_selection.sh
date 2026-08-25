#!/bin/bash
# Submit the COMPLETE cluster-resident six-path mechanism graph.
# Local terminal is used only to submit; scientific work does not depend on it.
# If preflight already PASS, skip it and resubmit families+assemble only (resume).
set -euo pipefail
cd /home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal

ROOT=output/toy_surrogate_ablation_v2
mkdir -p "$ROOT/logs" "$ROOT/manifests" "$ROOT/cache"
PART="${TOY_PARTITION:-sched_mit_sloan_batch_r8}"
TIME_FAM="${TOY_FAMILY_TIME:-12:00:00}"
PRE=scripts/run_toy_mechanism_preflight.sbatch
FAM=scripts/run_toy_mechanism_family.sbatch
ASM=scripts/run_toy_mechanism_assemble.sbatch

METHODS=(current_direct direct_mm_k1 moment_mm_k2 moment_mm_k3 local_slope_smooth quadratic)

SKIP_PRE=0
if [[ "${TOY_SKIP_PREFLIGHT:-}" == "1" ]]; then
  SKIP_PRE=1
elif [[ -f "$ROOT/preflight_report.json" ]]; then
  PRE_STATUS=$(/home/nacevedo/.conda/envs/fairness_env/bin/python -c "import json; print(json.load(open('$ROOT/preflight_report.json')).get('status',''))")
  if [[ "$PRE_STATUS" == "PASS" ]]; then
    SKIP_PRE=1
  fi
fi

if [[ "$SKIP_PRE" == "1" ]]; then
  PRE_ID=$(/home/nacevedo/.conda/envs/fairness_env/bin/python -c "import json; print(json.load(open('$ROOT/preflight_report.json')).get('slurm_job_id','resume'))")
  echo "Skipping preflight (already PASS, job ${PRE_ID})"
else
  PRE_ID=$(sbatch --parsable -p "$PART" "$PRE")
  echo "Submitted preflight ${PRE_ID}"
fi

FAM_IDS=()
for METHOD in "${METHODS[@]}"; do
  if [[ "$SKIP_PRE" == "1" ]]; then
    ID=$(sbatch --parsable -p "$PART" --time="$TIME_FAM" \
      --job-name="toy_m_${METHOD:0:12}" \
      --export=ALL,METHOD="$METHOD" \
      "$FAM")
  else
    ID=$(sbatch --parsable -p "$PART" --time="$TIME_FAM" --dependency=afterok:"$PRE_ID" \
      --job-name="toy_m_${METHOD:0:12}" \
      --export=ALL,METHOD="$METHOD" \
      "$FAM")
  fi
  FAM_IDS+=("$ID")
done

DEP=$(IFS=:; echo "${FAM_IDS[*]}")
ASM_ID=$(sbatch --parsable -p "$PART" --dependency=afterany:"$DEP" "$ASM")

export PRE_ID ASM_ID PART DEP ROOT TIME_FAM SKIP_PRE
FAM_IDS_CSV=$(IFS=,; echo "${FAM_IDS[*]}")
export FAM_IDS_CSV

/home/nacevedo/.conda/envs/fairness_env/bin/python - <<'PY'
import json, os
from pathlib import Path
methods = [
    "current_direct",
    "direct_mm_k1",
    "moment_mm_k2",
    "moment_mm_k3",
    "local_slope_smooth",
    "quadratic",
]
ids = os.environ["FAM_IDS_CSV"].split(",")
payload = {
    "experiment_label": "EXPERIMENTAL / TOY / NON-CANONICAL",
    "experiment": "toy_mechanism_selection_v2",
    "partition": os.environ["PART"],
    "preflight_job_id": os.environ["PRE_ID"],
    "preflight_skipped": os.environ.get("SKIP_PRE") == "1",
    "family_job_ids": dict(zip(methods, ids)),
    "family_job_id_list": ids,
    "assemble_job_id": os.environ["ASM_ID"],
    "dependency": {
        "families": "none_preflight_already_pass" if os.environ.get("SKIP_PRE") == "1" else "afterok:" + os.environ["PRE_ID"],
        "assemble": "afterany:" + os.environ["DEP"],
    },
    "cpus_per_task_family": 8,
    "mem_gb_family": 12,
    "time_family": os.environ["TIME_FAM"],
    "gpus": 0,
    "cluster_resident": True,
    "resume_from_family_metrics": True,
    "note": "No local scientific fallback. Assemble runs even if a family fails.",
}
path = Path(os.environ["ROOT"]) / "manifests" / "mechanism_job_graph.json"
path.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
PY
