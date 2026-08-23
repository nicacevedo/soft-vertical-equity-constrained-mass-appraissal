#!/bin/bash
# Waits for the rho-sweep array job to finish, verifies the q10 decile columns
# landed in all 12 experiment folders, then rebuilds the theory+empirical dashboard.
set -uo pipefail

REPO_ROOT="/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal"
PY="/home/nacevedo/.conda/envs/fairness_env/bin/python"
JOB_ID="${1:-15298685}"
ROOT="${REPO_ROOT}/output/rho_sweep_500_estimators"
cd "${REPO_ROOT}"

echo "[watch] waiting for SLURM job ${JOB_ID} to drain..."
for ((i=0; i<720; i++)); do          # up to 12h at 60s cadence
  n=$(squeue -j "${JOB_ID}" -h 2>/dev/null | wc -l | tr -d ' ')
  if [[ "${n}" == "0" ]]; then
    echo "[watch] job ${JOB_ID} no longer in queue (after ${i} min)"
    break
  fi
  sleep 60
done

# Verify every experiment folder has the new q10 decile median-ratio columns.
missing=$("${PY}" - <<'PYEOF'
import glob, os, sys
import pandas as pd
root = "output/rho_sweep_500_estimators"
bad = []
folders = [f for f in glob.glob(os.path.join(root, "*/")) if os.path.exists(os.path.join(f, "quick_test_metrics_assess.csv"))]
for f in sorted(folders):
    try:
        cols = pd.read_csv(os.path.join(f, "quick_test_metrics_assess.csv"), nrows=0).columns
    except Exception as exc:
        bad.append(f"{os.path.basename(f.rstrip('/'))}: read-error {exc}")
        continue
    if "MedianRatio_q10_bin10" not in cols:
        bad.append(f"{os.path.basename(f.rstrip('/'))}: no q10")
print(f"FOLDERS={len(folders)}")
for b in bad:
    print("BAD:", b)
sys.exit(1 if bad or len(folders) < 12 else 0)
PYEOF
)
status=$?
echo "${missing}"
if [[ ${status} -ne 0 ]]; then
  echo "DASHBOARD_REBUILD_FAILED: q10 columns missing or <12 folders ready"
  exit 1
fi

echo "[watch] all folders carry q10 columns; rebuilding dashboard..."
"${PY}" scripts/build_rho_sweep_dashboard_with_theory.py
rc=$?
if [[ ${rc} -eq 0 ]]; then
  echo "DASHBOARD_REBUILD_DONE: ${ROOT}/rho_sweep_dashboard.html"
else
  echo "DASHBOARD_REBUILD_FAILED: builder exit ${rc}"
fi
exit ${rc}
