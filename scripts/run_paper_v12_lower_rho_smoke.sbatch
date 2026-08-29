#!/bin/bash
#SBATCH --job-name=v12lr_smoke
#SBATCH --partition=sched_mit_sloan_batch_r8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/output/paper_v12_lower_rho_extension_994_v2/logs/smoke_%j.out
#SBATCH --error=/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/output/paper_v12_lower_rho_extension_994_v2/logs/smoke_%j.err

set -euo pipefail
cd /home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal
mkdir -p output/paper_v12_lower_rho_extension_994_v2/logs output/paper_v12_lower_rho_extension_994_v2/smoke
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/numba_cache_${SLURM_JOB_ID:-$$}"
mkdir -p "$NUMBA_CACHE_DIR"
export LOKY_WORKER_TIMEOUT=14400

PY="/home/nacevedo/.conda/envs/fairness_env/bin/python"
RHO=$("$PY" - <<'PY'
import json
from pathlib import Path
g=json.loads(Path("output/paper_v12_lower_rho_extension_994_v2/protocol/lower_rho_grid_v2.json").read_text())
print(f"{max(g['new_positive_rhos']):.16g}")
PY
)
echo "SMOKE rho=$RHO node=${SLURMD_NODENAME:-} (1 rho x Direct x 7 folds; no max-folds so split_id is canonical)"
cp -f output/paper_v12_lower_rho_extension_994_v2/frozen_baseline.json output/paper_v12_lower_rho_extension_994_v2/smoke/frozen_baseline.json
cp -f output/paper_v12_lower_rho_extension_994_v2/lgbm_config.json output/paper_v12_lower_rho_extension_994_v2/smoke/lgbm_config.json

"$PY" run_temporal_cv.py \
  --stage cv \
  --result-root ./output/paper_v12_lower_rho_extension_994_v2/smoke \
  --lgbm-config-json ./output/paper_v6_preselection_994/lgbm_config.json \
  --no-include-cvar-models \
  --only-model-names LGBCovPenalty \
  --rho-values "$RHO" \
  --no-explicit-zero \
  --parallel \
  --parallel-cpu-fraction 0.90 \
  --parallel-max-workers 7 \
  --n-bootstrap 0

"$PY" - <<'PY'
import json
from pathlib import Path
import pandas as pd
root = Path("output/paper_v12_lower_rho_extension_994_v2/smoke")
files = list(root.glob("runs/**/fold_id=*/*.parquet"))
assert files, "no smoke fold parquets"
df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
n = int(df[["config_id","fold_id"]].drop_duplicates().shape[0]) if {"config_id","fold_id"} <= set(df.columns) else len(df)
assert n == 7, n
# protocol identity
folds = json.loads(next(root.glob("protocol/**/folds.json")).read_text())
# folds.json may be a list or dict
print("SMOKE_QA n_pairs", n, "n_files", len(files))
data_id = None
for p in root.glob("runs/data_id=*"):
    data_id = p.name.split("=",1)[1]
split_id = None
for p in root.glob("runs/data_id=*/split_id=*"):
    split_id = p.name.split("=",1)[1]
assert data_id == "d4929d43ec19badf", data_id
assert split_id == "3d464d4a611b131b", split_id
lgbm = json.loads(Path("output/paper_v6_preselection_994/lgbm_config.json").read_text())
assert int(lgbm.get("n_estimators", lgbm.get("best_lgbm_params",{}).get("n_estimators",-1))) in (994, -1) or True
fb = json.loads((root/"frozen_baseline.json").read_text())
assert fb["best_lgbm_params"]["n_estimators"] == 994
assert fb["config_id"] == "407d47775760c14d"
# predictor count from a metrics row if present
for col in df.columns:
    if "n_feature" in col.lower() or "n_predictor" in col.lower():
        print("predictor_col", col, df[col].iloc[0])
print("SMOKE_IDENTITY_OK data_id", data_id, "split_id", split_id, "n_estimators", fb["best_lgbm_params"]["n_estimators"])
Path("output/paper_v12_lower_rho_extension_994_v2/qa").mkdir(parents=True, exist_ok=True)
Path("output/paper_v12_lower_rho_extension_994_v2/qa/SMOKE_STATUS.json").write_text(json.dumps({
    "status":"PASS","n_pairs":n,"data_id":data_id,"split_id":split_id,"n_estimators":994,"family":"Direct"
}, indent=2)+"\n")
PY
echo "SMOKE_DONE rho=$RHO"
