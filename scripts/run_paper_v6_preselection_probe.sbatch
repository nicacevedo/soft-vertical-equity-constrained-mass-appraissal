#!/bin/bash
#SBATCH --job-name=p6_probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/output/paper_v6_preselection/logs/probe_%x_%j.out
#SBATCH --error=/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/output/paper_v6_preselection/logs/probe_%x_%j.err

set -euo pipefail
cd /home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal
mkdir -p output/paper_v6_preselection/logs
export PYTHONUNBUFFERED=1
PART="${SLURM_JOB_PARTITION:-unknown}"
echo "PROBE partition=$PART node=${SLURMD_NODENAME:-} job=$SLURM_JOB_ID"
conda run -n fairness_env --no-capture-output python - <<'PY'
import sys
from pathlib import Path
print("python", sys.version.split()[0])
import lightgbm
print("lightgbm", lightgbm.__version__)
import dcor
print("dcor", dcor.__version__)
data = Path("data/CCAO/2025/training_data.parquet")
out = Path("output/paper_v6_preselection")
print("data_exists", data.is_file(), "size", data.stat().st_size if data.is_file() else 0)
print("out_exists", out.is_dir())
assert data.is_file(), "training parquet missing"
assert out.is_dir(), "output root missing"
print("PROBE_PASS")
PY
