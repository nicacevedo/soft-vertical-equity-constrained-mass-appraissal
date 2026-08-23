#!/bin/bash
# =============================================================================
# theory_rho_range_experiments.sh
#
# SLURM job-array driver for the theory-informed rho-range analysis.
# Each array task runs ONE independent experiment = (dataset, LGBM baseline
# config). This mirrors rho_sweep_experiments.sh, but it does NOT refit the
# covariance-penalty sweep. Instead, it fits/recycles the unpenalized LGBM
# baseline predictions and computes theory-implied rho bands for LGBCovPenalty[diff].
#
# Outputs per task are self-contained and crash-safe:
#   ${OUT_ROOT}/${src}_assess${year}__${cfg}_${cid}/
#     checkpoints/*__summary.csv, *__shrinkage.csv, *__prd_targets.csv, ...
#     plots/per_run/*__theory_tradeoff.png
#     theory_rho_summary_by_run.csv
#     theory_rho_report.md
#
# Usage:
#   sbatch scripts/theory_rho_range_experiments.sh              # all 12 tasks
#   sbatch --array=0-2 scripts/theory_rho_range_experiments.sh  # subset
#   bash scripts/theory_rho_range_experiments.sh list           # print matrix
#   THEORY_TASK_ID=0 bash scripts/theory_rho_range_experiments.sh
#   bash scripts/theory_rho_range_experiments.sh merge          # merge finished task outputs
# =============================================================================
#SBATCH --job-name=theory_rho
#SBATCH --partition=ou_sloan_batch # mit_normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96GB
#SBATCH --array=0-11
#SBATCH --output=temp/logs/theory_rho_%A_%a.out
#SBATCH --error=temp/logs/theory_rho_%A_%a.err
#SBATCH -t 0-04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nacevedo@mit.edu

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal}"
PY="${PY:-/home/nacevedo/.conda/envs/fairness_env/bin/python}"
SCRIPT="${SCRIPT:-scripts/theory_informed_rho_range_v2.py}"
cd "${REPO_ROOT}"

# --- Theory / model knobs ----------------------------------------------------
LGBM_N_ESTIMATORS="${LGBM_N_ESTIMATORS:-500}"
SAMPLE_FRAC="${SAMPLE_FRAC:-1}"
SEED="${SEED:-4050}"
OUT_ROOT="${OUT_ROOT:-output/theory_rho_range_500_estimators}"
RHO_SWEEP_ROOT="${RHO_SWEEP_ROOT:-output/rho_sweep_500_estimators}"
BASELINE_CACHE_ROOT="${BASELINE_CACHE_ROOT:-${OUT_ROOT}/baseline_cache_n${LGBM_N_ESTIMATORS}}"
HYPERPARAM_FILE="${HYPERPARAM_FILE:-best_lgbm_baseline_configs.yaml}"
PARAMS_PATH="${PARAMS_PATH:-params.yaml}"
MODEL_PARAMS_PATH="${MODEL_PARAMS_PATH:-model_params.yaml}"
EMPIRICAL_RHO_RANGE="${EMPIRICAL_RHO_RANGE:-2.56,3.54}"
SHRINKAGE_Q_VALUES="${SHRINKAGE_Q_VALUES:-0.75,0.50,0.33,0.25}"
PRD_TARGETS="${PRD_TARGETS:-1.03,1.02,1.01,0.99,0.98}"
ACCURACY_BUDGETS="${ACCURACY_BUDGETS:-0.001,0.005,0.01,0.02}"
PLOT_FORMAT="${PLOT_FORMAT:-png}"

# --- Experiment matrix -------------------------------------------------------
SRC_LABELS=(ccao2025 ccao_old ccao_sim2024 ccao_sim2023)
SRC_PATHS=(
  "data/CCAO/2025/training_data.parquet"
  "data/CCAO/2025/training_data_old.parquet"
  "data/CCAO/2025/training_data_sim2024.parquet"
  "data/CCAO/2025/training_data_sim2023.parquet"
)
SRC_ASSESS_YEARS=(2025 2024 2023 2022)

CFG_KEYS=(test_best_r2 cv_top1_r2 cv_top2_r2)
CFG_IDS=(dee08fa9 c6fc2c3b a1c87203)

N_DATASETS=${#SRC_LABELS[@]}
N_CONFIGS=${#CFG_KEYS[@]}
N_TOTAL=$(( N_DATASETS * N_CONFIGS ))

derive() {  # $1 = task id -> sets global EXP_* variables
  local tid="$1"
  local d=$(( tid / N_CONFIGS ))
  local c=$(( tid % N_CONFIGS ))
  EXP_SRC="${SRC_LABELS[$d]}"
  EXP_PATH="${SRC_PATHS[$d]}"
  EXP_YEAR="${SRC_ASSESS_YEARS[$d]}"
  EXP_KEY="${CFG_KEYS[$c]}"
  EXP_CID="${CFG_IDS[$c]}"
  EXP_OUT="${OUT_ROOT}/${EXP_SRC}_assess${EXP_YEAR}__${EXP_KEY}_${EXP_CID}"
  EXP_CACHE="${BASELINE_CACHE_ROOT}/${EXP_SRC}_assess${EXP_YEAR}__${EXP_KEY}_${EXP_CID}"
}

if [[ "${1:-}" == "list" ]]; then
  echo "theory-rho experiment matrix (${N_TOTAL} tasks):"
  for ((t=0; t<N_TOTAL; t++)); do
    derive "$t"
    printf "  [%2d] src=%-13s assess=%s  cfg=%-13s -> %s\n" "$t" "$EXP_SRC" "$EXP_YEAR" "$EXP_KEY" "$EXP_OUT"
  done
  exit 0
fi

if [[ "${1:-}" == "merge" ]]; then
  mkdir -p "${OUT_ROOT}/merged"
  "${PY}" - <<'PYMERGE' "${OUT_ROOT}"
import sys
from pathlib import Path
import numpy as np
import pandas as pd

root = Path(sys.argv[1])
merged = root / "merged"
merged.mkdir(parents=True, exist_ok=True)

files = {
    "summary": "theory_rho_summary_by_run.csv",
    "shrinkage": "theory_rho_shrinkage_targets.csv",
    "prd": "theory_rho_prd_targets.csv",
    "budget": "theory_rho_accuracy_budgets.csv",
    "comparison": "theory_empirical_comparison.csv",
    "ops": "theory_empirical_operating_points.csv",
}

def read_all(name):
    rows = []
    for p in sorted(root.glob(f"*/{name}")):
        if p.parent.name == "merged":
            continue
        try:
            df = pd.read_csv(p)
        except Exception as exc:
            print(f"[merge] skipping {p}: {exc}")
            continue
        df["task_output_dir"] = str(p.parent)
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

out = {}
for key, fname in files.items():
    df = read_all(fname)
    out[key] = df
    if not df.empty:
        df.to_csv(merged / fname, index=False)
        print(f"[merge] wrote {key}: rows={df.shape[0]} -> {merged / fname}")

summary = out.get("summary", pd.DataFrame())
if not summary.empty:
    rows = []
    for split_name, d in [("all", summary), *list(summary.groupby("split"))]:
        if isinstance(split_name, tuple):
            split_name = str(split_name[0])
        low = pd.to_numeric(d.get("theory_range_low"), errors="coerce")
        high = pd.to_numeric(d.get("theory_range_high"), errors="coerce")
        conf = pd.to_numeric(d.get("theory_confident_rho"), errors="coerce")
        rho50 = pd.to_numeric(d.get("rho_shrink_50pct"), errors="coerce")
        mse_band_low = 0.25 * rho50
        mse_ref = 0.50 * rho50
        mse_band_high = rho50
        cov_band_low = rho50 / 3.0
        cov_ref = rho50
        cov_band_high = 3.0 * rho50
        overlap_low = pd.concat([mse_band_low, cov_band_low], axis=1).max(axis=1)
        overlap_high = pd.concat([mse_band_high, cov_band_high], axis=1).min(axis=1)
        overlap_ref = np.sqrt(overlap_low * overlap_high).where(
            (overlap_low > 0.0) & (overlap_high >= overlap_low)
        )
        robust_low = float(np.nanquantile(low, 0.75)) if low.notna().any() else np.nan
        robust_high = float(np.nanquantile(high, 0.25)) if high.notna().any() else np.nan
        if np.isfinite(robust_low) and np.isfinite(robust_high) and robust_low > robust_high:
            robust_low = float(np.nanmedian(low)) if low.notna().any() else np.nan
            robust_high = float(np.nanmedian(high)) if high.notna().any() else np.nan
        rows.append({
            "split_group": str(split_name),
            "n_runs": int(d.shape[0]),
            "robust_theory_range_low": robust_low,
            "median_confident_rho": float(np.nanmedian(conf)) if conf.notna().any() else np.nan,
            "robust_theory_range_high": robust_high,
            "median_rho_shrink_25pct": float(np.nanmedian(pd.to_numeric(d.get("rho_shrink_25pct"), errors="coerce"))),
            "median_rho_shrink_50pct": float(np.nanmedian(rho50)),
            "median_rho_shrink_67pct": float(np.nanmedian(pd.to_numeric(d.get("rho_shrink_67pct"), errors="coerce"))),
            "median_orange_mse_band_low": float(np.nanmedian(mse_band_low)),
            "median_orange_mse_ref_rho": float(np.nanmedian(mse_ref)),
            "median_orange_mse_band_high": float(np.nanmedian(mse_band_high)),
            "median_orange_cov_band_low": float(np.nanmedian(cov_band_low)),
            "median_orange_cov_ref_rho": float(np.nanmedian(cov_ref)),
            "median_orange_cov_band_high": float(np.nanmedian(cov_band_high)),
            "median_orange_overlap_low": float(np.nanmedian(overlap_low)),
            "median_orange_overlap_ref_rho": float(np.nanmedian(overlap_ref)),
            "median_orange_overlap_high": float(np.nanmedian(overlap_high)),
            "median_bayes_diagnostic_C0_over_minus_B": float(np.nanmedian(pd.to_numeric(d.get("bayes_optimality_diagnostic_C0_over_minus_B"), errors="coerce"))),
            "empirical_overlap_rate": float(pd.Series(d.get("empirical_range_overlaps_theory", np.nan)).mean()) if "empirical_range_overlaps_theory" in d else np.nan,
        })
    agg = pd.DataFrame(rows)
    agg.to_csv(merged / "theory_rho_aggregate_recommendation.csv", index=False)
    report = ["# Merged theory-informed rho-range report", "", "## Aggregate recommendation", ""]
    report.append(agg.to_markdown(index=False, floatfmt=".4f"))
    report.append("")
    if not agg.empty and "median_orange_overlap_low" in agg.columns:
        report.append("## Orange-band overlap recommendation")
        report.append("")
        orange_cols = [
            "split_group",
            "median_orange_mse_band_low", "median_orange_mse_ref_rho", "median_orange_mse_band_high",
            "median_orange_cov_band_low", "median_orange_cov_ref_rho", "median_orange_cov_band_high",
            "median_orange_overlap_low", "median_orange_overlap_ref_rho", "median_orange_overlap_high",
        ]
        report.append(agg.loc[:, [c for c in orange_cols if c in agg.columns]].to_markdown(index=False, floatfmt=".4f"))
        report.append("")
        all_row = agg.loc[agg["split_group"].eq("all")]
        if not all_row.empty:
            r = all_row.iloc[0]
            report.append(
                f"Recommended orange-overlap rho range: "
                f"[{r['median_orange_overlap_low']:.4f}, {r['median_orange_overlap_high']:.4f}], "
                f"with reference rho {r['median_orange_overlap_ref_rho']:.4f}."
            )
            report.append("")
    report.append("## Inputs merged")
    for key, df in out.items():
        report.append(f"- {key}: {df.shape[0]} rows")
    (merged / "theory_rho_merged_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"[merge] wrote aggregate/report -> {merged}")
else:
    print("[merge] no summary files found")
PYMERGE
  echo "[merge] building global merged theory-vs-empirical rho-evolution plots"
  "${PY}" - <<'PYMERGEPLOTS' "${OUT_ROOT}" "${SCRIPT}"
import importlib.util
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
script_path = Path(sys.argv[2])
merged = root / "merged"

comparison_path = merged / "theory_empirical_comparison.csv"
if not comparison_path.exists():
    print(f"[merge-plots] missing {comparison_path}; skipping global plots")
    raise SystemExit(0)

spec = importlib.util.spec_from_file_location("theory_rho_module", script_path)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)

comparison_df = pd.read_csv(comparison_path)
plot_dir = merged / "plots" / "rho_evolution_theory_empirical_global"
paths = mod.plot_rho_evolution_theory_empirical_overlays(
    comparison_df,
    plot_dir,
    splits=("assessment", "test"),
)

print(f"[merge-plots] wrote {len(paths)} global rho-evolution overlay plots")
for path in paths:
    print(f"[merge-plots] {path}")
PYMERGEPLOTS
  exit 0
fi

# Resolve which task to run: SLURM array id, explicit env, or first CLI arg.
TASK_ID="${SLURM_ARRAY_TASK_ID:-${THEORY_TASK_ID:-${1:-}}}"
if [[ -z "${TASK_ID}" ]]; then
  echo "ERROR: no task id. Run under sbatch (--array) or pass THEORY_TASK_ID / a CLI index." >&2
  exit 2
fi
if (( TASK_ID < 0 || TASK_ID >= N_TOTAL )); then
  echo "ERROR: task id ${TASK_ID} out of range [0, $((N_TOTAL-1))]." >&2
  exit 2
fi

# --- Thread hygiene ----------------------------------------------------------
# This theory job fits at most one LightGBM at a time inside each array task, so
# give LightGBM the allocated cores. BLAS libraries stay single-threaded to avoid
# accidental nested oversubscription.
CPUS="${SLURM_CPUS_PER_TASK:-$(nproc)}"
export OMP_NUM_THREADS="${CPUS}"
export LGBM_NUM_THREADS="${CPUS}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p temp/logs "${OUT_ROOT}" "${BASELINE_CACHE_ROOT}"
derive "${TASK_ID}"

echo "==============================================================="
echo "[theory_rho] task=${TASK_ID}/${N_TOTAL} started $(date)"
echo "[theory_rho] host=$(hostname) cpus=${CPUS} python=${PY}"
echo "[theory_rho] script=${SCRIPT}"
echo "[theory_rho] source=${EXP_SRC} data=${EXP_PATH} assess_year=${EXP_YEAR}"
echo "[theory_rho] lgbm_config=${EXP_KEY} (id=${EXP_CID}) n_estimators=${LGBM_N_ESTIMATORS}"
echo "[theory_rho] empirical_rho_range=${EMPIRICAL_RHO_RANGE}"
echo "[theory_rho] out_dir=${EXP_OUT}"
echo "==============================================================="

if [[ ! -x "${PY}" ]]; then
  echo "ERROR: Python executable not available: ${PY}" >&2
  exit 127
fi
if [[ ! -f "${SCRIPT}" ]]; then
  echo "ERROR: theory script not found: ${SCRIPT}" >&2
  exit 2
fi
if [[ ! -f "${EXP_PATH}" ]]; then
  echo "ERROR: data file not found: ${EXP_PATH}" >&2
  exit 2
fi

set +e
"${PY}" "${SCRIPT}" \
  --data-source-specs "${EXP_SRC}:${EXP_YEAR}:${EXP_PATH}" \
  --lgbm-config-keys "${EXP_KEY}" \
  --lgbm-hyperparameter-file "${HYPERPARAM_FILE}" \
  --lgbm-n-jobs "${CPUS}" \
  --lgbm-n-estimators "${LGBM_N_ESTIMATORS}" \
  --params-path "${PARAMS_PATH}" \
  --model-params-path "${MODEL_PARAMS_PATH}" \
  --out-dir "${EXP_OUT}" \
  --baseline-cache-dir "${EXP_CACHE}" \
  --rho-sweep-root "${RHO_SWEEP_ROOT}" \
  --sample-frac "${SAMPLE_FRAC}" \
  --seed "${SEED}" \
  --empirical-rho-range "${EMPIRICAL_RHO_RANGE}" \
  --shrinkage-q-values "${SHRINKAGE_Q_VALUES}" \
  --prd-targets "${PRD_TARGETS}" \
  --accuracy-budgets "${ACCURACY_BUDGETS}" \
  --plot-format "${PLOT_FORMAT}" \
  --write-intermediate-results \
  --write-intermediate-plots

ec=$?
echo "[theory_rho] task=${TASK_ID} (${EXP_SRC}/${EXP_KEY}) finished $(date) exit=${ec}"
exit ${ec}
