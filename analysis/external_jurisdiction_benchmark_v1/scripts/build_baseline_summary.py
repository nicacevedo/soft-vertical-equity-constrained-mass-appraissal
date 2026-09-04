#!/usr/bin/env python3
"""Step 4: aggregate all_jurisdiction_baseline_summary.csv from each county's
{key}_baseline_cv_summary.csv (full metric set) filtered to the selected
config, plus role assignment from history_source_resolution.yaml.

Role assignment (predeclared, not outcome-driven):
- PRIMARY_FULL_7_FOLD: n_folds_used == 7
- otherwise, the label recorded in history_source_resolution.yaml's
  fallback_roles, or the protocol-predeclared default for Cook/Allegheny.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from analysis.external_jurisdiction_benchmark_v1.scripts.v1_common import ALL_KEYS, ANALYSIS, JURISDICTION_BY_KEY  # noqa: E402


def role_for(key: str, n_folds: int, resolution: dict) -> str:
    if n_folds >= 7:
        return "PRIMARY_FULL_7_FOLD"
    fallback = resolution.get("fallback_roles", {}) if resolution else {}
    if key in fallback:
        return fallback[key]
    if key == "cook":
        return "SHORT_HISTORY_BRIDGE_SENSITIVITY"
    if key == "allegheny":
        return "TEMPORAL_COVERAGE_SENSITIVITY_ONLY"
    return "SHORT_HISTORY_PROVISIONAL"


def main() -> int:
    resolution_path = ANALYSIS / "audits" / "history_source_resolution.yaml"
    resolution = yaml.safe_load(resolution_path.read_text()) if resolution_path.exists() else {}

    rows = []
    for key in ALL_KEYS:
        config_path = ANALYSIS / "baseline" / f"{key}_baseline_config.json"
        summary_path = ANALYSIS / "baseline" / f"{key}_baseline_cv_summary.csv"
        table_meta_path = ROOT / "output/external_jurisdiction_benchmark_v1/modeling_tables" / key / "modeling_table_meta_dev.json"
        if not (config_path.exists() and summary_path.exists()):
            rows.append({"county_key": key, "label": JURISDICTION_BY_KEY[key]["label"], "status": "NOT_YET_RUN"})
            continue
        cfg = json.loads(config_path.read_text())
        summary = pd.read_csv(summary_path)
        sel = summary.loc[summary["config_name"] == cfg["selected_lgbm_config"]].iloc[0]
        n_folds = int(sel["n_folds"])
        model_n = None
        if table_meta_path.exists():
            model_n = json.loads(table_meta_path.read_text()).get("n_final_primary_residential")
        rows.append({
            "county_key": key, "label": JURISDICTION_BY_KEY[key]["label"], "status": cfg.get("status"),
            "role": role_for(key, n_folds, resolution),
            "model_N": model_n, "n_folds_completed": n_folds,
            "selected_lgbm_config": cfg.get("selected_lgbm_config"),
            "R2_price": sel["mean_R2_price"], "R2_log": sel.get("mean_R2_log"), "NMSE": sel["mean_NMSE"],
            "RMSE_log": sel["mean_RMSE_log"], "MAE": sel["mean_MAE"], "MAPE": sel["mean_MAPE"],
            "COD": sel["mean_COD"], "PRD": sel["mean_PRD"], "PRB": sel["mean_PRB"], "MKI": sel["mean_MKI"],
            "VEI": sel["mean_VEI"], "beta_log": sel["mean_Beta_log"], "Delta_NL": sel["mean_Delta_NL"],
            "dCor": sel["mean_dCor_e_y"],
            "structural_core_R2_price": cfg.get("structural_core_cv_mean_R2_price"),
        })
    df = pd.DataFrame(rows)
    ANALYSIS.joinpath("baseline").mkdir(parents=True, exist_ok=True)
    df.to_csv(ANALYSIS / "baseline" / "all_jurisdiction_baseline_summary.csv", index=False)
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
