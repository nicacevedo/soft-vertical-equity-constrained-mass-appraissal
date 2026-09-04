#!/usr/bin/env python3
"""Step 4: freeze BASELINE_FREEZE.yaml. No penalty path may begin before this
file exists. Written only after all_jurisdiction_baseline_summary.csv exists
for every jurisdiction that will be run."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from analysis.external_jurisdiction_benchmark_v1.scripts.v1_common import (  # noqa: E402
    ALL_KEYS, ANALYSIS, JURISDICTION_BY_KEY, OUTPUT, sha256_file, write_json,
)


def main() -> int:
    summary_path = ANALYSIS / "baseline" / "all_jurisdiction_baseline_summary.csv"
    if not summary_path.exists():
        raise SystemExit("all_jurisdiction_baseline_summary.csv missing; run build_baseline_summary.py first")
    summary = pd.read_csv(summary_path)

    resolution_path = ANALYSIS / "audits" / "history_source_resolution.yaml"
    resolution = yaml.safe_load(resolution_path.read_text()) if resolution_path.exists() else {}

    units = []
    for key in ALL_KEYS:
        row = summary.loc[summary.county_key == key]
        if not len(row) or row.iloc[0].get("status") != "OK":
            units.append({"county_key": key, "status": row.iloc[0].get("status") if len(row) else "NOT_RUN"})
            continue
        row = row.iloc[0]
        j = JURISDICTION_BY_KEY[key]
        cache_manifest_path = OUTPUT / "cache" / key / "cache_manifest.json"
        table_meta_path = OUTPUT / "modeling_tables" / key / "modeling_table_meta_dev.json"
        cache_manifest = json.loads(cache_manifest_path.read_text()) if cache_manifest_path.exists() else []
        table_meta = json.loads(table_meta_path.read_text()) if table_meta_path.exists() else {}
        canonical_source = resolution.get("canonical_sources", {}).get(key, str(j["assessor_dir"]))
        units.append({
            "county_key": key, "label": j["label"], "role": row["role"], "status": "OK",
            "canonical_history_source": canonical_source,
            "cache_hashes": {c.get("product"): c.get("sha256") for c in cache_manifest},
            "modeling_table_hash": table_meta.get("table_sha256"),
            "development_period": table_meta.get("development_period"),
            "n_folds_completed": int(row["n_folds_completed"]),
            "selected_lgbm_config": row["selected_lgbm_config"],
            "cv_metrics": {
                m: (None if pd.isna(row.get(m)) else float(row.get(m)))
                for m in ["R2_price", "R2_log", "NMSE", "RMSE_log", "MAE", "MAPE",
                          "COD", "PRD", "PRB", "MKI", "VEI", "beta_log", "Delta_NL", "dCor"]
                if m in row.index
            },
        })

    freeze = {
        "schema_version": 1, "status": "FROZEN",
        "frozen_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rule": "No penalty path (Direct/Surrogate) may begin before this file exists.",
        "history_source_resolution_file": str(resolution_path) if resolution_path.exists() else None,
        "units": units,
    }
    ANALYSIS.joinpath("baseline").mkdir(parents=True, exist_ok=True)
    out_path = ANALYSIS / "baseline" / "BASELINE_FREEZE.yaml"
    out_path.write_text(yaml.safe_dump(freeze, sort_keys=False, default_flow_style=False))
    print(f"wrote {out_path}, sha256={sha256_file(out_path)}")
    print(f"units OK: {sum(1 for u in units if u.get('status') == 'OK')} / {len(units)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
