#!/usr/bin/env python3
"""Step 14: freeze the v2 panel BEFORE any positive-rho artifacts.

Uses only pre-Direct/Surrogate information. A non-regressive baseline is a valid
null/boundary case. Direct/Surrogate require at least two independent new modeling
units to pass.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from analysis.berry_attom_validation_v2.scripts.v2_common import ANALYSIS, COUNTIES, OUTPUT

FREEZE = ANALYSIS / "panel_freeze" / "final_panel_freeze_v2.yaml"
METHOD = OUTPUT / "method_transfer"


def load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def classify(county_key: str, payload: dict) -> str:
    """Pre-rho classification. Never uses Direct/Surrogate outcomes."""
    if payload.get("n_model", 0) < 2000:
        return "EXCLUDE"
    if payload.get("linkage_selection_flag"):
        # Material change in Berry regressivity under ATTOM matching.
        if payload.get("baseline_r2_price") and payload["baseline_r2_price"] >= 0.40:
            return "INCLUDE_BOUNDARY"
        return "SOURCE_VALIDATION_ONLY"
    if county_key == "wayne":
        # Detroit is the Berry unit; Wayne is the AVM unit. That geographic
        # mismatch is a boundary condition even when the AVM is credible.
        if payload.get("baseline_r2_price", 0) >= 0.40 and payload.get("parcel_match_rate", 0) >= 0.50:
            return "INCLUDE_BOUNDARY"
        if payload.get("parcel_match_rate", 0) >= 0.20:
            return "SOURCE_VALIDATION_ONLY"
        return "EXCLUDE"
    if payload.get("baseline_r2_price", 0) >= 0.50 and payload.get("safe_history_rate", 0) >= 0.80:
        return "INCLUDE_PRIMARY"
    if payload.get("baseline_r2_price", 0) >= 0.40:
        return "INCLUDE_BOUNDARY"
    if payload.get("n_model", 0) >= 2000:
        return "SOURCE_VALIDATION_ONLY"
    return "EXCLUDE"


def main() -> int:
    if any(METHOD.glob("*")):
        raise SystemExit("Refusing to write a freeze file after method_transfer artifacts exist.")
    linkage = load_json(ANALYSIS / "linkage" / "linkage_summary.json") or []
    link_map = {row.get("key"): row for row in linkage if isinstance(row, dict)}
    units = []
    for c in COUNTIES:
        key = c["key"]
        base = load_json(ANALYSIS / "baselines" / key / "run_meta.json") or {}
        table = load_json(OUTPUT / "modeling_tables" / key / "modeling_table_meta.json") or {}
        link = link_map.get(key, {})
        payload = {
            "jurisdiction_key": key,
            "geographic_unit": c["label"],
            "berry_unit": c["berry_unit"],
            "attom_fips": c["fips"],
            "naming_rule": "NEVER label Wayne as Detroit" if key == "wayne" else (
                "NEVER conflate 29189 with 29510" if key == "st_louis_county" else ""
            ),
            "berry_n": link.get("berry_n"),
            "parcel_match_rate": link.get("parcel_match_rate"),
            "high_conf_transaction_match_rate": link.get("high_conf_rate"),
            "safe_history_rate": link.get("safe_history_rate_among_unique_apn"),
            "linkage_selection_flag": link.get("linkage_selection_flag") or (link.get("bias_summary") or {}).get("linkage_selection_flag"),
            "n_model": base.get("n") or table.get("n_property_use_385") or 0,
            "model_period": " — ".join(base.get("sale_date_range") or []),
            "baseline_r2_price": base.get("primary_test_R2_price"),
            "baseline_prb": base.get("primary_test_PRB"),
            "baseline_beta_log": base.get("primary_test_Beta_log"),
            "baseline_delta_nl": base.get("primary_test_Delta_NL"),
            "baseline_dcor": base.get("primary_test_dCor"),
            "modeling_table_sha256": table.get("table_sha256"),
            "recorder_cache_sha256": table.get("recorder_cache_sha256"),
            "history_cache_sha256": table.get("history_cache_sha256"),
        }
        decision = classify(key, payload)
        payload["panel_decision"] = decision
        payload["decision_uses_positive_rho"] = False
        payload["non_regressive_baseline_is_valid_null"] = True
        units.append(payload)

    passing = [u for u in units if u["panel_decision"] in {"INCLUDE_PRIMARY", "INCLUDE_BOUNDARY"}]
    authorize = len(passing) >= 2
    freeze = {
        "schema_version": 2,
        "frozen_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rule": (
            "Classification uses only pre-Direct/Surrogate information. "
            "Direct/Surrogate require at least TWO independent new modeling units "
            "with INCLUDE_PRIMARY or INCLUDE_BOUNDARY. If fewer than two pass, stop."
        ),
        "direct_surrogate_authorized": authorize,
        "n_passing_modeling_units": len(passing),
        "passing_units": [u["jurisdiction_key"] for u in passing],
        "units": units,
        "existing_six_attom": {
            "role": "separate_exploratory_sensitivity",
            "do_not_overwrite": True,
            "not_part_of_this_freeze": True,
        },
        "if_not_authorized": "Preserve audit + baselines as the final v2 scientific outcome.",
    }
    FREEZE.parent.mkdir(parents=True, exist_ok=True)
    FREEZE.write_text(yaml.safe_dump(freeze, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(yaml.safe_dump({"authorized": authorize, "passing": freeze["passing_units"]}, sort_keys=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
