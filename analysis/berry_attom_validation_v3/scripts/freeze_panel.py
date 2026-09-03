#!/usr/bin/env python3
"""Two-dimension panel freeze. Uses validation metrics only. No held-out test. No rho."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from analysis.berry_attom_validation_v3.scripts.v3_common import ANALYSIS, COUNTIES, OUTPUT

FREEZE = ANALYSIS / "panel_freeze" / "final_panel_freeze_v3.yaml"
METHOD = OUTPUT / "method_transfer"
HELD = OUTPUT / "final_models"


def load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def model_transfer_status(payload: dict) -> str:
    if payload.get("fatal_construction"):
        return "NOT_ELIGIBLE"
    n = payload.get("n_model") or 0
    r2 = payload.get("validation_r2")
    if n < 2000 or r2 is None:
        return "NOT_ELIGIBLE"
    if payload.get("test_block_scored"):
        return "NOT_ELIGIBLE"
    if r2 >= 0.50 and n >= 5000:
        return "PRIMARY"
    if r2 >= 0.35:
        return "BOUNDARY"
    return "NOT_ELIGIBLE"


def berry_anchor_status(payload: dict) -> str:
    if payload.get("key") == "st_louis_county" and not payload.get("berry_has_assessed_value"):
        # Local sales exist; official assessment-ratio benchmark is not reproduced.
        base = "WEAK"
    else:
        base = "MODERATE"
    r_full = payload.get("r_fully_validated_unconditional") or 0
    prb_flip = payload.get("prb_sign_flip")
    if payload.get("berry_reproduction") == "PYTHON_TRANSLATION_MATCHES_V1_FILTERS" and r_full >= 0.4 and not prb_flip:
        return "STRONG"
    if payload.get("berry_reproduction") == "RESOLVED_USE_ARMS_LENGTH_FILE_NOT_RBIND" and r_full >= 0.2:
        return "MODERATE" if not prb_flip else "WEAK"
    if payload.get("key") == "st_louis_county":
        return "WEAK" if r_full < 0.2 else "MODERATE"
    return base


def main() -> int:
    if any(METHOD.glob("*")) or any(HELD.glob("*")):
        raise SystemExit("Refusing to freeze after held-out or method_transfer artifacts exist.")
    link = load_json(ANALYSIS / "linkage" / "linkage_summary.json") or []
    link_map = {row.get("key"): row for row in link if isinstance(row, dict)}
    units = []
    for c in COUNTIES:
        key = c["key"]
        base = load_json(ANALYSIS / "baselines_pre_freeze" / key / "run_meta.json") or {}
        table = load_json(OUTPUT / "modeling_tables" / key / "modeling_table_meta.json") or {}
        lk = link_map.get(key, {})
        val = (base.get("validation_lgbm_HISTORY_MARKET_CORE") or {})
        payload = {
            "key": key,
            "attom_unit": c["label"],
            "berry_unit": c["berry_unit"],
            "attom_fips": c["fips"],
            "naming_rule": "NEVER label Wayne as Detroit" if key == "wayne" else (
                "NEVER conflate 29189 with 29510" if key == "st_louis_county" else ""
            ),
            "n_model": table.get("n_final_model") or 0,
            "validation_r2": val.get("R2"),
            "validation_prb": val.get("PRB"),
            "validation_beta_log": val.get("Beta_log"),
            "test_block_scored": bool(base.get("test_block_scored")),
            "fatal_construction": (not table) or table.get("status") not in {None, "OK"},
            "retention_decile_spread": table.get("decile_p_final_spread"),
            "r_apn_unconditional": lk.get("r_apn_unconditional"),
            "r_transaction_unconditional": lk.get("r_transaction_unconditional"),
            "r_safe_history_unconditional": lk.get("r_safe_history_unconditional"),
            "r_fully_validated_unconditional": lk.get("r_fully_validated_unconditional"),
            "berry_PRB_eligible": lk.get("berry_PRB_eligible"),
            "berry_PRB_safe_history": lk.get("berry_PRB_safe_history"),
            "berry_n_eligible_linkage": lk.get("berry_n_eligible_linkage"),
            "berry_has_assessed_value": key != "st_louis_county",
            "berry_reproduction": {
                "wayne": "PYTHON_TRANSLATION_MATCHES_V1_FILTERS",
                "philadelphia": "RESOLVED_USE_ARMS_LENGTH_FILE_NOT_RBIND",
                "st_louis_county": "SALES_EXTRACT_NOT_FULL_BERRY_RATIO_BENCHMARK",
            }[key],
            "prb_sign_flip": (
                lk.get("berry_PRB_eligible") is not None
                and lk.get("berry_PRB_safe_history") is not None
                and lk.get("berry_PRB_eligible") != 0
                and lk.get("berry_PRB_safe_history") != 0
                and (lk.get("berry_PRB_eligible") or 0) * (lk.get("berry_PRB_safe_history") or 0) < 0
            ),
        }
        payload["MODEL_TRANSFER_STATUS"] = model_transfer_status(payload)
        payload["BERRY_ANCHOR_STATUS"] = berry_anchor_status(payload)
        payload["uses_heldout_test"] = False
        payload["uses_positive_rho"] = False
        units.append(payload)

    passing = [u for u in units if u["MODEL_TRANSFER_STATUS"] in {"PRIMARY", "BOUNDARY"}]
    authorize = len(passing) >= 2
    freeze = {
        "schema_version": 3,
        "frozen_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "direct_surrogate_authorized": authorize,
        "n_model_transfer_primary_or_boundary": len(passing),
        "passing_model_transfer_units": [u["key"] for u in passing],
        "if_not_authorized": "STOP. Finalize v3 as source-validation + baseline/provider-robustness. Do not force penalty results.",
        "heldout_test_not_used": True,
        "units": units,
        "existing_six_attom": {"role": "separate_exploratory_sensitivity", "do_not_overwrite": True},
    }
    FREEZE.parent.mkdir(parents=True, exist_ok=True)
    FREEZE.write_text(yaml.safe_dump(freeze, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(yaml.safe_dump({"authorized": authorize, "passing": freeze["passing_model_transfer_units"]}, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
