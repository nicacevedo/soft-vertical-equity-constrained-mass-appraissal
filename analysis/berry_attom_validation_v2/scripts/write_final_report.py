#!/usr/bin/env python3
"""Write reports/FINAL_V2_REPORT.md from whatever v2 artifacts currently exist.

Safe to re-run. Does not edit the manuscript. Does not invent metrics.
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from analysis.berry_attom_validation_v2.scripts.v2_common import ANALYSIS, COUNTIES, OUTPUT

REPORT = ANALYSIS / "reports" / "FINAL_V2_REPORT.md"


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def cell(value) -> str:
    if value is None or value == "":
        return "pending"
    return str(value)


def main() -> int:
    freeze = {}
    fp = ANALYSIS / "panel_freeze" / "final_panel_freeze_v2.yaml"
    if fp.exists():
        freeze = yaml.safe_load(fp.read_text()) or {}
    freeze_units = {u["jurisdiction_key"]: u for u in freeze.get("units", [])}
    link = {row.get("key"): row for row in (read_json(ANALYSIS / "linkage" / "linkage_summary.json") or []) if isinstance(row, dict)}
    repro = {r.get("jurisdiction"): r for r in read_csv(ANALYSIS / "berry_reproduction" / "reproduction_summary.csv")}
    inv_cov = read_csv(ANALYSIS / "inventory" / "field_coverage.csv")
    inv_counts = read_csv(ANALYSIS / "inventory" / "fips_year_counts.csv")
    method = read_csv(ANALYSIS / "method_transfer" / "all_metrics.csv")
    authorized = bool(freeze.get("direct_surrogate_authorized"))
    rows = []
    for c in COUNTIES:
        key = c["key"]
        berry_key = {"wayne": "detroit_mi", "philadelphia": "philadelphia_pa", "st_louis_county": "st_louis_county_mo"}[key]
        r = repro.get(berry_key, {})
        b = read_json(ANALYSIS / "baselines" / key / "run_meta.json") or {}
        t = read_json(OUTPUT / "modeling_tables" / key / "modeling_table_meta.json") or {}
        u = freeze_units.get(key, {})
        lk = link.get(key, {})
        rows.append({
            "jurisdiction": c["label"],
            "geographic_unit": c["label"],
            "berry_role": c["berry_unit"],
            "berry_reproduction_status": r.get("status", "pending"),
            "berry_n": r.get("n") or lk.get("berry_n"),
            "attom_recorder_n": t.get("n_recorder_raw"),
            "attom_history_n": "",
            "berry_parcel_match_rate": lk.get("parcel_match_rate"),
            "high_conf_transaction_match_rate": lk.get("high_conf_rate"),
            "safe_history_match_rate": lk.get("safe_history_rate_among_unique_apn"),
            "linkage_selection_risk": lk.get("linkage_selection_flag") or (lk.get("bias_summary") or {}).get("linkage_selection_flag"),
            "model_n": b.get("n") or t.get("n_property_use_385"),
            "model_period": " — ".join(b.get("sale_date_range") or [t.get("sale_date_min", ""), t.get("sale_date_max", "")]),
            "baseline_r2_price": b.get("primary_test_R2_price"),
            "baseline_prb": b.get("primary_test_PRB"),
            "baseline_beta_log": b.get("primary_test_Beta_log"),
            "baseline_delta_nl": b.get("primary_test_Delta_NL"),
            "baseline_dcor": b.get("primary_test_dCor"),
            "panel_decision": u.get("panel_decision", "pending"),
            "direct_run": "yes" if authorized and method else "no",
            "surrogate_run": "yes" if authorized and method else "no",
            "main_scientific_role": (
                "Wayne County AVM + Detroit Berry validation" if key == "wayne"
                else "Philadelphia AVM + Berry validation" if key == "philadelphia"
                else "St. Louis County AVM + Berry validation + provider robustness"
            ),
            "confidence": "pending_until_gates_complete" if not b else "see freeze notes",
        })

    header = "| " + " | ".join(rows[0].keys()) + " |\n| " + " | ".join("---" for _ in rows[0]) + " |\n"
    body = "".join("| " + " | ".join(cell(r[k]) for k in r) + " |\n" for r in rows)
    text = f"""# Berry/ATTOM validation v2 — final report

Written: {datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}  
Protocol: `analysis/berry_attom_validation_v2/protocol_v2.yaml`  
**Manuscript was not edited. Frozen v1 under `analysis/berry_cmf_validation/` was not modified.**

This report distinguishes Berry *official assessment/sale* ratios from model
*valuation/sale* ratios. Those estimands are never mixed.

## 1. Berry/local external assessment-regressivity validation

See `berry_reproduction/REPRODUCTION_V2_NOTES.md` and `reproduction_summary.csv`.

- Detroit: Python translation of the Rmd filters (class 401, VALID ARMS LENGTH,
  `filter=FALSE` in `cmfproperty::reformat_data`). Native R comparison is
  attempted separately when the `cmfproperty` sources can be sourced.
- Philadelphia: canonical table is `Ratio_Analysis_arms_length.dta` only. The
  total file is a disjoint universe, not a duplicate sample to rbind.
- St. Louis County: rebuilt from the 2019 cumulative `sales.csv` extract with
  actual `SALEDT` and `PRICE`. `joined.csv` and `SALE_PRICE=APRTOT` were not used.

## 2. Berry↔ATTOM transaction/linkage validation

See `linkage/linkage_waterfall.csv`, `linkage/linkage_summary.json`, and
`linkage/transaction_concordance.parquet` (large; gitignored).

Parcel identity uses APN hierarchy only. One-to-many APN maps are `AMBIGUOUS_APN`
and are not collapsed. Price/date is never used as a parcel identifier.

## 3. Linkage-selection bias

See `linkage/matched_unmatched_balance.csv` and
`source_concordance/berry_full_vs_attom_linkable.csv`.
`figures/linkage_match_rate_by_price_decile.pdf` plots P(match | price decile).

If ATTOM matching flips the Berry PRB sign or shows a large price-decile
match-rate spread, the jurisdiction is flagged and is not quietly promoted.

## 4. ATTOM standardized baseline AVMs

Wayne County, Philadelphia County, and St. Louis County, 2016–2025, log
Recorder target, HISTORY_MARKET_CORE primary, HISTORY_STRUCTURAL_CORE secondary.
LR + ordinary LightGBM only, validation selection, 200 monthly block bootstraps.

Wayne County models are never labeled Detroit.

See `reports/BASELINE_REPORT.md`.

## 5. St. Louis local-vs-ATTOM provider robustness

See `reports/ST_LOUIS_SOURCE_ROBUSTNESS.md`. Local table uses actual PRICE and
SALEDT plus the latest dwelling snapshot strictly before sale, with an explicit
2012-folder/TAXYR=2013 sensitivity.

## 6. Frozen panel decision

`panel_freeze/final_panel_freeze_v2.yaml`

Direct/Surrogate authorized: **{authorized}**
Passing units: {freeze.get("passing_units")}

## 7. Direct/Surrogate transfer results

{"Run. See `method_transfer/all_metrics.csv`. Horizontal axes use achieved mechanism reduction, not raw rho." if authorized and method else "Not run. Either the freeze is not yet written or fewer than two independent new modeling units passed. Audit/baselines are the scientific outcome in that case."}

## 8. Failure/boundary cases

- St. Louis City FIPS 29510 must not appear in the 29189 cache.
- Detroit Berry is a city sample; ATTOM AVM is Wayne County unless a validated
  city field supports a later sensitivity.
- Philadelphia v1 rbind concern is resolved: arms-length and total files are
  disjoint universes.
- Step 9 ATTOM-vs-Berry assessment-field comparison is skipped; see
  `source_concordance/STEP9_ASSESSMENT_CONCORDANCE.md`.

## 9. Relation to the existing six-county ATTOM analysis

See `source_concordance/existing_six_vs_v2_metadata.csv`. Those runs include
Tax Assessor/ACS/location features that v2 does not have. Do not present the
two pipelines as identical.

## 10. What Tax Assessor could add later

A later Tax Assessor extract could support a **separately labeled** location/ACS
enrichment sensitivity (coordinates, tract, current APN crosswalk). It must not
be folded into HISTORY_MARKET_CORE unless a new protocol is frozen first. Tax
values still must not become primary predictors.

## 11. Exact recommended paper additions (DO NOT EDIT THE PAPER IN THIS PASS)

1. Keep Cook County / CCAO as the primary application.
2. If v2 freeze authorizes transfer, add Wayne / Philadelphia / St. Louis County
   as a literature-anchored external-validation appendix, with Wayne **not**
   labeled Detroit, using HISTORY_MARKET_CORE (no Tax Assessor/ACS).
3. If the freeze does not authorize transfer, state that the new Dewey delivery
   was used to validate Berry transactions against ATTOM and to build leakage-safe
   baselines, and that Direct/Surrogate were not transferred because fewer than
   two new modeling units passed the pre-rho gates.
4. Keep the existing six-county ATTOM results as a separate sensitivity layer
   with a different feature class.
5. Do not cite Berry official assessment ratios as if they were model valuation
   ratios.

Inventory coverage rows: {len(inv_cov)}; FIPS-year rows: {len(inv_counts)}.

## Canonical table

{header}{body}
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text, encoding="utf-8")
    print("wrote", REPORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
