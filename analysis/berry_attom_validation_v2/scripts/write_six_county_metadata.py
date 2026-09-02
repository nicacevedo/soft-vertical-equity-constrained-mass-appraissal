#!/usr/bin/env python3
"""Step 18: metadata comparison vs the existing six-county ATTOM sensitivity layer.

Does not overwrite or recompute those results. Does not present old-six and
new-v2 metrics as if they came from the same feature pipeline.
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "analysis/berry_attom_validation_v2/source_concordance"

ROWS = [
    {
        "jurisdiction": "Cook County, IL",
        "fips": "17031",
        "data_sources": "ATTOM Recorder + Assessor History + Tax Assessor + ACS",
        "feature_class": "ccao_core_acs / attom_market_history / status_quo_augmented (includes location/ACS)",
        "years": "2016-2025 (recommended $50k floor)",
        "target": "log/raw Recorder TRANSFERAMOUNT (script may select scale)",
        "split": "chronological 20% test; 10% of pre-test validation",
        "method_families": "LGBM baseline + LGBCovPenalty[diff] (existing run)",
        "role_in_paper": "exploratory six-county ATTOM sensitivity; NOT overwritten by v2",
        "pipeline": "existing_six",
    },
    {
        "jurisdiction": "Allegheny County, PA",
        "fips": "42003",
        "data_sources": "ATTOM Recorder + Assessor History + Tax Assessor + ACS",
        "feature_class": "same as Cook existing-six (includes location/ACS)",
        "years": "2016-2025",
        "target": "Recorder TRANSFERAMOUNT",
        "split": "same chronological protocol",
        "method_families": "LGBM + LGBCovPenalty[diff]",
        "role_in_paper": "exploratory six-county ATTOM sensitivity; NOT overwritten by v2",
        "pipeline": "existing_six",
    },
    {
        "jurisdiction": "Maricopa County, AZ",
        "fips": "04013",
        "data_sources": "ATTOM Recorder + Assessor History + Tax Assessor + ACS",
        "feature_class": "includes location/ACS",
        "years": "2016-2025",
        "target": "Recorder TRANSFERAMOUNT",
        "split": "same chronological protocol",
        "method_families": "LGBM + LGBCovPenalty[diff]",
        "role_in_paper": "exploratory six-county ATTOM sensitivity; NOT overwritten by v2",
        "pipeline": "existing_six",
    },
    {
        "jurisdiction": "King County, WA",
        "fips": "53033",
        "data_sources": "ATTOM Recorder + Assessor History + Tax Assessor + ACS",
        "feature_class": "includes location/ACS",
        "years": "2016-2025",
        "target": "Recorder TRANSFERAMOUNT",
        "split": "same chronological protocol",
        "method_families": "LGBM + LGBCovPenalty[diff]",
        "role_in_paper": "exploratory six-county ATTOM sensitivity; NOT overwritten by v2",
        "pipeline": "existing_six",
    },
    {
        "jurisdiction": "Miami-Dade County, FL",
        "fips": "12086",
        "data_sources": "ATTOM Recorder + Assessor History + Tax Assessor + ACS",
        "feature_class": "includes location/ACS",
        "years": "2016-2025",
        "target": "Recorder TRANSFERAMOUNT",
        "split": "same chronological protocol",
        "method_families": "LGBM + LGBCovPenalty[diff]",
        "role_in_paper": "exploratory six-county ATTOM sensitivity / v1 boundary case; NOT overwritten by v2",
        "pipeline": "existing_six",
    },
    {
        "jurisdiction": "Middlesex County, MA",
        "fips": "25017",
        "data_sources": "ATTOM Recorder + Assessor History + Tax Assessor + ACS",
        "feature_class": "includes location/ACS",
        "years": "2016-2025",
        "target": "Recorder TRANSFERAMOUNT",
        "split": "same chronological protocol",
        "method_families": "LGBM + LGBCovPenalty[diff]",
        "role_in_paper": "exploratory six-county ATTOM sensitivity; NOT overwritten by v2",
        "pipeline": "existing_six",
    },
    {
        "jurisdiction": "Wayne County, MI",
        "fips": "26163",
        "data_sources": "ATTOM Recorder + Assessor History only (new Dewey 2003/2004-2025)",
        "feature_class": "HISTORY_MARKET_CORE / HISTORY_STRUCTURAL_CORE; no Tax Assessor, ACS, or neighbor features",
        "years": "2016-2025 modeled sales; 2003-2015 Recorder only as strictly prior history",
        "target": "log qualified Recorder TRANSFERAMOUNT (scale fixed)",
        "split": "chronological 20% test; 10% of pre-test validation; seed 2025",
        "method_families": "LR + LGBM; Direct/Surrogate only if v2 freeze authorizes",
        "role_in_paper": "v2 standardized AVM unit for Detroit Berry external validation; NEVER labeled Detroit",
        "pipeline": "v2_new",
    },
    {
        "jurisdiction": "Philadelphia County, PA",
        "fips": "42101",
        "data_sources": "ATTOM Recorder + Assessor History only",
        "feature_class": "HISTORY_MARKET_CORE / HISTORY_STRUCTURAL_CORE; no Tax Assessor/ACS",
        "years": "2016-2025 modeled sales",
        "target": "log qualified Recorder TRANSFERAMOUNT",
        "split": "same v2 chronological protocol",
        "method_families": "LR + LGBM; Direct/Surrogate only if freeze authorizes",
        "role_in_paper": "v2 standardized AVM unit + Berry external validation",
        "pipeline": "v2_new",
    },
    {
        "jurisdiction": "St. Louis County, MO",
        "fips": "29189",
        "data_sources": "ATTOM Recorder + Assessor History; plus repaired local dwelling/sales extracts",
        "feature_class": "HISTORY_MARKET_CORE / HISTORY_STRUCTURAL_CORE; local dwelling sensitivity; no Tax Assessor/ACS",
        "years": "2016-2025 ATTOM; local AVM 2016-2019 (sales extract ends 2019)",
        "target": "ATTOM: log Recorder TRANSFERAMOUNT; local: log actual PRICE (not APRTOT)",
        "split": "same chronological protocol on each table",
        "method_families": "LR + LGBM; local-vs-ATTOM provider robustness; Direct/Surrogate only if freeze authorizes",
        "role_in_paper": "v2 standardized AVM unit + Berry external validation + provider robustness; not St. Louis City 29510",
        "pipeline": "v2_new",
    },
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "existing_six_vs_v2_metadata.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ROWS[0].keys()))
        w.writeheader()
        w.writerows(ROWS)
    (OUT / "EXISTING_SIX_VS_V2.md").write_text(
        "# Existing six-county ATTOM vs new v2 jurisdictions\n\n"
        "These are **not** the same feature pipeline. The existing six-county "
        "runs include Tax Assessor / ACS / location information that is not "
        "available for the new Dewey Wayne / Philadelphia / St. Louis County "
        "delivery. Do not pool or rank metrics across the two pipelines.\n\n"
        "When Tax Assessor becomes available for the v2 counties, add it only "
        "as a **separately labeled enrichment sensitivity** unless a new "
        "protocol is frozen before use.\n",
        encoding="utf-8",
    )
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
