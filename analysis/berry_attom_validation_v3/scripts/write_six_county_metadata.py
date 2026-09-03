#!/usr/bin/env python3
"""Metadata comparison vs the existing six-county ATTOM sensitivity layer.

Does not overwrite those results. v3 currently lacks Tax Assessor/ACS/location.
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "analysis/berry_attom_validation_v3/source_concordance"

ROWS = [
    {
        "jurisdiction": "Cook County, IL",
        "fips": "17031",
        "data_sources": "ATTOM Recorder + Assessor History + Tax Assessor + ACS",
        "feature_class": "includes Tax Assessor / ACS / location",
        "years": "2016-2025",
        "target": "Recorder TRANSFERAMOUNT",
        "chronology": "chronological 20% test; 10% of pre-test validation",
        "penalty_methods": "LGBM + LGBCovPenalty[diff] (existing run)",
        "validation_role": "exploratory six-county ATTOM sensitivity; NOT overwritten by v3",
        "pipeline": "existing_six",
    },
    {
        "jurisdiction": "Allegheny County, PA",
        "fips": "42003",
        "data_sources": "ATTOM Recorder + Assessor History + Tax Assessor + ACS",
        "feature_class": "includes Tax Assessor / ACS / location",
        "years": "2016-2025",
        "target": "Recorder TRANSFERAMOUNT",
        "chronology": "same chronological protocol",
        "penalty_methods": "LGBM + LGBCovPenalty[diff]",
        "validation_role": "exploratory six-county ATTOM sensitivity; NOT overwritten by v3",
        "pipeline": "existing_six",
    },
    {
        "jurisdiction": "Maricopa County, AZ",
        "fips": "04013",
        "data_sources": "ATTOM Recorder + Assessor History + Tax Assessor + ACS",
        "feature_class": "includes Tax Assessor / ACS / location",
        "years": "2016-2025",
        "target": "Recorder TRANSFERAMOUNT",
        "chronology": "same chronological protocol",
        "penalty_methods": "LGBM + LGBCovPenalty[diff]",
        "validation_role": "exploratory six-county ATTOM sensitivity; NOT overwritten by v3",
        "pipeline": "existing_six",
    },
    {
        "jurisdiction": "King County, WA",
        "fips": "53033",
        "data_sources": "ATTOM Recorder + Assessor History + Tax Assessor + ACS",
        "feature_class": "includes Tax Assessor / ACS / location",
        "years": "2016-2025",
        "target": "Recorder TRANSFERAMOUNT",
        "chronology": "same chronological protocol",
        "penalty_methods": "LGBM + LGBCovPenalty[diff]",
        "validation_role": "exploratory six-county ATTOM sensitivity; NOT overwritten by v3",
        "pipeline": "existing_six",
    },
    {
        "jurisdiction": "Miami-Dade County, FL",
        "fips": "12086",
        "data_sources": "ATTOM Recorder + Assessor History + Tax Assessor + ACS",
        "feature_class": "includes Tax Assessor / ACS / location",
        "years": "2016-2025",
        "target": "Recorder TRANSFERAMOUNT",
        "chronology": "same chronological protocol",
        "penalty_methods": "LGBM + LGBCovPenalty[diff]",
        "validation_role": "exploratory six-county ATTOM sensitivity; NOT overwritten by v3",
        "pipeline": "existing_six",
    },
    {
        "jurisdiction": "Middlesex County, MA",
        "fips": "25017",
        "data_sources": "ATTOM Recorder + Assessor History + Tax Assessor + ACS",
        "feature_class": "includes Tax Assessor / ACS / location",
        "years": "2016-2025",
        "target": "Recorder TRANSFERAMOUNT",
        "chronology": "same chronological protocol",
        "penalty_methods": "LGBM + LGBCovPenalty[diff]",
        "validation_role": "exploratory six-county ATTOM sensitivity; NOT overwritten by v3",
        "pipeline": "existing_six",
    },
    {
        "jurisdiction": "Wayne County, MI",
        "fips": "26163",
        "data_sources": "ATTOM Recorder + Assessor History only (new Dewey 2003/2004-2025)",
        "feature_class": "HISTORY_MARKET_CORE / HISTORY_STRUCTURAL_CORE; no Tax Assessor, ACS, or location enrichment",
        "years": "2016-2025 modeled sales; older Recorder only as strictly prior history",
        "target": "log qualified Recorder TRANSFERAMOUNT",
        "chronology": "chronological 20% test held until after panel freeze; 10% of pre-test validation; seed 2025",
        "penalty_methods": "LR + LGBM; Direct/Surrogate only if v3 freeze authorizes",
        "validation_role": "v3 standardized AVM unit; Berry anchor is Detroit city, NEVER labeled Detroit",
        "pipeline": "v3_new",
    },
    {
        "jurisdiction": "Philadelphia County, PA",
        "fips": "42101",
        "data_sources": "ATTOM Recorder + Assessor History only",
        "feature_class": "HISTORY_MARKET_CORE / HISTORY_STRUCTURAL_CORE; no Tax Assessor/ACS",
        "years": "2016-2025 modeled sales",
        "target": "log qualified Recorder TRANSFERAMOUNT",
        "chronology": "same v3 chronological protocol (test after freeze)",
        "penalty_methods": "LR + LGBM; Direct/Surrogate only if freeze authorizes",
        "validation_role": "v3 standardized AVM unit + Berry external validation",
        "pipeline": "v3_new",
    },
    {
        "jurisdiction": "St. Louis County, MO",
        "fips": "29189",
        "data_sources": "ATTOM Recorder + Assessor History; plus repaired local dwelling/sales extracts",
        "feature_class": "HISTORY_MARKET_CORE / HISTORY_STRUCTURAL_CORE; local dwelling sensitivity; no Tax Assessor/ACS",
        "years": "2016-2025 ATTOM; local AVM 2016-2019 (sales extract ends 2019)",
        "target": "ATTOM: log Recorder TRANSFERAMOUNT; local: log actual PRICE (not APRTOT)",
        "chronology": "same chronological protocol; test after freeze",
        "penalty_methods": "LR + LGBM; local-vs-ATTOM provider robustness; Direct/Surrogate only if freeze authorizes",
        "validation_role": "v3 standardized AVM unit + Berry/local source validation; not St. Louis City 29510",
        "pipeline": "v3_new",
    },
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "existing_six_vs_v3_metadata.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ROWS[0].keys()))
        w.writeheader()
        w.writerows(ROWS)
    (OUT / "EXISTING_SIX_VS_V3.md").write_text(
        "# Existing six-county ATTOM vs new v3 jurisdictions\n\n"
        "These are **not** the same feature pipeline. The existing six-county "
        "runs include Tax Assessor / ACS / location information that is not "
        "part of the v3 primary design. Do not pool or rank metrics across "
        "the two pipelines. Old-six results are a separate exploratory/"
        "sensitivity layer and were not overwritten.\n",
        encoding="utf-8",
    )
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
