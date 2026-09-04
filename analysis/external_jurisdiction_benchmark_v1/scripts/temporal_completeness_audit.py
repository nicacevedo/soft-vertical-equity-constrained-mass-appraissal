#!/usr/bin/env python3
"""Step 3: monthly transaction completeness audit, 2016-2025.

Predeclared rule (stated here BEFORE looking at any county's numbers):
  - "complete" for a calendar year means all 12 months present with a sale
    count in every month, and that year's total is not a small fraction
    (< 50%) of the median of the two preceding years' totals (rules out a
    partial-year artifact where all 12 month buckets exist but almost empty).
  - forward = calendar 2025 iff EVERY primary jurisdiction is complete in 2025
    under that rule; otherwise the latest common complete year is used, chosen
    by month coverage and sale counts alone -- never by any model outcome.

Reads each jurisdiction's Recorder cache (built by build_county_caches.py) and
applies the qualified-sale-validation eligibility filter so completeness is
assessed on the same population the modeling cohort will draw from, not on
raw unfiltered transfers.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.other_counties_benchmars import (  # noqa: E402
    SALE_VALIDATION_DICTIONARY_PATH, SaleValidationPolicy, apply_sale_validation,
    clean_fips, load_sale_validation_dictionary,
)
from analysis.external_jurisdiction_benchmark_v1.scripts.v1_common import (  # noqa: E402
    ALL_KEYS, ANALYSIS, JURISDICTION_BY_KEY, MIN_SALE_PRICE, OUTPUT, ST_LOUIS_CITY_FIPS, write_json,
)

MIN_YEAR_RATIO_VS_PRECEDING_MEDIAN = 0.50


def policy() -> SaleValidationPolicy:
    decisions, sha = load_sale_validation_dictionary(SALE_VALIDATION_DICTIONARY_PATH)
    return SaleValidationPolicy(
        minimum_sale_price=MIN_SALE_PRICE, dictionary_path=str(SALE_VALIDATION_DICTIONARY_PATH),
        dictionary_sha256=sha, code_decisions=decisions, arms_length_only=True,
        single_parcel_only=True, cohort="broad",
    )


def load_eligible(key: str, pol: SaleValidationPolicy) -> pd.DataFrame | None:
    j = JURISDICTION_BY_KEY[key]
    cache = OUTPUT / "cache" / key / "recorder.parquet"
    if not cache.exists():
        return None
    rec = pq.read_table(cache).to_pandas()
    rec["sale_date"] = pd.to_datetime(rec["INSTRUMENTDATE"], errors="coerce").fillna(
        pd.to_datetime(rec.get("RECORDINGDATE"), errors="coerce")
    )
    rec["sale_price"] = pd.to_numeric(rec.get("TRANSFERAMOUNT"), errors="coerce")
    rec = rec.loc[clean_fips(rec["DOCUMENTRECORDINGCOUNTYFIPS"]).eq(j["fips"])].copy()
    if j["fips"] == "29189":
        assert not clean_fips(rec["DOCUMENTRECORDINGCOUNTYFIPS"]).eq(ST_LOUIS_CITY_FIPS).any()
    eligible, _audit, _wf = apply_sale_validation(rec, pol)
    return eligible


def monthly_completeness(eligible: pd.DataFrame) -> dict:
    eligible = eligible.copy()
    eligible["year"] = eligible["sale_date"].dt.year
    eligible["month"] = eligible["sale_date"].dt.month
    by_year_month = eligible.groupby(["year", "month"]).size()
    by_year = eligible.groupby("year").size()
    years = {}
    for y in range(2016, 2026):
        months_present = sorted(int(m) for (yy, m) in by_year_month.index if yy == y)
        n_year = int(by_year.get(y, 0))
        preceding = [int(by_year.get(y - 1, 0)), int(by_year.get(y - 2, 0))]
        preceding = [p for p in preceding if p > 0]
        preceding_median = float(pd.Series(preceding).median()) if preceding else float("nan")
        ratio_vs_preceding = (n_year / preceding_median) if preceding_median else float("nan")
        complete = (
            len(months_present) == 12
            and (pd.isna(ratio_vs_preceding) or ratio_vs_preceding >= MIN_YEAR_RATIO_VS_PRECEDING_MEDIAN)
        )
        years[str(y)] = {
            "n_months_present": len(months_present), "months_present": months_present,
            "n_sales": n_year, "ratio_vs_preceding_2yr_median": ratio_vs_preceding,
            "complete": bool(complete),
        }
    return years


def main() -> int:
    pol = policy()
    all_years: dict[str, dict] = {}
    missing_cache = []
    for key in ALL_KEYS:
        eligible = load_eligible(key, pol)
        if eligible is None:
            missing_cache.append(key)
            continue
        all_years[key] = monthly_completeness(eligible)
        print(json.dumps({
            "county_key": key,
            "2025_complete": all_years[key]["2025"]["complete"],
            "n_2025": all_years[key]["2025"]["n_sales"],
        }), flush=True)

    rows = []
    for key, years in all_years.items():
        for y, rec in years.items():
            rows.append({"county_key": key, "year": int(y), **rec})
    df = pd.DataFrame(rows)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    df.to_csv(ANALYSIS / "audits" / "monthly_completeness.csv", index=False)

    all_2025_complete = bool(all_years) and all(
        all_years[k]["2025"]["complete"] for k in all_years
    ) and not missing_cache
    if all_years:
        forward_year = 2025 if all_2025_complete else max(
            y for y in range(2016, 2025)
            if all(all_years[k][str(y)]["complete"] for k in all_years)
        )
    else:
        forward_year = None

    decision = {
        "written_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rule": (
            "complete year = 12 months present AND n_sales >= "
            f"{MIN_YEAR_RATIO_VS_PRECEDING_MEDIAN} * median(preceding 2 years' n_sales)"
        ),
        "jurisdictions_evaluated": sorted(all_years.keys()),
        "jurisdictions_missing_cache": missing_cache,
        "all_jurisdictions_2025_complete": all_2025_complete,
        "forward_year_decision": forward_year,
        "decided_by_month_coverage_and_counts_only": True,
    }
    write_json(ANALYSIS / "audits" / "temporal_completeness_decision.json", decision)
    print(json.dumps(decision, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
