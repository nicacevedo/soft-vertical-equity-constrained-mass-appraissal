#!/usr/bin/env python3
"""Step 10: standardized 2016-2025 ATTOM modeling tables (HISTORY_MARKET_CORE / STRUCTURAL_CORE).

Target: log qualified Recorder TRANSFERAMOUNT. Latest Assessor History strictly before sale.
No assessed/market/tax-value predictors. No Tax Assessor / ACS / neighbor features.
"""
from __future__ import annotations

import argparse
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
    attach_recorder_prior_sales, clean_fips, load_sale_validation_dictionary,
    match_history, normalize_property_use,
)
from analysis.berry_attom_validation_v2.scripts.v2_common import (  # noqa: E402
    ANALYSIS, ASSESSMENT_VALUE_COLUMNS, COUNTIES, MIN_SALE_PRICE, OUTPUT,
    PROPERTY_USE_CODES, SALE_WINDOW, sha256_file,
)

FEAT = ANALYSIS / "feature_audit"
FORBIDDEN_PREDICTORS = ASSESSMENT_VALUE_COLUMNS | {
    "tax_assessor_latitude", "tax_assessor_longitude", "tax_assessor_geoid",
    "LATITUDE", "LONGITUDE",
}


def policy() -> SaleValidationPolicy:
    decisions, sha = load_sale_validation_dictionary(SALE_VALIDATION_DICTIONARY_PATH)
    return SaleValidationPolicy(
        minimum_sale_price=MIN_SALE_PRICE,
        dictionary_path=str(SALE_VALIDATION_DICTIONARY_PATH),
        dictionary_sha256=sha,
        code_decisions=decisions,
        arms_length_only=True,
        single_parcel_only=True,
        cohort="broad",
    )


def build_county(key: str, fips: str, pol: SaleValidationPolicy) -> dict:
    rec_path = OUTPUT / "cache" / key / "recorder.parquet"
    hist_path = OUTPUT / "cache" / key / "history.parquet"
    if not rec_path.exists() or not hist_path.exists():
        return {"county_key": key, "status": "SKIPPED_NO_CACHE"}
    rec = pq.read_table(rec_path).to_pandas()
    rec["sale_date"] = pd.to_datetime(rec["INSTRUMENTDATE"], errors="coerce").fillna(
        pd.to_datetime(rec["RECORDINGDATE"], errors="coerce")
    )
    rec["sale_price"] = pd.to_numeric(rec["TRANSFERAMOUNT"], errors="coerce")
    rec["ATTOMID"] = pd.to_numeric(rec["ATTOMID"], errors="coerce").astype("Int64")
    rec = rec.loc[clean_fips(rec["DOCUMENTRECORDINGCOUNTYFIPS"]).eq(fips)].copy()
    if fips == "29189":
        assert not clean_fips(rec["DOCUMENTRECORDINGCOUNTYFIPS"]).eq("29510").any()
    eligible, audit, waterfall = apply_sale_validation(rec, pol)
    start, end = pd.Timestamp(SALE_WINDOW[0]), pd.Timestamp(SALE_WINDOW[1])
    targets = eligible.loc[eligible["sale_date"].between(start, end)].copy()
    for col in ("PROPERTYUSESTANDARDIZED", "PROPERTYADDRESSFULL", "PROPERTYADDRESSCITY", "PROPERTYADDRESSZIP"):
        if col in targets.columns:
            targets = targets.drop(columns=[col])
    hist = pq.read_table(hist_path).to_pandas()
    hist["ATTOMID"] = pd.to_numeric(hist["ATTOMID"], errors="coerce").astype("Int64")
    year = pd.to_numeric(hist["ASSESSORHISTORYYEAR"], errors="coerce").astype("Int64")
    hist["assessed_through"] = pd.to_datetime(year.astype("string") + "-12-31", errors="coerce")
    hist = hist.loc[hist["ATTOMID"].notna() & hist["assessed_through"].notna()].copy()
    if "SITUSSTATECOUNTYFIPS" in hist.columns:
        hist = hist.loc[clean_fips(hist["SITUSSTATECOUNTYFIPS"]).eq(fips)].copy()
        if fips == "29189":
            assert not clean_fips(hist["SITUSSTATECOUNTYFIPS"]).eq("29510").any()
    n_id_overlap = int(targets["ATTOMID"].isin(hist["ATTOMID"]).sum())
    matched = match_history(targets, hist)
    assert (matched["assessed_through"] < matched["sale_date"]).all()
    n_hist = len(matched)
    use = set(PROPERTY_USE_CODES)
    matched = matched.loc[normalize_property_use(matched["PROPERTYUSESTANDARDIZED"]).isin(use)].copy()
    matched = attach_recorder_prior_sales(matched, [rec_path], fips, pol)
    # Current sale must never appear as its own prior-sale feature.
    if "recorder_prior_sale_age_years" in matched.columns:
        bad_age = matched["recorder_prior_sale_age_years"].fillna(1) <= 0
        if bad_age.any():
            raise RuntimeError(f"{key}: non-positive prior-sale age on {int(bad_age.sum())} rows")
    present_forbidden = sorted(c for c in matched.columns if c in FORBIDDEN_PREDICTORS)
    out_dir = OUTPUT / "modeling_tables" / key
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / "history_market_core.parquet"
    matched.sort_values("sale_date").reset_index(drop=True).to_parquet(table_path, index=False)
    lag = matched["history_lag_days"] if "history_lag_days" in matched else pd.Series(dtype=float)
    rec = {
        "county_key": key,
        "fips": fips,
        "status": "OK",
        "n_recorder_raw": int(len(rec)),
        "n_sale_validation_eligible": int(len(eligible)),
        "n_window_2016_2025": int(len(targets)),
        "n_attomid_history_overlap": n_id_overlap,
        "n_strict_history_match": n_hist,
        "n_property_use_385": int(len(matched)),
        "sale_date_min": str(matched["sale_date"].min().date()) if len(matched) else "",
        "sale_date_max": str(matched["sale_date"].max().date()) if len(matched) else "",
        "history_lag_days_p50": float(lag.median()) if len(lag) else None,
        "share_lag_gt_1yr": float((matched["history_lag_years"] > 1).mean()) if len(matched) else None,
        "share_lag_gt_2yr": float((matched["history_lag_years"] > 2).mean()) if len(matched) else None,
        "share_lag_gt_3yr": float((matched["history_lag_years"] > 3).mean()) if len(matched) else None,
        "prior_sale_feature_nonnull_share": float(matched["recorder_prior_sale_amount"].notna().mean())
        if "recorder_prior_sale_amount" in matched else None,
        "forbidden_value_columns_present_in_table": "|".join(present_forbidden),
        "forbidden_columns_used_as_predictors": "NO_IN_PRIMARY_FEATURE_SETS",
        "table_path": str(table_path),
        "table_sha256": sha256_file(table_path),
        "recorder_cache_sha256": sha256_file(rec_path),
        "history_cache_sha256": sha256_file(hist_path),
        "written_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "detroit_city_note": (
            "Wayne County table is NOT a Detroit-city model. PROPERTYJURISDICTIONNAME retained for an optional later sensitivity."
            if key == "wayne" else ""
        ),
    }
    waterfall.assign(county_key=key).to_csv(out_dir / "sale_validation_waterfall.csv", index=False)
    (out_dir / "modeling_table_meta.json").write_text(json.dumps(rec, indent=2, default=str) + "\n")
    print(json.dumps({k: rec[k] for k in ["county_key", "n_property_use_385", "table_sha256"]}, sort_keys=True), flush=True)
    return rec


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--county-key", default="all")
    args = parser.parse_args()
    pol = policy()
    counties = [c for c in COUNTIES if args.county_key in {"all", c["key"]}]
    rows = [build_county(c["key"], c["fips"], pol) for c in counties]
    FEAT.mkdir(parents=True, exist_ok=True)
    existing = []
    for c in COUNTIES:
        meta = OUTPUT / "modeling_tables" / c["key"] / "modeling_table_meta.json"
        if meta.exists():
            existing.append(json.loads(meta.read_text()))
    if existing:
        pd.DataFrame(existing).to_csv(FEAT / "modeling_table_summary.csv", index=False)
        pd.DataFrame([{
            "county_key": r["county_key"], "table_sha256": r.get("table_sha256"),
            "recorder_cache_sha256": r.get("recorder_cache_sha256"),
            "history_cache_sha256": r.get("history_cache_sha256"),
            "n": r.get("n_property_use_385"),
        } for r in existing]).to_csv(FEAT / "modeling_table_hashes.csv", index=False)
    print("modeling tables done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
