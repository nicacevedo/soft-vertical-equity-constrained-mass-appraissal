#!/usr/bin/env python3
"""Step 5 cohort construction: HISTORY_MARKET_CORE modeling tables under the
FROZEN residential_code_mapping.yaml. Development-period only (sale_date in
[2016-01-01, --end-date], default 2024-12-31) -- 2025 is never read or written
by this script, per temporal_design.yaml's forward-lock rule. The full
2016-2025 table for the frozen refit/forward evaluation is built separately at
Step 14/15, only after every freeze in this protocol.

PRIMARY_RESIDENTIAL codes are read from residential_code_mapping.yaml at
runtime, never hardcoded here, so cohort membership has exactly one source of
truth.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.other_counties_benchmars import (  # noqa: E402
    SALE_VALIDATION_DICTIONARY_PATH, SaleValidationPolicy, apply_sale_validation,
    attach_recorder_prior_sales, clean_fips, load_sale_validation_dictionary,
    match_history, normalize_property_use,
)
from analysis.external_jurisdiction_benchmark_v1.scripts.v1_common import (  # noqa: E402
    ALL_KEYS, ANALYSIS, ASSESSMENT_VALUE_COLUMNS, JURISDICTION_BY_KEY, MIN_SALE_PRICE, OUTPUT,
    PILOT_KEYS, SALE_WINDOW, ST_LOUIS_CITY_FIPS, sha256_file, write_json,
)

MAPPING_PATH = ANALYSIS / "cohort" / "residential_code_mapping.yaml"
DEV_END_DATE_DEFAULT = "2024-12-31"


def load_primary_residential_codes() -> set[str]:
    mapping = yaml.safe_load(MAPPING_PATH.read_text())
    if mapping.get("status") != "FROZEN":
        raise SystemExit(f"residential_code_mapping.yaml status={mapping.get('status')!r}, not FROZEN")
    return {str(c) for c in mapping["PRIMARY_RESIDENTIAL"]["codes"]}


def policy() -> SaleValidationPolicy:
    decisions, sha = load_sale_validation_dictionary(SALE_VALIDATION_DICTIONARY_PATH)
    return SaleValidationPolicy(
        minimum_sale_price=MIN_SALE_PRICE, dictionary_path=str(SALE_VALIDATION_DICTIONARY_PATH),
        dictionary_sha256=sha, code_decisions=decisions, arms_length_only=True,
        single_parcel_only=True, cohort="broad",
    )


def retention_table(eligible: pd.DataFrame, window: pd.DataFrame, hist: pd.DataFrame,
                     final: pd.DataFrame, key: str) -> pd.DataFrame:
    base = eligible.copy()
    base["sale_price"] = pd.to_numeric(base["sale_price"], errors="coerce")
    base = base.loc[base["sale_price"].gt(0)]
    base["price_decile"] = pd.qcut(base["sale_price"], 10, labels=False, duplicates="drop")
    win_ids = set(window["TRANSACTIONID"]) if "TRANSACTIONID" in window else set()
    hist_ids = set(hist["TRANSACTIONID"]) if len(hist) and "TRANSACTIONID" in hist else set()
    fin_ids = set(final["TRANSACTIONID"]) if len(final) and "TRANSACTIONID" in final else set()
    rows = []
    for dec, g in base.groupby("price_decile"):
        rows.append({
            "county_key": key, "price_decile": int(dec) if pd.notna(dec) else -1,
            "n_eligible_recorder": int(len(g)),
            "p_in_dev_window": float(g["TRANSACTIONID"].isin(win_ids).mean()) if "TRANSACTIONID" in g else np.nan,
            "p_safe_history": float(g["TRANSACTIONID"].isin(hist_ids).mean()) if "TRANSACTIONID" in g else np.nan,
            "p_primary_residential_final": float(g["TRANSACTIONID"].isin(fin_ids).mean()) if "TRANSACTIONID" in g else np.nan,
            "median_price": float(g["sale_price"].median()),
        })
    return pd.DataFrame(rows)


def build_county(key: str, fips: str, pol: SaleValidationPolicy, primary_codes: set[str],
                  end_date: str) -> dict:
    j = JURISDICTION_BY_KEY[key]
    rec_path = OUTPUT / "cache" / key / "recorder.parquet"
    hist_path = OUTPUT / "cache" / key / "history.parquet"
    if not rec_path.exists() or not hist_path.exists():
        return {"county_key": key, "status": "SKIPPED_NO_CACHE"}

    rec = pq.read_table(rec_path).to_pandas()
    rec["sale_date"] = pd.to_datetime(rec["INSTRUMENTDATE"], errors="coerce").fillna(
        pd.to_datetime(rec.get("RECORDINGDATE"), errors="coerce")
    )
    rec["sale_price"] = pd.to_numeric(rec.get("TRANSFERAMOUNT"), errors="coerce")
    rec["ATTOMID"] = pd.to_numeric(rec["ATTOMID"], errors="coerce").astype("Int64")
    rec = rec.loc[clean_fips(rec["DOCUMENTRECORDINGCOUNTYFIPS"]).eq(fips)].copy()
    if fips == "29189":
        assert not clean_fips(rec["DOCUMENTRECORDINGCOUNTYFIPS"]).eq(ST_LOUIS_CITY_FIPS).any()

    eligible, _audit, waterfall = apply_sale_validation(rec, pol)
    start = pd.Timestamp(SALE_WINDOW[0])
    end = pd.Timestamp(end_date)
    assert end < pd.Timestamp("2025-01-01"), (
        f"{key}: end_date={end_date} would include 2025 data before the Step 14 freeze -- refusing"
    )
    targets = eligible.loc[eligible["sale_date"].between(start, end)].copy()
    # PROPERTYUSESTANDARDIZED/address come from History post-match, not Recorder.
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
            assert not clean_fips(hist["SITUSSTATECOUNTYFIPS"]).eq(ST_LOUIS_CITY_FIPS).any()

    matched = match_history(targets, hist)
    assert (matched["assessed_through"] < matched["sale_date"]).all(), (
        f"{key}: history strictly-before-sale assertion failed"
    )
    n_hist = len(matched)

    final = matched.loc[normalize_property_use(matched["PROPERTYUSESTANDARDIZED"]).isin(primary_codes)].copy()
    final = attach_recorder_prior_sales(final, [rec_path], fips, pol)
    if "recorder_prior_sale_age_years" in final.columns:
        bad = final["recorder_prior_sale_age_years"].fillna(1) <= 0
        if bad.any():
            raise RuntimeError(f"{key}: non-positive prior-sale age on {int(bad.sum())} rows")

    # The History cache carries TAXASSESSEDVALUETOTAL/TAXMARKETVALUETOTAL (part
    # of HISTORY_CACHE_COLUMNS) because other pipelines need them; this one
    # must not. Drop them from the STORED table itself (not just from the
    # feature matrix built later by feature_frame/v1_features) so the parquet
    # on disk can never be a source of assessment-value leakage regardless of
    # how it is read downstream.
    drop_cols = [c for c in final.columns if c in ASSESSMENT_VALUE_COLUMNS]
    if drop_cols:
        final = final.drop(columns=drop_cols)
    leak = ASSESSMENT_VALUE_COLUMNS & set(final.columns)
    if leak:
        raise RuntimeError(f"{key}: assessment-value columns present in modeling table: {sorted(leak)}")
    if len(final) and final["sale_date"].max() >= pd.Timestamp("2025-01-01"):
        raise RuntimeError(f"{key}: modeling table contains sale_date >= 2025-01-01 before freeze")

    out_dir = OUTPUT / "modeling_tables" / key
    out_dir.mkdir(parents=True, exist_ok=True)
    ANALYSIS.joinpath("cohort").mkdir(parents=True, exist_ok=True)
    table_path = out_dir / "history_market_core_dev.parquet"
    final_sorted = final.sort_values("sale_date").reset_index(drop=True)
    final_sorted.to_parquet(table_path, index=False)

    ret = retention_table(eligible, targets, matched, final, key)
    ret.to_csv(ANALYSIS / "cohort" / f"{key}_modeling_retention_by_decile.csv", index=False)

    lag = final["history_lag_days"] if "history_lag_days" in final else pd.Series(dtype=float)
    rec_out = {
        "county_key": key, "fips": fips, "status": "OK",
        "development_period": [str(start.date()), str(end.date())],
        "forward_period_excluded": "2025 (not read, not written, per temporal_design.yaml)",
        "primary_residential_codes": sorted(primary_codes),
        "mapping_source": str(MAPPING_PATH),
        "n_recorder_raw": int(len(rec)),
        "n_sale_validation_eligible": int(len(eligible)),
        "n_dev_window": int(len(targets)),
        "n_strict_history_match": n_hist,
        "n_final_primary_residential": int(len(final)),
        "retention_eligible_to_final": float(len(final) / max(len(eligible), 1)),
        "retention_window_to_final": float(len(final) / max(len(targets), 1)),
        "decile_p_final_spread": float(ret["p_primary_residential_final"].max() - ret["p_primary_residential_final"].min()) if len(ret) else None,
        "median_price_eligible": float(eligible["sale_price"].median()) if len(eligible) else None,
        "median_price_final": float(final["sale_price"].median()) if len(final) else None,
        "sale_date_min": str(final_sorted["sale_date"].min().date()) if len(final_sorted) else "",
        "sale_date_max": str(final_sorted["sale_date"].max().date()) if len(final_sorted) else "",
        "history_lag_p10": float(lag.quantile(0.1)) if len(lag) else None,
        "history_lag_p50": float(lag.median()) if len(lag) else None,
        "history_lag_p90": float(lag.quantile(0.9)) if len(lag) else None,
        "table_path": str(table_path),
        "table_sha256": sha256_file(table_path),
        "wayne_is_not_detroit": key == "wayne",
        "written_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    waterfall.assign(county_key=key).to_csv(out_dir / "sale_validation_waterfall.csv", index=False)
    write_json(out_dir / "modeling_table_meta_dev.json", rec_out)
    print(json.dumps({k: rec_out[k] for k in ["county_key", "n_final_primary_residential", "decile_p_final_spread"]}, default=str), flush=True)
    return rec_out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--county-key", default="pilot")
    parser.add_argument("--end-date", default=DEV_END_DATE_DEFAULT)
    args = parser.parse_args()
    primary_codes = load_primary_residential_codes()
    if args.county_key == "pilot":
        keys = list(PILOT_KEYS)
    elif args.county_key == "all":
        keys = list(ALL_KEYS)
    else:
        keys = [args.county_key]
    pol = policy()
    rows = [build_county(k, JURISDICTION_BY_KEY[k]["fips"], pol, primary_codes, args.end_date) for k in keys]
    existing = []
    for k in ALL_KEYS:
        meta = OUTPUT / "modeling_tables" / k / "modeling_table_meta_dev.json"
        if meta.exists():
            existing.append(json.loads(meta.read_text()))
    if existing:
        ANALYSIS.joinpath("cohort").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(existing).to_csv(ANALYSIS / "cohort" / "modeling_table_summary_dev.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
