#!/usr/bin/env python3
"""2016-2025 ATTOM modeling tables + Recorder-to-model retention audit."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.other_counties_benchmars import (  # noqa: E402
    SALE_VALIDATION_DICTIONARY_PATH, SaleValidationPolicy, apply_sale_validation,
    attach_recorder_prior_sales, clean_fips, load_sale_validation_dictionary,
    match_history, normalize_property_use,
)
from analysis.berry_attom_validation_v3.scripts.v3_common import (  # noqa: E402
    ANALYSIS, ASSESSMENT_VALUE_COLUMNS, BROAD_RESIDENTIAL_RULE, COUNTIES, MIN_SALE_PRICE, OUTPUT,
    PROPERTY_USE_CODES, PROPERTY_USE_SET_NAMES, SALE_WINDOW, ST_LOUIS_CITY_FIPS, sha256_file,
    write_json,
)

FORBIDDEN = ASSESSMENT_VALUE_COLUMNS | {
    "tax_assessor_latitude", "tax_assessor_longitude", "tax_assessor_geoid", "LATITUDE", "LONGITUDE",
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


def retention_table(eligible: pd.DataFrame, window: pd.DataFrame, hist: pd.DataFrame, final: pd.DataFrame, key: str) -> pd.DataFrame:
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
            "county_key": key,
            "price_decile": int(dec) if pd.notna(dec) else -1,
            "n_eligible_recorder": int(len(g)),
            "p_in_2016_2025": float(g["TRANSACTIONID"].isin(win_ids).mean()) if "TRANSACTIONID" in g else np.nan,
            "p_safe_history": float(g["TRANSACTIONID"].isin(hist_ids).mean()) if "TRANSACTIONID" in g else np.nan,
            "p_final_model": float(g["TRANSACTIONID"].isin(fin_ids).mean()) if "TRANSACTIONID" in g else np.nan,
            "median_price": float(g["sale_price"].median()),
        })
    return pd.DataFrame(rows)


def property_use_profile(matched: pd.DataFrame, key: str) -> pd.DataFrame:
    """Per-property-use-code structural profile of the safe-history matched sales.

    Published so every include/exclude decision behind the broad_residential
    sensitivity cohort is auditable. Dewey ships no PROPERTYUSESTANDARDIZED code
    dictionary, so this table describes structure and never names a code.
    """
    frame = matched.copy()
    frame["use_code"] = normalize_property_use(frame["PROPERTYUSESTANDARDIZED"]).fillna("MISSING")
    total = max(len(frame), 1)

    def numeric(group: pd.DataFrame, column: str) -> pd.Series:
        """An absent column reads as all-missing rather than crashing the audit."""
        if column not in group.columns:
            return pd.Series(np.nan, index=group.index, dtype=float)
        return pd.to_numeric(group[column], errors="coerce")

    rows = []
    for code, g in frame.groupby("use_code"):
        units = numeric(g, "UNITSCOUNT")
        area = numeric(g, "AREABUILDING")
        rows.append({
            "county_key": key,
            "use_code": code,
            "n_matched_sales": int(len(g)),
            "share_of_matched_rows": float(len(g) / total),
            "share_area_building_present": float(area.notna().mean()),
            "share_year_built_present": float(numeric(g, "YEARBUILT").notna().mean()),
            "median_units_count": float(units.median()) if units.notna().any() else np.nan,
            "median_area_building": float(area.median()),
            "median_bedrooms": float(numeric(g, "BEDROOMSCOUNT").median()),
            "median_sale_price": float(numeric(g, "sale_price").median()),
        })
    profile = pd.DataFrame(rows).sort_values("n_matched_sales", ascending=False)
    profile["in_primary_385"] = profile["use_code"].isin(set(PROPERTY_USE_CODES))
    profile["in_broad_residential"] = broad_residential_mask(profile)
    return profile.reset_index(drop=True)


def broad_residential_mask(profile: pd.DataFrame) -> pd.Series:
    """Apply BROAD_RESIDENTIAL_RULE to a property-use profile table.

    Structural conditions only: the code's matched sales must mostly carry
    building area and a year built, must have a *positive* median building area
    (a zero median means land, not a dwelling), must not look multi-family
    (median UNITSCOUNT <= 4, or absent), and must carry a minimum share of the
    county's matched rows so single stray codes cannot enter. No price,
    sale-outcome, or model-performance quantity appears anywhere in this rule.
    """
    rule = BROAD_RESIDENTIAL_RULE
    units = profile["median_units_count"]
    area = profile["median_area_building"]
    return (
        profile["share_area_building_present"].ge(rule["min_share_area_building_present"])
        & area.notna() & area.ge(rule["min_median_area_building"])
        & profile["share_year_built_present"].ge(rule["min_share_year_built_present"])
        & (units.isna() | units.le(rule["max_median_units_count"]))
        & profile["share_of_matched_rows"].ge(rule["min_share_of_matched_rows"])
    )


def use_set_retention_by_decile(matched: pd.DataFrame, use_sets: dict, key: str) -> pd.DataFrame:
    """P(kept by each property-use set | safe-history sale-price decile).

    The primary cohort's value-dependence is the point of this table: a
    value-dependent use filter moves vertical-equity diagnostics.
    """
    frame = matched.copy()
    frame["sale_price"] = pd.to_numeric(frame["sale_price"], errors="coerce")
    frame = frame.loc[frame["sale_price"].gt(0)].copy()
    frame["use_code"] = normalize_property_use(frame["PROPERTYUSESTANDARDIZED"]).fillna("MISSING")
    frame["price_decile"] = pd.qcut(frame["sale_price"], 10, labels=False, duplicates="drop")
    rows = []
    for dec, g in frame.groupby("price_decile"):
        row = {
            "county_key": key,
            "price_decile": int(dec) if pd.notna(dec) else -1,
            "n_safe_history": int(len(g)),
            "median_price": float(g["sale_price"].median()),
        }
        for name, codes in use_sets.items():
            row[f"p_kept_{name}"] = float(g["use_code"].isin(codes).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def build_county(
    key: str,
    fips: str,
    pol: SaleValidationPolicy,
    use_set: str = "primary_385",
    audit_only: bool = False,
) -> dict:
    rec_path = OUTPUT / "cache" / key / "recorder.parquet"
    hist_path = OUTPUT / "cache" / key / "history.parquet"
    if not rec_path.exists():
        return {"county_key": key, "status": "SKIPPED_NO_CACHE"}
    rec = pq.read_table(rec_path).to_pandas()
    rec["sale_date"] = pd.to_datetime(rec["INSTRUMENTDATE"], errors="coerce").fillna(
        pd.to_datetime(rec["RECORDINGDATE"], errors="coerce")
    )
    rec["sale_price"] = pd.to_numeric(rec["TRANSFERAMOUNT"], errors="coerce")
    rec["ATTOMID"] = pd.to_numeric(rec["ATTOMID"], errors="coerce").astype("Int64")
    rec = rec.loc[clean_fips(rec["DOCUMENTRECORDINGCOUNTYFIPS"]).eq(fips)].copy()
    if fips == "29189":
        assert not clean_fips(rec["DOCUMENTRECORDINGCOUNTYFIPS"]).eq(ST_LOUIS_CITY_FIPS).any()
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
            assert not clean_fips(hist["SITUSSTATECOUNTYFIPS"]).eq(ST_LOUIS_CITY_FIPS).any()
    matched = match_history(targets, hist)
    assert (matched["assessed_through"] < matched["sale_date"]).all()
    n_hist = len(matched)
    ANALYSIS.joinpath("feature_audit").mkdir(parents=True, exist_ok=True)
    profile = property_use_profile(matched, key)
    profile.to_csv(ANALYSIS / "feature_audit" / f"{key}_property_use_profile.csv", index=False)
    primary_codes = set(PROPERTY_USE_CODES)
    broad_codes = set(profile.loc[profile["in_broad_residential"], "use_code"])
    use_sets = {"primary_385": primary_codes, "broad_residential": broad_codes}
    use_set_retention_by_decile(matched, use_sets, key).to_csv(
        ANALYSIS / "feature_audit" / f"{key}_property_use_retention_by_decile.csv", index=False,
    )
    if audit_only:
        # Property-use audits only. The primary modeling tables must not be
        # rewritten: the panel freeze, held-out baselines and Direct path were
        # all computed against their recorded sha256.
        summary = {
            "county_key": key,
            "status": "AUDIT_ONLY",
            "n_safe_history_matched": n_hist,
            "n_primary_385": int(
                normalize_property_use(matched["PROPERTYUSESTANDARDIZED"]).isin(primary_codes).sum()
            ),
            "n_broad_residential": int(
                normalize_property_use(matched["PROPERTYUSESTANDARDIZED"]).isin(broad_codes).sum()
            ),
            "broad_residential_codes": sorted(broad_codes),
        }
        print(json.dumps(summary, default=str), flush=True)
        return summary
    use = primary_codes if use_set == "primary_385" else broad_codes
    final = matched.loc[normalize_property_use(matched["PROPERTYUSESTANDARDIZED"]).isin(use)].copy()
    final = attach_recorder_prior_sales(final, [rec_path], fips, pol)
    if "recorder_prior_sale_age_years" in final.columns:
        bad = final["recorder_prior_sale_age_years"].fillna(1) <= 0
        if bad.any():
            raise RuntimeError(f"{key}: non-positive prior-sale age on {int(bad.sum())} rows")
    out_dir = OUTPUT / "modeling_tables" / key
    out_dir.mkdir(parents=True, exist_ok=True)
    # The sensitivity cohort writes beside the primary table, never over it.
    suffix = "" if use_set == "primary_385" else f"_{use_set}"
    table_path = out_dir / f"history_market_core{suffix}.parquet"
    final.sort_values("sale_date").reset_index(drop=True).to_parquet(table_path, index=False)
    ret = retention_table(eligible, targets, matched, final, key)
    ret.to_csv(
        ANALYSIS / "feature_audit" / f"{key}_modeling_retention_by_decile{suffix}.csv", index=False,
    )
    lag = final["history_lag_days"] if "history_lag_days" in final else pd.Series(dtype=float)
    rec_out = {
        "county_key": key,
        "fips": fips,
        "status": "OK",
        "property_use_set": use_set,
        "property_use_codes": sorted(use),
        "n_property_use_codes": len(use),
        "freeze_status": (
            "PRIMARY_FROZEN_COHORT" if use_set == "primary_385"
            else "SENSITIVITY_ONLY_NOT_A_FREEZE_REVISION"
        ),
        "n_recorder_raw": int(len(rec)),
        "n_sale_validation_eligible": int(len(eligible)),
        "n_window_2016_2025": int(len(targets)),
        "n_strict_history_match": n_hist,
        "n_final_model": int(len(final)),
        "retention_eligible_to_final": float(len(final) / max(len(eligible), 1)),
        "retention_window_to_final": float(len(final) / max(len(targets), 1)),
        "decile_p_final_spread": float(ret["p_final_model"].max() - ret["p_final_model"].min()) if len(ret) else None,
        "median_price_eligible": float(eligible["sale_price"].median()) if len(eligible) else None,
        "median_price_final": float(final["sale_price"].median()) if len(final) else None,
        "sale_date_min": str(final["sale_date"].min().date()) if len(final) else "",
        "sale_date_max": str(final["sale_date"].max().date()) if len(final) else "",
        "history_lag_p10": float(lag.quantile(0.1)) if len(lag) else None,
        "history_lag_p50": float(lag.median()) if len(lag) else None,
        "history_lag_p90": float(lag.quantile(0.9)) if len(lag) else None,
        "share_lag_gt_1yr": float((final["history_lag_years"] > 1).mean()) if len(final) and "history_lag_years" in final else None,
        "share_lag_gt_2yr": float((final["history_lag_years"] > 2).mean()) if len(final) and "history_lag_years" in final else None,
        "share_lag_gt_3yr": float((final["history_lag_years"] > 3).mean()) if len(final) and "history_lag_years" in final else None,
        "table_path": str(table_path),
        "table_sha256": sha256_file(table_path),
        "wayne_is_not_detroit": key == "wayne",
        "written_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if use_set == "primary_385":
        waterfall.assign(county_key=key).to_csv(out_dir / "sale_validation_waterfall.csv", index=False)
    write_json(out_dir / f"modeling_table_meta{suffix}.json", rec_out)
    print(json.dumps({k: rec_out[k] for k in ["county_key", "property_use_set", "n_final_model", "decile_p_final_spread"]}, default=str), flush=True)
    return rec_out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--county-key", default="all")
    parser.add_argument(
        "--property-use-set", default="primary_385", choices=PROPERTY_USE_SET_NAMES,
        help="primary_385 is the frozen cohort; broad_residential is a labeled sensitivity only.",
    )
    parser.add_argument(
        "--audit-only", action="store_true",
        help="Write property-use audits only; never rewrite a modeling table.",
    )
    args = parser.parse_args()
    pol = policy()
    counties = [c for c in COUNTIES if args.county_key in {"all", c["key"]}]
    ANALYSIS.joinpath("feature_audit").mkdir(parents=True, exist_ok=True)
    for c in counties:
        build_county(c["key"], c["fips"], pol, args.property_use_set, args.audit_only)
    if args.audit_only:
        return 0
    suffix = "" if args.property_use_set == "primary_385" else f"_{args.property_use_set}"
    existing = []
    for c in COUNTIES:
        meta = OUTPUT / "modeling_tables" / c["key"] / f"modeling_table_meta{suffix}.json"
        if meta.exists():
            existing.append(json.loads(meta.read_text()))
    if existing:
        pd.DataFrame(existing).to_csv(
            ANALYSIS / "feature_audit" / f"modeling_table_summary{suffix}.csv", index=False,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
