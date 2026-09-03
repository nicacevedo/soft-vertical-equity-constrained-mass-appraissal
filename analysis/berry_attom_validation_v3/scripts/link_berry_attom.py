#!/usr/bin/env python3
"""Berry parcel -> ATTOMID -> Recorder corroboration -> strict pre-sale history.

Unconditional AND conditional rates. Nested selection audit.
Never identifies a parcel from price/date alone.
Never silently collapses one-to-many APN maps.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from analysis.berry_cmf_validation.scripts.reproduce_berry import cmf_cod_prd_prb, cmf_reformat  # noqa: E402
from analysis.berry_attom_validation_v2.scripts.apn_normalize import normalize_apn_series  # noqa: E402
from scripts.other_counties_benchmars import match_history  # noqa: E402
from analysis.berry_attom_validation_v3.scripts.v3_common import (  # noqa: E402
    ANALYSIS, COUNTIES, OUTPUT, STL_LINKAGE_END, STL_LINKAGE_START, UNIQUE_APN_STATUSES,
    write_json,
)

LINK = ANALYSIS / "linkage"
FIG = ANALYSIS / "figures"
HISTORY_APN_COLS = ["ATTOMID", "PARCELNUMBERFORMATTED", "PARCELNUMBERPREVIOUS", "PARCELNUMBERRAW"]
HISTORY_ASOF_COLS = [
    "ATTOMID", "ASSESSORHISTORYYEAR", "PARCELNUMBERFORMATTED", "PARCELNUMBERPREVIOUS", "PARCELNUMBERRAW",
    "PROPERTYJURISDICTIONNAME", "MINORCIVILDIVISIONNAME", "PROPERTYADDRESSCITY",
    "TAXASSESSEDVALUETOTAL", "TAXMARKETVALUETOTAL",
]


def eligible_berry(key: str, berry: pd.DataFrame) -> pd.DataFrame:
    b = berry.copy()
    b["berry_sale_date"] = pd.to_datetime(b["berry_sale_date"])
    if key == "st_louis_county":
        start, end = pd.Timestamp(STL_LINKAGE_START), pd.Timestamp(STL_LINKAGE_END)
        b["attom_linkage_cohort"] = b["berry_sale_date"].between(start, end)
        b["eligible_for_attom_linkage"] = b["attom_linkage_cohort"] & b["berry_sale_date"].notna()
    else:
        b["attom_linkage_cohort"] = True
        b["eligible_for_attom_linkage"] = b["berry_sale_date"].notna()
    return b


def history_apn_map(hist: pd.DataFrame) -> pd.DataFrame:
    rows = []
    hist = hist.copy()
    hist["ATTOMID"] = pd.to_numeric(hist["ATTOMID"], errors="coerce").astype("Int64")
    for col, kind in [
        ("PARCELNUMBERRAW", "raw"),
        ("PARCELNUMBERFORMATTED", "formatted"),
        ("PARCELNUMBERPREVIOUS", "previous"),
    ]:
        if col not in hist.columns:
            continue
        tmp = hist[["ATTOMID", col]].dropna()
        tmp["apn_norm"] = normalize_apn_series(tmp[col])
        tmp = tmp.loc[tmp["apn_norm"].notna() & tmp["ATTOMID"].notna()]
        tmp["apn_source"] = kind
        tmp["apn_raw"] = tmp[col].astype(str)
        rows.append(tmp[["ATTOMID", "apn_raw", "apn_norm", "apn_source"]].drop_duplicates())
    return pd.concat(rows, ignore_index=True).drop_duplicates() if rows else pd.DataFrame(
        columns=["ATTOMID", "apn_raw", "apn_norm", "apn_source"]
    )


def classify_matches(berry: pd.DataFrame, apn_map: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    berry = berry.copy()
    berry["apn_norm"] = normalize_apn_series(berry["berry_parcel_raw"])
    berry["berry_parcel_raw_str"] = berry["berry_parcel_raw"].astype(str).str.strip()
    m = berry[["berry_txn_id", "apn_norm", "berry_parcel_raw_str"]].merge(
        apn_map, on="apn_norm", how="left",
    )
    status = []
    for txn, g in m.groupby("berry_txn_id", sort=False):
        ids = pd.unique(g["ATTOMID"].dropna())
        sources = set(g.loc[g["ATTOMID"].notna(), "apn_source"].dropna())
        if len(ids) == 0:
            status.append((txn, "NO_APN_MATCH", pd.NA, None))
        elif len(ids) > 1:
            status.append((txn, "AMBIGUOUS_APN", pd.NA, ",".join(sorted(sources))))
        else:
            berry_raw = str(g["berry_parcel_raw_str"].iloc[0])
            raw_hits = g.loc[g["apn_source"].eq("raw") & g["apn_raw"].astype(str).str.strip().eq(berry_raw)]
            if len(raw_hits) and pd.notna(raw_hits["ATTOMID"].iloc[0]):
                st = "EXACT_RAW_APN"
            elif "previous" in sources and "raw" not in sources and "formatted" not in sources:
                st = "EXACT_PREVIOUS_APN"
            elif "previous" in sources and "raw" not in sources:
                st = "EXACT_PREVIOUS_APN"
            else:
                st = "EXACT_NORMALIZED_APN"
            status.append((txn, st, ids[0], ",".join(sorted(sources))))
    st = pd.DataFrame(status, columns=["berry_txn_id", "match_status", "ATTOMID", "apn_sources"])
    return berry.merge(st, on="berry_txn_id", how="left"), m


def recorder_corroboration(berry: pd.DataFrame, rec: pd.DataFrame) -> pd.DataFrame:
    rec = rec.copy()
    rec["ATTOMID"] = pd.to_numeric(rec["ATTOMID"], errors="coerce").astype("Int64")
    rec["instrument_date"] = pd.to_datetime(rec["INSTRUMENTDATE"], errors="coerce")
    rec["recording_date"] = pd.to_datetime(rec["RECORDINGDATE"], errors="coerce")
    rec["transfer_amount"] = pd.to_numeric(rec["TRANSFERAMOUNT"], errors="coerce")
    rec["rec_date"] = rec["instrument_date"].fillna(rec["recording_date"])
    linked = berry.loc[berry["match_status"].isin(UNIQUE_APN_STATUSES) & berry["ATTOMID"].notna()].copy()
    if linked.empty:
        return linked.assign(tier="NO_LINKED_PARCEL")
    rec = rec.loc[rec["ATTOMID"].isin(linked["ATTOMID"].unique())]
    rec = rec.loc[rec["rec_date"].notna(), [
        "ATTOMID", "TRANSACTIONID", "instrument_date", "recording_date", "rec_date", "transfer_amount",
        "ARMSLENGTHFLAG", "DOCUMENTTYPECODE",
    ]]
    cand = linked.merge(rec, on="ATTOMID", how="left")
    cand["date_diff_instrument"] = (cand["instrument_date"] - cand["berry_sale_date"]).dt.days.abs()
    cand["date_diff_recording"] = (cand["recording_date"] - cand["berry_sale_date"]).dt.days.abs()
    cand["date_diff"] = cand[["date_diff_instrument", "date_diff_recording"]].min(axis=1)
    cand["price_rel"] = (
        (cand["transfer_amount"] - cand["berry_sale_price"]).abs()
        / cand["berry_sale_price"].replace(0, np.nan)
    )
    cand["exact_date"] = cand["date_diff"].eq(0)
    cand["date_pm_1"] = cand["date_diff"].le(1)
    cand["date_pm_7"] = cand["date_diff"].le(7)
    cand["date_pm_30"] = cand["date_diff"].le(30)
    cand["exact_price"] = cand["price_rel"].eq(0)
    cand["price_le_1pct"] = cand["price_rel"].le(0.01)
    cand["price_le_5pct"] = cand["price_rel"].le(0.05)

    def tier(g: pd.DataFrame) -> pd.Series:
        ok = g.loc[g["date_diff"].notna()]
        if len(ok) == 0:
            return pd.Series({"tier": "CONFLICT", "n_candidates": 0})
        high = ok.loc[ok["date_pm_7"].fillna(False) & ok["price_le_5pct"].fillna(False)]
        if len(high) == 1:
            return pd.Series({"tier": "TIER_1_HIGH_CONFIDENCE", "n_candidates": 1})
        if len(high) > 1:
            return pd.Series({"tier": "AMBIGUOUS", "n_candidates": int(len(high))})
        plaus = ok.loc[ok["date_pm_30"].fillna(False) | ok["price_le_5pct"].fillna(False)]
        if len(plaus) == 1:
            return pd.Series({"tier": "TIER_2_PLAUSIBLE", "n_candidates": 1})
        if len(plaus) > 1:
            return pd.Series({"tier": "AMBIGUOUS", "n_candidates": int(len(plaus))})
        return pd.Series({"tier": "CONFLICT", "n_candidates": int(len(ok))})

    tiers = cand.groupby("berry_txn_id", sort=False).apply(tier, include_groups=False).reset_index()
    return cand.merge(tiers, on="berry_txn_id", how="left")


def ratio_metrics(frame: pd.DataFrame) -> dict:
    if "berry_assessed_value" not in frame.columns or frame["berry_assessed_value"].notna().sum() < 30:
        return {"N": int(len(frame)), "COD": np.nan, "PRD": np.nan, "PRB": np.nan, "note": "no_berry_assessed_value"}
    # Build a 1-d table. The merged Berry frame may already contain SALE_PRICE,
    # and renaming berry_sale_price onto that name makes cmf_reformat see a
    # DataFrame rather than a Series.
    work = pd.DataFrame({
        "SALE_PRICE": pd.to_numeric(frame["berry_sale_price"], errors="coerce"),
        "ASSESSED_VALUE": pd.to_numeric(frame["berry_assessed_value"], errors="coerce"),
        "SALE_YEAR": pd.to_datetime(frame["berry_sale_date"], errors="coerce").dt.year,
    })
    ratios = cmf_reformat(work, "SALE_PRICE", "ASSESSED_VALUE", "SALE_YEAR", filter_data=False)
    out = cmf_cod_prd_prb(ratios)
    y = np.log(pd.to_numeric(frame["berry_sale_price"], errors="coerce"))
    r = pd.to_numeric(frame["berry_assessment_ratio"], errors="coerce")
    ok = y.notna() & r.notna() & np.isfinite(y) & np.isfinite(r)
    if ok.sum() >= 30:
        b = np.polyfit(y[ok], r[ok], 1)[0]
        out["beta_log"] = float(b)
    else:
        out["beta_log"] = np.nan
    return out


def attach_berry_history(berry: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    linked = berry.loc[berry["match_status"].isin(UNIQUE_APN_STATUSES) & berry["ATTOMID"].notna()].copy()
    if linked.empty:
        return linked
    hist = hist.copy()
    hist["ATTOMID"] = pd.to_numeric(hist["ATTOMID"], errors="coerce").astype("Int64")
    year = pd.to_numeric(hist["ASSESSORHISTORYYEAR"], errors="coerce").astype("Int64")
    hist["assessed_through"] = pd.to_datetime(year.astype("string") + "-12-31", errors="coerce")
    hist = hist.dropna(subset=["ATTOMID", "assessed_through"])
    left = linked.rename(columns={"berry_sale_date": "sale_date"})
    left["sale_date"] = pd.to_datetime(left["sale_date"])
    left["ATTOMID"] = pd.to_numeric(left["ATTOMID"], errors="coerce").astype("Int64")
    keep_hist = [c for c in HISTORY_ASOF_COLS if c in hist.columns] + ["assessed_through"]
    matched = match_history(left, hist[keep_hist].drop_duplicates())
    if len(matched) and "sale_date" in matched.columns:
        assert (matched["assessed_through"] < matched["sale_date"]).all()
    return matched


def nested_audit(berry: pd.DataFrame, conc: pd.DataFrame, safe: pd.DataFrame) -> pd.DataFrame:
    elig = berry.loc[berry["eligible_for_attom_linkage"]].copy()
    n0 = set(elig["berry_txn_id"])
    unique = set(elig.loc[elig["match_status"].isin(UNIQUE_APN_STATUSES), "berry_txn_id"])
    if "tier" in conc and len(conc):
        t1 = set(conc.drop_duplicates("berry_txn_id").loc[
            conc.drop_duplicates("berry_txn_id")["tier"].eq("TIER_1_HIGH_CONFIDENCE"), "berry_txn_id"
        ])
    else:
        t1 = set()
    safe_ids = set(safe["berry_txn_id"]) if len(safe) and "berry_txn_id" in safe else set()
    stages = {
        0: n0,
        1: unique,
        2: unique & t1,
        3: unique & safe_ids,
        4: unique & t1 & safe_ids,
    }
    rows = []
    elig = elig.copy()
    elig["log_price"] = np.log(elig["berry_sale_price"].clip(lower=1))
    elig["year"] = pd.to_datetime(elig["berry_sale_date"]).dt.year
    elig["price_decile"] = pd.qcut(elig["berry_sale_price"], 10, labels=False, duplicates="drop")
    names = {
        0: "full_eligible_berry_cohort",
        1: "unique_apn_attomid",
        2: "high_confidence_recorder",
        3: "safe_history",
        4: "fully_validated",
    }
    for s, ids in stages.items():
        sub = elig.loc[elig["berry_txn_id"].isin(ids)]
        mets = ratio_metrics(sub)
        for dec, g in elig.groupby("price_decile"):
            rows.append({
                "stage": s,
                "stage_name": names[s],
                "n_stage": int(len(sub)),
                "n_eligible": int(len(elig)),
                "unconditional_rate": float(len(sub) / max(len(elig), 1)),
                "price_decile": int(dec) if pd.notna(dec) else -1,
                "p_stage_given_decile": float(g["berry_txn_id"].isin(ids).mean()),
                "COD": mets.get("COD"),
                "PRD": mets.get("PRD"),
                "PRB": mets.get("PRB"),
                "beta_log": mets.get("beta_log"),
                "median_log_price": float(sub["log_price"].median()) if len(sub) else np.nan,
                "year_min": int(sub["year"].min()) if len(sub) else None,
                "year_max": int(sub["year"].max()) if len(sub) else None,
            })
    return pd.DataFrame(rows)


def process_jurisdiction(county: dict) -> dict:
    key, fips, berry_file = county["key"], county["fips"], county["berry_file"]
    berry = eligible_berry(key, pd.read_parquet(ANALYSIS / "berry_reproduction" / berry_file))
    hist_path = OUTPUT / "cache" / key / "history.parquet"
    rec_path = OUTPUT / "cache" / key / "recorder.parquet"
    hist_apn = pq.read_table(hist_path, columns=HISTORY_APN_COLS).to_pandas()
    rec = pq.read_table(rec_path).to_pandas()
    work = berry.loc[berry["eligible_for_attom_linkage"]].copy()
    apn_map = history_apn_map(hist_apn)
    work, candidates = classify_matches(work, apn_map)
    conc = recorder_corroboration(work, rec)
    schema = pq.ParquetFile(hist_path).schema_arrow.names
    hist_asof = pq.read_table(hist_path, columns=[c for c in HISTORY_ASOF_COLS if c in schema]).to_pandas()
    safe = attach_berry_history(work, hist_asof)
    nested = nested_audit(work, conc, safe)
    n_elig = int(len(work))
    unique_n = int(work["match_status"].isin(UNIQUE_APN_STATUSES).sum())
    if "tier" in conc and len(conc):
        one = conc.drop_duplicates("berry_txn_id")
        t1_n = int(one["tier"].eq("TIER_1_HIGH_CONFIDENCE").sum())
    else:
        t1_n = 0
        one = pd.DataFrame()
    safe_n = int(len(safe))
    fully = 0
    if len(safe) and "berry_txn_id" in safe and "tier" in conc:
        t1_ids = set(one.loc[one["tier"].eq("TIER_1_HIGH_CONFIDENCE"), "berry_txn_id"])
        fully = int(safe["berry_txn_id"].isin(t1_ids).sum())
    full_m = ratio_metrics(work)
    linked_m = ratio_metrics(work.loc[work["berry_txn_id"].isin(set(safe["berry_txn_id"]) if len(safe) else set())])
    lag = safe["history_lag_years"] if len(safe) and "history_lag_years" in safe else pd.Series(dtype=float)
    return {
        "key": key,
        "fips": fips,
        "berry_n_full_source": int(len(berry)),
        "berry_n_eligible_linkage": n_elig,
        "stl_linkage_window": f"{STL_LINKAGE_START}/{STL_LINKAGE_END}" if key == "st_louis_county" else "full_berry_sample",
        "r_apn_unconditional": unique_n / max(n_elig, 1),
        "r_transaction_unconditional": t1_n / max(n_elig, 1),
        "r_safe_history_unconditional": safe_n / max(n_elig, 1),
        "r_fully_validated_unconditional": fully / max(n_elig, 1),
        "r_transaction_conditional_on_unique_apn": t1_n / max(unique_n, 1),
        "r_safe_history_conditional_on_unique_apn": safe_n / max(unique_n, 1),
        "status_counts": work["match_status"].value_counts(dropna=False).to_dict(),
        "tier_counts": one["tier"].value_counts(dropna=False).to_dict() if len(one) else {},
        "berry_PRB_eligible": full_m.get("PRB"),
        "berry_PRB_safe_history": linked_m.get("PRB"),
        "berry_COD_eligible": full_m.get("COD"),
        "history_lag_p10": float(lag.quantile(0.1)) if len(lag) else None,
        "history_lag_p50": float(lag.median()) if len(lag) else None,
        "history_lag_p90": float(lag.quantile(0.9)) if len(lag) else None,
        "share_lag_gt_1yr": float((lag > 1).mean()) if len(lag) else None,
        "share_lag_gt_2yr": float((lag > 2).mean()) if len(lag) else None,
        "share_lag_gt_3yr": float((lag > 3).mean()) if len(lag) else None,
        "ambiguous_apn_n": int(work["match_status"].eq("AMBIGUOUS_APN").sum()),
        "work": work,
        "candidates": candidates,
        "concordance": conc,
        "safe": safe,
        "nested": nested.assign(jurisdiction=key),
    }


def main() -> int:
    LINK.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    summaries, nested_all, waters = [], [], []
    for county in COUNTIES:
        cache = OUTPUT / "cache" / county["key"] / "history.parquet"
        if not cache.exists():
            summaries.append({"key": county["key"], "status": "SKIPPED_NO_CACHE"})
            continue
        print("linking", county["key"], flush=True)
        out = process_jurisdiction(county)
        slim = {k: v for k, v in out.items() if k not in {"work", "candidates", "concordance", "safe", "nested"}}
        summaries.append(slim)
        nested_all.append(out["nested"])
        out["work"].assign(county_key=county["key"]).to_parquet(LINK / f"{county['key']}_crosswalk.parquet", index=False)
        if len(out["concordance"]):
            out["concordance"].to_parquet(LINK / f"{county['key']}_transaction_concordance.parquet", index=False)
        if len(out["safe"]):
            out["safe"].to_parquet(LINK / f"{county['key']}_safe_history.parquet", index=False)
        waters.append(pd.DataFrame([{
            "jurisdiction": county["key"],
            "berry_n_eligible": slim["berry_n_eligible_linkage"],
            "r_apn": slim["r_apn_unconditional"],
            "r_transaction": slim["r_transaction_unconditional"],
            "r_safe_history": slim["r_safe_history_unconditional"],
            "r_fully_validated": slim["r_fully_validated_unconditional"],
            "r_transaction_conditional_on_apn": slim["r_transaction_conditional_on_unique_apn"],
            "r_safe_history_conditional_on_apn": slim["r_safe_history_conditional_on_unique_apn"],
            "note": "unconditional rates use eligible Berry N as denominator, not the matched subset",
        }]))
        print(json.dumps(slim, default=str), flush=True)
    if waters:
        pd.concat(waters, ignore_index=True).to_csv(LINK / "unconditional_waterfall.csv", index=False)
    if nested_all:
        pd.concat(nested_all, ignore_index=True).to_csv(LINK / "nested_selection_audit.csv", index=False)
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        nest = pd.concat(nested_all, ignore_index=True)
        stage4 = nest.loc[nest["stage"].eq(4)]
        for j, g in stage4.groupby("jurisdiction"):
            ax.plot(g["price_decile"] + 1, g["p_stage_given_decile"], marker="o", label=j)
        ax.set_xlabel("sale-price decile")
        ax.set_ylabel("P(fully validated | decile)")
        ax.set_ylim(0, 1)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG / "fully_validated_rate_by_price_decile.pdf")
        plt.close(fig)
    write_json(LINK / "linkage_summary.json", summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
