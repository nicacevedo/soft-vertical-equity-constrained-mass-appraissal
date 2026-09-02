#!/usr/bin/env python3
"""Steps 5-8: Berry parcel -> ATTOMID -> Recorder corroboration -> strict pre-sale history.

Requires county caches. Never identifies a parcel from price/date alone.
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
from scripts.other_counties_benchmars import match_history  # noqa: E402
from analysis.berry_attom_validation_v2.scripts.apn_normalize import normalize_apn_series  # noqa: E402
from analysis.berry_attom_validation_v2.scripts.v2_common import ANALYSIS, OUTPUT, FIPS  # noqa: E402

BERRY_DIR = ANALYSIS / "berry_reproduction"
LINK = ANALYSIS / "linkage"
FEAT = ANALYSIS / "feature_audit"
FIG = ANALYSIS / "figures"
SRC = ANALYSIS / "source_concordance"

UNIQUE_STATUSES = ("EXACT_FORMATTED_APN", "EXACT_PREVIOUS_APN", "DOCUMENTED_NORMALIZED_APN")
HISTORY_APN_COLS = ["ATTOMID", "PARCELNUMBERFORMATTED", "PARCELNUMBERPREVIOUS", "PARCELNUMBERRAW"]
HISTORY_ASOF_COLS = [
    "ATTOMID", "ASSESSORHISTORYYEAR", "PARCELNUMBERFORMATTED", "PARCELNUMBERPREVIOUS",
    "PROPERTYJURISDICTIONNAME", "MINORCIVILDIVISIONNAME", "PROPERTYADDRESSCITY",
    "TAXASSESSEDVALUETOTAL", "TAXMARKETVALUETOTAL",
]


def history_apn_map(hist: pd.DataFrame) -> pd.DataFrame:
    rows = []
    hist = hist.copy()
    hist["ATTOMID"] = pd.to_numeric(hist["ATTOMID"], errors="coerce").astype("Int64")
    for col, kind in [
        ("PARCELNUMBERFORMATTED", "formatted"),
        ("PARCELNUMBERPREVIOUS", "previous"),
        ("PARCELNUMBERRAW", "raw"),
    ]:
        if col not in hist.columns:
            continue
        tmp = hist[["ATTOMID", col]].dropna()
        tmp["apn_norm"] = normalize_apn_series(tmp[col])
        tmp = tmp.loc[tmp["apn_norm"].notna() & tmp["ATTOMID"].notna(), ["ATTOMID", col, "apn_norm"]]
        tmp["apn_source"] = kind
        tmp["apn_raw"] = tmp[col].astype(str)
        rows.append(tmp[["ATTOMID", "apn_raw", "apn_norm", "apn_source"]].drop_duplicates())
    return pd.concat(rows, ignore_index=True).drop_duplicates()


def classify_matches(berry: pd.DataFrame, apn_map: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    berry = berry.copy()
    berry["apn_norm"] = normalize_apn_series(berry["berry_parcel_raw"])
    m = berry[["berry_txn_id", "apn_norm", "berry_parcel_raw"]].merge(
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
            if "formatted" in sources:
                st = "EXACT_FORMATTED_APN"
            elif "raw" in sources:
                st = "DOCUMENTED_NORMALIZED_APN"
            elif "previous" in sources:
                st = "EXACT_PREVIOUS_APN"
            else:
                st = "DOCUMENTED_NORMALIZED_APN"
            status.append((txn, st, ids[0], ",".join(sorted(sources))))
    st = pd.DataFrame(status, columns=["berry_txn_id", "match_status", "ATTOMID", "apn_sources"])
    berry = berry.merge(st, on="berry_txn_id", how="left")
    return berry, m


def _date_price_flags(cand: pd.DataFrame) -> pd.DataFrame:
    cand = cand.copy()
    cand["exact_date"] = cand["date_diff"].eq(0)
    cand["date_pm_1"] = cand["date_diff"].le(1)
    cand["date_pm_7"] = cand["date_diff"].le(7)
    cand["date_pm_30"] = cand["date_diff"].le(30)
    cand["exact_price"] = cand["price_rel"].eq(0)
    cand["price_le_1pct"] = cand["price_rel"].le(0.01)
    cand["price_le_5pct"] = cand["price_rel"].le(0.05)
    return cand


def recorder_corroboration(berry: pd.DataFrame, rec: pd.DataFrame) -> pd.DataFrame:
    rec = rec.copy()
    rec["ATTOMID"] = pd.to_numeric(rec["ATTOMID"], errors="coerce").astype("Int64")
    rec["instrument_date"] = pd.to_datetime(rec["INSTRUMENTDATE"], errors="coerce")
    rec["recording_date"] = pd.to_datetime(rec["RECORDINGDATE"], errors="coerce")
    rec["transfer_amount"] = pd.to_numeric(rec["TRANSFERAMOUNT"], errors="coerce")
    rec["rec_date"] = rec["instrument_date"].fillna(rec["recording_date"])
    linked = berry.loc[berry["match_status"].isin(UNIQUE_STATUSES) & berry["ATTOMID"].notna()].copy()
    if linked.empty:
        return linked.assign(tier="NO_LINKED_PARCEL")
    rec = rec.loc[rec["ATTOMID"].isin(linked["ATTOMID"].unique())].copy()
    rec = rec.loc[rec["rec_date"].notna(), [
        "ATTOMID", "TRANSACTIONID", "instrument_date", "recording_date", "rec_date", "transfer_amount",
        "ARMSLENGTHFLAG", "DOCUMENTTYPECODE",
    ]]
    cand = linked.merge(rec, on="ATTOMID", how="left", suffixes=("", "_rec"))
    cand["date_diff_instrument"] = (cand["instrument_date"] - cand["berry_sale_date"]).dt.days.abs()
    cand["date_diff_recording"] = (cand["recording_date"] - cand["berry_sale_date"]).dt.days.abs()
    cand["date_diff"] = cand[["date_diff_instrument", "date_diff_recording"]].min(axis=1)
    cand["price_rel"] = (
        (cand["transfer_amount"] - cand["berry_sale_price"]).abs()
        / cand["berry_sale_price"].replace(0, np.nan)
    )
    cand = _date_price_flags(cand)

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
    work = frame.rename(columns={
        "berry_sale_price": "SALE_PRICE", "berry_assessed_value": "ASSESSED_VALUE",
    }).assign(SALE_YEAR=pd.to_datetime(frame["berry_sale_date"]).dt.year)
    ratios = cmf_reformat(work, "SALE_PRICE", "ASSESSED_VALUE", "SALE_YEAR", filter_data=False)
    return cmf_cod_prd_prb(ratios)


def smd(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    denom = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    if not np.isfinite(denom) or denom == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / denom)


def selection_bias(berry: pd.DataFrame) -> tuple[pd.DataFrame, dict, bool]:
    b = berry.copy()
    b["matched"] = b["match_status"].isin(UNIQUE_STATUSES)
    b["log_price"] = np.log(b["berry_sale_price"].clip(lower=1))
    b["year"] = pd.to_datetime(b["berry_sale_date"]).dt.year
    b["price_decile"] = pd.qcut(b["berry_sale_price"], 10, labels=False, duplicates="drop")
    rows = []
    for dec, g in b.groupby("price_decile"):
        rows.append({
            "jurisdiction": b["jurisdiction"].iloc[0],
            "price_decile": int(dec) if pd.notna(dec) else -1,
            "n": int(len(g)),
            "n_matched": int(g["matched"].sum()),
            "p_match": float(g["matched"].mean()),
            "median_price": float(g["berry_sale_price"].median()),
            "median_berry_ratio": float(g["berry_assessment_ratio"].median())
            if "berry_assessment_ratio" in g else np.nan,
        })
    decile = pd.DataFrame(rows)
    matched = b.loc[b["matched"]]
    unmatched = b.loc[~b["matched"]]
    full_m = ratio_metrics(b)
    match_m = ratio_metrics(matched)
    prb_flip = (
        np.isfinite(full_m.get("PRB", np.nan))
        and np.isfinite(match_m.get("PRB", np.nan))
        and np.sign(full_m["PRB"]) != 0
        and np.sign(match_m["PRB"]) != 0
        and np.sign(full_m["PRB"]) != np.sign(match_m["PRB"])
    )
    match_spread = float(decile["p_match"].max() - decile["p_match"].min()) if len(decile) else np.nan
    flag = bool(prb_flip or (np.isfinite(match_spread) and match_spread > 0.25))
    summary = {
        "jurisdiction": b["jurisdiction"].iloc[0],
        "n_full": int(len(b)),
        "n_matched": int(len(matched)),
        "n_unmatched": int(len(unmatched)),
        "p_match": float(b["matched"].mean()),
        "smd_log_price": smd(matched["log_price"], unmatched["log_price"]),
        "smd_berry_ratio": smd(matched.get("berry_assessment_ratio"), unmatched.get("berry_assessment_ratio"))
        if "berry_assessment_ratio" in b else np.nan,
        "match_rate_decile_spread": match_spread,
        "full_COD": full_m.get("COD"), "full_PRD": full_m.get("PRD"), "full_PRB": full_m.get("PRB"),
        "matched_COD": match_m.get("COD"), "matched_PRD": match_m.get("PRD"), "matched_PRB": match_m.get("PRB"),
        "prb_sign_flip": prb_flip,
        "linkage_selection_flag": flag,
        "flag_rule": "PRB sign flip OR price-decile match-rate spread > 0.25 (diagnostic, not a tuned cutoff)",
    }
    return decile, summary, flag


def attach_berry_history(berry: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    linked = berry.loc[berry["match_status"].isin(UNIQUE_STATUSES) & berry["ATTOMID"].notna()].copy()
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
    if "sale_date" in matched.columns and "assessed_through" in matched.columns:
        assert (matched["assessed_through"] < matched["sale_date"]).all(), "history row is not strictly before sale"
    return matched


def concordance_diagnostics(conc: pd.DataFrame) -> dict:
    one = conc.drop_duplicates("berry_txn_id") if "berry_txn_id" in conc and len(conc) else conc
    if one.empty or "date_diff" not in one:
        return {}
    best = conc.sort_values(["berry_txn_id", "date_diff", "price_rel"]).drop_duplicates("berry_txn_id")
    return {
        "share_exact_date": float(best["exact_date"].mean()) if "exact_date" in best else np.nan,
        "share_date_pm_1": float(best["date_pm_1"].mean()) if "date_pm_1" in best else np.nan,
        "share_date_pm_7": float(best["date_pm_7"].mean()) if "date_pm_7" in best else np.nan,
        "share_date_pm_30": float(best["date_pm_30"].mean()) if "date_pm_30" in best else np.nan,
        "share_exact_price": float(best["exact_price"].mean()) if "exact_price" in best else np.nan,
        "share_price_le_1pct": float(best["price_le_1pct"].mean()) if "price_le_1pct" in best else np.nan,
        "share_price_le_5pct": float(best["price_le_5pct"].mean()) if "price_le_5pct" in best else np.nan,
    }


def plot_match_rate(deciles: list[pd.DataFrame]) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for d in deciles:
        ax.plot(d["price_decile"] + 1, d["p_match"], marker="o", label=d["jurisdiction"].iloc[0])
    ax.set_xlabel("Berry sale-price decile (1=lowest)")
    ax.set_ylabel("P(unique APN match)")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("ATTOM parcel-match rate by Berry price decile")
    fig.tight_layout()
    fig.savefig(FIG / "linkage_match_rate_by_price_decile.pdf")
    plt.close(fig)


def process_jurisdiction(key: str, berry_file: str, fips: str) -> dict:
    berry = pd.read_parquet(BERRY_DIR / berry_file)
    hist_path = OUTPUT / "cache" / key / "history.parquet"
    rec_path = OUTPUT / "cache" / key / "recorder.parquet"
    hist_apn = pq.read_table(hist_path, columns=[c for c in HISTORY_APN_COLS if True]).to_pandas()
    rec = pq.read_table(rec_path).to_pandas()
    apn_map = history_apn_map(hist_apn)
    berry, candidates = classify_matches(berry, apn_map)
    conc = recorder_corroboration(berry, rec)
    hist_asof = pq.read_table(
        hist_path, columns=[c for c in HISTORY_ASOF_COLS if c in pq.ParquetFile(hist_path).schema_arrow.names],
    ).to_pandas()
    safe = attach_berry_history(berry, hist_asof)
    water = berry["match_status"].value_counts(dropna=False).rename_axis("stage").reset_index(name="n")
    water.insert(0, "jurisdiction", key)
    if "tier" in conc:
        tw = conc.drop_duplicates("berry_txn_id")["tier"].value_counts(dropna=False).rename_axis("stage").reset_index(name="n")
        tw.insert(0, "jurisdiction", key)
        water = pd.concat([water, tw], ignore_index=True)
    decile, bias_sum, flag = selection_bias(berry)
    city_counts = {}
    if "PROPERTYJURISDICTIONNAME" in hist_asof.columns:
        city_counts = (
            hist_asof["PROPERTYJURISDICTIONNAME"].astype("string").value_counts(dropna=False).head(15).to_dict()
        )
    return {
        "key": key,
        "fips": fips,
        "berry_n": int(len(berry)),
        "parcel_match_n": int(berry["match_status"].isin(UNIQUE_STATUSES).sum()),
        "parcel_match_rate": float(berry["match_status"].isin(UNIQUE_STATUSES).mean()),
        "ambiguous_apn_n": int(berry["match_status"].eq("AMBIGUOUS_APN").sum()),
        "status_counts": berry["match_status"].value_counts(dropna=False).to_dict(),
        "tier_counts": conc.drop_duplicates("berry_txn_id")["tier"].value_counts(dropna=False).to_dict()
        if "tier" in conc and len(conc) else {},
        "high_conf_rate": float(
            conc.drop_duplicates("berry_txn_id")["tier"].eq("TIER_1_HIGH_CONFIDENCE").mean()
        ) if "tier" in conc and len(conc) else np.nan,
        "safe_history_n": int(len(safe)),
        "safe_history_rate_among_unique_apn": (
            float(len(safe) / max(int(berry["match_status"].isin(UNIQUE_STATUSES).sum()), 1))
        ),
        "history_lag_days_p50": float(safe["history_lag_days"].median()) if len(safe) else None,
        "concordance_diagnostics": concordance_diagnostics(conc),
        "jurisdiction_name_top": city_counts,
        "linkage_selection_flag": flag,
        "bias_summary": bias_sum,
        "berry": berry,
        "candidates": candidates,
        "concordance": conc,
        "waterfall": water,
        "bias": decile,
        "safe_history": safe,
    }


def main() -> int:
    specs = [
        ("wayne", "detroit_mi_transactions.parquet", FIPS["wayne"]),
        ("philadelphia", "philadelphia_pa_transactions.parquet", FIPS["philadelphia"]),
        ("st_louis_county", "st_louis_county_mo_transactions.parquet", FIPS["st_louis_county"]),
    ]
    berries, cands, concs, waters, biases, hist_audits, bias_rows = [], [], [], [], [], [], []
    summaries = []
    for key, fn, fips in specs:
        cache = OUTPUT / "cache" / key / "history.parquet"
        if not cache.exists():
            print("SKIP no cache", key, flush=True)
            summaries.append({"key": key, "status": "SKIPPED_NO_CACHE"})
            continue
        print("linking", key, flush=True)
        out = process_jurisdiction(key, fn, fips)
        summaries.append({k: v for k, v in out.items() if k not in {
            "berry", "candidates", "concordance", "waterfall", "bias", "safe_history",
        }})
        b = out["berry"].copy()
        b["county_key"] = key
        berries.append(b)
        cands.append(out["candidates"].assign(county_key=key))
        concs.append(out["concordance"].assign(county_key=key))
        waters.append(out["waterfall"])
        biases.append(out["bias"])
        bias_rows.append(out["bias_summary"])
        sh = out["safe_history"]
        if len(sh):
            hist_audits.append({
                "jurisdiction": key,
                "n_unique_apn_matches": int(out["parcel_match_n"]),
                "n_history_matched_berry": int(len(sh)),
                "strict_history_match_rate": float(out["safe_history_rate_among_unique_apn"]),
                "lag_p50": float(sh["history_lag_days"].median()),
                "lag_p90": float(sh["history_lag_days"].quantile(0.9)),
                "lag_p10": float(sh["history_lag_days"].quantile(0.1)),
                "share_lag_gt_1yr": float((sh["history_lag_years"] > 1).mean()),
                "share_lag_gt_2yr": float((sh["history_lag_years"] > 2).mean()),
                "share_lag_gt_3yr": float((sh["history_lag_years"] > 3).mean()),
            })
            sh.to_parquet(LINK / f"{key}_berry_attom_safe_history.parquet", index=False)
        print(json.dumps({k: summaries[-1][k] for k in summaries[-1] if k != "jurisdiction_name_top"}, default=str), flush=True)
    LINK.mkdir(parents=True, exist_ok=True)
    FEAT.mkdir(parents=True, exist_ok=True)
    SRC.mkdir(parents=True, exist_ok=True)
    if berries:
        pd.concat(berries, ignore_index=True).to_parquet(LINK / "berry_attomid_crosswalk.parquet", index=False)
        pd.concat(cands, ignore_index=True).to_parquet(LINK / "parcel_linkage_candidates.parquet", index=False)
        pd.concat(concs, ignore_index=True).to_parquet(LINK / "transaction_concordance.parquet", index=False)
        pd.concat(waters, ignore_index=True).to_csv(LINK / "linkage_waterfall.csv", index=False)
        pd.concat(biases, ignore_index=True).to_csv(LINK / "matched_unmatched_balance.csv", index=False)
        pd.DataFrame(bias_rows).to_csv(SRC / "berry_full_vs_attom_linkable.csv", index=False)
        plot_match_rate(biases)
    if hist_audits:
        pd.DataFrame(hist_audits).to_csv(FEAT / "history_temporal_audit.csv", index=False)
    (LINK / "linkage_summary.json").write_text(json.dumps(summaries, indent=2, default=str) + "\n")
    print("wrote linkage outputs", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
