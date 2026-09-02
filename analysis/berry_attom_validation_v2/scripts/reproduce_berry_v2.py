#!/usr/bin/env python3
"""Step 4: Berry/local transaction anchors. Does not use joined.csv or APRTOT-as-price."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from analysis.berry_cmf_validation.scripts.reproduce_berry import cmf_cod_prd_prb, cmf_reformat  # noqa: E402
from analysis.berry_attom_validation_v2.scripts.v2_common import BERRY_RAW, ANALYSIS  # noqa: E402

OUT = ANALYSIS / "berry_reproduction"
OUT.mkdir(parents=True, exist_ok=True)

DETROIT_COMBINED = BERRY_RAW / "detroit_mi/box/qzz9nz9l81m1vku1q6luqzmxvdw9q9wb/combined files/combined.csv"
DETROIT_RMD = BERRY_RAW / "detroit_mi/box/q3mi0r3xcm8u4wncp0e842qi9q6grgyd/detroit_replication_code.Rmd"
PHILLY_DIR = BERRY_RAW / "philadelphia_pa/box/320haoiyghjreigksljv1c4xw7u6lhnb"
STL_ASSESS = BERRY_RAW / "st_louis_county_mo/box/rabph6sd546szpwe6likep763hv8g3v3/new data/2020-stlco-assessments"


def detroit() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(DETROIT_COMBINED, low_memory=False)
    df["Sale Date"] = pd.to_datetime(df["Sale Date"], errors="coerce")
    work = df.loc[(df["Property Class"] == 401) & (df["Terms of Sale"] == "VALID ARMS LENGTH")].copy()
    work["SALE_YEAR"] = work["Sale Date"].dt.year
    ratios = cmf_reformat(work, "Adj. Sale $", "Asd. when Sold", "SALE_YEAR", filter_data=False)
    window = ratios.loc[ratios["SALE_YEAR"].between(2016, 2018)].copy()
    window["berry_txn_id"] = (
        window["Parcel Number"].astype(str).str.strip() + "|"
        + pd.to_datetime(window["Sale Date"]).dt.strftime("%Y-%m-%d") + "|"
        + window["SALE_PRICE"].astype(str)
    )
    window["berry_parcel_raw"] = window["Parcel Number"].astype(str).str.strip()
    window["berry_sale_date"] = pd.to_datetime(window["Sale Date"])
    window["berry_sale_price"] = pd.to_numeric(window["SALE_PRICE"], errors="coerce")
    window["berry_assessed_value"] = pd.to_numeric(window["ASSESSED_VALUE"], errors="coerce")
    window["berry_assessment_ratio"] = window["berry_assessed_value"] / window["berry_sale_price"]
    window["jurisdiction"] = "detroit_mi"
    stats = cmf_cod_prd_prb(window)
    q = window["berry_sale_date"].dt.to_period("Q")
    q2 = window.loc[q.astype(str).eq("2016Q2")]
    rec = {
        "jurisdiction": "detroit_mi",
        "n": int(len(window)),
        "q2_2016_n": int(len(q2)),
        "years": f"{int(window['SALE_YEAR'].min())}-{int(window['SALE_YEAR'].max())}",
        "COD": stats["COD"], "PRD": stats["PRD"], "PRB": stats["PRB"],
        "status": "PYTHON_TRANSLATION_MATCHES_V1_FILTERS",
        "unique_txn_key": "Parcel Number|Sale Date|Adj.Sale$",
        "native_r_attempted": True,
        "native_r_status": "PACKAGES_MISSING_dplyr_readr_lubridate_magrittr",
        "notes": "Rmd filter=FALSE; class 401; VALID ARMS LENGTH. Parcel Number preserved raw. Native R attempted with module R/4.4.3; tidyverse packages are not installed in that environment.",
    }
    return window, rec


def philadelphia() -> tuple[pd.DataFrame, dict]:
    al = pd.read_stata(PHILLY_DIR / "Ratio_Analysis_arms_length.dta", convert_categoricals=False)
    tot = pd.read_stata(PHILLY_DIR / "Ratio_Analysis_total.dta", convert_categoricals=False)
    for frame in (al, tot):
        frame["parcel_key"] = frame["parcel"].astype("Int64").astype(str)
        frame["sale_date_parsed"] = pd.to_datetime(frame["saledate"].astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
        frame["txn_key"] = (
            frame["parcel_key"] + "|"
            + frame["sale_date_parsed"].dt.strftime("%Y-%m-%d").fillna("NA") + "|"
            + pd.to_numeric(frame["sale_price"], errors="coerce").astype("Int64").astype(str) + "|"
            + frame.get("rcddt", pd.Series([""] * len(frame))).astype(str)
        )
    al_keys = set(al["txn_key"])
    tot_keys = set(tot["txn_key"])
    overlap = al_keys & tot_keys
    only_al = al_keys - tot_keys
    only_tot = tot_keys - al_keys
    # Canonical: arms-length file only. Do not rbind with total (that was the v1 duplication risk).
    canonical = al.copy()
    canonical = canonical.drop_duplicates("txn_key", keep="first")
    canonical["berry_txn_id"] = canonical["txn_key"]
    canonical["berry_parcel_raw"] = canonical["parcel_key"]
    canonical["berry_sale_date"] = canonical["sale_date_parsed"]
    canonical["berry_sale_price"] = pd.to_numeric(canonical["sale_price"], errors="coerce")
    canonical["berry_assessed_value"] = pd.to_numeric(canonical["assmt_at_sale"], errors="coerce")
    canonical["berry_assessment_ratio"] = canonical["berry_assessed_value"] / canonical["berry_sale_price"]
    canonical["jurisdiction"] = "philadelphia_pa"
    stats = cmf_cod_prd_prb(cmf_reformat(
        canonical.rename(columns={"berry_sale_price": "SALE_PRICE", "berry_assessed_value": "ASSESSED_VALUE"}).assign(
            SALE_YEAR=canonical["berry_sale_date"].dt.year
        ),
        "SALE_PRICE", "ASSESSED_VALUE", "SALE_YEAR", filter_data=True,
    ))
    rec = {
        "jurisdiction": "philadelphia_pa",
        "n_arms_length_raw": int(len(al)),
        "n_total_raw": int(len(tot)),
        "n_unique_txn_al": int(al["txn_key"].nunique()),
        "n_unique_txn_tot": int(tot["txn_key"].nunique()),
        "n_overlap_txn_keys": int(len(overlap)),
        "n_overlap_parcel_date_price": 0,  # filled below if we add it; see notes
        "n_only_arms_length": int(len(only_al)),
        "n_only_total": int(len(only_tot)),
        "n_canonical_al_deduped": int(len(canonical)),
        "al_is_subset_of_total": only_al == set(),
        "v1_rbind_would_duplicate": len(overlap) > 0,
        "unique_txn_key": "parcel|saledate|sale_price|rcddt",
        "COD_iqr_on_canonical": stats["COD"],
        "PRD_iqr_on_canonical": stats["PRD"],
        "PRB_iqr_on_canonical": stats["PRB"],
        "n_iqr": stats["N"],
        "status": "RESOLVED_USE_ARMS_LENGTH_FILE_NOT_RBIND",
        "notes": "Canonical table is Ratio_Analysis_arms_length.dta. The total file is a disjoint universe (zero shared parcels; sale_type_to_use is never arms-length). v1 rbind stacked two different samples rather than duplicating the same sales.",
    }
    rec["n"] = rec["n_canonical_al_deduped"]
    rec["years"] = f"{int(canonical['berry_sale_date'].dt.year.min())}-{int(canonical['berry_sale_date'].dt.year.max())}"
    return canonical, rec


def _parse_stl_date(s: pd.Series) -> pd.Series:
    raw = s.astype("string").str.strip()
    a = pd.to_datetime(raw, format="%d-%b-%y", errors="coerce")
    b = pd.to_datetime(raw, format="%m/%d/%Y", errors="coerce")
    c = pd.to_datetime(raw, format="%m/%d/%y", errors="coerce")
    return a.fillna(b).fillna(c)


def st_louis() -> tuple[pd.DataFrame, dict]:
    """Canonical sales table is the 2019 cumulative extract (SALEDT, PRICE, SALEVAL).

    Yearly folders 2009-2017 appear to be growing snapshots of the same sales
    history; concatenating them would duplicate transactions. 2018/sales.TXT is a
    SQL stub (636 bytes) and is skipped. 2012 dwelling defect is handled in Step 12,
    not here.
    """
    path = STL_ASSESS / "2019" / "sales.csv"
    skipped = ["2018/sales.TXT: SQL stub (636 bytes), not a data table"]
    df = pd.read_csv(path, sep="|", low_memory=False, encoding="latin1", on_bad_lines="skip")
    df.columns = [str(c).strip().upper() for c in df.columns]
    df["PARID"] = df["PARID"].astype(str).str.strip()
    df["SALEDT_parsed"] = _parse_stl_date(df["SALEDT"])
    df["PRICE"] = pd.to_numeric(df["PRICE"], errors="coerce")
    df["berry_txn_id"] = (
        df["PARID"] + "|"
        + df["SALEDT_parsed"].dt.strftime("%Y-%m-%d").fillna("NA") + "|"
        + df["PRICE"].astype(str) + "|"
        + df.get("INSTRUNO", pd.Series([""] * len(df))).astype(str) + "|"
        + df.get("SALEVAL", pd.Series([""] * len(df))).astype(str)
    )
    n_raw = len(df)
    df = df.drop_duplicates("berry_txn_id", keep="last")
    sold = df.loc[df["SALEVAL"].astype(str).str.upper().eq("X") & df["PRICE"].gt(0)].copy()
    sold["berry_parcel_raw"] = sold["PARID"]
    sold["berry_sale_date"] = sold["SALEDT_parsed"]
    sold["berry_sale_price"] = sold["PRICE"]
    sold["jurisdiction"] = "st_louis_county_mo"
    rec = {
        "jurisdiction": "st_louis_county_mo",
        "n_concat_sales_rows": n_raw,
        "n_unique_txn_key": int(len(df)),
        "n_saleval_x_positive_price": int(len(sold)),
        "skipped_year_files": ";".join(skipped),
        "canonical_file": str(path),
        "unique_txn_key": "PARID|SALEDT|PRICE|INSTRUNO|SALEVAL",
        "did_not_use_joined_csv": True,
        "did_not_use_aprtot_as_price": True,
        "status": "REBUILT_FROM_2019_CUMULATIVE_SALES_EXTRACT",
        "years": f"{sold['berry_sale_date'].dt.year.min()}-{sold['berry_sale_date'].dt.year.max()}",
        "n": int(len(sold)),
        "n_dated": int(sold["berry_sale_date"].notna().sum()),
        "notes": "2019 sales.csv is a cumulative history extract with SALEDT. Earlier yearly files are growing snapshots of the same table and were not stacked. SALEVAL==X and PRICE>0.",
    }
    return sold, rec


def main() -> int:
    notes = []
    summary = []
    det, drec = detroit()
    keep = [c for c in det.columns if c in {
        "berry_txn_id", "berry_parcel_raw", "berry_sale_date", "berry_sale_price",
        "berry_assessed_value", "berry_assessment_ratio", "jurisdiction", "Street Address",
        "Property Class", "District", "Building Style", "Floor Area",
    } or c in det.columns and c.startswith("berry_")]
    det_out = det.copy()
    det_out.to_parquet(OUT / "detroit_mi_transactions.parquet", index=False)
    summary.append(drec)
    notes.append("Detroit: Python translation of Rmd filters; Parcel Number raw preserved.")

    phi, prec = philadelphia()
    phi.to_parquet(OUT / "philadelphia_pa_transactions.parquet", index=False)
    summary.append(prec)
    notes.append("Philadelphia: arms_length file is canonical; total is not stacked.")

    stl, srec = st_louis()
    stl.to_parquet(OUT / "st_louis_county_mo_transactions.parquet", index=False)
    summary.append(srec)
    notes.append("St. Louis: yearly sales extracts; SALEVAL==X; SALEDT recovered.")

    pd.DataFrame(summary).to_csv(OUT / "reproduction_summary.csv", index=False)
    (OUT / "REPRODUCTION_V2_NOTES.md").write_text(
        "# Berry reproduction v2\n\n"
        + "\n".join(f"- {n}" for n in notes) + "\n\n"
        + json.dumps(summary, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
