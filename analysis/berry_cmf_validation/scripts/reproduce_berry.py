#!/usr/bin/env python3
"""Reproduce Berry/CMF local assessment-ratio studies.

Translates supplied CMF filters and cmfproperty::reformat_data / calc_iaao_stats.
Does not use the paper AVM metrics. Does not alter filters silently.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
RAW = REPO / "data" / "berry_cmf" / "raw"
OUT = REPO / "analysis" / "berry_cmf_validation" / "reproduction"
LOG = REPO / "analysis" / "berry_cmf_validation" / "logs" / "reproduction"
OUT.mkdir(parents=True, exist_ok=True)
LOG.mkdir(parents=True, exist_ok=True)


def cmf_reformat(df: pd.DataFrame, sale_col: str, assessment_col: str, sale_year_col: str,
                 filter_data: bool = True, source: str = "") -> pd.DataFrame:
    """Map of cmfproperty/R/reformat_data.R (IQR arms-length if filter_data)."""
    out = df.copy()
    if sale_col != "SALE_PRICE":
        out = out.rename(columns={sale_col: "SALE_PRICE"})
    if assessment_col != "ASSESSED_VALUE":
        out = out.rename(columns={assessment_col: "ASSESSED_VALUE"})
    if sale_year_col != "SALE_YEAR":
        out = out.rename(columns={sale_year_col: "SALE_YEAR"})
    out["SALE_PRICE"] = pd.to_numeric(out["SALE_PRICE"], errors="coerce")
    out["ASSESSED_VALUE"] = pd.to_numeric(out["ASSESSED_VALUE"], errors="coerce")
    out["SALE_YEAR"] = pd.to_numeric(out["SALE_YEAR"], errors="coerce")
    out["TAX_YEAR"] = out["SALE_YEAR"]
    out["RATIO"] = np.where(
        out["SALE_PRICE"].notna() & (out["SALE_PRICE"] > 100),
        out["ASSESSED_VALUE"] / out["SALE_PRICE"],
        np.nan,
    )
    # IQR fence by SALE_YEAR
    def _iqr_flag(g):
        r = g["RATIO"]
        q1, q3 = r.quantile(0.25), r.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        g = g.copy()
        g["arms_length_iqr"] = np.where(r.notna(), ((r >= lo) & (r <= hi)).astype(int), np.nan)
        return g
    out = out.groupby("SALE_YEAR", group_keys=False).apply(_iqr_flag)
    if filter_data:
        out = out.loc[out["arms_length_iqr"] == 1].copy()
        out = out.loc[out["RATIO"].notna()].copy()
    return out.reset_index(drop=True)


def cmf_cod_prd_prb(df: pd.DataFrame) -> Dict[str, float]:
    """Point estimates from cmfproperty/R/iaao_stats.R (no bootstrap)."""
    r = pd.to_numeric(df["RATIO"], errors="coerce").to_numpy(dtype=float)
    sp = pd.to_numeric(df["SALE_PRICE"], errors="coerce").to_numpy(dtype=float)
    av = pd.to_numeric(df["ASSESSED_VALUE"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(r) & np.isfinite(sp) & np.isfinite(av) & (sp > 0)
    r, sp, av = r[mask], sp[mask], av[mask]
    n = int(r.size)
    if n == 0:
        return {"N": 0, "COD": np.nan, "PRD": np.nan, "PRB": np.nan}
    med = float(np.median(r))
    cod = 100.0 * float(np.sum(np.abs(r - med))) / (n * med) if med != 0 else np.nan
    wmean = float(np.average(r, weights=sp))
    prd = float(np.mean(r) / wmean) if wmean != 0 else np.nan
    # PRB: ((ratio-median)/median) ~ log(0.5*(sale + av/median))/log(2)
    y = (r - med) / med
    x = np.log(0.5 * (sp + av / med)) / math.log(2.0)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() >= 3:
        coeff = np.polyfit(x[ok], y[ok], 1)[0]
        prb = float(coeff)
    else:
        prb = np.nan
    return {"N": n, "COD": round(cod, 4) if np.isfinite(cod) else np.nan,
            "PRD": round(prd, 4) if np.isfinite(prd) else np.nan,
            "PRB": round(prb, 4) if np.isfinite(prb) else np.nan}


def write_log(name: str, text: str) -> None:
    (LOG / name).write_text(text, encoding="utf-8")


def detroit() -> dict:
    src = "data/berry_cmf/raw/_shared/../detroit_mi combined.csv + detroit_replication_code.Rmd"
    path = RAW / "detroit_mi/box/qzz9nz9l81m1vku1q6luqzmxvdw9q9wb/combined files/combined.csv"
    df = pd.read_csv(path)
    # Rmd lines 23-26
    df["Sale Date"] = pd.to_datetime(df["Sale Date"], errors="coerce")
    df["SALE_YEAR"] = df["Sale Date"].dt.year
    sub = df.loc[(df["Property Class"] == 401) & (df["Terms of Sale"] == "VALID ARMS LENGTH")].copy()
    # Rmd: reformat_data(..., filter=FALSE)
    ratios = cmf_reformat(sub, "Adj. Sale $", "Asd. when Sold", "SALE_YEAR", filter_data=False,
                          source="detroit_replication_code.Rmd:28")
    ratios["Month"] = ratios["Sale Date"].dt.month
    ratios["Quarter"] = np.where(ratios["Month"] < 4, 1,
                          np.where(ratios["Month"] < 7, 2,
                          np.where(ratios["Month"] < 10, 3, 4)))
    ratios["quarter_num"] = ratios["Quarter"] + 4 * (ratios["SALE_YEAR"] - 2016)
    # report window Q2 2016 - Q1 2018
    window = ratios.loc[(ratios["Sale Date"] >= "2016-04-01") & (ratios["Sale Date"] <= "2018-03-31")].copy()
    before = window.loc[window["quarter_num"] <= 5].copy()
    after = window.loc[window["quarter_num"] > 5].copy()
    qb = window.groupby(["SALE_YEAR", "Quarter"]).size().reset_index(name="n")
    stats = {
        "before": cmf_cod_prd_prb(before),
        "after": cmf_cod_prd_prb(after),
        "window_n": int(len(window)),
        "class401_arms_n": int(len(sub)),
        "quarterly": qb.to_dict("records"),
    }
    write_log("detroit.log", json.dumps(stats, indent=2, default=str))
    orig = {"COD_before": 46.26, "COD_after": 50.08, "PRD_before": 1.30, "PRD_after": 1.35,
            "PRB_before": -0.26, "PRB_after": -0.45, "years": "2016Q2-2018Q1",
            "q2_2016_n": 975}
    rec_n_q2 = int(((window["SALE_YEAR"] == 2016) & (window["Quarter"] == 2)).sum())
    # relative discrepancies
    def rel(a, b):
        return None if b in (0, None) or not np.isfinite(a) else abs(a - b) / abs(b)
    status = "SUBSTANTIVE_MATCH"
    if abs(stats["before"]["COD"] - 46.26) > 3 or abs(stats["after"]["PRD"] - 1.35) > 0.05:
        status = "PARTIAL_REPRODUCTION"
    return {
        "jurisdiction": "detroit_mi",
        "original_reported_n": "Table1 quarterly; 2016Q2 N=975; total ~9653",
        "reproduced_n": stats["window_n"],
        "original_years": orig["years"],
        "reproduced_years": f"{int(window['SALE_YEAR'].min())}-{int(window['SALE_YEAR'].max())}",
        "original_key_results": json.dumps(orig),
        "reproduced_results": json.dumps(stats["before"] | {"after": stats["after"], "q2_2016_n": rec_n_q2}),
        "abs_rel_discrepancy": json.dumps({
            "COD_before_rel": rel(stats["before"]["COD"], 46.26),
            "COD_after_rel": rel(stats["after"]["COD"], 50.08),
            "PRD_before_rel": rel(stats["before"]["PRD"], 1.30),
            "PRD_after_rel": rel(stats["after"]["PRD"], 1.35),
            "PRB_before_rel": rel(stats["before"]["PRB"], -0.26),
            "PRB_after_rel": rel(stats["after"]["PRB"], -0.45),
            "q2_2016_n_abs": rec_n_q2 - 975,
        }),
        "reproduction_status": status,
        "explanation": (
            "Translated detroit_replication_code.Rmd: Property Class==401, Terms of Sale==VALID ARMS LENGTH, "
            "reformat_data(filter=FALSE), before=quarter_num<=5 (through 2017Q1). "
            "Original Table 2 from Detroit Ratio Study 2020.pdf. Inflation adjustment in reformat_data "
            "does not alter SALE_PRICE/RATIO used by calc_iaao_stats."
        ),
        "source_code": "detroit_replication_code.Rmd",
    }


def philadelphia() -> dict:
    raw_dir = RAW / "philadelphia_pa/box/320haoiyghjreigksljv1c4xw7u6lhnb"
    cols = ["sale_year", "year_assessed", "assmt_at_sale", "sale_price", "sale_type_to_use"]
    d1 = pd.read_stata(raw_dir / "Ratio_Analysis_arms_length.dta", columns=cols, convert_categoricals=False)
    d2 = pd.read_stata(raw_dir / "Ratio_Analysis_total.dta", columns=cols, convert_categoricals=False)
    df = pd.concat([d1, d2], ignore_index=True)
    df["arms_length_transaction"] = (df["sale_type_to_use"].astype(str).str.lower() == "arms length sale confirmed").astype(int)
    # page code then reformat_data with default filter_data=TRUE
    ratios = cmf_reformat(df, "sale_price", "assmt_at_sale", "sale_year", filter_data=True,
                          source="philadelphia-raw-data-code page")
    overall = cmf_cod_prd_prb(ratios)
    by_year = {int(y): cmf_cod_prd_prb(g) for y, g in ratios.groupby(ratios["SALE_YEAR"])}
    write_log("philadelphia.log", json.dumps({"overall": overall, "by_year": by_year, "n_raw_rbind": len(df)}, indent=2))
    return {
        "jurisdiction": "philadelphia_pa",
        "original_reported_n": "not extracted from nationwide CoreLogic HTML; local PDF not linked",
        "reproduced_n": overall["N"],
        "original_years": "sale_year in Stata files (inspected after load)",
        "reproduced_years": f"{int(ratios['SALE_YEAR'].min())}-{int(ratios['SALE_YEAR'].max())}" if len(ratios) else "",
        "original_key_results": "local report PDF not directly linked on CMF local-reports page",
        "reproduced_results": json.dumps({"overall": overall, "year_min_max_n": {str(k): v["N"] for k, v in by_year.items()}}),
        "abs_rel_discrepancy": "",
        "reproduction_status": "PARTIAL_REPRODUCTION",
        "explanation": (
            "Followed philadelphia-raw-data-code: rbind arms_length and total Stata files, then "
            "cmfproperty::reformat_data default IQR filter. Could not compare to a local CMF HTML/PDF "
            "with published N/COD/PRD/PRB; page Report link is nationwide CoreLogic HTML. "
            "Note: rbind of total+arms_length may duplicate transactions."
        ),
        "source_code": "https://propertytaxproject.uchicago.edu/philadelphia-raw-data-code/",
    }


def orleans() -> dict:
    # Reconstruct nola.R join from raw CSVs (RDS is processed intermediate).
    loc = RAW / "orleans_la/box/hz5rv02dpgw61e0qp3kvbyz0je1omi6v/nola_data/raw"
    properties = pd.read_csv(loc / "properties.csv", low_memory=False)
    sales = pd.read_csv(loc / "sales.csv", low_memory=False)
    values = pd.read_csv(loc / "values.csv", low_memory=False)
    values = values.loc[values["assessed_land_value"].notna() & values["assessed_building_value"].notna()].copy()
    values["total_assessed_value"] = values["assessed_land_value"] + values["assessed_building_value"]
    values = values[["property_id", "year", "assessed_land_value", "assessed_building_value", "total_assessed_value"]]
    properties = properties.loc[properties["property_class"] == "Residential"].copy()
    sales["date"] = pd.to_datetime(sales["date"], errors="coerce")
    sales["sale_year"] = sales["date"].dt.year
    sales = sales.loc[sales["price"] < 10_000_000, ["property_id", "sale_year", "price"]].rename(columns={"price": "sale_price"})
    sales = sales.sort_values("sale_price", ascending=False).drop_duplicates(["sale_year", "property_id"])
    joined = properties.merge(values, left_on="id", right_on="property_id")
    joined = joined.merge(sales, left_on=["id", "year"], right_on=["property_id", "sale_year"])
    joined = joined.loc[(joined["sale_price"] >= 2000) & (joined["total_assessed_value"] > 0)].copy()
    # make_nola_report.R divides total_assessed_value by 0.1 before? that's on joined_reassessed after ACS join.
    # Report path: readRDS nola_joined.RDS which already divided by 0.1 (nola.R line 115).
    joined["total_assessed_value_report"] = joined["total_assessed_value"] / 0.1
    ratios = cmf_reformat(joined, "sale_price", "total_assessed_value_report", "year", filter_data=True,
                          source="make_nola_report.R + nola.R")
    overall = cmf_cod_prd_prb(ratios)
    write_log("orleans.log", json.dumps({"n_joined": len(joined), "overall": overall,
                                         "years": sorted(joined["year"].dropna().unique().tolist())}, indent=2, default=str))
    return {
        "jurisdiction": "orleans_la",
        "original_reported_n": "not parsed from Orleans Parish.html in this pass",
        "reproduced_n": overall["N"],
        "original_years": "FOIA/github scrape; nola.R does not restrict years explicitly",
        "reproduced_years": f"{int(joined['year'].min())}-{int(joined['year'].max())}" if len(joined) else "",
        "original_key_results": "see Box Orleans Parish.html",
        "reproduced_results": json.dumps(overall),
        "abs_rel_discrepancy": "ACS tract join skipped (tidycensus); nola.R inner-joins tracts and drops non-matches",
        "reproduction_status": "PARTIAL_REPRODUCTION",
        "explanation": (
            "Rebuilt nola.R CSV join (residential, price<1e7, distinct max price per year, sale>=2000). "
            "Applied /0.1 assessed-value scaling from nola.R line 115 used in nola_joined.RDS. "
            "Skipped tidycensus tract join so N may exceed the saved RDS. README: data scraped via "
            "github.com/bhelx/nola-assessor-data after $75k FOIA quote."
        ),
        "source_code": "nola.R; make_nola_report.R; README.txt",
    }


def franklin() -> dict:
    path = RAW / "franklin_oh/box/2jn1707wbpxdd98m1lke6igkqmdvg5t3/columbusonly.csv"
    df = pd.read_csv(path)
    # columnbs-raw-data-code: filter LAND_CLASS 510-530
    df["LAND_CLASS"] = pd.to_numeric(df["LAND_CLASS"], errors="coerce")
    df = df.loc[df["LAND_CLASS"].between(510, 530)].copy()
    ratios = cmf_reformat(df, "PRICE", "ASSESSED_VALUE", "SALE_YEAR", filter_data=True,
                          source="columnbs-raw-data-code")
    overall = cmf_cod_prd_prb(ratios)
    write_log("franklin.log", json.dumps({"n_landclass": len(df), "overall": overall,
                                          "years": sorted(df["SALE_YEAR"].dropna().unique().tolist())}, indent=2, default=str))
    return {
        "jurisdiction": "franklin_oh",
        "original_reported_n": "no local report numbers; page Report link is Columbus County NC CoreLogic HTML",
        "reproduced_n": overall["N"],
        "original_years": "unknown from CMF page",
        "reproduced_years": f"{int(df['SALE_YEAR'].min())}-{int(df['SALE_YEAR'].max())}" if len(df) else "",
        "original_key_results": "unavailable (wrong-geography CoreLogic HTML; franklin-code uses example_data)",
        "reproduced_results": json.dumps(overall),
        "abs_rel_discrepancy": "",
        "reproduction_status": "PARTIAL_REPRODUCTION",
        "explanation": (
            "Box file is columbusonly.csv (city extract), not Franklin County full.csv. "
            "Applied LAND_CLASS 510-530 and default reformat_data IQR filter. "
            "Cannot match a published local COD/PRD/PRB. franklin-raw-data-code overwrites with example_data."
        ),
        "source_code": "https://propertytaxproject.uchicago.edu/columnbs-raw-data-code/",
    }


def st_louis() -> dict:
    # As-written page code uses APRTOT as SALE_PRICE on fullroll (defect).
    full = RAW / "st_louis_county_mo/box/rabph6sd546szpwe6likep763hv8g3v3/new data/fullroll.csv"
    sold = RAW / "st_louis_county_mo/box/rabph6sd546szpwe6likep763hv8g3v3/new data/joined.csv"
    df_full = pd.read_csv(full, low_memory=False, nrows=None)
    as_written = cmf_reformat(df_full, "APRTOT", "ASMTOT", "TAXYR", filter_data=True,
                              source="stlouis page: SALE_PRICE=APRTOT on fullroll")
    written_stats = cmf_cod_prd_prb(as_written)
    df_sold = pd.read_csv(sold, low_memory=False)
    # intended: PRICE vs ASMTOT, SALEVAL==X
    intended = df_sold.loc[df_sold["SALEVAL"].astype(str) == "X"].copy()
    intended_ratios = cmf_reformat(intended, "PRICE", "ASMTOT", "TAXYR", filter_data=True,
                                   source="sold_only construction on page, then ignored by make_report")
    intended_stats = cmf_cod_prd_prb(intended_ratios)
    write_log("st_louis.log", json.dumps({
        "as_written_fullroll_APRTOT": written_stats,
        "intended_sold_SALEVAL_X": intended_stats,
        "fullroll_n": len(df_full),
        "joined_n": len(df_sold),
        "defect": "make_report uses APRTOT as sale price on unsold-inclusive fullroll",
    }, indent=2, default=str))
    return {
        "jurisdiction": "st_louis_county_mo",
        "original_reported_n": "see Box St. Louis County.html (not parsed numerically here)",
        "reproduced_n": written_stats["N"],
        "original_years": "assessments 2009-2019; sales 2005-2020 in page filters",
        "reproduced_years": f"{int(as_written['SALE_YEAR'].min())}-{int(as_written['SALE_YEAR'].max())}" if len(as_written) else "",
        "original_key_results": json.dumps({"as_written": written_stats, "intended_sold": intended_stats}),
        "reproduced_results": json.dumps({"as_written": written_stats, "intended_sold": intended_stats}),
        "abs_rel_discrepancy": "as-written estimand is appraisal/assessment on full roll, not sale ratio",
        "reproduction_status": "PARTIAL_REPRODUCTION",
        "explanation": (
            "Faithfully ran published snippet: SALE_PRICE=APRTOT, ASSESSED_VALUE=ASMTOT on fullroll.csv. "
            "That snippet ignores sold_only/joined.csv (PRICE, SALEVAL=='X'). "
            "Also computed the sold-only construction created two lines earlier. "
            "Do not treat as-written APRTOT ratios as Berry sale-ratio evidence."
        ),
        "source_code": "https://propertytaxproject.uchicago.edu/stlouis-county-raw-data-code/",
    }


def cook() -> dict:
    d = RAW / "cook_il/box/5j9offt7kv763i62duvhvi6hk5rrh1ok"
    files = sorted(d.glob("res_sales*.csv"))
    # R list.files alphabetical: CB_merged, joined, layout, then res_sales2002-2015 = files[4:17]
    frames = [pd.read_csv(f, low_memory=False) for f in files]
    tmp = pd.concat(frames, ignore_index=True)
    good = {202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 234, 278, 295, 299}
    tmp["bor_class"] = pd.to_numeric(tmp["bor_class"], errors="coerce")
    df = tmp.loc[tmp["bor_class"].isin(good)].copy()
    ratios = cmf_reformat(df, "NetConsideration", "bor_CCAO_ass", "joinyr", filter_data=True,
                          source="cook-county-raw-data-code")
    overall = cmf_cod_prd_prb(ratios)
    by_year = {int(y): cmf_cod_prd_prb(g) for y, g in ratios.groupby(ratios["SALE_YEAR"])}
    write_log("cook.log", json.dumps({"n_concat": len(tmp), "n_class": len(df), "overall": overall,
                                      "by_year_n": {str(k): v["N"] for k, v in by_year.items()}}, indent=2))
    return {
        "jurisdiction": "cook_il",
        "original_reported_n": "local CMF numeric report not linked; page Report is CoreLogic nationwide HTML",
        "reproduced_n": overall["N"],
        "original_years": "res_sales2002-2015 (interpreted files[4:17] as those CSVs)",
        "reproduced_years": f"{int(ratios['SALE_YEAR'].min())}-{int(ratios['SALE_YEAR'].max())}" if len(ratios) else "",
        "original_key_results": "unavailable from local PDF; CoreLogic HTML is a different estimand/source",
        "reproduced_results": json.dumps({"overall": overall, "by_year": {str(k): v for k, v in by_year.items()}}),
        "abs_rel_discrepancy": "",
        "reproduction_status": "PARTIAL_REPRODUCTION",
        "explanation": (
            "Concatenated res_sales2002-2015, filtered bor_class in CMF good_classes, "
            "SALE_PRICE=NetConsideration, ASSESSED_VALUE=bor_CCAO_ass, SALE_YEAR=joinyr, "
            "default IQR reformat_data. No local published COD/PRD/PRB to match; "
            "do not compare to CoreLogic Cook County_Illinois.html."
        ),
        "source_code": "https://propertytaxproject.uchicago.edu/cook-county-raw-data-code/",
    }


def main() -> int:
    rows = []
    for fn in (detroit, philadelphia, orleans, franklin, st_louis, cook):
        print("RUN", fn.__name__, flush=True)
        try:
            rows.append(fn())
        except Exception as e:
            rows.append({
                "jurisdiction": fn.__name__,
                "original_reported_n": "",
                "reproduced_n": "",
                "original_years": "",
                "reproduced_years": "",
                "original_key_results": "",
                "reproduced_results": "",
                "abs_rel_discrepancy": "",
                "reproduction_status": "FAILED",
                "explanation": str(e),
                "source_code": "",
            })
            print("FAIL", fn.__name__, e, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "reproduction_results.csv", index=False)
    print("wrote", OUT / "reproduction_results.csv", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
