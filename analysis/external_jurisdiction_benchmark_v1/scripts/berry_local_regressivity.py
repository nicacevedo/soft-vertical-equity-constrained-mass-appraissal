#!/usr/bin/env python3
"""Step 4: Berry/local external regressivity, read-only reuse of the audited
v3 constructions plus a corrected St. Louis reconstruction.

Detroit: full sample = analysis/berry_attom_validation_v3/berry_reproduction/
detroit_mi_transactions.parquet (already SALE_PRICE/ASSESSED_VALUE/RATIO under
Property Class==401 & Terms of Sale=="VALID ARMS LENGTH", filter_data=False,
per the Rmd). ATTOM-linkable subset = linkage/wayne_safe_history.parquet.

Philadelphia: full sample = berry_reproduction/philadelphia_pa_transactions.parquet
(arms-length file only; "total" file confirmed a disjoint universe with 0
transaction-key overlap and is never stacked). ATTOM-linkable subset =
linkage/philadelphia_safe_history.parquet. cmf_reformat applied here with
filter_data=True (canonical IQR spec), NOT reusing reproduce_berry.py's
naive-rbind branch.

St. Louis County: corrects the v3 wording. v3 said "no official
assessed-value series"; true only of the 2019 cumulative sales.csv it chose.
The acquired joined.csv carries PRICE + ASMTOT + SALEVAL for 2009-2019.
Full sample = joined.csv filtered SALEVAL=='X' (documented "VALID SALE" in
STLCO_REAL_DATA_DICTIONARY.txt), via the already-implemented intended_sold
specification (reproduce_berry.py:267-306) -- not invented here.
ATTOM-linkable subset = joined.csv rows whose (PARID, TAXYR) also appears in
linkage/st_louis_county_safe_history.parquet (by (PARID, year(SALEDT))),
since that linkage file carries no ASMTOT of its own.

beta_log (OLS slope of log(ratio) on log(sale_price)) mirrors
analysis/berry_attom_validation_v3/scripts/link_berry_attom.py::ratio_metrics.
Never mixes official assessment ratios with AVM valuation ratios.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from analysis.berry_cmf_validation.scripts.reproduce_berry import (  # noqa: E402
    cmf_cod_prd_prb, cmf_reformat,
)
from analysis.external_jurisdiction_benchmark_v1.scripts.v1_common import ANALYSIS, write_json  # noqa: E402

V3 = ROOT / "analysis" / "berry_attom_validation_v3"
STL_RAW = ROOT / "data/berry_cmf/raw/st_louis_county_mo/box/rabph6sd546szpwe6likep763hv8g3v3/new data"
STL_COVERAGE_START, STL_COVERAGE_END = 2009, 2019


def beta_log(ratios: pd.DataFrame) -> float:
    r = pd.to_numeric(ratios["RATIO"], errors="coerce")
    sp = pd.to_numeric(ratios["SALE_PRICE"], errors="coerce")
    mask = r.notna() & sp.notna() & (r > 0) & (sp > 0)
    if mask.sum() < 3:
        return float("nan")
    x = np.log(sp[mask].to_numpy())
    y = np.log(r[mask].to_numpy())
    return float(np.polyfit(x, y, 1)[0])


def decile_profile(ratios: pd.DataFrame) -> pd.DataFrame:
    frame = ratios.copy()
    frame["SALE_PRICE"] = pd.to_numeric(frame["SALE_PRICE"], errors="coerce")
    frame["RATIO"] = pd.to_numeric(frame["RATIO"], errors="coerce")
    frame = frame.loc[frame["SALE_PRICE"].gt(0) & frame["RATIO"].notna()].copy()
    frame["price_decile"] = pd.qcut(frame["SALE_PRICE"], 10, labels=False, duplicates="drop")
    out = frame.groupby("price_decile").agg(
        n=("RATIO", "size"), median_ratio=("RATIO", "median"),
        median_price=("SALE_PRICE", "median"),
    ).reset_index()
    return out


def summarize(jurisdiction: str, sample_name: str, ratios: pd.DataFrame, note: str = "") -> dict:
    stats = cmf_cod_prd_prb(ratios)
    return {
        "jurisdiction": jurisdiction, "sample": sample_name,
        "N": stats["N"], "COD": stats["COD"], "PRD": stats["PRD"], "PRB": stats["PRB"],
        "beta_log": beta_log(ratios), "note": note,
    }


def detroit() -> tuple[list[dict], list[pd.DataFrame]]:
    full = pd.read_parquet(V3 / "berry_reproduction" / "detroit_mi_transactions.parquet")
    linked = pd.read_parquet(V3 / "linkage" / "wayne_safe_history.parquet")
    rows = [
        summarize("detroit_mi", "full_local_sample", full),
        summarize("detroit_mi", "attom_linkable_subset", linked),
    ]
    profiles = [
        decile_profile(full).assign(jurisdiction="detroit_mi", sample="full_local_sample"),
        decile_profile(linked).assign(jurisdiction="detroit_mi", sample="attom_linkable_subset"),
    ]
    return rows, profiles


def philadelphia() -> tuple[list[dict], list[pd.DataFrame]]:
    full_full = pd.read_parquet(V3 / "berry_reproduction" / "philadelphia_pa_transactions.parquet")
    full = cmf_reformat(full_full, "sale_price", "assmt_at_sale", "sale_year", filter_data=True,
                         source="external_jurisdiction_benchmark_v1: canonical spec, arms-length file only")
    linked_raw = pd.read_parquet(V3 / "linkage" / "philadelphia_safe_history.parquet")
    linked = cmf_reformat(linked_raw, "sale_price", "assmt_at_sale", "sale_year", filter_data=True,
                           source="external_jurisdiction_benchmark_v1: canonical spec, ATTOM-linkable subset")
    rows = [
        summarize("philadelphia_pa", "full_local_sample", full,
                   note="arms_length file only; total file confirmed disjoint (0 overlap), never stacked"),
        summarize("philadelphia_pa", "attom_linkable_subset", linked),
    ]
    profiles = [
        decile_profile(full).assign(jurisdiction="philadelphia_pa", sample="full_local_sample"),
        decile_profile(linked).assign(jurisdiction="philadelphia_pa", sample="attom_linkable_subset"),
    ]
    return rows, profiles


def st_louis() -> tuple[list[dict], list[pd.DataFrame]]:
    joined_path = STL_RAW / "joined.csv"
    if not joined_path.exists():
        return [{
            "jurisdiction": "st_louis_county_mo", "sample": "full_local_sample",
            "N": 0, "COD": np.nan, "PRD": np.nan, "PRB": np.nan, "beta_log": np.nan,
            "note": f"BLOCKER: {joined_path} not found",
        }], []
    joined = pd.read_csv(joined_path, low_memory=False)
    joined["TAXYR"] = pd.to_numeric(joined["TAXYR"], errors="coerce")
    intended = joined.loc[joined["SALEVAL"].astype(str) == "X"].copy()
    full = cmf_reformat(intended, "PRICE", "ASMTOT", "TAXYR", filter_data=True,
                         source="reproduce_berry.py:267-306 intended_sold specification, reused not invented")

    safe_history = pd.read_parquet(V3 / "linkage" / "st_louis_county_safe_history.parquet")
    safe_history["TAXYR"] = pd.to_datetime(
        safe_history["SALEDT_parsed"], errors="coerce"
    ).dt.year
    linked_keys = set(
        zip(safe_history["PARID"].astype(str), safe_history["TAXYR"])
    )
    intended["_key"] = list(zip(intended["PARID"].astype(str), intended["TAXYR"]))
    linked_intended = intended.loc[intended["_key"].isin(linked_keys)].copy()
    linked = cmf_reformat(linked_intended, "PRICE", "ASMTOT", "TAXYR", filter_data=True,
                           source="external_jurisdiction_benchmark_v1: joined.csv restricted to "
                                  "(PARID, TAXYR) pairs also present in the safe-history ATTOM linkage")

    coverage_note = (
        f"coverage {STL_COVERAGE_START}-{STL_COVERAGE_END} only (joined.csv TAXYR range); "
        "ATTOM linkage cohort is predeclared 2005-2019 in v3, so 2005-2008 sales have no "
        "official-ratio counterpart here. as_written APRTOT-as-sale-price variant is a "
        "documented defect and is NOT used."
    )
    rows = [
        summarize("st_louis_county_mo", "full_local_sample", full, note=coverage_note),
        summarize("st_louis_county_mo", "attom_linkable_subset", linked, note=coverage_note),
    ]
    profiles = [
        decile_profile(full).assign(jurisdiction="st_louis_county_mo", sample="full_local_sample"),
        decile_profile(linked).assign(jurisdiction="st_louis_county_mo", sample="attom_linkable_subset"),
    ]
    return rows, profiles


def make_figure(profiles: list[pd.DataFrame]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_prof = pd.concat(profiles, ignore_index=True)
    jurisdictions = sorted(all_prof["jurisdiction"].unique())
    fig, axes = plt.subplots(1, len(jurisdictions), figsize=(5.2 * len(jurisdictions), 4.2), sharey=True)
    if len(jurisdictions) == 1:
        axes = [axes]
    for ax, j in zip(axes, jurisdictions):
        sub = all_prof.loc[all_prof["jurisdiction"] == j]
        for sample, g in sub.groupby("sample"):
            ax.plot(g["price_decile"], g["median_ratio"], marker="o", label=sample)
        ax.axhline(1.0, color="gray", lw=0.8)
        ax.set_xlabel("value/sale-price decile")
        ax.set_title(j)
    axes[0].set_ylabel("median OFFICIAL assessment/sale ratio")
    axes[0].legend(fontsize=8)
    fig.suptitle("Berry/local official ratio: full sample vs ATTOM-linkable subset")
    fig.tight_layout()
    out = ANALYSIS / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "berry_ratio_profiles.pdf")
    plt.close(fig)


def main() -> int:
    ANALYSIS.joinpath("berry").mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    all_profiles: list[pd.DataFrame] = []
    for fn in (detroit, philadelphia, st_louis):
        rows, profiles = fn()
        all_rows.extend(rows)
        all_profiles.extend(profiles)

    metrics_df = pd.DataFrame(all_rows)
    metrics_df.to_csv(ANALYSIS / "berry" / "berry_external_metrics.csv", index=False)

    preservation_rows = []
    for j in metrics_df["jurisdiction"].unique():
        sub = metrics_df.loc[metrics_df["jurisdiction"] == j].set_index("sample")
        if "full_local_sample" in sub.index and "attom_linkable_subset" in sub.index:
            preservation_rows.append({
                "jurisdiction": j,
                "N_full": sub.loc["full_local_sample", "N"],
                "N_linked": sub.loc["attom_linkable_subset", "N"],
                "linkage_rate": (
                    sub.loc["attom_linkable_subset", "N"] / sub.loc["full_local_sample", "N"]
                    if sub.loc["full_local_sample", "N"] else float("nan")
                ),
                "PRB_full": sub.loc["full_local_sample", "PRB"],
                "PRB_linked": sub.loc["attom_linkable_subset", "PRB"],
                "PRD_full": sub.loc["full_local_sample", "PRD"],
                "PRD_linked": sub.loc["attom_linkable_subset", "PRD"],
                "sign_flip_PRB": bool(
                    np.sign(sub.loc["full_local_sample", "PRB"]) != np.sign(sub.loc["attom_linkable_subset", "PRB"])
                ) if pd.notna(sub.loc["full_local_sample", "PRB"]) and pd.notna(sub.loc["attom_linkable_subset", "PRB"]) else None,
            })
    pd.DataFrame(preservation_rows).to_csv(ANALYSIS / "berry" / "berry_linkage_preservation.csv", index=False)

    for p in all_profiles:
        pass
    profile_df = pd.concat(all_profiles, ignore_index=True) if all_profiles else pd.DataFrame()
    profile_df.to_csv(ANALYSIS / "berry" / "berry_ratio_decile_profiles.csv", index=False)
    if all_profiles:
        make_figure(all_profiles)

    write_json(ANALYSIS / "berry" / "berry_regressivity_run_meta.json", {
        "written_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "jurisdictions": sorted(metrics_df["jurisdiction"].unique().tolist()),
        "st_louis_correction": (
            "v3's 'no official assessed-value series' claim held only for the file it chose. "
            "joined.csv (PRICE, ASMTOT, SALEVAL) supports the already-implemented intended_sold "
            "specification. Coverage 2009-2019 only."
        ),
    })
    print(metrics_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
