#!/usr/bin/env python3
"""Compare date/year coverage between the two Cook assessor-history extracts."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/dewey-downloads"
DIRS = {
    "cook-2006-2025-all-features": DATA / "cook-2006-2025-all-features",
    "cookcounty-2016-2025-all-features": DATA / "cookcounty-2016-2025-all-features",
}

# Columns the benchmark actually uses to decide a history row is usable.
KEY_YEAR = "ASSESSORHISTORYYEAR"
ALT_YEAR = "TAXYEARASSESSED"
DATE_COLS = [
    "ASSESSORLASTSALEDATE",
    "ASSESSORPRIORSALEDATE",
    "PUBLICATIONDATE",
    "ASSRLASTUPDATED",
    "LASTASSESSORTAXROLLUPDATE",
    "DEEDLASTSALEDATE",
    "LASTOWNERSHIPTRANSFERDATE",
]
FEATURE_PROBE = [
    "AREABUILDING",
    "BEDROOMSCOUNT",
    "BATHCOUNT",
    "TAXASSESSEDVALUETOTAL",
    "TAXMARKETVALUETOTAL",
    "PROPERTYUSESTANDARDIZED",
    "YEARBUILT",
    "LATITUDE",
    "LONGITUDE",
]
COOK_FIPS = "17031"
BATCH = 250_000


def year_stats(dataset: ds.Dataset) -> dict:
    """Aggregate row counts and validity by assessment year."""
    available = set(dataset.schema.names)
    cols = [c for c in [KEY_YEAR, ALT_YEAR, "SITUSSTATECOUNTYFIPS", *FEATURE_PROBE] if c in available]
    scanner = dataset.scanner(columns=cols, batch_size=BATCH)

    year_rows: dict[int, int] = {}
    alt_year_rows: dict[int, int] = {}
    valid_rows: dict[int, int] = {}
    feature_nonnull: dict[int, dict[str, int]] = {}
    fips_ok = 0
    total = 0

    for batch in scanner.to_batches():
        frame = batch.to_pandas()
        total += len(frame)
        if "SITUSSTATECOUNTYFIPS" in frame.columns:
            fips = frame["SITUSSTATECOUNTYFIPS"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
            frame = frame.loc[fips.eq(COOK_FIPS)]
        if frame.empty:
            continue
        fips_ok += len(frame)

        ay = pd.to_numeric(frame[KEY_YEAR], errors="coerce")
        for year, count in ay.dropna().astype(int).value_counts().items():
            year_rows[int(year)] = year_rows.get(int(year), 0) + int(count)

        if ALT_YEAR in frame.columns:
            ty = pd.to_numeric(frame[ALT_YEAR], errors="coerce")
            for year, count in ty.dropna().astype(int).value_counts().items():
                alt_year_rows[int(year)] = alt_year_rows.get(int(year), 0) + int(count)

        probe_cols = [c for c in FEATURE_PROBE if c in frame.columns]
        for year in ay.dropna().astype(int).unique():
            y = int(year)
            mask = ay.eq(y)
            n = int(mask.sum())
            if n == 0:
                continue
            sub = frame.loc[mask, probe_cols]
            # "Valid for modeling": at least one core numeric feature present.
            usable = sub[["AREABUILDING", "TAXASSESSEDVALUETOTAL", "YEARBUILT"]].notna().any(axis=1)
            valid_rows[y] = valid_rows.get(y, 0) + int(usable.sum())
            if y not in feature_nonnull:
                feature_nonnull[y] = {c: 0 for c in probe_cols}
            for col in probe_cols:
                feature_nonnull[y][col] += int(sub[col].notna().sum())

    return {
        "total_rows_scanned": total,
        "cook_fips_rows": fips_ok,
        "year_rows": year_rows,
        "alt_year_rows": alt_year_rows,
        "valid_rows": valid_rows,
        "feature_nonnull": feature_nonnull,
    }


def date_col_stats(dataset: ds.Dataset) -> pd.DataFrame:
    available = [c for c in DATE_COLS if c in dataset.schema.names]
    if not available:
        return pd.DataFrame()
    scanner = dataset.scanner(columns=["SITUSSTATECOUNTYFIPS", *available], batch_size=BATCH)
    mins, maxs, nonnull = {c: None for c in available}, {c: None for c in available}, {c: 0 for c in available}
    rows = 0
    for batch in scanner.to_batches():
        frame = batch.to_pandas()
        fips = frame["SITUSSTATECOUNTYFIPS"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
        frame = frame.loc[fips.eq(COOK_FIPS)]
        rows += len(frame)
        for col in available:
            parsed = pd.to_datetime(frame[col], errors="coerce")
            nn = int(parsed.notna().sum())
            nonnull[col] += nn
            if nn:
                cmin, cmax = parsed.min(), parsed.max()
                mins[col] = cmin if mins[col] is None or cmin < mins[col] else mins[col]
                maxs[col] = cmax if maxs[col] is None or cmax > maxs[col] else maxs[col]
    out = []
    for col in available:
        out.append({
            "column": col,
            "nonnull_rows": nonnull[col],
            "pct_of_cook_rows": 100.0 * nonnull[col] / rows if rows else np.nan,
            "min_date": mins[col],
            "max_date": maxs[col],
        })
    return pd.DataFrame(out)


def year_table(stats: dict) -> pd.DataFrame:
    years = sorted(set(stats["year_rows"]) | set(stats["valid_rows"]))
    rows = []
    for y in years:
        n = stats["year_rows"].get(y, 0)
        v = stats["valid_rows"].get(y, 0)
        alt = stats["alt_year_rows"].get(y, 0)
        probe = stats["feature_nonnull"].get(y, {})
        rows.append({
            "year": y,
            "rows": n,
            "valid_feature_rows": v,
            "valid_pct": 100.0 * v / n if n else np.nan,
            "taxyear_rows": alt,
            "livingarea_nonnull": probe.get("AREABUILDING", 0),
            "assessed_total_nonnull": probe.get("TAXASSESSEDVALUETOTAL", 0),
            "yearbuilt_nonnull": probe.get("YEARBUILT", 0),
        })
    return pd.DataFrame(rows)


def main() -> None:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)
    for name, folder in DIRS.items():
        files = sorted(folder.glob("*.snappy.parquet"))
        print("\n" + "=" * 90)
        print(name, "| files:", len(files))
        dataset = ds.dataset([str(f) for f in files], format="parquet")
        stats = year_stats(dataset)
        table = year_table(stats)
        print("total scanned:", stats["total_rows_scanned"], "| cook fips rows:", stats["cook_fips_rows"])
        print(table.to_string(index=False))
        if not table.empty:
            first_valid = table.loc[table["valid_feature_rows"].gt(0), "year"].min()
            last_valid = table.loc[table["valid_feature_rows"].gt(0), "year"].max()
            print(f"valid feature years: {first_valid} .. {last_valid}")
        dates = date_col_stats(dataset)
        if not dates.empty:
            print("\nDate columns (Cook FIPS only):")
            print(dates.to_string(index=False))

    # Side-by-side diff on overlapping years
    print("\n" + "=" * 90)
    print("YEAR-BY-YEAR COMPARISON (Cook FIPS rows by ASSESSORHISTORYYEAR)")
    tables = {}
    for name, folder in DIRS.items():
        dataset = ds.dataset([str(f) for f in sorted(folder.glob("*.snappy.parquet"))], format="parquet")
        tables[name] = year_table(year_stats(dataset)).set_index("year")
    years = sorted(set(tables["cook-2006-2025-all-features"].index) | set(tables["cookcounty-2016-2025-all-features"].index))
    comp = []
    for y in years:
        a = tables["cook-2006-2025-all-features"].loc[y] if y in tables["cook-2006-2025-all-features"].index else None
        b = tables["cookcounty-2016-2025-all-features"].loc[y] if y in tables["cookcounty-2016-2025-all-features"].index else None
        comp.append({
            "year": y,
            "cook_rows": None if a is None else int(a["rows"]),
            "cook_valid": None if a is None else int(a["valid_feature_rows"]),
            "cookcounty_rows": None if b is None else int(b["rows"]),
            "cookcounty_valid": None if b is None else int(b["valid_feature_rows"]),
        })
    print(pd.DataFrame(comp).to_string(index=False))


if __name__ == "__main__":
    main()
