#!/usr/bin/env python3
"""Profile date/year coverage in Allegheny (and Cook) assessor-history extracts.

Scans every parquet shard with PyArrow aggregations only — no full pandas load.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "temp/allegheny_date_coverage.json"

DATASETS = {
    "allegheny_2016_2025": ROOT / "data/dewey-downloads/allegheny-county-2016-2025-all-features",
    "allegheny_2006_2025": ROOT / "data/dewey-downloads/allegheny-county-2006-2025-all-features",
    "cook_2016_2025": ROOT / "data/dewey-downloads/cookcounty-2016-2025-all-features",
    "cook_2006_2025": ROOT / "data/dewey-downloads/cook-2006-2025-all-features",
}

DATE_COLUMNS = [
    "PUBLICATIONDATE",
    "ASSRLASTUPDATED",
    "LASTASSESSORTAXROLLUPDATE",
    "ASSESSORLASTSALEDATE",
    "ASSESSORPRIORSALEDATE",
    "DEEDLASTSALEDATE",
    "LASTOWNERSHIPTRANSFERDATE",
]
YEAR_COLUMNS = [
    "ASSESSORHISTORYYEAR",
    "TAXYEARASSESSED",
    "TAXFISCALYEAR",
    "TAXMARKETVALUEYEAR",
    "TAXDELINQUENTYEAR",
]
VALUE_COLUMNS = [
    "TAXASSESSEDVALUETOTAL",
    "TAXMARKETVALUETOTAL",
    "AREABUILDING",
    "YEARBUILT",
]


def _files(folder: Path) -> tuple[list[Path], list[dict]]:
    good: list[Path] = []
    bad: list[dict] = []
    for path in sorted(folder.glob("*.snappy.parquet")):
        try:
            pq.read_metadata(path)
            good.append(path)
        except Exception as exc:  # noqa: BLE001
            bad.append({"path": str(path), "error": str(exc)})
    return good, bad


def _year_from_timestamp(col: pa.ChunkedArray) -> pa.ChunkedArray:
    cast = pc.cast(col, pa.timestamp("us"))
    return pc.year(cast)


def _year_from_numeric(col: pa.ChunkedArray) -> pa.ChunkedArray:
    numeric = pc.cast(col, pa.float64())
    return pc.cast(pc.round(numeric), pa.int32())


def _count_by_year(table: pa.Table, year_col: str) -> dict[int, int]:
    years = table[year_col]
    if years.null_count == len(years):
        return {}
    counts = (
        pd.Series(years.to_numpy(zero_copy_only=False))
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
    )
    return {int(k): int(v) for k, v in counts.items()}


def _validity_mask(table: pa.Table) -> pa.Array:
    """Rows with a usable assessor-history year and positive total assessed value."""
    year = pc.cast(table["ASSESSORHISTORYYEAR"], pa.float64())
    value = pc.cast(table["TAXASSESSEDVALUETOTAL"], pa.float64())
    year_ok = pc.and_(pc.is_valid(year), pc.and_(pc.greater_equal(year, 1990), pc.less_equal(year, 2030)))
    value_ok = pc.and_(pc.is_valid(value), pc.greater(value, 0))
    return pc.and_(year_ok, value_ok)


def profile_dataset(name: str, folder: Path) -> dict:
    files, bad_files = _files(folder)
    if not files:
        return {"name": name, "folder": str(folder), "error": "no readable parquet files", "bad_files": bad_files}

    dataset = ds.dataset(files, format="parquet")
    schema_names = set(dataset.schema.names)
    columns = [c for c in DATE_COLUMNS + YEAR_COLUMNS + VALUE_COLUMNS if c in schema_names]
    scanner = dataset.scanner(columns=columns, batch_size=250_000)

    total_rows = 0
    valid_rows = 0
    year_hist: dict[int, int] = {}
    tax_year_hist: dict[int, int] = {}
    date_year_hists: dict[str, dict[int, int]] = {c: {} for c in DATE_COLUMNS if c in schema_names}
    date_min: dict[str, str | None] = {c: None for c in DATE_COLUMNS if c in schema_names}
    date_max: dict[str, str | None] = {c: None for c in DATE_COLUMNS if c in schema_names}
    value_positive = 0
    yearbuilt_present = 0

    for batch in scanner.to_batches():
        table = pa.Table.from_batches([batch])
        n = table.num_rows
        total_rows += n

        mask = _validity_mask(table)
        valid_rows += int(pc.sum(pc.cast(mask, pa.int64())).as_py())

        assessor_year = _year_from_numeric(table["ASSESSORHISTORYYEAR"])
        for y, c in zip(assessor_year.to_numpy(zero_copy_only=False), mask.to_numpy(zero_copy_only=False)):
            if c and y is not None and not np.isnan(y):
                yi = int(y)
                year_hist[yi] = year_hist.get(yi, 0) + 1

        if "TAXYEARASSESSED" in schema_names:
            tax_year = _year_from_numeric(table["TAXYEARASSESSED"])
            for y in tax_year.to_numpy(zero_copy_only=False):
                if y is not None and not np.isnan(y):
                    yi = int(y)
                    tax_year_hist[yi] = tax_year_hist.get(yi, 0) + 1

        for col in date_year_hists:
            arr = table[col]
            non_null = pc.drop_null(arr)
            if len(non_null):
                ts = pc.cast(non_null, pa.timestamp("us"))
                mn = pc.min(ts).as_py()
                mx = pc.max(ts).as_py()
                date_min[col] = str(min(filter(None, [date_min[col], str(mn)])))
                date_max[col] = str(max(filter(None, [date_max[col], str(mx)])))
                yrs = _year_from_timestamp(non_null)
                for y in yrs.to_numpy(zero_copy_only=False):
                    if y is not None:
                        yi = int(y)
                        date_year_hists[col][yi] = date_year_hists[col].get(yi, 0) + 1

        if "TAXASSESSEDVALUETOTAL" in schema_names:
            value = pc.cast(table["TAXASSESSEDVALUETOTAL"], pa.float64())
            value_positive += int(pc.sum(pc.cast(pc.and_(pc.is_valid(value), pc.greater(value, 0)), pa.int64())).as_py())
        if "YEARBUILT" in schema_names:
            yb = pc.cast(table["YEARBUILT"], pa.float64())
            yearbuilt_present += int(pc.sum(pc.cast(pc.and_(pc.is_valid(yb), pc.greater(yb, 0)), pa.int64())).as_py())

    years_sorted = sorted(year_hist)
    valid_years = [y for y, c in year_hist.items() if c > 0]
    first_year = min(valid_years) if valid_years else None
    last_year = max(valid_years) if valid_years else None

    return {
        "name": name,
        "folder": str(folder),
        "n_files": len(files),
        "bad_files": bad_files,
        "total_rows": total_rows,
        "valid_rows_assessor_year_and_value": valid_rows,
        "valid_row_share": valid_rows / total_rows if total_rows else 0.0,
        "assessor_history_year_hist": year_hist,
        "assessor_history_year_min": first_year,
        "assessor_history_year_max": last_year,
        "tax_year_assessed_hist": tax_year_hist,
        "date_min": date_min,
        "date_max": date_max,
        "date_year_hists": date_year_hists,
        "positive_assessed_value_rows": value_positive,
        "yearbuilt_present_rows": yearbuilt_present,
        "years_with_any_rows": years_sorted,
        "years_with_valid_rows": sorted(y for y in year_hist if year_hist[y] > 0),
    }


def compare_pair(newer: dict, older: dict) -> dict:
    newer_years = set(newer.get("assessor_history_year_hist", {}))
    older_years = set(older.get("assessor_history_year_hist", {}))
    older_only = sorted(older_years - newer_years)
    newer_only = sorted(newer_years - older_years)
    shared = sorted(newer_years & older_years)
    deltas = {}
    for y in shared:
        n = newer["assessor_history_year_hist"].get(y, 0)
        o = older["assessor_history_year_hist"].get(y, 0)
        deltas[y] = {"newer": n, "older": o, "delta_older_minus_newer": o - n}
    return {
        "older_only_years": older_only,
        "newer_only_years": newer_only,
        "shared_years": shared,
        "shared_year_counts": deltas,
    }


def summarize(results: dict) -> None:
    print("=== Assessor history year coverage (valid = year + positive assessed value) ===")
    for key in ["allegheny_2016_2025", "allegheny_2006_2025", "cook_2016_2025", "cook_2006_2025"]:
        r = results.get(key)
        if not r or "error" in r:
            print(key, "MISSING", r)
            continue
        hist = r["assessor_history_year_hist"]
        print(
            f"{key}: rows={r['total_rows']:,} valid={r['valid_rows_assessor_year_and_value']:,} "
            f"({100*r['valid_row_share']:.1f}%) years={r['assessor_history_year_min']}-{r['assessor_history_year_max']}"
        )
        for y in sorted(hist):
            print(f"  {y}: {hist[y]:,}")

    print("\n=== Allegheny 2016 vs 2006 comparison ===")
    cmp_a = results.get("allegheny_compare")
    if cmp_a:
        print("older-only years:", cmp_a["older_only_years"])
        print("newer-only years:", cmp_a["newer_only_years"])

    print("\n=== Cook 2016 vs 2006 comparison ===")
    cmp_c = results.get("cook_compare")
    if cmp_c:
        print("older-only years:", cmp_c["older_only_years"])
        print("newer-only years:", cmp_c["newer_only_years"])

    print("\n=== Date column ranges (Allegheny 2016-2025) ===")
    a16 = results.get("allegheny_2016_2025") or {}
    if "date_min" not in a16:
        print("(date ranges unavailable)")
        return
    for col in DATE_COLUMNS:
        if col in a16["date_min"]:
            print(f"  {col}: {a16['date_min'][col]} .. {a16['date_max'][col]}")


def main() -> None:
    results = {}
    for key, folder in DATASETS.items():
        if not folder.exists():
            results[key] = {"name": key, "folder": str(folder), "error": "missing directory"}
            continue
        files, bad = _files(folder)
        print(f"[scan] {key} ({len(files)} readable / {len(bad)} bad files)...", flush=True)
        results[key] = profile_dataset(key, folder)

    if "allegheny_2016_2025" in results and "allegheny_2006_2025" in results:
        results["allegheny_compare"] = compare_pair(results["allegheny_2016_2025"], results["allegheny_2006_2025"])
    if "cook_2016_2025" in results and "cook_2006_2025" in results:
        results["cook_compare"] = compare_pair(results["cook_2016_2025"], results["cook_2006_2025"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    summarize(results)
    print(f"\n[wrote] {OUT}")


if __name__ == "__main__":
    main()
