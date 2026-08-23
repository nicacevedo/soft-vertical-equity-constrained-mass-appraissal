#!/usr/bin/env python3
"""One-off profiling of the four ATTOM county extracts (not part of the pipeline)."""
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.other_counties_benchmars import (  # noqa: E402
    HISTORY_COLUMNS, fips_scan_filter, files_or_sample, readable_parquet_files,
)

COUNTIES = {
    "17031": "data/dewey-downloads/cookcounty-2016-2025-all-features",
    "42003": "data/dewey-downloads/allegheny-county-2016-2025-all-features",
    "04013": "data/dewey-downloads/maricopa-county-2016-2025-all-features",
    "53033": "data/dewey-downloads/king-county-2015-2025-all-features",
}


def profile(fips: str, rel: str) -> None:
    directory = ROOT / rel
    files = files_or_sample(directory, "assessor-history_*.parquet", 0)
    files, unreadable = readable_parquet_files(files)
    source = ds.dataset(files, format="parquet")
    missing = [c for c in HISTORY_COLUMNS if c not in source.schema.names]

    counts = {}
    years = set()
    total = 0
    nonnull = {c: 0 for c in ["AREABUILDING", "YEARBUILT", "BEDROOMSCOUNT", "LATITUDE", "CENSUSTRACT"]}
    scanner = source.scanner(
        columns=["PROPERTYUSESTANDARDIZED", "ASSESSORHISTORYYEAR", *nonnull],
        filter=fips_scan_filter(source, "SITUSSTATECOUNTYFIPS", fips),
        batch_size=200_000,
    )
    for batch in scanner.to_batches():
        total += batch.num_rows
        for code, n in zip(*[x.to_pylist() for x in batch.column("PROPERTYUSESTANDARDIZED").value_counts().flatten()]):
            key = str(code).strip().upper().removesuffix(".0") if code is not None else "<NULL>"
            counts[key] = counts.get(key, 0) + n
        years.update(y for y in batch.column("ASSESSORHISTORYYEAR").to_pylist() if y is not None)
        for column in nonnull:
            nonnull[column] += batch.num_rows - batch.column(column).null_count

    top = sorted(counts.items(), key=lambda kv: -kv[1])[:10]
    print("=" * 78)
    print(f"fips={fips} dir={rel}")
    print(f"  shards={len(files)} unreadable={unreadable} county_rows={total:,}")
    print(f"  missing_history_columns ({len(missing)}): {missing}")
    print(f"  assessor_years: {sorted(str(y) for y in years)}")
    print(f"  top_property_use: {[(c, n, round(n / max(total, 1), 4)) for c, n in top]}")
    print(f"  nonnull_rates: { {c: round(v / max(total, 1), 4) for c, v in nonnull.items()} }")


if __name__ == "__main__":
    for fips, rel in COUNTIES.items():
        profile(fips, rel)
