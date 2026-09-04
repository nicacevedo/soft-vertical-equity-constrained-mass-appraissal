#!/usr/bin/env python3
"""Step: per-jurisdiction Recorder/History caches via predicate pushdown.

Generalizes analysis/berry_attom_validation_v3/scripts/build_county_caches.py
to all 9 jurisdictions (v3 covered only 3). Two differences from the v3
builder, both deliberate:

1. Uses readable_parquet_files() to skip corrupt/unreadable shards rather than
   letting ds.dataset() abort the whole scan -- attom_county_benchmark.sh
   documents a real corrupt-shard incident in one of these same folders.
2. Reads directly from each jurisdiction's assessor_dir/recorder_dir pair in
   v1_common.JURISDICTIONS, so no folder is hard-coded per product.

Tax Assessor and ACS are never touched -- only fips_scan_filter,
readable_parquet_files and column lists are imported from
scripts/other_counties_benchmars.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.other_counties_benchmars import fips_scan_filter, readable_parquet_files  # noqa: E402
from analysis.external_jurisdiction_benchmark_v1.scripts.v1_common import (  # noqa: E402
    ANALYSIS, ALL_KEYS, HISTORY_CACHE_COLUMNS, JURISDICTION_BY_KEY, OUTPUT, RECORDER_CACHE_COLUMNS,
    ST_LOUIS_CITY_FIPS, sha256_file, write_json,
)


def present_columns(schema_names: list[str], wanted: list[str]) -> list[str]:
    have = set(schema_names)
    return [c for c in wanted if c in have]


def write_slice(product: str, directory: Path, fips: str, key: str, columns: list[str]) -> dict:
    all_files = sorted(p for p in directory.iterdir() if p.suffix == ".parquet")
    files, unreadable = readable_parquet_files(all_files)
    source = ds.dataset(files, format="parquet")
    cols = present_columns(source.schema.names, columns)
    fips_col = "DOCUMENTRECORDINGCOUNTYFIPS" if product == "recorder" else "SITUSSTATECOUNTYFIPS"
    table = source.to_table(columns=cols, filter=fips_scan_filter(source, fips_col, fips))
    if fips_col in table.column_names:
        fips_vals = {str(v) for v in table[fips_col].to_pylist() if v is not None}
        if fips == "29189" and ST_LOUIS_CITY_FIPS in fips_vals:
            raise RuntimeError("St. Louis County cache contains St. Louis City FIPS 29510")
    out_dir = OUTPUT / "cache" / key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{product}.parquet"
    pq.write_table(table, out_path, compression="zstd")
    rec = {
        "county_key": key,
        "fips": fips,
        "product": product,
        "n_rows": table.num_rows,
        "n_columns": table.num_columns,
        "path": str(out_path),
        "sha256": sha256_file(out_path),
        "source_dir": str(directory),
        "source_n_shards": len(all_files),
        "n_unreadable_shards": len(unreadable),
        "unreadable_shards": unreadable,
        "written_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    print(json.dumps(
        {k: rec[k] for k in ["county_key", "product", "n_rows", "n_unreadable_shards", "sha256"]},
        sort_keys=True,
    ), flush=True)
    return rec


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--county-key", default="all")
    args = parser.parse_args()
    keys = list(ALL_KEYS) if args.county_key == "all" else [args.county_key]
    for key in keys:
        if key not in JURISDICTION_BY_KEY:
            raise SystemExit(f"unknown county-key {key}")
    for key in keys:
        j = JURISDICTION_BY_KEY[key]
        rec_r = write_slice("recorder", j["recorder_dir"], j["fips"], key, RECORDER_CACHE_COLUMNS)
        rec_h = write_slice("history", j["assessor_dir"], j["fips"], key, HISTORY_CACHE_COLUMNS)
        write_json(OUTPUT / "cache" / key / "cache_manifest.json", [rec_r, rec_h])

    all_meta = []
    for key in ALL_KEYS:
        part = OUTPUT / "cache" / key / "cache_manifest.json"
        if part.exists():
            all_meta.extend(json.loads(part.read_text()))
    if all_meta:
        write_json(OUTPUT / "cache" / "cache_manifest.json", all_meta)
        ANALYSIS.joinpath("audits").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_meta).to_csv(ANALYSIS / "audits" / "county_cache_manifest.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
