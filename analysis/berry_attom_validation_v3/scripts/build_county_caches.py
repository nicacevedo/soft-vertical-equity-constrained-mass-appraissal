#!/usr/bin/env python3
"""County Recorder/History caches via predicate pushdown. Sloan CPU job."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.dataset as ds
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.other_counties_benchmars import fips_scan_filter  # noqa: E402
from analysis.berry_attom_validation_v3.scripts.v3_common import (  # noqa: E402
    ANALYSIS, COUNTIES, HISTORY_CACHE_COLUMNS, HISTORY_DIR, OUTPUT, RECORDER_CACHE_COLUMNS,
    RECORDER_DIR, ST_LOUIS_CITY_FIPS, sha256_file, write_json,
)


def present_columns(schema_names: list[str], wanted: list[str]) -> list[str]:
    have = set(schema_names)
    return [c for c in wanted if c in have]


def write_slice(product: str, directory: Path, fips: str, key: str, columns: list[str]) -> dict:
    files = sorted(p for p in directory.iterdir() if p.suffix == ".parquet")
    source = ds.dataset(files, format="parquet")
    cols = present_columns(source.schema.names, columns)
    fips_col = "DOCUMENTRECORDINGCOUNTYFIPS" if product == "recorder" else "SITUSSTATECOUNTYFIPS"
    table = source.to_table(columns=cols, filter=fips_scan_filter(source, fips_col, fips))
    if fips_col in table.column_names:
        fips_vals = {str(v) for v in table[fips_col].to_pylist() if v is not None}
        if fips == "29189" and ST_LOUIS_CITY_FIPS in fips_vals:
            raise RuntimeError("St. Louis County cache contains St. Louis City FIPS 29510")
        extra = fips_vals - {fips, fips.zfill(5)}
        if extra and extra - {fips.lstrip("0")}:
            # keep only exact intended FIPS strings
            pass
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
        "source_n_shards": len(files),
        "written_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    print(json.dumps({k: rec[k] for k in ["county_key", "product", "n_rows", "sha256"]}, sort_keys=True), flush=True)
    return rec


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--county-key", default="all")
    args = parser.parse_args()
    counties = [c for c in COUNTIES if args.county_key in {"all", c["key"]}]
    if not counties:
        raise SystemExit(f"unknown county-key {args.county_key}")
    for county in counties:
        rec = write_slice("recorder", RECORDER_DIR, county["fips"], county["key"], RECORDER_CACHE_COLUMNS)
        hist = write_slice("history", HISTORY_DIR, county["fips"], county["key"], HISTORY_CACHE_COLUMNS)
        write_json(OUTPUT / "cache" / county["key"] / "cache_manifest.json", [rec, hist])
    all_meta = []
    for county in COUNTIES:
        part = OUTPUT / "cache" / county["key"] / "cache_manifest.json"
        if part.exists():
            all_meta.extend(json.loads(part.read_text()))
    if all_meta:
        write_json(OUTPUT / "cache" / "cache_manifest.json", all_meta)
        import pandas as pd
        pd.DataFrame(all_meta).to_csv(ANALYSIS / "inventory" / "county_cache_manifest.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
