#!/usr/bin/env python3
"""Step 2: inventory new Dewey Recorder and Assessor History shards.

Uses Parquet footer metadata first, then projected PyArrow scans for FIPS/year.
Does not trust folder names as evidence of contents.
"""
from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.berry_attom_validation_v2.scripts.v2_common import (  # noqa: E402
    ANALYSIS, HISTORY_DIR, RECORDER_DIR, sha256_file,
)

OUT = ANALYSIS / "inventory"
NOW = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def list_parquet(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.suffix == ".parquet" and p.is_file())


def inventory_files(product: str, directory: Path) -> list[dict]:
    rows = []
    for path in list_parquet(directory):
        md = pq.ParquetFile(path).metadata
        schema = pq.read_schema(path)
        rows.append({
            "product": product,
            "filename": path.name,
            "relpath": str(path.relative_to(directory.parent.parent.parent) if False else path),
            "byte_size": path.stat().st_size,
            "sha256": sha256_file(path),
            "num_rows_footer": md.num_rows,
            "num_row_groups": md.num_row_groups,
            "num_columns": len(schema.names),
            "schema_names": "|".join(schema.names),
            "created_at_utc": NOW,
        })
        print(f"{product} {path.name} rows={md.num_rows} bytes={path.stat().st_size}", flush=True)
    return rows


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = fields or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def schema_rows(product: str, directory: Path) -> list[dict]:
    files = list_parquet(directory)
    first = pq.read_schema(files[0])
    union = list(first.names)
    seen = set(union)
    mismatches = []
    for path in files[1:]:
        names = pq.read_schema(path).names
        if names != first.names:
            mismatches.append(path.name)
        for n in names:
            if n not in seen:
                seen.add(n)
                union.append(n)
    rows = []
    sch = {f.name: str(f.type) for f in first}
    for i, name in enumerate(union):
        rows.append({
            "product": product,
            "column": name,
            "arrow_type": sch.get(name, "MISSING_FROM_FIRST_SHARD"),
            "position": i,
            "n_shards_schema_mismatch": len(mismatches),
            "mismatch_example": mismatches[0] if mismatches else "",
        })
    return rows


def scan_product_counts(
    files: list[Path],
    *,
    product: str,
    fips_col: str,
    year_from,
) -> tuple[list[dict], dict]:
    """Per-shard projected scans so a 6.6GB history extract is not fully materialized."""
    counts: Counter = Counter()
    fips_set: set[str] = set()
    n_rows = 0
    attom_missing = 0
    extras: dict = defaultdict(int)
    date_mins, date_maxs = [], []
    rec_mins, rec_maxs = [], []
    year_mins, year_maxs = [], []
    tid_nonnull = 0
    # TRANSACTIONID uniqueness is approximate: count per-shard distinct + collisions across shards later
    for path in files:
        pf = pq.ParquetFile(path)
        schema_names = pf.schema_arrow.names
        cols = [c for c in year_from["columns"] if c in schema_names]
        table = pf.read(columns=cols)
        n_rows += table.num_rows
        fips = pc.utf8_trim_whitespace(pc.cast(table[fips_col], pa.string()))
        fips_py = fips.to_pylist()
        fips_set.update(x for x in fips_py if x)
        years = year_from["year"](table)
        year_py = years if isinstance(years, list) else years.to_pylist()
        for f, y in zip(fips_py, year_py):
            counts[(f, y)] += 1
        if "ATTOMID" in cols:
            attom_missing += int(pc.sum(pc.invert(pc.is_valid(table["ATTOMID"]))).as_py() or 0)
        year_from["accumulate"](table, extras, date_mins, date_maxs, rec_mins, rec_maxs, year_mins, year_maxs)
        print(f"scan {product} {path.name} rows={table.num_rows}", flush=True)
        del table
    rows = [{"product": product, "fips": f, "year": y, "n": n_}
            for (f, y), n_ in sorted(counts.items(), key=lambda kv: (str(kv[0][0] or ""), kv[0][1] if kv[0][1] is not None else -1))]
    summary = {
        "n_rows": n_rows,
        "fips_values": sorted(fips_set),
        "attomid_missing": attom_missing,
        **year_from["summarize"](extras, date_mins, date_maxs, rec_mins, rec_maxs, year_mins, year_maxs, fips_set, n_rows),
    }
    return rows, summary


def recorder_year_spec() -> dict:
    def year(table):
        inst = table["INSTRUMENTDATE"] if "INSTRUMENTDATE" in table.column_names else None
        rec = table["RECORDINGDATE"] if "RECORDINGDATE" in table.column_names else None
        if inst is None:
            return [None] * table.num_rows
        y = pc.year(inst)
        if rec is not None:
            y = pc.if_else(pc.is_valid(y), y, pc.year(rec))
        return y

    def accumulate(table, extras, date_mins, date_maxs, rec_mins, rec_maxs, year_mins, year_maxs):
        if "INSTRUMENTDATE" in table.column_names:
            date_mins.append(pc.min(table["INSTRUMENTDATE"]).as_py())
            date_maxs.append(pc.max(table["INSTRUMENTDATE"]).as_py())
        if "RECORDINGDATE" in table.column_names:
            rec_mins.append(pc.min(table["RECORDINGDATE"]).as_py())
            rec_maxs.append(pc.max(table["RECORDINGDATE"]).as_py())
        if "TRANSACTIONID" in table.column_names:
            extras["tid_nonnull"] += int(pc.sum(pc.is_valid(table["TRANSACTIONID"])).as_py() or 0)
            extras["tid_distinct_sum_shards"] += int(pc.count_distinct(table["TRANSACTIONID"]).as_py() or 0)
        if "APNFORMATTED" in table.column_names:
            extras["apnformatted_nonnull"] += int(pc.sum(pc.is_valid(table["APNFORMATTED"])).as_py() or 0)
        if "APNORIGINAL" in table.column_names:
            extras["apnoriginal_nonnull"] += int(pc.sum(pc.is_valid(table["APNORIGINAL"])).as_py() or 0)

    def summarize(extras, date_mins, date_maxs, rec_mins, rec_maxs, year_mins, year_maxs, fips_set, n_rows):
        def _m(xs, fn):
            xs = [x for x in xs if x is not None]
            return str(fn(xs)) if xs else ""
        return {
            "instrument_min": _m(date_mins, min),
            "instrument_max": _m(date_maxs, max),
            "recording_min": _m(rec_mins, min),
            "recording_max": _m(rec_maxs, max),
            "transactionid_nonnull": extras.get("tid_nonnull", 0),
            "transactionid_distinct_sum_over_shards": extras.get("tid_distinct_sum_shards", 0),
            "apnformatted_nonnull": extras.get("apnformatted_nonnull", 0),
            "apnoriginal_nonnull": extras.get("apnoriginal_nonnull", 0),
            "contains_st_louis_city_29510": "29510" in fips_set,
            "contains_wayne_26163": "26163" in fips_set,
            "contains_philadelphia_42101": "42101" in fips_set,
            "contains_st_louis_county_29189": "29189" in fips_set,
        }

    return {
        "columns": [
            "DOCUMENTRECORDINGCOUNTYFIPS", "INSTRUMENTDATE", "RECORDINGDATE",
            "ATTOMID", "TRANSACTIONID", "APNFORMATTED", "APNORIGINAL",
        ],
        "year": year,
        "accumulate": accumulate,
        "summarize": summarize,
    }


def history_year_spec() -> dict:
    def year(table):
        return pc.cast(table["ASSESSORHISTORYYEAR"], pa.int32())

    def accumulate(table, extras, date_mins, date_maxs, rec_mins, rec_maxs, year_mins, year_maxs):
        y = pc.cast(table["ASSESSORHISTORYYEAR"], pa.int32())
        year_mins.append(pc.min(y).as_py())
        year_maxs.append(pc.max(y).as_py())
        n = table.num_rows
        for col, key in [
            ("PARCELNUMBERFORMATTED", "nn_parcel_formatted"),
            ("PARCELNUMBERPREVIOUS", "nn_parcel_previous"),
            ("PARCELNUMBERRAW", "nn_parcel_raw"),
            ("AREABUILDING", "nn_areabuilding"),
            ("YEARBUILT", "nn_yearbuilt"),
            ("BEDROOMSCOUNT", "nn_bedrooms"),
            ("BATHCOUNT", "nn_baths"),
            ("AREALOTSF", "nn_lot"),
            ("PROPERTYUSESTANDARDIZED", "nn_use"),
            ("TAXASSESSEDVALUETOTAL", "nn_tav"),
            ("PROPERTYJURISDICTIONNAME", "nn_jurisdiction"),
        ]:
            if col in table.column_names:
                extras[key] += int(pc.sum(pc.is_valid(table[col])).as_py() or 0)
        extras["n_acc"] += n

    def summarize(extras, date_mins, date_maxs, rec_mins, rec_maxs, year_mins, year_maxs, fips_set, n_rows):
        def share(key):
            return 1.0 - extras.get(key, 0) / max(n_rows, 1)
        return {
            "history_year_min": min(x for x in year_mins if x is not None) if year_mins else "",
            "history_year_max": max(x for x in year_maxs if x is not None) if year_maxs else "",
            "missing_share_PARCELNUMBERFORMATTED": share("nn_parcel_formatted"),
            "missing_share_PARCELNUMBERPREVIOUS": share("nn_parcel_previous"),
            "missing_share_PARCELNUMBERRAW": share("nn_parcel_raw"),
            "missing_share_AREABUILDING": share("nn_areabuilding"),
            "missing_share_YEARBUILT": share("nn_yearbuilt"),
            "missing_share_BEDROOMSCOUNT": share("nn_bedrooms"),
            "missing_share_BATHCOUNT": share("nn_baths"),
            "missing_share_AREALOTSF": share("nn_lot"),
            "missing_share_PROPERTYUSESTANDARDIZED": share("nn_use"),
            "missing_share_TAXASSESSEDVALUETOTAL": share("nn_tav"),
            "missing_share_PROPERTYJURISDICTIONNAME": share("nn_jurisdiction"),
            "contains_st_louis_city_29510": "29510" in fips_set,
            "contains_wayne_26163": "26163" in fips_set,
            "contains_philadelphia_42101": "42101" in fips_set,
            "contains_st_louis_county_29189": "29189" in fips_set,
        }

    return {
        "columns": [
            "SITUSSTATECOUNTYFIPS", "ASSESSORHISTORYYEAR", "ATTOMID",
            "PARCELNUMBERFORMATTED", "PARCELNUMBERPREVIOUS", "PARCELNUMBERRAW",
            "PROPERTYJURISDICTIONNAME", "AREABUILDING", "YEARBUILT", "BEDROOMSCOUNT",
            "BATHCOUNT", "AREALOTSF", "PROPERTYUSESTANDARDIZED", "TAXASSESSEDVALUETOTAL",
        ],
        "year": year,
        "accumulate": accumulate,
        "summarize": summarize,
    }


def field_coverage(product: str, summary: dict, extra: dict) -> list[dict]:
    rows = []
    for k, v in {**summary, **extra}.items():
        rows.append({"product": product, "field_or_check": k, "value": v})
    return rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rec_files = list_parquet(RECORDER_DIR)
    hist_files = list_parquet(HISTORY_DIR)
    print(f"recorder shards={len(rec_files)} history shards={len(hist_files)}", flush=True)
    manifest = inventory_files("recorder", RECORDER_DIR) + inventory_files("history", HISTORY_DIR)
    write_csv(OUT / "new_dewey_file_manifest.csv", manifest)
    schema = schema_rows("recorder", RECORDER_DIR) + schema_rows("history", HISTORY_DIR)
    write_csv(OUT / "new_dewey_schema.csv", schema)
    rec_counts, rec_sum = scan_product_counts(
        rec_files, product="recorder", fips_col="DOCUMENTRECORDINGCOUNTYFIPS", year_from=recorder_year_spec(),
    )
    hist_counts, hist_sum = scan_product_counts(
        hist_files, product="history", fips_col="SITUSSTATECOUNTYFIPS", year_from=history_year_spec(),
    )
    write_csv(OUT / "fips_year_counts.csv", rec_counts + hist_counts)
    coverage = field_coverage("recorder", rec_sum, {"n_shards": len(rec_files)}) + field_coverage(
        "history", hist_sum, {"n_shards": len(hist_files)}
    )
    write_csv(OUT / "field_coverage.csv", coverage)
    print("recorder fips", rec_sum["fips_values"], "n", rec_sum["n_rows"], flush=True)
    print("history fips", hist_sum["fips_values"], "n", hist_sum["n_rows"], flush=True)
    print("wrote", OUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
