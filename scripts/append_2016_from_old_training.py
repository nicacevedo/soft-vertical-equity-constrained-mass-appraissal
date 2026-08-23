"""Append 2016 sales from training_data_old.parquet into training_data.parquet.

The two files share the same 201-column CCAO schema. The current file is a later
extract (2017-2025) with two extra columns and a later time_sale_day origin.
This script copies only calendar-2016 rows from the old file, aligns them to the
current schema, concatenates, and recomputes time_sale_day on the merged file
so day 1 is the first sale date in the combined universe.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
CURRENT_PATH = REPO / "data/CCAO/2025/training_data.parquet"
OLD_PATH = REPO / "data/CCAO/2025/training_data_old.parquet"
BACKUP_PATH = REPO / "data/CCAO/2025/pre_2016_append.parquet"
MANIFEST_PATH = REPO / "data/CCAO/2025/training_data_2016_append_manifest.json"


def _cast_column(col: pa.ChunkedArray, field: pa.Field) -> pa.ChunkedArray:
    if col.type.equals(field.type):
        return col
    if pa.types.is_dictionary(col.type):
        col = col.dictionary_decode()
    try:
        return col.cast(field.type, safe=True)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError):
        return col.cast(field.type, safe=False)


def align_to_schema(table: pa.Table, schema: pa.Schema) -> pa.Table:
    arrays = []
    for field in schema:
        if field.name in table.column_names:
            arrays.append(_cast_column(table[field.name], field))
        else:
            arrays.append(pa.nulls(table.num_rows, type=field.type))
    return pa.Table.from_arrays(arrays, schema=schema)


def main() -> None:
    current = pq.read_table(CURRENT_PATH)
    old = pq.read_table(OLD_PATH)
    sale_year = pc.year(old["meta_sale_date"])
    old_2016 = old.filter(pc.equal(sale_year, 2016))
    if old_2016.num_rows == 0:
        raise SystemExit("No 2016 rows found in training_data_old.parquet.")

    current_keys = pa.Table.from_arrays(
        [current["meta_pin"], current["meta_sale_document_num"]],
        names=["meta_pin", "meta_sale_document_num"],
    )
    old_keys = pa.Table.from_arrays(
        [old_2016["meta_pin"], old_2016["meta_sale_document_num"]],
        names=["meta_pin", "meta_sale_document_num"],
    )
    overlap = current_keys.join(old_keys, keys=["meta_pin", "meta_sale_document_num"], join_type="inner")
    if overlap.num_rows:
        raise SystemExit(f"Refusing to append: {overlap.num_rows} 2016 keys already exist in the current file.")

    aligned_2016 = align_to_schema(old_2016, current.schema)
    merged = pa.concat_tables([aligned_2016, current], promote_options="none")
    min_date = pc.min(merged["meta_sale_date"])
    time_sale_day = pc.add(pc.cast(pc.days_between(min_date, merged["meta_sale_date"]), pa.float64()), 1.0)
    merged = merged.set_column(
        merged.schema.get_field_index("time_sale_day"),
        "time_sale_day",
        time_sale_day,
    )
    merged = merged.sort_by([("meta_sale_date", "ascending"), ("meta_pin", "ascending")])

    if merged.schema.names != current.schema.names:
        raise SystemExit("Merged schema names do not match the current file.")
    for field in current.schema:
        if not merged.schema.field(field.name).type.equals(field.type):
            raise SystemExit(f"Merged dtype mismatch for {field.name}: {merged.schema.field(field.name).type} vs {field.type}")
    if merged.num_rows != current.num_rows + old_2016.num_rows:
        raise SystemExit("Merged row count does not equal current + 2016 rows.")

    years = pc.year(merged["meta_sale_date"])
    year_counts = {
        str(int(year)): int(pc.sum(pc.equal(years, int(year))).as_py() or 0)
        for year in range(2016, 2026)
    }
    # 2016 is a leap year, so 2017-01-01 must be day 367 under the combined origin.
    day_2017_start = pc.min(pc.filter(merged["time_sale_day"], pc.equal(years, 2017))).as_py()
    if day_2017_start != 367.0:
        raise SystemExit(f"time_sale_day origin check failed: 2017 start is {day_2017_start}, expected 367.")

    shutil.copy2(CURRENT_PATH, BACKUP_PATH)
    pq.write_table(merged, CURRENT_PATH, compression="snappy")
    manifest = {
        "current_path": str(CURRENT_PATH),
        "old_path": str(OLD_PATH),
        "backup_of_pre_append_current": str(BACKUP_PATH),
        "appended_year": 2016,
        "rows_appended": int(old_2016.num_rows),
        "rows_before": int(current.num_rows),
        "rows_after": int(merged.num_rows),
        "schema_notes": {
            "shared_columns": 201,
            "current_only_columns_filled_null_for_2016": ["sv_outlier_reason", "sv_review_json"],
            "dtype_cast": {"other_affordability_risk_index": "int32 -> double"},
            "time_sale_day_origin": "min(meta_sale_date) so 2016-01-01 == 1 and 2017-01-01 == 367",
        },
        "compatibility": {
            "all_model_predictors_present_in_old": True,
            "2016_keys_already_in_current": 0,
            "overlapping_2017_2023_are_same_sales_with_later_extract_revisions": True,
        },
        "year_counts_after": year_counts,
        "time_sale_day_2017_start": day_2017_start,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
