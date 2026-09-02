#!/usr/bin/env python3
"""Schema-first inspection of Berry/CMF raw tables. Metadata before full loads."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[3]
RAW = REPO / "data" / "berry_cmf" / "raw"
OUT = REPO / "analysis" / "berry_cmf_validation"
LOG = OUT / "logs"

SKIP_SUFFIXES = {".headers", ".part", ".rmd", ".r", ".md", ".pdf", ".html", ".lock"}
SKIP_NAMES = {".git", "__pycache__"}
MAX_SAMPLE_ROWS = 5000
CSV_BYTES = 2_000_000

SEMANTIC_RULES = [
    # (keywords in lower name, semantic, role)
    (("sale_price", "saleprice", "price", "netconsideration", "transferamount", "adj. sale", "ls1saleamount"), "sale_price", "outcome"),
    (("sale_date", "saledate", "saledt", "instrumentdate", "date of sale", "sale date"), "sale_date", "outcome_timing"),
    (("sale_year", "saleyear", "year_of_sale", "joinyr"), "sale_year", "outcome_timing"),
    (("assessed", "assmt", "asd. when sold", "bor_ccao_ass", "asmtot", "totval", "lpv", "total_assessed", "assessed_value"), "assessed_value", "diagnostic_only"),
    (("market_value", "apr tot", "aprtot", "totfcv", "full_market"), "market_or_appraised_value", "diagnostic_only"),
    (("parcel", "parid", "ain", "pin", "apn", "sbl", "property_id", "pid"), "parcel_id", "identifier"),
    (("transaction", "sale_id", "document"), "transaction_id", "identifier"),
    (("arms", "arm's", "arms_length", "saletype", "saleval", "terms of sale", "sale_type"), "sale_validity", "diagnostic_only"),
    (("class", "luc", "property_class", "proptype", "land_class", "bor_class", "propertyuse"), "property_class", "predictor_candidate"),
    (("bldg", "building", "area", "sqft", "sq_ft", "living", "resarea"), "building_area", "predictor_candidate"),
    (("lot", "land_area", "acres", "frontage"), "lot_area", "predictor_candidate"),
    (("year_built", "yrblt", "yr_blt", "yearbuilt"), "year_built", "predictor_candidate"),
    (("bed", "bedroom"), "bedrooms", "predictor_candidate"),
    (("bath", "bathroom"), "bathrooms", "predictor_candidate"),
    (("grade", "quality", "condition", "cdu"), "quality_condition", "predictor_candidate"),
    (("style", "stories", "structure"), "style", "predictor_candidate"),
    (("nbhd", "neighborhood", "tract", "township", "geoid", "census"), "geography_id", "predictor_candidate"),
    (("address", "situs", "street"), "address", "identifier"),
    (("lat", "lon", "long", "x_coord", "y_coord", "wgs84"), "coordinates", "predictor_candidate"),
    (("tax", "levy"), "tax_variable", "diagnostic_only"),
]


def infer_semantic(col: str) -> Tuple[str, str, str]:
    n = col.lower().strip()
    for keys, semantic, role in SEMANTIC_RULES:
        if any(k in n for k in keys):
            return semantic, role, f"name_match:{[k for k in keys if k in n][0]}"
    return "unresolved", "unresolved", "no_keyword_match"


def file_kind(path: Path) -> str:
    suf = path.suffix.lower()
    if suf in {".csv", ".txt"}:
        return "csv_or_delimited"
    if suf in {".dta"}:
        return "stata"
    if suf in {".xlsx", ".xls"}:
        return "excel"
    if suf in {".parquet"}:
        return "parquet"
    if suf in {".rds"}:
        return "r_rds"
    if suf in {".zip"}:
        return "zip"
    if suf in {".gpkg", ".gdb"}:
        return "geodata"
    return "other"


def cheap_csv(path: Path) -> Dict[str, Any]:
    raw = path.read_bytes()[:8000]
    sample = raw.decode("utf-8", "replace")
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",|\t;")
        delim = dialect.delimiter
    except Exception:
        delim = "," if sample.count(",") >= sample.count("|") else "|"
    header = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter=delim)
        try:
            header = next(reader)
        except StopIteration:
            return {"n_rows": 0, "columns": [], "delim": delim, "sample": []}
        sample_rows = []
        for i, row in enumerate(reader):
            if i < 8:
                sample_rows.append(row[:40])
            if i >= 8:
                break
    path_bytes = path.stat().st_size
    if path_bytes > 5_000_000:
        # estimate rows from mean line length in the 8k sample
        nlines = max(sample.count("\n"), 1)
        est = int(path_bytes / max(len(raw) / nlines, 1)) - 1
        return {
            "n_rows": None,
            "n_rows_estimated": max(est, 0),
            "columns": header,
            "delim": delim,
            "sample": sample_rows,
            "row_count_method": "byte_over_mean_line_length",
        }
    # small files: exact line count without csv parser
    n = 0
    with path.open("rb") as f:
        for _ in f:
            n += 1
    return {
        "n_rows": max(n - 1, 0),
        "columns": header,
        "delim": delim,
        "sample": sample_rows,
        "row_count_method": "full_line_count",
    }


def inspect_csv(path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    meta = cheap_csv(path)
    cols = [c.strip() for c in meta.get("columns") or []]
    delim = meta.get("delim", ",")
    try:
        df = pd.read_csv(
            path,
            sep=delim,
            nrows=MAX_SAMPLE_ROWS,
            low_memory=False,
            dtype=str,
            encoding="utf-8",
            encoding_errors="replace",
        )
    except Exception as e:
        return pd.DataFrame(), {**meta, "load_error": str(e), "columns": cols}
    return df, meta


def inspect_stata(path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    meta: Dict[str, Any] = {"file_kind": "stata"}
    try:
        reader = pd.read_stata(path, iterator=True, convert_categoricals=False)
        df = reader.read(MAX_SAMPLE_ROWS)
        try:
            n = int(getattr(reader, "nobs", None) or getattr(reader, "num_rows", None) or -1)
        except Exception:
            n = -1
        meta["n_rows"] = n if n >= 0 else None
        meta["stata_columns"] = list(df.columns)
        return df, meta
    except Exception as e:
        meta["load_error"] = str(e)
        # try pyreadstat if present
        try:
            import pyreadstat
            df, md = pyreadstat.read_dta(path, metadataonly=True)
            meta["n_rows"] = md.number_rows
            meta["columns"] = md.column_names
            return pd.DataFrame(), meta
        except Exception as e2:
            meta["load_error2"] = str(e2)
            return pd.DataFrame(), meta


def inspect_excel(path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    meta: Dict[str, Any] = {"file_kind": "excel"}
    try:
        xl = pd.ExcelFile(path)
        meta["sheets"] = xl.sheet_names
        df = pd.read_excel(path, sheet_name=0, nrows=MAX_SAMPLE_ROWS, dtype=str)
        return df, meta
    except Exception as e:
        meta["load_error"] = str(e)
        return pd.DataFrame(), meta


def inspect_zip(path: Path) -> Dict[str, Any]:
    members = []
    try:
        with zipfile.ZipFile(path) as z:
            for i in z.infolist():
                members.append({"name": i.filename, "file_size": i.file_size, "compress_size": i.compress_size})
    except Exception as e:
        return {"load_error": str(e), "members": []}
    return {"n_members": len(members), "members": members[:200], "members_truncated": len(members) > 200}


def field_rows(jur: str, table: str, path: Path, df: pd.DataFrame, table_meta: Dict[str, Any]) -> List[dict]:
    rows = []
    cols = list(df.columns) if len(df.columns) else list(table_meta.get("columns") or table_meta.get("stata_columns") or [])
    n_sample = len(df)
    for col in cols:
        s = df[col] if col in df.columns else pd.Series(dtype=str)
        semantic, role, evidence = infer_semantic(str(col))
        missing = float(s.isna().mean()) if n_sample else None
        if n_sample:
            nonempty = s.dropna().astype(str)
            nonempty = nonempty[nonempty.str.strip() != ""]
            missing = 1.0 - (len(nonempty) / max(n_sample, 1))
            examples = nonempty.head(5).tolist()
            dtype = str(s.dtype)
        else:
            examples = []
            dtype = "unknown"
        # timing: only mark if date-like name
        timing = "unknown"
        usable_before_sale = "unknown"
        if semantic in {"sale_date", "sale_year"}:
            timing = "sale_event"
            usable_before_sale = "n/a_outcome"
        elif semantic == "assessed_value":
            timing = "assessment_stage_unresolved"
            usable_before_sale = "diagnostic_only"
        elif role == "predictor_candidate":
            timing = "TIMING_UNRESOLVED"
            usable_before_sale = "not_until_temporal_audit"
        rows.append({
            "jurisdiction": jur,
            "table_name": table,
            "source_path": str(path.relative_to(RAW)),
            "n_rows": table_meta.get("n_rows"),
            "n_rows_estimated": table_meta.get("n_rows_estimated"),
            "n_columns": len(cols),
            "year_min": None,
            "year_max": None,
            "column": str(col),
            "inferred_semantic": semantic,
            "inferred_role": role,
            "evidence": evidence,
            "dtype_sample": dtype,
            "missing_share_sample": missing,
            "example_values": json.dumps(examples, ensure_ascii=False)[:500],
            "timing_note": timing,
            "usable_before_sale": usable_before_sale,
            "sample_n": n_sample,
            "file_bytes": path.stat().st_size,
            "file_kind": file_kind(path),
            "row_count_method": table_meta.get("row_count_method", ""),
            "load_error": table_meta.get("load_error", ""),
        })
    if not cols:
        rows.append({
            "jurisdiction": jur,
            "table_name": table,
            "source_path": str(path.relative_to(RAW)),
            "n_rows": table_meta.get("n_rows"),
            "n_rows_estimated": table_meta.get("n_rows_estimated"),
            "n_columns": 0,
            "year_min": None,
            "year_max": None,
            "column": "",
            "inferred_semantic": "unusable",
            "inferred_role": "unusable",
            "evidence": table_meta.get("load_error", "no_columns"),
            "dtype_sample": "",
            "missing_share_sample": None,
            "example_values": "",
            "timing_note": "unknown",
            "usable_before_sale": "unknown",
            "sample_n": 0,
            "file_bytes": path.stat().st_size,
            "file_kind": file_kind(path),
            "row_count_method": table_meta.get("row_count_method", ""),
            "load_error": table_meta.get("load_error", ""),
        })
    return rows


def year_range_from_df(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    if df.empty:
        return None, None
    candidates = [c for c in df.columns if any(k in str(c).lower() for k in ["year", "date", "saledt", "sale_date"])]
    vals = []
    for c in candidates:
        s = pd.to_datetime(df[c], errors="coerce")
        if s.notna().any():
            vals.append(s)
            continue
        num = pd.to_numeric(df[c], errors="coerce")
        if num.between(1800, 2035).any():
            vals.append(num)
    if not vals:
        return None, None
    stacked = pd.concat(vals, axis=0)
    stacked = stacked.dropna()
    if stacked.empty:
        return None, None
    return str(stacked.min()), str(stacked.max())


def iter_data_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in SKIP_NAMES for part in p.parts):
            continue
        if p.suffix.lower() in SKIP_SUFFIXES:
            continue
        if p.name.startswith(".~") or p.name in {".DS_Store", "Thumbs.db"}:
            continue
        if p.name.upper().startswith("README"):
            continue
        yield p


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows: List[dict] = []
    coverage_rows: List[dict] = []
    wanted_jurs = {
        "detroit_mi", "philadelphia_pa", "orleans_la", "franklin_oh",
        "st_louis_county_mo", "cook_il", "nyc_ny", "erie_ny", "clark_nv",
        "los_angeles_ca", "maricopa_az",
    }
    files = [p for p in iter_data_files(RAW) if any(j in p.parts for j in wanted_jurs)]
    print(f"inspecting {len(files)} files", flush=True)
    for path in sorted(files):
        jur = next(j for j in wanted_jurs if j in path.parts)
        kind = file_kind(path)
        table = str(path.relative_to(RAW / jur))
        print(f"  {jur} {kind} {path.name} {path.stat().st_size}", flush=True)
        df = pd.DataFrame()
        meta: Dict[str, Any] = {}
        try:
            if kind == "csv_or_delimited":
                df, meta = inspect_csv(path)
            elif kind == "stata":
                df, meta = inspect_stata(path)
            elif kind == "excel":
                df, meta = inspect_excel(path)
            elif kind == "zip":
                zmeta = inspect_zip(path)
                (OUT / "logs" / "archive_listings").mkdir(parents=True, exist_ok=True)
                (OUT / "logs" / "archive_listings" / (path.name + ".json")).write_text(json.dumps(zmeta, indent=2))
                all_rows.append({
                    "jurisdiction": jur, "table_name": table, "source_path": str(path.relative_to(RAW)),
                    "n_rows": zmeta.get("n_members"), "n_rows_estimated": None, "n_columns": None,
                    "year_min": None, "year_max": None, "column": "__ARCHIVE__",
                    "inferred_semantic": "archive", "inferred_role": "container",
                    "evidence": f"zip_members={zmeta.get('n_members')}",
                    "dtype_sample": "zip", "missing_share_sample": None,
                    "example_values": json.dumps([m["name"] for m in (zmeta.get("members") or [])[:10]]),
                    "timing_note": "n/a", "usable_before_sale": "n/a", "sample_n": 0,
                    "file_bytes": path.stat().st_size, "file_kind": "zip",
                    "row_count_method": "zip_infolist", "load_error": zmeta.get("load_error", ""),
                })
                continue
            elif kind == "r_rds":
                all_rows.append({
                    "jurisdiction": jur, "table_name": table, "source_path": str(path.relative_to(RAW)),
                    "n_rows": None, "n_rows_estimated": None, "n_columns": None,
                    "year_min": None, "year_max": None, "column": "__RDS__",
                    "inferred_semantic": "r_serialized", "inferred_role": "processed_intermediate",
                    "evidence": "rds_not_parsed_in_python_schema_pass; inspect via R",
                    "dtype_sample": "RDS", "missing_share_sample": None, "example_values": "",
                    "timing_note": "unknown", "usable_before_sale": "unknown", "sample_n": 0,
                    "file_bytes": path.stat().st_size, "file_kind": "r_rds",
                    "row_count_method": "deferred_to_R", "load_error": "",
                })
                continue
            else:
                continue
        except Exception as e:
            meta = {"load_error": str(e), "columns": []}
            df = pd.DataFrame()
        ymin, ymax = year_range_from_df(df)
        frows = field_rows(jur, table, path, df, meta)
        for r in frows:
            r["year_min"] = ymin
            r["year_max"] = ymax
        all_rows.extend(frows)

    inv = pd.DataFrame(all_rows)
    inv_path = OUT / "schema_inventory.parquet"
    inv.to_parquet(inv_path, index=False)
    print("wrote", inv_path, "rows", len(inv), flush=True)

    # feature coverage matrix: jurisdictions x semantic groups
    semantics = sorted(inv["inferred_semantic"].dropna().unique())
    jurs = sorted(inv["jurisdiction"].dropna().unique())
    cov = []
    for j in jurs:
        sub = inv[inv["jurisdiction"] == j]
        row = {"jurisdiction": j, "n_tables": sub["table_name"].nunique(), "n_columns_inspected": len(sub)}
        for s in semantics:
            hit = sub[sub["inferred_semantic"] == s]
            row[f"has_{s}"] = int(len(hit) > 0)
            row[f"n_{s}"] = int(len(hit))
            if len(hit):
                row[f"ex_{s}"] = ";".join(sorted(hit["column"].astype(str).unique())[:8])
        cov.append(row)
    cov_df = pd.DataFrame(cov)
    cov_path = OUT / "feature_coverage_matrix.csv"
    cov_df.to_csv(cov_path, index=False)
    print("wrote", cov_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
