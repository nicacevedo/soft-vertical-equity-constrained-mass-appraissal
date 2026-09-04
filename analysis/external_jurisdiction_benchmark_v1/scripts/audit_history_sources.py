#!/usr/bin/env python3
"""Read-only audit: narrow ("-2016-2025-all-features") vs broad
("-2006-2025-all-features") Assessor History sources for Cook and Allegheny.

Writes analysis/external_jurisdiction_benchmark_v1/audits/history_source_resolution.yaml.
Does NOT change the sale window (2016-2025) or write any cache. Purely diagnostic.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.other_counties_benchmars import clean_fips, fips_scan_filter, readable_parquet_files  # noqa: E402
from analysis.external_jurisdiction_benchmark_v1.scripts.v1_common import ANALYSIS, sha256_file  # noqa: E402

DEWEY = ROOT / "data" / "dewey-downloads"
TARGETS = {
    "cook": {
        "fips": "17031",
        "narrow": DEWEY / "cookcounty-2016-2025-all-features",
        "broad": DEWEY / "cook-2006-2025-all-features",
    },
    "allegheny": {
        "fips": "42003",
        "narrow": DEWEY / "allegheny-county-2016-2025-all-features",
        "broad": DEWEY / "allegheny-county-2006-2025-all-features",
    },
}
COMPARE_COLS = [
    "ATTOMID", "SITUSSTATECOUNTYFIPS", "ASSESSORHISTORYYEAR", "PROPERTYUSESTANDARDIZED",
    "AREABUILDING", "YEARBUILT", "BEDROOMSCOUNT", "BATHCOUNT",
]


def shard_manifest(folder: Path, label: str) -> dict:
    files = sorted(p for p in folder.iterdir() if p.suffix == ".parquet")
    readable, unreadable = readable_parquet_files(files)
    hashes = {p.name: sha256_file(p) for p in readable}
    return {
        "folder": str(folder), "n_shards_total": len(files), "n_readable": len(readable),
        "n_unreadable": len(unreadable), "unreadable_shards": unreadable,
        "shard_sha256": hashes,
    }


def load_slice(folder: Path, fips: str, readable_files: list[Path]) -> pd.DataFrame:
    source = ds.dataset(readable_files, format="parquet")
    table = source.to_table(
        columns=COMPARE_COLS, filter=fips_scan_filter(source, "SITUSSTATECOUNTYFIPS", fips),
    )
    df = table.to_pandas()
    df["ATTOMID"] = pd.to_numeric(df["ATTOMID"], errors="coerce").astype("Int64")
    df["ASSESSORHISTORYYEAR"] = pd.to_numeric(df["ASSESSORHISTORYYEAR"], errors="coerce").astype("Int64")
    return df


def audit_one(key: str, spec: dict) -> dict:
    fips = spec["fips"]
    result = {"county_key": key, "fips": fips}

    narrow_manifest = shard_manifest(spec["narrow"], "narrow")
    broad_manifest = shard_manifest(spec["broad"], "broad")
    result["narrow_manifest"] = {k: v for k, v in narrow_manifest.items() if k != "shard_sha256"}
    result["broad_manifest"] = {k: v for k, v in broad_manifest.items() if k != "shard_sha256"}
    result["narrow_shard_sha256_sample"] = dict(list(narrow_manifest["shard_sha256"].items())[:3])
    result["broad_shard_sha256_sample"] = dict(list(broad_manifest["shard_sha256"].items())[:3])

    narrow_files = [spec["narrow"] / n for n in narrow_manifest["shard_sha256"]]
    broad_files = [spec["broad"] / n for n in broad_manifest["shard_sha256"]]
    narrow_df = load_slice(spec["narrow"], fips, narrow_files)
    broad_df = load_slice(spec["broad"], fips, broad_files)

    result["narrow_n_rows"] = int(len(narrow_df))
    result["broad_n_rows"] = int(len(broad_df))
    result["narrow_year_min_max"] = [int(narrow_df["ASSESSORHISTORYYEAR"].min()), int(narrow_df["ASSESSORHISTORYYEAR"].max())]
    result["broad_year_min_max"] = [int(broad_df["ASSESSORHISTORYYEAR"].min()), int(broad_df["ASSESSORHISTORYYEAR"].max())]
    result["narrow_rows_by_year"] = {int(k): int(v) for k, v in narrow_df["ASSESSORHISTORYYEAR"].value_counts().sort_index().items()}
    result["broad_rows_by_year"] = {int(k): int(v) for k, v in broad_df["ASSESSORHISTORYYEAR"].value_counts().sort_index().items()}

    narrow_dupe = narrow_df.duplicated(subset=["ATTOMID", "ASSESSORHISTORYYEAR"]).sum()
    broad_dupe = broad_df.duplicated(subset=["ATTOMID", "ASSESSORHISTORYYEAR"]).sum()
    result["narrow_duplicate_attomid_year_pairs"] = int(narrow_dupe)
    result["broad_duplicate_attomid_year_pairs"] = int(broad_dupe)

    narrow_keys = set(zip(narrow_df["ATTOMID"].dropna(), narrow_df["ASSESSORHISTORYYEAR"].dropna()))
    broad_keys = set(zip(broad_df["ATTOMID"].dropna(), broad_df["ASSESSORHISTORYYEAR"].dropna()))
    overlap_years = sorted(set(narrow_df["ASSESSORHISTORYYEAR"].dropna().unique()) & set(broad_df["ASSESSORHISTORYYEAR"].dropna().unique()))
    result["overlapping_years"] = [int(y) for y in overlap_years]
    result["narrow_keys_n"] = len(narrow_keys)
    result["broad_keys_n"] = len(broad_keys)
    result["key_overlap_n"] = len(narrow_keys & broad_keys)
    result["key_overlap_frac_of_narrow"] = len(narrow_keys & broad_keys) / len(narrow_keys) if narrow_keys else None

    if overlap_years:
        n_ov = narrow_df.loc[narrow_df["ASSESSORHISTORYYEAR"].isin(overlap_years)].set_index(["ATTOMID", "ASSESSORHISTORYYEAR"])
        b_ov = broad_df.loc[broad_df["ASSESSORHISTORYYEAR"].isin(overlap_years)].set_index(["ATTOMID", "ASSESSORHISTORYYEAR"])
        common_idx = n_ov.index.intersection(b_ov.index)
        common_idx = common_idx[~common_idx.duplicated()]
        n_ov = n_ov.loc[common_idx]
        b_ov = b_ov.loc[common_idx]
        feature_agreement = {}
        for col in ("PROPERTYUSESTANDARDIZED", "AREABUILDING", "YEARBUILT", "BEDROOMSCOUNT", "BATHCOUNT"):
            if col not in n_ov.columns:
                continue
            a = n_ov[col].astype(str)
            b = b_ov[col].astype(str)
            agree = float((a.to_numpy() == b.to_numpy()).mean()) if len(a) else None
            feature_agreement[col] = agree
        result["feature_agreement_on_overlap"] = feature_agreement
        result["n_common_keys_compared"] = int(len(common_idx))
        result["narrow_missingness_on_overlap"] = {c: float(n_ov[c].isna().mean()) for c in COMPARE_COLS if c in n_ov.columns}
        result["broad_missingness_on_overlap"] = {c: float(b_ov[c].isna().mean()) for c in COMPARE_COLS if c in b_ov.columns}
    else:
        result["feature_agreement_on_overlap"] = {}
        result["n_common_keys_compared"] = 0

    return result


def main() -> int:
    out = {"written_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "audits": {}}
    for key, spec in TARGETS.items():
        print(f"auditing {key}...", flush=True)
        out["audits"][key] = audit_one(key, spec)
        print(f"done {key}", flush=True)
    ANALYSIS.joinpath("audits").mkdir(parents=True, exist_ok=True)
    # Defense in depth: force every key/scalar to a plain, YAML-representable
    # Python type before dumping (numpy/pandas Int64 scalars are not).
    clean = json.loads(json.dumps(out, default=lambda o: int(o) if hasattr(o, "__int__") else str(o)))
    (ANALYSIS / "audits" / "history_source_resolution_raw.yaml").write_text(
        yaml.safe_dump(clean, sort_keys=False, default_flow_style=False)
    )
    print("wrote history_source_resolution_raw.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
