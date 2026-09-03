#!/usr/bin/env python3
"""Copy completed v2 inventory + Berry anchors into v3 with provenance hashes.

Does not rerun Dewey shard hashing. Does not modify v1 or v2.
"""
from __future__ import annotations

import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "analysis" / "berry_attom_validation_v3" / "scripts"))
from v3_common import ANALYSIS, V2, sha256_file, write_json

INVENTORY = [
    "inventory/new_dewey_file_manifest.csv",
    "inventory/new_dewey_schema.csv",
    "inventory/fips_year_counts.csv",
    "inventory/field_coverage.csv",
    "inventory/INVENTORY_NOTES.md",
    "inventory/environment.json",
    "berry_reproduction/detroit_mi_transactions.parquet",
    "berry_reproduction/philadelphia_pa_transactions.parquet",
    "berry_reproduction/st_louis_county_mo_transactions.parquet",
    "berry_reproduction/reproduction_summary.csv",
    "berry_reproduction/REPRODUCTION_V2_NOTES.md",
    "berry_reproduction/detroit_native_r.json",
    "source_concordance/STEP9_ASSESSMENT_CONCORDANCE.md",
    "source_concordance/existing_six_vs_v2_metadata.csv",
]


def main() -> int:
    rows = []
    for rel in INVENTORY:
        src = V2 / rel
        dest = ANALYSIS / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        rec = {"relative": rel, "v2_path": str(src), "exists": src.exists()}
        if src.exists():
            rec["v2_sha256"] = sha256_file(src)
            rec["v2_bytes"] = src.stat().st_size
            shutil.copy2(src, dest)
            rec["v3_path"] = str(dest)
            rec["v3_sha256"] = sha256_file(dest)
            rec["copied"] = True
            rec["rerun"] = False
        else:
            rec["copied"] = False
            rec["rerun"] = True
        rows.append(rec)
    notes = ANALYSIS / "berry_reproduction" / "PROVENANCE.md"
    notes.write_text(
        "# v3 Berry/inventory provenance\n\n"
        "Copied from frozen v2 without re-filtering. STL 1975-2019 full table is "
        "retained; ATTOM linkage uses the 2005-2019 cohort declared in protocol_v3.yaml.\n"
        "St. Louis is **not** a fully reproduced Berry assessment-ratio benchmark "
        "(no official assessed-value series in the sales extract).\n",
        encoding="utf-8",
    )
    write_json(ANALYSIS / "inventory" / "V2_PROVENANCE.json", {
        "copied_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
    })
    missing = [r["relative"] for r in rows if not r.get("copied")]
    print("copied", len(rows) - len(missing), "missing", missing)
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
