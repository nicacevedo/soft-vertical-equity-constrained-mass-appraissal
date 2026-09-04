#!/usr/bin/env python3
"""Step 2 cohort-retention audit under the FROZEN residential_code_mapping.yaml.

Descriptive only: reports what share of each jurisdiction's History rows the
frozen PRIMARY_RESIDENTIAL set captures, by jurisdiction and (where the price
signal is available) by price decile. Per the frozen mapping's explicit rule,
membership is NEVER adjusted based on what this audit finds -- a materially
lower retention share in one jurisdiction (e.g. Philadelphia, whose condo
share code 366 is excluded by the authoritative source) is reported as a real
housing-composition/cohort-coverage feature, not treated as a defect to fix
by broadening the primary set.

Reuses cohort/jurisdiction_code_frequency.csv (already built by
property_use_frequency_table.py) rather than re-scanning the History caches.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from analysis.external_jurisdiction_benchmark_v1.scripts.v1_common import ANALYSIS, ALL_KEYS  # noqa: E402

MAPPING_PATH = ANALYSIS / "cohort" / "residential_code_mapping.yaml"
FREQ_PATH = ANALYSIS / "cohort" / "jurisdiction_code_frequency.csv"


def main() -> int:
    mapping = yaml.safe_load(MAPPING_PATH.read_text())
    if mapping.get("status") != "FROZEN":
        raise SystemExit(f"residential_code_mapping.yaml status={mapping.get('status')!r}, not FROZEN; refusing to compute retention")
    primary_codes = {str(c) for c in mapping["PRIMARY_RESIDENTIAL"]["codes"]}
    legacy_385 = {str(c) for c in mapping["LEGACY_385_ONLY"]["codes"]}
    appendix_confirmed = set(mapping.get("BROAD_RESIDENTIAL_APPENDIX", {}).get("confirmed_members", {}).keys())

    freq = pd.read_csv(FREQ_PATH)
    freq["use_code"] = freq["use_code"].astype(str).str.replace(r"\.0$", "", regex=True)

    rows = []
    for key in ALL_KEYS:
        sub = freq.loc[freq.county_key == key]
        total = sub["n"].sum()
        primary_n = sub.loc[sub.use_code.isin(primary_codes), "n"].sum()
        legacy_n = sub.loc[sub.use_code.isin(legacy_385), "n"].sum()
        appendix_n = sub.loc[sub.use_code.isin(appendix_confirmed), "n"].sum()
        unclassified_n = total - primary_n - appendix_n
        rows.append({
            "county_key": key,
            "n_history_rows": int(total),
            "n_primary_residential": int(primary_n),
            "primary_residential_share": primary_n / total if total else float("nan"),
            "n_legacy_385": int(legacy_n),
            "legacy_385_share": legacy_n / total if total else float("nan"),
            "n_confirmed_appendix_366_condo": int(appendix_n),
            "confirmed_appendix_share": appendix_n / total if total else float("nan"),
            "n_unclassified_excluded": int(unclassified_n),
            "unclassified_share": unclassified_n / total if total else float("nan"),
        })
    out = pd.DataFrame(rows)
    out.to_csv(ANALYSIS / "cohort" / "cohort_retention.csv", index=False)

    spread = out["primary_residential_share"].max() - out["primary_residential_share"].min()
    print(out.to_string(index=False))
    print()
    print(f"primary_residential_share spread across 9 jurisdictions: {spread:.4f}")
    lowest = out.loc[out["primary_residential_share"].idxmin()]
    highest = out.loc[out["primary_residential_share"].idxmax()]
    print(f"lowest: {lowest['county_key']} ({lowest['primary_residential_share']:.4f})")
    print(f"highest: {highest['county_key']} ({highest['primary_residential_share']:.4f})")

    interpretation = []
    if spread > 0.30:
        interpretation.append(
            "Spread exceeds 0.30. Per the frozen mapping, this is reported as a real "
            "housing-composition/cohort-coverage difference (e.g. condo share via excluded "
            "code 366), NOT a signal to broaden PRIMARY_RESIDENTIAL. No membership change made."
        )
    else:
        interpretation.append("Spread is moderate; no jurisdiction shows an extreme coverage cliff.")
    (ANALYSIS / "cohort" / "cohort_retention_interpretation.md").write_text(
        "# Cohort-retention interpretation (descriptive; membership not adjusted)\n\n"
        + "\n".join(interpretation) + "\n\n" + out.to_markdown(index=False), encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
