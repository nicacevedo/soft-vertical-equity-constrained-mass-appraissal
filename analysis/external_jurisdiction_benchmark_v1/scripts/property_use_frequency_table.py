#!/usr/bin/env python3
"""Step 2 prerequisite: jurisdiction x PROPERTYUSESTANDARDIZED code frequency
table, built from the already-constructed History caches. Pure data profiling
-- no cohort decision is made here. Written before the residential mapping is
applied to any model outcome.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from analysis.external_jurisdiction_benchmark_v1.scripts.v1_common import ALL_KEYS, ANALYSIS, OUTPUT  # noqa: E402


def main() -> int:
    rows = []
    for key in ALL_KEYS:
        p = OUTPUT / "cache" / key / "history.parquet"
        table = pq.read_table(p, columns=["PROPERTYUSESTANDARDIZED"])
        s = table.column(0).to_pandas()
        s = s.astype("string").str.strip().str.upper().str.replace(r"\.0$", "", regex=True)
        total = len(s)
        vc = s.value_counts(dropna=False)
        for code, n in vc.items():
            rows.append({
                "county_key": key, "use_code": (code if pd.notna(code) else "MISSING"),
                "n": int(n), "share_of_county_history_rows": float(n) / total,
            })
        print(f"{key}: n_history_rows={total} n_distinct_codes={s.nunique(dropna=True)}", flush=True)

    df = pd.DataFrame(rows)
    ANALYSIS.joinpath("cohort").mkdir(parents=True, exist_ok=True)
    df.to_csv(ANALYSIS / "cohort" / "jurisdiction_code_frequency.csv", index=False)

    pivot = df.pivot_table(index="use_code", columns="county_key", values="n", fill_value=0)
    pivot["total_n"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("total_n", ascending=False)
    pivot.to_csv(ANALYSIS / "cohort" / "jurisdiction_code_frequency_pivot.csv")
    print(pivot.head(40).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
