"""Step 2 deliverable: aggregate per-jurisdiction as-of History coverage.

Reads the per-year coverage files already written by build_modeling_tables.py
(``audits/{key}_history_asof_coverage_by_year.csv``, one row per sale_year,
computed against the strict pre-sale History match before the residential
cohort filter) and the History cache manifest, and adds:

  - earliest/latest ASSESSORHISTORYYEAR actually present in each jurisdiction's
    canonical History cache (read directly off the cached parquet, one column
    only -- cheap even for Cook's 32.7M-row broad cache);
  - one pooled "ALL" row per jurisdiction across the full 2016-2024 qualified-
    sales window, with match rate weighted by n_qualified_sales and lag
    percentiles pooled from the modeling table's final (cohort-filtered)
    ``history_lag_days`` column -- documented as such since it is a slightly
    different (post-cohort-filter) population than the strict per-year rows;
  - the frozen fold role from baseline/BASELINE_FREEZE.yaml (PRIMARY_FULL_7_FOLD
    or a predeclared fallback), never invented here.

Read-only with respect to modeling tables and caches; writes only
audits/history_asof_coverage.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

ANALYSIS = Path(__file__).resolve().parents[1]
OUTPUT = ANALYSIS.parents[1] / "output" / "external_jurisdiction_benchmark_v1"
sys.path.insert(0, str(ANALYSIS / "scripts"))
from v1_common import ALL_KEYS  # noqa: E402

FOLD_WINDOW = (2018, 2024)  # frozen 7 calendar-year folds
QUALIFIED_SALES_WINDOW = (2016, 2024)


def history_year_range(key: str) -> tuple[int, int]:
    # Cook's broad History cache is 32.7M rows of decimal128 -- .to_pandas() on
    # this column boxes every value as a Python Decimal and OOMs a login-node
    # shell (confirmed: killed at default memory, exit 137). pc.min_max stays
    # entirely in Arrow's native decimal128 representation, no Python boxing.
    hist_path = OUTPUT / "cache" / key / "history.parquet"
    years = pq.read_table(hist_path, columns=["ASSESSORHISTORYYEAR"])["ASSESSORHISTORYYEAR"]
    mm = pc.min_max(years)
    lo, hi = mm["min"], mm["max"]
    if lo.as_py() is None or hi.as_py() is None:
        raise RuntimeError(f"{key}: ASSESSORHISTORYYEAR is all-null in {hist_path}")
    return int(lo.as_py()), int(hi.as_py())


def fold_role(key: str, freeze: dict) -> str:
    for row in freeze.get("units", freeze) if isinstance(freeze, dict) else freeze:
        if row.get("county_key") == key:
            return row.get("role", "UNKNOWN")
    return "UNKNOWN"


def pooled_lag_stats(key: str) -> dict:
    table_path = OUTPUT / "modeling_tables" / key / "history_market_core_dev.parquet"
    lag = pq.read_table(table_path, columns=["history_lag_days"])["history_lag_days"].to_pandas()
    lag = pd.to_numeric(lag, errors="coerce").dropna()
    if not len(lag):
        return {"median_history_lag_days": None, "p90_history_lag_days": None,
                "share_lag_gt_1yr": None, "share_lag_gt_2yr": None, "share_lag_gt_3yr": None}
    return {
        "median_history_lag_days": float(lag.median()),
        "p90_history_lag_days": float(lag.quantile(0.9)),
        "share_lag_gt_1yr": float((lag > 365).mean()),
        "share_lag_gt_2yr": float((lag > 730).mean()),
        "share_lag_gt_3yr": float((lag > 1095).mean()),
    }


def main() -> None:
    freeze_path = ANALYSIS / "baseline" / "BASELINE_FREEZE.yaml"
    freeze = yaml.safe_load(freeze_path.read_text()) if freeze_path.exists() else []

    rows = []
    for key in ALL_KEYS:
        by_year_path = ANALYSIS / "audits" / f"{key}_history_asof_coverage_by_year.csv"
        if not by_year_path.exists():
            raise RuntimeError(f"{key}: missing {by_year_path.name}; run build_modeling_tables.py first")
        by_year = pd.read_csv(by_year_path)
        by_year = by_year.loc[by_year["sale_year"].between(*QUALIFIED_SALES_WINDOW)].copy()

        hist_min, hist_max = history_year_range(key)
        role = fold_role(key, freeze)

        for _, r in by_year.iterrows():
            rows.append({
                "county_key": key,
                "role": role,
                "history_earliest_year": hist_min,
                "history_latest_year": hist_max,
                "row_type": "by_year",
                "sale_year": int(r["sale_year"]),
                "n_qualified_sales": int(r["n_qualified_sales"]),
                "n_strict_prehistory_match": int(r["n_strict_prehistory_match"]),
                "match_rate": r["share_with_strict_prehistory_match"],
                "median_history_lag_days": r["median_history_lag_days"],
                "p90_history_lag_days": r["p90_history_lag_days"],
                "share_lag_gt_1yr": r["share_lag_gt_1yr"],
                "share_lag_gt_2yr": r["share_lag_gt_2yr"],
                "share_lag_gt_3yr": r["share_lag_gt_3yr"],
            })

        n_q = int(by_year["n_qualified_sales"].sum())
        n_m = int(by_year["n_strict_prehistory_match"].sum())
        pooled = pooled_lag_stats(key)
        rows.append({
            "county_key": key,
            "role": role,
            "history_earliest_year": hist_min,
            "history_latest_year": hist_max,
            "row_type": "ALL_2016_2024_POOLED",
            "sale_year": None,
            "n_qualified_sales": n_q,
            "n_strict_prehistory_match": n_m,
            "match_rate": n_m / n_q if n_q else float("nan"),
            **pooled,
        })

    out = pd.DataFrame(rows)
    out_path = ANALYSIS / "audits" / "history_asof_coverage.csv"
    out.to_csv(out_path, index=False)

    print(f"wrote {out_path} ({len(out)} rows, {out['county_key'].nunique()} jurisdictions)")
    all_rows = out.loc[out["row_type"] == "ALL_2016_2024_POOLED"]
    for _, r in all_rows.sort_values("county_key").iterrows():
        n_folds = min(r["history_latest_year"], FOLD_WINDOW[1]) - FOLD_WINDOW[0] + 1
        print(f"  {r['county_key']:16s} role={r['role']:24s} hist=[{r['history_earliest_year']}-{r['history_latest_year']}] "
              f"match_rate={r['match_rate']:.4f} n_qualified={r['n_qualified_sales']}")


if __name__ == "__main__":
    main()
