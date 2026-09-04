#!/usr/bin/env python3
"""Step 6: automated pilot QA gate. PASS => full panel launches automatically,
no further user confirmation. FAIL => write BLOCKER.md, do not launch panel.

Checks only engineering/scientific-integrity conditions (Step 15's stop
rules), never scientific favorability. A weak or empty candidate region is
NOT a gate failure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from analysis.external_jurisdiction_benchmark_v1.scripts.run_candidate_region_screen import (  # noqa: E402
    load_direct_path, load_surrogate_path, run_family,
)
from analysis.external_jurisdiction_benchmark_v1.scripts.v1_common import ANALYSIS, OUTPUT  # noqa: E402

FORWARD_LOCK_DATE = pd.Timestamp("2025-01-01")


def check(name: str, ok: bool, detail: str, results: list) -> bool:
    results.append({"check": name, "passed": bool(ok), "detail": detail})
    return ok


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-keys", nargs="+", required=True)
    args = parser.parse_args()
    pilot = args.pilot_keys
    results = []
    all_ok = True

    for key in pilot:
        for family, loader in (("direct", load_direct_path), ("surrogate", load_surrogate_path)):
            meta_path = ANALYSIS / "cv" / f"{key}_{family}_cv_meta.json"
            df = loader(key)
            if df is None or not meta_path.exists():
                all_ok = check(f"{key}/{family}: artifacts exist", False, "missing CV path or meta", results) and all_ok
                continue
            meta = json.loads(meta_path.read_text())

            all_ok = check(f"{key}/{family}: all 7 folds completed", meta.get("n_folds") == 7,
                            f"n_folds={meta.get('n_folds')}", results) and all_ok

            max_date_path = OUTPUT / "modeling_tables" / key / "history_market_core_dev.parquet"
            table = pd.read_parquet(max_date_path, columns=["sale_date"])
            no_2025 = pd.to_datetime(table["sale_date"]).max() < FORWARD_LOCK_DATE
            all_ok = check(f"{key}/{family}: no 2025 data in source table", no_2025,
                            f"max sale_date={pd.to_datetime(table['sale_date']).max()}", results) and all_ok

            zero_rows = df.loc[df["rho_tilde"] == 0.0] if family == "direct" else None
            if family == "direct" and len(zero_rows):
                r2_ok = zero_rows["R2_price"].between(-1, 1).all() and zero_rows["R2_price"].notna().all()
                all_ok = check(f"{key}/direct: rho=0 metrics finite/interpretable", bool(r2_ok),
                                f"R2_price range at rho=0: {zero_rows['R2_price'].min():.3f}-{zero_rows['R2_price'].max():.3f}",
                                results) and all_ok

            if family == "direct":
                sample = df.loc[df["rho_tilde"] > 0].sample(min(20, len(df.loc[df["rho_tilde"] > 0])), random_state=1)
                exact = np.allclose(sample["raw_rho"] * sample["Var_training_y"], sample["rho_tilde"], rtol=1e-9)
                all_ok = check(f"{key}/direct: rho_tilde = raw_rho * Var_training(y) exact",
                                bool(exact), "sampled 20 rows", results) and all_ok

            needed = ["R2_price", "PRD", "PRB", "MKI", "VEI", "Beta_log", "RMSE_log", "MAE", "MAPE"]
            present = [c for c in needed if c in df.columns]
            finite_share = df[present].apply(lambda s: np.isfinite(pd.to_numeric(s, errors="coerce"))).mean().mean() if present else 0
            all_ok = check(f"{key}/{family}: candidate-screen metrics mostly finite", finite_share > 0.8,
                            f"finite_share={finite_share:.3f}", results) and all_ok

            if family == "surrogate":
                branch_path = ANALYSIS / "cv" / f"{key}_surrogate_first_branch_by_fold.csv"
                if branch_path.exists():
                    branch_df = pd.read_csv(branch_path)
                    has_reason = "branch_terminated_by" in branch_df.columns
                    all_ok = check(f"{key}/surrogate: frozen first-branch calibrator used", has_reason,
                                    "branch_terminated_by column present" if has_reason else "MISSING", results) and all_ok

    # Determinism: run the screen twice on the same pilot data, diff.
    for key in pilot:
        for family, loader in (("direct", load_direct_path), ("surrogate", load_surrogate_path)):
            r1 = run_family(key, family, loader)
            r2 = run_family(key, family, loader)
            if r1 is None or r2 is None:
                continue
            same = r1.get("activity_rho_tilde") == r2.get("activity_rho_tilde") and r1.get("guardrail_rho_tilde") == r2.get("guardrail_rho_tilde")
            all_ok = check(f"{key}/{family}: candidate-region screen deterministic", same,
                            f"run1={r1.get('activity_rho_tilde')},{r1.get('guardrail_rho_tilde')} "
                            f"run2={r2.get('activity_rho_tilde')},{r2.get('guardrail_rho_tilde')}", results) and all_ok

    # Grid-censoring: report, do not fail the gate for it (predeclared extension handles it).
    cand_path = ANALYSIS / "candidate_regions" / "candidate_regions.csv"
    censored = []
    if cand_path.exists():
        cand = pd.read_csv(cand_path)
        cand = cand.loc[cand.county_key.isin(pilot)]
        censored = cand.loc[cand.grid_censored_activity | cand.grid_censored_guardrail].to_dict("records")

    verdict = {
        "pilot_keys": pilot, "n_checks": len(results), "n_passed": sum(r["passed"] for r in results),
        "all_passed": all_ok, "checks": results, "grid_censored_endpoints": censored,
    }
    ANALYSIS.joinpath("candidate_regions").mkdir(parents=True, exist_ok=True)
    (ANALYSIS / "candidate_regions" / "pilot_qa_gate.json").write_text(json.dumps(verdict, indent=2, default=str))
    print(json.dumps({"all_passed": all_ok, "n_passed": verdict["n_passed"], "n_checks": verdict["n_checks"],
                       "n_grid_censored": len(censored)}, indent=2))
    for r in results:
        if not r["passed"]:
            print("FAIL:", r["check"], "--", r["detail"])
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
