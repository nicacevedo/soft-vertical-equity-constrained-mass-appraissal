#!/usr/bin/env python3
"""Step 13: achieved-mechanism anchors. A_beta = 1 - |beta(rho)|/|beta(0)|.

Descriptive cross-method coordinates only, never a model-selection criterion.
If baseline beta is unstable in sign or near-neutral across folds, reports
BASELINE_MECHANISM_NEAR_NEUTRAL with absolute beta instead of a percentage.
Interpolates only between adjacent points on the SAME path (fold-mean curve),
never across a Surrogate branch break.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from analysis.external_jurisdiction_benchmark_v1.scripts.v1_common import ALL_KEYS, ANALYSIS, write_json  # noqa: E402

ANCHORS = (0.25, 0.50, 0.75, 0.90)
NEAR_NEUTRAL_ABS_BETA = 0.02  # |beta| below this is treated as near-neutral


def beta_stable(fold_betas: np.ndarray) -> bool:
    fold_betas = fold_betas[np.isfinite(fold_betas)]
    if len(fold_betas) < 2:
        return False
    same_sign = np.all(fold_betas > 0) or np.all(fold_betas < 0)
    not_near_neutral = np.median(np.abs(fold_betas)) > NEAR_NEUTRAL_ABS_BETA
    return bool(same_sign and not_near_neutral)


def anchors_for(path_csv: Path, family: str) -> list[dict]:
    if not path_csv.exists():
        return []
    df = pd.read_csv(path_csv)
    if "Beta_log" not in df.columns:
        return []
    # Use only numerically valid fits, matching exactly what the Step 8-9 screen
    # aggregates. Without this the anchors would be measured on a curve that
    # includes path points the screen excluded (Middlesex Direct at
    # rho_tilde >= 71.22: 9 DIVERGED_OUTSIDE_TRAINING_SUPPORT + 10
    # NUMERICALLY_UNSTABLE_RHO), leaving two frozen artifacts describing
    # different curves for the same jurisdiction.
    if "fit_status" in df.columns:
        df = df.loc[df["fit_status"].astype(str) == "OK"].copy()
    df = df.loc[np.isfinite(pd.to_numeric(df["Beta_log"], errors="coerce"))].copy()
    if not len(df):
        return []
    fold0 = df.loc[df["rho_tilde"] == 0.0] if family == "direct" else None
    baseline_betas = (
        fold0.groupby("fold")["Beta_log"].first().to_numpy() if fold0 is not None and len(fold0)
        else df.loc[df["rho_tilde"] == df["rho_tilde"].min()].groupby("fold")["Beta_log"].first().to_numpy()
    )
    stable = beta_stable(baseline_betas)
    curve = df.groupby("rho_tilde")["Beta_log"].mean().reset_index().sort_values("rho_tilde")
    beta0 = float(curve["Beta_log"].iloc[0])
    rows = []
    if not stable:
        rows.append({
            "status": "BASELINE_MECHANISM_NEAR_NEUTRAL", "beta0": beta0,
            "fold_betas": baseline_betas.tolist(),
        })
        return rows
    a_beta = 1.0 - (curve["Beta_log"].abs() / abs(beta0))
    rho_arr = curve["rho_tilde"].to_numpy()
    a_arr = a_beta.to_numpy()
    for target in ANCHORS:
        rho_hit = None
        for i in range(len(a_arr) - 1):
            lo, hi = min(a_arr[i], a_arr[i + 1]), max(a_arr[i], a_arr[i + 1])
            if lo - 1e-9 <= target <= hi + 1e-9 and abs(a_arr[i + 1] - a_arr[i]) > 1e-12:
                w = (target - a_arr[i]) / (a_arr[i + 1] - a_arr[i])
                rho_hit = float(rho_arr[i] + w * (rho_arr[i + 1] - rho_arr[i]))
                break
        rows.append({
            "status": "OK", "target_A_beta": target, "rho_tilde": rho_hit,
            "beta0": beta0, "attained": rho_hit is not None,
        })
    return rows


def main() -> int:
    all_rows = []
    for key in ALL_KEYS:
        for family in ("direct", "surrogate"):
            path_csv = ANALYSIS / "cv" / f"{key}_{family}_normalized_cv_path_summary.csv"
            for r in anchors_for(path_csv, family):
                all_rows.append({"county_key": key, "family": family, **r})
    df = pd.DataFrame(all_rows)
    ANALYSIS.joinpath("candidate_regions").mkdir(parents=True, exist_ok=True)
    df.to_csv(ANALYSIS / "candidate_regions" / "achieved_mechanism_anchors.csv", index=False)
    print(df.to_string(index=False) if len(df) else "no CV path data yet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
