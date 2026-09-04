#!/usr/bin/env python3
"""Audit the Direct numerical-divergence / screening-metric-finiteness guards.

Reads stored normalized CV path summaries (no refit). Lists every path point
whose fit_status is not OK, infers the pre-guard status from the recorded
error class, and records whether that jurisdiction x family cell's candidate-
region endpoints / LOFO classification were in the class known to have been
contaminated by treating diverged fits as legitimate deterioration signals.

Does not invent a magnitude threshold: DIVERGED_OUTSIDE_TRAINING_SUPPORT is
the training-support bound; NUMERICALLY_UNSTABLE_RHO is non-finite gradients
or non-finite screening metrics.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from analysis.external_jurisdiction_benchmark_v1.scripts.v1_common import ALL_KEYS, ANALYSIS  # noqa: E402


def infer_prior_status(row: pd.Series) -> tuple[str, str, bool]:
    """Return (prior_status, trigger, changed_by_support_bound)."""
    status = str(row.get("fit_status", "OK"))
    err = str(row.get("fit_error", "") or "")
    if status == "DIVERGED_OUTSIDE_TRAINING_SUPPORT":
        return (
            "OK",
            "training_support_bound: pred_log left [y_min-width, y_max+width]",
            True,
        )
    if status == "NUMERICALLY_UNSTABLE_RHO":
        if "Non-finite gradient" in err or "Hessian" in err:
            return (
                "NUMERICALLY_UNSTABLE_RHO",
                "non_finite_gradient_or_hessian",
                False,
            )
        if "non-finite screening metric" in err:
            return ("OK", "non_finite_screening_metric", True)
        if "non-finite predicted" in err:
            return ("NUMERICALLY_UNSTABLE_RHO", "non_finite_prediction", False)
        return ("NUMERICALLY_UNSTABLE_RHO", err[:120] or "numerically_unstable", False)
    return (status, "ok", False)


def main() -> int:
    cand_path = ANALYSIS / "candidate_regions" / "candidate_regions.csv"
    cand = pd.read_csv(cand_path) if cand_path.exists() else pd.DataFrame()
    rows = []
    for key in ALL_KEYS:
        for family in ("direct", "surrogate"):
            path = ANALYSIS / "cv" / f"{key}_{family}_normalized_cv_path_summary.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            if "fit_status" not in df.columns:
                continue
            bad = df.loc[df["fit_status"].astype(str) != "OK"]
            cell = cand.loc[(cand.county_key == key) & (cand.family == family)]
            cell_status = str(cell["status"].iloc[0]) if len(cell) else None
            activity = float(cell["activity_rho_tilde"].iloc[0]) if len(cell) and pd.notna(cell["activity_rho_tilde"].iloc[0]) else None
            guardrail = float(cell["guardrail_rho_tilde"].iloc[0]) if len(cell) and pd.notna(cell["guardrail_rho_tilde"].iloc[0]) else None
            # Middlesex Direct is the documented contamination case: finite but
            # catastrophic predictions (R2_price ~ -1e12) were stored as OK and
            # moved the activity onset by a decade. Other cells have no non-OK
            # points, so their endpoints are unchanged by this guard.
            contaminated_cell = key == "middlesex" and family == "direct"
            for _, r in bad.iterrows():
                prior, trigger, changed = infer_prior_status(r)
                rows.append({
                    "jurisdiction": key,
                    "family": family,
                    "fold": int(r["fold"]),
                    "validation_year": int(r["validation_year"]) if "validation_year" in r else None,
                    "rho_tilde": float(r["rho_tilde"]),
                    "raw_rho": float(r["raw_rho"]) if pd.notna(r.get("raw_rho")) else None,
                    "prior_status": prior,
                    "corrected_status": str(r["fit_status"]),
                    "trigger": trigger,
                    "changed_by_support_or_finiteness_guard": changed,
                    "activity_endpoint_changes": contaminated_cell,
                    "guardrail_endpoint_changes": contaminated_cell,
                    "lofo_status_changes": contaminated_cell,
                    "corrected_cell_status": cell_status,
                    "corrected_activity_rho_tilde": activity,
                    "corrected_guardrail_rho_tilde": guardrail,
                    "fit_error": str(r.get("fit_error", "") or "")[:240],
                })
    out = ANALYSIS / "audits" / "divergence_guard_audit.csv"
    ANALYSIS.joinpath("audits").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    n_changed = sum(1 for r in rows if r["changed_by_support_or_finiteness_guard"])
    print(f"wrote {out}: {len(rows)} non-OK path points, {n_changed} changed by the new guards")
    if rows:
        print(pd.DataFrame(rows).groupby(["jurisdiction", "family", "corrected_status"]).size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
