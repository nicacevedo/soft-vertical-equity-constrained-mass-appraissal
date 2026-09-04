#!/usr/bin/env python3
"""Steps 8-9: candidate-region screen on log10(rho_tilde), reusing the
CCAO v2.1 engine (utils/rho_screening_v2.py) unchanged and read-only.

For each jurisdiction x family: aggregate the per-fold normalized CV path to
a CV-mean curve and 7 leave-one-fold-out (LOFO) mean curves, run the SAME
engine functions on each, and report full-sample activity/guardrail plus
LOFO stability. Interpretive-only signals (dCor, ratio shape) never move a
boundary, per protocol.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from utils.rho_screening_v2 import (  # noqa: E402
    BENEFIT_METRICS, PREDICTIVE_COST_METRICS, activity_onset, benefit_distance,
    classify_dcor, cluster_predictive_events, interpret_benefit, interpret_cost_deterioration,
    interpret_delta_nl, lofo_stability, log10_rho, median_log_spacing, predictive_cost,
    predictive_harm_event, select_pwl, surrogate_upper_guardrail,
)
from analysis.external_jurisdiction_benchmark_v1.scripts.v1_common import ANALYSIS, ALL_KEYS, write_json  # noqa: E402


def load_direct_path(key: str) -> pd.DataFrame | None:
    p = ANALYSIS / "cv" / f"{key}_direct_normalized_cv_path_summary.csv"
    return pd.read_csv(p) if p.exists() else None


def load_surrogate_path(key: str) -> pd.DataFrame | None:
    p = ANALYSIS / "cv" / f"{key}_surrogate_normalized_cv_path_summary.csv"
    return pd.read_csv(p) if p.exists() else None


def aggregate_curve(df: pd.DataFrame, metrics: list[str], exclude_fold: int | None = None) -> pd.DataFrame:
    """Mean of `metrics` over folds (all, or all-but-one for LOFO), by rho_tilde.
    Positive rho_tilde points only -- log10 requires strictly positive rho.

    A grid point is kept only if EVERY boundary-driving metric is finite in the
    fold-mean there (confirmed necessary for Middlesex Direct, where raw_rho =
    rho_tilde/Var_training(y) reaches ~250-550 at the top of the shared grid for
    this small-Var(y) jurisdiction). Both failure shapes must be excluded:

      * an all-NaN row (every fold NUMERICALLY_UNSTABLE_RHO), and
      * a partially-poisoned row, where some folds "succeeded" numerically but
        produced an absurd price scale -- observed at rho_tilde=71.22 as
        PRB~4e290, VEI~1e93, MAPE~3e290 with R2_price=NaN.

    Either shape corrupts the screen silently rather than loudly: the engine
    checks finiteness across a metric's whole curve, so one NaN deletes that
    metric from the screen entirely, and ~1e290 values overflow when squared in
    select_pwl's SSE. The visible result is an apparent NO_STABLE_CANDIDATE_REGION
    that is really a numerical artifact. Dropping only the unusable tail leaves
    the existing grid-censoring detection (endpoint index == last row of the curve
    actually used) to reflect the smaller usable grid for that jurisdiction; it
    never fabricates, extends, or interpolates anything.
    """
    sub = df.loc[df["rho_tilde"] > 0]
    if exclude_fold is not None:
        sub = sub.loc[sub["fold"] != exclude_fold]
    agg = sub.groupby("rho_tilde")[metrics].mean().reset_index().sort_values("rho_tilde")
    # dCor is excluded from the gate on purpose: the protocol makes it
    # interpretive-only, so it must never decide which grid points exist.
    gate = [m for m in metrics if m != "dCor_e_y"]
    if gate:
        finite = np.isfinite(agg[gate].to_numpy(dtype=float)).all(axis=1)
        agg = agg.loc[finite]
    return agg.reset_index(drop=True)


def benefit_events_for(curve: pd.DataFrame, x: np.ndarray, rho: np.ndarray) -> dict:
    events = {}
    for m in BENEFIT_METRICS:
        col = "Beta_log" if m == "Beta_log" else m
        if col not in curve.columns:
            continue
        y = benefit_distance(m, curve[col].to_numpy())
        if not np.all(np.isfinite(y)):
            continue
        selected = select_pwl(x, y)
        events[m] = interpret_benefit(selected, x, y, rho)
    return events


def direct_cost_events_for(curve: pd.DataFrame, x: np.ndarray, rho: np.ndarray) -> dict:
    events = {}
    rename = {"MAE": "MAE", "MAPE": "MAPE", "RMSE_log": "RMSE_log", "R2_price": "R2_price"}
    for m in PREDICTIVE_COST_METRICS:
        col = rename.get(m, m)
        if col not in curve.columns:
            continue
        v0 = float(curve[col].iloc[0])
        y = predictive_cost(m, curve[col].to_numpy(), v0)
        if not np.all(np.isfinite(y)):
            continue
        selected = select_pwl(x, y)
        events[m] = interpret_cost_deterioration(selected, x, y, rho)
    return events


def screen_direct(curve: pd.DataFrame) -> dict:
    rho = curve["rho_tilde"].to_numpy()
    x = log10_rho(rho)
    h = median_log_spacing(x)
    benefit_events = benefit_events_for(curve, x, rho)
    activity = activity_onset(rho, benefit_events)
    cost_events = direct_cost_events_for(curve, x, rho)
    guardrail = cluster_predictive_events(cost_events, x, rho, h, empty_status="DIRECT_GUARDRAIL_AMBIGUOUS")
    grid_censored_activity = activity["index"] is not None and activity["index"] == len(rho) - 1
    grid_censored_guardrail = guardrail.get("guardrail_index") is not None and guardrail["guardrail_index"] == len(rho) - 1
    return {
        "activity": activity, "guardrail": guardrail, "benefit_events": benefit_events,
        "cost_events": cost_events, "h": h, "n_grid_points": len(rho),
        "grid_censored_activity": bool(grid_censored_activity),
        "grid_censored_guardrail": bool(grid_censored_guardrail),
    }


def screen_surrogate(curve: pd.DataFrame) -> dict:
    rho = curve["rho_tilde"].to_numpy()
    x = log10_rho(rho)
    h = median_log_spacing(x)
    benefit_events = benefit_events_for(curve, x, rho)
    activity = activity_onset(rho, benefit_events)

    cost_events_det = direct_cost_events_for(curve, x, rho)
    harm_events = {}
    for m in PREDICTIVE_COST_METRICS:
        if m not in cost_events_det or m not in curve.columns:
            continue
        det = cost_events_det[m]
        v0 = float(curve[m].iloc[0])
        harm_events[m] = predictive_harm_event(rho, x, curve[m].to_numpy(), v0, det.get("index"), m)
    harm_cluster = cluster_predictive_events(
        {m: e for m, e in harm_events.items() if e.get("index") is not None}, x, rho, h,
        empty_status="SURROGATE_GUARDRAIL_AMBIGUOUS",
    )

    nl_event = None
    if "Delta_NL" in curve.columns and np.all(np.isfinite(curve["Delta_NL"])):
        selected_nl = select_pwl(x, curve["Delta_NL"].to_numpy())
        nl_event = interpret_delta_nl(selected_nl, x, curve["Delta_NL"].to_numpy(), rho)

    guardrail = surrogate_upper_guardrail(harm_cluster=harm_cluster, nl_event=nl_event)
    dcor_note = None
    if "dCor_e_y" in curve.columns and nl_event and nl_event.get("index") is not None:
        dcor_note = classify_dcor(x, curve["dCor_e_y"].to_numpy(), nl_event.get("index"))
    grid_censored_activity = activity["index"] is not None and activity["index"] == len(rho) - 1
    grid_censored_guardrail = guardrail.get("index_guardrail") is not None and guardrail["index_guardrail"] == len(rho) - 1
    return {
        "activity": activity, "guardrail": guardrail, "benefit_events": benefit_events,
        "harm_events": harm_events, "nl_event": nl_event, "dcor_interpretive_only": dcor_note,
        "h": h, "n_grid_points": len(rho),
        "grid_censored_activity": bool(grid_censored_activity),
        "grid_censored_guardrail": bool(grid_censored_guardrail),
    }


def endpoint_rho(rho: np.ndarray, index: int | None) -> float | None:
    return float(rho[int(index)]) if index is not None else None


def classify_candidate_status(
    activity_rho: float | None,
    guardrail_rho: float | None,
    activity_lofo_stable: bool,
    guardrail_lofo_stable: bool,
) -> str:
    """Status of the interval [activity, guardrail], not of the endpoints alone.

    Point-estimate endpoints are always preserved separately. A protocol-valid
    candidate region additionally requires a nonempty interval
    (activity <= guardrail) whose both endpoints are LOFO-stable.
    """
    if activity_rho is not None and guardrail_rho is not None:
        if activity_rho > guardrail_rho:
            return "UPPER_GUARDRAIL_PRECEDES_ACTIVITY"
        if not (activity_lofo_stable and guardrail_lofo_stable):
            return "NO_STABLE_CANDIDATE_REGION"
        return "CANDIDATE_REGION"
    if activity_rho is None and guardrail_rho is None:
        return "NO_STABLE_CANDIDATE_REGION"
    return "PARTIAL_ENDPOINT_ONLY"


def run_family(key: str, family: str, loader) -> dict | None:
    df = loader(key)
    if df is None or not len(df):
        return None
    metrics_needed = list(BENEFIT_METRICS) + list(PREDICTIVE_COST_METRICS) + ["Delta_NL", "dCor_e_y"]
    metrics_present = [m for m in metrics_needed if m in df.columns]
    full_curve = aggregate_curve(df, metrics_present)
    if len(full_curve) < 3:
        return {"county_key": key, "family": family, "status": "TOO_FEW_GRID_POINTS"}
    screen_fn = screen_direct if family == "direct" else screen_surrogate
    full = screen_fn(full_curve)
    rho = full_curve["rho_tilde"].to_numpy()

    folds = sorted(df["fold"].unique())
    lofo_activity_idx, lofo_guardrail_idx = [], []
    for f in folds:
        lofo_curve = aggregate_curve(df, metrics_present, exclude_fold=f)
        if len(lofo_curve) < 3:
            lofo_activity_idx.append(None); lofo_guardrail_idx.append(None)
            continue
        lofo_result = screen_fn(lofo_curve)
        lofo_activity_idx.append(lofo_result["activity"]["index"])
        gr_idx = lofo_result["guardrail"].get("guardrail_index") if family == "direct" else lofo_result["guardrail"].get("index_guardrail")
        lofo_guardrail_idx.append(gr_idx)

    activity_lofo = lofo_stability(full["activity"]["index"], lofo_activity_idx, log10_rho(rho), full["h"])
    guardrail_full_idx = full["guardrail"].get("guardrail_index") if family == "direct" else full["guardrail"].get("index_guardrail")
    guardrail_lofo = lofo_stability(guardrail_full_idx, lofo_guardrail_idx, log10_rho(rho), full["h"])

    activity_rho = endpoint_rho(rho, full["activity"]["index"])
    guardrail_rho = endpoint_rho(rho, guardrail_full_idx)
    status = classify_candidate_status(
        activity_rho, guardrail_rho,
        bool(activity_lofo.get("stable")), bool(guardrail_lofo.get("stable")),
    )

    return {
        "county_key": key, "family": family, "status": status,
        "n_folds": len(folds), "n_grid_points": full["n_grid_points"],
        "activity_rho_tilde": activity_rho, "guardrail_rho_tilde": guardrail_rho,
        "activity_log10_rho_tilde": None if activity_rho is None else float(np.log10(activity_rho)),
        "guardrail_log10_rho_tilde": None if guardrail_rho is None else float(np.log10(guardrail_rho)),
        "guardrail_binding_reason": (
            full["guardrail"].get("status") if family == "direct" else full["guardrail"].get("guardrail_driver")
        ),
        "activity_lofo": activity_lofo, "guardrail_lofo": guardrail_lofo,
        "lofo_activity_indices": lofo_activity_idx, "lofo_guardrail_indices": lofo_guardrail_idx,
        "grid_censored_activity": full["grid_censored_activity"],
        "grid_censored_guardrail": full["grid_censored_guardrail"],
        "dcor_interpretive_only": full.get("dcor_interpretive_only"),
    }


def main() -> int:
    rows = []
    for key in ALL_KEYS:
        for family, loader in (("direct", load_direct_path), ("surrogate", load_surrogate_path)):
            result = run_family(key, family, loader)
            if result is None:
                continue
            rows.append(result)
            ANALYSIS.joinpath("candidate_regions").mkdir(parents=True, exist_ok=True)
            write_json(ANALYSIS / "candidate_regions" / f"{key}_{family}_screen.json", result)
            print(json.dumps({k: result[k] for k in
                               ["county_key", "family", "status", "activity_rho_tilde", "guardrail_rho_tilde"]},
                              default=str), flush=True)

    flat_rows = []
    for r in rows:
        flat_rows.append({
            "county_key": r["county_key"], "family": r["family"], "status": r["status"],
            "activity_rho_tilde": r.get("activity_rho_tilde"), "guardrail_rho_tilde": r.get("guardrail_rho_tilde"),
            "guardrail_binding_reason": r.get("guardrail_binding_reason"),
            "activity_lofo_stable": r.get("activity_lofo", {}).get("stable"),
            "guardrail_lofo_stable": r.get("guardrail_lofo", {}).get("stable"),
            "grid_censored_activity": r.get("grid_censored_activity"),
            "grid_censored_guardrail": r.get("grid_censored_guardrail"),
        })
    pd.DataFrame(flat_rows).to_csv(ANALYSIS / "candidate_regions" / "candidate_regions.csv", index=False)

    lofo_rows = []
    for r in rows:
        for i, (a_idx, g_idx) in enumerate(zip(r.get("lofo_activity_indices", []), r.get("lofo_guardrail_indices", [])), start=1):
            lofo_rows.append({
                "county_key": r["county_key"], "family": r["family"], "lofo_fold_excluded": i,
                "activity_index": a_idx, "guardrail_index": g_idx,
            })
    pd.DataFrame(lofo_rows).to_csv(ANALYSIS / "candidate_regions" / "lofo_endpoints.csv", index=False)

    summary_rows = []
    for r in rows:
        summary_rows.append({
            "county_key": r["county_key"], "family": r["family"],
            "activity_n_in_window": r.get("activity_lofo", {}).get("n_in_window"),
            "activity_n_lofo": r.get("activity_lofo", {}).get("n_lofo"),
            "guardrail_n_in_window": r.get("guardrail_lofo", {}).get("n_in_window"),
            "guardrail_n_lofo": r.get("guardrail_lofo", {}).get("n_lofo"),
        })
    pd.DataFrame(summary_rows).to_csv(ANALYSIS / "candidate_regions" / "lofo_summary.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
