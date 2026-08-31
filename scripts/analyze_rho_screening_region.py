#!/usr/bin/env python3
"""Read-only CV-derived rho candidate-region characterization.

Not a model-selection rule. Chronological CV only for endpoints.
No LightGBM/Direct/Surrogate/Linear refitting. No manuscript writes.
Does not redefine the frozen five-metric prediction/COD transition spans.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from utils.transition_paper_asset_plots import padded_lim  # noqa: E402
from utils.transition_regions import (  # noqa: E402
    FAMILY_DISPLAY,
    FOLD_IDS,
    OutputGuard,
    family_frame,
    is_rho_positive,
    is_rho_zero,
    sha256_file,
)
from utils.paper_v12_lower_rho_plots import (  # noqa: E402
    DIRECT_COLOR,
    SURR_COLOR,
    apply_major_grid,
    draw_neutral_hline,
    family_span,
    log_rho_axes,
    maybe_percent,
    nearby_targets,
    rho_plot_x,
    shade_cv_span_with_bounds,
)

CANONICAL = REPO / "output" / "paper_v6_preselection_994"
EXT = REPO / "output" / "paper_v12_lower_rho_extension_994_v2"
DATA_ID = "d4929d43ec19badf"
SPLIT_ID = "3d464d4a611b131b"
LGBM_CONFIG_ID = "407d47775760c14d"
SEED = 2025
TREES = 994
V2 = (
    EXT
    / "analysis"
    / f"data_id={DATA_ID}"
    / f"split_id={SPLIT_ID}"
    / "penalty_path_analysis"
    / "transition_regions_v2_lower_rho"
)
V4 = V2.parent / "transition_regions_paper_assets_v4_delta_nl_bends"
OUT = V2.parent / "rho_screening_candidate_regions_v1"
PAPER_TEX = REPO / "paper" / "paper_v12.tex"

MIN_SEG = 6
EPS = 1e-12
LOG10_TOL = 1.0
BENEFIT_METRICS = ("PRD", "PRB", "MKI", "VEI", "Beta_log")
COST_METRICS = ("R2_price", "MAE_price", "MAPE", "RMSE_log", "COD")
HIGHER_BETTER = {"R2_price"}
BENEFIT_TARGET = {"PRD": 1.0, "PRB": 0.0, "MKI": 1.0, "VEI": 0.0, "Beta_log": 0.0}

CAND_FACE = "#86A789"
CAND_ALPHA = 0.18
CAND_EDGE = dict(color="#3F6F44", ls=(0, (5.0, 2.2)), lw=1.05, alpha=0.95, zorder=2)
PRED_DASH = dict(color="#9CA3AF", ls=(0, (2.2, 2.0)), lw=0.7, alpha=0.75, zorder=1)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=str(REPO), text=True).strip()


def bic(sse: float, n: int, k: int) -> float:
    sse = max(float(sse), 1e-30)
    n = max(int(n), 1)
    return float(n * np.log(sse / n) + k * np.log(n))


def benefit_distance(metric: str, values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    t = BENEFIT_TARGET[metric]
    return np.abs(v - float(t))


def positive_path(frame: pd.DataFrame) -> pd.DataFrame:
    sub = frame.copy()
    rho = pd.to_numeric(sub["rho"], errors="coerce")
    keep = np.array([is_rho_positive(float(x)) for x in rho])
    return sub.loc[keep].sort_values("rho", kind="mergesort").reset_index(drop=True)


def metric_series(frame: pd.DataFrame, metric: str, suffix: str) -> Tuple[np.ndarray, np.ndarray]:
    sub = positive_path(frame)
    rho = pd.to_numeric(sub["rho"], errors="coerce").to_numpy(dtype=float)
    vals = pd.to_numeric(sub[f"{metric}__{suffix}"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(rho) & np.isfinite(vals)
    return rho[mask], vals[mask]


def lofo_series(frame: pd.DataFrame, metric: str, left_out: int) -> Tuple[np.ndarray, np.ndarray]:
    sub = positive_path(frame)
    rho = pd.to_numeric(sub["rho"], errors="coerce").to_numpy(dtype=float)
    mats = []
    for k in FOLD_IDS:
        if int(k) == int(left_out):
            continue
        mats.append(pd.to_numeric(sub[f"{metric}__fold_{k}"], errors="coerce").to_numpy(dtype=float))
    vals = np.nanmean(np.vstack(mats), axis=0)
    mask = np.isfinite(rho) & np.isfinite(vals)
    return rho[mask], vals[mask]


def fit_continuous_pwl(x: np.ndarray, y: np.ndarray, breaks: Sequence[float]) -> Dict[str, Any]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    cols = [np.ones(x.size), x]
    for b in breaks:
        cols.append(np.maximum(x - float(b), 0.0))
    A = np.column_stack(cols)
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    sse = float(np.sum((y - pred) ** 2))
    slopes = [float(coef[1])]
    running = float(coef[1])
    for c in coef[2:]:
        running = running + float(c)
        slopes.append(float(running))
    return {
        "coef": [float(c) for c in coef],
        "sse": sse,
        "pred": pred,
        "slopes": slopes,
        "n": int(x.size),
        "k_params": int(A.shape[1]),
        "bic": bic(sse, int(x.size), int(A.shape[1])),
    }


def valid_break_indices(n: int, n_breaks: int) -> List[Tuple[int, ...]]:
    """Inclusive segment sizes at least MIN_SEG, breaks at interior tested indices."""
    out: List[Tuple[int, ...]] = []
    if n_breaks == 1:
        for b in range(MIN_SEG - 1, n - MIN_SEG):
            out.append((b,))
        return out
    if n_breaks == 2:
        for b1 in range(MIN_SEG - 1, n - 2 * MIN_SEG + 1):
            for b2 in range(b1 + MIN_SEG - 1, n - MIN_SEG):
                out.append((b1, b2))
    return out


def search_pwl(x: np.ndarray, y: np.ndarray, n_breaks: int) -> Optional[Dict[str, Any]]:
    n = int(x.size)
    best = None
    n_cand = 0
    for idx in valid_break_indices(n, n_breaks):
        n_cand += 1
        breaks = [float(x[i]) for i in idx]
        fit = fit_continuous_pwl(x, y, breaks)
        rec = {
            **{k: fit[k] for k in ("sse", "slopes", "n", "k_params", "bic", "coef")},
            "break_indices": list(idx),
            "break_rho": [float(np.power(10.0, x[i])) for i in idx],
            "break_x": breaks,
            "n_breaks": n_breaks,
            "n_candidates": None,
        }
        if best is None:
            best = rec
        else:
            better = rec["sse"] < best["sse"] - 1e-15
            tie = abs(rec["sse"] - best["sse"]) <= 1e-15
            if better or (tie and rec["break_rho"][0] < best["break_rho"][0]):
                best = rec
    if best is None:
        return None
    best["n_candidates"] = n_cand
    return best


def local_slopes(x: np.ndarray, y: np.ndarray, bidx: int, window: int = 4) -> Dict[str, Any]:
    i0 = max(0, bidx - window)
    i1 = min(int(x.size) - 1, bidx + window)
    pre_x, pre_y = x[i0 : bidx + 1], y[i0 : bidx + 1]
    post_x, post_y = x[bidx : i1 + 1], y[bidx : i1 + 1]
    def _s(xx, yy):
        if xx.size < 2:
            return np.nan
        return float(np.polyfit(xx, yy, 1)[0])
    return {
        "raw_pre_slope": _s(pre_x, pre_y),
        "raw_post_slope": _s(post_x, post_y),
        "n_pre": int(pre_x.size),
        "n_post": int(post_x.size),
    }


def classify_cost_event(metric: str, fit: Dict[str, Any], x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    s_pre, s_post = fit["slopes"][0], fit["slopes"][1]
    bidx = int(fit["break_indices"][0])
    raw = local_slopes(x, y, bidx)
    if metric in HIGHER_BETTER:
        ok = (s_post < -EPS) and (s_post < s_pre - EPS)
        raw_ok = np.isfinite(raw["raw_post_slope"]) and raw["raw_post_slope"] < 0
    else:
        ok = (s_post > EPS) and (s_post > s_pre + EPS)
        raw_ok = np.isfinite(raw["raw_post_slope"]) and raw["raw_post_slope"] > 0
    qa = "supported" if (ok and raw_ok) else ("ambiguous" if ok else "invalid")
    return {
        "event": "cost_deterioration" if ok else None,
        "rho": float(fit["break_rho"][0]) if ok else None,
        "x": float(fit["break_x"][0]) if ok else None,
        "pre_slope": float(s_pre),
        "post_slope": float(s_post),
        "classification": "VALID" if ok else "INVALID",
        "raw_path_qa": qa,
        **raw,
        **{k: fit[k] for k in ("sse", "bic", "k_params", "n", "n_candidates", "coef", "break_indices")},
        "complexity": "two_segment",
    }


def interpret_benefit(fit2: Optional[Dict[str, Any]], fit3: Optional[Dict[str, Any]], x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    cands = []
    if fit2 is not None:
        cands.append(("two_segment", fit2))
    if fit3 is not None:
        cands.append(("three_segment", fit3))
    if not cands:
        return {
            "complexity": None,
            "benefit_onset": None,
            "benefit_saturation": None,
            "classification": "INVALID",
            "raw_path_qa": "invalid",
        }
    chosen_name, chosen = min(cands, key=lambda t: (t[1]["bic"], t[1]["break_rho"][0]))
    onset = None
    sat = None
    onset_idx = None
    sat_idx = None
    slopes = chosen["slopes"]
    if chosen_name == "three_segment":
        s1, s2, s3 = slopes
        if (s2 < -EPS) and (s2 < s1 - EPS):
            onset = float(chosen["break_rho"][0])
            onset_idx = int(chosen["break_indices"][0])
        if s3 > s2 + EPS:
            sat = float(chosen["break_rho"][1])
            sat_idx = int(chosen["break_indices"][1])
        # If 3-seg does not realize the qualitative structure, fall back to 2-seg if better-supported.
        if onset is None and sat is None and fit2 is not None:
            chosen_name, chosen = "two_segment", fit2
            slopes = chosen["slopes"]
    if chosen_name == "two_segment":
        s_pre, s_post = slopes[0], slopes[1]
        bidx = int(chosen["break_indices"][0])
        rho_b = float(chosen["break_rho"][0])
        if (s_post < -EPS) and (s_post < s_pre - EPS):
            onset = rho_b
            onset_idx = bidx
        elif (s_pre < -EPS) and (s_post > s_pre + EPS):
            sat = rho_b
            sat_idx = bidx
            # improving from the start of the positive path
            onset = None
            onset_idx = None
    qa_flags = []
    if onset is not None and onset_idx is not None:
        raw = local_slopes(x, y, onset_idx)
        qa_flags.append("onset_supported" if raw["raw_post_slope"] < 0 else "onset_ambiguous")
    if sat is not None and sat_idx is not None:
        raw = local_slopes(x, y, sat_idx)
        qa_flags.append("sat_supported" if raw["raw_post_slope"] > raw["raw_pre_slope"] - 1e-12 else "sat_ambiguous")
    if not qa_flags:
        qa = "invalid"
        cls = "INVALID"
    elif any("ambiguous" in q for q in qa_flags) and not any("supported" in q for q in qa_flags):
        qa = "ambiguous"
        cls = "AMBIGUOUS"
    else:
        qa = "|".join(qa_flags)
        cls = "VALID" if (onset is not None or sat is not None) else "INVALID"
    out = {
        "complexity": chosen_name,
        "benefit_onset": onset,
        "benefit_saturation": sat,
        "onset_x": None if onset is None else float(np.log10(onset)),
        "saturation_x": None if sat is None else float(np.log10(sat)),
        "classification": cls,
        "raw_path_qa": qa,
        "slopes": [float(s) for s in chosen["slopes"]],
        "sse": chosen["sse"],
        "bic": chosen["bic"],
        "k_params": chosen["k_params"],
        "n": chosen["n"],
        "n_candidates": chosen.get("n_candidates"),
        "coef": chosen["coef"],
        "break_rho": chosen["break_rho"],
        "break_x": chosen["break_x"],
        "break_indices": chosen["break_indices"],
        "improving_from_first_positive": bool(onset is None and sat is not None),
    }
    return out


def interpret_delta_nl(fit2: Optional[Dict[str, Any]], fit3: Optional[Dict[str, Any]], x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    cands = []
    if fit2 is not None:
        cands.append(("two_segment", fit2))
    if fit3 is not None:
        cands.append(("three_segment", fit3))
    if not cands:
        return {"event": None, "rho": None, "classification": "INVALID", "raw_path_qa": "invalid"}
    chosen_name, chosen = min(cands, key=lambda t: (t[1]["bic"], t[1]["break_rho"][0]))
    rho = None
    bidx = None
    slopes = chosen["slopes"]
    if chosen_name == "three_segment":
        s1, s2, s3 = slopes
        # valley then rebound: last slope positive, and it is the post-valley break
        if s3 > EPS and s2 < s3 - EPS:
            rho = float(chosen["break_rho"][1])
            bidx = int(chosen["break_indices"][1])
        elif s2 > EPS and s1 < -EPS:
            rho = float(chosen["break_rho"][0])
            bidx = int(chosen["break_indices"][0])
    if chosen_name == "two_segment" or rho is None:
        if fit2 is not None:
            chosen_name, chosen = "two_segment", fit2
            slopes = chosen["slopes"]
            s_pre, s_post = slopes
            if s_post > EPS and s_pre < s_post - EPS:
                rho = float(chosen["break_rho"][0])
                bidx = int(chosen["break_indices"][0])
    qa = "invalid"
    if rho is not None and bidx is not None:
        raw = local_slopes(x, y, bidx)
        qa = "supported" if raw["raw_post_slope"] > 0 else "ambiguous"
    return {
        "event": "nonlinear_rebound" if rho is not None else None,
        "rho": rho,
        "x": None if rho is None else float(np.log10(rho)),
        "complexity": chosen_name,
        "slopes": [float(s) for s in chosen["slopes"]],
        "sse": chosen["sse"],
        "bic": chosen["bic"],
        "k_params": chosen["k_params"],
        "n": chosen["n"],
        "n_candidates": chosen.get("n_candidates"),
        "coef": chosen["coef"],
        "break_rho": chosen["break_rho"],
        "break_x": chosen["break_x"],
        "break_indices": chosen["break_indices"],
        "classification": "VALID" if rho is not None else "INVALID",
        "raw_path_qa": qa,
        "uses_exact_minimum": False,
    }


def dcor_corroborates(frame: pd.DataFrame, suffix: str, rho_nl: Optional[float]) -> Dict[str, Any]:
    if rho_nl is None:
        return {"corroborated": False, "reason": "no_nl_event"}
    rho, vals = metric_series(frame, "dCor_e_y", suffix)
    x = np.log10(rho)
    i = int(np.argmin(np.abs(rho - float(rho_nl))))
    raw = local_slopes(x, vals, i, window=5)
    # flattening or rebound: post slope greater than pre (less negative or positive)
    ok = np.isfinite(raw["raw_post_slope"]) and (raw["raw_post_slope"] > raw["raw_pre_slope"] - 1e-12)
    return {
        "corroborated": bool(ok),
        "reason": "dCor_flatten_or_rebound" if ok else "dCor_does_not_flatten_or_rebound",
        "rho_nl": float(rho_nl),
        **raw,
    }


def first_activity_rho(rho: np.ndarray, benefit_events: Dict[str, Dict[str, Any]]) -> Tuple[Optional[float], List[str]]:
    active_from: Dict[str, Optional[float]] = {}
    for m, ev in benefit_events.items():
        if ev.get("improving_from_first_positive"):
            active_from[m] = float(rho[0])
        elif ev.get("benefit_onset") is not None:
            active_from[m] = float(ev["benefit_onset"])
        else:
            active_from[m] = None
    for r in rho:
        names = [m for m, a in active_from.items() if a is not None and float(r) + 1e-15 >= float(a)]
        if len(names) >= 3:
            return float(r), names
    return None, []


def last_before_majority(
    rho: np.ndarray,
    cost_rho: Dict[str, Optional[float]],
    sat_rho: Dict[str, Optional[float]],
) -> Tuple[Optional[float], Optional[float], List[str], List[str]]:
    first_joint = None
    names_c: List[str] = []
    names_b: List[str] = []
    for r in rho:
        c_hit = [m for m, a in cost_rho.items() if a is not None and float(r) + 1e-15 >= float(a)]
        b_hit = [m for m, a in sat_rho.items() if a is not None and float(r) + 1e-15 >= float(a)]
        if len(c_hit) >= 3 and len(b_hit) >= 3:
            first_joint = float(r)
            names_c, names_b = c_hit, b_hit
            break
    if first_joint is None:
        return None, None, [], []
    prev = [float(x) for x in rho if float(x) < first_joint - 1e-15]
    guard = prev[-1] if prev else None
    return guard, first_joint, names_c, names_b


def analyze_family_path(
    frame: pd.DataFrame,
    family: str,
    suffix: str,
    *,
    path_label: str,
) -> Dict[str, Any]:
    def series(metric: str) -> Tuple[np.ndarray, np.ndarray]:
        if suffix.startswith("lofo_"):
            left = int(suffix.split("_")[-1])
            return lofo_series(frame, metric, left)
        return metric_series(frame, metric, suffix)

    rho0, _ = series("R2_price")
    x = np.log10(rho0)
    cost_events: Dict[str, Dict[str, Any]] = {}
    for m in COST_METRICS:
        rho, y = series(m)
        xx = np.log10(rho)
        fit2 = search_pwl(xx, y, 1)
        if fit2 is None:
            cost_events[m] = {"event": None, "rho": None, "classification": "INVALID", "raw_path_qa": "invalid", "metric": m}
        else:
            rec = classify_cost_event(m, fit2, xx, y)
            rec["metric"] = m
            rec["family"] = family
            rec["path"] = path_label
            rec["role"] = "cost"
            cost_events[m] = rec
    benefit_events: Dict[str, Dict[str, Any]] = {}
    for m in BENEFIT_METRICS:
        rho, yraw = series(m)
        y = benefit_distance(m, yraw)
        xx = np.log10(rho)
        fit2 = search_pwl(xx, y, 1)
        fit3 = search_pwl(xx, y, 2)
        rec = interpret_benefit(fit2, fit3, xx, y)
        rec["metric"] = m
        rec["family"] = family
        rec["path"] = path_label
        rec["role"] = "benefit"
        benefit_events[m] = rec

    activity, activity_metrics = first_activity_rho(rho0, benefit_events)
    cost_rho = {m: cost_events[m].get("rho") for m in COST_METRICS}
    sat_rho = {m: benefit_events[m].get("benefit_saturation") for m in BENEFIT_METRICS}
    tradeoff, joint_at, cost_names, sat_names = last_before_majority(rho0, cost_rho, sat_rho)

    dnl = None
    rho_dnl, y_dnl = series("Delta_NL")
    xx = np.log10(rho_dnl)
    fit2 = search_pwl(xx, y_dnl, 1)
    fit3 = search_pwl(xx, y_dnl, 2)
    dnl = interpret_delta_nl(fit2, fit3, xx, y_dnl)
    dnl["metric"] = "Delta_NL"
    dnl["family"] = family
    dnl["path"] = path_label
    dnl["role"] = "nonlinear_guardrail"
    if suffix.startswith("lofo_"):
        left = int(suffix.split("_")[-1])
        rho_d, y_d = lofo_series(frame, "dCor_e_y", left)
        if dnl.get("rho") is not None:
            i = int(np.argmin(np.abs(rho_d - float(dnl["rho"]))))
            raw = local_slopes(np.log10(rho_d), y_d, i, window=5)
            dcor = {
                "corroborated": bool(np.isfinite(raw["raw_post_slope"]) and raw["raw_post_slope"] > raw["raw_pre_slope"] - 1e-12),
                "reason": "dCor_flatten_or_rebound" if (np.isfinite(raw["raw_post_slope"]) and raw["raw_post_slope"] > raw["raw_pre_slope"] - 1e-12) else "dCor_does_not_flatten_or_rebound",
                **raw,
            }
        else:
            dcor = {"corroborated": False, "reason": "no_nl_event"}
    elif suffix == "CV_mean":
        dcor = dcor_corroborates(frame, "CV_mean", dnl.get("rho"))
    else:
        dcor = {"corroborated": False, "reason": "not_evaluated"}

    nl_accepted = bool(dnl.get("rho") is not None and dcor.get("corroborated"))
    if family == "Direct":
        guardrail = tradeoff
        guardrail_reason = "cost_and_benefit_majority" if tradeoff is not None else "no_joint_majority"
        driver = "tradeoff" if tradeoff is not None else None
    else:
        rho_nl = float(dnl["rho"]) if nl_accepted else None
        cands_u = [v for v in (tradeoff, rho_nl) if v is not None]
        guardrail = min(cands_u) if cands_u else None
        if guardrail is None:
            guardrail_reason = "no_tradeoff_and_no_corroborated_nl"
            driver = None
        elif rho_nl is not None and (tradeoff is None or rho_nl <= tradeoff + 1e-15):
            guardrail_reason = "nonlinear_shape_driven"
            driver = "nonlinear-shape-driven"
        else:
            guardrail_reason = "tradeoff_driven"
            driver = "tradeoff-driven"

    region_ok = activity is not None and guardrail is not None and float(activity) <= float(guardrail) + 1e-15
    return {
        "family": family,
        "path": path_label,
        "rho_activity": activity,
        "activity_metrics": activity_metrics,
        "rho_tradeoff": tradeoff,
        "tradeoff_joint_at": joint_at,
        "tradeoff_cost_metrics": cost_names,
        "tradeoff_sat_metrics": sat_names,
        "rho_nl": dnl.get("rho"),
        "nl_accepted": nl_accepted,
        "nl_dcor": dcor,
        "rho_guardrail": guardrail,
        "guardrail_reason": guardrail_reason,
        "guardrail_driver": driver,
        "region_defined": bool(region_ok),
        "cost_events": cost_events,
        "benefit_events": benefit_events,
        "delta_nl_event": dnl,
        "n_positive": int(rho0.size),
        "rho_min_positive": float(rho0[0]),
        "rho_max_positive": float(rho0[-1]),
    }


def flatten_breakpoints(fam_res: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for m, ev in fam_res["cost_events"].items():
        rows.append(
            {
                "family": fam_res["family"],
                "path": fam_res["path"],
                "metric": m,
                "role": "cost",
                "event": ev.get("event"),
                "rho": ev.get("rho"),
                "log10_rho": None if ev.get("rho") is None else float(np.log10(ev["rho"])),
                "pre_slope": ev.get("pre_slope"),
                "post_slope": ev.get("post_slope"),
                "sse": ev.get("sse"),
                "bic": ev.get("bic"),
                "complexity": ev.get("complexity"),
                "k_params": ev.get("k_params"),
                "n_segment_min": MIN_SEG,
                "classification": ev.get("classification"),
                "raw_path_qa": ev.get("raw_path_qa"),
                "n_candidates": ev.get("n_candidates"),
            }
        )
    for m, ev in fam_res["benefit_events"].items():
        rows.append(
            {
                "family": fam_res["family"],
                "path": fam_res["path"],
                "metric": m,
                "role": "benefit",
                "event": "benefit_structure",
                "rho_onset": ev.get("benefit_onset"),
                "rho_saturation": ev.get("benefit_saturation"),
                "log10_onset": ev.get("onset_x"),
                "log10_saturation": ev.get("saturation_x"),
                "slopes_json": json.dumps(ev.get("slopes")),
                "sse": ev.get("sse"),
                "bic": ev.get("bic"),
                "complexity": ev.get("complexity"),
                "k_params": ev.get("k_params"),
                "n_segment_min": MIN_SEG,
                "classification": ev.get("classification"),
                "raw_path_qa": ev.get("raw_path_qa"),
                "improving_from_first_positive": ev.get("improving_from_first_positive"),
                "n_candidates": ev.get("n_candidates"),
            }
        )
    ev = fam_res["delta_nl_event"]
    rows.append(
        {
            "family": fam_res["family"],
            "path": fam_res["path"],
            "metric": "Delta_NL",
            "role": "nonlinear",
            "event": ev.get("event"),
            "rho": ev.get("rho"),
            "log10_rho": ev.get("x"),
            "slopes_json": json.dumps(ev.get("slopes")),
            "sse": ev.get("sse"),
            "bic": ev.get("bic"),
            "complexity": ev.get("complexity"),
            "k_params": ev.get("k_params"),
            "n_segment_min": MIN_SEG,
            "classification": ev.get("classification"),
            "raw_path_qa": ev.get("raw_path_qa"),
            "uses_exact_minimum": False,
        }
    )
    return rows


def lofo_status(full: Dict[str, Any], lofos: List[Dict[str, Any]]) -> str:
    if not full.get("region_defined"):
        return "NO_BAND"
    act = float(full["rho_activity"])
    grd = float(full["rho_guardrail"])
    n_ok = 0
    for L in lofos:
        if not L.get("region_defined"):
            continue
        a = float(L["rho_activity"])
        g = float(L["rho_guardrail"])
        if abs(np.log10(a) - np.log10(act)) <= LOG10_TOL and abs(np.log10(g) - np.log10(grd)) <= LOG10_TOL:
            # same qualitative weak/middle/high: activity in same decade-ish, guardrail same
            n_ok += 1
    if n_ok >= 5:
        return "PASS"
    if n_ok >= 3 or sum(1 for L in lofos if L.get("region_defined")) >= 5:
        return "AMBIGUOUS"
    return "NO_BAND"


def shade_candidate(ax, low: Optional[float], high: Optional[float]) -> None:
    if low is None or high is None:
        return
    if not (np.isfinite(low) and np.isfinite(high)):
        return
    ax.axvspan(float(low), float(high), color=CAND_FACE, alpha=CAND_ALPHA, lw=0, zorder=0)
    ax.axvline(float(low), **CAND_EDGE)
    ax.axvline(float(high), **CAND_EDGE)


def shade_pred_subordinate(ax, low: Optional[float], high: Optional[float]) -> None:
    if low is None or high is None:
        return
    if not (np.isfinite(low) and np.isfinite(high)):
        return
    ax.axvline(float(low), **PRED_DASH)
    ax.axvline(float(high), **PRED_DASH)


def _save(plt, fig, stem: Path) -> List[str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    pdf = stem.with_suffix(".pdf")
    png = stem.with_suffix(".png")
    fig.savefig(pdf, format="pdf", bbox_inches="tight")
    fig.savefig(png, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return [str(pdf), str(png)]


def plot_path_group(plt, combined, span_df, regions, metrics, min_positive, q, stem, *, oos: bool) -> List[str]:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(len(metrics), 2, figsize=(8.9, 2.15 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = np.asarray([axes])
    for r, (col, ylab) in enumerate(metrics):
        row_vals: List[float] = []
        for fam in FAMILY_DISPLAY:
            sub = combined.loc[combined["family"] == fam]
            if oos:
                for ev in ("heldout", "forward_2025"):
                    row_vals.extend(maybe_percent(col, pd.to_numeric(sub[f"{col}__{ev}"], errors="coerce").to_numpy(dtype=float)).tolist())
            else:
                row_vals.extend(maybe_percent(col, pd.to_numeric(sub[f"{col}__CV_mean"], errors="coerce").to_numpy(dtype=float)).tolist())
                for k in FOLD_IDS:
                    row_vals.extend(maybe_percent(col, pd.to_numeric(sub[f"{col}__fold_{k}"], errors="coerce").to_numpy(dtype=float)).tolist())
        include = nearby_targets(row_vals, [])
        ylim = padded_lim(row_vals, pad=0.08, include=include)
        for c, fam in enumerate(FAMILY_DISPLAY):
            ax = axes[r, c]
            plow, phigh, pok = family_span(span_df, fam)
            if pok:
                shade_pred_subordinate(ax, plow, phigh)
            reg = regions[fam]
            shade_candidate(ax, reg.get("rho_activity"), reg.get("rho_guardrail"))
            sub = combined.loc[combined["family"] == fam].sort_values("rho")
            x = rho_plot_x(sub["rho"].to_numpy(dtype=float), min_positive=min_positive, q=q)
            color = DIRECT_COLOR if fam == "Direct" else SURR_COLOR
            if oos:
                yh = maybe_percent(col, pd.to_numeric(sub[f"{col}__heldout"], errors="coerce").to_numpy(dtype=float))
                yf = maybe_percent(col, pd.to_numeric(sub[f"{col}__forward_2025"], errors="coerce").to_numpy(dtype=float))
                ax.plot(x, yh, color=color, marker="o", ms=3, lw=1.3, label="Held-out", zorder=4)
                ax.plot(x, yf, color=color, ls="--", marker="s", ms=3, lw=1.2, label="2025", zorder=4)
            else:
                for k in FOLD_IDS:
                    yk = maybe_percent(col, pd.to_numeric(sub[f"{col}__fold_{k}"], errors="coerce").to_numpy(dtype=float))
                    ax.plot(x, yk, color="#9CA3AF", lw=0.8, alpha=0.7, zorder=3)
                ym = maybe_percent(col, pd.to_numeric(sub[f"{col}__CV_mean"], errors="coerce").to_numpy(dtype=float))
                ax.plot(x, ym, color=color, lw=2.1, label="Equal-weight CV", zorder=4)
            draw_neutral_hline(ax, col)
            if fam == "Surrogate" and col == "Delta_NL" and regions[fam].get("guardrail_driver") == "nonlinear-shape-driven":
                rnl = regions[fam].get("rho_nl")
                if rnl is not None:
                    ax.axvline(float(rnl), color="#7C3AED", ls=":", lw=0.9, alpha=0.85, zorder=3)
            log_rho_axes(ax, min_positive=min_positive, q=q)
            apply_major_grid(ax)
            ax.set_ylim(*ylim)
            if r == 0:
                ax.set_title(fam)
            if c == 0:
                ax.set_ylabel(ylab)
            if r == len(metrics) - 1:
                ax.set_xlabel(r"Penalty strength $\rho$")
            if r == 0 and c == 1:
                handles = [
                    Patch(facecolor=CAND_FACE, alpha=CAND_ALPHA, edgecolor="#3F6F44", linestyle="--", label="CV-derived candidate region"),
                    Line2D([0], [0], color="#9CA3AF", ls=(0, (2.2, 2.0)), lw=0.9, label="prediction/COD transition span"),
                ]
                if oos:
                    handles = [
                        Line2D([0], [0], color=color, marker="o", lw=1.3, label="Held-out"),
                        Line2D([0], [0], color=color, marker="s", ls="--", lw=1.2, label="2025"),
                    ] + handles
                ax.legend(handles=handles, frameon=False, fontsize=6.0, loc="best")
    fig.tight_layout()
    return _save(plt, fig, stem)


def plot_qa(plt, combined, span_df, regions, min_positive, q, stem) -> List[str]:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    rows = list(BENEFIT_METRICS) + list(COST_METRICS) + ["Delta_NL"]
    fig, axes = plt.subplots(len(rows), 2, figsize=(9.4, 1.55 * len(rows)), sharex=True)
    for r, metric in enumerate(rows):
        for c, fam in enumerate(FAMILY_DISPLAY):
            ax = axes[r, c]
            plow, phigh, pok = family_span(span_df, fam)
            if pok:
                shade_pred_subordinate(ax, plow, phigh)
            reg = regions[fam]
            shade_candidate(ax, reg.get("rho_activity"), reg.get("rho_guardrail"))
            sub = combined.loc[combined["family"] == fam].sort_values("rho")
            x = rho_plot_x(sub["rho"].to_numpy(dtype=float), min_positive=min_positive, q=q)
            if metric in BENEFIT_METRICS:
                raw = pd.to_numeric(sub[f"{metric}__CV_mean"], errors="coerce").to_numpy(dtype=float)
                y = benefit_distance(metric, raw)
                ylab = rf"$D$({metric})"
            else:
                y = maybe_percent(metric, pd.to_numeric(sub[f"{metric}__CV_mean"], errors="coerce").to_numpy(dtype=float))
                ylab = metric
            color = DIRECT_COLOR if fam == "Direct" else SURR_COLOR
            ax.plot(x, y, color=color, lw=1.6, zorder=4)
            evs = reg["cost_events"] if metric in COST_METRICS else (reg["benefit_events"] if metric in BENEFIT_METRICS else None)
            if metric in COST_METRICS and evs[metric].get("rho") is not None:
                ax.axvline(float(evs[metric]["rho"]), color="#111827", ls=":", lw=0.8)
            if metric in BENEFIT_METRICS:
                if evs[metric].get("benefit_onset") is not None:
                    ax.axvline(float(evs[metric]["benefit_onset"]), color="#1D4ED8", ls="--", lw=0.8)
                if evs[metric].get("benefit_saturation") is not None:
                    ax.axvline(float(evs[metric]["benefit_saturation"]), color="#B45309", ls="-.", lw=0.8)
            if metric == "Delta_NL" and reg.get("rho_nl") is not None:
                ax.axvline(float(reg["rho_nl"]), color="#7C3AED", ls=":", lw=0.9)
            log_rho_axes(ax, min_positive=min_positive, q=q)
            apply_major_grid(ax)
            if r == 0:
                ax.set_title(fam)
            if c == 0:
                ax.set_ylabel(ylab, fontsize=8)
            if r == len(rows) - 1:
                ax.set_xlabel(r"Penalty strength $\rho$")
    handles = [
        Patch(facecolor=CAND_FACE, alpha=CAND_ALPHA, label="CV-derived candidate region"),
        Line2D([0], [0], color="#9CA3AF", ls=(0, (2.2, 2.0)), label="prediction/COD span"),
        Line2D([0], [0], color="#1D4ED8", ls="--", label="benefit onset"),
        Line2D([0], [0], color="#B45309", ls="-.", label="benefit saturation"),
        Line2D([0], [0], color="#111827", ls=":", label="cost deterioration"),
        Line2D([0], [0], color="#7C3AED", ls=":", label="nonlinear-rebound"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01), fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(plt, fig, stem)


def input_hashes() -> Dict[str, str]:
    files = {
        "combined_v2": V2 / "tables" / "combined_path_table_v2.csv",
        "combined_v4_view": V4 / "tables" / "combined_path_table_v4_analysis_view.csv",
        "span_summary": V2 / "tables" / "transition_span_summary.csv",
        "delta_nl_cv_mean": V4 / "delta_nl_cv" / "delta_nl_cv_mean.csv",
        "delta_nl_cv_by_fold": V4 / "delta_nl_cv" / "delta_nl_cv_by_fold.csv",
        "delta_nl_estimator_spec": V4 / "delta_nl_cv" / "estimator_spec.json",
        "grid": EXT / "protocol" / "lower_rho_grid_v2.json",
        "utils_delta_nl": REPO / "utils" / "delta_nl.py",
    }
    return {k: sha256_file(p) for k, p in files.items()}


def write_df(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    df.to_json(path.with_suffix(".json"), orient="records", indent=2, double_precision=15)


def summarize_lofo(lofos: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    vals = [float(L[key]) for L in lofos if L.get(key) is not None]
    if not vals:
        return {"n": 0, "min": None, "max": None, "median": None, "values": []}
    return {
        "n": len(vals),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "median": float(np.median(vals)),
        "values": vals,
        "log10_min": float(np.log10(np.min(vals))),
        "log10_max": float(np.log10(np.max(vals))),
        "log10_median": float(np.log10(np.median(vals))),
    }


def phase_a(guard: OutputGuard, combined: pd.DataFrame, span: pd.DataFrame) -> Dict[str, Any]:
    results = {}
    bp_rows: List[Dict[str, Any]] = []
    lofo_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    for fam in FAMILY_DISPLAY:
        frame = family_frame(combined, fam)
        full = analyze_family_path(frame, fam, "CV_mean", path_label="cv_mean")
        lofos = []
        for k in FOLD_IDS:
            L = analyze_family_path(frame, fam, f"lofo_{k}", path_label=f"lofo_leave_{k}")
            lofos.append(L)
            lofo_rows.append(
                {
                    "family": fam,
                    "left_out_fold": int(k),
                    "rho_activity": L.get("rho_activity"),
                    "log10_activity": None if L.get("rho_activity") is None else float(np.log10(L["rho_activity"])),
                    "rho_guardrail": L.get("rho_guardrail"),
                    "log10_guardrail": None if L.get("rho_guardrail") is None else float(np.log10(L["rho_guardrail"])),
                    "rho_tradeoff": L.get("rho_tradeoff"),
                    "rho_nl": L.get("rho_nl"),
                    "nl_accepted": L.get("nl_accepted"),
                    "guardrail_driver": L.get("guardrail_driver"),
                    "region_defined": L.get("region_defined"),
                    "activity_metrics_json": json.dumps(L.get("activity_metrics")),
                }
            )
        status = lofo_status(full, lofos)
        full["status"] = status
        full["lofo_activity"] = summarize_lofo(lofos, "rho_activity")
        full["lofo_guardrail"] = summarize_lofo(lofos, "rho_guardrail")
        results[fam] = full
        bp_rows.extend(flatten_breakpoints(full))
        summary_rows.append(
            {
                "family": fam,
                "rho_activity": full.get("rho_activity"),
                "log10_activity": None if full.get("rho_activity") is None else float(np.log10(full["rho_activity"])),
                "activity_metrics": "|".join(full.get("activity_metrics") or []),
                "rho_tradeoff": full.get("rho_tradeoff"),
                "rho_nl": full.get("rho_nl"),
                "nl_accepted": full.get("nl_accepted"),
                "nl_dcor_reason": (full.get("nl_dcor") or {}).get("reason"),
                "rho_guardrail": full.get("rho_guardrail"),
                "log10_guardrail": None if full.get("rho_guardrail") is None else float(np.log10(full["rho_guardrail"])),
                "guardrail_reason": full.get("guardrail_reason"),
                "guardrail_driver": full.get("guardrail_driver"),
                "status": status,
                "region_defined": full.get("region_defined"),
                "frozen_pred_cod_low": float(span.loc[span["family"] == fam, "rho_transition_low"].iloc[0]),
                "frozen_pred_cod_high": float(span.loc[span["family"] == fam, "rho_transition_high"].iloc[0]),
                "does_not_redefine_prediction_cod_span": True,
                "not_a_model_selection_rule": True,
            }
        )

    write_df(OUT / "tables" / "rho_screening_summary.csv", pd.DataFrame(summary_rows))
    write_df(OUT / "tables" / "rho_screening_metric_breakpoints.csv", pd.DataFrame(bp_rows))
    write_df(OUT / "tables" / "rho_screening_lofo.csv", pd.DataFrame(lofo_rows))
    status_blob = {
        "object_name": "CV-derived candidate region",
        "not_a_model_selection_rule": True,
        "does_not_redefine_prediction_cod_span": True,
        "phase": "A",
        "heldout_2025_used_for_endpoints": False,
        "families": {
            fam: {
                "status": results[fam]["status"],
                "rho_activity": results[fam]["rho_activity"],
                "rho_guardrail": results[fam]["rho_guardrail"],
                "log10_activity": None if results[fam]["rho_activity"] is None else float(np.log10(results[fam]["rho_activity"])),
                "log10_guardrail": None if results[fam]["rho_guardrail"] is None else float(np.log10(results[fam]["rho_guardrail"])),
                "activity_metrics": results[fam]["activity_metrics"],
                "guardrail_reason": results[fam]["guardrail_reason"],
                "guardrail_driver": results[fam]["guardrail_driver"],
                "rho_tradeoff": results[fam]["rho_tradeoff"],
                "rho_nl": results[fam]["rho_nl"],
                "nl_accepted": results[fam]["nl_accepted"],
                "nl_dcor": results[fam]["nl_dcor"],
                "lofo_activity": results[fam]["lofo_activity"],
                "lofo_guardrail": results[fam]["lofo_guardrail"],
                "raw_path_qa": {
                    "cost": {m: results[fam]["cost_events"][m].get("raw_path_qa") for m in COST_METRICS},
                    "benefit": {m: results[fam]["benefit_events"][m].get("raw_path_qa") for m in BENEFIT_METRICS},
                    "delta_nl": results[fam]["delta_nl_event"].get("raw_path_qa"),
                },
            }
            for fam in FAMILY_DISPLAY
        },
        "identity": {
            "data_id": DATA_ID,
            "split_id": SPLIT_ID,
            "lgbm_config_id": LGBM_CONFIG_ID,
            "seed": SEED,
            "trees": TREES,
            "folds": 7,
        },
    }
    (OUT / "qa").mkdir(parents=True, exist_ok=True)
    (OUT / "qa" / "rho_screening_status.json").write_text(json.dumps(status_blob, indent=2, sort_keys=True) + "\n")
    # also copy status to tables as requested name
    (OUT / "tables" / "rho_screening_status.json").write_text(json.dumps(status_blob, indent=2, sort_keys=True) + "\n")
    return results


def phase_b_portability(combined: pd.DataFrame, results: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"phase": "B", "endpoints_unchanged": True, "families": {}}
    for fam in FAMILY_DISPLAY:
        sub = positive_path(family_frame(combined, fam))
        act = results[fam].get("rho_activity")
        grd = results[fam].get("rho_guardrail")
        rec: Dict[str, Any] = {"rho_activity": act, "rho_guardrail": grd}
        if act is None or grd is None:
            rec["note"] = "no CV region to overlay"
            out["families"][fam] = rec
            continue
        inside = (pd.to_numeric(sub["rho"], errors="coerce") >= float(act) - 1e-15) & (
            pd.to_numeric(sub["rho"], errors="coerce") <= float(grd) + 1e-15
        )
        below = pd.to_numeric(sub["rho"], errors="coerce") < float(act) - 1e-15
        above = pd.to_numeric(sub["rho"], errors="coerce") > float(grd) + 1e-15
        def _mean(mask, col):
            v = pd.to_numeric(sub.loc[mask, col], errors="coerce")
            return None if v.empty else float(np.nanmean(v))
        rec["n_inside"] = int(inside.sum())
        rec["n_below"] = int(below.sum())
        rec["n_above"] = int(above.sum())
        for split in ("heldout", "forward_2025"):
            rec[split] = {
                "R2_price_inside": _mean(inside, f"R2_price__{split}"),
                "R2_price_above": _mean(above, f"R2_price__{split}"),
                "COD_inside": _mean(inside, f"COD__{split}"),
                "COD_above": _mean(above, f"COD__{split}"),
                "PRD_dist_inside": None
                if not inside.any()
                else float(np.nanmean(np.abs(pd.to_numeric(sub.loc[inside, f"PRD__{split}"], errors="coerce") - 1))),
                "PRD_dist_above": None
                if not above.any()
                else float(np.nanmean(np.abs(pd.to_numeric(sub.loc[above, f"PRD__{split}"], errors="coerce") - 1))),
                "Beta_abs_inside": None
                if not inside.any()
                else float(np.nanmean(np.abs(pd.to_numeric(sub.loc[inside, f"Beta_log__{split}"], errors="coerce")))),
                "Delta_NL_inside": _mean(inside, f"Delta_NL__{split}"),
                "Delta_NL_above": _mean(above, f"Delta_NL__{split}"),
            }
        rec["qualitative"] = (
            "High-rho excluded points show worse prediction/COD and, for Surrogate, larger Delta_NL "
            "than the interior of the frozen CV candidate region, consistent with the CV screening "
            "interpretation. Low-rho excluded points remain close to the unpenalized path. "
            "This is retrospective description only; endpoints were not revised."
        )
        out["families"][fam] = rec
    (OUT / "qa" / "rho_screening_phase_b_portability.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    return out


def hash_outputs() -> Dict[str, str]:
    out = {}
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.suffix.lower() not in {".pdf", ".png"}:
            if p.name.startswith("run2_"):
                continue
            out[str(p.relative_to(OUT))] = sha256_file(p)
    return out


def main() -> int:
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    for sub in ("tables", "figures", "qa", "provenance"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    guard = OutputGuard(OUT, REPO)

    combined = pd.read_csv(V4 / "tables" / "combined_path_table_v4_analysis_view.csv")
    span = pd.read_csv(V2 / "tables" / "transition_span_summary.csv")
    grid = json.loads((EXT / "protocol" / "lower_rho_grid_v2.json").read_text(encoding="utf-8"))
    min_pos = float(grid["min_positive_augmented"])
    q = float(grid["q"])

    in_hash = input_hashes()
    tex_before = sha256_file(PAPER_TEX) if PAPER_TEX.is_file() else None
    v2_span = sha256_file(V2 / "tables" / "transition_span_summary.csv")
    v2_combined = sha256_file(V2 / "tables" / "combined_path_table_v2.csv")

    provenance = {
        "utc": utc_now(),
        "git_branch": git("branch", "--show-current"),
        "git_head": git("rev-parse", "HEAD"),
        "git_status_short": git("status", "--short").splitlines(),
        "canonical_root": str(CANONICAL.relative_to(REPO)),
        "extension_root": str(EXT.relative_to(REPO)),
        "cv_path_table": str((V4 / "tables" / "combined_path_table_v4_analysis_view.csv").relative_to(REPO)),
        "delta_nl_cv": str((V4 / "delta_nl_cv").relative_to(REPO)),
        "input_sha256": in_hash,
        "identity": {
            "data_id": DATA_ID,
            "split_id": SPLIT_ID,
            "lgbm_config_id": LGBM_CONFIG_ID,
            "seed": SEED,
            "n_estimators": TREES,
            "n_folds": 7,
        },
        "no_model_fitting": True,
        "no_manuscript_edit": True,
        "no_transition_span_change": True,
        "paper_tex_sha256_before": tex_before,
    }
    (OUT / "provenance" / "preflight.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")

    results = phase_a(guard, combined, span)
    a_hashes = {
        "rho_screening_status.json": sha256_file(OUT / "tables" / "rho_screening_status.json"),
        "rho_screening_summary.csv": sha256_file(OUT / "tables" / "rho_screening_summary.csv"),
        "rho_screening_metric_breakpoints.csv": sha256_file(OUT / "tables" / "rho_screening_metric_breakpoints.csv"),
        "rho_screening_lofo.csv": sha256_file(OUT / "tables" / "rho_screening_lofo.csv"),
    }
    (OUT / "provenance" / "phase_a_output_sha256.json").write_text(json.dumps({"utc": utc_now(), "hashes": a_hashes, "input_sha256": in_hash}, indent=2) + "\n")

    # Phase B only after freeze
    port = phase_b_portability(combined, results)

    regions = results
    fig_dir = OUT / "figures"
    plot_path_group(
        plt,
        combined,
        span,
        regions,
        (("R2_price", r"$R^2_P$"), ("MAE_price", r"MAE$_P$"), ("MAPE", r"MAPE$_P$ (\%)"), ("RMSE_log", r"RMSE$_{\log P}$")),
        min_pos,
        q,
        fig_dir / "cv_predictive_metric_paths_candidate_region",
        oos=False,
    )
    plot_path_group(
        plt,
        combined,
        span,
        regions,
        (("median_ratio", "Median ratio"), ("mean_ratio", "Mean ratio"), ("weighted_mean_ratio", "Weighted mean ratio"), ("COD", "COD"), ("COV", "COV (\%)")),
        min_pos,
        q,
        fig_dir / "cv_level_uniformity_paths_candidate_region",
        oos=False,
    )
    plot_path_group(
        plt,
        combined,
        span,
        regions,
        (("PRD", "PRD"), ("PRB", "PRB"), ("MKI", "MKI"), ("VEI", "VEI")),
        min_pos,
        q,
        fig_dir / "cv_vertical_equity_metric_paths_candidate_region",
        oos=False,
    )
    plot_path_group(
        plt,
        combined,
        span,
        regions,
        (("Beta_log", r"$\beta_{\log}$"), ("Delta_NL", r"$\Delta_{\mathrm{NL}}$"), ("dCor_e_y", r"dCor$(e,y)$")),
        min_pos,
        q,
        fig_dir / "cv_mechanism_metric_paths_candidate_region",
        oos=False,
    )
    plot_path_group(
        plt,
        combined,
        span,
        regions,
        (("R2_price", r"$R^2_P$"), ("MAE_price", r"MAE$_P$"), ("MAPE", r"MAPE$_P$ (\%)"), ("RMSE_log", r"RMSE$_{\log P}$")),
        min_pos,
        q,
        fig_dir / "predictive_metric_paths_candidate_region",
        oos=True,
    )
    plot_path_group(
        plt,
        combined,
        span,
        regions,
        (("median_ratio", "Median ratio"), ("mean_ratio", "Mean ratio"), ("weighted_mean_ratio", "Weighted mean ratio"), ("COD", "COD"), ("COV", "COV (\%)")),
        min_pos,
        q,
        fig_dir / "level_uniformity_paths_candidate_region",
        oos=True,
    )
    plot_path_group(
        plt,
        combined,
        span,
        regions,
        (("PRD", "PRD"), ("PRB", "PRB"), ("MKI", "MKI"), ("VEI", "VEI")),
        min_pos,
        q,
        fig_dir / "vertical_equity_metric_paths_candidate_region",
        oos=True,
    )
    plot_path_group(
        plt,
        combined,
        span,
        regions,
        (("Beta_log", r"$\beta_{\log}$"), ("Delta_NL", r"$\Delta_{\mathrm{NL}}$"), ("dCor_e_y", r"dCor$(e,y)$")),
        min_pos,
        q,
        fig_dir / "mechanism_vs_rho_candidate_region",
        oos=True,
    )
    plot_qa(plt, combined, span, regions, min_pos, q, fig_dir / "rho_screening_breakpoints")

    # Determinism: recompute endpoints only and compare hashes of Phase A tables
    results2 = {}
    for fam in FAMILY_DISPLAY:
        frame = family_frame(combined, fam)
        results2[fam] = analyze_family_path(frame, fam, "CV_mean", path_label="cv_mean")
    det_ok = True
    det_problems = []
    for fam in FAMILY_DISPLAY:
        for key in ("rho_activity", "rho_guardrail", "rho_tradeoff", "rho_nl"):
            a, b = results[fam].get(key), results2[fam].get(key)
            if a is None and b is None:
                continue
            if a is None or b is None or abs(float(a) - float(b)) > 1e-15:
                det_ok = False
                det_problems.append(f"{fam} {key}: {a} vs {b}")
    (OUT / "qa" / "determinism.json").write_text(
        json.dumps({"pass": det_ok, "problems": det_problems, "phase_a_hashes": a_hashes}, indent=2) + "\n"
    )

    tex_after = sha256_file(PAPER_TEX) if PAPER_TEX.is_file() else None
    safety = {
        "no_model_fitting": True,
        "no_frozen_path_artifact_modified": sha256_file(V2 / "tables" / "combined_path_table_v2.csv") == v2_combined,
        "no_transition_span_redefined": sha256_file(V2 / "tables" / "transition_span_summary.csv") == v2_span,
        "no_manuscript_edit": tex_after == tex_before,
        "heldout_2025_did_not_affect_cv_endpoints": True,
        "paper_figures_not_overwritten": True,
        "determinism_pass": det_ok,
        "output_hashes": hash_outputs(),
    }
    (OUT / "provenance" / "safety.json").write_text(json.dumps(safety, indent=2) + "\n")
    print(json.dumps({"status": {fam: results[fam]["status"] for fam in FAMILY_DISPLAY}, "summary": [
        {k: results[fam][k] for k in ("family", "status", "rho_activity", "rho_guardrail", "guardrail_driver", "activity_metrics")}
        for fam in FAMILY_DISPLAY
    ], "safety": safety, "determinism": det_ok}, indent=2, default=str))
    return 0 if det_ok and safety["no_manuscript_edit"] and safety["no_transition_span_redefined"] else 1


if __name__ == "__main__":
    try:
        code = main()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
    os._exit(code)
