"""Generic CV-derived rho candidate-region engine (v2).

Screening procedure, not model selection. Knows nothing about CCAO rho values.
All clustering, neighborhoods, and stability windows are grid-index / log-spacing
quantities. The only rho values consumed are those supplied by the caller.
"""

from __future__ import annotations

from math import ceil
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

METHOD_VERSION = "v2"
EPS = 1e-12

# Methodological constants only (no observed rho results).
BENEFIT_MAJORITY = 3
PREDICTIVE_CLUSTER_MIN_METRICS = 2
CLUSTER_WIDTH_H_MULT = 2.0
LOFO_STABLE_MIN = 5
LOFO_N_FOLDS = 7
LOFO_WINDOW_H_MULT = 4.0
MIN_SEGMENT_FRAC = 0.05
MIN_SEGMENT_ABS = 5
RAW_NEIGHBOR_STEPS = 4
SCALE_EQUIVARIANCE_FACTOR = 31.0  # unused by fitting; QA multiplier only

BENEFIT_METRICS = ("PRD", "PRB", "MKI", "VEI", "Beta_log")
PREDICTIVE_COST_METRICS = ("R2_price", "MAE_price", "MAPE", "RMSE_log")
UNIFORMITY_SUPPORT_METRIC = "COD"
BENEFIT_TARGET = {"PRD": 1.0, "PRB": 0.0, "MKI": 1.0, "VEI": 0.0, "Beta_log": 0.0}
HIGHER_BETTER_RAW = {"R2_price"}

METHOD_SPEC: Dict[str, Any] = {
    "version": METHOD_VERSION,
    "object_name": "CV-derived candidate region",
    "purpose": (
        "Soft-screen positive rho values that have moved beyond the low-activity/"
        "near-baseline regime but have not yet entered the first robust high-rho "
        "deterioration/pathology regime. Not a model-selection or deployment rule."
    ),
    "not_a_model_selection_rule": True,
    "coordinate": "x_j = log10(rho_j) on observed positive-grid points only",
    "rho_zero_role": "path origin / context; excluded from log-rho breakpoint fitting",
    "metric_sets": {
        "benefit": list(BENEFIT_METRICS),
        "predictive_cost": list(PREDICTIVE_COST_METRICS),
        "uniformity_support": UNIFORMITY_SUPPORT_METRIC,
        "surrogate_structural": "Delta_NL",
        "secondary_qa_only": [
            "dCor_e_y",
            "COV",
            "median_ratio",
            "mean_ratio",
            "weighted_mean_ratio",
            "ratio_shape_profiles",
        ],
    },
    "neutral_distance": {
        "PRD_dist": "abs(PRD - 1)",
        "PRB_dist": "abs(PRB)",
        "MKI_dist": "abs(MKI - 1)",
        "VEI_dist": "abs(VEI)",
        "beta_dist": "abs(beta_log)",
    },
    "predictive_cost_transform": {
        "R2_cost": "R2_at_rho0 - R2",
        "MAE_cost": "MAE - MAE_at_rho0",
        "MAPE_cost": "MAPE - MAPE_at_rho0",
        "RMSElog_cost": "RMSElog - RMSElog_at_rho0",
        "reference": "within-family custom-objective rho=0",
    },
    "piecewise_linear": {
        "complexities": [1, 2, 3],
        "breakpoints": "observed grid locations only",
        "search": "deterministic exhaustive",
        "selection": "BIC; ties prefer fewer segments then smaller first break index",
        "continuity": "hinge basis max(x-xb, 0)",
        "forbidden": [
            "smoothing splines",
            "manual smoothing",
            "numerical second derivatives",
            "interpolated rho values",
            "hand-adjusted breakpoints",
        ],
    },
    "min_segment_points": "max(5, ceil(0.05 * N_positive_rho))",
    "benefit_onset_rule": (
        "A metric has benefit_onset only if the selected PWL supports a breakpoint, "
        "the post-break fitted slope is negative and more favorable than the pre-break "
        "slope, and raw-path QA supports the interpretation. Family activity onset is "
        "the first observed positive-rho grid point at which at least 3 of 5 benefit "
        "metrics have entered their supported active-improvement regime."
    ),
    "benefit_saturation": (
        "Descriptive validation only in v2. Does not define the Direct upper guardrail."
    ),
    "direct_deterioration_cluster_rule": (
        "Cluster supported predictive-cost deterioration breakpoints when log-rho "
        "separation is at most 2*h, h = median positive log-grid spacing. Direct "
        "upper guardrail is the earliest observed-grid breakpoint in the earliest "
        "cluster that contains at least 2 of the 4 predictive metrics. COD may "
        "corroborate but cannot form or move the cluster by itself."
    ),
    "surrogate_nonlinear_rebound_rule": (
        "Delta_NL valley then persistent increasing regime. Rebound is the breakpoint "
        "into the increasing segment, not the exact minimum. Ratio-shape QA assigns "
        "CONFIRMED / CAUTION_ONLY / AMBIGUOUS. Only CONFIRMED may bind the Surrogate "
        "upper guardrail; CAUTION_ONLY is reported as a caution threshold."
    ),
    "breakpoint_cluster_tolerance": "2 * h, h = median_j (x_{j+1}-x_j)",
    "lofo_stability_rule": (
        "Repeat the full procedure on each of 7 leave-one-fold-out CV aggregates. "
        "An endpoint is LOFO-stable if at least 5 of 7 detect the same qualitative "
        "event and the location lies within +/- 4 median log-grid steps of the "
        "full-CV event. Do not center LOFO search on the full-CV rho."
    ),
    "phase_separation": (
        "Phase A uses chronological CV only. Phase B overlays held-out/2025 after "
        "Phase A hashes are frozen and must not revise endpoints or statuses."
    ),
    "status_definitions": {
        "PASS": "activity onset and upper guardrail defined; LOFO-stable broad regime",
        "AMBIGUOUS": "broad regime visible but numerical endpoint not LOFO-stable",
        "NO_BAND": "procedure does not identify a defensible candidate region",
        "DIRECT_GUARDRAIL_AMBIGUOUS": "no robust predictive-deterioration cluster",
        "SURROGATE_GUARDRAIL_AMBIGUOUS": "neither predictive nor confirmed NL guardrail is stable",
        "nonlinear_CONFIRMED": "supported persistent Delta_NL rebound, LOFO-stable, ratio-shape deformation near the event across a majority of folds",
        "nonlinear_CAUTION_ONLY": "Delta_NL rebound clear and LOFO-stable but ratio-shape deformation not yet substantively visible near the event",
        "nonlinear_AMBIGUOUS": "nonlinear rebound unsupported or LOFO-unstable",
        "dCor_REBOUND": "dCor turns upward/worsens near the NL event",
        "dCor_FLATTENING": "dCor still improving but at a smaller rate; not pathology confirmation",
        "dCor_STILL_IMPROVING": "no meaningful dCor deterioration",
        "dCor_AMBIGUOUS": "dCor local behavior not classifiable",
    },
    "methodological_constants": {
        "BENEFIT_MAJORITY": BENEFIT_MAJORITY,
        "PREDICTIVE_CLUSTER_MIN_METRICS": PREDICTIVE_CLUSTER_MIN_METRICS,
        "CLUSTER_WIDTH_H_MULT": CLUSTER_WIDTH_H_MULT,
        "LOFO_STABLE_MIN": LOFO_STABLE_MIN,
        "LOFO_N_FOLDS": LOFO_N_FOLDS,
        "LOFO_WINDOW_H_MULT": LOFO_WINDOW_H_MULT,
        "MIN_SEGMENT_FRAC": MIN_SEGMENT_FRAC,
        "MIN_SEGMENT_ABS": MIN_SEGMENT_ABS,
        "RAW_NEIGHBOR_STEPS": RAW_NEIGHBOR_STEPS,
        "SCALE_EQUIVARIANCE_FACTOR": SCALE_EQUIVARIANCE_FACTOR,
        "EPS": EPS,
    },
    "forbidden": [
        "CCAO-specific rho endpoints as algorithm inputs",
        "absolute rho-distance tolerances",
        "model refitting",
        "grid extension",
        "redefining the frozen prediction/COD transition span",
    ],
}


def min_segment_points(n_positive: int) -> int:
    n = int(n_positive)
    return int(max(MIN_SEGMENT_ABS, ceil(MIN_SEGMENT_FRAC * n)))


def log10_rho(rho: np.ndarray) -> np.ndarray:
    r = np.asarray(rho, dtype=float)
    if np.any(~np.isfinite(r) | (r <= 0)):
        raise ValueError("log10_rho requires strictly positive finite rho")
    return np.log10(r)


def median_log_spacing(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return float("nan")
    d = np.diff(x)
    d = d[np.isfinite(d) & (d > 0)]
    if d.size == 0:
        return float("nan")
    return float(np.median(d))


def benefit_distance(metric: str, values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    return np.abs(v - float(BENEFIT_TARGET[metric]))


def predictive_cost(metric: str, values: np.ndarray, value_at_rho0: float) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    ref = float(value_at_rho0)
    if metric in HIGHER_BETTER_RAW:
        return ref - v
    return v - ref


def bic(sse: float, n: int, k: int) -> float:
    sse = max(float(sse), 1e-30)
    n = max(int(n), 1)
    return float(n * np.log(sse / n) + int(k) * np.log(n))


def fit_continuous_pwl(x: np.ndarray, y: np.ndarray, breaks: Sequence[float]) -> Dict[str, Any]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    cols = [np.ones(x.size), x]
    for b in breaks:
        cols.append(np.maximum(x - float(b), 0.0))
    a = np.column_stack(cols)
    coef, _, _, _ = np.linalg.lstsq(a, y, rcond=None)
    pred = a @ coef
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
        "k_params": int(a.shape[1]),
        "bic": bic(sse, int(x.size), int(a.shape[1])),
        "n_breaks": int(len(breaks)),
    }


def valid_break_index_tuples(n: int, n_breaks: int, min_seg: int) -> List[Tuple[int, ...]]:
    out: List[Tuple[int, ...]] = []
    if n_breaks == 1:
        for b in range(min_seg - 1, n - min_seg):
            out.append((b,))
        return out
    if n_breaks == 2:
        for b1 in range(min_seg - 1, n - 2 * min_seg + 1):
            for b2 in range(b1 + min_seg - 1, n - min_seg):
                out.append((b1, b2))
    return out


def search_pwl(x: np.ndarray, y: np.ndarray, n_breaks: int, min_seg: int) -> Optional[Dict[str, Any]]:
    n = int(x.size)
    candidates = valid_break_index_tuples(n, n_breaks, min_seg)
    if not candidates:
        return None
    best: Optional[Dict[str, Any]] = None
    for idx in candidates:
        breaks = [float(x[i]) for i in idx]
        fit = fit_continuous_pwl(x, y, breaks)
        rec = {
            **{k: fit[k] for k in ("sse", "slopes", "n", "k_params", "bic", "coef", "n_breaks")},
            "break_indices": list(idx),
            "break_x": breaks,
            "n_candidates": len(candidates),
        }
        if best is None:
            best = rec
            continue
        better = rec["sse"] < best["sse"] - 1e-15
        tie = abs(rec["sse"] - best["sse"]) <= 1e-15
        if better or (tie and rec["break_indices"] < best["break_indices"]):
            best = rec
    return best


def select_pwl(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """BIC choice among 1/2/3 continuous segments. Breaks only at grid points."""
    n = int(x.size)
    min_seg = min_segment_points(n)
    fits: List[Dict[str, Any]] = []
    one = fit_continuous_pwl(x, y, [])
    fits.append(
        {
            **{k: one[k] for k in ("sse", "slopes", "n", "k_params", "bic", "coef", "n_breaks")},
            "break_indices": [],
            "break_x": [],
            "complexity": "one_segment",
            "n_candidates": 1,
            "min_seg": min_seg,
            "available": True,
        }
    )
    two = search_pwl(x, y, 1, min_seg)
    if two is None:
        fits.append({"complexity": "two_segment", "available": False, "min_seg": min_seg, "bic": float("inf")})
    else:
        two["complexity"] = "two_segment"
        two["available"] = True
        two["min_seg"] = min_seg
        fits.append(two)
    three = search_pwl(x, y, 2, min_seg)
    if three is None:
        fits.append({"complexity": "three_segment", "available": False, "min_seg": min_seg, "bic": float("inf")})
    else:
        three["complexity"] = "three_segment"
        three["available"] = True
        three["min_seg"] = min_seg
        fits.append(three)
    available = [f for f in fits if f.get("available")]
    chosen = min(
        available,
        key=lambda f: (float(f["bic"]), int(f.get("n_breaks", 0)), tuple(f.get("break_indices") or [])),
    )
    return {"chosen": chosen, "candidates": fits, "min_seg": min_seg, "n": n}


def _finite_diff_majority(y: np.ndarray, i0: int, i1: int, want_positive: bool) -> Dict[str, Any]:
    seg = np.asarray(y[i0 : i1 + 1], dtype=float)
    if seg.size < 3:
        return {"ok": False, "n_steps": 0, "n_agree": 0, "majority": False, "isolated": True}
    diffs = np.diff(seg)
    diffs = diffs[np.isfinite(diffs)]
    n_steps = int(diffs.size)
    if want_positive:
        n_agree = int(np.sum(diffs > EPS))
    else:
        n_agree = int(np.sum(diffs < -EPS))
    majority = n_steps > 0 and n_agree > n_steps / 2.0
    # Drop the last point: remaining post-segment should still have the same majority sign.
    if seg.size >= 4:
        diffs2 = np.diff(seg[:-1])
        diffs2 = diffs2[np.isfinite(diffs2)]
        if want_positive:
            n2 = int(np.sum(diffs2 > EPS))
        else:
            n2 = int(np.sum(diffs2 < -EPS))
        isolated = not (diffs2.size > 0 and n2 > diffs2.size / 2.0)
    else:
        isolated = True
    return {
        "ok": bool(majority and (not isolated) and n_agree >= 2),
        "n_steps": n_steps,
        "n_agree": n_agree,
        "majority": bool(majority),
        "isolated": bool(isolated),
    }


def raw_path_qa(x: np.ndarray, y: np.ndarray, bidx: int, *, want_positive: bool, end_idx: Optional[int] = None) -> Dict[str, Any]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    bidx = int(bidx)
    end = int(x.size - 1 if end_idx is None else end_idx)
    i0 = max(0, bidx - RAW_NEIGHBOR_STEPS)
    i1 = min(int(x.size) - 1, bidx + RAW_NEIGHBOR_STEPS)
    def _slope(xx, yy):
        if xx.size < 2 or not np.all(np.isfinite(xx)) or not np.all(np.isfinite(yy)):
            return float("nan")
        return float(np.polyfit(xx, yy, 1)[0])
    raw_pre = _slope(x[i0 : bidx + 1], y[i0 : bidx + 1])
    raw_post = _slope(x[bidx : i1 + 1], y[bidx : i1 + 1])
    post = _finite_diff_majority(y, bidx, end, want_positive)
    slope_ok = np.isfinite(raw_post) and ((raw_post > EPS) if want_positive else (raw_post < -EPS))
    supported = bool(slope_ok and post["ok"])
    return {
        "raw_pre_slope": raw_pre,
        "raw_post_slope": raw_post,
        "n_pre": int(bidx - i0 + 1),
        "n_post_window": int(i1 - bidx + 1),
        "post_majority": post,
        "supported": supported,
        "status": "supported" if supported else "ambiguous",
    }


def interpret_benefit(selected: Dict[str, Any], x: np.ndarray, y: np.ndarray, rho: np.ndarray) -> Dict[str, Any]:
    chosen = selected["chosen"]
    rho = np.asarray(rho, dtype=float)
    slopes = [float(s) for s in chosen.get("slopes") or []]
    idx = [int(i) for i in (chosen.get("break_indices") or [])]
    onset = None
    sat = None
    onset_idx = None
    sat_idx = None
    if chosen.get("complexity") == "three_segment" and len(slopes) == 3 and len(idx) == 2:
        s1, s2, s3 = slopes
        if (s2 < -EPS) and (s2 < s1 - EPS):
            onset = float(rho[idx[0]])
            onset_idx = idx[0]
        if onset is not None and (s3 > s2 + EPS):
            sat = float(rho[idx[1]])
            sat_idx = idx[1]
    elif chosen.get("complexity") == "two_segment" and len(slopes) == 2 and len(idx) == 1:
        s_pre, s_post = slopes
        bidx = idx[0]
        rho_b = float(rho[bidx])
        if (s_post < -EPS) and (s_post < s_pre - EPS):
            onset = rho_b
            onset_idx = bidx
        elif (s_pre < -EPS) and (s_post > s_pre + EPS):
            sat = rho_b
            sat_idx = bidx
    qa_onset = None
    qa_sat = None
    if onset_idx is not None:
        qa_onset = raw_path_qa(x, y, onset_idx, want_positive=False, end_idx=sat_idx if sat_idx is not None else None)
        if not qa_onset["supported"]:
            onset, onset_idx = None, None
    if sat_idx is not None and onset is not None:
        qa_sat = raw_path_qa(x, y, sat_idx, want_positive=True)
        # saturation may flatten (post slope less negative) rather than reverse
        if qa_sat["status"] == "ambiguous" and np.isfinite(qa_sat["raw_post_slope"]) and np.isfinite(qa_sat["raw_pre_slope"]):
            if qa_sat["raw_post_slope"] > qa_sat["raw_pre_slope"] - 1e-12:
                qa_sat["status"] = "supported"
                qa_sat["supported"] = True
        if not qa_sat["supported"]:
            sat, sat_idx = None, None
            qa_sat["status"] = "ambiguous"
    flags = []
    if onset is not None:
        flags.append("onset_supported")
    elif qa_onset is not None:
        flags.append("onset_ambiguous")
    if sat is not None:
        flags.append("sat_supported")
    elif qa_sat is not None:
        flags.append("sat_ambiguous")
    return {
        "complexity": chosen.get("complexity"),
        "available": bool(chosen.get("available", True)),
        "benefit_onset": onset,
        "benefit_onset_index": onset_idx,
        "benefit_saturation": sat,
        "benefit_saturation_index": sat_idx,
        "slopes": slopes,
        "sse": chosen.get("sse"),
        "bic": chosen.get("bic"),
        "k_params": chosen.get("k_params"),
        "n": chosen.get("n"),
        "min_seg": selected.get("min_seg"),
        "break_indices": idx,
        "break_x": [float(v) for v in (chosen.get("break_x") or [])],
        "n_candidates": chosen.get("n_candidates"),
        "raw_path_qa": "|".join(flags) if flags else "invalid",
        "qa_onset": qa_onset,
        "qa_sat": qa_sat,
        "classification": "VALID" if onset is not None else "INVALID",
        "label": "descriptive CV path breakpoint",
    }


def interpret_cost_deterioration(selected: Dict[str, Any], x: np.ndarray, y: np.ndarray, rho: np.ndarray) -> Dict[str, Any]:
    chosen = selected["chosen"]
    slopes = [float(s) for s in chosen.get("slopes") or []]
    idx = [int(i) for i in (chosen.get("break_indices") or [])]
    event_idx = None
    # Walk breaks from left; first transition into a more adverse (larger) positive slope.
    if chosen.get("complexity") in {"two_segment", "three_segment"} and idx and len(slopes) >= 2:
        for k, bidx in enumerate(idx):
            s_pre = slopes[k]
            s_post = slopes[k + 1]
            if (s_post > EPS) and (s_post > s_pre + EPS):
                event_idx = int(bidx)
                break
    qa = None
    rho_ev = None
    rho_arr = np.asarray(rho, dtype=float)
    if event_idx is not None:
        qa = raw_path_qa(x, y, event_idx, want_positive=True)
        if qa["supported"]:
            rho_ev = float(rho_arr[event_idx])
        else:
            event_idx = None
            rho_ev = None
    return {
        "complexity": chosen.get("complexity"),
        "event": "cost_deterioration" if rho_ev is not None else None,
        "rho": rho_ev,
        "index": event_idx,
        "x": None if event_idx is None else float(x[event_idx]),
        "slopes": slopes,
        "sse": chosen.get("sse"),
        "bic": chosen.get("bic"),
        "k_params": chosen.get("k_params"),
        "n": chosen.get("n"),
        "min_seg": selected.get("min_seg"),
        "break_indices": idx,
        "break_x": [float(v) for v in (chosen.get("break_x") or [])],
        "n_candidates": chosen.get("n_candidates"),
        "raw_path_qa": None if qa is None else qa["status"],
        "qa": qa,
        "classification": "VALID" if rho_ev is not None else "INVALID",
        "label": "descriptive CV path breakpoint",
    }


def interpret_delta_nl(selected: Dict[str, Any], x: np.ndarray, y: np.ndarray, rho: np.ndarray) -> Dict[str, Any]:
    chosen = selected["chosen"]
    slopes = [float(s) for s in chosen.get("slopes") or []]
    idx = [int(i) for i in (chosen.get("break_indices") or [])]
    rebound_idx = None
    if chosen.get("complexity") == "three_segment" and len(slopes) == 3 and len(idx) == 2:
        s1, s2, s3 = slopes
        improving = (s1 < -EPS) or (s2 < -EPS)
        if improving and (s3 > EPS) and (s3 > s2 + EPS):
            rebound_idx = idx[1]
        elif (s1 < -EPS) and (s2 > EPS) and (s2 > s1 + EPS):
            rebound_idx = idx[0]
    elif chosen.get("complexity") == "two_segment" and len(slopes) == 2 and len(idx) == 1:
        s_pre, s_post = slopes
        if (s_pre < -EPS) and (s_post > EPS) and (s_post > s_pre + EPS):
            rebound_idx = idx[0]
    valley_idx = int(np.argmin(y)) if y.size else None
    qa = None
    rho_ev = None
    rho = np.asarray(rho, dtype=float)
    if rebound_idx is not None:
        if valley_idx is not None and rebound_idx < valley_idx:
            rebound_idx = None
        else:
            qa = raw_path_qa(x, y, rebound_idx, want_positive=True)
            if qa["supported"]:
                rho_ev = float(rho[rebound_idx])
            else:
                rebound_idx = None
    return {
        "complexity": chosen.get("complexity"),
        "event": "nonlinear_rebound" if rho_ev is not None else None,
        "rho": rho_ev,
        "index": rebound_idx,
        "x": None if rebound_idx is None else float(x[rebound_idx]),
        "valley_index": valley_idx,
        "valley_rho": None if valley_idx is None else float(rho[valley_idx]),
        "uses_exact_minimum": False,
        "slopes": slopes,
        "sse": chosen.get("sse"),
        "bic": chosen.get("bic"),
        "k_params": chosen.get("k_params"),
        "n": chosen.get("n"),
        "min_seg": selected.get("min_seg"),
        "break_indices": idx,
        "break_x": [float(v) for v in (chosen.get("break_x") or [])],
        "n_candidates": chosen.get("n_candidates"),
        "raw_path_qa": None if qa is None else qa["status"],
        "qa": qa,
        "classification": "VALID" if rho_ev is not None else "INVALID",
        "label": "descriptive CV path breakpoint",
    }


def activity_onset(rho: np.ndarray, benefit_events: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    rho = np.asarray(rho, dtype=float)
    active_from: Dict[str, Optional[int]] = {}
    for m, ev in benefit_events.items():
        active_from[m] = ev.get("benefit_onset_index") if ev.get("benefit_onset") is not None else None
    for j, r in enumerate(rho):
        names = [m for m, i in active_from.items() if i is not None and j >= int(i)]
        if len(names) >= BENEFIT_MAJORITY:
            return {
                "rho": float(r),
                "index": int(j),
                "metrics": names,
                "n_active": len(names),
            }
    return {"rho": None, "index": None, "metrics": [], "n_active": 0}


def cluster_predictive_events(
    events: Dict[str, Dict[str, Any]],
    x: np.ndarray,
    rho: np.ndarray,
    h: float,
) -> Dict[str, Any]:
    pts: List[Tuple[int, str, float]] = []
    for m, ev in events.items():
        if ev.get("rho") is None or ev.get("index") is None:
            continue
        if ev.get("raw_path_qa") != "supported":
            continue
        pts.append((int(ev["index"]), str(m), float(x[int(ev["index"])])))
    pts.sort(key=lambda t: (t[0], t[1]))
    if len(pts) < PREDICTIVE_CLUSTER_MIN_METRICS or not np.isfinite(h):
        return {
            "clusters": [],
            "qualifying": None,
            "guardrail_index": None,
            "guardrail_rho": None,
            "guardrail_x": None,
            "metrics": [],
            "status": "DIRECT_GUARDRAIL_AMBIGUOUS",
        }
    clusters: List[List[Tuple[int, str, float]]] = []
    cur = [pts[0]]
    for rec in pts[1:]:
        if rec[2] - cur[-1][2] <= CLUSTER_WIDTH_H_MULT * h + 1e-15:
            cur.append(rec)
        else:
            clusters.append(cur)
            cur = [rec]
    clusters.append(cur)
    qualifying = None
    for cl in clusters:
        metrics = sorted({m for _i, m, _x in cl})
        if len(metrics) >= PREDICTIVE_CLUSTER_MIN_METRICS:
            qualifying = {"members": cl, "metrics": metrics}
            break
    if qualifying is None:
        return {
            "clusters": [
                {
                    "indices": [i for i, _m, _x in cl],
                    "metrics": sorted({m for _i, m, _x in cl}),
                    "x_min": float(cl[0][2]),
                    "x_max": float(cl[-1][2]),
                }
                for cl in clusters
            ],
            "qualifying": None,
            "guardrail_index": None,
            "guardrail_rho": None,
            "guardrail_x": None,
            "metrics": [],
            "status": "DIRECT_GUARDRAIL_AMBIGUOUS",
        }
    gidx = min(i for i, _m, _x in qualifying["members"])
    return {
        "clusters": [
            {
                "indices": [i for i, _m, _x in cl],
                "metrics": sorted({m for _i, m, _x in cl}),
                "x_min": float(cl[0][2]),
                "x_max": float(cl[-1][2]),
            }
            for cl in clusters
        ],
        "qualifying": {
            "indices": [i for i, _m, _x in qualifying["members"]],
            "metrics": qualifying["metrics"],
            "x_min": float(min(x for _i, _m, x in qualifying["members"])),
            "x_max": float(max(x for _i, _m, x in qualifying["members"])),
        },
        "guardrail_index": int(gidx),
        "guardrail_rho": float(np.asarray(rho, dtype=float)[gidx]),
        "guardrail_x": float(x[gidx]),
        "metrics": qualifying["metrics"],
        "status": "OK",
    }


def classify_dcor(x: np.ndarray, y: np.ndarray, event_index: Optional[int]) -> Dict[str, Any]:
    if event_index is None or y.size < 4:
        return {"status": "AMBIGUOUS", "reason": "no_nl_event"}
    j = int(event_index)
    qa = raw_path_qa(x, y, j, want_positive=True)
    pre, post = qa["raw_pre_slope"], qa["raw_post_slope"]
    if not (np.isfinite(pre) and np.isfinite(post)):
        return {"status": "AMBIGUOUS", "reason": "nonfinite_local_slopes", **qa}
    if post > EPS:
        status = "REBOUND"
        reason = "dCor_turns_upward"
    elif post > pre + EPS and post < -EPS:
        status = "FLATTENING"
        reason = "dCor_still_improving_smaller_rate"
    elif post <= pre + EPS and post < -EPS:
        status = "STILL_IMPROVING"
        reason = "dCor_still_improving"
    else:
        status = "AMBIGUOUS"
        reason = "dCor_unclassifiable"
    qa_out = {k: v for k, v in qa.items() if k != "status"}
    qa_out["raw_status"] = qa.get("status")
    return {"status": status, "reason": reason, "pre_slope": pre, "post_slope": post, "qa": qa_out}


def lofo_in_window(full_index: Optional[int], lofo_index: Optional[int], x: np.ndarray, h: float) -> bool:
    if full_index is None or lofo_index is None or not np.isfinite(h):
        return False
    return abs(float(x[int(lofo_index)]) - float(x[int(full_index)])) <= LOFO_WINDOW_H_MULT * h + 1e-15


def lofo_stability(
    full_index: Optional[int],
    lofo_indices: Sequence[Optional[int]],
    x: np.ndarray,
    h: float,
    *,
    require_event: bool = True,
) -> Dict[str, Any]:
    vals = list(lofo_indices)
    detected = [i for i in vals if i is not None]
    in_win = [i for i in detected if lofo_in_window(full_index, i, x, h)]
    n_ok = len(in_win)
    stable = full_index is not None and n_ok >= LOFO_STABLE_MIN
    if require_event and full_index is None:
        stable = False
    return {
        "n_detected": len(detected),
        "n_in_window": n_ok,
        "n_lofo": len(vals),
        "stable": bool(stable),
        "window_h_mult": LOFO_WINDOW_H_MULT,
        "full_index": full_index,
        "lofo_indices": [None if v is None else int(v) for v in vals],
    }


def family_status(*, activity_ok: bool, guardrail_ok: bool, activity_lofo: bool, guardrail_lofo: bool, guardrail_ambiguous_name: str) -> str:
    if not activity_ok and not guardrail_ok:
        return "NO_BAND"
    if not guardrail_ok:
        return guardrail_ambiguous_name
    if not activity_ok:
        return "NO_BAND"
    if activity_lofo and guardrail_lofo:
        return "PASS"
    return "AMBIGUOUS"


def screen_positive_path(
    rho: np.ndarray,
    *,
    benefit_raw: Dict[str, np.ndarray],
    predictive_raw: Dict[str, np.ndarray],
    predictive_rho0: Dict[str, float],
    delta_nl: Optional[np.ndarray] = None,
    dcor: Optional[np.ndarray] = None,
    cod: Optional[np.ndarray] = None,
    cod_rho0: Optional[float] = None,
) -> Dict[str, Any]:
    """Screen one positive-rho path. No CCAO constants; rho comes from the caller grid."""
    rho = np.asarray(rho, dtype=float)
    x = log10_rho(rho)
    h = median_log_spacing(x)
    min_seg = min_segment_points(int(rho.size))
    benefit_events: Dict[str, Dict[str, Any]] = {}
    for m in BENEFIT_METRICS:
        y = benefit_distance(m, np.asarray(benefit_raw[m], dtype=float))
        selected = select_pwl(x, y)
        rec = interpret_benefit(selected, x, y, rho)
        rec["metric"] = m
        rec["role"] = "benefit"
        benefit_events[m] = rec
    pred_events: Dict[str, Dict[str, Any]] = {}
    for m in PREDICTIVE_COST_METRICS:
        y = predictive_cost(m, np.asarray(predictive_raw[m], dtype=float), float(predictive_rho0[m]))
        selected = select_pwl(x, y)
        rec = interpret_cost_deterioration(selected, x, y, rho)
        rec["metric"] = m
        rec["role"] = "predictive_cost"
        pred_events[m] = rec
    cod_event = None
    if cod is not None:
        y_cod = np.asarray(cod, dtype=float)
        if cod_rho0 is not None:
            y_cod = y_cod - float(cod_rho0)
        selected = select_pwl(x, y_cod)
        cod_event = interpret_cost_deterioration(selected, x, y_cod, rho)
        cod_event["metric"] = UNIFORMITY_SUPPORT_METRIC
        cod_event["role"] = "uniformity_support"
    act = activity_onset(rho, benefit_events)
    cluster = cluster_predictive_events(pred_events, x, rho, h)
    dnl = None
    if delta_nl is not None:
        y_nl = np.asarray(delta_nl, dtype=float)
        selected = select_pwl(x, y_nl)
        dnl = interpret_delta_nl(selected, x, y_nl, rho)
        dnl["metric"] = "Delta_NL"
        dnl["role"] = "nonlinear"
    dcor_cls = None
    if dcor is not None:
        dcor_cls = classify_dcor(x, np.asarray(dcor, dtype=float), None if dnl is None else dnl.get("index"))
    return {
        "n": int(rho.size),
        "min_segment_points": min_seg,
        "h": h,
        "activity": act,
        "benefit_events": benefit_events,
        "predictive_events": pred_events,
        "cod_event": cod_event,
        "predictive_cluster": cluster,
        "rho_predictive_guardrail": cluster.get("guardrail_rho"),
        "index_predictive_guardrail": cluster.get("guardrail_index"),
        "delta_nl": dnl,
        "dcor": dcor_cls,
        "cluster_width": None if not np.isfinite(h) else CLUSTER_WIDTH_H_MULT * h,
    }


def json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        if not np.isfinite(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if not np.isfinite(v) else v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [json_safe(v) for v in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return str(obj)
