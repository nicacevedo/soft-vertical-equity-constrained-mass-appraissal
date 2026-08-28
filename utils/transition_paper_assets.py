"""Follow-up descriptive analyses on the frozen 994-tree transition-region result.

No model fitting, no protocol change, no rho selection, and no manuscript writes.
v1 tables are copied, not re-derived with a different event rule.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

MappingLike = Dict[str, Any]

import numpy as np
import pandas as pd

from utils.transition_regions import (
    ATOL,
    EXPECTED_IDENTITY,
    FAMILY_DISPLAY,
    FOLD_IDS,
    PRIMARY_METRICS,
    RTOL,
    extract_discrete_event,
    is_rho_positive,
    is_rho_zero,
    log10_distance_to_span,
    numerically_equal,
    rho_in_closed_span,
    sha256_file,
)

IAAO_PRD_RANGE = (0.98, 1.03)
IAAO_PRB_RANGE = (-0.05, 0.05)
IAAO_MKI_RANGE = (0.95, 1.05)
IAAO_VEI_RANGE = (-10.0, 10.0)  # percent, matching the combined table
POSITIVE_ANCHOR_TARGETS = (0.1, 1.0, 10.0, 100.0)
RATIO_SHAPE_ANCHOR_TARGETS = (0.0, 0.1, 1.0, 10.0, 100.0)

V1_REQUIRED_TABLES = (
    "transition_events_cv_mean.csv",
    "transition_events_by_fold.csv",
    "transition_span_summary.csv",
    "transition_lofo_sensitivity.csv",
    "transition_temporal_concordance.csv",
    "transition_mechanism_summary.csv",
    "transition_band_configs.csv",
    "transition_region_performance_envelope.csv",
    "transition_fold_stability_summary.csv",
)

ENVELOPE_METRICS_PAPER: Tuple[str, ...] = (
    "R2_price",
    "MAE_price",
    "MAPE",
    "RMSE_log",
    "COD",
    "PRD",
    "PRB",
    "MKI",
    "VEI",
    "Beta_log",
    "dCor_e_y",
    "Delta_NL",
)

ANCHOR_METRIC_SPECS: Tuple[Tuple[str, Optional[bool], Optional[float], bool], ...] = (
    ("R2_price", True, None, False),
    ("MAE_price", False, None, False),
    ("PRD", None, 1.0, True),
    ("PRB", None, 0.0, True),
    ("MKI", None, 1.0, True),
    ("VEI", None, 0.0, True),
    ("Beta_log", None, 0.0, False),
    ("Delta_NL", False, None, False),
    ("dCor_e_y", False, None, False),
)

REFERENCE_RANGES = {
    "PRD": IAAO_PRD_RANGE,
    "PRB": IAAO_PRB_RANGE,
    "MKI": IAAO_MKI_RANGE,
    "VEI": IAAO_VEI_RANGE,
}


def nearest_grid_rho(grid: Sequence[float], target: float) -> float:
    arr = np.asarray(list(grid), dtype=float)
    if arr.size == 0:
        raise ValueError("empty rho grid")
    return float(arr[int(np.argmin(np.abs(arr - float(target))))])


def positive_display_anchors(grid: Sequence[float]) -> List[float]:
    positives = [float(x) for x in grid if is_rho_positive(float(x))]
    return [nearest_grid_rho(positives, t) for t in POSITIVE_ANCHOR_TARGETS]


def ratio_shape_anchors(grid: Sequence[float]) -> List[float]:
    return [nearest_grid_rho(grid, t) for t in RATIO_SHAPE_ANCHOR_TARGETS]


def endpoint_equals_first_positive(rho: Optional[float], min_positive_rho: float) -> Optional[bool]:
    if rho is None or not np.isfinite(float(rho)):
        return None
    return bool(numerically_equal(float(rho), float(min_positive_rho)))


def endpoint_equals_last_positive(rho: Optional[float], max_positive_rho: float) -> Optional[bool]:
    if rho is None or not np.isfinite(float(rho)):
        return None
    return bool(numerically_equal(float(rho), float(max_positive_rho)))


def direction_aware_gap(best: float, other: float, direction: str) -> float:
    if direction == "max":
        return float(best - other)
    if direction == "min":
        return float(other - best)
    raise ValueError(f"direction must be max or min, got {direction!r}")


def path_range(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.max(finite) - np.min(finite))


def global_opt_index(values: Sequence[float], direction: str) -> int:
    arr = np.asarray(list(values), dtype=float)
    if direction == "max":
        return int(np.nanargmax(arr))
    if direction == "min":
        return int(np.nanargmin(arr))
    raise ValueError(f"direction must be max or min, got {direction!r}")


def second_best_event(
    rhos: Sequence[float],
    values: Sequence[float],
    *,
    direction: str,
    atol: float = ATOL,
    rtol: float = RTOL,
) -> Dict[str, Any]:
    """Next-best distinct value after the discrete global optimum."""
    r = np.asarray(list(rhos), dtype=float)
    v = np.asarray(list(values), dtype=float)
    order = np.argsort(r, kind="mergesort")
    r = r[order]
    v = v[order]
    if not np.all(np.isfinite(v)):
        return {
            "second_best_rho": None,
            "second_best_value": None,
            "best_minus_second_gap": None,
            "n_tied_best": 0,
        }
    best = float(np.max(v) if direction == "max" else np.min(v))
    tied = np.array([numerically_equal(float(x), best, atol=atol, rtol=rtol) for x in v], dtype=bool)
    remaining = v[~tied]
    remaining_r = r[~tied]
    if remaining.size == 0:
        return {
            "second_best_rho": None,
            "second_best_value": None,
            "best_minus_second_gap": None,
            "n_tied_best": int(tied.sum()),
        }
    j = int(np.argmax(remaining) if direction == "max" else np.argmin(remaining))
    second_val = float(remaining[j])
    second_rho = float(remaining_r[j])
    return {
        "second_best_rho": second_rho,
        "second_best_value": second_val,
        "best_minus_second_gap": direction_aware_gap(best, second_val, direction),
        "n_tied_best": int(tied.sum()),
    }


def neighbor_gaps(
    rhos: Sequence[float],
    values: Sequence[float],
    *,
    opt_rho: float,
    opt_value: float,
    direction: str,
) -> Dict[str, Any]:
    r = np.asarray(list(rhos), dtype=float)
    v = np.asarray(list(values), dtype=float)
    order = np.argsort(r, kind="mergesort")
    r = r[order]
    v = v[order]
    idx = [i for i, x in enumerate(r) if numerically_equal(float(x), float(opt_rho))]
    if not idx:
        return {
            "lower_neighbor_rho": None,
            "lower_neighbor_value": None,
            "lower_neighbor_gap": None,
            "higher_neighbor_rho": None,
            "higher_neighbor_value": None,
            "higher_neighbor_gap": None,
        }
    i0 = min(idx)
    i1 = max(idx)
    out: Dict[str, Any] = {
        "lower_neighbor_rho": float(r[i0 - 1]) if i0 > 0 else None,
        "lower_neighbor_value": float(v[i0 - 1]) if i0 > 0 else None,
        "higher_neighbor_rho": float(r[i1 + 1]) if i1 < len(r) - 1 else None,
        "higher_neighbor_value": float(v[i1 + 1]) if i1 < len(r) - 1 else None,
    }
    if out["lower_neighbor_value"] is None:
        out["lower_neighbor_gap"] = None
    else:
        out["lower_neighbor_gap"] = direction_aware_gap(opt_value, float(out["lower_neighbor_value"]), direction)
    if out["higher_neighbor_value"] is None:
        out["higher_neighbor_gap"] = None
    else:
        out["higher_neighbor_gap"] = direction_aware_gap(opt_value, float(out["higher_neighbor_value"]), direction)
    return out


def ordinal_rank_of_value(
    values: Sequence[float],
    candidate: float,
    *,
    direction: str,
    atol: float = ATOL,
    rtol: float = RTOL,
) -> Optional[int]:
    """1 = best. Rank is 1 + count of strictly better path points."""
    arr = np.asarray(list(values), dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0 or not np.isfinite(candidate):
        return None
    n_better = 0
    for x in finite:
        if numerically_equal(float(x), float(candidate), atol=atol, rtol=rtol):
            continue
        if direction == "max" and float(x) > float(candidate):
            n_better += 1
        if direction == "min" and float(x) < float(candidate):
            n_better += 1
    return int(1 + n_better)


def span_regret_row(
    rhos: Sequence[float],
    values: Sequence[float],
    *,
    family: str,
    split: str,
    metric: str,
    direction: str,
    rho_low: float,
    rho_high: float,
    event_classification: Optional[str] = None,
) -> Dict[str, Any]:
    r = np.asarray(list(rhos), dtype=float)
    v = np.asarray(list(values), dtype=float)
    if r.size == 0 or r.size != v.size:
        raise ValueError("rho/value path is empty or misaligned")
    if not np.all(np.isfinite(r)) or not np.all(np.isfinite(v)):
        raise ValueError(f"non-finite path for {family} {split} {metric}")
    if direction == "max":
        g_idx = int(np.argmax(v))
        global_opt_value = float(v[g_idx])
    else:
        g_idx = int(np.argmin(v))
        global_opt_value = float(v[g_idx])
    global_opt_rho = float(r[g_idx])
    mask = np.array([rho_in_closed_span(float(x), float(rho_low), float(rho_high)) for x in r], dtype=bool)
    if not bool(mask.any()):
        raise ValueError(f"no grid points inside frozen CV span for {family} {split} {metric}")
    inside_r = r[mask]
    inside_v = v[mask]
    if direction == "max":
        i_idx = int(np.argmax(inside_v))
        best_inside = float(inside_v[i_idx])
    else:
        i_idx = int(np.argmin(inside_v))
        best_inside = float(inside_v[i_idx])
    best_inside_rho = float(inside_r[i_idx])
    raw_regret = direction_aware_gap(global_opt_value, best_inside, direction)
    if raw_regret < -1e-12:
        raise ValueError(f"negative span regret for {family} {split} {metric}: {raw_regret}")
    raw_regret = max(0.0, float(raw_regret))
    rng = path_range(v)
    if np.isfinite(rng) and rng > 0:
        normalized_regret: Optional[float] = float(raw_regret / rng)
    else:
        normalized_regret = None
    zero_idx = [i for i, x in enumerate(r) if is_rho_zero(float(x))]
    rho0_value = float(v[zero_idx[0]]) if zero_idx else None
    log_dist = None
    if not rho_in_closed_span(global_opt_rho, float(rho_low), float(rho_high)):
        log_dist = log10_distance_to_span(global_opt_rho, float(rho_low), float(rho_high))
    else:
        log_dist = 0.0
    return {
        "family": family,
        "split": split,
        "metric": metric,
        "direction": direction,
        "event_classification": event_classification,
        "global_opt_rho": global_opt_rho,
        "global_opt_value": global_opt_value,
        "best_inside_rho": best_inside_rho,
        "best_inside_value": best_inside,
        "raw_regret": raw_regret,
        "path_range": rng,
        "normalized_regret": normalized_regret,
        "best_inside_ordinal_rank": ordinal_rank_of_value(v, best_inside, direction=direction),
        "log10_distance_global_opt_to_cv_span": log_dist,
        "rho0_value": rho0_value,
        "best_inside_minus_rho0": None if rho0_value is None else float(best_inside - rho0_value),
        "global_opt_minus_rho0": None if rho0_value is None else float(global_opt_value - rho0_value),
        "n_span_grid_points": int(mask.sum()),
        "cv_rho_transition_low": float(rho_low),
        "cv_rho_transition_high": float(rho_high),
        "no_interpolation": True,
        "not_a_selected_rho": True,
    }


def event_sharpness_row(
    rhos: Sequence[float],
    values: Sequence[float],
    *,
    family: str,
    split: str,
    metric: str,
    direction: str,
) -> Dict[str, Any]:
    ev = extract_discrete_event(rhos, values, metric=metric, direction=direction)
    second = second_best_event(rhos, values, direction=direction)
    opt_rho = ev.rho_low if ev.rho_low is not None else float("nan")
    opt_val = ev.metric_value if ev.metric_value is not None else float("nan")
    neigh = neighbor_gaps(
        rhos,
        values,
        opt_rho=float(opt_rho) if np.isfinite(opt_rho) else float("nan"),
        opt_value=float(opt_val) if np.isfinite(opt_val) else float("nan"),
        direction=direction,
    )
    rng = path_range(values)
    def _norm(gap: Optional[float]) -> Optional[float]:
        if gap is None or not np.isfinite(rng) or rng <= 0 or not np.isfinite(gap):
            return None
        return float(gap / rng)

    r = np.asarray(list(rhos), dtype=float)
    v = np.asarray(list(values), dtype=float)
    zero_idx = [i for i, x in enumerate(r) if is_rho_zero(float(x))]
    rho0_value = float(v[zero_idx[0]]) if zero_idx else None
    rec: Dict[str, Any] = {
        "family": family,
        "split": split,
        "metric": metric,
        "direction": direction,
        "classification": ev.classification,
        "optimum_rho": ev.rho_low,
        "optimum_rho_high": ev.rho_high,
        "optimum_value": ev.metric_value,
        "n_tied": ev.n_tied,
        "local_turn_verified": ev.local_turn_verified,
        "second_best_rho": second["second_best_rho"],
        "second_best_value": second["second_best_value"],
        "best_vs_second_gap": second["best_minus_second_gap"],
        "lower_neighbor_rho": neigh["lower_neighbor_rho"],
        "lower_neighbor_value": neigh["lower_neighbor_value"],
        "lower_neighbor_gap": neigh["lower_neighbor_gap"],
        "higher_neighbor_rho": neigh["higher_neighbor_rho"],
        "higher_neighbor_value": neigh["higher_neighbor_value"],
        "higher_neighbor_gap": neigh["higher_neighbor_gap"],
        "full_path_range": rng,
        "best_vs_second_gap_over_range": _norm(second["best_minus_second_gap"]),
        "lower_neighbor_gap_over_range": _norm(neigh["lower_neighbor_gap"]),
        "higher_neighbor_gap_over_range": _norm(neigh["higher_neighbor_gap"]),
        "rho0_value": rho0_value,
        "change_from_rho0": None if rho0_value is None or ev.metric_value is None else float(ev.metric_value - rho0_value),
        "no_smoothing": True,
    }
    if family == "Surrogate" and metric == "RMSE_log":
        rec.update(surrogate_rmse_log_zero_vs_positive(rhos, values))
    return rec


def surrogate_rmse_log_zero_vs_positive(
    rhos: Sequence[float],
    values: Sequence[float],
) -> Dict[str, Any]:
    r = np.asarray(list(rhos), dtype=float)
    v = np.asarray(list(values), dtype=float)
    zero_idx = [i for i, x in enumerate(r) if is_rho_zero(float(x))]
    pos_mask = np.array([is_rho_positive(float(x)) for x in r], dtype=bool)
    if not zero_idx or not bool(pos_mask.any()):
        return {
            "rmse_log_rho0": None,
            "best_positive_rmse_log": None,
            "best_positive_rho": None,
            "best_positive_minus_zero": None,
            "best_positive_minus_zero_over_path_range": None,
        }
    rho0 = float(v[zero_idx[0]])
    pos_r = r[pos_mask]
    pos_v = v[pos_mask]
    j = int(np.argmin(pos_v))
    best_pos = float(pos_v[j])
    best_rho = float(pos_r[j])
    gap = float(best_pos - rho0)
    rng = path_range(v)
    return {
        "rmse_log_rho0": rho0,
        "best_positive_rmse_log": best_pos,
        "best_positive_rho": best_rho,
        "best_positive_minus_zero": gap,
        "best_positive_minus_zero_over_path_range": None if (not np.isfinite(rng) or rng <= 0) else float(gap / rng),
    }


def better_than(val: float, other: float, *, higher: Optional[bool], target: Optional[float], tol: float = 1e-10) -> bool:
    if not (np.isfinite(val) and np.isfinite(other)):
        return False
    if higher is True:
        return bool(val + tol >= other)
    if higher is False:
        return bool(val - tol <= other)
    if target is None:
        return False
    return bool(abs(val - float(target)) <= abs(other - float(target)) + tol)


def in_reference_range(metric: str, val: float) -> Optional[bool]:
    if metric not in REFERENCE_RANGES:
        return None
    if not np.isfinite(val):
        return False
    lo, hi = REFERENCE_RANGES[metric]
    return bool(lo <= val <= hi)


def manuscript_format_flags(
    val: float,
    *,
    metric: str,
    family: str,
    linear_val: float,
    lgbm_val: float,
    higher: Optional[bool],
    target: Optional[float],
    can_star: bool,
) -> Dict[str, Any]:
    """paper_v12 live rule: bold = beats both baselines; asterisk = beats LightGBM only and in-range (penalized)."""
    beats_linear = better_than(val, linear_val, higher=higher, target=target)
    beats_lgbm = better_than(val, lgbm_val, higher=higher, target=target)
    within = in_reference_range(metric, val)
    if family in {"Linear", "LightGBM"}:
        other = lgbm_val if family == "Linear" else linear_val
        beats_both = better_than(val, other, higher=higher, target=target)
        beats_ordinary_only = False
        star = False
        bold = bool(beats_both)
    else:
        beats_both = bool(beats_linear and beats_lgbm)
        beats_ordinary_only = bool(beats_lgbm and not beats_both)
        star = bool(can_star and beats_ordinary_only and within is True)
        bold = bool(beats_both)
    return {
        "beats_both_baselines": beats_both,
        "beats_ordinary_only": beats_ordinary_only,
        "within_reference_range": within,
        "manuscript_bold": bold,
        "manuscript_asterisk": star,
    }


def load_v1_tables(v1_root: Path) -> Dict[str, pd.DataFrame]:
    tables = {}
    tdir = Path(v1_root) / "tables"
    for name in V1_REQUIRED_TABLES:
        path = tdir / name
        if not path.is_file():
            raise FileNotFoundError(f"missing immutable v1 table: {path}")
        tables[name] = pd.read_csv(path)
    return tables


def hash_v1_inputs(v1_root: Path, result_root: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    v1 = Path(v1_root)
    protocol = v1 / "protocol" / "transition_analysis_protocol.json"
    final = v1 / "qa" / "FINAL_STATUS.json"
    for path in (protocol, final):
        if not path.is_file():
            raise FileNotFoundError(f"missing v1 artifact: {path}")
        out[str(path)] = sha256_file(path)
    tdir = v1 / "tables"
    for path in sorted(tdir.glob("*")):
        if path.is_file():
            out[str(path)] = sha256_file(path)
    combined = Path(result_root) / "analysis" / "combined_path_table.csv"
    if not combined.is_file():
        raise FileNotFoundError(f"missing combined path table: {combined}")
    out[str(combined)] = sha256_file(combined)
    return out


def assert_v1_pass(v1_root: Path) -> Dict[str, Any]:
    path = Path(v1_root) / "qa" / "FINAL_STATUS.json"
    blob = json.loads(path.read_text(encoding="utf-8"))
    if str(blob.get("status")) != "PASS":
        raise RuntimeError(f"transition_regions_v1 FINAL_STATUS is not PASS: {blob.get('status')}")
    return blob


def frozen_direct_span(span_df: pd.DataFrame) -> Tuple[float, float, str]:
    row = span_df.loc[span_df["family"] == "Direct"].iloc[0]
    status = str(row["status"])
    if status != "VALID_POSITIVE_INTERIOR_SPAN":
        raise RuntimeError(f"Direct v1 span status is {status}, expected VALID_POSITIVE_INTERIOR_SPAN")
    return float(row["rho_transition_low"]), float(row["rho_transition_high"]), status


def lofo_envelope(lofo: pd.DataFrame, family: str = "Direct") -> Dict[str, Any]:
    sub = lofo.loc[lofo["family"] == family].copy()
    valid = sub.loc[sub["valid_positive_interior_five_event_span"].astype(bool)]
    out = {
        "n_lofo": int(len(sub)),
        "n_valid": int(len(valid)),
        "envelope_low": None,
        "envelope_high": None,
        "valid_low_min": None,
        "valid_low_max": None,
        "valid_high_min": None,
        "valid_high_max": None,
        "sensitivity_diagnostic_only": True,
        "does_not_replace_frozen_cv_span": True,
    }
    if not valid.empty:
        out["valid_low_min"] = float(valid["rho_transition_low"].min())
        out["valid_low_max"] = float(valid["rho_transition_low"].max())
        out["valid_high_min"] = float(valid["rho_transition_high"].min())
        out["valid_high_max"] = float(valid["rho_transition_high"].max())
        out["envelope_low"] = out["valid_low_min"]
        out["envelope_high"] = out["valid_high_max"]
    return out


def rho_inside_lofo_envelope(rho: Optional[float], envelope: MappingLike) -> Optional[bool]:
    if rho is None or envelope.get("envelope_low") is None:
        return None
    return bool(rho_in_closed_span(float(rho), float(envelope["envelope_low"]), float(envelope["envelope_high"])))


def classify_direct_interpretation(regret_df: pd.DataFrame) -> str:
    """A/B/C from exact zero vs strictly positive regret. No magnitude threshold."""
    sub = regret_df.loc[regret_df["family"] == "Direct"]
    if sub.empty:
        return "D"
    flags = []
    for val in sub["raw_regret"].to_numpy(dtype=float):
        flags.append(bool(np.isfinite(val) and val > ATOL))
    if not flags:
        return "D"
    if not any(flags):
        return "A"
    if all(flags):
        return "B"
    return "C"


def split_suffix(split: str) -> str:
    if split == "cv_mean":
        return "CV_mean"
    if split.startswith("fold_"):
        return split
    if split == "heldout":
        return "heldout"
    if split in {"forward_2025", "2025"}:
        return "forward_2025"
    raise ValueError(f"unknown split {split!r}")


def metric_series(df: pd.DataFrame, metric: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
    rhos = df["rho"].to_numpy(dtype=float)
    col = f"{metric}__{split_suffix(split)}"
    if col not in df.columns:
        raise KeyError(col)
    vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    return rhos, vals


def expected_identity_payload() -> Dict[str, Any]:
    return dict(EXPECTED_IDENTITY)


def fmt_num(v: Any, nd: int = 6) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)) or pd.isna(v):
        return "NA"
    if isinstance(v, (bool, np.bool_)):
        return "true" if bool(v) else "false"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    try:
        return f"{float(v):.{nd}g}"
    except (TypeError, ValueError):
        return str(v)

