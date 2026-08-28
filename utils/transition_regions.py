"""Descriptive CV transition-region analysis of frozen 994-tree penalty paths.

No model fitting, no rho/family/configuration selection, and no manuscript writes.
Primary event locations are discrete observed grid points only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


class CanonicalIdentityError(RuntimeError):
    """Raised when the canonical 994-tree experiment identity is inconsistent."""


class TransitionProtocolError(RuntimeError):
    """Raised when the frozen protocol is missing, mutated, or violated."""


class OutputConfineError(RuntimeError):
    """Raised when a write would escape the dedicated output directory."""


class PathDataError(RuntimeError):
    """Raised when a required frozen path point is missing or non-finite."""


# ---------------------------------------------------------------------------
# Frozen protocol (must be written to disk before event extraction)
# ---------------------------------------------------------------------------

ATOL = 1e-12
RTOL = 1e-12
Q_ATOL = 1e-10
Q_RTOL = 1e-8

PRIMARY_METRICS: Tuple[Tuple[str, str], ...] = (
    ("R2_price", "max"),
    ("MAE_price", "min"),
    ("MAPE", "min"),
    ("RMSE_log", "min"),
    ("COD", "min"),
)

FAMILY_DISPLAY = ("Direct", "Surrogate")
FAMILY_MODEL = {
    "Direct": "LGBCovPenalty",
    "Surrogate": "LGBSmoothPenalty",
}
BASELINE_FAMILIES = ("Linear", "LightGBM")
BASELINE_MODEL = {
    "Linear": "LinearRegression",
    "LightGBM": "LGBMRegressor",
}
FOLD_IDS: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)
EVAL_SPLITS: Tuple[str, ...] = ("cv_mean", "heldout", "forward_2025")

METRIC_COLUMNS = {
    "R2_price": "R2_price",
    "MAE_price": "MAE_price",
    "MAPE": "MAPE",
    "RMSE_log": "RMSE_log",
    "COD": "COD",
    "Median ratio": "median_ratio",
    "Mean ratio": "mean_ratio",
    "W. Mean ratio": "weighted_mean_ratio",
    "PRD": "PRD",
    "PRB": "PRB",
    "VEI": "VEI",
    "MKI": "MKI",
    "Beta_log": "Beta_log",
    "Cov_log_residual_log_price": "Cov_log_residual_log_price",
    "Corr_log_residual_log_price": "Corr_log_residual_log_price",
    "dCor_e_y": "dCor_e_y",
    "Delta_NL": "Delta_NL",
}

BAND_METRICS: Tuple[str, ...] = (
    "R2_price",
    "MAE_price",
    "MAPE",
    "RMSE_log",
    "COD",
    "median_ratio",
    "mean_ratio",
    "weighted_mean_ratio",
    "PRD",
    "PRB",
    "VEI",
    "MKI",
    "Beta_log",
    "Cov_log_residual_log_price",
    "Corr_log_residual_log_price",
    "dCor_e_y",
    "Delta_NL",
)

ENVELOPE_METRICS: Tuple[str, ...] = (
    "R2_price",
    "MAE_price",
    "RMSE_log",
    "COD",
    "PRD",
    "PRB",
    "VEI",
    "Beta_log",
    "dCor_e_y",
)

HISTORICAL_500_ROOT_NAME = "paper_v6_preselection"
CANONICAL_ROOT_NAME = "paper_v6_preselection_994"

EXPECTED_IDENTITY: Dict[str, Any] = {
    "result_root_name": CANONICAL_ROOT_NAME,
    "baseline_gate": "ADOPT_994",
    "lgbm_config_id": "407d47775760c14d",
    "split_id": "3d464d4a611b131b",
    "data_id": "d4929d43ec19badf",
    "seed": 2025,
    "n_estimators": 994,
    "n_folds": 7,
    "n_linear": 1,
    "n_lgbm": 1,
    "n_direct": 51,
    "n_surrogate": 51,
    "n_configs": 104,
    "n_cv_fits": 728,
    "n_positive_rho": 50,
    "min_positive_rho": 0.1,
    "max_positive_rho": 100.0,
}

SPAN_OK_CLASSES = frozenset({"interior_positive", "numerical_plateau"})
SPAN_BLOCK_CLASSES = frozenset({"boundary_zero", "boundary_high", "ambiguous"})

FORBIDDEN_SPAN_DESCRIPTORS = (
    "optimal",
    "recommended",
    "selected",
    "safe",
    "deployment-ready",
    "preferred",
)

PROTOCOL: Dict[str, Any] = {
    "name": "transition_regions_v1",
    "descriptive_only": True,
    "no_model_fitting": True,
    "no_rho_or_family_selection": True,
    "span_label": "CV-derived descriptive transition span",
    "alternate_span_label": "CV transition span",
    "forbidden_span_descriptors": list(FORBIDDEN_SPAN_DESCRIPTORS),
    "primary_metrics": [{"metric": m, "event": d} for m, d in PRIMARY_METRICS],
    "event_estimation": "discrete_observed_canonical_grid",
    "not_used_for_events": [
        "smoothing_splines",
        "LOWESS",
        "polynomial_approximation",
        "changepoint_packages",
        "second_derivatives",
        "theory_derived_rho",
        "held_out_information",
        "forward_2025_information",
        "IAAO_thresholds",
        "Pareto_Nash_rules",
        "historical_500_tree_ranges",
        "six_county_rho_values",
    ],
    "numerical_equality": {
        "formula": "abs(a-b) <= atol + rtol * abs(b)",
        "atol": ATOL,
        "rtol": RTOL,
        "note": "Floating-point equality only; not a near-optimality tolerance.",
    },
    "q_equality": {"atol": Q_ATOL, "rtol": Q_RTOL},
    "event_classes": [
        "interior_positive",
        "boundary_zero",
        "boundary_high",
        "numerical_plateau",
        "ambiguous",
    ],
    "span_rule": (
        "If all five primary full-CV events are positive and interior "
        "(unique interior_positive or positive-interior numerical_plateau), "
        "rho_transition_low is the minimum lower event location and "
        "rho_transition_high is the maximum upper event location. "
        "Otherwise flag FULL_COMMON_SPAN_NOT_SUPPORTED and do not redefine the band."
    ),
    "log10_width": "log10(rho_transition_high) - log10(rho_transition_low)",
    "fraction_of_full_positive_log_grid": (
        "log10_width / (log10(max_positive_rho) - log10(min_positive_rho))"
    ),
    "folds_are_temporal_diagnostics": True,
    "no_iid_confidence_intervals": True,
    "no_t_tests": True,
    "heldout_2025_label": "retrospective temporal concordance",
    "not_prospective_confirmation": True,
    "q_beta": "Beta_log(rho) / Beta_log(rho=0); keep signed; flag overcorrection if q<0",
    "q_cov": (
        "Cov_log_residual_log_price(rho) / Cov_log_residual_log_price(rho=0); "
        "keep signed"
    ),
    "surrogate_q_name": "empirical remaining first-order fraction",
    "direct_q_name": "q_beta",
    "theory_does_not_set_events_or_span": True,
    "canonical_identity": EXPECTED_IDENTITY,
    "primary_metrics_are_not_independent_confirmations": True,
}


def protocol_canonical_json() -> str:
    return json.dumps(PROTOCOL, indent=2, sort_keys=True, default=str) + "\n"


def protocol_sha256() -> str:
    return hashlib.sha256(protocol_canonical_json().encode("utf-8")).hexdigest()


def expected_canonical_rhos() -> List[float]:
    positives = [float(x) for x in np.geomspace(0.1, 100.0, 50).tolist()]
    return [0.0] + positives


def numerically_equal(a: Any, b: Any, *, atol: float = ATOL, rtol: float = RTOL) -> bool:
    try:
        x = float(a)
        y = float(b)
    except (TypeError, ValueError):
        return False
    if not (np.isfinite(x) and np.isfinite(y)):
        return False
    return bool(abs(x - y) <= (float(atol) + float(rtol) * abs(y)))


def rho_sort_key(rho: float) -> float:
    return float(rho)


def is_rho_zero(rho: float, *, atol: float = ATOL, rtol: float = RTOL) -> bool:
    return numerically_equal(rho, 0.0, atol=atol, rtol=rtol)


def is_rho_positive(rho: float, *, atol: float = ATOL, rtol: float = RTOL) -> bool:
    try:
        x = float(rho)
    except (TypeError, ValueError):
        return False
    return np.isfinite(x) and x > 0.0 and not is_rho_zero(x, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Output confinement
# ---------------------------------------------------------------------------

class OutputGuard:
    """Permit writes only under a dedicated output root; never .tex or paper/."""

    def __init__(self, output_root: Path, repo_root: Path):
        self.output_root = Path(output_root).resolve()
        self.repo_root = Path(repo_root).resolve()
        self.paper_root = (self.repo_root / "paper").resolve()

    def allowed(self, path: Path) -> Path:
        dest = Path(path)
        if dest.suffix.lower() == ".tex":
            raise OutputConfineError(f"refusing to write .tex file: {dest}")
        resolved = dest if dest.is_absolute() else (self.output_root / dest)
        resolved = resolved.resolve()
        try:
            resolved.relative_to(self.output_root)
        except ValueError as err:
            raise OutputConfineError(
                f"refusing write outside output root {self.output_root}: {resolved}"
            ) from err
        try:
            resolved.relative_to(self.paper_root)
            raise OutputConfineError(f"refusing to write under paper/: {resolved}")
        except ValueError:
            pass
        if self.paper_root in resolved.parents:
            raise OutputConfineError(f"refusing to write under paper/: {resolved}")
        return resolved

    def ensure_subdir(self, rel: str) -> Path:
        dest = (self.output_root / rel).resolve()
        try:
            dest.relative_to(self.output_root)
        except ValueError as err:
            raise OutputConfineError(
                f"refusing mkdir outside output root {self.output_root}: {dest}"
            ) from err
        dest.mkdir(parents=True, exist_ok=True)
        return dest

    def write_bytes(self, path: Path, data: bytes) -> Path:
        dest = self.allowed(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return dest

    def write_text(self, path: Path, text: str, encoding: str = "utf-8") -> Path:
        return self.write_bytes(path, text.encode(encoding))

    def write_json(self, path: Path, payload: Any) -> Path:
        blob = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
        return self.write_text(path, blob)

    def write_df(self, df: pd.DataFrame, path: Path, *, parquet: bool = True) -> List[Path]:
        written: List[Path] = []
        csv_path = Path(path)
        if csv_path.suffix.lower() != ".csv":
            csv_path = csv_path.with_suffix(".csv")
        dest_csv = self.allowed(csv_path)
        dest_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(dest_csv, index=False)
        written.append(dest_csv)
        if parquet:
            pq = dest_csv.with_suffix(".parquet")
            dest_pq = self.allowed(pq)
            df.to_parquet(dest_pq, index=False)
            written.append(dest_pq)
        return written


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Discrete event extraction
# ---------------------------------------------------------------------------

@dataclass
class DiscreteEvent:
    metric: str
    direction: str
    classification: str
    rho_low: Optional[float]
    rho_high: Optional[float]
    metric_value: Optional[float]
    n_tied: int
    tied_rhos: List[float]
    prev_rho: Optional[float]
    prev_value: Optional[float]
    next_rho: Optional[float]
    next_value: Optional[float]
    local_turn_verified: bool
    expected_optimization_direction: str
    used_midpoint: bool = False
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["tied_rhos_json"] = json.dumps(self.tied_rhos)
        return d


def _finite_path(rhos: Sequence[float], values: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    r = np.asarray(rhos, dtype=float)
    v = np.asarray(values, dtype=float)
    if r.size == 0 or v.size == 0 or r.size != v.size:
        raise PathDataError("rho/value path is empty or misaligned")
    order = np.argsort(r, kind="mergesort")
    r = r[order]
    v = v[order]
    if not np.all(np.isfinite(r)):
        raise PathDataError("non-finite rho in path")
    return r, v


def extract_discrete_event(
    rhos: Sequence[float],
    values: Sequence[float],
    *,
    metric: str,
    direction: str,
    atol: float = ATOL,
    rtol: float = RTOL,
) -> DiscreteEvent:
    """Global discrete optimum on the actual grid, including rho=0."""
    if direction not in {"max", "min"}:
        raise ValueError(f"direction must be max or min, got {direction!r}")
    r, v = _finite_path(rhos, values)
    if not np.all(np.isfinite(v)):
        return DiscreteEvent(
            metric=metric,
            direction=direction,
            classification="ambiguous",
            rho_low=None,
            rho_high=None,
            metric_value=None,
            n_tied=0,
            tied_rhos=[],
            prev_rho=None,
            prev_value=None,
            next_rho=None,
            next_value=None,
            local_turn_verified=False,
            expected_optimization_direction=direction,
            notes="non_finite_metric_values",
        )

    best = float(np.max(v) if direction == "max" else np.min(v))
    tied_idx = [int(i) for i, val in enumerate(v) if numerically_equal(val, best, atol=atol, rtol=rtol)]
    if not tied_idx:
        return DiscreteEvent(
            metric=metric,
            direction=direction,
            classification="ambiguous",
            rho_low=None,
            rho_high=None,
            metric_value=best,
            n_tied=0,
            tied_rhos=[],
            prev_rho=None,
            prev_value=None,
            next_rho=None,
            next_value=None,
            local_turn_verified=False,
            expected_optimization_direction=direction,
            notes="no_tied_index",
        )

    contiguous = all(tied_idx[i + 1] == tied_idx[i] + 1 for i in range(len(tied_idx) - 1))
    i0 = min(tied_idx)
    i1 = max(tied_idx)
    tied_rhos = [float(r[i]) for i in tied_idx]
    rho_low = float(min(tied_rhos))
    rho_high = float(max(tied_rhos))
    prev_rho = float(r[i0 - 1]) if i0 > 0 else None
    prev_value = float(v[i0 - 1]) if i0 > 0 else None
    next_rho = float(r[i1 + 1]) if i1 < len(r) - 1 else None
    next_value = float(v[i1 + 1]) if i1 < len(r) - 1 else None

    def neighbor_worse(val: Optional[float]) -> bool:
        if val is None:
            return False
        if direction == "max":
            return (not numerically_equal(val, best, atol=atol, rtol=rtol)) and val < best
        return (not numerically_equal(val, best, atol=atol, rtol=rtol)) and val > best

    local_turn = (
        prev_value is not None
        and next_value is not None
        and neighbor_worse(prev_value)
        and neighbor_worse(next_value)
    )

    rho_max = float(np.max(r))
    has_zero = any(is_rho_zero(x, atol=atol, rtol=rtol) for x in tied_rhos)
    has_high = any(numerically_equal(x, rho_max, atol=atol, rtol=rtol) for x in tied_rhos)
    all_zero = all(is_rho_zero(x, atol=atol, rtol=rtol) for x in tied_rhos)
    all_high = all(numerically_equal(x, rho_max, atol=atol, rtol=rtol) for x in tied_rhos)
    all_interior_positive = (not has_zero) and (not has_high) and all(
        is_rho_positive(x, atol=atol, rtol=rtol) for x in tied_rhos
    )

    notes = []
    if not contiguous:
        classification = "ambiguous"
        notes.append("noncontiguous_tied_optima")
    elif all_zero:
        classification = "boundary_zero"
    elif all_high:
        classification = "boundary_high"
    elif has_zero or has_high:
        classification = "ambiguous"
        notes.append("tied_interval_touches_grid_boundary")
    elif len(tied_idx) > 1 and all_interior_positive:
        classification = "numerical_plateau"
        notes.append("positive_interior_tied_interval")
    elif len(tied_idx) == 1 and all_interior_positive:
        classification = "interior_positive"
    else:
        classification = "ambiguous"
        notes.append("unclassified")

    return DiscreteEvent(
        metric=metric,
        direction=direction,
        classification=classification,
        rho_low=rho_low,
        rho_high=rho_high,
        metric_value=best,
        n_tied=len(tied_idx),
        tied_rhos=tied_rhos,
        prev_rho=prev_rho,
        prev_value=prev_value,
        next_rho=next_rho,
        next_value=next_value,
        local_turn_verified=bool(local_turn),
        expected_optimization_direction=direction,
        used_midpoint=False,
        notes=";".join(notes),
    )


def event_supports_common_span(event: DiscreteEvent) -> bool:
    if event.classification not in SPAN_OK_CLASSES:
        return False
    if event.rho_low is None or event.rho_high is None:
        return False
    if not (is_rho_positive(event.rho_low) and is_rho_positive(event.rho_high)):
        return False
    return True


@dataclass
class TransitionSpan:
    family: str
    status: str
    rho_transition_low: Optional[float]
    rho_transition_high: Optional[float]
    log10_width: Optional[float]
    fraction_of_full_positive_log_grid: Optional[float]
    n_primary_events: int
    n_supporting_events: int
    blocking_metrics: List[str]
    plateau_metrics: List[str]
    min_positive_rho: float
    max_positive_rho: float
    used_tied_intervals: bool
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def log10_width(low: float, high: float) -> float:
    if not (is_rho_positive(low) and is_rho_positive(high)):
        raise ValueError("log10_width requires positive endpoints")
    if high < low and not numerically_equal(low, high):
        raise ValueError("rho_transition_high < rho_transition_low")
    return float(np.log10(high) - np.log10(low))


def fraction_of_full_positive_log_grid(
    width: float,
    min_positive_rho: float,
    max_positive_rho: float,
) -> float:
    denom = float(np.log10(max_positive_rho) - np.log10(min_positive_rho))
    if denom <= 0:
        raise ValueError("positive log-grid width must be positive")
    return float(width / denom)


def construct_transition_span(
    family: str,
    events: Sequence[DiscreteEvent],
    *,
    min_positive_rho: float,
    max_positive_rho: float,
) -> TransitionSpan:
    supporting = [e for e in events if event_supports_common_span(e)]
    blocking = [e.metric for e in events if not event_supports_common_span(e)]
    plateaus = [e.metric for e in events if e.classification == "numerical_plateau"]
    n = len(events)
    if n != len(PRIMARY_METRICS):
        return TransitionSpan(
            family=family,
            status="FULL_COMMON_SPAN_NOT_SUPPORTED",
            rho_transition_low=None,
            rho_transition_high=None,
            log10_width=None,
            fraction_of_full_positive_log_grid=None,
            n_primary_events=n,
            n_supporting_events=len(supporting),
            blocking_metrics=blocking + ["incorrect_primary_event_count"],
            plateau_metrics=plateaus,
            min_positive_rho=float(min_positive_rho),
            max_positive_rho=float(max_positive_rho),
            used_tied_intervals=bool(plateaus),
            notes="expected_five_primary_events",
        )
    if blocking:
        return TransitionSpan(
            family=family,
            status="FULL_COMMON_SPAN_NOT_SUPPORTED",
            rho_transition_low=None,
            rho_transition_high=None,
            log10_width=None,
            fraction_of_full_positive_log_grid=None,
            n_primary_events=n,
            n_supporting_events=len(supporting),
            blocking_metrics=blocking,
            plateau_metrics=plateaus,
            min_positive_rho=float(min_positive_rho),
            max_positive_rho=float(max_positive_rho),
            used_tied_intervals=bool(plateaus),
            notes="one_or_more_primary_events_are_boundary_or_ambiguous",
        )
    low = min(float(e.rho_low) for e in events)
    high = max(float(e.rho_high) for e in events)
    width = log10_width(low, high)
    frac = fraction_of_full_positive_log_grid(width, min_positive_rho, max_positive_rho)
    return TransitionSpan(
        family=family,
        status="VALID_POSITIVE_INTERIOR_SPAN",
        rho_transition_low=low,
        rho_transition_high=high,
        log10_width=width,
        fraction_of_full_positive_log_grid=frac,
        n_primary_events=n,
        n_supporting_events=len(supporting),
        blocking_metrics=[],
        plateau_metrics=plateaus,
        min_positive_rho=float(min_positive_rho),
        max_positive_rho=float(max_positive_rho),
        used_tied_intervals=bool(plateaus),
        notes="span_uses_full_tied_intervals_when_plateau_present"
        if plateaus
        else "unique_event_locations",
    )


def rho_in_closed_span(
    rho: float,
    low: float,
    high: float,
    *,
    atol: float = ATOL,
    rtol: float = RTOL,
) -> bool:
    if not np.isfinite(rho):
        return False
    if numerically_equal(rho, low, atol=atol, rtol=rtol) or numerically_equal(rho, high, atol=atol, rtol=rtol):
        return True
    return bool(low < rho < high)


def log10_distance_to_span(rho: float, low: float, high: float) -> Optional[float]:
    """Non-negative log10 distance to the nearest span boundary if outside.

    rho=0 cannot be represented in log-rho; returns None.
    """
    if rho_in_closed_span(rho, low, high):
        return 0.0
    if not is_rho_positive(rho):
        return None
    if rho < low:
        return float(np.log10(low) - np.log10(rho))
    return float(np.log10(rho) - np.log10(high))


def span_segment_mask(
    rhos: Sequence[float],
    low: float,
    high: float,
    *,
    atol: float = ATOL,
    rtol: float = RTOL,
) -> np.ndarray:
    """Boolean mask of an ordered-rho path that lies in the frozen CV span."""
    arr = np.asarray(list(rhos), dtype=float)
    return np.array([rho_in_closed_span(float(x), low, high, atol=atol, rtol=rtol) for x in arr], dtype=bool)


# ---------------------------------------------------------------------------
# Fold / LOFO / concordance
# ---------------------------------------------------------------------------

def summarize_fold_events_logrho(events: Sequence[DiscreteEvent]) -> Dict[str, Any]:
    classes = [e.classification for e in events]
    positive_lows = [float(e.rho_low) for e in events if e.rho_low is not None and is_rho_positive(e.rho_low)]
    positive_highs = [float(e.rho_high) for e in events if e.rho_high is not None and is_rho_positive(e.rho_high)]
    log_low = np.log10(positive_lows) if positive_lows else np.array([])
    log_high = np.log10(positive_highs) if positive_highs else np.array([])

    def _stats(arr: np.ndarray) -> Dict[str, Optional[float]]:
        if arr.size == 0:
            return {"median": None, "iqr": None, "min": None, "max": None}
        q1, q3 = np.percentile(arr, [25, 75])
        return {
            "median": float(np.median(arr)),
            "iqr": float(q3 - q1),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    low_stats = _stats(log_low)
    high_stats = _stats(log_high)
    return {
        "n_folds": len(events),
        "n_interior_positive": sum(c == "interior_positive" for c in classes),
        "n_boundary_zero": sum(c == "boundary_zero" for c in classes),
        "n_boundary_high": sum(c == "boundary_high" for c in classes),
        "n_numerical_plateau": sum(c == "numerical_plateau" for c in classes),
        "n_ambiguous": sum(c == "ambiguous" for c in classes),
        "log10_rho_low_median": low_stats["median"],
        "log10_rho_low_iqr": low_stats["iqr"],
        "log10_rho_low_min": low_stats["min"],
        "log10_rho_low_max": low_stats["max"],
        "log10_rho_high_median": high_stats["median"],
        "log10_rho_high_iqr": high_stats["iqr"],
        "log10_rho_high_min": high_stats["min"],
        "log10_rho_high_max": high_stats["max"],
        "log10_stats_exclude_nonpositive_endpoints": True,
        "folds_are_temporal_diagnostics": True,
    }


def lofo_means(fold_matrix: np.ndarray, omit_index: int) -> np.ndarray:
    """Equal-weight mean of remaining folds. fold_matrix shape (n_rho, n_folds)."""
    x = np.asarray(fold_matrix, dtype=float)
    if x.ndim != 2:
        raise ValueError("fold_matrix must be 2-D")
    if omit_index < 0 or omit_index >= x.shape[1]:
        raise IndexError("omit_index out of range")
    keep = [i for i in range(x.shape[1]) if i != omit_index]
    return np.mean(x[:, keep], axis=1)


def q_ratio(value: float, value_at_zero: float) -> Tuple[Optional[float], str]:
    if not (np.isfinite(value) and np.isfinite(value_at_zero)):
        return None, "non_finite"
    if numerically_equal(value_at_zero, 0.0, atol=Q_ATOL, rtol=Q_RTOL):
        return None, "undefined_zero_denominator"
    q = float(value / value_at_zero)
    note = "ok"
    if q < 0:
        note = "overcorrection_sign_flip"
    return q, note


def q_beta_cov_agree(q_beta: Optional[float], q_cov: Optional[float]) -> Tuple[bool, Optional[float], str]:
    if q_beta is None or q_cov is None:
        return False, None, "missing_q"
    delta = float(q_beta - q_cov)
    ok = bool(abs(delta) <= (Q_ATOL + Q_RTOL * abs(q_cov)))
    return ok, delta, "ok" if ok else "q_beta_q_cov_disagree"


def attenuation(q: Optional[float]) -> Optional[float]:
    if q is None:
        return None
    return float(1.0 - q)


# ---------------------------------------------------------------------------
# Combined-table helpers
# ---------------------------------------------------------------------------

def metric_col(metric: str, suffix: str) -> str:
    return f"{metric}__{suffix}"


def fold_col(metric: str, fold_id: int) -> str:
    return f"{metric}__fold_{int(fold_id)}"


def family_frame(combined: pd.DataFrame, family: str) -> pd.DataFrame:
    sub = combined.loc[combined["family"].astype(str) == str(family)].copy()
    sub = sub.sort_values("rho", kind="mergesort").reset_index(drop=True)
    return sub


def path_from_frame(df: pd.DataFrame, metric: str, suffix: str) -> Tuple[np.ndarray, np.ndarray]:
    col = metric_col(metric, suffix)
    if "rho" not in df.columns or col not in df.columns:
        raise PathDataError(f"missing rho or {col}")
    rhos = df["rho"].to_numpy(dtype=float)
    vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    return rhos, vals


def fold_matrix_from_frame(df: pd.DataFrame, metric: str) -> Tuple[np.ndarray, np.ndarray]:
    rhos = df["rho"].to_numpy(dtype=float)
    cols = []
    for fid in FOLD_IDS:
        col = fold_col(metric, fid)
        if col not in df.columns:
            raise PathDataError(f"missing {col}")
        cols.append(pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float))
    return rhos, np.column_stack(cols)


def extract_primary_events_from_path(
    rhos: Sequence[float],
    value_lookup: Mapping[str, Sequence[float]],
) -> List[DiscreteEvent]:
    events = []
    for metric, direction in PRIMARY_METRICS:
        if metric not in value_lookup:
            raise PathDataError(f"missing primary metric {metric}")
        events.append(
            extract_discrete_event(rhos, value_lookup[metric], metric=metric, direction=direction)
        )
    return events


def extract_primary_events_from_frame(df: pd.DataFrame, suffix: str) -> List[DiscreteEvent]:
    rhos = df["rho"].to_numpy(dtype=float)
    lookup = {}
    for metric, _direction in PRIMARY_METRICS:
        col = metric_col(metric, suffix)
        if col not in df.columns:
            raise PathDataError(f"missing {col}")
        lookup[metric] = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    return extract_primary_events_from_path(rhos, lookup)


def lofo_events_and_span(
    df: pd.DataFrame,
    family: str,
    *,
    min_positive_rho: float,
    max_positive_rho: float,
) -> pd.DataFrame:
    rows = []
    rhos = df["rho"].to_numpy(dtype=float)
    matrices = {metric: fold_matrix_from_frame(df, metric)[1] for metric, _d in PRIMARY_METRICS}
    for omit_i, fold_id in enumerate(FOLD_IDS):
        lookup = {metric: lofo_means(mat, omit_i) for metric, mat in matrices.items()}
        events = extract_primary_events_from_path(rhos, lookup)
        span = construct_transition_span(
            family,
            events,
            min_positive_rho=min_positive_rho,
            max_positive_rho=max_positive_rho,
        )
        rec: Dict[str, Any] = {
            "family": family,
            "omitted_fold_id": int(fold_id),
            "n_remaining_folds": len(FOLD_IDS) - 1,
            "span_status": span.status,
            "valid_positive_interior_five_event_span": span.status == "VALID_POSITIVE_INTERIOR_SPAN",
            "rho_transition_low": span.rho_transition_low,
            "rho_transition_high": span.rho_transition_high,
            "log10_width": span.log10_width,
            "blocking_metrics": ",".join(span.blocking_metrics),
        }
        for ev in events:
            rec[f"{ev.metric}__classification"] = ev.classification
            rec[f"{ev.metric}__rho_low"] = ev.rho_low
            rec[f"{ev.metric}__rho_high"] = ev.rho_high
            rec[f"{ev.metric}__value"] = ev.metric_value
        rows.append(rec)
    return pd.DataFrame(rows)


def lofo_span_summary(lofo: pd.DataFrame) -> Dict[str, Any]:
    valid = lofo.loc[lofo["valid_positive_interior_five_event_span"].astype(bool)]
    out: Dict[str, Any] = {
        "n_lofo": int(len(lofo)),
        "n_valid_all_five_interior": int(len(valid)),
        "rho_low_min": None,
        "rho_low_max": None,
        "rho_high_min": None,
        "rho_high_max": None,
        "log10_width_min": None,
        "log10_width_max": None,
    }
    if not valid.empty:
        out["rho_low_min"] = float(valid["rho_transition_low"].min())
        out["rho_low_max"] = float(valid["rho_transition_low"].max())
        out["rho_high_min"] = float(valid["rho_transition_high"].min())
        out["rho_high_max"] = float(valid["rho_transition_high"].max())
        out["log10_width_min"] = float(valid["log10_width"].min())
        out["log10_width_max"] = float(valid["log10_width"].max())
    return out


def concordance_row(
    family: str,
    split: str,
    event: DiscreteEvent,
    span: TransitionSpan,
) -> Dict[str, Any]:
    inside: Optional[bool] = None
    dist: Optional[float] = None
    membership_note = "cv_span_not_defined"
    if span.status == "VALID_POSITIVE_INTERIOR_SPAN" and span.rho_transition_low is not None:
        if event.rho_low is None or event.rho_high is None:
            inside = False
            membership_note = "event_location_undefined"
        else:
            inside_low = rho_in_closed_span(event.rho_low, span.rho_transition_low, span.rho_transition_high)
            inside_high = rho_in_closed_span(event.rho_high, span.rho_transition_low, span.rho_transition_high)
            inside = bool(inside_low and inside_high)
            if inside:
                dist = 0.0
                membership_note = "inside_frozen_cv_span"
            else:
                d1 = log10_distance_to_span(event.rho_low, span.rho_transition_low, span.rho_transition_high)
                d2 = log10_distance_to_span(event.rho_high, span.rho_transition_low, span.rho_transition_high)
                candidates = [d for d in (d1, d2) if d is not None]
                dist = float(min(candidates)) if candidates else None
                membership_note = "outside_frozen_cv_span"
    return {
        "family": family,
        "split": split,
        "metric": event.metric,
        "direction": event.direction,
        "classification": event.classification,
        "rho_low": event.rho_low,
        "rho_high": event.rho_high,
        "metric_value": event.metric_value,
        "inside_frozen_cv_span": inside,
        "log10_distance_to_nearest_cv_span_boundary": dist,
        "membership_note": membership_note,
        "cv_span_status": span.status,
        "cv_rho_transition_low": span.rho_transition_low,
        "cv_rho_transition_high": span.rho_transition_high,
        "retrospective_temporal_concordance": True,
        "not_prospective_confirmation": True,
    }


# ---------------------------------------------------------------------------
# Canonical identity
# ---------------------------------------------------------------------------

def assert_not_historical_500(result_root: Path) -> None:
    root = Path(result_root).resolve()
    if root.name == HISTORICAL_500_ROOT_NAME:
        raise CanonicalIdentityError(
            f"historical 500-tree root is forbidden: {root}. "
            f"Use only {CANONICAL_ROOT_NAME}."
        )
    if HISTORICAL_500_ROOT_NAME in root.parts and CANONICAL_ROOT_NAME not in root.parts:
        raise CanonicalIdentityError(
            f"path uses historical 500-tree root component: {root}"
        )


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise CanonicalIdentityError(f"missing required identity file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def rhos_match_canonical(got: Sequence[float], expected: Sequence[float]) -> Tuple[bool, List[float], List[float]]:
    got_l = [float(x) for x in got]
    exp_l = [float(x) for x in expected]
    missing = [e for e in exp_l if not any(numerically_equal(e, g) for g in got_l)]
    extra = [g for g in got_l if not any(numerically_equal(g, e) for e in exp_l)]
    return (not missing) and (not extra) and len(got_l) == len(exp_l), missing, extra


def validate_combined_counts(combined: pd.DataFrame, expected_rhos: Sequence[float]) -> Dict[str, Any]:
    problems: List[str] = []
    counts = combined["family"].astype(str).value_counts().to_dict()
    expect_counts = {
        "Linear": EXPECTED_IDENTITY["n_linear"],
        "LightGBM": EXPECTED_IDENTITY["n_lgbm"],
        "Direct": EXPECTED_IDENTITY["n_direct"],
        "Surrogate": EXPECTED_IDENTITY["n_surrogate"],
    }
    for fam, n in expect_counts.items():
        got = int(counts.get(fam, 0))
        if got != int(n):
            problems.append(f"{fam} count {got} != {n}")
    if int(len(combined)) != int(EXPECTED_IDENTITY["n_configs"]):
        problems.append(f"total configs {len(combined)} != {EXPECTED_IDENTITY['n_configs']}")
    rho_checks = {}
    for fam in FAMILY_DISPLAY:
        sub = family_frame(combined, fam)
        rhos = [float(x) for x in sub["rho"].tolist()]
        ok, missing, extra = rhos_match_canonical(rhos, expected_rhos)
        rho_checks[fam] = {"ok": ok, "n": len(rhos), "missing": missing, "extra": extra}
        if not ok:
            problems.append(f"{fam} rho grid mismatch missing={missing[:5]} extra={extra[:5]}")
        if sub["rho"].duplicated().any():
            problems.append(f"{fam} duplicate rho values")
    data_ids = set(combined["data_id"].astype(str)) if "data_id" in combined.columns else set()
    split_ids = set(combined["split_id"].astype(str)) if "split_id" in combined.columns else set()
    if data_ids != {EXPECTED_IDENTITY["data_id"]}:
        problems.append(f"combined data_id {data_ids!r} != {EXPECTED_IDENTITY['data_id']}")
    if split_ids != {EXPECTED_IDENTITY["split_id"]}:
        problems.append(f"combined split_id {split_ids!r} != {EXPECTED_IDENTITY['split_id']}")
    return {
        "family_counts": {k: int(v) for k, v in counts.items()},
        "rho_checks": rho_checks,
        "problems": problems,
        "ok": not problems,
    }


def validate_canonical_result_root(result_root: Path) -> Dict[str, Any]:
    root = Path(result_root).resolve()
    assert_not_historical_500(root)
    if root.name != CANONICAL_ROOT_NAME:
        raise CanonicalIdentityError(
            f"result root name {root.name!r} != {CANONICAL_ROOT_NAME!r}"
        )

    baseline_gate = _load_json(root / "baseline_gate.json")
    experiment_manifest = _load_json(root / "experiment_manifest.json")
    lgbm_config = _load_json(root / "lgbm_config.json")
    frozen_baseline = _load_json(root / "frozen_baseline.json")
    cv_completion = _load_json(root / "cv_completion.json")

    problems: List[str] = []

    def expect(label: str, got: Any, want: Any) -> None:
        if got != want:
            problems.append(f"{label}: got {got!r} want {want!r}")

    expect("baseline_gate.decision", baseline_gate.get("decision"), EXPECTED_IDENTITY["baseline_gate"])
    expect("lgbm_config.config_id", lgbm_config.get("config_id"), EXPECTED_IDENTITY["lgbm_config_id"])
    expect("frozen_baseline.config_id", frozen_baseline.get("config_id"), EXPECTED_IDENTITY["lgbm_config_id"])
    expect("baseline_gate.config_id_994", baseline_gate.get("config_id_994"), EXPECTED_IDENTITY["lgbm_config_id"])
    expect("cv_completion.split_id", cv_completion.get("split_id"), EXPECTED_IDENTITY["split_id"])
    expect("baseline_gate.comparability.split_id", (baseline_gate.get("comparability") or {}).get("split_id"), EXPECTED_IDENTITY["split_id"])
    expect("cv_completion.data_id", cv_completion.get("data_id"), EXPECTED_IDENTITY["data_id"])
    expect("baseline_gate.comparability.experiment_data_id", (baseline_gate.get("comparability") or {}).get("experiment_data_id"), EXPECTED_IDENTITY["data_id"])
    expect("lgbm_config.n_estimators", int(lgbm_config.get("n_estimators", -1)), EXPECTED_IDENTITY["n_estimators"])
    params = lgbm_config.get("lgbm_params") or {}
    expect("lgbm_params.n_estimators", int(params.get("n_estimators", -1)), EXPECTED_IDENTITY["n_estimators"])
    expect("lgbm_params.random_state", int(params.get("random_state", -1)), EXPECTED_IDENTITY["seed"])
    expect("frozen_baseline.seed", int(frozen_baseline.get("seed", -1)), EXPECTED_IDENTITY["seed"])
    expect("cv_completion.n_expected_pairs", int(cv_completion.get("n_expected_pairs", -1)), EXPECTED_IDENTITY["n_cv_fits"])
    expect("cv_completion.n_completed_pairs", int(cv_completion.get("n_completed_pairs", -1)), EXPECTED_IDENTITY["n_cv_fits"])
    expect("cv_completion.status", cv_completion.get("status"), "complete")
    if cv_completion.get("failed_config_fold"):
        problems.append("cv_completion has failed_config_fold")
    if cv_completion.get("missing_valid_config_fold"):
        problems.append("cv_completion has missing_valid_config_fold")
    grid = experiment_manifest.get("canonical_grid") or {}
    expect("canonical_grid.LinearRegression", int(grid.get("LinearRegression", -1)), 1)
    expect("canonical_grid.LGBMRegressor", int(grid.get("LGBMRegressor", -1)), 1)
    expect("canonical_grid.LGBCovPenalty", int(grid.get("LGBCovPenalty", -1)), 51)
    expect("canonical_grid.LGBSmoothPenalty", int(grid.get("LGBSmoothPenalty", -1)), 51)
    expect("canonical_grid.total_configs", int(grid.get("total_configs", -1)), 104)
    expect("canonical_grid.expected_cv_fits", int(grid.get("expected_cv_fits", -1)), 728)
    expect("canonical_grid.folds", int(grid.get("folds", -1)), 7)
    folds = cv_completion.get("expected_fold_ids") or []
    if list(folds) != [0, 1, 2, 3, 4, 5, 6]:
        problems.append(f"expected_fold_ids {folds!r} != [0..6]")

    if problems:
        raise CanonicalIdentityError("canonical identity failed: " + "; ".join(problems))

    return {
        "ok": True,
        "result_root": str(root),
        "data_id": EXPECTED_IDENTITY["data_id"],
        "split_id": EXPECTED_IDENTITY["split_id"],
        "lgbm_config_id": EXPECTED_IDENTITY["lgbm_config_id"],
        "baseline_gate": EXPECTED_IDENTITY["baseline_gate"],
        "n_estimators": EXPECTED_IDENTITY["n_estimators"],
        "seed": EXPECTED_IDENTITY["seed"],
        "n_cv_fits": EXPECTED_IDENTITY["n_cv_fits"],
        "problems": [],
        "baseline_gate_json": baseline_gate,
        "experiment_manifest": experiment_manifest,
        "lgbm_config": lgbm_config,
        "frozen_baseline": frozen_baseline,
        "cv_completion_status": cv_completion.get("status"),
        "cv_n_completed_pairs": cv_completion.get("n_completed_pairs"),
        "heldout_identity": {
            "split_id": EXPECTED_IDENTITY["split_id"],
            "evaluation": "heldout",
        },
        "forward_2025_identity": {
            "split_id": EXPECTED_IDENTITY["split_id"],
            "evaluation": "forward_2025",
        },
    }


def default_output_root(result_root: Path, data_id: str, split_id: str) -> Path:
    return (
        Path(result_root).resolve()
        / "analysis"
        / f"data_id={data_id}"
        / f"split_id={split_id}"
        / "penalty_path_analysis"
        / "transition_regions_v1"
    )


def load_combined_path_table(result_root: Path) -> Path:
    path = Path(result_root).resolve() / "analysis" / "combined_path_table.csv"
    if not path.is_file():
        raise CanonicalIdentityError(f"missing canonical combined path table: {path}")
    return path


def assert_cv_mean_is_equal_weight(combined: pd.DataFrame, metrics: Sequence[str]) -> List[str]:
    problems = []
    for fam in FAMILY_DISPLAY:
        sub = family_frame(combined, fam)
        for metric in metrics:
            mean_col = metric_col(metric, "CV_mean")
            if mean_col not in sub.columns:
                problems.append(f"missing {mean_col}")
                continue
            rhos, mat = fold_matrix_from_frame(sub, metric)
            recomputed = np.mean(mat, axis=1)
            stored = pd.to_numeric(sub[mean_col], errors="coerce").to_numpy(dtype=float)
            if not np.allclose(recomputed, stored, atol=1e-10, rtol=1e-10, equal_nan=True):
                problems.append(f"{fam} {metric} CV_mean is not equal-weight mean of seven folds")
    return problems


def df_to_markdown(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_empty table_"
    cols = [str(c) for c in df.columns]

    def cell(v: Any) -> str:
        if v is None or (isinstance(v, float) and not np.isfinite(v)) or pd.isna(v):
            return ""
        if isinstance(v, (float, np.floating)):
            return f"{float(v):.8g}"
        if isinstance(v, (bool, np.bool_)):
            return "true" if bool(v) else "false"
        text = str(v).replace("|", "\\|").replace("\n", " ")
        return text

    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(cell(row[c]) for c in df.columns) + " |")
    return "\n".join(lines)


def event_table_row(family: str, split: str, event: DiscreteEvent, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    rec = {
        "family": family,
        "split": split,
        "metric": event.metric,
        "expected_optimization_direction": event.expected_optimization_direction,
        "classification": event.classification,
        "rho_low": event.rho_low,
        "rho_high": event.rho_high,
        "n_tied": event.n_tied,
        "tied_rhos_json": json.dumps(event.tied_rhos),
        "metric_value": event.metric_value,
        "prev_rho": event.prev_rho,
        "prev_value": event.prev_value,
        "next_rho": event.next_rho,
        "next_value": event.next_value,
        "local_turn_verified": event.local_turn_verified,
        "used_midpoint": event.used_midpoint,
        "notes": event.notes,
    }
    if extra:
        rec.update(extra)
    return rec
