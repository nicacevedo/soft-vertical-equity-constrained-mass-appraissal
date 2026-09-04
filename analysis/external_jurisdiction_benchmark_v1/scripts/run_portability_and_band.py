#!/usr/bin/env python3
"""Steps 10-12: normalization-portability test, cross-jurisdiction candidate
band, and achieved-mechanism anchors.

Reads candidate_regions/candidate_regions.csv (Step 8-9 output). For each
jurisdiction's endpoint, raw_rho = rho_tilde_endpoint / mean_Var_training(y)
across that jurisdiction's own folds (each jurisdiction's own representative
training variance, read back from its normalized CV path file) -- this is
the "raw rho equivalent" the portability test compares against.

Does not claim normalization improves portability unless dispersion in
log10(rho_tilde) is genuinely lower than in log10(raw rho).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from analysis.external_jurisdiction_benchmark_v1.scripts.v1_common import ANALYSIS, write_json  # noqa: E402

BERRY_ANCHORED = {"wayne", "philadelphia", "st_louis_county"}


def mean_training_vy(county_key: str, family: str) -> float | None:
    p = ANALYSIS / "cv" / f"{county_key}_{family}_normalized_cv_path_summary.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if "Var_training_y" not in df.columns or not len(df):
        return None
    return float(df.drop_duplicates("fold")["Var_training_y"].mean())


def dispersion(values: np.ndarray) -> dict:
    v = values[np.isfinite(values)]
    if len(v) < 2:
        return {"n": int(len(v)), "sd": None, "iqr": None, "range": None}
    return {
        "n": int(len(v)), "sd": float(np.std(v, ddof=1)),
        "iqr": float(np.percentile(v, 75) - np.percentile(v, 25)),
        "range": float(v.max() - v.min()),
    }


COVERAGE_GRID_POINTS = 200
MAJORITY_BAND_SEMANTICS = (
    ">=75% coverage is continuous interval membership (activity <= r <= guardrail) "
    "evaluated on a 200-point geometric grid spanning [min activity, max guardrail] "
    "of the included regions. It is NOT coverage counted only at the 33 tested CV "
    "rho_tilde points. The intersection uses the continuous endpoints "
    "(max activity, min guardrail) exactly."
)
PROTOCOL_VALID_STATUS = "CANDIDATE_REGION"


def _fmt_interval(pair: list[float] | None) -> str | None:
    if pair is None:
        return None
    return f"[{pair[0]}, {pair[1]}]"


def intersection_status(intersection: list[float] | None, n_with_region: int) -> str:
    if n_with_region == 0 or intersection is None:
        return "NO_INTERSECTION"
    lo, hi = intersection
    if not np.isfinite(lo) or not np.isfinite(hi) or hi < lo:
        return "NO_INTERSECTION"
    # Knife-edge: width is numerically a single grid/endpoint value, not a band.
    if hi <= lo * (1.0 + 1e-12):
        return "NO_NONDEGENERATE_ALL_JURISDICTION_INTERSECTION"
    return "NONEMPTY_INTERSECTION"


def region_overlap(regions: list[tuple[float, float]]) -> dict:
    """Intersection of closed intervals and >=75% coverage on an evaluation grid.

    `regions` must already be the intended sample (protocol-valid, or a labeled
    sensitivity sample). Empty/inverted intervals are dropped, not repaired.
    """
    valid = [(float(a), float(b)) for a, b in regions if a is not None and b is not None
             and np.isfinite(a) and np.isfinite(b) and a <= b]
    if not valid:
        return {
            "intersection": None, "n_with_region": 0,
            "intersection_status": "NO_INTERSECTION",
            "majority_band_75pct": None,
            "majority_band_semantics": MAJORITY_BAND_SEMANTICS,
            "coverage_curve": {"rho_tilde": [], "coverage": []},
        }
    max_activity = max(a for a, _ in valid)
    min_guardrail = min(b for _, b in valid)
    intersection = [max_activity, min_guardrail] if max_activity <= min_guardrail else None
    lo = min(a for a, _ in valid)
    hi = max(b for _, b in valid)
    grid = np.geomspace(max(lo, 1e-6), hi, COVERAGE_GRID_POINTS)
    coverage = np.array([sum(1 for a, b in valid if a <= r <= b) / len(valid) for r in grid])
    band_mask = coverage >= 0.75
    band = None
    if band_mask.any():
        idx = np.where(band_mask)[0]
        band = [float(grid[idx[0]]), float(grid[idx[-1]])]
    return {
        "intersection": intersection, "n_with_region": len(valid),
        "intersection_status": intersection_status(intersection, len(valid)),
        "majority_band_75pct": band,
        "majority_band_semantics": MAJORITY_BAND_SEMANTICS,
        "coverage_curve": {"rho_tilde": grid.tolist(), "coverage": coverage.tolist()},
    }


def _portability_for(frame: pd.DataFrame, family: str, endpoint: str, sample: str) -> dict | None:
    rho_tilde_vals, raw_rho_vals, keys = [], [], []
    for _, r in frame.iterrows():
        if pd.isna(r[endpoint]):
            continue
        vy = mean_training_vy(r.county_key, family)
        if vy is None or vy <= 0:
            continue
        rho_tilde_vals.append(float(r[endpoint]))
        raw_rho_vals.append(float(r[endpoint]) / vy)
        keys.append(r.county_key)
    if not keys:
        return None
    d_norm = dispersion(np.log10(np.array(rho_tilde_vals)))
    d_raw = dispersion(np.log10(np.array(raw_rho_vals)))
    improves = (
        d_norm["sd"] is not None and d_raw["sd"] is not None and d_norm["sd"] < d_raw["sd"]
    )
    return {
        "family": family, "endpoint": endpoint, "sample": sample,
        "jurisdictions": ",".join(keys),
        "n": d_norm["n"], "sd_log10_rho_tilde": d_norm["sd"], "sd_log10_raw_rho": d_raw["sd"],
        "iqr_log10_rho_tilde": d_norm["iqr"], "iqr_log10_raw_rho": d_raw["iqr"],
        "range_log10_rho_tilde": d_norm["range"], "range_log10_raw_rho": d_raw["range"],
        "normalization_reduces_dispersion": improves,
    }


def _band_row(family: str, subset: str, region_rule: str, result: dict) -> dict:
    inter = result.get("intersection")
    band = result.get("majority_band_75pct")
    return {
        "family": family, "subset": subset, "region_rule": region_rule,
        "n_with_region": result.get("n_with_region", 0),
        "intersection": _fmt_interval(inter),
        "intersection_status": result.get("intersection_status"),
        "majority_band_75pct": _fmt_interval(band),
        "majority_band_semantics": result.get("majority_band_semantics"),
    }


def main() -> int:
    cand = pd.read_csv(ANALYSIS / "candidate_regions" / "candidate_regions.csv")
    portability_rows = []
    band_rows = []

    for family in ("direct", "surrogate"):
        sub = cand.loc[cand.family == family].copy()
        protocol_valid = sub.loc[sub["status"] == PROTOCOL_VALID_STATUS].copy()
        for endpoint in ("activity_rho_tilde", "guardrail_rho_tilde"):
            # Point-estimate portability uses every finite endpoint, including
            # jurisdictions whose interval is empty or LOFO-unstable. That is a
            # statement about the scale of detected onsets, not about a valid
            # candidate region.
            row = _portability_for(sub, family, endpoint, "all_point_estimates")
            if row:
                portability_rows.append(row)
            row = _portability_for(protocol_valid, family, endpoint, "protocol_valid_regions")
            if row:
                portability_rows.append(row)

        def regions_for(frame):
            return [
                (row.activity_rho_tilde, row.guardrail_rho_tilde)
                for row in frame.itertuples()
                if pd.notna(row.activity_rho_tilde) and pd.notna(row.guardrail_rho_tilde)
            ]

        # Protocol-valid intersection: only nonempty, LOFO-stable candidate regions.
        valid_all = region_overlap(regions_for(protocol_valid))
        valid_berry = region_overlap(regions_for(protocol_valid.loc[protocol_valid.county_key.isin(BERRY_ANCHORED)]))
        valid_no_miami = region_overlap(regions_for(protocol_valid.loc[protocol_valid.county_key != "miami_dade"]))
        # Sensitivity: numerical point-estimate endpoints even when the cell is
        # not a protocol-valid region (e.g. Allegheny Direct LOFO-unstable).
        # Inverted intervals (activity > guardrail) are still excluded.
        sens_all = region_overlap(regions_for(sub))

        band_rows.append(_band_row(family, "all_primary", "protocol_valid", valid_all))
        band_rows.append(_band_row(family, "berry_anchored", "protocol_valid", valid_berry))
        band_rows.append(_band_row(family, "excl_miami_dade", "protocol_valid", valid_no_miami))
        band_rows.append(_band_row(family, "all_primary", "point_estimate_sensitivity", sens_all))
        write_json(
            ANALYSIS / "candidate_regions" / f"{family}_coverage_curve.json",
            {
                **valid_all.get("coverage_curve", {}),
                "region_rule": "protocol_valid",
                "semantics": MAJORITY_BAND_SEMANTICS,
            },
        )

    ANALYSIS.joinpath("tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(portability_rows).to_csv(ANALYSIS / "tables" / "normalization_portability.csv", index=False)
    pd.DataFrame(band_rows).to_csv(ANALYSIS / "candidate_regions" / "cross_jurisdiction_band.csv", index=False)
    print(pd.DataFrame(portability_rows).to_string(index=False))
    print()
    print(pd.DataFrame(band_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
