"""Synthetic tests for the generic v2 rho-screening engine.

Arrays only; no ML model fitting. Geometric rho grids are intentionally
unlike current CCAO endpoints.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils.rho_screening_v2 import (  # noqa: E402
    BENEFIT_METRICS,
    SCALE_EQUIVARIANCE_FACTOR,
    log10_rho,
    min_segment_points,
    screen_positive_path,
    select_pwl,
    surrogate_upper_guardrail,
)


def _pwl(x: np.ndarray, breaks: list, slopes: list, intercept: float) -> np.ndarray:
    y = intercept + slopes[0] * x
    for b, s_prev, s_next in zip(breaks, slopes[:-1], slopes[1:]):
        y = y + (s_next - s_prev) * np.maximum(x - b, 0.0)
    return y


def _geo_rho(n: int = 40, start: float = 0.003, ratio: float = 1.22) -> np.ndarray:
    return start * (ratio ** np.arange(n, dtype=float))


def _benefit_bundle(x: np.ndarray, onset_x: float, sat_x: float) -> dict:
    dist = _pwl(x, [onset_x, sat_x], [0.015, -0.18, -0.02], 0.55)
    return {
        "PRD": 1.0 + dist,
        "PRB": -dist,
        "MKI": 1.0 + dist,
        "VEI": dist,
        "Beta_log": -dist,
    }


def _predictive_bundle(
    x: np.ndarray,
    early_x: float,
    late_x: float,
    *,
    r2_0: float = 0.90,
    mae_0: float = 100.0,
    mape_0: float = 0.20,
    rmse_0: float = 0.30,
) -> tuple:
    early_cost = _pwl(x, [early_x], [0.01, 0.55], 0.02)
    late_cost = _pwl(x, [late_x], [0.00, 0.70], 0.01)
    raw = {
        "R2_price": r2_0 - early_cost,
        "MAE_price": mae_0 + early_cost * 80.0,
        "MAPE": mape_0 + late_cost * 0.15,
        "RMSE_log": rmse_0 + late_cost * 0.20,
    }
    rho0 = {"R2_price": r2_0, "MAE_price": mae_0, "MAPE": mape_0, "RMSE_log": rmse_0}
    return raw, rho0


def _interp_path(x: np.ndarray, x_knots: list, y_knots: list) -> np.ndarray:
    return np.interp(x, np.asarray(x_knots, dtype=float), np.asarray(y_knots, dtype=float))


def test_min_segment_is_grid_length_rule():
    assert min_segment_points(40) == 5
    assert min_segment_points(200) == 10
    assert min_segment_points(3) == 5


def test_a_direct_like_early_predictive_cluster_not_saturation():
    rho = _geo_rho()
    x = log10_rho(rho)
    onset_x = x[8]
    sat_x = x[32]
    early_x = x[16]
    late_x = x[28]
    benefit = _benefit_bundle(x, onset_x, sat_x)
    pred_raw, pred0 = _predictive_bundle(x, early_x, late_x)
    out = screen_positive_path(rho, benefit_raw=benefit, predictive_raw=pred_raw, predictive_rho0=pred0)
    assert out["activity"]["index"] is not None
    assert out["activity"]["n_active"] >= 3
    gidx = out["index_predictive_guardrail"]
    assert gidx is not None
    assert gidx <= 18
    sat_idx = [out["benefit_events"][m].get("benefit_saturation_index") for m in BENEFIT_METRICS]
    sat_idx = [i for i in sat_idx if i is not None]
    assert sat_idx, "synthetic path should support later saturation"
    assert gidx < min(sat_idx)
    assert out["predictive_cluster"]["status"] == "OK"
    assert len(out["predictive_cluster"]["metrics"]) >= 2


def test_b_surrogate_like_nl_from_shape_and_scale_equivariance():
    rho = _geo_rho(n=36, start=0.007, ratio=1.18)
    x = log10_rho(rho)
    onset_x = x[7]
    sat_x = x[30]
    pred_x = x[22]
    nl_x = x[14]
    benefit = _benefit_bundle(x, onset_x, sat_x)
    pred_raw, pred0 = _predictive_bundle(x, pred_x, pred_x)
    dnl = _pwl(x, [x[8], nl_x], [0.01, -0.45, 0.60], 0.22)
    out = screen_positive_path(
        rho,
        benefit_raw=benefit,
        predictive_raw=pred_raw,
        predictive_rho0=pred0,
        delta_nl=dnl,
    )
    nl = out["delta_nl"]
    assert nl["event"] == "nonlinear_rebound"
    assert nl["uses_exact_minimum"] is False
    assert nl["index"] is not None
    valley = int(nl["valley_index"])
    assert nl["index"] >= valley
    idx_unscaled = int(nl["index"])
    pred_idx = out["index_predictive_guardrail"]

    scaled = rho * SCALE_EQUIVARIANCE_FACTOR
    out_s = screen_positive_path(
        scaled,
        benefit_raw=benefit,
        predictive_raw=pred_raw,
        predictive_rho0=pred0,
        delta_nl=dnl,
    )
    assert out_s["delta_nl"]["index"] == idx_unscaled
    assert out_s["activity"]["index"] == out["activity"]["index"]
    assert out_s["index_predictive_guardrail"] == pred_idx
    np.testing.assert_allclose(out_s["delta_nl"]["rho"], nl["rho"] * SCALE_EQUIVARIANCE_FACTOR, rtol=1e-12)
    if out["activity"]["rho"] is not None:
        np.testing.assert_allclose(out_s["activity"]["rho"], out["activity"]["rho"] * SCALE_EQUIVARIANCE_FACTOR, rtol=1e-12)


def test_c_no_pathology_does_not_invent_guardrail():
    rho = _geo_rho(n=32, start=0.02, ratio=1.15)
    x = log10_rho(rho)
    dist = 0.30 - 0.04 * (x - x[0])
    benefit = {
        m: (1.0 + dist if m in {"PRD", "MKI"} else (-dist if m in {"PRB", "Beta_log"} else dist))
        for m in BENEFIT_METRICS
    }
    pred_raw = {
        "R2_price": np.full(x.size, 0.88),
        "MAE_price": np.full(x.size, 120.0),
        "MAPE": np.full(x.size, 0.22),
        "RMSE_log": np.full(x.size, 0.31),
    }
    pred0 = {"R2_price": 0.88, "MAE_price": 120.0, "MAPE": 0.22, "RMSE_log": 0.31}
    dnl = 0.12 - 0.01 * (x - x[0])
    out = screen_positive_path(
        rho,
        benefit_raw=benefit,
        predictive_raw=pred_raw,
        predictive_rho0=pred0,
        delta_nl=dnl,
    )
    assert out["index_predictive_guardrail"] is None
    assert out["predictive_cluster"]["status"] == "DIRECT_GUARDRAIL_AMBIGUOUS"
    assert out["delta_nl"]["event"] is None
    assert out["delta_nl"]["classification"] == "INVALID"


def test_one_segment_available_when_grid_is_short():
    rho = _geo_rho(n=8, start=0.5, ratio=1.3)
    x = log10_rho(rho)
    y = 0.1 * x
    sel = select_pwl(x, y)
    assert sel["chosen"]["complexity"] == "one_segment"
    two = [c for c in sel["candidates"] if c["complexity"] == "two_segment"][0]
    assert two["available"] is False


def test_d_surrogate_nl_caution_binds_not_early_bend():
    """Early predictive bend while still better than rho=0; NL rebound next; harm later.

    The early bend must not bind. Stable NL caution must bind. Scale-equivariant indices.
    """
    rho = _geo_rho(n=40, start=0.005, ratio=1.20)
    x = log10_rho(rho)
    onset_x = x[6]
    sat_x = x[34]
    benefit = _benefit_bundle(x, onset_x, sat_x)
    r2_0, mae_0, mape_0, rmse_0 = 0.90, 100.0, 0.20, 0.30
    pred_raw = {
        "R2_price": _interp_path(x, [x[0], x[10], x[28], x[-1]], [0.925, 0.96, 0.90, 0.84]),
        "MAE_price": _interp_path(x, [x[0], x[10], x[28], x[-1]], [95.0, 88.0, 100.0, 118.0]),
        "MAPE": _interp_path(x, [x[0], x[32], x[-1]], [0.18, 0.185, 0.22]),
        "RMSE_log": _interp_path(x, [x[0], x[32], x[-1]], [0.27, 0.275, 0.34]),
    }
    pred0 = {"R2_price": r2_0, "MAE_price": mae_0, "MAPE": mape_0, "RMSE_log": rmse_0}
    dnl = _pwl(x, [x[8], x[18]], [0.01, -0.45, 0.60], 0.22)
    out = screen_positive_path(
        rho,
        benefit_raw=benefit,
        predictive_raw=pred_raw,
        predictive_rho0=pred0,
        delta_nl=dnl,
    )
    bend_idx = out["index_predictive_guardrail"]
    nl_idx = out["delta_nl"]["index"]
    harm_idx = out["index_predictive_harm_guardrail"]
    assert bend_idx is not None
    assert nl_idx is not None
    assert out["delta_nl"]["event"] == "nonlinear_rebound"
    assert out["delta_nl"]["raw_path_qa"] == "supported"
    assert bend_idx < int(nl_idx)
    if harm_idx is not None:
        assert int(harm_idx) > int(nl_idx)
    final = surrogate_upper_guardrail(
        harm_cluster=out["predictive_harm_cluster"],
        nl_event=out["delta_nl"],
    )
    assert final["guardrail_driver"] == "nonlinear-structure-caution"
    assert final["index_guardrail"] == int(nl_idx)
    assert final["index_guardrail"] != bend_idx

    scaled = rho * SCALE_EQUIVARIANCE_FACTOR
    out_s = screen_positive_path(
        scaled,
        benefit_raw=benefit,
        predictive_raw=pred_raw,
        predictive_rho0=pred0,
        delta_nl=dnl,
    )
    final_s = surrogate_upper_guardrail(
        harm_cluster=out_s["predictive_harm_cluster"],
        nl_event=out_s["delta_nl"],
    )
    assert out_s["index_predictive_guardrail"] == bend_idx
    assert out_s["delta_nl"]["index"] == nl_idx
    assert out_s["index_predictive_harm_guardrail"] == harm_idx
    assert final_s["index_guardrail"] == final["index_guardrail"]
    assert final_s["guardrail_driver"] == "nonlinear-structure-caution"


def test_e_surrogate_predictive_harm_binds_before_nl():
    """Predictive harm occurs before any Delta_NL rebound and must bind."""
    rho = _geo_rho(n=40, start=0.004, ratio=1.19)
    x = log10_rho(rho)
    onset_x = x[6]
    sat_x = x[34]
    benefit = _benefit_bundle(x, onset_x, sat_x)
    r2_0, mae_0, mape_0, rmse_0 = 0.90, 100.0, 0.20, 0.30
    pred_raw = {
        "R2_price": _interp_path(x, [x[0], x[8], x[14], x[-1]], [0.915, 0.935, 0.90, 0.80]),
        "MAE_price": _interp_path(x, [x[0], x[8], x[14], x[-1]], [96.0, 90.0, 100.0, 135.0]),
        "MAPE": _interp_path(x, [x[0], x[30], x[-1]], [0.18, 0.185, 0.23]),
        "RMSE_log": _interp_path(x, [x[0], x[30], x[-1]], [0.27, 0.275, 0.36]),
    }
    pred0 = {"R2_price": r2_0, "MAE_price": mae_0, "MAPE": mape_0, "RMSE_log": rmse_0}
    dnl = _pwl(x, [x[8], x[30]], [0.01, -0.45, 0.60], 0.22)
    out = screen_positive_path(
        rho,
        benefit_raw=benefit,
        predictive_raw=pred_raw,
        predictive_rho0=pred0,
        delta_nl=dnl,
    )
    harm_idx = out["index_predictive_harm_guardrail"]
    nl_idx = out["delta_nl"]["index"]
    assert harm_idx is not None
    assert nl_idx is not None
    assert int(harm_idx) < int(nl_idx)
    final = surrogate_upper_guardrail(
        harm_cluster=out["predictive_harm_cluster"],
        nl_event=out["delta_nl"],
    )
    assert final["guardrail_driver"] == "predictive-harm"
    assert final["index_guardrail"] == int(harm_idx)


def run_all() -> dict:
    results = []
    ok = True
    for name, fn in [
        ("min_segment", test_min_segment_is_grid_length_rule),
        ("A_direct_like", test_a_direct_like_early_predictive_cluster_not_saturation),
        ("B_surrogate_like_scale", test_b_surrogate_like_nl_from_shape_and_scale_equivariance),
        ("C_no_pathology", test_c_no_pathology_does_not_invent_guardrail),
        ("short_grid", test_one_segment_available_when_grid_is_short),
        ("D_nl_caution_not_early_bend", test_d_surrogate_nl_caution_binds_not_early_bend),
        ("E_predictive_harm_before_nl", test_e_surrogate_predictive_harm_binds_before_nl),
    ]:
        rec = {"name": name, "pass": False, "error": None}
        try:
            fn()
            rec["pass"] = True
        except Exception as exc:  # noqa: BLE001 — collect synthetic QA failures
            ok = False
            rec["error"] = f"{type(exc).__name__}: {exc}"
        results.append(rec)
    return {"pass": ok, "tests": results}


if __name__ == "__main__":
    payload = run_all()
    print(payload)
    raise SystemExit(0 if payload["pass"] else 1)
