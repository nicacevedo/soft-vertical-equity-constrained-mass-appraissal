"""Utilities for the projection-path covariance-penalty experiments.

The manuscript in ``tmp/rho_projection_approach`` uses log residuals
``e = f(x) - log(price)`` and centered log price ``c`` as the direct target:

    C(f) = mean(e * c).

This module keeps those quantities in one place so linear projection-path
experiments and retrained LightGBM rho sweeps are evaluated on the same scale.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def safe_mean(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else float("nan")


def safe_var(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if not x.size:
        return float("nan")
    xc = x - float(np.mean(x))
    return float(np.mean(xc * xc))


def safe_cov(x_values: np.ndarray, y_values: np.ndarray) -> float:
    x = np.asarray(x_values, dtype=float).reshape(-1)
    y = np.asarray(y_values, dtype=float).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if not x.size:
        return float("nan")
    return float(np.mean((x - float(np.mean(x))) * (y - float(np.mean(y)))))


def safe_corr(x_values: np.ndarray, y_values: np.ndarray) -> float:
    x = np.asarray(x_values, dtype=float).reshape(-1)
    y = np.asarray(y_values, dtype=float).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2 or safe_var(x) <= 0.0 or safe_var(y) <= 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def q_from_rho(rho: float, capacity: float) -> float:
    """Projection convention: q(rho) = 1 / (1 + rho * A / 2)."""
    rho = finite_float(rho)
    capacity = finite_float(capacity)
    if not np.isfinite(rho) or not np.isfinite(capacity) or rho < 0.0 or capacity <= 0.0:
        return float("nan")
    return float(1.0 / (1.0 + 0.5 * rho * capacity))


def rho_from_q(q: float, capacity: float) -> float:
    """Projection convention: rho(q) = 2(1-q)/(q A)."""
    q = finite_float(q)
    capacity = finite_float(capacity)
    if not np.isfinite(q) or not np.isfinite(capacity) or q <= 0.0 or q >= 1.0 or capacity <= 0.0:
        return float("nan")
    return float(2.0 * (1.0 - q) / (q * capacity))


def mse_cost_from_q(q: float, baseline_cov: float, capacity: float) -> float:
    q = finite_float(q)
    baseline_cov = finite_float(baseline_cov)
    capacity = finite_float(capacity)
    if not all(np.isfinite(v) for v in (q, baseline_cov, capacity)) or capacity <= 0.0:
        return float("nan")
    return float((baseline_cov * baseline_cov / capacity) * ((1.0 - q) ** 2))


def _rel_abs_error(estimate: float, truth: float) -> float:
    estimate = finite_float(estimate)
    truth = finite_float(truth)
    if not np.isfinite(estimate) or not np.isfinite(truth):
        return float("nan")
    return float(abs(estimate - truth) / max(abs(truth), 1e-12))


def _fixed_value_decile_log_vei(y_log: np.ndarray, residual: np.ndarray) -> float:
    y = np.asarray(y_log, dtype=float).reshape(-1)
    e = np.asarray(residual, dtype=float).reshape(-1)
    mask = np.isfinite(y) & np.isfinite(e)
    y = y[mask]
    e = e[mask]
    if y.size < 20:
        return float("nan")
    order = np.argsort(y, kind="mergesort")
    chunks = np.array_split(order, 10)
    if not chunks or chunks[0].size == 0 or chunks[-1].size == 0:
        return float("nan")
    return float(100.0 * (np.mean(e[chunks[-1]]) - np.mean(e[chunks[0]])))


def _direct_projection_quantities(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y_true_log, dtype=float).reshape(-1)
    f = np.asarray(y_pred_log, dtype=float).reshape(-1)
    mask = np.isfinite(y) & np.isfinite(f)
    y = y[mask]
    f = f[mask]
    if y.size == 0:
        return {}

    e = f - y
    c = y - float(np.mean(y))
    price = np.exp(y)
    ratio = np.exp(np.clip(e, -50.0, 50.0))
    ratio_taylor1 = 1.0 + e
    ratio_taylor2 = 1.0 + e + 0.5 * e * e

    var_y = float(np.mean(c * c))
    mse_log = float(np.mean(e * e))
    cov_log = float(np.mean(e * c))
    slope_log = cov_log / var_y if var_y > 0.0 else float("nan")

    mu_ratio = safe_mean(ratio)
    mu_price = safe_mean(price)
    cov_ratio_price = safe_cov(ratio, price)
    cov_ratio_price_t1 = safe_cov(ratio_taylor1, price)
    cov_ratio_price_t2 = safe_cov(ratio_taylor2, price)
    cov_ratio_logprice = safe_cov(ratio, y)
    prb_proxy = (
        cov_ratio_logprice / (mu_ratio * var_y)
        if np.isfinite(cov_ratio_logprice) and np.isfinite(mu_ratio) and abs(mu_ratio) > 1e-12 and var_y > 0.0
        else float("nan")
    )

    prd_from_cov = (
        1.0 / (1.0 + cov_ratio_price / (mu_ratio * mu_price))
        if np.isfinite(cov_ratio_price) and np.isfinite(mu_ratio) and np.isfinite(mu_price)
        and abs(mu_ratio * mu_price) > 1e-12
        else float("nan")
    )
    prd_t1 = (
        1.0 / (1.0 + cov_ratio_price_t1 / (mu_ratio * mu_price))
        if np.isfinite(cov_ratio_price_t1) and np.isfinite(mu_ratio) and np.isfinite(mu_price)
        and abs(mu_ratio * mu_price) > 1e-12
        else float("nan")
    )
    prd_t2 = (
        1.0 / (1.0 + cov_ratio_price_t2 / (mu_ratio * mu_price))
        if np.isfinite(cov_ratio_price_t2) and np.isfinite(mu_ratio) and np.isfinite(mu_price)
        and abs(mu_ratio * mu_price) > 1e-12
        else float("nan")
    )

    return {
        "n_projection_theory": int(y.size),
        "MSE_log": mse_log,
        "RMSE_log": float(math.sqrt(mse_log)),
        "Var_logprice": var_y,
        "Mean_log_residual": safe_mean(e),
        "Std_log_residual": float(np.std(e)),
        "C_log_resid_logprice": cov_log,
        "Slope_log_resid_logprice": slope_log,
        "Corr_log_resid_logprice": safe_corr(e, y),
        "C_log_resid_price": safe_cov(e, price),
        "C_ratio_price": cov_ratio_price,
        "C_ratio_logprice": cov_ratio_logprice,
        "PRB_proxy_ratio_logprice": prb_proxy,
        "VEI_log_proxy_fixed_deciles": _fixed_value_decile_log_vei(y, e),
        "Mean_ratio_theory": mu_ratio,
        "Mean_price_theory": mu_price,
        "PRD_from_cov_ratio_price": prd_from_cov,
        "C_ratio_price_taylor1": cov_ratio_price_t1,
        "C_ratio_price_taylor2": cov_ratio_price_t2,
        "C_ratio_price_taylor1_abs_error": abs(cov_ratio_price_t1 - cov_ratio_price)
        if np.isfinite(cov_ratio_price_t1) and np.isfinite(cov_ratio_price) else float("nan"),
        "C_ratio_price_taylor2_abs_error": abs(cov_ratio_price_t2 - cov_ratio_price)
        if np.isfinite(cov_ratio_price_t2) and np.isfinite(cov_ratio_price) else float("nan"),
        "C_ratio_price_taylor1_rel_error": _rel_abs_error(cov_ratio_price_t1, cov_ratio_price),
        "C_ratio_price_taylor2_rel_error": _rel_abs_error(cov_ratio_price_t2, cov_ratio_price),
        "PRD_taylor1_from_cov": prd_t1,
        "PRD_taylor2_from_cov": prd_t2,
        "PRD_taylor1_abs_error": abs(prd_t1 - prd_from_cov)
        if np.isfinite(prd_t1) and np.isfinite(prd_from_cov) else float("nan"),
        "PRD_taylor2_abs_error": abs(prd_t2 - prd_from_cov)
        if np.isfinite(prd_t2) and np.isfinite(prd_from_cov) else float("nan"),
        "Ratio_taylor1_mae": float(np.mean(np.abs(ratio - ratio_taylor1))),
        "Ratio_taylor2_mae": float(np.mean(np.abs(ratio - ratio_taylor2))),
        "Ratio_taylor1_rmse": float(np.sqrt(np.mean((ratio - ratio_taylor1) ** 2))),
        "Ratio_taylor2_rmse": float(np.sqrt(np.mean((ratio - ratio_taylor2) ** 2))),
    }


def compute_projection_theory_metrics(
    *,
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    baseline_pred_log: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute direct mechanism and approximation diagnostics.

    If ``baseline_pred_log`` is supplied, the returned metrics include empirical
    q-shrinkage and log-MSE deltas relative to that baseline on the same split.
    """
    metrics = _direct_projection_quantities(y_true_log, y_pred_log)
    if not metrics:
        return metrics

    if baseline_pred_log is None:
        return metrics

    baseline = _direct_projection_quantities(y_true_log, baseline_pred_log)
    if not baseline:
        return metrics

    c0 = baseline.get("C_log_resid_logprice", float("nan"))
    b0 = baseline.get("MSE_log", float("nan"))
    slope0 = baseline.get("Slope_log_resid_logprice", float("nan"))
    prb0 = baseline.get("PRB_proxy_ratio_logprice", float("nan"))
    vei0 = baseline.get("VEI_log_proxy_fixed_deciles", float("nan"))

    c = metrics.get("C_log_resid_logprice", float("nan"))
    b = metrics.get("MSE_log", float("nan"))
    slope = metrics.get("Slope_log_resid_logprice", float("nan"))
    prb_proxy = metrics.get("PRB_proxy_ratio_logprice", float("nan"))
    vei_proxy = metrics.get("VEI_log_proxy_fixed_deciles", float("nan"))

    q_emp = c / c0 if np.isfinite(c) and np.isfinite(c0) and abs(c0) > 1e-12 else float("nan")
    q_abs = abs(c) / abs(c0) if np.isfinite(c) and np.isfinite(c0) and abs(c0) > 1e-12 else float("nan")
    delta_mse = b - b0 if np.isfinite(b) and np.isfinite(b0) else float("nan")

    metrics.update({
        "baseline_C_log_resid_logprice": c0,
        "baseline_MSE_log": b0,
        "baseline_Slope_log_resid_logprice": slope0,
        "baseline_PRB_proxy_ratio_logprice": prb0,
        "baseline_VEI_log_proxy_fixed_deciles": vei0,
        "q_empirical_signed": q_emp,
        "q_empirical_abs": q_abs,
        "covariance_reduction_empirical": 1.0 - q_emp if np.isfinite(q_emp) else float("nan"),
        "covariance_abs_reduction_empirical": 1.0 - q_abs if np.isfinite(q_abs) else float("nan"),
        "delta_MSE_log": delta_mse,
        "delta_MSE_log_frac": delta_mse / b0 if np.isfinite(delta_mse) and np.isfinite(b0) and b0 > 0.0 else float("nan"),
        "slope_shrink_empirical": slope / slope0
        if np.isfinite(slope) and np.isfinite(slope0) and abs(slope0) > 1e-12 else float("nan"),
        "PRB_proxy_shrink_empirical": prb_proxy / prb0
        if np.isfinite(prb_proxy) and np.isfinite(prb0) and abs(prb0) > 1e-12 else float("nan"),
        "VEI_log_proxy_shrink_empirical": vei_proxy / vei0
        if np.isfinite(vei_proxy) and np.isfinite(vei0) and abs(vei0) > 1e-12 else float("nan"),
    })
    return metrics


def add_projection_theory_predictions(row: Dict[str, Any]) -> Dict[str, Any]:
    """Add theory-predicted quantities to a comparison row in-place style."""
    out = dict(row)
    rho = finite_float(out.get("rho"))
    capacity = finite_float(out.get("A_projection_capacity", out.get("A_var_f0_log")))
    c0 = finite_float(out.get("baseline_C_log_resid_logprice", out.get("C0_cov_log_residual_logprice")))
    b0 = finite_float(out.get("baseline_MSE_log", out.get("B_mse_log")))
    slope0 = finite_float(out.get("baseline_Slope_log_resid_logprice"))
    prb0 = finite_float(out.get("baseline_PRB_proxy_ratio_logprice"))
    vei0 = finite_float(out.get("baseline_VEI_log_proxy_fixed_deciles"))

    q_theory = q_from_rho(rho, capacity)
    d_mse = mse_cost_from_q(q_theory, c0, capacity)
    out["q_theory"] = q_theory
    out["covariance_reduction_theory"] = 1.0 - q_theory if np.isfinite(q_theory) else float("nan")
    out["delta_MSE_log_theory"] = d_mse
    out["delta_MSE_log_frac_theory"] = d_mse / b0 if np.isfinite(d_mse) and np.isfinite(b0) and b0 > 0.0 else float("nan")
    out["C_log_resid_logprice_theory"] = q_theory * c0 if np.isfinite(q_theory) and np.isfinite(c0) else float("nan")
    out["Slope_log_resid_logprice_theory"] = q_theory * slope0 if np.isfinite(q_theory) and np.isfinite(slope0) else float("nan")
    out["PRB_proxy_ratio_logprice_theory"] = q_theory * prb0 if np.isfinite(q_theory) and np.isfinite(prb0) else float("nan")
    out["VEI_log_proxy_fixed_deciles_theory"] = q_theory * vei0 if np.isfinite(q_theory) and np.isfinite(vei0) else float("nan")
    return out
