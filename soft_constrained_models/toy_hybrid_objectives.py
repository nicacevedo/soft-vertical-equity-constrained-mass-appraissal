"""EXPERIMENTAL / TOY / NON-CANONICAL hybrid objectives.

Quadratic + Direct continuation, and Quadratic + nonlinear-moment guardrail.
Canonical Direct scaling is imported unchanged. Quadratic curvature is exact.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from soft_constrained_models.boosting_models import (
    _as_1d_float,
    _assert_finite_grad_hess,
    _canonical_mean_init_enabled,
    _lgbm_regressor_from_params,
    canonical_direct_exact_scaled_hessian,
    canonical_direct_scaled_grad_hess,
    canonical_surrogate_scaled_grad_hess,
)
from soft_constrained_models.toy_mechanism_objectives import EXPERIMENT_LABEL

HYBRID_METHODS: Tuple[str, ...] = ("quadratic_direct_cap", "quadratic_nl_guardrail")
HYBRID_TITLES = {
    "quadratic_direct_cap": "Quadratic + Direct cap",
    "quadratic_nl_guardrail": "Quadratic + NL guardrail",
}
QD_GRAD = "grad_i = e_i + alpha e_i c_i^2 + (lambda/2) C c_i"
QD_HESS_SUPPLIED = "hess_i = 1 + alpha c_i^2 + (lambda/(2n)) c_i^2"
QD_HESS_EXACT = "H = I + alpha diag(c^2) + (lambda/(2n)) c c^T"
QNL_GRAD = "grad_i = e_i + rho e_i c_i^2 + gamma (m2 phi2_i + m3 phi3_i)"
QNL_HESS_EXACT = "H = I + rho diag(c^2) + (gamma/n)(phi2 phi2^T + phi3 phi3^T)"
QNL_HESS_SUPPLIED = "hess_i = 1 + rho c_i^2 + gamma d23_i"
QNL_CURVATURE = (
    "exact Quadratic Hessian + exact NL-moment gradient + tighter diagonal MM "
    "majorizer for the rank-two guardrail only"
)


def nl_moment_pair(e: np.ndarray, phi: np.ndarray) -> Tuple[float, float]:
    e = _as_1d_float(e)
    phi = np.asarray(phi, dtype=np.float64)
    n = float(e.size)
    m2 = float(np.dot(phi[:, 1], e) / n)
    m3 = float(np.dot(phi[:, 2], e) / n)
    return m2, m3


def m23_norm2(m2: float, m3: float) -> float:
    return float(m2) ** 2 + float(m3) ** 2


def guardrail_d23(phi: np.ndarray) -> np.ndarray:
    """Frozen training majorizer weights d23_i = |phi2_i| mean(|phi2|) + |phi3_i| mean(|phi3|)."""
    phi = np.asarray(phi, dtype=np.float64)
    if phi.ndim != 2 or phi.shape[1] < 3:
        raise ValueError("Need training Phi with at least 3 columns.")
    p2 = phi[:, 1]
    p3 = phi[:, 2]
    d23 = np.abs(p2) * float(np.mean(np.abs(p2))) + np.abs(p3) * float(np.mean(np.abs(p3)))
    if (not np.all(np.isfinite(d23))) or np.any(d23 < -1e-15):
        raise RuntimeError("d23 is not finite and nonnegative.")
    return np.ascontiguousarray(np.maximum(d23, 0.0), dtype=np.float64)


def quadratic_direct_cap_scaled_grad_hess(
    e: np.ndarray,
    c: np.ndarray,
    *,
    alpha: float,
    lam: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Exact Quadratic diagonal + canonical Direct gradient/diagonal treatment."""
    e = _as_1d_float(e)
    c = _as_1d_float(c)
    n = int(e.size)
    alpha = float(alpha)
    lam = float(lam)
    C = float(np.mean(e * c)) if n else 0.0
    q_w = 1.0 + alpha * np.square(c)
    grad = e * q_w + 0.5 * lam * C * c
    hess = q_w + (lam / (2.0 * float(n))) * np.square(c) if n else q_w.copy()
    extras = {"n": float(n), "C": C, "alpha": alpha, "lambda": lam}
    return grad, hess, extras


def quadratic_direct_cap_exact_hessian(c: np.ndarray, *, alpha: float, lam: float) -> np.ndarray:
    c = _as_1d_float(c)
    n = int(c.size)
    if n == 0:
        return np.zeros((0, 0), dtype=float)
    H = np.diag(1.0 + float(alpha) * np.square(c))
    H = H + (float(lam) / (2.0 * float(n))) * np.outer(c, c)
    return H


def quadratic_direct_cap_scaled_objective(e: np.ndarray, c: np.ndarray, *, alpha: float, lam: float) -> float:
    e = _as_1d_float(e)
    c = _as_1d_float(c)
    n = float(e.size)
    C = float(np.mean(e * c)) if n else 0.0
    return 0.5 * float(np.dot(e, e)) + 0.5 * float(alpha) * float(np.dot(np.square(e), np.square(c))) + (float(lam) * n / 4.0) * (C ** 2)


def quadratic_nl_guardrail_scaled_grad_hess(
    e: np.ndarray,
    c: np.ndarray,
    phi: np.ndarray,
    d23: np.ndarray,
    *,
    rho: float,
    gamma: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    e = _as_1d_float(e)
    c = _as_1d_float(c)
    phi = np.asarray(phi, dtype=np.float64)
    d23 = _as_1d_float(d23)
    rho = float(rho)
    gamma = float(gamma)
    n = int(e.size)
    m2, m3 = nl_moment_pair(e, phi)
    q_w = 1.0 + rho * np.square(c)
    grad = e * q_w + gamma * (m2 * phi[:, 1] + m3 * phi[:, 2])
    hess = q_w + gamma * d23
    extras = {
        "n": float(n),
        "rho": rho,
        "gamma": gamma,
        "m2": m2,
        "m3": m3,
        "M23": m23_norm2(m2, m3),
        "curvature": QNL_CURVATURE,
    }
    return grad, hess, extras


def quadratic_nl_guardrail_exact_hessian(
    c: np.ndarray,
    phi: np.ndarray,
    *,
    rho: float,
    gamma: float,
) -> np.ndarray:
    c = _as_1d_float(c)
    phi = np.asarray(phi, dtype=np.float64)
    n = int(c.size)
    if n == 0:
        return np.zeros((0, 0), dtype=float)
    H = np.diag(1.0 + float(rho) * np.square(c))
    H = H + (float(gamma) / float(n)) * (np.outer(phi[:, 1], phi[:, 1]) + np.outer(phi[:, 2], phi[:, 2]))
    return H


def quadratic_nl_guardrail_scaled_objective(
    e: np.ndarray,
    c: np.ndarray,
    phi: np.ndarray,
    *,
    rho: float,
    gamma: float,
) -> float:
    e = _as_1d_float(e)
    c = _as_1d_float(c)
    m2, m3 = nl_moment_pair(e, phi)
    n = float(e.size)
    return (
        0.5 * float(np.dot(e, e))
        + 0.5 * float(rho) * float(np.dot(np.square(e), np.square(c)))
        + 0.5 * float(gamma) * n * m23_norm2(m2, m3)
    )


def majorizer_gap(phi: np.ndarray, d23: np.ndarray, x: np.ndarray) -> float:
    """x^T P23 x - sum d23_i x_i^2; must be <= 0 for the PSD majorizer."""
    phi = np.asarray(phi, dtype=np.float64)
    x = _as_1d_float(x)
    d23 = _as_1d_float(d23)
    n = float(x.size)
    p23x = (np.dot(phi[:, 1], x) * phi[:, 1] + np.dot(phi[:, 2], x) * phi[:, 2]) / n
    return float(np.dot(x, p23x) - np.dot(d23, np.square(x)))


class ToyHybridLGB:
    """LightGBM wrapper for the two hybrid corrections."""

    def __init__(
        self,
        *,
        method: str,
        lgbm_params,
        c_train: np.ndarray,
        phi_train: np.ndarray,
        d23: Optional[np.ndarray] = None,
        alpha: float = 0.0,
        lam: float = 0.0,
        rho: float = 0.0,
        gamma: float = 0.0,
        verbose: bool = False,
        match_native_init: bool = True,
    ):
        if method not in HYBRID_METHODS:
            raise ValueError(f"Unknown hybrid method {method!r}.")
        self.method = method
        self.verbose = bool(verbose)
        self.match_native_init = bool(match_native_init)
        self.alpha = float(alpha)
        self.lam = float(lam)
        self.rho = float(rho)
        self.gamma = float(gamma)
        self.c_train = _as_1d_float(c_train)
        self.phi_train = np.ascontiguousarray(phi_train, dtype=np.float64)
        self.d23 = None if d23 is None else _as_1d_float(d23)
        self.base_score_ = 0.0
        self.y_mean_ = 0.0
        self.experiment_label = EXPERIMENT_LABEL
        self.model = _lgbm_regressor_from_params(
            lgbm_params,
            disable_boost_from_average=_canonical_mean_init_enabled("diff", self.match_native_init),
        )

    def fit(self, X, y):
        y = _as_1d_float(y)
        self.base_score_ = float(np.mean(y))
        use_mean_init = _canonical_mean_init_enabled("diff", self.match_native_init)
        y_fit = y - self.base_score_ if use_mean_init else y
        self.y_mean_ = float(np.mean(y_fit))
        if int(self.c_train.size) != int(y_fit.size) or int(self.phi_train.shape[0]) != int(y_fit.size):
            raise ValueError("Hybrid Phi/c must align with training y.")
        if self.method == "quadratic_nl_guardrail":
            if self.d23 is None or int(self.d23.size) != int(y_fit.size):
                raise ValueError("quadratic_nl_guardrail requires frozen d23.")
        self.model.set_params(objective=self.fobj, metric="None")
        fit_kwargs = {}
        if use_mean_init:
            fit_kwargs["init_score"] = np.zeros(y_fit.shape[0], dtype=float)
        self.model.fit(X, y_fit, **fit_kwargs)
        return self

    def predict(self, X):
        pred = np.asarray(self.model.predict(X), dtype=float)
        if _canonical_mean_init_enabled("diff", self.match_native_init):
            return pred + float(self.base_score_)
        return pred

    def fobj(self, y_true, y_pred):
        y_true = _as_1d_float(y_true)
        y_pred = _as_1d_float(y_pred)
        e = y_pred - y_true
        c = y_true - float(self.y_mean_)
        if self.method == "quadratic_direct_cap":
            grad, hess, extras = quadratic_direct_cap_scaled_grad_hess(e, c, alpha=self.alpha, lam=self.lam)
            name = "ToyHybridLGB.quadratic_direct_cap"
        else:
            grad, hess, extras = quadratic_nl_guardrail_scaled_grad_hess(
                e, c, self.phi_train, self.d23, rho=self.rho, gamma=self.gamma
            )
            name = "ToyHybridLGB.quadratic_nl_guardrail"
        _assert_finite_grad_hess(grad, hess, name=name)
        if np.any(hess <= 0.0):
            raise FloatingPointError(f"Non-positive supplied Hessian in {name}.")
        if self.verbose:
            print(f"[{name}] extras={extras}", flush=True)
        return grad, hess

    def best_iteration(self) -> Optional[int]:
        model = getattr(self, "model", None)
        if model is None:
            return None
        value = getattr(model, "best_iteration_", None)
        if value is None:
            booster = getattr(model, "booster_", None)
            value = getattr(booster, "best_iteration", None) if booster is not None else None
        if value is None:
            return None
        try:
            iv = int(value)
        except Exception:
            return None
        return iv if iv > 0 else None
