"""EXPERIMENTAL / TOY / NON-CANONICAL surrogate-mechanism ablation.

This module is isolated from the paper's canonical Direct and Surrogate
objectives. It is not a proposed production method.

Canonical log-residual setup (ratio_mode='diff', identity c_i)
    e_i = yhat_i - y_i
    c_i = y_i - mean(y)

Complete objective (unscaled)
    J = (1/n) sum_i e_i^2 + rho * Psi(e)

LightGBM receives derivatives of (n/2) * J so that rho=0 matches native L2:
    g_i = e_i,  h_i = 1.

Level-invariant quadratic/capped variants: the gradient of the profiled
penalty is exact. The exact profiled Hessian is
    W - w w^T / (sum_j w_j).
LightGBM cannot consume the off-diagonal rank-one block; this toy
implementation supplies only the exact diagonal
    h_i^pen = w_i - w_i^2 / (sum_j w_j).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from soft_constrained_models.boosting_models import (
    _as_1d_float,
    _assert_finite_grad_hess,
    _canonical_mean_init_enabled,
    _lgbm_regressor_from_params,
    canonical_surrogate_scaled_grad_hess,
)

EXPERIMENT_LABEL = "EXPERIMENTAL / TOY / NON-CANONICAL"
PENALTY_SHAPES: Tuple[str, ...] = ("quadratic", "absolute", "capped_quadratic", "huber")
CAP_QUANTILE = 0.80
HUBER_Q_QUANTILE = 0.80
LAMBDA_GRID: Tuple[float, ...] = (0.0, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
RATIO_SHAPE_LAMBDAS: Tuple[float, ...] = (0.0, 1.0, 10.0, 100.0)

VARIANT_SPECS: Tuple[Tuple[str, bool], ...] = (
    ("quadratic", False),
    ("quadratic", True),
    ("absolute", False),
    ("absolute", True),
    ("capped_quadratic", False),
    ("capped_quadratic", True),
)


def variant_name(penalty_shape: str, level_invariant: bool) -> str:
    shape = str(penalty_shape).strip().lower()
    if shape not in PENALTY_SHAPES:
        raise ValueError(f"Unknown penalty_shape={penalty_shape!r}.")
    return f"{shape}_{'li' if level_invariant else 'fixed'}"


def parse_variant_name(name: str) -> Tuple[str, bool]:
    raw = str(name).strip().lower()
    for shape, li in VARIANT_SPECS:
        if variant_name(shape, li) == raw:
            return shape, li
    raise ValueError(f"Unknown variant {name!r}.")


def sign_zero(x: np.ndarray) -> np.ndarray:
    """sign(0) = 0."""
    out = np.sign(np.asarray(x, dtype=float))
    out[out == 0.0] = 0.0
    return out


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """|c|-weighted median of residuals: minimizer of sum_i w_i |x_i - a|."""
    values = _as_1d_float(values)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if values.size == 0:
        return 0.0
    if weights.shape != values.shape:
        raise ValueError("values and weights must have the same shape.")
    total = float(np.sum(weights))
    if (not np.isfinite(total)) or total <= 0.0:
        return float(np.median(values))
    order = np.argsort(values, kind="mergesort")
    v = values[order]
    cdf = np.cumsum(weights[order])
    idx = int(np.searchsorted(cdf, 0.5 * total, side="left"))
    idx = min(max(idx, 0), int(v.size) - 1)
    return float(v[idx])


def quadratic_a_star(e: np.ndarray, weights: np.ndarray) -> float:
    e = _as_1d_float(e)
    w = np.asarray(weights, dtype=float).reshape(-1)
    sw = float(np.sum(w))
    if (not np.isfinite(sw)) or sw <= 0.0:
        return 0.0
    return float(np.dot(w, e) / sw)


def capped_tau(c: np.ndarray, quantile: float = CAP_QUANTILE) -> float:
    abs_c = np.abs(_as_1d_float(c))
    if abs_c.size == 0:
        return 0.0
    return float(np.quantile(abs_c, float(quantile)))


def capped_weights(c: np.ndarray, tau: Optional[float] = None, quantile: float = CAP_QUANTILE) -> Tuple[np.ndarray, float]:
    c = _as_1d_float(c)
    if tau is None:
        tau = capped_tau(c, quantile=quantile)
    tau = float(tau)
    w = np.minimum(c ** 2, tau ** 2)
    return w, tau


def covariance_C(e: np.ndarray, c: np.ndarray) -> float:
    return float(np.mean(_as_1d_float(e) * _as_1d_float(c)))


def huber_delta_from_q(q: np.ndarray, quantile: float = HUBER_Q_QUANTILE) -> float:
    aq = np.abs(_as_1d_float(q))
    if aq.size == 0:
        return 0.0
    return float(np.quantile(aq, float(quantile)))


def huber_phi(q: np.ndarray, delta: float) -> np.ndarray:
    """phi_delta(q) = q^2 if |q|<=delta, else 2 delta |q| - delta^2.

    Scaled so the central region coincides with the quadratic penalty on q.
    """
    q = _as_1d_float(q)
    delta = float(delta)
    aq = np.abs(q)
    out = np.empty_like(q)
    inside = aq <= delta
    out[inside] = q[inside] ** 2
    out[~inside] = 2.0 * delta * aq[~inside] - (delta ** 2)
    return out


def huber_linear_share(e: np.ndarray, c: np.ndarray, delta: float) -> float:
    q = _as_1d_float(e) * _as_1d_float(c)
    n = int(q.size)
    if n == 0:
        return float("nan")
    return float(np.mean(np.abs(q) > float(delta)))


def penalty_value(
    e: np.ndarray,
    c: np.ndarray,
    *,
    penalty_shape: str,
    level_invariant: bool,
    tau: Optional[float] = None,
    huber_delta: Optional[float] = None,
) -> Dict[str, float]:
    """Unscaled Psi(e). Used for bounds and shift-invariance checks."""
    e = _as_1d_float(e)
    c = _as_1d_float(c)
    shape = str(penalty_shape).strip().lower()
    extras: Dict[str, float] = {"C": covariance_C(e, c), "n": float(e.size)}
    if shape == "quadratic":
        w = c ** 2
        a_star = quadratic_a_star(e, w) if level_invariant else 0.0
        resid = e - a_star if level_invariant else e
        extras.update({"a_star": float(a_star), "tau": float("nan")})
        extras["Psi"] = float(np.mean(w * resid ** 2))
        return extras
    if shape == "absolute":
        abs_c = np.abs(c)
        a_star = weighted_median(e, abs_c) if level_invariant else 0.0
        resid = e - a_star if level_invariant else e
        extras.update({"a_star": float(a_star), "tau": float("nan")})
        extras["Psi"] = float(np.mean(abs_c * np.abs(resid)))
        return extras
    if shape == "capped_quadratic":
        w, tau_val = capped_weights(c, tau=tau)
        a_star = quadratic_a_star(e, w) if level_invariant else 0.0
        resid = e - a_star if level_invariant else e
        extras.update({"a_star": float(a_star), "tau": float(tau_val)})
        extras["Psi"] = float(np.mean(w * resid ** 2))
        extras["K_tau"] = k_tau_constant(c, w)
        return extras
    if shape == "huber":
        if level_invariant:
            raise ValueError("Huber is fixed-level only in this experiment.")
        q = e * c
        if huber_delta is None:
            raise ValueError("huber_delta is required for the Huber penalty.")
        delta = float(huber_delta)
        extras.update({"a_star": 0.0, "tau": float("nan"), "huber_delta": delta})
        extras["Psi"] = float(np.mean(huber_phi(q, delta)))
        extras["p_linear"] = huber_linear_share(e, c, delta)
        return extras
    raise ValueError(f"Unknown penalty_shape={penalty_shape!r}.")


def k_tau_constant(c: np.ndarray, w: np.ndarray) -> float:
    c = _as_1d_float(c)
    w = np.asarray(w, dtype=float).reshape(-1)
    n = float(c.size)
    if n <= 0:
        return float("nan")
    mask = c != 0.0
    if not np.any(mask):
        return 0.0
    ww = w[mask]
    cc = c[mask]
    if np.any(ww <= 0.0):
        return float("nan")
    return float(np.sum((cc ** 2) / ww) / n)


def scaled_objective(
    e: np.ndarray,
    c: np.ndarray,
    rho: float,
    *,
    penalty_shape: str,
    level_invariant: bool,
    tau: Optional[float] = None,
    huber_delta: Optional[float] = None,
    profile: bool = True,
) -> float:
    """(n/2) * J, matching LightGBM-native L2 scaling."""
    e = _as_1d_float(e)
    c = _as_1d_float(c)
    rho = float(rho)
    mse = 0.5 * float(np.sum(e ** 2))
    use_li = bool(level_invariant) and bool(profile)
    psi_info = penalty_value(
        e, c, penalty_shape=penalty_shape, level_invariant=use_li, tau=tau, huber_delta=huber_delta
    )
    return mse + 0.5 * rho * float(np.size(e)) * float(psi_info["Psi"])


def exact_profiled_hessian(w: np.ndarray, rho: float) -> np.ndarray:
    """Exact Hessian of the scaled profiled quadratic/capped objective.

    H = I + rho * (W - w w^T / (sum w)).
    LightGBM is given only diag(H); the off-diagonal rank-one block is omitted.
    """
    w = _as_1d_float(w)
    n = int(w.size)
    H = np.eye(n, dtype=float)
    sw = float(np.sum(w))
    if n == 0 or (not np.isfinite(sw)) or sw <= 0.0:
        return H
    return H + float(rho) * (np.diag(w) - np.outer(w, w) / sw)


def _penalty_weights(
    c: np.ndarray,
    *,
    penalty_shape: str,
    tau: Optional[float],
) -> Tuple[np.ndarray, float]:
    shape = str(penalty_shape).strip().lower()
    c = _as_1d_float(c)
    if shape == "quadratic":
        return c ** 2, float("nan")
    if shape == "capped_quadratic":
        return capped_weights(c, tau=tau)
    if shape == "absolute":
        return np.abs(c), float("nan")
    raise ValueError(f"Unknown penalty_shape={penalty_shape!r}.")


def toy_surrogate_scaled_grad_hess(
    y_true,
    y_pred,
    *,
    y_mean: float,
    rho: float,
    penalty_shape: str,
    level_invariant: bool,
    tau: Optional[float] = None,
    huber_delta: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Scaled (n/2)J gradient and supplied LightGBM Hessian for one toy variant.

    Fairness-only contributions at unit rho are documented in the experiment
    brief. Total Hessian is always strictly positive under this scaling.
    """
    y_true = _as_1d_float(y_true)
    y_pred = _as_1d_float(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    n = int(y_true.size)
    rho = float(rho)
    shape = str(penalty_shape).strip().lower()
    li = bool(level_invariant)
    if n == 0:
        empty = np.asarray([], dtype=float)
        return empty, empty, {"n": 0.0, "rho": rho, "a_star": 0.0, "tau": float("nan"), "C": 0.0, "huber_delta": float("nan")}

    e = y_pred - y_true
    c = y_true - float(y_mean)
    C = covariance_C(e, c)

    if shape == "huber":
        if li:
            raise ValueError("Huber is fixed-level only in this experiment.")
        if huber_delta is None:
            raise ValueError("huber_delta is required for the Huber penalty.")
        delta = float(huber_delta)
        q = e * c
        abs_c = np.abs(c)
        inside = np.abs(q) <= delta
        g_pen = np.empty(n, dtype=float)
        h_pen = np.zeros(n, dtype=float)
        g_pen[inside] = e[inside] * (c[inside] ** 2)
        h_pen[inside] = c[inside] ** 2
        outside = ~inside
        g_pen[outside] = delta * abs_c[outside] * sign_zero(e[outside])
        a_star = 0.0
        tau_val = float("nan")
        grad = e + rho * g_pen
        hess = np.ones(n, dtype=float) + rho * h_pen
        extras = {
            "n": float(n),
            "rho": rho,
            "C": float(C),
            "a_star": 0.0,
            "tau": float("nan"),
            "huber_delta": delta,
            "p_linear": float(np.mean(outside)) if n else 0.0,
            "penalty_shape": shape,
            "level_invariant": 0.0,
            "sum_w": float("nan"),
            "g_pen_rms": float(np.sqrt(np.mean(g_pen ** 2))) if n else 0.0,
        }
        return grad, hess, extras

    w, tau_val = _penalty_weights(c, penalty_shape=shape, tau=tau)

    if shape in {"quadratic", "capped_quadratic"}:
        if li:
            a_star = quadratic_a_star(e, w)
            resid = e - a_star
            g_pen = w * resid
            sw = float(np.sum(w))
            if sw > 0.0:
                h_pen = w - (w ** 2) / sw
            else:
                h_pen = np.zeros(n, dtype=float)
        else:
            a_star = 0.0
            g_pen = w * e
            h_pen = w.copy()
        grad = e + rho * g_pen
        hess = np.ones(n, dtype=float) + rho * h_pen
    elif shape == "absolute":
        # Scaled penalty is (rho/2) sum_i |c_i| |e_i - a|, so g_pen uses 1/2.
        if li:
            a_star = weighted_median(e, w)
            resid = e - a_star
        else:
            a_star = 0.0
            resid = e
        g_pen = 0.5 * w * sign_zero(resid)
        h_pen = np.zeros(n, dtype=float)
        grad = e + rho * g_pen
        hess = np.ones(n, dtype=float)
    else:
        raise ValueError(f"Unknown penalty_shape={penalty_shape!r}.")

    extras = {
        "n": float(n),
        "rho": rho,
        "C": float(C),
        "a_star": float(a_star),
        "tau": float(tau_val) if np.isfinite(tau_val) else float("nan"),
        "penalty_shape": shape,
        "level_invariant": float(li),
        "sum_w": float(np.sum(w)),
        "g_pen_rms": float(np.sqrt(np.mean(g_pen ** 2))) if n else 0.0,
    }
    return grad, hess, extras


def fairness_penalty_gradient_unit_rho(
    e: np.ndarray,
    c: np.ndarray,
    *,
    penalty_shape: str,
    level_invariant: bool,
    tau: Optional[float] = None,
) -> np.ndarray:
    """g^{pen} at unit raw rho (baseline squared-error term excluded)."""
    e = _as_1d_float(e)
    dummy_true = np.zeros_like(e)
    dummy_pred = e.copy()
    dummy_mean = 0.0
    # Reconstruct y_true, y_pred, y_mean so that e' = e and c' = c.
    y_true = dummy_true + c
    y_pred = y_true + e
    y_mean = float(np.mean(y_true) - dummy_mean)
    # c_out = y_true - y_mean = c - (mean(c) - 0). Require mean(c)=0 in callers.
    grad, _hess, _ = toy_surrogate_scaled_grad_hess(
        y_true,
        y_pred,
        y_mean=y_mean,
        rho=1.0,
        penalty_shape=penalty_shape,
        level_invariant=level_invariant,
        tau=tau,
    )
    return grad - e


def fairness_penalty_gradient_unit_rho_from_ec(
    e: np.ndarray,
    c: np.ndarray,
    *,
    penalty_shape: str,
    level_invariant: bool,
    tau: Optional[float] = None,
    huber_delta: Optional[float] = None,
) -> np.ndarray:
    """Direct g^{pen}(e) at unit rho from residual/center vectors."""
    e = _as_1d_float(e)
    c = _as_1d_float(c)
    shape = str(penalty_shape).strip().lower()
    if shape == "huber":
        if level_invariant:
            raise ValueError("Huber is fixed-level only in this experiment.")
        if huber_delta is None:
            raise ValueError("huber_delta is required for the Huber penalty.")
        dummy_true = c.copy()
        dummy_pred = dummy_true + e
        y_mean = 0.0
        g, _h, _ = toy_surrogate_scaled_grad_hess(
            dummy_true,
            dummy_pred,
            y_mean=y_mean,
            rho=1.0,
            penalty_shape="huber",
            level_invariant=False,
            huber_delta=huber_delta,
        )
        return g - e
    w, _tau = _penalty_weights(c, penalty_shape=shape, tau=tau)
    if shape in {"quadratic", "capped_quadratic"}:
        if level_invariant:
            a_star = quadratic_a_star(e, w)
            return w * (e - a_star)
        return w * e
    if shape == "absolute":
        if level_invariant:
            a_star = weighted_median(e, w)
            return 0.5 * w * sign_zero(e - a_star)
        return 0.5 * w * sign_zero(e)
    raise ValueError(f"Unknown penalty_shape={penalty_shape!r}.")


class ToyLGBSurrogate:
    """EXPERIMENTAL / TOY / NON-CANONICAL LightGBM wrapper.

    Mirrors the canonical custom-objective initialization (mean init, identity
    log-residual, no early stopping) without changing canonical classes.
    """

    def __init__(
        self,
        *,
        rho: float,
        penalty_shape: str,
        level_invariant: bool,
        tau: Optional[float] = None,
        huber_delta: Optional[float] = None,
        lgbm_params=None,
        verbose: bool = False,
        match_native_init: bool = True,
        early_stopping_rounds=None,
        cap_quantile: float = CAP_QUANTILE,
    ):
        self.rho = float(rho)
        self.penalty_shape = str(penalty_shape).strip().lower()
        if self.penalty_shape not in PENALTY_SHAPES:
            raise ValueError(f"penalty_shape must be one of {PENALTY_SHAPES}.")
        self.level_invariant = bool(level_invariant)
        self.tau_arg = None if tau is None else float(tau)
        self.huber_delta_arg = None if huber_delta is None else float(huber_delta)
        self.huber_delta_ = float("nan")
        self.cap_quantile = float(cap_quantile)
        self.verbose = bool(verbose)
        self.match_native_init = bool(match_native_init)
        self.early_stopping_rounds = None if early_stopping_rounds is None else int(early_stopping_rounds)
        self.ratio_mode = "diff"
        self.base_score_ = 0.0
        self.y_mean_ = 0.0
        self.tau_ = float("nan")
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
        c_train = y_fit - self.y_mean_
        if self.penalty_shape == "capped_quadratic":
            if self.tau_arg is None:
                self.tau_ = capped_tau(c_train, quantile=self.cap_quantile)
            else:
                self.tau_ = float(self.tau_arg)
        else:
            self.tau_ = float("nan")
        if self.penalty_shape == "huber":
            if self.level_invariant:
                raise ValueError("Huber is fixed-level only in this experiment.")
            if self.huber_delta_arg is None:
                raise ValueError("huber_delta must be supplied for Huber fits (from the shared rho=0 residuals).")
            self.huber_delta_ = float(self.huber_delta_arg)
        else:
            self.huber_delta_ = float("nan")
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
        tau = None if not np.isfinite(self.tau_) else float(self.tau_)
        huber_delta = None if not np.isfinite(self.huber_delta_) else float(self.huber_delta_)
        grad, hess, extras = toy_surrogate_scaled_grad_hess(
            y_true,
            y_pred,
            y_mean=float(self.y_mean_),
            rho=self.rho,
            penalty_shape=self.penalty_shape,
            level_invariant=self.level_invariant,
            tau=tau,
            huber_delta=huber_delta,
        )
        _assert_finite_grad_hess(
            grad,
            hess,
            name=f"ToyLGBSurrogate.{variant_name(self.penalty_shape, self.level_invariant)}",
        )
        if np.any(hess <= 0.0):
            raise FloatingPointError("Non-positive supplied Hessian in ToyLGBSurrogate.")
        if self.verbose:
            print(
                f"[TOY {variant_name(self.penalty_shape, self.level_invariant)}] "
                f"rho={self.rho:.6g} C={extras['C']:.6e} a*={extras['a_star']:.6e}"
            )
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

    def __str__(self):
        return (
            f"ToyLGBSurrogate(EXPERIMENTAL, shape={self.penalty_shape}, "
            f"level_invariant={self.level_invariant}, rho={self.rho})"
        )


def quadratic_fixed_matches_canonical(y_true, y_pred, y_mean: float, rho: float) -> bool:
    g, h, _ = toy_surrogate_scaled_grad_hess(
        y_true,
        y_pred,
        y_mean=y_mean,
        rho=rho,
        penalty_shape="quadratic",
        level_invariant=False,
    )
    g0, h0, _ = canonical_surrogate_scaled_grad_hess(y_true, y_pred, y_mean=y_mean, rho=rho)
    return bool(np.allclose(g, g0, rtol=1e-12, atol=1e-12) and np.allclose(h, h0, rtol=1e-12, atol=1e-12))
