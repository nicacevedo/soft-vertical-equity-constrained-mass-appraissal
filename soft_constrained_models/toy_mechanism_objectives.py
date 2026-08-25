"""EXPERIMENTAL / TOY / NON-CANONICAL six-path mechanism objectives.

Isolated from paper-method selection. Canonical Direct is imported unchanged.
Quadratic uses the existing scaled identity-surrogate formulas.

LightGBM receives derivatives of the scaled objectives so that rho=0 matches
native L2: grad = e, hess = 1.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, Optional, Tuple

import numpy as np

from soft_constrained_models.boosting_models import (
    LGBCovPenalty,
    _as_1d_float,
    _assert_finite_grad_hess,
    _canonical_mean_init_enabled,
    _lgbm_regressor_from_params,
    canonical_direct_exact_scaled_hessian,
    canonical_direct_scaled_grad_hess,
    canonical_surrogate_scaled_grad_hess,
)

EXPERIMENT_LABEL = "EXPERIMENTAL / TOY / NON-CANONICAL"
METHODS: Tuple[str, ...] = (
    "current_direct",
    "direct_mm_k1",
    "moment_mm_k2",
    "moment_mm_k3",
    "local_slope_smooth",
    "quadratic",
)
METHOD_TITLES = {
    "current_direct": "Current Direct",
    "direct_mm_k1": "Direct-MM K=1",
    "moment_mm_k2": "Moment-MM K=2",
    "moment_mm_k3": "Moment-MM K=3",
    "local_slope_smooth": "Local-Slope Smooth",
    "quadratic": "Quadratic",
}
MOMENT_K = {"direct_mm_k1": 1, "moment_mm_k2": 2, "moment_mm_k3": 3}
MM_CURVATURE_DESCRIPTION = (
    "exact statistical-target gradient + rigorous diagonal MM majorized curvature"
)
DIRECT_CANONICAL_GRAD = "grad_i = e_i + (rho/2) * C * c_i"
DIRECT_CANONICAL_HESS_SUPPLIED = "hess_i = 1 + (rho / (2n)) * c_i^2"
DIRECT_CANONICAL_HESS_EXACT = "H = I + (rho / (2n)) * c c^T"
GRAM_ATOL = 1e-8
MEAN_ATOL = 1e-8
PHI1_ATOL = 1e-8
COND_FAIL = 1e10


def method_k(method: str) -> int:
    if method not in MOMENT_K:
        return 0
    return int(MOMENT_K[method])


def design_monomials(z: np.ndarray) -> np.ndarray:
    z = _as_1d_float(z)
    ones = np.ones(z.shape[0], dtype=np.float64)
    return np.column_stack([ones, z, z ** 2, z ** 3]).astype(np.float64, copy=False)


def build_training_moment_basis(y_train: np.ndarray) -> Dict[str, Any]:
    """Nested empirically orthonormal {phi_1, phi_2, phi_3} from training y only."""
    y = _as_1d_float(y_train).astype(np.float64, copy=False)
    n = int(y.size)
    if n < 8:
        raise ValueError("Need at least 8 training observations to build the moment basis.")
    y_mean = float(np.mean(y))
    c = y - y_mean
    sigma_c = float(np.sqrt(np.mean(c ** 2)))
    if (not np.isfinite(sigma_c)) or sigma_c <= 0.0:
        raise RuntimeError("Training sigma_c is not a positive finite scale.")
    z = c / sigma_c
    A = design_monomials(z)
    scale = float(np.sqrt(n))
    q, r = np.linalg.qr(A / scale, mode="reduced")
    if q.shape != (n, 4) or r.shape != (4, 4):
        raise RuntimeError("Reduced Householder QR did not return a 4-column basis.")
    cond_r = float(np.linalg.cond(r))
    if (not np.isfinite(cond_r)) or cond_r > COND_FAIL:
        raise RuntimeError(f"Moment-basis R is ill-conditioned: cond={cond_r:.6g}.")
    try:
        rinv = np.linalg.inv(r)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("Moment-basis R is not invertible.") from exc
    phi_full = A @ rinv
    raw_powers = [np.ones(n, dtype=np.float64), z, z ** 2, z ** 3]
    signs = np.ones(4, dtype=np.float64)
    for k in range(1, 4):
        corr = float(np.dot(phi_full[:, k], raw_powers[k]))
        if corr < 0.0:
            phi_full[:, k] *= -1.0
            rinv[:, k] *= -1.0
            signs[k] = -1.0
    phi = np.ascontiguousarray(phi_full[:, 1:4], dtype=np.float64)
    if not np.all(np.isfinite(phi)):
        raise RuntimeError("Moment basis contains non-finite values.")
    means = np.mean(phi, axis=0)
    gram = (phi.T @ phi) / float(n)
    phi1_err = float(np.max(np.abs(phi[:, 0] - z)))
    diagnostics = {
        "n": n,
        "y_mean": y_mean,
        "sigma_c": sigma_c,
        "qr_cond": cond_r,
        "signs": signs.tolist(),
        "empirical_means": means.tolist(),
        "gram": gram.tolist(),
        "max_abs_phi": np.max(np.abs(phi), axis=0).tolist(),
        "phi1_max_abs_err_vs_z": phi1_err,
        "gram_offdiag_max": float(np.max(np.abs(gram - np.eye(3)))),
        "mean_abs_max": float(np.max(np.abs(means))),
    }
    if float(np.max(np.abs(means))) > 1e-6:
        raise RuntimeError(f"Moment basis means are not ~0: {means}.")
    if float(np.max(np.abs(gram - np.eye(3)))) > 1e-6:
        raise RuntimeError("Moment basis fails empirical orthonormality Phi.T Phi / n ~= I.")
    if phi1_err > 1e-6:
        raise RuntimeError(f"K=1 identity failed: max|phi_1 - z|={phi1_err:.6g}.")
    return {
        "y_mean": y_mean,
        "sigma_c": sigma_c,
        "rinv": np.ascontiguousarray(rinv, dtype=np.float64),
        "phi_train": phi,
        "z_train": np.ascontiguousarray(z, dtype=np.float64),
        "c_train": np.ascontiguousarray(c, dtype=np.float64),
        "diagnostics": diagnostics,
        "construction": "training_only_reduced_householder_qr_empirical_inner_product",
    }


def apply_moment_basis(y: np.ndarray, basis: Dict[str, Any]) -> np.ndarray:
    """Evaluate the TRAINING basis map on any log-price vector. Does not refit."""
    y = _as_1d_float(y).astype(np.float64, copy=False)
    c = y - float(basis["y_mean"])
    sigma_c = float(basis["sigma_c"])
    z = c / sigma_c
    a = design_monomials(z)
    phi_full = a @ np.asarray(basis["rinv"], dtype=np.float64)
    return np.ascontiguousarray(phi_full[:, 1:4], dtype=np.float64)


def moments_from_residual(e: np.ndarray, phi: np.ndarray) -> np.ndarray:
    e = _as_1d_float(e)
    phi = np.asarray(phi, dtype=np.float64)
    if phi.ndim != 2 or phi.shape[0] != e.size:
        raise ValueError("phi must be (n, K) aligned with e.")
    return (phi.T @ e) / float(e.size)


def projector_apply(v: np.ndarray, phi_k: np.ndarray) -> np.ndarray:
    """P v = Phi (Phi.T v / n) without forming the n x n matrix."""
    v = _as_1d_float(v)
    m = moments_from_residual(v, phi_k)
    return phi_k @ m


def scaled_moment_objective(e: np.ndarray, phi_k: np.ndarray, rho: float) -> float:
    e = _as_1d_float(e)
    n = float(e.size)
    m = moments_from_residual(e, phi_k)
    return 0.5 * float(np.dot(e, e)) + 0.5 * float(rho) * n * float(np.dot(m, m))


def moment_mm_scaled_grad_hess(
    e: np.ndarray,
    phi_k: np.ndarray,
    rho: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Exact gradient of F_K; MM diagonal curvature hess_i = 1 + rho.

    Not the exact Hessian of the global moment objective.
    """
    e = _as_1d_float(e)
    phi_k = np.asarray(phi_k, dtype=np.float64)
    rho = float(rho)
    n = int(e.size)
    m = moments_from_residual(e, phi_k)
    grad = e + rho * (phi_k @ m)
    hess = np.full(n, 1.0 + rho, dtype=np.float64)
    extras = {
        "n": float(n),
        "rho": rho,
        "m": m,
        "curvature": MM_CURVATURE_DESCRIPTION,
    }
    return grad, hess, extras


def k1_squared_covariance_identity(e: np.ndarray, c: np.ndarray, phi1: np.ndarray, sigma_c: float) -> Dict[str, float]:
    e = _as_1d_float(e)
    c = _as_1d_float(c)
    phi1 = _as_1d_float(phi1)
    n = float(e.size)
    cov = float(np.mean(e * c))
    p_quad = float(np.dot(e, projector_apply(e, phi1.reshape(-1, 1))))
    expected = n * (cov / float(sigma_c)) ** 2
    return {
        "eT_P1_e": p_quad,
        "n_cov_over_sigma_sq": expected,
        "abs_err": abs(p_quad - expected),
        "C": cov,
    }


def local_slope_penalty(e: np.ndarray, c: np.ndarray, epsilon: float) -> float:
    e = _as_1d_float(e)
    c = _as_1d_float(c)
    eps = float(epsilon)
    abs_c = np.abs(c)
    inner = np.sqrt(np.square(e) + np.square(eps * c))
    return float(np.sum(abs_c * inner - eps * np.square(c)))


def local_slope_scaled_objective(e: np.ndarray, c: np.ndarray, rho: float, epsilon: float) -> float:
    e = _as_1d_float(e)
    return 0.5 * float(np.dot(e, e)) + float(rho) * local_slope_penalty(e, c, epsilon)


def local_slope_fairness_grad_hess(
    e: np.ndarray,
    c: np.ndarray,
    epsilon: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact fairness derivatives of Psi_LS. c_i=0 => 0, 0."""
    e = _as_1d_float(e)
    c = _as_1d_float(c)
    eps = float(epsilon)
    n = int(e.size)
    fair_g = np.zeros(n, dtype=np.float64)
    fair_h = np.zeros(n, dtype=np.float64)
    abs_c = np.abs(c)
    nz = abs_c > 0.0
    if not np.any(nz):
        return fair_g, fair_h
    e_nz = e[nz]
    c_nz = c[nz]
    abs_nz = abs_c[nz]
    den2 = np.square(e_nz) + np.square(eps * c_nz)
    den = np.sqrt(den2)
    fair_g[nz] = abs_nz * e_nz / den
    fair_h[nz] = (eps ** 2) * (abs_nz ** 3) / (den2 * den)
    return fair_g, fair_h


def local_slope_scaled_grad_hess(
    e: np.ndarray,
    c: np.ndarray,
    rho: float,
    epsilon: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    e = _as_1d_float(e)
    rho = float(rho)
    fair_g, fair_h = local_slope_fairness_grad_hess(e, c, epsilon)
    grad = e + rho * fair_g
    hess = 1.0 + rho * fair_h
    extras = {"n": float(e.size), "rho": rho, "epsilon": float(epsilon)}
    return grad, hess, extras


def quadratic_scaled_grad_hess(e: np.ndarray, c: np.ndarray, rho: float) -> Tuple[np.ndarray, np.ndarray]:
    e = _as_1d_float(e)
    c = _as_1d_float(c)
    weight = 1.0 + float(rho) * np.square(c)
    return e * weight, weight.copy()


def quadratic_pointwise_prox(e0: np.ndarray, c: np.ndarray, rho: float) -> np.ndarray:
    e0 = _as_1d_float(e0)
    c = _as_1d_float(c)
    return e0 / (1.0 + float(rho) * np.square(c))


def moment_prox(e0: np.ndarray, phi_k: np.ndarray, rho: float) -> np.ndarray:
    """Unrestricted-space proximal / exact linear solve: (I + rho P)^{-1} e0."""
    e0 = _as_1d_float(e0)
    rho = float(rho)
    return e0 - (rho / (1.0 + rho)) * projector_apply(e0, phi_k)


def eval_nonlinear_subspace(c_eval: np.ndarray) -> np.ndarray:
    """Held-out-only 2D subspace: [c^2, c^3] after removing span{1, c}, empirically orthonormal."""
    c = _as_1d_float(c_eval).astype(np.float64, copy=False)
    n = int(c.size)
    if n < 6:
        raise ValueError("Need at least 6 evaluation observations for L_NL.")
    # Nested QR on [1, c, c^2, c^3] in the empirical inner product; keep last two columns.
    a = np.column_stack([np.ones(n), c, c ** 2, c ** 3]).astype(np.float64, copy=False)
    scale = float(np.sqrt(n))
    q, r = np.linalg.qr(a / scale, mode="reduced")
    rinv = np.linalg.inv(r)
    full = a @ rinv
    phi_nl = np.ascontiguousarray(full[:, 2:4], dtype=np.float64)
    for j, raw in enumerate([c ** 2, c ** 3]):
        if float(np.dot(phi_nl[:, j], raw)) < 0.0:
            phi_nl[:, j] *= -1.0
    return phi_nl


def smooth_nl_metrics(e: np.ndarray, phi_nl: np.ndarray) -> Dict[str, float]:
    e = _as_1d_float(e)
    n = float(e.size)
    pe = projector_apply(e, phi_nl)
    l_nl = float(np.dot(e, pe) / n)
    mse = float(np.mean(np.square(e)))
    share = float(l_nl / mse) if mse > 0.0 and np.isfinite(mse) else float("nan")
    return {"L_NL": l_nl, "NL_share": share}


def unit_fairness_gradient(
    *,
    method: str,
    e: np.ndarray,
    c: np.ndarray,
    phi: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    e = _as_1d_float(e)
    c = _as_1d_float(c)
    if method == "current_direct":
        cov = float(np.mean(e * c))
        return 0.5 * cov * c
    if method in MOMENT_K:
        k = method_k(method)
        g, _h, _ = moment_mm_scaled_grad_hess(e, phi[:, :k], rho=1.0)
        return g - e
    if method == "local_slope_smooth":
        g, _h = local_slope_fairness_grad_hess(e, c, epsilon)
        return g
    if method == "quadratic":
        return e * np.square(c)
    raise ValueError(f"Unknown method {method!r}.")


def inspect_current_direct() -> Dict[str, Any]:
    src = inspect.getsource(canonical_direct_scaled_grad_hess)
    hess_src = inspect.getsource(canonical_direct_exact_scaled_hessian)
    cls_src = inspect.getsource(LGBCovPenalty.fobj)
    return {
        "class": "LGBCovPenalty",
        "imported_unchanged": True,
        "module": "soft_constrained_models.boosting_models",
        "canonical_grad": DIRECT_CANONICAL_GRAD,
        "canonical_supplied_hessian": DIRECT_CANONICAL_HESS_SUPPLIED,
        "canonical_exact_hessian": DIRECT_CANONICAL_HESS_EXACT,
        "scaled_grad_hess_qualname": canonical_direct_scaled_grad_hess.__qualname__,
        "uses_canonical_diff_branch": "canonical_direct_scaled_grad_hess" in cls_src,
        "source_excerpt_grad": src.strip().splitlines()[0:12],
        "source_excerpt_exact_hessian": hess_src.strip().splitlines()[0:8],
    }


class ToyMechanismLGB:
    """LightGBM wrapper for Direct-MM / Moment-MM / Local-Slope / Quadratic.

    Current Direct uses LGBCovPenalty instead of this class.
    """

    def __init__(
        self,
        *,
        method: str,
        rho: float,
        lgbm_params,
        phi_train: Optional[np.ndarray] = None,
        c_train: Optional[np.ndarray] = None,
        epsilon: Optional[float] = None,
        verbose: bool = False,
        match_native_init: bool = True,
    ):
        if method not in METHODS:
            raise ValueError(f"Unknown method {method!r}.")
        if method == "current_direct":
            raise ValueError("current_direct must use canonical LGBCovPenalty.")
        self.method = method
        self.rho = float(rho)
        self.verbose = bool(verbose)
        self.match_native_init = bool(match_native_init)
        self.base_score_ = 0.0
        self.y_mean_ = 0.0
        self.epsilon_ = float("nan") if epsilon is None else float(epsilon)
        self.phi_train = None if phi_train is None else np.ascontiguousarray(phi_train, dtype=np.float64)
        self.c_train = None if c_train is None else _as_1d_float(c_train)
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
        if self.method in MOMENT_K:
            if self.phi_train is None or int(self.phi_train.shape[0]) != int(y_fit.size):
                raise ValueError("Moment methods require training Phi aligned with y.")
        if self.method in {"local_slope_smooth", "quadratic"}:
            if self.c_train is None or int(self.c_train.size) != int(y_fit.size):
                raise ValueError("Local-slope/quadratic require training c aligned with y.")
        if self.method == "local_slope_smooth":
            if not np.isfinite(self.epsilon_) or self.epsilon_ <= 0.0:
                raise ValueError("local_slope_smooth requires a frozen positive epsilon.")
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
        if self.method in MOMENT_K:
            k = method_k(self.method)
            grad, hess, extras = moment_mm_scaled_grad_hess(e, self.phi_train[:, :k], self.rho)
            name = f"ToyMechanismLGB.{self.method}"
        elif self.method == "local_slope_smooth":
            grad, hess, extras = local_slope_scaled_grad_hess(e, self.c_train, self.rho, self.epsilon_)
            name = "ToyMechanismLGB.local_slope_smooth"
        elif self.method == "quadratic":
            dummy_true = y_true
            dummy_pred = y_pred
            grad, hess, extras = canonical_surrogate_scaled_grad_hess(
                dummy_true, dummy_pred, y_mean=float(self.y_mean_), rho=self.rho
            )
            name = "ToyMechanismLGB.quadratic"
        else:
            raise ValueError(self.method)
        _assert_finite_grad_hess(grad, hess, name=name)
        if np.any(hess <= 0.0):
            raise FloatingPointError(f"Non-positive supplied Hessian in {name}.")
        if self.verbose:
            print(f"[{name}] rho={self.rho:.6g} extras={ {k: extras[k] for k in extras if k != 'm'} }")
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


def make_current_direct(rho: float, lgbm_params, verbose: bool = False) -> LGBCovPenalty:
    """Canonical practical Direct, unchanged."""
    return LGBCovPenalty(
        rho=float(rho),
        ratio_mode="diff",
        early_stopping_rounds=None,
        lgbm_params=dict(lgbm_params),
        verbose=bool(verbose),
        match_native_init=True,
    )
