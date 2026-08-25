"""EXPERIMENTAL / TOY follow-up metrics: orthogonal N3 shape diagnostics.

Does not change canonical or first-hybrid objectives. The six-path / first-QNL
basis is already the training-only QR orthonormal polynomial map; N3_orth uses
phi_2 and phi_3 (orthogonal to intercept and linear trend).
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from soft_constrained_models.boosting_models import _as_1d_float
from soft_constrained_models.toy_mechanism_objectives import design_monomials


def reconstruct_phi_full(c_train: np.ndarray, sigma_c: float, rinv: np.ndarray) -> np.ndarray:
    """Rebuild the 4-column training basis [phi0, phi1, phi2, phi3] from frozen QR R^{-1}."""
    c = _as_1d_float(c_train)
    z = c / float(sigma_c)
    a = design_monomials(z)
    rinv = np.asarray(rinv, dtype=np.float64)
    phi = a @ rinv
    if phi.shape != (int(c.size), 4):
        raise RuntimeError(f"Expected phi_full shape (n,4); got {phi.shape}.")
    return np.ascontiguousarray(phi, dtype=np.float64)


def orthonormality_diagnostics(phi_full: np.ndarray) -> Dict[str, Any]:
    phi = np.asarray(phi_full, dtype=np.float64)
    n = float(phi.shape[0])
    gram = (phi.T @ phi) / n
    means = np.mean(phi, axis=0)
    z = phi[:, 1]
    p2, p3 = phi[:, 2], phi[:, 3]
    return {
        "n": int(phi.shape[0]),
        "gram": gram.tolist(),
        "gram_offdiag_max": float(np.max(np.abs(gram - np.eye(4)))),
        "gram_diag_max_abs_err": float(np.max(np.abs(np.diag(gram) - 1.0))),
        "means": means.tolist(),
        "phi2_mean": float(means[2]),
        "phi3_mean": float(means[3]),
        "phi2_dot_z": float(np.dot(p2, z) / n),
        "phi3_dot_z": float(np.dot(p3, z) / n),
        "phi2_dot_phi3": float(np.dot(p2, p3) / n),
        "phi0_mean": float(means[0]),
    }


def n3_orth_from_phi3col(e: np.ndarray, phi3: np.ndarray) -> Tuple[float, float, float, float]:
    """phi3 is the six-path (n,3) matrix [phi1, phi2, phi3]. Returns m2, m3, N3, N3_rel."""
    e = _as_1d_float(e)
    phi = np.asarray(phi3, dtype=np.float64)
    n = float(e.size)
    m2 = float(np.dot(phi[:, 1], e) / n)
    m3 = float(np.dot(phi[:, 2], e) / n)
    n3 = m2 ** 2 + m3 ** 2
    var_e = float(np.mean(np.square(e - np.mean(e)))) if e.size else float("nan")
    n3_rel = float(n3 / var_e) if var_e > 0 and np.isfinite(var_e) else float("nan")
    return m2, m3, float(n3), n3_rel
