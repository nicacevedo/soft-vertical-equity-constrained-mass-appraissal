"""LightGBM regressor with an epsilon-soft penalty on Cov(ratio, z).

- Trains in log-space: f(x) \approx log(y)
- Predicts in original space: y_hat = exp(f)
- Ratio: r_i = y_hat_i / y_i
- Penalty: rho * softplus((|Cov_w(r, z)| - eps)/tau)
  where z is either log(y) or y.

This is intended as a sklearn-like wrapper around `lightgbm.train`.

Notes
-----
- Requires y > 0 (so log is defined).
- Uses a diagonal Hessian approximation (base squared-error Hessian only) for stability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

try:
    import lightgbm as lgb
except ImportError as e:  # pragma: no cover
    raise ImportError("This module requires lightgbm. Install via `pip install lightgbm`.") from e

try:
    from sklearn.base import BaseEstimator, RegressorMixin
except ImportError as e:  # pragma: no cover
    raise ImportError("This module requires scikit-learn. Install via `pip install scikit-learn`.") from e


# ============================= MAIN MODELS =============================

# ==========================================================
# 0) Direct covariance penalty (non-separable but usable in LightGBM via indep. assumption)
# ==========================================================

# V2: with diff/div inputs
class LGBCovPenalty:
    """LightGBM objective: MSE + rho * (Cov(r, y))^2

    r is chosen by ratio_mode:
      - "div"  : r = y_pred / max(|y_true|, eps_y)    (DEFAULT, preserves old behavior)
      - "diff" : r = y_pred - y_true                 (useful when y is log-price -> log-residual)

    Cov is computed as: cov = mean( r_eff * (y_true - y_mean_) ),
    where r_eff may optionally be shifted by an "anchor" (see below).

    Anchor note (important):
      Because yc = (y_true - y_mean_) is mean-centered, subtracting any *constant* anchor
      from r does not change cov (up to floating error). Therefore anchor_mode/target_value
      are effectively no-ops for this specific cov definition. Included only for API symmetry.

    Diagonal Hessian approximation (as before):
      cov = (1/n) * sum_i r_i * yc_i
      dc/dy_pred_i = (1/n) * yc_i * d r_i / d y_pred_i
      penalty = 0.5 * rho * n * cov^2
      grad_pen_i = rho * n * cov * dc/dy_pred_i
      hess_pen_i = rho * n * (dc/dy_pred_i)^2
    """

    def __init__(
        self,
        rho=1e-3,
        ratio_mode="div",          # "div" or "diff"
        anchor_mode="target",        # "none" | "target" | "iter_mean"  (no-op here; see note)
        target_value=None,         # if anchor_mode="target": default 1.0 (div) or 0.0 (diff)
        zero_grad_tol=1e-6,
        eps_y=1e-12,
        lgbm_params=None,
        verbose=True,
    ):
        self.rho = float(rho)
        self.ratio_mode = ratio_mode
        self.anchor_mode = anchor_mode
        self.target_value = target_value
        self.zero_grad_tol = float(zero_grad_tol)
        self.eps_y = float(eps_y)
        self.verbose = bool(verbose)
        self.model = lgb.LGBMRegressor(**(lgbm_params or {}))

    def fit(self, X, y):
        self.y_mean_ = float(np.mean(y))
        self.model.set_params(objective=self.fobj)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def fobj(self, y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        n = y_pred.size

        yc = (y_true - self.y_mean_)  # centered y (note: mean(yc)≈0 on training data)

        # ---- choose r and dr/dy_pred ----
        if self.ratio_mode == "div":
            denom = np.maximum(np.abs(y_true), self.eps_y)
            r = y_pred / denom
            dr = 1.0 / denom
        elif self.ratio_mode == "diff":
            r = y_pred - y_true
            dr = np.ones_like(y_pred)
        else:
            raise ValueError("ratio_mode must be 'div' or 'diff'.")

        # ---- optional anchor (no-op for centered yc; kept for API symmetry) ----
        anchor = 0.0
        if self.anchor_mode == "none":
            anchor = 0.0
        elif self.anchor_mode == "iter_mean":
            anchor = float(np.mean(r))
        elif self.anchor_mode == "target":
            if self.target_value is None:
                anchor = 1.0 if self.ratio_mode == "div" else 0.0
            else:
                anchor = float(self.target_value)
        else:
            raise ValueError("anchor_mode must be 'none', 'iter_mean', or 'target'.")

        r_eff = r - anchor  # (effectively no change to cov because mean(yc)=0)

        # ---- covariance (note: E[yc]=0 on training set) ----
        cov = float(np.mean(r_eff * yc))

        # ---- objective pieces (for prints) ----
        mse_vec = (y_true - y_pred) ** 2
        mse_mean = float(np.mean(mse_vec))
        pen_value = 0.5 * self.rho * float(n) * (cov ** 2)

        try:
            corr = float(np.corrcoef(r, y_true)[0, 1])
        except Exception:
            corr = float("nan")

        if self.verbose:
            model_name = self.__str__().split("(")[0]
            print(
                f"[{model_name}] "
                f"Loss: {(mse_mean + pen_value):.6f} | MSE: {mse_mean:.6f} | "
                f"Cov: {cov:.6e} | Pen: {pen_value:.6f} | Corr(r,y): {corr:.6f}"
            )

        # ---- base MSE grads/hess ----
        grad_base = 2.0 * (y_pred - y_true)
        hess_base = 2.0 * np.ones_like(y_pred)

        # ---- cov penalty grads/hess (diag approx) ----
        # dc/dy_pred_i = (1/n) * yc_i * dr_i
        a = (yc * dr) / float(n)

        grad_pen = self.rho * float(n) * cov * a
        hess_pen = self.rho * float(n) * (a ** 2)

        grad = grad_base + grad_pen
        hess = hess_base + hess_pen

        grad[np.abs(grad) < self.zero_grad_tol] = self.zero_grad_tol
        hess[hess < self.zero_grad_tol] = self.zero_grad_tol

        return grad, hess

    def __str__(self):
        return f"LGBCovPenalty(rho={self.rho}, ratio_mode={self.ratio_mode})" #, anchor_mode={self.anchor_mode})"



# # CVaR Surrogate of the MSE
# class LGBCovPenaltyCVaR:
#     """LightGBM objective: (robust) MSE + rho * (Cov(r, y))^2

#     Change vs original:
#       - Adds CVaR on the MSE term ONLY (penalty unchanged).
#       - When mse_keep < 1, we focus on the worst mse_keep fraction of squared errors.
#       - Proper scaling: use (n / n_keep) multiplier on the selected tail so mse_keep=1
#         matches the original mean-MSE gradient scale, and mse_keep<1 increases emphasis
#         on the tail (roughly by 1/mse_keep).

#     Existing penalty (unchanged):
#       penalty = 0.5 * rho * n * cov^2
#       cov = mean(r_eff * (y_true - y_mean_))
#       grad_pen_i = rho * n * cov * a_i,    a_i = (yc_i * dr_i) / n
#       hess_pen_i = rho * n * a_i^2         (diag approx)
#     """

#     def __init__(
#         self,
#         rho=1e-3,
#         ratio_mode="div",            # "div" or "diff"
#         anchor_mode="target",        # "none" | "target" | "iter_mean"  (no-op here; see note)
#         target_value=None,           # if anchor_mode="target": default 1.0 (div) or 0.0 (diff)

#         # --- CVaR on MSE ---
#         mse_keep=1.0,                # 1.0 => original mean MSE; <1 => CVaR tail mean of squared errors
#         mse_mix_uniform=0.0,         # optional mixing with uniform weights in [0,1); default 0.0

#         zero_grad_tol=1e-6,
#         eps_y=1e-12,
#         lgbm_params=None,
#         verbose=True,
#     ):
#         self.rho = float(rho)
#         self.ratio_mode = ratio_mode
#         self.anchor_mode = anchor_mode
#         self.target_value = target_value

#         self.mse_keep = float(mse_keep)
#         self.mse_mix_uniform = float(mse_mix_uniform)

#         self.zero_grad_tol = float(zero_grad_tol)
#         self.eps_y = float(eps_y)
#         self.verbose = bool(verbose)
#         self.model = lgb.LGBMRegressor(**(lgbm_params or {}))

#     def fit(self, X, y):
#         self.y_mean_ = float(np.mean(y))
#         self.model.set_params(objective=self.fobj)
#         self.model.fit(X, y)
#         return self

#     def predict(self, X):
#         return self.model.predict(X)

#     # ------------------------
#     # CVaR weights for MSE
#     # ------------------------
#     def _cvar_mse_weights(self, sq_err):
#         """
#         Returns:
#           w_eff: per-sample multiplier applied to base MSE grad/hess
#           n_keep: number of samples in tail set
#           cvar_mse: tail-mean squared error (for logging)
#         """
#         n = sq_err.size
#         keep = float(self.mse_keep)

#         if keep >= 1.0:
#             w_eff = np.ones(n, dtype=float)
#             return w_eff, n, float(np.mean(sq_err))

#         keep = max(min(keep, 1.0), 1.0 / n)
#         n_keep = int(np.ceil(keep * n))

#         idx = np.argpartition(sq_err, -n_keep)[-n_keep:]
#         mask = np.zeros(n, dtype=float)
#         mask[idx] = 1.0

#         # scaling so keep=1 recovers original mean-MSE scale
#         w_eff = (n / float(n_keep)) * mask

#         mix = float(self.mse_mix_uniform)
#         if mix > 0.0:
#             mix = min(max(mix, 0.0), 0.999)
#             w_eff = (1.0 - mix) * w_eff + mix * np.ones(n, dtype=float)

#         return w_eff, n_keep, float(np.mean(sq_err[idx]))

#     def fobj(self, y_true, y_pred):
#         y_true = np.asarray(y_true, dtype=float)
#         y_pred = np.asarray(y_pred, dtype=float)
#         n = y_pred.size

#         yc = (y_true - self.y_mean_)  # centered y (mean ~ 0 on training)

#         # ---- choose r and dr/dy_pred ----
#         if self.ratio_mode == "div":
#             denom = np.maximum(np.abs(y_true), self.eps_y)
#             r = y_pred / denom
#             dr = 1.0 / denom
#         elif self.ratio_mode == "diff":
#             r = y_pred - y_true
#             dr = np.ones_like(y_pred)
#         else:
#             raise ValueError("ratio_mode must be 'div' or 'diff'.")

#         # ---- optional anchor (no-op for centered yc; kept for API symmetry) ----
#         anchor = 0.0
#         if self.anchor_mode == "none":
#             anchor = 0.0
#         elif self.anchor_mode == "iter_mean":
#             anchor = float(np.mean(r))
#         elif self.anchor_mode == "target":
#             if self.target_value is None:
#                 anchor = 1.0 if self.ratio_mode == "div" else 0.0
#             else:
#                 anchor = float(self.target_value)
#         else:
#             raise ValueError("anchor_mode must be 'none', 'iter_mean', or 'target'.")

#         r_eff = r - anchor

#         # ---- covariance ----
#         cov = float(np.mean(r_eff * yc))

#         # ---- MSE pieces ----
#         mse_vec = (y_true - y_pred) ** 2
#         mse_mean = float(np.mean(mse_vec))

#         # CVaR weights for MSE only
#         w_mse_eff, n_keep, cvar_mse = self._cvar_mse_weights(mse_vec)

#         # ---- penalty value for logging (unchanged) ----
#         pen_value = 0.5 * self.rho * float(n) * (cov ** 2)

#         try:
#             corr = float(np.corrcoef(r, y_true)[0, 1])
#         except Exception:
#             corr = float("nan")

#         if self.verbose:
#             model_name = self.__str__().split("(")[0]
#             # a proxy "loss" using CVaR-MSE + penalty
#             loss_proxy = cvar_mse + pen_value
#             print(
#                 f"[{model_name}] "
#                 f"Loss~: {loss_proxy:.6f} | MSE(mean): {mse_mean:.6f} | "
#                 f"MSE(CVaR@keep={self.mse_keep:.2f}): {cvar_mse:.6f} | "
#                 f"Cov: {cov:.6e} | Pen: {pen_value:.6f} | Corr(r,y): {corr:.6f} | "
#                 f"n_keep: {n_keep}"
#             )

#         # ---- base MSE grads/hess (weighted by CVaR) ----
#         grad_base = 2.0 * (y_pred - y_true)
#         hess_base = 2.0 * np.ones_like(y_pred)

#         grad_base = w_mse_eff * grad_base
#         hess_base = w_mse_eff * hess_base

#         # ---- cov penalty grads/hess (unchanged) ----
#         a = (yc * dr) / float(n)                 # dc/dy_pred_i
#         grad_pen = self.rho * float(n) * cov * a
#         hess_pen = self.rho * float(n) * (a ** 2)

#         grad = grad_base + grad_pen
#         hess = hess_base + hess_pen

#         grad[np.abs(grad) < self.zero_grad_tol] = self.zero_grad_tol
#         hess[hess < self.zero_grad_tol] = self.zero_grad_tol

#         return grad, hess

#     def __str__(self):
#         return f"LGBCovPenalty(rho={self.rho}, ratio_mode={self.ratio_mode}, mse_keep={self.mse_keep})"


# ==========================================================
# 1) Plain (no dual/adversary): minimize MSE + rho * Cov-surrogate (separable)
# ==========================================================

# V2: with diff/div inputs
class LGBSmoothPenalty:
    """LightGBM custom objective: per-sample MSE + rho * separable surrogate.

    Default behavior (unchanged):
      ratio_mode="div": surrogate = ((y_pred / denom) - 1)^2 * (y_true - y_mean)^2

    Added minimal option:
      ratio_mode="diff": surrogate = ((y_pred - y_true) - 0)^2 * (y_true - y_mean)^2

    Optional minimal target:
      - If ratio_mode="div": target_value defaults to 1.0
      - If ratio_mode="diff": target_value defaults to 0.0
      surrogate = (r - target_value)^2 * zc^2
    where:
      - r = y_pred / denom   (div)
      - r = y_pred - y_true  (diff)

    Notes:
      - This remains separable (per-sample), so grad/hess are exact and diagonal.
      - This does *not* exactly penalize covariance; it penalizes "ratio/residual magnitude"
        weighted by (y - mean(y))^2 (your original structure).
    """

    def __init__(
        self,
        rho=1e-3,
        ratio_mode="div",        # "div" (default) or "diff"
        target_value=None,       # default: 1.0 for div, 0.0 for diff
        zero_grad_tol=1e-6,
        eps_y=1e-12,
        lgbm_params=None,
        verbose=True,
    ):
        self.rho = float(rho)
        self.ratio_mode = ratio_mode
        self.target_value = target_value
        self.zero_grad_tol = float(zero_grad_tol)
        self.eps_y = float(eps_y)
        self.verbose = bool(verbose)
        self.model = lgb.LGBMRegressor(**(lgbm_params or {}))

    def fit(self, X, y):
        self.y_mean_ = float(np.mean(y))
        self.model.set_params(objective=self.fobj)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def fobj(self, y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        z = y_true
        zc = (y_true - self.y_mean_)
        denom = np.maximum(np.abs(z), self.eps_y)

        # choose r, dr/dy_pred, and default target
        if self.ratio_mode == "div":
            r = y_pred / denom
            dr = 1.0 / denom
            t = 1.0 if self.target_value is None else float(self.target_value)
        elif self.ratio_mode == "diff":
            r = y_pred - z
            dr = np.ones_like(y_pred)
            t = 0.0 if self.target_value is None else float(self.target_value)
        else:
            raise ValueError("ratio_mode must be 'div' or 'diff'.")

        # losses
        mse_value = (y_true - y_pred) ** 2
        cov_surr_value = (r - t) ** 2 * (zc ** 2)
        loss_value = mse_value + self.rho * cov_surr_value

        # print (kept close to yours)
        if self.verbose:
            model_name = self.__str__()
            try:
                corr = float(np.corrcoef(r, y_true)[0, 1])
            except Exception:
                corr = float("nan")
            print(
                f"[{model_name.split('(')[0]}] "
                f"Loss value: {np.mean(loss_value):.6f} "
                f"| MSE value: {np.mean(mse_value):.6f} "
                f"| CovSurr value: {np.mean(cov_surr_value):.6f} "
                f"| Corr(r,y): {corr:.6f} "
                f"| mode: {self.ratio_mode} | target: {t:.6f}"
            )

        # base gradients/hessians for (pred-y)^2
        grad_base = 2.0 * (y_pred - y_true)
        hess_base = 2.0 * np.ones_like(y_pred)

        # penalty gradients/hessians (separable, exact)
        # pen_i = (r_i - t)^2 * zc_i^2
        # d pen_i / d pred_i = 2 * (r_i - t) * dr_i * zc_i^2
        scale = (zc ** 2)
        grad_pen = 2.0 * (r - t) * dr * scale
        hess_pen = 2.0 * (dr ** 2) * scale

        grad = grad_base + self.rho * grad_pen
        hess = hess_base + self.rho * hess_pen

        # zero tol
        grad[np.abs(grad) < self.zero_grad_tol] = self.zero_grad_tol
        hess[hess < self.zero_grad_tol] = self.zero_grad_tol

        return grad, hess

    def __str__(self):
        return f"LGBSmoothPenalty(rho={self.rho}, mode={self.ratio_mode})"



# CVaR Surrogate of the
class LGBSmoothPenaltyCVaR:
    """LightGBM custom objective: robust MSE (CVaR) + rho * separable surrogate penalty.

    Base objective:
      - If mse_keep == 1.0: mean MSE (original behavior)
      - If mse_keep <  1.0: CVaR_keep of per-sample squared errors, implemented via top-k reweighting

    Penalty term (unchanged, global average, non-adversarial):
      ratio_mode="div": surrogate = ((y_pred / denom) - t)^2 * (y_true - y_mean)^2
      ratio_mode="diff": surrogate = ((y_pred - y_true) - t)^2 * (y_true - y_mean)^2
      with defaults: t=1.0 for div, t=0.0 for diff (unless target_value provided).

    Notes on CVaR implementation:
      - We compute squared errors s_i = (y_true - y_pred)^2.
      - Let n_keep = ceil(mse_keep * n). We select the top-n_keep errors.
      - We apply per-sample weights:
            w_eff_i = (n / n_keep) * 1{i in top-k}
        so that when mse_keep=1, w_eff_i = 1 (original scaling).
      - Optional mild mixing with uniform weights:
            w_eff_i = (1-mix)* (n/n_keep)*1{i in top-k} + mix*1
        This reduces oscillations when the top-k set changes a lot.
      - This is a piecewise-smooth objective (the top-k set can change as y_pred changes).
        In practice it works well as a robustification heuristic.
    """

    def __init__(
        self,
        rho=1e-3,
        ratio_mode="div",        # "div" (default) or "diff"
        target_value=None,       # default: 1.0 for div, 0.0 for diff

        # --- CVaR on MSE ---
        mse_keep=1.0,            # keep fraction for CVaR on squared error; 1.0 => original mean MSE
        mse_mix_uniform=0.0,     # optional mixing with uniform weights in [0,1); default 0.0

        # --- numerical ---
        zero_grad_tol=1e-6,
        eps_y=1e-12,
        verbose=True,

        # --- LightGBM ---
        lgbm_params=None,
    ):
        self.rho = float(rho)
        self.ratio_mode = ratio_mode
        self.target_value = target_value

        self.mse_keep = float(mse_keep)
        self.mse_mix_uniform = float(mse_mix_uniform)

        self.zero_grad_tol = float(zero_grad_tol)
        self.eps_y = float(eps_y)
        self.verbose = bool(verbose)

        self.model = lgb.LGBMRegressor(**(lgbm_params or {}))

    def fit(self, X, y):
        self.y_mean_ = float(np.mean(y))
        self.model.set_params(objective=self.fobj)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    # ------------------------
    # CVaR weights for MSE
    # ------------------------
    def _cvar_mse_weights(self, sq_err):
        """
        Returns:
          w_eff: per-sample multiplier applied to base grad/hess (MSE part only)
          n_keep: number of samples in the tail set
          cvar_mse: tail mean of squared errors (for logging)
        """
        n = sq_err.size
        keep = float(self.mse_keep)

        # Original mean MSE
        if keep >= 1.0:
            w_eff = np.ones(n, dtype=float)
            cvar_mse = float(np.mean(sq_err))
            return w_eff, n, cvar_mse

        # Clamp keep to at least 1 sample
        keep = max(min(keep, 1.0), 1.0 / n)
        n_keep = int(np.ceil(keep * n))

        # Top-k indices of squared error
        idx = np.argpartition(sq_err, -n_keep)[-n_keep:]

        mask = np.zeros(n, dtype=float)
        mask[idx] = 1.0

        # CVaR scaling: mean over top-k is (1/n_keep) sum_{idx} s_i.
        # To match the original mean-MSE gradient scale (which corresponds to (1/n) sum s_i),
        # we multiply selected points by n/n_keep.
        w_eff = (n / float(n_keep)) * mask

        # Optional mild mixing with uniform (stability)
        mix = float(self.mse_mix_uniform)
        if mix > 0.0:
            mix = min(max(mix, 0.0), 0.999)
            # convex combo: (1-mix)*CVaR + mix*mean
            w_eff = (1.0 - mix) * w_eff + mix * np.ones(n, dtype=float)

        cvar_mse = float(np.mean(sq_err[idx]))
        return w_eff, n_keep, cvar_mse

    def fobj(self, y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        z = y_true
        zc = (y_true - self.y_mean_)
        denom = np.maximum(np.abs(z), self.eps_y)

        # choose r, dr/dy_pred, and default target
        if self.ratio_mode == "div":
            r = y_pred / denom
            dr = 1.0 / denom
            t = 1.0 if self.target_value is None else float(self.target_value)
        elif self.ratio_mode == "diff":
            r = y_pred - z
            dr = np.ones_like(y_pred)
            t = 0.0 if self.target_value is None else float(self.target_value)
        else:
            raise ValueError("ratio_mode must be 'div' or 'diff'.")

        # ------------------------
        # MSE + CVaR(MSE) weighting
        # ------------------------
        sq_err = (y_true - y_pred) ** 2
        w_mse_eff, n_keep, cvar_mse = self._cvar_mse_weights(sq_err)

        # base gradients/hessians for squared error
        grad_base = 2.0 * (y_pred - y_true)
        hess_base = 2.0 * np.ones_like(y_pred)

        # Apply CVaR weights ONLY to the MSE term
        grad_base = w_mse_eff * grad_base
        hess_base = w_mse_eff * hess_base

        # ------------------------
        # penalty (unchanged, global mean behavior via uniform aggregation)
        # ------------------------
        # pen_i = (r_i - t)^2 * zc_i^2
        scale = (zc ** 2)
        cov_surr_value = (r - t) ** 2 * scale

        grad_pen = 2.0 * (r - t) * dr * scale
        hess_pen = 2.0 * (dr ** 2) * scale

        grad = grad_base + self.rho * grad_pen
        hess = hess_base + self.rho * hess_pen

        # Logging (kept close, now includes CVaR info)
        if self.verbose:
            try:
                corr = float(np.corrcoef(r, y_true)[0, 1])
            except Exception:
                corr = float("nan")

            # "Mean MSE" (unweighted) is still useful to track, plus CVaR tail mean
            mse_mean = float(np.mean(sq_err))
            cov_surr_mean = float(np.mean(cov_surr_value))

            # A comparable "objective-like" scalar (rough): mean(MSE) replaced by CVaR(MSE)
            # penalty stays as mean
            loss_proxy = cvar_mse + self.rho * cov_surr_mean

            print(
                f"[{self.__str__().split('(')[0]}] "
                f"Loss~: {loss_proxy:.6f} "
                f"| MSE(mean): {mse_mean:.6f} "
                f"| MSE(CVaR@keep={self.mse_keep:.2f}): {cvar_mse:.6f} "
                f"| CovSurr(mean): {cov_surr_mean:.6f} "
                f"| Corr(r,y): {corr:.6f} "
                f"| n_keep: {n_keep} | mode: {self.ratio_mode} | target: {t:.6f}"
            )

        # numerical floor
        grad[np.abs(grad) < self.zero_grad_tol] = self.zero_grad_tol
        hess[hess < self.zero_grad_tol] = self.zero_grad_tol

        return grad, hess

    def __str__(self):
        return f"LGBSmoothPenalty(rho={self.rho}, mode={self.ratio_mode}, mse_keep={self.mse_keep})"



# CVar on the whole objective (MSE + rho*penalty)
class LGBSmoothPenaltyCVaRTotal:
    """LightGBM custom objective: CVaR_keep of (MSE + rho*penalty) using top-k reweighting.

    Compared to LGBSmoothPenaltyCVaR (CVaR only on MSE):
      - We select the worst tail based on per-sample total score:
            score_i = (y_true - y_pred)^2 + rho * ((r - t)^2 * zc^2)
      - We apply the resulting CVaR weights to BOTH MSE and penalty gradients/hessians.

    Scaling:
      - weights w_eff_i = (n / n_keep) * 1{i in top-k}
        so keep=1 => w_eff_i=1 (original scale).

    Optional mild mixing:
      - w_eff = (1-mix)*w_eff + mix*1
        stabilizes training when the top-k set changes abruptly.
    """

    def __init__(
        self,
        rho=1e-3,
        ratio_mode="div",        # "div" or "diff"
        target_value=None,       # default: 1.0 for div, 0.0 for diff

        # --- CVaR on total (MSE + rho*penalty) ---
        keep=1.0,                # keep fraction for CVaR tail; 1.0 => original mean objective scaling
        mix_uniform=0.0,         # optional mixing with uniform weights in [0,1); default 0.0

        # --- numerical ---
        zero_grad_tol=1e-6,
        eps_y=1e-12,
        verbose=True,

        # --- LightGBM ---
        lgbm_params=None,
    ):
        self.rho = float(rho)
        self.ratio_mode = ratio_mode
        self.target_value = target_value

        self.keep = float(keep)
        self.mix_uniform = float(mix_uniform)

        self.zero_grad_tol = float(zero_grad_tol)
        self.eps_y = float(eps_y)
        self.verbose = bool(verbose)

        self.model = lgb.LGBMRegressor(**(lgbm_params or {}))

    def fit(self, X, y):
        self.y_mean_ = float(np.mean(y))
        self.model.set_params(objective=self.fobj)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    # ------------------------
    # CVaR weights for a generic per-sample score
    # ------------------------
    def _cvar_weights(self, score):
        """
        Returns:
          w_eff: per-sample multiplier applied to (grad_total, hess_total)
          n_keep: number of samples in the tail set
          cvar_score: tail mean of score (for logging)
        """
        n = score.size
        keep = float(self.keep)

        # Original: no CVaR (equivalent to keep=1)
        if keep >= 1.0:
            w_eff = np.ones(n, dtype=float)
            return w_eff, n, float(np.mean(score))

        keep = max(min(keep, 1.0), 1.0 / n)
        n_keep = int(np.ceil(keep * n))

        idx = np.argpartition(score, -n_keep)[-n_keep:]
        mask = np.zeros(n, dtype=float)
        mask[idx] = 1.0

        w_eff = (n / float(n_keep)) * mask

        mix = float(self.mix_uniform)
        if mix > 0.0:
            mix = min(max(mix, 0.0), 0.999)
            w_eff = (1.0 - mix) * w_eff + mix * np.ones(n, dtype=float)

        cvar_score = float(np.mean(score[idx]))
        return w_eff, n_keep, cvar_score

    def fobj(self, y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        z = y_true
        zc = (y_true - self.y_mean_)
        denom = np.maximum(np.abs(z), self.eps_y)

        # choose r, dr/dy_pred, and default target
        if self.ratio_mode == "div":
            r = y_pred / denom
            dr = 1.0 / denom
            t = 1.0 if self.target_value is None else float(self.target_value)
        elif self.ratio_mode == "diff":
            r = y_pred - z
            dr = np.ones_like(y_pred)
            t = 0.0 if self.target_value is None else float(self.target_value)
        else:
            raise ValueError("ratio_mode must be 'div' or 'diff'.")

        # per-sample pieces
        sq_err = (y_true - y_pred) ** 2
        scale = (zc ** 2)
        pen_i = (r - t) ** 2 * scale

        # CVaR selection score: total per-sample objective
        score = sq_err + self.rho * pen_i
        w_eff, n_keep, cvar_score = self._cvar_weights(score)

        # gradients/hessians (separable parts)
        grad_mse = 2.0 * (y_pred - y_true)
        hess_mse = 2.0 * np.ones_like(y_pred)

        grad_pen = 2.0 * (r - t) * dr * scale
        hess_pen = 2.0 * (dr ** 2) * scale

        # Apply CVaR weights to the TOTAL gradient/hessian
        grad = w_eff * (grad_mse + self.rho * grad_pen)
        hess = w_eff * (hess_mse + self.rho * hess_pen)

        # Logging
        if self.verbose:
            mse_mean = float(np.mean(sq_err))
            pen_mean = float(np.mean(pen_i))
            score_mean = float(np.mean(score))

            try:
                corr = float(np.corrcoef(r, y_true)[0, 1])
            except Exception:
                corr = float("nan")

            print(
                f"[{self.__str__().split('(')[0]}] "
                f"Score(mean): {score_mean:.6f} | Score(CVaR@keep={self.keep:.2f}): {cvar_score:.6f} | "
                f"MSE(mean): {mse_mean:.6f} | Pen(mean): {pen_mean:.6f} | "
                f"rho: {self.rho:.3g} | Corr(r,y): {corr:.6f} | n_keep: {n_keep} | "
                f"mode: {self.ratio_mode} | target: {t:.6f}"
            )

        # numerical floor
        grad[np.abs(grad) < self.zero_grad_tol] = self.zero_grad_tol
        hess[hess < self.zero_grad_tol] = self.zero_grad_tol

        return grad, hess

    def __str__(self):
        return f"LGBSmoothPenaltyCVaRTotal(rho={self.rho}, mode={self.ratio_mode}, keep={self.keep})"






# ============================= EXPERIMENTAL MODELS =============================

# ==========================================================
# 2) Improved primal-dual: mirror ascent adversary + KL projection onto capped simplex
# ==========================================================

# class LGBPrimalDualImproved:
#     """Primal-dual robust boosting with a *principled* dual step on the capped simplex.

#     Objective (overall):  min_F max_{w in Delta_K} sum_i w_i [mse_i + rho*surr_i]
#     Objective (individual): min_F max_p sum_i p_i mse_i + rho max_q sum_i q_i surr_i

#     Dual update options:
#       - dual_update="topk": exact best-response (uniform on worst-K)
#       - dual_update="mirror": exponentiated-gradient + KL projection onto capped simplex

#     Notes vs your current code:
#       - Uses a KL/Bregman projection consistent with exponentiated-gradient:
#             w = min(cap, u / Z) with Z chosen so sum w = 1
#         (found by bisection on Z).
#       - Adds eps_y to avoid division blow-ups.
#       - Keeps everything else close to your original structure.
#     """

#     def __init__(
#         self,
#         rho=1e-3,
#         keep=0.7,
#         adversary_type="overall",
#         dual_update="mirror",  # "mirror" or "topk"
#         eta_adv=0.1,
#         zero_grad_tol=1e-6,
#         eps_y=1e-12,
#         lgbm_params=None,
#     ):
#         self.rho = rho
#         self.keep = keep
#         self.adversary_type = adversary_type
#         self.dual_update = dual_update
#         self.eta_adv = eta_adv
#         self.zero_grad_tol = zero_grad_tol
#         self.eps_y = eps_y
#         self.model = lgb.LGBMRegressor(**(lgbm_params or {}))

#     def fit(self, X, y):
#         # cache for the callback
#         self.X_ = X
#         self.y_ = np.asarray(y)
#         self.y_mean_ = float(np.mean(self.y_))
#         self.n_ = int(self.y_.size)

#         # cached ensemble predictions (match boost_from_average=True for regression)
#         self.y_hat_ = np.ones(self.n_) * self.y_mean_

#         # CVaR/top-k cap: w_i <= 1/K, with K = keep*n
#         self.K_ = max(1, int(self.keep * self.n_))
#         self.cap_ = 1.0 / float(self.K_)

#         w0 = np.ones(self.n_) / float(self.n_)
#         if self.adversary_type == "overall":
#             self.w_ = w0
#         elif self.adversary_type == "individual":
#             self.p_ = w0.copy()
#             self.q_ = w0.copy()
#         else:
#             raise ValueError(f"No adversary_type called: {self.adversary_type}")

#         self.model.set_params(objective=self.fobj)
#         self.model.fit(X, y, callbacks=[self._adv_callback])
#         return self

#     def predict(self, X):
#         return self.model.predict(X)

#     # ---------- Dual helpers ----------

#     def _project_capped_simplex_kl(self, u):
#         """KL/Bregman projection of u onto {w>=0, sum w=1, w_i<=cap_}.

#         The solution has the form w_i = min(cap, u_i / Z) with Z>0 chosen so sum w = 1.
#         """
#         u = np.asarray(u, dtype=float)
#         u = np.maximum(u, 0.0) + 1e-300
#         cap = float(self.cap_)

#         # If already feasible after normalization and capping, quick exit
#         # (not strictly necessary, but cheap)
#         Z_hi = float(np.sum(u))
#         if not np.isfinite(Z_hi) or Z_hi <= 0:
#             return np.ones_like(u) / u.size

#         def s(Z):
#             return float(np.sum(np.minimum(cap, u / Z)))

#         # We want s(Z)=1. s is decreasing in Z.
#         Z_lo = 1e-300
#         # Ensure s(Z_lo) >= 1 (should hold if cap*n >= 1)
#         if s(Z_lo) < 1.0 - 1e-12:
#             # fallback to top-k feasible point
#             w = np.zeros_like(u)
#             idx = np.argpartition(u, -self.K_)[-self.K_:]
#             w[idx] = 1.0 / float(self.K_)
#             return w

#         # Z_hi gives s(Z_hi) <= 1
#         if s(Z_hi) > 1.0 + 1e-12:
#             # expand hi until it is above the root
#             for _ in range(60):
#                 Z_hi *= 2.0
#                 if s(Z_hi) <= 1.0 + 1e-12:
#                     break

#         # bisection
#         for _ in range(80):
#             Z_mid = 0.5 * (Z_lo + Z_hi)
#             sm = s(Z_mid)
#             if abs(sm - 1.0) <= 1e-12:
#                 Z_lo = Z_hi = Z_mid
#                 break
#             if sm > 1.0:
#                 Z_lo = Z_mid
#             else:
#                 Z_hi = Z_mid

#         Z_star = 0.5 * (Z_lo + Z_hi)
#         w = np.minimum(cap, u / Z_star)
#         w_sum = float(np.sum(w))
#         if not np.isfinite(w_sum) or w_sum <= 0:
#             w = np.ones_like(u) / u.size
#         else:
#             w /= w_sum
#         return w

#     def _dual_step(self, w, v):
#         """Update weights for max_{w in capped simplex} <w, v>."""
#         if self.dual_update == "topk":
#             # exact best response: uniform mass on worst-K v_i
#             idx = np.argpartition(v, -self.K_)[-self.K_:]
#             w_new = np.zeros_like(w)
#             w_new[idx] = 1.0 / float(self.K_)
#             return w_new

#         # mirror ascent (exponentiated-gradient) + KL projection
#         z = self.eta_adv * (v - np.max(v))
#         z = np.clip(z, -50.0, 50.0)
#         u = w * np.exp(z)
#         return self._project_capped_simplex_kl(u)

#     # ---------- Callback (dual update) ----------

#     def _adv_callback(self, env):
#         it = int(env.iteration) + 1

#         # Incremental prediction update: add only the latest tree contribution
#         delta = env.model.predict(self.X_, start_iteration=it - 1, num_iteration=1)
#         self.y_hat_ = self.y_hat_ + delta
#         y_hat = self.y_hat_

#         denom = np.maximum(np.abs(self.y_), self.eps_y)
#         mse_value = (self.y_ - y_hat) ** 2
#         cov_surr_value = ((y_hat / denom) - 1.0) ** 2 * (self.y_ - self.y_mean_) ** 2

#         if self.adversary_type == "overall":
#             v = mse_value + self.rho * cov_surr_value
#             self.w_ = self._dual_step(self.w_, v)
#         else:
#             self.p_ = self._dual_step(self.p_, mse_value)
#             self.q_ = self._dual_step(self.q_, cov_surr_value)

#     # ---------- Objective ----------

#     def fobj(self, y_true, y_pred):
#         y_true = np.asarray(y_true)
#         y_pred = np.asarray(y_pred)

#         # stable pieces
#         z = y_true
#         zc = (y_true - self.y_mean_)
#         denom = np.maximum(np.abs(z), self.eps_y)

#         # per-sample values (for prints)
#         mse_value = (y_true - y_pred) ** 2
#         cov_surr_value = ((y_pred / denom) - 1.0) ** 2 * (zc ** 2)
#         loss_value = mse_value + self.rho * cov_surr_value

#         model_name = self.__str__()
#         r = y_pred / denom
#         try:
#             corr = float(np.corrcoef(r, y_true)[0, 1])
#         except Exception:
#             corr = float('nan')
#         print(
#             f"[{model_name.split('(')[0]}] "
#             f"Loss value: {np.mean(loss_value):.6f} "
#             f"| MSE value: {np.mean(mse_value):.6f} "
#             f"| CovSurr value: {np.mean(cov_surr_value):.6f} "
#             f"| Corr(r,y): {corr:.6f} "
#         )

#         # base gradients/hessians
#         grad_base = 2.0 * (y_pred - y_true)
#         hess_base = 2.0 * np.ones_like(y_pred)

#         # surrogate grads/hess (separable)
#         scale = (zc / denom) ** 2
#         grad_pen = 2.0 * (y_pred - z) * scale
#         hess_pen = 2.0 * scale

#         n = y_pred.size
#         if self.adversary_type == "overall":
#             w_eff = float(n) * self.w_
#             grad = w_eff * (grad_base + self.rho * grad_pen)
#             hess = w_eff * (hess_base + self.rho * hess_pen)
#         else:
#             p_eff = float(n) * self.p_
#             q_eff = float(n) * self.q_
#             grad = p_eff * grad_base + self.rho * q_eff * grad_pen
#             hess = p_eff * hess_base + self.rho * q_eff * hess_pen

#         # tolerances
#         grad[np.abs(grad) < self.zero_grad_tol] = self.zero_grad_tol
#         hess[hess < self.zero_grad_tol] = self.zero_grad_tol

#         return grad, hess

#     def __str__(self):
#         return f"LGBPrimalDualImproved({self.rho}, {self.adversary_type}, {self.dual_update}, {self.eta_adv})"




# ==========================================================
# Primal–Dual (smooth) version (kept very close to the other ones)
# ==========================================================

class LGBPrimalDual:
    def __init__(self, rho=1e-3, keep=0.7, adversary_type="overall", eta_adv=0.1, zero_grad_tol=1e-6, lgbm_params=None):
        self.rho = rho
        self.keep = keep
        self.adversary_type = adversary_type
        self.eta_adv = eta_adv
        self.zero_grad_tol = zero_grad_tol
        self.model = lgb.LGBMRegressor(**lgbm_params)

    def fit(self, X, y):
        # cache for the callback
        self.X_ = X
        self.y_ = y
        self.y_mean_ = np.mean(y)
        self.n_ = y.size
        # Update: cache current ensemble predictions so we can update them incrementally
        self.y_hat_ = np.ones(self.n_) * self.y_mean_   # matches boost_from_average=True default

        # CVaR cap: w_i <= 1/(alpha*n), with alpha = keep
        self.cap_ = 1.0 / (max(1, int(self.keep * self.n_)) )  # = 1/K

        # initialize adversary weights (uniform)
        w0 = np.ones(self.n_) / self.n_
        if self.adversary_type == "overall":
            self.w_ = w0
        elif self.adversary_type == "individual":
            self.p_ = w0
            self.q_ = w0
        else:
            raise ValueError(f"No adversary_type called: {self.adversary_type}")

        # Update lgbm params
        self.model.set_params(objective=self.fobj)
        self.model.fit(X, y, callbacks=[self._adv_callback])

    def predict(self, X):
        return self.model.predict(X)

    def _project_capped_simplex(self, w):
        """Project to {w>=0, sum w=1, w_i<=cap_} (simple cap + redistribute)."""
        w = np.maximum(w, 0)
        if w.sum() <= 0:
            w = np.ones_like(w) / w.size
        else:
            w = w / w.sum()

        cap = self.cap_
        # cap-and-redistribute until feasible (usually 1-2 passes)
        for _ in range(10):
            over = w > cap
            if not np.any(over):
                break
            excess = w[over].sum() - cap * over.sum()
            w[over] = cap
            under = ~over
            if not np.any(under):
                # everything capped -> already sums to 1 by definition of cap=1/K
                break
            w[under] += excess * (w[under] / w[under].sum())
        return w

    def _mirror_step(self, w, v):
        # exponentiated-gradient / mirror-ascent step
        z = self.eta_adv * (v - np.max(v))
        w_new = w * np.exp(z)
        return self._project_capped_simplex(w_new)

    def _adv_callback(self, env):
        # update adversary once per boosting iteration using current predictions
        it = env.iteration + 1
        # y_hat = env.model.predict(self.X_, num_iteration=it)
        # Update: predict ONLY the new tree’s contribution and add it to cached predictions
        delta = env.model.predict(self.X_, start_iteration=it-1, num_iteration=1)
        self.y_hat_ = self.y_hat_ + delta
        y_hat = self.y_hat_

        mse_value = (self.y_ - y_hat) ** 2
        cov_surr_value = (y_hat / self.y_ - 1) ** 2 * (self.y_ - self.y_mean_) ** 2

        if self.adversary_type == "overall":
            v = mse_value + self.rho * cov_surr_value
            self.w_ = self._mirror_step(self.w_, v)
        else:
            self.p_ = self._mirror_step(self.p_, mse_value)
            self.q_ = self._mirror_step(self.q_, cov_surr_value)

    def fobj(self, y_true, y_pred):
        # Loss function value (same prints as yours)
        mse_value = (y_true - y_pred) ** 2
        cov_surr_value = (y_pred / y_true - 1) ** 2 * (y_true - np.mean(y_true)) ** 2
        loss_value = mse_value + self.rho * cov_surr_value
        model_name = self.__str__()
        print(
            f"[{model_name.split('(')[0]}] "
            f"Loss value: {np.mean(loss_value):.6f} "
            f"| MSE value: {np.mean(mse_value):.6f} "
            f"| CovSurr value: {np.mean(cov_surr_value):.6f} "
            f"| Corr(r,y): {np.corrcoef(y_pred / y_true, y_true)[0, 1]:.6f} "
        )

        # base gradients/hessians for 0.5*(pred-y)^2  (now for ALL samples)
        grad_base = 2 * (y_pred - y_true)
        hess_base = 2 * np.ones_like(y_pred)

        # penalty gradients/hessians (same structure as yours, for ALL samples)
        z = y_true
        z_c = (y_true - np.mean(y_true))
        grad_pen = 2 * (y_pred - z) * (z_c / z) ** 2
        hess_pen = 2 * (z_c / z) ** 2

        n = y_pred.size

        # primal step uses current adversary weights (scaled by n so magnitudes stay reasonable)
        if self.adversary_type == "overall":
            w_eff = n * self.w_
            grad = w_eff * (grad_base + self.rho * grad_pen)
            hess = w_eff * (hess_base + self.rho * hess_pen)
        else:
            p_eff = n * self.p_
            q_eff = n * self.q_
            grad = p_eff * grad_base + self.rho * q_eff * grad_pen
            hess = p_eff * hess_base + self.rho * q_eff * hess_pen

        # zero grad/hess tol (same as yours)
        grad[grad == 0] += self.zero_grad_tol
        hess[hess == 0] += self.zero_grad_tol

        return grad, hess

    def __str__(self):
        return f"LGBPrimalDual({self.rho}, {self.adversary_type}, {self.eta_adv})" #adversary_type={self.adversary_type})" #, eta_adv={self.eta_adv}, tol={self.zero_grad_tol})"


# Experimental binning
import numpy as np
import lightgbm as lgb
from sklearn.base import BaseEstimator, RegressorMixin


class LGBBinnedAdversarialRegressivityPenalty(BaseEstimator, RegressorMixin):
    """
    LightGBM objective:
        MSE + (rho/2) * sum_{b=1..B} w_b * (m_b)^2

    where bins b partition samples by a binning variable (typically the target value y_true,
    or log(y_true) if y_true is price). Within each bin:
        m_b = mean_{i in bin b} t_i

    and t_i is a "regressivity signal" you choose:

    penalty_signal="residual" (Option 2, recommended stable):
        t_i = (y_pred - y_true) - target_residual      (default target_residual = 0)

    penalty_signal="ratio" (Option 1):
        define r_i via ratio_mode, then:
        t_i = r_i - target_ratio                       (default target_ratio = 1)

        ratio_mode="div":
            r_i = y_pred / max(|y_true|, eps_y)
            (NOTE: this is only meaningful if y_true is on the same scale as y_pred and y_true>0)

        ratio_mode="expdiff" (recommended if y_true is log-price):
            r_i = exp(y_pred - y_true)

    Adversary:
        weights w over bins are updated by mirror ascent on v_b = (m_b)^2 (fairness only),
        with capped simplex projection and mild uniform mixing:
            w <- (1-mix) * Proj_CapSimplex(w * exp(eta_adv*(v - max(v)))) + mix * uniform

    Binning:
        binning="quantile" | "uniform" | "quantile_tails" | "custom"
        hard bins by default; optionally soft bins via soft_bins=True, which assigns each sample
        to two neighboring bins with linear interpolation (smooth across bin boundaries).

    Notes:
      * This penalty controls *binwise mean* violation, i.e. piecewise-constant approximation
        to E[t | value_bin]. It can strongly reduce nonlinear regressivity patterns.
      * Use capped simplex + mixing + update_every to improve stability/generalization.
      * For log targets, ratio_mode="expdiff" aligns with original-scale ratios.
    """

    def __init__(
        self,
        rho=1e-3,

        # --- penalty definition ---
        penalty_signal="residual",      # "residual" or "ratio"
        ratio_mode="div",           # "expdiff" or "div" (used if penalty_signal="ratio")
        target_residual=0.0,            # residual anchor
        target_ratio=1.0,               # ratio anchor
        eps_y=1e-12,                    # for div ratio

        # --- binning ---
        n_bins=20,
        binning="quantile",             # "quantile" | "uniform" | "quantile_tails" | "custom"
        bin_on="y",                     # "y" (use y_true) | "logy" (use log(y_true), requires y_true>0)
        custom_edges=None,              # array-like length B+1 if binning="custom"
        tails_frac=0.2,                 # for quantile_tails: fraction in each tail
        tails_bins_each=6,              # for quantile_tails: bins per tail (middle gets remaining)
        soft_bins=False,                # smooth bin assignment (linear interpolation between adjacent bins)

        # --- adversary (bins) ---
        eta_adv=0.2,                    # mirror-ascent step size for adversary
        keep=0.7,                       # CVaR-style cap: w_b <= 1/K with K = ceil(keep * B_eff)
        mix_uniform=0.05,               # mild uniform mixing (0..1)
        update_every=1,                 # update adversary every k boosting rounds

        # --- curvature / numerical ---
        hess_mode="gauss_newton",       # "gauss_newton" | "full_diag"
        zero_grad_tol=1e-8,
        verbose=True,

        # --- LightGBM params ---
        lgbm_params=None,
    ):
        self.rho = float(rho)

        self.penalty_signal = penalty_signal
        self.ratio_mode = ratio_mode
        self.target_residual = float(target_residual)
        self.target_ratio = float(target_ratio)
        self.eps_y = float(eps_y)

        self.n_bins = int(n_bins)
        self.binning = binning
        self.bin_on = bin_on
        self.custom_edges = custom_edges
        self.tails_frac = float(tails_frac)
        self.tails_bins_each = int(tails_bins_each)
        self.soft_bins = bool(soft_bins)

        self.eta_adv = float(eta_adv)
        self.keep = float(keep)
        self.mix_uniform = float(mix_uniform)
        self.update_every = int(update_every)

        self.hess_mode = hess_mode
        self.zero_grad_tol = float(zero_grad_tol)
        self.verbose = bool(verbose)

        self.lgbm_params = dict(lgbm_params or {})
        # ensure boost_from_average=True for stable cached-pred updates
        self.lgbm_params.setdefault("boost_from_average", True)

        self.model = lgb.LGBMRegressor(**self.lgbm_params)

    # ------------------------
    # sklearn API
    # ------------------------
    def fit(self, X, y):
        self.X_ = X
        self.y_ = np.asarray(y).astype(float)
        self.n_ = self.y_.size

        # initialize cached predictions to mean label if boost_from_average=True
        self.y_mean_ = float(np.mean(self.y_))
        self.y_hat_ = np.ones(self.n_) * self.y_mean_

        # prepare bins (on y or logy)
        self._prepare_bins(self.y_)

        # initialize adversary weights over bins (uniform)
        self.w_bins_ = np.ones(self.B_) / self.B_
        self.cap_ = self._cap_from_keep(self.keep, self.B_)

        # attach custom objective and callback
        self.model.set_params(objective=self.fobj)
        self.model.fit(X, y, callbacks=[self._adv_callback])

        return self

    def predict(self, X):
        return self.model.predict(X)

    # ------------------------
    # binning helpers
    # ------------------------
    def _cap_from_keep(self, keep, B):
        K = max(1, int(np.ceil(keep * B)))
        return 1.0 / K

    def _prepare_bins(self, y_true):
        # choose binning variable
        if self.bin_on == "y":
            v = y_true
        elif self.bin_on == "logy":
            if np.any(y_true <= 0):
                raise ValueError("bin_on='logy' requires y_true > 0.")
            v = np.log(y_true)
        else:
            raise ValueError("bin_on must be 'y' or 'logy'.")

        v = np.asarray(v, dtype=float)
        self.bin_values_ = v

        # build edges
        if self.binning == "custom":
            if self.custom_edges is None:
                raise ValueError("custom_edges must be provided when binning='custom'.")
            edges = np.asarray(self.custom_edges, dtype=float)
            if edges.ndim != 1 or edges.size < 3:
                raise ValueError("custom_edges must be a 1D array of length >= 3 (B+1).")
            # allow finite edges; we'll clamp endpoints
            edges = edges.copy()
        elif self.binning == "uniform":
            lo, hi = np.min(v), np.max(v)
            if lo == hi:
                edges = np.array([lo - 1.0, lo + 1.0])
            else:
                edges = np.linspace(lo, hi, self.n_bins + 1)
        elif self.binning == "quantile":
            qs = np.linspace(0.0, 1.0, self.n_bins + 1)
            edges = np.quantile(v, qs)
        elif self.binning == "quantile_tails":
            # More bins in tails, fewer in middle
            frac = self.tails_frac
            k_tail = self.tails_bins_each
            if not (0.0 < frac < 0.5):
                raise ValueError("tails_frac must be in (0, 0.5).")
            if 2 * k_tail >= self.n_bins:
                raise ValueError("tails_bins_each too large for n_bins.")
            k_mid = self.n_bins - 2 * k_tail

            # quantiles for low tail, mid, high tail
            q_low = np.linspace(0.0, frac, k_tail + 1)
            q_mid = np.linspace(frac, 1.0 - frac, k_mid + 1)
            q_high = np.linspace(1.0 - frac, 1.0, k_tail + 1)

            edges = np.unique(np.concatenate([
                np.quantile(v, q_low),
                np.quantile(v, q_mid[1:]),   # avoid duplicate at frac
                np.quantile(v, q_high[1:]),  # avoid duplicate at 1-frac
            ]))
        else:
            raise ValueError("binning must be 'quantile', 'uniform', 'quantile_tails', or 'custom'.")

        # ensure edges strictly increasing as much as possible
        edges = np.asarray(edges, dtype=float)
        edges = np.unique(edges)
        if edges.size < 3:
            # degenerate: all values same
            lo = float(np.min(v))
            edges = np.array([lo - 1.0, lo, lo + 1.0])

        # add infinite endpoints for safe searchsorted-based hard binning
        edges[0] = -np.inf
        edges[-1] = np.inf
        self.bin_edges_ = edges
        self.B_ = edges.size - 1  # number of bins

        # precompute hard bin ids for training points
        # bin id in {0,...,B_-1}
        self.bin_id_ = np.searchsorted(self.bin_edges_, v, side="right") - 1
        self.bin_id_ = np.clip(self.bin_id_, 0, self.B_ - 1)

        # counts for hard bins (used even if soft_bins=True as fallback)
        self.bin_counts_ = np.bincount(self.bin_id_, minlength=self.B_).astype(float)

    # ------------------------
    # adversary helpers
    # ------------------------
    def _project_capped_simplex(self, w):
        """Project to {w>=0, sum w=1, w_i<=cap_} with cap-and-redistribute."""
        w = np.maximum(w, 0.0)
        s = w.sum()
        if s <= 0:
            w = np.ones_like(w) / w.size
        else:
            w = w / s

        cap = float(self.cap_)
        for _ in range(20):
            over = w > cap
            if not np.any(over):
                break
            excess = w[over].sum() - cap * over.sum()
            w[over] = cap
            under = ~over
            if not np.any(under):
                break
            w[under] += excess * (w[under] / w[under].sum())
        # numerical normalization
        w = np.maximum(w, 0.0)
        w = w / w.sum()
        return w

    def _mirror_ascent_step(self, w, v):
        """Exponentiated-gradient / mirror ascent on simplex with capped projection."""
        v = np.asarray(v, dtype=float)
        z = self.eta_adv * (v - np.max(v))  # stabilize
        w_new = w * np.exp(z)
        w_new = self._project_capped_simplex(w_new)
        # mild uniform mixing (stability / generalization)
        if self.mix_uniform > 0:
            u = np.ones_like(w_new) / w_new.size
            w_new = (1.0 - self.mix_uniform) * w_new + self.mix_uniform * u
            w_new = self._project_capped_simplex(w_new)
        return w_new

    def _adv_callback(self, env):
        """Update bin weights once per boosting iteration using current predictions."""
        it = env.iteration + 1  # 1-index
        if self.update_every <= 0:
            return
        if (it % self.update_every) != 0:
            # still update cached preds if we rely on delta
            delta = env.model.predict(self.X_, start_iteration=it-1, num_iteration=1)
            self.y_hat_ = self.y_hat_ + delta
            return

        # Update cached predictions by adding only the newest tree contribution
        delta = env.model.predict(self.X_, start_iteration=it-1, num_iteration=1)
        self.y_hat_ = self.y_hat_ + delta
        y_pred = self.y_hat_

        # compute bin violations m_b under current predictions
        m_b, _ = self._compute_bin_means_and_derivatives(self.y_, y_pred, need_derivatives=False)

        v = m_b ** 2  # adversary focuses on fairness only
        self.w_bins_ = self._mirror_ascent_step(self.w_bins_, v)

        if self.verbose:
            worst = float(np.max(v))
            avg = float(np.mean(v))
            print(f"[ADV bins] it={it:04d} | mean(m_b^2)={avg:.3e} | max(m_b^2)={worst:.3e}")

    # ------------------------
    # penalty signal: t, dt, d2t
    # ------------------------
    def _signal_and_derivatives(self, y_true, y_pred):
        """
        Returns:
          t:   per-sample violation signal
          dt:  dt/dy_pred (same shape)
          d2t: d^2t/dy_pred^2 (same shape)
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        if self.penalty_signal == "residual":
            # t = (pred - true) - target
            t = (y_pred - y_true) - self.target_residual
            dt = np.ones_like(y_pred)
            d2t = np.zeros_like(y_pred)
            return t, dt, d2t

        if self.penalty_signal != "ratio":
            raise ValueError("penalty_signal must be 'residual' or 'ratio'.")

        # ratio signals
        if self.ratio_mode == "div":
            denom = np.maximum(np.abs(y_true), self.eps_y)
            r = y_pred / denom
            t = r - self.target_ratio
            dt = 1.0 / denom
            d2t = np.zeros_like(y_pred)
            return t, dt, d2t

        if self.ratio_mode == "expdiff":
            # r = exp(pred - true)  (true original-scale ratio if targets are log-prices)
            e = y_pred - y_true
            r = np.exp(np.clip(e, -50, 50))  # clip for numerical stability
            t = r - self.target_ratio
            dt = r
            d2t = r
            return t, dt, d2t

        raise ValueError("ratio_mode must be 'div' or 'expdiff'.")

    # ------------------------
    # bin aggregation (hard / soft) and derivatives
    # ------------------------
    def _compute_bin_means_and_derivatives(self, y_true, y_pred, need_derivatives=True):
        """
        Computes:
          m_b: bin mean of t
        If need_derivatives=True, also returns per-sample:
          dm_i: d m_{bin(i)} / d y_pred_i (hard bins)
          (for soft bins, returns dm_coeff_i such that grad_pen_i = rho * dt_i * dm_coeff_i)
        """
        t, dt, d2t = self._signal_and_derivatives(y_true, y_pred)

        if not self.soft_bins:
            # hard bins: each sample belongs to exactly one bin
            bid = self.bin_id_
            B = self.B_
            counts = np.bincount(bid, minlength=B).astype(float)
            sum_t = np.bincount(bid, weights=t, minlength=B).astype(float)

            # avoid division by zero
            denom = np.maximum(counts, 1.0)
            m_b = sum_t / denom

            if not need_derivatives:
                return m_b, None

            # per-sample mean derivative: d m_b / d y_pred_i = dt_i / n_b
            n_b = denom[bid]
            dm_i = dt / n_b
            # also provide d2t for full_diag option
            return m_b, (dm_i, dt, d2t, n_b)

        # soft bins: linear interpolation between adjacent hard bins (smooth across edges)
        # Each sample contributes to its primary bin j and neighbor j+1 by fraction.
        v = self.bin_values_
        edges = self.bin_edges_
        B = self.B_

        # primary bin index j (hard)
        j = self.bin_id_.copy()
        j = np.clip(j, 0, B - 1)

        # compute fractional position within the bin, for bins that have finite width
        left = edges[j]
        right = edges[j + 1]
        width = right - left
        # where width is inf or 0, set frac = 0 (all mass on j)
        frac = np.zeros_like(v, dtype=float)
        finite = np.isfinite(width) & (width > 0)
        frac[finite] = (v[finite] - left[finite]) / width[finite]
        frac = np.clip(frac, 0.0, 1.0)

        # weights to j and j+1 (if exists)
        wj = 1.0 - frac
        jp1 = np.clip(j + 1, 0, B - 1)
        wjp1 = np.where(jp1 != j, frac, 0.0)  # last bin has no +1 neighbor

        # aggregate weighted sums and denominators
        denom = np.zeros(B, dtype=float)
        sum_t = np.zeros(B, dtype=float)

        np.add.at(denom, j, wj)
        np.add.at(sum_t, j, wj * t)

        np.add.at(denom, jp1, wjp1)
        np.add.at(sum_t, jp1, wjp1 * t)

        denom_safe = np.maximum(denom, 1e-12)
        m_b = sum_t / denom_safe

        if not need_derivatives:
            return m_b, None

        # For gradients, we need coefficient per sample:
        # grad_pen_i = rho * dt_i * sum_b w_adv_b * m_b * (w_ib / denom_b)
        w_adv = self.w_bins_
        term_j = w_adv[j] * m_b[j] * (wj / denom_safe[j])
        term_jp1 = w_adv[jp1] * m_b[jp1] * (wjp1 / denom_safe[jp1])
        dm_coeff = term_j + term_jp1
        return m_b, (dm_coeff, dt, d2t, denom_safe, j, jp1, wj, wjp1)

    # ------------------------
    # LightGBM custom objective
    # ------------------------
    def fobj(self, y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        n = y_pred.size
        n_f = float(n)

        # base MSE
        grad_base = 2.0 * (y_pred - y_true)
        hess_base = 2.0 * np.ones_like(y_pred)

        # penalty pieces
        m_b, aux = self._compute_bin_means_and_derivatives(y_true, y_pred, need_derivatives=True)
        w_adv = self.w_bins_

        # compute penalty value (for logging)
        pen_value = 0.5 * self.rho * n_f * float(np.sum(w_adv * (m_b ** 2)))

        # gradients/hessians for penalty
        if not self.soft_bins:
            dm_i, dt, d2t, n_b = aux  # dt = dt/dy_pred, d2t = second derivative
            bid = self.bin_id_

            m_i = m_b[bid]
            w_i = w_adv[bid]

            # grad: rho * w_b * m_b * (dt / n_b)
            grad_pen = self.rho * n_f * w_i * m_i * dm_i

            if self.hess_mode == "full_diag":
                # diag Hess: rho*w_b*(dm_i^2 + m_b*(d2t/n_b))
                hess_pen = self.rho * n_f * w_i * (dm_i ** 2 + m_i * (d2t / n_b))
            else:
                # Gauss-Newton style: rho*w_b*dm_i^2
                hess_pen = self.rho * n_f * w_i * (dm_i ** 2)

        else:
            dm_coeff, dt, d2t, denom_safe, j, jp1, wj, wjp1 = aux
            # grad: rho * dt_i * dm_coeff_i
            grad_pen = self.rho * n_f * dt * dm_coeff

            if self.hess_mode == "full_diag":
                # full diag is messy for soft case with expdiff; use GN for stability
                pass

            # Gauss-Newton style diag Hess:
            # rho * dt_i^2 * sum_b w_b*(w_ib/denom_b)^2
            # where only j and j+1 bins contribute
            term_j = w_adv[j] * (wj / denom_safe[j]) ** 2
            term_jp1 = w_adv[jp1] * (wjp1 / denom_safe[jp1]) ** 2
            hess_pen = self.rho * n_f * (dt ** 2) * (term_j + term_jp1)

        grad = grad_base + grad_pen
        hess = hess_base + hess_pen

        # numerical floors
        grad[np.abs(grad) < self.zero_grad_tol] = np.sign(grad[np.abs(grad) < self.zero_grad_tol]) * self.zero_grad_tol
        hess[hess < self.zero_grad_tol] = self.zero_grad_tol

        # optional prints
        if self.verbose:
            mse_mean = float(np.mean((y_true - y_pred) ** 2))
            # simple diagnostic: worst bin violation magnitude
            worst_bin = float(np.max(np.abs(m_b))) if m_b.size else float("nan")
            print(
                f"[{self.__class__.__name__}] "
                f"MSE={mse_mean:.6f} | Pen={pen_value:.6f} | "
                f"mean(m_b^2)={float(np.mean(m_b**2)):.3e} | max|m_b|={worst_bin:.3e}"
            )

        return grad, hess

    def __str__(self):
        return f"LGBBinnedAdversarialRegressivityPenalty(rho={self.rho}, penalty_signal={self.penalty_signal}, ratio_mode={self.ratio_mode}, target_residual={self.target_residual}, target_ratio={self.target_ratio}, eps_y={self.eps_y}, n_bins={self.n_bins}, binning={self.binning}, custom_edges={self.custom_edges}, tails_frac={self.tails_frac}, tails_bins_each={self.tails_bins_each}, soft_bins={self.soft_bins}, eta_adv={self.eta_adv}, keep={self.keep}, mix_uniform={self.mix_uniform}, update_every={self.update_every}, hess_mode={self.hess_mode}, zero_grad_tol={self.zero_grad_tol})"







### 

# Models to compare primal-dual method

###

# 0) Just MSE but binning by y-real
class LGBBinnedMSEWeights:
    """
    LightGBM custom objective: weighted MSE only.

    Purpose:
      A super simple baseline to test how *re-weighting by y-bins* affects LightGBM,
      without changing the loss beyond weights.

    We build bins on y_true (ground truth) using either:
      - binning="quantile": equal-count bins (by quantiles)
      - binning="uniform" : equal-width bins in y-scale

    Then assign each sample a weight w_i based on its bin.
    By default, we use inverse-frequency weights so each bin contributes equally.

    Objective (per sample):
      loss_i = w_i * (y_pred_i - y_true_i)^2

    Grad/Hess:
      grad_i = 2 * w_i * (y_pred_i - y_true_i)
      hess_i = 2 * w_i

    Notes:
      - This is equivalent to passing sample_weight into LightGBM training,
        but implemented as a custom objective to keep the same "structure" style.
      - We clamp weights to be >= weight_floor for numerical stability.
    """

    def __init__(
        self,
        n_bins=10,
        binning="quantile",            # "quantile" or "uniform"
        weight_mode="inv_freq",        # "inv_freq" or "freq" or "none"
        weight_floor=1e-8,
        zero_grad_tol=1e-6,
        lgbm_params=None,
        verbose=True,
    ):
        self.n_bins = int(n_bins)
        self.binning = str(binning)
        self.weight_mode = str(weight_mode)
        self.weight_floor = float(weight_floor)
        self.zero_grad_tol = float(zero_grad_tol)
        self.verbose = bool(verbose)

        self.model = lgb.LGBMRegressor(**(lgbm_params or {}))

    def fit(self, X, y):
        y = np.asarray(y)

        if self.n_bins < 1:
            raise ValueError("n_bins must be >= 1")
        if self.binning not in ("quantile", "uniform"):
            raise ValueError("binning must be 'quantile' or 'uniform'")
        if self.weight_mode not in ("inv_freq", "freq", "none"):
            raise ValueError("weight_mode must be 'inv_freq', 'freq', or 'none'")

        self.bin_edges_ = self._make_bin_edges(y, self.n_bins, self.binning)
        self.bin_idx_, self.n_bins_eff_ = self._bin_index(y, self.bin_edges_)
        self.weights_, self.bin_counts_ = self._compute_weights(self.bin_idx_, self.n_bins_eff_)

        if self.verbose:
            counts = self.bin_counts_.astype(int).tolist()
            w_stats = (float(np.min(self.weights_)), float(np.mean(self.weights_)), float(np.max(self.weights_)))
            print(f"[LGBBinnedMSEWeights] bins={self.n_bins_eff_} | binning={self.binning} | "
                  f"weight_mode={self.weight_mode} | counts={counts} | "
                  f"w(min/mean/max)={w_stats[0]:.3e}/{w_stats[1]:.3e}/{w_stats[2]:.3e}")

        self.model.set_params(objective=self.fobj)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def fobj(self, y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        w = self.weights_

        # weighted MSE grads/hess
        grad = 2.0 * w * (y_pred - y_true)
        hess = 2.0 * w * np.ones_like(y_pred)

        # numerical guards (same style as your other code)
        grad[np.abs(grad) < self.zero_grad_tol] = self.zero_grad_tol
        hess[hess < self.zero_grad_tol] = self.zero_grad_tol

        if self.verbose:
            mse = float(np.mean((y_pred - y_true) ** 2))
            wmse = float(np.mean(w * (y_pred - y_true) ** 2))
            print(f"[LGBBinnedMSEWeights] MSE: {mse:.6f} | weighted MSE: {wmse:.6f}")

        return grad, hess

    # ----------------------------
    # helpers
    # ----------------------------
    @staticmethod
    def _make_bin_edges(y, n_bins, binning):
        y = np.asarray(y)

        if n_bins == 1:
            # single bin covers all
            lo = float(np.min(y))
            hi = float(np.max(y))
            if hi <= lo:
                hi = lo + 1e-12
            return np.array([lo, hi], dtype=float)

        if binning == "quantile":
            qs = np.linspace(0.0, 1.0, n_bins + 1)
            edges = np.quantile(y, qs)
            edges = np.unique(edges)
        else:  # "uniform"
            lo = float(np.min(y))
            hi = float(np.max(y))
            if hi <= lo:
                hi = lo + 1e-12
            edges = np.linspace(lo, hi, n_bins + 1)

        # ensure strictly increasing edges
        edges = np.unique(np.asarray(edges, dtype=float))
        if edges.size < 2:
            lo = float(np.min(y))
            edges = np.array([lo, lo + 1e-12], dtype=float)

        return edges

    @staticmethod
    def _bin_index(y, edges):
        edges = np.asarray(edges, dtype=float)
        K = int(max(edges.size - 1, 1))

        if K == 1:
            return np.zeros_like(y, dtype=int), 1

        # digitize on internal edges
        idx = np.digitize(y, edges[1:-1], right=False)
        idx = np.clip(idx, 0, K - 1).astype(int)

        return idx, K

    def _compute_weights(self, bin_idx, K):
        n = bin_idx.size
        counts = np.bincount(bin_idx, minlength=K).astype(float)
        counts_safe = np.maximum(counts, 1.0)

        if self.weight_mode == "none":
            w_b = np.ones(K, dtype=float)
        elif self.weight_mode == "freq":
            # proportional to bin frequency (mostly pointless, but included)
            w_b = counts_safe / float(n)
        else:
            # inv_freq: equalize bin contributions
            # Each bin gets total weight ≈ 1/K:
            # w_i = (n / (K * n_b))
            w_b = float(n) / (float(K) * counts_safe)

        w = w_b[bin_idx]
        w = np.maximum(w, self.weight_floor)

        return w, counts
    
    def __str__(self):
        return f"LGBBinnedMSEWeights(n_bins={self.n_bins}, binning={self.binning})"

# ==========================================================
# 4) Direct K-moments penalty (non-separable but usable in LightGBM via global stats ???)
# ==========================================================

# 4.5) Cov but Var(E(R|Y))
class LGBVarCondMeanPenalty:
    """LightGBM objective: MSE + rho * Var( E[r | y] )

    We approximate E[r|y] by *quantile bins* on y (computed once in fit()).

    r is chosen by ratio_mode:
      - "div"  : r = y_pred / max(|y_true|, eps_y)
      - "diff" : r = y_pred - y_true
      - "ratio": r = y_pred / max(|y_true|, eps_y)   (alias for "div"; kept for symmetry)

    Vertical penalty (binned):
      - Let bins B_k partition y. Let m_k = mean_{i in B_k} r_i and m_bar = mean_i r_i.
      - Var(E[r|y]) ≈ sum_k (n_k/n) * (m_k - m_bar)^2
      - penalty = 0.5 * rho * n * Var(E[r|y])

    Grad/Hess (diagonal approximation):
      Exact (in r-space):
        d penalty / d r_i = rho * (m_{bin(i)} - m_bar)
        d^2 penalty / d r_i^2 = rho * (1/n_{bin(i)} - 1/n)   (>=0)

      Chain rule with r = r(y_pred):
        grad_pen_i = rho * (m_bin - m_bar) * dr_i
        hess_pen_i = rho * (1/n_bin - 1/n) * (dr_i)^2
    """

    def __init__(
        self,
        rho=1e-3,
        n_bins=10,                # NEW: number of quantile bins on y_true
        ratio_mode="div",         # "div" | "diff" | "ratio"
        anchor_mode="none",       # "none" | "target" | "iter_mean" (treated as constant; see note below)
        target_value=None,
        zero_grad_tol=1e-6,
        eps_y=1e-12,
        lgbm_params=None,
        verbose=True,
    ):
        self.rho = float(rho)
        self.n_bins = int(n_bins)
        if self.n_bins < 1:
            raise ValueError("n_bins must be >= 1.")
        self.ratio_mode = ratio_mode
        self.anchor_mode = anchor_mode
        self.target_value = target_value
        self.zero_grad_tol = float(zero_grad_tol)
        self.eps_y = float(eps_y)
        self.verbose = bool(verbose)
        self.model = lgb.LGBMRegressor(**(lgbm_params or {}))

    def fit(self, X, y):
        y = np.asarray(y)
        self.y_mean_ = float(np.mean(y))

        # ---- quantile bin edges on y (log-price world) ----
        qs = np.linspace(0.0, 1.0, self.n_bins + 1)
        edges = np.quantile(y, qs)

        # Guard against duplicate edges (e.g., many identical y values)
        edges = np.unique(edges)
        if edges.size < 2:
            # Degenerate: everything in one bin
            edges = np.array([float(np.min(y)), float(np.max(y))], dtype=float)

        self.bin_edges_ = edges
        self.K_ = int(self.bin_edges_.size - 1)

        self.model.set_params(objective=self.fobj)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def _bin_index(self, y_true):
        # bins defined by edges; cutpoints are interior edges
        if self.K_ <= 1:
            return np.zeros_like(y_true, dtype=int), 1
        cutpoints = self.bin_edges_[1:-1]  # length K-1
        bin_idx = np.searchsorted(cutpoints, y_true, side="right").astype(int)  # 0..K-1
        return bin_idx, self.K_

    def fobj(self, y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        n = y_pred.size
        n_f = float(n)

        # ---- choose r and dr/dy_pred ----
        if self.ratio_mode in ("div", "ratio"):
            denom = np.maximum(np.abs(y_true), self.eps_y)
            r = y_pred / denom
            dr = 1.0 / denom
        elif self.ratio_mode == "diff":
            r = y_pred - y_true
            dr = np.ones_like(y_pred)
        else:
            raise ValueError("ratio_mode must be 'div', 'ratio', or 'diff'.")

        # ---- optional anchor ----
        # NOTE: for this penalty, an anchor *does* change r, but we treat the anchor as a constant
        # (no gradient through anchor), consistent with the diagonal approximations used elsewhere.
        anchor = 0.0
        if self.anchor_mode == "none":
            anchor = 0.0
        elif self.anchor_mode == "iter_mean":
            anchor = float(np.mean(r))
        elif self.anchor_mode == "target":
            if self.target_value is None:
                anchor = 1.0 if self.ratio_mode in ("div", "ratio") else 0.0
            else:
                anchor = float(self.target_value)
        else:
            raise ValueError("anchor_mode must be 'none', 'iter_mean', or 'target'.")

        r_eff = r - anchor

        # ---- quantile bins on y_true ----
        bin_idx, K = self._bin_index(y_true)

        n_k = np.bincount(bin_idx, minlength=K).astype(float)
        n_k_safe = np.maximum(n_k, 1.0)

        sum_r_k = np.bincount(bin_idx, weights=r_eff, minlength=K).astype(float)
        m_k = sum_r_k / n_k_safe

        m_bar = float(np.mean(r_eff))
        m_bin = m_k[bin_idx]

        # ---- Var(E[r|y]) (binned) ----
        # V = sum_k (n_k/n) (m_k - m_bar)^2
        # (empty bins contribute 0 because n_k=0)
        diff_k = (m_k - m_bar)
        V = float(np.sum((n_k / n_f) * (diff_k ** 2)))

        pen_value = 0.5 * self.rho * n_f * V

        # ---- prints ----
        mse_vec = (y_true - y_pred) ** 2
        mse_mean = float(np.mean(mse_vec))

        if self.verbose:
            model_name = self.__str__().split("(")[0]
            print(
                f"[{model_name}] "
                f"Loss: {(mse_mean + pen_value):.6f} | MSE: {mse_mean:.6f} | "
                f"Var(E[r|y]): {V:.6e} | Pen: {pen_value:.6f} | K: {K}"
            )

        # ---- base MSE grads/hess ----
        grad_base = 2.0 * (y_pred - y_true)
        hess_base = 2.0 * np.ones_like(y_pred)

        # ---- penalty grads/hess (diag Hessian in r-space, chain through r) ----
        # d penalty / d r_i = rho * (m_bin(i) - m_bar)
        grad_pen = self.rho * (m_bin - m_bar) * dr

        # d^2 penalty / d r_i^2 = rho * (1/n_bin - 1/n)
        inv_n_bin = 1.0 / n_k_safe[bin_idx]
        hess_pen = self.rho * (inv_n_bin - (1.0 / n_f)) * (dr ** 2)

        grad = grad_base + grad_pen
        hess = hess_base + hess_pen

        grad[np.abs(grad) < self.zero_grad_tol] = self.zero_grad_tol
        hess[hess < self.zero_grad_tol] = self.zero_grad_tol

        return grad, hess

    def __str__(self):
        return (
            f"LGBVarCondMeanPenalty(rho={self.rho}, n_bins={self.n_bins}, "
            f"ratio_mode={self.ratio_mode}, anchor_mode={self.anchor_mode})"
        )

# Both Var(E(R|Y)) and E(Var(R|Y))
class LGBCondMeanVarPenalty:
    """LightGBM objective:
        MSE
      + rho_cov  * Var( E[r | y] )
      + rho_disp * E[ Var(r | y) ]

    We approximate conditioning on y by *quantile bins* on y (computed once in fit()).

    r is chosen by ratio_mode:
      - "div"  : r = y_pred / max(|y_true|, eps_y)
      - "diff" : r = y_pred - y_true
      - "ratio": alias for "div"

    Definitions (binned):
      - Let bins B_k partition y. Let m_k = mean_{i in B_k} r_i and m_bar = mean_i r_i.
      - Vertical term:
            Var(E[r|y]) ≈ sum_k (n_k/n) * (m_k - m_bar)^2
      - Horizontal term:
            E[Var(r|y)] ≈ sum_k (n_k/n) * ( (1/n_k) * sum_{i in B_k} (r_i - m_k)^2 )
                         = (1/n) * sum_i (r_i - m_{bin(i)})^2

    Scaling (same style as your covariance penalty):
      pen_cov  = 0.5 * rho_cov  * n * Var(E[r|y])
      pen_disp = 0.5 * rho_disp * n * E[Var(r|y)]

    Grad/Hess (diagonal approximation, chain through r = r(y_pred)):
      Let k(i)=bin(i), n_k = count in bin, m_k = bin mean, m_bar = global mean.

      Vertical (between-bin) penalty:
        d/d r_i  [0.5*rho_cov*n*Var(E[r|y])] = rho_cov * (m_{k(i)} - m_bar)
        d^2/d r_i^2                          = rho_cov * (1/n_{k(i)} - 1/n)   (>=0)

      Horizontal (within-bin) penalty:
        d/d r_i  [0.5*rho_disp*n*EVar] = rho_disp * (r_i - m_{k(i)})
        d^2/d r_i^2                    = rho_disp * (1 - 1/n_{k(i)})          (>=0)

      Chain rule:
        grad_pen_i = (grad_r_i) * dr_i
        hess_pen_i = (hess_r_i) * (dr_i)^2
    """

    def __init__(
        self,
        rho_cov=1e-3,             # NEW: vertical penalty weight
        rho_disp=0.0,             # NEW: horizontal penalty weight
        n_bins=10,
        ratio_mode="div",         # "div" | "diff" | "ratio"
        anchor_mode="none",       # "none" | "target" | "iter_mean" (treated as constant; see note below)
        target_value=None,
        zero_grad_tol=1e-6,
        eps_y=1e-12,
        lgbm_params=None,
        verbose=True,
    ):
        self.rho_cov = float(rho_cov)
        self.rho_disp = float(rho_disp)
        self.n_bins = int(n_bins)
        if self.n_bins < 1:
            raise ValueError("n_bins must be >= 1.")
        self.ratio_mode = ratio_mode
        self.anchor_mode = anchor_mode
        self.target_value = target_value
        self.zero_grad_tol = float(zero_grad_tol)
        self.eps_y = float(eps_y)
        self.verbose = bool(verbose)
        self.model = lgb.LGBMRegressor(**(lgbm_params or {}))

    def fit(self, X, y):
        y = np.asarray(y)
        self.y_mean_ = float(np.mean(y))

        # ---- quantile bin edges on y ----
        qs = np.linspace(0.0, 1.0, self.n_bins + 1)
        edges = np.quantile(y, qs)

        # Guard against duplicate edges (e.g., many identical y values)
        edges = np.unique(edges)
        if edges.size < 2:
            edges = np.array([float(np.min(y)), float(np.max(y))], dtype=float)

        self.bin_edges_ = edges
        self.K_ = int(self.bin_edges_.size - 1)

        self.model.set_params(objective=self.fobj)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def _bin_index(self, y_true):
        if self.K_ <= 1:
            return np.zeros_like(y_true, dtype=int), 1
        cutpoints = self.bin_edges_[1:-1]  # length K-1
        bin_idx = np.searchsorted(cutpoints, y_true, side="right").astype(int)  # 0..K-1
        return bin_idx, self.K_

    def fobj(self, y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        n = y_pred.size
        n_f = float(n)

        # ---- choose r and dr/dy_pred ----
        if self.ratio_mode in ("div", "ratio"):
            denom = np.maximum(np.abs(y_true), self.eps_y)
            r = y_pred / denom
            dr = 1.0 / denom
        elif self.ratio_mode == "diff":
            r = y_pred - y_true
            dr = np.ones_like(y_pred)
        else:
            raise ValueError("ratio_mode must be 'div', 'ratio', or 'diff'.")

        # ---- optional anchor (treated as constant; no gradient through anchor) ----
        anchor = 0.0
        if self.anchor_mode == "none":
            anchor = 0.0
        elif self.anchor_mode == "iter_mean":
            anchor = float(np.mean(r))
        elif self.anchor_mode == "target":
            if self.target_value is None:
                anchor = 1.0 if self.ratio_mode in ("div", "ratio") else 0.0
            else:
                anchor = float(self.target_value)
        else:
            raise ValueError("anchor_mode must be 'none', 'iter_mean', or 'target'.")

        r_eff = r - anchor

        # ---- quantile bins on y_true ----
        bin_idx, K = self._bin_index(y_true)

        n_k = np.bincount(bin_idx, minlength=K).astype(float)
        n_k_safe = np.maximum(n_k, 1.0)

        sum_r_k = np.bincount(bin_idx, weights=r_eff, minlength=K).astype(float)
        m_k = sum_r_k / n_k_safe

        m_bar = float(np.mean(r_eff))
        m_bin = m_k[bin_idx]

        # ---- vertical: Var(E[r|y]) ----
        diff_k = (m_k - m_bar)
        V_cov = float(np.sum((n_k / n_f) * (diff_k ** 2)))

        # ---- horizontal: E[Var(r|y)] ----
        # EVar = (1/n) * sum_i (r_i - m_{bin(i)})^2
        resid_within = (r_eff - m_bin)
        V_disp = float(np.mean(resid_within ** 2))

        pen_cov = 0.5 * self.rho_cov * n_f * V_cov
        pen_disp = 0.5 * self.rho_disp * n_f * V_disp
        pen_value = pen_cov + pen_disp

        # ---- prints ----
        mse_vec = (y_true - y_pred) ** 2
        mse_mean = float(np.mean(mse_vec))

        if self.verbose:
            model_name = self.__str__().split("(")[0]
            print(
                f"[{model_name}] "
                f"Loss: {(mse_mean + pen_value):.6f} | MSE: {mse_mean:.6f} | "
                f"Var(E[r|y]): {V_cov:.6e} | EVar(r|y): {V_disp:.6e} | "
                f"Pen(cov): {pen_cov:.6f} | Pen(disp): {pen_disp:.6f} | K: {K}"
            )

        # ---- base MSE grads/hess ----
        grad_base = 2.0 * (y_pred - y_true)
        hess_base = 2.0 * np.ones_like(y_pred)

        # ---- penalty grads/hess in r-space ----
        # Vertical:
        grad_r_cov = self.rho_cov * (m_bin - m_bar)
        inv_n_bin = 1.0 / n_k_safe[bin_idx]
        hess_r_cov = self.rho_cov * (inv_n_bin - (1.0 / n_f))

        # Horizontal:
        grad_r_disp = self.rho_disp * (r_eff - m_bin)
        hess_r_disp = self.rho_disp * (1.0 - inv_n_bin)

        grad_pen = (grad_r_cov + grad_r_disp) * dr
        hess_pen = (hess_r_cov + hess_r_disp) * (dr ** 2)

        grad = grad_base + grad_pen
        hess = hess_base + hess_pen

        grad[np.abs(grad) < self.zero_grad_tol] = self.zero_grad_tol
        hess[hess < self.zero_grad_tol] = self.zero_grad_tol

        return grad, hess

    def __str__(self):
        return (
            f"LGBCondMeanVarPenalty(rho_cov={self.rho_cov}, rho_disp={self.rho_disp}, "
            f"n_bins={self.n_bins}, ratio_mode={self.ratio_mode}, anchor_mode={self.anchor_mode})"
        )


# 5) Covariance AND variance of r. It shouldn't make much sense, because min MSE is already seeking a low dispersion in this case
# V4: With binning and div/diff
class LGBCovDispPenalty:
    """LightGBM objective:
        (optionally binned) MSE
        + rho_cov  * Cov(r, y)^2   (or only negative direction)
        + rho_disp * mean_dispersion_loss(u)

    r is chosen by ratio_mode:
      - "div"  : r = y_pred / max(|y_true|, eps_y)   (old behavior)
      - "diff" : r = y_pred - y_true                (log-residual if y is log-price)

    Dispersion variable u:
      - if ratio_mode="div"  : u = r - 1
      - if ratio_mode="diff" : u = r - 0 = r

    Cov term workflow (cov_mode):
      - "cov"     : penalize Cov(r,y)^2
      - "neg_cov" : penalize max(0, -Cov(r,y))^2   (only regressivity direction)

    Scaling (kept similar to your other models):
      cov_pen  = 0.5 * rho_cov  * n * cov_term^2
      disp_pen =       rho_disp * n * mean(ell(u))   (ell(u) ~ 0.5 u^2 near 0)

    ---- Binning (minimal, optional) ----
    Instead of CVaR-style weighting, you can "equalize representation" by y-bins.

    If enabled for a term, we aggregate that term as an *equal average across bins*:

      - For MSE / Disp:
          term = (1/K) * sum_b mean_{i in b}(...)    (K = # nonempty bins)
        This is equivalent to per-sample weights w_i = n / (K * n_b(i)) for that term.

      - For Cov:
          cov_b = mean_{i in b}( r_i * (y_i - y_mean) )
          cov_pen = 0.5 * rho_cov * n * (1/K) * sum_b cov_term_b^2
        (cov_term_b = cov_b or max(0,-cov_b) depending on cov_mode)

    Controls:
      n_bins: int
      binning: "quantile" or "uniform"
      bin_mse / bin_cov / bin_disp: bool flags for which terms use bin aggregation
    """

    def __init__(
        self,
        rho_cov=1e-3,
        rho_disp=1e-3,
        ratio_mode="div",          # "div" or "diff"
        cov_mode="cov",            # "cov" or "neg_cov"
        disp_mode="l2",            # "l2" or "pseudohuber"
        huber_delta=0.10,          # only used if disp_mode="pseudohuber"
        # binning controls
        n_bins=1,                  # <=1 disables binning
        binning="quantile",        # "quantile" or "uniform"
        weight_mode="inv_freq",        # "inv_freq" or "freq" or "none"
        bin_mse=False,
        bin_cov=False,
        bin_disp=False,
        # numerics
        zero_grad_tol=1e-6,
        eps_y=1e-12,
        eps_delta=1e-12,
        lgbm_params=None,
        verbose=True,
    ):
        self.rho_cov = float(rho_cov)
        self.rho_disp = float(rho_disp)

        self.ratio_mode = str(ratio_mode)
        self.cov_mode = str(cov_mode)
        self.disp_mode = str(disp_mode)
        self.huber_delta = float(huber_delta)

        self.n_bins = int(n_bins)
        self.binning = str(binning)
        self.weight_mode = str(weight_mode)
        self.bin_mse = bool(bin_mse)
        self.bin_cov = bool(bin_cov)
        self.bin_disp = bool(bin_disp)

        self.zero_grad_tol = float(zero_grad_tol)
        self.eps_y = float(eps_y)
        self.eps_delta = float(eps_delta)
        self.verbose = bool(verbose)

        self.model = lgb.LGBMRegressor(**(lgbm_params or {}))

    def fit(self, X, y):
        y = np.asarray(y)
        self.y_mean_ = float(np.mean(y))

        if self.ratio_mode not in ("div", "diff"):
            raise ValueError(f"ratio_mode must be 'div' or 'diff', got {self.ratio_mode!r}")
        if self.cov_mode not in ("cov", "neg_cov"):
            raise ValueError(f"cov_mode must be 'cov' or 'neg_cov', got {self.cov_mode!r}")
        if self.disp_mode not in ("l2", "pseudohuber"):
            raise ValueError(f"disp_mode must be 'l2' or 'pseudohuber', got {self.disp_mode!r}")
        if self.disp_mode == "pseudohuber" and self.huber_delta <= 0:
            raise ValueError(f"huber_delta must be > 0, got {self.huber_delta}")
        if self.n_bins < 1:
            raise ValueError("n_bins must be >= 1")
        if self.binning not in ("quantile", "uniform"):
            raise ValueError("binning must be 'quantile' or 'uniform'")

        # store bin edges (used by fobj). If n_bins<=1, no binning.
        self.bin_edges_ = None
        if self.n_bins > 1 and (self.bin_mse or self.bin_cov or self.bin_disp):
            self.bin_edges_ = self._make_bin_edges(y, self.n_bins, self.binning)

        self.model.set_params(objective=self.fobj)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    # ----------------------------
    # dispersion loss
    # ----------------------------
    def _disp_loss_and_derivs(self, u):
        """Return (ell(u), ell'(u), ell''(u)) elementwise."""
        if self.disp_mode == "l2":
            # ell(u)=0.5u^2 -> ell'(u)=u, ell''(u)=1
            ell = 0.5 * (u ** 2)
            ell_p = u
            ell_pp = np.ones_like(u)
            return ell, ell_p, ell_pp

        # pseudo-Huber:
        # ell(u) = d^2 (sqrt(1+(u/d)^2)-1)
        # ell'(u)= u / sqrt(1+(u/d)^2)
        # ell''(u)= 1 / (1+(u/d)^2)^(3/2)
        d = max(self.huber_delta, self.eps_delta)
        t = u / d
        s = np.sqrt(1.0 + t * t)
        ell = (d * d) * (s - 1.0)
        ell_p = u / s
        ell_pp = 1.0 / (s ** 3)
        return ell, ell_p, ell_pp

    # ----------------------------
    # objective
    # ----------------------------
    def fobj(self, y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        n = y_pred.size

        yc = (y_true - self.y_mean_)  # centered y

        # ---- r and dr/dpred ----
        if self.ratio_mode == "div":
            denom = np.maximum(np.abs(y_true), self.eps_y)
            dr = 1.0 / denom
            r = y_pred * dr
            u = r - 1.0
        else:  # "diff"
            dr = np.ones_like(y_pred)
            r = y_pred - y_true
            u = r  # = r - 0

        # ---- binning setup (optional) ----
        use_bins = (self.bin_edges_ is not None)
        if use_bins:
            bin_idx, K = self._bin_index(y_true, self.bin_edges_)
            n_b = np.bincount(bin_idx, minlength=K).astype(float)
            nonempty = n_b > 0
            K_eff = int(np.sum(nonempty))
            if K_eff <= 0:
                K_eff = 1
            n_b_safe = np.maximum(n_b, 1.0)

            # # per-sample factor: n/(K_eff*n_b(i))  (so each nonempty bin contributes equally)
            # w_bin = float(n) / (float(K_eff) * n_b_safe[bin_idx])
            # if self.weight_mode == "freq":
            #     w_bin = 1/w_bin
            if self.weight_mode == "inv_freq":
                w_raw = 1.0 / n_b_safe[bin_idx]
            elif self.weight_mode == "freq":
                w_raw = n_b_safe[bin_idx]
            elif self.weight_mode == "none":
                w_raw = np.ones_like(y_pred, dtype=float)
            else:
                raise ValueError("weight_mode must be 'inv_freq', 'freq', or 'none'")

            w_bin = w_raw / float(np.mean(w_raw))  # normalize so mean weight = 1

        else:
            bin_idx = None
            K_eff = 1
            n_b_safe = None
            w_bin = np.ones_like(y_pred, dtype=float)

        # ----------------
        # MSE term (optionally binned)
        # ----------------
        if use_bins and self.bin_mse:
            w_mse = w_bin
        else:
            w_mse = 1.0

        grad_base = 2.0 * w_mse * (y_pred - y_true)
        hess_base = 2.0 * w_mse * np.ones_like(y_pred)

        # ----------------
        # Cov term (optionally binned)
        # cov_b = mean_{i in b}(r_i * yc_i)
        # ----------------
        s = r * yc  # contribution to covariance

        if use_bins and self.bin_cov:
            sum_s_b = np.bincount(bin_idx, weights=s, minlength=K).astype(float)
            cov_b = sum_s_b / n_b_safe
            cov_i = cov_b[bin_idx]

            if self.cov_mode == "cov":
                active = np.ones_like(y_pred, dtype=bool)
            else:  # "neg_cov"
                active = (cov_i < 0.0)

            # d cov_b / d pred_i = (1/n_b) * yc_i * dr_i
            dcb = (yc * dr) / n_b_safe[bin_idx]

            # penalty is 0.5*rho_cov*n*(1/K)*sum_b cov_term_b^2
            # => grad_i = rho_cov*n*(1/K)*cov_b * d cov_b / d pred_i
            factor = float(n) / float(K_eff)
            grad_cov = np.where(active, self.rho_cov * factor * cov_i * dcb, 0.0)
            hess_cov = np.where(active, self.rho_cov * factor * (dcb ** 2), 0.0)

            # for prints
            cov_term_print = float(np.mean((cov_b[nonempty] ** 2))) ** 0.5 if np.any(nonempty) else float("nan")
            pen_cov_value = 0.5 * self.rho_cov * float(n) * float(np.mean((cov_b[nonempty] ** 2))) if np.any(nonempty) else 0.0
        else:
            cov = float(np.mean(s))  # global cov since E[yc]=0
            if self.cov_mode == "cov":
                cov_active = True
            else:
                cov_active = (cov < 0.0)

            # d cov / d pred_i = (1/n) * yc_i * dr_i
            dc = (yc * dr) / float(n)

            if cov_active:
                grad_cov = self.rho_cov * float(n) * cov * dc
                hess_cov = self.rho_cov * float(n) * (dc ** 2)
            else:
                grad_cov = np.zeros_like(y_pred)
                hess_cov = np.zeros_like(y_pred)

            cov_term_print = cov
            pen_cov_value = 0.5 * self.rho_cov * float(n) * (cov ** 2) if cov_active else 0.0

        # ----------------
        # Dispersion term (optionally binned)
        # pen_disp = rho_disp * n * mean(ell(u))   OR binned mean across bins
        # ----------------
        ell_u, ell_p_u, ell_pp_u = self._disp_loss_and_derivs(u)

        if use_bins and self.bin_disp:
            w_disp = w_bin  # same equal-bin factor
        else:
            w_disp = 1.0

        # pen_disp = rho_disp * sum (w_disp * ell(u))  (since w_disp sums to n if binned)
        # u depends on r, and dr/dpred = dr
        grad_disp = self.rho_disp * w_disp * ell_p_u * dr
        hess_disp = self.rho_disp * w_disp * ell_pp_u * (dr ** 2)

        # ----------------
        # combine
        # ----------------
        grad = grad_base + grad_cov + grad_disp
        hess = hess_base + hess_cov + hess_disp

        # numerical guards
        grad[np.abs(grad) < self.zero_grad_tol] = self.zero_grad_tol
        hess[hess < self.zero_grad_tol] = self.zero_grad_tol

        # ----------------
        # logging (simple)
        # ----------------
        if self.verbose:
            mse_mean = float(np.mean((y_true - y_pred) ** 2))
            # helpful: (r - target)^2 mean for div, r^2 mean for diff
            u2_mean = float(np.mean(u ** 2))
            if use_bins and self.bin_disp:
                disp_mean = float(np.mean(w_disp * ell_u)) / float(np.mean(w_disp))
            else:
                disp_mean = float(np.mean(ell_u))
            pen_disp_value = self.rho_disp * float(n) * disp_mean

            try:
                corr = float(np.corrcoef(r, y_true)[0, 1])
            except Exception:
                corr = float("nan")

            model_name = self.__str__().split("(")[0]
            bin_info = f" | bins={K_eff}({self.binning}) mse/cov/disp={int(self.bin_mse)}/{int(self.bin_cov)}/{int(self.bin_disp)}" if use_bins else ""
            print(
                f"[{model_name}] "
                f"MSE: {mse_mean:.6f} | "
                f"CovTerm: {cov_term_print:.6e} | CovMode: {self.cov_mode} | "
                f"DispMode: {self.disp_mode} | DispMean: {disp_mean:.6e} | u^2 mean: {u2_mean:.6e} | "
                f"PenCov: {pen_cov_value:.6f} | PenDisp: {pen_disp_value:.6f} | "
                f"Corr(r,y): {corr:.6f}"
                f"{bin_info}"
            )

        return grad, hess

    # ----------------------------
    # helpers
    # ----------------------------
    @staticmethod
    def _make_bin_edges(y, n_bins, binning):
        y = np.asarray(y)

        if n_bins <= 1:
            lo = float(np.min(y))
            hi = float(np.max(y))
            if hi <= lo:
                hi = lo + 1e-12
            return np.array([lo, hi], dtype=float)

        if binning == "quantile":
            qs = np.linspace(0.0, 1.0, n_bins + 1)
            edges = np.quantile(y, qs)
            edges = np.unique(edges)
        else:  # "uniform"
            lo = float(np.min(y))
            hi = float(np.max(y))
            if hi <= lo:
                hi = lo + 1e-12
            edges = np.linspace(lo, hi, n_bins + 1)
            edges = np.unique(edges)

        if edges.size < 2:
            lo = float(np.min(y))
            edges = np.array([lo, lo + 1e-12], dtype=float)

        return edges.astype(float)

    @staticmethod
    def _bin_index(y, edges):
        edges = np.asarray(edges, dtype=float)
        K = int(max(edges.size - 1, 1))
        if K == 1:
            return np.zeros_like(y, dtype=int), 1
        idx = np.digitize(y, edges[1:-1], right=False)
        idx = np.clip(idx, 0, K - 1).astype(int)
        return idx, K

    def __str__(self):
        extra = f", disp={self.disp_mode}"
        if self.disp_mode == "pseudohuber":
            extra += f"(d={self.huber_delta})"
        return (
            f"LGBCovDispPenalty("
            f"rho_cov={self.rho_cov}, rho_disp={self.rho_disp}, "
            f"bins={self.n_bins}, weightmode={self.weight_mode}" #bindisp={self.bin_disp}"
            # f"ratio={self.ratio_mode}, cov_mode={self.cov_mode}{extra})"
        )


# 6) Surrogate of full-distributional independence of r and y

class LGBBinIndepSurrogatePenalty:
    """
    LightGBM objective: MSE + rho * (bin-independence surrogate)

    Goal: make an error-like quantity r "independent" of y by forcing its *bin-wise means*
    (across bins of y) to be the same (or equal to a target).

    Two common choices for r:
      - ratio_mode="div":  r_i = y_pred_i / max(|y_true_i|, eps_y)   (your old ratio idea)
      - ratio_mode="diff": r_i = y_pred_i - y_true_i                (log-residual if y is log-price)

    If y is log-price, and you care about *price-scale* ratios exp(y_pred)/exp(y_true),
    then using ratio_mode="diff" is usually the right primitive:
        exp(y_pred)/exp(y_true) = exp(y_pred - y_true) = exp(r)

    Penalty (mean-matching across y-bins):
      Let bins be formed on y_true.
      For each bin b:
        mu_b = mean_{i in b} r_i
      Anchor:
        - anchor_mode="global":  anchor = mu = mean(r)  (enforces mu_b ~ mu across bins)
        - anchor_mode="target":  anchor = target_value  (enforces mu_b ~ target_value in every bin)

      weights w_b:
        - weight_mode="proportional": w_b = n_b / n
        - weight_mode="uniform":      w_b = 1 / (#nonempty bins)

      penalty = 0.5 * rho * n * sum_b w_b * (mu_b - anchor)^2

    Derivatives:
      Base MSE:
        grad_base = 2 * (y_pred - y_true)
        hess_base = 2

      Let dr_i/dy_pred_i be:
        - "diff": 1
        - "div" : 1/denom_i

      The penalty is quadratic in r, so the Hessian is dense in principle,
      but we can compute an *exact diagonal* Hessian efficiently.

    Notes:
      - This is a *surrogate* for full independence: it only matches first moments across bins.
      - You can later extend it by also matching variances/quantiles per bin (not included here).
    """

    def __init__(
        self,
        rho=1e-3,
        bins=10,                    # int (#quantile bins) or array-like of bin edges
        ratio_mode="diff",          # "diff" or "div"
        anchor_mode="target",       # "global" or "target"
        target_value=None,          # default: 0 for diff, 1 for div
        weight_mode="proportional", # "proportional" or "uniform"
        eps_y=1e-12,
        zero_grad_tol=1e-6,
        lgbm_params=None,
        verbose=True,
    ):
        self.rho = float(rho)
        self.bins = bins
        self.ratio_mode = ratio_mode
        self.anchor_mode = anchor_mode
        self.target_value = target_value
        self.weight_mode = weight_mode
        self.eps_y = float(eps_y)
        self.zero_grad_tol = float(zero_grad_tol)
        self.verbose = bool(verbose)

        self.model = lgb.LGBMRegressor(**(lgbm_params or {}))
        self._call_count = 0

    # ----------------------------
    # sklearn-style API
    # ----------------------------
    def fit(self, X, y):
        y = np.asarray(y)
        self.bin_edges_ = self._make_bin_edges(y, self.bins)
        self.model.set_params(objective=self.fobj)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def __str__(self):
        return f"LGBBinIndepSurrogatePenalty(rho={self.rho}, mode={self.ratio_mode}, bins={self.bins})" #self._n_bins()})"

    # ----------------------------
    # Custom objective for LightGBM
    # ----------------------------
    def fobj(self, y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        n = y_pred.size
        self._call_count += 1

        # ---- define r and dr/dy_pred ----
        if self.ratio_mode == "diff":
            r = y_pred - y_true
            dr = np.ones_like(y_pred)
        elif self.ratio_mode == "div":
            denom = np.maximum(np.abs(y_true), self.eps_y)
            r = y_pred / denom
            dr = 1.0 / denom
        else:
            raise ValueError("ratio_mode must be 'diff' or 'div'.")

        # ---- bins on y_true ----
        bin_idx, K = self._bin_index(y_true, self.bin_edges_)
        # bin counts and sums
        n_b = np.bincount(bin_idx, minlength=K).astype(float)
        sum_r = np.bincount(bin_idx, weights=r, minlength=K).astype(float)

        # safe counts to avoid division by 0 (empty bins)
        n_b_safe = np.maximum(n_b, 1.0)
        mu_b = sum_r / n_b_safe
        mu = float(np.mean(r))

        # ---- anchor ----
        if self.anchor_mode == "global":
            anchor = mu
        elif self.anchor_mode == "target":
            if self.target_value is None:
                anchor = 1.0 if self.ratio_mode == "div" else 0.0
            else:
                anchor = float(self.target_value)
        else:
            raise ValueError("anchor_mode must be 'global' or 'target'.")

        d_b = mu_b - anchor  # bin-wise deviations from anchor

        # ---- bin weights ----
        w_b = self._bin_weights(n_b, n, mode=self.weight_mode)  # sums to 1 over nonempty bins
        Wsum = float(np.sum(w_b))

        # ---- penalty value (for logging) ----
        pen_value = 0.5 * self.rho * float(n) * float(np.sum(w_b * (d_b ** 2)))

        # ---- base MSE grads/hess ----
        grad_base = 2.0 * (y_pred - y_true)
        hess_base = 2.0 * np.ones_like(y_pred)

        # ---- penalty grad/hess (diagonal) wrt r ----
        b_i = bin_idx
        w_i = w_b[b_i]
        d_i = d_b[b_i]
        n_i = n_b_safe[b_i]

        if self.anchor_mode == "global":
            # S = sum_b w_b * (mu_b - mu)  (mu is current mean of r)
            # when anchor=mu, d_b = mu_b - mu, so:
            S = float(np.sum(w_b * d_b))
            # grad_r_i = rho*n*( w_{b(i)}*d_{b(i)}/n_{b(i)} - S/n )
            grad_r = self.rho * float(n) * (w_i * d_i / n_i - S / float(n))

            # exact diagonal of Hessian in r-space:
            # diag = w_b0*(1/n_b0 - 1/n)^2 + (Wsum - w_b0)*(1/n^2)
            diagB2 = w_i * ((1.0 / n_i - 1.0 / float(n)) ** 2) + (Wsum - w_i) * (1.0 / (float(n) ** 2))
            hess_r = self.rho * float(n) * diagB2
        else:
            # anchor is constant target: d_b = mu_b - target
            # grad_r_i = rho*n * w_{b(i)} * d_{b(i)} / n_{b(i)}
            grad_r = self.rho * float(n) * (w_i * d_i / n_i)
            # exact diagonal of Hessian in r-space:
            # diag = sum_b w_b * (d(mu_b)/dr_i)^2 = w_{b(i)}*(1/n_b)^2
            hess_r = self.rho * float(n) * (w_i * (1.0 / (n_i ** 2)))

        # ---- chain rule to y_pred ----
        grad_pen = grad_r * dr
        hess_pen = hess_r * (dr ** 2)

        grad = grad_base + grad_pen
        hess = hess_base + hess_pen

        # ---- numerical safety ----
        grad[np.abs(grad) < self.zero_grad_tol] = self.zero_grad_tol
        hess[hess < self.zero_grad_tol] = self.zero_grad_tol

        # ---- optional logging ----
        if self.verbose:
            nonempty = n_b > 0
            max_dev = float(np.max(np.abs(d_b[nonempty]))) if np.any(nonempty) else float("nan")
            mse_mean = float(np.mean((y_true - y_pred) ** 2))
            model_name = self.__str__().split("(")[0]
            print(
                f"[{model_name}] "
                f"MSE: {mse_mean:.6f} | Pen: {pen_value:.6f} | "
                f"max|bin_mean-anchor|: {max_dev:.6e} | "
                f"anchor: {anchor:.6f} | K(nonempty): {int(np.sum(nonempty))}"
            )

        return grad, hess

    # ----------------------------
    # Helpers
    # ----------------------------
    def _n_bins(self):
        edges = getattr(self, "bin_edges_", None)
        return max(int(edges.size - 1), 1) if edges is not None else None

    @staticmethod
    def _make_bin_edges(y, bins):
        y = np.asarray(y)

        # bins as explicit edges
        if hasattr(bins, "__len__") and not isinstance(bins, (str, bytes)):
            edges = np.asarray(bins, dtype=float)
            if edges.ndim != 1 or edges.size < 2:
                raise ValueError("If bins is array-like, it must be 1D with >= 2 edges.")
            edges = np.unique(np.sort(edges))
            if edges.size < 2:
                raise ValueError("Bin edges must contain at least two distinct values.")
            return edges

        # bins as int -> quantile bins
        K = int(bins)
        if K < 1:
            raise ValueError("If bins is an int, it must be >= 1.")
        qs = np.linspace(0.0, 1.0, K + 1)
        edges = np.quantile(y, qs)
        edges = np.unique(edges)
        if edges.size < 2:
            # degenerate case: all y are equal
            edges = np.array([float(np.min(y)), float(np.max(y) + 1e-12)], dtype=float)
        return edges

    @staticmethod
    def _bin_index(y, edges):
        """
        Map each y to a bin index in [0, K-1], where K = len(edges)-1.
        """
        edges = np.asarray(edges, dtype=float)
        K = int(max(edges.size - 1, 1))
        if K == 1:
            return np.zeros_like(y, dtype=int), 1
        # internal cut points exclude the ends
        idx = np.digitize(y, edges[1:-1], right=False)
        idx = np.clip(idx, 0, K - 1)
        return idx.astype(int), K

    @staticmethod
    def _bin_weights(n_b, n, mode="proportional"):
        """
        Return weights w_b that sum to 1 over nonempty bins.
        """
        n_b = np.asarray(n_b, dtype=float)
        nonempty = n_b > 0

        w = np.zeros_like(n_b)
        if not np.any(nonempty):
            return w

        if mode == "proportional":
            w[nonempty] = n_b[nonempty] / float(n)
            # sums to 1 automatically if bins cover all samples
        elif mode == "uniform":
            w[nonempty] = 1.0 / float(np.sum(nonempty))
        else:
            raise ValueError("weight_mode must be 'proportional' or 'uniform'.")
        # normalize defensively
        s = float(np.sum(w))
        if s > 0:
            w /= s
        return w












