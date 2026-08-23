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
import pandas as pd

try:
    import lightgbm as lgb
except ImportError as e:  # pragma: no cover
    raise ImportError("This module requires lightgbm. Install via `pip install lightgbm`.") from e

try:
    from sklearn.base import BaseEstimator, RegressorMixin
except ImportError as e:  # pragma: no cover
    raise ImportError("This module requires scikit-learn. Install via `pip install scikit-learn`.") from e


def _logistic_scale_from_std(std: float) -> float:
    std = float(std)
    if not np.isfinite(std) or std <= 0.0:
        return 0.0
    return float(std * np.sqrt(3.0) / np.pi)


def _empirical_logistic_quantile_proxy(values: np.ndarray, *, loc: float, scale: float) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    n = int(values.size)
    if n == 0:
        return np.asarray([], dtype=float)

    loc = float(loc)
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0.0:
        return np.full(n, loc, dtype=float)

    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    avg_ranks = np.empty(n, dtype=float)

    group_starts = np.r_[0, 1 + np.flatnonzero(sorted_values[1:] != sorted_values[:-1])]
    group_ends = np.r_[group_starts[1:], n]
    for start, end in zip(group_starts, group_ends):
        avg_rank = 0.5 * (float(start) + float(end - 1))
        avg_ranks[order[start:end]] = avg_rank

    probs = (avg_ranks + 0.5) / float(n)
    probs = np.clip(probs, 1e-12, 1.0 - 1e-12)
    return loc + scale * np.log(probs / (1.0 - probs))


def _training_early_stopping_callback(stopping_rounds: int, verbose: bool = True):
    """Stop when the tracked training metric stops improving."""
    rounds = int(stopping_rounds)
    if rounds <= 0:
        raise ValueError("stopping_rounds must be >= 1.")

    class _Callback:
        order = 30
        before_iteration = False

        def __init__(self):
            self.best_iteration = 0
            self.best_score = None
            self.best_score_list = None

        def __call__(self, env):
            if not env.evaluation_result_list:
                return

            metric = env.evaluation_result_list[0]
            score = float(metric[2])
            higher_better = bool(metric[3])

            if self.best_score is None:
                improved = True
            elif higher_better:
                improved = score > self.best_score
            else:
                improved = score < self.best_score

            if improved:
                self.best_score = score
                self.best_iteration = env.iteration
                self.best_score_list = list(env.evaluation_result_list)
                return

            if (env.iteration - self.best_iteration) >= rounds:
                if verbose:
                    metric_name = f"{metric[0]}'s {metric[1]}"
                    print(
                        f"Early stopping at iteration {env.iteration + 1}; "
                        f"best iteration was {self.best_iteration + 1} "
                        f"with {metric_name}: {self.best_score:.6f}"
                    )
                raise lgb.callback.EarlyStopException(self.best_iteration, self.best_score_list)

    return _Callback()


# ============================= MAIN MODELS =============================

# 0) Separable linear covariance penalty
class LGBCovLinearPenalty:
    """LightGBM objective: MSE - rho * n * Cov(r, y)

    Directional linear covariance penalty.

    r is chosen by ratio_mode:
      - "div"  : r = y_pred / max(|y_true|, eps_y)    (preserves old behavior)
      - "diff" : r = y_pred - y_true                  (prediction - truth)
      - "ratio": r = exp(y_pred - y_true)             (true price ratio when y is log-price)

    Cov is computed as: cov = mean( r_eff * (y_true - y_mean_) ),
    where r_eff may optionally be shifted by an "anchor" (see below).

    With objective minimization, the term - rho * n * Cov(r, y):
      - penalizes negative covariance,
      - rewards positive covariance.

    This matches the intended direction when r is defined so that negative
    covariance is the undesirable pattern (e.g. r = y_pred - y_true under
    your convention).

    Anchor note (important):
      Because yc = (y_true - y_mean_) is mean-centered, subtracting any *constant*
      anchor from r does not change cov (up to floating error). Therefore
      anchor_mode/target_value are effectively no-ops for this specific cov
      definition. Included only for API symmetry.

    Separable penalty:
      cov = mean( r_eff * (y_true - y_mean_) )
      penalty = - rho * n * cov
      grad_pen_i = - rho * (y_true_i - y_mean_) * d r_i / d y_pred_i
      hess_pen_i = - rho * (y_true_i - y_mean_) * d²r_i / d y_pred_i²

    Notes:
      - For "div" and "diff", d²r/dy_pred² = 0, so the penalty Hessian is zero.
      - For "ratio", d²r/dy_pred² = r (same clipping convention as before).
    """

    def __init__(
        self,
        rho=1e-3,
        ratio_mode="div",            # "div", "diff", or "ratio"
        anchor_mode="target",        # "none" | "target" | "iter_mean"  (no-op here; see note)
        target_value=None,           # if anchor_mode="target": default 1.0 (div/ratio) or 0.0 (diff)
        early_stopping_rounds=10,
        zero_grad_tol=1e-6,
        eps_y=1e-12,
        lgbm_params=None,
        verbose=True,
    ):
        self.rho = float(rho)
        self.ratio_mode = ratio_mode
        self.anchor_mode = anchor_mode
        self.target_value = target_value
        self.early_stopping_rounds = None if early_stopping_rounds is None else int(early_stopping_rounds)
        self.zero_grad_tol = float(zero_grad_tol)
        self.eps_y = float(eps_y)
        self.verbose = bool(verbose)
        self.model = lgb.LGBMRegressor(**(lgbm_params or {}))

    def fit(self, X, y):
        self.y_mean_ = float(np.mean(y))
        self.model.set_params(objective=self.fobj, metric="None")

        fit_kwargs = {}
        if self.early_stopping_rounds is not None and self.early_stopping_rounds > 0:
            fit_kwargs["eval_set"] = [(X, y)]
            fit_kwargs["eval_names"] = ["train"]
            fit_kwargs["eval_metric"] = self.feval
            fit_kwargs["callbacks"] = [
                _training_early_stopping_callback(
                    stopping_rounds=self.early_stopping_rounds,
                    verbose=self.verbose,
                )
            ]

        self.model.fit(X, y, **fit_kwargs)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def fobj(self, y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        n = y_pred.size

        yc = (y_true - self.y_mean_)  # centered y

        # ---- choose r, dr/dy_pred, d2r/dy_pred^2 ----
        if self.ratio_mode == "div":
            denom = np.maximum(np.abs(y_true), self.eps_y)
            r = y_pred / denom
            dr = 1.0 / denom
            d2r = np.zeros_like(y_pred)
        elif self.ratio_mode == "diff":
            r = y_pred - y_true
            dr = np.ones_like(y_pred)
            d2r = np.zeros_like(y_pred)
        elif self.ratio_mode == "ratio":
            log_ratio = np.clip(y_pred - y_true, -50.0, 50.0)
            r = np.exp(log_ratio)
            dr = r
            d2r = r
        else:
            raise ValueError("ratio_mode must be 'div', 'diff', or 'ratio'.")

        # ---- optional anchor (no-op for centered yc; kept for API symmetry) ----
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

        r_eff = r - anchor  # (effectively no change to cov because mean(yc)=0)

        # ---- covariance ----
        cov = float(np.mean(r_eff * yc))

        # ---- objective pieces (for prints) ----
        mse_vec = (y_true - y_pred) ** 2
        mse_mean = float(np.mean(mse_vec))
        pen_value = - self.rho * float(n) * cov

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

        # ---- linear cov penalty grads/hess ----
        # cov = mean(r_eff * yc), so:
        # d cov / d y_pred_i   = (1/n) * yc_i * dr_i
        # d²cov / d y_pred_i²  = (1/n) * yc_i * d2r_i
        #
        # penalty = - rho * n * cov
        # grad_pen_i = - rho * yc_i * dr_i
        # hess_pen_i = - rho * yc_i * d2r_i
        grad_pen = - self.rho * yc * dr
        hess_pen = - self.rho * yc * d2r

        grad = grad_base + grad_pen
        hess = hess_base + hess_pen

        grad[np.abs(grad) < self.zero_grad_tol] = self.zero_grad_tol
        hess[hess < self.zero_grad_tol] = self.zero_grad_tol

        return grad, hess

    def feval(self, y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        n = y_pred.size

        yc = (y_true - self.y_mean_)

        if self.ratio_mode == "div":
            denom = np.maximum(np.abs(y_true), self.eps_y)
            r = y_pred / denom
        elif self.ratio_mode == "diff":
            r = y_pred - y_true
        elif self.ratio_mode == "ratio":
            log_ratio = np.clip(y_pred - y_true, -50.0, 50.0)
            r = np.exp(log_ratio)
        else:
            raise ValueError("ratio_mode must be 'div', 'diff', or 'ratio'.")

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

        cov = float(np.mean((r - anchor) * yc))
        mse_mean = float(np.mean((y_true - y_pred) ** 2))
        pen_value = - self.rho * float(n) * cov
        return "train_loss", mse_mean + pen_value, False

    def __str__(self):
        return f"LGBCovLinearPenalty(rho={self.rho}, ratio_mode={self.ratio_mode})"

# ==========================================================
# 1) Direct covariance penalty (non-separable but usable in LightGBM via indep. assumption)
# ==========================================================

# V2: with diff/div/ratio inputs
class LGBCovPenalty:
    """LightGBM objective: MSE + rho * (Cov(r, y))^2

    r is chosen by ratio_mode:
      - "div"  : r = y_pred / max(|y_true|, eps_y)    (DEFAULT, preserves old behavior)
      - "diff" : r = y_pred - y_true                 (useful when y is log-price -> log-residual)
      - "ratio": r = exp(y_pred - y_true)            (true price ratio when y is log-price)

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
        ratio_mode="diff",          # "div", "diff", or "ratio"
        anchor_mode="target",        # "none" | "target" | "iter_mean"  (no-op here; see note)
        target_value=None,         # if anchor_mode="target": default 1.0 (div/ratio) or 0.0 (diff)
        early_stopping_rounds=10,
        zero_grad_tol=1e-6,
        eps_y=1e-12,
        lgbm_params=None,
        verbose=True,
    ):
        self.rho = float(rho)
        self.ratio_mode = ratio_mode
        self.anchor_mode = anchor_mode
        self.target_value = target_value
        self.early_stopping_rounds = None if early_stopping_rounds is None else int(early_stopping_rounds)
        self.zero_grad_tol = float(zero_grad_tol)
        self.eps_y = float(eps_y)
        self.verbose = bool(verbose)
        self.model = lgb.LGBMRegressor(**(lgbm_params or {}))

    def fit(self, X, y):
        self.y_mean_ = float(np.mean(y))
        self.model.set_params(objective=self.fobj, metric="None")

        fit_kwargs = {}
        if self.early_stopping_rounds is not None and self.early_stopping_rounds > 0:
            fit_kwargs["eval_set"] = [(X, y)]
            fit_kwargs["eval_names"] = ["train"]
            fit_kwargs["eval_metric"] = self.feval
            fit_kwargs["callbacks"] = [
                _training_early_stopping_callback(
                    stopping_rounds=self.early_stopping_rounds,
                    verbose=self.verbose,
                )
            ]

        self.model.fit(X, y, **fit_kwargs)
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
        elif self.ratio_mode == "ratio":
            log_ratio = np.clip(y_pred - y_true, -50.0, 50.0)
            r = np.exp(log_ratio)
            dr = r
        else:
            raise ValueError("ratio_mode must be 'div', 'diff', or 'ratio'.")

        # ---- optional anchor (no-op for centered yc; kept for API symmetry) ----
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

    def feval(self, y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        n = y_pred.size

        yc = (y_true - self.y_mean_)

        if self.ratio_mode == "div":
            denom = np.maximum(np.abs(y_true), self.eps_y)
            r = y_pred / denom
        elif self.ratio_mode == "diff":
            r = y_pred - y_true
        elif self.ratio_mode == "ratio":
            log_ratio = np.clip(y_pred - y_true, -50.0, 50.0)
            r = np.exp(log_ratio)
        else:
            raise ValueError("ratio_mode must be 'div', 'diff', or 'ratio'.")

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

        cov = float(np.mean((r - anchor) * yc))
        mse_mean = float(np.mean((y_true - y_pred) ** 2))
        pen_value = 0.5 * self.rho * float(n) * (cov ** 2)
        return "train_loss", mse_mean + pen_value, False

    def __str__(self):
        return f"LGBCovPenalty(rho={self.rho}, ratio_mode={self.ratio_mode})" #, anchor_mode={self.anchor_mode})"


# 1.1) CVaR for the MSE:CVaR_keep(MSE) + rho * Cov-surrogate
# Assumes: import numpy as np, import lightgbm as lgb
# Reuses your helper _cvar_topk_weights(values, keep)

class LGBCovPenaltyCVaR:
    """LightGBM objective: CVaR_keep(MSE) + rho * (Cov(r, y))^2

    This is the CVaR-robust version of LGBCovPenalty where:
      - The MSE term is replaced by a top-k (CVaR-like) average over per-sample MSE.
      - The covariance penalty remains exactly the same structure as LGBCovPenalty
        (non-separable, with diagonal Hessian approximation).

    Objective used (for reporting / gradient logic):
      J ~= CVaR_keep( (y_true - y_pred)^2 ) + 0.5 * rho * n * cov^2

    where:
      cov = mean( r_eff * (y_true - y_mean_) ),
      r depends on ratio_mode:
        - "div":  r = y_pred / max(|y_true|, eps_y)
        - "diff": r = y_pred - y_true

    Notes:
      - CVaR weights are recomputed from current mse_vec and treated as fixed inside
        this fobj call (same approximation style as your LGBSmoothPenaltyCVaR).
      - anchor_mode/target_value remain effectively no-ops for this centered covariance,
        kept only for API symmetry with LGBCovPenalty.
    """

    def __init__(
        self,
        rho=1e-3,
        mse_keep=1.0,             # same input name as in LGBSmoothPenaltyCVaR
        ratio_mode="div",         # "div" or "diff"
        anchor_mode="target",     # "none" | "target" | "iter_mean" (no-op here; API symmetry)
        target_value=None,        # if anchor_mode="target": default 1.0 (div) or 0.0 (diff)
        zero_grad_tol=1e-6,
        eps_y=1e-12,
        lgbm_params=None,
        verbose=True,
    ):
        self.rho = float(rho)
        self.mse_keep = float(mse_keep)
        self.ratio_mode = ratio_mode
        self.anchor_mode = anchor_mode
        self.target_value = target_value
        self.zero_grad_tol = float(zero_grad_tol)
        self.eps_y = float(eps_y)
        self.verbose = bool(verbose)

        if not (0.0 < self.mse_keep <= 1.0):
            raise ValueError("mse_keep must be in (0, 1].")

        self.model = lgb.LGBMRegressor(**(lgbm_params or {}))

    def fit(self, X, y):
        self.y_mean_ = float(np.mean(y))
        self.model.set_params(objective=self.fobj)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def fobj(self, y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        n = y_pred.size
        n_float = float(n)

        yc = (y_true - self.y_mean_)  # centered y

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

        r_eff = r - anchor  # effectively no change to cov because mean(yc)≈0 on training data

        # ---- covariance + penalty (same as LGBCovPenalty) ----
        cov = float(np.mean(r_eff * yc))
        pen_value = 0.5 * self.rho * n_float * (cov ** 2)

        # ---- MSE pieces + CVaR weights on MSE ----
        mse_vec = (y_true - y_pred) ** 2
        mse_mean = float(np.mean(mse_vec))
        cvar_w = _cvar_topk_weights(mse_vec, self.mse_keep)      # sums to 1
        mse_cvar = float(np.sum(cvar_w * mse_vec))               # average over worst keep-fraction
        tail_scale = n_float * cvar_w                            # converts mean-form grad to CVaR-form

        # ---- prints ----
        try:
            corr = float(np.corrcoef(r, y_true)[0, 1])
        except Exception:
            corr = float("nan")

        if self.verbose:
            model_name = self.__str__().split("(")[0]
            print(
                f"[{model_name}] "
                f"Loss: {(mse_cvar + pen_value):.6f} | "
                f"MSE(mean): {mse_mean:.6f} | MSE CVaR: {mse_cvar:.6f} | "
                f"Cov: {cov:.6e} | Pen: {pen_value:.6f} | Corr(r,y): {corr:.6f} | "
                f"mode: {self.ratio_mode} | keep: {self.mse_keep:.3f}"
            )

        # ==========================================================
        # Base term gradients/hessians: CVaR_keep(MSE)
        # Original per-sample MSE grad/hess:
        #   grad_i = 2*(pred_i - y_i)
        #   hess_i = 2
        # Replace mean MSE with sum_i cvar_w_i * mse_i
        # => per-sample scaling by n * cvar_w_i in LightGBM's mean-style convention
        # ==========================================================
        grad_base = (2.0 * (y_pred - y_true)) * tail_scale
        hess_base = (2.0 * np.ones_like(y_pred)) * tail_scale

        # ==========================================================
        # Cov penalty grads/hess (same diagonal approx as LGBCovPenalty)
        # cov = (1/n) sum_i r_i * yc_i   (anchor dropped due centering)
        # dc/dpred_i = (1/n) * yc_i * dr_i
        # penalty = 0.5 * rho * n * cov^2
        # grad_pen_i = rho * n * cov * dc/dpred_i
        # hess_pen_i = rho * n * (dc/dpred_i)^2
        # ==========================================================
        a = (yc * dr) / n_float  # dc/dpred_i

        grad_pen = self.rho * n_float * cov * a
        hess_pen = self.rho * n_float * (a ** 2)

        grad = grad_base + grad_pen
        hess = hess_base + hess_pen

        # same tolerance behavior as your original code
        grad[np.abs(grad) < self.zero_grad_tol] = self.zero_grad_tol
        hess[hess < self.zero_grad_tol] = self.zero_grad_tol

        return grad, hess

    def __str__(self):
        return (
            f"LGBCovPenaltyCVaR(rho={self.rho}, ratio_mode={self.ratio_mode}, "
            f"mse_keep={self.mse_keep})"
        )

# 1.2) CVaR for the whole objective: minimize CVaR_keep(MSE + rho * Cov-surrogate))
	# Assumes: that _cvar_topk_weights(values, keep) is already defined (same as in your code).
class LGBCovPenaltyCVaRTotal:
    """LightGBM objective: CVaR-tail selection based on a proxy of the TOTAL objective,
    while keeping the direct covariance penalty gradient/hessian from LGBCovPenalty.

    Core idea:
      - The direct covariance penalty in LGBCovPenalty is non-separable:
            0.5 * rho * n * cov^2
        so there is no exact per-sample "CVaR of total objective" decomposition.
      - We therefore build a per-sample proxy of the total objective:
            total_proxy_i = mse_i + rho * smooth_proxy_i
        where
            smooth_proxy_i = (r_i - t)^2 * (y_i - y_mean)^2
      - We compute CVaR top-k weights from total_proxy_i and apply them to the MSE term.
      - The direct covariance penalty term is kept exactly as in LGBCovPenalty
        (same diagonal Hessian approximation).

    This means:
      - Tail selection depends on BOTH accuracy and fairness-proxy ("Total/proxy" behavior).
      - The fairness gradient still comes from the direct covariance penalty (not the surrogate).

    Objective used for updates (practical approximation):
      J ~= sum_i w_i * mse_i + 0.5 * rho * n * cov^2
      where w comes from top-k(total_proxy).

    Parameters
    ----------
    rho : float
        Penalty strength for covariance term.
    mse_keep : float in (0,1]
        CVaR keep-fraction (same input name as your other CVaR class).
        keep=1.0 recovers mean-MSE weighting.
    ratio_mode : {"div","diff","ratio"}
        Same as LGBCovPenalty.
    anchor_mode : {"none","target","iter_mean"}
        Same as LGBCovPenalty (effectively no-op for centered covariance; kept for API symmetry).
    target_value : float or None
        Anchor target for direct covariance class (same semantics as LGBCovPenalty; effectively no-op).
    proxy_target_value : float or None
        Target used ONLY in the smooth proxy for tail selection:
          - default 1.0 if ratio_mode in {"div","ratio"}
          - default 0.0 if ratio_mode="diff"
    zero_grad_tol : float
        Numerical floor.
    eps_y : float
        Denominator protection for ratio_mode="div".
    lgbm_params : dict or None
        LightGBM regressor params.
    verbose : bool
        Print diagnostics each boosting call.
    """

    def __init__(
        self,
        rho=1e-3,
        mse_keep=1.0,
        ratio_mode="div",          # "div", "diff", or "ratio"
        anchor_mode="target",      # "none" | "target" | "iter_mean" (no-op for centered cov)
        target_value=None,         # direct-cov anchor target (API symmetry)
        proxy_target_value=None,   # target ONLY for total proxy tail selection
        zero_grad_tol=1e-6,
        eps_y=1e-12,
        lgbm_params=None,
        verbose=True,
    ):
        self.rho = float(rho)
        self.mse_keep = float(mse_keep)
        self.ratio_mode = ratio_mode
        self.anchor_mode = anchor_mode
        self.target_value = target_value
        self.proxy_target_value = proxy_target_value
        self.zero_grad_tol = float(zero_grad_tol)
        self.eps_y = float(eps_y)
        self.verbose = bool(verbose)

        if not (0.0 < self.mse_keep <= 1.0):
            raise ValueError("mse_keep must be in (0, 1].")

        self.model = lgb.LGBMRegressor(**(lgbm_params or {}))

    def fit(self, X, y):
        self.y_mean_ = float(np.mean(y))
        self.model.set_params(objective=self.fobj)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def fobj(self, y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        n = y_pred.size
        n_float = float(n)

        yc = (y_true - self.y_mean_)  # centered y

        # ---- choose r and dr/dy_pred ----
        if self.ratio_mode == "div":
            denom = np.maximum(np.abs(y_true), self.eps_y)
            r = y_pred / denom
            dr = 1.0 / denom
            t_proxy = 1.0 if self.proxy_target_value is None else float(self.proxy_target_value)
        elif self.ratio_mode == "diff":
            r = y_pred - y_true
            dr = np.ones_like(y_pred)
            t_proxy = 0.0 if self.proxy_target_value is None else float(self.proxy_target_value)
        elif self.ratio_mode == "ratio":
            log_ratio = np.clip(y_pred - y_true, -50.0, 50.0)
            r = np.exp(log_ratio)
            dr = r
            t_proxy = 1.0 if self.proxy_target_value is None else float(self.proxy_target_value)
        else:
            raise ValueError("ratio_mode must be 'div', 'diff', or 'ratio'.")

        # ---- optional anchor for direct covariance (effectively no-op due centered yc) ----
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

        r_eff = r - anchor  # no change to cov in exact arithmetic because mean(yc)=0

        # ---- direct covariance penalty (same as LGBCovPenalty) ----
        cov = float(np.mean(r_eff * yc))
        pen_value = 0.5 * self.rho * n_float * (cov ** 2)

        # ---- separable pieces for CVaR tail selection proxy ----
        mse_vec = (y_true - y_pred) ** 2
        smooth_proxy_vec = (r - t_proxy) ** 2 * (yc ** 2)   # same shape/idea as LGBSmoothPenalty surrogate
        total_proxy_vec = mse_vec + self.rho * smooth_proxy_vec

        # CVaR weights based on "total proxy"
        cvar_w = _cvar_topk_weights(total_proxy_vec, self.mse_keep)  # sums to 1
        tail_scale = n_float * cvar_w

        # Useful diagnostics
        mse_mean = float(np.mean(mse_vec))
        mse_cvar = float(np.sum(cvar_w * mse_vec))
        proxy_cvar = float(np.sum(cvar_w * total_proxy_vec))
        smooth_proxy_cvar = float(np.sum(cvar_w * smooth_proxy_vec))

        try:
            corr = float(np.corrcoef(r, y_true)[0, 1])
        except Exception:
            corr = float("nan")

        if self.verbose:
            model_name = self.__str__().split("(")[0]
            print(
                f"[{model_name}] "
                f"Loss(proxy-tail): {(mse_cvar + pen_value):.6f} | "
                f"MSE(mean): {mse_mean:.6f} | MSE CVaR: {mse_cvar:.6f} | "
                f"TotalProxy CVaR: {proxy_cvar:.6f} | SmoothProxy CVaR: {smooth_proxy_cvar:.6f} | "
                f"Cov: {cov:.6e} | Pen: {pen_value:.6f} | Corr(r,y): {corr:.6f} | "
                f"mode: {self.ratio_mode} | keep: {self.mse_keep:.3f} | proxy_t: {t_proxy:.6f}"
            )

        # ==========================================================
        # Base term gradients/hessians: CVaR-weighted MSE
        # ==========================================================
        grad_base = 2.0 * (y_pred - y_true) * tail_scale
        hess_base = 2.0 * np.ones_like(y_pred) * tail_scale

        # ==========================================================
        # Direct covariance penalty grads/hess (same diagonal approx as LGBCovPenalty)
        # cov = (1/n) sum_i r_i * yc_i
        # dc/dpred_i = (1/n) * yc_i * dr_i
        # penalty = 0.5 * rho * n * cov^2
        # ==========================================================
        a = (yc * dr) / n_float  # dc/dpred_i

        grad_pen = self.rho * n_float * cov * a
        hess_pen = self.rho * n_float * (a ** 2)

        # NOTE:
        # We do NOT CVaR-weight grad_pen/hess_pen here, because they come from the
        # direct non-separable covariance objective. Weighting them would be a heuristic
        # that does not correspond to this direct cov penalty derivative.
        grad = grad_base + grad_pen
        hess = hess_base + hess_pen

        # Numerical floors (mimic original behavior)
        grad[np.abs(grad) < self.zero_grad_tol] = self.zero_grad_tol
        hess[hess < self.zero_grad_tol] = self.zero_grad_tol

        return grad, hess

    def __str__(self):
        return (
            f"LGBCovPenaltyCVaRTotal(rho={self.rho}, ratio_mode={self.ratio_mode}, "
            f"mse_keep={self.mse_keep})"
        )


# ==========================================================
# 2) Plain (no dual/adversary): minimize MSE + rho * Cov-surrogate (separable)
# ==========================================================

# V2: with diff/div/ratio inputs
class LGBSmoothPenalty:
    """LightGBM custom objective: per-sample MSE + rho * separable surrogate.

    Default behavior (unchanged):
      ratio_mode="div": surrogate = ((y_pred / denom) - 1)^2 * (y_true - y_mean)^2

    Added minimal option:
      ratio_mode="diff": surrogate = ((y_pred - y_true) - 0)^2 * (y_true - y_mean)^2
      ratio_mode="ratio": surrogate = (exp(y_pred - y_true) - 1)^2 * (y_true - y_mean)^2

    Optional minimal target:
      - If ratio_mode="div": target_value defaults to 1.0
      - If ratio_mode="diff": target_value defaults to 0.0
      - If ratio_mode="ratio": target_value defaults to 1.0
      surrogate = (r - target_value)^2 * zc^2
    where:
      - r = y_pred / denom   (div)
      - r = y_pred - y_true  (diff)
      - r = exp(y_pred - y_true) (ratio)

    Notes:
      - This remains separable (per-sample), so grad/hess are exact and diagonal.
      - This does *not* exactly penalize covariance; it penalizes "ratio/residual magnitude"
        weighted by (y - mean(y))^2 (your original structure).
      - ``weighting_proxy_mode="logistic_quantile"`` keeps the same separable form but
        replaces ``(y - mean(y))`` with a rank-preserving logistic quantile proxy centered
        at the empirical log-price mean and scaled by ``sigma * sqrt(3) / pi``.
    """

    def __init__(
        self,
        rho=1e-3,
        ratio_mode="diff",        # "div" (default), "diff", or "ratio"
        weighting_proxy_mode="logistic_quantile",  # "identity" or "logistic_quantile"
        target_value=None,       # default: 1.0 for div/ratio, 0.0 for diff
        early_stopping_rounds=10,
        zero_grad_tol=1e-6,
        eps_y=1e-12,
        lgbm_params=None,
        verbose=True,
    ):
        self.rho = float(rho)
        self.ratio_mode = ratio_mode
        self.weighting_proxy_mode = str(weighting_proxy_mode).strip().lower()
        self.target_value = target_value
        self.early_stopping_rounds = None if early_stopping_rounds is None else int(early_stopping_rounds)
        self.zero_grad_tol = float(zero_grad_tol)
        self.eps_y = float(eps_y)
        self.verbose = bool(verbose)
        self.model = lgb.LGBMRegressor(**(lgbm_params or {}))

    def fit(self, X, y):
        self.y_mean_ = float(np.mean(y))
        self.y_std_ = float(np.std(y, ddof=0))
        self.y_logistic_scale_ = _logistic_scale_from_std(self.y_std_)
        if self.weighting_proxy_mode not in {"identity", "logistic_quantile"}:
            raise ValueError("weighting_proxy_mode must be 'identity' or 'logistic_quantile'.")
        if self.weighting_proxy_mode == "logistic_quantile":
            y_proxy = _empirical_logistic_quantile_proxy(
                y,
                loc=self.y_mean_,
                scale=self.y_logistic_scale_,
            )
            self.y_weight_centered_ = y_proxy - float(np.mean(y_proxy))
        else:
            self.y_weight_centered_ = np.asarray(y, dtype=float) - self.y_mean_
        self.model.set_params(objective=self.fobj, metric="None")

        fit_kwargs = {}
        if self.early_stopping_rounds is not None and self.early_stopping_rounds > 0:
            fit_kwargs["eval_set"] = [(X, y)]
            fit_kwargs["eval_names"] = ["train"]
            fit_kwargs["eval_metric"] = self.feval
            fit_kwargs["callbacks"] = [
                _training_early_stopping_callback(
                    stopping_rounds=self.early_stopping_rounds,
                    verbose=self.verbose,
                )
            ]

        self.model.fit(X, y, **fit_kwargs)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def fobj(self, y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        z = y_true
        if self.weighting_proxy_mode == "identity":
            zc = y_true - self.y_mean_
        else:
            zc = self.y_weight_centered_
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
        elif self.ratio_mode == "ratio":
            log_ratio = np.clip(y_pred - z, -50.0, 50.0)
            r = np.exp(log_ratio)
            dr = r
            t = 1.0 if self.target_value is None else float(self.target_value)
        else:
            raise ValueError("ratio_mode must be 'div', 'diff', or 'ratio'.")

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

    def feval(self, y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        z = y_true
        if self.weighting_proxy_mode == "identity":
            zc = y_true - self.y_mean_
        else:
            zc = self.y_weight_centered_
        denom = np.maximum(np.abs(z), self.eps_y)

        if self.ratio_mode == "div":
            r = y_pred / denom
            t = 1.0 if self.target_value is None else float(self.target_value)
        elif self.ratio_mode == "diff":
            r = y_pred - z
            t = 0.0 if self.target_value is None else float(self.target_value)
        elif self.ratio_mode == "ratio":
            log_ratio = np.clip(y_pred - z, -50.0, 50.0)
            r = np.exp(log_ratio)
            t = 1.0 if self.target_value is None else float(self.target_value)
        else:
            raise ValueError("ratio_mode must be 'div', 'diff', or 'ratio'.")

        mse_value = (y_true - y_pred) ** 2
        cov_surr_value = (r - t) ** 2 * (zc ** 2)
        loss_value = mse_value + self.rho * cov_surr_value
        return "train_loss", float(np.mean(loss_value)), False

    def __str__(self):
        return (
            f"LGBSmoothPenalty(rho={self.rho}, mode={self.ratio_mode}, "
            f"weighting_proxy={self.weighting_proxy_mode})"
        )



# ==========================================================
# 3) CVaR for the Surrogate: minimize MSE + rho*penalty (CVaR(MSE + rho*penalty))
# ==========================================================
def _cvar_topk_weights(values: np.ndarray, keep: float) -> np.ndarray:
    """Uniform weights over the worst keep-fraction values."""
    v = np.asarray(values, dtype=float).reshape(-1)
    n = int(v.size)
    if n == 0:
        return np.asarray([], dtype=float)
    if not (0.0 < float(keep) <= 1.0):
        raise ValueError("keep must be in (0, 1].")
    k = max(1, int(np.ceil(float(keep) * float(n))))
    idx = np.argpartition(v, -k)[-k:]
    w = np.zeros(n, dtype=float)
    w[idx] = 1.0 / float(k)
    return w


class LGBSmoothPenaltyCVaR:
    """CVaR on fairness surrogate only: mean(MSE) + rho * CVaR_keep(surrogate)."""

    def __init__(
        self,
        rho=1e-3,
        mse_keep=0.7,
        ratio_mode="div",
        target_value=None,
        zero_grad_tol=1e-6,
        eps_y=1e-12,
        lgbm_params=None,
        verbose=True,
    ):
        self.rho = float(rho)
        self.mse_keep = float(mse_keep)
        self.ratio_mode = ratio_mode
        self.target_value = target_value
        self.zero_grad_tol = float(zero_grad_tol)
        self.eps_y = float(eps_y)
        self.verbose = bool(verbose)
        if not (0.0 < self.mse_keep <= 1.0):
            raise ValueError("mse_keep must be in (0, 1].")
        self.model = lgb.LGBMRegressor(**(lgbm_params or {}))

    def fit(self, X, y):
        self.y_mean_ = float(np.mean(y))
        self.model.set_params(objective=self.fobj)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def fobj(self, y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        z = y_true
        zc = (y_true - self.y_mean_)
        denom = np.maximum(np.abs(z), self.eps_y)

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

        mse_value = (y_true - y_pred) ** 2
        cov_surr_value = (r - t) ** 2 * (zc ** 2)
        cvar_w = _cvar_topk_weights(cov_surr_value, self.mse_keep)

        if self.verbose:
            model_name = self.__str__().split("(")[0]
            try:
                corr = float(np.corrcoef(r, y_true)[0, 1])
            except Exception:
                corr = float("nan")
            print(
                f"[{model_name}] "
                f"Loss value: {(np.mean(mse_value) + self.rho * float(np.sum(cvar_w * cov_surr_value))):.6f} "
                f"| MSE value: {np.mean(mse_value):.6f} "
                f"| CovSurr CVaR: {float(np.sum(cvar_w * cov_surr_value)):.6f} "
                f"| Corr(r,y): {corr:.6f} "
                f"| mode: {self.ratio_mode} | target: {t:.6f} | keep: {self.mse_keep:.3f}"
            )

        grad_base = 2.0 * (y_pred - y_true)
        hess_base = 2.0 * np.ones_like(y_pred)

        scale = (zc ** 2)
        grad_pen = 2.0 * (r - t) * dr * scale
        hess_pen = 2.0 * (dr ** 2) * scale

        n = float(y_pred.size)
        tail_scale = n * cvar_w
        grad = grad_base + self.rho * tail_scale * grad_pen
        hess = hess_base + self.rho * tail_scale * hess_pen

        grad[np.abs(grad) < self.zero_grad_tol] = self.zero_grad_tol
        hess[hess < self.zero_grad_tol] = self.zero_grad_tol
        return grad, hess

    def __str__(self):
        return f"LGBSmoothPenaltyCVaR(rho={self.rho}, mode={self.ratio_mode}, mse_keep={self.mse_keep})"


class LGBSmoothPenaltyCVaRTotal:
    """CVaR on the total point loss: CVaR_keep(MSE + rho * surrogate)."""

    def __init__(
        self,
        rho=1e-3,
        keep=0.7,
        ratio_mode="div",
        target_value=None,
        zero_grad_tol=1e-6,
        eps_y=1e-12,
        lgbm_params=None,
        verbose=True,
    ):
        self.rho = float(rho)
        self.keep = float(keep)
        self.ratio_mode = ratio_mode
        self.target_value = target_value
        self.zero_grad_tol = float(zero_grad_tol)
        self.eps_y = float(eps_y)
        self.verbose = bool(verbose)
        if not (0.0 < self.keep <= 1.0):
            raise ValueError("keep must be in (0, 1].")
        self.model = lgb.LGBMRegressor(**(lgbm_params or {}))

    def fit(self, X, y):
        self.y_mean_ = float(np.mean(y))
        self.model.set_params(objective=self.fobj)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def fobj(self, y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        z = y_true
        zc = (y_true - self.y_mean_)
        denom = np.maximum(np.abs(z), self.eps_y)

        if self.ratio_mode == "div":
            r = y_pred / denom
            dr = 1.0 / denom
            t = 1.0 if self.target_value is None else float(self.target_value)
        elif self.ratio_mode == "diff":
            r = y_pred - z
            dr = np.ones_like(y_pred)
            t = 0.0 if self.target_value is None else float(self.target_value)
        elif self.ratio_mode == "ratio":
            log_ratio = np.clip(y_pred - z, -50.0, 50.0)
            r = np.exp(log_ratio)
            dr = r
            t = 1.0 if self.target_value is None else float(self.target_value)
        else:
            raise ValueError("ratio_mode must be 'div', 'diff', or 'ratio'.")

        mse_value = (y_true - y_pred) ** 2
        cov_surr_value = (r - t) ** 2 * (zc ** 2)
        total_point_loss = mse_value + self.rho * cov_surr_value
        cvar_w = _cvar_topk_weights(total_point_loss, self.keep)

        if self.verbose:
            model_name = self.__str__().split("(")[0]
            try:
                corr = float(np.corrcoef(r, y_true)[0, 1])
            except Exception:
                corr = float("nan")
            print(
                f"[{model_name}] "
                f"CVaR(loss): {float(np.sum(cvar_w * total_point_loss)):.6f} "
                f"| MSE value: {np.mean(mse_value):.6f} "
                f"| CovSurr value: {np.mean(cov_surr_value):.6f} "
                f"| Corr(r,y): {corr:.6f} "
                f"| mode: {self.ratio_mode} | target: {t:.6f} | keep: {self.keep:.3f}"
            )

        grad_base = 2.0 * (y_pred - y_true)
        hess_base = 2.0 * np.ones_like(y_pred)

        scale = (zc ** 2)
        grad_pen = 2.0 * (r - t) * dr * scale
        hess_pen = 2.0 * (dr ** 2) * scale

        n = float(y_pred.size)
        tail_scale = n * cvar_w
        grad = tail_scale * (grad_base + self.rho * grad_pen)
        hess = tail_scale * (hess_base + self.rho * hess_pen)

        grad[np.abs(grad) < self.zero_grad_tol] = self.zero_grad_tol
        hess[hess < self.zero_grad_tol] = self.zero_grad_tol
        return grad, hess

    def __str__(self):
        return f"LGBSmoothPenaltyCVaRTotal(rho={self.rho}, mode={self.ratio_mode}, keep={self.keep})"




# # 3) ABsolute deviation version of the surrogate
# class LGBAbsDevPenalty:
#     """LightGBM custom objective: per-sample MSE + rho * calibrated abs-deviation surrogate.

#     L1-style analogue of LGBSmoothPenalty.

#     Default behavior:
#       ratio_mode="div":
#           surrogate ≈ scale_match * y_scale^2 * |(y_pred / denom) - target| * |y_true - y_mean| / y_scale

#       ratio_mode="diff":
#           surrogate ≈ scale_match * y_scale^2 * |(y_pred - y_true) - target| / y_scale * |y_true - y_mean| / y_scale
#                    = scale_match * |(y_pred - y_true) - target| * |y_true - y_mean|

#     More precisely, to keep LightGBM stable, the abs on the prediction-dependent term is smoothed via:
#         soft_abs(u) = sqrt(u^2 + delta^2) - delta

#     where delta is small (in normalized units).

#     Parameters
#     ----------
#     rho : float
#         Penalty weight.
#     ratio_mode : str
#         "div" (default) or "diff".
#     target_value : float or None
#         Default: 1.0 for "div", 0.0 for "diff".
#     smooth_abs_delta : float
#         Small smoothing for the abs term on the prediction-dependent deviation.
#         Applied to the normalized deviation u = (r - t) / r_scale.
#     scale_mode : str
#         "rms" (default) or "mad". Determines y_scale used for normalization.
#     match_l2_scale : bool
#         If True, multiply penalty by pi/2. This roughly matches the expected magnitude
#         of |U||V| to U^2 V^2 when U,V are standardized Gaussian-like variables.
#         If False, uses literal L1-style product scale.
#     custom_match_factor : float or None
#         If provided, overrides the automatic match factor.
#     zero_grad_tol : float
#         Small floor for gradients/hessians (kept close to your current class behavior).
#     eps_y : float
#         Numerical epsilon for safe division in "div" mode.
#     eps_scale : float
#         Numerical epsilon for safe scale normalization.
#     lgbm_params : dict or None
#         Parameters passed to lgb.LGBMRegressor.
#     verbose : bool
#         Whether to print loss diagnostics inside fobj.

#     Notes
#     -----
#     - This remains separable across samples.
#     - This is NOT the same Jensen-type squared upper bound; it is the L1-style analogue.
#     - Because the regularization geometry changes (L2-like -> L1-like), you should still
#       expect the best rho for this model to differ from the squared version, even after scaling.
#     """

#     def __init__(
#         self,
#         rho=1e-3,
#         ratio_mode="div",              # "div" (default) or "diff"
#         target_value=None,             # default: 1.0 for div, 0.0 for diff
#         smooth_abs_delta=1e-3,         # smoothing for |u| on normalized deviation
#         scale_mode="rms",              # "rms" or "mad"
#         match_l2_scale=True,
#         custom_match_factor=None,
#         zero_grad_tol=1e-6,
#         eps_y=1e-12,
#         eps_scale=1e-12,
#         lgbm_params=None,
#         verbose=True,
#     ):
#         self.rho = float(rho)
#         self.ratio_mode = ratio_mode
#         self.target_value = target_value
#         self.smooth_abs_delta = float(smooth_abs_delta)
#         self.scale_mode = scale_mode
#         self.match_l2_scale = bool(match_l2_scale)
#         self.custom_match_factor = custom_match_factor
#         self.zero_grad_tol = float(zero_grad_tol)
#         self.eps_y = float(eps_y)
#         self.eps_scale = float(eps_scale)
#         self.verbose = bool(verbose)
#         self.model = lgb.LGBMRegressor(**(lgbm_params or {}))

#     def fit(self, X, y):
#         y = np.asarray(y, dtype=float)
#         self.y_mean_ = float(np.mean(y))

#         zc = y - self.y_mean_
#         if self.scale_mode == "rms":
#             self.y_scale_ = float(np.sqrt(np.mean(zc ** 2)))
#         elif self.scale_mode == "mad":
#             # robust std-like scale from MAD around mean-centered data
#             self.y_scale_ = float(np.median(np.abs(zc)) / 0.6744897501960817)
#         else:
#             raise ValueError("scale_mode must be 'rms' or 'mad'.")

#         self.y_scale_ = max(self.y_scale_, self.eps_scale)

#         if self.custom_match_factor is not None:
#             self.match_factor_ = float(self.custom_match_factor)
#         else:
#             self.match_factor_ = float(np.pi / 2.0) if self.match_l2_scale else 1.0

#         self.model.set_params(objective=self.fobj)
#         self.model.fit(X, y)
#         return self

#     def predict(self, X):
#         return self.model.predict(X)

#     def _soft_abs_with_derivatives(self, u):
#         """Smooth approximation of |u| with first and second derivatives."""
#         delta = max(self.smooth_abs_delta, self.eps_scale)
#         root = np.sqrt(u ** 2 + delta ** 2)

#         # value approximates |u| and is exactly 0 at u=0
#         val = root - delta

#         # d/du
#         grad = u / root

#         # d2/du2
#         hess = (delta ** 2) / (root ** 3)

#         return val, grad, hess

#     def fobj(self, y_true, y_pred):
#         y_true = np.asarray(y_true, dtype=float)
#         y_pred = np.asarray(y_pred, dtype=float)

#         z = y_true
#         zc = y_true - self.y_mean_
#         abs_zc_norm = np.abs(zc) / self.y_scale_

#         denom = np.maximum(np.abs(z), self.eps_y)

#         # choose r, dr/dy_pred, default target, and normalization scale for r
#         if self.ratio_mode == "div":
#             r = y_pred / denom
#             dr = 1.0 / denom
#             t = 1.0 if self.target_value is None else float(self.target_value)
#             r_scale = 1.0
#         elif self.ratio_mode == "diff":
#             r = y_pred - z
#             dr = np.ones_like(y_pred)
#             t = 0.0 if self.target_value is None else float(self.target_value)
#             r_scale = self.y_scale_
#         else:
#             raise ValueError("ratio_mode must be 'div' or 'diff'.")

#         r_scale = max(float(r_scale), self.eps_scale)

#         # normalized prediction-dependent deviation
#         u = (r - t) / r_scale

#         # smooth |u|
#         abs_u, dabs_u, ddabs_u = self._soft_abs_with_derivatives(u)

#         # calibrated penalty scale:
#         # y_scale_^2 turns the normalized L1-style product back to MSE-like units
#         pen_scale = self.match_factor_ * (self.y_scale_ ** 2)

#         # per-sample penalty
#         cov_surr_value = pen_scale * abs_u * abs_zc_norm

#         # total loss
#         mse_value = (y_true - y_pred) ** 2
#         loss_value = mse_value + self.rho * cov_surr_value

#         if self.verbose:
#             model_name = self.__str__()
#             try:
#                 corr = float(np.corrcoef(r, y_true)[0, 1])
#             except Exception:
#                 corr = float("nan")
#             print(
#                 f"[{model_name.split('(')[0]}] "
#                 f"Loss value: {np.mean(loss_value):.6f} "
#                 f"| MSE value: {np.mean(mse_value):.6f} "
#                 f"| AbsDevSurr value: {np.mean(cov_surr_value):.6f} "
#                 f"| Corr(r,y): {corr:.6f} "
#                 f"| mode: {self.ratio_mode} | target: {t:.6f} "
#                 f"| y_scale: {self.y_scale_:.6f} | match: {self.match_factor_:.6f}"
#             )

#         # base gradients/hessians for (pred - y)^2
#         grad_base = 2.0 * (y_pred - y_true)
#         hess_base = 2.0 * np.ones_like(y_pred)

#         # penalty derivatives
#         # pen_i = pen_scale * soft_abs(u_i) * |zc_i| / y_scale
#         # u_i = (r_i - t) / r_scale
#         # d pen_i / d pred_i
#         du_dpred = dr / r_scale
#         grad_pen = pen_scale * abs_zc_norm * dabs_u * du_dpred

#         # d2 pen_i / d pred_i^2
#         # since r is affine in y_pred, only ddabs_u term remains
#         hess_pen = pen_scale * abs_zc_norm * ddabs_u * (du_dpred ** 2)

#         grad = grad_base + self.rho * grad_pen
#         hess = hess_base + self.rho * hess_pen

#         # kept intentionally close to your current class for apples-to-apples comparison
#         grad[np.abs(grad) < self.zero_grad_tol] = self.zero_grad_tol
#         hess[hess < self.zero_grad_tol] = self.zero_grad_tol

#         return grad, hess

#     def __str__(self):
#         return f"LGBAbsDevPenalty(rho={self.rho}, mode={self.ratio_mode})"



# 3) Surrogate with a categorical feature control by group.
class LGBSmoothPenaltyGrouped:
    """
    LightGBM custom objective:
        MSE
      + rho * global separable surrogate
      + rho_group * grouped separable surrogate

    Global surrogate (same as original):
      ratio_mode="div":  ((y_pred / denom) - target)^2 * (y_true - global_mean)^2
      ratio_mode="diff": ((y_pred - y_true) - target)^2 * (y_true - global_mean)^2
      ratio_mode="ratio": (exp(y_pred - y_true) - target)^2 * (y_true - global_mean)^2

    Grouped surrogate (new):
      same structure, but centered within each group:
        ((r - target)^2) * (y_true - group_mean)^2

    Group aggregation:
      - "mean": each group contributes equally in aggregate (default)
      - "sum" : larger groups contribute more

    -----
    - Exact original behavior is preserved when:
        rho_group = 0.0
        or group_feature is None
    - The grouped term remains separable because group means / row weights are
      precomputed once in fit() and treated as constants during boosting.
    - This is NOT a true covariance penalty within each group; it is the same
      smooth separable surrogate structure as your current model, applied by group.
    """

    def __init__(
        self,
        rho=1e-3,
        rho_group=0.0,
        ratio_mode="div",          # "div" (default), "diff", or "ratio"
        target_value=None,         # default: 1.0 for div/ratio, 0.0 for diff
        group_feature=None,        # column name (DataFrame) or column index (ndarray)
        group_aggregation="mean",  # "mean" or "sum"
        min_group_size=2,
        zero_grad_tol=1e-6,
        eps_y=1e-12,
        lgbm_params=None,
        verbose=True,
    ):
        self.rho = float(rho)
        self.rho_group = float(rho_group)
        self.ratio_mode = ratio_mode
        self.target_value = target_value

        self.group_feature = group_feature
        self.group_aggregation = group_aggregation
        self.min_group_size = int(min_group_size)

        self.zero_grad_tol = float(zero_grad_tol)
        self.eps_y = float(eps_y)
        self.verbose = bool(verbose)

        self.model = lgb.LGBMRegressor(**(lgbm_params or {}))

        # caches built in fit()
        self.y_mean_ = None
        self.group_centered_y_ = None   # row-aligned: y_i - mean(y in group_i)
        self.group_row_weight_ = None   # row-aligned aggregation weights
        self.group_labels_ = None
        self.active_groups_ = None

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        self.y_mean_ = float(np.mean(y))

        self._prepare_group_cache(X, y)

        self.model.set_params(objective=self.fobj)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def _extract_group_values(self, X):
        """Extract the grouping feature from X."""
        if self.group_feature is None:
            return None

        # pandas DataFrame case
        if hasattr(X, "columns"):
            if isinstance(self.group_feature, str):
                if self.group_feature not in X.columns:
                    raise ValueError(
                        f"group_feature='{self.group_feature}' not found in X columns."
                    )
                g = X[self.group_feature]
            else:
                g = X.iloc[:, int(self.group_feature)]
            return pd.Series(g)

        # ndarray / array-like case
        X_arr = np.asarray(X)
        if isinstance(self.group_feature, str):
            raise ValueError(
                "group_feature as a string requires X to be a pandas DataFrame."
            )
        return pd.Series(X_arr[:, int(self.group_feature)])

    def _prepare_group_cache(self, X, y):
        """
        Precompute row-aligned grouped penalty quantities:
          group_centered_y_[i] = y_i - mean(y in group_i)
          group_row_weight_[i] = aggregation weight for group_i
        """
        n = len(y)

        self.group_centered_y_ = np.zeros(n, dtype=float)
        self.group_row_weight_ = np.zeros(n, dtype=float)
        self.group_labels_ = None
        self.active_groups_ = []

        if self.group_feature is None or self.rho_group == 0.0:
            return

        g = self._extract_group_values(X)
        g = g.astype("object").where(~pd.isna(g), "__MISSING__")

        codes, uniques = pd.factorize(g, sort=False)
        self.group_labels_ = list(uniques)

        for code, label in enumerate(uniques):
            idx = np.flatnonzero(codes == code)
            n_g = idx.size

            # Ignore tiny groups
            if n_g < self.min_group_size:
                continue

            y_g = y[idx]
            mu_g = float(np.mean(y_g))
            zc_g = y_g - mu_g

            # Equal aggregate contribution per group by default
            if self.group_aggregation == "mean":
                agg_weight = 1.0 / n_g
            elif self.group_aggregation == "sum":
                agg_weight = 1.0
            else:
                raise ValueError("group_aggregation must be 'mean' or 'sum'.")

            self.group_centered_y_[idx] = zc_g
            self.group_row_weight_[idx] = agg_weight
            self.active_groups_.append(label)

        if self.verbose:
            print(
                f"[{self.__class__.__name__}] "
                f"group_feature={self.group_feature!r} | "
                f"active_groups={len(self.active_groups_)} | "
                f"group_aggregation={self.group_aggregation} | "
                f"min_group_size={self.min_group_size} | "
                f"rho_group={self.rho_group}"
            )

    def fobj(self, y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        z = y_true
        zc_global = y_true - self.y_mean_
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
        elif self.ratio_mode == "ratio":
            log_ratio = np.clip(y_pred - z, -50.0, 50.0)
            r = np.exp(log_ratio)
            dr = r
            t = 1.0 if self.target_value is None else float(self.target_value)
        else:
            raise ValueError("ratio_mode must be 'div', 'diff', or 'ratio'.")

        # -------------------------
        # base MSE
        # -------------------------
        mse_value = (y_true - y_pred) ** 2
        grad_base = 2.0 * (y_pred - y_true)
        hess_base = 2.0 * np.ones_like(y_pred)

        # -------------------------
        # original global penalty
        # -------------------------
        global_scale = zc_global ** 2
        cov_surr_value = (r - t) ** 2 * global_scale
        grad_pen_global = 2.0 * (r - t) * dr * global_scale
        hess_pen_global = 2.0 * (dr ** 2) * global_scale

        # -------------------------
        # new grouped penalty
        # -------------------------
        if self.group_centered_y_ is not None and self.group_row_weight_ is not None:
            group_scale = self.group_centered_y_ ** 2
            row_weight = self.group_row_weight_

            group_cov_surr_value = row_weight * ((r - t) ** 2) * group_scale
            grad_pen_group = row_weight * 2.0 * (r - t) * dr * group_scale
            hess_pen_group = row_weight * 2.0 * (dr ** 2) * group_scale
        else:
            group_cov_surr_value = np.zeros_like(y_true)
            grad_pen_group = np.zeros_like(y_true)
            hess_pen_group = np.zeros_like(y_true)

        # full loss
        loss_value = (
            mse_value
            + self.rho * cov_surr_value
            + self.rho_group * group_cov_surr_value
        )

        # full gradient / hessian
        grad = (
            grad_base
            + self.rho * grad_pen_global
            + self.rho_group * grad_pen_group
        )

        hess = (
            hess_base
            + self.rho * hess_pen_global
            + self.rho_group * hess_pen_group
        )

        # diagnostics
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
                f"| GroupCovSurr value: {np.sum(group_cov_surr_value):.6f} "
                f"| Corr(r,y): {corr:.6f} "
                f"| mode: {self.ratio_mode} | target: {t:.6f}"
            )

        # zero tol (kept to preserve your original behavior)
        grad[np.abs(grad) < self.zero_grad_tol] = self.zero_grad_tol
        hess[hess < self.zero_grad_tol] = self.zero_grad_tol

        return grad, hess

    def __str__(self):
        return (
            f"LGBSmoothPenaltyGrouped("
            f"rho={self.rho}, "
            f"rho_group={self.rho_group}, "
            f"mode={self.ratio_mode}, "
            f"group_feature={self.group_feature}"
            f")"
        )



class LGBSmoothPenaltyGroupCVaR:
    """LightGBM custom objective:
        per-sample MSE
      + rho * separable surrogate
      + eta * worst-group group-MSE CVaR-like term.

    Original behavior (unchanged):
      ratio_mode="div":
          surrogate = ((y_pred / denom) - 1)^2 * (y_true - y_mean)^2

    Added minimal options:
      ratio_mode="diff":
          surrogate = ((y_pred - y_true) - 0)^2 * (y_true - y_mean)^2

      ratio_mode="ratio":
          surrogate = (exp(y_pred - y_true) - 1)^2 * (y_true - y_mean)^2

    Optional target_value:
      - div   -> default 1.0
      - diff  -> default 0.0
      - ratio -> default 1.0

    New worst-group accuracy term:
      - Groups are defined by `group_feature`
      - If `group_feature` is a list/tuple of features, the group CVaR term uses
        their interaction (cartesian combination of values). Interaction groups
        with fewer than 100 samples are pooled into one fallback bucket.
      - Group loss:
            L_g = mean_{i in g} (y_pred_i - y_true_i)^2
      - keep in (0,1] is the fraction of groups to keep as "worst groups"
        (if keep > 1 and <= 100, it is interpreted as a percentage and divided by 100)
      - Let k = round(keep * G), clipped to [1, G]
      - Group CVaR-like term:
            worst_group_value = mean of top-k group losses
      - Final objective:
            mse + rho * cov_surr + eta * worst_group_value

    Notes:
      - The new group term is non-separable because the active top-k groups depend
        on the current prediction vector.
      - The gradient/Hessian below are exact conditional on the current active set.
        At ties / active-set changes, this acts as a practical subgradient.
      - The grouping feature is only used to define the groups; it is not removed
        from X and can still be used by the model as a predictive feature.
    """

    def __init__(
        self,
        rho=1e-3,
        eta=0.0,
        keep=0.10,               # top fraction of worst groups to keep
        group_feature=None,      # column/index or list/tuple of them for interaction groups
        ratio_mode="div",        # "div" (default), "diff", or "ratio"
        target_value=None,       # default: 1.0 for div/ratio, 0.0 for diff
        early_stopping_rounds=10,
        zero_grad_tol=1e-6,
        eps_y=1e-12,
        lgbm_params=None,
        verbose=True,
    ):
        self.rho = float(rho)
        self.eta = float(eta)
        self.keep = float(keep)
        self.group_feature = group_feature

        # accept either fraction in (0,1] or percentage in (1,100]
        if self.keep <= 0:
            raise ValueError("keep must be > 0.")
        if self.keep > 1.0:
            if self.keep <= 100.0:
                self.keep = self.keep / 100.0
            else:
                raise ValueError("keep must be in (0,1] or in (0,100].")

        self.ratio_mode = ratio_mode
        self.target_value = target_value
        self.early_stopping_rounds = (
            None if early_stopping_rounds is None else int(early_stopping_rounds)
        )
        self.zero_grad_tol = float(zero_grad_tol)
        self.eps_y = float(eps_y)
        self.verbose = bool(verbose)

        self.model = lgb.LGBMRegressor(**(lgbm_params or {}))
        self.interaction_min_group_size_ = 100

        # filled in fit()
        self.y_mean_ = None
        self.group_ids_ = None
        self.group_sizes_ = None
        self.group_inv_sizes_per_sample_ = None
        self.group_labels_ = None
        self.n_groups_ = 0
        self.n_kept_groups_ = 0

    def fit(self, X, y):
        y = np.asarray(y)
        self.y_mean_ = float(np.mean(y))

        self._prepare_groups(X)

        self.model.set_params(objective=self.fobj, metric="None")

        fit_kwargs = {}
        if self.early_stopping_rounds is not None and self.early_stopping_rounds > 0:
            fit_kwargs["eval_set"] = [(X, y)]
            fit_kwargs["eval_names"] = ["train"]
            fit_kwargs["eval_metric"] = self.feval
            fit_kwargs["callbacks"] = [
                _training_early_stopping_callback(
                    stopping_rounds=self.early_stopping_rounds,
                    verbose=self.verbose,
                )
            ]

        self.model.fit(X, y, **fit_kwargs)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def _prepare_groups(self, X):
        """Extract grouping feature from X and precompute group structures."""
        self.group_ids_ = None
        self.group_sizes_ = None
        self.group_inv_sizes_per_sample_ = None
        self.group_labels_ = None
        self.n_groups_ = 0
        self.n_kept_groups_ = 0

        if self.eta <= 0.0:
            return

        if self.group_feature is None:
            raise ValueError(
                "group_feature must be provided when eta > 0 "
                "to define the worst-group CVaR-like term."
            )

        group_values = self._extract_group_values(X)
        group_ids, group_labels = self._factorize_group_values(group_values)

        if self._is_interaction_group_feature():
            group_sizes = np.bincount(group_ids, minlength=len(group_labels))
            small_group_mask = group_sizes < self.interaction_min_group_size_
            if np.any(small_group_mask):
                pooled_values = np.asarray(group_values, dtype=object).copy()
                pooled_values[small_group_mask[group_ids]] = "__SMALL_INTERACTION_GROUP__"
                group_ids, group_labels = self._factorize_group_values(pooled_values)

        self.group_ids_ = group_ids.astype(np.int32, copy=False)
        self.group_labels_ = np.asarray(group_labels, dtype=object)
        self.n_groups_ = int(len(self.group_labels_))

        if self.n_groups_ <= 0:
            raise ValueError("No groups were found from group_feature.")

        self.n_kept_groups_ = int(np.round(self.keep * self.n_groups_))
        self.n_kept_groups_ = max(1, min(self.n_groups_, self.n_kept_groups_))

        self.group_sizes_ = np.bincount(
            self.group_ids_, minlength=self.n_groups_
        ).astype(float)

        if np.any(self.group_sizes_ <= 0):
            raise ValueError("At least one group has zero size, which should not happen.")

        self.group_inv_sizes_per_sample_ = 1.0 / self.group_sizes_[self.group_ids_]

        if self.verbose:
            print(
                f"[{self.__class__.__name__}] "
                f"group_feature={self.group_feature!r} | "
                f"total groups={self.n_groups_} | "
                f"min_group_size={self.interaction_min_group_size_ if self._is_interaction_group_feature() else 'n/a'} | "
                f"keep={self.keep:.2%} | "
                f"groups kept={self.n_kept_groups_}"
            )

    def _is_interaction_group_feature(self):
        return isinstance(self.group_feature, (list, tuple)) and len(self.group_feature) > 1

    def _extract_group_values(self, X):
        """Read the grouping feature from DataFrame or ndarray-like X."""
        # pandas DataFrame-like
        if hasattr(X, "iloc"):
            if isinstance(self.group_feature, (list, tuple)):
                feature_specs = list(self.group_feature)
                if not feature_specs:
                    raise ValueError("group_feature list cannot be empty.")
                if len(feature_specs) == 1:
                    feature_specs = [feature_specs[0]]
                parts = []
                for feature in feature_specs:
                    if isinstance(feature, str):
                        if feature not in X.columns:
                            raise ValueError(
                                f"group_feature={feature!r} not found in DataFrame columns."
                            )
                        part = pd.Series(X[feature], copy=False)
                    else:
                        idx = int(feature)
                        part = pd.Series(X.iloc[:, idx], copy=False)
                    part = part.astype(object).where(~pd.isna(part), "__MISSING__").astype(str)
                    parts.append(part.to_numpy(dtype=object))
                if len(parts) == 1:
                    return np.asarray(parts[0], dtype=object)
                return np.asarray(
                    ["__X__".join(values) for values in zip(*parts)],
                    dtype=object,
                )
            if isinstance(self.group_feature, str):
                if self.group_feature not in X.columns:
                    raise ValueError(
                        f"group_feature={self.group_feature!r} not found in DataFrame columns."
                    )
                return np.asarray(X[self.group_feature])
            idx = int(self.group_feature)
            return np.asarray(X.iloc[:, idx])

        # ndarray-like
        X_arr = np.asarray(X)
        if X_arr.ndim != 2:
            raise ValueError("X must be 2D.")
        if isinstance(self.group_feature, (list, tuple)):
            feature_specs = list(self.group_feature)
            if not feature_specs:
                raise ValueError("group_feature list cannot be empty.")
            parts = []
            for feature in feature_specs:
                if not isinstance(feature, (int, np.integer)):
                    raise ValueError(
                        "For ndarray inputs, group_feature list entries must be integer column indices."
                    )
                parts.append(np.asarray(X_arr[:, int(feature)], dtype=object))
            if len(parts) == 1:
                return np.asarray(parts[0], dtype=object)
            return np.asarray(
                ["__X__".join("__MISSING__" if value is None else str(value) for value in values) for values in zip(*parts)],
                dtype=object,
            )
        if not isinstance(self.group_feature, (int, np.integer)):
            raise ValueError(
                "For ndarray inputs, group_feature must be an integer column index."
            )
        idx = int(self.group_feature)
        return np.asarray(X_arr[:, idx])

    def _factorize_group_values(self, values):
        """Factorize group values, treating missing values as one explicit group."""
        if pd is not None:
            s = pd.Series(values, copy=False)
            s = s.astype(object)
            s = s.where(~pd.isna(s), "__MISSING__")
            codes, uniques = pd.factorize(s, sort=False)
            return np.asarray(codes), list(uniques)

        # fallback without pandas
        vals = np.asarray(values, dtype=object).copy()
        for i, v in enumerate(vals):
            if v is None:
                vals[i] = "__MISSING__"
            else:
                try:
                    if np.isnan(v):
                        vals[i] = "__MISSING__"
                except Exception:
                    pass
        uniques, codes = np.unique(vals, return_inverse=True)
        return np.asarray(codes), list(uniques)

    def _compute_group_cvar_term(self, y_true, y_pred):
        """Compute worst-group group-MSE top-k average and its conditional grad/hess."""
        if self.eta <= 0.0 or self.group_ids_ is None:
            zero = np.zeros_like(y_pred, dtype=float)
            return 0.0, zero, zero, None, None

        err = y_pred - y_true
        sq_err = err ** 2

        group_sse = np.bincount(
            self.group_ids_,
            weights=sq_err,
            minlength=self.n_groups_,
        )
        group_mse = group_sse / self.group_sizes_

        k = self.n_kept_groups_

        if k >= self.n_groups_:
            active_group_ids = np.arange(self.n_groups_, dtype=np.int32)
        else:
            # top-k largest group MSEs
            active_group_ids = np.argpartition(group_mse, -k)[-k:].astype(np.int32)

        active_group_mask = np.zeros(self.n_groups_, dtype=bool)
        active_group_mask[active_group_ids] = True
        active_sample_mask = active_group_mask[self.group_ids_]

        worst_group_value = float(np.mean(group_mse[active_group_ids]))

        # Conditional gradient / Hessian for:
        #   (1/k) * sum_{g in active} (1/n_g) * sum_{i in g} err_i^2
        # So for i in active group g:
        #   d/d pred_i = (1/k) * 2 * err_i / n_g
        #   d2/d pred_i2 = (1/k) * 2 / n_g
        grad = np.zeros_like(y_pred, dtype=float)
        hess = np.zeros_like(y_pred, dtype=float)

        coeff = (2.0 / k) * self.group_inv_sizes_per_sample_
        grad[active_sample_mask] = coeff[active_sample_mask] * err[active_sample_mask]
        hess[active_sample_mask] = coeff[active_sample_mask]

        return worst_group_value, grad, hess, group_mse, active_group_ids

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
        elif self.ratio_mode == "ratio":
            log_ratio = np.clip(y_pred - z, -50.0, 50.0)
            r = np.exp(log_ratio)
            dr = r
            t = 1.0 if self.target_value is None else float(self.target_value)
        else:
            raise ValueError("ratio_mode must be 'div', 'diff', or 'ratio'.")

        # losses
        mse_value = (y_true - y_pred) ** 2
        cov_surr_value = (r - t) ** 2 * (zc ** 2)

        worst_group_value, grad_group, hess_group, group_mse, active_group_ids = (
            self._compute_group_cvar_term(y_true, y_pred)
        )

        loss_value = (
            mse_value
            + self.rho * cov_surr_value
            + self.eta * worst_group_value
        )

        if self.verbose:
            model_name = self.__str__()
            try:
                corr = float(np.corrcoef(r, y_true)[0, 1])
            except Exception:
                corr = float("nan")

            msg = (
                f"[{model_name.split('(')[0]}] "
                f"Loss value: {np.mean(loss_value):.6f} "
                f"| MSE value: {np.mean(mse_value):.6f} "
                f"| CovSurr value: {np.mean(cov_surr_value):.6f} "
                f"| Corr(r,y): {corr:.6f} "
                f"| mode: {self.ratio_mode} | target: {t:.6f}"
            )

            if self.eta > 0.0 and self.group_ids_ is not None:
                msg += (
                    f" | GroupCVaR value: {worst_group_value:.6f} "
                    f"| groups: {self.n_groups_} "
                    f"| kept: {self.n_kept_groups_}"
                )
                if group_mse is not None and active_group_ids is not None:
                    msg += (
                        f" | max group MSE: {np.max(group_mse):.6f} "
                        f"| mean kept group MSE: {np.mean(group_mse[active_group_ids]):.6f}"
                    )

            print(msg)

        # base gradients/hessians for (pred-y)^2
        grad_base = 2.0 * (y_pred - y_true)
        hess_base = 2.0 * np.ones_like(y_pred)

        # penalty gradients/hessians (separable, exact)
        # pen_i = (r_i - t)^2 * zc_i^2
        # d pen_i / d pred_i = 2 * (r_i - t) * dr_i * zc_i^2
        scale = (zc ** 2)
        grad_pen = 2.0 * (r - t) * dr * scale
        hess_pen = 2.0 * (dr ** 2) * scale

        grad = grad_base + self.rho * grad_pen + self.eta * grad_group
        hess = hess_base + self.rho * hess_pen + self.eta * hess_group

        # zero tol
        grad[np.abs(grad) < self.zero_grad_tol] = self.zero_grad_tol
        hess[hess < self.zero_grad_tol] = self.zero_grad_tol

        return grad, hess

    def feval(self, y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        z = y_true
        zc = (y_true - self.y_mean_)
        denom = np.maximum(np.abs(z), self.eps_y)

        if self.ratio_mode == "div":
            r = y_pred / denom
            t = 1.0 if self.target_value is None else float(self.target_value)
        elif self.ratio_mode == "diff":
            r = y_pred - z
            t = 0.0 if self.target_value is None else float(self.target_value)
        elif self.ratio_mode == "ratio":
            log_ratio = np.clip(y_pred - z, -50.0, 50.0)
            r = np.exp(log_ratio)
            t = 1.0 if self.target_value is None else float(self.target_value)
        else:
            raise ValueError("ratio_mode must be 'div', 'diff', or 'ratio'.")

        mse_value = (y_true - y_pred) ** 2
        cov_surr_value = (r - t) ** 2 * (zc ** 2)

        worst_group_value, _, _, _, _ = self._compute_group_cvar_term(y_true, y_pred)

        loss_value = (
            mse_value
            + self.rho * cov_surr_value
            + self.eta * worst_group_value
        )
        return "train_loss", float(np.mean(loss_value)), False

    def __str__(self):
        return (
            f"LGBSmoothPenaltyGroupCVaR("
            f"rho={self.rho}, eta={self.eta}, keep={self.keep}, "
            f"group_feature={self.group_feature}, mode={self.ratio_mode})"
        )






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

# class LGBPrimalDual:
#     def __init__(self, rho=1e-3, keep=0.7, adversary_type="overall", eta_adv=0.1, zero_grad_tol=1e-6, lgbm_params=None):
#         self.rho = rho
#         self.keep = keep
#         self.adversary_type = adversary_type
#         self.eta_adv = eta_adv
#         self.zero_grad_tol = zero_grad_tol
#         self.model = lgb.LGBMRegressor(**lgbm_params)

#     def fit(self, X, y):
#         # cache for the callback
#         self.X_ = X
#         self.y_ = y
#         self.y_mean_ = np.mean(y)
#         self.n_ = y.size
#         # Update: cache current ensemble predictions so we can update them incrementally
#         self.y_hat_ = np.ones(self.n_) * self.y_mean_   # matches boost_from_average=True default

#         # CVaR cap: w_i <= 1/(alpha*n), with alpha = keep
#         self.cap_ = 1.0 / (max(1, int(self.keep * self.n_)) )  # = 1/K

#         # initialize adversary weights (uniform)
#         w0 = np.ones(self.n_) / self.n_
#         if self.adversary_type == "overall":
#             self.w_ = w0
#         elif self.adversary_type == "individual":
#             self.p_ = w0
#             self.q_ = w0
#         else:
#             raise ValueError(f"No adversary_type called: {self.adversary_type}")

#         # Update lgbm params
#         self.model.set_params(objective=self.fobj)
#         self.model.fit(X, y, callbacks=[self._adv_callback])

#     def predict(self, X):
#         return self.model.predict(X)

#     def _project_capped_simplex(self, w):
#         """Project to {w>=0, sum w=1, w_i<=cap_} (simple cap + redistribute)."""
#         w = np.maximum(w, 0)
#         if w.sum() <= 0:
#             w = np.ones_like(w) / w.size
#         else:
#             w = w / w.sum()

#         cap = self.cap_
#         # cap-and-redistribute until feasible (usually 1-2 passes)
#         for _ in range(10):
#             over = w > cap
#             if not np.any(over):
#                 break
#             excess = w[over].sum() - cap * over.sum()
#             w[over] = cap
#             under = ~over
#             if not np.any(under):
#                 # everything capped -> already sums to 1 by definition of cap=1/K
#                 break
#             w[under] += excess * (w[under] / w[under].sum())
#         return w

#     def _mirror_step(self, w, v):
#         # exponentiated-gradient / mirror-ascent step
#         z = self.eta_adv * (v - np.max(v))
#         w_new = w * np.exp(z)
#         return self._project_capped_simplex(w_new)

#     def _adv_callback(self, env):
#         # update adversary once per boosting iteration using current predictions
#         it = env.iteration + 1
#         # y_hat = env.model.predict(self.X_, num_iteration=it)
#         # Update: predict ONLY the new tree’s contribution and add it to cached predictions
#         delta = env.model.predict(self.X_, start_iteration=it-1, num_iteration=1)
#         self.y_hat_ = self.y_hat_ + delta
#         y_hat = self.y_hat_

#         mse_value = (self.y_ - y_hat) ** 2
#         cov_surr_value = (y_hat / self.y_ - 1) ** 2 * (self.y_ - self.y_mean_) ** 2

#         if self.adversary_type == "overall":
#             v = mse_value + self.rho * cov_surr_value
#             self.w_ = self._mirror_step(self.w_, v)
#         else:
#             self.p_ = self._mirror_step(self.p_, mse_value)
#             self.q_ = self._mirror_step(self.q_, cov_surr_value)

#     def fobj(self, y_true, y_pred):
#         # Loss function value (same prints as yours)
#         mse_value = (y_true - y_pred) ** 2
#         cov_surr_value = (y_pred / y_true - 1) ** 2 * (y_true - np.mean(y_true)) ** 2
#         loss_value = mse_value + self.rho * cov_surr_value
#         model_name = self.__str__()
#         print(
#             f"[{model_name.split('(')[0]}] "
#             f"Loss value: {np.mean(loss_value):.6f} "
#             f"| MSE value: {np.mean(mse_value):.6f} "
#             f"| CovSurr value: {np.mean(cov_surr_value):.6f} "
#             f"| Corr(r,y): {np.corrcoef(y_pred / y_true, y_true)[0, 1]:.6f} "
#         )

#         # base gradients/hessians for 0.5*(pred-y)^2  (now for ALL samples)
#         grad_base = 2 * (y_pred - y_true)
#         hess_base = 2 * np.ones_like(y_pred)

#         # penalty gradients/hessians (same structure as yours, for ALL samples)
#         z = y_true
#         z_c = (y_true - np.mean(y_true))
#         grad_pen = 2 * (y_pred - z) * (z_c / z) ** 2
#         hess_pen = 2 * (z_c / z) ** 2

#         n = y_pred.size

#         # primal step uses current adversary weights (scaled by n so magnitudes stay reasonable)
#         if self.adversary_type == "overall":
#             w_eff = n * self.w_
#             grad = w_eff * (grad_base + self.rho * grad_pen)
#             hess = w_eff * (hess_base + self.rho * hess_pen)
#         else:
#             p_eff = n * self.p_
#             q_eff = n * self.q_
#             grad = p_eff * grad_base + self.rho * q_eff * grad_pen
#             hess = p_eff * hess_base + self.rho * q_eff * hess_pen

#         # zero grad/hess tol (same as yours)
#         grad[grad == 0] += self.zero_grad_tol
#         hess[hess == 0] += self.zero_grad_tol

#         return grad, hess

#     def __str__(self):
#         return f"LGBPrimalDual({self.rho}, {self.adversary_type}, {self.eta_adv})" #adversary_type={self.adversary_type})" #, eta_adv={self.eta_adv}, tol={self.zero_grad_tol})"



### 

# Models to compare primal-dual method

###

# # 0) Just MSE but binning by y-real
# class LGBBinnedMSEWeights:
#     """
#     LightGBM custom objective: weighted MSE only.

#     Purpose:
#       A super simple baseline to test how *re-weighting by y-bins* affects LightGBM,
#       without changing the loss beyond weights.

#     We build bins on y_true (ground truth) using either:
#       - binning="quantile": equal-count bins (by quantiles)
#       - binning="uniform" : equal-width bins in y-scale

#     Then assign each sample a weight w_i based on its bin.
#     By default, we use inverse-frequency weights so each bin contributes equally.

#     Objective (per sample):
#       loss_i = w_i * (y_pred_i - y_true_i)^2

#     Grad/Hess:
#       grad_i = 2 * w_i * (y_pred_i - y_true_i)
#       hess_i = 2 * w_i

#     Notes:
#       - This is equivalent to passing sample_weight into LightGBM training,
#         but implemented as a custom objective to keep the same "structure" style.
#       - We clamp weights to be >= weight_floor for numerical stability.
#     """

#     def __init__(
#         self,
#         n_bins=10,
#         binning="quantile",            # "quantile" or "uniform"
#         weight_mode="inv_freq",        # "inv_freq" or "freq" or "none"
#         weight_floor=1e-8,
#         zero_grad_tol=1e-6,
#         lgbm_params=None,
#         verbose=True,
#     ):
#         self.n_bins = int(n_bins)
#         self.binning = str(binning)
#         self.weight_mode = str(weight_mode)
#         self.weight_floor = float(weight_floor)
#         self.zero_grad_tol = float(zero_grad_tol)
#         self.verbose = bool(verbose)

#         self.model = lgb.LGBMRegressor(**(lgbm_params or {}))

#     def fit(self, X, y):
#         y = np.asarray(y)

#         if self.n_bins < 1:
#             raise ValueError("n_bins must be >= 1")
#         if self.binning not in ("quantile", "uniform"):
#             raise ValueError("binning must be 'quantile' or 'uniform'")
#         if self.weight_mode not in ("inv_freq", "freq", "none"):
#             raise ValueError("weight_mode must be 'inv_freq', 'freq', or 'none'")

#         self.bin_edges_ = self._make_bin_edges(y, self.n_bins, self.binning)
#         self.bin_idx_, self.n_bins_eff_ = self._bin_index(y, self.bin_edges_)
#         self.weights_, self.bin_counts_ = self._compute_weights(self.bin_idx_, self.n_bins_eff_)

#         if self.verbose:
#             counts = self.bin_counts_.astype(int).tolist()
#             w_stats = (float(np.min(self.weights_)), float(np.mean(self.weights_)), float(np.max(self.weights_)))
#             print(f"[LGBBinnedMSEWeights] bins={self.n_bins_eff_} | binning={self.binning} | "
#                   f"weight_mode={self.weight_mode} | counts={counts} | "
#                   f"w(min/mean/max)={w_stats[0]:.3e}/{w_stats[1]:.3e}/{w_stats[2]:.3e}")

#         self.model.set_params(objective=self.fobj)
#         self.model.fit(X, y)
#         return self

#     def predict(self, X):
#         return self.model.predict(X)

#     def fobj(self, y_true, y_pred):
#         y_true = np.asarray(y_true)
#         y_pred = np.asarray(y_pred)

#         w = self.weights_

#         # weighted MSE grads/hess
#         grad = 2.0 * w * (y_pred - y_true)
#         hess = 2.0 * w * np.ones_like(y_pred)

#         # numerical guards (same style as your other code)
#         grad[np.abs(grad) < self.zero_grad_tol] = self.zero_grad_tol
#         hess[hess < self.zero_grad_tol] = self.zero_grad_tol

#         if self.verbose:
#             mse = float(np.mean((y_pred - y_true) ** 2))
#             wmse = float(np.mean(w * (y_pred - y_true) ** 2))
#             print(f"[LGBBinnedMSEWeights] MSE: {mse:.6f} | weighted MSE: {wmse:.6f}")

#         return grad, hess

#     # ----------------------------
#     # helpers
#     # ----------------------------
#     @staticmethod
#     def _make_bin_edges(y, n_bins, binning):
#         y = np.asarray(y)

#         if n_bins == 1:
#             # single bin covers all
#             lo = float(np.min(y))
#             hi = float(np.max(y))
#             if hi <= lo:
#                 hi = lo + 1e-12
#             return np.array([lo, hi], dtype=float)

#         if binning == "quantile":
#             qs = np.linspace(0.0, 1.0, n_bins + 1)
#             edges = np.quantile(y, qs)
#             edges = np.unique(edges)
#         else:  # "uniform"
#             lo = float(np.min(y))
#             hi = float(np.max(y))
#             if hi <= lo:
#                 hi = lo + 1e-12
#             edges = np.linspace(lo, hi, n_bins + 1)

#         # ensure strictly increasing edges
#         edges = np.unique(np.asarray(edges, dtype=float))
#         if edges.size < 2:
#             lo = float(np.min(y))
#             edges = np.array([lo, lo + 1e-12], dtype=float)

#         return edges

#     @staticmethod
#     def _bin_index(y, edges):
#         edges = np.asarray(edges, dtype=float)
#         K = int(max(edges.size - 1, 1))

#         if K == 1:
#             return np.zeros_like(y, dtype=int), 1

#         # digitize on internal edges
#         idx = np.digitize(y, edges[1:-1], right=False)
#         idx = np.clip(idx, 0, K - 1).astype(int)

#         return idx, K

#     def _compute_weights(self, bin_idx, K):
#         n = bin_idx.size
#         counts = np.bincount(bin_idx, minlength=K).astype(float)
#         counts_safe = np.maximum(counts, 1.0)

#         if self.weight_mode == "none":
#             w_b = np.ones(K, dtype=float)
#         elif self.weight_mode == "freq":
#             # proportional to bin frequency (mostly pointless, but included)
#             w_b = counts_safe / float(n)
#         else:
#             # inv_freq: equalize bin contributions
#             # Each bin gets total weight ≈ 1/K:
#             # w_i = (n / (K * n_b))
#             w_b = float(n) / (float(K) * counts_safe)

#         w = w_b[bin_idx]
#         w = np.maximum(w, self.weight_floor)

#         return w, counts
    
#     def __str__(self):
#         return f"LGBBinnedMSEWeights(n_bins={self.n_bins}, binning={self.binning})"

# # ==========================================================
# # 4) Direct K-moments penalty (non-separable but usable in LightGBM via global stats ???)
# # ==========================================================

# # 4.5) Cov but Var(E(R|Y))
# class LGBVarCondMeanPenalty:
#     """LightGBM objective: MSE + rho * Var( E[r | y] )

#     We approximate E[r|y] by *quantile bins* on y (computed once in fit()).

#     r is chosen by ratio_mode:
#       - "div"  : r = y_pred / max(|y_true|, eps_y)
#       - "diff" : r = y_pred - y_true
#       - "ratio": r = y_pred / max(|y_true|, eps_y)   (alias for "div"; kept for symmetry)

#     Vertical penalty (binned):
#       - Let bins B_k partition y. Let m_k = mean_{i in B_k} r_i and m_bar = mean_i r_i.
#       - Var(E[r|y]) ≈ sum_k (n_k/n) * (m_k - m_bar)^2
#       - penalty = 0.5 * rho * n * Var(E[r|y])

#     Grad/Hess (diagonal approximation):
#       Exact (in r-space):
#         d penalty / d r_i = rho * (m_{bin(i)} - m_bar)
#         d^2 penalty / d r_i^2 = rho * (1/n_{bin(i)} - 1/n)   (>=0)

#       Chain rule with r = r(y_pred):
#         grad_pen_i = rho * (m_bin - m_bar) * dr_i
#         hess_pen_i = rho * (1/n_bin - 1/n) * (dr_i)^2
#     """

#     def __init__(
#         self,
#         rho=1e-3,
#         n_bins=10,                # NEW: number of quantile bins on y_true
#         ratio_mode="div",         # "div" | "diff" | "ratio"
#         anchor_mode="none",       # "none" | "target" | "iter_mean" (treated as constant; see note below)
#         target_value=None,
#         zero_grad_tol=1e-6,
#         eps_y=1e-12,
#         lgbm_params=None,
#         verbose=True,
#     ):
#         self.rho = float(rho)
#         self.n_bins = int(n_bins)
#         if self.n_bins < 1:
#             raise ValueError("n_bins must be >= 1.")
#         self.ratio_mode = ratio_mode
#         self.anchor_mode = anchor_mode
#         self.target_value = target_value
#         self.zero_grad_tol = float(zero_grad_tol)
#         self.eps_y = float(eps_y)
#         self.verbose = bool(verbose)
#         self.model = lgb.LGBMRegressor(**(lgbm_params or {}))

#     def fit(self, X, y):
#         y = np.asarray(y)
#         self.y_mean_ = float(np.mean(y))

#         # ---- quantile bin edges on y (log-price world) ----
#         qs = np.linspace(0.0, 1.0, self.n_bins + 1)
#         edges = np.quantile(y, qs)

#         # Guard against duplicate edges (e.g., many identical y values)
#         edges = np.unique(edges)
#         if edges.size < 2:
#             # Degenerate: everything in one bin
#             edges = np.array([float(np.min(y)), float(np.max(y))], dtype=float)

#         self.bin_edges_ = edges
#         self.K_ = int(self.bin_edges_.size - 1)

#         self.model.set_params(objective=self.fobj)
#         self.model.fit(X, y)
#         return self

#     def predict(self, X):
#         return self.model.predict(X)

#     def _bin_index(self, y_true):
#         # bins defined by edges; cutpoints are interior edges
#         if self.K_ <= 1:
#             return np.zeros_like(y_true, dtype=int), 1
#         cutpoints = self.bin_edges_[1:-1]  # length K-1
#         bin_idx = np.searchsorted(cutpoints, y_true, side="right").astype(int)  # 0..K-1
#         return bin_idx, self.K_

#     def fobj(self, y_true, y_pred):
#         y_true = np.asarray(y_true)
#         y_pred = np.asarray(y_pred)
#         n = y_pred.size
#         n_f = float(n)

#         # ---- choose r and dr/dy_pred ----
#         if self.ratio_mode in ("div", "ratio"):
#             denom = np.maximum(np.abs(y_true), self.eps_y)
#             r = y_pred / denom
#             dr = 1.0 / denom
#         elif self.ratio_mode == "diff":
#             r = y_pred - y_true
#             dr = np.ones_like(y_pred)
#         else:
#             raise ValueError("ratio_mode must be 'div', 'ratio', or 'diff'.")

#         # ---- optional anchor ----
#         # NOTE: for this penalty, an anchor *does* change r, but we treat the anchor as a constant
#         # (no gradient through anchor), consistent with the diagonal approximations used elsewhere.
#         anchor = 0.0
#         if self.anchor_mode == "none":
#             anchor = 0.0
#         elif self.anchor_mode == "iter_mean":
#             anchor = float(np.mean(r))
#         elif self.anchor_mode == "target":
#             if self.target_value is None:
#                 anchor = 1.0 if self.ratio_mode in ("div", "ratio") else 0.0
#             else:
#                 anchor = float(self.target_value)
#         else:
#             raise ValueError("anchor_mode must be 'none', 'iter_mean', or 'target'.")

#         r_eff = r - anchor

#         # ---- quantile bins on y_true ----
#         bin_idx, K = self._bin_index(y_true)

#         n_k = np.bincount(bin_idx, minlength=K).astype(float)
#         n_k_safe = np.maximum(n_k, 1.0)

#         sum_r_k = np.bincount(bin_idx, weights=r_eff, minlength=K).astype(float)
#         m_k = sum_r_k / n_k_safe

#         m_bar = float(np.mean(r_eff))
#         m_bin = m_k[bin_idx]

#         # ---- Var(E[r|y]) (binned) ----
#         # V = sum_k (n_k/n) (m_k - m_bar)^2
#         # (empty bins contribute 0 because n_k=0)
#         diff_k = (m_k - m_bar)
#         V = float(np.sum((n_k / n_f) * (diff_k ** 2)))

#         pen_value = 0.5 * self.rho * n_f * V

#         # ---- prints ----
#         mse_vec = (y_true - y_pred) ** 2
#         mse_mean = float(np.mean(mse_vec))

#         if self.verbose:
#             model_name = self.__str__().split("(")[0]
#             print(
#                 f"[{model_name}] "
#                 f"Loss: {(mse_mean + pen_value):.6f} | MSE: {mse_mean:.6f} | "
#                 f"Var(E[r|y]): {V:.6e} | Pen: {pen_value:.6f} | K: {K}"
#             )

#         # ---- base MSE grads/hess ----
#         grad_base = 2.0 * (y_pred - y_true)
#         hess_base = 2.0 * np.ones_like(y_pred)

#         # ---- penalty grads/hess (diag Hessian in r-space, chain through r) ----
#         # d penalty / d r_i = rho * (m_bin(i) - m_bar)
#         grad_pen = self.rho * (m_bin - m_bar) * dr

#         # d^2 penalty / d r_i^2 = rho * (1/n_bin - 1/n)
#         inv_n_bin = 1.0 / n_k_safe[bin_idx]
#         hess_pen = self.rho * (inv_n_bin - (1.0 / n_f)) * (dr ** 2)

#         grad = grad_base + grad_pen
#         hess = hess_base + hess_pen

#         grad[np.abs(grad) < self.zero_grad_tol] = self.zero_grad_tol
#         hess[hess < self.zero_grad_tol] = self.zero_grad_tol

#         return grad, hess

#     def __str__(self):
#         return (
#             f"LGBVarCondMeanPenalty(rho={self.rho}, n_bins={self.n_bins}, "
#             f"ratio_mode={self.ratio_mode}, anchor_mode={self.anchor_mode})"
#         )

# # Both Var(E(R|Y)) and E(Var(R|Y))
# class LGBCondMeanVarPenalty:
#     """LightGBM objective:
#         MSE
#       + rho_cov  * Var( E[r | y] )
#       + rho_disp * E[ Var(r | y) ]

#     We approximate conditioning on y by *quantile bins* on y (computed once in fit()).

#     r is chosen by ratio_mode:
#       - "div"  : r = y_pred / max(|y_true|, eps_y)
#       - "diff" : r = y_pred - y_true
#       - "ratio": alias for "div"

#     Definitions (binned):
#       - Let bins B_k partition y. Let m_k = mean_{i in B_k} r_i and m_bar = mean_i r_i.
#       - Vertical term:
#             Var(E[r|y]) ≈ sum_k (n_k/n) * (m_k - m_bar)^2
#       - Horizontal term:
#             E[Var(r|y)] ≈ sum_k (n_k/n) * ( (1/n_k) * sum_{i in B_k} (r_i - m_k)^2 )
#                          = (1/n) * sum_i (r_i - m_{bin(i)})^2

#     Scaling (same style as your covariance penalty):
#       pen_cov  = 0.5 * rho_cov  * n * Var(E[r|y])
#       pen_disp = 0.5 * rho_disp * n * E[Var(r|y)]

#     Grad/Hess (diagonal approximation, chain through r = r(y_pred)):
#       Let k(i)=bin(i), n_k = count in bin, m_k = bin mean, m_bar = global mean.

#       Vertical (between-bin) penalty:
#         d/d r_i  [0.5*rho_cov*n*Var(E[r|y])] = rho_cov * (m_{k(i)} - m_bar)
#         d^2/d r_i^2                          = rho_cov * (1/n_{k(i)} - 1/n)   (>=0)

#       Horizontal (within-bin) penalty:
#         d/d r_i  [0.5*rho_disp*n*EVar] = rho_disp * (r_i - m_{k(i)})
#         d^2/d r_i^2                    = rho_disp * (1 - 1/n_{k(i)})          (>=0)

#       Chain rule:
#         grad_pen_i = (grad_r_i) * dr_i
#         hess_pen_i = (hess_r_i) * (dr_i)^2
#     """

#     def __init__(
#         self,
#         rho_cov=1e-3,             # NEW: vertical penalty weight
#         rho_disp=0.0,             # NEW: horizontal penalty weight
#         n_bins=10,
#         ratio_mode="div",         # "div" | "diff" | "ratio"
#         anchor_mode="none",       # "none" | "target" | "iter_mean" (treated as constant; see note below)
#         target_value=None,
#         zero_grad_tol=1e-6,
#         eps_y=1e-12,
#         lgbm_params=None,
#         verbose=True,
#     ):
#         self.rho_cov = float(rho_cov)
#         self.rho_disp = float(rho_disp)
#         self.n_bins = int(n_bins)
#         if self.n_bins < 1:
#             raise ValueError("n_bins must be >= 1.")
#         self.ratio_mode = ratio_mode
#         self.anchor_mode = anchor_mode
#         self.target_value = target_value
#         self.zero_grad_tol = float(zero_grad_tol)
#         self.eps_y = float(eps_y)
#         self.verbose = bool(verbose)
#         self.model = lgb.LGBMRegressor(**(lgbm_params or {}))

#     def fit(self, X, y):
#         y = np.asarray(y)
#         self.y_mean_ = float(np.mean(y))

#         # ---- quantile bin edges on y ----
#         qs = np.linspace(0.0, 1.0, self.n_bins + 1)
#         edges = np.quantile(y, qs)

#         # Guard against duplicate edges (e.g., many identical y values)
#         edges = np.unique(edges)
#         if edges.size < 2:
#             edges = np.array([float(np.min(y)), float(np.max(y))], dtype=float)

#         self.bin_edges_ = edges
#         self.K_ = int(self.bin_edges_.size - 1)

#         self.model.set_params(objective=self.fobj)
#         self.model.fit(X, y)
#         return self

#     def predict(self, X):
#         return self.model.predict(X)

#     def _bin_index(self, y_true):
#         if self.K_ <= 1:
#             return np.zeros_like(y_true, dtype=int), 1
#         cutpoints = self.bin_edges_[1:-1]  # length K-1
#         bin_idx = np.searchsorted(cutpoints, y_true, side="right").astype(int)  # 0..K-1
#         return bin_idx, self.K_

#     def fobj(self, y_true, y_pred):
#         y_true = np.asarray(y_true)
#         y_pred = np.asarray(y_pred)
#         n = y_pred.size
#         n_f = float(n)

#         # ---- choose r and dr/dy_pred ----
#         if self.ratio_mode in ("div", "ratio"):
#             denom = np.maximum(np.abs(y_true), self.eps_y)
#             r = y_pred / denom
#             dr = 1.0 / denom
#         elif self.ratio_mode == "diff":
#             r = y_pred - y_true
#             dr = np.ones_like(y_pred)
#         else:
#             raise ValueError("ratio_mode must be 'div', 'ratio', or 'diff'.")

#         # ---- optional anchor (treated as constant; no gradient through anchor) ----
#         anchor = 0.0
#         if self.anchor_mode == "none":
#             anchor = 0.0
#         elif self.anchor_mode == "iter_mean":
#             anchor = float(np.mean(r))
#         elif self.anchor_mode == "target":
#             if self.target_value is None:
#                 anchor = 1.0 if self.ratio_mode in ("div", "ratio") else 0.0
#             else:
#                 anchor = float(self.target_value)
#         else:
#             raise ValueError("anchor_mode must be 'none', 'iter_mean', or 'target'.")

#         r_eff = r - anchor

#         # ---- quantile bins on y_true ----
#         bin_idx, K = self._bin_index(y_true)

#         n_k = np.bincount(bin_idx, minlength=K).astype(float)
#         n_k_safe = np.maximum(n_k, 1.0)

#         sum_r_k = np.bincount(bin_idx, weights=r_eff, minlength=K).astype(float)
#         m_k = sum_r_k / n_k_safe

#         m_bar = float(np.mean(r_eff))
#         m_bin = m_k[bin_idx]

#         # ---- vertical: Var(E[r|y]) ----
#         diff_k = (m_k - m_bar)
#         V_cov = float(np.sum((n_k / n_f) * (diff_k ** 2)))

#         # ---- horizontal: E[Var(r|y)] ----
#         # EVar = (1/n) * sum_i (r_i - m_{bin(i)})^2
#         resid_within = (r_eff - m_bin)
#         V_disp = float(np.mean(resid_within ** 2))

#         pen_cov = 0.5 * self.rho_cov * n_f * V_cov
#         pen_disp = 0.5 * self.rho_disp * n_f * V_disp
#         pen_value = pen_cov + pen_disp

#         # ---- prints ----
#         mse_vec = (y_true - y_pred) ** 2
#         mse_mean = float(np.mean(mse_vec))

#         if self.verbose:
#             model_name = self.__str__().split("(")[0]
#             print(
#                 f"[{model_name}] "
#                 f"Loss: {(mse_mean + pen_value):.6f} | MSE: {mse_mean:.6f} | "
#                 f"Var(E[r|y]): {V_cov:.6e} | EVar(r|y): {V_disp:.6e} | "
#                 f"Pen(cov): {pen_cov:.6f} | Pen(disp): {pen_disp:.6f} | K: {K}"
#             )

#         # ---- base MSE grads/hess ----
#         grad_base = 2.0 * (y_pred - y_true)
#         hess_base = 2.0 * np.ones_like(y_pred)

#         # ---- penalty grads/hess in r-space ----
#         # Vertical:
#         grad_r_cov = self.rho_cov * (m_bin - m_bar)
#         inv_n_bin = 1.0 / n_k_safe[bin_idx]
#         hess_r_cov = self.rho_cov * (inv_n_bin - (1.0 / n_f))

#         # Horizontal:
#         grad_r_disp = self.rho_disp * (r_eff - m_bin)
#         hess_r_disp = self.rho_disp * (1.0 - inv_n_bin)

#         grad_pen = (grad_r_cov + grad_r_disp) * dr
#         hess_pen = (hess_r_cov + hess_r_disp) * (dr ** 2)

#         grad = grad_base + grad_pen
#         hess = hess_base + hess_pen

#         grad[np.abs(grad) < self.zero_grad_tol] = self.zero_grad_tol
#         hess[hess < self.zero_grad_tol] = self.zero_grad_tol

#         return grad, hess

#     def __str__(self):
#         return (
#             f"LGBCondMeanVarPenalty(rho_cov={self.rho_cov}, rho_disp={self.rho_disp}, "
#             f"n_bins={self.n_bins}, ratio_mode={self.ratio_mode}, anchor_mode={self.anchor_mode})"
#         )


# # 5) Covariance AND variance of r. It shouldn't make much sense, because min MSE is already seeking a low dispersion in this case
# # V4: With binning and div/diff
# class LGBCovDispPenalty:
#     """LightGBM objective:
#         (optionally binned) MSE
#         + rho_cov  * Cov(r, y)^2   (or only negative direction)
#         + rho_disp * mean_dispersion_loss(u)

#     r is chosen by ratio_mode:
#       - "div"  : r = y_pred / max(|y_true|, eps_y)   (old behavior)
#       - "diff" : r = y_pred - y_true                (log-residual if y is log-price)

#     Dispersion variable u:
#       - if ratio_mode="div"  : u = r - 1
#       - if ratio_mode="diff" : u = r - 0 = r

#     Cov term workflow (cov_mode):
#       - "cov"     : penalize Cov(r,y)^2
#       - "neg_cov" : penalize max(0, -Cov(r,y))^2   (only regressivity direction)

#     Scaling (kept similar to your other models):
#       cov_pen  = 0.5 * rho_cov  * n * cov_term^2
#       disp_pen =       rho_disp * n * mean(ell(u))   (ell(u) ~ 0.5 u^2 near 0)

#     ---- Binning (minimal, optional) ----
#     Instead of CVaR-style weighting, you can "equalize representation" by y-bins.

#     If enabled for a term, we aggregate that term as an *equal average across bins*:

#       - For MSE / Disp:
#           term = (1/K) * sum_b mean_{i in b}(...)    (K = # nonempty bins)
#         This is equivalent to per-sample weights w_i = n / (K * n_b(i)) for that term.

#       - For Cov:
#           cov_b = mean_{i in b}( r_i * (y_i - y_mean) )
#           cov_pen = 0.5 * rho_cov * n * (1/K) * sum_b cov_term_b^2
#         (cov_term_b = cov_b or max(0,-cov_b) depending on cov_mode)

#     Controls:
#       n_bins: int
#       binning: "quantile" or "uniform"
#       bin_mse / bin_cov / bin_disp: bool flags for which terms use bin aggregation
#     """

#     def __init__(
#         self,
#         rho_cov=1e-3,
#         rho_disp=1e-3,
#         ratio_mode="div",          # "div" or "diff"
#         cov_mode="cov",            # "cov" or "neg_cov"
#         disp_mode="l2",            # "l2" or "pseudohuber"
#         huber_delta=0.10,          # only used if disp_mode="pseudohuber"
#         # binning controls
#         n_bins=1,                  # <=1 disables binning
#         binning="quantile",        # "quantile" or "uniform"
#         weight_mode="inv_freq",        # "inv_freq" or "freq" or "none"
#         bin_mse=False,
#         bin_cov=False,
#         bin_disp=False,
#         # numerics
#         zero_grad_tol=1e-6,
#         eps_y=1e-12,
#         eps_delta=1e-12,
#         lgbm_params=None,
#         verbose=True,
#     ):
#         self.rho_cov = float(rho_cov)
#         self.rho_disp = float(rho_disp)

#         self.ratio_mode = str(ratio_mode)
#         self.cov_mode = str(cov_mode)
#         self.disp_mode = str(disp_mode)
#         self.huber_delta = float(huber_delta)

#         self.n_bins = int(n_bins)
#         self.binning = str(binning)
#         self.weight_mode = str(weight_mode)
#         self.bin_mse = bool(bin_mse)
#         self.bin_cov = bool(bin_cov)
#         self.bin_disp = bool(bin_disp)

#         self.zero_grad_tol = float(zero_grad_tol)
#         self.eps_y = float(eps_y)
#         self.eps_delta = float(eps_delta)
#         self.verbose = bool(verbose)

#         self.model = lgb.LGBMRegressor(**(lgbm_params or {}))

#     def fit(self, X, y):
#         y = np.asarray(y)
#         self.y_mean_ = float(np.mean(y))

#         if self.ratio_mode not in ("div", "diff"):
#             raise ValueError(f"ratio_mode must be 'div' or 'diff', got {self.ratio_mode!r}")
#         if self.cov_mode not in ("cov", "neg_cov"):
#             raise ValueError(f"cov_mode must be 'cov' or 'neg_cov', got {self.cov_mode!r}")
#         if self.disp_mode not in ("l2", "pseudohuber"):
#             raise ValueError(f"disp_mode must be 'l2' or 'pseudohuber', got {self.disp_mode!r}")
#         if self.disp_mode == "pseudohuber" and self.huber_delta <= 0:
#             raise ValueError(f"huber_delta must be > 0, got {self.huber_delta}")
#         if self.n_bins < 1:
#             raise ValueError("n_bins must be >= 1")
#         if self.binning not in ("quantile", "uniform"):
#             raise ValueError("binning must be 'quantile' or 'uniform'")

#         # store bin edges (used by fobj). If n_bins<=1, no binning.
#         self.bin_edges_ = None
#         if self.n_bins > 1 and (self.bin_mse or self.bin_cov or self.bin_disp):
#             self.bin_edges_ = self._make_bin_edges(y, self.n_bins, self.binning)

#         self.model.set_params(objective=self.fobj)
#         self.model.fit(X, y)
#         return self

#     def predict(self, X):
#         return self.model.predict(X)

#     # ----------------------------
#     # dispersion loss
#     # ----------------------------
#     def _disp_loss_and_derivs(self, u):
#         """Return (ell(u), ell'(u), ell''(u)) elementwise."""
#         if self.disp_mode == "l2":
#             # ell(u)=0.5u^2 -> ell'(u)=u, ell''(u)=1
#             ell = 0.5 * (u ** 2)
#             ell_p = u
#             ell_pp = np.ones_like(u)
#             return ell, ell_p, ell_pp

#         # pseudo-Huber:
#         # ell(u) = d^2 (sqrt(1+(u/d)^2)-1)
#         # ell'(u)= u / sqrt(1+(u/d)^2)
#         # ell''(u)= 1 / (1+(u/d)^2)^(3/2)
#         d = max(self.huber_delta, self.eps_delta)
#         t = u / d
#         s = np.sqrt(1.0 + t * t)
#         ell = (d * d) * (s - 1.0)
#         ell_p = u / s
#         ell_pp = 1.0 / (s ** 3)
#         return ell, ell_p, ell_pp

#     # ----------------------------
#     # objective
#     # ----------------------------
#     def fobj(self, y_true, y_pred):
#         y_true = np.asarray(y_true)
#         y_pred = np.asarray(y_pred)
#         n = y_pred.size

#         yc = (y_true - self.y_mean_)  # centered y

#         # ---- r and dr/dpred ----
#         if self.ratio_mode == "div":
#             denom = np.maximum(np.abs(y_true), self.eps_y)
#             dr = 1.0 / denom
#             r = y_pred * dr
#             u = r - 1.0
#         else:  # "diff"
#             dr = np.ones_like(y_pred)
#             r = y_pred - y_true
#             u = r  # = r - 0

#         # ---- binning setup (optional) ----
#         use_bins = (self.bin_edges_ is not None)
#         if use_bins:
#             bin_idx, K = self._bin_index(y_true, self.bin_edges_)
#             n_b = np.bincount(bin_idx, minlength=K).astype(float)
#             nonempty = n_b > 0
#             K_eff = int(np.sum(nonempty))
#             if K_eff <= 0:
#                 K_eff = 1
#             n_b_safe = np.maximum(n_b, 1.0)

#             # # per-sample factor: n/(K_eff*n_b(i))  (so each nonempty bin contributes equally)
#             # w_bin = float(n) / (float(K_eff) * n_b_safe[bin_idx])
#             # if self.weight_mode == "freq":
#             #     w_bin = 1/w_bin
#             if self.weight_mode == "inv_freq":
#                 w_raw = 1.0 / n_b_safe[bin_idx]
#             elif self.weight_mode == "freq":
#                 w_raw = n_b_safe[bin_idx]
#             elif self.weight_mode == "none":
#                 w_raw = np.ones_like(y_pred, dtype=float)
#             else:
#                 raise ValueError("weight_mode must be 'inv_freq', 'freq', or 'none'")

#             w_bin = w_raw / float(np.mean(w_raw))  # normalize so mean weight = 1

#         else:
#             bin_idx = None
#             K_eff = 1
#             n_b_safe = None
#             w_bin = np.ones_like(y_pred, dtype=float)

#         # ----------------
#         # MSE term (optionally binned)
#         # ----------------
#         if use_bins and self.bin_mse:
#             w_mse = w_bin
#         else:
#             w_mse = 1.0

#         grad_base = 2.0 * w_mse * (y_pred - y_true)
#         hess_base = 2.0 * w_mse * np.ones_like(y_pred)

#         # ----------------
#         # Cov term (optionally binned)
#         # cov_b = mean_{i in b}(r_i * yc_i)
#         # ----------------
#         s = r * yc  # contribution to covariance

#         if use_bins and self.bin_cov:
#             sum_s_b = np.bincount(bin_idx, weights=s, minlength=K).astype(float)
#             cov_b = sum_s_b / n_b_safe
#             cov_i = cov_b[bin_idx]

#             if self.cov_mode == "cov":
#                 active = np.ones_like(y_pred, dtype=bool)
#             else:  # "neg_cov"
#                 active = (cov_i < 0.0)

#             # d cov_b / d pred_i = (1/n_b) * yc_i * dr_i
#             dcb = (yc * dr) / n_b_safe[bin_idx]

#             # penalty is 0.5*rho_cov*n*(1/K)*sum_b cov_term_b^2
#             # => grad_i = rho_cov*n*(1/K)*cov_b * d cov_b / d pred_i
#             factor = float(n) / float(K_eff)
#             grad_cov = np.where(active, self.rho_cov * factor * cov_i * dcb, 0.0)
#             hess_cov = np.where(active, self.rho_cov * factor * (dcb ** 2), 0.0)

#             # for prints
#             cov_term_print = float(np.mean((cov_b[nonempty] ** 2))) ** 0.5 if np.any(nonempty) else float("nan")
#             pen_cov_value = 0.5 * self.rho_cov * float(n) * float(np.mean((cov_b[nonempty] ** 2))) if np.any(nonempty) else 0.0
#         else:
#             cov = float(np.mean(s))  # global cov since E[yc]=0
#             if self.cov_mode == "cov":
#                 cov_active = True
#             else:
#                 cov_active = (cov < 0.0)

#             # d cov / d pred_i = (1/n) * yc_i * dr_i
#             dc = (yc * dr) / float(n)

#             if cov_active:
#                 grad_cov = self.rho_cov * float(n) * cov * dc
#                 hess_cov = self.rho_cov * float(n) * (dc ** 2)
#             else:
#                 grad_cov = np.zeros_like(y_pred)
#                 hess_cov = np.zeros_like(y_pred)

#             cov_term_print = cov
#             pen_cov_value = 0.5 * self.rho_cov * float(n) * (cov ** 2) if cov_active else 0.0

#         # ----------------
#         # Dispersion term (optionally binned)
#         # pen_disp = rho_disp * n * mean(ell(u))   OR binned mean across bins
#         # ----------------
#         ell_u, ell_p_u, ell_pp_u = self._disp_loss_and_derivs(u)

#         if use_bins and self.bin_disp:
#             w_disp = w_bin  # same equal-bin factor
#         else:
#             w_disp = 1.0

#         # pen_disp = rho_disp * sum (w_disp * ell(u))  (since w_disp sums to n if binned)
#         # u depends on r, and dr/dpred = dr
#         grad_disp = self.rho_disp * w_disp * ell_p_u * dr
#         hess_disp = self.rho_disp * w_disp * ell_pp_u * (dr ** 2)

#         # ----------------
#         # combine
#         # ----------------
#         grad = grad_base + grad_cov + grad_disp
#         hess = hess_base + hess_cov + hess_disp

#         # numerical guards
#         grad[np.abs(grad) < self.zero_grad_tol] = self.zero_grad_tol
#         hess[hess < self.zero_grad_tol] = self.zero_grad_tol

#         # ----------------
#         # logging (simple)
#         # ----------------
#         if self.verbose:
#             mse_mean = float(np.mean((y_true - y_pred) ** 2))
#             # helpful: (r - target)^2 mean for div, r^2 mean for diff
#             u2_mean = float(np.mean(u ** 2))
#             if use_bins and self.bin_disp:
#                 disp_mean = float(np.mean(w_disp * ell_u)) / float(np.mean(w_disp))
#             else:
#                 disp_mean = float(np.mean(ell_u))
#             pen_disp_value = self.rho_disp * float(n) * disp_mean

#             try:
#                 corr = float(np.corrcoef(r, y_true)[0, 1])
#             except Exception:
#                 corr = float("nan")

#             model_name = self.__str__().split("(")[0]
#             bin_info = f" | bins={K_eff}({self.binning}) mse/cov/disp={int(self.bin_mse)}/{int(self.bin_cov)}/{int(self.bin_disp)}" if use_bins else ""
#             print(
#                 f"[{model_name}] "
#                 f"MSE: {mse_mean:.6f} | "
#                 f"CovTerm: {cov_term_print:.6e} | CovMode: {self.cov_mode} | "
#                 f"DispMode: {self.disp_mode} | DispMean: {disp_mean:.6e} | u^2 mean: {u2_mean:.6e} | "
#                 f"PenCov: {pen_cov_value:.6f} | PenDisp: {pen_disp_value:.6f} | "
#                 f"Corr(r,y): {corr:.6f}"
#                 f"{bin_info}"
#             )

#         return grad, hess

#     # ----------------------------
#     # helpers
#     # ----------------------------
#     @staticmethod
#     def _make_bin_edges(y, n_bins, binning):
#         y = np.asarray(y)

#         if n_bins <= 1:
#             lo = float(np.min(y))
#             hi = float(np.max(y))
#             if hi <= lo:
#                 hi = lo + 1e-12
#             return np.array([lo, hi], dtype=float)

#         if binning == "quantile":
#             qs = np.linspace(0.0, 1.0, n_bins + 1)
#             edges = np.quantile(y, qs)
#             edges = np.unique(edges)
#         else:  # "uniform"
#             lo = float(np.min(y))
#             hi = float(np.max(y))
#             if hi <= lo:
#                 hi = lo + 1e-12
#             edges = np.linspace(lo, hi, n_bins + 1)
#             edges = np.unique(edges)

#         if edges.size < 2:
#             lo = float(np.min(y))
#             edges = np.array([lo, lo + 1e-12], dtype=float)

#         return edges.astype(float)

#     @staticmethod
#     def _bin_index(y, edges):
#         edges = np.asarray(edges, dtype=float)
#         K = int(max(edges.size - 1, 1))
#         if K == 1:
#             return np.zeros_like(y, dtype=int), 1
#         idx = np.digitize(y, edges[1:-1], right=False)
#         idx = np.clip(idx, 0, K - 1).astype(int)
#         return idx, K

#     def __str__(self):
#         extra = f", disp={self.disp_mode}"
#         if self.disp_mode == "pseudohuber":
#             extra += f"(d={self.huber_delta})"
#         return (
#             f"LGBCovDispPenalty("
#             f"rho_cov={self.rho_cov}, rho_disp={self.rho_disp}, "
#             f"bins={self.n_bins}, weightmode={self.weight_mode}" #bindisp={self.bin_disp}"
#             # f"ratio={self.ratio_mode}, cov_mode={self.cov_mode}{extra})"
#         )


# # 6) Surrogate of full-distributional independence of r and y

# class LGBBinIndepSurrogatePenalty:
#     """
#     LightGBM objective: MSE + rho * (bin-independence surrogate)

#     Goal: make an error-like quantity r "independent" of y by forcing its *bin-wise means*
#     (across bins of y) to be the same (or equal to a target).

#     Two common choices for r:
#       - ratio_mode="div":  r_i = y_pred_i / max(|y_true_i|, eps_y)   (your old ratio idea)
#       - ratio_mode="diff": r_i = y_pred_i - y_true_i                (log-residual if y is log-price)

#     If y is log-price, and you care about *price-scale* ratios exp(y_pred)/exp(y_true),
#     then using ratio_mode="diff" is usually the right primitive:
#         exp(y_pred)/exp(y_true) = exp(y_pred - y_true) = exp(r)

#     Penalty (mean-matching across y-bins):
#       Let bins be formed on y_true.
#       For each bin b:
#         mu_b = mean_{i in b} r_i
#       Anchor:
#         - anchor_mode="global":  anchor = mu = mean(r)  (enforces mu_b ~ mu across bins)
#         - anchor_mode="target":  anchor = target_value  (enforces mu_b ~ target_value in every bin)

#       weights w_b:
#         - weight_mode="proportional": w_b = n_b / n
#         - weight_mode="uniform":      w_b = 1 / (#nonempty bins)

#       penalty = 0.5 * rho * n * sum_b w_b * (mu_b - anchor)^2

#     Derivatives:
#       Base MSE:
#         grad_base = 2 * (y_pred - y_true)
#         hess_base = 2

#       Let dr_i/dy_pred_i be:
#         - "diff": 1
#         - "div" : 1/denom_i

#       The penalty is quadratic in r, so the Hessian is dense in principle,
#       but we can compute an *exact diagonal* Hessian efficiently.

#     Notes:
#       - This is a *surrogate* for full independence: it only matches first moments across bins.
#       - You can later extend it by also matching variances/quantiles per bin (not included here).
#     """

#     def __init__(
#         self,
#         rho=1e-3,
#         bins=10,                    # int (#quantile bins) or array-like of bin edges
#         ratio_mode="diff",          # "diff" or "div"
#         anchor_mode="target",       # "global" or "target"
#         target_value=None,          # default: 0 for diff, 1 for div
#         weight_mode="proportional", # "proportional" or "uniform"
#         eps_y=1e-12,
#         zero_grad_tol=1e-6,
#         lgbm_params=None,
#         verbose=True,
#     ):
#         self.rho = float(rho)
#         self.bins = bins
#         self.ratio_mode = ratio_mode
#         self.anchor_mode = anchor_mode
#         self.target_value = target_value
#         self.weight_mode = weight_mode
#         self.eps_y = float(eps_y)
#         self.zero_grad_tol = float(zero_grad_tol)
#         self.verbose = bool(verbose)

#         self.model = lgb.LGBMRegressor(**(lgbm_params or {}))
#         self._call_count = 0

#     # ----------------------------
#     # sklearn-style API
#     # ----------------------------
#     def fit(self, X, y):
#         y = np.asarray(y)
#         self.bin_edges_ = self._make_bin_edges(y, self.bins)
#         self.model.set_params(objective=self.fobj)
#         self.model.fit(X, y)
#         return self

#     def predict(self, X):
#         return self.model.predict(X)

#     def __str__(self):
#         return f"LGBBinIndepSurrogatePenalty(rho={self.rho}, mode={self.ratio_mode}, bins={self.bins})" #self._n_bins()})"

#     # ----------------------------
#     # Custom objective for LightGBM
#     # ----------------------------
#     def fobj(self, y_true, y_pred):
#         y_true = np.asarray(y_true)
#         y_pred = np.asarray(y_pred)
#         n = y_pred.size
#         self._call_count += 1

#         # ---- define r and dr/dy_pred ----
#         if self.ratio_mode == "diff":
#             r = y_pred - y_true
#             dr = np.ones_like(y_pred)
#         elif self.ratio_mode == "div":
#             denom = np.maximum(np.abs(y_true), self.eps_y)
#             r = y_pred / denom
#             dr = 1.0 / denom
#         else:
#             raise ValueError("ratio_mode must be 'diff' or 'div'.")

#         # ---- bins on y_true ----
#         bin_idx, K = self._bin_index(y_true, self.bin_edges_)
#         # bin counts and sums
#         n_b = np.bincount(bin_idx, minlength=K).astype(float)
#         sum_r = np.bincount(bin_idx, weights=r, minlength=K).astype(float)

#         # safe counts to avoid division by 0 (empty bins)
#         n_b_safe = np.maximum(n_b, 1.0)
#         mu_b = sum_r / n_b_safe
#         mu = float(np.mean(r))

#         # ---- anchor ----
#         if self.anchor_mode == "global":
#             anchor = mu
#         elif self.anchor_mode == "target":
#             if self.target_value is None:
#                 anchor = 1.0 if self.ratio_mode == "div" else 0.0
#             else:
#                 anchor = float(self.target_value)
#         else:
#             raise ValueError("anchor_mode must be 'global' or 'target'.")

#         d_b = mu_b - anchor  # bin-wise deviations from anchor

#         # ---- bin weights ----
#         w_b = self._bin_weights(n_b, n, mode=self.weight_mode)  # sums to 1 over nonempty bins
#         Wsum = float(np.sum(w_b))

#         # ---- penalty value (for logging) ----
#         pen_value = 0.5 * self.rho * float(n) * float(np.sum(w_b * (d_b ** 2)))

#         # ---- base MSE grads/hess ----
#         grad_base = 2.0 * (y_pred - y_true)
#         hess_base = 2.0 * np.ones_like(y_pred)

#         # ---- penalty grad/hess (diagonal) wrt r ----
#         b_i = bin_idx
#         w_i = w_b[b_i]
#         d_i = d_b[b_i]
#         n_i = n_b_safe[b_i]

#         if self.anchor_mode == "global":
#             # S = sum_b w_b * (mu_b - mu)  (mu is current mean of r)
#             # when anchor=mu, d_b = mu_b - mu, so:
#             S = float(np.sum(w_b * d_b))
#             # grad_r_i = rho*n*( w_{b(i)}*d_{b(i)}/n_{b(i)} - S/n )
#             grad_r = self.rho * float(n) * (w_i * d_i / n_i - S / float(n))

#             # exact diagonal of Hessian in r-space:
#             # diag = w_b0*(1/n_b0 - 1/n)^2 + (Wsum - w_b0)*(1/n^2)
#             diagB2 = w_i * ((1.0 / n_i - 1.0 / float(n)) ** 2) + (Wsum - w_i) * (1.0 / (float(n) ** 2))
#             hess_r = self.rho * float(n) * diagB2
#         else:
#             # anchor is constant target: d_b = mu_b - target
#             # grad_r_i = rho*n * w_{b(i)} * d_{b(i)} / n_{b(i)}
#             grad_r = self.rho * float(n) * (w_i * d_i / n_i)
#             # exact diagonal of Hessian in r-space:
#             # diag = sum_b w_b * (d(mu_b)/dr_i)^2 = w_{b(i)}*(1/n_b)^2
#             hess_r = self.rho * float(n) * (w_i * (1.0 / (n_i ** 2)))

#         # ---- chain rule to y_pred ----
#         grad_pen = grad_r * dr
#         hess_pen = hess_r * (dr ** 2)

#         grad = grad_base + grad_pen
#         hess = hess_base + hess_pen

#         # ---- numerical safety ----
#         grad[np.abs(grad) < self.zero_grad_tol] = self.zero_grad_tol
#         hess[hess < self.zero_grad_tol] = self.zero_grad_tol

#         # ---- optional logging ----
#         if self.verbose:
#             nonempty = n_b > 0
#             max_dev = float(np.max(np.abs(d_b[nonempty]))) if np.any(nonempty) else float("nan")
#             mse_mean = float(np.mean((y_true - y_pred) ** 2))
#             model_name = self.__str__().split("(")[0]
#             print(
#                 f"[{model_name}] "
#                 f"MSE: {mse_mean:.6f} | Pen: {pen_value:.6f} | "
#                 f"max|bin_mean-anchor|: {max_dev:.6e} | "
#                 f"anchor: {anchor:.6f} | K(nonempty): {int(np.sum(nonempty))}"
#             )

#         return grad, hess

#     # ----------------------------
#     # Helpers
#     # ----------------------------
#     def _n_bins(self):
#         edges = getattr(self, "bin_edges_", None)
#         return max(int(edges.size - 1), 1) if edges is not None else None

#     @staticmethod
#     def _make_bin_edges(y, bins):
#         y = np.asarray(y)

#         # bins as explicit edges
#         if hasattr(bins, "__len__") and not isinstance(bins, (str, bytes)):
#             edges = np.asarray(bins, dtype=float)
#             if edges.ndim != 1 or edges.size < 2:
#                 raise ValueError("If bins is array-like, it must be 1D with >= 2 edges.")
#             edges = np.unique(np.sort(edges))
#             if edges.size < 2:
#                 raise ValueError("Bin edges must contain at least two distinct values.")
#             return edges

#         # bins as int -> quantile bins
#         K = int(bins)
#         if K < 1:
#             raise ValueError("If bins is an int, it must be >= 1.")
#         qs = np.linspace(0.0, 1.0, K + 1)
#         edges = np.quantile(y, qs)
#         edges = np.unique(edges)
#         if edges.size < 2:
#             # degenerate case: all y are equal
#             edges = np.array([float(np.min(y)), float(np.max(y) + 1e-12)], dtype=float)
#         return edges

#     @staticmethod
#     def _bin_index(y, edges):
#         """
#         Map each y to a bin index in [0, K-1], where K = len(edges)-1.
#         """
#         edges = np.asarray(edges, dtype=float)
#         K = int(max(edges.size - 1, 1))
#         if K == 1:
#             return np.zeros_like(y, dtype=int), 1
#         # internal cut points exclude the ends
#         idx = np.digitize(y, edges[1:-1], right=False)
#         idx = np.clip(idx, 0, K - 1)
#         return idx.astype(int), K

#     @staticmethod
#     def _bin_weights(n_b, n, mode="proportional"):
#         """
#         Return weights w_b that sum to 1 over nonempty bins.
#         """
#         n_b = np.asarray(n_b, dtype=float)
#         nonempty = n_b > 0

#         w = np.zeros_like(n_b)
#         if not np.any(nonempty):
#             return w

#         if mode == "proportional":
#             w[nonempty] = n_b[nonempty] / float(n)
#             # sums to 1 automatically if bins cover all samples
#         elif mode == "uniform":
#             w[nonempty] = 1.0 / float(np.sum(nonempty))
#         else:
#             raise ValueError("weight_mode must be 'proportional' or 'uniform'.")
#         # normalize defensively
#         s = float(np.sum(w))
#         if s > 0:
#             w /= s
#         return w










