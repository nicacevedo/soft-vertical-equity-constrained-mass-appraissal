"""
Optimize convex stacking weights from CV artifacts.

The optimizer treats fold-level metrics as linear in the weights:
  metric_ensemble(fold) ≈ Σ_j w_j * metric_j(fold)

This approximation makes the problem convex and very easy to edit.
For final evaluation on the held-out test set, use `analyze_results.py`,
which can compute *true* stacked test metrics from per-row predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import cvxpy as cp
except ImportError as e:  # pragma: no cover
    raise ImportError("`optimize_stacking_weights.py` requires cvxpy. Install via `pip install cvxpy`.") from e

from utils.motivation_utils import IAAO_PRB_RANGE, IAAO_PRD_RANGE, IAAO_VEI_RANGE, _compute_extended_metrics


def _load_runs_df(*, result_root: str, data_id: str, split_id: str) -> pd.DataFrame:
    runs_dir = Path(result_root) / "runs" / f"data_id={data_id}" / f"split_id={split_id}"
    if not runs_dir.exists():
        raise FileNotFoundError(f"CV runs directory not found: {runs_dir}")
    paths = sorted(runs_dir.rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No run parquet files found under: {runs_dir}")
    dfs = [pd.read_parquet(p) for p in paths]
    return pd.concat(dfs, ignore_index=True)


def _load_predictions_df(*, result_root: str, data_id: str, split_id: str) -> pd.DataFrame:
    preds_dir = Path(result_root) / "predictions" / f"data_id={data_id}" / f"split_id={split_id}"
    if not preds_dir.exists():
        raise FileNotFoundError(f"Predictions directory not found: {preds_dir}")
    paths = sorted(preds_dir.rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No prediction parquet files found under: {preds_dir}")
    dfs = [pd.read_parquet(p) for p in paths]
    out = pd.concat(dfs, ignore_index=True)
    for c in ["run_id", "row_id", "y_true", "y_pred"]:
        if c not in out.columns:
            raise KeyError(f"Missing required prediction column: {c}")
    return out


def _align_complete_grid(
    df: pd.DataFrame,
    *,
    required_metrics: List[str],
    fold_col: str = "fold_id",
    model_col: str = "config_id",
) -> Tuple[pd.DataFrame, List[int], List[str]]:
    """
    Filter to the largest fully-observed fold×model grid across required metrics.
    """
    base_cols = [fold_col, model_col]
    for c in base_cols + required_metrics:
        if c not in df.columns:
            raise KeyError(f"Missing required column in runs_df: {c}")

    keep_cols = base_cols + required_metrics + ["model_name"]
    if "run_id" in df.columns:
        keep_cols.append("run_id")
    dfx = df.loc[:, keep_cols].copy()
    dfx = dfx.dropna(subset=required_metrics)

    # Drop duplicate rows per (fold, model) before building the presence grid.
    dfx = dfx.drop_duplicates(subset=base_cols, keep="first")

    # Build pivot presence grid: rows=folds, cols=models, value=1 if present.
    present = (
        dfx.assign(_present=1)
        .pivot_table(index=fold_col, columns=model_col, values="_present", aggfunc="max", fill_value=0)
    )

    # Keep only folds that have all models, and models that have all folds.
    folds_keep = present.index[present.sum(axis=1) == present.shape[1]].tolist()
    models_keep = present.columns[present.sum(axis=0) == present.shape[0]].tolist()

    # If nothing survives (e.g., no config ran across every fold), fall back.
    if len(folds_keep) == 0 or len(models_keep) == 0:
        folds_keep = sorted(dfx[fold_col].unique().tolist())
        models_keep = sorted(dfx[model_col].unique().tolist())

    dfx = dfx[dfx[fold_col].isin(folds_keep) & dfx[model_col].isin(models_keep)].copy()
    return dfx, folds_keep, models_keep


def _pivot_metric(df: pd.DataFrame, *, metric: str, folds: List[int], models: List[str]) -> np.ndarray:
    p = df.pivot_table(index="fold_id", columns="config_id", values=metric, aggfunc="first")
    p = p.reindex(index=folds, columns=models)
    if p.isna().any().any():
        missing = int(p.isna().sum().sum())
        raise ValueError(f"Missing values after alignment for metric '{metric}': {missing}")
    return p.to_numpy(dtype=float)


def _build_utopia_solution(
    *,
    runs_df: pd.DataFrame,
    folds: List[int],
    models: List[str],
    solver: Optional[str],
    solver_verbose: bool,
    aggregation: str,
    cvar_alpha: float,
    accuracy_weight: float,
    fairness_weight: float,
) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, Any]]:
    """
    Solve simplex weights for closeness to fold-wise utopia after ideal/nadir normalization.

    Accuracy objective (maximize): OOS R2 (fallback R2)
      a_tilde = (a_star - a_cfg) / (a_star - a_nad)
    Fairness objective (minimize): abs(PRD - 1)
      u_tilde = (u_cfg - u_star) / (u_nad - u_star)

    Aggregation options (all convex with pre-normalized constants):
      - average   : mean_s [ wa * p_s(x) + wu * q_s(x) ]
      - chebyshev : mean_s max(wa * p_s(x), wu * q_s(x))
      - cvar      : CVaR_alpha over z_s(x) := wa * p_s(x) + wu * q_s(x)
      - euclidean : mean_s || [sqrt(wa) p_s(x), sqrt(wu) q_s(x)] ||_2

    Here p_s(x), q_s(x) are affine in x, where x lies on the simplex.
    """
    req = ["OOS R2", "R2", "PRD", "PRB", "VEI"]
    aggregation = str(aggregation).lower().strip()
    if aggregation not in {"average", "chebyshev", "cvar", "euclidean"}:
        raise ValueError("aggregation must be one of: average, chebyshev, cvar, euclidean")
    alpha = float(cvar_alpha)
    if aggregation == "cvar" and not (0.0 < alpha < 1.0):
        raise ValueError("utopia cvar_alpha must be in (0, 1) when aggregation=cvar.")
    wa = float(accuracy_weight)
    wu = float(fairness_weight)
    if wa < 0.0 or wu < 0.0 or (wa + wu) <= 0.0:
        raise ValueError("accuracy_weight and fairness_weight must be >=0 and not both zero.")
    aligned_df, _, _ = _align_complete_grid(runs_df, required_metrics=req)
    aligned_df = aligned_df.copy()
    aligned_df["config_id"] = aligned_df["config_id"].astype(str)
    keep_models = [str(m) for m in models]
    keep_folds = [int(f) for f in folds]
    aligned_df = aligned_df[
        aligned_df["config_id"].isin(keep_models) & aligned_df["fold_id"].isin(keep_folds)
    ].copy()
    if aligned_df.empty:
        raise RuntimeError("No aligned rows available to build utopia solution.")

    acc_col = "OOS R2" if "OOS R2" in aligned_df.columns else "R2"
    acc = _pivot_metric(aligned_df, metric=acc_col, folds=keep_folds, models=keep_models)
    prd = _pivot_metric(aligned_df, metric="PRD", folds=keep_folds, models=keep_models)
    # IMPORTANT: use per-config absolute deviation as constants before optimization
    # so q_s(x) remains affine and DCP-safe.
    unfair = np.abs(prd - 1.0)

    eps = 1e-12
    a_star = np.max(acc, axis=1, keepdims=True)
    a_nad = np.min(acc, axis=1, keepdims=True)
    u_star = np.min(unfair, axis=1, keepdims=True)
    u_nad = np.max(unfair, axis=1, keepdims=True)
    a_den = np.maximum(a_star - a_nad, eps)
    u_den = np.maximum(u_nad - u_star, eps)
    a_tilde = (a_star - acc) / a_den
    u_tilde = (unfair - u_star) / u_den
    n_models = len(keep_models)
    n_folds = len(keep_folds)

    w = cp.Variable(n_models, nonneg=True)
    constraints = [cp.sum(w) == 1]

    acc_ens = acc @ w
    unfair_ens = unfair @ w

    a_star_v = a_star.reshape(-1)
    u_star_v = u_star.reshape(-1)
    a_den_v = a_den.reshape(-1)
    u_den_v = u_den.reshape(-1)
    a_tilde_ens = cp.multiply(1.0 / a_den_v, a_star_v - acc_ens)
    u_tilde_ens = cp.multiply(1.0 / u_den_v, unfair_ens - u_star_v)

    z = wa * a_tilde_ens + wu * u_tilde_ens
    if aggregation == "average":
        objective_expr = cp.sum(z) / float(max(1, n_folds))
    elif aggregation == "chebyshev":
        d = cp.Variable(n_folds, nonneg=True)
        constraints += [d >= wa * a_tilde_ens, d >= wu * u_tilde_ens]
        objective_expr = cp.sum(d) / float(max(1, n_folds))
    elif aggregation == "cvar":
        eta = cp.Variable()
        u = cp.Variable(n_folds, nonneg=True)
        constraints += [u >= z - eta]
        objective_expr = eta + (1.0 / ((1.0 - alpha) * float(max(1, n_folds)))) * cp.sum(u)
    else:  # euclidean
        terms: List[Any] = []
        s_wa = float(np.sqrt(max(wa, 0.0)))
        s_wu = float(np.sqrt(max(wu, 0.0)))
        for i in range(n_folds):
            terms.append(cp.norm(cp.hstack([s_wa * a_tilde_ens[i], s_wu * u_tilde_ens[i]]), 2))
        objective_expr = cp.sum(cp.hstack(terms)) / float(max(1, n_folds))
    prob = cp.Problem(cp.Minimize(objective_expr), constraints)

    preferred = _choose_solver(solver)
    installed = set(cp.installed_solvers())
    fallback_chain = [s for s in ["ECOS", "SCS", "OSQP", "CLARABEL"] if s != preferred and s in installed]
    solvers_to_try = [preferred] + fallback_chain
    solved = False
    last_exc: Optional[Exception] = None
    solver_name = preferred
    for candidate in solvers_to_try:
        try:
            prob.solve(solver=candidate, verbose=bool(solver_verbose))
            if prob.status not in (None, "infeasible", "infeasible_inaccurate", "unbounded") and prob.value is not None:
                solver_name = candidate
                solved = True
                last_exc = None
                break
        except Exception as exc:
            last_exc = exc
    if not solved:
        if last_exc is not None:
            raise RuntimeError(f"Utopia optimization failed across solvers {solvers_to_try}. Last error: {last_exc}") from last_exc
        raise RuntimeError(f"Utopia optimization not solved. tried={solvers_to_try} status={prob.status} value={prob.value}")

    weights = np.asarray(w.value, dtype=float).reshape(-1)
    if not np.all(np.isfinite(weights)):
        raise RuntimeError(f"Utopia optimization returned non-finite weights. status={prob.status}")
    if float(np.sum(weights)) <= 0:
        raise RuntimeError(f"Utopia optimization returned zero-sum weights. status={prob.status}")
    weights = np.maximum(weights, 0.0)
    weights = weights / np.sum(weights)

    r2 = _pivot_metric(aligned_df, metric="R2", folds=keep_folds, models=keep_models)
    prb = _pivot_metric(aligned_df, metric="PRB", folds=keep_folds, models=keep_models)
    vei = _pivot_metric(aligned_df, metric="VEI", folds=keep_folds, models=keep_models)
    cod = _pivot_metric(aligned_df, metric="COD", folds=keep_folds, models=keep_models) if "COD" in aligned_df.columns else None
    corr = _pivot_metric(aligned_df, metric="Corr(r,price)", folds=keep_folds, models=keep_models) if "Corr(r,price)" in aligned_df.columns else None

    oos_r2_ens = acc @ weights if acc_col == "OOS R2" else np.full(n_folds, np.nan, dtype=float)
    r2_ens = r2 @ weights
    prd_ens_np = prd @ weights
    prb_ens = prb @ weights
    vei_ens = vei @ weights
    cod_ens = cod @ weights if cod is not None else np.full(n_folds, np.nan, dtype=float)
    corr_ens = corr @ weights if corr is not None else np.full(n_folds, np.nan, dtype=float)

    if acc_col != "OOS R2":
        oos_r2_ens = r2_ens.copy()

    fold_df = pd.DataFrame(
        {
            "fold_id": np.asarray(keep_folds, dtype=int),
            "objective_mode": "utopia",
            "OOS R2_ensemble": oos_r2_ens,
            "R2_ensemble": r2_ens,
            "PRD_ensemble": prd_ens_np,
            "PRB_ensemble": prb_ens,
            "VEI_ensemble": vei_ens,
            "COD_ensemble": cod_ens,
            "Corr(r,price)_ensemble": corr_ens,
        }
    )

    # Report realized normalized distance under the optimized weights.
    a_tilde_realized = (a_star_v - (acc @ weights)) / a_den_v
    u_tilde_realized = ((unfair @ weights) - u_star_v) / u_den_v
    mean_dist = float(np.mean(np.sqrt(np.maximum(0.0, a_tilde_realized) ** 2 + np.maximum(0.0, u_tilde_realized) ** 2)))

    return weights, fold_df, {
        "utopia_accuracy_metric": str(acc_col),
        "utopia_fairness_metric": "abs(PRD-1)",
        "utopia_aggregation": str(aggregation),
        "utopia_distance_metric": "mean_l2(a_tilde, u_tilde) (diagnostic)",
        "utopia_accuracy_weight": float(wa),
        "utopia_fairness_weight": float(wu),
        "utopia_cvar_alpha": float(alpha),
        "objective_value": float(prob.value) if prob.value is not None else None,
        "status": str(prob.status),
        "solver": str(solver_name),
        "selected_mean_distance": float(mean_dist),
        "n_models_considered": int(len(keep_models)),
        "n_folds_considered": int(len(keep_folds)),
    }


def _cvar_of_values(values: np.ndarray, alpha: float) -> float:
    vals = np.asarray(values, dtype=float).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError("alpha must be in (0,1) for CVaR.")
    k = int(np.ceil((1.0 - float(alpha)) * vals.size))
    k = max(1, min(int(vals.size), k))
    worst = np.sort(vals)[-k:]
    return float(np.mean(worst))


def _build_single_utopia_solution(
    *,
    runs_df: pd.DataFrame,
    folds: List[int],
    models: List[str],
    aggregation: str,
    cvar_alpha: float,
    accuracy_weight: float,
    fairness_weight: float,
) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, Any]]:
    """
    Integer/simplex-vertex counterpart of utopia:
    pick one config_id that minimizes the chosen normalized utopia aggregation.
    """
    req = ["OOS R2", "R2", "PRD", "PRB", "VEI"]
    aggregation = str(aggregation).lower().strip()
    if aggregation not in {"average", "chebyshev", "cvar", "euclidean"}:
        raise ValueError("aggregation must be one of: average, chebyshev, cvar, euclidean")
    alpha = float(cvar_alpha)
    wa = float(accuracy_weight)
    wu = float(fairness_weight)
    if wa < 0.0 or wu < 0.0 or (wa + wu) <= 0.0:
        raise ValueError("accuracy_weight and fairness_weight must be >=0 and not both zero.")

    aligned_df, _, _ = _align_complete_grid(runs_df, required_metrics=req)
    aligned_df = aligned_df.copy()
    aligned_df["config_id"] = aligned_df["config_id"].astype(str)
    keep_models = [str(m) for m in models]
    keep_folds = [int(f) for f in folds]
    aligned_df = aligned_df[
        aligned_df["config_id"].isin(keep_models) & aligned_df["fold_id"].isin(keep_folds)
    ].copy()
    if aligned_df.empty:
        raise RuntimeError("No aligned rows available to build single utopia solution.")

    acc_col = "OOS R2" if "OOS R2" in aligned_df.columns else "R2"
    acc = _pivot_metric(aligned_df, metric=acc_col, folds=keep_folds, models=keep_models)
    prd = _pivot_metric(aligned_df, metric="PRD", folds=keep_folds, models=keep_models)
    unfair = np.abs(prd - 1.0)
    eps = 1e-12
    a_star = np.max(acc, axis=1, keepdims=True)
    a_nad = np.min(acc, axis=1, keepdims=True)
    u_star = np.min(unfair, axis=1, keepdims=True)
    u_nad = np.max(unfair, axis=1, keepdims=True)
    a_den = np.maximum(a_star - a_nad, eps)
    u_den = np.maximum(u_nad - u_star, eps)
    p = (a_star - acc) / a_den
    q = (unfair - u_star) / u_den
    z = wa * p + wu * q

    n_models = len(keep_models)
    best_idx = -1
    best_obj = None
    for j in range(n_models):
        if aggregation == "average":
            obj_j = float(np.mean(z[:, j]))
        elif aggregation == "chebyshev":
            d = np.maximum(wa * p[:, j], wu * q[:, j])
            obj_j = float(np.mean(d))
        elif aggregation == "cvar":
            obj_j = _cvar_of_values(z[:, j], alpha)
        else:
            dist = np.sqrt(np.maximum(0.0, wa * (p[:, j] ** 2) + wu * (q[:, j] ** 2)))
            obj_j = float(np.mean(dist))
        if best_obj is None or (np.isfinite(obj_j) and obj_j < float(best_obj)):
            best_obj = obj_j
            best_idx = int(j)
    if best_idx < 0:
        raise RuntimeError("Could not select a single utopia model.")

    w = np.zeros(n_models, dtype=float)
    w[best_idx] = 1.0
    r2 = _pivot_metric(aligned_df, metric="R2", folds=keep_folds, models=keep_models)
    prb = _pivot_metric(aligned_df, metric="PRB", folds=keep_folds, models=keep_models)
    vei = _pivot_metric(aligned_df, metric="VEI", folds=keep_folds, models=keep_models)
    cod = _pivot_metric(aligned_df, metric="COD", folds=keep_folds, models=keep_models) if "COD" in aligned_df.columns else None
    corr = _pivot_metric(aligned_df, metric="Corr(r,price)", folds=keep_folds, models=keep_models) if "Corr(r,price)" in aligned_df.columns else None
    oos_r2_ens = acc @ w if acc_col == "OOS R2" else np.full(len(keep_folds), np.nan, dtype=float)
    r2_ens = r2 @ w
    prd_ens = prd @ w
    prb_ens = prb @ w
    vei_ens = vei @ w
    cod_ens = cod @ w if cod is not None else np.full(len(keep_folds), np.nan, dtype=float)
    corr_ens = corr @ w if corr is not None else np.full(len(keep_folds), np.nan, dtype=float)
    if acc_col != "OOS R2":
        oos_r2_ens = r2_ens.copy()
    fold_df = pd.DataFrame(
        {
            "fold_id": np.asarray(keep_folds, dtype=int),
            "objective_mode": "single_utopia",
            "OOS R2_ensemble": oos_r2_ens,
            "R2_ensemble": r2_ens,
            "PRD_ensemble": prd_ens,
            "PRB_ensemble": prb_ens,
            "VEI_ensemble": vei_ens,
            "COD_ensemble": cod_ens,
            "Corr(r,price)_ensemble": corr_ens,
        }
    )
    return w, fold_df, {
        "utopia_aggregation": str(aggregation),
        "utopia_cvar_alpha": float(alpha),
        "utopia_accuracy_weight": float(wa),
        "utopia_fairness_weight": float(wu),
        "selected_config_id": str(keep_models[best_idx]),
        "objective_value": float(best_obj) if best_obj is not None else None,
    }


def _build_single_requested_solution(
    *,
    mode: str,
    folds: List[int],
    models: List[str],
    fold_panels: Dict[int, Tuple[np.ndarray, np.ndarray]],
    time_decay_gamma: float,
    cvar_alpha: float,
    prd_constraint_aggregation: str,
    prd_constraint_cvar_alpha: float,
    prd_feasibility_mode: str,
    prd_gap_penalty: float,
    use_bootstrap_scenarios: bool,
    n_bootstrap_scenarios: int,
    bootstrap_seed: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Integer/simplex-vertex counterpart of PRD-SOCP:
    evaluate all one-hot selections and keep the best objective under the same
    PRD aggregation + feasibility mode semantics.
    """
    if mode not in {"average", "worst", "time_decay", "cvar"}:
        raise ValueError("mode must be one of: average, worst, time_decay, cvar")
    prd_constraint_aggregation = str(prd_constraint_aggregation)
    if prd_constraint_aggregation not in {"worst", "average", "cvar"}:
        raise ValueError("prd_constraint_aggregation must be one of: worst, average, cvar")
    prd_feasibility_mode = str(prd_feasibility_mode)
    if prd_feasibility_mode not in {"priced_gap", "hard"}:
        raise ValueError("prd_feasibility_mode must be one of: priced_gap, hard")

    rng = np.random.default_rng(int(bootstrap_seed))
    n_models = len(models)
    n_folds = len(folds)
    loss_vals = np.full((n_folds, n_models), np.nan, dtype=float)
    low_vals = np.full((n_folds, n_models), np.nan, dtype=float)
    high_vals = np.full((n_folds, n_models), np.nan, dtype=float)
    bootstrap_losses: List[np.ndarray] = []
    bootstrap_lows: List[np.ndarray] = []
    bootstrap_highs: List[np.ndarray] = []
    bootstrap_fold_idx: List[int] = []
    prd_lo, prd_hi = float(IAAO_PRD_RANGE[0]), float(IAAO_PRD_RANGE[1])

    for fi, f in enumerate(folds):
        P, y = fold_panels[int(f)]
        y_pos = np.maximum(y, 1e-9)
        fold_scale = max(float(np.median(y_pos)), 1.0)
        Pn = P / fold_scale
        y_norm = y_pos / fold_scale
        a = np.sum(Pn / y_norm[:, None], axis=0)
        b = np.sum(Pn, axis=0)
        sy = float(np.sum(y_norm))
        low_vals[fi, :] = prd_lo * b - sy * a
        high_vals[fi, :] = sy * a - prd_hi * b
        ratio_res = (Pn / y_norm[:, None]) - 1.0
        loss_vals[fi, :] = np.mean(ratio_res ** 2, axis=0)

        if use_bootstrap_scenarios and int(n_bootstrap_scenarios) > 0:
            n = int(y_norm.shape[0])
            for _ in range(int(n_bootstrap_scenarios)):
                idx = rng.integers(0, n, size=n)
                Pb = Pn[idx, :]
                yb = np.maximum(y_norm[idx], 1e-9)
                ab = np.sum(Pb / yb[:, None], axis=0)
                bb = np.sum(Pb, axis=0)
                syb = float(np.sum(yb))
                bootstrap_lows.append(prd_lo * bb - syb * ab)
                bootstrap_highs.append(syb * ab - prd_hi * bb)
                bootstrap_losses.append(np.mean(((Pb / yb[:, None]) - 1.0) ** 2, axis=0))
                bootstrap_fold_idx.append(fi)

    gamma = float(time_decay_gamma)
    if mode == "time_decay" and not (0.0 < gamma <= 1.0):
        raise ValueError("time_decay_gamma must be in (0, 1].")
    obj_alpha = float(cvar_alpha)
    prd_alpha = float(prd_constraint_cvar_alpha)

    best_idx = -1
    best_obj = None
    best_gap = 0.0
    best_prd_mode = "hard"
    for j in range(n_models):
        scen_losses = [float(v) for v in loss_vals[:, j].tolist() if np.isfinite(v)]
        scen_low = [float(v) for v in low_vals[:, j].tolist() if np.isfinite(v)]
        scen_high = [float(v) for v in high_vals[:, j].tolist() if np.isfinite(v)]
        if bootstrap_losses:
            for k in range(len(bootstrap_losses)):
                lv = float(bootstrap_losses[k][j])
                lo = float(bootstrap_lows[k][j])
                hi = float(bootstrap_highs[k][j])
                if np.isfinite(lv):
                    scen_losses.append(lv)
                if np.isfinite(lo):
                    scen_low.append(lo)
                if np.isfinite(hi):
                    scen_high.append(hi)
        if not scen_losses or not scen_low or not scen_high:
            continue

        if mode == "average":
            obj_core = float(np.mean(scen_losses))
        elif mode == "worst":
            obj_core = float(np.max(scen_losses))
        elif mode == "time_decay":
            fw = np.asarray([gamma ** (n_folds - 1 - i) for i in range(n_folds)], dtype=float)
            fw = fw / np.sum(fw)
            weighted_losses = [float(fw[i] * loss_vals[i, j]) for i in range(n_folds)]
            if bootstrap_losses:
                reps = int(n_bootstrap_scenarios) if use_bootstrap_scenarios else 0
                for k, fi in enumerate(bootstrap_fold_idx):
                    weighted_losses.append(float(fw[int(fi)] * bootstrap_losses[k][j] / float(max(1, reps))))
            obj_core = float(np.sum(weighted_losses))
        else:
            obj_core = _cvar_of_values(np.asarray(scen_losses, dtype=float), obj_alpha)

        if prd_constraint_aggregation == "worst":
            low_agg = float(np.max(scen_low))
            high_agg = float(np.max(scen_high))
        elif prd_constraint_aggregation == "average":
            low_agg = float(np.mean(scen_low))
            high_agg = float(np.mean(scen_high))
        else:
            low_agg = _cvar_of_values(np.asarray(scen_low, dtype=float), prd_alpha)
            high_agg = _cvar_of_values(np.asarray(scen_high, dtype=float), prd_alpha)

        gap_needed = float(max(0.0, low_agg, high_agg))
        if prd_feasibility_mode == "hard":
            if gap_needed > 0.0:
                continue
            obj_total = obj_core
            prd_mode_j = "hard"
        else:
            obj_total = float(obj_core + float(prd_gap_penalty) * gap_needed)
            prd_mode_j = "priced_gap"

        if best_obj is None or (np.isfinite(obj_total) and obj_total < float(best_obj)):
            best_obj = obj_total
            best_idx = int(j)
            best_gap = float(gap_needed)
            best_prd_mode = str(prd_mode_j)

    if best_idx < 0:
        raise RuntimeError("No feasible single-model solution found under requested constraints/objective.")
    w = np.zeros(n_models, dtype=float)
    w[best_idx] = 1.0
    return w, {
        "objective_mode": str(mode),
        "prd_constraint_mode": str(best_prd_mode),
        "prd_feasibility_mode": str(prd_feasibility_mode),
        "prd_gap_penalty": float(prd_gap_penalty),
        "prd_gap_value": float(best_gap),
        "prd_constraint_aggregation": str(prd_constraint_aggregation),
        "prd_constraint_cvar_alpha": float(prd_alpha),
        "objective_value": float(best_obj) if best_obj is not None else None,
        "selected_config_id": str(models[best_idx]),
    }


def _choose_solver(preferred: str | None = None) -> str:
    installed = set(cp.installed_solvers())
    if preferred is not None and preferred.upper() in installed:
        return preferred.upper()
    for s in ["GUROBI", "ECOS", "OSQP", "SCS", "CLARABEL"]:
        if s in installed:
            return s
    raise RuntimeError("No cvxpy solvers installed. Install one of: ecos, scs, osqp, clarabel, mosek.")


def _build_fold_prediction_panels(
    *,
    runs_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    accuracy_metric: str,
    max_models: Optional[int],
) -> Tuple[List[int], List[str], Dict[int, Tuple[np.ndarray, np.ndarray]], pd.DataFrame]:
    """
    Build per-fold prediction matrices P_f (n_f x M) and targets y_f (n_f,)
    aligned across a common model set.
    Returns:
      folds, models, fold_panels, meta_df(config_id->model_name)
    """
    req = [str(accuracy_metric), "R2", "PRD", "PRB", "VEI"]
    aligned_df, folds, models = _align_complete_grid(runs_df, required_metrics=req)

    # Optional pruning by mean accuracy for scalability.
    if max_models is not None and int(max_models) > 0 and len(models) > int(max_models):
        p_acc = aligned_df.pivot_table(index="fold_id", columns="config_id", values=str(accuracy_metric), aggfunc="first")
        mean_acc = p_acc.mean(axis=0).sort_values(ascending=False)
        models = mean_acc.index[: int(max_models)].tolist()
        aligned_df = aligned_df[aligned_df["config_id"].isin(models)].copy()
        aligned_df, folds, models = _align_complete_grid(aligned_df, required_metrics=req)

    aligned_df = aligned_df.copy()
    aligned_df["config_id"] = aligned_df["config_id"].astype(str)
    if "run_id" not in aligned_df.columns:
        raise KeyError(
            "Missing 'run_id' after alignment. "
            "Ensure run artifacts include run_id and alignment keeps it."
        )
    aligned_df["run_id"] = aligned_df["run_id"].astype(str)
    preds_df = preds_df.copy()
    preds_df["run_id"] = preds_df["run_id"].astype(str)

    run_map = aligned_df.loc[:, ["fold_id", "config_id", "run_id", "model_name"]].drop_duplicates()
    pred_by_run = {str(rid): g.copy() for rid, g in preds_df.groupby("run_id", sort=False)}
    fold_panels: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    for f in sorted([int(x) for x in folds]):
        sub = run_map[run_map["fold_id"] == f].copy()
        sub = sub[sub["config_id"].isin(models)].copy()
        if sub.empty:
            continue

        run_ids = sub["run_id"].tolist()
        # Ensure all required run IDs have predictions.
        if any(rid not in pred_by_run for rid in run_ids):
            continue

        # Intersect row IDs across all models for this fold.
        row_sets = [set(pred_by_run[rid]["row_id"].astype(int).tolist()) for rid in run_ids]
        common_rows = sorted(set.intersection(*row_sets)) if row_sets else []
        if len(common_rows) < 2:
            continue

        # Build y and P aligned by common row order.
        cfg_to_run = dict(zip(sub["config_id"].astype(str), sub["run_id"].astype(str)))
        y_ref = pred_by_run[cfg_to_run[models[0]]].set_index("row_id").loc[common_rows, "y_true"].to_numpy(dtype=float)
        if not np.all(np.isfinite(y_ref)):
            continue
        y_ref = np.maximum(y_ref, 1e-9)

        cols: List[np.ndarray] = []
        ok = True
        for cfg in models:
            rid = cfg_to_run.get(str(cfg))
            if rid is None:
                ok = False
                break
            p = pred_by_run[rid].set_index("row_id").loc[common_rows, "y_pred"].to_numpy(dtype=float)
            if not np.all(np.isfinite(p)):
                ok = False
                break
            cols.append(np.maximum(p, 1e-9))
        if not ok:
            continue

        P = np.column_stack(cols)
        fold_panels[f] = (P, y_ref)

    folds_final = sorted(fold_panels.keys())
    if not folds_final:
        raise RuntimeError("Could not build any complete fold panels from predictions.")

    # Keep only models present for all retained folds.
    # (By construction with _align_complete_grid and common-row intersection, this should hold.)
    meta_df = run_map.loc[:, ["config_id", "model_name"]].drop_duplicates("config_id").copy()
    meta_df["config_id"] = meta_df["config_id"].astype(str)
    meta_df = meta_df[meta_df["config_id"].isin(models)].copy()
    return folds_final, [str(m) for m in models], fold_panels, meta_df


def _solve_prd_socp_for_mode(
    *,
    mode: str,
    folds: List[int],
    models: List[str],
    fold_panels: Dict[int, Tuple[np.ndarray, np.ndarray]],
    solver: Optional[str],
    solver_verbose: bool,
    time_decay_gamma: float,
    cvar_alpha: float,
    prd_constraint_aggregation: str,
    prd_constraint_cvar_alpha: float,
    prd_feasibility_mode: str,
    prd_gap_penalty: float,
    use_bootstrap_scenarios: bool,
    n_bootstrap_scenarios: int,
    bootstrap_seed: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Solve PRD-constrained convex stacking with one objective mode.
    """
    if mode not in {"average", "worst", "time_decay", "cvar"}:
        raise ValueError("mode must be one of: average, worst, time_decay, cvar")

    n_models = len(models)
    w = cp.Variable(n_models, nonneg=True)
    simplex_constraints = [cp.sum(w) == 1]
    denom_constraints: List[Any] = []
    objective_link_constraints: List[Any] = []
    prd_constraints: List[Any] = []
    prd_low_affine_terms: List[Any] = []
    prd_high_affine_terms: List[Any] = []

    prd_lo, prd_hi = float(IAAO_PRD_RANGE[0]), float(IAAO_PRD_RANGE[1])
    prd_constraint_aggregation = str(prd_constraint_aggregation)
    if prd_constraint_aggregation not in {"worst", "average", "cvar"}:
        raise ValueError("prd_constraint_aggregation must be one of: worst, average, cvar")
    prd_feasibility_mode = str(prd_feasibility_mode)
    if prd_feasibility_mode not in {"priced_gap", "hard"}:
        raise ValueError("prd_feasibility_mode must be one of: priced_gap, hard")
    prd_gap_penalty = float(prd_gap_penalty)
    if prd_feasibility_mode == "priced_gap" and prd_gap_penalty < 0.0:
        raise ValueError("prd_gap_penalty must be >= 0.")
    prd_cvar_alpha = float(prd_constraint_cvar_alpha)
    if prd_constraint_aggregation == "cvar" and not (0.0 < prd_cvar_alpha < 1.0):
        raise ValueError("prd_constraint_cvar_alpha must be in (0, 1) when prd_constraint_aggregation=cvar")

    scenario_losses: List[Any] = []
    rng = np.random.default_rng(int(bootstrap_seed))

    for f in folds:
        P, y = fold_panels[int(f)]
        y_pos = np.maximum(y, 1e-9)
        # Numeric stabilization: normalize each fold to keep objective/constraints well-scaled.
        # PRD constraints are invariant to common scaling of (P, y), so this does not alter feasibility geometry.
        fold_scale = max(float(np.median(y_pos)), 1.0)
        Pn = P / fold_scale
        y_norm = y_pos / fold_scale

        # Exact PRD linear constraints after clearing denominators:
        #   l * sum(yhat) <= sum(y) * sum(yhat/y) <= u * sum(yhat)
        a = np.sum(Pn / y_norm[:, None], axis=0)
        b = np.sum(Pn, axis=0)
        sy = float(np.sum(y_norm))
        low_aff = prd_lo * (b @ w) - sy * (a @ w)
        high_aff = sy * (a @ w) - prd_hi * (b @ w)
        prd_low_affine_terms.append(low_aff)
        prd_high_affine_terms.append(high_aff)
        denom_constraints += [(b @ w) >= 1e-9]

        # Scale-stable objective: mean squared ratio residual, ((yhat / y) - 1)^2.
        # With y treated as fixed positive data, this remains a convex quadratic in w.
        loss_f = cp.sum_squares(cp.multiply(1.0 / y_norm, Pn @ w) - 1.0) / float(y_norm.shape[0])
        scenario_losses.append(loss_f)

        if use_bootstrap_scenarios and int(n_bootstrap_scenarios) > 0:
            n = int(y_norm.shape[0])
            for _ in range(int(n_bootstrap_scenarios)):
                idx = rng.integers(0, n, size=n)
                Pb = Pn[idx, :]
                yb = y_norm[idx]
                yb_pos = np.maximum(yb, 1e-9)
                ab = np.sum(Pb / yb_pos[:, None], axis=0)
                bb = np.sum(Pb, axis=0)
                syb = float(np.sum(yb_pos))
                low_aff_b = prd_lo * (bb @ w) - syb * (ab @ w)
                high_aff_b = syb * (ab @ w) - prd_hi * (bb @ w)
                prd_low_affine_terms.append(low_aff_b)
                prd_high_affine_terms.append(high_aff_b)
                denom_constraints += [(bb @ w) >= 1e-9]
                scenario_losses.append(
                    cp.sum_squares(cp.multiply(1.0 / yb_pos, Pb @ w) - 1.0) / float(yb_pos.shape[0])
                )

    gap_var = cp.Variable(nonneg=True) if prd_feasibility_mode == "priced_gap" else None

    if prd_constraint_aggregation == "worst":
        for low_aff, high_aff in zip(prd_low_affine_terms, prd_high_affine_terms):
            if gap_var is None:
                prd_constraints.append(low_aff <= 0.0)
                prd_constraints.append(high_aff <= 0.0)
            else:
                prd_constraints.append(low_aff <= gap_var)
                prd_constraints.append(high_aff <= gap_var)
    elif prd_constraint_aggregation == "average":
        if gap_var is None:
            prd_constraints.append(cp.sum(cp.hstack(prd_low_affine_terms)) / float(len(prd_low_affine_terms)) <= 0.0)
            prd_constraints.append(cp.sum(cp.hstack(prd_high_affine_terms)) / float(len(prd_high_affine_terms)) <= 0.0)
        else:
            prd_constraints.append(cp.sum(cp.hstack(prd_low_affine_terms)) / float(len(prd_low_affine_terms)) <= gap_var)
            prd_constraints.append(cp.sum(cp.hstack(prd_high_affine_terms)) / float(len(prd_high_affine_terms)) <= gap_var)
    else:  # cvar
        eta_low = cp.Variable()
        u_low = cp.Variable(len(prd_low_affine_terms), nonneg=True)
        for i, low_aff in enumerate(prd_low_affine_terms):
            prd_constraints.append(u_low[i] >= low_aff - eta_low)
        if gap_var is None:
            prd_constraints.append(
                eta_low + (1.0 / ((1.0 - prd_cvar_alpha) * float(len(prd_low_affine_terms)))) * cp.sum(u_low) <= 0.0
            )
        else:
            prd_constraints.append(
                eta_low + (1.0 / ((1.0 - prd_cvar_alpha) * float(len(prd_low_affine_terms)))) * cp.sum(u_low) <= gap_var
            )

        eta_high = cp.Variable()
        u_high = cp.Variable(len(prd_high_affine_terms), nonneg=True)
        for i, high_aff in enumerate(prd_high_affine_terms):
            prd_constraints.append(u_high[i] >= high_aff - eta_high)
        if gap_var is None:
            prd_constraints.append(
                eta_high + (1.0 / ((1.0 - prd_cvar_alpha) * float(len(prd_high_affine_terms)))) * cp.sum(u_high) <= 0.0
            )
        else:
            prd_constraints.append(
                eta_high + (1.0 / ((1.0 - prd_cvar_alpha) * float(len(prd_high_affine_terms)))) * cp.sum(u_high) <= gap_var
            )

    objective_expr = None
    if mode == "average":
        objective_expr = cp.sum(cp.hstack(scenario_losses)) / float(len(scenario_losses))
    elif mode == "worst":
        t = cp.Variable()
        for lf in scenario_losses:
            objective_link_constraints.append(lf <= t)
        objective_expr = t
    elif mode == "time_decay":
        gamma = float(time_decay_gamma)
        if not (0.0 < gamma <= 1.0):
            raise ValueError("time_decay_gamma must be in (0, 1].")
        # fold-order weights (recent folds get higher weight)
        K = len(folds)
        fold_weights = np.asarray([gamma ** (K - 1 - i) for i in range(K)], dtype=float)
        fold_weights = fold_weights / np.sum(fold_weights)
        # Only fold scenarios are time-weighted; bootstrap scenarios keep the fold's weight.
        weighted_terms = []
        idx = 0
        for i, _f in enumerate(folds):
            rep = 1 + (int(n_bootstrap_scenarios) if use_bootstrap_scenarios else 0)
            for _ in range(rep):
                weighted_terms.append(float(fold_weights[i]) * scenario_losses[idx] / float(rep))
                idx += 1
        objective_expr = cp.sum(cp.hstack(weighted_terms))
    else:  # cvar
        alpha = float(cvar_alpha)
        if not (0.0 < alpha < 1.0):
            raise ValueError("cvar_alpha must be in (0, 1).")
        eta = cp.Variable()
        u = cp.Variable(len(scenario_losses), nonneg=True)
        for i, lf in enumerate(scenario_losses):
            objective_link_constraints.append(u[i] >= lf - eta)
        objective_expr = eta + (1.0 / ((1.0 - alpha) * float(len(scenario_losses)))) * cp.sum(u)

    if gap_var is not None:
        objective_expr = objective_expr + float(prd_gap_penalty) * gap_var
    objective = cp.Minimize(objective_expr)

    hard_constraints = simplex_constraints + denom_constraints + objective_link_constraints + prd_constraints
    prob = cp.Problem(objective, hard_constraints)
    preferred = _choose_solver(solver)
    installed = set(cp.installed_solvers())
    fallback_chain = [s for s in ["ECOS", "SCS", "OSQP", "CLARABEL"] if s != preferred and s in installed]
    solvers_to_try = [preferred] + fallback_chain

    solver_name = preferred
    last_exc: Optional[Exception] = None
    solved = False
    for candidate in solvers_to_try:
        print(f"[stacking][{mode}] trying solver={candidate} (hard PRD)")
        try:
            prob.solve(solver=candidate, verbose=bool(solver_verbose))
            if prob.status not in (None, "infeasible", "infeasible_inaccurate", "unbounded") and prob.value is not None:
                print(
                    f"[stacking][{mode}] solver={candidate} SUCCESS "
                    f"status={prob.status} objective={prob.value}"
                )
                solver_name = candidate
                solved = True
                last_exc = None
                break
            print(
                f"[stacking][{mode}] solver={candidate} NOT_SOLVED "
                f"status={prob.status} objective={prob.value} "
                f"reason=status_not_acceptable_or_no_objective_value"
            )
        except Exception as exc:
            last_exc = exc
            print(
                f"[stacking][{mode}] solver={candidate} EXCEPTION "
                f"type={type(exc).__name__} reason={exc}"
            )

    prd_constraint_mode = "hard"
    if not solved:
        # Final fallback: drop PRD constraints and solve simplex-only objective.
        # Runtime evidence showed some splits are infeasible under hard PRD bounds.
        fallback_constraints = simplex_constraints + objective_link_constraints
        fallback_prob = cp.Problem(cp.Minimize(objective_expr), fallback_constraints)
        for candidate in solvers_to_try:
            print(f"[stacking][{mode}] trying solver={candidate} (unconstrained fallback)")
            try:
                fallback_prob.solve(solver=candidate, verbose=bool(solver_verbose))
                if fallback_prob.status not in (None, "infeasible", "infeasible_inaccurate", "unbounded") and fallback_prob.value is not None:
                    print(
                        f"[stacking][{mode}] solver={candidate} SUCCESS "
                        f"status={fallback_prob.status} objective={fallback_prob.value} "
                        f"path=unconstrained_fallback"
                    )
                    solver_name = candidate
                    solved = True
                    prob = fallback_prob
                    prd_constraint_mode = "unconstrained_fallback"
                    break
                print(
                    f"[stacking][{mode}] solver={candidate} NOT_SOLVED "
                    f"status={fallback_prob.status} objective={fallback_prob.value} "
                    f"reason=status_not_acceptable_or_no_objective_value "
                    f"path=unconstrained_fallback"
                )
            except Exception as exc:
                last_exc = exc
                print(
                    f"[stacking][{mode}] solver={candidate} EXCEPTION "
                    f"type={type(exc).__name__} reason={exc} "
                    f"path=unconstrained_fallback"
                )

    if not solved:
        if last_exc is not None:
            raise RuntimeError(
                f"All candidate solvers failed for mode={mode}. "
                f"tried={solvers_to_try}. Last error: {last_exc}"
            ) from last_exc
        raise RuntimeError(
            f"No candidate solver produced a valid solution for mode={mode}. "
            f"tried={solvers_to_try}, status={prob.status}, value={prob.value}"
        )

    weights = np.asarray(w.value, dtype=float).reshape(-1)
    if not np.all(np.isfinite(weights)):
        raise RuntimeError(f"Optimization failed for mode={mode} (non-finite weights). status={prob.status}")
    if float(np.sum(weights)) <= 0:
        raise RuntimeError(f"Optimization failed for mode={mode} (weights sum to 0). status={prob.status}")
    weights = np.maximum(weights, 0.0)
    weights = weights / np.sum(weights)
    return weights, {
        "status": str(prob.status),
        "solver": solver_name,
        "objective_mode": mode,
        "prd_constraint_mode": str(prd_constraint_mode),
        "prd_feasibility_mode": str(prd_feasibility_mode),
        "prd_gap_penalty": float(prd_gap_penalty),
        "prd_gap_value": (float(gap_var.value) if (gap_var is not None and gap_var.value is not None) else 0.0),
        "prd_constraint_aggregation": str(prd_constraint_aggregation),
        "prd_constraint_cvar_alpha": float(prd_cvar_alpha),
        "objective_value": float(prob.value) if prob.value is not None else None,
        "n_constraints": int(len(hard_constraints)),
        "n_prd_bound_constraints": int(len(prd_constraints)),
        "n_scenarios": int(len(scenario_losses)),
    }


def run_stacking_pf_optimization(
    *,
    result_root: str,
    data_id: str,
    split_id: str,
    accuracy_metric: str = "OOS R2",
    objective_mode: str = "worst_fold",
    max_models: Optional[int] = 100,
    solver: Optional[str] = None,
    solver_verbose: bool = True,
) -> Dict[str, str]:
    """
    Solve:
      max_w  min_f  acc_f(w)        (objective_mode="worst_fold")
      max_w  mean_f acc_f(w)        (objective_mode="mean_fold")
    s.t.  w >= 0, sum(w)=1,
          PRD in range, PRB in range, VEI in range  (for every fold)
    """
    objective_mode = str(objective_mode)
    if objective_mode not in {"worst_fold", "mean_fold"}:
        raise ValueError("objective_mode must be one of: worst_fold, mean_fold")

    runs_df = _load_runs_df(result_root=result_root, data_id=data_id, split_id=split_id)

    required = [str(accuracy_metric), "PRD", "PRB", "VEI", "R2"]
    aligned_df, folds, models = _align_complete_grid(runs_df, required_metrics=required)

    # Optional pruning by mean accuracy to keep problem small.
    if max_models is not None and int(max_models) > 0 and len(models) > int(max_models):
        p_acc = aligned_df.pivot_table(index="fold_id", columns="config_id", values=str(accuracy_metric), aggfunc="first")
        mean_acc = p_acc.mean(axis=0).sort_values(ascending=False)
        models = mean_acc.index[: int(max_models)].tolist()
        aligned_df = aligned_df[aligned_df["config_id"].isin(models)].copy()
        aligned_df, folds, models = _align_complete_grid(aligned_df, required_metrics=required)

    A = _pivot_metric(aligned_df, metric=str(accuracy_metric), folds=folds, models=models)
    R2 = A if str(accuracy_metric) == "R2" else _pivot_metric(aligned_df, metric="R2", folds=folds, models=models)
    PRD = _pivot_metric(aligned_df, metric="PRD", folds=folds, models=models)
    PRB = _pivot_metric(aligned_df, metric="PRB", folds=folds, models=models)
    VEI = _pivot_metric(aligned_df, metric="VEI", folds=folds, models=models)

    n_models = len(models)
    n_folds = len(folds)

    w = cp.Variable(n_models, nonneg=True)
    constraints = [cp.sum(w) == 1]

    prd_lo, prd_hi = float(IAAO_PRD_RANGE[0]), float(IAAO_PRD_RANGE[1])
    prb_lo, prb_hi = float(IAAO_PRB_RANGE[0]), float(IAAO_PRB_RANGE[1])
    vei_lo, vei_hi = float(IAAO_VEI_RANGE[0]), float(IAAO_VEI_RANGE[1])

    for f in range(n_folds):
        constraints += [
            # PRD[f, :] @ w >= prd_lo,
            # PRD[f, :] @ w <= prd_hi,
            # PRB[f, :] @ w >= prb_lo,
            # PRB[f, :] @ w <= prb_hi,
            VEI[f, :] @ w >= vei_lo,
            VEI[f, :] @ w <= vei_hi,
        ]

    if objective_mode == "worst_fold":
        t = cp.Variable()
        for f in range(n_folds):
            constraints.append(A[f, :] @ w >= t)
        objective = cp.Maximize(t)
    else:
        objective = cp.Maximize(cp.sum(A @ w) / float(n_folds))

    prob = cp.Problem(objective, constraints)
    solver_name = _choose_solver(solver)
    prob.solve(solver=solver_name, verbose=bool(solver_verbose))

    weights = np.asarray(w.value, dtype=float).reshape(-1)
    if not np.all(np.isfinite(weights)):
        raise RuntimeError(f"Optimization failed (non-finite weights). status={prob.status}")
    if weights.sum() <= 0:
        raise RuntimeError(f"Optimization failed (weights sum to 0). status={prob.status}")
    weights = np.maximum(weights, 0.0)
    weights = weights / weights.sum()

    weights_df = pd.DataFrame({"config_id": models, "weight": weights})
    weights_df = weights_df.merge(
        aligned_df[["config_id", "model_name"]].drop_duplicates("config_id"), on="config_id", how="left"
    ).sort_values("weight", ascending=False, ignore_index=True)

    # Ensemble fold metrics under the linear approximation.
    acc_ens = A @ weights
    r2_ens = R2 @ weights
    prd_ens = PRD @ weights
    prb_ens = PRB @ weights
    vei_ens = VEI @ weights
    fold_df = pd.DataFrame(
        {
            "fold_id": folds,
            f"{accuracy_metric}_ensemble": acc_ens,
            "R2_ensemble": r2_ens,
            "PRD_ensemble": prd_ens,
            "PRB_ensemble": prb_ens,
            "VEI_ensemble": vei_ens,
        }
    )

    out_dir = Path(result_root) / "analysis" / f"data_id={data_id}" / f"split_id={split_id}" / "stacking_pf_opt"
    out_dir.mkdir(parents=True, exist_ok=True)

    weights_csv = out_dir / "weights.csv"
    fold_metrics_csv = out_dir / "fold_ensemble_metrics.csv"
    summary_json = out_dir / "optimization_summary.json"

    weights_df.to_csv(weights_csv, index=False)
    fold_df.to_csv(fold_metrics_csv, index=False)
    summary = {
        "status": str(prob.status),
        "objective_mode": objective_mode,
        "accuracy_metric": str(accuracy_metric),
        "solver": solver_name,
        "n_models": int(n_models),
        "n_folds": int(n_folds),
        "objective_value": float(prob.value) if prob.value is not None else None,
    }
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    return {"weights_csv": str(weights_csv), "fold_metrics_csv": str(fold_metrics_csv), "summary_json": str(summary_json)}


def run_stacking_prd_socp_optimization(
    *,
    result_root: str,
    data_id: str,
    split_id: str,
    objective_mode: str = "average",
    max_models: Optional[int] = 100,
    solver: Optional[str] = None,
    solver_verbose: bool = True,
    prd_constraint_aggregation: str = "worst",
    prd_constraint_cvar_alpha: float = 0.8,
    use_bootstrap_scenarios: bool = False,
    n_bootstrap_scenarios: int = 0,
    bootstrap_seed: int = 2025,
    time_decay_gamma: float = 0.9,
    cvar_alpha: float = 0.8,
    solve_all_modes: bool = False,
    prd_feasibility_mode: str = "priced_gap",
    prd_gap_penalty: float = 1000.0,
    utopia_aggregation: str = "euclidean",
    utopia_cvar_alpha: float = 0.8,
    utopia_accuracy_weight: float = 1.0,
    utopia_fairness_weight: float = 1.0,
) -> Dict[str, str]:
    """
    PRD-constrained exact convex stacking using row-level fold predictions.
    By default, solves only the requested objective_mode.
    Set solve_all_modes=True to also solve and store `average` and `worst`.
    """
    objective_mode = str(objective_mode)
    if objective_mode not in {"average", "worst", "time_decay", "cvar"}:
        raise ValueError("objective_mode must be one of: average, worst, time_decay, cvar")
    runs_df = _load_runs_df(result_root=result_root, data_id=data_id, split_id=split_id)
    preds_df = _load_predictions_df(result_root=result_root, data_id=data_id, split_id=split_id)
    folds, models, fold_panels, meta_df = _build_fold_prediction_panels(
        runs_df=runs_df,
        preds_df=preds_df,
        accuracy_metric="OOS R2",
        max_models=max_models,
    )

    if bool(solve_all_modes):
        modes_to_run: List[str] = []
        for m in ["average", "worst", objective_mode]:
            if m not in modes_to_run:
                modes_to_run.append(m)
    else:
        modes_to_run = [objective_mode]

    out_dir = Path(result_root) / "analysis" / f"data_id={data_id}" / f"split_id={split_id}" / "stacking_prd_socp_opt"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _build_fold_metrics_from_weights(weights: np.ndarray, mode_tag: str) -> pd.DataFrame:
        fold_rows: List[Dict[str, Any]] = []
        for f in folds:
            P, y = fold_panels[int(f)]
            y_hat = np.maximum(P @ weights, 1e-9)
            y_true = np.maximum(y, 1e-9)
            met = _compute_extended_metrics(
                y_true_log=np.log(y_true),
                y_pred_log=np.log(y_hat),
                y_train_log=np.log(y_true),
                ratio_mode="diff",
            )
            fold_rows.append(
                {
                    "fold_id": int(f),
                    "objective_mode": str(mode_tag),
                    "OOS R2_ensemble": float(met.get("OOS R2", np.nan)),
                    "R2_ensemble": float(met.get("R2", np.nan)),
                    "PRD_ensemble": float(met.get("PRD", np.nan)),
                    "PRB_ensemble": float(met.get("PRB", np.nan)),
                    "VEI_ensemble": float(met.get("VEI", np.nan)),
                    "COD_ensemble": float(met.get("COD", np.nan)),
                    "Corr(r,price)_ensemble": float(met.get("Corr(r,price)", np.nan)),
                }
            )
        return pd.DataFrame(fold_rows)

    utopia_w, utopia_fold_df, utopia_meta = _build_utopia_solution(
        runs_df=runs_df,
        folds=folds,
        models=models,
        solver=solver,
        solver_verbose=solver_verbose,
        aggregation=utopia_aggregation,
        cvar_alpha=utopia_cvar_alpha,
        accuracy_weight=utopia_accuracy_weight,
        fairness_weight=utopia_fairness_weight,
    )
    utopia_weights_df = pd.DataFrame({"config_id": models, "weight": utopia_w})
    utopia_weights_df = utopia_weights_df.merge(meta_df, on="config_id", how="left").sort_values(
        "weight", ascending=False, ignore_index=True
    )
    utopia_weights_path = out_dir / "weights_utopia.csv"
    utopia_fold_path = out_dir / "fold_ensemble_metrics_utopia.csv"
    utopia_weights_df.to_csv(utopia_weights_path, index=False)
    utopia_fold_df.to_csv(utopia_fold_path, index=False)

    single_utopia_w, single_utopia_fold_df, single_utopia_meta = _build_single_utopia_solution(
        runs_df=runs_df,
        folds=folds,
        models=models,
        aggregation=utopia_aggregation,
        cvar_alpha=utopia_cvar_alpha,
        accuracy_weight=utopia_accuracy_weight,
        fairness_weight=utopia_fairness_weight,
    )
    single_utopia_weights_df = pd.DataFrame({"config_id": models, "weight": single_utopia_w})
    single_utopia_weights_df = single_utopia_weights_df.merge(meta_df, on="config_id", how="left").sort_values(
        "weight", ascending=False, ignore_index=True
    )
    single_utopia_weights_path = out_dir / "weights_single_utopia.csv"
    single_utopia_fold_path = out_dir / "fold_ensemble_metrics_single_utopia.csv"
    single_utopia_weights_df.to_csv(single_utopia_weights_path, index=False)
    single_utopia_fold_df.to_csv(single_utopia_fold_path, index=False)

    all_summaries: Dict[str, Any] = {}
    for mode in modes_to_run:
        w_opt, summ = _solve_prd_socp_for_mode(
            mode=mode,
            folds=folds,
            models=models,
            fold_panels=fold_panels,
            solver=solver,
            solver_verbose=solver_verbose,
            time_decay_gamma=time_decay_gamma,
            cvar_alpha=cvar_alpha,
            prd_constraint_aggregation=prd_constraint_aggregation,
            prd_constraint_cvar_alpha=prd_constraint_cvar_alpha,
            prd_feasibility_mode=prd_feasibility_mode,
            prd_gap_penalty=prd_gap_penalty,
            use_bootstrap_scenarios=bool(use_bootstrap_scenarios),
            n_bootstrap_scenarios=int(n_bootstrap_scenarios),
            bootstrap_seed=int(bootstrap_seed),
        )

        weights_df = pd.DataFrame({"config_id": models, "weight": w_opt})
        weights_df = weights_df.merge(meta_df, on="config_id", how="left").sort_values("weight", ascending=False, ignore_index=True)

        fold_df = _build_fold_metrics_from_weights(w_opt, str(mode))

        w_path = out_dir / f"weights_{mode}.csv"
        f_path = out_dir / f"fold_ensemble_metrics_{mode}.csv"
        s_path = out_dir / f"optimization_summary_{mode}.json"
        weights_df.to_csv(w_path, index=False)
        fold_df.to_csv(f_path, index=False)
        mode_summary = {
            **summ,
            "mode": mode,
            "n_models": int(len(models)),
            "n_folds": int(len(folds)),
            "weights_csv": str(w_path),
            "fold_metrics_csv": str(f_path),
        }
        s_path.write_text(json.dumps(mode_summary, indent=2, sort_keys=True), encoding="utf-8")
        all_summaries[mode] = mode_summary

    # Single-model alternative for the requested PRD-SOCP objective.
    single_w, single_meta = _build_single_requested_solution(
        mode=objective_mode,
        folds=folds,
        models=models,
        fold_panels=fold_panels,
        time_decay_gamma=time_decay_gamma,
        cvar_alpha=cvar_alpha,
        prd_constraint_aggregation=prd_constraint_aggregation,
        prd_constraint_cvar_alpha=prd_constraint_cvar_alpha,
        prd_feasibility_mode=prd_feasibility_mode,
        prd_gap_penalty=prd_gap_penalty,
        use_bootstrap_scenarios=bool(use_bootstrap_scenarios),
        n_bootstrap_scenarios=int(n_bootstrap_scenarios),
        bootstrap_seed=int(bootstrap_seed),
    )
    single_weights_df = pd.DataFrame({"config_id": models, "weight": single_w})
    single_weights_df = single_weights_df.merge(meta_df, on="config_id", how="left").sort_values(
        "weight", ascending=False, ignore_index=True
    )
    single_fold_df = _build_fold_metrics_from_weights(single_w, f"single_{objective_mode}")
    single_weights_path = out_dir / f"weights_single_{objective_mode}.csv"
    single_fold_path = out_dir / f"fold_ensemble_metrics_single_{objective_mode}.csv"
    single_weights_df.to_csv(single_weights_path, index=False)
    single_fold_df.to_csv(single_fold_path, index=False)
    # Stable aliases for downstream readers.
    (out_dir / "weights_single_requested.csv").write_text(single_weights_path.read_text(encoding="utf-8"), encoding="utf-8")
    (out_dir / "fold_ensemble_metrics_single_requested.csv").write_text(
        single_fold_path.read_text(encoding="utf-8"), encoding="utf-8"
    )

    # Compatibility aliases for downstream readers: use requested default objective.
    default_weights = out_dir / f"weights_{objective_mode}.csv"
    default_folds = out_dir / f"fold_ensemble_metrics_{objective_mode}.csv"
    (out_dir / "weights.csv").write_text(default_weights.read_text(encoding="utf-8"), encoding="utf-8")
    (out_dir / "fold_ensemble_metrics.csv").write_text(default_folds.read_text(encoding="utf-8"), encoding="utf-8")

    summary_json = out_dir / "optimization_summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "stacking_method": "prd_socp",
                "default_objective_mode": objective_mode,
                "solved_modes": modes_to_run,
                "use_bootstrap_scenarios": bool(use_bootstrap_scenarios),
                "n_bootstrap_scenarios": int(n_bootstrap_scenarios),
                "time_decay_gamma": float(time_decay_gamma),
                "cvar_alpha": float(cvar_alpha),
                "prd_constraint_aggregation": str(prd_constraint_aggregation),
                "prd_constraint_cvar_alpha": float(prd_constraint_cvar_alpha),
                "solve_all_modes": bool(solve_all_modes),
                "prd_feasibility_mode": str(prd_feasibility_mode),
                "prd_gap_penalty": float(prd_gap_penalty),
                "utopia_aggregation": str(utopia_aggregation),
                "utopia_cvar_alpha": float(utopia_cvar_alpha),
                "utopia_accuracy_weight": float(utopia_accuracy_weight),
                "utopia_fairness_weight": float(utopia_fairness_weight),
                "utopia": {
                    **utopia_meta,
                    "weights_csv": str(utopia_weights_path),
                    "fold_metrics_csv": str(utopia_fold_path),
                },
                "single_requested": {
                    **single_meta,
                    "weights_csv": str(single_weights_path),
                    "fold_metrics_csv": str(single_fold_path),
                },
                "single_utopia": {
                    **single_utopia_meta,
                    "weights_csv": str(single_utopia_weights_path),
                    "fold_metrics_csv": str(single_utopia_fold_path),
                },
                "modes": all_summaries,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return {
        "stacking_dir": str(out_dir),
        "weights_csv": str(out_dir / "weights.csv"),
        "fold_metrics_csv": str(out_dir / "fold_ensemble_metrics.csv"),
        "summary_json": str(summary_json),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Convex stacking optimization (metric-level approximation).")
    p.add_argument("--result-root", type=str, default="./output/robust_rolling_origin_cv")
    p.add_argument("--data-id", type=str, required=True)
    p.add_argument("--split-id", type=str, required=True)
    p.add_argument("--stacking-method", type=str, default="prd_socp", choices=["prd_socp", "legacy_linear"])
    p.add_argument("--objective-mode", type=str, default="average", choices=["average", "worst", "time_decay", "cvar"])
    p.add_argument(
        "--only-requested-objective",
        dest="only_requested_objective",
        action="store_true",
        default=True,
        help=(
            "Run only --objective-mode for PRD-SOCP (default behavior)."
        ),
    )
    p.add_argument(
        "--all-objectives",
        dest="only_requested_objective",
        action="store_false",
        help="Run average, worst, and requested objective for PRD-SOCP (legacy behavior).",
    )
    p.add_argument("--use-bootstrap-scenarios", action="store_true", help="Augment fold scenarios with row-bootstrap scenarios (PRD-SOCP method only).")
    p.add_argument("--n-bootstrap-scenarios", type=int, default=0)
    p.add_argument("--bootstrap-seed", type=int, default=2025)
    p.add_argument("--time-decay-gamma", type=float, default=0.9, help="Decay in (0,1]; lower gives more weight to recent folds.")
    p.add_argument("--cvar-alpha", type=float, default=0.8, help="CVaR tail level in (0,1) for objective_mode=cvar.")
    p.add_argument(
        "--prd-constraint-aggregation",
        type=str,
        default="worst",
        choices=["worst", "average", "cvar"],
        help="How to aggregate PRD feasibility across folds/scenarios for PRD-SOCP.",
    )
    p.add_argument(
        "--prd-constraint-cvar-alpha",
        type=float,
        default=0.8,
        help="CVaR alpha in (0,1) when --prd-constraint-aggregation=cvar.",
    )
    p.add_argument(
        "--prd-feasibility-mode",
        type=str,
        default="priced_gap",
        choices=["priced_gap", "hard"],
        help=(
            "PRD feasibility handling for PRD-SOCP. "
            "'priced_gap' minimizes objective + xi*gap with softened PRD constraints; "
            "'hard' enforces hard PRD constraints."
        ),
    )
    p.add_argument(
        "--prd-gap-penalty",
        type=float,
        default=1000.0,
        help="Penalty xi for PRD gap when --prd-feasibility-mode=priced_gap (>=0).",
    )
    p.add_argument(
        "--utopia-aggregation",
        type=str,
        default="euclidean",
        choices=["average", "chebyshev", "cvar", "euclidean"],
        help="Aggregation used by utopia stacking optimization.",
    )
    p.add_argument(
        "--utopia-cvar-alpha",
        type=float,
        default=0.8,
        help="CVaR alpha in (0,1) when --utopia-aggregation=cvar.",
    )
    p.add_argument(
        "--utopia-accuracy-weight",
        type=float,
        default=1.0,
        help="Weight for normalized accuracy-gap term in utopia optimization.",
    )
    p.add_argument(
        "--utopia-fairness-weight",
        type=float,
        default=1.0,
        help="Weight for normalized unfairness term in utopia optimization.",
    )
    p.add_argument("--accuracy-metric", type=str, default="OOS R2", help="Metric column name to maximize (e.g., 'OOS R2' or 'R2').")
    p.add_argument("--legacy-objective-mode", type=str, default="worst_fold", choices=["worst_fold", "mean_fold"])
    p.add_argument("--max-models", type=int, default=100, help="Optional pruning: keep top-K models by mean accuracy.")
    p.add_argument("--solver", type=str, default=None, help="Preferred solver (e.g. MOSEK, ECOS, SCS).")
    p.add_argument("--solver-verbose", dest="solver_verbose", action="store_true", default=True)
    p.add_argument("--no-solver-verbose", dest="solver_verbose", action="store_false")
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    if str(args.stacking_method) == "legacy_linear":
        out = run_stacking_pf_optimization(
            result_root=str(args.result_root),
            data_id=str(args.data_id),
            split_id=str(args.split_id),
            accuracy_metric=str(args.accuracy_metric),
            objective_mode=str(args.legacy_objective_mode),
            max_models=(None if args.max_models is None else int(args.max_models)),
            solver=(None if args.solver is None else str(args.solver)),
            solver_verbose=bool(args.solver_verbose),
        )
    else:
        out = run_stacking_prd_socp_optimization(
            result_root=str(args.result_root),
            data_id=str(args.data_id),
            split_id=str(args.split_id),
            objective_mode=str(args.objective_mode),
            max_models=(None if args.max_models is None else int(args.max_models)),
            solver=(None if args.solver is None else str(args.solver)),
            solver_verbose=bool(args.solver_verbose),
            prd_constraint_aggregation=str(args.prd_constraint_aggregation),
            prd_constraint_cvar_alpha=float(args.prd_constraint_cvar_alpha),
            use_bootstrap_scenarios=bool(args.use_bootstrap_scenarios),
            n_bootstrap_scenarios=int(args.n_bootstrap_scenarios),
            bootstrap_seed=int(args.bootstrap_seed),
            time_decay_gamma=float(args.time_decay_gamma),
            cvar_alpha=float(args.cvar_alpha),
            solve_all_modes=not bool(args.only_requested_objective),
            prd_feasibility_mode=str(args.prd_feasibility_mode),
            prd_gap_penalty=float(args.prd_gap_penalty),
            utopia_aggregation=str(args.utopia_aggregation),
            utopia_cvar_alpha=float(args.utopia_cvar_alpha),
            utopia_accuracy_weight=float(args.utopia_accuracy_weight),
            utopia_fairness_weight=float(args.utopia_fairness_weight),
        )
    print("=" * 90)
    print("STACKING WEIGHTS OPTIMIZATION COMPLETED")
    print("=" * 90)
    print(f"weights_csv={out['weights_csv']}")
    print(f"fold_metrics_csv={out['fold_metrics_csv']}")
    print(f"summary_json={out['summary_json']}")

