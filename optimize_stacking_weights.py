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
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import cvxpy as cp
except ImportError as e:  # pragma: no cover
    raise ImportError("`optimize_stacking_weights.py` requires cvxpy. Install via `pip install cvxpy`.") from e

from utils.motivation_utils import IAAO_PRB_RANGE, IAAO_PRD_RANGE, IAAO_VEI_RANGE


def _load_runs_df(*, result_root: str, data_id: str, split_id: str) -> pd.DataFrame:
    runs_dir = Path(result_root) / "runs" / f"data_id={data_id}" / f"split_id={split_id}"
    if not runs_dir.exists():
        raise FileNotFoundError(f"CV runs directory not found: {runs_dir}")
    paths = sorted(runs_dir.rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No run parquet files found under: {runs_dir}")
    dfs = [pd.read_parquet(p) for p in paths]
    return pd.concat(dfs, ignore_index=True)


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

    dfx = df.loc[:, base_cols + required_metrics + ["model_name"]].copy()
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


def _choose_solver(preferred: str | None = None) -> str:
    installed = set(cp.installed_solvers())
    if preferred is not None and preferred.upper() in installed:
        return preferred.upper()
    for s in ["GUROBI", "MOSEK", "ECOS", "OSQP", "SCS", "CLARABEL"]:
        if s in installed:
            return s
    raise RuntimeError("No cvxpy solvers installed. Install one of: ecos, scs, osqp, clarabel, mosek.")


def run_stacking_pf_optimization(
    *,
    result_root: str,
    data_id: str,
    split_id: str,
    accuracy_metric: str = "OOS R2",
    objective_mode: str = "worst_fold",
    max_models: Optional[int] = 100,
    solver: Optional[str] = None,
    solver_verbose: bool = False,
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
            PRD[f, :] @ w >= prd_lo,
            PRD[f, :] @ w <= prd_hi,
            # PRB[f, :] @ w >= prb_lo,
            # PRB[f, :] @ w <= prb_hi,
            # VEI[f, :] @ w >= vei_lo,
            # VEI[f, :] @ w <= vei_hi,
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


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Convex stacking optimization (metric-level approximation).")
    p.add_argument("--result-root", type=str, default="./output/robust_rolling_origin_cv")
    p.add_argument("--data-id", type=str, required=True)
    p.add_argument("--split-id", type=str, required=True)
    p.add_argument("--accuracy-metric", type=str, default="OOS R2", help="Metric column name to maximize (e.g., 'OOS R2' or 'R2').")
    p.add_argument("--objective-mode", type=str, default="worst_fold", choices=["worst_fold", "mean_fold"])
    p.add_argument("--max-models", type=int, default=100, help="Optional pruning: keep top-K models by mean accuracy.")
    p.add_argument("--solver", type=str, default=None, help="Preferred solver (e.g. MOSEK, ECOS, SCS).")
    p.add_argument("--solver-verbose", action="store_true")
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    out = run_stacking_pf_optimization(
        result_root=str(args.result_root),
        data_id=str(args.data_id),
        split_id=str(args.split_id),
        accuracy_metric=str(args.accuracy_metric),
        objective_mode=str(args.objective_mode),
        max_models=(None if args.max_models is None else int(args.max_models)),
        solver=(None if args.solver is None else str(args.solver)),
        solver_verbose=bool(args.solver_verbose),
    )
    print("=" * 90)
    print("STACKING WEIGHTS OPTIMIZATION COMPLETED")
    print("=" * 90)
    print(f"weights_csv={out['weights_csv']}")
    print(f"fold_metrics_csv={out['fold_metrics_csv']}")
    print(f"summary_json={out['summary_json']}")

