from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import cvxpy as cp
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "This script requires cvxpy. Install with `pip install cvxpy` "
        "(and MOSEK if desired)."
    ) from exc

from results_analysis import load_cv_artifacts
from utils.motivation_utils import IAAO_PRB_RANGE, IAAO_PRD_RANGE, IAAO_VEI_RANGE


def _sanitize_metric_name(name: str) -> str:
    return (
        name.replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "_")
    )


def _build_metric_pivot(
    runs_df: pd.DataFrame,
    metric_col: str,
    id_cols: List[str],
) -> pd.DataFrame:
    if metric_col not in runs_df.columns:
        raise ValueError(f"Metric column not found in runs_df: {metric_col}")
    grp = (
        runs_df[id_cols + ["fold_id", metric_col]]
        .groupby(id_cols + ["fold_id"], dropna=False)[metric_col]
        .mean()
        .reset_index()
    )
    grp["model_id"] = grp[id_cols].astype(str).agg(" | ".join, axis=1)
    piv = grp.pivot(index="fold_id", columns="model_id", values=metric_col)
    return piv


def _align_pivots(pivots: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    common_folds = None
    common_models = None
    for p in pivots.values():
        idx = set(p.index.tolist())
        cols = set(p.columns.tolist())
        common_folds = idx if common_folds is None else (common_folds & idx)
        common_models = cols if common_models is None else (common_models & cols)

    if not common_folds or not common_models:
        raise ValueError("No common folds/models across requested metrics.")

    folds_sorted = sorted(common_folds)
    models_sorted = sorted(common_models)
    out: Dict[str, pd.DataFrame] = {}
    for k, p in pivots.items():
        out[k] = p.loc[folds_sorted, models_sorted].copy()

    # If there are missing entries (common when some fold×model runs failed),
    # filter down to the largest fully-observed rectangle across all metrics.
    any_nan = any(df.isna().any().any() for df in out.values())
    if any_nan:
        # Columns/models must be fully observed in ALL metrics.
        keep_cols = None
        for df in out.values():
            cols_ok = ~df.isna().any(axis=0)
            keep_cols = cols_ok if keep_cols is None else (keep_cols & cols_ok)
        keep_cols = keep_cols if keep_cols is not None else pd.Series([], dtype=bool)

        # Rows/folds must be fully observed in ALL metrics after col filtering.
        keep_rows = None
        for df in out.values():
            df2 = df.loc[:, keep_cols.index[keep_cols]].copy() if keep_cols.size else df.copy()
            rows_ok = ~df2.isna().any(axis=1)
            keep_rows = rows_ok if keep_rows is None else (keep_rows & rows_ok)
        keep_rows = keep_rows if keep_rows is not None else pd.Series([], dtype=bool)

        before = {k: (int(v.shape[0]), int(v.shape[1]), int(v.isna().to_numpy().sum())) for k, v in out.items()}
        for k in list(out.keys()):
            out[k] = out[k].loc[out[k].index[keep_rows], out[k].columns[keep_cols]].copy()
        after = {k: (int(v.shape[0]), int(v.shape[1]), int(v.isna().to_numpy().sum())) for k, v in out.items()}

        # If still NaNs, fail with details.
        if any(df.isna().any().any() for df in out.values()):
            bad = {k: int(v.isna().to_numpy().sum()) for k, v in out.items()}
            raise ValueError(f"Missing values after filtering complete grid: {bad}")

        # If filtering removed everything, fail loudly.
        if any(v.shape[0] == 0 or v.shape[1] == 0 for v in out.values()):
            raise ValueError("No fully-observed fold×model grid remains after filtering (too many failed/incomplete runs).")
    return out


def _solve_weight_optimization(
    acc: np.ndarray,
    prd: np.ndarray,
    prb: np.ndarray,
    vei: np.ndarray,
    objective_mode: str = "worst_fold",
) -> Tuple[np.ndarray, Dict[str, float], str]:
    n_folds, n_models = acc.shape
    w = cp.Variable(n_models, nonneg=True)
    constraints = [cp.sum(w) == 1.0]

    # Enforce IAAO fairness ranges on every fold (robust per-window constraints).
    prd_lo, prd_hi = IAAO_PRD_RANGE
    prb_lo, prb_hi = IAAO_PRB_RANGE
    vei_lo, vei_hi = IAAO_VEI_RANGE
    for i in range(n_folds):
        constraints += [
            prd[i, :] @ w >= float(prd_lo),
            prd[i, :] @ w <= float(prd_hi),
            # prb[i, :] @ w >= float(prb_lo),
            # prb[i, :] @ w <= float(prb_hi),
            # vei[i, :] @ w >= float(vei_lo),
            # vei[i, :] @ w <= float(vei_hi),
        ]

    if objective_mode == "worst_fold":
        t = cp.Variable()
        for i in range(n_folds):
            constraints.append(t <= acc[i, :] @ w)
        objective = cp.Maximize(t)
    elif objective_mode == "mean_fold":
        objective = cp.Maximize(cp.sum(acc @ w) / float(n_folds))
    else:
        raise ValueError("objective_mode must be 'worst_fold' or 'mean_fold'")

    problem = cp.Problem(objective, constraints)
    tried: List[str] = []
    solved_solver = ""
    for solver in [cp.MOSEK, cp.ECOS, cp.SCS]:
        try:
            problem.solve(solver=solver, verbose=False)
            tried.append(str(solver))
            if problem.status in ("optimal", "optimal_inaccurate"):
                solved_solver = str(solver)
                break
        except Exception:
            tried.append(str(solver))
            continue

    if problem.status not in ("optimal", "optimal_inaccurate") or w.value is None:
        raise RuntimeError(
            "Optimization did not solve successfully. "
            f"status={problem.status}, solvers_tried={tried}"
        )

    w_val = np.maximum(np.asarray(w.value).reshape(-1), 0.0)
    s = float(w_val.sum())
    if s <= 0:
        raise RuntimeError("Degenerate solution: weight sum <= 0.")
    w_val = w_val / s

    info = {
        "status": str(problem.status),
        "objective_value": float(problem.value) if problem.value is not None else np.nan,
    }
    return w_val, info, solved_solver


def run_stacking_pf_optimization(
    result_root: str = "./output/robust_rolling_origin_cv",
    data_id: Optional[str] = None,
    split_id: Optional[str] = None,
    accuracy_metric: str = "OOS R2",
    objective_mode: str = "worst_fold",
    max_models: Optional[int] = None,
) -> Dict[str, str]:
    loaded = load_cv_artifacts(result_root=result_root, data_id=data_id, split_id=split_id)
    runs_df = loaded["runs_df"]
    data_id = loaded["data_id"]
    split_id = loaded["split_id"]
    if runs_df.empty:
        raise ValueError("No completed run artifacts were found for this data_id/split_id.")

    id_cols = [c for c in ["config_id", "model_name"] if c in runs_df.columns]
    if len(id_cols) < 2:
        raise ValueError("Expected 'config_id' and 'model_name' columns in runs_df.")

    required = [accuracy_metric, "PRD", "PRB", "VEI", "fold_id"]
    missing = [c for c in required if c not in runs_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in runs_df: {missing}")

    pivots = {
        "acc": _build_metric_pivot(runs_df, accuracy_metric, id_cols),
        "prd": _build_metric_pivot(runs_df, "PRD", id_cols),
        "prb": _build_metric_pivot(runs_df, "PRB", id_cols),
        "vei": _build_metric_pivot(runs_df, "VEI", id_cols),
    }
    aligned = _align_pivots(pivots)

    # Optional trimming to top-k models by mean accuracy for speed/readability.
    if max_models is not None and int(max_models) > 0 and aligned["acc"].shape[1] > int(max_models):
        mean_acc = aligned["acc"].mean(axis=0).sort_values(ascending=False)
        keep_cols = mean_acc.head(int(max_models)).index.tolist()
        for k in aligned:
            aligned[k] = aligned[k][keep_cols]

    model_ids = aligned["acc"].columns.tolist()
    folds = aligned["acc"].index.tolist()
    acc = aligned["acc"].to_numpy(dtype=float)
    prd = aligned["prd"].to_numpy(dtype=float)
    prb = aligned["prb"].to_numpy(dtype=float)
    vei = aligned["vei"].to_numpy(dtype=float)

    w_opt, solve_info, solver_name = _solve_weight_optimization(
        acc=acc,
        prd=prd,
        prb=prb,
        vei=vei,
        objective_mode=objective_mode,
    )

    # Summaries by fold for diagnostics.
    fold_acc = acc @ w_opt
    fold_prd = prd @ w_opt
    fold_prb = prb @ w_opt
    fold_vei = vei @ w_opt
    fold_payload: Dict[str, object] = {
        "fold_id": folds,
        f"{_sanitize_metric_name(accuracy_metric)}_ensemble": fold_acc,
        "PRD_ensemble": fold_prd,
        "PRB_ensemble": fold_prb,
        "VEI_ensemble": fold_vei,
    }

    # Also export both accuracy metrics (if present) so downstream plots can overlay
    # the same stacking solution on both OOS R2 and R2 panels.
    for extra_acc in ["OOS R2", "R2"]:
        if extra_acc == accuracy_metric:
            continue
        if extra_acc in runs_df.columns:
            extra_piv = _build_metric_pivot(runs_df, extra_acc, id_cols)
            extra_piv = extra_piv.loc[folds, model_ids]
            fold_payload[f"{_sanitize_metric_name(extra_acc)}_ensemble"] = extra_piv.to_numpy(dtype=float) @ w_opt

    fold_df = pd.DataFrame(fold_payload)
    fold_df["PRD_ok"] = (fold_prd >= IAAO_PRD_RANGE[0]) & (fold_prd <= IAAO_PRD_RANGE[1])
    fold_df["PRB_ok"] = (fold_prb >= IAAO_PRB_RANGE[0]) & (fold_prb <= IAAO_PRB_RANGE[1])
    fold_df["VEI_ok"] = (fold_vei >= IAAO_VEI_RANGE[0]) & (fold_vei <= IAAO_VEI_RANGE[1])
    fold_df["all_fairness_ok"] = fold_df[["PRD_ok", "PRB_ok", "VEI_ok"]].all(axis=1)

    weights_df = pd.DataFrame({"model_id": model_ids, "weight": w_opt})
    weights_df = weights_df.sort_values("weight", ascending=False).reset_index(drop=True)

    out_dir = (
        Path(result_root)
        / "analysis"
        / f"data_id={data_id}"
        / f"split_id={split_id}"
        / "stacking_pf_opt"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    weights_path = out_dir / "weights.csv"
    fold_path = out_dir / "fold_ensemble_metrics.csv"
    summary_path = out_dir / "optimization_summary.json"
    weights_df.to_csv(weights_path, index=False)
    fold_df.to_csv(fold_path, index=False)

    summary = {
        "data_id": data_id,
        "split_id": split_id,
        "accuracy_metric": accuracy_metric,
        "objective_mode": objective_mode,
        "solver": solver_name,
        "n_models": int(len(model_ids)),
        "n_folds": int(len(folds)),
        "objective_value": solve_info.get("objective_value"),
        "status": solve_info.get("status"),
        "worst_fold_accuracy": float(np.min(fold_acc)),
        "mean_fold_accuracy": float(np.mean(fold_acc)),
        "all_folds_fairness_ok": bool(fold_df["all_fairness_ok"].all()),
        "iaao_ranges": {
            "PRD": [float(IAAO_PRD_RANGE[0]), float(IAAO_PRD_RANGE[1])],
            "PRB": [float(IAAO_PRB_RANGE[0]), float(IAAO_PRB_RANGE[1])],
            "VEI": [float(IAAO_VEI_RANGE[0]), float(IAAO_VEI_RANGE[1])],
        },
        "note": (
            "Fairness and accuracy are modeled as convex combinations of per-fold "
            "model metrics (readable convex surrogate). This script does not "
            "recompute non-linear fairness metrics directly from blended predictions."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "weights_csv": str(weights_path),
        "fold_metrics_csv": str(fold_path),
        "summary_json": str(summary_path),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Robust stacking optimization over CV models using cvxpy. "
            "Maximizes fold-level accuracy under per-fold IAAO fairness constraints."
        )
    )
    p.add_argument("--result-root", type=str, default="./output/robust_rolling_origin_cv")
    p.add_argument("--data-id", type=str, default=None)
    p.add_argument("--split-id", type=str, default=None)
    p.add_argument(
        "--accuracy-metric",
        type=str,
        default="OOS R2",
        help="Fold-level accuracy column from runs artifacts (e.g., 'OOS R2' or 'R2').",
    )
    p.add_argument(
        "--objective-mode",
        type=str,
        default="worst_fold",
        choices=["worst_fold", "mean_fold"],
        help="Optimize worst-fold accuracy (robust) or mean-fold accuracy.",
    )
    p.add_argument(
        "--max-models",
        type=int,
        default=None,
        help="Optional top-k models by mean accuracy to include for speed.",
    )
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    out = run_stacking_pf_optimization(
        result_root=args.result_root,
        data_id=args.data_id,
        split_id=args.split_id,
        accuracy_metric=args.accuracy_metric,
        objective_mode=args.objective_mode,
        max_models=args.max_models,
    )
    print("=" * 90)
    print("STACKING PF OPTIMIZATION COMPLETED")
    print("=" * 90)
    print(f"weights_csv={out['weights_csv']}")
    print(f"fold_metrics_csv={out['fold_metrics_csv']}")
    print(f"summary_json={out['summary_json']}")

