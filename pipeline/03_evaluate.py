#!/usr/bin/env python3
"""
Evaluate (CCAO ``03-evaluate`` analog).

Cook County aggregates ratio-study and ML metrics by geography for both the
held-out test set and the assessment universe.

This stage focuses on the configurations selected by stage 02-assess and
produces a side-by-side report combining:

- per-fold CV metrics from ``runs/``
- aggregated CV stats (mean / std / max-or-min depending on direction)
- held-out test metrics from ``analysis/.../test_metrics.csv``
- an IAAO-style "in-bounds?" check on average fold values

Outputs (under ``analysis/data_id=…/split_id=…/selected/``):

- ``selected_models_evaluation.csv`` — wide table, one row per selection rule
- ``selected_models_folds.csv``      — long table, one row per (rule, fold)

Optionally also runs the existing heavy ``analyze_results.py`` with
``--also-analyze-all`` (default off — that script is slow and writes a large
suite of plots/tables to the same analysis directory).

Usage::

  python pipeline/03_evaluate.py
  python pipeline/03_evaluate.py --also-analyze-all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from pipeline._helpers import DEFAULT_RESULT_ROOT, parse_data_split_ids, run_repo_script
from pipeline._selection import (
    CONSTRAINT_SPECS,
    load_runs_df,
    load_test_metrics,
)


# Columns we summarize per fold for each selected config.
_FOLD_NUMERIC_COLS = (
    "R2",
    "RMSE",
    "MAE",
    "MAPE",
    "PRD",
    "PRB",
    "VEI",
    "COD",
    "Mean ratio",
    "Median ratio",
    "W. Mean ratio",
    "Corr(r,price)",
    "Corr(r,logprice)",
)


def _read_selected(analysis_dir: Path) -> Dict[str, Any]:
    json_path = analysis_dir / "selected" / "selected_models.json"
    if not json_path.is_file():
        raise FileNotFoundError(
            f"selected_models.json not found at {json_path}. Run stage 02_assess first."
        )
    return json.loads(json_path.read_text(encoding="utf-8"))


def _summarize_folds(runs_df: pd.DataFrame, *, config_id: str) -> Dict[str, Any]:
    sub = runs_df.loc[runs_df["config_id"].astype(str) == str(config_id)].copy()
    sub = sub.sort_values("fold_id")
    summary: Dict[str, Any] = {"n_folds": int(sub["fold_id"].nunique())}
    for col in _FOLD_NUMERIC_COLS:
        if col not in sub.columns:
            continue
        vals = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            summary[f"{col}_mean"] = np.nan
            summary[f"{col}_std"] = np.nan
            summary[f"{col}_min"] = np.nan
            summary[f"{col}_max"] = np.nan
            continue
        summary[f"{col}_mean"] = float(np.mean(vals))
        summary[f"{col}_std"] = float(np.std(vals, ddof=0))
        summary[f"{col}_min"] = float(np.min(vals))
        summary[f"{col}_max"] = float(np.max(vals))
    if "RMSE" in sub.columns:
        rmse_vals = pd.to_numeric(sub["RMSE"], errors="coerce").to_numpy(dtype=float)
        mse_vals = rmse_vals[np.isfinite(rmse_vals)] ** 2
        if mse_vals.size:
            summary["MSE_mean"] = float(np.mean(mse_vals))
            summary["MSE_std"] = float(np.std(mse_vals, ddof=0))
            summary["MSE_min"] = float(np.min(mse_vals))
            summary["MSE_max"] = float(np.max(mse_vals))
    return summary


def _bounds_label(metric_id: str) -> str:
    spec = CONSTRAINT_SPECS.get(metric_id)
    if spec is None:
        return ""
    lo = "(-inf" if spec.lower is None else f"[{spec.lower}"
    hi = "+inf)" if spec.upper is None else f"{spec.upper}]"
    return f"{lo}, {hi}"


def _in_bounds(value: float, metric_id: str) -> bool:
    spec = CONSTRAINT_SPECS.get(metric_id)
    if spec is None or not np.isfinite(value):
        return False
    if spec.lower is not None and value < spec.lower:
        return False
    if spec.upper is not None and value > spec.upper:
        return False
    return True


def _evaluate(
    *,
    result_root: Path,
    data_id: str,
    split_id: str,
    selected: Dict[str, Any],
) -> Dict[str, Path]:
    runs_df = load_runs_df(result_root, data_id, split_id)
    runs_df["config_id"] = runs_df["config_id"].astype(str)
    test_df = load_test_metrics(result_root, data_id, split_id)
    test_df["config_id"] = test_df["config_id"].astype(str)
    test_lookup = test_df.set_index("config_id").to_dict(orient="index")

    out_dir = result_root / "analysis" / f"data_id={data_id}" / f"split_id={split_id}" / "selected"
    out_dir.mkdir(parents=True, exist_ok=True)

    wide_rows: List[Dict[str, Any]] = []
    long_rows: List[Dict[str, Any]] = []
    for rule, sel in selected["selections"].items():
        cfg_id = str(sel["config_id"])
        cv_summary = _summarize_folds(runs_df, config_id=cfg_id)
        test_metrics = test_lookup.get(cfg_id, {})

        row: Dict[str, Any] = {
            "rule": rule,
            "config_id": cfg_id,
            "model_name": sel["model_name"],
            "model_family": sel.get("model_family", ""),
            "selector_label": sel.get("selector_label", ""),
            "n_folds": cv_summary.get("n_folds", 0),
            "model_config_json": sel.get("model_config_json", ""),
        }
        for col in (*_FOLD_NUMERIC_COLS, "MSE"):
            row[f"cv_{col}_mean"] = cv_summary.get(f"{col}_mean", np.nan)
            row[f"cv_{col}_std"] = cv_summary.get(f"{col}_std", np.nan)
            row[f"cv_{col}_min"] = cv_summary.get(f"{col}_min", np.nan)
            row[f"cv_{col}_max"] = cv_summary.get(f"{col}_max", np.nan)
        for cid in selected.get("constraint_metrics", []):
            mean_val = float(row.get(f"cv_{cid}_mean", np.nan))
            row[f"cv_{cid}_in_bounds_mean"] = _in_bounds(mean_val, cid)
            row[f"cv_{cid}_bounds"] = _bounds_label(cid)
        for col in _FOLD_NUMERIC_COLS:
            if col in test_metrics:
                try:
                    row[f"test_{col}"] = float(test_metrics[col])
                except (TypeError, ValueError):
                    row[f"test_{col}"] = np.nan
        if "nash_log_utility" in sel:
            row["nash_log_utility"] = float(sel.get("nash_log_utility", np.nan))
        # Legacy JSON keys
        if rule == "utopia":
            row["utopia_distance"] = float(sel.get("utopia_distance", np.nan))
        wide_rows.append(row)

        sub = runs_df.loc[runs_df["config_id"] == cfg_id].sort_values("fold_id")
        for _, fold_row in sub.iterrows():
            entry: Dict[str, Any] = {
                "rule": rule,
                "config_id": cfg_id,
                "model_name": sel["model_name"],
                "fold_id": int(fold_row["fold_id"]) if pd.notna(fold_row["fold_id"]) else -1,
            }
            for col in _FOLD_NUMERIC_COLS:
                if col in fold_row:
                    try:
                        entry[col] = float(fold_row[col])
                    except (TypeError, ValueError):
                        entry[col] = np.nan
            long_rows.append(entry)

    wide_df = pd.DataFrame(wide_rows)
    long_df = pd.DataFrame(long_rows)
    wide_path = out_dir / "selected_models_evaluation.csv"
    long_path = out_dir / "selected_models_folds.csv"
    wide_df.to_csv(wide_path, index=False)
    long_df.to_csv(long_path, index=False)
    return {"evaluation_csv": wide_path, "folds_csv": long_path}


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate stage — focused report on selected models.")
    p.add_argument("--result-root", type=str, default=str(DEFAULT_RESULT_ROOT))
    p.add_argument("--data-id", type=str, default=None)
    p.add_argument("--split-id", type=str, default=None)
    p.add_argument("--no-context", action="store_true")
    p.add_argument(
        "--also-analyze-all",
        action="store_true",
        help="Also invoke analyze_results.py for the heavy full-suite analysis.",
    )
    args = p.parse_args()

    result_root = Path(args.result_root).resolve()
    data_id, split_id = parse_data_split_ids(
        data_id=args.data_id,
        split_id=args.split_id,
        result_root=result_root,
        prefer_context=not args.no_context,
    )
    analysis_dir = result_root / "analysis" / f"data_id={data_id}" / f"split_id={split_id}"

    selected = _read_selected(analysis_dir)
    paths = _evaluate(
        result_root=result_root,
        data_id=data_id,
        split_id=split_id,
        selected=selected,
    )

    print("=" * 70)
    print("EVALUATE — focused report on selected models")
    print("=" * 70)
    print(f"  data_id={data_id}  split_id={split_id}")
    for k, v in paths.items():
        print(f"  → {k}: {v}")

    if args.also_analyze_all:
        run_repo_script(
            "analyze_results.py",
            ["--result-root", str(result_root), "--data-id", data_id, "--split-id", split_id],
        )


if __name__ == "__main__":
    main()
