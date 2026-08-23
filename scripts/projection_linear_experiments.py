#!/usr/bin/env python
"""Run exact projection-path covariance experiments for linear regression.

This script is intentionally separate from ``quick_test_models.py`` because the
manuscript's preferred grid is in remaining-covariance fraction ``q`` rather than
raw rho, and rho(q) depends on the fitted linear prediction space.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression

from preprocessing.recipes_pipelined import build_model_pipeline
from quick_test_models import _compute_quick_test_metrics, _load_and_split_data
from soft_constrained_models.linear_models import LinearProjectionCovariancePath
from utils.projection_theory_utils import (
    add_projection_theory_predictions,
    compute_projection_theory_metrics,
)


T0 = time.perf_counter()


def log(message: str, **fields: Any) -> None:
    suffix = " | " + " | ".join(f"{k}={v}" for k, v in fields.items()) if fields else ""
    print(f"[projection-linear +{time.perf_counter() - T0:7.1f}s] {message}{suffix}", flush=True)


def parse_float_csv(raw: str) -> List[float]:
    values = [float(token.strip()) for token in str(raw).split(",") if token.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated float.")
    return values


def _cast_categoricals(X: pd.DataFrame, categorical_cols: Sequence[str]) -> pd.DataFrame:
    X = X.copy()
    for col in categorical_cols:
        if col in X.columns:
            X[col] = X[col].astype("category")
    return X


def _metric_row(
    *,
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    y_train_log: np.ndarray,
    baseline_pred_log: np.ndarray,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    metrics = _compute_quick_test_metrics(
        y_true_log=np.asarray(y_true_log, dtype=float),
        y_pred_log=np.asarray(y_pred_log, dtype=float),
        y_train_log=np.asarray(y_train_log, dtype=float),
        ratio_mode="diff",
    )
    metrics.update(
        compute_projection_theory_metrics(
            y_true_log=np.asarray(y_true_log, dtype=float),
            y_pred_log=np.asarray(y_pred_log, dtype=float),
            baseline_pred_log=np.asarray(baseline_pred_log, dtype=float),
        )
    )
    row = {**meta, **metrics}
    return add_projection_theory_predictions(row)


def _evaluate_linear_fit(
    *,
    fit_label: str,
    data_source: str,
    assessment_year: int,
    X_train: pd.DataFrame,
    y_train_log: np.ndarray,
    eval_sets: Sequence[Tuple[str, pd.DataFrame, np.ndarray]],
    linear_pipeline_builder,
    q_grid: Sequence[float],
) -> pd.DataFrame:
    log("building linear design", fit_label=fit_label, train_rows=len(y_train_log))
    pipe = linear_pipeline_builder()
    X_train_m = pipe.fit_transform(X_train, y_train_log)
    eval_mats = [(split, pipe.transform(X_eval), y_eval_log) for split, X_eval, y_eval_log in eval_sets]
    log("linear design ready", fit_label=fit_label, shape=str(getattr(X_train_m, "shape", "")))

    baseline = LinearRegression(fit_intercept=True)
    baseline.fit(X_train_m, y_train_log)
    baseline_train_pred = np.asarray(baseline.predict(X_train_m), dtype=float).reshape(-1)
    baseline_eval_preds = {
        split: np.asarray(baseline.predict(X_eval_m), dtype=float).reshape(-1)
        for split, X_eval_m, _ in eval_mats
    }

    rows: List[Dict[str, Any]] = []

    def add_baseline_row(split: str, y_true: np.ndarray, pred: np.ndarray, baseline_pred: np.ndarray) -> None:
        rows.append(
            _metric_row(
                y_true_log=y_true,
                y_pred_log=pred,
                y_train_log=y_train_log,
                baseline_pred_log=baseline_pred,
                meta={
                    "data_source": data_source,
                    "assessment_year": int(assessment_year),
                    "fit_label": fit_label,
                    "split": split,
                    "model_name": "LinearRegression",
                    "model_family": "LinearRegression",
                    "q_target": 1.0,
                    "rho": 0.0,
                    "is_exact_training_projection_check": split.startswith("train_"),
                },
            )
        )

    add_baseline_row(f"train_{fit_label}", y_train_log, baseline_train_pred, baseline_train_pred)
    for split, _, y_eval_log in eval_mats:
        add_baseline_row(split, y_eval_log, baseline_eval_preds[split], baseline_eval_preds[split])

    q_values = sorted({float(q) for q in q_grid if float(q) < 1.0}, reverse=True)
    for q in q_values:
        log("fitting projection-path model", fit_label=fit_label, q=q)
        model = LinearProjectionCovariancePath(target_q=float(q), fit_intercept=True)
        model.fit(X_train_m, y_train_log)
        train_pred = np.asarray(model.predict(X_train_m), dtype=float).reshape(-1)
        summary = model.theory_summary()
        common = {
            "data_source": data_source,
            "assessment_year": int(assessment_year),
            "fit_label": fit_label,
            "model_name": f"LinearProjectionCovariancePath_q_{q:.3f}",
            "model_family": "LinearProjectionCovariancePath",
            "q_target": float(q),
            "rho": float(summary["rho"]),
            "A_projection_capacity": float(summary["A_projection_capacity"]),
            "train_baseline_C_log_resid_logprice": float(summary["baseline_C_log_resid_logprice"]),
            "train_baseline_MSE_log": float(summary["baseline_MSE_log"]),
            "train_delta_MSE_log_theory": float(summary["delta_MSE_log_theory"]),
            "train_delta_MSE_log_frac_theory": float(summary["delta_MSE_log_frac_theory"]),
        }
        rows.append(
            _metric_row(
                y_true_log=y_train_log,
                y_pred_log=train_pred,
                y_train_log=y_train_log,
                baseline_pred_log=baseline_train_pred,
                meta={**common, "split": f"train_{fit_label}", "is_exact_training_projection_check": True},
            )
        )
        for split, X_eval_m, y_eval_log in eval_mats:
            pred = np.asarray(model.predict(X_eval_m), dtype=float).reshape(-1)
            rows.append(
                _metric_row(
                    y_true_log=y_eval_log,
                    y_pred_log=pred,
                    y_train_log=y_train_log,
                    baseline_pred_log=baseline_eval_preds[split],
                    meta={**common, "split": split, "is_exact_training_projection_check": False},
                )
            )

    return pd.DataFrame(rows)


def _write_verification_tables(metrics_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame()
    d = metrics_df.loc[metrics_df["model_family"].eq("LinearProjectionCovariancePath")].copy()
    if d.empty:
        return pd.DataFrame()
    d["q_error"] = pd.to_numeric(d["q_empirical_signed"], errors="coerce") - pd.to_numeric(d["q_theory"], errors="coerce")
    d["q_error_empirical_minus_theory"] = d["q_error"]
    d["delta_MSE_log_error"] = pd.to_numeric(d["delta_MSE_log"], errors="coerce") - pd.to_numeric(d["delta_MSE_log_theory"], errors="coerce")
    d["C_log_error"] = pd.to_numeric(d["C_log_resid_logprice"], errors="coerce") - pd.to_numeric(d["C_log_resid_logprice_theory"], errors="coerce")
    d["C_log_resid_logprice_error"] = d["C_log_error"]
    d["slope_error"] = pd.to_numeric(d["Slope_log_resid_logprice"], errors="coerce") - pd.to_numeric(d["Slope_log_resid_logprice_theory"], errors="coerce")
    d["Slope_log_resid_logprice_error"] = d["slope_error"]

    checks = []
    for (fit_label, split), g in d.groupby(["fit_label", "split"], dropna=False):
        exact = bool(str(split).startswith("train_"))
        row = {
            "fit_label": fit_label,
            "split": split,
            "is_exact_training_projection_check": exact,
            "n_rows": int(g.shape[0]),
        }
        for col in ["q_error", "delta_MSE_log_error", "C_log_error", "slope_error"]:
            values = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            row[f"{col}_max_abs"] = float(np.max(np.abs(values))) if values.size else np.nan
            row[f"{col}_rmse"] = float(np.sqrt(np.mean(values * values))) if values.size else np.nan
        for col in ["C_ratio_price_taylor1_rel_error", "C_ratio_price_taylor2_rel_error"]:
            values = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            row[f"{col}_median"] = float(np.median(values)) if values.size else np.nan
        checks.append(row)
    out = pd.DataFrame(checks)
    out.to_csv(out_dir / "linear_projection_verification_summary.csv", index=False)
    d.to_csv(out_dir / "linear_projection_theory_empirical_comparison.csv", index=False)
    return out


def _plot_linear_diagnostics(metrics_df: pd.DataFrame, out_dir: Path) -> List[Path]:
    paths: List[Path] = []
    d = metrics_df.loc[metrics_df["model_family"].eq("LinearProjectionCovariancePath")].copy()
    if d.empty:
        return paths
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    d["q_theory"] = pd.to_numeric(d["q_theory"], errors="coerce")
    d["q_empirical_signed"] = pd.to_numeric(d["q_empirical_signed"], errors="coerce")
    d["delta_MSE_log"] = pd.to_numeric(d["delta_MSE_log"], errors="coerce")
    d["delta_MSE_log_theory"] = pd.to_numeric(d["delta_MSE_log_theory"], errors="coerce")

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    for split, g in d.groupby("split"):
        ax.scatter(g["q_theory"], g["q_empirical_signed"], s=24, label=str(split), alpha=0.85)
    ax.axline((0, 0), slope=1, color="#111827", ls="--", lw=1.0)
    ax.set_xlabel("theory q")
    ax.set_ylabel("empirical C/C0")
    ax.set_title("Linear projection path: q check")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    p = plot_dir / "linear_q_empirical_vs_theory.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    for split, g in d.groupby("split"):
        ax.scatter(g["delta_MSE_log_theory"], g["delta_MSE_log"], s=24, label=str(split), alpha=0.85)
    finite = d[["delta_MSE_log_theory", "delta_MSE_log"]].replace([np.inf, -np.inf], np.nan).dropna()
    if not finite.empty:
        lo = float(finite.min().min())
        hi = float(finite.max().max())
        ax.plot([lo, hi], [lo, hi], color="#111827", ls="--", lw=1.0)
    ax.set_xlabel("theory delta MSE_log")
    ax.set_ylabel("empirical delta MSE_log")
    ax.set_title("Linear projection path: second-order MSE cost")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    p = plot_dir / "linear_mse_cost_empirical_vs_theory.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    for split, g in d.groupby("split"):
        ax.plot(
            g.sort_values("q_theory")["q_theory"],
            g.sort_values("q_theory")["C_ratio_price_taylor1_rel_error"],
            "-o",
            ms=3,
            lw=1.2,
            label=f"{split}: first order",
            alpha=0.75,
        )
        ax.plot(
            g.sort_values("q_theory")["q_theory"],
            g.sort_values("q_theory")["C_ratio_price_taylor2_rel_error"],
            "--o",
            ms=3,
            lw=1.2,
            label=f"{split}: second order",
            alpha=0.75,
        )
    ax.set_xlabel("theory q")
    ax.set_ylabel("relative error for Cov(r, price)")
    ax.set_title("Taylor bridge check: ratio-price covariance")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    p = plot_dir / "linear_taylor_bridge_errors_vs_q.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    return paths


def run(args: argparse.Namespace) -> Dict[str, str]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.params_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    q_grid = parse_float_csv(args.q_grid)
    df_tv, df_test, df_assess, predictor_cols, categorical_cols = _load_and_split_data(
        data_path=args.data_path,
        params=params,
        target_column=args.target_column,
        date_column=args.date_column,
        sample_frac=args.sample_frac,
        sample_seed=args.seed,
        assessment_year=int(args.assessment_year),
    )

    linear_pipeline_builder = lambda: build_model_pipeline(
        pred_vars=list(predictor_cols),
        cat_vars=list(categorical_cols),
        id_vars=params["model"]["predictor"]["id"],
    )

    X_tv = _cast_categoricals(df_tv[predictor_cols], categorical_cols)
    y_tv_log = np.log(df_tv[args.target_column].to_numpy(dtype=float))
    X_test = _cast_categoricals(df_test[predictor_cols], categorical_cols)
    y_test_log = np.log(df_test[args.target_column].to_numpy(dtype=float))

    frames = [
        _evaluate_linear_fit(
            fit_label="test_fit",
            data_source=args.data_source_label,
            assessment_year=int(args.assessment_year),
            X_train=X_tv,
            y_train_log=y_tv_log,
            eval_sets=[("test", X_test, y_test_log)],
            linear_pipeline_builder=linear_pipeline_builder,
            q_grid=q_grid,
        )
    ]

    if not df_assess.empty:
        df_pre = pd.concat([df_tv, df_test], ignore_index=True)
        X_pre = _cast_categoricals(df_pre[predictor_cols], categorical_cols)
        y_pre_log = np.log(df_pre[args.target_column].to_numpy(dtype=float))
        X_assess = _cast_categoricals(df_assess[predictor_cols], categorical_cols)
        y_assess_log = np.log(df_assess[args.target_column].to_numpy(dtype=float))
        frames.append(
            _evaluate_linear_fit(
                fit_label="assess_fit",
                data_source=args.data_source_label,
                assessment_year=int(args.assessment_year),
                X_train=X_pre,
                y_train_log=y_pre_log,
                eval_sets=[("assessment", X_assess, y_assess_log)],
                linear_pipeline_builder=linear_pipeline_builder,
                q_grid=q_grid,
            )
        )

    metrics_df = pd.concat(frames, ignore_index=True)
    metrics_path = out_dir / "linear_projection_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    verification_df = _write_verification_tables(metrics_df, out_dir)
    plot_paths = _plot_linear_diagnostics(metrics_df, out_dir)

    report = [
        "# Linear projection-path covariance experiment",
        "",
        "This run verifies the exact fixed-space projection identities on the training splits and reports out-of-sample behavior on test/assessment splits.",
        "",
        f"- Data source: `{args.data_source_label}`",
        f"- Assessment year: `{args.assessment_year}`",
        f"- q grid: `{','.join(str(q) for q in q_grid)}`",
        f"- Metric rows: {metrics_df.shape[0]}",
        "",
        "## Verification Summary",
        "",
    ]
    if not verification_df.empty:
        report.append(verification_df.to_markdown(index=False, floatfmt=".6g"))
    else:
        report.append("No covariance-path rows were available.")
    report.extend(["", "## Artifacts", "", f"- `{metrics_path.name}`", "- `linear_projection_theory_empirical_comparison.csv`", "- `linear_projection_verification_summary.csv`"])
    for path in plot_paths:
        report.append(f"- `{path.relative_to(out_dir)}`")
    report_path = out_dir / "linear_projection_report.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    log("done", out_dir=str(out_dir))
    return {
        "metrics": str(metrics_path),
        "verification": str(out_dir / "linear_projection_verification_summary.csv"),
        "comparison": str(out_dir / "linear_projection_theory_empirical_comparison.csv"),
        "report": str(report_path),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Exact linear projection-path covariance experiment.")
    p.add_argument("--data-source-label", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--assessment-year", type=int, required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--q-grid", default="1.00,0.90,0.80,0.70,0.60,0.50,0.40,0.30,0.20")
    p.add_argument("--params-path", default="params.yaml")
    p.add_argument("--target-column", default="meta_sale_price")
    p.add_argument("--date-column", default="meta_sale_date")
    p.add_argument("--sample-frac", type=float, default=None)
    p.add_argument("--seed", type=int, default=4050)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
