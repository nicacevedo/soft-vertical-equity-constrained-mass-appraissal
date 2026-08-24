#!/usr/bin/env python3
"""Paper-v6 pre-selection reporting: merge, figures, populate, QA. No selection."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from canonical_experiment import git_state, lgbm_params_hash, package_versions
from utils.motivation_utils import IAAO_PRD_RANGE, IAAO_VEI_RANGE

ROOT = REPO / "output" / "paper_v6_preselection"
CV_ROOT = ROOT
BASELINE_ROOT = ROOT / "baseline_reporting"
PREVIEW_ROOT = ROOT / "reporting_preview"
ANALYSIS_ROOT = ROOT / "analysis"
PAPER_OUT = ROOT / "paper_outputs"
FIG_OUT = PAPER_OUT / "figures"
TAB_OUT = PAPER_OUT / "tables"
SRC_OUT = PAPER_OUT / "sources"
MANIFEST_ROOT = ROOT / "manifests"
PAPER_TEX = REPO / "paper" / "paper_v6.tex"
PAPER_IMG = REPO / "paper" / "img" / "generated_v6_preselection"
SECTION2_JSON = ROOT / "section2_lgbm_config.json"
SELECTION_TERMS = (
    r"\bbest\b",
    r"\boptimal\b",
    r"\bpreferred\b",
    r"\bselected\b",
    r"we select",
    r"recommended rho",
    r"feasible winner",
    r"\bnash\b",
    r"\butopia\b",
)
ANCHORS = (0.0, 0.1, 1.0, 10.0, 100.0)
DIRECT = "LGBCovPenalty"
SURROGATE = "LGBSmoothPenalty"
NATIVE = "LGBMRegressor"
LINEAR = "LinearRegression"
METRIC_MAP = {
    "R2_price": "R2_price",
    "MAE_price": "MAE_price",
    "MAPE": "MAPE",
    "RMSE_log": "RMSE_log",
    "Median ratio": "median_ratio",
    "Mean ratio": "mean_ratio",
    "W. Mean ratio": "weighted_mean_ratio",
    "COD": "COD",
    "COV_IAAO": "COV",
    "PRD": "PRD",
    "PRB": "PRB",
    "MKI": "MKI",
    "VEI": "VEI",
    "Beta_log": "Beta_log",
    "Cov_log_residual_log_price": "Cov_log_residual_log_price",
    "dCor_e_y": "dCor_e_y",
}
PATH_METRICS = list(METRIC_MAP.values())
DIRECT_COLOR = "#1D4ED8"
SURR_COLOR = "#C2410C"
NATIVE_COLOR = "#111827"
LINEAR_COLOR = "#6B7280"
RHO_COLORS = ("#93C5FD", "#60A5FA", "#2563EB", "#1E40AF", "#1E3A8A")


def _parse_config(raw: Any) -> Dict[str, Any]:
    if raw is None or (isinstance(raw, float) and not np.isfinite(raw)):
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    text = str(raw).strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _rho_from_row(row: pd.Series) -> float:
    if "rho" in row.index and pd.notna(row.get("rho")):
        try:
            return float(row["rho"])
        except (TypeError, ValueError):
            pass
    cfg = _parse_config(row.get("model_config_json", row.get("model_config_json", None)))
    if "rho" in cfg:
        try:
            return float(cfg["rho"])
        except (TypeError, ValueError):
            return float("nan")
    return float("nan")


def _family_from_name(name: str) -> str:
    if name == LINEAR:
        return "Linear"
    if name == NATIVE:
        return "LightGBM"
    if name == DIRECT:
        return "Direct"
    if name == SURROGATE:
        return "Surrogate"
    return str(name)


def _nearest(grid: Sequence[float], target: float) -> float:
    arr = np.asarray(list(grid), dtype=float)
    return float(arr[int(np.argmin(np.abs(arr - float(target))))])


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _standardize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    name_col = "model_name" if "model_name" in out.columns else "name"
    out["model_name"] = out[name_col].astype(str)
    out["family"] = out["model_name"].map(_family_from_name)
    if "config_id" not in out.columns and "config_id" in out.columns:
        out["config_id"] = out["config_id"]
    out["config_id"] = out.get("config_id", pd.Series([""] * len(out))).astype(str)
    out["rho"] = out.apply(_rho_from_row, axis=1)
    for src, dst in METRIC_MAP.items():
        if src in out.columns:
            out[dst] = pd.to_numeric(out[src], errors="coerce")
    return out


def _concat_parquets(paths: Iterable[Path]) -> pd.DataFrame:
    files = [p for p in paths if p.is_file()]
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)


def load_baseline_metrics() -> pd.DataFrame:
    rows = []
    for split, fname in (("heldout", "test_metrics.csv"), ("forward_2025", "assess_metrics.csv")):
        matches = list(BASELINE_ROOT.glob(f"analysis/**/{fname}"))
        for path in matches:
            df = pd.read_csv(path)
            df = _standardize_metrics(df)
            df["evaluation"] = split
            df["source"] = "baseline_reporting"
            rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_oos_shards(preview_root: Path = PREVIEW_ROOT) -> pd.DataFrame:
    frames = []
    mapping = {
        "heldout": "test_run_metrics",
        "forward_2025": "assess_run_metrics",
    }
    for eval_name, shard in mapping.items():
        files = list(preview_root.glob(f"{eval_name}/**/{shard}/*.parquet"))
        if not files:
            continue
        df = _concat_parquets(files)
        df = _standardize_metrics(df)
        df["evaluation"] = eval_name
        df["source"] = "oos_preview"
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_cv_runs(cv_root: Path = CV_ROOT) -> pd.DataFrame:
    files = list(cv_root.glob("runs/**/*.parquet"))
    if not files:
        return pd.DataFrame()
    df = _concat_parquets(files)
    df = _standardize_metrics(df)
    df["evaluation"] = "cv_fold"
    df["source"] = "cv"
    if "fold_id" in df.columns:
        df["fold_id"] = pd.to_numeric(df["fold_id"], errors="coerce").astype("Int64")
    return df


def expected_canonical_rhos() -> List[float]:
    positives = np.geomspace(0.1, 100.0, 50)
    return [0.0] + [float(x) for x in positives]


def build_combined_table(cv: pd.DataFrame, oos: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    penalty_oos = oos.loc[oos["family"].isin(["Direct", "Surrogate"])].copy() if not oos.empty else pd.DataFrame()
    base = baseline.copy() if not baseline.empty else pd.DataFrame()
    rows: List[Dict[str, Any]] = []

    def _metric_value(df: pd.DataFrame, family: str, rho: Optional[float], metric: str, evaluation: str) -> float:
        if df.empty:
            return float("nan")
        sub = df.loc[df["family"] == family]
        if rho is not None:
            sub = sub.loc[np.isclose(sub["rho"].astype(float), float(rho), atol=1e-12, equal_nan=False)]
        else:
            sub = sub.loc[sub["rho"].isna() | (sub["rho"] == 0) & (family in {"Linear", "LightGBM"})]
            if family in {"Linear", "LightGBM"}:
                sub = df.loc[(df["family"] == family) & (df["evaluation"] == evaluation)]
        sub = sub.loc[sub["evaluation"] == evaluation] if "evaluation" in sub.columns else sub
        if sub.empty or metric not in sub.columns:
            return float("nan")
        return float(pd.to_numeric(sub[metric], errors="coerce").iloc[0])

    records: List[Tuple[str, Optional[float], str]] = [("Linear", None, LINEAR), ("LightGBM", None, NATIVE)]
    rhos = expected_canonical_rhos()
    for rho in rhos:
        records.append(("Direct", float(rho), DIRECT))
        records.append(("Surrogate", float(rho), SURROGATE))

    git = git_state()
    freeze = _load_json(CV_ROOT / "frozen_baseline.json")
    spec = _load_json(CV_ROOT / "experiment_spec.json")
    baseline_hash = freeze.get("lgbm_params_sha256") or spec.get("frozen_baseline_hash")
    grid_hash = spec.get("canonical_model_grid_hash") or spec.get("model_grid_hash")

    for family, rho, model_name in records:
        row: Dict[str, Any] = {
            "family": family,
            "rho": rho if rho is not None else np.nan,
            "model_name": model_name,
            "config_id": "",
            "git_commit": git.get("git_commit"),
            "git_dirty": git.get("git_dirty"),
            "git_diff_sha256": git.get("git_diff_sha256"),
            "baseline_hash": baseline_hash,
            "grid_hash": grid_hash,
            "stage": "preselection_path",
        }
        src_cv = cv.loc[cv["family"] == family].copy() if not cv.empty else pd.DataFrame()
        if rho is not None and not src_cv.empty:
            src_cv = src_cv.loc[np.isclose(src_cv["rho"].astype(float), float(rho), atol=1e-10)]
        if not src_cv.empty and "config_id" in src_cv.columns:
            row["config_id"] = str(src_cv["config_id"].iloc[0])
        for metric in PATH_METRICS:
            fold_vals = []
            if not src_cv.empty and metric in src_cv.columns and "fold_id" in src_cv.columns:
                for fold in range(7):
                    part = src_cv.loc[src_cv["fold_id"] == fold, metric]
                    val = float(pd.to_numeric(part, errors="coerce").iloc[0]) if not part.empty else float("nan")
                    row[f"fold_{fold + 1}"] = row.get(f"fold_{fold + 1}", {})
                    # store per metric with suffix
                    row[f"{metric}__fold_{fold + 1}"] = val
                    if np.isfinite(val):
                        fold_vals.append(val)
            row[f"{metric}__CV_mean"] = float(np.mean(fold_vals)) if fold_vals else float("nan")
            row[f"{metric}__CV_sd"] = float(np.std(fold_vals, ddof=1)) if len(fold_vals) > 1 else float("nan")
            oos_src = penalty_oos if family in {"Direct", "Surrogate"} else base
            row[f"{metric}__heldout"] = _metric_value(oos_src if family in {"Direct", "Surrogate"} else base, family, rho, metric, "heldout")
            row[f"{metric}__forward_2025"] = _metric_value(oos_src if family in {"Direct", "Surrogate"} else base, family, rho, metric, "forward_2025")
        rows.append(row)
    return pd.DataFrame(rows)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.color": "#E5E7EB",
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
        }
    )


def _save(fig: plt.Figure, stem: str) -> None:
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    PAPER_IMG.mkdir(parents=True, exist_ok=True)
    pdf = FIG_OUT / f"{stem}.pdf"
    png = FIG_OUT / f"{stem}.png"
    fig.savefig(pdf)
    fig.savefig(png)
    shutil.copy2(pdf, PAPER_IMG / f"{stem}.pdf")
    shutil.copy2(png, PAPER_IMG / f"{stem}.png")
    plt.close(fig)


def plot_mechanism(oos: pd.DataFrame) -> None:
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.8), constrained_layout=True)
    metrics = [("Beta_log", r"$\beta_{\log}$"), ("dCor_e_y", r"$\mathrm{dCor}(e,y)$")]
    families = [("Direct", DIRECT_COLOR), ("Surrogate", SURR_COLOR)]
    for col, (fam, color) in enumerate(families):
        sub = oos.loc[oos["family"] == fam].copy()
        for row, (metric, ylab) in enumerate(metrics):
            ax = axes[row, col]
            for eval_name, ls, lw in (("heldout", "-", 1.8), ("forward_2025", "--", 1.5)):
                part = sub.loc[sub["evaluation"] == eval_name].sort_values("rho")
                if part.empty or metric not in part.columns:
                    continue
                zero = part.loc[np.isclose(part["rho"], 0.0)]
                pos = part.loc[part["rho"] > 0]
                if not zero.empty:
                    ax.scatter([0.07], zero[metric], color=color, s=18, zorder=3)
                if not pos.empty:
                    ax.plot(pos["rho"], pos[metric], color=color, ls=ls, lw=lw, label=eval_name.replace("_", " "))
            ax.set_xscale("log")
            ax.set_xlim(0.06, 120)
            ax.set_xlabel(r"$\rho$")
            ax.set_ylabel(ylab)
            ax.set_title(fam)
            if metric == "Beta_log":
                ax.axhline(0.0, color="#111827", lw=0.8, ls=":")
            if row == 0 and col == 1:
                ax.legend(frameon=False, loc="best")
    _save(fig, "mechanism_vs_rho")


def plot_accuracy_equity(oos: pd.DataFrame, baseline: pd.DataFrame) -> None:
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.2), constrained_layout=True)
    evals = ("heldout", "forward_2025")
    ymetrics = [("PRD", r"PRD", IAAO_PRD_RANGE), ("VEI", r"VEI", IAAO_VEI_RANGE)]
    for r, ev in enumerate(evals):
        for c, (metric, ylab, band) in enumerate(ymetrics):
            ax = axes[r, c]
            ax.axhspan(band[0], band[1], color="#D1FAE5", alpha=0.45, lw=0)
            for fam, color, marker in (("Direct", DIRECT_COLOR, "o"), ("Surrogate", SURR_COLOR, "s")):
                part = oos.loc[(oos["family"] == fam) & (oos["evaluation"] == ev)].sort_values("rho")
                if part.empty:
                    continue
                ax.plot(part["R2_price"], part[metric], color=color, marker=marker, ms=3, lw=1.4, label=fam)
                if len(part) > 1:
                    x0, y0 = float(part["R2_price"].iloc[0]), float(part[metric].iloc[0])
                    x1, y1 = float(part["R2_price"].iloc[-1]), float(part[metric].iloc[-1])
                    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=color, lw=0.6))
            b = baseline.loc[baseline["evaluation"] == ev] if not baseline.empty else pd.DataFrame()
            for fam, color, marker, label in (
                ("LightGBM", NATIVE_COLOR, "D", "LightGBM"),
                ("Linear", LINEAR_COLOR, "P", "Linear"),
            ):
                bb = b.loc[b["family"] == fam]
                if not bb.empty and metric in bb.columns:
                    ax.scatter(bb["R2_price"], bb[metric], color=color, marker=marker, s=28, zorder=4, label=label)
            ax.set_xlabel(r"$R^2_P$")
            ax.set_ylabel(ylab)
            title_ev = "Held-out" if ev == "heldout" else "2025 forward"
            ax.set_title(f"{title_ev}: $R^2_P$ vs {ylab}")
            if r == 0 and c == 1:
                ax.legend(frameon=False, loc="best", ncol=2)
    _save(fig, "accuracy_equity_trajectories")


def plot_fold_stability(cv: pd.DataFrame) -> None:
    if cv.empty:
        return
    _style()
    metrics = [("R2_price", r"$R^2_P$"), ("PRD", "PRD"), ("VEI", "VEI"), ("Beta_log", r"$\beta_{\log}$")]
    fig, axes = plt.subplots(4, 2, figsize=(8.4, 9.2), constrained_layout=True)
    for c, (fam, color) in enumerate((("Direct", DIRECT_COLOR), ("Surrogate", SURR_COLOR))):
        sub = cv.loc[cv["family"] == fam]
        for r, (metric, ylab) in enumerate(metrics):
            ax = axes[r, c]
            if sub.empty:
                continue
            for fold, part in sub.groupby("fold_id"):
                part = part.sort_values("rho")
                pos = part.loc[part["rho"] > 0]
                ax.plot(pos["rho"], pos[metric], color=color, alpha=0.25, lw=0.8)
            mean = sub.groupby("rho", as_index=False)[metric].mean().sort_values("rho")
            pos = mean.loc[mean["rho"] > 0]
            ax.plot(pos["rho"], pos[metric], color=color, lw=2.0)
            ax.set_xscale("log")
            ax.set_xlabel(r"$\rho$")
            ax.set_ylabel(ylab)
            if r == 0:
                ax.set_title(fam)
    _save(fig, "cv_fold_stability")


def plot_metric_paths(oos: pd.DataFrame, metrics: Sequence[Tuple[str, str]], stem: str) -> None:
    _style()
    n = len(metrics)
    fig, axes = plt.subplots(n, 2, figsize=(8.4, 1.7 * n + 0.8), constrained_layout=True, sharex=True)
    if n == 1:
        axes = np.array([axes])
    for c, (fam, color) in enumerate((("Direct", DIRECT_COLOR), ("Surrogate", SURR_COLOR))):
        sub = oos.loc[oos["family"] == fam]
        for r, (metric, ylab) in enumerate(metrics):
            ax = axes[r, c]
            for ev, ls in (("heldout", "-"), ("forward_2025", "--")):
                part = sub.loc[sub["evaluation"] == ev].sort_values("rho")
                pos = part.loc[part["rho"] > 0]
                zero = part.loc[np.isclose(part["rho"], 0.0)]
                if not zero.empty:
                    ax.scatter([0.07], zero[metric], color=color, s=14)
                if not pos.empty:
                    ax.plot(pos["rho"], pos[metric], color=color, ls=ls, lw=1.5, label=ev.replace("_", " "))
            ax.set_xscale("log")
            ax.set_xlim(0.06, 120)
            ax.set_ylabel(ylab)
            if r == n - 1:
                ax.set_xlabel(r"$\rho$")
            if r == 0:
                ax.set_title(fam)
            if r == 0 and c == 1:
                ax.legend(frameon=False)
    _save(fig, stem)


def _load_oos_predictions(eval_name: str, family: str, rhos: Sequence[float]) -> Dict[float, pd.DataFrame]:
    shard = "test_run_predictions" if eval_name == "heldout" else "assess_run_predictions"
    files = list(PREVIEW_ROOT.glob(f"{eval_name}/**/{shard}/*.parquet"))
    out: Dict[float, pd.DataFrame] = {}
    if not files:
        return out
    df = _concat_parquets(files)
    df = _standardize_metrics(df)
    df = df.loc[df["family"] == family]
    for target in rhos:
        if df.empty:
            continue
        part = df.loc[np.isclose(df["rho"].astype(float), float(target), atol=1e-8)]
        if part.empty:
            continue
        out[float(target)] = part
    return out


def plot_ratio_shape(oos: pd.DataFrame) -> None:
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.4), sharex=True, sharey=True, constrained_layout=True)
    evals = ("heldout", "forward_2025")
    fams = ("Direct", "Surrogate")
    display_rhos = list(ANCHORS)
    for r, ev in enumerate(evals):
        for c, fam in enumerate(fams):
            ax = axes[r, c]
            preds = _load_oos_predictions(ev, fam, display_rhos)
            for color, rho in zip(RHO_COLORS, display_rhos):
                part = preds.get(float(_nearest(preds.keys(), rho))) if preds else None
                # keys may not match; try exact nearest among available
                if preds:
                    have = list(preds.keys())
                    use = _nearest(have, rho)
                    part = preds.get(use)
                if part is None or part.empty:
                    continue
                y_true = pd.to_numeric(part.get("y_true", part.get("y_true")), errors="coerce")
                y_pred = pd.to_numeric(part.get("y_pred", part.get("y_pred")), errors="coerce")
                ratio = (y_pred / y_true).to_numpy()
                order = np.argsort(y_true.to_numpy())
                y_sorted = y_true.to_numpy()[order]
                r_sorted = ratio[order]
                n = len(y_sorted)
                if n < 30:
                    continue
                edges = np.array_split(np.arange(n), 30)
                xs, ys = [], []
                for idx in edges:
                    xs.append(float(np.median(y_sorted[idx])))
                    ys.append(float(np.median(r_sorted[idx])))
                ax.plot(xs, ys, color=color, lw=1.4, marker="o", ms=2, label=fr"$\rho$={rho:g}")
            ax.axhline(1.0, color="#111827", ls="--", lw=0.8)
            ax.set_xscale("log")
            ax.set_ylim(0.55, 1.45)
            ev_lab = "Held-out" if ev == "heldout" else "2025 forward"
            ax.set_title(f"{fam} / {ev_lab}")
            if c == 0:
                ax.set_ylabel("Valuation-to-sale ratio")
            if r == 1:
                ax.set_xlabel("Sale price")
            if r == 0 and c == 1:
                ax.legend(frameon=False, loc="best")
    _save(fig, "ratio_shape_evolution")


def write_tables(combined: pd.DataFrame, oos: pd.DataFrame, baseline: pd.DataFrame) -> None:
    TAB_OUT.mkdir(parents=True, exist_ok=True)
    SRC_OUT.mkdir(parents=True, exist_ok=True)
    combined.to_csv(TAB_OUT / "combined_path_table.csv", index=False)
    combined.to_parquet(TAB_OUT / "combined_path_table.parquet", index=False)
    combined.to_csv(SRC_OUT / "combined_path_table.csv", index=False)

    def _fmt(x: Any, nd: int = 3) -> str:
        try:
            v = float(x)
        except (TypeError, ValueError):
            return "---"
        if not np.isfinite(v):
            return "---"
        return f"{v:.{nd}f}"

    # rho=0 control uses OOS + baseline LightGBM
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Sample & Model & $R^2_P$ & $\operatorname{RMSE}_{\log P}$ & $\beta_{\log}$ & Mean $|\Delta\widehat y|$ & Max $|\Delta\widehat y|$ \\",
        r"\midrule",
    ]
    for ev, label in (("heldout", "Held-out"), ("forward_2025", "2025 forward")):
        b = baseline.loc[(baseline["evaluation"] == ev) & (baseline["family"] == "LightGBM")]
        for fam, name in (("LightGBM", "Ordinary LightGBM"), ("Direct", r"Direct, $\rho=0$"), ("Surrogate", r"Surrogate, $\rho=0$")):
            src = b if fam == "LightGBM" else oos.loc[(oos["evaluation"] == ev) & (oos["family"] == fam) & np.isclose(oos["rho"].fillna(-1), 0.0)]
            if src.empty:
                lines.append(f"{label} & {name} & " + " & ".join(["---"] * 5) + r" \\")
                continue
            r2 = _fmt(src["R2_price"].iloc[0])
            rmse = _fmt(src["RMSE_log"].iloc[0])
            beta = _fmt(src["Beta_log"].iloc[0])
            lines.append(f"{label} & {name} & {r2} & {rmse} & {beta} & --- & --- \\\\")
        if ev == "heldout":
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB_OUT / "rho_zero_control.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    grid = sorted(set(float(x) for x in oos["rho"].dropna().unique())) if not oos.empty else list(ANCHORS)
    chosen = [_nearest(grid, a) if grid else a for a in ANCHORS]
    anchor_lines = [
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        r"Family & $\rho$ & $R^2_P$ & MAE & PRD & PRB & MKI & VEI & $\beta_{\log}$ & dCor \\",
        r"\midrule",
        r"\multicolumn{10}{l}{\textit{Panel A: held-out evaluation}} \\",
    ]

    def _anchor_row(ev: str, fam: str, rho: Optional[float], display: str) -> str:
        if fam == "LightGBM":
            src = baseline.loc[(baseline["evaluation"] == ev) & (baseline["family"] == "LightGBM")]
            rho_tex = "--"
        else:
            src = oos.loc[(oos["evaluation"] == ev) & (oos["family"] == fam) & np.isclose(oos["rho"].astype(float), float(rho), atol=1e-8)]
            rho_tex = f"{float(rho):g}"
        if src.empty:
            return f"{display} & {rho_tex} & " + " & ".join(["---"] * 8) + r" \\"
        vals = [
            _fmt(src["R2_price"].iloc[0]),
            _fmt(src["MAE_price"].iloc[0], 0) if "MAE_price" in src.columns else "---",
            _fmt(src["PRD"].iloc[0]),
            _fmt(src["PRB"].iloc[0]),
            _fmt(src["MKI"].iloc[0]),
            _fmt(src["VEI"].iloc[0], 1),
            _fmt(src["Beta_log"].iloc[0]),
            _fmt(src["dCor_e_y"].iloc[0]),
        ]
        return f"{display} & {rho_tex} & " + " & ".join(vals) + r" \\"

    for ev, panel in (("heldout", "A"), ("forward_2025", "B")):
        if ev == "forward_2025":
            anchor_lines += [r"\midrule", r"\multicolumn{10}{l}{\textit{Panel B: 2025 forward evaluation}} \\"]
        anchor_lines.append(_anchor_row(ev, "LightGBM", None, "Ordinary LightGBM"))
        for rho, lab in zip(chosen, [r"0", r"$\approx0.1$", r"$\approx1$", r"$\approx10$", r"$100$"]):
            anchor_lines.append(_anchor_row(ev, "Direct", rho, "Direct"))
        for rho in chosen:
            anchor_lines.append(_anchor_row(ev, "Surrogate", rho, "Surrogate"))
    anchor_lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB_OUT / "path_anchor_summary.tex").write_text("\n".join(anchor_lines) + "\n", encoding="utf-8")


def cmd_cv_qa() -> int:
    cv = load_cv_runs()
    payload: Dict[str, Any] = {"selection_performed": False}
    n = 0 if cv.empty else len(cv)
    payload["n_run_rows"] = int(n)
    expected = 728
    payload["expected_runs"] = expected
    payload["complete"] = bool(n >= expected)
    if not cv.empty:
        folds = sorted(cv["fold_id"].dropna().unique().tolist())
        payload["folds"] = [int(x) for x in folds]
        payload["n_folds"] = len(folds)
        fam = cv["family"].value_counts().to_dict()
        payload["family_counts"] = {str(k): int(v) for k, v in fam.items()}
        # truncation flag at endpoints
        warns = []
        for family in ("Direct", "Surrogate"):
            sub = cv.loc[cv["family"] == family]
            if sub.empty:
                continue
            mean = sub.groupby("rho")["Beta_log"].mean().sort_index()
            pos = mean.loc[mean.index > 0]
            if len(pos) >= 2:
                last = float(pos.iloc[-1])
                prev = float(pos.iloc[-2])
                if abs(last - prev) > 0.02:
                    warns.append(f"{family} Beta_log still moving at rho={float(pos.index[-1]):.3g}")
        payload["endpoint_truncation_warnings"] = warns
        payload["blocker"] = bool(warns)
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    _write_json(MANIFEST_ROOT / "cv_qa.json", payload)
    print(json.dumps(payload, indent=2))
    if n < expected:
        print("FAIL CV QA: incomplete runs")
        return 1
    print("PASS CV QA (integrity only; no selection)")
    return 0


def cmd_preview() -> int:
    oos = load_oos_shards()
    baseline = load_baseline_metrics()
    if oos.empty:
        print("FAIL preview: no OOS shards")
        return 1
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    oos.to_csv(ANALYSIS_ROOT / "oos_preview_metrics.csv", index=False)
    oos.to_parquet(ANALYSIS_ROOT / "oos_preview_metrics.parquet", index=False)
    plot_mechanism(oos)
    plot_accuracy_equity(oos, baseline)
    plot_ratio_shape(oos)
    plot_metric_paths(oos, [("R2_price", r"$R^2_P$"), ("MAE_price", "MAE"), ("MAPE", "MAPE"), ("RMSE_log", r"RMSE$_{\log}$")], "predictive_metric_paths")
    plot_metric_paths(oos, [("PRB", "PRB"), ("MKI", "MKI")], "prb_mki_paths")
    plot_metric_paths(
        oos,
        [("median_ratio", "Median ratio"), ("mean_ratio", "Mean ratio"), ("weighted_mean_ratio", "Weighted mean"), ("COD", "COD"), ("COV", "COV")],
        "level_uniformity_paths",
    )
    write_tables(pd.DataFrame(), oos, baseline)
    print("PASS preview figures/tables written (no selection)")
    return 0


def cmd_merge() -> int:
    cv = load_cv_runs()
    oos = load_oos_shards()
    baseline = load_baseline_metrics()
    combined = build_combined_table(cv, oos, baseline)
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    combined.to_csv(ANALYSIS_ROOT / "combined_path_table.csv", index=False)
    combined.to_parquet(ANALYSIS_ROOT / "combined_path_table.parquet", index=False)
    write_tables(combined, oos, baseline)
    plot_mechanism(oos)
    plot_accuracy_equity(oos, baseline)
    plot_ratio_shape(oos)
    plot_fold_stability(cv)
    plot_metric_paths(oos, [("R2_price", r"$R^2_P$"), ("MAE_price", "MAE"), ("MAPE", "MAPE"), ("RMSE_log", r"RMSE$_{\log}$")], "predictive_metric_paths")
    plot_metric_paths(oos, [("PRB", "PRB"), ("MKI", "MKI")], "prb_mki_paths")
    plot_metric_paths(
        oos,
        [("median_ratio", "Median ratio"), ("mean_ratio", "Mean ratio"), ("weighted_mean_ratio", "Weighted mean"), ("COD", "COD"), ("COV", "COV")],
        "level_uniformity_paths",
    )
    print(f"PASS merge n_rows={len(combined)}")
    return 0


def _replace_tabular(tex: str, label: str, new_tabular: str) -> str:
    pattern = rf"(\\label\{{{re.escape(label)}\}}\s*)\\begin\{{tabular\}}.*?\\end\{{tabular\}}"
    repl = r"\1" + new_tabular.replace("\\", "\\\\")
    updated, n = re.subn(pattern, repl, tex, count=1, flags=re.S)
    if n != 1:
        print(f"WARN could not replace tabular for {label}")
        return tex
    return updated


def _replace_includegraphics(tex: str, label: str, rel_path: str) -> str:
    # replace first includegraphics after the label by searching backwards from label's figure
    return tex.replace(
        "results_reference_assets/ratio_shape_layout_reference.jpg",
        rel_path if "ratio_shape" in rel_path else "results_reference_assets/ratio_shape_layout_reference.jpg",
        1 if "ratio_shape" in rel_path else 0,
    )


def cmd_populate() -> int:
    combined_path = ANALYSIS_ROOT / "combined_path_table.csv"
    oos = load_oos_shards()
    baseline = load_baseline_metrics()
    tex = PAPER_TEX.read_text(encoding="utf-8")
    rho0 = (TAB_OUT / "rho_zero_control.tex").read_text(encoding="utf-8") if (TAB_OUT / "rho_zero_control.tex").is_file() else ""
    anchors = (TAB_OUT / "path_anchor_summary.tex").read_text(encoding="utf-8") if (TAB_OUT / "path_anchor_summary.tex").is_file() else ""
    if rho0:
        tex = _replace_tabular(tex, "tab:rho_zero_control", rho0)
    if anchors:
        tex = _replace_tabular(tex, "tab:path_anchor_summary", anchors)
    replacements = {
        "results_reference_assets/ratio_shape_layout_reference.jpg": "img/generated_v6_preselection/ratio_shape_evolution.pdf",
        "results_reference_assets/metric_path_layout_reference.jpg": "img/generated_v6_preselection/mechanism_vs_rho.pdf",
        "results_reference_assets/PRD_vs_R2_layout_reference.pdf": "img/generated_v6_preselection/accuracy_equity_trajectories.pdf",
        "results_reference_assets/VEI_vs_R2_layout_reference.pdf": "img/generated_v6_preselection/accuracy_equity_trajectories.pdf",
        "results_reference_assets/multimetric_path_layout_reference.png": "img/generated_v6_preselection/predictive_metric_paths.pdf",
        "results_reference_assets/vei_group_layout_reference.png": "img/generated_v6_preselection/cv_fold_stability.pdf",
    }
    for old, new in replacements.items():
        tex = tex.replace(old, new)
    # descriptive, selection-free notes
    note = (
        "As $\\rho$ increases along the prespecified grid, $\\beta_{\\log}$ and "
        "$\\operatorname{dCor}(e,y)$ are reported for the complete Direct and Surrogate paths "
        "on held-out and 2025 samples. No configuration is selected."
    )
    tex = tex.replace(
        r"\todo{Populate Table~\ref{tab:rho_zero_control} from the complete held-out and 2025",
        note + r" \todo{Populate Table~\ref{tab:rho_zero_control} from the complete held-out and 2025",
        1,
    )
    PAPER_TEX.write_text(tex, encoding="utf-8")
    print("PASS paper populate (no selection language added as a claim)")
    return 0


def cmd_final_qa() -> int:
    oos = load_oos_shards()
    cv = load_cv_runs()
    problems = []
    if oos.empty:
        problems.append("missing OOS metrics")
    else:
        for fam in ("Direct", "Surrogate"):
            for ev in ("heldout", "forward_2025"):
                n = int(((oos["family"] == fam) & (oos["evaluation"] == ev)).sum())
                if n < 51:
                    problems.append(f"{fam} {ev} has {n} rows, expected 51")
    if not cv.empty and len(cv) < 728:
        problems.append(f"CV rows {len(cv)} < 728")
    tex = PAPER_TEX.read_text(encoding="utf-8")
    results = tex.split(r"\section{Results}", 1)[-1].split(r"\section{Discussion and Conclusions}", 1)[0]
    for pat in SELECTION_TERMS:
        if re.search(pat, results, flags=re.I):
            problems.append(f"selection-like term in Results: {pat}")
    if "results_reference_assets/" in results:
        problems.append("Results still references old layout-only assets")
    compile_ok = False
    try:
        proc = subprocess.run(
            ["latexmk", "-pdf", "-interaction=nonstopmode", "paper_v6.tex"],
            cwd=str(REPO / "paper"),
            check=False,
        )
        compile_ok = proc.returncode == 0
    except FileNotFoundError:
        proc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "paper_v6.tex"],
            cwd=str(REPO / "paper"),
            check=False,
        )
        compile_ok = proc.returncode == 0
    if not compile_ok:
        problems.append("paper compile failed")
    pdf = REPO / "paper" / "paper_v6.pdf"
    if pdf.is_file():
        shutil.copy2(pdf, PAPER_OUT / "paper_v6.pdf")
    payload = {"problems": problems, "compile_ok": compile_ok, "selection_performed": False}
    _write_json(MANIFEST_ROOT / "final_qa.json", payload)
    print(json.dumps(payload, indent=2))
    if problems:
        print("FAIL final QA")
        return 1
    print("PASS final QA")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["cv-qa", "preview", "merge", "populate", "final-qa"])
    args = p.parse_args()
    if args.command == "cv-qa":
        return cmd_cv_qa()
    if args.command == "preview":
        return cmd_preview()
    if args.command == "merge":
        return cmd_merge()
    if args.command == "populate":
        return cmd_populate()
    return cmd_final_qa()


if __name__ == "__main__":
    raise SystemExit(main())
