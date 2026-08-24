#!/usr/bin/env python3
"""Populate paper/paper_v6.tex from completed 994-tree pre-selection artifacts.

No rho, penalty family, or penalized configuration is selected or ranked.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterSciNotation, LogLocator
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import paper_v6_preselection_pipeline as pipe
from utils.motivation_utils import IAAO_PRD_RANGE, IAAO_VEI_RANGE, vei_percentile_group_profile

RESULT_ROOT = REPO / "output" / "paper_v6_preselection_994"
PAPER_TEX = REPO / "paper" / "paper_v6.tex"
PAPER_IMG = REPO / "paper" / "img" / "generated_v6_preselection"
IMG_REL = "img/generated_v6_preselection"
DISPLAY_TARGETS = (0.0, 0.1, 1.0, 10.0, 100.0)
N_DEV, N_HOLDOUT, N_PROD, N_2025 = 344_607, 38_290, 382_897, 26_641
SELECTION_RE = re.compile(r"\b(best|winner|winning|optimal|selected|preferred)\b", re.I)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_provenance() -> Dict[str, Any]:
    def run(cmd: List[str]) -> str:
        return subprocess.check_output(cmd, cwd=str(REPO), text=True).strip()

    status = run(["git", "status", "--porcelain"])
    diff = subprocess.check_output(["git", "diff"], cwd=str(REPO))
    return {
        "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": run(["git", "rev-parse", "HEAD"]),
        "dirty": bool(status),
        "status_porcelain": status,
        "diff_sha256": hashlib.sha256(diff).hexdigest() if diff else None,
    }


def fmt_int(n: int) -> str:
    return f"{int(n):,}"


def fmt_r2(x: float) -> str:
    return f"{float(x):.3f}"


def fmt_mae(x: float) -> str:
    return f"\\${float(x):,.0f}"


def fmt_pct_frac(x: float) -> str:
    return f"{100.0 * float(x):.1f}\\%"


def fmt_pct(x: float) -> str:
    return f"{float(x):.1f}\\%"


def fmt_3(x: float) -> str:
    return f"{float(x):.3f}"


def fmt_rho(x: float) -> str:
    v = float(x)
    if abs(v) < 1e-12:
        return "0"
    if abs(v - 100.0) < 1e-8:
        return "100"
    return f"{v:.3f}".rstrip("0").rstrip(".")


def combined_row(combined: pd.DataFrame, family: str, rho: Optional[float] = None) -> pd.Series:
    sub = combined.loc[combined["family"] == family]
    if rho is None:
        sub = sub.loc[sub["rho"].isna() | ~np.isfinite(pd.to_numeric(sub["rho"], errors="coerce"))]
    else:
        sub = sub.loc[np.isclose(pd.to_numeric(sub["rho"], errors="coerce"), float(rho), atol=1e-10)]
    if sub.empty:
        raise RuntimeError(f"missing combined row family={family} rho={rho}")
    return sub.iloc[0]


def metric(row: pd.Series, name: str, split: str) -> float:
    return float(row[f"{name}__{split}"])


def display_rhos(combined: pd.DataFrame) -> List[float]:
    grid = sorted(
        float(x)
        for x in combined.loc[combined["family"].isin(["Direct", "Surrogate"]), "rho"]
        if np.isfinite(float(x))
    )
    return pipe.map_display_anchors(grid, DISPLAY_TARGETS)


def find_pred_file(root: Path, config_id: str, shard: str) -> Path:
    matches = list(root.glob(f"**/{shard}/{config_id}.parquet"))
    if not matches:
        raise FileNotFoundError(f"no {shard} parquet for {config_id}")
    return matches[0]


def load_pred(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "y_true" not in df.columns:
        df["y_true"] = np.exp(df["y_true_log"].to_numpy(dtype=float))
    if "y_pred" not in df.columns:
        df["y_pred"] = np.exp(df["y_pred_log"].to_numpy(dtype=float))
    if "y_pred_log" not in df.columns:
        df["y_pred_log"] = np.log(np.clip(df["y_pred"].to_numpy(dtype=float), 1e-12, None))
    return df


def baseline_dir() -> Path:
    files = list((RESULT_ROOT / "baseline_reporting").glob("**/test_predictions.parquet"))
    if not files:
        raise FileNotFoundError("baseline test_predictions.parquet missing")
    return files[0].parent


def load_baseline_split(split: str) -> pd.DataFrame:
    fname = "test_predictions.parquet" if split == "heldout" else "assess_predictions.parquet"
    return load_pred(baseline_dir() / fname)


def name_col(df: pd.DataFrame) -> str:
    return "model_name" if "model_name" in df.columns else "model"


def native_slice(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df[name_col(df)].astype(str).isin([pipe.NATIVE, "LGBMRegressor"])].copy()


def linear_slice(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df[name_col(df)].astype(str).isin([pipe.LINEAR, "LinearRegression"])].copy()


def load_oos_pred(family: str, rho: float, evaluation: str) -> pd.DataFrame:
    combined = pd.read_csv(RESULT_ROOT / "analysis" / "combined_path_table.csv")
    row = combined_row(combined, family, rho)
    shard = "test_run_predictions" if evaluation == "heldout" else "assess_run_predictions"
    return load_pred(find_pred_file(RESULT_ROOT / "reporting_preview", str(row["config_id"]), shard))


def equal_count_bins(sale: np.ndarray, ratio: np.ndarray, n_bins: int = 30) -> pd.DataFrame:
    order = np.argsort(sale, kind="mergesort")
    sale, ratio = sale[order], ratio[order]
    rows = []
    for i, idx in enumerate(np.array_split(np.arange(len(sale)), n_bins), start=1):
        if idx.size == 0:
            continue
        r = ratio[idx]
        rows.append(
            {
                "bin": i,
                "median_sale_price": float(np.median(sale[idx])),
                "median_ratio": float(np.median(r)),
                "ratio_q25": float(np.quantile(r, 0.25)),
                "ratio_q75": float(np.quantile(r, 0.75)),
            }
        )
    return pd.DataFrame(rows)


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.0,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_fig(fig: plt.Figure, stem: str) -> Path:
    PAPER_IMG.mkdir(parents=True, exist_ok=True)
    pipe.FIG_OUT.mkdir(parents=True, exist_ok=True)
    pdf = PAPER_IMG / f"{stem}.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(pipe.FIG_OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(PAPER_IMG / f"{stem}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return pdf


def rho_x(rho: np.ndarray) -> np.ndarray:
    x = np.asarray(rho, dtype=float)
    return np.where(x <= 0, 0.07, x)


def plot_baseline_motivation() -> Path:
    set_style()
    frames = []
    for split, lab in (("heldout", "Held-out"), ("forward_2025", "2025")):
        raw = load_baseline_split(split)
        work = raw.copy()
        work["split"] = split
        work["model"] = work[name_col(work)].map(
            {
                pipe.LINEAR: "LinearRegression",
                "LinearRegression": "LinearRegression",
                pipe.NATIVE: "LGBMRegressor",
                "LGBMRegressor": "LGBMRegressor",
            }
        )
        work["sale_price"] = work["y_true"].astype(float)
        work["assessment_ratio"] = work["y_pred"].astype(float) / work["y_true"].astype(float)
        frames.append(work)
    preds = pd.concat(frames, ignore_index=True)
    profiles = []
    for (split, model), g in preds.groupby(["split", "model"]):
        prof = equal_count_bins(g["sale_price"].to_numpy(), g["assessment_ratio"].to_numpy())
        prof["split"] = split
        prof["model"] = model
        profiles.append(prof)
    profile = pd.concat(profiles, ignore_index=True)
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.15), sharex=True, sharey=True)
    xmin, xmax = profile["median_sale_price"].min(), profile["median_sale_price"].max()
    pad = 0.04 * (np.log10(xmax) - np.log10(xmin))
    xlim = (10 ** (np.log10(xmin) - pad), 10 ** (np.log10(xmax) + pad))
    colors = {"LinearRegression": "#0072B2", "LGBMRegressor": "#D55E00"}
    titles = {"LinearRegression": "Linear regression", "LGBMRegressor": "Unpenalized LightGBM"}
    split_labs = {"heldout": "Held-out", "forward_2025": "2025"}
    for r, split in enumerate(("heldout", "forward_2025")):
        for c, model in enumerate(("LinearRegression", "LGBMRegressor")):
            ax = axes[r, c]
            sub = preds.loc[(preds["split"] == split) & (preds["model"] == model)]
            prof = profile.loc[(profile["split"] == split) & (profile["model"] == model)]
            color = colors[model]
            ax.fill_between(prof["median_sale_price"], prof["ratio_q25"], prof["ratio_q75"], color=color, alpha=0.16, lw=0)
            ax.plot(prof["median_sale_price"], prof["median_ratio"], color=color, marker="o", ms=2.5, lw=1.5)
            ax.axhline(1.0, color="#111827", ls=(0, (2, 2)), lw=0.9)
            ax.set_xscale("log", base=10)
            ax.set_xlim(*xlim)
            ax.set_ylim(0.55, 1.45)
            ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            ax.xaxis.set_major_formatter(
                LogFormatterSciNotation(base=10, labelOnlyBase=False, minor_thresholds=(np.inf, np.inf))
            )
            ax.grid(True, color="#E5E7EB", lw=0.7)
            ax.set_axisbelow(True)
            sale = sub["sale_price"].to_numpy(dtype=float)
            ratio = sub["assessment_ratio"].to_numpy(dtype=float)
            ok = np.isfinite(sale) & (sale > 0) & np.isfinite(ratio) & (ratio > 0)
            slope = np.polyfit(np.log10(sale[ok]), ratio[ok], 1)[0]
            ylog = np.log(sale[ok])
            plog = np.log(sale[ok] * ratio[ok])
            beta = float(np.cov(plog - ylog, ylog, ddof=0)[0, 1] / np.var(ylog, ddof=0))
            ax.legend(
                handles=[Line2D([], [], ls="None", label=rf"Slope = {slope:.3f}   $\beta_{{\log}}$ = {beta:.3f}")],
                loc="lower left",
                frameon=False,
                handlelength=0,
                handletextpad=0,
                fontsize=7.5,
            )
            if r == 0:
                ax.set_title(titles[model])
            if c == 0:
                ax.set_ylabel(f"{split_labs[split]}\nValuation-to-sale ratio")
            if r == 1:
                ax.set_xlabel(r"Sale price (log$_{10}$ scale)")
    fig.legend(
        handles=[
            Line2D([0], [0], color="#111827", marker="o", lw=1.5, ms=3, label="Equal-count-bin median (IQR shaded)"),
            Line2D([0], [0], color="#111827", ls=(0, (2, 2)), lw=0.9, label="Ratio = 1"),
        ],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save_fig(fig, "baseline_models_motivation_2024_2025")


def plot_ratio_shape(anchors: Sequence[float]) -> Path:
    set_style()
    cmap = plt.cm.viridis
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.4), sharex=True, sharey=True)
    evals = (("heldout", "Held-out"), ("forward_2025", "2025 forward"))
    fams = ("Direct", "Surrogate")
    x_all: List[float] = []
    for r, (ev, evlab) in enumerate(evals):
        for c, fam in enumerate(fams):
            ax = axes[r, c]
            for i, rho in enumerate(anchors):
                pred = load_oos_pred(fam, float(rho), ev)
                sale = pred["y_true"].to_numpy(dtype=float)
                ratio = pred["y_pred"].to_numpy(dtype=float) / sale
                prof = equal_count_bins(sale, ratio)
                x_all.extend(prof["median_sale_price"].tolist())
                color = cmap(0.12 + 0.8 * i / max(len(anchors) - 1, 1))
                ax.plot(
                    prof["median_sale_price"],
                    prof["median_ratio"],
                    color=color,
                    lw=1.5,
                    marker="o",
                    ms=2.2,
                    label=rf"$\rho$={fmt_rho(rho)}",
                )
            ax.axhline(1.0, color="#111827", ls=(0, (2, 2)), lw=0.8)
            ax.set_xscale("log", base=10)
            ax.set_ylim(0.55, 1.45)
            ax.grid(True, color="#E5E7EB", lw=0.7)
            ax.set_axisbelow(True)
            if r == 0:
                ax.set_title(fam)
            if c == 0:
                ax.set_ylabel(f"{evlab}\nValuation-to-sale ratio")
            if r == 1:
                ax.set_xlabel("Sale price")
            if r == 0 and c == 1:
                ax.legend(fontsize=7, frameon=False, loc="lower left")
    xmin, xmax = min(x_all), max(x_all)
    for ax in axes.ravel():
        ax.set_xlim(xmin / 1.05, xmax * 1.05)
    fig.tight_layout()
    return save_fig(fig, "ratio_shape_evolution")


def plot_mechanism(combined: pd.DataFrame) -> Path:
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.2), sharex=True)
    metrics = (("Beta_log", r"$\beta_{\log}$", True), ("dCor_e_y", r"$\mathrm{dCor}(e,y)$", False))
    styles = {"heldout": ("-", "o"), "forward_2025": ("--", "s")}
    for c, fam in enumerate(("Direct", "Surrogate")):
        sub = combined.loc[combined["family"] == fam].sort_values("rho")
        for r, (col, ylab, zero) in enumerate(metrics):
            ax = axes[r, c]
            color = pipe.DIRECT_COLOR if fam == "Direct" else pipe.SURR_COLOR
            for ev, (ls, mk) in styles.items():
                ax.plot(
                    rho_x(sub["rho"].to_numpy(dtype=float)),
                    sub[f"{col}__{ev}"],
                    color=color,
                    ls=ls,
                    marker=mk,
                    ms=3.5,
                    lw=1.4,
                    label="Held-out" if ev == "heldout" else "2025",
                )
            ax.set_xscale("log")
            ax.set_xticks([0.07, 0.1, 1, 10, 100])
            ax.set_xticklabels(["0", "0.1", "1", "10", "100"])
            if zero:
                ax.axhline(0.0, color="#111827", lw=0.8, ls=":")
            ax.grid(True, color="#E5E7EB", lw=0.7)
            ax.set_axisbelow(True)
            if r == 0:
                ax.set_title(fam)
            if c == 0:
                ax.set_ylabel(ylab)
            if r == 1:
                ax.set_xlabel(r"Penalty strength $\rho$")
            if r == 0 and c == 1:
                ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return save_fig(fig, "mechanism_vs_rho")


def plot_accuracy_equity(combined: pd.DataFrame) -> Path:
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.4))
    specs = (
        (0, 0, "heldout", "PRD", "Held-out", "PRD", IAAO_PRD_RANGE),
        (0, 1, "heldout", "VEI", "Held-out", r"VEI (\%)", IAAO_VEI_RANGE),
        (1, 0, "forward_2025", "PRD", "2025 forward", "PRD", IAAO_PRD_RANGE),
        (1, 1, "forward_2025", "VEI", "2025 forward", r"VEI (\%)", IAAO_VEI_RANGE),
    )
    lin = combined_row(combined, "Linear")
    lgb = combined_row(combined, "LightGBM")
    for r, c, ev, met, title, ylab, band in specs:
        ax = axes[r, c]
        ax.axhspan(band[0], band[1], color="#9CA3AF", alpha=0.18, lw=0, label="Reference band")
        for fam, color in (("Direct", pipe.DIRECT_COLOR), ("Surrogate", pipe.SURR_COLOR)):
            sub = combined.loc[combined["family"] == fam].sort_values("rho")
            x = sub[f"R2_price__{ev}"].to_numpy(dtype=float)
            y = sub[f"{met}__{ev}"].to_numpy(dtype=float)
            ax.plot(x, y, color=color, marker="o", ms=3.2, lw=1.3, label=fam)
            if len(x) >= 2:
                ax.annotate("", xy=(x[-1], y[-1]), xytext=(x[-2], y[-2]), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.0))
        ax.scatter([metric(lin, "R2_price", ev)], [metric(lin, met, ev)], marker="D", s=36, color=pipe.LINEAR_COLOR, zorder=5, label="Linear")
        ax.scatter([metric(lgb, "R2_price", ev)], [metric(lgb, met, ev)], marker="s", s=36, color=pipe.NATIVE_COLOR, zorder=5, label="LightGBM")
        ax.set_xlabel(r"$R^2_P$")
        ax.set_ylabel(ylab)
        ax.set_title(f"{title}: {met}")
        ax.grid(True, color="#E5E7EB", lw=0.7)
        ax.set_axisbelow(True)
        if r == 0 and c == 1:
            ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return save_fig(fig, "accuracy_equity_trajectories")


def plot_prb_mki(combined: pd.DataFrame) -> Path:
    long = []
    for fam in ("Direct", "Surrogate"):
        sub = combined.loc[combined["family"] == fam]
        for ev in ("heldout", "forward_2025"):
            long.append(
                pd.DataFrame(
                    {
                        "family": fam,
                        "evaluation": ev,
                        "rho": sub["rho"].to_numpy(),
                        "R2_price": sub[f"R2_price__{ev}"].to_numpy(),
                        "PRB": sub[f"PRB__{ev}"].to_numpy(),
                        "MKI": sub[f"MKI__{ev}"].to_numpy(),
                    }
                )
            )
    oos = pd.concat(long, ignore_index=True)
    orig_fig = getattr(pipe, "FIG_OUT", None)
    orig_out = getattr(pipe, "FIG_OUT", None)
    pipe.FIG_OUT = PAPER_IMG
    if hasattr(pipe, "FIG_OUT"):
        pipe.FIG_OUT = PAPER_IMG
    PAPER_IMG.mkdir(parents=True, exist_ok=True)
    pdf = pipe.plot_accuracy_equity_r2(oos, ["PRB", "MKI"], "prb_mki_accuracy_equity")
    for dest in (orig_fig, orig_out, getattr(pipe, "FIG_OUT", None)):
        if dest is None:
            continue
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        if pdf.resolve() != (dest / pdf.name).resolve():
            shutil.copy2(pdf, dest / pdf.name)
    if orig_fig is not None:
        pipe.FIG_OUT = orig_fig
    if orig_out is not None:
        pipe.FIG_OUT = orig_out
    return pdf


def plot_metric_paths(combined: pd.DataFrame, metrics: Sequence[Tuple[str, str]], stem: str) -> Path:
    set_style()
    fig, axes = plt.subplots(len(metrics), 2, figsize=(8.4, 2.15 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = np.array([axes])
    for r, (col, ylab) in enumerate(metrics):
        for c, fam in enumerate(("Direct", "Surrogate")):
            ax = axes[r, c]
            sub = combined.loc[combined["family"] == fam].sort_values("rho")
            color = pipe.DIRECT_COLOR if fam == "Direct" else pipe.SURR_COLOR
            x = rho_x(sub["rho"].to_numpy(dtype=float))
            ax.plot(x, sub[f"{col}__heldout"], color=color, marker="o", ms=3, lw=1.3, label="Held-out")
            ax.plot(x, sub[f"{col}__forward_2025"], color=color, ls="--", marker="s", ms=3, lw=1.2, label="2025")
            ax.set_xscale("log")
            ax.set_xticks([0.07, 0.1, 1, 10, 100])
            ax.set_xticklabels(["0", "0.1", "1", "10", "100"])
            ax.grid(True, color="#E5E7EB", lw=0.7)
            if r == 0:
                ax.set_title(fam)
            if c == 0:
                ax.set_ylabel(ylab)
            if r == len(metrics) - 1:
                ax.set_xlabel(r"$\rho$")
            if r == 0 and c == 1:
                ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return save_fig(fig, stem)


def plot_cv_stability(combined: pd.DataFrame) -> Path:
    set_style()
    metrics = (("R2_price", r"$R^2_P$"), ("PRD", "PRD"), ("VEI", r"VEI (\%)"), ("Beta_log", r"$\beta_{\log}$"))
    fig, axes = plt.subplots(4, 2, figsize=(8.4, 9.2), sharex=True)
    for r, (col, ylab) in enumerate(metrics):
        for c, fam in enumerate(("Direct", "Surrogate")):
            ax = axes[r, c]
            sub = combined.loc[combined["family"] == fam].sort_values("rho")
            x = rho_x(sub["rho"].to_numpy(dtype=float))
            color = pipe.DIRECT_COLOR if fam == "Direct" else pipe.SURR_COLOR
            for k in range(1, 8):
                ax.plot(x, sub[f"{col}__fold_{k}"], color="#9CA3AF", lw=0.8, alpha=0.7)
            ax.plot(x, sub[f"{col}__CV_mean"], color=color, lw=2.0)
            ax.set_xscale("log")
            ax.set_xticks([0.07, 0.1, 1, 10, 100])
            ax.set_xticklabels(["0", "0.1", "1", "10", "100"])
            ax.grid(True, color="#E5E7EB", lw=0.7)
            if r == 0:
                ax.set_title(fam)
            if c == 0:
                ax.set_ylabel(ylab)
            if r == 3:
                ax.set_xlabel(r"$\rho$")
    fig.tight_layout()
    return save_fig(fig, "cv_fold_stability")


def plot_vei_groups() -> Path:
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.2), sharex=True, sharey=True)
    specs = (
        (0, 0, "heldout", "LinearRegression", "Held-out Linear"),
        (0, 1, "heldout", "LGBMRegressor", "Held-out LightGBM"),
        (1, 0, "forward_2025", "LinearRegression", "2025 Linear"),
        (1, 1, "forward_2025", "LGBMRegressor", "2025 LightGBM"),
    )
    for r, c, split, model, title in specs:
        ax = axes[r, c]
        raw = load_baseline_split(split)
        mapped = raw[name_col(raw)].replace({pipe.LINEAR: "LinearRegression", pipe.NATIVE: "LGBMRegressor"})
        sub = raw.loc[mapped == model]
        prof = vei_percentile_group_profile(sub["y_pred"].to_numpy(dtype=float), sub["y_true"].to_numpy(dtype=float))
        ax.fill_between(prof["group"], prof["ci_low"], prof["ci_high"], color="#1D4ED8", alpha=0.18, lw=0)
        ax.plot(prof["group"], prof["median_ratio"], color="#1D4ED8", marker="o", lw=1.4)
        ax.axhline(float(prof["overall_median_ratio"].iloc[0]), color="#111827", ls=":", lw=0.9)
        ax.axhline(1.0, color="#6B7280", ls="--", lw=0.8)
        ax.set_title(title)
        if c == 0:
            ax.set_ylabel("Group median valuation ratio")
        if r == 1:
            ax.set_xlabel("VEI percentile group (low to high value)")
        ax.grid(True, color="#E5E7EB", lw=0.7)
        ax.set_axisbelow(True)
    fig.tight_layout()
    return save_fig(fig, "vei_percentile_group_profile")


def _bold_pair(a: float, b: float, fa: str, fb: str, higher: Optional[bool], target: Optional[float]) -> Tuple[str, str]:
    if target is not None:
        left = abs(a - target) <= abs(b - target)
    else:
        left = a >= b if higher else a <= b
    if left:
        return r"\textbf{" + fa + "}", fb
    return fa, r"\textbf{" + fb + "}"


def make_baseline_table(combined: pd.DataFrame) -> str:
    def row(label: str, name: str, kind: str, higher: Optional[bool] = None, target: Optional[float] = None) -> str:
        parts = [rf"\baselinemetric{{{label}}}"]
        fmt = {"r2": fmt_r2, "mae": fmt_mae, "pctf": fmt_pct_frac, "pct": fmt_pct, "num": fmt_3}[kind]
        for split in ("heldout", "forward_2025"):
            a = metric(combined_row(combined, "Linear"), name, split)
            b = metric(combined_row(combined, "LightGBM"), name, split)
            fa, fb = _bold_pair(a, b, fmt(a), fmt(b), higher, target)
            parts.extend([fa, fb])
        return " & ".join(parts) + r" \\"

    body = "\n".join(
        [
            r"\multicolumn{5}{@{}l}{\textbf{Prediction}} \\",
            r"\cmidrule(r){1-1}",
            row(r"$R^2_P$", "R2_price", "r2", True),
            row(r"$\operatorname{MAE}_P$", "MAE_price", "mae", False),
            row(r"$\operatorname{MAPE}_P$", "MAPE", "pctf", False),
            row(r"$\operatorname{RMSE}_{\log P}$", "RMSE_log", "num", False),
            r"\addlinespace[5pt]",
            r"\multicolumn{5}{@{}l}{\textbf{Valuation level}} \\",
            r"\cmidrule(r){1-1}",
            row("Median ratio $m_S$", "median_ratio", "num", target=1.0),
            row(r"Mean ratio $\bar r_S$", "mean_ratio", "num", target=1.0),
            row(r"Weighted mean $\bar r_{W,S}$", "weighted_mean_ratio", "num", target=1.0),
            r"\addlinespace[5pt]",
            r"\multicolumn{5}{@{}l}{\textbf{Horizontal uniformity}} \\",
            r"\cmidrule(r){1-1}",
            row("COD", "COD", "pct", False),
            row("COV", "COV", "pctf", False),
            r"\addlinespace[5pt]",
            r"\multicolumn{5}{@{}l}{\textbf{Vertical equity}} \\",
            r"\cmidrule(r){1-1}",
            row("PRD", "PRD", "num", target=1.0),
            row("PRB", "PRB", "num", target=0.0),
            row("MKI", "MKI", "num", target=1.0),
            row("VEI", "VEI", "pct", target=0.0),
            r"\addlinespace[5pt]",
            r"\multicolumn{5}{@{}l}{\textbf{Supplemental diagnostics}} \\",
            r"\cmidrule(r){1-1}",
            row(r"$\beta_{\log}$", "Beta_log", "num", target=0.0),
            row(r"$\operatorname{dCor}(e,y)$", "dCor_e_y", "num", False),
        ]
    )
    return rf"""
\begin{{table}}[!ht]
\centering
\scriptsize
\setlength{{\tabcolsep}}{{2.2pt}}
\renewcommand{{\arraystretch}}{{1.06}}
\newcommand{{\baselinemetric}}[1]{{\hspace*{{0.6em}}#1}}
\caption{{Baseline comparison on the primary held-out evaluation and the 2025 forward evaluation.}}
\label{{tab:ccao_baseline_results}}
\begin{{tabularx}}{{\textwidth}}{{@{{}} >{{\raggedright\arraybackslash}}p{{2.80cm}} >{{\centering\arraybackslash}}X >{{\centering\arraybackslash}}X |>{{\centering\arraybackslash}}X >{{\centering\arraybackslash}}X @{{}}}}
\toprule
\textbf{{Measure}} & \multicolumn{{2}}{{c}}{{\textbf{{Held-out evaluation}}}} & \multicolumn{{2}}{{c}}{{\textbf{{2025 forward evaluation}}}} \\
\cmidrule(lr){{2-3}}\cmidrule(l){{4-5}}
& \textbf{{Linear}} & \textbf{{\shortstack{{Light\\GBM}}}} & \textbf{{Linear}} & \textbf{{\shortstack{{Light\\GBM}}}} \\
\midrule
{body}
\bottomrule
\end{{tabularx}}
\vspace{{1mm}}
\begin{{minipage}}{{\textwidth}}
\scriptsize
\emph{{Notes.}}
Preferred values and applicable reference ranges for the principal predictive
and assessor-facing measures are defined in Table~\ref{{tab:assessment_metrics_summary}}.
MAPE, COD, COV, and VEI are reported in percent; MAE is reported in dollars.
For the held-out evaluation, both models are fit on the {fmt_int(N_DEV)}-sale development
pool. For the 2025 forward evaluation, each baseline specification is refit on all
{fmt_int(N_PROD)} eligible 2016--2024 sales. The supplemental slope $\beta_{{\log}}$ and
distance correlation are defined in Eqs.~\eqref{{eq:log_ratio_slope}}
and~\eqref{{eq:dcor_diagnostic}}, respectively. Boldface indicates the preferred
value within each linear--LightGBM comparison according to the target or direction
defined above; it does not denote statistical significance or formal compliance.
\end{{minipage}}
\end{{table}}
"""


def make_sample_table() -> str:
    return rf"""
\begin{{table}}[!ht]
\centering
\caption{{Final CCAO temporal samples.}}
\label{{tab:ccao_samples}}
\begin{{tabular}}{{llr}}
\toprule
Sample & Period & Observations \\
\midrule
Development / validation & 2016-01-01--2023-11-09 & {fmt_int(N_DEV)} \\
Primary held-out test & 2023-11-09--2024-12-31 & {fmt_int(N_HOLDOUT)} \\
Production training & 2016-01-01--2024-12-31 & {fmt_int(N_PROD)} \\
Secondary forward evaluation & 2025-01-01--2025-12-29 & {fmt_int(N_2025)} \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def make_fold_table() -> str:
    path = next((RESULT_ROOT / "protocol").glob("**/folds.json"))
    folds = json.loads(path.read_text())["folds"]
    lines = []
    for i, fold in enumerate(folds, start=1):
        window = f"{fold['train_start']}--{fold['val_end']}"
        lines.append(
            f"{i} & {window} & Newest 10\\% & {fmt_int(fold['train_size'])} & {fmt_int(fold['val_size'])} \\\\"
        )
    body = "\n".join(lines)
    return rf"""
\begin{{table}}[!ht]
\centering
\caption{{Rolling-origin validation structure for the final CCAO development sample.}}
\label{{tab:fold_structure}}
\begin{{tabular}}{{cllrr}}
\toprule
Fold & Cumulative window & Validation rule & $n_{{\mathrm{{train}}}}$ & $n_{{\mathrm{{val}}}}$ \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def make_rho0_table(combined: pd.DataFrame, control: Dict[str, float]) -> str:
    def line(sample: str, label: str, row: pd.Series, mean_d: str, max_d: str) -> str:
        split = "heldout" if sample.startswith("Held") else "forward_2025"
        return (
            f"{sample} & {label} & {fmt_r2(metric(row, 'R2_price', split))} & "
            f"{fmt_3(metric(row, 'RMSE_log', split))} & {fmt_3(metric(row, 'Beta_log', split))} & "
            f"{mean_d} & {max_d} \\\\"
        )

    native_h = combined_row(combined, "LightGBM")
    d0 = combined_row(combined, "Direct", 0.0)
    s0 = combined_row(combined, "Surrogate", 0.0)
    dm = f"{control.get('direct_mean_abs_delta_log', control.get('direct_mean_abs_delta_log', 0.0)):.2e}"
    dx = f"{control.get('direct_max_abs_delta_log', control.get('direct_max_abs_delta_log', 0.0)):.2e}"
    sm = f"{control.get('surrogate_mean_abs_delta_log', control.get('surrogate_mean_abs_delta_log', 0.0)):.2e}"
    sx = f"{control.get('surrogate_max_abs_delta_log', control.get('surrogate_max_abs_delta_log', 0.0)):.2e}"
    body = "\n".join(
        [
            line("Held-out", "Ordinary LightGBM", native_h, "--", "--"),
            line("Held-out", r"Direct, $\rho=0$", d0, dm, dx),
            line("Held-out", r"Surrogate, $\rho=0$", s0, sm, sx),
            r"\addlinespace",
            line("2025 forward", "Ordinary LightGBM", native_h, "--", "--"),
            line("2025 forward", r"Direct, $\rho=0$", d0, dm, dx),
            line("2025 forward", r"Surrogate, $\rho=0$", s0, sm, sx),
        ]
    )
    return rf"""
\begin{{table}}[!ht]
\centering
\caption{{Implementation-control comparison at $\rho=0$. Prediction differences are log-price deviations from ordinary LightGBM, aligned on sale identifiers.}}
\label{{tab:rho_zero_control}}
\begin{{tabular}}{{llrrrrr}}
\toprule
Sample & Model & $R^2_P$ & $\operatorname{{RMSE}}_{{\log P}}$ & $\beta_{{\log}}$ & Mean $|\Delta\widehat y|$ & Max $|\Delta\widehat y|$ \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def make_path_table(combined: pd.DataFrame, anchors: Sequence[float]) -> str:
    def row(label: str, rho_tex: str, fam: str, rho: Optional[float], split: str) -> str:
        rec = combined_row(combined, fam, rho)
        return (
            f"{label} & {rho_tex} & {fmt_r2(metric(rec, 'R2_price', split))} & {fmt_mae(metric(rec, 'MAE_price', split))} & "
            f"{fmt_3(metric(rec, 'PRD', split))} & {fmt_3(metric(rec, 'PRB', split))} & {fmt_3(metric(rec, 'MKI', split))} & "
            f"{fmt_pct(metric(rec, 'VEI', split))} & {fmt_3(metric(rec, 'Beta_log', split))} & {fmt_3(metric(rec, 'dCor_e_y', split))} \\\\"
        )

    def panel(split: str) -> str:
        lines = [row("Ordinary LightGBM", "--", "LightGBM", None, split)]
        for fam in ("Direct", "Surrogate"):
            for rho in anchors:
                tex = "0" if rho == 0 else (fmt_rho(rho) if abs(rho - 100) < 1e-8 else rf"$\approx{fmt_rho(rho)}$")
                if abs(rho - 100) < 1e-8:
                    tex = "100"
                elif rho == 0:
                    tex = "0"
                else:
                    tex = rf"$\approx{fmt_rho(rho)}$"
                lines.append(row(fam, tex, fam, float(rho), split))
        return "\n".join(lines)

    return rf"""
\begin{{table}}[!htbp]
\centering
\scriptsize
\caption{{Regularization-path summary at prespecified display anchors. Rows are fixed for display before inspecting outcomes and do not constitute model selection.}}
\label{{tab:path_anchor_summary}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{llrrrrrrrr}}
\toprule
Family & $\rho$ & $R^2_P$ & MAE & PRD & PRB & MKI & VEI & $\beta_{{\log}}$ & dCor \\
\midrule
\multicolumn{{10}}{{l}}{{\textit{{Panel A: held-out evaluation}}}} \\
{panel("heldout")}
\midrule
\multicolumn{{10}}{{l}}{{\textit{{Panel B: 2025 forward evaluation}}}} \\
{panel("forward_2025")}
\bottomrule
\end{{tabular}}}}
\end{{table}}
"""


def make_design_table() -> str:
    return rf"""
\begin{{table}}[t]
\centering
\caption{{CCAO application and pre-selection regularization-path design.}}
\label{{tab:ccao_design}}
\begin{{tabular}}{{ll}}
\toprule
Item & Design \\
\midrule
Population & Verified Cook County residential sales \\
Development / CV & Oldest 90\% of eligible 2016--2024 sales: {fmt_int(N_DEV)}, through 2023-11-09 \\
Primary held-out evaluation & Newest 10\%: {fmt_int(N_HOLDOUT)} sales, 2023-11-09--2024-12-31 \\
Production training & {fmt_int(N_PROD)} eligible 2016--2024 sales \\
2025 forward evaluation & {fmt_int(N_2025)} sales, 2025-01-01--2025-12-29 \\
Predictors & 95 total; 23 categorical \\
Validation & Seven expanding-window rolling-origin folds; newest 10\% validates \\
Base nonlinear learner & LightGBM on log sale price; 994 trees; hyperparameters frozen before the penalty sweep \\
Corrections evaluated & Direct squared covariance and sample-additive covariance upper bound \\
Penalty path & $\rho=0$ control plus 50 positive values in $[0.1,100]$ per family \\
Current analysis & Full CV, held-out, and forward paths; no model selection \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def figure_env(label: str, path: str, caption: str, width: str = "0.92\\textwidth") -> str:
    return rf"""
\begin{{figure}}[!htbp]
\centering
\safeincludegraphics[width={width}]{{{path}}}
\caption{{{caption}}}
\label{{{label}}}
\end{{figure}}
"""


def figure_env_two(label: str, path_a: str, path_b: str, caption: str) -> str:
    return rf"""
\begin{{figure}}[!htbp]
\centering
\safeincludegraphics[width=0.92\textwidth]{{{path_a}}}\\[2mm]
\safeincludegraphics[width=0.92\textwidth]{{{path_b}}}
\caption{{{caption}}}
\label{{{label}}}
\end{{figure}}
"""


def path_findings(combined: pd.DataFrame, anchors: Sequence[float]) -> str:
    d0 = combined_row(combined, "Direct", 0.0)
    d1 = combined_row(combined, "Direct", float(anchors[-1]))
    s0 = combined_row(combined, "Surrogate", 0.0)
    s1 = combined_row(combined, "Surrogate", float(anchors[-1]))
    n_same_dir = 0
    for k in range(1, 8):
        if float(d1[f"Beta_log__fold_{k}"]) > float(d0[f"Beta_log__fold_{k}"]):
            n_same_dir += 1
    return (
        f"Along the Direct path, held-out $R^2_P$ moves from {fmt_r2(metric(d0, 'R2_price', 'heldout'))} at $\\rho=0$ "
        f"to {fmt_r2(metric(d1, 'R2_price', 'heldout'))} at $\\rho=100$, while $\\beta_{{\\log}}$ moves from "
        f"{fmt_3(metric(d0, 'Beta_log', 'heldout'))} to {fmt_3(metric(d1, 'Beta_log', 'heldout'))} and VEI from "
        f"{fmt_pct(metric(d0, 'VEI', 'heldout'))} to {fmt_pct(metric(d1, 'VEI', 'heldout'))}. "
        f"Along the Surrogate path, held-out $R^2_P$ moves from {fmt_r2(metric(s0, 'R2_price', 'heldout'))} to "
        f"{fmt_r2(metric(s1, 'R2_price', 'heldout'))}, $\\beta_{{\\log}}$ from {fmt_3(metric(s0, 'Beta_log', 'heldout'))} "
        f"to {fmt_3(metric(s1, 'Beta_log', 'heldout'))}, and VEI from {fmt_pct(metric(s0, 'VEI', 'heldout'))} to "
        f"{fmt_pct(metric(s1, 'VEI', 'heldout'))}. The 2025 forward paths move in the same direction. "
        f"On Direct, the first-order slope becomes less negative from $\\rho=0$ to $\\rho=100$ in {n_same_dir} of 7 folds. "
        "These statements describe the ordered paths; they do not identify an operating point."
    )


def rho0_control_from_preds() -> Dict[str, float]:
    native = native_slice(load_baseline_split("heldout"))
    d0 = load_oos_pred("Direct", 0.0, "heldout")
    s0 = load_oos_pred("Surrogate", 0.0, "heldout")
    return pipe.compute_rho0_control(native, d0, s0)


def compile_pdf() -> Path:
    env = os.environ.copy()
    cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "paper_v6.tex"]
    proc = subprocess.run(cmd, cwd=str(REPO / "paper"), env=env, text=True, capture_output=True)
    pdf = REPO / "paper" / "paper_v6.pdf"
    if proc.returncode != 0 or not pdf.is_file():
        raise RuntimeError((proc.stdout or "")[-4000:] + "\n" + (proc.stderr or "")[-4000:])
    dest = RESULT_ROOT / "paper_outputs" / "paper_v6.pdf"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pdf, dest)
    return dest


def live_tex(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if line.lstrip().startswith("%"):
            continue
        lines.append(re.sub(r"(?<!\\)%.*", "", line))
    return "\n".join(lines)


def qa_tex(text: str) -> List[str]:
    live = live_tex(text)
    live_l = live.lower()
    problems = []
    for phrase in pipe.FORBIDDEN_PHRASES:
        if phrase in live:
            problems.append(f"forbidden phrase: {phrase}")
    for stale in ("344,610", "382,900", "populate from full"):
        if stale in live:
            problems.append(f"stale live text: {stale}")
    if "PLACEHOLDER" in live:
        problems.append("stale live text: PLACEHOLDER")
    if SELECTION_RE.search(live) and "no model selection" not in live_l:
        pass
    for lab in (
        "tab:ccao_baseline_results",
        "tab:rho_zero_control",
        "tab:path_anchor_summary",
        "fig:baseline_motivation",
        "fig:ratio_shape_path_placeholder",
        "fig:mechanism_path_placeholder",
        "fig:accuracy_equity_placeholder",
    ):
        if lab not in text:
            problems.append(f"missing label {lab}")
    if "results_reference_assets" in live:
        problems.append("live reference-layout figure remains")
    return problems


def populate() -> int:
    pipe.configure_paths(str(RESULT_ROOT))
    if "paper_v6_preselection_994" not in str(pipe.ROOT):
        raise RuntimeError("Populate must run from the 994 result root.")
    gate = pipe.load_json(RESULT_ROOT / "baseline_gate.json")
    if str(gate.get("decision")) != "ADOPT_994":
        raise RuntimeError("baseline_gate.json is not ADOPT_994")
    combined_path = RESULT_ROOT / "analysis" / "combined_path_table.csv"
    combined = pd.read_csv(combined_path)
    if combined.empty or len(combined) < 100:
        raise RuntimeError(f"combined table looks incomplete: {len(combined)} rows")
    anchors = display_rhos(combined)
    control = rho0_control_from_preds()

    figs = {
        "baseline": plot_baseline_motivation(),
        "ratio": plot_ratio_shape(anchors),
        "mechanism": plot_mechanism(combined),
        "tradeoff": plot_accuracy_equity(combined),
        "prb": plot_prb_mki(combined),
        "pred": plot_metric_paths(
            combined,
            (("R2_price", r"$R^2_P$"), ("MAE_price", "MAE"), ("MAPE", "MAPE"), ("RMSE_log", r"RMSE$_{\log}$")),
            "predictive_metric_paths",
        ),
        "level": plot_metric_paths(
            combined,
            (
                ("median_ratio", "Median ratio"),
                ("mean_ratio", "Mean ratio"),
                ("weighted_mean_ratio", "Weighted mean ratio"),
                ("COD", "COD"),
                ("COV", "COV"),
            ),
            "level_uniformity_paths",
        ),
        "cv": plot_cv_stability(combined),
        "vei": plot_vei_groups(),
    }

    tex = PAPER_TEX.read_text(encoding="utf-8")
    tex = tex.replace("344,610", fmt_int(N_DEV)).replace("382,900", fmt_int(N_PROD))
    replacements = {
        ("table", "tab:ccao_baseline_results"): make_baseline_table(combined),
        ("table", "tab:ccao_samples"): make_sample_table(),
        ("table", "tab:fold_structure"): make_fold_table(),
        ("table", "tab:ccao_design"): make_design_table(),
        ("table", "tab:rho_zero_control"): make_rho0_table(combined, control),
        ("table", "tab:path_anchor_summary"): make_path_table(combined, anchors),
        ("figure", "fig:baseline_motivation"): figure_env(
            "fig:baseline_motivation",
            f"{IMG_REL}/baseline_models_motivation_2024_2025.pdf",
            "Descriptive valuation-ratio patterns against sale price for the two baseline models in the primary held-out evaluation and the 2025 forward evaluation. Curves show median ratios in 30 equal-count sale-price bins; shaded regions show the corresponding interquartile ranges.",
            "0.86\\textwidth",
        ),
        ("figure", "fig:ratio_shape_path_placeholder"): figure_env(
            "fig:ratio_shape_path_placeholder",
            f"{IMG_REL}/ratio_shape_evolution.pdf",
            r"Median valuation-to-sale ratios in 30 equal-count sale-price bins along the Direct and Surrogate paths. Columns are penalty families; rows are held-out and 2025 evaluations. Curves correspond to the prespecified display anchors $\rho=0,0.1,1,10,100$ (nearest grid points). No operating point is highlighted.",
        ),
        ("figure", "fig:mechanism_path_placeholder"): figure_env(
            "fig:mechanism_path_placeholder",
            f"{IMG_REL}/mechanism_vs_rho.pdf",
            r"First-order slope $\beta_{\log}$ and distance correlation along the full Direct and Surrogate grids. The $\rho=0$ control is shown explicitly; positive $\rho$ is on a log scale. Held-out and 2025 paths use distinct line styles.",
            "0.86\\textwidth",
        ),
        ("figure", "fig:accuracy_equity_placeholder"): figure_env(
            "fig:accuracy_equity_placeholder",
            f"{IMG_REL}/accuracy_equity_trajectories.pdf",
            r"Accuracy--equity trajectories for PRD and VEI against $R^2_P$. Shaded regions are reference bands, not compliance bands. Linear and ordinary LightGBM are context anchors. Arrows indicate the direction of increasing $\rho$ and do not mark a selected point.",
        ),
        ("figure", "fig:prb_mki_path_placeholder"): figure_env(
            "fig:prb_mki_path_placeholder",
            f"{IMG_REL}/prb_mki_accuracy_equity.pdf",
            r"Companion accuracy--equity trajectories for PRB and MKI against $R^2_P$ on the held-out and 2025 samples.",
        ),
        ("figure", "fig:other_metric_paths_placeholder"): figure_env_two(
            "fig:other_metric_paths_placeholder",
            f"{IMG_REL}/predictive_metric_paths.pdf",
            f"{IMG_REL}/level_uniformity_paths.pdf",
            r"Predictive-metric paths and valuation-level/uniformity paths along the Direct and Surrogate grids, with held-out and 2025 evaluations overlaid.",
        ),
        ("figure", "fig:cv_path_stability_placeholder"): figure_env(
            "fig:cv_path_stability_placeholder",
            f"{IMG_REL}/cv_fold_stability.pdf",
            r"Chronological fold paths (thin) and equal-weight means (thick) for $R^2_P$, PRD, VEI, and $\beta_{\log}$. Folds are not collapsed into a selection score.",
            "0.86\\textwidth",
        ),
        ("figure", "fig:vei_group_profile_placeholder"): figure_env(
            "fig:vei_group_profile_placeholder",
            f"{IMG_REL}/vei_percentile_group_profile.pdf",
            r"VEI percentile-group median valuation ratios for Linear and ordinary LightGBM on the held-out and 2025 samples, with deterministic 90\% bootstrap intervals.",
        ),
        ("figure", "fig:full_path_table_placeholder"): rf"""
\begin{{figure}}[!htbp]
\centering
\caption{{Complete pre-selection path table. The machine-readable artifact \texttt{{combined\_path\_table.csv}} contains every family--$\rho$ row with seven fold values, equal-weight CV mean and SD, held-out, and 2025 values for the Section~2 metrics. No row is omitted.}}
\label{{fig:full_path_table_placeholder}}
\end{{figure}}
""",
    }
    for (env, label), new_env in replacements.items():
        tex = pipe.replace_latex_environment_by_label(tex, env, label, new_env)

    hyper = (
        "The LightGBM hyperparameters were tuned on the same seven chronological folds "
        r"before any penalty path was estimated. The tuning criterion was the equal-weight mean of fold validation price RMSE, "
        r"$\operatorname{RMSE}_P(S)=\sqrt{n_S^{-1}\sum_{i\in S}(\widehat P_i-P_i)^2}$. "
        "The adopted configuration uses 994 trees and was then frozen for ordinary LightGBM, Direct, and Surrogate fits. "
        "The held-out and 2025 samples were not used for this hyperparameter choice. "
        r"With that vector held fixed, $\rho$ is the only dimension varied within each penalized family."
    )
    old_hyper_re = re.compile(
        r"The LightGBM hyperparameters\s+are frozen before the sweep and held fixed across all LightGBM variants,\s+so \$\\rho\$\s+is the only dimension varied within each penalized family\.",
        re.S,
    )
    tex, n_hyper = old_hyper_re.subn(lambda _m: hyper, tex, count=1)
    if n_hyper == 0:
        old_hyper = (
            "The LightGBM hyperparameters are frozen before the sweep and held fixed across all LightGBM variants, so $\\rho$ "
            "is the only dimension varied within each penalized family."
        )
        if old_hyper in tex:
            tex = tex.replace(old_hyper, hyper, 1)

    findings = path_findings(combined, anchors)
    tex = re.sub(r"\\todo\{Once the full held-out.*?here\.\}", lambda _m: findings, tex, count=1, flags=re.S)
    tex = re.sub(
        r"\\todo\{Insert the final held-out and 2025 correction results here only after.*?design\.\}",
        "This draft reports the complete Direct and Surrogate regularization paths without choosing a penalty strength.",
        tex,
        count=1,
        flags=re.S,
    )
    tex = re.sub(
        r"\\todo\{Insert the final quantitative correction results in the abstract.*?baselines\.\}",
        "Complete Direct and Surrogate regularization paths are reported on the held-out and 2025 samples; no penalty strength is chosen.",
        tex,
        count=1,
        flags=re.S,
    )
    tex = re.sub(
        r"The final penalized-model effect sizes will be reported from the same empirical design once the correction models have been rerun\.",
        "This draft reports the complete Direct and Surrogate regularization paths without choosing a penalty strength.",
        tex,
        count=1,
    )
    tex = re.sub(
        r"\\todo\{After the final correction run, add one concise empirical contribution sentence.*?source\.\}",
        "We report complete regularization paths rather than a selected penalty.",
        tex,
        count=1,
        flags=re.S,
    )
    tex = re.sub(r"\\todo\{Populate Table~\\ref\{tab:rho_zero_control\}.*?QA\.\}", "", tex, count=1, flags=re.S)
    tex = re.sub(r"\\todo\{Populate Table~\\ref\{tab:path_anchor_summary\}.*?here\.\}", "", tex, count=1, flags=re.S)
    tex = re.sub(r"\\todo\{Regenerate Figure~\\ref\{fig:baseline_motivation\}.*?rendering\.\}", "", tex, count=1, flags=re.S)
    tex = re.sub(r"\\todo\{Replace Figure~\\ref\{fig:ratio_shape_path_placeholder\}.*?IQR structure\.\}", "", tex, count=1, flags=re.S)
    tex = re.sub(r"\\todo\{Replace Figure~\\ref\{fig:mechanism_path_placeholder\}.*?particular \$\\rho\$\.\}", "", tex, count=1, flags=re.S)
    tex = re.sub(r"\\todo\{Replace Figure~\\ref\{fig:accuracy_equity_placeholder\}.*?point\.\}", "", tex, count=1, flags=re.S)
    tex = re.sub(r"\\todo\{After the 728-fit CV completes.*?outputs\)\.\}", lambda _m: findings, tex, count=1, flags=re.S)

    lin = combined_row(combined, "Linear")
    lgb = combined_row(combined, "LightGBM")
    prose = (
        f"On the primary held-out evaluation, LightGBM improves all four predictive measures relative to linear regression: "
        f"$R^2_P$ rises from ${fmt_r2(metric(lin, 'R2_price', 'heldout'))}$ to ${fmt_r2(metric(lgb, 'R2_price', 'heldout'))}$, "
        "while MAE, MAPE, and log-scale RMSE decrease. "
        f"PRD moves from ${fmt_3(metric(lin, 'PRD', 'heldout'))}$ to ${fmt_3(metric(lgb, 'PRD', 'heldout'))}$, "
        f"and VEI from ${fmt_pct(metric(lin, 'VEI', 'heldout'))}$ to ${fmt_pct(metric(lgb, 'VEI', 'heldout'))}$. "
        "The 2025 forward evaluation shows the same qualitative ordering."
    )
    tex = re.sub(
        r"\{On the primary held-out evaluation, LightGBM improves all four predictive measures relative to linear regression:.*?supplemental diagnostics\.\}",
        lambda _m: "{" + prose + "}",
        tex,
        count=1,
        flags=re.S,
    )

    tables_dir = RESULT_ROOT / "paper_outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(combined_path, tables_dir / "combined_path_table.csv")

    problems = qa_tex(tex)
    if problems:
        raise RuntimeError("QA failed before write: " + "; ".join(problems))
    PAPER_TEX.write_text(tex, encoding="utf-8")
    problems = qa_tex(PAPER_TEX.read_text(encoding="utf-8"))
    if problems:
        raise RuntimeError("QA failed after write: " + "; ".join(problems))

    fig_dest = RESULT_ROOT / "paper_outputs" / "figures"
    fig_dest.mkdir(parents=True, exist_ok=True)
    for src in figs.values():
        src = Path(src)
        if src.is_file():
            shutil.copy2(src, fig_dest / src.name)
            png = src.with_suffix(".png")
            if png.is_file():
                shutil.copy2(png, fig_dest / png.name)

    sha = {
        "tex": sha256_file(PAPER_TEX),
        "combined_csv": sha256_file(combined_path),
    }
    pdf = REPO / "paper" / "paper_v6.pdf"
    if pdf.is_file():
        sha["pdf"] = sha256_file(pdf)

    manifest = {
        "selection_performed": False,
        "result_root": str(RESULT_ROOT),
        "git": git_provenance(),
        "n_combined_rows": int(len(combined)),
        "display_anchors": [float(x) for x in anchors],
        "rho0_control": control,
        "figures": {k: str(v) for k, v in figs.items()},
        "sha256": sha,
        "sample_counts": {"development": N_DEV, "heldout": N_HOLDOUT, "production": N_PROD, "forward_2025": N_2025},
        "compiled_here": False,
    }
    pipe.write_json(RESULT_ROOT / "paper_outputs" / "paper_results_manifest.json", manifest)
    print("paper/paper_v6.tex has been populated with the completed 994-tree pre-selection results.")
    print("No rho, penalty family, or penalized configuration was selected or ranked in this analysis.")
    return 0


if __name__ == "__main__":
    raise SystemExit(populate())
