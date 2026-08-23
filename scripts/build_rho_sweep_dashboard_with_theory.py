#!/usr/bin/env python
"""
Build an HTML dashboard for the covariance-penalty rho-sweep experiments under
output/rho_sweep_500_estimators/.

The dashboard is fully data-driven: it scans the result folders, parses the
(data source, assessment year, LGBM baseline config) identity from each folder
name, reads the metric tables, and renders:

  * an Overview tab  : experiment matrix, auto-generated key findings, a global
                       "best covariance result per experiment" table, and an
                       at-a-glance small-multiples grid of the rho effect.
  * Overview comparison sections: rho effect across years (per config) and
                       across configs (per year), so rho can be compared directly.
  * one tab per experiment (year x config) with its own identifier, holding the
    metric-vs-rho evolution panels, the accuracy/equity tradeoff plot, and the
    full per-rho metric table with the best values highlighted.

Summary plots are generated here with matplotlib. Detailed rho-evolution and
tradeoff plots are linked directly from each experiment output folder. Run:

    python scripts/build_rho_sweep_dashboard.py
"""
from __future__ import annotations

import base64
import glob
import html
import io
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.motivation_utils import (
    IAAO_COD_RANGES,
    IAAO_LEVEL_RANGE,
    IAAO_PRB_RANGE,
    IAAO_PRD_RANGE,
    IAAO_VEI_RANGE,
)

ROOT = "output/rho_sweep_500_estimators"
OUT_HTML = os.path.join(ROOT, "rho_sweep_dashboard.html")

# Optional outputs from scripts/theory_informed_rho_range_v2.py + merge step.
THEORY_ROOT = os.environ.get(
    "THEORY_ROOT",
    os.path.join(os.path.dirname(ROOT), "theory_rho_range_500_estimators"),
)
THEORY_MERGED = os.path.join(THEORY_ROOT, "merged")
THEORY_PLOT_DIR = os.path.join(THEORY_MERGED, "plots", "rho_evolution_theory_empirical_global")
EMPIRICAL_RHO_RANGE_DEFAULT = os.environ.get("EMPIRICAL_RHO_RANGE", "2.56,3.54")

FOLDER_RE = re.compile(
    r"^(?P<src>.+?)_assess(?P<year>\d{4})__(?P<cfg>.+)_(?P<cid>[0-9a-f]{8})$"
)

SRC_LABELS = {
    "ccao2025": "CCAO 2025 (full sales history)",
    "ccao_old": "CCAO old-2024 vintage",
    "ccao_sim2024": "CCAO sim-2024 vintage",
    "ccao_sim2023": "CCAO sim-2023 vintage",
}
CFG_LABELS = {
    "test_best_r2": "test_best_r2 \u2014 tuned baseline (lr 0.013, 1209 leaves)",
    "cv_top1_r2": "cv_top1_r2 \u2014 CV-best #1 (lr 0.036, 1554 leaves)",
    "cv_top2_r2": "cv_top2_r2 \u2014 CV-best #2 (lr 0.021, 1245 leaves)",
}
CFG_ORDER = {"cv_top1_r2": 0, "test_best_r2": 1, "cv_top2_r2": 2}

# (display, csv column, ideal, fmt) ; ideal in {"up","down","one","zero"}
METRICS = [
    ("R\u00b2", "R2", "up", "{:.3f}"),
    ("RMSE", "RMSE", "down", "{:,.0f}"),
    ("MAE", "MAE", "down", "{:,.0f}"),
    ("MAPE (%)", "MAPE", "down", "{:.2f}"),
    ("MdAPE (%)", "MdAPE", "down", "{:.2f}"),
    ("COD", "COD", "down", "{:.2f}"),
    ("PRD", "PRD", "one", "{:.3f}"),
    ("PRB", "PRB", "zero", "{:.3f}"),
    ("MKI", "MKI", "one", "{:.3f}"),
    ("VEI", "VEI", "zero", "{:.2f}"),
    ("Median ratio", "Median ratio", "one", "{:.3f}"),
]

NASH_METRICS = [
    ("R2", "up"),
    ("MdAPE", "down"),
    ("COD", "down"),
    ("PRB", "zero"),
]

# Number-free label for the recommended-rho operating point. The actual rho value is
# resolved dynamically (geometric centre of the empirical-vs-theory overlap) and shown
# in the rho column / legends, so the label itself carries no hard-coded number.
REC_LABEL = "Cov rec-\u03c1"

OVERVIEW_MARKS = {
    "Linear": "\u2605",
    "LGBM": "\u25cf",
    REC_LABEL: "\u25ce",
    "Cov min-RMSE": "\u25c6",
    "Cov Nash-volume": "\u2726",
}

METRIC_DISPLAY_SCALE = {"MAPE": 100.0}
METRIC_ROUND_DIGITS = {
    "R2": 3,
    "RMSE": 0,
    "MAE": 0,
    "MAPE": 2,
    "MdAPE": 2,
    "COD": 2,
    "PRD": 3,
    "PRB": 3,
    "MKI": 3,
    "VEI": 2,
    "Median ratio": 3,
}

IAAO_BANDS = {
    "COD": (*IAAO_COD_RANGES["Residential Improved"], None),
    "PRD": (*IAAO_PRD_RANGE, 1.0),
    "PRB": (*IAAO_PRB_RANGE, 0.0),
    "VEI": (*IAAO_VEI_RANGE, 0.0),
    "Median ratio": (*IAAO_LEVEL_RANGE, 1.0),
}

R2LAB = "R\u00b2"
COL_LINEAR = "#6c757d"
COL_LGBM = "#0d6efd"
COL_COV = "#d6336c"
COL_COV_TEST = "#f59f00"

# Recommended rho and operating range. These are DEFAULTS only; build_html() overrides
# them at render time with the values derived from the (corrected) empirical and theory
# criteria via load_theory_context(). The recommended rho is the geometric centre of the
# empirical-vs-theory overlap, and the range is that overlap [both_low, both_high].
RECOMMENDED_RHO = 3.38
REC_RHO_RANGE = (3.24, 3.54)
REC_RANGE_COLOR = "#6f42c1"  # violet: distinct from the green/red metric highlights

# Decile-ratio matrix: the most distinguishable basic colours + distinct markers,
# so the five overlaid series stay readable even on a small panel.
DECILE_SERIES_COLORS = {
    "Linear": "#000000",        # black
    "LGBM": "#1f77b4",          # blue
    REC_LABEL: "#e41a1c",       # red
    "Cov min-RMSE": "#1a9e1a",  # green
    "Cov Nash": "#ff7f0e",      # orange
}
DECILE_SERIES_MARKERS = {
    "Linear": "o",
    "LGBM": "s",
    REC_LABEL: "D",
    "Cov min-RMSE": "^",
    "Cov Nash": "v",
}
METRIC_COLORS = {
    "R2": "#111827",
    "MdAPE": "#F97316",
    "COD": "#7C3AED",
    "PRD": "#16A34A",
    "PRB": "#2563EB",
    "VEI": "#DC2626",
    "Median ratio": "#0F766E",
}


@dataclass
class Experiment:
    folder: str
    src: str
    year: str
    cfg: str
    cid: str
    test: pd.DataFrame
    assess: pd.DataFrame
    boot: Optional[pd.DataFrame] = None

    @property
    def key(self) -> str:
        return f"{self.src}_{self.year}_{self.cfg}"

    @property
    def title(self) -> str:
        return f"{self.year} \u00b7 {self.cfg}"


def _experiment_sort_key(exp: Experiment):
    return (int(exp.year), CFG_ORDER.get(exp.cfg, 99), exp.src)


def _split_models(df: pd.DataFrame):
    lin = df[df.model_name == "LinearRegression"]
    lgb = df[df.model_name == "LGBMRegressor"]
    cov = df[df.model_name.str.startswith("LGBCovPenalty")].copy()
    cov = cov.sort_values("rho")
    lin = lin.iloc[0] if len(lin) else None
    lgb = lgb.iloc[0] if len(lgb) else None
    return lin, lgb, cov


def load_experiments() -> List[Experiment]:
    exps: List[Experiment] = []
    for fo in sorted(glob.glob(os.path.join(ROOT, "*/"))):
        name = os.path.basename(fo.rstrip("/"))
        m = FOLDER_RE.match(name)
        f_assess = os.path.join(fo, "quick_test_metrics_assess.csv")
        f_test = os.path.join(fo, "quick_test_metrics_test.csv")
        if not m or not (os.path.exists(f_assess) and os.path.exists(f_test)):
            continue
        boot_path = os.path.join(fo, "quick_test_metrics_validation_bootstrap_avg.csv")
        boot = pd.read_csv(boot_path) if os.path.exists(boot_path) else None
        exps.append(
            Experiment(
                folder=fo,
                src=m["src"],
                year=m["year"],
                cfg=m["cfg"],
                cid=m["cid"],
                test=pd.read_csv(f_test),
                assess=pd.read_csv(f_assess),
                boot=boot,
            )
        )
    return sorted(exps, key=_experiment_sort_key)


# --------------------------------------------------------------------------- #
# plotting helpers
# --------------------------------------------------------------------------- #
# Render quality. PNG is used for crisp line/scatter panels; JPEG (high quality)
# for the large multi-panel grids to keep the embedded payload manageable.
PNG_DPI = 165
JPG_DPI = 160
JPG_QUALITY = 92


def _png(fig, dpi: int = PNG_DPI) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")


def _jpg(fig, dpi: int = JPG_DPI) -> str:
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="jpg",
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"quality": JPG_QUALITY, "optimize": True},
    )
    plt.close(fig)
    buf.seek(0)
    return "data:image/jpeg;base64," + base64.b64encode(buf.read()).decode("ascii")


def _hline(ax, val, color, label):
    if val is not None and np.isfinite(val):
        ax.axhline(val, color=color, ls="--", lw=1.4, label=label, alpha=0.9)


def _iaao_band(ax, col: str) -> bool:
    band = IAAO_BANDS.get(col)
    if not band:
        return False
    lo, hi, target = band
    ax.axhspan(lo, hi, color="#2f9e44", alpha=0.12, label="IAAO band", zorder=0)
    if target is not None:
        ax.axhline(target, color="#2f9e44", ls=":", lw=1.2, label="IAAO target", alpha=0.9)
    return True


def _iaao_xband(ax, col: str) -> bool:
    band = IAAO_BANDS.get(col)
    if not band:
        return False
    lo, hi, target = band
    ax.axvspan(lo, hi, color="#2f9e44", alpha=0.12, label="IAAO band", zorder=0)
    if target is not None:
        ax.axvline(target, color="#2f9e44", ls=":", lw=1.2, alpha=0.9)
    return True


def fig_rho_evolution(exp: Experiment) -> str:
    panels = [
        ("R\u00b2", "R2", None),
        ("RMSE", "RMSE", None),
        ("COD (uniformity)", "COD", None),
        ("PRD (vertical equity)", "PRD", 1.0),
        ("PRB", "PRB", 0.0),
        ("MdAPE", "MdAPE", None),
    ]
    lin_a, lgb_a, cov_a = _split_models(exp.assess)
    lin_t, lgb_t, cov_t = _split_models(exp.test)
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2))
    for ax, (title, col, ideal) in zip(axes.ravel(), panels):
        ax.plot(cov_a["rho"], cov_a[col], "-o", ms=3, color=COL_COV, label="Cov (assessment)")
        ax.plot(cov_t["rho"], cov_t[col], "-o", ms=2.5, color=COL_COV_TEST, alpha=0.75, label="Cov (test)")
        _hline(ax, None if lin_a is None else lin_a[col], COL_LINEAR, "Linear (assess)")
        _hline(ax, None if lgb_a is None else lgb_a[col], COL_LGBM, "LGBM (assess)")
        if not _iaao_band(ax, col) and ideal is not None:
            ax.axhline(ideal, color="#2f9e44", ls=":", lw=1.2, alpha=0.8)
        ax.set_xlabel("rho")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(alpha=0.25)
    axes.ravel()[0].legend(fontsize=7, loc="best")
    fig.suptitle(f"Metric evolution vs rho \u2014 {exp.title}", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return _png(fig)


def fig_tradeoff(exp: Experiment) -> str:
    lin, lgb, cov = _split_models(exp.assess)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    specs = [("COD (lower = more uniform)", "COD"), ("PRD (1.0 = vertically equitable)", "PRD")]
    for ax, (xlab, xcol) in zip(axes, specs):
        sc = ax.scatter(cov[xcol], cov["R2"], c=cov["rho"], cmap="viridis", s=40, zorder=3)
        ax.plot(cov[xcol], cov["R2"], "-", color=COL_COV, lw=1, alpha=0.5, zorder=2)
        if lin is not None:
            ax.scatter([lin[xcol]], [lin["R2"]], marker="*", s=320, color=COL_LINEAR,
                       edgecolor="black", linewidth=0.6, zorder=5, label="Linear")
        if lgb is not None:
            ax.scatter([lgb[xcol]], [lgb["R2"]], marker="^", s=160, color=COL_LGBM,
                       edgecolor="black", linewidth=0.6, zorder=5, label="LGBM baseline")
        _iaao_xband(ax, xcol)
        ax.set_xlabel(xlab)
        ax.set_ylabel("R\u00b2 (accuracy)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="best")
    cb = fig.colorbar(sc, ax=axes, fraction=0.046, pad=0.02)
    cb.set_label("rho")
    fig.suptitle(f"Accuracy / equity tradeoff (assessment) \u2014 {exp.title}", fontsize=12, fontweight="bold")
    return _png(fig)


def fig_compare(exps: List[Experiment], group_by: str, fixed: str, col: str, ylabel: str, ideal=None) -> str:
    """Overlay metric-vs-rho lines. group_by='cfg' fixes a config, varies year (and vice versa)."""
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    cmap = plt.get_cmap("tab10")
    i = 0
    for e in exps:
        if group_by == "cfg" and e.cfg != fixed:
            continue
        if group_by == "year" and e.year != fixed:
            continue
        _, _, cov = _split_models(e.assess)
        lab = e.year if group_by == "cfg" else e.cfg
        ax.plot(cov["rho"], cov[col], "-o", ms=3, color=cmap(i % 10), label=lab)
        i += 1
    if not _iaao_band(ax, col) and ideal is not None:
        ax.axhline(ideal, color="#2f9e44", ls=":", lw=1.2)
    ax.set_xlabel("rho")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, title=("year" if group_by == "cfg" else "config"))
    ttl = (CFG_LABELS.get(fixed, fixed).split(" \u2014")[0] if group_by == "cfg"
           else f"assessment year {fixed}")
    ax.set_title(f"{ylabel} vs rho \u2014 {ttl}", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _png(fig)


def fig_small_multiples(exps: List[Experiment], col: str, ylabel: str, ideal=None) -> str:
    years = sorted({e.year for e in exps})
    cfgs = ["cv_top1_r2", "test_best_r2", "cv_top2_r2"]
    cfgs = [c for c in cfgs if any(e.cfg == c for e in exps)]
    by = {(e.year, e.cfg): e for e in exps}
    fig, axes = plt.subplots(len(years), len(cfgs), figsize=(3.6 * len(cfgs), 2.5 * len(years)),
                             squeeze=False, sharex=True)
    for r, yr in enumerate(years):
        for c, cf in enumerate(cfgs):
            ax = axes[r][c]
            e = by.get((yr, cf))
            if e is None:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center"); ax.set_axis_off(); continue
            lin, lgb, cov = _split_models(e.assess)
            ax.plot(cov["rho"], cov[col], "-", color=COL_COV, lw=1.6, label="cov")
            _hline(ax, None if lin is None else lin[col], COL_LINEAR, "lin")
            _hline(ax, None if lgb is None else lgb[col], COL_LGBM, "lgbm")
            if not _iaao_band(ax, col) and ideal is not None:
                ax.axhline(ideal, color="#2f9e44", ls=":", lw=1.0)
            ax.grid(alpha=0.2)
            if r == 0:
                ax.set_title(cf, fontsize=9, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"{yr}\n{ylabel}", fontsize=9)
    axes[0][0].legend(fontsize=6, loc="best")
    fig.suptitle(f"{ylabel} vs rho \u2014 all experiments (rows = assessment year, cols = config)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return _png(fig)


def _split_df(exp: Experiment, split: str) -> Optional[pd.DataFrame]:
    if split in ("assess", "assessment"):
        return exp.assess
    if split == "test":
        return exp.test
    if split in ("validation", "validation_bootstrap_average"):
        return exp.boot
    return None


def _split_label(split: str) -> str:
    return {
        "assess": "Assessment",
        "assessment": "Assessment",
        "test": "Test",
        "validation": "Validation bootstrap avg",
        "validation_bootstrap_average": "Validation bootstrap avg",
    }.get(split, split)


def _metric_spec(col: str):
    for disp, c, ideal, fmt in METRICS:
        if c == col:
            return disp, ideal, fmt
    return col, "down", "{:.3f}"


def _rec_rho_overlay(ax, *, label: bool = True) -> None:
    """Shade the recommended rho range and mark the recommended rho on a rho x-axis."""
    lo, hi = REC_RHO_RANGE
    ax.axvspan(lo, hi, color=REC_RANGE_COLOR, alpha=0.12, zorder=0,
               label=(f"recommended rho range [{lo:g}, {hi:g}]" if label else None))
    ax.axvline(RECOMMENDED_RHO, color=REC_RANGE_COLOR, ls="-.", lw=1.5, alpha=0.9,
               label=(f"recommended rho = {RECOMMENDED_RHO:g}" if label else None))


def fig_normalized_evolution(exp: Experiment, split: str) -> Optional[str]:
    df = _split_df(exp, split)
    if df is None:
        return None
    _, _, cov = _split_models(df)
    if cov.empty:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    for col, color in METRIC_COLORS.items():
        disp, ideal, _ = _metric_spec(col)
        q = np.array([_metric_quality(v, col, ideal) for v in cov[col]], dtype=float)
        if not np.isfinite(q).any():
            continue
        lo, hi = np.nanmin(q), np.nanmax(q)
        norm = np.full_like(q, 0.5, dtype=float) if np.isclose(hi, lo) else (q - lo) / (hi - lo)
        ax.plot(cov["rho"], norm, "-o", ms=2.6, lw=1.4, color=color, label=disp.replace(" (%)", ""))
    _rec_rho_overlay(ax)
    ax.set_xlabel("rho")
    ax.set_ylabel("normalized score (0=worst, 1=best)")
    ax.set_ylim(-0.04, 1.04)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.42), frameon=False)
    ax.set_title(f"Normalized metric evolution vs rho - {exp.title} ({_split_label(split)})",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    return _jpg(fig)


def fig_single_metric_evolution(exp: Experiment, split: str, metric: str) -> Optional[str]:
    col = {"r2": "R2", "mdape": "MdAPE", "cod": "COD", "prd": "PRD", "prb": "PRB", "vei": "VEI"}[metric]
    df = _split_df(exp, split)
    if df is None:
        return None
    lin, lgb, cov = _split_models(df)
    if cov.empty:
        return None
    disp, _, _ = _metric_spec(col)
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    ax.plot(cov["rho"], cov[col], "-o", ms=2.8, lw=1.4, color=METRIC_COLORS.get(col, COL_COV), label="Cov")
    _hline(ax, None if lin is None else lin[col], COL_LINEAR, "Linear")
    _hline(ax, None if lgb is None else lgb[col], COL_LGBM, "LGBM")
    _iaao_band(ax, col)
    ax.set_xlabel("rho")
    ax.set_ylabel(disp)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="best")
    ax.set_title(f"{disp} vs rho - {exp.title} ({_split_label(split)})", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return _jpg(fig)


def fig_metric_tradeoff(exp: Experiment, split: str, metric_col: str) -> Optional[str]:
    df = _split_df(exp, split)
    if df is None:
        return None
    lin, lgb, cov = _split_models(df)
    if cov.empty:
        return None
    metric_disp, _, _ = _metric_spec(metric_col)
    y_specs = [("PRD", "PRD"), ("PRB", "PRB"), ("VEI", "VEI")]
    rec = _cov_row_at_rho(cov, RECOMMENDED_RHO)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), squeeze=False)
    axes = axes.ravel()
    sc = None
    for ax, (ylabel, ycol) in zip(axes, y_specs):
        sc = ax.scatter(cov[metric_col], cov[ycol], c=cov["rho"], cmap="viridis", s=24, zorder=3)
        ax.plot(cov[metric_col], cov[ycol], "-", color=COL_COV, lw=1.0, alpha=0.55, zorder=2)
        if lin is not None:
            ax.scatter([lin[metric_col]], [lin[ycol]], marker="*", s=140, color=COL_LINEAR,
                       edgecolor="black", linewidth=0.5, zorder=5, label="Linear")
        if lgb is not None:
            ax.scatter([lgb[metric_col]], [lgb[ycol]], marker="^", s=80, color=COL_LGBM,
                       edgecolor="black", linewidth=0.5, zorder=5, label="LGBM")
        if rec is not None:
            ax.scatter([rec[metric_col]], [rec[ycol]], marker="*", s=300, color=REC_RANGE_COLOR,
                       edgecolor="black", linewidth=0.8, zorder=7,
                       label=f"recommended rho={RECOMMENDED_RHO:g}")
        _iaao_xband(ax, metric_col)
        _iaao_band(ax, ycol)
        ax.set_xlabel(metric_disp)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7, loc="best")
    if sc is not None:
        cb = fig.colorbar(sc, ax=axes, fraction=0.035, pad=0.02)
        cb.set_label("rho")
    fig.suptitle(f"Tradeoff vs {metric_disp} - {exp.title} ({_split_label(split)})",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    return _jpg(fig)


# --------------------------------------------------------------------------- #
# table helpers
# --------------------------------------------------------------------------- #
def _ideal_target(ideal: str):
    return {"one": 1.0, "zero": 0.0}.get(ideal)


def _metric_scale(col: Optional[str]) -> float:
    return METRIC_DISPLAY_SCALE.get(col or "", 1.0)


def _metric_display_value(v, col: Optional[str] = None) -> float:
    return float(v) * _metric_scale(col)


def _fmt(v, fmt, col: Optional[str] = None):
    if v is None:
        return "&ndash;"
    try:
        x = _metric_display_value(v, col)
        if not np.isfinite(x):
            return "&ndash;"
        return fmt.format(x)
    except Exception:
        return html.escape(str(v))


def _metric_quality(v, col: str, ideal: str) -> float:
    try:
        x = _metric_display_value(v, col)
    except Exception:
        return np.nan
    if not np.isfinite(x):
        return np.nan
    tgt = _ideal_target(ideal)
    if ideal == "up":
        return x
    if ideal == "down":
        return -x
    if tgt is not None:
        return -abs(x - tgt * _metric_scale(col))
    return np.nan


def _highlight_sets(rows, metric_specs):
    best: Dict[str, set] = {}
    worst: Dict[str, set] = {}
    for _, col, ideal, _ in metric_specs:
        vals = []
        digits = METRIC_ROUND_DIGITS.get(col, 3)
        for i, r in enumerate(rows):
            if r is None or col not in r:
                continue
            q = _metric_quality(r[col], col, ideal)
            if np.isfinite(q):
                vals.append((i, round(q, digits)))
        if not vals:
            continue
        best_q = max(q for _, q in vals)
        worst_q = min(q for _, q in vals)
        best[col] = {i for i, q in vals if q == best_q}
        worst[col] = {i for i, q in vals if q == worst_q}
    return best, worst


def _cell_class(i: int, col: str, best, worst) -> str:
    classes = []
    if i in best.get(col, set()):
        classes.append("best")
    if i in worst.get(col, set()):
        classes.append("worst")
    return f" class='{' '.join(classes)}'" if classes else ""


def _cov_row_at_rho(cov: pd.DataFrame, rho):
    if cov.empty or rho is None or not np.isfinite(float(rho)):
        return None
    dist = (cov["rho"].astype(float) - float(rho)).abs()
    return cov.loc[dist.idxmin()]


def _quality_matrix(cov: pd.DataFrame, specs) -> tuple[np.ndarray, np.ndarray]:
    quality = []
    valid = np.ones(len(cov), dtype=bool)
    for col, ideal in specs:
        q = cov[col].map(lambda v: _metric_quality(v, col, ideal)).to_numpy(dtype=float)
        valid &= np.isfinite(q)
        quality.append(q)
    return np.column_stack(quality), valid


def _pareto_efficient(quality: np.ndarray) -> np.ndarray:
    efficient = np.ones(len(quality), dtype=bool)
    for i, q in enumerate(quality):
        if not efficient[i]:
            continue
        dominated = np.all(quality >= q, axis=1) & np.any(quality > q, axis=1)
        if dominated.any():
            efficient[i] = False
    return efficient


def _nash_volume_idx(cov: pd.DataFrame):
    quality, valid = _quality_matrix(cov, NASH_METRICS)
    if not valid.any():
        return None

    valid_pos = np.flatnonzero(valid)
    qv = quality[valid]
    pareto = _pareto_efficient(qv)
    eps = 1e-6
    log_product = np.full(len(qv), -np.inf)
    utility_cols = []
    for j in range(qv.shape[1]):
        col = qv[:, j]
        lo, hi = np.nanmin(col), np.nanmax(col)
        if np.isclose(hi, lo):
            utility = np.ones(len(col))
        else:
            utility = eps + (1.0 - eps) * (col - lo) / (hi - lo)
        utility_cols.append(utility)
    utilities = np.column_stack(utility_cols)
    log_product[pareto] = np.log(utilities[pareto]).sum(axis=1)
    return cov.index[valid_pos[int(np.argmax(log_product))]]


def best_cov_row(exp: Experiment, by: str = "COD", result_split: str = "assess"):
    """Return result-split row at the rho selected on validation by the given criterion."""
    select_df = exp.boot if exp.boot is not None else exp.assess
    _, _, select_cov = _split_models(select_df)
    result_df = exp.test if result_split == "test" else exp.assess
    _, _, result_cov = _split_models(result_df)
    if select_cov.empty or result_cov.empty:
        return None
    if by == "Nash":
        idx = _nash_volume_idx(select_cov)
    elif by in ("COD", "RMSE", "MAE", "MAPE", "MdAPE"):
        idx = select_cov["COD"].idxmin()
        if by != "COD":
            idx = select_cov[by].idxmin()
    elif by == "PRD":
        idx = (select_cov["PRD"] - 1.0).abs().idxmin()
    else:
        idx = select_cov[by].idxmax()
    if idx is None:
        return None
    return _cov_row_at_rho(result_cov, select_cov.loc[idx]["rho"])


def _overview_model_label(tag: str) -> str:
    mark = OVERVIEW_MARKS.get(tag)
    if not mark:
        return html.escape(tag)
    return f"<span class='criterion-mark'>{mark}</span> {html.escape(tag)}"


def _overview_mark_legend() -> str:
    items = [REC_LABEL, "Cov min-RMSE", "Cov Nash-volume"]
    return " ".join(
        f"<span class='legend-mark'><span class='criterion-mark'>{OVERVIEW_MARKS[x]}</span> {html.escape(x)}</span>"
        for x in items
    )


def metric_table_html(exp: Experiment, split: str) -> str:
    df = exp.assess if split == "assess" else exp.test
    lin, lgb, cov = _split_models(df)
    lo, hi = REC_RHO_RANGE
    rows = []
    if lin is not None:
        rows.append(("Linear", "&ndash;", lin, "row-linear"))
    if lgb is not None:
        rows.append(("LGBM baseline", "&ndash;", lgb, "row-lgbm"))
    for _, r in cov.iterrows():
        rho_v = float(r["rho"])
        in_rng = lo <= rho_v <= hi
        # Subtle marker (thin violet left rule + small caret) only on the rho cell, so it
        # does not compete with the green/red metric highlights.
        rho_cell = (f"<td class='rho-rec' title='inside recommended rho range "
                    f"[{lo:g}, {hi:g}]'><span class='rec-tick'>\u2023</span>{rho_v:.3g}</td>"
                    if in_rng else f"<td>{rho_v:.3g}</td>")
        rows.append(("Cov", rho_cell, r, "row-cov"))

    best, worst = _highlight_sets([r for _, _, r, _ in rows], METRICS)

    head = "<tr><th>Model</th><th>rho</th>" + "".join(
        f"<th>{html.escape(d)}</th>" for d, *_ in METRICS) + "</tr>"
    body = []
    for i, (mname, rho, r, cls) in enumerate(rows):
        rho_td = rho if str(rho).startswith("<td") else f"<td>{rho}</td>"
        tds = [f"<td class='mname'>{mname}</td>{rho_td}"]
        for disp, col, ideal, fmt in METRICS:
            tds.append(f"<td{_cell_class(i, col, best, worst)}>{_fmt(r[col], fmt, col)}</td>")
        body.append(f"<tr class='{cls}'>" + "".join(tds) + "</tr>")
    return (f"<div class='tablewrap'><table class='metrics'><thead>{head}</thead>"
            f"<tbody>{''.join(body)}</tbody></table></div>")


def _delta_fmt(d: float, col: Optional[str]) -> str:
    digits = METRIC_ROUND_DIGITS.get(col or "", 3)
    if digits <= 0:
        return f"{d:+,.0f}"
    return f"{d:+,.{digits}f}"


def delta_badge(value, baseline, ideal, col: Optional[str] = None):
    """Coloured delta vs baseline (green = improvement)."""
    if value is None or baseline is None:
        return ""
    value = _metric_display_value(value, col)
    baseline = _metric_display_value(baseline, col)
    if not np.isfinite(value) or not np.isfinite(baseline):
        return ""
    tgt = _ideal_target(ideal)
    if ideal == "up":
        better = value > baseline
        d = value - baseline
    elif ideal == "down":
        better = value < baseline
        d = value - baseline
    elif tgt is not None:
        tgt = tgt * _metric_scale(col)
        better = abs(value - tgt) < abs(baseline - tgt)
        d = abs(value - tgt) - abs(baseline - tgt)
    else:
        return ""
    cls = "flat" if np.isclose(d, 0.0) else ("up" if better else "down")
    return f"<span class='delta {cls}'>{_delta_fmt(d, col)}</span>"


def _img_embed(src: Optional[str], title: str, height: int = 280) -> str:
    if not src:
        return "<div class='missing'>plot not found</div>"
    ttl = html.escape(title)
    return (
        f"<img class='plot artifactplot' loading='lazy' style='max-height:{int(height)}px;"
        f"object-fit:contain' title='{ttl}' src='{src}'>"
    )


def artifact_table_html(
    exps: List[Experiment],
    *,
    title: str,
    plot_getter,
    height: int = 250,
) -> str:
    rows = []
    for exp in sorted(exps, key=_experiment_sort_key):
        src = plot_getter(exp)
        rows.append(
            "<tr>"
            f"<td class='mname'>{html.escape(exp.year)}</td>"
            f"<td>{html.escape(exp.cfg)}</td>"
            f"<td class='plotcell'>{_img_embed(src, f'{title} {exp.title}', height=height)}</td>"
            "</tr>"
        )
    return (
        "<div class='tablewrap plot-table'><table class='metrics'>"
        f"<thead><tr><th>Data year</th><th>Baseline model</th><th>{html.escape(title)}</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def compact_plot_links(exp: Experiment, split: str) -> str:
    return ""


# --------------------------------------------------------------------------- #
# decile median-ratio matrix (year x config)
# --------------------------------------------------------------------------- #
DECILE_BAND = (0.95, 1.05)  # IAAO level target shown on the decile plots (per reference figure)


def _decile_ratios(row, n: int = 10) -> Optional[np.ndarray]:
    """Median prediction-to-price ratio per actual-price decile from a metric row."""
    if row is None:
        return None
    vals = []
    for k in range(1, n + 1):
        col = f"MedianRatio_q{n}_bin{k}"
        try:
            vals.append(float(row[col]) if col in row else np.nan)
        except Exception:
            vals.append(np.nan)
    arr = np.asarray(vals, dtype=float)
    return arr if np.isfinite(arr).any() else None


def _decile_series_for(exp: Experiment, split: str = "assess"):
    """Ordered (label, color, selected_rho, decile_vector) for the two baselines and three rhos."""
    result_df = exp.assess if split in ("assess", "assessment") else exp.test
    lin, lgb, result_cov = _split_models(result_df)
    cov_final = _cov_row_at_rho(result_cov, RECOMMENDED_RHO)
    rmse_row = best_cov_row(exp, "RMSE", split)
    nash_row = best_cov_row(exp, "Nash", split)

    def _rho_of(row):
        if row is None or "rho" not in row or pd.isna(row["rho"]):
            return None
        return float(row["rho"])

    return [
        ("Linear", DECILE_SERIES_COLORS["Linear"], None, _decile_ratios(lin)),
        ("LGBM", DECILE_SERIES_COLORS["LGBM"], None, _decile_ratios(lgb)),
        (REC_LABEL, DECILE_SERIES_COLORS[REC_LABEL], _rho_of(cov_final), _decile_ratios(cov_final)),
        ("Cov min-RMSE", DECILE_SERIES_COLORS["Cov min-RMSE"], _rho_of(rmse_row), _decile_ratios(rmse_row)),
        ("Cov Nash", DECILE_SERIES_COLORS["Cov Nash"], _rho_of(nash_row), _decile_ratios(nash_row)),
    ]


def fig_decile_matrix(exps: List[Experiment], split: str = "assess") -> str:
    years = sorted({e.year for e in exps})
    cfgs = [c for c in ("cv_top1_r2", "test_best_r2", "cv_top2_r2") if any(e.cfg == c for e in exps)]
    by = {(e.year, e.cfg): e for e in exps}
    nyears, ncfgs = len(years), len(cfgs)
    fig, axes = plt.subplots(nyears, ncfgs, figsize=(4.2 * ncfgs, 2.8 * nyears),
                             squeeze=False, sharex=True, sharey=True)
    x = np.arange(1, 11)
    lo, hi = DECILE_BAND
    legend_handles: Dict[str, Any] = {}
    for r, yr in enumerate(years):
        for c, cf in enumerate(cfgs):
            ax = axes[r][c]
            e = by.get((yr, cf))
            if e is None:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center")
                ax.set_axis_off()
                continue
            ax.axhspan(lo, hi, color="#2f9e44", alpha=0.12, zorder=0)
            ax.axhline(1.0, color="#2f9e44", ls=":", lw=1.1, alpha=0.9, zorder=1)
            ann = []
            for label, color, rho_v, vec in _decile_series_for(e, split):
                if vec is None:
                    continue
                mk = DECILE_SERIES_MARKERS.get(label, "o")
                ln, = ax.plot(x, vec, "-", marker=mk, ms=4.2, lw=1.5, color=color,
                              markeredgecolor="white", markeredgewidth=0.4, label=label)
                legend_handles.setdefault(label, ln)
                if rho_v is not None and label != REC_LABEL:
                    ann.append(f"{label.replace('Cov ', '')} \u03c1={rho_v:.3g}")
            if ann:
                ax.text(0.02, 0.04, "   ".join(ann), transform=ax.transAxes, fontsize=6.4,
                        va="bottom", ha="left", color="#444",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#ddd", alpha=0.85))
            ax.grid(alpha=0.2)
            ax.set_xticks(x)
            if r == 0:
                ax.set_title(cf, fontsize=9.5, fontweight="bold")
            if r == nyears - 1:
                ax.set_xticklabels([f"D{i}" for i in x], fontsize=7)
                ax.set_xlabel("sale-price decile (D1=cheapest)", fontsize=8)
            if c == 0:
                ax.set_ylabel(f"{yr}\nmedian ratio", fontsize=9)
    if legend_handles:
        fig.legend(list(legend_handles.values()), list(legend_handles.keys()),
                   loc="upper center", ncol=len(legend_handles), fontsize=9,
                   frameon=False, bbox_to_anchor=(0.5, 1.005))
    fig.suptitle("Median prediction-to-price ratio by sale-price decile \u2014 assessment split "
                 "(rows = assessment year, cols = LGBM baseline config)",
                 fontsize=11, fontweight="bold", y=1.035)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    return _jpg(fig, dpi=150)


# --------------------------------------------------------------------------- #
# HTML assembly
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# theory-vs-empirical recommendation helpers
# --------------------------------------------------------------------------- #
def _parse_float_pair(raw: str) -> Tuple[float, float]:
    vals = []
    for token in str(raw or "").split(","):
        token = token.strip()
        if token:
            try:
                vals.append(float(token))
            except ValueError:
                pass
    if len(vals) != 2 or not all(np.isfinite(vals)):
        return (np.nan, np.nan)
    return (float(min(vals)), float(max(vals)))


def _num(v: Any) -> float:
    try:
        out = float(v)
    except Exception:
        return np.nan
    return out if np.isfinite(out) else np.nan


def _rho_fmt(v: Any, digits: int = 3) -> str:
    x = _num(v)
    return "n/a" if not np.isfinite(x) else f"{x:.{digits}g}"


def _range_text(lo: Any, hi: Any, digits: int = 3) -> str:
    return f"[{_rho_fmt(lo, digits)}, {_rho_fmt(hi, digits)}]"


def _geom_mid(lo: float, hi: float) -> float:
    if np.isfinite(lo) and np.isfinite(hi) and lo > 0.0 and hi >= lo:
        return float(np.sqrt(lo * hi))
    return np.nan


def compute_empirical_range(
    exps: List[Experiment], budget: float = 0.01, equity_frac: float = 0.25
) -> Tuple[float, float]:
    """Derive the empirical operating range from the validation sweep (median across runs).

    Upper edge (accuracy budget): largest cov rho whose validation RMSE stays within
    ``budget`` (1%) of the model's best cov RMSE.

    Lower edge (equity onset, genuinely empirical): smallest accuracy-feasible rho at
    which the median normalized improvement of the vertical-equity errors
    |PRD-1|, |PRB|, |VEI| reaches ``equity_frac`` (25%) of their achievable gain, where
    the per-metric normalized improvement is
        I_m(rho) = (E_m(baseline) - E_m(rho)) / (E_m(baseline) - min_rho E_m(rho)),
    and E_m(baseline) is the unpenalised LGBM error.
    """
    lows: List[float] = []
    uppers: List[float] = []
    equity_specs = [("PRD", 1.0), ("PRB", 0.0), ("VEI", 0.0)]
    for e in exps:
        df = e.boot if e.boot is not None else e.assess
        if df is None or df.empty:
            continue
        _, lgb, cov = _split_models(df)
        if cov.empty or "RMSE" not in cov.columns:
            continue
        rmse = pd.to_numeric(cov["RMSE"], errors="coerce").to_numpy(dtype=float)
        rho = pd.to_numeric(cov["rho"], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(rmse) & np.isfinite(rho)
        if int(ok.sum()) < 2:
            continue
        rmse, rho = rmse[ok], rho[ok]
        feasible = rmse <= float(np.nanmin(rmse)) * (1.0 + budget)
        if feasible.any():
            uppers.append(float(np.nanmax(rho[feasible])))

        if lgb is None:
            continue
        improvements = []
        for m, ideal in equity_specs:
            if m not in cov.columns or m not in lgb.index:
                continue
            e0 = abs(float(lgb[m]) - ideal)
            er = np.abs(pd.to_numeric(cov[m], errors="coerce").to_numpy(dtype=float)[ok] - ideal)
            denom = e0 - float(np.nanmin(er))
            if denom > 1e-12:
                improvements.append((e0 - er) / denom)
            else:
                improvements.append(np.where(er <= e0, 1.0, 0.0))
        if not improvements:
            continue
        i_equity = np.nanmedian(np.vstack(improvements), axis=0)
        qualifies = (i_equity >= equity_frac) & feasible
        if qualifies.any():
            lows.append(float(np.nanmin(rho[qualifies])))

    emp_low = float(np.nanmedian(lows)) if lows else float("nan")
    emp_high = float(np.nanmedian(uppers)) if uppers else float("nan")
    return emp_low, emp_high


def load_theory_context(
    empirical_low_override: Optional[float] = None,
    empirical_high_override: Optional[float] = None,
) -> Dict[str, Any]:
    """Load merged theory outputs, if present, and compute dashboard recommendations.

    Empirical operating range (data-derived, passed in from ``compute_empirical_range``):
      lower = smallest accuracy-feasible rho where median equity improvement reaches 25%;
      upper = largest rho within 1% of the best cov RMSE on validation (median across runs).
    """
    ctx: Dict[str, Any] = {
        "available": False,
        "theory_merged_dir": THEORY_MERGED,
        "theory_plot_dir": THEORY_PLOT_DIR,
    }
    agg_path = os.path.join(THEORY_MERGED, "theory_rho_aggregate_recommendation.csv")
    summary_path = os.path.join(THEORY_MERGED, "theory_rho_summary_by_run.csv")
    comparison_path = os.path.join(THEORY_MERGED, "theory_empirical_comparison.csv")

    agg = pd.DataFrame()
    summary = pd.DataFrame()
    if os.path.exists(agg_path):
        try:
            agg = pd.read_csv(agg_path)
        except Exception:
            agg = pd.DataFrame()
    if os.path.exists(summary_path):
        try:
            summary = pd.read_csv(summary_path)
        except Exception:
            summary = pd.DataFrame()

    # Fallback empirical range (only used if the aggregate CSV / override are absent).
    emp_low, emp_high = _parse_float_pair(EMPIRICAL_RHO_RANGE_DEFAULT)
    ctx.update({"empirical_low": emp_low, "empirical_high": emp_high})

    if agg.empty:
        return ctx

    row_df = agg.loc[agg.get("split_group", pd.Series(dtype=str)).astype(str).eq("all")]
    row = row_df.iloc[0] if not row_df.empty else agg.iloc[0]

    rho50 = _num(row.get("median_rho_shrink_50pct"))
    rho25 = _num(row.get("median_rho_shrink_25pct"))
    mse_low = _num(row.get("median_orange_mse_band_low"))
    mse_ref = _num(row.get("median_orange_mse_ref_rho"))
    mse_high = _num(row.get("median_orange_mse_band_high"))
    cov_low = _num(row.get("median_orange_cov_band_low"))
    cov_ref = _num(row.get("median_orange_cov_ref_rho"))
    cov_high = _num(row.get("median_orange_cov_band_high"))

    # Backward-compatible fallback if the merged theory file predates orange-band columns.
    if np.isfinite(rho50):
        if not np.isfinite(mse_low):
            mse_low = 0.25 * rho50
        if not np.isfinite(mse_ref):
            mse_ref = 0.50 * rho50
        if not np.isfinite(mse_high):
            mse_high = rho50
        if not np.isfinite(cov_low):
            cov_low = rho50 / 3.0
        if not np.isfinite(cov_ref):
            cov_ref = rho50
        if not np.isfinite(cov_high):
            cov_high = 3.0 * rho50

    th_low = _num(row.get("median_orange_overlap_low"))
    th_ref = _num(row.get("median_orange_overlap_ref_rho"))
    th_high = _num(row.get("median_orange_overlap_high"))
    if not (np.isfinite(th_low) and np.isfinite(th_high)):
        th_low = max(mse_low, cov_low) if np.isfinite(mse_low) and np.isfinite(cov_low) else np.nan
        th_high = min(mse_high, cov_high) if np.isfinite(mse_high) and np.isfinite(cov_high) else np.nan
    if not np.isfinite(th_ref):
        th_ref = _geom_mid(th_low, th_high)

    # Derived empirical operating range (passed in from the sweep; replaces the old constant).
    emp_derived = False
    if empirical_low_override is not None and np.isfinite(float(empirical_low_override)):
        emp_low = float(empirical_low_override)
        emp_derived = True
    if empirical_high_override is not None and np.isfinite(float(empirical_high_override)):
        emp_high = float(empirical_high_override)
        emp_derived = True
    ctx["empirical_low"], ctx["empirical_high"] = emp_low, emp_high

    both_low = max(emp_low, th_low) if np.isfinite(emp_low) and np.isfinite(th_low) else np.nan
    both_high = min(emp_high, th_high) if np.isfinite(emp_high) and np.isfinite(th_high) else np.nan
    both_ref = _geom_mid(both_low, both_high)

    ctx.update({
        "available": True,
        "rho25": rho25,
        "empirical_derived": emp_derived,
        "aggregate": agg,
        "summary": summary,
        "comparison_path": comparison_path,
        "mse_low": mse_low,
        "mse_ref": mse_ref,
        "mse_high": mse_high,
        "cov_low": cov_low,
        "cov_ref": cov_ref,
        "cov_high": cov_high,
        "theory_low": th_low,
        "theory_ref": th_ref,
        "theory_high": th_high,
        "both_low": both_low,
        "both_ref": both_ref,
        "both_high": both_high,
        "rho50": rho50,
        "bayes_diag": _num(row.get("median_bayes_diagnostic_C0_over_minus_B")),
        "empirical_overlap_rate": _num(row.get("empirical_overlap_rate")),
    })
    return ctx


def recommendation_card(ctx: Dict[str, Any]) -> str:
    if not ctx.get("available", False):
        return (
            "<div class='card'><h2>Bottom line: rho recommendation</h2>"
            "<p class='muted'>Merged theory outputs were not found. Run the theory merge step, then rebuild "
            "the dashboard to populate the theory-informed recommendation.</p></div>"
        )
    emp_low, emp_high = ctx["empirical_low"], ctx["empirical_high"]
    th_low, th_ref, th_high = ctx["theory_low"], ctx["theory_ref"], ctx["theory_high"]
    both_low, both_ref, both_high = ctx["both_low"], ctx["both_ref"], ctx["both_high"]
    rho50 = ctx.get("rho50", np.nan)
    rho25 = ctx.get("rho25", np.nan)
    knee_note = ""
    if np.isfinite(rho50) and np.isfinite(both_ref):
        knee_note = (
            "<p class='muted'>Why the recommended rho (\u2248"
            f"{_rho_fmt(both_ref, 3)}) is <b>below</b> the theory covariance knee (rho\u2248{_rho_fmt(rho50, 3)}, "
            "where 50% of the error\u2013price covariance is removed): equity keeps improving with rho out to the "
            "knee and beyond, but the <b>accuracy budget binds first</b>. The upper edge of the empirical range is "
            "the largest rho that keeps RMSE within 1% of the model's best achievable accuracy (median \u2248"
            f"{_rho_fmt(emp_high, 3)}), so the operating point lands short of the knee.</p>"
        )
    return (
        "<div class='card'><h2>Bottom line: rho recommendation</h2>"
        "<p class='muted'>The operating range is the agreement between the <b>data-derived</b> empirical range and the "
        "theory covariance-knee overlap. The per-run theory interval (<code>theory_range_low/high</code>) is not used "
        "because it is degenerate (low &gt; high) on this data.</p>"
        "<div class='grid3'>"
        f"<div class='recobox'><h3>Empirical range</h3><div class='big'>{_range_text(emp_low, emp_high, 4)}</div>"
        "<p class='muted'>Derived from the validation sweep (not assumed): <b>lower</b> = smallest "
        "accuracy-feasible rho where the median normalized improvement of the equity errors "
        "(|PRD\u22121|, |PRB|, |VEI|) reaches 25% of its achievable gain vs the unpenalised LGBM; "
        "<b>upper</b> = largest rho keeping validation RMSE within 1% of the best cov RMSE. Medians across the 12 runs.</p></div>"
        f"<div class='recobox'><h3>Theory knee overlap</h3><div class='big'>{_range_text(th_low, th_high, 4)}</div>"
        "<p class='muted'>Overlap of the log-MSE safe band and the covariance log-rho knee transition band "
        f"([rho/3, 3\u00b7rho] around rho\u2248{_rho_fmt(rho50, 3)}, ~25\u201375% covariance reduction). "
        f"Reference rho: <b>{_rho_fmt(th_ref, 4)}</b>.</p></div>"
        f"<div class='recobox strong'><h3>Empirical \u2229 theory</h3><div class='big'>{_range_text(both_low, both_high, 4)}</div>"
        f"<p class='muted'>Recommended reference rho: <b>{_rho_fmt(both_ref, 4)}</b>.</p></div>"
        "</div>"
        + knee_note +
        "</div>"
    )


def theory_definitions_card() -> str:
    return (
        "<div class='card'><h3>Core theory in one paragraph</h3>"
        "<p class='muted'>The covariance-penalized objective is "
        "<code>MSE_log + 0.5 · rho · Cov(e, Y)^2</code>, where "
        "<code>Y = log(price)</code> and <code>e = f(X)-Y</code>. Under the local Bayes/rank-one approximation, "
        "rho controls the remaining covariance fraction "
        "<code>q(rho)=1/(1+rho·A/2)</code>, so covariance reduction is <code>1-q(rho)</code>. "
        "The theory also predicts a log-MSE cost proportional to <code>(1-q)^2</code>. "
        "The orange MSE band marks the log-MSE transition region; the orange covariance band marks the log-rho "
        "knee transition ([rho/3, 3\u00b7rho], ~25\u201375% covariance reduction) where multiplicative increases in "
        "rho stop giving increasing covariance-correction returns. "
        "Their overlap is used as the theory-informed rho range.</p>"
        "</div>"
    )


def theory_band_table(ctx: Dict[str, Any]) -> str:
    if not ctx.get("available", False):
        return "<div class='missing'>Theory aggregate CSV not found.</div>"
    rows = [
        ("Empirical range", ctx["empirical_low"], np.nan, ctx["empirical_high"], "Derived: equity-25% onset rho (low) to 1%-accuracy-budget rho (high)."),
        ("MSE orange band", ctx["mse_low"], ctx["mse_ref"], ctx["mse_high"], "Theory log-MSE transition band."),
        ("Covariance orange band", ctx["cov_low"], ctx["cov_ref"], ctx["cov_high"], "Log-rho knee transition band ([rho/3, 3rho], ~25-75% cov reduction)."),
        ("Theory orange overlap", ctx["theory_low"], ctx["theory_ref"], ctx["theory_high"], "Overlap of MSE and covariance theory bands."),
        ("Empirical ∩ theory", ctx["both_low"], ctx["both_ref"], ctx["both_high"], "Final common range."),
    ]
    body = []
    for name, lo, ref, hi, note in rows:
        body.append(
            "<tr>"
            f"<td class='mname'>{html.escape(name)}</td>"
            f"<td>{_rho_fmt(lo, 4)}</td>"
            f"<td>{_rho_fmt(ref, 4)}</td>"
            f"<td>{_rho_fmt(hi, 4)}</td>"
            f"<td class='mname'>{html.escape(note)}</td>"
            "</tr>"
        )
    return (
        "<div class='tablewrap'><table>"
        "<thead><tr><th>range</th><th>low</th><th>reference rho</th><th>high</th><th>meaning</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def theory_aggregate_table(ctx: Dict[str, Any]) -> str:
    if not ctx.get("available", False) or ctx.get("aggregate", pd.DataFrame()).empty:
        return "<div class='missing'>Merged aggregate recommendation table not found.</div>"
    cols = [
        "split_group", "n_runs", "median_orange_mse_band_low", "median_orange_mse_ref_rho",
        "median_orange_mse_band_high", "median_orange_cov_band_low", "median_orange_cov_ref_rho",
        "median_orange_cov_band_high", "median_orange_overlap_low", "median_orange_overlap_ref_rho",
        "median_orange_overlap_high", "median_bayes_diagnostic_C0_over_minus_B",
    ]
    df = ctx["aggregate"][[c for c in cols if c in ctx["aggregate"].columns]].copy()
    for c in df.columns:
        if c != "split_group":
            df[c] = pd.to_numeric(df[c], errors="ignore")
    return "<div class='tablewrap'>" + df.to_html(index=False, escape=True, float_format=lambda x: f"{x:.4f}") + "</div>"


def _file_img_embed(path: str, title: str) -> str:
    if not os.path.exists(path):
        return f"<div class='missing'>Missing plot:<br><code>{html.escape(path)}</code></div>"
    ext = os.path.splitext(path)[1].lower().lstrip(".") or "png"
    mime = "jpeg" if ext in {"jpg", "jpeg"} else "png"
    with open(path, "rb") as f:
        src = "data:image/" + mime + ";base64," + base64.b64encode(f.read()).decode("ascii")
    return f"<img class='plot' src='{src}' alt='{html.escape(title)}'>"


def theory_plot_grid(ctx: Dict[str, Any]) -> str:
    plot_dir = ctx.get("theory_plot_dir", THEORY_PLOT_DIR)
    specs = [
        ("Test: original-scale RMSE", "rho_evolution_test_rmse_frac_delta.png"),
        ("Test: log-MSE", "rho_evolution_test_mse_log_frac_delta.png"),
        ("Test: PRD improvement", "rho_evolution_test_prd_error_reduction.png"),
        ("Test: COD side effect", "rho_evolution_test_cod_frac_delta.png"),
        ("Assessment: original-scale RMSE", "rho_evolution_assessment_rmse_frac_delta.png"),
        ("Assessment: log-MSE", "rho_evolution_assessment_mse_log_frac_delta.png"),
        ("Assessment: PRD improvement", "rho_evolution_assessment_prd_error_reduction.png"),
        ("Assessment: COD side effect", "rho_evolution_assessment_cod_frac_delta.png"),
    ]
    items = []
    for title, fname in specs:
        path = os.path.join(plot_dir, fname)
        items.append(f"<div class='plotitem'><h4>{html.escape(title)}</h4>{_file_img_embed(path, title)}</div>")
    return "<div class='plotgrid compact'>" + "".join(items) + "</div>"


def theory_empirical_tab(ctx: Dict[str, Any]) -> str:
    return "".join([
        recommendation_card(ctx),
        theory_definitions_card(),
        "<div class='card'><h3>Range decomposition</h3>", theory_band_table(ctx), "</div>",
        "<div class='card'><h3>Merged aggregate orange-band table</h3>", theory_aggregate_table(ctx), "</div>",
        "<div class='card'><h3>Empirical-vs-theory comparison plots</h3>",
        "<p class='muted'>Green curves are empirical metric changes from the rho sweep. Red dashed curves are theory-implied quantities. Blue bands are the theory recommended range; orange bands mark the theory transition/knee region used above.</p>",
        theory_plot_grid(ctx),
        "</div>",
    ])

CSS = """
:root{--bg:#f6f8fa;--card:#fff;--ink:#1f2933;--muted:#647280;--line:#e2e8f0;
--blue:#0d6efd;--pink:#d6336c;--green:#2f9e44;--accent:#0b7285;}
*{box-sizing:border-box}
body{margin:0;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
background:var(--bg);color:var(--ink);line-height:1.5;font-size:15px}
header{background:linear-gradient(120deg,#0b7285,#1098ad);color:#fff;padding:26px 32px}
header h1{margin:0 0 6px;font-size:24px}
header p{margin:0;opacity:.92;font-size:14px;max-width:1000px}
nav{position:sticky;top:0;z-index:20;background:#0b3d4a;display:flex;flex-wrap:wrap;
gap:4px;padding:8px 16px;box-shadow:0 2px 6px rgba(0,0,0,.15)}
nav .grp{display:flex;gap:4px;align-items:center;margin-right:10px}
nav .lbl{color:#8fd3df;font-size:11px;text-transform:uppercase;letter-spacing:.04em;margin-right:4px}
nav button{background:#0e5566;color:#dff3f7;border:1px solid #15788c;border-radius:6px;
padding:5px 10px;font-size:12.5px;cursor:pointer}
nav button:hover{background:#13708a}
nav button.active{background:#fff;color:#0b3d4a;font-weight:700;border-color:#fff}
main{padding:22px 32px;max-width:1280px;margin:0 auto}
section.tab{display:none}
section.tab.active{display:block}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:18px 20px;margin:0 0 20px;box-shadow:0 1px 2px rgba(0,0,0,.04)}
h2{font-size:19px;margin:2px 0 12px}
h3{font-size:15px;margin:18px 0 8px;color:var(--accent)}
img.plot{width:100%;height:auto;border:1px solid var(--line);border-radius:8px;background:#fff}
.plotcell{min-width:560px}
.plot-table{max-height:760px}
.plotgrid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}
.plotgrid.compact{grid-template-columns:repeat(2,minmax(0,1fr))}
.plotitem h4{font-size:13px;margin:0 0 6px;color:var(--accent)}
.missing{border:1px dashed var(--line);border-radius:8px;color:var(--muted);padding:24px;text-align:center}
.plotlinks a{display:inline-block;margin:0 6px 6px 0;font-size:12px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{border:1px solid var(--line);padding:5px 8px;text-align:right}
th{background:#eef3f6;position:sticky;top:0}
td.mname,th:first-child{text-align:left}
.tablewrap{max-height:430px;overflow:auto;border:1px solid var(--line);border-radius:8px}
tr.row-linear td{background:#f3f4f6}
tr.row-lgbm td{background:#e7f0ff}
tr.row-cov:nth-child(odd) td{background:#fff}
	tr.row-cov:nth-child(even) td{background:#fcf5f8}
	td.best{outline:2px solid var(--green);font-weight:700;background:#e9fbef!important}
	td.worst{outline:2px solid #dc2626;font-weight:700;background:#fdeaea!important}
	td.best.worst{outline:2px solid var(--green);box-shadow:inset 0 0 0 2px #dc2626;
	background:linear-gradient(135deg,#e9fbef 0 50%,#fdeaea 50% 100%)!important}
	td.rho-rec{border-left:3px solid #6f42c1;background:#f5f0fb}
	.rec-tick{color:#6f42c1;font-weight:800;margin-right:2px}
	.criterion-mark{display:inline-block;min-width:16px;text-align:center;font-weight:800;color:#0b7285}
	.legend-mark{white-space:nowrap;margin-right:12px}
	.delta{font-size:11px;padding:1px 5px;border-radius:9px;margin-left:6px}
	.delta.up{background:#e6f7ed;color:#1b7a3d}
	.delta.down{background:#fdeaea;color:#b42323}
.delta.flat{background:#edf2f7;color:#4a5568}
	.grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}
	.grid3{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}
	.pill{display:inline-block;background:#e3fafc;color:#0b7285;border:1px solid #99e9f2;
	border-radius:20px;padding:2px 10px;font-size:12px;margin:0 6px 6px 0}
.recobox{border:1px solid var(--line);border-radius:10px;background:#f8fbfc;padding:14px 16px}
.recobox.strong{background:#ecfeff;border-color:#67e8f9}
.recobox h3{margin-top:0}
.big{font-size:24px;font-weight:800;color:#0b3d4a;margin:4px 0 8px}
.muted{color:var(--muted);font-size:13px}
ul.findings{margin:6px 0 0;padding-left:20px}
ul.findings li{margin:4px 0}
.legend span{display:inline-block;margin-right:14px;font-size:12px}
.dot{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:5px;vertical-align:middle}
	@media(max-width:900px){.grid2,.grid3,.plotgrid,.plotgrid.compact{grid-template-columns:1fr}}
"""

JS = """
function showTab(id){
  document.querySelectorAll('section.tab').forEach(s=>s.classList.remove('active'));
  document.querySelectorAll('nav button').forEach(b=>b.classList.remove('active'));
  var el=document.getElementById(id); if(el)el.classList.add('active');
  var bt=document.querySelector('nav button[data-tab="'+id+'"]'); if(bt)bt.classList.add('active');
  window.scrollTo({top:0,behavior:'instant'});
}
document.addEventListener('DOMContentLoaded',function(){
  var f=document.querySelector('nav button'); if(f)showTab(f.getAttribute('data-tab'));
});
"""


def overview_findings(exps: List[Experiment]) -> str:
    n = len(exps)
    cod_improved = prd_improved = 0
    for e in exps:
        lin, lgb, cov = _split_models(e.assess)
        bc = best_cov_row(e, "COD")
        if bc is None or lgb is None:
            continue
        if bc["COD"] < lgb["COD"]:
            cod_improved += 1
        # best PRD-closeness cov vs lgbm
        bp = best_cov_row(e, "PRD")
        if bp is not None and abs(bp["PRD"] - 1) < abs(lgb["PRD"] - 1):
            prd_improved += 1
    items = [
        f"<b>{n}</b> experiments loaded ({len({e.year for e in exps})} assessment years \u00d7 "
        f"{len({e.cfg for e in exps})} LGBM baseline configs).",
        f"The validation-selected covariance penalty reduced assessment <b>COD</b> below the LGBM baseline in "
        f"<b>{cod_improved}/{n}</b> experiments, and moved <b>PRD</b> closer to 1.0 in "
        f"<b>{prd_improved}/{n}</b> \u2014 i.e. it consistently buys vertical-equity / uniformity.",
        f"At the fixed budget of <code>n_estimators=500</code>, the comparison plots should be read "
        f"as an operating-point search: choose rho where R\u00b2 / MdAPE remain stable while COD, PRD, "
        f"PRB, and VEI move toward their targets.",
        "The embedded plot panels below are rebuilt from the 500-estimator sweep metrics. Rho is "
        "displayed on its original numeric scale, not a log axis, so local jumps and plateaus are "
        "visible directly.",
    ]
    return "<ul class='findings'>" + "".join(f"<li>{x}</li>" for x in items) + "</ul>"


def overview_matrix(exps: List[Experiment]) -> str:
    years = sorted({e.year for e in exps})
    cfgs = ["cv_top1_r2", "test_best_r2", "cv_top2_r2"]
    cfgs = [c for c in cfgs if any(e.cfg == c for e in exps)]
    by = {(e.year, e.cfg): e for e in exps}
    head = "<tr><th>assessment year \\ config</th>" + "".join(f"<th>{c}</th>" for c in cfgs) + "</tr>"
    body = []
    for yr in years:
        tds = [f"<td class='mname'>{yr}</td>"]
        for cf in cfgs:
            e = by.get((yr, cf))
            if e is None:
                tds.append("<td>&ndash;</td>")
            else:
                tds.append(f"<td><a href='#' onclick=\"showTab('{e.key}');return false;\">open</a></td>")
        body.append("<tr>" + "".join(tds) + "</tr>")
    return f"<table>{head}{''.join(body)}</table>"


def best_table(
    exps: List[Experiment],
    result_split: str = "assess",
    final_recommended_rho: Optional[float] = None,
) -> str:
    """Per experiment: baselines and selected cov operating points on a result split, with deltas."""
    head = "<tr><th>Experiment</th><th>Model</th><th>rho</th>" + "".join(
        f"<th>{html.escape(d)}</th>" for d, *_ in METRICS
    ) + "</tr>"
    body = []
    for e in sorted(exps, key=_experiment_sort_key):
        result_df = e.test if result_split == "test" else e.assess
        lin, lgb, result_cov = _split_models(result_df)
        rows = [
            ("Linear", lin, "row-linear"),
            ("LGBM", lgb, "row-lgbm"),
        ]
        final_rho = RECOMMENDED_RHO if final_recommended_rho is None else float(final_recommended_rho)
        final_row = _cov_row_at_rho(result_cov, final_rho)
        if final_row is not None:
            rows.append((REC_LABEL, final_row, "row-cov"))
        rows.extend([
            ("Cov min-RMSE", best_cov_row(e, "RMSE", result_split), "row-cov"),
            ("Cov Nash-volume", best_cov_row(e, "Nash", result_split), "row-cov"),
        ])
        for tag, r, cls in rows:
            if r is None:
                continue
            rho_val = r["rho"] if "rho" in r else np.nan
            rho = "&ndash;" if pd.isna(rho_val) else f"{float(rho_val):.3g}"
            cells = []
            for disp, col, ideal, fmt in METRICS:
                cell = _fmt(r[col], fmt, col)
                if tag.startswith("Cov") and lgb is not None:
                    cell += delta_badge(r[col], lgb[col], ideal, col)
                cells.append(f"<td>{cell}</td>")
            body.append(
                f"<tr class='{cls}'><td class='mname'>{e.title}</td><td class='mname'>{_overview_model_label(tag)}</td>"
                f"<td>{rho}</td>{''.join(cells)}</tr>"
            )
    return f"<div class='tablewrap'><table class='metrics'><thead>{head}</thead><tbody>{''.join(body)}</tbody></table></div>"


def cfg_summary(exp: Experiment) -> str:
    lin, lgb, cov = _split_models(exp.assess)
    pills = [
        f"<span class='pill'>source: {html.escape(SRC_LABELS.get(exp.src, exp.src))}</span>",
        f"<span class='pill'>assessment year: {exp.year}</span>",
        f"<span class='pill'>config: {exp.cfg} ({exp.cid})</span>",
        f"<span class='pill'>rho sweep: {cov['rho'].min():.3g} \u2192 {cov['rho'].max():.3g} ({len(cov)} pts)</span>",
    ]
    return "<div>" + "".join(pills) + "</div>"


def build_html(exps: List[Experiment]) -> str:
    global RECOMMENDED_RHO, REC_RHO_RANGE
    exps = sorted(exps, key=_experiment_sort_key)
    # Empirical operating range is derived from the validation sweep: lower edge = equity
    # onset (median equity improvement reaches 25%, accuracy-feasible); upper edge = accuracy
    # window (within 1% of best cov RMSE). Both medians across the 12 runs.
    emp_low, emp_high = compute_empirical_range(exps)
    theory_ctx = load_theory_context(
        empirical_low_override=emp_low, empirical_high_override=emp_high
    )
    # Reconcile the recommended rho / range with the corrected criteria, so every plot,
    # table marker and label below uses the empirical-vs-theory overlap.
    if theory_ctx.get("available") and np.isfinite(theory_ctx.get("both_low", np.nan)) \
            and np.isfinite(theory_ctx.get("both_high", np.nan)):
        REC_RHO_RANGE = (round(float(theory_ctx["both_low"]), 2), round(float(theory_ctx["both_high"]), 2))
        if np.isfinite(theory_ctx.get("both_ref", np.nan)):
            RECOMMENDED_RHO = round(float(theory_ctx["both_ref"]), 2)
    # nav
    nav = [
        "<button data-tab='overview' onclick=\"showTab('overview')\">Overview</button>",
        "<button data-tab='theory_empirical' onclick=\"showTab('theory_empirical')\">Theory × empirical</button>",
    ]
    years = sorted({e.year for e in exps})
    nav.append("<span class='grp'><span class='lbl'>experiments</span></span>")
    for yr in years:
        grp = ["<span class='grp'><span class='lbl'>" + yr + "</span>"]
        for e in [x for x in exps if x.year == yr]:
            grp.append(f"<button data-tab='{e.key}' onclick=\"showTab('{e.key}')\">{e.cfg}</button>")
        grp.append("</span>")
        nav.append("".join(grp))

    sections = []

    # ---- Overview ----
    legend = ("<div class='legend'>"
              f"<span><span class='dot' style='background:{COL_LINEAR}'></span>Linear baseline</span>"
              f"<span><span class='dot' style='background:{COL_LGBM}'></span>LGBM baseline</span>"
              f"<span><span class='dot' style='background:{COL_COV}'></span>Covariance penalty (assessment)</span>"
              f"<span><span class='dot' style='background:{COL_COV_TEST}'></span>Covariance penalty (test)</span>"
              "<span><span class='dot' style='background:#2f9e44'></span>IAAO band / target</span></div>")
    ov = [
        "<div class='card'><h2>Covariance-penalty rho sweep \u2014 results dashboard</h2>",
        "<p class='muted'>Each experiment fits a Linear baseline, an unpenalised LGBM baseline, "
        "and a covariance-penalised LGBM across a rho grid "
        "(<code>rho \u2208 [0.1, 20]</code>, 51-point geometric grid with an extra node at <code>rho = 3.01</code>, "
        "diff mode, <code>n_estimators=500</code>). The recommended operating rho is "
        f"<b>derived below</b> from the empirical\u2229theory overlap (currently <code>rho \u2248 {RECOMMENDED_RHO:g}</code>, "
        f"range {_range_text(REC_RHO_RANGE[0], REC_RHO_RANGE[1], 3)}); every recommended-rho row/marker adapts to it "
        "and is shown at the nearest swept rho. Metrics are reported on validation bootstrap averages, the test "
        "split, and the held-out <b>assessment</b> year.</p>",
        legend, "</div>",
        recommendation_card(theory_ctx),
        "<div class='card'><h3>Metric color key</h3><div class='legend'>",
        "".join(
            f"<span><span class='dot' style='background:{color}'></span>{html.escape(metric)}</span>"
            for metric, color in METRIC_COLORS.items()
        ),
        "</div></div>",
        "<div class='card'><h3>Key findings</h3>", overview_findings(exps), "</div>",
        "<div class='card'><h3>Experiment matrix</h3>", overview_matrix(exps), "</div>",
        "<div class='card'><h3>Best covariance result per experiment (assessment, vs LGBM baseline)</h3>",
        f"<p class='muted'>Rows show assessment metrics at the recommended <b>rho \u2248 {RECOMMENDED_RHO:g}</b> (the "
        f"empirical\u2229theory operating point {_range_text(REC_RHO_RANGE[0], REC_RHO_RANGE[1], 3)}, shown at the "
        "nearest swept rho), then at the rho values selected on the validation bootstrap averages by two "
        "criteria: minimum RMSE, and maximum Pareto-efficient Nash volume across R\u00b2, MdAPE, COD, and PRB. "
        "Deltas compare each penalised model to the unpenalised LGBM baseline for every metric "
        "(green = improvement).</p>",
        f"<p class='muted'>{_overview_mark_legend()}</p>",
        best_table(exps, final_recommended_rho=RECOMMENDED_RHO), "</div>",
        "<div class='card'><h3>Best covariance result per experiment (test, vs LGBM baseline)</h3>",
        f"<p class='muted'>Same recommended rho \u2248 {RECOMMENDED_RHO:g} and validation-selected rho criteria "
        "(min-RMSE, Nash volume) as above, but displaying test-split metrics and test-split deltas against the "
        "unpenalised LGBM baseline.</p>",
        f"<p class='muted'>{_overview_mark_legend()}</p>",
        best_table(exps, "test", final_recommended_rho=RECOMMENDED_RHO), "</div>",
        "<div class='card'><h2>Median ratio by sale-price decile</h2>"
        "<p class='muted'>For each assessment year (rows) and LGBM baseline config (columns), the median "
        "prediction-to-price ratio across the ten sale-price deciles (D1 = cheapest \u2192 D10 = most expensive) "
        "on the held-out assessment split. Each cell overlays the two baselines (Linear, LGBM) and the three "
        f"reported covariance operating points: the recommended rho \u2248 {RECOMMENDED_RHO:g}, the validation "
        "min-RMSE rho, and the validation Nash-volume rho (the two selected rho values are annotated in each "
        "cell). A flat line sitting on the "
        "dotted parity at 1.0, inside the green IAAO 0.95\u20131.05 band, is the equity target; a downward tilt "
        "from D1 to D10 is regressivity (cheap homes over-assessed relative to expensive ones).</p>"
        f"<img class='plot' src='{fig_decile_matrix(exps, 'assess')}'></div>",
    ]
    ov.extend([
        "<div class='card'><h3>All-metric normalized evolution \u2014 validation bootstrap average</h3>",
        "<p class='muted'>Each metric is rescaled to [0, 1] <b>within its own observed range across this "
        "experiment's rho sweep</b> (0 = worst, 1 = best). These curves show where each metric is most/least "
        "favourable <i>relative to the sweep</i> and let differently-scaled metrics share one axis; they are "
        "<b>not</b> absolute quality scores and are not comparable in level across experiments. Use the metric "
        "tables for absolute values.</p>",
        artifact_table_html(
            exps,
            title="Validation Avg Behavior",
            plot_getter=lambda e: fig_normalized_evolution(e, "validation"),
            height=280,
        ),
        "</div>",
        "<div class='card'><h3>All-metric normalized evolution \u2014 test split</h3>",
        "<p class='muted'>Same within-sweep [0, 1] rescaling as above (relative, not absolute, quality).</p>",
        artifact_table_html(
            exps,
            title="Test Behavior",
            plot_getter=lambda e: fig_normalized_evolution(e, "test"),
            height=280,
        ),
        "</div>",
        "<div class='card'><h3>MdAPE tradeoff \u2014 validation bootstrap average</h3>",
        artifact_table_html(
            exps,
            title="PRD / PRB / VEI vs MdAPE (Validation Avg)",
            plot_getter=lambda e: fig_metric_tradeoff(e, "validation", "MdAPE"),
            height=260,
        ),
        "</div>",
        "<div class='card'><h3>MdAPE tradeoff \u2014 test split</h3>",
        artifact_table_html(
            exps,
            title="PRD / PRB / VEI vs MdAPE (Test)",
            plot_getter=lambda e: fig_metric_tradeoff(e, "test", "MdAPE"),
            height=260,
        ),
        "</div>",
    ])
    sections.append(f"<section class='tab' id='overview'>{''.join(ov)}</section>")
    sections.append(f"<section class='tab' id='theory_empirical'>{theory_empirical_tab(theory_ctx)}</section>")

    # ---- per experiment ----
    for e in exps:
        rel = os.path.basename(e.folder.rstrip("/"))
        links = (
            f"<p class='muted'>Underlying artifacts: "
            f"<a href='{rel}/plots/'>detailed per-model plots (PDF)</a> &middot; "
            f"<a href='{rel}/quick_test_metrics_assess.csv'>assessment CSV</a> &middot; "
            f"<a href='{rel}/quick_test_metrics_test.csv'>test CSV</a> &middot; "
            f"<a href='{rel}/quick_test_metrics_validation_bootstrap_avg.csv'>bootstrap CSV</a></p>"
        )
        body = [
            f"<div class='card'><h2>{e.title}</h2>", cfg_summary(e), links, "</div>",
            "<div class='card'><h3>Normalized metric evolution vs rho</h3>",
            "<p class='muted'>Rebuilt from the split metric CSVs. Metric-specific "
            "original-scale plots: ",
            compact_plot_links(e, "validation"), " (validation) &middot; ",
            compact_plot_links(e, "test"), " (test) &middot; ",
            compact_plot_links(e, "assess"), " (assessment).</p>",
            "<div class='plotgrid'>",
            f"<div class='plotitem'><h4>Validation bootstrap average</h4>{_img_embed(fig_normalized_evolution(e, 'validation'), 'validation rho evolution')}</div>",
            f"<div class='plotitem'><h4>Test</h4>{_img_embed(fig_normalized_evolution(e, 'test'), 'test rho evolution')}</div>",
            f"<div class='plotitem'><h4>Assessment</h4>{_img_embed(fig_normalized_evolution(e, 'assess'), 'assessment rho evolution')}</div>",
            "</div></div>",
            "<div class='card'><h3>Tradeoff plots</h3>",
            "<p class='muted'>Rebuilt from the split metric CSVs with IAAO band/target styling.</p>",
            "<div class='plotgrid'>",
            f"<div class='plotitem'><h4>Validation: MdAPE</h4>{_img_embed(fig_metric_tradeoff(e, 'validation', 'MdAPE'), 'validation mdape tradeoff')}</div>",
            f"<div class='plotitem'><h4>Test: MdAPE</h4>{_img_embed(fig_metric_tradeoff(e, 'test', 'MdAPE'), 'test mdape tradeoff')}</div>",
            f"<div class='plotitem'><h4>Assessment: MdAPE</h4>{_img_embed(fig_metric_tradeoff(e, 'assessment', 'MdAPE'), 'assessment mdape tradeoff')}</div>",
            f"<div class='plotitem'><h4>Assessment: R\u00b2</h4>{_img_embed(fig_metric_tradeoff(e, 'assessment', 'R2'), 'assessment r2 tradeoff')}</div>",
            f"<div class='plotitem'><h4>Assessment: COD</h4>{_img_embed(fig_metric_tradeoff(e, 'assessment', 'COD'), 'assessment cod tradeoff')}</div>",
            "</div></div>",
            "<div class='card'><h3>Full metric table \u2014 assessment year</h3>",
            "<p class='muted'>Green outline = best value and red outline = worst value in the column "
            "across Linear, LGBM and all rho rows; ties use the displayed decimals. A violet \u2023 mark and "
            "left rule on the <b>rho</b> column flags rho values inside the recommended range "
            f"[{REC_RHO_RANGE[0]:g}, {REC_RHO_RANGE[1]:g}].</p>",
            metric_table_html(e, "assess"), "</div>",
            "<div class='card'><h3>Full metric table \u2014 test split</h3>",
            "<p class='muted'>Green outline = best value and red outline = worst value in the column "
            "across Linear, LGBM and all rho rows; ties use the displayed decimals. The violet \u2023 mark on the "
            "<b>rho</b> column flags rho inside the recommended range "
            f"[{REC_RHO_RANGE[0]:g}, {REC_RHO_RANGE[1]:g}].</p>",
            metric_table_html(e, "test"), "</div>",
        ]
        sections.append(f"<section class='tab' id='{e.key}'>{''.join(body)}</section>")

    return f"""<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Rho Sweep Dashboard</title><style>{CSS}</style></head>
<body>
<header><h1>Covariance-Penalty Rho Sweep \u2014 Results Dashboard</h1>
<p>Effect of the vertical-equity covariance penalty (rho) on LightGBM mass-appraisal models,
across assessment years and baseline configurations. Compare accuracy vs equity, the evolution
of each metric with rho, and the best operating point per experiment.</p></header>
<nav>{''.join(nav)}</nav>
<main>{''.join(sections)}</main>
<script>{JS}</script></body></html>"""


def main():
    exps = load_experiments()
    if not exps:
        raise SystemExit(f"No experiments found under {ROOT}/")
    print(f"loaded {len(exps)} experiments")
    htmltxt = build_html(exps)
    os.makedirs(ROOT, exist_ok=True)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(htmltxt)
    size = os.path.getsize(OUT_HTML) / 1e6
    print(f"wrote {OUT_HTML} ({size:.1f} MB) covering: "
          + ", ".join(sorted(e.key for e in exps)))


if __name__ == "__main__":
    main()
