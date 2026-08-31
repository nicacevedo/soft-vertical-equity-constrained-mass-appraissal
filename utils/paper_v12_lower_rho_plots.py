"""v2 lower-rho paper figures. Caller must set MPLBACKEND before importing pyplot use."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.transition_paper_asset_plots import (
    DIRECT_COLOR,
    FAMILY_DISPLAY,
    LINEAR_COLOR,
    NATIVE_COLOR,
    PERCENT_PATH_METRICS,
    SURR_COLOR,
    combined_row,
    equal_count_bins,
    load_oos_pred,
    metric_val,
    padded_lim,
)
from utils.transition_paper_assets import (
    IAAO_PRB_RANGE,
    IAAO_PRD_RANGE,
    IAAO_VEI_RANGE,
    decade_ratio_shape_anchors,
    ratio_shape_anchors,
)
from utils.transition_regions import FOLD_IDS, PRIMARY_METRICS, family_frame, is_rho_zero, numerically_equal

SPAN_FACE = "#9CA3AF"
SPAN_ALPHA = 0.15
SPAN_DASH = dict(color="#6B7280", ls=(0, (3.0, 2.2)), lw=0.75, alpha=0.9, zorder=1)
# v4 two-band styling (used only when two_band=True / FINAL_BEND_STATUS=PASS)
PRED_COD_SPAN_FACE = "#94A3B8"
PRED_COD_SPAN_ALPHA = 0.16
PRED_COD_SPAN_DASH = dict(color="#64748B", ls=(0, (3.0, 2.2)), lw=0.85, alpha=0.95, zorder=1)
BEND_SPAN_FACE = "#D6B56A"
BEND_SPAN_ALPHA = 0.16
BEND_SPAN_DASH = dict(color="#B45309", ls=(0, (3.0, 1.4, 0.8, 1.4)), lw=0.85, alpha=0.95, zorder=1)
PRED_COD_SPAN_LABEL = "CV prediction/COD transition span"
BEND_SPAN_LABEL = "Post-hoc CV equity/mechanism bend span"
GRID_COLOR = "#D1D5DB"
NEUTRAL_LINE = dict(color="#111827", ls=":", lw=1.05, zorder=3)
IAAO_MKI_RANGE = (0.95, 1.05)
NEUTRAL_HLINE = {
    "PRD": 1.0,
    "PRB": 0.0,
    "MKI": 1.0,
    "VEI": 0.0,
    "median_ratio": 1.0,
    "mean_ratio": 1.0,
    "weighted_mean_ratio": 1.0,
    "Beta_log": 0.0,
}
NEUTRAL_VLINE = {
    "PRD": 1.0,
    "PRB": 0.0,
    "MKI": 1.0,
    "VEI": 0.0,
    "Beta_log": 0.0,
}

# Okabe–Ito-inspired categorical mapping for the five/six display anchors.
# Historical keys 0/0.1/1/10/100 are unchanged; 0.01 is additive for v4 decade display.
ANCHOR_COLOR = {
    0.0: "#000000",
    0.01: "#CC79A7",
    0.1: "#0072B2",
    1.0: "#E69F00",
    10.0: "#009E73",
    100.0: "#D55E00",
}
ANCHOR_STYLE = {
    0.0: ("-", "o"),
    0.01: ((0, (1, 1.4)), "P"),
    0.1: ("--", "s"),
    1.0: ("-.", "D"),
    10.0: (":", "^"),
    100.0: ((0, (3, 1, 1, 1)), "v"),
}
ANCHOR_TARGETS = (0.0, 0.1, 1.0, 10.0, 100.0)
DECADE_ANCHOR_TARGETS = (0.0, 0.01, 0.1, 1.0, 10.0, 100.0)
DECADE_LEGEND_NOMINAL = {
    0.0: r"$\rho$=0",
    0.01: r"$\rho\approx0.01$",
    0.1: r"$\rho$=0.1",
    1.0: r"$\rho\approx1$",
    10.0: r"$\rho\approx10$",
    100.0: r"$\rho$=100",
}


def rho_plot_x(rho, *, min_positive: float, q: float) -> np.ndarray:
    x = np.asarray(rho, dtype=float)
    x_zero = float(min_positive) / float(q)
    if not (x_zero < float(min_positive)):
        raise RuntimeError(f"x_zero={x_zero} is not strictly left of min_positive={min_positive}")
    return np.where(x <= 0, x_zero, x)


def log_rho_axes(ax, *, min_positive: float, q: float, xmax: float = 130.0) -> None:
    x_zero = float(min_positive) / float(q)
    ax.set_xscale("log")
    ticks = [x_zero, 0.01, 0.1, 1.0, 10.0, 100.0]
    ticks = [t for t in ticks if t >= x_zero / 1.05]
    labels = ["0" if abs(t - x_zero) < 1e-18 * max(1.0, x_zero) or t == x_zero else (f"{t:g}" if t < 1 else f"{int(t) if t in (1, 10, 100) else t:g}") for t in ticks]
    labels = []
    for t in ticks:
        if abs(t - x_zero) <= 1e-18 * max(1.0, abs(x_zero)) or np.isclose(t, x_zero):
            labels.append("0")
        elif abs(t - 0.01) < 1e-12:
            labels.append("0.01")
        elif abs(t - 0.1) < 1e-12:
            labels.append("0.1")
        elif abs(t - 1) < 1e-12:
            labels.append("1")
        elif abs(t - 10) < 1e-12:
            labels.append("10")
        elif abs(t - 100) < 1e-12:
            labels.append("100")
        else:
            labels.append(f"{t:g}")
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlim(x_zero / 1.15, xmax)


def shade_cv_span(ax, low: Optional[float], high: Optional[float]) -> None:
    if low is None or high is None:
        return
    if not (np.isfinite(low) and np.isfinite(high)):
        return
    ax.axvspan(float(low), float(high), color=SPAN_FACE, alpha=SPAN_ALPHA, lw=0, zorder=0)


def shade_cv_span_with_bounds(ax, low: Optional[float], high: Optional[float]) -> None:
    """Frozen CV-derived descriptive transition span: fill plus dashed endpoints."""
    shade_cv_span(ax, low, high)
    if low is None or high is None:
        return
    if not (np.isfinite(low) and np.isfinite(high)):
        return
    ax.axvline(float(low), **SPAN_DASH)
    ax.axvline(float(high), **SPAN_DASH)


def shade_pred_cod_span(ax, low: Optional[float], high: Optional[float]) -> None:
    if low is None or high is None:
        return
    if not (np.isfinite(low) and np.isfinite(high)):
        return
    ax.axvspan(float(low), float(high), color=PRED_COD_SPAN_FACE, alpha=PRED_COD_SPAN_ALPHA, lw=0, zorder=0)
    ax.axvline(float(low), **PRED_COD_SPAN_DASH)
    ax.axvline(float(high), **PRED_COD_SPAN_DASH)


def shade_bend_span(ax, low: Optional[float], high: Optional[float]) -> None:
    if low is None or high is None:
        return
    if not (np.isfinite(low) and np.isfinite(high)):
        return
    ax.axvspan(float(low), float(high), color=BEND_SPAN_FACE, alpha=BEND_SPAN_ALPHA, lw=0, zorder=0)
    ax.axvline(float(low), **BEND_SPAN_DASH)
    ax.axvline(float(high), **BEND_SPAN_DASH)


def family_bend_span(bend_df: Optional[pd.DataFrame], family: str) -> Tuple[Optional[float], Optional[float], bool]:
    if bend_df is None or bend_df.empty:
        return None, None, False
    row = bend_df.loc[bend_df["family"] == family]
    if row.empty:
        return None, None, False
    r = row.iloc[0]
    status = str(r.get("status", r.get("classification", "")))
    if status not in {"VALID", "VALID_POSITIVE_INTERIOR_SPAN", "PASS"}:
        if not bool(r.get("valid", False)):
            return None, None, False
    low = r.get("rho_bend_low", r.get("rho_low"))
    high = r.get("rho_bend_high", r.get("rho_high"))
    if low is None or high is None or not (np.isfinite(float(low)) and np.isfinite(float(high))):
        return None, None, False
    return float(low), float(high), True


def shade_spans_for_path(
    ax,
    pred_low: Optional[float],
    pred_high: Optional[float],
    pred_ok: bool,
    *,
    v3: bool = False,
    two_band: bool = False,
    bend_df: Optional[pd.DataFrame] = None,
    family: Optional[str] = None,
) -> None:
    if two_band and pred_ok:
        shade_pred_cod_span(ax, pred_low, pred_high)
        blow, bhigh, bok = family_bend_span(bend_df, str(family))
        if bok:
            shade_bend_span(ax, blow, bhigh)
        return
    if pred_ok:
        if v3:
            shade_cv_span_with_bounds(ax, pred_low, pred_high)
        else:
            shade_cv_span(ax, pred_low, pred_high)


def apply_major_grid(ax) -> None:
    """Subtle major-only reference grid behind data; no minor-log clutter."""
    from matplotlib.ticker import NullLocator

    ax.set_axisbelow(True)
    ax.grid(True, which="major", axis="both", color=GRID_COLOR, linewidth=0.45, alpha=0.42, zorder=0)
    ax.grid(False, which="minor")
    if str(ax.get_xscale()) == "log":
        ax.xaxis.set_minor_locator(NullLocator())


def nearby_targets(values: Sequence[float], targets: Sequence[Optional[float]], *, rel: float = 0.35) -> Tuple[float, ...]:
    arr = np.asarray([float(v) for v in values if v is not None and np.isfinite(float(v))], dtype=float)
    if arr.size == 0:
        return ()
    lo, hi = float(np.min(arr)), float(np.max(arr))
    span = hi - lo
    if span <= 0:
        span = max(abs(hi), 1e-6)
    pad = rel * span
    keep: List[float] = []
    for t in targets:
        if t is None:
            continue
        tv = float(t)
        if not np.isfinite(tv):
            continue
        if (lo - pad) <= tv <= (hi + pad):
            keep.append(tv)
    return tuple(keep)


def draw_neutral_hline(ax, metric: str) -> None:
    if metric in {"Delta_NL", "dCor_e_y"}:
        return
    if metric not in NEUTRAL_HLINE:
        return
    ax.axhline(float(NEUTRAL_HLINE[metric]), **NEUTRAL_LINE)


def draw_neutral_vline(ax, metric: str) -> None:
    if metric in {"Delta_NL", "dCor_e_y"}:
        return
    if metric not in NEUTRAL_VLINE:
        return
    ax.axvline(float(NEUTRAL_VLINE[metric]), **NEUTRAL_LINE)


def maybe_percent(col: str, vals: np.ndarray) -> np.ndarray:
    out = np.asarray(vals, dtype=float)
    if col in PERCENT_PATH_METRICS:
        return 100.0 * out
    return out


def family_span(span_df: pd.DataFrame, family: str) -> Tuple[Optional[float], Optional[float], bool]:
    row = span_df.loc[span_df["family"] == family]
    if row.empty:
        return None, None, False
    r = row.iloc[0]
    ok = str(r.get("status", "")) == "VALID_POSITIVE_INTERIOR_SPAN"
    if not ok:
        return None, None, False
    return float(r["rho_transition_low"]), float(r["rho_transition_high"]), True


def anchor_key(rho: float, *, decade: bool = False) -> float:
    targets = DECADE_ANCHOR_TARGETS if decade else ANCHOR_TARGETS
    for t in targets:
        if abs(float(rho) - float(t)) < 1e-8 or (t == 0.0 and is_rho_zero(float(rho))):
            return float(t)
        if t > 0 and numerically_equal(float(rho), float(t), atol=1e-8, rtol=1e-8):
            return float(t)
    return float(min(targets, key=lambda t: abs(float(rho) - float(t))))


def _save(plt, fig, stem: Path) -> List[str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    pdf = stem.with_suffix(".pdf")
    png = stem.with_suffix(".png")
    fig.savefig(pdf, format="pdf", bbox_inches="tight")
    fig.savefig(png, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    if pdf.stat().st_size <= 0 or png.stat().st_size <= 0:
        raise RuntimeError(f"empty figure {stem}")
    return [str(pdf), str(png)]


def _oos_path_figure(
    plt,
    combined,
    span_df,
    metrics,
    min_positive,
    q,
    stem,
    *,
    mechanism=False,
    v3: bool = False,
    two_band: bool = False,
    bend_span_df: Optional[pd.DataFrame] = None,
):
    fig, axes = plt.subplots(len(metrics), 2, figsize=(8.8, 2.2 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = np.asarray([axes])
    for r, spec in enumerate(metrics):
        col, ylab, zero_line, force_zero_ylim = spec
        row_vals: List[float] = []
        for fam in FAMILY_DISPLAY:
            sub = combined.loc[combined["family"] == fam]
            for ev in ("heldout", "forward_2025"):
                vals = pd.to_numeric(sub[f"{col}__{ev}"], errors="coerce").to_numpy(dtype=float)
                vals = maybe_percent(col, vals)
                row_vals.extend(vals.tolist())
        include = (0.0,) if force_zero_ylim else ()
        if v3:
            include = include + nearby_targets(row_vals, [NEUTRAL_HLINE.get(col)])
        ylim = padded_lim(row_vals, pad=0.08, include=include)
        for c, fam in enumerate(FAMILY_DISPLAY):
            ax = axes[r, c]
            low, high, ok = family_span(span_df, fam)
            shade_spans_for_path(
                ax, low, high, ok, v3=v3, two_band=two_band, bend_df=bend_span_df, family=fam
            )
            sub = combined.loc[combined["family"] == fam].sort_values("rho")
            color = DIRECT_COLOR if fam == "Direct" else SURR_COLOR
            x = rho_plot_x(sub["rho"].to_numpy(dtype=float), min_positive=min_positive, q=q)
            y_h = maybe_percent(col, pd.to_numeric(sub[f"{col}__heldout"], errors="coerce").to_numpy(dtype=float))
            y_f = maybe_percent(col, pd.to_numeric(sub[f"{col}__forward_2025"], errors="coerce").to_numpy(dtype=float))
            ax.plot(x, y_h, color=color, marker="o", ms=3, lw=1.3, label="Held-out", zorder=4)
            ax.plot(x, y_f, color=color, ls="--", marker="s", ms=3, lw=1.2, label="2025", zorder=4)
            if v3:
                draw_neutral_hline(ax, col)
            elif zero_line:
                ax.axhline(0.0, color="#111827", ls=":", lw=0.8)
            log_rho_axes(ax, min_positive=min_positive, q=q)
            if v3:
                apply_major_grid(ax)
            ax.set_ylim(*ylim)
            if r == 0:
                ax.set_title(fam)
            if c == 0:
                ax.set_ylabel(ylab)
            if r == len(metrics) - 1:
                ax.set_xlabel(r"Penalty strength $\rho$")
            if r == 0 and c == 1:
                if two_band:
                    from matplotlib.lines import Line2D
                    from matplotlib.patches import Patch

                    handles = [
                        Line2D([0], [0], color=color, marker="o", lw=1.3, label="Held-out"),
                        Line2D([0], [0], color=color, marker="s", ls="--", lw=1.2, label="2025"),
                        Patch(facecolor=PRED_COD_SPAN_FACE, alpha=PRED_COD_SPAN_ALPHA, edgecolor="#64748B", linestyle="--", label=PRED_COD_SPAN_LABEL),
                        Patch(facecolor=BEND_SPAN_FACE, alpha=BEND_SPAN_ALPHA, edgecolor="#B45309", linestyle="-.", label=BEND_SPAN_LABEL),
                    ]
                    ax.legend(handles=handles, frameon=False, fontsize=6.2, loc="best")
                else:
                    ax.legend(frameon=False, fontsize=7, loc="best")
    fig.tight_layout()
    return _save(plt, fig, stem)


def plot_predictive(plt, combined, span_df, min_positive, q, stem, *, v3: bool = False, two_band: bool = False, bend_span_df: Optional[pd.DataFrame] = None):
    metrics = (
        ("R2_price", r"$R^2_P$", False, False),
        ("MAE_price", r"MAE$_P$", False, False),
        ("MAPE", r"MAPE$_P$ (\%)", False, False),
        ("RMSE_log", r"RMSE$_{\log P}$", False, False),
    )
    return _oos_path_figure(plt, combined, span_df, metrics, min_positive, q, stem, v3=v3, two_band=two_band, bend_span_df=bend_span_df)


def plot_level_uniformity(plt, combined, span_df, min_positive, q, stem, *, v3: bool = False, two_band: bool = False, bend_span_df: Optional[pd.DataFrame] = None):
    metrics = (
        ("median_ratio", "Median ratio", False, False),
        ("mean_ratio", "Mean ratio", False, False),
        ("weighted_mean_ratio", "Weighted mean ratio", False, False),
        ("COD", "COD", False, False),
        ("COV", "COV (\%)", False, False),
    )
    return _oos_path_figure(plt, combined, span_df, metrics, min_positive, q, stem, v3=v3, two_band=two_band, bend_span_df=bend_span_df)


def plot_vertical_equity(plt, combined, span_df, min_positive, q, stem, *, v3: bool = False, two_band: bool = False, bend_span_df: Optional[pd.DataFrame] = None):
    metrics = (
        ("PRD", "PRD", False, False),
        ("PRB", "PRB", True, False),
        ("MKI", "MKI", False, False),
        ("VEI", "VEI", True, False),
    )
    return _oos_path_figure(plt, combined, span_df, metrics, min_positive, q, stem, v3=v3, two_band=two_band, bend_span_df=bend_span_df)


def plot_mechanism(plt, combined, span_df, min_positive, q, stem, *, v3: bool = False, two_band: bool = False, bend_span_df: Optional[pd.DataFrame] = None):
    metrics = (
        ("Beta_log", r"$\beta_{\log}$", True, True),
        ("Delta_NL", r"$\Delta_{\mathrm{NL}}$", False, False),
        ("dCor_e_y", r"$\mathrm{dCor}(e,y)$", False, False),
    )
    return _oos_path_figure(plt, combined, span_df, metrics, min_positive, q, stem, mechanism=True, v3=v3, two_band=two_band, bend_span_df=bend_span_df)


def plot_cv_group(
    plt,
    combined,
    span_df,
    metrics,
    min_positive,
    q,
    stem,
    qa_mean: List[Dict[str, Any]],
    *,
    v3: bool = False,
    two_band: bool = False,
    bend_span_df: Optional[pd.DataFrame] = None,
):
    fig, axes = plt.subplots(len(metrics), 2, figsize=(8.8, 2.15 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = np.asarray([axes])
    for r, (col, ylab) in enumerate(metrics):
        row_vals: List[float] = []
        for fam in FAMILY_DISPLAY:
            sub = family_frame(combined, fam)
            for k in FOLD_IDS:
                vals = maybe_percent(col, pd.to_numeric(sub[f"{col}__fold_{k}"], errors="coerce").to_numpy(dtype=float)) if v3 else pd.to_numeric(sub[f"{col}__fold_{k}"], errors="coerce").to_numpy(dtype=float)
                row_vals.extend(vals.tolist())
            cvv = pd.to_numeric(sub[f"{col}__CV_mean"], errors="coerce").to_numpy(dtype=float)
            row_vals.extend((maybe_percent(col, cvv) if v3 else cvv).tolist())
        include = nearby_targets(row_vals, [NEUTRAL_HLINE.get(col)]) if v3 else ()
        ylim = padded_lim(row_vals, pad=0.08, include=include)
        for c, fam in enumerate(FAMILY_DISPLAY):
            ax = axes[r, c]
            low, high, ok = family_span(span_df, fam)
            shade_spans_for_path(
                ax, low, high, ok, v3=v3, two_band=two_band, bend_df=bend_span_df, family=fam
            )
            sub = family_frame(combined, fam).sort_values("rho")
            x = rho_plot_x(sub["rho"].to_numpy(dtype=float), min_positive=min_positive, q=q)
            folds = []
            for k in FOLD_IDS:
                yk = pd.to_numeric(sub[f"{col}__fold_{k}"], errors="coerce").to_numpy(dtype=float)
                folds.append(yk)
                y_plot = maybe_percent(col, yk) if v3 else yk
                ax.plot(x, y_plot, color="#9CA3AF", lw=0.85, alpha=0.75, zorder=3)
            mean = np.nanmean(np.vstack(folds), axis=0)
            cv_mean = pd.to_numeric(sub[f"{col}__CV_mean"], errors="coerce").to_numpy(dtype=float)
            if not np.allclose(mean, cv_mean, equal_nan=True, rtol=1e-10, atol=1e-10):
                qa_mean.append({"family": fam, "metric": col, "ok": False})
            else:
                qa_mean.append({"family": fam, "metric": col, "ok": True, "n_folds": 7})
            color = DIRECT_COLOR if fam == "Direct" else SURR_COLOR
            y_mean = maybe_percent(col, cv_mean) if v3 else cv_mean
            ax.plot(x, y_mean, color=color, lw=2.15, label="Equal-weight CV", zorder=4)
            if v3:
                draw_neutral_hline(ax, col)
            log_rho_axes(ax, min_positive=min_positive, q=q)
            if v3:
                apply_major_grid(ax)
            ax.set_ylim(*ylim)
            if r == 0:
                ax.set_title(fam)
            if c == 0:
                ax.set_ylabel(ylab)
            if r == len(metrics) - 1:
                ax.set_xlabel(r"Penalty strength $\rho$")
            if two_band and r == 0 and c == 1:
                from matplotlib.lines import Line2D
                from matplotlib.patches import Patch

                handles = [
                    Line2D([0], [0], color=color, lw=2.15, label="Equal-weight CV"),
                    Line2D([0], [0], color="#9CA3AF", lw=0.85, alpha=0.75, label="Chronological folds"),
                    Patch(facecolor=PRED_COD_SPAN_FACE, alpha=PRED_COD_SPAN_ALPHA, edgecolor="#64748B", linestyle="--", label=PRED_COD_SPAN_LABEL),
                    Patch(facecolor=BEND_SPAN_FACE, alpha=BEND_SPAN_ALPHA, edgecolor="#B45309", linestyle="-.", label=BEND_SPAN_LABEL),
                ]
                ax.legend(handles=handles, frameon=False, fontsize=5.8, loc="best")
    fig.tight_layout()
    return _save(plt, fig, stem)


def _ratio_shape_core(
    plt,
    result_root,
    combined,
    anchors_by_family,
    stem,
    *,
    empty_note: Optional[str] = None,
    v3: bool = False,
    extra_roots: Sequence[Path] = (),
    decade: bool = False,
):
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.6), sharex=True, sharey=True)
    x_all: List[float] = []
    y_all: List[float] = []
    evals = (("heldout", "Held-out"), ("forward_2025", "2025 forward"))
    for r, (ev, evlab) in enumerate(evals):
        for c, fam in enumerate(FAMILY_DISPLAY):
            ax = axes[r, c]
            anchors = list(anchors_by_family.get(fam, []))
            if not anchors:
                ax.text(0.5, 0.5, empty_note or "No common positive CV transition span\nunder the frozen five-metric rule.", ha="center", va="center", transform=ax.transAxes, fontsize=8, color="#374151")
            for rho in anchors:
                pred = load_oos_pred(result_root, combined, fam, float(rho), ev, extra_roots=extra_roots)
                sale = pred["y_true"].to_numpy(dtype=float)
                ratio = pred["y_pred"].to_numpy(dtype=float) / sale
                prof = equal_count_bins(sale, ratio)
                x_all.extend(prof["median_sale_price"].tolist())
                y_all.extend(prof["median_ratio"].tolist())
                key = anchor_key(float(rho), decade=decade)
                ls, mk = ANCHOR_STYLE[key]
                if decade:
                    lab = DECADE_LEGEND_NOMINAL.get(key, rf"$\rho$={float(rho):.3g}")
                else:
                    lab = rf"$\rho$={0 if is_rho_zero(float(rho)) else (100 if abs(float(rho)-100)<1e-8 else f'{float(rho):.3g}')}"
                ax.plot(
                    prof["median_sale_price"],
                    prof["median_ratio"],
                    color=ANCHOR_COLOR[key],
                    lw=1.7,
                    ls=ls,
                    marker=mk,
                    ms=3.2,
                    zorder=4,
                    label=lab,
                )
            ax.axhline(1.0, color="#111827", ls="-", lw=1.15, zorder=2)
            ax.axhline(0.9, color="#9CA3AF", ls=":", lw=0.8, zorder=1)
            ax.axhline(1.1, color="#9CA3AF", ls=":", lw=0.8, zorder=1)
            ax.set_xscale("log", base=10)
            if v3:
                apply_major_grid(ax)
            if r == 0:
                ax.set_title(fam)
            if c == 0:
                ax.set_ylabel(f"{evlab}\nValuation-to-sale ratio")
            if r == 1:
                ax.set_xlabel("Sale price")
            if r == 0 and c == 1 and anchors:
                ax.legend(fontsize=6.5, frameon=False, loc="lower left")
    if x_all:
        xmin, xmax = min(x_all), max(x_all)
        ymin = min(y_all)
        pad = 0.04 * max(2.0 - ymin, 0.2)
        ylo = ymin - pad
        if ylo >= min(y_all):
            ylo = min(y_all) - 0.01
        for ax in axes.ravel():
            ax.set_xlim(xmin / 1.05, xmax * 1.05)
            ax.set_ylim(ylo, 2.0)
    else:
        for ax in axes.ravel():
            ax.set_ylim(0.6, 2.0)
    fig.tight_layout()
    return _save(plt, fig, stem)


def plot_ratio_shape(
    plt,
    result_root,
    combined,
    stem,
    *,
    v3: bool = False,
    decade: bool = False,
    extra_roots: Sequence[Path] = (),
):
    grid = combined.loc[combined["family"] == "Direct", "rho"].to_numpy(dtype=float)
    anchors = decade_ratio_shape_anchors(grid) if decade else ratio_shape_anchors(grid)
    by = {fam: anchors for fam in FAMILY_DISPLAY}
    return _ratio_shape_core(
        plt, result_root, combined, by, stem, v3=v3, extra_roots=extra_roots, decade=decade
    )


def plot_ratio_shape_span_only(
    plt,
    result_root,
    combined,
    span_df,
    stem,
    *,
    v3: bool = False,
    decade: bool = False,
    extra_roots: Sequence[Path] = (),
):
    grid = combined.loc[combined["family"] == "Direct", "rho"].to_numpy(dtype=float)
    full = decade_ratio_shape_anchors(grid) if decade else ratio_shape_anchors(grid)
    by: Dict[str, List[float]] = {}
    for fam in FAMILY_DISPLAY:
        low, high, ok = family_span(span_df, fam)
        if not ok:
            by[fam] = []
            continue
        keep = []
        for rho in full:
            if is_rho_zero(float(rho)):
                continue
            if float(low) - 1e-12 <= float(rho) <= float(high) + 1e-12:
                keep.append(float(rho))
        by[fam] = keep
    return _ratio_shape_core(
        plt,
        result_root,
        combined,
        by,
        stem,
        empty_note="No common positive CV transition span\nunder the frozen five-metric rule.",
        v3=v3,
        extra_roots=extra_roots,
        decade=decade,
    )


def _band(ax, lo, hi, label=None):
    if lo is None:
        return
    ax.axvspan(lo, hi, color="#9CA3AF", alpha=0.16, lw=0, label=label)


def plot_main_tradeoff(plt, combined, stem, *, v3: bool = False, omit_linear: bool = False):
    fig, axes = plt.subplots(2, 4, figsize=(11.6, 5.6))
    cols = (
        ("PRD", "PRD", IAAO_PRD_RANGE, 1.0),
        ("PRB", "PRB", IAAO_PRB_RANGE, 0.0),
        ("MKI", "MKI", IAAO_MKI_RANGE, 1.0),
        ("VEI", r"VEI (\%)", IAAO_VEI_RANGE, 0.0),
    )
    evals = (("heldout", "Held-out"), ("forward_2025", "2025"))
    lin = combined_row(combined, "Linear")
    lgb = combined_row(combined, "LightGBM")
    for r, (ev, evlab) in enumerate(evals):
        for c, (met, xlab, band, _legacy_neutral) in enumerate(cols):
            ax = axes[r, c]
            if band is not None:
                _band(ax, band[0], band[1], "Reference band")
            if v3:
                draw_neutral_vline(ax, met)
            elif met == "MKI":
                ax.axvline(1.0, color="#111827", ls=":", lw=0.9)
            r2s: List[float] = []
            xs: List[float] = []
            for fam, color in (("Direct", DIRECT_COLOR), ("Surrogate", SURR_COLOR)):
                sub = combined.loc[combined["family"] == fam].sort_values("rho")
                x = pd.to_numeric(sub[f"{met}__{ev}"], errors="coerce").to_numpy(dtype=float)
                y = pd.to_numeric(sub[f"R2_price__{ev}"], errors="coerce").to_numpy(dtype=float)
                ax.plot(x, y, color=color, marker="o", ms=2.8, lw=1.2, label=fam, zorder=4)
                if len(x) >= 2:
                    ax.annotate("", xy=(x[-1], y[-1]), xytext=(x[-2], y[-2]), arrowprops=dict(arrowstyle="-|>", color=color, lw=0.9))
                xs.extend(x.tolist())
                r2s.extend(y.tolist())
            points = [(lgb, "s", NATIVE_COLOR, "LightGBM")]
            if not omit_linear:
                points = [(lin, "D", LINEAR_COLOR, "Linear")] + points
            for row, mk, colr, lab in points:
                ax.scatter([metric_val(row, met, ev)], [metric_val(row, "R2_price", ev)], marker=mk, s=32, color=colr, zorder=5, label=lab)
                xs.append(metric_val(row, met, ev))
                r2s.append(metric_val(row, "R2_price", ev))
            x_include = ()
            if v3:
                refs = [NEUTRAL_VLINE.get(met)]
                if band is not None:
                    refs.extend(list(band))
                x_include = nearby_targets(xs, refs)
            ax.set_xlim(*padded_lim(xs, pad=0.08, include=x_include))
            ax.set_ylim(*padded_lim(r2s, pad=0.06))
            if v3:
                apply_major_grid(ax)
            if r == 1:
                ax.set_xlabel(xlab)
            if c == 0:
                ax.set_ylabel(rf"{evlab}: $R^2_P$")
            if r == 0:
                ax.set_title(met)
            if r == 0 and c == 3:
                ax.legend(frameon=False, fontsize=6.2, loc="best")
    fig.tight_layout()
    return _save(plt, fig, stem)


def plot_prb_mki(plt, combined, stem, *, v3: bool = False):
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 5.6))
    cols = (
        ("PRB", "PRB", IAAO_PRB_RANGE, 0.0),
        ("MKI", "MKI", IAAO_MKI_RANGE, 1.0),
    )
    evals = (("heldout", "Held-out"), ("forward_2025", "2025"))
    lin = combined_row(combined, "Linear")
    lgb = combined_row(combined, "LightGBM")
    for r, (ev, evlab) in enumerate(evals):
        for c, (met, xlab, band, _legacy_neutral) in enumerate(cols):
            ax = axes[r, c]
            if band is not None:
                _band(ax, band[0], band[1], "Reference band")
            if v3:
                draw_neutral_vline(ax, met)
            elif met == "MKI":
                ax.axvline(1.0, color="#111827", ls=":", lw=0.9)
            xs: List[float] = []
            ys: List[float] = []
            for fam, color in (("Direct", DIRECT_COLOR), ("Surrogate", SURR_COLOR)):
                sub = combined.loc[combined["family"] == fam].sort_values("rho")
                x = pd.to_numeric(sub[f"{met}__{ev}"], errors="coerce").to_numpy(dtype=float)
                y = pd.to_numeric(sub[f"R2_price__{ev}"], errors="coerce").to_numpy(dtype=float)
                ax.plot(x, y, color=color, marker="o", ms=2.8, lw=1.2, label=fam, zorder=4)
                if len(x) >= 2:
                    ax.annotate("", xy=(x[-1], y[-1]), xytext=(x[-2], y[-2]), arrowprops=dict(arrowstyle="-|>", color=color, lw=0.9))
                xs.extend(x.tolist())
                ys.extend(y.tolist())
            for row, mk, colr, lab in ((lin, "D", LINEAR_COLOR, "Linear"), (lgb, "s", NATIVE_COLOR, "LightGBM")):
                ax.scatter([metric_val(row, met, ev)], [metric_val(row, "R2_price", ev)], marker=mk, s=32, color=colr, zorder=5, label=lab)
                xs.append(metric_val(row, met, ev))
                ys.append(metric_val(row, "R2_price", ev))
            x_include = ()
            if v3:
                refs = [NEUTRAL_VLINE.get(met)]
                if band is not None:
                    refs.extend(list(band))
                x_include = nearby_targets(xs, refs)
            ax.set_xlim(*padded_lim(xs, pad=0.08, include=x_include))
            ax.set_ylim(*padded_lim(ys, pad=0.06))
            if v3:
                apply_major_grid(ax)
            if r == 1:
                ax.set_xlabel(xlab)
            if c == 0:
                ax.set_ylabel(rf"{evlab}: $R^2_P$")
            if r == 0:
                ax.set_title(met)
            if r == 0 and c == 1:
                ax.legend(frameon=False, fontsize=6.2, loc="best")
    fig.tight_layout()
    return _save(plt, fig, stem)


def plot_tradeoff_atlas(plt, combined, xmetrics, stem, *, ymetrics=None, zero_x=None, no_zero=None, v3: bool = False):
    ymetrics = ymetrics or (("R2_price", r"$R^2_P$"), ("MAE_price", r"MAE$_P$"), ("MAPE", r"MAPE$_P$"), ("RMSE_log", r"RMSE$_{\log P}$"))
    zero_x = set(zero_x or [])
    no_zero = set(no_zero or [])
    fig, axes = plt.subplots(len(ymetrics), len(xmetrics), figsize=(3.1 * len(xmetrics), 2.35 * len(ymetrics)))
    if len(xmetrics) == 1:
        axes = np.expand_dims(axes, 1)
    lin = combined_row(combined, "Linear")
    lgb = combined_row(combined, "LightGBM")
    ev = "heldout" if "heldout" in str(stem) else "forward_2025"
    if "2025" in stem.name:
        ev = "forward_2025"
    if "heldout" in stem.name:
        ev = "heldout"
    for r, (ymet, ylab) in enumerate(ymetrics):
        for c, (xmet, xlab, band) in enumerate(xmetrics):
            ax = axes[r, c]
            if band is not None:
                _band(ax, band[0], band[1])
            if v3:
                draw_neutral_vline(ax, xmet)
            elif xmet in zero_x:
                ax.axvline(0.0, color="#111827", ls=":", lw=0.8)
            xs: List[float] = []
            ys: List[float] = []
            for fam, color in (("Direct", DIRECT_COLOR), ("Surrogate", SURR_COLOR)):
                sub = combined.loc[combined["family"] == fam].sort_values("rho")
                x = pd.to_numeric(sub[f"{xmet}__{ev}"], errors="coerce").to_numpy(dtype=float)
                y = pd.to_numeric(sub[f"{ymet}__{ev}"], errors="coerce").to_numpy(dtype=float)
                ax.plot(x, y, color=color, marker="o", ms=2.4, lw=1.05, label=fam, zorder=4)
                xs.extend(x.tolist())
                ys.extend(y.tolist())
            for row, mk, colr in ((lin, "D", LINEAR_COLOR), (lgb, "s", NATIVE_COLOR)):
                ax.scatter([metric_val(row, xmet, ev)], [metric_val(row, ymet, ev)], marker=mk, s=22, color=colr, zorder=5)
                xs.append(metric_val(row, xmet, ev))
                ys.append(metric_val(row, ymet, ev))
            include_x = (0.0,) if xmet in zero_x and xmet not in no_zero else ()
            if v3:
                refs = [NEUTRAL_VLINE.get(xmet)]
                if band is not None:
                    refs.extend(list(band))
                include_x = nearby_targets(xs, refs)
            ax.set_xlim(*padded_lim(xs, pad=0.08, include=include_x))
            ax.set_ylim(*padded_lim(ys, pad=0.08))
            if v3:
                apply_major_grid(ax)
            if r == len(ymetrics) - 1:
                ax.set_xlabel(xlab)
            if c == 0:
                ax.set_ylabel(ylab)
            if r == 0:
                ax.set_title(xlab)
            if r == 0 and c == len(xmetrics) - 1:
                ax.legend(frameon=False, fontsize=6, loc="best")
    fig.tight_layout()
    return _save(plt, fig, stem)


def plot_event_locations(plt, combined, tables, min_positive, q, stem, *, v3: bool = False):
    from matplotlib.lines import Line2D

    events_cv = tables["transition_events_cv_mean.csv"]
    events_fold = tables["transition_events_by_fold.csv"]
    conc = tables["transition_temporal_concordance.csv"]
    span_df = tables["transition_span_summary.csv"]
    metrics = [m for m, _d in PRIMARY_METRICS]
    ymap = {m: i for i, m in enumerate(reversed(metrics))}
    labels = {
        "R2_price": r"$R^2$ max",
        "MAE_price": "MAE min",
        "MAPE": "MAPE min",
        "RMSE_log": r"RMSE$_{\log}$ min",
        "COD": "COD min",
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.6), sharey=True)
    for ax, fam in zip(axes, FAMILY_DISPLAY):
        color = DIRECT_COLOR if fam == "Direct" else SURR_COLOR
        low, high, ok = family_span(span_df, fam)
        if ok:
            if v3:
                shade_cv_span_with_bounds(ax, low, high)
            else:
                shade_cv_span(ax, low, high)
        for metric in metrics:
            y = ymap[metric]
            folds = events_fold.loc[(events_fold["family"] == fam) & (events_fold["metric"] == metric)]
            if not folds.empty and "rho_low" in folds.columns:
                xf = rho_plot_x(pd.to_numeric(folds["rho_low"], errors="coerce").to_numpy(dtype=float), min_positive=min_positive, q=q)
                ax.scatter(xf, np.full_like(xf, y, dtype=float), s=12, color=color, alpha=0.45, zorder=4)
            cv = events_cv.loc[(events_cv["family"] == fam) & (events_cv["metric"] == metric)]
            if not cv.empty and pd.notna(cv.iloc[0]["rho_low"]):
                ax.scatter(rho_plot_x([float(cv.iloc[0]["rho_low"])], min_positive=min_positive, q=q)[0], y, s=70, marker="o", color=color, zorder=5, edgecolors="white", linewidths=0.6)
            for split, mk in (("heldout", "s"), ("forward_2025", "^")):
                part = conc.loc[(conc["family"] == fam) & (conc["split"] == split) & (conc["metric"] == metric)]
                if part.empty or pd.isna(part.iloc[0]["rho_low"]):
                    continue
                ax.scatter(
                    rho_plot_x([float(part.iloc[0]["rho_low"])], min_positive=min_positive, q=q)[0],
                    y,
                    s=42,
                    marker=mk,
                    facecolors="white",
                    edgecolors="#111827",
                    linewidths=1.0,
                    zorder=6,
                )
        log_rho_axes(ax, min_positive=min_positive, q=q)
        if v3:
            apply_major_grid(ax)
        ax.set_ylim(-0.9, 4.6)
        ax.set_yticks(list(ymap.values()))
        ax.set_yticklabels([labels[m] for m in reversed(metrics)])
        ax.set_title(fam)
        ax.set_xlabel(r"Penalty strength $\rho$")
        if ok:
            ax.text(0.02, 0.97, "Gray: CV-derived descriptive transition span", transform=ax.transAxes, fontsize=6.2, va="top", color="#374151")
        else:
            ax.text(0.02, 0.97, "No common five-metric positive CV span", transform=ax.transAxes, fontsize=6.2, va="top", color="#374151")
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1D4ED8", ms=8, label="Full-CV event"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1D4ED8", ms=4, alpha=0.5, label="Fold events"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="white", markeredgecolor="#111827", ms=6, label="Held-out"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="white", markeredgecolor="#111827", ms=6, label="2025"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _save(plt, fig, stem)


def plot_descriptive_event_locations(
    plt,
    events: pd.DataFrame,
    span_df: pd.DataFrame,
    min_positive: float,
    q: float,
    stem: Path,
    *,
    labels: Dict[str, str],
    metric_order: Sequence[str],
    note: str,
    two_band: bool = False,
    bend_span_df: Optional[pd.DataFrame] = None,
    extra_legend: Optional[Sequence[Any]] = None,
    span_note: Optional[str] = None,
):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    ymap = {m: i for i, m in enumerate(reversed(list(metric_order)))}
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.8), sharey=True)
    for ax, fam in zip(axes, FAMILY_DISPLAY):
        color = DIRECT_COLOR if fam == "Direct" else SURR_COLOR
        low, high, ok = family_span(span_df, fam)
        shade_spans_for_path(
            ax, low, high, ok, v3=True, two_band=two_band, bend_df=bend_span_df, family=fam
        )
        part = events.loc[events["family"] == fam]
        for metric in metric_order:
            y = ymap[metric]
            folds = part.loc[(part["metric"] == metric) & (part["split"].astype(str).str.startswith("fold_"))]
            if not folds.empty:
                xf = rho_plot_x(pd.to_numeric(folds["event_rho"], errors="coerce").to_numpy(dtype=float), min_positive=min_positive, q=q)
                ax.scatter(xf, np.full_like(xf, y, dtype=float), s=12, color=color, alpha=0.45, zorder=4)
            cv = part.loc[(part["metric"] == metric) & (part["split"] == "cv_mean")]
            if not cv.empty and pd.notna(cv.iloc[0]["event_rho"]):
                ax.scatter(
                    rho_plot_x([float(cv.iloc[0]["event_rho"])], min_positive=min_positive, q=q)[0],
                    y,
                    s=70,
                    marker="o",
                    color=color,
                    zorder=5,
                    edgecolors="white",
                    linewidths=0.6,
                )
            for split, mk in (("heldout", "s"), ("forward_2025", "^")):
                row = part.loc[(part["metric"] == metric) & (part["split"] == split)]
                if row.empty or pd.isna(row.iloc[0]["event_rho"]):
                    continue
                ax.scatter(
                    rho_plot_x([float(row.iloc[0]["event_rho"])], min_positive=min_positive, q=q)[0],
                    y,
                    s=42,
                    marker=mk,
                    facecolors="white",
                    edgecolors="#111827",
                    linewidths=1.0,
                    zorder=6,
                )
        log_rho_axes(ax, min_positive=min_positive, q=q)
        apply_major_grid(ax)
        ax.set_ylim(-0.9, float(len(metric_order)) - 0.4)
        ax.set_yticks(list(ymap.values()))
        ax.set_yticklabels([labels[m] for m in reversed(list(metric_order))])
        ax.set_title(fam)
        ax.set_xlabel(r"Penalty strength $\rho$")
        if two_band:
            ax.text(0.02, 0.97, "Cool dashed: prediction/COD span; warm dash-dot: post-hoc bend span", transform=ax.transAxes, fontsize=5.8, va="top", color="#374151")
        elif ok:
            ax.text(0.02, 0.97, span_note or "Gray: CV-derived descriptive transition span", transform=ax.transAxes, fontsize=6.2, va="top", color="#374151")
        ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=5.8, va="bottom", color="#374151")
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1D4ED8", ms=8, label="Full-CV event"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1D4ED8", ms=4, alpha=0.5, label="Fold events"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="white", markeredgecolor="#111827", ms=6, label="Held-out"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="white", markeredgecolor="#111827", ms=6, label="2025"),
    ]
    if two_band:
        handles.extend(
            [
                Patch(facecolor=PRED_COD_SPAN_FACE, alpha=PRED_COD_SPAN_ALPHA, edgecolor="#64748B", linestyle="--", label=PRED_COD_SPAN_LABEL),
                Patch(facecolor=BEND_SPAN_FACE, alpha=BEND_SPAN_ALPHA, edgecolor="#B45309", linestyle="-.", label=BEND_SPAN_LABEL),
            ]
        )
    if extra_legend:
        handles.extend(list(extra_legend))
    ncol = 4 if not two_band else 3
    fig.legend(handles=handles, loc="upper center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.90 if two_band else 0.92))
    return _save(plt, fig, stem)


def plot_regret(plt, regret: pd.DataFrame, stem: Path):
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), sharey=False)
    for ax, split, title in zip(axes, ("heldout", "forward_2025"), ("Held-out", "2025")):
        sub = regret.loc[regret["split"] == split]
        ax.bar(sub["metric"], pd.to_numeric(sub["raw_regret"], errors="coerce"), color=DIRECT_COLOR, alpha=0.85)
        ax.set_title(title)
        ax.set_ylabel("Raw span regret")
        ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    return _save(plt, fig, stem)
