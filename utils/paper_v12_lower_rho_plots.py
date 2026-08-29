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
from utils.transition_paper_assets import IAAO_PRB_RANGE, IAAO_PRD_RANGE, IAAO_VEI_RANGE, ratio_shape_anchors
from utils.transition_regions import FOLD_IDS, PRIMARY_METRICS, family_frame, is_rho_zero, numerically_equal

SPAN_FACE = "#9CA3AF"
SPAN_ALPHA = 0.15
IAAO_MKI_RANGE = (0.95, 1.05)

# Okabe–Ito-inspired categorical mapping for the five display anchors.
ANCHOR_COLOR = {
    0.0: "#000000",
    0.1: "#0072B2",
    1.0: "#E69F00",
    10.0: "#009E73",
    100.0: "#D55E00",
}
ANCHOR_STYLE = {
    0.0: ("-", "o"),
    0.1: ("--", "s"),
    1.0: ("-.", "D"),
    10.0: (":", "^"),
    100.0: ((0, (3, 1, 1, 1)), "v"),
}
ANCHOR_TARGETS = (0.0, 0.1, 1.0, 10.0, 100.0)


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


def family_span(span_df: pd.DataFrame, family: str) -> Tuple[Optional[float], Optional[float], bool]:
    row = span_df.loc[span_df["family"] == family]
    if row.empty:
        return None, None, False
    r = row.iloc[0]
    ok = str(r.get("status", "")) == "VALID_POSITIVE_INTERIOR_SPAN"
    if not ok:
        return None, None, False
    return float(r["rho_transition_low"]), float(r["rho_transition_high"]), True


def anchor_key(rho: float) -> float:
    for t in ANCHOR_TARGETS:
        if abs(float(rho) - float(t)) < 1e-8 or (t == 0.0 and is_rho_zero(float(rho))):
            return float(t)
        if t > 0 and numerically_equal(float(rho), float(t), atol=1e-8, rtol=1e-8):
            return float(t)
    # nearest display target
    return float(min(ANCHOR_TARGETS, key=lambda t: abs(float(rho) - float(t))))


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


def _oos_path_figure(plt, combined, span_df, metrics, min_positive, q, stem, *, mechanism=False):
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
                if col in PERCENT_PATH_METRICS:
                    vals = 100.0 * vals
                row_vals.extend(vals.tolist())
        include = (0.0,) if force_zero_ylim else ()
        ylim = padded_lim(row_vals, pad=0.08, include=include)
        for c, fam in enumerate(FAMILY_DISPLAY):
            ax = axes[r, c]
            low, high, ok = family_span(span_df, fam)
            if ok:
                shade_cv_span(ax, low, high)
            sub = combined.loc[combined["family"] == fam].sort_values("rho")
            color = DIRECT_COLOR if fam == "Direct" else SURR_COLOR
            x = rho_plot_x(sub["rho"].to_numpy(dtype=float), min_positive=min_positive, q=q)
            y_h = pd.to_numeric(sub[f"{col}__heldout"], errors="coerce").to_numpy(dtype=float)
            y_f = pd.to_numeric(sub[f"{col}__forward_2025"], errors="coerce").to_numpy(dtype=float)
            if col in PERCENT_PATH_METRICS:
                y_h = 100.0 * y_h
                y_f = 100.0 * y_f
            ax.plot(x, y_h, color=color, marker="o", ms=3, lw=1.3, label="Held-out")
            ax.plot(x, y_f, color=color, ls="--", marker="s", ms=3, lw=1.2, label="2025")
            if zero_line:
                ax.axhline(0.0, color="#111827", ls=":", lw=0.8)
            log_rho_axes(ax, min_positive=min_positive, q=q)
            ax.set_ylim(*ylim)
            if r == 0:
                ax.set_title(fam)
            if c == 0:
                ax.set_ylabel(ylab)
            if r == len(metrics) - 1:
                ax.set_xlabel(r"Penalty strength $\rho$")
            if r == 0 and c == 1:
                ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return _save(plt, fig, stem)


def plot_predictive(plt, combined, span_df, min_positive, q, stem):
    metrics = (
        ("R2_price", r"$R^2_P$", False, False),
        ("MAE_price", r"MAE$_P$", False, False),
        ("MAPE", r"MAPE$_P$ (\%)", False, False),
        ("RMSE_log", r"RMSE$_{\log P}$", False, False),
    )
    return _oos_path_figure(plt, combined, span_df, metrics, min_positive, q, stem)


def plot_level_uniformity(plt, combined, span_df, min_positive, q, stem):
    metrics = (
        ("median_ratio", "Median ratio", False, False),
        ("mean_ratio", "Mean ratio", False, False),
        ("weighted_mean_ratio", "Weighted mean ratio", False, False),
        ("COD", "COD", False, False),
        ("COV", "COV (\%)", False, False),
    )
    return _oos_path_figure(plt, combined, span_df, metrics, min_positive, q, stem)


def plot_vertical_equity(plt, combined, span_df, min_positive, q, stem):
    metrics = (
        ("PRD", "PRD", False, False),
        ("PRB", "PRB", True, False),
        ("MKI", "MKI", False, False),
        ("VEI", "VEI", True, False),
    )
    return _oos_path_figure(plt, combined, span_df, metrics, min_positive, q, stem)


def plot_mechanism(plt, combined, span_df, min_positive, q, stem):
    metrics = (
        ("Beta_log", r"$\beta_{\log}$", True, True),
        ("Delta_NL", r"$\Delta_{\mathrm{NL}}$", False, False),
        ("dCor_e_y", r"$\mathrm{dCor}(e,y)$", False, False),
    )
    return _oos_path_figure(plt, combined, span_df, metrics, min_positive, q, stem, mechanism=True)


def plot_cv_group(plt, combined, span_df, metrics, min_positive, q, stem, qa_mean: List[Dict[str, Any]]):
    fig, axes = plt.subplots(len(metrics), 2, figsize=(8.8, 2.15 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = np.asarray([axes])
    for r, (col, ylab) in enumerate(metrics):
        row_vals: List[float] = []
        for fam in FAMILY_DISPLAY:
            sub = family_frame(combined, fam)
            for k in FOLD_IDS:
                vals = pd.to_numeric(sub[f"{col}__fold_{k}"], errors="coerce").to_numpy(dtype=float)
                row_vals.extend(vals.tolist())
            row_vals.extend(pd.to_numeric(sub[f"{col}__CV_mean"], errors="coerce").tolist())
        ylim = padded_lim(row_vals, pad=0.08)
        for c, fam in enumerate(FAMILY_DISPLAY):
            ax = axes[r, c]
            low, high, ok = family_span(span_df, fam)
            if ok:
                shade_cv_span(ax, low, high)
            sub = family_frame(combined, fam).sort_values("rho")
            x = rho_plot_x(sub["rho"].to_numpy(dtype=float), min_positive=min_positive, q=q)
            folds = []
            for k in FOLD_IDS:
                yk = pd.to_numeric(sub[f"{col}__fold_{k}"], errors="coerce").to_numpy(dtype=float)
                ax.plot(x, yk, color="#9CA3AF", lw=0.85, alpha=0.75)
                folds.append(yk)
            mean = np.nanmean(np.vstack(folds), axis=0)
            cv_mean = pd.to_numeric(sub[f"{col}__CV_mean"], errors="coerce").to_numpy(dtype=float)
            if not np.allclose(mean, cv_mean, equal_nan=True, rtol=1e-10, atol=1e-10):
                qa_mean.append({"family": fam, "metric": col, "ok": False})
            else:
                qa_mean.append({"family": fam, "metric": col, "ok": True, "n_folds": 7})
            color = DIRECT_COLOR if fam == "Direct" else SURR_COLOR
            ax.plot(x, cv_mean, color=color, lw=2.15, label="Equal-weight CV")
            log_rho_axes(ax, min_positive=min_positive, q=q)
            ax.set_ylim(*ylim)
            if r == 0:
                ax.set_title(fam)
            if c == 0:
                ax.set_ylabel(ylab)
            if r == len(metrics) - 1:
                ax.set_xlabel(r"Penalty strength $\rho$")
    fig.tight_layout()
    return _save(plt, fig, stem)


def _ratio_shape_core(plt, result_root, combined, anchors_by_family, stem, *, empty_note: Optional[str] = None):
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
                pred = load_oos_pred(result_root, combined, fam, float(rho), ev)
                sale = pred["y_true"].to_numpy(dtype=float)
                ratio = pred["y_pred"].to_numpy(dtype=float) / sale
                prof = equal_count_bins(sale, ratio)
                x_all.extend(prof["median_sale_price"].tolist())
                y_all.extend(prof["median_ratio"].tolist())
                key = anchor_key(float(rho))
                ls, mk = ANCHOR_STYLE[key]
                ax.plot(
                    prof["median_sale_price"],
                    prof["median_ratio"],
                    color=ANCHOR_COLOR[key],
                    lw=1.7,
                    ls=ls,
                    marker=mk,
                    ms=3.2,
                    label=rf"$\rho$={0 if is_rho_zero(float(rho)) else (100 if abs(float(rho)-100)<1e-8 else f'{float(rho):.3g}')}",
                )
            ax.axhline(1.0, color="#111827", ls="-", lw=1.15, zorder=2)
            ax.axhline(0.9, color="#9CA3AF", ls=":", lw=0.8, zorder=1)
            ax.axhline(1.1, color="#9CA3AF", ls=":", lw=0.8, zorder=1)
            ax.set_xscale("log", base=10)
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


def plot_ratio_shape(plt, result_root, combined, stem):
    grid = combined.loc[combined["family"] == "Direct", "rho"].to_numpy(dtype=float)
    anchors = ratio_shape_anchors(grid)
    by = {fam: anchors for fam in FAMILY_DISPLAY}
    return _ratio_shape_core(plt, result_root, combined, by, stem)


def plot_ratio_shape_span_only(plt, result_root, combined, span_df, stem):
    grid = combined.loc[combined["family"] == "Direct", "rho"].to_numpy(dtype=float)
    full = ratio_shape_anchors(grid)
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
    )


def _band(ax, lo, hi, label=None):
    if lo is None:
        return
    ax.axvspan(lo, hi, color="#9CA3AF", alpha=0.16, lw=0, label=label)


def plot_main_tradeoff(plt, combined, stem):
    fig, axes = plt.subplots(2, 4, figsize=(11.6, 5.6))
    cols = (
        ("PRD", "PRD", IAAO_PRD_RANGE, None),
        ("PRB", "PRB", IAAO_PRB_RANGE, None),
        ("MKI", "MKI", IAAO_MKI_RANGE, 1.0),
        ("VEI", r"VEI (\%)", IAAO_VEI_RANGE, None),
    )
    evals = (("heldout", "Held-out"), ("forward_2025", "2025"))
    lin = combined_row(combined, "Linear")
    lgb = combined_row(combined, "LightGBM")
    for r, (ev, evlab) in enumerate(evals):
        for c, (met, xlab, band, neutral) in enumerate(cols):
            ax = axes[r, c]
            if band is not None:
                _band(ax, band[0], band[1], "Reference band")
            if neutral is not None:
                ax.axvline(neutral, color="#111827", ls=":", lw=0.9)
            r2s: List[float] = []
            xs: List[float] = []
            for fam, color in (("Direct", DIRECT_COLOR), ("Surrogate", SURR_COLOR)):
                sub = combined.loc[combined["family"] == fam].sort_values("rho")
                x = pd.to_numeric(sub[f"{met}__{ev}"], errors="coerce").to_numpy(dtype=float)
                y = pd.to_numeric(sub[f"R2_price__{ev}"], errors="coerce").to_numpy(dtype=float)
                ax.plot(x, y, color=color, marker="o", ms=2.8, lw=1.2, label=fam)
                if len(x) >= 2:
                    ax.annotate("", xy=(x[-1], y[-1]), xytext=(x[-2], y[-2]), arrowprops=dict(arrowstyle="-|>", color=color, lw=0.9))
                xs.extend(x.tolist())
                r2s.extend(y.tolist())
            for row, mk, colr, lab in ((lin, "D", LINEAR_COLOR, "Linear"), (lgb, "s", NATIVE_COLOR, "LightGBM")):
                ax.scatter([metric_val(row, met, ev)], [metric_val(row, "R2_price", ev)], marker=mk, s=32, color=colr, zorder=5, label=lab)
                xs.append(metric_val(row, met, ev))
                r2s.append(metric_val(row, "R2_price", ev))
            ax.set_xlim(*padded_lim(xs, pad=0.08))
            ax.set_ylim(*padded_lim(r2s, pad=0.06))
            if r == 1:
                ax.set_xlabel(xlab)
            if c == 0:
                ax.set_ylabel(rf"{evlab}: $R^2_P$")
            if r == 0:
                ax.set_title(met)
            if r == 0 and c == 3:
                ax.legend(frameon=False, fontsize=6.2)
    fig.tight_layout()
    return _save(plt, fig, stem)


def plot_prb_mki(plt, combined, stem):
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 5.6))
    cols = (
        ("PRB", "PRB", IAAO_PRB_RANGE, None),
        ("MKI", "MKI", IAAO_MKI_RANGE, 1.0),
    )
    evals = (("heldout", "Held-out"), ("forward_2025", "2025"))
    lin = combined_row(combined, "Linear")
    lgb = combined_row(combined, "LightGBM")
    for r, (ev, evlab) in enumerate(evals):
        for c, (met, xlab, band, neutral) in enumerate(cols):
            ax = axes[r, c]
            if band is not None:
                _band(ax, band[0], band[1], "Reference band")
            if neutral is not None:
                ax.axvline(neutral, color="#111827", ls=":", lw=0.9)
            xs: List[float] = []
            ys: List[float] = []
            for fam, color in (("Direct", DIRECT_COLOR), ("Surrogate", SURR_COLOR)):
                sub = combined.loc[combined["family"] == fam].sort_values("rho")
                x = pd.to_numeric(sub[f"{met}__{ev}"], errors="coerce").to_numpy(dtype=float)
                y = pd.to_numeric(sub[f"R2_price__{ev}"], errors="coerce").to_numpy(dtype=float)
                ax.plot(x, y, color=color, marker="o", ms=2.8, lw=1.2, label=fam)
                if len(x) >= 2:
                    ax.annotate("", xy=(x[-1], y[-1]), xytext=(x[-2], y[-2]), arrowprops=dict(arrowstyle="-|>", color=color, lw=0.9))
                xs.extend(x.tolist())
                ys.extend(y.tolist())
            for row, mk, colr, lab in ((lin, "D", LINEAR_COLOR, "Linear"), (lgb, "s", NATIVE_COLOR, "LightGBM")):
                ax.scatter([metric_val(row, met, ev)], [metric_val(row, "R2_price", ev)], marker=mk, s=32, color=colr, zorder=5, label=lab)
                xs.append(metric_val(row, met, ev))
                ys.append(metric_val(row, "R2_price", ev))
            ax.set_xlim(*padded_lim(xs, pad=0.08))
            ax.set_ylim(*padded_lim(ys, pad=0.06))
            if r == 1:
                ax.set_xlabel(xlab)
            if c == 0:
                ax.set_ylabel(rf"{evlab}: $R^2_P$")
            if r == 0:
                ax.set_title(met)
            if r == 0 and c == 1:
                ax.legend(frameon=False, fontsize=6.2)
    fig.tight_layout()
    return _save(plt, fig, stem)


def plot_tradeoff_atlas(plt, combined, xmetrics, stem, *, ymetrics=None, zero_x=None, no_zero=None):
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
            if xmet in zero_x:
                ax.axvline(0.0, color="#111827", ls=":", lw=0.8)
            xs: List[float] = []
            ys: List[float] = []
            for fam, color in (("Direct", DIRECT_COLOR), ("Surrogate", SURR_COLOR)):
                sub = combined.loc[combined["family"] == fam].sort_values("rho")
                x = pd.to_numeric(sub[f"{xmet}__{ev}"], errors="coerce").to_numpy(dtype=float)
                y = pd.to_numeric(sub[f"{ymet}__{ev}"], errors="coerce").to_numpy(dtype=float)
                ax.plot(x, y, color=color, marker="o", ms=2.4, lw=1.05, label=fam)
                xs.extend(x.tolist())
                ys.extend(y.tolist())
            for row, mk, colr in ((lin, "D", LINEAR_COLOR), (lgb, "s", NATIVE_COLOR)):
                ax.scatter([metric_val(row, xmet, ev)], [metric_val(row, ymet, ev)], marker=mk, s=22, color=colr, zorder=5)
                xs.append(metric_val(row, xmet, ev))
                ys.append(metric_val(row, ymet, ev))
            include_x = (0.0,) if xmet in zero_x and xmet not in no_zero else ()
            ax.set_xlim(*padded_lim(xs, pad=0.08, include=include_x))
            ax.set_ylim(*padded_lim(ys, pad=0.08))
            if r == len(ymetrics) - 1:
                ax.set_xlabel(xlab)
            if c == 0:
                ax.set_ylabel(ylab)
            if r == 0:
                ax.set_title(xlab)
            if r == 0 and c == len(xmetrics) - 1:
                ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    return _save(plt, fig, stem)


def plot_event_locations(plt, combined, tables, min_positive, q, stem):
    from matplotlib.lines import Line2D

    events_cv = tables["transition_events_cv_mean.csv"]
    events_fold = tables["transition_events_by_fold.csv"]
    conc = tables["transition_temporal_concordance.csv"]
    span_df = tables["transition_span_summary.csv"]
    lofo = tables["transition_lofo_sensitivity.csv"]
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
            shade_cv_span(ax, low, high)
        if fam in set(lofo["family"].astype(str)) and ok:
            part = lofo.loc[(lofo["family"] == fam) & (lofo.get("valid_span", True) == True)] if "valid_span" in lofo.columns else lofo.loc[lofo["family"] == fam]
            # LOFO envelope from summary columns if present
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
