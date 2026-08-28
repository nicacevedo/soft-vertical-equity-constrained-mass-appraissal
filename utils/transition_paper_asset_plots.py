"""Paper-asset figures. Matplotlib must be configured by the caller before import use."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.transition_paper_assets import IAAO_PRB_RANGE, IAAO_PRD_RANGE, IAAO_VEI_RANGE, ratio_shape_anchors
from utils.transition_regions import FAMILY_DISPLAY, FOLD_IDS, PRIMARY_METRICS, family_frame, is_rho_zero, numerically_equal

DIRECT_COLOR = "#1D4ED8"
SURR_COLOR = "#C2410C"
NATIVE_COLOR = "#111827"
LINEAR_COLOR = "#6B7280"
SPAN_FACE = "#9CA3AF"
LINEAR_NAME = "LinearRegression"
NATIVE_NAME = "LGBMRegressor"
PERCENT_PATH_METRICS = {"MAPE", "COV"}


def rho_x(rho) -> np.ndarray:
    x = np.asarray(rho, dtype=float)
    return np.where(x <= 0, 0.055, x)


def padded_lim(values, *, pad: float = 0.08, include: Sequence[float] = ()) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    extra = np.asarray(list(include), dtype=float)
    extra = extra[np.isfinite(extra)]
    if extra.size:
        arr = np.concatenate([arr, extra]) if arr.size else extra
    if arr.size == 0:
        return (0.0, 1.0)
    lo, hi = float(np.min(arr)), float(np.max(arr))
    span = hi - lo
    if span <= 0:
        span = max(abs(hi), 0.05)
    return lo - pad * span, hi + pad * span


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


def load_pred(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "y_true" not in df.columns:
        df["y_true"] = np.exp(df["y_true_log"].to_numpy(dtype=float))
    if "y_pred" not in df.columns:
        df["y_pred"] = np.exp(df["y_pred_log"].to_numpy(dtype=float))
    return df


def name_col(df: pd.DataFrame) -> str:
    return "model_name" if "model_name" in df.columns else "model"


def find_pred_file(result_root: Path, config_id: str, shard: str) -> Path:
    matches = list((result_root / "reporting_preview").glob(f"**/{shard}/{config_id}.parquet"))
    if not matches:
        raise FileNotFoundError(f"no {shard} parquet for {config_id}")
    return matches[0]


def combined_row(combined: pd.DataFrame, family: str, rho: Optional[float] = None) -> pd.Series:
    sub = combined.loc[combined["family"] == family]
    if rho is None:
        sub = sub.loc[sub["rho"].isna() | ~np.isfinite(pd.to_numeric(sub["rho"], errors="coerce"))]
    else:
        sub = sub.loc[np.isclose(pd.to_numeric(sub["rho"], errors="coerce"), float(rho), atol=1e-10)]
    if sub.empty:
        raise RuntimeError(f"missing combined row family={family} rho={rho}")
    return sub.iloc[0]


def metric_val(row: pd.Series, name: str, split: str) -> float:
    return float(row[f"{name}__{split}"])


def baseline_dir(result_root: Path) -> Path:
    files = list((result_root / "baseline_reporting").glob("**/test_predictions.parquet"))
    if not files:
        raise FileNotFoundError("baseline test_predictions.parquet missing")
    return files[0].parent


def load_baseline_split(result_root: Path, split: str) -> pd.DataFrame:
    fname = "test_predictions.parquet" if split == "heldout" else "assess_predictions.parquet"
    return load_pred(baseline_dir(result_root) / fname)


def load_oos_pred(result_root: Path, combined: pd.DataFrame, family: str, rho: float, evaluation: str) -> pd.DataFrame:
    row = combined_row(combined, family, rho)
    shard = "test_run_predictions" if evaluation == "heldout" else "assess_run_predictions"
    return load_pred(find_pred_file(result_root, str(row["config_id"]), shard))


def fmt_rho(x: float) -> str:
    v = float(x)
    if abs(v) < 1e-12:
        return "0"
    if abs(v - 100.0) < 1e-8:
        return "100"
    return f"{v:.3f}".rstrip("0").rstrip(".")


def log_rho_axes(ax) -> None:
    ax.set_xscale("log")
    ax.set_xticks([0.055, 0.1, 1, 10, 100])
    ax.set_xticklabels(["0", "0.1", "1", "10", "100"])
    ax.axvline(0.078, color="#D1D5DB", lw=0.8, ls=":")


def shade_direct_span(ax, low: float, high: float) -> None:
    ax.axvspan(low, high, color=SPAN_FACE, alpha=0.18, lw=0, zorder=0)


def plot_event_locations(plt, combined, v1_tables, guard, out_stem: Path) -> List[str]:
    from matplotlib.lines import Line2D

    events_cv = v1_tables["transition_events_cv_mean.csv"]
    events_fold = v1_tables["transition_events_by_fold.csv"]
    conc = v1_tables["transition_temporal_concordance.csv"]
    span_df = v1_tables["transition_span_summary.csv"]
    lofo = v1_tables["transition_lofo_sensitivity.csv"]
    metrics = [m for m, _d in PRIMARY_METRICS]
    ymap = {m: i for i, m in enumerate(reversed(metrics))}
    labels = {
        "R2_price": r"$R^2$ max",
        "MAE_price": "MAE min",
        "MAPE": "MAPE min",
        "RMSE_log": r"RMSE$_{\log}$ min",
        "COD": "COD min",
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 5.4), sharey=True)
    rng = np.random.default_rng(994)
    dspan = span_df.loc[span_df["family"] == "Direct"].iloc[0]
    for ax, fam in zip(axes, FAMILY_DISPLAY):
        color = DIRECT_COLOR if fam == "Direct" else SURR_COLOR
        if fam == "Direct" and str(dspan["status"]) == "VALID_POSITIVE_INTERIOR_SPAN":
            shade_direct_span(ax, float(dspan["rho_transition_low"]), float(dspan["rho_transition_high"]))
        valid = lofo.loc[(lofo["family"] == fam) & (lofo["valid_positive_interior_five_event_span"].astype(bool))]
        if not valid.empty:
            ax.plot(
                [float(valid["rho_transition_low"].min()), float(valid["rho_transition_high"].max())],
                [-0.55, -0.55],
                color="#6B7280",
                lw=2.0,
                solid_capstyle="butt",
            )
            ax.plot(
                [float(valid["rho_transition_low"].min()), float(valid["rho_transition_low"].min())],
                [-0.68, -0.42],
                color="#6B7280",
                lw=1.2,
            )
            ax.plot(
                [float(valid["rho_transition_high"].max()), float(valid["rho_transition_high"].max())],
                [-0.68, -0.42],
                color="#6B7280",
                lw=1.2,
            )
        for metric in metrics:
            y = ymap[metric]
            folds = events_fold.loc[(events_fold["family"] == fam) & (events_fold["metric"] == metric)]
            for _, row in folds.iterrows():
                if pd.isna(row["rho_low"]):
                    continue
                ax.scatter(
                    rho_x([float(row["rho_low"])])[0],
                    y + float(rng.uniform(-0.14, 0.14)),
                    s=12,
                    color=color,
                    alpha=0.45,
                    zorder=3,
                    linewidths=0,
                )
            cv = events_cv.loc[(events_cv["family"] == fam) & (events_cv["metric"] == metric)].iloc[0]
            if pd.notna(cv["rho_low"]):
                ax.scatter(rho_x([float(cv["rho_low"])])[0], y, s=70, marker="o", color=color, zorder=5, edgecolors="white", linewidths=0.6)
            for split, mk in (("heldout", "s"), ("forward_2025", "^")):
                part = conc.loc[(conc["family"] == fam) & (conc["split"] == split) & (conc["metric"] == metric)]
                if part.empty or pd.isna(part.iloc[0]["rho_low"]):
                    continue
                ax.scatter(
                    rho_x([float(part.iloc[0]["rho_low"])])[0],
                    y,
                    s=42,
                    marker=mk,
                    facecolors="white",
                    edgecolors="#111827",
                    linewidths=1.0,
                    zorder=6,
                )
        log_rho_axes(ax)
        ax.set_xlim(0.04, 130)
        ax.set_ylim(-0.9, 4.6)
        ax.set_yticks(list(ymap.values()))
        ax.set_yticklabels([labels[m] for m in reversed(metrics)])
        ax.set_title(fam)
        ax.set_xlabel(r"Penalty strength $\rho$")
        if fam == "Direct":
            ax.text(0.12, 4.35, r"shaded: frozen CV span; $\rho=0.1$ is first positive grid point", fontsize=6.5, color="#374151")
        else:
            ax.text(0.12, 4.35, "common five-metric CV span not supported", fontsize=6.5, color="#374151")
            ax.text(0.045, ymap["RMSE_log"] + 0.32, r"CV RMSE$_{\log}$ at $\rho=0$", fontsize=6.2, color="#374151")
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1D4ED8", ms=8, label="Full-CV event"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1D4ED8", ms=4, alpha=0.5, label="Fold events"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="white", markeredgecolor="#111827", ms=6, label="Held-out"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="white", markeredgecolor="#111827", ms=6, label="2025"),
        Line2D([0], [0], color="#6B7280", lw=2, label="Valid LOFO endpoint range"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(plt, fig, out_stem, guard)


def plot_primary_paths(plt, combined, v1_tables, guard, out_stem: Path) -> List[str]:
    span_df = v1_tables["transition_span_summary.csv"]
    events_cv = v1_tables["transition_events_cv_mean.csv"]
    conc = v1_tables["transition_temporal_concordance.csv"]
    dspan = span_df.loc[span_df["family"] == "Direct"].iloc[0]
    fig, axes = plt.subplots(5, 2, figsize=(8.8, 10.2), sharex=True)
    for r, (metric, _d) in enumerate(PRIMARY_METRICS):
        ylab = {"R2_price": r"$R^2_P$", "MAE_price": "MAE", "MAPE": "MAPE", "RMSE_log": r"RMSE$_{\log}$", "COD": "COD"}[metric]
        row_vals = []
        for fam in FAMILY_DISPLAY:
            sub = family_frame(combined, fam)
            for suf in ["CV_mean"] + [f"fold_{k}" for k in FOLD_IDS] + ["heldout", "forward_2025"]:
                row_vals.extend(pd.to_numeric(sub[f"{metric}__{suf}"], errors="coerce").tolist())
        ylim = padded_lim(row_vals, pad=0.08)
        for c, fam in enumerate(FAMILY_DISPLAY):
            ax = axes[r, c]
            sub = family_frame(combined, fam).sort_values("rho")
            x = rho_x(sub["rho"].to_numpy(dtype=float))
            color = DIRECT_COLOR if fam == "Direct" else SURR_COLOR
            if fam == "Direct" and str(dspan["status"]) == "VALID_POSITIVE_INTERIOR_SPAN":
                shade_direct_span(ax, float(dspan["rho_transition_low"]), float(dspan["rho_transition_high"]))
            for k in FOLD_IDS:
                ax.plot(x, sub[f"{metric}__fold_{k}"], color="#9CA3AF", lw=0.7, alpha=0.65)
            ax.plot(x, sub[f"{metric}__CV_mean"], color=color, lw=2.0, label="Equal-weight CV")
            ax.plot(x, sub[f"{metric}__heldout"], color=color, ls="--", lw=1.2, marker="s", ms=2.5, label="Held-out")
            ax.plot(x, sub[f"{metric}__forward_2025"], color=color, ls=":", lw=1.2, marker="^", ms=2.5, label="2025")
            cv = events_cv.loc[(events_cv["family"] == fam) & (events_cv["metric"] == metric)].iloc[0]
            if pd.notna(cv["rho_low"]) and pd.notna(cv["metric_value"]):
                ax.scatter(rho_x([float(cv["rho_low"])])[0], float(cv["metric_value"]), s=36, marker="o", color=color, zorder=5)
            for split, mk in (("heldout", "s"), ("forward_2025", "^")):
                part = conc.loc[(conc["family"] == fam) & (conc["split"] == split) & (conc["metric"] == metric)]
                if part.empty or pd.isna(part.iloc[0]["rho_low"]):
                    continue
                ax.scatter(rho_x([float(part.iloc[0]["rho_low"])])[0], float(part.iloc[0]["metric_value"]), s=28, marker=mk, facecolors="white", edgecolors="#111827", zorder=6)
            log_rho_axes(ax)
            ax.set_ylim(*ylim)
            if r == 0:
                ax.set_title(fam)
            if c == 0:
                ax.set_ylabel(ylab)
            if r == 4:
                ax.set_xlabel(r"$\rho$")
            if r == 0 and c == 1:
                ax.legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    return _save(plt, fig, out_stem, guard)


def plot_regret(plt, regret: pd.DataFrame, guard, out_stem: Path) -> List[str]:
    metrics = [m for m, _d in PRIMARY_METRICS]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = np.arange(len(metrics))
    w = 0.35
    h = regret.loc[regret["split"] == "heldout"].set_index("metric").loc[metrics]
    f = regret.loc[regret["split"] == "forward_2025"].set_index("metric").loc[metrics]
    ax.bar(x - w / 2, h["normalized_regret"].fillna(0).to_numpy(dtype=float), w, color=DIRECT_COLOR, label="Held-out")
    ax.bar(x + w / 2, f["normalized_regret"].fillna(0).to_numpy(dtype=float), w, color="#93C5FD", label="2025")
    ax.set_xticks(x)
    ax.set_xticklabels(["R2 max", "MAE min", "MAPE min", r"RMSE$_{\log}$ min", "COD min"])
    ax.set_ylabel("Normalized span regret")
    ax.set_title("Direct frozen CV span: out-of-time normalized regret")
    ax.legend(frameon=False)
    fig.tight_layout()
    return _save(plt, fig, out_stem, guard)


def plot_baseline_motivation(plt, result_root: Path, guard, out_stem: Path) -> List[str]:
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogFormatterSciNotation, LogLocator

    frames = []
    for split in ("heldout", "forward_2025"):
        raw = load_baseline_split(result_root, split)
        work = raw.copy()
        work["split"] = split
        work["model"] = work[name_col(work)].map(
            {"LinearRegression": LINEAR_NAME, LINEAR_NAME: LINEAR_NAME, NATIVE_NAME: NATIVE_NAME, "LGBMRegressor": NATIVE_NAME}
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
    ymin, ymax = padded_lim(profile[["median_ratio", "ratio_q25", "ratio_q75"]].to_numpy().ravel(), pad=0.08)
    colors = {LINEAR_NAME: "#0072B2", NATIVE_NAME: "#D55E00"}
    titles = {LINEAR_NAME: "Linear regression", NATIVE_NAME: "Unpenalized LightGBM"}
    split_labs = {"heldout": "Held-out", "forward_2025": "2025"}
    for r, split in enumerate(("heldout", "forward_2025")):
        for c, model in enumerate((LINEAR_NAME, NATIVE_NAME)):
            ax = axes[r, c]
            sub = preds.loc[(preds["split"] == split) & (preds["model"] == model)]
            prof = profile.loc[(profile["split"] == split) & (profile["model"] == model)]
            color = colors[model]
            ax.fill_between(prof["median_sale_price"], prof["ratio_q25"], prof["ratio_q75"], color=color, alpha=0.16, lw=0)
            ax.plot(prof["median_sale_price"], prof["median_ratio"], color=color, marker="o", ms=2.5, lw=1.5)
            ax.axhline(1.0, color="#111827", ls=(0, (2, 2)), lw=0.9)
            ax.set_xscale("log", base=10)
            ax.set_xlim(*xlim)
            ax.set_ylim(ymin, ymax)
            ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            ax.xaxis.set_major_formatter(LogFormatterSciNotation(base=10, labelOnlyBase=False, minor_thresholds=(np.inf, np.inf)))
            sale = sub["sale_price"].to_numpy(dtype=float)
            ratio = sub["assessment_ratio"].to_numpy(dtype=float)
            ok = np.isfinite(sale) & (sale > 0) & np.isfinite(ratio) & (ratio > 0)
            ylog = np.log(sale[ok])
            plog = np.log(sale[ok] * ratio[ok])
            beta = float(np.cov(plog - ylog, ylog, ddof=0)[0, 1] / np.var(ylog, ddof=0))
            ax.legend(handles=[Line2D([], [], ls="None", label=rf"$\beta_{{\log}}$ = {beta:.3f}")], loc="lower left", frameon=False, handlelength=0, handletextpad=0, fontsize=7.5)
            if r == 0:
                ax.set_title(titles[model])
            if c == 0:
                ax.set_ylabel(f"{split_labs[split]}\nValuation-to-sale ratio")
            if r == 1:
                ax.set_xlabel(r"Sale price (log$_{10}$ scale)")
    fig.tight_layout()
    return _save(plt, fig, out_stem, guard)


def plot_ratio_shape(plt, result_root: Path, combined: pd.DataFrame, guard, out_stem: Path) -> List[str]:
    grid = combined.loc[combined["family"] == "Direct", "rho"].to_numpy(dtype=float)
    anchors = ratio_shape_anchors(grid)
    cmap = plt.cm.viridis
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.4), sharex=True, sharey=True)
    x_all: List[float] = []
    y_all: List[float] = []
    evals = (("heldout", "Held-out"), ("forward_2025", "2025 forward"))
    for r, (ev, evlab) in enumerate(evals):
        for c, fam in enumerate(FAMILY_DISPLAY):
            ax = axes[r, c]
            for i, rho in enumerate(anchors):
                pred = load_oos_pred(result_root, combined, fam, float(rho), ev)
                sale = pred["y_true"].to_numpy(dtype=float)
                ratio = pred["y_pred"].to_numpy(dtype=float) / sale
                prof = equal_count_bins(sale, ratio)
                x_all.extend(prof["median_sale_price"].tolist())
                y_all.extend(prof["median_ratio"].tolist())
                color = cmap(0.12 + 0.8 * i / max(len(anchors) - 1, 1))
                ax.plot(prof["median_sale_price"], prof["median_ratio"], color=color, lw=1.5, marker="o", ms=2.2, label=rf"$\rho$={fmt_rho(rho)}")
            ax.axhline(1.0, color="#111827", ls=(0, (2, 2)), lw=0.8)
            ax.set_xscale("log", base=10)
            if r == 0:
                ax.set_title(fam)
            if c == 0:
                ax.set_ylabel(f"{evlab}\nValuation-to-sale ratio")
            if r == 1:
                ax.set_xlabel("Sale price")
            if r == 0 and c == 1:
                ax.legend(fontsize=7, frameon=False, loc="lower left")
    xmin, xmax = min(x_all), max(x_all)
    ymin, ymax = padded_lim(y_all, pad=0.08, include=(1.0,))
    for ax in axes.ravel():
        ax.set_xlim(xmin / 1.05, xmax * 1.05)
        ax.set_ylim(ymin, ymax)
    fig.tight_layout()
    return _save(plt, fig, out_stem, guard)


def plot_mechanism(plt, combined: pd.DataFrame, guard, out_stem: Path) -> List[str]:
    fig, axes = plt.subplots(3, 2, figsize=(8.4, 8.2), sharex=True)
    metrics = (("Beta_log", r"$\beta_{\log}$", True), ("Delta_NL", r"$\Delta_{\mathrm{NL}}$", True), ("dCor_e_y", r"$\mathrm{dCor}(e,y)$", False))
    styles = {"heldout": ("-", "o"), "forward_2025": ("--", "s")}
    for r, (col, ylab, zero) in enumerate(metrics):
        row_vals: List[float] = []
        for fam in FAMILY_DISPLAY:
            sub = combined.loc[combined["family"] == fam]
            for ev in ("heldout", "forward_2025"):
                row_vals.extend(pd.to_numeric(sub[f"{col}__{ev}"], errors="coerce").tolist())
        ylim = padded_lim(row_vals, pad=0.08, include=(0.0,) if zero else ())
        for c, fam in enumerate(FAMILY_DISPLAY):
            ax = axes[r, c]
            sub = combined.loc[combined["family"] == fam].sort_values("rho")
            color = DIRECT_COLOR if fam == "Direct" else SURR_COLOR
            for ev, (ls, mk) in styles.items():
                ax.plot(rho_x(sub["rho"].to_numpy(dtype=float)), sub[f"{col}__{ev}"], color=color, ls=ls, marker=mk, ms=3.5, lw=1.4, label="Held-out" if ev == "heldout" else "2025")
            log_rho_axes(ax)
            ax.set_ylim(*ylim)
            if zero:
                ax.axhline(0.0, color="#111827", lw=0.8, ls=":")
            if r == 0:
                ax.set_title(fam)
            if c == 0:
                ax.set_ylabel(ylab)
            if r == 2:
                ax.set_xlabel(r"Penalty strength $\rho$")
            if r == 0 and c == 1:
                ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return _save(plt, fig, out_stem, guard)


def _tradeoff(plt, combined, ymet, ylab, band, band_label, stem, guard, add_mki_line: bool = False):
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.4))
    specs = (
        (0, 0, "heldout", "Held-out"),
        (0, 1, "heldout", "Held-out"),
        (1, 0, "forward_2025", "2025"),
        (1, 1, "forward_2025", "2025"),
    )
    # rebuilt below
    lin = combined_row(combined, "Linear")
    lgb = combined_row(combined, "LightGBM")
    r2_by_col: Dict[int, List[float]] = {0: [], 1: []}
    y_by_col: Dict[int, List[float]] = {0: [], 1: []}
    panels = (
        (0, 0, "heldout", ymet[0], ylab[0], band[0], band_label[0], False),
        (0, 1, "heldout", ymet[1], ylab[1], band[1], band_label[1], add_mki_line),
        (1, 0, "forward_2025", ymet[0], ylab[0], band[0], band_label[0], False),
        (1, 1, "forward_2025", ymet[1], ylab[1], band[1], band_label[1], add_mki_line),
    )
    for r, c, ev, met, ylb, bnd, blab, mki in panels:
        ax = axes[r, c]
        if bnd is not None:
            ax.axhspan(bnd[0], bnd[1], color="#9CA3AF", alpha=0.18, lw=0, label=blab)
        if mki:
            ax.axhline(1.0, color="#111827", ls=":", lw=0.9, label="MKI = 1")
        for fam, color in (("Direct", DIRECT_COLOR), ("Surrogate", SURR_COLOR)):
            sub = combined.loc[combined["family"] == fam].sort_values("rho")
            x = sub[f"R2_price__{ev}"].to_numpy(dtype=float)
            y = sub[f"{met}__{ev}"].to_numpy(dtype=float)
            ax.plot(x, y, color=color, marker="o", ms=3.2, lw=1.3, label=fam)
            if len(x) >= 2:
                ax.annotate("", xy=(x[-1], y[-1]), xytext=(x[-2], y[-2]), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.0))
            r2_by_col[c].extend(x.tolist())
            y_by_col[c].extend(y.tolist())
        r2_by_col[c].extend([metric_val(lin, "R2_price", ev), metric_val(lgb, "R2_price", ev)])
        y_by_col[c].extend([metric_val(lin, met, ev), metric_val(lgb, met, ev)])
        ax.scatter([metric_val(lin, "R2_price", ev)], [metric_val(lin, met, ev)], marker="D", s=36, color=LINEAR_COLOR, zorder=5, label="Linear")
        ax.scatter([metric_val(lgb, "R2_price", ev)], [metric_val(lgb, met, ev)], marker="s", s=36, color=NATIVE_COLOR, zorder=5, label="LightGBM")
        ax.set_xlabel(r"$R^2_P$")
        ax.set_ylabel(ylb)
        ax.set_title(f"{'Held-out' if ev=='heldout' else '2025'}: {met}")
        if r == 0 and c == 1:
            ax.legend(frameon=False, fontsize=7)
    for c in (0, 1):
        xlim = padded_lim(r2_by_col[c], pad=0.06)
        ylim = padded_lim(y_by_col[c], pad=0.08)
        axes[0, c].set_xlim(*xlim)
        axes[1, c].set_xlim(*xlim)
        axes[0, c].set_ylim(*ylim)
        axes[1, c].set_ylim(*ylim)
    fig.tight_layout()
    return _save(plt, fig, stem, guard)


def plot_accuracy_equity(plt, combined, guard, out_stem: Path) -> List[str]:
    return _tradeoff(
        plt, combined,
        ("PRD", "VEI"),
        ("PRD", r"VEI (\%)"),
        (IAAO_PRD_RANGE, IAAO_VEI_RANGE),
        ("Reference band", "Reference band"),
        out_stem, guard, False,
    )


def plot_prb_mki(plt, combined, guard, out_stem: Path) -> List[str]:
    return _tradeoff(
        plt, combined,
        ("PRB", "MKI"),
        ("PRB", "MKI"),
        (IAAO_PRB_RANGE, None),
        ("PRB reference band", ""),
        out_stem, guard, True,
    )


def plot_metric_paths(plt, combined, metrics: Sequence[Tuple[str, str]], guard, out_stem: Path) -> List[str]:
    fig, axes = plt.subplots(len(metrics), 2, figsize=(8.4, 2.15 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = np.array([axes])
    for r, (col, ylab) in enumerate(metrics):
        row_vals: List[float] = []
        for fam in FAMILY_DISPLAY:
            sub = combined.loc[combined["family"] == fam]
            for ev in ("heldout", "forward_2025"):
                vals = pd.to_numeric(sub[f"{col}__{ev}"], errors="coerce").to_numpy(dtype=float)
                if col in PERCENT_PATH_METRICS:
                    vals = 100.0 * vals
                row_vals.extend(vals.tolist())
        ylim = padded_lim(row_vals, pad=0.08)
        for c, fam in enumerate(FAMILY_DISPLAY):
            ax = axes[r, c]
            sub = combined.loc[combined["family"] == fam].sort_values("rho")
            color = DIRECT_COLOR if fam == "Direct" else SURR_COLOR
            x = rho_x(sub["rho"].to_numpy(dtype=float))
            y_h = sub[f"{col}__heldout"].to_numpy(dtype=float)
            y_f = sub[f"{col}__forward_2025"].to_numpy(dtype=float)
            if col in PERCENT_PATH_METRICS:
                y_h = 100.0 * y_h
                y_f = 100.0 * y_f
            ax.plot(x, y_h, color=color, marker="o", ms=3, lw=1.3, label="Held-out")
            ax.plot(x, y_f, color=color, ls="--", marker="s", ms=3, lw=1.2, label="2025")
            log_rho_axes(ax)
            ax.set_ylim(*ylim)
            if r == 0:
                ax.set_title(fam)
            if c == 0:
                ax.set_ylabel(ylab)
            if r == len(metrics) - 1:
                ax.set_xlabel(r"$\rho$")
            if r == 0 and c == 1:
                ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return _save(plt, fig, out_stem, guard)


def plot_cv_stability(plt, combined, guard, out_stem: Path) -> List[str]:
    metrics = (("R2_price", r"$R^2_P$"), ("PRD", "PRD"), ("VEI", r"VEI (\%)"), ("Beta_log", r"$\beta_{\log}$"))
    fig, axes = plt.subplots(4, 2, figsize=(8.4, 9.2), sharex=True)
    for r, (col, ylab) in enumerate(metrics):
        for c, fam in enumerate(FAMILY_DISPLAY):
            ax = axes[r, c]
            sub = combined.loc[combined["family"] == fam].sort_values("rho")
            x = rho_x(sub["rho"].to_numpy(dtype=float))
            color = DIRECT_COLOR if fam == "Direct" else SURR_COLOR
            for k in range(1, 8):
                ax.plot(x, sub[f"{col}__fold_{k}"], color="#9CA3AF", lw=0.8, alpha=0.7)
            ax.plot(x, sub[f"{col}__CV_mean"], color=color, lw=2.0)
            log_rho_axes(ax)
            if r == 0:
                ax.set_title(fam)
            if c == 0:
                ax.set_ylabel(ylab)
            if r == 3:
                ax.set_xlabel(r"$\rho$")
    fig.tight_layout()
    return _save(plt, fig, out_stem, guard)


def plot_vei_groups(plt, result_root: Path, guard, out_stem: Path) -> List[str]:
    from utils.motivation_utils import vei_percentile_group_profile

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.2), sharex=True, sharey=True)
    specs = (
        (0, 0, "heldout", LINEAR_NAME, "Held-out Linear"),
        (0, 1, "heldout", NATIVE_NAME, "Held-out LightGBM"),
        (1, 0, "forward_2025", LINEAR_NAME, "2025 Linear"),
        (1, 1, "forward_2025", NATIVE_NAME, "2025 LightGBM"),
    )
    for r, c, split, model, title in specs:
        ax = axes[r, c]
        raw = load_baseline_split(result_root, split)
        mapped = raw[name_col(raw)].replace({"LinearRegression": LINEAR_NAME, "LGBMRegressor": NATIVE_NAME})
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
    fig.tight_layout()
    return _save(plt, fig, out_stem, guard)


def _save(plt, fig, stem: Path, guard) -> List[str]:
    pdf = guard.allowed(stem.with_suffix(".pdf"))
    png = guard.allowed(stem.with_suffix(".png"))
    fig.savefig(pdf, format="pdf", bbox_inches="tight")
    fig.savefig(png, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return [str(pdf), str(png)]
