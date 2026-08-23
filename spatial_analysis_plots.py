"""Aggregate SLURM-array shard outputs from spatial_analysis.py and render figures.

Rendering is intentionally separated from the experiment runner: this script only
reads the per-shard CSVs (`neighbor_experiment_{oos,train}_metrics_*.csv`),
concatenates them, rebuilds a best-configuration overview, and writes analysis
figures that answer the core questions:

  * Do neighbor features help accuracy and vertical-equity/fairness OOS?
  * How do metrics evolve with the neighbor count k?
  * Does categorical filtering (spatial vs spatial_nofilter) matter?
  * What is the accuracy<->vertical-equity tradeoff across configurations?
  * Which neighbor groups improve over the no-neighbor baseline?

Usage:
    python spatial_analysis_plots.py --input-dir output/neighbor_experiments/<run> \
        --out-dir output/neighbor_experiments/<run>/plots
"""

from __future__ import annotations

from pathlib import Path
import argparse
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from utils.motivation_utils import (
    IAAO_LEVEL_RANGE,
    IAAO_PRB_RANGE,
    IAAO_PRD_RANGE,
    IAAO_VEI_RANGE,
)


# ---------------------------------------------------------------------
# Metric groupings. ACCURACY_R2 is the price-scale OOS R2 used to rank
# configurations; the IAAO ranges come from the shared motivation utils.
# ---------------------------------------------------------------------
ACCURACY_R2 = "R2"
RATIO_DECILE_COLS = [f"MedianRatio_q10_bin{i}" for i in range(1, 11)]
K_CURVE_METRICS = [ACCURACY_R2, "COD", "MAPE", "PRD", "PRB", "COV_IAAO"]
SUMMARY_METRICS = [ACCURACY_R2, "COD", "MAPE", "MdAPE", "PRD", "PRB", "VEI"]
TOP5_METRICS = [
    ACCURACY_R2, "R2 (log)", "RMSE", "MAE", "MAPE", "MdAPE", "COD",
    "COV_IAAO", "PRD", "PRB", "VEI", "Median ratio", "W. Mean ratio",
]
TARGET_METRICS = {
    "PRD": 1.0,
    "PRB": 0.0,
    "VEI": 0.0,
    "Median ratio": 1.0,
    "Mean ratio": 1.0,
    "W. Mean ratio": 1.0,
}
TRADEOFF_METRICS = {
    "COD": {
        "ylabel": "COD (lower = better uniformity)",
        "filename": "tradeoff_r2_vs_cod.png",
        "title": "Accuracy vs COD tradeoff (OOS)",
    },
    "PRD": {
        "ylabel": "PRD (target near 1.00)",
        "filename": "tradeoff_r2_vs_prd.png",
        "title": "Accuracy vs PRD tradeoff (OOS)",
        "target": 1.0,
        "band": IAAO_PRD_RANGE,
    },
    "PRB": {
        "ylabel": "PRB (target near 0.00)",
        "filename": "tradeoff_r2_vs_prb.png",
        "title": "Accuracy vs PRB tradeoff (OOS)",
        "target": 0.0,
        "band": IAAO_PRB_RANGE,
    },
}


# ---------------------------------------------------------------------
# Loading and summary tables
# ---------------------------------------------------------------------
def _concat(input_dir: Path, split: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(input_dir / f"neighbor_experiment_{split}_metrics_*.csv")))
    if not files:
        # A single-shard run writes the untagged file name instead.
        single = input_dir / f"neighbor_experiment_{split}_metrics.csv"
        files = [str(single)] if single.exists() else []
    if not files:
        return pd.DataFrame()
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)

    subset = [c for c in ("dataset_label", "experiment", "model", "k", "split") if c in df.columns]
    if subset:
        df = df.drop_duplicates(subset=subset, keep="last").reset_index(drop=True)
    numeric_cols = [
        "k", "bandwidth", "bandwidth_scale", "geo_weight", "feature_weight",
        "feature_bandwidth", "time_weight", "time_bandwidth_days", "rho", "N", "n_boot",
        *SUMMARY_METRICS, "OOS R2", "R2 (log)", "RMSE", "MAE", "COV_IAAO", "MKI",
    ]
    numeric_cols.extend(
        c for c in df.columns
        if c in RATIO_DECILE_COLS
    )
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_overview(oos: pd.DataFrame) -> pd.DataFrame:
    """Best k per (dataset, experiment, model): max OOS R2, then vertical-equity ties."""
    if oos.empty:
        return oos
    df = oos.copy()
    df["abs_PRD_dev"] = (df["PRD"] - 1.0).abs()
    df["abs_PRB_dev"] = df["PRB"].abs()
    keys = ["dataset_label", "experiment", "model"]
    df = df.sort_values(by=[ACCURACY_R2, "abs_PRD_dev", "abs_PRB_dev", "COD", "MAPE"],
                        ascending=[False, True, True, True, True])
    return df.groupby(keys, dropna=False, as_index=False).first()


def build_best_neighbor_summary(overview: pd.DataFrame, oos: pd.DataFrame) -> pd.DataFrame:
    """Best overall neighbor config vs no-neighbor baseline per dataset/model."""
    if overview.empty:
        return overview
    best = overview[overview["experiment"] != "baseline_no_neighbors"].sort_values(
        ACCURACY_R2, ascending=False
    ).groupby(["dataset_label", "model"], as_index=False).first()

    base = oos[oos["experiment"] == "baseline_no_neighbors"].sort_values(
        ACCURACY_R2, ascending=False
    ).groupby(["dataset_label", "model"], as_index=False).first()

    if best.empty or base.empty:
        return pd.DataFrame()

    id_cols = [
        "dataset_label", "model", "experiment", "group", "k", "kernel",
        "rho", "ratio_mode", "bandwidth", "bandwidth_scale", "geo_weight",
        "feature_weight", "feature_bandwidth", "time_trend", "time_decay",
        "time_weight", "time_bandwidth_days", "neighbor_time_rule",
    ]
    best_cols = [c for c in id_cols + SUMMARY_METRICS if c in best.columns]
    base_cols = [c for c in ["dataset_label", "model", *SUMMARY_METRICS] if c in base.columns]
    summary = best[best_cols].merge(
        base[base_cols],
        on=["dataset_label", "model"],
        how="left",
        suffixes=("_best_neighbor", "_baseline"),
    )
    summary = summary.rename(columns={
        "experiment": "best_experiment",
        "group": "best_group",
        "k": "best_k",
        "kernel": "best_kernel",
    })
    for metric in SUMMARY_METRICS:
        best_col = f"{metric}_best_neighbor"
        base_col = f"{metric}_baseline"
        if best_col in summary.columns and base_col in summary.columns:
            summary[f"delta_{metric}"] = summary[best_col] - summary[base_col]
    if "PRD_best_neighbor" in summary.columns:
        summary["abs_PRD_dev_best_neighbor"] = (summary["PRD_best_neighbor"] - 1.0).abs()
    if "PRD_baseline" in summary.columns:
        summary["abs_PRD_dev_baseline"] = (summary["PRD_baseline"] - 1.0).abs()
    if {"abs_PRD_dev_best_neighbor", "abs_PRD_dev_baseline"}.issubset(summary.columns):
        summary["delta_abs_PRD_dev"] = (
            summary["abs_PRD_dev_best_neighbor"] - summary["abs_PRD_dev_baseline"]
        )
    if "PRB_best_neighbor" in summary.columns:
        summary["abs_PRB_dev_best_neighbor"] = summary["PRB_best_neighbor"].abs()
    if "PRB_baseline" in summary.columns:
        summary["abs_PRB_dev_baseline"] = summary["PRB_baseline"].abs()
    if {"abs_PRB_dev_best_neighbor", "abs_PRB_dev_baseline"}.issubset(summary.columns):
        summary["delta_abs_PRB_dev"] = (
            summary["abs_PRB_dev_best_neighbor"] - summary["abs_PRB_dev_baseline"]
        )
    return summary


def build_best_group_summary(overview: pd.DataFrame, oos: pd.DataFrame) -> pd.DataFrame:
    """Best config within each neighbor group vs baseline per dataset/model."""
    if overview.empty:
        return overview
    group_best = overview[overview["experiment"] != "baseline_no_neighbors"].sort_values(
        ACCURACY_R2, ascending=False
    ).groupby(["dataset_label", "model", "group"], as_index=False).first()

    base = oos[oos["experiment"] == "baseline_no_neighbors"].sort_values(
        ACCURACY_R2, ascending=False
    ).groupby(["dataset_label", "model"], as_index=False).first()

    if group_best.empty or base.empty:
        return pd.DataFrame()
    cols = [
        "dataset_label", "model", "group", "experiment", "k", "kernel",
        "rho", "ratio_mode", "bandwidth", "bandwidth_scale", "geo_weight", "feature_weight",
        "feature_bandwidth", "time_weight", "time_bandwidth_days",
        *SUMMARY_METRICS,
    ]
    group_cols = [c for c in cols if c in group_best.columns]
    base_cols = [c for c in ["dataset_label", "model", *SUMMARY_METRICS] if c in base.columns]
    summary = group_best[group_cols].merge(
        base[base_cols],
        on=["dataset_label", "model"],
        how="left",
        suffixes=("_best_group", "_baseline"),
    )
    summary = summary.rename(columns={"experiment": "best_experiment", "k": "best_k"})
    for metric in SUMMARY_METRICS:
        best_col = f"{metric}_best_group"
        base_col = f"{metric}_baseline"
        if best_col in summary.columns and base_col in summary.columns:
            summary[f"delta_{metric}"] = summary[best_col] - summary[base_col]
    if "PRD_best_group" in summary.columns:
        summary["abs_PRD_dev_best_group"] = (summary["PRD_best_group"] - 1.0).abs()
    if "PRD_baseline" in summary.columns:
        summary["abs_PRD_dev_baseline"] = (summary["PRD_baseline"] - 1.0).abs()
    if {"abs_PRD_dev_best_group", "abs_PRD_dev_baseline"}.issubset(summary.columns):
        summary["delta_abs_PRD_dev"] = (
            summary["abs_PRD_dev_best_group"] - summary["abs_PRD_dev_baseline"]
        )
    if "PRB_best_group" in summary.columns:
        summary["abs_PRB_dev_best_group"] = summary["PRB_best_group"].abs()
    if "PRB_baseline" in summary.columns:
        summary["abs_PRB_dev_baseline"] = summary["PRB_baseline"].abs()
    if {"abs_PRB_dev_best_group", "abs_PRB_dev_baseline"}.issubset(summary.columns):
        summary["delta_abs_PRB_dev"] = (
            summary["abs_PRB_dev_best_group"] - summary["abs_PRB_dev_baseline"]
        )
    return summary


# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------
def _savefig(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _metric_band(metric: str) -> tuple[float, float, float] | None:
    if metric == "PRD":
        return float(IAAO_PRD_RANGE[0]), float(IAAO_PRD_RANGE[1]), 1.0
    if metric == "PRB":
        return float(IAAO_PRB_RANGE[0]), float(IAAO_PRB_RANGE[1]), 0.0
    if metric == "VEI":
        return float(IAAO_VEI_RANGE[0]), float(IAAO_VEI_RANGE[1]), 0.0
    if metric in {"Median ratio", "Mean ratio", "W. Mean ratio"}:
        return float(IAAO_LEVEL_RANGE[0]), float(IAAO_LEVEL_RANGE[1]), 1.0
    return None


def _config_label(row: pd.Series) -> str:
    parts = [str(row.get("model", "model")), str(row.get("group", row.get("experiment", "")))]
    params = []
    for col, label in (
        ("k", "k"),
        ("bandwidth_scale", "bs"),
        ("geo_weight", "gw"),
        ("feature_weight", "fw"),
        ("time_weight", "tw"),
        ("rho", "rho"),
    ):
        value = pd.to_numeric(row.get(col), errors="coerce")
        if pd.notna(value):
            params.append(f"{label}={value:g}")
    return " | ".join([p for p in parts if p] + ([", ".join(params)] if params else []))


def _best_rows_for_metric(df: pd.DataFrame, metric: str, n: int = 5) -> pd.DataFrame:
    if metric not in df.columns:
        return pd.DataFrame()
    work = df.copy()
    work["_metric_value"] = pd.to_numeric(work[metric], errors="coerce")
    work = work[work["_metric_value"].notna()].copy()
    if work.empty:
        return work
    if metric in TARGET_METRICS:
        work["_metric_rank"] = (work["_metric_value"] - TARGET_METRICS[metric]).abs()
        return work.sort_values(["_metric_rank", ACCURACY_R2], ascending=[True, False]).head(n)
    ascending = metric != ACCURACY_R2 and metric != "R2 (log)"
    return work.sort_values(["_metric_value", ACCURACY_R2], ascending=[ascending, False]).head(n)


def plot_tradeoff(oos: pd.DataFrame, out: Path, metric: str) -> None:
    """Accuracy (OOS R2) vs one vertical-equity metric; one panel per dataset."""
    if metric not in oos.columns:
        return
    spec = TRADEOFF_METRICS[metric]
    datasets = sorted(oos["dataset_label"].dropna().unique())
    if not datasets:
        return
    ncol = min(2, len(datasets))
    nrow = (len(datasets) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(7 * ncol, 5 * nrow),
                             squeeze=False, constrained_layout=True)
    for ax, ds in zip(axes.ravel(), datasets):
        sub = oos[oos["dataset_label"] == ds]
        marker_map = {"linear": "o", "lgbm": "^", "cov": "s"}
        for model in sorted(sub["model"].dropna().unique()):
            mk = marker_map.get(str(model), "o")
            m = sub[sub["model"] == model]
            if m.empty:
                continue
            base = m[m["experiment"] == "baseline_no_neighbors"]
            neigh = m[m["experiment"] != "baseline_no_neighbors"]
            ax.scatter(neigh[ACCURACY_R2], neigh[metric], marker=mk, alpha=0.55,
                       s=28, label=f"{model} (neighbor)")
            if not base.empty:
                ax.scatter(base[ACCURACY_R2], base[metric], marker=mk, s=160,
                           edgecolor="black", linewidth=1.5, color="red",
                           label=f"{model} (baseline)", zorder=5)
        if "band" in spec:
            ax.axhspan(*spec["band"], color="green", alpha=0.08, zorder=0)
        if "target" in spec:
            ax.axhline(spec["target"], color="green", lw=1.0, ls=":", alpha=0.8)
        ax.set_title(ds)
        ax.set_xlabel("OOS R2 (price scale, higher = better accuracy)")
        ax.set_ylabel(spec["ylabel"])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    for ax in axes.ravel()[len(datasets):]:
        ax.axis("off")
    fig.suptitle(spec["title"], fontsize=14)
    _savefig(fig, out, spec["filename"])


def plot_ratio_decile_curves(overview: pd.DataFrame, out: Path) -> None:
    """Median predicted/actual ratio by actual-price decile for best configs."""
    cols = [c for c in RATIO_DECILE_COLS if c in overview.columns]
    if overview.empty or len(cols) != 10:
        return
    x = list(range(1, 11))
    for ds in sorted(overview["dataset_label"].dropna().unique()):
        sub = overview[overview["dataset_label"] == ds].copy()
        models = sorted(sub["model"].dropna().unique())
        if not models:
            continue
        fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 5),
                                 squeeze=False, constrained_layout=True)
        for ax, model in zip(axes.ravel(), models):
            model_df = sub[sub["model"] == model].sort_values(
                ["experiment", ACCURACY_R2], ascending=[True, False]
            )
            for _, row in model_df.iterrows():
                y = pd.to_numeric(row[cols], errors="coerce")
                if y.notna().sum() == 0:
                    continue
                is_base = row.get("experiment") == "baseline_no_neighbors"
                ax.plot(
                    x, y.to_numpy(dtype=float),
                    marker="o",
                    lw=2.4 if is_base else 1.4,
                    ls="--" if is_base else "-",
                    alpha=0.95 if is_base else 0.75,
                    label=_config_label(row),
                )
            ax.axhspan(*IAAO_LEVEL_RANGE, color="limegreen", alpha=0.16, zorder=0)
            ax.axhline(1.0, color="forestgreen", lw=1.2, ls="--", alpha=0.9)
            ax.set_title(str(model))
            ax.set_xlabel("Actual-price decile")
            ax.set_ylabel("Median predicted / actual ratio")
            ax.set_xticks(x)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=6)
        fig.suptitle(f"Ratio curves by actual-price decile — {ds} (OOS)", fontsize=14)
        _savefig(fig, out, f"ratio_decile_curves_{ds}.png")


def plot_k_curves(oos: pd.DataFrame, out: Path) -> None:
    """Metric vs k, one figure per dataset; lines per (group, model)."""
    metrics = [m for m in K_CURVE_METRICS if m in oos.columns]
    neigh = oos[oos["experiment"] != "baseline_no_neighbors"].copy()
    base = oos[oos["experiment"] == "baseline_no_neighbors"].copy()
    model_styles = {
        "cov": {"marker": "s", "linestyle": ":"},
        "lgbm": {"marker": "^", "linestyle": "--"},
        "linear": {"marker": "o", "linestyle": "-"},
    }
    for ds in sorted(neigh["dataset_label"].dropna().unique()):
        sub = neigh[neigh["dataset_label"] == ds]
        x_values = sorted(pd.to_numeric(sub["k"], errors="coerce").dropna().unique())
        ncol = 3
        nrow = (len(metrics) + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 4 * nrow),
                                 squeeze=False, constrained_layout=True)
        for ax, metric in zip(axes.ravel(), metrics):
            for (grp, model), g in sub.groupby(["group", "model"]):
                curve = g.groupby("k")[metric].mean().sort_index()
                style = model_styles.get(str(model), {"marker": "o", "linestyle": "-"})
                ax.plot(curve.index, curve.values, marker=style["marker"],
                        ls=style["linestyle"], ms=3,
                        label=f"{grp}/{model}", alpha=0.8)
            for model in sub["model"].unique():
                b = base[(base["dataset_label"] == ds) & (base["model"] == model)]
                if not b.empty:
                    style = model_styles.get(str(model), {"marker": "o", "linestyle": "--"})
                    if x_values:
                        ax.plot([x_values[0], x_values[-1]], [b[metric].iloc[0]] * 2,
                                marker=style["marker"], ls=style["linestyle"], lw=1, ms=3,
                                alpha=0.5, label=f"baseline/{model}")
                    else:
                        ax.axhline(b[metric].iloc[0], ls=style["linestyle"], lw=1, alpha=0.5,
                                   label=f"baseline/{model}")
            ax.set_title(metric)
            ax.set_xlabel("k (neighbors)")
            ax.grid(True, alpha=0.3)
        axes.ravel()[0].legend(fontsize=7, ncol=2)
        for ax in axes.ravel()[len(metrics):]:
            ax.axis("off")
        fig.suptitle(f"Metric vs k — {ds} (OOS)", fontsize=14)
        _savefig(fig, out, f"k_curves_{ds}.png")


def plot_filter_ablation(oos: pd.DataFrame, out: Path) -> None:
    """spatial (filtered) vs spatial_nofilter, mean OOS metrics by dataset/model."""
    sub = oos[oos["group"].isin(["spatial", "spatial_nofilter"])].copy()
    if sub.empty:
        return
    metrics = [m for m in (ACCURACY_R2, "COD", "PRD", "PRB") if m in sub.columns]
    agg = sub.groupby(["dataset_label", "model", "group"])[metrics].mean().reset_index()

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4.5),
                             squeeze=False, constrained_layout=True)
    for ax, metric in zip(axes.ravel(), metrics):
        pivot = agg.pivot_table(index=["dataset_label", "model"], columns="group",
                                values=metric)
        pivot.plot(kind="bar", ax=ax, width=0.8)
        ax.set_title(metric)
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelsize=7, rotation=75)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Filtering ablation: spatial vs spatial_nofilter (OOS)", fontsize=14)
    _savefig(fig, out, "filter_ablation.png")


def plot_baseline_vs_best(overview: pd.DataFrame, oos: pd.DataFrame, out: Path) -> None:
    """Best neighbor config vs baseline per (dataset, model) for key metrics."""
    if overview.empty:
        return
    metrics = [m for m in (ACCURACY_R2, "COD", "PRD", "MAPE") if m in oos.columns]
    base = oos[oos["experiment"] == "baseline_no_neighbors"].sort_values(
        ACCURACY_R2, ascending=False
    ).groupby(["dataset_label", "model"], as_index=False).first()

    neigh = oos[oos["experiment"] != "baseline_no_neighbors"].copy()
    if base.empty or neigh.empty:
        return
    rows = []
    for (ds, model), g in neigh.groupby(["dataset_label", "model"]):
        b = base[(base["dataset_label"] == ds) & (base["model"] == model)]
        if b.empty:
            continue
        for metric in metrics:
            best = _best_rows_for_metric(g, metric, n=1)
            if best.empty:
                continue
            rows.append({"key": f"{ds}\n{model}", "metric": metric,
                         "kind": "best_neighbor", "value": best[metric].iloc[0]})
            rows.append({"key": f"{ds}\n{model}", "metric": metric,
                         "kind": "baseline", "value": b[metric].iloc[0]})
    long = pd.DataFrame(rows)
    if long.empty:
        return
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4.5),
                             squeeze=False, constrained_layout=True)
    for ax, metric in zip(axes.ravel(), metrics):
        pivot = long[long["metric"] == metric].pivot_table(
            index="key", columns="kind", values="value")
        pivot.plot(kind="bar", ax=ax, width=0.8)
        ax.set_title(metric)
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelsize=7, rotation=0)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Metric-best neighbor config vs baseline (OOS)", fontsize=14)
    _savefig(fig, out, "baseline_vs_best.png")


def plot_top5_by_metric(oos: pd.DataFrame, out: Path) -> None:
    """Top five raw configurations per metric, using target distance where relevant."""
    metrics = [m for m in TOP5_METRICS if m in oos.columns]
    if oos.empty or not metrics:
        return
    for ds in sorted(oos["dataset_label"].dropna().unique()):
        sub = oos[oos["dataset_label"] == ds].copy()
        ncol = 3
        nrow = (len(metrics) + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(7 * ncol, 3.7 * nrow),
                                 squeeze=False, constrained_layout=True)
        for ax, metric in zip(axes.ravel(), metrics):
            top = _best_rows_for_metric(sub, metric, n=5)
            if top.empty:
                ax.axis("off")
                continue
            labels = [_config_label(row) for _, row in top.iterrows()]
            values = top["_metric_value"].to_numpy(dtype=float)
            bars = ax.barh(range(len(top)), values, color="#4C78A8", alpha=0.82)
            band = _metric_band(metric)
            if band is not None:
                lo, hi, target = band
                ax.axvspan(min(lo, hi), max(lo, hi), color="limegreen", alpha=0.16, zorder=0)
                ax.axvline(target, color="forestgreen", lw=1.1, ls="--", alpha=0.9)
            ax.set_title(metric)
            ax.set_yticks(range(len(top)))
            ax.set_yticklabels(labels, fontsize=6)
            ax.invert_yaxis()
            ax.grid(True, axis="x", alpha=0.3)
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_width(),
                    bar.get_y() + bar.get_height() / 2,
                    f" {value:.4g}",
                    va="center",
                    fontsize=7,
                )
        for ax in axes.ravel()[len(metrics):]:
            ax.axis("off")
        fig.suptitle(f"Top 5 configurations by metric — {ds} (OOS)", fontsize=14)
        _savefig(fig, out, f"top5_by_metric_{ds}.png")


def plot_best_group_delta_heatmaps(summary: pd.DataFrame, out: Path) -> None:
    """Heatmaps of best-by-group deltas relative to baseline."""
    if summary.empty:
        return
    metrics = [
        ("delta_R2", "Delta R2"),
        ("delta_COD", "Delta COD"),
        ("delta_abs_PRD_dev", "Delta |PRD - 1|"),
        ("delta_abs_PRB_dev", "Delta |PRB|"),
    ]
    metrics = [(col, label) for col, label in metrics if col in summary.columns]
    if not metrics:
        return

    data = summary.copy()
    data["dataset_model"] = data["dataset_label"].astype(str) + "\n" + data["model"].astype(str)
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 5.6),
                             squeeze=False, constrained_layout=True)
    for ax, (metric, label) in zip(axes.ravel(), metrics):
        pivot = data.pivot_table(index="dataset_model", columns="group", values=metric)
        if pivot.empty:
            ax.axis("off")
            continue
        values = pivot.to_numpy(dtype=float)
        finite_values = pd.Series(values.ravel()).dropna()
        vmax = float(finite_values.abs().max()) if not finite_values.empty else 1.0
        vmax = vmax if vmax > 0 else 1.0
        im = ax.imshow(values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(label)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        for i, row in enumerate(values):
            for j, value in enumerate(row):
                if pd.notna(value):
                    ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.suptitle("Best config in each neighbor group vs baseline (OOS)", fontsize=14)
    _savefig(fig, out, "best_group_delta_heatmaps.png")


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate + plot spatial neighbor experiments.")
    p.add_argument("--input-dir", required=True, help="Directory holding per-shard CSVs.")
    p.add_argument("--out-dir", default=None, help="Defaults to <input-dir>/plots.")
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir) if args.out_dir else input_dir / "plots"

    oos = _concat(input_dir, "oos")
    train = _concat(input_dir, "train")
    if oos.empty:
        raise SystemExit(f"No OOS metric CSVs found under {input_dir}")

    overview = build_overview(oos)
    best_neighbor_summary = build_best_neighbor_summary(overview, oos)
    best_group_summary = build_best_group_summary(overview, oos)
    oos.to_csv(input_dir / "combined_oos_metrics.csv", index=False)
    if not train.empty:
        train.to_csv(input_dir / "combined_train_metrics.csv", index=False)
    overview.to_csv(input_dir / "combined_overview_oos.csv", index=False)
    if not best_neighbor_summary.empty:
        best_neighbor_summary.to_csv(input_dir / "best_neighbor_vs_baseline_oos.csv", index=False)
    if not best_group_summary.empty:
        best_group_summary.to_csv(input_dir / "best_group_vs_baseline_oos.csv", index=False)

    for metric in TRADEOFF_METRICS:
        plot_tradeoff(oos, out_dir, metric)
    plot_ratio_decile_curves(overview, out_dir)
    plot_k_curves(oos, out_dir)
    plot_filter_ablation(oos, out_dir)
    plot_baseline_vs_best(overview, oos, out_dir)
    plot_top5_by_metric(oos, out_dir)
    plot_best_group_delta_heatmaps(best_group_summary, out_dir)

    print(f"[plots] rows(oos)={len(oos)} datasets={oos['dataset_label'].nunique()} -> combined CSVs in "
          f"{input_dir} and figures in {out_dir}")


if __name__ == "__main__":
    main()
