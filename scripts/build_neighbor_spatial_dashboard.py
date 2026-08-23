#!/usr/bin/env python
"""Build a self-contained dashboard for spatial neighbor experiments.

The dashboard scans output/neighbor_experiments/spatial_* runs, reads the
aggregate metric CSVs produced by spatial_analysis.py/spatial_analysis_plots.py,
and writes one standalone HTML file with embedded plots and summary tables.

Run:
    conda run -n fairness_env python scripts/build_neighbor_spatial_dashboard.py
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("output/neighbor_experiments")
OUT_HTML = ROOT / "neighbor_spatial_dashboard.html"

METRICS = ["R2", "COD", "MAPE", "MdAPE", "PRD", "PRB", "VEI", "Median ratio"]
NUMERIC_COLS = [
    "k",
    "N",
    "n_boot",
    "R2",
    "OOS R2",
    "R2 (log)",
    "RMSE",
    "MAE",
    "MAPE",
    "MdAPE",
    "COD",
    "COV_IAAO",
    "VEI",
    "PRD",
    "PRB",
    "MKI",
    "Median ratio",
    "Mean ratio",
    "W. Mean ratio",
]
PLOT_ORDER = [
    ("tradeoff_r2_vs_cod.png", "R2 vs COD"),
    ("tradeoff_r2_vs_prd.png", "R2 vs PRD"),
    ("tradeoff_r2_vs_prb.png", "R2 vs PRB"),
    ("baseline_vs_best.png", "Best neighbor vs baseline"),
    ("filter_ablation.png", "Spatial filter ablation"),
    ("k_curves_all_filtered.png", "k curves: all filtered"),
    ("k_curves_arms_length_or_missing.png", "k curves: arms length or missing"),
    ("k_curves_deed_01_02.png", "k curves: deed 01/02"),
    ("k_curves_single_family.png", "k curves: single family"),
]


@dataclass
class NeighborRun:
    name: str
    folder: Path
    oos: pd.DataFrame
    train: pd.DataFrame
    overview: pd.DataFrame


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _build_overview(oos: pd.DataFrame) -> pd.DataFrame:
    if oos.empty:
        return pd.DataFrame()
    df = oos.copy()
    df["abs_PRD_dev"] = (df["PRD"] - 1.0).abs()
    df["abs_PRB_dev"] = df["PRB"].abs()
    df = df.sort_values(
        ["R2", "abs_PRD_dev", "abs_PRB_dev", "COD", "MAPE"],
        ascending=[False, True, True, True, True],
    )
    return df.groupby(["dataset_label", "experiment", "model"], dropna=False, as_index=False).first()


def load_runs(root: Path, selected_runs: Iterable[str] | None = None) -> list[NeighborRun]:
    selected = set(selected_runs or [])
    folders = sorted(p for p in root.glob("spatial_*") if p.is_dir())
    runs: list[NeighborRun] = []
    for folder in folders:
        if selected and folder.name not in selected:
            continue
        oos = _read_csv(folder / "combined_oos_metrics.csv")
        if oos.empty:
            continue
        train = _read_csv(folder / "combined_train_metrics.csv")
        overview = _read_csv(folder / "combined_overview_oos.csv")
        if overview.empty:
            overview = _build_overview(oos)
        runs.append(NeighborRun(folder.name, folder, oos, train, overview))
    return runs


def _png_data_uri(fig, *, dpi: int = 115) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")


def _image_file_data_uri(path: Path) -> str | None:
    if not path.exists() or path.stat().st_size <= 0:
        return None
    suffix = path.suffix.lower().lstrip(".")
    mime = "image/png" if suffix == "png" else f"image/{suffix}"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _fmt(value, digits: int = 3, missing: str = "-") -> str:
    try:
        x = float(value)
    except Exception:
        return missing
    if not np.isfinite(x):
        return missing
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    return f"{x:.{digits}f}"


def _html_table(df: pd.DataFrame, *, max_rows: int = 20, classes: str = "table") -> str:
    if df.empty:
        return "<p class='muted'>No rows available.</p>"
    show = df.head(max_rows).copy()
    for col in show.columns:
        if pd.api.types.is_numeric_dtype(show[col]):
            show[col] = show[col].map(lambda x: _fmt(x, 4))
    return show.to_html(index=False, escape=True, classes=classes, border=0)


def _baseline_rows(oos: pd.DataFrame) -> pd.DataFrame:
    base = oos[oos["experiment"] == "baseline_no_neighbors"].copy()
    if base.empty:
        return base
    return (
        base.sort_values("R2", ascending=False)
        .groupby(["dataset_label", "model"], as_index=False, dropna=False)
        .first()
    )


def _neighbor_rows(oos: pd.DataFrame) -> pd.DataFrame:
    return oos[oos["experiment"] != "baseline_no_neighbors"].copy()


def best_examples(run: NeighborRun) -> pd.DataFrame:
    """Best examples by objective, measured against each dataset/model baseline."""
    oos = run.oos.copy()
    base = _baseline_rows(oos)
    neigh = _neighbor_rows(oos)
    if base.empty or neigh.empty:
        return pd.DataFrame()
    merged = neigh.merge(
        base[["dataset_label", "model", "R2", "COD", "PRD", "PRB", "MAPE"]],
        on=["dataset_label", "model"],
        how="left",
        suffixes=("", "_baseline"),
    )
    merged["run"] = run.name
    merged["R2_gain"] = merged["R2"] - merged["R2_baseline"]
    merged["COD_reduction"] = merged["COD_baseline"] - merged["COD"]
    merged["PRD_abs_error_reduction"] = (merged["PRD_baseline"] - 1.0).abs() - (merged["PRD"] - 1.0).abs()
    merged["PRB_abs_error_reduction"] = merged["PRB_baseline"].abs() - merged["PRB"].abs()

    objectives = [
        ("accuracy", "R2_gain"),
        ("COD", "COD_reduction"),
        ("PRD", "PRD_abs_error_reduction"),
        ("PRB", "PRB_abs_error_reduction"),
    ]
    rows = []
    for objective, col in objectives:
        valid = merged[np.isfinite(merged[col])].copy()
        if valid.empty:
            continue
        r = valid.sort_values(col, ascending=False).iloc[0].copy()
        r["objective"] = objective
        r["improvement"] = r[col]
        rows.append(r)
    if not rows:
        return pd.DataFrame()
    keep = [
        "run",
        "objective",
        "dataset_label",
        "model",
        "group",
        "experiment",
        "k",
        "kernel",
        "R2",
        "R2_baseline",
        "R2_gain",
        "COD",
        "COD_baseline",
        "COD_reduction",
        "PRD",
        "PRD_baseline",
        "PRD_abs_error_reduction",
        "PRB",
        "PRB_baseline",
        "PRB_abs_error_reduction",
    ]
    return pd.DataFrame(rows)[[c for c in keep if c in pd.DataFrame(rows).columns]]


def best_group_summary(run: NeighborRun) -> pd.DataFrame:
    overview = run.overview.copy()
    base = _baseline_rows(run.oos)
    neigh = overview[overview["experiment"] != "baseline_no_neighbors"].copy()
    if base.empty or neigh.empty:
        return pd.DataFrame()
    best = (
        neigh.sort_values("R2", ascending=False)
        .groupby(["dataset_label", "model", "group"], as_index=False, dropna=False)
        .first()
    )
    out = best.merge(
        base[["dataset_label", "model", "R2", "COD", "PRD", "PRB", "MAPE"]],
        on=["dataset_label", "model"],
        how="left",
        suffixes=("", "_baseline"),
    )
    out["run"] = run.name
    out["R2_gain"] = out["R2"] - out["R2_baseline"]
    out["COD_reduction"] = out["COD_baseline"] - out["COD"]
    out["PRD_abs_error_reduction"] = (out["PRD_baseline"] - 1.0).abs() - (out["PRD"] - 1.0).abs()
    out["PRB_abs_error_reduction"] = out["PRB_baseline"].abs() - out["PRB"].abs()
    return out


def generalization_summary(run: NeighborRun) -> pd.DataFrame:
    """Train-vs-OOS gaps for baseline and the best OOS neighbor per dataset/model."""
    if run.train.empty or run.overview.empty:
        return pd.DataFrame()
    base = run.overview[run.overview["experiment"] == "baseline_no_neighbors"].copy()
    neigh = run.overview[run.overview["experiment"] != "baseline_no_neighbors"].copy()
    best_neigh = (
        neigh.sort_values("R2", ascending=False)
        .groupby(["dataset_label", "model"], as_index=False, dropna=False)
        .first()
        if not neigh.empty
        else pd.DataFrame()
    )
    chosen = pd.concat([base.assign(selection="baseline"), best_neigh.assign(selection="best_neighbor")])
    if chosen.empty:
        return pd.DataFrame()
    train = run.train.copy()
    for df in (chosen, train):
        df["_k_key"] = pd.to_numeric(df.get("k", np.nan), errors="coerce").round(8).astype(str)
        df.loc[df["_k_key"].isin(["nan", "<NA>"]), "_k_key"] = ""
    keys = ["dataset_label", "model", "experiment", "_k_key"]
    train_cols = [c for c in keys + ["R2", "COD", "PRD", "PRB", "MAPE"] if c in train.columns]
    out = chosen.merge(
        train[train_cols],
        on=keys,
        how="left",
        suffixes=("_oos", "_train"),
    )
    out["run"] = run.name
    for metric in ["R2", "COD", "PRD", "PRB", "MAPE"]:
        oos_col = f"{metric}_oos"
        train_col = f"{metric}_train"
        if oos_col in out.columns and train_col in out.columns:
            out[f"train_minus_oos_{metric}"] = out[train_col] - out[oos_col]
    keep = [
        "run",
        "selection",
        "dataset_label",
        "model",
        "group",
        "experiment",
        "k",
        "R2_train",
        "R2_oos",
        "train_minus_oos_R2",
        "COD_train",
        "COD_oos",
        "train_minus_oos_COD",
        "PRD_train",
        "PRD_oos",
        "PRB_train",
        "PRB_oos",
    ]
    return out[[c for c in keep if c in out.columns]].sort_values(
        ["run", "dataset_label", "model", "selection"]
    )


def run_inventory(runs: list[NeighborRun]) -> pd.DataFrame:
    rows = []
    for run in runs:
        oos = run.oos
        neigh = _neighbor_rows(oos)
        k_vals = pd.to_numeric(neigh.get("k", pd.Series(dtype=float)), errors="coerce").dropna()
        n_boot = pd.to_numeric(oos.get("n_boot", pd.Series(dtype=float)), errors="coerce").dropna()
        rows.append(
            {
                "run": run.name,
                "oos_rows": len(oos),
                "train_rows": len(run.train),
                "overview_rows": len(run.overview),
                "datasets": oos["dataset_label"].nunique(),
                "models": ", ".join(sorted(map(str, oos["model"].dropna().unique()))),
                "groups": ", ".join(sorted(map(str, oos["group"].dropna().unique()))),
                "k_range": f"{int(k_vals.min())}-{int(k_vals.max())}" if not k_vals.empty else "-",
                "kernels": ", ".join(sorted(k for k in map(str, neigh["kernel"].dropna().unique()) if k)),
                "n_boot": int(n_boot.max()) if not n_boot.empty else 0,
            }
        )
    return pd.DataFrame(rows)


def fig_group_improvements(all_group: pd.DataFrame) -> str | None:
    if all_group.empty:
        return None
    agg = (
        all_group.groupby(["run", "group"], as_index=False)
        [["R2_gain", "COD_reduction", "PRD_abs_error_reduction", "PRB_abs_error_reduction"]]
        .mean()
    )
    metrics = [
        ("R2_gain", "Mean R2 gain (higher is better)"),
        ("COD_reduction", "Mean COD reduction (higher is better)"),
        ("PRD_abs_error_reduction", "Mean |PRD-1| reduction (higher is better)"),
        ("PRB_abs_error_reduction", "Mean |PRB| reduction (higher is better)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), squeeze=False)
    for ax, (metric, label) in zip(axes.ravel(), metrics):
        pivot = agg.pivot_table(index="group", columns="run", values=metric)
        pivot = pivot.reindex(sorted(pivot.index))
        pivot.plot(kind="bar", ax=ax, width=0.78)
        ax.axhline(0, color="#111827", lw=0.8)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelrotation=35)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle("Average best-by-group improvements vs no-neighbor baseline", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return _png_data_uri(fig)


def fig_run_metric_ranges(runs: list[NeighborRun]) -> str | None:
    rows = []
    for run in runs:
        neigh = _neighbor_rows(run.oos)
        for metric in ["R2", "COD", "PRD", "PRB"]:
            if metric not in neigh:
                continue
            rows.append(
                {
                    "run": run.name,
                    "metric": metric,
                    "min": neigh[metric].min(),
                    "median": neigh[metric].median(),
                    "max": neigh[metric].max(),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return None
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.6), squeeze=False)
    for ax, metric in zip(axes.ravel(), ["R2", "COD", "PRD", "PRB"]):
        sub = df[df["metric"] == metric].copy()
        x = np.arange(len(sub))
        ax.vlines(x, sub["min"], sub["max"], color="#94a3b8", lw=5, alpha=0.75)
        ax.scatter(x, sub["median"], color="#0f172a", s=32, zorder=3, label="median")
        if metric == "PRD":
            ax.axhspan(0.98, 1.03, color="#16a34a", alpha=0.12)
            ax.axhline(1.0, color="#16a34a", ls=":", lw=1.1)
        if metric == "PRB":
            ax.axhspan(-0.05, 0.05, color="#16a34a", alpha=0.12)
            ax.axhline(0.0, color="#16a34a", ls=":", lw=1.1)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["run"], rotation=35, ha="right", fontsize=8)
        ax.set_title(metric, fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Neighbor configuration metric ranges across current full runs", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return _png_data_uri(fig)


def _linked_plots(run: NeighborRun) -> str:
    cards = []
    for filename, label in PLOT_ORDER:
        uri = _image_file_data_uri(run.folder / "plots" / filename)
        if not uri:
            continue
        cards.append(
            "<figure class='plot-card'>"
            f"<img src='{uri}' alt='{html.escape(label)}'>"
            f"<figcaption>{html.escape(label)}</figcaption>"
            "</figure>"
        )
    return "\n".join(cards) if cards else "<p class='muted'>No plot PNGs found for this run.</p>"


def _top_tables(all_examples: pd.DataFrame) -> str:
    if all_examples.empty:
        return "<p class='muted'>No best-example rows available.</p>"
    tables = []
    specs = [
        ("accuracy", "Largest R2 gains"),
        ("COD", "Largest COD reductions"),
        ("PRD", "Largest reductions in |PRD - 1|"),
        ("PRB", "Largest reductions in |PRB|"),
    ]
    cols = [
        "run",
        "dataset_label",
        "model",
        "group",
        "experiment",
        "k",
        "R2_gain",
        "COD_reduction",
        "PRD_abs_error_reduction",
        "PRB_abs_error_reduction",
        "R2",
        "COD",
        "PRD",
        "PRB",
    ]
    for obj, title in specs:
        sub = all_examples[all_examples["objective"] == obj].copy()
        tables.append(f"<h3>{html.escape(title)}</h3>")
        tables.append(_html_table(sub[[c for c in cols if c in sub.columns]], max_rows=10))
    return "\n".join(tables)


def build_html(runs: list[NeighborRun]) -> str:
    inventory = run_inventory(runs)
    all_examples = pd.concat([best_examples(run) for run in runs], ignore_index=True)
    all_group = pd.concat([best_group_summary(run) for run in runs], ignore_index=True)
    generalization = pd.concat([generalization_summary(run) for run in runs], ignore_index=True)
    group_plot = fig_group_improvements(all_group)
    ranges_plot = fig_run_metric_ranges(runs)

    group_mean_cols = [
        "run",
        "group",
        "R2_gain",
        "COD_reduction",
        "PRD_abs_error_reduction",
        "PRB_abs_error_reduction",
    ]
    group_mean = (
        all_group.groupby(["run", "group"], as_index=False)
        [["R2_gain", "COD_reduction", "PRD_abs_error_reduction", "PRB_abs_error_reduction"]]
        .mean()
        .sort_values(["run", "R2_gain"], ascending=[True, False])
        if not all_group.empty
        else pd.DataFrame(columns=group_mean_cols)
    )

    run_sections = []
    for run in runs:
        ex = best_examples(run)
        section = f"""
        <section class='run-section' id='{html.escape(run.name)}'>
          <h2>{html.escape(run.name)}</h2>
          <p class='muted'>Embedded figures are generated from existing aggregate spatial-experiment outputs.</p>
          <h3>Best single examples</h3>
          {_html_table(ex, max_rows=8)}
          <h3>Outcome plots</h3>
          <div class='plot-grid'>{_linked_plots(run)}</div>
        </section>
        """
        run_sections.append(section)

    generated_at = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    css = """
    <style>
    :root { --bg:#f8fafc; --fg:#0f172a; --muted:#64748b; --line:#dbe3ef; --card:#ffffff; --accent:#1d4ed8; }
    body { margin:0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color:var(--fg); background:var(--bg); }
    header { padding:28px 36px; background:#0f172a; color:white; }
    header h1 { margin:0 0 8px; font-size:30px; }
    header p { margin:4px 0; color:#dbeafe; max-width:1120px; line-height:1.45; }
    nav { position:sticky; top:0; z-index:2; background:white; border-bottom:1px solid var(--line); padding:10px 36px; }
    nav a { color:var(--accent); margin-right:18px; text-decoration:none; font-weight:600; font-size:14px; }
    main { max-width:1280px; margin:0 auto; padding:24px 28px 48px; }
    section { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:22px; margin:0 0 20px; box-shadow:0 1px 2px rgba(15,23,42,0.04); }
    h2 { margin:0 0 12px; font-size:22px; }
    h3 { margin:22px 0 10px; font-size:16px; }
    p, li { line-height:1.5; }
    .muted { color:var(--muted); }
    .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:12px; }
    .card { border:1px solid var(--line); border-radius:8px; padding:14px; background:#fbfdff; }
    .card b { display:block; font-size:22px; margin-bottom:4px; }
    .table { border-collapse:collapse; width:100%; font-size:13px; }
    .table th, .table td { border-bottom:1px solid var(--line); padding:7px 8px; text-align:right; vertical-align:top; }
    .table th { background:#f1f5f9; color:#334155; position:sticky; top:39px; z-index:1; }
    .table td:first-child, .table th:first-child { text-align:left; }
    .plot-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(430px,1fr)); gap:16px; align-items:start; }
    .plot-card { margin:0; border:1px solid var(--line); border-radius:8px; overflow:hidden; background:white; }
    .plot-card img { display:block; width:100%; height:auto; }
    .plot-card figcaption { padding:8px 10px; font-size:13px; color:#334155; border-top:1px solid var(--line); }
    .figure-wide { width:100%; border:1px solid var(--line); border-radius:8px; background:white; }
    code { background:#e2e8f0; padding:1px 4px; border-radius:4px; }
    </style>
    """
    nav_links = " ".join(
        ["<a href='#overview'>Overview</a>", "<a href='#setup'>Setup</a>", "<a href='#findings'>Findings</a>"]
        + [f"<a href='#{html.escape(run.name)}'>{html.escape(run.name)}</a>" for run in runs]
    )

    total_oos = int(sum(len(run.oos) for run in runs))
    total_train = int(sum(len(run.train) for run in runs))
    total_configs = int(sum(len(_neighbor_rows(run.oos)) for run in runs))

    group_plot_html = f"<img class='figure-wide' src='{group_plot}' alt='Group improvements'>" if group_plot else ""
    ranges_plot_html = f"<img class='figure-wide' src='{ranges_plot}' alt='Metric ranges'>" if ranges_plot else ""

    return f"""<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>Spatial Neighbor Experiment Dashboard</title>
{css}
</head>
<body>
<header>
  <h1>Spatial Neighbor Experiment Dashboard</h1>
  <p>Self-contained review of the current non-smoke spatial neighbor experiments. Generated {html.escape(generated_at)} using aggregate OOS/train metric outputs only.</p>
</header>
<nav>{nav_links}</nav>
<main>
<section id='overview'>
  <h2>Overview</h2>
  <div class='cards'>
    <div class='card'><b>{len(runs)}</b><span>full spatial runs</span></div>
    <div class='card'><b>{total_oos:,}</b><span>OOS metric rows</span></div>
    <div class='card'><b>{total_train:,}</b><span>train metric rows</span></div>
    <div class='card'><b>{total_configs:,}</b><span>neighbor OOS rows evaluated</span></div>
  </div>
  <h3>Run inventory</h3>
  {_html_table(inventory, max_rows=20)}
</section>

<section id='setup'>
  <h2>Theoretical And Experimental Setup</h2>
  <p>The experiment augments standard CCAO mass-appraisal features with leakage-controlled comparable-sale target features. For every query sale, the transformer is fit only on training rows, selects comparable training sales by a composite distance, applies a kernel, and returns weighted log-price neighbor target features. The model then predicts log sale price and metrics are evaluated on the price scale.</p>
  <p>The key leakage controls are: chronological train/test split, fit-only-on-train preprocessing, self-exclusion for training rows, and <code>neighbor_time_rule='past'</code> so neighbor candidates must predate the query sale when dates are used. The OOS slice is common within each dataset filter, so model configurations are compared on the same held-out rows.</p>
  <p>Neighbor families: <code>spatial</code> uses geographic distance with categorical pooling; <code>spatial_nofilter</code> removes categorical pooling; <code>feature</code> adds standardized property-characteristic distance; <code>trend</code> adjusts neighbor targets by a fitted global time trend; <code>time</code> also adds temporal distance to the composite kernel.</p>
  <p class='muted'>Current outcome limitation: row-level predictions are not persisted by <code>spatial_analysis.py</code>, so prediction-dependent ratio-curve plots cannot be reconstructed from these outputs. The dashboard therefore focuses on all available aggregate accuracy and vertical-equity outcomes.</p>
</section>

<section id='findings'>
  <h2>Key Findings</h2>
  <p>The tables below identify the most direct examples of improvement over the no-neighbor baseline. Direction is explicit: <code>R2_gain = neighbor R2 - baseline R2</code>, <code>COD_reduction = baseline COD - neighbor COD</code>, <code>PRD_abs_error_reduction = |baseline PRD - 1| - |neighbor PRD - 1|</code>, and <code>PRB_abs_error_reduction = |baseline PRB| - |neighbor PRB|</code>. Positive values are better in all four improvement columns.</p>
  {_top_tables(all_examples)}
  <h3>Average best-by-group changes</h3>
  {_html_table(group_mean, max_rows=80)}
  <h3>Generalization check</h3>
  <p class='muted'>Train-vs-OOS gaps are shown for the no-neighbor baseline and the OOS-best neighbor configuration for each dataset/model. Large positive R2 gaps indicate stronger in-sample fit than held-out performance.</p>
  {_html_table(generalization, max_rows=40)}
  <h3>Summary figures</h3>
  {group_plot_html}
  {ranges_plot_html}
</section>

{''.join(run_sections)}
</main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a self-contained dashboard for spatial neighbor experiments.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=OUT_HTML)
    parser.add_argument("--runs", nargs="*", default=None, help="Optional run folder names, e.g. spatial_20260629_002858.")
    args = parser.parse_args()

    runs = load_runs(args.root, args.runs)
    if not runs:
        raise SystemExit(f"No spatial runs with combined_oos_metrics.csv found under {args.root}")

    html_text = build_html(runs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html_text, encoding="utf-8")
    size_mb = args.out.stat().st_size / (1024 * 1024)
    print(f"[neighbor-dashboard] wrote {args.out} ({size_mb:.1f} MB), runs={len(runs)}")


if __name__ == "__main__":
    main()
