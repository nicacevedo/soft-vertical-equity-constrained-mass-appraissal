"""Reproduce the paper's leakage-free CCAO-protocol baseline comparison.

The implementation deliberately reuses the project pipeline:

* predictors and categorical columns from ``params.yaml``;
* linear preprocessing from ``preprocessing.recipes_pipelined``;
* LightGBM parameters from ``model_params.yaml`` via ``run_temporal_cv``;
* expanding-window rolling-origin CV from ``utils.motivation_utils``; and
* assessor metrics from ``utils.motivation_utils``.

Model selection uses rolling-origin CV on the oldest 90% of eligible 2016--2024
sales. The held-out test set is the newest 10% of that universe. Final
production models are then fitted on 100% of 2016--2024 sales and evaluated on
the 2025 assessment year.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Allow the documented ``python scripts/run_paper_baseline.py`` invocation from
# the repository root to import the existing project modules.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocessing.recipes_pipelined import build_model_pipeline
from run_temporal_cv import _build_lgbm_params_from_files
from utils.motivation_utils import (
    build_rolling_origin_protocol,
    cod,
    cov_iaao,
    mki,
    prb,
    prd,
    run_robust_rolling_origin_cv,
    split_ccao_assessment_universe,
    vei,
)


MODEL_LABELS = {
    "LinearRegression": "Linear regression",
    "LGBMRegressor": "Unpenalized LightGBM",
}
MODEL_COLORS = {
    "LinearRegression": "#0072B2",
    "LGBMRegressor": "#D55E00",
}
YEAR_COLORS = {2024: "#0072B2", 2025: "#D55E00"}
EVAL_SPLITS = ("test", "assessment")
SPLIT_LABELS = {
    "test": "Test",
    "assessment": "2025",
}
METRIC_COLUMNS = [
    "r2_price",
    "mae_price",
    "mape_price_pct",
    "rmse_log_price",
    "median_ratio",
    "mean_ratio",
    "weighted_mean_ratio",
    "cod_pct",
    "cov_pct",
    "prd",
    "prb",
    "mki",
    "vei_pct",
    "beta_log",
    "distance_correlation_log_residual_log_price",
]


def _log(message: str) -> None:
    print(f"[paper-baseline] {message}", flush=True)


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_value(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _training_file_inventory(data_path: Path) -> List[Dict[str, Any]]:
    import pyarrow.parquet as pq

    rows: List[Dict[str, Any]] = []
    # Compare all locally available CCAO vintages and 2025 variants.
    for path in sorted(data_path.parent.parent.glob("*/training_data*.parquet")):
        metadata = pq.ParquetFile(path).metadata
        rows.append(
            {
                "path": str(path),
                "bytes": int(path.stat().st_size),
                "mtime_ns": int(path.stat().st_mtime_ns),
                "parquet_rows": int(metadata.num_rows),
                "parquet_columns": int(metadata.num_columns),
            }
        )
    return rows


def _load_data(
    data_path: Path,
    predictor_cols: List[str],
    target_col: str,
    date_col: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    filter_cols = ["ind_pin_is_multicard", "sv_is_outlier"]
    required = list(dict.fromkeys(predictor_cols + [target_col, date_col] + filter_cols))
    _log(f"loading {data_path} ({len(required)} selected columns)")
    df = pd.read_parquet(data_path, columns=required, engine="pyarrow")
    raw_rows = int(len(df))
    df.insert(0, "source_row", np.arange(raw_rows, dtype=np.int64))
    df[date_col] = pd.to_datetime(df[date_col])
    # ``TypeConverter`` in the established linear recipe explicitly performs
    # this conversion.  The newest parquet stores the Boolean field as object,
    # which LightGBM 4.6 rejects unless the same conversion is applied before
    # its raw-feature path.
    if "char_recent_renovation" in df.columns:
        df["char_recent_renovation"] = pd.to_numeric(
            df["char_recent_renovation"], errors="coerce"
        )
    raw_year_counts = {
        str(int(k)): int(v)
        for k, v in df.groupby(df[date_col].dt.year, dropna=False).size().items()
        if pd.notna(k)
    }

    keep = (
        ~df["ind_pin_is_multicard"].astype("bool").fillna(True)
        & ~df["sv_is_outlier"].astype("bool").fillna(True)
    )
    df = df.loc[keep].drop(columns=filter_cols)
    df = df.sort_values([date_col, "source_row"], kind="mergesort").reset_index(drop=True)
    if df[target_col].isna().any() or not np.all(df[target_col].to_numpy(dtype=float) > 0.0):
        raise ValueError("The filtered target must be complete and strictly positive.")

    filtered_year_counts = {
        str(int(k)): int(v)
        for k, v in df.groupby(df[date_col].dt.year, dropna=False).size().items()
        if pd.notna(k)
    }
    audit = {
        "raw_rows": raw_rows,
        "filtered_rows": int(len(df)),
        "removed_multicard_or_outlier_rows": int(raw_rows - len(df)),
        "raw_year_counts": raw_year_counts,
        "filtered_year_counts": filtered_year_counts,
        "filtered_min_date": df[date_col].min(),
        "filtered_max_date": df[date_col].max(),
        "dtype_compatibility_conversion": {
            "char_recent_renovation": "object/Boolean to numeric, matching TypeConverter"
        },
    }
    return df, audit


def _categorical_frames(
    frames: Iterable[pd.DataFrame], categorical_cols: List[str]
) -> List[pd.DataFrame]:
    converted: List[pd.DataFrame] = []
    for frame in frames:
        out = frame.copy()
        for col in categorical_cols:
            if col in out.columns:
                out[col] = out[col].astype("category")
        converted.append(out)
    return converted


def _distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    # dcor's exact one-dimensional mergesort algorithm is O(n log n).  A
    # writable Numba cache avoids package-cache failures on the cluster.
    os.environ.setdefault(
        "NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "paper_baseline_numba_cache")
    )
    import dcor

    return float(dcor.distance_correlation(x, y, method="mergesort"))


def _beta_log(y_log: np.ndarray, pred_log: np.ndarray) -> float:
    y = np.asarray(y_log, dtype=float)
    residual = np.asarray(pred_log, dtype=float) - y
    var_y = float(np.var(y, ddof=0))
    if not np.isfinite(var_y) or var_y <= 0.0:
        raise ValueError("beta_log is undefined when log-price variance is not positive.")
    return float(np.cov(residual, y, ddof=0)[0, 1] / var_y)


def _paper_metrics(y_log: np.ndarray, pred_log: np.ndarray) -> Dict[str, float]:
    actual = np.exp(np.asarray(y_log, dtype=float))
    predicted = np.exp(np.asarray(pred_log, dtype=float))
    ratio = predicted / actual
    median_ratio = float(np.median(ratio))
    metrics = {
        "r2_price": float(r2_score(actual, predicted)),
        "mae_price": float(np.mean(np.abs(predicted - actual))),
        "mape_price_pct": float(100.0 * np.mean(np.abs(predicted - actual) / actual)),
        "rmse_log_price": float(np.sqrt(np.mean((pred_log - y_log) ** 2))),
        "median_ratio": median_ratio,
        "mean_ratio": float(np.mean(ratio)),
        "weighted_mean_ratio": float(np.sum(predicted) / np.sum(actual)),
        "cod_pct": float(cod(ratio, na_rm=True)),
        "cov_pct": float(100.0 * cov_iaao(predicted, actual, na_rm=True)),
        "prd": float(prd(predicted, actual, na_rm=True)),
        "prb": float(prb(predicted, actual, na_rm=True)),
        "mki": float(mki(predicted, actual, na_rm=True)),
        "vei_pct": float(vei(predicted, actual, na_rm=True)),
        "beta_log": _beta_log(
            np.asarray(y_log, dtype=float), np.asarray(pred_log, dtype=float)
        ),
        "distance_correlation_log_residual_log_price": _distance_correlation(
            np.asarray(pred_log - y_log, dtype=float), np.asarray(y_log, dtype=float)
        ),
    }
    if not all(np.isfinite(metrics[name]) for name in METRIC_COLUMNS):
        raise ValueError(f"Non-finite paper metric encountered: {metrics}")
    if not np.isclose(metrics["prd"], metrics["mean_ratio"] / metrics["weighted_mean_ratio"]):
        raise AssertionError("PRD identity check failed.")
    return metrics


def _bootstrap_median_interval(
    values: np.ndarray, *, rng: np.random.Generator, replicates: int
) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    medians = np.empty(replicates, dtype=float)
    batch = 100
    for start in range(0, replicates, batch):
        stop = min(start + batch, replicates)
        sampled = rng.choice(values, size=(stop - start, values.size), replace=True)
        medians[start:stop] = np.median(sampled, axis=1)
    low, high = np.quantile(medians, [0.025, 0.975])
    return float(low), float(high)


def _decile_profile(
    predictions: pd.DataFrame, *, seed: int, bootstrap_replicates: int
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (split, model), group in predictions.groupby(["split", "model"], sort=True):
        ratio = group["assessment_ratio"].to_numpy(dtype=float)
        predicted = group["predicted_price"].to_numpy(dtype=float)
        actual = group["sale_price"].to_numpy(dtype=float)
        overall_median = float(np.median(ratio))
        proxy = 0.5 * (actual + predicted / overall_median)
        order = np.argsort(proxy, kind="mergesort")
        chunks = np.array_split(order, 10)
        model_offset = 0 if model == "LinearRegression" else 1000
        split_offset = 0 if split == "test" else 20000
        for decile, idx in enumerate(chunks, start=1):
            values = ratio[idx]
            rng = np.random.default_rng(int(seed + split_offset + model_offset + decile))
            ci_low, ci_high = _bootstrap_median_interval(
                values, rng=rng, replicates=bootstrap_replicates
            )
            rows.append(
                {
                    "split": str(split),
                    "year": int(group["year"].iloc[0]),
                    "model": str(model),
                    "decile": int(decile),
                    "n": int(values.size),
                    "proxy_min": float(np.min(proxy[idx])),
                    "proxy_max": float(np.max(proxy[idx])),
                    "median_ratio": float(np.median(values)),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "overall_median_ratio": overall_median,
                }
            )
    return pd.DataFrame(rows)


def _sale_price_profile(predictions: pd.DataFrame, bins: int = 30) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (split, model), group in predictions.groupby(["split", "model"], sort=True):
        ordered = group.sort_values(["sale_price", "source_row"], kind="mergesort")
        for bin_id, idx in enumerate(np.array_split(np.arange(len(ordered)), bins), start=1):
            part = ordered.iloc[idx]
            ratios = part["assessment_ratio"].to_numpy(dtype=float)
            rows.append(
                {
                    "split": str(split),
                    "year": int(group["year"].iloc[0]),
                    "model": str(model),
                    "bin": int(bin_id),
                    "n": int(len(part)),
                    "median_sale_price": float(part["sale_price"].median()),
                    "ratio_q25": float(np.quantile(ratios, 0.25)),
                    "median_ratio": float(np.median(ratios)),
                    "ratio_q75": float(np.quantile(ratios, 0.75)),
                }
            )
    return pd.DataFrame(rows)


# def _plot_motivation(
#     predictions: pd.DataFrame, price_profile: pd.DataFrame, out_path: Path, seed: int
# ) -> None:
#     plt.rcParams.update(
#         {
#             "font.size": 9,
#             "axes.titlesize": 10.5,
#             "axes.labelsize": 9.5,
#             "legend.fontsize": 8.5,
#             "pdf.fonttype": 42,
#         }
#     )
#     fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.15), sharex=True, sharey=True)
#     for row, year in enumerate((2024, 2025)):
#         for col, model in enumerate(("LinearRegression", "LGBMRegressor")):
#             ax = axes[row, col]
#             subset = predictions.loc[
#                 (predictions["year"] == year) & (predictions["model"] == model)
#             ]
#             sample_n = min(2500, len(subset))
#             sample = subset.sample(
#                 n=sample_n, random_state=int(seed + year + col), replace=False
#             )
#             ax.scatter(
#                 sample["sale_price"],
#                 sample["assessment_ratio"],
#                 s=4,
#                 color="#6B7280",
#                 alpha=0.10,
#                 linewidths=0,
#                 clip_on=True,
#             )
#             prof = price_profile.loc[
#                 (price_profile["year"] == year) & (price_profile["model"] == model)
#             ]
#             color = MODEL_COLORS[model]
#             ax.fill_between(
#                 prof["median_sale_price"],
#                 prof["ratio_q25"],
#                 prof["ratio_q75"],
#                 color=color,
#                 alpha=0.16,
#                 linewidth=0,
#             )
#             ax.plot(
#                 prof["median_sale_price"],
#                 prof["median_ratio"],
#                 color=color,
#                 marker="o",
#                 markersize=2.5,
#                 linewidth=1.5,
#             )
#             ax.axhline(1.0, color="#111827", linestyle=(0, (2, 2)), linewidth=0.9)
#             ax.set_xscale("log", base=10)
#             ax.set_ylim(0.55, 1.45)
#             ax.grid(True, color="#E5E7EB", linewidth=0.55)
#             ax.set_axisbelow(True)
#             if row == 0:
#                 ax.set_title(MODEL_LABELS[model])
#             if col == 0:
#                 ax.set_ylabel(f"{year}\nAssessment ratio")
#             if row == 1:
#                 ax.set_xlabel("Sale price (log$_{10}$ scale)")
#     handles = [
#         Line2D([0], [0], color="#6B7280", marker="o", linestyle="None", markersize=3,
#                alpha=0.45, label="Sampled sales"),
#         Line2D([0], [0], color="#111827", marker="o", linewidth=1.5,
#                markersize=3, label="Equal-count-bin median (IQR shaded)"),
#         Line2D([0], [0], color="#111827", linestyle=(0, (2, 2)), linewidth=0.9,
#                label="Ratio = 1"),
#     ]
#     fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
#                bbox_to_anchor=(0.5, 1.005))
#     fig.tight_layout(rect=(0, 0, 1, 0.94))
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     fig.savefig(out_path, format="pdf", bbox_inches="tight")
#     plt.close(fig)


from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, LogFormatterSciNotation
import numpy as np


# from matplotlib.lines import Line2D
# from matplotlib.ticker import LogLocator, LogFormatterSciNotation
# import numpy as np


def _plot_motivation(
    predictions: pd.DataFrame, price_profile: pd.DataFrame, out_path: Path, seed: int
) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.15), sharex=True, sharey=True)

    # Shared x-range based on the plotted profiles, with a small log-space margin.
    x_min = price_profile["median_sale_price"].min()
    x_max = price_profile["median_sale_price"].max()
    log_pad = 0.04 * (np.log10(x_max) - np.log10(x_min))
    x_lim = (
        10 ** (np.log10(x_min) - log_pad),
        10 ** (np.log10(x_max) + log_pad),
    )

    for row, split in enumerate(EVAL_SPLITS):
        for col, model in enumerate(("LinearRegression", "LGBMRegressor")):
            ax = axes[row, col]

            subset = predictions.loc[
                (predictions["split"] == split) & (predictions["model"] == model)
            ]

            sample_n = min(2500, len(subset))
            sample = subset.sample(
                n=sample_n,
                random_state=int(seed + row + col),
                replace=False,
            )

            # # White-filled circles with black borders.
            # ax.scatter(
            #     sample["sale_price"],
            #     sample["assessment_ratio"],
            #     s=5,
            #     facecolors="white",
            #     edgecolors="#111827",
            #     alpha=0.20,
            #     linewidths=0.35,
            #     clip_on=True,
            # )

            prof = price_profile.loc[
                (price_profile["split"] == split) & (price_profile["model"] == model)
            ]

            color = MODEL_COLORS[model]

            ax.fill_between(
                prof["median_sale_price"],
                prof["ratio_q25"],
                prof["ratio_q75"],
                color=color,
                alpha=0.16,
                linewidth=0,
            )

            ax.plot(
                prof["median_sale_price"],
                prof["median_ratio"],
                color=color,
                marker="o",
                markersize=2.5,
                linewidth=1.5,
            )

            ax.axhline(
                1.0,
                color="#111827",
                linestyle=(0, (2, 2)),
                linewidth=0.9,
            )

            ax.set_xscale("log", base=10)
            ax.set_xlim(*x_lim)
            ax.set_ylim(0.55, 1.45)

            # Show and label 1, 2, and 5 × 10^n ticks.
            ax.xaxis.set_major_locator(
                LogLocator(base=10, subs=(1.0, 2.0, 5.0))
            )
            ax.xaxis.set_major_formatter(
                LogFormatterSciNotation(
                    base=10,
                    labelOnlyBase=False,
                    minor_thresholds=(np.inf, np.inf),
                )
            )

            # Slightly stronger grid for clearer log-scale cells.
            ax.grid(True, color="#E5E7EB", linewidth=0.70)
            ax.set_axisbelow(True)

            # Full-data statistics: assessment ratio vs log10(sale price), plus beta_log.
            valid = subset.loc[
                (subset["sale_price"] > 0)
                & subset["sale_price"].notna()
                & subset["assessment_ratio"].notna(),
                ["sale_price", "assessment_ratio"],
            ]
            sale = valid["sale_price"].to_numpy(dtype=float)
            ratio = valid["assessment_ratio"].to_numpy(dtype=float)
            finite = (
                np.isfinite(sale)
                & (sale > 0)
                & np.isfinite(ratio)
                & (ratio > 0)
            )
            sale = sale[finite]
            ratio = ratio[finite]
            slope = np.polyfit(np.log10(sale), ratio, 1)[0]
            beta_log = _beta_log(np.log(sale), np.log(sale * ratio))

            ax.legend(
                handles=[
                    Line2D(
                        [],
                        [],
                        linestyle="None",
                        label=rf"Slope = {slope:.3f}   $\beta_{{\log}}$ = {beta_log:.3f}",
                    )
                ],
                loc="lower left",
                frameon=False,
                handlelength=0,
                handletextpad=0,
                fontsize=7.5,
            )

            if row == 0:
                ax.set_title(MODEL_LABELS[model])
            if col == 0:
                ax.set_ylabel(f"{SPLIT_LABELS[split]}\nAssessment ratio")
            if row == 1:
                ax.set_xlabel("Sale price (log$_{10}$ scale)")

    handles = [
        # Line2D(
        #     [0],
        #     [0],
        #     marker="o",
        #     linestyle="None",
        #     markersize=3.5,
        #     markerfacecolor="white",
        #     markeredgecolor="#111827",
        #     markeredgewidth=0.6,
        #     label="Sampled sales",
        # ),
        Line2D(
            [0],
            [0],
            color="#111827",
            marker="o",
            linewidth=1.5,
            markersize=3,
            label="Equal-count-bin median (IQR shaded)",
        ),
        Line2D(
            [0],
            [0],
            color="#111827",
            linestyle=(0, (2, 2)),
            linewidth=0.9,
            label="Ratio = 1",
        ),
    ]

    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.005),
    )

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

def _plot_deciles(
    profile: pd.DataFrame, predictions: pd.DataFrame, out_path: Path
) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.15), sharex=True, sharey=True)
    panel_boxes: Dict[Tuple[str, str], List[np.ndarray]] = {}
    for split in EVAL_SPLITS:
        for model in ("LinearRegression", "LGBMRegressor"):
            group = predictions.loc[
                (predictions["split"] == split) & (predictions["model"] == model)
            ]
            ratio = group["assessment_ratio"].to_numpy(dtype=float)
            predicted = group["predicted_price"].to_numpy(dtype=float)
            actual = group["sale_price"].to_numpy(dtype=float)
            overall_median = float(np.median(ratio))
            proxy = 0.5 * (actual + predicted / overall_median)
            order = np.argsort(proxy, kind="mergesort")
            panel_boxes[(split, model)] = [ratio[idx] for idx in np.array_split(order, 10)]
    y_limits = (0.0, 2.0)
    for row, split in enumerate(EVAL_SPLITS):
        for col, model in enumerate(("LinearRegression", "LGBMRegressor")):
            ax = axes[row, col]
            part = profile.loc[(profile["split"] == split) & (profile["model"] == model)]
            color = MODEL_COLORS[model]
            ax.axhspan(0.9, 1.1, color="#16A34A", alpha=0.16, linewidth=0, zorder=0)
            ax.boxplot(
                panel_boxes[(split, model)],
                positions=range(1, 11),
                widths=0.62,
                patch_artist=True,
                showfliers=False,
                boxprops={"facecolor": color, "edgecolor": color, "alpha": 0.28, "linewidth": 0.8},
                medianprops={"color": color, "linewidth": 1.2},
                whiskerprops={"color": color, "linewidth": 0.8},
                capprops={"color": color, "linewidth": 0.8},
                zorder=2,
            )
            yerr = np.vstack(
                [
                    part["median_ratio"] - part["ci95_low"],
                    part["ci95_high"] - part["median_ratio"],
                ]
            )
            ax.errorbar(
                part["decile"],
                part["median_ratio"],
                yerr=yerr,
                color=color,
                marker="o",
                markersize=3.5,
                linewidth=0,
                elinewidth=0.8,
                capsize=2,
                zorder=3,
            )
            ax.axhline(1.0, color="#111827", linestyle=":", linewidth=0.9)
            ax.axhline(
                float(part["overall_median_ratio"].iloc[0]),
                color=color,
                linestyle="--",
                linewidth=1.0,
            )
            ax.set_xticks(range(1, 11))
            ax.set_ylim(*y_limits)
            ax.grid(True, axis="y", color="#E5E7EB", linewidth=0.55)
            ax.set_axisbelow(True)
            if row == 0:
                ax.set_title(MODEL_LABELS[model])
            if col == 0:
                ax.set_ylabel(f"{SPLIT_LABELS[split]}\nAssessment ratio")
            if row == 1:
                ax.set_xlabel("Sale-price decile (low to high)")
    handles = [
        Patch(facecolor="#9CA3AF", edgecolor="#4B5563", alpha=0.35,
              label="Decile ratio distribution"),
        Line2D([0], [0], color="#111827", marker="o", linewidth=0, markersize=3.5,
               label="Decile median (95% bootstrap CI)"),
        Line2D([0], [0], color="#111827", linestyle="--", linewidth=1.0,
               label="Overall median"),
        Line2D([0], [0], color="#111827", linestyle=":", linewidth=0.9,
               label="Ratio = 1"),
        Patch(facecolor="#16A34A", alpha=0.16, edgecolor="none",
              label=r"$\pm$10% ratio band"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def _write_table_snippet(metrics: pd.DataFrame, out_path: Path) -> None:
    lookup = metrics.set_index(["split", "model"])

    def value(split: str, model: str, metric: str, fmt: str) -> str:
        return format(float(lookup.loc[(split, model), metric]), fmt)

    specs = [
        ("$R^2_P$", "r2_price", ".3f"),
        ("$\\operatorname{MAE}_P$", "mae_price", ",.0f"),
        ("$\\operatorname{MAPE}_P$", "mape_price_pct", ".1f"),
        ("$\\operatorname{RMSE}_{\\log P}$", "rmse_log_price", ".3f"),
        ("Median ratio $m_S$", "median_ratio", ".3f"),
        ("Mean ratio $\\bar r_S$", "mean_ratio", ".3f"),
        ("Weighted mean $\\bar r_{W,S}$", "weighted_mean_ratio", ".3f"),
        ("COD", "cod_pct", ".1f"),
        ("COV", "cov_pct", ".1f"),
        ("PRD", "prd", ".3f"),
        ("PRB", "prb", ".3f"),
        ("MKI", "mki", ".3f"),
        ("VEI", "vei_pct", ".1f"),
        (r"$\beta_{\log}$", "beta_log", ".3f"),
        ("$\\operatorname{dCor}(e,y)$", "distance_correlation_log_residual_log_price", ".3f"),
    ]
    lines = []
    for label, metric, fmt in specs:
        vals = [value(split, model, metric, fmt) for split in EVAL_SPLITS
                for model in ("LinearRegression", "LGBMRegressor")]
        if metric == "mae_price":
            vals = ["\\$" + v for v in vals]
        if metric in {"mape_price_pct", "cod_pct", "cov_pct", "vei_pct"}:
            vals = [v + "\\%" for v in vals]
        lines.append(f"{label} & " + " & ".join(vals) + r" \\")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_cv(
    development: pd.DataFrame,
    *,
    predictor_cols: List[str],
    categorical_cols: List[str],
    target_col: str,
    date_col: str,
    params: Dict[str, Any],
    lgbm_params: Dict[str, Any],
    data_signature: Dict[str, Any],
    result_root: Path,
    split_protocol: Dict[str, Any],
    max_workers: int,
) -> Dict[str, Any]:
    linear_pipeline_builder = lambda: build_model_pipeline(
        pred_vars=predictor_cols,
        cat_vars=categorical_cols,
        id_vars=params["model"]["predictor"]["id"],
    )
    model_specs = [
        {
            "name": "LinearRegression",
            "config": {},
            "requires_linear_pipeline": True,
            "factory": lambda: LinearRegression(fit_intercept=True),
        },
        {
            "name": "LGBMRegressor",
            "config": {"lgbm_params": dict(lgbm_params)},
            "requires_linear_pipeline": False,
            "factory": lambda: lgb.LGBMRegressor(**dict(lgbm_params)),
        },
    ]
    return run_robust_rolling_origin_cv(
        df_train_validate=development,
        date_col=date_col,
        target_col=target_col,
        predictor_cols=predictor_cols,
        categorical_cols=categorical_cols,
        model_specs=model_specs,
        linear_pipeline_builder=linear_pipeline_builder,
        result_root=str(result_root),
        data_signature=data_signature,
        split_protocol=split_protocol,
        bootstrap_protocol={"n_bootstrap": 0, "bootstrap_block_freq": "M", "rng_seed": 2025},
        fairness_ratio_mode="diff",
        predict_store=True,
        parquet_engine="pyarrow",
        log_progress=True,
        parallel_enabled=max_workers > 1,
        parallel_cpu_fraction=1.0,
        parallel_max_workers=max_workers,
        parallel_backend="loky",
        numeric_sanity_abs_cap=1e10,
    )


def run(args: argparse.Namespace) -> Dict[str, Any]:
    data_path = Path(args.data_path)
    out_dir = Path(args.out_dir)
    figures_dir = Path(args.figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with Path(args.params).open("r", encoding="utf-8") as handle:
        params = yaml.safe_load(handle)
    with Path(args.model_params).open("r", encoding="utf-8") as handle:
        model_params = yaml.safe_load(handle)
    predictor_cols = list(params["model"]["predictor"]["all"])
    categorical_cols = list(params["model"]["predictor"]["categorical"])
    target_col = "meta_sale_price"
    date_col = "meta_sale_date"

    inventory = _training_file_inventory(data_path)
    newest = max(inventory, key=lambda row: row["mtime_ns"])
    if Path(newest["path"]).resolve() != data_path.resolve():
        raise ValueError(f"Requested data file is not the newest training_data source: {newest}")
    data_sha256 = _sha256(data_path)
    df, data_audit = _load_data(data_path, predictor_cols, target_col, date_col)
    with Path(args.cv_config).open("r", encoding="utf-8") as handle:
        cv_config = yaml.safe_load(handle)
    splits = split_ccao_assessment_universe(
        df,
        date_col,
        split_prop=float(params["cv"]["split_prop"]),
        universe_start=str(cv_config.get("universe_start", "2016-01-01")),
        pre_assessment_end=str(cv_config.get("pre_assessment_end", "2024-12-31")),
        assessment_year=int(cv_config.get("assessment_year", 2025)),
        sort_cols=[date_col, "source_row"],
    )
    development = splits["development"]
    test = splits["test"]
    production = splits["production"]
    assessment = splits["assessment"]
    _log(
        "applied CCAO splits: "
        f"development={len(development)} test={len(test)} "
        f"production={len(production)} assessment={len(assessment)}"
    )

    lgbm_params = _build_lgbm_params_from_files(
        model_params=model_params,
        ccao_params=params,
        seed=int(args.seed),
        use_ccao_fallback=False,
    )
    if args.lgbm_n_jobs is not None:
        lgbm_params["n_jobs"] = int(args.lgbm_n_jobs)

    split_protocol = dict(cv_config["split_protocol"])
    folds = build_rolling_origin_protocol(development, date_col, **split_protocol)
    fold_rows = [
        {k: fold[k] for k in (
            "fold_id", "train_start", "train_end", "val_start", "val_end",
            "train_size", "val_size", "train_index_hash", "val_index_hash"
        )}
        for fold in folds
    ]
    pd.DataFrame(fold_rows).to_csv(out_dir / "rolling_origin_folds.csv", index=False)
    if not fold_rows:
        raise ValueError("Rolling-origin protocol produced no folds on the development pool.")
    if pd.Timestamp(max(row["val_end"] for row in fold_rows)) > pd.Timestamp(development[date_col].max()):
        raise AssertionError("Rolling-origin validation used sales outside the development pool.")
    if pd.Timestamp(test[date_col].iloc[0]) < pd.Timestamp(development[date_col].iloc[-1]):
        raise AssertionError("Test split is not later than the development pool.")
    if int(assessment[date_col].dt.year.min()) < 2025:
        raise AssertionError("Assessment split entered pre-2025 sales.")

    data_signature = {
        "data_path": str(data_path),
        "data_sha256": data_sha256,
        "target_col": target_col,
        "date_col": date_col,
        "predictor_cols": predictor_cols,
        "categorical_cols": categorical_cols,
        "filters": {"drop_multicard": True, "drop_outliers": True},
        "universe_start": str(cv_config.get("universe_start", "2016-01-01")),
        "pre_assessment_end": str(cv_config.get("pre_assessment_end", "2024-12-31")),
        "split_prop": float(params["cv"]["split_prop"]),
        "assessment_year": 2025,
        "sample_frac": None,
    }
    cv_summary: Dict[str, Any]
    if args.skip_cv:
        cv_summary = {"skipped": True, "fold_count": len(folds)}
    else:
        _log(f"running {len(folds)} existing rolling-origin folds for both baselines")
        cv_out = _run_cv(
            development,
            predictor_cols=predictor_cols,
            categorical_cols=categorical_cols,
            target_col=target_col,
            date_col=date_col,
            params=params,
            lgbm_params=lgbm_params,
            data_signature=data_signature,
            result_root=out_dir / "cv",
            split_protocol=split_protocol,
            max_workers=int(args.cv_workers),
        )
        cv_summary = {
            "skipped": False,
            "data_id": cv_out["data_id"],
            "split_id": cv_out["split_id"],
            "fold_count": int(cv_out["fold_count"]),
            "flagged_config_ids": list(cv_out.get("flagged_config_ids", [])),
        }

    prediction_rows: List[pd.DataFrame] = []
    metric_rows: List[Dict[str, Any]] = []
    stages = [
        {
            "split": "test",
            "panel_year": int(test[date_col].dt.year.max()),
            "train": development,
            "eval": test,
            "message": "fitting on development (oldest 90% of 2016-2024) for test evaluation",
        },
        {
            "split": "assessment",
            "panel_year": 2025,
            "train": production,
            "eval": assessment,
            "message": "fitting production models on 100% of eligible 2016-2024 sales for 2025",
        },
    ]
    for stage in stages:
        train_df = stage["train"]
        eval_df = stage["eval"]
        _log(str(stage["message"]))
        X_train = train_df[predictor_cols].copy()
        X_eval = eval_df[predictor_cols].copy()
        y_train_log = np.log(train_df[target_col].to_numpy(dtype=float))
        y_eval_log = np.log(eval_df[target_col].to_numpy(dtype=float))
        linear_pipeline = build_model_pipeline(
            pred_vars=predictor_cols,
            cat_vars=categorical_cols,
            id_vars=params["model"]["predictor"]["id"],
        )
        X_train_linear = linear_pipeline.fit_transform(X_train, y_train_log)
        linear_model = LinearRegression(fit_intercept=True).fit(X_train_linear, y_train_log)
        X_train_lgbm, X_eval_lgbm = _categorical_frames(
            [X_train, X_eval], categorical_cols
        )
        lgbm_model = lgb.LGBMRegressor(**lgbm_params).fit(X_train_lgbm, y_train_log)
        linear_pred = np.asarray(
            linear_model.predict(linear_pipeline.transform(X_eval)), dtype=float
        )
        lgbm_pred = np.asarray(lgbm_model.predict(X_eval_lgbm), dtype=float)
        for model, pred_log in (
            ("LinearRegression", linear_pred), ("LGBMRegressor", lgbm_pred)
        ):
            _log(f"computing paper metrics: {stage['split']} / {MODEL_LABELS[model]}")
            metric_rows.append(
                {
                    "year": int(stage["panel_year"]),
                    "split": stage["split"],
                    "train_rows": int(len(train_df)),
                    "model": model,
                    "n": int(len(eval_df)),
                    **_paper_metrics(y_eval_log, pred_log),
                }
            )
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "source_row": eval_df["source_row"].to_numpy(dtype=np.int64),
                        "sale_date": eval_df[date_col].to_numpy(),
                        "year": int(stage["panel_year"]),
                        "split": stage["split"],
                        "model": model,
                        "sale_price": eval_df[target_col].to_numpy(dtype=float),
                        "actual_log_price": y_eval_log,
                        "predicted_log_price": pred_log,
                        "predicted_price": np.exp(pred_log),
                        "log_residual": pred_log - y_eval_log,
                        "assessment_ratio": np.exp(pred_log - y_eval_log),
                    }
                )
            )

    metrics = pd.DataFrame(metric_rows).sort_values(["split", "model"]).reset_index(drop=True)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    metrics.to_csv(out_dir / "table2_metrics_unrounded.csv", index=False, float_format="%.17g")
    _write_json(
        out_dir / "table2_metrics_unrounded.json",
        {"metrics": metrics.to_dict(orient="records")},
    )
    predictions.to_parquet(out_dir / "baseline_predictions.parquet", index=False, engine="pyarrow")

    _log("computing market-value-proxy deciles and confidence intervals")
    deciles = _decile_profile(
        predictions, seed=int(args.seed), bootstrap_replicates=int(args.bootstrap_replicates)
    )
    price_profile = _sale_price_profile(predictions)
    deciles.to_csv(out_dir / "decile_ratio_profiles_unrounded.csv", index=False, float_format="%.17g")
    price_profile.to_csv(out_dir / "sale_price_profiles_unrounded.csv", index=False, float_format="%.17g")
    _write_table_snippet(metrics, out_dir / "table2_rows.tex")

    motivation_path = figures_dir / "baseline_models_motivation_2024_2025.pdf"
    decile_path = figures_dir / "baseline_models_decile_ratios_2024_2025.pdf"
    _plot_motivation(predictions, price_profile, motivation_path, int(args.seed))
    _plot_deciles(deciles, predictions, decile_path)

    manifest = {
        "dataset": {
            "path": str(data_path),
            "sha256": data_sha256,
            "bytes": int(data_path.stat().st_size),
            "inventory": inventory,
            "newest_training_file_by_mtime": newest,
            **data_audit,
        },
        "design": {
            "universe_start": str(cv_config.get("universe_start", "2016-01-01")),
            "pre_assessment_end": str(cv_config.get("pre_assessment_end", "2024-12-31")),
            "split_prop": float(params["cv"]["split_prop"]),
            "development_years": sorted(development[date_col].dt.year.unique().astype(int).tolist()),
            "development_rows": int(len(development)),
            "development_min_date": development[date_col].min(),
            "development_max_date": development[date_col].max(),
            "test_rows": int(len(test)),
            "test_min_date": test[date_col].min(),
            "test_max_date": test[date_col].max(),
            "production_rows": int(len(production)),
            "assessment_year": 2025,
            "assessment_rows": int(len(assessment)),
            "test_fit_on_development_only": True,
            "assessment_fit_on_full_pre_assessment_universe": True,
            "evaluation_years_used_in_cv_or_selection": [],
            "target": target_col,
            "training_scale": "natural log sale price",
            "price_retransformation": "direct exponentiation",
            "predictor_count": len(predictor_cols),
            "categorical_predictor_count": len(categorical_cols),
            "filters": data_signature["filters"],
            "split_protocol": split_protocol,
            "rolling_origin_folds": fold_rows,
        },
        "models": {
            "LinearRegression": {"fit_intercept": True, "preprocessing": "build_model_pipeline"},
            "LGBMRegressor": lgbm_params,
        },
        "cv": cv_summary,
        "metric_units": {
            "mape_price_pct": "percent",
            "cod_pct": "percent",
            "cov_pct": "percent",
            "vei_pct": "percent",
            "mae_price": "dollars",
            "rmse_log_price": "natural-log dollars",
        },
        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lgb.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "outputs": {
            "metrics_csv": str(out_dir / "table2_metrics_unrounded.csv"),
            "predictions_parquet": str(out_dir / "baseline_predictions.parquet"),
            "decile_profiles_csv": str(out_dir / "decile_ratio_profiles_unrounded.csv"),
            "motivation_figure": str(motivation_path),
            "decile_figure": str(decile_path),
        },
    }
    _write_json(out_dir / "experiment_manifest.json", manifest)
    _log("completed")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="data/CCAO/2025/training_data.parquet")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--model-params", default="model_params.yaml")
    parser.add_argument("--cv-config", default="cv_config.yaml")
    parser.add_argument("--out-dir", default="output/paper_baseline_2024_2025")
    parser.add_argument("--figures-dir", default="paper/img")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--cv-workers", type=int, default=8)
    parser.add_argument("--lgbm-n-jobs", type=int, default=None)
    parser.add_argument("--skip-cv", action="store_true")
    return parser


if __name__ == "__main__":
    run(_parser().parse_args())
