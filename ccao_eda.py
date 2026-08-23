"""
Exploratory data analysis for the CCAO sale-price target used by `quick_test_models.py`.

This script mirrors the same data-loading, filtering, and splitting logic used in
`quick_test_models.py`, then produces distribution plots for:

1. Original sale prices.
2. Log-transformed sale prices used by the predictive models.

Outputs are written under a dedicated `eda/` directory instead of `output/`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import yaml


_LOG_T0 = time.perf_counter()


def _log(message: str, **fields: Any) -> None:
    dt = time.perf_counter() - _LOG_T0
    suffix = ""
    if fields:
        suffix = " | " + " | ".join(f"{k}={v}" for k, v in fields.items())
    print(f"[ccao_eda +{dt:7.1f}s] {message}{suffix}", flush=True)


def _load_and_split_data(
    *,
    data_path: str,
    params: dict,
    target_column: str,
    date_column: str,
    sample_frac: float | None,
    sample_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """
    Mirrors `quick_test_models.py`:
      - load parquet
      - filter out multicard and outliers
      - keep only predictor + target + date
      - sort by date
      - split into assess (2024), and pre-assess (<2024) then train/validate + test
    """
    _log("loading parquet", data_path=data_path)
    df = pd.read_parquet(data_path, engine="fastparquet")
    _log("parquet loaded", rows=int(df.shape[0]), cols=int(df.shape[1]))
    df = df[
        (~df["ind_pin_is_multicard"].astype("bool").fillna(True))
        & (~df["sv_is_outlier"].astype("bool").fillna(True))
    ].copy()
    _log("row filters applied", rows=int(df.shape[0]))

    predictor_cols = list(params["model"]["predictor"]["all"])
    categorical_cols = list(params["model"]["predictor"]["categorical"])
    keep_cols = predictor_cols + [target_column, date_column]
    df = df.loc[:, keep_cols].copy()
    _log("projected columns", kept_cols=int(len(keep_cols)))

    if sample_frac is not None:
        if not (0.0 < float(sample_frac) <= 1.0):
            raise ValueError("sample_frac must be in (0, 1]. Use None to disable sampling.")
        if float(sample_frac) < 1.0:
            df = df.sample(frac=float(sample_frac), random_state=int(sample_seed)).copy()
            _log("sampling applied", sample_frac=float(sample_frac), rows=int(df.shape[0]))

    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(date_column).reset_index(drop=True)
    _log("date sort completed", rows=int(df.shape[0]))

    df_assess = df.loc[df[date_column].dt.year == 2024, :].copy()
    df_train_all = df.loc[df[date_column].dt.year < 2024, :].copy()

    train_prop = float(params["cv"]["split_prop"])
    split_idx = int(train_prop * df_train_all.shape[0])
    df_test = df_train_all.iloc[split_idx:, :].copy()
    df_train_validate = df_train_all.iloc[:split_idx, :].copy()
    _log(
        "data split completed",
        train_validate_rows=int(df_train_validate.shape[0]),
        test_rows=int(df_test.shape[0]),
        assess_rows=int(df_assess.shape[0]),
    )

    return df_train_validate, df_test, df_assess, predictor_cols, categorical_cols


def _bounded_hist_bin_count(values: np.ndarray, min_bins: int = 30, max_bins: int = 120) -> int:
    if values.size < 2:
        return min_bins
    fd_edges = np.histogram_bin_edges(values, bins="fd")
    bin_count = max(len(fd_edges) - 1, min_bins)
    return int(min(max(bin_count, min_bins), max_bins))


def _summary_stats(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    median = float(np.median(values))
    std = float(np.std(values, ddof=0))
    variance = float(std**2)
    centered = values - mean
    abs_centered = np.abs(centered)
    median_abs_dev = np.abs(values - median)

    quantiles = {
        "p001": float(np.quantile(values, 0.001)),
        "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)),
        "p10": float(np.quantile(values, 0.10)),
        "p25": float(np.quantile(values, 0.25)),
        "p50": median,
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "p999": float(np.quantile(values, 0.999)),
    }

    central_moment_3 = float(np.mean(centered**3))
    central_moment_4 = float(np.mean(centered**4))
    central_moment_5 = float(np.mean(centered**5))
    central_moment_6 = float(np.mean(centered**6))
    skew = float(central_moment_3 / (std**3)) if std > 0.0 else 0.0
    kurtosis = float(central_moment_4 / (std**4) - 3.0) if std > 0.0 else 0.0
    standardized_moment_5 = float(central_moment_5 / (std**5)) if std > 0.0 else 0.0
    standardized_moment_6 = float(central_moment_6 / (std**6)) if std > 0.0 else 0.0

    iqr = float(quantiles["p75"] - quantiles["p25"])
    idr = float(quantiles["p90"] - quantiles["p10"])
    bowley_skewness = (
        float((quantiles["p75"] + quantiles["p25"] - 2.0 * quantiles["p50"]) / iqr) if iqr > 0.0 else 0.0
    )
    crow_siddiqui_skewness = (
        float((quantiles["p90"] + quantiles["p10"] - 2.0 * quantiles["p50"]) / idr) if idr > 0.0 else 0.0
    )
    jarque_bera = float(values.size / 6.0 * (skew**2 + 0.25 * kurtosis**2))
    coeff_var = float(std / mean) if mean != 0.0 else np.nan
    mean_median_ratio = float(mean / median) if median != 0.0 else np.nan

    return {
        "count": float(values.size),
        "mean": mean,
        "median": median,
        "std": std,
        "variance": variance,
        "min": float(np.min(values)),
        **quantiles,
        "max": float(np.max(values)),
        "range": float(np.max(values) - np.min(values)),
        "iqr": iqr,
        "interdecile_range": idr,
        "mean_abs_deviation_from_mean": float(np.mean(abs_centered)),
        "median_abs_deviation_from_median": float(np.median(median_abs_dev)),
        "mad_scaled_to_sigma": float(1.4826 * np.median(median_abs_dev)),
        "coeff_var": coeff_var,
        "mean_median_gap": float(mean - median),
        "mean_median_ratio": mean_median_ratio,
        "pearson_median_skewness": float(3.0 * (mean - median) / std) if std > 0.0 else 0.0,
        "bowley_skewness": bowley_skewness,
        "crow_siddiqui_skewness": crow_siddiqui_skewness,
        "central_moment_2": variance,
        "central_moment_3": central_moment_3,
        "central_moment_4": central_moment_4,
        "central_moment_5": central_moment_5,
        "central_moment_6": central_moment_6,
        "skewness": skew,
        "excess_kurtosis": kurtosis,
        "standardized_moment_5": standardized_moment_5,
        "standardized_moment_6": standardized_moment_6,
        "jarque_bera": jarque_bera,
        "share_within_1std": float(np.mean(np.abs(values - mean) <= std)),
        "share_within_2std": float(np.mean(np.abs(values - mean) <= 2.0 * std)),
        "share_within_3std": float(np.mean(np.abs(values - mean) <= 3.0 * std)),
    }


def _format_value(value: float, style: str) -> str:
    if style == "currency":
        return f"${value:,.0f}"
    if style == "percent":
        return f"{100.0 * value:.1f}%"
    return f"{value:,.3f}"


def _summary_text(stats: Dict[str, float], style: str) -> str:
    return "\n".join(
        [
            f"n = {int(stats['count']):,}",
            f"mean = {_format_value(stats['mean'], style)}",
            f"median = {_format_value(stats['median'], style)}",
            f"std = {_format_value(stats['std'], style)}",
            f"min / max = {_format_value(stats['min'], style)} / {_format_value(stats['max'], style)}",
            f"p01 / p99 = {_format_value(stats['p01'], style)} / {_format_value(stats['p99'], style)}",
            f"skewness = {stats['skewness']:.3f}",
            f"within +/-1 std = {_format_value(stats['share_within_1std'], 'percent')}",
            f"within +/-2 std = {_format_value(stats['share_within_2std'], 'percent')}",
        ]
    )


def _quantile_shape_metrics(values: np.ndarray, n_points: int = 1001) -> Dict[str, float]:
    q = np.linspace(0.001, 0.999, n_points, dtype=float)
    q_values = np.quantile(values, q)
    scale = max(float(np.std(q_values, ddof=0)), 1e-12)
    normalized_curve = (q_values - float(np.median(q_values))) / scale
    first_diff = np.diff(normalized_curve)
    second_diff = np.diff(normalized_curve, n=2)
    denom = max(float(np.mean(np.abs(first_diff))), 1e-12)
    roughness = float(np.mean(np.abs(second_diff)) / denom) if second_diff.size else 0.0
    smoothness = float(1.0 / (1.0 + roughness))
    return {
        "quantile_roughness": roughness,
        "quantile_smoothness": smoothness,
    }


def _compute_center_difference_metrics(
    values: np.ndarray,
    *,
    center_name: str,
    center_value: float,
) -> Dict[str, float | str]:
    diffs = np.asarray(values, dtype=float) - float(center_value)
    rel_diffs = diffs / float(center_value)
    below_mask = diffs < 0.0
    above_mask = diffs > 0.0
    at_center_mask = ~(below_mask | above_mask)

    below_gaps = -diffs[below_mask]
    above_gaps = diffs[above_mask]
    rel_below_gaps = -rel_diffs[below_mask]
    rel_above_gaps = rel_diffs[above_mask]

    mean_below_gap = float(np.mean(below_gaps)) if below_gaps.size else np.nan
    mean_above_gap = float(np.mean(above_gaps)) if above_gaps.size else np.nan
    median_below_gap = float(np.median(below_gaps)) if below_gaps.size else np.nan
    median_above_gap = float(np.median(above_gaps)) if above_gaps.size else np.nan
    mean_rel_below_gap = float(np.mean(rel_below_gaps)) if rel_below_gaps.size else np.nan
    mean_rel_above_gap = float(np.mean(rel_above_gaps)) if rel_above_gaps.size else np.nan
    median_rel_below_gap = float(np.median(rel_below_gaps)) if rel_below_gaps.size else np.nan
    median_rel_above_gap = float(np.median(rel_above_gaps)) if rel_above_gaps.size else np.nan

    return {
        "center_name": center_name,
        "center_value": float(center_value),
        "function_mean": float(np.mean(diffs)),
        "function_median": float(np.median(diffs)),
        "relative_function_mean": float(np.mean(rel_diffs)),
        "relative_function_median": float(np.median(rel_diffs)),
        "share_below_center": float(np.mean(below_mask)),
        "share_above_center": float(np.mean(above_mask)),
        "share_at_center": float(np.mean(at_center_mask)),
        "crossing_percentile": float(100.0 * np.mean(values <= center_value)),
        "mean_gap_below": mean_below_gap,
        "mean_gap_above": mean_above_gap,
        "median_gap_below": median_below_gap,
        "median_gap_above": median_above_gap,
        "mean_relative_gap_below": mean_rel_below_gap,
        "mean_relative_gap_above": mean_rel_above_gap,
        "median_relative_gap_below": median_rel_below_gap,
        "median_relative_gap_above": median_rel_above_gap,
        "gap_ratio_mean": float(mean_above_gap / mean_below_gap) if np.isfinite(mean_below_gap) and mean_below_gap > 0.0 else np.nan,
        "gap_ratio_median": float(median_above_gap / median_below_gap) if np.isfinite(median_below_gap) and median_below_gap > 0.0 else np.nan,
        "relative_gap_ratio_mean": float(mean_rel_above_gap / mean_rel_below_gap) if np.isfinite(mean_rel_below_gap) and mean_rel_below_gap > 0.0 else np.nan,
        "relative_gap_ratio_median": float(median_rel_above_gap / median_rel_below_gap) if np.isfinite(median_rel_below_gap) and median_rel_below_gap > 0.0 else np.nan,
    }


def _difference_summary_text(
    *,
    stats: Dict[str, float],
    shape_metrics: Dict[str, float],
    mean_metrics: Dict[str, float | str],
    median_metrics: Dict[str, float | str],
    text_style: str,
) -> str:
    return "\n".join(
        [
            f"mean = {_format_value(stats['mean'], text_style)}",
            f"median = {_format_value(stats['median'], text_style)}",
            f"std = {_format_value(stats['std'], text_style)}",
            f"skewness = {stats['skewness']:.3f}",
            f"quantile smoothness = {shape_metrics['quantile_smoothness']:.3f}",
            f"F(mean) = {float(mean_metrics['crossing_percentile']):.1f}%",
            f"F(median) = {float(median_metrics['crossing_percentile']):.1f}%",
            f"avg gap below/above mean = {_format_value(float(mean_metrics['mean_gap_below']), text_style)} / {_format_value(float(mean_metrics['mean_gap_above']), text_style)}",
            f"avg gap below/above median = {_format_value(float(median_metrics['mean_gap_below']), text_style)} / {_format_value(float(median_metrics['mean_gap_above']), text_style)}",
            f"median[y - mean(y)] = {_format_value(float(mean_metrics['function_median']), text_style)}",
            f"mean[y - median(y)] = {_format_value(float(median_metrics['function_mean']), text_style)}",
        ]
    )

def _add_reference_lines(
    ax: plt.Axes, 
    mean: float, 
    median: float, 
    std: float, 
    data_min: float=False, 
    data_max: float=False
) -> None:
    line_specs = [
        (mean, "#C1121F", "-", 2.2, "Mean"),
        (median, "#003049", "-", 2.2, "Median"),
        (mean - std, "#F77F00", "--", 1.8, "Mean - 1 std"),
        (mean + std, "#F77F00", "--", 1.8, "Mean + 1 std"),
        (mean - 2.0 * std, "#FCBF49", ":", 2.0, "Mean - 2 std"),
        (mean + 2.0 * std, "#FCBF49", ":", 2.0, "Mean + 2 std"),
        (data_min, "#2A9D8F", "-.", 1.8, "Min"),  # Added Min
        (data_max, "#2A9D8F", "-.", 1.8, "Max"),  # Added Max
    ]
    
    for x_value, color, linestyle, linewidth, label in line_specs:
        ax.axvline(
            x_value,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=0.95,
            label=label,
            zorder=3,
        )


def _write_histogram(
    *,
    values: np.ndarray,
    out_path: Path,
    title: str,
    subtitle: str,
    x_label: str,
    text_style: str,
) -> Dict[str, float]:
    stats = _summary_stats(values)
    mean = stats["mean"]
    median = stats["median"]
    std = stats["std"]
    n_bins = _bounded_hist_bin_count(values)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        values,
        bins=n_bins,
        color="#669BBC",
        edgecolor="white",
        linewidth=0.6,
        alpha=0.9,
        zorder=1,
    )
    _add_reference_lines(ax, mean=mean, median=median, std=std, data_min=stats["min"], data_max=stats["max"])
    ax.set_title(f"{title}\n{subtitle}", fontsize=15, pad=14)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)

    if text_style == "currency":
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))

    ax.text(
        0.985,
        0.985,
        _summary_text(stats, text_style),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#D9D9D9", "alpha": 0.96},
    )
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    _log("histogram written", path=str(out_path), bins=n_bins, rows=int(stats["count"]))
    return stats


def _write_difference_function_plot(
    *,
    values: np.ndarray,
    out_path: Path,
    title: str,
    subtitle: str,
    x_label: str,
    text_style: str,
    scale_name: str,
) -> List[Dict[str, float | str]]:
    stats = _summary_stats(values)
    shape_metrics = _quantile_shape_metrics(values)
    mean_metrics = _compute_center_difference_metrics(values, center_name="mean", center_value=stats["mean"])
    median_metrics = _compute_center_difference_metrics(values, center_name="median", center_value=stats["median"])

    q = np.linspace(0.001, 0.999, 1200, dtype=float)
    q_values = np.quantile(values, q)

    mean_center = stats["mean"]
    median_center = stats["median"]
    mean_centered = q_values - mean_center
    median_centered = q_values - median_center
    mean_centered_rel = mean_centered / mean_center
    median_centered_rel = median_centered / median_center

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(q_values, mean_centered, color="#C1121F", linewidth=2.2, label="y - mean(y)")
    axes[0].plot(q_values, median_centered, color="#003049", linewidth=2.2, label="y - median(y)")
    axes[0].axhline(0.0, color="#4B5563", linewidth=1.2, alpha=0.8)
    axes[0].axvline(mean_center, color="#C1121F", linestyle="--", linewidth=1.3, alpha=0.7)
    axes[0].axvline(median_center, color="#003049", linestyle="--", linewidth=1.3, alpha=0.7)
    axes[0].set_ylabel("Centered difference")
    axes[0].grid(alpha=0.25, linewidth=0.8)
    axes[0].legend(loc="upper left", frameon=True)

    axes[1].plot(q_values, mean_centered_rel, color="#C1121F", linewidth=2.2, label="(y - mean(y)) / mean(y)")
    axes[1].plot(q_values, median_centered_rel, color="#003049", linewidth=2.2, label="(y - median(y)) / median(y)")
    axes[1].axhline(0.0, color="#4B5563", linewidth=1.2, alpha=0.8)
    axes[1].axvline(mean_center, color="#C1121F", linestyle="--", linewidth=1.3, alpha=0.7)
    axes[1].axvline(median_center, color="#003049", linestyle="--", linewidth=1.3, alpha=0.7)
    axes[1].set_ylabel("Relative centered difference")
    axes[1].set_xlabel(x_label)
    axes[1].grid(alpha=0.25, linewidth=0.8)
    axes[1].legend(loc="upper left", frameon=True)

    if scale_name == "original_sale_price":
        abs_linthresh = max(stats["std"] * 0.03, 1_000.0)
        rel_linthresh = max(float(np.nanstd(mean_centered_rel)) * 0.08, 0.02)
        axes[0].set_yscale("symlog", linthresh=abs_linthresh)
        axes[1].set_yscale("symlog", linthresh=rel_linthresh)
        for ax in axes:
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))

    axes[0].set_title(f"{title}\n{subtitle}", fontsize=15, pad=12)
    fig.text(
        0.985,
        0.5,
        _difference_summary_text(
            stats=stats,
            shape_metrics=shape_metrics,
            mean_metrics=mean_metrics,
            median_metrics=median_metrics,
            text_style=text_style,
        ),
        ha="right",
        va="center",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#D9D9D9", "alpha": 0.96},
    )
    fig.tight_layout(rect=(0.0, 0.0, 0.8, 1.0))
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    _log("difference plot written", path=str(out_path), scale=scale_name)
    return [
        {"scale": scale_name, **shape_metrics, **mean_metrics},
        {"scale": scale_name, **shape_metrics, **median_metrics},
    ]


def _write_cdf_comparison_plot(
    *,
    real_values: np.ndarray,
    normal_values: np.ndarray,
    logistic_values: np.ndarray,
    out_path: Path,
    title: str,
    subtitle: str,
    x_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    series = [
        ("Real log sale price", real_values, "#003049"),
        ("Normal approximation", normal_values, "#C1121F"),
        ("Logistic approximation", logistic_values, "#F77F00"),
    ]
    real_values = np.asarray(real_values, dtype=float)
    real_quantiles = {
        "p10": float(np.quantile(real_values, 0.10)),
        "p50": float(np.quantile(real_values, 0.50)),
        "p90": float(np.quantile(real_values, 0.90)),
    }

    def _empirical_cdf_distance(reference: np.ndarray, candidate: np.ndarray) -> Tuple[float, float]:
        reference = np.sort(np.asarray(reference, dtype=float))
        candidate = np.sort(np.asarray(candidate, dtype=float))
        grid = np.sort(np.concatenate([reference, candidate]))
        reference_cdf = np.searchsorted(reference, grid, side="right") / reference.size
        candidate_cdf = np.searchsorted(candidate, grid, side="right") / candidate.size
        ks_distance = float(np.max(np.abs(reference_cdf - candidate_cdf)))
        mean_cdf_gap = float(np.mean(np.abs(reference_cdf - candidate_cdf)))
        return ks_distance, mean_cdf_gap

    for label, values, color in series:
        sorted_values = np.sort(np.asarray(values, dtype=float))
        cumulative_prob = np.arange(1, sorted_values.size + 1, dtype=float) / sorted_values.size
        ax.plot(sorted_values, cumulative_prob, color=color, linewidth=2.1, label=label)

    quantile_line_specs = [
        (real_quantiles["p10"], "Real p10"),
        (real_quantiles["p50"], "Real median"),
        (real_quantiles["p90"], "Real p90"),
    ]
    for x_value, label in quantile_line_specs:
        ax.axvline(
            x_value,
            color="#6B7280",
            linestyle="--",
            linewidth=1.1,
            alpha=0.8,
            zorder=1,
            label=label,
        )

    normal_ks, normal_mean_cdf_gap = _empirical_cdf_distance(real_values, normal_values)
    logistic_ks, logistic_mean_cdf_gap = _empirical_cdf_distance(real_values, logistic_values)

    ax.set_title(f"{title}\n{subtitle}", fontsize=15, pad=14)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Cumulative probability")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.text(
        0.985,
        0.03,
        "\n".join(
            [
                f"Real p10 / median / p90 = {real_quantiles['p10']:.3f} / {real_quantiles['p50']:.3f} / {real_quantiles['p90']:.3f}",
                f"Normal KS / mean |dCDF| = {normal_ks:.4f} / {normal_mean_cdf_gap:.4f}",
                f"Logistic KS / mean |dCDF| = {logistic_ks:.4f} / {logistic_mean_cdf_gap:.4f}",
            ]
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#D9D9D9", "alpha": 0.96},
    )
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    _log("cdf comparison plot written", path=str(out_path))


def _build_split_summary(
    *,
    df_train_validate: pd.DataFrame,
    df_test: pd.DataFrame,
    df_assess: pd.DataFrame,
    target_column: str,
    date_column: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    split_frames = {
        "train_validate_pre2024": df_train_validate,
        "test_pre2024": df_test,
        "assess_2024": df_assess,
    }
    for split_name, split_df in split_frames.items():
        if split_df.empty:
            rows.append(
                {
                    "split": split_name,
                    "rows": 0,
                    "sale_price_mean": np.nan,
                    "sale_price_median": np.nan,
                    "sale_price_min": np.nan,
                    "sale_price_max": np.nan,
                    "sale_date_min": pd.NaT,
                    "sale_date_max": pd.NaT,
                }
            )
            continue
        prices = split_df[target_column].to_numpy(dtype=float)
        rows.append(
            {
                "split": split_name,
                "rows": int(split_df.shape[0]),
                "sale_price_mean": float(np.mean(prices)),
                "sale_price_median": float(np.median(prices)),
                "sale_price_min": float(np.min(prices)),
                "sale_price_max": float(np.max(prices)),
                "sale_date_min": pd.to_datetime(split_df[date_column]).min(),
                "sale_date_max": pd.to_datetime(split_df[date_column]).max(),
            }
        )
    return pd.DataFrame(rows)


def run_eda(
    *,
    eda_dir: str,
    data_path: str,
    sample_frac: float | None,
    seed: int,
) -> Dict[str, str]:
    target_column = "meta_sale_price"
    date_column = "meta_sale_date"
    eda_path = Path(eda_dir)
    eda_path.mkdir(parents=True, exist_ok=True)

    _log("eda start", eda_dir=str(eda_path), data_path=data_path, sample_frac=sample_frac)

    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    _log("configuration loaded")

    df_train_validate, df_test, df_assess, _, _ = _load_and_split_data(
        data_path=data_path,
        params=params,
        target_column=target_column,
        date_column=date_column,
        sample_frac=sample_frac,
        sample_seed=seed,
    )

    df_all = pd.concat([df_train_validate, df_test, df_assess], axis=0, ignore_index=True)
    sale_prices = df_all[target_column].to_numpy(dtype=float)
    sale_prices_log = np.log(sale_prices)
    split_summary = _build_split_summary(
        df_train_validate=df_train_validate,
        df_test=df_test,
        df_assess=df_assess,
        target_column=target_column,
        date_column=date_column,
    )

    split_counts_text = ", ".join(
        [
            f"train/validate={int(df_train_validate.shape[0]):,}",
            f"test={int(df_test.shape[0]):,}",
            f"assess={int(df_assess.shape[0]):,}",
            # f"total={int(df_all.shape[0]):,}",
        ]
    )

    original_plot_path = eda_path / "sale_price_distribution_original.pdf"
    log_plot_path = eda_path / "sale_price_distribution_log.pdf"
    log_normal_plot_path = eda_path / "sale_price_distribution_log_normal_approx.pdf"
    log_logistic_plot_path = eda_path / "sale_price_distribution_log_logistic_approx.pdf"
    log_cdf_comparison_plot_path = eda_path / "sale_price_distribution_log_cdf_comparison.pdf"
    original_diff_plot_path = eda_path / "sale_price_difference_functions_original.pdf"
    log_diff_plot_path = eda_path / "sale_price_difference_functions_log.pdf"
    stats_csv_path = eda_path / "sale_price_distribution_summary.csv"
    split_csv_path = eda_path / "sale_price_split_summary.csv"
    diff_csv_path = eda_path / "sale_price_difference_function_summary.csv"

    raw_stats = _write_histogram(
        values=sale_prices,
        out_path=original_plot_path,
        title="CCAO Sale Price Distribution",
        subtitle=f"{split_counts_text}",
        x_label="Sale price",
        text_style="currency",
    )
    log_stats = _write_histogram(
        values=sale_prices_log,
        out_path=log_plot_path,
        title="CCAO Log Sale Price Distribution",
        subtitle=f"{split_counts_text}",
        x_label="log(sale price)",
        text_style="number",
    )
    log_normal_stats = _summary_stats(sale_prices_log)
    sale_prices_log_normal = np.random.default_rng(seed).normal(
        loc=log_normal_stats["mean"],
        scale=log_normal_stats["std"],
        size=sale_prices_log.shape[0],
    )
    log_normal_stats = _write_histogram(
        values=sale_prices_log_normal,
        out_path=log_normal_plot_path,
        title="CCAO Approximate Normal Log Sale Price Distribution",
        subtitle=f"{split_counts_text}",
        x_label="log(sale price)",
        text_style="number",
    )
    sale_prices_log_logistic = np.random.default_rng(seed).logistic(
        loc=log_stats["mean"],
        scale=log_stats["std"] * np.sqrt(3.0) / np.pi,
        size=sale_prices_log.shape[0],
    )
    log_logistic_stats = _write_histogram(
        values=sale_prices_log_logistic,
        out_path=log_logistic_plot_path,
        title="CCAO Approximate Logistic Log Sale Price Distribution",
        subtitle=f"{split_counts_text}",
        x_label="log(sale price)",
        text_style="number",
    )
    _write_cdf_comparison_plot(
        real_values=sale_prices_log,
        normal_values=sale_prices_log_normal,
        logistic_values=sale_prices_log_logistic,
        out_path=log_cdf_comparison_plot_path,
        title="CCAO Log Sale Price CDF Comparison",
        subtitle=f"{split_counts_text}",
        x_label="log(sale price)",
    )
    diff_rows: List[Dict[str, float | str]] = []
    diff_rows.extend(
        _write_difference_function_plot(
            values=sale_prices,
            out_path=original_diff_plot_path,
            title="CCAO Sale Price Difference Functions",
            subtitle=f"{split_counts_text}",
            x_label="Sale Price P",
            text_style="currency",
            scale_name="original_sale_price",
        )
    )
    diff_rows.extend(
        _write_difference_function_plot(
            values=sale_prices_log,
            out_path=log_diff_plot_path,
            title="CCAO Log Sale Price Difference Functions",
            subtitle=f"{split_counts_text}",
            x_label="Log Sale Price y = log(P)",
            text_style="number",
            scale_name="log_sale_price",
        )
    )

    summary_df = pd.DataFrame(
        [
            {"distribution": "original_sale_price", **raw_stats},
            {"distribution": "log_sale_price", **log_stats},
            {"distribution": "log_sale_price_normal_approx", **log_normal_stats},
            {"distribution": "log_sale_price_logistic_approx", **log_logistic_stats},
        ]
    )
    diff_summary_df = pd.DataFrame(diff_rows)
    summary_df.to_csv(stats_csv_path, index=False)
    split_summary.to_csv(split_csv_path, index=False)
    diff_summary_df.to_csv(diff_csv_path, index=False)
    _log(
        "summary tables written",
        stats_csv=str(stats_csv_path),
        split_csv=str(split_csv_path),
        diff_csv=str(diff_csv_path),
    )
    _log("eda finished", eda_dir=str(eda_path))

    return {
        "eda_dir": str(eda_path),
        "original_plot": str(original_plot_path),
        "log_plot": str(log_plot_path),
        "log_normal_plot": str(log_normal_plot_path),
        "log_logistic_plot": str(log_logistic_plot_path),
        "log_cdf_comparison_plot": str(log_cdf_comparison_plot_path),
        "original_difference_plot": str(original_diff_plot_path),
        "log_difference_plot": str(log_diff_plot_path),
        "stats_csv": str(stats_csv_path),
        "split_csv": str(split_csv_path),
        "difference_csv": str(diff_csv_path),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="EDA for the cleaned CCAO sale-price target distribution used by quick_test_models.py."
    )
    p.add_argument(
        "--eda-dir",
        type=str,
        default="./eda",
        help="Directory to write EDA plots and summary tables.",
    )
    p.add_argument(
        "--data-path",
        type=str,
        default="./data/CCAO/2025/training_data.parquet",
        help="Path to the training parquet used by quick_test_models.py.",
    )
    p.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        help="Optional down-sampling fraction in (0,1] applied after cleaning, matching quick_test_models.py behavior.",
    )
    p.add_argument("--seed", type=int, default=4050, help="Random seed used when --sample-frac is provided.")
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    out = run_eda(
        eda_dir=str(args.eda_dir),
        data_path=str(args.data_path),
        sample_frac=(None if args.sample_frac is None else float(args.sample_frac)),
        seed=int(args.seed),
    )
    print("=" * 90)
    for key, value in out.items():
        print(f"{key}: {value}")
