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
    centered = values - mean
    skew = float(np.mean(centered**3) / (std**3)) if std > 0.0 else 0.0
    kurtosis = float(np.mean(centered**4) / (std**4) - 3.0) if std > 0.0 else 0.0

    return {
        "count": float(values.size),
        "mean": mean,
        "median": median,
        "std": std,
        "min": float(np.min(values)),
        "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
        "iqr": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
        "skewness": skew,
        "excess_kurtosis": kurtosis,
        "share_within_1std": float(np.mean(np.abs(values - mean) <= std)),
        "share_within_2std": float(np.mean(np.abs(values - mean) <= 2.0 * std)),
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


def _add_reference_lines(ax: plt.Axes, mean: float, median: float, std: float) -> None:
    line_specs = [
        (mean, "#C1121F", "-", 2.2, "Mean"),
        (median, "#003049", "-", 2.2, "Median"),
        (mean - std, "#F77F00", "--", 1.8, "Mean - 1 std"),
        (mean + std, "#F77F00", "--", 1.8, "Mean + 1 std"),
        (mean - 2.0 * std, "#FCBF49", ":", 2.0, "Mean - 2 std"),
        (mean + 2.0 * std, "#FCBF49", ":", 2.0, "Mean + 2 std"),
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

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.hist(
        values,
        bins=n_bins,
        color="#669BBC",
        edgecolor="white",
        linewidth=0.6,
        alpha=0.9,
        zorder=1,
    )
    _add_reference_lines(ax, mean=mean, median=median, std=std)
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
            f"total={int(df_all.shape[0]):,}",
        ]
    )

    original_plot_path = eda_path / "sale_price_distribution_original.png"
    log_plot_path = eda_path / "sale_price_distribution_log.png"
    stats_csv_path = eda_path / "sale_price_distribution_summary.csv"
    split_csv_path = eda_path / "sale_price_split_summary.csv"

    raw_stats = _write_histogram(
        values=sale_prices,
        out_path=original_plot_path,
        title="CCAO Sale Price Distribution",
        subtitle=f"Original sale prices after the same cleaning used by quick_test_models.py. {split_counts_text}",
        x_label="Sale price",
        text_style="currency",
    )
    log_stats = _write_histogram(
        values=sale_prices_log,
        out_path=log_plot_path,
        title="CCAO Log Sale Price Distribution",
        subtitle=f"Natural-log target used by the predictive models. {split_counts_text}",
        x_label="log(sale price)",
        text_style="number",
    )

    summary_df = pd.DataFrame(
        [
            {"distribution": "original_sale_price", **raw_stats},
            {"distribution": "log_sale_price", **log_stats},
        ]
    )
    summary_df.to_csv(stats_csv_path, index=False)
    split_summary.to_csv(split_csv_path, index=False)
    _log("summary tables written", stats_csv=str(stats_csv_path), split_csv=str(split_csv_path))
    _log("eda finished", eda_dir=str(eda_path))

    return {
        "eda_dir": str(eda_path),
        "original_plot": str(original_plot_path),
        "log_plot": str(log_plot_path),
        "stats_csv": str(stats_csv_path),
        "split_csv": str(split_csv_path),
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
