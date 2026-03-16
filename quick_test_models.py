"""
Quick test runner.

Goal
----
Fit and evaluate the 4 core models on:
  - held-out test split (most recent pre-2024 sales; ~2023 by CCAO-style split)
  - assessment split (2024 sales)

This script intentionally mirrors the preprocessing + split logic in `main.py`,
but avoids CV and bootstrapping to stay fast and easy to read.

Models
------
1) LinearRegression (baseline)
2) LGBMRegressor (baseline; defaults from `model_params.yaml` + fallback `params.yaml`)
3) LGBSmoothPenalty (fairness-regularized; uses `rho`)
4) LGBCovPenalty (fairness-regularized; uses `rho`)

Outputs
-------
Writes 2 CSV tables under `--out-dir`:
  - quick_test_metrics_test.csv
  - quick_test_metrics_assess.csv
  - quick_test_metrics_validation_bootstrap_avg.csv

Each table contains accuracy + vertical equity metrics computed with the same
metric routine used elsewhere in this repo (`_compute_extended_metrics`).

Usage
-----
From the `soft-vertical-equity-constrained-mass-appraissal/` directory:

  python quick_test_models.py --rho 1.0
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression
import lightgbm as lgb

from preprocessing.recipes_pipelined import build_model_pipeline
from soft_constrained_models.boosting_models import LGBCovPenalty, LGBSmoothPenalty, LGBSmoothPenaltyGrouped, LGBCovPenaltyCVaR, LGBSmoothPenaltyCVaR, LGBCovPenaltyCVaRTotal, LGBSmoothPenaltyCVaRTotal
from utils.plotting_utils import plot_ratio_vs_logprice
from utils.motivation_utils import _build_time_block_bootstrap_indices, _compute_extended_metrics


_PAIRWISE_DEPENDENCE_SAMPLE_N = 1024
_MAIN_SWEEP_FAMILIES = ("LGBSmoothPenalty", "LGBCovPenalty", "LGBSmoothPenaltyGrouped")
_RHO_PLOT_METRICS = [
    "Corr(r,price)",
    "Corr(r,logprice)",
    "dCor(r,logprice)_sampled",
    "ChatterjeeXi(r,logprice)",
    "nHSIC(r,logprice)_sampled",
]
_META_TOWNSHIP_TRIAD_COL = "meta_township_triad"
_CHAR_CLASS_BUCKET_COL = "char_class_bucket"
_TOWNSHIP_TRIAD_MAP = {
    "10": "north_northwest_suburbs",
    "16": "north_northwest_suburbs",
    "17": "north_northwest_suburbs",
    "18": "north_northwest_suburbs",
    "20": "north_northwest_suburbs",
    "22": "north_northwest_suburbs",
    "23": "north_northwest_suburbs",
    "24": "north_northwest_suburbs",
    "25": "north_northwest_suburbs",
    "26": "north_northwest_suburbs",
    "29": "north_northwest_suburbs",
    "35": "north_northwest_suburbs",
    "38": "north_northwest_suburbs",
    "11": "south_southwest_suburbs",
    "12": "south_southwest_suburbs",
    "13": "south_southwest_suburbs",
    "14": "south_southwest_suburbs",
    "15": "south_southwest_suburbs",
    "19": "south_southwest_suburbs",
    "21": "south_southwest_suburbs",
    "27": "south_southwest_suburbs",
    "28": "south_southwest_suburbs",
    "30": "south_southwest_suburbs",
    "31": "south_southwest_suburbs",
    "32": "south_southwest_suburbs",
    "33": "south_southwest_suburbs",
    "34": "south_southwest_suburbs",
    "36": "south_southwest_suburbs",
    "37": "south_southwest_suburbs",
    "39": "south_southwest_suburbs",
    "70": "city_of_chicago",
    "71": "city_of_chicago",
    "72": "city_of_chicago",
    "73": "city_of_chicago",
    "74": "city_of_chicago",
    "75": "city_of_chicago",
    "76": "city_of_chicago",
    "77": "city_of_chicago",
}
_CHAR_CLASS_BUCKET_MAP = {
    "202": "one_story_residences",
    "203": "one_story_residences",
    "204": "one_story_residences",
    "205": "two_story_older_than_62_years",
    "206": "two_story_older_than_62_years",
    "207": "two_story_newer_than_62_years",
    "208": "two_story_newer_than_62_years",
    "209": "two_story_newer_than_62_years",
    "278": "two_story_newer_than_62_years",
    "210": "townhomes_rowhouses",
    "295": "townhomes_rowhouses",
    "211": "multi_family_mixed_use_2_to_6_units",
    "212": "multi_family_mixed_use_2_to_6_units",
    "234": "split_level_residences",
    "299": "condominiums",
    "218": "char_class_anomaly",
    "219": "char_class_anomaly",
    "297": "char_class_anomaly",
    "NA": "NA",
}


def _build_lgbm_params_from_files(model_params: dict, ccao_params: dict, seed: int) -> dict:
    """
    Match `main.py`: use `model_params.yaml` as primary defaults and fall back to
    `params.yaml` for any missing keys.
    """
    model_default = dict(model_params.get("LGBMRegressor", {}))
    hp_default = dict(ccao_params["model"]["hyperparameter"]["default"])

    num_leaves = int(model_default.get("num_leaves", hp_default["num_leaves"]))
    if "max_depth" in model_default and model_default["max_depth"] is not None:
        max_depth = int(model_default["max_depth"])
    else:
        add_to_linked_depth = int(hp_default.get("add_to_linked_depth", 4))
        max_depth = int(np.floor(np.log2(max(num_leaves, 2))) + add_to_linked_depth)

    return {
        "boosting_type": str(model_default.get("boosting_type", "gbdt")),
        "objective": str(model_default.get("objective", "mse")),
        "n_estimators": int(model_default.get("n_estimators", hp_default["num_iterations"])),
        "learning_rate": float(model_default.get("learning_rate", hp_default["learning_rate"])),
        "num_leaves": num_leaves,
        "max_depth": max_depth,
        "max_bin": int(model_default.get("max_bin", hp_default["max_bin"])),
        "min_child_samples": int(model_default.get("min_child_samples", hp_default["min_data_in_leaf"])),
        "min_split_gain": float(model_default.get("min_split_gain", hp_default["min_gain_to_split"])),
        "colsample_bytree": float(model_default.get("colsample_bytree", hp_default["feature_fraction"])),
        "reg_alpha": float(model_default.get("reg_alpha", hp_default["lambda_l1"])),
        "reg_lambda": float(model_default.get("reg_lambda", hp_default["lambda_l2"])),
        "max_cat_threshold": int(model_default.get("max_cat_threshold", hp_default["max_cat_threshold"])),
        "min_data_per_group": int(model_default.get("min_data_per_group", hp_default["min_data_per_group"])),
        "cat_smooth": float(model_default.get("cat_smooth", hp_default["cat_smooth"])),
        "cat_l2": float(model_default.get("cat_l2", hp_default["cat_l2"])),
        "random_state": int(model_default.get("random_state", seed)),
        "n_jobs": int(model_default.get("n_jobs", 1)),
        "verbosity": int(model_default.get("verbosity", -1)),
        "importance_type": str(model_default.get("importance_type", "split")),
    }


def _series_to_string_with_na(series: pd.Series) -> pd.Series:
    return series.astype("object").where(series.notna(), "NA").astype(str)


def _map_meta_township_triad(series: pd.Series) -> pd.Series:
    return _series_to_string_with_na(series).map(_TOWNSHIP_TRIAD_MAP).fillna("other_or_unmapped_township")


def _map_char_class_bucket(series: pd.Series) -> pd.Series:
    return _series_to_string_with_na(series).map(_CHAR_CLASS_BUCKET_MAP).fillna("other_or_unmapped_char_class")


def _add_quick_test_grouped_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "meta_township_code" in df.columns:
        df[_META_TOWNSHIP_TRIAD_COL] = _map_meta_township_triad(df["meta_township_code"])
    if "char_class" in df.columns:
        df[_CHAR_CLASS_BUCKET_COL] = _map_char_class_bucket(df["char_class"])
    return df


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
    Mirrors `main.py`:
      - load parquet
      - filter out multicard and outliers
      - keep only predictor + target + date
      - sort by date
      - split into assess (2024), and pre-assess (<2024) then train/validate + test
    """
    df = pd.read_parquet(data_path, engine="fastparquet")
    df = df[
        (~df["ind_pin_is_multicard"].astype("bool").fillna(True))
        & (~df["sv_is_outlier"].astype("bool").fillna(True))
    ].copy()

    predictor_cols = list(params["model"]["predictor"]["all"])
    categorical_cols = list(params["model"]["predictor"]["categorical"])
    keep_cols = predictor_cols + [target_column, date_column]
    df = df.loc[:, keep_cols].copy()

    if sample_frac is not None:
        if not (0.0 < float(sample_frac) <= 1.0):
            raise ValueError("sample_frac must be in (0, 1]. Use None to disable sampling.")
        if float(sample_frac) < 1.0:
            df = df.sample(frac=float(sample_frac), random_state=int(sample_seed)).copy()

    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(date_column).reset_index(drop=True)

    df_assess = df.loc[df[date_column].dt.year == 2024, :].copy()
    df_train_all = df.loc[df[date_column].dt.year < 2024, :].copy()

    train_prop = float(params["cv"]["split_prop"])
    split_idx = int(train_prop * df_train_all.shape[0])
    df_test = df_train_all.iloc[split_idx:, :].copy()
    df_train_validate = df_train_all.iloc[:split_idx, :].copy()

    return df_train_validate, df_test, df_assess, predictor_cols, categorical_cols


def _build_rho_sweep(
    rho: float,
    rho_range_raw: str,
    rho_count: int,
    rho_scale: str,
) -> List[float]:
    if str(rho_range_raw).strip() == "":
        return [float(rho)]

    bounds = [float(token.strip()) for token in str(rho_range_raw).split(",") if token.strip()]
    if len(bounds) != 2:
        raise ValueError("rho_range must contain exactly two comma-separated values: min,max.")

    count = int(rho_count)
    if count < 1:
        raise ValueError("rho_count must be >= 1.")

    lo, hi = float(bounds[0]), float(bounds[1])
    if count == 1:
        return [lo]

    scale = str(rho_scale).strip().lower()
    if scale == "linear":
        values = np.linspace(lo, hi, count, dtype=float)
    elif scale in {"log", "geom"}:
        if lo <= 0.0 or hi <= 0.0:
            raise ValueError("rho_range bounds must be > 0 for rho_scale=log/geom.")
        values = np.geomspace(lo, hi, count, dtype=float)
    else:
        raise ValueError("rho_scale must be one of: linear, log, geom.")

    return [float(v) for v in values.tolist()]


def _pairwise_metric_subsample(x: np.ndarray, y: np.ndarray, max_n: int) -> Tuple[np.ndarray, np.ndarray]:
    n = int(x.size)
    if n <= max_n:
        return x, y
    order = np.argsort(x, kind="mergesort")
    take = (np.arange(max_n, dtype=int) * n) // max_n
    idx = order[take]
    return x[idx], y[idx]


def _double_center(matrix: np.ndarray) -> np.ndarray:
    row_mean = matrix.mean(axis=1, keepdims=True)
    col_mean = matrix.mean(axis=0, keepdims=True)
    grand_mean = float(matrix.mean())
    return matrix - row_mean - col_mean + grand_mean


def _distance_correlation_sampled(x: np.ndarray, y: np.ndarray, max_n: int) -> Tuple[float, int]:
    if x.size < 2 or y.size < 2:
        return np.nan, int(min(x.size, y.size))
    xs, ys = _pairwise_metric_subsample(x, y, max_n=max_n)
    a = np.abs(xs[:, None] - xs[None, :])
    b = np.abs(ys[:, None] - ys[None, :])
    a = _double_center(a)
    b = _double_center(b)
    dcov2 = float(np.mean(a * b))
    dvar_x = float(np.mean(a * a))
    dvar_y = float(np.mean(b * b))
    if dvar_x <= 0.0 or dvar_y <= 0.0:
        return 0.0, int(xs.size)
    dcor2 = float(np.clip(dcov2 / np.sqrt(dvar_x * dvar_y), 0.0, 1.0))
    return float(np.sqrt(dcor2)), int(xs.size)


def _rbf_kernel_from_1d(values: np.ndarray) -> np.ndarray:
    diffs = values[:, None] - values[None, :]
    sqdist = diffs * diffs
    upper = sqdist[np.triu_indices(values.size, k=1)]
    positive = upper[upper > 0.0]
    if positive.size == 0:
        return np.ones_like(sqdist)
    sigma2 = float(np.median(positive))
    gamma = 1.0 / max(2.0 * sigma2, 1e-12)
    return np.exp(-gamma * sqdist)


def _normalized_hsic_sampled(x: np.ndarray, y: np.ndarray, max_n: int) -> Tuple[float, int]:
    if x.size < 2 or y.size < 2:
        return np.nan, int(min(x.size, y.size))
    xs, ys = _pairwise_metric_subsample(x, y, max_n=max_n)
    kx = _double_center(_rbf_kernel_from_1d(xs))
    ky = _double_center(_rbf_kernel_from_1d(ys))
    denom = float(np.sqrt(np.sum(kx * kx) * np.sum(ky * ky)))
    if denom <= 0.0:
        return 0.0, int(xs.size)
    nhsic = float(np.sum(kx * ky) / denom)
    return float(np.clip(nhsic, 0.0, 1.0)), int(xs.size)


def _chatterjee_xi(x: np.ndarray, y: np.ndarray) -> float:
    n = int(min(x.size, y.size))
    if n < 2:
        return np.nan
    if np.all(x == x[0]) or np.all(y == y[0]):
        return 0.0

    order = np.argsort(x, kind="mergesort")
    y_ord = y[order]
    y_sorted = np.sort(y)
    r = np.searchsorted(y_sorted, y_ord, side="right").astype(float)
    l = (n - np.searchsorted(y_sorted, y_ord, side="left")).astype(float)

    denom = float(2.0 * np.sum(l * (n - l)))
    if denom <= 0.0:
        return 0.0

    xi = 1.0 - (float(n) * float(np.sum(np.abs(np.diff(r))))) / denom
    return float(np.clip(xi, -1.0, 1.0))


def _compute_logprice_dependence_metrics(
    *,
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    ratio_mode: str,
    eps_y: float = 1e-12,
) -> Dict[str, Any]:
    y_true_log = np.asarray(y_true_log, dtype=float).reshape(-1)
    y_pred_log = np.asarray(y_pred_log, dtype=float).reshape(-1)

    if ratio_mode == "diff":
        r = y_pred_log - y_true_log
    else:
        y_true = np.exp(y_true_log)
        y_pred = np.exp(y_pred_log)
        r = y_pred / np.maximum(np.abs(y_true), eps_y)

    mask = np.isfinite(y_true_log) & np.isfinite(r)
    x = y_true_log[mask]
    y = r[mask]

    dcor, sample_n = _distance_correlation_sampled(x, y, max_n=_PAIRWISE_DEPENDENCE_SAMPLE_N)
    nhsic, _ = _normalized_hsic_sampled(x, y, max_n=_PAIRWISE_DEPENDENCE_SAMPLE_N)
    xi = _chatterjee_xi(x, y)

    return {
        "dCor(r,logprice)_sampled": dcor,
        "ChatterjeeXi(r,logprice)": xi,
        "nHSIC(r,logprice)_sampled": nhsic,
        "pairwise_dependence_sample_n": int(sample_n),
    }


def _compute_quick_test_metrics(
    *,
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    y_train_log: np.ndarray,
    ratio_mode: str,
) -> Dict[str, Any]:
    metrics = _compute_extended_metrics(
        y_true_log=y_true_log,
        y_pred_log=y_pred_log,
        y_train_log=y_train_log,
        ratio_mode=ratio_mode,
    )
    metrics.update(
        _compute_logprice_dependence_metrics(
            y_true_log=y_true_log,
            y_pred_log=y_pred_log,
            ratio_mode=ratio_mode,
        )
    )
    return metrics


def _build_quick_test_models(
    *,
    rho_values: List[float],
    lgbm_params: dict,
) -> List[Dict[str, Any]]:
    models: List[Dict[str, Any]] = [
        {
            "model_name": "LinearRegression",
            "model_family": "LinearRegression",
            "rho": np.nan,
            "rho_group": np.nan,
            "estimator": LinearRegression(fit_intercept=True),
            "requires_linear_pipeline": True,
        },
        {
            "model_name": "LGBMRegressor",
            "model_family": "LGBMRegressor",
            "rho": np.nan,
            "rho_group": np.nan,
            "estimator": lgb.LGBMRegressor(**lgbm_params),
            "requires_linear_pipeline": False,
        },
    ]

    rho_list = [float(r) for r in rho_values]
    multi_rho_mode = len(rho_list) > 1

    for rho_value in rho_list:
        models.append(
            {
                "model_name": f"LGBSmoothPenalty_rho_{rho_value}",
                "model_family": "LGBSmoothPenalty",
                "rho": float(rho_value),
                "rho_group": np.nan,
                "estimator": LGBSmoothPenalty(
                    rho=float(rho_value),
                    ratio_mode="diff",
                    zero_grad_tol=1e-12,
                    lgbm_params=lgbm_params,
                    verbose=True,
                ),
                "requires_linear_pipeline": False,
            }
        )
        models.append(
            {
                "model_name": f"LGBCovPenalty_rho_{rho_value}",
                "model_family": "LGBCovPenalty",
                "rho": float(rho_value),
                "rho_group": np.nan,
                "estimator": LGBCovPenalty(
                    rho=float(rho_value),
                    ratio_mode="diff",
                    zero_grad_tol=1e-12,
                    lgbm_params=lgbm_params,
                    verbose=True,
                ),
                "requires_linear_pipeline": False,
            }
        )

    for rho_value in rho_list:
        for rho_group_value in rho_list:
            models.append(
                {
                    "model_name": f"LGBSmoothPenaltyGrouped_rho_{rho_value}_rho_group_{rho_group_value}",
                    "model_family": "LGBSmoothPenaltyGrouped",
                    "rho": float(rho_value),
                    "rho_group": float(rho_group_value),
                    "estimator": LGBSmoothPenaltyGrouped(
                        rho=float(rho_value),
                        rho_group=float(rho_group_value),
                        ratio_mode="diff",
                        group_feature=_META_TOWNSHIP_TRIAD_COL,
                        group_aggregation="mean",
                        min_group_size=2,
                        zero_grad_tol=1e-12,
                        lgbm_params=lgbm_params,
                        verbose=True,
                    ),
                    "requires_linear_pipeline": False,
                }
            )

    if multi_rho_mode:
        return models

    rho_single = float(rho_list[0])
    models.extend(
        [
            {
                "model_name": f"LGBCovPenaltyCVaRTotal_rho_{rho_single}_keep_0.9",
                "model_family": "LGBCovPenaltyCVaRTotal",
                "rho": float(rho_single),
                "rho_group": np.nan,
                "estimator": LGBCovPenaltyCVaRTotal(
                    rho=6.58,
                    mse_keep=0.9,
                    ratio_mode="diff",
                    zero_grad_tol=1e-12,
                    lgbm_params=lgbm_params,
                    verbose=True,
                ),
                "requires_linear_pipeline": False,
            },
            {
                "model_name": f"LGBCovPenaltyCVaRTotal_rho_{rho_single}_keep_0.7",
                "model_family": "LGBCovPenaltyCVaRTotal",
                "rho": float(rho_single),
                "rho_group": np.nan,
                "estimator": LGBCovPenaltyCVaRTotal(
                    rho=6.58,
                    mse_keep=0.7,
                    ratio_mode="diff",
                    zero_grad_tol=1e-12,
                    lgbm_params=lgbm_params,
                    verbose=True,
                ),
                "requires_linear_pipeline": False,
            },
            {
                "model_name": f"LGBCovPenaltyCVaRTotal_rho_{rho_single}_keep_0.5",
                "model_family": "LGBCovPenaltyCVaRTotal",
                "rho": float(rho_single),
                "rho_group": np.nan,
                "estimator": LGBCovPenaltyCVaRTotal(
                    rho=6.58,
                    mse_keep=0.5,
                    ratio_mode="diff",
                    zero_grad_tol=1e-12,
                    lgbm_params=lgbm_params,
                    verbose=True,
                ),
                "requires_linear_pipeline": False,
            },
        ]
    )
    return models


def _write_rho_evolution_plot(
    df: pd.DataFrame,
    *,
    split_label: str,
    out_path: Path,
) -> None:
    if df.empty or "model_family" not in df.columns or "rho" not in df.columns:
        return

    plot_df = df.loc[df["model_family"].isin(_MAIN_SWEEP_FAMILIES), :].copy()
    if plot_df.empty:
        return

    plot_df["rho"] = pd.to_numeric(plot_df["rho"], errors="coerce")
    plot_df = plot_df.loc[np.isfinite(plot_df["rho"]), :]
    if plot_df.empty:
        return

    fig, axes = plt.subplots(1, len(_MAIN_SWEEP_FAMILIES), figsize=(6 * len(_MAIN_SWEEP_FAMILIES), 5), sharey=True)
    if len(_MAIN_SWEEP_FAMILIES) == 1:
        axes = [axes]

    for ax, family in zip(axes, _MAIN_SWEEP_FAMILIES):
        family_df = plot_df.loc[plot_df["model_family"] == family, :].copy()
        if family_df.empty:
            ax.set_visible(False)
            continue

        metric_names = [metric_name for metric_name in _RHO_PLOT_METRICS if metric_name in family_df.columns]
        if family_df["rho"].duplicated().any() and metric_names:
            family_df = (
                family_df.groupby("rho", as_index=False)[metric_names]
                .mean(numeric_only=True)
                .sort_values("rho")
            )
        else:
            family_df = family_df.sort_values("rho")

        for metric_name in metric_names:
            y = pd.to_numeric(family_df[metric_name], errors="coerce")
            if not np.isfinite(y.to_numpy(dtype=float)).any():
                continue
            ax.plot(
                family_df["rho"].to_numpy(dtype=float),
                y.to_numpy(dtype=float),
                marker="o",
                linewidth=1.8,
                linestyle="--",
                label=metric_name,
            )

        ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        if family == "LGBSmoothPenaltyGrouped":
            ax.set_title("LGBSmoothPenaltyGrouped\n(avg over rho_group)")
        else:
            ax.set_title(family)
        ax.set_xlabel("rho")
        ax.grid(True, linestyle=":", alpha=0.4)

    axes[0].set_ylabel("metric value")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Correlation Metric Evolution vs rho ({split_label})")
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.92))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _sanitize_plot_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name).strip())
    return safe or "plot"


def _write_ratio_vs_logprice_plots(
    *,
    split_label: str,
    results_df: pd.DataFrame,
    y_true_log: np.ndarray,
    pred_logs: Dict[str, np.ndarray],
    out_dir: Path,
) -> None:
    if results_df.empty or not pred_logs:
        return

    split_dir = out_dir / split_label.lower()
    split_dir.mkdir(parents=True, exist_ok=True)

    for _, row in results_df.iterrows():
        model_name = str(row.get("model_name", "model"))
        y_pred_log = pred_logs.get(model_name)
        if y_pred_log is None:
            continue
        out_path = split_dir / f"{_sanitize_plot_filename(model_name)}.pdf"
        plot_ratio_vs_logprice(
            y_true_log=y_true_log,
            y_pred_log=y_pred_log,
            out_path=out_path,
            model_label=model_name,
            split_label=split_label,
            metrics=row.to_dict(),
        )


def _fit_predict_and_score(
    *,
    model_name: str,
    estimator: Any,
    requires_linear_pipeline: bool,
    linear_pipeline_builder,
    X_train: pd.DataFrame,
    y_train_log: np.ndarray,
    X_eval: pd.DataFrame,
    y_eval_log: np.ndarray,
    fairness_ratio_mode: str,
    return_prediction_log: bool = False,
) -> Dict[str, Any]:
    if requires_linear_pipeline:
        pipe = linear_pipeline_builder()
        X_train_m = pipe.fit_transform(X_train, y_train_log)
        X_eval_m = pipe.transform(X_eval)
    else:
        X_train_m = X_train
        X_eval_m = X_eval

    estimator.fit(X_train_m, y_train_log)
    y_pred_eval_log = np.asarray(estimator.predict(X_eval_m), dtype=float).reshape(-1)
    metrics = _compute_quick_test_metrics(
        y_true_log=y_eval_log,
        y_pred_log=y_pred_eval_log,
        y_train_log=y_train_log,
        ratio_mode=fairness_ratio_mode,
    )
    out = {"model_name": model_name, **metrics}
    if bool(return_prediction_log):
        out["_y_pred_eval_log"] = y_pred_eval_log
    return out


def run_quick_test(
    *,
    rho: float,
    rho_values: List[float] | None,
    out_dir: str,
    data_path: str,
    sample_frac: float | None,
    seed: int,
    skip_delete_analysis: bool = True,
    n_bootstrap_validation: int = 5,
    bootstrap_block_freq: str = "M",
) -> Dict[str, str]:
    """
    Runs the 4-model quick test and writes the output CSV tables.
    """
    target_column = "meta_sale_price"
    date_column = "meta_sale_date"

    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    with open("model_params.yaml", "r", encoding="utf-8") as f:
        model_params = yaml.safe_load(f)

    df_train_validate, df_test, df_assess, predictor_cols, categorical_cols = _load_and_split_data(
        data_path=data_path,
        params=params,
        target_column=target_column,
        date_column=date_column,
        sample_frac=sample_frac,
        sample_seed=seed,
    )
    df_train_validate = _add_quick_test_grouped_features(df_train_validate)
    df_test = _add_quick_test_grouped_features(df_test)
    df_assess = _add_quick_test_grouped_features(df_assess)

    predictor_cols = list(predictor_cols)
    categorical_cols = list(categorical_cols)
    for engineered_col in (_META_TOWNSHIP_TRIAD_COL, _CHAR_CLASS_BUCKET_COL):
        if engineered_col in df_train_validate.columns and engineered_col not in predictor_cols:
            predictor_cols.append(engineered_col)
        if engineered_col in df_train_validate.columns and engineered_col not in categorical_cols:
            categorical_cols.append(engineered_col)

    # Pipeline builder for linear models (matches `main.py`).
    linear_pipeline_builder = lambda: build_model_pipeline(
        pred_vars=predictor_cols,
        cat_vars=categorical_cols,
        id_vars=params["model"]["predictor"]["id"],
    )

    # Train/eval matrices.
    X_tv = df_train_validate[predictor_cols].copy()
    y_tv_log = np.log(df_train_validate[target_column].to_numpy())

    if not skip_delete_analysis:
        # DELETE
        from sklearn.cluster import AgglomerativeClustering, KMeans
        from sklearn.metrics import normalized_mutual_info_score, silhouette_score
        from sklearn.preprocessing import StandardScaler

        def _cramers_v_from_crosstab(contingency: pd.DataFrame) -> float:
            observed = contingency.to_numpy(dtype=float)
            n_obs = float(observed.sum())
            if n_obs <= 1.0 or observed.shape[0] < 2 or observed.shape[1] < 2:
                return 0.0
            row_sums = observed.sum(axis=1, keepdims=True)
            col_sums = observed.sum(axis=0, keepdims=True)
            expected = (row_sums @ col_sums) / n_obs
            valid = expected > 0.0
            if not np.any(valid):
                return 0.0
            chi2 = float(np.sum(((observed - expected) ** 2 / np.where(valid, expected, 1.0))[valid]))
            phi2 = chi2 / n_obs
            n_rows, n_cols = observed.shape
            phi2_corr = max(0.0, phi2 - ((n_cols - 1.0) * (n_rows - 1.0)) / max(n_obs - 1.0, 1.0))
            n_rows_corr = n_rows - ((n_rows - 1.0) ** 2) / max(n_obs - 1.0, 1.0)
            n_cols_corr = n_cols - ((n_cols - 1.0) ** 2) / max(n_obs - 1.0, 1.0)
            denom = min(n_cols_corr - 1.0, n_rows_corr - 1.0)
            if denom <= 0.0:
                return 0.0
            return float(np.sqrt(phi2_corr / denom))

        def _eta_squared_by_cluster(values: np.ndarray, labels: np.ndarray) -> float:
            if values.size == 0 or np.all(labels == labels[0]):
                return 0.0
            overall_mean = float(np.mean(values))
            ss_total = float(np.sum((values - overall_mean) ** 2))
            if ss_total <= 0.0:
                return 0.0
            ss_between = 0.0
            for label in np.unique(labels):
                cluster_values = values[labels == label]
                if cluster_values.size == 0:
                    continue
                ss_between += float(cluster_values.size) * (float(np.mean(cluster_values)) - overall_mean) ** 2
            return float(np.clip(ss_between / ss_total, 0.0, 1.0))

        analysis_out_dir = Path(out_dir)
        analysis_out_dir.mkdir(parents=True, exist_ok=True)

        max_categorical_levels = 50
        cat_cols = [c for c in categorical_cols if c in X_tv.columns]
        low_card_cat_data: Dict[str, pd.Series] = {}
        cat_cardinality_rows: List[Dict[str, Any]] = []
        for col in cat_cols:
            series = _series_to_string_with_na(X_tv[col])
            n_levels = int(series.nunique(dropna=False))
            cat_cardinality_rows.append({"feature": col, "n_levels": n_levels})
            if 2 <= n_levels <= max_categorical_levels:
                low_card_cat_data[col] = series

        for raw_feature in ("meta_township_code", "char_class"):
            low_card_cat_data.pop(raw_feature, None)
            cat_cardinality_rows = [row for row in cat_cardinality_rows if row["feature"] != raw_feature]

        low_card_cat_cols = list(low_card_cat_data.keys())
        X_cat_raw = pd.DataFrame(low_card_cat_data, index=X_tv.index)
        cat_cardinality_df = pd.DataFrame(cat_cardinality_rows)
        if not cat_cardinality_df.empty:
            cat_cardinality_df = cat_cardinality_df.sort_values(["n_levels", "feature"]).reset_index(drop=True)
        else:
            cat_cardinality_df = pd.DataFrame(columns=["feature", "n_levels"])

        numeric_cols = [
            c for c in X_tv.columns
            if c not in cat_cols and pd.api.types.is_numeric_dtype(X_tv[c])
        ]
        X_num_raw = X_tv[numeric_cols].apply(pd.to_numeric, errors="coerce")
        X_num_imputed = X_num_raw.copy()
        if not X_num_imputed.empty:
            X_num_imputed = X_num_imputed.fillna(X_num_imputed.median())
            X_num_cluster = pd.DataFrame(
                StandardScaler().fit_transform(X_num_imputed),
                columns=numeric_cols,
                index=X_tv.index,
            )
        else:
            X_num_cluster = pd.DataFrame(index=X_tv.index)

        cat_blocks: List[pd.DataFrame] = []
        for col in low_card_cat_cols:
            block = pd.get_dummies(X_cat_raw[col], prefix=col, drop_first=False, dtype=float)
            block = block / np.sqrt(max(block.shape[1], 1))
            cat_blocks.append(block)
        X_cat_cluster = pd.concat(cat_blocks, axis=1) if cat_blocks else pd.DataFrame(index=X_tv.index)

        X_cluster = pd.concat([X_num_cluster, X_cat_cluster], axis=1)
        X_cluster = X_cluster.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
        valid_rows = X_cluster.notna().all(axis=1)
        X_cluster = X_cluster.loc[valid_rows].copy()
        X_num_imputed = X_num_imputed.loc[valid_rows].copy()
        X_num_cluster = X_num_cluster.loc[valid_rows].copy()
        X_cat_raw = X_cat_raw.loc[valid_rows].copy()

        if X_cluster.empty:
            print("Temporary cluster separability analysis could not run: no valid clustering features remained.")
            exit()

        analysis_sample_n = min(10000, X_cluster.shape[0])
        sampled_index = (
            X_cluster.sample(n=analysis_sample_n, random_state=seed).index
            if X_cluster.shape[0] > analysis_sample_n
            else X_cluster.index
        )
        X_cluster_sample = X_cluster.loc[sampled_index].copy()
        X_num_cluster_sample = X_num_cluster.loc[sampled_index].copy()
        X_cat_sample = X_cat_raw.loc[sampled_index].copy()

        print("=" * 90)
        print("TEMPORARY CLUSTER SEPARABILITY ANALYSIS")
        print("=" * 90)
        print(f"rows_available={X_cluster.shape[0]}")
        print(f"rows_sampled={X_cluster_sample.shape[0]}")
        print(f"numeric_features_used={len(numeric_cols)}")
        print(f"low_cardinality_categorical_features_used={len(low_card_cat_cols)}")
        print("Categorical cardinalities:")
        print(cat_cardinality_df.to_string(index=False))

        solution_rows: List[Dict[str, Any]] = []
        categorical_rows: List[Dict[str, Any]] = []
        category_level_rows: List[Dict[str, Any]] = []
        numeric_rows: List[Dict[str, Any]] = []
        cluster_settings = [
            ("kmeans", lambda k: KMeans(n_clusters=k, n_init=20, random_state=seed)),
            ("agglomerative_ward", lambda k: AgglomerativeClustering(n_clusters=k, linkage="ward")),
        ]

        for method_name, cluster_factory in cluster_settings:
            for n_clusters in (2, 3, 4, 5):
                if X_cluster_sample.shape[0] <= n_clusters:
                    continue
                clusterer = cluster_factory(n_clusters)
                labels = clusterer.fit_predict(X_cluster_sample)
                unique_labels, cluster_sizes = np.unique(labels, return_counts=True)
                if unique_labels.size < 2:
                    continue

                silhouette = np.nan
                if X_cluster_sample.shape[0] > unique_labels.size:
                    silhouette = float(
                        silhouette_score(
                            X_cluster_sample,
                            labels,
                            metric="euclidean",
                            sample_size=min(1000, X_cluster_sample.shape[0]),
                            random_state=seed,
                        )
                    )

                solution_rows.append(
                    {
                        "method": method_name,
                        "n_clusters": int(n_clusters),
                        "silhouette": silhouette,
                        "cluster_sizes": ",".join(str(int(v)) for v in cluster_sizes.tolist()),
                    }
                )

                if not X_cat_sample.empty:
                    for col in low_card_cat_cols:
                        feature_values = X_cat_sample[col]
                        contingency = pd.crosstab(feature_values, labels)
                        nmi = float(normalized_mutual_info_score(feature_values, labels))
                        cramers_v = _cramers_v_from_crosstab(contingency)
                        feature_purity = float(contingency.max(axis=1).sum() / contingency.to_numpy().sum())
                        purity_baseline = 1.0 / float(n_clusters)
                        purity_gain = float(
                            np.clip(
                                (feature_purity - purity_baseline) / max(1.0 - purity_baseline, 1e-12),
                                0.0,
                                1.0,
                            )
                        )
                        separation_score = float(np.mean([nmi, cramers_v, purity_gain]))
                        categorical_rows.append(
                            {
                                "method": method_name,
                                "n_clusters": int(n_clusters),
                                "feature": col,
                                "n_levels": int(feature_values.nunique()),
                                "nmi": nmi,
                                "cramers_v": cramers_v,
                                "purity": feature_purity,
                                "purity_gain": purity_gain,
                                "separation_score": separation_score,
                            }
                        )

                        for category_value, counts in contingency.iterrows():
                            counts = counts.astype(float)
                            category_n = float(counts.sum())
                            dominant_cluster = int(counts.idxmax())
                            dominant_share = float(counts.max() / max(category_n, 1.0))
                            dominance_gain = float(
                                np.clip(
                                    (dominant_share - purity_baseline) / max(1.0 - purity_baseline, 1e-12),
                                    0.0,
                                    1.0,
                                )
                            )
                            support_share = float(category_n / X_cluster_sample.shape[0])
                            category_level_rows.append(
                                {
                                    "method": method_name,
                                    "n_clusters": int(n_clusters),
                                    "feature": col,
                                    "category": str(category_value),
                                    "category_n": int(category_n),
                                    "support_share": support_share,
                                    "dominant_cluster": dominant_cluster,
                                    "dominant_share": dominant_share,
                                    "dominance_gain": dominance_gain,
                                    "support_weighted_split_score": float(dominance_gain * np.sqrt(support_share)),
                                }
                            )

                if not X_num_cluster_sample.empty:
                    for col in numeric_cols:
                        eta2 = _eta_squared_by_cluster(X_num_cluster_sample[col].to_numpy(), labels)
                        numeric_rows.append(
                            {
                                "method": method_name,
                                "n_clusters": int(n_clusters),
                                "feature": col,
                                "eta_squared": eta2,
                            }
                        )

        solution_df = pd.DataFrame(solution_rows)
        if not solution_df.empty:
            solution_df = solution_df.sort_values(["method", "n_clusters"]).reset_index(drop=True)
        else:
            solution_df = pd.DataFrame(columns=["method", "n_clusters", "silhouette", "cluster_sizes"])
        categorical_detail_df = pd.DataFrame(categorical_rows)
        category_detail_df = pd.DataFrame(category_level_rows)
        numeric_detail_df = pd.DataFrame(numeric_rows)

        if categorical_detail_df.empty:
            categorical_rank_df = pd.DataFrame(
                columns=["feature", "n_levels", "avg_nmi", "avg_cramers_v", "avg_purity_gain", "avg_separation_score"]
            )
        else:
            categorical_rank_df = (
                categorical_detail_df.groupby("feature", as_index=False)
                .agg(
                    n_levels=("n_levels", "first"),
                    avg_nmi=("nmi", "mean"),
                    avg_cramers_v=("cramers_v", "mean"),
                    avg_purity_gain=("purity_gain", "mean"),
                    avg_separation_score=("separation_score", "mean"),
                    best_separation_score=("separation_score", "max"),
                )
                .sort_values(["avg_separation_score", "best_separation_score", "avg_nmi"], ascending=False)
                .reset_index(drop=True)
            )

        if category_detail_df.empty:
            category_rank_df = pd.DataFrame(
                columns=[
                    "feature",
                    "category",
                    "avg_dominance_gain",
                    "avg_dominant_share",
                    "support_share",
                    "avg_support_weighted_split_score",
                ]
            )
        else:
            category_rank_df = (
                category_detail_df.groupby(["feature", "category"], as_index=False)
                .agg(
                    support_share=("support_share", "mean"),
                    avg_dominant_share=("dominant_share", "mean"),
                    avg_dominance_gain=("dominance_gain", "mean"),
                    avg_support_weighted_split_score=("support_weighted_split_score", "mean"),
                    best_support_weighted_split_score=("support_weighted_split_score", "max"),
                    avg_category_n=("category_n", "mean"),
                )
                .sort_values(
                    ["avg_support_weighted_split_score", "avg_dominance_gain", "support_share"],
                    ascending=False,
                )
                .reset_index(drop=True)
            )

        if numeric_detail_df.empty:
            numeric_rank_df = pd.DataFrame(columns=["feature", "avg_eta_squared", "best_eta_squared"])
        else:
            numeric_rank_df = (
                numeric_detail_df.groupby("feature", as_index=False)
                .agg(
                    avg_eta_squared=("eta_squared", "mean"),
                    best_eta_squared=("eta_squared", "max"),
                )
                .sort_values(["avg_eta_squared", "best_eta_squared"], ascending=False)
                .reset_index(drop=True)
            )

        solution_path = analysis_out_dir / "temp_cluster_solution_summary.csv"
        categorical_path = analysis_out_dir / "temp_cluster_categorical_feature_ranking.csv"
        categorical_detail_path = analysis_out_dir / "temp_cluster_categorical_feature_detail.csv"
        category_path = analysis_out_dir / "temp_cluster_category_ranking.csv"
        category_detail_path = analysis_out_dir / "temp_cluster_category_detail.csv"
        numeric_path = analysis_out_dir / "temp_cluster_numeric_feature_ranking.csv"
        numeric_detail_path = analysis_out_dir / "temp_cluster_numeric_feature_detail.csv"

        solution_df.to_csv(solution_path, index=False)
        categorical_rank_df.to_csv(categorical_path, index=False)
        categorical_detail_df.to_csv(categorical_detail_path, index=False)
        category_rank_df.to_csv(category_path, index=False)
        category_detail_df.to_csv(category_detail_path, index=False)
        numeric_rank_df.to_csv(numeric_path, index=False)
        numeric_detail_df.to_csv(numeric_detail_path, index=False)

        print("\nClustering solutions:")
        print(solution_df.to_string(index=False))
        print("\nCategorical feature ranking:")
        print(categorical_rank_df.head(25).to_string(index=False))
        print("\nMost separating category values:")
        print(category_rank_df.head(25).to_string(index=False))
        print("\nNumeric feature separation context:")
        print(numeric_rank_df.head(25).to_string(index=False))
        print("\nTemporary analysis files written to:")
        print(solution_path)
        print(categorical_path)
        print(categorical_detail_path)
        print(category_path)
        print(category_detail_path)
        print(numeric_path)
        print(numeric_detail_path)

        exit()
        # DELETE

    X_test = df_test[predictor_cols].copy()
    y_test_log = np.log(df_test[target_column].to_numpy())

    X_assess = df_assess[predictor_cols].copy()
    y_assess_log = np.log(df_assess[target_column].to_numpy()) if not df_assess.empty else np.array([], dtype=float)

    # Categorical handling (matches `main.py`).
    cat_cols = [c for c in categorical_cols if c in X_tv.columns]
    for c in cat_cols:
        X_tv[c] = X_tv[c].astype("category")
        X_test[c] = X_test[c].astype("category")
        if not df_assess.empty:
            X_assess[c] = X_assess[c].astype("category")

    # Model parameterization (baseline LGBM defaults).
    lgbm_params = _build_lgbm_params_from_files(model_params=model_params, ccao_params=params, seed=seed)
    rho_sweep = [float(r) for r in (rho_values if rho_values is not None else [rho])]
    models = _build_quick_test_models(
        rho_values=rho_sweep,
        lgbm_params=lgbm_params,
    )

    fairness_ratio_mode = "diff"

    # --- Evaluate on TEST (train on df_train_validate only; strict out-of-time).
    test_rows = []
    test_pred_logs: Dict[str, np.ndarray] = {}
    for spec in models:
        row = _fit_predict_and_score(
            model_name=str(spec["model_name"]),
            estimator=spec["estimator"],
            requires_linear_pipeline=bool(spec["requires_linear_pipeline"]),
            linear_pipeline_builder=linear_pipeline_builder,
            X_train=X_tv,
            y_train_log=y_tv_log,
            X_eval=X_test,
            y_eval_log=y_test_log,
            fairness_ratio_mode=fairness_ratio_mode,
            return_prediction_log=True,
        )
        row["model_family"] = str(spec["model_family"])
        row["rho"] = spec["rho"]
        row["rho_group"] = spec.get("rho_group", np.nan)
        test_pred_logs[str(spec["model_name"])] = np.asarray(row.pop("_y_pred_eval_log"), dtype=float).reshape(-1)
        test_rows.append(row)
    test_df = pd.DataFrame(test_rows)

    # --- Evaluate on ASSESS (train on ALL pre-2024 sales, i.e., train_validate + test).
    assess_df = pd.DataFrame()
    assess_pred_logs: Dict[str, np.ndarray] = {}
    if not df_assess.empty:
        df_pre2024 = pd.concat([df_train_validate, df_test], ignore_index=True)
        X_pre = df_pre2024[predictor_cols].copy()
        y_pre_log = np.log(df_pre2024[target_column].to_numpy())
        for c in cat_cols:
            X_pre[c] = X_pre[c].astype("category")

        assess_rows = []
        for spec in models:
            row = _fit_predict_and_score(
                model_name=str(spec["model_name"]),
                estimator=spec["estimator"],
                requires_linear_pipeline=bool(spec["requires_linear_pipeline"]),
                linear_pipeline_builder=linear_pipeline_builder,
                X_train=X_pre,
                y_train_log=y_pre_log,
                X_eval=X_assess,
                y_eval_log=y_assess_log,
                fairness_ratio_mode=fairness_ratio_mode,
                return_prediction_log=True,
            )
            row["model_family"] = str(spec["model_family"])
            row["rho"] = spec["rho"]
            row["rho_group"] = spec.get("rho_group", np.nan)
            assess_pred_logs[str(spec["model_name"])] = np.asarray(row.pop("_y_pred_eval_log"), dtype=float).reshape(-1)
            assess_rows.append(row)
        assess_df = pd.DataFrame(assess_rows)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    test_path = out / "quick_test_metrics_test.csv"
    assess_path = out / "quick_test_metrics_assess.csv"
    bootstrap_val_path = out / "quick_test_metrics_validation_bootstrap_avg.csv"
    test_df.to_csv(test_path, index=False)
    if not assess_df.empty:
        assess_df.to_csv(assess_path, index=False)

    # --- Small bootstrap summary over the quick-test split (validation-like diagnostic).
    bootstrap_rows: List[Dict[str, Any]] = []
    n_bs = max(0, int(n_bootstrap_validation))
    if n_bs > 0 and y_test_log.size > 1 and test_pred_logs:
        bs_indices = _build_time_block_bootstrap_indices(
            val_dates=pd.to_datetime(df_test[date_column]),
            n_bootstrap=n_bs,
            block_freq=str(bootstrap_block_freq),
            rng_seed=int(seed),
        )
        for model_name, y_pred_log in test_pred_logs.items():
            per_bs: List[Dict[str, Any]] = []
            for sample_idx in bs_indices:
                idx = np.asarray(sample_idx, dtype=int)
                if idx.size < 2:
                    continue
                m = _compute_quick_test_metrics(
                    y_true_log=y_test_log[idx],
                    y_pred_log=y_pred_log[idx],
                    y_train_log=y_tv_log,
                    ratio_mode=fairness_ratio_mode,
                )
                per_bs.append(m)
            if not per_bs:
                continue
            bs_df = pd.DataFrame(per_bs)
            row: Dict[str, Any] = {
                "model_name": str(model_name),
                "n_bootstrap": int(len(per_bs)),
                "bootstrap_block_freq": str(bootstrap_block_freq),
            }
            if not test_df.empty:
                match = test_df.loc[test_df["model_name"] == str(model_name), ["model_family", "rho", "rho_group"]]
                if not match.empty:
                    row["model_family"] = str(match.iloc[0]["model_family"])
                    row["rho"] = match.iloc[0]["rho"]
                    row["rho_group"] = match.iloc[0]["rho_group"]
            for c in bs_df.columns:
                s = pd.to_numeric(bs_df[c], errors="coerce")
                v = s.to_numpy(dtype=float)
                if np.isfinite(v).any():
                    row[c] = float(np.nanmean(v))
            bootstrap_rows.append(row)
    bootstrap_df = pd.DataFrame(bootstrap_rows)
    bootstrap_df.to_csv(bootstrap_val_path, index=False)

    plots_dir = out / "plots"
    rho_plots_dir = plots_dir / "rho_evolution"
    ratio_plots_dir = plots_dir / "ratio_vs_logprice"
    _write_rho_evolution_plot(
        bootstrap_df,
        split_label="Validation bootstrap average",
        out_path=rho_plots_dir / "quick_test_rho_evolution_validation.pdf",
    )
    _write_rho_evolution_plot(
        test_df,
        split_label="Test",
        out_path=rho_plots_dir / "quick_test_rho_evolution_test.pdf",
    )
    if not assess_df.empty:
        _write_rho_evolution_plot(
            assess_df,
            split_label="Assessment",
            out_path=rho_plots_dir / "quick_test_rho_evolution_assess.pdf",
        )

    _write_ratio_vs_logprice_plots(
        split_label="Test",
        results_df=test_df,
        y_true_log=y_test_log,
        pred_logs=test_pred_logs,
        out_dir=ratio_plots_dir,
    )
    if not assess_df.empty:
        _write_ratio_vs_logprice_plots(
            split_label="Assessment",
            results_df=assess_df,
            y_true_log=y_assess_log,
            pred_logs=assess_pred_logs,
            out_dir=ratio_plots_dir,
        )

    return {
        "test_csv": str(test_path),
        "assess_csv": str(assess_path),
        "bootstrap_validation_avg_csv": str(bootstrap_val_path),
        "plots_dir": str(plots_dir),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Quick test: 4 core models on test (~2023) + assessment (2024).")
    p.add_argument("--rho", type=float, default=1.0, help="Rho used for the two regularized models.")
    p.add_argument(
        "--rho-range",
        type=str,
        default="",
        help="Optional comma-separated rho range for LGBSmoothPenalty and LGBCovPenalty in the form min,max. If omitted, uses --rho.",
    )
    p.add_argument("--rho-count", type=int, default=5, help="Number of rho values to generate when --rho-range is provided.")
    p.add_argument(
        "--rho-scale",
        type=str,
        default="log",
        help="Scale for rho sweep when --rho-range is provided. Allowed: linear, log, geom.",
    )
    p.add_argument("--out-dir", type=str, default="./output/quick_test", help="Directory to write CSV outputs.")
    p.add_argument(
        "--data-path",
        type=str,
        default="./data/CCAO/2025/training_data.parquet",
        help="Path to training_data.parquet (same file used by main.py).",
    )
    p.add_argument(
        "--skip-delete-analysis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip the temporary analysis block between the DELETE flags. Use --no-skip-delete-analysis to run it.",
    )
    p.add_argument("--sample-frac", type=float, default=None, help="Optional down-sampling fraction in (0,1].")
    p.add_argument("--seed", type=int, default=4050, help="Random seed (mirrors main.py default).")
    p.add_argument("--n-bootstrap-validation", type=int, default=5, help="Small number of bootstrap resamples for quick validation-like summary.")
    p.add_argument("--bootstrap-block-freq", type=str, default="M", help="Time block frequency for bootstrap resampling (e.g., M, W, Q).")
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    out = run_quick_test(
        rho=float(args.rho),
        rho_values=_build_rho_sweep(
            float(args.rho),
            str(args.rho_range),
            int(args.rho_count),
            str(args.rho_scale),
        ),
        out_dir=str(args.out_dir),
        data_path=str(args.data_path),
        sample_frac=(None if args.sample_frac is None else float(args.sample_frac)),
        seed=int(args.seed),
        skip_delete_analysis=bool(args.skip_delete_analysis),
        n_bootstrap_validation=int(args.n_bootstrap_validation),
        bootstrap_block_freq=str(args.bootstrap_block_freq),
    )
    print("=" * 90)
    print("QUICK TEST COMPLETED")
    print("=" * 90)
    print(f"test_csv={out['test_csv']}")
    print(f"assess_csv={out['assess_csv']}")
    print(f"bootstrap_validation_avg_csv={out['bootstrap_validation_avg_csv']}")
    print(f"plots_dir={out['plots_dir']}")
