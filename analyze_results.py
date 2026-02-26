"""
Analyze CV artifacts: summary tables + validation/test plots.

This is a lightweight, self-contained analysis script intended to work with the
artifact structure produced by `run_temporal_cv.py`:

  output/robust_rolling_origin_cv/
    runs/
    bootstrap_metrics/
    predictions/
    analysis/data_id=.../split_id=.../
      test_metrics.csv
      test_predictions.parquet
      stacking_pf_opt/ (optional)
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402

from utils.motivation_utils import (
    IAAO_PRB_RANGE,
    IAAO_PRD_RANGE,
    IAAO_VEI_RANGE,
    _build_time_block_bootstrap_indices,
    _compute_extended_metrics,
    compute_taxation_metrics,
)
from utils.plotting_utils import plot_vertical_equity_lowess


def _progress_log(message: str, *, t0: Optional[float] = None) -> None:
    if t0 is None:
        print(f"[analyze_results] {message}", flush=True)
        return
    dt = time.time() - float(t0)
    print(f"[analyze_results +{dt:7.1f}s] {message}", flush=True)


def _analysis_dir(result_root: str, data_id: str, split_id: str) -> Path:
    return Path(result_root) / "analysis" / f"data_id={data_id}" / f"split_id={split_id}"


def _load_runs_df(*, result_root: str, data_id: str, split_id: str) -> pd.DataFrame:
    runs_dir = Path(result_root) / "runs" / f"data_id={data_id}" / f"split_id={split_id}"
    paths = sorted(runs_dir.rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No run parquet files found under: {runs_dir}")
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    dfx = df.copy()
    for c in ["R2", "OOS R2", "PRD", "PRB", "VEI", "COD", "Corr(r,price)"]:
        if c in dfx.columns:
            dfx[c] = pd.to_numeric(dfx[c], errors="coerce")
    if "PRD" in dfx.columns:
        dfx["abs_PRD_dev"] = (dfx["PRD"] - 1.0).abs()
    if "PRB" in dfx.columns:
        dfx["abs_PRB_dev"] = dfx["PRB"].abs()
    if "VEI" in dfx.columns:
        dfx["abs_VEI_dev"] = dfx["VEI"].abs()
    if "Corr(r,price)" in dfx.columns:
        dfx["abs_Corr_r_price"] = dfx["Corr(r,price)"].abs()
    return dfx


def _summary_by_config(runs_df: pd.DataFrame) -> pd.DataFrame:
    dfx = _prepare_df(runs_df)
    agg_cols = [
        c
        for c in [
            "R2",
            "OOS R2",
            "PRD",
            "PRB",
            "VEI",
            "Corr(r,price)",
            "abs_PRD_dev",
            "abs_PRB_dev",
            "abs_VEI_dev",
            "abs_Corr_r_price",
        ]
        if c in dfx.columns
    ]
    keys = ["config_id", "model_name"]

    gb = dfx.groupby(keys, as_index=True)[agg_cols]
    mean_df = gb.mean().add_suffix("_mean")
    std_df = gb.std().add_suffix("_std")
    out = mean_df.join(std_df, how="outer").reset_index()
    return out


def _load_bootstrap_df(*, result_root: str, data_id: str, split_id: str) -> pd.DataFrame:
    boots_dir = Path(result_root) / "bootstrap_metrics" / f"data_id={data_id}" / f"split_id={split_id}"
    paths = sorted(boots_dir.rglob("*.parquet"))
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def _load_validation_predictions_df(*, result_root: str, data_id: str, split_id: str) -> pd.DataFrame:
    preds_dir = Path(result_root) / "predictions" / f"data_id={data_id}" / f"split_id={split_id}"
    paths = sorted(preds_dir.rglob("*.parquet"))
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def _load_bootstrap_protocol(*, result_root: str, data_id: str, split_id: str) -> Dict[str, Any]:
    protocol_path = Path(result_root) / "protocol" / f"data_id={data_id}" / f"split_id={split_id}" / "folds.json"
    if not protocol_path.exists():
        return {"n_bootstrap": 0, "block_freq": "M", "seed": 2025}
    try:
        payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    except Exception:
        return {"n_bootstrap": 0, "block_freq": "M", "seed": 2025}
    bp = payload.get("bootstrap_protocol", {}) if isinstance(payload, dict) else {}
    return {
        "n_bootstrap": int(bp.get("n_bootstrap", 0) or 0),
        "block_freq": str(bp.get("block_freq", "M")),
        "seed": int(bp.get("seed", 2025)),
    }


def _bootstrap_metric_columns(df: pd.DataFrame) -> List[str]:
    preferred = [
        "R2",
        "OOS R2",
        "PRD",
        "PRB",
        "VEI",
        "COD",
        "Corr(r,price)",
        "RMSE",
        "MAE",
        "MAPE",
        "MdAPE",
    ]
    return [c for c in preferred if c in df.columns]


def _summarize_bootstrap_statistics(
    df: pd.DataFrame,
    *,
    group_cols: List[str],
    metric_cols: List[str],
    meta_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    if not metric_cols:
        return pd.DataFrame()

    meta_cols = [c for c in (meta_cols or []) if c in df.columns and c not in group_cols]
    out_rows: List[pd.DataFrame] = []
    for metric in metric_cols:
        if metric not in df.columns:
            continue
        tmp = df.loc[:, group_cols + meta_cols + [metric]].copy()
        tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
        tmp = tmp.dropna(subset=[metric])
        if tmp.empty:
            continue
        gb = tmp.groupby(group_cols, dropna=False)[metric]
        agg = gb.agg(
            n_bootstrap="count",
            mean="mean",
            std="std",
            min="min",
            max="max",
        ).reset_index()
        q05 = gb.quantile(0.05).rename("q05").reset_index()
        q25 = gb.quantile(0.25).rename("q25").reset_index()
        q50 = gb.quantile(0.50).rename("q50").reset_index()
        q75 = gb.quantile(0.75).rename("q75").reset_index()
        q95 = gb.quantile(0.95).rename("q95").reset_index()
        stat = agg.merge(q05, on=group_cols, how="left")
        stat = stat.merge(q25, on=group_cols, how="left")
        stat = stat.merge(q50, on=group_cols, how="left")
        stat = stat.merge(q75, on=group_cols, how="left")
        stat = stat.merge(q95, on=group_cols, how="left")
        if meta_cols:
            meta_df = tmp.groupby(group_cols, dropna=False)[meta_cols].first().reset_index()
            stat = stat.merge(meta_df, on=group_cols, how="left")
        stat.insert(len(group_cols), "metric", metric)
        out_rows.append(stat)
    if not out_rows:
        return pd.DataFrame()
    return pd.concat(out_rows, ignore_index=True)


def _average_stats_across_folds(
    df: pd.DataFrame,
    *,
    group_cols: List[str],
    stat_cols: Optional[List[str]] = None,
    fold_col: str = "fold_id",
    skip_first_folds_for_stats: int = 0,
) -> pd.DataFrame:
    """
    Aggregate an existing fold-level bootstrap-stats table into a compact
    across-folds view by averaging each statistic over folds.
    """
    if df.empty:
        return pd.DataFrame()
    dfx = df.copy()
    if fold_col not in dfx.columns:
        return pd.DataFrame()

    if int(skip_first_folds_for_stats) > 0:
        # Apply skip only for numeric fold ids (validation folds). Non-numeric
        # fold ids (e.g., "TEST") are kept to avoid altering test workflows.
        fold_num = pd.to_numeric(dfx[fold_col], errors="coerce")
        dfx = dfx[fold_num.isna() | (fold_num >= int(skip_first_folds_for_stats))].copy()
        if dfx.empty:
            return pd.DataFrame()

    stat_cols = stat_cols or [
        "n_bootstrap",
        "mean",
        "std",
        "min",
        "max",
        "q05",
        "q25",
        "q50",
        "q75",
        "q95",
    ]
    keep_stats = [c for c in stat_cols if c in dfx.columns]
    if not keep_stats:
        return pd.DataFrame()

    keep_groups = [c for c in group_cols if c in dfx.columns and c != fold_col]
    if not keep_groups:
        return pd.DataFrame()

    for c in keep_stats:
        dfx[c] = pd.to_numeric(dfx[c], errors="coerce")

    out = dfx.groupby(keep_groups, dropna=False)[keep_stats].mean().reset_index()
    out = out.rename(columns={c: f"{c}_avg_over_folds" for c in keep_stats})
    n_folds = dfx.groupby(keep_groups, dropna=False)[fold_col].nunique().reset_index(name="n_folds_aggregated")
    out = out.merge(n_folds, on=keep_groups, how="left")

    # Carry simple provenance columns if present and constant per group.
    for meta in ["bootstrap_block_freq", "bootstrap_seed"]:
        if meta in dfx.columns:
            m = dfx.groupby(keep_groups, dropna=False)[meta].first().reset_index()
            out = out.merge(m, on=keep_groups, how="left")
    return out


def _build_test_bootstrap_metrics(
    *,
    analysis_dir: Path,
    protocol: Dict[str, Any],
    config_meta: pd.DataFrame,
) -> pd.DataFrame:
    preds_path = analysis_dir / "test_predictions.parquet"
    meta_path = analysis_dir / "test_eval_metadata.json"
    if (not preds_path.exists()) or (not meta_path.exists()):
        return pd.DataFrame()

    pred_df = pd.read_parquet(preds_path)
    pred_df = pred_df.copy()
    if pred_df.empty:
        return pd.DataFrame()
    required = {"config_id", "row_id", "sale_date", "y_true_log", "y_pred_log"}
    if not required.issubset(set(pred_df.columns)):
        return pd.DataFrame()

    test_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    ratio_mode = str(test_meta.get("fairness_ratio_mode", "diff"))
    y_train_log_mean = float(test_meta.get("y_train_log_mean", np.nan))
    if not np.isfinite(y_train_log_mean):
        return pd.DataFrame()

    n_bootstrap = int(protocol.get("n_bootstrap", 0))
    if n_bootstrap <= 0:
        return pd.DataFrame()
    block_freq = str(protocol.get("block_freq", "M"))
    seed = int(protocol.get("seed", 2025))

    # Common row/frame for block bootstrap indexing.
    ref = (
        pred_df.loc[:, ["row_id", "sale_date", "y_true_log"]]
        .drop_duplicates("row_id")
        .sort_values("row_id")
        .reset_index(drop=True)
    )
    if ref.empty:
        return pd.DataFrame()
    ref["sale_date"] = pd.to_datetime(ref["sale_date"], errors="coerce")
    ref = ref.dropna(subset=["sale_date", "y_true_log"]).reset_index(drop=True)
    if ref.shape[0] < 2:
        return pd.DataFrame()

    row_order = ref["row_id"].to_numpy()
    y_true_log = ref["y_true_log"].to_numpy(dtype=float)
    y_train_log = np.array([y_train_log_mean], dtype=float)
    bs_indices = _build_time_block_bootstrap_indices(
        val_dates=ref["sale_date"],
        n_bootstrap=n_bootstrap,
        block_freq=block_freq,
        rng_seed=seed,
    )
    if not bs_indices:
        return pd.DataFrame()

    pred_wide = pred_df.pivot_table(index="row_id", columns="config_id", values="y_pred_log", aggfunc="first")
    pred_wide = pred_wide.reindex(index=row_order)
    pred_wide.columns = pred_wide.columns.astype(str)

    meta = config_meta.copy()
    meta["config_id"] = meta["config_id"].astype(str)
    rows: List[Dict[str, Any]] = []
    for cfg in pred_wide.columns.tolist():
        y_pred_log = pd.to_numeric(pred_wide[cfg], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(y_pred_log)):
            continue
        mrow = meta.loc[meta["config_id"] == str(cfg)].head(1)
        model_name = str(mrow["model_name"].iloc[0]) if not mrow.empty and "model_name" in mrow.columns else str(cfg)
        rho = float(mrow["rho"].iloc[0]) if (not mrow.empty and "rho" in mrow.columns and np.isfinite(mrow["rho"].iloc[0])) else np.nan
        for b_idx, sample_idx in enumerate(bs_indices):
            y_true_bs = y_true_log[sample_idx]
            y_pred_bs = y_pred_log[sample_idx]
            met = _compute_extended_metrics(
                y_true_log=y_true_bs,
                y_pred_log=y_pred_bs,
                y_train_log=y_train_log,
                ratio_mode=ratio_mode,
            )
            rows.append(
                {
                    "split": "test",
                    "fold_id": "TEST",
                    "config_id": str(cfg),
                    "model_name": model_name,
                    "rho": rho,
                    "bootstrap_id": int(b_idx),
                    "bootstrap_seed": int(seed),
                    "bootstrap_block_freq": block_freq,
                    "bootstrap_sample_size": int(sample_idx.size),
                    **met,
                }
            )
    if not rows:
        return pd.DataFrame()
    return _prepare_df(pd.DataFrame(rows))


def _select_configs_for_evolution(
    runs_df: pd.DataFrame,
    *,
    top_k: Optional[int],
) -> List[str]:
    """
    Select configs to include in evolution plots.

    Strategy:
      - Top-K configs by mean validation accuracy (prefer OOS R2, fallback R2)
      - Always include the two baseline families if present
    """
    dfx = _prepare_df(runs_df)
    acc_col = "OOS R2" if "OOS R2" in dfx.columns else "R2"
    mean_acc = dfx.groupby(["config_id", "model_name"], as_index=False)[acc_col].mean()
    mean_acc = mean_acc.sort_values(acc_col, ascending=False, ignore_index=True)

    k = None if top_k is None else int(top_k)
    if k is not None and k > 0:
        picked = mean_acc.head(k)["config_id"].tolist()
    else:
        picked = mean_acc["config_id"].tolist()

    baselines = dfx.loc[dfx["model_name"].isin(["LinearRegression", "LGBMRegressor"]), "config_id"].unique().tolist()
    out = list(dict.fromkeys(picked + baselines))  # stable unique
    return [str(x) for x in out]


def _config_label_from_runs(runs_df: pd.DataFrame, config_id: str) -> str:
    row = runs_df.loc[runs_df["config_id"] == config_id, ["model_name", "model_config_json"]].drop_duplicates().head(1)
    if row.empty:
        return str(config_id)
    model_name = _short_model_alias(str(row["model_name"].iloc[0]))
    cfg_raw = row["model_config_json"].iloc[0] if "model_config_json" in row.columns else None
    tag = ""
    if isinstance(cfg_raw, str) and cfg_raw.strip():
        try:
            cfg = json.loads(cfg_raw)
        except Exception:
            cfg = {}
        if isinstance(cfg, dict):
            parts = []
            if "rho" in cfg:
                parts.append(f"r={cfg['rho']}")
            if "keep" in cfg:
                parts.append(f"k={cfg['keep']}")
            if parts:
                tag = " (" + ", ".join(parts) + ")"
    return f"{model_name}{tag}"


def _make_model_color_map(model_names: List[str]) -> Dict[str, Any]:
    palette = list(plt.cm.tab20.colors)
    uniq = sorted({str(m) for m in model_names})
    if not uniq:
        return {}
    return {m: palette[i % len(palette)] for i, m in enumerate(uniq)}


def _extract_rho_from_config_json(cfg_raw: Any) -> float:
    if not isinstance(cfg_raw, str) or not cfg_raw.strip():
        return np.nan
    try:
        cfg = json.loads(cfg_raw)
    except Exception:
        return np.nan
    if not isinstance(cfg, dict):
        return np.nan
    try:
        val = float(cfg.get("rho", np.nan))
    except Exception:
        return np.nan
    return val if np.isfinite(val) else np.nan


def _extract_keep_from_config_json(cfg_raw: Any) -> float:
    if not isinstance(cfg_raw, str) or not cfg_raw.strip():
        return np.nan
    try:
        cfg = json.loads(cfg_raw)
    except Exception:
        return np.nan
    if not isinstance(cfg, dict):
        return np.nan
    try:
        val = float(cfg.get("keep", np.nan))
    except Exception:
        return np.nan
    return val if np.isfinite(val) else np.nan


def _short_model_alias(model_name: str) -> str:
    aliases = {
        "LinearRegression": "LR",
        "LGBMRegressor": "LGBM",
        "LGBSmoothPenalty": "LGBSmooth",
        "LGBCovPenalty": "LGBCov",
        "LGBSmoothPenaltyCVaR": "LGBCVaR",
        "LGBSmoothPenaltyCVaRTotal": "LGBCVaRTotal",
        "LGBPrimalDual": "LGBPD",
    }
    return aliases.get(str(model_name), str(model_name))


def _plot_model_name(model_name: Any, cfg_raw: Any) -> str:
    """Keep-aware display key used only for plotting."""
    base = _short_model_alias(str(model_name))
    if base in {"LGBCVaR", "LGBCVaRTotal"}:
        keep = _extract_keep_from_config_json(cfg_raw)
        if np.isfinite(keep):
            return f"{base} k={keep:g}"
    return base


def _build_plot_color_map(model_names: List[str]) -> Dict[str, Any]:
    """
    Distinct plotting colors by displayed model key.
    CVaR keep-level variants get deterministic colors per keep.
    """
    cmap = _make_model_color_map(model_names)
    cmap.update(
        {
            "LR": "royalblue",
            "LGBM": "darkorange",
            "LGBSmooth": "mediumorchid",
            "LGBCov": "sienna",
            "LGBCVaR": "firebrick",
            "LGBCVaRTotal": "teal",
            "LGBPD": "slategray",
        }
    )

    cvar_palette = ["crimson", "tomato", "indianred", "salmon", "darkred"]
    total_palette = ["teal", "cadetblue", "steelblue", "darkcyan", "deepskyblue"]
    cvar_keys = sorted([k for k in cmap if k.startswith("LGBCVaR k=")])
    total_keys = sorted([k for k in cmap if k.startswith("LGBCVaRTotal k=")])
    for i, k in enumerate(cvar_keys):
        cmap[k] = cvar_palette[i % len(cvar_palette)]
    for i, k in enumerate(total_keys):
        cmap[k] = total_palette[i % len(total_palette)]
    return cmap


def _blend_with_white(color: Any, intensity: float) -> Tuple[float, float, float]:
    """
    intensity in [0,1]: 0 => white, 1 => original color.
    """
    rgb = np.asarray(mcolors.to_rgb(color), dtype=float)
    t = float(np.clip(intensity, 0.0, 1.0))
    return tuple(((1.0 - t) * np.ones(3) + t * rgb).tolist())


def _add_iaao_reference_band(ax, metric_col: str, axis: str = "y") -> None:
    """
    Draw objective line + transparent green IAAO acceptance band for fairness axes.
    Supports raw metrics and their absolute-deviation transforms, with or without _mean suffix.
    """
    y_base = str(metric_col).replace("_mean", "")
    band_color = "limegreen"
    line_color = "forestgreen"
    use_x = str(axis).lower() == "x"

    def _draw_band(lo: float, hi: float) -> None:
        if use_x:
            ax.axvspan(min(lo, hi), max(lo, hi), color=band_color, alpha=0.18, label="IAAO band")
        else:
            ax.axhspan(min(lo, hi), max(lo, hi), color=band_color, alpha=0.18, label="IAAO band")

    def _draw_target(v: float) -> None:
        if use_x:
            ax.axvline(v, color=line_color, linewidth=1.2, linestyle="--", label="IAAO target")
        else:
            ax.axhline(v, color=line_color, linewidth=1.2, linestyle="--", label="IAAO target")

    if y_base == "PRD":
        lo, hi = float(IAAO_PRD_RANGE[0]), float(IAAO_PRD_RANGE[1])
        target = 1.0
        _draw_band(lo, hi)
        _draw_target(target)
    elif y_base == "PRB":
        lo, hi = float(IAAO_PRB_RANGE[0]), float(IAAO_PRB_RANGE[1])
        target = 0.0
        _draw_band(lo, hi)
        _draw_target(target)
    elif y_base == "VEI":
        lo, hi = float(IAAO_VEI_RANGE[0]), float(IAAO_VEI_RANGE[1])
        target = 0.0
        _draw_band(lo, hi)
        _draw_target(target)
    elif y_base == "abs_PRD_dev":
        lo = abs(float(IAAO_PRD_RANGE[0]) - 1.0)
        hi = abs(float(IAAO_PRD_RANGE[1]) - 1.0)
        _draw_band(0.0, max(lo, hi))
        _draw_target(0.0)
    elif y_base == "abs_PRB_dev":
        hi = max(abs(float(IAAO_PRB_RANGE[0])), abs(float(IAAO_PRB_RANGE[1])))
        _draw_band(0.0, hi)
        _draw_target(0.0)
    elif y_base == "abs_VEI_dev":
        hi = max(abs(float(IAAO_VEI_RANGE[0])), abs(float(IAAO_VEI_RANGE[1])))
        _draw_band(0.0, hi)
        _draw_target(0.0)


def _ensure_tradeoff_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Ensure derived fairness columns exist when possible.
    Useful for overlay tables that may only include raw metrics.
    """
    dfx = df.copy()
    if col in dfx.columns:
        return dfx

    if col == "abs_PRD_dev" and "PRD" in dfx.columns:
        dfx[col] = (pd.to_numeric(dfx["PRD"], errors="coerce") - 1.0).abs()
    elif col == "abs_PRB_dev" and "PRB" in dfx.columns:
        dfx[col] = pd.to_numeric(dfx["PRB"], errors="coerce").abs()
    elif col == "abs_VEI_dev" and "VEI" in dfx.columns:
        dfx[col] = pd.to_numeric(dfx["VEI"], errors="coerce").abs()
    elif col == "abs_Corr_r_price" and "Corr(r,price)" in dfx.columns:
        dfx[col] = pd.to_numeric(dfx["Corr(r,price)"], errors="coerce").abs()
    elif col == "abs_Corr_r_price_mean" and "Corr(r,price)_mean" in dfx.columns:
        dfx[col] = pd.to_numeric(dfx["Corr(r,price)_mean"], errors="coerce").abs()
    elif col == "abs_PRD_dev_mean" and "PRD_mean" in dfx.columns:
        dfx[col] = (pd.to_numeric(dfx["PRD_mean"], errors="coerce") - 1.0).abs()
    elif col == "abs_PRB_dev_mean" and "PRB_mean" in dfx.columns:
        dfx[col] = pd.to_numeric(dfx["PRB_mean"], errors="coerce").abs()
    elif col == "abs_VEI_dev_mean" and "VEI_mean" in dfx.columns:
        dfx[col] = pd.to_numeric(dfx["VEI_mean"], errors="coerce").abs()

    return dfx


def _bootstrap_summary_long(
    boots_df: pd.DataFrame,
    runs_df: pd.DataFrame,
    *,
    metrics: List[str],
) -> pd.DataFrame:
    """
    Returns long-form summary:
      fold_id, config_id, model_name, metric, mean, std
    computed across bootstrap draws within each fold/config.
    """
    if boots_df.empty:
        return pd.DataFrame()

    dfx = _prepare_df(boots_df)
    # Bootstrap rows don't include model_name; join via run_id or (config_id, fold_id).
    meta_cols = ["run_id", "model_name", "model_config_json"]
    meta = runs_df.loc[:, [c for c in meta_cols if c in runs_df.columns]].drop_duplicates()
    if "run_id" not in meta.columns:
        # fallback: join on fold_id+config_id
        meta = runs_df.loc[:, [c for c in ["config_id", "fold_id", "model_name", "model_config_json"] if c in runs_df.columns]].drop_duplicates()
        dfx = dfx.merge(meta, on=["config_id", "fold_id"], how="left")
    else:
        dfx = dfx.merge(meta, on="run_id", how="left")

    keep_cols = ["fold_id", "config_id", "model_name", "bootstrap_id"] + [m for m in metrics if m in dfx.columns]
    dfx = dfx.loc[:, [c for c in keep_cols if c in dfx.columns]].copy()
    if dfx.empty:
        return pd.DataFrame()

    out_rows: List[pd.DataFrame] = []
    for m in metrics:
        if m not in dfx.columns:
            continue
        g = dfx.groupby(["fold_id", "config_id", "model_name"])[m].agg(mean="mean", std="std").reset_index()
        g["metric"] = m
        out_rows.append(g.loc[:, ["fold_id", "config_id", "model_name", "metric", "mean", "std"]])
    if not out_rows:
        return pd.DataFrame()
    out = pd.concat(out_rows, ignore_index=True)
    out["fold_id"] = pd.to_numeric(out["fold_id"], errors="coerce").astype("Int64")
    out["config_id"] = out["config_id"].astype(str)
    out["model_name"] = out["model_name"].astype(str)
    return out


def _stacking_bootstrap_evolution(
    boots_df: pd.DataFrame,
    *,
    analysis_dir: Path,
    metrics: List[str],
) -> pd.DataFrame:
    """
    If weights exist, compute ensemble metric per bootstrap draw by weighting config-level metrics:
      metric_ens(fold, bs) = Σ_j w_j metric_j(fold, bs)
    Then summarize mean/std across bootstrap draws per fold.

    Returns long form like `_bootstrap_summary_long` but with model_name='STACKING_OPTIMUM_BOOTSTRAP'
    and config_id='STACKING_OPTIMUM_BOOTSTRAP'.
    """
    sdirs = _resolve_stacking_dirs(analysis_dir)
    if not sdirs:
        return pd.DataFrame()
    if boots_df.empty:
        return pd.DataFrame()
    rows: List[pd.DataFrame] = []
    for source_tag, sdir in sdirs:
        weights_path = sdir / "weights_average.csv"
        if not weights_path.exists():
            weights_path = sdir / "weights.csv"
        if not weights_path.exists():
            continue

        wdf = pd.read_csv(weights_path)
        wdf["config_id"] = wdf["config_id"].astype(str)
        wdf["weight"] = pd.to_numeric(wdf["weight"], errors="coerce").fillna(0.0)
        wdf = wdf[wdf["weight"] > 0.0].copy()
        if wdf.empty:
            continue

        dfx = boots_df.copy()
        dfx["config_id"] = dfx["config_id"].astype(str)
        dfx = dfx[dfx["config_id"].isin(wdf["config_id"])].copy()
        if dfx.empty:
            continue

        dfx = dfx.merge(wdf.loc[:, ["config_id", "weight"]], on="config_id", how="inner")
        for m in metrics:
            if m not in dfx.columns:
                continue
            tmp = dfx.loc[:, ["fold_id", "bootstrap_id", "weight", m]].copy()
            tmp[m] = pd.to_numeric(tmp[m], errors="coerce")
            tmp = tmp.dropna(subset=[m])
            if tmp.empty:
                continue
            tmp["weighted"] = tmp["weight"] * tmp[m]
            # ensemble per fold, per bootstrap draw
            ens = tmp.groupby(["fold_id", "bootstrap_id"], as_index=False)["weighted"].sum()
            # summarize across bootstrap draws within fold
            summ = ens.groupby("fold_id")["weighted"].agg(mean="mean", std="std").reset_index()
            summ["metric"] = m
            summ["config_id"] = f"STACKING_OPTIMUM_BOOTSTRAP_{source_tag}"
            summ["model_name"] = f"STACKING_OPTIMUM_BOOTSTRAP_{source_tag}"
            rows.append(summ.loc[:, ["fold_id", "config_id", "model_name", "metric", "mean", "std"]])
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["fold_id"] = pd.to_numeric(out["fold_id"], errors="coerce").astype("Int64")
    return out


def _load_stacking_linear_fold_metrics(*, analysis_dir: Path, metrics: List[str]) -> pd.DataFrame:
    """
    Load deterministic fold-level ensemble metrics from optimizer output (no bootstrap).

    Returns long form with model_name='STACKING_OPTIMUM_LINEAR' and std=NaN.
    """
    sdirs = _resolve_stacking_dirs(analysis_dir)
    if not sdirs:
        return pd.DataFrame()
    rows: List[pd.DataFrame] = []
    rename_map = {
        "R2_ensemble": "R2",
        "PRD_ensemble": "PRD",
        "PRB_ensemble": "PRB",
        "VEI_ensemble": "VEI",
        "COD_ensemble": "COD",
        "OOS R2_ensemble": "OOS R2",
        "Corr(r,price)_ensemble": "Corr(r,price)",
    }

    for source_tag, sdir in sdirs:
        candidates: List[Tuple[str, Path]] = [
            (f"STACKING_OPTIMUM_AVERAGE_{source_tag}", sdir / "fold_ensemble_metrics_average.csv"),
            (f"STACKING_OPTIMUM_WORST_{source_tag}", sdir / "fold_ensemble_metrics_worst.csv"),
            (f"STACKING_OPTIMUM_UTOPIA_{source_tag}", sdir / "fold_ensemble_metrics_utopia.csv"),
            (f"SINGLE_OPTIMUM_REQUESTED_{source_tag}", sdir / "fold_ensemble_metrics_single_requested.csv"),
            (f"SINGLE_OPTIMUM_UTOPIA_{source_tag}", sdir / "fold_ensemble_metrics_single_utopia.csv"),
            (f"STACKING_OPTIMUM_LINEAR_{source_tag}", sdir / "fold_ensemble_metrics.csv"),
        ]
        for model_tag, fold_path in candidates:
            if not fold_path.exists():
                continue
            df = pd.read_csv(fold_path).rename(columns=rename_map)
            df = _prepare_df(df)
            for m in metrics:
                if m not in df.columns:
                    continue
                tmp = df.loc[:, ["fold_id", m]].copy()
                tmp = tmp.rename(columns={m: "mean"})
                tmp["std"] = np.nan
                tmp["metric"] = m
                tmp["config_id"] = model_tag
                tmp["model_name"] = model_tag
                rows.append(tmp.loc[:, ["fold_id", "config_id", "model_name", "metric", "mean", "std"]])

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["fold_id"] = pd.to_numeric(out["fold_id"], errors="coerce").astype("Int64")
    return out


def _plot_metric_evolution(
    evo_long: pd.DataFrame,
    *,
    metric: str,
    title: str,
    out_path: Path,
    color_map: Dict[str, Any],
    config_labels: Dict[str, str],
    skip_first_folds: int = 0,
) -> None:
    dfm = evo_long[evo_long["metric"] == metric].copy()
    if dfm.empty:
        return

    dfm["fold_id"] = pd.to_numeric(dfm["fold_id"], errors="coerce")
    dfm = dfm.dropna(subset=["fold_id", "mean"])
    if skip_first_folds > 0:
        dfm = dfm[dfm["fold_id"] >= skip_first_folds]
    dfm = dfm.sort_values(["fold_id", "model_name", "config_id"], ignore_index=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.grid(True, alpha=0.35)

    # IAAO reference bands (only for the raw fairness metrics)
    if metric == "PRD":
        ax.axhspan(float(IAAO_PRD_RANGE[0]), float(IAAO_PRD_RANGE[1]), color="gray", alpha=0.18, label="IAAO band")
        ax.axhline(1.0, color="gray", linewidth=1.0)
    elif metric == "PRB":
        ax.axhspan(float(IAAO_PRB_RANGE[0]), float(IAAO_PRB_RANGE[1]), color="gray", alpha=0.18, label="IAAO band")
        ax.axhline(0.0, color="gray", linewidth=1.0)
    elif metric == "VEI":
        ax.axhspan(float(IAAO_VEI_RANGE[0]), float(IAAO_VEI_RANGE[1]), color="gray", alpha=0.18, label="IAAO band")
        ax.axhline(0.0, color="gray", linewidth=1.0)

    # Plot each config as a separate curve.
    for (cfg_id, model_name), g in dfm.groupby(["config_id", "model_name"], sort=False):
        x = g["fold_id"].to_numpy(dtype=float)
        y = pd.to_numeric(g["mean"], errors="coerce").to_numpy(dtype=float)
        s = pd.to_numeric(g["std"], errors="coerce").to_numpy(dtype=float) if "std" in g.columns else np.full_like(y, np.nan)

        base_color = color_map.get(str(model_name), "C0")
        label = config_labels.get(str(cfg_id), str(cfg_id))

        is_stacking = str(model_name).startswith("STACKING_OPTIMUM")
        is_single = str(model_name).startswith("SINGLE_OPTIMUM")
        lw = 2.6 if is_stacking else 1.6
        if is_single:
            lw = 2.8
        mname = str(model_name)
        if "LINEAR_LEGACY" in mname:
            ls = ":"
        elif "LINEAR_ROBUST" in mname or mname == "STACKING_OPTIMUM_LINEAR":
            ls = "--"
        elif is_single:
            ls = "-."
        else:
            ls = "-"

        marker = "^" if is_single else None
        ax.plot(x, y, color=base_color, linewidth=lw, linestyle=ls, marker=marker, markersize=5, label=label)
        if np.any(np.isfinite(s)) and "LINEAR" not in str(model_name):
            ax.fill_between(x, y - s, y + s, color=base_color, alpha=0.12)

    ax.set_title(title)
    ax.set_xlabel("fold_id")
    ax.set_ylabel(metric)
    ax.legend(loc="best", frameon=True, ncol=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _compute_2d_pareto_mask(x: np.ndarray, y: np.ndarray, *, maximize_x: bool, minimize_y: bool) -> np.ndarray:
    """
    Returns True for points on the 2D Pareto frontier.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    n = x.size
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            better_or_eq_x = (x[j] >= x[i]) if maximize_x else (x[j] <= x[i])
            better_or_eq_y = (y[j] <= y[i]) if minimize_y else (y[j] >= y[i])
            strictly_better = ((x[j] > x[i]) if maximize_x else (x[j] < x[i])) or ((y[j] < y[i]) if minimize_y else (y[j] > y[i]))
            if better_or_eq_x and better_or_eq_y and strictly_better:
                mask[i] = False
                break
    return mask


def _plot_tradeoff(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    title: str,
    out_path: Path,
    show_top_k: Optional[int] = None,
    overlay_points: Optional[pd.DataFrame] = None,
) -> int:
    dfx = df.copy()
    dfx = dfx[np.isfinite(dfx[x_col]) & np.isfinite(dfx[y_col])].copy()

    x = dfx[x_col].to_numpy(dtype=float)
    y = dfx[y_col].to_numpy(dtype=float)
    # Tradeoff convention: x=fairness deviation (lower is better), y=accuracy (higher is better).
    pareto = _compute_2d_pareto_mask(x, y, maximize_x=False, minimize_y=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(True, alpha=0.35)

    if "model_config_json" in dfx.columns:
        dfx["_plot_model_name"] = [
            _plot_model_name(m, cfg) for m, cfg in zip(dfx["model_name"], dfx["model_config_json"])
        ]
    else:
        dfx["_plot_model_name"] = dfx["model_name"].astype(str)
    model_color_map = _build_plot_color_map(dfx["_plot_model_name"].astype(str).tolist())
    dfx["_rho"] = dfx["model_config_json"].apply(_extract_rho_from_config_json) if "model_config_json" in dfx.columns else np.nan
    dfx["_model_name"] = dfx["_plot_model_name"].astype(str)

    for model_name, g in dfx.groupby("_model_name", sort=True):
        is_baseline = str(model_name) in {"LR", "LGBM"}
        marker = "*" if is_baseline else "o"
        size = 120 if is_baseline else 30
        base_color = model_color_map.get(model_name, "C0")

        rho_vals = pd.to_numeric(g["_rho"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(rho_vals)
        intensities = np.full(g.shape[0], 0.80 if is_baseline else 0.55, dtype=float)
        if np.any(finite):
            # Higher rho -> more intense color; use log scale for stability across decades.
            log_rho = np.log10(np.maximum(rho_vals[finite], 1e-12))
            lo, hi = float(np.nanmin(log_rho)), float(np.nanmax(log_rho))
            if hi > lo:
                norm = (log_rho - lo) / (hi - lo)
            else:
                norm = np.ones_like(log_rho)
            intensities[finite] = 0.35 + 0.65 * norm

        colors = [_blend_with_white(base_color, t) for t in intensities.tolist()]
        ax.scatter(
            g[x_col].to_numpy(dtype=float),
            g[y_col].to_numpy(dtype=float),
            s=size,
            marker=marker,
            c=colors,
            alpha=0.95,
            label=model_name,
        )

    ax.scatter(x[pareto], y[pareto], s=100, facecolors="none", edgecolors="black", linewidths=1.6, label="pareto (2D)")

    _add_iaao_reference_band(ax, x_col, axis="x")

    if overlay_points is not None and not overlay_points.empty:
        o = overlay_points.copy()
        o = _ensure_tradeoff_column(o, x_col)
        o = _ensure_tradeoff_column(o, y_col)
        if (x_col not in o.columns) or (y_col not in o.columns):
            o = pd.DataFrame()
        else:
            o = o[np.isfinite(pd.to_numeric(o[x_col], errors="coerce")) & np.isfinite(pd.to_numeric(o[y_col], errors="coerce"))]
        if not o.empty:
            names = o["model_name"].astype(str) if "model_name" in o.columns else pd.Series(["STACKING_OPTIMUM_AVG"] * len(o), index=o.index)
            o = o.assign(_overlay_name=names)
            for name, og in o.groupby("_overlay_name", sort=True):
                n = str(name)
                n_up = n.upper()
                if n_up.startswith("SINGLE_OPTIMUM_UTOPIA"):
                    marker = "^"
                    if "LEGACY" in n_up:
                        color = "mediumpurple"
                        label = "single optimum (legacy utopia)"
                    else:
                        color = "darkviolet"
                        label = "single optimum (robust utopia)"
                elif n_up.startswith("SINGLE_OPTIMUM_REQUESTED"):
                    marker = "v"
                    if "LEGACY" in n_up:
                        color = "cornflowerblue"
                        label = "single optimum (legacy requested)"
                    else:
                        color = "navy"
                        label = "single optimum (robust requested)"
                elif "UTOPIA" in n_up:
                    marker = "D"
                    if "LEGACY" in n_up:
                        color = "slateblue"
                        label = "stacking optimum (legacy utopia)"
                    else:
                        color = "darkcyan"
                        label = "stacking optimum (robust utopia)"
                elif "WORST" in n_up:
                    marker = "X"
                    if "LEGACY" in n_up:
                        color = "purple"
                        label = "stacking optimum (legacy worst-block)"
                    else:
                        color = "firebrick"
                        label = "stacking optimum (robust worst-block)"
                else:
                    marker = "P"
                    if "LEGACY" in n_up:
                        color = "dodgerblue"
                        label = "stacking optimum (legacy average)"
                    else:
                        color = "goldenrod"
                        label = "stacking optimum (robust average)"
                ax.scatter(
                    og[x_col].to_numpy(dtype=float),
                    og[y_col].to_numpy(dtype=float),
                    s=150,
                    marker=marker,
                    color=color,
                    alpha=0.95,
                    label=label,
                )

    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend(loc="best", frameon=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return int(dfx.shape[0])


def _pick_validation_single_run_fold_id(runs_df: pd.DataFrame) -> int:
    dfx = _prepare_df(runs_df)
    acc_col = "OOS R2" if "OOS R2" in dfx.columns else "R2"
    fold_mean = dfx.groupby("fold_id")[acc_col].mean()
    return int(fold_mean.idxmin())


def _resolve_stacking_dirs(analysis_dir: Path) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    for tag, name in [("ROBUST", "stacking_prd_socp_opt"), ("LEGACY", "stacking_pf_opt")]:
        p = analysis_dir / name
        if p.exists():
            out.append((tag, p))
    return out


def _resolve_stacking_dir(analysis_dir: Path) -> Optional[Path]:
    dirs = _resolve_stacking_dirs(analysis_dir)
    return dirs[0][1] if dirs else None


def _load_stacking_validation_overlay(*, analysis_dir: Path, fold_id: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sdirs = _resolve_stacking_dirs(analysis_dir)
    if not sdirs:
        return pd.DataFrame(), pd.DataFrame()

    avg_rows: List[pd.DataFrame] = []
    single_rows: List[pd.DataFrame] = []
    for source_tag, sdir in sdirs:
        candidates: List[Tuple[str, Path]] = [
            (f"STACKING_OPTIMUM_AVG_{source_tag}", sdir / "fold_ensemble_metrics_average.csv"),
            (f"STACKING_OPTIMUM_UTOPIA_{source_tag}", sdir / "fold_ensemble_metrics_utopia.csv"),
            (f"SINGLE_OPTIMUM_REQUESTED_{source_tag}", sdir / "fold_ensemble_metrics_single_requested.csv"),
            (f"SINGLE_OPTIMUM_UTOPIA_{source_tag}", sdir / "fold_ensemble_metrics_single_utopia.csv"),
        ]
        if not (sdir / "fold_ensemble_metrics_average.csv").exists():
            candidates.append((f"STACKING_OPTIMUM_AVG_{source_tag}", sdir / "fold_ensemble_metrics.csv"))
        for model_tag, fold_path in candidates:
            if not fold_path.exists():
                continue
            fold_df = pd.read_csv(fold_path)
            fold_df = _prepare_df(
                fold_df.rename(
                    columns={
                        "OOS R2_ensemble": "OOS R2",
                        "R2_ensemble": "R2",
                        "PRD_ensemble": "PRD",
                        "PRB_ensemble": "PRB",
                        "VEI_ensemble": "VEI",
                        "COD_ensemble": "COD",
                        "Corr(r,price)_ensemble": "Corr(r,price)",
                    }
                )
            )
            avg = fold_df.drop(columns=["fold_id"], errors="ignore").mean(numeric_only=True).to_frame().T
            avg.columns = [f"{c}_mean" for c in avg.columns]  # match column names used in validation avg plots
            avg["model_name"] = model_tag
            avg_rows.append(avg)
            single = fold_df.loc[fold_df["fold_id"] == int(fold_id)].copy()
            if not single.empty:
                single["model_name"] = model_tag
                single_rows.append(single)
    return (
        pd.concat(avg_rows, ignore_index=True) if avg_rows else pd.DataFrame(),
        pd.concat(single_rows, ignore_index=True) if single_rows else pd.DataFrame(),
    )


def _load_stacking_validation_worst_overlay(*, analysis_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build validation overlays for a robust (worst-fold) stacking point:
      - avg-style row with *_mean columns (for validation average tradeoffs)
      - single-style raw row (for single-fold tradeoffs)
    """
    sdirs = _resolve_stacking_dirs(analysis_dir)
    if not sdirs:
        return pd.DataFrame(), pd.DataFrame()

    worst_avg_rows: List[pd.DataFrame] = []
    worst_single_rows: List[pd.DataFrame] = []
    for source_tag, sdir in sdirs:
        fold_path = sdir / "fold_ensemble_metrics_worst.csv"
        if not fold_path.exists():
            # legacy fallback (derive "worst" from single stored curve)
            fold_path = sdir / "fold_ensemble_metrics.csv"
        if not fold_path.exists():
            continue

        fold_df = pd.read_csv(fold_path)
        fold_df = _prepare_df(
            fold_df.rename(
                columns={
                    "OOS R2_ensemble": "OOS R2",
                    "R2_ensemble": "R2",
                    "PRD_ensemble": "PRD",
                    "PRB_ensemble": "PRB",
                    "VEI_ensemble": "VEI",
                    "COD_ensemble": "COD",
                    "Corr(r,price)_ensemble": "Corr(r,price)",
                }
            )
        )
        if fold_df.empty:
            continue

        acc_col = "OOS R2" if "OOS R2" in fold_df.columns else "R2"
        fold_df[acc_col] = pd.to_numeric(fold_df[acc_col], errors="coerce")
        fold_df = fold_df[np.isfinite(fold_df[acc_col])].copy()
        if fold_df.empty:
            continue

        worst_idx = int(fold_df[acc_col].idxmin())
        worst_single = fold_df.loc[[worst_idx]].copy()
        worst_single["model_name"] = f"STACKING_OPTIMUM_WORST_BLOCK_{source_tag}"
        worst_single_rows.append(worst_single)

        worst_avg = worst_single.drop(columns=["fold_id"], errors="ignore").copy()
        worst_avg.columns = [f"{c}_mean" if c != "model_name" else c for c in worst_avg.columns]
        worst_avg_rows.append(worst_avg)
    return (
        pd.concat(worst_avg_rows, ignore_index=True) if worst_avg_rows else pd.DataFrame(),
        pd.concat(worst_single_rows, ignore_index=True) if worst_single_rows else pd.DataFrame(),
    )


def _compute_test_stacking_overlay(
    *,
    analysis_dir: Path,
    y_train_log_mean: float,
    worst_block_freq: str = "Q",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sdirs = _resolve_stacking_dirs(analysis_dir)
    if not sdirs:
        return pd.DataFrame(), pd.DataFrame()
    preds_path = analysis_dir / "test_predictions.parquet"
    if not preds_path.exists():
        return pd.DataFrame(), pd.DataFrame()

    avg_rows: List[pd.DataFrame] = []
    worst_rows: List[pd.DataFrame] = []
    for source_tag, sdir in sdirs:
        weights_avg_path = sdir / "weights_average.csv"
        if not weights_avg_path.exists():
            weights_avg_path = sdir / "weights.csv"
        if not weights_avg_path.exists():
            continue

        pdf = pd.read_parquet(preds_path)
        if pdf.empty:
            continue
        y_train_log = np.array([float(y_train_log_mean)], dtype=float)

        avg_like_specs: List[Tuple[str, Path]] = [(f"STACKING_OPTIMUM_AVG_{source_tag}", weights_avg_path)]
        weights_utopia_path = sdir / "weights_utopia.csv"
        if weights_utopia_path.exists():
            avg_like_specs.append((f"STACKING_OPTIMUM_UTOPIA_{source_tag}", weights_utopia_path))
        weights_single_requested_path = sdir / "weights_single_requested.csv"
        if weights_single_requested_path.exists():
            avg_like_specs.append((f"SINGLE_OPTIMUM_REQUESTED_{source_tag}", weights_single_requested_path))
        weights_single_utopia_path = sdir / "weights_single_utopia.csv"
        if weights_single_utopia_path.exists():
            avg_like_specs.append((f"SINGLE_OPTIMUM_UTOPIA_{source_tag}", weights_single_utopia_path))

        y_pred_stack_log = None
        y_true_log = None
        sale_date = None
        for model_tag, wpath in avg_like_specs:
            wdf = pd.read_csv(wpath)
            wdf = wdf[pd.to_numeric(wdf["weight"], errors="coerce").fillna(0.0) > 0.0].copy()
            if wdf.empty:
                continue
            pdf_w = pdf[pdf["config_id"].isin(wdf["config_id"])].copy()
            if pdf_w.empty:
                continue

            mat = pdf_w.pivot_table(index="row_id", columns="config_id", values="y_pred_log", aggfunc="first")
            mat = mat.reindex(columns=wdf["config_id"].tolist())
            if mat.isna().any().any():
                mat = mat.dropna(axis=0)
            if mat.empty:
                continue

            w = wdf.set_index("config_id")["weight"].reindex(mat.columns).to_numpy(dtype=float)
            w = np.maximum(w, 0.0)
            if float(np.sum(w)) <= 0:
                continue
            w = w / np.sum(w)

            y_pred_log_cur = mat.to_numpy(dtype=float) @ w
            y_true_log_cur = (
                pdf_w.drop_duplicates("row_id")
                .set_index("row_id")
                .reindex(mat.index)["y_true_log"]
                .to_numpy(dtype=float)
            )
            sale_date_cur = (
                pd.to_datetime(
                    pdf_w.drop_duplicates("row_id")
                    .set_index("row_id")
                    .reindex(mat.index)["sale_date"]
                )
            )
            avg_metrics = _compute_extended_metrics(
                y_true_log=y_true_log_cur, y_pred_log=y_pred_log_cur, y_train_log=y_train_log, ratio_mode="diff"
            )
            avg_rows.append(_prepare_df(pd.DataFrame([{**avg_metrics, "model_name": model_tag}])))

            if model_tag.startswith("STACKING_OPTIMUM_AVG_"):
                y_pred_stack_log = y_pred_log_cur
                y_true_log = y_true_log_cur
                sale_date = sale_date_cur

        # Worst temporal block on test (by OOS R2 if present else R2)
        if y_pred_stack_log is None or y_true_log is None or sale_date is None:
            continue
        blocks = sale_date.dt.to_period(worst_block_freq).astype(str)
        acc_key = "OOS R2" if "OOS R2" in avg_metrics else "R2"
        worst_row = None
        worst_acc = None
        for b in pd.unique(blocks):
            m = blocks == b
            if int(np.sum(m)) < 2:
                continue
            met = _compute_extended_metrics(y_true_log=y_true_log[m], y_pred_log=y_pred_stack_log[m], y_train_log=y_train_log, ratio_mode="diff")
            acc = float(met.get(acc_key, np.nan))
            if worst_acc is None or (np.isfinite(acc) and acc < worst_acc):
                worst_acc = acc
                worst_row = {**met, "model_name": f"STACKING_OPTIMUM_WORST_BLOCK_{source_tag}", "block": str(b)}
        worst_df = _prepare_df(pd.DataFrame([worst_row])) if worst_row is not None else pd.DataFrame()

        # If a dedicated robust-worst optimization exists, compute that on test too.
        robust_weights_path = sdir / "weights_worst.csv"
        if robust_weights_path.exists():
            try:
                rw = pd.read_csv(robust_weights_path)
                rw = rw[pd.to_numeric(rw.get("weight", np.nan), errors="coerce").fillna(0.0) > 0.0].copy()
                if not rw.empty:
                    mat_r = pdf.pivot_table(index="row_id", columns="config_id", values="y_pred_log", aggfunc="first")
                    mat_r = mat_r.reindex(columns=rw["config_id"].tolist())
                    if mat_r.isna().any().any():
                        mat_r = mat_r.dropna(axis=0)
                    wr = rw.set_index("config_id")["weight"].reindex(mat_r.columns).to_numpy(dtype=float)
                    wr = np.maximum(wr, 0.0)
                    wr = wr / wr.sum()
                    y_pred_r = mat_r.to_numpy(dtype=float) @ wr
                    y_true_r = (
                        pdf.drop_duplicates("row_id")
                        .set_index("row_id")
                        .reindex(mat_r.index)["y_true_log"]
                        .to_numpy(dtype=float)
                    )
                    met_r = _compute_extended_metrics(y_true_log=y_true_r, y_pred_log=y_pred_r, y_train_log=y_train_log, ratio_mode="diff")
                    robust_row = _prepare_df(pd.DataFrame([{**met_r, "model_name": f"STACKING_OPTIMUM_WORST_{source_tag}"}]))
                    if worst_df.empty:
                        worst_df = robust_row
                    else:
                        # Keep both temporal-worst-block and robust-worst weights evaluations.
                        worst_df = pd.concat([worst_df, robust_row], ignore_index=True)
            except Exception:
                pass
        if not worst_df.empty:
            worst_rows.append(worst_df)

    return (
        pd.concat(avg_rows, ignore_index=True) if avg_rows else pd.DataFrame(),
        pd.concat(worst_rows, ignore_index=True) if worst_rows else pd.DataFrame(),
    )


def _build_stacking_optima_summary(
    *,
    val_avg: pd.DataFrame,
    val_worst_avg: pd.DataFrame,
    val_single: pd.DataFrame,
    val_worst_single: pd.DataFrame,
    test_avg: pd.DataFrame,
    test_worst: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a compact table with available stacking optima across validation/test views.
    """

    def _pick_metric_cols(df: pd.DataFrame) -> List[str]:
        preferred = [
            "R2",
            "OOS R2",
            "PRD",
            "PRB",
            "VEI",
            "COD",
            "Corr(r,price)",
            "abs_PRD_dev",
            "abs_PRB_dev",
            "abs_VEI_dev",
            "abs_Corr_r_price",
            "R2_mean",
            "OOS R2_mean",
            "PRD_mean",
            "PRB_mean",
            "VEI_mean",
            "COD_mean",
            "Corr(r,price)_mean",
            "abs_PRD_dev_mean",
            "abs_PRB_dev_mean",
            "abs_VEI_dev_mean",
            "abs_Corr_r_price_mean",
        ]
        return [c for c in preferred if c in df.columns]

    rows: List[pd.DataFrame] = []
    sources = [
        ("validation_avg", val_avg),
        ("validation_avg_worst", val_worst_avg),
        ("validation_single", val_single),
        ("validation_single_worst", val_worst_single),
        ("test_avg", test_avg),
        ("test_worst", test_worst),
    ]
    for source_name, df in sources:
        if df is None or df.empty:
            continue
        dfx = _prepare_df(df.copy())
        keep = [c for c in ["model_name", "block"] if c in dfx.columns] + _pick_metric_cols(dfx)
        if not keep:
            continue
        out = dfx.loc[:, keep].copy()
        out.insert(0, "source", source_name)
        rows.append(out)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _build_final_model_comparison_taxation_metrics(
    *,
    result_root: str,
    analysis_dir: Path,
    data_id: str,
    split_id: str,
    runs_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a compact table of all compute_taxation_metrics outputs for:
      - Baselines: LinearRegression, LGBMRegressor
      - Final stacking solutions available in stacking output dirs
    across:
      - validation_avg (pooled validation predictions across folds)
      - test (held-out test predictions)
    """
    val_preds = _load_validation_predictions_df(result_root=result_root, data_id=data_id, split_id=split_id)
    test_preds_path = analysis_dir / "test_predictions.parquet"
    test_preds = pd.read_parquet(test_preds_path) if test_preds_path.exists() else pd.DataFrame()

    if val_preds.empty and test_preds.empty:
        return pd.DataFrame()

    runs_meta = (
        runs_df.loc[:, [c for c in ["run_id", "config_id", "model_name", "fold_id"] if c in runs_df.columns]]
        .drop_duplicates(subset=["run_id"])
        .copy()
    )
    if not val_preds.empty and "run_id" in val_preds.columns and not runs_meta.empty:
        val_preds = val_preds.merge(runs_meta, on="run_id", how="left")
    for dfx in [val_preds, test_preds]:
        if not dfx.empty and "config_id" in dfx.columns:
            dfx["config_id"] = dfx["config_id"].astype(str)

    dfx_runs = _prepare_df(runs_df)
    acc_col = "OOS R2" if "OOS R2" in dfx_runs.columns else "R2"
    baseline_cfg: Dict[str, str] = {}
    for m in ["LinearRegression", "LGBMRegressor"]:
        sub = dfx_runs[dfx_runs["model_name"].astype(str) == m].copy()
        if sub.empty:
            continue
        best = (
            sub.groupby("config_id", as_index=False)[acc_col]
            .mean()
            .sort_values(acc_col, ascending=False, ignore_index=True)
            .head(1)
        )
        if not best.empty:
            baseline_cfg[m] = str(best["config_id"].iloc[0])

    meta_path = analysis_dir / "test_eval_metadata.json"
    y_train_test = None
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            y_train_log_mean = float(meta.get("y_train_log_mean", np.nan))
            if np.isfinite(y_train_log_mean):
                y_train_test = np.array([float(np.exp(y_train_log_mean))], dtype=float)
        except Exception:
            y_train_test = None

    def _metrics_from_arrays(y_true: np.ndarray, y_pred: np.ndarray, y_train_ref: Optional[np.ndarray]) -> Dict[str, float]:
        y_true = np.asarray(y_true, dtype=float).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
        ok = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0.0) & (y_pred > 0.0)
        y_true = y_true[ok]
        y_pred = y_pred[ok]
        if y_true.size < 2:
            return {}
        if y_train_ref is None:
            y_train_ref = y_true
        try:
            out = compute_taxation_metrics(y_real=y_true, y_pred=y_pred, scale="price", y_train=np.asarray(y_train_ref, dtype=float))
            return {str(k): float(v) for k, v in out.items() if np.isscalar(v) and np.isfinite(v)}
        except Exception:
            return {}

    rows: List[Dict[str, Any]] = []

    # Baseline rows
    for model_name, cfg_id in baseline_cfg.items():
        if not val_preds.empty and {"config_id", "y_true", "y_pred"}.issubset(set(val_preds.columns)):
            v = val_preds[val_preds["config_id"] == str(cfg_id)].copy()
            mv = _metrics_from_arrays(v["y_true"].to_numpy(), v["y_pred"].to_numpy(), None)
            if mv:
                rows.append({"split": "validation_avg", "solution": str(model_name), "family": "baseline", "config_id": str(cfg_id), **mv})
        if not test_preds.empty and {"config_id", "y_true", "y_pred"}.issubset(set(test_preds.columns)):
            t = test_preds[test_preds["config_id"] == str(cfg_id)].copy()
            mt = _metrics_from_arrays(t["y_true"].to_numpy(), t["y_pred"].to_numpy(), y_train_test)
            if mt:
                rows.append({"split": "test", "solution": str(model_name), "family": "baseline", "config_id": str(cfg_id), **mt})

    # Stacking rows from available weights files
    for source_tag, sdir in _resolve_stacking_dirs(analysis_dir):
        weight_specs: List[Tuple[str, Path]] = []
        p_avg = sdir / "weights_average.csv"
        if p_avg.exists():
            weight_specs.append((f"STACKING_OPTIMUM_AVG_{source_tag}", p_avg))
        elif (sdir / "weights.csv").exists():
            weight_specs.append((f"STACKING_OPTIMUM_AVG_{source_tag}", sdir / "weights.csv"))
        p_uto = sdir / "weights_utopia.csv"
        if p_uto.exists():
            weight_specs.append((f"STACKING_OPTIMUM_UTOPIA_{source_tag}", p_uto))
        p_worst = sdir / "weights_worst.csv"
        if p_worst.exists():
            weight_specs.append((f"STACKING_OPTIMUM_WORST_{source_tag}", p_worst))
        p_single_req = sdir / "weights_single_requested.csv"
        if p_single_req.exists():
            weight_specs.append((f"SINGLE_OPTIMUM_REQUESTED_{source_tag}", p_single_req))
        p_single_uto = sdir / "weights_single_utopia.csv"
        if p_single_uto.exists():
            weight_specs.append((f"SINGLE_OPTIMUM_UTOPIA_{source_tag}", p_single_uto))
        if source_tag == "LEGACY" and (sdir / "weights.csv").exists():
            weight_specs.append((f"STACKING_OPTIMUM_LINEAR_{source_tag}", sdir / "weights.csv"))

        for sol_name, wpath in weight_specs:
            wdf = pd.read_csv(wpath)
            if "config_id" not in wdf.columns or "weight" not in wdf.columns:
                continue
            wdf["config_id"] = wdf["config_id"].astype(str)
            wdf["weight"] = pd.to_numeric(wdf["weight"], errors="coerce").fillna(0.0)
            wdf = wdf[wdf["weight"] > 0.0].copy()
            if wdf.empty:
                continue

            if not val_preds.empty and {"config_id", "run_id", "row_id", "y_true", "y_pred"}.issubset(set(val_preds.columns)):
                pv = val_preds[val_preds["config_id"].isin(wdf["config_id"])].copy()
                if not pv.empty:
                    mat = pv.pivot_table(index=["run_id", "row_id"], columns="config_id", values="y_pred", aggfunc="first")
                    mat = mat.reindex(columns=wdf["config_id"].tolist())
                    mat = mat.dropna(axis=0, how="any")
                    if not mat.empty:
                        w = wdf.set_index("config_id")["weight"].reindex(mat.columns).to_numpy(dtype=float)
                        w = np.maximum(w, 0.0)
                        if float(np.sum(w)) > 0:
                            w = w / np.sum(w)
                            y_hat = mat.to_numpy(dtype=float) @ w
                            y_true = (
                                pv.drop_duplicates(subset=["run_id", "row_id"])
                                .set_index(["run_id", "row_id"])
                                .reindex(mat.index)["y_true"]
                                .to_numpy(dtype=float)
                            )
                            mv = _metrics_from_arrays(y_true, y_hat, None)
                            if mv:
                                fam = "single" if str(sol_name).startswith("SINGLE_OPTIMUM") else "stacking"
                                rows.append({"split": "validation_avg", "solution": sol_name, "family": fam, "config_id": "", **mv})

            if not test_preds.empty and {"config_id", "row_id", "y_true", "y_pred"}.issubset(set(test_preds.columns)):
                pt = test_preds[test_preds["config_id"].isin(wdf["config_id"])].copy()
                if not pt.empty:
                    mat = pt.pivot_table(index="row_id", columns="config_id", values="y_pred", aggfunc="first")
                    mat = mat.reindex(columns=wdf["config_id"].tolist())
                    mat = mat.dropna(axis=0, how="any")
                    if not mat.empty:
                        w = wdf.set_index("config_id")["weight"].reindex(mat.columns).to_numpy(dtype=float)
                        w = np.maximum(w, 0.0)
                        if float(np.sum(w)) > 0:
                            w = w / np.sum(w)
                            y_hat = mat.to_numpy(dtype=float) @ w
                            y_true = (
                                pt.drop_duplicates("row_id")
                                .set_index("row_id")
                                .reindex(mat.index)["y_true"]
                                .to_numpy(dtype=float)
                            )
                            mt = _metrics_from_arrays(y_true, y_hat, y_train_test)
                            if mt:
                                fam = "single" if str(sol_name).startswith("SINGLE_OPTIMUM") else "stacking"
                                rows.append({"split": "test", "solution": sol_name, "family": fam, "config_id": "", **mt})

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    pref = ["split", "family", "solution", "config_id"]
    other = [c for c in out.columns if c not in pref]
    return out.loc[:, pref + other].copy()


def _collect_final_solution_predictions_for_vertical_equity(
    *,
    result_root: str,
    analysis_dir: Path,
    data_id: str,
    split_id: str,
    runs_df: pd.DataFrame,
) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """
    Collect (y_true, y_pred) arrays for:
      - validation_avg
      - test
    for baselines and final stacking solutions.
    """
    out: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {"validation_avg": {}, "test": {}}
    val_preds = _load_validation_predictions_df(result_root=result_root, data_id=data_id, split_id=split_id)
    test_preds_path = analysis_dir / "test_predictions.parquet"
    test_preds = pd.read_parquet(test_preds_path) if test_preds_path.exists() else pd.DataFrame()
    if val_preds.empty and test_preds.empty:
        return out

    runs_meta = (
        runs_df.loc[:, [c for c in ["run_id", "config_id", "model_name", "fold_id"] if c in runs_df.columns]]
        .drop_duplicates(subset=["run_id"])
        .copy()
    )
    if not val_preds.empty and "run_id" in val_preds.columns and not runs_meta.empty:
        val_preds = val_preds.merge(runs_meta, on="run_id", how="left")
    for dfx in [val_preds, test_preds]:
        if not dfx.empty and "config_id" in dfx.columns:
            dfx["config_id"] = dfx["config_id"].astype(str)

    dfx_runs = _prepare_df(runs_df)
    acc_col = "OOS R2" if "OOS R2" in dfx_runs.columns else "R2"
    baseline_cfg: Dict[str, str] = {}
    for m in ["LinearRegression", "LGBMRegressor"]:
        sub = dfx_runs[dfx_runs["model_name"].astype(str) == m].copy()
        if sub.empty:
            continue
        best = (
            sub.groupby("config_id", as_index=False)[acc_col]
            .mean()
            .sort_values(acc_col, ascending=False, ignore_index=True)
            .head(1)
        )
        if not best.empty:
            baseline_cfg[m] = str(best["config_id"].iloc[0])

    for model_name, cfg_id in baseline_cfg.items():
        if not val_preds.empty and {"config_id", "y_true", "y_pred"}.issubset(set(val_preds.columns)):
            v = val_preds[val_preds["config_id"] == str(cfg_id)].copy()
            if not v.empty:
                out["validation_avg"][model_name] = (
                    v["y_true"].to_numpy(dtype=float),
                    v["y_pred"].to_numpy(dtype=float),
                )
        if not test_preds.empty and {"config_id", "y_true", "y_pred"}.issubset(set(test_preds.columns)):
            t = test_preds[test_preds["config_id"] == str(cfg_id)].copy()
            if not t.empty:
                out["test"][model_name] = (
                    t["y_true"].to_numpy(dtype=float),
                    t["y_pred"].to_numpy(dtype=float),
                )

    for source_tag, sdir in _resolve_stacking_dirs(analysis_dir):
        weight_specs: List[Tuple[str, Path]] = []
        p_avg = sdir / "weights_average.csv"
        if p_avg.exists():
            weight_specs.append((f"STACKING_OPTIMUM_AVG_{source_tag}", p_avg))
        elif (sdir / "weights.csv").exists():
            weight_specs.append((f"STACKING_OPTIMUM_AVG_{source_tag}", sdir / "weights.csv"))
        p_uto = sdir / "weights_utopia.csv"
        if p_uto.exists():
            weight_specs.append((f"STACKING_OPTIMUM_UTOPIA_{source_tag}", p_uto))
        p_worst = sdir / "weights_worst.csv"
        if p_worst.exists():
            weight_specs.append((f"STACKING_OPTIMUM_WORST_{source_tag}", p_worst))
        p_single_req = sdir / "weights_single_requested.csv"
        if p_single_req.exists():
            weight_specs.append((f"SINGLE_OPTIMUM_REQUESTED_{source_tag}", p_single_req))
        p_single_uto = sdir / "weights_single_utopia.csv"
        if p_single_uto.exists():
            weight_specs.append((f"SINGLE_OPTIMUM_UTOPIA_{source_tag}", p_single_uto))
        if source_tag == "LEGACY" and (sdir / "weights.csv").exists():
            weight_specs.append((f"STACKING_OPTIMUM_LINEAR_{source_tag}", sdir / "weights.csv"))

        for sol_name, wpath in weight_specs:
            wdf = pd.read_csv(wpath)
            if "config_id" not in wdf.columns or "weight" not in wdf.columns:
                continue
            wdf["config_id"] = wdf["config_id"].astype(str)
            wdf["weight"] = pd.to_numeric(wdf["weight"], errors="coerce").fillna(0.0)
            wdf = wdf[wdf["weight"] > 0.0].copy()
            if wdf.empty:
                continue

            if not val_preds.empty and {"config_id", "run_id", "row_id", "y_true", "y_pred"}.issubset(set(val_preds.columns)):
                pv = val_preds[val_preds["config_id"].isin(wdf["config_id"])].copy()
                if not pv.empty:
                    mat = pv.pivot_table(index=["run_id", "row_id"], columns="config_id", values="y_pred", aggfunc="first")
                    mat = mat.reindex(columns=wdf["config_id"].tolist())
                    mat = mat.dropna(axis=0, how="any")
                    if not mat.empty:
                        w = wdf.set_index("config_id")["weight"].reindex(mat.columns).to_numpy(dtype=float)
                        w = np.maximum(w, 0.0)
                        if float(np.sum(w)) > 0:
                            w = w / np.sum(w)
                            y_hat = mat.to_numpy(dtype=float) @ w
                            y_true = (
                                pv.drop_duplicates(subset=["run_id", "row_id"])
                                .set_index(["run_id", "row_id"])
                                .reindex(mat.index)["y_true"]
                                .to_numpy(dtype=float)
                            )
                            out["validation_avg"][sol_name] = (y_true, y_hat)

            if not test_preds.empty and {"config_id", "row_id", "y_true", "y_pred"}.issubset(set(test_preds.columns)):
                pt = test_preds[test_preds["config_id"].isin(wdf["config_id"])].copy()
                if not pt.empty:
                    mat = pt.pivot_table(index="row_id", columns="config_id", values="y_pred", aggfunc="first")
                    mat = mat.reindex(columns=wdf["config_id"].tolist())
                    mat = mat.dropna(axis=0, how="any")
                    if not mat.empty:
                        w = wdf.set_index("config_id")["weight"].reindex(mat.columns).to_numpy(dtype=float)
                        w = np.maximum(w, 0.0)
                        if float(np.sum(w)) > 0:
                            w = w / np.sum(w)
                            y_hat = mat.to_numpy(dtype=float) @ w
                            y_true = (
                                pt.drop_duplicates("row_id")
                                .set_index("row_id")
                                .reindex(mat.index)["y_true"]
                                .to_numpy(dtype=float)
                            )
                            out["test"][sol_name] = (y_true, y_hat)
    return out


def run_results_analysis(
    *,
    result_root: str,
    data_id: str,
    split_id: str,
    plot_top_k: Optional[int] = None,
    skip_first_folds: int = 0,
    skip_first_folds_for_stats: int = 0,
    exclude_models_tradeoff: Optional[List[str]] = None,
) -> Dict[str, Any]:
    t0 = time.time()
    _progress_log(
        f"starting analysis | data_id={data_id} split_id={split_id} "
        f"plot_top_k={plot_top_k} skip_first_folds={skip_first_folds}",
        t0=t0,
    )
    analysis_dir = _analysis_dir(result_root, data_id, split_id)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    _progress_log(f"analysis directory ready: {analysis_dir}", t0=t0)

    _progress_log("loading run artifacts (parquet) ...", t0=t0)
    runs_df = _load_runs_df(result_root=result_root, data_id=data_id, split_id=split_id)
    runs_df = _prepare_df(runs_df)
    _progress_log(f"loaded runs: {runs_df.shape[0]} rows x {runs_df.shape[1]} cols", t0=t0)

    summary_df = _summary_by_config(runs_df)
    summary_path = analysis_dir / "summary_by_config.csv"
    summary_df.to_csv(summary_path, index=False)
    _progress_log(f"saved summary_by_config.csv with {summary_df.shape[0]} configs", t0=t0)

    # Choose fold for single-run view
    fold_id = _pick_validation_single_run_fold_id(runs_df)
    single_df = _prepare_df(runs_df[runs_df["fold_id"] == fold_id].copy())

    # Load test metrics (if present)
    test_metrics_path = analysis_dir / "test_metrics.csv"
    test_df = pd.DataFrame()
    if test_metrics_path.exists():
        _progress_log("loading test_metrics.csv ...", t0=t0)
        test_df = _prepare_df(pd.read_csv(test_metrics_path))
        _progress_log(f"loaded test metrics rows: {test_df.shape[0]}", t0=t0)

    exclude_models_tradeoff = [str(m).strip() for m in (exclude_models_tradeoff or []) if str(m).strip()]
    excluded_set = set(exclude_models_tradeoff)

    # Bootstrap protocol metadata (used for both validation and test summaries).
    bootstrap_protocol = _load_bootstrap_protocol(result_root=result_root, data_id=data_id, split_id=split_id)
    _progress_log(
        "bootstrap protocol: "
        f"n_bootstrap={bootstrap_protocol.get('n_bootstrap', 0)} "
        f"block_freq={bootstrap_protocol.get('block_freq', 'M')} "
        f"seed={bootstrap_protocol.get('seed', 2025)}",
        t0=t0,
    )

    # Config metadata (rho/model) used in bootstrap summary tables.
    config_meta = (
        runs_df.loc[:, [c for c in ["config_id", "model_name", "model_config_json", "run_id"] if c in runs_df.columns]]
        .drop_duplicates(subset=[c for c in ["config_id", "run_id"] if c in runs_df.columns], keep="first")
        .copy()
    )
    if "config_id" in config_meta.columns:
        config_meta["config_id"] = config_meta["config_id"].astype(str)
    if "model_config_json" in config_meta.columns:
        config_meta["rho"] = config_meta["model_config_json"].apply(_extract_rho_from_config_json)
    else:
        config_meta["rho"] = np.nan

    # ------------------------------------
    # Bootstrap statistics tables
    # ------------------------------------
    _progress_log("loading validation bootstrap metrics ...", t0=t0)
    boots_df = _load_bootstrap_df(result_root=result_root, data_id=data_id, split_id=split_id)
    _progress_log(f"validation bootstrap rows loaded: {boots_df.shape[0]}", t0=t0)
    if not boots_df.empty:
        _progress_log("computing validation bootstrap statistics tables ...", t0=t0)
        val_boot = _prepare_df(boots_df)
        if "run_id" in val_boot.columns and "run_id" in config_meta.columns:
            val_boot = val_boot.merge(
                config_meta.loc[:, [c for c in ["run_id", "config_id", "model_name", "rho"] if c in config_meta.columns]].drop_duplicates("run_id"),
                on="run_id",
                how="left",
                suffixes=("", "_meta"),
            )
            if "config_id_meta" in val_boot.columns:
                val_boot["config_id"] = val_boot["config_id"].astype(str)
                val_boot["config_id"] = val_boot["config_id"].where(val_boot["config_id"].notna(), val_boot["config_id_meta"].astype(str))
                val_boot = val_boot.drop(columns=["config_id_meta"], errors="ignore")
            if "model_name_meta" in val_boot.columns:
                val_boot["model_name"] = val_boot["model_name"].where(val_boot["model_name"].notna(), val_boot["model_name_meta"])
                val_boot = val_boot.drop(columns=["model_name_meta"], errors="ignore")
        elif {"config_id", "fold_id"}.issubset(set(val_boot.columns)):
            val_boot = val_boot.merge(
                runs_df.loc[:, [c for c in ["config_id", "fold_id", "model_name", "model_config_json"] if c in runs_df.columns]]
                .drop_duplicates(subset=["config_id", "fold_id"]),
                on=["config_id", "fold_id"],
                how="left",
            )
            if "rho" not in val_boot.columns and "model_config_json" in val_boot.columns:
                val_boot["rho"] = val_boot["model_config_json"].apply(_extract_rho_from_config_json)

        if "rho" not in val_boot.columns:
            val_boot["rho"] = np.nan
        val_boot["split"] = "validation"
        val_metric_cols = _bootstrap_metric_columns(val_boot)

        val_stats_cfg = _summarize_bootstrap_statistics(
            val_boot,
            group_cols=[c for c in ["split", "fold_id", "model_name", "config_id", "rho"] if c in val_boot.columns],
            metric_cols=val_metric_cols,
            meta_cols=[c for c in ["bootstrap_block_freq", "bootstrap_seed", "bootstrap_sample_size"] if c in val_boot.columns],
        )
        if not val_stats_cfg.empty:
            val_stats_cfg.to_csv(analysis_dir / "validation_bootstrap_metric_stats_by_config_fold.csv", index=False)
            _progress_log("saved validation stats by config/fold", t0=t0)
            val_stats_cfg_across = _average_stats_across_folds(
                val_stats_cfg,
                group_cols=["split", "model_name", "config_id", "rho", "metric"],
                skip_first_folds_for_stats=skip_first_folds_for_stats,
            )
            if not val_stats_cfg_across.empty:
                val_stats_cfg_across.to_csv(
                    analysis_dir / "validation_bootstrap_metric_stats_by_config_across_folds.csv",
                    index=False,
                )
                _progress_log("saved validation stats by config/across-folds", t0=t0)

        val_stats_rho = _summarize_bootstrap_statistics(
            val_boot,
            group_cols=[c for c in ["split", "fold_id", "model_name", "rho"] if c in val_boot.columns],
            metric_cols=val_metric_cols,
            meta_cols=[c for c in ["bootstrap_block_freq", "bootstrap_seed"] if c in val_boot.columns],
        )
        if not val_stats_rho.empty:
            val_stats_rho.to_csv(analysis_dir / "validation_bootstrap_metric_stats_by_rho_fold.csv", index=False)
            _progress_log("saved validation stats by rho/fold", t0=t0)
            val_stats_rho_across = _average_stats_across_folds(
                val_stats_rho,
                group_cols=["split", "model_name", "rho", "metric"],
                skip_first_folds_for_stats=skip_first_folds_for_stats,
            )
            if not val_stats_rho_across.empty:
                val_stats_rho_across.to_csv(
                    analysis_dir / "validation_bootstrap_metric_stats_by_rho_across_folds.csv",
                    index=False,
                )
                _progress_log("saved validation stats by rho/across-folds", t0=t0)

    _progress_log("building test bootstrap metrics from test_predictions.parquet ...", t0=t0)
    test_boot = _build_test_bootstrap_metrics(
        analysis_dir=analysis_dir,
        protocol=bootstrap_protocol,
        config_meta=config_meta.loc[:, [c for c in ["config_id", "model_name", "rho"] if c in config_meta.columns]].drop_duplicates("config_id"),
    )
    if not test_boot.empty:
        _progress_log(f"computing test bootstrap statistics tables ... rows={test_boot.shape[0]}", t0=t0)
        test_metric_cols = _bootstrap_metric_columns(test_boot)
        test_stats_cfg = _summarize_bootstrap_statistics(
            test_boot,
            group_cols=[c for c in ["split", "fold_id", "model_name", "config_id", "rho"] if c in test_boot.columns],
            metric_cols=test_metric_cols,
            meta_cols=[c for c in ["bootstrap_block_freq", "bootstrap_seed", "bootstrap_sample_size"] if c in test_boot.columns],
        )
        if not test_stats_cfg.empty:
            test_stats_cfg.to_csv(analysis_dir / "test_bootstrap_metric_stats_by_config_fold.csv", index=False)
            _progress_log("saved test stats by config/fold", t0=t0)
            test_stats_cfg_across = _average_stats_across_folds(
                test_stats_cfg,
                group_cols=["split", "model_name", "config_id", "rho", "metric"],
                skip_first_folds_for_stats=skip_first_folds_for_stats,
            )
            if not test_stats_cfg_across.empty:
                test_stats_cfg_across.to_csv(
                    analysis_dir / "test_bootstrap_metric_stats_by_config_across_folds.csv",
                    index=False,
                )
                _progress_log("saved test stats by config/across-folds", t0=t0)

        test_stats_rho = _summarize_bootstrap_statistics(
            test_boot,
            group_cols=[c for c in ["split", "fold_id", "model_name", "rho"] if c in test_boot.columns],
            metric_cols=test_metric_cols,
            meta_cols=[c for c in ["bootstrap_block_freq", "bootstrap_seed"] if c in test_boot.columns],
        )
        if not test_stats_rho.empty:
            test_stats_rho.to_csv(analysis_dir / "test_bootstrap_metric_stats_by_rho_fold.csv", index=False)
            _progress_log("saved test stats by rho/fold", t0=t0)
            test_stats_rho_across = _average_stats_across_folds(
                test_stats_rho,
                group_cols=["split", "model_name", "rho", "metric"],
                skip_first_folds_for_stats=skip_first_folds_for_stats,
            )
            if not test_stats_rho_across.empty:
                test_stats_rho_across.to_csv(
                    analysis_dir / "test_bootstrap_metric_stats_by_rho_across_folds.csv",
                    index=False,
                )
                _progress_log("saved test stats by rho/across-folds", t0=t0)

    (analysis_dir / "bootstrap_statistics_metadata.json").write_text(
        json.dumps(
            {
                "bootstrap_protocol": bootstrap_protocol,
                "validation_bootstrap_source": str(
                    Path(result_root) / "bootstrap_metrics" / f"data_id={data_id}" / f"split_id={split_id}"
                ),
                "test_bootstrap_source": str(analysis_dir / "test_predictions.parquet"),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _progress_log("saved bootstrap_statistics_metadata.json", t0=t0)

    # Stacking overlays
    val_overlay_avg, val_overlay_single = _load_stacking_validation_overlay(analysis_dir=analysis_dir, fold_id=fold_id)
    val_overlay_worst_avg, val_overlay_worst_single = _load_stacking_validation_worst_overlay(analysis_dir=analysis_dir)
    _progress_log(
        "loaded stacking validation overlays "
        f"(avg={val_overlay_avg.shape[0]}, worst={val_overlay_worst_avg.shape[0]})",
        t0=t0,
    )

    # Test stacking overlay needs y_train_log_mean metadata (written by run_temporal_cv.py)
    test_overlay_avg = pd.DataFrame()
    test_overlay_worst = pd.DataFrame()
    meta_path = analysis_dir / "test_eval_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        y_train_log_mean = float(meta.get("y_train_log_mean", np.nan))
        if np.isfinite(y_train_log_mean):
            _progress_log("computing test stacking overlays ...", t0=t0)
            test_overlay_avg, test_overlay_worst = _compute_test_stacking_overlay(analysis_dir=analysis_dir, y_train_log_mean=y_train_log_mean, worst_block_freq="Q")
            _progress_log(
                "computed test stacking overlays "
                f"(avg={test_overlay_avg.shape[0]}, worst={test_overlay_worst.shape[0]})",
                t0=t0,
            )

    # Save compact stacking summary table (if available)
    stacking_summary_df = _build_stacking_optima_summary(
        val_avg=val_overlay_avg,
        val_worst_avg=val_overlay_worst_avg,
        val_single=val_overlay_single,
        val_worst_single=val_overlay_worst_single,
        test_avg=test_overlay_avg,
        test_worst=test_overlay_worst,
    )
    if not stacking_summary_df.empty:
        stacking_summary_df.to_csv(analysis_dir / "stacking_optima_summary.csv", index=False)
        _progress_log("saved stacking_optima_summary.csv", t0=t0)

    _progress_log("building final_model_comparison_taxation_metrics.csv ...", t0=t0)
    final_cmp_df = _build_final_model_comparison_taxation_metrics(
        result_root=result_root,
        analysis_dir=analysis_dir,
        data_id=data_id,
        split_id=split_id,
        runs_df=runs_df,
    )
    final_results_dir = analysis_dir / "final_results"
    final_results_dir.mkdir(parents=True, exist_ok=True)
    if not final_cmp_df.empty:
        cmp_path = analysis_dir / "final_model_comparison_taxation_metrics.csv"
        final_cmp_df.to_csv(cmp_path, index=False)
        shutil.copy2(cmp_path, final_results_dir / "final_model_comparison_taxation_metrics.csv")
        _progress_log("saved and copied final_model_comparison_taxation_metrics.csv", t0=t0)

    # Final vertical-equity plots for baselines and selected stacking solutions.
    _progress_log("collecting predictions for final vertical-equity plots ...", t0=t0)
    pred_sets = _collect_final_solution_predictions_for_vertical_equity(
        result_root=result_root,
        analysis_dir=analysis_dir,
        data_id=data_id,
        split_id=split_id,
        runs_df=runs_df,
    )
    for split_name in ["validation_avg", "test"]:
        split_map = pred_sets.get(split_name, {})
        if not split_map:
            continue
        split_out = final_results_dir / "plots" / split_name
        _progress_log(f"rendering final vertical-equity plots for {split_name}: {len(split_map)} solutions", t0=t0)
        for sol_name, (y_true, y_pred) in split_map.items():
            y_true = np.asarray(y_true, dtype=float).reshape(-1)
            y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
            mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0.0) & (y_pred > 0.0)
            if int(np.sum(mask)) < 2:
                continue
            ratios = y_pred[mask] / y_true[mask]
            y_log = np.log(y_true[mask])
            safe_name = "".join(ch if (ch.isalnum() or ch in {"_", "-", "."}) else "_" for ch in str(sol_name))
            out_path = split_out / f"vertical_equity_{safe_name}.png"
            plot_vertical_equity_lowess(
                y_log=y_log,
                ratios=ratios,
                out_path=out_path,
                model_label=f"{sol_name} [{split_name}]",
                sample_size=3000,
                random_seed=2025,
                lowess_frac=0.4,
                y_limits=(0.0, 4.0),
            )
        _progress_log(f"finished final vertical-equity plots for {split_name}", t0=t0)

    # Plot set (tradeoffs)
    plot_pairs = [
        ("abs_PRD_dev_mean", "OOS R2_mean"),
        ("abs_PRB_dev_mean", "OOS R2_mean"),
        ("abs_VEI_dev_mean", "OOS R2_mean"),
        ("abs_Corr_r_price_mean", "OOS R2_mean"),
        ("abs_PRD_dev_mean", "R2_mean"),
        ("abs_PRB_dev_mean", "R2_mean"),
        ("abs_VEI_dev_mean", "R2_mean"),
        ("abs_Corr_r_price_mean", "R2_mean"),
    ]

    plots_root = analysis_dir / "plots"
    val_dir = plots_root / "validation"
    single_dir = plots_root / "validation_single_run"
    test_dir = plots_root / "test"
    evolution_dir = plots_root / "evolution"

    # Validation: use summary table means (one point per config) + config json
    # so tradeoff coloring can encode rho intensity.
    cfg_meta = (
        runs_df.loc[:, [c for c in ["config_id", "model_config_json"] if c in runs_df.columns]]
        .drop_duplicates(subset=["config_id"])
        .copy()
    )
    val_plot_df = summary_df.copy()
    if not cfg_meta.empty:
        val_plot_df = val_plot_df.merge(cfg_meta, on="config_id", how="left")
    if excluded_set and "model_name" in val_plot_df.columns:
        val_plot_df = val_plot_df[~val_plot_df["model_name"].astype(str).isin(excluded_set)].copy()
    if excluded_set and "model_name" in single_df.columns:
        single_df = single_df[~single_df["model_name"].astype(str).isin(excluded_set)].copy()
    if excluded_set and (not test_df.empty) and ("model_name" in test_df.columns):
        test_df = test_df[~test_df["model_name"].astype(str).isin(excluded_set)].copy()
    _progress_log(
        "tradeoff inputs ready "
        f"(validation={val_plot_df.shape[0]}, single_fold={single_df.shape[0]}, test={test_df.shape[0]})",
        t0=t0,
    )
    for x_col, y_col in plot_pairs:
        if x_col not in val_plot_df.columns or y_col not in val_plot_df.columns:
            continue
        _plot_tradeoff(
            val_plot_df,
            x_col=x_col,
            y_col=y_col,
            title=f"Validation (avg folds): {x_col} vs {y_col}",
            out_path=val_dir / f"tradeoff_{x_col}__{y_col}.png",
            show_top_k=None,
            overlay_points=(
                pd.concat([val_overlay_avg, val_overlay_worst_avg], ignore_index=True)
                if (not val_overlay_avg.empty or not val_overlay_worst_avg.empty)
                else None
            ),
        )
    _progress_log("finished validation tradeoff plots", t0=t0)

    # Single-run validation: one fold
    for x_col, y_col in [
        ("abs_PRD_dev", "OOS R2"),
        ("abs_PRB_dev", "OOS R2"),
        ("abs_VEI_dev", "OOS R2"),
        ("abs_Corr_r_price", "OOS R2"),
        ("abs_PRD_dev", "R2"),
        ("abs_PRB_dev", "R2"),
        ("abs_VEI_dev", "R2"),
        ("abs_Corr_r_price", "R2"),
    ]:
        if x_col not in single_df.columns or y_col not in single_df.columns:
            continue
        _plot_tradeoff(
            single_df,
            x_col=x_col,
            y_col=y_col,
            title=f"Validation (fold {fold_id}): {x_col} vs {y_col}",
            out_path=single_dir / f"tradeoff_fold{fold_id}_{x_col}__{y_col}.png",
            show_top_k=None,
            overlay_points=(
                pd.concat([val_overlay_single, val_overlay_worst_single], ignore_index=True)
                if (not val_overlay_single.empty or not val_overlay_worst_single.empty)
                else None
            ),
        )
    _progress_log("finished single-fold validation tradeoff plots", t0=t0)

    # Test plots
    if not test_df.empty:
        for x_col, y_col in [
            ("abs_PRD_dev", "OOS R2"),
            ("abs_PRB_dev", "OOS R2"),
            ("abs_VEI_dev", "OOS R2"),
            ("abs_Corr_r_price", "OOS R2"),
            ("abs_PRD_dev", "R2"),
            ("abs_PRB_dev", "R2"),
            ("abs_VEI_dev", "R2"),
            ("abs_Corr_r_price", "R2"),
        ]:
            if x_col not in test_df.columns or y_col not in test_df.columns:
                continue
            overlay = pd.concat([test_overlay_avg, test_overlay_worst], ignore_index=True) if (not test_overlay_avg.empty or not test_overlay_worst.empty) else None
            _plot_tradeoff(
                test_df,
                x_col=x_col,
                y_col=y_col,
                title=f"Test (held-out): {x_col} vs {y_col}",
                out_path=test_dir / f"tradeoff_{x_col}__{y_col}.png",
                show_top_k=None,
                overlay_points=overlay,
            )
        _progress_log("finished test tradeoff plots", t0=t0)

    # -----------------------
    # Evolution plots (bootstrap mean ± std across windows/folds)
    # -----------------------
    metrics_for_evolution = ["R2", "OOS R2", "VEI", "PRD", "PRB", "COD"]
    if not boots_df.empty:
        _progress_log("building evolution plots (bootstrap summaries) ...", t0=t0)
        selected_config_ids = _select_configs_for_evolution(runs_df, top_k=plot_top_k)
        (analysis_dir / "evolution_models_selected.json").write_text(
            json.dumps({"selected_config_ids": selected_config_ids}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        _progress_log(f"selected {len(selected_config_ids)} configs for evolution", t0=t0)

        evo_long = _bootstrap_summary_long(boots_df, runs_df, metrics=metrics_for_evolution)
        evo_long = evo_long[evo_long["config_id"].isin(selected_config_ids)].copy()

        # Optional stacking curves (bootstrap-weighted + deterministic fold metrics)
        evo_stack_boot = _stacking_bootstrap_evolution(boots_df, analysis_dir=analysis_dir, metrics=metrics_for_evolution)
        evo_stack_lin = _load_stacking_linear_fold_metrics(analysis_dir=analysis_dir, metrics=metrics_for_evolution)
        if not evo_stack_boot.empty:
            evo_long = pd.concat([evo_long, evo_stack_boot], ignore_index=True)
        if not evo_stack_lin.empty:
            evo_long = pd.concat([evo_long, evo_stack_lin], ignore_index=True)

        # Keep-aware model names for CVaR families so each keep level is visualized
        # as a distinct model line/color without altering stored run schemas.
        cfg_plot_name_map: Dict[str, str] = {}
        cfg_meta_for_plot = (
            runs_df.loc[:, [c for c in ["config_id", "model_name", "model_config_json"] if c in runs_df.columns]]
            .drop_duplicates(subset=["config_id"])
            .copy()
        )
        if not cfg_meta_for_plot.empty:
            cfg_meta_for_plot["config_id"] = cfg_meta_for_plot["config_id"].astype(str)
            cfg_meta_for_plot["_plot_model_name"] = [
                _plot_model_name(m, cfg) for m, cfg in zip(cfg_meta_for_plot["model_name"], cfg_meta_for_plot["model_config_json"])
            ]
            cfg_plot_name_map = dict(zip(cfg_meta_for_plot["config_id"], cfg_meta_for_plot["_plot_model_name"]))
        evo_long["config_id"] = evo_long["config_id"].astype(str)
        evo_long["model_name"] = [
            cfg_plot_name_map.get(str(cfg_id), str(mname))
            for cfg_id, mname in zip(evo_long["config_id"], evo_long["model_name"])
        ]

        # Save aggregated table used for plotting
        evo_long.to_csv(analysis_dir / "evolution_bootstrap_summary.csv", index=False)
        _progress_log(f"saved evolution_bootstrap_summary.csv rows={evo_long.shape[0]}", t0=t0)

        # Color mapping by model family (model_name)
        model_color_map = _build_plot_color_map(evo_long["model_name"].tolist())
        # Force stacking colors to be distinct and consistent
        if "STACKING_OPTIMUM_BOOTSTRAP" in model_color_map:
            model_color_map["STACKING_OPTIMUM_BOOTSTRAP"] = "crimson"
        if "STACKING_OPTIMUM_LINEAR" in model_color_map:
            model_color_map["STACKING_OPTIMUM_LINEAR"] = "black"
        if "STACKING_OPTIMUM_BOOTSTRAP_ROBUST" in model_color_map:
            model_color_map["STACKING_OPTIMUM_BOOTSTRAP_ROBUST"] = "crimson"
        if "STACKING_OPTIMUM_BOOTSTRAP_LEGACY" in model_color_map:
            model_color_map["STACKING_OPTIMUM_BOOTSTRAP_LEGACY"] = "firebrick"
        if "STACKING_OPTIMUM_LINEAR_ROBUST" in model_color_map:
            model_color_map["STACKING_OPTIMUM_LINEAR_ROBUST"] = "black"
        if "STACKING_OPTIMUM_LINEAR_LEGACY" in model_color_map:
            model_color_map["STACKING_OPTIMUM_LINEAR_LEGACY"] = "dimgray"
        if "STACKING_OPTIMUM_AVERAGE_ROBUST" in model_color_map:
            model_color_map["STACKING_OPTIMUM_AVERAGE_ROBUST"] = "goldenrod"
        if "STACKING_OPTIMUM_AVERAGE_LEGACY" in model_color_map:
            model_color_map["STACKING_OPTIMUM_AVERAGE_LEGACY"] = "dodgerblue"
        if "STACKING_OPTIMUM_WORST_ROBUST" in model_color_map:
            model_color_map["STACKING_OPTIMUM_WORST_ROBUST"] = "firebrick"
        if "STACKING_OPTIMUM_WORST_LEGACY" in model_color_map:
            model_color_map["STACKING_OPTIMUM_WORST_LEGACY"] = "purple"
        if "STACKING_OPTIMUM_UTOPIA_ROBUST" in model_color_map:
            model_color_map["STACKING_OPTIMUM_UTOPIA_ROBUST"] = "darkcyan"
        if "STACKING_OPTIMUM_UTOPIA_LEGACY" in model_color_map:
            model_color_map["STACKING_OPTIMUM_UTOPIA_LEGACY"] = "slateblue"
        if "SINGLE_OPTIMUM_REQUESTED_ROBUST" in model_color_map:
            model_color_map["SINGLE_OPTIMUM_REQUESTED_ROBUST"] = "navy"
        if "SINGLE_OPTIMUM_REQUESTED_LEGACY" in model_color_map:
            model_color_map["SINGLE_OPTIMUM_REQUESTED_LEGACY"] = "cornflowerblue"
        if "SINGLE_OPTIMUM_UTOPIA_ROBUST" in model_color_map:
            model_color_map["SINGLE_OPTIMUM_UTOPIA_ROBUST"] = "darkviolet"
        if "SINGLE_OPTIMUM_UTOPIA_LEGACY" in model_color_map:
            model_color_map["SINGLE_OPTIMUM_UTOPIA_LEGACY"] = "mediumpurple"

        # Labels per config_id
        labels: Dict[str, str] = {}
        for cfg_id in sorted(set(evo_long["config_id"].astype(str).tolist())):
            if cfg_id.startswith("STACKING_OPTIMUM"):
                labels[cfg_id] = cfg_id
            else:
                labels[cfg_id] = _config_label_from_runs(runs_df, cfg_id)

        for m in metrics_for_evolution:
            _progress_log(f"[evolution] plotting {m} ...", t0=t0)
            _plot_metric_evolution(
                evo_long,
                metric=m,
                title=f"Evolution across folds (bootstrap mean ± std): {m}",
                out_path=evolution_dir / f"evolution_{m.replace(' ', '_')}.png",
                color_map=model_color_map,
                config_labels=labels,
                skip_first_folds=skip_first_folds,
            )
        _progress_log("finished evolution plots", t0=t0)

    _progress_log("analysis completed", t0=t0)

    return {
        "data_id": data_id,
        "split_id": split_id,
        "analysis_dir": str(analysis_dir),
        "n_completed_runs": int(runs_df.shape[0]),
        "n_summary_configs": int(summary_df.shape[0]),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze rolling-origin CV artifacts (tables + plots).")
    p.add_argument("--result-root", type=str, default="./output/robust_rolling_origin_cv")
    p.add_argument("--data-id", type=str, required=True)
    p.add_argument("--split-id", type=str, required=True)
    p.add_argument("--plot-top-k", type=int, default=None, help="If set, only limit model selection for evolution plots.")
    p.add_argument("--skip-first-folds", type=int, default=0,
                   help="Exclude the first N fold IDs from evolution plots (useful when early folds are noisy).")
    p.add_argument(
        "--skip-first-folds-for-stats",
        type=int,
        default=0,
        help="Exclude the first N numeric fold IDs when computing *across-folds* stats tables only.",
    )
    p.add_argument(
        "--exclude-models-tradeoff",
        type=str,
        default="",
        help="Comma-separated model_name values to exclude from tradeoff plots only (e.g., 'LinearRegression').",
    )
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    exclude_models_tradeoff = [s.strip() for s in str(args.exclude_models_tradeoff).split(",") if s.strip()]
    out = run_results_analysis(
        result_root=str(args.result_root),
        data_id=str(args.data_id),
        split_id=str(args.split_id),
        plot_top_k=(None if args.plot_top_k is None else int(args.plot_top_k)),
        skip_first_folds=int(args.skip_first_folds),
        skip_first_folds_for_stats=int(args.skip_first_folds_for_stats),
        exclude_models_tradeoff=exclude_models_tradeoff,
    )
    print("=" * 90)
    print("RESULTS ANALYSIS COMPLETED")
    print("=" * 90)
    print(f"data_id={out['data_id']} | split_id={out['split_id']}")
    print(f"analysis_dir={out['analysis_dir']}")
    print(f"n_completed_runs={out['n_completed_runs']} | n_summary_configs={out['n_summary_configs']}")

