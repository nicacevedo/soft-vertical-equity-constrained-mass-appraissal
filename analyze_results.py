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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from utils.motivation_utils import IAAO_PRB_RANGE, IAAO_PRD_RANGE, IAAO_VEI_RANGE, _compute_extended_metrics


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
    for c in ["R2", "OOS R2", "PRD", "PRB", "VEI", "COD"]:
        if c in dfx.columns:
            dfx[c] = pd.to_numeric(dfx[c], errors="coerce")
    if "PRD" in dfx.columns:
        dfx["abs_PRD_dev"] = (dfx["PRD"] - 1.0).abs()
    if "PRB" in dfx.columns:
        dfx["abs_PRB_dev"] = dfx["PRB"].abs()
    if "VEI" in dfx.columns:
        dfx["abs_VEI_dev"] = dfx["VEI"].abs()
    return dfx


def _summary_by_config(runs_df: pd.DataFrame) -> pd.DataFrame:
    dfx = _prepare_df(runs_df)
    agg_cols = [c for c in ["R2", "OOS R2", "PRD", "PRB", "VEI", "abs_PRD_dev", "abs_PRB_dev", "abs_VEI_dev"] if c in dfx.columns]
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
    model_name = str(row["model_name"].iloc[0])
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
                parts.append(f"rho={cfg['rho']}")
            if "keep" in cfg:
                parts.append(f"keep={cfg['keep']}")
            if parts:
                tag = " (" + ", ".join(parts) + ")"
    return f"{model_name}{tag}"


def _make_model_color_map(model_names: List[str]) -> Dict[str, Any]:
    palette = list(plt.cm.tab20.colors)
    uniq = sorted({str(m) for m in model_names})
    if not uniq:
        return {}
    return {m: palette[i % len(palette)] for i, m in enumerate(uniq)}


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
    weights_path = analysis_dir / "stacking_pf_opt" / "weights.csv"
    if not weights_path.exists():
        return pd.DataFrame()
    if boots_df.empty:
        return pd.DataFrame()

    wdf = pd.read_csv(weights_path)
    wdf["config_id"] = wdf["config_id"].astype(str)
    wdf["weight"] = pd.to_numeric(wdf["weight"], errors="coerce").fillna(0.0)
    wdf = wdf[wdf["weight"] > 0.0].copy()
    if wdf.empty:
        return pd.DataFrame()

    dfx = boots_df.copy()
    dfx["config_id"] = dfx["config_id"].astype(str)
    dfx = dfx[dfx["config_id"].isin(wdf["config_id"])].copy()
    if dfx.empty:
        return pd.DataFrame()

    dfx = dfx.merge(wdf.loc[:, ["config_id", "weight"]], on="config_id", how="inner")

    rows: List[pd.DataFrame] = []
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
        summ["config_id"] = "STACKING_OPTIMUM_BOOTSTRAP"
        summ["model_name"] = "STACKING_OPTIMUM_BOOTSTRAP"
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
    fold_path = analysis_dir / "stacking_pf_opt" / "fold_ensemble_metrics.csv"
    if not fold_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(fold_path)
    # normalize expected names from optimizer output
    rename_map = {
        "R2_ensemble": "R2",
        "PRD_ensemble": "PRD",
        "PRB_ensemble": "PRB",
        "VEI_ensemble": "VEI",
        "COD_ensemble": "COD",
        "OOS R2_ensemble": "OOS R2",
    }
    df = df.rename(columns=rename_map)
    df = _prepare_df(df)

    rows: List[pd.DataFrame] = []
    for m in metrics:
        if m not in df.columns:
            continue
        tmp = df.loc[:, ["fold_id", m]].copy()
        tmp = tmp.rename(columns={m: "mean"})
        tmp["std"] = np.nan
        tmp["metric"] = m
        tmp["config_id"] = "STACKING_OPTIMUM_LINEAR"
        tmp["model_name"] = "STACKING_OPTIMUM_LINEAR"
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
) -> None:
    dfm = evo_long[evo_long["metric"] == metric].copy()
    if dfm.empty:
        return

    dfm["fold_id"] = pd.to_numeric(dfm["fold_id"], errors="coerce")
    dfm = dfm.dropna(subset=["fold_id", "mean"])
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
        lw = 2.6 if is_stacking else 1.6
        ls = "--" if str(model_name) == "STACKING_OPTIMUM_LINEAR" else "-"

        ax.plot(x, y, color=base_color, linewidth=lw, linestyle=ls, label=label)
        if np.any(np.isfinite(s)) and str(model_name) != "STACKING_OPTIMUM_LINEAR":
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
    if show_top_k is not None and int(show_top_k) > 0 and x_col in dfx.columns:
        dfx = dfx.sort_values(x_col, ascending=False).head(int(show_top_k)).copy()

    x = dfx[x_col].to_numpy(dtype=float)
    y = dfx[y_col].to_numpy(dtype=float)
    pareto = _compute_2d_pareto_mask(x, y, maximize_x=True, minimize_y=True)

    baselines = dfx["model_name"].isin(["LinearRegression", "LGBMRegressor"])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(True, alpha=0.35)

    ax.scatter(x[~baselines], y[~baselines], s=28, alpha=0.85, label="models", marker="o")
    if np.any(baselines):
        ax.scatter(x[baselines], y[baselines], s=120, alpha=0.9, label="baselines", marker="*")

    ax.scatter(x[pareto], y[pareto], s=60, facecolors="none", edgecolors="black", linewidths=1.2, label="pareto (2D)")

    # IAAO reference for fairness axis (if applicable)
    if y_col == "abs_PRD_dev":
        band_lo = abs(float(IAAO_PRD_RANGE[0]) - 1.0)
        band_hi = abs(float(IAAO_PRD_RANGE[1]) - 1.0)
        ax.axhspan(min(band_lo, band_hi), max(band_lo, band_hi), color="gray", alpha=0.25, label="IAAO band (PRD)")
        ax.axhline(0.0, color="gray", linewidth=1.0)
    elif y_col == "abs_PRB_dev":
        ax.axhspan(0.0, max(abs(float(IAAO_PRB_RANGE[0])), abs(float(IAAO_PRB_RANGE[1]))), color="gray", alpha=0.25, label="IAAO band (PRB)")
        ax.axhline(0.0, color="gray", linewidth=1.0)
    elif y_col == "abs_VEI_dev":
        ax.axhspan(0.0, max(abs(float(IAAO_VEI_RANGE[0])), abs(float(IAAO_VEI_RANGE[1]))), color="gray", alpha=0.25, label="IAAO band (VEI)")
        ax.axhline(0.0, color="gray", linewidth=1.0)

    if overlay_points is not None and not overlay_points.empty:
        o = overlay_points.copy()
        o = o[np.isfinite(o[x_col]) & np.isfinite(o[y_col])]
        if not o.empty:
            ax.scatter(o[x_col], o[y_col], s=140, marker="X", color="crimson", label="stacking optimum")

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


def _load_stacking_validation_overlay(*, analysis_dir: Path, fold_id: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fold_path = analysis_dir / "stacking_pf_opt" / "fold_ensemble_metrics.csv"
    if not fold_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    fold_df = pd.read_csv(fold_path)
    fold_df = _prepare_df(fold_df.rename(columns={"OOS R2_ensemble": "OOS R2", "R2_ensemble": "R2", "PRD_ensemble": "PRD", "PRB_ensemble": "PRB", "VEI_ensemble": "VEI"}))
    avg = fold_df.drop(columns=["fold_id"], errors="ignore").mean(numeric_only=True).to_frame().T
    avg.columns = [f"{c}_mean" for c in avg.columns]  # match column names used in validation avg plots
    single = fold_df.loc[fold_df["fold_id"] == int(fold_id)].copy()
    return avg, single


def _compute_test_stacking_overlay(*, analysis_dir: Path, y_train_log_mean: float, worst_block_freq: str = "Q") -> Tuple[pd.DataFrame, pd.DataFrame]:
    weights_path = analysis_dir / "stacking_pf_opt" / "weights.csv"
    preds_path = analysis_dir / "test_predictions.parquet"
    if (not weights_path.exists()) or (not preds_path.exists()):
        return pd.DataFrame(), pd.DataFrame()

    wdf = pd.read_csv(weights_path)
    wdf = wdf[pd.to_numeric(wdf["weight"], errors="coerce").fillna(0.0) > 0.0].copy()
    if wdf.empty:
        return pd.DataFrame(), pd.DataFrame()

    pdf = pd.read_parquet(preds_path)
    pdf = pdf[pdf["config_id"].isin(wdf["config_id"])].copy()
    if pdf.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Wide matrix of log-preds
    mat = pdf.pivot_table(index="row_id", columns="config_id", values="y_pred_log", aggfunc="first")
    mat = mat.reindex(columns=wdf["config_id"].tolist())
    if mat.isna().any().any():
        mat = mat.dropna(axis=0)

    w = wdf.set_index("config_id")["weight"].reindex(mat.columns).to_numpy(dtype=float)
    w = np.maximum(w, 0.0)
    w = w / w.sum()

    y_pred_stack_log = mat.to_numpy(dtype=float) @ w
    y_true_log = (
        pdf.drop_duplicates("row_id")
        .set_index("row_id")
        .reindex(mat.index)["y_true_log"]
        .to_numpy(dtype=float)
    )
    sale_date = (
        pd.to_datetime(
            pdf.drop_duplicates("row_id")
            .set_index("row_id")
            .reindex(mat.index)["sale_date"]
        )
    )

    y_train_log = np.array([float(y_train_log_mean)], dtype=float)
    avg_metrics = _compute_extended_metrics(y_true_log=y_true_log, y_pred_log=y_pred_stack_log, y_train_log=y_train_log, ratio_mode="diff")
    avg_row = pd.DataFrame([{**avg_metrics, "model_name": "STACKING_OPTIMUM"}])
    avg_row = _prepare_df(avg_row)

    # Worst temporal block on test (by OOS R2 if present else R2)
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
            worst_row = {**met, "model_name": "STACKING_OPTIMUM_WORST_BLOCK", "block": str(b)}
    worst_df = _prepare_df(pd.DataFrame([worst_row])) if worst_row is not None else pd.DataFrame()
    return avg_row, worst_df


def run_results_analysis(
    *,
    result_root: str,
    data_id: str,
    split_id: str,
    plot_top_k: Optional[int] = None,
) -> Dict[str, Any]:
    analysis_dir = _analysis_dir(result_root, data_id, split_id)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    runs_df = _load_runs_df(result_root=result_root, data_id=data_id, split_id=split_id)
    runs_df = _prepare_df(runs_df)

    summary_df = _summary_by_config(runs_df)
    summary_path = analysis_dir / "summary_by_config.csv"
    summary_df.to_csv(summary_path, index=False)

    # Choose fold for single-run view
    fold_id = _pick_validation_single_run_fold_id(runs_df)
    single_df = _prepare_df(runs_df[runs_df["fold_id"] == fold_id].copy())

    # Load test metrics (if present)
    test_metrics_path = analysis_dir / "test_metrics.csv"
    test_df = pd.DataFrame()
    if test_metrics_path.exists():
        test_df = _prepare_df(pd.read_csv(test_metrics_path))

    # Stacking overlays
    val_overlay_avg, val_overlay_single = _load_stacking_validation_overlay(analysis_dir=analysis_dir, fold_id=fold_id)

    # Test stacking overlay needs y_train_log_mean metadata (written by run_temporal_cv.py)
    test_overlay_avg = pd.DataFrame()
    test_overlay_worst = pd.DataFrame()
    meta_path = analysis_dir / "test_eval_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        y_train_log_mean = float(meta.get("y_train_log_mean", np.nan))
        if np.isfinite(y_train_log_mean):
            test_overlay_avg, test_overlay_worst = _compute_test_stacking_overlay(analysis_dir=analysis_dir, y_train_log_mean=y_train_log_mean, worst_block_freq="Q")

    # Plot set (tradeoffs)
    plot_pairs = [
        ("OOS R2_mean", "abs_PRD_dev_mean"),
        ("OOS R2_mean", "abs_PRB_dev_mean"),
        ("OOS R2_mean", "abs_VEI_dev_mean"),
        ("R2_mean", "abs_PRD_dev_mean"),
        ("R2_mean", "abs_PRB_dev_mean"),
        ("R2_mean", "abs_VEI_dev_mean"),
    ]

    plots_root = analysis_dir / "plots"
    val_dir = plots_root / "validation"
    single_dir = plots_root / "validation_single_run"
    test_dir = plots_root / "test"
    evolution_dir = plots_root / "evolution"

    # Validation: use summary table means (one point per config)
    val_plot_df = summary_df.copy()
    for x_col, y_col in plot_pairs:
        if x_col not in val_plot_df.columns or y_col not in val_plot_df.columns:
            continue
        _plot_tradeoff(
            val_plot_df,
            x_col=x_col,
            y_col=y_col,
            title=f"Validation (avg folds): {x_col} vs {y_col}",
            out_path=val_dir / f"tradeoff_{x_col}__{y_col}.png",
            show_top_k=plot_top_k,
            overlay_points=val_overlay_avg if not val_overlay_avg.empty else None,
        )

    # Single-run validation: one fold
    for x_col, y_col in [("OOS R2", "abs_PRD_dev"), ("OOS R2", "abs_PRB_dev"), ("OOS R2", "abs_VEI_dev"), ("R2", "abs_PRD_dev"), ("R2", "abs_PRB_dev"), ("R2", "abs_VEI_dev")]:
        if x_col not in single_df.columns or y_col not in single_df.columns:
            continue
        _plot_tradeoff(
            single_df,
            x_col=x_col,
            y_col=y_col,
            title=f"Validation (fold {fold_id}): {x_col} vs {y_col}",
            out_path=single_dir / f"tradeoff_fold{fold_id}_{x_col}__{y_col}.png",
            show_top_k=plot_top_k,
            overlay_points=val_overlay_single if not val_overlay_single.empty else None,
        )

    # Test plots
    if not test_df.empty:
        for x_col, y_col in [("OOS R2", "abs_PRD_dev"), ("OOS R2", "abs_PRB_dev"), ("OOS R2", "abs_VEI_dev"), ("R2", "abs_PRD_dev"), ("R2", "abs_PRB_dev"), ("R2", "abs_VEI_dev")]:
            if x_col not in test_df.columns or y_col not in test_df.columns:
                continue
            overlay = pd.concat([test_overlay_avg, test_overlay_worst], ignore_index=True) if (not test_overlay_avg.empty or not test_overlay_worst.empty) else None
            _plot_tradeoff(
                test_df,
                x_col=x_col,
                y_col=y_col,
                title=f"Test (held-out): {x_col} vs {y_col}",
                out_path=test_dir / f"tradeoff_{x_col}__{y_col}.png",
                show_top_k=plot_top_k,
                overlay_points=overlay,
            )

    # -----------------------
    # Evolution plots (bootstrap mean ± std across windows/folds)
    # -----------------------
    metrics_for_evolution = ["R2", "OOS R2", "VEI", "PRD", "PRB", "COD"]
    boots_df = _load_bootstrap_df(result_root=result_root, data_id=data_id, split_id=split_id)
    if not boots_df.empty:
        selected_config_ids = _select_configs_for_evolution(runs_df, top_k=plot_top_k)
        (analysis_dir / "evolution_models_selected.json").write_text(
            json.dumps({"selected_config_ids": selected_config_ids}, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        evo_long = _bootstrap_summary_long(boots_df, runs_df, metrics=metrics_for_evolution)
        evo_long = evo_long[evo_long["config_id"].isin(selected_config_ids)].copy()

        # Optional stacking curves (bootstrap-weighted + deterministic fold metrics)
        evo_stack_boot = _stacking_bootstrap_evolution(boots_df, analysis_dir=analysis_dir, metrics=metrics_for_evolution)
        evo_stack_lin = _load_stacking_linear_fold_metrics(analysis_dir=analysis_dir, metrics=metrics_for_evolution)
        if not evo_stack_boot.empty:
            evo_long = pd.concat([evo_long, evo_stack_boot], ignore_index=True)
        if not evo_stack_lin.empty:
            evo_long = pd.concat([evo_long, evo_stack_lin], ignore_index=True)

        # Save aggregated table used for plotting
        evo_long.to_csv(analysis_dir / "evolution_bootstrap_summary.csv", index=False)

        # Color mapping by model family (model_name)
        model_color_map = _make_model_color_map(evo_long["model_name"].tolist())
        # Force stacking colors to be distinct and consistent
        if "STACKING_OPTIMUM_BOOTSTRAP" in model_color_map:
            model_color_map["STACKING_OPTIMUM_BOOTSTRAP"] = "crimson"
        if "STACKING_OPTIMUM_LINEAR" in model_color_map:
            model_color_map["STACKING_OPTIMUM_LINEAR"] = "black"

        # Labels per config_id
        labels: Dict[str, str] = {}
        for cfg_id in sorted(set(evo_long["config_id"].astype(str).tolist())):
            if cfg_id.startswith("STACKING_OPTIMUM"):
                labels[cfg_id] = cfg_id
            else:
                labels[cfg_id] = _config_label_from_runs(runs_df, cfg_id)

        for m in metrics_for_evolution:
            print(f"  [evolution] plotting {m} ...")
            _plot_metric_evolution(
                evo_long,
                metric=m,
                title=f"Evolution across folds (bootstrap mean ± std): {m}",
                out_path=evolution_dir / f"evolution_{m.replace(' ', '_')}.png",
                color_map=model_color_map,
                config_labels=labels,
            )

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
    p.add_argument("--plot-top-k", type=int, default=None, help="If set, only plot top-K by accuracy per figure.")
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    out = run_results_analysis(
        result_root=str(args.result_root),
        data_id=str(args.data_id),
        split_id=str(args.split_id),
        plot_top_k=(None if args.plot_top_k is None else int(args.plot_top_k)),
    )
    print("=" * 90)
    print("RESULTS ANALYSIS COMPLETED")
    print("=" * 90)
    print(f"data_id={out['data_id']} | split_id={out['split_id']}")
    print(f"analysis_dir={out['analysis_dir']}")
    print(f"n_completed_runs={out['n_completed_runs']} | n_summary_configs={out['n_summary_configs']}")

