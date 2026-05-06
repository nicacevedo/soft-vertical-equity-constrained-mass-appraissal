"""End-to-end post-CV pipeline that converts a rolling-origin CV output into
the tables + figures needed for ``Paper/GE slides/slides_v1_5 {YEAR}.tex``.

For each year we:
  1. Locate the single (data_id, split_id) produced by run_temporal_cv.py.
  2. Run ``analyze_results.py`` with test bootstrap enabled to obtain metric
     CSVs, per-fold stats, and tradeoff plots.
  3. Run ``simple_model_selection.py`` with the four constraint subsets that
     populate Appendix A4 (``PRD``; ``PRD,PRB,VEI``; ``PRD,MEAN_RATIO,COD``;
     full set).
  4. Emit all slide figures in the format the .tex files expect
     (``img{year}/motivation/...``, ``img{year}/tradeoffs/...``,
     ``img{year}/correlations/...``, ``img{year}/model_selection/...``).
  5. Write ``slide_summary_{year}.json`` containing every scalar needed by
     ``update_slides_tex.py``.

This script is intentionally idempotent: it writes everything into a year
specific ``slide_build/{tag}/`` sandbox and only touches the slide deck in a
follow-up step (``update_slides_tex.py``).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = "/home/nacevedo/.conda/envs/fairness_env/bin/python"


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _log(msg: str) -> None:
    print(f"[pipeline {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _run(cmd: List[str], cwd: Optional[Path] = None) -> None:
    _log("running: " + " ".join(cmd))
    rc = subprocess.call(cmd, cwd=str(cwd or REPO_ROOT))
    if rc != 0:
        raise RuntimeError(f"command failed (rc={rc}): {cmd}")


def _format_currency(val: float) -> str:
    if not np.isfinite(val):
        return "---"
    return f"\\${int(round(val)):,}".replace(",", "{,}")


def _fmt_pct(val: float, digits: int = 1, signed: bool = True) -> str:
    if not np.isfinite(val):
        return "---"
    pct = float(val) * 100.0
    fmt = f"{pct:+.{digits}f}" if signed else f"{pct:.{digits}f}"
    return fmt


# ---------------------------------------------------------------------------
# discovery of data_id / split_id
# ---------------------------------------------------------------------------
def resolve_split(result_root: Path) -> Tuple[str, str, Path]:
    """Return (data_id, split_id, analysis_dir)."""
    analysis_root = result_root / "analysis"
    data_dirs = sorted(p for p in analysis_root.glob("data_id=*") if p.is_dir())
    if not data_dirs:
        raise FileNotFoundError(f"no data_id=* under {analysis_root}")
    # pick the newest data_id (in case previous smoke outputs linger)
    data_dir = max(data_dirs, key=lambda p: p.stat().st_mtime)
    split_dirs = sorted(p for p in data_dir.glob("split_id=*") if p.is_dir())
    if not split_dirs:
        raise FileNotFoundError(f"no split_id=* under {data_dir}")
    split_dir = max(split_dirs, key=lambda p: p.stat().st_mtime)
    data_id = data_dir.name.split("=", 1)[1]
    split_id = split_dir.name.split("=", 1)[1]
    return data_id, split_id, split_dir


# ---------------------------------------------------------------------------
# configuration helpers: read model_config_json to extract rho
# ---------------------------------------------------------------------------
def _extract_rho(config_json: str) -> float:
    try:
        d = json.loads(config_json)
    except Exception:
        return np.nan
    for k in ("rho", "Rho"):
        if k in d:
            try:
                return float(d[k])
            except Exception:
                pass
    return np.nan


def augment_test_metrics(analysis_dir: Path) -> pd.DataFrame:
    """Read test_metrics.csv and add ``rho`` + handy cleaned columns."""
    df = pd.read_csv(analysis_dir / "test_metrics.csv")
    df["rho"] = df.get("model_config_json", pd.Series([""] * len(df))).apply(_extract_rho)
    return df


# ---------------------------------------------------------------------------
# metric selection for the main slide tables
# ---------------------------------------------------------------------------
def pick_baseline_rows(test_df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Return one representative row per baseline model (LinearRegression, LGBMRegressor)."""
    out: Dict[str, pd.Series] = {}
    for m in ("LinearRegression", "LGBMRegressor"):
        sub = test_df[test_df["model_name"] == m]
        if sub.empty:
            continue
        # there should be exactly one baseline config per model_name
        out[m] = sub.iloc[0]
    return out


def pick_representative_penalized(
    test_df: pd.DataFrame,
    *,
    model_name: str = "LGBSmoothPenalty",
    target_metric: str = "R2",
    min_target: float | None = 0.85,
    max_prd_dev: float | None = 0.04,
) -> Optional[pd.Series]:
    """Pick a penalized config that keeps OOS R2 high while lowering PRD toward 1.

    We want the PRD-closest-to-1 point whose ``target_metric`` exceeds
    ``min_target`` (if provided) and whose |PRD-1| is below ``max_prd_dev``.
    Falls back to smallest |PRD-1| if nothing matches.
    """
    sub = test_df[test_df["model_name"] == model_name].copy()
    if sub.empty:
        return None
    sub["prd_dev"] = (sub["PRD"] - 1.0).abs()
    if min_target is not None and target_metric in sub.columns:
        mask = sub[target_metric] >= float(min_target)
        if mask.any():
            sub = sub[mask]
    if max_prd_dev is not None:
        mask = sub["prd_dev"] <= float(max_prd_dev)
        if mask.any():
            filt = sub[mask]
            # among feasible: keep highest R2
            return filt.sort_values(target_metric, ascending=False).iloc[0]
    return sub.sort_values("prd_dev").iloc[0]


# ---------------------------------------------------------------------------
# figure generation
# ---------------------------------------------------------------------------
def _load_predictions(analysis_dir: Path, model_name: str, config_id: str) -> Optional[pd.DataFrame]:
    """Load the held-out test predictions for a (model_name, config_id)."""
    pred_path = analysis_dir / "test_predictions.parquet"
    if not pred_path.exists():
        return None
    df = pd.read_parquet(pred_path)
    sub = df[(df["model_name"] == model_name) & (df["config_id"] == config_id)]
    if sub.empty:
        return None
    return sub.reset_index(drop=True)


def make_motivation_plots(
    analysis_dir: Path,
    baseline_lgb_row: pd.Series,
    penalized_row: pd.Series,
    out_dir: Path,
) -> Dict[str, Path]:
    """Regenerate the two motivation ratio-vs-logprice plots."""
    sys.path.insert(0, str(REPO_ROOT))
    from utils.plotting_utils import plot_ratio_vs_logprice  # noqa: WPS433

    out_dir.mkdir(parents=True, exist_ok=True)

    outputs: Dict[str, Path] = {}
    try:
        pen_rho = float(penalized_row.get("rho", np.nan))
    except Exception:
        pen_rho = np.nan
    pen_label = (
        f"Covariance-Penalized LightGBM  (rho={pen_rho:.3f})"
        if np.isfinite(pen_rho) else "Covariance-Penalized LightGBM"
    )
    for tag, row, label in [
        ("baseline", baseline_lgb_row, "Baseline LightGBM"),
        ("penalized", penalized_row, pen_label),
    ]:
        pred_df = _load_predictions(analysis_dir, row["model_name"], row["config_id"])
        if pred_df is None or pred_df.empty:
            _log(f"[motivation] no predictions for {row['model_name']} {row['config_id']}")
            continue
        if "y_true_log" in pred_df.columns and "y_pred_log" in pred_df.columns:
            y_true_log = pred_df["y_true_log"].to_numpy()
            y_pred_log = pred_df["y_pred_log"].to_numpy()
        else:
            y_true_log = np.log(pred_df["y_true"].to_numpy())
            y_pred_log = np.log(pred_df["y_pred"].to_numpy())
        out_path = out_dir / f"{tag}_motivation.pdf"
        plot_ratio_vs_logprice(
            y_true_log=y_true_log,
            y_pred_log=y_pred_log,
            out_path=out_path,
            model_label=label,
            split_label="Held-out test",
            metrics={
                "Corr(r,logprice)": row.get("Corr(r,logprice)", np.nan),
                "PRD": row.get("PRD", np.nan),
                "PRB": row.get("PRB", np.nan),
                "VEI": row.get("VEI", np.nan),
            },
            sample_size=15000,
            lowess_frac=0.4,
        )
        outputs[tag] = out_path
    return outputs


def make_tradeoff_band_plots(
    test_df: pd.DataFrame,
    out_dir: Path,
    *,
    rho_col: str = "rho",
    penalized_name: str = "LGBSmoothPenalty",
    also_family: str = "LGBCovPenalty",
) -> Dict[str, Path]:
    """Build tradeoff band plots (VEI vs R2, |PRD-1| vs R2, PRB vs R2)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plots: Dict[str, Path] = {}

    base_lgb = test_df[test_df["model_name"] == "LGBMRegressor"].iloc[0]
    base_lr = test_df[test_df["model_name"] == "LinearRegression"].iloc[0]

    pairs: List[Tuple[str, str, callable, str]] = [
        ("PRD", r"$|\mathrm{PRD}-1|$", lambda s: np.abs(s["PRD"] - 1.0), "PRD_vs_R2_band"),
        ("VEI", r"|VEI| (%)", lambda s: np.abs(s["VEI"]), "VEI_vs_R2_band"),
        ("PRB", r"|PRB|", lambda s: np.abs(s["PRB"]), "PRB_vs_R2_band"),
    ]

    for _, y_label, transform, fname in pairs:
        fig, ax = plt.subplots(figsize=(6.2, 4.6))

        for family, marker, color in [
            (penalized_name, "o", "tab:blue"),
            (also_family, "s", "tab:orange"),
        ]:
            sub = test_df[test_df["model_name"] == family].copy()
            if sub.empty:
                continue
            sub = sub.dropna(subset=[rho_col])
            sub = sub.sort_values(rho_col)
            ax.plot(
                sub["R2"],
                transform(sub),
                marker=marker,
                color=color,
                alpha=0.75,
                label=family,
                markersize=5,
                linewidth=1.2,
            )

        ax.scatter(
            [base_lgb["R2"]],
            [transform(pd.DataFrame([base_lgb]).iloc[0])],
            color="red",
            marker="X",
            s=110,
            label="LGBM baseline",
            zorder=3,
        )
        ax.scatter(
            [base_lr["R2"]],
            [transform(pd.DataFrame([base_lr]).iloc[0])],
            color="grey",
            marker="D",
            s=80,
            label="Linear baseline",
            zorder=3,
        )

        ax.set_xlabel(r"Held-out $R^2$")
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=8, loc="best")
        ax.set_title("Accuracy-equity tradeoff (test set)")

        out_path = out_dir / f"{fname}.pdf"
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        plots[fname] = out_path

    return plots


def make_correlation_evolution_plot(test_df: pd.DataFrame, out_dir: Path) -> Dict[str, Path]:
    """Corr(r, price) vs rho for penalized families; horizontal lines for baselines."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.8, 4.4))

    base_lgb = test_df[test_df["model_name"] == "LGBMRegressor"].iloc[0]
    base_lr = test_df[test_df["model_name"] == "LinearRegression"].iloc[0]

    for family, marker, color in [
        ("LGBSmoothPenalty", "o", "tab:blue"),
        ("LGBCovPenalty", "s", "tab:orange"),
    ]:
        sub = test_df[test_df["model_name"] == family].dropna(subset=["rho"]).sort_values("rho")
        if sub.empty:
            continue
        ax.plot(sub["rho"], sub["Corr(r,price)"], marker=marker, color=color, label=f"{family}: Corr(r,P)", linewidth=1.2)
        ax.plot(sub["rho"], sub["Corr(r,logprice)"], marker=marker, color=color, linestyle="--", alpha=0.6,
                label=f"{family}: Corr(r,logP)", linewidth=1.0)

    ax.axhline(base_lgb["Corr(r,price)"], color="red", linestyle=":", linewidth=1.2, label="LGBM Corr(r,P)")
    ax.axhline(base_lr["Corr(r,price)"], color="grey", linestyle=":", linewidth=1.2, label="LR Corr(r,P)")
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_xscale("log")
    ax.set_xlabel(r"Penalty $\rho$")
    ax.set_ylabel("Correlation between ratio and price")
    ax.grid(True, which="both", alpha=0.4)
    ax.set_title("Ratio-price correlation vs penalty strength")
    ax.legend(fontsize=7, loc="best")

    out_path = out_dir / "correlations.pdf"
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return {"correlations": out_path}


def make_pareto_plot(
    test_df: pd.DataFrame,
    selection_summary_path: Path,
    out_dir: Path,
) -> Dict[str, Path]:  # noqa: C901

    """Scatter of LGBSmoothPenalty + LGBCovPenalty on (abs(PRD-1), OOS R2), with
    annotated CONSTRAINED / UTOPIA / NASH picks from the full-subset selection."""
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.8, 4.6))

    colours = {"LGBSmoothPenalty": "tab:blue", "LGBCovPenalty": "tab:orange"}
    for family, color in colours.items():
        sub = test_df[test_df["model_name"] == family].copy().sort_values("rho")
        if sub.empty:
            continue
        ax.plot(
            (sub["PRD"] - 1.0).abs(),
            sub["R2"],
            marker="o" if family == "LGBSmoothPenalty" else "s",
            color=color,
            alpha=0.75,
            linewidth=1.0,
            markersize=5,
            label=family,
        )
    # baselines
    base_lgb = test_df[test_df["model_name"] == "LGBMRegressor"].iloc[0]
    base_lr = test_df[test_df["model_name"] == "LinearRegression"].iloc[0]
    ax.scatter(
        abs(base_lgb["PRD"] - 1.0), base_lgb["R2"], marker="X", s=120, color="red", label="LGBM baseline", zorder=5,
    )
    ax.scatter(
        abs(base_lr["PRD"] - 1.0), base_lr["R2"], marker="D", s=80, color="grey", label="Linear baseline", zorder=5,
    )

    # annotate selections for the FULL constraint set (richest specification)
    if selection_summary_path.exists():
        sel = pd.read_csv(selection_summary_path)
        full_key = "PRD,PRB,VEI,COD,COV,MEAN_RATIO,MEDIAN_RATIO,WEIGHTED_MEAN_RATIO"
        if "row_kind" in sel.columns:
            sel = sel[sel["row_kind"] == "summary"]
        if "constraint_metrics" in sel.columns:
            sel_full = sel[sel["constraint_metrics"].astype(str) == full_key]
        else:
            sel_full = sel
        # Join with test metrics by config_id to recover held-out PRD/R2
        test_keyed = test_df.drop_duplicates(subset=["config_id"]).set_index("config_id")
        for solution, color, marker in [
            ("constrained", "green", "P"),
            ("utopia", "purple", "^"),
            ("nash", "gold", "*"),
        ]:
            match = sel_full[sel_full["selection_method"] == solution]
            if match.empty:
                continue
            row = match.iloc[0]
            cid = str(row.get("config_id", ""))
            if cid not in test_keyed.index:
                continue
            tr = test_keyed.loc[cid]
            try:
                prd = abs(float(tr["PRD"]) - 1.0)
                r2 = float(tr["R2"])
                ax.scatter([prd], [r2], color=color, marker=marker, s=180, edgecolors="black", linewidths=1.2,
                           label=solution.upper(), zorder=10)
            except Exception:
                continue

    ax.set_xlabel(r"$|\mathrm{PRD}-1|$")
    ax.set_ylabel(r"Held-out $R^2$")
    ax.set_title("Pareto view: tradeoff + selected solutions")
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=7, loc="best")

    out_path = out_dir / "pareto_optimal.pdf"
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return {"pareto_optimal": out_path}


# ---------------------------------------------------------------------------
# summary-json construction
# ---------------------------------------------------------------------------
def _load_bootstrap_ci(analysis_dir: Path) -> pd.DataFrame:
    """Load test-bootstrap across-folds percentile table (may be empty)."""
    p = analysis_dir / "test_bootstrap_metric_stats_by_config_across_folds.csv"
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


# --- targeted bootstrap CI (month-block) for only a few critical configs ---
def _ratio_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute the IAAO metrics cited in the slides exactly as in motivation_utils."""
    sys.path.insert(0, str(REPO_ROOT))
    from utils.motivation_utils import vei as _vei  # type: ignore
    from utils.motivation_utils import prd as _prd  # type: ignore
    from utils.motivation_utils import prb as _prb  # type: ignore
    from utils.motivation_utils import cod as _cod  # type: ignore
    from utils.motivation_utils import cov_iaao as _cov_iaao  # type: ignore

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0) & (y_pred > 0)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size < 20:
        return {}
    r = y_pred / y_true
    ss_res = float(np.sum((y_pred - y_true) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    out: Dict[str, float] = {
        "R2": float(1.0 - ss_res / max(ss_tot, 1e-12)),
        "MAE": float(np.mean(np.abs(y_pred - y_true))),
        "Mean ratio": float(np.mean(r)),
        "Median ratio": float(np.median(r)),
        "W. Mean ratio": float(np.sum(y_pred) / np.sum(y_true)),
        "COD": float(_cod(r)),
        "COV_IAAO": float(_cov_iaao(y_pred, y_true)),
        "PRD": float(_prd(y_pred, y_true)),
        "PRB": float(_prb(y_pred, y_true)),
        "VEI": float(_vei(y_pred, y_true)),
    }
    return out


def _month_block_bootstrap_cis(
    pred_df: pd.DataFrame,
    *,
    n_bootstrap: int = 200,
    seed: int = 2025,
) -> Dict[str, Dict[str, float]]:
    """Block-bootstrap by sale_month and return {metric: {q05,q50,q95,mean,std}}."""
    if pred_df.empty:
        return {}
    df = pred_df.copy()
    # Preferred: y_true / y_pred in original units
    if {"y_true_log", "y_pred_log"}.issubset(df.columns):
        y_true = np.exp(df["y_true_log"].to_numpy(dtype=float))
        y_pred = np.exp(df["y_pred_log"].to_numpy(dtype=float))
    elif {"y_true", "y_pred"}.issubset(df.columns):
        y_true = df["y_true"].to_numpy(dtype=float)
        y_pred = df["y_pred"].to_numpy(dtype=float)
    else:
        return {}
    if "sale_date" in df.columns:
        month = pd.to_datetime(df["sale_date"], errors="coerce").dt.to_period("M")
    else:
        month = pd.Series(["none"] * len(df))
    groups = month.to_numpy()
    unique_months = pd.unique(groups)
    rng = np.random.default_rng(seed)
    per_boot: Dict[str, List[float]] = {}
    for _ in range(n_bootstrap):
        sampled = rng.choice(unique_months, size=len(unique_months), replace=True)
        idx_parts = [np.where(groups == m)[0] for m in sampled]
        if not idx_parts:
            continue
        idx = np.concatenate(idx_parts)
        if idx.size < 10:
            continue
        m = _ratio_metrics(y_true[idx], y_pred[idx])
        for k, v in m.items():
            per_boot.setdefault(k, []).append(v)
    out: Dict[str, Dict[str, float]] = {}
    for k, vals in per_boot.items():
        arr = np.array(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size < 10:
            continue
        out[k] = {
            "q05": float(np.quantile(arr, 0.05)),
            "q50": float(np.quantile(arr, 0.50)),
            "q95": float(np.quantile(arr, 0.95)),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)),
            "n_bootstrap": int(arr.size),
        }
    return out


def bootstrap_selected_configs(
    analysis_dir: Path,
    config_ids: Iterable[str],
    *,
    n_bootstrap: int = 200,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Block-bootstrap the listed configs from test_predictions.parquet."""
    pred_path = analysis_dir / "test_predictions.parquet"
    if not pred_path.exists():
        return {}
    all_preds = pd.read_parquet(pred_path)
    if all_preds.empty or "config_id" not in all_preds.columns:
        return {}
    wanted = [cid for cid in {str(c) for c in config_ids} if cid]
    if not wanted:
        return {}
    cis: Dict[str, Dict[str, Dict[str, float]]] = {}
    for cid in wanted:
        sub = all_preds[all_preds["config_id"].astype(str) == cid]
        if sub.empty:
            continue
        cis[cid] = _month_block_bootstrap_cis(sub, n_bootstrap=n_bootstrap)
    return cis


def _cis_for_config(bs_df: pd.DataFrame, config_id: str) -> Dict[str, Dict[str, float]]:
    """Return {metric: {q05, q50, q95, mean, std}} for a config_id."""
    if bs_df.empty or not config_id:
        return {}
    sub = bs_df[bs_df["config_id"].astype(str) == str(config_id)]
    out: Dict[str, Dict[str, float]] = {}
    for _, r in sub.iterrows():
        metric = str(r.get("metric", ""))
        if not metric:
            continue
        out[metric] = {
            "q05": float(r.get("q05_avg_over_folds", np.nan)),
            "q50": float(r.get("q50_avg_over_folds", np.nan)),
            "q95": float(r.get("q95_avg_over_folds", np.nan)),
            "mean": float(r.get("mean_avg_over_folds", np.nan)),
            "std": float(r.get("std_avg_over_folds", np.nan)),
        }
    return out


def build_summary_json(
    *,
    year_tag: str,
    analysis_dir: Path,
    test_df: pd.DataFrame,
    penalized_row: pd.Series,
    selection_csvs: Dict[str, Path],
    figure_paths: Dict[str, Path],
) -> Dict[str, object]:
    baselines = pick_baseline_rows(test_df)
    lr = baselines.get("LinearRegression")
    lgb = baselines.get("LGBMRegressor")
    bs_df = _load_bootstrap_ci(analysis_dir)

    # Targeted block-bootstrap only for configs cited in the slides.
    cited_config_ids: List[str] = []
    for r in (lr, lgb, penalized_row):
        if r is None:
            continue
        cid = str(r.get("config_id", ""))
        if cid:
            cited_config_ids.append(cid)
    # A2 / A4 representative configs
    for r in (
        pick_representative_penalized(test_df, model_name="LGBSmoothPenalty", min_target=0.85, max_prd_dev=0.04),
        pick_representative_penalized(test_df, model_name="LGBSmoothPenalty", min_target=0.80, max_prd_dev=0.10),
        pick_representative_penalized(test_df, model_name="LGBCovPenalty", min_target=0.85, max_prd_dev=0.04),
        pick_representative_penalized(test_df, model_name="LGBCovPenalty", min_target=0.80, max_prd_dev=0.10),
    ):
        if r is None:
            continue
        cid = str(r.get("config_id", ""))
        if cid:
            cited_config_ids.append(cid)
    cited_config_ids = list(dict.fromkeys(cited_config_ids))
    _log(f"[summary] running targeted block-bootstrap (n=200) on {len(cited_config_ids)} configs ...")
    targeted_cis = bootstrap_selected_configs(analysis_dir, cited_config_ids, n_bootstrap=200)

    def _row_to_dict(r: pd.Series | None) -> Dict[str, float]:
        if r is None:
            return {}
        d: Dict[str, float] = {}
        for c in (
            "OOS R2", "R2", "MAE", "COD", "COV_IAAO", "PRD", "PRB", "VEI", "MKI",
            "Mean ratio", "Median ratio", "W. Mean ratio",
            "Corr(r,price)", "Corr(r,logprice)", "Slope(r~logy)", "rho",
        ):
            v = r.get(c, np.nan)
            try:
                v = float(v)
            except Exception:
                v = np.nan
            d[c] = v
        d["model_name"] = str(r.get("model_name", ""))
        d["config_id"] = str(r.get("config_id", ""))
        d["bootstrap_ci"] = (
            targeted_cis.get(d["config_id"], {}) or _cis_for_config(bs_df, d["config_id"])
        )
        return d

    appendix_rows = []
    for r in (
        pick_representative_penalized(test_df, model_name="LGBSmoothPenalty", min_target=0.85, max_prd_dev=0.04),
        pick_representative_penalized(test_df, model_name="LGBSmoothPenalty", min_target=0.80, max_prd_dev=0.10),
        pick_representative_penalized(test_df, model_name="LGBCovPenalty", min_target=0.85, max_prd_dev=0.04),
        pick_representative_penalized(test_df, model_name="LGBCovPenalty", min_target=0.80, max_prd_dev=0.10),
    ):
        if r is None:
            continue
        appendix_rows.append(_row_to_dict(r))

    # Appendix A4 rows: join simple_model_selection rows with test_metrics by config_id
    a4: Dict[str, List[Dict[str, object]]] = {}
    sel_csv = selection_csvs.get("SINGLE_CSV")
    if sel_csv and sel_csv.exists():
        sel_df = pd.read_csv(sel_csv)
        if "row_kind" in sel_df.columns:
            sel_df = sel_df[sel_df["row_kind"] == "summary"].copy()
        # Ensure the selection-picked configs also have bootstrap CIs.
        extra_cids = [
            str(c) for c in sel_df.get("config_id", pd.Series(dtype=str)).dropna().unique()
        ]
        missing = [c for c in extra_cids if c and c not in targeted_cis]
        if missing:
            extra_cis = bootstrap_selected_configs(analysis_dir, missing, n_bootstrap=200)
            targeted_cis.update(extra_cis)
        test_keyed = test_df.drop_duplicates(subset=["config_id"]).set_index("config_id")
        wanted_subsets = {
            "PRD": "PRD",
            "PRD_PRB_VEI": "PRD,PRB,VEI",
            "PRD_MEAN_COD": "PRD,MEAN_RATIO,COD",
            "FULL": "PRD,PRB,VEI,COD,COV,MEAN_RATIO,MEDIAN_RATIO,WEIGHTED_MEAN_RATIO",
        }
        for tag, constraint_key in wanted_subsets.items():
            sub = sel_df[sel_df["constraint_metrics"].astype(str) == constraint_key]
            rows: List[Dict[str, object]] = []
            for _, r in sub.iterrows():
                cid = str(r.get("config_id", ""))
                test_row = test_keyed.loc[cid] if cid in test_keyed.index else None
                merged = {
                    "selection_method": str(r.get("selection_method", "")),
                    "status": str(r.get("status", "")),
                    "model_name": str(r.get("model_name", "")),
                    "rho": _extract_rho(str(r.get("model_config_json", "{}"))),
                }
                if test_row is not None:
                    for c in ("OOS R2", "MAE", "COD", "COV_IAAO", "PRD", "PRB", "VEI",
                              "Median ratio", "Mean ratio", "W. Mean ratio", "MKI", "R2"):
                        v = test_row.get(c, np.nan) if hasattr(test_row, "get") else (test_row[c] if c in test_row.index else np.nan)
                        try:
                            v = float(v)
                        except Exception:
                            v = np.nan
                        merged[c] = v
                merged["bootstrap_ci"] = targeted_cis.get(cid, {})
                rows.append(merged)
            a4[tag] = rows

    summary = {
        "year_tag": year_tag,
        "analysis_dir": str(analysis_dir),
        "n_test": int(len(test_df)) if "n_test" not in test_df.columns else int(test_df["n_test"].iloc[0]),
        "slide10": {
            "LinearRegression": _row_to_dict(lr),
            "LGBMRegressor": _row_to_dict(lgb),
        },
        "slide15": {
            "baseline": _row_to_dict(lgb),
            "penalized": _row_to_dict(penalized_row),
        },
        "slide16": {
            "baseline": _row_to_dict(lgb),
            "penalized": _row_to_dict(penalized_row),
        },
        "appendix_a2": appendix_rows,
        "appendix_a4": a4,
        "figures": {k: str(v) for k, v in figure_paths.items()},
    }
    return summary


# ---------------------------------------------------------------------------
# top-level orchestration
# ---------------------------------------------------------------------------
def run_for_year(
    *,
    year_tag: str,
    result_root: Path,
    data_path: Path,
    img_out_dir: Path,
    build_dir: Path,
    skip_analyze: bool = False,
    skip_selection: bool = False,
) -> Path:
    """Return the path to ``slide_summary_{year_tag}.json``."""
    build_dir.mkdir(parents=True, exist_ok=True)
    img_out_dir.mkdir(parents=True, exist_ok=True)

    data_id, split_id, analysis_dir = resolve_split(result_root)
    _log(f"[{year_tag}] data_id={data_id} split_id={split_id}")
    _log(f"[{year_tag}] analysis_dir={analysis_dir}")

    # Step 0: enable test bootstrap by patching the CV protocol file (we ran
    # CV with n_bootstrap=0 to save compute, but the indices are regenerated
    # from the sale-date column inside analyze_results.py).
    protocol_path = (
        result_root / "protocol" / f"data_id={data_id}" / f"split_id={split_id}" / "folds.json"
    )
    target_n_bootstrap = 100
    if protocol_path.exists():
        try:
            payload = json.loads(protocol_path.read_text(encoding="utf-8"))
            bp = dict(payload.get("bootstrap_protocol", {}) or {})
            if int(bp.get("n_bootstrap", 0) or 0) != target_n_bootstrap:
                bp["n_bootstrap"] = target_n_bootstrap
                bp["block_freq"] = bp.get("block_freq", "M")
                bp["seed"] = int(bp.get("seed", 2025) or 2025)
                payload["bootstrap_protocol"] = bp
                protocol_path.write_text(json.dumps(payload, indent=2))
                _log(
                    f"[{year_tag}] patched bootstrap_protocol.n_bootstrap={target_n_bootstrap} "
                    f"in {protocol_path}"
                )
        except Exception as exc:
            _log(f"[{year_tag}] WARN: could not patch bootstrap protocol ({exc})")

    # Step 1: analyze_results.py (skip test bootstrap -- too slow when the
    # number of configs x test rows is large; we'll compute CIs ourselves
    # only for the configs cited in the slides).
    if not skip_analyze:
        _run([
            PYTHON_BIN, "analyze_results.py",
            "--result-root", str(result_root),
            "--data-id", data_id,
            "--split-id", split_id,
            "--skip-test-bootstrap",
            "--skip-final-vertical-equity",
        ])

    # Step 2: simple_model_selection.py for four constraint subsets
    subset_spec = "PRD;PRD,PRB,VEI;PRD,MEAN_RATIO,COD;PRD,PRB,VEI,COD,COV,MEAN_RATIO,MEDIAN_RATIO,WEIGHTED_MEAN_RATIO"
    if not skip_selection:
        _run([
            PYTHON_BIN, "simple_model_selection.py",
            "--result-root", str(result_root),
            "--data-id", data_id,
            "--split-id", split_id,
            "--constraint-metric-subsets", subset_spec,
            "--selection-method", "both",
        ])

    # Step 3: load data
    test_df = augment_test_metrics(analysis_dir)
    penalized_row = pick_representative_penalized(test_df)
    if penalized_row is None:
        raise RuntimeError("could not pick a representative penalized model")
    _log(
        f"[{year_tag}] picked penalized: {penalized_row['model_name']} "
        f"rho={penalized_row['rho']:.4g} R2={penalized_row['R2']:.4f} PRD={penalized_row['PRD']:.4f}"
    )
    baseline_rows = pick_baseline_rows(test_df)

    # Step 4: figure generation
    figure_paths: Dict[str, Path] = {}
    if "LGBMRegressor" in baseline_rows:
        mot = make_motivation_plots(
            analysis_dir=analysis_dir,
            baseline_lgb_row=baseline_rows["LGBMRegressor"],
            penalized_row=penalized_row,
            out_dir=img_out_dir / "motivation",
        )
        for k, v in mot.items():
            figure_paths[f"motivation_{k}"] = v

    tr = make_tradeoff_band_plots(test_df, img_out_dir / "tradeoffs")
    figure_paths.update({f"tradeoff_{k}": v for k, v in tr.items()})

    co = make_correlation_evolution_plot(test_df, img_out_dir / "correlations")
    figure_paths.update({f"correlation_{k}": v for k, v in co.items()})

    sel_csv = analysis_dir / "simple_model_selection" / "selection_summary.csv"
    pa = make_pareto_plot(test_df, sel_csv, img_out_dir / "model_selection")
    figure_paths.update({f"pareto_{k}": v for k, v in pa.items()})

    # Subset-level A4 rows are all inside the single selection_summary.csv.
    subset_tags = {"SINGLE_CSV": sel_csv}

    # Step 5: summary json
    summary = build_summary_json(
        year_tag=year_tag,
        analysis_dir=analysis_dir,
        test_df=test_df,
        penalized_row=penalized_row,
        selection_csvs=subset_tags,
        figure_paths=figure_paths,
    )
    out_json = build_dir / f"slide_summary_{year_tag}.json"
    out_json.write_text(json.dumps(summary, indent=2, default=str))
    _log(f"[{year_tag}] wrote {out_json}")
    return out_json


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--year-tag", required=True, choices=["2023", "2024"])
    ap.add_argument("--result-root", required=True)
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--img-out-dir", required=True, help="e.g. Paper/GE slides/img2023")
    ap.add_argument("--build-dir", default="slide_build")
    ap.add_argument("--skip-analyze", action="store_true")
    ap.add_argument("--skip-selection", action="store_true")
    args = ap.parse_args()

    run_for_year(
        year_tag=args.year_tag,
        result_root=Path(args.result_root),
        data_path=Path(args.data_path),
        img_out_dir=Path(args.img_out_dir),
        build_dir=Path(args.build_dir),
        skip_analyze=bool(args.skip_analyze),
        skip_selection=bool(args.skip_selection),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
