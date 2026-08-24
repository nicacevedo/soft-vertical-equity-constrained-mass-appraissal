"""Read-only analysis of frozen paper-v6 penalty paths.

No fitting and no rho/model selection. Representative ratio-shape plots use a
prespecified rho set only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterSciNotation, LogLocator
import numpy as np
import pandas as pd

from utils.plotting_utils import plot_vertical_equity_lowess


HEADLINE_METRICS: Tuple[str, ...] = (
    "R2_price",
    "R2_log",
    "RMSE_price",
    "RMSE_log",
    "MAE_price",
    "MAPE",
    "MdAPE",
    "Median ratio",
    "Mean ratio",
    "W. Mean ratio",
    "COD",
    "COV_IAAO",
    "PRD",
    "PRB",
    "VEI",
    "MKI",
    "Cov_log_residual_log_price",
    "Beta_log",
    "Corr_log_residual_log_price",
    "dCor_e_y",
)

ACCURACY_EQUITY_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("R2_price", "PRD"),
    ("R2_price", "PRB"),
    ("R2_price", "VEI"),
)

PENALTY_FAMILIES: Tuple[str, ...] = ("LGBCovPenalty", "LGBSmoothPenalty")
SHAPE_RHOS: Tuple[float, ...] = (0.0, 0.1, 1.0, 10.0, 100.0)


def _parse_config(raw: Any) -> Dict[str, Any]:
    if raw is None or (isinstance(raw, float) and not np.isfinite(raw)):
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    text = str(raw).strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except Exception:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _rho_from_row(row: pd.Series) -> float:
    if "rho" in row.index and pd.notna(row.get("rho")):
        try:
            return float(row["rho"])
        except (TypeError, ValueError):
            pass
    cfg = _parse_config(row.get("model_config_json"))
    if "rho" in cfg:
        try:
            return float(cfg["rho"])
        except (TypeError, ValueError):
            return np.nan
    return np.nan


def _discover_data_split_pairs(result_root: Path) -> List[Tuple[str, str]]:
    runs_root = result_root / "runs"
    pairs = []
    if runs_root.is_dir():
        for data_dir in sorted(runs_root.glob("data_id=*")):
            if not data_dir.is_dir():
                continue
            data_id = data_dir.name.split("data_id=", 1)[-1]
            for split_dir in sorted(data_dir.glob("split_id=*")):
                if split_dir.is_dir():
                    pairs.append((data_id, split_dir.name.split("split_id=", 1)[-1]))
    for analysis_root in (
        result_root / "analysis",
        result_root / "baseline_reporting" / "analysis",
    ):
        if not analysis_root.is_dir():
            continue
        for data_dir in sorted(analysis_root.glob("data_id=*")):
            data_id = data_dir.name.split("data_id=", 1)[-1]
            for split_dir in sorted(data_dir.glob("split_id=*")):
                pair = (data_id, split_dir.name.split("split_id=", 1)[-1])
                if pair not in pairs:
                    pairs.append(pair)
    return pairs


def resolve_experiment_ids(result_root: Path, data_id: Optional[str], split_id: Optional[str]) -> Tuple[str, str]:
    if data_id and split_id:
        return str(data_id), str(split_id)
    completion = result_root / "cv_completion.json"
    if completion.is_file():
        blob = json.loads(completion.read_text(encoding="utf-8"))
        cid, sid = blob.get("data_id"), blob.get("split_id")
        if cid and sid:
            if data_id and str(data_id) != str(cid):
                raise SystemExit(f"--data-id {data_id!r} does not match cv_completion.json {cid!r}")
            if split_id and str(split_id) != str(sid):
                raise SystemExit(f"--split-id {split_id!r} does not match cv_completion.json {sid!r}")
            return str(cid), str(sid)
    pairs = _discover_data_split_pairs(result_root)
    if data_id:
        pairs = [p for p in pairs if p[0] == str(data_id)]
    if split_id:
        pairs = [p for p in pairs if p[1] == str(split_id)]
    uniq = sorted(set(pairs))
    if len(uniq) == 1:
        return uniq[0]
    if not uniq:
        raise SystemExit(f"No data_id/split_id artifacts found under {result_root}")
    raise SystemExit(
        "Multiple experiments found; pass --data-id and --split-id explicitly. "
        f"Candidates: {uniq}"
    )


def _load_cv_runs(result_root: Path, data_id: str, split_id: str) -> pd.DataFrame:
    runs_root = result_root / "runs" / f"data_id={data_id}" / f"split_id={split_id}"
    files = sorted(runs_root.glob("fold_id=*/*.parquet"))
    if not files:
        return pd.DataFrame()
    cv = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    cv["split"] = "cv"
    cv["rho"] = cv.apply(_rho_from_row, axis=1)
    return cv


def _load_stage_metrics(analysis_dir: Path, prefix: str, split_name: str) -> pd.DataFrame:
    path = analysis_dir / f"{prefix}_metrics.csv"
    if not path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["split"] = split_name
    df["rho"] = df.apply(_rho_from_row, axis=1)
    return df


def _metric_columns(df: pd.DataFrame) -> List[str]:
    present = [m for m in HEADLINE_METRICS if m in df.columns]
    for extra in ("OOS_R2_price", "OOS_R2_log", "RMSE", "MAE", "R2", "R2 (log)"):
        if extra in df.columns and extra not in present:
            present.append(extra)
    return present


def build_combined_path_table(
    cv: pd.DataFrame,
    test: pd.DataFrame,
    forward: pd.DataFrame,
) -> pd.DataFrame:
    frames = [d for d in (cv, test, forward) if d is not None and not d.empty]
    if not frames:
        return pd.DataFrame()
    metrics = _metric_columns(pd.concat(frames, ignore_index=True, sort=False))
    identities: List[Tuple[str, float]] = []
    for src in (cv, test, forward):
        if src is None or src.empty:
            continue
        for model_name, rho in src[["model_name", "rho"]].drop_duplicates().itertuples(index=False, name=None):
            identities.append((str(model_name), float(rho) if pd.notna(rho) else np.nan))
    uniq: List[Tuple[str, float]] = []
    seen = set()
    for model_name, rho in identities:
        key = (model_name, None if not np.isfinite(rho) else round(float(rho), 12))
        if key in seen:
            continue
        seen.add(key)
        uniq.append((model_name, rho))

    rows: List[Dict[str, Any]] = []
    for model_name, rho in uniq:
        rec: Dict[str, Any] = {"model_name": model_name, "rho": rho}
        if not cv.empty:
            mask = cv["model_name"].astype(str) == model_name
            mask &= np.isclose(cv["rho"].astype(float), float(rho), rtol=0.0, atol=1e-12) if np.isfinite(rho) else cv["rho"].isna()
            cv_sub = cv.loc[mask]
            if not cv_sub.empty and "fold_id" in cv_sub.columns:
                rec["n_folds"] = int(cv_sub["fold_id"].nunique())
                for metric in metrics:
                    if metric not in cv_sub.columns:
                        continue
                    by_fold = cv_sub.groupby("fold_id", dropna=False)[metric].mean()
                    for fold_id, val in by_fold.items():
                        rec[f"{metric}__cv_fold_{int(fold_id)}"] = float(val) if pd.notna(val) else np.nan
                    rec[f"{metric}__cv_mean"] = float(by_fold.mean()) if by_fold.size else np.nan
                    rec[f"{metric}__cv_sd"] = float(by_fold.std(ddof=1)) if by_fold.size > 1 else np.nan
        for split_name, src in (("test", test), ("forward", forward)):
            if src is None or src.empty:
                continue
            mask = src["model_name"].astype(str) == model_name
            mask &= np.isclose(src["rho"].astype(float), float(rho), rtol=0.0, atol=1e-12) if np.isfinite(rho) else src["rho"].isna()
            sub = src.loc[mask]
            if sub.empty:
                continue
            for metric in metrics:
                if metric in sub.columns:
                    rec[f"{metric}__{split_name}"] = float(sub[metric].iloc[0])
        rows.append(rec)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["model_name", "rho"], na_position="first").reset_index(drop=True)
    return out


def _family_color(name: str) -> str:
    return {
        "LinearRegression": "#4B5563",
        "LGBMRegressor": "#111827",
        "LGBCovPenalty": "#1D4ED8",
        "LGBSmoothPenalty": "#0F766E",
    }.get(str(name), "#6B7280")


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    pdf_path = path.with_suffix(".pdf")
    fig.savefig(pdf_path)
    plt.close(fig)


def _plot_rho_traces(combined: pd.DataFrame, metrics: Sequence[str], out_dir: Path) -> None:
    families = [n for n in PENALTY_FAMILIES if n in set(combined["model_name"].astype(str))] or list(PENALTY_FAMILIES)
    for metric in metrics:
        mean_col = f"{metric}__cv_mean"
        sd_col = f"{metric}__cv_sd"
        if mean_col not in combined.columns:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
        for ax, family in zip(axes, families[:2]):
            sub = combined.loc[combined["model_name"].astype(str) == family].copy()
            if sub.empty:
                ax.set_axis_off()
                continue
            zero = sub.loc[np.isclose(sub["rho"].fillna(-1.0), 0.0)]
            pos = sub.loc[sub["rho"] > 0].sort_values("rho")
            color = _family_color(family)
            if not pos.empty:
                x = pos["rho"].to_numpy(dtype=float)
                y = pos[mean_col].to_numpy(dtype=float)
                ax.plot(x, y, color=color, marker="o", linewidth=1.6, label="CV mean")
                if sd_col in pos.columns:
                    sd = pos[sd_col].to_numpy(dtype=float)
                    ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.18, label="CV SD")
                if f"{metric}__test" in pos.columns:
                    ax.plot(x, pos[f"{metric}__test"], color=color, linestyle="--", marker="s", label="Held-out test")
                if f"{metric}__forward" in pos.columns:
                    ax.plot(x, pos[f"{metric}__forward"], color=color, linestyle=":", marker="^", label="2025 forward")
            if not zero.empty:
                x0 = pos["rho"].min() * 0.7 if not pos.empty else 0.1
                ax.scatter(np.full(len(zero), x0), zero[mean_col], color=color, marker="D", zorder=5, label="rho=0 (CV)")
            ax.set_xscale("log")
            ax.set_title(family)
            ax.set_xlabel("rho")
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        _save_fig(fig, out_dir / f"rho_trace_{metric.replace(' ', '_')}.png")


def _plot_accuracy_equity(combined: pd.DataFrame, out_dir: Path) -> None:
    splits = (
        ("cv_mean", "CV mean"),
        ("test", "Held-out"),
        ("forward", "2025 forward"),
    )
    for x_metric, y_metric in ACCURACY_EQUITY_PAIRS:
        fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharex=False, sharey=False)
        drew = False
        for ax, (suffix, title) in zip(axes, splits):
            x_col = f"{x_metric}__{suffix}"
            y_col = f"{y_metric}__{suffix}"
            any_series = False
            for family in PENALTY_FAMILIES:
                sub = combined.loc[combined["model_name"].astype(str) == family].copy()
                if sub.empty or x_col not in sub.columns or y_col not in sub.columns:
                    continue
                sub = sub.sort_values("rho")
                ax.plot(
                    sub[x_col],
                    sub[y_col],
                    color=_family_color(family),
                    marker="o",
                    linewidth=1.5,
                    label=family,
                )
                any_series = True
                drew = True
            ax.set_title(title)
            ax.set_xlabel(x_metric)
            ax.set_ylabel(y_metric)
            ax.grid(True, alpha=0.3)
            if any_series:
                ax.legend(fontsize=7)
            else:
                ax.set_axis_off()
        if drew:
            fig.suptitle(f"{x_metric} vs {y_metric} along the ordered rho path", fontsize=11)
            _save_fig(fig, out_dir / f"path_{x_metric}_vs_{y_metric}.png")
        else:
            plt.close(fig)


def _plot_fold_stability(cv: pd.DataFrame, metrics: Sequence[str], out_dir: Path) -> None:
    if cv.empty or "fold_id" not in cv.columns:
        return
    for family in PENALTY_FAMILIES:
        sub = cv.loc[cv["model_name"].astype(str) == family].copy()
        if sub.empty:
            continue
        for metric in metrics:
            if metric not in sub.columns:
                continue
            fig, ax = plt.subplots(figsize=(7.5, 4.6))
            pos = sub.loc[sub["rho"] > 0]
            for fold_id, fold_df in pos.groupby("fold_id"):
                fold_df = fold_df.sort_values("rho")
                ax.plot(fold_df["rho"], fold_df[metric], alpha=0.45, linewidth=1.0, label=f"fold {int(fold_id)}")
            mean_df = pos.groupby("rho", as_index=False)[metric].mean().sort_values("rho")
            ax.plot(mean_df["rho"], mean_df[metric], color="black", linewidth=2.2, label="equal-weight mean")
            ax.set_xscale("log")
            ax.set_xlabel("rho")
            ax.set_ylabel(metric)
            ax.set_title(f"{family}: seven-fold {metric} paths")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, ncol=2)
            _save_fig(fig, out_dir / f"fold_paths_{family}_{metric.replace(' ', '_')}.png")


def _closest_rho(values: Iterable[float], target: float) -> Optional[float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(arr[np.argmin(np.abs(arr - float(target)))])


def _plot_ratio_shapes(result_root: Path, cv: pd.DataFrame, out_dir: Path) -> None:
    pred_root = result_root / "predictions"
    if not pred_root.is_dir() or cv.empty or "fold_id" not in cv.columns:
        return
    last_fold = int(cv["fold_id"].max())
    for family in PENALTY_FAMILIES:
        available = cv.loc[cv["model_name"].astype(str) == family, "rho"].dropna().unique().tolist()
        for target in SHAPE_RHOS:
            chosen = _closest_rho(available, target)
            if chosen is None:
                continue
            if (not np.isclose(target, 0.0)) and abs(chosen - target) > 0.25 * max(target, 0.1):
                continue
            match = cv.loc[
                (cv["model_name"].astype(str) == family)
                & np.isclose(cv["rho"].astype(float), chosen, rtol=0.0, atol=1e-12)
                & (cv["fold_id"] == last_fold)
            ]
            if match.empty or "run_id" not in match.columns:
                continue
            run_id = str(match["run_id"].iloc[0])
            files = list(pred_root.glob(f"**/{run_id}.parquet"))
            if not files:
                continue
            pred = pd.read_parquet(files[0])
            y_true = pred["y_true"].to_numpy() if "y_true" in pred.columns else np.exp(pred["y_true_log"].to_numpy())
            y_pred = pred["y_pred"].to_numpy() if "y_pred" in pred.columns else np.exp(pred["y_pred_log"].to_numpy())
            y_log = np.log(y_true)
            ratios = y_pred / y_true
            plot_vertical_equity_lowess(
                y_log,
                ratios,
                out_path=out_dir / f"ratio_shape_{family}_rho_{chosen:g}.png",
                model_label=f"{family}, rho={chosen:g} (prespecified; not selected)",
                sample_size=8000,
                lowess_frac=0.4,
            )


BASELINE_MODELS: Tuple[str, ...] = ("LinearRegression", "LGBMRegressor")
SECTION2_EXPORT_COLUMNS: Tuple[str, ...] = (
    "evaluation",
    "model",
    "R2_price",
    "MAE_price",
    "MAPE_price_pct",
    "RMSE_log",
    "Median_ratio",
    "Mean_ratio",
    "Weighted_mean_ratio",
    "COD_pct",
    "COV_pct",
    "PRD",
    "PRB",
    "MKI",
    "VEI_pct",
    "Beta_log",
    "dCor_e_y",
)


def _section2_row(metrics: Dict[str, Any], *, evaluation: str, model: str) -> Dict[str, Any]:
    return {
        "evaluation": evaluation,
        "model": model,
        "R2_price": metrics.get("R2_price"),
        "MAE_price": metrics.get("MAE_price", metrics.get("MAE")),
        "MAPE_price_pct": 100.0 * float(metrics["MAPE"]) if metrics.get("MAPE") is not None else np.nan,
        "RMSE_log": metrics.get("RMSE_log"),
        "Median_ratio": metrics.get("Median ratio"),
        "Mean_ratio": metrics.get("Mean ratio"),
        "Weighted_mean_ratio": metrics.get("W. Mean ratio"),
        "COD_pct": metrics.get("COD"),
        "COV_pct": 100.0 * float(metrics["COV_IAAO"]) if metrics.get("COV_IAAO") is not None else np.nan,
        "PRD": metrics.get("PRD"),
        "PRB": metrics.get("PRB"),
        "MKI": metrics.get("MKI"),
        "VEI_pct": metrics.get("VEI"),
        "Beta_log": metrics.get("Beta_log"),
        "dCor_e_y": metrics.get("dCor_e_y"),
    }


def _sale_price_bin_profile(pred: pd.DataFrame, n_bins: int = 30) -> pd.DataFrame:
    rows = []
    for (evaluation, model), group in pred.groupby(["evaluation", "model"], sort=True):
        ordered = group.sort_values("sale_price", kind="mergesort")
        if ordered.empty:
            continue
        for bin_id, idx in enumerate(np.array_split(np.arange(len(ordered)), n_bins), start=1):
            part = ordered.iloc[idx]
            ratios = part["valuation_to_sale_ratio"].to_numpy(dtype=float)
            rows.append(
                {
                    "evaluation": evaluation,
                    "model": model,
                    "bin": int(bin_id),
                    "n": int(len(part)),
                    "median_sale_price": float(part["sale_price"].median()),
                    "median_ratio": float(np.median(ratios)),
                    "ratio_q25": float(np.quantile(ratios, 0.25)),
                    "ratio_q75": float(np.quantile(ratios, 0.75)),
                    "beta_log": float(part["beta_log"].iloc[0]) if "beta_log" in part.columns else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _plot_section2_baseline_bins(profile: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.2), sharex=True, sharey=True)
    evals = ("Held-out evaluation", "2025 forward evaluation")
    models = BASELINE_MODELS
    if profile.empty:
        return
    x_min = float(profile["median_sale_price"].min())
    x_max = float(profile["median_sale_price"].max())
    for row, evaluation in enumerate(evals):
        for col, model in enumerate(models):
            ax = axes[row, col]
            sub = profile.loc[(profile["evaluation"] == evaluation) & (profile["model"] == model)]
            if sub.empty:
                ax.set_axis_off()
                continue
            ax.fill_between(sub["median_sale_price"], sub["ratio_q25"], sub["ratio_q75"], color="#1D4ED8", alpha=0.18, linewidth=0)
            ax.plot(sub["median_sale_price"], sub["median_ratio"], color="#1D4ED8", marker="o", markersize=2.5, linewidth=1.4)
            ax.axhline(1.0, color="#111827", linestyle="--", linewidth=0.9)
            ax.set_xscale("log", base=10)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(0.55, 1.45)
            ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            ax.xaxis.set_major_formatter(
                LogFormatterSciNotation(base=10, labelOnlyBase=False, minor_thresholds=(np.inf, np.inf))
            )
            beta = sub["beta_log"].iloc[0] if "beta_log" in sub.columns else np.nan
            if np.isfinite(beta):
                ax.text(0.04, 0.08, rf"$\beta_{{\log}}={beta:.3f}$", transform=ax.transAxes, fontsize=8)
            display = {"LinearRegression": "Linear", "LGBMRegressor": "LightGBM"}.get(model, model)
            short_eval = "Held-out" if "Held-out" in evaluation else "2025 forward"
            ax.set_title(f"{short_eval} {display}")
            if col == 0:
                ax.set_ylabel("Valuation-to-sale ratio")
            if row == 1:
                ax.set_xlabel("Sale price")
            ax.grid(True, color="#E5E7EB", linewidth=0.7)
            ax.set_axisbelow(True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_vei_group_profile(profile: pd.DataFrame, out_path: Path) -> None:
    if profile.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.2), sharex=True, sharey=True)
    evals = ("Held-out evaluation", "2025 forward evaluation")
    for row, evaluation in enumerate(evals):
        for col, model in enumerate(BASELINE_MODELS):
            ax = axes[row, col]
            sub = profile.loc[(profile["evaluation"] == evaluation) & (profile["model"] == model)]
            if sub.empty:
                ax.set_axis_off()
                continue
            group_col = "group" if "group" in sub.columns else "group"
            med_col = "median_ratio" if "median_ratio" in sub.columns else "median_ratio"
            low_col = "ci_low" if "ci_low" in sub.columns else "ci_low"
            high_col = "ci_high" if "ci_high" in sub.columns else "ci_high"
            overall_col = "overall_median_ratio" if "overall_median_ratio" in sub.columns else "overall_median_ratio"
            x = sub[group_col].to_numpy(dtype=float)
            ax.fill_between(x, sub[low_col], sub[high_col], color="#1D4ED8", alpha=0.18, linewidth=0)
            ax.plot(x, sub[med_col], color="#1D4ED8", marker="o", linewidth=1.4)
            overall = float(sub[overall_col].iloc[0])
            if np.isfinite(overall):
                ax.axhline(overall, color="#111827", linestyle=":", linewidth=0.9, label="overall median ratio")
            ax.axhline(1.0, color="#6B7280", linestyle="--", linewidth=0.8)
            display = {"LinearRegression": "Linear", "LGBMRegressor": "LightGBM"}.get(model, model)
            short_eval = "Held-out" if "Held-out" in evaluation else "2025 forward"
            ax.set_title(f"{short_eval} {display}")
            if col == 0:
                ax.set_ylabel("Group median valuation-to-sale ratio")
            if row == 1:
                ax.set_xlabel("VEI percentile group (low to high value)")
            ax.grid(True, color="#E5E7EB", linewidth=0.7)
            ax.set_axisbelow(True)
            if row == 0 and col == 1:
                ax.legend(fontsize=7, loc="best")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def write_section2_outputs(analysis_dir: Path, out_dir: Path) -> None:
    from utils.motivation_utils import compute_taxation_metrics, vei_percentile_group_profile

    frames = []
    pred_frames = []
    vei_frames = []
    mapping = (
        ("test_predictions.parquet", "Held-out evaluation"),
        ("assess_predictions.parquet", "2025 forward evaluation"),
    )
    for fname, evaluation in mapping:
        path = analysis_dir / fname
        if not path.is_file():
            continue
        pred = pd.read_parquet(path)
        name_col = "model_name" if "model_name" in pred.columns else "model"
        y_true = pred["y_true"] if "y_true" in pred.columns else np.exp(pred["y_true_log"])
        y_pred = pred["y_pred"] if "y_pred" in pred.columns else np.exp(pred["y_pred_log"])
        y_true_log = pred["y_true_log"] if "y_true_log" in pred.columns else np.log(y_true)
        y_pred_log = pred["y_pred_log"] if "y_pred_log" in pred.columns else np.log(y_pred)
        work = pred.copy()
        work["evaluation"] = evaluation
        work["model"] = work[name_col].astype(str)
        work["sale_price"] = np.asarray(y_true, dtype=float)
        work["predicted_price"] = np.asarray(y_pred, dtype=float)
        work["valuation_to_sale_ratio"] = work["predicted_price"] / work["sale_price"]
        for model in BASELINE_MODELS:
            sub = work.loc[work["model"] == model]
            if sub.empty:
                continue
            metrics = compute_taxation_metrics(
                sub["y_true_log"] if "y_true_log" in sub.columns else np.log(sub["sale_price"]),
                sub["y_pred_log"] if "y_pred_log" in sub.columns else np.log(sub["predicted_price"]),
                scale="log",
            )
            frames.append(_section2_row(metrics, evaluation=evaluation, model=model))
            sub = sub.copy()
            sub["beta_log"] = metrics.get("Beta_log", np.nan)
            pred_frames.append(sub)
            profile = vei_percentile_group_profile(sub["predicted_price"], sub["sale_price"])
            if not profile.empty:
                profile["evaluation"] = evaluation
                profile["model"] = model
                vei_frames.append(profile)
    if frames:
        table = pd.DataFrame(frames)[list(SECTION2_EXPORT_COLUMNS)]
        table_path = out_dir / "section2_baseline_table.csv"
        table.to_csv(table_path, index=False)
        _write_section2_latex(table, out_dir / "section2_baseline_table.tex")
        print(f"wrote {table_path}")
    if pred_frames:
        profile = _sale_price_bin_profile(pd.concat(pred_frames, ignore_index=True), n_bins=30)
        profile_path = out_dir / "section2_baseline_ratio_bins.csv"
        profile.to_csv(profile_path, index=False)
        fig_path = out_dir / "section2_baseline_ratio_bins.png"
        _plot_section2_baseline_bins(profile, fig_path)
        print(f"wrote {fig_path}")
    if vei_frames:
        vei_df = pd.concat(vei_frames, ignore_index=True)
        vei_path = out_dir / "vei_percentile_group_profile.csv"
        vei_df.to_csv(vei_path, index=False)
        _plot_vei_group_profile(vei_df, out_dir / "vei_percentile_group_profile.png")
        print(f"wrote {vei_path}")


def _write_section2_latex(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in SECTION2_EXPORT_COLUMNS if c in table.columns]
    lines = [
        r"\begin{tabular}{l" + "r" * (len(cols) - 1) + r"}",
        r"\toprule",
        " & ".join(cols) + r" \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        cells = []
        for c in cols:
            val = row[c]
            if isinstance(val, (float, np.floating)):
                cells.append(f"{float(val):.3f}")
            else:
                cells.append(str(val))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _endpoint_path_summary(combined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family in PENALTY_FAMILIES:
        sub = combined.loc[combined["model_name"].astype(str) == family].copy()
        if sub.empty or "rho" not in sub.columns:
            continue
        sub = sub.sort_values("rho")
        zero = sub.loc[np.isclose(sub["rho"].fillna(-1.0), 0.0)]
        pos = sub.loc[sub["rho"] > 0]
        picks = []
        if not zero.empty:
            picks.append(("rho=0", zero.iloc[0]))
        if not pos.empty:
            picks.append(("lowest_positive_rho", pos.iloc[0]))
            picks.append(("highest_positive_rho", pos.iloc[-1]))
        for label, rec in picks:
            row = {"family": family, "grid_location": label, "rho": rec.get("rho")}
            for metric in ("R2_price", "PRD", "PRB", "MKI", "VEI", "Beta_log"):
                for suffix in ("cv_mean", "cv_sd", "test", "forward"):
                    col = f"{metric}__{suffix}"
                    if col in rec.index:
                        row[col] = rec[col]
            rows.append(row)
    return pd.DataFrame(rows)


def _copy_paper_assets(src_dir: Path, paper_dir: Path, manuscript_dir: Path) -> None:
    paper_dir.mkdir(parents=True, exist_ok=True)
    (paper_dir / "tables").mkdir(exist_ok=True)
    (paper_dir / "figures").mkdir(exist_ok=True)
    (paper_dir / "sources").mkdir(exist_ok=True)
    manuscript_dir.mkdir(parents=True, exist_ok=True)
    for csv in src_dir.glob("*.csv"):
        target = paper_dir / "tables" / csv.name if "table" in csv.name or "path" in csv.name else paper_dir / "sources" / csv.name
        target.write_bytes(csv.read_bytes())
    for tex in src_dir.glob("*.tex"):
        (paper_dir / "tables" / tex.name).write_bytes(tex.read_bytes())
    for fig in list(src_dir.glob("*.png")) + list(src_dir.glob("*.pdf")):
        if fig.name.startswith("ratio_shape_"):
            continue
        (paper_dir / "figures" / fig.name).write_bytes(fig.read_bytes())
        (manuscript_dir / fig.name).write_bytes(fig.read_bytes())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze frozen paper-v6 penalty paths with no fitting or selection.")
    p.add_argument("--result-root", type=str, default="./output/paper_v6_preselection")
    p.add_argument("--data-id", type=str, default=None)
    p.add_argument("--split-id", type=str, default=None)
    p.add_argument(
        "--paper-asset-dir",
        type=str,
        default="./output/paper_v6_preselection/paper_outputs",
        help="Directory for manuscript-ready copies of tables/figures.",
    )
    p.add_argument(
        "--manuscript-figure-dir",
        type=str,
        default="./paper/img/generated_v6_preselection",
    )
    p.add_argument(
        "--preview",
        action="store_true",
        help="Mark outputs as smoke/preview; do not treat numbers as final paper results.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result_root = Path(args.result_root)
    data_id, split_id = resolve_experiment_ids(result_root, args.data_id, args.split_id)
    analysis_dir = result_root / "analysis" / f"data_id={data_id}" / f"split_id={split_id}"
    baseline_analysis_dir = (
        result_root / "baseline_reporting" / "analysis" / f"data_id={data_id}" / f"split_id={split_id}"
    )
    out_dir = analysis_dir / "penalty_path_analysis"
    if args.preview:
        out_dir = result_root / "preview" / "penalty_path_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    cv = _load_cv_runs(result_root, data_id, split_id)
    test = _load_stage_metrics(analysis_dir, "test", "test")
    forward = _load_stage_metrics(analysis_dir, "assess", "forward")
    combined = build_combined_path_table(cv, test, forward)
    table_path = out_dir / "combined_penalty_path_table.csv"
    if not combined.empty:
        combined.to_csv(table_path, index=False)
        endpoints = _endpoint_path_summary(combined)
        if not endpoints.empty:
            endpoints.to_csv(out_dir / "rho_path_endpoint_summary.csv", index=False)

    trace_metrics = [
        m
        for m in (
            "R2_price",
            "MAE_price",
            "MAPE",
            "RMSE_log",
            "PRD",
            "PRB",
            "MKI",
            "VEI",
            "Beta_log",
            "Cov_log_residual_log_price",
            "dCor_e_y",
        )
        if combined.empty or f"{m}__cv_mean" in combined.columns or f"{m}__test" in combined.columns
    ]
    if not combined.empty:
        _plot_rho_traces(combined, trace_metrics, out_dir)
        _plot_accuracy_equity(combined, out_dir)
        _plot_fold_stability(cv, ["R2_price", "PRD", "PRB", "VEI", "Beta_log"], out_dir)
        _plot_ratio_shapes(result_root, cv, out_dir)

    section2_dir = baseline_analysis_dir if (baseline_analysis_dir / "test_predictions.parquet").is_file() else analysis_dir
    write_section2_outputs(section2_dir, out_dir)
    if not args.preview:
        _copy_paper_assets(out_dir, Path(args.paper_asset_dir), Path(args.manuscript_figure_dir))
    print(f"wrote {table_path}")
    print(f"figures under {out_dir}")
    print(f"experiment data_id={data_id} split_id={split_id}")
    if args.preview:
        print("PREVIEW/SMOKE outputs only; numbers are not final paper results.")


if __name__ == "__main__":
    main()
