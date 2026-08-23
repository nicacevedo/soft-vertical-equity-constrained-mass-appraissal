#!/usr/bin/env python
"""Collect linear and LGBM projection-theory comparison artifacts.

The output is a manuscript-facing comparison layer: exact linear projection-path
checks and retrained-LGBM approximation checks in one table with shared error
summaries and diagnostic plots.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


FAMILY_COLORS = {
    "lgbm_retrained_local_approx": "#2563eb",
    "linear_exact_projection": "#dc2626",
}

SPLIT_MARKERS = {
    "assessment": "o",
    "test": "s",
    "train_assess_fit": "^",
    "train_test_fit": "v",
}

CONFIG_MARKERS = {
    "cv_top1_r2": "o",
    "cv_top2_r2": "s",
    "test_best_r2": "^",
}


def _read_csvs(paths: List[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"[collect] skipping {path}: {exc}", flush=True)
            continue
        df["source_file"] = str(path)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_linear(root: Path) -> pd.DataFrame:
    paths = sorted(root.glob("*/linear_projection_theory_empirical_comparison.csv"))
    df = _read_csvs(paths)
    if df.empty:
        return df
    df["comparison_family"] = "linear_exact_projection"
    if "q_theory" not in df.columns and "q_target" in df.columns:
        df["q_theory"] = pd.to_numeric(df["q_target"], errors="coerce")
    baseline_rows = []
    for comparison_path in paths:
        metrics_path = comparison_path.parent / "linear_projection_metrics.csv"
        if not metrics_path.exists():
            continue
        try:
            metrics = pd.read_csv(metrics_path)
        except Exception as exc:
            print(f"[collect] skipping linear baseline metrics {metrics_path}: {exc}", flush=True)
            continue
        baseline = metrics.loc[metrics["model_family"].astype(str).eq("LinearRegression")].copy()
        if baseline.empty:
            continue
        keep = [
            "data_source", "assessment_year", "fit_label", "split",
            "R2", "R2 (log)", "RMSE", "MAE", "MAPE", "MdAPE", "COD", "PRD", "PRB", "VEI",
            "MSE_log", "RMSE_log",
        ]
        baseline = baseline[[c for c in keep if c in baseline.columns]].rename(
            columns={c: f"baseline_{c}" for c in keep if c not in {"data_source", "assessment_year", "fit_label", "split"}}
        )
        baseline_rows.append(baseline)
    if baseline_rows:
        baseline_df = pd.concat(baseline_rows, ignore_index=True)
        key = [c for c in ["data_source", "assessment_year", "fit_label", "split"] if c in df.columns and c in baseline_df.columns]
        if key:
            df = df.merge(baseline_df, on=key, how="left")
    return df


def _parse_rho_sweep_task_name(path: Path) -> Dict[str, Any]:
    task_name = path.parent.name
    left, sep, right = task_name.partition("__")
    if not sep:
        return {}
    data_source, assess_sep, year_text = left.rpartition("_assess")
    if not assess_sep:
        return {}
    config_key = right.rsplit("_", 1)[0] if "_" in right else right
    try:
        assessment_year: Any = int(year_text)
    except ValueError:
        assessment_year = year_text
    return {"data_source": data_source, "assessment_year": assessment_year, "config_key": config_key}


def _infer_lgbm_rho_sweep_root(theory_root: Path) -> Optional[Path]:
    text = str(theory_root)
    if "theory_rho_range" not in text:
        return None
    candidate = Path(text.replace("theory_rho_range", "rho_sweep", 1))
    return candidate if candidate.exists() else None


def _load_lgbm_decile_metrics(rho_sweep_root: Optional[Path]) -> pd.DataFrame:
    if rho_sweep_root is None or not rho_sweep_root.exists():
        return pd.DataFrame()
    frames = []
    for split, pattern in [("assessment", "*/quick_test_metrics_assess.csv"), ("test", "*/quick_test_metrics_test.csv")]:
        for path in sorted(rho_sweep_root.glob(pattern)):
            try:
                metrics = pd.read_csv(path)
            except Exception as exc:
                print(f"[collect] skipping LGBM decile metrics {path}: {exc}", flush=True)
                continue
            meta = _parse_rho_sweep_task_name(path)
            if not meta:
                continue
            if "rho" not in metrics.columns:
                continue
            metrics = metrics[pd.to_numeric(metrics["rho"], errors="coerce").notna()].copy()
            if metrics.empty:
                continue
            metrics["split"] = split
            for col, val in meta.items():
                metrics[col] = val
            q10_cols = [col for col in metrics.columns if "_q10_" in col or col == "effective_bins_q10"]
            keep = _present(metrics, ["data_source", "assessment_year", "config_key", "split", "rho"] + q10_cols)
            frames.append(metrics[keep])
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["_rho_key"] = _numeric(out, "rho").round(10)
    return out.drop(columns=["rho"])


def _merge_lgbm_decile_metrics(df: pd.DataFrame, rho_sweep_root: Optional[Path]) -> pd.DataFrame:
    deciles = _load_lgbm_decile_metrics(rho_sweep_root)
    if df.empty or deciles.empty:
        return df
    out = df.copy()
    out["_rho_key"] = _numeric(out, "rho").round(10)
    key = _present(out, ["data_source", "assessment_year", "config_key", "split", "_rho_key"])
    if len(key) < 5:
        out = out.drop(columns=["_rho_key"], errors="ignore")
        return out
    decile_cols = [col for col in deciles.columns if col not in key]
    merged = out.merge(deciles[key + decile_cols], on=key, how="left", suffixes=("", "_rho_sweep"))
    for col in decile_cols:
        sweep_col = f"{col}_rho_sweep"
        if sweep_col in merged.columns:
            if col in merged.columns:
                merged[col] = merged[col].where(merged[col].notna(), merged[sweep_col])
            else:
                merged[col] = merged[sweep_col]
            merged = merged.drop(columns=[sweep_col])
    return merged.drop(columns=["_rho_key"], errors="ignore")


def _load_lgbm_baseline_accuracy_metrics(rho_sweep_root: Optional[Path]) -> pd.DataFrame:
    if rho_sweep_root is None or not rho_sweep_root.exists():
        return pd.DataFrame()
    frames = []
    for split, pattern in [("assessment", "*/quick_test_metrics_assess.csv"), ("test", "*/quick_test_metrics_test.csv")]:
        for path in sorted(rho_sweep_root.glob(pattern)):
            try:
                metrics = pd.read_csv(path)
            except Exception as exc:
                print(f"[collect] skipping LGBM baseline accuracy metrics {path}: {exc}", flush=True)
                continue
            meta = _parse_rho_sweep_task_name(path)
            if not meta:
                continue
            model_name = metrics["model_name"].astype(str) if "model_name" in metrics.columns else pd.Series("", index=metrics.index)
            model_family = metrics["model_family"].astype(str) if "model_family" in metrics.columns else pd.Series("", index=metrics.index)
            baseline = metrics[model_name.eq("LGBMRegressor") | model_family.eq("LGBMRegressor")].copy()
            if baseline.empty:
                continue
            if "rho" in baseline.columns:
                baseline_rho = pd.to_numeric(baseline["rho"], errors="coerce")
                if baseline_rho.isna().any():
                    baseline = baseline[baseline_rho.isna()].copy()
            if baseline.empty:
                continue
            baseline = baseline.head(1).copy()
            baseline["split"] = split
            for col, val in meta.items():
                baseline[col] = val
            accuracy_cols = _present(baseline, ["RMSE", "MAE", "MAPE", "MdAPE", "MSE_log", "RMSE_log"])
            keep = _present(baseline, ["data_source", "assessment_year", "config_key", "split"] + accuracy_cols)
            baseline = baseline[keep].rename(columns={col: f"baseline_{col}" for col in accuracy_cols})
            frames.append(baseline)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    key = _present(out, ["data_source", "assessment_year", "config_key", "split"])
    return out.drop_duplicates(key) if key else out


def _merge_lgbm_baseline_accuracy_metrics(df: pd.DataFrame, rho_sweep_root: Optional[Path]) -> pd.DataFrame:
    baseline = _load_lgbm_baseline_accuracy_metrics(rho_sweep_root)
    if df.empty or baseline.empty:
        return df
    key = _present(df, ["data_source", "assessment_year", "config_key", "split"])
    if len(key) < 4:
        return df
    baseline_cols = [col for col in baseline.columns if col not in key]
    merged = df.merge(baseline[key + baseline_cols], on=key, how="left", suffixes=("", "_rho_sweep_baseline"))
    for col in baseline_cols:
        sweep_col = f"{col}_rho_sweep_baseline"
        if sweep_col in merged.columns:
            if col in merged.columns:
                merged[col] = merged[col].where(merged[col].notna(), merged[sweep_col])
            else:
                merged[col] = merged[sweep_col]
            merged = merged.drop(columns=[sweep_col])
    return merged


def _load_lgbm(root: Path, rho_sweep_root: Optional[Path] = None) -> pd.DataFrame:
    merged = root / "merged" / "theory_empirical_comparison.csv"
    if merged.exists():
        df = _read_csvs([merged])
    else:
        df = _read_csvs(sorted(root.glob("*/theory_empirical_comparison.csv")))
    if df.empty:
        return df
    df["comparison_family"] = "lgbm_retrained_local_approx"
    if "q_theory" not in df.columns and "q_theory_remaining_covariance" in df.columns:
        df["q_theory"] = pd.to_numeric(df["q_theory_remaining_covariance"], errors="coerce")
    if "delta_MSE_log_theory" not in df.columns and "delta_mse_log_frac_theory" in df.columns and "B_mse_log" in df.columns:
        df["delta_MSE_log_theory"] = pd.to_numeric(df["delta_mse_log_frac_theory"], errors="coerce") * pd.to_numeric(df["B_mse_log"], errors="coerce")
    df = _merge_lgbm_decile_metrics(df, rho_sweep_root)
    df = _merge_lgbm_baseline_accuracy_metrics(df, rho_sweep_root)
    return df


def _numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _present(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [col for col in cols if col in df.columns]


def _short_family_label(family: Any) -> str:
    text = str(family)
    if text == "lgbm_retrained_local_approx":
        return "LGBM"
    if text == "linear_exact_projection":
        return "Linear"
    return text


def _style_for_family_split(family: Any, split: Any) -> Dict[str, Any]:
    family_text = str(family)
    split_text = str(split)
    return {
        "color": FAMILY_COLORS.get(family_text, "#6b7280"),
        "marker": SPLIT_MARKERS.get(split_text, "o"),
        "label": f"{_short_family_label(family_text)} | {split_text}",
    }


def _robust_limits(values: List[pd.Series], *, q_low: float = 0.02, q_high: float = 0.98, include_zero: bool = False) -> Optional[tuple[float, float, int]]:
    arrays = []
    for value in values:
        arr = pd.to_numeric(value, errors="coerce").to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            arrays.append(arr)
    if not arrays:
        return None
    all_values = np.concatenate(arrays)
    if all_values.size < 4:
        lo = float(np.nanmin(all_values))
        hi = float(np.nanmax(all_values))
    else:
        lo = float(np.nanquantile(all_values, q_low))
        hi = float(np.nanquantile(all_values, q_high))
    if include_zero:
        lo = min(lo, 0.0)
        hi = max(hi, 0.0)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if hi <= lo:
        span = max(abs(hi), abs(lo), 1.0)
        lo -= 0.05 * span
        hi += 0.05 * span
    pad = 0.06 * (hi - lo)
    lo -= pad
    hi += pad
    clipped = int(np.sum((all_values < lo) | (all_values > hi)))
    return lo, hi, clipped


def _full_limits(values: List[pd.Series], *, include_zero: bool = False) -> Optional[tuple[float, float]]:
    arrays = []
    for value in values:
        arr = pd.to_numeric(value, errors="coerce").to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            arrays.append(arr)
    if not arrays:
        return None
    all_values = np.concatenate(arrays)
    lo = float(np.nanmin(all_values))
    hi = float(np.nanmax(all_values))
    if include_zero:
        lo = min(lo, 0.0)
        hi = max(hi, 0.0)
    if hi <= lo:
        span = max(abs(hi), abs(lo), 1.0)
        lo -= 0.05 * span
        hi += 0.05 * span
    pad = 0.04 * (hi - lo)
    return lo - pad, hi + pad


def _finite_values(values: List[pd.Series]) -> np.ndarray:
    arrays = []
    for value in values:
        arr = pd.to_numeric(value, errors="coerce").to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            arrays.append(arr)
    return np.concatenate(arrays) if arrays else np.array([], dtype=float)


def _needs_symlog(values: List[pd.Series]) -> bool:
    arr = _finite_values(values)
    if arr.size < 12:
        return False
    abs_arr = np.abs(arr)
    abs_arr = abs_arr[abs_arr > 0.0]
    if abs_arr.size < 12:
        return False
    p50 = float(np.nanquantile(abs_arr, 0.50))
    p95 = float(np.nanquantile(abs_arr, 0.95))
    max_abs = float(np.nanmax(abs_arr))
    return bool(max_abs > max(8.0 * p95, 30.0 * p50))


def _symlog_linthresh(values: List[pd.Series]) -> float:
    abs_arr = np.abs(_finite_values(values))
    abs_arr = abs_arr[abs_arr > 0.0]
    if abs_arr.size == 0:
        return 1e-6
    p25 = float(np.nanquantile(abs_arr, 0.25))
    p50 = float(np.nanquantile(abs_arr, 0.50))
    p95 = float(np.nanquantile(abs_arr, 0.95))
    return max(min(p50 * 0.25, p95 * 0.05), p25 * 0.10, 1e-8)


def _add_error_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "delta_MSE_log" not in out.columns:
        out["delta_MSE_log"] = np.nan
    if "empirical_delta_MSE_log" in out.columns:
        empirical_delta = _numeric(out, "empirical_delta_MSE_log")
        out["delta_MSE_log"] = _numeric(out, "delta_MSE_log").where(
            _numeric(out, "delta_MSE_log").notna(),
            empirical_delta,
        )
    theory_frac = _numeric(out, "delta_MSE_log_frac_theory")
    if "delta_mse_log_frac_theory" in out.columns:
        theory_frac = theory_frac.where(theory_frac.notna(), _numeric(out, "delta_mse_log_frac_theory"))
    out["delta_MSE_log_frac_theory_unified"] = theory_frac
    empirical_frac = _numeric(out, "delta_MSE_log_frac")
    if "empirical_MSE_log_frac_delta" in out.columns:
        empirical_frac = empirical_frac.where(empirical_frac.notna(), _numeric(out, "empirical_MSE_log_frac_delta"))
    out["delta_MSE_log_frac_empirical"] = empirical_frac
    if "q_empirical_signed" in out.columns and "q_theory" in out.columns:
        out["q_error_empirical_minus_theory"] = _numeric(out, "q_empirical_signed") - _numeric(out, "q_theory")
    if "delta_MSE_log" in out.columns and "delta_MSE_log_theory" in out.columns:
        out["delta_MSE_log_error"] = _numeric(out, "delta_MSE_log") - _numeric(out, "delta_MSE_log_theory")
    elif "empirical_delta_MSE_log" in out.columns and "delta_MSE_log_theory" in out.columns:
        out["delta_MSE_log_error"] = _numeric(out, "empirical_delta_MSE_log") - _numeric(out, "delta_MSE_log_theory")
    if "Slope_log_resid_logprice" in out.columns and "Slope_log_resid_logprice_theory" in out.columns:
        out["slope_error_empirical_minus_theory"] = (
            _numeric(out, "Slope_log_resid_logprice") - _numeric(out, "Slope_log_resid_logprice_theory")
        )
    for metric in ["R2", "RMSE", "MAE", "MAPE", "MdAPE", "COD", "PRD", "PRB", "VEI", "MSE_log"]:
        if metric in out.columns:
            out[metric] = _numeric(out, metric)
        bcol = f"baseline_{metric}"
        if bcol in out.columns:
            out[bcol] = _numeric(out, bcol)
    if {"R2", "baseline_R2"}.issubset(out.columns):
        out["empirical_R2_delta"] = _numeric(out, "R2") - _numeric(out, "baseline_R2")
    if {"RMSE", "baseline_RMSE"}.issubset(out.columns):
        out["empirical_RMSE_frac_delta"] = _numeric(out, "RMSE") / _numeric(out, "baseline_RMSE") - 1.0
    if {"MAE", "baseline_MAE"}.issubset(out.columns):
        out["empirical_MAE_frac_delta"] = _numeric(out, "MAE") / _numeric(out, "baseline_MAE") - 1.0
    if {"MAPE", "baseline_MAPE"}.issubset(out.columns):
        out["empirical_MAPE_frac_delta"] = _numeric(out, "MAPE") / _numeric(out, "baseline_MAPE") - 1.0
    if {"MdAPE", "baseline_MdAPE"}.issubset(out.columns):
        out["empirical_MdAPE_frac_delta"] = _numeric(out, "MdAPE") / _numeric(out, "baseline_MdAPE") - 1.0
    if {"COD", "baseline_COD"}.issubset(out.columns):
        out["empirical_COD_frac_delta"] = _numeric(out, "COD") / _numeric(out, "baseline_COD") - 1.0
    if {"MSE_log", "baseline_MSE_log"}.issubset(out.columns):
        out["empirical_MSE_log_frac_delta"] = _numeric(out, "MSE_log") / _numeric(out, "baseline_MSE_log") - 1.0

    def abs_target_ratio(values: pd.Series, baseline: pd.Series, target: float) -> pd.Series:
        num = (pd.to_numeric(values, errors="coerce") - target).abs()
        den = (pd.to_numeric(baseline, errors="coerce") - target).abs()
        return num / den.replace(0.0, np.nan)

    if {"PRD", "baseline_PRD"}.issubset(out.columns):
        out["empirical_abs_PRD_minus_1_ratio"] = abs_target_ratio(out["PRD"], out["baseline_PRD"], 1.0)
        out["empirical_PRD_error_reduction"] = 1.0 - out["empirical_abs_PRD_minus_1_ratio"]
    if {"PRB", "baseline_PRB"}.issubset(out.columns):
        out["empirical_abs_PRB_ratio"] = abs_target_ratio(out["PRB"], out["baseline_PRB"], 0.0)
        out["empirical_PRB_error_reduction"] = 1.0 - out["empirical_abs_PRB_ratio"]
    if {"VEI", "baseline_VEI"}.issubset(out.columns):
        out["empirical_abs_VEI_ratio"] = abs_target_ratio(out["VEI"], out["baseline_VEI"], 0.0)
        out["empirical_VEI_error_reduction"] = 1.0 - out["empirical_abs_VEI_ratio"]
    return out


def _rmse(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x * x))) if x.size else np.nan


def _mae(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(np.abs(x))) if x.size else np.nan


def _median_abs(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    return float(np.median(np.abs(x))) if x.size else np.nan


def _corr(df: pd.DataFrame, xcol: str, ycol: str) -> float:
    if xcol not in df.columns or ycol not in df.columns:
        return np.nan
    d = df[[xcol, ycol]].apply(pd.to_numeric, errors="coerce").dropna()
    if d.shape[0] < 2:
        return np.nan
    return float(d.corr().iloc[0, 1])


def _summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if df.empty:
        return pd.DataFrame()
    group_cols = ["comparison_family", "split"]
    for key, g in df.groupby(group_cols, dropna=False):
        comparison_family, split = key
        rows.append({
            "comparison_family": comparison_family,
            "split": split,
            "n_rows": int(g.shape[0]),
            "q_error_mae": _mae(g.get("q_error_empirical_minus_theory", pd.Series(dtype=float))),
            "q_error_rmse": _rmse(g.get("q_error_empirical_minus_theory", pd.Series(dtype=float))),
            "q_corr": _corr(g, "q_theory", "q_empirical_signed"),
            "delta_MSE_log_error_mae": _mae(g.get("delta_MSE_log_error", pd.Series(dtype=float))),
            "delta_MSE_log_error_rmse": _rmse(g.get("delta_MSE_log_error", pd.Series(dtype=float))),
            "delta_MSE_log_corr": _corr(g, "delta_MSE_log_theory", "delta_MSE_log")
            if "delta_MSE_log" in g.columns else _corr(g, "delta_MSE_log_theory", "empirical_delta_MSE_log"),
            "slope_error_mae": _mae(g.get("slope_error_empirical_minus_theory", pd.Series(dtype=float))),
            "slope_corr": _corr(g, "Slope_log_resid_logprice_theory", "Slope_log_resid_logprice"),
            "median_taylor1_cov_ratio_price_rel_error": _median_abs(g.get("C_ratio_price_taylor1_rel_error", pd.Series(dtype=float))),
            "median_taylor2_cov_ratio_price_rel_error": _median_abs(g.get("C_ratio_price_taylor2_rel_error", pd.Series(dtype=float))),
        })
    return pd.DataFrame(rows)


def _diagnostic_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for (family, split), g in df.groupby(["comparison_family", "split"], dropna=False):
        row: Dict[str, Any] = {
            "comparison_family": family,
            "split": split,
            "n_rows": int(g.shape[0]),
            "q_corr": _corr(g, "q_theory", "q_empirical_signed"),
            "mse_corr": _corr(g, "delta_MSE_log_theory", "delta_MSE_log"),
            "slope_corr": _corr(g, "Slope_log_resid_logprice_theory", "Slope_log_resid_logprice"),
        }
        for label, col in [
            ("q_error", "q_error_empirical_minus_theory"),
            ("delta_MSE_log_error", "delta_MSE_log_error"),
            ("slope_error", "slope_error_empirical_minus_theory"),
            ("taylor1_cov_ratio_price_rel_error", "C_ratio_price_taylor1_rel_error"),
            ("taylor2_cov_ratio_price_rel_error", "C_ratio_price_taylor2_rel_error"),
            ("PRD_error_reduction", "empirical_PRD_error_reduction"),
            ("PRB_error_reduction", "empirical_PRB_error_reduction"),
            ("VEI_error_reduction", "empirical_VEI_error_reduction"),
        ]:
            values = _numeric(g, col).replace([np.inf, -np.inf], np.nan).dropna()
            row[f"{label}_median"] = float(values.median()) if not values.empty else np.nan
            row[f"{label}_mean"] = float(values.mean()) if not values.empty else np.nan
            row[f"{label}_mae"] = float(values.abs().mean()) if not values.empty else np.nan
            row[f"{label}_p90_abs"] = float(values.abs().quantile(0.9)) if not values.empty else np.nan
        if {"C_ratio_price_taylor1_rel_error", "C_ratio_price_taylor2_rel_error"}.issubset(g.columns):
            t1 = _numeric(g, "C_ratio_price_taylor1_rel_error").abs()
            t2 = _numeric(g, "C_ratio_price_taylor2_rel_error").abs()
            finite = t1.notna() & t2.notna()
            row["taylor2_better_share"] = float((t2[finite] < t1[finite]).mean()) if finite.any() else np.nan
        if "q_empirical_signed" in g.columns:
            q = _numeric(g, "q_empirical_signed")
            row["q_negative_share"] = float((q < 0.0).mean())
            row["q_greater_than_one_share"] = float((q > 1.0).mean())
            row["q_abs_greater_than_one_share"] = float((q.abs() > 1.0).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _save_scatter(
    df: pd.DataFrame,
    *,
    xcol: str,
    ycol: str,
    xlabel: str,
    ylabel: str,
    title: str,
    path: Path,
    focus_axes: bool = False,
    identity: bool = True,
) -> Optional[Path]:
    if df.empty or xcol not in df.columns or ycol not in df.columns:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    plotted = False
    group_cols = _present(df, ["comparison_family", "split"])
    for key, g in df.groupby(group_cols, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        key_map = {col: val for col, val in zip(group_cols, key_tuple)}
        style = _style_for_family_split(key_map.get("comparison_family", "model"), key_map.get("split", "split"))
        x = _numeric(g, xcol)
        y = _numeric(g, ycol)
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.any():
            plotted = True
            ax.scatter(
                x[finite],
                y[finite],
                s=28,
                alpha=0.78,
                color=style["color"],
                marker=style["marker"],
                edgecolor="white",
                linewidth=0.35,
                label=style["label"],
            )
    if not plotted:
        plt.close(fig)
        return None
    all_xy = df[[xcol, ycol]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if not all_xy.empty:
        if focus_axes:
            lims = _robust_limits([all_xy[xcol], all_xy[ycol]], include_zero=True)
            if lims is not None:
                lo, hi, clipped = lims
                ax.set_xlim(lo, hi)
                ax.set_ylim(lo, hi)
                if clipped:
                    ax.text(
                        0.02,
                        0.98,
                        f"focus axis: robust central scale; {clipped} values outside view",
                        transform=ax.transAxes,
                        va="top",
                        ha="left",
                        fontsize=8,
                        bbox=dict(facecolor="white", alpha=0.78, edgecolor="none"),
                    )
            else:
                lo = float(all_xy.min().min())
                hi = float(all_xy.max().max())
        else:
            lims_full = _full_limits([all_xy[xcol], all_xy[ycol]], include_zero=True)
            lo, hi = lims_full if lims_full is not None else (float(all_xy.min().min()), float(all_xy.max().max()))
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
        if identity and np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            ax.plot([lo, hi], [lo, hi], color="#111827", ls="--", lw=1.0, alpha=0.75)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, loc="upper right")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_faceted_scatter(
    df: pd.DataFrame,
    *,
    xcol: str,
    ycol: str,
    xlabel: str,
    ylabel: str,
    title: str,
    path: Path,
    focus_axes: bool = False,
) -> Optional[Path]:
    if df.empty or xcol not in df.columns or ycol not in df.columns:
        return None
    groups = []
    for key, g in df.groupby(["comparison_family", "split"], dropna=False):
        x = _numeric(g, xcol)
        y = _numeric(g, ycol)
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.any():
            groups.append((key, g.loc[finite].copy(), x[finite], y[finite]))
    if not groups:
        return None

    ncols = 2
    nrows = int(np.ceil(len(groups) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2 * ncols, 5.2 * nrows), squeeze=False)
    cmap = plt.get_cmap("viridis")
    marker_handles: Dict[str, Line2D] = {}
    sc = None
    for ax in axes.ravel():
        ax.set_axis_off()
    for ax, ((family, split), g, x, y) in zip(axes.ravel(), groups):
        ax.set_axis_on()
        color_values = _numeric(g, "rho") if "rho" in g.columns else pd.Series(np.nan, index=g.index)
        if color_values.notna().any():
            marker_col = "config_key" if "config_key" in g.columns and g["config_key"].notna().any() else None
            if marker_col:
                for marker_value, mg in g.groupby(marker_col, dropna=False):
                    idx = mg.index
                    marker = CONFIG_MARKERS.get(str(marker_value), "o")
                    sc = ax.scatter(
                        x.loc[idx],
                        y.loc[idx],
                        c=color_values.loc[idx],
                        cmap=cmap,
                        s=22,
                        alpha=0.82,
                        marker=marker,
                        edgecolor="white",
                        linewidth=0.25,
                    )
                    label = str(marker_value)
                    if label not in marker_handles:
                        marker_handles[label] = Line2D(
                            [0], [0], marker=marker, linestyle="None", color="#111827",
                            markerfacecolor="#9ca3af", markeredgecolor="white", markersize=6, label=label
                        )
            else:
                sc = ax.scatter(x, y, c=color_values, cmap=cmap, s=20, alpha=0.8)
        else:
            sc = ax.scatter(x, y, s=20, alpha=0.8)
        if focus_axes:
            lims = _robust_limits([x, y], include_zero=True)
            if lims is not None:
                lo, hi, clipped = lims
                if clipped:
                    ax.text(
                        0.02,
                        0.98,
                        f"focus: {clipped} outside",
                        transform=ax.transAxes,
                        va="top",
                        ha="left",
                        fontsize=7,
                        bbox=dict(facecolor="white", alpha=0.72, edgecolor="none"),
                    )
            else:
                lo = float(np.nanmin([x.min(), y.min()]))
                hi = float(np.nanmax([x.max(), y.max()]))
        else:
            lims_full = _full_limits([x, y], include_zero=True)
            lo, hi = lims_full if lims_full is not None else (
                float(np.nanmin([x.min(), y.min()])),
                float(np.nanmax([x.max(), y.max()])),
            )
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            ax.plot([lo, hi], [lo, hi], color="#111827", ls="--", lw=1.0, alpha=0.75)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
        corr = np.corrcoef(x, y)[0, 1] if len(x) > 1 else np.nan
        ax.set_title(f"{family}\n{split} | n={len(x)} | corr={corr:.3g}", fontsize=10, pad=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    if any("sc" in locals() for _ in [0]) and "rho" in df.columns:
        try:
            cb = fig.colorbar(sc, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
            cb.set_label("rho")
        except Exception:
            pass
    if marker_handles:
        fig.legend(
            list(marker_handles.values()),
            list(marker_handles.keys()),
            loc="lower center",
            ncol=min(3, len(marker_handles)),
            frameon=False,
            fontsize=8,
            bbox_to_anchor=(0.5, 0.005),
        )
    fig.suptitle(title, fontweight="bold", y=0.995)
    fig.subplots_adjust(hspace=0.55, wspace=0.28, top=0.94, bottom=0.07 if marker_handles else 0.05)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _direct_theory_metric_specs() -> List[Dict[str, Any]]:
    return [
        {
            "xcol": "q_theory",
            "ycol": "q_empirical_signed",
            "title": "Remaining covariance fraction q",
            "xlabel": "theory q",
            "ylabel": "empirical q = C/C0",
            "identity": True,
            "equal_axes": True,
        },
        {
            "xcol": "covariance_reduction_theory",
            "ycol": "covariance_reduction_empirical",
            "title": "Covariance reduction",
            "xlabel": "theory reduction",
            "ylabel": "empirical reduction",
            "identity": True,
            "equal_axes": True,
        },
        {
            "xcol": "C_log_resid_logprice_theory",
            "ycol": "C_log_resid_logprice",
            "title": "Log covariance C(f)",
            "xlabel": "theory C(f)",
            "ylabel": "empirical C(f)",
            "identity": True,
            "equal_axes": True,
        },
        {
            "xcol": "Slope_log_resid_logprice_theory",
            "ycol": "Slope_log_resid_logprice",
            "title": "Log-residual slope",
            "xlabel": "theory slope",
            "ylabel": "empirical slope",
            "identity": True,
            "equal_axes": True,
        },
        {
            "xcol": "PRB_proxy_ratio_logprice_theory",
            "ycol": "PRB_proxy_ratio_logprice",
            "title": "PRB log-price proxy",
            "xlabel": "theory proxy",
            "ylabel": "empirical proxy",
            "identity": True,
            "equal_axes": True,
        },
        {
            "xcol": "VEI_log_proxy_fixed_deciles_theory",
            "ycol": "VEI_log_proxy_fixed_deciles",
            "title": "VEI log proxy",
            "xlabel": "theory proxy",
            "ylabel": "empirical proxy",
            "identity": True,
            "equal_axes": True,
        },
        {
            "xcol": "delta_MSE_log_theory",
            "ycol": "delta_MSE_log",
            "title": "Second-order log-MSE cost",
            "xlabel": "theory delta MSE_log",
            "ylabel": "empirical delta MSE_log",
            "identity": True,
            "equal_axes": True,
        },
        {
            "xcol": "delta_MSE_log_frac_theory_unified",
            "ycol": "delta_MSE_log_frac_empirical",
            "title": "Fractional log-MSE cost",
            "xlabel": "theory frac delta MSE_log",
            "ylabel": "empirical frac delta MSE_log",
            "identity": True,
            "equal_axes": True,
        },
    ]


def _metric_driver_specs() -> List[Dict[str, Any]]:
    return [
        {
            "xcol": "covariance_reduction_theory",
            "ycol": "empirical_PRD_error_reduction",
            "title": "PRD error reduction",
            "xlabel": "theory covariance reduction",
            "ylabel": "empirical PRD error reduction",
            "identity": True,
            "equal_axes": True,
        },
        {
            "xcol": "covariance_reduction_theory",
            "ycol": "empirical_PRB_error_reduction",
            "title": "PRB error reduction",
            "xlabel": "theory covariance reduction",
            "ylabel": "empirical PRB error reduction",
            "identity": True,
            "equal_axes": True,
        },
        {
            "xcol": "covariance_reduction_theory",
            "ycol": "empirical_VEI_error_reduction",
            "title": "VEI error reduction",
            "xlabel": "theory covariance reduction",
            "ylabel": "empirical VEI error reduction",
            "identity": True,
            "equal_axes": True,
        },
        {
            "xcol": "delta_MSE_log_frac_theory_unified",
            "ycol": "delta_MSE_log_frac_empirical",
            "title": "Log-MSE fractional movement",
            "xlabel": "theory log-MSE cost / baseline",
            "ylabel": "empirical log-MSE fractional change",
            "identity": True,
            "equal_axes": True,
        },
        {
            "xcol": "delta_MSE_log_frac_theory_unified",
            "ycol": "empirical_RMSE_frac_delta",
            "title": "RMSE movement",
            "xlabel": "theory log-MSE cost / baseline",
            "ylabel": "empirical RMSE fractional change",
            "identity": False,
            "equal_axes": False,
        },
        {
            "xcol": "delta_MSE_log_frac_theory_unified",
            "ycol": "empirical_MAPE_frac_delta",
            "title": "MAPE movement",
            "xlabel": "theory log-MSE cost / baseline",
            "ylabel": "empirical MAPE fractional change",
            "identity": False,
            "equal_axes": False,
        },
        {
            "xcol": "delta_MSE_log_frac_theory_unified",
            "ycol": "empirical_MdAPE_frac_delta",
            "title": "MdAPE movement",
            "xlabel": "theory log-MSE cost / baseline",
            "ylabel": "empirical MdAPE fractional change",
            "identity": False,
            "equal_axes": False,
        },
        {
            "xcol": "covariance_reduction_theory",
            "ycol": "empirical_COD_frac_delta",
            "title": "COD movement",
            "xlabel": "theory covariance reduction",
            "ylabel": "empirical COD fractional change",
            "identity": False,
            "equal_axes": False,
        },
        {
            "xcol": "delta_MSE_log_frac_theory_unified",
            "ycol": "empirical_R2_delta",
            "title": "R2 movement",
            "xlabel": "theory log-MSE cost / baseline",
            "ylabel": "empirical R2 change",
            "identity": False,
            "equal_axes": False,
        },
    ]


def _save_metric_matrix(
    df: pd.DataFrame,
    *,
    specs: List[Dict[str, Any]],
    title: str,
    path: Path,
    focus_axes: bool,
) -> Optional[Path]:
    from matplotlib.ticker import MaxNLocator

    available = [
        spec for spec in specs
        if spec["xcol"] in df.columns
        and spec["ycol"] in df.columns
        and _numeric(df, spec["xcol"]).notna().any()
        and _numeric(df, spec["ycol"]).notna().any()
    ]
    if not available:
        return None

    ncols = 4
    nrows = int(np.ceil(len(available) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.2 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.set_axis_off()

    legend_handles: Dict[str, Line2D] = {}
    for ax, spec in zip(axes.ravel(), available):
        ax.set_axis_on()
        x_all = _numeric(df, spec["xcol"])
        y_all = _numeric(df, spec["ycol"])
        finite_all = np.isfinite(x_all) & np.isfinite(y_all)
        if not finite_all.any():
            ax.set_axis_off()
            continue
        x_symlog = False
        y_symlog = False

        group_cols = _present(df, ["comparison_family", "split"])
        for key, g in df.loc[finite_all].groupby(group_cols, dropna=False):
            key_tuple = key if isinstance(key, tuple) else (key,)
            key_map = {col: val for col, val in zip(group_cols, key_tuple)}
            style = _style_for_family_split(key_map.get("comparison_family", "model"), key_map.get("split", "split"))
            x = _numeric(g, spec["xcol"])
            y = _numeric(g, spec["ycol"])
            finite = np.isfinite(x) & np.isfinite(y)
            if finite.any():
                ax.scatter(
                    x[finite],
                    y[finite],
                    s=23,
                    alpha=0.72,
                    color=style["color"],
                    marker=style["marker"],
                    edgecolor="white",
                    linewidth=0.3,
                )
                if style["label"] not in legend_handles:
                    legend_handles[style["label"]] = Line2D(
                        [0], [0], marker=style["marker"], linestyle="None",
                        markerfacecolor=style["color"], markeredgecolor="white",
                        markersize=7, label=style["label"],
                    )

        if bool(spec.get("equal_axes", False)):
            if focus_axes:
                lims = _robust_limits([x_all[finite_all], y_all[finite_all]], include_zero=True)
                if lims is not None:
                    lo, hi, clipped = lims
                else:
                    limits = _full_limits([x_all[finite_all], y_all[finite_all]], include_zero=True)
                    lo, hi = limits if limits is not None else (0.0, 1.0)
                    clipped = 0
            else:
                limits = _full_limits([x_all[finite_all], y_all[finite_all]], include_zero=True)
                lo, hi = limits if limits is not None else (0.0, 1.0)
                clipped = 0
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            if bool(spec.get("identity", False)):
                ax.plot([lo, hi], [lo, hi], color="#111827", ls="--", lw=1.0, alpha=0.75)
            if not focus_axes and _needs_symlog([x_all[finite_all], y_all[finite_all]]):
                linthresh = _symlog_linthresh([x_all[finite_all], y_all[finite_all]])
                ax.set_xscale("symlog", linthresh=linthresh)
                ax.set_yscale("symlog", linthresh=linthresh)
                x_symlog = True
                y_symlog = True
        else:
            if focus_axes:
                x_lims = _robust_limits([x_all[finite_all]], include_zero=True)
                y_lims = _robust_limits([y_all[finite_all]], include_zero=True)
                clipped = (x_lims[2] if x_lims is not None else 0) + (y_lims[2] if y_lims is not None else 0)
                if x_lims is not None:
                    ax.set_xlim(x_lims[0], x_lims[1])
                if y_lims is not None:
                    ax.set_ylim(y_lims[0], y_lims[1])
            else:
                clipped = 0
                x_lims_full = _full_limits([x_all[finite_all]], include_zero=True)
                y_lims_full = _full_limits([y_all[finite_all]], include_zero=True)
                if x_lims_full is not None:
                    ax.set_xlim(x_lims_full)
                if y_lims_full is not None:
                    ax.set_ylim(y_lims_full)
                if _needs_symlog([x_all[finite_all]]):
                    ax.set_xscale("symlog", linthresh=_symlog_linthresh([x_all[finite_all]]))
                    x_symlog = True
                if _needs_symlog([y_all[finite_all]]):
                    ax.set_yscale("symlog", linthresh=_symlog_linthresh([y_all[finite_all]]))
                    y_symlog = True
        ax.axhline(0.0, color="#9ca3af", lw=0.8, alpha=0.55)
        ax.axvline(0.0, color="#9ca3af", lw=0.8, alpha=0.55)
        if focus_axes and clipped:
            ax.text(
                0.02,
                0.98,
                f"focus: {clipped} values outside",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=7,
                bbox=dict(facecolor="white", alpha=0.74, edgecolor="none"),
            )
        elif (x_symlog or y_symlog):
            axis_text = "symlog: "
            axis_text += "x/y" if x_symlog and y_symlog else ("x" if x_symlog else "y")
            ax.text(
                0.02,
                0.98,
                axis_text,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=7,
                bbox=dict(facecolor="white", alpha=0.74, edgecolor="none"),
            )
        corr = _corr(df.loc[finite_all], spec["xcol"], spec["ycol"])
        ax.set_title(f"{spec['title']}\nn={int(finite_all.sum())}, corr={corr:.3g}", fontsize=9)
        ax.set_xlabel(spec["xlabel"], fontsize=8)
        ax.set_ylabel(spec["ylabel"], fontsize=8)
        if not x_symlog:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        if not y_symlog:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.tick_params(axis="both", labelsize=7.5)
        ax.grid(alpha=0.25)

    if legend_handles:
        fig.legend(
            list(legend_handles.values()),
            list(legend_handles.keys()),
            loc="lower center",
            ncol=min(4, len(legend_handles)),
            frameon=False,
            fontsize=8,
        )
    suffix = "focus axes: robust central scale; outliers remain in CSVs/tables" if focus_axes else "full axes"
    fig.suptitle(f"{title}\n{suffix}", fontweight="bold", y=0.995)
    fig.subplots_adjust(hspace=0.52, wspace=0.32, bottom=0.12 if legend_handles else 0.06, top=0.88)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _lgbm_evolution_specs() -> List[Dict[str, Any]]:
    return [
        {
            "filename": "q",
            "empirical_col": "q_empirical_signed",
            "theory_col": "q_theory",
            "empirical_label": "empirical q",
            "theory_label": "theory q",
            "title": "Remaining covariance fraction q",
            "same_axis": True,
            "focus_y": False,
        },
        {
            "filename": "covariance",
            "empirical_col": "C_log_resid_logprice",
            "theory_col": "C_log_resid_logprice_theory",
            "empirical_label": "empirical C(f)",
            "theory_label": "theory C(f)",
            "title": "Log covariance C(f)",
            "same_axis": True,
            "focus_y": False,
        },
        {
            "filename": "covariance_reduction",
            "empirical_col": "covariance_reduction_empirical",
            "theory_col": "covariance_reduction_theory",
            "empirical_label": "empirical covariance reduction",
            "theory_label": "theory covariance reduction",
            "title": "Covariance reduction",
            "same_axis": True,
            "focus_y": True,
        },
        {
            "filename": "slope",
            "empirical_col": "Slope_log_resid_logprice",
            "theory_col": "Slope_log_resid_logprice_theory",
            "empirical_label": "empirical slope",
            "theory_label": "theory slope",
            "title": "Log-residual slope",
            "same_axis": True,
            "focus_y": False,
        },
        {
            "filename": "prb_proxy",
            "empirical_col": "PRB_proxy_ratio_logprice",
            "theory_col": "PRB_proxy_ratio_logprice_theory",
            "empirical_label": "empirical PRB proxy",
            "theory_label": "theory PRB proxy",
            "title": "PRB log-price proxy",
            "same_axis": True,
            "focus_y": False,
        },
        {
            "filename": "vei_proxy",
            "empirical_col": "VEI_log_proxy_fixed_deciles",
            "theory_col": "VEI_log_proxy_fixed_deciles_theory",
            "empirical_label": "empirical VEI proxy",
            "theory_label": "theory VEI proxy",
            "title": "VEI log proxy",
            "same_axis": True,
            "focus_y": True,
        },
        {
            "filename": "delta_mse_log",
            "empirical_col": "delta_MSE_log",
            "theory_col": "delta_MSE_log_theory",
            "empirical_label": "empirical delta MSE_log",
            "theory_label": "theory delta MSE_log",
            "title": "Second-order log-MSE cost",
            "same_axis": True,
            "focus_y": True,
        },
        {
            "filename": "delta_mse_log_frac",
            "empirical_col": "delta_MSE_log_frac_empirical",
            "theory_col": "delta_MSE_log_frac_theory_unified",
            "empirical_label": "empirical fractional log-MSE movement",
            "theory_label": "theory fractional log-MSE cost",
            "title": "Fractional log-MSE cost",
            "same_axis": True,
            "focus_y": True,
        },
        {
            "filename": "prd_error_reduction",
            "empirical_col": "empirical_PRD_error_reduction",
            "theory_col": "covariance_reduction_theory",
            "empirical_label": "empirical PRD error reduction",
            "theory_label": "theory covariance reduction",
            "title": "PRD bridge",
            "same_axis": True,
            "focus_y": True,
        },
        {
            "filename": "prb_error_reduction",
            "empirical_col": "empirical_PRB_error_reduction",
            "theory_col": "covariance_reduction_theory",
            "empirical_label": "empirical PRB error reduction",
            "theory_label": "theory covariance reduction",
            "title": "PRB bridge",
            "same_axis": True,
            "focus_y": True,
        },
        {
            "filename": "vei_error_reduction",
            "empirical_col": "empirical_VEI_error_reduction",
            "theory_col": "covariance_reduction_theory",
            "empirical_label": "empirical VEI error reduction",
            "theory_label": "theory covariance reduction",
            "title": "VEI bridge",
            "same_axis": True,
            "focus_y": True,
        },
        {
            "filename": "r2_delta",
            "empirical_col": "empirical_R2_delta",
            "theory_col": "delta_MSE_log_frac_theory_unified",
            "empirical_label": "empirical R2 change",
            "theory_label": "theory log-MSE cost / baseline",
            "title": "R2 collateral movement",
            "same_axis": False,
            "focus_y": True,
        },
        {
            "filename": "rmse_frac_delta",
            "empirical_col": "empirical_RMSE_frac_delta",
            "theory_col": "delta_MSE_log_frac_theory_unified",
            "empirical_label": "empirical RMSE fractional change",
            "theory_label": "theory log-MSE cost / baseline",
            "title": "RMSE collateral movement",
            "same_axis": False,
            "focus_y": True,
        },
        {
            "filename": "mape_frac_delta",
            "empirical_col": "empirical_MAPE_frac_delta",
            "theory_col": "delta_MSE_log_frac_theory_unified",
            "empirical_label": "empirical MAPE fractional change",
            "theory_label": "theory log-MSE cost / baseline",
            "title": "MAPE collateral movement",
            "same_axis": False,
            "focus_y": True,
        },
        {
            "filename": "mdape_frac_delta",
            "empirical_col": "empirical_MdAPE_frac_delta",
            "theory_col": "delta_MSE_log_frac_theory_unified",
            "empirical_label": "empirical MdAPE fractional change",
            "theory_label": "theory log-MSE cost / baseline",
            "title": "MdAPE collateral movement",
            "same_axis": False,
            "focus_y": True,
        },
        {
            "filename": "cod_frac_delta",
            "empirical_col": "empirical_COD_frac_delta",
            "theory_col": "covariance_reduction_theory",
            "empirical_label": "empirical COD fractional change",
            "theory_label": "theory covariance reduction",
            "title": "COD collateral movement",
            "same_axis": False,
            "focus_y": True,
        },
    ]


def _save_lgbm_rho_evolution_plot(
    df: pd.DataFrame,
    *,
    split: str,
    spec: Dict[str, Any],
    path: Path,
) -> Optional[Path]:
    from matplotlib.ticker import MaxNLocator

    required = {"comparison_family", "split", "data_source", "config_key", "rho", spec["empirical_col"], spec["theory_col"]}
    if df.empty or not required.issubset(df.columns):
        return None
    d = df[
        df["comparison_family"].astype(str).eq("lgbm_retrained_local_approx")
        & df["split"].astype(str).eq(split)
    ].copy()
    if d.empty:
        return None
    d["_rho"] = _numeric(d, "rho")
    d["_empirical"] = _numeric(d, spec["empirical_col"])
    d["_theory"] = _numeric(d, spec["theory_col"])
    d = d[np.isfinite(d["_rho"])].copy()
    if d.empty or (not d["_empirical"].notna().any() and not d["_theory"].notna().any()):
        return None

    data_sources = sorted(d["data_source"].dropna().astype(str).unique())
    configs = sorted(d["config_key"].dropna().astype(str).unique())
    if not data_sources or not configs:
        return None

    same_axis = bool(spec.get("same_axis", True))
    focus_y = bool(spec.get("focus_y", False))
    clipped_note = 0
    if same_axis:
        y_lims = _robust_limits([d["_empirical"], d["_theory"]], include_zero=True) if focus_y else _full_limits([d["_empirical"], d["_theory"]], include_zero=True)
        if focus_y and y_lims is not None:
            clipped_note = y_lims[2]
            y_lims = (y_lims[0], y_lims[1])
    else:
        emp_lims = _robust_limits([d["_empirical"]], include_zero=True) if focus_y else _full_limits([d["_empirical"]], include_zero=True)
        theory_lims = _robust_limits([d["_theory"]], include_zero=True) if focus_y else _full_limits([d["_theory"]], include_zero=True)
        if focus_y:
            clipped_note = (emp_lims[2] if emp_lims is not None else 0) + (theory_lims[2] if theory_lims is not None else 0)
            emp_lims = (emp_lims[0], emp_lims[1]) if emp_lims is not None else None
            theory_lims = (theory_lims[0], theory_lims[1]) if theory_lims is not None else None

    fig, axes = plt.subplots(
        len(data_sources),
        len(configs),
        figsize=(4.5 * len(configs), 3.05 * len(data_sources)),
        sharex=True,
        squeeze=False,
    )
    legend_handles: Dict[str, Any] = {}
    plotted = False
    for i, data_source in enumerate(data_sources):
        for j, config in enumerate(configs):
            ax = axes[i, j]
            g = d[d["data_source"].astype(str).eq(data_source) & d["config_key"].astype(str).eq(config)].sort_values("_rho")
            if g.empty:
                ax.set_axis_off()
                continue
            emp_mask = np.isfinite(g["_rho"]) & np.isfinite(g["_empirical"])
            theory_mask = np.isfinite(g["_rho"]) & np.isfinite(g["_theory"])
            if emp_mask.any():
                (emp_line,) = ax.plot(
                    g.loc[emp_mask, "_rho"],
                    g.loc[emp_mask, "_empirical"],
                    color="#2563eb",
                    lw=1.55,
                    marker="o",
                    ms=2.5,
                    label=spec["empirical_label"],
                )
                legend_handles[spec["empirical_label"]] = emp_line
                plotted = True
            ax.axhline(0.0, color="#9ca3af", lw=0.8, alpha=0.65)
            if same_axis:
                if theory_mask.any():
                    (theory_line,) = ax.plot(
                        g.loc[theory_mask, "_rho"],
                        g.loc[theory_mask, "_theory"],
                        color="#111827",
                        lw=1.45,
                        ls="--",
                        label=spec["theory_label"],
                    )
                    legend_handles[spec["theory_label"]] = theory_line
                    plotted = True
                if y_lims is not None:
                    ax.set_ylim(y_lims)
                ax.set_ylabel(spec["empirical_label"] if j == 0 else "", fontsize=8)
            else:
                ax2 = ax.twinx()
                if theory_mask.any():
                    (theory_line,) = ax2.plot(
                        g.loc[theory_mask, "_rho"],
                        g.loc[theory_mask, "_theory"],
                        color="#dc2626",
                        lw=1.4,
                        ls="--",
                        label=spec["theory_label"],
                    )
                    legend_handles[spec["theory_label"]] = theory_line
                    plotted = True
                if emp_lims is not None:
                    ax.set_ylim(emp_lims)
                if theory_lims is not None:
                    ax2.set_ylim(theory_lims)
                if j == 0:
                    ax.set_ylabel(spec["empirical_label"], color="#1d4ed8", fontsize=8)
                if j == len(configs) - 1:
                    ax2.set_ylabel(spec["theory_label"], color="#b91c1c", fontsize=8)
                else:
                    ax2.tick_params(axis="y", labelright=False)
                ax.tick_params(axis="y", labelcolor="#1d4ed8")
                ax2.tick_params(axis="y", labelcolor="#b91c1c")
                ax2.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.set_xscale("log")
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.tick_params(axis="both", labelsize=8)
            ax.grid(alpha=0.23)
            ax.set_title(f"{data_source} | {config}", fontsize=8.8)
            if i == len(data_sources) - 1:
                ax.set_xlabel("rho")
    if not plotted:
        plt.close(fig)
        return None
    if legend_handles:
        fig.legend(
            list(legend_handles.values()),
            list(legend_handles.keys()),
            loc="upper center",
            ncol=min(2, len(legend_handles)),
            frameon=False,
            bbox_to_anchor=(0.5, 0.995),
        )
    focus_text = f" | focus y-axis, {clipped_note} values outside panels" if focus_y and clipped_note else (" | focus y-axis" if focus_y else "")
    fig.suptitle(f"LGBM {split}: {spec['title']} vs rho{focus_text}", fontweight="bold", y=1.025)
    fig.subplots_adjust(hspace=0.45, wspace=0.34, top=0.91)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_metric_comparison_plots(df: pd.DataFrame, plot_dir: Path) -> List[Path]:
    paths: List[Path] = []
    for focus_axes, suffix in [(False, "full"), (True, "focus")]:
        for maybe in [
            _save_metric_matrix(
                df,
                specs=_direct_theory_metric_specs(),
                title="Direct projection-theory checks",
                path=plot_dir / "metric_comparison" / f"direct_theory_empirical_checks_{suffix}.png",
                focus_axes=focus_axes,
            ),
            _save_metric_matrix(
                df,
                specs=_metric_driver_specs(),
                title="Theory drivers vs empirical assessment and accuracy metrics",
                path=plot_dir / "metric_comparison" / f"theory_driver_empirical_metric_response_{suffix}.png",
                focus_axes=focus_axes,
            ),
        ]:
            if maybe is not None:
                paths.append(maybe)
    evolution_dir = plot_dir / "rho_metric_evolution"
    for split in ["assessment", "test"]:
        for spec in _lgbm_evolution_specs():
            maybe = _save_lgbm_rho_evolution_plot(
                df,
                split=split,
                spec=spec,
                path=evolution_dir / f"lgbm_{spec['filename']}_evolution_{split}.png",
            )
            if maybe is not None:
                paths.append(maybe)
    return paths


def _operating_point_table(df: pd.DataFrame, out_dir: Path, targets: tuple[float, ...] = (1.62, 2.807, 4.861)) -> Optional[Path]:
    if df.empty or "rho" not in df.columns:
        return None
    d = df.copy()
    d["_rho_numeric"] = _numeric(d, "rho")
    d = d[np.isfinite(d["_rho_numeric"])].copy()
    if d.empty:
        return None

    group_cols = _present(d, ["comparison_family", "data_source", "assessment_year", "split", "config_key", "fit_label"])
    value_cols = _present(d, [
        "rho", "q_theory", "q_empirical_signed", "q_error_empirical_minus_theory",
        "covariance_reduction_theory", "covariance_reduction_empirical",
        "C_log_resid_logprice", "C_log_resid_logprice_theory", "C_log_resid_logprice_error",
        "delta_MSE_log", "delta_MSE_log_theory", "delta_MSE_log_error",
        "R2", "baseline_R2", "empirical_R2_delta",
        "MSE_log", "baseline_MSE_log", "empirical_MSE_log_frac_delta",
        "RMSE_log",
        "RMSE", "baseline_RMSE", "empirical_RMSE_frac_delta",
        "MAE", "baseline_MAE", "empirical_MAE_frac_delta",
        "MAPE", "baseline_MAPE", "empirical_MAPE_frac_delta",
        "MdAPE", "baseline_MdAPE", "empirical_MdAPE_frac_delta",
        "COD", "PRD", "baseline_PRD", "empirical_PRD_error_reduction",
        "PRB", "baseline_PRB", "empirical_PRB_error_reduction",
        "VEI", "baseline_VEI", "empirical_VEI_error_reduction",
    ])

    rows: List[Dict[str, Any]] = []
    for key, g in d.groupby(group_cols, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        rho = _numeric(g, "rho")
        if not np.isfinite(rho).any():
            continue
        for target in targets:
            idx = (rho - target).abs().idxmin()
            source = d.loc[idx]
            row: Dict[str, Any] = {col: val for col, val in zip(group_cols, key_tuple)}
            row["target_rho"] = target
            row["selected_rho"] = float(source["_rho_numeric"])
            row["rho_distance"] = abs(row["selected_rho"] - target)
            for col in value_cols:
                row[col] = source.get(col, np.nan)
            rows.append(row)

    if not rows:
        return None
    out = pd.DataFrame(rows)
    sort_cols = _present(out, ["comparison_family", "data_source", "assessment_year", "split", "config_key", "fit_label", "target_rho"])
    if sort_cols:
        out = out.sort_values(sort_cols)
    path = out_dir / "projection_theory_operating_points_rho_targets.csv"
    out.to_csv(path, index=False)
    return path


def _high_rho_failure_table(df: pd.DataFrame, out_dir: Path, rho_min: float = 10.0) -> Optional[Path]:
    needed = {"rho", "delta_MSE_log", "delta_MSE_log_theory"}
    if df.empty or not needed.issubset(df.columns):
        return None
    d = df[df["comparison_family"].astype(str).eq("lgbm_retrained_local_approx")].copy()
    if d.empty:
        return None
    d["_rho_numeric"] = _numeric(d, "rho")
    d["_delta_mse_empirical"] = _numeric(d, "delta_MSE_log")
    d["_delta_mse_theory"] = _numeric(d, "delta_MSE_log_theory")
    d = d[np.isfinite(d["_rho_numeric"]) & (d["_rho_numeric"] >= rho_min)].copy()
    if d.empty:
        return None

    d["mse_delta_ratio_empirical_to_theory"] = d["_delta_mse_empirical"] / d["_delta_mse_theory"].replace(0.0, np.nan)
    d["mse_delta_abs_error"] = (d["_delta_mse_empirical"] - d["_delta_mse_theory"]).abs()
    d["mse_delta_signed_error"] = d["_delta_mse_empirical"] - d["_delta_mse_theory"]
    d["q_abs_error"] = _numeric(d, "q_error_empirical_minus_theory").abs()
    d["highlight_ccao_sim2023_cv_top1_r2"] = (
        d["data_source"].astype(str).eq("ccao_sim2023")
        & d.get("config_key", pd.Series("", index=d.index)).astype(str).eq("cv_top1_r2")
    )

    keep_cols = _present(d, [
        "comparison_family", "data_source", "assessment_year", "split", "config_key", "rho",
        "q_theory", "q_empirical_signed", "q_error_empirical_minus_theory", "q_abs_error",
        "C_log_resid_logprice", "C_log_resid_logprice_theory",
        "delta_MSE_log", "delta_MSE_log_theory", "mse_delta_signed_error",
        "mse_delta_abs_error", "mse_delta_ratio_empirical_to_theory",
        "R2", "baseline_R2", "empirical_R2_delta", "MSE_log", "baseline_MSE_log",
        "empirical_MSE_log_frac_delta", "PRD", "PRB", "VEI", "COD", "MdAPE",
        "highlight_ccao_sim2023_cv_top1_r2",
    ])
    out = d[keep_cols].sort_values(
        _present(d, ["split", "data_source", "config_key", "rho"])
    )
    path = out_dir / "projection_theory_high_rho_failures.csv"
    out.to_csv(path, index=False)
    return path


def _decile_local_diagnostics(df: pd.DataFrame, out_dir: Path) -> Optional[Path]:
    if df.empty:
        return None
    d = df.copy()
    derived: Dict[str, pd.Series] = {}
    metric_specs = {
        "mean_ratio": "MeanRatio",
        "median_ratio": "MedianRatio",
        "weighted_mean_ratio": "WeightedMeanRatio",
        "prd": "PRD",
        "prb": "PRB",
        "mpe": "MPE",
        "mean_error": "MeanError",
        "mape": "MAPE",
        "mdape": "MdAPE",
    }
    for label, prefix in metric_specs.items():
        cols = [f"{prefix}_q10_bin{i}" for i in range(1, 11) if f"{prefix}_q10_bin{i}" in d.columns]
        if not cols:
            continue
        vals = d[cols].apply(pd.to_numeric, errors="coerce")
        derived[f"q10_{label}_min"] = vals.min(axis=1)
        derived[f"q10_{label}_max"] = vals.max(axis=1)
        derived[f"q10_{label}_spread"] = vals.max(axis=1) - vals.min(axis=1)
        derived[f"q10_{label}_abs_max"] = vals.abs().max(axis=1)

    count_cols = [f"Count_q10_bin{i}" for i in range(1, 11) if f"Count_q10_bin{i}" in d.columns]
    if count_cols:
        counts = d[count_cols].apply(pd.to_numeric, errors="coerce")
        derived["q10_count_min"] = counts.min(axis=1)
        derived["q10_count_max"] = counts.max(axis=1)
        derived["q10_count_total"] = counts.sum(axis=1)

    if not derived:
        return None

    id_cols = _present(d, [
        "comparison_family", "data_source", "assessment_year", "split", "config_key", "fit_label",
        "rho", "q_theory", "q_empirical_signed", "q_error_empirical_minus_theory",
        "delta_MSE_log", "delta_MSE_log_theory", "R2", "MSE_log", "RMSE_log",
        "PRD", "PRB", "VEI", "COD", "MdAPE",
    ])
    out = pd.concat([d[id_cols].reset_index(drop=True), pd.DataFrame(derived).reset_index(drop=True)], axis=1)
    sort_cols = _present(out, ["comparison_family", "data_source", "assessment_year", "split", "config_key", "fit_label", "rho"])
    if sort_cols:
        out = out.sort_values(sort_cols)
    path = out_dir / "projection_theory_price_decile_diagnostics.csv"
    out.to_csv(path, index=False)
    return path


def _save_lgbm_rho_path_plot(
    df: pd.DataFrame,
    *,
    split: str,
    empirical_col: str,
    theory_col: str,
    ylabel: str,
    title: str,
    path: Path,
) -> Optional[Path]:
    required = {"comparison_family", "split", "data_source", "config_key", "rho", empirical_col, theory_col}
    if df.empty or not required.issubset(df.columns):
        return None
    d = df[
        df["comparison_family"].astype(str).eq("lgbm_retrained_local_approx")
        & df["split"].astype(str).eq(split)
    ].copy()
    if d.empty:
        return None
    data_sources = sorted(d["data_source"].dropna().astype(str).unique())
    configs = sorted(d["config_key"].dropna().astype(str).unique())
    if not data_sources or not configs:
        return None

    fig, axes = plt.subplots(
        len(data_sources),
        len(configs),
        figsize=(4.4 * len(configs), 3.0 * len(data_sources)),
        sharex=True,
        squeeze=False,
    )
    plotted = False
    for i, data_source in enumerate(data_sources):
        for j, config in enumerate(configs):
            ax = axes[i, j]
            g = d[d["data_source"].astype(str).eq(data_source) & d["config_key"].astype(str).eq(config)].copy()
            if g.empty:
                ax.set_axis_off()
                continue
            g["_rho"] = _numeric(g, "rho")
            g["_empirical"] = _numeric(g, empirical_col)
            g["_theory"] = _numeric(g, theory_col)
            g = g[np.isfinite(g["_rho"])].sort_values("_rho")
            if g.empty:
                ax.set_axis_off()
                continue
            if np.isfinite(g["_empirical"]).any():
                ax.plot(g["_rho"], g["_empirical"], color="#2563eb", lw=1.8, label="empirical")
                plotted = True
            if np.isfinite(g["_theory"]).any():
                ax.plot(g["_rho"], g["_theory"], color="#111827", lw=1.5, ls="--", label="theory")
                plotted = True
            ax.axhline(0.0, color="#9ca3af", lw=0.8, alpha=0.65)
            ax.set_xscale("log")
            ax.grid(alpha=0.25)
            ax.set_title(f"{data_source} | {config}", fontsize=9)
            if i == len(data_sources) - 1:
                ax.set_xlabel("rho")
            if j == 0:
                ax.set_ylabel(ylabel)
    if not plotted:
        plt.close(fig)
        return None
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle(title, fontweight="bold", y=1.02)
    fig.subplots_adjust(hspace=0.45, wspace=0.28, top=0.92)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_high_rho_failure_plot(df: pd.DataFrame, path: Path, rho_min: float = 10.0) -> Optional[Path]:
    needed = {"comparison_family", "split", "data_source", "config_key", "rho", "delta_MSE_log", "delta_MSE_log_theory"}
    if df.empty or not needed.issubset(df.columns):
        return None
    d = df[df["comparison_family"].astype(str).eq("lgbm_retrained_local_approx")].copy()
    if d.empty:
        return None
    d["_rho"] = _numeric(d, "rho")
    d["_ratio"] = _numeric(d, "delta_MSE_log") / _numeric(d, "delta_MSE_log_theory").replace(0.0, np.nan)
    d["_abs_error"] = (_numeric(d, "delta_MSE_log") - _numeric(d, "delta_MSE_log_theory")).abs()
    d = d[np.isfinite(d["_rho"]) & (d["_rho"] >= rho_min) & np.isfinite(d["_ratio"]) & (d["_ratio"] > 0.0)].copy()
    if d.empty:
        return None

    splits = sorted(d["split"].dropna().astype(str).unique())
    fig, axes = plt.subplots(1, len(splits), figsize=(6.3 * len(splits), 5.0), squeeze=False, sharey=True)
    markers = {"cv_top1_r2": "o", "cv_top2_r2": "s", "test_best_r2": "^"}
    colors = dict(zip(sorted(d["data_source"].dropna().astype(str).unique()), plt.get_cmap("tab10").colors))
    for ax, split in zip(axes.ravel(), splits):
        gsplit = d[d["split"].astype(str).eq(split)]
        for (data_source, config), g in gsplit.groupby(["data_source", "config_key"], dropna=False):
            label = f"{data_source} | {config}"
            is_focus = str(data_source) == "ccao_sim2023" and str(config) == "cv_top1_r2"
            ax.scatter(
                g["_rho"],
                g["_ratio"],
                s=46 if is_focus else 25,
                marker=markers.get(str(config), "o"),
                color=colors.get(str(data_source), "#6b7280"),
                edgecolor="#111827" if is_focus else "none",
                linewidth=1.0 if is_focus else 0.0,
                alpha=0.95 if is_focus else 0.65,
                label=label if split == splits[0] else None,
            )
        ax.axhline(1.0, color="#111827", ls="--", lw=1.0, alpha=0.75)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("rho")
        ax.set_title(split)
        ax.grid(alpha=0.25)
    axes[0, 0].set_ylabel("empirical / theory delta MSE_log")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    fig.suptitle(f"High-rho LGBM MSE-cost deviations (rho >= {rho_min:g})", fontweight="bold")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_taylor_error_vs_rmse_plot(df: pd.DataFrame, path: Path) -> Optional[Path]:
    from matplotlib.ticker import MaxNLocator

    required = {"comparison_family", "split", "RMSE_log", "C_ratio_price_taylor1_rel_error", "C_ratio_price_taylor2_rel_error"}
    if df.empty or not required.issubset(df.columns):
        return None
    rows: List[Dict[str, Any]] = []
    for order, col in [("first order", "C_ratio_price_taylor1_rel_error"), ("second order", "C_ratio_price_taylor2_rel_error")]:
        d = df[["comparison_family", "split", "RMSE_log", col]].copy()
        d["order"] = order
        d["abs_relative_error"] = _numeric(d, col).abs()
        d["rmse_log"] = _numeric(d, "RMSE_log")
        rows.extend(d[["comparison_family", "split", "order", "rmse_log", "abs_relative_error"]].to_dict("records"))
    plot_df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).dropna(subset=["rmse_log", "abs_relative_error"])
    if plot_df.empty:
        return None

    groups = list(plot_df.groupby(["comparison_family", "split"], dropna=False))
    ncols = 2
    nrows = int(np.ceil(len(groups) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4.2 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.set_axis_off()
    markers = {"first order": "o", "second order": "^"}
    colors = {"first order": "#dc2626", "second order": "#2563eb"}
    for ax, ((family, split), g) in zip(axes.ravel(), groups):
        ax.set_axis_on()
        x = pd.to_numeric(g["rmse_log"], errors="coerce")
        y = pd.to_numeric(g["abs_relative_error"], errors="coerce")
        x_vals = x[np.isfinite(x)]
        y_vals = y[np.isfinite(y) & (y > 0.0)]
        x_hi = float(np.nanquantile(x_vals, 0.98)) if x_vals.size >= 8 else float(np.nanmax(x_vals)) if x_vals.size else np.nan
        x_lo = float(np.nanmin(x_vals)) if x_vals.size else np.nan
        if np.isfinite(x_lo) and np.isfinite(x_hi) and x_hi > x_lo:
            pad = 0.04 * (x_hi - x_lo)
            ax.set_xlim(x_lo - pad, x_hi + pad)
        y_lo = float(np.nanquantile(y_vals, 0.02)) if y_vals.size >= 8 else float(np.nanmin(y_vals)) if y_vals.size else np.nan
        y_hi = float(np.nanquantile(y_vals, 0.98)) if y_vals.size >= 8 else float(np.nanmax(y_vals)) if y_vals.size else np.nan
        if np.isfinite(y_lo) and np.isfinite(y_hi) and y_hi > y_lo:
            ax.set_ylim(max(y_lo * 0.75, 1e-8), y_hi * 1.25)
        outside = 0
        if np.isfinite(x_hi):
            outside += int((x > x_hi).sum())
        if np.isfinite(y_hi):
            outside += int((y > y_hi).sum())
        for order, order_df in g.groupby("order", dropna=False):
            ax.scatter(
                order_df["rmse_log"],
                order_df["abs_relative_error"].clip(lower=1e-8),
                s=22,
                alpha=0.72,
                marker=markers.get(str(order), "o"),
                color=colors.get(str(order), "#6b7280"),
                label=str(order),
            )
        ax.set_yscale("log")
        ax.set_xlabel("RMSE_log")
        ax.set_ylabel("abs relative error")
        ax.set_title(f"{family}\n{split}", fontsize=10)
        ax.grid(alpha=0.25)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.tick_params(axis="both", labelsize=8)
        if outside:
            ax.text(
                0.02,
                0.98,
                f"focus scale: {outside} outside",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=7,
                bbox=dict(facecolor="white", alpha=0.74, edgecolor="none"),
            )
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Taylor bridge error vs log-residual RMSE\nfocus axes use robust RMSE/error scale; outliers remain in CSVs", fontweight="bold", y=0.995)
    fig.subplots_adjust(hspace=0.55, wspace=0.28, top=0.92)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_decile_diagnostic_plot(decile_path: Optional[Path], plot_dir: Path) -> List[Path]:
    if decile_path is None or not decile_path.exists():
        return []
    d = pd.read_csv(decile_path)
    if d.empty or "rho" not in d.columns:
        return []
    metrics = [
        ("q10_mean_ratio_spread", "q10 mean-ratio spread"),
        ("q10_prb_abs_max", "max absolute q10 PRB"),
    ]
    if not any(col in d.columns for col, _ in metrics):
        return []
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.4 * len(metrics), 5.0), squeeze=False)
    colors = {
        "lgbm_retrained_local_approx": "#2563eb",
        "linear_exact_projection": "#dc2626",
    }
    markers = {"assessment": "o", "test": "s", "train_assess_fit": "^", "train_test_fit": "v"}
    for ax, (metric, ylabel) in zip(axes.ravel(), metrics):
        if metric not in d.columns:
            ax.set_axis_off()
            continue
        for (family, split), g in d.groupby(["comparison_family", "split"], dropna=False):
            x = _numeric(g, "rho")
            y = _numeric(g, metric)
            finite = np.isfinite(x) & np.isfinite(y)
            if finite.any():
                ax.scatter(
                    x[finite],
                    y[finite],
                    s=22,
                    alpha=0.68,
                    color=colors.get(str(family), "#6b7280"),
                    marker=markers.get(str(split), "o"),
                    label=f"{family} | {split}",
                )
        ax.set_xscale("log")
        ax.set_xlabel("rho")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    fig.suptitle("Price-decile local diagnostics along the rho path", fontweight="bold")
    path = plot_dir / "price_decile_diagnostics_vs_rho.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return [path]


def _extra_outputs(df: pd.DataFrame, out_dir: Path) -> Dict[str, List[Path]]:
    plot_dir = out_dir / "plots"
    table_paths: List[Path] = []
    plot_paths: List[Path] = []
    for maybe in [
        _operating_point_table(df, out_dir),
        _high_rho_failure_table(df, out_dir),
        _decile_local_diagnostics(df, out_dir),
    ]:
        if maybe is not None:
            table_paths.append(maybe)

    decile_path = next((p for p in table_paths if p.name == "projection_theory_price_decile_diagnostics.csv"), None)
    for split in ["assessment", "test"]:
        for maybe in [
            _save_lgbm_rho_path_plot(
                df,
                split=split,
                empirical_col="q_empirical_signed",
                theory_col="q_theory",
                ylabel="q = C(r, log price) / C0",
                title=f"LGBM q path vs rho ({split})",
                path=plot_dir / f"lgbm_q_vs_rho_by_dataset_config_{split}.png",
            ),
            _save_lgbm_rho_path_plot(
                df,
                split=split,
                empirical_col="C_log_resid_logprice",
                theory_col="C_log_resid_logprice_theory",
                ylabel="Cov(log residual, log price)",
                title=f"LGBM covariance path vs rho ({split})",
                path=plot_dir / f"lgbm_covariance_vs_rho_by_dataset_config_{split}.png",
            ),
        ]:
            if maybe is not None:
                plot_paths.append(maybe)
    for maybe in [
        _save_high_rho_failure_plot(df, plot_dir / "high_rho_lgbm_mse_cost_deviation.png"),
        _save_taylor_error_vs_rmse_plot(df, plot_dir / "taylor_error_vs_rmse_log.png"),
    ]:
        if maybe is not None:
            plot_paths.append(maybe)
    plot_paths.extend(_save_decile_diagnostic_plot(decile_path, plot_dir))
    plot_paths.extend(_save_metric_comparison_plots(df, plot_dir))
    return {"tables": table_paths, "plots": plot_paths}


def _plots(df: pd.DataFrame, out_dir: Path) -> List[Path]:
    plot_dir = out_dir / "plots"
    paths: List[Path] = []
    for maybe in [
        _save_scatter(
            df,
            xcol="q_theory",
            ycol="q_empirical_signed",
            xlabel="theory q",
            ylabel="empirical C/C0",
            title="Projection theory q check",
            path=plot_dir / "combined_q_empirical_vs_theory.png",
            focus_axes=True,
        ),
        _save_scatter(
            df,
            xcol="delta_MSE_log_theory",
            ycol="delta_MSE_log",
            xlabel="theory delta MSE_log",
            ylabel="empirical delta MSE_log",
            title="Second-order log-MSE cost check",
            path=plot_dir / "combined_mse_cost_empirical_vs_theory.png",
            focus_axes=True,
        ),
        _save_scatter(
            df,
            xcol="covariance_reduction_theory",
            ycol="empirical_PRD_error_reduction",
            xlabel="theory covariance reduction",
            ylabel="empirical PRD error reduction",
            title="First-order PRD bridge check",
            path=plot_dir / "combined_prd_error_reduction_vs_theory.png",
            focus_axes=True,
        ),
        _save_faceted_scatter(
            df,
            xcol="q_theory",
            ycol="q_empirical_signed",
            xlabel="theory q",
            ylabel="empirical C/C0",
            title="Projection theory q check by family and split",
            path=plot_dir / "faceted_q_empirical_vs_theory.png",
            focus_axes=True,
        ),
        _save_faceted_scatter(
            df,
            xcol="delta_MSE_log_theory",
            ycol="delta_MSE_log",
            xlabel="theory delta MSE_log",
            ylabel="empirical delta MSE_log",
            title="Second-order log-MSE cost by family and split",
            path=plot_dir / "faceted_mse_cost_empirical_vs_theory.png",
            focus_axes=True,
        ),
        _save_faceted_scatter(
            df,
            xcol="covariance_reduction_theory",
            ycol="empirical_PRD_error_reduction",
            xlabel="theory covariance reduction",
            ylabel="empirical PRD error reduction",
            title="First-order PRD bridge by family and split",
            path=plot_dir / "faceted_prd_error_reduction_vs_theory.png",
            focus_axes=True,
        ),
    ]:
        if maybe is not None:
            paths.append(maybe)

    if {"C_ratio_price_taylor1_rel_error", "C_ratio_price_taylor2_rel_error"}.issubset(df.columns):
        d = df.copy()
        rows = []
        for family, g in d.groupby("comparison_family", dropna=False):
            for order, col in [("first order", "C_ratio_price_taylor1_rel_error"), ("second order", "C_ratio_price_taylor2_rel_error")]:
                vals = _numeric(g, col)
                for val in vals[np.isfinite(vals)]:
                    rows.append({"comparison_family": family, "order": order, "rel_error": float(val)})
        box = pd.DataFrame(rows)
        if not box.empty:
            fig, ax = plt.subplots(figsize=(8.0, 5.2))
            labels = []
            series = []
            for (family, order), g in box.groupby(["comparison_family", "order"], dropna=False):
                labels.append(f"{family}\n{order}")
                series.append(g["rel_error"].to_numpy(dtype=float))
            ax.boxplot(series, tick_labels=labels, showfliers=False)
            ax.set_ylabel("relative error for Cov(r, price)")
            ax.set_title("Taylor bridge error distribution")
            ax.grid(axis="y", alpha=0.25)
            plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
            p = plot_dir / "combined_taylor_bridge_error_boxplot.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, dpi=180, bbox_inches="tight")
            plt.close(fig)
            paths.append(p)
    return paths


def run(args: argparse.Namespace) -> Dict[str, str]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    linear_df = _load_linear(Path(args.linear_root)) if args.linear_root else pd.DataFrame()
    lgbm_theory_root = Path(args.lgbm_theory_root) if args.lgbm_theory_root else None
    lgbm_rho_sweep_root = Path(args.lgbm_rho_sweep_root) if args.lgbm_rho_sweep_root else (
        _infer_lgbm_rho_sweep_root(lgbm_theory_root) if lgbm_theory_root is not None else None
    )
    lgbm_df = _load_lgbm(lgbm_theory_root, lgbm_rho_sweep_root) if lgbm_theory_root is not None else pd.DataFrame()
    combined = pd.concat([d for d in [linear_df, lgbm_df] if not d.empty], ignore_index=True) if (not linear_df.empty or not lgbm_df.empty) else pd.DataFrame()
    combined = _add_error_columns(combined)

    comparison_path = out_dir / "projection_theory_combined_comparison.csv"
    summary_path = out_dir / "projection_theory_combined_summary.csv"
    diagnostic_path = out_dir / "projection_theory_diagnostic_summary.csv"
    combined.to_csv(comparison_path, index=False)
    summary_df = _summary(combined)
    summary_df.to_csv(summary_path, index=False)
    diagnostic_df = _diagnostic_summary(combined)
    diagnostic_df.to_csv(diagnostic_path, index=False)
    plot_paths = _plots(combined, out_dir)
    extra = _extra_outputs(combined, out_dir)
    extra_table_paths = extra["tables"]
    plot_paths.extend(extra["plots"])

    missing_direct = []
    for required in ["q_empirical_signed", "delta_MSE_log", "C_log_resid_logprice"]:
        if required not in combined.columns or combined[required].notna().sum() == 0:
            missing_direct.append(required)

    report = [
        "# Projection-Theory Empirical Comparison",
        "",
        f"- Linear rows loaded: {linear_df.shape[0]}",
        f"- LGBM rows loaded: {lgbm_df.shape[0]}",
        f"- Combined rows: {combined.shape[0]}",
        "",
        "## Summary",
        "",
    ]
    if not summary_df.empty:
        report.append(summary_df.to_markdown(index=False, floatfmt=".6g"))
    else:
        report.append("No comparison rows were loaded.")
    if not diagnostic_df.empty:
        report.extend(["", "## Diagnostic Summary", ""])
        display_cols = [
            "comparison_family", "split", "n_rows", "q_corr", "q_error_mae",
            "delta_MSE_log_error_mae", "mse_corr", "taylor2_better_share",
            "PRD_error_reduction_median", "PRB_error_reduction_median", "VEI_error_reduction_median",
            "q_negative_share", "q_abs_greater_than_one_share",
        ]
        report.append(diagnostic_df[[c for c in display_cols if c in diagnostic_df.columns]].to_markdown(index=False, floatfmt=".6g"))
    if missing_direct:
        report.extend([
            "",
            "## Missing Direct Columns",
            "",
            "The following direct theory columns were absent or empty. Re-run `rho_sweep_experiments.sh` after the quick-test metric patch, then re-run `theory_rho_range_experiments.sh merge`.",
            "",
            *[f"- `{col}`" for col in missing_direct],
        ])
    report.extend(["", "## Extra Outcome Notes", ""])
    report.extend([
        "- Rho-target operating points use the nearest available fitted rho for each model/dataset/split/configuration.",
        "- Price-decile diagnostics use saved q10 metric columns; LGBM q10 columns are merged from the matching rho-sweep metrics when that sibling run is available.",
        "- Metric-comparison matrices include full-axis and focus-axis versions; focus axes use robust central scaling and keep outliers in the CSVs/tables.",
        "- Original-scale RMSE and MAPE diagnostics are treated as collateral accuracy checks against the log-MSE theory driver, not as direct theorem targets.",
        "- Rho-evolution plots separate direct mechanism metrics, first-order bridge metrics, and collateral accuracy metrics; collateral plots use theory drivers on a second axis rather than implying exact theory prediction.",
        "- Township diagnostics are not generated here because the current artifacts do not save penalized row-level predictions or township group metrics.",
    ])
    report.extend(["", "## Artifacts", "", f"- `{comparison_path.name}`", f"- `{summary_path.name}`", f"- `{diagnostic_path.name}`"])
    for path in extra_table_paths:
        report.append(f"- `{path.name}`")
    for path in plot_paths:
        report.append(f"- `{path.relative_to(out_dir)}`")
    report_path = out_dir / "projection_theory_combined_report.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"[collect] wrote {comparison_path}", flush=True)
    print(f"[collect] wrote {summary_path}", flush=True)
    print(f"[collect] wrote {diagnostic_path}", flush=True)
    print(f"[collect] wrote {report_path}", flush=True)
    return {"comparison": str(comparison_path), "summary": str(summary_path), "diagnostic": str(diagnostic_path), "report": str(report_path)}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Collect projection-theory comparison outputs.")
    p.add_argument("--linear-root", default="output/projection_linear")
    p.add_argument("--lgbm-theory-root", default="output/theory_rho_range_500_estimators")
    p.add_argument("--lgbm-rho-sweep-root", default=None)
    p.add_argument("--out-dir", default="output/projection_theory_comparison")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
