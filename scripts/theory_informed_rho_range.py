#!/usr/bin/env python
"""
Theory-informed rho range analysis for LGBCovPenalty[diff].

The covariance-penalty LightGBM objective has gradients equivalent to minimizing

    E[(f(X) - Y)^2] + 0.5 * rho * Cov(f(X) - Y, Y)^2

up to a constant multiplier. Under the local rank-one approximation
f0(X) ~= E[Y|X], the remaining covariance fraction is

    q(rho) = 1 / (1 + rho * Var(f0) / 2).

This script reuses existing rho-sweep artifacts first. Newer sweeps may contain
baseline_predictions_*.parquet with exact LGBM log predictions. Older sweeps
only contain quick_test_metrics_*.csv; for those, the script uses the LGBM row
plus the raw target split and applies the Bayes-style approximation
Var(f0) ~= Var(Y) - MSE_log and Cov(f0-Y, Y) ~= -MSE_log. No baseline model is
fit when --no-fit-baseline is set.
"""
from __future__ import annotations

import argparse
import math
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

T0 = time.perf_counter()
TIMING: List[Dict[str, Any]] = []
FOLDER_RE = re.compile(r"^(?P<src>.+?)_assess(?P<year>\d{4})__(?P<cfg>.+)_(?P<cid>[0-9a-f]{8})$")
SPLIT_SUFFIX = {"test": "test", "assessment": "assess"}


def log(msg: str, **fields: Any) -> None:
    elapsed = time.perf_counter() - T0
    suffix = " | " + " | ".join(f"{k}={v}" for k, v in fields.items()) if fields else ""
    print(f"[theory-rho +{elapsed:.1f}s] {msg}{suffix}", flush=True)


@contextmanager
def timed(step: str, **fields: Any):
    start = time.perf_counter()
    log(f"start: {step}", **fields)
    status = "ok"
    try:
        yield
    except Exception:
        status = "failed"
        raise
    finally:
        seconds = time.perf_counter() - start
        TIMING.append({"step": step, "seconds": seconds, "status": status, **fields})
        log(f"end: {step}", seconds=f"{seconds:.3f}", status=status, **fields)


def parse_csv(raw: str, cast=str) -> List[Any]:
    return [cast(x.strip()) for x in str(raw or "").split(",") if x.strip()]


def finite_float(x: Any, default: float = float("nan")) -> float:
    try:
        y = float(x)
    except (TypeError, ValueError):
        return default
    return y if np.isfinite(y) else default


def row_float(row: Any, key: str, default: float = float("nan")) -> float:
    try:
        return finite_float(row.get(key, default), default)
    except AttributeError:
        return default


def safe_var(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean((x - np.mean(x)) ** 2)) if x.size else float("nan")


def safe_cov(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    return float(np.mean((x - np.mean(x)) * (y - np.mean(y)))) if x.size else float("nan")


def safe_mean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else float("nan")


def rho_from_q(q: float, A: float) -> float:
    q = finite_float(q)
    A = finite_float(A)
    if not np.isfinite(q) or not np.isfinite(A) or q <= 0.0 or q >= 1.0 or A <= 0.0:
        return float("nan")
    return float(2.0 * (1.0 - q) / (q * A))


def q_from_rho(rho: float, A: float) -> float:
    rho = finite_float(rho)
    A = finite_float(A)
    if not np.isfinite(rho) or not np.isfinite(A) or rho < 0.0 or A <= 0.0:
        return float("nan")
    return float(1.0 / (1.0 + rho * A / 2.0))


def rho_from_accuracy_budget(alpha: float, B: float, C0: float, A: float) -> float:
    alpha, B, C0, A = map(finite_float, [alpha, B, C0, A])
    if not all(np.isfinite(v) for v in [alpha, B, C0, A]) or alpha <= 0 or B <= 0 or A <= 0 or C0 == 0:
        return float("nan")
    q_min = 1.0 - math.sqrt(max(0.0, alpha * B * A / (C0 ** 2)))
    q_min = float(np.clip(q_min, 1e-9, 1.0 - 1e-9))
    return rho_from_q(q_min, A)


def rho_grid(lo: float, hi: float, count: int, scale: str) -> np.ndarray:
    if count <= 0:
        return np.asarray([], dtype=float)
    if count == 1:
        return np.asarray([float(lo)], dtype=float)
    scale = str(scale).lower()
    if scale in {"log", "geom", "geometric"}:
        return np.geomspace(float(lo), float(hi), int(count))
    if scale == "linear":
        return np.linspace(float(lo), float(hi), int(count))
    raise ValueError("horizon rho scale must be log/geom/geometric or linear")


def default_specs() -> str:
    return ";".join(
        [
            "ccao2025:2025:./data/CCAO/2025/training_data.parquet",
            "ccao_old:2024:./data/CCAO/2025/training_data_old.parquet",
            "ccao_sim2024:2023:./data/CCAO/2025/training_data_sim2024.parquet",
            "ccao_sim2023:2022:./data/CCAO/2025/training_data_sim2023.parquet",
        ]
    )


def parse_specs(raw: str) -> List[Tuple[str, int, str]]:
    out: List[Tuple[str, int, str]] = []
    for chunk in str(raw or "").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        src, year, path = chunk.split(":", 2)
        out.append((src.strip(), int(year), path.strip()))
    return out


def analysis_specs(args: argparse.Namespace) -> List[Tuple[str, int, str]]:
    specs = parse_specs(args.data_source_specs)
    if specs:
        return specs
    label = args.data_source_label or Path(args.data_path).stem
    return [(label, y, args.data_path) for y in parse_csv(args.assessment_years, int)]


def read_parquet(path: str, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, columns=columns, engine="fastparquet")
    except TypeError:
        df = pd.read_parquet(path, engine="fastparquet")
        return df.loc[:, list(columns)].copy() if columns is not None else df


def load_target_splits(
    data_path: str,
    target_col: str,
    date_col: str,
    sample_frac: Optional[float],
    seed: int,
    assessment_year: int,
    train_prop: float,
) -> Dict[str, Optional[np.ndarray]]:
    cols = ["ind_pin_is_multicard", "sv_is_outlier", target_col, date_col]
    with timed("data: read target columns", data_path=data_path, assessment_year=assessment_year):
        df = read_parquet(data_path, columns=cols)
    df = df[
        (~df["ind_pin_is_multicard"].astype("bool").fillna(True))
        & (~df["sv_is_outlier"].astype("bool").fillna(True))
    ].copy()
    if sample_frac is not None and float(sample_frac) < 1.0:
        df = df.sample(frac=float(sample_frac), random_state=int(seed)).copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    assess = df.loc[df[date_col].dt.year == int(assessment_year), :]
    pre = df.loc[df[date_col].dt.year < int(assessment_year), :]
    split_idx = int(float(train_prop) * pre.shape[0])
    test = pre.iloc[split_idx:, :]

    def ylog(frame: pd.DataFrame) -> Optional[np.ndarray]:
        y = pd.to_numeric(frame[target_col], errors="coerce").to_numpy(dtype=float)
        y = y[np.isfinite(y) & (y > 0)]
        return np.log(y) if y.size else None

    out = {"test": ylog(test), "assessment": ylog(assess)}
    log(
        "target split ready",
        assessment_year=assessment_year,
        test_rows=0 if out["test"] is None else int(out["test"].size),
        assess_rows=0 if out["assessment"] is None else int(out["assessment"].size),
    )
    return out


def find_sweep_folder(root: str, source: str, year: int, config_key: str) -> Optional[Path]:
    root_path = Path(root)
    if not root_path.exists():
        return None
    for folder in sorted(p for p in root_path.iterdir() if p.is_dir()):
        m = FOLDER_RE.match(folder.name)
        if m and m["src"] == source and int(m["year"]) == int(year) and m["cfg"] == config_key:
            return folder
    return None


def read_predictions(path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
    except Exception:
        return None
    if "y_log" not in df.columns or "f0_log" not in df.columns:
        return None
    y = pd.to_numeric(df["y_log"], errors="coerce").to_numpy(dtype=float)
    f = pd.to_numeric(df["f0_log"], errors="coerce").to_numpy(dtype=float)
    return (y, f) if y.size and y.size == f.size else None


def read_lgbm_metric_row(folder: Optional[Path], split: str) -> Optional[pd.Series]:
    if folder is None:
        return None
    path = folder / f"quick_test_metrics_{SPLIT_SUFFIX[split]}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    rows = df.loc[df["model_name"].astype(str).eq("LGBMRegressor")]
    return rows.iloc[0] if not rows.empty else None


def prd_from_arrays(pred_price: np.ndarray, true_price: np.ndarray) -> float:
    mask = np.isfinite(pred_price) & np.isfinite(true_price) & (true_price > 0)
    pred = pred_price[mask]
    true = true_price[mask]
    if not true.size or np.sum(true) <= 0:
        return float("nan")
    return float(np.mean(pred / true) / (np.sum(pred) / np.sum(true)))


def prd_info_from_predictions(y_log: np.ndarray, f_log: np.ndarray) -> Dict[str, float]:
    price = np.exp(y_log)
    pred = np.exp(f_log)
    ratio = pred / price
    prd = prd_from_arrays(pred, price)
    mu_r = safe_mean(ratio)
    mu_p = safe_mean(price)
    cov_from_prd = mu_r * mu_p * (1.0 / prd - 1.0) if np.isfinite(mu_r * mu_p * prd) and prd != 0 else float("nan")
    cov_emp = safe_cov(ratio, price)
    return {
        "mu_ratio_price": mu_r,
        "mu_price": mu_p,
        "prd": prd,
        "cov_ratio_price_empirical": cov_emp,
        "cov_ratio_price_from_prd": cov_from_prd,
        "cov_ratio_price_identity_abs_error": abs(cov_emp - cov_from_prd) if np.isfinite(cov_emp) and np.isfinite(cov_from_prd) else float("nan"),
    }


def prd_info_from_metrics(y_log: np.ndarray, row: pd.Series) -> Dict[str, float]:
    prd = row_float(row, "PRD")
    mu_r = row_float(row, "Mean ratio")
    mu_p = safe_mean(np.exp(y_log))
    cov_from_prd = mu_r * mu_p * (1.0 / prd - 1.0) if np.isfinite(mu_r * mu_p * prd) and prd != 0 else float("nan")
    return {
        "mu_ratio_price": mu_r,
        "mu_price": mu_p,
        "prd": prd,
        "cov_ratio_price_empirical": float("nan"),
        "cov_ratio_price_from_prd": cov_from_prd,
        "cov_ratio_price_identity_abs_error": float("nan"),
    }


def valid_prd_target(prd0: float, target: float) -> bool:
    if not np.isfinite(prd0) or not np.isfinite(target):
        return False
    if prd0 > 1.0:
        return 1.0 <= target <= prd0
    if prd0 < 1.0:
        return prd0 <= target <= 1.0
    return False


def theory_from_quantities(
    *,
    source: str,
    year: int,
    config_key: str,
    split: str,
    baseline_source: str,
    n: int,
    A: float,
    B: float,
    C0: float,
    var_y: float,
    prd_info: Dict[str, float],
    prd_targets: Sequence[float],
    shrink_qs: Sequence[float],
    budgets: Sequence[float],
    empirical_range: Optional[Tuple[float, float]],
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = {"data_source": source, "assessment_year": year, "config_key": config_key, "split": split, "baseline_source": baseline_source}
    prd0 = prd_info["prd"]
    cov_rp0 = prd_info["cov_ratio_price_from_prd"]
    shrink_rows: List[Dict[str, Any]] = []
    for q in shrink_qs:
        rho = rho_from_q(q, A)
        d_mse = (C0 ** 2 / A) * ((1.0 - q) ** 2) if A > 0 and np.isfinite(C0) else float("nan")
        shrink_rows.append({**base, "q_remaining_covariance": q, "covariance_reduction": 1.0 - q, "rho_theory": rho, "delta_mse_log_theory": d_mse, "delta_mse_log_frac_of_baseline": d_mse / B if B > 0 and np.isfinite(d_mse) else float("nan")})
    prd_rows: List[Dict[str, Any]] = []
    for target in prd_targets:
        if not valid_prd_target(prd0, target):
            continue
        mu_r = prd_info["mu_ratio_price"]
        mu_p = prd_info["mu_price"]
        cov_target = mu_r * mu_p * (1.0 / target - 1.0) if np.isfinite(mu_r * mu_p * target) and target != 0 else float("nan")
        q_prd = abs(cov_target) / abs(cov_rp0) if np.isfinite(cov_target) and np.isfinite(cov_rp0) and cov_rp0 != 0 else float("nan")
        prd_rows.append({**base, "prd_baseline": prd0, "prd_target": target, "cov_ratio_price_baseline_from_prd": cov_rp0, "cov_ratio_price_target_from_prd": cov_target, "q_log_ratio_price_bridge": q_prd, "rho_prd_ratio_price_bridge": rho_from_q(q_prd, A)})
    budget_rows: List[Dict[str, Any]] = []
    for alpha in budgets:
        rho = rho_from_accuracy_budget(alpha, B, C0, A)
        q = q_from_rho(rho, A)
        budget_rows.append({**base, "accuracy_budget_frac_of_baseline_mse": alpha, "rho_max_under_budget": rho, "q_remaining_covariance_at_budget": q, "covariance_reduction_at_budget": 1.0 - q if np.isfinite(q) else float("nan")})
    prd_df = pd.DataFrame(prd_rows)
    budget_df = pd.DataFrame(budget_rows)
    rho_25 = rho_from_q(0.75, A)
    rho_50 = rho_from_q(0.50, A)
    rho_67 = rho_from_q(0.33, A)
    guidance_target = 1.03 if np.isfinite(prd0) and prd0 > 1.03 else (0.98 if np.isfinite(prd0) and prd0 < 0.98 else float("nan"))
    rho_prd = float("nan")
    if np.isfinite(guidance_target) and not prd_df.empty:
        match = prd_df[np.isclose(prd_df["prd_target"].astype(float), guidance_target)]
        if not match.empty:
            rho_prd = float(match.iloc[0]["rho_prd_ratio_price_bridge"])
    rho_budget_1pct = float("nan")
    if not budget_df.empty:
        match = budget_df[np.isclose(budget_df["accuracy_budget_frac_of_baseline_mse"].astype(float), 0.01)]
        if not match.empty:
            rho_budget_1pct = float(match.iloc[0]["rho_max_under_budget"])
    low = np.nanmax([rho_25, rho_prd]) if np.isfinite(rho_prd) else rho_25
    high = np.nanmin([rho_67, rho_budget_1pct]) if np.isfinite(rho_budget_1pct) else rho_67
    confident = float(np.clip(rho_50, low, high)) if np.isfinite(low) and np.isfinite(high) and low <= high else rho_50
    summary = {
        **base,
        "n": int(n),
        "var_y_log": var_y,
        "baseline_r2_log_from_quantities": 1.0 - B / var_y if var_y > 0 and np.isfinite(B) else float("nan"),
        "A_var_f0_log": A,
        "B_mse_log": B,
        "C0_cov_log_residual_logprice": C0,
        "bayes_optimality_diagnostic_C0_over_minus_B": C0 / (-B) if B > 0 and np.isfinite(C0) else float("nan"),
        **prd_info,
        "rho_shrink_25pct": rho_25,
        "rho_shrink_50pct": rho_50,
        "rho_shrink_67pct": rho_67,
        "prd_guidance_target": guidance_target,
        "rho_prd_guidance": rho_prd,
        "rho_budget_1pct_mse": rho_budget_1pct,
        "theory_range_low": low,
        "theory_confident_rho": confident,
        "theory_range_high": high,
        "theory_range_valid": bool(np.isfinite(low) and np.isfinite(high) and low <= high),
    }
    if empirical_range is not None:
        lo_emp, hi_emp = empirical_range
        summary["empirical_rho_low"] = lo_emp
        summary["empirical_rho_high"] = hi_emp
        summary["empirical_range_overlaps_theory"] = bool(np.isfinite(low) and np.isfinite(high) and max(lo_emp, low) <= min(hi_emp, high))
    return summary, pd.DataFrame(shrink_rows), prd_df, budget_df


def theory_from_predictions(y: np.ndarray, f: np.ndarray, **kwargs: Any):
    y = np.asarray(y, dtype=float).reshape(-1)
    f = np.asarray(f, dtype=float).reshape(-1)
    mask = np.isfinite(y) & np.isfinite(f)
    y, f = y[mask], f[mask]
    e = f - y
    return theory_from_quantities(
        n=int(y.size),
        A=safe_var(f),
        B=float(np.mean(e ** 2)) if e.size else float("nan"),
        C0=safe_cov(e, y),
        var_y=safe_var(y),
        prd_info=prd_info_from_predictions(y, f),
        **kwargs,
    )


def theory_from_metric_baseline(y: np.ndarray, row: pd.Series, **kwargs: Any):
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    var_y = safe_var(y)
    r2_log = row_float(row, "R2 (log)")
    B = (1.0 - r2_log) * var_y if np.isfinite(r2_log) and var_y > 0 else float("nan")
    if not np.isfinite(B):
        std_r = row_float(row, "Std(r)")
        B = std_r ** 2 if np.isfinite(std_r) else float("nan")
    B = max(0.0, B) if np.isfinite(B) else float("nan")
    A = max(var_y - B, 1e-12) if np.isfinite(var_y) and np.isfinite(B) else float("nan")
    summary, shrink, prd, budget = theory_from_quantities(
        n=int(y.size),
        A=A,
        B=B,
        C0=-B if np.isfinite(B) else float("nan"),
        var_y=var_y,
        prd_info=prd_info_from_metrics(y, row),
        **kwargs,
    )
    summary["metric_baseline_r2_log"] = r2_log
    summary["metric_baseline_R2_price"] = row_float(row, "R2")
    summary["metric_baseline_RMSE_price"] = row_float(row, "RMSE")
    summary["metric_baseline_MdAPE"] = row_float(row, "MdAPE")
    return summary, shrink, prd, budget


def load_empirical(root: str) -> pd.DataFrame:
    root_path = Path(root)
    if not root_path.exists():
        return pd.DataFrame()
    files = {"assessment": "quick_test_metrics_assess.csv", "test": "quick_test_metrics_test.csv", "validation": "quick_test_metrics_validation_bootstrap_avg.csv"}
    rows: List[pd.DataFrame] = []
    for folder in sorted(p for p in root_path.iterdir() if p.is_dir()):
        m = FOLDER_RE.match(folder.name)
        if not m:
            continue
        for split, name in files.items():
            path = folder / name
            if not path.exists():
                continue
            df = pd.read_csv(path)
            df["data_source"] = m["src"]
            df["assessment_year"] = int(m["year"])
            df["config_key"] = m["cfg"]
            df["config_id"] = m["cid"]
            df["split"] = split
            rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def aggregate_ranges(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    groups = [("all", summary)] + [(str(k), v) for k, v in summary.groupby("split")]
    rows = []
    for name, df in groups:
        low = pd.to_numeric(df["theory_range_low"], errors="coerce")
        high = pd.to_numeric(df["theory_range_high"], errors="coerce")
        conf = pd.to_numeric(df["theory_confident_rho"], errors="coerce")
        robust_low = float(np.nanquantile(low, 0.75)) if low.notna().any() else float("nan")
        robust_high = float(np.nanquantile(high, 0.25)) if high.notna().any() else float("nan")
        if np.isfinite(robust_low) and np.isfinite(robust_high) and robust_low > robust_high:
            robust_low = float(np.nanmedian(low))
            robust_high = float(np.nanmedian(high))
        rows.append({
            "split_group": name,
            "n_runs": int(df.shape[0]),
            "robust_theory_range_low": robust_low,
            "robust_theory_range_high": robust_high,
            "median_theory_range_low": float(np.nanmedian(low)),
            "median_confident_rho": float(np.nanmedian(conf)),
            "median_theory_range_high": float(np.nanmedian(high)),
            "median_rho_shrink_25pct": float(np.nanmedian(pd.to_numeric(df["rho_shrink_25pct"], errors="coerce"))),
            "median_rho_shrink_50pct": float(np.nanmedian(pd.to_numeric(df["rho_shrink_50pct"], errors="coerce"))),
            "median_rho_shrink_67pct": float(np.nanmedian(pd.to_numeric(df["rho_shrink_67pct"], errors="coerce"))),
            "median_bayes_diagnostic_C0_over_minus_B": float(np.nanmedian(pd.to_numeric(df["bayes_optimality_diagnostic_C0_over_minus_B"], errors="coerce"))),
        })
    return pd.DataFrame(rows)


def build_horizon(summary: pd.DataFrame, empirical: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    grid = rho_grid(args.horizon_rho_min, args.horizon_rho_max, args.horizon_rho_count, args.horizon_rho_scale)
    cov_emp = empirical[empirical.get("model_family", "").astype(str).eq("LGBCovPenalty[diff]")].copy() if not empirical.empty else pd.DataFrame()
    if not cov_emp.empty:
        cov_emp["rho"] = pd.to_numeric(cov_emp["rho"], errors="coerce")
    rows: List[Dict[str, Any]] = []
    for r in summary.to_dict("records"):
        rhos = list(grid)
        if not cov_emp.empty:
            mask = (
                cov_emp["data_source"].eq(r["data_source"])
                & cov_emp["assessment_year"].eq(r["assessment_year"])
                & cov_emp["config_key"].eq(r["config_key"])
                & cov_emp["split"].eq(r["split"])
            )
            rhos.extend(cov_emp.loc[mask, "rho"].dropna().astype(float).tolist())
        for rho in sorted(set(round(float(x), 12) for x in rhos if np.isfinite(float(x)))):
            q = q_from_rho(rho, r["A_var_f0_log"])
            d_mse = (r["C0_cov_log_residual_logprice"] ** 2 / r["A_var_f0_log"]) * ((1.0 - q) ** 2) if np.isfinite(q) and r["A_var_f0_log"] > 0 else float("nan")
            cov_rp = q * r["cov_ratio_price_from_prd"] if np.isfinite(q) and np.isfinite(r["cov_ratio_price_from_prd"]) else float("nan")
            denom = 1.0 + cov_rp / (r["mu_ratio_price"] * r["mu_price"]) if np.isfinite(cov_rp) and np.isfinite(r["mu_ratio_price"] * r["mu_price"]) and r["mu_ratio_price"] * r["mu_price"] != 0 else float("nan")
            rows.append({
                "data_source": r["data_source"],
                "assessment_year": r["assessment_year"],
                "config_key": r["config_key"],
                "split": r["split"],
                "baseline_source": r["baseline_source"],
                "rho": rho,
                "q_remaining_covariance_theory": q,
                "covariance_reduction_theory": 1.0 - q if np.isfinite(q) else float("nan"),
                "predicted_delta_mse_log": d_mse,
                "predicted_delta_mse_log_frac_of_baseline": d_mse / r["B_mse_log"] if r["B_mse_log"] > 0 and np.isfinite(d_mse) else float("nan"),
                "predicted_mse_log": r["B_mse_log"] + d_mse if np.isfinite(d_mse) else float("nan"),
                "predicted_r2_log": 1.0 - (r["B_mse_log"] + d_mse) / r["var_y_log"] if r["var_y_log"] > 0 and np.isfinite(d_mse) else float("nan"),
                "predicted_cov_ratio_price": cov_rp,
                "predicted_PRD_ratio_price_bridge": 1.0 / denom if np.isfinite(denom) and denom != 0 else float("nan"),
                "inside_theory_range": bool(np.isfinite(r["theory_range_low"]) and np.isfinite(r["theory_range_high"]) and r["theory_range_low"] <= rho <= r["theory_range_high"]),
            })
    return pd.DataFrame(rows)


def empirical_comparison(summary: pd.DataFrame, empirical: pd.DataFrame) -> pd.DataFrame:
    if summary.empty or empirical.empty:
        return pd.DataFrame()
    key = ["data_source", "assessment_year", "config_key", "split"]
    cov = empirical[empirical["model_family"].astype(str).eq("LGBCovPenalty[diff]") & empirical["split"].isin(["assessment", "test"])].copy()
    base = empirical[empirical["model_name"].astype(str).eq("LGBMRegressor") & empirical["split"].isin(["assessment", "test"])].copy()
    if cov.empty or base.empty:
        return pd.DataFrame()
    theory = summary[key + ["A_var_f0_log", "B_mse_log", "C0_cov_log_residual_logprice", "theory_range_low", "theory_range_high"]].copy()
    base = base[key + ["R2", "RMSE", "MdAPE", "COD", "PRD", "PRB", "VEI"]].rename(columns={c: f"baseline_{c}" for c in ["R2", "RMSE", "MdAPE", "COD", "PRD", "PRB", "VEI"]})
    out = cov.merge(theory, on=key, how="inner").merge(base, on=key, how="left")
    out["rho"] = pd.to_numeric(out["rho"], errors="coerce")
    out["q_theory_remaining_covariance"] = [q_from_rho(r, a) for r, a in zip(out["rho"], out["A_var_f0_log"])]
    out["covariance_reduction_theory"] = 1.0 - out["q_theory_remaining_covariance"]
    out["delta_mse_log_frac_theory"] = ((out["C0_cov_log_residual_logprice"] ** 2 / out["A_var_f0_log"]) * ((1.0 - out["q_theory_remaining_covariance"]) ** 2) / out["B_mse_log"])
    out["empirical_R2_delta"] = pd.to_numeric(out["R2"], errors="coerce") - pd.to_numeric(out["baseline_R2"], errors="coerce")
    out["empirical_RMSE_frac_delta"] = pd.to_numeric(out["RMSE"], errors="coerce") / pd.to_numeric(out["baseline_RMSE"], errors="coerce") - 1.0
    out["empirical_MdAPE_frac_delta"] = pd.to_numeric(out["MdAPE"], errors="coerce") / pd.to_numeric(out["baseline_MdAPE"], errors="coerce") - 1.0
    out["empirical_COD_frac_delta"] = pd.to_numeric(out["COD"], errors="coerce") / pd.to_numeric(out["baseline_COD"], errors="coerce") - 1.0
    out["inside_theory_range"] = (out["rho"] >= out["theory_range_low"]) & (out["rho"] <= out["theory_range_high"])
    return out


def operating_points(empirical: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    if empirical.empty or summary.empty:
        return pd.DataFrame()
    key = ["data_source", "assessment_year", "config_key"]
    val = empirical[empirical["split"].eq("validation") & empirical["model_family"].astype(str).eq("LGBCovPenalty[diff]")].copy()
    theory = summary[summary["split"].eq("assessment")][key + ["theory_range_low", "theory_confident_rho", "theory_range_high"]]
    rows = []
    for g, d in val.groupby(key):
        choices = {
            "validation min COD": d["COD"].idxmin(),
            "validation min MdAPE": d["MdAPE"].idxmin(),
            "validation PRD closest": (d["PRD"] - 1.0).abs().idxmin(),
            "validation max R2": d["R2"].idxmax(),
        }
        for criterion, idx in choices.items():
            row = d.loc[idx]
            rows.append({"data_source": g[0], "assessment_year": int(g[1]), "config_key": g[2], "criterion": criterion, "selected_rho": float(row["rho"]), "validation_R2": row_float(row, "R2"), "validation_COD": row_float(row, "COD"), "validation_PRD": row_float(row, "PRD")})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.merge(theory, on=key, how="left")
    out["inside_theory_range"] = (out["selected_rho"] >= out["theory_range_low"]) & (out["selected_rho"] <= out["theory_range_high"])
    return out


def make_range_plot(ops: pd.DataFrame, path: Path) -> Optional[Path]:
    if ops.empty:
        return None
    base = ops.drop_duplicates(["data_source", "assessment_year", "config_key"]).sort_values(["assessment_year", "config_key", "data_source"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, max(4, 0.35 * len(base) + 1)))
    for i, row in base.iterrows():
        ax.hlines(i, row["theory_range_low"], row["theory_range_high"], color="#2563EB", lw=5, alpha=0.55)
        ax.scatter(row["theory_confident_rho"], i, color="#111827", marker="D", s=35)
    for criterion, d in ops.groupby("criterion"):
        y = []
        x = []
        for row in d.itertuples(index=False):
            idx = base.index[(base["data_source"].eq(row.data_source)) & (base["assessment_year"].eq(row.assessment_year)) & (base["config_key"].eq(row.config_key))]
            if len(idx):
                y.append(int(idx[0]))
                x.append(float(row.selected_rho))
        ax.scatter(x, y, s=28, label=criterion)
    ax.set_yticks(range(len(base)))
    ax.set_yticklabels([f"{int(r.assessment_year)} {r.config_key}\n{r.data_source}" for r in base.itertuples(index=False)], fontsize=8)
    ax.set_xlabel("rho")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def write_outputs(out_dir: Path, summary: pd.DataFrame, shrink: pd.DataFrame, prd: pd.DataFrame, budget: pd.DataFrame, args: argparse.Namespace) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    empirical = load_empirical(args.rho_sweep_root) if args.rho_sweep_root else pd.DataFrame()
    aggregate = aggregate_ranges(summary)
    horizon = build_horizon(summary, empirical, args)
    comp = empirical_comparison(summary, empirical)
    ops = operating_points(empirical, summary)
    paths = {
        "summary": out_dir / "theory_rho_summary_by_run.csv",
        "shrinkage": out_dir / "theory_rho_shrinkage_targets.csv",
        "prd": out_dir / "theory_rho_prd_targets.csv",
        "budget": out_dir / "theory_rho_accuracy_budgets.csv",
        "aggregate": out_dir / "theory_rho_aggregate_recommendation.csv",
        "horizon": out_dir / "theory_rho_horizon.csv",
        "comparison": out_dir / "theory_empirical_comparison.csv",
        "operating_points": out_dir / "theory_empirical_operating_points.csv",
        "report": out_dir / "theory_rho_report.md",
        "timing_detail": out_dir / "theory_rho_timing_detail.csv",
        "timing_summary": out_dir / "theory_rho_timing_summary.csv",
    }
    summary.to_csv(paths["summary"], index=False)
    shrink.to_csv(paths["shrinkage"], index=False)
    prd.to_csv(paths["prd"], index=False)
    budget.to_csv(paths["budget"], index=False)
    aggregate.to_csv(paths["aggregate"], index=False)
    horizon.to_csv(paths["horizon"], index=False)
    comp.to_csv(paths["comparison"], index=False)
    ops.to_csv(paths["operating_points"], index=False)
    plot = make_range_plot(ops, out_dir / "plots" / "theory_ranges_vs_empirical_operating_points.png")
    report = render_report(summary, aggregate, comp, ops, paths, plot, args)
    paths["report"].write_text(report, encoding="utf-8")
    pd.DataFrame(TIMING).to_csv(paths["timing_detail"], index=False)
    timing = pd.DataFrame(TIMING)
    if not timing.empty:
        timing.groupby("step")["seconds"].agg(calls="count", total_seconds="sum", mean_seconds="mean", max_seconds="max").reset_index().sort_values("total_seconds", ascending=False).to_csv(paths["timing_summary"], index=False)
    else:
        pd.DataFrame().to_csv(paths["timing_summary"], index=False)
    return paths


def fmt(x: Any, digits: int = 4) -> str:
    x = finite_float(x)
    return "NA" if not np.isfinite(x) else f"{x:.{digits}f}"


def render_report(summary: pd.DataFrame, aggregate: pd.DataFrame, comp: pd.DataFrame, ops: pd.DataFrame, paths: Dict[str, Path], plot: Optional[Path], args: argparse.Namespace) -> str:
    lines = ["# Theory-informed rho-range report", ""]
    if not aggregate.empty:
        allrow = aggregate[aggregate["split_group"].eq("all")].iloc[0]
        lines += [
            "## Main aggregate recommendation",
            "",
            f"- Robust theory range: **[{fmt(allrow['robust_theory_range_low'], 3)}, {fmt(allrow['robust_theory_range_high'], 3)}]**",
            f"- Median confident rho: **{fmt(allrow['median_confident_rho'], 3)}**",
            f"- Median rho for 25%, 50%, 67% covariance reduction: {fmt(allrow['median_rho_shrink_25pct'], 3)}, {fmt(allrow['median_rho_shrink_50pct'], 3)}, {fmt(allrow['median_rho_shrink_67pct'], 3)}",
            "",
            "## Aggregate table",
            "",
            aggregate.to_markdown(index=False, floatfmt=".4f"),
            "",
        ]
    show_cols = ["data_source", "assessment_year", "config_key", "split", "baseline_source", "n", "prd", "A_var_f0_log", "B_mse_log", "C0_cov_log_residual_logprice", "rho_shrink_50pct", "rho_prd_guidance", "rho_budget_1pct_mse", "theory_range_low", "theory_confident_rho", "theory_range_high"]
    lines += ["## Per-run summary", "", summary[[c for c in show_cols if c in summary.columns]].to_markdown(index=False, floatfmt=".4f"), ""]
    lines += [
        "## Interpretation notes",
        "",
        "- Exact saved predictions are used when `baseline_predictions_*.parquet` exists.",
        "- Older `rho_sweep_500_estimators` folders are used through the LGBM metric rows; this uses A ~= Var(Y)-MSE_log and C0 ~= -MSE_log, so the rho scale is theory-informed but approximate.",
        "- PRD targets use Cov(ratio, price) = E[ratio]E[price](1/PRD - 1) and assume proportional movement with the penalized log covariance.",
        "",
        "## Theory vs empirical sweep checks",
        "",
    ]
    if comp.empty:
        lines.append(f"No empirical sweep rows were matched from `{args.rho_sweep_root}`.")
    else:
        assess = comp[comp["split"].eq("assessment")]
        lines.append(f"- Matched empirical covariance-penalty rows: **{comp.shape[0]}**.")
        lines.append(f"- Share of assessment sweep rho values inside the per-run theory range: **{fmt(assess['inside_theory_range'].mean(), 3)}**.")
        if "empirical_R2_delta" in assess:
            corr = assess[["covariance_reduction_theory", "empirical_R2_delta"]].corr().iloc[0, 1]
            lines.append(f"- Corr(theory covariance reduction, empirical R2 delta): **{fmt(corr, 3)}**.")
    if not ops.empty:
        lines.append(f"- Validation-selected operating points inside theory range: **{fmt(ops['inside_theory_range'].mean(), 3)}**.")
        cols = ["data_source", "assessment_year", "config_key", "criterion", "selected_rho", "theory_range_low", "theory_confident_rho", "theory_range_high", "inside_theory_range"]
        lines += ["", ops[cols].to_markdown(index=False, floatfmt=".4f")]
    lines += ["", "Generated artifacts:"]
    for key in ["summary", "aggregate", "horizon", "comparison", "operating_points"]:
        lines.append(f"- `{os.path.relpath(paths[key], args.out_dir)}`")
    if plot is not None:
        lines.append(f"- `{os.path.relpath(plot, args.out_dir)}`")
    return "\n".join(lines) + "\n"


def run_analysis(args: argparse.Namespace) -> Dict[str, Path]:
    out_dir = Path(args.out_dir)
    with open(args.params_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    configs = parse_csv(args.lgbm_config_keys, str)
    shrink_qs = parse_csv(args.shrinkage_q_values, float)
    prd_targets = parse_csv(args.prd_targets, float)
    budgets = parse_csv(args.accuracy_budgets, float)
    empirical_parts = parse_csv(args.empirical_rho_range, float) if args.empirical_rho_range else []
    empirical_range = (empirical_parts[0], empirical_parts[1]) if len(empirical_parts) == 2 else None
    summaries: List[Dict[str, Any]] = []
    shrinks: List[pd.DataFrame] = []
    prds: List[pd.DataFrame] = []
    budgets_all: List[pd.DataFrame] = []
    for source, year, data_path in analysis_specs(args):
        target_cache: Optional[Dict[str, Optional[np.ndarray]]] = None
        for config in configs:
            folder = find_sweep_folder(args.rho_sweep_root, source, year, config)
            for split in ("test", "assessment"):
                pred = read_predictions(folder / f"baseline_predictions_{SPLIT_SUFFIX[split]}.parquet") if folder else None
                common = dict(source=source, year=year, config_key=config, split=split, prd_targets=prd_targets, shrink_qs=shrink_qs, budgets=budgets, empirical_range=empirical_range)
                if pred is not None:
                    with timed("theory: exact sweep predictions", source=source, year=year, config=config, split=split):
                        summary, shrink, prd, budget = theory_from_predictions(pred[0], pred[1], baseline_source="rho_sweep_predictions", **common)
                else:
                    row = read_lgbm_metric_row(folder, split)
                    if row is None:
                        if args.no_fit_baseline:
                            raise RuntimeError(f"Missing sweep baseline predictions and metrics for {source}/{year}/{config}/{split}")
                        raise RuntimeError("Fallback LGBM fitting is intentionally disabled in this cluster workflow. Re-run the rho sweep or use existing metrics.")
                    if target_cache is None:
                        target_cache = load_target_splits(data_path, args.target_column, args.date_column, args.sample_frac, args.seed, year, float(params["cv"]["split_prop"]))
                    y = target_cache.get(split)
                    if y is None:
                        continue
                    with timed("theory: sweep metric baseline", source=source, year=year, config=config, split=split):
                        summary, shrink, prd, budget = theory_from_metric_baseline(y, row, baseline_source="rho_sweep_metrics_bayes_approx", **common)
                summaries.append(summary)
                shrinks.append(shrink)
                prds.append(prd)
                budgets_all.append(budget)
    summary_df = pd.DataFrame(summaries)
    shrink_df = pd.concat(shrinks, ignore_index=True) if shrinks else pd.DataFrame()
    prd_df = pd.concat(prds, ignore_index=True) if prds else pd.DataFrame()
    budget_df = pd.concat(budgets_all, ignore_index=True) if budgets_all else pd.DataFrame()
    return write_outputs(out_dir, summary_df, shrink_df, prd_df, budget_df, args)


def aggregate_existing(args: argparse.Namespace) -> Dict[str, Path]:
    root = Path(args.aggregate_input_root)

    def cat(name: str) -> pd.DataFrame:
        frames = [pd.read_csv(p) for p in sorted(root.glob(f"*/{name}"))]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    summary = cat("theory_rho_summary_by_run.csv")
    if summary.empty:
        raise RuntimeError(f"No theory_rho_summary_by_run.csv files under {root}")
    return write_outputs(Path(args.out_dir), summary, cat("theory_rho_shrinkage_targets.csv"), cat("theory_rho_prd_targets.csv"), cat("theory_rho_accuracy_budgets.csv"), args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compute theory-informed rho ranges for LGBCovPenalty[diff].")
    p.add_argument("--data-path", default="./data/CCAO/2025/training_data.parquet")
    p.add_argument("--data-source-label", default="")
    p.add_argument("--data-source-specs", default=default_specs())
    p.add_argument("--params-path", default="params.yaml")
    p.add_argument("--model-params-path", default="model_params.yaml")
    p.add_argument("--lgbm-hyperparameter-file", default="best_lgbm_baseline_configs.yaml")
    p.add_argument("--lgbm-config-keys", default="cv_top1_r2,test_best_r2,cv_top2_r2")
    p.add_argument("--lgbm-n-jobs", type=int, default=None)
    p.add_argument("--lgbm-n-estimators", type=int, default=None)
    p.add_argument("--assessment-years", default="2022,2023,2024,2025")
    p.add_argument("--target-column", default="meta_sale_price")
    p.add_argument("--date-column", default="meta_sale_date")
    p.add_argument("--sample-frac", type=float, default=None)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out-dir", default="output/theory_rho_range")
    p.add_argument("--baseline-cache-dir", default="")
    p.add_argument("--rho-sweep-root", default="output/rho_sweep_500_estimators")
    p.add_argument("--shrinkage-q-values", default="0.75,0.50,0.33,0.25")
    p.add_argument("--prd-targets", default="1.03,1.02,1.01,0.99,0.98")
    p.add_argument("--accuracy-budgets", default="0.001,0.005,0.01,0.02")
    p.add_argument("--empirical-rho-range", default="2.56,3.54")
    p.add_argument("--horizon-rho-min", type=float, default=0.1)
    p.add_argument("--horizon-rho-max", type=float, default=20.0)
    p.add_argument("--horizon-rho-count", type=int, default=200)
    p.add_argument("--horizon-rho-scale", default="log")
    p.add_argument("--no-fit-baseline", action="store_true")
    p.add_argument("--aggregate-input-root", default="")
    return p


def main() -> None:
    args = build_parser().parse_args()
    start = time.perf_counter()
    status = "ok"
    try:
        paths = aggregate_existing(args) if args.aggregate_input_root else run_analysis(args)
    except Exception:
        status = "failed"
        raise
    finally:
        TIMING.append({"step": "run: total", "seconds": time.perf_counter() - start, "status": status})
    for key, path in paths.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
