#!/usr/bin/env python3
"""Steps 15-17: Direct (LGBCovPenalty[diff]) and Surrogate (LGBSmoothPenalty).

Runs ONLY if panel_freeze/final_panel_freeze_v2.yaml authorizes the transfer
experiment. County-specific rho is calibrated on development/validation only.
Plots use achieved mechanism reduction, not raw rho.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMRegressor

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.other_counties_benchmars import (  # noqa: E402
    bootstrap_scores, feature_frame, load_lgbm_configs, plan_rho_grid, score_predictions,
)
from soft_constrained_models.boosting_models import LGBCovPenalty, LGBSmoothPenalty
from utils.delta_nl import estimate_delta_nl
from utils.motivation_utils import paper_mechanism_metrics
from analysis.berry_attom_validation_v2.scripts.v2_common import (  # noqa: E402
    ANALYSIS, ASSESSMENT_VALUE_COLUMNS, LGBM_CONFIG_PATH, N_BOOTSTRAP, OUTPUT, SEED,
    TEST_FRACTION, VALIDATION_FRACTION,
)

TARGETS = (0.10, 0.25, 0.50, 0.67, 0.80, 0.90, 0.97)
FREEZE = ANALYSIS / "panel_freeze" / "final_panel_freeze_v2.yaml"


def enrich(actual, predicted, train_actual):
    m = score_predictions(actual, predicted, train_actual)
    m.update(paper_mechanism_metrics(np.log(actual), np.log(predicted)))
    try:
        m["Delta_NL"] = float(estimate_delta_nl(np.log(actual), np.log(predicted), row_ids=np.arange(len(actual))).get("Delta_NL", np.nan))
    except Exception as exc:
        m["Delta_NL"] = np.nan
        m["Delta_NL_error"] = str(exc)
    return m


def first_order_reduction(base_cov: float, new_cov: float) -> float:
    if not np.isfinite(base_cov) or abs(base_cov) < 1e-18:
        return float("nan")
    return float(1.0 - new_cov / base_cov)


def fit_direct(features, y_log, split, rho, params):
    model = LGBCovPenalty(
        rho=float(rho), ratio_mode="diff", early_stopping_rounds=None,
        zero_grad_tol=1e-12, lgbm_params=params, verbose=False,
    )
    model.fit(features.iloc[:split], y_log.iloc[:split])
    return np.exp(model.predict(features))


def fit_surrogate(features, y_log, split, rho, params):
    model = LGBSmoothPenalty(
        rho=float(rho), ratio_mode="diff", early_stopping_rounds=None,
        lgbm_params=params, verbose=False,
    )
    model.fit(features.iloc[:split], y_log.iloc[:split])
    return np.exp(model.predict(features))


def calibrate_surrogate_rho(features, y_log, data, development, val_end, params, baseline_cov):
    """Coarse log-rho bracket on validation; interpolate to portable targets; freeze before test."""
    grid = np.geomspace(1e-6, 1e2, 12)
    achieved = []
    for rho in grid:
        try:
            pred = fit_surrogate(features.iloc[:val_end], y_log.iloc[:val_end], development, rho, params)
        except Exception:
            achieved.append((float(rho), np.nan))
            continue
        mets = score_predictions(
            data.sale_price.iloc[development:val_end], pred[development:], data.sale_price.iloc[:development],
        )
        cov = mets.get("Cov(e,logprice)", np.nan)
        achieved.append((float(rho), first_order_reduction(baseline_cov, cov)))
    frame = pd.DataFrame(achieved, columns=["rho", "achieved_reduction"]).dropna()
    frozen = []
    for target in TARGETS:
        if frame.empty or not np.isfinite(frame["achieved_reduction"]).any():
            frozen.append({"requested_reduction": target, "rho": np.nan, "status": "unattained"})
            continue
        # interpolate rho onto achieved reduction where monotone enough
        work = frame.sort_values("achieved_reduction")
        if target < work["achieved_reduction"].min() or target > work["achieved_reduction"].max():
            nearest = work.iloc[(work["achieved_reduction"] - target).abs().argmin()]
            if abs(nearest["achieved_reduction"] - target) > 0.15:
                frozen.append({"requested_reduction": target, "rho": np.nan, "status": "unattained"})
                continue
            frozen.append({
                "requested_reduction": target, "rho": float(nearest["rho"]),
                "status": "nearest_on_validation",
            })
            continue
        rho = float(np.interp(target, work["achieved_reduction"], work["rho"]))
        frozen.append({"requested_reduction": target, "rho": rho, "status": "interpolated_on_validation"})
    return pd.DataFrame(frozen), frame


def run_county(key: str, fips: str, config_name: str, threads: int) -> pd.DataFrame:
    data = pd.read_parquet(OUTPUT / "modeling_tables" / key / "history_market_core.parquet")
    data = data.sort_values("sale_date").reset_index(drop=True)
    n = len(data)
    split = int(n * (1 - TEST_FRACTION))
    development = int(split * (1 - VALIDATION_FRACTION))
    y_log = np.log(data["sale_price"].astype(float))
    configs = load_lgbm_configs(LGBM_CONFIG_PATH, config_name, threads)
    params = configs[config_name]
    features, categorical = feature_frame(data, split, True, False)
    if set(features.columns) & ASSESSMENT_VALUE_COLUMNS:
        raise RuntimeError("assessment-value predictors in Direct/Surrogate features")
    # Baseline refit on full pre-test (frozen from Step 11 selected config).
    base = LGBMRegressor(**params)
    base.fit(features.iloc[:split], y_log.iloc[:split], categorical_feature=categorical)
    base_pred = np.exp(base.predict(features))
    train_price = data.sale_price.iloc[:split].to_numpy()
    test = data.iloc[split:]
    base_train_m = score_predictions(data.sale_price.iloc[:development], base_pred[:development], train_price[:development])
    plan, theory, _ = plan_rho_grid(
        y_log.to_numpy()[:development], np.log(base_pred[:development]),
        county_fips=fips, config_key=config_name, shrinkage_targets=TARGETS, include_anchors=False,
    )
    # Surrogate calibration uses a development-only baseline covariance.
    base_dev = LGBMRegressor(**params)
    base_dev.fit(features.iloc[:development], y_log.iloc[:development], categorical_feature=categorical)
    base_dev_pred = np.exp(base_dev.predict(features.iloc[:split]))
    base_val_m = score_predictions(
        data.sale_price.iloc[development:split], base_dev_pred[development:], data.sale_price.iloc[:development],
    )
    surr_plan, surr_curve = calibrate_surrogate_rho(
        features, y_log, data, development, split, params, base_val_m.get("Cov(e,logprice)", np.nan),
    )
    surr_plan.to_csv(OUTPUT / "method_transfer" / key / "surrogate_rho_frozen_before_test.csv", index=False)
    plan.to_csv(OUTPUT / "method_transfer" / key / "direct_rho_plan.csv", index=False)

    rows = []
    base_test = enrich(data.sale_price.iloc[split:].to_numpy(), base_pred[split:], train_price)
    base_cov = base_test.get("Cov(e,logprice)", np.nan)
    rows.append({
        "county_key": key, "path": "baseline", "requested_first_order_reduction": 0.0,
        "achieved_first_order_reduction": 0.0, "rho": 0.0, "status": "baseline",
        **{k: v for k, v in base_test.items() if not isinstance(v, dict)},
    })
    for row in plan.itertuples(index=False):
        try:
            pred = fit_direct(features, y_log, split, float(row.rho), params)
            mets = enrich(data.sale_price.iloc[split:].to_numpy(), pred[split:], train_price)
            rows.append({
                "county_key": key, "path": "direct",
                "requested_first_order_reduction": float(row.requested_covariance_reduction),
                "achieved_first_order_reduction": first_order_reduction(base_cov, mets.get("Cov(e,logprice)", np.nan)),
                "rho": float(row.rho), "status": "ok",
                **{k: v for k, v in mets.items() if not isinstance(v, dict)},
            })
        except Exception as exc:
            rows.append({
                "county_key": key, "path": "direct",
                "requested_first_order_reduction": float(row.requested_covariance_reduction),
                "rho": float(row.rho), "status": f"unstable:{exc}",
            })
    for row in surr_plan.itertuples(index=False):
        if not np.isfinite(row.rho):
            rows.append({
                "county_key": key, "path": "surrogate",
                "requested_first_order_reduction": float(row.requested_reduction),
                "rho": np.nan, "status": row.status,
            })
            continue
        try:
            pred = fit_surrogate(features, y_log, split, float(row.rho), params)
            mets = enrich(data.sale_price.iloc[split:].to_numpy(), pred[split:], train_price)
            rows.append({
                "county_key": key, "path": "surrogate",
                "requested_first_order_reduction": float(row.requested_reduction),
                "achieved_first_order_reduction": first_order_reduction(base_cov, mets.get("Cov(e,logprice)", np.nan)),
                "rho": float(row.rho), "status": row.status,
                **{k: v for k, v in mets.items() if not isinstance(v, dict)},
            })
        except Exception as exc:
            rows.append({
                "county_key": key, "path": "surrogate",
                "requested_first_order_reduction": float(row.requested_reduction),
                "rho": float(row.rho), "status": f"unstable:{exc}",
            })
    return pd.DataFrame(rows)


def plot_paths(metrics: pd.DataFrame) -> None:
    fig_dir = ANALYSIS / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ycol, fname, title in (
        ("R2", "cross_jurisdiction_accuracy_vs_mechanism.pdf", "R2_price vs achieved first-order reduction"),
        ("PRB", "cross_jurisdiction_prb_vs_mechanism.pdf", "PRB vs achieved first-order reduction"),
        ("Delta_NL", "cross_jurisdiction_delta_nl_vs_mechanism.pdf", "Delta_NL vs achieved first-order reduction"),
    ):
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        for (county, path), g in metrics.groupby(["county_key", "path"]):
            g = g.dropna(subset=["achieved_first_order_reduction", ycol])
            if g.empty:
                continue
            ax.plot(g["achieved_first_order_reduction"], g[ycol], marker="o", label=f"{county} {path}")
        ax.set_xlabel("Achieved first-order / covariance reduction (test)")
        ax.set_ylabel(ycol)
        ax.set_title(title)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / fname)
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lgbm-threads", type=int, default=8)
    args = parser.parse_args()
    if not FREEZE.exists():
        print("STOP: freeze file missing; Direct/Surrogate not run.", flush=True)
        return 0
    freeze = yaml.safe_load(FREEZE.read_text())
    if not freeze.get("direct_surrogate_authorized"):
        print("STOP: freeze did not authorize Direct/Surrogate.", flush=True)
        return 0
    passing = freeze.get("passing_units") or []
    out_root = OUTPUT / "method_transfer"
    out_root.mkdir(parents=True, exist_ok=True)
    frames = []
    for unit in freeze.get("units", []):
        key = unit["jurisdiction_key"]
        if key not in passing:
            continue
        meta = json.loads((ANALYSIS / "baselines" / key / "run_meta.json").read_text())
        config_name = meta["selected_lgbm_config"]
        (out_root / key).mkdir(parents=True, exist_ok=True)
        print("path", key, config_name, flush=True)
        frames.append(run_county(key, str(unit["attom_fips"]), config_name, args.lgbm_threads))
    if not frames:
        print("no passing units executed", flush=True)
        return 0
    metrics = pd.concat(frames, ignore_index=True)
    compact = ANALYSIS / "method_transfer"
    compact.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(compact / "all_metrics.csv", index=False)
    metrics.to_csv(out_root / "all_metrics.csv", index=False)
    path_summary = metrics.groupby(["county_key", "path", "status"], dropna=False).size().reset_index(name="n")
    path_summary.to_csv(compact / "path_summary.csv", index=False)
    plot_paths(metrics)
    print("wrote method_transfer", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
