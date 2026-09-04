#!/usr/bin/env python3
"""Step 7 (Surrogate): normalized-rho CV path over the frozen seven folds,
using the clean first-branch calibration protocol.

Reuses (read-only import, no modification) the noise-floor-adaptive
first_branch_calibrate from analysis/berry_attom_validation_v3/scripts/v3_common.py
-- the "new clean first-branch protocol already frozen for this benchmark"
per the user's instruction: a branch opens only once achieved first-order
reduction clears a noise floor estimated from the curve's own inactive tail,
and closes only on a material reversal. Never globally sorts rho by achieved
reduction; never jumps to a later high-rho branch.

raw_rho = rho_tilde / Var_training_fold(y). Canonical objective only:
ratio_mode="diff", weighting_proxy_mode="identity", match_native_init=True.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMRegressor

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from soft_constrained_models.boosting_models import LGBSmoothPenalty  # noqa: E402
from scripts.other_counties_benchmars import score_predictions  # noqa: E402
from analysis.berry_attom_validation_v3.scripts.v3_common import first_branch_calibrate  # noqa: E402
from analysis.external_jurisdiction_benchmark_v1.scripts.run_baseline_cv import (  # noqa: E402
    build_folds, enrich, v1_features,
)
from analysis.external_jurisdiction_benchmark_v1.scripts.v1_common import (  # noqa: E402
    ANALYSIS, LGBM_CONFIG_PATH, OUTPUT, SURROGATE_CANONICAL_KWARGS, population_variance,
    normalized_rho_tilde_grid, write_json,
)

FORWARD_LOCK_DATE = pd.Timestamp("2025-01-01")


def first_order_reduction(base_cov: float, new_cov: float) -> float:
    if not np.isfinite(base_cov) or abs(base_cov) < 1e-18:
        return float("nan")
    return float(1.0 - new_cov / base_cov)


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--county-key", required=True)
    parser.add_argument("--lgbm-threads", type=int, default=8)
    parser.add_argument("--extra-decades", type=float, default=0.0)
    args = parser.parse_args()
    key = args.county_key

    config_path = ANALYSIS / "baseline" / f"{key}_baseline_config.json"
    table_path = OUTPUT / "modeling_tables" / key / "history_market_core_dev.parquet"
    if not (config_path.exists() and table_path.exists()):
        print(json.dumps({"county_key": key, "path": "surrogate", "skipped": True, "reason": "no baseline/table"}))
        return 0
    cfg = json.loads(config_path.read_text())
    grid = yaml.safe_load(LGBM_CONFIG_PATH.read_text())["lgbm_grid"]
    params = dict(grid[cfg["selected_lgbm_config"]])
    params["n_jobs"] = args.lgbm_threads

    data = pd.read_parquet(table_path).sort_values("sale_date").reset_index(drop=True)
    data["sale_date"] = pd.to_datetime(data["sale_date"])
    assert data["sale_date"].max() < FORWARD_LOCK_DATE, f"{key}: table contains >=2025 data"

    rho_tilde_grid = normalized_rho_tilde_grid(include_zero=False, extra_decades=args.extra_decades)
    full_rows, branch_rows = [], []
    for fold_idx, (dev_start, train_end, val_end) in enumerate(build_folds(data["sale_date"]), start=1):
        train_mask = data["sale_date"] < train_end
        val_mask = (data["sale_date"] >= train_end) & (data["sale_date"] < val_end)
        train = data.loc[train_mask].reset_index(drop=True)
        val = data.loc[val_mask].reset_index(drop=True)
        if len(train) < 500 or len(val) < 50:
            continue
        assert val["sale_date"].max() < FORWARD_LOCK_DATE
        assert train["sale_date"].max() <= val["sale_date"].min()

        combined = pd.concat([train, val], ignore_index=True)
        features, cats = v1_features(combined, len(train), True)
        y_log_train = np.log(train["sale_price"].astype(float))
        vy_train = population_variance(y_log_train)
        train_price = train["sale_price"].to_numpy()
        val_price = val["sale_price"].to_numpy()

        base = LGBMRegressor(**params)
        base.fit(features.iloc[:len(train)], y_log_train, categorical_feature=cats)
        base_pred = np.exp(base.predict(features.iloc[len(train):]))
        base_cov = score_predictions(val_price, base_pred, train_price).get("Cov(e,logprice)", np.nan)

        achieved, fold_rows = [], []
        for rho_tilde in rho_tilde_grid:
            raw_rho = float(rho_tilde) / vy_train if vy_train > 0 else float("nan")
            try:
                model = LGBSmoothPenalty(
                    rho=raw_rho, early_stopping_rounds=None, lgbm_params=dict(params),
                    verbose=False, **SURROGATE_CANONICAL_KWARGS,
                )
                model.fit(features.iloc[:len(train)], y_log_train)  # LGBSmoothPenalty.fit(X,y) has no categorical_feature kwarg; category dtype is auto-detected
                pred_price = np.exp(model.predict(features.iloc[len(train):]))
                metrics = enrich(val_price, pred_price, train_price)
                red = first_order_reduction(base_cov, metrics.get("Cov(e,logprice)", np.nan))
            except Exception as exc:
                metrics, red = {"fit_error": str(exc)}, float("nan")
            achieved.append(red)
            fold_rows.append({
                "county_key": key, "fold": fold_idx, "validation_year": train_end.year,
                "rho_tilde": float(rho_tilde), "raw_rho": raw_rho, "Var_training_y": vy_train,
                "base_cov_val": float(base_cov), "achieved_first_order_reduction": red,
                "n_train": len(train), "n_val": len(val), **metrics,
            })
        full_rows.extend(fold_rows)

        frozen, branch = first_branch_calibrate(rho_tilde_grid, np.array(achieved, dtype=float))
        frozen["county_key"] = key
        frozen["fold"] = fold_idx
        frozen["validation_year"] = train_end.year
        branch_rows.append(frozen)
        print(f"{key} surrogate fold {fold_idx} ({train_end.year}) done, branch_n={len(branch)}, "
              f"terminated_by={frozen['branch_terminated_by'].iloc[0] if len(frozen) else 'n/a'}", flush=True)

    full_df = pd.DataFrame(full_rows)
    branch_df = pd.concat(branch_rows, ignore_index=True) if branch_rows else pd.DataFrame()
    ANALYSIS.joinpath("cv").mkdir(parents=True, exist_ok=True)
    OUTPUT.joinpath("cv").mkdir(parents=True, exist_ok=True)
    full_df.to_parquet(OUTPUT / "cv" / f"{key}_surrogate_normalized_cv_path.parquet", index=False)
    full_df.to_csv(ANALYSIS / "cv" / f"{key}_surrogate_normalized_cv_path_summary.csv", index=False)
    branch_df.to_csv(ANALYSIS / "cv" / f"{key}_surrogate_first_branch_by_fold.csv", index=False)
    write_json(ANALYSIS / "cv" / f"{key}_surrogate_cv_meta.json", {
        "county_key": key, "family": "surrogate", "n_folds": int(full_df["fold"].nunique()) if len(full_df) else 0,
        "n_rho_points": len(rho_tilde_grid), "selected_lgbm_config": cfg["selected_lgbm_config"],
        "rho_tilde_min": float(rho_tilde_grid.min()), "rho_tilde_max": float(rho_tilde_grid.max()),
        "calibration_method": "first_branch_calibrate (noise-floor-adaptive), reused read-only from "
                               "analysis/berry_attom_validation_v3/scripts/v3_common.py",
        "no_2025_data_used": True, "never_globally_sorted_by_achieved_reduction": True,
    })
    print(json.dumps({"county_key": key, "path": "surrogate", "n_rows": len(full_df)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
