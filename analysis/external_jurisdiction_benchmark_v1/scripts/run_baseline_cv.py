#!/usr/bin/env python3
"""Step 5: shared 12-config LightGBM baseline search on the frozen seven-fold
expanding calendar-year CV (temporal_design.yaml), HISTORY_MARKET_CORE.

HARD GUARD: reads only history_market_core_dev.parquet (built with
--end-date 2024-12-31, never containing 2025 rows), and additionally asserts
every fold's validation dates are < 2025-01-01 before scoring anything.

Selection rule (frozen in baseline/shared_lgbm_grid.yaml): the config with the
lowest mean CV RMSE_log wins per jurisdiction. No penalty path is run here.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMRegressor

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.other_counties_benchmars import feature_frame, score_predictions  # noqa: E402
from utils.delta_nl import estimate_delta_nl  # noqa: E402
from utils.motivation_utils import paper_mechanism_metrics  # noqa: E402
from analysis.external_jurisdiction_benchmark_v1.scripts.v1_common import (  # noqa: E402
    ANALYSIS, ASSESSMENT_VALUE_COLUMNS, JURISDICTION_BY_KEY, LGBM_CONFIG_PATH, OUTPUT, PILOT_KEYS,
    population_variance, write_json,
)

CV_VALIDATION_YEARS = (2018, 2019, 2020, 2021, 2022, 2023, 2024)
FORWARD_LOCK_DATE = pd.Timestamp("2025-01-01")
ACS_PREFIXES = ("acs_",)


def v1_features(data: pd.DataFrame, train_rows: int, include_prior: bool) -> tuple[pd.DataFrame, list[str]]:
    feats, cats = feature_frame(data, train_rows, include_prior, False)
    drop = [c for c in feats.columns if c.startswith(ACS_PREFIXES) or c in ASSESSMENT_VALUE_COLUMNS]
    feats = feats.drop(columns=drop, errors="ignore")
    overlap = set(feats.columns) & ASSESSMENT_VALUE_COLUMNS
    if overlap:
        raise RuntimeError(f"assessment-value predictors leaked into features: {sorted(overlap)}")
    cats = [c for c in cats if c in feats.columns]
    return feats, cats


def nmse(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    """NMSE_S = MSE_log,S / Var_S(y), ddof=0, on the EVALUATION sample S.
    Never confused with the training-block Vy_T used for rho_tilde (that
    normalization is not used anywhere in this baseline-selection script)."""
    mse = float(np.mean((y_pred_log - y_true_log) ** 2))
    var_s = population_variance(y_true_log)
    return mse / var_s if var_s > 0 else float("nan")


def enrich(actual_price: np.ndarray, predicted_price: np.ndarray, train_price: np.ndarray) -> dict:
    m = score_predictions(actual_price, predicted_price, train_price)
    y_true_log = np.log(actual_price)
    y_pred_log = np.log(predicted_price)
    m.update(paper_mechanism_metrics(y_true_log, y_pred_log))
    m["NMSE"] = nmse(y_true_log, y_pred_log)
    try:
        m["Delta_NL"] = float(estimate_delta_nl(y_true_log, y_pred_log, row_ids=np.arange(len(actual_price))).get("Delta_NL", np.nan))
    except Exception as exc:
        m["Delta_NL"] = np.nan
        m["Delta_NL_error"] = str(exc)
    return {k: v for k, v in m.items() if k != "ratio_by_decile"}


def load_configs() -> dict:
    d = yaml.safe_load(LGBM_CONFIG_PATH.read_text())
    return d["lgbm_grid"]


def build_folds(dates: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """fold t: train < Jan1(t); validate in [Jan1(t), Jan1(t+1))."""
    folds = []
    for t in CV_VALIDATION_YEARS:
        train_end = pd.Timestamp(f"{t}-01-01")
        val_end = pd.Timestamp(f"{t + 1}-01-01")
        folds.append((pd.Timestamp("2016-01-01"), train_end, val_end))
    return folds


def run_county(key: str, threads: int) -> dict:
    table_path = OUTPUT / "modeling_tables" / key / "history_market_core_dev.parquet"
    if not table_path.exists():
        return {"county_key": key, "status": "SKIPPED_NO_TABLE"}
    data = pd.read_parquet(table_path).sort_values("sale_date").reset_index(drop=True)
    data["sale_date"] = pd.to_datetime(data["sale_date"])
    assert data["sale_date"].max() < FORWARD_LOCK_DATE, (
        f"{key}: modeling table contains sale_date >= 2025-01-01; refusing to run CV"
    )
    configs = load_configs()

    fold_rows = []
    for fold_idx, (dev_start, train_end, val_end) in enumerate(build_folds(data["sale_date"]), start=1):
        train_mask = data["sale_date"] < train_end
        val_mask = (data["sale_date"] >= train_end) & (data["sale_date"] < val_end)
        train = data.loc[train_mask].reset_index(drop=True)
        val = data.loc[val_mask].reset_index(drop=True)
        if len(train) < 500 or len(val) < 50:
            print(f"skip fold {fold_idx} for {key}: n_train={len(train)} n_val={len(val)} too small", flush=True)
            continue
        assert val["sale_date"].max() < FORWARD_LOCK_DATE
        assert train["sale_date"].max() <= val["sale_date"].min()

        combined = pd.concat([train, val], ignore_index=True)
        features, cats = v1_features(combined, len(train), True)
        y_log_all = np.log(combined["sale_price"].astype(float))
        train_price = train["sale_price"].to_numpy()
        val_price = val["sale_price"].to_numpy()

        for config_name, params in configs.items():
            lgbm_params = dict(params)
            lgbm_params["n_jobs"] = threads
            model = LGBMRegressor(**lgbm_params)
            model.fit(features.iloc[:len(train)], y_log_all.iloc[:len(train)], categorical_feature=cats)
            pred_price = np.exp(model.predict(features.iloc[len(train):]))
            metrics = enrich(val_price, pred_price, train_price)
            fold_rows.append({
                "county_key": key, "fold": fold_idx, "validation_year": CV_VALIDATION_YEARS[fold_idx - 1],
                "config_name": config_name, "n_train": len(train), "n_val": len(val), **metrics,
            })
        print(f"{key} fold {fold_idx} ({CV_VALIDATION_YEARS[fold_idx-1]}): n_train={len(train)} n_val={len(val)} done", flush=True)

    fold_df = pd.DataFrame(fold_rows)
    if not len(fold_df):
        return {"county_key": key, "status": "NO_FOLDS_RAN"}

    ANALYSIS.joinpath("baseline").mkdir(parents=True, exist_ok=True)
    fold_df.to_csv(ANALYSIS / "baseline" / f"{key}_baseline_cv_folds.csv", index=False)

    summary = fold_df.groupby("config_name").agg(
        n_folds=("fold", "nunique"), mean_RMSE_log=("RMSE_log", "mean"), std_RMSE_log=("RMSE_log", "std"),
        mean_R2_price=("R2_price", "mean"), mean_R2_log=("R2_log", "mean"), mean_NMSE=("NMSE", "mean"),
        mean_COD=("COD", "mean"), mean_PRD=("PRD", "mean"), mean_PRB=("PRB", "mean"), mean_MKI=("MKI", "mean"),
        mean_VEI=("VEI", "mean"), mean_Beta_log=("Beta_log", "mean"), mean_Delta_NL=("Delta_NL", "mean"),
        mean_dCor_e_y=("dCor_e_y", "mean"), mean_MAE=("MAE", "mean"), mean_MAPE=("MAPE", "mean"),
    ).reset_index().sort_values("mean_RMSE_log")
    summary.to_csv(ANALYSIS / "baseline" / f"{key}_baseline_cv_summary.csv", index=False)

    selected_name = str(summary.iloc[0]["config_name"])
    selected_row = summary.iloc[0].to_dict()

    # HISTORY_STRUCTURAL_CORE secondary sensitivity, selected config only, all folds.
    structural_rows = []
    for fold_idx, (dev_start, train_end, val_end) in enumerate(build_folds(data["sale_date"]), start=1):
        train_mask = data["sale_date"] < train_end
        val_mask = (data["sale_date"] >= train_end) & (data["sale_date"] < val_end)
        train = data.loc[train_mask].reset_index(drop=True)
        val = data.loc[val_mask].reset_index(drop=True)
        if len(train) < 500 or len(val) < 50:
            continue
        combined = pd.concat([train, val], ignore_index=True)
        struct_features, struct_cats = v1_features(combined, len(train), False)
        y_log_all = np.log(combined["sale_price"].astype(float))
        lgbm_params = dict(configs[selected_name]); lgbm_params["n_jobs"] = threads
        model = LGBMRegressor(**lgbm_params)
        model.fit(struct_features.iloc[:len(train)], y_log_all.iloc[:len(train)], categorical_feature=struct_cats)
        pred_price = np.exp(model.predict(struct_features.iloc[len(train):]))
        metrics = enrich(val["sale_price"].to_numpy(), pred_price, train["sale_price"].to_numpy())
        structural_rows.append({"county_key": key, "fold": fold_idx, **metrics})
    structural_df = pd.DataFrame(structural_rows)
    if len(structural_df):
        structural_df.to_csv(ANALYSIS / "baseline" / f"{key}_structural_core_cv_folds.csv", index=False)

    result = {
        "county_key": key, "status": "OK", "selected_lgbm_config": selected_name,
        "n_folds_used": int(fold_df["fold"].nunique()),
        "cv_mean_RMSE_log": selected_row["mean_RMSE_log"], "cv_mean_R2_price": selected_row["mean_R2_price"],
        "cv_mean_NMSE": selected_row["mean_NMSE"], "cv_mean_COD": selected_row["mean_COD"],
        "cv_mean_PRD": selected_row["mean_PRD"], "cv_mean_PRB": selected_row["mean_PRB"],
        "cv_mean_MKI": selected_row["mean_MKI"], "cv_mean_VEI": selected_row["mean_VEI"],
        "cv_mean_Beta_log": selected_row["mean_Beta_log"], "cv_mean_Delta_NL": selected_row["mean_Delta_NL"],
        "structural_core_cv_mean_RMSE_log": float(structural_df["RMSE_log"].mean()) if len(structural_df) else None,
        "structural_core_cv_mean_R2_price": float(structural_df["R2_price"].mean()) if len(structural_df) else None,
        "selection_rule": "lowest mean CV RMSE_log across 7 expanding calendar-year folds",
        "written_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "no_2025_data_used": True, "no_penalty_path_run": True,
    }
    write_json(ANALYSIS / "baseline" / f"{key}_baseline_config.json", result)
    print(json.dumps({k: result[k] for k in ["county_key", "selected_lgbm_config", "cv_mean_R2_price", "cv_mean_RMSE_log"]}, default=str), flush=True)
    return result


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--county-key", required=True)
    parser.add_argument("--lgbm-threads", type=int, default=8)
    args = parser.parse_args()
    run_county(args.county_key, args.lgbm_threads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
