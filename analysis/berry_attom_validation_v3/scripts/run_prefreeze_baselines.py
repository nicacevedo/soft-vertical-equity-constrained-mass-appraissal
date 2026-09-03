#!/usr/bin/env python3
"""Pre-freeze LR + ordinary LightGBM on DEVELOPMENT and VALIDATION only.

NEVER scores, stores, or even predicts the chronological held-out test block.
Categorical levels for LR are determined from the development prefix only.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.other_counties_benchmars import feature_frame, load_lgbm_configs, score_predictions  # noqa: E402
from utils.delta_nl import estimate_delta_nl
from utils.motivation_utils import paper_mechanism_metrics
from analysis.berry_attom_validation_v3.scripts.v3_common import (  # noqa: E402
    ANALYSIS, ASSESSMENT_VALUE_COLUMNS, COUNTIES, LGBM_CONFIG_PATH, OUTPUT,
    PROPERTY_USE_SET_NAMES, SEED, chronological_splits, lr_feature_groups, write_json,
)

LGBM_KEYS = "test_best_r2,cv_top1_r2,cv_top2_r2"
ACS_PREFIXES = ("acs_",)


def v3_features(data: pd.DataFrame, train_rows: int, include_prior: bool) -> tuple[pd.DataFrame, list[str]]:
    feats, cats = feature_frame(data, train_rows, include_prior, False)
    drop = [c for c in feats.columns if c.startswith(ACS_PREFIXES) or c in ASSESSMENT_VALUE_COLUMNS]
    feats = feats.drop(columns=drop, errors="ignore")
    overlap = set(feats.columns) & ASSESSMENT_VALUE_COLUMNS
    if overlap:
        raise RuntimeError(f"assessment-value predictors leaked: {sorted(overlap)}")
    cats = [c for c in cats if c in feats.columns]
    return feats, cats


def enrich(actual, predicted, train_actual) -> dict:
    m = score_predictions(actual, predicted, train_actual)
    m.update(paper_mechanism_metrics(np.log(actual), np.log(predicted)))
    try:
        m["Delta_NL"] = float(estimate_delta_nl(np.log(actual), np.log(predicted), row_ids=np.arange(len(actual))).get("Delta_NL", np.nan))
    except Exception as exc:
        m["Delta_NL"] = np.nan
        m["Delta_NL_error"] = str(exc)
    return {k: v for k, v in m.items() if k != "ratio_by_decile"}


def fit_lr_dev_only(features: pd.DataFrame, categorical: list[str], y_log: pd.Series, n_dev: int, n_val_end: int):
    """Fit on development; score validation. Categorical cardinality from development only."""
    train = features.iloc[:n_dev]
    numeric, cats, dropped = lr_feature_groups(train, categorical)
    transformers = [("num", SimpleImputer(strategy="median"), numeric)]
    if cats:
        transformers.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cats))
    pre = ColumnTransformer(transformers, remainder="drop")
    model = Pipeline([("pre", pre), ("lr", LinearRegression())])
    model.fit(train, y_log.iloc[:n_dev])
    val_pred = np.exp(model.predict(features.iloc[n_dev:n_val_end]))
    return val_pred, {"lr_cats": cats, "lr_numeric": numeric, "lr_dropped_high_card": dropped}


def run_county(key: str, threads: int, use_set: str = "primary_385") -> dict:
    freeze = ANALYSIS / "panel_freeze" / "final_panel_freeze_v3.yaml"
    # Pre-freeze path must run whether or not freeze exists; it must not write test preds.
    # A non-primary use set is a labeled sensitivity: it reads its own modeling table
    # and writes to its own directories so the frozen primary artifacts stay untouched.
    suffix = "" if use_set == "primary_385" else f"_{use_set}"
    label = key if use_set == "primary_385" else f"{key}_{use_set}_sensitivity"
    table = OUTPUT / "modeling_tables" / key / f"history_market_core{suffix}.parquet"
    if not table.exists():
        return {"county_key": key, "status": "SKIPPED_NO_TABLE"}
    data = pd.read_parquet(table).sort_values("sale_date").reset_index(drop=True)
    split, validation_split = chronological_splits(len(data))
    assert data["sale_date"].iloc[:split].max() <= data["sale_date"].iloc[split:].min()
    # Explicitly never slice test rows for scoring.
    pretest = data.iloc[:split].copy()
    y_log = np.log(pretest["sale_price"].astype(float))
    configs = load_lgbm_configs(LGBM_CONFIG_PATH, LGBM_KEYS, threads)
    features, cats = v3_features(pretest, validation_split, True)
    candidate_rows = []
    for name, params in configs.items():
        model = LGBMRegressor(**params)
        model.fit(features.iloc[:validation_split], y_log.iloc[:validation_split], categorical_feature=cats)
        pred = np.exp(model.predict(features.iloc[validation_split:split]))
        scores = enrich(
            pretest.sale_price.iloc[validation_split:split].to_numpy(),
            pred,
            pretest.sale_price.iloc[:validation_split].to_numpy(),
        )
        candidate_rows.append({"lgbm_config": name, **scores})
    candidates = pd.DataFrame(candidate_rows)
    best = candidates.sort_values("R2", ascending=False).iloc[0]
    selected = str(best["lgbm_config"])
    params = configs[selected]
    lgbm = LGBMRegressor(**params)
    lgbm.fit(features.iloc[:validation_split], y_log.iloc[:validation_split], categorical_feature=cats)
    lgbm_val = np.exp(lgbm.predict(features.iloc[validation_split:split]))
    lr_val, lr_spec = fit_lr_dev_only(features, cats, y_log, validation_split, split)
    lr_cats = lr_spec["lr_cats"]
    lgbm_m = enrich(
        pretest.sale_price.iloc[validation_split:split].to_numpy(),
        lgbm_val,
        pretest.sale_price.iloc[:validation_split].to_numpy(),
    )
    lr_m = enrich(
        pretest.sale_price.iloc[validation_split:split].to_numpy(),
        lr_val,
        pretest.sale_price.iloc[:validation_split].to_numpy(),
    )
    struct, struct_cats = v3_features(pretest, validation_split, False)
    lgbm_s = LGBMRegressor(**params)
    lgbm_s.fit(struct.iloc[:validation_split], y_log.iloc[:validation_split], categorical_feature=struct_cats)
    s_val = np.exp(lgbm_s.predict(struct.iloc[validation_split:split]))
    s_m = enrich(
        pretest.sale_price.iloc[validation_split:split].to_numpy(),
        s_val,
        pretest.sale_price.iloc[:validation_split].to_numpy(),
    )
    out_dir = ANALYSIS / "baselines_pre_freeze" / label
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(out_dir / "lgbm_config_validation.csv", index=False)
    meta = {
        "county_key": key,
        "property_use_set": use_set,
        "freeze_status": (
            "PRIMARY_FROZEN_COHORT" if use_set == "primary_385"
            else "SENSITIVITY_ONLY_NOT_A_FREEZE_REVISION"
        ),
        "modeling_table": str(table),
        "n_full": int(len(data)),
        "n_pretest": int(split),
        "n_development": int(validation_split),
        "n_validation": int(split - validation_split),
        "n_test_held_out_unscored": int(len(data) - split),
        "test_block_scored": False,
        "test_predictions_written": False,
        "selected_lgbm_config": selected,
        "lr_cats_from_development_only": lr_cats,
        "lr_dropped_high_card": lr_spec["lr_dropped_high_card"],
        "validation_lgbm_HISTORY_MARKET_CORE": lgbm_m,
        "validation_lr_HISTORY_MARKET_CORE": lr_m,
        "validation_lgbm_HISTORY_STRUCTURAL_CORE": s_m,
        "wayne_is_not_detroit": key == "wayne",
        "written_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": SEED,
    }
    write_json(out_dir / "run_meta.json", meta)
    pd.DataFrame([
        {"model": "LGBM_MARKET", **lgbm_m},
        {"model": "LR_MARKET", **lr_m},
        {"model": "LGBM_STRUCTURAL", **s_m},
    ]).to_csv(out_dir / "validation_metrics.csv", index=False)
    pred_dir = OUTPUT / "pre_freeze_models" / label
    pred_dir.mkdir(parents=True, exist_ok=True)
    val_idx = pretest.index[validation_split:split]
    pd.DataFrame({
        "row_index": val_idx,
        "sale_date": pretest.sale_date.iloc[validation_split:split].astype(str).to_numpy(),
        "y": pretest.sale_price.iloc[validation_split:split].to_numpy(),
        "lgbm_market": lgbm_val,
        "lr_market": lr_val,
        "split": "validation",
    }).to_parquet(pred_dir / "validation_predictions.parquet", index=False)
    print(json.dumps({"county_key": key, "property_use_set": use_set, "val_R2": lgbm_m.get("R2"), "test_scored": False}, default=str), flush=True)
    return meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--county-key", required=True)
    parser.add_argument("--lgbm-threads", type=int, default=8)
    parser.add_argument(
        "--property-use-set", default="primary_385", choices=PROPERTY_USE_SET_NAMES,
        help="primary_385 is the frozen cohort; broad_residential is a labeled sensitivity only.",
    )
    args = parser.parse_args()
    run_county(args.county_key, args.lgbm_threads, args.property_use_set)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
