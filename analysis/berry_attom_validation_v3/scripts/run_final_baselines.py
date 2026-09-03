#!/usr/bin/env python3
"""Held-out test baselines AFTER freeze file exists. Scores test once."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.other_counties_benchmars import bootstrap_scores, load_lgbm_configs, score_predictions  # noqa: E402
from utils.delta_nl import estimate_delta_nl
from utils.motivation_utils import paper_mechanism_metrics
from analysis.berry_attom_validation_v3.scripts.run_prefreeze_baselines import v3_features  # noqa: E402
from analysis.berry_attom_validation_v3.scripts.v3_common import (  # noqa: E402
    ANALYSIS, COUNTIES, LGBM_CONFIG_PATH, N_BOOTSTRAP, OUTPUT, SEED,
    chronological_splits, lr_feature_groups, write_json,
)

FREEZE = ANALYSIS / "panel_freeze" / "final_panel_freeze_v3.yaml"


def enrich(actual, predicted, train_actual) -> dict:
    m = score_predictions(actual, predicted, train_actual)
    m.update(paper_mechanism_metrics(np.log(actual), np.log(predicted)))
    try:
        m["Delta_NL"] = float(estimate_delta_nl(np.log(actual), np.log(predicted), row_ids=np.arange(len(actual))).get("Delta_NL", np.nan))
    except Exception as exc:
        m["Delta_NL"] = np.nan
        m["Delta_NL_error"] = str(exc)
    ratio = predicted / actual
    q = pd.qcut(actual, 10, labels=False, duplicates="drop")
    m["ratio_by_decile"] = {str(int(i)): float(np.median(ratio[q == i])) for i in np.unique(q)}
    return m


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--county-key", required=True)
    parser.add_argument("--lgbm-threads", type=int, default=8)
    args = parser.parse_args()
    if not FREEZE.exists():
        raise SystemExit("Freeze file missing; refusing held-out evaluation.")
    freeze = yaml.safe_load(FREEZE.read_text())
    pre = json.loads((ANALYSIS / "baselines_pre_freeze" / args.county_key / "run_meta.json").read_text())
    if pre.get("test_block_scored"):
        raise SystemExit("Pre-freeze run already scored test; abort.")
    data = pd.read_parquet(OUTPUT / "modeling_tables" / args.county_key / "history_market_core.parquet")
    data = data.sort_values("sale_date").reset_index(drop=True)
    split, validation_split = chronological_splits(len(data))
    y_log = np.log(data["sale_price"].astype(float))
    selected = pre["selected_lgbm_config"]
    configs = load_lgbm_configs(LGBM_CONFIG_PATH, "test_best_r2,cv_top1_r2,cv_top2_r2", args.lgbm_threads)
    params = configs[selected]
    features, cats = v3_features(data, split, True)
    lgbm = LGBMRegressor(**params)
    lgbm.fit(features.iloc[:split], y_log.iloc[:split], categorical_feature=cats)
    pred = np.exp(lgbm.predict(features))
    train = features.iloc[:split]
    numeric, lr_cats, dropped = lr_feature_groups(train, cats)
    transformers = [("num", SimpleImputer(strategy="median"), numeric)]
    if lr_cats:
        transformers.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), lr_cats))
    preproc = ColumnTransformer(transformers, remainder="drop")
    lr = Pipeline([("pre", preproc), ("lr", LinearRegression())])
    lr.fit(train, y_log.iloc[:split])
    lr_pred = np.exp(lr.predict(features))
    train_price = data.sale_price.iloc[:split].to_numpy()
    test = data.iloc[split:]
    lgbm_m = enrich(test.sale_price.to_numpy(), pred[split:], train_price)
    lr_m = enrich(test.sale_price.to_numpy(), lr_pred[split:], train_price)
    _, boot = bootstrap_scores(test, pred[split:], train_price, N_BOOTSTRAP, "M", SEED)
    out_a = ANALYSIS / "final_baselines" / args.county_key
    out_b = OUTPUT / "final_models" / args.county_key
    out_a.mkdir(parents=True, exist_ok=True)
    out_b.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"y": test.sale_price, "lgbm": pred[split:], "lr": lr_pred[split:], "sale_date": test.sale_date.astype(str)}).to_parquet(
        out_b / "heldout_predictions.parquet", index=False
    )
    boot.to_csv(out_a / "bootstrap_lgbm.csv", index=False)
    write_json(out_a / "heldout_metrics.json", {
        "county_key": args.county_key,
        "freeze_authorized_note": freeze.get("direct_surrogate_authorized"),
        "selected_lgbm_config": selected,
        "lgbm": {k: v for k, v in lgbm_m.items() if k != "ratio_by_decile"},
        "lr": {k: v for k, v in lr_m.items() if k != "ratio_by_decile"},
        "ratio_by_decile_lgbm": lgbm_m.get("ratio_by_decile"),
        "n_test": int(len(test)),
    })
    print(json.dumps({"county_key": args.county_key, "heldout_R2": lgbm_m.get("R2")}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
