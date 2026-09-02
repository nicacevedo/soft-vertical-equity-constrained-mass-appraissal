#!/usr/bin/env python3
"""Step 11: Linear Regression + ordinary LightGBM only. No positive-rho models.

PRIMARY feature set HISTORY_MARKET_CORE is selected among the frozen LGBM config
keys on validation only, then refit on the full pre-test block. Target scale is
fixed to log. HISTORY_STRUCTURAL_CORE is a secondary reported sensitivity and is
not used to choose hyperparameters.
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
from scripts.other_counties_benchmars import (  # noqa: E402
    bootstrap_scores, feature_frame, load_lgbm_configs, score_predictions,
)
from utils.delta_nl import estimate_delta_nl
from utils.motivation_utils import paper_mechanism_metrics
from analysis.berry_attom_validation_v2.scripts.v2_common import (  # noqa: E402
    ANALYSIS, ASSESSMENT_VALUE_COLUMNS, COUNTIES, LGBM_CONFIG_PATH, N_BOOTSTRAP,
    OUTPUT, SEED, TEST_FRACTION, VALIDATION_FRACTION,
)

LGBM_KEYS = "test_best_r2,cv_top1_r2,cv_top2_r2"
PRIMARY = "HISTORY_MARKET_CORE"
SECONDARY = "HISTORY_STRUCTURAL_CORE"


def chronological_splits(n: int) -> tuple[int, int]:
    split = int(n * (1 - TEST_FRACTION))
    validation_split = int(split * (1 - VALIDATION_FRACTION))
    if not (1 <= validation_split < split < n):
        raise ValueError(f"sample too small for chronological splits: n={n}")
    return split, validation_split


def assert_no_value_predictors(columns: list[str]) -> None:
    overlap = set(columns) & ASSESSMENT_VALUE_COLUMNS
    if overlap:
        raise RuntimeError(f"assessment-value predictors leaked into features: {sorted(overlap)}")


def enrich_metrics(actual: np.ndarray, predicted: np.ndarray, train_actual: np.ndarray, row_ids=None) -> dict:
    metrics = score_predictions(actual, predicted, train_actual)
    y_log = np.log(actual)
    p_log = np.log(predicted)
    metrics.update(paper_mechanism_metrics(y_log, p_log))
    try:
        dnl = estimate_delta_nl(y_log, p_log, row_ids=row_ids if row_ids is not None else np.arange(len(y_log)))
        metrics["Delta_NL"] = float(dnl.get("Delta_NL", np.nan))
    except Exception as exc:
        metrics["Delta_NL"] = np.nan
        metrics["Delta_NL_error"] = str(exc)
    ratio = predicted / actual
    q = pd.qcut(actual, 10, labels=False, duplicates="drop")
    metrics["ratio_by_decile"] = {str(int(i)): float(np.median(ratio[q == i])) for i in np.unique(q)}
    return metrics


def fit_lr(features: pd.DataFrame, categorical: list[str], y_log: pd.Series, split: int, validation_split: int):
    numeric = [c for c in features.columns if c not in categorical]
    cats = [c for c in categorical if features[c].nunique(dropna=True) <= 32]
    pre = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), numeric),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), cats),
        ],
        remainder="drop",
    )
    model = Pipeline([("pre", pre), ("lr", LinearRegression())])
    model.fit(features.iloc[:split], y_log.iloc[:split])
    pred_log = model.predict(features)
    return np.exp(pred_log)


def run_county(key: str, threads: int) -> dict:
    table = OUTPUT / "modeling_tables" / key / "history_market_core.parquet"
    if not table.exists():
        return {"county_key": key, "status": "SKIPPED_NO_TABLE"}
    data = pd.read_parquet(table).sort_values("sale_date").reset_index(drop=True)
    split, validation_split = chronological_splits(len(data))
    assert data["sale_date"].iloc[:split].max() <= data["sale_date"].iloc[split:].min()
    target_log = np.log(data["sale_price"].astype(float))
    configs = load_lgbm_configs(LGBM_CONFIG_PATH, LGBM_KEYS, threads)
    out_dir = ANALYSIS / "baselines" / key
    out_big = OUTPUT / "baselines" / key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_big.mkdir(parents=True, exist_ok=True)

    rows = []
    # Secondary structural-only feature set is scored with the *same* selected
    # primary LGBM config after primary selection, plus a parallel validation
    # search reported separately. Hyperparameters for PRIMARY are chosen first.
    primary_features, primary_cat = feature_frame(data, validation_split, True, False)
    assert_no_value_predictors(list(primary_features.columns))
    candidate_rows = []
    for name, params in configs.items():
        model = LGBMRegressor(**params)
        model.fit(
            primary_features.iloc[:validation_split], target_log.iloc[:validation_split],
            categorical_feature=primary_cat,
        )
        pred = np.exp(model.predict(primary_features.iloc[validation_split:split]))
        scores = enrich_metrics(
            data.sale_price.iloc[validation_split:split].to_numpy(),
            pred,
            data.sale_price.iloc[:validation_split].to_numpy(),
        )
        candidate_rows.append({"feature_set": PRIMARY, "lgbm_config": name, "n_features": int(primary_features.shape[1]), **{
            k: v for k, v in scores.items() if k != "ratio_by_decile"
        }})
    candidates = pd.DataFrame(candidate_rows)
    # Select on validation R2_price only among the frozen config set (not raw vs log,
    # not feature-set shopping). R2_price is the protocol accuracy metric.
    best = candidates.sort_values("R2", ascending=False).iloc[0]
    selected_name = str(best["lgbm_config"])
    params = configs[selected_name]

    features, categorical = feature_frame(data, split, True, False)
    assert_no_value_predictors(list(features.columns))
    lgbm = LGBMRegressor(**params)
    lgbm.fit(features.iloc[:split], target_log.iloc[:split], categorical_feature=categorical)
    lgbm_pred = np.exp(lgbm.predict(features))
    lr_pred = fit_lr(features, categorical, target_log, split, validation_split)

    structural, structural_cat = feature_frame(data, split, False, False)
    assert_no_value_predictors(list(structural.columns))
    lgbm_s = LGBMRegressor(**params)
    lgbm_s.fit(structural.iloc[:split], target_log.iloc[:split], categorical_feature=structural_cat)
    structural_pred = np.exp(lgbm_s.predict(structural))

    train_price = data.sale_price.iloc[:split].to_numpy()
    test = data.iloc[split:]
    models = {
        "LinearRegression": lr_pred,
        "LGBMRegressor_HISTORY_MARKET_CORE": lgbm_pred,
        "LGBMRegressor_HISTORY_STRUCTURAL_CORE": structural_pred,
    }
    metric_rows = []
    boot_summaries = []
    for model_name, pred in models.items():
        train_m = enrich_metrics(data.sale_price.iloc[:split].to_numpy(), pred[:split], train_price)
        test_m = enrich_metrics(data.sale_price.iloc[split:].to_numpy(), pred[split:], train_price)
        draws, summary = bootstrap_scores(test, pred[split:], train_price, N_BOOTSTRAP, "M", SEED)
        for split_name, mets in (("train", train_m), ("test", test_m)):
            row = {"county_key": key, "model": model_name, "split": split_name}
            row.update({k: v for k, v in mets.items() if k != "ratio_by_decile"})
            row["ratio_by_decile"] = json.dumps(mets.get("ratio_by_decile"))
            metric_rows.append(row)
        summary = summary.assign(county_key=key, model=model_name)
        boot_summaries.append(summary)
        draws.to_parquet(out_big / f"bootstrap_{model_name}.parquet", index=False)

    pred_out = data[["ATTOMID", "TRANSACTIONID", "sale_date", "sale_price"]].copy() if "TRANSACTIONID" in data else data[["ATTOMID", "sale_date", "sale_price"]].copy()
    pred_out["split"] = np.where(np.arange(len(data)) < split, "train", "test")
    for model_name, pred in models.items():
        pred_out[f"pred_{model_name}"] = pred
    pred_out.to_parquet(out_big / "predictions.parquet", index=False)
    candidates.to_csv(out_dir / "lgbm_validation_candidates.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(out_dir / "metrics.csv", index=False)
    pd.concat(boot_summaries, ignore_index=True).to_csv(out_dir / "bootstrap_summary.csv", index=False)

    primary_test = next(r for r in metric_rows if r["model"] == "LGBMRegressor_HISTORY_MARKET_CORE" and r["split"] == "test")
    report = {
        "county_key": key,
        "status": "OK",
        "n": int(len(data)),
        "n_train": split,
        "n_validation": split - validation_split,
        "n_test": int(len(data) - split),
        "sale_date_range": [str(data.sale_date.min().date()), str(data.sale_date.max().date())],
        "test_date_range": [str(test.sale_date.min().date()), str(test.sale_date.max().date())],
        "selected_lgbm_config": selected_name,
        "selection_rule": "max validation R2_price among frozen LGBM configs; log target fixed; PRIMARY feature set only",
        "primary_test_R2_price": primary_test.get("R2"),
        "primary_test_PRB": primary_test.get("PRB"),
        "primary_test_Beta_log": primary_test.get("Beta_log"),
        "primary_test_Delta_NL": primary_test.get("Delta_NL"),
        "primary_test_dCor": primary_test.get("dCor_e_y"),
        "written_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=2, default=str), flush=True)
    return report


def write_baseline_report() -> None:
    parts = []
    for c in COUNTIES:
        meta = ANALYSIS / "baselines" / c["key"] / "run_meta.json"
        metrics = ANALYSIS / "baselines" / c["key"] / "metrics.csv"
        if not meta.exists():
            continue
        rec = json.loads(meta.read_text())
        parts.append(f"## {c['label']} (`{c['key']}`)\n")
        parts.append(f"- Berry unit: {c['berry_unit']}")
        parts.append(f"- Naming: Wayne models are **not** Detroit." if c["key"] == "wayne" else "")
        parts.append(f"- N={rec['n']} train={rec['n_train']} val={rec['n_validation']} test={rec['n_test']}")
        parts.append(f"- Selected LGBM config: `{rec['selected_lgbm_config']}` ({rec['selection_rule']})")
        parts.append(
            f"- PRIMARY test: R2_price={rec.get('primary_test_R2_price')} "
            f"PRB={rec.get('primary_test_PRB')} Beta_log={rec.get('primary_test_Beta_log')} "
            f"Delta_NL={rec.get('primary_test_Delta_NL')} dCor={rec.get('primary_test_dCor')}"
        )
        if metrics.exists():
            df = pd.read_csv(metrics)
            keep = df.loc[df["split"].eq("test"), [
                "model", "R2", "MAE", "COD", "PRD", "PRB", "Beta_log", "Delta_NL", "dCor_e_y",
            ]]
            parts.append("\n```\n" + keep.to_string(index=False) + "\n```\n")
    text = "# Baseline AVM report (LR + ordinary LightGBM only)\n\nNo Direct/Surrogate models are included.\n\n" + "\n".join(p for p in parts if p)
    (ANALYSIS / "reports" / "BASELINE_REPORT.md").write_text(text + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--county-key", default="all")
    parser.add_argument("--lgbm-threads", type=int, default=8)
    args = parser.parse_args()
    counties = [c for c in COUNTIES if args.county_key in {"all", c["key"]}]
    for c in counties:
        print("baseline", c["key"], flush=True)
        run_county(c["key"], args.lgbm_threads)
    write_baseline_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
