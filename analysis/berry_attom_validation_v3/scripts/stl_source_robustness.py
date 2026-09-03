#!/usr/bin/env python3
"""Step 13: St. Louis local vs ATTOM predictor robustness on a common transaction set.

Holds fixed: rows, target, chronology, model family, tuning protocol.
Compares local dwelling predictors vs ATTOM Assessor History predictors.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.other_counties_benchmars import (  # noqa: E402
    bootstrap_scores, feature_frame, load_lgbm_configs, score_predictions,
)
from utils.delta_nl import estimate_delta_nl
from utils.motivation_utils import paper_mechanism_metrics
from analysis.berry_attom_validation_v2.scripts.apn_normalize import normalize_apn_series  # noqa: E402
from analysis.berry_attom_validation_v3.scripts.v3_common import (  # noqa: E402
    ANALYSIS, LGBM_CONFIG_PATH, N_BOOTSTRAP, OUTPUT, SEED, TEST_FRACTION, VALIDATION_FRACTION,
)

OUT_A = ANALYSIS / "st_louis_source_robustness"
OUT_B = OUTPUT / "st_louis_source_robustness"
FIG = ANALYSIS / "figures"
LOCAL_NUM = ["STORIES", "YRBLT", "RMBED", "FIXBATH", "FIXHALF", "SFLA"]
LOCAL_CAT = ["STYLE", "GRADE", "CDU"]


def enrich(actual, predicted, train_actual):
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


def common_key(frame: pd.DataFrame, apn_col: str, date_col: str) -> pd.Series:
    """Parcel+date identity. Price is corroboration, not the join key."""
    apn = normalize_apn_series(frame[apn_col])
    dt = pd.to_datetime(frame[date_col]).dt.strftime("%Y-%m-%d")
    return apn + "|" + dt


def fit_lgbm(data: pd.DataFrame, features: pd.DataFrame, categorical: list[str], split: int, validation_split: int):
    y_log = np.log(data["sale_price"].astype(float))
    configs = load_lgbm_configs(LGBM_CONFIG_PATH, "test_best_r2,cv_top1_r2,cv_top2_r2", 8)
    best_name, best_r2 = None, -np.inf
    feat_val, cat_val = features, categorical
    for name, params in configs.items():
        model = LGBMRegressor(**params)
        model.fit(feat_val.iloc[:validation_split], y_log.iloc[:validation_split], categorical_feature=cat_val)
        pred = np.exp(model.predict(feat_val.iloc[validation_split:split]))
        r2 = score_predictions(
            data.sale_price.iloc[validation_split:split], pred, data.sale_price.iloc[:validation_split],
        )["R2"]
        if r2 > best_r2:
            best_name, best_r2 = name, r2
    params = configs[str(best_name)]
    model = LGBMRegressor(**params)
    # Pre-freeze: fit on development only; never score or store the test block.
    model.fit(features.iloc[:validation_split], y_log.iloc[:validation_split], categorical_feature=categorical)
    pred = np.full(len(features), np.nan)
    pred[validation_split:split] = np.exp(model.predict(features.iloc[validation_split:split]))
    return pred, str(best_name)


def main() -> int:
    local_path = OUT_B / "local_modeling_table.parquet"
    attom_path = OUTPUT / "modeling_tables" / "st_louis_county" / "history_market_core.parquet"
    if not local_path.exists() or not attom_path.exists():
        print("SKIP missing local or ATTOM modeling table", flush=True)
        return 0
    OUT_A.mkdir(parents=True, exist_ok=True)
    OUT_B.mkdir(parents=True, exist_ok=True)
    local = pd.read_parquet(local_path)
    attom = pd.read_parquet(attom_path)
    local["common_key"] = common_key(local, "PARID", "sale_date")
    # PARCELNUMBERFORMATTED is 100% missing in this Dewey History delivery.
    if "PARCELNUMBERRAW" not in attom.columns:
        raise RuntimeError("ATTOM modeling table lacks PARCELNUMBERRAW")
    attom["common_key"] = common_key(attom, "PARCELNUMBERRAW", "sale_date")
    # Local sales extract ends 2019; do not form the common cohort on 2020-2025 ATTOM-only years.
    attom = attom.loc[pd.to_datetime(attom["sale_date"]).between("2016-01-01", "2019-12-31")].copy()
    local_u = local.drop_duplicates("common_key", keep="first")
    attom_u = attom.drop_duplicates("common_key", keep="first")
    keys = sorted(set(local_u["common_key"].dropna()) & set(attom_u["common_key"].dropna()))
    local_c = local_u.loc[local_u["common_key"].isin(keys)].copy()
    attom_c = attom_u.loc[attom_u["common_key"].isin(keys)].copy()
    # Same row order / chronology / target: sort by local sale_date then key.
    local_c = local_c.sort_values(["sale_date", "common_key"]).reset_index(drop=True)
    attom_c = attom_c.set_index("common_key").loc[local_c["common_key"]].reset_index()
    attom_c["attom_transfer_amount"] = pd.to_numeric(attom_c["sale_price"], errors="coerce")
    price_rel = (
        (attom_c["attom_transfer_amount"] - local_c["sale_price"]).abs()
        / local_c["sale_price"].replace(0, np.nan)
    )
    price_agree = {
        "share_exact_price": float(price_rel.eq(0).mean()) if len(local_c) else None,
        "share_price_le_1pct": float(price_rel.le(0.01).mean()) if len(local_c) else None,
        "share_price_le_5pct": float(price_rel.le(0.05).mean()) if len(local_c) else None,
    }
    # Same target for the predictor-source comparison: local PRICE.
    attom_c["sale_price"] = local_c["sale_price"].to_numpy()
    attom_c["sale_date"] = local_c["sale_date"].to_numpy()
    n = len(local_c)
    split = int(n * (1 - TEST_FRACTION))
    validation_split = int(split * (1 - VALIDATION_FRACTION))
    if not (1 <= validation_split < split < n):
        OUT_A.mkdir(parents=True, exist_ok=True)
        note = {
            "n_common": n,
            "status": "SKIPPED_COMMON_SAMPLE_TOO_SMALL",
            "join": "normalized PARCELNUMBERRAW/PARID + sale_date; 2016-2019",
            "test_block_scored": False,
        }
        (OUT_A / "metrics.csv").write_text("source,n_common,status\n")
        (OUT_A / "ST_LOUIS_SOURCE_ROBUSTNESS.md").write_text(
            "# St. Louis County local vs ATTOM source robustness\n\n"
            f"Common N={n}. Too small for chronological development/validation/test.\n"
            "Join is normalized raw APN + sale date on 2016-2019 (formatted APN is missing).\n"
        )
        print(json.dumps(note), flush=True)
        return 0

    for col in LOCAL_NUM:
        local_c[col] = pd.to_numeric(local_c[col], errors="coerce")
    for col in LOCAL_CAT:
        local_c[col] = local_c[col].astype("string").fillna("__missing__").astype("category")
    local_feat = local_c[LOCAL_NUM + LOCAL_CAT]
    attom_feat, attom_cat = feature_frame(attom_c, validation_split, False, False)

    local_pred, local_cfg = fit_lgbm(local_c, local_feat, LOCAL_CAT, split, validation_split)
    attom_pred, attom_cfg = fit_lgbm(attom_c, attom_feat, attom_cat, split, validation_split)

    train_price = local_c.sale_price.iloc[:validation_split].to_numpy()
    val = local_c.iloc[validation_split:split]
    rows = []
    for name, pred, cfg in (
        ("local_historical_dwelling", local_pred, local_cfg),
        ("attom_assessor_history", attom_pred, attom_cfg),
    ):
        mets = enrich(val.sale_price.to_numpy(), pred[validation_split:split], train_price)
        row = {"source": name, "selected_lgbm_config": cfg, "n_common": n, "n_validation": int(len(val)),
               "n_test_unscored": int(n - split), "test_block_scored": False, **price_agree}
        row.update({k: v for k, v in mets.items() if k != "ratio_by_decile"})
        row["ratio_by_decile"] = json.dumps(mets.get("ratio_by_decile"))
        rows.append(row)

    common = local_c[["common_key", "PARID", "sale_date", "sale_price"]].copy()
    common["attom_transfer_amount"] = attom_c["attom_transfer_amount"].to_numpy()
    common["pred_local"] = local_pred
    common["pred_attom"] = attom_pred
    common["split"] = np.where(np.arange(n) < validation_split, "development",
                               np.where(np.arange(n) < split, "validation", "test_unscored"))
    common.to_parquet(OUT_B / "common_transactions.parquet", index=False)
    common_dir = OUTPUT / "common_cohorts"
    common_dir.mkdir(parents=True, exist_ok=True)
    common.to_parquet(common_dir / "st_louis_county_local_attom.parquet", index=False)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT_A / "metrics.csv", index=False)

    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    labels = ["R2", "PRB", "Beta_log"]
    x = np.arange(len(labels))
    a = metrics.loc[metrics["source"].eq("local_historical_dwelling")].iloc[0]
    b = metrics.loc[metrics["source"].eq("attom_assessor_history")].iloc[0]
    axes[0].bar(x - 0.2, [a.get("R2", np.nan), a.get("PRB", np.nan), a.get("Beta_log", np.nan)], 0.4, label="local")
    axes[0].bar(x + 0.2, [b.get("R2", np.nan), b.get("PRB", np.nan), b.get("Beta_log", np.nan)], 0.4, label="ATTOM")
    axes[0].set_xticks(x, labels)
    axes[0].legend()
    axes[0].set_title("Validation metrics (test unscored)")
    for ax, src, pred in (
        (axes[1], "local", local_pred),
        (axes[2], "ATTOM", attom_pred),
    ):
        ratio = pred[validation_split:split] / local_c.sale_price.iloc[validation_split:split].to_numpy()
        q = pd.qcut(local_c.sale_price.iloc[validation_split:split], 10, labels=False, duplicates="drop")
        ax.plot(np.arange(1, 11), [np.median(ratio[q == i]) for i in range(10)], marker="o")
        ax.axhline(1.0, color="gray", lw=0.8)
        ax.set_title(f"{src} decile ratio")
        ax.set_xlabel("price decile")
    fig.tight_layout()
    fig.savefig(FIG / "st_louis_source_comparison.pdf")
    fig.savefig(OUT_A / "source_comparison.pdf")
    plt.close(fig)

    report = [
        "# St. Louis County local vs ATTOM source robustness",
        "",
        "Same transactions, same sale target (local PRICE), same chronology, same LGBM config search.",
        "Local predictors: dwelling snapshot strictly before SALEDT.",
        "ATTOM predictors: HISTORY_STRUCTURAL_CORE-style Assessor History (no prior-sale market history, no tax values).",
        "",
        f"Common N={n}; validation N={int(len(val))}; test unscored until freeze.",
        f"Join: normalized PARCELNUMBERRAW/PARID + sale date, 2016-2019 (formatted APN missing).",
        f"Price corroboration vs ATTOM TRANSFERAMOUNT: exact={price_agree['share_exact_price']}; "
        f"<=1%={price_agree['share_price_le_1pct']}; <=5%={price_agree['share_price_le_5pct']}.",
        "",
        metrics.drop(columns=["ratio_by_decile"]).to_string(index=False),
        "",
        "Berry official assessment ratios are not used here. This is a predictor-source comparison on a common sale target.",
        "",
    ]
    (OUT_A / "ST_LOUIS_SOURCE_ROBUSTNESS.md").write_text("\n".join(report) + "\n")
    (ANALYSIS / "reports" / "ST_LOUIS_SOURCE_ROBUSTNESS.md").write_text("\n".join(report) + "\n")
    print(metrics.drop(columns=["ratio_by_decile"]).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
