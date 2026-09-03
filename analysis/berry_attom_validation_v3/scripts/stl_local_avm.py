#!/usr/bin/env python3
"""Step 12: corrected St. Louis County local-government AVM.

Uses actual SALEDT and PRICE from the 2019 cumulative sales extract (SALEVAL==X).
Never uses APRTOT as sale price. Never uses joined.csv as the canonical table.
Handles the 2012 dwelling folder / TAXYR=2013 labeling defect explicitly.
"""
from __future__ import annotations

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
    bootstrap_scores, load_lgbm_configs, score_predictions,
)
from utils.delta_nl import estimate_delta_nl
from utils.motivation_utils import paper_mechanism_metrics
from analysis.berry_attom_validation_v3.scripts.v3_common import (  # noqa: E402
    ANALYSIS, BERRY_RAW, LGBM_CONFIG_PATH, N_BOOTSTRAP, OUTPUT, SEED,
    TEST_FRACTION, VALIDATION_FRACTION,
)

STL_ASSESS = BERRY_RAW / "st_louis_county_mo/box/rabph6sd546szpwe6likep763hv8g3v3/new data/2020-stlco-assessments"
TXN = ANALYSIS / "berry_reproduction" / "st_louis_county_mo_transactions.parquet"
OUT_A = ANALYSIS / "st_louis_source_robustness"
OUT_B = OUTPUT / "st_louis_source_robustness"

COLS_39 = [
    "PARID", "CARD", "TAXYR", "STORIES", "EXTWALL", "STYLE", "YRBLT", "EFFYR", "YRREMOD",
    "RMTOT", "RMBED", "RMFAM", "FIXBATH", "FIXHALF", "FIXADDL", "FIXTOT", "REMKIT", "REMBATH",
    "BSMT", "HEAT", "FUEL", "HEATSYS", "ATTIC", "UNFINAREA", "RECROMAREA", "FINBSMTAREA",
    "WBFP_O", "WBFP_S", "WBFP_PF", "BSMTCAR", "CONDOLVL", "CONDOTYP", "CONDOVW", "UNITNO",
    "CNDBASEVAL", "MGFA", "SFLA", "GRADE", "CDU",
]
COLS_37 = [c for c in COLS_39 if c not in {"WBFP_S", "CNDBASEVAL"}]
USE = ["PARID", "TAXYR", "STORIES", "STYLE", "YRBLT", "RMBED", "FIXBATH", "FIXHALF", "SFLA", "GRADE", "CDU"]
NUMERIC = ["STORIES", "YRBLT", "RMBED", "FIXBATH", "FIXHALF", "SFLA"]
CATEGORICAL = ["STYLE", "GRADE", "CDU"]


def _first_data_line(path: Path) -> tuple[str, str]:
    with path.open(encoding="latin1", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.upper().startswith("PARID"):
                continue
            delim = "|" if s.count("|") >= s.count(",") else ","
            return s, delim
    raise ValueError(f"no data lines in {path}")


def load_dwelling(sold_parids: set[str]) -> pd.DataFrame:
    cache = OUT_B / "local_dwelling_use_cols.parquet"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        out = pd.read_parquet(cache)
        return out.loc[out["PARID"].isin(sold_parids)].copy()
    frames = []
    notes = []
    for yrdir in sorted(p for p in STL_ASSESS.iterdir() if p.is_dir() and p.name.isdigit()):
        cands = [
            p for p in yrdir.iterdir()
            if p.is_file() and p.name.lower().startswith("dwelling") and not p.name.endswith(".headers")
        ]
        if not cands:
            notes.append(f"{yrdir.name}: no dwelling file")
            continue
        path = cands[0]
        sample, delim = _first_data_line(path)
        nfields = len(sample.split(delim))
        names = COLS_39 if nfields == 39 else COLS_37 if nfields == 37 else None
        if names is None:
            notes.append(f"{yrdir.name}: unexpected nfields={nfields}")
            continue
        df = pd.read_csv(
            path, sep=delim, header=None, names=names, usecols=[c for c in USE if c in names],
            dtype=str, encoding="latin1", skip_blank_lines=True, engine="c",
        )
        df = df.loc[~df["PARID"].astype(str).str.upper().str.startswith("PARID")].copy()
        df["PARID"] = df["PARID"].astype(str).str.strip()
        df["TAXYR"] = pd.to_numeric(df["TAXYR"], errors="coerce")
        df["source_folder_year"] = int(yrdir.name)
        df = df.loc[df["PARID"].isin(sold_parids) & df["TAXYR"].notna()].copy()
        taxyr_mode = int(df["TAXYR"].mode(dropna=True).iloc[0]) if len(df) else None
        notes.append(
            f"folder={yrdir.name} file={path.name} n={len(df)} nfields={nfields} taxyr_mode={taxyr_mode}"
        )
        if int(yrdir.name) == 2012 and taxyr_mode == 2013:
            notes.append("DEFECT: 2012 dwelling folder has TAXYR mode 2013")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["PARID", "TAXYR", "source_folder_year"]).drop_duplicates(["PARID", "TAXYR"], keep="last")
    out.to_parquet(cache, index=False)
    (OUT_A / "dwelling_load_notes.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")
    return out.loc[out["PARID"].isin(sold_parids)].copy()


def enrich(actual, predicted, train_actual):
    m = score_predictions(actual, predicted, train_actual)
    m.update(paper_mechanism_metrics(np.log(actual), np.log(predicted)))
    try:
        m["Delta_NL"] = float(estimate_delta_nl(np.log(actual), np.log(predicted), row_ids=np.arange(len(actual))).get("Delta_NL", np.nan))
    except Exception as exc:
        m["Delta_NL"] = np.nan
        m["Delta_NL_error"] = str(exc)
    return m


def chronological_splits(n: int) -> tuple[int, int]:
    split = int(n * (1 - TEST_FRACTION))
    validation_split = int(split * (1 - VALIDATION_FRACTION))
    if not (1 <= validation_split < split < n):
        raise ValueError(f"STL local sample too small: n={n}")
    return split, validation_split


def fit_models(data: pd.DataFrame, label: str) -> pd.DataFrame:
    data = data.sort_values("sale_date").reset_index(drop=True)
    split, validation_split = chronological_splits(len(data))
    y_log = np.log(data["sale_price"].astype(float))
    for col in NUMERIC:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    for col in CATEGORICAL:
        data[col] = data[col].astype("string").fillna("__missing__").astype("category")
    features = data[NUMERIC + CATEGORICAL]
    configs = load_lgbm_configs(LGBM_CONFIG_PATH, "test_best_r2,cv_top1_r2,cv_top2_r2", 8)
    cand = []
    for name, params in configs.items():
        model = LGBMRegressor(**params)
        model.fit(features.iloc[:validation_split], y_log.iloc[:validation_split], categorical_feature=CATEGORICAL)
        pred = np.exp(model.predict(features.iloc[validation_split:split]))
        scores = score_predictions(data.sale_price.iloc[validation_split:split], pred, data.sale_price.iloc[:validation_split])
        cand.append({"lgbm_config": name, "R2": scores["R2"]})
    selected = pd.DataFrame(cand).sort_values("R2", ascending=False).iloc[0]["lgbm_config"]
    params = configs[str(selected)]
    lgbm = LGBMRegressor(**params)
    lgbm.fit(features.iloc[:validation_split], y_log.iloc[:validation_split], categorical_feature=CATEGORICAL)
    lgbm_pred_val = np.exp(lgbm.predict(features.iloc[validation_split:split]))
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAL),
    ])
    lr = Pipeline([("pre", pre), ("lr", LinearRegression())])
    lr.fit(features.iloc[:validation_split], y_log.iloc[:validation_split])
    lr_pred_val = np.exp(lr.predict(features.iloc[validation_split:split]))
    train_price = data.sale_price.iloc[:validation_split].to_numpy()
    val = data.iloc[validation_split:split]
    rows = []
    for model_name, pred in (("LinearRegression", lr_pred_val), ("LGBMRegressor", lgbm_pred_val)):
        val_m = enrich(val.sale_price.to_numpy(), pred, train_price)
        row = {"sample": label, "model": model_name, "n": int(len(data)), "n_validation": int(len(val)),
               "n_test_unscored": int(len(data) - split),
               "test_block_scored": False,
               "selected_lgbm_config": selected if model_name == "LGBMRegressor" else "",
               **{k: v for k, v in val_m.items() if not isinstance(v, dict)}}
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    OUT_A.mkdir(parents=True, exist_ok=True)
    OUT_B.mkdir(parents=True, exist_ok=True)
    sold = pd.read_parquet(TXN)
    sold["sale_date"] = pd.to_datetime(sold["berry_sale_date"])
    sold["sale_price"] = pd.to_numeric(sold["berry_sale_price"], errors="coerce")
    sold["PARID"] = sold["berry_parcel_raw"].astype(str).str.strip()
    # Local AVM window: dated 2016-2019 (sales extract ends 2019; standardized ATTOM window starts 2016).
    sold = sold.loc[sold["sale_date"].between("2016-01-01", "2019-12-31") & sold["sale_price"].gt(0)].copy()
    print("sold 2016-2019 SALEVAL=X", len(sold), flush=True)
    dwell = load_dwelling(set(sold["PARID"]))
    dwell["assessed_through"] = pd.to_datetime(
        pd.to_numeric(dwell["TAXYR"], errors="coerce").astype("Int64").astype("string") + "-12-31",
        errors="coerce",
    )
    left = sold.sort_values(["sale_date", "PARID"])
    right = dwell.dropna(subset=["PARID", "assessed_through"]).sort_values(["assessed_through", "PARID"])
    matched = pd.merge_asof(
        left, right, left_on="sale_date", right_on="assessed_through", by="PARID",
        direction="backward", allow_exact_matches=False,
    ).dropna(subset=["assessed_through"])
    matched["history_lag_days"] = (matched["sale_date"] - matched["assessed_through"]).dt.days
    matched["affected_2012_defect"] = matched["source_folder_year"].eq(2012)
    assert (matched["assessed_through"] < matched["sale_date"]).all()
    matched.to_parquet(OUT_B / "local_modeling_table.parquet", index=False)
    primary = fit_models(matched, "primary_include_2012_snapshot")
    sensitivity = fit_models(matched.loc[~matched["affected_2012_defect"]].copy(), "exclude_2012_snapshot_defect")
    metrics = pd.concat([primary, sensitivity], ignore_index=True)
    metrics.to_csv(OUT_A / "local_avm_metrics.csv", index=False)
    meta = {
        "n_sold_2016_2019": int(len(sold)),
        "n_matched_dwelling_strictly_before": int(len(matched)),
        "n_affected_2012_defect": int(matched["affected_2012_defect"].sum()),
        "did_not_use_joined_csv": True,
        "did_not_use_aprtot_as_price": True,
        "history_rule": "latest dwelling TAXYR year-end strictly before SALEDT",
        "written_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (OUT_A / "local_avm_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
