#!/usr/bin/env python3
"""St. Louis County baseline smoke test (LR + ordinary LightGBM). No Direct/Surrogate."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from utils.delta_nl import estimate_delta_nl
from utils.motivation_utils import compute_taxation_metrics, paper_mechanism_metrics

RAW = REPO / "data/berry_cmf/raw/st_louis_county_mo/box/rabph6sd546szpwe6likep763hv8g3v3"
OUT = REPO / "analysis/berry_cmf_validation/modeling_readiness"
LOG = REPO / "analysis/berry_cmf_validation/logs"
OUT.mkdir(parents=True, exist_ok=True)

# Dictionary (STLCO_REAL_DATA_DICTIONARY.pdf) has 39 fields. 2009-2012 extracts omit WBFP_S and CNDBASEVAL (37 fields).
COLS_39 = [
    "PARID", "CARD", "TAXYR", "STORIES", "EXTWALL", "STYLE", "YRBLT", "EFFYR", "YRREMOD",
    "RMTOT", "RMBED", "RMFAM", "FIXBATH", "FIXHALF", "FIXADDL", "FIXTOT", "REMKIT", "REMBATH",
    "BSMT", "HEAT", "FUEL", "HEATSYS", "ATTIC", "UNFINAREA", "RECROMAREA", "FINBSMTAREA",
    "WBFP_O", "WBFP_S", "WBFP_PF", "BSMTCAR", "CONDOLVL", "CONDOTYP", "CONDOVW", "UNITNO",
    "CNDBASEVAL", "MGFA", "SFLA", "GRADE", "CDU",
]
COLS_37 = [c for c in COLS_39 if c not in {"WBFP_S", "CNDBASEVAL"}]
USE = ["PARID", "TAXYR", "STORIES", "STYLE", "YRBLT", "RMBED", "FIXBATH", "FIXHALF", "SFLA", "GRADE", "CDU"]


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
    cache = REPO / "data/berry_cmf/derived/st_louis_county_mo/dwelling_use_cols.parquet"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        print("loading dwelling cache", cache, flush=True)
        out = pd.read_parquet(cache)
        return out.loc[out["PARID"].isin(sold_parids)].copy()
    root = RAW / "new data/2020-stlco-assessments"
    frames = []
    notes = []
    for yrdir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.isdigit()):
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
        if nfields == 39:
            names = COLS_39
        elif nfields == 37:
            names = COLS_37
        else:
            notes.append(f"{yrdir.name}: unexpected nfields={nfields} delim={delim!r} file={path.name}")
            print(notes[-1], flush=True)
            continue
        print(f"loading {yrdir.name}/{path.name} nfields={nfields} delim={delim!r}", flush=True)
        df = pd.read_csv(
            path, sep=delim, header=None, names=names, usecols=USE, dtype=str,
            encoding="latin1", skip_blank_lines=True, engine="c",
        )
        print(f"read {path.name} rows={len(df)}", flush=True)
        df = df.loc[~df["PARID"].astype(str).str.upper().str.startswith("PARID")].copy()
        df["PARID"] = df["PARID"].astype(str).str.strip()
        df["TAXYR"] = pd.to_numeric(df["TAXYR"], errors="coerce")
        df = df.loc[df["PARID"].isin(sold_parids) & df["TAXYR"].notna(), USE].copy()
        taxyr_mode = int(df["TAXYR"].mode(dropna=True).iloc[0]) if len(df) else None
        frames.append(df)
        msg = f"dwelling folder={yrdir.name} file={path.name} n_sold_match={len(df)} nfields={nfields} taxyr_mode={taxyr_mode}"
        notes.append(msg)
        print(msg, flush=True)
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["PARID", "TAXYR"]).drop_duplicates(["PARID", "TAXYR"], keep="last")
    out.to_parquet(cache, index=False)
    Path(LOG / "stl_dwelling_load.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def metrics(y_log, pred_log, train_log=None, row_ids=None) -> dict:
    m = compute_taxation_metrics(y_log, pred_log, scale="log", y_train=train_log)
    m.update(paper_mechanism_metrics(y_log, pred_log))
    ids = row_ids if row_ids is not None else np.arange(len(y_log))
    try:
        dnl = estimate_delta_nl(y_log, pred_log, row_ids=ids)
        m["Delta_NL"] = float(dnl.get("Delta_NL", np.nan))
    except Exception as e:
        m["Delta_NL"] = np.nan
        m["Delta_NL_error"] = str(e)
    y = np.exp(y_log)
    p = np.exp(pred_log)
    ratio = p / y
    q = pd.qcut(y, 10, labels=False, duplicates="drop")
    m["ratio_by_decile"] = {str(int(i)): float(np.median(ratio[q == i])) for i in np.unique(q)}
    return m


def main() -> int:
    LOG.mkdir(parents=True, exist_ok=True)
    sold = pd.read_csv(RAW / "new data/joined.csv", low_memory=False)
    sold = sold.loc[sold["SALEVAL"].astype(str) == "X"].copy()
    sold["PRICE"] = pd.to_numeric(sold["PRICE"], errors="coerce")
    sold["TAXYR"] = pd.to_numeric(sold["TAXYR"], errors="coerce")
    sold["PARID"] = sold["PARID"].astype(str).str.strip()
    sold = sold.loc[sold["PRICE"].between(10000, 2_000_000) & sold["TAXYR"].notna()].copy()
    print("sold X after price filter", len(sold), flush=True)
    derived = REPO / "data/berry_cmf/derived/st_louis_county_mo"
    derived.mkdir(parents=True, exist_ok=True)
    model_path = derived / "smoke_modeling_table.parquet"
    if model_path.exists():
        print("loading frozen modeling table", model_path, flush=True)
        m = pd.read_parquet(model_path)
        n_sold = len(sold)
        unmatched = n_sold - len(m)
        print("cached modeling rows", len(m), "sold_x", n_sold, flush=True)
    else:
        dwell = load_dwelling(set(sold["PARID"]))
        m = sold.merge(dwell, on=["PARID", "TAXYR"], how="inner")
        print("matched dwelling same year", len(m), "unmatched", len(sold) - len(m), flush=True)
        for c in ["YRBLT", "RMBED", "FIXBATH", "FIXHALF", "SFLA", "STORIES"]:
            m[c] = pd.to_numeric(m[c], errors="coerce")
        m = m.loc[m["SFLA"].between(200, 8000) & m["YRBLT"].between(1800, 2019)].copy()
        m = m.drop_duplicates(["PARID", "TAXYR"], keep="last")
        m["row_id"] = m["PARID"].astype(str) + "|" + m["TAXYR"].astype(int).astype(str)
        m = m.sort_values(["TAXYR", "PARID"]).reset_index(drop=True)
        keep = ["row_id", "PARID", "TAXYR", "PRICE", "SALEVAL", "CLASS", "LUC",
                "SFLA", "YRBLT", "RMBED", "FIXBATH", "FIXHALF", "STORIES", "STYLE", "GRADE", "CDU"]
        m[keep].to_parquet(model_path, index=False)
    for c in ["PRICE", "TAXYR", "YRBLT", "RMBED", "FIXBATH", "FIXHALF", "SFLA", "STORIES"]:
        if c in m.columns:
            m[c] = pd.to_numeric(m[c], errors="coerce")
    year_counts = m["TAXYR"].value_counts().sort_index().to_dict()
    print("year counts", year_counts, flush=True)
    # Prespecified year split (sale day is unavailable in joined.csv).
    split_rule = "development TAXYR<=2016; validation 2017-2018; heldout TAXYR>=2019; year-only because joined.csv dropped SALEDT"
    parts = {
        "development": m.loc[m["TAXYR"] <= 2016].copy(),
        "validation": m.loc[m["TAXYR"].isin([2017, 2018])].copy(),
        "heldout": m.loc[m["TAXYR"] >= 2019].copy(),
    }
    for k, v in parts.items():
        print("split", k, len(v), "years", sorted(v["TAXYR"].unique().tolist()) if len(v) else [], flush=True)
        if len(v) == 0:
            raise RuntimeError(f"empty {k} split")
    n = len(m)
    table_sha = sha256_file(model_path)

    num_cols = ["SFLA", "YRBLT", "RMBED", "FIXBATH", "STORIES"]  # FIXHALF omitted: 47% missing in extract
    cat_cols = ["STYLE", "GRADE", "CDU"]
    print("cat nunique", {c: int(m[c].astype(str).nunique()) for c in cat_cols}, flush=True)
    for d in parts.values():
        for c in cat_cols:
            d[c] = d[c].astype(str).fillna("NA")
    y = {k: np.log(v["PRICE"].to_numpy(dtype=float)) for k, v in parts.items()}
    X = {k: v[num_cols + cat_cols].copy() for k, v in parts.items()}
    ids = {k: v["row_id"].to_numpy() for k, v in parts.items()}
    cat_maps = {c: sorted(X["development"][c].astype(str).unique().tolist()) for c in cat_cols}

    def encode(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in cat_cols:
            out[c] = pd.Categorical(out[c].astype(str), categories=cat_maps[c])
        return out

    miss = {c: float(pd.to_numeric(m[c], errors="coerce").isna().mean()) for c in num_cols}
    print("numeric missing shares", miss, flush=True)
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])
    lr = Pipeline([("pre", pre), ("model", LinearRegression())])
    print("fitting LinearRegression", flush=True)
    lr.fit(X["development"], y["development"])
    print("LinearRegression fit done", flush=True)

    grid = [
        dict(n_estimators=200, learning_rate=0.05, num_leaves=31, min_child_samples=50, random_state=2025, n_jobs=1),
        dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=20, random_state=2025, n_jobs=1),
        dict(n_estimators=400, learning_rate=0.02, num_leaves=31, min_child_samples=100, random_state=2025, n_jobs=1),
    ]
    best = None
    best_rmse = np.inf
    best_model = None
    for hp in grid:
        print("fitting LGBM", hp, flush=True)
        model = LGBMRegressor(objective="rmse", verbose=-1, force_row_wise=True, **hp)
        Xt, Xv = encode(X["development"]), encode(X["validation"])
        model.fit(Xt, y["development"], categorical_feature=cat_cols)
        pred = model.predict(Xv)
        rmse = float(np.sqrt(np.mean((pred - y["validation"]) ** 2)))
        print("lgbm val rmse", hp, rmse, flush=True)
        if rmse < best_rmse:
            best_rmse, best, best_model = rmse, hp, model

    rows = []
    for name, pred_fn in [
        ("LinearRegression", lambda spl: lr.predict(X[spl])),
        ("LGBMRegressor", lambda spl: best_model.predict(encode(X[spl]))),
    ]:
        for spl in ["validation", "heldout"]:
            print(f"metrics {name} {spl} n={len(y[spl])}", flush=True)
            pred = pred_fn(spl)
            met = metrics(y[spl], pred, train_log=y["development"], row_ids=ids[spl])
            rec = {"jurisdiction": "st_louis_county_mo", "model": name, "split": spl,
                   "n": int(len(y[spl])), "n_total_modeled": n,
                   "n_development": int(len(y["development"])),
                   "split_rule": split_rule,
                   "features": ",".join(num_cols + cat_cols),
                   "lgbm_hparams": json.dumps(best) if name.startswith("LGBM") else "",
                   "modeling_table_sha256": table_sha,
                   "r2_price": met.get("R2_price"), "rmse_log": met.get("RMSE_log"),
                   "mae": met.get("MAE"), "mape": met.get("MAPE"),
                   "median_ratio": met.get("Median ratio"), "mean_ratio": met.get("Mean ratio"),
                   "wmean_ratio": met.get("W. Mean ratio"),
                   "COD": met.get("COD"), "PRD": met.get("PRD"), "PRB": met.get("PRB"),
                   "MKI": met.get("MKI"), "VEI": met.get("VEI"),
                   "beta_log": met.get("Beta_log"), "Delta_NL": met.get("Delta_NL"),
                   "dCor": met.get("dCor_e_y"),
                   "ratio_by_decile": json.dumps(met.get("ratio_by_decile")),
                   }
            rows.append(rec)
            print(name, spl, {k: rec[k] for k in ["n", "r2_price", "rmse_log", "PRD", "PRB", "beta_log", "Delta_NL"]}, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "baseline_smoke_metrics.csv", index=False)
    status = [
        {
            "jurisdiction": "st_louis_county_mo",
            "status": "RAN",
            "n_modeled": n,
            "n_sold_x_price_filter": len(sold),
            "match_rate_dwelling": n / max(len(sold), 1),
            "heldout_used_for_selection": False,
            "modeling_table_sha256": table_sha,
            "year_counts": json.dumps({str(int(k)): int(v) for k, v in year_counts.items()}),
            "numeric_missing_share": json.dumps(miss),
            "note": "Year-only sale dates because joined.csv dropped SALEDT. Dwelling headers ignored; 37/39-field dictionary mapping. 2012 folder contains TAXYR=2013 rows so 2012 sales have no same-year dwelling snapshot. LR uses development-median imputation; LGBM uses native NaN handling.",
        },
        {"jurisdiction": "detroit_mi", "status": "NOT_RUN_FAILED_GATE", "note": "A_EXTERNAL_REGRESSIVITY_ONLY; TIMING_UNRESOLVED characteristics"},
        {"jurisdiction": "philadelphia_pa", "status": "NOT_RUN_FAILED_GATE", "note": "C_PUBLIC_ENRICHMENT_CANDIDATE; public history not acquired this pass"},
        {"jurisdiction": "orleans_la", "status": "NOT_RUN_FAILED_GATE", "note": "A_EXTERNAL_REGRESSIVITY_ONLY; scrape leakage"},
        {"jurisdiction": "franklin_oh", "status": "NOT_RUN_FAILED_GATE", "note": "A_EXTERNAL_REGRESSIVITY_ONLY; no structural predictors"},
        {"jurisdiction": "cook_il", "status": "NOT_RUN_FAILED_GATE", "note": "CMF 2002-2015 is not the canonical CCAO AVM sample; do not replace CCAO experiment"},
    ]
    pd.DataFrame(status).to_csv(OUT / "baseline_smoke_status.csv", index=False)
    print("wrote smoke metrics", n, "sha", table_sha, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
