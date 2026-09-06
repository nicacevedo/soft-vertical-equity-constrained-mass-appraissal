#!/usr/bin/env python3
"""P0-2 -- native vs custom rho=0 parity, in TWO STRICTLY SEPARATE TRACKS.

Track H (historical settings)  -- the only track that speaks to the frozen artifacts.
Track P (pinned determinism)   -- an implementation statement ONLY.

A Track-P result may never be cited as validating a historical positive-rho artifact.

Cells (fixed names; never abbreviate, never substitute):
  A  "Ordinary LightGBM (standard raw-label native)"  native mse, raw y, boost_from_average=True
  B  "Parity-aligned native L2"                       native mse, centered y, bfa=False, init_score=0
  C  "Custom rho=0 origin"                            LGBCovPenalty / LGBSmoothPenalty at rho=0

Modes
-----
  --mode ladder      A/B/C x capacity {T,M,F} x split {heldout, forward_2025}
  --mode reproduce   re-fit the 7 named frozen configs and compare to cached predictions

Tiers
-----
  parity_tier      T1..T4   (A/B/C comparisons)          -- separate column
  reproduction_tier R1..R4  (frozen-artifact reproduction) -- separate column, separate table
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p0_common as c

import lightgbm as lgb

CAPACITIES = {
    "T": {"n_estimators": 60, "num_leaves": 15, "max_depth": 4},
    "M": {"n_estimators": 200, "num_leaves": 63, "max_depth": 8},
    "F": {},   # frozen 994 / 573 / 11 -- no override
}
PIN_PARAMS = {"deterministic": True, "force_row_wise": True, "num_threads": 1}

# (label, family, rho, config_id) for the frozen-artifact reproduction sample.
REPRODUCE_SAMPLE = [
    ("cellA_native", "native", None, "252a25d9c0ce796b"),
    ("direct_rho0", "direct", 0.0, "1fb838f7d6bfda88"),
    ("surrogate_rho0", "surrogate", 0.0, "5b7875e55e58ac62"),
    ("direct_rho_0p954095", "direct", 0.954095476349994, "e732a8e35bd1a796"),
    ("surrogate_rho_0p954095", "surrogate", 0.954095476349994, "4a39ef84979943f2"),
    ("direct_rho100", "direct", 100.0, "a72f544c5161823f"),
    ("surrogate_rho100", "surrogate", 100.0, "d5704816049673ad"),
]

METRIC_KEYS = ["R2_price", "MAE_price", "MAPE", "RMSE_log", "Median ratio", "Mean ratio",
               "W. Mean ratio", "COD", "COV_IAAO", "PRD", "PRB", "MKI", "VEI"]


# --------------------------------------------------------------------------- models
def build_params(track: str, capacity: str) -> dict:
    p = dict(c.frozen_lgbm_config()["lgbm_params"])
    p.update(CAPACITIES[capacity])
    if track == "pinned":
        p.update(PIN_PARAMS)
    return p


def fit_predict_cell(cell: str, params: dict, X_tr, y_tr_log, X_te, family: str = "direct",
                     rho: float = 0.0):
    """Return log-scale predictions on X_te for the requested cell."""
    from soft_constrained_models.boosting_models import LGBCovPenalty, LGBSmoothPenalty
    from run_temporal_cv import _native_lgbm_estimator

    if cell == "A":
        est = _native_lgbm_estimator(dict(params))
        est.fit(X_tr, y_tr_log)
        return np.asarray(est.predict(X_te), dtype=float).reshape(-1), est

    if cell == "B":
        # Parity-aligned native L2: same native objective, centered labels, zero init.
        base = float(np.mean(y_tr_log))
        y_c = y_tr_log - base
        p = {k: v for k, v in dict(params).items()
             if k not in {"early_stopping_rounds", "early_stopping_round"}}
        try:
            est = lgb.LGBMRegressor(boost_from_average=False, early_stopping_rounds=None, **p)
        except TypeError:
            est = lgb.LGBMRegressor(boost_from_average=False, **p)
        est.fit(X_tr, y_c, init_score=np.zeros(y_c.shape[0], dtype=float))
        return np.asarray(est.predict(X_te), dtype=float).reshape(-1) + base, est

    if cell == "C":
        if family == "direct":
            m = LGBCovPenalty(rho=float(rho), ratio_mode="diff", match_native_init=True,
                              zero_grad_tol=1e-12, early_stopping_rounds=None,
                              lgbm_params=dict(params), verbose=False)
        else:
            m = LGBSmoothPenalty(rho=float(rho), ratio_mode="diff",
                                 weighting_proxy_mode="identity", match_native_init=True,
                                 zero_grad_tol=1e-12, early_stopping_rounds=None,
                                 lgbm_params=dict(params), verbose=False)
        m.fit(X_tr, y_tr_log)
        return np.asarray(m.predict(X_te), dtype=float).reshape(-1), m

    raise ValueError(cell)


# --------------------------------------------------------------------------- tiers
def delta_stats(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))
    return {
        "n": int(d.size),
        "mean_abs_delta_log": float(np.mean(d)),
        "median_abs_delta_log": float(np.median(d)),
        "p95_abs_delta_log": float(np.percentile(d, 95)),
        "max_abs_delta_log": float(np.max(d)),
        "pearson": float(np.corrcoef(a, b)[0, 1]) if d.size > 1 else float("nan"),
        "n_exact_equal": int(np.sum(d == 0.0)),
        "frac_exact_equal": float(np.mean(d == 0.0)),
    }


def metrics_from(y_true_log, y_pred_log, y_train_log) -> dict:
    """Canonical metric suite via the SAME function the pipeline used
    (utils.motivation_utils._compute_extended_metrics with ratio_mode='diff'),
    so every number is directly comparable to the frozen artifacts."""
    from utils.motivation_utils import _compute_extended_metrics, paper_mechanism_metrics
    m = _compute_extended_metrics(
        y_true_log=np.asarray(y_true_log, dtype=float),
        y_pred_log=np.asarray(y_pred_log, dtype=float),
        y_train_log=np.asarray(y_train_log, dtype=float),
        ratio_mode="diff",
    )
    out = {}
    for k in METRIC_KEYS:
        if k in m:
            try:
                out[k] = float(m[k])
            except (TypeError, ValueError):
                pass
    mech = paper_mechanism_metrics(np.asarray(y_true_log, dtype=float),
                                   np.asarray(y_pred_log, dtype=float))
    out["Beta_log"] = float(mech["Beta_log"])
    out["Cov_log_residual_log_price"] = float(mech["Cov_log_residual_log_price"])
    if "RMSE_log" not in out:
        out["RMSE_log"] = float(np.sqrt(np.mean(
            (np.asarray(y_pred_log, float) - np.asarray(y_true_log, float)) ** 2)))
    return out


def metrics_agree(m1: dict, m2: dict, digits: int = 3) -> tuple:
    """Agreement 'at displayed precision' -- 3 decimals for ratios/indices, 0 for prices."""
    disagree = []
    for k in set(m1) & set(m2):
        v1, v2 = m1[k], m2[k]
        if not (np.isfinite(v1) and np.isfinite(v2)):
            continue
        nd = 0 if ("MAE" in k or "RMSE_price" in k) else digits
        if round(float(v1), nd) != round(float(v2), nd):
            disagree.append(k)
    return (len(disagree) == 0), sorted(disagree)


def parity_tier(stats: dict, agree: bool, named_cause: str | None) -> str:
    mx = stats["max_abs_delta_log"]
    if mx == 0.0:
        return "T1"
    if mx <= 1e-6 and agree:
        return "T2"
    if mx <= 1e-3 and agree and named_cause:
        return "T3"
    return "T4"


def reproduction_tier(stats: dict, agree: bool, named_cause: str | None) -> str:
    mx = stats["max_abs_delta_log"]
    if mx == 0.0:
        return "R1"
    if mx <= 1e-6 and agree:
        return "R2"
    if mx <= 1e-3 and agree and named_cause:
        return "R3"
    return "R4"


# --------------------------------------------------------------------------- data
def prepare_frames():
    df_tv, df_test, df_assess, pred_cols, cat_cols = c.load_canonical_splits()
    production = pd.concat([df_tv, df_test], ignore_index=True)

    def _xy(train_df, test_df):
        X_tr = train_df[pred_cols].copy()
        X_te = test_df[pred_cols].copy()
        cc = [k for k in cat_cols if k in X_tr.columns]
        for k in cc:
            X_tr[k] = X_tr[k].astype("category")
            X_te[k] = X_te[k].astype("category")
        y_tr = np.log(train_df[c.TARGET_COL].to_numpy())
        y_te = np.log(test_df[c.TARGET_COL].to_numpy())
        return X_tr, y_tr, X_te, y_te

    return {
        "heldout": _xy(df_tv, df_test),
        "forward_2025": _xy(production, df_assess),
    }, df_tv, df_test, df_assess


# --------------------------------------------------------------------------- runners
def run_ladder(track: str, capacities: list, splits: list) -> int:
    frames, *_ = prepare_frames()
    rows = []
    for split in splits:
        X_tr, y_tr, X_te, y_te = frames[split]
        for cap in capacities:
            params = build_params(track, cap)
            preds = {}
            for cell in ("A", "B", "C"):
                t0 = time.perf_counter()
                p, _est = fit_predict_cell(cell, params, X_tr, y_tr, X_te,
                                           family="direct", rho=0.0)
                preds[cell] = p
                print(f"[{track}/{cap}/{split}] cell {cell} fit+predict "
                      f"{time.perf_counter()-t0:.1f}s", flush=True)
            # Surrogate rho=0 as an extra Cell-C check (must equal Direct rho=0)
            p_surr, _ = fit_predict_cell("C", params, X_tr, y_tr, X_te,
                                         family="surrogate", rho=0.0)
            preds["C_surrogate"] = p_surr

            mets = {k: metrics_from(y_te, v, y_tr) for k, v in preds.items()}
            for a, b in (("A", "B"), ("B", "C"), ("A", "C"),
                         ("C", "C_surrogate")):
                st = delta_stats(preds[a], preds[b])
                agree, dis = metrics_agree(mets[a], mets[b])
                cause = None
                if {a, b} == {"A", "B"} or {a, b} == {"A", "C"}:
                    cause = ("float32 label representation: Cell A is fed float32(y) with "
                             "y~12.4 (ULP 9.5e-7) while Cells B/C are fed float32(y-ybar) "
                             "(ULP 1.2e-7)") if st["max_abs_delta_log"] <= 1e-3 else None
                rows.append({
                    "track": track, "capacity": cap, "split": split,
                    "cell_pair": f"{a}<->{b}",
                    "cell_a_name": {"A": c.CELL_A_NAME, "B": c.CELL_B_NAME,
                                    "C": c.CELL_C_NAME, "C_surrogate": c.CELL_C_NAME + " (Surrogate)"}[a],
                    "cell_b_name": {"A": c.CELL_A_NAME, "B": c.CELL_B_NAME,
                                    "C": c.CELL_C_NAME, "C_surrogate": c.CELL_C_NAME + " (Surrogate)"}[b],
                    **st,
                    "metrics_agree_at_displayed_precision": agree,
                    "metrics_disagreeing": ";".join(dis),
                    "named_cause": cause or "",
                    "parity_tier": parity_tier(st, agree, cause),
                    "n_estimators": params.get("n_estimators"),
                    "num_leaves": params.get("num_leaves"),
                    "max_depth": params.get("max_depth"),
                    "deterministic": params.get("deterministic", False),
                    "force_row_wise": params.get("force_row_wise", False),
                    "num_threads": params.get("num_threads", None),
                })
            for k, m in mets.items():
                rows.append({"track": track, "capacity": cap, "split": split,
                             "cell_pair": f"METRICS:{k}", "cell_a_name": k, "cell_b_name": "",
                             **{f"metric_{kk}": vv for kk, vv in m.items()}})
    df = pd.DataFrame(rows)
    out = c.TABLES / f"parity_ladder_{track}.csv"
    c.write_table(df, out)
    print(f"[done] wrote {out}")
    return 0


def run_reproduce(splits: list) -> int:
    """Track H only: re-fit named frozen configs and compare to cached predictions."""
    cfgmap = pd.read_csv(c.CONFIGS / "frozen_config_map.csv")
    frames, *_ = prepare_frames()
    params = build_params("historical", "F")
    rows = []
    for split in splits:
        X_tr, y_tr, X_te, y_te = frames[split]
        for label, family, rho, cfg_id in REPRODUCE_SAMPLE:
            sel = cfgmap[(cfgmap.stage == split) & (cfgmap.config_id == cfg_id)]
            if sel.empty:
                raise c.ProtocolViolation(f"no cached artifact for {cfg_id} / {split}")
            pred_file = sel.iloc[0]["pred_file"]
            cached = pd.read_parquet(pred_file)
            t0 = time.perf_counter()
            if family == "native":
                p, _ = fit_predict_cell("A", params, X_tr, y_tr, X_te)
            else:
                p, _ = fit_predict_cell("C", params, X_tr, y_tr, X_te, family=family, rho=rho)
            el = time.perf_counter() - t0
            cached_pred = cached["y_pred_log"].to_numpy(dtype=float)
            if len(cached_pred) != len(p):
                raise c.ProtocolViolation(
                    f"{label}/{split}: cached n={len(cached_pred)} vs refit n={len(p)}")
            if not np.allclose(cached["y_true_log"].to_numpy(dtype=float), y_te,
                               rtol=0, atol=0):
                raise c.ProtocolViolation(f"{label}/{split}: y_true_log alignment failed")
            st = delta_stats(cached_pred, p)
            m_cached = metrics_from(y_te, cached_pred, y_tr)
            m_refit = metrics_from(y_te, p, y_tr)
            agree, dis = metrics_agree(m_cached, m_refit)
            cause = None
            if st["max_abs_delta_log"] <= 1e-3 and st["max_abs_delta_log"] > 1e-6:
                cause = ("same committed source (objective/split/metric files byte-identical "
                         "at the provenance commits); residual bounded by F-DIRTY "
                         "unreconstructable dirty-state uncertainty")
            rtier = reproduction_tier(st, agree, cause)
            if rtier == "R1":
                fclass = "none"
            elif rtier in ("R2", "R3"):
                fclass = "F-DIRTY"
            else:
                fclass = "PENDING_CLASSIFICATION"
            rows.append({
                "label": label, "family": family, "rho": rho, "config_id": cfg_id,
                "split": split, "cached_pred_file": pred_file, **st,
                "metrics_agree_at_displayed_precision": agree,
                "metrics_disagreeing": ";".join(dis),
                "reproduction_tier": rtier, "failure_class": fclass,
                "named_cause": cause or "",
                "fit_seconds": el, "track": "historical",
                **{f"cached_{k}": v for k, v in m_cached.items()},
                **{f"refit_{k}": v for k, v in m_refit.items()},
            })
            print(f"[reproduce/{split}] {label:<24s} max|d|={st['max_abs_delta_log']:.3e} "
                  f"mean={st['mean_abs_delta_log']:.3e} tier={rtier} ({el:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    c.write_table(df, c.TABLES / "frozen_artifact_reproduction.csv")
    print("[done] wrote frozen_artifact_reproduction.csv")
    return 0


def run_replicate(out_suffix: str = "") -> int:
    """F-NUM test: Cell A twice under identical historical settings, same host."""
    frames, *_ = prepare_frames()
    params = build_params("historical", "F")
    rows = []
    for split in ("heldout", "forward_2025"):
        X_tr, y_tr, X_te, y_te = frames[split]
        p1, _ = fit_predict_cell("A", params, X_tr, y_tr, X_te)
        p2, _ = fit_predict_cell("A", params, X_tr, y_tr, X_te)
        st = delta_stats(p1, p2)
        m1, m2 = metrics_from(y_te, p1, y_tr), metrics_from(y_te, p2, y_tr)
        agree, dis = metrics_agree(m1, m2)
        rows.append({"split": split, "test": "cellA_same_host_replicate", "track": "historical",
                     **st, "metrics_agree_at_displayed_precision": agree,
                     "metrics_disagreeing": ";".join(dis),
                     "f_num_floor_max_abs_delta_log": st["max_abs_delta_log"],
                     "interpretation": ("run-to-run numerical nondeterminism floor under "
                                        "identical historical settings on one host")})
        print(f"[replicate/{split}] max|d|={st['max_abs_delta_log']:.3e} "
              f"exact_equal_frac={st['frac_exact_equal']:.6f}", flush=True)
    c.write_table(pd.DataFrame(rows), c.TABLES / f"fnum_same_host_replicate{out_suffix}.csv")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["ladder", "reproduce", "replicate"])
    ap.add_argument("--track", default="historical", choices=["historical", "pinned"])
    ap.add_argument("--capacities", default="T,M,F")
    ap.add_argument("--splits", default="heldout,forward_2025")
    ap.add_argument("--out-suffix", default="")
    a = ap.parse_args()
    caps = [x.strip() for x in a.capacities.split(",") if x.strip()]
    splits = [x.strip() for x in a.splits.split(",") if x.strip()]
    print(f"[p0-2] mode={a.mode} track={a.track} capacities={caps} splits={splits}", flush=True)
    print(f"[p0-2] env={c.thread_env()}", flush=True)
    if a.mode == "ladder":
        return run_ladder(a.track, caps, splits)
    if a.mode == "reproduce":
        return run_reproduce(splits)
    return run_replicate(a.out_suffix)


if __name__ == "__main__":
    raise SystemExit(main())
