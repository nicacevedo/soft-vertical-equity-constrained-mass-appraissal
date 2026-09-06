#!/usr/bin/env python3
"""Stage 1.5 -- the 27 canonical zero-reference fits: A + B + C across 9 fitting blocks.

HISTORICAL execution settings only.  No Track-P deterministic pins.

Cells (post-G1 forward names, configs/post_g1_reference_convention.yaml):
  A  "Ordinary LightGBM (standard raw-label native)"   assessor-facing benchmark
  B  "Centered-label native L2 (initialization-aligned)" implementation-decomposition control
     (legacy_stage1_label = "Parity-aligned native L2")
  C  "Custom-objective rho=0 origin"                    PRIMARY within-path penalty-isolating ref

Cell C uses ONE canonical implementation -- Direct rho=0 (LGBCovPenalty) -- recorded explicitly.
Stage 1 established Direct-rho0 == Surrogate-rho0 bitwise inside the custom path (T1).

Blocks: fold 1..7 training blocks (paired eval = that fold's validation block),
        full development pool (paired eval = held-out),
        full 2016-2024 production block (paired eval = 2025 forward).

Retains per fit: in-sample training log predictions, paired evaluation log predictions,
row_id, y_true_log, prediction hashes, config hash, block/split hash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p0_common as c

import lightgbm as lgb

OUT = c.P0_OUTPUT_ROOT / "zero_reference_fits"

CELLS = {
    "A": {"forward_display_name": "Ordinary LightGBM (standard raw-label native)",
          "role": "assessor-facing / workflow benchmark",
          "implementation": "run_temporal_cv._native_lgbm_estimator, raw labels, boost_from_average default"},
    "B": {"forward_display_name": "Centered-label native L2 (initialization-aligned)",
          "legacy_stage1_label": "Parity-aligned native L2",
          "role": "implementation-decomposition control only",
          "implementation": "lgb.LGBMRegressor(objective=mse, boost_from_average=False), centered labels, init_score=0, ybar added back"},
    "C": {"forward_display_name": "Custom-objective rho=0 origin",
          "role": "PRIMARY within-path penalty-isolating reference",
          "implementation": "LGBCovPenalty(rho=0, ratio_mode='diff', match_native_init=True)  [Direct rho=0]"},
}
CELL_C_IMPL = "LGBCovPenalty (Direct rho=0)"


def _hash_arr(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()


def _hash_idx(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.int64).tobytes()).hexdigest()


def fit_cell(cell: str, params: dict, X_tr, y_tr_log, X_ev):
    from soft_constrained_models.boosting_models import LGBCovPenalty
    from run_temporal_cv import _native_lgbm_estimator

    if cell == "A":
        est = _native_lgbm_estimator(dict(params))
        est.fit(X_tr, y_tr_log)
        return (np.asarray(est.predict(X_tr), dtype=float).reshape(-1),
                np.asarray(est.predict(X_ev), dtype=float).reshape(-1))
    if cell == "B":
        base = float(np.mean(y_tr_log))
        yc = y_tr_log - base
        p = {k: v for k, v in dict(params).items()
             if k not in {"early_stopping_rounds", "early_stopping_round"}}
        try:
            est = lgb.LGBMRegressor(boost_from_average=False, early_stopping_rounds=None, **p)
        except TypeError:
            est = lgb.LGBMRegressor(boost_from_average=False, **p)
        est.fit(X_tr, yc, init_score=np.zeros(yc.shape[0], dtype=float))
        return (np.asarray(est.predict(X_tr), dtype=float).reshape(-1) + base,
                np.asarray(est.predict(X_ev), dtype=float).reshape(-1) + base)
    if cell == "C":
        m = LGBCovPenalty(rho=0.0, ratio_mode="diff", match_native_init=True,
                          zero_grad_tol=1e-12, early_stopping_rounds=None,
                          lgbm_params=dict(params), verbose=False)
        m.fit(X_tr, y_tr_log)
        return (np.asarray(m.predict(X_tr), dtype=float).reshape(-1),
                np.asarray(m.predict(X_ev), dtype=float).reshape(-1))
    raise ValueError(cell)


def diagnostics(y: np.ndarray, f0: np.ndarray) -> dict:
    """Paper conventions: population moments (ddof=0); c = y - ybar."""
    n = int(y.size)
    ybar = float(np.mean(y)); f0bar = float(np.mean(f0))
    cy = y - ybar
    var_y = float(np.mean(cy ** 2))
    var_f0 = float(np.mean((f0 - f0bar) ** 2))
    cov_f0_y = float(np.mean((f0 - f0bar) * cy))
    e = f0 - y
    cov_e_y = float(np.mean(e * cy))                 # Cov_T(f0-y, y), paper convention
    beta_log_train = cov_e_y / var_y if var_y > 0 else float("nan")
    sse = float(np.sum((f0 - y) ** 2)); sst = float(np.sum(cy ** 2))
    r2_log = 1.0 - sse / sst if sst > 0 else float("nan")
    b_star = var_y / cov_f0_y if cov_f0_y > 0 else float("nan")
    return {
        "n_T": n, "ybar_T": ybar, "f0bar_T": f0bar, "f0bar_minus_ybar": f0bar - ybar,
        "Var_T_y_ddof0": var_y, "Var_T_f0_ddof0": var_f0,
        "Cov_T_f0_y": cov_f0_y, "Cov_T_f0_minus_y_y": cov_e_y,
        "beta_log_train": beta_log_train,
        "R2_log_insample": r2_log,
        "one_over_R2_log_theoretical_diagnostic_only": (1.0 / r2_log if r2_log not in (0.0,) and np.isfinite(r2_log) else float("nan")),
        "b_star_train": b_star,
        "cov_positive": bool(cov_f0_y > 0),
        "b_star_finite": bool(np.isfinite(b_star)),
        # internal identity check: beta_log = 1/b_star - 1  (since cov_e_y = cov_f0_y - var_y)
        "identity_beta_log_vs_bstar_absdiff": float(abs(beta_log_train - (1.0 / b_star - 1.0)))
        if (np.isfinite(b_star) and b_star != 0) else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block-index", type=int, default=-1,
                    help="0..8 to run one block; -1 for all nine")
    ap.add_argument("--cells", default="A,B,C")
    a = ap.parse_args()
    cells = [x.strip() for x in a.cells.split(",") if x.strip()]

    print(f"[zref] env={c.thread_env()}", flush=True)
    df_tv, df_test, df_assess, pred_cols, cat_cols = c.load_canonical_splits()
    folds, archive = c.rebuild_folds(df_tv)
    production = pd.concat([df_tv, df_test], ignore_index=True)
    params = dict(c.frozen_lgbm_config()["lgbm_params"])
    cfg_hash = c.frozen_lgbm_config()["lgbm_params_sha256"]

    def prep(train_df, eval_df):
        Xtr = train_df[pred_cols].copy(); Xev = eval_df[pred_cols].copy()
        for k in [k for k in cat_cols if k in Xtr.columns]:
            Xtr[k] = Xtr[k].astype("category"); Xev[k] = Xev[k].astype("category")
        return (Xtr, np.log(train_df[c.TARGET_COL].to_numpy()),
                Xev, np.log(eval_df[c.TARGET_COL].to_numpy()))

    blocks = []
    for rec in folds:
        tr = np.asarray(rec["train_indices"], dtype=int)
        va = np.asarray(rec["val_indices"], dtype=int)
        blocks.append({
            "block_id": f"fold_{int(rec['fold_id'])+1}_train", "block_kind": "cv_fold_train",
            "fold_1based": int(rec["fold_id"]) + 1,
            "eval_name": f"fold_{int(rec['fold_id'])+1}_val", "eval_kind": "cv_validation",
            "train_df": df_tv.iloc[tr], "eval_df": df_tv.iloc[va],
            "train_row_id": tr, "eval_row_id": va,
            "train_index_hash": rec["train_index_hash"], "eval_index_hash": rec["val_index_hash"],
        })
    blocks.append({
        "block_id": "development_pool", "block_kind": "development_pool", "fold_1based": None,
        "eval_name": "heldout", "eval_kind": "out_of_time",
        "train_df": df_tv, "eval_df": df_test,
        "train_row_id": np.arange(len(df_tv)), "eval_row_id": np.arange(len(df_test)),
        "train_index_hash": _hash_idx(np.arange(len(df_tv))),
        "eval_index_hash": _hash_idx(np.arange(len(df_test))),
    })
    blocks.append({
        "block_id": "production_2016_2024", "block_kind": "production_block", "fold_1based": None,
        "eval_name": "forward_2025", "eval_kind": "out_of_time",
        "train_df": production, "eval_df": df_assess,
        "train_row_id": np.arange(len(production)), "eval_row_id": np.arange(len(df_assess)),
        "train_index_hash": _hash_idx(np.arange(len(production))),
        "eval_index_hash": _hash_idx(np.arange(len(df_assess))),
    })
    if a.block_index >= 0:
        blocks = [blocks[a.block_index]]

    rows = []
    for blk in blocks:
        Xtr, ytr, Xev, yev = prep(blk["train_df"], blk["eval_df"])
        for cell in cells:
            t0 = time.perf_counter()
            f0_tr, f0_ev = fit_cell(cell, params, Xtr, ytr, Xev)
            el = time.perf_counter() - t0
            d = diagnostics(ytr, f0_tr)

            odir = OUT / f"cell={cell}" / f"block={blk['block_id']}"
            odir.mkdir(parents=True, exist_ok=True)
            tr_df = pd.DataFrame({"row_id": blk["train_row_id"], "y_true_log": ytr,
                                  "y_pred_log": f0_tr})
            ev_df = pd.DataFrame({"row_id": blk["eval_row_id"], "y_true_log": yev,
                                  "y_pred_log": f0_ev})
            tr_df.to_parquet(odir / "train_predictions.parquet", index=False)
            ev_df.to_parquet(odir / "eval_predictions.parquet", index=False)

            meta = {
                "cell_id": cell, **CELLS[cell],
                "cell_c_implementation": CELL_C_IMPL if cell == "C" else None,
                "block_id": blk["block_id"], "block_kind": blk["block_kind"],
                "fold_1based": blk["fold_1based"],
                "eval_name": blk["eval_name"], "eval_kind": blk["eval_kind"],
                "n_train": int(len(ytr)), "n_eval": int(len(yev)),
                "lgbm_params_sha256": cfg_hash,
                "train_index_hash": blk["train_index_hash"],
                "eval_index_hash": blk["eval_index_hash"],
                "train_pred_sha256": _hash_arr(f0_tr),
                "eval_pred_sha256": _hash_arr(f0_ev),
                "y_train_log_sha256": _hash_arr(ytr),
                "y_eval_log_sha256": _hash_arr(yev),
                "execution_settings": "HISTORICAL (no deterministic/force_row_wise/num_threads pins)",
                "fit_seconds": el,
                "train_predictions": str((odir / "train_predictions.parquet").relative_to(c.REPO)),
                "eval_predictions": str((odir / "eval_predictions.parquet").relative_to(c.REPO)),
                **d,
            }
            c.write_json(odir / "fit_meta.json", meta)
            rows.append(meta)
            print(f"[zref] {cell} {blk['block_id']:<24s} n={len(ytr):>7d} "
                  f"b*={d['b_star_train']:.6f} cov={d['Cov_T_f0_y']:.6f} "
                  f"f0bar-ybar={d['f0bar_minus_ybar']:+.3e} ({el:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    tag = "" if a.block_index < 0 else f"_block{a.block_index}"
    c.write_table(df, c.TABLES / f"zero_reference_fits{tag}.csv")
    print(f"[zref] wrote zero_reference_fits{tag}.csv  ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
