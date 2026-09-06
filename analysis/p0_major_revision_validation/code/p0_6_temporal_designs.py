#!/usr/bin/env python3
"""Stage-3 temporal robustness: D-SNAP (strict-date) and D-PURGE (ORACLE repeat-parcel).

D-SNAP   snap-back at all 8 chronological boundaries so max(train_date) < min(eval_date).
D-PURGE  ORACLE DIAGNOSTIC -- evaluation sets preserved bitwise; from each TRAINING block
         every row whose meta_pin appears in the paired evaluation block is removed.
         NOT prospectively implementable and NOT a proposed CCAO split.
D-UNSEEN no fits; a row subset of frozen cached predictions (handled in the assembler).

Canonical split utilities are IMPORTED and wrapped, never modified in place.
Historical 994-tree settings; no Track-P determinism pins.

Modes
-----
  protocols  build + hash configs/split_protocol_{dsnap,dpurge}.json, robustness_rho_grid.json
  fit        --shard S : fit one (design, block, family) cell of the screening grid
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

OUT = c.P0_OUTPUT_ROOT / "temporal_robustness"
IDX = OUT / "protocols"


def _save_idx(name: str, arr: np.ndarray) -> dict:
    """Bulky index arrays live untracked under output/, hashed and indexed."""
    IDX.mkdir(parents=True, exist_ok=True)
    p = IDX / f"{name}.npy"
    np.save(p, np.ascontiguousarray(arr, dtype=np.int64))
    return {"npy": str(p.relative_to(c.REPO)), "n": int(arr.size), "sha256": _hash_idx(arr)}


def _load_idx(meta: dict) -> np.ndarray:
    a = np.load(c.REPO / meta["npy"])
    if _hash_idx(a) != meta["sha256"]:
        raise c.ProtocolViolation(f"index hash mismatch for {meta['npy']}")
    return a
DESIGNS = ["dsnap", "dpurge"]
BLOCKS = [f"fold_{k}" for k in range(1, 8)] + ["heldout", "forward_2025"]
FAMILIES = ["native", "direct", "surrogate"]

DISPLAY_ANCHORS = [0.0104811313415468, 0.1, 0.954095476349994, 10.481131341546853, 100.0]
CANDIDATE_ENDPOINTS = [0.2023589647725157, 0.3556480306223128,
                       2.2229964825261943, 2.559547922699543]


def _hash_arr(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()


def _hash_idx(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.int64).tobytes()).hexdigest()


# ------------------------------------------------------------------ rho grid
def screening_grid() -> dict:
    full = c.frozen_rho_grid()          # [0.0] + 82 positive
    pos = np.asarray(full[1:], dtype=float)
    every4 = pos[::4]                                      # 21 points
    forced = np.asarray(DISPLAY_ANCHORS + CANDIDATE_ENDPOINTS, dtype=float)
    keep = [pos[int(np.argmin(np.abs(pos - f)))] for f in forced]
    grid = np.unique(np.concatenate([every4, np.asarray(keep)]))
    return {
        "construction": ("frozen 82-point positive grid: rho=0 plus every 4th grid index "
                         "(21 points), force-including the five display anchors and the four "
                         "candidate-region endpoints, then deduplicated. Exact scientific "
                         "anchors outrank an arbitrary row count."),
        "n_every_4th": int(every4.size),
        "n_positive_screening": int(grid.size),
        "forced_display_anchors": DISPLAY_ANCHORS,
        "forced_candidate_endpoints": CANDIDATE_ENDPOINTS,
        "positive_rhos": [float(x) for x in grid],
        "includes_rho_zero": True,
        "n_configs_per_design": int(1 + 2 * grid.size),
    }


# --------------------------------------------------------------- PIN loading
def load_pins(df_tv, df_test, df_assess):
    """meta_pin is not loaded by the pipeline; read it with the SAME pyarrow pushdown
    filters and the same ordering, then verify alignment on date and price."""
    import pyarrow.parquet as pq
    tbl = pq.read_table(str(c.DATA_PATH), columns=[c.PIN_COL, c.DATE_COL, c.TARGET_COL],
                        filters=[("ind_pin_is_multicard", "==", False),
                                 ("sv_is_outlier", "==", False)])
    raw = tbl.to_pandas()
    raw[c.DATE_COL] = pd.to_datetime(raw[c.DATE_COL])
    ordered = raw.sort_values([c.DATE_COL], kind="mergesort").reset_index(drop=True)
    d = ordered[c.DATE_COL]
    universe = ordered.loc[(d >= "2016-01-01") & (d <= "2024-12-31")].reset_index(drop=True)
    assess = ordered.loc[d.dt.year == 2025].reset_index(drop=True)
    split_idx = int(0.9 * len(universe))
    dev = universe.iloc[:split_idx].reset_index(drop=True)
    test = universe.iloc[split_idx:].reset_index(drop=True)
    for name, a, b in (("development", dev, df_tv), ("heldout", test, df_test),
                       ("forward_2025", assess, df_assess)):
        if not np.array_equal(a[c.DATE_COL].to_numpy(),
                              pd.to_datetime(b[c.DATE_COL]).to_numpy()):
            raise c.ProtocolViolation(f"{name}: PIN join date alignment failed")
        if not np.allclose(a[c.TARGET_COL].to_numpy(float), b[c.TARGET_COL].to_numpy(float),
                           rtol=0, atol=0):
            raise c.ProtocolViolation(f"{name}: PIN join price alignment failed")
    all_pins = pd.concat([dev[c.PIN_COL], test[c.PIN_COL], assess[c.PIN_COL]],
                         ignore_index=True)
    codes, uniques = pd.factorize(all_pins, sort=False)
    n1, n2 = len(dev), len(test)
    return codes[:n1], codes[n1:n1 + n2], codes[n1 + n2:], int(len(uniques))


# ------------------------------------------------------------------ D-SNAP
def snap_back(dates: np.ndarray, cut: int) -> int:
    """Move the positional cut BACK to the first row sharing the boundary date, so all
    boundary-date rows land on the evaluation side and max(train) < min(eval) strictly."""
    d_b = dates[cut]
    new = cut
    while new > 0 and dates[new - 1] == d_b:
        new -= 1
    return new


def build_protocols() -> int:
    df_tv, df_test, df_assess, pred_cols, cat_cols = c.load_canonical_splits()
    folds, archive = c.rebuild_folds(df_tv)
    dates_dev = pd.to_datetime(df_tv[c.DATE_COL]).to_numpy()
    dates_test = pd.to_datetime(df_test[c.DATE_COL]).to_numpy()
    dates_as = pd.to_datetime(df_assess[c.DATE_COL]).to_numpy()
    pin_dev, pin_test, pin_as, n_pins = load_pins(df_tv, df_test, df_assess)
    pin_prod = np.concatenate([pin_dev, pin_test])

    # ---- D-SNAP -----------------------------------------------------------
    snap_folds, snap_rows = [], []
    for rec in folds:
        tr = np.asarray(rec["train_indices"], dtype=int)
        va = np.asarray(rec["val_indices"], dtype=int)
        cum = np.concatenate([tr, va])                      # positional, date-ordered
        cut = tr.size
        new_cut = snap_back(dates_dev[cum], cut)
        ntr, nva = cum[:new_cut], cum[new_cut:]
        assert dates_dev[ntr].max() < dates_dev[nva].min(), "snap-back failed"
        k1 = int(rec["fold_id"]) + 1
        snap_folds.append({"fold_1based": k1,
                           "train_idx": _save_idx(f"dsnap_fold{k1}_train", ntr),
                           "val_idx": _save_idx(f"dsnap_fold{k1}_val", nva),
                           "train_size": int(ntr.size), "val_size": int(nva.size),
                           "train_index_hash": _hash_idx(ntr), "val_index_hash": _hash_idx(nva)})
        snap_rows.append({"design": "dsnap", "boundary": f"fold_{int(rec['fold_id'])+1}",
                          "primary_train": int(tr.size), "primary_val": int(va.size),
                          "snap_train": int(ntr.size), "snap_val": int(nva.size),
                          "rows_moved_to_eval": int(cut - new_cut),
                          "boundary_date": str(pd.Timestamp(dates_dev[cum][cut]).date()),
                          "train_max_date": str(pd.Timestamp(dates_dev[ntr].max()).date()),
                          "val_min_date": str(pd.Timestamp(dates_dev[nva].min()).date()),
                          "strict": True})
    # dev/held-out boundary: move boundary-date rows from dev into held-out
    all_dates = np.concatenate([dates_dev, dates_test])
    cut = len(dates_dev)
    new_cut = snap_back(all_dates, cut)
    snap_dev_idx = np.arange(new_cut)
    snap_test_idx = np.arange(new_cut, len(all_dates))
    assert all_dates[snap_dev_idx].max() < all_dates[snap_test_idx].min()
    snap_rows.append({"design": "dsnap", "boundary": "development_heldout",
                      "primary_train": len(dates_dev), "primary_val": len(dates_test),
                      "snap_train": int(snap_dev_idx.size), "snap_val": int(snap_test_idx.size),
                      "rows_moved_to_eval": int(cut - new_cut),
                      "boundary_date": str(pd.Timestamp(all_dates[cut]).date()),
                      "train_max_date": str(pd.Timestamp(all_dates[snap_dev_idx].max()).date()),
                      "val_min_date": str(pd.Timestamp(all_dates[snap_test_idx].min()).date()),
                      "strict": True})
    snap_rows.append({"design": "dsnap", "boundary": "production_2025",
                      "primary_train": len(all_dates), "primary_val": len(dates_as),
                      "snap_train": len(all_dates), "snap_val": len(dates_as),
                      "rows_moved_to_eval": 0, "boundary_date": "2025-01-01",
                      "train_max_date": str(pd.Timestamp(all_dates.max()).date()),
                      "val_min_date": str(pd.Timestamp(dates_as.min()).date()), "strict": True})
    c.write_json(c.CONFIGS / "split_protocol_dsnap.json", {
        "design": "dsnap", "design_type": "strict_date_robustness",
        "rule": ("snap the positional cut BACK to the first row sharing the boundary date, so "
                 "every boundary-date row lands on the evaluation side and "
                 "max(train_date) < min(eval_date) strictly at all 8 boundaries"),
        "canonical_utilities": "imported from utils.motivation_utils; never modified in place",
        "folds": snap_folds,
        "development_idx": _save_idx("dsnap_development", snap_dev_idx),
        "heldout_idx": _save_idx("dsnap_heldout", snap_test_idx),
        "development_indices_hash": _hash_idx(snap_dev_idx),
        "heldout_indices_hash": _hash_idx(snap_test_idx),
        "n_development": int(snap_dev_idx.size), "n_heldout": int(snap_test_idx.size),
        "n_forward_2025": int(len(dates_as)),
        "boundary_audit": snap_rows,
        "provenance": c.preflight_block(),
    })

    # ---- D-PURGE (ORACLE) --------------------------------------------------
    purge, prows = [], []
    for rec in folds:
        tr = np.asarray(rec["train_indices"], dtype=int)
        va = np.asarray(rec["val_indices"], dtype=int)
        evalset = np.unique(pin_dev[va])
        keep = ~np.isin(pin_dev[tr], evalset)
        ntr = tr[keep]
        k1 = int(rec["fold_id"]) + 1
        purge.append({"fold_1based": k1,
                      "train_idx": _save_idx(f"dpurge_fold{k1}_train", ntr),
                      "val_idx": _save_idx(f"dpurge_fold{k1}_val", va),
                      "train_size": int(ntr.size), "val_size": int(va.size),
                      "train_index_hash": _hash_idx(ntr), "val_index_hash": _hash_idx(va)})
        prows.append({"design": "dpurge", "block": f"fold_{int(rec['fold_id'])+1}",
                      "primary_train": int(tr.size), "purged_train": int(ntr.size),
                      "rows_removed": int(tr.size - ntr.size),
                      "share_removed": float((tr.size - ntr.size) / tr.size),
                      "eval_size_unchanged": int(va.size),
                      "eval_pins_in_train_after_purge": int(np.isin(pin_dev[ntr], evalset).sum())})
    dev_keep = ~np.isin(pin_dev, np.unique(pin_test))
    purge_dev = np.arange(len(pin_dev))[dev_keep]
    prows.append({"design": "dpurge", "block": "development_pool",
                  "primary_train": len(pin_dev), "purged_train": int(purge_dev.size),
                  "rows_removed": int(len(pin_dev) - purge_dev.size),
                  "share_removed": float((len(pin_dev) - purge_dev.size) / len(pin_dev)),
                  "eval_size_unchanged": len(pin_test),
                  "eval_pins_in_train_after_purge":
                      int(np.isin(pin_dev[purge_dev], np.unique(pin_test)).sum())})
    prod_keep = ~np.isin(pin_prod, np.unique(pin_as))
    purge_prod = np.arange(len(pin_prod))[prod_keep]
    prows.append({"design": "dpurge", "block": "production_2016_2024",
                  "primary_train": len(pin_prod), "purged_train": int(purge_prod.size),
                  "rows_removed": int(len(pin_prod) - purge_prod.size),
                  "share_removed": float((len(pin_prod) - purge_prod.size) / len(pin_prod)),
                  "eval_size_unchanged": len(pin_as),
                  "eval_pins_in_train_after_purge":
                      int(np.isin(pin_prod[purge_prod], np.unique(pin_as)).sum())})
    c.write_json(c.CONFIGS / "split_protocol_dpurge.json", {
        "design": "dpurge", "design_type": "oracle_diagnostic",
        "NOT_PROSPECTIVELY_IMPLEMENTABLE": True,
        "not_a_proposed_ccao_split": True,
        "rule": ("evaluation sets preserved BITWISE from the frozen primary design; from each "
                 "TRAINING block remove every row whose meta_pin appears in the paired "
                 "evaluation block. Requires knowing future evaluation PINs at training time, "
                 "which no assessor could know -- hence oracle."),
        "purpose": ("bound the contribution of parcel-specific history to the reported results; "
                    "it deliberately over-corrects"),
        "folds": purge,
        "development_train_idx": _save_idx("dpurge_development_train", purge_dev),
        "production_train_idx": _save_idx("dpurge_production_train", purge_prod),
        "development_indices_hash": _hash_idx(purge_dev),
        "production_indices_hash": _hash_idx(purge_prod),
        "n_development_train": int(purge_dev.size), "n_production_train": int(purge_prod.size),
        "purge_audit": prows, "n_unique_pins": n_pins,
        "provenance": c.preflight_block(),
    })
    c.write_table(pd.DataFrame(snap_rows), c.TABLES / "dsnap_boundary_audit.csv")
    c.write_table(pd.DataFrame(prows), c.TABLES / "dpurge_purge_audit.csv")

    # ---- D-UNSEEN definition (no fits) ------------------------------------
    unseen = {"design": "dunseen", "design_type": "evaluation_subset_secondary", "fits": 0,
              "rule": ("restrict each evaluation block of the FROZEN primary models to parcels "
                       "whose meta_pin never occurred in that model's training block"),
              "denominator_changes": True,
              "not_differenced_against_frozen_numbers": True, "blocks": {}}
    for rec in folds:
        tr = np.asarray(rec["train_indices"], dtype=int)
        va = np.asarray(rec["val_indices"], dtype=int)
        m = ~np.isin(pin_dev[va], np.unique(pin_dev[tr]))
        unseen["blocks"][f"fold_{int(rec['fold_id'])+1}"] = {
            "n_eval": int(va.size), "n_unseen": int(m.sum()),
            "share_unseen": float(m.mean()), "mask_hash": _hash_idx(va[m])}
    m = ~np.isin(pin_test, np.unique(pin_dev))
    unseen["blocks"]["heldout"] = {"n_eval": int(pin_test.size), "n_unseen": int(m.sum()),
                                   "share_unseen": float(m.mean()),
                                   "mask_hash": _hash_idx(np.arange(pin_test.size)[m])}
    m = ~np.isin(pin_as, np.unique(pin_prod))
    unseen["blocks"]["forward_2025"] = {"n_eval": int(pin_as.size), "n_unseen": int(m.sum()),
                                        "share_unseen": float(m.mean()),
                                        "mask_hash": _hash_idx(np.arange(pin_as.size)[m])}
    c.write_json(c.CONFIGS / "unseen_subset_definition.json", unseen)

    grid = screening_grid()
    grid["provenance"] = c.preflight_block()
    c.write_json(c.CONFIGS / "robustness_rho_grid.json", grid)

    print("D-SNAP boundary audit:")
    print(pd.DataFrame(snap_rows)[["boundary", "primary_train", "snap_train",
                                   "rows_moved_to_eval", "boundary_date", "strict"]].to_string(index=False))
    print("\nD-PURGE audit:")
    print(pd.DataFrame(prows)[["block", "primary_train", "purged_train", "rows_removed",
                               "share_removed", "eval_pins_in_train_after_purge"]].to_string(index=False))
    print(f"\nscreening grid: {grid['n_positive_screening']} positive rhos "
          f"+ rho=0 -> {grid['n_configs_per_design']} configs per design")
    print("\nD-UNSEEN unseen shares:",
          {k: round(v["share_unseen"], 4) for k, v in unseen["blocks"].items()})
    return 0


# --------------------------------------------------------------------- fits
def _shards():
    return [(d, b, f) for d in DESIGNS for b in BLOCKS for f in FAMILIES]


def fit_shard(shard: int) -> int:
    design, block, family = _shards()[shard]
    print(f"[temporal] design={design} block={block} family={family}", flush=True)
    print(f"[temporal] env={c.thread_env()}", flush=True)

    proto = json.loads((c.CONFIGS / f"split_protocol_{design}.json").read_text())
    grid = json.loads((c.CONFIGS / "robustness_rho_grid.json").read_text())
    params = dict(c.frozen_lgbm_config()["lgbm_params"])
    cfg_hash = c.frozen_lgbm_config()["lgbm_params_sha256"]

    df_tv, df_test, df_assess, pred_cols, cat_cols = c.load_canonical_splits()
    production = pd.concat([df_tv, df_test], ignore_index=True)

    if block.startswith("fold_"):
        k = int(block.split("_")[1])
        fr = [f for f in proto["folds"] if f["fold_1based"] == k][0]
        tri = _load_idx(fr["train_idx"]); vai = _load_idx(fr["val_idx"])
        tr_df, ev_df, ev_rid = df_tv.iloc[tri], df_tv.iloc[vai], vai
    elif block == "heldout":
        if design == "dsnap":
            dev_i = _load_idx(proto["development_idx"])
            ho_i = _load_idx(proto["heldout_idx"])
            tr_df = production.iloc[dev_i]
            ev_df = production.iloc[ho_i]
            ev_rid = np.arange(ho_i.size)          # positional within the eval block
        else:
            tri = _load_idx(proto["development_train_idx"])
            tr_df, ev_df = df_tv.iloc[tri], df_test
            ev_rid = np.arange(len(df_test))       # identical to the primary design
    elif block == "forward_2025":
        if design == "dsnap":
            tr_df, ev_df = production, df_assess    # the 2025 boundary is already year-based
            ev_rid = np.arange(len(df_assess))
        else:
            tri = _load_idx(proto["production_train_idx"])
            tr_df, ev_df = production.iloc[tri], df_assess
            ev_rid = np.arange(len(df_assess))
    else:
        raise c.ProtocolViolation(f"unknown block {block}")

    return _run_cell(design, block, family, tr_df, ev_df, ev_rid, params, cfg_hash,
                     grid, pred_cols, cat_cols)


def _run_cell(design, block, family, tr_df, ev_df, ev_rid, params, cfg_hash, grid,
              pred_cols, cat_cols):
    from soft_constrained_models.boosting_models import LGBCovPenalty, LGBSmoothPenalty
    from run_temporal_cv import _native_lgbm_estimator
    from utils.motivation_utils import _compute_extended_metrics, paper_mechanism_metrics
    from utils.delta_nl import estimate_delta_nl

    X_tr = tr_df[pred_cols].copy(); X_ev = ev_df[pred_cols].copy()
    for kk in [kk for kk in cat_cols if kk in X_tr.columns]:
        X_tr[kk] = X_tr[kk].astype("category"); X_ev[kk] = X_ev[kk].astype("category")
    y_tr = np.log(tr_df[c.TARGET_COL].to_numpy())
    y_ev = np.log(ev_df[c.TARGET_COL].to_numpy())

    rhos = ([0.0] if family == "native" else
            [0.0] + [float(x) for x in grid["positive_rhos"]])
    if family == "native":
        rhos = [None]

    rows = []
    for rho in rhos:
        t0 = time.perf_counter()
        if family == "native":
            est = _native_lgbm_estimator(dict(params)); est.fit(X_tr, y_tr)
            p = np.asarray(est.predict(X_ev), dtype=float).reshape(-1)
        elif family == "direct":
            m = LGBCovPenalty(rho=float(rho), ratio_mode="diff", match_native_init=True,
                              zero_grad_tol=1e-12, early_stopping_rounds=None,
                              lgbm_params=dict(params), verbose=False)
            m.fit(X_tr, y_tr); p = np.asarray(m.predict(X_ev), dtype=float).reshape(-1)
        else:
            m = LGBSmoothPenalty(rho=float(rho), ratio_mode="diff",
                                 weighting_proxy_mode="identity", match_native_init=True,
                                 zero_grad_tol=1e-12, early_stopping_rounds=None,
                                 lgbm_params=dict(params), verbose=False)
            m.fit(X_tr, y_tr); p = np.asarray(m.predict(X_ev), dtype=float).reshape(-1)
        el = time.perf_counter() - t0

        mm = _compute_extended_metrics(y_true_log=y_ev, y_pred_log=p, y_train_log=y_tr,
                                       ratio_mode="diff")
        ren = {"Median ratio": "median_ratio", "Mean ratio": "mean_ratio",
               "W. Mean ratio": "weighted_mean_ratio", "COV_IAAO": "COV"}
        out = {}
        for kk in ["R2_price", "MAE_price", "MAPE", "RMSE_log", "Median ratio", "Mean ratio",
                   "W. Mean ratio", "COD", "COV_IAAO", "PRD", "PRB", "MKI", "VEI"]:
            if kk in mm:
                try:
                    out[ren.get(kk, kk)] = float(mm[kk])
                except (TypeError, ValueError):
                    pass
        mech = paper_mechanism_metrics(y_ev, p)
        out["beta_log"] = float(mech["Beta_log"])
        out["Cov_log_residual_log_price"] = float(mech["Cov_log_residual_log_price"])
        out["dCor_e_y"] = float(mech["dCor_e_y"])
        dn = estimate_delta_nl(y_ev, p, ev_rid)
        out["Delta_NL"] = float(dn["Delta_NL"]); out["Delta_NL_raw"] = float(dn["Delta_NL_raw"])

        rows.append({"design": design, "design_type": ("oracle_diagnostic" if design == "dpurge"
                                                       else "strict_date_robustness"),
                     "block": block, "family": family, "rho": rho,
                     "n_train": int(len(y_tr)), "n_eval": int(len(y_ev)),
                     "grid": "screening", "lgbm_params_sha256": cfg_hash,
                     "pred_sha256": _hash_arr(p), "fit_seconds": el,
                     "execution_settings": "HISTORICAL (no determinism pins)", **out})
        print(f"[temporal] {design}/{block}/{family} rho={rho} beta={out['beta_log']:+.5f} "
              f"R2={out['R2_price']:.5f} ({el:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    c.write_table(df, c.TABLES / f"robustness_shard__{design}__{block}__{family}.csv")
    print(f"[temporal] wrote shard {design}/{block}/{family} ({len(df)} rows)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["protocols", "fit", "list"])
    ap.add_argument("--shard", type=int, default=-1)
    a = ap.parse_args()
    if a.mode == "protocols":
        return build_protocols()
    if a.mode == "list":
        for i, s in enumerate(_shards()):
            print(i, s)
        return 0
    return fit_shard(a.shard)


if __name__ == "__main__":
    raise SystemExit(main())
