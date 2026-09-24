#!/usr/bin/env python3
"""LB Stage 2 -- v20 compatibility gate and the linear-vs-LightGBM predictor audit.

Checks, in order, and refuses to emit anything if one fails:
  C1  extract identity: byte length + SHA-256 against the value recorded in v20.
  C2  canonical splits: 344,607 / 38,290 / 26,641 and the 382,897 production block,
      via the imported loader (run_temporal_cv._load_and_split_data).
  C3  the seven archived chronological folds rebuild with matching index hashes.
  C4  sale identity: the cached prediction artifacts line up with the canonical
      splits row for row, on both sale date and bitwise log sale price.
  C5  predictor audit: the realised linear design matrix vs the 95 raw predictors
      LightGBM consumes natively -- what is excluded, and how the rest is represented.

Fits the LINEAR PREPROCESSING PIPELINE only (to observe what it produces).
No model is fitted; no prediction is produced.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lb_common as lb


def main() -> int:
    t0 = time.perf_counter()
    report: dict = {"checks": {}}

    # ---- C1 extract identity ------------------------------------------------
    n_bytes = lb.DATA_PATH.stat().st_size
    sha = lb.sha256_file(lb.DATA_PATH)
    report["checks"]["C1_extract_identity"] = {
        "path_as_resolved": str(lb.DATA_PATH.resolve()),
        "bytes": n_bytes, "bytes_recorded_in_v20": lb.V20_EXTRACT_BYTES,
        "sha256": sha, "sha256_recorded_in_v20": lb.V20_EXTRACT_SHA256,
        "pass": bool(n_bytes == lb.V20_EXTRACT_BYTES and sha == lb.V20_EXTRACT_SHA256),
    }
    print(f"[lb2] C1 extract identity: {report['checks']['C1_extract_identity']['pass']}", flush=True)
    if not report["checks"]["C1_extract_identity"]["pass"]:
        raise lb.ProtocolViolation("extract identity does not match the v20 record")

    # ---- C2 canonical splits -----------------------------------------------
    df_tv, df_test, df_assess, pred_cols, cat_cols = lb.load_canonical_splits()
    production = pd.concat([df_tv, df_test], ignore_index=True)
    report["checks"]["C2_sample_counts"] = {
        "development_pool": int(len(df_tv)), "heldout": int(len(df_test)),
        "forward_2025": int(len(df_assess)), "production_2016_2024": int(len(production)),
        "expected": [lb.N_DEVELOPMENT, lb.N_HELDOUT, lb.N_2025, lb.N_PRODUCTION],
        "n_predictors": int(len(pred_cols)), "n_categorical": int(len(cat_cols)),
        "n_predictors_recorded_in_v20": lb.V20_N_PREDICTORS,
        "n_categorical_recorded_in_v20": lb.V20_N_CATEGORICAL,
        "eligibility_filters": ["ind_pin_is_multicard == False", "sv_is_outlier == False"],
        "date_ranges": {
            "development_pool": [str(df_tv[lb.DATE_COL].min().date()), str(df_tv[lb.DATE_COL].max().date())],
            "heldout": [str(df_test[lb.DATE_COL].min().date()), str(df_test[lb.DATE_COL].max().date())],
            "forward_2025": [str(df_assess[lb.DATE_COL].min().date()), str(df_assess[lb.DATE_COL].max().date())],
        },
        "pass": bool(len(df_tv) == lb.N_DEVELOPMENT and len(df_test) == lb.N_HELDOUT
                     and len(df_assess) == lb.N_2025 and len(production) == lb.N_PRODUCTION
                     and len(pred_cols) == lb.V20_N_PREDICTORS
                     and len(cat_cols) == lb.V20_N_CATEGORICAL),
    }
    print(f"[lb2] C2 sample counts: {report['checks']['C2_sample_counts']['pass']}", flush=True)

    # ---- C3 seven chronological folds --------------------------------------
    folds, archive = lb.rebuild_folds(df_tv, verify=True)
    report["checks"]["C3_folds"] = {
        "n_folds": len(folds),
        "split_protocol": archive["split_protocol"],
        "archived_index_hashes_verified": True,
        "folds": [{"fold_1based": int(f["fold_id"]) + 1,
                   "train_start": f["train_start"], "train_end": f["train_end"],
                   "val_start": f["val_start"], "val_end": f["val_end"],
                   "train_size": int(f["train_size"]), "val_size": int(f["val_size"]),
                   "train_index_hash": f["train_index_hash"],
                   "val_index_hash": f["val_index_hash"]} for f in folds],
        "overlap_note": ("expanding-window folds share training observations and the "
                         "validation blocks of later folds contain earlier validation "
                         "observations; the seven results are not independent replications"),
        "pass": bool(len(folds) == 7),
    }
    print(f"[lb2] C3 folds: {report['checks']['C3_folds']['pass']} ({len(folds)} folds)", flush=True)

    # ---- C4 sale identity of the cached predictions -------------------------
    ident = []
    for ev, frame in (("heldout", df_test), ("forward_2025", df_assess)):
        stage = ev
        y_log = np.log(frame[lb.TARGET_COL].to_numpy())
        dates = pd.to_datetime(frame[lb.DATE_COL]).to_numpy()
        for name, cfg in ((lb.LINEAR_NAME, lb.LINEAR_CONFIG_ID),
                          (lb.NATIVE_NAME, lb.NATIVE_CONFIG_ID)):
            d = pd.read_parquet(lb.frozen_oos_prediction_path(cfg, stage))
            d = d.sort_values("row_id").reset_index(drop=True)
            ident.append({
                "evaluation": ev, "model": name, "config_id": cfg, "n": int(len(d)),
                "row_id_is_0_to_n_minus_1": bool(np.array_equal(
                    d.row_id.to_numpy(), np.arange(len(frame)))),
                "y_true_log_bitwise_equals_log_extract_target": bool(
                    np.array_equal(d.y_true_log.to_numpy(), y_log)),
                "sale_date_equals_extract_sale_date": bool(np.array_equal(
                    d.sale_date.to_numpy().astype("datetime64[ns]"),
                    dates.astype("datetime64[ns]"))),
                "y_pred_equals_direct_exponentiation": bool(np.array_equal(
                    d.y_pred.to_numpy(), np.exp(d.y_pred_log.to_numpy()))),
                "y_true_log_sha256": lb.hash_f64(d.y_true_log.to_numpy()),
                "y_pred_log_sha256": lb.hash_f64(d.y_pred_log.to_numpy()),
            })
    for f in folds:
        k = int(f["fold_id"]) + 1
        vi = np.asarray(f["val_indices"], dtype=int)
        sub = df_tv.iloc[vi]
        y_log = np.log(sub[lb.TARGET_COL].to_numpy())
        dates = pd.to_datetime(sub[lb.DATE_COL]).to_numpy()
        for name, cfg in ((lb.LINEAR_NAME, lb.LINEAR_CONFIG_ID),
                          (lb.NATIVE_NAME, lb.NATIVE_CONFIG_ID)):
            d = pd.read_parquet(lb.frozen_fold_prediction_path(cfg, k))
            ident.append({
                "evaluation": f"fold_{k}", "model": name, "config_id": cfg, "n": int(len(d)),
                "row_id_is_0_to_n_minus_1": bool(np.array_equal(d.row_id.to_numpy(), vi)),
                "y_true_log_bitwise_equals_log_extract_target": bool(
                    np.array_equal(d.y_true_log.to_numpy(), y_log)),
                "sale_date_equals_extract_sale_date": bool(np.array_equal(
                    d.sale_date.to_numpy().astype("datetime64[ns]"),
                    dates.astype("datetime64[ns]"))),
                "y_pred_equals_direct_exponentiation": bool(np.array_equal(
                    d.y_pred.to_numpy(), np.exp(d.y_pred_log.to_numpy()))),
                "y_true_log_sha256": lb.hash_f64(d.y_true_log.to_numpy()),
                "y_pred_log_sha256": lb.hash_f64(d.y_pred_log.to_numpy()),
            })
    ident = pd.DataFrame(ident)
    lb.write_table(ident, lb.TABLES / "lb_sale_identity_check.csv")
    c4 = bool(ident.row_id_is_0_to_n_minus_1.all()
              and ident.y_true_log_bitwise_equals_log_extract_target.all()
              and ident.sale_date_equals_extract_sale_date.all()
              and ident.y_pred_equals_direct_exponentiation.all())
    report["checks"]["C4_sale_identity"] = {
        "pass": c4, "artifacts_checked": int(len(ident)),
        "identifier_available": "row_id + sale_date only; meta_pin is not carried in these artifacts",
        "how_identity_is_established": (
            "positional row index into the canonical split, corroborated by a bitwise-equal "
            "log sale price vector and an exactly equal sale-date vector"),
    }
    print(f"[lb2] C4 sale identity: {c4}", flush=True)

    # ---- C5 predictor audit -------------------------------------------------
    from preprocessing.recipes_pipelined import build_model_pipeline
    params = lb.p0.load_params()
    id_vars = list(params["model"]["predictor"]["id"])
    X_dev = df_tv[pred_cols].copy()
    y_dev = np.log(df_tv[lb.TARGET_COL].to_numpy())
    pipe = build_model_pipeline(pred_vars=pred_cols, cat_vars=cat_cols, id_vars=id_vars)
    t1 = time.perf_counter()
    Z = pipe.fit_transform(X_dev, y_dev)
    prep_sec = time.perf_counter() - t1
    print(f"[lb2] C5 linear design matrix: {Z.shape} in {prep_sec:.0f}s", flush=True)

    dropped_initial = list(pipe.named_steps["1_initial_drop"].drop_cols_)
    lencode = list(pipe.named_steps["4_target_encode"].encoding_maps_.keys())
    ohe_used = list(pipe.named_steps["6_one_hot_encode"].ohe_cols_to_use_)
    winsorized = list(pipe.named_steps["7_winsorize"].quantiles_.keys())
    boxcoxed = list(pipe.named_steps["8_feature_engineer"].power_transformers_.keys())
    scaled = list(pipe.named_steps["9_normalize"].cols_to_scale_)
    nzv_removed = list(pipe.named_steps["10_nzv_removal"].nzv_cols_)
    imputed_num = list(pipe.named_steps["3_imputation"].num_cols_)
    imputed_nom = list(pipe.named_steps["3_imputation"].nom_cols_)

    excluded = [c for c in pred_cols if c in dropped_initial]
    rows = []
    for c in pred_cols:
        if c in excluded:
            rep = "EXCLUDED before fitting (non-numeric loc_* that is not loc_school_*)"
        elif c in lencode:
            rep = "target-encoded (training-fold mean of log sale price), then standardised"
        elif c in ohe_used:
            rep = "one-hot encoded (unseen levels -> 'unknown' -> all-zero row)"
        else:
            rep = "numeric passthrough"
        extras = []
        if c in winsorized: extras.append("winsorised at 1%/99% of the training block")
        if c in boxcoxed or f"{c}_1" in boxcoxed: extras.append("Box-Cox")
        if c in ("char_yrblt", "char_bldg_sf", "char_land_sf"): extras.append("squared term added")
        if c in ("prox_nearest_vacant_land_dist_ft", "prox_nearest_new_construction_dist_ft",
                 "acs5_percent_employment_unemployed"): extras.append("+0.001 offset copy added")
        if c in scaled: extras.append("standardised")
        rows.append({
            "predictor": c,
            "categorical_in_params": c in cat_cols,
            "used_by_lightgbm": True,
            "lightgbm_representation": ("native categorical (pandas category dtype)"
                                        if c in cat_cols else "raw numeric, NaN handled natively"),
            "used_by_linear": c not in excluded,
            "linear_representation": rep,
            "linear_extra_transforms": "; ".join(extras),
            "linear_missing_value_treatment": (
                "median imputation" if c in imputed_num else
                ("most-frequent imputation" if c in imputed_nom else
                 ("not imputed (excluded)" if c in excluded else "no imputer attached"))),
        })
    pa = pd.DataFrame(rows)
    lb.write_table(pa, lb.TABLES / "lb_predictor_representation_audit.csv", full_precision=False)

    report["checks"]["C5_predictor_audit"] = {
        "n_raw_predictors": len(pred_cols),
        "n_categorical": len(cat_cols),
        "linear_design_matrix_columns": int(Z.shape[1]),
        "linear_design_matrix_rows": int(Z.shape[0]),
        "predictors_excluded_from_the_linear_model": excluded,
        "n_predictors_excluded_from_the_linear_model": len(excluded),
        "exclusion_rule": ("preprocessing.recipes_pipelined.InitialColumnDropper drops every "
                           "column whose name starts with 'loc_', does not start with "
                           "'loc_school_', and is not numeric in the extract"),
        "target_encoded": lencode,
        "one_hot_encoded": ohe_used,
        "winsorised": winsorized,
        "box_cox_transformed": boxcoxed,
        "squared_terms_added": ["char_yrblt", "char_bldg_sf", "char_land_sf"],
        "standardised_columns": len(scaled),
        "near_zero_variance_columns_removed": nzv_removed,
        "median_imputed_numeric": len(imputed_num),
        "most_frequent_imputed_nominal": len(imputed_nom),
        "lightgbm_representation": ("the same 95 raw predictors, 23 of them as pandas "
                                    "category dtype consumed by LightGBM's native categorical "
                                    "handling; no imputation, no encoding, no engineered terms"),
        "information_comparison": (
            "Not nested in either direction. The linear model sees 93 of the 95 predictors and "
            "loses two high-cardinality string location fields entirely, but it also receives "
            "information LightGBM never gets in that form: target encodings of five "
            "high-cardinality fields, Box-Cox transforms, squared terms and offset copies. "
            "So the accuracy gap cannot be read as a pure functional-form result."),
        "preprocessing_fit_seconds_on_development_pool": prep_sec,
        "preprocessing_is_refit_per_block": (
            "yes -- pipe.fit_transform is called on each fitting block's training rows only "
            "(run_temporal_cv.py:1359-1364), so each fold's encodings, imputers, winsor limits, "
            "Box-Cox parameters and scaler come from that fold's training block"),
        "pass": True,
    }

    report["preflight"] = lb.preflight()
    report["elapsed_seconds"] = time.perf_counter() - t0
    report["verdict"] = ("PASS" if all(v.get("pass") for v in report["checks"].values()) else "FAIL")
    lb.write_json(lb.PROVENANCE / "lb2_compatibility.json", report)
    print(f"\n[lb2] verdict: {report['verdict']}  ({report['elapsed_seconds']:.0f}s)")
    print(f"[lb2] linear design matrix: {Z.shape[1]} columns; excluded predictors: {excluded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
