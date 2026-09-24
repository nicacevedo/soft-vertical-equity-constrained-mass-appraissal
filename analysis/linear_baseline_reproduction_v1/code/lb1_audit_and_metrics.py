#!/usr/bin/env python3
"""LB Stage 1 -- audit the cached ordinary-linear predictions and recompute every metric.

What this does
--------------
(1) PARITY.  Establishes, at full precision and at row level, that the ordinary
    LightGBM series in output/paper_v6_preselection_994/baseline_reporting is the
    SAME object as the frozen zero-penalty benchmark (P0 cell A) on all nine
    canonical blocks.  This is the gate the manuscript-ready comparison depends on.
(2) AUDIT.  Re-executes the frozen metric code on the cached row-level linear
    predictions and compares against the values the frozen run recorded, so the
    cached predictions are validated rather than trusted.
(3) METRICS.  Emits the full-precision metric suite for both models on all nine
    blocks, plus the model-specific equal-weight PRB/VEI value-proxy summary.

Fits nothing.  Selects nothing.  Writes only inside the LB stage.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lb_common as lb

MAIN_PRED = ["R2_price", "MAE_price", "MAPE", "RMSE_log"]
MAIN_EQUITY = ["PRD", "PRB", "MKI", "VEI"]
LEVEL_UNIF = ["median_ratio", "mean_ratio", "weighted_mean_ratio", "COD", "COV"]
MECHANISM = ["beta_log", "Cov_log_residual_log_price", "Delta_NL", "Delta_NL_raw", "dCor_e_y"]
ALL_METRICS = MAIN_PRED + MAIN_EQUITY + LEVEL_UNIF + MECHANISM

# Column names the frozen baseline_reporting metric CSVs use for the same quantities.
FROZEN_COLMAP = {
    "R2_price": "R2_price", "MAE_price": "MAE_price", "MAPE": "MAPE", "RMSE_log": "RMSE_log",
    "PRD": "PRD", "PRB": "PRB", "MKI": "MKI", "VEI": "VEI",
    "median_ratio": "Median ratio", "mean_ratio": "Mean ratio",
    "weighted_mean_ratio": "W. Mean ratio", "COD": "COD", "COV": "COV_IAAO",
    "beta_log": "Beta_log", "Cov_log_residual_log_price": "Cov_log_residual_log_price",
    "dCor_e_y": "dCor_e_y",
}


def frozen_oos_metric_row(stage: str, config_id: str) -> pd.Series:
    fn = "test_metrics.csv" if stage == "heldout" else "assess_metrics.csv"
    d = pd.read_csv(lb.BASELINE_REPORTING / fn)
    sel = d[d.config_id == config_id]
    if len(sel) != 1:
        raise lb.ProtocolViolation(f"{fn}: {len(sel)} rows for {config_id}")
    return sel.iloc[0]


def frozen_fold_metric_row(fold_1based: int, config_id: str) -> pd.Series:
    m = pd.read_csv(lb.FROZEN_CV_RUN_MAP)
    sel = m[(m.config_id == config_id) & (m.fold_1based == fold_1based)
            & (m.root == "paper_v6_preselection_994")]
    if config_id == lb.NATIVE_CONFIG_ID:
        sel = sel[sel.model_name == "LGBMRegressor"]
    run_id = str(sel.iloc[0]["run_id"])
    p = (lb.V6 / "runs" / "data_id=d4929d43ec19badf" / "split_id=3d464d4a611b131b"
         / f"fold_id={fold_1based - 1}" / f"{run_id}.parquet")
    if not p.exists():
        raise lb.ProtocolViolation(f"frozen fold metric artifact missing: {p}")
    return pd.read_parquet(p).iloc[0]


def frozen_zero_control_row(evaluation: str) -> pd.Series:
    d = pd.read_csv(lb.ZERO_CONTROL_FULL)
    sel = d[(d.cell_id == "A") & (d.evaluation == evaluation)]
    if len(sel) != 1:
        raise lb.ProtocolViolation(f"zero_control_full.csv: {len(sel)} rows for A/{evaluation}")
    return sel.iloc[0]


def main() -> int:
    parity_rows, metric_rows, audit_rows, proxy_rows = [], [], [], []

    for blk in lb.BLOCKS:
        ev = blk["eval"]
        d = lb.load_pair(ev)
        print(f"[lb1] {ev:<14s} n={d['n']:>6d} n_train={d['n_train']:>6d} "
              f"native_bitwise={d['native_pred_matches_frozen_bitwise']}", flush=True)

        # ---- (1) frozen-LightGBM parity -------------------------------------
        nat = d["pred"][lb.NATIVE_NAME]
        ref_pred_sha = lb.hash_f64(nat)
        parity = {
            "evaluation": ev, "block_id": d["block_id"], "fold_1based": d["fold_1based"],
            "n": d["n"], "n_fitting_block": d["n_train"],
            "row_id_matches_frozen_benchmark": True,
            "y_true_log_bitwise_equal_to_benchmark": True,
            "native_pred_bitwise_equal_to_benchmark": d["native_pred_matches_frozen_bitwise"],
            "native_pred_max_abs_delta_log": d["native_pred_max_abs_delta"],
            "native_pred_sha256_recomputed": ref_pred_sha,
            "frozen_benchmark_eval_pred_sha256": d["frozen_eval_pred_sha256"],
            "native_pred_sha256_agrees": ref_pred_sha == d["frozen_eval_pred_sha256"],
            "y_eval_log_sha256_recomputed": lb.hash_f64(d["y_true_log"]),
            "frozen_benchmark_y_eval_log_sha256": d["frozen_y_eval_log_sha256"],
            "eval_index_hash_recomputed": lb.eval_index_hash(d["row_id"], d["kind"]),
            "eval_index_hash_convention": (
                "sha256(int64 bytes)" if d["kind"] == "out_of_time"
                else "motivation_utils._stable_hash({'idx': [...]}) -- the archived protocol convention"),
            "frozen_benchmark_eval_index_hash": d["frozen_eval_index_hash"],
            "archived_protocol_val_index_hash": d["archived_val_index_hash"],
            "linear_pred_sha256": lb.hash_f64(d["pred"][lb.LINEAR_NAME]),
            "lgbm_params_sha256": d["frozen_config_hash"],
            "lgbm_params_sha256_matches_expected": (
                d["frozen_config_hash"] == lb.EXPECTED_LGBM_PARAMS_SHA256),
            "sale_date_min": d["sale_date_min"], "sale_date_max": d["sale_date_max"],
        }
        parity["y_eval_log_sha256_agrees"] = (
            parity["y_eval_log_sha256_recomputed"] == parity["frozen_benchmark_y_eval_log_sha256"])
        parity["eval_index_hash_agrees"] = (
            parity["eval_index_hash_recomputed"] == parity["frozen_benchmark_eval_index_hash"])
        parity_rows.append(parity)

        # ---- (2)+(3) metrics + audit ----------------------------------------
        for model, pred in d["pred"].items():
            cfg = lb.LINEAR_CONFIG_ID if model == lb.LINEAR_NAME else lb.NATIVE_CONFIG_ID
            m = lb.metrics_from(d["y_true_log"], pred, d["y_train_log"], d["row_id"])
            row = {"model": model, "config_id": cfg, "evaluation": ev,
                   "block_id": d["block_id"], "fold_1based": d["fold_1based"],
                   "kind": d["kind"], "n": d["n"], "n_fitting_block": d["n_train"],
                   "prediction_sha256": lb.hash_f64(pred)}
            row.update({k: m[k] for k in ALL_METRICS})
            metric_rows.append(row)

            prox = lb.value_proxy_summary(d["y_true_log"], pred)
            proxy_rows.append({"model": model, "evaluation": ev, "n": d["n"], **prox})

            # audit against whatever the frozen run recorded for this cell
            if d["kind"] == "out_of_time":
                fr = frozen_oos_metric_row(ev, cfg)
                src = f"baseline_reporting/{'test' if ev == 'heldout' else 'assess'}_metrics.csv"
            else:
                fr = frozen_fold_metric_row(d["fold_1based"], cfg)
                src = f"runs/fold_id={d['fold_1based'] - 1}/<run_id>.parquet"
            for k, fk in FROZEN_COLMAP.items():
                if fk not in fr.index:
                    continue
                fv = float(fr[fk])
                rv = float(m[k])
                den = max(abs(fv), 1e-300)
                audit_rows.append({
                    "model": model, "config_id": cfg, "evaluation": ev, "metric": k,
                    "recomputed": rv, "frozen_recorded": fv,
                    "abs_delta": abs(rv - fv), "rel_delta": abs(rv - fv) / den,
                    "frozen_source": src,
                })

            # additional audit: cell-A OOS rows also appear in the P0 zero-control table
            if model == lb.NATIVE_NAME and d["kind"] == "out_of_time":
                zc = frozen_zero_control_row(ev)
                for k in ALL_METRICS:
                    if k not in zc.index or pd.isna(zc[k]):
                        continue
                    fv = float(zc[k]); rv = float(m[k])
                    audit_rows.append({
                        "model": model, "config_id": cfg, "evaluation": ev, "metric": k,
                        "recomputed": rv, "frozen_recorded": fv,
                        "abs_delta": abs(rv - fv), "rel_delta": abs(rv - fv) / max(abs(fv), 1e-300),
                        "frozen_source": "p0_major_revision_validation/tables/zero_control_full.csv (cell A)",
                    })

    parity = pd.DataFrame(parity_rows)
    metrics = pd.DataFrame(metric_rows)
    audit = pd.DataFrame(audit_rows)
    proxy = pd.DataFrame(proxy_rows)

    # ---- descriptive CV summaries (equal-weight over the seven folds) --------
    cv = metrics[metrics.kind == "cv_validation"]
    cvsum = []
    for model, g in cv.groupby("model", sort=False):
        for stat, fn in (("CV_mean", np.mean), ("CV_SD", lambda a: np.std(a, ddof=1)),
                         ("CV_min", np.min), ("CV_max", np.max)):
            r = {"model": model, "evaluation": stat, "kind": "cv_summary",
                 "n_folds": int(len(g)),
                 "note": "equal-weight over seven OVERLAPPING expanding-window folds; "
                         "not independent replications; descriptive only"}
            for k in ALL_METRICS:
                r[k] = float(fn(g[k].to_numpy()))
            cvsum.append(r)
    cvsum = pd.DataFrame(cvsum)

    lb.write_table(parity, lb.TABLES / "lb_frozen_lightgbm_parity.csv")
    lb.write_table(metrics, lb.TABLES / "lb_metrics_full_precision_all_blocks.csv")
    lb.write_table(audit, lb.TABLES / "lb_cached_prediction_audit.csv")
    lb.write_table(cvsum, lb.TABLES / "lb_cv_fold_summary.csv")
    lb.write_table(proxy, lb.TABLES / "lb_prb_vei_value_proxy.csv")

    # ---- verdicts -----------------------------------------------------------
    parity_ok = bool(
        parity.native_pred_sha256_agrees.all()
        and parity.y_eval_log_sha256_agrees.all()
        and parity.eval_index_hash_agrees.all()
        and parity.native_pred_bitwise_equal_to_benchmark.all()
        and parity.lgbm_params_sha256_matches_expected.all())
    worst = audit.sort_values("rel_delta", ascending=False).head(15)
    audit_ok = bool(audit.rel_delta.max() <= 1e-9)

    verdict = {
        "frozen_lightgbm_parity": {
            "verdict": "PASS -- bitwise identical on all nine blocks" if parity_ok else "FAIL",
            "blocks_checked": int(len(parity)),
            "max_native_pred_abs_delta_log": float(parity.native_pred_max_abs_delta_log.max()),
            "what_was_compared": (
                "output/paper_v6_preselection_994/baseline_reporting + CV predictions "
                "(config_id=252a25d9c0ce796b, LGBMRegressor) vs "
                "output/p0_major_revision_validation/zero_reference_fits/cell=A "
                "(frozen zero-penalty benchmark), row-level"),
            "shared_sample_identity": (
                "row_id arrays equal, y_true_log bitwise equal, eval_index_hash equal; "
                "the linear series is drawn from the same artifacts and therefore shares "
                "that identity exactly"),
            "identity_limitation": (
                "the prediction artifacts carry row_id and sale_date, not meta_pin: sale "
                "identity is established positionally against the canonical split plus the "
                "bitwise log-price vector, not by parcel identifier"),
        },
        "cached_linear_prediction_audit": {
            "verdict": ("PASS -- recomputed metrics reproduce the frozen recorded values to "
                        "float tolerance" if audit_ok else "REVIEW -- see worst rows"),
            "comparisons": int(len(audit)),
            "max_abs_delta": float(audit.abs_delta.max()),
            "max_rel_delta": float(audit.rel_delta.max()),
            "worst_rows": worst.to_dict(orient="records"),
            "rerun_needed": not (parity_ok and audit_ok),
        },
        "rounded_agreement_note": (
            "Parity was established on unrounded float64 values and prediction content "
            "hashes, not by agreement after rounding R2_P to 0.894."),
        "preflight": lb.preflight(),
    }
    lb.write_json(lb.PROVENANCE / "lb1_verdict.json", verdict)

    print("\n=== frozen-LightGBM parity:", verdict["frozen_lightgbm_parity"]["verdict"])
    print("=== cached-linear audit:   ", verdict["cached_linear_prediction_audit"]["verdict"],
          f"(max rel delta {audit.rel_delta.max():.3e})")
    pd.set_option("display.width", 220)
    print("\nHeld-out / 2025, main measures:")
    show = metrics[metrics.kind == "out_of_time"][
        ["model", "evaluation", "n"] + MAIN_PRED + MAIN_EQUITY]
    print(show.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
