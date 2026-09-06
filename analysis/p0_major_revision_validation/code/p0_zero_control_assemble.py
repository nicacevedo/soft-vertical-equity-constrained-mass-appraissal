#!/usr/bin/env python3
"""Stage 1.5 -- reproduction QC for the 27 fits + complete zero-control evidence.

Two parts:
  (1) QC: compare the NEW A and C paired-evaluation predictions against the exact frozen
      artifacts using the existing R1-R4 rubric.  Cell B is checked against the Stage-1
      aggregate ladder values (no per-row Stage-1 B artifact exists); B is NOT required
      to equal C.
  (2) Zero-control evidence for A/B/C across fold 1..7, CV mean, CV SD, held-out and
      2025 forward, with the full canonical metric suite plus the frozen Delta_NL estimator.

No PRB inference, no VEI significance, no smearing -- those remain later P1 extras.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p0_common as c

T = c.TABLES
OUT = c.P0_OUTPUT_ROOT / "zero_reference_fits"

METRIC_KEYS = ["R2_price", "MAE_price", "MAPE", "RMSE_log", "Median ratio", "Mean ratio",
               "W. Mean ratio", "COD", "COV_IAAO", "PRD", "PRB", "MKI", "VEI"]
RENAME = {"Median ratio": "median_ratio", "Mean ratio": "mean_ratio",
          "W. Mean ratio": "weighted_mean_ratio", "COV_IAAO": "COV"}

CELL_META = {
    "A": ("Ordinary LightGBM (standard raw-label native)", "assessor-facing / workflow benchmark"),
    "B": ("Centered-label native L2 (initialization-aligned)",
          "implementation-decomposition control only"),
    "C": ("Custom-objective rho=0 origin", "PRIMARY within-path penalty-isolating reference"),
}


def metrics_from(y_log, p_log, y_train_log, row_ids) -> dict:
    from utils.motivation_utils import _compute_extended_metrics, paper_mechanism_metrics
    from utils.delta_nl import estimate_delta_nl
    m = _compute_extended_metrics(y_true_log=np.asarray(y_log, float),
                                  y_pred_log=np.asarray(p_log, float),
                                  y_train_log=np.asarray(y_train_log, float),
                                  ratio_mode="diff")
    out = {}
    for k in METRIC_KEYS:
        if k in m:
            try:
                out[RENAME.get(k, k)] = float(m[k])
            except (TypeError, ValueError):
                pass
    mech = paper_mechanism_metrics(np.asarray(y_log, float), np.asarray(p_log, float))
    out["beta_log"] = float(mech["Beta_log"])
    out["Cov_log_residual_log_price"] = float(mech["Cov_log_residual_log_price"])
    out["dCor_e_y"] = float(mech["dCor_e_y"])
    if "RMSE_log" not in out:
        out["RMSE_log"] = float(np.sqrt(np.mean(
            (np.asarray(p_log, float) - np.asarray(y_log, float)) ** 2)))
    dn = estimate_delta_nl(np.asarray(y_log, float), np.asarray(p_log, float), row_ids)
    out["Delta_NL"] = float(dn["Delta_NL"])
    out["Delta_NL_raw"] = float(dn["Delta_NL_raw"])
    return out


def r_tier(mx: float, agree: bool, cause: str | None) -> str:
    if mx == 0.0:
        return "R1"
    if mx <= 1e-6 and agree:
        return "R2"
    if mx <= 1e-3 and agree and cause:
        return "R3"
    return "R4"


def main() -> int:
    zf = pd.read_csv(T / "zero_reference_fits.csv")
    if len(zf) != 27:
        raise c.ProtocolViolation(f"expected 27 fits, found {len(zf)}")
    cvmap = pd.read_csv(c.CONFIGS / "frozen_cv_run_map.csv")
    cfgmap = pd.read_csv(c.CONFIGS / "frozen_config_map.csv")

    # ---------------------------------------------------------------- (1) QC
    qc = []
    frozen_for = {
        "A": {"cv_model": "LGBMRegressor", "cv_rho": None, "oos_cfg": c.NATIVE_CONFIG_ID},
        "C": {"cv_model": "LGBCovPenalty", "cv_rho": 0.0, "oos_cfg": c.DIRECT_RHO0_ID},
    }
    for cell in ("A", "C"):
        spec = frozen_for[cell]
        for _, r in zf[zf.cell_id == cell].iterrows():
            newp = pd.read_parquet(c.REPO / r.eval_predictions)
            if r.block_kind == "cv_fold_train":
                sel = cvmap[(cvmap.fold_1based == r.fold_1based)
                            & (cvmap.model_name == spec["cv_model"])]
                if spec["cv_rho"] is not None:
                    sel = sel[sel.rho == spec["cv_rho"]]
                if sel.empty:
                    qc.append({"cell_id": cell, "block_id": r.block_id, "eval_name": r.eval_name,
                               "frozen_counterpart": None, "reproduction_tier": "NO_COUNTERPART"})
                    continue
                fp = sel.iloc[0]["pred_file"]
            else:
                sel = cfgmap[(cfgmap.stage == r.eval_name)
                             & (cfgmap.config_id == spec["oos_cfg"])]
                fp = sel.iloc[0]["pred_file"]
            fz = pd.read_parquet(fp)
            a = newp.sort_values("row_id").reset_index(drop=True)
            b = fz.sort_values("row_id").reset_index(drop=True)
            if len(a) != len(b):
                raise c.ProtocolViolation(f"{cell}/{r.block_id}: n {len(a)} vs {len(b)}")
            if not np.array_equal(a.row_id.to_numpy(), b.row_id.to_numpy()):
                raise c.ProtocolViolation(f"{cell}/{r.block_id}: row_id mismatch")
            if not np.allclose(a.y_true_log.to_numpy(), b.y_true_log.to_numpy(), rtol=0, atol=0):
                raise c.ProtocolViolation(f"{cell}/{r.block_id}: y_true_log mismatch")
            d = np.abs(a.y_pred_log.to_numpy() - b.y_pred_log.to_numpy())
            mx = float(np.max(d))
            m1 = metrics_from(a.y_true_log, a.y_pred_log, a.y_true_log, a.row_id)
            m2 = metrics_from(b.y_true_log, b.y_pred_log, b.y_true_log, b.row_id)
            dis = [k for k in set(m1) & set(m2)
                   if np.isfinite(m1[k]) and np.isfinite(m2[k])
                   and round(m1[k], 0 if "MAE" in k else 3) != round(m2[k], 0 if "MAE" in k else 3)]
            agree = not dis
            cause = ("same committed source; F-DIRTY bound" if 1e-6 < mx <= 1e-3 else None)
            tier = r_tier(mx, agree, cause)
            qc.append({
                "cell_id": cell, "forward_display_name": CELL_META[cell][0],
                "block_id": r.block_id, "eval_name": r.eval_name, "n": int(len(a)),
                "frozen_counterpart": str(fp),
                "mean_abs_delta_log": float(np.mean(d)),
                "p95_abs_delta_log": float(np.percentile(d, 95)),
                "max_abs_delta_log": mx,
                "frac_exact_equal": float(np.mean(d == 0.0)),
                "metrics_agree_at_displayed_precision": agree,
                "metrics_disagreeing": ";".join(sorted(dis)),
                "reproduction_tier": tier,
                "failure_class": "none" if tier == "R1" else ("F-DIRTY" if tier in ("R2", "R3")
                                                              else "PENDING_CLASSIFICATION"),
            })
            print(f"[qc] {cell} {r.block_id:<24s} {r.eval_name:<16s} max|d|={mx:.3e} tier={tier}",
                  flush=True)

    # Cell B: aggregate consistency against the Stage-1 ladder (capacity F)
    lh = pd.read_csv(T / "parity_ladder_historical.csv")
    lh = lh[(~lh.cell_pair.astype(str).str.startswith("METRICS")) & (lh.capacity == "F")]
    for blk, split in (("development_pool", "heldout"), ("production_2016_2024", "forward_2025")):
        pr = {}
        for cell in ("A", "B", "C"):
            row = zf[(zf.cell_id == cell) & (zf.block_id == blk)].iloc[0]
            pr[cell] = pd.read_parquet(c.REPO / row.eval_predictions).sort_values("row_id")
        for pair in (("A", "B"), ("B", "C"), ("A", "C")):
            d = np.abs(pr[pair[0]].y_pred_log.to_numpy() - pr[pair[1]].y_pred_log.to_numpy())
            ref = lh[(lh.split == split) & (lh.cell_pair == f"{pair[0]}<->{pair[1]}")]
            ref_mean = float(ref.mean_abs_delta_log.iloc[0]) if len(ref) else float("nan")
            ref_max = float(ref.max_abs_delta_log.iloc[0]) if len(ref) else float("nan")
            qc.append({
                "cell_id": f"{pair[0]}<->{pair[1]}", "forward_display_name": "stage1_ladder_crosscheck",
                "block_id": blk, "eval_name": split, "n": int(d.size),
                "frozen_counterpart": "tables/parity_ladder_historical.csv (capacity F)",
                "mean_abs_delta_log": float(np.mean(d)), "max_abs_delta_log": float(np.max(d)),
                "p95_abs_delta_log": float(np.percentile(d, 95)),
                "frac_exact_equal": float(np.mean(d == 0.0)),
                "stage1_ladder_mean": ref_mean, "stage1_ladder_max": ref_max,
                "matches_stage1_ladder": bool(
                    np.isclose(np.mean(d), ref_mean, rtol=0, atol=1e-12)
                    and np.isclose(np.max(d), ref_max, rtol=0, atol=1e-12)),
                "metrics_agree_at_displayed_precision": None,
                "reproduction_tier": "AGGREGATE_CROSSCHECK",
                "failure_class": "n/a",
            })
            print(f"[qc] ladder x-check {pair[0]}<->{pair[1]} {split:<14s} "
                  f"mean {np.mean(d):.6g} vs {ref_mean:.6g} -> "
                  f"{np.isclose(np.mean(d), ref_mean, rtol=0, atol=1e-12)}", flush=True)

    qdf = pd.DataFrame(qc)
    c.write_table(qdf, T / "zero_reference_reproduction_qc.csv")

    # ------------------------------------------------- (2) zero-control evidence
    rows = []
    for cell in ("A", "B", "C"):
        per_fold = {}
        for _, r in zf[zf.cell_id == cell].iterrows():
            ev = pd.read_parquet(c.REPO / r.eval_predictions)
            tr = pd.read_parquet(c.REPO / r.train_predictions)
            m = metrics_from(ev.y_true_log, ev.y_pred_log, tr.y_true_log, ev.row_id)
            evname = ("fold_%d" % r.fold_1based) if r.block_kind == "cv_fold_train" else r.eval_name
            base = {
                "cell_id": cell, "display_name": CELL_META[cell][0], "role": CELL_META[cell][1],
                "family": {"A": "native_builtin", "B": "native_builtin_centered",
                           "C": "custom_objective"}[cell],
                "rho": 0.0 if cell == "C" else None,
                "evaluation": evname, "n": int(len(ev)), **m,
                "source_artifact": r.eval_predictions,
                "config_hash": r.lgbm_params_sha256,
                "prediction_hash": r.eval_pred_sha256,
                "block_id": r.block_id,
                "cell_c_implementation": r.get("cell_c_implementation"),
                "legacy_stage1_label": ("Parity-aligned native L2" if cell == "B" else None),
            }
            rows.append(base)
            if r.block_kind == "cv_fold_train":
                per_fold[r.fold_1based] = m
        # equal-weight CV mean / SD across the seven folds
        keys = sorted({k for v in per_fold.values() for k in v})
        agg_mean, agg_sd = {}, {}
        for k in keys:
            vals = np.array([per_fold[f][k] for f in sorted(per_fold)], dtype=float)
            agg_mean[k] = float(np.mean(vals)); agg_sd[k] = float(np.std(vals, ddof=1))
        for label, agg in (("CV_mean", agg_mean), ("CV_SD", agg_sd)):
            rows.append({
                "cell_id": cell, "display_name": CELL_META[cell][0], "role": CELL_META[cell][1],
                "family": {"A": "native_builtin", "B": "native_builtin_centered",
                           "C": "custom_objective"}[cell],
                "rho": 0.0 if cell == "C" else None,
                "evaluation": label, "n": 7, **agg,
                "source_artifact": "equal-weight aggregate over fold_1..fold_7 (this table)",
                "config_hash": zf[zf.cell_id == cell].iloc[0].lgbm_params_sha256,
                "prediction_hash": None, "block_id": None,
                "cell_c_implementation": None,
                "legacy_stage1_label": ("Parity-aligned native L2" if cell == "B" else None),
            })

    # frozen cross-reference rows from the canonical path table
    v4 = (c.V12 / "analysis" / "data_id=d4929d43ec19badf" / "split_id=3d464d4a611b131b"
          / "penalty_path_analysis" / "transition_regions_paper_assets_v4_delta_nl_bends"
          / "tables" / "combined_path_table_v4_analysis_view.csv")
    fp = pd.read_csv(v4)
    frozen_sel = fp[((fp.family.isin(["Direct", "Surrogate"])) & (fp.rho == 0)) |
                    (fp.family == "LightGBM")]
    metmap = {"R2_price": "R2_price", "MAE_price": "MAE_price", "MAPE": "MAPE",
              "RMSE_log": "RMSE_log", "median_ratio": "median_ratio", "mean_ratio": "mean_ratio",
              "weighted_mean_ratio": "weighted_mean_ratio", "COD": "COD", "COV": "COV",
              "PRD": "PRD", "PRB": "PRB", "MKI": "MKI", "VEI": "VEI", "beta_log": "Beta_log",
              "Cov_log_residual_log_price": "Cov_log_residual_log_price",
              "Delta_NL": "Delta_NL", "dCor_e_y": "dCor_e_y"}
    for _, r in frozen_sel.iterrows():
        for evname, suffix in [("fold_%d" % k, "fold_%d" % k) for k in range(1, 8)] + \
                              [("CV_mean", "CV_mean"), ("CV_SD", "CV_sd"),
                               ("heldout", "heldout"), ("forward_2025", "forward_2025")]:
            row = {"cell_id": "FROZEN", "display_name": f"frozen path table: {r.family}",
                   "role": "frozen cross-reference (not a Stage-1.5 fit)",
                   "family": r.family, "rho": r.rho, "evaluation": evname, "n": None,
                   "source_artifact": str(v4.relative_to(c.REPO)),
                   "config_hash": r.get("config_id"), "prediction_hash": None,
                   "block_id": None, "cell_c_implementation": None,
                   "legacy_stage1_label": None}
            for out_k, base_k in metmap.items():
                col = f"{base_k}__{suffix}"
                row[out_k] = float(r[col]) if col in fp.columns and pd.notna(r.get(col)) else None
            rows.append(row)

    zc = pd.DataFrame(rows)
    order = ["cell_id", "display_name", "role", "family", "rho", "evaluation", "n",
             "R2_price", "MAE_price", "MAPE", "RMSE_log", "median_ratio", "mean_ratio",
             "weighted_mean_ratio", "COD", "COV", "PRD", "PRB", "MKI", "VEI",
             "beta_log", "Cov_log_residual_log_price", "Delta_NL", "Delta_NL_raw", "dCor_e_y",
             "source_artifact", "config_hash", "prediction_hash", "block_id",
             "cell_c_implementation", "legacy_stage1_label"]
    zc = zc.reindex(columns=[k for k in order if k in zc.columns]
                    + [k for k in zc.columns if k not in order])
    c.write_table(zc, T / "zero_control_full.csv")
    print(f"[zc] wrote zero_control_full.csv  ({len(zc)} rows)")

    # b-star diagnostics summary
    bs = zf[["cell_id", "forward_display_name", "role", "block_id", "block_kind", "n_T",
             "ybar_T", "f0bar_T", "f0bar_minus_ybar", "Var_T_y_ddof0", "Var_T_f0_ddof0",
             "Cov_T_f0_y", "Cov_T_f0_minus_y_y", "beta_log_train", "R2_log_insample",
             "one_over_R2_log_theoretical_diagnostic_only", "b_star_train", "cov_positive",
             "b_star_finite", "identity_beta_log_vs_bstar_absdiff"]].copy()
    c.write_table(bs, T / "b_star_diagnostics.csv")

    gate = {
        "n_fits": int(len(zf)),
        "all_27_complete": bool(len(zf) == 27),
        "cov_positive_all_A_and_C": bool(zf[zf.cell_id.isin(["A", "C"])].cov_positive.all()),
        "cov_positive_all_cells": bool(zf.cov_positive.all()),
        "b_star_finite_all_A_and_C": bool(zf[zf.cell_id.isin(["A", "C"])].b_star_finite.all()),
        "b_star_finite_all_cells": bool(zf.b_star_finite.all()),
        "identity_max_absdiff": float(zf.identity_beta_log_vs_bstar_absdiff.max()),
        "qc_A_C_tiers": qdf[qdf.reproduction_tier.isin(["R1", "R2", "R3", "R4"])]
                        .reproduction_tier.value_counts().to_dict(),
        "qc_ladder_crosschecks_all_match": bool(
            qdf[qdf.reproduction_tier == "AGGREGATE_CROSSCHECK"].matches_stage1_ladder.all()),
        "zero_control_rows": int(len(zc)),
        "cell_c_implementation": zf[zf.cell_id == "C"].iloc[0].get("cell_c_implementation"),
    }
    c.write_json(T / "gate_g2_checks.json", gate)
    print(json.dumps(gate, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
