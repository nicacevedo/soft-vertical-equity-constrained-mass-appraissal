#!/usr/bin/env python3
"""P0-1 -- Direct/Surrogate objective and curvature audit on REAL fold data.

No model fits.  One target column read plus canonical index reconstruction.

Emits (plan E.3):
  tables/objective_scaling_audit.csv       per fitting block: n, ybar, Var(y), c moments
  tables/direct_hessian_magnitudes.csv     per block x rho: supplied diag vs exact rank-one
  tables/surrogate_weight_distribution.csv per block x rho: w=1+rho c^2 concentration
  tables/effective_leaf_shrinkage.csv      STYLIZED nominal-leaf fixed-lambda diagnostic
  tables/label_quantization.csv            float32 label representation, raw vs centered
  reports/P0_IMPLEMENTATION_AUDIT.md

E.4 verdict (ACCEPT / REJECT / INDETERMINATE) for:
  "Direct is effectively gradient-only, retaining essentially native curvature".
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p0_common as c


# ---------------------------------------------------------------- E.4 rule
M_ACCEPT_MAX = 1e-2     # max_i (rho/2n) c_i^2 must stay below this for ACCEPT
M_REJECT_MIN = 5e-2     # at/above this the supplied diagonal is doing real work
R_ACCEPT_MIN = 1.20     # exact/supplied directional curvature ratio at rho >= 1
R_REJECT_MAX = 1.05     # below this throughout, the diagonal is a good approximation

# Frozen absolute L2 leaf penalty (stylized diagnostic only).
REG_LAMBDA = 10.857207234911057
MIN_CHILD_SAMPLES = 93
MIN_SUM_HESSIAN_DEFAULT = 1e-3   # LightGBM default; never set in the frozen config


def _moments(c_vec: np.ndarray) -> dict:
    n = int(c_vec.size)
    c2 = c_vec ** 2
    return {
        "n": n,
        "var_y_ddof0": float(np.mean(c2)),          # == Var_T(y), c is already centered
        "sum_c2": float(np.sum(c2)),
        "sum_c4": float(np.sum(c2 ** 2)),
        "c2_min": float(np.min(c2)),
        "c2_median": float(np.median(c2)),
        "c2_mean": float(np.mean(c2)),
        "c2_p99": float(np.percentile(c2, 99)),
        "c2_max": float(np.max(c2)),
    }


def run() -> int:
    print("[S2] loading canonical splits ...", flush=True)
    df_tv, df_test, df_assess, _pred, _cat = c.load_canonical_splits()
    folds, archive = c.rebuild_folds(df_tv)

    y_dev_log = np.log(df_tv[c.TARGET_COL].to_numpy(dtype=float))
    y_prod_log = np.concatenate(
        [y_dev_log, np.log(df_test[c.TARGET_COL].to_numpy(dtype=float))]
    )
    blocks = c.fitting_blocks(df_tv, df_test, folds)
    rho_grid = c.frozen_rho_grid()
    anchors = set(c.DISPLAY_ANCHORS)

    scaling_rows, direct_rows, surr_rows, leaf_rows, quant_rows = [], [], [], [], []

    for blk in blocks:
        y = c.block_log_target(blk, y_dev_log, y_prod_log)
        ybar = float(np.mean(y))
        cv = y - ybar
        mom = _moments(cv)
        n = mom["n"]
        var_y = mom["var_y_ddof0"]

        if n != blk["n"]:
            raise c.ProtocolViolation(f"block {blk['block_id']}: n mismatch {n} vs {blk['n']}")

        scaling_rows.append({
            "block_id": blk["block_id"], "block_kind": blk["block_kind"],
            "fold_id": blk["fold_id"], "n_T": n, "ybar_T": ybar, **{
                k: v for k, v in mom.items() if k != "n"},
            "objective_scaling": "n/2 applied analytically: grad=e, hess=1 at rho=0",
            "direct_supplied_hess": "1 + (rho/(2n)) c_i^2",
            "direct_exact_scaled_hessian": "I + (rho/(2n)) c c^T",
            "surrogate_supplied_hess": "1 + rho c_i^2",
        })

        # ---- float32 label quantisation (H1 evidence, real data) -------------
        err_raw = np.float32(y).astype(np.float64) - y
        err_cen = (np.float32(cv).astype(np.float64) + ybar) - y
        quant_rows.append({
            "block_id": blk["block_id"], "n_T": n, "ybar_T": ybar,
            "raw_label_abs_err_mean": float(np.mean(np.abs(err_raw))),
            "raw_label_abs_err_max": float(np.max(np.abs(err_raw))),
            "centered_label_abs_err_mean": float(np.mean(np.abs(err_cen))),
            "centered_label_abs_err_max": float(np.max(np.abs(err_cen))),
            "effective_label_gap_mean": float(np.mean(np.abs(err_raw - err_cen))),
            "effective_label_gap_max": float(np.max(np.abs(err_raw - err_cen))),
            "note": "LightGBM Dataset stores labels as float32; Cell A sees float32(y), Cells B/C see float32(y-ybar).",
        })

        for rho in rho_grid:
            is_anchor = any(abs(rho - a) < 1e-12 for a in anchors)
            # ---- Direct -------------------------------------------------------
            incr = (rho / (2.0 * n)) * cv ** 2          # penalty-only diagonal increment
            h = 1.0 + incr
            # exact rank-one eigenvalue along c/||c||
            exact_dir = 1.0 + 0.5 * rho * var_y
            # supplied directional curvature along the same unit vector
            supplied_dir = 1.0 + (rho / (2.0 * n)) * (mom["sum_c4"] / mom["sum_c2"])
            direct_rows.append({
                "block_id": blk["block_id"], "fold_id": blk["fold_id"], "n_T": n,
                "rho": rho, "is_display_anchor": is_anchor, "var_y_T": var_y,
                "h_min": float(np.min(h)), "h_median": float(np.median(h)),
                "h_mean": float(np.mean(h)), "h_p99": float(np.percentile(h, 99)),
                "h_max": float(np.max(h)),
                "incr_min": float(np.min(incr)), "incr_median": float(np.median(incr)),
                "incr_max": float(np.max(incr)),
                "max_penalty_contribution_M": float(np.max(incr)),
                "penalty_share_of_curvature_median": float(np.median(incr / h)),
                "penalty_share_of_curvature_max": float(np.max(incr / h)),
                "exact_directional_curvature": exact_dir,
                "supplied_directional_curvature": supplied_dir,
                "curvature_ratio_exact_over_supplied": exact_dir / supplied_dir,
                "min_sum_hessian_in_leaf_default": MIN_SUM_HESSIAN_DEFAULT,
                "h_min_ge_1": bool(np.min(h) >= 1.0),
            })

            # ---- Surrogate ----------------------------------------------------
            w = 1.0 + rho * cv ** 2
            sw = float(np.sum(w))
            order = np.argsort(w)[::-1]
            k1 = max(1, int(np.ceil(0.01 * n)))
            top1_share = float(np.sum(w[order[:k1]]) / sw)
            ess = float(sw ** 2 / np.sum(w ** 2))
            surr_rows.append({
                "block_id": blk["block_id"], "fold_id": blk["fold_id"], "n_T": n,
                "rho": rho, "is_display_anchor": is_anchor,
                "w_min": float(np.min(w)), "w_median": float(np.median(w)),
                "w_mean": float(np.mean(w)), "w_p99": float(np.percentile(w, 99)),
                "w_max": float(np.max(w)),
                "top1pct_weight_share": top1_share,
                "effective_sample_size": ess,
                "ess_fraction_of_n": ess / n,
            })

            # ---- STYLIZED fixed-lambda nominal-leaf diagnostic ----------------
            # Three nominal leaves of MIN_CHILD_SAMPLES rows: uniform-random,
            # lowest-c^2 decile, highest-c^2 decile.  Illustrative only.
            rng = np.random.default_rng(2025)
            c2 = cv ** 2
            dec = np.argsort(c2)
            lo = dec[: max(k1, MIN_CHILD_SAMPLES)]
            hi = dec[-max(k1, MIN_CHILD_SAMPLES):]
            for tag, idx in (
                ("uniform_random", rng.choice(n, size=MIN_CHILD_SAMPLES, replace=False)),
                ("low_c2_decile", rng.choice(lo, size=MIN_CHILD_SAMPLES, replace=False)),
                ("high_c2_decile", rng.choice(hi, size=MIN_CHILD_SAMPLES, replace=False)),
            ):
                Hd = float(np.sum(1.0 + (rho / (2.0 * n)) * c2[idx]))
                Hs = float(np.sum(1.0 + rho * c2[idx]))
                leaf_rows.append({
                    "block_id": blk["block_id"], "fold_id": blk["fold_id"],
                    "rho": rho, "is_display_anchor": is_anchor, "leaf_draw": tag,
                    "min_child_samples": MIN_CHILD_SAMPLES, "reg_lambda": REG_LAMBDA,
                    "H_leaf_direct": Hd, "H_leaf_surrogate": Hs,
                    "shrink_direct": Hd / (Hd + REG_LAMBDA),
                    "shrink_surrogate": Hs / (Hs + REG_LAMBDA),
                    "shrink_ratio_surr_over_direct": (Hs / (Hs + REG_LAMBDA)) / (Hd / (Hd + REG_LAMBDA)),
                    "min_sum_hessian_binding_direct": bool(Hd < MIN_SUM_HESSIAN_DEFAULT),
                    "min_sum_hessian_binding_surrogate": bool(Hs < MIN_SUM_HESSIAN_DEFAULT),
                    "diagnostic_type": "stylized_nominal_leaf",
                })
        print(f"[S2] block {blk['block_id']:<24s} n={n:>7d} Var(y)={var_y:.6f}", flush=True)

    df_scaling = pd.DataFrame(scaling_rows)
    df_direct = pd.DataFrame(direct_rows)
    df_surr = pd.DataFrame(surr_rows)
    df_leaf = pd.DataFrame(leaf_rows)
    df_quant = pd.DataFrame(quant_rows)

    c.write_table(df_scaling, c.TABLES / "objective_scaling_audit.csv")
    c.write_table(df_direct, c.TABLES / "direct_hessian_magnitudes.csv")
    c.write_table(df_surr, c.TABLES / "surrogate_weight_distribution.csv")
    c.write_table(df_leaf, c.TABLES / "effective_leaf_shrinkage.csv")
    c.write_table(df_quant, c.TABLES / "label_quantization.csv")

    # ------------------------------------------------------------- E.4 verdict
    disp = df_direct[df_direct["is_display_anchor"]]
    M_max = float(disp["max_penalty_contribution_M"].max())
    ratios_ge1 = disp[disp["rho"] >= 1.0]["curvature_ratio_exact_over_supplied"]
    R_min_at_rho_ge1 = float(ratios_ge1.min()) if len(ratios_ge1) else float("nan")
    R_max_all = float(disp["curvature_ratio_exact_over_supplied"].max())

    if M_max >= M_REJECT_MIN or R_max_all < R_REJECT_MAX:
        verdict = "REJECT"
    elif M_max < M_ACCEPT_MAX and (len(ratios_ge1) > 0 and R_min_at_rho_ge1 >= R_ACCEPT_MIN):
        verdict = "ACCEPT"
    else:
        verdict = "INDETERMINATE"

    min_h = float(df_direct["h_min"].min())
    msh_binding = bool(df_leaf["min_sum_hessian_binding_direct"].any() or
                       df_leaf["min_sum_hessian_binding_surrogate"].any())

    summary = {
        "e4_verdict": verdict,
        "statement": "Direct is effectively gradient-only, retaining essentially native curvature",
        "rule": {"M_accept_max": M_ACCEPT_MAX, "M_reject_min": M_REJECT_MIN,
                 "R_accept_min_at_rho_ge_1": R_ACCEPT_MIN, "R_reject_max": R_REJECT_MAX},
        "evidence_at_display_anchors": {
            "max_penalty_contribution_M_over_all_blocks_and_anchors": M_max,
            "min_curvature_ratio_at_rho_ge_1": R_min_at_rho_ge1,
            "max_curvature_ratio": R_max_all,
        },
        "min_supplied_hessian_over_everything": min_h,
        "min_sum_hessian_in_leaf": {
            "value_used_by_lightgbm": MIN_SUM_HESSIAN_DEFAULT,
            "set_in_frozen_config": False,
            "binding_anywhere_in_stylized_leaves": msh_binding,
            "conclusion": "checked and non-binding" if not msh_binding else "BINDING -- escalate",
        },
        "n_blocks": int(len(blocks)), "n_rho": int(len(rho_grid)),
        "provenance": c.preflight_block(),
    }
    c.write_json(c.TABLES / "e4_verdict.json", summary)
    print(json.dumps({k: summary[k] for k in
                      ("e4_verdict", "evidence_at_display_anchors",
                       "min_supplied_hessian_over_everything")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
