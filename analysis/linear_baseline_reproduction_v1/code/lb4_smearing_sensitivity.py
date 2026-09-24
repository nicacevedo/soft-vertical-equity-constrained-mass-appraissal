#!/usr/bin/env python3
"""LB Stage 4 -- does the paper's retransformation alternative flip the accuracy ordering?

The manuscript's alternative to direct exponentiation is the frozen development-only
Duan smearing factor of Appendix~\\ref{app:smearing}:
    u = y_true_log - y_pred_log,  s = sum(w u-exp) / sum(w),  y_pred_log -> y_pred_log + log s,
with w the D3 row-balanced weights (one development sale row, one vote) estimated on
the seven development out-of-fold validation blocks and never re-estimated on an
evaluation block.  Both the estimator and the weighting helper are IMPORTED from
analysis/p1_inferential_reporting, not re-implemented.

Question answered: whether the linear-versus-LightGBM ordering on R2_P, MAE_P and
MAPE survives the alternative convention.  This is a convention check, not a result,
and it selects nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lb_common as lb

sys.path.insert(0, str(lb.P1_DIR / "code"))
import p1_common as p1                                                   # noqa: E402
from p1_4_smearing_sensitivity import smearing_factor                    # noqa: E402

DEV_FOLDS = [f"fold_{k}" for k in range(1, 8)]


def main() -> int:
    p1.assert_d3_multiplicity_identities()

    # ---- estimate s on development out-of-fold residuals only ---------------
    pooled = {lb.LINEAR_NAME: [], lb.NATIVE_NAME: []}
    for ev in DEV_FOLDS:
        d = lb.load_pair(ev)
        for model, pred in d["pred"].items():
            pooled[model].append(pd.DataFrame({"row_id": d["row_id"],
                                               "y_true_log": d["y_true_log"],
                                               "y_pred_log": pred}))
    srows = []
    for model, parts in pooled.items():
        dev = pd.concat(parts, ignore_index=True)
        rid = dev.row_id.to_numpy()
        u = dev.y_true_log.to_numpy() - dev.y_pred_log.to_numpy()        # = -e, Duan's sign
        w = p1.d3_weights_for_pooled(rid)
        s = smearing_factor(u, w)
        s_naive = smearing_factor(u, np.ones_like(w))
        uniq, cnt = np.unique(rid, return_counts=True)
        if int(rid.size) != p1.D3_N_APPEARANCES or int(uniq.size) != p1.D3_N_UNIQUE:
            raise lb.ProtocolViolation(
                f"{model}: dev OOF structure {rid.size}/{uniq.size} != "
                f"{p1.D3_N_APPEARANCES}/{p1.D3_N_UNIQUE}")
        srows.append({"model": model, "s": s, "s_naive_pooled": s_naive,
                      "log_s": float(np.log(s)),
                      "n_appearances": int(rid.size), "n_unique": int(uniq.size),
                      "estimation_sample": "|".join(DEV_FOLDS),
                      "oos_used_in_estimation": False,
                      "weighting": "D3 row-balanced, w_ik = 1/m_i"})
        print(f"[lb4] {model:<46s} s={s:.6f}  s_naive={s_naive:.6f}", flush=True)
    sf = pd.DataFrame(srows)

    # cross-check the LightGBM arm against the frozen P1 value
    frozen = pd.read_csv(lb.P1_DIR / "tables" / "smearing_factor_provenance.csv")
    fa = frozen[frozen.realization_key == "fit:LGBMRegressor:252a25d9c0ce796b"]
    s_frozen = float(fa.iloc[0]["s"]) if len(fa) else float("nan")
    s_mine = float(sf[sf.model == lb.NATIVE_NAME].iloc[0]["s"])
    sf["frozen_p1_s_for_this_arm"] = [np.nan, s_frozen]
    sf["agrees_with_frozen_p1"] = [None, bool(abs(s_mine - s_frozen) <= 1e-12)]
    print(f"[lb4] LightGBM s vs frozen P1: {s_mine:.12f} vs {s_frozen:.12f} "
          f"(|delta| {abs(s_mine - s_frozen):.3e})")

    # ---- apply unchanged to both out-of-time evaluations --------------------
    base = pd.read_csv(lb.TABLES / "lb_metrics_full_precision_all_blocks.csv")
    rows = []
    for ev in ("heldout", "forward_2025"):
        d = lb.load_pair(ev)
        for model, pred in d["pred"].items():
            s = float(sf[sf.model == model].iloc[0]["s"])
            m0 = lb.metrics_from(d["y_true_log"], pred, d["y_train_log"], d["row_id"])
            m1 = lb.metrics_from(d["y_true_log"], pred + np.log(s), d["y_train_log"], d["row_id"])
            r = {"model": model, "evaluation": ev, "n": d["n"], "s": s}
            for k in ("R2_price", "MAE_price", "MAPE", "median_ratio", "mean_ratio",
                      "weighted_mean_ratio", "COD", "COV", "PRD", "PRB", "MKI", "VEI",
                      "beta_log", "Delta_NL", "dCor_e_y"):
                r[f"{k}_direct_exp"] = m0[k]
                r[f"{k}_smeared"] = m1[k]
            rows.append(r)
    sens = pd.DataFrame(rows)

    # ---- invariance audit and ordering verdict ------------------------------
    inv_keys = ["COD", "COV", "PRD", "PRB", "MKI", "VEI", "beta_log", "Delta_NL", "dCor_e_y"]
    scale_keys = ["median_ratio", "mean_ratio", "weighted_mean_ratio"]
    inv = []
    for _, r in sens.iterrows():
        for k in inv_keys:
            a, b = r[f"{k}_direct_exp"], r[f"{k}_smeared"]
            inv.append({"model": r.model, "evaluation": r.evaluation, "metric": k,
                        "expected": "invariant", "direct_exp": a, "smeared": b,
                        "abs_delta": abs(b - a),
                        "rel_delta": abs(b - a) / max(abs(a), 1e-300)})
        for k in scale_keys:
            a, b = r[f"{k}_direct_exp"], r[f"{k}_smeared"]
            inv.append({"model": r.model, "evaluation": r.evaluation, "metric": k,
                        "expected": "scales by s", "direct_exp": a, "smeared": b,
                        "abs_delta": abs(b - a * r.s),
                        "rel_delta": abs(b - a * r.s) / max(abs(a * r.s), 1e-300)})
    inv = pd.DataFrame(inv)

    verdicts = []
    for ev in ("heldout", "forward_2025"):
        sub = sens[sens.evaluation == ev].set_index("model")
        for metric, better in (("R2_price", "higher"), ("MAE_price", "lower"),
                               ("MAPE", "lower")):
            a = sub.loc[lb.LINEAR_NAME, f"{metric}_direct_exp"]
            b = sub.loc[lb.NATIVE_NAME, f"{metric}_direct_exp"]
            a2 = sub.loc[lb.LINEAR_NAME, f"{metric}_smeared"]
            b2 = sub.loc[lb.NATIVE_NAME, f"{metric}_smeared"]
            win = (lambda x, y: lb.NATIVE_NAME if ((y > x) if better == "higher" else (y < x))
                   else lb.LINEAR_NAME)
            verdicts.append({
                "evaluation": ev, "metric": metric, "better_is": better,
                "linear_direct_exp": a, "lightgbm_direct_exp": b,
                "linear_smeared": a2, "lightgbm_smeared": b2,
                "winner_direct_exp": win(a, b), "winner_smeared": win(a2, b2),
                "ordering_preserved": win(a, b) == win(a2, b2),
            })
    verdicts = pd.DataFrame(verdicts)

    lb.write_table(sf, lb.TABLES / "lb_smearing_factors.csv")
    lb.write_table(sens, lb.TABLES / "lb_smearing_sensitivity.csv")
    lb.write_table(inv, lb.TABLES / "lb_smearing_invariance_audit.csv")
    lb.write_table(verdicts, lb.TABLES / "lb_smearing_ordering_verdict.csv")

    lb.write_json(lb.PROVENANCE / "lb4_smearing.json", {
        "estimator": "frozen Duan smearing, analysis/p1_inferential_reporting/configs/"
                     "smearing_estimator_frozen.json",
        "estimator_imported_from": "p1_4_smearing_sensitivity.smearing_factor",
        "weights_imported_from": "p1_common.d3_weights_for_pooled",
        "s_linear": float(sf[sf.model == lb.LINEAR_NAME].iloc[0]["s"]),
        "s_lightgbm": s_mine,
        "s_lightgbm_frozen_p1": s_frozen,
        "s_lightgbm_reproduces_frozen_p1": bool(abs(s_mine - s_frozen) <= 1e-12),
        "max_invariance_rel_delta": float(inv[inv.expected == "invariant"].rel_delta.max()),
        "max_scaling_rel_delta": float(inv[inv.expected == "scales by s"].rel_delta.max()),
        "accuracy_ordering_preserved_everywhere": bool(verdicts.ordering_preserved.all()),
        "verdict": ("The qualitative linear-versus-LightGBM accuracy ordering is unchanged "
                    "under the paper's retransformation alternative on both evaluations."
                    if bool(verdicts.ordering_preserved.all()) else
                    "The accuracy ordering changes under the alternative retransformation."),
        "scope": "convention check only; nothing refitted, nothing selected, no headline replaced",
        "limitation": ("s is estimated on development out-of-fold residuals and applied "
                       "unchanged to the 2025 block, whose fitting set has no out-of-fold "
                       "analogue; that is the assumption the frozen P1 design already records."),
    })

    pd.set_option("display.width", 220)
    print("\n" + verdicts.to_string(index=False))
    print(f"\n[lb4] max invariance rel delta: "
          f"{inv[inv.expected=='invariant'].rel_delta.max():.3e}")
    print(f"[lb4] max scaling  rel delta: "
          f"{inv[inv.expected=='scales by s'].rel_delta.max():.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
