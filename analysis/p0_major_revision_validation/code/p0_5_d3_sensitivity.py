#!/usr/bin/env python3
"""Bounded D3 matched sensitivity: do the D1-primary conclusions survive on the
row-balanced development coordinate?

D1 stays PRIMARY. This asks only whether re-running the identical deterministic K=6
construction on D3 would change any conclusion the matched-beta comparison draws.
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
MET = ["R2_price", "MAE_price", "MAPE", "RMSE_log", "median_ratio", "mean_ratio",
       "weighted_mean_ratio", "COD", "COV", "PRD", "PRB", "MKI", "VEI",
       "beta_log", "Cov_log_residual_log_price", "Delta_NL", "dCor_e_y"]
HEADLINE = ["R2_price", "RMSE_log", "MAE_price", "COD", "Delta_NL", "dCor_e_y", "beta_log"]
# Materiality floor per metric: the smallest difference that could change a digit the
# manuscript actually displays. A sign disagreement between D1 and D3 in which BOTH
# differences sit below this floor is not a conclusion change -- it is two ways of
# writing "no difference". Frozen here, not tuned after inspecting the flips.
FLOOR = {"R2_price": 5e-4, "RMSE_log": 5e-4, "MAE_price": 100.0, "COD": 5e-2,
         "Delta_NL": 1e-3, "dCor_e_y": 1e-3, "beta_log": 1e-3}
STRUCTURAL_ZERO = 1e-12   # the matched coordinate forces an exact zero
PRIMARY = ["Direct", "Surrogate", "C-posthoc"]
PAIRS = ["Direct_minus_Cposthoc", "Surrogate_minus_Cposthoc", "Surrogate_minus_Direct"]


def main() -> int:
    d1 = pd.read_csv(T / "matched_beta_comparison.csv")
    d3 = pd.read_csv(T / "matched_beta_comparison_d3.csv")
    p1 = pd.read_csv(T / "matched_beta_pairwise_deltas.csv")
    p3 = pd.read_csv(T / "matched_beta_pairwise_deltas_d3.csv")
    F1 = json.loads((c.CONFIGS / "matched_beta_frozen.json").read_text())
    F3 = json.loads((c.CONFIGS / "matched_beta_d3_frozen.json").read_text())

    # ---------------------------------------------- per-cell selection + metric shift
    rows = []
    for j in range(F1["K_core"]):
        for fam in PRIMARY + ["A-posthoc"]:
            s1 = {r["family"]: r for r in F1["matched_configurations"] if r["j"] == j}[fam]
            s3 = {r["family"]: r for r in F3["matched_configurations"] if r["j"] == j}[fam]
            same = ((s1["rho"] == s3["rho"]) if s1["rho"] is not None else
                    (abs(float(s1["b"]) - float(s3["b"])) < 1e-12))
            for reg in ["CV_mean", "CV_SD", "heldout", "forward_2025"]:
                a = d1[(d1.j == j) & (d1.family == fam) & (d1.evaluation == reg)]
                b = d3[(d3.j == j) & (d3.family == fam) & (d3.evaluation == reg)]
                if not len(a) or not len(b):
                    continue
                a, b = a.iloc[0], b.iloc[0]
                r = {"j": j, "family": fam, "role": s1["role"], "evaluation": reg,
                     "D1_target": s1["target"], "D3_target": s3["target"],
                     "D1_rho": s1["rho"], "D3_rho": s3["rho"],
                     "D1_b": s1["b"], "D3_b": s3["b"],
                     "same_configuration_selected": bool(same),
                     "D1_match_mode": s1["match_mode"], "D3_match_mode": s3["match_mode"]}
                for m in HEADLINE:
                    va, vb = a[m], b[m]
                    r[f"{m}__D1"] = None if pd.isna(va) else float(va)
                    r[f"{m}__D3"] = None if pd.isna(vb) else float(vb)
                    r[f"{m}__D3_minus_D1"] = (None if (pd.isna(va) or pd.isna(vb))
                                              else float(vb) - float(va))
                rows.append(r)
    sens = pd.DataFrame(rows)
    c.write_table(sens, T / "matched_beta_d3_sensitivity.csv")

    # ------------------------------------ conclusion-level check: do delta SIGNS agree?
    sign_rows = []
    for reg in ["CV_mean", "heldout", "forward_2025"]:
        for pair in PAIRS:
            for m in HEADLINE:
                for j in range(1, F1["K_core"]):     # j=0 is the structural origin identity
                    a = p1[(p1.j == j) & (p1.pair == pair) & (p1.evaluation == reg)]
                    b = p3[(p3.j == j) & (p3.pair == pair) & (p3.evaluation == reg)]
                    if not len(a) or not len(b):
                        continue
                    va, vb = a.iloc[0][m], b.iloc[0][m]
                    if pd.isna(va) or pd.isna(vb):
                        continue
                    va, vb = float(va), float(vb)
                    fl = FLOOR[m]
                    if abs(va) <= STRUCTURAL_ZERO or abs(vb) <= STRUCTURAL_ZERO:
                        cls = "structural_zero"
                    elif abs(va) < fl and abs(vb) < fl:
                        cls = "both_below_floor"
                    elif np.sign(va) == np.sign(vb):
                        cls = "sign_agrees"
                    else:
                        cls = "MATERIAL_SIGN_FLIP"
                    sign_rows.append({
                        "evaluation": reg, "pair": pair, "metric": m, "j": j,
                        "delta_D1": va, "delta_D3": vb,
                        "sign_D1": int(np.sign(va)), "sign_D3": int(np.sign(vb)),
                        "materiality_floor": fl,
                        "classification": cls,
                        "sign_agrees_raw": bool(np.sign(va) == np.sign(vb)),
                        "abs_change": abs(vb - va)})
    sg = pd.DataFrame(sign_rows)
    c.write_table(sg, T / "matched_beta_d3_sign_agreement.csv")

    flips = sg[sg.classification == "MATERIAL_SIGN_FLIP"]
    cls_counts = sg.classification.value_counts().to_dict()
    summary = {
        "primary_coordinate": "D1 (equal-weight seven-fold mean) -- UNCHANGED",
        "sensitivity_coordinate": "D3 (row-balanced pooled OOF, w_ik = 1/m_i)",
        "D3_MATERIAL": True,
        "why_material": ("max|D1-D3| = 0.004867 > tau = 0.002, and the K=6 construction "
                         "selects a different fitted Direct config at j=2 "
                         "(rho 1.930698 under D3 vs 1.676833 under D1)"),
        "configurations_reselected": int((~sens.drop_duplicates(["j", "family"])
                                          .same_configuration_selected).sum()),
        "configurations_total": int(len(sens.drop_duplicates(["j", "family"]))),
        "headline_sign_comparisons": int(len(sg)),
        "classification_counts": {k: int(v) for k, v in cls_counts.items()},
        "materiality_floors": FLOOR,
        "material_sign_flips": int(len(flips)),
        "raw_sign_agreement_rate": (float(sg.sign_agrees_raw.mean()) if len(sg) else None),
        "note_on_structural_zeros": (
            "Under D1 the match forces the CV_mean beta_log delta between a family and "
            "C-posthoc to be exactly zero by construction, so its 'sign' carries no "
            "information; those comparisons are classed structural_zero, not flips."),
        "max_headline_metric_shift": {
            m: float(np.nanmax(np.abs(sens[f"{m}__D3_minus_D1"].astype(float)))) for m in HEADLINE},
        "material_sign_flips_detail": flips[["evaluation", "pair", "metric", "j",
                                             "delta_D1", "delta_D3"]].to_dict("records"),
    }
    c.write_json(c.TABLES.parent / "tables" / "matched_beta_d3_sensitivity_summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items()
                      if k != "material_sign_flips_detail"}, indent=2))
    print(f"\nMATERIAL sign flips ({len(flips)} of {len(sg)}):")
    if len(flips):
        print(flips[["evaluation", "pair", "metric", "j", "delta_D1", "delta_D3"]]
              .to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
