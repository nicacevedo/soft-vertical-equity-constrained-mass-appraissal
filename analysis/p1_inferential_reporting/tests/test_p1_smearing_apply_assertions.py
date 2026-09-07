#!/usr/bin/env python3
"""Assertions for P1 Task 4 (Duan smearing): freeze, estimate and APPLY.

Read-only: recomputes no smearing factor and re-reads no prediction artifact.
Complements tests/test_smearing_sign.py, which covers the sign convention with
synthetic cases.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
import p1_common as c                                                   # noqa: E402

FZ = json.loads((c.CONFIGS / "smearing_estimator_frozen.json").read_text())
FZH = json.loads((c.CONFIGS / "smearing_estimator_frozen_hash.json").read_text())
PROV = pd.read_csv(c.TABLES / "smearing_factor_provenance.csv")
INV = pd.read_csv(c.TABLES / "smearing_apply_invariance.csv")
ED2 = pd.read_csv(c.TABLES / "smearing_apply_ed2_stability.csv")
DNL = pd.read_csv(c.TABLES / "smearing_apply_delta_nl_subset.csv")
FCR = pd.read_csv(c.TABLES / "smearing_apply_fastpath_reconciliation.csv")
SM = json.loads((c.TABLES / "smearing_apply_summary.json").read_text())

FOLDS = [f"fold_{k}" for k in range(1, 8)]
ALL_EVALS = FOLDS + ["pooled_oof", "heldout", "forward_2025"]
TOL = 1e-9


# ------------------------------------------------------------------- freeze
def test_frozen_estimator_hash_is_intact():
    p = c.CONFIGS / "smearing_estimator_frozen.json"
    assert FZH["file"] == p.name
    assert FZH["file_sha256"] == c.sha256_file(p), "frozen estimator changed after hashing"
    assert FZH["frozen_before_any_oos_read"] is True
    assert FZ["frozen_before_any_heldout_or_2025_output_is_read"] is True


def test_duan_sign_is_the_opposite_of_the_repo_residual_convention():
    assert FZ["residual_convention_in_repo"] == "e = y_pred_log - y_true_log"
    assert FZ["duan_log_error"] == "u_ik = y_true_log - y_pred_log = -e_ik"
    assert FZ["formula"] == "s = sum_ik( w_ik * exp(u_ik) ) / sum_ik( w_ik )"
    assert "exp(-(y_pred_log - y_true_log))" in FZ["formula_expanded"]
    assert FZ["forbidden_formula"].startswith("s = mean(exp(e_ik))")
    assert "WRONG SIGN" in FZ["forbidden_formula"]
    # the live source really implements exp(+u), never exp(e)
    src = (c.P1_DIR / "code" / "p1_4_smearing_sensitivity.py").read_text()
    assert "u = dev.y_true_log.to_numpy() - dev.y_pred_log.to_numpy()" in src
    assert "np.sum(w * np.exp(u)) / np.sum(w)" in src
    assert "np.exp(e)" not in src
    # and the applied shift is +log(s), i.e. y_hat -> s * y_hat
    assert "p_s = p + math.log(s)" in src
    assert FZ["application"].startswith("y_hat_level = s * exp(f(x))")


def test_sign_direction_is_consistent_with_the_estimated_factors():
    """u = y_true - y_pred, so s > 1 iff predictions sit below sale prices on
    average. A sign flip would give s' = E[exp(-u)] and s*s' >= 1 by Jensen, so
    a wrong sign could not produce these values together with mean_u > 0."""
    assert (PROV.s > 0).all()
    pos = PROV[PROV.mean_u > 0]
    assert len(pos) > 0
    assert (pos.s > 1.0).all(), "s must exceed 1 where the weighted mean log error is positive"
    # s >= exp(mean_u) by Jensen, for every realization
    assert (PROV.s >= np.exp(PROV.mean_u) - 1e-12).all()


# ----------------------------------------------------------------- estimate
def test_estimation_used_development_out_of_fold_data_only():
    assert FZ["estimation_sample"].startswith("DEVELOPMENT out-of-fold residuals only")
    assert "prohibited" in FZ and "heldout" in FZ["prohibited"]
    assert not PROV.oos_used_in_estimation.any()
    assert (PROV.s_source == "dev_oof_row_balanced_D3").all()
    assert (PROV.estimation_evaluations == "|".join(FOLDS)).all()
    for bad in ("heldout", "forward_2025", "pooled_oof"):
        assert not PROV.estimation_evaluations.str.contains(bad).any(), bad


def test_freeze_preceded_the_estimate_and_the_application():
    """Content evidence first: the config is hash-pinned, so it cannot have been
    rewritten after the fact, and it names a development-only estimation sample.
    Filesystem mtimes corroborate the ordering freeze -> estimate -> apply."""
    ts = pd.Timestamp(FZH["frozen_at_utc"]).tz_convert("UTC")
    for name in ("smearing_factor_provenance.csv", "smearing_apply_invariance.csv"):
        m = pd.Timestamp(( c.TABLES / name).stat().st_mtime, unit="s", tz="UTC")
        assert ts <= m, (name, ts, m)
    cfg = pd.Timestamp((c.CONFIGS / "smearing_estimator_frozen.json").stat().st_mtime,
                       unit="s", tz="UTC")
    assert cfg <= pd.Timestamp((c.TABLES / "smearing_factor_provenance.csv").stat().st_mtime,
                               unit="s", tz="UTC")


def test_d3_multiplicity_identities_and_row_balance():
    c.assert_d3_multiplicity_identities()
    assert (PROV.n_appearances == c.D3_N_APPEARANCES).all()
    assert (PROV.n_unique == c.D3_N_UNIQUE).all()
    assert (PROV.unique_rows_multiplicity_1 == c.D3_UNIQUE_ROWS_MULT_1).all()
    assert (PROV.duplicated_unique_rows == c.D3_DUPLICATED_UNIQUE_ROWS).all()
    assert (PROV.duplicated_appearances == c.D3_DUPLICATED_APPEARANCES).all()
    assert (PROV.max_multiplicity == c.D3_MAX_MULTIPLICITY).all()
    assert np.allclose(PROV.sum_weights, c.D3_N_UNIQUE, atol=1e-6)
    assert (PROV.max_abs_rowweight_minus_one < 1e-12).all()
    # the identity that makes the P0 bookkeeping trap visible
    assert c.D3_UNIQUE_ROWS_MULT_1 + 2 * c.D3_DUPLICATED_UNIQUE_ROWS == c.D3_N_APPEARANCES
    assert "APPEARANCES" in FZ["d3_multiplicity"]["warning"]


def test_naive_duplicate_weighted_factor_is_reported_but_never_applied():
    assert (PROV.s != PROV.s_naive_pooled).all()
    assert PROV.s_minus_s_naive.abs().max() > 1e-6
    assert "never applied" in FZ["also_reported_never_applied"]["s_naive_pooled"] \
        or "transparency only" in FZ["also_reported_never_applied"]["s_naive_pooled"]
    # the APPLIED factor is s, not s_naive
    applied = dict(zip(INV.realization_key, INV.s))
    frozen = dict(zip(PROV.realization_key, PROV.s))
    for k, v in applied.items():
        assert v == frozen[k], k


# -------------------------------------------------------------------- apply
def test_apply_covered_every_realization_and_every_evaluation_block():
    assert INV.realization_key.nunique() == 43 == len(PROV)
    assert sorted(INV.evaluation.unique()) == sorted(ALL_EVALS)
    assert INV.groupby(["realization_key", "evaluation"]).ngroups == 430 == 43 * 10
    assert SM["n_cells"] == 430
    assert SM["s_recomputed_in_apply"] is False


def test_applied_factor_is_the_frozen_factor_unchanged():
    for _, r in INV.iterrows():
        assert abs(r.log_s - math.log(r.s)) < 1e-15
    j = INV.merge(PROV[["realization_key", "s"]], on="realization_key", suffixes=("", "_frozen"))
    assert (j.s == j.s_frozen).all()


def test_level_metrics_scale_by_exactly_s():
    g = INV[INV.invariance_class == "scales_by_s"]
    assert set(g.metric.unique()) == {"median_ratio", "mean_ratio", "weighted_mean_ratio"}
    assert len(g) == 430 * 3
    assert g.rel_diff_from_expected.max() < TOL, g.rel_diff_from_expected.max()
    assert (g.flag == "OK").all()
    # and they really did move -- the audit must not be vacuous
    assert (g.delta_smeared_minus_baseline.abs() > 1e-6).all()


def test_equity_and_mechanism_metrics_are_invariant():
    g = INV[INV.invariance_class == "invariant"]
    expect = {"COD", "COV", "PRD", "PRB", "VEI", "MKI", "beta_log",
              "dCor_e_y", "Cov_log_residual_log_price"}
    assert set(g.metric.unique()) == expect, set(g.metric.unique()) ^ expect
    assert g.rel_diff_from_expected.max() < TOL, (
        g.loc[g.rel_diff_from_expected.idxmax(), ["metric", "evaluation",
                                                  "rel_diff_from_expected"]].to_dict())
    assert (g.flag == "OK").all()
    for m, sub in g.groupby("metric"):
        assert sub.rel_diff_from_expected.max() < TOL, m


def test_no_cell_is_flagged():
    tested = INV[INV.flag != "NA_MOVES_BY_DESIGN"]
    assert len(tested) == SM["n_invariance_or_scale_checks"]
    assert SM["n_flagged"] == 0
    assert (tested.flag == "OK").all()
    assert SM["max_rel_diff_scales_by_s"] < TOL
    assert SM["max_rel_diff_invariant"] < TOL


def test_price_scale_accuracy_metrics_move_and_are_labelled_as_such():
    g = INV[INV.invariance_class == "moves"]
    assert set(g.metric.unique()) == {"R2_price", "MAE_price", "MAPE"}
    assert (g.flag == "NA_MOVES_BY_DESIGN").all()
    assert g.expected.isna().all()
    for m in ("R2_price", "MAE_price", "MAPE"):
        assert SM["moves_by_design"][m]["max_abs_delta"] > 0.0
    assert "NOT mathematically invariant" in SM["RMSE_log_note"]


def test_ed2_verdicts_are_unchanged_where_ed2_applies():
    ap = ED2[ED2.ed2_applicability == "ED2_APPLIED"]
    assert sorted(ap.evaluation.unique()) == sorted(FOLDS + ["heldout", "forward_2025"])
    assert len(ap) == 43 * 9 == 387
    assert ap.verdict_unchanged.astype(bool).all()
    assert ap.vei_rel_diff.max() < TOL
    for col in ("step5_gate", "step6_result", "step7_outcome", "ed2_verdict"):
        assert (ap[f"base_{col}"] == ap[f"smeared_{col}"]).all(), col
    assert SM["ed2_stability"]["all_verdicts_unchanged"] is True
    assert SM["ed2_stability"]["n_cells_ed2_applied"] == 387


def test_ed2_inference_is_refused_on_pooled_oof_consistently_with_task_3():
    na = ED2[ED2.ed2_applicability == "NOT_APPLICABLE_FOR_ED2_INFERENCE"]
    assert len(na) == 43
    assert set(na.evaluation.unique()) == {"pooled_oof"}
    assert na.verdict_unchanged.isna().all()
    assert na.ed2_not_applicable_reason.str.contains("no row-balanced analogue").all()
    assert na.ed2_not_applicable_reason.str.contains("No weighted variant was invented").all()
    # identical wording to the Task-3 artifact: one source of truth
    import p1_3_vei_ed2_inference as v
    assert set(na.ed2_not_applicable_reason.unique()) == {v.POOLED_OOF_REASON}
    vs = json.loads((c.TABLES / "vei_significance_summary.json").read_text())
    assert vs["pooled_oof"] == v.POOLED_OOF_REASON
    # but the VEI POINT ESTIMATE is still audited on pooled_oof
    pv = INV[(INV.evaluation == "pooled_oof") & (INV.metric == "VEI")]
    assert len(pv) == 43
    assert (pv.flag == "OK").all()


def test_delta_nl_subset_is_invariant_and_the_rule_is_declared():
    assert SM["delta_nl_subset"]["rule"].startswith("the two included reference cells")
    assert len(DNL) == 30 == 3 * 10
    assert DNL.realization_key.nunique() == 3
    assert sorted(DNL.evaluation.unique()) == sorted(ALL_EVALS)
    assert (DNL.flag == "OK").all()
    assert DNL.Delta_NL_rel_diff.max() < TOL, DNL.Delta_NL_rel_diff.max()
    assert DNL.Delta_NL_raw_rel_diff.max() < TOL
    assert "intercept" in SM["delta_nl_subset"]["analytic_reason"]
    # the subset must include the largest-s realization -- the strongest stress case
    assert PROV.loc[PROV.s.idxmax(), "realization_key"] in set(DNL.realization_key)


def test_fast_path_reproduces_the_frozen_p0_canonical_suite():
    assert len(FCR) > 0
    assert FCR.rel_diff.max() < 1e-9, FCR.loc[FCR.rel_diff.idxmax()].to_dict()
    assert SM["fastpath_reconciliation"]["max_rel_diff_vs_p0_metrics_from"] < 1e-9
    # every audited metric except Delta_NL was cross-checked
    audited = set(INV.metric.unique()) - {"Delta_NL"}
    assert audited <= set(FCR.metric.unique()) | {"COV"}, audited - set(FCR.metric.unique())


def test_role_is_sensitivity_only():
    assert "SENSITIVITY ONLY" in FZ["role"]
    assert "no headline replaced" in FZ["role"]
    assert SM["role"] == FZ["role"]
    assert "explicit assumption and a stated limitation" in FZ["forward_2025_assumption"]
    assert SM["forward_2025_assumption"] == FZ["forward_2025_assumption"]


def test_no_protected_path_written():
    import subprocess
    out = subprocess.run(
        ["git", "status", "--porcelain", "--", "paper/", "utils/", "soft_constrained_models/",
         "scripts/", "run_temporal_cv.py", "analysis/p0_major_revision_validation/"],
        cwd=str(c.REPO), capture_output=True, text=True).stdout.strip()
    assert out == "", out


if __name__ == "__main__":
    import traceback
    names = sorted(n for n in dir() if n.startswith("test_"))
    npass = nfail = 0
    for n in names:
        try:
            globals()[n](); print(f"  PASS  {n}"); npass += 1
        except Exception as ex:
            print(f"  FAIL  {n}: {type(ex).__name__}: {ex}"); traceback.print_exc(); nfail += 1
    print(f"\n{npass} passed, {nfail} failed")
    raise SystemExit(1 if nfail else 0)
