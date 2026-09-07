#!/usr/bin/env python3
"""PRB-specific assertions for P1 Task 2.  Read-only: reruns no inference."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
import p1_common as c                                                   # noqa: E402

DF = pd.read_csv(c.TABLES / "prb_inference.csv")
SM = json.loads((c.TABLES / "prb_inference_summary.json").read_text())
EVALS = [f"fold_{k}" for k in range(1, 8)] + ["pooled_oof", "heldout", "forward_2025"]


def test_row_and_attainability_counts():
    assert len(DF) == 480, len(DF)
    assert int(DF.attained.astype(bool).sum()) == 440
    assert int((~DF.attained.astype(bool)).sum()) == 40


def test_all_ten_evaluation_blocks_present():
    assert sorted(DF.evaluation.unique()) == sorted(EVALS)
    for ev in EVALS:
        assert (DF.evaluation == ev).sum() == 48, ev


def test_unweighted_ols_reproduces_canonical_prb():
    unw = DF[DF.weighting == "none"].ols_minus_canonical.dropna().abs()
    assert unw.size > 0
    assert unw.max() < 1e-10, unw.max()
    assert abs(SM["reconciliation_unweighted_rows"]["max_abs"] - unw.max()) < 1e-18


def test_frozen_reconciliation_is_exact():
    fz = DF.frozen_minus_recomputed.dropna().abs()
    assert fz.size == 234, fz.size
    assert fz.max() < 1e-12, fz.max()


def test_d3_effect_reported_as_sensitivity_not_error():
    d3 = DF[DF.weighting == "D3_row_balanced"]
    assert set(d3.evaluation.unique()) == {"pooled_oof"}
    assert set(d3.evaluation_role.unique()) == {"sensitivity"}
    assert SM["d3_pooled_oof_reweighting_effect"]["max_abs"] > 1e-6


def test_no_fold_as_iid_aggregation_anywhere():
    """No CV_mean / CV_SD / SEM-over-folds construct may appear."""
    banned = ("cv_mean", "cv_sd", "cv_se", "sem", "sqrt(7)", "std_over_folds")
    for col in DF.columns:
        assert not any(b in col.lower() for b in banned), col
    assert "CV_mean" not in set(DF.evaluation.unique())
    assert "CV_SD" not in set(DF.evaluation.unique())
    # every fold row carries its own observation-level SE and n, not 7
    folds = DF[DF.evaluation.str.startswith("fold_") & DF.attained.astype(bool)]
    assert (folds.n > 1000).all()
    assert folds.PRB_se_classical.notna().all()
    assert (folds.evaluation_role == "descriptive").all()


def test_classification_states_and_no_threshold_crossing_mislabel():
    states = {"within_pm005", "overlaps_pm005",
              "outside_pm005_but_not_pm010", "outside_pm010"}
    got = set(DF.iaao_2013_class.dropna().unique())
    assert got <= states, got - states
    a = DF[DF.attained.astype(bool)]
    for _, r in a.iterrows():
        lo, hi, k = r.PRB_ci_lo, r.PRB_ci_hi, r.iaao_2013_class
        if k == "outside_pm010":
            assert lo > 0.10 or hi < -0.10
        elif k == "outside_pm005_but_not_pm010":
            assert (lo > 0.05 or hi < -0.05) and not (lo > 0.10 or hi < -0.10)
        elif k == "within_pm005":
            assert lo >= -0.05 and hi <= 0.05
        else:                                   # overlaps_pm005: straddles a boundary
            assert not (lo > 0.05 or hi < -0.05)
            assert not (lo >= -0.05 and hi <= 0.05)


def test_ci_is_consistent_with_classical_se():
    a = DF[DF.attained.astype(bool)]
    width = (a.PRB_ci_hi - a.PRB_ci_lo) / (2 * a.PRB_se_classical)
    assert width.between(1.95, 1.97).all(), (width.min(), width.max())
    assert np.allclose(a.PRB, (a.PRB_ci_lo + a.PRB_ci_hi) / 2, atol=1e-12)


def test_not_attained_preserved_with_null_metrics():
    na = DF[~DF.attained.astype(bool)]
    assert len(na) == 40
    assert na.PRB.isna().all() and na.PRB_se_classical.isna().all()
    assert na.iaao_2013_class.isna().all()
    assert na.rho.isna().all()
    assert (na.match_mode == "NOT_ATTAINED").all()
    assert na.source.str.contains("NOT_ATTAINED").all()


def test_attainability_matches_the_frozen_matched_beta_config_exactly():
    F = c.matched_beta_frozen()
    frozen = {(e["family"], float(e["target"])): bool(e["attained"]) for e in F["ext_matched"]}
    got = {(r.family, float(r.ext_target)): bool(r.attained)
           for _, r in DF[DF.display_kind == "matched_beta_ext"].iterrows()}
    assert got == frozen, (got, frozen)
    expected_na = {("Direct", -0.06), ("Direct", -0.03), ("Direct", 0.0), ("Surrogate", 0.0)}
    assert {k for k, v in got.items() if not v} == expected_na


def test_roles_preserved_for_deduplicated_realizations():
    """Cell C and matched-beta j=0 are the same realization; both role labels survive."""
    g = DF[(DF.realization_key == "fit:LGBCovPenalty:1fb838f7d6bfda88")
           & (DF.evaluation == "heldout")]
    assert len(g) == 2
    assert set(g.display_kind) == {"reference_cell", "matched_beta_core"}
    assert g.PRB.nunique() == 1, "the same realization must yield one PRB value"


def test_no_protected_path_written():
    import subprocess
    out = subprocess.run(
        ["git", "status", "--porcelain", "--", "paper/", "utils/", "soft_constrained_models/",
         "scripts/", "run_temporal_cv.py", "analysis/p0_major_revision_validation/",
         "output/paper_v6_preselection_994/", "output/paper_v12_lower_rho_extension_994_v2/"],
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
