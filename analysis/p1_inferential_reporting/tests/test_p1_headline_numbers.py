#!/usr/bin/env python3
"""Guard the reports against hand-transcription drift.

Two jobs:
  1. re-derive provenance/p1_headline_numbers.json from the artifacts and compare;
  2. assert the critical figures quoted in MANUSCRIPT_EVIDENCE_PACKAGE.md and
     MANUSCRIPT_IMPACT_MEMO.md appear verbatim, so no number in either document
     can drift away from the artifact it came from.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
import p1_common as c                                                   # noqa: E402
import p1_9_headline_numbers as hn                                      # noqa: E402

H = json.loads((c.PROVENANCE / "p1_headline_numbers.json").read_text())
PKG = (c.REPORTS / "MANUSCRIPT_EVIDENCE_PACKAGE.md").read_text()
MEMO = (c.REPORTS / "MANUSCRIPT_IMPACT_MEMO.md").read_text()


def _strip_provenance(d: dict) -> dict:
    return {k: v for k, v in d.items() if k != "provenance"}


def test_headline_numbers_are_reproducible_from_the_artifacts():
    got = _strip_provenance(hn.build())
    want = _strip_provenance(H)
    assert got == want, [k for k in want if got.get(k) != want.get(k)]


def test_cell_count_identities_hold():
    d, t3 = H["display_set"], H["task3_vei"]
    assert d["entries"] == 48 and d["attained"] == 44 and d["not_attained"] == 4
    assert d["fitted_realizations"] == 43
    assert t3["rows"] == 480 == d["entries"] * d["evaluation_blocks"]
    assert t3["evaluated_cells"] == 396 == d["attained"] * 9
    assert t3["unevaluated_cells"] == 84
    b = t3["unevaluated_breakdown"]
    assert b["pooled_oof_attained"] + b["not_attained"] == 84
    g = t3["step5_gate_counts"]
    assert g["step5_within_pm10_stop"] == 117
    assert g["step5_within_pm10_stop"] + g["step5_outside_pm10_escalate"] == 396
    s7 = t3["step7_counts"]
    assert s7["reject_null"] + s7["fail_to_reject_null"] == 279
    assert t3["step6_counts"].get("ci_overlap_stop", 0) == 0
    sf = t3["standards_facing"]
    assert sf["cells"] == 88 == 44 * 2
    assert sf["stop_at_step5_within_band"] + sf["escalated_past_step5"] == 88
    assert sf["reject_null"] + sf["fail_to_reject_at_step7"] == sf["escalated_past_step5"]
    t4 = H["task4_smearing"]
    assert t4["cells"] == 430 == 43 * 10
    assert t4["ed2_cells_applied"] == 387 == 43 * 9
    assert t4["ed2_cells_not_applicable"] == 43


def test_pooled_oof_is_not_applicable_everywhere_it_is_mentioned():
    assert H["task3_vei"]["pooled_oof_treatment"] == "NOT_APPLICABLE_FOR_ED2_INFERENCE"
    r = H["task3_vei"]["pooled_oof_reason"]
    assert "No weighted variant was invented" in r
    ed2 = pd.read_csv(c.TABLES / "smearing_apply_ed2_stability.csv")
    na = ed2[ed2.ed2_applicability == "NOT_APPLICABLE_FOR_ED2_INFERENCE"]
    assert set(na.ed2_not_applicable_reason.unique()) == {r}
    for doc, name in ((PKG, "package"), (MEMO, "memo")):
        assert "NOT_APPLICABLE_FOR_ED2_INFERENCE" in doc, name
        assert "No weighted variant" in doc or "None was invented" in doc, name


def test_no_flags_and_tolerances_are_as_reported():
    t4 = H["task4_smearing"]
    assert t4["flagged"] == 0
    assert t4["invariance_or_scale_checks"] == 5160
    assert t4["metric_checks"] == 6450
    assert t4["max_rel_diff_scales_by_s"] < 1e-9
    assert t4["max_rel_diff_invariant"] < 1e-9
    assert t4["worst_invariant_metric"] == "dCor_e_y"
    assert t4["ed2_verdicts_unchanged"] is True
    assert t4["fastpath_max_rel_diff"] == 0.0
    assert t4["delta_nl_subset_cells"] == 30
    for m in ("R2_price", "MAE_price", "MAPE"):
        assert t4["moves_by_design"][m]["max_abs_delta"] > 0


def test_dcor_answer_distinguishes_the_two_estimators():
    d = H["task1_dcor"]
    assert d["estimator_class"] == "BIASED / V-statistic (double-centered)"
    assert d["bias_corrected"] is False
    assert d["abs_diff_vs_V_statistic"] < 1e-12
    assert d["abs_diff_vs_U_statistic"] > 1e-3      # the two are distinguishable
    assert d["subsampling"] == "none - full evaluation sample"


def test_critical_figures_appear_verbatim_in_both_reports():
    """A targeted anti-drift check on the numbers a reader would act on."""
    t3, t4, d1, t2 = (H["task3_vei"], H["task4_smearing"],
                      H["task1_dcor"], H["task2_prb"])
    sf = t3["standards_facing"]
    pkg_must = [
        "153 passed, 0 failed, 0 skipped",
        "152 passed, 1 failed",
        "72 passed, 0 failed",
        "64 scientific",
        f"{t3['evaluated_cells']} evaluated",
        "117", "279", "228", "51",
        f"{sf['escalated_past_step5']}", f"{sf['reject_null']}",
        f"{t4['cells']} cells", "5,160", "6,450", "0 FLAGGED",
        f"{t4['ed2_cells_applied']}/{t4['ed2_cells_applied']}",
        "1.042931", "1.142692", "1.060201",
        "151,153", "130,165", "20,988",
        H["ed2_source"]["sha256"],
    ]
    for s in pkg_must:
        assert s in PKG, f"evidence package is missing {s!r}"
    memo_must = [
        "V-statistic", "bias_corrected = False", "7.52e-03", "1.33e-15",
        f"**11 of 63**", "−29.79", "+14.77",
        "1.0429", "1.1427", "1.0602", "+4.3%", "+14.3%",
        "3.94e-03",
        f"{t4['ed2_cells_applied']}/{t4['ed2_cells_applied']}",
        "5,160", "0 failures",
        "RMSE_log", "382,897",
        "May 2026", H["ed2_source"]["sha256"],
    ]
    for s in memo_must:
        assert s in MEMO, f"impact memo is missing {s!r}"


def test_reports_do_not_claim_a_compliance_determination():
    for doc, name in ((PKG, "package"), (MEMO, "memo")):
        low = doc.lower()
        assert "not a compliance determination" in low or "no compliance determination" in low, name
        assert "exposure draft" in low or "draft" in low, name
        assert "2013" in doc, name


def test_reports_state_that_no_manuscript_file_was_edited():
    assert "No manuscript file was edited" in PKG
    assert "No manuscript file has been edited" in MEMO
    import subprocess
    out = subprocess.run(["git", "status", "--porcelain", "--", "paper/"],
                         cwd=str(c.REPO), capture_output=True, text=True).stdout.strip()
    assert out == "", out
    diff = subprocess.run(["git", "diff", "--name-only", c.P0_TAG, "HEAD", "--", "paper/"],
                          cwd=str(c.REPO), capture_output=True, text=True).stdout.strip()
    assert diff == "", diff


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
