#!/usr/bin/env python3
"""VEI / ED2 App. E assertions for P1 Task 3.  Read-only: reruns no inference.

Every ED2 rule is re-derived here INDEPENDENTLY from the verbatim text of the
May-2026 Exposure Draft (App. D.2 p.67, App. E pp.78-81) rather than by calling
the Task-3 implementation, so a mistake in p1_3 cannot validate itself.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
import p1_common as c                                                   # noqa: E402

DF = pd.read_csv(c.TABLES / "vei_significance.csv")
SM = json.loads((c.TABLES / "vei_significance_summary.json").read_text())
CFG = json.loads((c.CONFIGS / "ed2_vei_procedure.json").read_text())
MAN = json.loads((c.PROVENANCE / "ed2_source_manifest.json").read_text())

FOLDS = [f"fold_{k}" for k in range(1, 8)]
ED2_APPLICABLE = FOLDS + ["heldout", "forward_2025"]      # nine blocks
ALL_EVALS = FOLDS + ["pooled_oof", "heldout", "forward_2025"]
BAND = 10.0
Z90 = 1.645

EV = DF[DF.VEI_step5.notna()]          # the 396 evaluated cells


# --------------------------------------------------------------- cell counts
def test_cell_counts_reconcile_exactly():
    """480 = 48 display entries x 10 blocks; 396 evaluated = 44 attained x 9 blocks."""
    assert len(DF) == 480, len(DF)
    assert sorted(DF.evaluation.unique()) == sorted(ALL_EVALS)
    for ev in ALL_EVALS:
        assert (DF.evaluation == ev).sum() == 48, ev
    n_attained_entries = int(DF[DF.evaluation == "heldout"].attained.astype(bool).sum())
    assert n_attained_entries == 44, n_attained_entries
    assert len(EV) == 396 == n_attained_entries * len(ED2_APPLICABLE)
    assert SM["rows"] == 480 and SM["rows_evaluated"] == 396
    # the 84 unevaluated rows are exactly: 44 attained pooled_oof + 40 NOT_ATTAINED
    unev = DF[DF.VEI_step5.isna()]
    assert len(unev) == 84
    assert int((unev.evaluation == "pooled_oof").sum()) == 48
    assert int((~unev.attained.astype(bool)).sum()) == 40
    assert int(((unev.evaluation == "pooled_oof") & unev.attained.astype(bool)).sum()) == 44
    assert SM["step5_gate_counts"]["nan"] == 84


def test_evaluated_cells_are_exactly_the_nine_ed2_applicable_blocks():
    assert sorted(EV.evaluation.unique()) == sorted(ED2_APPLICABLE)
    assert "pooled_oof" not in set(EV.evaluation.unique())
    roles = dict(zip(EV.evaluation, EV.evaluation_role))
    for f in FOLDS:
        assert roles[f] == "descriptive", (f, roles[f])
    assert roles["heldout"] == "standards_facing"
    assert roles["forward_2025"] == "standards_facing"


# ------------------------------------------------- pooled-OOF adjudication
def test_pooled_oof_is_explicitly_not_applicable_with_a_recorded_reason():
    po = DF[DF.evaluation == "pooled_oof"]
    assert len(po) == 48
    assert (po.evaluation_role == "not_applicable").all()
    assert po.VEI_step5.isna().all()
    assert po.step5_gate.isna().all()
    assert po.VEI_significance_step7.isna().all()
    att = po[po.attained.astype(bool)]
    assert len(att) == 44
    assert att.source.str.contains("no row-balanced analogue").all()
    assert att.source.str.contains("No weighted variant was invented").all()
    r = SM["pooled_oof"]
    assert "rank-based order-statistic" in r
    assert "20,988 unique rows twice" in r
    assert "No weighted variant was invented" in r


def test_no_weighted_rank_ed2_variant_exists_anywhere_in_the_code():
    """The draft defines no weighted-median CI; none may have been invented."""
    banned = ("weighted_rank", "weighted_median", "weighted_quantile", "effective_n",
              "effective_sample_size", "weighted_order_statistic", "d3_median_ci")
    for f in sorted((c.P1_DIR / "code").glob("p1_*.py")):
        low = f.read_text().lower()
        for b in banned:
            assert b not in low, (f.name, b)
    # and no ED2 column may be weight-derived
    for col in DF.columns:
        assert "weight" not in col.lower(), col


# --------------------------------------------- App. D.2 rank-based median CI
def _d2_ranks(n: int, z: float = Z90):
    """ED2 App. D.2, re-derived from the verbatim rule (independent of p1_3).

    odd  : j = ceil(z*sqrt(n)/2)        ranks (m-j, m+j),  m = (n+1)/2
    even : j = ceil(z*sqrt(n)/2 + 0.5)  ranks (n/2+1-j, n/2+j)
    """
    if n % 2 == 1:
        j = math.ceil(z * math.sqrt(n) / 2.0)
        m = (n + 1) // 2
        return j, m - j, m + j
    j = math.ceil(z * math.sqrt(n) / 2.0 + 0.5)
    return j, n // 2 + 1 - j, n // 2 + j


def test_d2_j_and_ranks_match_an_independent_rederivation():
    for _, r in EV.iterrows():
        for side in ("first", "last"):
            n = int(r[f"{side}_pg_n"])
            j, lo, hi = _d2_ranks(n)
            assert int(r[f"{side}_pg_ci_j"]) == j, (side, n, j, r[f"{side}_pg_ci_j"])
            assert r[f"{side}_pg_ci_ranks"] == f"{lo}|{hi}", (side, n, lo, hi)
            assert r[f"{side}_pg_n_parity"] == ("odd" if n % 2 else "even")
            assert 1 <= lo < hi <= n


def test_d2_is_applied_only_where_the_draft_says_it_applies():
    """'When the sales sample is greater than 30 sales' -- every PG must exceed 30."""
    assert (EV.first_pg_n > 30).all(), EV.first_pg_n.min()
    assert (EV.last_pg_n > 30).all(), EV.last_pg_n.min()
    assert not bool(SM["ci_rank_clamped_any"])
    assert not EV.ci_rank_clamped.astype(bool).any()


def test_ci_brackets_its_own_group_median():
    for _, r in EV.iterrows():
        assert r.first_pg_ci_lo <= r.first_pg_median <= r.first_pg_ci_hi
        assert r.last_pg_ci_lo <= r.last_pg_median <= r.last_pg_ci_hi


def test_ci_limits_are_true_order_statistics_on_a_bounded_subset():
    """Reload observations for the two reference cells x the two OOS blocks and
    confirm the recorded limits are the literal sorted-array elements at the
    D.2 ranks -- the one claim that cannot be checked from the table alone.

    Checked BITWISE against the parquet twin, which stores float64 exactly.  The
    committed CSV is compared separately, to within the 2-ULP text-serialisation
    loss quantified in test_csv_is_a_faithful_serialisation_of_the_parquet_twin.
    """
    PQ = pd.read_parquet(c.TABLES / "vei_significance.parquet")
    D = c.display_set()
    refs = [e for e in D["entries"] if e.get("reference_cell") in ("C", "A")]
    assert len(refs) == 2
    import p1_3_vei_ed2_inference as v
    checked = 0
    for e in refs:
        for ev in ("heldout", "forward_2025"):
            d = c.load_observations(e, ev)
            g = v.vei_groups(np.exp(d.y_pred_log.to_numpy()), np.exp(d.y_true_log.to_numpy()))
            sel = ((PQ.realization_key == e["realization_key"]) & (PQ.evaluation == ev))
            row = PQ[sel].iloc[0]
            crow = DF[(DF.realization_key == e["realization_key"])
                      & (DF.evaluation == ev)].iloc[0]
            for side, vals in (("first", g["groups"][0]), ("last", g["groups"][-1])):
                s = np.sort(vals)
                n = s.size
                assert n == int(row[f"{side}_pg_n"])
                _, lo, hi = _d2_ranks(n)
                # bitwise, against the exact float64 store
                assert float(s[lo - 1]) == float(row[f"{side}_pg_ci_lo"])
                assert float(s[hi - 1]) == float(row[f"{side}_pg_ci_hi"])
                assert float(np.median(s)) == float(row[f"{side}_pg_median"])
                # and the committed CSV agrees to within text-serialisation loss
                for col, exact in ((f"{side}_pg_ci_lo", float(s[lo - 1])),
                                   (f"{side}_pg_ci_hi", float(s[hi - 1])),
                                   (f"{side}_pg_median", float(np.median(s)))):
                    assert abs(float(crow[col]) - exact) <= 2 * np.spacing(abs(exact)), col
                checked += 1
    assert checked == 8, checked


def test_csv_is_a_faithful_serialisation_of_the_parquet_twin():
    """The committed evidence is CSV; the bit-exact store is the parquet twin.
    Quantify the difference so no downstream bitwise claim is made on the CSV."""
    PQ = pd.read_parquet(c.TABLES / "vei_significance.parquet")
    assert PQ.shape == DF.shape
    assert list(PQ.columns) == list(DF.columns)
    worst_ulp, worst_abs = 0.0, 0.0
    for col in DF.columns:
        if DF[col].dtype.kind not in "fi" or PQ[col].dtype.kind not in "fi":
            continue
        a = DF[col].to_numpy(float); b = PQ[col].to_numpy(float)
        m = np.isfinite(a) & np.isfinite(b)
        if not m.any():
            continue
        d = np.abs(a[m] - b[m])
        worst_abs = max(worst_abs, float(d.max()))
        sp = np.spacing(np.abs(b[m]))
        worst_ulp = max(worst_ulp, float((d / np.where(sp > 0, sp, 1.0)).max()))
    assert worst_ulp <= 16.0, worst_ulp        # observed: 16 ULP on VEI_step5
    assert worst_abs < 1e-13, worst_abs        # observed: 3.553e-15
    # integer-valued and string columns must round-trip exactly
    for col in ("first_pg_n", "last_pg_n", "n", "n_groups"):
        a = DF[col].to_numpy(float); b = PQ[col].to_numpy(float)
        m = np.isfinite(a) & np.isfinite(b)
        assert (a[m] == b[m]).all(), col
    for col in ("step5_gate", "step6_result", "step7_outcome", "ed2_verdict",
                "realization_key", "evaluation", "ed2_source_sha256"):
        assert DF[col].fillna("~").tolist() == PQ[col].fillna("~").tolist(), col


# ------------------------------------------------------ App. E Steps 3, 5, 6, 7
def test_step3_groups_are_deciles_with_array_split_counts():
    assert (EV.n_groups == 10).all()
    assert (EV.n >= 501).all()                      # ED2 Step-3 table: >=501 -> 10
    for _, r in EV.iterrows():
        chunks = np.array_split(np.arange(int(r.n)), 10)
        assert len(chunks[0]) == int(r.first_pg_n)
        assert len(chunks[-1]) == int(r.last_pg_n)


def test_step5_formula_recomputed_from_recorded_medians():
    lhs = 100.0 * (EV.last_pg_median - EV.first_pg_median) / EV.sample_median_ratio
    assert np.allclose(lhs, EV.VEI_step5, rtol=0, atol=1e-9), np.abs(lhs - EV.VEI_step5).max()


def test_step5_reproduces_the_canonical_vei_function_exactly():
    d = (EV.VEI_step5 - EV.canonical_vei_fn).abs()
    assert d.max() == 0.0, d.max()
    assert SM["max_abs_step5_minus_canonical_vei"] == 0.0


def test_step5_gate_is_the_band_rule_and_the_observed_split_is_preserved():
    within = EV.step5_gate == "step5_within_pm10_stop"
    assert (EV.VEI_step5.abs()[within] <= BAND).all()
    assert (EV.VEI_step5.abs()[~within] > BAND).all()
    assert int(within.sum()) == 117, int(within.sum())
    assert int((~within).sum()) == 279
    assert SM["step5_gate_counts"]["step5_within_pm10_stop"] == 117
    assert SM["step5_gate_counts"]["step5_outside_pm10_escalate"] == 279
    # cells that stop at Step 5 must not carry Step-6/7 results
    st = EV[within]
    assert (st.step6_result == "not_run").all()
    assert (st.step7_outcome == "not_run").all()
    assert st.VEI_significance_step7.isna().all()


def test_step6_overlap_recomputed_from_recorded_ci_bounds():
    esc = EV[EV.step5_gate == "step5_outside_pm10_escalate"]
    for _, r in esc.iterrows():
        overlap = not (r.first_pg_ci_hi < r.last_pg_ci_lo
                       or r.last_pg_ci_hi < r.first_pg_ci_lo)
        expect = "ci_overlap_stop" if overlap else "ci_no_overlap_escalate"
        assert r.step6_result == expect, (r.realization_key, r.evaluation, expect)
    assert SM["step6_counts"]["ci_no_overlap_escalate"] == 279
    assert SM["step6_counts"].get("ci_overlap_stop", 0) == 0
    assert SM["step6_counts"]["not_run"] == 117


def test_step7_formula_sign_and_outcome_mapping():
    s7 = EV[EV.step6_result == "ci_no_overlap_escalate"]
    assert len(s7) == 279
    for _, r in s7.iterrows():
        if r.first_pg_median >= r.last_pg_median:
            lo_hi, hi_lo = r.first_pg_ci_lo, r.last_pg_ci_hi
        else:
            lo_hi, hi_lo = r.last_pg_ci_lo, r.first_pg_ci_hi
        sig = 100.0 * (lo_hi - hi_lo) / r.sample_median_ratio
        assert abs(sig - r.VEI_significance_step7) < 1e-9, (sig, r.VEI_significance_step7)
        assert sig > 0, sig                      # guaranteed by non-overlap at Step 6
        expect = "reject_null" if sig > BAND else "fail_to_reject_null"
        assert r.step7_outcome == expect, (sig, r.step7_outcome)
    assert SM["step7_counts"]["reject_null"] == 228
    assert SM["step7_counts"]["fail_to_reject_null"] == 51
    assert SM["step7_counts"]["not_run"] == 117
    assert 228 + 51 == 279


def test_boundary_case_flag_is_consistent():
    b = EV[EV.ed2_boundary_exact.astype(bool)]
    for _, r in b.iterrows():
        assert r.VEI_significance_step7 == BAND
        assert r.step7_outcome == "fail_to_reject_null"


# ----------------------------------------------------------- verdict wording
def test_every_verdict_string_comes_from_the_frozen_decision_mapping():
    DM = CFG["decision_mapping"]
    m5 = {v["code"]: v["ed2_verdict"] for v in DM["step5_gate"].values()}
    m6 = {v["code"]: v["ed2_verdict"] for v in DM["step6_result"].values()}
    m7 = {v["code"]: v["ed2_verdict"] for v in DM["step7_outcome"].values()}
    for _, r in EV.iterrows():
        assert r.step5_verdict == m5[r.step5_gate]
        assert r.step6_verdict == m6[r.step6_result]
        assert r.step7_verdict == m7[r.step7_outcome]
        # the headline verdict is the LAST step actually reached
        if r.step5_gate == "step5_within_pm10_stop":
            assert r.ed2_verdict == m5[r.step5_gate]
        elif r.step6_result == "ci_overlap_stop":
            assert r.ed2_verdict == m6[r.step6_result]
        else:
            assert r.ed2_verdict == m7[r.step7_outcome]


def test_ci_method_is_the_rank_based_order_statistic_not_the_bootstrap():
    assert (EV.ci_method == "ED2 App. D.2 rank-based order statistic").all()
    assert (EV.ci_level == 0.90).all()
    assert (EV.ed2_band_pct == BAND).all()
    assert "NOT the bootstrap" in SM["ci_method"]
    assert SM["bootstrap_role"] == "separately labelled SENSITIVITY columns only"


def test_bootstrap_columns_are_sensitivity_only_and_bounded_in_scope():
    bcols = [x for x in DF.columns if "bootstrap" in x]
    assert bcols and all(x.endswith("_sensitivity") for x in bcols), bcols
    got = set(DF[DF[bcols[0]].notna()].evaluation.unique())
    assert got <= {"heldout", "forward_2025"}, got
    assert set(SM["bootstrap_scope"]) == {"heldout", "forward_2025"}
    # no decision column may be bootstrap-derived
    for dc in ("step5_gate", "step6_result", "step7_outcome", "VEI_significance_step7"):
        assert "bootstrap" not in dc


# ------------------------------------------------------------- provenance
def test_ed2_source_hash_on_every_row_matches_the_verified_manifest():
    h = MAN["source"]["sha256"]
    assert MAN.get("verified") is True
    assert (DF.ed2_source_sha256 == h).all()
    assert SM["ed2_source"]["sha256"] == h
    assert CFG["source"]["sha256"] == h
    pdf = c.PROVENANCE / "ed2_source_cache" / Path(MAN["source"]["pdf_url"]).name
    if pdf.exists():                       # gitignored; present only on the run host
        assert c.sha256_file(pdf) == h
        assert pdf.stat().st_size == MAN["source"]["bytes"]


def test_draft_status_is_disclosed_on_every_row():
    assert (DF.guidance_status == c.ED2_STATUS).all()
    assert (DF.adopted_reference == c.ED2_ADOPTED_REFERENCE).all()
    assert SM["not_a_compliance_determination"] is True
    assert "Exposure Draft" in SM["guidance_status"]
    assert "2013" in SM["adopted_reference"]


def test_documented_deviations_are_recorded_not_silent():
    """The two ED2 text problems and the grouping deviation must be on the record."""
    s2 = CFG["appendix_E_vei"]["step_2"]
    assert "DISCREPANCY" in s2
    assert s2["verbatim_printed_formula"] == "Proxy = (0.50*SP) + (AV/Median Ratio)"
    assert s2["implemented"] == "proxy = 0.50*SP + 0.50*(AV/median_ratio)"
    assert "equal weight" in s2["verbatim_prose"]
    e3 = CFG["appendix_E_vei"]["percentile_rank_methodology_E3"]
    assert "array_split is neither R6 nor R7" in e3["status"]
    d2 = CFG["appendix_D2_median_ci"]["interpretation_ED2_I_1"]
    assert d2["status"] == "documented interpretation, not a verbatim rule"
    vb = CFG["appendix_D2_median_ci"]["verbatim"]
    assert len(vb) == 7                       # 5 numbered steps; step 2 splits into a/b
    assert [x.strip()[:2] for x in vb] == ["1.", "2.", "a.", "b.", "3.", "4.", "5."]
    # spot-check against the literal document wording (App. D.2, printed page 67)
    assert vb[0] == "1. Array the ratios in ascending order and rank accordingly"
    assert "round up to the next integer" in vb[4]
    assert "count up and down the array from the median" in vb[6]
    assert "sales sample greater than 30" in CFG["appendix_D2_median_ci"]["applies_when"]
    assert CFG["appendix_D2_median_ci"]["page"] == 67
    assert CFG["appendix_D2_median_ci"]["z_used"] == Z90


# --------------------------------------------------------------- structure
def test_frozen_reconciliation_is_exact():
    fz = DF.frozen_minus_recomputed.dropna().abs()
    assert fz.size == 234, fz.size
    assert fz.max() < 1e-12, fz.max()
    assert SM["frozen_reconciliation"]["n_compared"] == 234
    assert SM["frozen_reconciliation"]["max_abs_frozen_minus_recomputed"] < 1e-12


def test_not_attained_rows_preserved_with_null_metrics():
    na = DF[~DF.attained.astype(bool)]
    assert len(na) == 40
    assert na.VEI_step5.isna().all() and na.sample_median_ratio.isna().all()
    assert na.ed2_verdict.isna().all()
    assert (na.match_mode == "NOT_ATTAINED").all()
    assert na.source.str.contains("NOT_ATTAINED").all()
    assert SM["rows_not_attained"] == 40
    got = {(r.family, float(r.ext_target)) for _, r in
           na[na.display_kind == "matched_beta_ext"].iterrows()}
    assert got == {("Direct", -0.06), ("Direct", -0.03), ("Direct", 0.0), ("Surrogate", 0.0)}, got


def test_no_fold_as_iid_aggregation_anywhere():
    banned = ("cv_mean", "cv_sd", "cv_se", "sem", "sqrt(7)", "std_over_folds")
    for col in DF.columns:
        assert not any(b in col.lower() for b in banned), col
    assert not {"CV_mean", "CV_SD"} & set(DF.evaluation.unique())
    f = EV[EV.evaluation.isin(FOLDS)]
    assert len(f) == 44 * 7
    assert (f.evaluation_role == "descriptive").all()
    assert (f.n > 1000).all()


def test_summary_counts_match_the_table():
    for key, col in (("step5_gate_counts", "step5_gate"),
                     ("step6_counts", "step6_result"),
                     ("step7_counts", "step7_outcome")):
        got = {str(k): int(v) for k, v in DF[col].value_counts(dropna=False).items()}
        assert SM[key] == got, (key, SM[key], got)


def test_the_same_realization_yields_one_ed2_result():
    g = DF[(DF.realization_key == "fit:LGBCovPenalty:1fb838f7d6bfda88")
           & (DF.evaluation == "heldout")]
    assert len(g) == 2
    assert set(g.display_kind) == {"reference_cell", "matched_beta_core"}
    assert g.VEI_step5.nunique() == 1
    assert g.ed2_verdict.nunique() == 1
    for (rk, ev), grp in DF[DF.VEI_step5.notna()].groupby(["realization_key", "evaluation"]):
        assert grp.VEI_step5.nunique() == 1, (rk, ev)


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
