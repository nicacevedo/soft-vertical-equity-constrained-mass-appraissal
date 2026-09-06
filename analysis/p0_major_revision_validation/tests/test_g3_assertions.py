#!/usr/bin/env python3
"""Stage-2 / Gate-G3 assertions (additive to the Stage-1 and Gate-G2 suites)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
import p0_common as c            # noqa: E402
import p0_4_centered_spread as cs  # noqa: E402

T = c.TABLES
GRID = c.CONFIGS / "b_grid_frozen.json"
GRID_H = c.CONFIGS / "b_grid_frozen_hash.json"
CONV = c.CONFIGS / "posthoc_comparator_convention.yaml"


def _g():
    return json.loads(GRID.read_text())


def _conv():
    return yaml.safe_load(CONV.read_text())


def _path():
    return pd.read_csv(T / "centered_spread_path.csv")


# --------------------------------------------------------------------- roles
def test_C_primary_A_secondary_B_decomposition():
    v = _conv()["references"]
    assert v["C"]["posthoc_status"] == "PRIMARY"
    assert v["A"]["posthoc_status"] == "SECONDARY_PRACTICAL"
    assert v["B"]["posthoc_status"] == "NO_FULL_PATH"
    # Gate-G2 roles untouched
    g2 = yaml.safe_load((c.CONFIGS / "post_g1_reference_convention.yaml").read_text())
    assert g2["cells"]["C"]["role"] == "PRIMARY within-path penalty-isolating reference"
    assert g2["cells"]["A"]["role"] == "assessor-facing / workflow benchmark"
    assert g2["cells"]["B"]["role"] == "implementation-decomposition control only"


def test_no_b_full_path_exists():
    assert not list(T.glob("centered_spread_path__B__*.csv"))
    p = _path()
    assert set(p.reference.unique()) == {"A", "C"}


def test_future_matched_beta_primary_triple_exact():
    f = _conv()["future_matched_beta_convention"]
    assert f["primary_core_comparison"]["families"] == ["Direct", "Surrogate", "C-posthoc"]
    assert f["primary_core_comparison"]["matched_on"] == "DEVELOPMENT achieved beta_log"
    assert f["secondary_practical_comparison"]["families"] == ["A-posthoc"]
    assert f["cell_B"]["full_posthoc_path"] is False
    assert f["executed_in_stage_2"] is False
    for r in f["secondary_practical_comparison"]["restrictions"]:
        assert "must NOT" in r


# ------------------------------------------------------------------- no refit
def test_no_refit_performed():
    g = json.loads((T / "gate_g3_checks.json").read_text())
    assert g["no_refit_performed"] is True
    # the 27-fit table must be byte-identical to its Gate-G2 recorded hash
    base = json.loads((c.PROVENANCE / "output_artifact_hashes.json").read_text())
    k = "analysis/p0_major_revision_validation/tables/zero_reference_fits.csv"
    assert c.sha256_file(c.REPO / k) == base[k]["sha256"]


def test_no_new_fit_jobs_submitted_in_stage2():
    txt = (c.LOGS / "submitted_jobs.txt").read_text().lower()
    for bad in ("zero_reference_fits_2", "parity_track", "regen"):
        assert bad not in txt.split("centered_spread")[-1], bad


def test_input_prediction_hashes_verified():
    zf = pd.read_csv(T / "zero_reference_fits.csv")
    import hashlib
    for ref in ("A", "C"):
        for _, r in zf[zf.cell_id == ref].iterrows():
            d = pd.read_parquet(c.REPO / r.eval_predictions)
            h = hashlib.sha256(np.ascontiguousarray(
                d.y_pred_log.to_numpy(), dtype=np.float64).tobytes()).hexdigest()
            assert h == r.eval_pred_sha256, (ref, r.block_id)


# ----------------------------------------------------------------- b=1 exact
def test_b1_bitwise_and_fast_path():
    q = pd.read_csv(T / "centered_spread_b1_qc.csv")
    assert q.bitwise_identical_at_b1.all()
    assert (q.max_abs_delta == 0.0).all()
    non_pooled = q[q.evaluation != "pooled_oof"]
    assert non_pooled.shares_memory_fast_path.all()


def test_centered_map_b1_returns_same_object():
    f0 = np.array([1.0, 2.0, 3.0])
    out = cs.centered_map(f0, 12.4, 1.0)
    assert out is f0, "b == 1 must return the cached array unchanged, not recompute it"
    out2 = cs.centered_map(f0, 12.4, 1.05)
    assert out2 is not f0


def test_b1_path_metrics_equal_zero_control():
    """At b=1 the path must reproduce the Gate-G2 zero-control row for A and C."""
    zc = pd.read_csv(T / "zero_control_full.csv")
    p = _path()
    for ref in ("A", "C"):
        for ev in ("heldout", "forward_2025"):
            a = p[(p.reference == ref) & (p.evaluation == ev) & (p.b == 1.0)].iloc[0]
            b = zc[(zc.cell_id == ref) & (zc.evaluation == ev)].iloc[0]
            for k in ("R2_price", "MAE_price", "beta_log", "Delta_NL", "dCor_e_y", "COD"):
                assert abs(float(a[k]) - float(b[k])) < 1e-9, (ref, ev, k)


# --------------------------------------------------- development-only grid
def test_grid_construction_is_development_only():
    g = _g()
    assert g["development_only"] is True
    assert g["no_heldout_or_2025_outcome_read"] is True
    files = g["files_read_during_root_construction"]
    assert files, "the files read during root construction must be recorded"
    for f in files:
        assert "block=fold_" in f, f"root construction read a non-fold artifact: {f}"
        assert "development_pool" not in f and "production_2016_2024" not in f


def test_grid_frozen_before_oos_and_hash_matches():
    g, h = _g(), json.loads(GRID_H.read_text())
    assert h["file_sha256"] == c.sha256_file(GRID), "grid file changed after hashing"
    # grid must predate the out-of-time path shards
    oos = [T / f"centered_spread_path__{r}__{e}.csv"
           for r in ("A", "C") for e in ("heldout", "forward_2025")]
    gt = GRID_H.stat().st_mtime
    for p in oos:
        assert p.exists() and p.stat().st_mtime >= gt - 1, f"{p.name} predates the grid freeze"


def test_cvmean_root_formula():
    g = _g()
    for ref in ("C", "A"):
        r = g["development_roots"][ref]
        Rk = np.array([f["R_k"] for f in r["folds"]], dtype=float)
        assert len(Rk) == 7
        assert abs(r["mean_R_k"] - float(np.mean(Rk))) < 1e-15
        assert abs(r["b_zero_cvmean"] - 1.0 / float(np.mean(Rk))) < 1e-15
        # each fold R_k must equal Cov_Vk/Var_Vk
        for f in r["folds"]:
            assert abs(f["R_k"] - f["Cov_Vk_f0_y"] / f["Var_Vk_y"]) < 1e-15


def test_pooled_oof_root_formula():
    g = _g()
    for ref in ("C", "A"):
        r = g["development_roots"][ref]
        V, P, Q = r["pooled_V_var_y"], r["pooled_P_cov_ybar_c"], r["pooled_Q_cov_f0_c"]
        assert abs(r["b_zero_pooled_oof"] - (V - P) / (Q - P)) < 1e-12
        assert abs(P) > 0, "fold centers must correlate with c in the pooled sample"


def test_roots_computed_separately_per_reference():
    g = _g()
    assert (g["development_roots"]["C"]["b_zero_cvmean"]
            != g["development_roots"]["A"]["b_zero_cvmean"])
    assert (g["development_roots"]["C"]["b_zero_pooled_oof"]
            != g["development_roots"]["A"]["b_zero_pooled_oof"])
    assert _conv()["calibration"]["coordinate_D1_primary"]["computed_separately_per_reference"]
    assert _conv()["calibration"]["coordinate_D2_sensitivity"]["computed_separately_per_reference"]


def test_roots_zero_their_own_coordinate():
    rv = pd.read_csv(T / "centered_spread_root_verification.csv")
    assert len(rv) == 4
    assert rv.abs_from_zero.max() < 1e-12, rv.to_dict("records")


# ------------------------------------------------------ b_star_train excluded
def test_b_star_train_is_diagnostic_only():
    g, conv = _g(), _conv()
    assert "IN-SAMPLE" in g["b_star_train_status"]
    assert conv["calibration"]["b_star_train"]["used_for_empirical_endpoint"] is False
    assert conv["calibration"]["one_over_r2"]["used_for_endpoint"] is False
    chk = json.loads((T / "gate_g3_checks.json").read_text())
    assert chk["b_star_train_used_for_endpoint"] is False
    # b_max must NOT equal any b_star_train-derived value
    bs = pd.read_csv(T / "b_star_diagnostics.csv")
    for v in bs.b_star_train:
        assert abs(g["b_max"] - 1.25 * float(v)) > 1e-6
        assert abs(g["b_max"] - float(v)) > 1e-6


def test_overshoot_formula_is_25pct_of_b_minus_1():
    g = _g()
    roots = [g["development_roots"][r][k] for r in ("C", "A")
             for k in ("b_zero_cvmean", "b_zero_pooled_oof")]
    bref = max(roots)
    assert abs(g["b_reference_max_used"] - bref) < 1e-15
    assert abs(g["b_max"] - (1.0 + 1.25 * (bref - 1.0))) < 1e-15
    assert abs(g["b_max"] - 1.25 * bref) > 1e-3, "b_max must not be 1.25 * b_star"
    assert "NOT 1.25 * b_star" in g["b_max_rule"]


# ------------------------------------------------------------------ coverage
def test_path_spans_common_support_and_reaches_beta_zero():
    cov = _g()["coverage_verification"]
    assert cov["all_requirements_met"] is True
    assert all(cov["reaches_beta_zero"].values())
    assert all(cov["reaches_direct_upper_common_support"].values())
    for tgt in ("-0.06", "-0.03", "0.0"):
        for r in ("C", "A"):
            assert cov["ext_targets"][tgt][r]["within_grid"] is True
    lo, hi = cov["three_way_common_support_cvmean"]
    assert hi > lo


def test_grid_contains_all_special_anchors():
    g = _g()
    bs = np.array(g["b_values"], dtype=float)
    for k, v in g["special_anchors"].items():
        assert np.min(np.abs(bs - float(v))) < 1e-14, k


def test_grid_is_linear_base_plus_anchors():
    g = _g()
    assert "LINEAR" in g["construction_rule"]
    assert g["n_grid"] >= 121
    bs = np.array(g["b_values"], dtype=float)
    assert bs[0] == 1.0
    assert abs(bs[-1] - g["b_max"]) < 1e-15
    assert np.all(np.diff(bs) > 0), "grid must be sorted and deduplicated"


# ---------------------------------------------------------------- linearity
def test_beta_log_is_linear_in_b():
    lin = pd.read_csv(T / "centered_spread_linearity.csv")
    assert lin.max_abs_linear_residual.max() < 1e-9, lin.to_dict("records")
    assert lin.max_abs_dev_from_closed_form.dropna().max() < 1e-9
    assert lin.slope_abs_diff.max() < 1e-9
    assert lin.monotone_increasing.all()


def test_path_row_counts_complete():
    g, p = _g(), _path()
    for ref in ("A", "C"):
        for ev in cs.ALL_EVALS:
            n = len(p[(p.reference == ref) & (p.evaluation == ev)])
            assert n == g["n_grid"], (ref, ev, n)


def test_primary_map_label_on_every_row():
    p = _path()
    assert (p["map"] == "ybar_T-centered (theorem-matched primary)").all()


# ------------------------------------------------------- historical A QC
def test_historical_recalibration_reproduces():
    chk = json.loads((T / "gate_g3_checks.json").read_text())
    assert chk["historical_qc_worst_relative_metric_diff"] < 1e-6
    hq = pd.read_csv(T / "centered_spread_existing_qc.csv")
    assert len(hq) >= 102
    assert hq.ybar_absdiff.dropna().max() < 1e-9


def test_historical_endpoint_not_adopted_as_grid_endpoint():
    g = _g()
    assert abs(g["b_max"] - 1.1672656789694134) > 1e-3


# --------------------------------------------------------------- centering
def test_centering_sensitivity_immaterial():
    chk = json.loads((T / "gate_g3_checks.json").read_text())
    assert chk["centering_max_analytic_shift_log"] < 1e-4
    assert chk["centering_beta_log_absdiff_max"] < 1e-12
    s = pd.read_csv(T / "centered_spread_centering_sensitivity.csv")
    # scale-invariant metrics must be untouched
    for k in ("beta_log", "PRB", "PRD", "COD", "COV", "MKI", "VEI", "Delta_NL", "dCor_e_y",
              "Cov_log_residual_log_price"):
        col = f"{k}__absdiff"
        if col in s.columns:
            assert s[col].max() < 1e-9, (k, s[col].max())
    # level-sensitive metrics must move only negligibly in RELATIVE terms
    for k in ("MAE_price", "R2_price", "median_ratio", "mean_ratio", "weighted_mean_ratio"):
        a, b = f"{k}__absdiff", f"{k}__ybar"
        rel = (s[a] / s[b].abs().replace(0, np.nan)).max()
        assert rel < 1e-3, (k, rel)
    assert _conv()["centering_sensitivity"]["full_second_path_scheduled"] is False


def test_no_full_f0bar_path_generated():
    assert not list(T.glob("centered_spread_path_f0bar*"))
    p = _path()
    assert "center_f0bar" not in set(p["map"].unique())


# ------------------------------------------------- still-forbidden at every stage
# The original Gate-G3 versions of these three guards forbade the matched-beta and
# temporal artifacts outright, because at Gate G3 those stages had not been authorized.
# Stage 3 authorized them, so the guards are narrowed to the items that remain
# forbidden. Stage 3's own suite (test_stage3_assertions.py) carries the current,
# strictly stronger boundary for the newly authorized work.
def test_no_p1_output_from_this_stage():
    for bad in ("prb_inference.csv", "vei_significance.csv", "smearing_sensitivity.csv"):
        assert not (T / bad).exists(), bad


def test_no_full_regeneration_output_exists():
    for bad in ("combined_path_table_regen.csv", "combined_path_table_dsnap_full.csv"):
        assert not (T / bad).exists(), bad
    for d in (c.P0_OUTPUT_ROOT / "full_path_regen",
              c.P0_OUTPUT_ROOT / "dsnap_full_regen"):
        assert not d.exists(), str(d)


def test_no_regeneration_or_p1_jobs_submitted():
    log = c.LOGS / "submitted_jobs.txt"
    txt = log.read_text().lower() if log.exists() else ""
    for bad in ("full_path_regen", "regen_dsnap", "smearing", "inferential_extras"):
        assert bad not in txt, bad


def test_no_regeneration_trigger_fired():
    trg = pd.read_csv(T / "regeneration_triggers.csv")
    assert not trg.fired.any()


def test_no_manuscript_file_changed():
    prot = ("paper/", "utils/", "soft_constrained_models/", "run_temporal_cv.py", "scripts/",
            "output/paper_v6_preselection_994/", "output/paper_v12_")
    st = subprocess.run(["git", "status", "--porcelain"], cwd=str(c.REPO),
                        capture_output=True, text=True, check=True).stdout
    bad = [l for l in st.splitlines()
           if any(l[3:].strip().strip('"').startswith(p) for p in prot)]
    assert not bad, bad
    df = subprocess.run(["git", "diff", "--name-only", "HEAD"], cwd=str(c.REPO),
                        capture_output=True, text=True, check=True).stdout.splitlines()
    assert not [f for f in df if any(f.startswith(p) for p in prot)]


def test_stage1_and_g2_reports_untouched():
    base = json.loads((c.PROVENANCE / "stage1_frozen_hashes.json").read_text())
    allowed = {"analysis/p0_major_revision_validation/tables/regeneration_triggers.csv",
               "analysis/p0_major_revision_validation/provenance/output_artifact_hashes.json",
               "analysis/p0_major_revision_validation/provenance/POSTFLIGHT.json",
               "analysis/p0_major_revision_validation/provenance/slurm_graph.json",
               "analysis/p0_major_revision_validation/provenance/worktree_diff.patch"}
    bad = [k for k, v in base.items() if k not in allowed
           and (not (c.REPO / k).exists() or c.sha256_file(c.REPO / k) != v["sha256"])]
    assert not bad, bad


def test_approved_plan_still_readonly():
    p = c.P0_DIR / "APPROVED_EXECUTION_PLAN.md"
    assert (p.stat().st_mode & 0o222) == 0


# ------------------------------------------------------------------ profiles
def test_ratio_profiles_only_at_anchors_with_cis():
    pr = pd.read_csv(T / "centered_spread_ratio_profiles.csv")
    assert set(pr.profile.unique()) == {"iaao_proxy_group", "price_bin_30"}
    assert pr.b.nunique() <= 8, "profiles must be anchor-only, not the full path"
    g = pr[pr.profile == "iaao_proxy_group"]
    assert g.ci_low.notna().all() and g.ci_high.notna().all()
    assert (g.ci_level == 0.90).all()


def test_fold_overlap_finding_recorded():
    s = json.loads((T / "cv_fold_validation_overlap_summary.json").read_text())
    assert s["n_duplicated_rows"] > 0
    assert s["d1_cvmean_unaffected"] is True
    a = pd.read_csv(T / "cv_fold_validation_overlap_audit.csv")
    assert a.n_overlap.max() > 0
