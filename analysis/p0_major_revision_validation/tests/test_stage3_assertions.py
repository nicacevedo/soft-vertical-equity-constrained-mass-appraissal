#!/usr/bin/env python3
"""Stage-3 assertions: matched-beta + temporal robustness leakage and scope guards.

One test per mandated §20 item, plus the ordering guards. Run via run_all_tests.py
(pytest is not installed in fairness_env).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
import p0_common as c                                            # noqa: E402
import p0_6_temporal_designs as td                               # noqa: E402

T, CFG, REP, FIGD = c.TABLES, c.CONFIGS, c.REPORTS, c.FIGURES
MB = CFG / "matched_beta_frozen.json"
MBH = CFG / "matched_beta_frozen_hash.json"


def _mb():
    return json.loads(MB.read_text())


def _dev():
    return pd.read_csv(T / "development_beta_coordinate_audit.csv")


# ---------------------------------------------------- coordinate roles
def test_d1_remains_primary():
    F = _mb()
    assert F["development_coordinate"] == "beta_D1"
    assert "FROZEN PRIMARY" in F["coordinate_role"]
    s = json.loads((T / "development_beta_coordinate_summary.json").read_text())
    assert "FROZEN PRIMARY" in s["D1"]["role"]


def test_d2_labelled_duplicate_weighted_historical_sensitivity():
    s = json.loads((T / "development_beta_coordinate_summary.json").read_text())
    r = s["D2"]["role"].lower()
    assert "duplicate-weighted" in r and "sensitivity" in r
    assert "only" in r, "D2 must be explicitly restricted to a sensitivity role"


def test_d3_appearance_weights_sum_to_exactly_one_per_unique_row():
    s = json.loads((T / "development_beta_coordinate_summary.json").read_text())
    assert s["D3"]["weights_sum_to_one_max_dev"] == 0.0
    assert s["overlap_structure"]["max_abs_unique_row_weight_minus_one"] == 0.0


def test_d3_uses_development_data_only():
    s = json.loads((T / "development_beta_coordinate_summary.json").read_text())
    assert s["D3"]["development_only"] is True
    d = _dev()
    srcs = set(d.source.unique())
    assert srcs <= {"frozen CV fold predictions",
                    "Stage-1.5 cached fold predictions + theorem map"}, \
        f"unexpected D3 source: {srcs}"
    for s_ in srcs:                     # no out-of-sample block may appear as a source
        assert "heldout" not in s_.lower() and "2025" not in s_.lower()


# ---------------------------------------------------- role boundaries
def test_A_never_determines_primary_support_or_targets():
    F = _mb()
    assert "A-posthoc" not in F["primary_families"]
    assert F["secondary_families"] == ["A-posthoc"]
    for k in ("common support", "CORE target selection", "whether a CORE target exists"):
        assert k in F["A_does_not_determine"]
    # the recorded support must equal the support of the PRIMARY families alone
    d = _dev()
    L = max(d[d.family == f].beta_D1.min() for f in F["primary_families"])
    U = min(d[d.family == f].beta_D1.max() for f in F["primary_families"])
    assert abs(L - F["three_way_common_support"][0]) < 1e-15
    assert abs(U - F["three_way_common_support"][1]) < 1e-15


def test_B_cannot_enter_matched_beta():
    F = _mb()
    assert "B" in F["excluded"]
    fams = {r["family"] for r in F["matched_configurations"]} | \
           {r["family"] for r in F["ext_matched"]}
    assert not any(f in ("B", "B-posthoc", "Centered-label native L2") for f in fams)
    for name in ("matched_beta_comparison.csv", "matched_beta_ext_targets.csv",
                 "matched_beta_ratio_profiles.csv"):
        df = pd.read_csv(T / name)
        assert not df.family.astype(str).str.fullmatch(r"B|B-posthoc").any(), name
    # the pairwise table is keyed on `pair`, not `family`
    pw = pd.read_csv(T / "matched_beta_pairwise_deltas.csv")
    assert not pw.pair.astype(str).str.contains(r"(^|_)B(_|$)", regex=True).any()


# ---------------------------------------------------- CORE construction
def test_six_core_targets_are_deterministic_direct_achieved_values():
    F = _mb()
    assert F["K_core"] == 6 and len(F["core_targets"]) == 6
    d = _dev()
    for t in F["core_targets"]:
        s = d[(d.family == "Direct")
              & np.isclose(d.rho.astype(float), t["direct_rho"], rtol=0, atol=1e-12)]
        assert len(s) == 1, f"Direct rho={t['direct_rho']} not a unique fitted config"
        # the TARGET is the ACHIEVED beta of that fitted config, never the probe q_j
        assert abs(float(s.iloc[0].beta_D1) - t["target"]) < 1e-15
    rhos = [t["direct_rho"] for t in F["core_targets"]]
    assert len(set(rhos)) == 6, "each CORE anchor must be a distinct fitted config"
    # probes are equally spaced across the recorded support, endpoints included
    L, U = F["three_way_common_support"]
    for j, t in enumerate(F["core_targets"]):
        assert abs(t["q_j"] - (L + j / 5 * (U - L))) < 1e-12


def test_tau_is_frozen_and_respected():
    F = _mb()
    assert F["tau"] == 0.002
    for r in F["matched_configurations"]:
        if r["attained"]:
            assert r["match_gap"] <= F["tau"] + 1e-15, r
    h = json.loads(MBH.read_text())
    assert h["file_sha256"] == c.sha256_file(MB), "frozen config changed after hashing"


# ---------------------------------------------------- ordering / interpolation
def test_matched_config_hash_precedes_all_oos_reads():
    h = json.loads(MBH.read_text())
    assert "BEFORE any held-out or 2025" in h["assertion"]
    frozen_at = pd.Timestamp(json.loads(MB.read_text())["frozen_at_utc"]).tz_localize(None)
    for name in ("matched_beta_comparison.csv", "matched_beta_pairwise_deltas.csv",
                 "matched_beta_ext_targets.csv", "matched_beta_crossings.csv",
                 "matched_beta_ratio_profiles.csv"):
        p = T / name
        assert p.exists(), name
        mt = pd.Timestamp(p.stat().st_mtime, unit="s")
        assert mt > frozen_at, f"{name} predates the config freeze"
    assert json.loads(MB.read_text())["no_oos_read_during_freeze"] is True


def test_no_oos_metric_interpolation_anywhere():
    allowed = {"exact_anchor", "nearest_fitted", "exact_posthoc_solve",
               "targeted_fit", "NOT_ATTAINED"}
    for name in ("matched_beta_comparison.csv", "matched_beta_ext_targets.csv",
                 "matched_beta_comparison_d3.csv", "matched_beta_ext_targets_d3.csv"):
        p = T / name
        if not p.exists():
            continue
        df = pd.read_csv(p)
        assert "match_mode" in df.columns, name
        bad = set(df.match_mode.dropna().unique()) - allowed
        assert not bad, f"{name}: disallowed match_mode {bad}"
        assert not df.match_mode.astype(str).str.contains(
            "interp", case=False).any(), f"{name} contains an interpolated row"
        for col in ("source",):
            if col in df.columns:
                assert not df[col].astype(str).str.contains(
                    "interp", case=False).any(), f"{name}.{col} claims interpolation"


def test_direct_core_rows_are_actual_frozen_fits():
    df = pd.read_csv(T / "matched_beta_comparison.csv")
    s = df[(df.family == "Direct") & df.attained]
    assert len(s) and (s.match_mode == "exact_anchor").all()
    assert s.rho.notna().all() and s.b.isna().all()
    assert s.source.str.contains("actual fit").all()
    assert np.allclose(s.match_gap.astype(float), 0.0, atol=0)


def test_surrogate_core_rows_are_actual_fitted_configs():
    F = _mb()
    df = pd.read_csv(T / "matched_beta_comparison.csv")
    s = df[(df.family == "Surrogate") & df.attained]
    assert len(s) and (s.match_mode == "nearest_fitted").all()
    assert s.rho.notna().all()
    assert s.source.str.contains("actual fit").all()
    d = _dev()
    for r in F["matched_configurations"]:
        if r["family"] == "Surrogate" and r["attained"]:
            q = d[(d.family == "Surrogate")
                  & np.isclose(d.rho.astype(float), r["rho"], rtol=0, atol=1e-12)]
            assert len(q) == 1, f"Surrogate rho={r['rho']} is not a fitted config"
    assert not any(r["requires_new_fit"] for r in F["matched_configurations"])


def test_posthoc_rows_are_actual_evaluated_transformations():
    df = pd.read_csv(T / "matched_beta_comparison.csv")
    s = df[df.family.isin(["C-posthoc", "A-posthoc"])]
    assert len(s) and (s.match_mode == "exact_posthoc_solve").all()
    assert s.b.notna().all() and s.rho.isna().all()
    assert s.source.str.contains("actual evaluated").all()


def test_origin_identity_holds_at_j0():
    idf = pd.read_csv(T / "matched_beta_origin_identity.csv")
    assert len(idf) == 8
    assert idf.agrees_within_1e9_relative.all()
    assert idf.max_rel_metric_difference.max() < 1e-9


def test_crossings_exclude_the_structural_origin():
    cr = pd.read_csv(T / "matched_beta_crossings.csv")
    if cr.crosses_between_j.notna().any():
        lo = cr.crosses_between_j.dropna().str.split("->").str[0].astype(int)
        assert (lo >= 1).all(), "j=0 deltas are structurally zero and carry no sign"


def test_ext_non_attainment_is_reported_not_dropped():
    e = pd.read_csv(T / "matched_beta_ext_targets.csv")
    for tgt in (-0.06, -0.03, 0.0):
        s = e[np.isclose(e.target, tgt) & (e.family == "Direct")]
        assert len(s) and (~s.attained.astype(bool)).all()
        assert (s.match_mode == "NOT_ATTAINED").all()
        assert s.achieved_dev_beta.notna().all(), "must carry the max achieved correction"
    z = e[np.isclose(e.target, 0.0) & (e.family == "Surrogate")]
    assert len(z) and (~z.attained.astype(bool)).all()


# ---------------------------------------------------- temporal designs
def test_dsnap_strict_date_boundaries():
    a = pd.read_csv(T / "dsnap_boundary_audit.csv")
    assert len(a) >= 8
    assert a.strict.astype(bool).all(), "every D-SNAP boundary must satisfy max(train) < min(eval)"


def test_dpurge_evaluation_sets_bitwise_identical_to_primary():
    df_tv, _, _, _, _ = c.load_canonical_splits(verbose=False)
    folds, _ = c.rebuild_folds(df_tv)
    prim = {int(r["fold_id"]) + 1: np.asarray(r["val_indices"], dtype=int) for r in folds}
    pg = json.loads((CFG / "split_protocol_dpurge.json").read_text())
    for r in pg["folds"]:
        v = td._load_idx(r["val_idx"])
        assert np.array_equal(np.sort(v), np.sort(prim[int(r["fold_1based"])])), \
            f"D-PURGE fold {r['fold_1based']} evaluation set differs from the primary design"


def test_dpurge_removes_every_paired_evaluation_pin_from_training():
    a = pd.read_csv(T / "dpurge_purge_audit.csv")
    assert len(a) >= 9
    assert (a.eval_pins_in_train_after_purge.astype(int) == 0).all()


def test_dpurge_labelled_oracle_everywhere():
    pg = json.loads((CFG / "split_protocol_dpurge.json").read_text())
    assert pg["design_type"] == "oracle_diagnostic"
    assert pg["NOT_PROSPECTIVELY_IMPLEMENTABLE"] is True
    for p in (T / "robustness_path_dpurge.csv", T / "robustness_vs_frozen_deltas.csv"):
        if not p.exists():
            continue
        df = pd.read_csv(p)
        s = df[df.design == "dpurge"] if "design" in df.columns else df
        assert len(s) and (s.design_type == "oracle_diagnostic").all(), p.name
    rp = REP / "TEMPORAL_ROBUSTNESS_REPORT.md"
    if rp.exists():
        head = "\n".join(rp.read_text().splitlines()[:40]).lower()
        assert "not prospectively implementable" in head or \
               "not be implemented prospectively" in head


def test_dunseen_performs_no_fits():
    spec = json.loads((CFG / "unseen_subset_definition.json").read_text())
    assert spec["fits"] == 0
    assert spec["denominator_changes"] is True
    assert spec["not_differenced_against_frozen_numbers"] is True
    p = T / "robustness_unseen_subset.csv"
    if p.exists():
        df = pd.read_csv(p)
        assert (df.design == "dunseen").all()
        assert "fit_seconds" not in df.columns, "D-UNSEEN must not record any fit"
        assert df.not_differenced_against_frozen.astype(bool).all()


def test_temporal_overlap_audit_exists_and_asserts_dpurge_equality():
    a = pd.read_csv(T / "temporal_validation_overlap_audit.csv")
    assert set(a.design.unique()) == {"primary", "dsnap", "dpurge"}
    s = json.loads((T / "temporal_validation_overlap_summary.json").read_text())
    assert s["dpurge_evaluation_sets_identical_to_primary"] is True
    assert int(s["primary"]["n_duplicated_appearances"]) == 20988
    assert int(s["dpurge"]["n_duplicated_appearances"]) == \
        int(s["primary"]["n_duplicated_appearances"])
    txt = json.dumps(s).lower()
    for bad in ("folds are independent", "folds are disjoint", "disjoint validation"):
        assert bad not in txt


def test_g5a_refinement_required_before_a_minor_alert_can_promote():
    p = T / "gate_g5a_outcome.json"
    assert p.exists()
    g = json.loads(p.read_text())
    if g["n_dsnap_triggers_fired"] == 0:
        assert "NO_TRIGGER" in g["g5a_outcome"]
    else:
        assert "refinement" in g["g5a_outcome"].lower()
        assert (CFG / "dsnap_refinement_grid.json").exists(), \
            "a fired screening alert must build the refinement grid before any promotion"
    assert "not launched in this run" in g["hard_stop"].lower()
    b = T / "gate_g5b_outcome.json"
    if b.exists():
        gb = json.loads(b.read_text())
        assert gb["full_dsnap_regeneration_launched_in_this_run"] is False


def test_no_full_dsnap_regeneration_submitted_in_this_run():
    out = subprocess.run(["sacct", "-u", "nacevedo", "--starttime", "2026-09-05",
                          "--format=JobID,JobName%40,State", "-n", "-P"],
                         capture_output=True, text=True).stdout
    for ln in out.splitlines():
        name = ln.split("|")[1] if "|" in ln else ""
        assert "regen" not in name.lower(), f"a regeneration job was submitted: {ln}"
    # and no regeneration output root exists
    for d in (c.P0_OUTPUT_ROOT / "full_path_regen", c.P0_OUTPUT_ROOT / "dsnap_full_regen"):
        assert not d.exists(), f"{d} must not exist in this run"


# ---------------------------------------------------- scope guards
def test_no_p1_outputs_exist():
    for name in ("prb_inference.csv", "vei_significance.csv", "smearing_sensitivity.csv"):
        assert not (T / name).exists(), f"{name} is P1 and out of scope for Stage 3"
    assert not (c.P0_DIR / "code" / "p0_7_inferential_extras.py").exists()
    assert not (REP / "MANUSCRIPT_IMPACT_MEMO.md").exists()


def test_no_manuscript_file_changed():
    out = subprocess.run(["git", "status", "--porcelain", "--",
                          "paper/", "utils/", "soft_constrained_models/",
                          "run_temporal_cv.py", "scripts/",
                          "output/paper_v6_preselection_994/",
                          "output/paper_v12_lower_rho_extension_994_v2/"],
                         cwd=str(c.REPO), capture_output=True, text=True).stdout.strip()
    assert out == "", f"protected paths modified:\n{out}"


def test_required_stage3_artifacts_present():
    for p in [CFG / "matched_beta_frozen.json", MBH,
              T / "development_beta_coordinate_audit.csv",
              T / "matched_beta_comparison.csv", T / "matched_beta_pairwise_deltas.csv",
              T / "matched_beta_crossings.csv", T / "matched_beta_ext_targets.csv",
              T / "matched_beta_ratio_profiles.csv",
              CFG / "split_protocol_dsnap.json", CFG / "split_protocol_dpurge.json",
              CFG / "unseen_subset_definition.json", CFG / "robustness_rho_grid.json",
              T / "temporal_validation_overlap_audit.csv",
              FIGD / "matched_beta_accuracy_equity.pdf",
              FIGD / "matched_beta_mechanism.pdf",
              FIGD / "matched_beta_ratio_profiles.pdf",
              REP / "MATCHED_BETA_REPORT.md",
              c.PROVENANCE / "POST_G3_ADJUDICATION.md"]:
        assert p.exists(), f"missing required Stage-3 artifact: {p}"


def test_d3_sensitivity_present_because_material():
    s = json.loads((T / "development_beta_coordinate_summary.json").read_text())
    if s.get("D3_MATERIAL"):
        assert (T / "matched_beta_d3_sensitivity.csv").exists()
        d = json.loads((T / "matched_beta_d3_sensitivity_summary.json").read_text())
        assert "UNCHANGED" in d["primary_coordinate"]


def test_post_g3_adjudication_does_not_call_d1_simply_unaffected():
    txt = (c.PROVENANCE / "POST_G3_ADJUDICATION.md").read_text()
    low = txt.lower()
    assert "unaffected" not in low or "must not be described simply as" in low, \
        "D1 may not be described simply as 'unaffected'"
    for phrase in ("not independent", "non-uniform"):
        assert phrase in low, f"the overlap interpretation must state '{phrase}'"
