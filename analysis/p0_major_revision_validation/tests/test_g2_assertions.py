#!/usr/bin/env python3
"""Stage-1.5 / Gate-G2 assertions (additive to test_p0_assertions.py)."""
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
import p0_common as c  # noqa: E402

T = c.TABLES
CONV_PATH = c.CONFIGS / "post_g1_reference_convention.yaml"

FWD = {
    "A": "Ordinary LightGBM (standard raw-label native)",
    "B": "Centered-label native L2 (initialization-aligned)",
    "C": "Custom-objective rho=0 origin",
}
ROLES = {
    "A": "assessor-facing / workflow benchmark",
    "B": "implementation-decomposition control only",
    "C": "PRIMARY within-path penalty-isolating reference",
}
BLOCKS = {"fold_1_train", "fold_2_train", "fold_3_train", "fold_4_train", "fold_5_train",
          "fold_6_train", "fold_7_train", "development_pool", "production_2016_2024"}
PAIRING = {**{f"fold_{k}_train": f"fold_{k}_val" for k in range(1, 8)},
           "development_pool": "heldout", "production_2016_2024": "forward_2025"}


def _conv():
    return yaml.safe_load(CONV_PATH.read_text())


def _zf():
    return pd.read_csv(T / "zero_reference_fits.csv")


# ------------------------------------------------------------- naming and roles
def test_forward_display_names_and_roles():
    conv = _conv()
    for cid in ("A", "B", "C"):
        assert conv["cells"][cid]["forward_display_name"] == FWD[cid]
        assert conv["cells"][cid]["role"] == ROLES[cid]


def test_cell_b_legacy_label_is_metadata_only():
    conv = _conv()
    assert conv["cells"]["B"]["legacy_stage1_label"] == "Parity-aligned native L2"
    # the legacy label must not be the forward display name anywhere in NEW outputs
    zf = _zf()
    assert not (zf.forward_display_name == "Parity-aligned native L2").any()
    zc = pd.read_csv(T / "zero_control_full.csv")
    assert not (zc.display_name == "Parity-aligned native L2").any()
    b = zc[zc.cell_id == "B"]
    assert (b.legacy_stage1_label == "Parity-aligned native L2").all()


def test_new_outputs_never_call_cell_b_parity_aligned():
    for f in list(c.REPORTS.glob("ZERO_CONTROL_REPORT.md")) + [CONV_PATH]:
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            if "Parity-aligned native L2" in line:
                assert ("legacy" in line.lower() or "misleading" in line.lower()
                        or "Stage-1" in line or "no longer" in line.lower()), line[:140]


# ------------------------------------------------------------------- 27 fits
def test_each_cell_has_exactly_nine_canonical_blocks():
    zf = _zf()
    assert len(zf) == 27
    for cid in ("A", "B", "C"):
        sub = zf[zf.cell_id == cid]
        assert len(sub) == 9, f"cell {cid} has {len(sub)} blocks"
        assert set(sub.block_id) == BLOCKS


def test_evaluation_pairing_is_correct():
    zf = _zf()
    for _, r in zf.iterrows():
        assert r.eval_name == PAIRING[r.block_id], (r.block_id, r.eval_name)


def test_block_sizes_match_canonical_protocol():
    zf = _zf()
    exp = {"fold_1_train": 46888, "fold_2_train": 100776, "fold_3_train": 151187,
           "fold_4_train": 200908, "fold_5_train": 252486, "fold_6_train": 298022,
           "fold_7_train": 310147, "development_pool": c.N_DEVELOPMENT,
           "production_2016_2024": c.N_PRODUCTION}
    for _, r in zf.iterrows():
        assert int(r.n_train) == exp[r.block_id], (r.block_id, r.n_train)
    dev = zf[zf.block_id == "development_pool"]
    assert (dev.n_eval == c.N_HELDOUT).all()
    prod = zf[zf.block_id == "production_2016_2024"]
    assert (prod.n_eval == c.N_2025).all()


def test_fold_index_hashes_match_archive():
    arch = {int(f["fold_id"]) + 1: f for f in c.load_archived_folds()["folds"]}
    zf = _zf()
    for _, r in zf[zf.block_kind == "cv_fold_train"].iterrows():
        a = arch[int(r.fold_1based)]
        assert r.train_index_hash == a["train_index_hash"]
        assert r.eval_index_hash == a["val_index_hash"]


def test_all_fits_used_historical_settings():
    zf = _zf()
    assert zf.execution_settings.str.contains("HISTORICAL").all()
    assert not zf.execution_settings.str.contains("deterministic=True").any()


def test_cell_c_implementation_recorded_and_single():
    zf = _zf()
    cc = zf[zf.cell_id == "C"]
    impls = set(cc.cell_c_implementation.dropna().unique())
    assert impls == {"LGBCovPenalty (Direct rho=0)"}, impls


def test_frozen_lgbm_config_hash_on_every_fit():
    zf = _zf()
    assert (zf.lgbm_params_sha256 == c.EXPECTED_LGBM_PARAMS_SHA256).all()


# ------------------------------------------------------- reproduction QC (A / C)
def test_A_and_C_reproduce_frozen_counterparts():
    qc = pd.read_csv(T / "zero_reference_reproduction_qc.csv")
    real = qc[qc.reproduction_tier.isin(["R1", "R2", "R3", "R4"])]
    assert len(real) > 0
    assert not (real.reproduction_tier == "R4").any(), \
        real[real.reproduction_tier == "R4"][["cell_id", "block_id"]].to_dict("records")
    assert set(real.cell_id) <= {"A", "C"}


def test_qc_tiers_in_range_and_classified():
    qc = pd.read_csv(T / "zero_reference_reproduction_qc.csv")
    real = qc[qc.reproduction_tier.isin(["R1", "R2", "R3", "R4"])]
    bad = real[(real.reproduction_tier != "R1")
               & (real.failure_class.isna()
                  | (real.failure_class.astype(str).str.strip() == ""))]
    assert bad.empty


def test_cell_B_is_not_required_to_equal_cell_C():
    """B<->C may differ; the QC table records it as a cross-check, never as a failure."""
    qc = pd.read_csv(T / "zero_reference_reproduction_qc.csv")
    bc = qc[qc.cell_id == "B<->C"]
    assert len(bc) > 0
    assert (bc.reproduction_tier == "AGGREGATE_CROSSCHECK").all()


def test_stage1_ladder_crosschecks_match():
    qc = pd.read_csv(T / "zero_reference_reproduction_qc.csv")
    x = qc[qc.reproduction_tier == "AGGREGATE_CROSSCHECK"]
    assert len(x) == 6
    assert x.matches_stage1_ladder.all()


# --------------------------------------------------------------- b-star / covariance
def test_cov_positive_and_bstar_finite_for_A_and_C():
    bs = pd.read_csv(T / "b_star_diagnostics.csv")
    ac = bs[bs.cell_id.isin(["A", "C"])]
    assert len(ac) == 18
    assert ac.cov_positive.all(), ac[~ac.cov_positive][["cell_id", "block_id"]].to_dict("records")
    assert ac.b_star_finite.all()
    assert np.isfinite(ac.b_star_train.to_numpy()).all()


def test_bstar_identity_holds():
    """beta_log_train == 1/b_star_train - 1 by construction."""
    bs = pd.read_csv(T / "b_star_diagnostics.csv")
    assert bs.identity_beta_log_vs_bstar_absdiff.max() < 1e-9


def test_one_over_r2_is_diagnostic_only():
    bs = pd.read_csv(T / "b_star_diagnostics.csv")
    assert "one_over_R2_log_theoretical_diagnostic_only" in bs.columns
    conv = _conv()
    assert "theoretical diagnostic only" in conv["b_star_definition"]["one_over_r2"]
    assert conv["b_star_definition"]["definition"].startswith("b_star_train = Var_T(y) / Cov_T(f0, y)")


# ------------------------------------------------------------ reference convention
def test_C_is_the_only_primary_penalty_isolating_reference():
    conv = _conv()
    prim = [k for k in ("A", "B", "C") if conv["cells"][k]["primary_penalty_reference"]]
    assert prim == ["C"], prim
    assert conv["comparison_kinds"]["penalty_isolating"]["reference"] == "C"
    assert conv["comparison_kinds"]["penalty_isolating"]["may_attribute_to_rho"] is True


def test_A_is_the_assessor_facing_benchmark():
    conv = _conv()
    assert conv["comparison_kinds"]["assessor_facing_benchmark"]["reference"] == "A"
    assert conv["comparison_kinds"]["assessor_facing_benchmark"]["may_attribute_to_rho"] is False
    assert conv["cells"]["A"]["visible_in_headline_tables"] is True


def test_B_cannot_be_selected_as_primary_penalty_reference():
    conv = _conv()
    b = conv["cells"]["B"]
    assert b["primary_penalty_reference"] is False
    assert b["may_be_primary_paper_baseline"] is False
    assert b["may_be_primary_centered_spread_comparator"] is False
    assert conv["comparison_kinds"]["implementation_decomposition"]["diagnostic_only"] is True


def test_comparator_convention_frozen_C_primary_A_secondary_no_B():
    conv = _conv()["future_centered_spread_convention"]
    assert conv["primary"]["cell"] == "C" and conv["primary"]["status"] == "PRIMARY"
    assert conv["secondary"]["cell"] == "A" and conv["secondary"]["status"] == "SECONDARY"
    assert conv["cell_B_full_path"]["scheduled_by_default"] is False
    assert conv["executed_in_stage_1_5"] is False


def test_bc_discrepancy_wording_is_not_innocuous():
    """The B<->C discrepancy must never be CHARACTERIZED as an innocuous numerical
    effect.  Quoting the superseded rev.3 RG-1 condition in order to replace it, or
    declaring the phrase forbidden, is legitimate."""
    conv = _conv()["bc_discrepancy"]
    assert "innocuous numerical effect" in conv["forbidden_wording"]
    assert "innocuous" not in conv["accepted_characterization"].lower()
    assert "innocuous" not in conv["conservative_characterization"].lower()
    allowed = ("withdrawn", "not be used", "forbidden", "replaced", "superseded",
               "rev.3", "rev. 3", "no longer")
    for f in [(c.PROVENANCE / "POST_G1_ADJUDICATION.md"),
              (c.REPORTS / "ZERO_CONTROL_REPORT.md")]:
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            if "innocuous numerical effect" in line:
                assert any(k in line.lower() for k in allowed), f"{f.name}: {line[:170]}"


def test_conservative_wording_available_when_rng_not_identified():
    conv = _conv()["bc_discrepancy"]
    if not conv["rng_mechanism_directly_identified_by_stage1_evidence"]:
        assert "feature-subsampling execution-path divergence" in \
               conv["conservative_characterization"]


# ------------------------------------------------------------- regeneration gate
def test_no_rg_trigger_fires_under_post_g1_adjudication():
    trg = pd.read_csv(T / "regeneration_triggers.csv")
    assert set(trg.trigger_id) == {"RG-1", "RG-2", "RG-3", "RG-4"}
    assert not trg.fired.any(), trg[trg.fired][["trigger_id", "decision"]].to_dict("records")


def test_rg_not_set_merely_because_AC_or_BC_differ():
    trg = pd.read_csv(T / "regeneration_triggers.csv")
    for _, r in trg.iterrows():
        if bool(r.fired):
            ev = str(r.evidence)
            assert "A<->B" not in ev and "A<->C" not in ev
    conv = _conv()
    assert conv["historical_positive_rho_path"]["regenerate_82_point_path"] is False


def test_every_rg_row_cites_evidence_artifacts():
    trg = pd.read_csv(T / "regeneration_triggers.csv")
    assert "evidence_artifacts" in trg.columns
    assert trg.evidence_artifacts.astype(str).str.len().min() > 10


# ----------------------------------------------------------------- not-yet-run
def test_no_centered_spread_path_executed():
    for bad in ("centered_spread_path.csv", "b_grid_frozen.json",
                "centered_spread_ratio_profiles.csv", "matched_beta_comparison.csv",
                "matched_beta_frozen.json"):
        assert not (T / bad).exists() and not (c.CONFIGS / bad).exists(), bad


def test_no_temporal_robustness_jobs_submitted():
    log = c.LOGS / "submitted_jobs.txt"
    txt = log.read_text() if log.exists() else ""
    for bad in ("dsnap", "dpurge", "dunseen", "full_path_regen", "centered_spread",
                "matched_beta", "inferential_extras"):
        assert bad not in txt.lower(), bad


def test_no_p1_extras_produced():
    for bad in ("prb_inference.csv", "vei_significance.csv", "smearing_sensitivity.csv"):
        assert not (T / bad).exists(), bad


def test_no_manuscript_or_frozen_output_modified():
    out = subprocess.run(["git", "status", "--porcelain"], cwd=str(c.REPO),
                         capture_output=True, text=True, check=True).stdout
    prot = ("paper/", "utils/", "soft_constrained_models/", "run_temporal_cv.py", "scripts/",
            "output/paper_v6_preselection_994/", "output/paper_v12_")
    bad = [l for l in out.splitlines()
           if any(l[3:].strip().strip('"').startswith(p) for p in prot)]
    assert not bad, bad
    # and nothing changed vs HEAD among tracked protected files
    diff = subprocess.run(["git", "diff", "--name-only", "HEAD"], cwd=str(c.REPO),
                          capture_output=True, text=True, check=True).stdout.splitlines()
    bad2 = [f for f in diff if any(f.startswith(p) for p in prot)]
    assert not bad2, bad2


def test_only_gitignore_modified_outside_p0():
    out = subprocess.run(["git", "diff", "--name-only", "2732e653~1", "HEAD"], cwd=str(c.REPO),
                         capture_output=True, text=True, check=False).stdout.splitlines()
    outside = [f for f in out if not f.startswith("analysis/p0_major_revision_validation/")]
    assert outside == [".gitignore"] or outside == [], outside


# ------------------------------------------------- frozen Stage-1 artifacts intact
AUTHORIZED_UPDATES = {
    # explicitly authorized by the Stage-1.5 instruction ("create/update")
    "analysis/p0_major_revision_validation/tables/regeneration_triggers.csv",
    # provenance files that are regenerated by design at each freeze
    "analysis/p0_major_revision_validation/provenance/output_artifact_hashes.json",
    "analysis/p0_major_revision_validation/provenance/POSTFLIGHT.json",
    "analysis/p0_major_revision_validation/provenance/slurm_graph.json",
    "analysis/p0_major_revision_validation/provenance/worktree_diff.patch",
}


def test_frozen_stage1_artifacts_not_rewritten():
    base = json.loads((c.PROVENANCE / "stage1_frozen_hashes.json").read_text())
    changed = []
    for rel, meta in base.items():
        if rel in AUTHORIZED_UPDATES:
            continue
        p = c.REPO / rel
        if not p.exists():
            changed.append((rel, "MISSING"))
        elif c.sha256_file(p) != meta["sha256"]:
            changed.append((rel, "HASH CHANGED"))
    assert not changed, changed


def test_approved_plan_still_frozen_readonly():
    p = c.P0_DIR / "APPROVED_EXECUTION_PLAN.md"
    assert (p.stat().st_mode & 0o222) == 0
    pre = json.loads((c.PROVENANCE / "PREFLIGHT.json").read_text())
    assert pre["approved_execution_plan_sha256"] == c.sha256_file(p)
    conv = _conv()
    assert conv["frozen_plan_sha256"] == c.sha256_file(p)


# ------------------------------------------------------------------ zero control
def test_zero_control_complete_for_ABC():
    zc = pd.read_csv(T / "zero_control_full.csv")
    need = {f"fold_{k}" for k in range(1, 8)} | {"CV_mean", "CV_SD", "heldout", "forward_2025"}
    for cid in ("A", "B", "C"):
        got = set(zc[zc.cell_id == cid].evaluation)
        assert need <= got, (cid, need - got)


def test_zero_control_has_required_columns():
    zc = pd.read_csv(T / "zero_control_full.csv")
    req = ["cell_id", "display_name", "role", "family", "rho", "evaluation", "n",
           "R2_price", "MAE_price", "MAPE", "RMSE_log", "median_ratio", "mean_ratio",
           "weighted_mean_ratio", "COD", "COV", "PRD", "PRB", "MKI", "VEI", "beta_log",
           "Cov_log_residual_log_price", "Delta_NL", "dCor_e_y",
           "source_artifact", "config_hash", "prediction_hash"]
    missing = [k for k in req if k not in zc.columns]
    assert not missing, missing


def test_zero_control_delta_nl_present_for_all_ABC_evaluations():
    zc = pd.read_csv(T / "zero_control_full.csv")
    abc = zc[zc.cell_id.isin(["A", "B", "C"])]
    assert abc.Delta_NL.notna().all()
    assert abc.dCor_e_y.notna().all()
