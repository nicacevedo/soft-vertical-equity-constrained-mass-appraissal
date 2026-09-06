#!/usr/bin/env python3
"""Stage-1 assertions for the P0 major-revision validation.

Run:  /home/nacevedo/.conda/envs/fairness_env/bin/python -m pytest \
        analysis/p0_major_revision_validation/tests/test_p0_assertions.py -q
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
import p0_common as c  # noqa: E402

TABLES = c.TABLES
CELL_NAMES = {c.CELL_A_NAME, c.CELL_B_NAME, c.CELL_C_NAME}


# ------------------------------------------------------------------ canonical data
def test_canonical_counts_constants():
    assert (c.N_DEVELOPMENT, c.N_HELDOUT, c.N_2025) == (344607, 38290, 26641)
    assert c.N_PRODUCTION == 382897


def test_objective_audit_reproduces_canonical_block_sizes():
    df = pd.read_csv(TABLES / "objective_scaling_audit.csv")
    assert len(df) == 9, "expected the 9 canonical fitting blocks"
    assert int(df.loc[df.block_id == "development_pool", "n_T"].iloc[0]) == c.N_DEVELOPMENT
    assert int(df.loc[df.block_id == "production_2016_2024", "n_T"].iloc[0]) == c.N_PRODUCTION
    folds = df[df.block_kind == "cv_fold_train"].sort_values("fold_id")["n_T"].tolist()
    assert folds == [46888, 100776, 151187, 200908, 252486, 298022, 310147]


def test_fold_index_hashes_verified_live():
    """rebuild_folds raises ProtocolViolation on any archived-hash mismatch."""
    df_tv, _te, _as, _p, _c = c.load_canonical_splits(verbose=False)
    folds, archive = c.rebuild_folds(df_tv, verify=True)
    assert len(folds) == 7
    for rec, arch in zip(folds, archive["folds"]):
        assert rec["train_index_hash"] == arch["train_index_hash"]
        assert rec["val_index_hash"] == arch["val_index_hash"]


def test_frozen_lgbm_params_hash():
    cfg = c.frozen_lgbm_config()          # raises on mismatch
    assert cfg["lgbm_params_sha256"] == c.EXPECTED_LGBM_PARAMS_SHA256
    assert cfg["lgbm_params"]["n_estimators"] == 994
    assert cfg["lgbm_params"]["num_leaves"] == 573


def test_frozen_rho_grid_is_83_points():
    g = c.frozen_rho_grid()
    assert len(g) == 83 and g[0] == 0.0
    for a in c.DISPLAY_ANCHORS:
        assert any(abs(x - a) < 1e-12 for x in g)


# ------------------------------------------------------- canonical loader discipline
def test_p0_code_never_reimplements_the_parquet_filter():
    """No P0 module may apply the eligibility filter pandas-side; the pyarrow
    row-group pushdown in run_temporal_cv._load_and_split_data is part of the frozen
    experiment definition."""
    offenders = []
    for f in CODE.glob("p0_*.py"):
        src = f.read_text()
        if "ind_pin_is_multicard" in src and "_load_and_split_data" not in src:
            # p0_6_temporal reads meta_pin with the SAME pushdown filters -- allowed,
            # because it uses pyarrow filters, not a pandas mask.
            if ".astype(\"bool\").fillna(" in src or "~df[" in src:
                offenders.append(f.name)
    assert not offenders, f"pandas-side eligibility filtering found in {offenders}"


def test_every_p0_module_imports_canonical_loader():
    for name in ("p0_1_objective_audit.py", "p0_6_temporal.py", "p0_2_parity_tracks.py"):
        src = (CODE / name).read_text()
        assert "load_canonical_splits" in src, f"{name} must use the canonical loader"


# --------------------------------------------------------------------- cell naming
def test_fixed_cell_names_verbatim():
    assert c.CELL_A_NAME == "Ordinary LightGBM (standard raw-label native)"
    assert c.CELL_B_NAME == "Parity-aligned native L2"
    assert c.CELL_C_NAME == "Custom rho=0 origin"


def test_cell_b_never_printed_as_ordinary_lightgbm():
    for f in list(CODE.glob("*.py")) + list((c.P0_DIR / "reports").glob("*.md")):
        src = f.read_text()
        for line in src.splitlines():
            if "Parity-aligned native L2" in line and "Ordinary LightGBM" in line:
                # allowed only when both names appear as distinct labelled entities
                assert "never" in line.lower() or "|" in line or "Cell A" in line, (
                    f"ambiguous co-mention in {f.name}: {line[:120]}")


def test_parity_ladder_uses_only_fixed_cell_names():
    for trk in ("historical", "pinned"):
        p = TABLES / f"parity_ladder_{trk}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        rows = df[~df.cell_pair.astype(str).str.startswith("METRICS")]
        for col in ("cell_a_name", "cell_b_name"):
            vals = {v for v in rows[col].dropna().unique()}
            for v in vals:
                assert v in CELL_NAMES or v.startswith(c.CELL_C_NAME), v


# ------------------------------------------------------------------- tier separation
def test_parity_and_reproduction_tiers_live_in_separate_fields():
    for trk in ("historical", "pinned"):
        p = TABLES / f"parity_ladder_{trk}.csv"
        if p.exists():
            cols = set(pd.read_csv(p, nrows=1).columns)
            assert "parity_tier" in cols
            assert "reproduction_tier" not in cols, "R-tiers must not appear in a parity table"
    rp = TABLES / "frozen_artifact_reproduction.csv"
    if rp.exists():
        cols = set(pd.read_csv(rp, nrows=1).columns)
        assert "reproduction_tier" in cols
        assert "parity_tier" not in cols, "T-tiers must not appear in a reproduction table"


def test_tier_values_are_in_range():
    for trk in ("historical", "pinned"):
        p = TABLES / f"parity_ladder_{trk}.csv"
        if p.exists():
            df = pd.read_csv(p)
            vals = set(df["parity_tier"].dropna().unique())
            assert vals <= {"T1", "T2", "T3", "T4"}, vals
    rp = TABLES / "frozen_artifact_reproduction.csv"
    if rp.exists():
        vals = set(pd.read_csv(rp)["reproduction_tier"].dropna().unique())
        assert vals <= {"R1", "R2", "R3", "R4"}, vals


def test_track_h_and_track_p_are_separate_rows_with_explicit_track_field():
    for trk in ("historical", "pinned"):
        p = TABLES / f"parity_ladder_{trk}.csv"
        if p.exists():
            df = pd.read_csv(p)
            assert "track" in df.columns
            assert set(df["track"].unique()) == {trk}


def test_non_r1_reproduction_rows_carry_a_failure_class():
    rp = TABLES / "frozen_artifact_reproduction.csv"
    if not rp.exists():
        pytest.skip("reproduction table not yet produced")
    df = pd.read_csv(rp)
    bad = df[(df.reproduction_tier != "R1") &
             (df.failure_class.isna() | (df.failure_class.astype(str).str.strip() == ""))]
    assert bad.empty, f"non-R1 rows without a failure class: {bad['label'].tolist()}"


def test_failure_classes_come_from_the_approved_taxonomy():
    rp = TABLES / "frozen_artifact_reproduction.csv"
    if not rp.exists():
        pytest.skip("reproduction table not yet produced")
    allowed = {"none", "F-SRC", "F-DIRTY", "F-ENV", "F-NUM", "F-IMP",
               "PENDING_CLASSIFICATION"}
    vals = set(pd.read_csv(rp)["failure_class"].dropna().unique())
    assert vals <= allowed, vals


# ---------------------------------------------------------------- regeneration gate
def test_no_rg_trigger_can_be_set_by_A_B_or_A_C():
    p = TABLES / "regeneration_triggers.csv"
    if not p.exists():
        pytest.skip("trigger table not yet initialized")
    df = pd.read_csv(p)
    for _, r in df.iterrows():
        ev = str(r.get("evidence", ""))
        if bool(r.get("fired", False)):
            assert "A<->B" not in ev and "A<->C" not in ev, (
                f"trigger {r['trigger_id']} fired on an A<->B / A<->C comparison")


def test_stage1_submitted_no_regeneration_job():
    log = c.LOGS / "submitted_jobs.txt"
    if not log.exists():
        pytest.skip("no job log")
    txt = log.read_text()
    for forbidden in ("full_path_regen", "regen_dsnap", "temporal_dsnap", "temporal_dpurge"):
        assert forbidden not in txt, f"Stage 1 submitted a forbidden job: {forbidden}"


def test_source_equivalence_precedes_reproduction_interpretation():
    """The dirty-state limitation and source-equivalence verdict must exist and be
    older than (or equal in age to) the reproduction table."""
    se = TABLES / "source_equivalence_verdict.json"
    lim = c.PROVENANCE / "DIRTY_STATE_LIMITATION.md"
    assert se.exists() and lim.exists(), "source-equivalence audit must run first"
    rp = TABLES / "frozen_artifact_reproduction.csv"
    if rp.exists():
        assert se.stat().st_mtime <= rp.stat().st_mtime + 1


# ------------------------------------------------------------------------ isolation
def test_frozen_analysis_directories_unmodified():
    out = subprocess.run(["git", "status", "--porcelain"], cwd=str(c.REPO),
                         capture_output=True, text=True, check=True).stdout
    protected = ("paper/", "utils/", "soft_constrained_models/", "run_temporal_cv.py",
                 "scripts/", "output/paper_v6_preselection_994/",
                 "output/paper_v12_")
    bad = []
    for line in out.splitlines():
        path = line[3:].strip().strip('"')
        if any(path.startswith(p) for p in protected):
            bad.append(line)
    assert not bad, f"protected paths modified: {bad}"


def test_all_stage1_writes_are_inside_approved_locations():
    out = subprocess.run(["git", "status", "--porcelain"], cwd=str(c.REPO),
                         capture_output=True, text=True, check=True).stdout
    allowed = ("analysis/p0_major_revision_validation/", "output/p0_major_revision_validation/")
    bad = [l for l in out.splitlines()
           if l[3:].strip().strip('"') and not any(l[3:].strip().strip('"').startswith(a) for a in allowed)]
    assert not bad, f"writes outside the approved P0 locations: {bad}"


def test_write_guard_rejects_paths_outside_p0():
    with pytest.raises(c.ProtocolViolation):
        c.write_json(c.REPO / "paper" / "should_never_exist.json", {"x": 1})


def test_approved_plan_is_frozen_readonly():
    p = c.P0_DIR / "APPROVED_EXECUTION_PLAN.md"
    assert p.exists()
    assert (p.stat().st_mode & 0o222) == 0, "APPROVED_EXECUTION_PLAN.md must be read-only"
    pre = json.loads((c.PROVENANCE / "PREFLIGHT.json").read_text())
    assert pre["approved_execution_plan_sha256"] == c.sha256_file(p)


# ----------------------------------------------------------------- stylized labelling
def test_leaf_shrinkage_rows_are_labelled_stylized():
    df = pd.read_csv(TABLES / "effective_leaf_shrinkage.csv")
    assert set(df["diagnostic_type"].unique()) == {"stylized_nominal_leaf"}


def test_min_sum_hessian_reported_non_binding():
    df = pd.read_csv(TABLES / "effective_leaf_shrinkage.csv")
    assert not df["min_sum_hessian_binding_direct"].any()
    assert not df["min_sum_hessian_binding_surrogate"].any()


def test_e4_verdict_is_one_of_three_values():
    v = json.loads((TABLES / "e4_verdict.json").read_text())
    assert v["e4_verdict"] in {"ACCEPT", "REJECT", "INDETERMINATE"}


# ------------------------------------------------------------------ temporal exposure
def test_all_eight_chronological_boundaries_cross_same_date():
    df = pd.read_csv(TABLES / "boundary_exposure_audit.csv")
    cv = df[df.boundary_kind == "cv_fold_train_val"]
    assert len(cv) == 7 and cv["same_date_crossing"].all()
    dh = df[df.boundary_id == "development_heldout"].iloc[0]
    assert bool(dh["same_date_crossing"])
    fwd = df[df.boundary_id == "production_2025"].iloc[0]
    assert not bool(fwd["same_date_crossing"]), "the 2025 boundary is year-based and clean"


def test_boundary_exposure_is_small():
    df = pd.read_csv(TABLES / "boundary_exposure_audit.csv")
    assert df["share_of_val_block_on_boundary_date"].max() < 0.01


def test_repeat_pin_exposure_monotone_and_material():
    df = pd.read_csv(TABLES / "repeat_pin_exposure.csv")
    cv = df[df.boundary_kind == "cv_fold_train_val"].copy()
    cv["f"] = cv.boundary_id.str.replace("fold_", "").astype(int)
    cv = cv.sort_values("f")
    s = cv["share_val_pins_also_in_train"].to_numpy()
    assert np.all(np.diff(s) > 0), "repeat-PIN exposure should rise with expanding folds"
    assert s[0] > 0.09 and s[-1] > 0.26
    oot = df[df.boundary_kind == "out_of_time"].set_index("boundary_id")
    assert oot.loc["development_heldout", "share_val_pins_also_in_train"] > 0.29
    assert oot.loc["production_2025", "share_val_pins_also_in_train"] > 0.33
