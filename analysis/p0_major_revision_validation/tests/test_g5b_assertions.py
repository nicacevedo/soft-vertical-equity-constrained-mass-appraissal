#!/usr/bin/env python3
"""Gate-G5b / temporal-completion assertions.

Covers the checks the recovery stage adds on top of test_stage3_assertions.py:
screening completeness, refinement fidelity to the frozen config, non-overwrite of
frozen evidence, and matched-beta immutability.
"""
from __future__ import annotations

import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
import p0_common as c  # noqa: E402
import p0_6_temporal_designs as td  # noqa: E402

T = c.TABLES
CFG = c.CONFIGS
REP = c.P0_DIR / "reports"
DESIGNS = ["dsnap", "dpurge"]
BLOCKS = [f"fold_{k}" for k in range(1, 8)] + ["heldout", "forward_2025"]
FAMS = ["native", "direct", "surrogate"]
ATOL = 1e-12


def _refine_grid():
    return json.loads((CFG / "dsnap_refinement_grid.json").read_text())


def _screen_rhos():
    g = json.loads((CFG / "robustness_rho_grid.json").read_text())
    return [0.0] + [float(x) for x in g["positive_rhos"]]


def _eq_sets(a, b, atol=ATOL):
    """Set equality up to CSV float round-tripping."""
    a, b = sorted(float(x) for x in a), sorted(float(x) for x in b)
    return len(a) == len(b) and all(abs(x - y) <= atol for x, y in zip(a, b))


# ------------------------------------------------- original screening is intact
def test_original_screening_is_54_of_54():
    miss = [f"{d}/{b}/{f}" for d in DESIGNS for b in BLOCKS for f in FAMS
            if not (T / f"robustness_shard__{d}__{b}__{f}.csv").exists()]
    assert not miss, f"missing original screening shards: {miss}"
    assert len(glob.glob(str(T / "robustness_shard__*.csv"))) == 54


def test_original_screening_shards_carry_only_frozen_screening_rhos():
    want = _screen_rhos()
    for p in sorted(T.glob("robustness_shard__*.csv")):
        d = pd.read_csv(p)
        assert set(d.grid) == {"screening"}, f"{p.name}: not labelled screening"
        if set(d.family) == {"native"}:
            continue
        assert _eq_sets(d.rho, want), f"{p.name}: rho set is not the frozen screening grid"


def test_no_original_shard_was_overwritten():
    """The 54 screening shards are committed; they must be byte-identical to HEAD."""
    out = subprocess.run(
        ["git", "status", "--porcelain", "--",
         "analysis/p0_major_revision_validation/tables/robustness_shard__*"],
        cwd=str(c.REPO), capture_output=True, text=True).stdout.strip()
    assert out == "", f"a frozen screening shard changed:\n{out}"


# ---------------------------------------------------------- refinement fidelity
def test_every_required_refinement_shard_exists():
    g = _refine_grid()
    miss = []
    for r in g["regions"]:
        nch = td.N_CHUNK_A if r["id"] == "A" else 1
        for fam in r["families"]:
            for blk in BLOCKS:
                for ch in range(nch):
                    p = T / f"dsnap_refine_shard__{r['id']}__{blk}__{fam}__c{ch}.csv"
                    if not p.exists():
                        miss.append(p.name)
    assert not miss, f"missing refinement shards: {miss}"


def test_refinement_rhos_equal_the_frozen_refinement_config():
    g = _refine_grid()
    for r in g["regions"]:
        want = [float(x) for x in r["refinement_rhos"]]
        for fam in r["families"]:
            for blk in BLOCKS:
                got = []
                for p in T.glob(f"dsnap_refine_shard__{r['id']}__{blk}__{fam}__c*.csv"):
                    got += [float(x) for x in pd.read_csv(p).rho]
                assert _eq_sets(got, want), (
                    f"region {r['id']} {blk}/{fam}: refined rho set differs from the frozen "
                    "refinement grid")


def test_no_unplanned_rho_was_fit():
    """Union of every refined rho equals exactly the frozen refinement grid; and no
    refinement rho collides with a screening rho (they must be the SKIPPED points)."""
    g = _refine_grid()
    planned = sorted({float(x) for r in g["regions"] for x in r["refinement_rhos"]})
    got = sorted({float(x) for p in T.glob("dsnap_refine_shard__*.csv")
                  for x in pd.read_csv(p).rho})
    assert _eq_sets(got, planned), "a rho outside the frozen refinement grid was fit"
    scr = _screen_rhos()
    for x in got:
        assert not any(abs(x - s) <= ATOL for s in scr), (
            f"refinement rho {x} is already a screening rho; the refinement grid must "
            "contain only original-grid points the screen skipped")


def test_refinement_used_the_frozen_lgbm_config_and_historical_settings():
    hashes, settings = set(), set()
    for p in T.glob("dsnap_refine_shard__*.csv"):
        d = pd.read_csv(p)
        hashes |= set(d.lgbm_params_sha256)
        settings |= set(d.execution_settings)
        assert set(d.grid) == {"refinement"}
        assert set(d.design) == {"dsnap"}, "only D-SNAP feeds G5a/G5b refinement"
    assert len(hashes) == 1, f"refinement used more than one LightGBM config: {hashes}"
    assert hashes == {c.frozen_lgbm_config()["lgbm_params_sha256"]}
    assert settings == {"HISTORICAL (no determinism pins)"}


def test_no_dpurge_refinement_was_run():
    assert not list(T.glob("dpurge_refine_shard__*.csv")), \
        "D-PURGE is an oracle diagnostic and does not feed the promotion gate"


def test_refinement_provenance_table_is_complete():
    p = T / "dsnap_refinement_provenance.csv"
    assert p.exists(), "the compact refinement provenance table is required"
    d = pd.read_csv(p)
    g = _refine_grid()
    assert len(d) == len(glob.glob(str(T / "dsnap_refine_shard__*.csv")))
    assert int(d.n_rho.sum()) == g["n_refinement_fits"]
    assert d.rho_sha256.notna().all() and d.file_sha256.notna().all()


# ------------------------------------------------------------------- Gate G5b
def test_g5b_outcome_exists_and_is_well_formed():
    b = json.loads((T / "gate_g5b_outcome.json").read_text())
    assert b["status"] in ("NOT_CONFIRMED", "CONFIRMED_MATERIAL_CHANGE", "NOT_APPLICABLE")
    assert b["full_dsnap_regeneration_launched_in_this_run"] is False
    assert "not launched in this run" in b["hard_stop"].lower()


def test_g5b_materiality_tolerance_is_the_frozen_matched_beta_tau():
    b = json.loads((T / "gate_g5b_outcome.json").read_text())
    tau = json.loads((CFG / "matched_beta_frozen.json").read_text())["tau"]
    assert b["materiality_rule"]["TAU_MATCH"] == tau, \
        "G5b must key off the tolerance frozen BEFORE the refinement was seen"


def test_g5b_sign_test_actually_covers_the_region_b_support():
    """Region B refined the Surrogate only. A Direct-derived rho set would silently drop
    all 81 region-B fits, so the Surrogate support must exceed the Direct support."""
    b = json.loads((T / "gate_g5b_outcome.json").read_text())
    n = b["n_rho_after_refinement"]
    g = _refine_grid()
    nB = len([r for r in g["regions"] if r["id"] == "B"][0]["refinement_rhos"])
    assert n["Surrogate"] == n["Direct"] + nB, (
        f"Surrogate refined support {n['Surrogate']} should exceed Direct "
        f"{n['Direct']} by the {nB} region-B rhos")
    for e in b["evidence"]:
        assert e["frozen_sign"]["Surrogate"]["n_rho"] >= n["Direct"]


def test_dsnap_refinement_table_written():
    assert (T / "dsnap_refinement.csv").exists(), \
        "the G5a/G5b refinement outcome is recorded regardless of the verdict"


def test_temporal_status_consistent_with_g5b():
    b = json.loads((T / "gate_g5b_outcome.json").read_text())
    r = (REP / "TEMPORAL_ROBUSTNESS_REPORT.md").read_text()
    want = ("REQUIRES_FULL_DSNAP_REGEN" if b["status"] == "CONFIRMED_MATERIAL_CHANGE"
            else "PASS_PRIMARY_STANDS")
    assert f"TEMPORAL_STATUS    = {want}" in r, "report status disagrees with gate_g5b_outcome"


# -------------------------------------------------------------- matched-beta
def test_matched_beta_artifacts_unchanged():
    """Stage 3B must not touch the completed matched-beta evidence."""
    pats = ["tables/matched_beta*", "tables/development_beta_coordinate*",
            "configs/matched_beta*", "reports/MATCHED_BETA_REPORT.md", "figures/"]
    out = subprocess.run(
        ["git", "status", "--porcelain", "--",
         *[f"analysis/p0_major_revision_validation/{p}" for p in pats]],
        cwd=str(c.REPO), capture_output=True, text=True).stdout.strip()
    assert out == "", f"matched-beta artifacts changed in this stage:\n{out}"


def test_matched_beta_report_still_declares_its_primary_triple():
    r = (REP / "MATCHED_BETA_REPORT.md").read_text()
    assert "Direct / Surrogate / C-posthoc" in r
    assert "A-posthoc" in r and "SECONDARY" in r
    assert "Cell B has no post-hoc path" in r


# ------------------------------------------------------------------- report
def test_temporal_report_distinguishes_all_four_designs():
    p = REP / "TEMPORAL_ROBUSTNESS_REPORT.md"
    assert p.exists()
    r = p.read_text()
    flat = r.replace("*", "").replace("`", "")      # strip markdown emphasis before matching
    for k in ("PRIMARY frozen temporal design", "D-SNAP", "D-PURGE", "D-UNSEEN"):
        assert k in r, f"the report must name {k} explicitly"
    assert "oracle" in flat.lower() and "not a proposed sample-construction rule" in flat.lower(), \
        "D-PURGE must be labelled an oracle diagnostic in the report"
    assert "denominators change" in flat.lower(), \
        "D-UNSEEN denominator change must be stated"
    assert "not IID standard errors" in flat, \
        "the non-IID fold-SD reading must be preserved"
    assert "no row is ever predicted by a" in flat, \
        "the report must state predictions remain out-of-training-sample"


def test_report_does_not_claim_d1_simply_unaffected():
    r = (REP / "TEMPORAL_ROBUSTNESS_REPORT.md").read_text().replace("*", "")
    assert "fold-level results are not independent" in r
    assert "D1 is unaffected" not in r
