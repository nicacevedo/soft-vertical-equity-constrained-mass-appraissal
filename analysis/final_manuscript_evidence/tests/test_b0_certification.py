#!/usr/bin/env python3
"""Certification assertions.

The scientific certification is what the FROZEN TAGS carry plus subtree
byte-identity. Local reruns in this integration worktree are environment-
dependent diagnostics and are asserted to be RECORDED AS SUCH, never as the
certification.
"""
from __future__ import annotations

import subprocess

import b0_common as c

CERT = c.read_json(c.CERT / "certification.json")


def git(*a) -> str:
    return subprocess.run(["git", "-C", str(c.REPO), *a],
                          capture_output=True, text=True, check=True).stdout.strip()


def test_p0_final_tag_certification_is_153_of_153():
    p0 = CERT["primary_certification"]["p0"]
    assert p0["final_tag_certification"] == "153/153"
    src = c.P1_PROV / "p0_suite_at_tag.json"
    assert c.sha256_file(src) == p0["source_sha256"]
    rs = c.read_json(src)["result_structured"]
    assert rs["passed"] == 153 and rs["failed"] == 0 and rs["skipped"] == 0
    assert c.read_json(src)["n_tests_total"] == 153
    assert sum(rs["suites"].values()) == 153
    assert p0["certified"] is True


def test_p0_subtree_is_byte_identical_at_integration_head():
    assert CERT["primary_certification"]["p0"]["subtree_identity"]["byte_identical"]
    assert git("diff", "--stat", c.P0_TAG, "HEAD", "--",
               "analysis/p0_major_revision_validation") == ""


def test_p1_final_certification_is_72_of_72():
    p1 = CERT["primary_certification"]["p1"]
    assert p1["final_certification"] == "72/72"
    pkg = c.P1_REPORTS / "MANUSCRIPT_EVIDENCE_PACKAGE.md"
    assert c.sha256_file(pkg) == p1["source_sha256"]
    # the quote selector must still resolve, exactly once, under its heading
    import re
    m = re.match(r'^heading="(.*?)"; quote="(.*)"$', p1["selector"], re.S)
    assert m, p1["selector"]
    c.select_markdown_quote(pkg, m.group(1), m.group(2))
    b = p1["breakdown"]
    assert b["prb"] + b["smearing_sign"] + b["vei"] + b["smearing_apply"] == \
        b["scientific"] == 64
    assert b["scientific"] + b["report_consistency"] == 72
    # cross-check against the machine-readable headline numbers
    h = c.read_json(c.P1_PROV / "p1_headline_numbers.json")["test_suites"]
    assert h["total"] == 72 and h["scientific_subtotal"] == 64


def test_p1_provenance_limitation_is_recorded():
    """P1 committed no p1_suite_at_tag.json. That asymmetry must be stated, not
    papered over, and the superseded checkpoint must not be used as a source."""
    p1 = CERT["primary_certification"]["p1"]
    lim = p1["provenance_limitation"]
    assert "p1_suite_at_tag.json" in lim
    assert "SUPERSEDED" in lim
    assert not (c.P1_PROV / "p1_suite_at_tag.json").exists()
    assert "P1_CHECKPOINT" not in p1["source"]


def test_p1_subtree_is_byte_identical_at_integration_head():
    assert CERT["primary_certification"]["p1"]["subtree_identity"]["byte_identical"]
    assert git("diff", "--stat", c.P1_TAG, "HEAD", "--",
               "analysis/p1_inferential_reporting") == ""


def test_manuscript_is_untouched():
    ms = CERT["primary_certification"]["manuscript"]
    assert ms["paper_identity"]["byte_identical"]
    assert ms["canonical_tex_sha256_matches_pin"]
    assert ms["tier_b0_edits_to_manuscript"] == 0
    assert git("diff", "--stat", c.TIER_A_COMMIT, "HEAD", "--", "paper") == ""


def test_local_reruns_are_labelled_diagnostics_not_certification():
    d = CERT["diagnostic_reruns"]
    assert d["status"] == "NOT_THE_SCIENTIFIC_CERTIFICATION"
    m = d["measured_in_this_worktree"]
    for stage in ("p0_local_rerun", "p1_local_rerun"):
        rec = m[stage]
        assert rec["result"], stage
        # the class counts must add up to the failure count in `result`
        n_failed = int(rec["result"].split("passed,")[1].split("failed")[0])
        assert sum(x["n"] for x in rec["failure_classes"].values()) == n_failed, \
            f"{stage}: class counts do not sum to {n_failed}"
        # and every class must list exactly n named tests
        for cls, info in rec["failure_classes"].items():
            assert len(info["tests"]) == info["n"], f"{stage}/{cls}"
            assert info["why"], f"{stage}/{cls}"
        assert rec.get("p0_content_defects", rec.get("p1_content_defects")) == 0
    assert m["p0_local_rerun"]["result"] == "140 passed, 13 failed, 0 skipped"
    assert m["p1_local_rerun"]["result"] == "69 passed, 3 failed"
    # the correction must be recorded, not the outcome re-explained
    assert "152 passed" in m["p0_local_rerun"]["correction"]
    assert "corrected to the measured outcome" in m["p0_local_rerun"]["correction"]
    assert "re-explained" in d["rule"]
    # the precedent P1 recorded is the MAIN-worktree figure, and still says so
    rec = c.read_json(c.P1_PROV / "p0_suite_on_p1_branch.json")
    assert rec["passed"] == 152
    assert rec["failed_tests"] == ["test_only_gitignore_modified_outside_p0"]
    assert "main working tree" in rec["what"] or "main working tree" in str(rec)


def test_the_diagnostic_failure_classes_are_all_environmental():
    """No class may describe a content defect: the subtrees are byte-identical."""
    m = CERT["diagnostic_reruns"]["measured_in_this_worktree"]
    environmental = {"MISSING_GITIGNORED_ARTIFACT",
                     "MISSING_GITIGNORED_PARQUET_TWIN",
                     "FILE_MODE_NOT_STORED_BY_GIT",
                     "WORKTREE_OR_HEAD_RELATIVE_GUARD"}
    for stage in ("p0_local_rerun", "p1_local_rerun"):
        got = set(m[stage]["failure_classes"])
        assert got <= environmental, f"{stage}: non-environmental class {got}"
    # the file-mode class is checkable: git stores only the exec bit
    plan = c.P0 / "APPROVED_EXECUTION_PLAN.md"
    assert plan.exists()
    assert (plan.stat().st_mode & 0o777) != 0o444, (
        "the plan is 0444 here, so the FILE_MODE class no longer applies and "
        "the record must be re-measured")
    # the parquet twins really are absent
    assert not list(c.P1_TABLES.glob("*.parquet"))


def test_known_index_mismatches_are_recorded_rather_than_claimed_away():
    """"Every P0 index is byte-exact" would be false, and was never true."""
    k = CERT["known_preexisting_index_mismatches"]
    assert k["stage1_frozen_hashes_json"] == "48/49"
    assert k["output_artifact_hashes_json"] == "118/125"
    assert k["final_gate_index_stage3b"].startswith("28/28")
    assert k["identical_at_tag_and_on_branch"] is True
    # the reason must name the self-reference, which can never match
    assert "lists itself" in k["reason"]


def test_gate_g4_impact_memo_precondition_is_met():
    g = CERT["gate_g4"]
    memo = c.P1_REPORTS / "MANUSCRIPT_IMPACT_MEMO.md"
    assert memo.exists()
    assert c.sha256_file(memo) == g["impact_memo_sha256"]
    assert "no artifacts inside P0" in g["finding"]


def test_authorization_scope_caveat_is_recorded():
    assert "STAGE_1_THROUGH_GATE_G1" in CERT["scope_caveat"]
    proto = c.read_yaml(c.P0 / "protocol_p0_validation.yaml")
    assert proto["authorized_scope"]["this_run"] == "STAGE_1_THROUGH_GATE_G1"
    assert proto["authorized_scope"]["hard_stop_after"] == "GATE_G1"
    # ... yet Stage 3B / Gate G5b artifacts exist, which is exactly why the
    # stage-specific gate JSONs, not this file, state the authorization status.
    assert (c.P0_TABLES / "gate_g5b_outcome.json").exists()


def test_all_primary_checks_pass():
    assert CERT["all_primary_checks_pass"] is True


def test_the_recorded_head_is_a_pin_and_the_live_head_descends_from_it():
    """The artifact records a PIN, never the live HEAD: embedding a live value
    would make the file change on every commit, and the determinism check would
    then fail once after each one. The liveness check belongs here."""
    integ = CERT["integration"]
    assert integ["head"] == c.INTEGRATION_HEAD
    assert integ["head_is_a_pin_not_the_live_head"] is True
    assert "head_now" not in integ, (
        "the live HEAD must not be embedded in a deterministic artifact")
    live = git("rev-parse", "HEAD")
    if live != c.INTEGRATION_HEAD:
        # the pin must be an ancestor: Tier B0 is additive on top of it
        rc = subprocess.run(
            ["git", "-C", str(c.REPO), "merge-base", "--is-ancestor",
             c.INTEGRATION_HEAD, live]).returncode
        assert rc == 0, (
            f"live HEAD {live[:12]} does not descend from the recorded pin "
            f"{c.INTEGRATION_HEAD[:12]}; the frozen-evidence baseline moved")
