#!/usr/bin/env python3
"""Manifest assertions: every hash either recomputes or is quoted transitively."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import b0_common as c

MAN = c.read_json(c.B0 / "FINAL_EVIDENCE_MANIFEST.json")
ARTS = MAN["artifacts"]


def git(*a) -> str:
    return subprocess.run(["git", "-C", str(c.REPO), *a],
                          capture_output=True, text=True, check=True).stdout.strip()


def test_present_artifacts_exist_and_hashes_recompute():
    bad = []
    for path, rec in ARTS.items():
        if not rec["present_in_worktree"]:
            continue
        f = c.REPO / path
        if not f.exists():
            bad.append(f"{path}: marked present but absent")
            continue
        if rec["hash_source"] != "RECOMPUTED":
            bad.append(f"{path}: present artifacts must be RECOMPUTED, "
                       f"got {rec['hash_source']}")
        got = c.sha256_file(f)
        if got != rec["sha256"]:
            bad.append(f"{path}: sha256 {got} != recorded {rec['sha256']}")
        if f.stat().st_size != rec["bytes"]:
            bad.append(f"{path}: size mismatch")
    assert not bad, "\n".join(bad[:20])


def test_absent_artifacts_are_quoted_transitively():
    """An absent hash must match the frozen index named by hash_source, whose own
    sha256 is recorded in the manifest and recomputed here."""
    idx_paths = {v["role"]: k for k, v in MAN["hash_indices"].items()}
    bad = []
    for path, rec in ARTS.items():
        if rec["present_in_worktree"]:
            continue
        assert rec["hash_source"] != "RECOMPUTED", path
        src_role = rec["hash_source"]
        assert src_role in idx_paths, f"{path}: unknown hash_source {src_role}"
        idx_file = c.REPO / rec["hash_source_file"]
        # the index's own sha256 must recompute
        recorded = MAN["hash_indices"][c.rel(idx_file)]["sha256"]
        got = c.sha256_file(idx_file)
        if got != recorded:
            bad.append(f"{c.rel(idx_file)}: index sha256 {got} != {recorded}")
            continue
        quoted = c.select_json_value(idx_file, rec["hash_source_key"])
        if quoted != rec["sha256"]:
            bad.append(f"{path}: quoted {rec['sha256']} != index {quoted}")
        if (c.REPO / path).exists():
            bad.append(f"{path}: marked absent but exists in the worktree")
    assert not bad, "\n".join(bad[:20])
    assert sum(1 for r in ARTS.values() if not r["present_in_worktree"]) == 11


def test_scientific_stage_and_role_vocabularies_are_closed():
    bad = [f"{p}: stage {r['scientific_stage']}" for p, r in ARTS.items()
           if r["scientific_stage"] not in c.SCIENTIFIC_STAGES]
    assert not bad, "\n".join(bad[:20])
    roles = {r["role"] for r in ARTS.values()}
    allowed = {"primary_evidence", "intermediate", "provenance",
               "convention_or_config", "report", "figure", "code", "test",
               "slurm_script", "protocol", "manuscript_source",
               "wording_precedent_only", "bibliography",
               "bibliography_staging_source", "manuscript_analysis",
               "legacy_frozen_input"}
    assert roles <= allowed, f"unexpected roles: {roles - allowed}"


def test_tags_dereference_to_recorded_commits():
    assert git("rev-parse", c.P0_TAG + "^{commit}") == c.P0_COMMIT
    assert git("rev-parse", c.P1_TAG + "^{commit}") == c.P1_COMMIT
    coords = MAN["coordinates"]
    assert coords["p0_commit"] == c.P0_COMMIT
    assert coords["p1_commit"] == c.P1_COMMIT


def test_frozen_subtrees_are_byte_identical_to_their_tags():
    assert git("diff", "--stat", c.P0_TAG, "HEAD", "--",
               "analysis/p0_major_revision_validation") == ""
    assert git("diff", "--stat", c.P1_TAG, "HEAD", "--",
               "analysis/p1_inferential_reporting") == ""


def test_paper_is_byte_identical_to_the_tier_a_checkpoint():
    assert git("diff", "--stat", c.TIER_A_COMMIT, "HEAD", "--", "paper") == ""


def test_manuscript_sha256_is_pinned():
    assert c.sha256_file(c.TEX) == c.TEX_SHA256
    assert MAN["coordinates"]["manuscript_sha256"] == c.TEX_SHA256


def test_no_output_or_data_path_is_marked_present():
    bad = [p for p, r in ARTS.items()
           if r["present_in_worktree"]
           and (p.startswith("output/") or p.startswith("data/"))]
    assert not bad, f"output/ or data/ paths marked present: {bad}"


def test_no_external_benchmark_artifact_is_indexed():
    forbidden = [c.rel(d) for d in c.FORBIDDEN_EVIDENCE_DIRS]
    bad = [p for p in ARTS if any(p.startswith(f + "/") for f in forbidden)]
    assert not bad, f"external-benchmark artifacts in the manifest: {bad}"
    # and the exclusion is recorded, not merely absent
    reasons = {e["reason"] for e in MAN["excluded_streams"]}
    assert reasons == {"OUT_OF_SCOPE_FOR_B0"}
    paths = " ".join(e["path"] for e in MAN["excluded_streams"])
    assert "external_jurisdiction_benchmark_v1" in paths


def test_manifest_counts_are_consistent():
    n_present = sum(1 for r in ARTS.values() if r["present_in_worktree"])
    assert MAN["counts"]["present"] == n_present
    assert MAN["counts"]["total"] == len(ARTS)
    assert sum(MAN["counts"]["by_stage"].values()) == len(ARTS)


def test_manifest_is_deterministic():
    """Rebuilding must produce byte-identical output."""
    before = (c.B0 / "FINAL_EVIDENCE_MANIFEST.json").read_bytes()
    import b0_1_build_manifest
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        assert b0_1_build_manifest.main() == 0
    after = (c.B0 / "FINAL_EVIDENCE_MANIFEST.json").read_bytes()
    assert before == after, "manifest rebuild is not byte-identical"
