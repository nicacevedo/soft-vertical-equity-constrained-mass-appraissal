#!/usr/bin/env python3
"""Build the Tier-B0 certification record.

The scientific certification of P0 and P1 is what their FROZEN TAGS carry, not
what this integration worktree can re-execute. This worktree is missing the
gitignored parquet twins and the mtimes git does not store, and P0 carries a
HEAD-relative guard that any additive commit anywhere breaks. Re-running the
suites here therefore measures the environment, not the science.

So certification is assembled from three things that ARE checkable here:

  1. the frozen suite records committed at the tags
     (P0: provenance/p0_suite_at_tag.json -> 153/153, machine-readable;
      P1: reports/MANUSCRIPT_EVIDENCE_PACKAGE.md 0. Verification status ->
      72/72, prose -- P1 committed no p1_suite_at_tag.json, so this asymmetry
      is recorded as a provenance limitation rather than papered over);
  2. byte-identity of each stage subtree at integration HEAD against its tag;
  3. the frozen hash indices resolving, including the 8 early-stage index
     mismatches that the tag ITSELF already has -- claiming "every P0 index is
     byte-exact" would be false, and P1's report says so explicitly.

Local reruns are recorded separately, as environment-dependent DIAGNOSTICS with
their expected outcomes pre-declared, so a different outcome is a real finding.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import b0_common as c

CERT_JSON = c.CERT / "certification.json"
CERT_MD = c.CERT / "CERTIFICATION.md"

P1_72_QUOTE = ("**72 passed, 0 failed** — 64 scientific (12 PRB + 5 smearing-sign "
               "+ 28 VEI + 19 smearing-apply) + 8 report-consistency")
P1_72_HEADING = "0. Verification status"


def git(*args) -> str:
    return subprocess.run(["git", "-C", str(c.REPO), *args],
                          capture_output=True, text=True, check=True).stdout.strip()


def subtree_identical(tag: str, path: str) -> dict:
    out = git("diff", "--stat", tag, "HEAD", "--", path)
    return {"tag": tag, "path": path, "byte_identical": out == "",
            "git_diff_stat": out}


def main() -> int:
    # ---- 1. tags dereference to the recorded commits ----------------------
    tags = {}
    for tag, want in ((c.P0_TAG, c.P0_COMMIT), (c.P1_TAG, c.P1_COMMIT)):
        got = git("rev-parse", tag + "^{commit}")
        tags[tag] = {"annotated_object": git("rev-parse", tag),
                     "commit": got, "expected_commit": want,
                     "matches": got == want}

    # ---- 2. P0 primary certification --------------------------------------
    suite = c.read_json(c.P1_PROV / "p0_suite_at_tag.json")
    rs = suite["result_structured"]
    p0 = {
        "final_tag_certification": "153/153",
        "source": c.rel(c.P1_PROV / "p0_suite_at_tag.json"),
        "source_sha256": c.sha256_file(c.P1_PROV / "p0_suite_at_tag.json"),
        "selector": "$.result_structured",
        "passed": rs["passed"], "failed": rs["failed"], "skipped": rs["skipped"],
        "n_tests_total": suite["n_tests_total"],
        "suites": suite["suites"],
        "certified": rs["passed"] == 153 and rs["failed"] == 0
                     and suite["n_tests_total"] == 153,
        "measured_at": suite["what"],
        "subtree_identity": subtree_identical(
            c.P0_TAG, "analysis/p0_major_revision_validation"),
        "p0_not_modified": suite["p0_not_modified"],
        "p0_suite_not_modified": suite["p0_suite_not_modified"],
    }

    # ---- 3. P1 primary certification --------------------------------------
    pkg = c.P1_REPORTS / "MANUSCRIPT_EVIDENCE_PACKAGE.md"
    quote_ok = True
    try:
        c.select_markdown_quote(pkg, P1_72_HEADING, P1_72_QUOTE)
    except c.SelectorError as e:
        quote_ok = False
        print("WARNING:", e)
    p1 = {
        "final_certification": "72/72",
        "source": c.rel(pkg),
        "source_sha256": c.sha256_file(pkg),
        "selector": f'heading="{P1_72_HEADING}"; quote="{P1_72_QUOTE}"',
        "breakdown": {"scientific": 64, "prb": 12, "smearing_sign": 5, "vei": 28,
                      "smearing_apply": 19, "report_consistency": 8},
        "certified": quote_ok,
        "subtree_identity": subtree_identical(
            c.P1_TAG, "analysis/p1_inferential_reporting"),
        "provenance_limitation": (
            "P1 committed no machine-readable p1_suite_at_tag.json, the analogue "
            "of P0's. The 72/72 figure is therefore certified from the prose "
            "table in reports/MANUSCRIPT_EVIDENCE_PACKAGE.md section '0. "
            "Verification status', resolved by an exact-quote selector. "
            "P1_CHECKPOINT.md repeats 72/72 but is explicitly SUPERSEDED (its "
            "Stage-3B '86/86' is wrong; the freeze records 28/28), so it is not "
            "used as a source here."),
    }

    # ---- 4. manuscript side ------------------------------------------------
    manuscript = {
        "tier_a_commit": c.TIER_A_COMMIT,
        "paper_identity": subtree_identical(c.TIER_A_COMMIT, "paper"),
        "canonical_tex": c.rel(c.TEX),
        "canonical_tex_sha256": c.sha256_file(c.TEX),
        "canonical_tex_sha256_matches_pin": c.sha256_file(c.TEX) == c.TEX_SHA256,
        "tier_b0_edits_to_manuscript": 0,
    }

    # ---- 5. known, pre-existing index mismatches ---------------------------
    known = {
        "what": "Two early-stage P0 hash indexes do not match byte-for-byte, and "
                "did not at the tag either. Recorded so no Tier-B0 output claims "
                "otherwise.",
        "stage1_frozen_hashes_json": "48/49",
        "output_artifact_hashes_json": "118/125",
        "final_gate_index_stage3b": "28/28 byte-exact",
        "reason": "early-stage snapshots superseded by P0's own later gates: the "
                  "G2/G3 test modules were edited in later stages; POSTFLIGHT, "
                  "slurm_graph and large_local_artifacts are end-of-run "
                  "summaries; and output_artifact_hashes.json lists itself, a "
                  "self-reference that can never match.",
        "identical_at_tag_and_on_branch": True,
        "source": c.rel(pkg),
    }

    # ---- 6. diagnostics: expected local rerun outcomes ---------------------
    diagnostics = {
        "status": "NOT_THE_SCIENTIFIC_CERTIFICATION",
        "why": "environment-dependent. Reported separately; never quoted as "
               "certification.",
        "measured_in_this_worktree": {
            "p0_local_rerun": {
                "result": "140 passed, 13 failed, 0 skipped",
                "n_tests_total": 153,
                "correction":
                    "An earlier draft of this record pre-declared '152 passed, "
                    "1 failed'. That figure is real but belongs to a DIFFERENT "
                    "location: it was measured in the main working tree "
                    "(provenance/p0_suite_on_p1_branch.json), where the "
                    "gitignored prediction trees and the CCAO parquet are "
                    "present. This is a content-only git worktree, so eight "
                    "further assertions cannot run at all. The record was "
                    "corrected to the measured outcome rather than the outcome "
                    "re-explained.",
                "failure_classes": {
                    "MISSING_GITIGNORED_ARTIFACT": {
                        "n": 8,
                        "tests": [
                            "test_fold_index_hashes_verified_live",
                            "test_frozen_lgbm_params_hash",
                            "test_frozen_rho_grid_is_83_points",
                            "test_fold_index_hashes_match_archive",
                            "test_input_prediction_hashes_verified",
                            "test_dpurge_evaluation_sets_bitwise_identical_to_primary",
                            "test_race_used_the_frozen_configuration_and_dsnap_protocol",
                            "test_refinement_used_the_frozen_lgbm_config_and_historical_settings",
                        ],
                        "why": "they read output/paper_v6_preselection_994, "
                               "output/paper_v12_lower_rho_extension_994_v2, "
                               "output/p0_major_revision_validation or "
                               "data/CCAO/2025/training_data.parquet -- all "
                               "gitignored by design and in no checkout. Tier B0 "
                               "quotes their hashes from the frozen index rather "
                               "than reading them.",
                    },
                    "FILE_MODE_NOT_STORED_BY_GIT": {
                        "n": 3,
                        "tests": ["test_approved_plan_is_frozen_readonly",
                                  "test_approved_plan_still_frozen_readonly",
                                  "test_approved_plan_still_readonly"],
                        "why": "three tests assert APPROVED_EXECUTION_PLAN.md is "
                               "mode 0444. Git records only the exec bit "
                               "(100644), so a checkout creates it 0664 and no "
                               "checkout can reproduce 0444.",
                    },
                    "WORKTREE_OR_HEAD_RELATIVE_GUARD": {
                        "n": 2,
                        "tests": [
                            "test_only_gitignore_modified_outside_p0",
                            "test_all_stage1_writes_are_inside_approved_locations",
                        ],
                        "why": "both compare the repository against a P0-era "
                               "baseline and assert nothing changed outside the "
                               "P0 tree. They are doing exactly their job: "
                               "registering that additive areas exist -- P1, and "
                               "now Tier B0. Neither indicates a P0 change.",
                    },
                },
                "p0_content_defects": 0,
                "why_this_is_not_a_p0_regression":
                    "The P0 subtree at this HEAD is byte-identical to the P0 tag "
                    "(git diff over the tree is empty), so no failure here can be "
                    "a content defect. All 13 fall into the three environment "
                    "classes above -- the same classes "
                    "provenance/p0_suite_at_tag.json documents having had to "
                    "reconstitute in order to reach 153/153.",
                "recorded_precedent": c.rel(
                    c.P1_PROV / "p0_suite_on_p1_branch.json"),
            },
            "p1_local_rerun": {
                "result": "69 passed, 3 failed",
                "n_tests_total": 72,
                "failure_classes": {
                    "MISSING_GITIGNORED_PARQUET_TWIN": {
                        "n": 2,
                        "tests": [
                            "test_ci_limits_are_true_order_statistics_on_a_bounded_subset",
                            "test_csv_is_a_faithful_serialisation_of_the_parquet_twin",
                        ],
                        "why": "both read P1's gitignored parquet twins, which no "
                               "checkout contains. This was pre-declared.",
                    },
                    "WORKTREE_OR_HEAD_RELATIVE_GUARD": {
                        "n": 1,
                        "tests": [
                            "test_reports_state_that_no_manuscript_file_was_edited"],
                        "why": "the guard compares paper/ against P1's baseline "
                               "and fires on the TIER-A manuscript commit, which "
                               "is the approved baseline of this branch. Tier B0 "
                               "edited no manuscript file: paper/ at this HEAD is "
                               "byte-identical to the Tier-A checkpoint.",
                    },
                },
                "p1_content_defects": 0,
            },
        },
        "rule": "A DIFFERENT outcome than the one recorded here is a real finding "
                "and must be investigated, not re-explained. That rule was "
                "exercised: the first measured P0 outcome disagreed with the "
                "pre-declaration, and the pre-declaration was the thing that was "
                "wrong.",
    }

    cert = {
        "schema_version": 1,
        "generated_by": c.rel(Path(__file__)),
        "what": "Tier-B0 certification record. Primary certification comes from "
                "the frozen tags plus subtree identity, NOT from reruns in this "
                "integration worktree.",
        # `head` is the PIN: the frozen-evidence state Tier B0 was built
        # against. The live HEAD is deliberately NOT recorded here -- embedding
        # it would make this artifact change on every commit, so the
        # determinism check would fail once after each one. The liveness check
        # lives in tests/test_b0_certification.py instead, which asserts the
        # live HEAD is the pin or a descendant of it.
        "integration": {"branch": "paper-major-revision-integration",
                        "head": c.INTEGRATION_HEAD,
                        "head_is_a_pin_not_the_live_head": True},
        "tags": tags,
        "primary_certification": {"p0": p0, "p1": p1, "manuscript": manuscript,
                                  "tier_b0": {
                                      "requirement": "the complete Tier-B0 suite "
                                                     "must pass 100%",
                                      "runner": c.rel(c.TESTS / "run_all_tests.py")}},
        "known_preexisting_index_mismatches": known,
        "diagnostic_reruns": diagnostics,
        "gate_g4": {
            "finding": "Gate G4 has no artifacts inside P0.",
            "impact_memo_precondition_met_by": c.rel(
                c.P1_REPORTS / "MANUSCRIPT_IMPACT_MEMO.md"),
            "impact_memo_sha256": c.sha256_file(
                c.P1_REPORTS / "MANUSCRIPT_IMPACT_MEMO.md"),
            "note": "the Tier-B gate was conditioned on that memo existing; it does.",
        },
        "scope_caveat": "protocol_p0_validation.yaml still declares scope "
                        "STAGE_1_THROUGH_GATE_G1 and was never refreshed. Cite the "
                        "stage-specific gate JSONs for authorization state, not it.",
    }

    ok = (all(t["matches"] for t in tags.values())
          and p0["certified"] and p0["subtree_identity"]["byte_identical"]
          and p1["certified"] and p1["subtree_identity"]["byte_identical"]
          and manuscript["paper_identity"]["byte_identical"]
          and manuscript["canonical_tex_sha256_matches_pin"])
    cert["all_primary_checks_pass"] = ok

    c.write_json(CERT_JSON, cert)
    c.write_text(CERT_MD, _render_md(cert))
    print(f"wrote {c.rel(CERT_JSON)}")
    print(f"wrote {c.rel(CERT_MD)}")
    print(f"  P0 tag certification   {p0['final_tag_certification']}  "
          f"certified={p0['certified']}  subtree_identical="
          f"{p0['subtree_identity']['byte_identical']}")
    print(f"  P1 certification       {p1['final_certification']}  "
          f"certified={p1['certified']}  subtree_identical="
          f"{p1['subtree_identity']['byte_identical']}")
    print(f"  paper/ vs Tier-A       identical="
          f"{manuscript['paper_identity']['byte_identical']}")
    print(f"  ALL PRIMARY CHECKS     {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def _render_md(cert: dict) -> str:
    p0 = cert["primary_certification"]["p0"]
    p1 = cert["primary_certification"]["p1"]
    ms = cert["primary_certification"]["manuscript"]
    d = cert["diagnostic_reruns"]["measured_in_this_worktree"]
    L = []
    A = L.append
    A("# Tier-B0 certification")
    A("")
    A("Machine-readable twin: `certification/certification.json`. "
      "Regenerate with `code/b0_1b_build_certification.py`.")
    A("")
    A("The scientific certification of P0 and P1 is **what their frozen tags "
      "carry**, together with byte-identity of each stage subtree at integration "
      "HEAD. It is *not* what this worktree can re-execute: the gitignored "
      "parquet twins are absent, git stores no mtimes, and P0 carries a "
      "HEAD-relative guard that any additive commit anywhere breaks. Reruns here "
      "measure the environment, not the science.")
    A("")
    A("## Primary certification")
    A("")
    A("| Stage | Certification | Source | Subtree byte-identical to tag |")
    A("|---|---|---|---|")
    A(f"| P0 | **{p0['final_tag_certification']}** "
      f"({p0['passed']} passed / {p0['failed']} failed / {p0['skipped']} skipped) "
      f"| `{p0['source']}` `$.result_structured` | "
      f"{'yes' if p0['subtree_identity']['byte_identical'] else 'NO'} |")
    A(f"| P1 | **{p1['final_certification']}** "
      f"(64 scientific + 8 report-consistency) | `{p1['source']}` "
      f"section *{P1_72_HEADING}* | "
      f"{'yes' if p1['subtree_identity']['byte_identical'] else 'NO'} |")
    A(f"| Manuscript | no Tier-B0 edit | `{ms['canonical_tex']}` sha256 "
      f"`{ms['canonical_tex_sha256'][:16]}...` | "
      f"`paper/` vs Tier-A `{ms['tier_a_commit'][:8]}`: "
      f"{'yes' if ms['paper_identity']['byte_identical'] else 'NO'} |")
    A("| Tier B0 | complete suite must pass **100%** | "
      f"`{cert['primary_certification']['tier_b0']['runner']}` | n/a |")
    A("")
    A("P0's per-suite breakdown, as the tag records it:")
    A("")
    A("| suite | tests |")
    A("|---|---:|")
    for k, v in sorted(p0["suites"].items()):
        A(f"| {k} | {v} |")
    A("")
    A("### P1 provenance limitation")
    A("")
    A(p1["provenance_limitation"])
    A("")
    A("## Known pre-existing index mismatches (not Tier-B0 regressions)")
    A("")
    k = cert["known_preexisting_index_mismatches"]
    A(k["what"])
    A("")
    A(f"- `stage1_frozen_hashes.json` — **{k['stage1_frozen_hashes_json']}**")
    A(f"- `output_artifact_hashes.json` — **{k['output_artifact_hashes_json']}**")
    A(f"- final-gate index (Stage-3B / G5b) — **{k['final_gate_index_stage3b']}**")
    A("")
    A(f"Reason: {k['reason']}")
    A("")
    A("## Diagnostic reruns — NOT the certification")
    A("")
    A("These are what the frozen suites actually do **in this content-only git "
      "worktree**. They measure the environment, not the science. A different "
      "outcome than the one recorded here is a real finding, to be investigated "
      "rather than re-explained -- and that rule has already been exercised: the "
      "first measured P0 outcome disagreed with an earlier pre-declaration, and "
      "the pre-declaration was what turned out to be wrong.")
    A("")
    for stage, rec in (("P0", d["p0_local_rerun"]), ("P1", d["p1_local_rerun"])):
        A(f"### {stage} — `{rec['result']}` of {rec['n_tests_total']}")
        A("")
        if "correction" in rec:
            A(f"> **Correction.** {' '.join(rec['correction'].split())}")
            A("")
        A("| class | n | why |")
        A("|---|---:|---|")
        for cls, info in rec["failure_classes"].items():
            A(f"| `{cls}` | {info['n']} | {' '.join(info['why'].split())} |")
        A("")
        defects = rec.get("p0_content_defects", rec.get("p1_content_defects"))
        line = f"**Content defects: {defects}.**"
        if "why_this_is_not_a_p0_regression" in rec:
            line += " " + " ".join(rec["why_this_is_not_a_p0_regression"].split())
        A(line)
        A("")
        for cls, info in rec["failure_classes"].items():
            A(f"<details><summary>{cls} — the {info['n']} tests</summary>")
            A("")
            for t in info["tests"]:
                A(f"- `{t}`")
            A("")
            A("</details>")
            A("")
    A("")
    A("## Gate G4")
    A("")
    g = cert["gate_g4"]
    A(f"{g['finding']} {g['note'].capitalize()} Impact memo: "
      f"`{g['impact_memo_precondition_met_by']}` "
      f"(sha256 `{g['impact_memo_sha256'][:16]}...`).")
    A("")
    A("## Authorization-state caveat")
    A("")
    A(cert["scope_caveat"])
    A("")
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
