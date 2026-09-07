# Tier-B0 certification

Machine-readable twin: `certification/certification.json`. Regenerate with `code/b0_1b_build_certification.py`.

The scientific certification of P0 and P1 is **what their frozen tags carry**, together with byte-identity of each stage subtree at integration HEAD. It is *not* what this worktree can re-execute: the gitignored parquet twins are absent, git stores no mtimes, and P0 carries a HEAD-relative guard that any additive commit anywhere breaks. Reruns here measure the environment, not the science.

## Primary certification

| Stage | Certification | Source | Subtree byte-identical to tag |
|---|---|---|---|
| P0 | **153/153** (153 passed / 0 failed / 0 skipped) | `analysis/p1_inferential_reporting/provenance/p0_suite_at_tag.json` `$.result_structured` | yes |
| P1 | **72/72** (64 scientific + 8 report-consistency) | `analysis/p1_inferential_reporting/reports/MANUSCRIPT_EVIDENCE_PACKAGE.md` section *0. Verification status* | yes |
| Manuscript | no Tier-B0 edit | `paper/paper_v17_option1.tex` sha256 `13c84ce7e799d485...` | `paper/` vs Tier-A `097544ff`: yes |
| Tier B0 | complete suite must pass **100%** | `analysis/final_manuscript_evidence/tests/run_all_tests.py` | n/a |

P0's per-suite breakdown, as the tag records it:

| suite | tests |
|---|---:|
| Stage-1 assertions | 28 |
| Stage-1.5 / Gate-G2 assertions | 36 |
| Stage-2 / Gate-G3 assertions | 36 |
| Stage-3 matched-beta + temporal assertions | 29 |
| Stage-3B / Gate-G5b temporal-completion assertions | 24 |

### P1 provenance limitation

P1 committed no machine-readable p1_suite_at_tag.json, the analogue of P0's. The 72/72 figure is therefore certified from the prose table in reports/MANUSCRIPT_EVIDENCE_PACKAGE.md section '0. Verification status', resolved by an exact-quote selector. P1_CHECKPOINT.md repeats 72/72 but is explicitly SUPERSEDED (its Stage-3B '86/86' is wrong; the freeze records 28/28), so it is not used as a source here.

## Known pre-existing index mismatches (not Tier-B0 regressions)

Two early-stage P0 hash indexes do not match byte-for-byte, and did not at the tag either. Recorded so no Tier-B0 output claims otherwise.

- `stage1_frozen_hashes.json` — **48/49**
- `output_artifact_hashes.json` — **118/125**
- final-gate index (Stage-3B / G5b) — **28/28 byte-exact**

Reason: early-stage snapshots superseded by P0's own later gates: the G2/G3 test modules were edited in later stages; POSTFLIGHT, slurm_graph and large_local_artifacts are end-of-run summaries; and output_artifact_hashes.json lists itself, a self-reference that can never match.

## Diagnostic reruns — NOT the certification

These are what the frozen suites actually do **in this content-only git worktree**. They measure the environment, not the science. A different outcome than the one recorded here is a real finding, to be investigated rather than re-explained -- and that rule has already been exercised: the first measured P0 outcome disagreed with an earlier pre-declaration, and the pre-declaration was what turned out to be wrong.

### P0 — `140 passed, 13 failed, 0 skipped` of 153

> **Correction.** An earlier draft of this record pre-declared '152 passed, 1 failed'. That figure is real but belongs to a DIFFERENT location: it was measured in the main working tree (provenance/p0_suite_on_p1_branch.json), where the gitignored prediction trees and the CCAO parquet are present. This is a content-only git worktree, so eight further assertions cannot run at all. The record was corrected to the measured outcome rather than the outcome re-explained.

| class | n | why |
|---|---:|---|
| `MISSING_GITIGNORED_ARTIFACT` | 8 | they read output/paper_v6_preselection_994, output/paper_v12_lower_rho_extension_994_v2, output/p0_major_revision_validation or data/CCAO/2025/training_data.parquet -- all gitignored by design and in no checkout. Tier B0 quotes their hashes from the frozen index rather than reading them. |
| `FILE_MODE_NOT_STORED_BY_GIT` | 3 | three tests assert APPROVED_EXECUTION_PLAN.md is mode 0444. Git records only the exec bit (100644), so a checkout creates it 0664 and no checkout can reproduce 0444. |
| `WORKTREE_OR_HEAD_RELATIVE_GUARD` | 2 | both compare the repository against a P0-era baseline and assert nothing changed outside the P0 tree. They are doing exactly their job: registering that additive areas exist -- P1, and now Tier B0. Neither indicates a P0 change. |

**Content defects: 0.** The P0 subtree at this HEAD is byte-identical to the P0 tag (git diff over the tree is empty), so no failure here can be a content defect. All 13 fall into the three environment classes above -- the same classes provenance/p0_suite_at_tag.json documents having had to reconstitute in order to reach 153/153.

<details><summary>MISSING_GITIGNORED_ARTIFACT — the 8 tests</summary>

- `test_fold_index_hashes_verified_live`
- `test_frozen_lgbm_params_hash`
- `test_frozen_rho_grid_is_83_points`
- `test_fold_index_hashes_match_archive`
- `test_input_prediction_hashes_verified`
- `test_dpurge_evaluation_sets_bitwise_identical_to_primary`
- `test_race_used_the_frozen_configuration_and_dsnap_protocol`
- `test_refinement_used_the_frozen_lgbm_config_and_historical_settings`

</details>

<details><summary>FILE_MODE_NOT_STORED_BY_GIT — the 3 tests</summary>

- `test_approved_plan_is_frozen_readonly`
- `test_approved_plan_still_frozen_readonly`
- `test_approved_plan_still_readonly`

</details>

<details><summary>WORKTREE_OR_HEAD_RELATIVE_GUARD — the 2 tests</summary>

- `test_only_gitignore_modified_outside_p0`
- `test_all_stage1_writes_are_inside_approved_locations`

</details>

### P1 — `69 passed, 3 failed` of 72

| class | n | why |
|---|---:|---|
| `MISSING_GITIGNORED_PARQUET_TWIN` | 2 | both read P1's gitignored parquet twins, which no checkout contains. This was pre-declared. |
| `WORKTREE_OR_HEAD_RELATIVE_GUARD` | 1 | the guard compares paper/ against P1's baseline and fires on the TIER-A manuscript commit, which is the approved baseline of this branch. Tier B0 edited no manuscript file: paper/ at this HEAD is byte-identical to the Tier-A checkpoint. |

**Content defects: 0.**

<details><summary>MISSING_GITIGNORED_PARQUET_TWIN — the 2 tests</summary>

- `test_ci_limits_are_true_order_statistics_on_a_bounded_subset`
- `test_csv_is_a_faithful_serialisation_of_the_parquet_twin`

</details>

<details><summary>WORKTREE_OR_HEAD_RELATIVE_GUARD — the 1 tests</summary>

- `test_reports_state_that_no_manuscript_file_was_edited`

</details>


## Gate G4

Gate G4 has no artifacts inside P0. The tier-b gate was conditioned on that memo existing; it does. Impact memo: `analysis/p1_inferential_reporting/reports/MANUSCRIPT_IMPACT_MEMO.md` (sha256 `94316684d102bc83...`).

## Authorization-state caveat

protocol_p0_validation.yaml still declares scope STAGE_1_THROUGH_GATE_G1 and was never refreshed. Cite the stage-specific gate JSONs for authorization state, not it.
