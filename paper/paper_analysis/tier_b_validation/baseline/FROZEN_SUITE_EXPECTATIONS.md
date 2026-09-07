# The frozen suites are informational, and here is exactly why

Recorded at **B1.0**, before any manuscript edit. Machine-readable twin:
`frozen_suites_pre_edit_B1.0.json`; full transcripts in
`{tier_b0,p0,p1}_suite_pre_edit_B1.0.log`.

**Modifying any frozen P0 / P1 / Tier-B0 test is prohibited.** These suites are
run to record what the environment reports, never as a writing-stage pass/fail
criterion, and never cited as certification of this pass. What certifies the
science is what the three tags carry: **P0 153/153** and **P1 72/72** at
`p0-major-revision-final-20260907` and
`p1-inferential-reporting-final-20260907`, with each stage subtree
byte-identical to its tag. What certifies the writing is `validate.py` plus the
cumulative write-scope and subtree-immutability checks in `tb_scope.py`.

## Why they cannot be the gate

They contain paper-immutability and working-tree/HEAD-relative guards that fail
**by design** the moment anything is written under `paper/` -- which is the
entire job of the Tier-B writing pass. This is not a hypothesis: creating this
validator directory alone, with the manuscript still byte-identical to the
Tier-A baseline, moved

| suite | clean tree, before this directory existed | with `paper/paper_analysis/tier_b_validation/` present |
|---|---|---|
| Tier B0 | **110 / 110** | 107 / 110 |
| P0 | **141 / 153** | 135 / 153 |
| P1 | not measured on a clean tree this session | 65 / 72 |

Every additional failure is a scope guard reporting the untracked directory.
**Content defects: 0** -- proven independently of the suites, by the three tag
diffs being empty:

```
git diff --stat p0-major-revision-final-20260907        HEAD -- analysis/p0_major_revision_validation
git diff --stat p1-inferential-reporting-final-20260907 HEAD -- analysis/p1_inferential_reporting
git diff --stat tier-b0-final-20260907                  HEAD -- analysis/final_manuscript_evidence
```

## Classification of every failure in the recorded run

### Tier B0 -- 107 passed, 3 failed

| class | n | tests |
|---|---:|---|
| `WORKTREE_SCOPE_GUARD` | 3 | `test_frozen_stages_and_paper_are_untouched`, `test_git_status_shows_nothing_outside_the_tier_b0_area`, `test_only_the_gitignore_is_modified_outside_the_tier_b0_area` |

All three assert that nothing outside the Tier-B0 area has changed. The first
compares `paper/` against the Tier-A commit, so it fails permanently from B1.1
onward. `code/b0_common.py::TEX_SHA256` pins the baseline manuscript hash for
the same reason; it is quoted in `tb_common.BASELINE_TEX_SHA256` so the
divergence stays deliberate and visible rather than accidental.

### P0 -- 135 passed, 18 failed

| class | n | tests |
|---|---:|---|
| `WORKTREE_OR_HEAD_RELATIVE_GUARD` | 7 | `test_all_stage1_writes_are_inside_approved_locations`, `test_frozen_analysis_directories_unmodified`, `test_no_manuscript_file_changed` (defined in two modules), `test_no_manuscript_or_frozen_output_modified`, `test_no_manuscript_or_regeneration_work_from_the_g2_era`, `test_only_gitignore_modified_outside_p0` |
| `FILE_MODE_NOT_STORED_BY_GIT` | 3 | `test_approved_plan_is_frozen_readonly`, `test_approved_plan_still_frozen_readonly`, `test_approved_plan_still_readonly` |
| `MISSING_GITIGNORED_ARTIFACT_OR_DEPENDENCY` | 8 | `test_fold_index_hashes_verified_live`, `test_fold_index_hashes_match_archive`, `test_frozen_lgbm_params_hash`, `test_frozen_rho_grid_is_83_points`, `test_input_prediction_hashes_verified`, `test_dpurge_evaluation_sets_bitwise_identical_to_primary`, `test_race_used_the_frozen_configuration_and_dsnap_protocol`, `test_refinement_used_the_frozen_lgbm_config_and_historical_settings` |

The scope guards are doing exactly their job: registering that additive areas
exist. Git records only the exec bit, so no checkout can reproduce mode 0444.
The eight artifact tests read `output/paper_v6_preselection_994`,
`output/paper_v12_lower_rho_extension_994_v2` or
`data/CCAO/2025/training_data.parquet`, all gitignored by design and in no
checkout; two of them fail earlier still, on `sklearn` and a parquet engine
being absent from this interpreter. Tier B0 quotes those hashes from the frozen
index rather than reading the files, and so does this validator.

**One recorded delta, investigated rather than re-explained.** The Tier-B0
certification records the P0 diagnostic rerun in this worktree as **140 / 153**
(13 failures). Measured on a clean tree this session it is **141 / 153** (12).
The difference is exactly one HEAD-relative guard,
`test_all_stage1_writes_are_inside_approved_locations`, which now passes on a
clean tree because the commit graph moved on (the certification was written at
the B0 integration HEAD; B0.1 added a commit). It is in the
`WORKTREE_OR_HEAD_RELATIVE_GUARD` class either way, and content defects remain
0 because the P0 subtree is byte-identical to its tag. Nothing about P0's
science changed.

### P1 -- 65 passed, 7 failed

| class | n | tests |
|---|---:|---|
| `PAPER_WRITE_GUARD` | 4 | `test_no_protected_path_written` (three suites; `_PROTECTED` includes `REPO/"paper"`), `test_reports_state_that_no_manuscript_file_was_edited` |
| `MISSING_GITIGNORED_PARQUET_TWIN` | 2 | `test_ci_limits_are_true_order_statistics_on_a_bounded_subset`, `test_csv_is_a_faithful_serialisation_of_the_parquet_twin` |
| `MISSING_DEPENDENCY` | 1 | `test_headline_numbers_are_reproducible_from_the_artifacts` (no `lightgbm` in this interpreter) |

`test_reports_state_that_no_manuscript_file_was_edited` stays true only for as
long as nothing under `paper/` is touched, which is precisely what B1.1 onward
does. The parquet twins are gitignored, which is also why any *bitwise*
serialisation claim must reference the parquet rather than the committed CSV:
P1 records that the CSVs round-trip float64 to a worst 16 ULP / `3.553e-15`.

## What is expected from B1.1 onward

Once the manuscript is edited, the suites will additionally and permanently
report `test_frozen_stages_and_paper_are_untouched`,
`test_no_manuscript_file_changed`,
`test_no_manuscript_or_regeneration_work_from_the_g2_era`,
`test_reports_state_that_no_manuscript_file_was_edited`, and any assertion
resting on `TEX_SHA256`. That is the designed behaviour of a
paper-immutability guard during a paper rewrite. It is documented here and left
alone.
