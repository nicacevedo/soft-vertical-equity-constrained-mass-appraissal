# State

## Snapshot

This capsule was authored against, and is grounded only in, snapshot:

```
branch  research-os/r0-dogfood
commit  993db93  [Tier B2.5] Close temporal robustness and portability interpretation
tree    clean at authoring time
```

The sibling worktree on `paper-major-revision-write` is out of bounds for this capsule and
was not read. No later or uncommitted work informs any object here.

## Current manuscript state

`paper/paper_v17_option1.tex` is the live manuscript. It is mid-revision under the Tier-B
writing programme. At this snapshot `paper/paper_analysis/tier_b_validation/runs/B2.5.json`
reports `ok = true`, `n_unexpected_failures = 0`, `unsupported_tokens = 0`, with **25
declared expected failures** still open (21 owed to stage B2.6, 2 to B3.1, 1 to B4.2, 1 to
B4.3) and **4 active unsupported math-mode claims** (`MATH-001` … `MATH-004`).

`unsupported_tokens = 0` is a statement about the text-mode token population only. The
coverage audit masks math environments, so the four registered math-mode claims are outside
that population entirely and are **not** resolved by it.

## Human framing decision recorded for this capsule

Results section 5.9 (`paper_v17_option1.tex:2603`) is the live scientific state:

> the three families are not interchangeable at matched first-order correction, and no
> family is better on every diagnostic ... a diagnostic distinction between mechanisms,
> not a ranking, and no penalty strength is selected anywhere.

Two Discussion/Limitations statements in the same file are **stale** relative to the
Tier-B2 evidence and are known to be so:

1. `:2816` — "the study omits the closest post-processing comparator --- the centered
   spread map implied by the Direct fixed-space result". The comparator **was** run: it is
   reported in sections 5.8 and 5.9 and in `EVI-0005`, `EVI-0006`, `EVI-0007`. The
   *conclusion* attached to that sentence ("so it cannot claim that training-time
   regularization is necessary or superior") remains correct and is in fact strengthened by
   the comparator result; only its premise is false.
2. `:2733` and `:2816` — "native-to-penalized contrasts remain partly confounded **until**
   the initialization-aligned rho=0 parity rerun is propagated". The rerun was executed and
   `analysis/final_manuscript_evidence/spec/todo_closure.yaml` entry 16 records it as
   `CLOSED_WITH_CONTRARY_FINDING`: the conditional resolved in the negative, pinning does
   not remove the divergence, and the table is not regenerated. The adopted resolution is
   the two-reference convention (`DEC-0001`), not a future parity restoration.

Both sentences date to commit `a1b4f369` (2026-09-05), before `d06966f` (Tier B2.4,
2026-09-07) added the centered-spread and matched-beta comparisons. Section 6 is scheduled
for rewrite at stage B3.1, but neither sentence is individually registered as an expected
failure, so neither is enforced in either direction.

**No manuscript file was edited to record this.** The inconsistency is recorded here, in
this capsule, only.

## Open, and deliberately unclaimed

- The Direct curvature audit returns `INDETERMINATE` (`EVI-0008`). No closed curvature
  explanation is claimed, and its negation is not claimed either.
- Temporal portability of any raw-`rho` operating interval is **not** claimed. Gate G5a
  fired `ALERT`; refinement returned G5b `NOT_CONFIRMED` with "the primary design stands"
  (`EVI-0009`).
- Candidate-region, transition-span and span-regret numbers are unresolvable from the
  frozen evidence and back no object in this capsule.
- The Surrogate `dCor` non-monotonicity **as the manuscript currently prints it** rests on
  `MATH-004`, an active unsupported math-mode claim. `EVI-0009` therefore sources the
  rebound structure from the frozen trigger table instead, and no object asserts the
  manuscript's printed endpoints.

## Provenance disclosures that the R0 schema cannot carry in a digest-bound field

Recorded here because `notes` is excluded from every semantic projection, so a disclosure
placed only there would not be bound by any future Review.

1. **EXP-0001 ran from dirty working trees.** Its three provenance commits are
   `508dc1c2` (994-tree baseline/config and v6 experiment spec), `2aa0346a` (lower-rho
   extension spec) and `d3ef45f2` (rho=0 split audit, recalibration path, Delta_NL). All
   three trees were dirty; for `d3ef45f2` not even a diff hash was stored. The project
   classifies the residual as `F-DIRTY -- unreconstructable dirty-state uncertainty`, a
   disclosure item and not a regeneration trigger
   (`analysis/p0_major_revision_validation/provenance/DIRTY_STATE_LIMITATION.md`).
   `Experiment.provenance.git_commit` holds a single commit and cannot express any of this.
2. **EXP-0001's frozen configuration is not in Git.** `provenance.config` names
   `output/paper_v6_preselection_994/lgbm_config.json`, the correct repository-relative
   path, but `output/` is gitignored (`.gitignore:17`) and the file is absent from this
   worktree. Its content identity is `lgbm_params_sha256 =
   8f0f2acd83118de782604b5ca7143acfbd2af3fd186ea9376588f9bcf560585b`,
   `config_id = 407d47775760c14d`.
3. **EXP-0001's preregistration is retrospectively reconstructed.** Its `predictions` and
   `decision_rule` did not exist when the runs executed. Each such field is marked
   `[RECONSTRUCTED]` inline so the marker is inside the semantic digest. EXP-0002's
   equivalents are marked `[PRESPECIFIED]` and are genuinely frozen ahead of execution.
4. **EXP-0002 was a multi-stage programme.** `protocol_p0_validation.yaml` still declares
   `authorized_scope.this_run: STAGE_1_THROUGH_GATE_G1` and was never refreshed after
   Stages 2 and 3 ran under `configs/posthoc_comparator_convention.yaml` ("FROZEN AT GATE
   G3"). Authorization state must be read from the stage-specific gate JSONs, not from the
   protocol file.

## Not done, deliberately

No Review exists. `CLAIM-0001` stands at `evidence_linked` and must not be moved to
`accepted` until a bounded Research OS integrity patch addresses the demonstrated
Experiment-binding gap: an Evidence object of `kind: experiment` binds an Experiment **ID**,
and its digest does not cover that Experiment's provenance, predictions or metrics. Eight of
the nine Evidence objects here are experiment-kind, so a human Review of `CLAIM-0001` today
would bind the Evidence digests and none of the experimental content they summarise.
