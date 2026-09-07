# P1 DEVIATIONS — inferential-reporting stage

Append-only. Records every departure from the frozen
`analysis/p0_major_revision_validation/APPROVED_EXECUTION_PLAN.md` (rev. 3, sha256
`d3b7ae34…`) and every interpretive choice made in P1. Nothing in P0 is edited.

P0 is frozen at tag **`p0-major-revision-final-20260907`** = commit
`805c426e1587972a2a07dcaf60220603397c0d3e`.

---

## D-P1-1 — P1 lives in a new sibling area, not inside the P0 tree

**Plan §L** placed the P1 artifacts inside `analysis/p0_major_revision_validation/`
(`code/p0_7_inferential_extras.py`, `tables/{prb_inference,vei_significance,
smearing_sensitivity}.csv`, `reports/MANUSCRIPT_IMPACT_MEMO.md`).

**Four P0 assertions forbid exactly those paths**, all scoped to the P0 directory:
`tests/test_g2_assertions.py:282,305`, `tests/test_g3_assertions.py:310`,
`tests/test_stage3_assertions.py:319-322`; and
`test_g2_assertions.py::test_no_forbidden_jobs_ever_submitted` forbids the strings
`inferential_extras` and `smearing` in `logs/submitted_jobs.txt`. In addition
`protocol_p0_validation.yaml:26` lists PRB inference, VEI inference and smearing under
`not_authorized_in_this_run`, and `code/p0_common.py:143` `_ALLOWED_WRITE_ROOTS`
hard-refuses writes outside the P0 tree. Both of those files are hash-pinned in
`provenance/stage1_frozen_hashes.json`.

**Resolution.** P1 was created as `analysis/p1_inferential_reporting/` on branch
`p1-inferential-reporting`, cut from the tag, with its own `p1_common.py` write guard.
The P0 suite therefore remains **153 passed / 0 failed / 0 skipped verbatim** at the tag,
and no frozen P0 file was edited to make room for P1.

---

## D-P1-2 — the smearing factor is D3 row-balanced, not the plan's pooled-OOF mean

**Plan §J-bis** mandated `s` from **pooled development out-of-fold residuals** — the
concatenated seven validation blocks. That sample is the **D2** construction: it counts
20,988 unique rows twice, because the fold-6 and fold-7 validation blocks overlap.

**Resolution.** `s` is estimated with **D3 row-balanced weights** `w_ik = 1/m_i`, so each
unique development sale row carries total weight exactly one — the same rule that defines
D3 in P0 (`code/p0_5_beta_coordinates.py:30-43`). The naive duplicate-weighted factor is
still computed and reported as `s_naive_pooled` for transparency but is **never applied**.
Measured effect: max `|s − s_naive_pooled| = 3.936e-03` over 43 realizations.

Authorised explicitly by the user instruction of 2026-09-07.

---

## D-P1-3 — Duan smearing sign

The canonical residual convention in this repository is `e = y_pred_log − y_true_log`
(`utils/motivation_utils.py:1554`, verified by inspection). Duan's factor therefore uses
the **opposite-sign** log error `u = y_true_log − y_pred_log = −e`:

```
s = sum_ik( w_ik * exp(u_ik) ) / sum_ik( w_ik )
```

`exp(e)` is wrong and must never appear. Frozen in
`configs/smearing_estimator_frozen.json` **before** any held-out or 2025 output was read.

A symmetric error distribution cannot detect a sign flip (`E[exp(u)] = E[exp(−u)]`), so
`tests/test_smearing_sign.py` pairs the lognormal magnitude benchmark with an exact
asymmetric two-point case. A mutation check (estimator flipped to `exp(−u)`) confirms
three of the five tests fail, while the lognormal test passes — documenting precisely why
the asymmetric case is required.

---

## D-P1-4 — `RMSE_log` invariance statement corrected

An earlier draft asserted that both `Beta_log` and `RMSE_log` are invariant to a global
multiplicative level factor. That is wrong for `RMSE_log`.

- `Beta_log` **is** invariant: `e → e + log s`, and `Cov(e + k, c_y) = Cov(e, c_y)`
  because `c_y` is centered. Confirmed empirically (ratio exactly `1.000000000`).
- `RMSE_log` is **not** mathematically invariant to adding a constant. It is simply **not
  recomputed**: Duan smearing is a post-exponentiation price-scale retransformation
  sensitivity and does not alter the canonical log prediction.

---

## D-P1-5 — ED2 Step-2 proxy: the printed formula is not used

ED2 Appendix E Step 2 (page 78) **prints** `Proxy = (0.50*SP) + (AV/Median Ratio)`,
omitting the `0.50` on the AV term, while the prose one line above says the proxy "gives
equal weight" to both. Taken literally the printed formula roughly doubles the AV
contribution.

**Resolution.** The equal-weight form `0.50*SP + 0.50*(AV/Median)` is implemented. It is
(a) the 2013 Standard App. D p.56 formula, (b) what ED2's own Step-2 prose specifies, and
(c) what the frozen executed code `utils/motivation_utils.vei` implements — so it is
required for the P1 Step-5 value to reconcile with the frozen `VEI__*` artifacts. The
manuscript already discloses this draft inconsistency at `paper_v17_option1.tex:565`.
Corresponds to audit item P0-11.

---

## D-P1-6 — ED2 E.3 percentile-rank method

ED2 §E.3 documents the NIST/Hyndman-Fan **R6** and **R7** percentile-rank methods. The
frozen executed code assigns percentile groups with `numpy.array_split` over the
proxy-sorted index (contiguous near-equal-count groups, stable mergesort), which is
neither R6 nor R7. It is used here because P1 Step 5 must reconcile with the frozen
`VEI__*` values. Where `n` is divisible by 10 (held-out `n = 38,290` → ten groups of
3,829) all three coincide; where it is not (2025 forward `n = 26,641` → one group of
2,665 and nine of 2,664) group boundaries may shift by at most one observation.

---

## D-P1-7 — ED2 procedure not applied to `pooled_oof`

ED2 App. D.2 is a **rank-based order-statistic** confidence interval on a sample of
distinct observations. It has no row-balanced or weighted analogue, and the pooled-OOF
sample contains 20,988 unique rows twice. Rather than invent a weighted variant the draft
does not define, `pooled_oof` is carried as an explicit not-applicable row with the reason
recorded in `tables/vei_significance.csv`. **No weighted variant was invented.**

---

## D-P1-8 — resolved interpretations of the ED2 text

Recorded in full, with justification, in `configs/ed2_vei_procedure.json`:

- **ED2-I-1** — for even `n`, App. D.2's "count up and down the array from the median"
  does not by itself fix the anchor ranks, and the document contains no worked
  large-sample example (the only worked median-CI example is the small-sample D.4 case,
  `n = 17`, page 71). Counting outward from the two central order statistics is used;
  it reproduces the standard normal-approximation interval at `n = 100` → ranks (40, 61)
  and `n = 101` → ranks (41, 61), both verified.
- **ED2-I-2** — Step 7's "PG with the highest/lowest median" ranges over the two groups
  carried forward from Step 6 (first and last). ED2 §E.2 confirms this by offering
  other-group comparisons only as optional Further Analysis, with a multiple-comparisons
  warning.
- **ED2-I-3** — ED2 specifies `> 10%` for reject and `< 10%` for fail-to-reject; exactly
  10% is unspecified. Implemented as reject iff strictly `> 10.0`, with
  `ed2_boundary_exact` flagged so the edge case is visible rather than silent.

---

## R-P1-1 — reporting-layer repair (no scientific recomputation)

**What happened.** The first execution of `code/p1_2_prb_inference.py` computed and wrote
`tables/prb_inference.csv` (480 rows) successfully, then raised
`TypeError: '<' not supported between instances of 'NoneType' and 'str'` while serialising
`tables/prb_inference_summary.json`. Cause: `json.dumps(..., sort_keys=True)` on a
`class_counts` dict whose keys mixed `None` (the 40 NOT_ATTAINED rows, whose
`iaao_2013_class` is null by design) with strings.

**Scope.** Purely a **reporting-layer** failure in the summary serialiser. It occurred
strictly **after** the scientific table had been written, and the scientific table was
unaffected.

**Repair.** The `None` key is mapped to the literal `"NOT_ATTAINED"`, and a
`--summary-only` entry point was added that regenerates the summary **from the already
written `prb_inference.csv`**. **No PRB inference was recomputed**; no prediction artifact
was re-read; `prb_inference.csv` was not rewritten.

**Second, related reporting correction.** The first `--summary-only` run reported
`max |OLS − canonical prb()| = 2.723e-03`, which conflated two different quantities: the
reconciliation check (unweighted rows, which must be ~0) and the **D3 row-balanced
pooled-OOF reweighting effect** (a real, interpretable sensitivity). The summary now
reports them as two separate fields:

- `reconciliation_unweighted_rows.max_abs = 1.1790152187884928e-12`
- `d3_pooled_oof_reweighting_effect.max_abs = 0.0027225657356132`

Also purely a reporting-layer correction: no value in `prb_inference.csv` changed.

---

## Performance changes (no scientific effect)

- **P-1** — the cluster-robust variance in `p1_2_prb_inference.py` originally accumulated
  per-cluster score sums in a Python loop over ~130k clusters and did not complete. It was
  vectorised with `np.add.reduceat`, verified equal to the loop to `1.42e-14` on a
  synthetic case. Identical arithmetic.
- **P-2** — `p1_common.preflight_block()` and the Stage-1.5 zero-reference loader are
  memoised. `git_state()` runs `git status --porcelain` over a repository with a very large
  untracked `output/` tree on a network filesystem and costs minutes; it is invariant
  within a process. Hash verification in the zero-reference loader still runs once per
  (cell, block) pair.
- **P-3** — `pypdf 5.9.0` was installed into an **isolated scratch directory**
  (`pip install --no-deps --target=...`) purely to read the ED2 PDF. `fairness_env` is
  unmodified: numpy 1.26.4, pandas 2.3.1, scipy 1.13.1, scikit-learn 1.6.1, pyarrow
  14.0.1, lightgbm 4.6.0, dcor 0.6 all unchanged, and `pypdf` is not importable from it.

---

## C-P1-1 — correction: the P0 suite is 153/153 AT THE TAG, not on the P1 branch

The plan justifying D-P1-1 claimed a sibling P1 area would keep the P0 suite at
"153/153 verbatim, forever". That was **wrong**. Two P0 guards react to work
outside the P0 area regardless of which directory it lives in:

1. `test_p0_assertions.py:222::test_all_stage1_writes_are_inside_approved_locations`
   fails on any **dirty working-tree** entry outside the two P0 roots — including an
   untracked `analysis/p1_inferential_reporting/`. Committing resolves this one.
2. `test_g2_assertions.py:325::test_only_gitignore_modified_outside_p0` is
   **HEAD-relative** (`git diff --name-only 2732e653~1 HEAD`) and asserts the only
   file changed outside the P0 area is `.gitignore`. **Any** additive commit
   anywhere breaks it. A sibling directory does not help; committing makes it fail.

Measured: at the tag, files outside P0 since `2732e653~1` = `['.gitignore']`
→ **153 passed, 0 failed**. On `p1-inferential-reporting` @ `ccff55f0` → 25 files
→ **152 passed, 1 failed**.

**This is not a P0 regression.** P0 content is provably unchanged: `git diff`
against the tag over `analysis/p0_major_revision_validation/` is empty, the
Stage-3B index is 86/86 byte-exact, and every protected path is clean. The
certification the tag carries is *the suite at the tag*, and that is unaffected.

D-P1-1 still stands on its remaining grounds: a sibling area avoids editing any
frozen, hash-pinned P0 file (`p0_common.py`, `protocol_p0_validation.yaml`, the
four test modules) in order to make room for P1. Only the "153/153 forever on any
branch" claim is withdrawn.

---

## D-P1-9 — ED2 §D.3 strengthens D-P1-7: the draft weights the MEAN, never the median

D-P1-7 recorded that App. D.2 has no row-balanced analogue. Reading the literal document
at finalization makes the argument stronger than "no analogue is stated":

- **§D.3, immediately after D.2, is "Weighted Mean Confidence Interval"** and constructs an
  **effective sample size** from the weights. The draft is therefore demonstrably capable of
  specifying weighted inference — and does so **only for the mean**.
- The string **"weighted median" appears nowhere** in the 103-page document, and
  **"effective sample size" appears only inside that weighted-mean section**.
- Every App. E step is defined on a count of **distinct** sales. §E.3 states "*N is the
  number of sales ratios in the sample*" for both R6 and R7; Step 3's group table is keyed on
  "Number of Observations". No weight appears anywhere in App. E.
- The draft never addresses duplicate observations, repeat sales, or pooled samples.

Applying D.2 to the D3 pooled-OOF sample would require choosing an `n` (151,153 appearances
double-counts 20,988 rows and is not IID; the weight total 130,165 does not index the
151,153-element array) and then inverting a weighted quantile to integer ranks. Every
resolution invents a procedure the draft declines to define. **pooled_oof therefore stays
`NOT_APPLICABLE_FOR_ED2_INFERENCE`. No weighted variant was invented**, and
`tests/test_p1_vei_assertions.py` scans the P1 code for `weighted_median`, `weighted_rank`,
`weighted_quantile`, `effective_n` and similar and asserts none exists.

> **Naming caution.** This project's **D3** row-balanced construction and the draft's
> **§D.3** weighted-mean CI are unrelated. They must never be conflated in prose.

The same refusal is enforced in the smearing apply stage: its ED2 verdict-stability table
carries the 43 `pooled_oof` cells as `NOT_APPLICABLE_FOR_ED2_INFERENCE`, importing the reason
string from `p1_3_vei_ed2_inference.POOLED_OOF_REASON` so there is exactly one source of
truth (asserted). The VEI **point estimate** is still audited there, being a descriptive
statistic rather than the CI-based test.

---

## D-P1-10 — smearing apply: two-tier coverage, with `Delta_NL` on a pre-declared subset

`Delta_NL` costs ~34 s per call (five cross-fitted OOF spline fits). Auditing it on all
430 cells in both arms would have cost ~8 h of wall clock for a quantity that is exactly
invariant by construction.

**Resolution.** Two tiers, both declared in the artifact:

- **Tier 1, full coverage** — all 43 realizations × 10 evaluation blocks = **430 cells**,
  both arms, for the 15 remaining canonical metrics (level, uniformity, vertical-equity,
  mechanism and price-scale accuracy). 6,450 comparisons, 5,160 carrying an assertion.
- **Tier 2, `Delta_NL`** — a **pre-declared** subset: the two included reference cells (C, A)
  plus the realization with the **largest** `s` (the strongest stress on invariance), across
  all ten blocks. 30 cells, max relative deviation `1.81e-15`.

`Delta_NL`'s invariance is also exact analytically: `e → e + log s`, both the affine and the
cubic-spline head carry an intercept, and `Var(e)` is translation invariant, so the
cross-fitted OOF residuals are unchanged.

The Tier-1 fast path is **not a re-implementation**: every value is produced by the same
canonical function `utils.motivation_utils.compute_taxation_metrics` calls. On the Tier-2
subset it was compared against the frozen P0 suite
`p0_4_centered_spread.metrics_from`: **450 comparisons, max relative difference `0.0`**.

`code/p1_4_smearing_sensitivity.py` changed (its `--mode apply` branch was a `SystemExit(0)`
stub and is now implemented), so its sha256 differs from the checkpoint index. The **frozen
scientific artifact** `configs/smearing_estimator_frozen.json` is byte-identical and its hash
pin still validates; `--mode freeze` was **not** rerun.

---

## D-P1-11 — the committed CSVs lose 1 ULP; the parquet twins are bit-exact

Discovered while asserting that the ED2 CI limits are true order statistics. `pandas.to_csv`
does not always emit a fully round-trippable float64.

Measured over `vei_significance.csv` vs its parquet twin: **worst 16 ULP, worst absolute
deviation `3.553e-15`** (on `VEI_step5`, where a scaled difference amplifies the relative
rounding); most columns are within 1 ULP. Integer-valued and string columns round-trip
exactly.

**Consequence, now asserted rather than assumed:** bitwise claims must reference the parquet
twin, not the committed CSV. `tests/test_p1_vei_assertions.py` performs the order-statistic
exactness check bitwise on the parquet, compares the CSV to within 2 ULP, and
`test_csv_is_a_faithful_serialisation_of_the_parquet_twin` pins the whole-table bound.
Nothing scientific is affected — `3.55e-15` on a VEI expressed in percent is ~1e-16
relative — but no downstream document should claim CSV bit-exactness.

Note that summary values recorded as exactly `0.0` (for instance
`max_abs_step5_minus_canonical_vei`) are properties of the **in-memory** computation, written
to JSON, and are unaffected. Table-internal differences are also unaffected, because both
operands are serialised from the same in-memory float and round identically.

---

## D-P1-12 — reproducing 153/153 at the tag requires state git does not carry

The P0 suite was verified **at the immutable tag in a separate detached worktree**. A fresh
`git worktree add` carries only tracked git *content*, so three classes of state the suite
legitimately asserts had to be re-attached from the main working tree, where P0 actually ran.
**No P0 test was modified, skipped or weakened, and no file content was altered.**

1. **Gitignored derived artifacts** — `output/paper_v6_preselection_994`,
   `output/paper_v12_lower_rho_extension_994_v2`, `output/p0_major_revision_validation`.
2. **Gitignored source data** — `data/{ATTOM,berry_cmf,CCAO,CensusData,dewey-downloads}`,
   attached as *real directories with symlinked children*. A bare symlink is not a directory,
   so the existing `data/<X>/` gitignore patterns would not match it, and the resulting
   untracked path tripped `test_all_stage1_writes_are_inside_approved_locations` on the first
   attempt.
3. **File mode and mtimes, which git does not record.**
   `APPROVED_EXECUTION_PLAN.md` is `0444` in the main tree and `0664` after checkout, and
   three tests assert it is read-only.
   More substantively,
   `test_p0_assertions.py::test_source_equivalence_precedes_reproduction_interpretation`
   asserts a **relative mtime ordering** (`source_equivalence_verdict.json` no newer than
   `frozen_artifact_reproduction.csv` + 1 s). `git worktree add` stamps every file with the
   checkout time, so **this assertion is not reproducible from committed content by any
   checkout.** mtimes were reconstituted for the 309 tracked P0 files from the main working
   tree, the authoritative record of the P0 execution order.

Both measurements are reported in `provenance/p0_suite_at_tag.json`:

| checkout | result |
|---|---|
| content only | **152 passed, 1 failed** — the mtime-ordering test, sole failure |
| + mode and mtimes reconstituted | **153 passed, 0 failed, 0 skipped** |

This is a genuine reproducibility limit of the P0 suite and is disclosed as such: one of its
153 assertions is a property of the original run's filesystem metadata rather than of the
committed content.

---

## C-P1-2 — correction: the checkpoint's hash index was stale against the checkpoint

At the previous checkpoint, `provenance/p1_artifact_hashes.json` recorded
`P1_CHECKPOINT.md` at 12,207 bytes / `3153355a…`, while the committed file was 12,427 bytes
/ `b27ef763…`. Cause: the index hashed the checkpoint, and the checkpoint was then edited to
add the pointer *to* the index. 23 of 24 entries were byte-exact; only the self-referential
one was stale.

**Resolution.** The authoritative manifest is now generated by
`code/p1_9_freeze_evidence.py` as the **last** step before commit, after every file it
covers is final. It excludes exactly one path — **itself**, since a hash index cannot contain
its own hash — and records that exclusion and its reason explicitly in
`self_excluded` / `self_exclusion_reason`.

---

## P-4 — `pypdf` reinstalled into an isolated scratch target

The isolated `pypdf` install from the earlier session was gone (node-local `/tmp`), and the
leftover trees found there were incomplete 6.x copies with no importable package. `pypdf
5.9.0` — the version recorded in P-3 — was reinstalled with
`pip install --target <scratch>` purely to re-read the ED2 PDF and verify App. D.2 and
App. E verbatim at finalization.

`fairness_env` is unmodified and `pypdf` is still not importable from it: numpy 1.26.4,
pandas 2.3.1, scipy 1.13.1, scikit-learn 1.6.1, pyarrow 14.0.1, lightgbm 4.6.0, dcor 0.6,
Python 3.9.19, all unchanged. The cached PDF re-hashed to
`e950e00d0c3684dd067734d401e5278dcf659f471fcfb0d6db65c7584f7c56b8` (2,013,844 bytes), and
the extracted document confirms 103 pages, D.2 on printed page 67, App. E on 78–81, and the
title "EXPOSURE DRAFT – STANDARD ON RATIO STUDIES - MAY 2026".

---

## C-P1-3 — correction: two P0 hash indexes are NOT byte-exact, and never were

The first run of `code/p1_9_freeze_evidence.py` refused to freeze, reporting 8 mismatches:
`stage1_frozen_hashes.json` 48/49 and `output_artifact_hashes.json` 118/125. Earlier P1
documents (including this file's D-P1-1 and the checkpoint) claimed the frozen P0 hash
indexes were byte-exact. **That claim was too strong.**

**Investigated, not assumed.** The identical check was run inside the detached tag worktree:

- the mismatch set at the tag is **exactly the same 8 paths** as on the P1 branch;
- all 8 files are **byte-identical between the two trees**;
- `git diff` against the tag over the P0 tree is **empty**.

**Cause.** The two indexes are **early-stage snapshots that P0's own later gates legitimately
superseded**. `tests/run_all_tests.py`, `test_g2_assertions.py` and `test_g3_assertions.py`
were edited in later P0 stages; `POSTFLIGHT.json`, `slurm_graph.json` and
`large_local_artifacts.json` are end-of-run summaries written after the index; and
`output_artifact_hashes.json` **lists itself**, a self-reference that can never match — the
same failure class as C-P1-2. `tables/regeneration_triggers.csv` is the single Stage-1 entry.

**Resolution.** The gate criterion was wrong, not the evidence. Requiring zero mismatches in
those two indexes would demand a property that was never true. The freeze script now
enforces: (1) empty `git diff` vs the tag over the P0 tree; (2) the **final** gate index
(Stage-3B / G5b) fully byte-exact — it is, 28/28; (3) **no index acquired a mismatch the tag
did not already have**, checked by re-hashing the tag's own blobs via `git show`, so no extra
checkout is needed; (4) no index entry missing; (5) protected paths clean; (6) only the P1
directory and `.gitignore` changed since the tag. All six hold, and `P0_UNCHANGED` is
recorded as `true` with the full evidence in
`provenance/p1_artifact_hashes.json → p0_immutability`.

This strengthens rather than weakens the immutability claim: it now detects any P1-induced
change to a P0 file while being honest about P0's documented internal staleness. The
withdrawn wording is "every frozen P0 hash index is byte-exact"; what holds is "the final
gate index is byte-exact and no index acquired a new mismatch".
