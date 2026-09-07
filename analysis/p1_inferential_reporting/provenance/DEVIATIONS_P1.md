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
