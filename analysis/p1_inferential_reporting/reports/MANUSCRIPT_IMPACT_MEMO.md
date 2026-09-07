# MANUSCRIPT IMPACT MEMO — P1 inferential reporting

**Status:** FINAL. **No manuscript file has been edited.** This memo states what the P1
evidence permits, requires, and forbids. Every number traces to
`provenance/p1_headline_numbers.json`; the full argument is in
`reports/MANUSCRIPT_EVIDENCE_PACKAGE.md`.

Target file: `paper/paper_v17_option1.tex`.

---

## 1. Bottom line

P1 changes **no** substantive empirical claim in the manuscript. It closes one open
`\todo`, supplies the inferential layer the manuscript itself flags as missing, and answers
a robustness check the manuscript explicitly asks for. One claim can be **strengthened**,
one must be **narrowed**, and two pieces of **new** disclosure are now available.

| item | effect on the manuscript |
|---|---|
| dCor estimator | **closes a `\todo`.** Text to add is fully determined. |
| PRB inference | **strengthens** — adds CIs and a four-state rule; no reclassification of any headline. |
| VEI / ED2 inference | **strengthens, with one narrowing.** A point estimate outside ±10% is not a significant finding in 11 of 63 standards-facing escalations. |
| Duan smearing | **answers a check the manuscript requests**, and licenses a clean invariance statement. |
| pooled-OOF ED2 | **new limitation to disclose** — the draft's procedure cannot be applied there. |
| P0 suite reproducibility | **new limitation to disclose** — one of 153 assertions is not reproducible from committed content. |

---

## 2. Closes the open `\todo` (dCor)

`paper/paper_v17_option1.tex:3204-3206` asks to "confirm and archive the exact finite-sample
distance-correlation implementation … including whether the reported estimator is the
standard biased/V-statistic or an unbiased/U-centered variant".

**Answer, now archived.** `dcor 0.6`;
`dcor.distance_correlation(e, y_true_log, method="auto")`; `exponent = 1`;
`compile_mode = AUTO`; `bias_corrected = False`; full evaluation sample, no subsampling;
`e = y_pred_log − y_true_log`, `y = log P`. This is the **standard biased / V-statistic
(double-centered)** estimator. Pinned numerically on an n = 400 probe: it matches a
hand-rolled double-centered V-statistic to `1.33e-15` and differs from the U-centered
(unbiased) statistic by `7.52e-03`, so the two are distinguishable and the executed one is
unambiguous. `method="auto"` is an exact algorithm choice, not an estimator choice.

The `\todo` can be replaced by that specification. The definition at
`eq:dcor_diagnostic` (line 1163) needs **no** change: it defines the population object,
and the estimator note is a separate, additive sentence.

**Also available if wanted:** all reported path dCor values come from one function, so the
manuscript may state single-estimator consistency. The other two `dcor` call sites in the
repository feed no manuscript table, and one computes a different quantity — dCor(ratio,
log P), subsampled. Worth stating precisely, because "we report distance correlation" is
otherwise ambiguous across three call sites.

---

## 3. PRB — strengthens, reclassifies nothing

480 rows, 440 attained, 95% CIs. The classification rule is now explicit and conservative:
**the entire CI must lie outside a band before that band is deemed exceeded**; a CI that
merely crosses a threshold is `overlaps_pm005`, never evidence of exceeding it.

Standards-facing counts: 33 `within_pm005`, 8 `overlaps_pm005`, 44
`outside_pm005_but_not_pm010`, 3 `outside_pm010`, 8 `NOT_ATTAINED`.

**Nothing in the manuscript needs to change to accommodate this**, because the P1
regression reproduces the canonical `prb()` on identical observations to `1.18e-12` and the
frozen artifacts reconcile to `7.63e-16` over 234 comparisons. The point estimates the
manuscript already reports are unchanged; P1 only adds uncertainty around them.

Two things become *sayable* that were not before:

- **`±0.05`/`±0.10` statements can be made inferentially rather than by point estimate.**
  In particular, the 8 `overlaps_pm005` standards-facing cells should not be described as
  exceeding `±0.05`.
- **The D3 reweighting effect is a sensitivity, not an error:** max `2.72e-03` on pooled
  OOF, the measured consequence of one-sale-one-vote versus letting 20,988 overlapping
  fold-6/fold-7 rows count twice. If pooled-OOF PRB is quoted, this belongs beside it.

**Forbidden, and asserted absent:** no `mean ± SD/√7` over folds appears anywhere. Fold
values are per-fold descriptive quantities with their own observation-level SEs. The
manuscript must not present fold spread as an inferential interval.

---

## 4. VEI / ED2 — strengthens, with one genuine narrowing

### 4.1 What can now be said

The full App. E decision path (Steps 5, 6, 7) is computed for **396** cells — 44 attained
entries × 9 evaluation blocks — with 90% median CIs from App. D.2's **rank-based order
statistic**, not the bootstrap. Step 5 reproduces the canonical `vei()` **exactly** (max
`|Δ| = 0.0`); frozen artifacts reconcile to `7.11e-15`.

The manuscript already reports per-percentile-group median ratios with 90% CIs
(line 639) and already frames the draft as descriptive rather than a compliance standard.
P1 makes that framing *operational*: it emits the draft's own verdict wording, for every
cell, from a hash-verified copy of the document.

### 4.2 The narrowing — this is the one substantive finding

On the 88 standards-facing cells (heldout + forward_2025):

- 25 stop at Step 5 with |VEI| ≤ 10%;
- 63 escalate; of those **52 reject the null** and **11 fail to reject at Step 7**;
- **0** stop at Step 6 on CI overlap.

**Implication.** A VEI point estimate outside ±10% does **not** by itself establish
statistically significant vertical inequity under the draft's own test: in **11 of 63**
standards-facing escalations the test fails to reject. Any manuscript sentence that treats
"|VEI| > 10%" as a finding of unacceptable vertical inequity should be qualified. The
honest formulation distinguishes the point estimate from the significance outcome.

**Second, quieter implication.** No cell stopped at Step 6. With roughly 3,800 observations
per decile the first/last median CIs are narrow enough that overlap essentially never
occurs, so Step 6 is not the discriminating step at this sample size — Step 7's 10%
threshold is. If the manuscript describes the CI-overlap step as the significance test, that
is misleading at this n; it is a screen that never fires here.

VEI point estimates on standards-facing blocks span **−29.79 to +14.77**.

### 4.3 New limitation that must be disclosed: pooled-OOF is outside the draft

The ED2 procedure **cannot** be applied to the D3 row-balanced pooled-OOF construction, and
this was adjudicated against the literal text rather than assumed:

- every App. E step is defined on a count of **distinct** sales (§E.3: "*N is the number of
  sales ratios in the sample*"); no weight appears in App. E;
- App. D.2 is a **rank-based order statistic** — limits are array elements at integer ranks;
- the draft *does* define a weighted interval, but **only for the mean** (§D.3, via an
  effective sample size). **"Weighted median" appears nowhere in the 103-page document**,
  and "effective sample size" appears only in that weighted-mean section;
- the draft never addresses duplicate observations or pooled samples.

The pooled sample has 151,153 appearances over 130,165 unique rows. Applying D.2 literally
would require inventing a weighted-rank median CI. **None was invented.** pooled_oof is
carried as `NOT_APPLICABLE_FOR_ED2_INFERENCE` with the reason recorded on every row.

If the manuscript reports pooled-OOF VEI, it must be labelled a **descriptive point
estimate with no ED2 inferential status**. This costs nothing standards-facing: the
standards-facing inference rests on heldout and forward_2025, where it is fully available.

> Naming caution for drafting: the project's **D3** row-balanced construction and the
> draft's **§D.3** weighted-mean CI are unrelated. Do not let the labels collide in prose.

### 4.4 Bibliography — resolve the citation

The cached document's **title page reads "Exposure Draft May 2026"** even though the URL
filename says `Mar2026`. `IAAO2026ExposureRatio` should cite it by printed title and date
(May 2026), with the URL as given, sha256
`e950e00d0c3684dd067734d401e5278dcf659f471fcfb0d6db65c7584f7c56b8`, 2,013,844 bytes,
retrieved 2026-09-07. The existing statement at line 565 that the May-2026 document
"remains an exposure draft" and that the 2013 Standard "remains the recommended IAAO
guidance" is **confirmed** and needs no change.

The PDF is **not redistributable** — permission was not established — so replication
materials should ship the hash and URL, not the file.

### 4.5 Already disclosed, now corroborated

The Step-2 printed-formula discrepancy is already disclosed at line 565, and line 566
already records that the executed code uses the equal-weight proxy. P1 confirms both
against the document: the draft prints `Proxy = (0.50*SP) + (AV/Median Ratio)` while its own
prose one line above specifies equal weight. **No change needed** — the disclosure is
correct as written.

One deviation is *not* yet in the manuscript and should be added if VEI grouping is
described in detail: percentile groups come from `numpy.array_split` over the proxy-sorted
index, which is **neither R6 nor R7** of the draft's §E.3. It is required for Step 5 to
reconcile with the frozen values. Where n divides by 10 (heldout, 38,290 → ten groups of
3,829) all three coincide; where it does not (forward_2025, 26,641) boundaries shift by at
most one observation.

---

## 5. Duan smearing — answers a check the manuscript asks for

Line 3057 already states that "*direct exponentiation of log predictions is an operational
retransformation, not a smearing or conditional-mean correction*" and that "*the reported
price-scale path should be checked against smearing or other retransformation choices*".

**That check has now been run**, and the result is clean and strong.

A development-only, D3 row-balanced Duan factor was frozen **before** any out-of-sample
application, then applied unchanged to all 10 evaluation blocks for all 43 realizations —
430 cells, 5,160 invariance/scale assertions, **0 failures**:

- **level metrics scale by exactly `s`** (median, mean and weighted-mean ratio; max relative
  deviation `1.26e-15`);
- **every uniformity, vertical-equity and mechanism metric is unchanged** — COD, COV, PRD,
  PRB, VEI, MKI, β_log, Cov(e, log P), dCor(e, y); max relative deviation `1.00e-11`, that
  worst case being dCor's approximate-order algorithm rather than any change in dependence;
- **Δ_NL unchanged** on a pre-declared subset including the largest-`s` realization (30
  cells, max `1.81e-15`), and exactly invariant by construction — both regression heads
  carry an intercept and `Var(e)` is translation invariant;
- **every ED2 verdict is unchanged**: 387/387 applicable cells keep an identical Step-5
  gate, Step-6 result, Step-7 outcome and verdict string.

**What the manuscript may now claim.** Duan smearing changes the valuation **level** by
`s ∈ [1.0429, 1.1427]` (median `1.0602`), i.e. **+4.3% to +14.3%**, and leaves **every
reported uniformity, vertical-equity and residual-dependence diagnostic numerically
unchanged**. The equity conclusions are therefore invariant to this retransformation
choice, and the vertical-equity findings do not depend on direct exponentiation. This is
a genuine strengthening of the robustness discussion.

**What it may not claim.**

- The price-scale **accuracy** metrics do move, materially: max |Δ| `0.0577` (R²_price),
  `20,809` (MAE_price), `0.0581` (MAPE). Smearing is not accuracy-neutral, only
  equity-neutral.
- **`RMSE_log` is not invariant** to adding a constant, and was not recomputed. Do not list
  it among the invariant quantities. (An earlier internal draft made exactly this error; it
  is corrected and recorded as D-P1-4.)
- **forward_2025 rests on an explicit assumption**: its fitting set is the 382,897-row
  production block, which has no out-of-fold analogue, so the development-estimated `s` is
  applied there unchanged. This is a stated limitation, not a validated property, and must
  be disclosed if the 2025 smearing sensitivity is reported.
- The factor is **D3 row-balanced**, not the naive duplicate-weighted pooled-OOF mean. The
  naive value is reported but never applied; max `|s − s_naive| = 3.94e-03`. If a smearing
  factor is quoted, say which one.
- This is a **sensitivity**. The canonical model is unaltered, nothing was retuned, and no
  smeared value replaces a headline number.

---

## 6. New disclosure: a reproducibility limit of the P0 verification

Worth one honest sentence in replication materials. The P0 suite is **153/153 at the
immutable tag**, reproduced in an isolated detached worktree. But reproducing it requires
re-attaching state that git does not carry: two gitignored data trees, one read-only file
mode, and — the substantive one — a **relative file-mtime ordering** asserted by
`test_source_equivalence_precedes_reproduction_interpretation`. Git does not record mtimes,
so that single assertion is **not reproducible from committed content**; a content-only
checkout scores 152/153 with that test as the sole failure.

Separately, on the P1 branch the suite reports **152/153**, the sole failure being
`test_only_gitignore_modified_outside_p0` — a HEAD-relative scope guard that any additive
commit anywhere in the repository necessarily breaks. It is doing its job. **P0 was not
modified to make that number 153**, and P0 content is provably unchanged: the diff against
the tag over the P0 tree is empty, every frozen hash index is byte-exact, and all protected
paths are clean.

---

## 7. What P1 does *not* license

- No manuscript number changes. No headline is replaced, reclassified or retuned.
- No compliance determination. The May-2026 document is a **draft**; the 2013 Standard
  remains the adopted reference; every ED2 row says so.
- No fold-as-IID inference anywhere, for any metric.
- No weighted-rank ED2 extension to pooled OOF.
- The 4 NOT_ATTAINED display cells (Direct at ext targets −0.06, −0.03, 0.00; Surrogate at
  0.00) remain NOT_ATTAINED, with null metrics, exactly as frozen.

---

## 8. Suggested drafting order

1. Replace the dCor `\todo` (line 3204) with the archived specification — fully determined,
   no judgement required.
2. Qualify any "|VEI| > 10%" sentence to separate the point estimate from the Step-7
   significance outcome (§4.2).
3. Add the smearing invariance result to the robustness discussion near line 3057, with the
   accuracy-moves and forward_2025 caveats (§5).
4. Add the pooled-OOF ED2 non-applicability limitation (§4.3) and the `array_split`
   grouping deviation (§4.5).
5. Fix the `IAAO2026ExposureRatio` bibliography entry to the printed May-2026 title (§4.4).
6. Optionally add PRB CIs and the four-state rule (§3).
