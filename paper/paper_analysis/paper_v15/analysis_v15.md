# Updated audit: commit `9a28d685`

## Short answer

Yes—this is a meaningful positive update, but it changes only the external-validation part of the previous audit.

Commit [`9a28d685`](https://github.com/nicacevedo/soft-vertical-equity-constrained-mass-appraissal/commit/9a28d685f39f8cfc9ad3061478206f13b8dc5eac) contains the completed frozen 2025 evaluation for all nine jurisdictions, including:

* all Direct and Surrogate paths;
* frozen-input and training-period audits;
* CV–2025 comparison tables;
* paired monthly-block bootstrap intervals;
* ratio profiles;
* cross-jurisdiction figures;
* a forward-results report.

The attached [cursot_output_2.zip](sandbox:/workspace/scratch/60b9bdc97c34/upload/cursot_output_2.zip) exactly matches the committed forward artifacts.

This resolves my previous statement that the external benchmark was CV-only and lacked committed forward results. External temporal evidence is now substantially stronger.

However, it does not resolve the paper’s largest blockers:

* parity-correct rerunning of the primary CCAO paths;
* the missing post-hoc spread comparator;
* the sale-price-versus-latent-value interpretation;
* local/subgroup assessment;
* manuscript restructuring and claim narrowing;
* final reproducibility and PDF cleanup.

The overall verdict therefore remains “major revision,” although external validity improves from approximately B− to B/B+.

---

## 1. Exact change from the previous audit

| Previous audit finding                                   | Status at `9a28d685`                                                                                   | Does it change the audit?               |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ | --------------------------------------- |
| Nine-jurisdiction benchmark had CV results only          | Full calendar-2025 forward paths are now committed                                                     | Yes—major positive update               |
| No external forward uncertainty                          | Paired monthly-block bootstrap, 200 draws, is present                                                  | Partially resolved                      |
| Direct common normalized band was CV-only                | Frozen band \([0.387,0.562]\) was evaluated in 2025                                                    | Meaningfully strengthened               |
| Surrogate had no nondegenerate universal interval        | Still no such interval; no replacement was selected from 2025                                          | Reinforced                              |
| Forward inputs and model choices needed integrity checks | Freeze hash, cohort identity, cache hashes, configurations, training dates, and grid completeness pass | Largely resolved within the v1 pipeline |
| Old six-county benchmark should be superseded            | New benchmark now has forward evidence sufficient to replace it                                        | Yes; old section should now be removed  |
| Primary Cook CCAO paths have a \(\rho=0\) parity concern | Not addressed by this commit                                                                           | Unchanged P0 blocker                    |
| Direct needs a matched post-hoc spread comparator        | Not addressed                                                                                          | Unchanged P0 blocker                    |
| Fairness framing is too broad                            | Not addressed                                                                                          | Unchanged                               |
| Paper is unfinished and contains TODOs/tracked changes   | Attached paper is unchanged                                                                            | Unchanged                               |
| Statistical evidence needed strengthening                | External bootstrap helps, but is still limited                                                         | Partial improvement                     |
| External 2025 could not be called fully confirmatory     | Still true because overlapping 2025 data had been inspected in earlier experiments                     | Unchanged qualification                 |

A crucial distinction: the new Cook external benchmark uses the standardized ATTOM/common-model design with 278,479 training and 24,993 evaluation sales. It is not the same as the primary CCAO research translation with 382,897 pre-2025 sales and 26,641 2025 sales. The new Cook result cannot substitute for rerunning the primary CCAO experiments.

---

# 2. What the new evidence actually shows

## 2.1 Integrity and completion are strong

The external run appears operationally complete:

* Nine jurisdictions × two families completed.
* All expected frozen grid points are present.
* Pre-2025 development-table identities match their frozen hashes.
* Cache hashes match the freeze.
* Training samples end before January 1, 2025.
* Evaluation rows are calendar-2025 transactions.
* No 2025-derived candidate region was written.
* No automatic Surrogate recalibration was performed.
* The supplied archive and repository artifacts agree.

These are substantial strengths. The empty error log for job `21976709` confirms the postprocessing job completed cleanly, but the meaningful evidence is in the committed audits, metrics, and reports—not in the empty log itself.

The canonical report is [`FORWARD_2025_RESULTS.md`](https://github.com/nicacevedo/soft-vertical-equity-constrained-mass-appraissal/blob/9a28d685f39f8cfc9ad3061478206f13b8dc5eac/analysis/external_jurisdiction_benchmark_v1/forward_2025/reports/FORWARD_2025_RESULTS.md).

## 2.2 Baseline regressivity is geographically widespread

All nine 2025 baseline AVMs have negative \(\beta_{\log}\) under both the Direct and native-baseline branches. Baseline values range approximately from:

* \(-0.136\) in St. Louis County;
* to \(-0.305\) in Philadelphia.

This supports a broad empirical motivation: price-related residual slope is not unique to Cook County.

It should be described as nine jurisdictional findings, not “18 independent cases,” because the two family baselines within each jurisdiction use the same data and nearly the same underlying learner.

## 2.3 Low normalized penalties transfer directionally

Restricting attention to the 16 protocol-valid jurisdiction–family candidate regions:

* Every activity point improved the absolute 2025 \(\beta_{\log}\).
* Every activity point improved absolute PRB.
* Median attained \(\beta\)-correction was approximately 8.1%.
* The range was approximately 4.5%–11.9%.
* Median \(\Delta\)NMSE was only 0.00055 NMSE points.
* The range was \(-0.00253\) to \(0.00821\).

This is the strongest new finding:

> Small normalized penalties generate directionally consistent, modest reductions in price-related slope across heterogeneous jurisdictions, generally at small predictive cost.

That is more defensible than claiming that the full candidate regions “transfer.”

## 2.4 The frozen Direct common band performs well descriptively

For the eight protocol-valid Direct jurisdictions, excluding Allegheny from the common-band construction:

| Frozen coordinate        | 2025 \(A_\beta\) range | Median \(A_\beta\) | 2025 \(\Delta\)NMSE range | Median \(\Delta\)NMSE |
| ------------------------ | ---------------------: | -----------------: | ------------------------: | --------------------: |
| \(\widetilde\rho=0.387\) |             6.3%–10.9% |               9.8% |       −0.00341 to 0.00182 |              −0.00016 |
| \(\widetilde\rho=0.562\) |             8.7%–15.2% |              13.8% |       −0.00268 to 0.00220 |               0.00025 |

This is genuinely encouraging. The most accurate claim is:

> Both frozen Direct coordinates improved the absolute log-slope in all eight protocol-valid jurisdictions, while their 2025 NMSE changes remained small.

That is stronger and more transparent than the current “practically useful in 8/8” binary claim.

## 2.5 Direct now looks empirically stronger than Surrogate

At comparable CV-derived mechanism anchors:

| Frozen mechanism point     | Direct median 2025 \(A_\beta\) | Direct median \(\Delta\)NMSE | Surrogate median 2025 \(A_\beta\) | Surrogate median \(\Delta\)NMSE |
| -------------------------- | -----------------------------: | ---------------------------: | --------------------------------: | ------------------------------: |
| Activity                   |                           6.4% |                      0.00012 |                              8.3% |                         0.00193 |
| \(A_\beta=0.25\) CV anchor |                          29.0% |                      0.00444 |                             28.2% |                         0.01998 |
| \(A_\beta=0.50\) CV anchor |                          58.3% |                      0.02901 |                             48.3% |                         0.05686 |

The important result is not merely that both methods reduce the slope. It is that Direct appears much more efficient once a material correction is requested:

* At the 25% anchor, Surrogate’s median NMSE cost is approximately 4.5 times Direct’s.
* At the 50% anchor, Surrogate’s median cost is approximately twice Direct’s.
* Surrogate still exhibits stronger nonlinear/profile deformation in several jurisdictions.

This makes the appropriate paper architecture clearer:

1. Direct is the primary proposed method.
2. Surrogate is a simple, implementable upper-bound alternative.
3. Surrogate is useful at low penalties but has an earlier practical limitation.
4. The Surrogate failure is scientifically informative, not something to hide.

## 2.6 The 2025 results confirm heterogeneity, not universal tuning

The outcomes reinforce several limitations:

* No nondegenerate all-jurisdiction Surrogate interval exists.
* Direct guardrails remain highly heterogeneous.
* Allegheny Direct has no protocol-valid stable candidate region.
* Middlesex Surrogate has an inverted activity–guardrail ordering.
* Maricopa and Middlesex experience substantial CV-to-2025 baseline-level shifts.
* Some high-penalty Direct paths remain numerically unstable.
* Miami-Dade Direct overshoots PRB at its guardrail: approximately \(-0.047\) to \(+0.057\), even while \(\beta_{\log}\) improves.

Therefore normalized \(\rho\) improves portability, but does not produce a universal plug-in tuning rule.

---

# 3. Problems in the new reporting layer

The forward experiment is valuable, but several generated headline statements should not enter the paper as currently written.

| Issue                                                                  | Why it matters                                                                                                     | Required correction                                                                                |  Confidence |
| ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------- | ----------: |
| “Practically useful in 8/8” is almost vacuous                          | The code defines usefulness as a successful fit and \(A_\beta\geq0\), with no meaningful accuracy-cost requirement | Delete the binary claim; report the actual \(A_\beta\) and \(\Delta\)NMSE ranges                   |        High |
| `SUPPORTS/PARTIAL` relies on an unstated \(A_\beta\geq0.10\) threshold | The prose rubric says same-sign transfer but the code applies an arbitrary 10% threshold                           | Treat these labels as exploratory or remove them; do not retune the threshold after observing 2025 |        High |
| Rubric says PRB or beta reversal, but implementation checks only beta  | The classification can miss a PRB deterioration                                                                    | Report PRB and beta separately; do not collapse them into one label                                |        High |
| Invalid CV regions still receive forward labels                        | Allegheny Direct and Middlesex Surrogate are classified even though their candidate regions are not protocol-valid | Put them in a separate sensitivity/failure table, outside candidate-region transfer counts         |        High |
| Middlesex Surrogate is labeled `SUPPORTS_CV`                           | Its guardrail precedes activity, so no valid region exists to support                                              | Replace with “invalid CV ordering; forward anchors shown descriptively”                            |        High |
| `vertical_equity_benefit_2025` compares 2025 PRB with CV PRB           | That is cross-period drift, not within-2025 improvement                                                            | Use the same-split baseline-derived \(\Delta I_{\mathrm{PRB},2025}\), or remove the field          |        High |
| Full-scale \(\rho=0\) parity is not established                        | Direct and native 2025 baseline metrics differ, despite tight 4,000-row pilot parity                               | Run full-data prediction-level parity for all nine jurisdictions                                   |        High |
| The 200-draw bootstrap is thin                                         | Percentile endpoints are based on very few order statistics                                                        | Increase to at least 2,000 draws; retain exactly the same frozen anchors                           |        High |
| Only 12 calendar-month blocks exist                                    | Conventional bootstrap interpretation is fragile with 12 clusters                                                  | Add transaction-paired and weekly/monthly-block sensitivity; present all variants                  | Medium-high |
| Forward evidence is not genuinely analyst-blind project-wide           | Earlier overlapping experiments examined 2025 outcomes                                                             | Call it protocol-frozen forward reanalysis, not pristine confirmation                              |        High |
| Repository README still says 2025 is locked                            | It is now completed                                                                                                | Update status and canonical artifact list                                                          |        High |
| Paper-integration draft is only four comment lines                     | No usable external-results section was actually drafted                                                            | Produce a factual integration memo after the audit corrections                                     |        High |

## Full-scale parity warning

The new forward tables show differences between Direct custom \(\rho=0\) and native LightGBM baselines. Maximum observed metric gaps include approximately:

* Price \(R^2\): 0.0082 in Middlesex.
* NMSE: 0.00135 in Middlesex.
* PRB: 0.00329 in Philadelphia.
* \(\beta_{\log}\): 0.00260 in Philadelphia.

These do not by themselves prove that parity failed—price \(R^2\) can be sensitive to a few tail predictions—but the earlier 4,000-row pilot cannot be treated as sufficient.

The agent should calculate, on every full 2025 evaluation set:

$$
\max_i |f_i^{\text{native}}-f_i^{\text{custom}}|,
\qquad
\operatorname{mean}_i |f_i^{\text{native}}-f_i^{\text{custom}}|,
$$

for:

1. native LightGBM;
2. Direct custom objective at \(\rho=0\);
3. Surrogate custom objective at \(\rho=0\).

The current forward script uses native LightGBM directly for the Surrogate zero baseline, so custom-Surrogate zero parity is not actually tested in the full run.

---

# 4. Figure assessment

The figures are useful diagnostic artifacts but not yet all main-paper quality.

### `accuracy_mechanism_frontier_cv_vs_2025.pdf`

* Direct numerical tails dominate the vertical scale.
* Most scientifically relevant low-cost paths collapse near zero.
* Jurisdictions cannot be identified.
* Invalid/divergent regions are visually mixed with usable regions.

Use a restricted protocol-relevant panel in the main paper and retain the full numerical-tail version in the supplement.

### `forward_key_metric_paths_9jurisdictions.pdf`

* Scientifically rich but too dense: 36 small panels.
* Good as a supplement overview.
* Too small for a main-text figure.
* It needs cleaner mathematical labels, common-band markers, and uncertainty or fold-range indications.

### `forward_ratio_profile_examples.pdf`

This is one of the most valuable outputs because it makes clear that slope correction does not necessarily flatten the entire profile. However:

* add uncertainty bands;
* mark the frozen coordinates numerically;
* explain that the higher anchors can create U-shape or overspreading;
* retain Philadelphia, St. Louis County, and Middlesex because they were predeclared, not chosen from 2025 outcomes.

### `berry_local_vs_avm_ratio_profiles.pdf`

This figure correctly says that official assessment ratios and AVM ratios are different constructs and that Wayne County is not Detroit. Nevertheless:

* the St. Louis official level near 0.17 is institutionally scaled and not directly comparable to the AVM ratio level;
* the left and right panels use different populations, estimands, and geographic boundaries;
* it should be presented as contextual triangulation, not external validation.

### `cv_to_2025_path_drift.pdf`

Useful diagnostic, but it lacks a jurisdiction legend. As currently rendered, the colored curves cannot be interpreted. Add the legend and separate ordinary drift from numerically divergent tail behavior.

---

# 5. Highest-quality next steps

## Priority 1 — close the external forward audit

The local agent should not start new external model families or edit the live manuscript yet.

Its first goal should be a surgical audit/addendum under the existing isolated benchmark directory.

### Required tasks

1. Add full-data, prediction-level native/custom \(\rho=0\) parity checks for all nine jurisdictions.

2. Test three baselines:

   * native LightGBM;
   * `LGBCovPenalty(rho=0, match_native_init=True)`;
   * `LGBSmoothPenalty(rho=0, weighting_proxy_mode="identity", match_native_init=True)`.

3. Preserve the existing forward artifacts; write new parity artifacts rather than silently overwriting the committed results.

4. Remove or neutralize the “8/8 practically useful” boolean.

5. Replace it with the actual predeclared-band summary:

   * slope improvement range;
   * NMSE-change range;
   * bootstrap interval;
   * PRB direction;
   * nonlinear diagnostics.

6. Separate protocol-valid and invalid candidate statuses:

   * Allegheny Direct: sensitivity only;
   * Middlesex Surrogate: invalid region ordering;
   * neither belongs in candidate-region support counts.

7. Fix the PRB benefit variable so it compares each 2025 penalized model with its own 2025 baseline.

8. Preserve the original `SUPPORTS/PARTIAL` labels only as an archived exploratory summary, or remove them from the paper-facing report. Do not invent a new favorable threshold after seeing 2025.

9. Increase the bootstrap to at least 2,000 paired draws at exactly the same frozen coordinates. Add transaction-level and weekly/monthly-block sensitivity without selecting among them.

10. Update the benchmark README to state that the 2025 forward pass is complete.

### Definition of done

The external benchmark is closed only when there is one concise audit table showing:

| Jurisdiction | Family | CV status | Frozen coordinate | 2025 \(A_\beta\) | \(\Delta\)NMSE | \(\Delta\)PRB ideal-distance | \(\Delta\Delta_{\mathrm{NL}}\) | Bootstrap interval | Valid interpretation |
| ------------ | ------ | --------- | ----------------: | ---------------: | -------------: | ---------------------------: | -----------------------------: | ------------------ | -------------------- |

No qualitative label should be necessary to understand the result.

---

## Priority 2 — produce the external-results integration package

After the audit passes, the agent should prepare—but not yet insert—a paper-integration package containing:

* one compact main table;
* one main Direct portability figure;
* one Direct-versus-Surrogate frontier figure;
* one predeclared ratio-profile figure;
* one supplement-wide path figure;
* a concise results-section draft;
* exact limitations language.

The main message should be:

> The normalized Direct low-penalty region shows consistent but modest forward slope correction with near-zero median NMSE change. Larger corrections exhibit heterogeneous and increasing costs. Surrogate acts earlier but is less efficient at material correction and has no nondegenerate universal region.

The old six-county ATTOM manuscript section should be explicitly marked for deletion.

---

## Priority 3 — return to the primary CCAO blocker

Once the external package is frozen, the project’s highest-priority scientific task remains the primary CCAO rerun:

1. Rerun the complete Direct and Surrogate paths using parity-correct initialization.
2. Preserve the chronological CV, heldout, and 2025 structure.
3. Recompute all path metrics and \(\Delta_{\mathrm{NL}}\).
4. Do not redefine candidate regions from heldout or 2025.
5. Compare the new results against the frozen old paths.
6. Determine whether the apparent Direct accuracy improvement survives.

This is more important to the paper than adding another county or another penalty family.

---

## Priority 4 — add the mandatory post-hoc comparator

Immediately after the parity-correct CCAO rerun, add the centered-spread comparator implied by the fixed-space theorem.

This comparator should:

* use only training/CV data to estimate its parameter;
* generate a complete correction frontier;
* be evaluated on heldout and 2025 without retuning;
* be compared with Direct at matched \(A_\beta\), NMSE, PRB, and nonlinear diagnostics.

Until this is done, the paper cannot establish that Direct retraining adds value beyond a simple post-hoc dispersion correction.

---

# 6. What the local agent should not do now

The agent should not:

* rerun the quarantined v3 experiments;
* add more jurisdictions;
* redesign the normalized grid using 2025;
* find a new universal Surrogate band;
* change the 2025 result because a label looks unfavorable;
* edit `paper/paper_v12.tex` before the external audit is closed;
* present the generic ATTOM Cook model as the primary CCAO model;
* call the external forward pass fully confirmatory;
* treat the Berry/local official-ratio comparison as validation of the AVM;
* spend time polishing all 36 path plots before fixing the report logic and parity.

---

## Final updated verdict

This commit meaningfully improves the project. The external benchmark is no longer merely promising CV evidence: it now has a complete, frozen-coordinate 2025 evaluation with broad directional support.

The strongest new result is that the Direct normalized interval around \(\widetilde\rho=0.39\)–\(0.56\) produces modest slope reductions across all eight protocol-valid jurisdictions with very small forward NMSE changes. Direct also appears materially more efficient than Surrogate at the 25% and 50% correction anchors.

But the generated summary layer is currently more confident than the underlying definitions justify. The next local-agent task should therefore be audit closure—especially full-scale zero-penalty parity and removal of the weak binary success criteria—followed by the parity-correct primary CCAO rerun and post-hoc comparator.

In short:

> External validation has advanced substantially; the overall paper has not yet crossed its main methodological blockers. The agent should now consolidate and verify, not expand.
