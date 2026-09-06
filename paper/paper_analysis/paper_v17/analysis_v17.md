I completed the major-revision pass on the current manuscript source and treated the requested framing as the governing principle throughout:

> **CCAO-centered applied-methods and diagnostic paper; not a general fairness solution, not a penalty-selection paper.**

I used the current LaTeX source as the baseline  and also checked the rendered manuscript, where the revision-history layers were visibly interfering with the reader-facing story. 

## Revised artifacts

[**Revised main LaTeX — `paper_v15_major_revision.tex`**](sandbox:/mnt/data/paper_v15_major_revision.tex)

[**Full P0/P1 revision plan — problem / why / where / implementation / final message / confidence**](sandbox:/mnt/data/P0_P1_MAJOR_REVISION_PLAN.md)

[**Historical bibliography additions**](sandbox:/mnt/data/references_major_revision_additions.bib)

[**Major-revision changelog**](sandbox:/mnt/data/MAJOR_REVISION_CHANGELOG.md)

The revised `.tex` passes a two-pass LaTeX syntax check with balanced environments, no duplicate labels, no unresolved internal cross-references, and no fatal compilation errors. I did **not** present the temporary syntax-check PDF as a revised paper PDF: the uploaded package does not contain the repository bibliography/figure assets, and several figures/results must legitimately be regenerated after the P0 experiments. Producing a “final” PDF with stale figures or unresolved references would defeat the purpose of this revision.

---

# 1. What changed at the highest level

The paper now has one much clearer scientific identity.

### Primary research question

The manuscript is organized around:

> **Can a small, auditable change to the training objective attenuate the observed-sale price-related valuation-ratio pattern in a CCAO residential LightGBM research workflow while retaining the predictive advantages of the nonlinear learner—and what limitations appear as regularization strengthens and time moves forward?**

That replaces several competing stories that had accumulated in the previous draft:

* “fair property taxation”;
* generic ML fairness;
* a new penalty;
* penalty-region discovery;
* universal tuning;
* external transferability;
* nonlinear diagnostics;
* CCAO implementation.

Those components no longer have equal status.

---

# 2. Final contribution hierarchy

The revised manuscript now makes **three primary contributions**.

| Contribution                                           | Final role                                                                                                                                                                                                | Confidence |
| ------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------: |
| **Classical assessor diagnostic → training objective** | Translate a classical log-log vertical-equity relationship into an explicit training-time control in a CCAO-motivated LightGBM workflow                                                                   |        95% |
| **Direct versus Surrogate mechanisms**                 | Establish mathematically that the Direct objective and Jensen Surrogate are genuinely different interventions: signed sample-wide covariance versus symmetric price-tail error weighting                  |        99% |
| **CCAO temporal path and failure-mode audit**          | Trace complete fixed-base-learner paths through rolling CV, held-out, and 2025 evaluation and identify where improvement, predictive deterioration, nonlinear deformation, and temporal instability occur |        98% |

The secondary contributions are now clearly secondary:

* the Bayes residual–outcome covariance result;
* fixed-space rank-one Direct geometry;
* weighted-projection Surrogate geometry;
* \(\Delta_{\mathrm{NL}}\);
* normalized penalty coordinates;
* path-screening infrastructure;
* reproducibility machinery;
* the separate multi-jurisdiction stress test.

This is a substantially stronger hierarchy.

---

# 3. The most important novelty correction

One of the most important revisions was to stop implicitly treating the underlying log-ratio/log-price relationship as new.

The revised Related Work now makes the historical connection explicit. If the classical log-log specification is

$$
\log \widehat P=a+b\log P+u,
$$

then

$$
\log(\widehat P/P)=a+(b-1)\log P+u.
$$

Therefore, neutrality of the classical elasticity \(b=1\) corresponds directly to zero slope of the log valuation ratio against log sale price.

The paper now says, essentially:

> **The first-order relationship is classical; our contribution is converting it from an ex-post assessor diagnostic into a train-time regularization target and studying its behavior inside the CCAO-style ML workflow.**

I added the historical bridge through Paglin–Fogarty, Cheng, Edelstein, and Sunderman et al. to the supplied BibTeX additions.

This is both more accurate and more convincing than trying to claim a new definition of vertical equity.

---

# 4. Major structural changes actually implemented

The main body has been substantially compressed and reorganized.

### Introduction

It now follows one sequence:

**property valuation → observed-sale vertical-equity concern → CCAO baseline tension → training-time intervention → temporal design → contributions → strict boundaries.**

The obsolete six-county result is gone from the active motivation.

The introduction no longer suggests that the project is discovering a universally useful penalty region.

### Related Work

It is now organized around:

* ML in mass appraisal;
* historical vertical-equity measurement;
* model/local/segmented remedies;
* fair regression, dependence control, and post-processing;
* the precise gap filled by this paper.

This makes Candogan–Han–Lu and the post-hoc literature much more important comparators instead of peripheral citations.

### Evaluation metrics

This received a major compression.

The old main-text metric encyclopedia has been replaced by a hierarchy:

**Prediction**

* \(R^2_P\)
* MAE

**Headline assessor vertical equity**

* PRB
* VEI

**Mechanism / nonlinear shape**

* \(\beta_{\log}\)
* \(\Delta_{\mathrm{NL}}\)

MAPE, log RMSE, ratio level, COD, COV, PRD, MKI, and dCor remain available as complementary diagnostics in the appendix.

This substantially improves the reader's ability to understand what a result means.

### Method

The section is now titled:

> **Covariance-Guided Regularization of a Price-Related First-Order Pattern**

rather than implicitly presenting the method as a complete “regressivity correction.”

The manuscript now distinguishes consistently between:

* the **mathematical Direct objective**, and
* the empirical **Direct-diagonal** LightGBM implementation.

That distinction is essential because the implementation supplies the exact covariance gradient but not the dense rank-one cross-observation curvature.

### Results

The Results now tell four connected stories:

1. **CCAO baseline tension**
2. **Complete path behavior**
3. **Ratio-shape/nonlinear failure**
4. **Temporal instability of exact raw-\(\rho\) locations**

The candidate-region machinery is no longer one of the headline contributions.

### Discussion

The Discussion has been rewritten around three conclusions:

* predictive performance and price-related vertical equity are distinct;
* Direct and Surrogate attenuate the target through different mechanisms;
* complete paths and failure modes are the appropriate scientific object here—not a selected \(\rho\).

---

# 5. P0 items that are fixed in writing versus still empirical TODOs

A critical part of this exercise was **not pretending that wording can solve an experimental problem**.

### Already implemented

The following P0 corrections are resolved in the manuscript itself:

* clean reader-facing source;
* no tracked-revision layers;
* narrow title and estimand;
* CCAO-centered contribution framing;
* historical novelty correction;
* Direct versus Direct-diagonal terminology;
* removal of obsolete external evidence;
* no selected penalty claim;
* no general fairness claim;
* no in-processing superiority claim;
* removal of numeric winner boldface/asterisks;
* centered-spread comparator formally incorporated into the design;
* explicit external-evidence boundary;
* reproducibility section rewritten around an actual final freeze.

### Correctly left as explicit TODOs

These cannot responsibly be filled without experiments:

**P0 parity rerun.**
Native LightGBM, Direct-diagonal at \(\rho=0\), and a true custom-objective Surrogate at \(\rho=0\) must agree to near numerical precision before positive-\(\rho\) differences are attributed to the penalty.

**P0 complete parity-correct path regeneration.**
If initialization changes, all affected positive-\(\rho\) CCAO paths, tables, and figures must be regenerated.

**P0 centered-spread comparator.**
The paper now specifies

$$
f_b(x)
=
\bar y_{\mathcal T}
+
b\left(f_0(x)-\bar y_{\mathcal T}\right),
$$

with \(b\) estimated using training/CV information only.

This comparator is no longer optional. It follows directly from the paper's own fixed-space theorem.

**P0 paired uncertainty.**
The final tables need paired intervals before very small differences are interpreted.

**P0 CCAO provenance/scope freeze.**
The final research extract, sale filters, property-class/condominium scope, and exact Python-versus-published-workflow correspondence must be documented.

**P0 reproducibility freeze.**
The final paper must name the actual commit/diff, environment, data identity, model configuration, predictions, generation commands, and artifact hashes.

---

# 6. The centered-spread comparator is now central

This is perhaps the most consequential scientific change.

The fixed-space theorem implies that exact Direct regularization, with an intercept, behaves as a centered prediction-spread transformation.

Therefore the paper now explicitly acknowledges:

> If a validation-fitted one-dimensional post-hoc spread transformation reproduces the Direct-diagonal path, then the contribution is primarily diagnostic and mechanistic rather than evidence that retraining is intrinsically better.

Conversely:

> If parity-correct Direct-diagonal retraining produces a meaningfully better accuracy/equity/shape path than centered spread after matched first-order correction, that becomes a genuine additional applied-method contribution.

Both outcomes are scientifically useful.

This makes the paper considerably more robust to an “unfavorable” experiment.

---

# 7. Candidate-region / penalty-selection material

I demoted this deliberately.

The revised main paper does **not** claim:

* a safe interval;
* a candidate operating range;
* a universal penalty;
* a penalty-selection algorithm;
* temporal stability of raw-\(\rho\) endpoints.

The historical screening machinery is retained only in the appendix because it documents how high-\(\rho\) deterioration was diagnosed during development.

The headline object is now:

> **the complete fixed-base-learner regularization path.**

That is much cleaner scientifically.

---

# 8. CCAO focus after the revision

The CCAO focus is now considerably stronger.

The empirical narrative is explicitly:

> ordinary CCAO-motivated LightGBM versus historically motivated linear regression → CCAO temporal data → CCAO Direct/Surrogate paths → CCAO failure modes → implications for CCAO-style model development.

The normalized multi-jurisdiction experiment is deliberately prevented from becoming a second paper inside the first.

The CCAO language is also more accurate:

> **research translation of the CCAO workflow**

rather than:

> official CCAO assessments / deployed CCAO model.

One P0 issue remains because it requires source verification: the exact research property scope, especially condominium treatment, must be reconciled with the public CCAO residential AVM description.

---

# 9. P1 empirical work now specified precisely

The revised source contains detailed hidden TODOs for:

* VEI bootstrap with proxy/group re-estimation;
* MKI tie handling;
* \(\Delta_{\mathrm{NL}}\) spline/fold sensitivity;
* date-grouped chronology;
* repeated-PIN/resale overlap;
* parcel-blocked sensitivity;
* feature-timing audit;
* smearing/retransformation sensitivity;
* local CCAO geography analysis;
* residential property-class analysis;
* value-band analysis;
* worst-group reporting;
* sold-versus-unsold transport;
* one domain-specific comparator;
* limited learner-capacity/early-stopping sensitivity;
* path uncertainty;
* screen-free mechanism figure regeneration;
* later external-benchmark integration only after its separate audit closes.

The TODOs describe **what to compute, why, how to interpret it, and what claim it can or cannot support**. They are hidden in ordinary compilation by default.

---

# 10. Final storyline

The final paper should read approximately as follows:

> **CCAO's nonlinear model predicts substantially better than the linear benchmark but exhibits a stronger observed-sale price-related ratio pattern. Classical vertical-equity diagnostics suggest a simple trainable first-order target. We implement that target through two mathematically distinct regularization objectives. Complete chronological paths show where those interventions attenuate the first-order pattern and where their limitations emerge. Direct and Surrogate behave differently, and Surrogate demonstrates particularly clearly that first-order neutrality need not imply a well-behaved ratio profile. Exact operating locations are temporally unstable. The method is therefore an auditable model-development control and diagnostic—not a fairness certificate or automatic tuning rule.**

That is now the organizing logic of the manuscript.

---

# 11. Final assessment

### Before this revision

My assessment was approximately:

* scientific core: **strong**
* novelty positioning: **promising but blurred**
* CCAO focus: **good empirically but competed with other stories**
* storyline: **fragmented**
* writing: **technically strong but heavily layered**
* submission readiness: **low**
* danger: the manuscript looked simultaneously like a fairness paper, penalty-selection paper, external-validity paper, and CCAO application.

### After the implemented structural revision

My assessment of the **paper architecture** is now:

| Dimension                        |         Assessment |
| -------------------------------- | -----------------: |
| Research question                |               9/10 |
| CCAO empirical focus             |             9.5/10 |
| Contribution hierarchy           |               9/10 |
| Novelty accuracy                 |               9/10 |
| Mathematical story               |               9/10 |
| Direct–Surrogate interpretation  |             9.5/10 |
| Failure-mode contribution        |             9.5/10 |
| Storyline                        |               9/10 |
| Main-text clarity                |               9/10 |
| Scope discipline                 |             9.5/10 |
| Current empirical identification |    6/10 pending P0 |
| Comparator completeness          |    6/10 pending P0 |
| Statistical support              | 6/10 pending P0/P1 |
| Submission readiness now         |            ~6.5/10 |
| Expected readiness if P0/P1 pass |              ~9/10 |

The remaining weakness is no longer primarily the paper's story. It is empirical closure.

The **single most important next sequence** is now unambiguous:

**parity-correct CCAO paths → centered-spread comparator → paired uncertainty → CCAO subgroup/transport audits → final table/figure regeneration → final clean PDF.**

I would not add another method, another penalty, or another jurisdiction before those items are closed.
