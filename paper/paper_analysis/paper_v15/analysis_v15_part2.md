I reviewed the current [source](sandbox:/workspace/scratch/60b9bdc97c34/upload/Pasted text(20260905-040646).txt) and [65-page PDF](sandbox:/workspace/scratch/60b9bdc97c34/upload/Reducing_Regressivity_in_ML_based_Mass_Valuation (1)(1).pdf), including the abstract, mathematics, CCAO design, complete-path results, appendices, tracked revisions, and closest literature.

## Overall verdict

This is now a credible and potentially valuable applied-methods paper. It does not require a new conceptual foundation. It requires a major but focused revision that:

1. freezes and cleans the manuscript;
2. resolves several empirical attribution blockers;
3. centers the story on one CCAO research question;
4. narrows the novelty claim from “new equity objective” to “training-time translation of a classical vertical-equity relationship”;
5. removes the candidate-screening machinery from the central narrative; and
6. adds the closest post-hoc and CCAO subgroup comparisons.

The strongest paper is not:

> We solve regressivity in machine-learning property assessment.

It is:

> In a research translation of CCAO’s residential LightGBM workflow, we turn a classical price-related valuation-ratio slope into an auditable training-time control. The resulting paths can reduce the observed-sale first-order pattern before predictive performance deteriorates, but the effective penalty is temporally unstable and first-order neutrality can coexist with nonlinear and local problems.

That is honest, technically interesting, operationally relevant, and better differentiated from the literature.

### Current quality assessment

| Dimension                       |    Score | Judgment                                                                                                         |
| ------------------------------- | -------: | ---------------------------------------------------------------------------------------------------------------- |
| Importance of research problem  |     9/10 | High-stakes, operationally real, and under-addressed during model training                                       |
| Conceptual objective            |   8.5/10 | Now appropriately narrow and clearly defined                                                                     |
| Mathematical exposition         |   8.5/10 | Mostly precise, transparent, and unusually candid about limitations                                              |
| Novelty positioning             |   7.5/10 | Defensible, but the historical assessment-literature bridge needs correction                                     |
| CCAO empirical focus            |   8.5/10 | Clearly the main empirical application                                                                           |
| CCAO operational correspondence |   6.5/10 | Research translation is clear, but scope, data, and implementation correspondence remain incompletely documented |
| Empirical attribution           |   5.5/10 | Native/custom parity and Direct implementation issues remain blockers                                            |
| Comparator completeness         |     5/10 | Missing the mathematically implied post-hoc centered-spread comparator                                           |
| Uncertainty and robustness      |     5/10 | Extensive descriptive diagnostics, but weak inferential support                                                  |
| Intended storyline              |     8/10 | The underlying story is strong                                                                                   |
| Current rendered storyline      |     3/10 | Superseded text and contradictory versions are printed together                                                  |
| Submission readiness            |     4/10 | Major revision; not yet suitable for external review                                                             |
| Potential after required fixes  | 8–8.5/10 | Strong application-led OR/ML/mass-appraisal paper                                                                |

## 1. What the paper’s objectives should be

### Primary objective

The primary question should be stated as:

> Can a one-parameter modification of the training objective attenuate CCAO’s observed-sale, price-related valuation-ratio pattern while retaining the predictive benefits of its LightGBM model class—and what limitations appear across future periods and stronger regularization?

This formulation gets five things right:

* CCAO is the motivating and evidentiary center.
* The target is observed-sale price-related behavior, not latent fairness.
* The intervention is intentionally minimal.
* The empirical object is the regularization path, not a selected model.
* Failure modes are part of the contribution.

### Supporting objectives

1. Derive the Direct objective and explain its non-additive rank-one curvature.
2. Derive the Surrogate as a genuine Jensen upper bound and exact tail-weighted squared-error objective.
3. Show that Direct and Surrogate are substantively different interventions.
4. Evaluate the paths under rolling-origin, held-out, and 2025-forward designs.
5. distinguish first-order improvement from nonlinear residual structure, valuation level, horizontal uniformity, and local equity.

### Objectives the paper should explicitly reject

The paper is not establishing:

* equity relative to latent market value;
* racial, income, or protected-group fairness;
* equitable final tax liabilities;
* residual–sale-price independence;
* a production-ready CCAO model;
* a selected or safe penalty;
* superiority over post-processing;
* transportability to other jurisdictions; or
* the fully retuned accuracy–equity frontier.

The current abstract and limitations increasingly say this correctly. That discipline must become uniform throughout the manuscript.

## 2. What is genuinely novel

The novelty is meaningful, but principally translational, operational, and empirical—not foundational fairness theory.

| Candidate contribution                                                               | Real novelty                                             | Correct positioning                                                                                                  | Confidence |
| ------------------------------------------------------------------------------------ | -------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ---------: |
| Turning a price-related valuation-ratio slope into a train-time LightGBM regularizer | Moderate–high application novelty                        | “Training-time translation of an assessor diagnostic into an existing AVM workflow”                                  |        92% |
| Direct squared residual–price covariance penalty                                     | Low as general methodology; moderate in this application | Covariance/dependence penalties are established; the mass-appraisal estimand and implementation are the contribution |        98% |
| Jensen Surrogate and weighted-loss representation                                    | Moderate methodological contribution                     | The algebra is elementary, but its exact interpretation and implementation consequences are useful                   |        94% |
| Explicit Direct–Surrogate non-equivalence                                            | Moderate–high                                            | Probably the strongest methodological contribution: signed aggregate control versus symmetric tail-error reweighting |        94% |
| Fixed-space Direct centered-spread result                                            | Low–moderate                                             | Valuable mechanism theorem and limitation, not a headline new optimization theory                                    |        92% |
| Fixed-space Surrogate weighted projection                                            | Moderate                                                 | Useful explanation of its multi-directional behavior; still an interpretive benchmark                                |        88% |
| Full CCAO temporal path and failure-mode audit                                       | High applied novelty                                     | This is probably the strongest overall contribution                                                                  |        94% |
| Showing first-order neutrality can hide nonlinear deformation                        | High empirical/diagnostic value                          | A central negative result, especially for Surrogate                                                                  |        95% |
| Candidate-region screen                                                              | Low/uncertain                                            | A heuristic engineering artifact until independently validated                                                       |        96% |
| \(\Delta_{\mathrm{NL}}\) diagnostic                                                  | Low–moderate                                             | A paper-specific synthesis of established correlation-ratio ideas; useful but secondary                              |        90% |
| External multi-jurisdiction infrastructure                                           | Potential future contribution                            | Not evidence for the current manuscript until its audit is completed                                                 |        99% |

### The historical novelty correction

The manuscript currently presents

$$
\operatorname{Cov}\!\left(\log(\widehat P/P),\log P\right)
$$

as the explicit target developed for this paper. Its training-time use may be novel, but the underlying diagnostic relationship is not.

If the traditional log-log specification is

$$
\log \widehat P = a+b\log P+u,
$$

then

$$
\log(\widehat P/P)=a+(b-1)\log P+u.
$$

Therefore, the slope of the log valuation ratio on log sale price is \(b-1\). Penalizing the residual–price covariance is, up to the fixed variance of log price, penalizing the departure of the log-log valuation elasticity from one. This connects directly to classical vertical-inequity models associated with Cheng-type and subsequent testing frameworks.

The paper should say:

> We convert a classical log-log vertical-equity relationship from an ex-post diagnostic into a training-time regularization target.

That is more accurate and actually strengthens the literature bridge. The history of vertical-equity testing extends at least through Paglin–Fogarty, Edelstein, and Sunderman et al.; the current Related Work begins too late. See, for example, [Edelstein’s 1979 analysis](https://doi.org/10.2307/2330450) and [Sunderman et al.’s testing framework](https://www.tandfonline.com/doi/abs/10.1080/10835547.1990.12090625).

Covariance, correlation, continuous-variable fairness, and constrained regression are already established in fair regression, including [Komiyama et al.](https://proceedings.mlr.press/v80/komiyama18a.html), [Mary et al.](https://proceedings.mlr.press/v97/mary19a.html), and [Wei et al.](https://proceedings.mlr.press/v206/wei23a.html). The manuscript is correct not to claim those general ideas.

## 3. What the paper is doing correctly

| Asset                                                                      | Why it is correct and valuable                                                                            | Where it matters                                    | Mathematical/empirical intuition                                                                                   | Confidence |
| -------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- | --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | ---------: |
| Narrow observed-sale estimand                                              | Prevents a ratio diagnostic from being misrepresented as latent fairness or tax-incidence evidence        | Abstract, §§2–3, Limitations                        | Sale price is a noisy proxy and appears in both the outcome and ratio denominator                                  |        99% |
| Distinction between valuation ratio and statutory assessment/tax liability | Essential in Cook County, where several downstream transformations determine the tax bill                 | Introduction and §2.2                               | \(\widehat P/P\) evaluates the valuation stage only                                                                |        99% |
| CCAO baseline tension                                                      | Gives the method a concrete operational reason to exist                                                   | §2.4 and Table 2                                    | LightGBM improves prediction and COD while worsening price-related diagnostics relative to the linear benchmark    |        96% |
| Exact log-ratio bridge                                                     | Makes the training mechanism immediately interpretable                                                    | Eqs. defining \(e=\log(\widehat P/P)\) and \(C(f)\) | The residual is exactly the log valuation ratio                                                                    |        99% |
| Bayes residual-covariance identity                                         | This is one of the paper’s most intellectually important passages                                         | §3.1 and Appendix B                                 | \(\operatorname{Cov}(m(X)-Y,Y)=-E[\operatorname{Var}(Y\mid X)]\); zero covariance is not automatically fair        |        99% |
| Direct penalty’s sign intuition                                            | Clearly explains what a gradient step does at lower and higher sale prices                                | §3.2                                                | Under negative covariance, the penalty lowers lower-price overpredictions and raises higher-price underpredictions |        98% |
| Direct level invariance                                                    | Correctly separates vertical pattern from overall valuation level                                         | §3.2                                                | A common shift in log predictions changes level but not covariance                                                 |        99% |
| Honest dense-Hessian caveat                                                | Prevents the empirical Direct implementation from being presented as mathematically exact                 | §3.2 and Appendix E                                 | The rank-one \(cc^\top\) term couples observations and cannot be represented by ordinary per-row Hessians          |        99% |
| Genuine Jensen bound                                                       | The revised derivation is correct                                                                         | §3.3                                                | \((n^{-1}\sum e_ic_i)^2\le n^{-1}\sum e_i^2c_i^2\)                                                                 |        99% |
| Surrogate decomposition                                                    | Makes clear why the Surrogate is not merely a computational approximation                                 | Eq. 35                                              | \(\Psi^{\rm surr}=C^2+\operatorname{Var}_n(ec)\)                                                                   |        99% |
| Temporal evaluation                                                        | Much stronger than random splitting for a changing housing market                                         | §4.2                                                | Expanding-window validation tests later sales with earlier information                                             |        97% |
| Disclosure that the held-out set was previously inspected                  | Avoids false confirmatory language                                                                        | §4.1                                                | “Held out” is accurate; “untouched” or “preregistered” would not be                                                |        99% |
| Complete-path reporting                                                    | Appropriate because no defensible operating loss has been specified                                       | §§4–5                                               | It avoids retroactively choosing the most attractive \(\rho\)                                                      |        96% |
| Multiple assessor diagnostics                                              | Consistent with assessor literature, which warns that individual measures have different failure behavior | §2.3 and Results                                    | PRD, PRB, MKI, VEI, ratio curves, and mechanism diagnostics answer different questions                             |        97% |
| Separate nonlinear diagnostic                                              | Directly addresses a predictable failure of global covariance control                                     | §§2.3, 5.3–5.4                                      | A zero linear slope does not imply a flat conditional mean                                                         |        98% |
| Strong Surrogate failure analysis                                          | Scientifically more interesting than reporting only improvements                                          | §§5.3–5.4                                           | Symmetric tail weighting can reduce the linear trend while producing a non-monotone profile                        |        97% |
| Correct IAAO status language                                               | The paper appropriately distinguishes adopted 2013 guidance from the 2026 exposure draft                  | Tables and Appendix A                               | MKI/VEI bands should not be presented as adopted compliance standards                                              |        99% |
| No external-validity overclaim                                             | Appropriate given the incomplete external benchmark audit                                                 | §§4.4, 5.7, Limitations                             | Common commercial data do not reproduce another assessor’s production workflow                                     |        99% |
| No adoption or deployment claim                                            | Correct given the available evidence                                                                      | Abstract, Data/Institutional Statement              | Collaboration and code compatibility are not production adoption                                                   |        99% |

The current description of the 2026 ratio-study revision is accurate: IAAO still describes it as an [exposure draft under review](https://www.iaao.org/about/board-of-directors/governing-documents/ratio-studies-exposure-draft/). The decision to use a suite of vertical-equity diagnostics is also consistent with the [IAAO task-force review](https://researchexchange.iaao.org/jptaa/vol20/iss2/7/).

## 4. What is not yet correct

### Strict blockers

| Problem                                                          | Why it matters                                                                                                                          | Where                                                        | Required correction                                                                                                          | Confidence |
| ---------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------- | ---------: |
| The rendered PDF contains revision history as manuscript content | Multiple mutually inconsistent scientific claims are printed together; a reader cannot determine the operative design or conclusion     | Throughout; especially §§4.2, 5 opening, 5.2–5.6, Discussion | Produce a clean build in which `oldtext` and old revision blocks disappear and accepted text renders in black                |       100% |
| Massive textual duplication                                      | The Results begins with approximately five versions of the same paragraph; Discussion repeats incompatible transition conclusions       | §§5 and 6                                                    | Retain exactly one current version of every paragraph and caption                                                            |       100% |
| Broken headings/captions                                         | Examples include “What the Prespecified Regularization Paths DoWhat the Final…” and duplicated figure captions                          | §5.2, Appendix D, Figures 2–4                                | Replace tracked dual titles with one literal title in the clean source                                                       |       100% |
| Stale six-county claim                                           | The Introduction says six-county experiments examine transferability, while active §§4.4 and 5.7 say the legacy experiment was removed  | Introduction                                                 | Delete the sentence or state that external validation is future work                                                         |       100% |
| Native/custom \(\rho=0\) mismatch                                | Native-to-penalized differences cannot be attributed solely to the penalty                                                              | Results opening, Appendix E, Limitations                     | Complete the initialization-aligned parity run and regenerate all affected tables/figures                                    |       100% |
| Empirical “Direct” is not the exact Direct optimizer             | Exact gradient plus diagonal-only curvature can change tree gains, splits, and leaf updates                                             | §§3.2, 4.3, 5                                                | Call the empirical family “Direct-diagonal” until exactness is demonstrated; quantify approximation sensitivity              |        98% |
| Missing centered-spread comparator                               | The paper itself proves that exact Direct regularization collapses to a simple post-hoc centered-spread transformation in a fixed space | Related Work, §4.3, Limitations                              | Add a validation-fitted post-hoc centered-spread path using identical splits and metrics                                     |        99% |
| No paired uncertainty for headline differences                   | Boldface and stars imply reliable improvements even when differences are tiny and based on unrounded values                             | Tables 2 and 6                                               | Add paired block-bootstrap intervals or remove improvement typography and use effect sizes                                   |        98% |
| Exact CCAO data provenance is missing                            | The paper cannot presently be reproduced or independently scoped                                                                        | §4.1 and data statement                                      | Report extract identifier, retrieval/build date, eligibility rules, source commit, configuration hash, and artifact manifest |       100% |
| Paper promises a future replication package                      | “Should contain” is a project-management statement, not evidence of reproducibility                                                     | Appendix G                                                   | Release it and describe what is available, or remove unreleased promises                                                     |        99% |

The parity and comparator problems are especially important. Until they are fixed, the paper can support:

> These paths exhibit potentially favorable behavior.

It cannot cleanly support:

> The penalty causes improvements over native LightGBM or is preferable to simpler calibration.

### High-priority conceptual, CCAO, and presentation problems

| Problem                                                                     | Why/intuition                                                                                                                                         | Where                               | How to fix                                                                                                                           | Confidence |
| --------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ---------: |
| “Correction” is too normative                                               | Equation 28 proves that moving covariance toward zero can fit transaction noise; it is not always correcting an error                                 | Section 3 title and recurring prose | Prefer “regularization,” “intervention,” or “control”; reserve “correction” for the target pattern                                   |        96% |
| “Attainable path/frontier” is too broad                                     | Hyperparameters and tree count are fixed; this is not the family’s retuned attainable frontier                                                        | Abstract and Discussion             | Say “observed fixed-configuration paths”                                                                                             |        96% |
| Historical diagnostic connection is missing                                 | The target resembles the classical log-log vertical-equity slope                                                                                      | Related Work and §3.1               | Add the elasticity derivation and early vertical-equity literature                                                                   |        95% |
| CCAO property scope may be inconsistent                                     | The manuscript says condominiums are included, while CCAO’s published residential AVM describes its scope as class-200 properties excluding condos    | §4.1                                | Verify the research extract; explain explicitly why its scope differs, or correct the statement                                      |        97% |
| Python translation versus official workflow is underspecified               | “Fits the CCAO workflow” may be read as code-level equivalence, while the published model uses an R/Tidymodels pipeline                               | §§2.1, 4.1                          | List what is replicated, adapted, omitted, and independently implemented                                                             |        96% |
| Countywide evidence is insufficiently CCAO-operational                      | CCAO’s published evaluation includes geography and property-class breakouts; the paper largely reports countywide paths                               | Results and appendix                | Add township/triad, property-class, price-range, and period diagnostics                                                              |        98% |
| Candidate screen competes with the actual contribution                      | It introduces activity thresholds, guardrails, LOFO events, regret, and exact endpoints despite no selection objective                                | §§4.2, 5.2, 5.6 and Appendix F      | Move the entire screen to the supplement or remove it; retain one sentence that strong-\(\rho\) deterioration is temporally variable |        98% |
| The screen is called “generic” without general validation                   | Scale-equivariance and synthetic tests do not establish operational generality                                                                        | §5.6 and Appendix F                 | Call it “exploratory CCAO path-screening diagnostic”                                                                                 |        98% |
| Too many main-text metrics                                                  | Readers must remember four prediction measures, three level measures, two dispersion measures, four equity measures, and three dependence diagnostics | §2.3 and Results                    | Establish a strict hierarchy and move most formulas/tables to Appendix A                                                             |        99% |
| Tiny differences are overinterpreted                                        | An improvement from 0.904 to 0.905 may be sampling or implementation variation                                                                        | Table 6                             | Report paired differences and uncertainty; do not visually reward unrounded microscopic changes                                      |        98% |
| Same-date and repeated-parcel leakage remain                                | Same-day sales are exchangeable, and repeated parcels can create overly similar training/evaluation observations                                      | §4.1 and Limitations                | Group equal dates and block parcels across splits; report the sensitivity                                                            |        96% |
| Direct exponentiation is not a price-scale conditional-mean estimate        | Log-scale fitting plus \(\exp(\widehat y)\) generally targets a conditional geometric mean/median-like quantity                                       | §2.2 and Limitations                | Add smearing or alternative retransformation sensitivity                                                                             |        97% |
| \(\widetilde\rho=\rho\operatorname{Var}(y)\) is presented mainly for Direct | The same scaling applies to the Surrogate because \(e^2c^2\) also scales as \(a^4\)                                                                   | §3.4                                | State the normalization for both objectives, with qualification about other LightGBM regularizers                                    |        94% |
| Abstract is technically overloaded but empirically vague                    | It includes Hessian-interface detail but no sample size or magnitude                                                                                  | Abstract                            | Remove curvature detail; add sample sizes and one precise descriptive result                                                         |        94% |
| External-boundary language is repeated                                      | Scope is discussed in design, results, discussion, limitations, and Appendix H                                                                        | §§4.4, 5.7, 6, H                    | Keep one design sentence and one limitation paragraph; delete Appendix H                                                             |        99% |
| Appendix includes internal project-management material                      | Planned audits, versions, hashes, and future package requirements belong in a repository README                                                       | Appendices G–H                      | Convert completed items into reproducibility statements and move workflow notes outside the paper                                    |        99% |
| Sentence structure is overqualified                                         | Many sentences contain the claim, three caveats, design status, and non-claim simultaneously                                                          | Abstract, §§4–6                     | State result first; follow with one separate limitation sentence                                                                     |        96% |

CCAO’s published repository confirms that rolling-origin testing, held-out evaluation, assessor metrics, geography/property-class breakouts, and desk review are integral parts of its workflow. It also describes the model scope as excluding condominiums. These facts make the temporal alignment strong but the current scope and subgroup omissions important to resolve. See the [CCAO residential AVM documentation](https://github.com/ccao-data/model-res-avm).

## 5. Is the CCAO focus sufficiently clear?

### Empirically: yes

CCAO is unmistakably the central empirical application:

* the baseline problem is defined using CCAO data;
* the main sample contains 344,607 development sales, 38,290 held-out sales, and 26,641 2025 sales;
* the feature and LightGBM configuration are CCAO-motivated;
* the entire main Results section is CCAO;
* external evidence is excluded from inferential support; and
* CCAO staff are coauthors.

### Operationally: only partly

The paper establishes a CCAO-centered research translation, not a full CCAO implementation result. The remaining gaps are:

* exact extract/version not identified;
* condo/property scope potentially inconsistent;
* Python versus the official R/Tidymodels implementation not fully mapped;
* native/custom parity unresolved;
* no full-roll evaluation;
* limited township/property-class evidence;
* no assessor desk-review evidence;
* no selected model or production integration.

The manuscript should embrace the correct formulation:

> CCAO is the motivating institution, source of the empirical problem, and basis of the research workflow; the reported outputs are not official CCAO assessments or evidence of deployment.

That is already stated in places and should be consolidated, not repeated defensively throughout.

A subtitle could make the evidence boundary even clearer:

> Covariance-Guided Regularization for Price-Related Vertical Equity in Machine-Learning Mass Appraisal: Evidence from a Cook County Research Workflow

This is helpful but not strictly necessary.

## 6. Storyline assessment

### The strongest current storyline

1. CCAO’s LightGBM baseline improves prediction and horizontal uniformity relative to linear regression.
2. It nevertheless exhibits a stronger observed-sale price-related ratio pattern.
3. A classical log-log vertical-equity relationship can be translated into a training objective.
4. Direct and Surrogate implement that principle differently.
5. Moderate portions of the paths can improve several accuracy and equity-related measures together.
6. Stronger regularization reveals two limitations:

   * predictive and level deterioration;
   * nonlinear ratio-shape deformation, particularly for Surrogate.
7. Path direction is more stable than the exact penalty location.
8. Therefore the method is an auditable control knob, not an equity certificate or selected operating model.

This is a good paper. The negative findings in points 6–8 increase rather than decrease its research value.

### What currently obscures the story

* The Introduction spends too much space on the tax pipeline before presenting the empirical tension.
* Section 2 becomes a metric encyclopedia.
* The full-path idea, transition-span idea, candidate-screen idea, and external-benchmark idea compete for priority.
* The Results foregrounds screening endpoints with six-decimal precision even though no operating point is selected.
* The Discussion repeatedly restates the same qualifications.
* Tracked revisions make the compiled paper look internally contradictory.

## 7. Recommended main-body structure

A cleaner paper would be approximately 20–25 pages before references.

| Main section                        | Keep                                                                                           | Move out                                                           |
| ----------------------------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| 1. Introduction                     | CCAO problem, one baseline result, research question, contribution, boundaries                 | Long tax-system explanation and repeated non-claims                |
| 2. Literature and conceptual bridge | Classical vertical-equity tests; correction methods; fair regression; exact gap                | General AVM survey detail                                          |
| 3. Method                           | \(e=\log(\widehat P/P)\), \(C(f)\), Bayes caveat, Direct, Surrogate, mechanisms                | Fixed-space proofs and detailed LightGBM derivatives               |
| 4. CCAO empirical design            | Sample, temporal split, features, fixed learner, path design, primary diagnostics              | Exact grids, anchor mapping, screening algorithm, artifact history |
| 5. Results                          | Baseline tension; path efficacy; nonlinear failure mode; temporal instability                  | Candidate-region mechanics, regret tables, every metric trajectory |
| 6. Discussion                       | Interpretation, CCAO implications, limits, next evaluation                                     | Ten-step operational checklist and repeated external caveats       |
| Online appendix                     | Complete metrics, proofs, implementation audit, all path tables, uncertainty, subgroup results | Internal TODOs and obsolete ATTOM history                          |

### Main results should communicate only three findings

1. **Baseline finding:** prediction quality alone does not eliminate the CCAO ratio trend.
2. **Intervention finding:** both objectives can move the first-order and assessor-facing measures toward neutrality; moderate Direct configurations show the clearest descriptive joint improvement.
3. **Boundary finding:** no portable raw-\(\rho\) operating point is established, and Surrogate demonstrates how first-order improvement can coexist with nonlinear distortion.

Everything else should support those three findings.

## 8. Literature positioning

The Related Work is current and reasonably broad, but historically incomplete.

It correctly distinguishes the present approach from:

* quantile/spatial correction, such as [Quintos’s quantile-regression approach](https://researchexchange.iaao.org/jptaa/vol11/iss4/3/);
* the application-specific [\(K\)-segment valuation model](https://arxiv.org/abs/2312.05996);
* general fair-regression and dependence regularization;
* and post-hoc recalibration.

It also correctly incorporates recent evidence that accuracy and assessment equity need not be universally opposed. Smith et al.’s large-scale results emphasize that better information can improve both, making the manuscript’s “fixed information, change only the objective” design a controlled complement—not a universal prescription. See [Smith et al. 2026](https://arxiv.org/html/2605.15020v1).

The literature section should be reorganized around four solution classes:

| Literature class                                   | What it changes                    | Relationship to this paper               |
| -------------------------------------------------- | ---------------------------------- | ---------------------------------------- |
| Better property, spatial, and temporal information | Conditional valuation model        | Complementary and potentially preferable |
| Segmentation, quantile, and local models           | Model architecture/local structure | More flexible but more disruptive        |
| Post-hoc recalibration                             | Existing predictions               | Closest missing comparator               |
| Training-time dependence regularization            | Loss/objective                     | The paper’s chosen intervention          |

That comparison would make the research gap much clearer than the present serial review.

## 9. Primary versus secondary contributions

### Primary contributions

1. **CCAO problem translation:** turning a classical observed-sale vertical-equity slope into a train-time control within a realistic assessor-motivated workflow.
2. **Direct–Surrogate distinction:** showing algebraically and empirically that the Direct signed global target and the Surrogate tail-weighted loss are different interventions.
3. **Temporal CCAO evidence and failure audit:** documenting joint improvements, implementation limitations, nonlinear deformation, and weak operating-point portability.

### Secondary achievements

* fixed-space rank-one and weighted-projection results;
* the Bayes residual-covariance warning;
* normalized penalty coordinates;
* \(\Delta_{\mathrm{NL}}\) as a failure diagnostic;
* complete path infrastructure;
* 2025 forward evaluation;
* reproducibility manifests and audit machinery;
* external benchmark infrastructure.

### Not presently contributions

* the candidate-region screen as a generic method;
* external validity;
* deployment readiness;
* superiority to calibration;
* jurisdiction-wide or demographic equity;
* a validated penalty-selection procedure.

## 10. Highest-quality revision order

### P0: Required before scientific circulation

1. Generate a clean manuscript with no visible revision history, obsolete blocks, duplicated captions, or contradictory paragraphs.
2. Delete the active six-county sentence and consolidate the external-validity boundary.
3. Complete native/custom \(\rho=0\) parity and regenerate every affected artifact.
4. Rename or explicitly qualify the empirical Direct family as diagonal-curvature Direct.
5. Add the centered-spread post-hoc path.
6. Verify condo/property scope and exact CCAO extract provenance.
7. Add paired uncertainty or remove significance-like formatting.
8. Freeze the exact finite-sample dCor and \(\Delta_{\mathrm{NL}}\) implementations.

### P1: Required for a strong paper

9. Reframe the novelty as training-time translation of a classical log-log vertical-equity relationship.
10. Add historical vertical-equity literature.
11. Reduce the main metric hierarchy.
12. Move candidate-screen details completely to the supplement.
13. Add township/property-class/value-range/time subgroup diagnostics.
14. Run equal-date and repeated-parcel split audits.
15. Add retransformation sensitivity.
16. Replace “correction,” “safe region,” and “attainable frontier” with appropriately bounded terminology.

### P2: Valuable extensions, but not required for the present paper

17. Retune each objective family to estimate fuller attainable frontiers.
18. Compare richer information, regularization, and their combination.
19. Complete the normalized multi-jurisdiction benchmark audit.
20. Evaluate structured geographic/time/class penalties and structured post-processing.
21. Evaluate full-inventory transport and assessor desk-review implications.

## Final assessment

The project’s main intellectual asset is no longer merely that a penalty can improve several metrics. Its most valuable result is that an apparently simple equity-motivated objective has a transparent but narrow mechanism, can provide descriptive Pareto improvements in CCAO, and also has identifiable statistical and operational failure modes.

That is a publishable story.

The manuscript is currently strongest in its mathematics, scope discipline, temporal design, and failure analysis. It is weakest in the rendered presentation, causal attribution to the penalty, comparator completeness, uncertainty, and the correspondence between countywide research evidence and CCAO’s local operational evaluation.

My recommendation would be **major revision with a strong presumption in favor of the project’s core contribution**. After the P0 and P1 corrections, the paper should be presented as a CCAO-centered applied-methods and diagnostic paper—not as a general fairness solution and not as a penalty-selection paper.
