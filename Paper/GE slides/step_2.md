I did another full pass against the latest paper, the older slide deck, the revised deck draft, your preliminaries proposal, and the fairness lecture framing. These are the core sources I re-checked:

As design guardrails, I would explicitly follow four outside rules: prefer images over text, keep to about four bullets or fewer, reveal objects only when they are discussed and de-emphasize them afterward, and visually cue the most important part of a graph or figure. The Steve Jobs-style guidance points in the same direction: tell a story, use very simple slides, group ideas in threes when possible, and explain the story behind numbers instead of dumping raw tables. ([UC San Diego Multimedia][1])

## Final critical verdict

Your revised deck is **substantially better** than the old one. It is now much more aligned with the current paper’s actual message: property taxes matter, mass appraisal makes ML natural, stronger ML can worsen vertical equity, a covariance-based penalty is a lightweight correction layer, and deployment should be framed as constrained calibration. That high-level story is right.

But it is **not yet final-quality**. There are still five important issues.

First, the revised deck is still too text-heavy in several places. The main problem is no longer the overall section structure; it is slide density. Slides 2, 5, 7, 10, 13, and 19 still ask the audience to read too much while listening.

Second, the deck is still mixing **conceptual slides** with **speaker-note slides**. Theorems, full equations, and detailed calibration notation are still sitting in the main deck where they should mostly live in the appendix or in oral narration.

Third, the deck still has a few **internal consistency problems**. The biggest one is that the main body uses the latest paper numbers, but Appendix A2 still carries older baseline results from the old deck, and the model-selection appendix still looks more final than the current paper text actually is. That has to be cleaned up before this becomes a real presentation.

Fourth, the revised deck still has not fully decided which penalty variant is the **flagship operational result**. The current paper says the surrogate is the recommended deployment variant, while the revised main-result slide currently showcases a covariance-penalty example. That hierarchy should be made explicit and consistent.

Fifth, the model-selection section is better than before, but it is still slightly too formal for the current maturity of the paper. The paper still contains placeholders around the final calibration-constraints table and selected-model summary, so the main deck should keep constrained calibration conceptual and operational, not overly algebraic or overly final.

## The most important fixes still needed

### 1. Decide the main empirical flagship once

Right now, the paper’s current message is:

* direct penalty = closest conceptual target
* surrogate = recommended practical default

The deck should mirror that exactly.

So the method story should be:

* explain the direct penalty first because it is intuitive
* then say the surrogate is the operational version because it is separable and plug-and-play
* use the **surrogate** as the main deployment headline unless you change the paper

Do not let the main deck imply that the direct penalty is the recommended routine solution if the paper says otherwise.

### 2. Remove all old-number contamination

This is the highest-priority mechanical fix.

Your revised main deck is already using the current-paper baseline tension and headline result. But Appendix A2 still shows the old baseline table with (R^2=0.860), MAE $73,398, and older penalized variants. That is not consistent with the latest paper framing. Either update Appendix A2 fully or remove it until the numbers are frozen.

Same issue for the model-selection appendix: it is useful, but it currently looks more settled than the paper text itself.

### 3. Demote the “translation mismatch” caveat

That warning made sense in the old working deck, but it should not be visible in the main body now. Keep it as:

* a tiny appendix note
* a footnote on the CCAO slide
* or an oral caveat during Q&A

It should not occupy main-story real estate.

### 4. Reduce math in the main deck

The main audience does not need:

* the full squared-covariance formula
* the full surrogate formula
* the constrained-calibration optimization problem
* the Bayes theorem note on the main regressivity slide

Those are all useful, but they belong in appendix or oral narration. Main slides should stay conceptual.

### 5. Replace tables with visual summaries in the main deck

The revised deck is still too table-based in the core results. The audience will remember:

* one ratio plot
* one before/after comparison
* three metric cards
* one frontier plot
* one feasible-region selection diagram

They will not remember a 7-column table.

## Final full-deck outline I would recommend

I would now make the main deck **20 slides including Q&A**, with a much sharper “one message per slide” discipline.

### Part I. Why this problem matters

1. Title
2. Why property valuation matters
3. How valuation shapes the tax bill
4. Why assessors need mass appraisal
5. Why ML is attractive here
6. CCAO as the running case

### Part II. What fairness means here

7. Fairness in assessment: ratio, horizontal, vertical
8. Regressivity in taxpayer terms
9. Regressivity in diagnostic terms
10. The baseline tension: better ML, worse vertical equity

### Part III. What we propose

11. Main idea: add a regressivity penalty
12. Why covariance in log-space
13. Direct vs surrogate: conceptual vs operational
14. Why this fits assessor workflows

### Part IV. What changes empirically

15. Before/after: the trend flattens
16. Main payoff: vertical-equity metrics recover
17. Tradeoff frontier: accuracy vs equity
18. Deployment rule: constrained calibration

### Part V. Closing

19. Main takeaway + honest scope
20. Questions

### Appendix

A1. Metric definitions
A2. Full up-to-date results table
A3. Full formulas and boosting implementation
A4. Alternative selection criteria
A5. Extra diagnostics and robustness plots

That is the leanest version that still preserves the paper’s full story.

## Slide-by-slide final outline

## 1. Title

**Message:** Better ML can worsen vertical equity; a lightweight penalty can recover much of the gap.

What to show:

* title
* subtitle
* names
* one-line thesis under title

Keep:

* exactly as a clean opening
* no extra bullets

---

## 2. Why property valuation matters

**Message:** This is a high-stakes public-finance problem.

What to show:

* one big number block: 72%, 47%, $500B
* one simple services visual: schools, roads, safety, transit

What to say:

* property taxes fund local services
* assessment quality affects real burden distribution

Change from current draft:

* keep the numbers
* remove most of the explanatory text
* let the visual do the work

---

## 3. How valuation shapes the tax bill

**Message:** Valuation is upstream of the final tax burden.

What to show:

* 4-step flow:
  assess value → exemptions → tax rate → tax bill

What to say:

* distortions in assessment propagate into distortions in tax burden

Change from current draft:

* no explanatory bullet list under the flow
* at most one short takeaway sentence

---

## 4. Why assessors need mass appraisal

**Message:** Few homes sell, but all homes need values.

What to show:

* sold vs unsold parcel-map style visual
* 3–4 feature icons around parcels

What to say:

* assessors infer unsold values using common data and standardized methods

Change from current draft:

* this should be mostly picture + one definition, not bullets

---

## 5. Why ML is attractive here

**Message:** The problem naturally favors flexible tabular ML.

What to show:

* one clean visual contrast:
  linear structure vs nonlinear structure
* tiny icons for linear regression and boosted trees

What to say:

* many features, nonlinear interactions, large parcel inventories
* ML improves predictive fit but not automatically equity

Change from current draft:

* mention neural networks only verbally or in appendix
* keep the main contrast to linear vs tree-based models

---

## 6. CCAO as the running case

**Message:** This is a real assessor workflow, not a toy example.

What to show:

* one compact case-study card:
  1.5M parcels operational context
  349,661 verified sales
  95 features
  2016–2022 training
  2023 test
  rolling-origin validation

What to say:

* same diagnostics assessors actually use
* real deployment context

Change from current draft:

* remove any visible “mismatch” warning from the main body

---

## 7. Fairness in assessment

**Message:** The key fairness object is the assessment ratio.

What to show:

* ( r=\widehat P/P )
* one horizontal-equity mini visual
* one vertical-equity mini visual

What to say:

* horizontal equity = consistency among similar homes
* vertical equity = fairness across value tiers

Change from current draft:

* keep the ratio definition
* keep the illustrations
* remove appendix-style note from the slide itself

---

## 8. Regressivity in taxpayer terms

**Message:** Regressivity means burden shifts across price tiers.

What to show:

* not a table
* instead, a left-right house infographic or 5-tier strip
* low tier highlighted in red, high tier in green/blue
* percent over/under assessment annotations

What to say:

* low-value homes over-assessed
* high-value homes under-assessed
* this is the financial content behind the technical metric

Change from current draft:

* replace the current quantile table with a more visual infographic
* keep the 50%-higher-effective-tax-rate idea in the narration or a small callout

---

## 9. Regressivity in diagnostic terms

**Message:** The ratio-vs-price plot is the key diagnostic picture.

What to show:

* baseline LightGBM ratio-vs-log-price plot
* over-assessed region
* under-assessed region
* negative trend line

What to say:

* if the trend slopes downward, assessments are regressive

Change from current draft:

* remove the Bayes-predictor note from the bottom of the slide
* mention that orally or move it to appendix
* this slide should be fully visual

---

## 10. The baseline tension: better ML, worse vertical equity

**Message:** This is the central contradiction that motivates the method.

What to show:

* not the full table
* instead, two side-by-side model cards:

  * Linear Regression
  * LightGBM
* green arrows on (R^2), MAE, COD/COV
* red arrows on PRD, PRB, VEI

What to say:

* LightGBM predicts much better
* but it becomes more regressive

Change from current draft:

* replace the dense table with metric cards or arrow callouts
* keep the IAAO compliance ranges as a small subtitle or footer

---

## 11. Main idea: add a regressivity penalty

**Message:** This is a one-knob correction layer.

What to show:

* old paradigm vs new paradigm
* loss only → loss + ( \rho \cdot \Psi )

What to say:

* start from a strong learner
* add one dependence penalty
* tune (\rho) to balance accuracy and equity

Change from current draft:

* keep the structure
* reduce the text to three short lines
* no full notation beyond the symbolic objective

---

## 12. Why covariance in log-space

**Message:** Penalizing covariance targets the trend we care about.

What to show:

* one visual chain:
  residual (e=\hat y-y=\log r)
* one arrow to “ratio drift with price”

What to say:

* in log-space, residuals are log-ratios
* covariance with log-price is a natural first-order trend target

Change from current draft:

* keep only the identity
* remove extra bullets
* make it feel intuitive, not algebraic

---

## 13. Direct vs surrogate: conceptual vs operational

**Message:** There are two variants, but only one main hierarchy.

What to show:

* two clean boxes:

  * direct = closest conceptual target
  * surrogate = separable, plug-and-play, operational default

What to say:

* direct gives the cleanest conceptual picture
* surrogate is the implementation-friendly one recommended for routine use

Change from current draft:

* avoid formulas in the main deck
* move formulas to appendix
* make the recommended hierarchy explicit

---

## 14. Why this fits assessor workflows

**Message:** This is not a replacement pipeline.

What to show:

* simple pipeline graphic
* one highlighted added block: penalty / calibration

What to say:

* same preprocessing
* same base learner
* same diagnostics
* one extra hyperparameter

Change from current draft:

* strong slide already
* just trim text further

---

## 15. Before/after: the trend flattens

**Message:** The penalty changes the exact pattern it was designed to change.

What to show:

* before/after ratio plots
* one small arrow line with slope/correlation moving toward zero

What to say:

* the ratio–price relation becomes flatter after penalization

Change from current draft:

* keep only one penalty example in the main body
* if you want to compare direct vs surrogate visually, do it in appendix

---

## 16. Main payoff: vertical-equity metrics recover

**Message:** Most of the vertical-equity gap is recovered while predictive fit stays nearly flat.

What to show:

* three big metric cards:
  PRD 1.085 → 1.038
  PRB -0.118 → -0.011
  VEI -36.8% → -8.5%
* one smaller card:
  (R^2) essentially unchanged

What to say:

* this is the main empirical headline
* mention COD/COV only briefly

Change from current draft:

* remove the table
* keep the cards
* make the slope of the story visual: bad → near-compliant

---

## 17. Tradeoff frontier

**Message:** There is a frontier, not one universally best model.

What to show:

* one panel only in the main deck, preferably PRD vs (R^2)
* compliance band shaded
* baseline point, penalized frontier, selected region

What to say:

* increasing (\rho) traces a meaningful frontier
* the real question is which model best satisfies assessor tolerances while keeping strong fit

Change from current draft:

* move the second panel to appendix
* this slide should have one graph, not two

---

## 18. Deployment rule: constrained calibration

**Message:** Deployment should use the strongest predictive feasible model.

What to show:

* conceptual diagram:
  candidate pool → feasible region → selected model
* maybe one simple sentence:
  “best predictive candidate subject to ratio-study tolerances”

What to say:

* same logic assessors already apply through compliance diagnostics
* not “best one score,” but “best among feasible candidates”

Change from current draft:

* remove the optimization equation from the main slide
* move it to appendix
* keep the operational statement in plain English

---

## 19. Main takeaway + honest scope

**Message:** The method works, but its scope is specific.

What to show:

* left side: 3-line takeaway
* right side: 3-line limitations

Takeaway:

* stronger ML can worsen vertical equity
* covariance penalization can recover much of that gap
* it is lightweight enough for real workflows

Limitations:

* not direct PRD/PRB/VEI control
* no out-of-sample guarantee
* still global / first-order

Change from current draft:

* merge takeaway and limitations
* remove the translation caveat from this main closing slide
* keep the close crisp

---

## 20. Questions

**Message:** Clean finish.

What to show:

* “Questions?”
* one small footer: implementation / diagnostics / deployment / extensions

---

## What to move out of the main deck

These items are good, but they are not main-story slides:

* full metric formulas
* Bayes-predictor theorem note
* exact squared-covariance and surrogate equations
* constrained-calibration optimization formula
* alternative Pareto rules
* large multi-model result tables
* extensive appendix selection tables
* nonlinear-dependence extension material

Those are appendix or backup only.

## Full plan to make the slides efficient

This is the design plan I would actually apply to the whole deck.

### Rule 1: One message per slide

Every slide should answer one question only.

Bad:

* “What are property taxes, how does taxation work, why does ML help, and why is fairness important?”

Good:

* “Why does valuation matter?”
* “Why do assessors need mass appraisal?”
* “Why can stronger ML still be unfair?”

### Rule 2: Replace explanatory text with speaking text

If a sentence is something you would naturally say, it probably should not also sit as full text on the slide.

Slides should contain:

* titles
* one takeaway sentence
* one visual
* short labels

Your notes or your memory should contain the full explanation.

### Rule 3: Prefer pictures, process diagrams, and metric cards over tables

In the main deck:

* use parcel maps, workflow flows, house infographics, ratio plots, frontier plots, and metric cards
* use tables only in the appendix

### Rule 4: Use progressive disclosure

Reveal one element at a time:

* in the tax flow slide
* in the mass-appraisal parcel slide
* in the baseline tension cards
* in the metric payoff cards

This will help both the audience and the presenter.

### Rule 5: Keep color semantics fixed

Use one consistent language:

* red = regressivity / worsening
* green = improvement / compliance
* blue = method / penalized model
* gray = baseline context

### Rule 6: Put the story in the transitions

The transitions should sound like this:

* “Valuation matters because it affects tax burdens.”
* “But not every home sells, so assessors must estimate at scale.”
* “That makes ML attractive.”
* “But stronger ML creates a fairness problem.”
* “So we add a correction layer.”
* “Then we choose among candidate models using assessor-style tolerances.”

That is smoother than jumping between section names.

### Rule 7: Put all math beyond the first identity in the appendix

Main deck:

* at most (r=\widehat P/P)
* at most (e=\hat y-y=\log r)
* at most loss + ( \rho\Psi )

Everything else:

* appendix

### Rule 8: Tell the story behind numbers

Whenever you present a number:

* convert it into a contrast
* or convert it into a decision
* or convert it into a burden implication

Example:
Do not show “PRD = 1.038” alone.
Show “PRD: 1.085 → 1.038, moving toward the IAAO target.”

## Final list of changes to add, why, how, and solution

1. **Unify all numbers across main deck and appendix.**
   Why: the deck still mixes old and new results.
   How: refresh Appendix A2 and A4 from the latest paper outputs or temporarily remove them.
   Solution: one canonical results source for all tables and metric cards.

2. **Decide the flagship variant explicitly.**
   Why: the paper says the surrogate is the operational default, but the current main-result slide still spotlights a covariance-penalty example.
   How: present direct as conceptual, surrogate as deployment default.
   Solution: one sentence on Slide 13 and one consistent empirical headline on Slide 16.

3. **Replace dense tables in the main deck with cards or arrows.**
   Why: current tables slow reading and weaken smoothness.
   How: use 3–4 big metric cards and visual arrows.
   Solution: keep full tables only in appendix.

4. **Remove theorem-level content from the main narrative.**
   Why: it increases cognitive load and hurts flow.
   How: move Bayes/theorem note and formulas to appendix.
   Solution: keep only the intuitive identity and the one-line method objective in the main deck.

5. **Make Slide 8 fully visual.**
   Why: the current taxpayer-terms slide is still table-like.
   How: replace quantile table with a house/quantile infographic.
   Solution: use one low-tier home and one high-tier home, or a 5-tier strip.

6. **Simplify the baseline tension slide.**
   Why: it is the most important problem slide and should be remembered instantly.
   How: use two model cards instead of an 8-column table.
   Solution: keep just (R^2), MAE, PRD, PRB, VEI.

7. **Reduce the tradeoff slide to one graph in the main deck.**
   Why: two panels split attention.
   How: keep PRD vs (R^2) main; VEI vs (R^2) appendix.
   Solution: one message, one graph.

8. **Make constrained calibration plain-English in the main deck.**
   Why: the optimization statement is too formal for the current presentation goal.
   How: use “best predictive feasible model” and a feasible-region diagram.
   Solution: move the formal equation to appendix.

9. **Demote the translation caveat.**
   Why: it undermines the main story if shown prominently.
   How: footnote or appendix note only.
   Solution: main body stays confident and clean.

10. **Add progressive-reveal builds throughout.**
    Why: they improve smoothness and reduce overload.
    How: reveal pipeline steps, then candidate models, then metric cards, then selected feasible region.
    Solution: cleaner pacing for both presenter and audience.

## Final recommendation

The deck is now close. The high-level structure is right. The remaining work is mostly **presentation surgery**, not story surgery.

The strongest final version is:

* a **visual-first preliminaries section**
* a **single clean statement of the method**
* a **result section built around plots and metric cards**
* a **plain-English deployment rule**
* a **clean close with limited, honest scope**

That will make the presentation much easier to follow, much easier to deliver, and much more faithful to the current paper.

[1]: https://multimedia.ucsd.edu/best-practices/presentation-design.html "Evidence-Based Presentation Design Recommendations"
