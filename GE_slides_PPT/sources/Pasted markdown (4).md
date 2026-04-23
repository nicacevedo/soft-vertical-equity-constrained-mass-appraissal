After re-checking the current deck, paper, your preliminaries idea, and the code, my final view is:

**the best preliminaries section is an 8-slide sequence that ends at the baseline tension**: why property taxation matters, why mass appraisal exists, why ML is used, what fairness means for assessors, why the stronger model creates the problem, and what regressivity looks like. That is more intuitive than the current deck, and better aligned with the paper’s real core message. The paper’s strongest setup is the LR→LightGBM tension, the assessor-side equity framing, and the cleaned CCAO workflow with about **349,661 observations**, **95 predictors**, and the **2016–2022 / 2023 / 2024** temporal split.   

## Final preliminaries deck

### Slide 1 — **Why this matters: property taxes affect everyone**

**Goal:** civic relevance in one slide.

**Discuss**

* Property taxes are a major local-revenue source in the U.S.
* They fund schools, roads, transit, safety, and related services.
* The key intuition: if assessments are systematically biased, the tax burden is systematically shifted. 

**Include**

* One big headline number block:

  * “~72% of local tax revenue”
  * “~47% of local own-source general revenue”
  * “~$500B/year”
* One short sentence under it: “So valuation errors scale into real distributional consequences.” 

**Do not include**

* A Boston/NYC/Chicago comparison table.
* Too much local tax-rate detail.

**Visual**

* Clean infographic, not a table:

  * left: house/property icon,
  * middle: “assessed value”,
  * right: icons for school / road / bus / safety.
* Subtitle: “Property valuations are not just predictions; they shape public finance.”

---

### Slide 2 — **How taxes are computed: where assessment enters**

**Goal:** make the mechanism concrete.

**Discuss**

* Step 1: assessor estimates market value.
* Step 2: exemptions / base adjustments.
* Step 3: local tax rate.
* Step 4: final tax bill.

**Include**

* One simple formula only:
  [
  \text{Property Tax} \approx (\text{Assessed Value} - \text{Exemptions}) \times \text{Tax Rate}
  ]
* One small Chicago callout only if desired: “Chicago effective rates are materially larger than many other cities,” but do not build the slide around city rates. The current slides mention Chicago roughly 2–2.3%, but this should stay secondary. 

**Visual**

* Horizontal 4-step flow diagram with arrows.
* Use one example property card moving through the pipeline.

**Critical note**

* Keep this slide operational, not legalistic.

---

### Slide 3 — **Why assessors must estimate most homes**

**Goal:** this is the missing bridge in the current deck.

**Discuss**

* Only a small fraction of properties transact in a given year.
* Yet every property still needs an assessment.
* So assessors infer values for the unsold majority using observed sales plus common features.

**Include**

* One short definition:

  * **Mass appraisal:** estimating the value of many properties at once using common data and standardized methods. This is directly aligned with the paper’s definition. 

**Visual**

* Best visual in the preliminaries:

  * a stylized city/parcels map or rectangular grid of properties;
  * about 5% highlighted as “sold / observed price”;
  * the remaining majority shown as “unsold / must be estimated.”
* Under the map, a few feature tags:

  * size,
  * age,
  * bedrooms,
  * neighborhood,
  * distance to transit,
  * nearby amenities.

**Design note**

* This slide should visually explain the whole task in 5 seconds.

---

### Slide 4 — **Why ML is a natural tool for mass appraisal**

**Goal:** justify ML without turning this into a generic ML talk.

**Discuss**

* Many properties, many features, nonlinear interactions.
* Assessors have moved from simpler linear tools to stronger nonlinear models.
* In this project, the meaningful comparison is **Linear Regression vs LightGBM**, not a broad model zoo. The paper and current deck are built around that tension.  

**Include**

* A two-column comparison:

  * **Linear Regression**: simple, interpretable, limited nonlinear structure.
  * **Tree Ensembles / LightGBM**: stronger fit on complex tabular data.
* Mention neural nets in one phrase only; do not diagram them.

**Visual**

* Minimal “model evolution” graphic:

  * sold homes/features → LR,
  * sold homes/features → boosted trees,
  * arrow labeled “more flexible.”
* Or a simple icon pair: line vs tree ensemble.

**Do not include**

* “Sale price is only a proxy” here.
* Interpretability/fairness concerns yet; those belong next.

---

### Slide 5 — **Cook County case study: the workflow we analyze**

**Goal:** ground the story in one real assessor pipeline.

**Discuss**

* This project studies the Cook County Assessor’s Office residential workflow.
* The repository is open and operationally relevant.
* We follow the translated 2025-cycle residential pipeline.  

**Include**

* One compact facts box:

  * residential Cook County workflow,
  * ~349,661 filtered sales,
  * 95 predictors,
  * train/validation: 2016–2022,
  * test: 2023,
  * assessment year: 2024. 
* One sentence: “This is a realistic assessor-style temporal setup.”

**Visual**

* Left: small Cook County / Chicago locator map.
* Right: compact timeline strip:

  * 2016–2022 train/validate,
  * 2023 test,
  * 2024 assessment.

**Critical note**

* Use the **cleaned modeling-ready numbers**, not the rough “>350K / ~200 raw features” wording from the current slides, unless you explicitly separate raw repository features from final predictors.  

---

### Slide 6 — **What fairness means for assessors**

**Goal:** define the fairness language in the assessor’s terms.

**Discuss**

* Assessors care about two main forms of equity:

  * **Horizontal equity**: similar properties should be treated similarly.
  * **Vertical equity**: fairness across different value tiers.
* This talk focuses on **vertical equity**, especially regressivity. The paper centers the analysis on standard ratio studies. 

**Include**

* Very short definitions only.
* One line introducing the ratio:
  [
  r = \widehat P / P
  ]
  where ( \widehat P ) is assessed/predicted value and (P) is sale price. 

**Do not include**

* Full formulas for PRD, PRB, VEI, COD on this slide.
* A big metric table.

**Visual**

* Two mini-panels:

  * left: three similar houses with very dispersed ratios → “bad horizontal equity,” then same houses with tight ratios → “good horizontal equity.”
  * right: low-tier to high-tier houses with downward trend in ratio → “bad vertical equity.”

**Optional footer**

* “Standard diagnostics later: COD, PRD, PRB, VEI.”

---

### Slide 7 — **Baseline tension: better ML, worse vertical equity**

**Goal:** this is the crucial bridge into the method.

**Discuss**

* In the CCAO case, LightGBM improves predictive performance relative to linear regression.
* But it worsens the main vertical-equity diagnostics.
* This is the practical problem the paper solves. The paper states this sharply; the current deck already has the table, but it should appear in preliminaries, not later.  

**Include**

* One compact comparison table with only the most important metrics:

  * (R^2),
  * MAE,
  * COD,
  * PRD,
  * PRB,
  * VEI.
* Highlight:

  * accuracy gets better,
  * regressivity gets worse.

**Visual**

* Best design: a “green/red delta” table.

  * Green arrows for (R^2), MAE, COD improvements.
  * Red arrows for PRD, PRB, VEI worsening.
* Add one sentence under the table:

  * “The stronger predictor exits the assessor compliance region on the dimension the office most cares about.” This is almost exactly the paper’s framing. 

**Critical note**

* This slide is strictly necessary. Without it, the penalty appears as a generic fairness add-on rather than a targeted fix to a concrete workflow problem.

---

### Slide 8 — **What regressivity looks like**

**Goal:** land the intuition visually and financially.

**Discuss**

* Regressivity means lower-value homes tend to be over-assessed and higher-value homes under-assessed.
* The current LightGBM ratio-vs-log-price figure is still the best visual definition.
* This is the last preliminaries slide; it should naturally set up the method section.  

**Include**

* Main panel: ratio vs log-price plot with:

  * (r=1) line,
  * lowess trend,
  * labeled over-assessed and under-assessed regions.
* Minimal notation box:

  * (AV): assessed value,
  * (MV): market value,
  * (r=AV/MV).

**Add one financial interpretation panel**

* Small right-side or bottom callout:

  * “For lower-value homes: ratio > 1 → taxed as if worth too much.”
  * “For higher-value homes: ratio < 1 → taxed as if worth too little.”
* Use **two tiers only**: low-value vs high-value.
* Do not use a dense 5-quantile table unless it is extremely clean.

**Visual**

* Preferred layout:

  * 70% of slide: the existing scatter/lowess plot.
  * 30%: two simple house cards:

    * low-value home → “over-assessed,”
    * high-value home → “under-assessed.”
* Keep the message visual, not metric-heavy.

---

## Final critical pass

After one more review, I would make only these strict choices:

* **Do not split Slide 1 and Slide 2** into two separate “importance” and “mechanism” decks; the flow is better as two compact slides only.
* **Do not add a generic fairness-in-ML lecture slide**; keep fairness assessor-specific.
* **Do not overuse city comparisons**; national importance + Cook County case study is enough.
* **Do not overload metrics early**; one ratio definition early, one compact benchmark table, the full metric details later or in appendix.
* **Do not omit Slide 7**; the LR-vs-LGBM tension is the real hinge of the whole story.
* **Do not omit Slide 3**; the estimation-necessity slide is the main missing intuition in the current deck.

So the final preliminaries section should be:

1. Why property taxes matter
2. How assessment enters the tax bill
3. Why assessors must estimate most homes
4. Why ML is a natural tool
5. Cook County case study
6. What fairness means for assessors
7. Better ML, worse vertical equity
8. What regressivity looks like

That is the version I would build.
