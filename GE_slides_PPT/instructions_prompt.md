Here is the critical pass.

The attached review outline is **mostly right**, but not every point has the same weight. The strongest insight in it is that the new preliminaries are no longer failing at **story logic**; they are now failing mostly at **implementation fidelity**. That is the correct diagnosis. The paper’s actual core setup is: property taxes matter, mass appraisal is operationally necessary, the CCAO workflow is realistic, and the main hinge is that moving from linear regression to LightGBM improves fit but worsens vertical equity.  

## Critical review of the review outline

### The truly necessary, high-confidence changes

These are the ones I would treat as mandatory.

**1. Replace placeholder visuals on Slides 6 and 8.**
This is the most important point in the review, and I agree with it fully. Slide 6 is supposed to define assessor fairness visually, and Slide 8 is supposed to land the main regressivity intuition. If either is still placeholder text, the preliminaries remain a wireframe rather than a real section. The source recommendations already say Slide 6 should use two mini-panels for horizontal versus vertical equity, and Slide 8 should use the actual ratio-vs-log-price figure as the core visual.   

**2. Remove raw LaTeX leakage.**
Also fully correct, and high-priority. The current beamer necessarily contains TeX syntax, but the generated PPTX should not display raw `\approx`, `\text{}`, or similar. Slide 2 is especially vulnerable because the source recommendation explicitly includes the tax formula in TeX form, which is fine as a source spec but not as literal slide text.  

**3. Fix Slide 5’s factual conflation around 349,661 and the 2016–2024 timeline.**
This is a real correctness issue, not just polish. The current paper says the translated CCAO workflow uses **349,661 verified sales over 2016–2023**, with **95 features**, and separately describes the training/validation, test, and assessment-year structure.  The attached recommendation is also clear that the slide should present **349,661 filtered sales**, **95 predictors**, and the **2016–2022 / 2023 / 2024** temporal split, while avoiding the rough “>350K / ~200 raw features” wording from the current beamer.  The current beamer still uses the rough version, so this is definitely a necessary correction. 

**4. Keep Slide 7 anchored to the latest paper baseline numbers and make the hinge visually stronger.**
This is also high-confidence. The paper explicitly gives the updated baseline comparison: linear regression at (R^2=0.762), MAE $86,257, PRD 1.039, PRB -0.011, VEI -13.474%; LightGBM at (R^2=0.883), MAE $72,064, PRD 1.085, PRB -0.118, VEI -36.816%.  The old beamer still uses the earlier snapshot with 0.798 versus 0.860 and different metric values, so the review is right that the generated preliminaries are better aligned with the current paper here.   The part I would soften slightly is this: making the table a native editable PPT table is highly desirable, but not as important as using the correct numbers and visual emphasis.

**5. Tighten Slide 3 wording and add one concrete scale fact.**
I agree this is a real improvement, but it is slightly less urgent than the four above. The recommendation to soften “Most homes never sell in a given year” to something like “Only a minority of homes sell in a given year, yet every home must be valued” is sensible, because the paper strongly supports the mass-appraisal necessity logic without asserting that exact stronger phrasing. The paper also gives the scale fact that the CCAO produces residential valuations for roughly **1.5 million parcels** on a triennial cycle, which is perfect support for making the slide feel operational rather than abstract.  

### Important but slightly secondary

These matter, but they are not the first things I would fix.

**6. Preserve the cleaner notation (r=\widehat P/P) instead of older AV/MV-heavy notation.**
This is a good point, and I agree with the direction. The paper consistently uses the cleaner assessment-ratio notation (r_i=\widehat P_i/P_i), and the preliminaries recommendation also frames Slide 6 around that cleaner notation.   The current beamer still uses AV/MV on the regressivity slide, so preserving the updated notation is the right move.  This is not as urgent as fixing placeholders or factual mismatches, but it is definitely worth preserving.

**7. Add a tiny explanatory line on Slide 8 about the x-axis being log-price because prices are highly skewed.**
This is a small but good improvement. The paper states exactly that log-price is used because sale prices are highly right-skewed and log-price makes the trend comparable across value levels.  So this is evidence-based and worth adding, ideally as a very short subtitle or footnote rather than a full sentence block.

**8. Make Slide 7 visually scream “accuracy up, vertical equity down.”**
Yes, but this is about presentation strength rather than factual correction. The preliminaries recommendation already says the best design is a green/red delta table highlighting that the stronger predictor exits the compliance region on the dimension the office most cares about.  That is worth doing, but only after the underlying content and numbers are correct.

### Useful, but not strictly necessary right now

These are real ideas, but I would not prioritize them ahead of the others.

**9. Decide whether the preliminaries file is standalone or just an inserted section.**
This is valid, but it is mostly a packaging decision. The attached recommendation had already argued that the best preliminaries are an **8-slide sequence**, not necessarily a standalone mini-deck.  So the existence of an 8-slide content-only section is not inherently a problem. I would only force this decision now if the Cursor project is ambiguous in a way that affects layout or file naming.

**10. Add the “up to 50% higher effective tax rates” fact as a speaker-note hook.**
Good idea, but optional. The paper does explicitly mention that lower-valued property owners can face effective tax rates up to 50% higher than higher-valued owners.  This is excellent narrative material, but it belongs in notes or oral delivery, not on-slide. Since the user asked about the deck itself and the Cursor project, this is not one of the most urgent changes.

## My final priority ranking

If I compress this into the actual order I would implement:

1. Replace Slide 6 and Slide 8 placeholders with real visuals.
2. Remove all raw LaTeX from PPT text.
3. Fix Slide 5’s wording so the 349,661 count is not conflated with the 2016–2024 timeline.
4. Lock Slide 7 to the current paper baseline numbers and emphasize the tradeoff visually.
5. Soften Slide 3’s title and add the 1.5 million parcels fact.
6. Preserve (r=\widehat P/P) notation consistently.
7. Add the small log-price rationale note on Slide 8.
8. Only then worry about standalone-vs-insert packaging and speaker-note hooks. 

---

## Step-by-step guide to incorporate these changes in the Cursor project

I’m assuming your Cursor project already has the layered workflow we discussed:

* `.cursor/rules/`
* `brief.md`
* `storyboard.yaml`
* `deck.json`
* `src/generate_deck.js`
* `assets/`
* `sources/`

The key principle is: **fix each issue at the correct layer**. Do not patch everything directly in the generator.

## Phase 1: lock the project rules before editing content

### Step 1. Open the project and create a safe working branch

In Cursor:

* Open the project folder.
* Open the terminal.
* Create a branch for this pass, something like:

```bash
git checkout -b prelim-fidelity-fixes
```

This matters because you are now moving from broad outline design into source-of-truth and rendering corrections.

### Step 2. Update the project rule file

Open `.cursor/rules/slide-style.mdc` and add these instructions if they are not already there:

```md id="u5xgy8"
- Never leave placeholder text on final slides.
- Never render raw LaTeX syntax as visible slide text.
- Prefer current paper over old beamer when numbers differ.
- Use clean assessment-ratio notation r = P_hat / P.
- For preliminaries, Slides 6 and 8 must contain real visuals, not descriptions of visuals.
- Slide 5 must distinguish pre-2024 modeling sample from 2024 assessment period.
```

Why here: this stops Agent from reintroducing the same failure patterns in later edits.

### Step 3. Add a project workflow rule

In `.cursor/rules/presentation-workflow.mdc`, add:

```md id="3lk0e1"
- For factual slide corrections, update storyboard.yaml and deck.json before touching generate_deck.js.
- For rendering-only fixes, modify generate_deck.js only after the content spec is correct.
- Do not invent or infer dataset counts beyond the current paper.
- If a figure is referenced in the paper or beamer, try to insert the real asset rather than describing it in text.
```

This prevents the project from sliding back into code-first patching.

---

## Phase 2: fix the content layer first

### Step 4. Update `brief.md`

Open `brief.md` and add a short “current correction priorities” section:

```md id="5r7v8p"
## Current correction priorities
1. Replace placeholder visuals on Slides 6 and 8
2. Remove raw LaTeX text
3. Fix Slide 5 wording around 349,661 and 2024
4. Use current paper baseline numbers on Slide 7
5. Soften Slide 3 title and add CCAO scale fact
```

Why: this gives Agent the current target without you repeating it in every prompt.

### Step 5. Revise `storyboard.yaml`

This is the highest-leverage file to edit next.

Open `storyboard.yaml` and change these entries.

#### Slide 3

Change the title from any strong wording like “Most homes never sell in a given year” to:

```yaml id="7q1g1d"
title: "Only a minority of homes sell in a given year, yet every home must be valued"
```

Then add in `on_slide_text` or `speaker_notes`:

```yaml id="x5zlsi"
speaker_notes: "Cook County values roughly 1.5 million parcels on a triennial cycle, so valuation cannot rely only on observed sales."
```

This is grounded in the paper’s workflow description. 

#### Slide 5

Revise the slide so the count and timeline are not fused in the title. I would use:

```yaml id="wilk4t"
title: "Cook County gives us a realistic assessor workflow to study"
```

Then inside the facts box:

```yaml id="bi1ti4"
on_slide_text:
  - "349,661 verified sales in the pre-2024 modeling sample"
  - "95 predictors"
  - "Train/validate: 2016–2022 | Test: 2023 | Assessment: 2024"
```

This is better than “349,661 sales, 95 features, 2016–2024” because it respects the paper’s actual structure.  

#### Slide 6

Replace any placeholder description with an actual visual spec:

```yaml id="x0mq8s"
title: "Assessors care about fairness both within tiers and across tiers"
layout_type: "split_panel_equity"
key_visual: "left: horizontal equity mini-sketch; right: vertical equity mini-sketch"
on_slide_text:
  - "Horizontal equity: similar homes should receive similar ratios"
  - "Vertical equity: ratios should not systematically fall with value"
```

Do not add metric formulas here. The attached recommendation explicitly says Slide 6 should remain metric-light. 

#### Slide 7

Make sure the table content uses the current paper numbers, not the old beamer snapshot. Write them directly in the structured content:

```yaml id="xm5q1k"
on_slide_text:
  - "Linear Regression: R² 0.762 | MAE $86,257 | PRD 1.039 | PRB -0.011 | VEI -13.5%"
  - "LightGBM: R² 0.883 | MAE $72,064 | PRD 1.085 | PRB -0.118 | VEI -36.8%"
speaker_notes: "Accuracy improves, but vertical equity worsens sharply; this is the central empirical motivation."
```

These numbers match the current paper. 

#### Slide 8

Make the key visual explicit and force the real asset:

```yaml id="peoazm"
title: "The baseline model shifts tax burden across value tiers"
layout_type: "annotated_regressivity_plot"
key_visual: "real ratio-vs-log-price plot from source assets"
on_slide_text:
  - "Lower-value homes tend to be over-assessed"
  - "Higher-value homes tend to be under-assessed"
  - "x-axis is log-price because prices are highly skewed"
```

Again, this is directly grounded in the paper and the slide recommendations.  

---

## Phase 3: fix the render spec

### Step 6. Update `deck.json`

Once `storyboard.yaml` is correct, regenerate or manually update `deck.json`.

There are three main fixes to enforce here.

#### A. Remove raw LaTeX from visible text fields

Anywhere you see content like:

* `\approx`
* `\text{}`
* `\widehat P`
* TeX display blocks

replace visible text with plain slide text, for example:

* `≈`
* `Predicted value`
* `P̂`
* `Property tax ≈ (Assessed value - exemptions) × tax rate`

A good rule is: **if it must render as text, make it Unicode/plain text**. If you want equation styling, the generator should create a formatted text box, not dump raw TeX.

#### B. Force real asset usage on Slides 6 and 8

In `deck.json`, the visual blocks for Slides 6 and 8 should not be generic descriptions. They should reference either:

* a real asset path in `assets/`, or
* a named render function in the generator that draws the figure.

For example:

```json id="0r5uxt"
{
  "slide_number": 8,
  "layout_type": "annotated_regressivity_plot",
  "visual": {
    "type": "image",
    "asset": "assets/figures/baseline_regressivity_plot.png"
  }
}
```

#### C. Turn Slide 7 into an editable table spec

Do not leave Slide 7 as an image-only snapshot if you can avoid it. In `deck.json`, define the columns and rows explicitly, for example:

* Model
* R²
* MAE
* COD
* PRD
* PRB
* VEI

Then let the generator render it as a native PPT table. This is not the first priority, but it is the correct medium-term fix.

---

## Phase 4: gather the missing assets

### Step 7. Create or collect the actual visuals for Slide 6 and Slide 8

Open the `assets/` folder and add:

* `assets/figures/baseline_regressivity_plot.png` or `.pdf` export converted to a usable image
* `assets/figures/equity_mini_sketches.svg` or equivalent for Slide 6

For Slide 8, the real plot already exists in the paper / source slides and is clearly central to the project. The current paper uses the baseline LightGBM regressivity figure and explicitly says the assessment ratio declines with property value, with lower-value properties over-assessed and higher-value properties under-assessed. 

For Slide 6, you may not have a ready-made asset. In that case, do not leave placeholder text. Either:

* draw it directly in the PPT generator using shapes, or
* make a small SVG/PNG manually and place it in `assets/figures`.

The easiest robust option is to **draw Slide 6 directly in code** using simple houses / rectangles / labels / arrows.

---

## Phase 5: fix the generator only after the spec is correct

### Step 8. Open `src/generate_deck.js`

Now patch the rendering layer.

You want three concrete render changes.

#### A. Add a `sanitizeText` helper

Make a helper function that:

* converts `\approx` to `≈`
* strips `\text{...}` into plain content
* replaces `\widehat P` with `P̂`
* replaces `\times` with `×`
* removes stray `{}` and backslashes when needed

Then apply it to all visible text blocks before rendering.

This will kill the raw LaTeX leakage globally.

#### B. Add a real render function for Slide 6

Implement something like `renderSplitPanelEquity(slide, spec)` that draws:

* left panel: three similar homes, inconsistent ratios vs consistent ratios
* right panel: low-value to high-value homes with a downward ratio trend

This directly matches the attached recommendation and is much better than trying to insert descriptive placeholder text. 

#### C. Add a real render function for Slide 8

Implement `renderAnnotatedRegressivityPlot(slide, spec)` that:

* inserts the actual plot image from `assets/figures/baseline_regressivity_plot.png`
* adds a small notation box with `r = P̂ / P`
* adds two small burden cards:

  * lower-value home → over-assessed
  * higher-value home → under-assessed

That matches the recommendation almost exactly. 

### Step 9. Make Slide 7 a stronger hinge

In the generator:

* use green highlighting for accuracy improvements
* use red highlighting for worse vertical-equity metrics
* optionally add a small subtitle bar: “Stronger fit, worse vertical equity”

Even if you do not fully rebuild it as a native table immediately, at least enforce the right colors, spacing, and emphasis.

---

## Phase 6: use Cursor Agent surgically

### Step 10. Use targeted Agent prompts, not broad ones

Inside Cursor Agent, run prompts in this order.

First:

```text id="v6uxu3"
@brief.md
@storyboard.yaml
Revise only Slides 3, 5, 6, 7, and 8 in storyboard.yaml to match the current paper and current correction priorities. Do not change slide order.
```

Then:

```text id="kej5gn"
@storyboard.yaml
@deck.json
Update deck.json so that:
- Slides 6 and 8 require real visuals
- Slide 5 distinguishes the pre-2024 modeling sample from the 2024 assessment year
- Slide 7 uses the current paper baseline numbers
- visible text contains no raw LaTeX
```

Then:

```text id="g3ad8e"
@deck.json
@src/generate_deck.js
Patch generate_deck.js only for rendering:
- sanitize visible text
- render Slide 6 with shapes/diagrams
- render Slide 8 with the real regressivity plot asset
- keep slide layouts otherwise unchanged
```

That sequence keeps the workflow disciplined.

---

## Phase 7: regenerate and QA

### Step 11. Rebuild the PPTX

In the Cursor terminal:

```bash
node src/generate_deck.js
```

### Step 12. Run a structured QA check in Cursor chat

Use:

```text id="ub81m3"
@storyboard.yaml
@deck.json
Audit the preliminaries section for these specific issues only:
1. Any remaining placeholder text
2. Any raw LaTeX in visible text
3. Any mismatch between Slide 5 wording and the paper’s sample/timeline structure
4. Any use of old baseline numbers on Slide 7
5. Any missing real visual on Slide 8
Return only a short issue list.
```

Then patch only what still fails.

---

## Phase 8: what not to spend time on yet

Do not spend your next iteration on these before the must-fix items are resolved:

* deciding standalone vs inserted section
* adding the 50%-higher tax-rate hook visibly on a slide
* fancy transitions or animation
* extra metrics on Slide 6
* polishing appendices

Those are lower leverage than fixing fidelity and correctness.

---

## Final recommended execution order

If you want the shortest best path inside Cursor, do this exact sequence:

1. Update `.cursor/rules/` to prevent placeholders and raw LaTeX.
2. Edit `brief.md` to list the current correction priorities.
3. Edit `storyboard.yaml` for Slides 3, 5, 6, 7, and 8.
4. Update `deck.json` so Slides 6 and 8 require real visuals and Slide 7 uses current numbers.
5. Add the missing assets under `assets/figures/`.
6. Patch `generate_deck.js` to sanitize text and render the missing visuals.
7. Rebuild the PPTX.
8. Run a focused QA prompt in Cursor.
9. Only after that, decide whether to add a section-opener slide or speaker-note hooks.

That order best matches the real state of the deck: the storytelling backbone is already much better, so now you should prioritize **fidelity, correctness, and visual intuition delivery**.  

If you want, next I can turn this into exact ready-to-paste contents for:

* `storyboard.yaml` for slides 3–8
* `deck.json` entries for slides 5–8
* a `generate_deck.js` patch plan for the three render fixes
