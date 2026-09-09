# Tier B — Final major revision of `paper/paper_v17_option1.tex`

## Context

P0 (scientific validation) and P1 (inferential reporting) are frozen and certified.
Tier B0 built a deterministic evidence bridge that specifies, claim by claim and
number by number, what the manuscript may say. Tier B0 wrote **no** prose and touched
**no** file under `paper/`. This plan is the execution plan for that writing pass.

The manuscript's problem is not that it is wrong everywhere. It is that it mixes three
kinds of content that currently look identical to a reader:

1. **fully resolvable** results (Cell A and Cell C at ρ=0; Direct/Surrogate at the two
   frozen anchors ρ≈0.954095 and ρ=100; PRB and VEI at all five anchors);
2. **unresolvable** results — 356 active numeric tokens, concentrated in exactly eight
   anchors, which no frozen artifact reproduces;
3. **frozen results the manuscript never reports at all** — the A/B/C reference
   convention, the centered-spread comparator, matched-β, temporal robustness
   (D-SNAP / D-PURGE / D-UNSEEN / G5a / G5b), ED2 Step 4–7 inference, and Duan
   smearing sensitivity.

Intended outcome: a CCAO-centered applied-methods and diagnostic paper on controlling a
**first-order component** of price-related regressivity in ML mass appraisal, auditing
the predictive, nonlinear, temporal and standards-facing consequences — with every
printed number resolving through the frozen Tier-B0 maps, and with the closest prior art
(currently invisible) engaged explicitly.

### Verified starting state (read-only, this session)

| check | result |
|---|---|
| worktree | `/orcd/home/002/nacevedo/RA/soft-vertical-equity-paper-write` |
| branch | `paper-major-revision-write` |
| HEAD | `904976451bdd01aa3e6c8b59d4eee77d6f10ba74`, tree clean |
| `tier-b0-final-20260907` | resolves (`730fd7fc`, annotated → commit `90497645`), **ancestor of HEAD** |
| `p0-major-revision-final-20260907` | `805c426e`, **ancestor** |
| `p1-inferential-reporting-final-20260907` | `6caf6677`, **ancestor** |
| canonical manuscript | `paper/paper_v17_option1.tex`, 4,389 lines, 388,448 bytes |
| 8 plain figure replacements | all present on disk **and tracked in git** |

---

## 0. The two governing rules

Everything below is downstream of two rules. Getting either wrong invalidates the pass.

**Rule 1 — Reference-cell semantics (A and C are two roles, not a substitution).**

| cell | forward display name | frozen role | purpose |
|---|---|---|---|
| **A** | Ordinary LightGBM (standard raw-label native) | assessor-facing / workflow benchmark | `WORKFLOW_BENCHMARK` |
| **B** | Centered-label native L2 (initialization-aligned) | implementation-decomposition control only, no full path | `WITHIN_MODEL` |
| **C** | Custom-objective rho=0 origin | **PRIMARY** within-path penalty-isolating reference | `PENALTY_ISOLATING` |

The fix for the path tables is **not** to replace A with C. Keep the A-referenced
comparison (labelled a workflow-benchmark comparison) **and add** the C-referenced
penalty-isolating comparison as primary. Concretely: **both A and C appear as explicit
reference rows**, and two visibly different markers are used — **bold** for improvement
against A, **†** for improvement against C. No asterisk, because it reads as statistical
significance. Say which is which in the notes in words, naming both cells by forward
display name, and state that neither marker denotes significance, compliance, model
selection, or a recommended ρ. Never present an A-referenced difference as a pure penalty
effect; never call C a workflow benchmark; never leave the C comparison inferable only
from prose or a footnote.

**Rule 2 — Provenance, numeric and visual, is the same rule.**
An unsupported number is not publishable because it is relabelled "exploratory". A figure
whose *graphic* draws the unsupported candidate-region construction cannot be `KEEP`,
even if its caption numbers resolve.

A subtlety the writing pass must not trip over: the candidate-region ρ endpoints
(0.355648, 2.559548, 0.202359, 2.222996, 6.866488) are *not* individually flagged in the
coverage audit, because they exist in the frozen grid maps as **grid coordinates**. The
flagged tokens at `subsec:path_stability_results` are only the LOFO indices (53, 56, 53,
55). This does **not** license the prose: the claim that those coordinates *are* the
activity onset and upper guardrail is what no artifact reproduces
(`C-results-012`, `C-appendix-004`, `C-results-021`, `C-appendix-009`). Token-level
allowlisting of a grid coordinate is not support for the claim built on it.

---

## A. Proposed final storyline

> Machine-learning mass appraisal can be highly accurate and still price-regressive.
> In a research translation of the CCAO residential LightGBM workflow, the regressive
> pattern has a first-order component that is exactly the classical log-ratio/log-price
> vertical-equity slope, because on a fixed evaluation sample
> `beta_log = Cov(e,y)/Var(y)`. That component is directly controllable during training.
> We give two LightGBM-compatible objectives that control it — a Direct
> squared-covariance penalty (exact gradient; only the diagonal of a dense rank-one
> Hessian) and a Jensen-derived sample-additive Surrogate that is exactly weighted
> squared error — and we show they are *different mechanisms*, not two implementations of
> one.
>
> We then audit the consequences rather than declaring victory. (i) The ρ=0 controls are
> not numerically identical to ordinary LightGBM, so only the custom-objective ρ=0 origin
> isolates a penalty effect. (ii) Post-hoc centered rescaling of the same unregularized
> predictor reproduces or improves much of Direct's first-order accuracy/equity frontier
> over the Direct-attainable β range — and the accuracy comparison *crosses over* with
> correction strength: post-hoc beats Direct at moderate correction and loses badly to
> Surrogate at strong correction. (iii) At matched development β_log, Direct, Surrogate and
> C-posthoc are *not* interchangeable; over the evaluated path Direct did not attain the
> three extended targets that a one-parameter rescaling reaches exactly, and log-scale
> accuracy can reverse the price-level-accuracy impression. (iv) Under alternative temporal
> designs the primary conclusion stands, with one gate in ALERT that a denser grid did not
> confirm. (v) Standards-facing inference separates the adopted 2013 Standard (PRB) from
> the May-2026 Exposure Draft (VEI) — and shows that a VEI point estimate outside ±10% is
> a trigger, not a finding. (vi) A retransformation sensitivity shows which quantities are
> invariant, which scale, and which move.
>
> First-order neutrality is not fairness, not conditional unbiasedness, not independence,
> and not a flat conditional ratio profile. No penalty strength is selected anywhere.
> The contribution is the mass-appraisal translation, the Direct/Surrogate distinction,
> the joint β_log / Δ_NL / dCor audit, and the failure-mode-explicit temporal validation —
> not residual–outcome decorrelation, which is prior art.

Eight sentences the paper must be able to survive being reduced to:

1. Accuracy alone does not remove the CCAO ratio trend (Cell A: held-out `beta_log`
   −0.150, PRB −0.091, VEI −26.5%, COD 21.6% — outside the adopted [5,15] range; and on
   2025 the PRB 95% CI lies **entirely** outside ±0.10).
2. The targeted first-order component moves as intended along both paths, referenced to
   Cell C (held-out `beta_log` −0.147 at C → −0.079 Direct / −0.001 Surrogate at ρ=100).
3. It moves *unevenly across diagnostics*: at ρ=100 Direct's `beta_log` is −0.079 while
   Surrogate reaches −0.001, yet Surrogate's `RMSE_log` degrades from 0.289 to 0.357.
4. At matched first-order correction the three families are not interchangeable, and at
   the strongest attainable common target Direct is the *worst* of the three on price
   accuracy and COD.
5. Over the evaluated frozen path, Direct **did not attain** β_log ∈ {−0.06, −0.03, 0} and
   Surrogate did not attain 0, while both post-hoc families attained all three exactly.
   Those four NOT_ATTAINED states are reported, never interpolated — and never generalized
   into a claim about unexamined ρ values.
6. A one-parameter post-hoc centered rescaling is therefore a serious competitor on the
   first-order axis — which is why no superiority claim for either family is made.
7. The paths are not a simple monotone accuracy-for-equity exchange, and no configuration
   is selected.
8. Everything printed resolves to a frozen artifact; what does not, is gone.

---

## B. Proposed section structure

Front matter and Methods keep their shape. **Results is reorganized** around the frozen
evidence hierarchy instead of the historical five-anchor / candidate-region architecture.

| # | section | action |
|---|---|---|
| 1 | Introduction | REWRITE (last, Tier B3) |
| 1.1 | Related Work | **REWRITE + ADD prior-art paragraph** (Tier B1, first) |
| 2 | Preliminaries: the CCAO baseline and empirical motivation | KEEP structure |
| 2.1–2.2 | Workflow; prediction setting and baseline models | KEEP |
| 2.3 | Predictive, assessor-facing, and residual-structure diagnostics | UPDATE (guidance status per row) |
| 2.4 | Baseline comparison and empirical motivation | UPDATE (Cell A re-source; Linear decision) |
| 3 | Covariance-guided regressivity correction | KEEP structure |
| 3.1–3.3 | Target; Direct; Surrogate | KEEP + ADD ESS consequence |
| 3.4 | Interpretation, validation, and scope | **REWRITE (E.4 = INDETERMINATE)** |
| 4 | Application and evaluation design | KEEP structure |
| 4.1–4.2 | Pipeline/data; temporal validation and path design | UPDATE (extract hash; D1/D2/D3 statement) |
| 4.3 | Models and comparisons | **ADD the A/B/C convention + D1/D3 coordinates** |
| 4.4 | Scope of external evidence | KEEP verbatim |
| 5.1 | Accuracy alone does not remove the CCAO ratio trend | KEEP + UPDATE |
| 5.2 | *The ρ=0 reference cells and what they permit* | **ADD** (short; table stays in App. E.3) |
| 5.3 | Regularization paths: prediction and assessor-facing vertical equity | **REBUILD + SPLIT** (`tab:path_anchor_frozen` / `tab:path_anchor_standards`) |
| 5.4 | First-order mechanism, nonlinear shape, and broader dependence | **REWRITE** (merges old 5.3 + 5.4) |
| 5.5 | Accuracy–equity trajectories | KEEP |
| 5.6 | *What retraining buys over post-hoc centered rescaling* | **ADD** (centered-spread comparator) |
| 5.7 | *Matched-β_log comparison: Direct vs Surrogate vs C-posthoc* | **ADD** (central) |
| 5.8 | *Temporal robustness under alternative designs* | **ADD**, absorbing the repaired old §5.6 |
| 5.9 | *Standards-facing inference: PRB (adopted 2013) and VEI (May-2026 Exposure Draft)* | **ADD** |
| 5.10 | External-validity boundary | KEEP verbatim |
| 6 | Discussion and conclusions | REWRITE (Tier B3) |
| 6.1 | Practical implementation considerations | UPDATE (drop screen steps 6–7 wording) |
| 6.2 | Limitations | **REWRITE** (six frozen limitations added) |
| 6.3 | Future work | KEEP |

Section moves / merges:

- **Old §5.3 (ratio shape) merges into new §5.4.** Both are the same argument — that
  first-order neutrality does not imply a flat profile — and old §5.3's second half is
  built on the deleted CV screen.
- **Old §5.6 ("Chronological stability…") is replaced by new §5.8.** Its over-suppressed
  `oldrevisionblock` is repaired sentence by sentence (never unwrapped), and its active
  candidate-region block (L2912–2918) is deleted.
Inside Appendix F ("Primary CCAO Regularization-Path Details") — note these are the
*manuscript's* appendix subsections, not this plan's §F:

- **"Generic CV Candidate-Region Screen" is deleted entirely.**
- **"Cross-Metric Turning Points and Temporal Concordance" is deleted**, along with both
  its tables and its three figures.
- **New subsection: "Matched-β_log targets and attainment"** — full CORE/EXT tables,
  D1 primary and D3 sensitivity.
- **New subsection: "Centered-spread post-hoc comparator detail"**.
- **New subsection: "Alternative temporal designs"** — D-SNAP / D-PURGE / D-UNSEEN tables.
- **Appendix A gains** the ED2 Step 4–7 procedure with counting units, and the Duan
  smearing sensitivity (`C-appendix-002`, `C-appendix-003`).
- **Appendix G gains** the certification statement (`C-certification-001/002`).
- **Appendix E.3** keeps `tab:rho_zero_control` and gains the parity-audit *finding*
  replacing the two `\todo` promises.
- The three `\iffalse` regions (2081–2098, 2925–2993, 4320–4370) are **deleted**, which
  also removes the inert duplicate `\label{app:attom}`.

---

## C. Table/figure architecture after Tier B0.1

Tier-B0 totals: 20 tables + 19 figures → **13 DELETE, 12 KEEP, 2 REBUILD, 12 UPDATE**.

### Tables — final set

| label | action | note |
|---|---|---|
| `tab:assessment_metrics_summary` | UPDATE | add a guidance-status column: 2013 **ADOPTED** (PRD, PRB, COD, level ratios) vs May-2026 **EXPOSURE DRAFT** (MKI, VEI). COV range is a derived approximation — keep it labelled as such. |
| `tab:ccao_baseline_results` | UPDATE | re-source LightGBM from `zero_control_full.csv` cell A; **Linear columns deleted** (all 16 flagged tokens are Linear) → four columns become two. Label as workflow-benchmark. |
| `tab:ccao_samples` | KEEP | cite `PREFLIGHT.json` for the four counts. |
| `tab:feature_groups` | KEEP | say plainly these describe the upstream CCAO pipeline, not P0/P1-validated counts. |
| `tab:ccao_design` | KEEP | keep the two-stage grid history (50 + 32, not 82 prespecified). |
| `tab:path_anchor_summary` | **REBUILD → SPLIT** | 65 flagged tokens. Becomes **`tab:path_anchor_frozen`** (two frozen anchors: prediction + PRD + MKI, tier R1) and **`tab:path_anchor_standards`** (PRB + VEI at all five anchors, 2013-adopted vs ED2-proposed labelled separately). Two-device highlighting on both. |
| `tab:path_anchor_complementary` | **REBUILD** | 119 flagged tokens. Six resolvable complementary metrics at the two frozen anchors only; Δ_NL and dCor unsupported at **every** positive ρ — those columns dropped for positive ρ. Same two-device convention. |
| `tab:ccao_baseline_complementary` | UPDATE | re-source cell A (all 8 metrics incl. Δ_NL, dCor); **Linear columns deleted** (18 flagged tokens); state COD outside adopted [5,15]. |
| `tab:rho_zero_control` | KEEP | replace both `\todo`s with the parity finding + normalization-audit result. |
| `tab:fold_structure` | KEEP | add the fold-6/fold-7 validation-overlap finding. |
| `tab:roles` | KEEP | load-bearing novelty boundary — survives intact. |
| `tab:rho_candidate_regions` | **DELETE** | 12/12 flagged. |
| `tab:transition_summary` | **DELETE** | 22 flagged. |
| `tab:transition_regret` | **DELETE** | 100 flagged — largest single concentration. |
| 3 × `tab:attom_*` (IFFALSE) | **DELETE** | inert; do not revive. |
| 3 commented copies (`tab:assessment_metrics_summary`, `tab:path_anchor_summary` ×3, `tab:path_anchor_complementary`) | **DELETE** | archive preserves them; **do not mine them for values**. |
| **NEW** `tab:matched_beta` | ADD | D1 primary / D3 sensitivity; four NOT_ATTAINED states verbatim. |
| **NEW** `tab:centered_spread` | ADD | C-primary, A-secondary; B has no full path. |
| **NEW** `tab:temporal_robustness` | ADD | D-SNAP / D-PURGE / D-UNSEEN + G5a / G5b verdicts as recorded. |
| **NEW** `tab:ed2_steps` (App. A) | ADD | Step 4–7 with counting unit named on every count. |
| **NEW** `tab:smearing_sensitivity` (App. A) | ADD | SENSITIVITY only. |

### Figures — the candidate-region asset swap (8 assets across 7 figures)

All eight plain replacements verified present and git-tracked.

| figure | referenced today | replace with |
|---|---|---|
| `fig:mechanism_path_placeholder` | `mechanism_vs_rho_candidate_region.pdf` | `mechanism_vs_rho.pdf` |
| `fig:other_metric_paths_placeholder` | `predictive_metric_paths_candidate_region.pdf` | `predictive_metric_paths.pdf` |
| `fig:other_metric_paths_placeholder` | `level_uniformity_paths_candidate_region.pdf` | `level_uniformity_paths.pdf` |
| `fig:vertical_equity_metric_paths` | `vertical_equity_metric_paths_candidate_region.pdf` | `vertical_equity_metric_paths.pdf` |
| `fig:cv_predictive_metric_paths` | `cv_predictive_metric_paths_candidate_region.pdf` | `cv_predictive_metric_paths.pdf` |
| `fig:cv_level_uniformity_paths` | `cv_level_uniformity_paths_candidate_region.pdf` | `cv_level_uniformity_paths.pdf` |
| `fig:cv_vertical_equity_metric_paths` | `cv_vertical_equity_metric_paths_candidate_region.pdf` | `cv_vertical_equity_metric_paths.pdf` |
| `fig:cv_mechanism_metric_paths` | `cv_mechanism_metric_paths_candidate_region.pdf` | `cv_mechanism_metric_paths.pdf` |

For each: swap the asset, **delete every** candidate-region / activity-onset /
upper-guardrail / transition-span sentence from the caption, and **keep** the path curves
and the genuine metric-definition reference lines (PRD=1, PRB=0, MKI=1, VEI=0,
β_log=0, ratio=1).

**Figures DELETED** (defined by the unsupported construction, no plain counterpart):
`fig:transition_event_locations`, `fig:vertical_equity_event_locations`,
`fig:mechanism_event_locations`, `fig:ratio_shape_cv_transition_span_only`.

**Figures KEPT**: `fig:baseline_motivation`, `fig:accuracy_equity_placeholder`, the four
`fig:tradeoff_*`. **`fig:ratio_shape_path_placeholder`**: keep the asset, strip
candidate-region/guardrail language from caption and neighbouring prose.

**`fig:vei_group_profile_placeholder` — UPDATE, the trickiest caption in the paper.**
Three different intervals sit at the same nominal 90% level and the caption distinguishes
none of them:

1. **ED2's inferential interval** — `ci_method = "ED2 App. D.2 rank-based order statistic
   (NOT the bootstrap)"`, `ci_level 0.9`, `z = 1.645`, ranks
   `j = ceil(1.645·sqrt(n)/2)` counted outward from the median; this is what drives
   Steps 6 and 7. Guarded by
   `test_ci_method_is_the_rank_based_order_statistic_not_the_bootstrap`.
2. **P1's bootstrap sensitivity** — `_bootstrap_group_ci(vals, n_boot=1000, ci=0.90,
   seed=2025)`, docstring *"The existing deterministic percentile bootstrap — SENSITIVITY
   ONLY"*, written only to four `_sensitivity`-suffixed columns, scope
   `("heldout","forward_2025")`, populated on exactly **63 rows** (30 held-out + 33 2025) —
   the escalating standards-facing cells, i.e. 63 display entries = 61 unique
   realizations. It *"never enters a decision."*
3. **What this figure actually draws** — `utils/motivation_utils.vei_percentile_group_profile`
   with `n_bootstrap=1000, ci=0.90, rng_seed=2025`: descriptive percentile-bootstrap
   intervals on the plotted group medians for the **Linear and ordinary-LightGBM
   baselines**. These carry **no ED2 verdict at all.**

**Why this is worse than it looks, and why the disclaimer must be explicit rather than
implied:** (2) and (3) share the *same* bootstrap parameters — 1000 replicates, 90%,
seed 2025 — and differ only in what they are computed on. And (1) sits at the same nominal
90%. So all three are numerically plausible readings of a bare "90% interval" beside a VEI
figure, and a reader will most naturally assume (1).

Required: label the ±10% band a *proposed exposure-draft* diagnostic. If the intervals
are retained, the caption must say explicitly that they are descriptive visualization
intervals on the plotted group medians, that they are **not** the ED2 App. D.2 rank-based
order-statistic interval driving Steps 6 and 7, and that they are **not** P1's bootstrap
sensitivity on the 63 escalating standards-facing cells either. Otherwise drop them.
Attach no ED2 verdict to this figure.

*Generate no new scientific figures from new experiments.*

---

## D. Commit sequence

Each stage is one commit, verified before the next. Rewriting front-to-back is
explicitly not the plan.

### D.0 Execution control — two mandatory stops

**This authorization covers B1.0 through B1.4 only.**

Execute B1.0 → B1.4 in order, running **all prescribed validation and a compile after every
commit** (§H.2 validator, §H.3(a) cumulative write-scope, §H.3(b) three tag diffs, §H.4
token count, `latexmk`). Then **STOP. Do not begin B2.1.**

At the B1 stop, report:

| # | item |
|---|---|
| 1 | the five commit hashes (B1.0 … B1.4) |
| 2 | `git status` |
| 3 | cumulative diff summary from `tier-b0-final-20260907` |
| 4 | Tier-B validator status |
| 5 | compile status |
| 6 | unsupported-token count remaining |
| 7 | exact files changed |
| 8 | a concise scientific summary of what changed in Related Work, Methods, and Design |
| 9 | any warnings, or any deviation from the approved plan |

Then **wait for explicit approval before B2.**

**A second mandatory stop after B2.6, before beginning B3.** Same report shape, covering
B2.1 → B2.6: the six commit hashes, `git status`, cumulative diff from the tag, validator
status, compile status, remaining unsupported-token count (expected **0** from B2.2
onward), exact files changed, a concise scientific summary of the rebuilt tables and the
four added Results subsections, and any warnings or deviations. Wait for explicit approval
before B3.

Expected token trajectory across B1: 356 at B1.0 (no manuscript edit), **218 after B1.1**,
and 218 through B1.2–B1.4 — none of those three stages touches a flagged anchor. A
different number at the B1 stop is a finding to report, not to reconcile silently.

Anything discovered during B1 that would change B2 is reported at the stop rather than
acted on.

### Tier B1 — foundations (5 commits)

| # | commit | contents |
|---|---|---|
| **B1.0** | `[Tier B1.0] Tier-B manuscript validator` | build `paper/paper_analysis/tier_b_validation/` (§H.2): the twelve checks, the Tier-B provenance ledger, and the three-tag immutability assertions (§H.3). Reads the frozen B0 maps; writes nothing under `analysis/`. Also record the informational pre-edit run of the frozen B0/P1 suites, and document which of their guards are expected to fail once the manuscript is edited. **No manuscript edit in this commit** — the gate exists before the work it gates. |
| B1.1 | `[Tier B1.1] Delete unsupported candidate-region and transition material` | delete `tab:rho_candidate_regions`, `tab:transition_summary`, `tab:transition_regret`, the 4 overlay figures, Appendix F's candidate-region-screen and turning-point subsections, **the active candidate-region block at L2912–2918** (its 4 LOFO-index tokens — see §H.4), the 3 `\iffalse` regions, and the commented table copies. **Do this first** so later edits are never written against numbers about to disappear. Removes **138** of 356 flagged tokens plus the inert ATTOM material. |
| B1.2 | `[Tier B1.2] Related Work: engage the prior-art boundary` | copy the 4 validated historical entries into `paper/references_additions.bib` (**no** third `\addbibresource`, no `analysis/` dependency); `Edelstein1979` → `pages = {753}`; `Cheng1974` → minimal core, no DOI; **cite the six uncited prior-art entries**; add the residual–outcome-decorrelation paragraph. |
| B1.3 | `[Tier B1.3] Methods: E.4 INDETERMINATE, Surrogate ESS, post-hoc root` | §3.4 rewrite; §3.3 ESS addition; the `b_star_train = Var_T(y)/Cov_T(f0,y)` correction. |
| B1.4 | `[Tier B1.4] Design: A/B/C convention, D1/D3 coordinates, metric guidance status` | §4.3 addition; §4.1 extract hash; `tab:assessment_metrics_summary` guidance column; the D1/D2/D3 statement. |

> **── MANDATORY STOP (§D.0). Report, then wait for explicit approval before B2.1. ──**

### Tier B2 — empirical core (6 commits)

| # | commit | contents |
|---|---|---|
| B2.1 | `[Tier B2.1] Split and rebuild the path tables under the two-role convention` | `tab:path_anchor_summary` → `tab:path_anchor_frozen` + `tab:path_anchor_standards`; rebuild `tab:path_anchor_complementary`; drop all Linear rows. Removes 184 flagged tokens. |
| B2.2 | `[Tier B2.2] Re-source Cell A in the baseline tables; remove the Linear columns` | `tab:ccao_baseline_results` + `tab:ccao_baseline_complementary` (4 cols → 2); reframe §1 and §2.4 as a level statement; drop the Linear curve from `fig:baseline_motivation` and the VEI-profile figure; drop Linear as a context anchor in the four `fig:tradeoff_*` figures. Removes 34 flagged tokens → **0 remaining**. |
| B2.3 | `[Tier B2.3] Add the ρ=0 audit result and the A/B/C consequence` | new §5.2; App. E.3 `\todo` → findings (#16 contrary finding, #17 closed). |
| B2.4 | `[Tier B2.4] Add the centered-spread comparator and the matched-β comparison` | new §5.6 + §5.7, plus the two new Appendix-F subsections that hold their full tables. |
| B2.5 | `[Tier B2.5] Rebuild §5.6 as temporal robustness` | new §5.8; sentence-by-sentence repair of the `oldrevisionblock` (restore the operating-point conclusion, **not** the materiality sentence); state τ = 0.002 positively; new Appendix-F "Alternative temporal designs" subsection. The candidate-region block is already gone from B1.1. |
| B2.6 | `[Tier B2.6] Standards-facing inference, smearing, figure asset swap` | new §5.9; App. A additions (ED2 Steps + smearing); 8 asset swaps + caption strips; the VEI-figure caption. |

> **── MANDATORY STOP (§D.0). Report, then wait for explicit approval before B3.1. ──**

### Tier B3 — framing (2 commits)

| # | commit | contents |
|---|---|---|
| B3.1 | `[Tier B3.1] Discussion, limitations, and workflow interpretation` | §6 rewrite; §6.2 six frozen limitations; §6.1 step list. |
| B3.2 | `[Tier B3.2] Introduction and Conclusion` | written *after* the body is stable. |

### Tier B4 — closure (3 commits)

| # | commit | contents |
|---|---|---|
| B4.1 | `[Tier B4.1] Abstract and title` | last prose written. |
| B4.2 | `[Tier B4.2] Remove revision scaffolding` | remove the Tier-A override block (preamble), 336 active `\latesttext` + 67 `\newtext` call sites (unwrapped to plain prose), 169 active `\oldtext` call sites (deleted — already non-printing), 2 active `oldrevisionblock` environments, remaining `\todo`s, and the commented / `\iffalse` regions. **Confirmed for this pass, and only after B4.1** — git history plus `superseded_text_archive.md` preserve everything. |
| B4.3 | `[Tier B4.3] Compile, certification appendix, adversarial full-paper audit` | Appendix G (`MANUSCRIPT_RELEVANT_CERTIFICATION` layer only); then the **final gate** — see §H.7. A frozen B0/P1 suite rerun is **not** part of it. |

---

## E. Per-change specification

Notation: **D** = disposition, **P** = priority, **Conf** = confidence.
`C-*` ids are `manuscript_claim_map.csv` rows; `N-*` are `manuscript_numeric_map.csv` rows.

### E.1 Related Work — the single largest integrity gap

**`C-related-001` · REWRITE · P0 · Conf HIGH · §1.1 `subsec:related_work` · Tier B1.2**

*Problem.* Two bibliography gaps make real content invisible, and I confirmed both
mechanically this session.

- **Uncited prior art.** Six entries sit in `references_additions.bib` — which *is*
  loaded — and are cited **zero** times in the active manuscript:
  `TrederEtAl2021`, `RenEtAl2026`, `LeeChen2026`, `WangEtAl2023`, `SmithEtAl2019`,
  `BeheshtiEtAl2019`. They do not print. `KomiyamaEtAl2018` is cited once;
  `SmithEtAl2026` once.
- **Uncitable historical entries.** `paper_v17_option1.tex:30-31` loads only
  `references.bib` and `references_additions.bib`. `PaglinFogarty1972`, `Cheng1974`,
  `Edelstein1979`, `SundermanEtAl1990` live in
  `paper/references_major_revision_additions.txt`, which no `\addbibresource` loads, so
  `\cite` of any of them prints nothing.

*Why it matters.* The six uncited entries are the **closest prior art to this paper's
mechanism**, and they are not in mass appraisal at all — they are the brain-age
prediction literature, which independently developed residual–outcome decorrelation:

| reference | what it already does | which novelty claim it forecloses |
|---|---|---|
| `SmithEtAl2019` (NeuroImage) | brain-age *delta* estimation; the residual–outcome dependence problem and its regression-based correction | residual–outcome decorrelation itself |
| `BeheshtiEtAl2019` (NeuroImage: Clinical) | post-hoc bias-adjustment scheme | affine residual orthogonalization |
| `TrederEtAl2021` (Front. Psychiatry) | **"Correlation Constraints for Regression Models: Controlling Bias in Brain Age Prediction"** | covariance/correlation control generally; generic tunable decorrelation paths |
| `WangEtAl2023` (IEEE TMI) | skewed loss function correcting predictive bias | target-dependent corrective losses generally |
| `RenEtAl2026` (ISBI) | age-delta *correlation loss* inside a nonlinear learner | generic nonlinear in-processing correction |
| `LeeChen2026` (arXiv stat.ME) | outcome-calibrated regression + predicted-outcome-based inference | the statistical framing of decorrelation |
| `KomiyamaEtAl2018` | coefficient-of-determination fairness constraint | dependence-constrained regression generally |

*Required final message.* Four paragraphs:

1. ML in mass appraisal — KEEP essentially as is.
2. Vertical equity and regressivity — KEEP, and **add** the four historical citations
   (`PaglinFogarty1972`, `Cheng1974`, `Edelstein1979`, `SundermanEtAl1990`) where the
   classical vertical-equity testing literature is discussed. Keep `SmithEtAl2026` on
   accuracy and equity not being intrinsically opposed.
3. Fair regression and dependence-based constrained learning — KEEP.
4. **NEW: residual–outcome decorrelation outside mass appraisal.** State plainly that the
   brain-age literature has already formulated residual–outcome correlation control as a
   training constraint, a tunable path, a post-hoc affine adjustment, a target-dependent
   loss, and a nonlinear in-processing correction; and that `LeeChen2026` gives the
   statistical framing. Then state the boundary: what is new here is the CCAO-centered
   mass-appraisal translation, the exact tie to the classical log-ratio/log-price slope,
   the LightGBM-compatible Direct implementation with its exact/approximate curvature
   structure, the Jensen-derived additive Surrogate, the Direct/Surrogate *diagnostic*
   distinction, the joint β_log / Δ_NL / dCor audit, the chronological + forward
   validation with explicit failure modes, and the workflow interpretation.

*Must disappear.* Any implication that covariance/correlation control, tunable
decorrelation paths, affine residual orthogonalization, target-dependent corrective
losses, or nonlinear in-processing correction are novel here. The existing
`\latesttext` non-novelty paragraph at L309 is a good start but does not name the
foreclosing literature — it must now cite it.

*Citations to add.* The 6 uncited entries (already in the loaded
`references_additions.bib` — they need `\cite`s, nothing else) plus the 4 historical ones.

*How the historical entries get loaded — no third `\addbibresource`, and no dependency on
`analysis/`.* **Copy** the validated entries into `paper/references_additions.bib`, which
`paper_v17_option1.tex:31` already loads, and cite them normally. The Tier-B0 staged file
`analysis/final_manuscript_evidence/bib/staged_historical_entries.bib` is
**provenance/input only** — the compiled manuscript must never depend on a bibliography
file under `analysis/`, and `paper/references_major_revision_additions.txt` stays unloaded.
The per-field provenance stays where it is, under the frozen B0 evidence layer
(`bib/STAGED_BIB_PROVENANCE.md` + `bib/metadata_cache/`), and is referenced from the
replication appendix rather than reproduced in the paper. *(Corrected from an earlier draft
of this plan, which proposed a third `\addbibresource` pointing at the staging file.)*

*Bib hygiene — invent nothing.*
- **`PaglinFogarty1972`** and **`SundermanEtAl1990`** verify field-by-field against cached
  Crossref responses. Paste complete.
- **`Edelstein1979`**: paste with `pages = {753}` only. The start page 753 is Crossref-
  confirmed; the **end page 768 is confirmed by no authoritative source**, so it is omitted
  rather than carried.
- **`Cheng1974`**: Crossref returned no matching record, so **every** field rests on the
  in-repo transcription and **no DOI exists** — none may be invented. Paste the minimal
  citable core only (`author`, `title`, `journal`, `year`) and omit `volume`, `number`,
  `pages` and `doi`, all of which are unverified detail rather than what a citation needs
  to exist. Record in the replication appendix that this entry rests on the in-repo
  transcription alone. If you want the full fields, they need external verification I
  cannot perform in this pass — flag it and I will leave the entry minimal.

*Verification.* `biber` run with zero "not found" warnings; grep that each of the ten keys
appears in ≥1 active `\cite`; visually confirm the printed reference list contains all ten;
confirm `\addbibresource` count is still **2** and no path under `analysis/` appears in the
preamble.

---

### E.2 Methods and theory (Tier B1.3)

**`C-theory-004` · DELETE_OR_REPLACE · P0 · Conf HIGH · §3.4 `subsec:correction_scope`**

*Problem.* The current text argues the Direct curvature contribution is small enough that
the objective behaves close to gradient-only. Gate E.4 does **not** support this.
*Numbers to use:* `e4_verdict = INDETERMINATE`; `max_curvature_ratio = 40.04` against an
accept threshold of **1.2**; `max_penalty_contribution_M = 0.0118` against an accept
threshold of **0.01** (`N-e4-verdict`, `N-e4-max-curvature-ratio`, `N-e4-max-M`,
`N-e4-M-accept-threshold`).
*Must disappear.* "Direct is effectively gradient-only", and any softer paraphrase of it.
*Required final message.* Report E.4 as INDETERMINATE with its evidence, and describe the
structure accurately and proportionately: the exact Hessian is
`(2/n)I + (rho/n^2) c c^T`; the standard observation-wise interface admits only the
diagonal, so the **dense rank-one cross-observation term is omitted** and a **diagonal
approximation is supplied** — `1 + (rho/(2n)) c_i^2` for Direct, `1 + rho c_i^2` for
Surrogate (`objective_scaling_audit.csv`, all nine blocks, `n/2` applied analytically so
`grad = e`, `hess = 1` at ρ=0). Neither overstate nor minimize.
*Dependency.* None. *Verification.* Forbidden-phrase grep; E.4 numbers match
`e4_verdict.json`.

**`C-theory-003` · KEEP + ADD · P0 · Conf HIGH · §3.3 `subsec:surrogate`**
Keep the Jensen framing exactly. **Add** the effective-sample-size consequence at ρ=100
from `surrogate_weight_distribution.csv`. Keep the wording **associational** — an
implementation-side diagnostic or plausible explanation, never a proven causal mechanism.
*Must disappear.* Any phrasing that makes weight concentration the established cause of
the observed Surrogate behaviour.

**`C-theory-002` · KEEP · P0 · Conf HIGH · §3.2 / App. D**
The general post-hoc root is `b_star_train = Var_T(y)/Cov_T(f0,y)`.
*Must disappear.* "the post-hoc root is 1/R^2". `1/R^2` is a theoretical diagnostic
confined to the linear/projection special case and must not be presented as the general
LightGBM result.

**`C-theory-001` · KEEP · P0 · Conf HIGH · §3.1**
The paper may describe `Cov(e,y)` as a **first-order component / mechanism / diagnostic**
of price-related regressivity, with `e = prediction − y`, `y = log(sale price)`, and
`beta_log = Cov(e,y)/Var(y)` on a fixed evaluation sample. It may **not** define
regressivity itself as residual–outcome covariance, nor imply zero covariance means
fairness, conditional unbiasedness, independence, or a flat conditional ratio profile.
The active text already denies this four times — **those four denials must survive.**

**`C-theory-005`, `C-theory-006` · KEEP · P1 · Conf HIGH · App. D**
Keep both constraints verbatim. Never write "retrained Direct is equivalent to post-hoc
rescaling"; never attribute the empirical high-ρ S-shape to particular eigenmodes without
a diagnostic that projects fitted-path changes onto the fixed-space modes (none exists in
P0 or P1).

---

### E.3 Design (Tier B1.4)

**`C-design-007` · ADD · P0 · Conf HIGH · §4.3 `subsec:comparators`**

*Problem.* The manuscript uses **no** A / B / C or D1 / D3 token anywhere. The frozen
reference convention is the coordinate system the rest of the revision depends on.
*Required final message.* Introduce the three cells by **forward display name** with
their roles (table above), and state the frozen attribution rule: comparisons against A
are descriptively valid, but changes relative to A must not be attributed solely to ρ;
C(ρ=0) → Direct/Surrogate(ρ>0) is the only contrast from which a ρ effect may be
attributed. Note B has no full path.
*Must disappear.* Legacy Stage-1 label strings as manuscript-facing display names
("Parity-aligned native L2", "Custom rho=0 origin") — those may appear only as values
inside Stage-1 tables.

**Development coordinates · ADD · §4.2 `subsec:path_design`**
Use this statement, or an equivalently precise final version:

> Primary development coordinate D1 preserves the CCAO-inspired overlapping rolling-origin
> workflow. D3 is the one-sale-one-vote sensitivity. Fold-level SDs are descriptive
> chronological-window variation, not IID standard errors.

D2 is the duplicate-weighted pooled-OOF historical sensitivity and is **not** elevated
over D1/D3.
*Must disappear.* "mean +/- SD/sqrt(7)" and any IID significance language built on seven
overlapping chronological folds.

**`C-design-001` · REWRITE · P0 · Conf HIGH · §4.1**
State the extract with its sha256 from `PREFLIGHT.json` —
`data/CCAO/2025/training_data.parquet`, sha256 `b1fc00b5...`, 215,400,916 bytes — plus a
retrieval date and citable source. Do not imply the file is redistributed.
**Open gap:** the citable public source and build/retrieval date are still missing
(§F below).

**`C-design-002/003/004` · KEEP · P1 · Conf HIGH**
Cite the configuration hash (`8f0f2acd...`, `lgbm_params_sha256_reverified = true`) rather
than listing every parameter; keep the two-stage grid history explicit (50-point
[0.1,100] + 32-point lower-tail [0.00110, 0.08685]; `N-design-n_rho_total = 83` including
ρ=0); keep the four sample counts (344,607 / 38,290 / 382,897 / 26,641) with
`PREFLIGHT.json` cited. Do not later present all 82 positive values as prespecified.

**`C-design-005` · KEEP + ADD · P1 · Conf HIGH · `tab:fold_structure`**
Keep all seven `n_train`/`n_val` pairs (46,888/5,209 … 310,147/34,460). **Add** the frozen
finding that the fold-6 and fold-7 **validation blocks overlap** — origins about four
months apart (2023-07-01 → 2023-11-09) while 10% of the development pool spans about a
year — so 20,988 unique rows appear twice: 151,153 appearances over 130,165 distinct sales,
**13.885268568933465 %** duplicated, max multiplicity 2. Window end dates are allowlisted
design constants, not frozen evidence.

**Wording constraint — "unaffected" is prohibited here.** The Tier-B0 spec's own phrasing
for this claim ("the equal-weight CV-mean coordinate is unaffected") and
`CENTERED_SPREAD_COMPARATOR_REPORT.md` §4 ("D1 … is unaffected (True)") are both
**superseded** by `POST_G3_ADJUDICATION.md` §2: *"D1 must not be described simply as
'unaffected.' The precise statement is five-part."* Use the adjudicated interpretation:

> D1 remains the primary coordinate — the equal-weight mean of the seven chronological
> fold metrics — and its mathematical definition is unchanged; every fold-level prediction
> remains genuinely out of training sample. But the fold-6 and fold-7 validation blocks
> overlap, so D1 is **not an independent-observation aggregate**: some sales contribute to
> more than one fold-specific value, fold-level statistics are not independent, and D1 must
> not be described as unaffected by the overlap. D3 is the one-sale-one-vote sensitivity
> (130,165 distinct sales, `w_ik = 1/m_i`, development-only). D2 is the duplicate-weighted
> pooled coordinate and is the only one that double-counts by construction.

Supporting frozen strings available to cite: `POST_G3_ADJUDICATION.md` §2.2 — *"Any reading
of D1 as an average over seven separable pieces of evidence — and in particular any
standard-error, sampling-distribution, or 'seven replications' reading of `CV_sd` — is
unsupported"*; and `development_beta_coordinate_summary.json` → `D1.no_iid_interpretation
= true`.
*One further technical detail to state where Δ_NL on the pooled sample is mentioned:* for
the D2 sample only, a composite `"fold|row_id"` identifier is supplied to the Δ_NL
estimator (frozen estimator spec otherwise unchanged).

**`C-design-008` · KEEP verbatim · P0 · Conf HIGH · §4.4**

---

### E.4 Results — the rebuilt path tables (Tier B2.1)

**`C-results-001`, `C-results-002`, `C-results-003`, `C-results-004`, `C-results-005`,
`C-results-006` · `tab:path_anchor_summary` REBUILD → **SPLIT** · P0 · Conf HIGH**

**Decided (§F.0(2)): this becomes two tables**, so the provenance boundary is structural.
`tab:path_anchor_frozen` carries prediction + PRD + MKI at the two frozen anchors only;
`tab:path_anchor_standards` carries PRB and VEI at all five anchors, with PRB attributed to
the adopted 2013 Standard and VEI to the May-2026 Exposure Draft, labelled differently
because their standing differs. All Linear rows are removed from both.

Three kinds of cell, currently indistinguishable:

*Supported and kept.* Ordinary LightGBM (Cell A) rows; Direct and Surrogate at the **two
frozen anchors** (ρ≈0.954095 and ρ=100) at reproduction tier R1
(`frozen_artifact_reproduction.csv`, cached and refit agreeing at displayed precision);
and **PRB and VEI at all five anchors** (`prb_inference.csv`, `vei_significance.csv`) —
the strongest resolvability result in the audit: all 40 printed values re-derive exactly,
234 values reconciled per measure, max |Δ| 7.6e-16 (PRB) and 7.1e-15 (VEI).

*Unsupported and removed.* Every Linear-regression cell; and R²_P / MAE / MAPE /
RMSE_log / PRD / MKI at the **three legacy anchors** (0.0104811313415468, 0.1,
10.481131341546853) — the legacy V6/V12 prediction trees are gitignored, in no checkout,
hash-pinned only. **Do not relabel these "exploratory" and print them anyway.** Under the
split, the three legacy anchors simply do not appear in `tab:path_anchor_frozen` at all;
no blank cells and no footnoted absence.

*Cell C must be visible as a row, not inferable from prose.* Both reference cells get an
explicit row in each rebuilt table:

| row | display name | role |
|---|---|---|
| **A** | Ordinary LightGBM (standard raw-label native) | workflow benchmark |
| **C** | Custom-objective rho=0 origin | **penalty-isolating origin (primary)** |

Cell C resolves for the full 17-metric set on all three regimes from
`zero_control_full.csv` (the Stage-1.5 / Gate-G2 full zero control), so a C row is fully
supportable in both rebuilt tables and in `tab:path_anchor_complementary`. Held-out /
2025 C values to use: R²_P 0.893 / 0.904, RMSE_log 0.289 / 0.279, COD 21.599 / 21.229,
PRD 1.0690 / 1.0775, PRB −0.0884 / −0.1028, MKI 0.9230 / 0.9093, VEI −26.167 / −28.407,
β_log −0.147 / −0.161, Δ_NL 0.1164 / 0.1212, dCor 0.382 / 0.417.
If a table cannot carry a C row for layout reasons, it must instead carry an equally
explicit device — a Δ-vs-C column, or a paired panel — never a footnote. **The primary
comparison may not be left inferable only from prose.**

*Two markers, and no asterisk.* Asterisks conventionally read as statistical significance,
which is exactly the misreading the current table invites. The convention is:

- **bold** = improvement relative to **A**, the workflow benchmark;
- **†** = improvement relative to **C**, the penalty-isolating origin (the primary
  comparison);
- never one device for both, and both explained in the notes in words, naming each
  reference cell by its forward display name.

*"Improvement" is defined metric-by-metric, against each metric's own direction or ideal —
not by a single "higher/lower is better" rule.* The definition, which must be stated in the
notes:

| metric | comparison rule | ideal |
|---|---|---|
| R²_P | **higher** | 1 |
| MAE_P, MAPE_P, RMSE_logP | **lower** | 0 |
| PRB, VEI, β_log | **closer to zero** in absolute value | 0 |
| PRD, MKI | **closer to the stated ideal** in absolute distance | 1 |
| median / mean / weighted-mean ratio | **closer to the stated ideal** | 1 |
| COD, COV | **closer to less dispersion**, within the reference range | no universal ideal |
| Δ_NL, dCor | **lower** = less non-affine / less broad dependence | no assessor-standard ideal |

Two riders the notes must carry. First, **COD and COV have no universal ideal** — the
manuscript's own metric table records `--` for both, and the frozen guidance notes that
unusually *low* dispersion can itself warrant diagnostic review; so a marker on COD or COV
means "moved toward less dispersion", not "better". Second, **Δ_NL and dCor carry no
assessor-standard reference range at all** and are mechanism/residual-structure
diagnostics, so a marker on them is directional only and must not be read as an equity
ranking. Comparisons are evaluated on the **unrounded** canonical values, so two cells that
appear tied at displayed precision may still be marked differently — say so.

The notes must state explicitly that **neither marker denotes statistical significance,
standards compliance, model selection, or a recommended ρ** — only direction of movement
relative to the named reference cell. The current `\textsuperscript{*}` device ("improves
on both baselines") disappears entirely, which the Linear removal makes moot anyway.

*Two numeric-consistency items to fix in the same pass:*
- Direct ρ≈10 held-out **VEI = −10.5%** is currently starred, which reads as compliance,
  but it lies **outside** the proposed ±10% exposure-draft band. Keep the value; say it
  sits just outside the band (`C-results-005`).
- Surrogate ρ≈10 **2025 PRB = 0.000** is a *rounded display* of `0.0001345072515744`,
  not an exact zero. Show it as a rounded estimate, and state the 2013 classification
  rule: the **entire** 95% CI must lie outside a band before that band is deemed
  exceeded; a CI that merely crosses a threshold does not qualify (`C-results-006`).
- The display anchor is written `0.0104811` in one place and `0.0104811313` in another —
  pick one form and use it everywhere.

*Numbers that must be used (frozen anchors, held-out / 2025):*

| | Direct ρ≈0.954095 | Direct ρ=100 | Surrogate ρ≈0.954095 | Surrogate ρ=100 |
|---|---|---|---|---|
| R²_P | 0.899 / 0.910 | 0.869 / 0.873 | 0.897 / 0.908 | 0.889 / 0.900 |
| MAE_P | \$74,485 / \$77,139 | \$84,918 / \$89,309 | \$75,468 / \$78,196 | \$82,094 / \$85,257 |
| MAPE_P | 21.1 / 20.7 | 23.9 / 23.1 | 21.0 / 20.4 | 22.8 / 22.0 |
| RMSE_logP | 0.290 / 0.279 | 0.325 / 0.307 | 0.298 / 0.284 | **0.357 / 0.336** |
| PRD | 1.060 / 1.069 | 1.029 / 1.036 | 1.049 / 1.060 | 1.010 / 1.020 |
| MKI | 0.940 / 0.926 | 0.998 / 0.983 | 0.953 / 0.932 | 1.013 / 0.987 |
| PRB | −0.075 / −0.090 | −0.015 / −0.029 | −0.045 / −0.062 | 0.042 / 0.030 |
| VEI | −21.9 / −23.9 | 0.7 / −4.5 | −12.6 / −15.3 | 4.0 / 1.7 |

Cell A (workflow benchmark), held-out / 2025: R²_P 0.894 / 0.904; MAE \$75,655 / \$78,484;
MAPE 21.2 / 20.8; RMSE_log 0.289 / 0.278; PRD 1.069 / 1.079; PRB −0.091 / −0.106;
MKI 0.923 / 0.907; VEI −26.5 / −28.6; COD 21.6 / 21.3; COV 39.7 / 37.2;
β_log −0.150 / −0.164; Δ_NL 0.119 / 0.121; dCor 0.387 / 0.422; median r 0.929 / 0.950;
mean r 0.989 / 1.015; weighted mean r 0.924 / 0.941.

Cell C (ρ=0 origin, penalty-isolating), held-out / 2025: R²_P 0.893 / 0.904;
RMSE_log 0.289 / 0.279; β_log **−0.147 / −0.161**; dCor **0.382 / 0.417**.

PRB and VEI at the three legacy anchors — **kept**, from P1:

| anchor | Direct PRB h/f | Direct VEI h/f | Surrogate PRB h/f | Surrogate VEI h/f |
|---|---|---|---|---|
| 0.0104811313415468 | −0.089 / −0.105 | −26.9 / −29.8 | −0.088 / −0.101 | −25.8 / −27.4 |
| 0.1 | −0.088 / −0.102 | −27.0 / −27.7 | −0.079 / −0.095 | −22.7 / −24.7 |
| 10.481131341546853 | −0.040 / −0.049 | **−10.5** / −11.9 | 0.013 / **0.000** | 0.2 / −2.0 |

**`C-results-020` · `tab:path_anchor_complementary` REBUILD · P0 · Conf HIGH**
Same three-way split, and worse in one respect: `frozen_artifact_reproduction.csv` carries
**no Δ_NL and no dCor column at all**, so those two columns are unsupported at **every**
positive ρ, not only at the three legacy anchors. Keep the baseline rows and the six
remaining complementary metrics at the two frozen anchors (median/mean/weighted-mean
ratio, COD, COV, β_log — values in the numeric map). Drop or re-derive Δ_NL and dCor for
every positive ρ. Same two-role highlighting.

---

### E.5 Results — baseline tables (Tier B2.2)

**`C-baseline-001` / `C-baseline-002` · `tab:ccao_baseline_results` · P0 · Conf HIGH**
The LightGBM half resolves **exactly** from `zero_control_full.csv` cell A — all eight
metrics on both evaluations. The Linear half resolves from **nothing**, and prints
`R^2_P = 0.799` in **both** panels, which is itself a consistency flag: an identical value
to three decimals on two different samples is implausible enough to require re-derivation
rather than qualification. All 16 flagged tokens in this table are exactly the Linear
column (verified mechanically this session).
**Decided (§F.0(1)): the Linear columns are removed.** The table goes from four columns to
two — held-out and 2025 forward, Cell A only — and is labelled explicitly as an
assessor-facing workflow-benchmark comparison. The qualitative tension survives as a level
statement about Cell A rather than as a linear-vs-nonlinear contrast.

**`C-baseline-003` / `C-baseline-004` · `tab:ccao_baseline_complementary` · P0/P1 · Conf HIGH**
Re-source cell A for all eight complementary metrics including Δ_NL and dCor; remove the
Linear columns (18 flagged tokens). **State explicitly** that baseline COD (21.3–21.6% for
Cell A; up to 24.5% along the paths) lies **outside** the adopted IAAO [5,15] range for
single-family residential, so no compliance claim is implied — acknowledge it rather than
omit it. Note for provenance: this COD-exceeds-range statement is *not* recorded inside P1;
it is a Tier-B0 editorial requirement resting on values that do resolve from
`zero_control_full.csv`. Present it as interpretation, not as a P1 finding.

---

### E.6 Results — mechanism and shape (Tier B2.1/B2.6)

**`C-results-010` · KEEP · P0 · Conf HIGH · new §5.4**
Both ends of every quoted β_log range resolve — ρ=0 from `zero_control_full.csv` cell C,
ρ=100 from `frozen_artifact_reproduction.csv` — and the comparison is correctly referenced
to the custom-objective ρ=0 origin. Keep: held-out Direct β_log −0.147 → −0.079;
2025 −0.161 → −0.093; Surrogate reaching −0.001 (held-out) and −0.015 (2025) at ρ=100.

**`C-results-011` · REWRITE · P0 · Conf HIGH · new §5.4**
Keep the ρ=0 dCor values (0.382 held-out, 0.417 in 2025 — Cell C). The positive-ρ dCor
path values 0.250 (near ρ=10.481) and 0.267 (ρ=100), and the 2025 0.258 → 0.266 rebound,
resolve from **nothing**: dCor is not a column of
`frozen_artifact_reproduction.csv`. **The non-monotonicity claim currently rests on
unresolvable numbers** and must be re-derived or dropped. *Note:* Cell A dCor at ρ=100 in
`tab:path_anchor_complementary` (0.265/0.281 Direct, 0.267/0.266 Surrogate) is subject to
the same removal — do not keep it in one table while deleting it from prose.

**`C-results-021` · REWRITE · P0 · Conf HIGH · `fig:mechanism_path_placeholder`**
Swap `mechanism_vs_rho_candidate_region.pdf` → `mechanism_vs_rho.pdf`. Delete the
candidate-region, activity-onset and guardrail sentences from the caption. Keep the
mechanism path curves and the β_log = 0 neutrality reference.
*Must disappear.* "sweet spot", "safe region", "deployment point".

**Old §5.3 → merged into §5.4.** Keep the Δ_NL/ratio-shape argument that does not depend
on the deleted screen: Direct held-out Δ_NL stays in a narrow band and is not increasing
at the strongest penalty, while Surrogate's falls and then rebounds. **Delete** the
"1/7 folds vs 6/7 folds" event-relative ratio-profile QA sentence and the
`\Delta_{NL}`-rebound-at-the-guardrail interpretation — both are the deleted screen. The
descriptive CV Δ_NL minima sentence at L4269 (Direct 0.08685 at ρ=100; Surrogate 0.08677
at ρ≈1.677) is **not** in the numeric map — treat as unsupported unless it resolves.

---

### E.7 Results — the three additions that carry the revision (Tier B2.3/B2.4/B2.5/B2.6)

**`C-results-007` / `C-results-008` / `C-results-009` · new §5.2 + App. E.3 · P0 · Conf HIGH**

*Keep* `tab:rho_zero_control` — fully resolvable and central: cells A and C from
`zero_control_full.csv`, the A↔C prediction deltas from
`zero_reference_reproduction_qc.csv` (held-out mean |Δ| `3.24e-02`, max `3.09e-01`;
2025 `3.06e-02` / `2.83e-01`). Direct and Surrogate at ρ=0 **coincide exactly**
(`parity_ladder C↔C_surrogate` max |Δ| = 0). This table is the reason A-to-penalized
contrasts cannot isolate ρ.

*Replace the two `\todo` promises with the findings.* Todo #16 resolves in the
**negative**: on the pinned track A↔B agrees to about **7.7e-09** (native-vs-native parity
essentially holds) but B↔C does **not** — mean |Δ log| about **0.0176**, max about
**0.140** — and pinning does not remove it (`p4_pinning_does_not_remove_BC = true`). The
gap is the **deterministic built-in-vs-custom feature-subsampling execution-path
divergence**, not label initialization; B↔C becomes identical only when
`colsample_bytree = 1.0`, which specifies a *different learner* rather than reproducing
the frozen experiment. **The table is NOT regenerated**, and the surrounding prose must
stop promising that parity holds. Todo #17 is closed outright by
`objective_scaling_audit.csv` + `PREFLIGHT.json` + `provenance/worktree_diff.patch`.
*Must disappear.* "innocuous numerical effect"; and the current L2114 sentence
"The substantive tables must be regenerated from an initialization-aligned parity run
before any native-to-penalized contrast is interpreted as a pure penalty effect" —
the rerun happened and resolved negatively, so this promise is now false.

**`C-results-019` · ADD · P0 · Conf HIGH · new §5.6 (centered-spread comparator)**

*Required final message.* **Promote this analysis.** It demonstrates that post-hoc
centered rescaling can reproduce or improve much of Direct's first-order accuracy/equity
frontier over the **Direct-attainable β range**, while **not** reproducing all nonlinear
behaviour. The C-based map is **PRIMARY**, the A-based map **SECONDARY_PRACTICAL**, and
**B has no full post-hoc path**. `no_refit_performed = true`; all 20 input prediction
arrays hash-verified.

*The map — two centerings, and they must stay distinct.* Quoting
`configs/posthoc_comparator_convention.yaml`:

- **`primary_map.formula`** = **`f_b(x) = ybar_T + b * (f0(x) - ybar_T)`**, named the
  *theorem-matched centered-spread map*, because `cor:path_scaling` expresses the
  fixed-space Direct path as `f_rho = ybar·1 + b_rho·(f0 − ybar·1)`. The config states it
  is **"the only map used for the full path"**. `ybar_T` is the fitting block's mean log
  *target*: fold-k training mean for fold k, development-pool mean for held-out,
  production 2016–2024 mean for 2025. At `b = 1` there is an explicit fast path returning
  the cached prediction array **bitwise** unchanged.
- **`centering_sensitivity.alternative_map`** = **`f_b(x) = ybar_T + b * (f0(x) - f0bar_T)`**,
  centred on the fitting block's mean *prediction*. It is a frozen **sensitivity**, not the
  primary map: `full_second_path_scheduled: false`, evaluated only at four anchors
  (`b_1`, `b_direct_upper_common_support`, `b_zero_cvmean`, `b_max`) × three regimes × both
  references. Its analytic difference from the primary map is
  `b·(f0bar_T − ybar_T)`, **a constant in x**, so β_log, `Cov(e,y)`, COD, COV and dCor are
  *exactly* unaffected and only level-sensitive metrics can move; Gate G2 bounded
  `max |f0bar_T − ybar_T| ≤ 8.36e-06` over all 27 fits, giving a worst-case log shift of
  about `1.0e-05` (measured `centering_max_analytic_shift_log = 6.266e-06`,
  `centering_max_metric_absdiff = 1.5444` — \$1.54 of MAE).

> **Disagreement recorded, and the frozen evidence followed.** The review instruction
> stated that the frozen comparator is the `f0bar_T`-centered form and that `f0bar_T` must
> not be replaced by `ybar_T` because they are numerically close. The frozen config says
> the reverse: the `ybar_T`-centered map is `primary_map` and the theorem-matched form, and
> the `f0bar_T` map is `centering_sensitivity.alternative_map`, scheduled for anchors only
> **precisely because** the two are within `8.36e-06`. Per the standing rule that frozen
> evidence governs, the plan reports the `ybar_T` map as primary and the `f0bar_T` map as
> the named centering sensitivity. The instruction's substantive point is adopted in full:
> the two are never conflated, never presented as interchangeable, and the sensitivity is
> reported with its bound rather than waved away as "numerically close".

*Three distinct root concepts — never collapsed into one.*

| # | concept | definition | status |
|---|---|---|---|
| (a) | **fitting-block theoretical root** | `b_T* = Var_T(y) / Cov_T(f0,y)` on the fitting block | `IN-SAMPLE THEORY DIAGNOSTIC ONLY`, `used_for_empirical_endpoint: false`. Range C 1.065397 … 1.092817. Frozen reason: the 994-tree learner is far less regressive in sample (`beta_log_train ≈ −0.06 … −0.09`) than out of sample (`≈ −0.14 … −0.16`), so the in-sample root **understates** the required rescaling. |
| (b) | **D1 development-coordinate root** | `beta_k(b) = b·Cov_Vk(f0,y)/Var_Vk(y) − 1`; `beta_cvmean(b) = (1/7)Σ_k beta_k(b)`; **`b_zero_cvmean = 1 / mean_k[ Cov_Vk(f0,y)/Var_Vk(y) ]`** | The empirical D1 root. Built from **validation-block** quantities (`Cov_Vk`, `Var_Vk`) — **not** a training covariance/variance relation. C `1.160457945844737`, A `1.1603673641139836`; `mean_R_k` C `0.8617287714567422`. |
| (c) | **D2 pooled-OOF sensitivity root** | `b = (V − P)/(Q − P)` with `V = Var_pooled(y)`, `P = mean(ybar_T(k(i))·c)`, `Q = mean(f0·c)`; `P` is non-zero because the fold centers correlate with `c` | Sensitivity only. C `1.1670266398954645`, A `1.1672656789694134`. |

Root verification: achieved `beta_log` `−1.001e-16` (C, D1) against a 1e-12 target.
`b_max = 1 + 1.25*(max of all four development roots − 1) = 1.2090820987117668`;
`development_only: true`, `no_oos_information_used: true`. And `one_over_r2` is recorded
`status: theoretical diagnostic only`, `used_for_endpoint: false`.

*The frozen framing questions, worth borrowing verbatim.* For **C** (`PRIMARY`): *"At the
same first-order correction, what does retraining buy relative to globally rescaling the
same unregularized custom-path predictor?"* — the rationale being that C is the common
custom-objective ρ=0 origin of both penalized families, so a C-based map holds the
execution path fixed and isolates retraining. For **A** (`SECONDARY_PRACTICAL`): *"Could a
practitioner obtain a similar tradeoff simply by post-processing standard LightGBM?"*,
`display_rule: "may appear in a separate panel/table; descriptive wording only"`. Cell B is
`NO_FULL_PATH`. A-posthoc carries three explicit frozen restrictions: it must **not**
determine the three-way common support, must **not** determine CORE target selection, and
must **not** replace C-posthoc in the primary mechanistic test.

*The headline comparison — and it crosses over.* From
`CENTERED_SPREAD_COMPARATOR_REPORT.md` §7, **explicitly descriptive** and keyed on
held-out β (the report states the formal matched-β comparison is not performed in Stage 2):

| at held-out β_log ≈ | family | R²_price | MAE_price | Δ_NL | dCor |
|---|---|---|---|---|---|
| **−0.07941** | Direct (ρ=100) | 0.86868 | 84,918 | 0.11535 | 0.26493 |
| | C-posthoc (b=1.080148) | **0.89242** | **74,930** | 0.12946 | 0.28404 |
| **+0.00084** | Surrogate (ρ=75.4312) | **0.88908** | **81,930** | 0.12434 | 0.26804 |
| | C-posthoc (b=1.174235) | 0.77962 | 90,429 | 0.13438 | 0.25408 |

Report's own words: *"The accuracy comparison appears to cross over with correction
strength."* **This is the paper's most important honest finding and its scope limit at
once:** post-hoc rescaling *beats* Direct at moderate correction, and *loses badly* to
Surrogate at strong correction. The Surrogate claim is the one that survives.

*The cost of driving development β to zero post-hoc* (`b = 1.160458`), which the paper
should report because it bounds what the first-order axis can buy: held-out R²_price
0.89281 → **0.80628**, MAE \$75,976 → **\$87,182**; 2025 R²_price 0.90385 → 0.79906.
Δ_NL **rises** 0.11645 → 0.13430; dCor falls 0.38225 → 0.25175 then **rebounds** to
0.26813 at `b_max`. Assessor metrics **overshoot**: PRD 1.0690 → 0.9871, MKI 0.92305 →
1.07258, VEI −26.167 → **+14.770**, and COD *worsens* 21.599 → 22.587.

*Must disappear.* "retrained Direct is equivalent to post-hoc rescaling"; "b_max = 1.25 *
b_star"; "the post-hoc root is 1/R^2".
*Never quote these literals* — unrendered f-string placeholders in §8–9 of the report.
Resolve to `executed_in_stage_2 = false`, `n_grid = 124`, `len(path) = 2,480`.
*One superseded statement in a frozen report — do not quote it.* The report's §4 says
"D1, the paper's primary CV coordinate, **is unaffected** (True)". `POST_G3_ADJUDICATION.md`
§2 **supersedes** this: *"D1 must not be described simply as 'unaffected.' The precise
statement is five-part."* Use the adjudication, not the report, on this point.
*Consequence for the paper's stance.* No superiority claim for either family is available
— exactly what `C-design-006`, `C-theory-005`, `C-discussion-003` already say. Those stay.

**`C-results-018` · ADD · P0 · Conf HIGH · new §5.7 (matched-β) — the central comparison**

*Required final message.* Primary triple: **Direct vs Surrogate vs C-posthoc** at matched
development achieved β_log. A-posthoc is **secondary and practical only**. B has no full
path. **D1 is PRIMARY and D3 is the SENSITIVITY coordinate — never swap them.**
Frozen status: **`MATCHED_BETA_STATUS = PASS`**, `tau = 0.002`, three-way common support
`[−0.1382712285432577, −0.0786514918143895]`, and **zero targeted new fits were required**.

*The six CORE targets (`j`, target β_log, and the configuration each family used):*

| j | target β_log | Direct ρ | Surrogate ρ | C-posthoc b | A-posthoc b |
|---|---|---|---|---|---|
| 0 | −0.1382712285432577 | 0.0 | 0.0 | 1.0000000000000002 | 0.9999219431164414 |
| 1 | −0.1264903880813394 | 0.7196856730011519 | 0.1325711365590109 | 1.0136711699227623 | 1.013592045910285 |
| 2 | −0.1147325732441951 | 1.6768329368110082 | 0.4094915062380425 | 1.0273156195762976 | 1.0272354305206024 |
| 3 | −0.1020327443752032 | 3.906939937054617 | 0.8286427728546845 | 1.0420532368981876 | 1.0419718974700132 |
| 4 | −0.0909015214774399 | 9.102981779915218 | 1.4563484775012436 | 1.054970552956866 | 1.054888205243256 |
| 5 | −0.0786514918143895 | 86.85113737513521 | 2.559547922699536 | 1.0691861972161865 | 1.0691027398736879 |

All 24 CORE cells `attained = True`, worst gap `1.0569e-03` < τ. `match_mode`: Direct
`exact_anchor`, Surrogate `nearest_fitted`, both post-hoc `exact_posthoc_solve`.
**Origin identity QC at j=0:** Direct, Surrogate and C-posthoc coincide to max relative
**4.42e−12** across all 17 metrics × 4 evaluations — the j=0 row is a genuine identity
check, and should be presented as one.

*The result that carries the subsection — j=5, held-out:*

| family | R²_price | MAE_price | RMSE_log | COD | Δ_NL | dCor |
|---|---|---|---|---|---|---|
| Direct (ρ=86.85) | 0.8760 | \$82,764 | 0.3170 | 23.97 | 0.1198 | 0.2730 |
| Surrogate (ρ=2.5595) | 0.8963 | \$76,128 | 0.3094 | 21.40 | 0.0914 | 0.2714 |
| C-posthoc (b=1.06919) | 0.8966 | \$74,236 | 0.2920 | 21.61 | 0.1282 | 0.2944 |

At the strongest attainable common-support target, **Direct is worse than both Surrogate
and C-posthoc on price accuracy and on COD.** At matched first-order correction the three
are *not* interchangeable, and the ordering is not the one the current manuscript's
anchor table implies.

*The four `NOT_ATTAINED` states — verbatim, never interpolated.* EXT targets
`[−0.06, −0.03, 0.00]`: **Direct NOT_ATTAINED at all three** (max achieved
−0.0786514918143895; gaps 0.018651…, 0.048651…, 0.078651…), **Surrogate NOT_ATTAINED at
0.00** (max achieved −0.0186094650282942, gap 0.0186094650282942). Both post-hoc families
attain all three. NOT_ATTAINED rows carry **blank metric columns** — never dropped, never
filled. That is 16 `attained=False` rows in `matched_beta_ext_targets.csv` and 4 display
entries.

**State non-attainment as a bounded statement about the evaluated path, never as global
impossibility.** Required wording, or an equivalently bounded form:

> Direct did not attain β_log ∈ {−0.06, −0.03, 0} over the evaluated frozen path;
> Surrogate did not attain 0 over its evaluated path.

*(Corrected from an earlier draft of this plan, which wrote that Direct "cannot reach"
first-order neutrality "on this fixed base learner".)* That is an overclaim: the frozen
grid is 82 positive ρ values per family, so non-attainment is a property of **the evaluated
path**, not of the objective or the learner. Do **not** write that Direct can never reach
neutrality for this base learner at unexamined ρ values. The defensible contrast is
therefore also bounded: over the evaluated path a one-parameter post-hoc rescaling reaches
the three EXT targets exactly while the retrained Direct path does not — which is
informative about what the frozen experiment covers, not about what the objective could do
under a grid it never explored.

*The RMSE_log reversal — mandatory, and the report flags it itself.* Quote-worthy source
wording: *"In `RMSE_log` the ranking reverses — Surrogate is worse (0.3443 vs 0.2999 at
−0.03). The two accuracy metrics disagree because they weight the price distribution
differently; `R²` on price is dominated by the expensive tail that the Surrogate's
`w_i = 1 + ρc_i²` weighting protects, while `RMSE_log` weights all sales equally in logs.
Both are reported; neither alone settles 'predictive performance.'"* Exact values at
β_dev = −0.03: Surrogate R²_price **0.8917** / RMSE_log **0.3443**; C-posthoc R²_price
0.8569 / RMSE_log **0.2999** — the price-level ranking and the log-scale ranking point
opposite ways. At β_dev = −0.06: Surrogate 0.8934 / 0.3230; C-posthoc 0.8868 / 0.2944.
A second reversal at **j=5** is in the data but not called out in the report:
ΔR²_price(Surrogate − C-posthoc) = **−0.00035**, below the 0.0005 materiality floor, i.e.
a *tie* on price, while ΔRMSE_log = **+0.0174**, i.e. Surrogate clearly worse in logs.
The R²-parity impression reverses in logs. **Report both.**

*D3 sensitivity — report, do not bury.* `D3_MATERIAL = true`; **13 of 24 configurations
were reselected** under D3; of 315 headline sign comparisons, 259 agree, 41 fall below the
materiality floor, and **10 are MATERIAL_SIGN_FLIPs** (raw sign-agreement 0.9365);
`max |D1 − D3| = 0.004867498943410613`. Also `rank_order_preserved_D1_vs_D3 = false`, and
monotonicity in β is false for Direct and Surrogate but true for both post-hoc families.
D3 common support is `[−0.1377438408828227, −0.07670460033496714]`.

*Counting unit.* `display_entry` for every aggregate/count statement here (48 display
entries, 44 attained, 4 NOT_ATTAINED, 43 fitted realizations, 10 evaluation blocks → 480
rows in each P1 inference table).

**`C-results-017` + `C-results-015` · ADD + REWRITE · P0 · Conf HIGH · new §5.8 (temporal robustness)**

*Required final message.* Report the frozen outcomes **as recorded**, not paraphrased as a
clean pass. The frozen verbatim block is:

```
MATCHED_BETA_STATUS = PASS
TEMPORAL_STATUS     = PASS_PRIMARY_STANDS
G5a                 = ALERT (T1 only, D-SNAP)
G5b                 = NOT_CONFIRMED
```

*D-SNAP — strict-date separation.* The positional cut is moved back so
`max(train_date) < min(eval_date)` at all eight development/held-out boundaries (35–158
rows moved per boundary; `production_2025` moves **0** rows). Frozen answer to "does
strict-date separation change any core path conclusion?" — **"No."** ΔR²: CV_mean mean
+0.00001 (range −0.00090 … +0.00305); held-out mean −0.00124 (−0.00407 … +0.00225);
2025 **all +0.00000**. `max|Δβ_log|` 0.00284 / 0.00571 / 0.00000. Frozen wording:
*"The primary temporal design stands. No 82-point regeneration is warranted or authorized."*

*D-PURGE — an oracle sensitivity, not a design.* Every training row whose `meta_pin`
appears in the corresponding evaluation block is dropped; evaluation sets are bitwise
identical to the primary; training shrinks 1.077 %–3.828 %; `eval_pins_in_train_after_purge
= 0` in all nine blocks. What changes: β_log at ρ=0 becomes **more negative** by −0.01184
(CV_mean), −0.02949 (held-out), −0.02257 (2025), and dCor rises +0.021 / +0.052 / +0.043;
mean ΔR² −0.00091 on CV_mean. Frozen wording: *"D-PURGE is NOT promoted to the primary
temporal protocol"*, `design_type = oracle_diagnostic`, `feeds_g5_gate = False`, and
*"Absolute performance deterioration under D-PURGE is not, by itself, evidence against the
primary design."*
**Must disappear.** Any statement that repeat-parcel information "causes" or definitively
"masks" regressivity. The correct reading: under an oracle removal of parcel overlap the
measured first-order pattern is somewhat larger — a sensitivity, not an identification.

*D-UNSEEN — an evaluation subset, zero fits.* Frozen cached predictions restricted to
evaluation rows whose PIN never appears in that model's own training block (66.6 %–90.5 %
of each block survives; held-out n = **27,126**, 2025 n = **17,744**). Frozen answer to
"do the conclusions survive on never-before-seen parcels?" — **"Yes."** Direct β_log
ρ=0→100: CV_mean −0.1393→−0.0799, held-out −0.1609→−0.0936, 2025 −0.1801→−0.1129;
Surrogate −0.1386→−0.0195, −0.1590→−0.0142, −0.1794→−0.0288. Surrogate attains more
correction at 26/27, 24/27 and 24/27 ρ values.

*G5a / G5b — report the alert and its resolution honestly.* G5a fired on a **screening
grid** ("27 positive screening rhos + rho=0", 4× coarser than the frozen 82-point path)
with exactly one D-SNAP trigger, `T1_beta_log_sign_or_ordering`:
`"ALERT -- targeted local refinement required before any promotion"`. D-PURGE additionally
fired `T2_surrogate_dcor_rebound`, recorded **for information only**. Targeted refinement
then added 21 ρ (Direct) and 30 ρ (Surrogate) — 459 fits, 45 shards — and G5b returned
`status = NOT_CONFIRMED`:
`"no promotion -- the screening alert was a near-tie / near-zero artifact that did not
survive local refinement; the primary design stands"`. Per-evaluation:
`ordering_flips_any` 10 / 13 / 0 with **`ordering_flips_material` 0 everywhere**; on
held-out `sign_status_changed = true` but `sign_change_material = false` (the frozen
Surrogate path maximum `+0.0008383543937158` versus D-SNAP `−0.0001540536261184`, i.e.
**2.4× below τ = 0.002**). Frozen wording worth borrowing: *"the denser grid exposes
**more** near-ties, not fewer: 23 flips on the refined support, of which **0 are
material**."* `TAU_MATCH = 0.002` was frozen in `matched_beta_frozen.json` **before this
stage read any outcome** — say so, because it is what makes the non-confirmation
credible. `full_dsnap_regeneration_launched_in_this_run = false`; a confirmed promotion
would require separate authorization.

*The §5.6 repair — rebuild sentence by sentence, never unwrap.* An `oldrevisionblock` at
L2898–2910 wraps **both** superseded `\oldtext` **and accepted `\latesttext`** prose; the
Tier-A redefinition swallows the whole body, so accepted prose at 2900, 2903, 2906, 2907
and 2909 no longer prints. **Do not unwrap the block.** The suppressed prose also quotes
the span, regret and concordance values the audit finds unsupported (spans `[0.0494, 1.099]`,
`[0.00222, 0.954]`; 0/5 and 1/5 concordance; regret 0.005–0.056 etc.), so unwrapping
wholesale would resurrect them. Restore sentence by sentence, dropping every clause whose
numbers are unresolvable.

**Exactly one of the two anti-overclaim sentences is restored.**

- **Restore** the conclusion that the chronological folds *"do not identify a stable
  multi-metric operating point"* — recast so it stands on the surviving evidence rather
  than on the deleted transition analysis: the deleted analysis did not establish a stable
  multi-metric operating point, and nothing that survives establishes one either.
- **Delete** *"no materiality threshold is imposed."* **Corrected from an earlier draft of
  this plan, which proposed restoring it as a general statement of the paper's stance.
  That would be false.** The frozen analysis *does* impose a materiality threshold:
  `TAU_MATCH = 0.002`, frozen in `matched_beta_frozen.json` before Stage 3B read any
  outcome, and used to adjudicate matched-β attainment (worst CORE gap `1.0569e-03 < τ`)
  and every G5b sign and ordering decision (`ordering_flips_material = 0`,
  `sign_change_material = false`). A blanket "no materiality threshold is imposed" would
  contradict the two results the paper most depends on. The sentence had a strictly local
  referent — the span-regret table — and that referent is deleted, so the sentence goes
  with it. It may be restored **only** if a surviving passage gives it a local referent for
  which it is literally true.

Also delete the **active** candidate-region block at L2912–2918 in the same subsection —
see the coverage accounting in §H, where its four flagged tokens are itemized.

**Where τ = 0.002 must instead be stated positively.** Because the general
no-threshold sentence is gone, §5.7 and §5.8 must say what threshold *is* used and where it
came from: `TAU_MATCH = 0.002`, frozen before any outcome was read, applied to matched-β
attainment and to G5b materiality. That is a strength of the design, not a caveat, and
stating it is what makes the G5b non-confirmation credible.

**`C-results-004`, `C-results-005`, `C-results-006`, `C-appendix-002`, `C-metrics-003` ·
new §5.9 + App. A — standards-facing inference**

*Standing.* IAAO 2013 is **adopted/current** guidance; the May-2026 document is an
**Exposure Draft / proposed** guidance, never adopted. PRB's ±0.05 band belongs to the
adopted 2013 Standard; VEI's ±10% band to the Exposure Draft. Every frozen VEI row carries
`guidance_status = "May-2026 Exposure Draft / proposed guidance; not adopted IAAO
guidance."` and `not_a_compliance_determination = true`.
*The source, pinned.* sha256 `e950e00d…`, 2,013,844 bytes, 103 pages, internal title as
printed **"Exposure Draft May 2026"**, running header on every page
"EXPOSURE DRAFT - STANDARD ON RATIO STUDIES - MAY 2026", while the **filename says
`Mar2026`** — `ed2_source_manifest.json` is the authority. `committed_to_repo: false`,
`redistribution_permission_established: false`, so the PDF is cited, never redistributed.
Page anchors: App. D.2 p. 67, App. D.3 p. 67, App. D.4 p. 68, App. E from p. 78,
App. E.3 percentile-rank methodologies pp. 79–81.

**PRB under the adopted 2013 Standard.** `ci_level = 0.95`; display SE classical
(homoskedastic) with HC1 reported beside it as robustness. The classification rule,
verbatim: *"the ENTIRE 95% CI must lie outside a band before that band is deemed exceeded;
a CI that merely crosses a threshold is 'overlaps_pm005', never evidence of exceeding
it"*. The four state labels are `within_pm005`, `overlaps_pm005`,
`outside_pm005_but_not_pm010`, `outside_pm010`, plus null for NOT_ATTAINED. Counts over all
480 rows: 186 / 40 / 193 / 21 / 40. Over the 96 standards-facing rows: 33 / 8 / 44 / 3 / 8.

**Terminology constraint on the ±0.10 band.** The frozen config key is
`unacceptable_pm010`, but that is a **project reporting category**, and I have not verified
that "unacceptable" is the literal normative term for ±0.10 in the adopted 2013 Standard.
So the manuscript must **not** call ±0.10 an "unacceptable band". Report the arithmetic and
the frozen classification instead, and say plainly that the four state labels are this
project's reporting categories rather than IAAO terminology unless the adopted text is
checked and found to use them. *(Corrected from an earlier draft of this plan, which wrote
"inferentially outside the unacceptable band".)*

Two values worth reporting explicitly:
- **Cell A on the 2025 forward sample is classified `outside_pm010`** — PRB
  `−0.10604403342505853` with 95% CI `[−0.11092599784762935, −0.1011620690024877]`, so
  **the entire 95% CI lies below −0.10**. State it that way. It is the strongest single
  standards-facing statement the paper can make about the unpenalized model, and it needs
  no normative adjective to land. Two further cells share the classification:
  `A-posthoc b=0.9999219431164414` (`−0.10610859819500718`) and the Direct ρ≈0.01 display
  anchor (`−0.10489636870317365`).
- **Surrogate at the ρ≈10 anchor on 2025** is `within_pm005` with PRB
  `0.00013450725157726495` and CI `[−0.004271510858452601, 0.004540525361607131]` — this
  is what the displayed `0.000` actually is. Report it as a rounded estimate with its CI,
  and never as exact neutrality.
`pooled_oof` PRB is **D3 row-balanced WLS with SEs clustered on `row_id`, reported as a
SENSITIVITY, not the standards-facing result**; `d3_pooled_oof_reweighting_effect` max
|Δ| `0.0027225657356132`. And the frozen table records
`fold_as_iid_replicates: "NEVER -- no mean +/- SD/sqrt(7) appears in this table"`.

**VEI under the Exposure Draft — the Step-5/6/7 distinction, and what it actually shows.**
Step 5 gates on the point estimate against ±10%; Step 6 asks whether the first- and
last-percentile-group 90% median CIs overlap; Step 7 computes
`100 × (lower CI of the higher-median PG − upper CI of the lower-median PG) / sample median`
and rejects only if that exceeds 10%. Frozen gate counts over all 480 rows:

| step | outcome counts |
|---|---|
| Step 5 | 279 escalate · 117 stop within ±10% · 84 n/a |
| Step 6 | 279 no-overlap escalate · 117 not run · 84 n/a |
| Step 7 | **228 reject null** · **51 fail to reject** · 117 not run · 84 n/a |

Over the 88 evaluated standards-facing cells: 25 stop at Step 5, **63 escalate**, 52
reject the null, **11 fail to reject at Step 7**, and `ci_overlap_stop_at_step6 = 0`.
VEI range `−29.78552349345996` … `+14.769870796387837`; Step-7 significance on those cells
`7.941610153507715` … `26.592697932630788`.

**Two findings here that the paper should state, because they are exactly why
`|VEI| > 10%` is not an inferential finding:**

1. **11 of 63 standards-facing escalations fail to reject under the draft's own test.**
   A point estimate outside ±10% is a *trigger*, not a conclusion.
2. **The inferential distinction occurred at Step 7, not Step 6.** Supported wording, to
   be used as close to verbatim as the prose allows:

   > In this application, none of the 63 standards-facing configurations that escalated
   > beyond Step 5 stopped at Step 6; the inferential distinction therefore occurred at
   > Step 7.

   **Do not generalize this.** An earlier draft of this plan wrote "at CCAO sample sizes
   Step 6 is effectively non-binding" — that is a general claim about the procedure which
   this single application cannot support. Any explanation in terms of sample size or
   narrow median intervals (roughly 3,800 observations per decile) must be marked clearly
   as descriptive or suggestive of why this occurred here, never as an established property
   of Step 6.

**`pooled_oof` carries no ED2 inference.** All 48 pooled-OOF rows carry
`evaluation_role = not_applicable`, null VEI/step columns, and the single-source reason
string: *"ED2 App. D.2 is a rank-based order-statistic CI on distinct observations and has
no row-balanced analogue; the pooled-OOF sample contains 20,988 unique rows twice, so the
ED2 procedure is not applied there. No weighted variant was invented."* The literal token
is `NOT_APPLICABLE_FOR_ED2_INFERENCE` (`ed2_cells_applied = 387`,
`ed2_cells_not_applicable = 43`).
*Reinforcing evidence for the two prohibitions:* the string **"weighted median" appears
nowhere in the 103-page draft**, and "effective sample size" appears **only** inside the
§D.3 *weighted-mean* CI section. The project's **D3** row-balanced coordinate and the
draft's **§D.3** weighted-mean CI are unrelated constructions and must never be conflated.

**Five documented ED2 interpretations/deviations that must be disclosed, not smoothed over:**

| id | what | status |
|---|---|---|
| **ED2-I-1** | For even `n` the median is not an array element, so "count up and down the array from the median" does not fix the two anchor ranks; the draft gives no worked large-sample D.2 example. Resolved by counting outward from the two central order statistics (lower from rank `n/2+1`, upper from rank `n/2`). | *documented interpretation, not a verbatim rule* |
| **ED2-I-2** | Step-7 group scope = the first and last percentile groups carried forward from Step 6. | resolved from the document text |
| **ED2-I-3** | Reject iff Step-7 significance `> 10.0` strictly; exactly 10.0 yields fail-to-reject and sets `ed2_boundary_exact`. Observed `False` on every evaluated row. | implemented convention |
| **D-P1-5** | **ED2's printed Step-2 formula omits the 0.50 multiplier on the `AV/Median` term, contradicting its own prose ("gives equal weight"); taken literally it roughly doubles the AV contribution.** P1 implements the prose, `proxy = 0.50·SP + 0.50·(AV/median_ratio)`, conforming to the 2013 Standard. | documented deviation from the draft's *printed* formula |
| **D-P1-6** | The percentile-rank method used is `array_split`, which is neither R6 nor R7; it is required for the Step-5 value to reconcile with the frozen VEI artifacts. For `n` divisible by 10 (held-out 38,290 → ten groups of 3,829) all three coincide; where it is not (2025 forward 26,641 → one group of 2,665 and nine of 2,664) group boundaries may shift by at most one observation. | documented deviation |

The manuscript's existing `\newtext` paragraph on the equal-weight proxy (currently near
L587) already states most of D-P1-5 correctly — **keep it and cite
`configs/ed2_vei_procedure.json`.**
*Fidelity to quote:* `max_abs_step5_minus_canonical_vei = 0.0`; frozen reconciliation
`n_compared = 234`, `max |frozen − recomputed| = 7.105427357601002e-15`. And **D-P1-11**:
the committed CSV round-trips float64 to a worst 16 ULP / `3.553e-15` on `VEI_step5`, so
any bitwise claim must reference the parquet twin, not the CSV.
*Must disappear.* "ED2 is adopted guidance"; "weighted median"; "project D3 is ED2
Appendix D.3"; any equation of `|VEI| > 10%` with an inferential finding; and
`mean +/- SD/sqrt(7)`.

**Counting units — now four frames, never mixed in one sentence.** The vocabulary is
`display_entry`, `unique_realization`, `standards_facing_configuration`, `evaluation_cell`.

| frame | unit | counts |
|---|---|---|
| all ED2-applicable | `display_entry` | 396 evaluated · 279 escalated · 228 reject-null |
| all ED2-applicable | `unique_realization` | 387 · 270 · 219 |
| standards-facing only | `standards_facing_configuration` | 88 evaluated · 25 stop at Step 5 · 63 escalated · 52 reject · 11 fail to reject |
| bootstrap-sensitivity scope | `display_entry` → `unique_realization` | 63 → 61 |

The 396/387 and 279/270 and 228/219 gaps are all exactly **9**, because
`fit:LGBCovPenalty:1fb838f7d6bfda88` appears twice in the display set — once as the Cell-C
reference and once as core j=0 Direct — across the nine ED2-applicable blocks. The 63→61
gap is the same duplicated realization. Prefer `unique_realization` for methodological
summary statements; use `display_entry` only where the claim is specifically about display
roles. Structural shape: 48 display entries × 10 evaluation blocks = **480 rows** in each
P1 table; 44 attained × 9 ED2-applicable blocks = **396** evaluated; the 84 unevaluated
split as 44 pooled-OOF attained + 40 NOT_ATTAINED.
**One labelling trap:** the headline field
`display_set.distinct_realizations_incl_not_attained_key = 43` is computed as
`len([k for k in realizations if k != "NOT_ATTAINED"])` — it **excludes** the key its name
mentions. The `realizations` map itself has 44 keys. Write **43 fitted realizations**;
never "44 realizations".

**`C-appendix-003` · ADD · P0 · Conf HIGH · App. A (smearing), pointer in §5.9**

Report as **SENSITIVITY only**, never as a primary result or a new canonical model.
*Sign.* `s = Σ w·exp(u) / Σ w` with `u = y_true_log − y_pred_log = −e`; applied as
`y_pred_log → y_pred_log + log(s)`. The repo residual convention is
`e = y_pred_log − y_true_log`, so the Duan factor uses the **opposite** sign from `e` and
from the dCor estimator — the two must not be conflated. `s = mean(exp(e))` is recorded as
the `forbidden_formula`.
*Frozen values.* `s_min 1.0429306797423497` (A-posthoc at b=1.16037),
`s_median 1.0602010551302514`, `s_max 1.1426916234337632` (Surrogate ρ=100); reference
cells C `1.0623763218826567` and A `1.0623227151201418`; implied price-level shift
**4.293067974234965 % to 14.269162343376319 %**. Estimated on folds 1–7 only,
`sum_weights = 130165.0`, `max_abs_rowweight_minus_one = 0.0`,
`oos_used_in_estimation = false` on all 43 rows; `s_recomputed_in_apply = false`.
Estimating `s` on the held-out or 2025 block is *refused by assertion*, not by convention.

*The invariance result is three-way, not two-way — get this right.* Over 430 cells and
6,450 metric checks, `n_flagged = 0`:

| class | metrics | behaviour |
|---|---|---|
| `invariant` | COD, COV, PRD, PRB, VEI, MKI, β_log, dCor, Δ_NL, `Cov(e,y)` | unchanged; worst relative diff `1.0011069074835367e-11` (dCor) |
| `scales_by_s` | median ratio, mean ratio, weighted-mean ratio | scale exactly by `s`; worst relative diff `1.2638783260013826e-15` |
| `moves` | R²_price, MAE_price, MAPE | change materially **by design** |

Largest moves: R²_price |Δ| `0.05769600282085874` (A-posthoc b=1.16037, fold 3:
0.7605 → 0.7028); MAE_price |Δ| **\$20,809.35** (Surrogate ρ=100, fold 6:
\$72,973 → \$93,783); MAPE |Δ| `0.05811496835853305`. **All ED2 verdicts are unchanged**:
387 of 387, `all_verdicts_unchanged = true`, `max_vei_rel_diff 2.845607125668017e-13`.

*Why β_log is invariant and RMSE_log is not — state both, because a prior draft got it
wrong.* `e → e + log s`, and `Cov(e + k, c_y) = Cov(e, c_y)` because `c_y` is centered;
confirmed empirically at ratio exactly 1.000000000. `RMSE_log`, by contrast, is
*"NOT mathematically invariant to adding a constant. It is simply not recomputed: Duan
smearing is a post-exponentiation price-scale retransformation sensitivity and does not
alter the canonical log prediction."* P1 records this as deviation **D-P1-4**, an
explicitly corrected earlier error.
*Must disappear.* "s = mean(exp(e))"; "RMSE_log is smearing-invariant".

*The forward-2025 limitation, verbatim and non-negotiable:*
> The 2025 forward evaluation's fitting set is the 382,897-row production block, which has
> no out-of-fold analogue. The development-estimated `s` is applied there unchanged. This
> is an explicit assumption and a stated limitation, not a validated property.

*The appearances-versus-rows trap, with its frozen warning string:* the P0 artifact key
`m_i_distribution = {'1': 109177, '2': 41976}` is valued in **appearances**; 41,976
appearances come from **20,988 duplicated unique rows** (151,153 appearances → 130,165
unique). Never write "multiplicity 2 gives 41,976 rows". P0's own
`POST_G3_ADJUDICATION.md` §1 mislabels these as counts of *sales* — the manuscript must
not inherit that error.

**`C-appendix-001` · KEEP→ANSWER · P0 · Conf HIGH · App. A (dCor estimator)**
Replace the promise with the answer, field by field from `dcor_estimator_facts.json`:
`dcor` **0.6**, `dcor.distance_correlation(e, y, method="auto")`, exponent **1**,
**BIASED / V-statistic (double-centered)**, `bias_corrected_passed = false`, no
subsampling, residual convention `e = y_pred_log − y_true_log`. The artifact itself names
`paper_v17_option1.tex:3204-3206` as the manuscript site.

---

### E.8 Discussion, limitations, certification (Tier B3 / B4)

**`C-discussion-002` · REWRITE · P0 · Conf HIGH · §6.2 Limitations**
Add each of these explicitly — none is currently stated:

1. Gate **E.4 = INDETERMINATE** (curvature ratio 40.04 vs 1.2; M 0.0118 vs 0.01).
2. The **native-versus-custom execution-path divergence** (A↔B holds; B↔C does not; pinning
   does not fix it) — so A-to-penalized contrasts are workflow-benchmark comparisons.
3. The **fold-6/fold-7 validation overlap** affecting the pooled coordinate.
4. **G5a = ALERT** and **G5b = NOT_CONFIRMED**, with PASS_PRIMARY_STANDS.
5. The **candidate-region and transition numbers are not reproducible** from the frozen
   evidence, and are therefore not reported.
6. **Post-hoc centered rescaling matches much of the Direct path**, so no superiority claim
   for in-processing is available.

Keep the existing limitations on sale-based evidence, the mechanically negative
`Cov(e*,Y) = −E[Var(Y|X)]`, and the fixed-994-tree scope. Retire the two now-false
promises in §6.2 (the "requires an initialization-aligned parity rerun" clause and the
"omits the closest post-processing comparator" clause — both were executed).
*Must disappear.* "mean +/- SD/sqrt(7)"; "ED2 is adopted guidance"; presenting fold SD as
a standard error.

**`C-discussion-001`, `C-discussion-004`, `C-discussion-003` · KEEP · Conf HIGH**
Keep the bounded framing (fixed 994-tree custom-objective paths, not a deployment
decision). **`tab:roles` and the novelty boundary must survive intact — all five
non-novelty statements and all eight defensible-contribution items.** Option 2 may be read
for **wording precedent only**; no number, threshold, empirical result or conclusion may
be imported from it unless it independently resolves through the frozen maps.

**`C-certification-001` / `C-certification-002` · ADD · P0/P1 · Conf HIGH · App. G**
State: P0 **153/153** at its immutable tag (Stage-1 28 + G2 36 + G3 36 + Stage-3 29 +
Stage-3B/G5b 24 = 153), P1 **72/72** (64 scientific — 12 PRB + 5 smearing-sign + 28 VEI +
19 smearing-apply — plus 8 report-consistency), both subtrees byte-identical to their tags
at the integration HEAD, and **no manuscript file edited by either stage**. Frozen display
set: **48 display entries, 44 attained, 4 NOT_ATTAINED, 43 fitted realizations, 10
evaluation blocks → 480 rows** in each P1 inference table, with the selector
`(display_kind, family, j, ext_target, rho, role_label, evaluation)` verified unique on all
480 rows of both tables, and `frozen_before_any_result_is_read = true`.

**Scope discipline: two layers, and only one of them is in the paper.**
The appendix is a *manuscript-facing* reproducibility statement, not an internal audit log.
Split the material explicitly.

**`MANUSCRIPT_RELEVANT_CERTIFICATION` — belongs in Appendix G, concise:**
- the two immutable checkpoints, named: `p0-major-revision-final-20260907` and
  `p1-inferential-reporting-final-20260907`, plus `tier-b0-final-20260907` for the evidence
  bridge;
- the final certified suite outcomes **at those tags**: P0 153/153, P1 72/72;
- the frozen evidence manifest: `FINAL_EVIDENCE_MANIFEST.json`, 370 artifacts indexed, of
  which 11 legacy inputs are recorded `present_in_worktree = false` with hashes **quoted**
  from the frozen index rather than recomputed — so the legacy prediction trees and the
  CCAO parquet are hash-pinned but **not redistributed**;
- the frozen display-set shape with its counting unit (48 display entries / 44 attained /
  4 NOT_ATTAINED / 43 fitted realizations / 10 evaluation blocks → 480 rows per P1 table,
  selector verified unique), and `frozen_before_any_result_is_read = true`;
- manuscript-revision provenance: that Tier B0 edited no file under `paper/`, and that this
  Tier-B pass is the writing stage against a frozen evidence map.

**`REPOSITORY_AUDIT_DETAIL` — stays in the repository evidence package, out of the paper**
(it is already recorded in `certification/CERTIFICATION.md` and
`provenance/p1_artifact_hashes.json`; the appendix points there rather than reciting it):
the historical `stage1_frozen_hashes.json` 48/49 and `output_artifact_hashes.json` 118/125
index mismatches; the stale `Status:` / `authorized_scope` lines in P0's `README.md` and
`protocol_p0_validation.yaml`; the superseded `P1_CHECKPOINT.md` and its wrong Stage-3B
"86/86"; and the environment-specific suite variants (152/153 content-only checkout,
152/153 on the P1 branch, 140/153 in this worktree, all with **content defects: 0**).
*(Corrected from an earlier draft of this plan, which put all four of these in the
manuscript appendix.)*

**Three constraints that survive the trimming, because each is a negative claim the paper
must avoid making rather than a detail it must recite:**
1. Do **not** write that every P0 index is byte-exact. Only the final Stage-3B/G5b index
   is (28/28). Satisfied by not making the claim; the detail lives in the audit layer. This
   is the reconciliation with `C-certification-001`, whose required message is precisely
   that the byte-exactness claim would be false.
2. Do **not** quote a suite count measured in an environment lacking the gitignored parquet
   twins as if it were the certification. Cite the tag figures; if any other figure is
   mentioned at all, name its environment.
3. Do **not** cite `protocol_p0_validation.yaml` or P0's `README.md` for authorization
   state — cite the stage-specific gate JSONs and the `POST_G1` / `POST_G3` / `POST_G5`
   adjudications.

**`C-appendix-005` · REWRITE · P0 · Conf HIGH · App. G**
Check the artifact list item by item against `FINAL_EVIDENCE_MANIFEST.json` (370 artifacts;
11 legacy inputs recorded `present_in_worktree = false` with hashes **quoted** from the
frozen index, never recomputed). Replace "should contain" with an assertive description for
everything archived, and **delete items that are not released rather than promising them**.
State plainly that the legacy prediction trees and the CCAO parquet are hash-pinned but not
redistributed.

**Optional cleanup (Tier B4.2) · Conf MEDIUM**
`C-intro-001/003/004` (three redundancies), `C-metrics-001` (dangling pointer to the
deleted model-selection section — repoint or delete; **do not reintroduce a
model-selection section**), `C-appendix-004` (remove the v2.1 versioning promise along
with the unsupported screening numbers), `C-appendix-006/007` (delete the inert `\iffalse`
ATTOM material — **do not resurrect it**), `C-appendix-008` (keep Appendix H's boundary
statement; the duplicate `\label{app:attom}` goes with the inert block).

---

## F. Decisions taken, and what remains open

### F.0 Three decisions you have now made — folded into the plan above

**(1) The Linear-regression benchmark: remove the numbers, keep the framing.**
No frozen artifact resolves any Linear value (34 flagged tokens across
`tab:ccao_baseline_results` (16) and `tab:ccao_baseline_complementary` (18), plus the
Linear rows in both rebuilt path tables and the Linear curves in two figures). Every
Linear numeric cell and curve is deleted. A qualitative sentence survives — that
township-level linear regression historically preceded the LightGBM workflow, cited to
`CCAOModelResAVM2026` and the four newly loaded historical vertical-equity references — but
it carries **no numbers**. §1 and §2.4 are reframed from a *contrast* to a *level
statement* on Cell A alone, which establishes the motivation by itself:

> Ordinary LightGBM attains held-out R²_P = 0.894 while exhibiting PRB = −0.091,
> VEI = −26.5%, β_log = −0.150, and COD = 21.6% — outside the adopted [5,15] range; and on
> the 2025 forward sample the PRB 95% CI lies entirely outside ±0.10. Predictive accuracy
> alone does not remove the ratio trend.

Consequences to execute: `tab:ccao_baseline_results` and `tab:ccao_baseline_complementary`
drop from four columns to two (held-out, 2025); the Linear curve is dropped from
`fig:baseline_motivation` and from `fig:vei_group_profile_placeholder` — **note that this
makes the VEI-profile figure an ordinary-LightGBM-only panel, which also simplifies its
caption**; the four `fig:tradeoff_*` figures lose Linear as a context anchor; and §1's
"rather than choose between the two baselines" model-design framing is rewritten to a
within-LightGBM design question. No new computation; fully Tier-B0 compliant.
*Recorded for later:* re-deriving a frozen Linear baseline remains available as a
separately authorized follow-up if the contrast is ever wanted back.

**(2) The five display anchors: split into two tables.**
`tab:path_anchor_summary` becomes two tables, so the provenance boundary is structural
rather than a footnote:

- **`tab:path_anchor_frozen`** — the two frozen anchors (ρ≈0.954095, ρ=100) at reproduction
  tier R1, with prediction plus PRD and MKI. Cell A row retained as the workflow benchmark.
- **`tab:path_anchor_standards`** — PRB and VEI at **all five** anchors for both families,
  with PRB attributed to the **adopted 2013 Standard** (±0.05) and VEI to the **May-2026
  Exposure Draft** (±10%), labelled differently because they have different standing. This
  preserves the audit's strongest resolvability result: all 40 printed values re-derive
  exactly, 234 values reconciled per measure, max |Δ| 7.6e-16 (PRB) and 7.1e-15 (VEI) —
  including at the three legacy anchors that no other metric can reach.

Both carry the §0.1 two-device highlighting convention. The Direct ρ≈10 VEI = −10.5%
"just outside the proposed band" qualification and the Surrogate ρ≈10 PRB = 0.000
rounding/CI qualification both live in `tab:path_anchor_standards`.
`tab:path_anchor_complementary` follows the same split logic: the six resolvable
complementary metrics at the two frozen anchors only, with Δ_NL and dCor dropped for every
positive ρ.

**(3) Revision scaffolding: strip at Tier B4.2, after the prose is stable.**
Remove the Tier-A override block, the 336 active `\latesttext` and 67 `\newtext` call sites
(unwrapped to plain prose), the 169 active `\oldtext` call sites (deleted — they already do
not print), both active `oldrevisionblock` environments, the remaining `\todo` sites, and
the commented and `\iffalse` regions. Nothing is lost: git history and
`paper/paper_analysis/paper_v17/superseded_text_archive.md` preserve every suppressed
passage verbatim. Ordering is load-bearing — **B4.2 runs after B4.1**, so no replacement
prose is ever written against a macro that is about to disappear.

### F.1 Title — your call at Tier B4.1, not now

Current: *"Covariance-Based Regularization for Regressivity in Machine-Learning Real
Estate Valuation."* This reads as a method paper; the frozen scope is a CCAO-centered
applied-methods **and diagnostic** paper about a **first-order component**. Candidates:

- **(a)** *Controlling a First-Order Component of Price-Related Regressivity in
  Machine-Learning Mass Appraisal* — most faithful to the frozen scope.
- **(b)** *Covariance-Guided Regularization in Machine-Learning Mass Appraisal: Mechanism,
  Implementation, and a Temporal Failure-Mode Audit* — signals the audit contribution.
- **(c)** Keep the current title.

I recommend **(a)**. Your call at Tier B4.1, not now.

### F.2 The still-open CCAO extract provenance — the one input I need from you

`PREFLIGHT.json` hash-pins **which** extract
(`data/CCAO/2025/training_data.parquet`, sha256 `b1fc00b5…`, 215,400,916 bytes), so *which*
extract is unambiguous. But a **citable public source and a build or retrieval date are
still missing**, and the file is in no checkout. I will not invent either.

**This does not block the revision.** Supply them at any point before Tier B1.4 if a
citable/public source exists and a build/retrieval date is recoverable. If they are not
available by B1.4 I proceed with the explicit limitation rather than holding the pass: the
sha256, the byte count, and a plain statement that the extract is hash-pinned, not public,
not redistributed, and of unrecorded build date. Nothing is invented either way.

Two smaller release-side gaps of the same kind, both `REMAINING_GAP_RELEASE` in the Tier-B0
closure map, which I will resolve by checking against `FINAL_EVIDENCE_MANIFEST.json` and
deleting rather than promising anything not actually released (todo #18, #19).

---

## G. Claims that cannot yet be written from the frozen evidence

These are **not** to be written, hedged, or relabelled. Each needs new frozen evidence or
must stay out.

| claim | why it cannot be written |
|---|---|
| Any Linear-regression number | no frozen artifact resolves any of them (decided: removed, F.0) |
| R²_P / MAE / MAPE / RMSE_log / PRD / MKI at ρ ∈ {0.0105, 0.1, 10.481} | legacy V6/V12 trees gitignored, hash-pinned only |
| Δ_NL or dCor at **any** positive ρ | absent from `frozen_artifact_reproduction.csv` |
| The Surrogate dCor non-monotonicity (0.382 → 0.250 → 0.267) | positive-ρ endpoints unresolvable |
| Candidate-region activity onset / upper guardrail as a *result* | grid coordinates exist; the screen result is reproduced nowhere |
| 7/7 LOFO stability, guardrail indices 53–56 / 53–55, LOFO endpoint ranges | reproduced nowhere |
| Five-metric transition event locations, spans, log₁₀ widths | reproduced nowhere |
| Span-regret values (all 100 tokens) | reproduced nowhere |
| Held-out / 2025 exact concordance ratios (0/5, 1/5) | reproduced nowhere |
| CV Δ_NL minima (Direct 0.08685 at ρ=100; Surrogate 0.08677 at ρ≈1.677) | not in the numeric map |
| "Direct is effectively gradient-only" | E.4 INDETERMINATE |
| "retrained Direct ≡ post-hoc rescaling" | matching ≠ equivalence |
| Weight concentration / ESS as the *proven cause* of Surrogate behaviour | association only |
| Repeat-parcel information *causes* or *masks* regressivity | D-PURGE is an oracle sensitivity |
| Any IID significance statement over the seven folds | folds overlap; SD is not an SE |
| That D1 is "unaffected" by the fold-6/7 validation overlap | superseded by `POST_G3_ADJUDICATION.md` §2 |
| A general "no materiality threshold is imposed" | false: τ = 0.002 governs matched-β attainment and every G5b decision |
| "Step 6 is non-binding at CCAO sample sizes" or any general property of ED2 Step 6 | one application cannot establish it; only the in-this-application form is supported |
| ±0.10 described as an "unacceptable band" | `unacceptable_pm010` is a project reporting key, not verified IAAO normative wording |
| The `f0bar_T`-centered map presented as the primary comparator | it is `centering_sensitivity.alternative_map`, anchors only |
| That Direct *cannot* reach β_log neutrality for this base learner | non-attainment is a property of the **evaluated** 82-point path, not of the objective or the learner at unexamined ρ |
| `b = 1/R²` as a general LightGBM post-hoc root | linear/projection special case only |
| Any ED2 compliance determination, or `\|VEI\| > 10%` as an inferential finding | ED2 is an Exposure Draft; Steps 5–7 govern |
| Eigenmode attribution of the high-ρ S-shape | no projection diagnostic exists |
| Any external-jurisdiction / ATTOM evidence | `OUT_OF_SCOPE_FOR_B0` |
| CCAO institutional adoption | no public institutional record |
| A preferred ρ, safe region, deployment point, or recommended ρ | no penalty strength is selected anywhere |
| Bitwise/exact-serialisation claims about the committed P1 CSVs | they round-trip float64 to a worst 16 ULP / `3.553e-15`; bitwise claims must reference the parquet twin, which is gitignored |
| "44 realizations" | 43 fitted realizations; the 44th map key is `NOT_ATTAINED` |
| "multiplicity 2 gives 41,976 rows" | 41,976 is *appearances* from 20,988 duplicated unique rows |
| "every P0 index is byte-exact" | only the final Stage-3B gate index is (28/28); two early indexes are 48/49 and 118/125 |
| Any Δ_NL claim about the pooled-OOF sample without its composite identifier caveat | Δ_NL for D2 uses a composite `"fold\|row_id"` identifier supplied *only* for that sample |

---

## G.1 Cross-cutting hazards in the frozen record itself

Five places where a frozen artifact will mislead a careless writer. Each is a real trap I
hit while reading.

1. **A frozen report contains a statement its own adjudication supersedes.**
   `CENTERED_SPREAD_COMPARATOR_REPORT.md` §4 says *"D1, the paper's primary CV coordinate,
   is unaffected (True)"*. `POST_G3_ADJUDICATION.md` §2 overrides it: *"D1 must not be
   described simply as 'unaffected.' The precise statement is five-part."* **Adjudications
   beat reports.**
2. **Two authority files carry a stale scope line.** `protocol_p0_validation.yaml`
   (`authorized_scope.this_run = STAGE_1_THROUGH_GATE_G1`) and P0's `README.md` `Status:`
   line both predate Stages 1.5/2/3/3B. Their `not_authorized_in_this_run` lists name
   experiments that subsequently ran under later authorizations.
3. **P0 keeps the legacy Stage-1 cell names by design.**
   `protocol_p0_validation.yaml:fixed_cell_names` still says `B: "Parity-aligned native
   L2"`, `C: "Custom rho=0 origin"`. Forward display names live only in
   `configs/post_g1_reference_convention.yaml`. Stage-1 artifacts were deliberately not
   rewritten.
4. **Manuscript line references inside the P1 reports are stale in this worktree.** They
   cite `paper_v17_option1.tex:565`, `:639`, `:3057`, `:3204-3206`; the actual sites are
   now the dCor `\todo` at 3226–3228, the smearing sentence at 3079, the group-profile 90%
   CI sentence at 661, and the figure at 4302–4303. **Re-identify every passage by
   `latex_label` / `source_anchor` and the baseline excerpt sha256, never by line number** —
   which is exactly what `SELECTOR_GRAMMAR.md` §4 requires.
5. **Two field names lie about their contents.**
   `display_set.distinct_realizations_incl_not_attained_key = 43` *excludes* that key, and
   P0's `m_i_distribution` keys `{'1': 109177, '2': 41976}` are valued in *appearances*
   while `POST_G3_ADJUDICATION.md` §1 describes them as counts of *sales*.

Also worth recording: every P0 artifact records `cwd =
…/soft-vertical-equity-constrained-mass-appraissal` — P0 executed in the main repo, not in
this write worktree. Nothing in the plan depends on that, but a provenance sentence in
Appendix G should not claim otherwise.

---

## H. Verification — a Tier-B validator, not the frozen suites

### H.1 Why the frozen suites cannot be the gate

**The frozen B0 and P1 suites contain paper-immutability and HEAD-relative guards that
fail by design the moment the manuscript is intentionally edited.** Verified this session:

| guard | what it asserts | effect of a paper edit |
|---|---|---|
| `tests/test_b0_isolation.py::test_frozen_stages_and_paper_are_untouched` | `git diff --stat TIER_A_COMMIT HEAD -- paper` is empty | **fails on the first edit** |
| `tests/test_b0_isolation.py::test_git_status_shows_nothing_outside_the_tier_b0_area` | working tree clean outside the B0 area | fails while editing |
| `code/b0_common.py::TEX_SHA256` (`13c84ce7…`) | pins the baseline manuscript hash | any live-file rehash diverges |
| P1 `test_no_protected_path_written` (×3 suites), `_PROTECTED` includes `REPO/"paper"` | nothing under `paper/` is written | **fails on the first edit** |
| P1 `test_p1_headline_numbers::test_reports_state_that_no_manuscript_file_was_edited` | the P1 reports still say no manuscript file was edited | stays true only because P1 didn't edit it |
| P0 `test_g2_assertions::test_only_gitignore_modified_outside_p0` | HEAD-relative scope guard | already failing on any additive commit |

So requiring "the frozen suites stay at 100%" is incoherent for a writing pass, and
**modifying any frozen P0/P1/B0 test to make it pass is prohibited.** *(Corrected from an
earlier draft of this plan, which made a 100% B0 suite the primary gate.)* Those suites are
still run **once, informationally, before B1.1** to record the pre-edit baseline; after
that their paper-facing guards are expected to fail and that expectation is documented,
not fixed.

### H.2 The Tier-B manuscript validator

New, and the actual gate. Lives at **`paper/paper_analysis/tier_b_validation/`** — inside
`paper/`, so it is Tier-B-owned; it **reads** the frozen B0 evidence maps and **never
writes to or modifies anything under `analysis/`**. Built at **B1.0, before any manuscript
edit**, so every later stage has a gate to run.

Checks, at minimum:

1. **Numeric provenance — recomputed, never trusted.** Every manuscript-facing empirical
   number resolves to a frozen artifact. Two resolution paths, because the frozen numeric
   map indexes only the *baseline* manuscript's tokens: an existing `N-*` map row, **or** a
   Tier-B provenance ledger entry for numbers newly added by this pass (matched-β,
   centered-spread, temporal, ED2 counts, Cell C rows). The ledger lives beside the
   validator, never under `analysis/`.

   **The ledger is an input to be verified, not an authority.** A hand-authored value in it
   proves nothing. Each entry carries seven fields:

   | field | purpose |
   |---|---|
   | `artifact_path` | the frozen file, under `analysis/` |
   | `artifact_sha256` | pin, so a silently changed artifact is caught |
   | `selector` | the row/column address, in `SELECTOR_GRAMMAR.md` form |
   | `raw_value` | the literal decimal text as stored |
   | `value_transform` | from the closed vocabulary (default `identity`) |
   | `rounding_rule` | from the closed vocabulary |
   | `rendered_value` | what the manuscript actually prints |

   For every entry the validator must **reopen the frozen artifact, verify its sha256,
   execute the selector, re-extract `raw_value`, recompute the transform and the rounding,
   and compare the result against the manuscript-facing value it found in the `.tex`.** A
   mismatch at any step is a hard failure. Values are carried as literal decimal text and
   converted with `decimal.Decimal`, never through float — the same discipline Tier B0
   applies, and for the same reason: the committed CSVs round-trip float64 only to about
   `3.55e-15`. This makes the ledger a *checked* derivation rather than a second place a
   wrong number can live.
2. **No `FLAGGED_UNSUPPORTED` result remains** — see the accounting in H.3.
3. **A/C reference semantics** — no A-referenced difference presented as a penalty effect;
   no row pairing `reference_cell=A` with `PENALTY_ISOLATING` or `C` with
   `WORKFLOW_BENCHMARK`; both A and C present as explicit reference rows (or an equally
   explicit device); forward display names used, never legacy Stage-1 labels.
4. **D1 / D3 semantics** — D1 stated primary, D3 stated as the one-sale-one-vote
   sensitivity, D2 not elevated; the string "unaffected" never attached to D1 and the
   overlap qualification present; no `mean +/- SD/sqrt(7)` or IID reading of `CV_sd`.
5. **`NOT_ATTAINED` preservation** — the four frozen states present verbatim with blank
   metric cells; no interpolated value anywhere in the matched-β tables.
6. **ED2 guidance status and counting units** — every ED2 mention carries "Exposure Draft"
   or "proposed"; every count names its unit from the closed vocabulary; no sentence mixes
   `{396,279,228}` with `{387,270,219}`, nor either with the standards-facing frame
   `{88,25,63,52,11}` or the bootstrap scope `{63,61}`, without naming the unit.
7. **Forbidden wording** — all 17 phrases from `spec/forbidden_wording.yaml` plus the 3
   forbidden literals, read from that file rather than duplicated; an occurrence is legal
   only inside a sentence carrying a `prohibition_markers` cue.
8. **No unsupported candidate-region assets** — no `_candidate_region` graphic referenced;
   no candidate-region / activity-onset / upper-guardrail / transition-span sentence in any
   caption.
9. **Figure existence** — every `\safeincludegraphics` path resolves to a file that is
   tracked in git.
10. **Label / ref integrity** — no duplicate `\label`, no `\ref` or `\eqref` to a
    non-existent or deleted label.
11. **Citation-key integrity** — every active `\cite` key exists in a loaded `.bib`; every
    newly required key is cited at least once; `\addbibresource` count is 2 and no path
    under `analysis/` appears in the preamble.
12. **TODO closure** — no `\todo` remains after B4.2, and every one of the 19 historical
    sites is accounted for as closed, deleted, or converted to prose.

### H.3 Write-scope and frozen-subtree immutability — tested directly

Two independent checks at **every** checkpoint. Neither depends on a frozen guard that a
paper edit breaks.

**(a) Cumulative branch write-scope.** `git diff --name-only HEAD` sees only *uncommitted*
changes and is therefore insufficient once Tier-B commits exist. The binding check is
cumulative across the whole branch:

```
git diff --name-only tier-b0-final-20260907..HEAD
```

**Every** path it lists must be under `paper/`. Run it at every checkpoint, not just at the
end — a stray write in B1.2 is invisible to an uncommitted-changes check by the time B2.4
runs. Keep the uncommitted-scope check as well; it catches a stray write *before* it is
committed, which is when it is cheapest to undo.

**(b) Frozen-subtree immutability**, per tag, per directory:

```
git diff --stat p0-major-revision-final-20260907        HEAD -- analysis/p0_major_revision_validation
git diff --stat p1-inferential-reporting-final-20260907 HEAD -- analysis/p1_inferential_reporting
git diff --stat tier-b0-final-20260907                  HEAD -- analysis/final_manuscript_evidence
```

All three must be empty. This does exactly what the frozen guards were protecting.

(a) and (b) are complementary rather than redundant: (b) proves the three frozen subtrees
are untouched; (a) proves nothing *else* outside `paper/` was touched either — a new file
under `output/`, `utils/`, or a fourth `analysis/` subdirectory would pass (b) and fail (a).

### H.4 Coverage accounting — it closes at 4, and here they are

The arithmetic must close mechanically before any "zero unsupported" claim.
**356 − 134 − 184 − 34 = 4, not 0.** *(Corrected from an earlier draft of this plan, which
asserted 0 after B2.2.)* The four remaining tokens are, exactly:

| # | anchor | baseline line | token | what it is | disposition |
|---|---|---|---|---|---|
| 1 | `subsec:path_stability_results` | L2913 | `53` | Direct LOFO guardrail index range 53–56 | DELETE with the block |
| 2 | `subsec:path_stability_results` | L2913 | `56` | same | DELETE with the block |
| 3 | `subsec:path_stability_results` | L2915 | `53` | Surrogate LOFO guardrail index range 53–55 | DELETE with the block |
| 4 | `subsec:path_stability_results` | L2915 | `55` | same | DELETE with the block |

They are the leave-one-fold-out guardrail **indices** in the *active* candidate-region
block at L2912–2918 — the block that reports the screen result as fact. (The ρ endpoints in
the same sentences are not individually flagged because they exist in the frozen grid maps
as grid coordinates; the *claim* built on them is what no artifact reproduces. §0 Rule 2.)

**Resolution: the block deletion moves into B1.1**, where it belongs — it is unsupported
candidate-region material, and deleting it with the three candidate-region tables keeps the
"delete first" principle intact. Revised per-stage ledger:

| stage | anchors cleared | tokens | running total |
|---|---|---:|---:|
| — | starting | | **356** |
| B1.1 | `tab:rho_candidate_regions` 12 + `tab:transition_regret` 100 + `tab:transition_summary` 22 + `subsec:path_stability_results` 4 | **138** | 218 |
| B2.1 | `tab:path_anchor_summary` 65 + `tab:path_anchor_complementary` 119 | **184** | 34 |
| B2.2 | `tab:ccao_baseline_results` 16 + `tab:ccao_baseline_complementary` 18 | **34** | **0** |

`138 + 184 + 34 = 356`. The count reaches zero at B2.2, and the validator asserts it there
rather than being told to expect it.

### H.5 Other mechanical checks at every commit

- **Re-identification integrity** — every passage edited was located by `latex_label` /
  `source_anchor` / baseline-excerpt sha256, **not** by baseline line number
  (`SELECTOR_GRAMMAR.md` §4). After B1.1 every line number in the maps is stale by
  construction, including the ones in this plan.
- **Compile** — `latexmk` clean, no missing references, no new overfull hboxes in the
  rebuilt tables.

### H.6 Scientific checks, per stage

- **B1.0** — the validator runs green on the *unedited* manuscript except for the checks
  that are supposed to fail (356 flagged tokens, missing citations, unsupported assets). A
  validator that passes on the baseline is not validating anything.
- **B1.2** — `biber` reports zero missing keys; all ten newly cited keys print;
  `\addbibresource` count is 2 with no `analysis/` path in the preamble; `Cheng1974` has no
  DOI and no unverified fields; `Edelstein1979` shows `pages = {753}`.
- **B1.3** — E.4 numbers match `e4_verdict.json` exactly; the four zero-covariance denials
  still print; the three root concepts (`b_T*`, `b_zero_cvmean`, D2) are distinct and the
  D1 root is not described as a training relation.
- **B1.4** — the three forward display names appear verbatim; no legacy Stage-1 label is
  used as a display name; D1 primary / D3 one-sale-one-vote sensitivity, with the overlap
  qualification and **without** the word "unaffected".
- **B2.1 / B2.2** — every printed cell resolves via a map row or a ledger entry; **both A
  and C appear as reference rows**; bold-vs-A and †-vs-C are visually distinct, explained
  in the notes, and disclaimed as not-significance/not-compliance/not-selection; PRB and
  VEI present at all five anchors in `tab:path_anchor_standards`; the −10.5% band
  qualification and the `0.000`-with-CI qualification both present; flagged count reaches
  **0**.
- **B2.4** — the four `NOT_ATTAINED` states appear verbatim with blank metric cells and no
  interpolation; D1 primary / D3 sensitivity; the RMSE_log reversal stated at both
  β_dev = −0.03 and j=5; τ = 0.002 stated with its pre-registration.
- **B2.5** — the operating-point conclusion prints; **"no materiality threshold is
  imposed" does not appear**; no span/regret/concordance number reappears; the
  `oldrevisionblock` is gone rather than unwrapped; G5a ALERT / G5b NOT_CONFIRMED /
  PASS_PRIMARY_STANDS all present as recorded.
- **B2.6** — every ED2 count names its counting unit; the Step-6 statement is the bounded
  in-this-application form, not a general claim; ±0.10 is never called "unacceptable" and
  the Cell-A 2025 result is phrased as "the entire 95% CI lies below −0.10"; the smearing
  sign is `exp(y_true_log − y_pred_log)`; the three-way invariance split (`invariant` /
  `scales_by_s` / `moves`) is stated correctly and `RMSE_log` appears in none of them; the
  forward-2025 smearing limitation present; the VEI figure caption distinguishes all three
  90% interval concepts; all 8 assets swapped and all overlay sentences stripped.
### H.7 The B4.3 final gate — five items, and a frozen-suite rerun is not one of them

The pass/fail gate at B4.3 is exactly:

1. **Tier-B manuscript validator** (§H.2) — all twelve checks green, ledger entries
   recomputed from the artifacts.
2. **Compilation and reference checks** — `latexmk` clean; no missing or undefined
   references; no new overfull hboxes in the rebuilt tables.
3. **Cumulative paper-only write-scope check** — `git diff --name-only
   tier-b0-final-20260907..HEAD` lists paths under `paper/` and nothing else.
4. **Direct P0 / P1 / B0 subtree-immutability checks** — the three tag diffs of §H.3(b),
   all empty.
5. **Adversarial full-paper read** against `MANUSCRIPT_REVISION_SPEC.md` §0–§7, hunting six
   failure modes: a number with no verified provenance; an A-referenced difference
   presented as a penalty effect; an ED2 statement without "Exposure Draft" or "proposed";
   a count without its counting unit; a generalization from this application to a property
   of the ED2 procedure; and a path non-attainment stated as global impossibility.

Also at B4.3: Appendix G carries only the `MANUSCRIPT_RELEVANT_CERTIFICATION` layer and
points to the repository package for the audit detail.

**Frozen B0/P1 suite reruns are informational only.** *(Corrected from an earlier draft of
this plan, which listed "re-run the Tier-B0 suite" as a B4.3 action.)* They may be run to
record what the environment reports, but they are **never** a writing-stage pass/fail
criterion, **never** modified, and **never** cited as certification of this pass. The
certification of the science is what the three tags carry; the certification of the writing
is items 1–5 above.

**Never during this pass:** run model fits, LightGBM, or CV; submit Slurm; create new ρ
values; recompute matched-β; rerun P0/P1; read `output/` or `data/` to recover an
unsupported value; introduce external-jurisdiction evidence.

---

## Review pass 4 — execution control

One rule added, no other plan change: **§D.0**, two mandatory stops. This authorization
covers **B1.0 → B1.4 only**, with full validation and a compile after every commit, then a
hard stop before B2.1 and a nine-item report; and an analogous mandatory stop after B2.6
before B3.1. Stop markers are inlined in the commit tables so they cannot be missed while
working through the sequence.

## Review pass 3 — final execution-safety corrections

| # | correction | what changed |
|---|---|---|
| 1 | cumulative write-scope guard | `git diff --name-only HEAD` sees only uncommitted changes and is insufficient once Tier-B commits exist. **§H.3(a)** now runs `git diff --name-only tier-b0-final-20260907..HEAD` at every checkpoint and asserts every listed path is under `paper/`. The three frozen-subtree tag diffs are retained as **§H.3(b)**, and the two are documented as complementary: (b) proves the frozen subtrees are untouched, (a) proves nothing else outside `paper/` was touched either. |
| 2 | B4.3 gate wording | "Re-run the Tier-B0 suite" is **removed as a pass/fail action**. New **§H.7** states the five-item final gate: validator, compilation/reference checks, cumulative paper-only write-scope, direct P0/P1/B0 subtree immutability, adversarial read. Frozen suite reruns are informational only — never modified, never writing-stage certification. |
| 3 | NOT_ATTAINED interpretation | The global-impossibility phrasing is **removed** from all three places it appeared (matched-β section, storyline paragraph, key-sentence list). Bounded wording adopted: *Direct did not attain β_log ∈ {−0.06, −0.03, 0} over the evaluated frozen path; Surrogate did not attain 0 over its evaluated path.* Added to the unwritable-claims table, and to the B4.3 adversarial read as a sixth failure mode. |
| 4 | provenance ledger | The ledger is now **an input to be verified, not an authority**. Seven required fields (artifact path, artifact sha256, selector, raw value, transform, rounding rule, rendered value), and the validator must reopen the artifact, verify the hash, execute the selector, re-extract the raw value, recompute transform and rounding, and compare against what the `.tex` prints. `decimal.Decimal` throughout, never float. |
| 5 | table comparison semantics | "Improvement" is now defined **metric-by-metric** in a table: R² higher; MAE/MAPE/RMSE_log lower; PRB/VEI/β_log closer to zero; PRD/MKI/ratio metrics closer to their stated ideal; COD/COV toward less dispersion *within* the reference range, with the frozen caveat that unusually low dispersion warrants review; Δ_NL/dCor directional only, no assessor-standard range. Comparisons on unrounded canonical values. Markers still disclaim significance, compliance, selection, and any recommended ρ. |

## Review pass 2 — corrections applied after your earlier review

| # | correction | what changed |
|---|---|---|
| 1 | centered-spread map and roots | **Disagreement reported, frozen evidence followed.** The frozen `posthoc_comparator_convention.yaml` records `f_b(x) = ybar_T + b(f0 − ybar_T)` as `primary_map` / theorem-matched and "the only map used for the full path", with the `f0bar_T`-centered form as `centering_sensitivity.alternative_map` (anchors only, `full_second_path_scheduled: false`), bounded by `max\|f0bar_T − ybar_T\| ≤ 8.36e-06`. Your substantive point is adopted in full: both maps named, never conflated, sensitivity reported with its bound. The three root concepts are now tabulated separately, and **"the actual training covariance/variance relation" is removed** — `b_zero_cvmean` is built from *validation*-block quantities. |
| 2 | D1 overlap wording | Every "unaffected" removed, including the Tier-B0 spec's own phrasing for `C-design-005`. Replaced with the adjudicated statement: primary, definition unchanged, every prediction genuinely out of fold, but **not an independent-observation aggregate** and not to be called unaffected. D3 = one-sale-one-vote sensitivity. |
| 3 | materiality threshold | **"No materiality threshold is imposed" is now deleted, not restored.** It would contradict `TAU_MATCH = 0.002`, which governs matched-β attainment and every G5b decision. Only the operating-point conclusion is restored; τ is instead stated *positively*, with its pre-registration, as a design strength. |
| 4 | writing-branch validation | Frozen suites are no longer the gate — verified that `test_b0_isolation.py::test_frozen_stages_and_paper_are_untouched` and P1's `test_no_protected_path_written` fail by design on the first paper edit. New **§H.2 Tier-B validator** at `paper/paper_analysis/tier_b_validation/` (twelve checks, read-only against the frozen maps), built at a new **B1.0** commit; immutability tested by direct `git diff` against the three tags (**§H.3**). No frozen test is modified. |
| 5 | coverage arithmetic | Closed mechanically: **356 − 138 − 184 − 34 = 0**, not 356 − 134 − 184 − 34. The four unaccounted tokens are itemized in **§H.4** — the LOFO guardrail indices 53/56/53/55 at `subsec:path_stability_results` L2913/L2915 — and their block deletion moved into **B1.1**. |
| 6 | Cell C visibility | Both A and C are now **explicit reference rows** in each rebuilt table, with the full 17-metric C values available from `zero_control_full.csv`; a Δ-vs-C column or paired panel is the only permitted alternative. Never prose-only or footnote-only. |
| 7 | table markers | Asterisk dropped. **bold = vs A**, **† = vs C**, with an explicit note that neither denotes significance, compliance, model selection, or a recommended ρ. |
| 8 | bibliography | **No third `\addbibresource`.** Validated entries are copied into `paper/references_additions.bib`; the B0 staged file is provenance/input only, so the manuscript never depends on `analysis/`. `Cheng1974` minimal core, no DOI, unverified fields omitted; `Edelstein1979` `pages = {753}`. |
| 9 | VEI Step 6 | Generalization removed. Bounded wording adopted: none of the 63 escalating standards-facing configurations stopped at Step 6, so the inferential distinction occurred at Step 7 — with any sample-size explanation marked descriptive. |
| 10 | PRB ±0.10 | "Unacceptable band" removed. Reported as "the entire 95% CI lies below −0.10" plus the frozen `outside_pm010` classification, with the four state labels identified as project reporting categories unless the adopted text is verified. |
| 11 | certification scope | Split into `MANUSCRIPT_RELEVANT_CERTIFICATION` (concise, in Appendix G) and `REPOSITORY_AUDIT_DETAIL` (48/49, 118/125, stale scope lines, superseded checkpoint, environment-specific suite variants — kept in the repository evidence package). Three negative constraints survive as claims the paper must avoid making. |
| 12 | extract provenance | Explicitly **non-blocking**: if unavailable by B1.4, proceed with the hash-pinned / non-redistributed / unrecorded-date limitation. |

**No new blocker appeared.** One factual disagreement surfaced (item 1) and is resolved in
favour of the frozen record, with the instruction's intent preserved.

## Self-review of the first draft

Five things I checked and corrected while writing it:

1. **I nearly placed the A/B/C audit table in main-text Results.** The frozen todo-#16
   note says keep the audit in the appendix; the requested empirical backbone puts the
   A/B/C audit third. Resolved by splitting: the *convention* goes in §4.3, a short
   *result* subsection (§5.2) goes in Results, and the *table* stays in Appendix E.3.
2. **I nearly treated the candidate-region ρ endpoints as supported** because the coverage
   audit does not flag them. It does not flag them because they are grid coordinates; the
   claim built on them is what is unsupported. Recorded explicitly in §0, Rule 2.
3. **I nearly kept the positive-ρ dCor column in `tab:path_anchor_complementary` while
   deleting the same values from §5.4 prose.** Both must go together; noted in E.6.
4. **I first wrote the smearing invariance as a two-way split** (invariant vs moves). The
   frozen classification is **three-way**: the three level ratios `scale_by_s` — they are
   neither invariant nor "move by design". Corrected in E.7.
5. **I first described the centered-spread comparator as simply favouring post-hoc.** It
   *crosses over*: C-posthoc beats Direct at moderate correction and loses badly to
   Surrogate at strong correction. Writing it as a one-directional result would have
   inverted the paper's actual conclusion. Corrected in E.7 and in the storyline.

Two residual risks I want on the record:

- The centered-spread §7 comparison is **keyed on held-out β, not development β**, and the
  report says so explicitly ("the formal matched-beta comparison is not performed in Stage
  2"). §5.6 must present it as the descriptive comparison it is, and let §5.7 carry the
  formal matched-β result. Conflating the two would be the easiest serious error in this
  revision.
- Every number in this plan was read from the frozen artifacts or from the Tier-B0 numeric
  map during this session, but the P0/P1 breadth inventories were gathered by parallel
  agents. Before B2.4 and B2.5 I will re-read `matched_beta_comparison.csv`,
  `matched_beta_ext_targets.csv`, `CENTERED_SPREAD_COMPARATOR_REPORT.md` and
  `TEMPORAL_ROBUSTNESS_REPORT.md` directly and quote from them, so that no manuscript
  value ever traces to a digest rather than to an artifact.
