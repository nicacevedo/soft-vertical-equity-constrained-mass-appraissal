# Full Scientific Audit and Major-Revision Plan
## Manuscript: `v17_option1_current.tex` — Covariance-Based Regularization for Regressivity in ML Real Estate Valuation

**Audit basis.** Complete read of the 4,367-line source; page-text of the rendered 62-page PDF; full method sections of Treder 2021, Ren 2026, Wang 2023, Lee & Chen 2026, Smith 2019, Beheshti 2019; Candogan/Han/Lu 2023, Komiyama 2018, McMillen & Singh 2020/2023; IAAO 2013 Standard and May 2026 Exposure Draft 2; all three bibliography files; all three supplied memos. No web search used. No manuscript edits made. No experimental outcomes invented.

**Evidence convention.** `[file p.N]` = verified in the named source at that page/section. "Inferred" = my derivation from verified premises. "Unverified" = flagged, not asserted.

---

# A. Executive scientific verdict

### A.1 What is the strongest defensible contribution now?

Not the Direct penalty, and not "covariance regularization." The defensible core is a **four-part applied-methods and diagnostic package**:

1. **The mass-appraisal translation.** `β_log = Cov(e,y)/Var(y)` on a fixed sample, with `e = log r` and `y = log P`, so squared covariance is a fixed positive rescaling of squared `β_log`, which is the log-scale analogue of the regulated PRB-type slope. Generic "residual–target decorrelation" language does not carry this reading, because it does not identify the penalized quantity with a statistic that assessor standards already regulate.
2. **The covariance-versus-correlation argument.** Because `Corr(e,y) = Cov(e,y)/√(Var(e)Var(y))` and `Var(e)` is model-dependent, a correlation penalty can be reduced by *inflating residual variance* rather than by flattening the slope. Covariance cannot be gamed this way within a fixed sample. Treder and Ren both use correlation; neither makes this argument. This is a genuine, defensible, domain-motivated reason for the chosen objective.
3. **The Surrogate fixed-space theory.** Proposition `prop:surrogate_fixed_space`, Corollary `cor:surrogate_local`, and the spectral representation `eq:surrogate_spectral_path`. I found **no precedent for this in any supplied source.** It is the paper's strongest surviving theoretical result.
4. **The three-level residual-structure path audit** (`β_log` / `Δ_NL` / `dCor`) traced across rolling-origin CV, a held-out block, and a forward year, demonstrating empirically that driving the targeted first-order association toward zero leaves — and can even increase — non-affine and broader dependence. The Surrogate `dCor` rebound (held-out 0.382 → 0.250 → 0.267; 2025 0.258 → 0.266) is the single most valuable empirical object in the paper.

### A.2 What is definitely established prior art?

| Claim | Superseded by | Verified evidence |
|---|---|---|
| Negative residual–outcome covariance is mechanical under least squares | Treder Eq. (13): `−yᵀ(I−H)y ≤ 0`; attributed by Treder to Le et al. | `Treder p.6 §2.4` |
| In-sample OLS identity `C₀ = −‖f₀−y‖²` | Same result, different notation | `Treder p.6 Eq. 13` |
| Zero-residual-correlation solution is a **centered scalar rescaling** of the unconstrained fit | Treder Eq. (17)–(19): `b₁:ₚ = θ₀β₁:ₚ`, `b₀ = ȳ − x̄ᵀb₁:ₚ` | `Treder p.6 §2.5.1` |
| `b_∞ = 1/R²` | Treder `θ₀ = ‖y‖²/(yᵀHy)`, which is `1/R²` for OLS | `Treder p.6 Eq. 19` |
| Tunable soft accuracy-vs-residual-correlation path | Treder §2.5.2 bounded constraint `\|corr\|≤ρ`, with train/test tradeoff curves | `Treder p.7, Fig. 4` |
| Equivalence of in-training constraint and post-hoc affine correction | Treder §2.6: `δ₂ = θ₀ŷ − y`, "equivalent to our zero correlation solution" | `Treder p.7 §2.6` |
| Nonlinear, end-to-end soft residual–target correlation penalty with accuracy tradeoff | Ren `L_total = α·L_MSE + (1−α)·L_ADC`, `δᵢ = ŷᵢ − yᵢ`, ResNet-18 / VQ-VAE | `Ren p.3 §2.4–2.5` |
| Zero-covariance target as a hard in-processing constraint (+ intercept calibration) | Lee & Chen OCR: `η̂ = Cov(y,ŷ)/Var(y) = 1 ⟺ Cov(y, ŷ−y) = 0`, imposed as `Aβ = c` | `Lee & Chen p.4–5 §2` |
| Observation-additive, target-dependent loss weighting for this exact bias | Wang `L^□ = L·s(y−ŷ, y)`, `s = exp(sgn(x)λ(y))`, λ(y) linear in target | `Wang p.3 §III.A` |
| Post-hoc affine residual–target correction, training-fitted, applied out of sample | Beheshti offset `Ω = α + β`; Smith stage-2 correction | `Beheshti p.2 §2.4`; `Smith p.2 §2.2` |

**Consequence.** Appendix D's Direct subsection (Prop. `prop:path`, Cor. `cor:path_scaling`, Remark `rem:mechanical_covariance`, Prop. `prop:path_cost`) is **largely a reparameterization of Treder (2021)** and must be reframed accordingly. It should be *kept* — it is correct, it is the exact path for *this* objective, and it is the mathematical reason the post-hoc comparator is mandatory — but not presented as new geometry.

### A.3 Does the paper remain valuable under CCAO-centered applied-methods framing?

**Yes, conditional on P0-A.** The paper's own Corollary `cor:path_scaling` proves that in a fixed prediction space containing an intercept the Direct penalty is *exactly* a one-parameter centered rescaling — i.e., zero additional expressive power over post-hoc recalibration in the benchmark case. Having proved that, declining to run the post-hoc comparator (currently deferred to future work at `subsec:comparators` line 2045 and a rendered `\todo` at line 3382) is not a scope choice a referee will accept. With P0-A completed the paper has a real question with four scientifically interesting possible answers, none of which requires Direct to win.

### A.4 Five most serious rejection risks

1. **The paper proves its method reduces to post-hoc rescaling, then declines to test post-hoc.** Highest-severity. Compounded by a rendered `\todo` explicitly relegating the comparison to a future paper.
2. **The native/custom ρ=0 parity failure contaminates the headline table.** Line 2092 admits non-parity (held-out mean |Δŷ| = 3.24e−02, max 3.09e−01; β_log −0.150 vs −0.147). Yet `tab:path_anchor_summary`'s entire boldface/asterisk convention is defined **relative to Ordinary LightGBM**. The paper's most attractive result — Direct at ρ≈1 with held-out R² 0.899 vs 0.894 and MAE \$74,485 vs \$75,655 — is therefore stated against a contaminated reference. Worse, `tab:rho_zero_control` reports **only R², RMSE_log, and β_log** for the ρ=0 controls, so MAE, MAPE, PRD, PRB, MKI, VEI and COD at the custom-objective origin are **not reported anywhere in the manuscript**, and the reader cannot repair the comparison themselves.
3. **Treder overlap.** A referee who knows the brain-age literature will read Appendix D as re-derivation.
4. **Inference is absent exactly where the cited standards require it.** IAAO 2013 ties PRB interpretation to t-values and 95% CIs and reserves "unacceptable" for CIs outside ±0.10 `[2013 p.19]`; ED2's VEI test has a **mandatory second stage** (CI overlap, then VEI Significance vs 10%) `[ED2 App. E Steps 6–7]`. The paper reports point estimates across 82 grid points and draws directional conclusions, with CIs only on VEI group profiles.
5. **Nested folds treated as replicates.** The seven expanding windows are strictly nested (fold 7 trains on 310,147 of the 344,607 pool; folds 1–6 validation sets are inside fold 7's training set). "Equal-weight fold means and standard deviations" and "7/7 LOFO stability" are much weaker than presented.

### A.5 What is strongest and must be protected

- Proposition `prop:bayes_covariance` — verified: `Cov(e*,Y) = Var(E[Y|X]) − Var(Y) = −E[Var(Y|X)] ≤ 0`. Correct, well-proved, and the intellectual anchor of the paper's honesty about what zero covariance does and does not mean.
- Remark `rem:mechanical_covariance` and `C₀ = −‖f₀−y‖ₙ²` — verified correct; keep, re-attribute.
- The covariance-vs-correlation argument (§3.1 line 1410) — keep and **expand**; it is under-sold.
- The entire Surrogate fixed-space subsection — verified correct; this is the novel theory.
- `Δ_NL` construction, projection representation `eq:nonlinearity_gap_projection`, and the cross-fitted estimator with non-negative truncation `eq:nonlinearity_gap_estimator`.
- The Surrogate `dCor` rebound and non-monotone ratio trough.
- The scale-normalization coordinate `ρ̃ = ρ·V̂ar_train(y)` (`eq:normalized_rho`) — verified: squared error scales as `a²`, `C(f)²` as `a⁴` under `y ↦ ay`. Correct and genuinely useful.
- The proxy-aware, standards-grounded metric suite and its explicit refusal to infer compliance.

---

# B. P0 revision ledger

| # | Location | Current claim / problem | Why scientifically problematic | Source evidence | Required action | Final defensible message | Exp.-dep.? | Conf. |
|---|---|---|---|---|---|---|---|---|
| **P0-1** | `tab:path_anchor_summary` (L2671–2745) caption + Notes; all Results prose comparing to "Ordinary LightGBM" | Boldface/asterisk convention defined as improvement over Ordinary LightGBM | Line 2092 already concedes native ≠ custom at ρ=0. Every native-to-penalized contrast mixes penalty effect with API-path effect. | Manuscript `tab:rho_zero_control` L3667–3673 | Regenerate from parity run (P0-B). Until then, redefine bold/star **relative to the within-family ρ=0 control** and add the full metric suite for both ρ=0 controls. | Positive-ρ effects are measured against each family's own ρ=0 origin. | Yes | 97 |
| **P0-2** | `tab:rho_zero_control` (L3659–3676) | Reports only R², RMSE_log, β_log, mean/max Δŷ | The metrics used for every headline claim (MAE, MAPE, PRD, PRB, MKI, VEI, COD, Δ_NL, dCor) are absent at the ρ=0 origin, so no reader can reconstruct a clean contrast. | Direct inspection | Extend the table to the complete Section-2.3 metric suite for Ordinary, Direct ρ=0, Surrogate ρ=0, on both held-out and 2025. | The implementation control is auditable on every reported metric. | Yes | 96 |
| **P0-3** | `subsec:comparators` L2045; `\todo` L2046; `\todo` L3382; `Future Work` L3071 | Post-hoc calibration declared out of scope and deferred to a follow-up paper | Cor. `cor:path_scaling` proves Direct *is* a centered-spread map in the fixed-space benchmark. Deferring the matched comparator is not defensible after proving that. | Manuscript Cor. `cor:path_scaling`; Treder §2.6 | Run P0-A (Section G.1). Move the comparator into the paper. Delete both `\todo`s. | The centered-spread post-hoc map is a comparator in this paper, not future work. | Yes | 95 |
| **P0-4** | App. D `Direct penalty: rank-one covariance path` (L3278–3396); contribution ¶ L204; Discussion L3002 | Fixed-space Direct geometry presented as this paper's result | Anticipated by Treder: negative ADC identity, θ₀ scaling, centered rescaling, soft bounded path, post-hoc equivalence. | `Treder p.6 Eq. 13, Eq. 17–19; p.7 §2.5.2, §2.6` | Keep theorem; add explicit attribution; reframe as *the exact path of this specific soft squared-covariance objective* and as the motivation for the comparator. | The one-dimensional geometry is known; what is exact here is the squared-covariance Lagrangian path `C_ρ = C₀/(1+ρA/2)`, which never reaches zero at finite ρ. | No | 93 |
| **P0-5** | Related Work L277–287 | Closest lineage is generic fair regression / sensitive-attribute dependence | The genuinely closest work is residual–outcome correction (Treder, Ren, Wang, Lee & Chen, Smith, Beheshti). None is cited. | All six sources | Insert a dedicated *residual–outcome dependence and correction* paragraph **before** the fair-regression paragraph. Demote generic fair regression to broader context. | The method sits in the residual–outcome correction lineage; fair regression supplies only the general dependence-control principle. | No | 95 |
| **P0-6** | Abstract L123 | "Understanding regressivity as an undesired dependence between log-price residuals and the actual sale prices" | Equates regressivity with a dependence the paper only partially targets, and asserts a definition the paper elsewhere refutes (`Δ_NL`, dCor, Prop. `prop:bayes_covariance`). | Manuscript's own §3.1 L1403–1408 | Rewrite: regressivity is a value-related pattern in valuation ratios across the value distribution; the paper targets one signed **first-order component** of it. | The target is a component of regressivity, not its definition. | No | 94 |
| **P0-7** | §3.2 L1509; App. E L3630–3641, L3648–3650; Abstract L123; `subsec:comparators` L2041 | Direct described as using "the diagonal of its dense rank-one Hessian" | **New finding.** Supplied `hᵢ = 1 + (ρ/2n)cᵢ²`. With n ≈ 3.1×10⁵ and cᵢ² ~ O(0.5) this is ~10⁻⁷ρ — indistinguishable from 1 even at ρ=100. The retained diagonal is numerically inert; the *discarded* off-diagonal carries curvature `1 + (ρ/2)V̂ar(y)` ≈ 1+0.25ρ along the c direction alone. | Derived from manuscript Eqs. `eq:direct_gradient_hessian`, App. E L3638; `V̂ar(y)` inferred from `β_log = C/V_y` scale | State that Direct is effectively a **gradient-only modification retaining native curvature**; note that Newton steps along c are systematically over-stepped by up to ~(1+ρV̂ar(y)/2), which is an implementation-side candidate explanation for high-ρ Direct behavior currently attributed to the objective. | The Direct implementation supplies the exact covariance gradient with essentially native curvature; the omitted cross-observation term is what matters, and its omission biases step size along the penalized direction. | Partly (a curvature-corrected run would confirm) | 88 |
| **P0-8** | `\usepackage[textsize=tiny]{todonotes}` L20; source comment L18–19 | Comment asserts TODOs "are hidden by default"; `disable` is **not** in the options | **Verified in rendered PDF:** `\todo{also redundant with previous story}` and `\todo{This is also unclear or redundant}` appear on the Introduction contribution page; `\todo{Treat Eq. (49)...}` appears in the appendix. | `docs/v17_option1_current.txt` L113, L119, L2499 | Delete all `\todo` content before any external circulation; do not rely on `disable`. | — | No | 99 |
| **P0-9** | Intro L183; §4 opening L1858 | Both promise six-county / ATTOM experiments | The ATTOM material is inside `\iffalse` (L2059–2076, L2903–2971, L4298–4348) and never reported. Rendered PDF L107 and L1095 promise it; rendered L4046 says the paper does not report it. Direct self-contradiction. | Rendered PDF L107, L1095, L4046 | Remove both promises; keep only the External-Validity Boundary statement. | The paper makes no multi-jurisdiction claim. | No | 98 |
| **P0-10** | `tab:assessment_metrics_summary` MKI row L1040–1043 + Notes L1066–1068 | MKI reference band `[0.95, 1.05]` | ED2 states `[0.95,1.05]` is for **lower-variability strata** and `[0.90,1.10]` for **higher-variability or small strata**. This sample has COD ≈ 21.6%, COV ≈ 39.7% — unambiguously high-variability. The tighter band flatters the corrections and worsens the baseline. | `ED2 §Gini/Kakwani, L3168–3169` | Use `[0.90,1.10]`, or report both and justify. Recheck every MKI-based directional claim. | MKI is read against the high-variability band appropriate to this stratum. | No | 90 |
| **P0-11** | Eq. `eq:market_value_proxy` L555 and note L565 | Equal-weight proxy attribution | **Manuscript is correct.** 2013 App. D p.56: `Value = 0.50 × (AV/Median) + 0.50 × SP`. ED2 App. E Step 2 displays `Proxy = (0.50*SP) + (AV/Median Ratio)` — omitting the 0.50 — while its heading says "gives equal weight" and its Note says "the **average** of" the two. | `2013 p.56 App. D`; `ED2 App. E Step 2` | Keep wording; sharpen to note the draft specifies equal weight in **two** places, so the omission reads as typographical, and that the literal formula would roughly double the AV contribution. | Consistent with the 2013 formula and ED2's equal-weight prose; not a transcription of ED2's printed equation. | No | 96 |
| **P0-12** | Active `.bib` files | Treder, Wang, Ren, Lee & Chen, Smith 2019, Beheshti 2019 absent | Cannot execute P0-5 without them. | Key extraction: none present | Add six verified entries to an **actually loaded** `.bib` (`references.bib` or `references_additions.bib`). Do not rely on the historical supplement, which the `.tex` does not load. | — | No | 99 |
| **P0-13** | `\oldtext` / `oldrevisionblock` regions, esp. L2876–2888, L2987–2989, L3623 | Superseded prose renders struck-through in the compiled PDF, including three successive versions of the same Discussion paragraph | The rendered document contains mutually contradictory statements about Surrogate CV span support (L2883 says Surrogate fails the five-metric construction; L2884 says it now succeeds). | Direct inspection; rendered PDF | Resolve to one version each; delete superseded text from source before circulation. | — | No | 97 |

---

# C. P1 revision ledger

*(To be executed after the P0 experiments resolve.)*

| # | Location | Problem | Why it matters | Evidence | Required action | Exp.-dep.? | Conf. |
|---|---|---|---|---|---|---|---|
| P1-1 | §4.2 L1947–1955; App. F `tab:fold_structure` L4202–4219 | Seven **expanding-window** folds are strictly nested but summarized by equal-weight means/SDs and 7/7 LOFO counts | Nested folds are not exchangeable replicates; SDs understate uncertainty and LOFO agreement is near-mechanical | Verified: fold 7 train = 310,147 ⊂ 344,607 pool | State the nesting explicitly; report fold-level paths rather than SDs as if independent; downgrade LOFO language from "stability" to "not driven by a single fold's validation block" | No | 92 |
| P1-2 | `tab:assessment_metrics_summary` PRB row; all PRB reporting | Band `[-0.05,0.05]` given without the standard's inferential structure | 2013 requires t-value/CI reading and reserves "unacceptable" for 95% CIs outside ±0.10 | `2013 p.19` | Add the significance requirement and the ±0.10 threshold to the table note; report PRB standard errors for baselines and display anchors | Yes (cheap) | 94 |
| P1-3 | §2.3 VEI L626–639; all VEI reporting | ED2's VEI test is two-stage; paper computes only the point estimate | A point estimate inside/outside ±10% is **not** the draft's test outcome | `ED2 App. E Steps 5–7` | Compute VEI Significance for baselines and the five display anchors; state plainly that the paper reports the draft's Step-5 statistic only | Yes (cheap) | 93 |
| P1-4 | §2.3 COD L504–534; `tab:ccao_baseline_complementary` | Observed COD ≈ 21.6% (held-out) / 21.3% (2025) is outside the adopted `[5.0,15.0]` band for **every** model including all penalized configurations; never discussed | Undercuts operational-feasibility language; a referee will note no configuration meets the adopted uniformity standard | `2013 Table 1-3 p.17`; manuscript `tab:ccao_baseline_complementary` | Add one paragraph acknowledging that horizontal uniformity is outside adopted guidance throughout, and that the intervention targets vertical equity only | No | 93 |
| P1-5 | §2.3 PRD L540–549 | PRD band applied without the standard's own caveat | 2013: PRD "may be less meaningful ... when wide variation in prices exist. In such cases, statistical tests ... should be substituted" — exactly this sample | `2013 Table 1-3 note, p.17` | Add the caveat where PRD is introduced | No | 92 |
| P1-6 | Limitations L3053; entire empirical design | Sold-vs-unsold treated as a narrative limitation only | Both standards make sold/unsold equivalence a **validity requirement** with a named test (Mann-Whitney) and a sales-chasing detection appendix. The correction is fitted on sales; its effect on the unsold roll is entirely unvalidated | `2013 L1090, L538`; `ED2 L703, §5.3.1` | Either run the characteristic-comparison test on sold vs unsold, or state explicitly that the ratio-study validity condition is untested and that all claims are conditional on it | Yes | 91 |
| P1-7 | Surrogate §3.3 and all Surrogate claims | Surrogate weights are symmetric and sign-independent, so nothing guarantees signed covariance reduction; the fixed-space theory confirms this (L3576) | A referee will ask why a method with no directional guarantee is presented as "covariance-guided," and why Wang's directional observation-additive loss is not the natural comparator | Manuscript L1621, L3576; `Wang p.3 §III.A` | State the absence of a shrinkage guarantee up front; cite Wang as the directional alternative and note it is not evaluated here | No | 90 |
| P1-8 | §2.4 baseline motivation L1325–1329 | "LightGBM is more regressive than Linear" carried by PRB/VEI/MKI | McMillen & Singh show regression-based vertical-equity measures are biased toward finding regressivity when the sales used for evaluation were also used to fit the assessments, and are biased under model-form mismatch (log-linear fit evaluated with a linear diagnostic) | `McMillen & Singh 2023 abstract; §3 pp.1–2` | Note that the held-out/forward design mitigates the first bias; acknowledge the log-vs-linear mismatch; cite explicitly | No | 89 |
| P1-9 | App. B L3204 `\todo` | Distance-correlation estimator variant (biased/V-statistic vs U-centered) unrecorded | dCor values are reported to three decimals and a rebound of 0.250→0.267 carries an argument | Manuscript L3204 | Record the exact estimator and confirm consistency across all reported paths | Yes | 95 |
| P1-10 | §5 Discussion L2978–3002; Conclusion | Discussion still frames "moderate regularization need not impose an accuracy cost" as a finding | Depends entirely on the ρ=0 origin question (P0-1/P0-2) | — | Rewrite after parity; state improvements relative to the family origin, with the parity caveat resolved or retained | Yes | 94 |
| P1-11 | Bibliography | `SmithEtAl2026` (FAccT) and `SmithHarveyBerryGoldinHo2026` (arXiv) duplicate the same paper; only the former is cited | Duplicate entries risk double-listing | Key extraction | Merge; keep the published FAccT entry | No | 96 |
| P1-12 | Bibliography | `IAAO2026ExposureRatio` URL is `..._Exposure-Mar2026.pdf` while title/month/note say **May 2026** | The paper's entire standards argument depends on identifying ED2 precisely | Entry inspection | Verify and correct the URL/date; confirm this is ED2 | No | 92 |
| P1-13 | Bibliography | `IAAO2025ExposureRatio` (Sept 2025 draft) present but uncited; `10578d49-78f8-...` UUID key (Sirmans et al. 2008) present but uncited; `McMillen2020` key names a two-author McMillen & Singh paper | Stale/malformed entries invite miscitation | Key extraction | Remove or clearly quarantine the 2025 draft; rename the UUID key; rename `McMillen2020` → `McMillenSingh2020` | No | 94 |
| P1-14 | Title | "Covariance-Based Regularization for Regressivity..." | May become too narrow if the post-hoc comparator becomes co-equal | — | Reassess **only after** P0-A resolves | Yes | 85 |

---

# D. Novelty and prior-art matrix

`e = ŷ − y` throughout. "Target" = residual-vs-outcome association unless noted.

| Paper | Same statistical target? | Same objective/form? | Same model class? | In- vs post-processing? | Same application? | What it anticipates | What remains distinct here | Required citation/positioning change |
|---|---|---|---|---|---|---|---|---|
| **Smith et al. 2019** | Yes — `Cov(δ, y)` | No — two-stage OLS of δ on y | Linear (stage 2) | **Post** | No (brain age) | Residual–target orthogonalization as a corrective target; nonlinear-correction discussion | Nothing in the mechanism; only the application and the soft in-training path | Cite as post-hoc origin; add to new lineage ¶ |
| **Beheshti et al. 2019** | Yes | No — training-fitted affine offset `Ω = α + β` applied out of sample | Linear correction on any base model | **Post** | No | Correct predictive practice: fit correction on train, apply to test | Nothing in mechanism | Cite as the methodologically sound post-hoc variant |
| **Treder et al. 2021** | **Yes — identical** (`corr(y,δ)`) | **Very close**: hard `=0` and soft `\|corr\|≤ρ` constrained ERM | OLS, **Ridge, Kernel Ridge** (nonlinear via RKHS) | **In** | No | (i) mechanical negative ADC `−yᵀ(I−H)y≤0`; (ii) constrained solution = **centered scalar rescaling** `θ₀=‖y‖²/yᵀHy` = `1/R²` for OLS; (iii) tunable soft accuracy/ADC path; (iv) **explicit equivalence to post-hoc scaling** | Covariance rather than correlation, with a stated reason; the specific Lagrangian path `C_ρ=C₀/(1+ρA/2)` that never reaches zero at finite ρ; boosted trees; mass appraisal; the Surrogate | **Closest methodological prior work.** Must be cited at Abstract, Related Work, §3.1, §3.2, App. D, Discussion |
| **Wang et al. 2023** | Yes | Related but **directional**: `L·exp(sgn(y−ŷ)λ(y))`, λ linear in target | Deep CNNs (ResNet/VGG/GoogleNet) | **In** | No | Observation-additive, **target-dependent** loss weighting for exactly this bias; comparison against two-stage affine correction; transfer/shift concerns | Symmetric `1+ρ(y−ȳ)²`; **sign-independent**; derived as a Jensen bound on squared covariance; exact weighted-MSE identity; weighted-projection geometry | **Closest Surrogate prior work.** Cite; state explicitly that target-dependent weighting is not claimed as new |
| **Ren et al. 2026** | **Yes — identical**, same sign convention | Soft `α·L_MSE + (1−α)·\|corr(δ,y)\|` | ResNet-18, VQ-VAE (deep, nonlinear) | **In** | No | Nonlinear soft residual–target correlation penalty, end-to-end, with an explicit accuracy/bias tradeoff | Covariance vs correlation; slope interpretation; GBDT custom-objective implementation; ratio-study evaluation | Cite wherever "nonlinear in-processing correction" is characterized |
| **Lee & Chen 2026 (OCR)** | **Yes — exact endpoint**: `η̂=1 ⟺ Cov(y,ŷ−y)=0` | **Hard equality constraint** `Aβ=c`, **plus** intercept calibration `α̂=0` | **Linear only**, closed-form restricted LS | **In** | No | The zero-covariance target as a formal calibration condition; minimum-variance property among constrained unbiased estimators | Soft path rather than hard constraint; no level constraint imposed; nonlinear learner; mass appraisal | Cite as near-exact **endpoint** precedent. **Do not import "conditional unbiasedness"** — their Prop. 2 assumes `W ⊥ (Ŷ−Y) \| Y` |
| **Candogan, Han & Lu 2023** | No — segment-level ratio fairness, not residual–target covariance | No — K-segment architecture with smoothing; deviation-weighted fairness is an **evaluation** measure, not a training weight | LightGBM, CCAO | **In** (architectural) | **Yes — same office, same base learner, similar 90/10 chronological design** | The CCAO regressivity problem; the accuracy–fairness tradeoff framing; tail-weighted deviation measures; the empirical design template | Objective modification rather than segmentation; single global first-order target; rolling-origin CV + forward year; mechanism diagnostics | **Closest mass-appraisal intervention.** Already cited; strengthen the contrast and note the shared empirical-design lineage (shared coauthor: Lu) |
| **Komiyama et al. 2018** | No — CoD between **prediction and sensitive attributes** | Hard CoD constraint, nonconvex, exact global solution | Linear | **In** | No | Dependence-as-constraint with exact optimization and a fairness/accuracy tradeoff | Target is the outcome itself, not a protected attribute; soft penalty; nonlinear learner | Keep as general dependence-control context; explicitly **not** the closest ancestor |
| **McMillen & Singh 2020** | n/a | n/a | n/a | n/a | Yes | Regressivity's incidence consequences | — | Correct role; rename key |
| **McMillen & Singh 2023** | n/a — measurement | n/a | n/a | n/a | Yes | Regression-based vertical-equity measures are biased toward regressivity when evaluation sales were used to fit assessments; biased under model-form mismatch; PRD preferred; Gini/Suits alternatives | — | **Under-used.** Should be cited in §2.4 and Limitations, not only in Related Work |

### Closest methodological prior work
**Treder et al. (2021).** It shares the statistical target, the in-processing mechanism, the soft tunable path, the model-class generality (including a nonlinear kernel class), the scaling geometry, and the post-hoc equivalence.

### Closest mass-appraisal intervention
**Candogan, Han & Lu (2023).** Same office, same base learner, same problem, comparable empirical design — but a different mechanism (segmentation vs objective modification).

### Closest post-hoc prior work
**Beheshti et al. (2019)** — training-fitted affine offset applied out of sample; methodologically the correct template for the P0-A comparator. **Smith et al. (2019)** is the conceptual origin; **Treder §2.6** supplies the formal equivalence.

### Narrowest defensible method claim

> We translate a ratio-study vertical-equity concern into a soft squared-covariance objective on the log-error/log-price association, whose penalized quantity is a fixed rescaling of the log-ratio slope; we derive its exact Lagrangian path in a fixed prediction space and its Jensen-derived observation-additive surrogate, characterize the surrogate as a log-price-distance-weighted projection with generally multi-directional geometry, implement both through a standard gradient-boosting custom-objective interface, and audit their complete chronological regularization paths in a research translation of the CCAO residential workflow using first-order, non-affine conditional-mean, and omnibus dependence diagnostics.

Every clause is defensible after the P0 changes. Nothing in it claims that residual–target decorrelation, dependence regularization, one-dimensional scaling, target-dependent weighting, or accuracy/dependence tradeoffs are new.

---

# E. Mathematical audit

All derivations below were checked by hand.

### CORRECT — PRESERVE

| Item | Location | Verification |
|---|---|---|
| Covariance–slope identity | `eq:covariance_slope` | `C(f)=n⁻¹Σeᵢcᵢ`, `V_y=n⁻¹Σcᵢ²`; slope `= C/V_y`. Requires `V_y>0` and the same `1/n` convention in both; both hold. Centering of `e` is immaterial since `Σcᵢ=0` |
| Residual sign consistency | throughout | `e=ŷ−y` used consistently in `eq:prediction_scale`, `eq:training_covariance`, gradients, `β_log`, App. D. `C<0` ⟺ ratios decline in price ⟺ regressive. **No sign error found anywhere.** Matches Ren's δ convention and is the negative of Treder's `e` |
| Bayes/population identity | Prop. `prop:bayes_covariance` | `Cov(f*(X)−Y, Y) = Var(E[Y\|X]) − Var(Y) = −E[Var(Y\|X)]`. Needs only `E[Y²]<∞`. Correct |
| Finite-sample OLS analogue | Cor. `cor:path_scaling` | `C₀ = ⟨f₀−y,c⟩ₙ = −‖f₀−y‖ₙ²` via `1∈V` and residual ⟂ V. Correct — **and identical to Treder Eq. (13)** |
| Direct gradient | `eq:direct_penalty_gradient` | `∂/∂ŷᵢ[(ρ/2)C²] = ρC·cᵢ/n`. Correct |
| Direct exact Hessian | `eq:direct_gradient_hessian` | `(2/n)I + (ρ/n²)ccᵀ`. Correct |
| Jensen bound | `eq:surrogate_upper_bound` | `(n⁻¹Σuᵢ)² ≤ n⁻¹Σuᵢ²` with `uᵢ=eᵢcᵢ`, convexity of `t²`. Correct |
| Surrogate decomposition | `eq:surrogate_decomposition` | `Ψ = C² + n⁻¹Σ(uᵢ−C)²` is the variance identity. Correct |
| Weighted-MSE identity | `eq:surrogate_weighted_loss` | `L+ρΨ = n⁻¹Σ[1+ρcᵢ²]eᵢ²`. Exact, not approximate |
| Surrogate derivatives | `eq:surrogate_derivatives` | `gᵢ=(2/n)eᵢ[1+ρcᵢ²]`, `hᵢ=(2/n)[1+ρcᵢ²]`. Correct; `cᵢ` fixed given `ȳ` |
| `n/2` implementation scaling | App. E L3626–3646 | Direct: `gᵢ=eᵢ+(ρ/2)zcᵢ`, `hᵢ=1+(ρ/2n)cᵢ²`. Surrogate: `gᵢ=eᵢ(1+ρcᵢ²)`, `hᵢ=1+ρcᵢ²`. All consistent with the stated objectives and with App. E L3691–3739 |
| Fixed-space Direct path | Prop. `prop:path` | FOC `2⟨f_ρ−y,h⟩+ρC_ρ⟨c,h⟩=0` ⟹ `d_ρ=−(ρ/2)C_ρ g`; `⟨g,c⟩ₙ=‖P_V c‖ₙ²=A` ⟹ `C_ρ=C₀/(1+ρA/2)`. Requires `A>0`. Correct |
| Centered-spread form | Cor. `cor:path_scaling` | `1∈V ⟹ g=f₀−ȳ1`; `f_ρ=ȳ1+b_ρ(f₀−ȳ1)`, `b_ρ=1−(ρ/2)C_ρ`. `C₀<0 ⟹ b_ρ>1`, `C_ρ→0⁻`. Correct |
| Accuracy cost | Prop. `prop:path_cost` | `‖f_ρ−y‖²−‖f₀−y‖² = (C₀²/A)(1−q(ρ))²`; second-order near ρ=0. Correct |
| Surrogate weighted projection | Prop. `prop:surrogate_fixed_space` | `f_ρ^surr = P_V^{W_ρ}y`; `δ_ρ=−ρ(I+ρT)⁻¹g_surr`, `T=P_V D\|_V` self-adjoint PSD on V, `g_surr=P_V De₀`. Correct. **No precedent found in supplied literature** |
| Surrogate local direction and cost | Cor. `cor:surrogate_local` | `f_ρ=f₀−ρP_V[c^⊙2⊙e₀]+O(ρ²)`; cost `=‖δ_ρ‖ₙ²=ρ²‖g_surr‖ₙ²+O(ρ³)`. Correct |
| Spectral path | `eq:surrogate_spectral_path` | `δ_ρ=−Σⱼ ρ/(1+ρλⱼ)·aⱼuⱼ`. Correct; multi-directional unless `g_surr` lies in one eigenspace |
| PRD sign result | Prop. `prop:cov_prd` | `1−PRD = Cov(r,P)/E[rP]`. Correct; the manuscript correctly notes it acts on `Cov(r,P)`, not the penalized `Cov(log r, log P)` |
| `Δ_NL` projection form | `eq:nonlinearity_gap_projection` | `η²(e\|Z) − Corr(e,Z)² = E[(m−Π_aff m)²]/Var(e) ≥ 0`, using Z standardized so the affine component is exactly `Corr(e,Z)²`. Correct |
| Scale normalization | `eq:normalized_rho` | Under `y↦ay`: squared error `∝a²`, `C(f)²∝a⁴`, so `ρ̃=ρ·V̂ar(y)` is scale-invariant. Correct |

### CORRECT BUT NEEDS QUALIFICATION

- **Covariance vs correlation (§3.1 L1410).** The equivalence `Corr=0 ⟺ Cov=0` holds only for positive variances. Away from zero the objectives differ because `Var(e)` is model-dependent — so a correlation penalty admits a degenerate route (inflate `Var(e)`) that a covariance penalty does not. The manuscript states the premise in one sentence but never draws this conclusion. **Expand this; it is a load-bearing justification.**
- **`b_∞ = 1/R²` (not currently in the manuscript).** I verified it: `b_∞ = 1 − C₀/A = 1 + ‖e₀‖²/A`, and with `1∈V`, `‖c‖² = A + ‖e₀‖²`, so `b_∞ = 1/R²` for in-sample OLS on the log scale. Also verified that `b = 1/R²` is exactly the value making `Cov(e_b, y) = 0`. **Include it** — not for elegance, but because it pins the exact endpoint of the P0-A comparator grid. Attribute to Treder's `θ₀`.
- **Surrogate as "covariance-guided."** The upper-bound relation does **not** imply covariance shrinkage; the manuscript's own L3576 says monotone shrinkage is not guaranteed. Keep, but surface earlier.
- **`Δ_NL`.** Measures **conditional-mean** non-affinity only. Not a monotone-dependence measure, not a general dependence measure. Manuscript is mostly careful; enforce uniformly.
- **`dCor`.** Population zero ⟺ independence under moment conditions ✓. Unsigned, does not identify mechanism, does not rank severity across dependence forms. Manuscript's caveats are adequate.

### INCORRECT / MISLEADING AS WRITTEN

- **The "diagonal Hessian approximation" characterization** (P0-7). `hᵢ = 1 + (ρ/2n)cᵢ²` is numerically ≈ 1 at every tested ρ. Describing Direct as retaining "the diagonal" of the curvature implies a meaningful second-order approximation; in fact it is native curvature. The scientifically accurate statement is that the omitted rank-one term carries curvature `1+(ρ/2)V̂ar(y)` in the single direction `c`, so Newton steps along the penalized direction are over-taken by that factor. This is the one place where the manuscript's mathematical description does not match what the implementation does.

### UNVERIFIED

- Reported empirical values (paths, CV events, LOFO counts, regret table) — not independently recomputable from supplied materials.
- The `\todo`-flagged `motivation_utils.py` SHA256 (L566) and the executed code's normalization conventions.
- The exact dCor estimator variant (L3204).
- Whether the "later initialization-aligned parity experiment" referenced at L3679 in fact achieves parity.

---

# F. IAAO standards audit

| Topic | 2013 approved guidance | May 2026 Exposure Draft 2 | Current manuscript treatment | Required correction |
|---|---|---|---|---|
| **Market value** | Latent; a single sale is an indicator, not market value | Same, stated as a Principle: "Market value cannot be observed directly; a single sales transaction does not equate to market value and ratio study statistics do not apply to individual properties" `[ED2 §3, L232]` | Correctly stated at L335, L148, L538 | None. **Protect this language** |
| **Sales proxy** | Valid arm's-length, verified, screened sales | Same; adds time-adjusted sale price | `P_i` treated as verified transaction-based proxy | Consider noting whether time-adjusted sale price is used; ED2 permits TASP |
| **Ratio studies** | Level + uniformity; multiple complementary diagnostics | Same, with VEI elevated | Correct | None |
| **COD** | SF residential older/heterogeneous **5.0–15.0**; <5.0 may indicate sales chasing `[2013 Table 1-3, p.17]` | Adds: "Extremely low variability should be examined to detect possible sales chasing or model overfitting" `[ED2 L834]` | Band `[5.0,15.0]` in Table 1; low-COD caution stated correctly | **P1-4:** acknowledge that observed COD ≈ 21.3–21.6% is outside the adopted band for every model reported |
| **PRD** | **0.98–1.03**; "not absolute ... less meaningful when samples are small or when **wide variation in prices exist**. In such cases, statistical tests ... should be substituted" `[2013 Table 1-3 note]` | Retained as supplemental; noted as sensitive to value outliers and possibly "too insensitive" in very large samples `[ED2 L3263–3264]` | Band given; sensitivity to high-value observations noted | **P1-5:** add the wide-price-variation caveat, which applies directly here |
| **PRB** | "should fall between –0.05 and 0.05"; **95% CIs outside that range** support a >5% conclusion; **CIs outside ±0.10 indicate unacceptable vertical inequities** `[2013 p.19]` | Retained as supplemental; ±.05 "tend to indicate good" `[ED2 L936]` | Band `[-0.05,0.05]` only; no t-values, no CIs | **P1-2:** add the inferential structure and ±0.10 threshold; report standard errors |
| **MKI** | Not in 2013 | MKI=1 neutral; **0.95–1.05 for lower-variability strata; 0.90–1.10 for higher-variability or small strata** `[ED2 L3165–3169]` | Uses `[0.95,1.05]`, attributed to CCAO practice + "lower-variability strata" | **P0-10:** this is a high-variability stratum; use `[0.90,1.10]` or report both |
| **VEI** | Not in 2013 | **Primary** measure; supplemental measures "can potentially confirm the finding of the VEI" `[ED2 L910–913]`. Deciles for n≥501. `VEI = 100·(Median Last PG − Median First PG)/Sample Median`. Band ±10%. **Two-stage test**: CI overlap → VEI Significance vs 10% `[ED2 App. E]` | Formula, deciles, band, and primary status all stated **correctly**; 90% group CIs reported | **P1-3:** compute VEI Significance for baselines and anchors, or state explicitly that only Step 5 is reported |
| **Reference ranges** | Adopted | Proposed | Table 1 note distinguishes adopted / derived / draft; `tab:path_anchor_summary` note repeats it | Correct. Preserve; extend the same discipline to figure shading captions |
| **Compliance wording** | Level/uniformity conclusions require CIs and tests | Same, strengthened | L1196 and L3084–3094 disclaim compliance inference | Correct and unusually careful. **Protect** |
| **Equal-weight proxy** | `Value = 0.50×(AV/Median) + 0.50×SP` `[2013 App. D p.56]` | Prose says equal weight and "the **average** of"; **displayed formula omits the 0.50 on AV/Median** `[ED2 App. E Step 2]` | Manuscript implements the 2013 formula and describes the draft inconsistency accurately at L565 | **P0-11:** keep; sharpen to note the prose specifies equal weight twice |
| **Confidence intervals** | Required for level and PRB conclusions | Required; 90% CIs mandated per VEI percentile group | Only VEI group profiles carry CIs | **P1-2/P1-3** |
| **Sample representativeness** | Central validity condition | Extensive §5.3 treatment | Acknowledged in Limitations | Move a one-sentence version into the empirical design |
| **Sold vs unsold** | "Ratio study validity **requires** that sold and unsold parcels be appraised at the same level"; Mann-Whitney listed `[2013 L538, L1090]` | Identical requirement; §5.3.1 + sales-chasing Appendix B `[ED2 L317, L703]` | Narrative limitation only | **P1-6:** either test it or state that the validity condition is untested and all claims are conditional |
| **Temporal evaluation** | Sample-period selection to preclude sales chasing | Same | Strong chronological design | None — this is a strength |

---

# G. Empirical-blocker specification

## G.1 — P0-A: Centered-spread post-hoc comparator

**Verdict: REQUIRED.** Not because the audit prompt lists it, but because Corollary `cor:path_scaling` proves the Direct penalty *is* this map in the fixed-space benchmark, and Treder §2.6 independently establishes the constraint/post-hoc equivalence. The paper cannot prove that and then decline the comparison.

**Scientific reason.** To determine whether retraining a boosted ensemble under a covariance penalty buys anything over rescaling the unpenalized ensemble's centered predictions — in accuracy, in ratio shape, in nonlinear structure, or in temporal transfer.

**Minimal rigorous design.**
- Comparator: `f_b(x) = ȳ_T + b·(f₀(x) − ȳ_T)`, with `f₀` the unpenalized fit and `ȳ_T` computed **from the training block of the relevant fold only**.
- `b` is a **path**, not a selected value. Grid: `b ∈ [1, b_max]` on a log-spaced grid, with `b_max = 1/R²_T` (in-sample log-scale OLS-analogue `R²` on the training block) extended ~20% beyond to allow overshoot. Rationale: `b = 1/R²_T` is exactly the training-sample covariance-zeroing value (verified in §E). Match grid resolution to the 82-point ρ grids.
- Per fold: fit `f₀` on `T_v`, compute `ȳ_{T_v}` and `R²_{T_v}` on `T_v`, apply the whole `b` grid, evaluate on `V_v`. Held-out: refit `f₀` on the 344,607 pool, `ȳ` and `R²` from that pool. 2025: refit on 382,897, recompute both.
- Use the same frozen 994-tree configuration and the same `f₀` used for Ordinary LightGBM — after P0-B parity, so the comparator's base is the same object as the penalized families' origin.

**Comparison target.** The scientifically correct comparison is **at matched achieved first-order correction**, not at arbitrary `ρ ↔ b`. Procedure: for a set of target values `q` spanning the observed range (e.g. `β_log ∈ {−0.15, −0.12, −0.09, −0.06, −0.03, 0}`), interpolate along each family's path to the configuration achieving `β_log = q` **on the training/development block**, then report every metric on CV / held-out / 2025 for those matched configurations. Achieved-`β_log` matching must never be done on held-out or 2025 outcomes.

**Leakage controls.** `b`, `ȳ_T`, `R²_T`, and any matching target are estimated only on training/development data. No held-out or 2025 outcome enters `b` selection, grid construction, or matching. Freeze the comparator grid before evaluating out of time. Record that the held-out block was inspected during earlier development (already disclosed at L1888) so the comparator is not described as preregistered.

**Exact outputs required.** One machine-readable table row per (family ∈ {Direct, Surrogate, Post-hoc}, parameter value, evaluation ∈ {7 CV folds, held-out, 2025}) with the full metric suite; a matched-`β_log` table; and matched-`β_log` overlays on the ratio-shape, accuracy–equity, and mechanism figures.

**Metrics.** Predictive: `R²_P`, `MAE_P`, `MAPE_P`, `RMSE_log P`. Assessor-facing: median / mean / weighted-mean ratio, COD, COV, PRD, PRB, MKI, VEI (+ VEI Significance). Mechanism: `β_log`, `Δ_NL`, `dCor`, 30-bin ratio profile, VEI decile profile with 90% CIs.

**Acceptance / failure criterion.** There is no pass/fail — the comparator is descriptive. The criterion is *completeness*: every metric, at matched `β_log`, on all three evaluation regimes, with fold variation. The result is admissible whichever way it falls.

**Sections affected.** Abstract; contribution ¶ (L202–206); `subsec:comparators` (delete L2045 and `\todo` L2046); new Results subsection; `subsec:tradeoff_results`; Discussion L3002; Limitations L3061 (delete the "omits the closest post-processing comparator" sentence); Future Work L3071; App. D L3381 and `\todo` L3382.

**Consequences of each outcome.**
1. **Direct dominates post-hoc** — strongest result; supports in-processing, but must be attributed to retraining freedom (tree structure changes), not to the covariance objective per se, since the objectives coincide in the fixed-space limit.
2. **Roughly equivalent frontier** — the most likely outcome given Cor. `cor:path_scaling`. Not a negative result: it becomes the paper's cleanest finding, empirically confirming the theory in a fully retrained nonlinear learner, and it repositions the contribution as *mechanism + diagnostics* rather than *method superiority*. The paper survives and is more honest.
3. **Post-hoc dominates** — publishable and valuable; reframes as a cautionary applied-methods result. Requires rewriting Discussion and Conclusion, not retracting the theory.
4. **Same first-order behavior, different nonlinear/dependence behavior** — scientifically the most interesting. Would make `Δ_NL` / `dCor` the headline and justify the whole diagnostic apparatus.
5. **Different temporal transfer** — would justify the rolling-origin architecture as the paper's distinctive empirical contribution.

## G.2 — P0-B: Native vs custom-objective ρ=0 parity

**Verdict: REQUIRED, and the cheapest blocker.** L3679 indicates a parity run already exists in the project.

**Scientific reason.** At ρ=0 the supplied derivatives (`gᵢ=eᵢ`, `hᵢ=1`) are algebraically identical to LightGBM's L2 objective, so any prediction difference is an implementation artifact. Observed differences are large — held-out mean |Δŷ| = 3.24e−02 in log space (≈3.3% in price) and max 3.09e−01 (≈36%) — far beyond floating point. Until resolved, no native-to-penalized contrast is interpretable as a penalty effect.

**Alignment checklist (all must be verified identical).** Initial score / `boost_from_average` handling and the exact centering convention (`ỹ = y − b`, zero init, `b` added back — confirm `b` is the *fold-specific* training mean everywhere, including the 382,897-row refit); objective normalization (the `n/2` factor and whether it is applied to gradients only or gradients and Hessians); gradient and Hessian values at ρ=0, checked elementwise against native; `num_iterations` = 994 with early stopping **disabled or identically configured** (a differing `best_iteration` is the leading suspect); learning rate; all seeds (`seed`, `bagging_seed`, `feature_fraction_seed`, `data_random_seed`); `deterministic` and `num_threads` (thread count changes histogram accumulation order); identical feature matrix, categorical index set, and row order; row weights; `bagging_fraction`/`bagging_freq`; `feature_fraction`; `num_leaves`, `min_data_in_leaf`, `min_sum_hessian_in_leaf`, `max_depth`, `lambda_l1`, `lambda_l2`, `min_gain_to_split`, `max_bin`, `min_data_in_bin`; prediction transformation (`exp`, no smearing).

**Two Hessian-scale traps to check explicitly.** (i) `min_sum_hessian_in_leaf` and `lambda_l2` are absolute quantities — if the custom path supplies Hessians on a different scale than native (e.g. `2/n` instead of `1`), leaf-splitting and shrinkage thresholds change even though the minimizer does not. (ii) `min_gain_to_split` is likewise scale-dependent. These are the most common causes of exactly this symptom.

**Sufficient numerical agreement.** Target: max |Δŷ| ≤ 1e−6 on the log scale, with all reported metrics agreeing to the displayed precision. If exact parity is unattainable, the acceptance criterion is max |Δŷ| ≤ 1e−3 (≈0.1% in price) **and** a documented, named cause. Anything larger must be reported as an unresolved implementation difference, with all native-to-penalized comparisons removed.

**Outputs.** Extended `tab:rho_zero_control` with the full metric suite (P0-2); a parity diagnostic reporting max/mean/percentile |Δŷ| and `best_iteration` for both paths; the exact configuration hash (already requested at `\todo` L3681).

**Claims provisional until resolved.** Every bold/asterisk cell in `tab:path_anchor_summary` and `tab:path_anchor_complementary`; "moderate regularization need not impose an accuracy cost" (L2980, L2859); the ρ≈1 Direct accuracy-and-equity improvement (L2750); the baseline-vs-penalized narrative in Discussion L2978.

## G.3 — P0-C: Temporal boundaries and repeated parcels

**Verdict: REQUIRED but bounded.** This is a robustness audit, not a redesign.

**Same-date observations.** The manuscript already discloses the issue (L1890: the 90/10 split is on ordered observations, so cutoff-date sales appear on both sides). Yes — same-date transactions should be kept on one side. **Rationale:** the paper's central claim is about *chronological* transfer, so a boundary that is not a clean date boundary weakens the claim it is meant to support. **Cost:** trivial — moving one date's sales shifts the split by well under 0.1% of the pool. There is no reason not to fix it. Apply the same rule to all seven CV fold boundaries.

**Repeat sales.** Blocking is **warranted for a robustness check, not for the primary design.** Reasoning: (i) the risk is real — Cook County residential over 2016–2025 will contain a material share of repeat transactions, and a parcel's earlier sale in training gives the model parcel-specific information about its later sale, inflating held-out accuracy; (ii) but the effect on the *paper's actual claims* is likely second-order, because those claims concern **differences along a regularization path** with the base learner held fixed, and any repeat-sale advantage is shared across all ρ. Full parcel-blocking of the primary design would also break comparability with CCAO's production workflow, which does not block repeat sales. **Recommendation:** keep the primary design; add a parcel-blocked robustness path.

**If required, specify.** (a) Revised split: enforce same-date integrity at all boundaries; construct a parcel-blocked variant in which every PIN's transactions fall entirely on one side of each train/validation and train/held-out boundary. (b) Comparison: rerun the Direct and Surrogate paths (a coarser grid, e.g. the five display anchors plus the candidate-region endpoints, is sufficient) under the blocked design. (c) Recheck: the baseline comparison table; the anchor table; `β_log`, `Δ_NL`, `dCor` paths; the CV candidate-region endpoints. (d) **Threshold that would threaten conclusions:** if the sign or ordering of the `β_log` path changes, if the Surrogate `dCor` rebound disappears, or if the candidate-region endpoints move by more than roughly a factor of two in ρ, the temporal-portability conclusions require rewriting. Ordinary shifts in absolute accuracy do not threaten anything, because all claims are within-path.

**Do not** exclude repeat sales from the primary sample. That would be destructive, would break comparability with CCAO practice and with Candogan et al., and is not justified by the paper's claims.

---

# H. Full-paper independent referee findings

*(Additional to the prior-art audit.)*

| # | Location | Issue | Why it matters | Required fix | Conf. |
|---|---|---|---|---|---|
| H-1 | `tab:rho_zero_control` L3659–3676 | The ρ=0 controls are reported on only 3 of ~17 metrics | Makes the confounding in P0-1 unrepairable by the reader | Full metric suite (see P0-2) | 96 |
| H-2 | §4.2 L1947–1955; App. F | Nested expanding-window folds summarized as if exchangeable | SDs understate uncertainty; 7/7 LOFO is near-mechanical when folds share most data | Disclose nesting; reframe LOFO language | 92 |
| H-3 | Throughout Results | Descriptive point estimates only; no uncertainty on any path difference | 82 grid points × 2 families × 3 regimes × ~17 metrics is a very large descriptive surface; some reported movements are small (e.g. Δ_NL 0.116→0.119) | Add block-bootstrap or fold-level intervals for at least the headline anchors; already promised in Future Work L3071 — promote it | 93 |
| H-4 | §5.2 L2750; `tab:path_anchor_summary` | Direct at ρ≈1 improves **both** accuracy and every vertical-equity diagnostic vs Ordinary LightGBM | A free lunch on both axes is exactly the signature of an implementation confound, not of a penalty. Must not be presented as a substantive finding until P0-B closes | Hold the claim; re-derive against the ρ=0 origin | 94 |
| H-5 | §2.4 L1325–1333 vs App. D | The paper's own `C₀ = −‖f₀−y‖ₙ²` implies that in-sample, a **more accurate** model has `β_log` **closer to zero** | This directly tensions with the baseline motivation (LightGBM is both more accurate *and* more regressive). The tension is resolvable — the identity is in-sample and in a fixed linear space, while the table is out-of-sample and nonlinear — but the paper never addresses it, and a sharp referee will | Add a short reconciliation paragraph; report training-sample `β_log` for both baselines alongside the out-of-sample values | 90 |
| H-6 | §3.3, Surrogate | Weights `1+ρ(y−ȳ)²` are symmetric and sign-independent; no shrinkage guarantee | The paper's name for the method ("covariance-guided") over-promises for the Surrogate; the fixed-space theory (L3576) concedes it | Surface the concession in §3.3, not only in App. D | 91 |
| H-7 | §3.3 L1617; §5 | Quadratic weights are unbounded in log-price distance | At ρ=100 a sale 3 log-units from center carries weight 901. The paper notes the log transform "compresses this leverage ... but does not bound it," then does not diagnose it | Report the effective-weight distribution (e.g. top-1% weight share) at the display anchors; this is likely the mechanism behind the high-ρ Surrogate trough | 88 |
| H-8 | §2.2 L358; Limitations L3057 | Retransformation by direct exponentiation, correctly disclaimed | All price-scale metrics (`R²_P`, `MAE_P`, `MAPE_P`, PRD, MKI, VEI) depend on this choice, and the Surrogate changes tail fit and therefore the retransformation bias non-uniformly along ρ | Report at least one smearing-corrected sensitivity row at the display anchors — this is cheap and closes a real objection | 89 |
| H-9 | §5.2 L2750, §5.5 L2890–2895 | The candidate-region screen is a data-derived heuristic reported with five-significant-figure endpoints (0.355648, 2.559548, 0.202359, 2.222996) | False precision. The screen is explicitly not an estimator or CI, yet the presentation implies resolution the method cannot support | Round to grid-index resolution; state the grid spacing | 90 |
| H-10 | `subsec:external_results_boundary` L2901 vs Intro L183 | External-validity boundary correctly disclaims transfer; Introduction still promises six-county evidence | Self-contradiction in the rendered document (P0-9) | Remove the promise | 98 |
| H-11 | §2.3 MKI L619 | MKI orders observations by **sale price**, which the ED2 note identifies as the ordering biased toward regressivity | The manuscript flags this honestly, but then reports MKI against a band as a headline vertical-equity metric | Either compute MKI on the equal-weight proxy ordering as a sensitivity, or downgrade MKI's role | 87 |
| H-12 | §4.1 L1880 `\todo` | The exact CCAO extract/version/date is unspecified | Reproducibility; the paper's entire empirical basis is one data pull | Resolve before circulation | 95 |
| H-13 | §4.2 L1962; `tab:ccao_design` L2020; `\todo` L2025 | Two-stage grid history (50 points, then a 32-point lower-tail extension after seeing the first transition analysis) | Honestly disclosed — a genuine strength — but the extension was motivated by an observed result, so the augmented grid is not prespecified. The paper says this; the Results occasionally read as if it were | Keep the disclosure; ensure no Results sentence describes the 82-point grid as prespecified | 91 |
| H-14 | §5.5 L2884–2885 | Surrogate CV span conclusion **reverses** between the struck and active versions (fails vs supports the five-metric construction), both rendering | Reader cannot tell which conclusion holds | Resolve to one (P0-13) | 95 |
| H-15 | Discussion L2994 | "The value of the approach is operational" | No assessor-facing evaluation, desk review, or full-roll test supports an operational claim; Future Work L3071 concedes desk review is still needed | Downgrade to "the approach is *implementable* within a standard boosted-tree workflow"; reserve "operational value" for after review | 90 |
| H-16 | §2.3 L1196 | Subgroup diagnostics deferred to "later operational validation" | Both standards and the paper's own Related Work emphasize that countywide statistics conceal geographic and class-level inequity. A global correction could worsen subgroup equity while improving the countywide slope — and nothing here would detect it | Add at least one stratified check (township or property class) at the display anchors, or state explicitly that subgroup effects are unmeasured and could be adverse | 92 |

---

# I. Bibliography and citation corrections

**Overall status:** every `\cite` in active (non-commented, non-`\iffalse`) text resolves — **45 cited keys, 0 missing**. The problems are omissions, duplicates, and metadata.

### Must-add active references
Treder et al. 2021 (*Front. Psychiatry* 12:615754, doi 10.3389/fpsyt.2021.615754); Wang et al. 2023 (*IEEE Trans. Med. Imaging* 42(6):1577–, doi 10.1109/TMI.2022.3231730); Ren et al. 2026 (*IEEE ISBI 2026*, doi 10.1109/ISBI61048.2026.11515552); Lee & Chen 2026 (arXiv:2605.29255v1 [stat.ME], 28 May 2026 — **label as a preprint**); Smith et al. 2019 (brain-age delta estimation technical note); Beheshti et al. 2019 (bias-adjustment scheme). Metadata above is transcribed from the supplied PDFs; **volume/page details for Smith 2019 and Beheshti 2019 were not extracted and must be completed from the sources — do not fabricate.**

They must go into `references.bib` or `references_additions.bib`, the only two files `\addbibresource` loads (L30–31).

### References currently playing the wrong conceptual role
- `KomiyamaEtAl2018`, `MaryEtAl2019`, `PerezSuayEtAl2017`, `LiEtAl2022Fairness`, `lee2022maximal`, `ScutariPaneroProissl2022` are positioned (L283) as the closest dependence-control lineage. They control dependence between predictions and a **sensitive attribute**. The closest lineage is residual-vs-outcome. Demote to general context.
- `McMillenSingh2023` is cited only for "regression-based diagnostics can be biased." Its stronger, directly applicable result — bias toward finding regressivity when evaluation sales were used to fit the assessments, plus model-form mismatch bias — belongs in §2.4 and Limitations.
- `CandoganHanLu2023` is cited as a mechanism contrast. It is also the empirical-design precedent (CCAO, LightGBM, 90/10 chronological). Acknowledge that.

### Duplicate or inconsistent entries
- `SmithEtAl2026` (FAccT `@inproceedings`) **and** `SmithHarveyBerryGoldinHo2026` (arXiv `@misc`) — same paper. Keep the published entry, delete the preprint.
- `IAAO2025ExposureRatio` (Sept 2025 draft) vs `IAAO2026ExposureRatio` (May 2026 ED2) — the 2025 entry is uncited and is a *different, earlier* draft. Remove it, or the wrong draft will eventually be cited.
- `10578d49-78f8-3ef9-8277-9b6add2bd071` — reference-manager UUID key for Sirmans, Gatzlaff & Macpherson (2008). Uncited. Rename or remove.
- `McMillen2020` names a two-author McMillen & Singh paper; inconsistent with `McMillenSingh2023`. Rename to `McMillenSingh2020`.

### Publication-status corrections
- `IAAO2026ExposureRatio`: **URL says `StandardonRatioStudies_Exposure-Mar2026.pdf` while title, month (`may`) and note say May 2026.** Given that the paper's standards argument turns on identifying ED2 exactly, verify and correct.
- `CandoganHanLu2023` is listed as an arXiv preprint — confirm whether a published version now exists.
- Lee & Chen 2026 must be labelled a preprint, not a journal article.

### Claims needing different citations
- Every "training-time residual–target correlation control" statement → Treder (currently uncited).
- Every "nonlinear in-processing correction" statement → Ren, Wang (currently uncited).
- The `Cov(y, ŷ−y)=0` endpoint → Lee & Chen (currently uncited).
- Post-hoc affine correction (App. D L3381, Future Work L3071) → Smith, Beheshti, Treder §2.6 (currently uncited).
- The mechanical-negative-covariance result (Prop. `prop:bayes_covariance`, Remark `rem:mechanical_covariance`) → currently uncited; Treder Eq. (13) and the Le et al. result it cites should be acknowledged.

### Historical references that should remain secondary-source attributed
`references_supplement_historical.txt` (Paglin & Fogarty 1972, Cheng 1974, Edelstein 1979, Sunderman et al. 1990) is **not loaded** by the `.tex` and none are cited. That is correct and should stay that way — the historical vertical-equity regression tradition is adequately carried by `IAAO2023VerticalReview` and `McMillenSingh2023` as secondary sources. Do not add primary citations to papers not consulted.

---

# J. Document-integrity cleanup

### Active TODOs that render in the compiled PDF
Confirmed present in extracted PDF text: L113 ("also redundant with previous story") and L119 ("This is also unclear or redundant") — **both on the Introduction contribution page** — and L2499 ("Treat Eq. (49) as a..."). The package is loaded as `\usepackage[textsize=tiny]{todonotes}` with **no `disable` option**, contradicting the source comment at L18–19 which asserts TODOs are hidden.

Full inventory of live `\todo` calls: L204, L206, L1880, L1966, L2025, L2046, L3204, L3382, L3679, L3681. (L2930 and L2969 are inside `\iffalse` and do not render.)

### Contradictory claims in the rendered document
- Intro (PDF L107) and §4 opening (PDF L1095) promise six-county / ATTOM experiments; App. H (PDF L4046) states the paper does not report them. The material is `\iffalse`'d at .tex L2059–2076, L2903–2971, L4298–4348.
- `oldrevisionblock` L2876–2888: struck text says the Surrogate **does not** support the five-metric CV construction; active text says it **does**. Both render.
- Discussion L2987–2989: three successive struck versions of the same transition paragraph render before the active version.
- App. E L3623 (struck) says the initialization "does not produce prediction parity"; L3624 (active) says the audit is in the next subsection. Both render.

### Stale text and counts
- L1981 `\oldtext{All 50 positive values...}` renders struck alongside L1982's "82". Verified arithmetic: 50 + 32 = 82 ✓.
- L2790, L2803, L2808, L2850, L2856, L2863, L2871–2873, L3070, L3749–3752, L4021 all carry struck predecessors that render.
- L1176 (commented) and the large commented Related Work block L215–250 are source-only and do not render — acceptable, but should be removed before submission.

### Verified as *correct* (no action)
- No duplicate `\label` among the 124 active labels — the second `app:attom` at L4301 is inside `\iffalse`.
- Fold arithmetic: fold 7 = 310,147 + 34,460 = 344,607 ✓; 344,607 + 38,290 = 382,897 ✓; fold windows advance in exact 15-month steps from 2016-01-01 ✓.
- Display anchors {0.0104811, 0.1, 0.954095, 10.4811, 100} used consistently in `tab:path_anchor_summary`, `tab:path_anchor_complementary`, and the surrounding prose ✓.
- `β_log` values internally consistent between `tab:rho_zero_control` (−0.150 / −0.147 held-out; −0.164 / −0.161 in 2025), `tab:ccao_baseline_complementary`, and Results prose L2831–2834 ✓ — the paper correctly uses the ρ=0 custom control as the mechanism-path origin, which makes the `tab:path_anchor_summary` convention the anomaly rather than a systematic error.

### Unresolved placeholders
`\safeincludegraphics` fallback boxes will silently substitute "Figure file not found" if `img/generated_v12_994/` is absent at compile time. Verify every figure renders before circulation; several labels still carry `_placeholder` suffixes (`fig:ratio_shape_path_placeholder`, `fig:mechanism_path_placeholder`, `fig:accuracy_equity_placeholder`, `fig:other_metric_paths_placeholder`, `fig:vei_group_profile_placeholder`) — rename for clarity.

---

# K. Canonical revised paper identity

> This paper translates a specific ratio-study concern — the price-related decline in valuation ratios known as regressivity — into a trainable first-order objective, and audits what that objective does and fails to do inside a research translation of the Cook County Assessor's Office residential LightGBM workflow. On a fixed sample the log-error/log-price covariance is a fixed rescaling of the log-ratio slope, so penalizing squared covariance targets exactly the quantity that ratio studies regulate; we prefer covariance to correlation because a correlation penalty can be satisfied by inflating residual variance rather than by flattening the slope. We derive the exact Lagrangian path of this soft objective in a fixed prediction space — recovering, in this setting, the centered-rescaling geometry already known from correlation-constrained regression — and we derive a Jensen-based observation-additive surrogate that is exactly weighted squared error with symmetric log-price-distance weights, and whose fixed-space path is a weighted projection with generally multi-directional geometry. Tracing both objectives, alongside a matched centered-spread post-hoc comparator, across seven rolling-origin folds, a later held-out block, and a forward year, we find that the targeted first-order association and several assessor-facing vertical-equity diagnostics move toward neutrality over substantial portions of both paths before stronger regularization degrades prediction and ratio shape — and, critically, that first-order neutrality implies neither a flat conditional ratio profile nor residual–price independence. Because sale price is simultaneously the training target and the market-value proxy, and because even the squared-error Bayes predictor has negative residual–outcome covariance, the correction controls a price-related pattern among observed sales; it does not identify inequity relative to latent market value, does not extend automatically to unsold parcels, and does not determine tax burdens.

### What the paper IS
A CCAO-centered applied-methods and diagnostic study: an objective translation, an exact Direct/Surrogate mechanism distinction with fixed-space geometry for both, a standard-interface boosting implementation with an honest account of what its curvature approximation retains, and a complete temporal regularization-path and failure-mode audit against a matched post-hoc comparator.

### What the paper IS NOT
A general fairness method. A claim that residual–target decorrelation, dependence regularization, nonlinear in-processing correction, one-dimensional scaling, target-dependent loss weighting, or accuracy/dependence tradeoffs are new. A penalty-selection or deployment paper. A claim that training-time correction is necessary or superior to recalibration. A claim that zero covariance establishes conditional unbiasedness, independence, latent-value equity, or tax-burden fairness. A multi-jurisdiction or compliance study.

### Core contribution in one sentence
A precise mass-appraisal translation of the residual–outcome covariance target, an exact separation of its Direct and Surrogate realizations in both geometry and implementation, and a complete temporal path audit showing that reducing the targeted first-order association leaves substantial residual–price structure intact.

### Strongest theoretical contribution
The Surrogate weighted-projection theory — Proposition `prop:surrogate_fixed_space`, Corollary `cor:surrogate_local`, and the spectral representation — establishing that the Surrogate is a related but structurally distinct objective with a generally multi-directional path, not a computational approximation to Direct. No precedent found in the supplied literature.

### Strongest empirical contribution
The demonstration, on complete 82-point paths across three temporal regimes, that `β_log` can be driven to approximately zero (Surrogate: −0.001 held-out, −0.015 in 2025 at ρ=100) while `Δ_NL` rebounds (0.099 → 0.124 held-out) and `dCor` rebounds (0.250 → 0.267 held-out; 0.258 → 0.266 in 2025), with a visibly non-monotone ratio profile. This is the paper's most transferable finding and the strongest available warning against treating any single vertical-equity statistic as sufficient.

### Strongest applied contribution
The demonstration that the correction is implementable inside an existing production-style boosted-tree workflow with one additional hyperparameter and no change to data, features, or review — combined with an unusually disciplined standards treatment that separates adopted 2013 guidance from May 2026 exposure-draft proposals and refuses to infer compliance from point estimates.

---

# L. Ordered execution plan

### Before experiments

| # | Task | Why now | Depends on | Tool | Edit manuscript? | Output |
|---|---|---|---|---|---|---|
| L1 | Verify the exact executed objective code against Eqs. `eq:baseline_learning`, `eq:direct_penalty`, `eq:surrogate_learning` and the `n/2` scaling; record configuration hash | Everything downstream is indexed by ρ; a factor-of-two convention error would silently relabel every path | — | Code audit (already requested at `\todo` L3681) | No | Signed normalization memo + hash |
| L2 | Compute the numeric magnitude of `(ρ/2n)cᵢ²` at every tested ρ, and `1+(ρ/2)V̂ar(y)` along `c` | Confirms or refutes P0-7 before any prose is written about it | L1 | Arithmetic on training `V̂ar(y)` | No | One table; decides P0-7 wording |
| L3 | Diagnose the native/custom ρ=0 discrepancy against the G.2 checklist | Cheapest blocker; L3679 says a parity run exists | L1 | LightGBM config diff | No | Named cause + parity config |
| L4 | Add the six prior-art `.bib` entries; deduplicate `SmithEtAl2026`, remove `IAAO2025ExposureRatio`, fix the ED2 URL/date, rename `McMillen2020` and the UUID key | Independent of all experiments; unblocks P0-5 | — | Manual | Bibliography only | Clean `.bib` |
| L5 | Resolve MKI band (P0-10) and recheck every MKI-based directional statement | Changes which results count as improvements | — | Recompute from existing paths | No | Revised band decision |
| L6 | Delete all `\todo`s and superseded `\oldtext`/`oldrevisionblock` regions; remove the six-county promises | The rendered PDF is currently not circulable | — | Manual | **Yes — integrity only** | Clean compile, visually inspected |

### P0 experiments

| # | Task | Why now | Depends on | Output |
|---|---|---|---|---|
| L7 | **P0-B parity rerun** and propagation through all primary artifacts | Every headline comparison depends on it | L1, L3 | Extended `tab:rho_zero_control` (full metric suite) + regenerated path tables |
| L8 | **P0-A centered-spread comparator**, full `b` path on `[1, ~1.2/R²_T]`, all three regimes | The paper's central missing comparison | L7 (shared `f₀`) | Comparator path table + matched-`β_log` table + figure overlays |
| L9 | **P0-C robustness**: same-date boundary integrity at all 8 boundaries; parcel-blocked variant on a coarse ρ grid | Supports the temporal claim that is the paper's empirical identity | L7 | Robustness comparison table |
| L10 | Cheap inferential additions: PRB standard errors; VEI Significance at baselines and anchors; one smearing sensitivity row; effective-weight distribution for the Surrogate | Closes H-3, H-7, H-8, P1-2, P1-3 at low cost | L7 | Four small tables |

### After experiments

| # | Task | Depends on |
|---|---|---|
| L11 | Determine which of G.1's five outcomes obtained; write the comparator result **before** touching Discussion | L8 |
| L12 | Re-derive every "improves on baseline" claim against the parity-corrected ρ=0 origin | L7 |
| L13 | Decide whether `b_∞ = 1/R²` enters the main text (recommended: yes, as the comparator endpoint) | L8 |

### Manuscript implementation

Order matters: **Related Work and §3.1 first** (P0-5, P0-4, P0-6 — these are experiment-independent and fix the framing), then §3.2/App. E (P0-7), then §4 comparators, then Results (P0-1, P0-2, L11), then Discussion / Limitations / Future Work (P1-10), then Abstract and Introduction **last** (P0-6, P0-9), then the title (P1-14). Writing the Abstract before the comparator result is known would guarantee a rewrite.

### Final adversarial review

Re-read the compiled PDF end to end for: any surviving `\todo` or struck text; any sentence implying in-processing superiority; any sentence implying compliance; any novelty claim on the D-matrix prohibited list; every number reconciled against the machine-readable path table; every figure rendering (no `\safeincludegraphics` fallback boxes); all 124 labels resolving. Then a targeted read by someone who knows the brain-age bias literature, and a second by a property-tax practitioner.

---

# M. Final go/no-go decision

**DO NOT EDIT YET — RESOLVE THE NATIVE/CUSTOM ρ=0 PARITY (P0-B) FIRST.**

Parity is the cheapest blocker, a parity run reportedly already exists in the project (L3679), and it determines whether the manuscript's most attractive empirical claim — that moderate Direct regularization improves accuracy *and* every vertical-equity diagnostic simultaneously — is a finding or an artifact, which in turn determines how Results, Discussion, Abstract and title must read. The centered-spread comparator (P0-A) should be launched in parallel because it shares the unpenalized base model with the parity run and because Corollary `cor:path_scaling` makes it the matched competitor rather than an optional extra. Three revisions are genuinely experiment-independent and can proceed immediately — the prior-art lineage and Treder attribution, the bibliography repair, and the document-integrity cleanup that currently leaves drafting notes and struck contradictory prose in the compiled Introduction. The mathematics is in good shape: I found no sign errors, no incorrect derivations, and one materially misleading characterization (the numerically inert diagonal Hessian), so the revision is a framing, attribution, and evidence problem rather than a correctness problem. The paper is publishable as a CCAO-centered applied-methods and diagnostic study once the comparator exists and the novelty claims are narrowed to the boundary in Section D.
