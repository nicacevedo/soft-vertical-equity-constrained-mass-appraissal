# Superseded revision text — archive

Extracted from `paper/paper_v17_option1.tex` at git HEAD `faa2adae66fd1725de5ecfdb536bda2ff0ed522b`, before the Tier-A markup neutralization.

Every passage below is text the manuscript had **already marked as deleted** (struck-through `\oldtext`, crossed `oldrevisionblock`, or `%`-commented). Tier A stops rendering it; this file preserves it verbatim.

## Census

| macro | active | inert (comment/`\iffalse`) | total |
|---|---:|---:|---:|
| `\oldtext` | 169 | 273 | 442 |
| `\newtext` | 67 | 15 | 82 |
| `\latesttext` | 336 | 222 | 558 |
| `\advisorchange` | 4 | 2 | 6 |
| `oldrevisionblock` | 2 | 2 | 4 |

## Suppressed passages (`\oldtext`, active only)

### L1981 · near `subsec:path_design`

```latex
All 50 positive values per family remain in the path figures and machine-readable result table.
```

### L1983 · near `subsec:path_design`

```latex
Ratio-shape figures may additionally show the zero-penalty path origin for visual continuity. Because the direct and surrogate penalties have different scales, the anchors organize each family separately and are not interpreted as equal-strength interventions.
```

### L1985 · near `subsec:path_design`

```latex
Display-anchor refresh: before compilation, extract the four new nominal-$0.01$ rows (Direct/Surrogate on held-out/2025) from the frozen final augmented path at tested $\rho=0.0104811313$, recompute the table-formatting flags from unrounded values, and regenerate \texttt{ratio\_shape\_evolution.pdf} and \texttt{ratio\_shape\_cv\_transition\_span\_only.pdf} using the same nominal anchor convention.
```

### L1987 · near `subsec:path_design`

```latex
The same prespecified regularization paths are also evaluated on the two out-of-time samples.
```

### L1991 · near `subsec:path_design`

```latex
These held-out and forward paths are descriptive: they are not used to change the baseline LightGBM configuration, the $\rho$ grid, the objective definitions, or the training design. The chronological CV, held-out, and 2025 paths are therefore three complementary views of the same prespecified experiment rather than sequential model-selection stages.
```

### L2754 · near `tab:path_anchor_summary`

```latex
The favorable moderate-penalty anchors in Table~\ref{tab:path_anchor_summary} are therefore descriptive examples, not evidence of a stable multi-metric optimum. Section~\ref{subsec:path_stability_results} evaluates that question separately using the frozen transition rule.
```

### L2755 · near `tab:path_anchor_summary`

```latex
The decade-spaced rows in Table~\ref{tab:path_anchor_summary} are therefore descriptive snapshots of different penalty scales, not evidence of a stable multi-metric optimum. Section~\ref{subsec:path_stability_results} evaluates path stability separately using the frozen transition rule.
```

### L2790 · near `subsec:ratio_shape_results`

```latex
Scalar diagnostics do not show where along the value distribution the ratio pattern changes. We therefore complement the sweep table with a selection-free ratio-shape view using the same 30 equal-count-bin construction as the baseline motivation figure and the prespecified $\rho$ display anchors within each penalty family.
```

### L2796 · near `subsec:ratio_shape_results`

```latex
Valuation-ratio profiles against sale price at the prespecified display anchors. The horizontal line at 1 is the principal neutrality reference; lines at 0.9 and 1.1 are aggregate appraisal-level reference guides and are not binwise acceptance criteria.
```

### L2803 · near `fig:ratio_shape_path_placeholder`

```latex
At the prespecified anchors currently displayed, the Direct profile changes more smoothly than the Surrogate profile.
```

### L2808 · near `fig:ratio_shape_path_placeholder`

```latex
The fixed cross-fitted $\Delta_{\mathrm{NL}}$ nonlinearity gap separates these families. Along the Direct anchors, held-out $\Delta_{\mathrm{NL}}$ remains near $0.116$--$0.132$ and is $0.115$ at $\rho=100$; the 2025 Direct path is similarly stable, ending at $0.119$. The Surrogate path instead falls through moderate regularization, from $0.116$ at $\rho=0$ to $0.099$ near $\rho=0.954$ held-out and from $0.121$ to $0.092$ near $\rho=10.481$ in 2025, then rises at $\rho=100$ to $0.124$ and $0.111$, respectively. That rebound coincides with the non-monotone Surrogate ratio trough above. We do not infer a specific functional shape from the scalar gap alone; the shape label comes from the ratio-profile figure.
```

### L2826 · near `subsec:mechanism_results`

```latex
Mechanism and residual-structure paths versus $\rho$ for Direct and Surrogate on held-out (solid) and 2025 (dashed) evaluations. The $\beta_{\log}=0$ line is a first-order neutrality reference; $\Delta_{\mathrm{NL}}$ and dCor are not given assessor-style zero targets. Gray fill and dashed vertical boundaries mark the frozen family-specific CV-derived descriptive transition span; they are not a selected or recommended penalty interval.
```

### L2850 · near `subsec:tradeoff_results`

```latex
PRD and VEI provide complementary ratio-based and grouped views and are therefore the two main-text tradeoff panels. PRB and MKI are retained as companion diagnostics in Appendix~\ref{app:ccao_path_details}.
```

### L2856 · near `subsec:tradeoff_results`

```latex
Accuracy--equity trajectories with $R^2_P$ on the vertical axis and PRD, PRB, MKI, and VEI on the horizontal axis, for held-out and 2025 evaluations. Linear and ordinary LightGBM are context anchors. Arrows indicate increasing $\rho$ and do not mark a selected point.
```

### L2863 · near `fig:accuracy_equity_placeholder`

```latex
Accordingly, the main accuracy--equity figure does not highlight the Direct CV transition span as a preferred segment; temporal portability of that span is assessed separately below.
```

### L2871 · near `subsec:path_stability_results`

```latex
We therefore reserve the full fold-level stability traces and the
prediction/level/dispersion paths for Appendix~\ref{app:ccao_path_details}, while
reporting the main path findings in the figures above.
```

### L2877 · near `subsec:path_stability_results`

```latex
The chronological folds support the direction of the targeted first-order mechanism but do not imply a stable multi-metric operating point. On Direct, $\beta_{\log}$ becomes less negative from $\rho=0$ to $\rho=100$ in all seven folds. Applying the frozen five-metric transition rule to the equal-weight chronological-CV path yields a descriptive Direct span $\rho\in[0.1,1.099]$. All seven leave-one-fold-out aggregations also yield a five-event positive-interior span, so this within-CV pattern is not driven by any single fold. The lower endpoint requires an important qualification: $\rho=0.1$ is the smallest positive penalty on the tested grid and therefore is not an estimated lower transition threshold.
```

### L2880 · near `subsec:path_stability_results`

```latex
Temporal concordance of the exact event locations is substantially weaker. After freezing the Direct CV span, only one of the five held-out turning events and none of the five 2025 turning events fall inside it. Value-based span regret gives a more nuanced result than event locations alone: held-out $\operatorname{RMSE}_{\log P}$ has zero span regret, whereas the other four held-out primary metrics have strictly positive regret; all five 2025 primary metrics have strictly positive regret. Because no substantive ``small-regret'' threshold is imposed, these signs are not used to label the positive regrets as material or negligible. The evidence is therefore mixed by metric: the Direct span is a reproducible description of the development-period CV path, but it does not define a temporally invariant penalty region.
```

### L2883 · near `subsec:path_stability_results`

```latex
The Surrogate does not support the same five-metric construction in the full chronological-CV aggregate because $\operatorname{RMSE}_{\log P}$ is minimized at the $\rho=0$ boundary; only two of seven leave-one-fold-out aggregations produce a common positive-interior span. Thus, moderate Surrogate regularization can still improve several predictive and assessor-facing criteria, but those improvements do not organize into the same common five-metric CV transition structure as Direct. Appendix~\ref{app:ccao_path_details} reports the event locations, leave-one-fold-out diagnostics, and Direct regret values. These are descriptive path-stability results and do not select a model family or operating point.
```

### L2886 · near `subsec:path_stability_results`

```latex
The corresponding metric-value losses are generally limited in magnitude over the observed paths, although no materiality threshold is imposed.
```

### L2987 · near `sec:discussion`

```latex
The transition analysis adds a separate temporal qualification. A targeted mechanism can be directionally stable without yielding a stable multi-metric operating point. Direct exhibits an internally reproducible five-metric CV transition span, including seven-of-seven leave-one-fold-out support, but the exact held-out and forward turning locations show limited concordance with that interval and the value-based regret result is mixed across metrics. The Surrogate does not support the analogous common positive span in the full CV aggregate. Chronological validation is therefore valuable for characterizing the regularization path, but the evidence does not support treating a particular raw-$\rho$ interval as an intrinsic or temporally invariant property of either objective. Any later deployment calibration should specify explicitly which predictive, valuation, and equity criteria define the operating decision and re-evaluate that decision across time.
```

### L2988 · near `sec:discussion`

```latex
The transition analysis adds a separate temporal qualification. After resolving the lower tail of the grid, both Direct and Surrogate exhibit positive-interior five-metric CV transition spans with seven-of-seven leave-one-fold-out support. Exact out-of-time event portability remains weak, however: Direct has zero of five held-out and zero of five 2025 events inside its frozen span, while Surrogate has one of five in each period. Chronological validation is therefore useful for characterizing repeatable path structure, but the evidence does not support treating either raw-$\rho$ interval as an intrinsic or temporally invariant operating region. Any later deployment calibration should specify explicitly which predictive, valuation, and equity criteria define the operating decision and re-evaluate that decision across time.
```

### L2989 · near `sec:discussion`

```latex
The transition analysis adds a separate temporal qualification. After resolving the lower tail of the grid, both Direct and Surrogate exhibit positive-interior five-metric CV transition spans with seven-of-seven leave-one-fold-out support. Exact out-of-time event portability remains weak, however: Direct has zero of five held-out and zero of five 2025 events inside its frozen span, while Surrogate has one of five in each period. The augmented evidence distinguishes a reproducible prediction/COD transition structure from the much larger penalties at which assessor-facing measures or mechanism diagnostics become closest to neutrality. Therefore there is no single empirically supported ``safe'' $\rho$ interval, and a future deployment choice requires an explicit operating criterion. Chronological validation is therefore useful for characterizing repeatable path structure, but the evidence does not support treating either raw-$\rho$ interval as an intrinsic or temporally invariant operating region. Any later deployment calibration should specify explicitly which predictive, valuation, and equity criteria define the operating decision and re-evaluate that decision across time.
```

### L3070 · near `tab:roles`

```latex
The main methodological extension is structured covariance control by geography, property class, and time, with safeguards for small or unstable groups. That extension should be judged against stratified post-hoc recalibration and against a stronger base AVM.
```

### L3193 · near `eq:nonlinearity_gap_estimator`

```latex
The estimator is reported on the primary
held-out and 2025 forward paths and is not computed retrospectively for the already
completed chronological CV.
```

### L3272 · near `prop:cov_prd`

```latex
Exact Fixed-Space Regularization Path
```

### L3275 · near `app:theory`

```latex
This appendix isolates the geometry of the two penalties when the prediction space is held fixed. The Direct result provides an exact rank-one benchmark for the targeted covariance correction. The Surrogate result below provides the corresponding weighted-projection benchmark and makes precise why its path need not remain one-dimensional. These finite-sample projection results are used for mechanism interpretation only; they do not assert that a fully retrained boosting path remains in a fixed prediction space.
```

### L3380 · near `rem:mechanical_covariance`

```latex
This equivalence is specific to the exact fixed-prediction-space benchmark and motivates the centered recalibration comparator in the previous draft.
```

### L3581 · near `eq:surrogate_spectral_path`

```latex
Taken together, the fixed-space results give a parallel interpretation of the two methods. Direct regularization applies a rank-one correction aligned with the centered price direction and, with an intercept in the prediction space, becomes an exact centered spread rescaling. Surrogate regularization changes the projection geometry through the diagonal weights $1+\rho c_i^2$: locally it repairs the learnable tail-weighted residual pattern, and globally it can combine multiple correction modes. In both cases the baseline squared-error cost is second order near $\rho=0$, but only the Direct objective directly guarantees shrinkage of the targeted covariance within this benchmark.
```

### L3623 · near `app:implementation`

```latex
The empirical $\rho=0$ audit shows that this initialization does not produce prediction parity with native LightGBM beyond the initial score. Direct and Surrogate $\rho=0$ predictions coincide with each other, and the remaining native/custom gap is the custom-objective API path rather than a positive-$\rho$ objective error.
```

### L3749 · near `app:ccao_path_details`

```latex
This appendix contains the complete diagnostics underlying the compact main-text
regularization-path presentation. It is intentionally organized around the full
prespecified paths and not around a selected configuration.
```

### L3752 · near `app:ccao_path_details`

```latex
This appendix contains the complete diagnostics underlying the compact main-text regularization-path presentation. It is organized around the final augmented paths and not around a selected configuration; only the compact display anchors remain prespecified presentation choices.
```

### L3762 · near `app:ccao_path_details`

```latex
The compact main-text table uses only the prespecified positive display anchors together with the ordinary-LightGBM baseline; zero-penalty custom-objective diagnostics are kept in the implementation appendix. No path row is omitted from the machine-readable results because of its observed performance.
```

### L3976 · near `tab:path_anchor_complementary`

```latex
The machine-readable combined path table now includes held-out and 2025 $\Delta_{\mathrm{NL}}$ for Linear, ordinary LightGBM, and every Direct/Surrogate grid point. The fixed cross-fitted estimator specification in Appendix~\ref{app:metrics} is archived with those values. Fold-level $\Delta_{\mathrm{NL}}$ is not added to the already completed chronological CV. Split-specific $\rho=0$ prediction-parity audits and the centered-recalibration validation record, including $b^\star$ and row-level recalibrated predictions, are stored in the same result root.
```

### L3977 · near `tab:path_anchor_complementary`

```latex
The machine-readable combined path table includes held-out and 2025 $\Delta_{\mathrm{NL}}$ for Linear, ordinary LightGBM, and every Direct/Surrogate grid point. The fixed cross-fitted estimator specification in Appendix~\ref{app:metrics} is archived with those values. Fold-level $\Delta_{\mathrm{NL}}$ is not added retrospectively to the already completed chronological CV. Split-specific $\rho=0$ prediction-parity audits are stored in the same result root.
```

### L4000 · near `subsec:transition_appendix`

```latex
The main text reports only the interpretation of the five-metric transition diagnostic. The objects below provide the auditable event locations and value-based checks. The Direct interval is always described as a \emph{CV-derived descriptive transition span}, never as a recommended, safe, optimal, or deployment range.
```

### L4001 · near `subsec:transition_appendix`

```latex
The main text reports only the interpretation of the five-metric transition diagnostic. The objects below provide the auditable event locations and fold/leave-one-fold-out stability checks. Any family-specific interval is described only as a \emph{CV-derived descriptive transition span}, never as a recommended, safe, optimal, preferred, or deployment range.
```

### L4009 · near `subsec:transition_appendix`

```latex
Cross-metric turning-event summary for the frozen chronological-CV diagnostic and its retrospective temporal comparison. The Direct interval is a CV-derived descriptive transition span, not a selected or recommended penalty range.
```

### L4018 · near `tab:transition_summary`

```latex
$\rho=0.1$ (interior $+$)
```

### L4021 · near `tab:transition_summary`

```latex
$\rho=0.1$ (interior $+$)
```

### L4022 · near `tab:transition_summary`

```latex
$\rho=0$ (boundary ($\rho=0$))
```

### L4027 · near `tab:transition_summary`

```latex
valid positive-interior span $[0.1,\,1.099]$; $\log_{10}$-width $1.04$
```

### L4028 · near `tab:transition_summary`

```latex
not supported
```

### L4030 · near `tab:transition_summary`

```latex
$\rho=0.1$ is the smallest positive tested value, not an estimated lower threshold
```

### L4031 · near `tab:transition_summary`

```latex
---
```

### L4033 · near `tab:transition_summary`

```latex
7/7; valid endpoints $[0.1,\,0.153]$ to $[1.099,\,1.265]$
```

### L4033 · near `tab:transition_summary`

```latex
7/7
```

### L4034 · near `tab:transition_summary`

```latex
2/7
```

### L4034 · near `tab:transition_summary`

```latex
7/7
```

### L4036 · near `tab:transition_summary`

```latex
1/5
```

### L4037 · near `tab:transition_summary`

```latex
0/5
```

### L4040 · near `tab:transition_summary`

```latex
0/5
```

### L4041 · near `tab:transition_summary`

```latex
Value-based reading
```

### L4042 · near `tab:transition_summary`

```latex
mixed by metric
```

### L4043 · near `tab:transition_summary`

```latex
no common five-metric positive CV span
```

### L4062 · near `tab:transition_summary`

```latex
Turning-event locations for the five fixed criteria. Gray shading, where present, is the frozen CV-derived descriptive transition span and is not a selected or recommended penalty interval.
```

### L4069 · near `fig:transition_event_locations`

```latex
Closest-to-neutral observed-grid locations for PRD, PRB, MKI, and VEI. Gray fill and dashed vertical boundaries mark the already-frozen five-primary-metric CV-derived descriptive transition span; these events do not redefine that span and do not select $\rho$.
```

### L4076 · near `fig:vertical_equity_event_locations`

```latex
Mechanism turning-event locations corresponding to minimum $|\beta_{\log}|$ and minimum dCor. Gray fill and dashed vertical boundaries mark the frozen five-primary-metric CV-derived descriptive transition span as context only. No CV $\Delta_{\mathrm{NL}}$ event is shown because $\Delta_{\mathrm{NL}}$ was not computed retrospectively for chronological CV. These events do not define a selected or recommended range.
```

### L4076 · near `fig:vertical_equity_event_locations`

```latex
Mechanism turning-event locations corresponding to minimum $|\beta_{\log}|$ and minimum dCor. Gray fill and dashed vertical boundaries mark, for context only, the frozen CV-derived descriptive transition span constructed from the five prediction/COD criteria. No CV $\Delta_{\mathrm{NL}}$ event is shown because $\Delta_{\mathrm{NL}}$ was not computed retrospectively for chronological CV. These events do not define a selected or recommended range.
```

### L4084 · near `fig:mechanism_event_locations`

```latex
Value-based cost of restricting the Direct out-of-time paths to the frozen CV-derived descriptive span. No small/large regret threshold is imposed.
```

### L4092 · near `fig:mechanism_event_locations`

```latex
Metric
```

### L4092 · near `fig:mechanism_event_locations`

```latex
$\rho^{*}$
```

### L4092 · near `fig:mechanism_event_locations`

```latex
value$^{*}$
```

### L4092 · near `fig:mechanism_event_locations`

```latex
$\rho_{\mathrm{span}}$
```

### L4092 · near `fig:mechanism_event_locations`

```latex
value$_{\mathrm{span}}$
```

### L4092 · near `fig:mechanism_event_locations`

```latex
raw regret
```

### L4092 · near `fig:mechanism_event_locations`

```latex
norm. regret
```

### L4092 · near `fig:mechanism_event_locations`

```latex
$\log_{10}$ dist.
```

### L4094 · near `fig:mechanism_event_locations`

```latex
\textit{Panel A: held-out evaluation}
```

### L4096 · near `fig:mechanism_event_locations`

```latex
$R^2_P$
```

### L4096 · near `fig:mechanism_event_locations`

```latex
3.393
```

### L4096 · near `fig:mechanism_event_locations`

```latex
0.901
```

### L4096 · near `fig:mechanism_event_locations`

```latex
1.099
```

### L4096 · near `fig:mechanism_event_locations`

```latex
0.899
```

### L4096 · near `fig:mechanism_event_locations`

```latex
0.0018
```

### L4096 · near `fig:mechanism_event_locations`

```latex
0.056
```

### L4096 · near `fig:mechanism_event_locations`

```latex
0.490
```

### L4097 · near `fig:mechanism_event_locations`

```latex
$\operatorname{MAE}_P$
```

### L4097 · near `fig:mechanism_event_locations`

```latex
1.931
```

### L4097 · near `fig:mechanism_event_locations`

```latex
\$73,924
```

### L4097 · near `fig:mechanism_event_locations`

```latex
0.829
```

### L4097 · near `fig:mechanism_event_locations`

```latex
\$74,409
```

### L4097 · near `fig:mechanism_event_locations`

```latex
\$485
```

### L4097 · near `fig:mechanism_event_locations`

```latex
0.044
```

### L4097 · near `fig:mechanism_event_locations`

```latex
0.245
```

### L4098 · near `fig:mechanism_event_locations`

```latex
$\operatorname{MAPE}_P$
```

### L4098 · near `fig:mechanism_event_locations`

```latex
1.265
```

### L4098 · near `fig:mechanism_event_locations`

```latex
21.0\%
```

### L4098 · near `fig:mechanism_event_locations`

```latex
0.829
```

### L4098 · near `fig:mechanism_event_locations`

```latex
21.1\%
```

### L4098 · near `fig:mechanism_event_locations`

```latex
0.033 pp
```

### L4098 · near `fig:mechanism_event_locations`

```latex
0.011
```

### L4098 · near `fig:mechanism_event_locations`

```latex
0.061
```

### L4099 · near `fig:mechanism_event_locations`

```latex
$\operatorname{RMSE}_{\log P}$
```

### L4099 · near `fig:mechanism_event_locations`

```latex
0.720
```

### L4099 · near `fig:mechanism_event_locations`

```latex
0.289
```

### L4099 · near `fig:mechanism_event_locations`

```latex
0.720
```

### L4099 · near `fig:mechanism_event_locations`

```latex
0.289
```

### L4099 · near `fig:mechanism_event_locations`

```latex
0
```

### L4099 · near `fig:mechanism_event_locations`

```latex
0
```

### L4099 · near `fig:mechanism_event_locations`

```latex
0
```

### L4100 · near `fig:mechanism_event_locations`

```latex
COD
```

### L4100 · near `fig:mechanism_event_locations`

```latex
1.265
```

### L4100 · near `fig:mechanism_event_locations`

```latex
21.52
```

### L4100 · near `fig:mechanism_event_locations`

```latex
1.099
```

### L4100 · near `fig:mechanism_event_locations`

```latex
21.53
```

### L4100 · near `fig:mechanism_event_locations`

```latex
0.015
```

### L4100 · near `fig:mechanism_event_locations`

```latex
0.005
```

### L4100 · near `fig:mechanism_event_locations`

```latex
0.061
```

### L4102 · near `fig:mechanism_event_locations`

```latex
\textit{Panel B: 2025 forward evaluation}
```

### L4104 · near `fig:mechanism_event_locations`

```latex
$R^2_P$
```

### L4104 · near `fig:mechanism_event_locations`

```latex
1.931
```

### L4104 · near `fig:mechanism_event_locations`

```latex
0.912
```

### L4104 · near `fig:mechanism_event_locations`

```latex
1.099
```

### L4104 · near `fig:mechanism_event_locations`

```latex
0.910
```

### L4104 · near `fig:mechanism_event_locations`

```latex
0.0014
```

### L4104 · near `fig:mechanism_event_locations`

```latex
0.037
```

### L4104 · near `fig:mechanism_event_locations`

```latex
0.245
```

### L4105 · near `fig:mechanism_event_locations`

```latex
$\operatorname{MAE}_P$
```

### L4105 · near `fig:mechanism_event_locations`

```latex
2.560
```

### L4105 · near `fig:mechanism_event_locations`

```latex
\$76,558
```

### L4105 · near `fig:mechanism_event_locations`

```latex
1.099
```

### L4105 · near `fig:mechanism_event_locations`

```latex
\$76,848
```

### L4105 · near `fig:mechanism_event_locations`

```latex
\$289
```

### L4105 · near `fig:mechanism_event_locations`

```latex
0.023
```

### L4105 · near `fig:mechanism_event_locations`

```latex
0.367
```

### L4106 · near `fig:mechanism_event_locations`

```latex
$\operatorname{MAPE}_P$
```

### L4106 · near `fig:mechanism_event_locations`

```latex
2.223
```

### L4106 · near `fig:mechanism_event_locations`

```latex
20.6\%
```

### L4106 · near `fig:mechanism_event_locations`

```latex
0.309
```

### L4106 · near `fig:mechanism_event_locations`

```latex
20.6\%
```

### L4106 · near `fig:mechanism_event_locations`

```latex
0.052 pp
```

### L4106 · near `fig:mechanism_event_locations`

```latex
0.020
```

### L4106 · near `fig:mechanism_event_locations`

```latex
0.306
```

### L4107 · near `fig:mechanism_event_locations`

```latex
$\operatorname{RMSE}_{\log P}$
```

### L4107 · near `fig:mechanism_event_locations`

```latex
1.456
```

### L4107 · near `fig:mechanism_event_locations`

```latex
0.278
```

### L4107 · near `fig:mechanism_event_locations`

```latex
0.625
```

### L4107 · near `fig:mechanism_event_locations`

```latex
0.278
```

### L4107 · near `fig:mechanism_event_locations`

```latex
0.0001
```

### L4107 · near `fig:mechanism_event_locations`

```latex
0.003
```

### L4107 · near `fig:mechanism_event_locations`

```latex
0.122
```

### L4108 · near `fig:mechanism_event_locations`

```latex
COD
```

### L4108 · near `fig:mechanism_event_locations`

```latex
2.223
```

### L4108 · near `fig:mechanism_event_locations`

```latex
21.05
```

### L4108 · near `fig:mechanism_event_locations`

```latex
0.309
```

### L4108 · near `fig:mechanism_event_locations`

```latex
21.11
```

### L4108 · near `fig:mechanism_event_locations`

```latex
0.062
```

### L4108 · near `fig:mechanism_event_locations`

```latex
0.027
```

### L4108 · near `fig:mechanism_event_locations`

```latex
0.306
```

### L4114 · near `fig:mechanism_event_locations`

```latex
\emph{Notes.} This Direct-only table belongs to the v1 span analysis. Its values are not carried forward as final v2 evidence because the final augmented experiment gives both families valid CV spans and therefore requires the regret diagnostic to be regenerated symmetrically.
```

### L4164 · near `tab:transition_regret`

```latex
Zero means the out-of-time global observed-grid optimum is available within the frozen span.
```

### L4172 · near `tab:transition_regret`

```latex
Exact event concordance is deliberately interpreted jointly with Table~\ref{tab:transition_regret}: a turning point can move outside the CV interval even when the best value available inside the interval is close to the global discrete-grid optimum. Event-sharpness diagnostics are retained in the machine-readable replication package to document this issue without defining a post-hoc near-optimality tolerance.
```

### L4173 · near `tab:transition_regret`

```latex
Exact event concordance is used here only as a strict location diagnostic. Event-sharpness diagnostics remain in the machine-readable replication package rather than being promoted to an additional selection rule.
```

### L4176 · near `tab:transition_regret`

```latex
The appendix previously included a separate PRB/MKI-versus-$R^2_P$ companion figure.
```

### L4181 · near `tab:transition_regret`

```latex
Throughout the $\rho$-evolution figures below, gray shading denotes the family-specific CV-derived descriptive transition span when one exists. It is a path-description aid, not a selected, safe, preferred, or recommended penalty interval.
```

### L4188 · near `tab:transition_regret`

```latex
Predictive-metric paths and valuation-level/uniformity paths along the Direct and Surrogate grids, with held-out and 2025 evaluations overlaid. Gray fill and dashed vertical boundaries mark the frozen family-specific CV-derived descriptive transition span. Valuation-level panels include the ratio $=1$ reference.
```

### L4193 · near `fig:other_metric_paths_placeholder`

```latex
Vertical-equity metric paths versus $\rho$ for Direct and Surrogate. Gray fill and dashed vertical boundaries mark the frozen family-specific CV-derived descriptive transition span. Reference lines mark PRD $=1$, PRB $=0$, MKI $=1$, and VEI $=0$.
```

### L4224 · near `tab:fold_structure`

```latex
The previous appendix included a separate four-metric fold-stability figure for $R^2_P$, PRD, VEI, and $\beta_{\log}$.
```

### L4228 · near `tab:fold_structure`

```latex
Chronological-fold predictive-metric paths (thin gray) and equal-weight CV means (thick). Gray fill and dashed vertical boundaries mark the frozen family-specific CV-derived descriptive transition span.
```

### L4233 · near `fig:cv_predictive_metric_paths`

```latex
Chronological-fold valuation-level and uniformity paths (thin gray) and equal-weight CV means (thick). Gray fill and dashed vertical boundaries mark the frozen family-specific CV-derived descriptive transition span. Valuation-level panels include the ratio $=1$ reference.
```

### L4238 · near `fig:cv_level_uniformity_paths`

```latex
Chronological-fold vertical-equity paths (thin gray) and equal-weight CV means (thick). Gray fill and dashed vertical boundaries mark the frozen family-specific CV-derived descriptive transition span. Reference lines mark PRD $=1$, PRB $=0$, MKI $=1$, and VEI $=0$.
```

### L4243 · near `fig:cv_vertical_equity_metric_paths`

```latex
Chronological-fold mechanism paths for $\beta_{\log}$, $\Delta_{\mathrm{NL}}$, and distance correlation (thin gray) and equal-weight CV means (thick). The $\beta_{\log}=0$ line is a first-order neutrality reference; $\Delta_{\mathrm{NL}}$ and dCor are not given forced zero lines. Gray fill and dashed vertical boundaries mark the frozen family-specific CV-derived prediction/COD transition span. CV $\Delta_{\mathrm{NL}}$ was reconstructed from frozen validation predictions with the same fixed estimator used out of time; it does not define that span or select $\rho$.
```

### L4252 · near `fig:cv_mechanism_metric_paths`

```latex
Ratio-shape profiles restricted to the prespecified display anchors that lie inside each family's frozen CV-derived descriptive transition span. Families without a valid common positive span show no penalized curves.
```

### L4252 · near `fig:cv_mechanism_metric_paths`

```latex
Ratio-shape profiles restricted to the prespecified display anchors that lie inside each family's frozen CV-derived descriptive transition span. The horizontal line at 1 is the principal neutrality reference; lines at 0.9 and 1.1 are aggregate appraisal-level reference guides and are not binwise acceptance criteria.
```

### L4257 · near `fig:ratio_shape_cv_transition_span_only`

```latex
Held-out assessor-facing diagnostics (horizontal) versus predictive metrics (vertical) along the Direct and Surrogate paths.
```

### L4262 · near `fig:tradeoff_equity_vs_accuracy_heldout`

```latex
2025 assessor-facing diagnostics (horizontal) versus predictive metrics (vertical) along the Direct and Surrogate paths.
```

### L4267 · near `fig:tradeoff_equity_vs_accuracy_2025`

```latex
Held-out mechanism/residual diagnostics (horizontal) versus predictive metrics (vertical).
```

### L4272 · near `fig:tradeoff_mechanism_vs_accuracy_heldout`

```latex
2025 mechanism/residual diagnostics (horizontal) versus predictive metrics (vertical).
```

## Suppressed `oldrevisionblock` environments

### L2876 · near `subsec:path_stability_results`

```latex
\oldtext{The chronological folds support the direction of the targeted first-order mechanism but do not imply a stable multi-metric operating point. On Direct, $\beta_{\log}$ becomes less negative from $\rho=0$ to $\rho=100$ in all seven folds. Applying the frozen five-metric transition rule to the equal-weight chronological-CV path yields a descriptive Direct span $\rho\in[0.1,1.099]$. All seven leave-one-fold-out aggregations also yield a five-event positive-interior span, so this within-CV pattern is not driven by any single fold. The lower endpoint requires an important qualification: $\rho=0.1$ is the smallest positive penalty on the tested grid and therefore is not an estimated lower transition threshold.}
\latesttext{The chronological folds support reproducible within-development path structure but do not identify a stable multi-metric operating point. After the lower-tail extension, the five Direct CV events all occur at positive interior grid points and define the descriptive span $\rho\in[0.0494,1.099]$. The event that had previously been censored at the first positive point---$\operatorname{RMSE}_{\log P}$---moves inward to $\rho=0.0494$, while the other four Direct event locations are unchanged at the reported grid resolution. All seven Direct leave-one-fold-out aggregations again yield a valid positive-interior five-event span.}

\oldtext{Temporal concordance of the exact event locations is substantially weaker. After freezing the Direct CV span, only one of the five held-out turning events and none of the five 2025 turning events fall inside it. Value-based span regret gives a more nuanced result than event locations alone: held-out $\operatorname{RMSE}_{\log P}$ has zero span regret, whereas the other four held-out primary metrics have strictly positive regret; all five 2025 primary metrics have strictly positive regret. Because no substantive ``small-regret'' threshold is imposed, these signs are not used to label the positive regrets as material or negligible. The evidence is therefore mixed by metric: the Direct span is a reproducible description of the development-period CV path, but it does not define a temporally invariant penalty region.}
\latesttext{Exact temporal concordance is nevertheless weak. None of the five Direct held-out turning events and none of the five Direct 2025 turning events fall inside the frozen v2 CV span. Thus, the lower-tail extension strengthens the interpretation of the Direct span as an interior description of the development-period CV path while providing no evidence that the same event envelope is a temporally invariant operating region.}

\oldtext{The Surrogate does not support the same five-metric construction in the full chronological-CV aggregate because $\operatorname{RMSE}_{\log P}$ is minimized at the $\rho=0$ boundary; only two of seven leave-one-fold-out aggregations produce a common positive-interior span. Thus, moderate Surrogate regularization can still improve several predictive and assessor-facing criteria, but those improvements do not organize into the same common five-metric CV transition structure as Direct. Appendix~\ref{app:ccao_path_details} reports the event locations, leave-one-fold-out diagnostics, and Direct regret values. These are descriptive path-stability results and do not select a model family or operating point.}
\latesttext{The lower-tail extension also changes the Surrogate conclusion. Its CV $\operatorname{RMSE}_{\log P}$ event moves from the $\rho=0$ boundary to the positive interior point $\rho=0.00222$, and its MAE event moves below $0.1$. The five Surrogate CV events therefore define their own descriptive span, $\rho\in[0.00222,0.954]$, and all seven Surrogate leave-one-fold-out aggregations now support a positive-interior five-event span. This does not make the Surrogate span a selected or portable operating range: only one of five held-out events and one of five 2025 events (MAE in each case) fall inside it. Taken together, the two families show that reproducible within-CV transition structure is distinct from temporal portability of the exact operating location. Appendix~\ref{app:ccao_path_details} reports the event locations and leave-one-fold-out diagnostics. These are descriptive path-stability results and do not select a model family or operating point.}
\latesttext{The family-specific CV spans remain Direct $\rho\in[0.0494,1.099]$ and Surrogate $\rho\in[0.00222,0.954]$. They should be interpreted as envelopes of the five prediction/COD turning events, not as equity-neutrality intervals. Appendix Figures~\ref{fig:vertical_equity_event_locations} and~\ref{fig:mechanism_event_locations} show that the closest-to-neutral locations for PRD, PRB, MKI, and VEI generally occur at much larger $\rho$ for both families; predictive/COD turning and equity neutrality therefore occur at distinct scales. Exact out-of-time event concordance remains weak, but the family-symmetric value-based span-regret result in Table~\ref{tab:transition_regret} is more nuanced. Direct normalized regret is roughly $0.005$--$0.056$ on the held-out sample and $0.004$--$0.037$ in 2025. Surrogate MAE regret is $0$ on both held-out and 2025 evaluations; Surrogate $\operatorname{RMSE}_{\log P}$ normalized regret is at most $0.003$; the largest reported Surrogate normalized regret is 2025 $R^2_P\approx 0.083$.}
\oldtext{The corresponding metric-value losses are generally limited in magnitude over the observed paths, although no materiality threshold is imposed.}
\latesttext{These regret values show that exact event-location disagreement and value-based loss are distinct aspects of temporal portability; no materiality threshold is imposed. Direct $\operatorname{RMSE}_{\log P}$ and Surrogate $R^2_P$ CV events are particularly value-flat, while MAE events are comparatively sharper; exact event-$\rho$ movement therefore should not be interpreted as equal substantive movement for all criteria.}
```

### L4082 · near `fig:mechanism_event_locations`

```latex
\noindent\textbf{Previous Direct-only span-regret table.}\par\smallskip
\oldtext{Value-based cost of restricting the Direct out-of-time paths to the frozen CV-derived descriptive span. No small/large regret threshold is imposed.}\par\smallskip
\centering
\scriptsize
\setlength{\tabcolsep}{2.6pt}
\renewcommand{\arraystretch}{1.08}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrrrr}
\toprule
\oldtext{Metric} & \oldtext{$\rho^{*}$} & \oldtext{value$^{*}$} & \oldtext{$\rho_{\mathrm{span}}$} & \oldtext{value$_{\mathrm{span}}$} & \oldtext{raw regret} & \oldtext{norm. regret} & \oldtext{$\log_{10}$ dist.} \\
\midrule
\multicolumn{8}{l}{\oldtext{\textit{Panel A: held-out evaluation}}} \\
\addlinespace[2pt]
\oldtext{$R^2_P$} & \oldtext{3.393} & \oldtext{0.901} & \oldtext{1.099} & \oldtext{0.899} & \oldtext{0.0018} & \oldtext{0.056} & \oldtext{0.490} \\
\oldtext{$\operatorname{MAE}_P$} & \oldtext{1.931} & \oldtext{\$73,924} & \oldtext{0.829} & \oldtext{\$74,409} & \oldtext{\$485} & \oldtext{0.044} & \oldtext{0.245} \\
\oldtext{$\operatorname{MAPE}_P$} & \oldtext{1.265} & \oldtext{21.0\%} & \oldtext{0.829} & \oldtext{21.1\%} & \oldtext{0.033 pp} & \oldtext{0.011} & \oldtext{0.061} \\
\oldtext{$\operatorname{RMSE}_{\log P}$} & \oldtext{0.720} & \oldtext{0.289} & \oldtext{0.720} & \oldtext{0.289} & \oldtext{0} & \oldtext{0} & \oldtext{0} \\
\oldtext{COD} & \oldtext{1.265} & \oldtext{21.52} & \oldtext{1.099} & \oldtext{21.53} & \oldtext{0.015} & \oldtext{0.005} & \oldtext{0.061} \\
\midrule
\multicolumn{8}{l}{\oldtext{\textit{Panel B: 2025 forward evaluation}}} \\
\addlinespace[2pt]
\oldtext{$R^2_P$} & \oldtext{1.931} & \oldtext{0.912} & \oldtext{1.099} & \oldtext{0.910} & \oldtext{0.0014} & \oldtext{0.037} & \oldtext{0.245} \\
\oldtext{$\operatorname{MAE}_P$} & \oldtext{2.560} & \oldtext{\$76,558} & \oldtext{1.099} & \oldtext{\$76,848} & \oldtext{\$289} & \oldtext{0.023} & \oldtext{0.367} \\
\oldtext{$\operatorname{MAPE}_P$} & \oldtext{2.223} & \oldtext{20.6\%} & \oldtext{0.309} & \oldtext{20.6\%} & \oldtext{0.052 pp} & \oldtext{0.020} & \oldtext{0.306} \\
\oldtext{$\operatorname{RMSE}_{\log P}$} & \oldtext{1.456} & \oldtext{0.278} & \oldtext{0.625} & \oldtext{0.278} & \oldtext{0.0001} & \oldtext{0.003} & \oldtext{0.122} \\
\oldtext{COD} & \oldtext{2.223} & \oldtext{21.05} & \oldtext{0.309} & \oldtext{21.11} & \oldtext{0.062} & \oldtext{0.027} & \oldtext{0.306} \\
\bottomrule
\end{tabular}}
\vspace{1mm}
\begin{minipage}{\textwidth}
\scriptsize
\oldtext{\emph{Notes.} This Direct-only table belongs to the v1 span analysis. Its values are not carried forward as final v2 evidence because the final augmented experiment gives both families valid CV spans and therefore requires the regret diagnostic to be regenerated symmetrically.}
\end{minipage}
```

## Inert `%`-commented superseded passages (retained in source, never rendered)

- L118 · near `(before first label)` — `overall assessment level`
- L143 · near `sec:introduction` — `A central quantity in these studies is the ratio of a valuation estimate to a market-value proxy such as sale price. Because the CCAO model studied he`
- L173 · near `sec:introduction` — `overall assessment level`
- L176 · near `sec:introduction` — `For a later deployment decision, chronological validation can first be used to derive a family-specific \emph{candidate region} that screens the posit`
- L182 · near `sec:introduction` — `This draft reports the complete Direct and Surrogate regularization paths`
- L196 · near `sec:introduction` — `Our contribution is application-driven. Motivated by the regressive pattern in the controlled CCAO baseline comparison, we study whether that pattern `
- L198 · near `sec:introduction` — `Our contribution is application-driven and has three parts. First, we translate an assessor-facing regressivity pattern into a trainable first-order o`
- L205 · near `sec:introduction` — `We report complete regularization paths rather than a selected penalty.`
- L208 · near `sec:introduction` — `The remainder of this section reviews the related literature on machine learning in mass appraisal, vertical equity, and dependence-based fair regress`
- L279 · near `subsec:related_work` — `Although much of the fairness-in-ML literature initially focused on classification, \emph{fair regression} studies how fairness requirements can be in`
- L304 · near `sec:setting` — `This section fixes the CCAO application setting, prediction notation, and evaluation framework used throughout the paper. We first describe the reside`
- L430 · near `eq:boosted_tree_model` — `Predictive and Assessor-Facing Evaluation Criteria`
- L668 · near `tab:assessment_metrics_summary` — `\textbf{\shortstack{Reference range}}`
- L951 · near `tab:assessment_metrics_summary` — `\textbf{\shortstack{Reference range}}`
- L1057 · near `tab:assessment_metrics_summary` — ```--'' denotes that no universal ideal or reference range is imposed.`
- L1084 · near `tab:assessment_metrics_summary` — `Supplemental diagnostics.`
- L1217 · near `subsec:baseline_tension` — `The previous baseline presentation combined prediction, valuation level, horizontal uniformity, vertical equity, and mechanism/residual-structure diag`
- L1218 · near `subsec:baseline_tension` — `For readability and to keep the empirical hierarchy explicit, the baseline comparison is split into a primary table for prediction and assessor-facing`
- L1322 · near `fig:baseline_motivation` — `Table~\ref{tab:ccao_baseline_results} and Figure~\ref{fig:baseline_motivation} apply the evaluation framework in Section~\ref{subsec:metrics} to the t`
- L1322 · near `fig:baseline_motivation` — `Tables~\ref{tab:ccao_baseline_results} and~\ref{tab:ccao_baseline_complementary}, together with Figure~\ref{fig:baseline_motivation}, apply the evalua`
- L1490 · near `eq:direct_penalty_gradient` — `overall assessment level`
- L1493 · near `eq:direct_penalty_gradient` — `overall assessment level`
- L1591 · near `eq:surrogate_decomposition` — `Hence, the surrogate controls squared covariance together with the dispersion of the observation-level residual--price contributions. It is therefore `
- L1640 · near `eq:surrogate_derivatives` — `Both quantities depend only on observation $i$ once $\bar y$ has been computed. They can therefore be supplied through a standard second-order boostin`
- L1656 · near `eq:surrogate_derivatives` — `overall assessment level`
- L1667 · near `subsec:correction_scope` — `The two formulations intervene differently. The direct penalty responds to the aggregate signed first-order association between log valuation ratios a`
- L1672 · near `subsec:correction_scope` — `Neither formulation directly optimizes PRD, PRB, VEI, COD, or assessment level. Nor does zero training covariance imply the absence of nonlinear, geog`
- L1685 · near `eq:normalized_rho` — `Section~\ref{sec:empirical_design} specifies the chronological validation and regularization-path procedure used in the empirical application. Appendi`
- L1686 · near `eq:normalized_rho` — `Section~\ref{sec:empirical_design} specifies the chronological validation and regularization-path procedure used in the empirical application. Appendi`
- L1871 · near `subsec:ccao_design` — `frozen Section-2 LightGBM hyperparameter configuration`
- L1873 · near `subsec:ccao_design` — `This design isolates the practical effect of changing the objective while preserving the surrounding valuation workflow.`
- L1874 · near `subsec:ccao_design` — `Holding the surrounding model specification fixed makes the training-objective change as controlled as possible; the separate $\rho=0$ implementation `
- L1877 · near `subsec:ccao_design` — `The CCAO application uses verified Cook County residential sales from the CCAO residential modeling data used for the final evaluation. The residentia`
- L1884 · near `subsec:ccao_design` — `The sales universe is 2016--2024. The current training file contains 2016 sales, so the first observed sale is on 2016-01-01. The oldest 90\% of eligi`
- L1886 · near `subsec:ccao_design` — `Neither out-of-time sample is used to alter the baseline configuration, penalty grid, objective definitions, or training design.`
- L1902 · near `tab:ccao_samples` — `Primary held-out test`
- L1905 · near `tab:ccao_samples` — `Production training`
- L1911 · near `tab:ccao_samples` — `\footnotesize The current training file contains 2016 sales; observed sales begin on 2016-01-01.`
- L1954 · near `subsec:path_design` — `The exact seven-fold date, validation-rule, and sample-size table was displayed in the main text.`
- L1959 · near `subsec:path_design` — `The candidate set contains standard LightGBM, a custom-objective implementation control with $\rho=0$, and the direct and sample-additive covariance-p`
- L1961 · near `subsec:path_design` — `For each positive-penalty family, we evaluate 50 geometrically spaced values between $0.1$ and $100$.`
- L1972 · near `subsec:path_design` — `No $\rho$, model family, or penalized configuration is selected in the present analysis.`
- L1978 · near `subsec:path_design` — `For compact main-text displays only, we use a small set of \emph{prespecified display anchors}: $\rho=0$ together with the positive grid points neares`
- L1979 · near `subsec:path_design` — `For compact main-text tables, we use the positive grid points nearest $0.1$, $1$, $10$, and $100$ as prespecified display anchors and show ordinary Li`
- L2004 · near `subsec:path_design` — `CCAO application and pre-selection regularization-path design.`
- L2031 · near `subsec:comparators` — `The empirical comparison separates the baseline model class, the custom-objective implementation path, positive covariance regularization, and a cente`
- L2039 · near `subsec:comparators` — `The direct penalty tests the stated covariance target. The surrogate tests the implementation recommended for standard boosting software.`
- L2042 · near `subsec:comparators` — `The final comparison also includes a custom-objective fit at $\rho=0$. This implementation control separates changes caused by the custom-objective co`
- L2044 · near `subsec:comparators` — `\textbf{Post-hoc rescaling.} The previous draft included a validation-fitted centered rescaling of the baseline predictions as a primary and multi-cou`
- L2066 · near `subsec:attom_design` — `For each county, eligible recorder transfers are joined to the most recent assessor-history record available before the sale, together with lagged con`
- L2068 · near `subsec:attom_design` — `For each county, eligible recorder transfers are joined to the most recent assessor-history record available before the sale, together with lagged con`
- L2069 · near `subsec:attom_design` — `For each county, eligible recorder transfers are joined to the most recent assessor-history record available before the sale, together with lagged con`
- L2081 · near `sec:results` — `The primary CCAO Results below report three additions to the complete pre-selection analysis: the split-specific $\rho=0$ implementation-control audit`
- L2083 · near `sec:results` — `The primary CCAO Results report the complete Direct and Surrogate regularization paths together with three complementary checks: the $\rho=0$ implemen`
- L2084 · near `sec:results` — `The primary CCAO Results report the complete Direct and Surrogate regularization paths together with two complementary checks: the $\rho=0$ implementa`
- L2085 · near `sec:results` — `The primary CCAO Results report the complete Direct and Surrogate regularization paths together with the fixed cross-fitted correlation-ratio nonlinea`
- L2086 · near `sec:results` — `The primary CCAO Results report the complete Direct and Surrogate regularization paths together with the fixed cross-fitted correlation-ratio nonlinea`
- L2088 · near `sec:results` — `We also report the frozen five-metric transition diagnostic from Section~\ref{subsec:path_design}, including fold and leave-one-fold-out stability, re`
- L2089 · near `sec:results` — `We also report the frozen five-metric transition diagnostic from Section~\ref{subsec:path_design}, including fold and leave-one-fold-out stability and`
- L2105 · near `subsec:ccao_baseline_results` — `A dedicated main-text subsection and table previously reported the $\rho=0$ native/custom implementation check.`
- L2110 · near `subsec:ccao_baseline_results` — `What the Prespecified Regularization Paths Do`
- L2114 · near `subsec:path_results` — `The Direct and Surrogate objectives are evaluated over their complete prespecified grids.`
- L2121 · near `subsec:path_results` — `Table~\ref{tab:path_anchor_summary} is the compact main-text view. It uses only the prespecified display anchors defined in Section~\ref{subsec:path_d`
- L2122 · near `subsec:path_results` — `Table~\ref{tab:path_anchor_summary} is the compact main-text view of prediction and assessor-facing vertical equity at the fixed display anchors defin`
- L2574 · near `tab:path_anchor_summary` — `An asterisk indicates that the reported point estimate falls within the applicable reference range.`
- L2593 · near `tab:path_anchor_summary` — `Primary regularization-path summary at prespecified display anchors: prediction and assessor-facing vertical equity. Rows are fixed for display and do`
- L2617 · near `tab:path_anchor_summary` — `Direct`
- L2617 · near `tab:path_anchor_summary` — `$\approx0.01$`
- L2617 · near `tab:path_anchor_summary` — `\textit{pending canonical v2 extraction}`
- L2619 · near `tab:path_anchor_summary` — `$\approx0.1$`
- L2619 · near `tab:path_anchor_summary` — `\textit{source}`
- L2619 · near `tab:path_anchor_summary` — `\textit{source}`
- L2620 · near `tab:path_anchor_summary` — `$\approx0.954$`
- L2620 · near `tab:path_anchor_summary` — `\textit{source}`
- L2620 · near `tab:path_anchor_summary` — `\textit{source}`
- L2621 · near `tab:path_anchor_summary` — `$\approx10.481$`
- L2621 · near `tab:path_anchor_summary` — `\textit{source}`
- L2621 · near `tab:path_anchor_summary` — `\textit{source}`
- L2622 · near `tab:path_anchor_summary` — `\textit{source}`
- L2622 · near `tab:path_anchor_summary` — `\textit{source}`
- L2626 · near `tab:path_anchor_summary` — `Surrogate`
- L2626 · near `tab:path_anchor_summary` — `$\approx0.01$`
- L2626 · near `tab:path_anchor_summary` — `\textit{pending canonical v2 extraction}`
- L2628 · near `tab:path_anchor_summary` — `$\approx0.1$`
- L2628 · near `tab:path_anchor_summary` — `\textit{source}`
- L2628 · near `tab:path_anchor_summary` — `\textit{source}`
- L2629 · near `tab:path_anchor_summary` — `$\approx0.954$`
- L2629 · near `tab:path_anchor_summary` — `\textit{source}`
- L2629 · near `tab:path_anchor_summary` — `\textit{source}`
- L2630 · near `tab:path_anchor_summary` — `$\approx10.481$`
- L2630 · near `tab:path_anchor_summary` — `\textit{source}`
- L2630 · near `tab:path_anchor_summary` — `\textit{source}`
- L2631 · near `tab:path_anchor_summary` — `\textit{source}`
- L2631 · near `tab:path_anchor_summary` — `\textit{source}`
- L2643 · near `tab:path_anchor_summary` — `Direct`
- L2643 · near `tab:path_anchor_summary` — `$\approx0.01$`
- L2643 · near `tab:path_anchor_summary` — `\textit{pending canonical v2 extraction}`
- L2645 · near `tab:path_anchor_summary` — `$\approx0.1$`
- L2645 · near `tab:path_anchor_summary` — `\textit{source}`
- L2645 · near `tab:path_anchor_summary` — `\textit{source}`
- L2646 · near `tab:path_anchor_summary` — `$\approx0.954$`
- L2646 · near `tab:path_anchor_summary` — `\textit{source}`
- L2646 · near `tab:path_anchor_summary` — `\textit{source}`
- L2647 · near `tab:path_anchor_summary` — `$\approx10.481$`
- L2647 · near `tab:path_anchor_summary` — `\textit{source}`
- L2647 · near `tab:path_anchor_summary` — `\textit{source}`
- L2648 · near `tab:path_anchor_summary` — `\textit{source}`
- L2648 · near `tab:path_anchor_summary` — `\textit{source}`
- L2652 · near `tab:path_anchor_summary` — `Surrogate`
- L2652 · near `tab:path_anchor_summary` — `$\approx0.01$`
- L2652 · near `tab:path_anchor_summary` — `\textit{pending canonical v2 extraction}`
- L2654 · near `tab:path_anchor_summary` — `$\approx0.1$`
- L2654 · near `tab:path_anchor_summary` — `\textit{source}`
- L2654 · near `tab:path_anchor_summary` — `\textit{source}`
- L2655 · near `tab:path_anchor_summary` — `$\approx0.954$`
- L2655 · near `tab:path_anchor_summary` — `\textit{source}`
- L2655 · near `tab:path_anchor_summary` — `\textit{source}`
- L2656 · near `tab:path_anchor_summary` — `$\approx10.481$`
- L2656 · near `tab:path_anchor_summary` — `\textit{source}`
- L2656 · near `tab:path_anchor_summary` — `\textit{source}`
- L2657 · near `tab:path_anchor_summary` — `\textit{source}`
- L2657 · near `tab:path_anchor_summary` — `\textit{source}`
- L2664 · near `tab:path_anchor_summary` — `Display anchors are fixed presentation anchors and are not selected on the basis of observed performance. Positive $\rho$ values are shown at the corr`
- L2748 · near `tab:path_anchor_summary` — `Given the current implementation audit, within-family path changes are read relative to the custom-objective $\rho=0$ control. At the prespecified anc`
- L2749 · near `tab:path_anchor_summary` — `At the prespecified anchor near $\rho=0.954$, the Direct path has held-out $R^2_P=0.899$ and MAE of \$74,485, compared with $0.894$ and \$75,655 for o`
- L2752 · near `tab:path_anchor_summary` — `Thus the empirical paths are not a simple monotone exchange of prediction accuracy for vertical equity; at stronger penalties, however, price-scale ac`
- L2802 · near `fig:ratio_shape_path_placeholder` — `The Direct path modifies the shape comparatively smoothly.`
- L2806 · near `fig:ratio_shape_path_placeholder` — `Consequently, a near-neutral global slope or grouped high-versus-low statistic need not imply a locally flat valuation-ratio profile.`
- L2814 · near `fig:ratio_shape_path_placeholder` — `The Correction Acts on the Intended First-Order Mechanism`
- L2819 · near `subsec:mechanism_results` — `The direct training target is covariance, and $\beta_{\log}$ is its signed first-order slope equivalent up to the fixed variance of log sale price wit`
- L2835 · near `fig:mechanism_path_placeholder` — `The nonlinear dependence diagnostic is not monotone, however.`
- L2840 · near `fig:mechanism_path_placeholder` — `This behavior is consistent with the non-monotone ratio shapes above and shows that reducing the targeted first-order association does not eliminate b`
- L2852 · near `subsec:tradeoff_results` — `The previous headline accuracy--equity figure overlaid a centered-recalibration path with the Direct and Surrogate paths.`
- L2860 · near `fig:accuracy_equity_placeholder` — `At stronger regularization, the vertical-equity diagnostics continue toward neutrality while $R^2_P$ bends downward.`
- L2862 · near `fig:accuracy_equity_placeholder` — `the relevant pre-selection empirical object`
- L2907 · near `subsec:attom_results` — `Table~\ref{tab:attom_baselines} shows the unpenalized ATTOM benchmark results. Every county has at least one vertical-equity point estimate outside th`
- L2934 · near `tab:attom_baselines` — `Among the four counties whose baseline PRB point estimate is outside the benchmark band, at least one tested penalty places PRB inside the band in all`
- L2965 · near `tab:attom_selected_penalty` — `These results do not establish a universal or confirmatory compliance result and do not imply that the selected penalty dominates simpler post-hoc rec`
- L2968 · near `tab:attom_selected_penalty` — `The previous draft included a separate primary CCAO centered-recalibration results subsection, endpoint table, and overlays in the accuracy--equity fi`
- L2976 · near `sec:discussion` — `The CCAO application answers a limited question. The regenerated baseline comparison establishes a clear application-specific tension: relative to lin`
- L2984 · near `sec:discussion` — `Second, stronger regularization exposes the limits of a global first-order target. The Surrogate can drive $\beta_{\log}$ and grouped diagnostics clos`
- L2993 · near `sec:discussion` — `The value of the approach is operational. It converts a ratio-study concern into a model-training option, works through a standard boosted-tree interf`
- L2996 · near `sec:discussion` — `The previous version mixed the scope limitations, the native/custom implementation caveat, ATTOM external-validity caveats, and the centered-recalibra`
- L2998 · near `sec:discussion` — `The evidence also bounds the paper's claim. Better data can improve predictive accuracy and assessment equity together. The proposed penalties target `
- L3005 · near `sec:discussion` — `Assessor Implementation`
- L3009 · near `sec:implementation_workflow` — `The method can be inserted into an existing AVM with the following sequence.`
- L3036 · near `tab:roles` — `One-parameter slope/spread rescaling`
- L3036 · near `tab:roles` — `Low-cost post-hoc correction of a global value trend; benchmark for the penalty`
- L3038 · near `tab:roles` — `Stratified and local diagnostics`
- L3038 · near `tab:roles` — `Detect offsetting geographic, class-specific, or nonlinear patterns hidden by aggregate measures`
- L3039 · near `tab:roles` — `$\Delta_{\mathrm{NL}}$, ratio-shape, and stratified/local diagnostics`
- L3039 · near `tab:roles` — `Separate nonlinear conditional-mean departures from the first-order trend and detect offsetting geographic or class-specific patterns hidden by aggreg`
- L3050 · near `tab:roles` — `First, the CCAO application uses sold properties. Results on sales do not establish performance for the full unsold inventory, particularly when the s`
- L3055 · near `tab:roles` — `Second, the correction is deliberately narrow. Direct targets a global log-scale covariance, while the Surrogate penalizes a sample-additive upper bou`
- L3059 · near `tab:roles` — `Third, the present analysis characterizes regularization paths rather than a deployment decision. Fold-level heterogeneity and group-level uncertainty`
- L3068 · near `tab:roles` — `Preliminary tests of second- and third-degree global recalibration have shown little consistent improvement over a first-degree map, so higher-order c`
- L3092 · near `app:metrics` — `pre-selection`
- L3621 · near `app:implementation` — `This makes the custom-objective starting prediction match the native squared-error LightGBM initialization while leaving $e_i=f(x_i)-y_i$ and $c_i=y_i`
- L3745 · near `tab:rho_zero_control` — `Primary CCAO Pre-Selection Path Details`
- L3756 · near `app:ccao_path_details` — `The complete pre-selection path is provided`
- L3812 · near `tab:ccao_baseline_complementary` — `Complementary regularization-path summary at the same prespecified display anchors: valuation level, horizontal uniformity, and mechanism/residual str`
- L3836 · near `tab:path_anchor_complementary` — `Direct`
- L3836 · near `tab:path_anchor_complementary` — `$\approx0.01$`
- L3836 · near `tab:path_anchor_complementary` — `\textit{pending canonical v2 extraction}`
- L3838 · near `tab:path_anchor_complementary` — `$\approx0.1$`
- L3838 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3838 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3838 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3838 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3838 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3839 · near `tab:path_anchor_complementary` — `$\approx0.954$`
- L3839 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3839 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3839 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3839 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3839 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3840 · near `tab:path_anchor_complementary` — `$\approx10.481$`
- L3840 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3840 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3840 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3840 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3840 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3841 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3841 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3841 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3841 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3841 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3845 · near `tab:path_anchor_complementary` — `Surrogate`
- L3845 · near `tab:path_anchor_complementary` — `$\approx0.01$`
- L3845 · near `tab:path_anchor_complementary` — `\textit{pending canonical v2 extraction}`
- L3847 · near `tab:path_anchor_complementary` — `$\approx0.1$`
- L3847 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3847 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3847 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3847 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3847 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3848 · near `tab:path_anchor_complementary` — `$\approx0.954$`
- L3848 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3848 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3848 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3848 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3848 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3849 · near `tab:path_anchor_complementary` — `$\approx10.481$`
- L3849 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3849 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3849 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3849 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3849 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3850 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3850 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3850 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3850 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3850 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3862 · near `tab:path_anchor_complementary` — `Direct`
- L3862 · near `tab:path_anchor_complementary` — `$\approx0.01$`
- L3862 · near `tab:path_anchor_complementary` — `\textit{pending canonical v2 extraction}`
- L3864 · near `tab:path_anchor_complementary` — `$\approx0.1$`
- L3864 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3864 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3864 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3864 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3864 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3865 · near `tab:path_anchor_complementary` — `$\approx0.954$`
- L3865 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3865 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3865 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3865 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3865 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3866 · near `tab:path_anchor_complementary` — `$\approx10.481$`
- L3866 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3866 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3866 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3866 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3866 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3867 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3867 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3867 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3867 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3867 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3871 · near `tab:path_anchor_complementary` — `Surrogate`
- L3871 · near `tab:path_anchor_complementary` — `$\approx0.01$`
- L3871 · near `tab:path_anchor_complementary` — `\textit{pending canonical v2 extraction}`
- L3873 · near `tab:path_anchor_complementary` — `$\approx0.1$`
- L3873 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3873 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3873 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3873 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3873 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3874 · near `tab:path_anchor_complementary` — `$\approx0.954$`
- L3874 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3874 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3874 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3874 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3874 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3875 · near `tab:path_anchor_complementary` — `$\approx10.481$`
- L3875 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3875 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3875 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3875 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3875 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3876 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3876 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3876 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3876 · near `tab:path_anchor_complementary` — `\textit{source}`
- L3876 · near `tab:path_anchor_complementary` — `\textit{source}`
- L4288 · near `app:ccao_results` — `A replication package for the CCAO application should contain the complete penalty grid; fold-level, held-out, and forward predictions for every valid`
- L4299 · near `app:attom` — `Exploratory ATTOM Benchmark and Recalibration`
- L4303 · near `app:attom` — `The exploratory ATTOM benchmark uses single-family recorder transfers that pass the benchmark price and data-quality filters. Each transfer is joined `
- L4338 · near `tab:attom_design_appendix` — `The benchmark compares the ordinary LightGBM fit, the custom-objective zero-penalty control, positive covariance-penalty values, and centered recalibr`
- L4340 · near `tab:attom_design_appendix` — `The benchmark compares ordinary LightGBM, the custom-objective zero-penalty control, positive \texttt{LGBCovPenalty[diff]} values, and the centered on`
- L4341 · near `tab:attom_design_appendix` — `The benchmark compares ordinary LightGBM, the custom-objective zero-penalty control, and positive \texttt{LGBCovPenalty[diff]} values. County-specific`
- L4344 · near `tab:attom_design_appendix` — `A confirmatory version must freeze the feature set, LightGBM settings, penalty grid, recalibration family, and selection rule before opening any count`
- L4345 · near `tab:attom_design_appendix` — `assessment level`
- L1265 · near `tab:ccao_baseline_results` — `% \noindent\textbf{Previous main-text complementary baseline table.}\par\smallskip % \centering % \scriptsize % \setlength{\tabcolsep}{2.4pt} % \renew`
- L2502 · near `tab:path_anchor_summary` — `% \noindent\textbf{Previous combined representative-$\rho$ table.}\par\smallskip % \centering % \scriptsize % \renewcommand{\arraystretch}{1.08} % \re`
