# P0-2 — Native vs Custom rho=0 Parity (Stage-1 portion)

**Binding rule (plan A1 / F.0), stated verbatim:**

> A pinned Track-P equivalence result may never be cited as validating the historical
> positive-rho path artifacts. Track P characterises the implementation prospectively;
> only Track H speaks to the frozen artifacts.

Cell names are fixed strings and are used verbatim throughout:
**Cell A = "Ordinary LightGBM (standard raw-label native)"**, **Cell B = "Parity-aligned native L2"**, **Cell C = "Custom rho=0 origin"**.
Cell B is never printed as "Ordinary LightGBM". Cell C is never substituted for Cell B.

Parity tiers **T1–T4** (A/B/C comparisons) and reproduction tiers **R1–R4** (frozen-artifact
reproduction) are distinct concepts, live in separate tables and separate fields, and are never
combined into one verdict.

---

## 1. Source equivalence (must precede any interpretation of reproduction) — plan F.2b

* All three artifact-generating commits are ancestors of HEAD: **True**
* Executed-path source drift detected: **True** — in
  `run_temporal_cv.py, utils/delta_nl.py`
* A temporary detached provenance worktree was created for all three commits, byte-compared, and removed.

**`soft_constrained_models/boosting_models.py`, `utils/motivation_utils.py`,
`canonical_experiment.py`, `params.yaml`, `cv_config.yaml` and `model_params.yaml` are
BYTE-IDENTICAL at every provenance commit** (True).
The objective, split and metric machinery has not changed since the artifacts were generated.

Resolution of the two drifted files:

* **`run_temporal_cv.py`** — additive `--no-explicit-zero` feature only.
  `_finalize_rho_values(v, explicit_zero=True)` was proven **bitwise identical** to
  `_prepend_explicit_zero(v)` on 8 test inputs including the exact frozen 82-, 50- and 32-point
  grids. The flag defaults OFF. No model factory, objective, loader, split, fit/predict or metric
  code changed.
* **`utils/delta_nl.py`** — recorded as untracked (`?? utils/delta_nl.py`) in the `d3ef45f2`
  artifact's own `status_porcelain`, so the diff is untracked-then-committed-later, not a change.
  It consumes predictions and takes no part in training or prediction.

**Net effect on prediction reproduction: none.**

The irreducible dirty-state limitation is recorded in `provenance/DIRTY_STATE_LIMITATION.md`.

---

## 2. Frozen-artifact reproduction — Track H, tiers R1–R4

Seven frozen configurations re-fitted under historical settings and compared to their cached
predictions on both out-of-time blocks.

| label | rho | config_id | split | n | mean_abs_delta_log | max_abs_delta_log | frac_exact_equal | reproduction_tier | failure_class |
|---|---|---|---|---|---|---|---|---|---|
| cellA_native | nan | 252a25d9c0ce796b | heldout | 38290 | 0 | 0 | 1 | R1 | none |
| direct_rho0 | 0 | 1fb838f7d6bfda88 | heldout | 38290 | 0 | 0 | 1 | R1 | none |
| surrogate_rho0 | 0 | 5b7875e55e58ac62 | heldout | 38290 | 0 | 0 | 1 | R1 | none |
| direct_rho_0p954095 | 0.9541 | e732a8e35bd1a796 | heldout | 38290 | 0 | 0 | 1 | R1 | none |
| surrogate_rho_0p954095 | 0.9541 | 4a39ef84979943f2 | heldout | 38290 | 0 | 0 | 1 | R1 | none |
| direct_rho100 | 100 | a72f544c5161823f | heldout | 38290 | 0 | 0 | 1 | R1 | none |
| surrogate_rho100 | 100 | d5704816049673ad | heldout | 38290 | 0 | 0 | 1 | R1 | none |
| cellA_native | nan | 252a25d9c0ce796b | forward_2025 | 26641 | 0 | 0 | 1 | R1 | none |
| direct_rho0 | 0 | 1fb838f7d6bfda88 | forward_2025 | 26641 | 0 | 0 | 1 | R1 | none |
| surrogate_rho0 | 0 | 5b7875e55e58ac62 | forward_2025 | 26641 | 0 | 0 | 1 | R1 | none |
| direct_rho_0p954095 | 0.9541 | e732a8e35bd1a796 | forward_2025 | 26641 | 0 | 0 | 1 | R1 | none |
| surrogate_rho_0p954095 | 0.9541 | 4a39ef84979943f2 | forward_2025 | 26641 | 0 | 0 | 1 | R1 | none |
| direct_rho100 | 100 | a72f544c5161823f | forward_2025 | 26641 | 0 | 0 | 1 | R1 | none |
| surrogate_rho100 | 100 | d5704816049673ad | forward_2025 | 26641 | 0 | 0 | 1 | R1 | none |


> ### All 14 configurations reproduce at **R1 — exact**.
> `max|delta| = 0.0`, 100 % of rows bitwise identical, all metrics agree at displayed precision.

Consequences: the frozen positive-rho artifacts are bit-reproducible at HEAD; no residual is left
to attribute; **no failure class is required** (all `none`); and **RG-3 does not fire**.

---

## 3. Same-host reproducibility (F-NUM floor) — Track H

| split | max_abs_delta_log | mean_abs_delta_log | frac_exact_equal |
|---|---|---|---|
| heldout | 0 | 0 | 1 |
| forward_2025 | 0 | 0 | 1 |

Cell A fitted twice under identical historical settings on one host: **max|delta| = 0.0**, 100.0000 % of rows exactly equal. (source: `tables/fnum_same_host_replicate.csv`)

> **The F-NUM floor is exactly zero.** Run-to-run numerical nondeterminism does not exist for this pipeline on this hardware, so no observed difference anywhere in this report may be attributed to it.

---

## 4. Track H parity ladder — historical settings

Cells A, B, C at three capacities (**T** = 60 trees/15 leaves/depth 4, **M** = 200/63/8,
**F** = the frozen 994/573/11), on both out-of-time blocks. All other parameters are the frozen
vector; no determinism pins.

| capacity | split | cell_pair | mean_abs_delta_log | p95_abs_delta_log | max_abs_delta_log | pearson | metrics_agree_at_displayed_precision | parity_tier |
|---|---|---|---|---|---|---|---|---|
| T | forward_2025 | A<->B | 7.0637e-09 | 1.6677e-08 | 2.2196e-08 | 1 | True | T2 |
| T | forward_2025 | A<->C | 0.016447 | 0.043276 | 0.1195 | 0.99922 | False | T4 |
| T | forward_2025 | B<->C | 0.016447 | 0.043276 | 0.1195 | 0.99922 | False | T4 |
| T | forward_2025 | C<->C_surrogate | 0 | 0 | 0 | 1 | True | T1 |
| M | forward_2025 | A<->B | 1.1499e-08 | 2.7737e-08 | 5.377e-08 | 1 | True | T2 |
| M | forward_2025 | A<->C | 0.018666 | 0.049718 | 0.20216 | 0.99924 | False | T4 |
| M | forward_2025 | B<->C | 0.018666 | 0.049718 | 0.20216 | 0.99924 | False | T4 |
| M | forward_2025 | C<->C_surrogate | 0 | 0 | 0 | 1 | True | T1 |
| F | forward_2025 | A<->B | 1.0143e-05 | 5.1466e-08 | 0.011989 | 1 | False | T4 |
| F | forward_2025 | A<->C | 0.030563 | 0.086554 | 0.28252 | 0.99796 | False | T4 |
| F | forward_2025 | B<->C | 0.030562 | 0.086554 | 0.28252 | 0.99796 | False | T4 |
| F | forward_2025 | C<->C_surrogate | 0 | 0 | 0 | 1 | True | T1 |
| T | heldout | A<->B | 7.6893e-09 | 1.741e-08 | 2.583e-08 | 1 | True | T2 |
| T | heldout | A<->C | 0.017626 | 0.046467 | 0.14014 | 0.99906 | False | T4 |
| T | heldout | B<->C | 0.017626 | 0.046467 | 0.14014 | 0.99906 | False | T4 |
| T | heldout | C<->C_surrogate | 0 | 0 | 0 | 1 | True | T1 |
| M | heldout | A<->B | 1.1314e-08 | 2.5596e-08 | 6.9535e-08 | 1 | True | T2 |
| M | heldout | A<->C | 0.017523 | 0.04687 | 0.17475 | 0.99934 | False | T4 |
| M | heldout | B<->C | 0.017523 | 0.04687 | 0.17475 | 0.99934 | False | T4 |
| M | heldout | C<->C_surrogate | 0 | 0 | 0 | 1 | True | T1 |
| F | heldout | A<->B | 0.030322 | 0.086976 | 0.29858 | 0.99797 | False | T4 |
| F | heldout | A<->C | 0.03239 | 0.090955 | 0.30894 | 0.99774 | False | T4 |
| F | heldout | B<->C | 0.032594 | 0.093221 | 0.29705 | 0.99768 | False | T4 |
| F | heldout | C<->C_surrogate | 0 | 0 | 0 | 1 | True | T1 |


### Structure of the result

| comparison | T | M | F |
|---|---|---|---|
| A<->B | T2 | T2 | T4 |
| A<->C | T4 | T4 | T4 |
| B<->C | T4 | T4 | T4 |
| C Direct <-> C Surrogate | **T1** | **T1** | **T1** |

**Two separable mechanisms, both LightGBM-level, neither a defect in the paper's objective:**

1. **Feature-subsampling RNG stream (present at every capacity).** `B<->C` is T4 even at
   capacity T, where `A<->B` is a clean T2. Since Cells B and C are fed identical labels with
   identical initialisation, label representation cannot explain it.
2. **Float32 label representation amplified by capacity (present only at F).** `A<->B` is T2 at
   T and M (6.95e-08 max) but
   T4 at F (0.2986 max).

**Direct and Surrogate at rho=0 are bitwise identical (T1) at every capacity and split**,
confirming the frozen audit's finding independently.

### Exact match to the frozen artifact

`A<->C` at capacity F reproduces the frozen `tab:rho_zero_control` values to every digit:

| split | frozen mean\|d\| | refit mean\|d\| | frozen max\|d\| | refit max\|d\| | frozen Pearson | refit Pearson |
|---|---|---|---|---|---|---|
| heldout | 0.03238952394 | 0.03238952394 | 0.3089424505 | 0.3089424505 | 0.9977384232 | 0.9977384232 |
| forward_2025 | 0.03056298285 | 0.03056298285 | 0.2825197364 | 0.2825197364 | 0.997960774 | 0.997960774 |

---

## 5. Track P parity ladder

> ### PINNED IMPLEMENTATION DIAGNOSTIC — NOT HISTORICAL EVIDENCE
> This section characterises the implementation prospectively. It may not be cited as validating
> any historical positive-rho artifact, and it sets no regeneration trigger.

Same ladder with `deterministic=True`, `force_row_wise=True`, `num_threads=1` on **every** cell.

| capacity | split | cell_pair | mean_abs_delta_log | max_abs_delta_log | parity_tier |
|---|---|---|---|---|---|
| T | forward_2025 | A<->B | 7.0637e-09 | 2.2196e-08 | T2 |
| T | forward_2025 | A<->C | 0.016447 | 0.1195 | T4 |
| T | forward_2025 | B<->C | 0.016447 | 0.1195 | T4 |
| T | forward_2025 | C<->C_surrogate | 0 | 0 | T1 |
| M | forward_2025 | A<->B | 1.1499e-08 | 5.377e-08 | T2 |
| M | forward_2025 | A<->C | 0.018666 | 0.20216 | T4 |
| M | forward_2025 | B<->C | 0.018666 | 0.20216 | T4 |
| M | forward_2025 | C<->C_surrogate | 0 | 0 | T1 |
| F | forward_2025 | A<->B | 1.0143e-05 | 0.011989 | T4 |
| F | forward_2025 | A<->C | 0.030563 | 0.28252 | T4 |
| F | forward_2025 | B<->C | 0.030562 | 0.28252 | T4 |
| F | forward_2025 | C<->C_surrogate | 0 | 0 | T1 |
| T | heldout | A<->B | 7.6893e-09 | 2.583e-08 | T2 |
| T | heldout | A<->C | 0.017626 | 0.14014 | T4 |
| T | heldout | B<->C | 0.017626 | 0.14014 | T4 |
| T | heldout | C<->C_surrogate | 0 | 0 | T1 |
| M | heldout | A<->B | 1.1314e-08 | 6.9535e-08 | T2 |
| M | heldout | A<->C | 0.017523 | 0.17475 | T4 |
| M | heldout | B<->C | 0.017523 | 0.17475 | T4 |
| M | heldout | C<->C_surrogate | 0 | 0 | T1 |
| F | heldout | A<->B | 0.030322 | 0.29858 | T4 |
| F | heldout | A<->C | 0.03239 | 0.30894 | T4 |
| F | heldout | B<->C | 0.032594 | 0.29705 | T4 |
| F | heldout | C<->C_surrogate | 0 | 0 | T1 |


**Track P is bit-identical to Track H in all 24 comparisons** (`max|delta|` equal in every
cell: True; tiers equal in every cell: True).

> Determinism pinning changes nothing. **F-ENV and F-NUM are therefore excluded** as causes of the
> native/custom gap.

---

## 6. Root cause of `B<->C`

Prescribed per-iteration tracing (plan Gate G1 stop branch).

**Step 1 — the objective is exact.** At iteration 0 on the real development pool the supplied
derivatives are bit-identical to native L2:
`max |grad_custom - grad_native_L2| = 0.0`, all Hessians exactly 1.

**Step 2 — divergence begins in tree 0.** With bit-identical gradients and Hessians, the very
first tree already has a different split structure (max leaf-value difference
0.0129459) and no tree ever shares a split structure:

| n_trees | mean_abs_delta_log | max_abs_delta_log | tree0_same_split_structure | n_trees_with_identical_splits | n_trees_compared |
|---|---|---|---|---|---|
| 1 | 0.010178 | 0.066944 | False | 0 | 1 |
| 2 | 0.0076733 | 0.09124 | False | 0 | 2 |
| 5 | 0.013235 | 0.10268 | False | 0 | 5 |
| 20 | 0.013842 | 0.11878 | False | 0 | 20 |
| 60 | 0.017626 | 0.14014 | False | 0 | 60 |


**Step 3 — one knob isolates it.** Single-tree fits, varying one setting at a time:

| variant | tree0_identical | max_abs_delta_log | frac_exact_equal | colsample_bytree | deterministic | force_row_wise |
|---|---|---|---|---|---|---|
| V0_frozen_as_is | False | 0.066944 | 0 | 0.54101 | False | False |
| V1_pin_histogram | False | 0.066944 | 0 | 0.54101 | True | True |
| V2_no_feature_subsampling | True | 0 | 1 | 1 | False | False |
| V3_pin_and_no_subsampling | True | 0 | 1 | 1 | True | True |


Pinning the histogram path (V1) changes **nothing**. Setting `colsample_bytree = 1.0` (V2) makes
Cell B and Cell C **bitwise identical — `max|delta| = 0.0`, 100 % of rows exactly equal**.

> ### Named cause
> **LightGBM's per-tree feature-subsampling RNG stream differs between the built-in-objective
> path and the custom-objective path.** With `colsample_bytree = 0.5410105713520937`, the two
> learners draw different random feature subsets from the first tree onward and therefore fit
> different — though equally valid — members of the same model family.
>
> **This is LightGBM library behaviour, not a defect in the paper's objective code.** Given the
> same feature subsets, the custom rho=0 objective reproduces native L2 exactly.

### What this means for the manuscript

The native-to-penalized contrast in `tab:path_anchor_summary` conflates the penalty effect with a
feature-subsampling draw difference of mean 0.03239
(held-out) in log space. That confound is real, reproducible, and now precisely named — which is
exactly what P0-2 was commissioned to establish. **Interpreting it, and deciding the reference
convention, is Gate G2 work and is not part of Stage 1.**

---

## 7. Regeneration triggers (plan F.6) — none fired

| trigger_id | fired | description |
|---|---|---|
| RG-1 | False | Historical B<->C materially non-parity that cannot be attributed to an innocuous numerical effect |
| RG-2 | False | Defect in the custom-objective implementation AS IT STOOD AT THE PROVENANCE COMMIT |
| RG-3 | False | Frozen positive-rho artifacts fail historical reproduction at R4 |
| RG-4 | False | Explicit decision to make the pinned deterministic implementation canonical |


Detailed evidence is in `tables/regeneration_triggers.csv`. Note in particular that **A<->B and
A<->C non-parity can never set a trigger**, by rule; only `B<->C` is implementation-relevant, and
its T4 status is attributed to a named, reproduced, innocuous numerical effect.

**No full positive-rho path regeneration is required on Stage-1 evidence.**
No regeneration job was submitted.
