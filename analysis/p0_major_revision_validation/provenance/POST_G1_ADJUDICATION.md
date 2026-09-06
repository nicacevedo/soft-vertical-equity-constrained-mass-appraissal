# Post-G1 Adjudication — binding for Gate G2

**Status:** frozen. Created after `GATE G1 = PASS` was reviewed and accepted.

This document records a **new post-G1 interpretation**. It does **not** edit
`APPROVED_EXECUTION_PLAN.md` (rev. 3, sha256 `d3b7ae3419660a8f…`), which remains
read-only. Where this adjudication and the rev.3 wording differ, the difference is in
**interpretation of Stage-1 evidence that did not exist when rev.3 was written**, and it is
recorded here explicitly rather than by silently changing the historical plan.

Machine-readable companion: `configs/post_g1_reference_convention.yaml`.
Verified-premise evidence: `tables/post_g1_adjudication_evidence.json`.

---

## 1. Historical positive-rho path

**The historical Direct/Surrogate paths remain scientifically usable.**

Each reason below was recomputed from the Stage-1 artifacts, not asserted:

| # | Premise | Verified | Evidence |
|---|---|---|---|
| P1 | Sampled historical artifacts reproduce exactly at R1 | **True** — 14/14 configs, max\|d\| = 0.0 | `tables/frozen_artifact_reproduction.csv` |
| P2 | Direct/Surrogate objective equations match the executed code; rho=0 derivatives correct | **True** — max abs gradient difference vs native L2 = 0.0, all Hessians exactly 1 | `tables/bc_rootcause_verdict.json + reports/P0_IMPLEMENTATION_AUDIT.md` |
| P3 | Direct rho=0 and Surrogate rho=0 are identical within the custom-objective implementation | **True** — T1 in all 6 capacity x split cells, max\|d\| = 0 | `tables/parity_ladder_historical.csv (C<->C_surrogate)` |
| P4 | Deterministic pinning does not remove the B<->C difference | **True** — Track P max\|d\| 0.2971 equals Track H exactly | `tables/parity_ladder_{historical,pinned}.csv` |
| P5 | Changing `colsample_bytree` would define a DIFFERENT LightGBM specification | **True** — frozen value 0.5410105713520937 was tuned as part of the canonical 994-tree configuration | `tables/bc_discriminate.csv` |
| P6 | The run-to-run numerical floor is zero, so nothing is attributable to noise | **True** — max\|d\| = 0 on both heldout, forward_2025 | `tables/fnum_same_host_replicate.csv` |

> **The B<->C discrepancy does not imply corruption of the positive-rho custom path.**
>
> ### DO NOT regenerate the 82-point path at this stage.
> No full-path regeneration is authorized.

---

## 2. Interpretation of the B<->C discrepancy

The phrase **"innocuous numerical effect" is withdrawn** and must not be used.

### Accepted characterization

> A deterministic LightGBM implementation-level feature-subsampling-path difference between the built-in-objective and custom-objective execution paths under the canonical colsample_bytree < 1 setting. The supplied rho=0 gradients and Hessians are correct, and the discrepancy is not evidence of an error in the covariance objective. However, the resulting prediction difference is material enough that native-vs-custom comparisons cannot be interpreted as isolating the effect of rho.

### Conservative characterization

Stage-1 evidence isolates the *knob* (`colsample_bytree`), the *stage* (tree 0 split structure)
and the *source state* (objective files byte-identical), but it does **not** directly instrument
LightGBM's internal RNG stream. `rng_mechanism_directly_identified_by_stage1_evidence =
False`. Accordingly the
preferred short form is:

> **deterministic built-in-vs-custom feature-subsampling execution-path divergence**

No claim is made about LightGBM internals beyond what the tree/split/source evidence supports.

### The key scientific point

> **A or B versus positive-rho custom models is not a pure penalty-effect contrast.**

Measured magnitudes at the canonical 994-tree capacity (log scale):

| contrast | held-out mean \| max | 2025 mean \| max | tier |
|---|---|---|---|
| A <-> C | 0.03239 \| 0.3089 | 0.03056 \| 0.2825 | T4 |
| A <-> B | 0.03032 \| 0.2986 | 1.014e-05 \| 0.01199 | T4 |
| B <-> C | 0.03259 \| 0.2971 | 0.03056 \| 0.2825 | T4 |
| B <-> C at low capacity (60 trees) | 0.01763 \| 0.1401 | 0.01645 \| 0.1195 | T4 |

---

## 3. Reference convention — three frozen roles

### Cell A — `Ordinary LightGBM (standard raw-label native)`

* **Role: assessor-facing / workflow benchmark** — the standard LightGBM learner a practitioner would
  naturally run.
* Remains visible in all relevant headline tables.
* **Attribution rule:** comparisons against A are descriptively valid, but changes relative to A
  must **NOT** be attributed solely to rho, because the native/custom execution-path difference is
  also present.

### Cell B — `Centered-label native L2 (initialization-aligned)`

* The Stage-1 label **"Parity-aligned native L2"** is now scientifically
  misleading, because empirical parity with C failed. **Frozen Stage-1 artifacts are not
  rewritten**; the old label is preserved only as the metadata field
  `legacy_stage1_label = "Parity-aligned native L2"`.
* All NEW Stage-1.5 / G2 outputs use the forward name above.
* **Role: implementation-decomposition control only.** B separates
  **A -> B** (label / initialization representation effects within native L2) from
  **B -> C** (built-in-objective versus custom-objective execution-path effects).
* B is **not** the primary penalty reference and must not become a primary paper comparator.

### Cell C — `Custom-objective rho=0 origin`

* **Role: PRIMARY within-path penalty-isolating reference.**
* The clean contrast for statements about the incremental effect of rho is
  **`C(rho=0) -> Direct/Surrogate(rho>0)`**, because it holds the custom-objective execution
  path fixed and changes only rho.
* Terminology: use *"within-path penalty-isolating reference"* or
  *"custom-objective rho=0 origin"*. **Avoid the word "causal"** in
  manuscript-facing terminology.

---

## 4. Regeneration-gate adjudication

The rev.3 RG-1 wording was written before the Stage-1 root-cause evidence existed. Its condition
("cannot be attributed to an innocuous numerical effect") is replaced, for interpretation, by:

> **Historical B<->C materially non-parity whose cause is unknown OR which implies the custom
> positive-rho path is corrupted.**

Neither holds. The cause is identified, and the positive-rho path reproduces exactly.

| RG | Fired | Decision |
|---|---|---|
| **RG-1** | **False** | FALSE under the post-G1 adjudication. The attribution problem is resolved by using Cell C as the within-path penalty-isolating origin, not by regenerating the path. |
| **RG-2** | **False** | FALSE — no defect found. |
| **RG-3** | **False** | FALSE — reproduction is exact. |
| **RG-4** | **False** | FALSE — no decision, and pinning is a no-op. |

Full evidence strings and artifact citations are in `tables/regeneration_triggers.csv`.

**No full path regeneration is required under the accepted post-G1 reference convention.**
The scientific attribution problem is resolved by using **C** as the penalty-isolating origin.

---

## 5. Future comparator convention (frozen here, NOT executed in Stage 1.5)

| Comparator | Cell | Status | Map |
|---|---|---|---|
| Primary mechanistic | **C** | **PRIMARY** | `f_b(x) = ybar_T + b*(f_C(x) - ybar_T)` |
| Secondary practical | **A** | **SECONDARY** | `f_b(x) = ybar_T + b*(f_A(x) - ybar_T)` |
| Cell B full path | B | **not scheduled by default** | — |

* Primary question: At the same first-order correction, what does retraining buy relative to globally rescaling the same unregularized custom-path predictor?
* Secondary question: Could a practitioner obtain a similar tradeoff simply by post-processing standard LightGBM?
* B condition to schedule: only if Gate G2 identifies a concrete unresolved scientific question that A and C cannot answer
* `executed_in_stage_1_5 = False`

`b_star` definition: **b_star_train = Var_T(y) / Cov_T(f0, y) when Cov_T(f0,y) > 0**.
`1/R2` is retained as a **theoretical diagnostic only; NOT the LightGBM b-star definition**.

---

## 6. Carried forward unchanged from Stage 1

* E.4 verdict on "Direct is effectively gradient-only": **INDETERMINATE** — unchanged,
  and still not asserted as fact.
* The fixed-lambda leaf-shrinkage analysis remains a **stylized** diagnostic.
* `min_sum_hessian_in_leaf` remains **checked and non-binding**.
* Track P may never certify a historical artifact.
