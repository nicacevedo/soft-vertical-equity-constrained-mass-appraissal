# POST-G3 ADJUDICATION — development-coordinate overlap and the D3 row-balanced sensitivity

Stage: STAGE 3 (formal matched-β comparison + temporal robustness).
Written **before** any held-out or 2025 matched-β metric was read; the matched configuration
table was frozen and hashed first (`configs/matched_beta_frozen_hash.json`,
`file_sha256 = ace12b09e79b92511b01979620b48a49e0765f990de4b93ad6025174a069a7a1`,
frozen at `2026-09-06T16:38:45Z`).

This document adjudicates one repository finding discovered during Stage 2 and its consequences
for the matched-β design. It changes no frozen artifact and re-opens no earlier gate.

---

## 1. The finding

The seven rolling-origin **validation** blocks are **not mutually disjoint**. The validation
blocks of **fold 6 and fold 7 share 20,988 rows**.

Measured structure of the concatenated development out-of-fold sample:

| Quantity | Value |
|---|---|
| Concatenated fold-validation appearances | **151,153** |
| Distinct sales among them | **130,165** |
| Duplicated appearances | **20,988** (13.885 % of appearances) |
| Sales appearing in exactly one fold (`m_i = 1`) | **109,177** |
| Sales appearing in exactly two folds (`m_i = 2`) | **41,976** |
| Sales appearing in three or more folds | **0** |

Mechanism: `build_rolling_origin_protocol` advances the origin on 15-month **date** offsets but
cuts train/validation **positionally** (newest 10 % of each cumulative window). The final origin
step lands close to the end of the eligible universe, so the cumulative windows of folds 6 and 7
are similar in size and their newest-10 % tails intersect.

This is a property of the **frozen primary design** — the design that produced the manuscript's
existing `CV_mean` column — not something introduced by this stage. It was surfaced here because
`utils/delta_nl.estimate_delta_nl` correctly refuses non-unique `row_ids` within a split; that
guard is what exposed it. The Stage-2 fix was to pass a composite deterministic identifier
`"fold|row_id"` for the pooled-OOF sample only. That identifier is a function of observation
identity alone and is independent of every prediction, so the frozen Δ_NL estimator spec
(`e85069150b509a3518eeb2abff02b91d589bc13fd4c1f265da8978c9798c5243`) is unchanged.

---

## 2. Interpretation — what the overlap does and does not do to D1

**D1** = equal-weight mean of the seven fold-specific `β_log` values. It remains the **FROZEN
PRIMARY** development coordinate, because it is the coordinate the manuscript's `CV_mean` column
already reports and the coordinate in which the frozen 82-point path is expressed. Changing the
primary coordinate now would silently redefine every historical development number.

**D1 must not be described simply as "unaffected."** The precise statement is five-part:

1. **No D1 number is arithmetically wrong, and no fold-level prediction is contaminated.**
   Every fold's `β_log` is computed on that fold's own validation block using predictions from a
   model that never saw those rows in training. Each of the seven summands is individually a
   valid out-of-training-sample statistic, and the mean of seven such statistics is well defined.
   The overlap is between *evaluation* blocks of different folds, not between a fold's training
   and validation data, so it is **not** train/test leakage.

2. **The seven fold statistics are not independent, and D1 is therefore not a mean over
   disjoint evidence.** 41,976 sales (13.9 % of appearances) enter two of the seven summands.
   Any reading of D1 as an average over seven separable pieces of evidence — and in particular
   any standard-error, sampling-distribution, or "seven replications" reading of `CV_sd` — is
   unsupported. This **compounds** the nesting non-exchangeability already recorded as C-9
   (the folds share most of their *training* data): the folds share evaluation data as well.

3. **D1 carries an implicit non-uniform observation weighting.** Under equal fold weights, a
   sale in the fold-6 ∩ fold-7 intersection influences 2/7 of the mean while a sale appearing in
   one fold influences 1/7 of one summand. D1 is thus a weighted development functional whose
   weights were never chosen deliberately — they are a by-product of the positional cut. It is a
   legitimate and reproducible coordinate; it is not an equally-weighted average over sales.
   Note also (A9) that equal fold weighting already over-weights the small early folds relative
   to their sample size; the overlap adds a second, distinct non-uniformity on top of that.

4. **The magnitude of the resulting distortion is measured, not assumed, and it is not
   negligible at the matching tolerance.** Against the row-balanced D3 coordinate (§3),
   `max |D1 − D3| = 0.004867` in `β_log` units (Surrogate ρ=100), which is **2.4× the frozen
   matching tolerance τ = 0.002**. Per family: Direct 0.002419, Surrogate 0.004867,
   C-posthoc 0.002090, A-posthoc 0.002295. The three-way common support moves from
   `[−0.138271, −0.078651]` (D1) to `[−0.137744, −0.076705]` (D3). So the overlap is
   large enough to matter for *matching*, which is exactly why the D3 sensitivity is mandatory
   here rather than optional.

5. **The overlap is confined to the development coordinate.** The held-out block (38,290) and
   the 2025 forward block (26,641) are each a single evaluation set with no internal duplication,
   so every reported held-out and 2025 metric — the outcomes the matched-β comparison actually
   compares — is untouched by this finding. The overlap affects how configurations are *selected
   and aligned*, not how they are *scored*.

---

## 3. The three development coordinates and their frozen roles

| ID | Definition | Role |
|---|---|---|
| **D1** | equal-weight mean of the seven fold-specific `β_log` | **FROZEN PRIMARY** — matches the manuscript's existing `CV_mean`; defines the common support and the CORE targets |
| **D2** | `β_log` on the concatenated 151,153 fold-validation appearances | **duplicate-weighted historical / reproducibility sensitivity ONLY** — it explicitly gives overlapping sales double observation weight, so it is reported for continuity with the historical pooled-OOF calculation and is never used to select a configuration |
| **D3** | `β_log` on the 130,165 distinct sales with appearance weights `w_ik = 1/m_i` | **row-balanced robustness sensitivity** — each distinct sale carries total weight exactly one (measured max deviation from one: **0.0**) |

D3 uses development predictions only. The `A-posthoc` family is reported as SECONDARY throughout
and, per the frozen convention, does **not** participate in determining the common support, the
CORE target selection, or whether a target exists.

---

## 4. `D3_MATERIAL` verdict and the bounded sensitivity it triggers

**`D3_MATERIAL = TRUE`**, on two independent grounds:

1. `max |D1 − D3| = 0.004867 > τ = 0.002`.
2. Re-running the identical deterministic K=6 CORE construction under the D3 common support
   selects a **different fitted Direct configuration at j=2** (ρ = 1.930698 under D3 versus
   ρ = 1.676833 under D1). The other five CORE anchors are unchanged. All six D3 targets are
   still attained within τ (worst achieved gap 0.000619).

Consequence, per the Stage-3 authorization: a **bounded** D3 matched sensitivity is constructed —
the same deterministic K=6 procedure under the D3 common support, reusing existing fitted
configurations, with C-posthoc solving exactly on D3. **No targeted Surrogate fits are required:**
every D3 CORE target has an actually-fitted Surrogate configuration within τ. The experiment is
not redesigned; the sensitivity is emitted as `tables/matched_beta_d3_sensitivity.csv` alongside
the primary D1 table.

**Rank order is not preserved between D1 and D3** (`rank_order_preserved_D1_vs_D3 = false`).
This must not be read as the overlap scrambling the paths. Both Direct and Surrogate `β_log`
paths are **non-monotone in ρ under both D1 and D3** (`Direct: D1_monotone = false,
D3_monotone = false`; `Surrogate: likewise`; both post-hoc families are monotone under both).
Within-family rank shuffling between the two coordinates is therefore driven by pre-existing
non-monotonicity of the path in flat regions, where a 0.002–0.005 coordinate shift reorders
configurations that were already nearly tied — not by any distortion the overlap introduces into
the ordering itself.

---

## 5. What this does not change

- No frozen artifact is modified. The 82-point path table, the fold protocol, the Δ_NL estimator
  spec, and the A/B/C reference convention all stand exactly as adjudicated at G1/G2.
- No earlier gate is re-opened. G1, G2 and G3 remain ACCEPTED.
- The primary matching coordinate remains D1. D3 is a sensitivity, reported alongside, never
  substituted for D1.
- Held-out and 2025 metrics remain actual measured values of actually fitted configurations, or
  actual evaluations of the theorem-matched transformation at an exactly solved `b`. **Nothing is
  interpolated between ρ values**, under either coordinate.

## 6. Manuscript consequence (mapping only — no prose proposed, no edits made)

The overlap is a disclosure item for §4.2 and App. F (`tab:fold_structure`), alongside the
existing nesting disclosure (C-9). The accurate framing is that `CV_mean` / `CV_sd` describe
variation across **nested, partially overlapping** chronological validation windows — not
statistical uncertainty from seven independent replications. The measured overlap
(20,988 shared rows between folds 6 and 7; 13.9 % of pooled appearances) and the bounded
coordinate effect (`max |D1 − D3| = 0.0049` in `β_log`) are the numbers to state.
