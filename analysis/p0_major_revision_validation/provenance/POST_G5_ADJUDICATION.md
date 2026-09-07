# POST-G5 ADJUDICATION — the D-SNAP screening alert did not survive targeted refinement

Stage: **STAGE 3B** (Gate G5a → G5b, temporal robustness completion).
Append-only. This document re-opens no earlier gate and edits no frozen artifact.
`DEVIATIONS.md` (Stage-1, `sha256 026a0454…`) is untouched; Stage-3B mechanical notes live in
`DEVIATIONS_STAGE3B.md`.

---

## 1. What was adjudicated

Gate **G5a** evaluated the five §J.5 material-change criteria on the 21-point (28 ρ including
ρ=0) **screening** grid. Exactly one criterion fired on D-SNAP:
`T1_beta_log_sign_or_ordering`. Because the screening grid is 4× coarser than the frozen
82-point path, §J.6 classifies that as an **alert, not a verdict**, and mandates targeted local
refinement before any promotion.

`T1` is a compound criterion, and **two different sub-facts** fired it in two disjoint regions of
the ρ axis. Each received its own refinement region, built only from original 82-point grid
values the screen had skipped, extended one screening interval on each side.

| region | sub-fact | families | ρ added | fits |
|---|---|---|---|---|
| **A** | Direct/Surrogate `β_log` ordering flips at small ρ | Direct + Surrogate | 21 | 378 |
| **B** | Surrogate held-out `β_log` "all negative" status changed at large ρ | Surrogate only | 9 | 81 |

**459 refinement fits, 45 shards** — exactly the count declared in the frozen
`configs/dsnap_refinement_grid.json`. No ρ outside that config was fit; no refinement ρ collides
with a screening ρ. Both facts are asserted, not assumed
(`tests/test_g5b_assertions.py::test_no_unplanned_rho_was_fit`).

---

## 2. The materiality rule, and why it is not a post-hoc constant

`TAU_MATCH = 0.002` is the `β_log` matching tolerance **already frozen** in
`configs/matched_beta_frozen.json` (frozen `2026-09-06T16:38:45Z`, before any refinement result
existed). `mode_g5b` re-reads that file and raises if the two ever disagree. Throughout this P0
pass, two configurations closer than τ in `β_log` are treated as *matched*; it would be
incoherent to simultaneously call a reversal tighter than τ a change of finding.

- **Ordering** is material only if `min(|gap_frozen|, |gap_dsnap|) ≥ τ` — the families are
  genuinely separated on **both** paths and the ordering genuinely reversed.
- **Sign** is material only if the path maximum crossing zero exceeds τ in absolute value —
  otherwise the path merely grazes zero.

---

## 3. Outcome

**Region A — NOT_CONFIRMED.** On the screening grid there were 8 ordering flips (4 `CV_mean`,
4 `heldout`, 0 `forward_2025`). The refined support has **more** flips, not fewer — 23 across
the three evaluations — because filling in the skipped ρ exposes additional near-ties in exactly
the region where the two families cross. **Zero are material.** Every flip sits where the
families are separated by less than τ: the largest `min|gap|` over the original eight is
**1.22e−03**, against an overall gap range up to **0.084** on the same paths. This is the
signature of a coarse-grid artifact, and the denser grid confirms it rather than dissolving it.

**Region B — NOT_CONFIRMED.** The 9 filled-in ρ in [18.42, 75.43] leave the **D-SNAP** held-out
Surrogate path **negative throughout** (refined maximum **−1.54e−04** at ρ=100). The refinement
does change the *frozen* side: its maximum rises from +1.07e−04 (ρ=86.85, screening) to
**+8.38e−04** at ρ=75.43, a point the screen had skipped. So the "all negative" status
difference between the two designs is **real and not a grid artifact** — but it is a difference
between a path that grazes zero at +8.4e−04 and one that grazes it at −1.5e−04, i.e. **2.4×
below τ**. It is not a material change in the sign of the `β_log` path.

**Gate G5b = `NOT_CONFIRMED`. `TEMPORAL_STATUS = PASS_PRIMARY_STANDS`.**

The primary temporal design stands. D-SNAP is reported as strict-date robustness.
**No 82-point D-SNAP regeneration is warranted, and none was submitted**
(`full_dsnap_regeneration_launched_in_this_run = false`).

---

## 4. One implementation correction was required to reach this verdict honestly

`mode_g5b` originally derived its ρ support from the **Direct** family alone. Region B refined
the **Surrogate only**, and its 9 ρ are by construction absent from both the screening grid and
the Direct family — so all **81 region-B fits would have been silently discarded by the very
test they were computed for**, and the sign alert would have been "re-evaluated" on the
unchanged screening grid. Measured before the fix: `len(keep) = 49`, region-B ρ inside it = 0.

The gate now takes ρ support per family: the ordering test uses the 49 ρ common to both families
(numerically the same set as before), and the sign test uses each family's own refined support
(Direct 49, Surrogate 58). Two guards were added — the gate refuses to run on partial refinement
evidence, and refuses any refinement ρ that collides with the screening grid.

**No ρ, split, LightGBM parameter, seed, filter, metric or threshold changed.** Recorded as D-4
in `DEVIATIONS_STAGE3B.md`; asserted by
`test_g5b_sign_test_actually_covers_the_region_b_support`.

---

## 5. What this does not change

- **Matched-β is untouched.** No model was refit, no artifact regenerated; every committed
  matched-β table, figure and report is byte-identical to the checkpoint that produced it.
  `MATCHED_BETA_STATUS = PASS`. D1 remains the primary development coordinate, D2 the
  duplicate-weighted pooled-OOF sensitivity, D3 the row-balanced sensitivity.
- **Cell roles are unchanged**: A = Ordinary LightGBM (assessor-facing benchmark), B =
  centered-label native L2 (implementation-decomposition control only, no post-hoc path), C =
  custom-objective ρ=0 origin (primary within-path penalty-isolating origin).
- **D-PURGE is not promoted.** It fired `T1` and `T2`, but it is an **oracle diagnostic** with
  `feeds_g5_gate = False` and does not enter the promotion gate. Its one materially changed
  quantity — the attenuated Surrogate dCor rebound — is reported explicitly in
  `reports/TEMPORAL_ROBUSTNESS_REPORT.md` §6 rather than smoothed over.
- **The fold-6/fold-7 validation-block overlap reading is carried forward unchanged.** All
  predictions remain genuinely out-of-training-sample; D1 remains primary; but fold-level results
  are **not independent** and fold SDs are descriptive chronological-window variation, **not IID
  standard errors**. The overlap was **measured** under D-SNAP (13.919 % duplicated appearances
  vs 13.885 % primary, same `fold_6&fold_7` pair, same maximum multiplicity 2), and D-PURGE
  preserves the primary evaluation sets **exactly**, asserted elementwise.

---

## 6. Manuscript consequence (mapping only — no prose proposed, no edits made)

The temporal section can state that the reported path conclusions are robust to (i) strict-date
separation at all eight boundaries, (ii) oracle removal of all repeat-parcel information from
training with evaluation sets held fixed, and (iii) restriction to never-before-seen parcels.
The one quantity that moves materially under the oracle diagnostic is the magnitude of the
Surrogate dCor rebound, which must be reported as such. **No manuscript file was edited in this
stage.**
