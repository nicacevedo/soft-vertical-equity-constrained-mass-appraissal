# v3 quarantine review — canonical interpretation layer

Static review. No model was run, no held-out data scored, no rho calibrated, no job submitted.

- **Quarantine commit:** `52e26275e5feceb8caaeb09c8eaf5776f54bdbca` — *[Quarantine] Interrupted and
  continued workflow from Crsr to CC. Stopped the long process*
- **Parent:** `8729a6e7452467d57e9316da8cf391e611237335`
- **Branch:** `testing`

`reports/FINAL_V3_REPORT.md` remains the execution/provenance report. **This document governs the
canonical interpretation of those results.** The report is not replaced or invalidated; where the
two differ on interpretation, this document controls.

## 1. What `52e26275` changed

183 files, 18,376 insertions, in exactly three groups:

| | |
|---|---|
| 181 files **added** | `analysis/berry_attom_validation_v3/` (scripts, slurm, tests, audit CSV/JSON/MD/PDF) |
| 1 file **modified** | `.gitignore` — a block making the compact v3 audit artifacts trackable while large parquet stays ignored |
| 1 file **modified** | `analysis/berry_cmf_validation/scripts/__pycache__/reproduce_berry.cpython-39.pyc` |

The commit arose because a Cursor workflow hit its credit limit and was continued in Claude Code,
which then ran past the intended stopping point. The overrun is confined to the items in §4 and §5.

## 2. Core manuscript and model files were not modified

Verified by `git show --name-only 52e2627`. The commit touched **no** file under `paper/`, and did
not touch `paper/paper_v12.tex`, `scripts/analyze_rho_screening_region_v2_1.py`,
`paper/img/generated_v12_994/`, `soft_constrained_models/boosting_models.py`, or
`analysis/berry_attom_validation_v2/`.

Its single entry under frozen v1 (`analysis/berry_cmf_validation/`) is compiled Python bytecode,
not scientific content, and no v1 source `.py` was altered. That bytecode had been tracked since
`d6b07b7c`; `52e26275` only updated its bytes (status `M`, not `A`).

`panel_freeze/final_panel_freeze_v3.yaml` is unchanged:
`sha256 01495b232efeac9be1ee99e33af3aca9cf8c501578fa3ec26f7935e908bea2e6`.

An accidental Forest City / data-center prompt occurred in a prior session. A repository-wide scan
found no Forest City artifacts here. No Forest City analysis was or should be performed.

## 3. CANONICAL_PROTOCOL_CONFORMING

Aligned with the intended v3 design. Do not rerun or alter.

1. Berry/local source validation and provenance.
2. ATTOM parcel and transaction linkage.
3. Unconditional linkage waterfall.
4. Nested linkage-selection diagnostics.
5. Strict pre-sale Assessor History construction (`history_effective_date < sale_date`).
6. Standardized ATTOM 2016–2025 modeling cohorts.
7. Modeling-cohort retention diagnostics.
8. Development/validation-only baseline selection.
9. `panel_freeze/final_panel_freeze_v3.yaml` **exactly as frozen**.
10. Final untouched held-out LightGBM baselines (Wayne R² 0.795 / PRB −0.109; Philadelphia 0.505 /
    −0.218; St. Louis 0.681 / −0.056).
11. Direct `LGBCovPenalty[diff]` paths using full-pretest rho mapping.
12. Baseline and Direct bootstrap results (200 monthly time-block draws, shared indices).
13. Existing-six ATTOM analysis as a separate sensitivity layer.
14. Sloan execution provenance (`sched_mit_sloan_batch_r8`; `mit_normal` never used).

## 4. CANONICAL_BUT_QUALIFIED

**Philadelphia frozen code-385 cohort.** The frozen primary cohort keeps
`PROPERTYUSESTANDARDIZED = 385`, which retains **18.2%** of Philadelphia safe-history sales
(38,043 of 208,508) against 85.4% in Wayne and 84.3% in St. Louis, and retention is
value-dependent across sale-price deciles.

> Canonical interpretation: Philadelphia primary-path results are **conditional on the frozen
> code-385 cohort** and are **not** a clean full-Philadelphia residential-market claim.

**St. Louis source-provider comparison.** 476 common transactions, 38 validation rows.

> Canonical interpretation: **INCONCLUSIVE / UNDERPOWERED.** It supports no conclusion about either
> provider. The proposed 2005–2019 follow-up is deliberately not implemented.

## 5. EXPLORATORY_POSTHOC_OR_PROTOCOL_DEVIATION

**Philadelphia broad-residential sensitivity.** Useful, but **sensitivity only**: it does not revise
`final_panel_freeze_v3.yaml`, was never scored on held-out, and received no Direct or Surrogate
path. It must never be presented as a primary Philadelphia result.

**External v3 Surrogate family — classified `EXPLORATORY_METHOD_DIAGNOSTIC`.**

The original frozen v3 Surrogate implementation **was amended after pass 1**. This is stated
plainly rather than smoothed over; the amendment is recorded in `protocol_v3.yaml` and
`panel_freeze/SURROGATE_RECALIBRATION_LOG.md`, both retained as audit trail. The reasons were
diagnosable from pretest artifacts alone — an insufficient pass-1 rho-grid ceiling, and a branch
detector reacting to tiny numerical variation — but the amendment is still a protocol deviation.

**Philadelphia's held-out Surrogate rows were scored before the amendment and scored again in
pass 2. That is a second held-out look and is disclosed as such.**

All pass-1 and pass-2 files are preserved and must not be deleted or overwritten. Wayne has no
pass-1 snapshot because its pass-1 job failed before writing (`21882241_0`, stale NFS file handle).

Wayne and St. Louis Surrogate artifacts remain useful diagnostics, but because the
cross-jurisdiction calibration rule was amended after pass 1, the **entire external v3 Surrogate
family stays in the exploratory layer** for conservative interpretation. It must not be presented as
equally protocol-conforming with the Direct path.

## 6. DO_NOT_USE_AS_SCIENTIFIC_CLAIM

**St. Louis rho 1e3–1e10 extrapolation.** `reports/FINAL_V3_REPORT.md` (line 303) extrapolates the
terminal St. Louis Surrogate branch slope out to rho ≈ 1e3, 1e6, 1e8 and 1e10 and concludes that
widening the grid would not deliver stronger targets. **That extrapolation is not strong enough for
a canonical claim and is withdrawn here.** The historical report is retained unedited as execution
provenance. The canonical statement is:

> Under the tested first-branch grid, St. Louis attained the 10%, 25%, and 50% Surrogate anchors.
> Higher requested reductions were not attained within the tested grid. No claim is made about
> asymptotic attainability outside that grid.

**Degenerate LR level-space metrics as equity evidence.** The held-out linear-regression reference
produces level-space aggregates destroyed by a handful of log-space extrapolations — St. Louis
`R2_price` ≈ −4.2e8, `COD` ≈ 6586, `PRB` ≈ 1309, driven by a single prediction near $1.5e12;
Philadelphia `R2_price` ≈ −51 with two rows above 100×. Read `R2_log` for LR. These numbers are
artifacts of one or two rows and are **not** statements about vertical equity. Nothing was clipped;
LightGBM is the baseline carrying the science, and no frozen status or rho anchor depends on LR.

## 7. Recommended interpretation for paper integration

1. **Direct is the primary protocol-conforming external method-transfer evidence from v3.** It is
   protocol-conforming *within v3*; it is **not** elevated to a confirmatory jurisdiction claim for
   the paper as a whole, which treats the external multi-jurisdiction work as external/exploratory
   transfer evidence. Cook County / CCAO remains the primary application.
2. Surrogate results are exploratory method diagnostics only.
3. Philadelphia claims must be stated as conditional on the code-385 cohort.
4. The St. Louis provider comparison must be reported as underpowered, or omitted.
5. Wayne is **Wayne County**, never Detroit; the Berry anchor is Detroit city. St. Louis County
   (29189) is never St. Louis City (29510).
6. Official assessment ratios are never AVM valuation ratios.
7. Unconditional linkage rates are reported against eligible Berry N.

## 8. Provenance hazards recorded (documentation only)

- **Surrogate code/artifact mismatch.** `52e26275` contains pass-3 Surrogate code
  (`SURROGATE_PASS = 3`, `estimate_surrogate_noise_floor` in `scripts/v3_common.py`) while every
  Surrogate artifact on disk is **pass 2**. Pass 3 was never executed. The committed code therefore
  does not reproduce the committed artifacts. Documented only — no code reconstructed, modified, or
  rerun.
- **Report regeneration.** Re-running `scripts/write_final_report.py` would regenerate the
  withdrawn St. Louis extrapolation of §6. Treat `FINAL_V3_REPORT.md` as a frozen historical
  record, not a file to refresh.
- **Bytecode hygiene.** The three tracked `.pyc` files under
  `analysis/berry_cmf_validation/scripts/__pycache__/` were untracked with `git rm --cached`; the
  files remain on disk. The repo does carry global `__pycache__/` and `*.pyc` rules
  (`.gitignore` lines 2–3), but the blanket un-ignore `!analysis/berry_cmf_validation/**` at line 57
  defeated them for this subtree — which is how the bytecode became trackable in the first place.
  Two re-ignore lines were therefore added to the v1 block, mirroring the equivalents the v2 and v3
  blocks already carry. Without them the untracking would not hold: the next `git add -A` would
  restore all three. No source `.py` was touched.

## 9. Status

**NO MORE MODEL RUNS ARE REQUIRED BY THIS REVIEW.**
