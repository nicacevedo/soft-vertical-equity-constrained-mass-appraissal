# P0 Major-Revision Validation

Isolated, additive analysis closing the four P0 scientific blockers on
`paper/paper_v17_option1.tex` before any manuscript revision.

**Authority:** `APPROVED_EXECUTION_PLAN.md` (rev. 3, FINAL) —
sha256 `d3b7ae3419660a8fc13897d8b867abc8bee46f4e3bb7d59db0f98c2788fb8ba1`.
That file is frozen read-only and must not be edited. Deviations are recorded in
`provenance/DEVIATIONS.md`, never by amending the plan.

**Status:** STAGE 1 (through Gate G1). Nothing beyond Gate G1 is authorized in this run.

## What Stage 1 does

| Task | Question | Fits |
|---|---|---|
| S2 — P0-1 objective audit | Real-data magnitudes of the Direct supplied Hessian vs the omitted rank-one curvature; Surrogate weight concentration; stylized fixed-λ diagnostic | none |
| S3 — temporal exposure audit | Same-date crossing at all 8 chronological boundaries; repeat-PIN overlap per fold and out of time | none |
| S4-0 — source equivalence | Does today's executed training/prediction path differ from the commits that generated the frozen artifacts? | none |
| S4a — Track H | Do the frozen artifacts reproduce under **historical** settings (tiers R1–R4)? What is the A/B/C parity ladder (tiers T1–T4)? | yes |
| S4b — Track P | Are native-L2 and custom-ρ=0 equivalent under **pinned determinism**? (implementation statement only) | yes |

## Non-negotiables

* Canonical splits are 344,607 / 38,290 / 26,641 and are loaded **only** via
  `run_temporal_cv._load_and_split_data` — the pyarrow row-group pushdown is part of the
  frozen experiment definition.
* Cell names are fixed strings: `Ordinary LightGBM (standard raw-label native)`,
  `Parity-aligned native L2`, `Custom rho=0 origin`. Cell B is never printed as
  "Ordinary LightGBM"; Cell C is never substituted for Cell B.
* Parity tiers **T1–T4** (A/B/C comparisons) and reproduction tiers **R1–R4**
  (frozen-artifact reproduction) are distinct fields and never share a verdict.
* Track P may never certify a historical artifact.
* `A↔B` / `A↔C` non-parity can never trigger positive-ρ path regeneration.
* Every non-R1 reproduction result carries a failure class from
  {F-SRC, F-DIRTY, F-ENV, F-NUM, F-IMP}. "Nondeterminism" is not a catch-all.
* The fixed-λ leaf-shrinkage numbers are `diagnostic_type = stylized_nominal_leaf`
  until actual trained-leaf Hessian sums are measured.
* Writes are confined to `analysis/p0_major_revision_validation/` and
  `output/p0_major_revision_validation/`.

## Layout

```
APPROVED_EXECUTION_PLAN.md   frozen, read-only
protocol_p0_validation.yaml  frozen protocol: tiers, taxonomy, triggers, isolation
configs/                     frozen configs and grids
code/                        P0 code; imports canonical modules, never re-implements
slurm/                       Stage-1 job scripts (Sloan CPU partitions only)
logs/                        Slurm stdout/stderr
tables/                      machine-readable results (csv + parquet twins)
figures/                     P0-only figures; never writes into paper/img/
reports/                     P0 reports
provenance/                  preflight/postflight, hashes, diffs, limitations
tests/                       Stage-1 assertions
```
