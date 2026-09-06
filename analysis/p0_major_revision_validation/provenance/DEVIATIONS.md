# Deviations and implementation notes — Stage 1

The approved plan (`APPROVED_EXECUTION_PLAN.md`, rev. 3, sha256 `d3b7ae34…`) is frozen read-only.
Anything that differs from it in execution is recorded here, never by amending the plan.

## Authorized amendment carried into the protocol (not a deviation)

**AMD-1 — Step 7 / J4: 9 native refits → 18 native refits.** Cell A and Cell B across all 9
fitting blocks, retaining in-sample training predictions, the corresponding evaluation
predictions, `ybar_T`, `f0bar_T`, `Var_T(y)`, `Cov_T(f0,y)` and `b_star_train` for both cells.
Cell C is never substituted for Cell B. Issued with the Stage-1 execution instruction; recorded in
`protocol_p0_validation.yaml`. **Affects Stage-2 compute only; not executed in this run.**

## Deviations from the plan

**None affecting scientific design.** The three items below are implementation-level and change no
definition, no threshold, no sample and no reported quantity.

### D-1 — `meta_pin` membership computed on integer codes (performance only)
`meta_pin` is object dtype (14-character zero-padded strings; 327,052 distinct over 409,538
eligible rows). `np.isin` on object arrays falls back to an O(n·m) scan and did not complete in
10 minutes. The PIN column is now factorized once over the full eligible universe
(`pd.factorize`, a bijection on observed values) and all membership tests run on int64.
Identical results, ~60x faster. *No scientific change.*

### D-2 — canonical rho grid read from the generating protocol artifact
`frozen_rho_grid()` initially derived the 82-point grid from the two `experiment_spec.json`
files. It now reads `V12/protocol/lower_rho_grid_v2.json` (the artifact that *generated* the
grid, storing full float64 repr) and asserts agreement with the spec-derived grid to within
1 ULP. Measured maximum relative difference: **4.97e-16**. S2 was re-run under the canonical
grid; the E.4 verdict and every reported digit were unchanged. *No scientific change.*

### D-3 — Stage-1 test runner
`pytest` is not installed in `fairness_env`. `tests/run_stage1_tests.py` provides a minimal
runner with `pytest.raises` / `pytest.skip` shims so `tests/test_p0_assertions.py` runs unmodified
and can also be executed by pytest elsewhere. *No scientific change.*

## Corrections to plan narrative figures, established from repository evidence

### C-1 — development/held-out boundary counts on 2023-11-09
The plan's narrative (§J.1, from an earlier read-only reconstruction) records **43 development /
105 held-out** rows on the boundary date. Measured here from the canonical frames returned by
`run_temporal_cv._load_and_split_data`: **42 development / 106 held-out**. Total exposure is 148
rows either way, and every scientific statement is unchanged (same-date crossing at all 8
boundaries; < 0.5 % of any evaluation block). Per the authority order, the canonical-loader
measurement supersedes the narrative figure.

All other plan figures reproduced exactly, including all seven fold boundary counts
(46/46, 156/35, 121/54, 123/47, 35/166, 132/38, 158/2) and all nine repeat-PIN exposure shares.

## Mechanical job failures and retries

### M-1 — first `sbatch` submission rejected (`invalid time limit`)
A shell parameter-expansion bug in the sbatch generator truncated `--time=08:00:00` to
`--time=00`, because the time string itself contains colons. The generator was rewritten with
explicit positional fields and the four jobs resubmitted unchanged in every other respect.
No scientific parameter was altered.

---

# Deviations and implementation notes — Stage 3B (temporal completion / Gate G5b)

Appended, not amended. Nothing above this line was edited.

## Mechanical job failures and retries

### M-2 — VS Code agent host terminated with SIGBUS while polling array 22151066
The Claude Code frontend died while **polling** the already-running D-SNAP refinement array.
The Slurm jobs are independent of the frontend and continued unaffected; no scientific process
was interrupted and no artifact was left partially written (each shard is written once, at the
end of its cell, by `c.write_table`). State was reconstructed from repository artifacts, `sacct`,
and the job logs — not from conversation memory. *No scientific change.*

### M-3 — refinement array task 1 failed on a stale NFS file handle
`22151066_1` (region A / `fold_1` / `direct` / chunk 2 of 2) failed after 1:05 with
`OSError: [Errno 116] Stale file handle`, raised inside `numba.core.caching` while `dcor`
loaded its on-disk JIT cache index from the conda site-packages tree. This is a filesystem
error during library import, before any model was fit; it is not a numerical or scientific
failure. The shard was resubmitted as `22155049` with `sbatch --array=1` against the **same
frozen** `slurm/09_dsnap_refinement.sbatch` — identical script, identical ρ values, identical
LightGBM parameters, identical split protocol, identical metric definitions. Only the array
range differs. It completed in 5m34s and wrote the 10 expected rows.
*No scientific parameter was altered.*

### M-4 — two refinement tasks ran ~15x slower than their siblings (node contention)
`22151066_3` and `22151066_12` were co-scheduled on `node2621` and averaged ~820 s and ~2224 s
per fit against ~55 s and ~150 s for the identical cells on other nodes. Per the recovery rule,
a task that is RUNNING and demonstrably progressing is left alone; both were allowed to finish
inside their 12-hour wall clock. Wall-clock rate does not enter any fitted quantity.
*No scientific change.*

## Deviations from the plan

### D-4 — Gate G5b rho support is taken per family, not from the Direct family alone
**Implementation correction to unexecuted code, found before its first run.**

`mode_g5b` originally restricted both the ordering test and the sign test to
`keep = rho values present on the **Direct** family`. The frozen refinement grid
(`configs/dsnap_refinement_grid.json`, built at the previous checkpoint) defines **two** regions:

- **region A** refines *both* families (21 ρ), and
- **region B** refines the **Surrogate only** (9 ρ) — it exists specifically to test the
  Surrogate held-out `β_log` sign alert.

Because region B's 9 ρ are Surrogate-only and are (by construction) not screening ρ, a
Direct-derived `keep` contains none of them. All **81 region-B fits would have been silently
discarded by the very test they were computed for**, leaving the sign alert re-evaluated on the
unchanged screening grid. Verified empirically before the fix: `len(keep) = 49`, region-B ρ
inside `keep` = **0**.

The fix computes `rho_by_fam` per family; the **ordering** test (which needs both families at a
ρ) runs on their intersection — numerically the same set as before — and the **sign** test runs
on each family's own refined support. Two guards were added at the same time: `_refined_dsnap`
now raises unless every ρ of every region is present for all nine blocks on each declared family
(the gate may not run on partial evidence), and unless every refinement ρ is absent from the
screening grid (the refinement must contain only *skipped* original-grid points).

**This changes no ρ, no split, no LightGBM parameter, no random state, no data filter and no
metric definition.** It makes the already-frozen refinement grid actually reach the gate it was
built for. Covered by `tests/test_g5b_assertions.py::test_g5b_sign_test_actually_covers_the_region_b_support`.
