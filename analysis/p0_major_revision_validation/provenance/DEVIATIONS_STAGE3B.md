# Stage-3B deviations, mechanical retries and recovery record

**Append-only.** The Stage-1 record `DEVIATIONS.md` is a hash-pinned frozen artifact
(`sha256 026a0454...`, 3551 bytes) and is never edited. Everything Stage-3B needs to record
lives here instead, in the same format.

Scope: recovery of the P0 major-revision temporal branch after the agent host died mid-poll,
completion of the Gate-G5a D-SNAP refinement, and Gate G5b.

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

### M-5 — refinement task 12 redundantly re-executed in parallel (computational recovery)
`22151066_12` (region A / `fold_7` / Direct / chunk 1-of-2, 11 ρ) was **healthy and
progressing** but anomalously slow: ~2200 s per fit against ~150 s for the identical cell
(`...__c1`) on another node. Its node, `node2621`, was oversubscribed (`CPULoad=75.06`,
58/64 CPUs allocated). Projected remaining time ~3 h.

Because the 11 ρ fits are scientifically independent — `_run_cell` constructs a fresh
estimator per ρ and never warm-starts — **the same 11 fits were re-executed redundantly and
concurrently** as `22166048`, one ρ per array element, while the original continued
untouched as the fallback. All 11 elements completed in ~4 minutes each.

**No scientific setting changed.** The race wrapper
(`code/p0_6_race_task12.py`) calls the canonical `p0_6_temporal_designs._load_block` and
`._run_cell`; it reimplements no mathematics. Same design, region, block, family, ρ values,
split protocol, `lgbm_params_sha256 = 8f0f2acd…585b`, `random_state = 2025`, `n_jobs = 1`,
feature columns, filters, objective, metric code and `OMP/MKL/OPENBLAS_NUM_THREADS = 1`.
Scheduler-only differences, all recorded in `race/task12_parallel/race_manifest.json`:
`--array=0-10` (granularity), `--exclude=node2621`, `--mem 110G→16G` (measured `MaxRSS` of
every completed refinement task is 2.4–2.7 GB; the 110 G request was ~40x over-provisioned
and was itself blocking concurrency on a near-saturated partition), and a per-task
`NUMBA_CACHE_DIR` (cache *location* only — the dCor estimator and every metric
implementation are untouched).

**Output isolation.** Race tasks wrote only to `race/task12_parallel/rho_NN.csv`; the
wrapper repoints `c.TABLES` at that directory so no race task could ever write the canonical
filename while the original job was alive.

**Selection was on completion + validation only, never on favourable results.** The
candidate passed all 26 structural/provenance checks, and for the **six** ρ the original
execution had already logged, the two independent executions agree **exactly** at the logged
precision (worst |Δ| = 0.00e+00 on both `beta_log` and `R2_price`). Only then was
`22151066_12` cancelled (`CANCELLED`, 04:20:11 elapsed) and the candidate promoted to
`tables/dsnap_refine_shard__A__fold_7__direct__c0.csv`.

**No rows were mixed across executions.** The accepted shard is one complete lineage — all
11 rows from the race. The candidate, its per-ρ files and the manifest are preserved.
The preregistered G5a/G5b interpretation, τ, and every threshold are unchanged.
