# Surrogate recalibration: pass 1 → pass 2

Written 2026-09-03. Amends `protocol_v3.yaml` → `surrogate.calibration_amendment` only.

**No jurisdiction status changed.** `panel_freeze/final_panel_freeze_v3.yaml` is byte-identical
before and after this repair (`sha256 01495b232efeac9be1ee99e33af3aca9cf8c501578fa3ec26f7935e908bea2e6`).
No target anchor was added, dropped, or moved. Rho was never selected or refined from a test metric
in either pass.

## What pass 1 produced

Pass 1 ran as Slurm array `21882241` on `sched_mit_sloan_batch_r8`.

| county | array index | state | first-branch outcome |
| --- | --- | --- | --- |
| wayne | `21882241_0` | FAILED | never calibrated |
| philadelphia | `21882241_1` | COMPLETED | 10% and 25% attained; 50–97% UNATTAINED |
| st_louis_county | `21882241_2` | COMPLETED | every target UNATTAINED |

Wayne's failure was infrastructure, not logic: `dcor`'s numba cache index was read off the shared
filesystem and raised `OSError: [Errno 116] Stale file handle`. Because the downstream bootstrap and
report jobs hung off `--dependency=afterok` on this array, both were cancelled
(`21882242`, `21882243`) and never ran. Pass 2 exports `NUMBA_CACHE_DIR` to node-local disk.

## The two defects

Both are visible in pretest-only artifacts. Neither was found by looking at held-out metrics.

### 1. The grid ceiling sat below the Direct anchors

Pass 1 used a fixed `np.geomspace(1e-6, 1e2, 16)`. The Direct path, mapping rho from the same
pretest block, needs far more than 100 to reach its strong-penalty anchors:

| county | max Direct rho (97% anchor) | pass-1 surrogate grid ceiling |
| --- | --- | --- |
| wayne | 200.94 | 100 |
| philadelphia | 255.82 | 100 |
| st_louis_county | 135.41 | 100 |

Philadelphia's pass-1 branch was monotone increasing all the way to the last grid point and stopped
at 0.324 reduction *at rho = 100*:

```
rho      achieved_reduction
0.2154   0.0178
0.7356   0.0565
2.5119   0.1098
8.5770   0.1903
29.2864  0.2671
100.0    0.3238   <- last grid point, still rising
```

So the 50/67/80/90/97% rows were reported UNATTAINED because the grid ran out, not because the
branch bent. Pass 1 could not tell those two cases apart, and the pre-freeze report narrated the
result as if the branch had bent — which is the opposite of the CCAO finding it was meant to test.

### 2. The branch detector had no noise floor

Pass 1 opened a branch on any reduction `>= -1e-8` and closed it on any decrease beyond `1e-8`.
St. Louis's first grid point, rho = 1e-6, returned a reduction of 0.008 — numerically positive but
far below anything the penalty could actually be doing at that rho. That opened the branch; the next
point came back marginally lower and closed it. The result was a one-point branch:

```
rho     achieved_reduction
1e-06   0.008028
```

With `len(branch) < 2` there is nothing to interpolate between, so all seven targets returned
UNATTAINED. This is an artifact of float noise at negligible rho, not a property of
`LGBSmoothPenalty`.

## Pass 2

Declared in `protocol_v3.yaml` before running.

**Grid rule.** `geomspace(1e-3, 4 × max(Direct rho for this county), 25)`. The ceiling is tied to
this county's own Direct rho mapping, which is itself pretest-only, so it carries no test
information. Resulting spans: Wayne `1e-3 … 803.8`, Philadelphia `1e-3 … 1023.3`,
St. Louis `1e-3 … 541.7`.

**Branch detector.** A branch opens only once achieved reduction exceeds `noise_floor = 0.01`, and
closes only on a drop greater than `max(noise_floor, 0.05 × |previous reduction|)`. Every row now
carries `branch_terminated_by` ∈ {`MATERIAL_REVERSAL`, `GRID_CEILING`, `FIT_FAILURE`,
`NEVER_STARTED`} and, for UNATTAINED rows, an `unattained_reason` (adding `BRANCH_TOO_SHORT`), so a
genuine S-shaped bend is never again reported as if it were a grid limit or vice versa.

**Fit failures.** Pass 1 swallowed per-rho exceptions into a bare `np.nan`. Pass 2 records them in
`method_transfer/<county>/surrogate_fit_errors.json` alongside the grid bounds and the baseline
validation covariance, and writes the full unfiltered grid curve to
`surrogate_full_grid_curve.csv` next to the accepted branch.

## Disclosure of the second held-out look

Philadelphia's chronological test rows were scored under pass 1 at two attainable anchors (10% and
25%). Pass 2 scores them again under the repaired calibration. This is a second look at the same
held-out rows and is reported as such rather than silently overwritten:

- pass-1 outputs are preserved as `surrogate_pass1_rho_first_branch.csv`,
  `surrogate_pass1_branch_curve.csv`, `surrogate_pass1_heldout.csv`;
- pass-2 outputs carry a `surrogate_pass = 2` column;
- `preserve_pass1` refuses to overwrite an existing pass-1 snapshot, so a later run cannot
  retroactively turn pass-2 numbers into "pass 1";
- `test_surrogate_pass1_preserved_when_pass2_exists` fails the suite if a pass-1 snapshot goes
  missing while pass-2 output exists.

Wayne and St. Louis have no pass-1 held-out surrogate numbers to compare against — Wayne never ran,
and St. Louis attained no anchor — so for those two counties pass 2 is their first and only held-out
surrogate evaluation.

## What this repair cannot fix

If pass 2's branch still terminates by `MATERIAL_REVERSAL` before a requested target, that is a
scientific finding about `LGBSmoothPenalty` on this county and is reported as UNATTAINED with the
reversal reason attached. The repair widens the search and removes a noise artifact; it does not
guarantee attainment, and no target will be relabeled attained by moving the grid again.
