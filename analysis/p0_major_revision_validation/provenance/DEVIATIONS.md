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
