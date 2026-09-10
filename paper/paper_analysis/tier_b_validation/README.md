# Tier-B manuscript validator

The gate for the Tier-B writing pass on `paper/paper_v17_option1.tex`. Built at
**B1.0, before any manuscript edit**, so every later stage has something to run.

```
python3 validate.py --stage B1.1 --compile --json runs/B1.1.json
```

## What it is for

Tier B0 built a deterministic bridge from the frozen P0 / P1 evidence to
manuscript claims: which number may be printed, from which artifact, through
which selector, with which rounding. This validator is the enforcement side of
that bridge, applied to the live file at every stage.

It exists because **the frozen suites cannot be the gate.** They carry
paper-immutability and HEAD-relative guards that fail by design the moment the
manuscript is intentionally edited, and modifying a frozen test is prohibited.
`baseline/FROZEN_SUITE_EXPECTATIONS.md` records their pre-edit state and
classifies every failure. Frozen-suite reruns are informational only.

**Hardened at B1.5:** once the live manuscript is no longer byte-identical to the
pinned pre-writing baseline, `run_frozen_suites_informational.py` refuses **all**
frozen P0/P1/B0 suites in this worktree, not just the Tier-B0 one that was
measured writing into its own frozen subtree. A post-edit rerun has no
certification value here and their paper-facing guards are expected to fail, so
the refusal costs nothing while a mutation of a frozen subtree would cost the
pass. If an informational rerun is ever wanted, it belongs in a **disposable
clean checkout or throwaway worktree at the appropriate frozen tag**; the
runner's help text says so and gives the commands. No frozen test is modified by
any of this, and the binding immutability checks remain the direct tag/subtree
diffs in `tb_scope.py`.

## Hard boundaries

* **Reads** frozen material under `analysis/`; **never writes** there.
  `tb_common.guard_write` refuses any write inside the repository other than
  into this directory, and `sys.dont_write_bytecode` is set before any frozen
  module is imported so not even a `__pycache__` appears.
* Runs **no** science: no model fit, no CV, no Slurm, no new rho value, no
  recomputation of matched-beta. Every number is re-derived from a frozen
  artifact by selector.
* Never reads `output/` or `data/` to recover an unsupported manuscript value.

## The pass criterion

The canonical manuscript is intentionally **not** compliant with the final-state
checks. So the criterion for a stage is

```
UNEXPECTED_VALIDATOR_FAILURES == 0
```

and **not** "the unrevised manuscript passes every final-state check". A
validator that passes on the baseline is not validating anything, so no check is
weakened to make the baseline green. Instead every expected failure is declared
individually in `spec/expected_failures.yaml` with the stage that resolves it --
never as one blanket exemption -- which makes resolution enforceable in both
directions: a stage cannot quietly fail to do its job, and a failure that
vanishes early is reported rather than silently accepted.

At B1.0 that is **88 expected failures, 0 unexpected**, with the unsupported-token
count at its baseline **356**.

## Layout

| path | what it is |
|---|---|
| `tb_common.py` | geography, write guard, frozen-module access, frozen-spec loaders |
| `tb_text.py` | the ACTIVE build as sentence units, with LaTeX-aware normalization |
| `tb_coverage.py` | the numeric coverage audit, re-run against the live manuscript |
| `tb_ledger.py` | the Tier-B provenance ledger: the seven-step verifier and its self-test |
| `tb_checks.py` | the thirteen checks (C01-C13) |
| `tb_scope.py` | cumulative paper-only write scope; the three frozen-subtree tag diffs |
| `tb_compile.py` | `latexmk` plus the log diagnostics that matter |
| `validate.py` | the runner: classify, report, exit non-zero on an unexpected failure |
| `run_frozen_suites_informational.py` | records what the frozen suites say; never a gate |
| `spec/expected_failures.yaml` | the stage-aware expected-failure registry |
| `spec/tier_b_citations.yaml` | required citation keys, the two-`\addbibresource` rule, bib fields that may not be invented |
| `spec/ledger_selftest.yaml` | entries that prove the ledger verifier works |
| `spec/tier_b_required_statements.yaml` | statements the manuscript must keep printing |
| `spec/unsupported_math_claims.yaml` | known ACTIVE unsupported claims written inside math mode, which the coverage audit cannot see |
| `ledger/tier_b_numeric_ledger.yaml` | the ledger itself -- empty at B1.0 by design; 12 entries from B1.3 |
| `baseline/` | the pre-edit record: frozen-suite transcripts, compile diagnostics |
| `runs/` | one JSON per stage validation |

## The thirteen checks

| id | check | notes |
|---|---|---|
| C01 | numeric provenance | every ledger entry recomputed from the artifact: reopen, verify sha256 against the file AND the frozen manifest, execute the selector, re-extract `raw_value` byte-exactly, apply the transform, apply the rounding, compare with what the `.tex` prints. `decimal.Decimal` throughout, `ROUND_HALF_UP`, never float. The self-test also mutates each step and requires a rejection. |
| C02 | no `FLAGGED_UNSUPPORTED` token | the frozen algorithm, applied to the live file. Reproduces the certified baseline exactly: 846 ACTIVE tokens -> 300 SOURCED / 190 ALLOWLISTED / 356 FLAGGED, per-anchor identical. The stage budget is contractual: 356 -> 218 -> 34 -> 0. |
| C03 | A / B / C semantics | forward display names used, legacy Stage-1 labels not; the attribution rule stated; both A and C explicit reference rows in each rebuilt table; no row pairing A with `PENALTY_ISOLATING` or C with `WORKFLOW_BENCHMARK`. |
| C04 | D1 / D2 / D3 semantics | D1 primary, D3 the one-sale-one-vote sensitivity, D2 not elevated; the fold-6/fold-7 overlap qualification present; "unaffected" never attached to D1; no fold SD read as a standard error. |
| C05 | `NOT_ATTAINED` preservation | the four frozen states verbatim, with blank metric cells. Never interpolated. |
| C06 | ED2 status and counting units | every VEI/MKI band or standard attribution marked Exposure Draft or proposed; no statement mixing the two ED2 count families; every count naming its unit. |
| C07 | wording: forbidden and required | prohibitions are data-driven from `spec/forbidden_wording.yaml`, whitespace-squashed so re-spacing does not evade them, and legal only inside a sentence carrying a frozen prohibition marker. The positive counterpart is `spec/tier_b_required_statements.yaml`: the four denials the frozen spec relies on must keep printing, since deleting a denial breaks no pattern scan. |
| C08 | candidate-region provenance | no `_candidate_region` asset, no overlay caption, and prose keyed by sentence content hash so the identity survives deletion. Visual provenance obeys the same rule as numeric provenance. |
| C09 | figure existence | every included graphic exists and is tracked in git. |
| C10 | label / reference integrity | no duplicate compiled label, no reference to an undefined one. A `\ref` inside an `oldrevisionblock` DOES resolve (the body is typeset into a discarded box); inside `\oldtext` it does not (the argument is gobbled). Getting that wrong makes the check blind or noisy. |
| C11 | citation integrity | every active `\cite` key in a loaded `.bib`; the ten required prior-art keys cited; exactly two `\addbibresource` and no dependency on a bibliography file under `analysis/`; `Cheng1974` carries the primary-source-verified volume 29, number 3--4 and pages 268--284 exactly, and no DOI, because none was found or verified -- not because none is known to exist; `Edelstein1979` carries only the confirmed start page. |
| C12 | TODO closure | the 19-site crosswalk is intact, the count never increases, and the sites go with the scaffolding at B4.2. |
| C13 | known unsupported math-mode claims | the blind spot C02 cannot cover. `b0_tex` masks math environments before extracting tokens, so an unsupported value written `$0.08677$` is outside the token population entirely -- neither SOURCED nor FLAGGED. `spec/unsupported_math_claims.yaml` registers the known ones by content (all identifying literals in one ACTIVE sentence, plus a drift cue and the normalized-excerpt sha256), each with the stage that owes its removal or rewrite. Reported **alongside** the token budget, never inside it. |

### Two numeric populations, never merged

`UNSUPPORTED_TOKENS` counts the ordinary **text-mode** population, whose
trajectory 356 -> 218 -> 34 -> 0 is contractual. `KNOWN_UNSUPPORTED_MATH_CLAIMS`
counts the C13 registry. Keeping them apart is the whole point: reaching
`UNSUPPORTED_TOKENS = 0` at B2.2 is a true statement about the text-mode tokens
and **not** a statement that every printed number resolves. A registered claim
that is still active at or after its owning stage becomes an UNEXPECTED failure,
so B2 cannot leave one behind by accident. Absence from the registry is not
evidence that a math-mode number is supported -- if a later stage finds another,
it is added there in the same commit.

Alongside them, two checks that have no expected-failure entries because they
may never fail: **cumulative paper-only write scope**
(`git diff --name-only tier-b0-final-20260907..HEAD` lists nothing outside
`paper/`) and **frozen-subtree immutability** (the three tag diffs, all empty).

## Stable anchors, not line numbers

Every baseline line number in the frozen maps, the plan and the older reports is
stale from B1.1 onward, by construction. Passages are re-identified by
`latex_label`, `source_anchor`, the normalized baseline excerpt and its sha256,
as `SELECTOR_GRAMMAR.md` section 4 requires. The C08 prose keys are content
hashes for the same reason.
