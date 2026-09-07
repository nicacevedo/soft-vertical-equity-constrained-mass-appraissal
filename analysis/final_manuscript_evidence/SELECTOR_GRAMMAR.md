# Selector grammar and the value pipeline (normative)

How a number in the manuscript is tied to a byte in a frozen artifact, and how
that byte becomes the printed value. This file is the normative definition;
`code/b0_common.py` implements it and `tests/test_b0_numeric_map.py` re-derives
every row against it independently.

## 1. Selectors

| artifact kind | selector form | resolution rule |
|---|---|---|
| `csv` | `col=val;col2=val2` | must match **exactly one** row. Values are compared as **text**, never parsed as numbers. `column` (or `metric`) names the value column. |
| `csv`, float key | `rho~=0.954095476349994` | `abs(a-b) <= 1e-12 * max(1, abs(b))`, computed in `Decimal`. Must still match exactly one row. |
| `json` / `yaml` | `$.a.b[0].c` | dot/bracket path; must resolve to exactly one **scalar**. Resolving to a dict or list is an error. |
| `markdown` | `heading="..."; quote="..."` | the quote must occur **exactly once** within that heading's section, and the heading must match exactly one heading. |

A selector that matches zero rows, more than one row, or a non-scalar is a build
failure. Nothing falls back to a default.

**The unique row selector for both P1 inference tables** is
`(display_kind, family, j, ext_target, rho, role_label, evaluation)`, verified
unique on all 480 rows of `prb_inference.csv` and of `vei_significance.csv`.
`(realization_key, evaluation)` yields only 440 groups, because
`fit:LGBCovPenalty:1fb838f7d6bfda88` appears twice — once as the Cell-C
reference and once as core j=0 Direct. That single duplication is the whole
reason the entry-level and unique-realization ED2 counts differ by 9.

## 2. Raw values are literal text

`raw_value` is the **decimal literal as it appears in the artifact**, read with
Python's `csv` module and never converted to `float`. Two consequences:

- re-resolution is byte-exact, so a test can assert `raw_value` reproduces from
  the selector without any tolerance;
- rounding is deterministic from the recorded literal alone.

This matters because P1 records that its committed CSVs round-trip float64 to
about 3.55e-15. **No Tier-B0 output claims CSV bit-exactness.** Carrying the
literal text sidesteps the question instead of asserting something false.

## 3. The two-stage value pipeline

```
raw_value  ──value_transform──▶  transformed  ──rounding_rule──▶  display_value
```

The stages are separate and `rounding_rule` **never** performs a unit
conversion. The frozen artifacts genuinely mix units, and conflating the two
stages is how a percent gets multiplied twice or not at all:

| metric | in the artifact | printed | so |
|---|---|---|---|
| `MAPE` | `0.2120088150100442` (a fraction) | `21.2%` | `times_100` + `half_up:1` |
| `VEI` | `-26.459634217874914` (already percent) | `-26.5%` | `identity` + `half_up:1` |
| `COD` | `21.6265818314926` (already percent) | `21.6%` | `identity` + `half_up:1` |

### `value_transform` — closed vocabulary, default `identity`

| rule | meaning |
|---|---|
| `identity` | no change |
| `times_100` | exact decimal shift by +2 (fraction → percent) |
| `divide_100` | exact decimal shift by −2 (percent → fraction) |
| `absolute_value` | magnitude |

All four are exact `Decimal` operations. `times_100` is a `scaleb`, not a
multiplication, so no representation error is introduced.

### `rounding_rule` — closed vocabulary

| rule | meaning |
|---|---|
| `half_up:<n>` | `n` decimal places, `ROUND_HALF_UP` |
| `sig:<n>` | `n` significant digits, `ROUND_HALF_UP` |
| `sci:<n>` | `n` significant digits in exponential form with a two-digit exponent, e.g. `3.24e-02` |
| `usd_comma:0` | integer with thousands separators, e.g. `344,607` |
| `int` | integer |
| `verbatim` | pass a status string through unchanged |
| `none` | emit nothing — used only for `NOT_ATTAINED` |

Rounding always uses `decimal.ROUND_HALF_UP`, **never** Python's built-in
`round()`, which is banker's rounding: `0.025` must display as `0.03` at two
places, not `0.02`.

`raw_unit` and `display_unit` are recorded alongside, so a unit mismatch is
visible in the CSV rather than hidden in the rounding rule.

## 4. Manuscript anchors — why line numbers are not enough

Line numbers are **baseline coordinates only**. The moment the writing pass
deletes a table, every later line number is wrong. Each map row therefore
carries five additional anchors, all mechanically derived:

| column | how it is derived |
|---|---|
| `manuscript_section` | nearest preceding `\section` / `\subsection` / `\subsubsection`, or `\begin{abstract}`; prefixed `Appendix ` after `\appendix` |
| `latex_label` | the authored `\label{...}` of the table or figure, where there is one |
| `source_anchor` | nearest preceding `\label{...}` — the same algorithm `markup_preflight.py` uses |
| `baseline_excerpt_norm` | whitespace-normalized excerpt of the source line, ≤120 chars |
| `baseline_excerpt_sha256` | sha256 of that normalized excerpt |
| `baseline_line` | the line number, explicitly **non-authoritative** |

To re-find a passage after edits: search for `baseline_excerpt_norm`, or fall
back to `latex_label` / `source_anchor` within `manuscript_section`.

`manuscript_location` composes them: `L<line>|<source_anchor>` plus
`|row=<r>|col=<c>` for a table cell.

## 5. Identity

`claim_id` names a **scientific value** (`N-<domain>-<...>`), and is deliberately
**reused** wherever that value appears. The primary key of
`manuscript_numeric_map.csv` is the triple
`(claim_id, manuscript_location, metric)`: one value printed in a table and
quoted again in prose is two rows with one `claim_id`.

`manuscript_claim_map.csv` uses `C-<section-slug>-<nnn>`, and adds
`legacy_plan_id`, `todo_id`, `render_bucket`, `option2_precedent` and
`numeric_claim_ids`.

## 6. `NOT_ATTAINED`

`attained_status ∈ {ATTAINED, NOT_ATTAINED, NOT_APPLICABLE, UNRESOLVED}`.

For `NOT_ATTAINED`: `raw_value` is empty, `metric` is still named,
`value_transform` is `identity`, `rounding_rule` is `none`, and `display_value`
is the frozen token. The builder **re-reads the source `attained` field** and
fails if the row is in fact attained. The frozen states are exactly four
display entries — Direct at `ext_target` −0.06, −0.03 and 0.00, and Surrogate at
0.00 — giving 40 rows in each P1 table and 16 `attained=False` rows in
`matched_beta_ext_targets.csv`. **No interpolation, ever.**

## 7. `counting_unit`

Required, from a closed vocabulary, for every aggregate or count claim:
`display_entry`, `unique_realization`, `standards_facing_configuration`,
`evaluation_cell`, or `NOT_APPLICABLE`.

The two ED2 count families must never appear in one statement:

| unit | evaluated | escalated past Step 5 | Step-7 reject null |
|---|---:|---:|---:|
| `display_entry` | 396 | 279 | 228 |
| `unique_realization` | 387 | 270 | 219 |

Both triples are derived here from `vei_significance.csv`, not transcribed.
Prefer `unique_realization` for methodological summary statements; use
`display_entry` only where the claim is specifically about display roles.

## 8. Reference cell and comparison purpose

`reference_cell ∈ {A, B, C, NONE}` and
`comparison_purpose ∈ {WORKFLOW_BENCHMARK, PENALTY_ISOLATING, WITHIN_MODEL,
NOT_APPLICABLE}`, taken from the frozen convention in
`configs/post_g1_reference_convention.yaml`:

- **A** is the assessor-facing / workflow benchmark. Contrasts against A are
  descriptively valid; changes relative to A must **not** be attributed to rho.
- **C** is the PRIMARY within-path penalty-isolating reference. `C(rho=0) →
  Direct/Surrogate(rho>0)` is the clean penalty contrast.

Guards: no row may pair `A` with `PENALTY_ISOLATING`, and none may pair `C` with
`WORKFLOW_BENCHMARK`.

## 9. The coverage audit

`code/b0_2_build_numeric_map.py` extracts every numeric token in the manuscript
body and requires each one to resolve to exactly one status.

**Masking.** The audit ignores the preamble (everything before
`\begin{document}`), math environments, and the numeric arguments of reference
or typesetting macros. The mask is *per macro and per argument position*, which
matters: `\textcolor{impGreen}{...}` and `\multicolumn{4}{c}{...}` carry **real
prose** in their last argument, so masking the whole call would hide rendered
result numbers. Two subtleties worth recording, because both silently corrupted
earlier drafts of this audit:

- `\\[2mm]` is a line break with optional spacing, **not** the display-math
  opener `\[`. Without a negative lookbehind its `\[` pairs with a distant `\]`
  and masks hundreds of lines.
- an unmatched `$` must be bounded at the next blank line, since LaTeX forbids a
  paragraph break inside inline math. Otherwise one stray `$` masks the rest of
  the file.

**Statuses.**

| status | meaning |
|---|---|
| `SOURCED` | matched by a numeric-map row at that anchor |
| `ALLOWLISTED` | in `spec/unsupported_numbers_allowlist.csv` with a closed-vocabulary reason |
| `FLAGGED_UNSUPPORTED` | a result-shaped number with no frozen source |
| `NOT_RENDERED` | in the `SUPPRESSED`, `COMMENTED` or `IFFALSE` bucket |

Allowlist reasons: `IAAO_STANDARD_CONSTANT`, `METRIC_DEFINITION_CONSTANT`,
`DESIGN_CONSTANT`, `MATH_CONSTANT`, `CITATION_OR_YEAR`, `TYPESETTING`,
`EXTERNAL_LITERATURE_STATISTIC`. Each row also records a `frozen_source` — an
artifact path, or the literal `NOT_RECORDED_IN_FROZEN_EVIDENCE`, so a reviewer
can see which allowlisted constants have no frozen backing at all.

**The no-wildcard rule.** An allowlist row may carry `*` as its anchor, meaning
"the same fact wherever it appears". `spec/coverage_policy.yaml` lists the
anchors where that wildcard does **not** apply: the blocks under
`DELETE_OR_REPLACE` review. Inside `tab:transition_summary`, a bare `7` is far
more likely to be "7/7 LOFO events" — an unsupported result — than the
seven-fold design constant, and a wildcard match would quietly launder it. Every
number in those blocks must be accounted for individually.

This is the mechanical half of the binding rule: **an unsupported numerical
result is not publishable merely because it is relabelled "exploratory".** The
other half is the gate on `QUALIFY_AS_EXPLORATORY` in
`code/b0_3_build_claim_map.py`, which is legal only when the claim names
numeric ids that all resolve with an attained status.
