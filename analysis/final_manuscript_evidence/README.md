# Tier B0 — final manuscript evidence integration

A deterministic bridge from frozen empirical evidence to manuscript claims:

```
evidence → claim → artifact → selector → raw value → display value → required wording
```

Tier B0 runs **no experiments**, edits **no manuscript file**, and **flags**
whatever it cannot support. Its purpose is to make the Tier-B writing pass edit
against a hash-pinned source that can be re-verified mechanically, rather than
from memory — which is how superseded numbers come back.

## What is here

| file | what it is |
|---|---|
| `FINAL_EVIDENCE_INDEX.md` | what frozen evidence exists, by manuscript section, then by evidence domain |
| `MANUSCRIPT_REVISION_SPEC.md` | what to change, why, on what evidence, and what wording is forbidden |
| `TABLE_FIGURE_DISPOSITION.md` | every table and figure: KEEP / UPDATE / REBUILD / DEMOTE_TO_APPENDIX / DELETE |
| `FINAL_EVIDENCE_MANIFEST.json` | hash-pinned index of every citable artifact |
| `manuscript_numeric_map.csv` | 291 rows, 235 distinct scientific values, each derived from a selector |
| `manuscript_claim_map.csv` | the claims, with dispositions and required final messages |
| `SELECTOR_GRAMMAR.md` | **normative**: selectors, the value pipeline, anchors, the coverage audit |
| `certification/` | what the frozen tags certify, and what the local reruns do not |
| `coverage/` | every numeric token in the manuscript, classified |
| `spec/` | the **authored** source of truth — the science |
| `code/` | the **derivation** — the numbers |
| `bib/` | staged BibTeX for the four uncitable historical entries, with per-field provenance |
| `tests/` | the enforcement layer |

## Authored versus derived

This split is the whole design.

- **Authored**, in `spec/*.yaml`: which claim, which artifact, which selector,
  which units, which disposition, which required message, which counting unit,
  which reference cell.
- **Derived**, by `code/*.py`: every `raw_value`, every `display_value`, every
  `artifact_sha256`, and every manuscript anchor.

So "`raw_value` must reproduce from the named selector" and "`display_value`
must be mechanically obtained from `raw_value`" are *enforceable*, and the tests
re-derive both independently of the builders.

## The rules that are enforced, not merely stated

- **An unsupported number is not publishable by relabelling it "exploratory".**
  `QUALIFY_AS_EXPLORATORY` is legal only when the claim names numeric ids that
  all resolve with an attained status; otherwise the disposition must be
  `DELETE_OR_REPLACE` or `REWRITE_WITH_SUPPORTED_EVIDENCE`. The wildcard
  allowlist is also barred from the blocks under review, so a bare `7` inside
  `tab:transition_summary` cannot be laundered into a design constant.
- **A and C are two frozen roles, not a substitution.** A is the assessor-facing
  workflow benchmark and its contrasts stay reportable descriptively; C is the
  primary within-path penalty-isolating reference. No row may pair A with
  `PENALTY_ISOLATING`, or C with `WORKFLOW_BENCHMARK`.
- **Rounding never carries a unit conversion.** `raw_value → value_transform →
  rounding_rule → display_value`, all in `Decimal`. MAPE is a fraction shown as
  percent (`times_100`); VEI and COD are already percent (`identity`).
- **Every count names its unit.** The entry-level ED2 triple 396/279/228 and the
  unique-realization triple 387/270/219 differ by exactly 9 and may never
  co-occur in one statement.
- **Line numbers are baseline coordinates only.** Every row also carries
  `manuscript_section`, `latex_label`, `source_anchor`, and a normalized
  baseline excerpt with its sha256.
- **Certification comes from the frozen tags**, not from reruns in this
  worktree. See `certification/CERTIFICATION.md`.
- **No Tier-B0 output claims CSV bit-exactness.** P1 records ~3.55e-15 float
  round-trip, so values are carried as literal decimal text.

## Coverage, as measured

846 numeric tokens in the ACTIVE build (404 distinct):
**300 sourced**, **190 allowlisted** with a closed-vocabulary reason,
**356 flagged unsupported**.

The flagged set is confined to eight blocks, and each has a remedial claim:
the Linear-regression columns of both baseline tables; the three legacy anchors'
accuracy metrics and the `Delta_NL`/`dCor` columns of both path tables; and the
whole `tab:rho_candidate_regions` / `tab:transition_summary` /
`tab:transition_regret` family, whose numbers no frozen artifact reproduces.

## Regenerating

```bash
cd ~/RA/soft-vertical-equity-paper-integration
PY=/home/nacevedo/.conda/envs/fairness_env/bin/python
B0=analysis/final_manuscript_evidence

$PY $B0/tests/run_all_tests.py          # must be 100%

# deterministic, offline rebuild — the working tree must not change
for s in b0_1_build_manifest b0_1b_build_certification b0_2_build_numeric_map \
         b0_3_build_claim_map b0_4_build_bib b0_5_build_tf_disposition \
         b0_6_build_documents; do $PY $B0/code/$s.py; done
git diff --stat                           # must be empty
```

`bib/fetch_metadata.py` is the **only** file that touches the network, it lives
outside `code/` deliberately, and no builder or test imports it. It is run by
hand; everything else reads the cached responses under `bib/metadata_cache/`.

## Discipline

Failures are investigated at the cause and never silenced by weakening an
assertion. If a frozen artifact cannot support a claim, the **claim** is flagged;
the **test** stays strict.

The certification record pre-declares what the frozen suites do here, so that a
different outcome is a finding rather than something to explain away afterwards.
That rule earned its keep immediately: the first measured P0 rerun gave 140/153,
not the pre-declared 152/153, and **the pre-declaration was what was wrong** —
152/153 was measured in the main working tree, where the gitignored prediction
trees exist. The record was corrected to the measured outcome and the 13 failures
decomposed into three environment classes, none of them a content defect. See
`certification/CERTIFICATION.md`.

Three defects in the audit itself were found this way
and are recorded in `SELECTOR_GRAMMAR.md` §9, because each one silently shrank
the audited population: `\\[2mm]` being read as a display-math opener, an
unmatched `$` running to end of file, and `\textcolor{impGreen}{...}` being
masked as a whole call rather than only its colour argument.

## Out of scope

No refit, no LightGBM, no Slurm, no CV, no new rho, no matched-beta
recomputation. No edits to the frozen P0/P1 artifacts,
`paper/paper_v17_option1.tex`, the committed PDF, or the figures. No replacement
manuscript prose. No `output/` or `data/` reads. The external-benchmark stream
is `OUT_OF_SCOPE_FOR_B0` and enters no manifest. Tier B is not started.
