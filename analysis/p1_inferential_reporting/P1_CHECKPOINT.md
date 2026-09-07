# P1 CHECKPOINT — inferential reporting

> **SUPERSEDED — P1 IS COMPLETE.** This file is retained as the historical mid-stage
> record. The authoritative final documents are
> `reports/MANUSCRIPT_EVIDENCE_PACKAGE.md`, `reports/MANUSCRIPT_IMPACT_MEMO.md`,
> `provenance/p1_artifact_hashes.json` (regenerated at the freeze) and
> `provenance/p1_headline_numbers.json`. Where this file disagrees with those, they govern.
> The hash table in §3 below is the *checkpoint-time* extract and is now stale by design.

**Written:** 2026-09-07 (session stop point, by instruction)
**Finalized:** 2026-09-07 (all four tasks complete and asserted)
**Branch:** `p1-inferential-reporting`
**Base:** tag `p0-major-revision-final-20260907` = `805c426e1587972a2a07dcaf60220603397c0d3e`

P0 is **FINAL and immutable**. Nothing under `analysis/p0_major_revision_validation/`
or any protected path was modified. Verified below.

---

## 1. Status

| Task | Status |
|---|---|
| Phase 0 — P0 closure + tag | **COMPLETE** |
| Display-set freeze | **COMPLETE** |
| Task 1 — dCor estimator documentation | **COMPLETE** |
| Task 2 — PRB inferential reporting | **COMPLETE AND VERIFIED** |
| Task 3 — VEI ED2 inferential reporting | **COMPLETE AND VERIFIED** (28/28) |
| Task 4 — Duan smearing: freeze + estimate | **COMPLETE** |
| Task 4 — Duan smearing: apply + invariance audit | **COMPLETE AND VERIFIED** (19/19, 0 flags) |
| Final evidence freeze | **COMPLETE** |
| `MANUSCRIPT_IMPACT_MEMO.md` | **COMPLETE** |
| Manuscript edits | **NONE. Not permitted.** |

**Final:** 72/72 P1 assertions pass (64 scientific + 8 report-consistency).
P0 is **153/153 at the tag**, verified in an isolated detached worktree; the P1 branch
reports the single expected HEAD-relative guard failure. See the evidence package.

> **Disclosure.** Task 3 (VEI) and the Task 4 freeze+estimate steps completed
> *before* the instruction to stop arrived. Their artifacts exist on disk and are
> listed below. They were **not** deleted (that would destroy verified work), but
> **no** VEI-specific assertion suite has been written yet, so Task 3 is recorded
> as *computed, not yet asserted*. The smearing **apply** step was not run.

---

## 2. Task 2 — PRB, scientifically complete

- 480 rows; **440 attained, 40 NOT_ATTAINED**; all 10 evaluation blocks present (48 rows each)
- max unweighted `|P1 OLS − canonical PRB|` = **1.1790152187884928e-12**
- max frozen reconciliation discrepancy = **7.632783294297951e-16** (234 comparisons)
- D3 pooled-OOF reweighting effect = **0.0027225657356132**, reported as a *sensitivity*, not an error
- **12/12** PRB assertions pass (`tests/test_p1_prb_assertions.py`)
- classification uses four states; a CI that merely crosses a threshold is `overlaps_pm005`, never evidence of exceeding it
- **no fold-as-IID aggregation anywhere** — asserted

Counts (all rows / standards-facing rows):
`outside_pm005_but_not_pm010` 193/44 · `within_pm005` 186/33 · `overlaps_pm005` 40/8 ·
`outside_pm010` 21/3 · `NOT_ATTAINED` 40/8

**Do not rerun PRB.**

### Reporting-layer repair (no scientific recomputation) — R-P1-1

The first run wrote `prb_inference.csv` (480 rows) successfully, then raised
`TypeError: '<' not supported between instances of 'NoneType' and 'str'` while
serialising the **summary JSON** — `json.dumps(sort_keys=True)` on a `class_counts`
dict mixing a `None` key (the 40 NOT_ATTAINED rows, null by design) with strings.
The failure occurred strictly **after** the scientific table was written.

Repair: map the `None` key to `"NOT_ATTAINED"`, and add a `--summary-only` entry
point that regenerates the summary **from the already-written CSV**. **No PRB
inference was recomputed; no prediction artifact was re-read; `prb_inference.csv`
was not rewritten.**

A second reporting-layer correction split one conflated field into two:
`reconciliation_unweighted_rows.max_abs` (must be ~0) and
`d3_pooled_oof_reweighting_effect.max_abs` (a real sensitivity). No value in
`prb_inference.csv` changed.

The last `--summary-only` invocation completed normally (full expected output, no
error), and `prb_inference_summary.json` parses cleanly with 16 top-level keys.

---

## 3. Artifact hashes (SHA256)

> **Authoritative machine-readable index:** `provenance/p1_artifact_hashes.json`,
> regenerated at the end of the checkpoint over every *tracked* P1 file. The table
> below is a human-readable extract; where the two differ, the JSON governs.

### Committed evidence

| file | bytes | sha256 |
|---|---|---|
| `configs/display_set_frozen.json` | 107181 | `b050a2bbd08b09ac83b3cbce0678eb4e54cd27099f7b1072e4f0fba8c8bcd6a8` |
| `configs/display_set_frozen_hash.json` | 180 | `f30b79763d0094a5a5d114621fa9aeb5c186a993e65ea3f3f1dd6e1d26964d52` |
| `configs/ed2_vei_procedure.json` | 11702 | `5bf62c9de6b22d906abd299d54694b3389ef8e9229301bbd964f1ee85e813b63` |
| `configs/smearing_estimator_frozen.json` | 3598 | `6e00c9fa14b3fa5813be00691d48b36fb6917110b6cd8faa0ff9201fbd36bf18` |
| `configs/smearing_estimator_frozen_hash.json` | 225 | `132247499af161d1b7d67e4e438ddeb6f781b8890cc3c0baa56dfb4c42fd8dbb` |
| `provenance/DEVIATIONS_P1.md` | *(see `provenance/p1_artifact_hashes.json`)* | — |
| `provenance/ed2_source_manifest.json` | 2856 | `63306a00dc6aeaf4198a3397966d3f4f7096edbece345b2518553632d294eb82` |
| `tables/dcor_estimator_audit.csv` | 1004 | `735d26066d16f5a34b92da3b28fb9080174fbea94fd1f3d69ef401d0c5c6db45` |
| `tables/dcor_estimator_facts.json` | 3553 | `175bf4ca7eb597ecfb47872a2288766d36cbdaafdf5faaa0eb77761503463a94` |
| **`tables/prb_inference.csv`** | **250523** | **`fb7e8f117c7916d0a22f14f4c2e7f00d1e84eceaff63c12a4edd5833f7ef926a`** |
| **`tables/prb_inference_summary.json`** | **2999** | **`1458c8466af0a3ec7f1e961d9ecb039ff2cb1a38d77cf35acb39748a011ed0b9`** |
| `tables/smearing_factor_provenance.csv` | 12295 | `54a2c4332dbb23f9f3e533bc405a2c83a74612a8f4ae56625df6311116986d58` |
| `tables/vei_significance.csv` | 532784 | `8e27d0ae0bd25d337ff078f79273de9205f668134883a441119ffb6c622fa757` |
| `tables/vei_significance_summary.json` | 2734 | `2a0f93a965153741d9cb98ffee1d00aca9cdf42401fa207c21df1141759a5bab` |
| `code/p1_common.py` | 12459 | `9a2d175400c950a34542d622af70e3225daae1d3c40c92283bad4e609eaaa1e4` |
| `code/p1_0_freeze_display_set.py` | 10274 | `7832b9997063aeeeb31704c70975066bc4db4bc5e120d02e3610596f0d19ada2` |
| `code/p1_1_dcor_documentation.py` | 8219 | `182aba42bd93c55d0fa702874eded02ab6bf56f9a01915ba5a8d0de1ba62b1c5` |
| `code/p1_2_prb_inference.py` | 18537 | `2c84f194b71fead97e6877bc9d61e300be33c2ac2987487e6d162b0ca871609d` |
| `code/p1_3_vei_ed2_inference.py` | 16461 | `490291ce372e2373730da6601bb3a874a4426526f2f9a98470ba5c51b8e88280` |
| `code/p1_4_smearing_sensitivity.py` | 9718 | `dce53aad45000831c9a4931e23d8b9036992a64ec0b89a9f31a52046697eaa33` |
| `tests/test_p1_prb_assertions.py` | 5825 | `ed3d5b60cbecf2dfb2ac873262ccb6acdb025c4e06796cf5aff9f784075e3b51` |
| `tests/test_smearing_sign.py` | 5568 | `7f254a3f340140d2e0e777fabf1c3f2b3fbad536e2a344a4a0409336288150fc` |

Parquet twins exist alongside the CSVs but are gitignored by design (mirroring P0).

### ED2 source — NOT committed

| field | value |
|---|---|
| landing page | https://www.iaao.org/about/board-of-directors/governing-documents/ratio-studies-exposure-draft/ |
| PDF URL | https://www.iaao.org/wp-content/uploads/StandardonRatioStudies_Exposure-Mar2026.pdf |
| **SHA256** | **`e950e00d0c3684dd067734d401e5278dcf659f471fcfb0d6db65c7584f7c56b8`** |
| bytes | 2 013 844 |
| retrieved (UTC) | 2026-09-07T02:04:55Z |
| title page | **"STANDARD ON RATIO STUDIES / Exposure Draft May 2026"** (despite the `Mar2026` filename — resolves bibliography item P1-12) |
| pages | 103; App. D.2 p.67, App. E pp.78–81 |
| local cache | `provenance/ed2_source_cache/` — **gitignored; the PDF is NOT redistributed** |

Redistribution permission has **not** been established, so only
`provenance/ed2_source_manifest.json` is committed.

### Display set

`configs/display_set_frozen.json` sha256 `b050a2bb…` — 48 entries, **44 attained,
4 NOT_ATTAINED**, 43 distinct realizations. NOT_ATTAINED preserved exactly as
frozen: Direct at −0.06 / −0.03 / 0.00, Surrogate at 0.00.

---

## 4. P0 immutability — verified at this checkpoint

- `git diff p0-major-revision-final-20260907 -- analysis/p0_major_revision_validation/` → **empty**
- protected paths (`paper/ utils/ soft_constrained_models/ scripts/ run_temporal_cv.py
  output/paper_v6_preselection_994/ output/paper_v12_*`) → **clean**
- Stage-3B hash index: **86/86 byte-exact**
- HEAD == tag commit `805c426e…`

### The P0 suite is 153/153 AT THE TAG, and 152/153 on the P1 branch

The plan's premise — that putting P1 in a sibling directory would keep the P0
suite at 153/153 — was **wrong**, in two ways. Both are properties of the P0
guards, not P0 regressions. P0's *content* is provably unchanged.

1. `test_p0_assertions.py:222::test_all_stage1_writes_are_inside_approved_locations`
   fails on **any dirty working-tree entry** outside the two P0 roots, including an
   *untracked* `analysis/p1_inferential_reporting/`. Committing fixes this one.
2. `test_g2_assertions.py:325::test_only_gitignore_modified_outside_p0` is
   **HEAD-relative**: it runs `git diff --name-only 2732e653~1 HEAD` and asserts
   that the only file changed outside `analysis/p0_major_revision_validation/` is
   `.gitignore`. **Any** additive commit anywhere in the repository breaks it — a
   sibling directory does not help, and committing makes it fail rather than pass.

Measured:

| ref | files outside P0 since `2732e653~1` | P0 suite |
|---|---|---|
| tag `p0-major-revision-final-20260907` | `['.gitignore']` | **153 passed, 0 failed** |
| branch `p1-inferential-reporting` @ `ccff55f0` | 25 (`.gitignore` + the 24 P1 files) | **152 passed, 1 failed** |

**Correct framing.** The P0 freeze is certified **at the tag**, where the suite is
153/153. That is what the tag means and it does not change. On the P1 branch the
suite is 152/153 and the single failure is that scope guard correctly registering
that P1 exists — which is its designed behaviour, since its purpose was to stop
Stage-1/2/3 writing outside the P0 area. It is **not** evidence of a P0 change:

- `git diff p0-major-revision-final-20260907 -- analysis/p0_major_revision_validation/` → **empty**
- Stage-3B hash index → **86/86 byte-exact**
- all protected paths → **clean**

**To reproduce the 153/153 certification**, run the suite at the tag (e.g. in a
detached worktree), not on the P1 branch.
**Re-run the suite after checkout to confirm 153/153** (command below).

### `.gitignore`

The root `.gitignore` ignores `*.csv` / `*.json` globally with a narrow un-ignore
block for the P0 area. A **mirrored P1 block** was appended, or every P1 artifact
would have remained untracked. `.gitignore` is not a protected path, and there is
precedent for it being the one file changed outside the P0 area. Verified after the
change: the ED2 PDF is still ignored; parquet twins still ignored; all evidence
files trackable.

---

## 5. Environment

`fairness_env` **unmodified**: numpy 1.26.4, pandas 2.3.1, scipy 1.13.1,
scikit-learn 1.6.1, pyarrow 14.0.1, lightgbm 4.6.0, **dcor 0.6**, Python 3.9.19.
`pypdf 5.9.0` was installed into an isolated scratch `--target` directory purely to
read the ED2 PDF and is **not** importable from the environment.

---

## 6. Exact commands to resume

```bash
cd /orcd/home/002/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal
git checkout p1-inferential-reporting
PY=/home/nacevedo/.conda/envs/fairness_env/bin/python

# 0. confirm P0 is still final and the tree is clean (expect 153 passed, 0 failed)
$PY analysis/p0_major_revision_validation/tests/run_all_tests.py | tail -3
git diff --stat p0-major-revision-final-20260907 -- analysis/p0_major_revision_validation/

# 1. re-verify what is already done (no recomputation)
$PY analysis/p1_inferential_reporting/tests/test_p1_prb_assertions.py   # expect 12 passed
$PY analysis/p1_inferential_reporting/tests/test_smearing_sign.py       # expect 5 passed

# 2. NEXT: write and run VEI assertions for Task 3 (artifacts already computed)
#    -> analysis/p1_inferential_reporting/tests/test_p1_vei_assertions.py

# 3. THEN: Task 4 apply + invariance audit (NOT yet run)
$PY analysis/p1_inferential_reporting/code/p1_4_smearing_sensitivity.py --mode apply

# 4. ONLY AFTER Task 3 is asserted complete from the authoritative ED2 source:
#    the final evidence freeze and MANUSCRIPT_IMPACT_MEMO.md
```

**Re-running is not needed and is not advised for:** the display-set freeze
(`p1_0`), dCor (`p1_1`), PRB (`p1_2`), or the smearing freeze/estimate
(`p1_4 --mode freeze|estimate`). The smearing estimator config is hash-pinned;
re-running `--mode freeze` would rewrite it with a new timestamp and break the
pin ordering.

---

## 7. Gate on the final evidence freeze — SATISFIED

The gate required Task 3 to be complete and asserted from the authoritative ED2
source before the evidence package and impact memo could be produced. It is:
`tests/test_p1_vei_assertions.py` passes 28/28, re-deriving the App. D.2 rule
independently and verifying the source PDF hash; the pooled-OOF non-applicability was
adjudicated against the literal document (see `provenance/DEVIATIONS_P1.md` D-P1-9).

Both documents now exist under `reports/`. **No manuscript file has been touched** —
`git diff` over `paper/` against the tag is empty, and that is asserted by
`tests/test_p1_headline_numbers.py`.
