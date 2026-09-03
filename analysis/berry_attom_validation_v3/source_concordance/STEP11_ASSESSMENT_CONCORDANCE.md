# Step 11 — ATTOM vs Berry assessment concordance (v3)

**Status: SKIPPED**

Copied v2 skip rationale is in `STEP9_ASSESSMENT_CONCORDANCE.md`.

v3 does not assume `TAXASSESSEDVALUETOTAL` or `TAXMARKETVALUETOTAL` is the same
legal/timing construct as Berry/local assessed value. Dictionaries in this
delivery remain Recorder-focused and do not establish sale-time compatibility.

Primary validation remains:

- Berry/local full eligible sample
- vs Berry/local metrics on the ATTOM-linkable nested subsets

Official assessment ratios are never equated with AVM valuation ratios.
Those tax-value columns may exist in the History cache as audit fields only.
They are excluded from HISTORY_MARKET_CORE / HISTORY_STRUCTURAL_CORE.
