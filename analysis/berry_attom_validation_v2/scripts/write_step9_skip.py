#!/usr/bin/env python3
"""Step 9: ATTOM vs Berry assessment concordance — skip unless dictionary-defensible."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "analysis/berry_attom_validation_v2/source_concordance"
DICT_DIR = ROOT / "data/dewey-downloads/data_dictionaries"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(p.name for p in DICT_DIR.iterdir() if p.is_file()) if DICT_DIR.exists() else []
    has_history_dict = any("history" in n.lower() or "assessor" in n.lower() for n in files)
    text = f"""# Step 9 — ATTOM vs Berry assessment concordance

**Status: SKIPPED**

Primary validation in v2 is Berry-full vs Berry-on-ATTOM-linkable observations
(Step 8). That comparison uses the Berry official assessment/sale ratio only.

This optional ATTOM-vs-Berry *assessment-field* comparison is skipped because:

1. The current Dewey delivery does not include a Tax Assessor extract.
2. Available dictionaries in `data/dewey-downloads/data_dictionaries/` are
   Recorder-focused: {files}.
3. No dictionary in this delivery demonstrates that Assessor History
   `TAXASSESSEDVALUETOTAL` or `TAXMARKETVALUETOTAL` is conceptually or
   timing-compatible with the Berry local assessment measure
   (Detroit `Asd. when Sold`; Philadelphia `assmt_at_sale`).
4. Year-end `ASSESSORHISTORYYEAR` snapshots are not the same object as a
   jurisdiction's official ratio-study assessed value as of the sale.

Those tax-value columns are retained in the county history cache **only** as
an audit field. They are **not** AVM predictors and are **not** treated as
Berry assessment substitutes.

If a later delivery includes an Assessor History data dictionary that
establishes sale-time compatibility, this comparison can be added as a
separately labeled sensitivity without revising the frozen v2 protocol's
primary estimands.
"""
    (OUT / "STEP9_ASSESSMENT_CONCORDANCE.md").write_text(text, encoding="utf-8")
    print("wrote", OUT / "STEP9_ASSESSMENT_CONCORDANCE.md", "has_history_dict", has_history_dict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
