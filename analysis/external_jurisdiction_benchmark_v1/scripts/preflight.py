#!/usr/bin/env python3
"""Step 0: repo/protocol preflight. Records repo state and implementation
hashes BEFORE any model outcome. Writes inventory/preflight.json.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from analysis.external_jurisdiction_benchmark_v1.scripts.v1_common import (  # noqa: E402
    ANALYSIS, sha256_file, write_json,
)

DEPENDENCY_FILES = (
    "soft_constrained_models/boosting_models.py",
    "scripts/other_counties_benchmars.py",
    "scripts/theory_informed_rho_range_v2.py",
    "utils/motivation_utils.py",
    "utils/delta_nl.py",
    "utils/rho_screening_v2.py",
    "utils/paper_v12_lower_rho_plots.py",
    "scripts/analyze_rho_screening_region_v2_1.py",
    "analysis/berry_cmf_validation/scripts/reproduce_berry.py",
    "analysis/berry_attom_validation_v3/scripts/link_berry_attom.py",
    "tests/test_paper_v6_guards.py",
    "tests/test_canonical_objectives.py",
)


def run(cmd: list[str]) -> str:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()


def main() -> int:
    rec = {
        "written_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repo_root": str(ROOT),
        "branch": run(["git", "branch", "--show-current"]),
        "head_sha": run(["git", "rev-parse", "HEAD"]),
        "git_status_short": run(["git", "status", "--short"]),
        "python_executable": sys.executable,
        "python_version": sys.version,
    }
    try:
        import lightgbm, numpy, pandas, sklearn
        rec["package_versions"] = {
            "lightgbm": lightgbm.__version__, "numpy": numpy.__version__,
            "pandas": pandas.__version__, "scikit_learn": sklearn.__version__,
        }
    except Exception as exc:
        rec["package_versions_error"] = str(exc)
    rec["dependency_hashes"] = {}
    for rel in DEPENDENCY_FILES:
        p = ROOT / rel
        rec["dependency_hashes"][rel] = sha256_file(p) if p.exists() else "MISSING"
    missing = [k for k, v in rec["dependency_hashes"].items() if v == "MISSING"]
    if missing:
        rec["MISSING_DEPENDENCIES"] = missing
    ANALYSIS.joinpath("audits").mkdir(parents=True, exist_ok=True)
    write_json(ANALYSIS / "audits" / "preflight.json", rec)
    print(json.dumps({"branch": rec["branch"], "head_sha": rec["head_sha"],
                       "n_dependency_files": len(DEPENDENCY_FILES),
                       "n_missing": len(missing)}, indent=2))
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
