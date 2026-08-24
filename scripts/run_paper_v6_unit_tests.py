"""Run focused paper-v6 unit tests without requiring pytest."""

from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

(Path(REPO) / "tests" / "__init__.py").touch()

MODULES = [
    "tests.test_canonical_grid",
    "tests.test_canonical_metrics",
    "tests.test_canonical_objectives",
    "tests.test_paper_v6_guards",
]


def main() -> int:
    failures = []
    passed = 0
    for mod_name in MODULES:
        mod = importlib.import_module(mod_name)
        for name in sorted(dir(mod)):
            if not name.startswith("test_"):
                continue
            fn = getattr(mod, name)
            if not callable(fn):
                continue
            try:
                fn()
                print(f"PASS {mod_name}.{name}")
                passed += 1
            except Exception as exc:
                print(f"FAIL {mod_name}.{name}: {exc}")
                traceback.print_exc()
                failures.append(f"{mod_name}.{name}")
    print(f"{passed} PASS / {len(failures)} FAIL")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
