#!/usr/bin/env python3
"""Run v3 assertion tests without pytest (fairness_env does not include pytest)."""
from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path

TEST = Path(__file__).resolve().parents[1] / "tests" / "test_v3_assertions.py"


def main() -> int:
    spec = importlib.util.spec_from_file_location("test_v3_assertions", TEST)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    names = [n for n in dir(mod) if n.startswith("test_")]
    failed = []
    for name in names:
        try:
            getattr(mod, name)()
            print("ok", name)
        except Exception as exc:
            failed.append(name)
            print("FAIL", name, type(exc).__name__, exc)
            traceback.print_exc()
    print("passed", len(names) - len(failed), "failed", len(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
