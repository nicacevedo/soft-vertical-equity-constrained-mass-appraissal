#!/usr/bin/env python3
"""Minimal runner for the Stage-1 assertions (pytest is not installed in fairness_env)."""
from __future__ import annotations
import sys, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

class _Raises:
    def __init__(self, exc): self.exc = exc
    def __enter__(self): return self
    def __exit__(self, t, v, tb):
        if t is None:
            raise AssertionError(f"expected {self.exc.__name__} to be raised")
        return issubclass(t, self.exc)

class _Skip(Exception): pass
class _PytestShim:
    @staticmethod
    def raises(exc): return _Raises(exc)
    @staticmethod
    def skip(msg=""): raise _Skip(msg)
sys.modules["pytest"] = _PytestShim  # type: ignore

import test_p0_assertions as T  # noqa: E402

def main() -> int:
    names = sorted(n for n in dir(T) if n.startswith("test_"))
    npass = nfail = nskip = 0
    fails = []
    for n in names:
        try:
            getattr(T, n)()
            print(f"  PASS  {n}"); npass += 1
        except _Skip as e:
            print(f"  SKIP  {n} ({e})"); nskip += 1
        except Exception as e:
            print(f"  FAIL  {n}: {type(e).__name__}: {e}"); nfail += 1
            fails.append((n, traceback.format_exc()))
    print(f"\n{npass} passed, {nfail} failed, {nskip} skipped, {len(names)} total")
    for n, tb in fails:
        print(f"\n--- {n} ---\n{tb}")
    return 1 if nfail else 0

if __name__ == "__main__":
    raise SystemExit(main())
