#!/usr/bin/env python3
"""Run both P0 assertion suites (pytest is not installed in fairness_env)."""
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

import test_p0_assertions as S1          # noqa: E402
import test_g2_assertions as G2          # noqa: E402

def run(mod, label):
    names = sorted(n for n in dir(mod) if n.startswith("test_"))
    npass = nfail = nskip = 0; fails = []
    print(f"\n===== {label} ({len(names)} tests) =====")
    for n in names:
        try:
            getattr(mod, n)(); print(f"  PASS  {n}"); npass += 1
        except _Skip as e:
            print(f"  SKIP  {n} ({e})"); nskip += 1
        except Exception as e:
            print(f"  FAIL  {n}: {type(e).__name__}: {e}"); nfail += 1
            fails.append((n, traceback.format_exc()))
    return npass, nfail, nskip, fails

def main() -> int:
    tot = [0, 0, 0]; allf = []
    for mod, lbl in ((S1, "Stage-1 assertions"), (G2, "Stage-1.5 / Gate-G2 assertions")):
        p, f, s, fl = run(mod, lbl)
        tot[0] += p; tot[1] += f; tot[2] += s; allf += fl
    print(f"\nTOTAL: {tot[0]} passed, {tot[1]} failed, {tot[2]} skipped")
    for n, tb in allf:
        print(f"\n--- {n} ---\n{tb}")
    return 1 if tot[1] else 0

if __name__ == "__main__":
    raise SystemExit(main())
