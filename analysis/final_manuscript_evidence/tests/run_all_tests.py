#!/usr/bin/env python3
"""Run every Tier-B0 assertion suite (fairness_env has no pytest).

Mirrors analysis/p0_major_revision_validation/tests/run_all_tests.py: a small
pytest shim plus plain asserts, so the suites read like pytest tests but need no
dependency.

The Tier-B0 requirement is 100%: unlike the frozen P0/P1 suites, nothing here
depends on gitignored artifacts or on mtimes, so there is no legitimate
environment-dependent failure. A failure is investigated at the cause and never
silenced by weakening an assertion. If a frozen artifact cannot support a claim,
the CLAIM is flagged; the TEST stays strict.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "code"))


class _Raises:
    def __init__(self, exc):
        self.exc = exc

    def __enter__(self):
        return self

    def __exit__(self, t, v, tb):
        if t is None:
            raise AssertionError(f"expected {self.exc.__name__} to be raised")
        return issubclass(t, self.exc)


class _Skip(Exception):
    pass


class _PytestShim:
    @staticmethod
    def raises(exc):
        return _Raises(exc)

    @staticmethod
    def skip(msg=""):
        raise _Skip(msg)


sys.modules["pytest"] = _PytestShim  # type: ignore

import test_b0_manifest as MAN            # noqa: E402
import test_b0_certification as CERT      # noqa: E402
import test_b0_numeric_map as NUM         # noqa: E402
import test_b0_claim_map as CLAIM         # noqa: E402
import test_b0_role_guards as GUARD       # noqa: E402
import test_b0_isolation as ISO           # noqa: E402


def run(mod, label):
    names = sorted(n for n in dir(mod) if n.startswith("test_"))
    npass = nfail = nskip = 0
    fails = []
    print(f"\n===== {label} ({len(names)} tests) =====")
    for n in names:
        try:
            getattr(mod, n)()
            print(f"  PASS  {n}")
            npass += 1
        except _Skip as e:
            print(f"  SKIP  {n} ({e})")
            nskip += 1
        except Exception as e:
            print(f"  FAIL  {n}: {type(e).__name__}: {e}")
            nfail += 1
            fails.append((n, traceback.format_exc()))
    return npass, nfail, nskip, fails


def main() -> int:
    tot = [0, 0, 0]
    allf = []
    for mod, lbl in ((MAN, "Manifest and artifact hashes"),
                     (CERT, "Frozen-stage certification"),
                     (NUM, "Numeric map, value pipeline and coverage"),
                     (CLAIM, "Claim map, dispositions and coverage topics"),
                     (GUARD, "Scientific-integrity role guards"),
                     (ISO, "Isolation, guards and write discipline")):
        p, f, s, fl = run(mod, lbl)
        tot[0] += p
        tot[1] += f
        tot[2] += s
        allf += fl
    print(f"\nTOTAL: {tot[0]} passed, {tot[1]} failed, {tot[2]} skipped")
    for n, tb in allf:
        print(f"\n--- {n} ---\n{tb}")
    return 1 if tot[1] else 0


if __name__ == "__main__":
    raise SystemExit(main())
