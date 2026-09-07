#!/usr/bin/env python3
"""Run the frozen Tier-B0 / P0 / P1 suites INFORMATIONALLY, and record what they say.

These suites are not the gate for a writing pass, and cannot be. Verified
directly: they carry paper-immutability and HEAD-relative guards that fail BY
DESIGN the moment the manuscript is intentionally edited --

  * b0 tests/test_b0_isolation.py::test_frozen_stages_and_paper_are_untouched
      asserts `git diff --stat TIER_A_COMMIT HEAD -- paper` is empty;
  * b0 tests/test_b0_isolation.py::test_git_status_shows_nothing_outside_the_tier_b0_area
      asserts a clean working tree outside the Tier-B0 area;
  * b0 code/b0_common.py::TEX_SHA256 pins the baseline manuscript hash, so any
      live rehash diverges;
  * P1 test_no_protected_path_written (x3 suites), whose _PROTECTED includes
      REPO/"paper";
  * P1 test_p1_headline_numbers::test_reports_state_that_no_manuscript_file_was_edited;
  * P0 test_g2_assertions::test_only_gitignore_modified_outside_p0, a
      HEAD-relative scope guard that already fails on any additive commit.

So "the frozen suites stay at 100%" is incoherent here, and MODIFYING A FROZEN
TEST TO MAKE IT PASS IS PROHIBITED. They are run once before the first edit to
record the pre-edit baseline; after that their paper-facing guards are expected
to fail, and that expectation is documented in
baseline/FROZEN_SUITE_EXPECTATIONS.md rather than fixed.

What certifies the science is what the three tags carry. What certifies the
writing is validate.py plus the scope and immutability checks.

Nothing here writes under analysis/: sys.dont_write_bytecode is set before any
frozen module is imported, so not even a __pycache__ appears.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

import tb_common as tb

B0_RUNNER = tb.B0 / "tests" / "run_all_tests.py"
P0_RUNNER = tb.P0 / "tests" / "run_all_tests.py"
P1_TESTS = tb.P1 / "tests"

P1_MODULES = ("test_p1_headline_numbers", "test_p1_prb_assertions",
              "test_p1_vei_assertions", "test_p1_smearing_apply_assertions",
              "test_smearing_sign")

# A tiny pytest shim, mirroring the one the frozen runners already use, so the
# P1 suites can run without pytest installed. It is a RUNNER, not a change to
# any frozen test: no file under analysis/ is touched.
P1_SHIM = r'''
import sys, traceback
from pathlib import Path
sys.dont_write_bytecode = True
TESTS = Path(sys.argv[1])
sys.path.insert(0, str(TESTS))
sys.path.insert(0, str(TESTS.parent / "code"))

class _Raises:
    def __init__(self, exc): self.exc = exc
    def __enter__(self): return self
    def __exit__(self, t, v, tb):
        if t is None:
            raise AssertionError(f"expected {self.exc.__name__} to be raised")
        return issubclass(t, self.exc)
class _Skip(Exception): pass
class _Shim:
    @staticmethod
    def raises(exc): return _Raises(exc)
    @staticmethod
    def skip(msg=""): raise _Skip(msg)
sys.modules["pytest"] = _Shim

mods = sys.argv[2:]
tot = [0, 0, 0]
for name in mods:
    mod = __import__(name)
    names = sorted(n for n in dir(mod) if n.startswith("test_"))
    print(f"\n===== {name} ({len(names)} tests) =====")
    for n in names:
        try:
            getattr(mod, n)(); print(f"  PASS  {n}"); tot[0] += 1
        except _Skip as e:
            print(f"  SKIP  {n} ({e})"); tot[2] += 1
        except Exception as e:
            print(f"  FAIL  {n}: {type(e).__name__}: {e}"); tot[1] += 1
print(f"\nTOTAL: {tot[0]} passed, {tot[1]} failed, {tot[2]} skipped")
raise SystemExit(1 if tot[1] else 0)
'''


def _run(cmd, cwd=None):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run(cmd, capture_output=True, text=True, env=env,
                       cwd=cwd or str(tb.REPO))
    return r.returncode, r.stdout + r.stderr


def _parse(out: str) -> dict:
    m = re.search(r"TOTAL:\s*(\d+) passed,\s*(\d+) failed,\s*(\d+) skipped", out)
    fails = re.findall(r"^  FAIL  (\S+):", out, re.M)
    return {"passed": int(m.group(1)) if m else None,
            "failed": int(m.group(2)) if m else None,
            "skipped": int(m.group(3)) if m else None,
            "failing_tests": sorted(set(fails))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True,
                    help="what this run records, e.g. pre_edit_B1.0")
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args(argv)
    out = tb.Path(a.outdir) if a.outdir else tb.TB_BASELINE
    out.mkdir(parents=True, exist_ok=True)

    shim = out / "_p1_runner.py"
    tb.write_text(shim, P1_SHIM)

    results = {}
    for name, cmd in (
            ("tier_b0", [sys.executable, str(B0_RUNNER)]),
            ("p0", [sys.executable, str(P0_RUNNER)]),
            ("p1", [sys.executable, str(shim), str(P1_TESTS)] + list(P1_MODULES)),
    ):
        code, text = _run(cmd)
        tb.write_text(out / f"{name}_suite_{a.label}.log", text)
        results[name] = {"exit_code": code, **_parse(text)}
        r = results[name]
        print(f"{name:8s} exit={code}  passed={r['passed']} failed={r['failed']} "
              f"skipped={r['skipped']}")
        for t in r["failing_tests"]:
            print(f"           FAIL {t}")
    shim.unlink(missing_ok=True)

    tb.write_json(out / f"frozen_suites_{a.label}.json", {
        "label": a.label,
        "informational_only": True,
        "why_not_the_gate": (
            "These suites carry paper-immutability and HEAD-relative guards that "
            "fail by design once the manuscript is intentionally edited. They are "
            "never a writing-stage pass/fail criterion, never modified, and never "
            "cited as certification of this pass."),
        "manuscript_sha256": tb.sha256_file(tb.TEX),
        "head": tb.git("rev-parse", "HEAD").strip(),
        "results": results,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
