#!/usr/bin/env python3
"""Tier-B validation primitives: geography, guards, frozen-module access.

This package is the gate for the Tier-B writing pass. It lives under ``paper/``
because it is Tier-B-owned, and it is built at B1.0 -- BEFORE any manuscript
edit -- so every later stage has a gate to run against.

Two hard rules, enforced mechanically here rather than by convention:

  * It **reads** the frozen P0 / P1 / Tier-B0 evidence and **never writes**
    anything under ``analysis/``. ``guard_write`` refuses any path outside this
    validator's own directory.
  * It **never** re-runs science. No model fit, no CV, no new rho value, no
    recomputation of matched-beta. Every number it checks is re-derived from a
    frozen artifact by selector, exactly as Tier B0 derived it.

Why it exists at all: the frozen Tier-B0 and P1 suites carry paper-immutability
and HEAD-relative guards that fail *by design* the moment the manuscript is
intentionally edited (``test_frozen_stages_and_paper_are_untouched``,
``test_no_protected_path_written``, the pinned ``TEX_SHA256``). Requiring them to
stay green is incoherent for a writing pass, and modifying a frozen test is
prohibited. So the frozen suites are run once, informationally, and this
validator is the actual gate.

Decimal discipline: every value comparison routes through the frozen
``b0_common`` value pipeline, which carries literal decimal text and converts
with ``decimal.Decimal``. Binary float arithmetic is never used on a
manuscript-facing number -- P1's committed CSVs round-trip float64 only to about
3.55e-15, so a float detour could silently change a displayed digit.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

# Importing the frozen builders must not drop __pycache__ into analysis/:
# that would be a write outside paper/ and would fail the write-scope check.
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent           # paper/paper_analysis/tier_b_validation
PAPER_ANALYSIS = HERE.parent
PAPER = PAPER_ANALYSIS.parent
REPO = PAPER.parent

ANALYSIS = REPO / "analysis"
P0 = ANALYSIS / "p0_major_revision_validation"
P1 = ANALYSIS / "p1_inferential_reporting"
B0 = ANALYSIS / "final_manuscript_evidence"
B0_CODE = B0 / "code"
B0_SPEC = B0 / "spec"

TEX = PAPER / "paper_v17_option1.tex"
BIB_MAIN = PAPER / "references.bib"
BIB_ADDITIONS = PAPER / "references_additions.bib"

TB_SPEC = HERE / "spec"
TB_LEDGER = HERE / "ledger"
TB_BASELINE = HERE / "baseline"

# Frozen coordinates. These are read, never rewritten.
P0_TAG = "p0-major-revision-final-20260907"
P1_TAG = "p1-inferential-reporting-final-20260907"
B0_TAG = "tier-b0-final-20260907"
B0_COMMIT = "904976451bdd01aa3e6c8b59d4eee77d6f10ba74"

FROZEN_SUBTREES = (
    (P0_TAG, "analysis/p0_major_revision_validation"),
    (P1_TAG, "analysis/p1_inferential_reporting"),
    (B0_TAG, "analysis/final_manuscript_evidence"),
)

# The Tier-A baseline manuscript hash, quoted from the frozen b0_common. The live
# manuscript is EXPECTED to diverge from it from B1.1 onward; it is recorded so
# the divergence is deliberate and visible rather than accidental.
BASELINE_TEX_SHA256 = "13c84ce7e799d485cf33e20a96a53e1f7ff30ecbb505a2b7042f76fd124de19a"

STAGES = ("B1.0", "B1.1", "B1.2", "B1.3", "B1.4",
          "B2.1", "B2.2", "B2.3", "B2.4", "B2.5", "B2.6",
          "B3.1", "B3.2", "B4.1", "B4.2", "B4.3")


class TierBGuardError(RuntimeError):
    """A write left the region the Tier-B validator is permitted to touch."""


def _under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root)
        return True
    except ValueError:
        return False


def guard_write(path) -> Path:
    """Refuse any write inside the repository other than into this directory.

    Scratch paths outside the repository are fine -- they cannot violate the
    cumulative write-scope check. What must be impossible is a write anywhere
    else *in the repo*: under analysis/ above all, but equally under output/,
    utils/ or a stray new directory, since the binding scope rule is that every
    path changed since the Tier-B0 tag lies under paper/.
    """
    p = Path(path).resolve()
    if _under(p, HERE):
        return p
    if not _under(p, REPO):
        return p
    raise TierBGuardError(
        f"the Tier-B validator writes only under {HERE} (or outside the "
        f"repository); refused: {p}\n"
        "Nothing under analysis/ may be modified by the writing pass, and "
        "nothing outside paper/ may be modified at all.")


def write_text(path, text: str) -> Path:
    p = guard_write(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def write_json(path, obj) -> Path:
    return write_text(path, json.dumps(obj, indent=2, sort_keys=True,
                                       ensure_ascii=False) + "\n")


def read_text(path) -> str:
    return Path(path).read_text(encoding="utf-8")


def read_json(path):
    return json.loads(read_text(path))


def read_yaml(path):
    import yaml
    return yaml.safe_load(read_text(path))


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def rel(path) -> str:
    return Path(path).resolve().relative_to(REPO).as_posix()


def git(*args, check: bool = True) -> str:
    r = subprocess.run(("git", "-C", str(REPO)) + args,
                       capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout


# --------------------------------------------------------------------------
# Frozen Tier-B0 modules. Imported, not copied: the coverage audit must be the
# SAME algorithm Tier B0 certified, applied to the LIVE manuscript.
# --------------------------------------------------------------------------
def frozen_b0():
    """Return (b0_common, b0_tex) with the frozen code path on sys.path."""
    p = str(B0_CODE)
    if p not in sys.path:
        sys.path.insert(0, p)
    import b0_common          # noqa: E402
    import b0_tex             # noqa: E402
    return b0_common, b0_tex


def live_tex():
    """The live manuscript, partitioned by the frozen Tier-B0 algorithm."""
    _, b0_tex = frozen_b0()
    return b0_tex.Tex(TEX)


# Frozen specs the validator is data-driven from, rather than duplicating.
def frozen_numeric_map() -> list:
    import csv
    with open(B0 / "manuscript_numeric_map.csv", newline="",
              encoding="utf-8") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def frozen_claim_map() -> list:
    import csv
    with open(B0 / "manuscript_claim_map.csv", newline="",
              encoding="utf-8") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def frozen_allowlist() -> list:
    import csv
    with open(B0_SPEC / "unsupported_numbers_allowlist.csv", newline="",
              encoding="utf-8") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def frozen_coverage_policy() -> dict:
    return read_yaml(B0_SPEC / "coverage_policy.yaml")


def frozen_forbidden_wording() -> dict:
    return read_yaml(B0_SPEC / "forbidden_wording.yaml")


def frozen_todo_closure() -> dict:
    return read_yaml(B0_SPEC / "todo_closure.yaml")


def frozen_tf_disposition() -> dict:
    return read_yaml(B0_SPEC / "table_figure_disposition.yaml")


def frozen_manifest() -> dict:
    return read_json(B0 / "FINAL_EVIDENCE_MANIFEST.json")


def frozen_coverage_summary() -> dict:
    return read_json(B0 / "coverage" / "coverage_summary.json")
