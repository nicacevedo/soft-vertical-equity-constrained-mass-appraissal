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

# B1.5 is a bounded post-B1.4 hardening pass, not a new scientific stage: it
# corrects wording and tightens this validator's own boundaries. It clears no
# flagged-token anchor, so its budget is B1.4's -- see expected_token_trajectory.
STAGES = ("B1.0", "B1.1", "B1.2", "B1.3", "B1.4", "B1.5",
          "B2.1", "B2.2", "B2.3", "B2.4", "B2.5", "B2.6", "B2.7", "B2.8", "B2.9",
          "B3.1", "B3.2", "B3.3", "B4.1", "B4.2", "B4.3")
# B2.7 was added after the closure audit of B2.6, which found three reader-facing
# defects the thirteen checks cannot see: a matched-beta price-scale comparison
# that reverses under the frozen retransformation sensitivity, a retained path
# overlay whose endpoints no frozen artifact reproduces, and a VEI/band sentence
# that invites the inference the Exposure Draft's own procedure declines. Adding
# a stage only ever tightens the gate: an entry resolved by an earlier stage is
# already UNEXPECTED here, and B2.7 owns no expected-failure entry of its own.
# B2.8 closes the final Tier-B presentation/provenance pass identified by the
# empirical-core closure audit: unsupported positive-rho dCor path-shape claims,
# a Results promise of a transition diagnostic the paper no longer reports, a
# Limitations clause denying the centered-spread comparator B2.4 added, two stale
# "parity rerun outstanding" statements, and the legacy gray band itself, which is
# removed from the eight active path figures by
# remove_legacy_gray_bands.py rather than only disclosed. Like B2.7 this is an
# ordered insert that tolerates nothing new: B2.8 owns no expected-failure entry,
# and the four deferrals it inherits still resolve at B3.1/B4.2/B4.3.
# B2.9 is the final Tier-B consistency closure after independent B2.8
# acceptance review: it removes one stale reference to now-removed shaded
# landmarks and one stale conditional implying the completed rho=0 parity
# audit remained pending. It introduces no new scientific result and owns
# no expected-failure entry.
# B3.3 is the surgical body-consistency / prior-art closure that follows the
# independent B3.2 final acceptance and novelty audit. It removes the superseded
# fitted-linear-baseline design story from the ACTIVE body (the finalized study
# is within-model: the historical linear workflow is context, not an estimated
# arm), narrows a small number of novelty and literature statements to what the
# cited sources support, and integrates the 2026 Cook County prior art. It
# introduces NO new science -- no fit, no rho, no rerun, no regenerated figure,
# no changed table or ledger value -- and, like B2.7--B2.9 and B3.1--B3.2, it
# owns no expected-failure entry of its own. The only failures it may carry are
# the two it inherits: C12 (TODO/scaffolding closure, B4.2) and C08 (appendix
# screening-package wording, B4.3).


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


# Tier-B-owned specs. These are NOT frozen evidence: they are this pass's own
# registries, and they are authored and maintained under paper/.
def tier_b_math_claims() -> dict:
    """Known ACTIVE unsupported numeric claims that sit inside math mode.

    The frozen coverage audit masks math environments, so such a claim is never
    in the token population and can be neither SOURCED nor FLAGGED. Without this
    registry a later FLAGGED_UNSUPPORTED = 0 would read as "every printed number
    resolves", when what it actually means is "every TEXT-MODE token resolves".
    """
    return read_yaml(TB_SPEC / "unsupported_math_claims.yaml")
