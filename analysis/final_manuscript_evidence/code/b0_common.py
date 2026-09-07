#!/usr/bin/env python3
"""Tier-B0 shared primitives: path guards, hashing, selector resolution, value pipeline.

Tier B0 builds a deterministic bridge from frozen P0/P1 evidence to manuscript
claims. It runs NO experiments, fits NO models, and reads NO row-level prediction
data. Two guards enforce that mechanically:

  * READ  guard -- reads are confined to ALLOWED_READ_ROOTS. Anything resolving
    under ``output/`` or ``data/`` raises, unconditionally. Both roots exist in
    this worktree (holding unrelated smoke-test and geo material), so a
    "the directory is absent" argument would not be sound; the allowlist is.
  * WRITE guard -- writes are confined to analysis/final_manuscript_evidence/.
    A write to paper/, or to either frozen stage, raises.

Every number that reaches the manuscript map is DERIVED here from a frozen
artifact via a selector, never transcribed. Values are carried as the literal
decimal TEXT found in the artifact and converted with ``decimal.Decimal``, never
through float, so re-resolution is byte-exact and rounding is reproducible from
the recorded literal alone. (P1 records that its committed CSVs round-trip
float64 to ~3.55e-15, so no Tier-B0 output may claim CSV bit-exactness.)

No module in this package performs network I/O.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from decimal import Decimal, ROUND_HALF_UP, localcontext
from pathlib import Path

# --------------------------------------------------------------------------
# Repository geography
# --------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
B0 = HERE.parent                                  # analysis/final_manuscript_evidence
ANALYSIS = B0.parent
REPO = ANALYSIS.parent

P0 = ANALYSIS / "p0_major_revision_validation"
P1 = ANALYSIS / "p1_inferential_reporting"
PAPER = REPO / "paper"

P0_TABLES, P0_CONFIGS, P0_PROV, P0_REPORTS = (
    P0 / "tables", P0 / "configs", P0 / "provenance", P0 / "reports")
P1_TABLES, P1_CONFIGS, P1_PROV, P1_REPORTS = (
    P1 / "tables", P1 / "configs", P1 / "provenance", P1 / "reports")

TEX = PAPER / "paper_v17_option1.tex"
TEX_OPTION2 = PAPER / "paper_v17_option2.tex"
PAPER_ANALYSIS = PAPER / "paper_analysis" / "paper_v17"

SPEC, CODE, TESTS, COVERAGE, BIB, CERT = (
    B0 / "spec", B0 / "code", B0 / "tests", B0 / "coverage", B0 / "bib",
    B0 / "certification")

# Frozen stage coordinates. Recorded, and asserted by the certification test.
P0_TAG = "p0-major-revision-final-20260907"
P1_TAG = "p1-inferential-reporting-final-20260907"
P0_COMMIT = "805c426e1587972a2a07dcaf60220603397c0d3e"
P1_COMMIT = "6caf66774f2979f80d3cc74fa09c4d63b4e9f909"
TIER_A_COMMIT = "097544fff8c98bc864035674d038d9602ea667cb"
INTEGRATION_HEAD = "cd5bfce35e446d4ff2950cb2e077c0c33b5aaefd"
TEX_SHA256 = "13c84ce7e799d485cf33e20a96a53e1f7ff30ecbb505a2b7042f76fd124de19a"

# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------
ALLOWED_READ_ROOTS = (P0, P1, B0, PAPER, REPO / "configs")
FORBIDDEN_READ_ROOTS = (REPO / "output", REPO / "data")
# Streams that Tier B0 must never legitimize by indexing them as evidence.
FORBIDDEN_EVIDENCE_DIRS = (
    ANALYSIS / "external_jurisdiction_benchmark_v1",
    ANALYSIS / "berry_attom_validation_v2",
    ANALYSIS / "berry_attom_validation_v3",
    ANALYSIS / "berry_cmf_validation",
)


class GuardError(RuntimeError):
    """A read or write left the region Tier B0 is permitted to touch."""


def _under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def guard_read(path) -> Path:
    """Resolve ``path`` and raise unless it is inside an allowed read root."""
    p = Path(path).resolve()
    for bad in FORBIDDEN_READ_ROOTS:
        if _under(p, bad):
            raise GuardError(
                f"read from an unavailable legacy root is prohibited: {p}\n"
                "Legacy V6/V12 and raw-data artifacts are indexed by quoting the "
                "frozen hash indices, never by reading the files.")
    for bad in FORBIDDEN_EVIDENCE_DIRS:
        if _under(p, bad):
            raise GuardError(
                f"external-benchmark evidence is OUT_OF_SCOPE_FOR_B0: {p}")
    if not any(_under(p, root) for root in ALLOWED_READ_ROOTS):
        raise GuardError(f"read outside the Tier-B0 allowed roots: {p}")
    return p


def guard_write(path) -> Path:
    """Resolve ``path`` and raise unless it is inside the Tier-B0 output area."""
    p = Path(path).resolve()
    if not _under(p, B0):
        raise GuardError(
            f"Tier B0 writes only under {B0}; refused: {p}")
    return p


# --------------------------------------------------------------------------
# Hashing and guarded I/O
# --------------------------------------------------------------------------
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path) -> str:
    p = guard_read(path)
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_text(path) -> str:
    return guard_read(path).read_text(encoding="utf-8")


def read_json(path):
    return json.loads(read_text(path))


def read_yaml(path):
    import yaml
    return yaml.safe_load(read_text(path))


def read_csv_literal(path) -> list:
    """Read a CSV as a list of dicts of LITERAL TEXT. No float conversion, ever."""
    p = guard_read(path)
    with open(p, newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def write_text(path, text: str) -> Path:
    p = guard_write(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def write_json(path, obj) -> Path:
    """Deterministic JSON: sorted keys, 2-space indent, trailing newline."""
    return write_text(path, json.dumps(obj, indent=2, sort_keys=True,
                                       ensure_ascii=False) + "\n")


def write_csv(path, fieldnames, rows) -> Path:
    """Deterministic CSV: fixed column order, \\n line endings, no index."""
    p = guard_write(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fieldnames), lineterminator="\n",
                           extrasaction="raise")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fieldnames})
    return p


def rel(path) -> str:
    """Repo-relative POSIX path, the form recorded in every Tier-B0 artifact."""
    return Path(path).resolve().relative_to(REPO).as_posix()


# --------------------------------------------------------------------------
# Selector grammar (normative definition lives in SELECTOR_GRAMMAR.md)
# --------------------------------------------------------------------------
FLOAT_TOL_REL = Decimal("1e-12")


class SelectorError(RuntimeError):
    """A selector failed to resolve to exactly one value."""


def _parse_csv_selector(selector: str):
    """``col=val;col2~=1.5`` -> [(col, op, val), ...]  op in {'=', '~='}."""
    clauses = []
    for part in selector.split(";"):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^([^=~]+?)\s*(~=|=)\s*(.*)$", part)
        if not m:
            raise SelectorError(f"unparseable CSV selector clause: {part!r}")
        clauses.append((m.group(1).strip(), m.group(2), m.group(3).strip()))
    if not clauses:
        raise SelectorError("empty CSV selector")
    return clauses


def _float_close(a_text: str, b_text: str) -> bool:
    """abs(a-b) <= 1e-12 * max(1, abs(b)), computed in Decimal."""
    try:
        a, b = Decimal(a_text), Decimal(b_text)
    except Exception:
        return False
    with localcontext() as ctx:
        ctx.prec = 60
        scale = max(Decimal(1), abs(b))
        return abs(a - b) <= FLOAT_TOL_REL * scale


def select_csv_row(path, selector: str) -> dict:
    """Resolve a CSV selector to EXACTLY ONE row. Values compared as text."""
    rows = read_csv_literal(path)
    if not rows:
        raise SelectorError(f"{rel(path)} has no data rows")
    clauses = _parse_csv_selector(selector)
    for col, _op, _v in clauses:
        if col not in rows[0]:
            raise SelectorError(
                f"{rel(path)}: no column {col!r}; have {sorted(rows[0])}")
    hits = []
    for row in rows:
        ok = True
        for col, op, val in clauses:
            cell = row.get(col, "")
            if op == "=":
                if cell != val:
                    ok = False
                    break
            else:
                if not _float_close(cell, val):
                    ok = False
                    break
        if ok:
            hits.append(row)
    if len(hits) != 1:
        raise SelectorError(
            f"{rel(path)} selector {selector!r} matched {len(hits)} rows, need exactly 1")
    return hits[0]


def select_csv_value(path, selector: str, metric: str) -> str:
    """The literal text of one cell, uniquely selected."""
    row = select_csv_row(path, selector)
    if metric not in row:
        raise SelectorError(
            f"{rel(path)}: no metric column {metric!r}; have {sorted(row)}")
    return row[metric]


_JSON_STEP = re.compile(r"\.([^.\[\]]+)|\[(\d+)\]")


def select_json_value(path, selector: str):
    """``$.a.b[0].c`` -> exactly one scalar. Works for JSON and YAML mappings."""
    if not selector.startswith("$"):
        raise SelectorError(f"JSON selector must start with '$': {selector!r}")
    obj = read_yaml(path) if Path(path).suffix in (".yaml", ".yml") else read_json(path)
    cur, pos = obj, 1
    while pos < len(selector):
        m = _JSON_STEP.match(selector, pos)
        if not m:
            raise SelectorError(f"unparseable JSON selector at {selector[pos:]!r}")
        key, idx = m.group(1), m.group(2)
        if key is not None:
            if not isinstance(cur, dict) or key not in cur:
                raise SelectorError(
                    f"{rel(path)}: {selector!r} -- key {key!r} absent at "
                    f"{type(cur).__name__}")
            cur = cur[key]
        else:
            i = int(idx)
            if not isinstance(cur, list) or i >= len(cur):
                raise SelectorError(f"{rel(path)}: {selector!r} -- index {i} out of range")
            cur = cur[i]
        pos = m.end()
    if isinstance(cur, (dict, list)):
        raise SelectorError(
            f"{rel(path)}: {selector!r} resolved to {type(cur).__name__}, need a scalar")
    return cur


def select_markdown_quote(path, heading: str, quote: str) -> str:
    """A verbatim substring that occurs EXACTLY ONCE under a given heading."""
    text = read_text(path)
    lines = text.split("\n")
    starts = [i for i, ln in enumerate(lines)
              if ln.lstrip().startswith("#") and heading in ln]
    if len(starts) != 1:
        raise SelectorError(
            f"{rel(path)}: heading {heading!r} matched {len(starts)} headings, need 1")
    start = starts[0]
    head = lines[start].lstrip()
    level = len(head) - len(head.lstrip("#"))
    end = len(lines)
    for i in range(start + 1, len(lines)):
        s = lines[i].lstrip()
        if s.startswith("#"):
            lvl = len(s) - len(s.lstrip("#"))
            if lvl <= level:
                end = i
                break
    section = "\n".join(lines[start:end])
    n = section.count(quote)
    if n != 1:
        raise SelectorError(
            f"{rel(path)} under {heading!r}: quote occurs {n} times, need exactly 1")
    return quote


def resolve_selector(path, kind: str, selector: str, metric: str = ""):
    """Dispatch on artifact kind. Returns the literal value as text."""
    if kind == "csv":
        return select_csv_value(path, selector, metric)
    if kind in ("json", "yaml"):
        v = select_json_value(path, selector)
        return v if isinstance(v, str) else _scalar_to_text(v)
    if kind == "markdown":
        m = re.match(r'^heading="(.*?)"\s*;\s*quote="(.*)"$', selector, re.S)
        if not m:
            raise SelectorError(
                'markdown selector must be heading="..."; quote="..."')
        return select_markdown_quote(path, m.group(1), m.group(2))
    raise SelectorError(f"unknown artifact kind {kind!r}")


def _scalar_to_text(v) -> str:
    """JSON scalar -> the literal text Tier B0 records. repr(float) is stable."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return ""
    if isinstance(v, float):
        return repr(v)
    return str(v)


# --------------------------------------------------------------------------
# Value pipeline:  raw_value -> value_transform -> rounding_rule -> display_value
#
# The two stages are deliberately separate. rounding_rule NEVER carries a unit
# conversion; a percent display is value_transform=times_100 + half_up:<n>.
# --------------------------------------------------------------------------
VALUE_TRANSFORMS = ("identity", "times_100", "divide_100", "absolute_value")

ROUNDING_RULES_FIXED = ("int", "usd_comma:0", "verbatim", "none")
_ROUND_PARAM = re.compile(r"^(half_up|sig|sci):(\d+)$")


def valid_transform(name: str) -> bool:
    return name in VALUE_TRANSFORMS


def valid_rounding(rule: str) -> bool:
    return rule in ROUNDING_RULES_FIXED or bool(_ROUND_PARAM.match(rule))


def apply_transform(raw_value: str, transform: str) -> str:
    """Exact Decimal transform. Returns text; never touches float."""
    if not valid_transform(transform):
        raise ValueError(f"unknown value_transform {transform!r}")
    if transform == "identity" or raw_value == "":
        return raw_value
    with localcontext() as ctx:
        ctx.prec = 60
        d = Decimal(raw_value)
        if transform == "times_100":
            d = d.scaleb(2)
        elif transform == "divide_100":
            d = d.scaleb(-2)
        elif transform == "absolute_value":
            d = abs(d)
        return format(d, "f")


def apply_rounding(value_text: str, rule: str) -> str:
    """Round a decimal literal with ROUND_HALF_UP -- never Python's banker's round."""
    if not valid_rounding(rule):
        raise ValueError(f"unknown rounding_rule {rule!r}")
    if rule == "none":
        return ""
    if rule == "verbatim":
        return value_text
    if value_text == "":
        raise ValueError(f"rounding_rule {rule!r} needs a value, got empty text")
    with localcontext() as ctx:
        ctx.prec = 60
        d = Decimal(value_text)
        if rule == "int":
            return str(int(d.quantize(Decimal(1), rounding=ROUND_HALF_UP)))
        if rule == "usd_comma:0":
            n = int(d.quantize(Decimal(1), rounding=ROUND_HALF_UP))
            return f"{n:,}"
        m = _ROUND_PARAM.match(rule)
        kind, n = m.group(1), int(m.group(2))
        if kind == "sci":
            # n significant digits in exponential form with a 2-digit exponent,
            # e.g. sci:3 -> "3.24e-02". A DISPLAY FORMAT, not a unit conversion.
            q = Decimal(1).scaleb(-(n - 1))
            if d == 0:
                mant, exp = Decimal(0).quantize(q), 0
            else:
                exp = d.adjusted()
                mant = (d.scaleb(-exp)).quantize(q, rounding=ROUND_HALF_UP)
                if abs(mant) >= 10:            # rounding carried into the exponent
                    mant, exp = mant.scaleb(-1).quantize(q), exp + 1
            sign = "-" if mant < 0 else ""
            return (f"{sign}{format(abs(mant), 'f')}e"
                    f"{'-' if exp < 0 else '+'}{abs(exp):02d}")
        if kind == "half_up":
            q = Decimal(1).scaleb(-n)
            return format(d.quantize(q, rounding=ROUND_HALF_UP), "f")
        # sig:<n> -- n significant digits, half-up
        if d == 0:
            return format(Decimal(0).quantize(Decimal(1).scaleb(-(n - 1)),
                                              rounding=ROUND_HALF_UP), "f")
        exp = d.adjusted()                    # floor(log10(|d|))
        q = Decimal(1).scaleb(exp - (n - 1))
        return format(d.quantize(q, rounding=ROUND_HALF_UP), "f")


def to_display(raw_value: str, transform: str, rule: str) -> str:
    """The whole deterministic pipeline, in one call."""
    return apply_rounding(apply_transform(raw_value, transform), rule)


# --------------------------------------------------------------------------
# Closed vocabularies (normative; the tests read these, not literals)
# --------------------------------------------------------------------------
ATTAINED_STATUS = ("ATTAINED", "NOT_ATTAINED", "NOT_APPLICABLE", "UNRESOLVED")

RESOLUTION_STATUS = ("SOURCED", "ALLOWLISTED", "FLAGGED_UNSUPPORTED", "NOT_RENDERED")

ALLOWLIST_REASONS = ("IAAO_STANDARD_CONSTANT", "METRIC_DEFINITION_CONSTANT",
                     "DESIGN_CONSTANT", "MATH_CONSTANT", "CITATION_OR_YEAR",
                     "TYPESETTING", "EXTERNAL_LITERATURE_STATISTIC")

GUIDANCE_STATUS = ("IAAO_2013_ADOPTED", "IAAO_2026_ED2_PROPOSED", "NOT_APPLICABLE")

# Claim disposition. QUALIFY_AS_EXPLORATORY is gated: it is legal only when the
# exact underlying frozen number IS source-resolvable. An unsupported number is
# never publishable merely by being relabelled exploratory.
DISPOSITIONS = ("KEEP", "REWRITE_WITH_SUPPORTED_EVIDENCE", "QUALIFY_AS_EXPLORATORY",
                "DELETE_OR_REPLACE", "ADD")

# Table / figure disposition -- a SEPARATE vocabulary from claim disposition.
TF_DISPOSITIONS = ("KEEP", "UPDATE", "REBUILD", "DEMOTE_TO_APPENDIX", "DELETE")

TF_SUPPORT = ("FULLY_SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "NOT_RENDERED")

PRIMARY_OR_SENSITIVITY = ("PRIMARY", "SENSITIVITY", "DESCRIPTIVE", "NOT_APPLICABLE")

# A and C are two frozen roles, not a substitution.
#   A = assessor-facing / workflow benchmark
#   C = PRIMARY within-path penalty-isolating reference
REFERENCE_CELLS = ("A", "B", "C", "NONE")
COMPARISON_PURPOSES = ("WORKFLOW_BENCHMARK", "PENALTY_ISOLATING", "WITHIN_MODEL",
                       "NOT_APPLICABLE")

# Every ED2 / aggregate count statement must name its counting unit.
COUNTING_UNITS = ("display_entry", "unique_realization",
                  "standards_facing_configuration", "evaluation_cell",
                  "NOT_APPLICABLE")

RENDER_BUCKETS = ("ACTIVE", "SUPPRESSED", "COMMENTED", "IFFALSE")

SCIENTIFIC_STAGES = ("P0_STAGE_1", "P0_STAGE_1_5_G2", "P0_STAGE_2_G3",
                     "P0_STAGE_3_MATCHED_BETA", "P0_STAGE_3_TEMPORAL",
                     "P0_STAGE_3B_G5B", "P0_PROVENANCE", "P0_CONVENTION",
                     "P1_DISPLAY_SET", "P1_DCOR", "P1_PRB", "P1_VEI_ED2",
                     "P1_SMEARING", "P1_PROVENANCE", "TIER_A_MANUSCRIPT",
                     "LEGACY_FROZEN_PATH")

BIB_FIELD_SOURCES = ("CROSSREF_DOI", "PUBLISHER_LANDING", "IN_REPO_TXT",
                     "UNVERIFIED_FLAGGED")

BIB_AGREEMENT = ("EXACT", "NORMALIZED_MATCH", "CROSSREF_SILENT", "CONFLICT")

# The three \iffalse regions of the canonical manuscript, measured at the
# Tier-A baseline. Every ATTOM number lives inside them and is not compiled.
IFFALSE_REGIONS = ((2081, 2098), (2925, 2993), (4320, 4370))


def norm_excerpt(text: str, limit: int = 120) -> str:
    """Whitespace-normalized baseline excerpt -- the edit-stable anchor payload."""
    return " ".join(text.split())[:limit]
