#!/usr/bin/env python3
"""The thirteen Tier-B manuscript checks.

Each check returns Findings. A Finding is not automatically a failure: the
canonical manuscript is intentionally NOT compliant with the final-state checks
at B1.0, so ``validate.py`` classifies every Finding against a stage-aware
expected-failure registry. The criterion for a stage is
``UNEXPECTED_VALIDATOR_FAILURES == 0``, never "the unrevised manuscript passes
every final-state check". Checks are never weakened to make a baseline green.

Findings carry a stable ``key`` so the registry can name them, and an optional
``count`` so an expectation can be exact. An expectation with a count is what
makes the registry stage-aware in a useful way: when B1.1 deletes
``tab:transition_regret``, its 100 flagged tokens must be gone -- not merely
allowed to be fewer.

    C01  numeric provenance -- the Tier-B ledger, recomputed from the artifacts
    C02  no FLAGGED_UNSUPPORTED token remains
    C03  A / B / C reference-cell semantics
    C04  D1 / D2 / D3 development-coordinate semantics
    C05  NOT_ATTAINED preservation
    C06  ED2 guidance status and counting units
    C07  forbidden wording
    C08  no unsupported candidate-region assets, captions or prose
    C09  figure existence and git tracking
    C10  label / reference integrity
    C11  citation integrity
    C12  TODO closure
    C13  known ACTIVE unsupported claims that hide inside math mode

C13 exists because C02 cannot see them. The frozen coverage audit masks math
environments before extracting tokens, so ``$0.08677$`` is not in the token
population at all -- neither SOURCED nor FLAGGED. The contractual trajectory
356 -> 218 -> 34 -> 0 is therefore a statement about the TEXT-MODE population,
and a future ``FLAGGED_UNSUPPORTED = 0`` must not be read as "every printed
number resolves". C13 carries the known exceptions explicitly, by content rather
than by line number, each with the stage that owes its removal or rewrite.

``tb_scope`` supplies the two non-negotiable checks that sit alongside these --
cumulative paper-only write scope, and byte-identity of the three frozen
subtrees. Those have no registry entries: they may never fail.
"""
from __future__ import annotations

import os
import re
import subprocess

import tb_common as tb
import tb_coverage
import tb_ledger
import tb_text

# ---------------------------------------------------------------------------


class Finding:
    __slots__ = ("check", "key", "message", "count", "anchor", "line", "severity")

    def __init__(self, check, key, message, count=None, anchor="", line=None,
                 severity="FAIL"):
        self.check = check
        self.key = key
        self.message = message
        self.count = count
        self.anchor = anchor
        self.line = line
        self.severity = severity

    def as_dict(self):
        return {"check": self.check, "key": self.key, "message": self.message,
                "count": self.count, "anchor": self.anchor, "line": self.line,
                "severity": self.severity}

    def __repr__(self):
        c = f" (n={self.count})" if self.count is not None else ""
        loc = f" L{self.line}" if self.line else ""
        return f"[{self.check}] {self.key}{c}{loc}: {self.message}"


def _squash(s: str) -> str:
    return re.sub(r"\s+", "", s)


class Context:
    """Everything the checks share, computed once."""

    def __init__(self):
        self.tex = tb.live_tex()
        self.at = tb_text.ActiveText(self.tex)
        self.coverage = tb_coverage.audit(self.tex)
        self.fw = tb.frozen_forbidden_wording()
        self.tf = tb.frozen_tf_disposition()
        self.todo = tb.frozen_todo_closure()
        self.b0_common, _ = tb.frozen_b0()
        self._graphics = None
        self._bib = None
        self._caption_spans = None

    # ---- graphics -------------------------------------------------------
    @property
    def active_graphics(self):
        """(path, position) for every graphic the ACTIVE build includes.

        The preamble is excluded: \\safeincludegraphics is DEFINED there, and
        its `#2` placeholder is not a file name.
        """
        if self._graphics is None:
            out = []
            for m in re.finditer(
                    r"\\(?:safe)?includegraphics(?:\[[^\]]*\])?\{([^}]+)\}",
                    self.tex.src):
                if m.start() < self.tex.preamble_end:
                    continue
                if self.tex.bucket_at(m.start()) != "ACTIVE":
                    continue
                out.append((m.group(1), m.start()))
            self._graphics = out
        return self._graphics

    # ---- bibliography ---------------------------------------------------
    @property
    def bib(self):
        if self._bib is None:
            keys = {}
            for f in (tb.BIB_MAIN, tb.BIB_ADDITIONS):
                if not f.exists():
                    continue
                txt = f.read_text(encoding="utf-8", errors="replace")
                for m in re.finditer(r"^@(\w+)\s*\{\s*([^,\s]+)\s*,", txt, re.M):
                    body = _bib_entry_body(txt, m.end())
                    keys[m.group(2)] = {"file": tb.rel(f), "type": m.group(1),
                                        "body": body}
            self._bib = keys
        return self._bib

    def active_cite_keys(self):
        out = {}
        for m in re.finditer(
                r"\\(?:cite|citep|citet|citeauthor|citeyear|citealp|citealt|"
                r"parencite|textcite)\s*(?:\[[^\]]*\])*\s*\{([^}]*)\}",
                self.tex.src):
            if m.start() < self.tex.preamble_end:
                continue
            if self.tex.bucket_at(m.start()) != "ACTIVE":
                continue
            for k in m.group(1).split(","):
                k = k.strip()
                if k:
                    out.setdefault(k, []).append(self.tex.line_of(m.start()))
        return out

    # ---- label definedness ----------------------------------------------
    def _oldtext_spans(self):
        return [(c["start"], c["end"]) for c in self.tex.calls["oldtext"]
                if not c["inert"]]

    def is_compiled(self, pos: int) -> bool:
        """Does LaTeX actually process this position?

        COMMENTED and IFFALSE: no. An active ``\\oldtext{...}`` body: no -- the
        Tier-A redefinition is ``\\renewcommand{\\oldtext}[1]{}``, which GOBBLES
        the argument, so a ``\\label`` or ``\\ref`` inside it never executes.
        An ``oldrevisionblock`` body: YES -- the redefinition is
        ``\\setbox0\\vbox\\bgroup ... \\egroup``, which TYPESETS the body into a
        box that is then discarded. Labels there are defined and references
        there are resolved, so a suppressed sentence can still leave a dangling
        reference. Getting this distinction wrong makes the reference check
        either blind or full of false alarms.
        """
        b = self.tex.bucket_at(pos)
        if b in ("COMMENTED", "IFFALSE"):
            return False
        if b == "ACTIVE":
            return True
        return not any(a <= pos < z for a, z in self._oldtext_spans())

    def labels_defined(self):
        out = {}
        for pos, line, lab in self.tex.labels:
            if pos < self.tex.preamble_end or not self.is_compiled(pos):
                continue
            out.setdefault(lab, []).append(line)
        return out

    def refs_made(self):
        out = {}
        for m in re.finditer(
                r"\\(?:ref|autoref|eqref|pageref|cref|Cref|nameref)\{([^}]+)\}",
                self.tex.src):
            if m.start() < self.tex.preamble_end or not self.is_compiled(m.start()):
                continue
            out.setdefault(m.group(1), []).append(self.tex.line_of(m.start()))
        return out

    # ---- floats ---------------------------------------------------------
    def active_floats(self):
        return [f for f in self.tex.floats() if f["bucket"] == "ACTIVE"]

    # ---- paragraph scope ------------------------------------------------
    def paragraph_of(self, pos: int) -> str:
        """The blank-line-delimited paragraph around a position, normalized.

        Scope matters for the ED2 status check: a careful two-sentence
        disclaimer ("... reported as descriptive diagnostics rather than
        determinations of compliance") sits in a paragraph that DOES name the
        exposure draft, and demanding the qualifier inside every sentence would
        report it as a violation. The paragraph is the honest unit for "does
        this passage say what standing the guidance has".
        """
        m = self.at.masked
        a = m.rfind("\n\n", 0, pos)
        a = 0 if a < 0 else a + 2
        b = m.find("\n\n", pos)
        b = len(m) if b < 0 else b
        return tb_text.normalize(m[a:b])

    def caption_spans(self):
        """Character spans of every \\caption{...} argument.

        Caption prose is already judged by the caption check; counting it a
        second time in the prose scan would double-report the same sentence.
        """
        if self._caption_spans is None:
            spans = []
            for m in re.finditer(r"\\caption\s*(?:\[[^\]]*\])?\s*\{", self.tex.src):
                b = self.tex.src.index("{", m.end() - 1)
                try:
                    end = self.tex._skip_group(b, "{", "}")
                except ValueError:
                    continue
                spans.append((m.start(), end))
            self._caption_spans = spans
        return self._caption_spans

    def in_caption(self, pos: int) -> bool:
        return any(a <= pos < z for a, z in self.caption_spans())

    def states_a_prohibition(self, unit) -> bool:
        """Does this sentence state the prohibition rather than make the claim?

        The same test the frozen forbidden-wording guard uses: an occurrence is
        legal only inside a sentence carrying one of the frozen
        prohibition_markers.
        """
        return any(m.lower() in unit.norm
                   for m in self.fw["prohibition_markers"])


def _bib_entry_body(txt: str, start: int) -> str:
    depth, i = 1, start
    while i < len(txt) and depth:
        if txt[i] == "{":
            depth += 1
        elif txt[i] == "}":
            depth -= 1
        i += 1
    return txt[start:i]


# ---------------------------------------------------------------------------
# C01  numeric provenance -- the ledger, recomputed
# ---------------------------------------------------------------------------
def c01_numeric_provenance(ctx) -> list:
    out = []
    res = tb_ledger.verify_all(tex=ctx.tex, require_manuscript=True)
    for p in res["problems"]:
        eid = p.split(":", 1)[0].strip()
        out.append(Finding("C01", f"C01:ledger:{eid}", p))
    for p in tb_ledger.selftest():
        out.append(Finding("C01", "C01:selftest", p))
    # The ledger's own vocabulary discipline: an aggregate needs a counting unit.
    for e in tb_ledger.load():
        if e.get("counting_unit") in (None, ""):
            out.append(Finding("C01", f"C01:counting_unit:{e.get('entry_id')}",
                               "ledger entry does not name a counting_unit"))
        elif e["counting_unit"] not in ctx.b0_common.COUNTING_UNITS:
            out.append(Finding("C01", f"C01:counting_unit:{e.get('entry_id')}",
                               f"counting_unit {e['counting_unit']!r} is outside "
                               f"the closed vocabulary"))
    return out


# ---------------------------------------------------------------------------
# C02  no FLAGGED_UNSUPPORTED token remains
# ---------------------------------------------------------------------------
def c02_no_flagged_unsupported(ctx) -> list:
    # The TOTAL is not a Finding: validate.py compares it against the stage's
    # contractual budget (356 -> 218 -> ... -> 0) directly, so a per-anchor
    # entry and a total entry cannot disagree about the same fact.
    out = []
    cov = ctx.coverage
    for anchor, n in sorted(cov["flagged_unsupported_by_anchor"].items()):
        out.append(Finding("C02", f"C02:flagged:{anchor}",
                           f"{n} unsupported numeric token(s) at this anchor",
                           count=n, anchor=anchor))
    return out


# ---------------------------------------------------------------------------
# C03  A / B / C reference-cell semantics
# ---------------------------------------------------------------------------
# Tables that must carry BOTH reference roles once they exist. The primary
# comparison may never be left inferable only from prose or a footnote.
TWO_ROLE_TABLES = ("tab:path_anchor_frozen", "tab:path_anchor_standards",
                   "tab:path_anchor_complementary")
REQUIRED_NEW_TABLES = {"tab:path_anchor_frozen": "B2.1",
                       "tab:path_anchor_standards": "B2.1"}


def c03_reference_cell_semantics(ctx) -> list:
    out = []
    roles = ctx.fw["reference_roles"]
    plain = ctx.at.plain

    for cell in ("A", "B", "C"):
        name = roles[cell]["forward_display_name"]
        if tb_text.normalize(name) not in plain:
            out.append(Finding("C03", f"C03:forward_name_missing:{cell}",
                               f"cell {cell} forward display name is never used: "
                               f"{name!r}"))
    # Legacy Stage-1 label strings may appear only inside Stage-1 tables, never
    # as manuscript-facing display names. B's differs from its forward name.
    legacy = roles["B"]["legacy_stage1_label"]
    if tb_text.normalize(legacy) in plain:
        out.append(Finding("C03", f"C03:legacy_label:{legacy}",
                           f"legacy Stage-1 label used as a display name: "
                           f"{legacy!r}"))
    # The frozen attribution rule must be stated, not implied.
    markers = tuple(ctx.fw["prohibition_markers"])
    has_rule = any(
        ("attribut" in u.norm and any(m.lower() in u.norm for m in markers)
         and ("rho" in u.norm or "penalty" in u.norm))
        for u in ctx.at.units)
    if not has_rule:
        out.append(Finding("C03", "C03:attribution_rule_missing",
                           "no sentence states that changes relative to the "
                           "workflow benchmark must not be attributed to rho"))
    if "penalty-isolating" not in plain and "penalty isolating" not in plain:
        out.append(Finding("C03", "C03:penalty_isolating_missing",
                           "the penalty-isolating role of the custom-objective "
                           "rho=0 origin is never named"))
    # Both reference rows present in each two-role table that exists.
    floats_by_label = {}
    for f in ctx.active_floats():
        for lab in f["labels"]:
            floats_by_label[lab] = f
    for lab, stage in REQUIRED_NEW_TABLES.items():
        if lab not in floats_by_label:
            out.append(Finding("C03", f"C03:missing_table:{lab}",
                               f"required rebuilt table {lab} does not exist"))
    for lab in TWO_ROLE_TABLES:
        f = floats_by_label.get(lab)
        if not f:
            continue
        body = tb_text.normalize(ctx.tex.src[
            ctx.tex._line_starts()[f["line"] - 1]:
            ctx.tex._line_starts()[min(f["end_line"],
                                       len(ctx.tex._line_starts()) - 1)]])
        for cell in ("A", "C"):
            nm = tb_text.normalize(roles[cell]["forward_display_name"])
            if nm not in body:
                out.append(Finding(
                    "C03", f"C03:reference_rows:{lab}:{cell}",
                    f"{lab} does not carry an explicit cell-{cell} reference row "
                    f"({roles[cell]['forward_display_name']!r})", anchor=lab,
                    line=f["line"]))
    return out


# ---------------------------------------------------------------------------
# C04  D1 / D2 / D3 development-coordinate semantics
# ---------------------------------------------------------------------------
def c04_development_coordinates(ctx) -> list:
    out = []
    plain = ctx.at.plain
    src_norm = plain

    def _has(*alts):
        return any(tb_text.normalize(a) in src_norm for a in alts)

    if not re.search(r"\bd1\b", src_norm):
        out.append(Finding("C04", "C04:missing_token:D1",
                           "the primary development coordinate D1 is never named"))
    if not re.search(r"\bd3\b", src_norm):
        out.append(Finding("C04", "C04:missing_token:D3",
                           "the D3 sensitivity coordinate is never named"))
    if not _has("one-sale-one-vote", "one sale one vote"):
        out.append(Finding("C04", "C04:missing:one_sale_one_vote",
                           "D3 is not described as the one-sale-one-vote "
                           "sensitivity"))
    if not any(("d1" in u.norm and "primary" in u.norm) for u in ctx.at.units):
        out.append(Finding("C04", "C04:missing:d1_primary",
                           "D1 is not stated to be the primary coordinate"))
    if not any(("overlap" in u.norm and ("fold" in u.norm))
               for u in ctx.at.units):
        out.append(Finding("C04", "C04:missing:overlap_qualification",
                           "the fold-6 / fold-7 validation-overlap qualification "
                           "is absent"))
    if not _has("not an independent-observation aggregate",
                "not an independent observation aggregate"):
        out.append(Finding("C04", "C04:missing:not_independent_aggregate",
                           "D1 is not qualified as NOT an "
                           "independent-observation aggregate"))
    # "unaffected" is prohibited for D1: POST_G3_ADJUDICATION section 2
    # supersedes both the Tier-B0 spec's phrasing and the comparator report.
    for u in ctx.at.units:
        if "unaffected" not in u.norm:
            continue
        if not re.search(r"\bd1\b|cv-mean|cv mean", u.norm):
            continue
        # A sentence that STATES the prohibition -- "we do not describe D1 as
        # unaffected" -- is the compliant form, not a violation. Same exemption
        # C07 and C08 apply, from the same frozen prohibition_markers vocabulary.
        if ctx.states_a_prohibition(u):
            continue
        out.append(Finding("C04", "C04:d1_unaffected",
                           "D1 is described as 'unaffected' by the overlap",
                           line=u.line, anchor=u.anchor))
    # No IID reading of seven overlapping chronological folds.
    for u in ctx.at.units:
        if re.search(r"\bsd\b|standard deviation", u.norm) and \
                "standard error" in u.norm and "fold" in u.norm and \
                not any(m.lower() in u.norm
                        for m in ctx.fw["prohibition_markers"]):
            out.append(Finding("C04", "C04:iid_fold_inference",
                               "fold SD presented as a standard error",
                               line=u.line, anchor=u.anchor))
    return out


# ---------------------------------------------------------------------------
# C05  NOT_ATTAINED preservation
# ---------------------------------------------------------------------------
BASELINE_TODO_SITES = 19       # the frozen crosswalk accounts for exactly these
MATCHED_BETA_LABELS = ("tab:matched_beta",)
N_FROZEN_NOT_ATTAINED = 4      # Direct at -0.06, -0.03, 0.00; Surrogate at 0.00


def c05_not_attained(ctx) -> list:
    out = []
    labels = {l for f in ctx.active_floats() for l in f["labels"]}
    present = [l for l in MATCHED_BETA_LABELS if l in labels]
    if not present:
        out.append(Finding("C05", "C05:matched_beta_absent",
                           "the matched-beta table does not exist, so the four "
                           "frozen NOT_ATTAINED states cannot be checked"))
        return out
    n = len(re.findall(r"NOT_ATTAINED", ctx.at.masked))
    if n < N_FROZEN_NOT_ATTAINED:
        out.append(Finding("C05", "C05:not_attained_count",
                           f"{n} NOT_ATTAINED display states present; the frozen "
                           f"set has {N_FROZEN_NOT_ATTAINED}", count=n))
    # A NOT_ATTAINED row carries BLANK metric cells: never dropped, never filled.
    for u in ctx.at.units:
        if "not_attained" in u.norm and re.search(r"\d", u.norm):
            out.append(Finding("C05", "C05:not_attained_interpolated",
                               "a NOT_ATTAINED cell sits beside a numeric value; "
                               "the metric columns must stay blank",
                               line=u.line, anchor=u.anchor))
    return out


# ---------------------------------------------------------------------------
# C06  ED2 guidance status and counting units
# ---------------------------------------------------------------------------
DRAFT_MARKERS = ("exposure draft", "exposure-draft", "proposed", "draft")
COUNT_UNIT_WORDS = ("display entr", "unique realization", "unique realisation",
                    "standards-facing configuration", "standards facing "
                    "configuration", "evaluation cell", "evaluation block",
                    "fitted realization", "fitted realisation")


def c06_ed2_status_and_counts(ctx) -> list:
    out = []
    # (a) A band or standard attribution for VEI / MKI must be marked as the
    #     May-2026 EXPOSURE DRAFT, never as adopted guidance.
    for u in ctx.at.units:
        if not re.search(r"\bvei\b|\bmki\b", u.norm):
            continue
        if not re.search(r"standard|guidance|complian|band|threshold", u.norm):
            continue
        if any(d in u.norm for d in DRAFT_MARKERS):
            continue
        if any(d in ctx.paragraph_of(u.start) for d in DRAFT_MARKERS):
            continue
        if ctx.states_a_prohibition(u):
            continue
        out.append(Finding("C06", f"C06:ed2_status:{u.anchor}",
                           "a VEI/MKI band or standard attribution carries no "
                           "'Exposure Draft' or 'proposed' qualifier: "
                           f"{u.excerpt(110)!r}", line=u.line, anchor=u.anchor))
    # (b)+(c) counting units. The two ED2 count families differ by exactly 9 and
    #     may never appear in one statement; any count needs its unit named.
    sets = ctx.fw["mutually_exclusive_count_sets"]
    for u in ctx.at.units:
        hit = []
        for s in sets:
            if any(re.search(r"(?<![\d.])" + re.escape(v) + r"(?![\d.])", u.norm)
                   for v in s["values"]):
                hit.append(s["name"])
        if len(hit) > 1:
            out.append(Finding("C06", f"C06:count_mix:{u.line}",
                               f"one statement mixes the mutually exclusive count "
                               f"families {hit}: {u.excerpt(110)!r}",
                               line=u.line, anchor=u.anchor))
        elif hit and not any(w in u.norm for w in COUNT_UNIT_WORDS):
            out.append(Finding("C06", f"C06:count_unit:{u.line}",
                               f"an ED2 count from {hit[0]} appears without "
                               f"naming its counting unit: {u.excerpt(110)!r}",
                               line=u.line, anchor=u.anchor))
    return out


# ---------------------------------------------------------------------------
# C07  forbidden wording
# ---------------------------------------------------------------------------
def c07_forbidden_wording(ctx) -> list:
    """Data-driven from spec/forbidden_wording.yaml -- never a local copy.

    Matching squashes whitespace on both sides, so re-spacing a phrase does not
    evade the guard. An occurrence is legal only inside a sentence that carries
    one of the frozen prohibition_markers, i.e. a sentence stating the
    prohibition itself.
    """
    out = []
    markers = [m.lower() for m in ctx.fw["prohibition_markers"]]
    phrases = [(p["phrase"], p["why"]) for p in ctx.fw["forbidden_phrases"]]
    literals = [(l, ctx.fw["forbidden_literals_why"])
                for l in ctx.fw["forbidden_literals"]]
    # The positive counterpart: four denials the frozen spec relies on must keep
    # printing. A prohibition list cannot protect them -- deleting a denial
    # breaks no pattern scan.
    req = tb.read_yaml(tb.TB_SPEC / "tier_b_required_statements.yaml")
    for st in req["required_statements"]:
        need = [st["fragment"]] + ([st["also_requires"]]
                                   if st.get("also_requires") else [])
        if not all(tb_text.normalize(f) in ctx.at.plain for f in need):
            out.append(Finding("C07", f"C07:missing_required_denial:{st['id']}",
                               f"a required denial no longer prints: "
                               f"{st['why'].strip()}"))
    for phrase, why in phrases + literals:
        needle = _squash(tb_text.normalize(phrase))
        if not needle:
            continue
        for u in ctx.at.units:
            if needle not in _squash(u.norm):
                continue
            if any(m in u.norm for m in markers):
                continue
            out.append(Finding("C07", f"C07:phrase:{phrase}",
                               f"forbidden phrase {phrase!r} printed without a "
                               f"prohibition cue. {why.strip()}",
                               line=u.line, anchor=u.anchor))
    return out


# ---------------------------------------------------------------------------
# C08  no unsupported candidate-region assets, captions or prose
# ---------------------------------------------------------------------------
def c08_candidate_region(ctx) -> list:
    """Visual provenance obeys the same rule as numeric provenance.

    A path figure can have perfectly supported caption numbers while the GRAPHIC
    draws the candidate-region fill, the activity-onset boundary, the upper
    guardrail or the transition-span shading -- the CV screen result that no
    frozen artifact reproduces. The path curves and the genuine
    metric-definition reference lines (PRD=1, PRB=0, MKI=1, VEI=0, beta_log=0,
    ratio=1) are supported and stay; the overlay goes.

    The rho endpoints themselves are NOT individually flagged by the coverage
    audit, because they exist in the frozen grid maps as GRID COORDINATES. That
    is not a licence: the claim that those coordinates ARE the activity onset and
    the upper guardrail is what no artifact reproduces. Token-level allowlisting
    of a grid coordinate is not support for the claim built on it -- which is
    exactly why this check reads prose and captions, not just numbers.
    """
    out = []
    markers = tuple(ctx.b0_common.UNSUPPORTED_VISUAL_MARKERS)

    for path, pos in ctx.active_graphics:
        if "_candidate_region" in path:
            out.append(Finding("C08", f"C08:asset:{os.path.basename(path)}",
                               f"the ACTIVE build draws an unsupported "
                               f"candidate-region overlay asset: {path}",
                               line=ctx.tex.line_of(pos)))
    for f in ctx.active_floats():
        cap = (f.get("caption_full") or "").lower()
        hits = sorted({m for m in markers if m in cap})
        if hits:
            lab = ",".join(f["labels"]) or f"L{f['line']}"
            out.append(Finding("C08", f"C08:caption:{lab}",
                               f"caption states the unsupported screen result "
                               f"{hits}", anchor=lab, line=f["line"]))
    # Prose, one finding per sentence, keyed by the sha256 of its NORMALIZED
    # text rather than by anchor or line. Both of those are unstable by
    # construction: B1.1 deletes labels, so a surviving sentence's nearest-label
    # anchor changes, and every line number shifts. A content hash is the
    # edit-stable identity SELECTOR_GRAMMAR.md section 4 asks for -- a deleted
    # sentence's key disappears, and a NEW candidate-region sentence gets a key
    # nothing declares, which is exactly when the check should fire.
    for u in ctx.at.units:
        txt = re.sub(r"\\label\{[^}]*\}", " ", u.raw)
        txt = re.sub(r"\\(?:safe)?includegraphics(?:\[[^\]]*\])?\{[^}]*\}",
                     " ", txt)
        norm = tb_text.normalize(txt)
        hits = sorted({m for m in markers if m in norm})
        if not hits:
            continue            # the asset check owns graphics; labels are names
        if ctx.in_caption(u.start):
            continue            # already judged by the caption check above
        if ctx.states_a_prohibition(u):
            continue            # a sentence stating the prohibition is legal
        h = tb.sha256_text(norm)[:12]
        out.append(Finding("C08", f"C08:prose:{h}",
                           f"ACTIVE statement asserts the candidate-region / "
                           f"transition-span construction {hits} at "
                           f"{u.anchor}: {u.excerpt(120)!r}",
                           anchor=u.anchor, line=u.line))
    return out


# ---------------------------------------------------------------------------
# C09  figure existence and git tracking
# ---------------------------------------------------------------------------
def c09_figure_existence(ctx) -> list:
    out = []
    tracked = set(tb.git("ls-files", "paper").split())
    for path, pos in ctx.active_graphics:
        full = tb.PAPER / path
        rel = f"paper/{path}"
        if not full.exists():
            out.append(Finding("C09", f"C09:missing:{path}",
                               f"included graphic does not exist: {rel}",
                               line=ctx.tex.line_of(pos)))
        elif rel not in tracked:
            out.append(Finding("C09", f"C09:untracked:{path}",
                               f"included graphic is not tracked in git: {rel}",
                               line=ctx.tex.line_of(pos)))
    return out


# ---------------------------------------------------------------------------
# C10  label / reference integrity
# ---------------------------------------------------------------------------
def c10_label_ref_integrity(ctx) -> list:
    out = []
    defined = ctx.labels_defined()
    for lab, lines in sorted(defined.items()):
        if len(lines) > 1:
            out.append(Finding("C10", f"C10:duplicate_label:{lab}",
                               f"label defined {len(lines)} times in the compiled "
                               f"document at lines {lines}", count=len(lines),
                               line=lines[0]))
    for lab, lines in sorted(ctx.refs_made().items()):
        if lab not in defined:
            out.append(Finding("C10", f"C10:dangling_ref:{lab}",
                               f"reference to an undefined label, from lines "
                               f"{lines}", line=lines[0]))
    return out


# ---------------------------------------------------------------------------
# C11  citation integrity
# ---------------------------------------------------------------------------
def c11_citation_integrity(ctx) -> list:
    out = []
    spec = tb.read_yaml(tb.TB_SPEC / "tier_b_citations.yaml")
    bib = ctx.bib
    cited = ctx.active_cite_keys()

    for key, lines in sorted(cited.items()):
        if key not in bib:
            out.append(Finding("C11", f"C11:cite_not_in_bib:{key}",
                               f"cited at lines {lines} but present in no loaded "
                               f".bib, so it prints nothing", line=lines[0]))
    for req in spec["required_citations"]:
        key = req["key"]
        if key not in bib:
            out.append(Finding("C11", f"C11:required_key_not_loaded:{key}",
                               f"required prior-art entry is in no loaded .bib: "
                               f"{req['why']}"))
        if key not in cited:
            out.append(Finding("C11", f"C11:required_key_uncited:{key}",
                               f"required prior-art entry is never cited in the "
                               f"ACTIVE build: {req['why']}"))
    n_add = len(re.findall(r"\\addbibresource", ctx.tex.src))
    if n_add != spec["addbibresource_count"]:
        out.append(Finding("C11", "C11:addbibresource_count",
                           f"{n_add} \\addbibresource declarations; the frozen "
                           f"rule is exactly {spec['addbibresource_count']} and "
                           f"no third one", count=n_add))
    # analysis/ is the frozen evidence layer. paper_analysis/ is this
    # validator's own home under paper/ and is not a dependency of the build,
    # so the lookbehind matters; and a mention inside a comment is not a
    # dependency either.
    preamble = ctx.tex.src[:ctx.tex.preamble_end]
    dep = [m.start() for m in re.finditer(r"(?<![\w])analysis/", preamble)
           if not ctx.tex.comment[m.start()]]
    if dep:
        out.append(Finding("C11", "C11:analysis_path_in_preamble",
                           "the preamble references a path under analysis/; the "
                           "compiled manuscript must never depend on the frozen "
                           "evidence layer"))
    # Invent nothing: fields that no authoritative source verified stay absent.
    for key, rules in (spec.get("bib_field_rules") or {}).items():
        if key not in bib:
            continue
        body = bib[key]["body"]
        for field in (rules.get("must_be_absent") or []):
            if re.search(r"\b" + re.escape(field) + r"\s*=", body, re.I):
                out.append(Finding("C11", f"C11:bib_field_present:{key}:{field}",
                                   f"{key} carries an unverified field "
                                   f"{field!r}: {rules['why']}"))
        for field, want in (rules.get("must_equal") or {}).items():
            m = re.search(r"\b" + re.escape(field) + r"\s*=\s*[{\"]?([^,}\"\n]*)",
                          body, re.I)
            got = (m.group(1).strip() if m else None)
            if got != str(want):
                out.append(Finding("C11", f"C11:bib_field_value:{key}:{field}",
                                   f"{key} {field} is {got!r}, must be "
                                   f"{want!r}: {rules['why']}"))
    return out


# ---------------------------------------------------------------------------
# C12  TODO closure
# ---------------------------------------------------------------------------
def c12_todo_closure(ctx) -> list:
    out = []
    ids = [t["todo_id"] for t in ctx.todo["todos"]]
    if len(ids) != len(set(ids)):
        out.append(Finding("C12", "C12:todo_spec_shape",
                           "the frozen todo_closure crosswalk has duplicate ids"))
    if len(ids) != 19:
        out.append(Finding("C12", "C12:todo_spec_shape",
                           f"the frozen crosswalk covers {len(ids)} todo ids, "
                           f"expected 19", count=len(ids)))
    active = [t for t in ctx.tex.todos if t["bucket"] == "ACTIVE"]
    if len(ctx.tex.todos) > BASELINE_TODO_SITES:
        out.append(Finding("C12", "C12:new_todo_introduced",
                           f"{len(ctx.tex.todos)} \\todo sites, more than the "
                           f"{BASELINE_TODO_SITES} the frozen crosswalk accounts "
                           f"for: the writing pass has introduced a new promise",
                           count=len(ctx.tex.todos)))
    if ctx.tex.todos:
        out.append(Finding("C12", "C12:todos_present",
                           f"{len(ctx.tex.todos)} \\todo sites remain in the "
                           f"source ({len(active)} in the ACTIVE build)",
                           count=len(ctx.tex.todos)))
    return out


# ---------------------------------------------------------------------------
# C13  known ACTIVE unsupported claims inside math mode
# ---------------------------------------------------------------------------
MATH_CLAIM_REQUIRED_FIELDS = (
    "claim_id", "status", "disposition", "resolved_by_stage",
    "manuscript_anchor", "identify_by_literals", "drift_cue",
    "normalized_excerpt_sha256", "why_unsupported")


def math_claim_status(ctx) -> list:
    """Locate every registered math-mode claim in the LIVE manuscript.

    Returns one status dict per registry entry. ``validate.py`` reports these
    alongside -- never inside -- the ordinary flagged-token budget, because they
    are by construction outside the token population that budget counts.

    Identification is by content, never by line number: every literal in
    ``identify_by_literals`` must appear in one ACTIVE sentence unit. All of
    them, because one value alone can legitimately recur -- ``0.08685`` is also
    the upper end of the lower-tail grid extension in the design section, and a
    single-literal matcher would fire there.
    """
    reg = tb.tier_b_math_claims()
    out = []
    for c in reg.get("claims") or []:
        lits = [str(x) for x in (c.get("identify_by_literals") or [])]
        cue = tb_text.normalize(str(c.get("drift_cue") or ""))
        exact = [u for u in ctx.at.units if lits and all(l in u.raw for l in lits)]
        partial = []
        if not exact and cue:
            partial = [u for u in ctx.at.units
                       if cue in u.norm and any(l in u.raw for l in lits)]
        found = exact or partial
        st = {
            "claim_id": c.get("claim_id"),
            "resolved_by_stage": c.get("resolved_by_stage"),
            "disposition": c.get("disposition"),
            "active": bool(found),
            "matched_exactly": bool(exact),
            "n_matches": len(found),
            "line": found[0].line if found else None,
            "anchor": found[0].anchor if found else "",
            "excerpt_matches_registry": None,
            # Proof, not assertion, that this claim really is outside the token
            # budget: if the coverage audit ever starts seeing these literals,
            # the registry and the budget would be counting the same fact twice.
            "in_token_population": sorted(
                {t["token"] for t in ctx.coverage["tokens"] if t["token"] in lits}),
        }
        if found:
            st["excerpt_matches_registry"] = (
                tb.sha256_text(found[0].norm)
                == str(c.get("normalized_excerpt_sha256") or ""))
        out.append(st)
    return out


def c13_unsupported_math_claims(ctx) -> list:
    out = []
    reg = tb.tier_b_math_claims()
    claims = reg.get("claims") or []
    vocab = set(reg.get("dispositions") or ())

    # The registry's own shape. A malformed entry is a silent hole, so it fails
    # as itself rather than being skipped.
    seen = set()
    for c in claims:
        cid = c.get("claim_id") or "(unnamed)"
        missing = [f for f in MATH_CLAIM_REQUIRED_FIELDS if not c.get(f)]
        if missing:
            out.append(Finding("C13", f"C13:registry_shape:{cid}",
                               f"registry entry is missing required field(s) "
                               f"{missing}"))
        if cid in seen:
            out.append(Finding("C13", f"C13:registry_shape:{cid}",
                               "duplicate claim_id in the registry"))
        seen.add(cid)
        if c.get("disposition") and c["disposition"] not in vocab:
            out.append(Finding("C13", f"C13:registry_shape:{cid}",
                               f"disposition {c['disposition']!r} is outside the "
                               f"closed vocabulary {sorted(vocab)}"))
        if c.get("resolved_by_stage") not in tb.STAGES:
            out.append(Finding("C13", f"C13:registry_shape:{cid}",
                               f"resolved_by_stage {c.get('resolved_by_stage')!r} "
                               f"is not a known Tier-B stage"))

    by_id = {c.get("claim_id"): c for c in claims}
    for st in math_claim_status(ctx):
        cid = st["claim_id"]
        c = by_id.get(cid, {})
        if not st["active"]:
            continue                    # gone: the registry entry goes STALE
        out.append(Finding(
            "C13", f"C13:unsupported_math_claim:{cid}",
            f"a known unsupported math-mode claim is still ACTIVE and is "
            f"scheduled for {c.get('disposition')} at "
            f"{c.get('resolved_by_stage')}. It is INVISIBLE to the "
            f"flagged-token budget because the frozen coverage audit masks math "
            f"mode, so it must not be read as validated by any "
            f"FLAGGED_UNSUPPORTED count. {str(c.get('why_unsupported','')).strip()}",
            count=st["n_matches"], anchor=st["anchor"], line=st["line"]))
        if st["excerpt_matches_registry"] is False:
            out.append(Finding(
                "C13", f"C13:excerpt_drift:{cid}",
                "the registered claim is still active but its text no longer "
                "matches the registered excerpt. A rewrite of a registered "
                "claim must update spec/unsupported_math_claims.yaml in the "
                "same commit -- retire the entry if the claim was resolved, or "
                "re-register the new text and its sha256 if it was not.",
                anchor=st["anchor"], line=st["line"]))
        if st["in_token_population"]:
            out.append(Finding(
                "C13", f"C13:double_counted:{cid}",
                f"literal(s) {st['in_token_population']} are now IN the coverage "
                f"token population, so this claim is counted both by the "
                f"flagged-token budget and by this registry. Reconcile the two "
                f"before either number is reported.",
                anchor=st["anchor"], line=st["line"]))
    return out


CHECKS = (
    ("C01", "numeric provenance (Tier-B ledger, recomputed)", c01_numeric_provenance),
    ("C02", "no FLAGGED_UNSUPPORTED token remains", c02_no_flagged_unsupported),
    ("C03", "A / B / C reference-cell semantics", c03_reference_cell_semantics),
    ("C04", "D1 / D2 / D3 development-coordinate semantics", c04_development_coordinates),
    ("C05", "NOT_ATTAINED preservation", c05_not_attained),
    ("C06", "ED2 guidance status and counting units", c06_ed2_status_and_counts),
    ("C07", "forbidden wording", c07_forbidden_wording),
    ("C08", "no unsupported candidate-region assets/captions/prose", c08_candidate_region),
    ("C09", "figure existence and git tracking", c09_figure_existence),
    ("C10", "label / reference integrity", c10_label_ref_integrity),
    ("C11", "citation integrity", c11_citation_integrity),
    ("C12", "TODO closure", c12_todo_closure),
    ("C13", "known unsupported math-mode claims (outside the token budget)",
     c13_unsupported_math_claims),
)


def run_all(ctx=None) -> tuple:
    ctx = ctx or Context()
    findings = []
    for cid, _label, fn in CHECKS:
        findings += fn(ctx)
    return ctx, findings
