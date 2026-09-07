#!/usr/bin/env python3
"""Canonical-manuscript partition, anchors, numeric tokens and float inventory.

The manuscript has FOUR render buckets, not three. Tier A neutralised the
revision macros in the preamble without editing a single call site, so what
prints is decided by macro identity and by two kinds of inert region:

  ACTIVE      plain prose and the accepted macros (\\newtext, \\latesttext,
              \\advisorchange) -- prints.
  SUPPRESSED  \\oldtext{...} and oldrevisionblock -- in the source, prints nothing.
  COMMENTED   after an unescaped % -- prints nothing.
  IFFALSE     inside \\iffalse ... \\fi -- not even compiled. Every ATTOM number
              lives here.

This module re-implements the partition algorithm of
paper/paper_analysis/paper_v17/markup_preflight.py (comment mask, \\iffalse mask,
brace-balanced argument reader, nearest-label anchor). It does NOT import that
file: markup_preflight.py is a sys.argv script that writes its archive at import
time. ``census()`` self-checks this re-implementation against the census table
that script already committed in superseded_text_archive.md.

Anchors: line numbers are BASELINE COORDINATES ONLY and will go stale as soon as
the Tier-B writing pass edits the manuscript. Every recorded location therefore
also carries manuscript_section, latex_label, source_anchor and a normalized
baseline excerpt plus its sha256, so the intended passage can be re-identified
after earlier edits shift the lines.
"""
from __future__ import annotations

import re
from pathlib import Path

import b0_common as c

REVISION_MACROS = ("oldtext", "newtext", "latesttext", "advisorchange")
ACCEPTED_MACROS = ("newtext", "latesttext", "advisorchange")

SECTIONING = ("section", "subsection", "subsubsection")

# Macros whose numeric arguments are references or typesetting, never results.
# The value is how many following groups to mask; None means "mask every
# following [..] / {..} group". Partial counts matter: \textcolor{impGreen}{..}
# and \multicolumn{4}{c}{..} carry REAL PROSE in their last argument -- masking
# the whole call would hide rendered result numbers from the coverage audit.
MASK_MACROS = {
    # citations and cross-references: nothing printed comes from the source
    "cite": None, "citep": None, "citet": None, "citeauthor": None,
    "citeyear": None, "citealp": None, "citealt": None, "nocite": None,
    "ref": None, "autoref": None, "eqref": None, "pageref": None,
    "cref": None, "Cref": None, "label": None,
    # file, URL and package machinery
    "includegraphics": None, "safeincludegraphics": None,
    "url": None, "usepackage": None,
    "documentclass": None, "input": None, "include": None,
    "bibliography": None, "bibliographystyle": None, "addbibresource": None,
    "geometry": None, "captionsetup": None, "definecolor": None,
    "renewcommand": None, "newcommand": None, "newcolumntype": None,
    # pure typesetting lengths and rules
    "cmidrule": None, "cline": None, "arraystretch": None, "setlength": None,
    "hspace": None, "vspace": None, "rule": None, "addlinespace": None,
    "phantom": None, "hphantom": None, "vphantom": None,
    # \todo bodies never print (todonotes is loaded with [disable]); they are
    # audited separately by spec/todo_closure.yaml, not by the numeric coverage.
    "todo": None,
    # partial: the trailing group is real content
    "textcolor": 1, "color": 1, "rowcolor": 1, "columncolor": 1,
    "scalebox": 1, "href": 1,
    "multicolumn": 2, "multirow": 2, "resizebox": 2,
}

MATH_ENVS = ("equation", "equation*", "align", "align*", "gather", "gather*",
             "multline", "multline*", "eqnarray", "eqnarray*", "split",
             "aligned", "alignedat", "array", "pmatrix", "bmatrix", "vmatrix",
             "cases", "displaymath", "math", "IEEEeqnarray")

FLOAT_ENVS = ("table", "table*", "figure", "figure*", "sidewaystable",
              "longtable")

NUMERIC_TOKEN = re.compile(
    r"(?<![0-9A-Za-z._])"
    r"(?:\d{4}-\d{2}-\d{2}"                  # ISO date (a split boundary)
    r"|\d+(?:\.\d+)?[eE][-+]?\d+"           # scientific
    r"|\d{1,3}(?:,\d{3})+(?:\.\d+)?"       # comma-grouped
    r"|\d+\.\d+"                            # decimal
    r"|\.\d+"                               # bare decimal
    r"|\d+)"                                # integer
    # a trailing hyphen is only disqualifying when it continues a date
    # (2016-01-01); "344,607-sale" must still yield 344,607
    r"(?![0-9A-Za-z]|-\d)")


# --------------------------------------------------------------------------
class Tex:
    """The canonical manuscript, partitioned and indexed."""

    def __init__(self, path=None):
        self.path = Path(path) if path else c.TEX
        self.src = c.read_text(self.path)
        self.sha256 = c.sha256_file(self.path)
        self.n = len(self.src)
        self._build_masks()
        self._build_index()

    # ---- position helpers -------------------------------------------------
    def line_of(self, pos: int) -> int:
        return self.src.count("\n", 0, pos) + 1

    def line_text(self, line: int) -> str:
        return self.src.split("\n")[line - 1]

    # ---- masks ------------------------------------------------------------
    def _build_masks(self):
        src, n = self.src, self.n

        comment = bytearray(n)
        i = 0
        while i < n:
            ch = src[i]
            if ch == "\\":
                i += 2
                continue
            if ch == "%":
                j = src.find("\n", i)
                j = n if j < 0 else j
                for k in range(i, j):
                    comment[k] = 1
                i = j
                continue
            i += 1
        self.comment = comment

        iffalse = bytearray(n)
        self.iffalse_regions = []
        for m in re.finditer(r"^\\iffalse", src, re.M):
            end = re.search(r"^\\fi", src[m.start():], re.M)
            if not end:
                continue
            lo, hi = m.start(), m.start() + end.end()
            for k in range(lo, hi):
                iffalse[k] = 1
            self.iffalse_regions.append((self.line_of(lo), self.line_of(hi - 1)))
        self.iffalse = iffalse

        # Preamble: everything before \begin{document} is machinery, not prose.
        bd = src.find("\\begin{document}")
        self.preamble_end = bd if bd >= 0 else 0
        self.preamble_end_line = self.line_of(self.preamble_end)

        # Math regions (inline and display), for numeric-token masking only.
        math = bytearray(n)
        for lo, hi in self._math_spans():
            for k in range(lo, hi):
                math[k] = 1
        self.math = math

        # Reference / typesetting macro calls, for numeric-token masking only.
        refmask = bytearray(n)
        for mac, ngroups in MASK_MACROS.items():
            pat = (r"\\" + mac + r"\*?\s*(?:\([a-z]*\))?\s*(?=[\[{])")
            for m in re.finditer(pat, src):
                pos, taken = m.end(), 0
                while pos < n and src[pos] in "[{":
                    if ngroups is not None and taken >= ngroups:
                        break
                    close = "]" if src[pos] == "[" else "}"
                    try:
                        nxt = self._skip_group(pos, src[pos], close)
                    except ValueError:
                        break
                    # an optional [..] argument does not consume a mandatory slot
                    if src[pos] == "{":
                        taken += 1
                    pos = nxt
                for k in range(m.start(), min(pos, n)):
                    refmask[k] = 1
        # tabular / tabularx column specifications carry p{3.8cm}-style lengths,
        # and \begin{minipage}{0.98\textwidth} carries a width fraction
        for m in re.finditer(r"\\begin\{(?:tabular|tabularx|longtable|tabular\*"
                             r"|minipage|adjustbox|varwidth)\}", src):
            pos = m.end()
            groups = 0
            while pos < n and src[pos] in "[{" and groups < 2:
                close = "]" if src[pos] == "[" else "}"
                try:
                    nxt = self._skip_group(pos, src[pos], close)
                except ValueError:
                    break
                if src[pos] == "{":
                    groups += 1
                pos = nxt
            for k in range(m.start(), min(pos, n)):
                refmask[k] = 1
        self.refmask = refmask

    def _skip_group(self, pos: int, opn: str, close: str) -> int:
        src, n = self.src, self.n
        depth, i = 0, pos
        while i < n:
            ch = src[i]
            if ch == "\\":
                i += 2
                continue
            if ch == opn:
                depth += 1
            elif ch == close:
                depth -= 1
                if depth == 0:
                    return i + 1
            i += 1
        raise ValueError(f"unbalanced {opn} at line {self.line_of(pos)}")

    def _math_spans(self):
        src, n = self.src, self.n
        spans = []
        # $$...$$ and $...$  (skip escaped \$; skip inside comments)
        # LaTeX forbids a paragraph break inside inline math, so an unmatched $
        # is bounded at the next blank line instead of running to EOF.
        i = 0
        while i < n:
            ch = src[i]
            if ch == "\\":
                i += 2
                continue
            if ch == "$" and not self.comment[i]:
                dd = src.startswith("$$", i)
                close = "$$" if dd else "$"
                limit = n
                if not dd:
                    para = src.find("\n\n", i)
                    limit = n if para < 0 else para
                j, found = i + len(close), False
                while j < limit:
                    if src[j] == "\\":
                        j += 2
                        continue
                    if src.startswith(close, j):
                        found = True
                        break
                    j += 1
                if found:
                    spans.append((i, min(j + len(close), n)))
                    i = j + len(close)
                else:
                    i += len(close)          # unmatched: mask nothing
                continue
            i += 1
        # \[ ... \]  -- the lookbehind is essential: "\\[2mm]" is a line break
        # with optional spacing, NOT a display-math opener. Without it the \[ of
        # \\[2mm] pairs with a distant \] and masks hundreds of lines.
        for m in re.finditer(r"(?<!\\)\\\[", src):
            e = re.search(r"(?<!\\)\\\]", src[m.end():])
            if e:
                spans.append((m.start(), m.end() + e.end()))
        # \begin{env} ... \end{env}
        for env in MATH_ENVS:
            pat = (r"\\begin\{" + re.escape(env) + r"\}(.*?)\\end\{"
                   + re.escape(env) + r"\}")
            for m in re.finditer(pat, src, re.S):
                spans.append((m.start(), m.end()))
        return spans

    # ---- macro calls, labels, sections ------------------------------------
    def _read_arg(self, pos: int):
        """pos points at '{'. Return (arg_text, index_after_closing_brace)."""
        end = self._skip_group(pos, "{", "}")
        return self.src[pos + 1:end - 1], end

    def find_calls(self, macro: str):
        out = []
        for m in re.finditer(r"\\" + macro + r"\s*(\[[^\]]*\])?\s*\{", self.src):
            b = self.src.index("{", m.end() - 1)
            try:
                arg, end = self._read_arg(b)
            except ValueError:
                continue
            out.append({"start": m.start(), "end": end, "arg": arg,
                        "line": self.line_of(m.start()),
                        "inert": bool(self.comment[m.start()] or self.iffalse[m.start()])})
        return out

    def _build_index(self):
        src = self.src
        self.calls = {mac: self.find_calls(mac) for mac in REVISION_MACROS}
        self.blocks = [
            {"start": m.start(), "end": m.end(), "arg": m.group(1),
             "line": self.line_of(m.start()),
             "inert": bool(self.comment[m.start()] or self.iffalse[m.start()])}
            for m in re.finditer(
                r"\\begin\{oldrevisionblock\}(.*?)\\end\{oldrevisionblock\}", src, re.S)]

        # SUPPRESSED mask: active \oldtext bodies and active oldrevisionblocks.
        suppressed = bytearray(self.n)
        for item in self.calls["oldtext"] + self.blocks:
            if item["inert"]:
                continue
            for k in range(item["start"], item["end"]):
                suppressed[k] = 1
        self.suppressed = suppressed

        self.labels = [(m.start(), self.line_of(m.start()), m.group(1))
                       for m in re.finditer(r"\\label\{([^}]+)\}", src)]

        self.sections = []
        for m in re.finditer(r"\\(" + "|".join(SECTIONING) + r")\*?\s*\{", src):
            b = src.index("{", m.end() - 1)
            try:
                arg, _ = self._read_arg(b)
            except ValueError:
                arg = ""
            self.sections.append({"start": m.start(), "line": self.line_of(m.start()),
                                  "kind": m.group(1), "title": self._plain(arg)})
        ab = src.find("\\begin{abstract}")
        if ab >= 0:
            self.sections.append({"start": ab, "line": self.line_of(ab),
                                  "kind": "abstract", "title": "Abstract"})
        ap = src.find("\n\\appendix")
        self.appendix_start = ap if ap >= 0 else self.n
        self.sections.sort(key=lambda s: s["start"])

        self.todos = [{"line": self.line_of(m.start()), "start": m.start(),
                       "bucket": self.bucket_at(m.start())}
                      for m in re.finditer(r"\\todo\s*(\[[^\]]*\])?\s*\{", src)]

    @staticmethod
    def _plain(arg: str) -> str:
        """Strip revision/typesetting macros from a heading for a readable title."""
        t = arg
        for _ in range(6):
            t = re.sub(r"\\texorpdfstring\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"\2", t)
            t = re.sub(r"\\(?:newtext|latesttext|advisorchange|textcolor|color|"
                       r"oldtext|mbox|text|emph|textbf|textit)\s*(\{[^{}]*\})?"
                       r"\s*\{([^{}]*)\}", r"\2", t)
        t = re.sub(r"\\[A-Za-z]+\*?", "", t)
        return " ".join(t.replace("{", "").replace("}", "").split())

    # ---- buckets ----------------------------------------------------------
    def bucket_at(self, pos: int) -> str:
        if self.iffalse[pos]:
            return "IFFALSE"
        if self.comment[pos]:
            return "COMMENTED"
        if self.suppressed[pos]:
            return "SUPPRESSED"
        return "ACTIVE"

    def bucket_of_line(self, line: int) -> str:
        """The bucket a whole source line belongs to (ACTIVE if any part prints)."""
        starts = self._line_starts()
        lo = starts[line - 1]
        hi = starts[line] if line < len(starts) else self.n
        seen = {self.bucket_at(k) for k in range(lo, max(lo + 1, hi))}
        for b in ("ACTIVE", "SUPPRESSED", "IFFALSE", "COMMENTED"):
            if b in seen:
                return b
        return "COMMENTED"

    def _line_starts(self):
        if not hasattr(self, "_ls"):
            ls, pos = [0], 0
            while True:
                pos = self.src.find("\n", pos)
                if pos < 0:
                    break
                pos += 1
                ls.append(pos)
            self._ls = ls
        return self._ls

    # ---- anchors ----------------------------------------------------------
    def nearest_label(self, pos: int) -> str:
        prev = [l for p, _ln, l in self.labels if p < pos]
        return prev[-1] if prev else "(before first label)"

    def section_of(self, pos: int) -> str:
        prev = [s for s in self.sections if s["start"] <= pos]
        if not prev:
            return "(preamble)"
        s = prev[-1]
        prefix = "Appendix " if s["start"] >= self.appendix_start else ""
        return f"{prefix}{s['kind']}: {s['title']}" if s["title"] else \
            f"{prefix}{s['kind']}"

    def anchor(self, pos: int, span: int = 160) -> dict:
        """The full edit-stable anchor record for a source position."""
        line = self.line_of(pos)
        raw = self.line_text(line)
        excerpt = c.norm_excerpt(raw if raw.strip() else self.src[pos:pos + span])
        return {
            "baseline_line": line,
            "manuscript_section": self.section_of(pos),
            "source_anchor": self.nearest_label(pos),
            "render_bucket": self.bucket_at(pos),
            "baseline_excerpt_norm": excerpt,
            "baseline_excerpt_sha256": c.sha256_text(excerpt),
        }

    def anchor_for_line(self, line: int) -> dict:
        return self.anchor(self._line_starts()[line - 1])

    # ---- census self-check ------------------------------------------------
    def census(self) -> dict:
        out = {}
        for mac in REVISION_MACROS:
            items = self.calls[mac]
            a = sum(1 for x in items if not x["inert"])
            out["\\" + mac] = {"active": a, "inert": len(items) - a,
                               "total": len(items)}
        a = sum(1 for x in self.blocks if not x["inert"])
        out["oldrevisionblock"] = {"active": a, "inert": len(self.blocks) - a,
                                   "total": len(self.blocks)}
        return out

    def archive_census(self) -> dict:
        """The census table markup_preflight.py already committed, parsed back."""
        text = c.read_text(c.PAPER_ANALYSIS / "superseded_text_archive.md")
        out = {}
        for m in re.finditer(
                r"^\|\s*`?(\\\\?[A-Za-z]+|oldrevisionblock)`?\s*\|\s*(\d+)\s*\|"
                r"\s*(\d+)\s*\|\s*(\d+)\s*\|", text, re.M):
            key = m.group(1).replace("\\\\", "\\")
            out[key] = {"active": int(m.group(2)), "inert": int(m.group(3)),
                        "total": int(m.group(4))}
        return out

    # ---- numeric tokens ---------------------------------------------------
    def numeric_tokens(self, active_only: bool = False):
        """Every numeric token in the body, bucket-tagged and classified.

        Masked out: the preamble, math environments, and the numeric arguments
        of reference / typesetting macros (MASK_MACROS). What remains is the
        bounded population the coverage audit must account for.
        """
        rows = []
        for m in NUMERIC_TOKEN.finditer(self.src):
            pos = m.start()
            if pos < self.preamble_end:
                continue
            if self.math[pos] or self.refmask[pos]:
                continue
            bucket = self.bucket_at(pos)
            if active_only and bucket != "ACTIVE":
                continue
            tok = m.group(0)
            rows.append({"token": tok, "kind": classify_token(tok),
                         "start": pos, **self.anchor(pos)})
        return rows

    # ---- float (table / figure) inventory ---------------------------------
    def floats(self):
        """Every table/figure environment with its labels, bucket and caption."""
        out = []
        for env in FLOAT_ENVS:
            pat = (r"\\begin\{" + re.escape(env) + r"\}(.*?)\\end\{"
                   + re.escape(env) + r"\}")
            for m in re.finditer(pat, self.src, re.S):
                body = m.group(1)
                labels = re.findall(r"\\label\{([^}]+)\}", body)
                cap = re.search(r"\\caption\s*(\[[^\]]*\])?\s*\{", body)
                caption = ""
                if cap:
                    b = m.start(1) + cap.end() - 1
                    try:
                        caption = self._plain(self._read_arg(b)[0])
                    except ValueError:
                        caption = ""
                out.append({
                    "env": env, "labels": labels,
                    "line": self.line_of(m.start()),
                    "end_line": self.line_of(m.end() - 1),
                    "bucket": self.bucket_at(m.start()),
                    "caption": caption[:200],
                    # this project wraps \includegraphics in \safeincludegraphics
                    "graphics": re.findall(
                        r"\\(?:safe)?includegraphics(?:\[[^\]]*\])?\{([^}]+)\}",
                        body),
                })
        out.sort(key=lambda r: r["line"])
        return out


def classify_token(tok: str) -> str:
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", tok):
        return "date"
    if "e" in tok or "E" in tok:
        return "scientific"
    if "." in tok:
        return "decimal"
    plain = tok.replace(",", "")
    v = int(plain)
    if len(plain) == 4 and 1900 <= v <= 2100:
        return "year_like"
    if v >= 1000:
        return "int_ge_1000"
    if v >= 100:
        return "int_mid"
    return "int_small"


# --------------------------------------------------------------------------
def self_check(verbose: bool = True) -> list:
    """Verify the re-implemented partition against externally committed facts."""
    t = Tex()
    problems = []

    got, want = t.census(), t.archive_census()
    for key, w in want.items():
        g = got.get(key)
        if g != w:
            problems.append(f"census mismatch {key}: got {g}, archive says {w}")

    if t.sha256 != c.TEX_SHA256:
        problems.append(f"tex sha256 {t.sha256} != pinned {c.TEX_SHA256}")

    if t.iffalse_regions != list(c.IFFALSE_REGIONS):
        problems.append(f"iffalse regions {t.iffalse_regions} != "
                        f"{list(c.IFFALSE_REGIONS)}")

    if len(t.todos) != 19:
        problems.append(f"expected 19 \\todo sites, found {len(t.todos)}")

    if verbose:
        print(f"tex            {c.rel(t.path)}  sha256 {t.sha256[:16]}...")
        print(f"lines          {t.src.count(chr(10)) + 1}   chars {t.n}")
        print(f"preamble ends  L{t.preamble_end_line} (\\begin{{document}})")
        print(f"iffalse        {t.iffalse_regions}")
        print(f"todo sites     {len(t.todos)}  buckets "
              f"{ {b: sum(1 for x in t.todos if x['bucket'] == b) for b in sorted({x['bucket'] for x in t.todos})} }")
        print("census (re-implemented vs committed archive):")
        for k in sorted(want):
            flag = "ok " if got.get(k) == want[k] else "BAD"
            print(f"  {flag} {k:20} {got.get(k)}")
        toks = t.numeric_tokens()
        act = [r for r in toks if r["render_bucket"] == "ACTIVE"]
        from collections import Counter
        print(f"numeric tokens total {len(toks)}  by bucket "
              f"{dict(Counter(r['render_bucket'] for r in toks))}")
        print(f"  ACTIVE {len(act)} occurrences, "
              f"{len({r['token'] for r in act})} distinct; by kind "
              f"{dict(Counter(r['kind'] for r in act))}")
        fl = t.floats()
        print(f"floats {len(fl)}  by env+bucket "
              f"{dict(Counter((r['env'], r['bucket']) for r in fl))}")
        print(f"PROBLEMS: {len(problems)}")
        for p in problems:
            print("  -", p)
    return problems


if __name__ == "__main__":
    raise SystemExit(1 if self_check() else 0)
