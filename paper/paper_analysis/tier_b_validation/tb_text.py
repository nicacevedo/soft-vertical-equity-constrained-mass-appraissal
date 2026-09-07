#!/usr/bin/env python3
"""What the manuscript actually SAYS: active prose, sentence units, normalization.

The wording checks (reference-cell semantics, D1/D3 semantics, ED2 guidance
status, counting units, forbidden phrases) all need the same three things, and
getting any of them wrong makes a check either blind or noisy:

1. **Only the ACTIVE build.** A phrase inside ``\\oldtext{...}``, an
   ``oldrevisionblock``, a comment or an ``\\iffalse`` region does not print, so
   it is not a claim. The Tier-A preamble decides this by macro identity, and
   the frozen ``b0_tex`` partition is the authority; this module reuses it
   rather than re-deriving it.

2. **Sentence-level scope.** "must not" three paragraphs away does not license a
   forbidden phrase here, and two mutually exclusive count families are only a
   violation *in one statement*. So the unit of judgement is a sentence, and
   table cells (``&``-separated) and ``\\\\`` row breaks are sentence boundaries
   too -- a table cell is a statement.

3. **LaTeX-aware normalization that does not destroy the phrase.** Stripping
   every macro turns "Custom-objective $\\rho=0$ origin" into
   "custom-objective =0 origin", which would make the forward display-name check
   silently unsatisfiable. Named symbol macros are therefore mapped to plain
   words BEFORE generic macro stripping.

Character offsets are preserved through masking (inert characters become
spaces), so every finding can be reported at a real line number in the live
file rather than at a stale baseline coordinate.
"""
from __future__ import annotations

import re

# Named macros that carry MEANING in a phrase. Mapped to plain text before the
# generic macro strip, because dropping them corrupts the phrase itself.
SYMBOL_MACROS = {
    "rho": "rho", "beta": "beta", "Beta": "beta", "Delta": "Delta",
    "delta": "delta", "alpha": "alpha", "sigma": "sigma", "tau": "tau",
    "mu": "mu", "nu": "nu", "lambda": "lambda", "epsilon": "epsilon",
    "approx": "~=", "pm": "+/-", "times": "x", "cdot": "*",
    "ge": ">=", "geq": ">=", "le": "<=", "leq": "<=", "neq": "!=",
    "log": "log", "sqrt": "sqrt", "in": " in ", "to": " to ",
    "rightarrow": " to ", "leftarrow": " from ", "ldots": "...",
    "dots": "...", "cdots": "...", "infty": "inf",
}

# Escaped literals: these are characters, not macros.
ESCAPES = {r"\%": "%", r"\$": "$", r"\&": "&", r"\_": "_", r"\#": "#",
           r"\{": "{", r"\}": "}", r"\,": " ", r"\;": " ", r"\:": " ",
           r"\ ": " ", r"\-": ""}

_ABBREV = (
    "e.g", "i.e", "cf", "vs", "etc", "approx", "resp", "al", "Fig", "fig",
    "Eq", "eq", "Sec", "sec", "Tab", "tab", "No", "no", "Ref", "ref",
    "Appendix", "App", "Dr", "Mr", "Ms", "St", "Inc", "Ch", "ch", "p", "pp",
)
_ABBREV_RE = re.compile(r"(?:" + "|".join(re.escape(a) for a in _ABBREV) + r")\.$")


def normalize(text: str) -> str:
    """LaTeX source -> lowercase plain text suitable for phrase matching."""
    t = text
    for k, v in ESCAPES.items():
        t = t.replace(k, v)
    # Named symbol macros first, longest name first so \Delta beats \delta.
    for name in sorted(SYMBOL_MACROS, key=len, reverse=True):
        t = re.sub(r"\\" + name + r"(?![A-Za-z])", SYMBOL_MACROS[name], t)
    # \operatorname{RMSE}, \mathrm{NL}, \text{...}: keep the argument.
    for _ in range(6):
        t = re.sub(r"\\(?:operatorname|mathrm|mathbf|mathcal|mathit|text|textbf"
                   r"|textit|texttt|emph|textsc|textsuperscript|textcolor"
                   r"|latesttext|newtext|oldtext|advisorchange|makecell)"
                   r"\s*\*?\s*(?:\{[^{}]*\})?\s*\{([^{}]*)\}", r"\1", t)
    t = re.sub(r"\\[A-Za-z]+\*?", " ", t)        # remaining macros: drop
    t = t.replace("$", " ").replace("{", " ").replace("}", " ")
    t = t.replace("~", " ").replace("^", "").replace("--", "-")
    return " ".join(t.split()).lower()


class ActiveText:
    """The ACTIVE build of the live manuscript, split into sentence units."""

    def __init__(self, tex):
        self.tex = tex
        src = tex.src
        n = len(src)
        # Character-preserving mask: keep only what prints, replace the rest
        # with spaces so every offset stays valid.
        keep = []
        for i, ch in enumerate(src):
            if i < tex.preamble_end or tex.bucket_at(i) != "ACTIVE":
                keep.append("\n" if ch == "\n" else " ")
            else:
                keep.append(ch)
        self.masked = "".join(keep)
        assert len(self.masked) == n
        self.units = self._split()
        self.plain = normalize(self.masked)

    # ---- sentence units --------------------------------------------------
    def _split(self):
        m = self.masked
        # Hard boundaries: sentence end, paragraph break, LaTeX row break,
        # table cell separator, and the start of a \caption or \item.
        bounds = {0, len(m)}
        for mt in re.finditer(r"(?<=[.!?])[\)\]\}]?\s+", m):
            head = m[max(0, mt.start() - 12):mt.start()]
            if re.search(r"\d\.$", head):        # 21.6 -- a decimal, not a stop
                continue
            if _ABBREV_RE.search(head.strip()):
                continue
            bounds.add(mt.end())
        for pat in (r"\n\s*\n", r"\\\\", r"(?<!\\)&", r"\\caption", r"\\item",
                    r"\\begin\{", r"\\end\{", r"\\(?:sub)*section"):
            for mt in re.finditer(pat, m):
                bounds.add(mt.start())
                bounds.add(mt.end())
        bs = sorted(bounds)
        out = []
        for a, b in zip(bs, bs[1:]):
            raw = m[a:b]
            if not raw.strip():
                continue
            out.append(Unit(raw, a, self.tex.line_of(a), self.tex))
        return out

    def find(self, needle: str):
        """Units whose normalized text contains a normalized needle."""
        nd = normalize(needle)
        return [u for u in self.units if nd and nd in u.norm]

    def count(self, needle: str) -> int:
        nd = normalize(needle)
        return self.plain.count(nd) if nd else 0


class Unit:
    __slots__ = ("raw", "start", "line", "norm", "_tex")

    def __init__(self, raw, start, line, tex):
        self.raw = raw
        self.start = start
        self.line = line
        self._tex = tex
        self.norm = normalize(raw)

    def __repr__(self):
        return f"<Unit L{self.line} {self.norm[:70]!r}>"

    @property
    def anchor(self) -> str:
        return self._tex.nearest_label(self.start)

    @property
    def section(self) -> str:
        return self._tex.section_of(self.start)

    def excerpt(self, limit: int = 140) -> str:
        return self.norm[:limit]
