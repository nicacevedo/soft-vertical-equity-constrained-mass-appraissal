"""Archive superseded revision text and run the blocking empty-group pre-flight.

Usage: markup_tool.py <tex> <archive.md>
Exits non-zero if any enclosing caption/heading would become empty.
"""
import re, sys, json

TEX, ARCHIVE = sys.argv[1], sys.argv[2]
src = open(TEX, encoding="utf-8").read()
lines = src.split("\n")
n = len(src)

def line_of(pos):
    return src.count("\n", 0, pos) + 1

# ---- 1. map inert regions: % comments (unescaped) and \iffalse...\fi ----
comment = [False]*n
i = 0
while i < n:
    c = src[i]
    if c == "\\":
        i += 2; continue
    if c == "%":
        j = src.find("\n", i)
        j = n if j < 0 else j
        for k in range(i, j): comment[k] = True
        i = j
        continue
    i += 1

iffalse = [False]*n
for m in re.finditer(r"^\\iffalse", src, re.M):
    end = re.search(r"^\\fi", src[m.start():], re.M)
    if end:
        for k in range(m.start(), m.start()+end.end()): iffalse[k] = True

def active(pos): return not comment[pos] and not iffalse[pos]

# ---- 2. brace-balanced argument reader ----
def read_arg(pos):
    """pos points at '{'. Return (arg_text, index_after_closing_brace)."""
    assert src[pos] == "{"
    depth, i = 0, pos
    while i < n:
        c = src[i]
        if c == "\\": i += 2; continue
        if c == "{": depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0: return src[pos+1:i], i+1
        i += 1
    raise ValueError(f"unbalanced brace at line {line_of(pos)}")

def find_calls(macro):
    out = []
    for m in re.finditer(r"\\" + macro + r"\s*\{", src):
        b = src.index("{", m.start())
        try: arg, end = read_arg(b)
        except ValueError as e: 
            print(f"WARN: {e}"); continue
        out.append({"start": m.start(), "end": end, "arg": arg,
                    "line": line_of(m.start()), "active": active(m.start())})
    return out

old   = find_calls("oldtext")
new   = find_calls("newtext")
latest= find_calls("latesttext")
adv   = find_calls("advisorchange")

# oldrevisionblock environments
blocks = []
for m in re.finditer(r"\\begin\{oldrevisionblock\}(.*?)\\end\{oldrevisionblock\}", src, re.S):
    blocks.append({"start": m.start(), "end": m.end(), "arg": m.group(1),
                   "line": line_of(m.start()), "active": active(m.start())})

# ---- 3. section label context ----
labels = [(m.start(), m.group(1)) for m in re.finditer(r"\\label\{([^}]+)\}", src)]
def nearest_label(pos):
    prev = [l for p, l in labels if p < pos]
    return prev[-1] if prev else "(before first label)"

# ---- 4. simulate the preamble redefinition, then pre-flight ----
edits = []
for c in old + blocks:
    edits.append((c["start"], c["end"], ""))                 # \oldtext -> {}
for c in new + latest + adv:
    edits.append((c["start"], c["end"], c["arg"]))           # -> identity
edits.sort(key=lambda e: e[0])
buf, cur = [], 0
for s, e, rep in edits:
    if s < cur: continue          # nested call already consumed by an outer one
    buf.append(src[cur:s]); buf.append(rep); cur = e
buf.append(src[cur:])
sim = "".join(buf)

def scan_empty(text, macro):
    bad = []
    for m in re.finditer(r"\\" + macro + r"\s*(\[[^\]]*\])?\s*\{", text):
        b = text.index("{", m.end()-1)
        depth, i = 0, b
        while i < len(text):
            ch = text[i]
            if ch == "\\": i += 2; continue
            if ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0: break
            i += 1
        arg = text[b+1:i]
        stripped = re.sub(r"\\(label|texorpdfstring)\s*\{[^}]*\}", "", arg)
        stripped = re.sub(r"[\s{}\\]", "", stripped)
        if not stripped:
            bad.append((text.count("\n", 0, m.start())+1, macro, repr(arg[:80])))
    return bad

problems = []
for mac in ("caption", "section", "subsection", "subsubsection", "paragraph", "title"):
    problems += scan_empty(sim, mac)

# ---- 5. write archive ----
def census(name, items):
    a = sum(1 for x in items if x["active"])
    return f"| `{name}` | {a} | {len(items)-a} | {len(items)} |"

with open(ARCHIVE, "w", encoding="utf-8") as f:
    f.write("# Superseded revision text — archive\n\n")
    f.write(f"Extracted from `paper/{TEX.split('/')[-1]}` at git HEAD "
            f"`faa2adae66fd1725de5ecfdb536bda2ff0ed522b`, before the Tier-A markup neutralization.\n\n")
    f.write("Every passage below is text the manuscript had **already marked as deleted** "
            "(struck-through `\\oldtext`, crossed `oldrevisionblock`, or `%`-commented). "
            "Tier A stops rendering it; this file preserves it verbatim.\n\n")
    f.write("## Census\n\n| macro | active | inert (comment/`\\iffalse`) | total |\n|---|---:|---:|---:|\n")
    f.write("\n".join([census("\\oldtext", old), census("\\newtext", new),
                       census("\\latesttext", latest), census("\\advisorchange", adv),
                       census("oldrevisionblock", blocks)]) + "\n\n")
    f.write("## Suppressed passages (`\\oldtext`, active only)\n\n")
    for c in old:
        if not c["active"]: continue
        f.write(f"### L{c['line']} · near `{nearest_label(c['start'])}`\n\n```latex\n{c['arg'].strip()}\n```\n\n")
    if any(b["active"] for b in blocks):
        f.write("## Suppressed `oldrevisionblock` environments\n\n")
        for c in blocks:
            if not c["active"]: continue
            f.write(f"### L{c['line']} · near `{nearest_label(c['start'])}`\n\n```latex\n{c['arg'].strip()}\n```\n\n")
    f.write("## Inert `%`-commented superseded passages (retained in source, never rendered)\n\n")
    for c in old + blocks:
        if c["active"]: continue
        f.write(f"- L{c['line']} · near `{nearest_label(c['start'])}` — "
                f"`{' '.join(c['arg'].split())[:150]}`\n")

print(json.dumps({
    "oldtext_active": sum(1 for c in old if c["active"]),
    "oldtext_inert":  sum(1 for c in old if not c["active"]),
    "newtext_active": sum(1 for c in new if c["active"]),
    "latesttext_active": sum(1 for c in latest if c["active"]),
    "advisorchange_active": sum(1 for c in adv if c["active"]),
    "oldrevisionblock_active": sum(1 for c in blocks if c["active"]),
    "PREFLIGHT_empty_groups": problems,
}, indent=2))
sys.exit(1 if problems else 0)
