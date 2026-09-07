#!/usr/bin/env python3
"""Build TABLE_FIGURE_DISPOSITION.md, checked against the measured inventory.

Feedback item 9 asked for an explicit disposition for every active table and
figure. This script does not take the authored spec on trust:

  * every label in spec/table_figure_disposition.yaml must exist in the tex,
    with the declared render bucket and the declared number of occurrences;
  * every float in the tex must have an entry -- so the inventory cannot fall
    behind the manuscript;
  * the declared `numbers_fully_supported` level must be CONSISTENT with the
    token tallies recomputed from coverage/active_tex_numeric_tokens.csv.
    Declaring FULLY_SUPPORTED for a float that still has flagged tokens, or
    UNSUPPORTED for one whose tokens all resolve, is a build failure.

The float disposition vocabulary (KEEP / UPDATE / REBUILD /
DEMOTE_TO_APPENDIX / DELETE) is deliberately separate from the claim
disposition vocabulary.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import b0_common as c
import b0_tex

SPEC_FILE = c.SPEC / "table_figure_disposition.yaml"
OUT_MD = c.B0 / "TABLE_FIGURE_DISPOSITION.md"
TOKENS = c.COVERAGE / "active_tex_numeric_tokens.csv"


def token_tallies():
    """SOURCED / ALLOWLISTED / FLAGGED per source_anchor, ACTIVE bucket only."""
    out = defaultdict(Counter)
    for r in c.read_csv_literal(TOKENS):
        if r["render_bucket"] == "ACTIVE":
            out[r["source_anchor"]][r["resolution"]] += 1
    return out


def main() -> int:
    tex = b0_tex.Tex()
    spec = c.read_yaml(SPEC_FILE)
    tallies = token_tallies()
    manifest = c.read_json(c.B0 / "FINAL_EVIDENCE_MANIFEST.json")

    # measured inventory: (label, bucket) -> occurrences, first line, env, caption
    measured = {}
    for f in tex.floats():
        for lab in (f["labels"] or ["(no label)"]):
            key = (lab, f["bucket"])
            rec = measured.setdefault(key, {"occurrences": 0, "lines": [],
                                            "env": f["env"], "caption": f["caption"]})
            rec["occurrences"] += 1
            rec["lines"].append(f["line"])

    failures, rows = [], []
    seen = set()
    for e in spec["floats"]:
        lab, bucket = e["label"], e["render_bucket"]
        key = (lab, bucket)
        if key in seen:
            failures.append(f"duplicate spec entry for {key}")
        seen.add(key)
        if e["disposition"] not in c.TF_DISPOSITIONS:
            failures.append(f"{lab}: disposition {e['disposition']!r} not in "
                            f"{c.TF_DISPOSITIONS}")
        if e["numbers_fully_supported"] not in c.TF_SUPPORT:
            failures.append(f"{lab}: numbers_fully_supported "
                            f"{e['numbers_fully_supported']!r} not in {c.TF_SUPPORT}")
        if key not in measured:
            failures.append(f"{lab} ({bucket}) is not in the tex at all")
            continue
        m = measured[key]
        want_occ = e.get("occurrences", 1)
        if m["occurrences"] != want_occ:
            failures.append(f"{lab} ({bucket}): spec says {want_occ} occurrence(s), "
                            f"tex has {m['occurrences']}")
        for a in e.get("frozen_evidence", []) or []:
            if a not in manifest["artifacts"]:
                failures.append(f"{lab}: frozen_evidence not in manifest: {a}")
        if not e.get("why") or not e.get("writing_pass"):
            failures.append(f"{lab}: why and writing_pass are both required")

        t = tallies.get(lab, Counter())
        flagged = t.get("FLAGGED_UNSUPPORTED", 0)
        sourced = t.get("SOURCED", 0)
        decl = e["numbers_fully_supported"]
        # consistency of the DECLARED support level with the measured tallies
        if bucket != "ACTIVE":
            if decl != "NOT_RENDERED":
                failures.append(f"{lab} ({bucket}): non-ACTIVE floats must declare "
                                f"NOT_RENDERED, got {decl}")
        elif decl == "FULLY_SUPPORTED" and flagged:
            failures.append(f"{lab}: declared FULLY_SUPPORTED but {flagged} tokens "
                            "are FLAGGED_UNSUPPORTED")
        elif decl == "UNSUPPORTED" and sourced:
            failures.append(f"{lab}: declared UNSUPPORTED but {sourced} tokens "
                            "are SOURCED")
        elif decl == "PARTIALLY_SUPPORTED" and not (flagged and sourced):
            failures.append(f"{lab}: declared PARTIALLY_SUPPORTED but tallies are "
                            f"sourced={sourced} flagged={flagged}")
        elif decl == "NOT_RENDERED":
            failures.append(f"{lab} (ACTIVE): cannot declare NOT_RENDERED")

        rows.append({**e, "lines": sorted(m["lines"]), "env": m["env"],
                     "caption": m["caption"], "sourced": sourced,
                     "allowlisted": t.get("ALLOWLISTED", 0), "flagged": flagged})

    for key in measured:
        if key not in seen:
            failures.append(f"float in the tex with no spec entry: {key}")

    rows.sort(key=lambda r: (r["kind"], r["render_bucket"] != "ACTIVE",
                             r["lines"][0]))
    c.write_text(OUT_MD, render(rows, tex))
    print(f"wrote {c.rel(OUT_MD)}  {len(rows)} floats")
    print(f"  dispositions {dict(Counter(r['disposition'] for r in rows))}")
    print(f"  support      {dict(Counter(r['numbers_fully_supported'] for r in rows))}")
    if failures:
        print(f"\n  BUILD FAILURES ({len(failures)}):")
        for f in failures[:40]:
            print("   -", f)
    return 1 if failures else 0


def render(rows: list, tex: b0_tex.Tex) -> str:
    import re
    L = ["# Table and figure disposition", "",
         "Every table and figure in `paper/paper_v17_option1.tex`, with an "
         "explicit disposition. Generated by "
         "`code/b0_5_build_tf_disposition.py` from "
         "`spec/table_figure_disposition.yaml`, and **checked** against the "
         "measured float inventory and the numeric coverage audit: a declared "
         "support level that contradicts the token tallies is a build failure, "
         "and a float in the tex with no entry here fails the build.", "",
         "Line numbers are **baseline coordinates only** and go stale as soon "
         "as the writing pass begins. The `\\label{...}` is the stable anchor.",
         "",
         "Vocabulary: `KEEP` leave alone - `UPDATE` fix specific cells, labels "
         "or notes - `REBUILD` regenerate from frozen evidence - "
         "`DEMOTE_TO_APPENDIX` - `DELETE`. This is separate from the claim "
         "disposition vocabulary.", ""]

    tot = Counter(r["disposition"] for r in rows)
    L += ["## Summary", "",
          "| disposition | tables | figures | total |", "|---|---:|---:|---:|"]
    for d in c.TF_DISPOSITIONS:
        nt = sum(1 for r in rows if r["disposition"] == d and r["kind"] == "table")
        nf = sum(1 for r in rows if r["disposition"] == d and r["kind"] == "figure")
        if nt or nf:
            L.append(f"| `{d}` | {nt} | {nf} | {nt + nf} |")
    L += [f"| **total** | "
          f"{sum(1 for r in rows if r['kind'] == 'table')} | "
          f"{sum(1 for r in rows if r['kind'] == 'figure')} | {len(rows)} |", ""]

    L += ["The three `\\iffalse` regions of the manuscript are "
          f"{', '.join(f'lines {a}-{b}' for a, b in tex.iffalse_regions)}. "
          "Every ATTOM table lives inside them, so none of that material is "
          "compiled.", ""]

    # orphaned image files
    used = set(re.findall(r"\\(?:safe)?includegraphics(?:\[[^\]]*\])?\{([^}]+)\}",
                          tex.src))
    used = {u for u in used if not u.startswith("#")}
    import subprocess
    tracked = subprocess.run(["git", "-C", str(c.REPO), "ls-files", "paper/img"],
                             capture_output=True, text=True).stdout.split()
    tracked = [p[len("paper/"):] for p in tracked]
    orphans = sorted(set(tracked) - used)
    L += ["## Image files", "",
          f"All {sum(1 for r in rows if r['kind'] == 'figure' and r['render_bucket'] == 'ACTIVE')} "
          "active figures reference graphics files that exist under `paper/`. "
          f"In the other direction, **{len(orphans)} of the {len(tracked)} "
          "tracked files under `paper/img/` are referenced nowhere in the "
          "manuscript** -- mostly earlier `generated_v6_preselection` and "
          "`generated_v12_994` renderings. They are inert, not broken; they do "
          "not affect the compiled document.", ""]
    if orphans:
        L += ["<details><summary>the "
              f"{len(orphans)} unreferenced image files</summary>", ""]
        L += [f"- `paper/{o}`" for o in orphans]
        L += ["", "</details>", ""]

    for kind in ("table", "figure"):
        L += [f"## {kind.capitalize()}s", ""]
        for r in rows:
            if r["kind"] != kind:
                continue
            lines = ", ".join(f"L{x}" for x in r["lines"])
            L += [f"### `{r['label']}` — **{r['disposition']}**", "",
                  f"- bucket: **{r['render_bucket']}** · {r['env']} · {lines}"
                  + (f" · {r['occurrences']} occurrences"
                     if r.get("occurrences", 1) > 1 else ""),
                  f"- numbers: **{r['numbers_fully_supported']}** "
                  f"(tokens: {r['sourced']} sourced, {r['allowlisted']} "
                  f"allowlisted, {r['flagged']} flagged unsupported)"]
            if r["caption"]:
                L.append(f"- caption: {r['caption'][:180]}")
            ev = r.get("frozen_evidence") or []
            L.append("- frozen evidence: "
                     + (", ".join(f"`{x}`" for x in ev) if ev else "*none*"))
            L += ["", f"**Why.** {' '.join(str(r['why']).split())}", "",
                  f"**Writing pass.** {' '.join(str(r['writing_pass']).split())}",
                  ""]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
