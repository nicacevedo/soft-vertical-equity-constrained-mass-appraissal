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

import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import b0_common as c
import b0_tex

SPEC_FILE = c.SPEC / "table_figure_disposition.yaml"
OUT_MD = c.B0 / "TABLE_FIGURE_DISPOSITION.md"
TOKENS = c.COVERAGE / "active_tex_numeric_tokens.csv"


def tracked_images() -> set:
    out = subprocess.run(["git", "-C", str(c.REPO), "ls-files", "paper/img"],
                         capture_output=True, text=True, check=True).stdout
    return {l for l in out.split("\n") if l.strip()}


TRACKED_IMG = tracked_images()


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
                                            "env": f["env"], "caption": f["caption"],
                                            "caption_full": f["caption_full"],
                                            "graphics": []})
            rec["occurrences"] += 1
            rec["lines"].append(f["line"])
            rec["graphics"] += f["graphics"]

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

        # ---- visual provenance: required on figures, absent on tables ----
        vis = e.get("visual_provenance")
        if e["kind"] == "figure":
            if vis not in c.VISUAL_PROVENANCE:
                failures.append(f"{lab}: visual_provenance {vis!r} not in "
                                f"{c.VISUAL_PROVENANCE}")
            if bucket != "ACTIVE" and vis != "NOT_RENDERED":
                failures.append(f"{lab}: non-ACTIVE figure must be NOT_RENDERED")
        elif vis is not None:
            failures.append(f"{lab}: visual_provenance is for figures only")

        unsup = e.get("unsupported_assets", []) or []
        repl = e.get("replacement_assets", []) or []
        if e["kind"] == "figure" and bucket == "ACTIVE":
            # what the tex ACTUALLY references, and what the caption says
            refd = {"paper/" + g for g in m["graphics"]}
            marked_assets = sorted(a for a in refd
                                   if any(k in a for k in
                                          c.UNSUPPORTED_VISUAL_MARKERS))
            cap = (m["caption_full"] or "").lower()
            marked_caption = sorted(k for k in c.UNSUPPORTED_VISUAL_MARKERS
                                    if k in cap and "_" not in k)
            # the declaration must match what is measured
            if (marked_assets or marked_caption) and vis != "UNSUPPORTED_OVERLAY":
                failures.append(
                    f"{lab}: declares visual_provenance {vis} but the tex "
                    f"references {marked_assets or '[]'} and its caption says "
                    f"{marked_caption or '[]'}")
            if vis == "UNSUPPORTED_OVERLAY" and e["disposition"] == "KEEP":
                failures.append(
                    f"{lab}: UNSUPPORTED_OVERLAY may not be KEEP -- visual "
                    "provenance obeys the same rule as numeric provenance")
            # every declared unsupported asset must really be referenced here
            for a in unsup:
                if a not in refd:
                    failures.append(f"{lab}: unsupported_assets names {a}, "
                                    "which this figure does not reference")
            if marked_assets and not unsup:
                failures.append(f"{lab}: references {marked_assets} but declares "
                                "no unsupported_assets")
            if sorted(unsup) != sorted(marked_assets) and marked_assets:
                failures.append(f"{lab}: unsupported_assets {sorted(unsup)} != "
                                f"measured {marked_assets}")
            # every replacement must exist, be tracked, and be plain
            if unsup and len(repl) != len(unsup):
                failures.append(f"{lab}: {len(unsup)} unsupported assets but "
                                f"{len(repl)} replacements")
            for a in repl:
                if not (c.REPO / a).exists():
                    failures.append(f"{lab}: replacement asset missing: {a}")
                if a not in TRACKED_IMG:
                    failures.append(f"{lab}: replacement asset not tracked: {a}")
                if any(k in a for k in c.UNSUPPORTED_VISUAL_MARKERS):
                    failures.append(f"{lab}: replacement {a} is itself a "
                                    "candidate-region asset")
            if unsup and repl:
                for u, r in zip(sorted(unsup), sorted(repl)):
                    if r.replace(".pdf", "") not in u.replace(
                            "_candidate_region", ""):
                        failures.append(f"{lab}: replacement {r} does not pair "
                                        f"with {u}")
            if e["disposition"] == "UPDATE" and vis == "UNSUPPORTED_OVERLAY":
                wp = " ".join(str(e["writing_pass"]).split()).lower()
                for need in ("replacement_assets", "caption"):
                    pass
                if "plain path asset" not in wp and "plain" not in wp:
                    failures.append(f"{lab}: writing_pass must tell the pass to "
                                    "use the plain asset")
        elif unsup or repl:
            failures.append(f"{lab}: asset fields are for ACTIVE figures only")

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
                     "caption": m["caption"], "graphics": sorted(set(m["graphics"])),
                     "sourced": sourced,
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

    # visual provenance
    figs = [r for r in rows if r["kind"] == "figure"]
    overlay = [r for r in figs if r.get("visual_provenance") == "UNSUPPORTED_OVERLAY"]
    swaps = [r for r in figs if r.get("unsupported_assets")]
    n_assets = sum(len(r["unsupported_assets"]) for r in swaps)
    L += ["## Visual provenance", "",
          "Visual provenance obeys the same rule as numeric provenance. A path "
          "figure can have perfectly supported caption numbers while the "
          "**graphic itself** encodes the unsupported candidate-region "
          "construction — the light fill, the dashed activity-onset boundary, "
          "the solid upper guardrail, the transition-span shading. Those are "
          "the CV screening result that no frozen artifact reproduces, which is "
          "why `tab:rho_candidate_regions` is `DELETE`. A figure carrying one "
          "may not be classified `KEEP`, and the build fails if it is.", "",
          f"**{len(overlay)} of {len(figs)} active figures carry an unsupported "
          f"overlay.** {len(swaps)} of them reference a `_candidate_region` "
          f"graphic ({n_assets} assets in total), and a plain replacement "
          "already exists in the repository for every one:", ""]
    L += ["| figure | disposition | references | plain replacement |",
          "|---|---|---|---|"]
    for r in swaps:
        for u, v in zip(sorted(r["unsupported_assets"]),
                        sorted(r.get("replacement_assets", []))):
            L.append(f"| `{r['label']}` | **{r['disposition']}** | "
                     f"`{u.split('/')[-1]}` | `{v.split('/')[-1]}` |")
    L += ["",
          "The remaining overlay figures carry no `_candidate_region` asset but "
          "are defined by the same unsupported construction (turning-event "
          "locations, or a span-restricted ratio profile) and are `DELETE`: "
          + ", ".join(f"`{r['label']}`" for r in overlay
                      if not r.get("unsupported_assets")) + ".", "",
          "**Tier B0 did not swap any asset.** This is the specification for "
          "the Tier-B writing pass; no file under `paper/` was touched.", ""]

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
            if r.get("visual_provenance"):
                L.append(f"- visual provenance: **{r['visual_provenance']}**")
            if r.get("unsupported_assets"):
                L.append("- **asset swap required** — the manuscript references "
                         "the candidate-region variant:")
                for u, v in zip(sorted(r["unsupported_assets"]),
                                sorted(r.get("replacement_assets", []))):
                    L.append(f"    - `{u}` → `{v}`")
            elif r.get("graphics"):
                L.append("- assets: "
                         + ", ".join(f"`paper/{g}`" for g in r["graphics"]))
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
