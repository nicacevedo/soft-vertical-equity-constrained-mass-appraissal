#!/usr/bin/env python3
"""Build manuscript_claim_map.csv from spec/claims.yaml and the numeric map.

A CLAIM is a scientific statement the manuscript makes (or must make). Each one
carries a disposition from a closed vocabulary, and the gate that matters most is
on QUALIFY_AS_EXPLORATORY:

    QUALIFY_AS_EXPLORATORY is legal ONLY when the exact underlying frozen number
    is source-resolvable -- i.e. the claim names numeric_claim_ids and every one
    of them resolves in manuscript_numeric_map.csv with an attained status.

If the number is not resolvable, the disposition must be DELETE_OR_REPLACE or
REWRITE_WITH_SUPPORTED_EVIDENCE. Relabelling an unsupported numerical result
"exploratory" does not make it publishable. The qualitative idea may survive; the
unsupported table or cell may not. That rule is enforced here and re-checked in
tests/test_b0_claim_map.py.

Every claim also records, mechanically:
  * edit-stable anchors (manuscript_section, latex_label, source_anchor, a
    normalized baseline excerpt and its sha256) -- because line numbers are
    baseline coordinates only and go stale the moment Tier B starts editing;
  * reference_cell and comparison_purpose -- A is the assessor-facing workflow
    benchmark, C is the primary penalty-isolating reference, and neither may be
    silently substituted for the other;
  * counting_unit -- required for every aggregate/count claim, so an entry-level
    ED2 count can never be mixed with a unique-realization one.
"""
from __future__ import annotations

from pathlib import Path

import b0_common as c
import b0_tex

CLAIMS_SPEC = c.SPEC / "claims.yaml"
COVERAGE_SPEC = c.SPEC / "mandatory_coverage.yaml"
TODO_SPEC = c.SPEC / "todo_closure.yaml"
MAP_CSV = c.B0 / "manuscript_claim_map.csv"
NUMERIC_MAP = c.B0 / "manuscript_numeric_map.csv"

FIELDS = [
    "claim_id", "manuscript_section", "latex_label", "source_anchor",
    "manuscript_location", "baseline_line", "render_bucket",
    "baseline_excerpt_norm", "baseline_excerpt_sha256",
    "claim_text", "disposition", "reason", "required_final_message",
    "forbidden_wording", "primary_or_sensitivity", "reference_cell",
    "comparison_purpose", "counting_unit", "guidance_status",
    "supporting_artifacts", "numeric_claim_ids", "coverage_topics",
    "legacy_plan_id", "todo_id", "option2_precedent", "priority", "confidence",
]


def main() -> int:
    tex = b0_tex.Tex()
    manifest = c.read_json(c.B0 / "FINAL_EVIDENCE_MANIFEST.json")
    numeric = {r["claim_id"]: r for r in c.read_csv_literal(NUMERIC_MAP)}
    spec = c.read_yaml(CLAIMS_SPEC)
    todos = c.read_yaml(TODO_SPEC)
    coverage = c.read_yaml(COVERAGE_SPEC)

    rows, failures = [], []
    seen = set()

    for cl in spec["claims"]:
        cid = cl["claim_id"]
        if cid in seen:
            failures.append(f"duplicate claim_id {cid}")
        seen.add(cid)

        line = int(cl["baseline_line"])
        anchor = tex.anchor_for_line(line)
        disp = cl["disposition"]
        if disp not in c.DISPOSITIONS:
            failures.append(f"{cid}: disposition {disp!r} not in {c.DISPOSITIONS}")

        nids = cl.get("numeric_claim_ids", []) or []
        missing = [n for n in nids if n not in numeric]
        if missing:
            failures.append(f"{cid}: numeric_claim_ids not in the numeric map: {missing}")

        # --- the exploratory gate -------------------------------------------
        if disp == "QUALIFY_AS_EXPLORATORY":
            if not nids:
                failures.append(
                    f"{cid}: QUALIFY_AS_EXPLORATORY with no numeric_claim_ids. "
                    "The exact underlying frozen number must be source-resolvable; "
                    "otherwise the disposition must be DELETE_OR_REPLACE or "
                    "REWRITE_WITH_SUPPORTED_EVIDENCE.")
            for n in nids:
                st = numeric.get(n, {}).get("attained_status")
                if st not in ("ATTAINED", "NOT_ATTAINED"):
                    failures.append(
                        f"{cid}: QUALIFY_AS_EXPLORATORY but numeric claim {n} has "
                        f"attained_status={st!r}; only a resolvable number may be "
                        "qualified as exploratory.")

        arts = cl.get("supporting_artifacts", []) or []
        for a in arts:
            # TIER_B0_SELF: a Tier-B0 output cannot appear in its own manifest
            if a.startswith("analysis/final_manuscript_evidence/"):
                if not (c.REPO / a).exists():
                    failures.append(f"{cid}: Tier-B0 artifact does not exist: {a}")
            elif a != "NONE_REQUIRED" and a not in manifest["artifacts"]:
                failures.append(f"{cid}: supporting artifact not in manifest: {a}")
        if not arts:
            failures.append(f"{cid}: no supporting_artifacts (use NONE_REQUIRED)")

        for field, vocab, default in (
                ("primary_or_sensitivity", c.PRIMARY_OR_SENSITIVITY, "NOT_APPLICABLE"),
                ("reference_cell", c.REFERENCE_CELLS, "NONE"),
                ("comparison_purpose", c.COMPARISON_PURPOSES, "NOT_APPLICABLE"),
                ("counting_unit", c.COUNTING_UNITS, "NOT_APPLICABLE"),
                ("guidance_status", c.GUIDANCE_STATUS, "NOT_APPLICABLE")):
            v = cl.get(field, default)
            if v not in vocab:
                failures.append(f"{cid}: {field}={v!r} not in {vocab}")

        # A is never the penalty-isolating reference; C is never a benchmark
        if cl.get("reference_cell") == "A" and \
                cl.get("comparison_purpose") == "PENALTY_ISOLATING":
            failures.append(
                f"{cid}: reference_cell A with comparison_purpose "
                "PENALTY_ISOLATING. A is the assessor-facing workflow benchmark; "
                "the frozen convention forbids attributing changes relative to A "
                "to rho.")
        if cl.get("reference_cell") == "C" and \
                cl.get("comparison_purpose") == "WORKFLOW_BENCHMARK":
            failures.append(
                f"{cid}: reference_cell C labelled WORKFLOW_BENCHMARK. C is the "
                "PRIMARY within-path penalty-isolating reference.")

        conf = str(cl.get("confidence", ""))
        if conf not in ("HIGH", "MEDIUM", "LOW"):
            failures.append(f"{cid}: confidence={conf!r} must be HIGH/MEDIUM/LOW")
        if not cl.get("reason"):
            failures.append(f"{cid}: no reason")
        if not cl.get("required_final_message"):
            failures.append(f"{cid}: no required_final_message")

        tid = str(cl["todo_id"]) if "todo_id" in cl else ""
        loc = f"L{line}|{anchor['source_anchor']}"
        rows.append({
            "claim_id": cid,
            "manuscript_section": anchor["manuscript_section"],
            "latex_label": cl.get("latex_label", ""),
            "source_anchor": anchor["source_anchor"],
            "manuscript_location": loc,
            "baseline_line": line,
            "render_bucket": anchor["render_bucket"],
            "baseline_excerpt_norm": anchor["baseline_excerpt_norm"],
            "baseline_excerpt_sha256": anchor["baseline_excerpt_sha256"],
            "claim_text": " ".join(str(cl["claim_text"]).split()),
            "disposition": disp,
            "reason": " ".join(str(cl["reason"]).split()),
            "required_final_message": " ".join(
                str(cl["required_final_message"]).split()),
            "forbidden_wording": ";".join(cl.get("forbidden_wording", []) or []),
            "primary_or_sensitivity": cl.get("primary_or_sensitivity",
                                             "NOT_APPLICABLE"),
            "reference_cell": cl.get("reference_cell", "NONE"),
            "comparison_purpose": cl.get("comparison_purpose", "NOT_APPLICABLE"),
            "counting_unit": cl.get("counting_unit", "NOT_APPLICABLE"),
            "guidance_status": cl.get("guidance_status", "NOT_APPLICABLE"),
            "supporting_artifacts": ";".join(arts),
            "numeric_claim_ids": ";".join(nids),
            "coverage_topics": ";".join(cl.get("coverage_topics", []) or []),
            "legacy_plan_id": cl.get("legacy_plan_id", ""),
            "todo_id": tid,
            "option2_precedent": cl.get("option2_precedent", ""),
            "priority": cl.get("priority", ""),
            "confidence": conf,
        })

    # ---- every \todo site covered exactly once ---------------------------
    todo_ids = [t["todo_id"] for t in todos["todos"]]
    if sorted(todo_ids) != list(range(1, 20)):
        failures.append(f"todo_closure.yaml must cover todo_id 1..19, has {todo_ids}")
    tex_lines = [t["line"] for t in tex.todos]
    spec_lines = [t["baseline_line"] for t in todos["todos"]]
    if sorted(spec_lines) != sorted(tex_lines):
        failures.append(f"todo lines {sorted(spec_lines)} != tex {sorted(tex_lines)}")
    claimed = [r["todo_id"] for r in rows if r["todo_id"] != ""]
    for t in todos["todos"]:
        n = claimed.count(str(t["todo_id"]))
        if n != 1:
            failures.append(f"todo_id {t['todo_id']} referenced by {n} claims, need 1")

    # ---- every mandated coverage topic mapped ----------------------------
    mapped = set()
    for r in rows:
        mapped.update(x for x in r["coverage_topics"].split(";") if x)
    for topic in coverage["topics"]:
        if topic["topic_id"] not in mapped:
            failures.append(f"mandatory coverage topic unmapped: {topic['topic_id']}")

    rows.sort(key=lambda r: r["claim_id"])
    c.write_csv(MAP_CSV, FIELDS, rows)

    from collections import Counter
    print(f"wrote {c.rel(MAP_CSV)}  {len(rows)} claims")
    print(f"  dispositions   {dict(Counter(r['disposition'] for r in rows))}")
    print(f"  priorities     {dict(Counter(r['priority'] for r in rows))}")
    print(f"  purposes       {dict(Counter(r['comparison_purpose'] for r in rows))}")
    print(f"  todo sites     {len(claimed)}/19 referenced")
    print(f"  coverage       {len(mapped)}/{len(coverage['topics'])} topics mapped")
    if failures:
        print(f"\n  BUILD FAILURES ({len(failures)}):")
        for f in failures[:40]:
            print("    -", f)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
