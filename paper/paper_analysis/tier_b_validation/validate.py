#!/usr/bin/env python3
"""The Tier-B gate: run the checks, classify every finding, report, exit.

The canonical manuscript is intentionally NOT compliant with the final-state
checks when this validator is built. So the pass criterion for a stage is

    UNEXPECTED_VALIDATOR_FAILURES == 0

and NOT "the unrevised manuscript passes every final-state check". A validator
that passes on the baseline is not validating anything, so no check is weakened
to make the baseline green.

Every expected failure is declared individually in
``spec/expected_failures.yaml`` with the stage that resolves it -- never as one
blanket exemption. That is what makes a resolved failure REQUIRED to disappear:
once a stage has run, any finding whose registry entry named that stage is
reported as UNEXPECTED. A declared count that changes is likewise unexpected,
because the token trajectory (356 -> 218 -> ... -> 0) is contractual rather than
advisory.

Three outcomes per finding:

  EXPECTED     a registry entry matches, its stage has not run yet, and the
               count (if declared) agrees exactly.
  UNEXPECTED   anything else. This is the gate.
  STALE        a registry entry matched nothing while its stage was still
               pending -- reported, because a failure vanishing early is a
               change worth seeing, not a silent win.

Usage:  python3 validate.py --stage B1.1 [--compile] [--json out.json]
"""
from __future__ import annotations

import sys

# Set before ANY project import: running the gate must not leave a stray
# __pycache__ anywhere, least of all under analysis/.
sys.dont_write_bytecode = True

import argparse
import fnmatch
import json

import tb_common as tb
import tb_checks
import tb_scope

REGISTRY = tb.TB_SPEC / "expected_failures.yaml"


def _stage_index(stage: str) -> int:
    if stage not in tb.STAGES:
        raise SystemExit(f"unknown stage {stage!r}; known: {', '.join(tb.STAGES)}")
    return tb.STAGES.index(stage)


def load_registry() -> list:
    doc = tb.read_yaml(REGISTRY) or {}
    return list(doc.get("expected_failures") or [])


def classify(findings, stage: str, registry) -> dict:
    si = _stage_index(stage)
    used = set()
    expected, unexpected = [], []
    for f in findings:
        hit = None
        for i, e in enumerate(registry):
            if fnmatch.fnmatchcase(f.key, e["key"]):
                hit = (i, e)
                break
        if hit is None:
            unexpected.append((f, "no expected-failure entry declares this"))
            continue
        i, e = hit
        used.add(i)
        ri = _stage_index(e["resolved_by_stage"])
        if si >= ri:
            unexpected.append(
                (f, f"declared resolved by {e['resolved_by_stage']}, which has "
                    f"already run"))
            continue
        if "expected_count" in e and f.count != e["expected_count"]:
            unexpected.append(
                (f, f"expected count {e['expected_count']}, measured {f.count}"))
            continue
        expected.append((f, e))
    stale = [e for i, e in enumerate(registry)
             if i not in used and _stage_index(e["resolved_by_stage"]) > si]
    return {"expected": expected, "unexpected": unexpected, "stale": stale}


def expected_token_total(stage: str, registry) -> int:
    """The flagged-token budget this stage is contractually allowed."""
    si = _stage_index(stage)
    tot = 0
    for e in registry:
        if e["key"].startswith("C02:flagged:") and "expected_count" in e:
            if _stage_index(e["resolved_by_stage"]) > si:
                tot += e["expected_count"]
    return tot


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    help="the Tier-B stage just completed, e.g. B1.1")
    ap.add_argument("--compile", action="store_true",
                    help="also run latexmk and report reference/citation state")
    ap.add_argument("--json", default=None, help="write the full result as JSON")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    registry = load_registry()
    ctx, findings = tb_checks.run_all()
    cls = classify(findings, a.stage, registry)
    scope = tb_scope.all_scope_checks()

    cov = ctx.coverage
    budget = expected_token_total(a.stage, registry)
    token_ok = cov["flagged_unsupported"] == budget
    # Cross-check the per-anchor budget against the trajectory declared in the
    # registry header. If the two disagree the registry is internally
    # inconsistent, which must surface as a failure rather than as a number the
    # validator quietly prefers.
    declared = (tb.read_yaml(REGISTRY) or {}).get("expected_token_trajectory", {})
    traj = declared.get(a.stage)
    traj_ok = traj is None or traj == budget

    comp = None
    if a.compile:
        import tb_compile
        comp = tb_compile.compile_manuscript()
        comp.pop("stdout_tail", None)

    n_unexpected = len(cls["unexpected"])
    ok = (n_unexpected == 0 and scope["ok"] and token_ok and traj_ok
          and (comp is None or comp["ok"]))

    if not a.quiet:
        print("=" * 78)
        print(f"TIER-B MANUSCRIPT VALIDATOR   stage={a.stage}")
        print(f"manuscript {cov['manuscript']}  sha256 {cov['manuscript_sha256'][:16]}...")
        print(f"  (Tier-A baseline manuscript: "
              f"{'unchanged' if cov['manuscript_is_tier_a_baseline'] else 'edited'})")
        print("=" * 78)
        print(f"\nACTIVE numeric tokens {cov['active_tokens']}: "
              f"{cov['active_resolution']}")
        print(f"UNSUPPORTED_TOKENS  measured={cov['flagged_unsupported']}  "
              f"expected_at_{a.stage}={budget}  "
              f"{'OK' if token_ok else '<-- MISMATCH, a finding'}")
        if not traj_ok:
            print(f"  *** registry inconsistency: declared trajectory for "
                  f"{a.stage} is {traj}, per-anchor budget is {budget} ***")
        for anchor, n in sorted(cov["flagged_unsupported_by_anchor"].items(),
                                key=lambda kv: -kv[1]):
            print(f"    {n:5d}  {anchor}")

        print(f"\n--- checks ---")
        by_check = {}
        for f in findings:
            by_check.setdefault(f.check, []).append(f)
        for cid, label, _ in tb_checks.CHECKS:
            fs = by_check.get(cid, [])
            unexp = [f for f, _ in cls["unexpected"] if f.check == cid]
            state = ("PASS" if not fs else
                     ("EXPECTED_ONLY" if not unexp else "UNEXPECTED"))
            print(f"  {cid}  {state:14s} {len(fs):3d} finding(s)  {label}")

        print(f"\n--- write scope and frozen-subtree immutability ---")
        print(f"  cumulative ({scope['cumulative']['command']}): "
              f"{scope['cumulative']['n_paths']} path(s), "
              f"outside paper/: {scope['cumulative']['outside_paper'] or 'none'}")
        print(f"  uncommitted: {scope['uncommitted']['n_paths']} path(s), "
              f"outside paper/: {scope['uncommitted']['outside_paper'] or 'none'}")
        for s in scope["frozen_subtrees"]["subtrees"]:
            print(f"  {s['tag']}  {s['subtree']}: "
                  f"{'EMPTY (identical)' if s['empty'] else 'DIVERGED'}")

        if comp is not None:
            print(f"\n--- compile ---")
            print(f"  latexmk exit {comp['exit_code']}  pdf={comp['pdf_written']}  "
                  f"pages={comp['pages']}")
            print(f"  undefined references: {comp['undefined_references'] or 'none'}")
            print(f"  undefined citations : {comp['undefined_citations'] or 'none'}")
            print(f"  missing bib entries : {comp['missing_bib_entries'] or 'none'}")
            print(f"  LaTeX errors        : {comp['n_latex_errors']}")
            print(f"  overfull hbox {comp['n_overfull_hbox']}  "
                  f"vbox {comp['n_overfull_vbox']}  underfull {comp['n_underfull']}")
            if comp["latex_warnings"]:
                for w in comp["latex_warnings"][:10]:
                    print(f"    warning: {w}")

        if cls["unexpected"]:
            print(f"\n*** UNEXPECTED VALIDATOR FAILURES ({n_unexpected}) ***")
            for f, why in cls["unexpected"]:
                print(f"  {f}\n      -> {why}")
        if cls["stale"]:
            print(f"\n--- stale expectations ({len(cls['stale'])}): declared but "
                  f"not observed, while their stage is still pending ---")
            for e in cls["stale"]:
                print(f"  {e['key']}  (resolved_by {e['resolved_by_stage']})")

        print(f"\nEXPECTED_FAILURES            = {len(cls['expected'])}")
        print(f"UNEXPECTED_VALIDATOR_FAILURES = {n_unexpected}")
        print(f"UNSUPPORTED_TOKENS            = {cov['flagged_unsupported']}")
        print(f"\nTIER_B_VALIDATOR = {'PASS' if ok else 'FAIL'}  (stage {a.stage})")

    if a.json:
        payload = {
            "stage": a.stage,
            "manuscript": cov["manuscript"],
            "manuscript_sha256": cov["manuscript_sha256"],
            "active_resolution": cov["active_resolution"],
            "unsupported_tokens": cov["flagged_unsupported"],
            "unsupported_tokens_expected": budget,
            "unsupported_tokens_by_anchor": cov["flagged_unsupported_by_anchor"],
            "n_expected_failures": len(cls["expected"]),
            "n_unexpected_failures": n_unexpected,
            "expected_failures": [{"finding": f.as_dict(),
                                   "resolved_by_stage": e["resolved_by_stage"]}
                                  for f, e in cls["expected"]],
            "unexpected_failures": [{"finding": f.as_dict(), "why": w}
                                    for f, w in cls["unexpected"]],
            "stale_expectations": cls["stale"],
            "scope": scope,
            "compile": comp,
            "ok": ok,
        }
        tb.write_json(a.json, payload)
        if not a.quiet:
            print(f"\nwrote {a.json}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
