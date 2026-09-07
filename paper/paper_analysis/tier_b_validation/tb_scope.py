#!/usr/bin/env python3
"""Write scope and frozen-subtree immutability, tested directly against git.

Neither check depends on a frozen guard that a paper edit breaks, which is the
whole point: the frozen suites protect the same property with assertions that
fail as soon as the manuscript is legitimately edited, so the protection has to
be re-established here in a form that survives the writing pass.

Two checks, complementary rather than redundant:

**(a) Cumulative branch write-scope.** ``git diff --name-only HEAD`` sees only
UNCOMMITTED changes, so once Tier-B commits exist it is blind to a stray write
made two stages ago. The binding check is cumulative across the whole branch,
``tier-b0-final-20260907..HEAD``, and every path it lists must be under
``paper/``. The uncommitted view is kept as well -- it catches a stray write
before it is committed, which is when undoing it is cheapest.

**(b) Frozen-subtree immutability**, per tag and per directory: the P0, P1 and
Tier-B0 subtrees must be byte-identical to their own tags. This is exactly what
the frozen guards were protecting.

(a) proves nothing outside ``paper/`` was touched; (b) proves the three frozen
subtrees specifically are untouched. A new file under ``output/``, ``utils/`` or
a fourth ``analysis/`` subdirectory would pass (b) and fail (a).
"""
from __future__ import annotations

import sys

# Set before ANY project import: running the gate must not leave a stray
# __pycache__ anywhere, least of all under analysis/.
sys.dont_write_bytecode = True

import tb_common as tb


def cumulative_write_scope() -> dict:
    """Every path changed since the Tier-B0 tag must be under paper/."""
    out = tb.git("diff", "--name-only", f"{tb.B0_TAG}..HEAD")
    paths = [p for p in out.split("\n") if p.strip()]
    outside = [p for p in paths if not p.startswith("paper/")]
    return {"command": f"git diff --name-only {tb.B0_TAG}..HEAD",
            "paths": paths, "n_paths": len(paths), "outside_paper": outside,
            "ok": not outside}


def uncommitted_write_scope() -> dict:
    """Nothing uncommitted, tracked or untracked, may sit outside paper/."""
    tracked = [p for p in tb.git("diff", "--name-only", "HEAD").split("\n")
               if p.strip()]
    porcelain = [l for l in tb.git("status", "--porcelain").split("\n") if l.strip()]
    untracked = [l[3:] for l in porcelain if l.startswith("??")]
    paths = sorted(set(tracked) | set(untracked))
    outside = [p for p in paths if not p.startswith("paper/")]
    return {"paths": paths, "n_paths": len(paths), "untracked": untracked,
            "outside_paper": outside, "ok": not outside}


def frozen_subtree_identity() -> dict:
    """The three tag diffs. All three must be empty."""
    results = []
    for tag, subtree in tb.FROZEN_SUBTREES:
        stat = tb.git("diff", "--stat", tag, "HEAD", "--", subtree).strip()
        results.append({"tag": tag, "subtree": subtree,
                        "diff_stat": stat, "empty": stat == ""})
    return {"subtrees": results, "ok": all(r["empty"] for r in results)}


def all_scope_checks() -> dict:
    cum = cumulative_write_scope()
    unc = uncommitted_write_scope()
    frz = frozen_subtree_identity()
    return {"cumulative": cum, "uncommitted": unc, "frozen_subtrees": frz,
            "ok": cum["ok"] and unc["ok"] and frz["ok"]}


if __name__ == "__main__":
    import json
    print(json.dumps(all_scope_checks(), indent=2, sort_keys=True))
