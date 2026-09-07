#!/usr/bin/env python3
"""The numeric coverage audit, re-run against the LIVE manuscript.

Tier B0 accounted for every numeric token in the Tier-A baseline: 846 ACTIVE
tokens resolving to 300 SOURCED, 190 ALLOWLISTED and **356
FLAGGED_UNSUPPORTED**, concentrated in exactly eight anchors. That number is the
writing pass's odometer, and it must be produced mechanically at every stage
rather than asserted.

This module deliberately reuses the frozen algorithm instead of paraphrasing it:

  * the token population and the four render buckets come from ``b0_tex.Tex``,
    pointed at the live file;
  * the SOURCED index is rebuilt from ``manuscript_numeric_map.csv`` exactly as
    ``b0_2_build_numeric_map.coverage`` builds it -- keyed on
    ``(display_value, source_anchor)`` with the ``$``/``,`` and leading-``-``
    variants;
  * the allowlist and the no-wildcard policy come from ``spec/``.

The no-wildcard rule matters more than it looks. Inside a block under deletion
review a bare ``7`` is far likelier to be "7/7 LOFO events" -- an unsupported
RESULT -- than the seven-fold design constant, so wildcard allowlist rows are
refused at those eight anchors. Every number there is accounted for
individually, which is the mechanical half of the binding rule: an unsupported
numeric result is not publishable because it was relabelled "exploratory".

Verified at B1.0: run against the unedited manuscript this reproduces the frozen
tally and the frozen per-anchor breakdown exactly. A coverage audit that cannot
reproduce the frozen baseline is not measuring the same thing.
"""
from __future__ import annotations

import sys

# Set before ANY project import: running the gate must not leave a stray
# __pycache__ anywhere, least of all under analysis/.
sys.dont_write_bytecode = True

import tb_common as tb
import tb_ledger


def _sourced_index(rows):
    """(token, source_anchor) -> [claim_id], as the frozen builder keys it."""
    idx = {}
    for r in rows:
        dv = r["display_value"]
        key = dv.replace("$", "").replace(",", "")
        for tok in {dv, key, dv.lstrip("-"), key.lstrip("-")}:
            if tok:
                idx.setdefault((tok, r["source_anchor"]), []).append(r["claim_id"])
    return idx


def audit(tex=None) -> dict:
    """Resolve every numeric token in the live manuscript."""
    tex = tex or tb.live_tex()
    rows = tb.frozen_numeric_map()
    allow = tb.frozen_allowlist()
    policy = tb.frozen_coverage_policy()
    no_wild = set(policy.get("no_wildcard_anchors", []))

    exact, wild = {}, {}
    for a in allow:
        if a["source_anchor"] != "*":
            exact[(a["token"], a["source_anchor"])] = a
        else:
            wild[a["token"]] = a

    sourced = _sourced_index(rows)

    # The frozen numeric map indexes only the BASELINE manuscript's tokens, so a
    # number this pass adds has no map row. Section H.2 of the plan gives it a
    # second resolution path: a Tier-B provenance ledger entry. That cannot
    # launder an unsupported value -- C01 re-derives every entry from a frozen
    # artifact under analysis/ through all seven steps, so an entry can only
    # resolve a number that genuinely comes from the frozen evidence. (The
    # no-wildcard anchors restrict ALLOWLISTING, which asserts a number is a
    # constant; sourcing DERIVES it, so the restriction does not apply here.)
    ledger_index = {}
    for e in tb_ledger.load():
        anchor = e.get("manuscript_anchor")
        rendered = str(e.get("rendered_value", ""))
        if not anchor or not rendered:
            continue
        key = rendered.replace("$", "").replace(",", "")
        for tok in {rendered, key, rendered.lstrip("-"), key.lstrip("-")}:
            if tok:
                ledger_index.setdefault((tok, anchor), []).append(
                    e.get("entry_id", "(unnamed)"))

    tokens, tally, by_anchor = [], {}, {}
    for t in tex.numeric_tokens():
        bucket = t["render_bucket"]
        tok, anchor = t["token"], t["source_anchor"]
        plain = tok.replace(",", "")
        if bucket != "ACTIVE":
            res, reason, ids = "NOT_RENDERED", bucket, ""
        elif (tok, anchor) in sourced or (plain, anchor) in sourced:
            hit = sourced.get((tok, anchor)) or sourced.get((plain, anchor))
            res, reason, ids = "SOURCED", "", ";".join(sorted(set(hit)))
        elif (tok, anchor) in ledger_index or (plain, anchor) in ledger_index:
            hit = (ledger_index.get((tok, anchor))
                   or ledger_index.get((plain, anchor)))
            res = "SOURCED_LEDGER"
            reason = "Tier-B provenance ledger (recomputed by C01)"
            ids = ";".join(sorted(set(hit)))
        elif (tok, anchor) in exact:
            res, reason, ids = "ALLOWLISTED", exact[(tok, anchor)]["reason"], ""
        elif (plain, anchor) in exact:
            res, reason, ids = "ALLOWLISTED", exact[(plain, anchor)]["reason"], ""
        elif (tok in wild or plain in wild) and anchor not in no_wild:
            res = "ALLOWLISTED"
            reason = (wild.get(tok) or wild.get(plain))["reason"]
            ids = ""
        else:
            res, reason, ids = "FLAGGED_UNSUPPORTED", "no frozen source", ""
        tally[res] = tally.get(res, 0) + 1
        if res == "FLAGGED_UNSUPPORTED" and bucket == "ACTIVE":
            by_anchor[anchor] = by_anchor.get(anchor, 0) + 1
        tokens.append({"token": tok, "kind": t["kind"], "render_bucket": bucket,
                       "line": t["baseline_line"],
                       "manuscript_section": t["manuscript_section"],
                       "source_anchor": anchor,
                       "excerpt": t["baseline_excerpt_norm"],
                       "resolution": res, "reason": reason, "claim_ids": ids})

    active = [t for t in tokens if t["render_bucket"] == "ACTIVE"]

    def _tally(rs, field):
        out = {}
        for r in rs:
            out[r[field]] = out.get(r[field], 0) + 1
        return out

    return {
        "manuscript": tb.rel(tex.path),
        "manuscript_sha256": tex.sha256,
        "manuscript_is_tier_a_baseline": tex.sha256 == tb.BASELINE_TEX_SHA256,
        "tokens_total": len(tokens),
        "tokens_by_bucket": _tally(tokens, "render_bucket"),
        "active_tokens": len(active),
        "active_resolution": _tally(active, "resolution"),
        "flagged_unsupported": sum(1 for t in active
                                   if t["resolution"] == "FLAGGED_UNSUPPORTED"),
        "sourced_via_ledger": sum(1 for t in active
                                  if t["resolution"] == "SOURCED_LEDGER"),
        "flagged_unsupported_by_anchor": dict(sorted(by_anchor.items(),
                                                     key=lambda kv: -kv[1])),
        "tokens": tokens,
    }


def reproduces_frozen_baseline(result: dict) -> tuple:
    """Does this audit reproduce the certified Tier-B0 numbers? (only meaningful
    on the unedited manuscript, and that is exactly when it is checked)."""
    frozen = tb.frozen_coverage_summary()
    problems = []
    if result["manuscript_sha256"] != frozen["manuscript_sha256"]:
        return False, ["manuscript has been edited; frozen comparison not applicable"]
    if result["active_resolution"] != frozen["active_resolution"]:
        problems.append(f"active_resolution {result['active_resolution']} != "
                        f"frozen {frozen['active_resolution']}")
    if result["flagged_unsupported_by_anchor"] != frozen["flagged_unsupported_by_anchor"]:
        problems.append("per-anchor flagged breakdown differs from the frozen audit")
    if result["tokens_by_bucket"] != frozen["tokens_by_bucket"]:
        problems.append(f"tokens_by_bucket {result['tokens_by_bucket']} != "
                        f"frozen {frozen['tokens_by_bucket']}")
    return (not problems), problems


if __name__ == "__main__":
    import json
    r = audit()
    ok, probs = reproduces_frozen_baseline(r)
    print(json.dumps({k: v for k, v in r.items() if k != "tokens"},
                     indent=2, sort_keys=True))
    print("reproduces frozen baseline:", ok, probs)
