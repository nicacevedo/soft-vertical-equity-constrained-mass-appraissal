#!/usr/bin/env python3
"""Build manuscript_numeric_map.csv and the numeric coverage audit.

Authored vs derived, strictly:

  AUTHORED  in spec/numeric_claims.yaml -- WHICH manuscript cell, WHICH artifact,
            WHICH selector, which units, which transform, which rounding, which
            reference cell and comparison purpose, which counting unit.
  DERIVED   here -- raw_value (the literal decimal TEXT in the artifact),
            display_value (raw -> value_transform -> rounding_rule), the artifact
            sha256, and every manuscript anchor (section, latex_label,
            source_anchor, normalized excerpt + its sha256, baseline_line).

That split is what makes "raw_value must reproduce from the named selector" and
"display_value must be mechanically obtained from raw_value" enforceable rather
than aspirational.

The value pipeline has TWO separate stages. rounding_rule never carries a unit
conversion: a percent display is value_transform=times_100 plus half_up:<n>. The
frozen artifacts are inconsistent about units on purpose -- MAPE is a fraction
(0.2120...) shown as 21.2%, while VEI and COD are already in percent -- and that
is exactly the confusion the two-stage pipeline removes.

The coverage audit then requires EVERY numeric token in the ACTIVE build to
resolve to exactly one of SOURCED / ALLOWLISTED / FLAGGED_UNSUPPORTED. Tokens in
the SUPPRESSED, COMMENTED and IFFALSE buckets are recorded as NOT_RENDERED with
their bucket, so the writing pass can see what must not be resurrected.
"""
from __future__ import annotations

import re
from pathlib import Path

import b0_common as c
import b0_tex

SPEC_FILE = c.SPEC / "numeric_claims.yaml"
ALLOWLIST = c.SPEC / "unsupported_numbers_allowlist.csv"
POLICY = c.SPEC / "coverage_policy.yaml"
MAP_CSV = c.B0 / "manuscript_numeric_map.csv"
TOKENS_CSV = c.COVERAGE / "active_tex_numeric_tokens.csv"
COVERAGE_JSON = c.COVERAGE / "coverage_summary.json"

MAP_FIELDS = [
    "claim_id", "domain", "metric",
    # ---- manuscript side: anchors, not just line numbers ----
    "manuscript_section", "latex_label", "source_anchor",
    "manuscript_location", "baseline_line", "render_bucket",
    "baseline_excerpt_norm", "baseline_excerpt_sha256",
    "table_row", "table_col",
    # ---- evidence side ----
    "artifact_path", "artifact_kind", "artifact_sha256", "scientific_stage",
    "selector", "hash_source",
    # ---- the deterministic value pipeline ----
    "raw_value", "raw_unit", "value_transform", "display_unit",
    "rounding_rule", "display_value", "attained_status",
    # ---- semantics the guards read ----
    "reference_cell", "comparison_purpose", "primary_or_sensitivity",
    "counting_unit", "guidance_status", "evidence_note",
]


def _manifest():
    return c.read_json(c.B0 / "FINAL_EVIDENCE_MANIFEST.json")


def _resolve_axes(template: str, combo: dict) -> str:
    """Substitute {axis} placeholders. Used for display strings, which may
    legitimately contain LaTeX braces such as $n_{\\mathrm{train}}$."""
    out = str(template)
    for k, v in combo.items():
        out = out.replace("{" + k + "}", str(v))
    return out


def _resolve_selector(template: str, combo: dict) -> str:
    """Same substitution, but an UNFILLED placeholder is a hard error: a selector
    with a live {axis} in it would silently fail to match the intended row."""
    out = _resolve_axes(template, combo)
    left = re.findall(r"\{([a-z_]+)\}", out)
    if left:
        raise SystemExit(f"unfilled selector placeholders {left} in {out!r}")
    return out


def _combos(axes: dict):
    if not axes:
        yield {}
        return
    keys = list(axes)
    def rec(i, acc):
        if i == len(keys):
            yield dict(acc)
            return
        for v in axes[keys[i]]:
            acc[keys[i]] = v
            yield from rec(i + 1, acc)
    yield from rec(0, {})


def _expand_entries(spec: dict, block: dict) -> list:
    """Expand a block's declarative metric_set into explicit entries.

    A block either lists `entries` outright, or names a `metric_set` plus a
    `column_map`. The declarative form keeps the authored YAML compact without a
    code generator: units, transform, rounding and guidance status are defined
    ONCE per metric in `metric_defs`, so MAPE cannot pick up VEI's units by a
    copy-paste slip.
    """
    if "entries" in block:
        return block["entries"]
    defs = spec["metric_defs"]
    colmap = spec["column_maps"][block["column_map"]]
    names = (block["metrics"] if "metrics" in block
             else spec["metric_sets"][block["metric_set"]])
    tmpl = block["claim_id_template"]
    lines_by_metric = block.get("lines_by_metric", {})
    out = []
    for name in names:
        d = defs[name]
        if name not in colmap:
            raise SystemExit(f"{block['block_id']}: column_map "
                             f"{block['column_map']!r} has no metric {name!r}")
        ent = {
            "claim_id": tmpl.replace("{metric}", name),
            "metric": name,
            "column": colmap[name],
            "table_col": d["tex_col"],
            "raw_unit": d["raw_unit"],
            "display_unit": d["display_unit"],
            "value_transform": d.get("value_transform", "identity"),
            "rounding_rule": d["rounding_rule"],
            "guidance_status": d.get("guidance_status", "NOT_APPLICABLE"),
        }
        if name in lines_by_metric:
            ent["baseline_line"] = lines_by_metric[name]
        if "table_row" in block:
            ent["table_row"] = block["table_row"]
        for k, v in block.get("entry_overrides", {}).get(name, {}).items():
            ent[k] = v
        out.append(ent)
    return out


def build_rows(tex: b0_tex.Tex, manifest: dict):
    spec = c.read_yaml(SPEC_FILE)
    rows, failures = [], []
    seen_ids = set()

    for block in spec["blocks"]:
        bid = block["block_id"]
        domain = block["domain"]
        art = block["artifact"]
        kind = block.get("artifact_kind", "csv")
        if art not in manifest["artifacts"]:
            failures.append(f"{bid}: artifact not in manifest: {art}")
            continue
        arec = manifest["artifacts"][art]
        if not arec["present_in_worktree"]:
            failures.append(f"{bid}: artifact is absent-legacy and may not be read: {art}")
            continue
        apath = c.REPO / art

        defaults = dict(
            raw_unit=block.get("raw_unit", "dimensionless"),
            display_unit=block.get("display_unit", "dimensionless"),
            value_transform=block.get("value_transform", "identity"),
            rounding_rule=block.get("rounding_rule", "half_up:3"),
            reference_cell=block.get("reference_cell", "NONE"),
            comparison_purpose=block.get("comparison_purpose", "NOT_APPLICABLE"),
            primary_or_sensitivity=block.get("primary_or_sensitivity", "DESCRIPTIVE"),
            counting_unit=block.get("counting_unit", "NOT_APPLICABLE"),
            guidance_status=block.get("guidance_status", "NOT_APPLICABLE"),
            attained_status=block.get("attained_status", "ATTAINED"),
        )

        # A block enumerates its manuscript cells either as a cross product of
        # `axes`, or as an explicit `combos` list when the baseline line numbers
        # differ per combination (table rows are not a product of the axes).
        enumerated = (block["combos"] if "combos" in block
                      else list(_combos(block.get("axes", {}))))
        try:
            block_entries = _expand_entries(spec, block)
        except SystemExit as e:
            failures.append(str(e))
            continue
        for combo in enumerated:
            for ent in block_entries:
                spec_row = {**defaults, **{k: v for k, v in ent.items()
                                           if k in defaults}}
                metric = _resolve_axes(ent["metric"], combo)
                selector = _resolve_selector(
                    ent.get("selector", block.get("selector_template", "")), combo)
                line = ent.get("baseline_line",
                               combo.get("line", block.get("baseline_line", 0)))
                line = int(_resolve_axes(str(line), combo))
                if line <= 0:
                    failures.append(f"{ent['claim_id']}: no baseline_line resolved")
                    continue
                anchor = tex.anchor_for_line(line)
                claim_id = _resolve_axes(ent["claim_id"], combo)

                # ---- resolve the value from the frozen artifact ----
                status = spec_row["attained_status"]
                if status == "NOT_ATTAINED":
                    raw = ""
                    disp = ent.get("display_value", "NOT_ATTAINED")
                    spec_row["rounding_rule"] = "none"
                    spec_row["value_transform"] = "identity"
                    # a test re-reads the source status; verify it here too
                    try:
                        att = c.select_csv_value(apath, selector,
                                                 ent.get("attained_column", "attained"))
                        if att not in ("False", "false", "0"):
                            failures.append(
                                f"{claim_id}: declared NOT_ATTAINED but source "
                                f"attained={att!r}")
                    except c.SelectorError as e:
                        failures.append(f"{claim_id}: {e}")
                else:
                    try:
                        raw = c.resolve_selector(apath, kind, selector,
                                                 ent.get("column", metric))
                        raw = "" if raw is None else str(raw)
                    except c.SelectorError as e:
                        failures.append(f"{claim_id}: {e}")
                        continue
                    try:
                        disp = c.to_display(raw, spec_row["value_transform"],
                                            spec_row["rounding_rule"])
                    except Exception as e:
                        failures.append(f"{claim_id}: display pipeline: {e}")
                        continue
                    expect = ent.get("expect_display")
                    if expect is not None and disp != str(expect):
                        failures.append(
                            f"{claim_id}: display_value {disp!r} != manuscript "
                            f"value {expect!r} (raw {raw!r}, "
                            f"{spec_row['value_transform']} + "
                            f"{spec_row['rounding_rule']})")

                loc = f"L{line}|{anchor['source_anchor']}"
                if ent.get("table_row"):
                    loc += f"|row={_resolve_axes(ent['table_row'], combo)}"
                if ent.get("table_col"):
                    loc += f"|col={_resolve_axes(ent['table_col'], combo)}"

                # claim_id names a SCIENTIFIC VALUE and is deliberately reused
                # wherever that value appears (abstract, table, discussion).
                # The primary key is the (claim_id, location, metric) triple.
                pk = (claim_id, loc, metric)
                if pk in seen_ids:
                    failures.append(f"duplicate primary key: {pk}")
                seen_ids.add(pk)

                rows.append({
                    "claim_id": claim_id, "domain": domain, "metric": metric,
                    "manuscript_section": anchor["manuscript_section"],
                    "latex_label": block.get("latex_label", ""),
                    "source_anchor": anchor["source_anchor"],
                    "manuscript_location": loc,
                    "baseline_line": line,
                    "render_bucket": anchor["render_bucket"],
                    "baseline_excerpt_norm": anchor["baseline_excerpt_norm"],
                    "baseline_excerpt_sha256": anchor["baseline_excerpt_sha256"],
                    "table_row": _resolve_axes(ent.get("table_row", ""), combo),
                    "table_col": _resolve_axes(ent.get("table_col", ""), combo),
                    "artifact_path": art, "artifact_kind": kind,
                    "artifact_sha256": arec["sha256"],
                    "scientific_stage": arec["scientific_stage"],
                    "selector": selector, "hash_source": arec["hash_source"],
                    "raw_value": raw, "raw_unit": spec_row["raw_unit"],
                    "value_transform": spec_row["value_transform"],
                    "display_unit": spec_row["display_unit"],
                    "rounding_rule": spec_row["rounding_rule"],
                    "display_value": disp,
                    "attained_status": status,
                    "reference_cell": spec_row["reference_cell"],
                    "comparison_purpose": spec_row["comparison_purpose"],
                    "primary_or_sensitivity": spec_row["primary_or_sensitivity"],
                    "counting_unit": spec_row["counting_unit"],
                    "guidance_status": spec_row["guidance_status"],
                    "evidence_note": ent.get("evidence_note",
                                             block.get("evidence_note", "")),
                })
    rows.sort(key=lambda r: (r["claim_id"], r["manuscript_location"], r["metric"]))
    return rows, failures


# --------------------------------------------------------------------------
def coverage(tex: b0_tex.Tex, rows: list):
    """Account for every numeric token in the manuscript body."""
    allow = c.read_csv_literal(ALLOWLIST) if ALLOWLIST.exists() else []
    policy = c.read_yaml(POLICY) if POLICY.exists() else {}
    no_wild = set(policy.get("no_wildcard_anchors", []))
    # allowlist keys: (token, source_anchor) with '*' wildcards on the anchor
    exact, wild = {}, {}
    for a in allow:
        (exact if a["source_anchor"] != "*" else wild)[
            (a["token"], a["source_anchor"]) if a["source_anchor"] != "*"
            else a["token"]] = a

    # display values SOURCED by the map, indexed by (token, source_anchor)
    sourced = {}
    for r in rows:
        key = r["display_value"].replace("$", "").replace(",", "")
        for tok in {r["display_value"], key,
                    r["display_value"].lstrip("-"), key.lstrip("-")}:
            if tok:
                sourced.setdefault((tok, r["source_anchor"]), []).append(r["claim_id"])

    out, tally = [], {}
    # the wildcard allowlist may not reach blocks under DELETE/REWRITE review
    for t in tex.numeric_tokens():
        bucket = t["render_bucket"]
        tok, anchor = t["token"], t["source_anchor"]
        plain = tok.replace(",", "")
        if bucket != "ACTIVE":
            res, reason, ids = "NOT_RENDERED", bucket, ""
        elif (tok, anchor) in sourced or (plain, anchor) in sourced:
            hit = sourced.get((tok, anchor)) or sourced.get((plain, anchor))
            res, reason, ids = "SOURCED", "", ";".join(sorted(set(hit)))
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
        out.append({"token": tok, "kind": t["kind"], "render_bucket": bucket,
                    "baseline_line": t["baseline_line"],
                    "manuscript_section": t["manuscript_section"],
                    "source_anchor": anchor,
                    "baseline_excerpt_norm": t["baseline_excerpt_norm"],
                    "baseline_excerpt_sha256": t["baseline_excerpt_sha256"],
                    "resolution": res, "reason": reason, "claim_ids": ids})
    return out, tally, sorted(no_wild)


def main() -> int:
    tex = b0_tex.Tex()
    manifest = _manifest()
    rows, failures = build_rows(tex, manifest)
    toks, tally, no_wild = coverage(tex, rows)

    c.write_csv(MAP_CSV, MAP_FIELDS, rows)
    c.write_csv(TOKENS_CSV,
                ["token", "kind", "render_bucket", "baseline_line",
                 "manuscript_section", "source_anchor", "baseline_excerpt_norm",
                 "baseline_excerpt_sha256", "resolution", "reason", "claim_ids"],
                toks)

    active = [t for t in toks if t["render_bucket"] == "ACTIVE"]
    flagged = [t for t in active if t["resolution"] == "FLAGGED_UNSUPPORTED"]
    by_anchor = {}
    for t in flagged:
        by_anchor[t["source_anchor"]] = by_anchor.get(t["source_anchor"], 0) + 1
    summary = {
        "schema_version": 1,
        "generated_by": c.rel(Path(__file__)),
        "manuscript": c.rel(tex.path), "manuscript_sha256": tex.sha256,
        "numeric_map_rows": len(rows),
        "distinct_claim_ids": len({r["claim_id"] for r in rows}),
        "tokens_total": len(toks),
        "tokens_by_bucket": _tally(toks, "render_bucket"),
        "active_tokens": len(active),
        "active_distinct_tokens": len({t["token"] for t in active}),
        "active_by_kind": _tally(active, "kind"),
        "active_resolution": _tally(active, "resolution"),
        "resolution_all_buckets": tally,
        "flagged_unsupported_by_anchor": dict(sorted(by_anchor.items(),
                                                     key=lambda kv: -kv[1])),
        "no_wildcard_anchors": sorted(no_wild),
        "build_failures": failures,
    }
    c.write_json(COVERAGE_JSON, summary)

    print(f"wrote {c.rel(MAP_CSV)}          {len(rows)} rows")
    print(f"wrote {c.rel(TOKENS_CSV)}  {len(toks)} tokens")
    print(f"wrote {c.rel(COVERAGE_JSON)}")
    print(f"  ACTIVE tokens {len(active)}: {summary['active_resolution']}")
    if by_anchor:
        print("  FLAGGED_UNSUPPORTED by anchor:")
        for k, v in list(summary["flagged_unsupported_by_anchor"].items())[:20]:
            print(f"    {v:5d}  {k}")
    if failures:
        print(f"\n  BUILD FAILURES ({len(failures)}):")
        for f in failures[:40]:
            print("    -", f)
    return 1 if failures else 0


def _tally(rows, field):
    out = {}
    for r in rows:
        out[r[field]] = out.get(r[field], 0) + 1
    return out


if __name__ == "__main__":
    raise SystemExit(main())
