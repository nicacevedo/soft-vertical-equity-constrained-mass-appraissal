#!/usr/bin/env python3
"""Numeric-map assertions: every value re-derives, independently of the builder."""
from __future__ import annotations

import contextlib
import io
from collections import Counter

import b0_common as c
import b0_tex

ROWS = c.read_csv_literal(c.B0 / "manuscript_numeric_map.csv")
TOKENS = c.read_csv_literal(c.COVERAGE / "active_tex_numeric_tokens.csv")
SUMMARY = c.read_json(c.COVERAGE / "coverage_summary.json")
MAN = c.read_json(c.B0 / "FINAL_EVIDENCE_MANIFEST.json")
CLAIMS = c.read_csv_literal(c.B0 / "manuscript_claim_map.csv")
POLICY = c.read_yaml(c.SPEC / "coverage_policy.yaml")
ALLOW = c.read_csv_literal(c.SPEC / "unsupported_numbers_allowlist.csv")
TEX = b0_tex.Tex()


def test_primary_key_is_unique():
    keys = Counter((r["claim_id"], r["manuscript_location"], r["metric"])
                   for r in ROWS)
    dups = [k for k, n in keys.items() if n > 1]
    assert not dups, f"duplicate primary keys: {dups[:5]}"
    assert len(ROWS) > 0


def test_claim_ids_are_reused_for_the_same_scientific_value():
    """A claim_id names a value, so the same id at two locations must carry the
    same raw_value. A silent divergence would mean two different numbers."""
    by = {}
    bad = []
    for r in ROWS:
        k = (r["claim_id"], r["metric"])
        if k in by and by[k] != r["raw_value"]:
            bad.append(f"{k}: {by[k]!r} vs {r['raw_value']!r}")
        by[k] = r["raw_value"]
    assert not bad, "\n".join(bad[:10])


def test_every_artifact_is_in_the_manifest_and_present():
    bad = []
    for r in ROWS:
        rec = MAN["artifacts"].get(r["artifact_path"])
        if rec is None:
            bad.append(f"{r['claim_id']}: {r['artifact_path']} not in manifest")
            continue
        if not rec["present_in_worktree"]:
            bad.append(f"{r['claim_id']}: reads an absent artifact")
        if rec["sha256"] != r["artifact_sha256"]:
            bad.append(f"{r['claim_id']}: artifact sha256 mismatch")
        if rec["scientific_stage"] != r["scientific_stage"]:
            bad.append(f"{r['claim_id']}: stage mismatch")
    assert not bad, "\n".join(bad[:20])


def test_every_selector_resolves_uniquely_and_raw_value_is_byte_identical():
    bad = []
    for r in ROWS:
        if r["attained_status"] == "NOT_ATTAINED":
            continue
        path = c.REPO / r["artifact_path"]
        col = r["metric"]
        # the map records `metric` as a short name; resolve via the same
        # selector and compare the literal text
        try:
            if r["artifact_kind"] == "csv":
                row = c.select_csv_row(path, r["selector"])
                hits = [v for k, v in row.items() if v == r["raw_value"]]
                if not hits:
                    bad.append(f"{r['claim_id']}: raw_value {r['raw_value']!r} "
                               "not found in the uniquely selected row")
            else:
                got = c.resolve_selector(path, r["artifact_kind"], r["selector"])
                if str(got) != r["raw_value"]:
                    bad.append(f"{r['claim_id']}: {got!r} != {r['raw_value']!r}")
        except c.SelectorError as e:
            bad.append(f"{r['claim_id']}: {e}")
    assert not bad, "\n".join(bad[:20])


def test_display_value_recomputes_from_raw_value():
    bad = []
    for r in ROWS:
        if not c.valid_transform(r["value_transform"]):
            bad.append(f"{r['claim_id']}: bad transform {r['value_transform']}")
            continue
        if not c.valid_rounding(r["rounding_rule"]):
            bad.append(f"{r['claim_id']}: bad rounding {r['rounding_rule']}")
            continue
        if r["attained_status"] == "NOT_ATTAINED":
            continue
        got = c.to_display(r["raw_value"], r["value_transform"],
                           r["rounding_rule"])
        if got != r["display_value"]:
            bad.append(f"{r['claim_id']}: {got!r} != {r['display_value']!r}")
    assert not bad, "\n".join(bad[:20])


def test_rounding_never_carries_a_unit_conversion():
    """pct_half_up was removed: a percent display must go through times_100."""
    assert not any("pct" in r["rounding_rule"] for r in ROWS)
    bad = []
    for r in ROWS:
        ru, du, tr = r["raw_unit"], r["display_unit"], r["value_transform"]
        if ru == du and tr not in ("identity", "absolute_value"):
            bad.append(f"{r['claim_id']}: same unit {ru} but transform {tr}")
        if ru == "fraction" and du == "percent" and tr != "times_100":
            bad.append(f"{r['claim_id']}: fraction->percent needs times_100, "
                       f"got {tr}")
        if ru == "percent" and du == "percent" and tr != "identity":
            bad.append(f"{r['claim_id']}: percent->percent must be identity")
    assert not bad, "\n".join(bad[:20])


def test_the_mixed_unit_trap_is_handled_correctly():
    """MAPE is a fraction shown as percent; VEI and COD are already percent.
    This is the exact confusion the two-stage pipeline exists to prevent."""
    got = {}
    for r in ROWS:
        if r["metric"] in ("MAPEP", "VEI", "COD"):
            got[r["metric"]] = (r["raw_unit"], r["value_transform"])
    assert got.get("MAPEP") == ("fraction", "times_100"), got
    assert got.get("VEI") == ("percent", "identity"), got
    assert got.get("COD") == ("percent", "identity"), got


def test_rounding_is_half_up_not_bankers():
    assert c.apply_rounding("0.025", "half_up:2") == "0.03"
    assert c.apply_rounding("0.015", "half_up:2") == "0.02"
    assert c.apply_rounding("-0.025", "half_up:2") == "-0.03"


def test_closed_vocabularies():
    bad = []
    for r in ROWS:
        for field, vocab in (("attained_status", c.ATTAINED_STATUS),
                             ("reference_cell", c.REFERENCE_CELLS),
                             ("comparison_purpose", c.COMPARISON_PURPOSES),
                             ("counting_unit", c.COUNTING_UNITS),
                             ("guidance_status", c.GUIDANCE_STATUS),
                             ("primary_or_sensitivity", c.PRIMARY_OR_SENSITIVITY),
                             ("render_bucket", c.RENDER_BUCKETS)):
            if r[field] not in vocab:
                bad.append(f"{r['claim_id']}: {field}={r[field]!r}")
    assert not bad, "\n".join(bad[:20])


def test_not_attained_states_are_still_not_attained_at_source():
    """The four frozen non-attainment states, verified at source. No interpolation."""
    for tbl in ("prb_inference.csv", "vei_significance.csv"):
        rows = c.read_csv_literal(c.P1_TABLES / tbl)
        na = [r for r in rows if r["attained"] != "True"]
        assert len(na) == 40, f"{tbl}: {len(na)} not-attained rows, expected 40"
        states = {(r["family"], r["ext_target"]) for r in na}
        assert states == {("Direct", "-0.06"), ("Direct", "-0.03"),
                          ("Direct", "0.0"), ("Surrogate", "0.0")}, states
    ext = c.read_csv_literal(c.P0_TABLES / "matched_beta_ext_targets.csv")
    n_false = sum(1 for r in ext if r.get("attained") in ("False", "false"))
    assert n_false == 16, f"matched_beta_ext_targets: {n_false} unattained, want 16"
    hn = c.read_json(c.P1_PROV / "p1_headline_numbers.json")
    assert hn["display_set"]["not_attained"] == 4
    assert hn["display_set"]["attained"] == 44
    assert hn["display_set"]["entries"] == 48


def test_every_row_has_an_anchor_that_re_identifies_the_passage():
    bad = []
    for r in ROWS:
        line = int(r["baseline_line"])
        a = TEX.anchor_for_line(line)
        if a["baseline_excerpt_sha256"] != r["baseline_excerpt_sha256"]:
            bad.append(f"{r['claim_id']}: excerpt sha256 drifted at L{line}")
        if a["baseline_excerpt_norm"] != r["baseline_excerpt_norm"]:
            bad.append(f"{r['claim_id']}: excerpt text drifted at L{line}")
        if a["source_anchor"] != r["source_anchor"]:
            bad.append(f"{r['claim_id']}: source_anchor drifted at L{line}")
        if a["manuscript_section"] != r["manuscript_section"]:
            bad.append(f"{r['claim_id']}: section drifted at L{line}")
        if a["render_bucket"] != r["render_bucket"]:
            bad.append(f"{r['claim_id']}: bucket drifted at L{line}")
        if c.sha256_text(r["baseline_excerpt_norm"]) != r["baseline_excerpt_sha256"]:
            bad.append(f"{r['claim_id']}: excerpt sha256 does not match its text")
    assert not bad, "\n".join(bad[:20])


def test_every_cited_line_is_in_the_bucket_the_row_declares():
    bad = [f"{r['claim_id']}: declares {r['render_bucket']} at L{r['baseline_line']}"
           for r in ROWS
           if TEX.bucket_of_line(int(r["baseline_line"])) != r["render_bucket"]]
    assert not bad, "\n".join(bad[:20])


def test_b0_tex_reproduces_the_committed_archive_census():
    problems = b0_tex.self_check(verbose=False)
    assert not problems, "\n".join(problems)


def test_iffalse_regions_and_todo_count_are_as_recorded():
    assert TEX.iffalse_regions == list(c.IFFALSE_REGIONS)
    assert len(TEX.todos) == 19


def test_every_active_token_is_accounted_for():
    active = [t for t in TOKENS if t["render_bucket"] == "ACTIVE"]
    bad = [t for t in active if t["resolution"] not in
           ("SOURCED", "ALLOWLISTED", "FLAGGED_UNSUPPORTED")]
    assert not bad, f"unaccounted active tokens: {bad[:5]}"
    inert = [t for t in TOKENS if t["render_bucket"] != "ACTIVE"]
    assert all(t["resolution"] == "NOT_RENDERED" for t in inert)
    assert len(active) == SUMMARY["active_tokens"]


def test_allowlist_reasons_are_closed_and_carry_a_frozen_source_field():
    bad = []
    for a in ALLOW:
        if a["reason"] not in c.ALLOWLIST_REASONS:
            bad.append(f"{a['token']}: reason {a['reason']!r}")
        if a["guidance_status"] not in c.GUIDANCE_STATUS:
            bad.append(f"{a['token']}: guidance_status {a['guidance_status']!r}")
        if not a["frozen_source"]:
            bad.append(f"{a['token']}: no frozen_source")
        src = a["frozen_source"]
        if src != "NOT_RECORDED_IN_FROZEN_EVIDENCE" \
                and src not in MAN["artifacts"]:
            bad.append(f"{a['token']}: frozen_source not in manifest: {src}")
        if not a["note"]:
            bad.append(f"{a['token']}: no note")
    assert not bad, "\n".join(bad[:20])
    # IAAO constants must state which guidance they belong to
    iaao = [a for a in ALLOW if a["reason"] == "IAAO_STANDARD_CONSTANT"]
    assert iaao
    assert all(a["guidance_status"] in ("IAAO_2013_ADOPTED",
                                        "IAAO_2026_ED2_PROPOSED") for a in iaao)


def test_the_wildcard_allowlist_never_reaches_a_block_under_review():
    """An unsupported result must not be laundered by a wildcard constant."""
    no_wild = set(POLICY["no_wildcard_anchors"])
    assert no_wild, "the policy must name at least one block"
    exact = {(a["token"], a["source_anchor"]) for a in ALLOW
             if a["source_anchor"] != "*"}
    bad = []
    for t in TOKENS:
        if t["resolution"] != "ALLOWLISTED":
            continue
        if t["source_anchor"] in no_wild:
            key = (t["token"], t["source_anchor"])
            plain = (t["token"].replace(",", ""), t["source_anchor"])
            if key not in exact and plain not in exact:
                bad.append(f"L{t['baseline_line']} {t['token']!r} in "
                           f"{t['source_anchor']} allowlisted by a wildcard")
    assert not bad, "\n".join(bad[:20])
    assert SUMMARY["no_wildcard_anchors"] == sorted(no_wild)


def test_flagged_tokens_are_cross_listed_in_the_claim_map():
    """Every block that still has flagged numbers must carry a DELETE_OR_REPLACE
    or REWRITE_WITH_SUPPORTED_EVIDENCE claim. A flagged number with no claim
    would be an unsupported number nobody has been told to remove."""
    flagged_anchors = {t["source_anchor"] for t in TOKENS
                       if t["resolution"] == "FLAGGED_UNSUPPORTED"}
    remedial = {"DELETE_OR_REPLACE", "REWRITE_WITH_SUPPORTED_EVIDENCE"}
    covered = set()
    for cl in CLAIMS:
        if cl["disposition"] in remedial:
            covered.add(cl["source_anchor"])
            if cl["latex_label"]:
                covered.add(cl["latex_label"])
    missing = sorted(flagged_anchors - covered)
    assert not missing, f"flagged blocks with no remedial claim: {missing}"


def test_no_row_lacks_a_resolvable_source():
    bad = [r["claim_id"] for r in ROWS
           if r["attained_status"] == "UNRESOLVED"]
    assert not bad, f"UNRESOLVED rows must not exist in the map: {bad[:5]}"


def test_coverage_regenerates_byte_identically():
    before = {p: (c.B0 / p).read_bytes() for p in
              ("manuscript_numeric_map.csv",
               "coverage/active_tex_numeric_tokens.csv",
               "coverage/coverage_summary.json")}
    import b0_2_build_numeric_map
    with contextlib.redirect_stdout(io.StringIO()):
        assert b0_2_build_numeric_map.main() == 0
    for p, b in before.items():
        assert (c.B0 / p).read_bytes() == b, f"{p} is not byte-identical on rebuild"
