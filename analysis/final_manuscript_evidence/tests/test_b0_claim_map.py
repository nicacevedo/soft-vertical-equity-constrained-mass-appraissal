#!/usr/bin/env python3
"""Claim-map assertions, including the QUALIFY_AS_EXPLORATORY gate."""
from __future__ import annotations

import contextlib
import io
from collections import Counter

import b0_common as c
import b0_tex

CLAIMS = c.read_csv_literal(c.B0 / "manuscript_claim_map.csv")
NUM = {r["claim_id"]: r for r in c.read_csv_literal(c.B0 / "manuscript_numeric_map.csv")}
MAN = c.read_json(c.B0 / "FINAL_EVIDENCE_MANIFEST.json")
TODOS = c.read_yaml(c.SPEC / "todo_closure.yaml")
COV = c.read_yaml(c.SPEC / "mandatory_coverage.yaml")
TF = c.read_yaml(c.SPEC / "table_figure_disposition.yaml")
TEX = b0_tex.Tex()
INDEX_MD = c.read_text(c.B0 / "FINAL_EVIDENCE_INDEX.md")
SPEC_MD = c.read_text(c.B0 / "MANUSCRIPT_REVISION_SPEC.md")


def test_claim_ids_are_unique():
    dups = [k for k, n in Counter(r["claim_id"] for r in CLAIMS).items() if n > 1]
    assert not dups, dups


def test_disposition_vocabulary_is_closed_and_has_no_bare_qualify():
    bad = [r["claim_id"] for r in CLAIMS if r["disposition"] not in c.DISPOSITIONS]
    assert not bad, bad
    # the withdrawn generic values must not reappear
    assert "REWRITE" not in {r["disposition"] for r in CLAIMS}
    assert "QUALIFY" not in {r["disposition"] for r in CLAIMS}
    assert "DELETE" not in {r["disposition"] for r in CLAIMS}


def test_exploratory_is_gated_on_a_resolvable_frozen_number():
    """The binding rule: an unsupported numerical result is not publishable
    merely because it is relabelled exploratory."""
    bad = []
    for r in CLAIMS:
        if r["disposition"] != "QUALIFY_AS_EXPLORATORY":
            continue
        ids = [x for x in r["numeric_claim_ids"].split(";") if x]
        if not ids:
            bad.append(f"{r['claim_id']}: exploratory with no numeric_claim_ids")
            continue
        for i in ids:
            if i not in NUM:
                bad.append(f"{r['claim_id']}: numeric id {i} not in the map")
            elif NUM[i]["attained_status"] not in ("ATTAINED", "NOT_ATTAINED"):
                bad.append(f"{r['claim_id']}: {i} is "
                           f"{NUM[i]['attained_status']}, not resolvable")
    assert not bad, "\n".join(bad)
    assert any(r["disposition"] == "QUALIFY_AS_EXPLORATORY" for r in CLAIMS)


def test_the_unresolvable_blocks_are_delete_or_replace():
    """transition_summary, transition_regret and the candidate regions resolve
    from nothing, so they may not be qualified as exploratory."""
    want = {"tab:transition_summary", "tab:transition_regret",
            "tab:rho_candidate_regions"}
    seen = {}
    for r in CLAIMS:
        if r["latex_label"] in want:
            seen[r["latex_label"]] = r["disposition"]
    assert set(seen) == want, f"missing claims for {want - set(seen)}"
    bad = {k: v for k, v in seen.items() if v != "DELETE_OR_REPLACE"}
    assert not bad, f"must be DELETE_OR_REPLACE: {bad}"


def test_every_claim_has_reason_message_and_confidence():
    bad = []
    for r in CLAIMS:
        if not r["reason"]:
            bad.append(f"{r['claim_id']}: no reason")
        if not r["required_final_message"]:
            bad.append(f"{r['claim_id']}: no required_final_message")
        if r["confidence"] not in ("HIGH", "MEDIUM", "LOW"):
            bad.append(f"{r['claim_id']}: confidence {r['confidence']!r}")
        if r["priority"] not in ("P0", "P1", "optional cleanup"):
            bad.append(f"{r['claim_id']}: priority {r['priority']!r}")
    assert not bad, "\n".join(bad[:20])


def test_supporting_artifacts_are_in_the_manifest_or_explicitly_none():
    bad = []
    for r in CLAIMS:
        arts = [a for a in r["supporting_artifacts"].split(";") if a]
        if not arts:
            bad.append(f"{r['claim_id']}: no supporting_artifacts")
        for a in arts:
            if a == "NONE_REQUIRED":
                continue
            if a.startswith("analysis/final_manuscript_evidence/"):
                if not (c.REPO / a).exists():
                    bad.append(f"{r['claim_id']}: missing Tier-B0 artifact {a}")
            elif a not in MAN["artifacts"]:
                bad.append(f"{r['claim_id']}: {a} not in manifest")
    assert not bad, "\n".join(bad[:20])


def test_numeric_claim_ids_resolve():
    bad = [f"{r['claim_id']}: {i}" for r in CLAIMS
           for i in r["numeric_claim_ids"].split(";") if i and i not in NUM]
    assert not bad, "\n".join(bad[:20])


def test_all_nineteen_todo_sites_are_covered_exactly_once():
    ids = [r["todo_id"] for r in CLAIMS if r["todo_id"]]
    assert sorted(int(x) for x in ids) == list(range(1, 20)), sorted(ids)
    spec_ids = sorted(t["todo_id"] for t in TODOS["todos"])
    assert spec_ids == list(range(1, 20))
    assert sorted(t["baseline_line"] for t in TODOS["todos"]) == \
        sorted(t["line"] for t in TEX.todos)


def test_todo_closure_statuses_are_closed_and_buckets_match_the_tex():
    allowed = {"CLOSED_BY_FROZEN_EVIDENCE", "CLOSED_WITH_CONTRARY_FINDING",
               "PARTIALLY_CLOSED", "REMAINING_GAP_EDITORIAL",
               "REMAINING_GAP_RELEASE", "KEEP_AS_CONSTRAINT", "NOT_RENDERED"}
    by_line = {t["line"]: t["bucket"] for t in TEX.todos}
    bad = []
    for t in TODOS["todos"]:
        if t["closure_status"] not in allowed:
            bad.append(f"#{t['todo_id']}: {t['closure_status']}")
        if by_line[t["baseline_line"]] != t["render_bucket"]:
            bad.append(f"#{t['todo_id']}: bucket "
                       f"{t['render_bucket']} != {by_line[t['baseline_line']]}")
        for e in t.get("frozen_evidence", []) or []:
            if e.startswith("analysis/final_manuscript_evidence/"):
                assert (c.REPO / e).exists(), e
            elif e not in MAN["artifacts"]:
                bad.append(f"#{t['todo_id']}: evidence not in manifest: {e}")
    assert not bad, "\n".join(bad[:20])
    # a NOT_RENDERED status must correspond to a genuinely inert bucket
    for t in TODOS["todos"]:
        if t["closure_status"] == "NOT_RENDERED":
            assert t["render_bucket"] in ("COMMENTED", "IFFALSE"), t["todo_id"]


def test_the_parity_todo_is_closed_with_a_contrary_finding():
    """#16 promised the table could be regenerated if parity held. It does not."""
    t = next(x for x in TODOS["todos"] if x["todo_id"] == 16)
    assert t["closure_status"] == "CLOSED_WITH_CONTRARY_FINDING"
    conv = c.read_yaml(c.P0_CONFIGS / "post_g1_reference_convention.yaml")
    assert conv["verified_premises"]["p4_pinning_does_not_remove_BC"]["value"] is True
    assert conv["verified_premises"][
        "p5_colsample_change_is_a_different_specification"]["value"] is True
    # A<->B essentially agrees; B<->C does not
    rows = c.read_csv_literal(c.P0_TABLES / "parity_ladder_pinned.csv")
    ab = next(r for r in rows if r["cell_pair"] == "A<->B"
              and r["split"] == "heldout")
    bc = next(r for r in rows if r["cell_pair"] == "B<->C"
              and r["split"] == "heldout")
    from decimal import Decimal
    assert Decimal(ab["mean_abs_delta_log"]) < Decimal("1e-6")
    assert Decimal(bc["mean_abs_delta_log"]) > Decimal("1e-3")
    claim = next(r for r in CLAIMS if r["todo_id"] == "16")
    assert claim["disposition"] == "DELETE_OR_REPLACE"


def test_every_mandatory_coverage_topic_maps_and_appears_in_the_index():
    mapped = set()
    for r in CLAIMS:
        mapped.update(x for x in r["coverage_topics"].split(";") if x)
    bad = []
    for t in COV["topics"]:
        if t["topic_id"] not in mapped:
            bad.append(f"unmapped topic {t['topic_id']}")
        if t["topic_id"] not in INDEX_MD:
            bad.append(f"topic absent from FINAL_EVIDENCE_INDEX.md: {t['topic_id']}")
    assert not bad, "\n".join(bad)
    unknown = mapped - {t["topic_id"] for t in COV["topics"]}
    assert not unknown, f"claims cite unknown topics: {unknown}"


def test_counting_unit_is_present_for_every_aggregate_count_claim():
    """A claim that cites an aggregate over the display set or the evaluation
    structure must name its counting unit. The test keys on the counting unit
    the NUMBERS themselves carry, not on the raw unit `count`: 994 trees and an
    83-point grid are design constants, not aggregates over display entries."""
    bad = []
    for r in CLAIMS:
        ids = [i for i in r["numeric_claim_ids"].split(";") if i]
        units = {NUM[i]["counting_unit"] for i in ids if i in NUM}
        units.discard("NOT_APPLICABLE")
        if units and r["counting_unit"] == "NOT_APPLICABLE":
            bad.append(f"{r['claim_id']}: cites {sorted(units)} counts but "
                       "declares counting_unit NOT_APPLICABLE")
        if r["counting_unit"] != "NOT_APPLICABLE" and units \
                and r["counting_unit"] not in units:
            bad.append(f"{r['claim_id']}: declares {r['counting_unit']} but its "
                       f"numbers are {sorted(units)}")
    assert not bad, "\n".join(bad)


def test_the_two_ed2_count_families_are_never_mixed_in_one_claim():
    """396/279/228 are display_entry counts; 387/270/219 are unique_realization
    counts. They differ by exactly 9 and must never co-occur in a statement."""
    fw = c.read_yaml(c.SPEC / "forbidden_wording.yaml")
    fams = {s["counting_unit"]: set(s["values"])
            for s in fw["mutually_exclusive_count_sets"]}
    entry, uniq = fams["display_entry"], fams["unique_realization"]
    bad = []
    for r in CLAIMS:
        text = r["required_final_message"] + " " + r["claim_text"]
        import re
        toks = set(re.findall(r"\b\d{3}\b", text))
        if toks & entry and toks & uniq:
            bad.append(f"{r['claim_id']}: mixes {sorted(toks & entry)} with "
                       f"{sorted(toks & uniq)}")
    assert not bad, "\n".join(bad)
    # and the derived counts must actually be what the frozen table says
    rows = c.read_csv_literal(c.P1_TABLES / "vei_significance.csv")
    ed2 = [x for x in rows if x["attained"] == "True"
           and x["evaluation_role"] != "not_applicable"]
    assert len(ed2) == 396
    assert sum(1 for x in ed2
               if x["step5_gate"] == "step5_outside_pm10_escalate") == 279
    assert sum(1 for x in ed2 if x["step7_outcome"] == "reject_null") == 228
    seen = {(x["realization_key"], x["evaluation"]): x for x in ed2}
    assert len(seen) == 387
    assert sum(1 for x in seen.values()
               if x["step5_gate"] == "step5_outside_pm10_escalate") == 270
    assert sum(1 for x in seen.values()
               if x["step7_outcome"] == "reject_null") == 219
    # the 9-row gap is exactly one duplicated realization
    from collections import Counter as _C
    dup = {k: v for k, v in _C(x["realization_key"] for x in ed2).items() if v > 9}
    assert dup == {"fit:LGBCovPenalty:1fb838f7d6bfda88": 18}, dup


def test_anchors_still_identify_their_passages():
    bad = []
    for r in CLAIMS:
        a = TEX.anchor_for_line(int(r["baseline_line"]))
        if a["baseline_excerpt_sha256"] != r["baseline_excerpt_sha256"]:
            bad.append(f"{r['claim_id']}: excerpt drifted")
        if a["source_anchor"] != r["source_anchor"]:
            bad.append(f"{r['claim_id']}: source_anchor drifted")
        if a["render_bucket"] != r["render_bucket"]:
            bad.append(f"{r['claim_id']}: bucket drifted")
    assert not bad, "\n".join(bad[:20])


def test_declared_latex_labels_exist_in_the_tex():
    labels = {l for _p, _ln, l in TEX.labels}
    bad = [f"{r['claim_id']}: {r['latex_label']}" for r in CLAIMS
           if r["latex_label"] and r["latex_label"] not in labels]
    assert not bad, "\n".join(bad)


def test_every_active_float_has_a_disposition():
    spec_keys = {(f["label"], f["render_bucket"]) for f in TF["floats"]}
    measured = set()
    for f in TEX.floats():
        for lab in (f["labels"] or ["(no label)"]):
            measured.add((lab, f["bucket"]))
    assert measured == spec_keys, (
        f"missing {sorted(measured - spec_keys)}; "
        f"stale {sorted(spec_keys - measured)}")
    bad = [f["label"] for f in TF["floats"]
           if f["disposition"] not in c.TF_DISPOSITIONS
           or f["numbers_fully_supported"] not in c.TF_SUPPORT]
    assert not bad, bad


def test_the_table_highlighting_convention_is_specified():
    """Feedback item 2 requires an explicit recommendation, not a hint."""
    assert "highlighting and labelling convention" in SPEC_MD
    for label in ("tab:path_anchor_summary", "tab:path_anchor_complementary",
                  "tab:ccao_baseline_results", "tab:ccao_baseline_complementary"):
        assert label in SPEC_MD, label
    assert "two visibly different devices" in SPEC_MD
    assert "workflow-benchmark" in SPEC_MD and "penalty-isolating" in SPEC_MD


def test_claim_map_regenerates_byte_identically():
    before = (c.B0 / "manuscript_claim_map.csv").read_bytes()
    import b0_3_build_claim_map
    with contextlib.redirect_stdout(io.StringIO()):
        assert b0_3_build_claim_map.main() == 0
    assert (c.B0 / "manuscript_claim_map.csv").read_bytes() == before
