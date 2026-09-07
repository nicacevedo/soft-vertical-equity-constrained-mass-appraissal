#!/usr/bin/env python3
"""Scientific-integrity guards, data-driven from spec/forbidden_wording.yaml.

This module holds no copy of the forbidden strings or the reference-role
semantics: it reads them from the spec, which in turn quotes the frozen
convention files. So a change to the science has one place to happen.

Every Tier-B0 text output is scanned. A forbidden phrase may appear only inside
a sentence that states the prohibition, marked by one of the spec's
`prohibition_markers`.
"""
from __future__ import annotations

import re
from decimal import Decimal
from pathlib import Path

import b0_common as c

FW = c.read_yaml(c.SPEC / "forbidden_wording.yaml")
CONV = c.read_yaml(c.P0_CONFIGS / "post_g1_reference_convention.yaml")
POSTHOC = c.read_yaml(c.P0_CONFIGS / "posthoc_comparator_convention.yaml")
NUM = c.read_csv_literal(c.B0 / "manuscript_numeric_map.csv")
CLAIMS = c.read_csv_literal(c.B0 / "manuscript_claim_map.csv")

TEXT_OUTPUTS = sorted(
    p for p in c.B0.rglob("*")
    if p.is_file() and p.suffix in (".md", ".yaml", ".csv", ".bib")
    and "metadata_cache" not in p.parts)

# The DECLARATION SITES: places whose job is to quote a prohibited string.
# spec/forbidden_wording.yaml is the register itself; the `forbidden_wording`
# column of the maps lists what a claim must avoid; and the generated documents
# reproduce that register. Pattern scans skip these, and
# test_forbidden_register_states_its_own_prohibitions checks them instead.
DECLARATION_FILES = {"forbidden_wording.yaml"}
_DECL_LINE = re.compile(
    "^\\s*(-\\s*phrase:"                       # a YAML register entry
    "|\\|\\s*[\u201c\u201d\"']"             # a table cell that opens with a quote
    "|-\\s*\\*?\\*?(wording that )?must NOT" # a bullet listing prohibitions
    ")", re.I)


def _scannable(path: Path) -> str:
    """The text of `path` with declaration contexts removed."""
    if path.name in DECLARATION_FILES:
        return ""
    text = c.read_text(path)
    if path.suffix == ".csv":
        # drop the forbidden_wording column, which exists to list prohibitions
        rows = c.read_csv_literal(path)
        if rows and "forbidden_wording" in rows[0]:
            return "\n".join(
                " ".join(v for k, v in r.items() if k != "forbidden_wording")
                for r in rows)
        return text
    if path.suffix == ".md":
        out, skip = [], False
        for ln in text.split("\n"):
            if ln.startswith("## ") and "Forbidden wording" in ln:
                skip = True
                continue
            if skip and ln.startswith("## "):
                skip = False
            if skip or _DECL_LINE.match(ln):
                continue
            out.append(ln)
        return "\n".join(out)
    if path.suffix == ".yaml":
        # Walk the parsed structure rather than filtering lines: YAML block
        # scalars wrap sentences across lines, and a line filter would both miss
        # register entries and shred sentences into fragments.
        skip_keys = {"forbidden_wording", "forbidden_phrases",
                     "forbidden_literals", "phrase", "prohibition_markers",
                     "mutually_exclusive_count_sets", "forbidden_b_max_rule"}
        out = []

        def walk(node, key=None):
            if key in skip_keys:
                return
            if isinstance(node, dict):
                for k, v in node.items():
                    walk(v, k)
            elif isinstance(node, list):
                for v in node:
                    walk(v, key)
            elif isinstance(node, str):
                out.append(node)

        walk(c.read_yaml(path))
        return "\n".join(out)
    return text


def _sentences(text: str):
    return re.split(r"(?<=[.!?])\s+|\n", text)


def test_reference_roles_match_the_frozen_convention():
    """A is the workflow benchmark; C is the penalty-isolating reference. The
    spec must quote the frozen convention, not paraphrase it."""
    for cid in ("A", "B", "C"):
        spec = FW["reference_roles"][cid]
        frozen = CONV["cells"][cid]
        assert spec["forward_display_name"] == frozen["forward_display_name"], cid
        assert spec["role"] == frozen["role"], cid
        assert spec["primary_penalty_reference"] == \
            frozen["primary_penalty_reference"], cid
    assert CONV["cells"]["C"]["primary_penalty_reference"] is True
    assert CONV["cells"]["A"]["primary_penalty_reference"] is False
    assert CONV["comparison_kinds"]["penalty_isolating"]["reference"] == "C"
    assert CONV["comparison_kinds"]["penalty_isolating"]["may_attribute_to_rho"] is True
    assert CONV["comparison_kinds"]["assessor_facing_benchmark"]["reference"] == "A"
    assert CONV["comparison_kinds"][
        "assessor_facing_benchmark"]["may_attribute_to_rho"] is False


def test_a_and_c_are_never_swapped_in_either_map():
    """Two-sided: A may not be penalty-isolating, and C may not be a benchmark.
    Feedback item 2: these are two frozen roles, not a substitution."""
    bad = []
    for label, rows in (("numeric", NUM), ("claim", CLAIMS)):
        for r in rows:
            rc, cp = r["reference_cell"], r["comparison_purpose"]
            if rc == "A" and cp == "PENALTY_ISOLATING":
                bad.append(f"{label} {r['claim_id']}: A as PENALTY_ISOLATING")
            if rc == "C" and cp == "WORKFLOW_BENCHMARK":
                bad.append(f"{label} {r['claim_id']}: C as WORKFLOW_BENCHMARK")
    assert not bad, "\n".join(bad)
    # both roles must actually be exercised, or the guard proves nothing
    assert any(r["reference_cell"] == "A"
               and r["comparison_purpose"] == "WORKFLOW_BENCHMARK" for r in NUM)
    assert any(r["reference_cell"] == "C"
               and r["comparison_purpose"] == "PENALTY_ISOLATING" for r in NUM)


def test_a_referenced_claims_are_never_described_as_penalty_effects():
    pat = re.compile(r"(pure |isolat|attributab|effect of (the )?(rho|penalty)"
                     r"|penalty effect)", re.I)
    # A sentence that introduces the C-referenced comparison legitimately uses
    # penalty-isolating language; it is not asserting that an A-referenced
    # difference is a penalty effect.
    about_c = re.compile(r"(C-referenced|Cell C|custom-objective rho=0 origin"
                         r"|penalty-isolating reference)", re.I)
    bad = []
    for r in CLAIMS:
        if r["reference_cell"] != "A":
            continue
        for s in _sentences(r["required_final_message"]):
            if not pat.search(s):
                continue
            if about_c.search(s):
                continue
            if re.search(r"\bnot\b|never|rather than|descriptiv", s, re.I):
                continue
            bad.append(f"{r['claim_id']}: {s.strip()[:120]}")
    assert not bad, "\n".join(bad)
    # prove the guard can still fire
    assert pat.search("this is a pure penalty effect")
    assert not about_c.search("this is a pure penalty effect")


def test_forward_display_names_are_used_and_legacy_names_are_confined():
    """Legacy Stage-1 strings may appear only as table values, never as the
    manuscript-facing display name."""
    legacy = {FW["reference_roles"]["B"]["legacy_stage1_label"],
              FW["reference_roles"]["C"]["legacy_stage1_label"]}
    spec_md = c.read_text(c.B0 / "MANUSCRIPT_REVISION_SPEC.md")
    for cid in ("A", "B", "C"):
        assert FW["reference_roles"][cid]["forward_display_name"] in spec_md, cid
    # each legacy string that appears must be flagged as legacy where it appears
    for name in legacy:
        for s in _sentences(spec_md):
            if name in s:
                assert re.search(r"[Ll]egacy|Stage-1", s), \
                    f"legacy label {name!r} used without a legacy marker: {s[:120]}"


def test_d1_is_primary_and_d3_is_sensitivity():
    ph = POSTHOC["calibration"]
    assert "coordinate_D1_primary" in ph
    assert "coordinate_D2_sensitivity" in ph
    d3 = c.read_json(c.P0_TABLES / "matched_beta_d3_sensitivity_summary.json")
    assert d3, "the D3 summary must exist and be non-empty"
    bad = [r["claim_id"] for r in CLAIMS
           if "D3" in r["required_final_message"]
           and "SENSITIVITY" not in r["required_final_message"]
           and "sensitivity" not in r["required_final_message"]]
    assert not bad, f"D3 mentioned without marking it sensitivity: {bad}"


def test_c_posthoc_is_primary_and_a_posthoc_secondary():
    refs = POSTHOC["references"]
    assert refs["C"]["posthoc_status"] == "PRIMARY"
    assert refs["A"]["posthoc_status"] == "SECONDARY_PRACTICAL"
    assert refs["B"]["posthoc_status"] == "NO_FULL_PATH"


def test_no_forbidden_phrase_appears_outside_a_prohibition():
    markers = [m.lower() for m in FW["prohibition_markers"]]
    bad = []
    for p in TEXT_OUTPUTS:
        text = _scannable(p)
        for entry in FW["forbidden_phrases"]:
            phrase = entry["phrase"]
            if phrase not in text:
                continue
            for s in _sentences(text):
                if phrase in s and not any(m in s.lower() for m in markers):
                    bad.append(f"{c.rel(p)}: “{phrase}” without a prohibition "
                               f"marker in: {s.strip()[:140]}")
    assert not bad, "\n".join(bad[:15])


def test_the_unrendered_fstring_literals_are_never_quoted():
    bad = []
    for p in TEXT_OUTPUTS:
        text = _scannable(p)
        for lit in FW["forbidden_literals"]:
            if lit in text and "forbidden_literals" not in text[:400]:
                # allowed only in the spec that defines them, and in the doc that
                # states the prohibition alongside the resolved values
                if p.name in ("forbidden_wording.yaml",
                              "MANUSCRIPT_REVISION_SPEC.md"):
                    continue
                bad.append(f"{c.rel(p)}: quotes {lit!r}")
    assert not bad, "\n".join(bad)
    # and the report they come from really does contain them, unrendered
    rpt = c.read_text(c.P0_REPORTS / "CENTERED_SPREAD_COMPARATOR_REPORT.md")
    assert any(lit in rpt for lit in FW["forbidden_literals"])


def test_e4_is_indeterminate_and_gradient_only_is_only_prohibited():
    v = c.read_json(c.P0_TABLES / "e4_verdict.json")
    assert v["e4_verdict"] == "INDETERMINATE"
    ev = v["evidence_at_display_anchors"]
    rule = v["rule"]
    assert Decimal(str(ev["max_curvature_ratio"])) > Decimal(str(rule["R_accept_min_at_rho_ge_1"]))
    assert Decimal(str(ev["max_penalty_contribution_M_over_all_blocks_and_anchors"])) \
        > Decimal(str(rule["M_accept_max"]))
    row = next(r for r in NUM if r["claim_id"] == "N-e4-verdict")
    assert row["display_value"] == "INDETERMINATE"
    # the phrase appears only inside prohibitions (covered by the phrase guard),
    # and the claim that used to assert it is DELETE_OR_REPLACE
    cl = next(r for r in CLAIMS if r["claim_id"] == "C-theory-004")
    assert cl["disposition"] == "DELETE_OR_REPLACE"
    assert "Direct is effectively gradient-only" in cl["forbidden_wording"]


def test_no_fold_sd_is_labelled_a_standard_error():
    pat = re.compile(r"(SD\s*/\s*(sqrt|√)|standard error of the (folds|fold))", re.I)
    bad = []
    for p in TEXT_OUTPUTS:
        for s in _sentences(_scannable(p)):
            if pat.search(s) and not re.search(r"not|never|must NOT|forbidden",
                                               s, re.I):
                bad.append(f"{c.rel(p)}: {s.strip()[:140]}")
    assert not bad, "\n".join(bad)


def test_ed2_is_never_described_as_adopted():
    """Every ED2 mention must carry an exposure-draft / proposed qualifier, and
    IAAO 2013 must be the only *_ADOPTED guidance status in use."""
    rows = c.read_csv_literal(c.P1_TABLES / "vei_significance.csv")
    gs = {r["guidance_status"] for r in rows}
    assert gs == {"May-2026 Exposure Draft / proposed guidance; "
                  "not adopted IAAO guidance."}, gs
    adopted = {r["adopted_reference"] for r in rows}
    assert adopted == {"IAAO Standard on Ratio Studies (2013) remains the "
                       "adopted/current guidance."}, adopted
    used = {r["guidance_status"] for r in NUM} | {r["guidance_status"] for r in CLAIMS}
    assert used <= set(c.GUIDANCE_STATUS), used
    assert "IAAO_2013_ADOPTED" in used and "IAAO_2026_ED2_PROPOSED" in used
    # VEI must never be tagged adopted, and PRB never exposure-draft
    bad = [r["claim_id"] for r in NUM
           if r["metric"] == "VEI" and r["guidance_status"] == "IAAO_2013_ADOPTED"]
    bad += [r["claim_id"] for r in NUM
            if r["metric"] == "PRB"
            and r["guidance_status"] == "IAAO_2026_ED2_PROPOSED"]
    assert not bad, bad


def test_pooled_oof_carries_no_ed2_inference():
    """Encoded as actually frozen: vei_significance flags pooled_oof through
    evaluation_role == not_applicable, while the literal
    NOT_APPLICABLE_FOR_ED2_INFERENCE lives in the smearing stability table."""
    vei = c.read_csv_literal(c.P1_TABLES / "vei_significance.csv")
    na = [r for r in vei if r["evaluation_role"] == "not_applicable"]
    assert len(na) == 48, len(na)
    assert {r["evaluation"] for r in na} == {"pooled_oof"}
    stab = c.read_csv_literal(c.P1_TABLES / "smearing_apply_ed2_stability.csv")
    lit = [r for r in stab
           if r.get("ed2_applicability") == "NOT_APPLICABLE_FOR_ED2_INFERENCE"]
    assert len(lit) == 43, len(lit)
    hn = c.read_json(c.P1_PROV / "p1_headline_numbers.json")
    assert hn["task3_vei"]["pooled_oof_treatment"] == \
        "NOT_APPLICABLE_FOR_ED2_INFERENCE"
    assert hn["task4_smearing"]["ed2_cells_not_applicable"] == 43


def test_prb_uses_the_entire_ci_rule():
    hn = c.read_json(c.P1_PROV / "p1_headline_numbers.json")
    rule = hn["task2_prb"]["classification_rule"]
    assert "ENTIRE" in rule and "outside" in rule
    assert hn["task2_prb"]["ci_level"] == 0.95
    counts = hn["task2_prb"]["class_counts_all"]
    assert sum(counts.values()) == hn["task2_prb"]["rows"] == 480
    assert counts["NOT_ATTAINED"] == 40


def test_surrogate_wording_is_associational_not_causal():
    assert "causal" in CONV["cells"]["C"]["terminology"]["avoid"]
    pat = re.compile(r"\bcaus(e|es|ed|al|ally)\b", re.I)
    bad = []
    for r in CLAIMS:
        for s in _sentences(r["required_final_message"]):
            if pat.search(s) and not re.search(r"not|never|avoid", s, re.I):
                bad.append(f"{r['claim_id']}: {s.strip()[:140]}")
    assert not bad, "\n".join(bad)


def test_zero_covariance_is_never_equated_with_fairness():
    pat = re.compile(r"zero covariance[^.]{0,80}"
                     r"(implies|means|ensures|guarantees)[^.]{0,40}"
                     r"(fair|unbiased|independen)", re.I)
    bad = [c.rel(p) for p in TEXT_OUTPUTS if pat.search(_scannable(p))]
    assert not bad, bad
    cl = next(r for r in CLAIMS if r["claim_id"] == "C-theory-001")
    assert "zero covariance implies fairness" in cl["forbidden_wording"]


def test_the_post_hoc_root_is_b_star_train_not_one_over_r2():
    b = CONV["b_star_definition"]
    assert b["definition"].startswith("b_star_train = Var_T(y) / Cov_T(f0, y)")
    assert "NOT the LightGBM b-star definition" in b["one_over_r2"]
    assert POSTHOC["calibration"]["one_over_r2"]["used_for_endpoint"] is False
    assert POSTHOC["calibration"]["b_star_train"][
        "used_for_empirical_endpoint"] is False
    assert POSTHOC["calibration"]["forbidden_b_max_rule"] == "b_max = 1.25 * b_star"
    assert POSTHOC["calibration"]["b_max_rule"].startswith("b_max = 1 + 1.25 *")
    phrases = [x["phrase"] for x in FW["forbidden_phrases"]]
    assert POSTHOC["calibration"]["forbidden_b_max_rule"] in phrases


def test_no_deployment_or_penalty_selection_framing_survives():
    banned = ("sweet spot", "safe region", "deployment point", "recommended rho")
    markers = [m.lower() for m in FW["prohibition_markers"]]
    bad = []
    for p in TEXT_OUTPUTS:
        for s in _sentences(_scannable(p)):
            for b in banned:
                if b in s.lower() and not any(m in s.lower() for m in markers):
                    bad.append(f"{c.rel(p)}: “{b}” in {s.strip()[:120]}")
    assert not bad, "\n".join(bad[:10])
    # candidate-region claims must be remedial, never KEEP
    bad = [r["claim_id"] for r in CLAIMS
           if "candidate" in (r["claim_text"] + r["latex_label"]).lower()
           and r["disposition"] in ("KEEP", "QUALIFY_AS_EXPLORATORY")]
    assert not bad, f"candidate-region claims must not be kept as-is: {bad}"


def test_the_duan_sign_and_smearing_invariance_set_are_correct():
    cfg = c.read_json(c.P1_CONFIGS / "smearing_estimator_frozen.json")
    blob = str(cfg)
    assert "y_true_log - y_pred_log" in blob or "y_true - y_pred" in blob, blob[:400]
    facts = c.read_json(c.P1_TABLES / "dcor_estimator_facts.json")
    assert facts["residual_convention"] == "e = y_pred_log - y_true_log"
    # the two conventions have OPPOSITE signs and must not be conflated
    inv = c.read_json(c.P1_TABLES / "smearing_apply_summary.json")
    assert "RMSE_log" not in str(inv.get("invariant_metrics", []))
    hn = c.read_json(c.P1_PROV / "p1_headline_numbers.json")
    assert "RMSE_log" not in hn["task4_smearing"]["per_metric_max_rel_diff"]
    # smearing claims are sensitivity only
    bad = [r["claim_id"] for r in CLAIMS
           if "smear" in r["claim_text"].lower()
           and r["primary_or_sensitivity"] not in ("SENSITIVITY", "NOT_APPLICABLE")]
    assert not bad, bad
    bad = [r["claim_id"] for r in NUM
           if r["domain"] == "smearing" and r["primary_or_sensitivity"] != "SENSITIVITY"]
    assert not bad, bad


def test_dcor_facts_match_the_frozen_artifact_field_by_field():
    facts = c.read_json(c.P1_TABLES / "dcor_estimator_facts.json")
    want = {"N-dcor-estimator-class": ("estimator_class", None),
            "N-dcor-bias-corrected": ("bias_corrected_passed", "false"),
            "N-dcor-version": ("version_installed", None),
            "N-dcor-function": ("function_called", None),
            "N-dcor-residual-convention": ("residual_convention", None)}
    for cid, (field, expect) in want.items():
        row = next(r for r in NUM if r["claim_id"] == cid)
        val = facts[field]
        txt = ("true" if val is True else "false" if val is False else str(val))
        assert row["raw_value"] == txt, f"{cid}: {row['raw_value']!r} != {txt!r}"
        if expect is not None:
            assert txt == expect, f"{cid}: {txt!r} != {expect!r}"
    assert facts["estimator_class"].startswith("BIASED / V-statistic")
    assert facts["bias_corrected_passed"] is False
    assert facts["version_installed"] == "0.6"


def test_no_tier_b0_output_claims_csv_bit_exactness():
    pat = re.compile(r"bit[- ]exact", re.I)
    bad = []
    for p in TEXT_OUTPUTS:
        for s in _sentences(_scannable(p)):
            if pat.search(s):
                # allowed only where it is denied, or where it describes the
                # frozen Stage-3B gate index, which genuinely is byte-exact
                if re.search(r"\bno\b|never|not\b|28/28|final-gate|final_gate",
                             s, re.I):
                    continue
                bad.append(f"{c.rel(p)}: {s.strip()[:140]}")
    assert not bad, "\n".join(bad)
    man = c.read_json(c.B0 / "FINAL_EVIDENCE_MANIFEST.json")
    assert any("bit-exactness" in d for d in man["discipline"])


def test_option2_precedent_carries_no_number_or_conclusion():
    bad = []
    for r in CLAIMS:
        prec = r["option2_precedent"]
        if not prec:
            continue
        if re.search(r"\d+\.\d+|\b\d{3,}\b", prec):
            bad.append(f"{r['claim_id']}: numeric value in option2_precedent: "
                       f"{prec[:100]}")
        if not re.search(r"WORDING", prec, re.I):
            bad.append(f"{r['claim_id']}: option2_precedent must say WORDING "
                       f"ONLY: {prec[:100]}")
    assert not bad, "\n".join(bad)
    # no numeric-map row may be sourced from Option 2
    bad = [r["claim_id"] for r in NUM if "option2" in r["artifact_path"]]
    assert not bad, bad
    assert "wording may be reused" in FW["option2_rule"] or \
        "WORDING PRECEDENT ONLY" in FW["option2_rule"]


def test_no_external_benchmark_path_backs_any_row():
    forbidden = [c.rel(d) for d in c.FORBIDDEN_EVIDENCE_DIRS]
    bad = [r["claim_id"] for r in NUM
           if any(r["artifact_path"].startswith(f) for f in forbidden)]
    bad += [r["claim_id"] for r in CLAIMS
            if any(f in r["supporting_artifacts"] for f in forbidden)]
    assert not bad, bad


def test_every_staged_bib_field_is_traceable_or_flagged():
    bib = c.read_text(c.BIB / "staged_historical_entries.bib")
    prov = c.read_text(c.BIB / "STAGED_BIB_PROVENANCE.md")
    keys = re.findall(r"@\w+\{([^,]+),", bib)
    assert sorted(keys) == ["Cheng1974", "Edelstein1979", "PaglinFogarty1972",
                            "SundermanEtAl1990"], keys
    for k in keys:
        assert f"`{k}`" in prov, k
    # every UNVERIFIED_FLAGGED marker in the .bib must be explained in the
    # provenance document
    flagged = re.findall(r"% UNVERIFIED_FLAGGED: (\w+)", bib)
    assert flagged, "expected at least one flagged field"
    assert "UNVERIFIED_FLAGGED" in prov
    # no DOI may be invented: Cheng1974 has no authoritative record, so no doi
    cheng = re.search(r"@article\{Cheng1974,(.*?)\n\}", bib, re.S).group(1)
    assert "doi" not in cheng.replace("% UNVERIFIED_FLAGGED: doi", "")
    assert "no DOI is supplied" in prov or "none may be invented" in prov


def test_the_novelty_boundary_claims_are_all_present():
    labels = {r["claim_id"] for r in CLAIMS}
    for cid in ("C-intro-002", "C-related-001", "C-discussion-004"):
        assert cid in labels, cid
    nov = [r for r in CLAIMS if "T-novelty-boundary" in r["coverage_topics"]]
    assert len(nov) >= 3
    assert all(r["disposition"] in ("KEEP", "REWRITE_WITH_SUPPORTED_EVIDENCE")
               for r in nov)


def test_the_d3_and_appearances_cautions_are_recorded():
    phrases = [p["phrase"] for p in FW["forbidden_phrases"]]
    assert "project D3 is ED2 Appendix D.3" in phrases
    assert "multiplicity 2 gives 41,976 rows" in phrases
    hn = c.read_json(c.P1_PROV / "p1_headline_numbers.json")
    d3 = hn["task4_smearing"]["d3"]
    assert d3["duplicated_unique_rows"] == 20988
    assert d3["max_multiplicity"] == 2
    assert 20988 * 2 == 41976


def test_forbidden_register_states_its_own_prohibitions():
    """The register is excluded from pattern scanning, so it is checked here:
    every entry must carry a reason, a frozen source, and a prohibition cue."""
    markers = [m.lower() for m in FW["prohibition_markers"]]
    bad = []
    for e in FW["forbidden_phrases"]:
        # A `why` need not contain a prohibition cue: its POSITION in
        # forbidden_phrases is the prohibition, and requiring a cue word inside
        # an explanation would be a proxy for something already structural.
        # What it must be is substantive and sourced to a frozen artifact.
        if not e.get("why") or len(e["why"].split()) < 8:
            bad.append(f"{e['phrase']}: why is missing or too thin")
            continue
        src = e.get("source", "")
        if not src:
            bad.append(f"{e['phrase']}: no source")
        elif not (c.REPO / src).exists():
            bad.append(f"{e['phrase']}: source does not exist: {src}")
    assert not bad, "\n".join(bad)
    assert len(FW["forbidden_phrases"]) >= 15
    assert any(m in FW["forbidden_literals_why"].lower() for m in markers)
    assert any(m in FW["option2_rule"].lower() for m in markers)
    # the register must be reachable from the writing-pass document, or the
    # writer never sees it
    spec_md = c.read_text(c.B0 / "MANUSCRIPT_REVISION_SPEC.md")
    assert "spec/forbidden_wording.yaml" in spec_md
    for e in FW["forbidden_phrases"]:
        assert e["phrase"] in spec_md, f"{e['phrase']!r} absent from the spec doc"


def test_declaration_sites_are_actually_excluded_from_scanning():
    """Guard the guard: if _scannable stopped excluding the register, the phrase
    scan would trivially pass on everything, so prove the exclusion is live."""
    reg = c.SPEC / "forbidden_wording.yaml"
    assert _scannable(reg) == ""
    spec_md = c.B0 / "MANUSCRIPT_REVISION_SPEC.md"
    raw, scanned = c.read_text(spec_md), _scannable(spec_md)
    assert len(scanned) < len(raw)
    # ... and prove the scan still SEES the rest of that document
    assert "highlighting and labelling convention" in scanned
