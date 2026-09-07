#!/usr/bin/env python3
"""Build FINAL_EVIDENCE_MANIFEST.json -- the hash-pinned index of every artifact
Tier B0 is allowed to cite.

Three classes of artifact appear:

  PRESENT       tracked under analysis/p0_*, analysis/p1_* or paper/. Its SHA256
                is RECOMPUTED here from the bytes on disk.
  ABSENT LEGACY the V6/V12 prediction trees and the CCAO parquet. These are
                gitignored by design and are in no checkout. Their hashes are
                QUOTED from a frozen index (PREFLIGHT.json / large_local_artifacts
                .json), never re-derived from a file -- and the index file's own
                SHA256 is recomputed first, so the quotation is verified
                transitively.
  EXCLUDED      external-benchmark streams. OUT_OF_SCOPE_FOR_B0: recorded as an
                exclusion with a reason, never as evidence.

Stage membership is taken from the frozen indices wherever they state it
(stage1_frozen_hashes.json, stage3b_artifact_hashes.json) and from an ordered
pattern table otherwise. A file matching no rule is a hard error: nothing is
classified silently.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import b0_common as c

MANIFEST = c.B0 / "FINAL_EVIDENCE_MANIFEST.json"

# --------------------------------------------------------------------------
# Ordered (regex, stage, role) rules, applied to the path relative to the
# stage root. First match wins.
# --------------------------------------------------------------------------
P0_RULES = [
    # --- non-table areas -------------------------------------------------
    (r"^APPROVED_EXECUTION_PLAN\.md$",        "P0_PROVENANCE", "protocol"),
    (r"^protocol_p0_validation\.yaml$",       "P0_PROVENANCE", "protocol"),
    (r"^README\.md$",                         "P0_PROVENANCE", "protocol"),
    (r"^code/",                               "P0_PROVENANCE", "code"),
    (r"^tests/",                              "P0_PROVENANCE", "test"),
    (r"^slurm/",                              "P0_PROVENANCE", "slurm_script"),
    (r"^logs/",                               "P0_PROVENANCE", "intermediate"),
    (r"^race/",                               "P0_STAGE_3B_G5B", "intermediate"),
    (r"^provenance/",                         "P0_PROVENANCE", "provenance"),
    (r"^configs/",                            "P0_CONVENTION", "convention_or_config"),
    (r"^figures/matched_beta_",               "P0_STAGE_3_MATCHED_BETA", "figure"),
    (r"^reports/ZERO_CONTROL_REPORT\.md$",    "P0_STAGE_1_5_G2", "report"),
    (r"^reports/RHO_ZERO_PARITY_REPORT\.md$", "P0_STAGE_1_5_G2", "report"),
    (r"^reports/P0_IMPLEMENTATION_AUDIT\.md$", "P0_STAGE_1", "report"),
    (r"^reports/CENTERED_SPREAD_COMPARATOR_REPORT\.md$", "P0_STAGE_2_G3", "report"),
    (r"^reports/MATCHED_BETA_REPORT\.md$",    "P0_STAGE_3_MATCHED_BETA", "report"),
    (r"^reports/TEMPORAL_ROBUSTNESS_REPORT\.md$", "P0_STAGE_3B_G5B", "report"),

    # --- tables: Stage-1 implementation audit ----------------------------
    (r"^tables/(objective_scaling_audit|direct_hessian_magnitudes|"
     r"effective_leaf_shrinkage|label_quantization|surrogate_weight_distribution|"
     r"boundary_exposure_audit|b_star_diagnostics|posthoc_development_roots|"
     r"fnum_same_host_replicate|repeat_pin_exposure|regeneration_triggers|"
     r"cv_fold_validation_overlap_audit|cv_fold_validation_overlap_summary|"
     r"development_beta_coordinate_audit|development_beta_coordinate_summary|"
     r"e4_verdict|stage1_summary)\.(csv|json)$", "P0_STAGE_1", "primary_evidence"),

    # --- tables: Stage-1.5 / G2 zero control and parity ------------------
    (r"^tables/zero_reference_fits_block\d+\.csv$", "P0_STAGE_1_5_G2", "intermediate"),
    (r"^tables/(zero_control_full|zero_control_frozen_crosscheck|"
     r"zero_reference_fits|zero_reference_reproduction_qc|parity_ladder_pinned|"
     r"parity_ladder_historical|bc_discriminate|bc_discriminate_verdict|"
     r"bc_rootcause_trace|bc_rootcause_verdict|frozen_artifact_reproduction|"
     r"gate_g2_checks|post_g1_adjudication_evidence)\.(csv|json)$",
     "P0_STAGE_1_5_G2", "primary_evidence"),
    (r"^tables/source_equivalence_", "P0_STAGE_1_5_G2", "primary_evidence"),

    # --- tables: Stage-2 / G3 centered-spread comparator -----------------
    (r"^tables/centered_spread_path__[ABC]__", "P0_STAGE_2_G3", "intermediate"),
    (r"^tables/(centered_spread_|gate_g3_checks)", "P0_STAGE_2_G3", "primary_evidence"),

    # --- tables: Stage-3 matched beta ------------------------------------
    (r"^tables/matched_beta_", "P0_STAGE_3_MATCHED_BETA", "primary_evidence"),

    # --- tables: Stage-3 / 3B temporal robustness ------------------------
    # the 45 dsnap refinement shards are consumed directly by Gate G5b
    (r"^tables/dsnap_refine_shard__", "P0_STAGE_3B_G5B", "primary_evidence"),
    (r"^tables/robustness_shard__", "P0_STAGE_3_TEMPORAL", "intermediate"),
    (r"^tables/(dsnap_refinement|dsnap_refinement_provenance|dsnap_boundary_audit|"
     r"dpurge_purge_audit|robustness_path_dsnap|robustness_path_dpurge|"
     r"robustness_unseen_subset|robustness_vs_frozen_deltas|"
     r"temporal_material_change_triggers|temporal_material_change_triggers_detail|"
     r"temporal_validation_overlap_audit|temporal_validation_overlap_summary|"
     r"gate_g5a_outcome|gate_g5b_outcome)\.(csv|json)$",
     "P0_STAGE_3_TEMPORAL", "primary_evidence"),
]

P1_RULES = [
    (r"^P1_CHECKPOINT\.md$",              "P1_PROVENANCE", "intermediate"),
    (r"^code/",                           "P1_PROVENANCE", "code"),
    (r"^tests/",                          "P1_PROVENANCE", "test"),
    (r"^configs/display_set_frozen",      "P1_DISPLAY_SET", "convention_or_config"),
    (r"^configs/ed2_vei_procedure\.json$", "P1_VEI_ED2", "convention_or_config"),
    (r"^configs/smearing_estimator_frozen", "P1_SMEARING", "convention_or_config"),
    (r"^provenance/",                     "P1_PROVENANCE", "provenance"),
    (r"^reports/",                        "P1_PROVENANCE", "report"),
    (r"^tables/dcor_estimator_",          "P1_DCOR", "primary_evidence"),
    (r"^tables/prb_inference",            "P1_PRB", "primary_evidence"),
    (r"^tables/vei_significance",         "P1_VEI_ED2", "primary_evidence"),
    (r"^tables/smearing_apply_(delta_nl_subset|fastpath_reconciliation)\.csv$",
     "P1_SMEARING", "intermediate"),
    (r"^tables/smearing_",                "P1_SMEARING", "primary_evidence"),
]

# Manuscript side: an EXPLICIT list, not the whole paper/ tree. Superseded
# manuscript versions, the committed PDFs and the 60 img/ figures are not
# evidence and are not indexed. (Figure files are checked for existence by the
# table/figure disposition audit, which reads \includegraphics paths out of the
# tex -- a coverage question, not an evidence one.)
PAPER_FILES = {
    "paper/paper_v17_option1.tex": ("TIER_A_MANUSCRIPT", "manuscript_source"),
    "paper/paper_v17_option2.tex": ("TIER_A_MANUSCRIPT", "wording_precedent_only"),
    "paper/references.bib": ("TIER_A_MANUSCRIPT", "bibliography"),
    "paper/references_additions.bib": ("TIER_A_MANUSCRIPT", "bibliography"),
    "paper/references_major_revision_additions.txt":
        ("TIER_A_MANUSCRIPT", "bibliography_staging_source"),
    "paper/paper_analysis/paper_v17/superseded_text_archive.md":
        ("TIER_A_MANUSCRIPT", "manuscript_analysis"),
    "paper/paper_analysis/paper_v17/markup_preflight.py":
        ("TIER_A_MANUSCRIPT", "manuscript_analysis"),
    "paper/paper_analysis/paper_v17/MAJOR_REVISION_CHANGELOG.md":
        ("TIER_A_MANUSCRIPT", "manuscript_analysis"),
    "paper/paper_analysis/paper_v17/P0_P1_MAJOR_REVISION_PLAN.md":
        ("TIER_A_MANUSCRIPT", "manuscript_analysis"),
    "paper/paper_analysis/paper_v17/analysis_v17.md":
        ("TIER_A_MANUSCRIPT", "manuscript_analysis"),
}

# Legacy inputs the manuscript's CURRENT path tables came from. Absent here by
# design; hashes quoted from the frozen index named in hash_source.
LEGACY_ABSENT_KEYS = (
    "archived_folds_json", "canonical_data_parquet", "delta_nl_estimator_json",
    "frozen_lgbm_config_json", "frozen_path_table_v4_csv",
    "v12_experiment_spec_json", "v6_experiment_spec_json",
    "v6_recalibration_path_csv", "v6_recalibration_spec_json",
    "v6_rho0_split_audit_csv", "v6_rho0_split_audit_json",
)
# OUT_OF_SCOPE_FOR_B0. Present in the worktree, deliberately NOT indexed.
EXCLUDED_PREFLIGHT_KEYS = ("external_zero_rho_parity_csv",)


def classify(relpath: str, rules) -> tuple:
    for pat, stage, role in rules:
        if re.search(pat, relpath):
            return stage, role
    raise SystemExit(
        f"UNCLASSIFIED artifact: {relpath!r}\n"
        "Every artifact must carry an explicit scientific_stage and role. "
        "Add a rule rather than letting it default.")


def git_tracked(prefix: str):
    out = subprocess.run(["git", "-C", str(c.REPO), "ls-files", prefix],
                         capture_output=True, text=True, check=True).stdout
    return [l for l in out.split("\n") if l.strip()]


def main() -> int:
    artifacts = {}
    counts = {"present": 0, "absent_legacy": 0}

    # ---- 1. frozen hash indices, verified by their own recomputed SHA256 ----
    index_files = {
        c.rel(c.P0_PROV / "PREFLIGHT.json"): "P0_PREFLIGHT_INPUT_INDEX",
        c.rel(c.P0_PROV / "large_local_artifacts.json"): "P0_LARGE_LOCAL_INDEX",
        c.rel(c.P0_PROV / "stage1_frozen_hashes.json"): "P0_STAGE1_INDEX",
        c.rel(c.P0_PROV / "stage3b_artifact_hashes.json"): "P0_STAGE3B_INDEX",
        c.rel(c.P0_PROV / "output_artifact_hashes.json"): "P0_OUTPUT_INDEX",
        c.rel(c.P1_PROV / "p1_artifact_hashes.json"): "P1_ARTIFACT_INDEX",
    }
    hash_indices = {p: {"role": r, "sha256": c.sha256_file(c.REPO / p)}
                    for p, r in index_files.items()}

    stage1_index = c.read_json(c.P0_PROV / "stage1_frozen_hashes.json")
    stage3b = c.read_json(c.P0_PROV / "stage3b_artifact_hashes.json")
    stage3b_paths = {f"analysis/p0_major_revision_validation/{k}"
                     for k in stage3b["artifacts"]}

    # ---- 2. present, tracked artifacts --------------------------------------
    for prefix, rules, tag in (
            ("analysis/p0_major_revision_validation", P0_RULES, c.P0_TAG),
            ("analysis/p1_inferential_reporting", P1_RULES, c.P1_TAG),
            ("paper", PAPER_FILES, c.TIER_A_COMMIT)):
        tracked = git_tracked(prefix)
        if isinstance(rules, dict):
            missing = sorted(set(rules) - set(tracked))
            if missing:
                raise SystemExit(f"expected manuscript-side files absent: {missing}")
            paths = sorted(rules)
        else:
            paths = tracked
        for path in paths:
            sub = path[len(prefix) + 1:]
            if isinstance(rules, dict):
                stage, role = rules[path]
            else:
                stage, role = classify(sub, rules)
            # frozen indices state stage membership authoritatively; prefer them
            if path in stage3b_paths:
                stage = "P0_STAGE_3B_G5B"
            elif path in stage1_index and stage == "P0_PROVENANCE" \
                    and role in ("code", "test", "slurm_script"):
                stage = "P0_STAGE_1"
            f = c.REPO / path
            artifacts[path] = {
                "scientific_stage": stage,
                "role": role,
                "present_in_worktree": True,
                "sha256": c.sha256_file(f),
                "bytes": f.stat().st_size,
                "hash_source": "RECOMPUTED",
                "source_commit_or_tag": tag,
            }
            counts["present"] += 1

    # ---- 3. absent legacy inputs, hashes quoted transitively ----------------
    pre = c.read_json(c.P0_PROV / "PREFLIGHT.json")
    inputs = pre["input_artifact_hashes"]
    for key in LEGACY_ABSENT_KEYS:
        rec = inputs[key]
        path = rec["path"]
        if (c.REPO / path).exists():
            raise SystemExit(
                f"{path} was expected ABSENT but exists; re-verify before quoting "
                "its hash from the frozen index.")
        artifacts[path] = {
            "scientific_stage": "LEGACY_FROZEN_PATH",
            "role": "legacy_frozen_input",
            "present_in_worktree": False,
            "sha256": rec["sha256"],
            "bytes": rec["bytes"],
            "hash_source": "P0_PREFLIGHT_INPUT_INDEX",
            "hash_source_file": c.rel(c.P0_PROV / "PREFLIGHT.json"),
            "hash_source_key": f"$.input_artifact_hashes.{key}.sha256",
            "source_commit_or_tag": c.P0_TAG,
            "note": "gitignored by design; in no checkout. Hash quoted, never "
                    "re-derived. Tier B0 reads this file never.",
        }
        counts["absent_legacy"] += 1

    excluded = [
        {"path": inputs[k]["path"], "preflight_key": k,
         "reason": "OUT_OF_SCOPE_FOR_B0",
         "detail": "external-jurisdiction benchmark. Recorded so the exclusion is "
                   "explicit; no external-benchmark artifact may back a manuscript "
                   "claim, and any active manuscript number depending on that "
                   "stream is FLAGGED_UNSUPPORTED rather than legitimized here."}
        for k in EXCLUDED_PREFLIGHT_KEYS]
    for d in c.FORBIDDEN_EVIDENCE_DIRS:
        excluded.append({"path": c.rel(d) + "/", "preflight_key": None,
                         "reason": "OUT_OF_SCOPE_FOR_B0",
                         "detail": "external benchmark stream; not indexed."})

    manifest = {
        "schema_version": 1,
        "generated_by": c.rel(Path(__file__)),
        "what": "Hash-pinned index of every artifact Tier B0 may cite as evidence.",
        "discipline": [
            "PRESENT artifacts carry a RECOMPUTED sha256.",
            "ABSENT legacy artifacts carry a hash QUOTED from the frozen index "
            "named in hash_source, whose own sha256 is recorded in hash_indices "
            "and recomputed by the test suite. Absent files are never read.",
            "External-benchmark streams are recorded under excluded_streams, "
            "never under artifacts.",
            "No Tier-B0 output claims CSV bit-exactness: P1 records that the "
            "committed CSVs round-trip float64 to ~3.55e-15, so values are "
            "carried as the literal decimal text found in the artifact.",
        ],
        "coordinates": {
            "integration_branch": "paper-major-revision-integration",
            "integration_head": c.INTEGRATION_HEAD,
            "p0_tag": c.P0_TAG, "p0_commit": c.P0_COMMIT,
            "p1_tag": c.P1_TAG, "p1_commit": c.P1_COMMIT,
            "tier_a_commit": c.TIER_A_COMMIT,
            "manuscript": c.rel(c.TEX), "manuscript_sha256": c.TEX_SHA256,
        },
        "hash_indices": hash_indices,
        "excluded_streams": excluded,
        "counts": {**counts, "total": len(artifacts),
                   "by_stage": _tally(artifacts, "scientific_stage"),
                   "by_role": _tally(artifacts, "role")},
        "artifacts": artifacts,
    }
    c.write_json(MANIFEST, manifest)
    print(f"wrote {c.rel(MANIFEST)}")
    print(f"  artifacts     {len(artifacts)} "
          f"({counts['present']} present, {counts['absent_legacy']} absent legacy)")
    print(f"  hash indices  {len(hash_indices)}")
    print(f"  excluded      {len(excluded)}")
    for k, v in sorted(manifest["counts"]["by_stage"].items()):
        print(f"    {k:28} {v}")
    return 0


def _tally(artifacts: dict, field: str) -> dict:
    out = {}
    for rec in artifacts.values():
        out[rec[field]] = out.get(rec[field], 0) + 1
    return out


if __name__ == "__main__":
    raise SystemExit(main())
