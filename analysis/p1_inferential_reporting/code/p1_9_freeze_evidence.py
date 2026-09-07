#!/usr/bin/env python3
"""P1 final evidence freeze.

Two jobs, in this order:

  1. verify every frozen P0 scientific artifact is byte-identical to the tag;
  2. regenerate the authoritative P1 artifact-hash manifest.

The manifest is generated LAST, after every other P1 file is final, because it
hashes them.  It cannot contain its own hash, so it excludes exactly one path --
itself -- and records that exclusion explicitly.  (At the previous checkpoint the
manifest also went stale against P1_CHECKPOINT.md, which was edited afterwards to
point at the manifest; running this script last is what prevents a recurrence.)

Writes nothing outside analysis/p1_inferential_reporting/.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p1_common as c                                                   # noqa: E402

SELF_EXCLUDED = "provenance/p1_artifact_hashes.json"
P0_PROV = c.P0_DIR / "provenance"
P0_INDEXES = ("stage1_frozen_hashes.json", "output_artifact_hashes.json")
PROTECTED = ["paper/", "utils/", "soft_constrained_models/", "scripts/", "run_temporal_cv.py",
             "output/paper_v6_preselection_994/", "output/paper_v12_lower_rho_extension_994_v2/",
             "analysis/p0_major_revision_validation/"]


def _git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=str(c.REPO), capture_output=True,
                          text=True, check=True).stdout.strip()


def verify_p0() -> dict:
    """Byte-exactness of the frozen P0 scientific artifacts."""
    out: dict = {"tag": c.P0_TAG, "commit": c.P0_COMMIT}

    # (a) git: nothing under the P0 tree differs from the tag
    out["git_diff_vs_tag_p0_tree"] = _git(
        "diff", "--name-only", c.P0_TAG, "--", "analysis/p0_major_revision_validation/")
    out["p0_tree_identical_to_tag"] = (out["git_diff_vs_tag_p0_tree"] == "")

    # (b) protected paths clean in the working tree
    out["protected_paths_dirty"] = _git("status", "--porcelain", "--", *PROTECTED)
    out["protected_paths_clean"] = (out["protected_paths_dirty"] == "")

    # (c) every file changed since the tag, and the P1-only claim
    changed = [x for x in _git("diff", "--name-only", c.P0_TAG, "HEAD").splitlines() if x]
    outside = [x for x in changed
               if not x.startswith("analysis/p1_inferential_reporting/") and x != ".gitignore"]
    out["files_changed_since_tag"] = len(changed)
    out["files_changed_outside_p1_and_gitignore"] = outside
    out["only_p1_and_gitignore_changed"] = (outside == [])

    # (d) the frozen hash indexes, recomputed file by file.
    #
    #     Two of P0's three indexes are EARLY-STAGE snapshots that P0's own later gates
    #     legitimately superseded: the G2/G3 test modules were edited in those later
    #     stages, POSTFLIGHT/slurm_graph/large_local_artifacts are end-of-run summaries,
    #     and output_artifact_hashes.json LISTS ITSELF -- a self-reference that can never
    #     match. So "zero mismatches" is the wrong criterion; it was never true, not even
    #     at the tag.
    #
    #     The right criterion is that P1 introduced NO NEW mismatch. Each index is checked
    #     against the working tree AND against the tag's own blobs (via `git show`), and
    #     the two mismatch sets must be identical. That detects any P1-induced change while
    #     tolerating P0's documented internal staleness, and it needs no extra checkout.
    def _tag_blob_sha(rel: str):
        r = subprocess.run(["git", "show", f"{c.P0_TAG}:{rel}"], cwd=str(c.REPO),
                           capture_output=True)
        if r.returncode != 0:
            return None
        import hashlib
        return hashlib.sha256(r.stdout).hexdigest()

    idx: dict = {}
    for name in P0_INDEXES:
        H = json.loads((P0_PROV / name).read_text())
        ok = miss = 0
        bad_worktree, bad_at_tag = [], []
        for rel, meta in H.items():
            p = c.REPO / rel
            if not p.exists():
                miss += 1
                continue
            if c.sha256_file(p) == meta["sha256"] and p.stat().st_size == meta["bytes"]:
                ok += 1
            else:
                bad_worktree.append(rel)
            if _tag_blob_sha(rel) != meta["sha256"]:
                bad_at_tag.append(rel)
        idx[name] = {
            "n": len(H), "byte_exact": ok, "missing": miss,
            "MISMATCH": len(bad_worktree),
            "mismatched_paths": sorted(bad_worktree),
            "mismatched_paths_AT_TAG": sorted(bad_at_tag),
            "no_new_mismatch_vs_tag": sorted(bad_worktree) == sorted(bad_at_tag),
            "stale_by_p0_design": sorted(bad_at_tag),
            "why_stale": ("early-stage snapshot superseded by P0's own later gates; "
                          "output_artifact_hashes.json additionally lists itself"),
        }
    S3 = json.loads((P0_PROV / "stage3b_artifact_hashes.json").read_text())
    ok = bad = miss = 0
    bad_list = []
    for rel, meta in S3["artifacts"].items():
        p = c.P0_DIR / rel
        if not p.exists():
            miss += 1
            continue
        if c.sha256_file(p) == meta["sha256"] and p.stat().st_size == meta["bytes"]:
            ok += 1
        else:
            bad += 1
            bad_list.append(rel)
    idx["stage3b_artifact_hashes.json"] = {
        "n": len(S3["artifacts"]), "byte_exact": ok, "missing": miss, "MISMATCH": bad,
        "mismatched_paths": bad_list, "gate": S3.get("gate"),
        "no_new_mismatch_vs_tag": bad == 0,
        "note": "the FINAL P0 gate index; this one IS fully byte-exact"}
    out["frozen_hash_indexes"] = idx
    out["final_gate_index_byte_exact"] = bool(
        idx["stage3b_artifact_hashes.json"]["MISMATCH"] == 0
        and idx["stage3b_artifact_hashes.json"]["missing"] == 0)
    out["no_new_mismatch_in_any_index"] = all(v["no_new_mismatch_vs_tag"] for v in idx.values())
    out["no_index_entry_is_missing"] = all(v["missing"] == 0 for v in idx.values())
    out["immutability_criterion"] = (
        "P0 is unchanged iff: (1) git diff vs the tag over the P0 tree is empty; "
        "(2) the FINAL gate index (Stage-3B / G5b) is fully byte-exact; (3) no index "
        "acquired a mismatch that the tag did not already have; (4) no index entry went "
        "missing; (5) protected paths are clean; (6) the only files changed since the tag "
        "are the P1 directory and .gitignore. Requiring zero mismatches in the two "
        "early-stage indexes would be wrong: it was never true, not even at the tag.")

    out["P0_UNCHANGED"] = bool(out["p0_tree_identical_to_tag"]
                               and out["protected_paths_clean"]
                               and out["only_p1_and_gitignore_changed"]
                               and out["final_gate_index_byte_exact"]
                               and out["no_new_mismatch_in_any_index"]
                               and out["no_index_entry_is_missing"])
    return out


def build_manifest(p0: dict, p0_suite: dict) -> dict:
    tracked = [x for x in _git("ls-files", "analysis/p1_inferential_reporting/").splitlines() if x]
    arts = {}
    for rel in sorted(tracked):
        sub = rel.split("analysis/p1_inferential_reporting/", 1)[1]
        if sub == SELF_EXCLUDED:
            continue
        p = c.REPO / rel
        arts[sub] = {"bytes": p.stat().st_size, "sha256": c.sha256_file(p)}

    MAN = c.ed2_source_manifest()
    pdf = c.PROVENANCE / "ed2_source_cache" / Path(MAN["source"]["pdf_url"]).name
    return {
        "schema_version": 2,
        "stage": "P1_INFERENTIAL_REPORTING",
        "status": "FINAL",
        "generated_by": "code/p1_9_freeze_evidence.py",
        "note": ("SHA-256 index of every TRACKED P1 artifact at the final evidence freeze. "
                 "Parquet twins and logs are gitignored by design (mirroring P0). The IAAO ED2 "
                 "PDF is cached at a gitignored path and is NOT redistributed; its hash is "
                 "recorded separately below."),
        "self_excluded": SELF_EXCLUDED,
        "self_exclusion_reason": ("a hash index cannot contain its own hash; this file is the only "
                                  "tracked P1 path it does not cover, and it is generated after "
                                  "every file it does cover is final"),
        "n_artifacts": len(arts),
        "artifacts": arts,
        "p0_tag": c.P0_TAG,
        "p0_commit": c.P0_COMMIT,
        "p0_immutability": p0,
        "p0_suite": p0_suite,
        "ed2_source_not_committed": {
            "url": MAN["source"]["pdf_url"],
            "sha256": MAN["source"]["sha256"],
            "bytes": MAN["source"]["bytes"],
            "committed": False,
            "redistribution_permission_established": False,
            "path_gitignored": str(pdf.relative_to(c.REPO)),
            "present_on_run_host": pdf.exists(),
            "verified_on_run_host": bool(pdf.exists()
                                         and c.sha256_file(pdf) == MAN["source"]["sha256"]),
        },
        "provenance": c.preflight_block(),
    }


def main() -> int:
    print("=" * 78)
    print("P1 FINAL EVIDENCE FREEZE")
    print("=" * 78)
    p0 = verify_p0()
    print(f"\n[P0] tree identical to tag      : {p0['p0_tree_identical_to_tag']}")
    print(f"[P0] protected paths clean      : {p0['protected_paths_clean']}")
    print(f"[P0] only P1 + .gitignore added : {p0['only_p1_and_gitignore_changed']} "
          f"({p0['files_changed_since_tag']} files since tag)")
    for k, v in p0["frozen_hash_indexes"].items():
        print(f"[P0] {k:34s} {v['byte_exact']}/{v['n']} byte-exact "
              f"(missing {v['missing']}, MISMATCH {v['MISMATCH']}, "
              f"same-as-tag {v['no_new_mismatch_vs_tag']})")
    print(f"[P0] final gate index byte-exact: {p0['final_gate_index_byte_exact']}")
    print(f"[P0] no NEW mismatch vs tag     : {p0['no_new_mismatch_in_any_index']}")
    print(f"[P0] ==> P0_UNCHANGED           : {p0['P0_UNCHANGED']}")
    if not p0["P0_UNCHANGED"]:
        raise c.ProtocolViolation(f"P0 is NOT byte-identical: {json.dumps(p0, indent=2)}")

    suite_path = c.PROVENANCE / "p0_suite_at_tag.json"
    p0_suite = json.loads(suite_path.read_text()) if suite_path.exists() else {
        "status": "NOT_RECORDED — run the P0 suite in an isolated checkout at the tag"}

    man = build_manifest(p0, p0_suite)
    out = c.write_json(c.PROVENANCE / "p1_artifact_hashes.json", man)
    print(f"\n[manifest] {man['n_artifacts']} tracked artifacts hashed -> {out.name}")
    print(f"[manifest] self-excluded: {SELF_EXCLUDED}")
    print(f"[manifest] ED2 PDF verified on run host: "
          f"{man['ed2_source_not_committed']['verified_on_run_host']} (NOT redistributed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
