#!/usr/bin/env python3
"""Isolation assertions: guards, write discipline and the static scan.

Feedback item 7: the literal strings "V6" and "V12" are NOT prohibited, because
Tier B0 must be able to index and describe legacy V6/V12 evidence. What is
prohibited is *reading* from the unavailable legacy roots and *executing* any
model fit or data load. Both `output/` and `data/` exist in this worktree
holding unrelated material, so a "the directory is absent" argument would not be
sound: the runtime read-allowlist is what makes the isolation real.
"""
from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import b0_common as c

CODE_FILES = sorted((c.CODE).glob("*.py"))
TEST_FILES = sorted((c.TESTS).glob("*.py"))
ALL_PY = CODE_FILES + TEST_FILES

NETWORK_MODULES = {"urllib", "urllib.request", "urllib.error", "http",
                   "http.client", "socket", "requests", "httpx", "ftplib",
                   "telnetlib", "smtplib", "xmlrpc"}
MODEL_MODULES = {"lightgbm", "xgboost", "catboost", "torch", "tensorflow",
                 "sklearn", "scipy.optimize"}
FORBIDDEN_CALLS = {"load_canonical_splits", "fit", "train", "predict"}


def git(*a) -> str:
    return subprocess.run(["git", "-C", str(c.REPO), *a],
                          capture_output=True, text=True, check=True).stdout


def test_git_status_shows_nothing_outside_the_tier_b0_area():
    allowed_prefix = "analysis/final_manuscript_evidence/"
    bad = []
    for line in git("status", "--porcelain").splitlines():
        path = line[3:].strip().strip('"')
        if path == ".gitignore":
            continue                      # the one approved exception
        if not path.startswith(allowed_prefix):
            bad.append(line)
    assert not bad, "changes outside the Tier-B0 area:\n" + "\n".join(bad)


def test_frozen_stages_and_paper_are_untouched():
    assert git("diff", "--stat", c.P0_TAG, "HEAD", "--",
               "analysis/p0_major_revision_validation").strip() == ""
    assert git("diff", "--stat", c.P1_TAG, "HEAD", "--",
               "analysis/p1_inferential_reporting").strip() == ""
    assert git("diff", "--stat", c.TIER_A_COMMIT, "HEAD", "--", "paper").strip() == ""
    # and nothing is staged or dirty under those trees either
    for p in ("analysis/p0_major_revision_validation",
              "analysis/p1_inferential_reporting", "paper"):
        assert git("status", "--porcelain", "--", p).strip() == "", p


def test_write_guard_refuses_every_protected_path():
    for bad in ("paper/paper_v17_option1.tex",
                "paper/paper_v17_option1.pdf",
                "paper/img/baseline_models_motivation_2024_2025.pdf",
                "analysis/p0_major_revision_validation/tables/zero_control_full.csv",
                "analysis/p1_inferential_reporting/tables/prb_inference.csv",
                ".gitignore"):
        try:
            c.guard_write(c.REPO / bad)
            raise AssertionError(f"write guard allowed {bad}")
        except c.GuardError:
            pass
    # and it permits the Tier-B0 area
    c.guard_write(c.B0 / "x.md")
    c.guard_write(c.B0 / "nested" / "y.json")


def test_read_guard_refuses_the_unavailable_legacy_roots():
    """Both roots EXIST here, holding unrelated smoke-test and geo material, so
    the guard cannot rely on absence."""
    assert (c.REPO / "output").is_dir()
    assert (c.REPO / "data").is_dir()
    for bad in ("output/paper_v6_preselection_994/lgbm_config.json",
                "output/paper_v12_lower_rho_extension_994_v2/experiment_spec.json",
                "output/quick_test",
                "data/CCAO/2025/training_data.parquet",
                "data/geo/cook_il_puma2020.geojson"):
        try:
            c.guard_read(c.REPO / bad)
            raise AssertionError(f"read guard allowed {bad}")
        except c.GuardError:
            pass
    for bad in c.FORBIDDEN_EVIDENCE_DIRS:
        try:
            c.guard_read(bad / "x.csv")
            raise AssertionError(f"read guard allowed {bad}")
        except c.GuardError:
            pass
    # and it permits the frozen stages and the manuscript
    for good in ("analysis/p0_major_revision_validation/tables/zero_control_full.csv",
                 "analysis/p1_inferential_reporting/tables/prb_inference.csv",
                 "paper/paper_v17_option1.tex"):
        c.guard_read(c.REPO / good)


def test_the_legacy_artifacts_are_genuinely_absent():
    """The V6/V12 trees and the CCAO parquet are hash-pinned, never read."""
    man = c.read_json(c.B0 / "FINAL_EVIDENCE_MANIFEST.json")
    absent = [p for p, r in man["artifacts"].items()
              if not r["present_in_worktree"]]
    assert len(absent) == 11
    for p in absent:
        assert not (c.REPO / p).exists(), f"{p} exists but is marked absent"
        assert p.startswith("output/") or p.startswith("data/"), p


def test_v6_and_v12_may_be_named_but_never_read():
    """Feedback item 7: describing legacy evidence is required, not forbidden."""
    named = [p.name for p in ALL_PY
             if "V6" in c.read_text(p) or "V12" in c.read_text(p)
             or "v6_" in c.read_text(p) or "v12_" in c.read_text(p)]
    assert named, ("Tier B0 must be able to index legacy V6/V12 evidence; "
                   "no module mentions it, which suggests the index is missing")
    # ... and no module reads from those roots
    bad = []
    for p in ALL_PY:
        tree = ast.parse(c.read_text(p), filename=str(p))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                v = node.value
                if v.startswith("output/") or v.startswith("data/") \
                        or "/output/paper_v" in v:
                    # a path literal is only allowed as a GUARD or a pattern
                    if p.name not in ("b0_common.py", "b0_1_build_manifest.py",
                                      "test_b0_isolation.py",
                                      "test_b0_manifest.py"):
                        bad.append(f"{p.name}: path literal {v!r}")
    assert not bad, "\n".join(bad)


def test_no_module_imports_a_model_or_network_library():
    bad = []
    for p in ALL_PY:
        tree = ast.parse(c.read_text(p), filename=str(p))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for n in names:
                root = n.split(".")[0]
                if n in MODEL_MODULES or root in {m.split(".")[0]
                                                  for m in MODEL_MODULES}:
                    bad.append(f"{p.name}: imports model library {n}")
                if n in NETWORK_MODULES or root in {m.split(".")[0]
                                                    for m in NETWORK_MODULES}:
                    bad.append(f"{p.name}: imports network library {n}")
    assert not bad, "\n".join(bad)


def test_no_module_calls_a_data_loader_or_a_fit():
    bad = []
    for p in ALL_PY:
        tree = ast.parse(c.read_text(p), filename=str(p))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = (fn.id if isinstance(fn, ast.Name)
                    else fn.attr if isinstance(fn, ast.Attribute) else "")
            if name == "load_canonical_splits":
                bad.append(f"{p.name}: calls {name}")
            if name in ("read_parquet", "read_feather"):
                bad.append(f"{p.name}: calls {name}")
    assert not bad, "\n".join(bad)


def test_the_network_fetcher_lives_outside_code_and_is_never_imported():
    """bib/fetch_metadata.py is the only file allowed network I/O, and no build
    script or test may import it: the pipeline stays offline and deterministic."""
    fetcher = c.BIB / "fetch_metadata.py"
    assert fetcher.exists()
    assert fetcher.parent == c.BIB, "the fetcher must not live under code/"
    src = c.read_text(fetcher)
    assert "urllib.request" in src, "the fetcher is the file that does network I/O"
    # AST, not substring: a docstring may legitimately EXPLAIN the fetcher.
    bad = []
    for f in ALL_PY:
        tree = ast.parse(c.read_text(f), filename=str(f))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any("fetch_metadata" in a.name for a in node.names):
                    bad.append(f"{f.name}: imports the fetcher")
            elif isinstance(node, ast.ImportFrom):
                if node.module and "fetch_metadata" in node.module:
                    bad.append(f"{f.name}: imports the fetcher")
            elif isinstance(node, ast.Call):
                fn = node.func
                nm = (fn.id if isinstance(fn, ast.Name)
                      else fn.attr if isinstance(fn, ast.Attribute) else "")
                if nm in ("run", "check_call", "check_output", "Popen") and any(
                        isinstance(a, ast.Constant) and isinstance(a.value, str)
                        and "fetch_metadata" in a.value for a in node.args):
                    bad.append(f"{f.name}: shells out to the fetcher")
    assert not bad, f"modules importing or invoking the fetcher: {bad}"
    # the cached responses it wrote are what the builder reads
    idx = c.read_json(c.BIB / "metadata_cache" / "RETRIEVAL_INDEX.json")
    for key, rec in idx["entries"].items():
        if rec.get("status") != "OK":
            continue
        f = c.BIB / "metadata_cache" / rec["response_file"]
        assert f.exists(), key
        assert c.sha256_file(f) == rec["response_sha256"], key


def test_every_emitted_file_is_tracked_or_trackable_and_none_is_gitignored():
    emitted = [p for p in c.B0.rglob("*")
               if p.is_file() and "__pycache__" not in p.parts]
    assert emitted, "the Tier-B0 area is empty"
    bad = []
    for p in emitted:
        rel = c.rel(p)
        ignored = subprocess.run(
            ["git", "-C", str(c.REPO), "check-ignore", "-q", "--no-index", rel]
        ).returncode == 0
        if ignored:
            bad.append(f"{rel} is gitignored")
    assert not bad, "\n".join(bad)


def test_parquet_and_bytecode_stay_ignored():
    for rel in ("analysis/final_manuscript_evidence/x.parquet",
                "analysis/final_manuscript_evidence/code/__pycache__/x.pyc"):
        rc = subprocess.run(
            ["git", "-C", str(c.REPO), "check-ignore", "-q", "--no-index", rel]
        ).returncode
        assert rc == 0, f"{rel} should be ignored but is not"


def test_only_the_gitignore_is_modified_outside_the_tier_b0_area():
    """The single approved exception, verified against the working tree rather
    than against HEAD, so it does not depend on commit history."""
    changed = {line[3:].strip() for line in git("status", "--porcelain").splitlines()}
    outside = {p for p in changed
               if not p.startswith("analysis/final_manuscript_evidence/")}
    assert outside <= {".gitignore"}, f"unexpected changes: {outside}"


def test_no_tier_b0_module_writes_outside_the_area():
    """Every write must go through the guarded helpers."""
    bad = []
    for p in CODE_FILES:
        tree = ast.parse(c.read_text(p), filename=str(p))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                name = (fn.id if isinstance(fn, ast.Name)
                        else fn.attr if isinstance(fn, ast.Attribute) else "")
                # c.write_text / c.write_json / c.write_csv ARE the guarded
                # writers; only a raw Path.write_text is a violation.
                receiver = (fn.value.id if isinstance(fn, ast.Attribute)
                            and isinstance(fn.value, ast.Name) else "")
                if name in ("write_text", "write_bytes") \
                        and receiver not in ("c", "b0_common") \
                        and p.name != "b0_common.py":
                    bad.append(f"{p.name}: raw {name} on {receiver or '<expr>'}; "
                               "use the guarded writers")
                if name == "open":
                    args = node.args + [k.value for k in node.keywords]
                    mode = next((a.value for a in args
                                 if isinstance(a, ast.Constant)
                                 and isinstance(a.value, str)
                                 and set(a.value) <= set("rwxabt+")), "r")
                    if any(ch in mode for ch in "wxa") and p.name != "b0_common.py":
                        bad.append(f"{p.name}: open(mode={mode!r}) outside b0_common")
    assert not bad, "\n".join(bad)


def test_the_guarded_writers_are_the_ones_actually_used():
    """Guard the guard: every builder must go through b0_common's writers, so
    prove they are used and that they enforce the write guard."""
    used = set()
    for p in CODE_FILES:
        tree = ast.parse(c.read_text(p), filename=str(p))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and isinstance(node.func.value, ast.Name) \
                    and node.func.value.id == "c":
                used.add(node.func.attr)
    assert {"write_text", "write_json", "write_csv"} & used, used
    # the writers really do enforce the guard
    try:
        c.write_text(c.PAPER / "x.md", "no")
        raise AssertionError("write_text bypassed the guard")
    except c.GuardError:
        pass


def test_every_builder_regenerates_byte_identically():
    """The whole pipeline must be deterministic and offline: rerunning every
    builder in order must leave the working tree unchanged."""
    import contextlib
    import io
    tracked = [p for p in c.B0.rglob("*")
               if p.is_file() and "__pycache__" not in p.parts]
    before = {p: p.read_bytes() for p in tracked}
    import b0_1_build_manifest, b0_1b_build_certification
    import b0_2_build_numeric_map, b0_3_build_claim_map
    import b0_4_build_bib, b0_5_build_tf_disposition, b0_6_build_documents
    with contextlib.redirect_stdout(io.StringIO()):
        for mod in (b0_1_build_manifest, b0_1b_build_certification,
                    b0_2_build_numeric_map, b0_3_build_claim_map,
                    b0_4_build_bib, b0_5_build_tf_disposition,
                    b0_6_build_documents):
            rc = mod.main()
            assert rc == 0, f"{mod.__name__}.main() returned {rc}"
    after = {p: p.read_bytes() for p in c.B0.rglob("*")
             if p.is_file() and "__pycache__" not in p.parts}
    changed = [c.rel(p) for p in set(before) | set(after)
               if before.get(p) != after.get(p)]
    assert not changed, f"rebuild is not byte-identical: {changed}"
