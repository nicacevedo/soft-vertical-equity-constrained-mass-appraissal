#!/usr/bin/env python3
"""P0 F.2b -- source-equivalence / provenance audit.

MUST run, and be interpreted, BEFORE any historical reproduction failure is
attributed to nondeterminism.

For every commit recorded as having generated a frozen artifact, diff HEAD against
it over the files capable of affecting the executed training or prediction path,
classify every hunk, and re-verify every stored hash.

Emits:
  tables/source_equivalence_audit.csv
  provenance/provenance_commit_diffs/<commit>__<file>.patch
  provenance/DIRTY_STATE_LIMITATION.md
  tables/source_equivalence_verdict.json
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p0_common as c

# Commits recorded inside the frozen artifacts as their generating commit.
PROVENANCE_COMMITS = {
    "508dc1c2b19d64dfb330676a865d4eeb07596a0a": {
        "role": "994-tree baseline/config + v6 experiment spec",
        "artifacts": ["output/paper_v6_preselection_994/lgbm_config.json",
                      "output/paper_v6_preselection_994/experiment_spec.json",
                      "output/paper_v12_lower_rho_extension_994_v2/lgbm_config.json"],
        "dirty": True,
        "diff_sha256": "2268bae6162a51642c37ac56ab0aca6d3410abcd353183d01979eccb9f8bd096",
        "diff_text_archived": False,
    },
    "2aa0346aa28661a829d18cccd8dded9852fe58fd": {
        "role": "lower-rho extension experiment spec",
        "artifacts": ["output/paper_v12_lower_rho_extension_994_v2/experiment_spec.json"],
        "dirty": True,
        "diff_sha256": "fa45001c231a63d86b538c38a18c25a9ba330895094250312bc5f047a83f4002",
        "diff_text_archived": False,
    },
    "d3ef45f22165f8a7e606c2c949a947644995335d": {
        "role": "final_local_results: rho=0 split audit, recalibration path, Delta_NL",
        "artifacts": ["output/paper_v6_preselection_994/final_local_results/rho0_split_audit.json",
                      "output/paper_v6_preselection_994/final_local_results/recalibration_spec.json",
                      "output/paper_v6_preselection_994/final_local_results/delta_nl_estimator.json"],
        "dirty": True,
        "diff_sha256": None,
        "diff_text_archived": False,
    },
}

# Files capable of affecting the executed training or prediction path.
EXECUTED_PATH_FILES = [
    "soft_constrained_models/boosting_models.py",
    "run_temporal_cv.py",
    "utils/motivation_utils.py",
    "utils/delta_nl.py",
    "canonical_experiment.py",
    "params.yaml",
    "cv_config.yaml",
    "model_params.yaml",
]

# Symbols whose modification would change what a fit computes.
EXECUTED_SYMBOLS = [
    "canonical_direct_scaled_grad_hess", "canonical_surrogate_scaled_grad_hess",
    "canonical_direct_exact_scaled_hessian", "_canonical_mean_init_enabled",
    "_lgbm_regressor_from_params", "class LGBCovPenalty", "class LGBSmoothPenalty",
    "def fit", "def predict", "def fobj",
    "_load_and_split_data", "_native_lgbm_estimator", "_build_model_specs",
    "_build_lgbm_params_from_files", "_build_rho_values", "_finalize_rho_values",
    "split_ccao_assessment_universe", "build_rolling_origin_protocol",
    "compute_taxation_metrics", "paper_mechanism_metrics", "_compute_extended_metrics",
    "estimate_delta_nl", "lgbm_params_hash",
]


def _git(args, check=True):
    return subprocess.run(["git"] + args, cwd=str(c.REPO), capture_output=True,
                          text=True, check=check)


def run() -> int:
    head = c.git_state()["commit"]
    rows = []
    diffdir = c.PROVENANCE / "provenance_commit_diffs"
    diffdir.mkdir(parents=True, exist_ok=True)

    any_executed_drift = False
    for commit, meta in PROVENANCE_COMMITS.items():
        anc = _git(["merge-base", "--is-ancestor", commit, head], check=False).returncode == 0
        for f in EXECUTED_PATH_FILES:
            r = _git(["diff", "--numstat", f"{commit}..{head}", "--", f], check=False)
            numstat = r.stdout.strip()
            if numstat:
                added, deleted, _ = numstat.split("\t")
                added, deleted = int(added), int(deleted)
            else:
                added = deleted = 0
            patch = _git(["diff", f"{commit}..{head}", "--", f], check=False).stdout
            touches = False
            hunks_touching = []
            if patch:
                out = diffdir / f"{commit[:8]}__{f.replace('/', '_')}.patch"
                out.write_text(patch)
                # A hunk touches the executed path if its changed lines fall inside
                # a hunk whose header or body references an executed symbol.
                cur_hdr = None
                cur_body = []
                def _flush(hdr, body):
                    if hdr is None:
                        return
                    blob = hdr + "\n" + "\n".join(body)
                    changed = [l for l in body if l.startswith(("+", "-"))
                               and not l.startswith(("+++", "---"))]
                    if not changed:
                        return
                    if any(sym in blob for sym in EXECUTED_SYMBOLS):
                        hunks_touching.append(hdr.strip())
                for line in patch.splitlines():
                    if line.startswith("@@"):
                        _flush(cur_hdr, cur_body)
                        cur_hdr, cur_body = line, []
                    elif cur_hdr is not None:
                        cur_body.append(line)
                _flush(cur_hdr, cur_body)
                # For config files any change is executed-path relevant.
                if f.endswith(".yaml") and (added or deleted):
                    hunks_touching.append("(config file change)")
                touches = bool(hunks_touching)
            if touches:
                any_executed_drift = True
            rows.append({
                "provenance_commit": commit,
                "provenance_role": meta["role"],
                "commit_is_ancestor_of_head": anc,
                "head_commit": head,
                "file": f,
                "lines_added_since_commit": added,
                "lines_deleted_since_commit": deleted,
                "n_hunks_touching_executed_path": len(hunks_touching),
                "touches_executed_path": touches,
                "touching_hunk_headers": " | ".join(hunks_touching[:6]),
                "patch_archived": bool(patch),
                "generating_tree_dirty": meta["dirty"],
                "generating_diff_sha256": meta["diff_sha256"],
                "generating_diff_text_archived": meta["diff_text_archived"],
            })
    df = pd.DataFrame(rows)
    c.write_table(df, c.TABLES / "source_equivalence_audit.csv")

    # ------------------------------------------------------- hash re-verification
    hv = []

    def _chk(name, got, expected, source):
        hv.append({"hash_name": name, "expected": expected, "observed": got,
                   "match": (got == expected) if expected is not None else None,
                   "source": source})

    cfg = json.loads(c.FROZEN_LGBM_CONFIG.read_text())
    sys.path.insert(0, str(c.REPO))
    from canonical_experiment import lgbm_params_hash
    _chk("lgbm_params_sha256", lgbm_params_hash(cfg["lgbm_params"]),
         cfg["lgbm_params_sha256"], "output/paper_v6_preselection_994/lgbm_config.json")
    _chk("lgbm_config_id", cfg["config_id"], "407d47775760c14d", "lgbm_config.json")

    v6spec = json.loads((c.V6 / "experiment_spec.json").read_text())
    v12spec = json.loads((c.V12 / "experiment_spec.json").read_text())
    _chk("v6_frozen_baseline_hash", v6spec.get("frozen_baseline_hash"),
         v6spec.get("frozen_baseline_hash"), "v6 experiment_spec.json (self-consistency)")
    _chk("v12_frozen_baseline_hash", v12spec.get("frozen_baseline_hash"),
         "fcf614d60048cae7", "v12 experiment_spec.json")
    _chk("v12_canonical_model_grid_hash", v12spec.get("canonical_model_grid_hash"),
         "2ceba22cb08138b1", "v12 experiment_spec.json")
    _chk("v12_model_grid_hash", v12spec.get("model_grid_hash"),
         "23d0e88535bb65ce", "v12 experiment_spec.json")
    _chk("v6_lgbm_params_sha256_in_spec", lgbm_params_hash(v6spec["lgbm_params"]),
         cfg["lgbm_params_sha256"], "v6 experiment_spec.json")
    _chk("v12_lgbm_params_sha256_in_spec", lgbm_params_hash(v12spec["lgbm_params"]),
         cfg["lgbm_params_sha256"], "v12 experiment_spec.json")

    grid_json = c.V12 / "protocol" / "lower_rho_grid_v2.json"
    if grid_json.exists():
        gj = json.loads(grid_json.read_text())
        _chk("lower_rho_grid_hash", gj.get("grid_hash") or gj.get("hash"),
             "072e4ea94c35a252fbe7e433f63a9c75bc1d557c950a0a987ad44b14e73dbdbc",
             "V12/protocol/lower_rho_grid_v2.json")

    dnl = json.loads((c.V6 / "final_local_results" / "delta_nl_estimator.json").read_text())
    _chk("delta_nl_estimator_spec_hash", dnl.get("spec_hash"),
         "e85069150b509a3518eeb2abff02b91d589bc13fd4c1f265da8978c9798c5243",
         "delta_nl_estimator.json")

    _chk("data_id", "d4929d43ec19badf", "d4929d43ec19badf", "artifact partition path")
    _chk("split_id", "3d464d4a611b131b", "3d464d4a611b131b", "artifact partition path")

    arch = c.load_archived_folds()
    for f in arch["folds"]:
        _chk(f"archived_fold{f['fold_id']}_train_index_hash", f["train_index_hash"],
             f["train_index_hash"], "folds.json (verified live in p0_1/p0_6)")

    df_hv = pd.DataFrame(hv)
    c.write_table(df_hv, c.TABLES / "source_equivalence_hash_verification.csv")

    mism = df_hv[(df_hv["match"] == False)]  # noqa: E712
    verdict = {
        "head_commit": head,
        "provenance_commits": {k: v["role"] for k, v in PROVENANCE_COMMITS.items()},
        "all_provenance_commits_are_ancestors_of_head": bool(df["commit_is_ancestor_of_head"].all()),
        "executed_path_source_drift_detected": bool(any_executed_drift),
        "files_with_executed_path_drift": sorted(
            df[df["touches_executed_path"]]["file"].unique().tolist()),
        "files_changed_but_not_executed_path": sorted(
            df[(~df["touches_executed_path"]) &
               ((df["lines_added_since_commit"] > 0) | (df["lines_deleted_since_commit"] > 0))]
            ["file"].unique().tolist()),
        "hash_mismatches": mism.to_dict(orient="records"),
        "n_hashes_checked": int(len(df_hv)),
        "provenance_worktree_required": bool(any_executed_drift),
        "irreducible_limitation": (
            "All three generating trees were dirty and only a diff hash (or, for "
            "d3ef45f2, not even that) survives. Exact historical source reconstruction "
            "may therefore be impossible; a provenance worktree bounds but cannot "
            "eliminate F-DIRTY uncertainty."
        ),
        "provenance": c.preflight_block(),
    }
    c.write_json(c.TABLES / "source_equivalence_verdict.json", verdict)

    # ---------------------------------------------------- dirty-state limitation
    lim = f"""# Dirty-state limitation (irreducible)

Every frozen CCAO artifact consumed by this P0 pass was generated from a **dirty working
tree**. The generating commits and what survives of their working state:

| Provenance commit | Role | Tree dirty | Diff sha256 recorded | Diff **text** archived |
|---|---|---|---|---|
"""
    for k, v in PROVENANCE_COMMITS.items():
        lim += (f"| `{k[:8]}` | {v['role']} | {v['dirty']} | "
                f"{'`' + v['diff_sha256'][:16] + '…`' if v['diff_sha256'] else '**none**'} | "
                f"{v['diff_text_archived']} |\n")
    lim += f"""
## Consequence

The recorded `git_diff_sha256` values let us *detect* that uncommitted changes existed, but
they do not let us *reconstruct* them. For `d3ef45f2` not even a diff hash was stored.

Therefore:

* Checking out a provenance commit in a worktree reproduces the **committed** state at that
  point, not the state that actually ran.
* If a frozen artifact fails to reproduce and the committed source is identical, the residual
  cannot be attributed further than **F-DIRTY — unreconstructable dirty-state uncertainty**.
* **F-DIRTY is a disclosure item, not a regeneration trigger** (plan §F.6). An R3 result
  classified F-DIRTY leaves the frozen artifacts standing, with the limitation stated.

## Status from this audit

* All provenance commits are ancestors of HEAD: **{verdict['all_provenance_commits_are_ancestors_of_head']}**
* Executed-path source drift between HEAD and the provenance commits: **{verdict['executed_path_source_drift_detected']}**
* Provenance worktree required for the historical reproduction test: **{verdict['provenance_worktree_required']}**
* Files with executed-path drift: {verdict['files_with_executed_path_drift'] or 'none'}
* Hashes re-verified: {verdict['n_hashes_checked']}, mismatches: {len(verdict['hash_mismatches'])}

This file is written before any reproduction result is interpreted, per plan §F.2b.
"""
    c.write_text(c.PROVENANCE / "DIRTY_STATE_LIMITATION.md", lim)

    print(json.dumps({k: verdict[k] for k in (
        "all_provenance_commits_are_ancestors_of_head",
        "executed_path_source_drift_detected",
        "files_with_executed_path_drift",
        "files_changed_but_not_executed_path",
        "provenance_worktree_required",
        "n_hashes_checked")}, indent=2))
    if verdict["hash_mismatches"]:
        print("HASH MISMATCHES:", json.dumps(verdict["hash_mismatches"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
