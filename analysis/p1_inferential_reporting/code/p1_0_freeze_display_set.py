#!/usr/bin/env python3
"""P1 step 0: freeze the bounded display set BEFORE any result is read.

Enumerates display coordinates only -- reference cells, the frozen matched-beta
K_core and EXT entries, and the five plan display anchors -- and resolves each to
its cached prediction artifact.  No metric, no outcome and no prediction value is
read here; only coordinates, artifact paths and the frozen attainability state.

Attainability is carried VERBATIM from configs/matched_beta_frozen.json:
no interpolation, no synthetic rho, no substitute configuration, and
NOT_ATTAINED entries are preserved explicitly rather than dropped.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p1_common as c                                                   # noqa: E402

FAMILY_MODEL = {"Direct": "LGBCovPenalty", "Surrogate": "LGBSmoothPenalty"}
POSTHOC_REF = {"C-posthoc": "C", "A-posthoc": "A"}
FOLD_EVALS = [f"fold_{k}" for k in range(1, 8)]
EVALUATIONS = FOLD_EVALS + ["pooled_oof", "heldout", "forward_2025"]


def _resolve_fitted(cmap: pd.DataFrame, cvmap: pd.DataFrame, family: str, rho: float) -> dict:
    """Cached prediction artifacts for a fitted (family, rho); zero refits."""
    mn = FAMILY_MODEL[family]
    out = {"kind": "fitted", "model_name": mn, "rho": float(rho), "artifacts": {}}
    for stage in ("heldout", "forward_2025"):
        s = cmap[(cmap.model_name == mn) & (cmap.stage == stage)
                 & (cmap.rho.astype(float) - float(rho)).abs().le(1e-12)]
        if len(s) != 1:
            raise c.ProtocolViolation(f"{family} rho={rho} {stage}: {len(s)} matches, expected 1")
        r = s.iloc[0]
        out["config_id"] = str(r.config_id)
        out["artifacts"][stage] = str(r.pred_file)
    for k in range(1, 8):
        s = cvmap[(cvmap.model_name == mn) & (cvmap.fold_1based == k)
                  & (cvmap.rho.astype(float) - float(rho)).abs().le(1e-12)]
        if len(s) != 1:
            raise c.ProtocolViolation(f"{family} rho={rho} fold_{k}: {len(s)} matches, expected 1")
        out["artifacts"][f"fold_{k}"] = str(s.iloc[0].pred_file)
    return out


def _resolve_posthoc(family: str, b: float) -> dict:
    """Post-hoc entries are evaluated transformations of the Stage-1.5 zero-reference
    fits: f_b = ybar_T + b*(f0 - ybar_T).  Artifacts are resolved by cell/block."""
    ref = POSTHOC_REF[family]
    zf = pd.read_csv(c.P0_TABLES / "zero_reference_fits.csv")
    arts = {}
    block_for = {**{f"fold_{k}": f"fold_{k}_train" for k in range(1, 8)},
                 "heldout": "development_pool", "forward_2025": "production_2016_2024"}
    for ev, blk in block_for.items():
        r = zf[(zf.cell_id == ref) & (zf.block_id == blk)]
        if len(r) != 1:
            raise c.ProtocolViolation(f"no unique zero-reference fit for {ref}/{blk}")
        arts[ev] = str(r.iloc[0].eval_predictions)
    return {"kind": "posthoc", "posthoc_ref_cell": ref, "b": float(b), "artifacts": arts}


def main() -> int:
    c.assert_d3_multiplicity_identities()
    F = c.matched_beta_frozen()
    cmap = pd.read_csv(c.FROZEN_CONFIG_MAP)
    cvmap = pd.read_csv(c.FROZEN_CV_RUN_MAP)

    entries = []

    # ---- 1. reference cells (B is decomposition-only and is deliberately absent)
    entries.append({"display_kind": "reference_cell", "reference_cell": "C",
                    "display_name": "Custom-objective rho=0 origin",
                    "cell_role": "PRIMARY within-path penalty-isolating reference",
                    "family": "Direct", "role": "REFERENCE", "j": None, "ext_target": None,
                    "target": None, "rho": 0.0, "b": None, "attained": True,
                    "match_mode": "exact_anchor", "match_gap": 0.0,
                    "achieved_dev_beta": None, "max_achieved_dev_correction": None,
                    **_resolve_fitted(cmap, cvmap, "Direct", 0.0)})
    a = cmap[(cmap.model_name == "LGBMRegressor")]
    arts_a = {r.stage: str(r.pred_file) for _, r in a.iterrows()}
    zf = pd.read_csv(c.P0_TABLES / "zero_reference_fits.csv")
    for k in range(1, 8):
        r = zf[(zf.cell_id == "A") & (zf.block_id == f"fold_{k}_train")].iloc[0]
        arts_a[f"fold_{k}"] = str(r.eval_predictions)
    entries.append({"display_kind": "reference_cell", "reference_cell": "A",
                    "display_name": c.CELL_A_NAME,
                    "cell_role": "assessor-facing / workflow benchmark",
                    "family": "A-native", "role": "REFERENCE", "j": None, "ext_target": None,
                    "target": None, "rho": None, "b": None, "attained": True,
                    "match_mode": "native_baseline", "match_gap": None,
                    "achieved_dev_beta": None, "max_achieved_dev_correction": None,
                    "kind": "fitted", "model_name": "LGBMRegressor",
                    "config_id": str(a.iloc[0].config_id), "artifacts": arts_a})

    # ---- 2. matched-beta K_core (j = 0..5) and ---- 3. EXT targets
    for rec_list, kind, tf in ((F["matched_configurations"], "matched_beta_core", "target"),
                               (F["ext_matched"], "matched_beta_ext", "target")):
        for rec in rec_list:
            fam = rec["family"]
            e = {"display_kind": kind, "reference_cell": None, "display_name": None,
                 "cell_role": None, "family": fam, "role": rec["role"],
                 "j": rec.get("j"), "ext_target": rec[tf] if kind.endswith("ext") else None,
                 "target": rec[tf], "rho": rec.get("rho"), "b": rec.get("b"),
                 "attained": bool(rec["attained"]), "match_mode": rec["match_mode"],
                 "match_gap": rec.get("match_gap"),
                 "achieved_dev_beta": rec.get("achieved_dev_beta"),
                 "max_achieved_dev_correction": rec.get("max_achieved_dev_correction")}
            if not rec["attained"]:
                # Preserved explicitly. No interpolation, no synthetic rho, no substitute.
                e.update({"kind": "NOT_ATTAINED", "model_name": None, "config_id": None,
                          "artifacts": {},
                          "not_attained_note": "NOT_ATTAINED within the frozen design"})
            elif fam in FAMILY_MODEL:
                e.update(_resolve_fitted(cmap, cvmap, fam, float(rec["rho"])))
            else:
                e.update(_resolve_posthoc(fam, float(rec["b"])))
            entries.append(e)

    # ---- 4. the five plan display anchors, Direct + Surrogate
    for rho in c.DISPLAY_ANCHORS:
        for fam in ("Direct", "Surrogate"):
            entries.append({"display_kind": "plan_display_anchor", "reference_cell": None,
                            "display_name": None, "cell_role": None, "family": fam,
                            "role": "DISPLAY_ANCHOR", "j": None, "ext_target": None,
                            "target": None, "rho": float(rho), "b": None, "attained": True,
                            "match_mode": "exact_anchor", "match_gap": 0.0,
                            "achieved_dev_beta": None, "max_achieved_dev_correction": None,
                            **_resolve_fitted(cmap, cvmap, fam, float(rho))})

    # ---- dedup key: identical realized configuration (roles preserved separately)
    for e in entries:
        if e["kind"] == "fitted":
            e["realization_key"] = f"fit:{e['model_name']}:{e['config_id']}"
        elif e["kind"] == "posthoc":
            e["realization_key"] = f"posthoc:{e['posthoc_ref_cell']}:b={e['b']!r}"
        else:
            e["realization_key"] = "NOT_ATTAINED"

    real = {}
    for e in entries:
        real.setdefault(e["realization_key"], []).append(
            {k: e[k] for k in ("display_kind", "family", "role", "j", "ext_target", "rho", "b")})

    payload = {
        "schema_version": 1,
        "stage": "P1_INFERENTIAL_REPORTING",
        "frozen_before_any_result_is_read": True,
        "p0_tag": c.P0_TAG, "p0_commit": c.P0_COMMIT,
        "matched_beta_frozen_sha256": c.sha256_file(c.MATCHED_BETA_FROZEN),
        "matched_beta_frozen_at_utc": F["frozen_at_utc"],
        "evaluations": EVALUATIONS,
        "reference_cells_included": ["A", "C"],
        "reference_cell_B_excluded": ("Cell B is an implementation-decomposition control only "
                                      "(post_g1_reference_convention.yaml) and carries no P1 "
                                      "inference row."),
        "attainability_rules": [
            "attained / match_mode / match_gap are carried verbatim from matched_beta_frozen.json",
            "no interpolation", "no synthetic rho", "no substitute configuration",
            "NOT_ATTAINED entries are preserved explicitly with null metrics, never dropped",
            "identical realized configurations are computed once; all role labels are preserved",
        ],
        "n_entries": len(entries),
        "n_attained": sum(1 for e in entries if e["attained"]),
        "n_not_attained": sum(1 for e in entries if not e["attained"]),
        "n_distinct_realizations": len([k for k in real if k != "NOT_ATTAINED"]),
        "realizations": real,
        "entries": entries,
        "provenance": c.preflight_block(),
    }
    out = c.write_json(c.CONFIGS / "display_set_frozen.json", payload)
    h = {"file": out.name, "file_sha256": c.sha256_file(out),
         "frozen_at_utc": pd.Timestamp.utcnow().isoformat()}
    c.write_json(c.CONFIGS / "display_set_frozen_hash.json", h)

    print(f"[freeze] entries={len(entries)}  attained={payload['n_attained']}  "
          f"NOT_ATTAINED={payload['n_not_attained']}  "
          f"distinct realizations={payload['n_distinct_realizations']}")
    na = [(e["family"], e["ext_target"], e["match_mode"]) for e in entries if not e["attained"]]
    print("[freeze] NOT_ATTAINED entries preserved:")
    for x in na:
        print("        ", x)
    dup = {k: v for k, v in real.items() if len(v) > 1 and k != "NOT_ATTAINED"}
    print(f"[freeze] realizations reached by >1 display role: {len(dup)}")
    for k, v in dup.items():
        print(f"         {k}")
        for r in v:
            print(f"            {r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
