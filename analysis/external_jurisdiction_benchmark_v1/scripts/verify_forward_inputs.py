#!/usr/bin/env python3
"""Pre-2025 freeze/input identity check. Refuses to proceed on mismatch.

Does not build 2025 modeling tables and does not fit any 2025 path. It may
hash already-frozen development tables and caches.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from analysis.external_jurisdiction_benchmark_v1.scripts.forward_common import (  # noqa: E402
    EXPECTED_FORWARD_FREEZE_SHA256, ANALYSIS, OUTPUT, assert_dev_table_hash,
    verify_forward_freeze, write_input_freeze_check,
)
from analysis.external_jurisdiction_benchmark_v1.scripts.v1_common import (  # noqa: E402
    ALL_KEYS, sha256_file,
)

OUT_PATH = ANALYSIS / "forward_2025" / "audits" / "forward_input_freeze_check.json"


def cache_hashes(key: str) -> dict:
    manifest_path = OUTPUT / "cache" / key / "cache_manifest.json"
    if manifest_path.exists():
        recs = json.loads(manifest_path.read_text())
        return {c.get("product"): c.get("sha256") for c in recs}
    out = {}
    for product in ("recorder", "history"):
        p = OUTPUT / "cache" / key / f"{product}.parquet"
        if p.exists():
            out[product] = sha256_file(p)
    return out


def existing_2025_artifacts() -> list[str]:
    hits = []
    tracked = ANALYSIS / "forward_2025"
    if tracked.exists():
        for p in tracked.rglob("*"):
            if not p.is_file():
                continue
            rel = str(p.relative_to(ANALYSIS))
            if rel.startswith("forward_2025/audits/evaluation_layer_definition.yaml"):
                continue
            if rel.startswith("forward_2025/audits/forward_input_freeze_check.json"):
                continue
            hits.append(rel)
    out_root = OUTPUT / "forward_2025"
    if out_root.exists():
        for p in out_root.rglob("*"):
            if p.is_file():
                hits.append(str(p.relative_to(OUTPUT.parent.parent)))
    return sorted(hits)


def main() -> int:
    freeze = verify_forward_freeze()
    roles = freeze["jurisdiction_roles"]
    configs = freeze["baseline_configs"]
    unit_hashes = {}
    cache_ok = True
    for key in ALL_KEYS:
        dev_sha = assert_dev_table_hash(key, freeze)
        cfg_path = ANALYSIS / "baseline" / f"{key}_baseline_config.json"
        cfg = json.loads(cfg_path.read_text())
        selected = cfg["selected_lgbm_config"]
        if selected != configs[key]:
            raise RuntimeError(f"{key}: baseline config {selected} != freeze {configs[key]}")
        if cfg.get("no_2025_data_used") is not True:
            raise RuntimeError(f"{key}: baseline config does not record no_2025_data_used")
        caches = cache_hashes(key)
        expected_caches = freeze["cohort_and_model_table_hashes"][key]["cache_hashes"]
        cache_match = caches == expected_caches
        if not cache_match:
            cache_ok = False
        unit_hashes[key] = {
            "role": roles[key],
            "modeling_table_hash": dev_sha,
            "selected_lgbm_config": selected,
            "cache_hashes": caches,
            "cache_hashes_match_freeze": cache_match,
        }
    prior = existing_2025_artifacts()
    extra = {
        "checked_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "units": unit_hashes,
        "prior_2025_artifacts": prior,
        "cache_hashes_all_match": cache_ok,
        "direct_grid_points": 34,
        "surrogate_grid_points": 33,
        "independent_pre_2025_test_split": False,
        "evaluation_layers": ["CV_FOLD", "CV_OOF", "FORWARD_2025"],
    }
    if not cache_ok:
        rec = write_input_freeze_check(OUT_PATH, extra | {"ok": False})
        raise RuntimeError(f"cache hash mismatch; wrote {OUT_PATH} with ok=false")
    if prior:
        rec = write_input_freeze_check(OUT_PATH, extra | {"ok": False})
        raise RuntimeError(f"prior 2025 artifacts exist: {prior[:20]}")
    rec = write_input_freeze_check(OUT_PATH, extra)
    print(json.dumps({
        "ok": True,
        "forward_freeze_sha256": EXPECTED_FORWARD_FREEZE_SHA256,
        "n_jurisdictions": len(ALL_KEYS),
        "wrote": str(OUT_PATH),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
