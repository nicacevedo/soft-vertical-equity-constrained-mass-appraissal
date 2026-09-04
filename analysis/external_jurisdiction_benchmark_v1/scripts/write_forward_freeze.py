#!/usr/bin/env python3
"""Step 13: FORWARD_FREEZE.yaml. No 2025 outcome may be used to alter this
file, and the 2025 forward evaluation may not run until this file exists and
is hashed."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from analysis.external_jurisdiction_benchmark_v1.scripts.v1_common import ANALYSIS, sha256_file, write_json  # noqa: E402


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def yaml_safe(obj):
    """Recursively convert numpy/pandas scalars to plain Python for safe_dump.

    yaml.safe_dump cannot represent numpy.int64/float64/bool_, and every record
    here comes from pd.read_csv, so dumping directly raises RepresenterError.
    NaN/inf become None: YAML has no portable spelling for them and a freeze
    file must round-trip cleanly.

    Deliberately NOT a json.dumps round-trip: numpy.float64 subclasses Python
    float, so json serialises it directly and never consults a `default` hook,
    which makes allow_nan=False raise instead of sanitising (verified).
    """
    if isinstance(obj, dict):
        return {str(k): yaml_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [yaml_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [yaml_safe(v) for v in obj.tolist()]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if np.isfinite(f) else None
    if obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if pd.isna(obj):
        return None
    return str(obj)


def usable_grid_extent() -> dict:
    """Per-jurisdiction/family record of which path points were numerically
    usable, so the freeze states the ANALYSED grid, not just the declared one."""
    out = {}
    for path in sorted((ANALYSIS / "cv").glob("*_normalized_cv_path_summary.csv")):
        name = path.name.replace("_normalized_cv_path_summary.csv", "")
        df = pd.read_csv(path)
        if "rho_tilde" not in df.columns:
            continue
        status = df["fit_status"].astype(str) if "fit_status" in df.columns else None
        ok = df if status is None else df.loc[status == "OK"]
        excluded = {} if status is None else {
            str(k): int(v) for k, v in status.loc[status != "OK"].value_counts().items()
        }
        out[name] = {
            "n_rows": int(len(df)),
            "n_usable_rows": int(len(ok)),
            "excluded_by_status": excluded,
            "max_usable_rho_tilde": (float(ok["rho_tilde"].max()) if len(ok) else None),
            "declared_max_rho_tilde": float(df["rho_tilde"].max()),
        }
    return out


def main() -> int:
    baseline_freeze_path = ANALYSIS / "baseline" / "BASELINE_FREEZE.yaml"
    if not baseline_freeze_path.exists():
        raise SystemExit("BASELINE_FREEZE.yaml missing; cannot write FORWARD_FREEZE.yaml")
    baseline_freeze = yaml.safe_load(baseline_freeze_path.read_text())

    candidate_regions = read_csv(ANALYSIS / "candidate_regions" / "candidate_regions.csv")
    lofo_summary = read_csv(ANALYSIS / "candidate_regions" / "lofo_summary.csv")
    anchors = read_csv(ANALYSIS / "candidate_regions" / "achieved_mechanism_anchors.csv")
    portability = read_csv(ANALYSIS / "tables" / "normalization_portability.csv")
    band = read_csv(ANALYSIS / "candidate_regions" / "cross_jurisdiction_band.csv")

    direct_meta = {}
    surrogate_meta = {}
    for meta_path in (ANALYSIS / "cv").glob("*_direct_cv_meta.json"):
        d = json.loads(meta_path.read_text())
        direct_meta[d["county_key"]] = d
    for meta_path in (ANALYSIS / "cv").glob("*_surrogate_cv_meta.json"):
        d = json.loads(meta_path.read_text())
        surrogate_meta[d["county_key"]] = d

    ANALYSIS.joinpath("path_freeze").mkdir(parents=True, exist_ok=True)
    out_path = ANALYSIS / "path_freeze" / "FORWARD_FREEZE.yaml"
    provisional_archive = ANALYSIS / "path_freeze" / "FORWARD_FREEZE.provisional_pre_integrity_amendment.yaml"
    provisional_sha = None
    if out_path.exists() and not provisional_archive.exists():
        provisional_archive.write_text(out_path.read_text())
        provisional_sha = sha256_file(provisional_archive)
    elif provisional_archive.exists():
        provisional_sha = sha256_file(provisional_archive)

    freeze = {
        "schema_version": 1, "status": "FROZEN",
        "frozen_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rule": (
            "No 2025 outcome may be used to alter this file. The 2025 forward evaluation "
            "may not run until this file exists and is hashed."
        ),
        "integrity_amendment": {
            "name": "pre_forward_integrity_amendment",
            "reason": (
                "Candidate-region status now distinguishes a protocol-valid nonempty "
                "LOFO-stable interval from preserved point-estimate endpoints "
                "(UPPER_GUARDRAIL_PRECEDES_ACTIVITY / NO_STABLE_CANDIDATE_REGION). "
                "Cross-jurisdiction overlap is recomputed on protocol-valid regions "
                "only; a knife-edge Surrogate intersection is labeled "
                "NO_NONDEGENERATE_ALL_JURISDICTION_INTERSECTION. Continuous-endpoint "
                "intersection vs 200-point evaluation-grid >=75% coverage is explicit. "
                "An earlier provisional freeze is archived, not concealed. 2025 remains untouched."
            ),
            "supersedes_provisional_freeze": str(provisional_archive) if provisional_archive.exists() else None,
            "provisional_sha256": provisional_sha,
            "2025_still_untouched": True,
        },
        "baseline_freeze_file": str(baseline_freeze_path),
        "baseline_freeze_sha256": sha256_file(baseline_freeze_path),
        "jurisdiction_roles": {u["county_key"]: u.get("role") for u in baseline_freeze.get("units", [])},
        "canonical_history_sources": {
            u["county_key"]: u.get("canonical_history_source") for u in baseline_freeze.get("units", [])
        },
        "cohort_and_model_table_hashes": {
            u["county_key"]: {"modeling_table_hash": u.get("modeling_table_hash"), "cache_hashes": u.get("cache_hashes")}
            for u in baseline_freeze.get("units", []) if u.get("status") == "OK"
        },
        "baseline_configs": {
            u["county_key"]: u.get("selected_lgbm_config") for u in baseline_freeze.get("units", []) if u.get("status") == "OK"
        },
        "fold_dates": {"scheme": "expanding_calendar_year", "validation_years": [2018, 2019, 2020, 2021, 2022, 2023, 2024]},
        "normalized_path_grid": {"n_points": 34, "min_rho_tilde": 1e-3, "max_rho_tilde": 150.0, "includes_zero": True},
        "numerical_validity_rules": {
            "training_support_bound": (
                "A Direct path point is invalid where predicted log price leaves the training "
                "label range extended by one full range width on each side: pred_log must lie in "
                "[y_min-(y_max-y_min), y_max+(y_max-y_min)]. Defined only from the training "
                "support -- never from R2/PRD/PRB/MKI/VEI/beta -- so it cannot select on the "
                "outcome of interest. Approved 2026-09-04; applied uniformly to all "
                "jurisdictions. Recorded as fit_status=DIVERGED_OUTSIDE_TRAINING_SUPPORT."
            ),
            "screening_metric_finiteness": (
                "A path point is invalid if any boundary-driving metric "
                "(PRD, PRB, MKI, VEI, Beta_log, R2_price, MAE_price, MAPE, RMSE_log) is "
                "non-finite. Recorded as fit_status=NUMERICALLY_UNSTABLE_RHO."
            ),
            "surrogate_family_exposure": (
                "The Surrogate family is structurally not exposed to these failure modes: "
                "first_branch_calibrate terminates the branch (GRID_CEILING/MATERIAL_REVERSAL) "
                "before the divergent tail. Verified: min R2_price across all 9 jurisdictions "
                "is +0.02, max MAE_price 3.33e5, max MAPE 0.39 -- no divergence signature."
            ),
        },
        "usable_grid_extent": usable_grid_extent(),
        "band_semantics": (
            "Intersection = continuous [max activity, min guardrail] across the stated "
            "sample. >=75% coverage = continuous interval membership evaluated on a "
            "200-point geometric grid spanning [min activity, max guardrail] of that "
            "sample -- not coverage counted only at the 33 tested CV rho_tilde points."
        ),
        "direct_path_meta": direct_meta,
        "surrogate_first_branch_meta": surrogate_meta,
        "candidate_region_endpoints": candidate_regions.to_dict("records") if len(candidate_regions) else [],
        "lofo_diagnostics": lofo_summary.to_dict("records") if len(lofo_summary) else [],
        "normalization_portability": portability.to_dict("records") if len(portability) else [],
        "cross_jurisdiction_band": band.to_dict("records") if len(band) else [],
        "achieved_mechanism_anchors": anchors.to_dict("records") if len(anchors) else [],
    }
    ANALYSIS.joinpath("path_freeze").mkdir(parents=True, exist_ok=True)
    out_path = ANALYSIS / "path_freeze" / "FORWARD_FREEZE.yaml"
    out_path.write_text(yaml.safe_dump(yaml_safe(freeze), sort_keys=False, default_flow_style=False))
    freeze_hash = sha256_file(out_path)
    write_json(ANALYSIS / "path_freeze" / "FORWARD_FREEZE.sha256.json", {"path": str(out_path), "sha256": freeze_hash})
    print(f"wrote {out_path}")
    print(f"sha256={freeze_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
