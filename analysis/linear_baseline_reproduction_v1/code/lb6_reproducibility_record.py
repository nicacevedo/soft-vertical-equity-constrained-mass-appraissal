#!/usr/bin/env python3
"""LB Stage 6 -- the reproducibility record for the linear-baseline reproduction.

Binds every reported number to an input by content hash, records the split and
fold identities, the preprocessing, the metric-code version and the scripts, and
hash-identifies the restricted row-level prediction artifacts instead of copying
sale-level data anywhere.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lb_common as lb


def rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(lb.REPO.resolve()))
    except ValueError:
        return str(p)


def main() -> int:
    parity = pd.read_csv(lb.TABLES / "lb_frozen_lightgbm_parity.csv")
    lb2 = json.loads((lb.PROVENANCE / "lb2_compatibility.json").read_text())
    lb1 = json.loads((lb.PROVENANCE / "lb1_verdict.json").read_text())
    lb3 = json.loads((lb.PROVENANCE / "lb3_figure.json").read_text())
    lb4 = json.loads((lb.PROVENANCE / "lb4_smearing.json").read_text())
    lb5 = json.loads((lb.PROVENANCE / "lb5_assets.json").read_text())

    # restricted inputs: identified by hash, never copied
    restricted = []
    for _, r in parity.iterrows():
        ev = r["evaluation"]
        if ev in ("heldout", "forward_2025"):
            lp = lb.frozen_oos_prediction_path(lb.LINEAR_CONFIG_ID, ev)
            np_ = lb.frozen_oos_prediction_path(lb.NATIVE_CONFIG_ID, ev)
        else:
            k = int(r["fold_1based"])
            lp = lb.frozen_fold_prediction_path(lb.LINEAR_CONFIG_ID, k)
            np_ = lb.frozen_fold_prediction_path(lb.NATIVE_CONFIG_ID, k)
        for model, p, arrsha in ((lb.LINEAR_NAME, lp, r["linear_pred_sha256"]),
                                 (lb.NATIVE_NAME, np_, r["native_pred_sha256_recomputed"])):
            restricted.append({
                "evaluation": ev, "model": model, "role": "row-level predictions (RESTRICTED)",
                "path": rel(p), "file_sha256": lb.sha256_file(p),
                "bytes": int(Path(p).stat().st_size),
                "y_pred_log_float64_sha256": arrsha,
                "redistributed": False,
            })
        restricted.append({
            "evaluation": ev, "model": "shared", "role": "frozen benchmark eval predictions (RESTRICTED)",
            "path": rel(lb.ZREF / f"cell=A" / f"block={r['block_id']}" / "eval_predictions.parquet"),
            "file_sha256": lb.sha256_file(lb.ZREF / "cell=A" / f"block={r['block_id']}" / "eval_predictions.parquet"),
            "bytes": int((lb.ZREF / "cell=A" / f"block={r['block_id']}" / "eval_predictions.parquet").stat().st_size),
            "y_pred_log_float64_sha256": r["frozen_benchmark_eval_pred_sha256"],
            "redistributed": False,
        })
    restricted = pd.DataFrame(restricted)
    lb.write_table(restricted, lb.TABLES / "lb_restricted_input_manifest.csv", full_precision=False)

    tracked_inputs = {}
    for p in [lb.REPO / "params.yaml", lb.REPO / "run_temporal_cv.py",
              lb.REPO / "utils" / "motivation_utils.py", lb.REPO / "utils" / "delta_nl.py",
              lb.REPO / "preprocessing" / "recipes_pipelined.py",
              lb.FROZEN_CONFIG_MAP, lb.FROZEN_CV_RUN_MAP, lb.ZERO_CONTROL_FULL,
              lb.V6 / "lgbm_config.json", lb.p0.ARCHIVED_FOLDS,
              lb.P1_DIR / "configs" / "smearing_estimator_frozen.json",
              lb.P1_DIR / "tables" / "smearing_factor_provenance.csv",
              lb.REPO / "paper" / "paper_v20.tex", lb.REPO / "paper" / "paper_v20.pdf",
              lb.REPO / "paper" / "paper_v15.tex",
              lb.REPO / "paper" / "img" / "generated_v12_994" / "baseline_models_motivation_2024_2025.pdf",
              lb.REPO / "paper" / "img" / "generated_v6_preselection" / "baseline_models_motivation_2024_2025.pdf",
              lb.REPO / "paper" / "img" / "baseline_models_motivation_2024_2025.pdf",
              ]:
        if Path(p).exists():
            tracked_inputs[rel(p)] = {"sha256": lb.sha256_file(Path(p)),
                                      "bytes": int(Path(p).stat().st_size)}

    outputs = {}
    for p in sorted(list(lb.TABLES.glob("*.csv")) + list(lb.FIGURES.glob("*"))
                    + list(lb.SNIPPETS.glob("*.tex")) + list(lb.PROVENANCE.glob("*.json"))):
        outputs[rel(p)] = {"sha256": lb.sha256_file(p), "bytes": int(p.stat().st_size)}

    record = {
        "stage": "linear_baseline_reproduction_v1",
        "purpose": ("Audit, verify and report the ordinary unpenalized linear-regression "
                    "baseline against ordinary LightGBM under the v20 CCAO design. "
                    "No model was refitted: every number comes from cached row-level "
                    "predictions that this stage verified against the frozen benchmark."),
        "rerun_needed": lb1["cached_linear_prediction_audit"]["rerun_needed"],
        "extract": {
            "path": "data/CCAO/2025/training_data.parquet",
            "resolved_through": "a read-only symlink in this worktree to the main checkout",
            "bytes": lb2["checks"]["C1_extract_identity"]["bytes"],
            "sha256": lb2["checks"]["C1_extract_identity"]["sha256"],
            "matches_v20_record": lb2["checks"]["C1_extract_identity"]["pass"],
            "redistributed": False,
        },
        "samples": lb2["checks"]["C2_sample_counts"],
        "folds": lb2["checks"]["C3_folds"],
        "sale_identity": lb2["checks"]["C4_sale_identity"],
        "predictors_and_preprocessing": lb2["checks"]["C5_predictor_audit"],
        "model_specifications": {
            "linear": {
                "estimator": "sklearn.linear_model.LinearRegression(fit_intercept=True)",
                "defined_at": "run_temporal_cv.py:919-925",
                "target": "np.log(meta_sale_price)",
                "preprocessing": "preprocessing.recipes_pipelined.build_model_pipeline, "
                                 "10 steps, fit on each fitting block's training rows only",
                "config_id": lb.LINEAR_CONFIG_ID,
            },
            "ordinary_lightgbm": {
                "estimator": "run_temporal_cv._native_lgbm_estimator, raw labels, "
                             "boost_from_average default",
                "n_estimators": 994,
                "lgbm_params_sha256": lb.EXPECTED_LGBM_PARAMS_SHA256,
                "config_id": lb.NATIVE_CONFIG_ID,
                "preprocessing": "none; 23 predictors as pandas category dtype",
            },
            "prediction_convention": "direct exponentiation, P_hat = exp(f(x)); verified exact "
                                     "against the stored y_pred column on every artifact",
        },
        "frozen_lightgbm_parity": lb1["frozen_lightgbm_parity"],
        "cached_prediction_audit": {
            k: v for k, v in lb1["cached_linear_prediction_audit"].items() if k != "worst_rows"},
        "metric_code": {
            "functions": ["utils.motivation_utils._compute_extended_metrics",
                          "utils.motivation_utils.paper_mechanism_metrics",
                          "utils.delta_nl.estimate_delta_nl"],
            "called_exactly_as": "analysis/p0_major_revision_validation/code/"
                                 "p0_zero_control_assemble.metrics_from",
            "ratio_mode": "diff",
            "sha256": lb1["preflight"]["metric_code"],
            "delta_nl_estimator_spec_hash": None,
        },
        "regenerated_lightgbm_control": {
            "refitted": False,
            "reason": "the cached ordinary-LightGBM series is bitwise identical to the frozen "
                      "zero-penalty benchmark on all nine blocks, so refitting would only add "
                      "execution-path noise to an exact match",
        },
        "retransformation_sensitivity": lb4,
        "figure": lb3,
        "v15_comparison": {
            "cells_compared": lb5["v15_cells_compared"],
            "cells_differing": lb5["v15_cells_differing_at_displayed_precision"],
            "differences": lb5["v15_differences"],
        },
        "restricted_inputs": "see tables/lb_restricted_input_manifest.csv; identified by content "
                             "hash, never copied out of output/",
        "tracked_inputs": tracked_inputs,
        "outputs": outputs,
        "scripts": ["analysis/linear_baseline_reproduction_v1/code/lb_common.py",
                    "analysis/linear_baseline_reproduction_v1/code/lb1_audit_and_metrics.py",
                    "analysis/linear_baseline_reproduction_v1/code/lb2_compatibility_and_predictors.py",
                    "analysis/linear_baseline_reproduction_v1/code/lb3_figure.py",
                    "analysis/linear_baseline_reproduction_v1/code/lb4_smearing_sensitivity.py",
                    "analysis/linear_baseline_reproduction_v1/code/lb5_paper_assets.py",
                    "analysis/linear_baseline_reproduction_v1/code/lb6_reproducibility_record.py"],
        "run_order": "lb1 -> lb2 -> lb3 -> lb4 -> lb5 -> lb6",
        "environment": lb1["preflight"]["versions"],
        "thread_env": lb1["preflight"]["thread_env"],
        "git": lb1["preflight"]["git"],
        "write_isolation": ("lb_common._assert_write_allowed refuses every write outside "
                            "analysis/linear_baseline_reproduction_v1/ and "
                            "output/linear_baseline_reproduction_v1/, and explicitly refuses "
                            "paper/, utils/, scripts/, preprocessing/, data/, the P0/P1/B0 "
                            "analysis subtrees and the frozen output roots"),
        "known_limitations": [
            "The prediction artifacts carry row_id and sale_date but not meta_pin, so sale "
            "identity is positional against the canonical split plus a bitwise-equal log-price "
            "vector, not a parcel-identifier join.",
            "The seven chronological folds are expanding-window and overlapping; their spread is "
            "descriptive and is not a sampling distribution.",
            "The Duan factor is estimated on development out-of-fold residuals and applied "
            "unchanged to the 2025 block, which has no out-of-fold analogue.",
            "The linear and LightGBM predictor representations are not nested, so the accuracy "
            "gap is not a pure functional-form contrast.",
            "The linear specification is a research benchmark on this extract and design; it is "
            "not the office's historical production model and says nothing about how that "
            "workflow performed.",
        ],
    }
    from utils.delta_nl import estimator_spec_hash
    record["metric_code"]["delta_nl_estimator_spec_hash"] = estimator_spec_hash()

    lb.write_json(lb.PROVENANCE / "lb_reproducibility_record.json", record)
    print(f"[lb6] wrote reproducibility record: "
          f"{len(tracked_inputs)} tracked inputs, {len(outputs)} outputs, "
          f"{len(restricted)} restricted artifacts hash-identified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
