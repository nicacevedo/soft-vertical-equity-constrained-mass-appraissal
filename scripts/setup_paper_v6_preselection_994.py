#!/usr/bin/env python3
"""Freeze the 994-tree search winner as the paper-v6 preselection baseline.

No model selection among penalized configurations. This only records the
pre-penalty CV gate decision and writes the LightGBM freeze used by the
994 rerun.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from canonical_experiment import git_state, lgbm_params_hash, package_versions, write_json

ROOT_500 = REPO / "output" / "paper_v6_preselection"
ROOT_994 = REPO / "output" / "paper_v6_preselection_994"
SEARCH_BEST = (
    REPO
    / "output"
    / "robust_rolling_origin_cv_v2"
    / "baseline_lgbm_search"
    / "analysis"
    / "data_id=851e64027e3a1698"
    / "split_id=3d464d4a611b131b"
    / "baseline_lgbm_best_params.json"
)

# Equal-weight mean of RMSE_price on the seven chronological folds.
FOLDS_500 = [91654.177078, 99612.686890, 91170.166299, 94926.636487, 122295.315480, 119875.956925, 123584.937154]
FOLDS_994 = [88369.523244, 98465.559918, 89243.657976, 92667.212221, 115050.148635, 115295.185482, 117421.345527]


def main() -> None:
    best = json.loads(SEARCH_BEST.read_text(encoding="utf-8"))
    params = dict(best["best_lgbm_params"])
    param_hash = lgbm_params_hash(params)
    git = git_state()
    versions = package_versions()

    mean_500 = float(sum(FOLDS_500) / 7.0)
    mean_994 = float(sum(FOLDS_994) / 7.0)
    diffs = [a - b for a, b in zip(FOLDS_994, FOLDS_500)]

    gate = {
        "decision": "ADOPT_994",
        "reason": (
            "The 994-tree search winner strictly improves the original pre-penalty "
            "criterion mean_validation_RMSE_price on the exact same seven chronological "
            "folds. Fold train/validation index hashes match the completed 500-tree CV. "
            "data_id strings differ only because the search data signature includes "
            "mode=baseline_lgbm_search; the 500-tree default in that search reproduces "
            "the completed experiment's native LightGBM fold RMSE_price to machine precision."
        ),
        "comparability": {
            "search_data_id": "851e64027e3a1698",
            "experiment_data_id": "d4929d43ec19badf",
            "split_id": "3d464d4a611b131b",
            "split_id_identical": True,
            "fold_index_hashes_identical": True,
            "primary_metric": "mean_validation_RMSE_price",
            "metric_column": "RMSE_price",
            "data_id_differs_due_to_search_mode_flag": True,
        },
        "folds_500_RMSE_price": FOLDS_500,
        "folds_994_RMSE_price": FOLDS_994,
        "mean_500": mean_500,
        "mean_994": mean_994,
        "diff_994_minus_500": diffs,
        "strictly_improves": bool(mean_994 < mean_500 - 1e-12 and all(d < 0.0 for d in diffs)),
        "config_id_500_section2": "7ce5a0a0e6e4f38b",
        "config_id_500_search_default": "3f1c683387b597c1",
        "config_id_994": "407d47775760c14d",
        "n_estimators_500": 500,
        "n_estimators_994": 994,
        "final_result_root": str(ROOT_994),
        "do_not_use_500_penalty_paths_as_headline": True,
        "selection_among_penalized_models": False,
        **git,
        "versions": versions,
    }

    for root in (ROOT_500, ROOT_994):
        (root / "logs").mkdir(parents=True, exist_ok=True)
        (root / "manifests").mkdir(parents=True, exist_ok=True)
        write_json(root / "baseline_gate.json", gate)

    config_payload = {
        "source": "seven_fold_search_winner",
        "search_criterion": "mean_validation_RMSE_price",
        "config_id": "407d47775760c14d",
        "lgbm_params": params,
        "lgbm_params_sha256": param_hash,
        "n_estimators": int(params["n_estimators"]),
        "search_best_params_json": str(SEARCH_BEST),
        "search_best_mean_validation_rmse": float(best.get("best_mean_validation_rmse", mean_994)),
        "replaces_section2_500_tree_baseline": True,
        "section2_500_config_id": "7ce5a0a0e6e4f38b",
        "identification": (
            "Pre-penalty chronological-CV gate adopted the archived 994-tree "
            "baseline-search winner because it strictly improved mean_validation_RMSE_price "
            "on the identical seven folds used by the 500-tree Section-2 LightGBM."
        ),
        "versions": versions,
        **git,
    }
    write_json(ROOT_994 / "lgbm_config.json", config_payload)
    write_json(
        ROOT_994 / "frozen_baseline.json",
        {
            "best_lgbm_params": params,
            "source": "seven_fold_search_winner",
            "search_criterion": "mean_validation_RMSE_price",
            "n_folds_protocol": "paper_v6_seven_fold_expanding_15mo",
            "seed": 2025,
            "lgbm_params_sha256": param_hash,
            "lgbm_config_json": str(ROOT_994 / "lgbm_config.json"),
            "config_id": "407d47775760c14d",
            "versions": versions,
            **git,
        },
    )
    write_json(
        ROOT_994 / "experiment_manifest.json",
        {
            "experiment": "paper_v6_preselection_994",
            "selection_performed": False,
            "no_selection_confirmation": (
                "No rho, penalty family, or penalized configuration was selected or ranked in this analysis."
            ),
            "canonical_grid": {
                "LinearRegression": 1,
                "LGBMRegressor": 1,
                "LGBCovPenalty": 51,
                "LGBSmoothPenalty": 51,
                "total_configs": 104,
                "folds": 7,
                "expected_cv_fits": 728,
            },
            "result_root": str(ROOT_994),
            "lgbm_config_json": str(ROOT_994 / "lgbm_config.json"),
            "frozen_baseline": str(ROOT_994 / "frozen_baseline.json"),
            "baseline_gate": str(ROOT_994 / "baseline_gate.json"),
            **git,
            "versions": versions,
        },
    )
    print(json.dumps({"decision": "ADOPT_994", "mean_500": mean_500, "mean_994": mean_994, "root": str(ROOT_994)}, indent=2))


if __name__ == "__main__":
    main()
