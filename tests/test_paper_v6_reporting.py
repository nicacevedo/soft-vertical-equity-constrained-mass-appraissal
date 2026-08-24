"""Focused reporting-pipeline regression tests. No model selection."""

from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import paper_v6_preselection_pipeline as pipe


def test_exact_cv_expected_config_fold_set():
    rhos = pipe.expected_canonical_rhos()
    assert len(rhos) == 51
    assert rhos[0] == 0.0
    assert np.isclose(rhos[-1], 100.0)
    n_configs = 2 + 2 * len(rhos)
    assert n_configs * 7 == 728


def test_exact_oos_family_rho_set_and_duplicates():
    rhos = pipe.expected_canonical_rhos()
    rows = []
    for fam in ("Direct", "Surrogate"):
        for rho in rhos:
            rows.append({"family": fam, "evaluation": "heldout", "rho": rho, "config_id": f"{fam}-{rho}"})
    oos = pd.DataFrame(rows)
    chk = pipe.exact_oos_family_rho_set(oos, family="Direct", evaluation="heldout", expected_rhos=rhos)
    assert chk["ok"]
    pipe.assert_no_duplicate_pairs(oos, ["family", "rho", "evaluation"], "oos")
    bad = pd.concat([oos, oos.iloc[[0]]], ignore_index=True)
    try:
        pipe.assert_no_duplicate_pairs(bad, ["family", "rho", "evaluation"], "oos")
        raise AssertionError("expected duplicate detection")
    except RuntimeError:
        pass


def test_ratio_shape_config_id_rho_join():
    metrics = pd.DataFrame(
        {
            "config_id": ["c1", "c2"],
            "family": ["Direct", "Direct"],
            "rho": [0.0, 1.15],
            "model_name": [pipe.DIRECT, pipe.DIRECT],
        }
    )
    preds = pd.DataFrame(
        {
            "config_id": ["c1", "c1", "c2"],
            "y_true": [1, 2, 3],
            "y_pred": [1.1, 2.1, 3.1],
            "model_name": [pipe.DIRECT] * 3,
        }
    )
    joined = pipe.join_prediction_metadata(preds, metrics)
    assert set(joined["rho"]) == {0.0, 1.15}
    assert joined["family"].eq("Direct").all()


def test_nearest_anchor_mapping_for_1_and_10():
    grid = pipe.expected_canonical_rhos()
    a1 = pipe.nearest_anchor(grid, 1.0)
    a10 = pipe.nearest_anchor(grid, 10.0)
    assert abs(a1 - 1.0) <= min(abs(g - 1.0) for g in grid)
    assert abs(a10 - 10.0) <= min(abs(g - 10.0) for g in grid)
    mapped = pipe.map_display_anchors(grid, pipe.ANCHORS)
    assert mapped[0] == 0.0
    assert np.isclose(mapped[-1], 100.0)
    assert len(mapped) == 5


def test_ratio_shape_generation_requires_joined_anchor_curves():
    metrics = pd.DataFrame(
        {
            "config_id": [f"c{i}" for i in range(5)],
            "family": ["Direct"] * 5,
            "rho": [0.0, 0.1, 1.15, 9.8, 100.0],
            "model_name": [pipe.DIRECT] * 5,
        }
    )
    preds = pd.DataFrame(
        {
            "config_id": np.repeat(metrics["config_id"].to_numpy(), 40),
            "y_true": np.linspace(50, 400, 200),
            "y_pred": np.linspace(55, 380, 200),
            "model_name": [pipe.DIRECT] * 200,
        }
    )
    joined = pipe.join_prediction_metadata(preds, metrics)
    assert joined["rho"].notna().all()
    assert int(joined.groupby("rho").ngroups) == 5


def test_rho0_prediction_difference_computation():
    native = pd.DataFrame({"row_id": [1, 2, 3], "y_pred_log": [1.0, 1.1, 1.2]})
    out = pipe.compute_rho0_control(native, native.copy(), native.copy())
    assert out["direct_mean_abs_delta_log"] == 0.0
    assert out["surrogate_max_abs_delta_log"] == 0.0
    assert out["direct_vs_surrogate_mean_abs_delta_log"] == 0.0


def test_robust_table_replacement_with_resizebox():
    tex = r"""
\begin{table}[!ht]
\centering
\caption{Implementation control.}
\label{tab:rho_zero_control}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llrrrrr}
Held-out & Ordinary LightGBM & \multicolumn{5}{c}{\textit{populate from full path}} \\
\end{tabular}}
\end{table}
"""
    updated = pipe.replace_latex_environment_by_label(
        tex, "table", "tab:rho_zero_control", r"\begin{table}NEW TABLE\end{table}"
    )
    assert "NEW TABLE" in updated
    assert "populate from full path" not in updated
    assert updated.count(r"\begin{table}") == 1


def test_figure_replacement_by_latex_label():
    tex = r"""
\begin{figure}[!htbp]
\centering
\safeincludegraphics{results_reference_assets/ratio_shape_layout_reference.jpg}
\label{fig:ratio_shape_path_placeholder}
\end{figure}
\begin{figure}[!htbp]
\centering
\safeincludegraphics{results_reference_assets/ratio_shape_layout_reference.jpg}
\label{fig:other_metric_paths_placeholder}
\end{figure}
"""
    updated = pipe.replace_latex_environment_by_label(
        tex, "figure", "fig:ratio_shape_path_placeholder", r"\begin{figure}RATIO SHAPE\end{figure}"
    )
    assert "RATIO SHAPE" in updated
    assert "fig:other_metric_paths_placeholder" in updated
    assert updated.count("ratio_shape_layout_reference.jpg") == 1


def test_prb_mki_figure_is_r2_versus_metric():
    oos = pd.DataFrame(
        {
            "family": ["Direct"] * 4 + ["Surrogate"] * 4,
            "evaluation": ["heldout"] * 8,
            "rho": [0, 0.1, 1, 10] * 2,
            "R2_price": np.linspace(0.88, 0.84, 8),
            "PRB": np.linspace(-0.1, -0.02, 8),
            "MKI": np.linspace(1.2, 1.02, 8),
        }
    )
    with TemporaryDirectory() as td:
        pipe.FIG_OUT = Path(td)
        pdf = pipe.plot_accuracy_equity_r2(oos, ["PRB", "MKI"], "prb_mki_accuracy_equity")
        assert pdf.is_file() and pdf.stat().st_size > 0
        meta = pipe.load_json(Path(td) / "prb_mki_accuracy_equity.meta.json")
    assert meta["x"] == "R2_price"
    assert meta["vs_rho"] is False
    assert "PRB" in meta["y"] and "MKI" in meta["y"]


def test_final_qa_rejects_placeholder_phrases():
    live = "PLACEHOLDER populate from Reference layout only results_reference_assets after the 728-fit CV completes"
    for phrase in pipe.FORBIDDEN_PHRASES:
        assert phrase in live
