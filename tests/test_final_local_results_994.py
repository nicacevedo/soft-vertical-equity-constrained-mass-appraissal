"""Focused checks for split-specific rho=0 audit and b=1 recalibration identity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
FINAL = REPO / "output" / "paper_v6_preselection_994" / "final_local_results"
COMBINED = REPO / "output" / "paper_v6_preselection_994" / "analysis" / "combined_path_table.csv"


def test_rho0_audit_is_split_specific():
    audit = pd.read_csv(FINAL / "rho0_split_audit.csv")
    payload = json.loads((FINAL / "rho0_split_audit.json").read_text())
    assert payload["gate"] == "PASS_REPORT_CODE_PATH_DIFFERENCE"
    assert payload["positive_rho_paths_refit"] is False
    held = audit.loc[audit["split"] == "heldout"]
    fwd = audit.loc[audit["split"] == "forward_2025"]
    assert int(held.loc[held["model"] == "Direct rho=0", "n_aligned"].iloc[0]) == 38290
    assert int(fwd.loc[fwd["model"] == "Direct rho=0", "n_aligned"].iloc[0]) == 26641
    d_h = float(held.loc[held["model"] == "Direct rho=0", "mean_abs_delta_log"].iloc[0])
    d_f = float(fwd.loc[fwd["model"] == "Direct rho=0", "mean_abs_delta_log"].iloc[0])
    s_h = float(held.loc[held["model"] == "Surrogate rho=0", "mean_abs_delta_log"].iloc[0])
    s_f = float(fwd.loc[fwd["model"] == "Surrogate rho=0", "mean_abs_delta_log"].iloc[0])
    assert d_h == s_h
    assert d_f == s_f
    assert abs(d_h - d_f) > 1e-6
    assert d_h > 1e-8 and d_f > 1e-8


def test_b1_reproduces_native_lightgbm():
    spec = json.loads((FINAL / "recalibration_spec.json").read_text())
    path = pd.read_csv(FINAL / "recalibration_path.csv")
    comb = pd.read_csv(COMBINED)
    native = comb.loc[comb["family"] == "LightGBM"].iloc[0]
    assert spec["heldout_or_2025_used_to_choose_b"] is False
    assert spec["b_star_source"] == "validation-neutral"
    assert abs(float(spec["b_star_details"]["pooled_oof_beta_log_at_b_star"])) < 1e-10
    assert all(r["hashes_match_archive"] for r in spec["fold_training_means"])
    for ev, col in (("heldout", "heldout"), ("forward_2025", "forward_2025")):
        row = path.loc[(path["evaluation"] == ev) & (path["j"] == 0)].iloc[0]
        assert abs(float(row["b"]) - 1.0) < 1e-15
        assert abs(float(row["R2_price"]) - float(native[f"R2_price__{col}"])) < 1e-12
        assert abs(float(row["Beta_log"]) - float(native[f"Beta_log__{col}"])) < 1e-12
        assert abs(float(row["Delta_NL"]) - float(native[f"Delta_NL__{col}"])) < 1e-12
    star = path.loc[(path["evaluation"] == "heldout") & (path["j"] == 50)].iloc[0]
    assert np.isclose(float(star["b"]), float(spec["b_star"]))
    assert int(len(path)) == 102


def test_delta_nl_oos_coverage():
    dnl = pd.read_csv(FINAL / "delta_nl_oos.csv")
    comb = pd.read_csv(COMBINED)
    assert not comb["Delta_NL__heldout"].isna().any()
    assert not comb["Delta_NL__forward_2025"].isna().any()
    counts = dnl.groupby(["family", "evaluation"]).size()
    assert int(counts.loc[("Direct", "heldout")]) == 51
    assert int(counts.loc[("Surrogate", "forward_2025")]) == 51
    hashes = dnl.groupby("evaluation")["fold_assignment_hash"].nunique()
    assert (hashes == 1).all()
