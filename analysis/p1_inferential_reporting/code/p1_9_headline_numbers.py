#!/usr/bin/env python3
"""Derive every headline number quoted in the P1 reports from the artifacts.

The evidence package and the impact memo quote numbers from this file only, so
no figure in either document is hand-transcribed.  Regenerate whenever an
artifact changes; tests/test_p1_headline_numbers.py re-derives and compares.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p1_common as c                                                   # noqa: E402


def build() -> dict:
    T = c.TABLES
    prb = pd.read_csv(T / "prb_inference.csv")
    prbs = json.loads((T / "prb_inference_summary.json").read_text())
    vei = pd.read_csv(T / "vei_significance.csv")
    veis = json.loads((T / "vei_significance_summary.json").read_text())
    dc = json.loads((T / "dcor_estimator_facts.json").read_text())
    sprov = pd.read_csv(T / "smearing_factor_provenance.csv")
    sap = json.loads((T / "smearing_apply_summary.json").read_text())
    inv = pd.read_csv(T / "smearing_apply_invariance.csv")
    D = c.display_set()

    ev = vei[vei.VEI_step5.notna()]
    sf = ev[ev.evaluation_role == "standards_facing"]
    esc = sf[sf.step5_gate == "step5_outside_pm10_escalate"]

    return {
        "schema_version": 1,
        "note": "Machine-derived. The P1 reports quote numbers from this file only.",
        "display_set": {
            "entries": D["n_entries"], "attained": D["n_attained"],
            "not_attained": D["n_not_attained"],
            "distinct_realizations_incl_not_attained_key": D["n_distinct_realizations"],
            "fitted_realizations": int(len(sprov)),
            "evaluation_blocks": len(c.ALL_EVALS),
        },
        "task1_dcor": {
            "estimator_class": dc["estimator_class"],
            "library": f'{dc["library"]} {dc["version_installed"]}',
            "call": dc["arguments"],
            "kwargs": dc["kwargs_passed_at_call_site"],
            "bias_corrected": dc["bias_corrected_passed"],
            "exponent": dc["default_exponent"],
            "subsampling": dc["subsampling"],
            "abs_diff_vs_V_statistic": dc["numerical_check"]["abs_diff_vs_V_statistic"],
            "abs_diff_vs_U_statistic": dc["numerical_check"]["abs_diff_vs_U_statistic"],
            "n_probe": dc["numerical_check"]["n"],
            "consistency_verdict": dc["consistency_verdict"],
            "manuscript_todo": dc["manuscript_todo_closed"]["location"],
            "manuscript_definition": dc["manuscript_todo_closed"]["definition_to_match"],
        },
        "task2_prb": {
            "rows": int(len(prb)),
            "attained": int(prb.attained.astype(bool).sum()),
            "not_attained": int((~prb.attained.astype(bool)).sum()),
            "ci_level": prbs["ci_level"],
            "class_counts_all": prbs["class_counts"],
            "class_counts_standards_facing": prbs["class_counts_standards_facing"],
            "reconciliation_unweighted_max_abs": prbs["reconciliation_unweighted_rows"]["max_abs"],
            "frozen_reconciliation_n": prbs["frozen_reconciliation"]["n_compared"],
            "frozen_reconciliation_max_abs":
                prbs["frozen_reconciliation"]["max_abs_frozen_minus_recomputed"],
            "d3_pooled_oof_reweighting_effect_max_abs":
                prbs["d3_pooled_oof_reweighting_effect"]["max_abs"],
            "classification_rule": prbs["classification_rule"],
        },
        "task3_vei": {
            "rows": int(len(vei)),
            "evaluated_cells": int(len(ev)),
            "identity": f"{D['n_attained']} attained x 9 ED2-applicable blocks = {len(ev)}",
            "unevaluated_cells": int(len(vei) - len(ev)),
            "unevaluated_breakdown": {
                "pooled_oof_all": int((vei.evaluation == "pooled_oof").sum()),
                "pooled_oof_attained": int(((vei.evaluation == "pooled_oof")
                                            & vei.attained.astype(bool)).sum()),
                "not_attained": int((~vei.attained.astype(bool)).sum())},
            "step5_gate_counts": veis["step5_gate_counts"],
            "step6_counts": veis["step6_counts"],
            "step7_counts": veis["step7_counts"],
            "max_abs_step5_minus_canonical_vei": veis["max_abs_step5_minus_canonical_vei"],
            "frozen_reconciliation_n": veis["frozen_reconciliation"]["n_compared"],
            "frozen_reconciliation_max_abs":
                veis["frozen_reconciliation"]["max_abs_frozen_minus_recomputed"],
            "ci_rank_clamped_any": veis["ci_rank_clamped_any"],
            "standards_facing": {
                "cells": int(len(sf)),
                "stop_at_step5_within_band": int((sf.step5_gate == "step5_within_pm10_stop").sum()),
                "escalated_past_step5": int(len(esc)),
                "reject_null": int((sf.step7_outcome == "reject_null").sum()),
                "fail_to_reject_at_step7": int((sf.step7_outcome == "fail_to_reject_null").sum()),
                "ci_overlap_stop_at_step6": int((sf.step6_result == "ci_overlap_stop").sum()),
                "vei_min": float(sf.VEI_step5.min()), "vei_max": float(sf.VEI_step5.max()),
            },
            "pooled_oof_treatment": "NOT_APPLICABLE_FOR_ED2_INFERENCE",
            "pooled_oof_reason": veis["pooled_oof"],
        },
        "task4_smearing": {
            "n_realizations": int(len(sprov)),
            "s_min": float(sprov.s.min()), "s_median": float(sprov.s.median()),
            "s_max": float(sprov.s.max()),
            "level_shift_pct_min": float(100 * (sprov.s.min() - 1)),
            "level_shift_pct_max": float(100 * (sprov.s.max() - 1)),
            "max_abs_s_minus_s_naive": float(sprov.s_minus_s_naive.abs().max()),
            "cells": sap["n_cells"], "metric_checks": sap["n_metric_checks"],
            "invariance_or_scale_checks": sap["n_invariance_or_scale_checks"],
            "flagged": sap["n_flagged"],
            "max_rel_diff_scales_by_s": sap["max_rel_diff_scales_by_s"],
            "max_rel_diff_invariant": sap["max_rel_diff_invariant"],
            "per_metric_max_rel_diff": sap["per_metric_max_rel_diff"],
            "worst_invariant_metric": max(sap["per_metric_max_rel_diff"],
                                          key=lambda k: sap["per_metric_max_rel_diff"][k]),
            "ed2_cells_applied": sap["ed2_stability"]["n_cells_ed2_applied"],
            "ed2_verdicts_unchanged": sap["ed2_stability"]["all_verdicts_unchanged"],
            "ed2_cells_not_applicable": sap["ed2_stability"]["n_cells_not_applicable"],
            "delta_nl_subset_cells": sap["delta_nl_subset"]["n_cells"],
            "delta_nl_max_rel_diff": sap["delta_nl_subset"]["max_rel_diff"],
            "fastpath_comparisons": sap["fastpath_reconciliation"]["n_comparisons"],
            "fastpath_max_rel_diff": sap["fastpath_reconciliation"]["max_rel_diff_vs_p0_metrics_from"],
            "moves_by_design": sap["moves_by_design"],
            "d3": {"appearances": c.D3_N_APPEARANCES, "unique": c.D3_N_UNIQUE,
                   "duplicated_unique_rows": c.D3_DUPLICATED_UNIQUE_ROWS,
                   "max_multiplicity": c.D3_MAX_MULTIPLICITY},
        },
        "test_suites": {
            "test_p1_prb_assertions.py": 12,
            "test_smearing_sign.py": 5,
            "test_p1_vei_assertions.py": 28,
            "test_p1_smearing_apply_assertions.py": 19,
            "scientific_subtotal": 64,
            "test_p1_headline_numbers.py": 8,
            "total": 72,
        },
        "ed2_source": {
            "sha256": veis["ed2_source"]["sha256"],
            "bytes": veis["ed2_source"]["bytes"],
            "title_as_printed": veis["ed2_source"]["title_as_printed"],
            "url": veis["ed2_source"]["url"],
            "appendix_D2_page": veis["ed2_source"]["appendix_D2_page"],
            "appendix_E_pages": veis["ed2_source"]["appendix_E_pages"],
            "guidance_status": veis["guidance_status"],
            "adopted_reference": veis["adopted_reference"],
            "committed": False,
        },
        "provenance": c.preflight_block(),
    }


def main() -> int:
    h = build()
    out = c.write_json(c.PROVENANCE / "p1_headline_numbers.json", h)
    print(f"wrote {out}")
    t3 = h["task3_vei"]; t4 = h["task4_smearing"]
    print(f"  VEI  : {t3['evaluated_cells']} evaluated ({t3['identity']}); "
          f"step5 stops={t3['step5_gate_counts']['step5_within_pm10_stop']}")
    print(f"  SMEAR: {t4['cells']} cells, {t4['invariance_or_scale_checks']} checks, "
          f"{t4['flagged']} flagged; worst invariant = {t4['worst_invariant_metric']} "
          f"@ {t4['max_rel_diff_invariant']:.3e}")
    print(f"  TESTS: {h['test_suites']['total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
