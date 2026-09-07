#!/usr/bin/env python3
"""P1 Task 1: document the EXACT distance-correlation estimator actually executed.

Read-only.  Nothing is refit and no dCor value is recomputed or corrected: this
task establishes, from the installed library and the executed source, which
estimator produced every reported dCor_e_y number, and whether all reported path
metrics use the same one.

Closes the manuscript TODO at paper/paper_v17_option1.tex:3204-3206.
"""
from __future__ import annotations

import inspect
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p1_common as c                                                   # noqa: E402


def main() -> int:
    import dcor
    from dcor import _dcor as dcor_mod
    from utils import motivation_utils as mu

    rows, facts = [], {}

    # ---------------------------------------------------------------- library
    facts["library"] = "dcor"
    facts["version_installed"] = dcor.__version__
    facts["version_declared_requirements_txt"] = next(
        (l.strip() for l in (c.REPO / "requirements.txt").read_text().splitlines()
         if l.strip().lower().startswith("dcor")), None)
    facts["python"] = sys.version.split()[0]

    # ------------------------------------------------- the executed call site
    src = inspect.getsource(mu.distance_correlation_e_y)
    file_ = inspect.getsourcefile(mu.distance_correlation_e_y)
    line0 = inspect.getsourcelines(mu.distance_correlation_e_y)[1]
    facts["call_site"] = f"{Path(file_).relative_to(c.REPO)}:{line0}"
    facts["call_site_source"] = src
    m = re.search(r"dcor\.(\w+)\(([^)]*)\)", src)
    facts["function_called"] = f"dcor.{m.group(1)}"
    facts["kwargs_passed_at_call_site"] = m.group(2)
    facts["residual_convention"] = "e = y_pred_log - y_true_log"
    facts["arguments"] = "dcor.distance_correlation(e, y_true_log, ...)  ->  x=e, y=log P"
    facts["subsampling"] = "none - full evaluation sample"

    # ------------------------------------------ estimator class, from the lib
    sig = inspect.signature(dcor.distance_correlation)
    facts["signature"] = str(sig)
    facts["default_exponent"] = repr(sig.parameters["exponent"].default)
    facts["default_method"] = repr(sig.parameters["method"].default)
    facts["default_compile_mode"] = repr(sig.parameters["compile_mode"].default)
    facts["bias_corrected_passed"] = False
    inner = inspect.getsource(dcor_mod._distance_correlation_sqr_naive) \
        if hasattr(dcor_mod, "_distance_correlation_sqr_naive") else ""
    stats_sqr = inspect.getsource(dcor_mod.DistanceCovarianceMethod) \
        if hasattr(dcor_mod, "DistanceCovarianceMethod") else ""
    facts["library_docstring_first_line"] = (dcor.distance_correlation.__doc__ or "").strip().splitlines()[0]
    dsqr_doc = (dcor.distance_correlation_sqr.__doc__ or "").strip().splitlines()[0]
    facts["distance_correlation_sqr_docstring_first_line"] = dsqr_doc
    biased = ("biased" in facts["library_docstring_first_line"].lower()
              or "biased" in dsqr_doc.lower()
              or "Usual (biased)" in (dcor.distance_correlation.__doc__ or ""))
    facts["estimator_class"] = "BIASED / V-statistic (double-centered)" if biased else "UNRESOLVED"
    facts["unbiased_alternatives_exported_but_never_called"] = [
        "dcor.u_distance_correlation_sqr", "dcor.distance_correlation_af_inv", "dcor.u_centered"]

    # ------------- numerical proof: the executed call equals the biased V-stat
    rng = np.random.default_rng(20260907)
    x = rng.normal(size=400); y = 0.6 * x + rng.normal(size=400)
    a = np.abs(x[:, None] - x[None, :]); b = np.abs(y[:, None] - y[None, :])
    dc = lambda M: M - M.mean(axis=1, keepdims=True) - M.mean(axis=0, keepdims=True) + M.mean()
    A, B = dc(a), dc(b)
    v_stat = float(np.sqrt(np.mean(A * B) / np.sqrt(np.mean(A * A) * np.mean(B * B))))
    executed = float(dcor.distance_correlation(x, y, method="auto"))
    u_stat = float(np.sqrt(max(dcor.u_distance_correlation_sqr(x, y), 0.0)))
    facts["numerical_check"] = {
        "n": 400,
        "executed_dcor_distance_correlation_method_auto": executed,
        "hand_rolled_double_centered_V_statistic": v_stat,
        "abs_diff_vs_V_statistic": abs(executed - v_stat),
        "sqrt_U_centered_unbiased_statistic": u_stat,
        "abs_diff_vs_U_statistic": abs(executed - u_stat),
        "verdict": ("executed == double-centered V-statistic to float tolerance; "
                    "it is NOT the U-centered/unbiased statistic"),
    }
    if abs(executed - v_stat) > 1e-10:
        raise c.ProtocolViolation("executed dcor does not match the biased V-statistic")
    if abs(executed - u_stat) < 1e-6:
        raise c.ProtocolViolation("V- and U-statistics indistinguishable; check is not discriminating")

    # ------------------------------------------- consistency across call sites
    sites = [
        ("utils/motivation_utils.py", "distance_correlation_e_y",
         'dcor.distance_correlation(e, y_true_log, method="auto")',
         "dCor_e_y", "YES - this is the estimator behind every reported path metric",
         "biased V-statistic; method='auto' dispatches to the exact O(n log n) AVL algorithm"),
        ("scripts/run_paper_baseline.py", "_distance_correlation",
         'dcor.distance_correlation(x, y, method="mergesort")',
         "distance_correlation_log_residual_log_price", "no - separate baseline artifact",
         "same biased V-statistic; 'mergesort' is a different EXACT algorithm, not a different estimator"),
        ("quick_test_models.py", "_distance_correlation_sampled",
         "hand-rolled double-centering, subsampled",
         "dCor(r,logprice)_sampled", "no - never enters a paper table",
         "also a biased V-statistic, but SUBSAMPLED and a different quantity: dCor(ratio, log P), not dCor(e, y)"),
    ]
    for f, fn, call, col, feeds, note in sites:
        rows.append({"file": f, "function": fn, "call": call, "output_column": col,
                     "feeds_manuscript_path_tables": feeds, "estimator_class": facts["estimator_class"],
                     "note": note})

    facts["consistency_verdict"] = (
        "ALL reported path metrics use ONE estimator. dCor_e_y is produced only by "
        "utils/motivation_utils.distance_correlation_e_y, reached via paper_mechanism_metrics -> "
        "compute_taxation_metrics/_compute_extended_metrics from run_temporal_cv.py and every P0 "
        "script; the path tables are a pure merge of those shards with no recomputation. The two "
        "other call sites do not feed any manuscript table.")
    facts["benign_redundancy_noted_not_repaired"] = (
        "Three P0 scripts call _compute_extended_metrics (which already computes dCor_e_y) and then "
        "call paper_mechanism_metrics again, overwriting with the identical value: "
        "p0_4_centered_spread.py, p0_6_temporal_designs.py, p0_zero_control_assemble.py. "
        "dCor is therefore computed twice per cell. No effect on any value.")
    facts["manuscript_todo_closed"] = {
        "location": "paper/paper_v17_option1.tex:3204-3206",
        "answer": ("dcor v0.6, dcor.distance_correlation(e, y, method='auto'), exponent=1, "
                   "compile_mode=AUTO, bias_corrected=False -> the standard biased / V-statistic "
                   "(double-centered) estimator, computed on the full evaluation sample with "
                   "e = log(P_hat) - log(P) and y = log P."),
        "definition_to_match": "paper/paper_v17_option1.tex:1160-1168 (eq:dcor_diagnostic)"}

    c.write_table(pd.DataFrame(rows), c.TABLES / "dcor_estimator_audit.csv")
    c.write_json(c.TABLES / "dcor_estimator_facts.json", facts)
    print(json.dumps({k: facts[k] for k in
                      ("library", "version_installed", "call_site", "function_called",
                       "kwargs_passed_at_call_site", "default_exponent", "default_method",
                       "default_compile_mode", "estimator_class")}, indent=2))
    print("\nnumerical check:", json.dumps(facts["numerical_check"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
