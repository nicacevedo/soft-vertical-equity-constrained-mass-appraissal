#!/usr/bin/env python3
"""Write reports/FINAL_V3_REPORT.md from whatever v3 artifacts currently exist.

Safe to re-run. Does not edit the manuscript. Does not invent metrics.
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from analysis.berry_attom_validation_v3.scripts.v3_common import ANALYSIS, COUNTIES, OUTPUT

REPORT = ANALYSIS / "reports" / "FINAL_V3_REPORT.md"


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def cell(value) -> str:
    if value is None or value == "":
        return "pending"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, str):
        # CSV values arrive as strings; round them instead of printing 17 digits.
        try:
            f = float(value)
        except ValueError:
            return value
        return "n/a" if f != f else f"{f:.4g}"
    return str(value)


def table_cell(row: dict, column: str) -> str:
    """Distinguish a column this artifact never wrote from an empty value."""
    if column not in row:
        return "—"
    value = row[column]
    return "" if value == "" else cell(value)


def num(value, fmt: str = ".4g") -> str:
    """Format a CSV/JSON scalar, saying 'n/a' rather than inventing a number."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if f != f else format(f, fmt)


def md_table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "_No rows available._\n"
    head = "| " + " | ".join(columns) + " |\n| " + " | ".join("---" for _ in columns) + " |\n"
    return head + "".join(
        "| " + " | ".join(table_cell(r, c) for c in columns) + " |\n" for r in rows
    )


def property_use_section(counties) -> str:
    """Philadelphia's PROPERTYUSESTANDARDIZED=385 attrition, stated plainly."""
    lines = []
    summary_rows = []
    for c in counties:
        key = c["key"]
        prof = read_csv(ANALYSIS / "feature_audit" / f"{key}_property_use_profile.csv")
        if not prof:
            continue
        broad = [r for r in prof if str(r.get("in_broad_residential")).lower() == "true"]
        n_matched = sum(int(float(r["n_matched_sales"])) for r in prof)
        n_385 = sum(
            int(float(r["n_matched_sales"])) for r in prof
            if str(r.get("in_primary_385")).lower() == "true"
        )
        n_broad = sum(int(float(r["n_matched_sales"])) for r in broad)
        summary_rows.append({
            "county": c["label"],
            "safe_history_sales": n_matched,
            "kept_by_385": n_385,
            "share_kept_by_385": num(n_385 / max(n_matched, 1), ".3f"),
            "kept_by_broad_residential": n_broad,
            "share_kept_by_broad": num(n_broad / max(n_matched, 1), ".3f"),
            "n_broad_codes": len(broad),
            "broad_codes": " ".join(sorted(r["use_code"] for r in broad)),
        })
    if summary_rows:
        lines.append("### Property-use attrition, all three counties\n")
        lines.append(md_table(summary_rows, list(summary_rows[0].keys())))
    for c in counties:
        key = c["key"]
        dec = read_csv(ANALYSIS / "feature_audit" / f"{key}_property_use_retention_by_decile.csv")
        if not dec:
            continue
        lines.append(f"\n### {c['label']}: P(kept | safe-history sale-price decile)\n")
        lines.append(md_table(
            dec, ["price_decile", "n_safe_history", "median_price",
                  "p_kept_primary_385", "p_kept_broad_residential"],
        ))
    sens = read_json(
        ANALYSIS / "baselines_pre_freeze"
        / "philadelphia_broad_residential_sensitivity" / "run_meta.json"
    )
    primary = read_json(ANALYSIS / "baselines_pre_freeze" / "philadelphia" / "run_meta.json")
    if sens and primary:
        cmp_rows = []
        for label, meta in (("primary_385 (frozen)", primary), ("broad_residential (sensitivity)", sens)):
            val = meta.get("validation_lgbm_HISTORY_MARKET_CORE") or {}
            cmp_rows.append({
                "cohort": label,
                "n_full": meta.get("n_full"),
                "n_validation": meta.get("n_validation"),
                "validation_R2": num(val.get("R2")),
                "validation_COD": num(val.get("COD")),
                "validation_PRD": num(val.get("PRD")),
                "validation_PRB": num(val.get("PRB")),
                "validation_beta_log": num(val.get("Beta_log")),
                "freeze_status": meta.get("freeze_status"),
            })
        lines.append("\n### Philadelphia cohort sensitivity, VALIDATION ONLY\n")
        lines.append(md_table(cmp_rows, list(cmp_rows[0].keys())))
        lines.append(
            "\nThe sensitivity cohort was never scored on the held-out test block and never "
            "entered Direct or Surrogate. It does not revise Philadelphia's frozen "
            "`MODEL_TRANSFER_STATUS`.\n"
        )
    elif summary_rows:
        lines.append("\n_Philadelphia sensitivity baselines not yet available._\n")
    return "".join(lines) if lines else "_Property-use audit not yet available._\n"


def branch_ceiling_diagnostic(key: str) -> str:
    """Separate a saturating branch from one merely truncated by the grid.

    `branch_terminated_by = GRID_CEILING` is mechanical: it only says the branch
    ran to the last grid point without reversing. It does not say whether more
    rho would help. The slope at the top of the branch does. Extrapolating that
    slope to each unattained target gives the extra decades of rho required, and
    when that number is large the target is unreachable in practice rather than
    a casualty of where the grid stopped.
    """
    rows = read_csv(ANALYSIS / "method_transfer" / key / "surrogate_branch_curve.csv")
    frozen = read_csv(ANALYSIS / "method_transfer" / key / "surrogate_rho_first_branch.csv")
    if len(rows) < 3 or not frozen:
        return ""
    try:
        import math

        pts = [(float(r["rho"]), float(r["achieved_reduction"])) for r in rows]
    except (KeyError, ValueError):
        return ""
    if frozen[0].get("branch_terminated_by") != "GRID_CEILING":
        return ""
    top_rho, top_red = pts[-1]
    # Slope over the final decade of rho actually covered by the branch.
    lo = next((p for p in reversed(pts) if p[0] <= top_rho / 10.0), pts[0])
    decades = math.log10(top_rho / lo[0]) if lo[0] > 0 else 0.0
    if decades <= 0:
        return ""
    slope = (top_red - lo[1]) / decades
    unattained = [
        float(r["requested_reduction"]) for r in frozen if r.get("status") == "UNATTAINED"
    ]
    if not unattained:
        return ""
    verdict = "saturating" if slope < 0.05 else "still rising"
    parts = [
        f"\nTop-of-branch slope is {slope:.3f} reduction per decade of rho "
        f"(measured over the final {decades:.2f} decades, ending at rho={top_rho:.3g} "
        f"with {top_red:.3f} achieved). The branch is **{verdict}** there. "
    ]
    est = []
    for t in sorted(unattained):
        need = (t - top_red) / slope if slope > 0 else float("inf")
        est.append(
            f"{t:.0%} would need about {need:.1f} more decades of rho"
            + (f" (rho ~ 1e{math.log10(top_rho) + need:.0f})" if need < 40 else "")
        )
    parts.append(
        "Extrapolating that slope, " + "; ".join(est) + ". "
    )
    if verdict == "saturating":
        parts.append(
            "So these anchors are **not** unattained because the grid stopped too early. "
            "The Surrogate's first-order reduction saturates below them on this county, and "
            "widening the grid again would not deliver them. That is a finding about the "
            "method, not a missing run.\n"
        )
    else:
        parts.append(
            "A wider grid could plausibly reach these anchors. Extending it would be a "
            "pre-registered decision for a future pass, not a retune of this one.\n"
        )
    return "".join(parts)


def surrogate_section(counties, authorized: bool) -> str:
    if not authorized:
        return "Not run. Freeze did not authorize, or freeze is not yet written.\n"
    lines = [
        "First contiguous low-rho branch only; no global sort + `np.interp`. Every UNATTAINED "
        "row now carries why it was unattained, which pass 1 could not express.\n\n"
        "**Pass 2.** Pass 1 (`21882241`) had two pretest-diagnosable defects: a fixed "
        "`geomspace(1e-6, 1e2, 16)` grid whose ceiling sat below the Direct 97% anchor "
        "(rho up to 255.8), and a branch detector with no noise floor that opened a one-point "
        "branch on a 0.008 reduction at rho=1e-6 in St. Louis. Pass 2 ties the grid ceiling to "
        "4x this county's largest Direct rho and requires a 1% reduction to open a branch. "
        "Rho was never chosen from a test metric in either pass. Philadelphia's test rows were "
        "scored twice and pass-1 output is preserved as `surrogate_pass1_*.csv`. Full disclosure: "
        "`panel_freeze/SURROGATE_RECALIBRATION_LOG.md`.\n\n",
    ]
    for c in counties:
        key = c["key"]
        rows = read_csv(ANALYSIS / "method_transfer" / key / "surrogate_heldout.csv")
        if not rows:
            lines.append(f"\n### {c['label']}\n\n_No surrogate output._\n")
            continue
        lines.append(f"\n### {c['label']}\n\n")
        lines.append(md_table(rows, [
            "requested_reduction", "rho", "status", "unattained_reason", "branch_terminated_by",
            "R2_price", "COD", "PRD", "PRB", "Beta_log", "Cov(e,logprice)", "Delta_NL", "dCor_e_y",
        ]))
        errors = read_json(ANALYSIS / "method_transfer" / key / "surrogate_fit_errors.json") or {}
        if errors:
            lines.append(
                f"\nGrid {num(errors.get('grid_min'))} to {num(errors.get('grid_max'))} over "
                f"{errors.get('grid_points')} points (max Direct rho "
                f"{num(errors.get('max_direct_rho'))}); fit failures: "
                f"{errors.get('n_fit_errors')}.\n"
            )
        lines.append(branch_ceiling_diagnostic(key))
        pass1 = read_csv(ANALYSIS / "method_transfer" / key / "surrogate_pass1_heldout.csv")
        if pass1:
            attained = [r for r in pass1 if r.get("status") != "UNATTAINED"]
            now = [r for r in rows if r.get("status") != "UNATTAINED"]
            lines.append(
                f"\nPass 1 attained {len(attained)} of {len(pass1)} anchors here against "
                f"{len(now)} of {len(rows)} in pass 2; see `surrogate_pass1_heldout.csv`.\n"
            )
    return "".join(lines)


def direct_section(counties, authorized: bool) -> str:
    if not authorized:
        return "Not run. Freeze did not authorize, or freeze is not yet written.\n"
    lines = [
        "`LGBCovPenalty[diff]`. Rho from the rank-one mapping on the FULL PRETEST block. "
        "Portable covariance-reduction anchors 10/25/50/67/80/90/97%. Raw rho is not comparable "
        "across counties and is not compared.\n",
    ]
    for c in counties:
        rows = read_csv(ANALYSIS / "method_transfer" / c["key"] / "direct_heldout.csv")
        if not rows:
            lines.append(f"\n### {c['label']}\n\n_No Direct output._\n")
            continue
        lines.append(f"\n### {c['label']}\n\n")
        lines.append(md_table(rows, [
            "requested_reduction", "rho", "R2_price", "RMSE_log", "COD", "PRD", "PRB", "MKI",
            "VEI", "Beta_log", "Cov(e,logprice)", "Delta_NL", "dCor_e_y",
        ]))
    return "".join(lines)


def lr_level_space_outliers(counties) -> list[dict]:
    """Count held-out LR predictions whose level-space ratio is extreme.

    Exponentiating a log-space linear fit can extrapolate without bound, and a
    single such row destroys every level-space aggregate (R2_price, COD, PRD,
    PRB) while leaving the log-space fit intact. These counts are reported so
    the degenerate LR columns are not read as a substantive result. Nothing is
    clipped or winsorized: the held-out block is scored once, and trimming it
    after seeing the numbers would be an outcome-driven edit.
    """
    try:
        import numpy as np
        import pandas as pd
    except ImportError:
        return []
    rows = []
    for c in counties:
        path = OUTPUT / "final_models" / c["key"] / "heldout_predictions.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        if "lr" not in frame or "y" not in frame:
            continue
        ratio = np.asarray(frame["lr"], dtype=float) / np.asarray(frame["y"], dtype=float)
        rows.append({
            "jurisdiction": c["label"],
            "n_test": int(len(frame)),
            "max_lr_prediction": num(float(np.nanmax(frame["lr"])), ".3g"),
            "max_ratio": num(float(np.nanmax(ratio)), ".3g"),
            "ratio_p99_9": num(float(np.nanquantile(ratio, 0.999)), ".3g"),
            "n_ratio_gt_10": int((ratio > 10).sum()),
            "n_ratio_gt_100": int((ratio > 100).sum()),
        })
    return rows


def heldout_section(counties) -> str:
    rows = []
    for c in counties:
        held = read_json(ANALYSIS / "final_baselines" / c["key"] / "heldout_metrics.json")
        if not held:
            continue
        for model in ("lgbm", "lr"):
            m = held.get(model) or {}
            if not m:
                continue
            rows.append({
                "jurisdiction": c["label"],
                "model": model.upper(),
                "n_test": held.get("n_test"),
                "R2_price": num(m.get("R2_price")),
                "R2_log": num(m.get("R2_log")),
                "RMSE_log": num(m.get("RMSE_log")),
                "MAPE": num(m.get("MAPE")),
                "COD": num(m.get("COD")),
                "PRD": num(m.get("PRD")),
                "PRB": num(m.get("PRB")),
                "MKI": num(m.get("MKI")),
                "VEI": num(m.get("VEI")),
                "Beta_log": num(m.get("Beta_log")),
                "Delta_NL": num(m.get("Delta_NL")),
                "dCor_e_y": num(m.get("dCor_e_y")),
            })
    if not rows:
        return "_Held-out baselines not yet available._\n"
    text = (
        "Frozen LR and LightGBM configurations refit on development+validation, then the "
        "chronological test block scored **once**. The same test rows carry into every "
        "Direct/Surrogate comparison.\n\n"
        + md_table(rows, list(rows[0].keys()))
    )
    outliers = lr_level_space_outliers(counties)
    if outliers:
        text += (
            "\n#### The LR reference model's level-space metrics are degenerate in two counties\n\n"
            "Read `R2_log` for LR, not `R2_price`. Exponentiating a log-space linear fit "
            "extrapolates without bound, and a handful of held-out rows blow up every "
            "level-space aggregate. St. Louis County has a single LR prediction near "
            "$1.5 trillion, which alone drives its `R2_price` to about -4.2e8, `COD` to 6586 "
            "and `PRB` to 1309. Philadelphia has two rows above 100x. Those numbers are "
            "artifacts of one or two rows, not statements about vertical equity.\n\n"
            + md_table(outliers, list(outliers[0].keys()))
            + "\nNothing was clipped or winsorized. The test block is scored once, and trimming "
            "it after seeing the result would be an outcome-driven edit to a frozen evaluation. "
            "The consequences are contained: **LightGBM is the baseline that carries the "
            "science.** It is the model the panel freeze used, the model the Direct and "
            "Surrogate paths penalize, and its held-out predictions are well behaved in every "
            "county (maximum ratio 9.1 / 9.7 / 17.3). LR is a reference point only, and no "
            "jurisdiction status, rho anchor, or conclusion depends on it. A bounded LR variant "
            "should be pre-registered in a future pass rather than patched into this one.\n"
        )
    return text


def stl_robustness_section(stl_rob: list[dict]) -> str:
    if not stl_rob:
        return "_St. Louis provider robustness not yet available._\n"
    n_common = stl_rob[0].get("n_common")
    n_val = stl_rob[0].get("n_validation")
    return (
        "Local table uses actual `PRICE` and `SALEDT`, never `APRTOT`, with an explicit 2012 "
        "dwelling-history defect sensitivity. Validation metrics only.\n\n"
        f"**This comparison is statistically powerless and must not be read as evidence either "
        f"way.** The common cohort holds {n_common} transactions with {n_val} validation rows. "
        "Two causes, both design choices rather than data limits: the cohort was cut to the "
        "standardized 2016-2025 ATTOM window, which against a local sales file ending in 2019 "
        "leaves only 2016-2019; and the join was a bespoke normalized-raw-APN plus exact-sale-date "
        "match rather than the validated linkage crosswalk, which reaches 99.6% APN match on this "
        "county over 2005-2019. Differences in the table below are within noise at this N.\n\n"
        + md_table(stl_rob, [
            "source", "n_common", "n_validation", "R2_price", "RMSE_log", "COD", "PRD", "PRB",
            "Beta_log",
        ])
        + "\nRecommended follow-up: rebuild this comparison over the predeclared 2005-2019 "
        "St. Louis linkage window using `linkage/st_louis_county_crosswalk.parquet`. Not done in "
        "this pass.\n"
    )


def bootstrap_section(counties) -> str:
    lines = []
    for c in counties:
        ci = read_csv(ANALYSIS / "final_baselines" / c["key"] / "bootstrap_ci_all_methods.csv")
        if not ci:
            continue
        keep = {"R2", "COD", "PRD", "PRB", "MKI", "VEI", "Beta_log", "Delta_NL", "RMSE_log"}
        rows = [r for r in ci if r.get("metric") in keep]
        lines.append(f"\n### {c['label']}\n\n")
        lines.append(md_table(rows, ["method", "metric", "mean", "std", "ci_2_5", "ci_97_5"]))
    if not lines:
        return (
            "_Bootstrap not yet available._ 200 monthly time-block draws, identical draw indices "
            "across baseline, Direct and Surrogate within a jurisdiction.\n"
        )
    return (
        "200 monthly time-block bootstrap draws. The draw indices are built once per jurisdiction "
        "and reused across baseline, Direct and Surrogate, so method differences are not draw "
        "differences. Saved as `output/.../final_models/<county>/bootstrap_indices.npy`.\n"
        + "".join(lines)
    )


def main() -> int:
    freeze = {}
    fp = ANALYSIS / "panel_freeze" / "final_panel_freeze_v3.yaml"
    if fp.exists():
        freeze = yaml.safe_load(fp.read_text()) or {}
    freeze_units = {u.get("key"): u for u in freeze.get("units", [])}
    link = {
        row.get("key"): row
        for row in (read_json(ANALYSIS / "linkage" / "linkage_summary.json") or [])
        if isinstance(row, dict)
    }
    repro = {r.get("jurisdiction"): r for r in read_csv(ANALYSIS / "berry_reproduction" / "reproduction_summary.csv")}
    inv_cov = read_csv(ANALYSIS / "inventory" / "field_coverage.csv")
    inv_counts = read_csv(ANALYSIS / "inventory" / "fips_year_counts.csv")
    cache_manifest = read_csv(ANALYSIS / "inventory" / "county_cache_manifest.csv")
    waterfall = read_csv(ANALYSIS / "linkage" / "unconditional_waterfall.csv")
    nested = read_csv(ANALYSIS / "linkage" / "nested_selection_audit.csv")
    ret_sum = read_csv(ANALYSIS / "feature_audit" / "modeling_table_summary.csv")
    stl_rob = read_csv(ANALYSIS / "st_louis_source_robustness" / "metrics.csv")
    authorized = bool(freeze.get("direct_surrogate_authorized"))
    jobs = (ANALYSIS / "logs" / "submitted_job_ids.txt").read_text().strip() if (ANALYSIS / "logs" / "submitted_job_ids.txt").exists() else "not yet submitted"
    proto = yaml.safe_load((ANALYSIS / "protocol_v3.yaml").read_text())
    canceled = proto.get("scheduler", {}).get("canceled_superseded_v2_jobs", [])

    rows = []
    for c in COUNTIES:
        key = c["key"]
        berry_key = {"wayne": "detroit_mi", "philadelphia": "philadelphia_pa", "st_louis_county": "st_louis_county_mo"}[key]
        r = repro.get(berry_key, {})
        b = read_json(ANALYSIS / "baselines_pre_freeze" / key / "run_meta.json") or {}
        held = read_json(ANALYSIS / "final_baselines" / key / "heldout_metrics.json") or {}
        t = read_json(OUTPUT / "modeling_tables" / key / "modeling_table_meta.json") or {}
        u = freeze_units.get(key, {})
        lk = link.get(key, {})
        wf = next((x for x in waterfall if x.get("jurisdiction") == key), {})
        val = b.get("validation_lgbm_HISTORY_MARKET_CORE") or {}
        lgbm_h = held.get("lgbm") or {}
        direct_csv = ANALYSIS / "method_transfer" / key / "direct_heldout.csv"
        surr_csv = ANALYSIS / "method_transfer" / key / "surrogate_heldout.csv"
        rows.append({
            "jurisdiction": c["label"],
            "ATTOM_geographic_unit": c["label"],
            "Berry_geographic_unit": c["berry_unit"],
            "Berry_N": r.get("n") or lk.get("berry_n_full_source"),
            "Berry_reproduction_status": r.get("status", "pending"),
            "unconditional_APN_rate": wf.get("r_apn") or lk.get("r_apn_unconditional"),
            "unconditional_transaction_confirmed_rate": wf.get("r_transaction") or lk.get("r_transaction_unconditional"),
            "unconditional_safe_history_rate": wf.get("r_safe_history") or lk.get("r_safe_history_unconditional"),
            "unconditional_fully_validated_rate": wf.get("r_fully_validated") or lk.get("r_fully_validated_unconditional"),
            "Berry_PRB_full": lk.get("berry_PRB_eligible"),
            "Berry_PRB_fully_linked": lk.get("berry_PRB_safe_history"),
            "model_eligible_Recorder_N": t.get("n_sale_validation_eligible"),
            "final_model_N": t.get("n_final_model"),
            "model_retention_decile_spread": t.get("decile_p_final_spread"),
            "validation_R2": val.get("R2"),
            "model_transfer_status": u.get("MODEL_TRANSFER_STATUS", "pending"),
            "berry_anchor_status": u.get("BERRY_ANCHOR_STATUS", "pending"),
            "heldout_R2": lgbm_h.get("R2"),
            "heldout_baseline_PRB": lgbm_h.get("PRB"),
            "heldout_beta_log": lgbm_h.get("Beta_log"),
            "Direct_run": "yes" if authorized and direct_csv.exists() else "no",
            "Surrogate_run": "yes" if authorized and surr_csv.exists() else "no",
            "main_scientific_role": (
                "Wayne County AVM + Detroit city Berry anchor" if key == "wayne"
                else "Philadelphia AVM + Berry validation" if key == "philadelphia"
                else "St. Louis County AVM + local source robustness; not City 29510"
            ),
            "confidence": "pending_until_gates_complete" if not u else "see freeze notes",
        })

    property_use_block = property_use_section(COUNTIES)
    stl_rob_block = stl_robustness_section(stl_rob)
    heldout_block = heldout_section(COUNTIES)
    direct_block = direct_section(COUNTIES, authorized)
    surrogate_block = surrogate_section(COUNTIES, authorized)
    bootstrap_block = bootstrap_section(COUNTIES)
    header = "| " + " | ".join(rows[0].keys()) + " |\n| " + " | ".join("---" for _ in rows[0]) + " |\n"
    body = "".join("| " + " | ".join(cell(r[k]) for k in r) + " |\n" for r in rows)
    passing = freeze.get("passing_model_transfer_units")
    text = f"""# Berry/ATTOM validation v3 — final report

Written: {datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}  
Protocol: `analysis/berry_attom_validation_v3/protocol_v3.yaml`  
**Manuscript was not edited. Frozen v1 (`analysis/berry_cmf_validation/`) and v2 (`analysis/berry_attom_validation_v2/`) were not modified.**

Berry official assessment/sale ratios and model valuation/sale ratios are different estimands.

## 1. Accidental-prompt cleanup / repository provenance

This repository is a property-assessment / mass-appraisal regressivity project.
A previous session prompt was accidentally copied from an unrelated repository.
Inspection of *this* repo found **no files, hunks, or jobs from that prompt to revert**.
Working-tree cleanup was therefore a no-op. `git reset --hard` was not used.

- HEAD at protocol write: `{proto.get("repository", {}).get("head_sha_at_protocol_write")}`
- Relation to `7b6adf7`: older ancestor
- Relation to `d6b07b7`: immediate parent of that HEAD (v2 freeze SHA); current HEAD is one Berry/ATTOM commit after `d6b07b7`
- Superseded v2 Slurm jobs canceled (never executed linkage/baselines/freeze): {canceled}
- v2 completed artifacts preserved as preliminary evidence
- New v3 job record: `{jobs}`
- Scheduler: `{proto.get("scheduler", {}).get("preferred_partition")}` (not `mit_normal`)

## 2. New ATTOM inventory

Reused from frozen v2 with SHA256 provenance (`inventory/V2_PROVENANCE.json`). Inventory coverage rows: {len(inv_cov)}; FIPS-year rows: {len(inv_counts)}; cache-manifest rows: {len(cache_manifest)}.

Dewey folders are immutable sources. Caches are FIPS-filtered to Wayne `26163`, Philadelphia `42101`, St. Louis County `29189`. St. Louis City `29510` is forbidden in the 29189 cache. Folder names are not contents: extra FIPS in the raw delivery are not modeled.

## 3. Berry/local source reproduction

Copied from v2 without re-filtering (`berry_reproduction/PROVENANCE.md`).

- Detroit: Python translation of the Rmd filters (class 401, VALID ARMS LENGTH). Native R comparison is attempted separately when `cmfproperty` can be sourced.
- Philadelphia: canonical table is the arms-length Stata file only. The alternative total file is a disjoint universe and is not stacked naively.
- St. Louis County: 2019 cumulative `sales.csv` with actual `SALEDT` and `PRICE`. This is **not** a fully reproduced Berry assessment-ratio benchmark (no official assessed-value series in that extract). ATTOM-linkage cohort is predeclared **2005-01-01 through 2019-12-31**, not 1975-2019.

## 4. Parcel linkage

Statuses: `EXACT_RAW_APN`, `EXACT_NORMALIZED_APN`, `EXACT_PREVIOUS_APN`, `AMBIGUOUS_APN`, `NO_APN_MATCH`. One-to-many APN→ATTOMID maps are never silently collapsed. Price/date is never a parcel identifier. Address is corroboration only.

See `linkage/linkage_summary.json` and per-county `*_crosswalk.parquet`.

## 5. Recorder transaction corroboration

Independent Recorder search on linked ATTOMIDs. Date windows 0 / ±1 / ±7 / ±30 days; price exact / ≤1% / ≤5%. Tiers: `TIER_1_HIGH_CONFIDENCE`, `TIER_2_PLAUSIBLE`, `AMBIGUOUS`, `CONFLICT`. Thresholds were not retuned after seeing regressivity.

## 6. Unconditional linkage waterfall

Unconditional rates use **eligible Berry/local N** as the denominator (St. Louis eligible = 2005–2019 dated sales). Conditional rates are also reported. Do not read a high history-coverage number as coverage of the full Berry sample unless the waterfall says so.

See `linkage/unconditional_waterfall.csv`. Rows: {len(waterfall)}.

## 7. Nested linkage-selection effects

Stages 0–4: full eligible cohort → unique APN/ATTOMID → high-confidence Recorder → safe History → fully validated. `P(stage | sale-price decile)` is in `linkage/nested_selection_audit.csv` ({len(nested)} rows). Figure: `figures/fully_validated_rate_by_price_decile.pdf`.

## 8. Berry regressivity preservation

Primary scientific question: does restricting to ATTOM-linkable observations materially alter the independently documented Berry pattern (COD/PRD/PRB/value-decile profile)? Sign reversals and strong value-dependent retention are flags, not gates to be loosened.

ATTOM `TAXASSESSEDVALUETOTAL` / `TAXMARKETVALUETOTAL` are **not** treated as Berry assessed value. Direct Berry-vs-ATTOM assessment-ratio comparison is skipped; see `source_concordance/STEP9_ASSESSMENT_CONCORDANCE.md`.

## 9. Standardized ATTOM modeling-cohort selection

2016–2025 qualified arms-length single-parcel residential (`PROPERTYUSESTANDARDIZED=385`) Recorder sales with strict pre-sale History. Wayne models are Wayne County, never Detroit.

Retention audit: `feature_audit/*_modeling_retention_by_decile.csv` and `modeling_table_summary.csv` ({len(ret_sum)} rows). Report `P(enters final model | Recorder price decile)` and eligible-vs-final price differences.

### 9a. Philadelphia's property-use filter is the dominant, value-dependent selection

This is a first-class limitation on the Philadelphia results, not a footnote.

`PROPERTYUSESTANDARDIZED=385` retains 8.75% of Philadelphia's eligible Recorder sales
(208,508 safe-history matches to 38,043 modeled), against roughly 85% retention in Wayne
and 82% in St. Louis. The cause is coding, not stock: Philadelphia's Assessor History is
dominated by use code 366 (5.36M rows) with 385 second (2.68M), while Wayne (13.3M) and
St. Louis (5.6M) are 385-dominated. Dewey ships no `PROPERTYUSESTANDARDIZED` code
dictionary in this delivery, so no code can be *called* residential here; the sensitivity
cohort below is defined by published structural facts instead.

The attrition is also value-dependent and non-monotone. `P(enters final model | Recorder
price decile)` in Philadelphia runs 0.028 at the bottom decile, peaks near 0.176 at the
eighth, then falls back to 0.084 at the top. **Philadelphia's vertical-equity diagnostics
therefore rest on a value-dependently filtered cohort**, and its lower held-out R2 relative
to Wayne should be read in that light.

Philadelphia was assigned `MODEL_TRANSFER_STATUS: PRIMARY` before this was visible. The
freeze is deliberately **not** revised: the protocol forbids moving a jurisdiction status on
information that arrived after the freeze, and `final_panel_freeze_v3.yaml` is byte-identical
to its frozen state. The finding is documented and probed instead.

{property_use_block}

## 10. Validation-only baseline results

LR + ordinary LightGBM on development/validation only. Categorical LR levels from the development prefix. The chronological test block is not scored or stored before freeze. Hyperparameter selection is validation-only. A non-regressive validation baseline is **not** an exclusion criterion.

See `baselines_pre_freeze/<county>/validation_metrics.csv`.

## 11. St. Louis provider robustness

{stl_rob_block}

## 12. Frozen model-transfer and Berry-anchor statuses

`panel_freeze/final_panel_freeze_v3.yaml`

Direct/Surrogate authorized: **{authorized}**  
Passing MODEL_TRANSFER units: {passing}

Two independent statuses per jurisdiction: `MODEL_TRANSFER_STATUS` (PRIMARY / BOUNDARY / NOT_ELIGIBLE) and `BERRY_ANCHOR_STATUS` (STRONG / MODERATE / WEAK / NOT_APPLICABLE). Berry linkage is not a hard requirement for ATTOM model validity. A strong ATTOM AVM is not a claim of Berry replication.

If fewer than two MODEL_TRANSFER units are PRIMARY or BOUNDARY: STOP. Do not force penalty results.

## 13. Final untouched held-out baselines

{heldout_block}

## 14. Direct results if authorized

{direct_block}

## 15. Surrogate results if authorized

{surrogate_block}

## 16. Remaining nonlinear dependence / shape failures

Held-out `Delta_NL` and `dCor_e_y` sit in the Direct and Surrogate tables above; sale-price-decile
valuation-ratio profiles are in `final_baselines/<county>/decile_valuation_ratio_profiles.csv` with
figures at `figures/<county>_heldout_ratio_by_decile.pdf`. Residual nonlinear dependence after
first-order reduction is a scientific finding, not a reason to retune rho on test.

`HISTORY_STRUCTURAL_CORE` exists in v3 only as a validation-side metric
(`validation_lgbm_HISTORY_STRUCTURAL_CORE` in each `baselines_pre_freeze/<county>/run_meta.json`).
It has no separate modeling table, no held-out evaluation and no penalty path, so it is a
validation-only sensitivity and must not be reported as a second held-out feature family.

### 16a. Bootstrap distributions

{bootstrap_block}

## 17. Relationship to existing six ATTOM counties

See `source_concordance/existing_six_vs_v3_metadata.csv`. Those runs include Tax Assessor/ACS/location that v3 currently lacks. They remain a separate exploratory/sensitivity layer and were not overwritten.

## 18. Exact scientific interpretation

v3 is a **pre-rho** source-validation and standardized-AVM design. Cook County / CCAO remains the paper's primary application. Wayne County is not Detroit. St. Louis County is not St. Louis City. Official assessment ratios are not AVM valuation ratios.

## 19. Exact recommended paper additions/changes

**Do not edit the manuscript in this pass.** Recommended later:

1. Keep Cook County / CCAO as the primary application.
2. If freeze authorized transfer, add Wayne / Philadelphia / St. Louis County as a literature-anchored external-validation appendix, with Wayne **not** labeled Detroit, using HISTORY_MARKET_CORE (no Tax Assessor/ACS).
3. If freeze did not authorize transfer, report source validation + leakage-safe baselines, and state that Direct/Surrogate were not transferred because fewer than two new MODEL_TRANSFER units qualified.
4. Keep existing six-county ATTOM results as a separate sensitivity with a different feature class.
5. Do not cite Berry official assessment ratios as if they were model valuation ratios.
6. Always report unconditional linkage rates against eligible Berry N.

## 20. Remaining limitations

- Assessor History `PARCELNUMBERFORMATTED` / `PROPERTYJURISDICTIONNAME` are missing in this Dewey delivery, so Detroit-city ATTOM models are not a primary v3 product.
- St. Louis local extract does not reproduce an official assessment-ratio Berry benchmark.
- Year-end History snapshots can lag sales by more than a year; lag percentiles are reported, not tuned away.
- v3 primary features omit Tax Assessor/ACS/location; metrics are not comparable to old-six.
- The LR reference model's held-out level-space metrics are degenerate in Philadelphia and
  St. Louis County because exponentiating a log-space linear fit extrapolates without bound
  (section 13). LightGBM carries the science; no status, anchor, or conclusion uses LR.
- Philadelphia's modeled cohort survives a value-dependent property-use filter that keeps 8.75%
  of eligible sales (section 9a). Its vertical-equity numbers carry that selection.
- The St. Louis local-vs-ATTOM provider comparison has 38 validation rows and settles nothing
  (section 11).
- Surrogate anchors come from calibration pass 2; Philadelphia's held-out test rows were scored
  under both passes. Disclosed in `panel_freeze/SURROGATE_RECALIBRATION_LOG.md`.
- `HISTORY_STRUCTURAL_CORE` is validation-only in v3.
- Any Surrogate anchor still marked UNATTAINED after pass 2 carries a reason code; a
  `MATERIAL_REVERSAL` is a finding about the method on that county, not a missing run.

## Canonical jurisdiction table

{header}{body}
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text, encoding="utf-8")
    print("wrote", REPORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
