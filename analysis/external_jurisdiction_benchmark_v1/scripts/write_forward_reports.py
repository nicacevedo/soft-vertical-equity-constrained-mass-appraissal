#!/usr/bin/env python3
"""Paper-oriented findings and the full 2025 results report.

Labels are applied only after a fixed rubric. Numbers remain the evidence.
Does not write a 2025 candidate region.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from analysis.external_jurisdiction_benchmark_v1.scripts.forward_common import (  # noqa: E402
    ANALYSIS, DIRECT_COMMON_INTERVAL, SURROGATE_INTERSECTION_STATUS,
)
from analysis.external_jurisdiction_benchmark_v1.scripts.v1_common import ALL_KEYS, JURISDICTION_BY_KEY  # noqa: E402

REP = ANALYSIS / "forward_2025" / "reports"
TABLES = ANALYSIS / "forward_2025" / "tables"
METRICS = ANALYSIS / "forward_2025" / "metrics"
BOOT = ANALYSIS / "forward_2025" / "bootstrap"
AUDITS = ANALYSIS / "forward_2025" / "audits"

RUBRIC = """
FORWARD_SUPPORTS_CV: 2025 equity correction has the same sign as CV at activity
and at the 25% A_beta anchor; 50% A_beta remains attained or still reduces
|beta|; no reversal of PRB/beta toward worse regressivity; Delta_NMSE has the
same sign as CV (or remains near zero).
FORWARD_PARTIAL: some frozen anchors transfer, others do not, with no wholesale
reversal of the correction mechanism.
FORWARD_WEAKENS_CV: correction remains same-signed but is materially smaller, or
predictive cost is materially larger, at the frozen activity/25%/50% coordinates.
FORWARD_REVERSAL: at a frozen activity or 25%/50% anchor, 2025 PRB or beta_log
moves away from the ideal relative to that split's own baseline.
""".strip()


def _row(df, key, family, role):
    hit = df.loc[(df.county_key == key) & (df.family == family) & (df.role == role)]
    return None if not len(hit) else hit.iloc[0]


def _first_row(df, key, family, *roles):
    for role in roles:
        row = _row(df, key, family, role)
        if row is not None:
            return row
    return None


def classify(anc: pd.DataFrame, key: str, family: str) -> str:
    base = _first_row(anc, key, family, "baseline_rho0", "native_lgbm_baseline")
    act = _row(anc, key, family, "activity")
    a25 = _row(anc, key, family, "A_beta_0.25")
    a50 = _row(anc, key, family, "A_beta_0.5")
    if base is None:
        return "FORWARD_PARTIAL"
    b0 = base.get("beta_log_2025")
    def worse(row):
        if row is None or pd.isna(row.get("beta_log_2025")) or pd.isna(b0):
            return False
        return abs(row["beta_log_2025"]) > abs(b0) + 1e-4
    if worse(act) or worse(a25):
        return "FORWARD_REVERSAL"
    def transfers(row):
        if row is None or pd.isna(row.get("A_beta_2025")):
            return False
        return float(row["A_beta_2025"]) >= 0.10
    n_ok = sum(transfers(r) for r in (act, a25, a50))
    if transfers(act) and transfers(a25) and (a50 is None or pd.isna(a50.get("rho_tilde")) or transfers(a50)):
        # cost sign vs CV
        if a25 is not None and pd.notna(a25.get("Delta_NMSE_2025")) and pd.notna(a25.get("Delta_NMSE_cv_mean")):
            c25, ccv = float(a25["Delta_NMSE_2025"]), float(a25["Delta_NMSE_cv_mean"])
            if abs(c25) > max(0.05, 2.5 * abs(ccv)) and abs(ccv) > 1e-4:
                return "FORWARD_WEAKENS_CV"
        return "FORWARD_SUPPORTS_CV"
    if n_ok == 0:
        return "FORWARD_WEAKENS_CV"
    return "FORWARD_PARTIAL"


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "NA"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def main() -> int:
    anc = pd.read_csv(METRICS / "forward_anchor_metrics.csv")
    base = pd.read_csv(TABLES / "forward_baseline_summary.csv")
    cand = pd.read_csv(TABLES / "forward_candidate_validation.csv")
    band = pd.read_csv(TABLES / "direct_common_band_forward.csv")
    comp = pd.read_csv(AUDITS / "forward_fit_completeness.csv")
    mech = pd.read_csv(TABLES / "mechanism_anchor_forward.csv")
    boot = pd.read_csv(BOOT / "forward_anchor_bootstrap_ci.csv") if (BOOT / "forward_anchor_bootstrap_ci.csv").exists() else pd.DataFrame()

    labels = {(k, f): classify(anc, k, f) for k in ALL_KEYS for f in ("direct", "surrogate")}
    n_rev = sum(v == "FORWARD_REVERSAL" for v in labels.values())
    n_sup = sum(v == "FORWARD_SUPPORTS_CV" for v in labels.values())
    n_partial = sum(v == "FORWARD_PARTIAL" for v in labels.values())
    n_weak = sum(v == "FORWARD_WEAKENS_CV" for v in labels.values())

    lines = []
    lines.append("# Frozen 2025 forward evaluation")
    lines.append("")
    lines.append(f"Written at {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}.")
    lines.append("Candidate regions, baselines, grids, and the forward freeze were not modified from 2025 outcomes.")
    lines.append("")
    lines.append("## Evaluation layers")
    lines.append("")
    lines.append("No independent pre-2025 TEST split exists. Layers: `CV_FOLD`, `CV_OOF` (fold-mean of frozen 2018–2024 metrics; **not** an independent test), `FORWARD_2025`.")
    lines.append("")
    lines.append("## Rubric")
    lines.append("")
    lines.append("```")
    lines.append(RUBRIC)
    lines.append("```")
    lines.append("")
    lines.append("## Sample sizes")
    lines.append("")
    lines.append("| jurisdiction | n_train | n_eval_2025 | Var_train(log price) |")
    lines.append("|---|---:|---:|---:|")
    for key in ALL_KEYS:
        row = comp.loc[(comp.county_key == key) & (comp.family == "direct")]
        if not len(row):
            continue
        r = row.iloc[0]
        lines.append(f"| {key} | {int(r.n_train)} | {int(r.n_eval)} | {fmt(r.Var_train_y, 4)} |")
    lines.append("")
    lines.append("## Baseline 2025 metrics (rho = 0)")
    lines.append("")
    lines.append("| jurisdiction | family | R2_price | NMSE | PRB | beta_log | PRD | MKI | VEI |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for key in ALL_KEYS:
        for family in ("direct", "surrogate"):
            r = _first_row(anc, key, family, "baseline_rho0", "native_lgbm_baseline")
            if r is None:
                continue
            lines.append(
                f"| {key} | {family} | {fmt(r.R2_price_2025)} | {fmt(r.NMSE_2025)} | {fmt(r.PRB_2025)} | "
                f"{fmt(r.beta_log_2025)} | {fmt(r.PRD_2025)} | {fmt(r.MKI_2025)} | {fmt(r.VEI_2025)} |"
            )
    lines.append("")
    lines.append("## Jurisdiction × family findings")
    lines.append("")
    for key in ALL_KEYS:
        lines.append(f"### {JURISDICTION_BY_KEY[key]['label']} (`{key}`)")
        for family in ("direct", "surrogate"):
            lab = labels[(key, family)]
            r0 = _first_row(anc, key, family, "baseline_rho0", "native_lgbm_baseline")
            act = _row(anc, key, family, "activity")
            a25 = _row(anc, key, family, "A_beta_0.25")
            a50 = _row(anc, key, family, "A_beta_0.5")
            grd = _row(anc, key, family, "guardrail")
            lines.append(f"**{family}** — `{lab}`")
            lines.append("")
            lines.append(
                f"- baseline 2025 beta_log={fmt(None if r0 is None else r0.beta_log_2025)} "
                f"PRB={fmt(None if r0 is None else r0.PRB_2025)}"
            )
            if act is not None:
                lines.append(
                    f"- activity ρ̃={fmt(act.rho_tilde, 4)} A_beta_2025={fmt(act.A_beta_2025)} "
                    f"Delta_NMSE_2025={fmt(act.Delta_NMSE_2025)} (CV {fmt(act.Delta_NMSE_cv_mean)})"
                )
            if grd is not None:
                lines.append(
                    f"- guardrail ρ̃={fmt(grd.rho_tilde, 4)} A_beta_2025={fmt(grd.A_beta_2025)} "
                    f"Delta_NMSE_2025={fmt(grd.Delta_NMSE_2025)}"
                )
            if a25 is not None:
                lines.append(f"- 25% A_beta ρ̃={fmt(a25.rho_tilde, 4)} A_beta_2025={fmt(a25.A_beta_2025)} Delta_NMSE={fmt(a25.Delta_NMSE_2025)}")
            if a50 is not None:
                lines.append(f"- 50% A_beta ρ̃={fmt(a50.rho_tilde, 4)} A_beta_2025={fmt(a50.A_beta_2025)} Delta_NMSE={fmt(a50.Delta_NMSE_2025)}")
            lines.append("")

    # Direct common band
    stable = band.loc[band.in_protocol_valid_set == True]  # noqa: E712
    n_j = stable["county_key"].nunique()
    both_ok = 0
    fail_j = []
    for key, g in stable.groupby("county_key"):
        ok = bool(g["practically_useful_2025"].all()) if "practically_useful_2025" in g.columns else False
        if ok:
            both_ok += 1
        else:
            fail_j.append(key)
    lines.append("## Direct common-band forward test")
    lines.append("")
    lines.append(f"Frozen interval `[{DIRECT_COMMON_INTERVAL[0]:.6f}, {DIRECT_COMMON_INTERVAL[1]:.6f}]` across 8 protocol-valid Direct jurisdictions (Allegheny held out as sensitivity).")
    lines.append(f"Both endpoints practically useful in 2025 for **{both_ok}/{n_j}** jurisdictions.")
    if fail_j:
        lines.append(f"Failures/non-useful: {', '.join(fail_j)}.")
    lines.append("The interval was **not** redefined from 2025.")
    lines.append("")
    lines.append("## Surrogate forward test")
    lines.append("")
    lines.append(f"Preserved CV conclusion: `{SURROGATE_INTERSECTION_STATUS}`. No 2025 search for a replacement universal band.")
    lines.append("")
    lines.append("## Bootstrap (paired monthly-block, 200 draws)")
    lines.append("")
    if boot.empty:
        lines.append("Bootstrap CI file not yet written. Intervals will appear after `run_forward_bootstrap.py` finishes.")
    else:
        lines.append("Significance is never used to move a frozen coordinate. Percentile intervals are for frozen anchors only.")
        lines.append("")
        lines.append("| jurisdiction | family | role | Delta_NMSE mean [2.5, 97.5] | Delta_beta_log mean [2.5, 97.5] | excludes 0 (NMSE / beta) |")
        lines.append("|---|---|---|---|---|---|")
        roles_keep = {"activity", "A_beta_0.25", "A_beta_0.5", "direct_common_lo", "direct_common_hi", "guardrail"}
        boot_show = boot.loc[boot.role.isin(roles_keep)].copy()
        for key in ALL_KEYS:
            for family in ("direct", "surrogate"):
                sub = boot_show.loc[(boot_show.county_key == key) & (boot_show.family == family)]
                for _, r in sub.iterrows():
                    nm = f"{fmt(r.Delta_NMSE_mean, 4)} [{fmt(r.Delta_NMSE_p025, 4)}, {fmt(r.Delta_NMSE_p975, 4)}]"
                    bt = f"{fmt(r.Delta_beta_log_mean, 4)} [{fmt(r.Delta_beta_log_p025, 4)}, {fmt(r.Delta_beta_log_p975, 4)}]"
                    lines.append(
                        f"| {key} | {family} | {r.role} | {nm} | {bt} | "
                        f"{r.Delta_NMSE_excludes_zero} / {r.Delta_beta_log_excludes_zero} |"
                    )
    lines.append("")

    # Q&A
    still_reg = []
    for key in ALL_KEYS:
        for family in ("direct", "surrogate"):
            r0 = _first_row(anc, key, family, "baseline_rho0", "native_lgbm_baseline")
            if r0 is not None and pd.notna(r0.beta_log_2025) and r0.beta_log_2025 < -0.02:
                still_reg.append(f"{key}/{family}")
    lines.append("## Answers to the predeclared scientific questions")
    lines.append("")
    lines.append(f"1. Are all nine 2025 baseline AVMs still vertically regressive? **{len(still_reg)}/18** family-jurisdiction baselines have beta_log < -0.02: {', '.join(still_reg) if still_reg else 'none'}.")
    lines.append("2. Does normalized activity onset transfer? See activity rows above; judged per jurisdiction rather than forced.")
    lines.append("3. Lack of portable upper guardrails: Surrogate remains without a nondegenerate all-jurisdiction intersection; Direct guardrails still span a wide range. Not redefined.")
    lines.append(f"4. Frozen Direct common interval transfer: {both_ok}/{n_j} protocol-valid jurisdictions keep both endpoints practically useful.")
    lines.append("5. Direct vs Surrogate frontier: see `figures/paper/accuracy_mechanism_frontier_cv_vs_2025.pdf`.")
    lines.append("6. Surrogate early limitations: Middlesex CV ordering and Miami-Dade early upper bound are tested, not rewritten.")
    lines.append("7. 25%/50% A_beta: see `mechanism_anchor_forward.csv`.")
    lines.append("8. Temporal stability: see `cv_to_2025_path_drift.pdf` and anchor 2025-minus-CV columns.")
    lines.append("9. Ratio profiles: see `figures/ratio_profiles/` and `forward_ratio_profile_examples.pdf`.")
    lines.append("10. Berry/local vs AVM: official-assessment and AVM ratios remain separate constructs; see `berry_local_vs_avm_ratio_profiles.pdf`. Wayne is not Detroit.")
    lines.append(
        f"11. 2025 vs CV portability claims: SUPPORTS={n_sup} PARTIAL={n_partial} "
        f"WEAKENS={n_weak} REVERSAL={n_rev} of 18 family-jurisdiction pairs. Numbers in the tables dominate the labels."
    )
    lines.append("")
    lines.append("No 2025-derived candidate region was written.")
    REP.mkdir(parents=True, exist_ok=True)
    (REP / "FORWARD_2025_RESULTS.md").write_text("\n".join(lines) + "\n")

    paper = [
        "# Paper-oriented findings (frozen 2025 forward pass)",
        "",
        "This note is interpretive. The canonical numbers are the CSVs under `forward_2025/`.",
        "",
        "```",
        RUBRIC,
        "```",
        "",
        f"Direct common band frozen at [{DIRECT_COMMON_INTERVAL[0]:.3f}, {DIRECT_COMMON_INTERVAL[1]:.3f}].",
        f"Surrogate all-jurisdiction intersection remains `{SURROGATE_INTERSECTION_STATUS}`.",
        "",
        f"Family-jurisdiction rubric: SUPPORTS={n_sup}, PARTIAL={n_partial}, WEAKENS={n_weak}, REVERSAL={n_rev}.",
        "",
        "Predeclared paper figures (counties were not chosen from 2025 outcomes):",
        "",
        "1. `figures/paper/accuracy_mechanism_frontier_cv_vs_2025.pdf`",
        "2. `figures/paper/forward_key_metric_paths_9jurisdictions.pdf`",
        "3. `figures/paper/forward_ratio_profile_examples.pdf` (Philadelphia, St. Louis County, Middlesex — chosen from pre-2025 roles).",
        "",
        "Do not cherry-pick 2025 counties for the main figures.",
        "",
        "Level shifts that matter for interpretation, not for re-estimating regions:",
        "",
        "- Maricopa and Middlesex 2025 price-R2 sit well below their CV fold-means; the Direct correction still moves beta_log toward 0 at frozen activity/25%/50% coordinates.",
        "- Cook and Maricopa 2025 baselines are more regressive than CV; activity still reduces |beta| at near-zero Delta_NMSE.",
        "- Middlesex Direct 50% A_beta has a wild interpolated CV-mean R2 near the diverged tail; use the finite 2025 metric, not the interpolated CV R2.",
        "- Wayne is not Detroit. Official-assessment and AVM ratio profiles remain separate constructs.",
        "",
        "Do not write a 2025-derived candidate region. Allegheny Direct remains `NO_STABLE_CANDIDATE_REGION`. Surrogate remains `NO_NONDEGENERATE_ALL_JURISDICTION_INTERSECTION`.",
    ]
    (REP / "PAPER_ORIENTED_FINDINGS.md").write_text("\n".join(paper) + "\n")

    tex = ANALYSIS / "paper_integration" / "FORWARD_2025_SECTION_DRAFT.tex"
    tex.parent.mkdir(parents=True, exist_ok=True)
    tex.write_text(
        "% Draft only. Do not edit paper/paper_v12.tex from this pass.\n"
        "% Frozen Direct common interval: "
        f"[{DIRECT_COMMON_INTERVAL[0]:.6f}, {DIRECT_COMMON_INTERVAL[1]:.6f}].\n"
        f"% Surrogate intersection status: {SURROGATE_INTERSECTION_STATUS}.\n"
        "% Insert numbers from analysis/external_jurisdiction_benchmark_v1/forward_2025/.\n"
    )
    print(json.dumps({
        "supports": n_sup, "partial": n_partial, "weakens": n_weak, "reversal": n_rev,
        "still_regressive": len(still_reg), "bootstrap_rows": int(len(boot)),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
