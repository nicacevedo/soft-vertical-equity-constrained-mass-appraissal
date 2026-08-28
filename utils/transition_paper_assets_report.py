"""Markdown interpretation assembled from already-computed follow-up tables."""

from __future__ import annotations

from typing import Any

import pandas as pd

from utils.transition_paper_assets import classify_direct_interpretation, fmt_num
from utils.transition_regions import PRIMARY_METRICS


def build_interpretation(
    summary: pd.DataFrame,
    regret: pd.DataFrame,
    sharpness: pd.DataFrame,
    detail: pd.DataFrame,
    mech: pd.DataFrame,
) -> str:
    dsum = summary.loc[summary["family"] == "Direct"].iloc[0]
    ssum = summary.loc[summary["family"] == "Surrogate"].iloc[0]
    letter = classify_direct_interpretation(regret)
    letter_text = {
        "A": "location unstable but value robust",
        "B": "location and value both unstable",
        "C": "mixed by metric",
        "D": "insufficiently localized to justify a transition-region statement",
    }[letter]
    f = fmt_num
    lines = [
        "# Transition-region paper interpretation",
        "",
        "Descriptive follow-up on the frozen 994-tree CV transition result.",
        "No rho, family, or penalized configuration is selected or recommended.",
        "Exact event concordance is not treated as sufficient for out-of-time transfer.",
        "Span regret is reported without a materiality threshold.",
        "",
        "## Direct",
        "",
        "### 1. Exact five CV event rhos",
    ]
    for metric, _d in PRIMARY_METRICS:
        lines.append(
            f"- {metric}: rho={f(dsum[f'{metric}__cv_rho'])}, "
            f"class={dsum[f'{metric}__cv_classification']}, value={f(dsum[f'{metric}__cv_value'])}"
        )
    lines += ["", "### 2. Sharpness of each CV event", ""]
    for metric, _d in PRIMARY_METRICS:
        sh = sharpness.loc[
            (sharpness["family"] == "Direct") & (sharpness["split"] == "cv_mean") & (sharpness["metric"] == metric)
        ].iloc[0]
        lines.append(
            f"- {metric}: best-vs-second gap={f(sh['best_vs_second_gap'])}; "
            f"gap/range={f(sh['best_vs_second_gap_over_range'])}; "
            f"lower-neighbor gap={f(sh['lower_neighbor_gap'])}; "
            f"higher-neighbor gap={f(sh['higher_neighbor_gap'])}; "
            f"local_turn_verified={f(sh['local_turn_verified'])}"
        )
    lines += [
        "",
        f"### 3. Fraction of the positive log-rho grid spanned: {f(dsum['fraction_of_full_positive_log_grid'])}",
        f"log10_width={f(dsum['log10_width'])}; n_positive_rho_in_span={f(dsum['n_positive_rho_in_span'])}",
        "",
        f"### 4. Is 0.1 the smallest tested positive value? {f(dsum['lower_endpoint_equals_first_positive_grid'])}",
        "The lower endpoint equals the first tested positive rho. It is not an estimated lower threshold.",
        f"Upper endpoint equals last positive grid? {f(dsum['upper_endpoint_equals_last_positive_grid'])}",
        "",
        "### 5. Seven fold events",
        "",
    ]
    for metric, _d in PRIMARY_METRICS:
        part = sharpness.loc[
            (sharpness["family"] == "Direct")
            & (sharpness["metric"] == metric)
            & (sharpness["split"].astype(str).str.startswith("fold_"))
        ]
        rhos = ", ".join(f(x) for x in part["optimum_rho"].tolist())
        classes = ", ".join(str(x) for x in part["classification"].tolist())
        lines.append(f"- {metric} fold rhos: {rhos}")
        lines.append(f"  classes: {classes}")
    lines += [
        "",
        "### 6. LOFO span endpoint ranges (valid analyses only)",
        f"- valid: {int(dsum['lofo_valid_count'])} / 7",
        f"- lower endpoints [{f(dsum['lofo_valid_low_min'])}, {f(dsum['lofo_valid_low_max'])}]",
        f"- upper endpoints [{f(dsum['lofo_valid_high_min'])}, {f(dsum['lofo_valid_high_max'])}]",
        "This LOFO envelope is a sensitivity diagnostic and does not replace the frozen full-CV span.",
        "",
        "### 7. Held-out and 2025 events",
        f"- held-out exact concordance: {int(dsum['heldout_exact_concordance'])} / 5",
        f"- 2025 exact concordance: {int(dsum['forward_2025_exact_concordance'])} / 5",
        "",
    ]
    for metric, _d in PRIMARY_METRICS:
        row = detail.loc[(detail["family"] == "Direct") & (detail["metric"] == metric)].iloc[0]
        lines.append(
            f"- {metric}: held-out rho={f(row['heldout_rho'])} inside_cv={f(row['heldout_inside_cv_span'])} "
            f"log10_dist={f(row['heldout_log10_distance_from_span'])}; "
            f"2025 rho={f(row['forward_2025_rho'])} inside_cv={f(row['forward_2025_inside_cv_span'])} "
            f"log10_dist={f(row['forward_2025_log10_distance_from_span'])}; "
            f"held-out inside LOFO envelope={f(row.get('heldout_inside_lofo_endpoint_envelope'))}; "
            f"2025 inside LOFO envelope={f(row.get('forward_2025_inside_lofo_endpoint_envelope'))}"
        )
    lines += ["", "### 8. Out-of-time span regret (frozen Direct CV span only)", ""]
    for split in ("heldout", "forward_2025"):
        lines.append(f"**{split}**")
        part = regret.loc[regret["split"] == split]
        for _, row in part.iterrows():
            lines.append(
                f"- {row['metric']}: global_opt_rho={f(row['global_opt_rho'])}, "
                f"best_inside_rho={f(row['best_inside_rho'])}, "
                f"raw_regret={f(row['raw_regret'])}, "
                f"normalized_regret={f(row['normalized_regret'])}, "
                f"best_inside_rank={f(row['best_inside_ordinal_rank'])}, "
                f"path_range={f(row['path_range'])}"
            )
        lines.append("")
    lines += [
        "### 9. Exact concordance versus metric values",
        "",
        "No materiality threshold is applied. The letter below uses only whether raw regret is",
        "numerically zero or strictly positive for each Direct out-of-time primary metric.",
        f"Held-out exact concordance is {int(dsum['heldout_exact_concordance'])}/5 and 2025 is "
        f"{int(dsum['forward_2025_exact_concordance'])}/5.",
        "Zero regret means the CV span still contains a globally best grid value even if the exact turning rho moved.",
        "Positive regret means the span does not retain that out-of-time optimum value.",
        "",
        "### 10. First-order attenuation at span endpoints",
        "",
    ]
    for loc in ("span_low", "span_high"):
        part = mech.loc[(mech["family"] == "Direct") & (mech["location"] == loc)]
        if part.empty:
            continue
        r = part.iloc[0]
        lines.append(
            f"- {loc} rho={f(r['rho'])}: q_beta={f(r['q_beta'])}, "
            f"1-q_beta={f(r['attenuation_1_minus_q_beta'])}, "
            f"q_cov={f(r['q_cov'])}, q_beta/q_cov agree={f(r['q_beta_q_cov_agree'])}, "
            f"fold median q_beta={f(r['fold_median_q_beta'])}, fold IQR={f(r['fold_iqr_q_beta'])}, "
            f"held-out q_beta={f(r['heldout_q_beta'])}, 2025 q_beta={f(r['forward_2025_q_beta'])}, "
            f"overcorrection={f(r['overcorrection'])}"
        )
    lines += [
        "",
        "## Surrogate",
        "",
        "### 1. Exact five CV events",
    ]
    for metric, _d in PRIMARY_METRICS:
        lines.append(
            f"- {metric}: rho={f(ssum[f'{metric}__cv_rho'])}, "
            f"class={ssum[f'{metric}__cv_classification']}, value={f(ssum[f'{metric}__cv_value'])}"
        )
    sh0 = sharpness.loc[
        (sharpness["family"] == "Surrogate") & (sharpness["split"] == "cv_mean") & (sharpness["metric"] == "RMSE_log")
    ].iloc[0]
    lines += [
        "",
        "### 2-3. Best positive-rho RMSE_log versus rho=0",
        f"- RMSE_log(rho=0)={f(sh0.get('rmse_log_rho0'))}",
        f"- best positive RMSE_log={f(sh0.get('best_positive_rmse_log'))} at rho={f(sh0.get('best_positive_rho'))}",
        f"- best_positive_minus_zero={f(sh0.get('best_positive_minus_zero'))}",
        f"- that gap / full path range={f(sh0.get('best_positive_minus_zero_over_path_range'))}",
        "No near-optimality tolerance is applied; these are discrete-grid comparisons.",
        "",
        f"### 4. Why only {int(ssum['lofo_valid_count'])}/7 LOFO analyses support a common span",
        "The frozen full-CV Surrogate status is FULL_COMMON_SPAN_NOT_SUPPORTED because RMSE_log is a rho=0 boundary event.",
        "LOFO uses the same five-metric interior-span rule on six-fold means; five of seven replications remain blocked by RMSE_log.",
        "",
        "### 5. Sharpness of the remaining Surrogate CV events",
        "",
    ]
    for metric, _d in PRIMARY_METRICS:
        sh = sharpness.loc[
            (sharpness["family"] == "Surrogate") & (sharpness["split"] == "cv_mean") & (sharpness["metric"] == metric)
        ].iloc[0]
        lines.append(
            f"- {metric}: class={sh['classification']}; best-vs-second gap={f(sh['best_vs_second_gap'])}; "
            f"gap/range={f(sh['best_vs_second_gap_over_range'])}"
        )
    lines += [
        "",
        "### 6. Held-out and 2025 event locations (descriptive)",
        "There is no frozen Surrogate CV span, so exact concordance against a common span is 0/5 by construction.",
        "",
    ]
    for metric, _d in PRIMARY_METRICS:
        row = detail.loc[(detail["family"] == "Surrogate") & (detail["metric"] == metric)].iloc[0]
        lines.append(
            f"- {metric}: held-out rho={f(row['heldout_rho'])} class={row['heldout_classification']}; "
            f"2025 rho={f(row['forward_2025_rho'])} class={row['forward_2025_classification']}"
        )
    lines += [
        "",
        "## Overall interpretation letter",
        "",
        f"Direct out-of-time span-regret pattern: **{letter} — {letter_text}.**",
        "This letter uses only the sign of raw regret (zero vs strictly positive) across Direct held-out and 2025",
        "primary metrics. It is not a deployment recommendation and does not redefine the frozen CV span.",
        "Surrogate remains a negative/partial common-span result under the frozen protocol.",
        "The lower Direct endpoint is the first tested positive rho, so the span is left-censored by the grid.",
        "",
    ]
    return "\n".join(lines) + "\n"
