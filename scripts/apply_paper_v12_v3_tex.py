#!/usr/bin/env python3
"""Apply tracked v3 manuscript edits. No TeX compilation. No model fitting."""
from __future__ import annotations

import collections
import os
import re
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "paper" / "paper_v12.tex"
IMG = REPO / "paper" / "img" / "generated_v12_994"
sys.path.insert(0, str(REPO / "scripts"))
from promote_paper_v12_v3 import (  # noqa: E402
    METRIC_TEX,
    V3,
    fmt_logd,
    fmt_norm,
    fmt_raw,
    fmt_rho,
    tex_from_display,
)

PRIMARY_ROWS = [
    ("heldout", "Direct", 0.1, r"Direct & $\approx0.1$ & 0.894 & \$76,235 & \textit{source} & \textit{source} & 1.068 & -0.088 & 0.926 & -27.0\% \\"),
    ("heldout", "Direct", 1, r"Direct & $\approx0.954$ & \textbf{0.899} & \textbf{\$74,485} & \textit{source} & \textit{source} & 1.060 & -0.075 & 0.940 & -21.9\% \\"),
    ("heldout", "Direct", 10, r"Direct & $\approx10.481$ & \textbf{0.895} & \textbf{\$75,218} & \textit{source} & \textit{source} & \textbf{1.036} & -0.040\textsuperscript{*} & \textbf{0.984} & \textbf{-10.5\%} \\"),
    ("heldout", "Direct", 100, r"Direct & 100 & 0.869 & \$84,918 & \textit{source} & \textit{source} & \textbf{1.029} & \textbf{-0.015} & \textbf{0.998} & \textbf{0.7\%} \\"),
    ("heldout", "Surrogate", 0.1, r"Surrogate & $\approx0.1$ & 0.894 & \$75,897 & \textit{source} & \textit{source} & 1.064 & -0.079 & 0.931 & -22.7\% \\"),
    ("heldout", "Surrogate", 1, r"Surrogate & $\approx0.954$ & \textbf{0.897} & \textbf{\$75,468} & \textit{source} & \textit{source} & 1.049 & -0.045\textsuperscript{*} & 0.953\textsuperscript{*} & -12.6\% \\"),
    ("heldout", "Surrogate", 10, r"Surrogate & $\approx10.481$ & 0.893 & \$78,110 & \textit{source} & \textit{source} & \textbf{1.024} & \textbf{0.013} & \textbf{0.988} & \textbf{0.2\%} \\"),
    ("heldout", "Surrogate", 100, r"Surrogate & 100 & 0.889 & \$82,094 & \textit{source} & \textit{source} & \textbf{1.010} & 0.042\textsuperscript{*} & \textbf{1.013} & \textbf{4.0\%} \\"),
    ("forward_2025", "Direct", 0.1, r"Direct & $\approx0.1$ & \textbf{0.906} & \textbf{\$78,029} & \textit{source} & \textit{source} & 1.077 & -0.102 & 0.910 & -27.7\% \\"),
    ("forward_2025", "Direct", 1, r"Direct & $\approx0.954$ & \textbf{0.910} & \textbf{\$77,139} & \textit{source} & \textit{source} & 1.069 & -0.090 & 0.926 & -23.9\% \\"),
    ("forward_2025", "Direct", 10, r"Direct & $\approx10.481$ & 0.902 & \$79,187 & \textit{source} & \textit{source} & \textbf{1.042} & -0.049\textsuperscript{*} & \textbf{0.971} & \textbf{-11.9\%} \\"),
    ("forward_2025", "Direct", 100, r"Direct & 100 & 0.873 & \$89,309 & \textit{source} & \textit{source} & \textbf{1.036} & -0.029\textsuperscript{*} & \textbf{0.983} & \textbf{-4.5\%} \\"),
    ("forward_2025", "Surrogate", 0.1, r"Surrogate & $\approx0.1$ & \textbf{0.905} & \$78,546 & \textit{source} & \textit{source} & 1.074 & -0.095 & 0.914 & -24.7\% \\"),
    ("forward_2025", "Surrogate", 1, r"Surrogate & $\approx0.954$ & \textbf{0.908} & \textbf{\$78,196} & \textit{source} & \textit{source} & 1.060 & -0.062 & 0.932 & \textbf{-15.3\%} \\"),
    ("forward_2025", "Surrogate", 10, r"Surrogate & $\approx10.481$ & \textbf{0.905} & \$81,075 & \textit{source} & \textit{source} & \textbf{1.033} & \textbf{0.000} & \textbf{0.967} & \textbf{-2.0\%} \\"),
    ("forward_2025", "Surrogate", 100, r"Surrogate & 100 & 0.900 & \$85,257 & \textit{source} & \textit{source} & \textbf{1.020} & 0.030\textsuperscript{*} & \textbf{0.987} & \textbf{1.7\%} \\"),
]

COMP_ROWS = [
    ("heldout", "Direct", 0.1, r"Direct & $\approx0.1$ & \textit{source} & \textit{source} & \textit{source} & \textit{source} & \textit{source} & -0.147 & 0.119 & 0.382 \\"),
    ("heldout", "Direct", 1, r"Direct & $\approx0.954$ & \textit{source} & \textit{source} & \textit{source} & \textit{source} & \textit{source} & -0.134 & 0.122 & 0.359 \\"),
    ("heldout", "Direct", 10, r"Direct & $\approx10.481$ & \textit{source} & \textit{source} & \textit{source} & \textit{source} & \textit{source} & -0.096 & 0.132 & 0.302 \\"),
    ("heldout", "Direct", 100, r"Direct & 100 & \textit{source} & \textit{source} & \textit{source} & \textit{source} & \textit{source} & \textbf{-0.079} & \textbf{0.115} & 0.265 \\"),
    ("heldout", "Surrogate", 0.1, r"Surrogate & $\approx0.1$ & \textit{source} & \textit{source} & \textit{source} & \textit{source} & \textit{source} & -0.138 & \textbf{0.113} & 0.367 \\"),
    ("heldout", "Surrogate", 1, r"Surrogate & $\approx0.954$ & \textit{source} & \textit{source} & \textit{source} & \textit{source} & \textit{source} & -0.102 & \textbf{0.099} & 0.310 \\"),
    ("heldout", "Surrogate", 10, r"Surrogate & $\approx10.481$ & \textit{source} & \textit{source} & \textit{source} & \textit{source} & \textit{source} & \textbf{-0.036} & \textbf{0.103} & 0.250 \\"),
    ("heldout", "Surrogate", 100, r"Surrogate & 100 & \textit{source} & \textit{source} & \textit{source} & \textit{source} & \textit{source} & \textbf{-0.001} & 0.124 & 0.267 \\"),
    ("forward_2025", "Direct", 0.1, r"Direct & $\approx0.1$ & \textit{source} & \textit{source} & \textit{source} & \textit{source} & \textit{source} & -0.160 & 0.121 & 0.416 \\"),
    ("forward_2025", "Direct", 1, r"Direct & $\approx0.954$ & \textit{source} & \textit{source} & \textit{source} & \textit{source} & \textit{source} & -0.148 & 0.127 & 0.392 \\"),
    ("forward_2025", "Direct", 10, r"Direct & $\approx10.481$ & \textit{source} & \textit{source} & \textit{source} & \textit{source} & \textit{source} & \textbf{-0.105} & 0.134 & 0.318 \\"),
    ("forward_2025", "Direct", 100, r"Direct & 100 & \textit{source} & \textit{source} & \textit{source} & \textit{source} & \textit{source} & \textbf{-0.093} & \textbf{0.119} & 0.281 \\"),
    ("forward_2025", "Surrogate", 0.1, r"Surrogate & $\approx0.1$ & \textit{source} & \textit{source} & \textit{source} & \textit{source} & \textit{source} & -0.153 & \textbf{0.116} & 0.402 \\"),
    ("forward_2025", "Surrogate", 1, r"Surrogate & $\approx0.954$ & \textit{source} & \textit{source} & \textit{source} & \textit{source} & \textit{source} & -0.120 & \textbf{0.096} & 0.343 \\"),
    ("forward_2025", "Surrogate", 10, r"Surrogate & $\approx10.481$ & \textit{source} & \textit{source} & \textit{source} & \textit{source} & \textit{source} & \textbf{-0.050} & \textbf{0.092} & \textbf{0.258} \\"),
    ("forward_2025", "Surrogate", 100, r"Surrogate & 100 & \textit{source} & \textit{source} & \textit{source} & \textit{source} & \textit{source} & \textbf{-0.015} & \textbf{0.111} & \textbf{0.266} \\"),
]

COMP_METRICS = ("median_ratio", "mean_ratio", "weighted_mean_ratio", "COD", "COV")
RESOLVED_TODO_SNIPPETS = (
    "Canonical v2 table-source values for penalized-model MAPE",
    "Canonical v2 table-source values for penalized-model median ratio",
    "the old Direct-only span-regret result is no longer used",
    "Reintroduce value-based span regret only after the same v2 calculation",
    "Optional value-based portability check: only restore a span-regret table",
)


def once(text: str, old: str, new: str, name: str) -> str:
    n = text.count(old)
    if n != 1:
        raise RuntimeError(f"{name}: expected 1 occurrence, found {n}")
    return text.replace(old, new, 1)


def cell_lookup(ph: pd.DataFrame, table: str, split: str, family: str, anchor: float, metric: str) -> str:
    hit = ph.loc[
        (ph["table"] == table)
        & (ph["split"] == split)
        & (ph["family"] == family)
        & (pd.to_numeric(ph["nominal_display_anchor"], errors="coerce") == float(anchor))
        & (ph["metric"] == metric)
    ]
    if len(hit) != 1:
        raise RuntimeError(f"lookup failed {table} {split} {family} {anchor} {metric}: {len(hit)}")
    r = hit.iloc[0]
    return tex_from_display(str(r["value_display"]), bool(r["manuscript_bold"]), bool(r["manuscript_asterisk"]))


def fill_sources(old_line: str, cells: list[str]) -> str:
    parts = old_line.split(r"\textit{source}")
    if len(parts) != len(cells) + 1:
        raise RuntimeError(f"source count {len(parts) - 1} != {len(cells)} for {old_line[:80]}")
    out = parts[0]
    for cell, rest in zip(cells, parts[1:]):
        out += cell + rest
    return out


def build_regret_table() -> str:
    regret = pd.read_csv(V3 / "tables" / "transition_oos_span_regret_v3.csv")
    lines = {
        "heldout": [],
        "forward_2025": [],
    }
    for split in ("heldout", "forward_2025"):
        for fam in ("Direct", "Surrogate"):
            for met in ("R2_price", "MAE_price", "MAPE", "RMSE_log", "COD"):
                r = regret.loc[(regret["split"] == split) & (regret["family"] == fam) & (regret["metric"] == met)].iloc[0]
                lines[split].append(
                    f"{fam} & {METRIC_TEX[met]} & {fmt_rho(r['global_opt_rho'])} & {fmt_rho(r['best_inside_rho'])} & "
                    f"{fmt_raw(met, r['raw_regret'])} & {fmt_norm(r['normalized_regret'])} & {fmt_logd(r['log10_distance_global_opt_to_cv_span'])} \\\\"
                )
    held = "\n".join(lines["heldout"])
    fwd = "\n".join(lines["forward_2025"])
    return rf"""{{\color{{latestPurple}}
\begin{{table}}[!htbp]
\centering
\scriptsize
\setlength{{\tabcolsep}}{{2.6pt}}
\renewcommand{{\arraystretch}}{{1.08}}
\caption{{Value-based cost of restricting each out-of-time path to its family-specific frozen CV-derived descriptive transition span. No materiality threshold is imposed.}}
\label{{tab:transition_regret}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{llrrrrr}}
\toprule
Family & Metric & $\rho^{{*}}$ & $\rho_{{\mathrm{{span}}}}$ & raw regret & norm.\ regret & $\log_{{10}}$ dist. \\
\midrule
\multicolumn{{7}}{{l}}{{\textit{{Panel A: Held-out evaluation}}}} \\
\addlinespace[2pt]
{held}
\midrule
\multicolumn{{7}}{{l}}{{\textit{{Panel B: 2025 forward evaluation}}}} \\
\addlinespace[2pt]
{fwd}
\bottomrule
\end{{tabular}}}}
\vspace{{1mm}}
\begin{{minipage}}{{\textwidth}}
\scriptsize
\emph{{Notes.}}
Each family is restricted to its own frozen CV-derived descriptive transition span.
Normalized regret is raw regret divided by the full-path metric range when that range is nonzero.
Zero means the out-of-time global observed-grid optimum is available within the frozen span.
Exact event concordance and value-based regret answer different questions.
No interpolation, near-optimality tolerance, or materiality threshold is imposed, and no $\rho$ is selected.
\end{{minipage}}
\end{{table}}
}}"""


def apply_manuscript() -> dict:
    ph = pd.read_csv(V3 / "tables" / "paper_source_placeholder_replacements.csv")
    text = PAPER.read_text(encoding="utf-8")
    n_source_before = text.count(r"\textit{source}")
    replacements = []

    for split, fam, anc, old in PRIMARY_ROWS:
        cells = [cell_lookup(ph, "tab:path_anchor_summary", split, fam, anc, m) for m in ("MAPE", "RMSE_log")]
        new = fill_sources(old, cells)
        text = once(text, old, new, f"primary {split} {fam} {anc}")
        replacements.append(("primary", split, fam, anc))
    for split, fam, anc, old in COMP_ROWS:
        cells = [cell_lookup(ph, "tab:path_anchor_complementary", split, fam, anc, m) for m in COMP_METRICS]
        new = fill_sources(old, cells)
        text = once(text, old, new, f"comp {split} {fam} {anc}")
        replacements.append(("complementary", split, fam, anc))

    text = once(
        text,
        r"""{\color{latestPurple}
\todo[inline]{Canonical v2 table-source values for penalized-model MAPE and $\operatorname{RMSE}_{\log P}$ were not attached to this manuscript. Cells marked \emph{source} are intentionally left unpopulated rather than reconstructed from figures or older runs. Replace only those cells from \texttt{representative\_rho\_primary\_source.tex} (or its canonical CSV/parquet counterpart).}
\begin{table}[!htbp]""",
        r"""{\color{latestPurple}
\begin{table}[!htbp]""",
        "remove primary source TODO",
    )
    text = once(
        text,
        r"""\todo[inline]{Canonical v2 table-source values for penalized-model median ratio, mean ratio, weighted mean ratio, COD, and COV were not attached to this manuscript. Cells marked \emph{source} are intentionally left unpopulated rather than reconstructed from figures or older runs. Replace only those cells from \texttt{representative\_rho\_secondary\_source.tex} (or its canonical CSV/parquet counterpart).}
\begin{table}[!htbp]""",
        r"""\begin{table}[!htbp]""",
        "remove complementary source TODO",
    )

    text = once(
        text,
        r"""\oldtext{For Direct, we additionally report direction-aware \emph{span regret} on the held-out and 2025 paths: the difference between the split-specific global discrete-grid optimum of a metric and the best value attainable while restricting $\rho$ to the frozen CV span, with the sign defined so that regret is nonnegative. Normalized regret divides this quantity by the metric's full-path range when that range is nonzero. No smoothing, interpolation, near-optimality tolerance, or post-hoc threshold is used. The transition analysis is descriptive and does not select a penalty strength or redefine the prespecified display anchors.}
\latesttext{No smoothing, interpolation, near-optimality tolerance, or post-hoc threshold is used. The transition analysis is descriptive and does not select a penalty strength or redefine the prespecified display anchors.}
\todo{V2 transition update: the old Direct-only span-regret result is no longer used as a final paper result because Surrogate now also has a valid frozen CV span. Regenerate the same direction-aware regret diagnostic symmetrically for both families from the final augmented v2 paths before deciding whether to restore a regret table. Do not infer or hand-reconstruct the missing Surrogate values.}""",
        r"""\oldtext{For Direct, we additionally report direction-aware \emph{span regret} on the held-out and 2025 paths: the difference between the split-specific global discrete-grid optimum of a metric and the best value attainable while restricting $\rho$ to the frozen CV span, with the sign defined so that regret is nonnegative. Normalized regret divides this quantity by the metric's full-path range when that range is nonzero. No smoothing, interpolation, near-optimality tolerance, or post-hoc threshold is used. The transition analysis is descriptive and does not select a penalty strength or redefine the prespecified display anchors.}
\oldtext{No smoothing, interpolation, near-optimality tolerance, or post-hoc threshold is used. The transition analysis is descriptive and does not select a penalty strength or redefine the prespecified display anchors.}
\latesttext{For each family with a valid frozen CV transition span, report direction-aware out-of-time span regret as the difference between the split-specific global observed-grid optimum and the best value attainable within that family's frozen CV span, normalized by the full-path metric range when nonzero. No smoothing, interpolation, near-optimality tolerance, or post-hoc threshold is used. The transition analysis is descriptive and does not select a penalty strength or redefine the prespecified display anchors.}""",
        "section 4.2 protocol",
    )

    text = once(
        text,
        r"""\latesttext{We also report the frozen five-metric transition diagnostic from Section~\ref{subsec:path_design}, including fold and leave-one-fold-out stability and retrospective temporal concordance. This diagnostic characterizes the path and does not define a selected or recommended range.}""",
        r"""\oldtext{We also report the frozen five-metric transition diagnostic from Section~\ref{subsec:path_design}, including fold and leave-one-fold-out stability and retrospective temporal concordance. This diagnostic characterizes the path and does not define a selected or recommended range.}
\latesttext{We also report the frozen five-metric transition diagnostic from Section~\ref{subsec:path_design}, including fold and leave-one-fold-out stability, retrospective exact concordance, and family-symmetric value-based span regret. This diagnostic characterizes the path and does not define a selected or recommended range.}""",
        "results intro",
    )

    text = once(
        text,
        r"""\caption{Mechanism and residual-structure paths versus $\rho$ for Direct and Surrogate on held-out (solid) and 2025 (dashed) evaluations. Gray shading, where present, is the CV-derived descriptive transition span for that family; it is not a selected or recommended penalty interval.}""",
        r"""\caption{\oldtext{Mechanism and residual-structure paths versus $\rho$ for Direct and Surrogate on held-out (solid) and 2025 (dashed) evaluations. Gray shading, where present, is the CV-derived descriptive transition span for that family; it is not a selected or recommended penalty interval.}\latesttext{Mechanism and residual-structure paths versus $\rho$ for Direct and Surrogate on held-out (solid) and 2025 (dashed) evaluations. The $\beta_{\log}=0$ line is a first-order neutrality reference; $\Delta_{\mathrm{NL}}$ and dCor are not given assessor-style zero targets. Gray fill and dashed vertical boundaries mark the frozen family-specific CV-derived descriptive transition span; they are not a selected or recommended penalty interval.}}""",
        "mechanism caption",
    )

    text = once(
        text,
        r"""\caption{Accuracy--equity trajectories with $R^2_P$ on the vertical axis and PRD, PRB, MKI, and VEI on the horizontal axis, for held-out and 2025 evaluations. Linear and ordinary LightGBM are context anchors. Arrows indicate increasing $\rho$ and do not mark a selected point.}""",
        r"""\caption{\oldtext{Accuracy--equity trajectories with $R^2_P$ on the vertical axis and PRD, PRB, MKI, and VEI on the horizontal axis, for held-out and 2025 evaluations. Linear and ordinary LightGBM are context anchors. Arrows indicate increasing $\rho$ and do not mark a selected point.}\latesttext{Accuracy--equity trajectories with $R^2_P$ on the vertical axis and PRD, PRB, MKI, and VEI on the horizontal axis, for held-out and 2025 evaluations. Linear and ordinary LightGBM are context anchors. Dotted vertical lines mark metric neutrality where applicable, and shaded regions are assessor-facing guidance ranges rather than compliance claims. Arrows indicate increasing $\rho$ and do not mark a selected point.}}""",
        "accuracy-equity caption",
    )

    text = once(
        text,
        r"""\latesttext{The lower-tail extension also changes the Surrogate conclusion. Its CV $\operatorname{RMSE}_{\log P}$ event moves from the $\rho=0$ boundary to the positive interior point $\rho=0.00222$, and its MAE event moves below $0.1$. The five Surrogate CV events therefore define their own descriptive span, $\rho\in[0.00222,0.954]$, and all seven Surrogate leave-one-fold-out aggregations now support a positive-interior five-event span. This does not make the Surrogate span a selected or portable operating range: only one of five held-out events and one of five 2025 events (MAE in each case) fall inside it. Taken together, the two families show that reproducible within-CV transition structure is distinct from temporal portability of the exact operating location. Appendix~\ref{app:ccao_path_details} reports the event locations and leave-one-fold-out diagnostics. These are descriptive path-stability results and do not select a model family or operating point.}
\todo{V2 result update: this subsection now uses only the frozen augmented-grid findings supplied by the completed run. Reintroduce value-based span regret only after the same v2 calculation is available for both Direct and Surrogate.}""",
        r"""\latesttext{The lower-tail extension also changes the Surrogate conclusion. Its CV $\operatorname{RMSE}_{\log P}$ event moves from the $\rho=0$ boundary to the positive interior point $\rho=0.00222$, and its MAE event moves below $0.1$. The five Surrogate CV events therefore define their own descriptive span, $\rho\in[0.00222,0.954]$, and all seven Surrogate leave-one-fold-out aggregations now support a positive-interior five-event span. This does not make the Surrogate span a selected or portable operating range: only one of five held-out events and one of five 2025 events (MAE in each case) fall inside it. Taken together, the two families show that reproducible within-CV transition structure is distinct from temporal portability of the exact operating location. Appendix~\ref{app:ccao_path_details} reports the event locations and leave-one-fold-out diagnostics. These are descriptive path-stability results and do not select a model family or operating point.}
\latesttext{The family-specific CV spans remain Direct $\rho\in[0.0494,1.099]$ and Surrogate $\rho\in[0.00222,0.954]$. They should be interpreted as envelopes of the five prediction/COD turning events, not as equity-neutrality intervals. Appendix Figures~\ref{fig:vertical_equity_event_locations} and~\ref{fig:mechanism_event_locations} show that the closest-to-neutral locations for PRD, PRB, MKI, and VEI generally occur at much larger $\rho$ for both families; predictive/COD turning and equity neutrality therefore occur at distinct scales. Exact out-of-time event concordance remains weak, but the family-symmetric value-based span-regret result in Table~\ref{tab:transition_regret} is more nuanced. Direct normalized regret is roughly $0.005$--$0.056$ on the held-out sample and $0.004$--$0.037$ in 2025. Surrogate MAE regret is $0$ on both held-out and 2025 evaluations; Surrogate $\operatorname{RMSE}_{\log P}$ normalized regret is at most $0.003$; the largest reported Surrogate normalized regret is 2025 $R^2_P\approx 0.083$. The corresponding metric-value losses are generally limited in magnitude over the observed paths, although no materiality threshold is imposed. Direct $\operatorname{RMSE}_{\log P}$ and Surrogate $R^2_P$ CV events are particularly value-flat, while MAE events are comparatively sharper; exact event-$\rho$ movement therefore should not be interpreted as equal substantive movement for all criteria.}""",
        "section 5.6",
    )

    text = once(
        text,
        r"""\latesttext{The transition analysis adds a separate temporal qualification. After resolving the lower tail of the grid, both Direct and Surrogate exhibit positive-interior five-metric CV transition spans with seven-of-seven leave-one-fold-out support. Exact out-of-time event portability remains weak, however: Direct has zero of five held-out and zero of five 2025 events inside its frozen span, while Surrogate has one of five in each period. Chronological validation is therefore useful for characterizing repeatable path structure, but the evidence does not support treating either raw-$\rho$ interval as an intrinsic or temporally invariant operating region. Any later deployment calibration should specify explicitly which predictive, valuation, and equity criteria define the operating decision and re-evaluate that decision across time.}""",
        r"""\oldtext{The transition analysis adds a separate temporal qualification. After resolving the lower tail of the grid, both Direct and Surrogate exhibit positive-interior five-metric CV transition spans with seven-of-seven leave-one-fold-out support. Exact out-of-time event portability remains weak, however: Direct has zero of five held-out and zero of five 2025 events inside its frozen span, while Surrogate has one of five in each period. Chronological validation is therefore useful for characterizing repeatable path structure, but the evidence does not support treating either raw-$\rho$ interval as an intrinsic or temporally invariant operating region. Any later deployment calibration should specify explicitly which predictive, valuation, and equity criteria define the operating decision and re-evaluate that decision across time.}
\latesttext{The transition analysis adds a separate temporal qualification. After resolving the lower tail of the grid, both Direct and Surrogate exhibit positive-interior five-metric CV transition spans with seven-of-seven leave-one-fold-out support. Exact out-of-time event portability remains weak, however: Direct has zero of five held-out and zero of five 2025 events inside its frozen span, while Surrogate has one of five in each period. The augmented evidence distinguishes a reproducible prediction/COD transition structure from the much larger penalties at which assessor-facing measures or mechanism diagnostics become closest to neutrality. Therefore there is no single empirically supported ``safe'' $\rho$ interval, and a future deployment choice requires an explicit operating criterion. Chronological validation is therefore useful for characterizing repeatable path structure, but the evidence does not support treating either raw-$\rho$ interval as an intrinsic or temporally invariant operating region. Any later deployment calibration should specify explicitly which predictive, valuation, and equity criteria define the operating decision and re-evaluate that decision across time.}""",
        "discussion",
    )

    text = once(
        text,
        r"""\latesttext{The v2 transition analysis is computed read-only from the final augmented 994-tree path table. The original $0.1$--$100$ fits and v1 transition artifacts are preserved unchanged; v2 appends 32 lower-tail positive values per family under the same frozen data, split, seed, predictor set, and 994-tree configuration. The 448 new chronological-CV fits were completed first, the augmented CV transition state was frozen, and only then were the same 32 new values evaluated on the held-out and 2025 samples (128 additional out-of-time fits). The five-metric transition rule and the prespecified display anchors remain unchanged. The lower-grid protocol, event tables, fold and leave-one-fold-out summaries, retrospective temporal-concordance calculations, event-sharpness diagnostics, paper-asset manifest, and hashes linking v1 and v2 are archived in the transition-analysis replication artifacts.}""",
        r"""\oldtext{The v2 transition analysis is computed read-only from the final augmented 994-tree path table. The original $0.1$--$100$ fits and v1 transition artifacts are preserved unchanged; v2 appends 32 lower-tail positive values per family under the same frozen data, split, seed, predictor set, and 994-tree configuration. The 448 new chronological-CV fits were completed first, the augmented CV transition state was frozen, and only then were the same 32 new values evaluated on the held-out and 2025 samples (128 additional out-of-time fits). The five-metric transition rule and the prespecified display anchors remain unchanged. The lower-grid protocol, event tables, fold and leave-one-fold-out summaries, retrospective temporal-concordance calculations, event-sharpness diagnostics, paper-asset manifest, and hashes linking v1 and v2 are archived in the transition-analysis replication artifacts.}
\latesttext{The v2 transition analysis is computed read-only from the final augmented 994-tree path table. The original $0.1$--$100$ fits and v1 transition artifacts are preserved unchanged; v2 appends 32 lower-tail positive values per family under the same frozen data, split, seed, predictor set, and 994-tree configuration. The 448 new chronological-CV fits were completed first, the augmented CV transition state was frozen, and only then were the same 32 new values evaluated on the held-out and 2025 samples (128 additional out-of-time fits). The five-metric transition rule and the prespecified display anchors remain unchanged. The lower-grid protocol, event tables, fold and leave-one-fold-out summaries, retrospective temporal-concordance calculations, the completed v3 artifacts \texttt{transition\_oos\_span\_regret\_v3}, \texttt{transition\_lofo\_endpoint\_summary\_v3}, \texttt{vertical\_equity\_event\_locations}, \texttt{mechanism\_event\_locations}, and \texttt{event\_sharpness\_summary\_v3}, the v3 figure QA manifest, paper-asset manifest, and hashes linking v1, v2, and v3 are archived in the transition-analysis replication artifacts.}""",
        "appendix provenance paragraph",
    )

    text = once(
        text,
        r"""\latesttext{The main text reports only the interpretation of the five-metric transition diagnostic. The objects below provide the auditable event locations and fold/leave-one-fold-out stability checks. Any family-specific interval is described only as a \emph{CV-derived descriptive transition span}, never as a recommended, safe, optimal, preferred, or deployment range.}""",
        r"""\oldtext{The main text reports only the interpretation of the five-metric transition diagnostic. The objects below provide the auditable event locations and fold/leave-one-fold-out stability checks. Any family-specific interval is described only as a \emph{CV-derived descriptive transition span}, never as a recommended, safe, optimal, preferred, or deployment range.}
\latesttext{The main text reports only the interpretation of the five-metric transition diagnostic. The objects below provide the auditable event locations, fold/leave-one-fold-out stability checks, and family-symmetric value-based span regret. Any family-specific interval is described only as a \emph{CV-derived descriptive transition span}, never as a recommended, safe, optimal, preferred, or deployment range.}""",
        "transition appendix intro",
    )

    text = once(
        text,
        r"""LOFO five-event support
& \oldtext{7/7; valid endpoints $[0.1,\,0.153]$ to $[1.099,\,1.265]$}\latesttext{7/7}
& \oldtext{2/7}\latesttext{7/7} \\""",
        r"""LOFO five-event support
& \oldtext{7/7; valid endpoints $[0.1,\,0.153]$ to $[1.099,\,1.265]$}\oldtext{7/7}\latesttext{7/7; lower $[0.004498,\,0.065513]$, upper $[1.098541,\,1.264855]$}
& \oldtext{2/7}\oldtext{7/7}\latesttext{7/7; lower $[0.001099,\,0.010481]$, upper $[0.828643,\,1.676833]$} \\""",
        "LOFO row",
    )

    text = once(
        text,
        r"""Held-out and 2025 comparisons are retrospective temporal concordance, not prospective confirmation.
No $\rho$ is selected.""",
        r"""Held-out and 2025 comparisons are retrospective temporal concordance, not prospective confirmation.
No $\rho$ is selected.
\latesttext{Leave-one-fold-out 7/7 records that a positive-interior five-event span exists in every omitted-fold aggregation; it does not imply precise endpoint stability. Direct $\log_{10}$-width min/median/max is $1.286/1.347/2.449$ with $0/7$ first-positive lower-bound hits; Surrogate $\log_{10}$-width min/median/max is $2.204/2.633/2.939$ with $1/7$ first-positive lower-bound hits. Lower-end location is materially less stable than the upper end.}""",
        "LOFO note",
    )

    text = once(
        text,
        r"""\caption{Turning-event locations for the five fixed criteria. Gray shading, where present, is the frozen CV-derived descriptive transition span and is not a selected or recommended penalty interval.}
\label{fig:transition_event_locations}
\end{figure}""",
        r"""\caption{\oldtext{Turning-event locations for the five fixed criteria. Gray shading, where present, is the frozen CV-derived descriptive transition span and is not a selected or recommended penalty interval.}\latesttext{Turning-event locations for the five fixed prediction/COD criteria. Gray fill and dashed vertical boundaries mark the frozen family-specific CV-derived descriptive transition span; they are not a selected or recommended penalty interval.}}
\label{fig:transition_event_locations}
\end{figure}

\begin{figure}[!htbp]
\centering
\safeincludegraphics[width=0.95\textwidth]{img/generated_v12_994/vertical_equity_event_locations.pdf}
\caption{\latesttext{Closest-to-neutral observed-grid locations for PRD, PRB, MKI, and VEI. Gray fill and dashed vertical boundaries mark the already-frozen five-primary-metric CV-derived descriptive transition span; these events do not redefine that span and do not select $\rho$.}}
\label{fig:vertical_equity_event_locations}
\end{figure}

\begin{figure}[!htbp]
\centering
\safeincludegraphics[width=0.95\textwidth]{img/generated_v12_994/mechanism_event_locations.pdf}
\caption{\latesttext{Mechanism turning-event locations corresponding to minimum $|\beta_{\log}|$ and minimum dCor. Gray fill and dashed vertical boundaries mark the frozen five-primary-metric CV-derived descriptive transition span as context only. No CV $\Delta_{\mathrm{NL}}$ event is shown because $\Delta_{\mathrm{NL}}$ was not computed retrospectively for chronological CV. These events do not define a selected or recommended range.}}
\label{fig:mechanism_event_locations}
\end{figure}""",
        "event location figures",
    )

    old_regret = r"""\begin{table}[!htbp]
\centering
\scriptsize
\setlength{\tabcolsep}{2.6pt}
\renewcommand{\arraystretch}{1.08}
\caption{\oldtext{Value-based cost of restricting the Direct out-of-time paths to the frozen CV-derived descriptive span. No small/large regret threshold is imposed.}}
\label{tab:transition_regret}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrrrr}
\toprule
\oldtext{Metric} & \oldtext{$\rho^{*}$} & \oldtext{value$^{*}$} & \oldtext{$\rho_{\mathrm{span}}$} & \oldtext{value$_{\mathrm{span}}$} & \oldtext{raw regret} & \oldtext{norm. regret} & \oldtext{$\log_{10}$ dist.} \\
\midrule
\multicolumn{8}{l}{\oldtext{\textit{Panel A: held-out evaluation}}} \\
\addlinespace[2pt]
\oldtext{$R^2_P$} & \oldtext{3.393} & \oldtext{0.901} & \oldtext{1.099} & \oldtext{0.899} & \oldtext{0.0018} & \oldtext{0.056} & \oldtext{0.490} \\
\oldtext{$\operatorname{MAE}_P$} & \oldtext{1.931} & \oldtext{\$73,924} & \oldtext{0.829} & \oldtext{\$74,409} & \oldtext{\$485} & \oldtext{0.044} & \oldtext{0.245} \\
\oldtext{$\operatorname{MAPE}_P$} & \oldtext{1.265} & \oldtext{21.0\%} & \oldtext{0.829} & \oldtext{21.1\%} & \oldtext{0.033 pp} & \oldtext{0.011} & \oldtext{0.061} \\
\oldtext{$\operatorname{RMSE}_{\log P}$} & \oldtext{0.720} & \oldtext{0.289} & \oldtext{0.720} & \oldtext{0.289} & \oldtext{0} & \oldtext{0} & \oldtext{0} \\
\oldtext{COD} & \oldtext{1.265} & \oldtext{21.52} & \oldtext{1.099} & \oldtext{21.53} & \oldtext{0.015} & \oldtext{0.005} & \oldtext{0.061} \\
\midrule
\multicolumn{8}{l}{\oldtext{\textit{Panel B: 2025 forward evaluation}}} \\
\addlinespace[2pt]
\oldtext{$R^2_P$} & \oldtext{1.931} & \oldtext{0.912} & \oldtext{1.099} & \oldtext{0.910} & \oldtext{0.0014} & \oldtext{0.037} & \oldtext{0.245} \\
\oldtext{$\operatorname{MAE}_P$} & \oldtext{2.560} & \oldtext{\$76,558} & \oldtext{1.099} & \oldtext{\$76,848} & \oldtext{\$289} & \oldtext{0.023} & \oldtext{0.367} \\
\oldtext{$\operatorname{MAPE}_P$} & \oldtext{2.223} & \oldtext{20.6\%} & \oldtext{0.309} & \oldtext{20.6\%} & \oldtext{0.052 pp} & \oldtext{0.020} & \oldtext{0.306} \\
\oldtext{$\operatorname{RMSE}_{\log P}$} & \oldtext{1.456} & \oldtext{0.278} & \oldtext{0.625} & \oldtext{0.278} & \oldtext{0.0001} & \oldtext{0.003} & \oldtext{0.122} \\
\oldtext{COD} & \oldtext{2.223} & \oldtext{21.05} & \oldtext{0.309} & \oldtext{21.11} & \oldtext{0.062} & \oldtext{0.027} & \oldtext{0.306} \\
\bottomrule
\end{tabular}}
\vspace{1mm}
\begin{minipage}{\textwidth}
\scriptsize
\oldtext{\emph{Notes.} This Direct-only table belongs to the v1 span analysis. Its values are not carried forward as final v2 evidence because the final augmented experiment gives both families valid CV spans and therefore requires the regret diagnostic to be regenerated symmetrically.}
\end{minipage}
\end{table}
\todo{Optional value-based portability check: only restore a span-regret table if the same deterministic v2 calculation is generated for both Direct and Surrogate using their own frozen family-specific CV spans. The crossed-out Direct-only v1 table above is revision history and must not be used in the clean manuscript. Exact concordance is sufficient for the current descriptive transition result, so do not delay the paper solely to recreate this optional diagnostic.}"""
    new_regret = (
        "\\begin{oldrevisionblock}\n"
        + old_regret.replace("\\label{tab:transition_regret}\n", "").replace(
            "\n\\todo{Optional value-based portability check: only restore a span-regret table if the same deterministic v2 calculation is generated for both Direct and Surrogate using their own frozen family-specific CV spans. The crossed-out Direct-only v1 table above is revision history and must not be used in the clean manuscript. Exact concordance is sufficient for the current descriptive transition result, so do not delay the paper solely to recreate this optional diagnostic.}",
            "",
        )
        + "\n\\end{oldrevisionblock}\n\n"
        + build_regret_table()
    )
    text = once(text, old_regret, new_regret, "span-regret table")

    text = once(
        text,
        r"""\oldtext{Exact event concordance is deliberately interpreted jointly with Table~\ref{tab:transition_regret}: a turning point can move outside the CV interval even when the best value available inside the interval is close to the global discrete-grid optimum. Event-sharpness diagnostics are retained in the machine-readable replication package to document this issue without defining a post-hoc near-optimality tolerance.}
\latesttext{Exact event concordance is used here only as a strict location diagnostic. Event-sharpness diagnostics remain in the machine-readable replication package rather than being promoted to an additional selection rule.}""",
        r"""\oldtext{Exact event concordance is deliberately interpreted jointly with Table~\ref{tab:transition_regret}: a turning point can move outside the CV interval even when the best value available inside the interval is close to the global discrete-grid optimum. Event-sharpness diagnostics are retained in the machine-readable replication package to document this issue without defining a post-hoc near-optimality tolerance.}
\oldtext{Exact event concordance is used here only as a strict location diagnostic. Event-sharpness diagnostics remain in the machine-readable replication package rather than being promoted to an additional selection rule.}
\latesttext{Exact event concordance is used here only as a strict location diagnostic. Machine-readable event-sharpness diagnostics document flat versus sharp events without defining a near-optimality tolerance or a new $\rho$ interval.}""",
        "event-sharpness note",
    )

    text = once(
        text,
        r"""\latesttext{Throughout the $\rho$-evolution figures below, gray shading denotes the family-specific CV-derived descriptive transition span when one exists. It is a path-description aid, not a selected, safe, preferred, or recommended penalty interval.}""",
        r"""\oldtext{Throughout the $\rho$-evolution figures below, gray shading denotes the family-specific CV-derived descriptive transition span when one exists. It is a path-description aid, not a selected, safe, preferred, or recommended penalty interval.}
\latesttext{Throughout the $\rho$-evolution figures below, gray fill and dashed vertical boundaries mark the frozen family-specific CV-derived descriptive transition span when one exists. It is a path-description aid, not a selected, safe, preferred, or recommended penalty interval.}""",
        "rho-evolution intro",
    )

    text = once(
        text,
        r"""\caption{Predictive-metric paths and valuation-level/uniformity paths along the Direct and Surrogate grids, with held-out and 2025 evaluations overlaid. Gray shading, where present, is the CV-derived descriptive transition span.}""",
        r"""\caption{\oldtext{Predictive-metric paths and valuation-level/uniformity paths along the Direct and Surrogate grids, with held-out and 2025 evaluations overlaid. Gray shading, where present, is the CV-derived descriptive transition span.}\latesttext{Predictive-metric paths and valuation-level/uniformity paths along the Direct and Surrogate grids, with held-out and 2025 evaluations overlaid. Gray fill and dashed vertical boundaries mark the frozen family-specific CV-derived descriptive transition span. Valuation-level panels include the ratio $=1$ reference.}}""",
        "predictive/level caption",
    )

    text = once(
        text,
        r"""\caption{Vertical-equity metric paths versus $\rho$ for Direct and Surrogate. Gray shading, where present, is the CV-derived descriptive transition span.}""",
        r"""\caption{\oldtext{Vertical-equity metric paths versus $\rho$ for Direct and Surrogate. Gray shading, where present, is the CV-derived descriptive transition span.}\latesttext{Vertical-equity metric paths versus $\rho$ for Direct and Surrogate. Gray fill and dashed vertical boundaries mark the frozen family-specific CV-derived descriptive transition span. Reference lines mark PRD $=1$, PRB $=0$, MKI $=1$, and VEI $=0$.}}""",
        "vertical equity paths caption",
    )

    text = once(
        text,
        r"""\caption{Chronological-fold predictive-metric paths (thin gray) and equal-weight CV means (thick). Gray shading, where present, is the CV-derived descriptive transition span.}""",
        r"""\caption{\oldtext{Chronological-fold predictive-metric paths (thin gray) and equal-weight CV means (thick). Gray shading, where present, is the CV-derived descriptive transition span.}\latesttext{Chronological-fold predictive-metric paths (thin gray) and equal-weight CV means (thick). Gray fill and dashed vertical boundaries mark the frozen family-specific CV-derived descriptive transition span.}}""",
        "cv predictive caption",
    )

    text = once(
        text,
        r"""\caption{Chronological-fold valuation-level and uniformity paths (thin gray) and equal-weight CV means (thick).}""",
        r"""\caption{\oldtext{Chronological-fold valuation-level and uniformity paths (thin gray) and equal-weight CV means (thick).}\latesttext{Chronological-fold valuation-level and uniformity paths (thin gray) and equal-weight CV means (thick). Gray fill and dashed vertical boundaries mark the frozen family-specific CV-derived descriptive transition span. Valuation-level panels include the ratio $=1$ reference.}}""",
        "cv level caption",
    )

    text = once(
        text,
        r"""\caption{Chronological-fold vertical-equity paths (thin gray) and equal-weight CV means (thick).}""",
        r"""\caption{\oldtext{Chronological-fold vertical-equity paths (thin gray) and equal-weight CV means (thick).}\latesttext{Chronological-fold vertical-equity paths (thin gray) and equal-weight CV means (thick). Gray fill and dashed vertical boundaries mark the frozen family-specific CV-derived descriptive transition span. Reference lines mark PRD $=1$, PRB $=0$, MKI $=1$, and VEI $=0$.}}""",
        "cv vertical caption",
    )

    text = once(
        text,
        r"""\caption{Chronological-fold mechanism paths for $\beta_{\log}$ and distance correlation (thin gray) and equal-weight CV means (thick). $\Delta_{\mathrm{NL}}$ is not shown because it is not computed for CV.}""",
        r"""\caption{\oldtext{Chronological-fold mechanism paths for $\beta_{\log}$ and distance correlation (thin gray) and equal-weight CV means (thick). $\Delta_{\mathrm{NL}}$ is not shown because it is not computed for CV.}\latesttext{Chronological-fold mechanism paths for $\beta_{\log}$ and distance correlation (thin gray) and equal-weight CV means (thick). The $\beta_{\log}=0$ line is a first-order neutrality reference; dCor is not given an assessor-style zero target. Gray fill and dashed vertical boundaries mark the frozen family-specific CV-derived descriptive transition span. $\Delta_{\mathrm{NL}}$ is not shown because it is not computed for CV.}}""",
        "cv mechanism caption",
    )

    text = once(
        text,
        r"""\caption{Ratio-shape profiles restricted to the prespecified display anchors that lie inside each family's frozen CV-derived descriptive transition span. Families without a valid common positive span show no penalized curves.}""",
        r"""\caption{\oldtext{Ratio-shape profiles restricted to the prespecified display anchors that lie inside each family's frozen CV-derived descriptive transition span. Families without a valid common positive span show no penalized curves.}\latesttext{Ratio-shape profiles restricted to the prespecified display anchors that lie inside each family's frozen CV-derived descriptive transition span. The horizontal line at 1 is the principal neutrality reference; lines at 0.9 and 1.1 are aggregate appraisal-level reference guides and are not binwise acceptance criteria.}}""",
        "ratio-shape span-only caption",
    )

    text = once(
        text,
        r"""\caption{Held-out assessor-facing diagnostics (horizontal) versus predictive metrics (vertical) along the Direct and Surrogate paths.}""",
        r"""\caption{\oldtext{Held-out assessor-facing diagnostics (horizontal) versus predictive metrics (vertical) along the Direct and Surrogate paths.}\latesttext{Held-out assessor-facing diagnostics (horizontal) versus predictive metrics (vertical) along the Direct and Surrogate paths. Dotted vertical lines mark metric neutrality where applicable, and shaded regions are assessor-facing guidance ranges rather than compliance claims.}}""",
        "tradeoff equity heldout caption",
    )

    text = once(
        text,
        r"""\caption{2025 assessor-facing diagnostics (horizontal) versus predictive metrics (vertical) along the Direct and Surrogate paths.}""",
        r"""\caption{\oldtext{2025 assessor-facing diagnostics (horizontal) versus predictive metrics (vertical) along the Direct and Surrogate paths.}\latesttext{2025 assessor-facing diagnostics (horizontal) versus predictive metrics (vertical) along the Direct and Surrogate paths. Dotted vertical lines mark metric neutrality where applicable, and shaded regions are assessor-facing guidance ranges rather than compliance claims.}}""",
        "tradeoff equity 2025 caption",
    )

    text = once(
        text,
        r"""\caption{Held-out mechanism/residual diagnostics (horizontal) versus predictive metrics (vertical).}""",
        r"""\caption{\oldtext{Held-out mechanism/residual diagnostics (horizontal) versus predictive metrics (vertical).}\latesttext{Held-out mechanism/residual diagnostics (horizontal) versus predictive metrics (vertical). The dotted vertical line at $\beta_{\log}=0$ is a first-order neutrality reference where applicable; $\Delta_{\mathrm{NL}}$ and dCor are not given assessor-style zero targets.}}""",
        "tradeoff mechanism heldout caption",
    )

    text = once(
        text,
        r"""\caption{2025 mechanism/residual diagnostics (horizontal) versus predictive metrics (vertical).}""",
        r"""\caption{\oldtext{2025 mechanism/residual diagnostics (horizontal) versus predictive metrics (vertical).}\latesttext{2025 mechanism/residual diagnostics (horizontal) versus predictive metrics (vertical). The dotted vertical line at $\beta_{\log}=0$ is a first-order neutrality reference where applicable; $\Delta_{\mathrm{NL}}$ and dCor are not given assessor-style zero targets.}}""",
        "tradeoff mechanism 2025 caption",
    )

    text = once(
        text,
        r"""\oldtext{It should also archive the frozen transition-analysis protocol and status files, exact event tables, fold and leave-one-fold-out summaries, Direct span-regret and event-sharpness tables, the canonical paper-asset manifest, and hashes linking those outputs to the same 994-tree data/configuration/split identity.}\latesttext{It should also archive the original and augmented grid manifests and hashes, the v2 CV-freeze status, exact family-specific event tables, fold and leave-one-fold-out summaries, retrospective concordance outputs, event-sharpness diagnostics, any regenerated family-symmetric span-regret table, the canonical paper-asset manifest, and hashes linking all v2 outputs to the same 994-tree data/configuration/split identity.}""",
        r"""\oldtext{It should also archive the frozen transition-analysis protocol and status files, exact event tables, fold and leave-one-fold-out summaries, Direct span-regret and event-sharpness tables, the canonical paper-asset manifest, and hashes linking those outputs to the same 994-tree data/configuration/split identity.}\oldtext{It should also archive the original and augmented grid manifests and hashes, the v2 CV-freeze status, exact family-specific event tables, fold and leave-one-fold-out summaries, retrospective concordance outputs, event-sharpness diagnostics, any regenerated family-symmetric span-regret table, the canonical paper-asset manifest, and hashes linking all v2 outputs to the same 994-tree data/configuration/split identity.}
\latesttext{It should also archive the original and augmented grid manifests and hashes, the v2 CV-freeze status, exact family-specific event tables, fold and leave-one-fold-out summaries, retrospective concordance outputs, the completed v3 artifacts \texttt{transition\_oos\_span\_regret\_v3}, \texttt{transition\_lofo\_endpoint\_summary\_v3}, \texttt{vertical\_equity\_event\_locations}, \texttt{mechanism\_event\_locations}, and \texttt{event\_sharpness\_summary\_v3}, the v3 figure QA manifest, the canonical paper-asset manifest, and hashes linking all v2/v3 outputs to the same 994-tree data/configuration/split identity.}""",
        "replication provenance",
    )

    PAPER.write_text(text, encoding="utf-8")
    qa = static_qa(text)
    qa["n_source_before"] = n_source_before
    qa["n_source_after"] = text.count(r"\textit{source}")
    qa["n_row_replacements"] = len(replacements)
    return qa


def _active_labels(text: str) -> list[str]:
    labels = []
    for raw in text.splitlines():
        line = raw.split("%", 1)[0]
        labels.extend(re.findall(r"\\label\{([^}]+)\}", line))
    return labels


def static_qa(text: str) -> dict:
    problems = []
    if text.count("{") != text.count("}"):
        problems.append(f"unbalanced braces {text.count('{')} vs {text.count('}')}")
    begins = re.findall(r"\\begin\{([A-Za-z*]+)\}", text)
    ends = re.findall(r"\\end\{([A-Za-z*]+)\}", text)
    if collections.Counter(begins) != collections.Counter(ends):
        problems.append(f"begin/end mismatch {collections.Counter(begins) - collections.Counter(ends)} vs {collections.Counter(ends) - collections.Counter(begins)}")
    labels = _active_labels(text)
    dup = [k for k, v in collections.Counter(labels).items() if v > 1]
    if dup:
        problems.append(f"duplicate labels: {dup}")
    if labels.count("tab:transition_regret") != 1:
        problems.append(f"tab:transition_regret count={labels.count('tab:transition_regret')}")
    for lab in ("fig:vertical_equity_event_locations", "fig:mechanism_event_locations"):
        if labels.count(lab) != 1:
            problems.append(f"{lab} count={labels.count(lab)}")
    n_wrapped_source = text.count(r"\oldtext{\textit{source}}")
    n_raw_source = text.count(r"\textit{source}")
    if n_raw_source != n_wrapped_source:
        problems.append(f"unwrapped \\textit{{source}}: raw={n_raw_source} wrapped={n_wrapped_source}")
    if n_wrapped_source != 112:
        problems.append(f"expected 112 wrapped source cells, found {n_wrapped_source}")
    for snippet in RESOLVED_TODO_SNIPPETS:
        if snippet in text:
            problems.append(f"resolved TODO remains: {snippet[:60]}")
    for fig in (
        "ratio_shape_evolution.pdf",
        "mechanism_vs_rho.pdf",
        "accuracy_equity_trajectories_inprocessing_only.pdf",
        "predictive_metric_paths.pdf",
        "level_uniformity_paths.pdf",
        "vertical_equity_metric_paths.pdf",
        "cv_predictive_metric_paths.pdf",
        "cv_level_uniformity_paths.pdf",
        "cv_vertical_equity_metric_paths.pdf",
        "cv_mechanism_metric_paths.pdf",
        "ratio_shape_cv_transition_span_only.pdf",
        "tradeoff_equity_vs_accuracy_heldout.pdf",
        "tradeoff_equity_vs_accuracy_2025.pdf",
        "tradeoff_mechanism_vs_accuracy_heldout.pdf",
        "tradeoff_mechanism_vs_accuracy_2025.pdf",
        "paper_transition_event_locations.pdf",
        "vertical_equity_event_locations.pdf",
        "mechanism_event_locations.pdf",
        "baseline_models_motivation_2024_2025.pdf",
        "vei_percentile_group_profile.pdf",
    ):
        if not (IMG / fig).is_file():
            problems.append(f"missing figure {fig}")
    forbidden = (
        "sweet spot",
        "safe range",
        "operationally safe",
        "recommended range",
        "preferred range",
        "optimal range",
        "selected range",
    )
    latest_bodies = _macro_bodies(text, "latesttext")
    for body in latest_bodies:
        low = body.lower()
        if "sweet spot" in low:
            problems.append("latesttext sweet spot")
        if "operationally safe" in low:
            problems.append("latesttext operationally safe")
        if "safe range" in low:
            problems.append("latesttext safe range")
        if "optimal range" in low:
            problems.append("latesttext optimal range")
        if "preferred range" in low:
            problems.append("latesttext preferred range")
        if "recommended range" in low and "not" not in low and "never" not in low:
            problems.append("latesttext recommended range")
        if "selected range" in low and "not" not in low and "never" not in low and "do not" not in low:
            problems.append("latesttext selected range")
    refs = re.findall(r"\\(?:eq)?ref\{([^}]+)\}", text)
    labelset = set(labels)
    preexisting_unresolved = {
        "fig:baseline_value_groups",
        "sec:model_selection",
        "subsec:model_specific_implementations",
    }
    missing_refs = sorted({r for r in refs if r not in labelset and r not in preexisting_unresolved})
    if missing_refs:
        problems.append(f"unresolved refs: {missing_refs[:20]}")
    return {"problems": problems, "n_labels": len(labels), "n_latesttext": len(latest_bodies)}


def _macro_bodies(text: str, name: str) -> list[str]:
    bodies = []
    needle = "\\" + name + "{"
    i = 0
    while True:
        j = text.find(needle, i)
        if j < 0:
            break
        k = j + len(needle)
        depth = 1
        while k < len(text) and depth:
            if text[k] == "{":
                depth += 1
            elif text[k] == "}":
                depth -= 1
            k += 1
        bodies.append(text[j + len(needle) : k - 1])
        i = k
    return bodies


if __name__ == "__main__":
    rc = 1
    try:
        qa = apply_manuscript()
        print(qa)
        rc = 0 if not qa["problems"] else 1
    except Exception:
        import traceback

        traceback.print_exc()
        rc = 1
    finally:
        os._exit(rc)
