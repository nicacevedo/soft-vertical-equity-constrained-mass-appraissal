#!/usr/bin/env python3
"""LB Stage 5 -- paper-ready numerical assets, the v15 comparison, and typeset snippets.

Emits
  tables/lb_table1_main_baseline.csv          full-precision main table (4 predictive + 4 equity)
  tables/lb_table2_complementary_baseline.csv full-precision companion (level, COD/COV, mechanism)
  tables/lb_v15_comparison.csv                every v15 displayed baseline value vs the recomputation
  snippets/*.tex                              drop-in table and figure snippets, labels preserved

Writes nothing under paper/.  Proposes text; applies none of it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lb_common as lb

MAIN = [("R2_price", r"$R^2_P$", "{:.3f}"),
        ("MAE_price", r"$\operatorname{MAE}_P$", "\\${:,.0f}"),
        ("MAPE", r"$\operatorname{MAPE}_P$", "{:.1%}"),
        ("RMSE_log", r"$\operatorname{RMSE}_{\log P}$", "{:.3f}")]
EQUITY = [("PRD", r"PRD", "{:.3f}"), ("PRB", r"PRB", "{:.3f}"),
          ("MKI", r"MKI", "{:.3f}"), ("VEI", r"VEI", "{:.1f}\\%")]
LEVEL = [("median_ratio", r"Median ratio $m_S$", "{:.3f}"),
         ("mean_ratio", r"Mean ratio $\bar r_S$", "{:.3f}"),
         ("weighted_mean_ratio", r"Weighted mean $\bar r_{W,S}$", "{:.3f}")]
UNIF = [("COD", r"COD", "{:.1f}\\%"), ("COV", r"COV", "{:.1f}\\%")]
MECH = [("beta_log", r"$\beta_{\log}$", "{:.3f}"),
        ("Delta_NL", r"$\Delta_{\mathrm{NL}}$", "{:.3f}"),
        ("dCor_e_y", r"$\operatorname{dCor}(e,y)$", "{:.3f}")]

# v15 displayed values, transcribed from paper/paper_v15.tex
# (tab:ccao_baseline_results at l.1236-1260; the oldrevisionblock complementary table at l.1276-1310).
V15 = {
    ("Linear", "heldout"): {"R2_price": 0.799, "MAE_price": 90092, "MAPE": 0.241, "RMSE_log": 0.322,
                            "PRD": 1.042, "PRB": -0.016, "MKI": 0.975, "VEI": -11.6,
                            "median_ratio": 0.969, "mean_ratio": 1.020, "weighted_mean_ratio": 0.979,
                            "COD": 24.7, "COV": 45.2, "beta_log": -0.092, "Delta_NL": 0.131,
                            "dCor_e_y": 0.250},
    ("LightGBM", "heldout"): {"R2_price": 0.894, "MAE_price": 75655, "MAPE": 0.212, "RMSE_log": 0.289,
                              "PRD": 1.069, "PRB": -0.091, "MKI": 0.923, "VEI": -26.5,
                              "median_ratio": 0.929, "mean_ratio": 0.989, "weighted_mean_ratio": 0.924,
                              "COD": 21.6, "COV": 39.7, "beta_log": -0.150, "Delta_NL": 0.119,
                              "dCor_e_y": 0.387},
    ("Linear", "forward_2025"): {"R2_price": 0.799, "MAE_price": 99371, "MAPE": 0.249, "RMSE_log": 0.313,
                                 "PRD": 1.052, "PRB": -0.029, "MKI": 0.954, "VEI": -17.3,
                                 "median_ratio": 1.020, "mean_ratio": 1.081, "weighted_mean_ratio": 1.027,
                                 "COD": 24.3, "COV": 42.1, "beta_log": -0.109, "Delta_NL": 0.122,
                                 "dCor_e_y": 0.269},
    ("LightGBM", "forward_2025"): {"R2_price": 0.904, "MAE_price": 78484, "MAPE": 0.208, "RMSE_log": 0.278,
                                   "PRD": 1.079, "PRB": -0.106, "MKI": 0.907, "VEI": -28.6,
                                   "median_ratio": 0.950, "mean_ratio": 1.015, "weighted_mean_ratio": 0.941,
                                   "COD": 21.3, "COV": 37.2, "beta_log": -0.164, "Delta_NL": 0.121,
                                   "dCor_e_y": 0.422},
}
# how each v15 cell was rounded for display
DISPLAY = {"R2_price": 3, "MAE_price": 0, "MAPE": 3, "RMSE_log": 3, "PRD": 3, "PRB": 3,
           "MKI": 3, "VEI": 1, "median_ratio": 3, "mean_ratio": 3, "weighted_mean_ratio": 3,
           "COD": 1, "COV": 1, "beta_log": 3, "Delta_NL": 3, "dCor_e_y": 3}
PCT_STORED_AS_FRACTION = {"COV"}


def short(model: str) -> str:
    return "Linear" if model.startswith("Ordinary linear") else "LightGBM"


def fmt(metric: str, v: float, spec: str) -> str:
    """Format a cell and leave LaTeX-safe output: every percent sign escaped exactly once."""
    if metric == "COV":
        v = v * 100.0
    out = spec.format(v)
    return out.replace("\\%", "%").replace("%", "\\%")


def main() -> int:
    M = pd.read_csv(lb.TABLES / "lb_metrics_full_precision_all_blocks.csv")
    M["m"] = M.model.map(short)
    oos = M[M.kind == "out_of_time"]

    def cell(model_short: str, ev: str, metric: str) -> float:
        return float(oos[(oos.m == model_short) & (oos.evaluation == ev)].iloc[0][metric])

    # ---------------- Table 1 and Table 2 (machine-readable, full precision) --
    t1 = oos[["model", "config_id", "evaluation", "n", "n_fitting_block", "prediction_sha256",
              "R2_price", "MAE_price", "MAPE", "RMSE_log", "PRD", "PRB", "MKI", "VEI"]].copy()
    t1.insert(0, "table", "main: four predictive + four vertical-equity measures")
    t1["guidance_status_PRD_PRB"] = lb.ADOPTED_GUIDANCE
    t1["guidance_status_MKI_VEI"] = lb.ED2_GUIDANCE
    t1["compliance_claim"] = "none; point estimates do not establish standards compliance"
    lb.write_table(t1, lb.TABLES / "lb_table1_main_baseline.csv")

    t2 = oos[["model", "config_id", "evaluation", "n", "n_fitting_block",
              "median_ratio", "mean_ratio", "weighted_mean_ratio", "COD", "COV",
              "beta_log", "Cov_log_residual_log_price", "Delta_NL", "Delta_NL_raw",
              "dCor_e_y"]].copy()
    t2.insert(0, "table", "companion: valuation level, uniformity, mechanism/residual structure")
    t2["COV_units"] = "stored as a fraction; multiply by 100 for the displayed percent"
    t2["guidance_status_level_COD"] = lb.ADOPTED_GUIDANCE
    t2["guidance_status_beta_dNL_dCor"] = "no assessor reference range; residual-structure diagnostics"
    lb.write_table(t2, lb.TABLES / "lb_table2_complementary_baseline.csv")

    # ---------------- v15 comparison -----------------------------------------
    rows = []
    for (ms, ev), vals in V15.items():
        for metric, v15v in vals.items():
            rec = cell(ms, ev, metric)
            disp = rec * 100.0 if metric in PCT_STORED_AS_FRACTION else rec
            r = round(disp, DISPLAY[metric])
            rows.append({
                "model": ms, "evaluation": ev, "metric": metric,
                "v15_displayed": v15v, "recomputed_full_precision": rec,
                "recomputed_rounded_to_v15_display": r,
                "difference_at_v15_display_precision": round(r - v15v, 6),
                "matches_v15": bool(abs(r - v15v) < 10 ** (-DISPLAY[metric]) / 2 + 1e-12),
            })
    v15 = pd.DataFrame(rows)
    lb.write_table(v15, lb.TABLES / "lb_v15_comparison.csv")
    n_diff = int((~v15.matches_v15).sum())

    # ---------------- LaTeX snippets -----------------------------------------
    def row(label, metric, spec, indent=True):
        pre = "\\baselineprimarymetric{" if indent else "{"
        return (f"{pre}{label}}} & {fmt(metric, cell('Linear', EVAL, metric), spec)} "
                f"& {fmt(metric, cell('LightGBM', EVAL, metric), spec)} \\\\")

    panels = []
    for EVAL, title, fitnote in (
            ("heldout", "Panel A. Held-out evaluation ($n=38{,}290$; fit on the 344,607-sale development pool)", ""),
            ("forward_2025", "Panel B. 2025 forward evaluation ($n=26{,}641$; refit on all 382,897 eligible 2016--2024 sales)", "")):
        lines = [f"\\multicolumn{{3}}{{@{{}}l}}{{\\textbf{{{title}}}}} \\\\",
                 "\\cmidrule(r){1-1}",
                 "\\multicolumn{3}{@{}l}{\\emph{Prediction}} \\\\"]
        lines += [row(lab, k, sp) for k, lab, sp in MAIN]
        lines += ["\\addlinespace[3pt]",
                  "\\multicolumn{3}{@{}l}{\\emph{Vertical equity}} \\\\"]
        lines += [row(lab, k, sp) for k, lab, sp in EQUITY]
        panels.append("\n".join(lines))

    tab1 = r"""% ---------------------------------------------------------------------------
% PROPOSED REPLACEMENT for tab:ccao_baseline_results (paper/paper_v20.tex, l.617-654).
% Label preserved, so \ref{tab:ccao_baseline_results} keeps resolving.
% Source: analysis/linear_baseline_reproduction_v1/tables/lb_table1_main_baseline.csv
% Provenance: frozen cached predictions, config_id fd63507d2456c789 (linear) and
% 252a25d9c0ce796b (ordinary LightGBM), the latter bitwise identical to the frozen
% zero-penalty benchmark, cell A of
% analysis/p0_major_revision_validation/tables/zero_control_full.csv.
% ---------------------------------------------------------------------------
{\begin{table}[!ht]
\centering
\scriptsize
\setlength{\tabcolsep}{2.4pt}
\renewcommand{\arraystretch}{1.06}
\newcommand{\baselineprimarymetric}[1]{\hspace*{0.6em}#1}
\caption{Prediction and vertical-equity measures for the ordinary linear-regression
benchmark and ordinary LightGBM (standard raw-label native) on the held-out and 2025
forward evaluations. Both are unpenalized; neither is a CCAO production model.}
\label{tab:ccao_baseline_results}
\begin{tabularx}{\textwidth}{@{} >{\raggedright\arraybackslash}p{6.4cm} >{\centering\arraybackslash}X >{\centering\arraybackslash}X @{}}
\toprule
\textbf{Measure} & \textbf{Ordinary linear} & \textbf{Ordinary LightGBM} \\
\midrule
__PANEL_A__
\addlinespace[5pt]
__PANEL_B__
\bottomrule
\end{tabularx}
\vspace{1mm}
\begin{minipage}{\textwidth}
\scriptsize
\emph{Notes.}
The LightGBM benchmark is defined in Section~\ref{subsec:comparators}. The linear column is an
ordinary least-squares fit of log sale price, estimated on the same samples under the same
temporal design; it is a research benchmark, not a reconstruction of the office's historical
township-level production models, and no claim is made about how those models performed.
The guidance status and reference ranges are in Table~\ref{tab:assessment_metrics_summary};
MKI and VEI use proposed exposure-draft guidance, PRD and PRB adopted guidance. These point
estimates do not establish standards compliance or statistical significance. MAPE and VEI are
in percent; MAE is in dollars. Both models use direct exponentiation to return log predictions
to dollars; Appendix~\ref{app:smearing} records the alternative, under which the accuracy
ordering between these two columns is unchanged.
The two model classes share the outcome, the sales, the splits and the metric code, but not the
predictor representation: the linear pipeline drops two high-cardinality string location fields
and adds target encodings, Box-Cox transforms and squared terms, so the accuracy difference is
not a pure functional-form contrast (Appendix~\ref{app:implementation}).
\end{minipage}
\end{table}

}
"""
    tab1 = tab1.replace("__PANEL_A__", panels[0]).replace("__PANEL_B__", panels[1])
    lb.write_text(lb.SNIPPETS / "tab_ccao_baseline_results_panelled.tex", tab1)

    # wide (v15-style) alternative
    def wrow(label, metric, spec):
        return ("\\baselineprimarymetric{" + label + "} & "
                + fmt(metric, cell("Linear", "heldout", metric), spec) + " & "
                + fmt(metric, cell("LightGBM", "heldout", metric), spec) + " & "
                + fmt(metric, cell("Linear", "forward_2025", metric), spec) + " & "
                + fmt(metric, cell("LightGBM", "forward_2025", metric), spec) + " \\\\")

    wide_body = "\n".join(
        ["\\multicolumn{5}{@{}l}{\\textbf{Prediction}} \\\\", "\\cmidrule(r){1-1}"]
        + [wrow(lab, k, sp) for k, lab, sp in MAIN]
        + ["\\addlinespace[5pt]", "\\multicolumn{5}{@{}l}{\\textbf{Vertical equity}} \\\\",
           "\\cmidrule(r){1-1}"]
        + [wrow(lab, k, sp) for k, lab, sp in EQUITY])
    tab1w = r"""% ---------------------------------------------------------------------------
% ALTERNATIVE layout for tab:ccao_baseline_results: the v15-style four-column grid.
% Same numbers as tab_ccao_baseline_results_panelled.tex; use whichever reads better.
% ---------------------------------------------------------------------------
{\begin{table}[!ht]
\centering
\scriptsize
\setlength{\tabcolsep}{2.4pt}
\renewcommand{\arraystretch}{1.06}
\newcommand{\baselineprimarymetric}[1]{\hspace*{0.6em}#1}
\caption{Prediction and vertical-equity measures for the ordinary linear-regression benchmark
and ordinary LightGBM (standard raw-label native) on the held-out and 2025 forward evaluations.
Both are unpenalized; neither is a CCAO production model.}
\label{tab:ccao_baseline_results}
\begin{tabularx}{\textwidth}{@{} >{\raggedright\arraybackslash}p{2.80cm} >{\centering\arraybackslash}X >{\centering\arraybackslash}X >{\centering\arraybackslash}X >{\centering\arraybackslash}X @{}}
\toprule
\textbf{Measure} & \multicolumn{2}{c}{\textbf{Held-out evaluation}} & \multicolumn{2}{c}{\textbf{2025 forward evaluation}} \\
\cmidrule(lr){2-3}\cmidrule(l){4-5}
& \textbf{Linear} & \textbf{\shortstack{Light\\GBM}} & \textbf{Linear} & \textbf{\shortstack{Light\\GBM}} \\
\midrule
__BODY__
\bottomrule
\end{tabularx}
\vspace{1mm}
\begin{minipage}{\textwidth}
\scriptsize
\emph{Notes.}
As in the panelled layout: same predictions, same metric code, no boldface ranking.
For the held-out evaluation both models are fit on the 344,607-sale development pool; for the
2025 forward evaluation both are refit on all 382,897 eligible 2016--2024 sales.
\end{minipage}
\end{table}

}
"""
    lb.write_text(lb.SNIPPETS / "tab_ccao_baseline_results_wide.tex",
                  tab1w.replace("__BODY__", wide_body))

    # companion table
    def crow(label, metric, spec):
        return ("\\baselinecompmetric{" + label + "} & "
                + fmt(metric, cell("Linear", "heldout", metric), spec) + " & "
                + fmt(metric, cell("LightGBM", "heldout", metric), spec) + " & "
                + fmt(metric, cell("Linear", "forward_2025", metric), spec) + " & "
                + fmt(metric, cell("LightGBM", "forward_2025", metric), spec) + " \\\\")

    comp_body = "\n".join(
        ["\\multicolumn{5}{@{}l}{\\textbf{Valuation level}} \\\\", "\\cmidrule(r){1-1}"]
        + [crow(lab, k, sp) for k, lab, sp in LEVEL]
        + ["\\addlinespace[5pt]", "\\multicolumn{5}{@{}l}{\\textbf{Horizontal uniformity}} \\\\",
           "\\cmidrule(r){1-1}"]
        + [crow(lab, k, sp) for k, lab, sp in UNIF]
        + ["\\addlinespace[5pt]",
           "\\multicolumn{5}{@{}l}{\\textbf{Mechanism and residual structure}} \\\\",
           "\\cmidrule(r){1-1}"]
        + [crow(lab, k, sp) for k, lab, sp in MECH])
    tab2 = r"""% ---------------------------------------------------------------------------
% PROPOSED REPLACEMENT for tab:ccao_baseline_complementary (paper/paper_v20.tex, l.2359-2400).
% Label preserved.  Source: analysis/linear_baseline_reproduction_v1/tables/
% lb_table2_complementary_baseline.csv
% ---------------------------------------------------------------------------
\begin{table}[!htbp]
\centering
\scriptsize
\setlength{\tabcolsep}{2.4pt}
\renewcommand{\arraystretch}{1.06}
\newcommand{\baselinecompmetric}[1]{\hspace*{0.6em}#1}
\caption{Complementary baseline diagnostics for the ordinary linear benchmark and
\emph{Ordinary LightGBM (standard raw-label native)}: valuation level, horizontal uniformity,
and mechanism/residual structure. Descriptive levels for two unpenalized specifications, not a
recommendation between them.}
\label{tab:ccao_baseline_complementary}
\begin{tabularx}{\textwidth}{@{} >{\raggedright\arraybackslash}p{2.80cm} >{\centering\arraybackslash}X >{\centering\arraybackslash}X >{\centering\arraybackslash}X >{\centering\arraybackslash}X @{}}
\toprule
\textbf{Measure} & \multicolumn{2}{c}{\textbf{Held-out evaluation}} & \multicolumn{2}{c}{\textbf{2025 forward evaluation}} \\
\cmidrule(lr){2-3}\cmidrule(l){4-5}
& \textbf{Linear} & \textbf{\shortstack{Light\\GBM}} & \textbf{Linear} & \textbf{\shortstack{Light\\GBM}} \\
\midrule
__BODY__
\bottomrule
\end{tabularx}
\vspace{1mm}
\begin{minipage}{\textwidth}
\scriptsize
\emph{Notes.}
Median, mean, weighted-mean ratio, COD, and COV are assessor-facing valuation-level or
horizontal-uniformity diagnostics defined in Section~\ref{subsec:metrics}. The first-order
mechanism slope $\beta_{\log}$, correlation-ratio nonlinearity gap $\Delta_{\mathrm{NL}}$, and
distance correlation are defined in Eqs.~\eqref{eq:log_ratio_slope},
\eqref{eq:nonlinearity_gap}, and~\eqref{eq:dcor_diagnostic}; the last two carry no
assessor-standard reference range and are residual-structure diagnostics rather than equity
measures.
Both columns' COD exceeds the $[5,15]$ range the adopted Standard on Ratio Studies gives for
single-family residential property (Table~\ref{tab:assessment_metrics_summary}), and COD rises
further along the penalty paths (Appendix Table~\ref{tab:path_anchor_complementary}). No
compliance claim is implied anywhere in this paper. This range comparison is a descriptive
interpretation of the reported values rather than a formal inferential finding.
The PRB and VEI value proxy is model-specific by construction: it is
$0.5\,P_i+0.5\,\widehat P_i/m_S$, so each column is read against a value axis that embeds that
column's own valuations and its own median ratio. The two columns are therefore not ranked on a
shared value axis.
COD and COV are reported in percent; COV is stored as a fraction in the retained artifact and
converted to percent for display.
\end{minipage}
\end{table}
"""
    lb.write_text(lb.SNIPPETS / "tab_ccao_baseline_complementary.tex",
                  tab2.replace("__BODY__", comp_body))

    fig = r"""% ---------------------------------------------------------------------------
% PROPOSED REPLACEMENT for the fig:baseline_motivation float (paper/paper_v20.tex, l.656-662).
% Label preserved.  The new graphic must first be copied to
%   paper/img/generated_v20_linear_baseline/baseline_models_motivation_linear_vs_lgbm_2024_2025_v20r1.pdf
% from analysis/linear_baseline_reproduction_v1/figures/.  Do NOT overwrite
% img/generated_v12_994/baseline_models_motivation_2024_2025.pdf: that asset is the
% LightGBM-only figure produced by the P1-1 vector surgery in commit c630994d.
% ---------------------------------------------------------------------------
\begin{figure}[!htbp]
\centering
\safeincludegraphics[width=0.8\textwidth]{img/generated_v20_linear_baseline/baseline_models_motivation_linear_vs_lgbm_2024_2025_v20r1.pdf}
\caption{Descriptive valuation-ratio profiles against sale price for the ordinary linear
benchmark and ordinary LightGBM, in the held-out and 2025 forward evaluations. Curves show
median ratios in 30 equal-count sale-price bins; shaded regions show the corresponding
interquartile ranges. Binning, axis limits and the $\beta_{\log}$ annotation follow one
convention across all four panels. The panels describe observed sales; they are not a
standards-compliance finding and not evidence that either model class is intrinsically more
or less regressive.}
\label{fig:baseline_motivation}
\end{figure}
"""
    lb.write_text(lb.SNIPPETS / "fig_baseline_motivation.tex", fig)

    lb.write_json(lb.PROVENANCE / "lb5_assets.json", {
        "v15_cells_compared": int(len(v15)),
        "v15_cells_differing_at_displayed_precision": n_diff,
        "v15_differences": v15[~v15.matches_v15].to_dict(orient="records"),
        "snippets": sorted(p.name for p in lb.SNIPPETS.glob("*.tex")),
        "labels_preserved": ["tab:ccao_baseline_results", "tab:ccao_baseline_complementary",
                             "fig:baseline_motivation"],
    })
    print(f"[lb5] v15 comparison: {len(v15)} cells, {n_diff} differ at v15 display precision")
    if n_diff:
        print(v15[~v15.matches_v15].to_string(index=False))
    print("[lb5] snippets:", ", ".join(sorted(p.name for p in lb.SNIPPETS.glob('*.tex'))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
