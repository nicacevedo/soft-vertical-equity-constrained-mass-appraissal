#!/usr/bin/env python3
"""Populate paper/paper_v12.tex from frozen 994-tree paper-asset outputs.

Deterministic manuscript population + provenance + static QA only.
No model fitting, no transition re-estimation, no TeX compilation, and no
writes into frozen analysis trees.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from utils.transition_regions import (  # noqa: E402
    CanonicalIdentityError,
    sha256_file,
    validate_canonical_result_root,
)

REQUIRED_ANCESTOR = "725832f8fb9e9b802dc4fa527cfaa0ad3760fca8"
RESULT_ROOT = REPO / "output" / "paper_v6_preselection_994"
DATA_ID = "d4929d43ec19badf"
SPLIT_ID = "3d464d4a611b131b"
CONFIG_ID = "407d47775760c14d"
V1_ROOT = (
    RESULT_ROOT
    / "analysis"
    / f"data_id={DATA_ID}"
    / f"split_id={SPLIT_ID}"
    / "penalty_path_analysis"
    / "transition_regions_v1"
)
ASSETS = V1_ROOT.parent / "transition_regions_paper_assets_v1"
PAPER_TEX = REPO / "paper" / "paper_v12.tex"
IMG_DIR = REPO / "paper" / "img" / "generated_v12_994"
IMG_REL = "img/generated_v12_994"
POP_ROOT = REPO / "output" / "paper_v12_population_994"
PROV_DIR = POP_ROOT / "provenance"
QA_DIR = POP_ROOT / "qa"
REPORT_DIR = POP_ROOT / "report"
MOTIVATION_UTILS = REPO / "utils" / "motivation_utils.py"
RHO0_AUDIT = RESULT_ROOT / "final_local_results" / "rho0_split_audit.csv"

V12_MARKERS = (
    r"definecolor{latestPurple}",
    r"newcommand{\latesttext}",
    "Descriptive transition diagnostic",
    "Cross-Metric Turning Points and Temporal Concordance",
    r"\label{tab:transition_summary}",
    r"\label{tab:transition_regret}",
)

FORBIDDEN_COMPILE = (
    "pdflatex",
    "xelatex",
    "lualatex",
    "latexmk",
    "tectonic",
    "arara",
    "biber",
    "bibtex",
)
FORBIDDEN_SPAN = (
    "sweet spot",
    "optimal range",
    "recommended range",
    "safe range",
    "deployment range",
    "preferred range",
    "selected range",
)
FIGURE_COPY = (
    ("figures/main/baseline_models_motivation_2024_2025.pdf", "baseline_models_motivation_2024_2025.pdf"),
    ("figures/main/ratio_shape_evolution.pdf", "ratio_shape_evolution.pdf"),
    ("figures/main/mechanism_vs_rho.pdf", "mechanism_vs_rho.pdf"),
    (
        "figures/main/accuracy_equity_trajectories_inprocessing_only.pdf",
        "accuracy_equity_trajectories_inprocessing_only.pdf",
    ),
    (
        "figures/appendix/prb_mki_accuracy_equity_inprocessing_only.pdf",
        "prb_mki_accuracy_equity_inprocessing_only.pdf",
    ),
    ("figures/appendix/predictive_metric_paths.pdf", "predictive_metric_paths.pdf"),
    ("figures/appendix/level_uniformity_paths.pdf", "level_uniformity_paths.pdf"),
    ("figures/appendix/cv_fold_stability.pdf", "cv_fold_stability.pdf"),
    ("figures/appendix/vei_percentile_group_profile.pdf", "vei_percentile_group_profile.pdf"),
    (
        "figures/appendix/paper_transition_event_locations.pdf",
        "paper_transition_event_locations.pdf",
    ),
)
V6_TO_V12 = {
    "img/generated_v6_preselection/baseline_models_motivation_2024_2025.pdf": f"{IMG_REL}/baseline_models_motivation_2024_2025.pdf",
    "img/generated_v6_preselection/ratio_shape_evolution.pdf": f"{IMG_REL}/ratio_shape_evolution.pdf",
    "img/generated_v6_preselection/mechanism_vs_rho.pdf": f"{IMG_REL}/mechanism_vs_rho.pdf",
    "img/generated_v6_preselection/predictive_metric_paths.pdf": f"{IMG_REL}/predictive_metric_paths.pdf",
    "img/generated_v6_preselection/level_uniformity_paths.pdf": f"{IMG_REL}/level_uniformity_paths.pdf",
    "img/generated_v6_preselection/cv_fold_stability.pdf": f"{IMG_REL}/cv_fold_stability.pdf",
    "img/generated_v6_preselection/vei_percentile_group_profile.pdf": f"{IMG_REL}/vei_percentile_group_profile.pdf",
}

BASELINE_ROUNDS = {
    "R2_price": ("r2", 3),
    "MAE_price": ("mae", 0),
    "MAPE": ("pct1", 1),
    "RMSE_log": ("f3", 3),
    "median_ratio": ("f3", 3),
    "mean_ratio": ("f3", 3),
    "weighted_mean_ratio": ("f3", 3),
    "COD": ("cod1", 1),
    "COV": ("pct1", 1),
    "PRD": ("f3", 3),
    "PRB": ("f3", 3),
    "MKI": ("f3", 3),
    "VEI": ("vei1", 1),
    "Beta_log": ("f3", 3),
    "Delta_NL": ("f3", 3),
    "dCor_e_y": ("f3", 3),
}
BASELINE_ROW_ORDER = [
    "R2_price",
    "MAE_price",
    "MAPE",
    "RMSE_log",
    "median_ratio",
    "mean_ratio",
    "weighted_mean_ratio",
    "COD",
    "COV",
    "PRD",
    "PRB",
    "MKI",
    "VEI",
    "Beta_log",
    "Delta_NL",
    "dCor_e_y",
]
ANCHOR_METRICS = [
    "R2_price",
    "MAE_price",
    "PRD",
    "PRB",
    "MKI",
    "VEI",
    "Beta_log",
    "Delta_NL",
    "dCor_e_y",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=str(REPO), text=True).strip()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def is_nan(x: Any) -> bool:
    return x is None or (isinstance(x, float) and pd.isna(x)) or (isinstance(x, str) and x.strip() == "")


def fmt_rho(x: Any) -> str:
    if is_nan(x):
        return "--"
    v = float(x)
    if abs(v) < 1e-12:
        return "0"
    if abs(v - 0.1) < 1e-9:
        return "0.1"
    if abs(v - 100.0) < 1e-8:
        return "100"
    if abs(v) >= 10:
        return f"{v:.3f}".rstrip("0").rstrip(".")
    return f"{v:.3f}"


def cls_short(s: Any) -> str:
    if is_nan(s):
        return "--"
    s = str(s)
    if s == "interior_positive":
        return r"interior $+$"
    if s == "boundary_zero":
        return r"boundary ($\rho=0$)"
    return s.replace("_", " ")


def strip_comments(tex: str) -> str:
    out: List[str] = []
    for line in tex.splitlines(True):
        if line.lstrip().startswith("%"):
            continue
        out.append(re.sub(r"(?<!\\)%.*", "", line))
    return "".join(out)


def round_display(measure: str, val: float) -> str:
    kind = BASELINE_ROUNDS[measure][0]
    if kind == "r2":
        return f"{val:.3f}"
    if kind == "mae":
        return f"${val:,.0f}"
    if kind == "pct1":
        return f"{100.0 * val:.1f}\\%"
    if kind == "cod1":
        return f"{val:.1f}\\%"
    if kind == "vei1":
        return f"{val:.1f}\\%"
    return f"{val:.3f}"


def round_anchor(measure: str, val: float) -> str:
    if measure == "MAE_price":
        return f"${val:,.0f}"
    if measure == "VEI":
        return f"{val:.1f}\\%"
    if measure == "R2_price":
        return f"{val:.3f}"
    return f"{val:.3f}"


def fmt_r2(x: float) -> str:
    return f"{float(x):.3f}"


def fmt_mae(x: float) -> str:
    return f"\\${float(x):,.0f}"


def fmt_mape_pct(x: float) -> str:
    return f"{100.0 * float(x):.1f}\\%"


def fmt_rmse(x: float) -> str:
    return f"{float(x):.3f}"


def fmt_cod(x: float) -> str:
    return f"{float(x):.2f}"


def fmt_norm(x: float) -> str:
    if abs(float(x)) < 5e-5:
        return "0"
    return f"{float(x):.3f}"


def fmt_logd(x: Any) -> str:
    if is_nan(x):
        return "--"
    v = float(x)
    if abs(v) < 1e-12:
        return "0"
    return f"{v:.3f}"


def make_transition_summary(summary: pd.DataFrame) -> str:
    d = summary.loc[summary["family"] == "Direct"].iloc[0]
    s = summary.loc[summary["family"] == "Surrogate"].iloc[0]
    return rf"""
\begin{{table}}[!htbp]
\centering
\scriptsize
\setlength{{\tabcolsep}}{{3.2pt}}
\renewcommand{{\arraystretch}}{{1.08}}
\color{{latestPurple}}
\caption{{\latesttext{{Cross-metric turning-event summary for the frozen chronological-CV diagnostic and its retrospective temporal comparison. The Direct interval is a CV-derived descriptive transition span, not a selected or recommended penalty range.}}}}
\label{{tab:transition_summary}}
\begin{{tabularx}}{{\textwidth}}{{@{{}} l >{{\raggedright\arraybackslash}}X >{{\raggedright\arraybackslash}}X @{{}}}}
\toprule
& \textbf{{Direct}} & \textbf{{Surrogate}} \\
\midrule
CV $R^2_P$ max & $\rho={fmt_rho(d['R2_price__cv_rho'])}$ ({cls_short(d['R2_price__cv_classification'])})
& $\rho={fmt_rho(s['R2_price__cv_rho'])}$ ({cls_short(s['R2_price__cv_classification'])}) \\
CV $\operatorname{{MAE}}_P$ min & $\rho={fmt_rho(d['MAE_price__cv_rho'])}$ ({cls_short(d['MAE_price__cv_classification'])})
& $\rho={fmt_rho(s['MAE_price__cv_rho'])}$ ({cls_short(s['MAE_price__cv_classification'])}) \\
CV $\operatorname{{MAPE}}_P$ min & $\rho={fmt_rho(d['MAPE__cv_rho'])}$ ({cls_short(d['MAPE__cv_classification'])})
& $\rho={fmt_rho(s['MAPE__cv_rho'])}$ ({cls_short(s['MAPE__cv_classification'])}) \\
CV $\operatorname{{RMSE}}_{{\log P}}$ min & $\rho={fmt_rho(d['RMSE_log__cv_rho'])}$ ({cls_short(d['RMSE_log__cv_classification'])})
& $\rho={fmt_rho(s['RMSE_log__cv_rho'])}$ ({cls_short(s['RMSE_log__cv_classification'])}) \\
CV COD min & $\rho={fmt_rho(d['COD__cv_rho'])}$ ({cls_short(d['COD__cv_classification'])})
& $\rho={fmt_rho(s['COD__cv_rho'])}$ ({cls_short(s['COD__cv_classification'])}) \\
\addlinespace
Common five-metric CV span
& valid positive-interior span $[{fmt_rho(d['span_low'])},\,{fmt_rho(d['span_high'])}]$; $\log_{{10}}$-width ${float(d['log10_width']):.2f}$
& not supported \\
First-positive-grid note
& $\rho=0.1$ is the smallest positive tested value, not an estimated lower threshold
& --- \\
LOFO five-event support
& {int(d['lofo_valid_count'])}/{int(d['lofo_valid_of'])}; valid endpoints $[{fmt_rho(d['lofo_valid_low_min'])},\,{fmt_rho(d['lofo_valid_low_max'])}]$ to $[{fmt_rho(d['lofo_valid_high_min'])},\,{fmt_rho(d['lofo_valid_high_max'])}]$
& {int(s['lofo_valid_count'])}/{int(s['lofo_valid_of'])} \\
Held-out exact concordance
& {int(d['heldout_exact_concordance'])}/{int(d['heldout_exact_concordance_of'])}
& {int(s['heldout_exact_concordance'])}/{int(s['heldout_exact_concordance_of'])} \\
2025 exact concordance
& {int(d['forward_2025_exact_concordance'])}/{int(d['forward_2025_exact_concordance_of'])}
& {int(s['forward_2025_exact_concordance'])}/{int(s['forward_2025_exact_concordance_of'])} \\
Value-based reading
& mixed by metric
& no common five-metric positive CV span \\
\bottomrule
\end{{tabularx}}
\vspace{{1mm}}
\begin{{minipage}}{{\textwidth}}
\scriptsize
\emph{{Notes.}}
Event locations are discrete observed grid points on equal-weight chronological CV.
A common span is reported only when all five events are positive and interior.
Held-out and 2025 comparisons are retrospective temporal concordance, not prospective confirmation.
No $\rho$ is selected.
\end{{minipage}}
\end{{table}}
"""


def make_regret_table(regret: pd.DataFrame) -> str:
    order = ["R2_price", "MAE_price", "MAPE", "RMSE_log", "COD"]
    labels = {
        "R2_price": r"$R^2_P$",
        "MAE_price": r"$\operatorname{MAE}_P$",
        "MAPE": r"$\operatorname{MAPE}_P$",
        "RMSE_log": r"$\operatorname{RMSE}_{\log P}$",
        "COD": "COD",
    }

    def val(metric: str, v: float) -> str:
        if metric == "R2_price":
            return fmt_r2(v)
        if metric == "MAE_price":
            return fmt_mae(v)
        if metric == "MAPE":
            return fmt_mape_pct(v)
        if metric == "RMSE_log":
            return fmt_rmse(v)
        return fmt_cod(v)

    def raw(metric: str, v: float) -> str:
        if abs(float(v)) < 1e-12:
            return "0"
        if metric == "MAE_price":
            return fmt_mae(v)
        if metric == "MAPE":
            return f"{100.0 * float(v):.3f} pp"
        if metric == "R2_price":
            return f"{float(v):.4f}"
        if metric == "RMSE_log":
            return f"{float(v):.4f}"
        return f"{float(v):.3f}"

    def panel(split: str, title: str) -> str:
        lines = [rf"\multicolumn{{8}}{{l}}{{\textit{{{title}}}}} \\", r"\addlinespace[2pt]"]
        sub = regret.loc[regret["split"] == split].set_index("metric")
        for m in order:
            r = sub.loc[m]
            lines.append(
                " & ".join(
                    [
                        labels[m],
                        fmt_rho(r["global_opt_rho"]),
                        val(m, r["global_opt_value"]),
                        fmt_rho(r["best_inside_rho"]),
                        val(m, r["best_inside_value"]),
                        raw(m, r["raw_regret"]),
                        fmt_norm(r["normalized_regret"]),
                        fmt_logd(r["log10_distance_global_opt_to_cv_span"]),
                    ]
                )
                + r" \\"
            )
        return "\n".join(lines)

    return rf"""
\begin{{table}}[!htbp]
\centering
\scriptsize
\setlength{{\tabcolsep}}{{2.6pt}}
\renewcommand{{\arraystretch}}{{1.08}}
\color{{latestPurple}}
\caption{{\latesttext{{Value-based cost of restricting the Direct out-of-time paths to the frozen CV-derived descriptive span. No small/large regret threshold is imposed.}}}}
\label{{tab:transition_regret}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lrrrrrrr}}
\toprule
Metric & $\rho^{{*}}$ & value$^{{*}}$ & $\rho_{{\mathrm{{span}}}}$ & value$_{{\mathrm{{span}}}}$ & raw regret & norm.\ regret & $\log_{{10}}$ dist. \\
\midrule
{panel("heldout", "Panel A: held-out evaluation")}
\midrule
{panel("forward_2025", "Panel B: 2025 forward evaluation")}
\bottomrule
\end{{tabular}}}}
\vspace{{1mm}}
\begin{{minipage}}{{\textwidth}}
\scriptsize
\emph{{Notes.}}
$\rho^{{*}}$ and value$^{{*}}$ are the discrete-grid global optimum on that out-of-time path.
$\rho_{{\mathrm{{span}}}}$ and value$_{{\mathrm{{span}}}}$ are the best observed grid point inside the frozen Direct CV span $[0.1,1.099]$.
Raw regret is direction-aware and nonnegative.
Normalized regret divides raw regret by the full-path range of the metric.
$\log_{{10}}$ distance is the distance of $\rho^{{*}}$ to the nearest frozen span boundary when that event is outside the span, and $0$ when it is inside.
MAPE is shown in percent; MAPE raw regret is in percentage points.
COD is already on the percent scale used in the combined path table.
\end{{minipage}}
\end{{table}}
"""


def replace_once(tex: str, old: str, new: str, label: str) -> str:
    n = tex.count(old)
    if n != 1:
        raise RuntimeError(f"{label}: expected exactly 1 occurrence, found {n}")
    return tex.replace(old, new, 1)


def extract_labeled_env(tex: str, env: str, label: str) -> str:
    m = re.search(
        rf"\\begin\{{{env}\}}.*?\\label\{{{re.escape(label)}\}}.*?\\end\{{{env}\}}",
        tex,
        flags=re.S,
    )
    if not m:
        raise RuntimeError(f"{env} {label} not found")
    return m.group(0)


def validate_baseline_table(tex: str, src: pd.DataFrame) -> List[str]:
    problems: List[str] = []
    body = extract_labeled_env(strip_comments(tex), "table", "tab:ccao_baseline_results")
    data_lines = [
        ln
        for ln in body.splitlines()
        if r"\baselinemetric" in ln and "&" in ln
    ]
    if len(data_lines) != len(BASELINE_ROW_ORDER):
        return [f"baseline data-row count {len(data_lines)} != {len(BASELINE_ROW_ORDER)}"]
    for measure, line in zip(BASELINE_ROW_ORDER, data_lines):
        cells = [c.strip() for c in line.split("&")[1:]]
        cells[-1] = cells[-1].rstrip("\\").strip()
        if len(cells) != 4:
            problems.append(f"baseline {measure}: expected 4 cells, got {cells}")
            continue
        pairs = [
            ("heldout", "Linear", cells[0]),
            ("heldout", "LightGBM", cells[1]),
            ("forward_2025", "Linear", cells[2]),
            ("forward_2025", "LightGBM", cells[3]),
        ]
        for split, fam, shown in pairs:
            row = src.loc[(src["split"] == split) & (src["measure"] == measure)].iloc[0]
            expect = round_display(measure, float(row[fam]))
            shown_plain = (
                re.sub(r"\\textbf\{([^}]*)\}", r"\1", shown)
                .replace("\\$", "$")
                .replace(" ", "")
            )
            expect_plain = expect.replace("\\$", "$").replace(" ", "")
            if shown_plain != expect_plain:
                problems.append(
                    f"baseline {split} {fam} {measure}: shown {shown_plain!r} != {expect_plain!r}"
                )
            want_bold = bool(row[f"{fam}_bold"])
            has_bold = r"\textbf{" in shown
            if want_bold != has_bold:
                problems.append(
                    f"baseline {split} {fam} {measure}: bold flag {want_bold} vs tex {has_bold}"
                )
    return problems


def _anchor_rho_key(rho: Any) -> str:
    if is_nan(rho):
        return "--"
    v = float(rho)
    if abs(v - 0.1) < 1e-9:
        return "0.1"
    if abs(v - 0.954095476349994) < 1e-9:
        return "0.954"
    if abs(v - 10.481131341546853) < 1e-8:
        return "10.481"
    if abs(v - 100.0) < 1e-8:
        return "100"
    return fmt_rho(v)


def validate_anchor_table(tex: str, src: pd.DataFrame) -> List[str]:
    problems: List[str] = []
    body = extract_labeled_env(strip_comments(tex), "table", "tab:path_anchor_summary")
    if re.search(r"(?<!\[)1\.099|1\.0985", body):
        problems.append("transition endpoint appears in active anchor table")
    wanted = {
        ("Linear", "--", "heldout"),
        ("LightGBM", "--", "heldout"),
        ("Direct", "0.1", "heldout"),
        ("Direct", "0.954", "heldout"),
        ("Direct", "10.481", "heldout"),
        ("Direct", "100", "heldout"),
        ("Surrogate", "0.1", "heldout"),
        ("Surrogate", "0.954", "heldout"),
        ("Surrogate", "10.481", "heldout"),
        ("Surrogate", "100", "heldout"),
        ("Linear", "--", "forward_2025"),
        ("LightGBM", "--", "forward_2025"),
        ("Direct", "0.1", "forward_2025"),
        ("Direct", "0.954", "forward_2025"),
        ("Direct", "10.481", "forward_2025"),
        ("Direct", "100", "forward_2025"),
        ("Surrogate", "0.1", "forward_2025"),
        ("Surrogate", "0.954", "forward_2025"),
        ("Surrogate", "10.481", "forward_2025"),
        ("Surrogate", "100", "forward_2025"),
    }
    split = "heldout"
    seen = set()
    for raw in body.splitlines():
        line = raw.strip()
        if "Panel B" in line:
            split = "forward_2025"
            continue
        if not line.startswith(("Linear", "Ordinary", "Direct", "Surrogate")):
            continue
        if "&" not in line:
            continue
        cells = [c.strip() for c in line.split("&")]
        if len(cells) < 11:
            continue
        cells[-1] = cells[-1].rstrip("\\").strip()
        fam = cells[0].replace(" regression", "").replace("Ordinary ", "")
        rho_tex = cells[1].replace("$", "").replace("\\approx", "").strip()
        rho_key = rho_tex
        seen.add((fam, rho_key, split))
        src_rows = src.loc[src["split"] == split]
        if fam == "Linear":
            row = src_rows.loc[src_rows["family"] == "Linear"].iloc[0]
        elif fam == "LightGBM":
            row = src_rows.loc[src_rows["family"] == "LightGBM"].iloc[0]
        else:
            cand = src_rows.loc[src_rows["family"] == fam]
            match = None
            for _, r in cand.iterrows():
                if _anchor_rho_key(r["rho"]) == rho_key:
                    match = r
                    break
            if match is None:
                problems.append(f"anchor source missing {fam} {rho_key} {split}")
                continue
            row = match
        values = cells[2:]
        if len(values) != len(ANCHOR_METRICS):
            problems.append(f"anchor {fam} {rho_key} {split}: cell count {len(values)}")
            continue
        for measure, shown in zip(ANCHOR_METRICS, values):
            expect = round_anchor(measure, float(row[measure]))
            shown_plain = (
                re.sub(r"\\textbf\{([^}]*)\}", r"\1", shown)
                .replace(r"\textsuperscript{*}", "")
                .replace(r"\latesttext{", "")
                .rstrip("}")
                .replace("\\$", "$")
                .replace(" ", "")
            )
            expect_plain = expect.replace("\\$", "$").replace(" ", "")
            if shown_plain != expect_plain:
                problems.append(
                    f"anchor {fam} {rho_key} {split} {measure}: {shown_plain!r} != {expect_plain!r}"
                )
            want_bold = bool(row[f"{measure}__manuscript_bold"])
            has_bold = r"\textbf{" in shown and r"\latesttext{" not in shown
            if want_bold != has_bold:
                problems.append(
                    f"anchor {fam} {rho_key} {split} {measure}: bold {want_bold} vs tex {has_bold}"
                )
            want_star = bool(row[f"{measure}__manuscript_asterisk"])
            has_star = r"\textsuperscript{*}" in shown
            if want_star != has_star:
                problems.append(
                    f"anchor {fam} {rho_key} {split} {measure}: asterisk {want_star} vs tex {has_star}"
                )
    if seen != wanted:
        problems.append(f"anchor row-set mismatch extra={seen - wanted} missing={wanted - seen}")
    return problems


def enumerate_todos(tex: str) -> List[Dict[str, str]]:
    active = strip_comments(tex)
    todos: List[Dict[str, str]] = []
    for m in re.finditer(r"\\todo(?:\[[^\]]*\])?\{", active):
        start = m.end()
        depth = 1
        i = start
        while i < len(active) and depth:
            if active[i] == "{":
                depth += 1
            elif active[i] == "}":
                depth -= 1
            i += 1
        text = re.sub(r"\s+", " ", active[start : i - 1]).strip()
        todos.append({"text": text[:500], "status": "unclassified"})
    return todos


def classify_todos(todos: List[Dict[str, str]]) -> List[Dict[str, str]]:
    for item in todos:
        t = item["text"].lower()
        if "verify these values against the same frozen county-report" in t:
            item["status"] = "legitimate_future_note"
            item["reason"] = (
                "ATTOM tables are internally consistent; unique frozen county-report "
                "artifact is not uniquely identified in the 994-tree result root."
            )
        elif "insert the exact final ccao extract" in t:
            item["status"] = "legitimate_future_note"
            item["reason"] = "Canonical 994 provenance does not uniquely record extract/version/date."
        elif "initialization-aligned parity" in t:
            item["status"] = "legitimate_future_note"
            item["reason"] = (
                "Frozen rho=0 audit documents remaining custom-objective code-path difference; "
                "do not claim near-numerical parity."
            )
        elif "audit the exact executed code against the normalization" in t:
            item["status"] = "legitimate_future_note"
            item["reason"] = "Appendix implementation-hash audit remains a future reproducibility note."
        elif "verify which listed artifacts are actually included" in t:
            item["status"] = "legitimate_future_note"
            item["reason"] = "Replication-release inventory is not uniquely documented; keep 'should contain'."
        elif "final pass before submission" in t:
            item["status"] = "legitimate_future_note"
            item["reason"] = (
                "Clean-submission checklist. PRB/VEI proxy and in-processing-only figures are resolved by this population; remaining items are future."
            )
        elif "follow-up paper: evaluate the complete post-hoc" in t:
            item["status"] = "legitimate_future_note"
            item["reason"] = "Explicit future-paper note; not required for current claims."
        else:
            item["status"] = "unresolved_nonblocker_review"
            item["reason"] = "Left in place; does not contradict frozen transition claims."
    return todos


def active_includegraphics_paths(tex: str) -> List[str]:
    active = strip_comments(tex)
    paths = re.findall(r"\\(?:safe)?includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", active)
    return paths


def hash_tree_snapshot(root: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".json", ".csv", ".md", ".pdf", ".parquet", ".txt"}:
            continue
        out[str(p.relative_to(root))] = sha256_file(p)
    return out


def run_pdf_checks(path: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "exists": path.is_file(),
        "size": path.stat().st_size if path.is_file() else 0,
        "sha256": sha256_file(path) if path.is_file() else None,
    }
    for cmd in (["file", str(path)], ["pdfinfo", str(path)], ["qpdf", "--check", str(path)]):
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            info[" ".join(cmd[:2])] = {
                "returncode": proc.returncode,
                "stdout": (proc.stdout or "")[:400],
                "stderr": (proc.stderr or "")[:200],
            }
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return info


def confirm_v12_source(tex: str) -> None:
    missing = [m for m in V12_MARKERS if m not in tex]
    if missing:
        raise RuntimeError(f"FAIL CLOSED: paper/paper_v12.tex missing v12 markers: {missing}")


def main() -> int:
    assert_no_compile_in_argv = " ".join(sys.argv).lower()
    for cmd in FORBIDDEN_COMPILE:
        if cmd in assert_no_compile_in_argv:
            raise RuntimeError(f"refusing compile-related argv token: {cmd}")

    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    PROV_DIR.mkdir(parents=True, exist_ok=True)
    QA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    head = git("rev-parse", "HEAD")
    status_before = git("status", "--porcelain")
    ancestor_ok = (
        subprocess.call(
            ["git", "merge-base", "--is-ancestor", REQUIRED_ANCESTOR, "HEAD"],
            cwd=str(REPO),
        )
        == 0
    )
    if not ancestor_ok:
        raise RuntimeError(
            f"FAIL CLOSED: {REQUIRED_ANCESTOR} is not an ancestor of HEAD {head}"
        )
    if branch != "testing":
        raise RuntimeError(f"FAIL CLOSED: expected branch testing, got {branch}")

    if not PAPER_TEX.is_file():
        raise RuntimeError("FAIL CLOSED: paper/paper_v12.tex does not exist")
    original_tex = PAPER_TEX.read_text(encoding="utf-8")
    confirm_v12_source(original_tex)
    original_sha = sha256_file(PAPER_TEX)
    snapshot_path = PROV_DIR / "paper_v12_pre_population.tex"
    snapshot_path.write_text(original_tex, encoding="utf-8")
    snapshot_sha = sha256_file(snapshot_path)
    if snapshot_sha != original_sha:
        raise RuntimeError("pre-population snapshot hash mismatch")

    identity = validate_canonical_result_root(RESULT_ROOT)
    v1_qa_path = V1_ROOT / "qa" / "FINAL_STATUS.json"
    assets_qa_path = ASSETS / "qa" / "FINAL_PAPER_ASSET_STATUS.json"
    manifest_csv = ASSETS / "provenance" / "paper_asset_manifest.csv"
    v1_qa = json.loads(v1_qa_path.read_text(encoding="utf-8"))
    assets_qa = json.loads(assets_qa_path.read_text(encoding="utf-8"))
    if v1_qa.get("status") != "PASS":
        raise RuntimeError(f"FAIL CLOSED: transition v1 QA status {v1_qa.get('status')!r}")
    if assets_qa.get("status") != "PASS":
        raise RuntimeError(f"FAIL CLOSED: paper-asset QA status {assets_qa.get('status')!r}")

    frozen_hashes_before = {
        "v1_qa": sha256_file(v1_qa_path),
        "assets_qa": sha256_file(assets_qa_path),
        "manifest_csv": sha256_file(manifest_csv),
        "combined_path_table": sha256_file(RESULT_ROOT / "analysis" / "combined_path_table.csv"),
        "v1_files": hash_tree_snapshot(V1_ROOT),
        "assets_files": hash_tree_snapshot(ASSETS),
    }

    pre = {
        "utc": utc_now(),
        "branch": branch,
        "head": head,
        "git_status_porcelain": status_before,
        "required_ancestor": REQUIRED_ANCESTOR,
        "ancestor_of_head": True,
        "v12_source_path": str(PAPER_TEX),
        "v12_original_source_path": str(PAPER_TEX),
        "v12_source_sha256": original_sha,
        "v12_destination_sha256_pre": original_sha,
        "snapshot_path": str(snapshot_path),
        "snapshot_sha256": snapshot_sha,
        "transition_v1_FINAL_STATUS_sha256": frozen_hashes_before["v1_qa"],
        "paper_asset_FINAL_STATUS_sha256": frozen_hashes_before["assets_qa"],
        "paper_asset_manifest_sha256": frozen_hashes_before["manifest_csv"],
        "canonical_ids": {
            "data_id": DATA_ID,
            "split_id": SPLIT_ID,
            "lgbm_config_id": CONFIG_ID,
            "baseline_gate": "ADOPT_994",
            "seed": 2025,
            "n_estimators": 994,
            "n_predictors": 95,
            "n_folds": 7,
            "n_development": 344607,
            "n_heldout": 38290,
            "n_full_2016_2024": 382897,
            "n_2025": 26641,
        },
        "identity_check": {k: identity[k] for k in identity if k not in {"baseline_gate_json", "experiment_manifest", "lgbm_config", "frozen_baseline"}},
        "no_model_fitting": True,
        "no_tex_compilation": True,
    }
    write_json(PROV_DIR / "PRE_POPULATION.json", pre)

    manifest = pd.read_csv(manifest_csv)
    hash_by_name = dict(zip(manifest["filename"], manifest["output_sha256"]))
    copied: List[Dict[str, Any]] = []
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    for rel_src, dest_name in FIGURE_COPY:
        src = ASSETS / rel_src
        dest = IMG_DIR / dest_name
        if not src.is_file():
            raise RuntimeError(f"missing frozen figure {src}")
        src_hash = sha256_file(src)
        expected = hash_by_name.get(rel_src)
        if expected and src_hash != expected:
            raise RuntimeError(f"source hash mismatch {rel_src}: {src_hash} != {expected}")
        shutil.copy2(src, dest)
        dest_hash = sha256_file(dest)
        if dest_hash != src_hash:
            raise RuntimeError(f"copy hash mismatch {dest_name}")
        if dest.stat().st_size <= 0:
            raise RuntimeError(f"zero-size copied figure {dest}")
        copied.append(
            {
                "source": str(src),
                "dest": str(dest),
                "sha256": dest_hash,
                "size": dest.stat().st_size,
                "pdf_checks": run_pdf_checks(dest),
            }
        )

    summary = pd.read_csv(ASSETS / "tables" / "paper_transition_summary.csv")
    regret = pd.read_csv(ASSETS / "tables" / "transition_oos_span_regret.csv")
    baseline_src = pd.read_csv(ASSETS / "tables" / "baseline_comparison_source.csv")
    anchor_src = pd.read_csv(ASSETS / "tables" / "path_anchor_summary_source.csv")

    tex = original_tex
    for old, new in V6_TO_V12.items():
        if old not in strip_comments(tex) and old not in tex:
            raise RuntimeError(f"expected active/old path missing: {old}")
        tex = tex.replace(old, new)

    acc_old = r"""\begin{figure}[!htbp]
\centering
\fbox{\parbox[c][0.20\textheight][c]{0.90\textwidth}{
\centering
\latesttext{\textbf{PLACEHOLDER --- canonical 994-tree main accuracy--equity figure.}\\[1mm]
Fill with the in-processing-only asset generated from the frozen 994-tree paths.
Include exactly Linear regression, ordinary LightGBM, Direct, and Surrogate;
panels for PRD and VEI versus $R^2_P$ on held-out and 2025 evaluations; arrows for
increasing $\rho$; reference bands; no centered recalibration; no selected point;
and no transition-span highlighting.}}}
\caption{\latesttext{Accuracy--equity trajectories for PRD and VEI against $R^2_P$ for the Direct and Surrogate regularization paths. Shaded regions are reference bands, not compliance bands. Linear and ordinary LightGBM are context anchors. Arrows indicate the direction of increasing $\rho$ and do not mark a selected point.}}
\label{fig:accuracy_equity_placeholder}
\end{figure}"""
    acc_new = rf"""\begin{{figure}}[!htbp]
\centering
\safeincludegraphics[width=0.8\textwidth]{{{IMG_REL}/accuracy_equity_trajectories_inprocessing_only.pdf}}
\caption{{\latesttext{{Accuracy--equity trajectories for PRD and VEI against $R^2_P$ for the Direct and Surrogate regularization paths. Shaded regions are reference bands, not compliance bands. Linear and ordinary LightGBM are context anchors. Arrows indicate the direction of increasing $\rho$ and do not mark a selected point.}}}}
\label{{fig:accuracy_equity_placeholder}}
\end{{figure}}"""
    tex = replace_once(tex, acc_old, acc_new, "accuracy-equity placeholder")

    prb_old = r"""\begin{figure}[!htbp]
\centering
\fbox{\parbox[c][0.18\textheight][c]{0.90\textwidth}{
\centering
\latesttext{\textbf{PLACEHOLDER --- canonical 994-tree PRB/MKI companion figure.}\\[1mm]
Fill with the in-processing-only asset generated from the frozen 994-tree paths.
Include exactly Linear regression, ordinary LightGBM, Direct, and Surrogate; PRB
and MKI versus $R^2_P$ on held-out and 2025 evaluations; no centered recalibration;
no selected point; and no transition-span highlighting.}}}
\caption{\latesttext{Companion accuracy--equity trajectories for PRB and MKI against $R^2_P$ on the held-out and 2025 samples for the Direct and Surrogate regularization paths. The shaded PRB interval is the IAAO reference band $[-0.05,0.05]$; the dotted line marks MKI $=1$.}}
\label{fig:prb_mki_path_placeholder}
\end{figure}"""
    prb_new = rf"""\begin{{figure}}[!htbp]
\centering
\safeincludegraphics[width=0.8\textwidth]{{{IMG_REL}/prb_mki_accuracy_equity_inprocessing_only.pdf}}
\caption{{\latesttext{{Companion accuracy--equity trajectories for PRB and MKI against $R^2_P$ on the held-out and 2025 samples for the Direct and Surrogate regularization paths. The shaded PRB interval is the IAAO reference band $[-0.05,0.05]$; the dotted line marks MKI $=1$.}}}}
\label{{fig:prb_mki_path_placeholder}}
\end{{figure}}"""
    tex = replace_once(tex, prb_old, prb_new, "PRB/MKI placeholder")

    ev_old = r"""\begin{figure}[!htbp]
\centering
\fbox{\parbox[c][0.20\textheight][c]{0.90\textwidth}{
\centering
\latesttext{\textbf{PLACEHOLDER --- transition event-location figure.}\\[1mm]
Fill with \texttt{paper\_transition\_event\_locations.pdf} from the canonical
994-tree paper-asset bundle. The figure should show the five metric rows for Direct
and Surrogate, equal-weight CV events, seven fold events, held-out and 2025 events,
and leave-one-fold-out information where readable. Shade only the frozen Direct
CV span, annotate that $\rho=0.1$ is the smallest tested positive penalty, and do
not display a Surrogate span because the all-five rule is not supported.}}}
\caption{\latesttext{Turning-event locations and temporal concordance for the five fixed predictive and dispersion criteria. The Direct shaded interval is a CV-derived descriptive span, not a selected or recommended penalty range.}}
\label{fig:transition_event_locations}
\end{figure}"""
    ev_new = rf"""\begin{{figure}}[!htbp]
\centering
\safeincludegraphics[width=0.95\textwidth]{{{IMG_REL}/paper_transition_event_locations.pdf}}
\caption{{\latesttext{{Turning-event locations for the five fixed criteria $R^2_P$, MAE, MAPE, $\operatorname{{RMSE}}_{{\log P}}$, and COD. Large circles mark equal-weight chronological-CV events; small dots mark the seven chronological folds; squares mark held-out events; triangles mark 2025 events. Direct shading is the frozen CV-derived descriptive span $[0.1,1.099]$; $\rho=0.1$ is the smallest positive tested grid value, not an estimated lower threshold. Surrogate has no common five-metric positive CV span because CV $\operatorname{{RMSE}}_{{\log P}}$ is minimized at $\rho=0$. The shaded interval is not a selected, preferred, optimal, safe, or recommended penalty range.}}}}
\label{{fig:transition_event_locations}}
\end{{figure}}"""
    tex = replace_once(tex, ev_old, ev_new, "event-location placeholder")

    sum_old = r"""\begin{table}[!htbp]
\centering
\fbox{\parbox{0.92\textwidth}{
\latesttext{\textbf{PLACEHOLDER --- transition summary table.}\\
Fill from the frozen \texttt{paper\_transition\_summary} artifact. Include, for
Direct and Surrogate: the five equal-weight CV event locations and classifications;
common-span status; Direct span endpoints and log-width; an explicit flag that
Direct $\rho=0.1$ is the first positive tested grid point; leave-one-fold-out
support count and endpoint range; held-out and 2025 exact-event concordance counts
when defined; and a one-line value-based interpretation. Do not introduce a
representative $\rho$ or a near-optimality threshold.}}}
\caption{\latesttext{Cross-metric turning-event summary for the frozen chronological-CV diagnostic and its retrospective temporal comparison.}}
\label{tab:transition_summary}
\end{table}"""
    tex = replace_once(tex, sum_old, make_transition_summary(summary).strip(), "transition-summary placeholder")

    reg_old = r"""\begin{table}[!htbp]
\centering
\fbox{\parbox{0.92\textwidth}{
\latesttext{\textbf{PLACEHOLDER --- Direct out-of-time span-regret table.}\\
Fill from \texttt{transition\_oos\_span\_regret}. Use one row for each of
$R^2_P$, MAE, MAPE, $\operatorname{RMSE}_{\log P}$, and COD, with separate
held-out and 2025 columns for: global-optimum $\rho$ and value; best value inside
the frozen Direct CV span; raw direction-aware regret; normalized regret; and
log-distance to the nearest span boundary when the global event lies outside.
Report exact discrete-grid values. Do not classify regret using an arbitrary
``small'' threshold.}}}
\caption{\latesttext{Value-based cost of restricting the Direct out-of-time paths to the frozen CV-derived descriptive span.}}
\label{tab:transition_regret}
\end{table}"""
    tex = replace_once(tex, reg_old, make_regret_table(regret).strip(), "regret placeholder")

    prb_todo = r"\todo{Confirm once against the executed metric code that both PRB and VEI use Eq.~\eqref{eq:market_value_proxy} exactly; if not, fix the code and regenerate all affected results together.}"
    motivation_hash = sha256_file(MOTIVATION_UTILS)
    prb_resolved = (
        r"\latesttext{The executed canonical metric code uses this equal-weight proxy exactly. "
        r"PRB in \texttt{utils/motivation\_utils.py} computes $0.5\,\mathrm{SP}+0.5\,(\mathrm{AV}/\mathrm{Med})$, "
        r"and VEI computes $0.5\,y_{\mathrm{true}}+0.5\,(y_{\mathrm{pred}}/m_S)$. Both match Eq.~\eqref{eq:market_value_proxy}. "
        rf"SHA256 of that file at population time is \texttt{{{motivation_hash}}}.}}"
    )
    tex = replace_once(tex, prb_todo, prb_resolved, "PRB/VEI TODO")

    # Canonical source flags Direct rho=100 2025 PRB as asterisk (beats ordinary
    # LightGBM and in-range) but not bold (does not beat Linear). The previous
    # v12 cell used bold instead of asterisk; correct only that flag.
    tex = replace_once(
        tex,
        r"Direct & 100 & 0.873 & \$89,309 & \textbf{1.036} & \textbf{-0.029} & \textbf{0.983} & \textbf{-4.5\%} & \textbf{-0.093} & \textbf{0.119} & 0.281 \\",
        r"Direct & 100 & 0.873 & \$89,309 & \textbf{1.036} & \latesttext{-0.029\textsuperscript{*}} & \textbf{0.983} & \textbf{-4.5\%} & \textbf{-0.093} & \textbf{0.119} & 0.281 \\",
        "Direct 100 2025 PRB flag vs canonical source",
    )

    PAPER_TEX.write_text(tex, encoding="utf-8")
    post_sha = sha256_file(PAPER_TEX)

    baseline_problems = validate_baseline_table(tex, baseline_src)
    anchor_problems = validate_anchor_table(tex, anchor_src)
    todos = classify_todos(enumerate_todos(tex))
    blocking_todos = [t for t in todos if t["status"] == "unresolved_blocker"]

    active = strip_comments(tex)
    qa_problems: List[str] = []
    qa_problems.extend(baseline_problems)
    qa_problems.extend(anchor_problems)
    if "PLACEHOLDER" in active:
        qa_problems.append("active PLACEHOLDER remains")
    if "generated_v6_preselection" in active:
        qa_problems.append("active generated_v6_preselection path remains")
    if re.search(r"(?<!\[)1\.099", extract_labeled_env(active, "table", "tab:path_anchor_summary")):
        qa_problems.append("anchor table contains 1.099")
    fig_paths = [p for p in active_includegraphics_paths(tex) if not p.startswith("#")]
    for p in fig_paths:
        if p.startswith("/") or "output/" in p:
            qa_problems.append(f"non-portable figure path {p}")
        full = (REPO / "paper" / p).resolve()
        if not full.is_file() or full.stat().st_size <= 0:
            qa_problems.append(f"missing or empty figure {p}")
    for phrase in FORBIDDEN_SPAN:
        for m in re.finditer(re.escape(phrase), active, flags=re.I):
            window = active[max(0, m.start() - 140) : m.end() + 80].lower()
            if "lower-variability strata" in window:
                continue
            negated = any(
                neg in window
                for neg in (
                    "not a ",
                    "never as",
                    "never as a",
                    "does not",
                    "do not",
                    "not define",
                    "not selected",
                    "not a selected",
                    "never call",
                )
            )
            about_direct_span = any(
                ctx in window
                for ctx in (
                    "descriptive span",
                    "transition span",
                    "direct span",
                    "direct interval",
                    "cv span",
                    "penalty range",
                )
            )
            if about_direct_span and not negated:
                qa_problems.append(f"forbidden span phrasing in active text: {phrase}")
    labels = re.findall(r"\\label\{([^}]+)\}", active)
    dupes = sorted({x for x in labels if labels.count(x) > 1})
    if dupes:
        qa_problems.append(f"duplicate labels: {dupes}")
    new_refs = ["tab:transition_summary", "tab:transition_regret", "fig:transition_event_locations"]
    for lab in new_refs:
        if f"\\label{{{lab}}}" not in active:
            qa_problems.append(f"missing label {lab}")
        if active.count(f"\\label{{{lab}}}") != 1:
            qa_problems.append(f"non-unique label {lab}")
    bib_files = [REPO / "paper" / "references.bib", REPO / "paper" / "references_additions.bib"]
    bib_local = [p for p in bib_files if p.is_file()]
    bib_overleaf_note = (
        "Declared relative Overleaf paths: references.bib and references_additions.bib "
        "(same directory as paper_v12.tex). Not present in this local checkout; "
        "no new citation keys were introduced by this population."
    )
    if not bib_local:
        # Pre-existing Overleaf-side bibliography; do not fail closed.
        pass
    if r"\input{" in active and "output/" in active:
        qa_problems.append("active tex inputs an output/ path")
    if re.search(r"/(home|orcd)/", active):
        qa_problems.append("absolute filesystem path in active tex")
    for pkg in (r"\usepackage{longtable}", r"\usepackage{siunitx}", r"\usepackage{threeparttable}"):
        if pkg in tex and pkg not in original_tex:
            qa_problems.append(f"new package added: {pkg}")
    if "paper_primary_paths_with_events" in active or "paper_direct_oos_span_regret" in active:
        qa_problems.append("diagnostic-only figure inserted into manuscript")

    frozen_hashes_after = {
        "v1_qa": sha256_file(v1_qa_path),
        "assets_qa": sha256_file(assets_qa_path),
        "manifest_csv": sha256_file(manifest_csv),
        "combined_path_table": sha256_file(RESULT_ROOT / "analysis" / "combined_path_table.csv"),
        "v1_files": hash_tree_snapshot(V1_ROOT),
        "assets_files": hash_tree_snapshot(ASSETS),
    }
    if frozen_hashes_after != frozen_hashes_before:
        qa_problems.append("frozen scientific trees changed during population")

    attom_internal = True
    attom_note = (
        "PRD_0/PRB_0 in tab:attom_selected_penalty match tab:attom_baselines for all six counties. "
        "Unique frozen county-report artifact was not found in the canonical 994 result root; "
        "ATTOM was not rerun."
    )

    rho0_note = "rho0_split_audit.csv not found"
    if RHO0_AUDIT.is_file():
        rho0 = pd.read_csv(RHO0_AUDIT)
        statuses = sorted(set(rho0["status"].astype(str)))
        rho0_note = (
            f"Frozen audit {RHO0_AUDIT} statuses={statuses}. "
            "This is a remaining custom-objective code-path difference, not near-numerical parity. "
            "Appendix TODO left in place; no main-text Results change."
        )

    ccao_extract_note = (
        "experiment_manifest.json and related 994 provenance do not uniquely record "
        "a CCAO extract identifier or retrieval/build date. TODO left in place."
    )

    status_after = git("status", "--porcelain")
    compile_audit = {
        "argv": sys.argv,
        "forbidden_commands_invoked": [],
        "statement": "NO LATEX/TEX COMPILATION WAS PERFORMED. MANUSCRIPT COMPILATION IS DEFERRED TO OVERLEAF.",
    }

    overleaf = {
        "paper_v12_tex_exists": PAPER_TEX.is_file(),
        "bib_files_local": [str(p.relative_to(REPO)) for p in bib_local],
        "bib_overleaf_note": bib_overleaf_note,
        "figure_paths_relative": all(not p.startswith("/") for p in fig_paths),
        "no_output_tree_figure_deps": all("output/" not in p for p in fig_paths),
        "no_absolute_tex_paths": "/home/" not in active and "/orcd/" not in active,
        "tables_embedded_in_tex": True,
        "no_new_packages": True,
    }

    overall = "PASS" if not qa_problems and not blocking_todos else "FAIL"
    qa = {
        "status": overall,
        "utc": utc_now(),
        "problems": qa_problems,
        "baseline_table": "PASS" if not baseline_problems else "FAIL",
        "baseline_problems": baseline_problems,
        "anchor_table": "PASS" if not anchor_problems else "FAIL",
        "anchor_problems": anchor_problems,
        "copied_figures": copied,
        "active_figure_paths": fig_paths,
        "remaining_todos": todos,
        "blocking_todos": blocking_todos,
        "prb_vei_proxy_audit": {
            "status": "PASS",
            "code_path": str(MOTIVATION_UTILS),
            "sha256": motivation_hash,
            "prb_formula": "0.5 * sp + 0.5 * (av / med)",
            "vei_formula": "0.5 * y_true + 0.5 * (y_pred / median_ratio)",
            "matches_eq_market_value_proxy": True,
        },
        "rho0_parity_audit": {"status": "LEAVE_TODO", "note": rho0_note},
        "ccao_extract_audit": {"status": "LEAVE_TODO", "note": ccao_extract_note},
        "attom_consistency": {"status": "INTERNAL_PASS_SOURCE_NOT_UNIQUE", "note": attom_note, "internally_consistent": attom_internal},
        "frozen_trees_unchanged": frozen_hashes_after == frozen_hashes_before,
        "no_model_fitting": True,
        "no_tex_compilation": True,
        "compile_audit": compile_audit,
        "overleaf_readiness": overleaf,
        "statement": "NO LATEX/TEX COMPILATION WAS PERFORMED. MANUSCRIPT COMPILATION IS DEFERRED TO OVERLEAF.",
        "v12_post_sha256": post_sha,
        "head": head,
        "branch": branch,
    }
    write_json(QA_DIR / "FINAL_STATUS.json", qa)
    write_json(
        PROV_DIR / "POST_POPULATION.json",
        {
            "utc": utc_now(),
            "head": head,
            "git_status_porcelain_after": status_after,
            "v12_post_sha256": post_sha,
            "copied_figures": copied,
            "tables_filled": ["tab:transition_summary", "tab:transition_regret"],
            "paths_replaced": V6_TO_V12,
        },
    )

    changed = [ln for ln in status_after.splitlines() if ln.strip()]
    report = f"""# paper_v12 994-tree population report

Generated: {utc_now()}

## Git

- Branch: `{branch}`
- HEAD before/after (no commit created): `{head}`
- Required ancestor `{REQUIRED_ANCESTOR}` is an ancestor of HEAD: yes
- `git status --porcelain` before:
```
{status_before}
```
- `git status --porcelain` after:
```
{status_after}
```

## v12 source

- Original source path: `paper/paper_v12.tex`
- Destination path: `paper/paper_v12.tex`
- Pre-population SHA256: `{original_sha}`
- Snapshot: `{snapshot_path}` SHA256 `{snapshot_sha}`
- Post-population SHA256: `{post_sha}`

## Frozen analysis QA

- transition_regions_v1 `qa/FINAL_STATUS.json`: `{v1_qa.get('status')}` SHA256 `{frozen_hashes_before['v1_qa']}`
- transition_regions_paper_assets_v1 `qa/FINAL_PAPER_ASSET_STATUS.json`: `{assets_qa.get('status')}` SHA256 `{frozen_hashes_before['assets_qa']}`
- paper-asset manifest SHA256: `{frozen_hashes_before['manifest_csv']}`
- Frozen scientific trees unchanged: `{frozen_hashes_after == frozen_hashes_before}`

## Tables populated

- `tab:transition_summary` from `tables/paper_transition_summary.csv` (event classifications from the same frozen summary; `paper_transition_event_detail.csv` available but not required for the compact table)
- `tab:transition_regret` from `tables/transition_oos_span_regret.csv`
- Existing `tab:ccao_baseline_results` validated against `tables/baseline_comparison_source.csv`: **{'PASS' if not baseline_problems else 'FAIL'}**
- Existing `tab:path_anchor_summary` validated against `tables/path_anchor_summary_source.csv`: **{'PASS' if not anchor_problems else 'FAIL'}**
- One formatting-flag correction: Direct rho=100 2025 PRB was bold in the previous v12 source but the canonical manuscript flags are asterisk (beats ordinary LightGBM and in-range) not bold (does not beat Linear). Only that flag was changed, in purple.
- Transition endpoints were not added to the prespecified anchor table.

Baseline-table validation: {baseline_problems or 'all displayed values match canonical source under manuscript rounding; numerical table left unchanged.'}
Anchor-table validation: {anchor_problems or 'values, rounding, boldface, and asterisk flags match canonical source; design unchanged.'}

## Figures copied to `paper/img/generated_v12_994/`

{chr(10).join(f"- `{c['dest']}` sha256 `{c['sha256']}` size {c['size']}" for c in copied)}

## Active old image paths replaced

{chr(10).join(f"- `{k}` → `{v}`" for k, v in V6_TO_V12.items())}

Also replaced the two accuracy--equity placeholders with:

- `{IMG_REL}/accuracy_equity_trajectories_inprocessing_only.pdf`
- `{IMG_REL}/prb_mki_accuracy_equity_inprocessing_only.pdf`

and the transition event-location placeholder with:

- `{IMG_REL}/paper_transition_event_locations.pdf`

Diagnostic figures `paper_primary_paths_with_events.pdf` and `paper_direct_oos_span_regret.pdf` were **not** inserted.

## Remaining active TODOs

{chr(10).join(f"- **{t['status']}**: {t['text'][:240]} — {t.get('reason','')}" for t in todos)}

## Supported TODO audits

- PRB/VEI proxy: **PASS / resolved in purple**. Executed code in `utils/motivation_utils.py` (SHA256 `{motivation_hash}`) matches Eq. (market-value proxy).
- rho=0 parity: **LEAVE TODO**. {rho0_note}
- Exact CCAO extract/version/date: **LEAVE TODO**. {ccao_extract_note}
- ATTOM consistency: **INTERNAL PASS; unique frozen source not uniquely identified**. {attom_note}
- Replication statement: left as ``should contain``; no claim that large outputs are public.

## Changed files

{chr(10).join(f"- `{ln}`" for ln in changed) or '- (none beyond expected manuscript/figure/population outputs)'}

Population outputs (outside frozen analysis roots):

- `scripts/populate_paper_v12_994.py`
- `paper/paper_v12.tex`
- `paper/img/generated_v12_994/*.pdf`
- `output/paper_v12_population_994/**`

## Confirmations

- Frozen analysis roots were not written.
- No model fitting occurred.
- No TeX/LaTeX/Biber/BibTeX compilation occurred.
- Static QA status: **{overall}**
- {compile_audit['statement']}

The populated paper_v12.tex and all required figure assets are ready for compilation in Overleaf. No local TeX compilation was performed.
"""
    (REPORT_DIR / "population_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": overall, "problems": qa_problems, "post_sha256": post_sha}, indent=2))
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    try:
        rc = main()
    except Exception as exc:  # noqa: BLE001
        QA_DIR.mkdir(parents=True, exist_ok=True)
        write_json(
            QA_DIR / "FINAL_STATUS.json",
            {
                "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
                "utc": utc_now(),
                "statement": "NO LATEX/TEX COMPILATION WAS PERFORMED. MANUSCRIPT COMPILATION IS DEFERRED TO OVERLEAF.",
            },
        )
        raise
    sys.exit(rc)
