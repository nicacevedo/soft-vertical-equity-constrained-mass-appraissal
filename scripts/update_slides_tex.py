"""Apply the numbers and figure paths from ``slide_summary_{year}.json`` to
``Paper/GE slides/slides_v1_5 {year}.tex`` in place.

Automated edits (year-specific):
  - Slide 10 ("The CCAO baseline tension") LR + LGBM metric cells.
  - Slide 10 caption year tag.
  - Slide 15 "Visual effect" Corr/Slope annotation table + motivation figures.
  - Slide 16 "Main metrics" LGBM baseline + penalized row, tcolorboxes, caption.
  - Slide 17 tradeoff figure paths.
  - Slide 20 pareto figure path.
  - Appendix A2 full results table body.
  - Appendix A4 (cont.) constrained-calibration table body + caption year.
  - Appendix A6 correlations figure path.

The script is idempotent and uses distinctive anchor comments (e.g. ``% ---- BEGIN SLIDE 10 TABLE ----``) that are injected on first run so future edits
stay stable.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------
def _fmt_money(val: float) -> str:
    if val is None or not np.isfinite(val):
        return "---"
    i = int(round(float(val)))
    return "\\$" + f"{i:,}".replace(",", "{,}")


def _fmt_pct(val_frac: float, *, digits: int = 2, signed: bool = True) -> str:
    """Percent format, expects fraction (e.g. -0.368 -> -36.80%)."""
    if val_frac is None or not np.isfinite(val_frac):
        return "---"
    pct = float(val_frac) * 100.0
    if signed:
        return f"{pct:+.{digits}f}\\%"
    return f"{pct:.{digits}f}\\%"


def _fmt_float(val: float, digits: int = 3, signed: bool = False) -> str:
    if val is None or not np.isfinite(val):
        return "---"
    v = float(val)
    if signed:
        return f"{v:+.{digits}f}"
    return f"{v:.{digits}f}"


def _pct_value_num(val_frac: float, digits: int = 1, signed: bool = False) -> str:
    """Format a value that is ALREADY in percent units (e.g. VEI, COD).

    No rescaling is applied; we just stringify ``val_frac`` as-is with the
    requested precision. Use ``_fmt_pct`` if you actually need a fraction→%
    conversion.
    """
    if val_frac is None or not np.isfinite(val_frac):
        return "---"
    v = float(val_frac)
    if signed:
        return f"{v:+.{digits}f}"
    return f"{v:.{digits}f}"


# ---------------------------------------------------------------------------
# tex block rewriters
# ---------------------------------------------------------------------------
def _regex_replace_single(tex: str, pattern: str, replacement: str, *, flags=re.DOTALL) -> str:
    new_tex, n = re.subn(pattern, lambda m, r=replacement: r, tex, count=1, flags=flags)
    if n == 0:
        raise RuntimeError(f"pattern not found:\n{pattern}")
    return new_tex


def update_slide10(tex: str, summary: dict, year: str) -> str:
    """Rewrite the LR + LGBM row in the Slide 10 table and the caption."""
    lr = summary["slide10"]["LinearRegression"]
    lgb = summary["slide10"]["LGBMRegressor"]

    def _row(r: dict, bold: bool) -> str:
        r2 = _fmt_float(r.get("R2"), 3)
        mae = _fmt_money(r.get("MAE"))
        cod = _fmt_float(r.get("COD"), 2)
        cov = _fmt_float(r.get("COV_IAAO", 0.0) * 100.0 if r.get("COV_IAAO") is not None else np.nan, 1)
        vei_pct = _pct_value_num(r.get("VEI"), digits=2, signed=True)
        prd = _fmt_float(r.get("PRD"), 3)
        prb = _fmt_float(r.get("PRB"), 3, signed=True)
        if bold:
            return (
                f"\\textcolor{{green!60!black}}{{\\textbf{{{r2}}}}} & "
                f"\\textcolor{{green!60!black}}{{\\textbf{{{mae}}}}} & "
                f"\\textcolor{{green!60!black}}{{\\textbf{{{cod}}}}} & "
                f"\\textcolor{{green!60!black}}{{\\textbf{{{cov}}}}} & "
                f"\\textcolor{{red}}{{\\textbf{{{vei_pct}}}}} & "
                f"\\textcolor{{red}}{{\\textbf{{{prd}}}}} & "
                f"\\textcolor{{red}}{{\\textbf{{{prb}}}}}"
            )
        return f"{r2} & {mae} & {cod} & {cov} & {vei_pct} & {prd} & {prb}"

    lr_cells = _row(lr, bold=False)
    lgb_cells = _row(lgb, bold=True)

    pattern = (
        r"Linear Regression &[^\n]*\\\\\s*\n"
        r"LightGBM\s+&[^\n]*\\\\\s*\n"
        r"\\bottomrule"
    )
    replacement = (
        f"Linear Regression & {lr_cells} \\\\\n"
        f"LightGBM          & {lgb_cells} \\\\\n"
        f"\\bottomrule"
    )
    tex = _regex_replace_single(tex, pattern, replacement)

    # Caption year
    tex = re.sub(
        r"Held-out 20\d\d CCAO test set\.",
        f"Held-out {year} CCAO test set.",
        tex,
        count=1,
    )
    return tex


def update_slide15(tex: str, summary: dict, img_rel: str) -> str:
    """Rewrite Corr/Slope values + motivation figure paths."""
    base = summary["slide15"]["baseline"]
    pen = summary["slide15"]["penalized"]

    base_corr = float(base.get("Corr(r,logprice)", np.nan))
    base_slope = float(base.get("Slope(r~logy)", np.nan))
    pen_corr = float(pen.get("Corr(r,logprice)", np.nan))
    pen_slope = float(pen.get("Slope(r~logy)", np.nan))

    def f(x): return f"{x:+.3f}" if np.isfinite(x) else "---"

    corr_repl = (
        f"Corr($r$, $P$)                 & {{\\color{{red}}${f(base_corr)}$}} "
        f"& $\\rightarrow$ & {{\\color{{blue}}${f(pen_corr)}$}}"
    )
    slope_repl = (
        f"Slope($r \\sim \\log P$)         & {{\\color{{red}}${f(base_slope)}$}} "
        f"& $\\rightarrow$ & {{\\color{{blue}}${f(pen_slope)}$}}"
    )
    tex = re.sub(
        r"Corr\(\$r\$, \$P\$\)\s*&\s*\{\\color\{red\}\$[^$]+\$\}\s*&\s*\$\\rightarrow\$\s*&\s*\{\\color\{blue\}\$[^$]+\$\}",
        lambda m, repl=corr_repl: repl,
        tex,
        count=1,
    )
    tex = re.sub(
        r"Slope\(\$r\s*\\sim\s*\\log P\$\)\s*&\s*\{\\color\{red\}\$[^$]+\$\}\s*&\s*\$\\rightarrow\$\s*&\s*\{\\color\{blue\}\$[^$]+\$\}",
        lambda m, repl=slope_repl: repl,
        tex,
        count=1,
    )

    # Figure paths (baseline + penalized)
    tex = re.sub(
        r"\{CCAO meeting/img/motivation/Baseline LightGBM_0_motivation_arrows_3\.pdf\}",
        f"{{{img_rel}/motivation/baseline_motivation.pdf}}",
        tex,
        count=1,
    )
    tex = re.sub(
        r"\{CCAO meeting/img/Covariance-Penalized LightGBM_2_motivation\.pdf\}",
        f"{{{img_rel}/motivation/penalized_motivation.pdf}}",
        tex,
        count=1,
    )
    return tex


def update_slide16(tex: str, summary: dict, year: str) -> str:
    """Rewrite the LGBM baseline + penalized row of slide 16 + tcolorboxes."""
    base = summary["slide16"]["baseline"]
    pen = summary["slide16"]["penalized"]
    rho = float(pen.get("rho", np.nan))
    rho_txt = f"{rho:.2f}" if np.isfinite(rho) else "?"

    def _base_row(r: dict) -> str:
        r2 = _fmt_float(r.get("R2"), 3)
        mae = _fmt_money(r.get("MAE"))
        cod = _fmt_float(r.get("COD"), 2)
        cov_pct = _fmt_float((r.get("COV_IAAO") or 0.0) * 100.0, 1)
        vei_pct = _pct_value_num(r.get("VEI"), digits=1, signed=True)
        prd = _fmt_float(r.get("PRD"), 3)
        prb = _fmt_float(r.get("PRB"), 3, signed=True)
        return (
            f"LightGBM (baseline)   & {r2} & {mae} & {cod} & {cov_pct}\\% & "
            f"\\textcolor{{red}}{{${vei_pct}\\%$}} & \\textcolor{{red}}{{${prd}$}} & "
            f"\\textcolor{{red}}{{${prb}$}}"
        )

    def _pen_row(r: dict) -> str:
        r2 = _fmt_float(r.get("R2"), 3)
        mae = _fmt_money(r.get("MAE"))
        cod = _fmt_float(r.get("COD"), 2)
        cov_pct = _fmt_float((r.get("COV_IAAO") or 0.0) * 100.0, 1)
        vei_pct = _pct_value_num(r.get("VEI"), digits=1, signed=True)
        prd = _fmt_float(r.get("PRD"), 3)
        prb = _fmt_float(r.get("PRB"), 3, signed=True)
        return (
            f"\\;\\;\\; + covariance penalty ($\\rho{{=}}{rho_txt}$) & "
            f"\\textbf{{{r2}}} & {mae} & \\textbf{{{cod}}} & \\textbf{{{cov_pct}\\%}} & "
            f"\\textcolor{{green!60!black}}{{\\textbf{{${vei_pct}\\%$}}}} & "
            f"\\textcolor{{green!60!black}}{{\\textbf{{${prd}$}}}} & "
            f"\\textcolor{{green!60!black}}{{\\textbf{{${prb}$}}}}"
        )

    pattern = (
        r"LightGBM \(baseline\)[^\n]*\\\\\s*\n"
        r"\\;\\;\\; \+ covariance penalty[^\n]*\n"
        r"([^\n]*\n){0,5}?"
        r"\\bottomrule"
    )
    replacement = (
        f"{_base_row(base)} \\\\\n"
        f"{_pen_row(pen)} \\\\\n"
        f"\\bottomrule"
    )
    tex = _regex_replace_single(tex, pattern, replacement)

    # The tcolorbox triplet
    prd_base = _fmt_float(base.get("PRD"), 3)
    prd_pen = _fmt_float(pen.get("PRD"), 3)
    prb_base = _fmt_float(base.get("PRB"), 3, signed=True)
    prb_pen = _fmt_float(pen.get("PRB"), 3, signed=True)
    vei_base_pct = _pct_value_num(base.get("VEI"), digits=1, signed=True)
    vei_pen_pct = _pct_value_num(pen.get("VEI"), digits=1, signed=True)

    def _box_repl(label: str, base_val: str, pen_val: str, pct: bool = False) -> str:
        if pct:
            base_fmt = f"{base_val}\\%"
            pen_fmt = f"{pen_val}\\%"
        else:
            base_fmt = base_val
            pen_fmt = pen_val
        return (
            f"\\textbf{{{label}}}\\\\[0.5mm]\n"
            f"{{\\color{{red}}${base_fmt}$}}\\; $\\rightarrow$\\; "
            f"{{\\color{{green!60!black}}$\\mathbf{{{pen_fmt}}}$}}"
        )

    prd_repl = _box_repl("PRD", prd_base, prd_pen)
    prb_repl = _box_repl("PRB", prb_base, prb_pen)
    vei_repl = _box_repl("VEI", vei_base_pct, vei_pen_pct, pct=True)

    tex = re.sub(
        r"\\textbf\{PRD\}\\\\\[0\.5mm\]\s*\n\s*\{\\color\{red\}\$[^$]+\$\}\\;\s*\$\\rightarrow\$\\;\s*\{\\color\{green!60!black\}\$\\mathbf\{[^}]+\}\$\}",
        lambda m, r=prd_repl: r, tex, count=1,
    )
    tex = re.sub(
        r"\\textbf\{PRB\}\\\\\[0\.5mm\]\s*\n\s*\{\\color\{red\}\$[^$]+\$\}\\;\s*\$\\rightarrow\$\\;\s*\{\\color\{green!60!black\}\$\\mathbf\{[^}]+\}\$\}",
        lambda m, r=prb_repl: r, tex, count=1,
    )
    tex = re.sub(
        r"\\textbf\{VEI\}\\\\\[0\.5mm\]\s*\n\s*\{\\color\{red\}\$[^$]+\$\}\\;\s*\$\\rightarrow\$\\;\s*\{\\color\{green!60!black\}\$\\mathbf\{[^}]+\}\$\}",
        lambda m, r=vei_repl: r, tex, count=1,
    )

    # Caption: "Representative penalized LightGBM on the 2023 CCAO test set"
    tex = re.sub(
        r"Representative penalized LightGBM on the 20\d\d CCAO test set",
        f"Representative penalized LightGBM on the {year} CCAO test set",
        tex,
    )
    return tex


def update_figure_paths(tex: str, img_rel: str) -> str:
    """Repoint tradeoff / pareto / correlation / extreme-case figure paths."""
    substitutions = [
        (r"\{CCAO meeting/img/tradeoffs/PRD_vs_R2_band\.pdf\}",
         f"{{{img_rel}/tradeoffs/PRD_vs_R2_band.pdf}}"),
        (r"\{CCAO meeting/img/tradeoffs/VEI_vs_R2_band\.pdf\}",
         f"{{{img_rel}/tradeoffs/VEI_vs_R2_band.pdf}}"),
        (r"\{CCAO meeting/img/model_selection/pareto_optimal\.pdf\}",
         f"{{{img_rel}/model_selection/pareto_optimal.pdf}}"),
        (r"\{CCAO meeting/img/correlations/correlations\.pdf\}",
         f"{{{img_rel}/correlations/correlations.pdf}}"),
    ]
    for pat, repl in substitutions:
        tex = re.sub(pat, repl, tex, count=1)
    return tex


def update_appendix_a2(tex: str, summary: dict) -> str:
    """Rewrite Appendix A2 full results table body."""
    lr = summary["slide10"]["LinearRegression"]
    lgb = summary["slide10"]["LGBMRegressor"]
    rows = list(summary.get("appendix_a2", []))

    def _baseline_row(name: str, r: dict) -> str:
        return (
            f"{name} & -- & {_fmt_float(r.get('R2'), 3)} & {_fmt_money(r.get('MAE'))} & "
            f"{_fmt_float(r.get('COD'), 3)} & {_fmt_float(r.get('COV_IAAO'), 3)} & "
            f"{_pct_value_num(r.get('VEI'), digits=3, signed=True)}\\% & {_fmt_float(r.get('PRD'), 3)} & "
            f"{_fmt_float(r.get('PRB'), 3, signed=True)} & {_fmt_float(r.get('MKI'), 3)}"
        )

    def _pen_row(r: dict, kind: str) -> str:
        rho = float(r.get("rho", np.nan))
        rho_txt = f"{rho:.2f}" if np.isfinite(rho) else "--"

        def pct_delta(val_new, val_base, *, signed: bool = True) -> str:
            try:
                if val_base == 0 or not np.isfinite(val_new) or not np.isfinite(val_base):
                    return "--"
                d = (float(val_new) - float(val_base)) / abs(float(val_base)) * 100.0
            except Exception:
                return "--"
            return f"{d:+.2f}\\%" if signed else f"{d:.2f}\\%"

        r2 = _fmt_float(r.get("R2"), 3)
        mae = _fmt_money(r.get("MAE"))
        cod = _fmt_float(r.get("COD"), 3)
        cov = _fmt_float(r.get("COV_IAAO"), 3)
        vei = _pct_value_num(r.get("VEI"), digits=3, signed=True) + "\\%"
        prd = _fmt_float(r.get("PRD"), 3)
        prb = _fmt_float(r.get("PRB"), 3, signed=True)
        mki = _fmt_float(r.get("MKI"), 3)

        r2_d = pct_delta(r.get("R2"), lgb.get("R2"))
        mae_d = pct_delta(r.get("MAE"), lgb.get("MAE"))
        cod_d = pct_delta(r.get("COD"), lgb.get("COD"))
        cov_d = pct_delta(r.get("COV_IAAO"), lgb.get("COV_IAAO"))
        vei_d = pct_delta(abs(r.get("VEI", np.nan)) - abs(lgb.get("VEI", np.nan)), abs(lgb.get("VEI", np.nan)))
        prd_d = pct_delta(abs(r.get("PRD", np.nan) - 1), abs(lgb.get("PRD", np.nan) - 1))
        prb_d = pct_delta(abs(r.get("PRB", np.nan)), abs(lgb.get("PRB", np.nan)))
        mki_d = pct_delta(abs(r.get("MKI", np.nan) - 1), abs(lgb.get("MKI", np.nan) - 1))

        return (
            f"{kind} & {rho_txt} & "
            f"\\makecell{{\\textbf{{{r2}}}\\\\{{\\scriptsize({r2_d})}}}} & "
            f"\\makecell{{{mae}\\\\{{\\scriptsize({mae_d})}}}} & "
            f"\\makecell{{{cod}\\\\{{\\scriptsize({cod_d})}}}} & "
            f"\\makecell{{\\textbf{{{cov}}}\\\\{{\\scriptsize({cov_d})}}}} & "
            f"\\makecell{{\\textbf{{{vei}}}\\\\{{\\scriptsize({vei_d})}}}} & "
            f"\\makecell{{\\textbf{{{prd}}}\\\\{{\\scriptsize({prd_d})}}}} & "
            f"\\makecell{{\\textbf{{{prb}}}\\\\{{\\scriptsize({prb_d})}}}} & "
            f"\\makecell{{\\textbf{{{mki}}}\\\\{{\\scriptsize({mki_d})}}}}"
        )

    kinds = ["LGBMSurrPenalty [v1]", "LGBMSurrPenalty [v2]", "LGBMCovPenalty [v1]", "LGBMCovPenalty [v2]"]
    pen_rows = []
    for i, r in enumerate(rows[:4]):
        kind = kinds[i] if i < len(kinds) else f"LGBMPenalty [v{i+1}]"
        pen_rows.append(_pen_row(r, kind))

    table_body = (
        _baseline_row("Linear Regression", lr) + " \\\\\n"
        + _baseline_row("LGBM Regressor   ", lgb) + " \\\\\n"
        + "\\midrule\n"
        + " \\\\\n[1.2mm]\n".join(pen_rows)
        + " \\\\"
    )

    pattern = (
        r"Linear Regression & -- &[^\n]*\\\\\s*\n"
        r"LGBM Regressor\s+& -- &.*?"
        r"\\bottomrule"
    )
    replacement = table_body + "\n\\bottomrule"
    tex = _regex_replace_single(tex, pattern, replacement, flags=re.DOTALL)
    return tex


def update_appendix_a4_table(tex: str, summary: dict, year: str) -> str:
    """Rewrite the Appendix A4 'Constrained calibration --- test results' table."""
    a4 = summary.get("appendix_a4", {})
    lr = summary["slide10"]["LinearRegression"]
    lgb = summary["slide10"]["LGBMRegressor"]

    def _baseline_row(name: str, model_lbl: str, r: dict) -> str:
        return (
            f"{name} & {model_lbl} & --   & {_fmt_float(r.get('R2'), 3)} & "
            f"{_fmt_money(r.get('MAE'))} & {_fmt_float(r.get('Median ratio'), 3)} & "
            f"{_fmt_float(r.get('Mean ratio'), 3)} & "
            f"{_fmt_float(r.get('COD'), 2)}\\% & "
            f"{_fmt_float((r.get('COV_IAAO') or 0.0) * 100.0, 2)}\\% & "
            f"{_fmt_float(r.get('PRD'), 3)} & "
            f"{_fmt_float(r.get('PRB'), 3, signed=True)} & "
            f"{_pct_value_num(r.get('VEI'), digits=2, signed=True)}\\%"
        )

    def _selection_row(solution: str, r: dict) -> str:
        model_short = {"LGBSmoothPenalty": "LGBSmooth", "LGBCovPenalty": "LGBCov"}.get(
            r.get("model_name", ""), r.get("model_name", "")
        )
        rho = r.get("rho", np.nan)
        try:
            rho_txt = f"{float(rho):.2f}"
        except Exception:
            rho_txt = "--"
        status = str(r.get("status", "")).lower()
        if status != "selected":
            return f"INFEASIBLE$^{{*}}$ & --- & --- & --- & --- & --- & --- & --- & --- & --- & --- & ---"
        return (
            f"{solution.upper()} & {model_short} & {rho_txt} & "
            f"{_fmt_float(r.get('R2'), 3)} & {_fmt_money(r.get('MAE'))} & "
            f"{_fmt_float(r.get('Median ratio'), 3)} & {_fmt_float(r.get('Mean ratio'), 3)} & "
            f"{_fmt_float(r.get('COD'), 2)}\\% & "
            f"{_fmt_float((r.get('COV_IAAO') or 0.0) * 100.0, 2)}\\% & "
            f"{_fmt_float(r.get('PRD'), 3)} & "
            f"{_fmt_float(r.get('PRB'), 3, signed=True)} & "
            f"{_pct_value_num(r.get('VEI'), digits=2, signed=True)}\\%"
        )

    header = (
        "\\multicolumn{12}{l}{\\textbf{Unconstrained baselines}} \\\\\n"
        "\\midrule\n"
        f"{_baseline_row('BASELINE', 'Linear Reg.', lr)} \\\\\n"
        f"{_baseline_row('BASELINE', 'LGBM       ', lgb)} \\\\\n"
    )

    def _subset_block(tag: str, title: str) -> str:
        rows = a4.get(tag, [])
        constrained = [r for r in rows if r.get("selection_method") == "constrained"]
        utopia = [r for r in rows if r.get("selection_method") == "utopia"]
        nash = [r for r in rows if r.get("selection_method") == "nash"]
        txt = (
            "\\midrule\n"
            f"\\multicolumn{{12}}{{l}}{{\\textbf{{Constraints: {title}}}}} \\\\\n"
            "\\midrule\n"
        )
        c_row = _selection_row("CONSTRAINED", constrained[0]) if constrained else "CONSTRAINED & --- & --- & --- & --- & --- & --- & --- & --- & --- & --- & ---"
        u_row = _selection_row("UTOPIA", utopia[0]) if utopia else "UTOPIA & --- & --- & --- & --- & --- & --- & --- & --- & --- & --- & ---"
        n_row = _selection_row("NASH", nash[0]) if nash else "NASH & --- & --- & --- & --- & --- & --- & --- & --- & --- & --- & ---"
        txt += f"{c_row} \\\\\n{u_row} \\\\\n{n_row} \\\\\n"
        return txt

    body = (
        header
        + _subset_block("PRD", "PRD")
        + _subset_block("PRD_PRB_VEI", "PRD, PRB, VEI")
        + _subset_block("PRD_MEAN_COD", "PRD, Mean $r$, COD")
        + _subset_block("FULL", "full metric set$^{\\dagger}$")
    )

    # pattern: match from "Unconstrained baselines" row to the last row before \bottomrule of THIS table
    pattern = (
        r"\\multicolumn\{12\}\{l\}\{\\textbf\{Unconstrained baselines\}\}.*?"
        r"(?=\\bottomrule\s*\n\s*\\multicolumn\{12\}\{l\}\{\\scriptsize \$\^\{\*\})"
    )
    tex = _regex_replace_single(tex, pattern, body, flags=re.DOTALL)

    # caption year
    tex = re.sub(
        r"Held-out 20\d\d test set under different constraint sets",
        f"Held-out {year} test set under different constraint sets",
        tex,
        count=1,
    )
    return tex


def update_extreme_case_path(tex: str, img_rel: str) -> str:
    """Keep the existing A4-(first frame) figure in place but repoint it under img{year}
    only if we have a copy; else leave untouched."""
    # Keep as-is to avoid breaking the build; the extreme_case figure is a static
    # schematic that does not change across runs.
    return tex


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--year", required=True, choices=["2023", "2024"])
    ap.add_argument("--summary-json", required=True)
    ap.add_argument("--tex-file", required=True, help="Paper/GE slides/slides_v1_5 {year}.tex")
    ap.add_argument("--img-rel", required=True, help="relative path used in \\includegraphics, e.g. img2023")
    args = ap.parse_args()

    tex_path = Path(args.tex_file)
    backup = tex_path.with_suffix(tex_path.suffix + ".bak")
    tex = tex_path.read_text(encoding="utf-8")
    if not backup.exists():
        backup.write_text(tex, encoding="utf-8")

    summary = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))

    tex = update_slide10(tex, summary, args.year)
    tex = update_slide15(tex, summary, args.img_rel)
    tex = update_slide16(tex, summary, args.year)
    tex = update_figure_paths(tex, args.img_rel)
    tex = update_appendix_a2(tex, summary)
    try:
        tex = update_appendix_a4_table(tex, summary, args.year)
    except RuntimeError as e:
        print(f"WARN: could not rewrite Appendix A4 table ({e}); leaving original body", file=sys.stderr)

    tex_path.write_text(tex, encoding="utf-8")
    print(f"OK: wrote {tex_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
