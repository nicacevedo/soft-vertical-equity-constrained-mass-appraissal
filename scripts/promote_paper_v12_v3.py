#!/usr/bin/env python3
"""Promote approved v3 assets into paper/img and emit formatted table cells.

No TeX compilation. No model fitting.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
V3 = (
    REPO
    / "output/paper_v12_lower_rho_extension_994_v2/analysis/data_id=d4929d43ec19badf/split_id=3d464d4a611b131b/penalty_path_analysis/transition_regions_paper_assets_v3_followup"
)
PAPER_IMG = REPO / "paper/img/generated_v12_994"
CANONICAL = REPO / "output/paper_v6_preselection_994"

PROMOTE = [
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
]
NEW_FIGS = [
    ("vertical_equity_event_locations.pdf", "appendix_candidate"),
    ("mechanism_event_locations.pdf", "appendix_candidate"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tex_from_display(display: str, bold: bool, star: bool) -> str:
    if display.startswith("$"):
        inner = r"\$" + display[1:]
    else:
        inner = display.replace("%", r"\%")
    if bold:
        inner = r"\textbf{" + inner + "}"
    if star:
        inner = inner + r"\textsuperscript{*}"
    return r"\oldtext{\textit{source}}\latesttext{" + inner + "}"


def fmt_rho(x: float) -> str:
    v = float(x)
    if abs(v - 100.0) < 1e-8:
        return "100"
    if abs(v) < 1e-12:
        return "0"
    if abs(v - 1.0985411419875584) < 1e-8:
        return "1.099"
    if abs(v - 0.954095476349994) < 1e-8:
        return "0.954"
    if abs(v - 0.8286427728546845) < 1e-8:
        return "0.829"
    if v >= 1:
        return f"{v:.3f}".rstrip("0").rstrip(".")
    if v >= 0.1:
        return f"{v:.3f}".rstrip("0").rstrip(".")
    s = f"{v:.5g}"
    return s


def fmt_raw(metric: str, raw: float) -> str:
    x = float(raw)
    if metric == "MAE_price":
        return f"\\${x:,.0f}"
    if metric == "MAPE":
        pp = 100.0 * x
        if abs(pp) < 5e-4:
            return "0"
        return f"{pp:.3f} pp"
    if abs(x) < 5e-5:
        return "0"
    if metric == "R2_price":
        return f"{x:.4f}".rstrip("0").rstrip(".")
    if metric == "RMSE_log":
        return f"{x:.4g}"
    return f"{x:.4g}"


def fmt_norm(x: float) -> str:
    v = float(x)
    if abs(v) < 5e-5:
        return "0"
    return f"{v:.3f}"


def fmt_logd(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "--"
    v = float(x)
    if abs(v) < 1e-12:
        return "0"
    return f"{v:.3f}"


METRIC_TEX = {
    "R2_price": r"$R^2_P$",
    "MAE_price": r"$\operatorname{MAE}_P$",
    "MAPE": r"$\operatorname{MAPE}_P$",
    "RMSE_log": r"$\operatorname{RMSE}_{\log P}$",
    "COD": "COD",
}


TABLE_INPUTS = [
    "baseline_primary.csv",
    "baseline_secondary.csv",
    "representative_rho_primary.csv",
    "representative_rho_secondary.csv",
    "paper_source_placeholder_replacements.csv",
    "vertical_equity_event_locations.csv",
    "level_uniformity_event_locations.csv",
    "mechanism_event_locations.csv",
    "delta_nl_oos_only_event_locations.csv",
    "transition_oos_span_regret_v3.csv",
    "transition_lofo_endpoint_summary_v3.csv",
    "event_sharpness_summary_v3.csv",
]


def copy_promoted_figures(rec: dict) -> None:
    src_main = V3 / "figures" / "main_candidate"
    src_app = V3 / "figures" / "appendix_candidate"
    for name in PROMOTE:
        src = src_main / name if (src_main / name).is_file() else src_app / name
        dest = PAPER_IMG / name
        old = sha256(dest) if dest.is_file() else None
        new_src = sha256(src)
        shutil.copy2(src, dest)
        after = sha256(dest)
        if after != new_src:
            raise RuntimeError(f"copy mismatch {name}")
        rec["copies"].append({"name": name, "old_sha256": old, "v3_sha256": new_src, "new_sha256": after})
    for name, folder in NEW_FIGS:
        src = V3 / "figures" / folder / name
        dest = PAPER_IMG / name
        new_src = sha256(src)
        shutil.copy2(src, dest)
        after = sha256(dest)
        if after != new_src:
            raise RuntimeError(f"copy mismatch {name}")
        rec["new"].append({"name": name, "v3_sha256": new_src, "new_sha256": after})


def regenerate_grid_figures(rec: dict) -> None:
    rec["grid_refresh"] = {"status": "not_attempted"}
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sys.path.insert(0, str(REPO))
        import utils.transition_paper_asset_plots as tp
        from utils.paper_v12_lower_rho_plots import apply_major_grid

        orig_save = tp._save

        def save_grid(plt_mod, fig, stem, guard):
            for ax in fig.axes:
                apply_major_grid(ax)
            return orig_save(plt_mod, fig, stem, guard)

        tp._save = save_grid

        class Guard:
            def allowed(self, path):
                p = Path(path)
                p.parent.mkdir(parents=True, exist_ok=True)
                return p

        outdir = V3 / "figures" / "grid_refresh"
        outdir.mkdir(parents=True, exist_ok=True)
        guard = Guard()
        names = ("baseline_models_motivation_2024_2025", "vei_percentile_group_profile")
        old_hashes = {n: sha256(PAPER_IMG / f"{n}.pdf") for n in names}
        tp.plot_baseline_motivation(plt, CANONICAL, guard, outdir / names[0])
        tp.plot_vei_groups(plt, CANONICAL, guard, outdir / names[1])
        rec["grid_refresh"] = {"status": "copied", "items": []}
        for n in names:
            src = outdir / f"{n}.pdf"
            dest = PAPER_IMG / f"{n}.pdf"
            new_src = sha256(src)
            shutil.copy2(src, dest)
            after = sha256(dest)
            if after != new_src:
                raise RuntimeError(f"grid refresh copy mismatch {n}")
            rec["grid_refresh"]["items"].append(
                {"name": f"{n}.pdf", "old_sha256": old_hashes[n], "generated_sha256": new_src, "new_sha256": after}
            )
    except Exception as exc:
        rec["grid_refresh"] = {"status": "left_unchanged", "reason": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    rec = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "copies": [],
        "new": [],
        "baseline_qa": [],
        "paper_tex_sha256_before": sha256(REPO / "paper" / "paper_v12.tex"),
        "v3_table_sha256": {name: sha256(V3 / "tables" / name) for name in TABLE_INPUTS},
        "paper_img_sha256_before": {p.name: sha256(p) for p in sorted(PAPER_IMG.glob("*.pdf"))},
        "v3_candidate_sha256": {},
    }
    src_main = V3 / "figures" / "main_candidate"
    src_app = V3 / "figures" / "appendix_candidate"
    for name in PROMOTE:
        src = src_main / name if (src_main / name).is_file() else src_app / name
        rec["v3_candidate_sha256"][name] = sha256(src)
    for name, folder in NEW_FIGS:
        rec["v3_candidate_sha256"][name] = sha256(V3 / "figures" / folder / name)

    # Baseline QA
    bpri = pd.read_csv(V3 / "tables" / "baseline_primary.csv")
    bsec = pd.read_csv(V3 / "tables" / "baseline_secondary.csv")
    expected = {
        ("heldout", "Linear", "R2_price"): "0.799",
        ("heldout", "LightGBM", "R2_price"): "0.894",
        ("heldout", "Linear", "MAE_price"): "$90,092",
        ("heldout", "LightGBM", "MAE_price"): "$75,655",
        ("heldout", "Linear", "MAPE"): "24.1%",
        ("heldout", "LightGBM", "MAPE"): "21.2%",
        ("heldout", "Linear", "RMSE_log"): "0.322",
        ("heldout", "LightGBM", "RMSE_log"): "0.289",
        ("forward_2025", "Linear", "R2_price"): "0.799",
        ("forward_2025", "LightGBM", "R2_price"): "0.904",
        ("forward_2025", "Linear", "MAE_price"): "$99,371",
        ("forward_2025", "LightGBM", "MAE_price"): "$78,484",
        ("forward_2025", "Linear", "MAPE"): "24.9%",
        ("forward_2025", "LightGBM", "MAPE"): "20.8%",
        ("forward_2025", "Linear", "RMSE_log"): "0.313",
        ("forward_2025", "LightGBM", "RMSE_log"): "0.278",
        ("heldout", "Linear", "median_ratio"): "0.969",
        ("heldout", "LightGBM", "median_ratio"): "0.929",
        ("heldout", "Linear", "mean_ratio"): "1.020",
        ("heldout", "LightGBM", "mean_ratio"): "0.989",
        ("heldout", "Linear", "weighted_mean_ratio"): "0.979",
        ("heldout", "LightGBM", "weighted_mean_ratio"): "0.924",
        ("heldout", "Linear", "COD"): "24.7%",
        ("heldout", "LightGBM", "COD"): "21.6%",
        ("heldout", "Linear", "COV"): "45.2%",
        ("heldout", "LightGBM", "COV"): "39.7%",
        ("forward_2025", "Linear", "median_ratio"): "1.020",
        ("forward_2025", "LightGBM", "median_ratio"): "0.950",
        ("forward_2025", "Linear", "mean_ratio"): "1.081",
        ("forward_2025", "LightGBM", "mean_ratio"): "1.015",
        ("forward_2025", "Linear", "weighted_mean_ratio"): "1.027",
        ("forward_2025", "LightGBM", "weighted_mean_ratio"): "0.941",
        ("forward_2025", "Linear", "COD"): "24.3%",
        ("forward_2025", "LightGBM", "COD"): "21.3%",
        ("forward_2025", "Linear", "COV"): "42.1%",
        ("forward_2025", "LightGBM", "COV"): "37.2%",
        ("heldout", "Linear", "PRD"): "1.042",
        ("heldout", "LightGBM", "PRD"): "1.069",
        ("heldout", "Linear", "Beta_log"): "-0.092",
        ("heldout", "LightGBM", "Beta_log"): "-0.150",
        ("heldout", "Linear", "Delta_NL"): "0.131",
        ("heldout", "LightGBM", "Delta_NL"): "0.119",
        ("heldout", "Linear", "dCor_e_y"): "0.250",
        ("heldout", "LightGBM", "dCor_e_y"): "0.387",
        ("heldout", "Linear", "PRB"): "-0.016",
        ("heldout", "LightGBM", "PRB"): "-0.091",
        ("heldout", "Linear", "MKI"): "0.975",
        ("heldout", "LightGBM", "MKI"): "0.923",
        ("heldout", "Linear", "VEI"): "-11.6%",
        ("heldout", "LightGBM", "VEI"): "-26.5%",
        ("forward_2025", "Linear", "PRD"): "1.052",
        ("forward_2025", "LightGBM", "PRD"): "1.079",
        ("forward_2025", "Linear", "PRB"): "-0.029",
        ("forward_2025", "LightGBM", "PRB"): "-0.106",
        ("forward_2025", "Linear", "MKI"): "0.954",
        ("forward_2025", "LightGBM", "MKI"): "0.907",
        ("forward_2025", "Linear", "VEI"): "-17.3%",
        ("forward_2025", "LightGBM", "VEI"): "-28.6%",
        ("forward_2025", "Linear", "Beta_log"): "-0.109",
        ("forward_2025", "LightGBM", "Beta_log"): "-0.164",
        ("forward_2025", "Linear", "Delta_NL"): "0.122",
        ("forward_2025", "LightGBM", "Delta_NL"): "0.121",
        ("forward_2025", "Linear", "dCor_e_y"): "0.269",
        ("forward_2025", "LightGBM", "dCor_e_y"): "0.422",
    }
    both = pd.concat([bpri, bsec], ignore_index=True)
    problems = []
    for (split, fam, met), exp in expected.items():
        hit = both.loc[(both["split"] == split) & (both["family"] == fam) & (both["metric"] == met)]
        got = str(hit.iloc[0]["value_display"])
        if got != exp:
            problems.append(f"{split} {fam} {met}: {got} != {exp}")
    rec["baseline_qa"] = problems
    if problems:
        out = V3 / "qa" / "promote_paper_v12_v3.json"
        out.write_text(json.dumps(rec, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        print(json.dumps({"status": "FAIL_CLOSED_BASELINE", "baseline_qa": problems}, indent=2))
        return 2

    copy_promoted_figures(rec)
    regenerate_grid_figures(rec)

    ph = pd.read_csv(V3 / "tables" / "paper_source_placeholder_replacements.csv")
    rec["n_placeholders"] = int(len(ph))
    rec["primary_cells"] = []
    rec["secondary_cells"] = []
    for _, r in ph.iterrows():
        cell = tex_from_display(str(r["value_display"]), bool(r["manuscript_bold"]), bool(r["manuscript_asterisk"]))
        item = {
            "table": r["table"],
            "split": r["split"],
            "family": r["family"],
            "anchor": r["nominal_display_anchor"],
            "metric": r["metric"],
            "display": r["value_display"],
            "bold": bool(r["manuscript_bold"]),
            "star": bool(r["manuscript_asterisk"]),
            "tex": cell,
        }
        if r["table"] == "tab:path_anchor_summary":
            rec["primary_cells"].append(item)
        else:
            rec["secondary_cells"].append(item)

    # Cross-check complementary mechanism values already in paper
    sec = pd.read_csv(V3 / "tables" / "representative_rho_secondary.csv")
    rec["mechanism_crosscheck"] = []
    paper_mech = {
        ("heldout", "Direct", 0.1, "Beta_log"): "-0.147",
        ("heldout", "Direct", 0.1, "Delta_NL"): "0.119",
        ("heldout", "Direct", 0.1, "dCor_e_y"): "0.382",
        ("heldout", "Direct", 1.0, "Beta_log"): "-0.134",
        ("heldout", "Direct", 1.0, "Delta_NL"): "0.122",
        ("heldout", "Direct", 1.0, "dCor_e_y"): "0.359",
        ("heldout", "Direct", 10.0, "Beta_log"): "-0.096",
        ("heldout", "Direct", 10.0, "Delta_NL"): "0.132",
        ("heldout", "Direct", 10.0, "dCor_e_y"): "0.302",
        ("heldout", "Direct", 100.0, "Beta_log"): "-0.079",
        ("heldout", "Direct", 100.0, "Delta_NL"): "0.115",
        ("heldout", "Direct", 100.0, "dCor_e_y"): "0.265",
        ("heldout", "Surrogate", 0.1, "Beta_log"): "-0.138",
        ("heldout", "Surrogate", 0.1, "Delta_NL"): "0.113",
        ("heldout", "Surrogate", 0.1, "dCor_e_y"): "0.367",
        ("heldout", "Surrogate", 1.0, "Beta_log"): "-0.102",
        ("heldout", "Surrogate", 1.0, "Delta_NL"): "0.099",
        ("heldout", "Surrogate", 1.0, "dCor_e_y"): "0.310",
        ("heldout", "Surrogate", 10.0, "Beta_log"): "-0.036",
        ("heldout", "Surrogate", 10.0, "Delta_NL"): "0.103",
        ("heldout", "Surrogate", 10.0, "dCor_e_y"): "0.250",
        ("heldout", "Surrogate", 100.0, "Beta_log"): "-0.001",
        ("heldout", "Surrogate", 100.0, "Delta_NL"): "0.124",
        ("heldout", "Surrogate", 100.0, "dCor_e_y"): "0.267",
        ("forward_2025", "Direct", 0.1, "Beta_log"): "-0.160",
        ("forward_2025", "Direct", 0.1, "Delta_NL"): "0.121",
        ("forward_2025", "Direct", 0.1, "dCor_e_y"): "0.416",
        ("forward_2025", "Direct", 1.0, "Beta_log"): "-0.148",
        ("forward_2025", "Direct", 1.0, "Delta_NL"): "0.127",
        ("forward_2025", "Direct", 1.0, "dCor_e_y"): "0.392",
        ("forward_2025", "Direct", 10.0, "Beta_log"): "-0.105",
        ("forward_2025", "Direct", 10.0, "Delta_NL"): "0.134",
        ("forward_2025", "Direct", 10.0, "dCor_e_y"): "0.318",
        ("forward_2025", "Direct", 100.0, "Beta_log"): "-0.093",
        ("forward_2025", "Direct", 100.0, "Delta_NL"): "0.119",
        ("forward_2025", "Direct", 100.0, "dCor_e_y"): "0.281",
        ("forward_2025", "Surrogate", 0.1, "Beta_log"): "-0.153",
        ("forward_2025", "Surrogate", 0.1, "Delta_NL"): "0.116",
        ("forward_2025", "Surrogate", 0.1, "dCor_e_y"): "0.402",
        ("forward_2025", "Surrogate", 1.0, "Beta_log"): "-0.120",
        ("forward_2025", "Surrogate", 1.0, "Delta_NL"): "0.096",
        ("forward_2025", "Surrogate", 1.0, "dCor_e_y"): "0.343",
        ("forward_2025", "Surrogate", 10.0, "Beta_log"): "-0.050",
        ("forward_2025", "Surrogate", 10.0, "Delta_NL"): "0.092",
        ("forward_2025", "Surrogate", 10.0, "dCor_e_y"): "0.258",
        ("forward_2025", "Surrogate", 100.0, "Beta_log"): "-0.015",
        ("forward_2025", "Surrogate", 100.0, "Delta_NL"): "0.111",
        ("forward_2025", "Surrogate", 100.0, "dCor_e_y"): "0.266",
    }
    for (split, fam, anc, met), exp in paper_mech.items():
        hit = sec.loc[
            (sec["split"] == split)
            & (sec["family"] == fam)
            & (pd.to_numeric(sec["nominal_display_anchor"], errors="coerce") == float(anc))
            & (sec["metric"] == met)
        ]
        got = str(hit.iloc[0]["value_display"])
        if got != exp:
            rec["mechanism_crosscheck"].append(f"{split} {fam} {anc} {met}: {got} != {exp}")

    regret = pd.read_csv(V3 / "tables" / "transition_oos_span_regret_v3.csv")
    rec["regret_rows"] = []
    for split in ("heldout", "forward_2025"):
        for fam in ("Direct", "Surrogate"):
            for met in ("R2_price", "MAE_price", "MAPE", "RMSE_log", "COD"):
                r = regret.loc[(regret["split"] == split) & (regret["family"] == fam) & (regret["metric"] == met)].iloc[0]
                rec["regret_rows"].append(
                    {
                        "split": split,
                        "family": fam,
                        "metric": met,
                        "tex": (
                            f"{fam} & {METRIC_TEX[met]} & {fmt_rho(r['global_opt_rho'])} & {fmt_rho(r['best_inside_rho'])} & "
                            f"{fmt_raw(met, r['raw_regret'])} & {fmt_norm(r['normalized_regret'])} & {fmt_logd(r['log10_distance_global_opt_to_cv_span'])} \\\\"
                        ),
                    }
                )

    if rec["mechanism_crosscheck"]:
        out = V3 / "qa" / "promote_paper_v12_v3.json"
        out.write_text(json.dumps(rec, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        print(json.dumps({"status": "FAIL_CLOSED_MECHANISM", "mech": rec["mechanism_crosscheck"]}, indent=2))
        return 2

    paper_now = (REPO / "paper" / "paper_v12.tex").read_text(encoding="utf-8")
    if r"\textit{source}" in paper_now:
        from apply_paper_v12_v3_tex import apply_manuscript

        rec["tex_qa"] = apply_manuscript()
    rec["paper_tex_sha256_after"] = sha256(REPO / "paper" / "paper_v12.tex")

    out = V3 / "qa" / "promote_paper_v12_v3.json"
    out.write_text(json.dumps(rec, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({
        "baseline_qa": problems,
        "mech": rec["mechanism_crosscheck"],
        "n_ph": rec["n_placeholders"],
        "n_copies": len(rec["copies"]),
        "n_new": len(rec["new"]),
        "grid_refresh": rec.get("grid_refresh", {}).get("status"),
        "tex_qa": rec.get("tex_qa"),
    }, indent=2, default=str))
    tex_problems = (rec.get("tex_qa") or {}).get("problems") or []
    return 0 if not problems and not rec["mechanism_crosscheck"] and not tex_problems else 1


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:
        import traceback

        traceback.print_exc()
        rc = 1
    finally:
        os._exit(rc)
