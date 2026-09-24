#!/usr/bin/env python3
"""LB Stage 3 -- regenerate the paired baseline ratio figure from verified predictions.

Layout follows the v15-era two-model x two-period grid that
scripts/populate_paper_v6_994.plot_baseline_motivation produced: median
valuation-to-sale ratio in 30 equal-count sale-price bins with a shaded
interquartile band, common conventions across all four panels.

The binning helper, styling, colours, log10 axis and beta_log annotation are
transcribed from that generator so the plotted convention is unchanged; the only
substantive differences are that (a) the row-level inputs are the artifacts
LB Stage 1 verified against the frozen benchmark, (b) the annotated beta_log is
taken from the executed metric code rather than recomputed inline, and (c) the
output goes to a new versioned path inside this stage.

Writes nothing under paper/.  Overwrites no existing figure.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lb_common as lb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                          # noqa: E402
from matplotlib.lines import Line2D                                      # noqa: E402
from matplotlib.ticker import LogLocator, LogFormatterSciNotation        # noqa: E402

STEM = "baseline_models_motivation_linear_vs_lgbm_2024_2025_v20r1"
N_BINS = 30
COLORS = {"linear": "#0072B2", "native": "#D55E00"}
TITLES = {"linear": "Ordinary linear regression", "native": "Ordinary LightGBM"}
SPLIT_LABS = {"heldout": "Held-out", "forward_2025": "2025"}


def equal_count_bins(sale: np.ndarray, ratio: np.ndarray, n_bins: int = N_BINS) -> pd.DataFrame:
    """Verbatim from scripts/populate_paper_v6_994.equal_count_bins."""
    order = np.argsort(sale, kind="mergesort")
    sale, ratio = sale[order], ratio[order]
    rows = []
    for i, idx in enumerate(np.array_split(np.arange(len(sale)), n_bins), start=1):
        if idx.size == 0:
            continue
        r = ratio[idx]
        rows.append({"bin": i,
                     "n": int(idx.size),
                     "median_sale_price": float(np.median(sale[idx])),
                     "median_ratio": float(np.median(r)),
                     "ratio_q25": float(np.quantile(r, 0.25)),
                     "ratio_q75": float(np.quantile(r, 0.75))})
    return pd.DataFrame(rows)


def padded_lim(values: np.ndarray, pad: float = 0.08):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    lo, hi = float(v.min()), float(v.max())
    span = hi - lo
    return lo - pad * span, hi + pad * span


def set_style() -> None:
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10.5, "axes.labelsize": 9.5,
                         "legend.fontsize": 8.0, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})


def main() -> int:
    metrics = pd.read_csv(lb.TABLES / "lb_metrics_full_precision_all_blocks.csv")
    profiles, panels = [], {}
    for split in ("heldout", "forward_2025"):
        d = lb.load_pair(split)
        for key, model in (("linear", lb.LINEAR_NAME), ("native", lb.NATIVE_NAME)):
            sale = np.exp(d["y_true_log"])
            av = np.exp(d["pred"][model])
            ratio = av / sale
            prof = equal_count_bins(sale, ratio)
            prof["split"] = split
            prof["model"] = model
            profiles.append(prof)
            beta = float(metrics[(metrics.model == model)
                                 & (metrics.evaluation == split)].iloc[0]["beta_log"])
            # cross-check against the inline formula the original generator used
            ylog, plog = d["y_true_log"], d["pred"][model]
            beta_inline = float(np.cov(plog - ylog, ylog, ddof=0)[0, 1] / np.var(ylog, ddof=0))
            if abs(beta - beta_inline) > 1e-12:
                raise lb.ProtocolViolation(
                    f"{split}/{key}: annotated beta_log {beta} disagrees with the "
                    f"generator's inline value {beta_inline}")
            panels[(split, key)] = {"sale": sale, "ratio": ratio, "beta": beta,
                                    "prof": prof, "n": d["n"]}

    profile = pd.concat(profiles, ignore_index=True)
    lb.write_table(profile, lb.TABLES / "lb_figure_ratio_profile_bins.csv")

    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.15), sharex=True, sharey=True)
    xmin, xmax = profile["median_sale_price"].min(), profile["median_sale_price"].max()
    pad = 0.04 * (np.log10(xmax) - np.log10(xmin))
    xlim = (10 ** (np.log10(xmin) - pad), 10 ** (np.log10(xmax) + pad))
    ymin, ymax = padded_lim(profile[["median_ratio", "ratio_q25", "ratio_q75"]].to_numpy().ravel())

    for r, split in enumerate(("heldout", "forward_2025")):
        for c, key in enumerate(("linear", "native")):
            ax = axes[r, c]
            P = panels[(split, key)]
            prof, color = P["prof"], COLORS[key]
            ax.fill_between(prof["median_sale_price"], prof["ratio_q25"], prof["ratio_q75"],
                            color=color, alpha=0.16, lw=0)
            ax.plot(prof["median_sale_price"], prof["median_ratio"], color=color,
                    marker="o", ms=2.5, lw=1.5)
            ax.axhline(1.0, color="#111827", ls=(0, (2, 2)), lw=0.9)
            ax.set_xscale("log", base=10)
            ax.set_xlim(*xlim)
            ax.set_ylim(ymin, ymax)
            ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            ax.xaxis.set_major_formatter(LogFormatterSciNotation(
                base=10, labelOnlyBase=False, minor_thresholds=(np.inf, np.inf)))
            ax.grid(True, color="#E5E7EB", lw=0.7)
            ax.set_axisbelow(True)
            ax.legend(handles=[Line2D([], [], ls="None",
                                      label=rf"$\beta_{{\log}}$ = {P['beta']:.3f}")],
                      loc="lower left", frameon=False, handlelength=0,
                      handletextpad=0, fontsize=7.5)
            if r == 0:
                ax.set_title(TITLES[key])
            if c == 0:
                ax.set_ylabel(f"{SPLIT_LABS[split]}\nValuation-to-sale ratio")
            if r == 1:
                ax.set_xlabel(r"Sale price (log$_{10}$ scale)")

    fig.legend(handles=[
        Line2D([0], [0], color="#111827", marker="o", lw=1.5, ms=3,
               label="Equal-count-bin median (IQR shaded)"),
        Line2D([0], [0], color="#111827", ls=(0, (2, 2)), lw=0.9, label="Ratio = 1")],
        loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    lb.FIGURES.mkdir(parents=True, exist_ok=True)
    pdf = lb.FIGURES / f"{STEM}.pdf"
    png = lb.FIGURES / f"{STEM}.png"
    lb._assert_write_allowed(pdf)
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=160, bbox_inches="tight")
    plt.close(fig)

    lb.write_json(lb.PROVENANCE / "lb3_figure.json", {
        "figure_pdf": str(pdf.relative_to(lb.REPO)),
        "figure_png": str(png.relative_to(lb.REPO)),
        "figure_pdf_sha256": lb.sha256_file(pdf),
        "figure_png_sha256": lb.sha256_file(png),
        "determinism": ("The PNG is byte-identical across independent runs. The PDF is not: "
                        "matplotlib stamps /CreationDate into it, so its sha256 changes on every "
                        "run while the drawn content does not. Verify the figure by the PNG hash "
                        "or by the plotted profile in tables/lb_figure_ratio_profile_bins.csv, "
                        "not by the PDF hash."),
        "layout": "2 rows (held-out, 2025) x 2 columns (linear, ordinary LightGBM)",
        "binning": f"{N_BINS} equal-count sale-price bins, median ratio, shaded interquartile band",
        "conventions_common_across_panels": [
            "same 30 equal-count binning rule", "shared x and y limits",
            "log10 sale-price axis", "ratio = 1 reference line",
            "beta_log annotation taken from the executed metric code"],
        "status": "DESCRIPTIVE. Ratio profiles on observed sales; not a standards-compliance "
                  "finding and not a causal statement about either model class.",
        "source_predictions": {
            "linear": "output/paper_v6_preselection_994/baseline_reporting/.../"
                      "{test,assess}_run_predictions/fd63507d2456c789.parquet",
            "ordinary LightGBM": "output/paper_v6_preselection_994/baseline_reporting/.../"
                                 "{test,assess}_run_predictions/252a25d9c0ce796b.parquet"},
        "verified_by": "analysis/linear_baseline_reproduction_v1/provenance/lb1_verdict.json",
        "generator_transcribed_from": "scripts/populate_paper_v6_994.plot_baseline_motivation",
        "does_not_overwrite": [
            "paper/img/generated_v12_994/baseline_models_motivation_2024_2025.pdf",
            "paper/img/generated_v6_preselection/baseline_models_motivation_2024_2025.pdf",
            "paper/img/baseline_models_motivation_2024_2025.pdf"],
        "panel_beta_log": {f"{s}/{k}": panels[(s, k)]["beta"]
                           for s in ("heldout", "forward_2025") for k in ("linear", "native")},
        "panel_n": {f"{s}/{k}": panels[(s, k)]["n"]
                    for s in ("heldout", "forward_2025") for k in ("linear", "native")},
    })
    print(f"[lb3] wrote {pdf}")
    for k, v in sorted({f"{s}/{k}": panels[(s, k)]["beta"]
                        for s in ("heldout", "forward_2025")
                        for k in ("linear", "native")}.items()):
        print(f"   beta_log {k:<22s} {v:+.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
