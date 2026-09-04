#!/usr/bin/env python3
"""Steps 11-12 figures: candidate_region_heatmap.pdf and
raw_vs_normalized_region_alignment.pdf. Descriptive only -- does not select
or move any boundary."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from analysis.external_jurisdiction_benchmark_v1.scripts.v1_common import ANALYSIS  # noqa: E402


def heatmap_for_family(ax_top, ax_bottom, cand: pd.DataFrame, family: str, coverage_curve: dict | None) -> None:
    sub = cand.loc[cand.family == family].sort_values("county_key")
    keys = sub["county_key"].tolist()
    lo = max(1e-4, sub[["activity_rho_tilde", "guardrail_rho_tilde"]].min(numeric_only=True).min() or 1e-3)
    hi = sub[["activity_rho_tilde", "guardrail_rho_tilde"]].max(numeric_only=True).max() or 150.0
    lo, hi = lo / 3, hi * 3
    for i, (_, row) in enumerate(sub.iterrows()):
        a, g = row.get("activity_rho_tilde"), row.get("guardrail_rho_tilde")
        status = str(row.get("status", ""))
        if pd.isna(a) or pd.isna(g) or status != "CANDIDATE_REGION":
            label = status if status and status != "nan" else "NO_STABLE_CANDIDATE_REGION"
            ax_top.text(np.sqrt(lo * hi), i, label, ha="center", va="center", fontsize=7)
            if pd.notna(a) and pd.notna(g):
                ax_top.plot([a], [i], marker="o", color="#2c7bb6", markersize=4)
                ax_top.plot([g], [i], marker="s", color="#d7191c", markersize=4)
            continue
        ax_top.barh(i, g - a, left=a, height=0.6, color="#a6d96a")
        ax_top.plot([a], [i], marker="o", color="#2c7bb6", markersize=4)
        ax_top.plot([g], [i], marker="s", color="#d7191c", markersize=4)
    ax_top.set_xscale("log")
    ax_top.set_xlim(lo, hi)
    ax_top.set_yticks(range(len(keys)))
    ax_top.set_yticklabels(keys, fontsize=8)
    ax_top.set_title(f"{family} candidate regions (log10 rho_tilde)")
    if coverage_curve:
        ax_bottom.plot(coverage_curve["rho_tilde"], coverage_curve["coverage"], color="black")
        ax_bottom.axhline(0.75, color="red", ls="--", lw=0.8)
        ax_bottom.set_xscale("log")
        ax_bottom.set_xlim(lo, hi)
        ax_bottom.set_ylabel("coverage")
        ax_bottom.set_xlabel("rho_tilde")


def main() -> int:
    cand = pd.read_csv(ANALYSIS / "candidate_regions" / "candidate_regions.csv")
    fig, axes = plt.subplots(4, 1, figsize=(8, 10), gridspec_kw={"height_ratios": [3, 1, 3, 1]})
    for fam, (ax_top, ax_bottom) in zip(("direct", "surrogate"), ((axes[0], axes[1]), (axes[2], axes[3]))):
        cov_path = ANALYSIS / "candidate_regions" / f"{fam}_coverage_curve.json"
        cov = json.loads(cov_path.read_text()) if cov_path.exists() else None
        heatmap_for_family(ax_top, ax_bottom, cand, fam, cov)
    fig.tight_layout()
    ANALYSIS.joinpath("figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(ANALYSIS / "figures" / "candidate_region_heatmap.pdf")
    plt.close(fig)

    port = pd.read_csv(ANALYSIS / "tables" / "normalization_portability.csv")
    fig2, ax = plt.subplots(figsize=(6, 5))
    for _, row in port.iterrows():
        if row.get("sample", "all_point_estimates") != "all_point_estimates":
            continue
        if pd.isna(row["sd_log10_rho_tilde"]) or pd.isna(row["sd_log10_raw_rho"]):
            continue
        ax.plot([0, 1], [row["sd_log10_raw_rho"], row["sd_log10_rho_tilde"]], marker="o",
                label=f"{row['family']}/{row['endpoint']}")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["raw rho", "rho_tilde (normalized)"])
    ax.set_ylabel("cross-jurisdiction SD of log10(endpoint)")
    ax.set_title("Raw vs normalized endpoint dispersion (all point estimates)")
    ax.legend(fontsize=6)
    fig2.tight_layout()
    fig2.savefig(ANALYSIS / "figures" / "raw_vs_normalized_region_alignment.pdf")
    plt.close(fig2)
    print("wrote candidate_region_heatmap.pdf and raw_vs_normalized_region_alignment.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
