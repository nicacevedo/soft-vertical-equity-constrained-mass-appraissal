#!/usr/bin/env python3
"""Matched-beta figures. The horizontal coordinate is ALWAYS the matched development
beta_log target, never rho and never b -- that is the whole point of the comparison."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p0_common as c

T, FIG = c.TABLES, c.FIGURES
STYLE = {"Direct":     dict(color="#1b6ca8", marker="o", ls="-",  lw=1.7, ms=5, zorder=4),
         "Surrogate":  dict(color="#c0392b", marker="s", ls="-",  lw=1.7, ms=5, zorder=4),
         "C-posthoc":  dict(color="#2e7d32", marker="^", ls="--", lw=1.7, ms=5, zorder=3),
         "A-posthoc":  dict(color="#7f8c8d", marker="v", ls=":",  lw=1.4, ms=4, zorder=2)}
LBL = {"Direct": "Direct (retrained)", "Surrogate": "Surrogate (retrained)",
       "C-posthoc": "Centered-spread post-hoc on Cell C (PRIMARY comparator)",
       "A-posthoc": "Centered-spread post-hoc on Cell A (secondary)"}
EVL = {"CV_mean": "Development (7-fold CV mean)", "heldout": "Held-out (2023-11 to 2024-12)",
       "forward_2025": "2025 forward"}


def _panel(ax, d, metric, reg, ylabel, title=None):
    s = d[(d.evaluation == reg) & d.attained]
    for fam in ("A-posthoc", "C-posthoc", "Direct", "Surrogate"):
        g = s[s.family == fam].sort_values("target")
        v = pd.to_numeric(g[metric], errors="coerce")
        ok = v.notna()
        if not ok.any():
            continue
        ax.plot(g.target[ok], v[ok], label=LBL[fam], **STYLE[fam])
    ax.set_xlabel(r"matched development $\beta_{\log}$ target")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, lw=0.5)
    if title:
        ax.set_title(title, fontsize=9)


def fig_accuracy_equity(d):
    fig, axes = plt.subplots(3, 3, figsize=(13.2, 10.2))
    spec = [("R2_price", r"$R^2$ (price)"), ("RMSE_log", r"RMSE (log price)"), ("COD", "COD")]
    for i, (m, yl) in enumerate(spec):
        for k, reg in enumerate(("CV_mean", "heldout", "forward_2025")):
            _panel(axes[i][k], d, m, reg, yl, EVL[reg] if i == 0 else None)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.012))
    fig.suptitle("Accuracy and assessor-facing equity at matched development "
                 r"$\beta_{\log}$", fontsize=12)
    fig.tight_layout(rect=(0, 0.055, 1, 0.975))
    p = FIG / "matched_beta_accuracy_equity.pdf"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def fig_mechanism(d):
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.0))
    for i, (m, yl) in enumerate([("Delta_NL", r"$\Delta_{NL}$"),
                                 ("dCor_e_y", r"dCor$(e,y)$")]):
        for k, reg in enumerate(("CV_mean", "heldout", "forward_2025")):
            _panel(axes[i][k], d, m, reg, yl, EVL[reg] if i == 0 else None)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(r"Mechanism coordinates at matched development $\beta_{\log}$: "
                 "the first-order match does NOT equalize the higher-order structure",
                 fontsize=11.5)
    fig.tight_layout(rect=(0, 0.08, 1, 0.955))
    p = FIG / "matched_beta_mechanism.pdf"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def fig_ratio_profiles():
    pr = pd.read_csv(T / "matched_beta_ratio_profiles.csv")
    js = [0, 3, 5]
    fig, axes = plt.subplots(2, len(js), figsize=(4.35 * len(js), 7.2), sharey="row")
    for r, ev in enumerate(("heldout", "forward_2025")):
        for cix, j in enumerate(js):
            ax = axes[r][cix]
            s = pr[(pr.evaluation == ev) & (pr.j == j)]
            for fam in ("A-posthoc", "C-posthoc", "Direct", "Surrogate"):
                g = s[s.family == fam].sort_values("group")
                if not len(g):
                    continue
                st = dict(STYLE[fam]); st.pop("zorder", None)
                ax.plot(range(1, len(g) + 1), g.median_ratio, label=LBL[fam], **st)
                if fam in ("Direct", "C-posthoc"):
                    ax.fill_between(range(1, len(g) + 1), g.ci_low, g.ci_high,
                                    color=STYLE[fam]["color"], alpha=0.13, lw=0)
            ax.axhline(1.0, color="k", lw=0.8, ls="-", alpha=0.55)
            ax.grid(alpha=0.25, lw=0.5)
            ax.set_xlabel("IAAO VEI proxy group (low $\\rightarrow$ high value)")
            if cix == 0:
                ax.set_ylabel(f"{EVL[ev]}\nmedian valuation ratio")
            tg = s.target.iloc[0] if len(s) else float("nan")
            if r == 0:
                ax.set_title(f"$j={j}$   target $\\beta_{{\\log}}={tg:.4f}$", fontsize=9.5)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.015))
    fig.suptitle("Ratio shape at matched development $\\beta_{\\log}$ "
                 "(90% bootstrap CI shown for Direct and the primary comparator)",
                 fontsize=11.5)
    fig.tight_layout(rect=(0, 0.075, 1, 0.955))
    p = FIG / "matched_beta_ratio_profiles.pdf"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def main() -> int:
    d = pd.read_csv(T / "matched_beta_comparison.csv")
    for p in (fig_accuracy_equity(d), fig_mechanism(d), fig_ratio_profiles()):
        print(f"[fig] {p.relative_to(c.REPO)}  ({p.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
