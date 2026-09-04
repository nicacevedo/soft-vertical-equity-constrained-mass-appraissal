#!/usr/bin/env python3
"""Forward-2025 evolution atlas, ratio profiles, stability, and paper figures.

Uses frozen CV coordinates only. Does not select rho from 2025.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from analysis.external_jurisdiction_benchmark_v1.scripts.forward_common import (  # noqa: E402
    ANALYSIS, DIRECT_COMMON_INTERVAL, OUTPUT, frozen_anchor_points,
)
from analysis.external_jurisdiction_benchmark_v1.scripts.v1_common import ALL_KEYS, JURISDICTION_BY_KEY  # noqa: E402

FIG = ANALYSIS / "figures"
PRED = OUTPUT / "forward_2025" / "predictions"
LABELS = {k: JURISDICTION_BY_KEY[k]["label"] for k in ALL_KEYS}
CV_COLOR = "#4c78a8"
FWD_COLOR = "#e45756"
IDEAL = {
    "median_ratio": 1.0, "mean_ratio": 1.0, "PRD": 1.0, "PRB": 0.0, "MKI": 1.0,
    "VEI": 0.0, "beta_log": 0.0, "COD": None, "COV": None,
}


def _x(rho: np.ndarray) -> np.ndarray:
    rho = np.asarray(rho, dtype=float)
    out = np.full_like(rho, np.nan, dtype=float)
    pos = rho > 0
    out[pos] = np.log10(rho[pos])
    if (~pos).any():
        lo = np.nanmin(out) if np.isfinite(out).any() else -3.0
        out[~pos] = lo - 0.45
    return out


def _load():
    fwd = pd.read_csv(ANALYSIS / "forward_2025" / "metrics" / "forward_2025_path_metrics.csv")
    cmp_ = pd.read_csv(ANALYSIS / "forward_2025" / "metrics" / "cv_forward_path_comparison.csv")
    cand = pd.read_csv(ANALYSIS / "candidate_regions" / "candidate_regions.csv")
    return fwd, cmp_, cand


def _cand_row(cand, key, family):
    hit = cand.loc[(cand.county_key == key) & (cand.family == family)]
    return None if not len(hit) else hit.iloc[0]


def mark_region(ax, crow, family: str) -> None:
    if crow is None:
        return
    status = str(crow.get("status", ""))
    a, g = crow.get("activity_rho_tilde"), crow.get("guardrail_rho_tilde")
    if pd.notna(a):
        ax.axvline(_x(np.array([a]))[0], color="#2c7bb6", ls=":", lw=0.9, zorder=0)
    if pd.notna(g):
        ax.axvline(_x(np.array([g]))[0], color="#d7191c", ls=":", lw=0.9, zorder=0)
    if status == "CANDIDATE_REGION" and pd.notna(a) and pd.notna(g) and a <= g:
        ax.axvspan(_x(np.array([a]))[0], _x(np.array([g]))[0], color="#a6d96a", alpha=0.12, zorder=0)
    elif pd.notna(a) and pd.notna(g):
        ax.axvspan(_x(np.array([min(a, g)]))[0], _x(np.array([max(a, g)]))[0], color="#cccccc", alpha=0.18, zorder=0)
    if family == "direct":
        lo, hi = DIRECT_COMMON_INTERVAL
        ax.axvspan(_x(np.array([lo]))[0], _x(np.array([hi]))[0], color="#fdae61", alpha=0.08, zorder=0)


def plot_metric(ax, cmp_sub, fwd_sub, metric, crow, family, ylabel=None):
    sub = cmp_sub.sort_values("rho_tilde")
    x = _x(sub["rho_tilde"].to_numpy())
    mean = sub.get(f"{metric}_cv_mean")
    lo = sub.get(f"{metric}_cv_min")
    hi = sub.get(f"{metric}_cv_max")
    if mean is not None:
        if lo is not None and hi is not None:
            ax.fill_between(x, lo, hi, color=CV_COLOR, alpha=0.18, linewidth=0)
        ax.plot(x, mean, color=CV_COLOR, lw=1.4, label="CV mean")
    fsub = fwd_sub.loc[fwd_sub.fit_status.astype(str).eq("OK")].sort_values("rho_tilde")
    if len(fsub) and metric in fsub.columns:
        ax.plot(_x(fsub["rho_tilde"].to_numpy()), fsub[metric], color=FWD_COLOR, lw=1.4, label="2025")
    if metric in IDEAL and IDEAL[metric] is not None:
        ax.axhline(IDEAL[metric], color="black", ls="--", lw=0.6)
    mark_region(ax, crow, family)
    ax.set_ylabel(ylabel or metric, fontsize=8)
    ax.tick_params(labelsize=7)


def path_figure(key, cmp_, fwd, cand, metrics, stem, title):
    fig, axes = plt.subplots(len(metrics), 2, figsize=(8.8, 1.55 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = np.array([axes])
    for j, family in enumerate(("direct", "surrogate")):
        crow = _cand_row(cand, key, family)
        csub = cmp_.loc[(cmp_.county_key == key) & (cmp_.family == family)]
        fsub = fwd.loc[(fwd.county_key == key) & (fwd.family == family)]
        for i, metric in enumerate(metrics):
            ax = axes[i, j]
            plot_metric(ax, csub, fsub, metric, crow, family)
            if i == 0:
                ax.set_title(family.capitalize(), fontsize=10)
            if i == len(metrics) - 1:
                ax.set_xlabel(r"$\log_{10}(\tilde\rho)$  (0 at left)", fontsize=8)
    fig.suptitle(f"{LABELS[key]} — {title}", fontsize=11)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=7, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = FIG / "path_evolution" / f"{key}_{stem}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def ratio_profiles_one(key, family, roles):
    fwd_path = PRED / f"{key}_{family}_forward_preds.parquet"
    oof_path = PRED / f"{key}_{family}_oof_anchor_preds.parquet"
    if not fwd_path.exists():
        return None
    fwd = pd.read_parquet(fwd_path)
    oof = pd.read_parquet(oof_path) if oof_path.exists() else None
    anchors = frozen_anchor_points(key, family)
    want = [p for p in anchors if p["role"] in roles]
    colors = {"baseline_rho0": "#4c78a8", "native_lgbm_baseline": "#4c78a8",
              "activity": "#54a24b", "A_beta_0.25": "#f58518",
              "A_beta_0.5": "#e45756", "guardrail": "#b279a2"}
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.4), sharex=True)
    panels = [
        (axes[0, 0], oof, "CV OOF (fold-pooled anchors)", False),
        (axes[0, 1], fwd, "2025 raw ratio", False),
        (axes[1, 0], oof, "CV OOF shape (ratio / median)", True),
        (axes[1, 1], fwd, "2025 shape (ratio / median)", True),
    ]
    for ax, src, title, shape in panels:
        ax.set_title(title, fontsize=9)
        ax.axhline(1.0, color="black", ls="--", lw=0.7)
        if src is None or not len(src):
            ax.text(5, 1.0, "not available", ha="center", fontsize=8)
            continue
        for p in want:
            sub = src.loc[np.isclose(src.rho_tilde.astype(float), p["rho_tilde"], atol=1e-10)].copy()
            if not len(sub):
                continue
            sub["ratio"] = sub["pred_price"] / sub["sale_price"]
            if shape:
                med = sub["ratio"].median()
                if pd.notna(med) and med != 0:
                    sub["ratio"] = sub["ratio"] / med
            sub = sub.loc[np.isfinite(sub["ratio"]) & sub["sale_price"].gt(0)]
            if not len(sub):
                continue
            sub["decile"] = pd.qcut(sub["sale_price"], 10, labels=False, duplicates="drop")
            prof = sub.groupby("decile")["ratio"].median()
            ax.plot(prof.index + 1, prof.values, marker="o", ms=3, lw=1.2,
                    color=colors.get(p["role"], "gray"), label=p["role"])
        ax.set_ylabel("median ratio" if not shape else "median ratio*")
        ax.set_xticks(range(1, 11))
    axes[1, 0].set_xlabel("sale-price decile")
    axes[1, 1].set_xlabel("sale-price decile")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=7, frameon=False)
    fig.suptitle(f"{LABELS[key]} {family} ratio profiles (frozen anchors)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = FIG / "ratio_profiles" / f"{key}_{family}_ratio_profiles.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def stability_figure(cmp_):
    metrics = ["Delta_NMSE", "Delta_I_PRB", "Delta_I_beta_log", "Delta_NL", "dCor"]
    fig, axes = plt.subplots(len(metrics), 2, figsize=(8.8, 8.5), sharex=True)
    for j, family in enumerate(("direct", "surrogate")):
        for i, metric in enumerate(metrics):
            ax = axes[i, j]
            col = f"{metric}_2025_minus_cv"
            if col not in cmp_.columns:
                # fall back: 2025 - cv mean of the metric itself
                col = f"{metric}_2025_minus_cv"
            for key in ALL_KEYS:
                sub = cmp_.loc[(cmp_.county_key == key) & (cmp_.family == family)].sort_values("rho_tilde")
                ycol = f"{metric}_2025_minus_cv"
                if ycol not in sub.columns:
                    continue
                ax.plot(_x(sub.rho_tilde.to_numpy()), sub[ycol], lw=1.0, alpha=0.85, label=key if i == 0 and j == 0 else None)
            ax.axhline(0, color="black", ls="--", lw=0.6)
            ax.set_ylabel(r"$\Delta$" + metric, fontsize=8)
            if i == 0:
                ax.set_title(family.capitalize(), fontsize=10)
            if i == len(metrics) - 1:
                ax.set_xlabel(r"$\log_{10}(\tilde\rho)$", fontsize=8)
    fig.suptitle("2025 minus CV-mean path drift", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = FIG / "forward_stability" / "cv_to_2025_path_drift.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def frontier(cmp_):
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.2), sharey=True)
    anc = pd.read_csv(ANALYSIS / "forward_2025" / "metrics" / "forward_anchor_metrics.csv")
    for ax, family in zip(axes, ("direct", "surrogate")):
        ax.set_title(family.capitalize())
        ax.set_xlabel(r"$A_\beta$")
        ax.set_ylabel(r"$\Delta$ NMSE")
        ax.axhline(0, color="black", ls="--", lw=0.6)
        for key in ALL_KEYS:
            sub = cmp_.loc[(cmp_.county_key == key) & (cmp_.family == family)].sort_values("rho_tilde")
            ax.plot(sub["A_beta_cv_mean"], sub["Delta_NMSE_cv_mean"], color=CV_COLOR, lw=1.0, alpha=0.7)
            ax.plot(sub["A_beta_2025"], sub["Delta_NMSE_2025"], color=FWD_COLOR, lw=1.0, alpha=0.85)
            a = anc.loc[(anc.county_key == key) & (anc.family == family)]
            for role, mk in (("A_beta_0.25", "o"), ("A_beta_0.5", "s"), ("A_beta_0.75", "^")):
                r = a.loc[a.role == role]
                if not len(r) or pd.isna(r.iloc[0].get("A_beta_2025")):
                    continue
                ax.scatter(r.iloc[0]["A_beta_2025"], r.iloc[0]["Delta_NMSE_2025"],
                           marker=mk, color=FWD_COLOR, s=18, zorder=3)
        ax.set_xlim(-0.05, 1.05)
    fig.suptitle("Accuracy–mechanism frontier (CV vs 2025)", fontsize=11)
    fig.tight_layout()
    out = FIG / "paper" / "accuracy_mechanism_frontier_cv_vs_2025.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def key_metric_small_multiples(cmp_):
    metrics = ["beta_log", "PRB", "NMSE", "Delta_NL"]
    fig, axes = plt.subplots(9, 4, figsize=(10.5, 12.5), sharex=True)
    for i, key in enumerate(ALL_KEYS):
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            for family, ls in (("direct", "-"), ("surrogate", "--")):
                sub = cmp_.loc[(cmp_.county_key == key) & (cmp_.family == family)].sort_values("rho_tilde")
                ax.plot(_x(sub.rho_tilde.to_numpy()), sub[f"{metric}_cv_mean"], color=CV_COLOR, ls=ls, lw=1.0)
                ax.plot(_x(sub.rho_tilde.to_numpy()), sub[f"{metric}_2025"], color=FWD_COLOR, ls=ls, lw=1.0)
            if metric in IDEAL and IDEAL[metric] is not None:
                ax.axhline(IDEAL[metric], color="black", ls=":", lw=0.5)
            if i == 0:
                ax.set_title(metric, fontsize=9)
            if j == 0:
                ax.set_ylabel(key, fontsize=7)
    fig.suptitle("Nine-jurisdiction key paths (solid Direct, dashed Surrogate; blue CV, red 2025)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = FIG / "paper" / "forward_key_metric_paths_9jurisdictions.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def paper_ratio_examples():
    """Predeclared examples: Philadelphia, St. Louis County, Middlesex."""
    keys = ["philadelphia", "st_louis_county", "middlesex"]
    fig, axes = plt.subplots(3, 2, figsize=(8.6, 8.0), sharex=True)
    for i, key in enumerate(keys):
        for j, family in enumerate(("direct", "surrogate")):
            ax = axes[i, j]
            path = PRED / f"{key}_{family}_forward_preds.parquet"
            ax.axhline(1.0, color="black", ls="--", lw=0.6)
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            for role, color in (("baseline_rho0", "#4c78a8"), ("native_lgbm_baseline", "#4c78a8"),
                                ("A_beta_0.25", "#f58518"), ("A_beta_0.5", "#e45756")):
                pts = [p for p in frozen_anchor_points(key, family) if p["role"] == role]
                if not pts:
                    continue
                sub = df.loc[np.isclose(df.rho_tilde.astype(float), pts[0]["rho_tilde"], atol=1e-10)].copy()
                if not len(sub):
                    continue
                sub["ratio"] = sub["pred_price"] / sub["sale_price"]
                sub = sub.loc[np.isfinite(sub.ratio) & sub.sale_price.gt(0)]
                sub["decile"] = pd.qcut(sub.sale_price, 10, labels=False, duplicates="drop")
                prof = sub.groupby("decile")["ratio"].median()
                ax.plot(prof.index + 1, prof.values, marker="o", ms=3, color=color, label=role)
            if i == 0:
                ax.set_title(family.capitalize())
            if j == 0:
                ax.set_ylabel(key)
            ax.set_xticks(range(1, 11))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Forward ratio profiles at frozen anchors (predeclared examples)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = FIG / "paper" / "forward_ratio_profile_examples.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def berry_context():
    berry = pd.read_csv(ANALYSIS / "berry" / "berry_ratio_decile_profiles.csv")
    mapping = [
        ("detroit_mi", "wayne", "Detroit (official) vs Wayne County AVM"),
        ("philadelphia_pa", "philadelphia", "Philadelphia official vs AVM"),
        ("st_louis_county_mo", "st_louis_county", "St. Louis County official vs AVM"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(8.8, 8.2))
    for i, (berry_j, key, title) in enumerate(mapping):
        ax_l, ax_r = axes[i]
        b = berry.loc[berry.jurisdiction == berry_j]
        for sample, ls in (("full_local_sample", "-"), ("attom_linkable_subset", "--")):
            sub = b.loc[b["sample"] == sample]
            if not len(sub):
                continue
            ax_l.plot(sub.price_decile + 1, sub.median_ratio, ls=ls, marker="o", ms=3, label=sample)
        ax_l.axhline(1.0, color="black", ls="--", lw=0.6)
        ax_l.set_ylabel("official assessment ratio")
        ax_l.set_title(title, fontsize=8)
        ax_l.legend(fontsize=6, frameon=False)
        path = PRED / f"{key}_direct_forward_preds.parquet"
        ax_r.axhline(1.0, color="black", ls="--", lw=0.6)
        ax_r.set_title(f"{key} Direct AVM 2025", fontsize=8)
        if path.exists():
            df = pd.read_parquet(path)
            for role, color in (("baseline_rho0", "#4c78a8"), ("A_beta_0.25", "#f58518"), ("A_beta_0.5", "#e45756")):
                pts = [p for p in frozen_anchor_points(key, "direct") if p["role"] == role]
                if not pts:
                    continue
                sub = df.loc[np.isclose(df.rho_tilde.astype(float), pts[0]["rho_tilde"], atol=1e-10)].copy()
                if not len(sub):
                    continue
                sub["ratio"] = sub["pred_price"] / sub["sale_price"]
                sub = sub.loc[np.isfinite(sub.ratio) & sub.sale_price.gt(0)]
                sub["decile"] = pd.qcut(sub.sale_price, 10, labels=False, duplicates="drop")
                prof = sub.groupby("decile")["ratio"].median()
                ax_r.plot(prof.index + 1, prof.values, marker="o", ms=3, color=color, label=role)
            ax_r.legend(fontsize=6, frameon=False)
        ax_r.set_ylabel("AVM model ratio")
        ax_l.set_xticks(range(1, 11))
        ax_r.set_xticks(range(1, 11))
    axes[2, 0].set_xlabel("value/sale-price decile")
    axes[2, 1].set_xlabel("sale-price decile")
    fig.suptitle("Official assessment ratios (left) are not AVM model ratios (right).\nWayne ≠ Detroit.", fontsize=10)
    fig.tight_layout()
    out = FIG / "paper" / "berry_local_vs_avm_ratio_profiles.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> int:
    fwd, cmp_, cand = _load()
    FIG.mkdir(parents=True, exist_ok=True)
    written = []
    for key in ALL_KEYS:
        written.append(path_figure(key, cmp_, fwd, cand,
            ["R2_price", "R2_log", "NMSE", "RMSE_log", "MAE", "MAPE"],
            "predictive_paths", "predictive paths"))
        written.append(path_figure(key, cmp_, fwd, cand,
            ["median_ratio", "mean_ratio", "COD", "COV"],
            "level_uniformity_paths", "level / uniformity"))
        written.append(path_figure(key, cmp_, fwd, cand,
            ["PRD", "PRB", "MKI", "VEI", "beta_log"],
            "vertical_equity_paths", "vertical equity"))
        written.append(path_figure(key, cmp_, fwd, cand,
            ["beta_log", "Delta_NL", "dCor", "A_beta"],
            "mechanism_paths", "mechanism"))
        for family in ("direct", "surrogate"):
            written.append(ratio_profiles_one(
                key, family,
                ["baseline_rho0", "native_lgbm_baseline", "activity", "A_beta_0.25", "A_beta_0.5", "guardrail"],
            ))
    written.append(stability_figure(cmp_))
    written.append(frontier(cmp_))
    written.append(key_metric_small_multiples(cmp_))
    written.append(paper_ratio_examples())
    written.append(berry_context())
    print("wrote", sum(p is not None for p in written), "figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
