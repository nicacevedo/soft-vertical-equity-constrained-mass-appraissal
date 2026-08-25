"""Unified follow-up figures: every comparable toy model on the same axes.

Reads already-assembled V2/V3 artifacts. Does not refit. V1/V2 sources are
immutable; figures are written only under the V3 follow-up root.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import toy_mechanism_selection as six
import toy_surrogate_ablation as v1

REPO = Path(__file__).resolve().parents[1]
V2 = REPO / "output" / "toy_surrogate_ablation_v2"
V3 = REPO / "output" / "toy_surrogate_followup_v3"

SIX_PATH = (
    "current_direct",
    "direct_mm_k1",
    "quadratic",
    "moment_mm_k2",
    "moment_mm_k3",
    "local_slope_smooth",
)
V3_FAMS = ("qd_a25", "qd_a30", "qnl_a25", "qnl_a30")
RATIO_DISPLAY_S: Tuple[float, ...] = (1.00, 0.80, 0.60, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10)
MATCHED_S: Tuple[float, ...] = (0.20, 0.15, 0.10)

SERIES_ORDER = (
    "current_direct",
    "direct_mm_k1",
    "quadratic",
    "moment_mm_k2",
    "moment_mm_k3",
    "local_slope_smooth",
    "huber",
    "absolute",
    "qd_a25",
    "qd_a30",
    "qnl_a25",
    "qnl_a30",
)
TITLES = {
    "current_direct": "Current Direct",
    "direct_mm_k1": "Direct-MM K=1",
    "quadratic": "Quadratic",
    "moment_mm_k2": "Moment-MM K=2",
    "moment_mm_k3": "Moment-MM K=3",
    "local_slope_smooth": "Local-Slope",
    "huber": "Huber",
    "absolute": "Absolute",
    "qd_a25": "QD (anchor 0.25)",
    "qd_a30": "QD (anchor 0.30)",
    "qnl_a25": "QNL (anchor 0.25)",
    "qnl_a30": "QNL (anchor 0.30)",
}
COLORS = {
    "current_direct": "#111827",
    "direct_mm_k1": "#1D4ED8",
    "quadratic": "#BE123C",
    "moment_mm_k2": "#047857",
    "moment_mm_k3": "#6D28D9",
    "local_slope_smooth": "#B45309",
    "huber": "#CA8A04",
    "absolute": "#9CA3AF",
    "qd_a25": "#0F766E",
    "qd_a30": "#2DD4BF",
    "qnl_a25": "#A16207",
    "qnl_a30": "#F59E0B",
}
LINESTYLES = {
    "current_direct": "-",
    "direct_mm_k1": "-",
    "quadratic": "-",
    "moment_mm_k2": "-",
    "moment_mm_k3": "-",
    "local_slope_smooth": "-",
    "huber": ":",
    "absolute": ":",
    "qd_a25": "--",
    "qd_a30": "--",
    "qnl_a25": "-.",
    "qnl_a30": "-.",
}
PRIMARY_YLIM = {
    "current_direct",
    "direct_mm_k1",
    "quadratic",
    "moment_mm_k2",
    "moment_mm_k3",
    "local_slope_smooth",
    "qd_a25",
    "qd_a30",
    "qnl_a25",
    "qnl_a30",
}


def _finite(val: Any) -> bool:
    return six._finite(val)


def _attained(val: Any) -> bool:
    if val is True or val == 1:
        return True
    return str(val).strip().lower() in {"true", "1", "yes"}


def _target_close(series: pd.Series, sstar: float) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").apply(lambda x: bool(_finite(x) and abs(float(x) - float(sstar)) < 1e-8))


def resolve_pred_path(row: pd.Series, v3_root: Path) -> Optional[Path]:
    tag = str(row.get("pred_tag") or "")
    if not tag or tag.lower() in {"nan", "none", ""}:
        return None
    family = str(row.get("family") or "")
    method = str(row.get("method") or "")
    candidates = [
        v3_root / "families" / family / f"pred_{tag}.parquet",
        six.pred_path(V2, method, tag),
        six.pred_path(V2, family, tag),
        V2 / "families" / method / f"pred_{tag}.parquet",
        V2 / "families" / family / f"pred_{tag}.parquet",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def _load_huber_absolute() -> pd.DataFrame:
    frames = []
    for fam in ("huber", "absolute"):
        path = V2 / "families" / fam / "metrics.csv"
        if not path.is_file():
            continue
        df = pd.read_csv(path)
        if "level_invariant" in df.columns:
            li = df["level_invariant"]
            if li.dtype == bool:
                df = df.loc[~li]
            else:
                df = df.loc[~li.astype(str).str.lower().isin({"true", "1", "yes"})]
        if "status" in df.columns:
            df = df.loc[df["status"].astype(str).eq("ok")]
        df = df.copy()
        df["method"] = fam
        df["family"] = fam
        df["series_id"] = fam
        df["provenance"] = "reused_v2_family_metrics"
        df["attained"] = True
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _row_at_s(sub: pd.DataFrame, sstar: float, tol: float = 0.03) -> Optional[pd.Series]:
    """Prefer an exact target_s lock; otherwise the nearest s_train within `tol`."""
    if sub.empty:
        return None
    if "target_s" in sub.columns:
        hit = sub.loc[_target_close(sub["target_s"], sstar)]
        if not hit.empty:
            return hit.iloc[0]
    dist = (pd.to_numeric(sub["s_train"], errors="coerce") - float(sstar)).abs()
    if dist.notna().any() and float(dist.min()) <= tol:
        return sub.loc[dist.idxmin()]
    return None


def build_series_table(shape: pd.DataFrame) -> pd.DataFrame:
    """One comparable series per model/family, with duplicates from hybrid reuse dropped."""
    rows: List[pd.DataFrame] = []
    work = shape.copy()
    if "provenance" not in work.columns:
        work["provenance"] = ""
    six_df = work.loc[work["method"].isin(SIX_PATH)].copy()
    six_df = six_df.loc[~six_df["provenance"].astype(str).str.contains("first_hybrid|new_v3|hybrid_v2", na=False)]
    if "attained" in six_df.columns and "R2_price" in six_df.columns:
        six_df = six_df.loc[six_df["attained"].map(_attained) | six_df["R2_price"].map(_finite)]
    elif "attained" in six_df.columns:
        six_df = six_df.loc[six_df["attained"].map(_attained)]
    six_df["series_id"] = six_df["method"]
    rows.append(six_df)

    v3 = work.loc[work["family"].isin(V3_FAMS)].copy()
    if not v3.empty:
        v3 = v3.loc[v3["attained"].map(_attained)]
        v3["series_id"] = v3["family"]
        rows.append(v3)

    # V3 A25 continuation does not include s=0.10; keep the hybrid QD lock on qd_a25.
    extra = work.loc[work["method"].eq("quadratic_direct_cap")].copy()
    if not extra.empty:
        have = set()
        if not v3.empty:
            a25 = v3.loc[v3["family"].eq("qd_a25")]
            if "target_s" in a25.columns:
                have = set(float(x) for x in a25["target_s"] if _finite(x))
        add = extra.loc[extra["attained"].map(_attained)]
        add = add.loc[add["target_s"].apply(lambda s: _finite(s) and abs(float(s) - 0.10) < 1e-8)]
        add = add.loc[~add["provenance"].astype(str).str.contains("first_hybrid", na=False)]
        add = add.loc[~add["target_s"].apply(lambda s: any(abs(float(s) - h) < 1e-8 for h in have))]
        if not add.empty:
            add = add.copy()
            add["series_id"] = "qd_a25"
            add["family"] = "qd_a25"
            rows.append(add)

    habs = _load_huber_absolute()
    if not habs.empty:
        rows.append(habs)

    out = pd.concat(rows, ignore_index=True, sort=False)
    out = out.loc[out["series_id"].isin(SERIES_ORDER)].copy()
    if "R2_price" in out.columns:
        out = out.loc[out["R2_price"].map(_finite)]
    if "s_train" in out.columns:
        out["_sk"] = (
            out["series_id"].astype(str)
            + "|"
            + pd.to_numeric(out["s_train"], errors="coerce").round(8).astype(str)
            + "|"
            + out.get("pred_tag", pd.Series([""] * len(out))).fillna("").astype(str)
        )
        out = out.drop_duplicates("_sk").drop(columns=["_sk"])
    return out.reset_index(drop=True)


def _ylim_from_primary(df: pd.DataFrame, col: str) -> Tuple[float, float]:
    if col not in df.columns:
        return (0.0, 1.0)
    prim = df.loc[df["series_id"].isin(PRIMARY_YLIM)]
    vals = pd.to_numeric(prim[col], errors="coerce")
    vals = vals[np.isfinite(vals)]
    if vals.empty:
        vals = pd.to_numeric(df[col], errors="coerce")
        vals = vals[np.isfinite(vals)]
    return six._padded(vals) if len(vals) else (0.0, 1.0)


def _draw_series(
    ax,
    df: pd.DataFrame,
    *,
    xcol: str,
    ycol: str,
    ylim: Tuple[float, float],
    xlim: Optional[Tuple[float, float]] = None,
    clip_rows: List[dict],
    figure: str,
) -> None:
    if ycol not in df.columns:
        return
    for sid in SERIES_ORDER:
        sub = df.loc[df["series_id"] == sid].sort_values(xcol)
        if sub.empty:
            continue
        x = pd.to_numeric(sub[xcol], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(sub[ycol], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]
        if x.size == 0:
            continue
        yplot = np.clip(y, ylim[0], ylim[1])
        xplot = np.clip(x, xlim[0], xlim[1]) if xlim is not None else x
        clipped = ~np.isclose(y, yplot, atol=1e-12)
        if xlim is not None:
            clipped = clipped | (~np.isclose(x, xplot, atol=1e-12))
        lw = 1.0 if sid in {"huber", "absolute"} else 1.45
        alpha = 0.55 if sid in {"huber", "absolute"} else 1.0
        ax.plot(
            xplot,
            yplot,
            color=COLORS[sid],
            ls=LINESTYLES[sid],
            marker="o",
            ms=3.6,
            lw=lw,
            alpha=alpha,
            label=TITLES[sid],
        )
        if np.any(clipped):
            ax.scatter(xplot[clipped], yplot[clipped], marker="^", s=22, color="#B91C1C", zorder=5)
            for xi, yi, ypi in zip(x[clipped], y[clipped], yplot[clipped]):
                clip_rows.append(
                    {
                        "figure": figure,
                        "series_id": sid,
                        "metric": ycol,
                        "s_train": float(xi) if xcol == "s_train" else float("nan"),
                        "x": float(xi),
                        "exact_value": float(yi),
                        "displayed_value": float(ypi),
                    }
                )


def _mark_anchors(ax) -> None:
    ax.axvline(0.30, color="#6B7280", ls=":", lw=0.8)
    ax.axvline(0.25, color="#6B7280", ls="--", lw=0.8)


def plot_lines_vs_s(
    df: pd.DataFrame,
    root: Path,
    *,
    stem: str,
    panels: Sequence[Tuple[str, str]],
    title: str,
    clip_rows: List[dict],
) -> Path:
    import matplotlib.pyplot as plt

    six._set_style()
    fig, axes = plt.subplots(len(panels), 1, figsize=(8.8, 2.15 * len(panels) + 1.2), sharex=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, (col, ylab) in zip(axes, panels):
        ylim = _ylim_from_primary(df, col)
        _draw_series(ax, df, xcol="s_train", ycol=col, ylim=ylim, clip_rows=clip_rows, figure=stem)
        _mark_anchors(ax)
        ax.set_xlim(1.05, -0.02)
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylab)
        ax.grid(True, color="#E5E7EB", lw=0.6)
        ax.set_axisbelow(True)
        if col == "Beta_log":
            ax.axhline(0.0, color="#111827", lw=0.6, ls=":")
    axes[-1].set_xlabel(r"Training signed retention $s$  ($1\rightarrow 0$)")
    axes[0].legend(frameon=False, fontsize=6.4, ncol=4, loc="best")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out = root / "figures" / stem
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out.with_suffix(".pdf")


def plot_tradeoff(df: pd.DataFrame, root: Path, clip_rows: List[dict]) -> Path:
    import matplotlib.pyplot as plt

    six._set_style()
    panels = [
        ("R2_price", r"held-out $R^2_P$"),
        ("MAE_price", r"held-out MAE"),
        ("dCor_e_y", r"held-out dCor"),
        ("NL_shape", r"NL$_{\mathrm{shape}}$"),
        ("N3_rel_eval", r"$N_{3,\mathrm{rel}}$"),
    ]
    beta_lim = _ylim_from_primary(df, "Beta_log")
    fig, axes = plt.subplots(1, 5, figsize=(14.6, 3.35))
    for ax, (col, ylab) in zip(axes, panels):
        ylim = _ylim_from_primary(df, col)
        _draw_series(
            ax,
            df,
            xcol="Beta_log",
            ycol=col,
            ylim=ylim,
            xlim=beta_lim,
            clip_rows=clip_rows,
            figure="tradeoff_vs_beta",
        )
        ax.set_xlabel(r"held-out $\beta_{\log}$")
        ax.set_ylabel(ylab)
        ax.set_xlim(*beta_lim)
        ax.set_ylim(*ylim)
        ax.grid(True, color="#E5E7EB", lw=0.6)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=5.8, loc="best", ncol=1)
    fig.suptitle("EXPERIMENTAL / TOY all-model tradeoffs vs held-out $\\beta_{\\log}$", fontsize=11)
    fig.tight_layout()
    out = root / "figures" / "tradeoff_vs_beta"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out.with_suffix(".pdf")


def plot_matched_bars(df: pd.DataFrame, root: Path) -> Optional[Path]:
    import matplotlib.pyplot as plt

    six._set_style()
    q = df.loc[df["series_id"].eq("quadratic")].copy()
    metrics = [
        ("R2_price", r"$R^2_P$"),
        ("MAE_price", "MAE"),
        ("Beta_log", r"$\beta_{\log}$"),
        ("dCor_e_y", "dCor"),
        ("NL_shape", r"NL$_{\mathrm{shape}}$"),
        ("N3_rel_eval", r"$N_{3,\mathrm{rel}}$"),
    ]
    fig, axes = plt.subplots(len(metrics), 3, figsize=(12.4, 2.05 * len(metrics) + 0.6), sharex="col")
    series_present = [s for s in SERIES_ORDER if s != "quadratic" and (df["series_id"] == s).any()]
    for col_i, sstar in enumerate(MATCHED_S):
        qrow = _row_at_s(q, sstar, tol=0.03)
        for row_i, (col, ylab) in enumerate(metrics):
            ax = axes[row_i, col_i]
            if col_i == 0:
                ax.set_ylabel(ylab, fontsize=8)
            if row_i == 0:
                ax.set_title(rf"$s^\star={sstar:.2f}$")
            ax.axhline(0.0, color="#111827", lw=0.55)
            if qrow is None or not _finite(qrow.get(col)):
                continue
            qval = float(qrow[col])
            xs, ys, cs = [], [], []
            for sid in series_present:
                hit = _row_at_s(df.loc[df["series_id"].eq(sid)], sstar, tol=0.03)
                if hit is None:
                    continue
                val = hit.get(col)
                if not _finite(val):
                    continue
                xs.append(TITLES[sid])
                if col == "Beta_log":
                    ys.append(abs(float(val)) - abs(qval))
                else:
                    ys.append(float(val) - qval)
                cs.append(COLORS[sid])
            if not xs:
                continue
            ax.bar(np.arange(len(xs)), ys, color=cs)
            ax.set_xticks(np.arange(len(xs)))
            ax.set_xticklabels(xs, rotation=65, ha="right", fontsize=5.8)
            ax.grid(True, axis="y", color="#E5E7EB")
    fig.suptitle("EXPERIMENTAL / TOY all-model matched difference vs Quadratic", fontsize=11)
    fig.tight_layout()
    out = root / "figures" / "matched_vs_quadratic"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out.with_suffix(".pdf")


def plot_ratio_grid(df: pd.DataFrame, root: Path, clip_rows: List[dict]) -> Optional[Path]:
    import matplotlib.pyplot as plt

    six._set_style()
    frames = []
    base = V2 / "predictions" / "lambda0_shared.parquet"
    if base.is_file():
        pred = pd.read_parquet(base)
        if "y_pred" not in pred.columns:
            pred["y_pred"] = np.exp(pred["y_pred_log"])
            pred["y_true"] = np.exp(pred["y_true_log"])
        b = v1.equal_count_bins(pred["y_true"].to_numpy(), pred["y_pred"].to_numpy() / np.clip(pred["y_true"].to_numpy(), 1e-12, None))
        b["series_id"] = "baseline"
        b["target_s"] = 1.0
        frames.append(b)
    for _, rec in df.iterrows():
        path = resolve_pred_path(rec, root)
        if path is None:
            continue
        pred = pd.read_parquet(path)
        if "y_pred" not in pred.columns:
            pred["y_pred"] = np.exp(pred["y_pred_log"])
            pred["y_true"] = np.exp(pred["y_true_log"])
        ratio = pred["y_pred"].to_numpy(dtype=float) / np.clip(pred["y_true"].to_numpy(dtype=float), 1e-12, None)
        if not np.all(np.isfinite(ratio)):
            continue
        b = v1.equal_count_bins(pred["y_true"].to_numpy(dtype=float), ratio)
        b["series_id"] = rec["series_id"]
        b["target_s"] = float(rec["target_s"]) if _finite(rec.get("target_s")) else float(rec.get("s_train", np.nan))
        frames.append(b)
    if not frames:
        return None
    bins = pd.concat(frames, ignore_index=True)
    six.atomic_csv(bins, root / "unified_ratio_bins.csv")
    series_with_pred = [s for s in SERIES_ORDER if (bins["series_id"] == s).any()]
    n = len(series_with_pred)
    ncols = 5
    nrows = int(np.ceil(n / ncols))
    axis_src = bins.loc[bins["series_id"].isin(["baseline", "quadratic"] + [s for s in series_with_pred if s in PRIMARY_YLIM])]
    ylim = six._padded(axis_src["median_ratio"].to_numpy(dtype=float))
    x_all = bins["median_sale_price"].to_numpy(dtype=float)
    xmin, xmax = float(np.nanmin(x_all)), float(np.nanmax(x_all))
    cmap = plt.cm.viridis
    display = list(RATIO_DISPLAY_S)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.05 * ncols, 2.7 * nrows + 0.4), sharex=True, sharey=True)
    axes_f = np.atleast_1d(axes).ravel()
    for ax, sid in zip(axes_f, series_with_pred):
        sub = bins.loc[bins["series_id"] == sid]
        baseb = bins.loc[bins["series_id"] == "baseline"].sort_values("bin")
        if not baseb.empty:
            ax.plot(baseb["median_sale_price"], np.clip(baseb["median_ratio"], ylim[0], ylim[1]), color="#9CA3AF", lw=1.1, label=r"$s=1$")
        for i, sstar in enumerate(display):
            if abs(sstar - 1.0) < 1e-12:
                continue
            part = sub.loc[np.isclose(sub["target_s"].astype(float), float(sstar), atol=8e-3)].sort_values("bin")
            if part.empty:
                continue
            y = part["median_ratio"].to_numpy(dtype=float)
            yplot = np.clip(y, ylim[0], ylim[1])
            clipped = ~np.isclose(y, yplot, atol=1e-12)
            ax.plot(
                part["median_sale_price"],
                yplot,
                color=cmap(0.12 + 0.8 * i / max(len(display) - 1, 1)),
                lw=1.25,
                marker="o",
                ms=1.6,
                label=rf"$s={sstar:.2f}$",
            )
            if np.any(clipped):
                for j in np.flatnonzero(clipped):
                    clip_rows.append(
                        {
                            "figure": "matched_ratio_shape",
                            "series_id": sid,
                            "target_s": float(sstar),
                            "bin": int(part.iloc[int(j)]["bin"]),
                            "exact_value": float(y[int(j)]),
                            "displayed_value": float(yplot[int(j)]),
                        }
                    )
        ax.axhline(1.0, color="#111827", ls=(0, (2, 2)), lw=0.65)
        ax.set_xscale("log")
        ax.set_ylim(*ylim)
        ax.set_xlim(xmin / 1.05, xmax * 1.05)
        ax.set_title(TITLES[sid], fontsize=8)
        ax.grid(True, color="#E5E7EB", lw=0.55)
        ax.set_xlabel("Sale price", fontsize=7)
        if sid == series_with_pred[0]:
            ax.set_ylabel("Valuation-to-sale ratio")
    for ax in axes_f[len(series_with_pred) :]:
        ax.axis("off")
    axes_f[len(series_with_pred) - 1].legend(fontsize=5.4, frameon=False, ncol=2)
    fig.suptitle("EXPERIMENTAL / TOY all-model matched ratio shapes", fontsize=11)
    fig.tight_layout()
    out = root / "figures" / "matched_ratio_shape"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out.with_suffix(".pdf")


def render_unified_figures(root: Path, shape: pd.DataFrame) -> Dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    v1._ensure_dir(root / "figures")
    series = build_series_table(shape)
    six.atomic_csv(series, root / "unified_plot_series.csv")
    clip_rows: List[dict] = []
    paths = {}
    paths["mechanism_vs_s"] = str(
        plot_lines_vs_s(
            series,
            root,
            stem="mechanism_vs_s",
            panels=[
                ("Beta_log", r"held-out $\beta_{\log}$"),
                ("dCor_e_y", r"held-out dCor$(e,y)$"),
                ("NL_shape", r"NL$_{\mathrm{shape}}$"),
                ("N3_rel_eval", r"$N_{3,\mathrm{rel}}$"),
                ("L_NL", r"smooth $L_{\mathrm{NL}}$"),
                ("R2_price", r"held-out $R^2_P$"),
            ],
            title="EXPERIMENTAL / TOY all-model mechanism vs training $s$",
            clip_rows=clip_rows,
        )
    )
    paths["assessor_vs_s"] = str(
        plot_lines_vs_s(
            series,
            root,
            stem="assessor_vs_s",
            panels=[
                ("PRD", "PRD"),
                ("PRB", "PRB"),
                ("COD", "COD"),
                ("VEI", "VEI"),
                ("MKI", "MKI"),
            ],
            title="EXPERIMENTAL / TOY all-model assessor-facing metrics vs training $s$",
            clip_rows=clip_rows,
        )
    )
    paths["moments_vs_s"] = str(
        plot_lines_vs_s(
            series,
            root,
            stem="moments_vs_s",
            panels=[
                ("m1_train", r"train $m_1$"),
                ("m2_train", r"train $m_2$"),
                ("m3_train", r"train $m_3$"),
                ("N3_orth_train", r"train $N_{3,\mathrm{orth}}$"),
                ("N3_rel_train", r"train $N_{3,\mathrm{rel}}$"),
            ],
            title="EXPERIMENTAL / TOY all-model training moment / shape vs $s$",
            clip_rows=clip_rows,
        )
    )
    paths["eval_moments_vs_s"] = str(
        plot_lines_vs_s(
            series,
            root,
            stem="eval_moments_vs_s",
            panels=[
                ("m1_eval", r"held-out $m_1$"),
                ("m2_eval", r"held-out $m_2$"),
                ("m3_eval", r"held-out $m_3$"),
                ("N3_orth_eval", r"held-out $N_{3,\mathrm{orth}}$"),
                ("N3_rel_eval", r"$N_{3,\mathrm{rel}}$"),
            ],
            title="EXPERIMENTAL / TOY all-model held-out moment / shape vs $s$",
            clip_rows=clip_rows,
        )
    )
    paths["tradeoff_vs_beta"] = str(plot_tradeoff(series, root, clip_rows))
    ratio = plot_ratio_grid(series, root, clip_rows)
    if ratio is not None:
        paths["matched_ratio_shape"] = str(ratio)
    matched = plot_matched_bars(series, root)
    if matched is not None:
        paths["matched_vs_quadratic"] = str(matched)
    # Keep the previous V3 filenames as copies of the all-model mechanism/ratio
    # so existing references still open a current comparison.
    import shutil

    for src, dest in (
        ("mechanism_vs_s", "main_paths"),
        ("mechanism_vs_s", "context_all_toy"),
        ("matched_ratio_shape", "ratio_shape"),
    ):
        sp = root / "figures" / f"{src}.pdf"
        if sp.is_file():
            shutil.copy2(sp, root / "figures" / f"{dest}.pdf")
            png = root / "figures" / f"{src}.png"
            if png.is_file():
                shutil.copy2(png, root / "figures" / f"{dest}.png")
            paths[dest] = str(root / "figures" / f"{dest}.pdf")
    if clip_rows:
        six.atomic_csv(pd.DataFrame(clip_rows), root / "plot_clipped_points.csv")
    return paths
