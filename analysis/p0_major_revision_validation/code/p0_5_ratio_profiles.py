#!/usr/bin/env python3
"""Ratio-shape diagnostics at the six matched-beta CORE targets.

Two profiles per (target, family, evaluation):
  * the canonical IAAO VEI proxy-group profile with 90% bootstrap CIs
    (utils.motivation_utils.vei_percentile_group_profile -- imported, never reimplemented);
  * 30 equal-count sale-price bins with the median valuation ratio per bin.

Every prediction vector is either an actual fitted configuration's cached predictions
or an actual evaluation of the theorem-matched transformation. Nothing is interpolated.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p0_common as c
import p0_4_centered_spread as cs

sys.path.insert(0, str(c.REPO))
from utils.motivation_utils import vei_percentile_group_profile   # noqa: E402

T = c.TABLES
N_BINS = 30
EVALS = ["heldout", "forward_2025"]
FAM_MODEL = {"Direct": "LGBCovPenalty", "Surrogate": "LGBSmoothPenalty"}


def _fitted_preds(family, rho, evaluation):
    m = pd.read_csv(c.CONFIGS / "frozen_config_map.csv")
    s = m[(m.model_name == FAM_MODEL[family]) & (m.stage == evaluation)
          & np.isclose(m.rho.astype(float), float(rho), rtol=0, atol=1e-12)]
    if len(s) != 1:
        raise c.ProtocolViolation(
            f"{family} rho={rho} {evaluation}: {len(s)} cached prediction files (need 1)")
    r = s.iloc[0]
    if not bool(r.pred_exists):
        raise c.ProtocolViolation(f"cached prediction file missing: {r.pred_file}")
    d = pd.read_parquet(c.REPO / r.pred_file)
    return d, str(r.config_id)


def _posthoc_preds(ref, b, evaluation):
    blk = cs.BLOCK_FOR[evaluation]
    d, yb, _ = cs.load_eval(ref, blk)
    out = d.copy()
    out["y_pred_log"] = cs.centered_map(d.y_pred_log.to_numpy(), yb, b)
    # the Stage-1.5 paired-evaluation files carry log columns only; retransform by
    # direct exponentiation, exactly as run_temporal_cv does (no smearing) so the
    # price-level ratios sit on the same footing as the cached fitted-config files
    out["y_pred"] = np.exp(out.y_pred_log.to_numpy())
    if "y_true" not in out.columns:
        out["y_true"] = np.exp(d.y_true_log.to_numpy())
    return out, f"{ref}-posthoc-b={b:.12f}"


def _price_bin_profile(assessed, price, n_bins=N_BINS):
    ok = np.isfinite(assessed) & np.isfinite(price) & (assessed > 0) & (price > 0)
    a, p = assessed[ok], price[ok]
    ratio = a / p
    order = np.argsort(p, kind="stable")
    edges = np.array_split(order, n_bins)
    rows = []
    for i, idx in enumerate(edges, start=1):
        if not len(idx):
            continue
        rows.append({"bin": i, "n": int(len(idx)),
                     "price_min": float(p[idx].min()), "price_max": float(p[idx].max()),
                     "price_median": float(np.median(p[idx])),
                     "median_ratio": float(np.median(ratio[idx])),
                     "mean_ratio": float(np.mean(ratio[idx]))})
    return pd.DataFrame(rows)


def main() -> int:
    F = json.loads((c.CONFIGS / "matched_beta_frozen.json").read_text())
    h = json.loads((c.CONFIGS / "matched_beta_frozen_hash.json").read_text())
    if h["file_sha256"] != c.sha256_file(c.CONFIGS / "matched_beta_frozen.json"):
        raise c.ProtocolViolation("matched_beta_frozen.json changed after being hashed")

    iaao, bins = [], []
    for rec in F["matched_configurations"]:
        if not rec["attained"]:
            continue
        fam, j = rec["family"], rec["j"]
        for ev in EVALS:
            if fam in ("Direct", "Surrogate"):
                d, cid = _fitted_preds(fam, rec["rho"], ev)
                src = "cached predictions of an actually fitted configuration"
            else:
                d, cid = _posthoc_preds("C" if fam == "C-posthoc" else "A", float(rec["b"]), ev)
                src = "actual evaluation of the theorem-matched transformation"
            a = d.y_pred.to_numpy(dtype=float); p = d.y_true.to_numpy(dtype=float)
            base = {"j": j, "target": rec["target"], "family": fam, "role": rec["role"],
                    "rho": rec["rho"], "b": rec["b"], "evaluation": ev,
                    "config_id": cid, "source": src, "n_total": int(len(d))}
            prof = vei_percentile_group_profile(a, p)
            for _, r in prof.iterrows():
                iaao.append({**base, **{k: (float(v) if isinstance(v, (int, float, np.floating))
                                            else v) for k, v in r.items()}})
            for _, r in _price_bin_profile(a, p).iterrows():
                bins.append({**base, **r.to_dict()})
            print(f"[profiles] j={j} {fam:<11} {ev:<13} n={len(d)}", flush=True)

    idf = pd.DataFrame(iaao); bdf = pd.DataFrame(bins)
    c.write_table(idf, T / "matched_beta_ratio_profiles.csv")
    c.write_table(bdf, T / "matched_beta_ratio_profiles_price_bins.csv")
    print(f"[profiles] wrote {len(idf)} IAAO proxy-group rows and {len(bdf)} price-bin rows")
    print(f"[profiles] IAAO group columns: {sorted(idf.columns.tolist())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
