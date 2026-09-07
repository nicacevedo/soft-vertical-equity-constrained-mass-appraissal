#!/usr/bin/env python3
"""P1 Task 3: VEI inference per the IAAO May-2026 Exposure Draft, Appendix E.

Implemented literally from the authoritative source (see
provenance/ed2_source_manifest.json and configs/ed2_vei_procedure.json).

  Step 4  90% median CI per percentile group by App. D.2 RANK-BASED order statistics
          -- NOT the bootstrap.  The deterministic bootstrap is retained only in
          separately labelled *_sensitivity columns.
  Step 5  VEI = 100 * (MEDIAN Last PG - MEDIAN First PG) / Sample MEDIAN.
          |VEI| <= 10 stops here; the gate is evaluated independently for every
          display cell from its own computed value.
  Step 6  first/last PG CI overlap -> stop; no overlap -> Step 7.
  Step 7  VEI Significance = 100 * (Lower CI of PG with highest median
          - Upper CI of PG with lowest median) / Sample MEDIAN, compared to 10.

All verdict strings are ED2's own wording, emitted from the decision mapping in
configs/ed2_vei_procedure.json.

STATUS: May-2026 Exposure Draft / proposed guidance; not adopted IAAO guidance.
        Not a compliance determination.  The adopted reference is the 2013 Standard.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p1_common as c                                                   # noqa: E402

CI_LEVEL = 0.90
Z = {0.90: 1.645, 0.95: 1.96}
BAND = 10.0
BOOTSTRAP_SCOPE = ("heldout", "forward_2025")   # sensitivity only; bounded for cost


# ------------------------------------------------- ED2 App. D.2 rank-based CI
def ed2_median_ci(x: np.ndarray, ci_level: float = CI_LEVEL) -> dict:
    """IAAO ED2 Appendix D.2 (page 67), verbatim rule, for n > 30.

    j = z*sqrt(n)/2          (n odd)
    j = z*sqrt(n)/2 + 0.5    (n even)
    round j UP to the next integer if it is not already one, then count j up and
    down the array from the median.

    Interpretation ED2-I-1 (recorded in configs/ed2_vei_procedure.json): for even n
    the median is not an array element, so the count is taken outward from the two
    central order statistics -- lower from rank n/2+1, upper from rank n/2.
    """
    v = np.sort(np.asarray(x, dtype=float))
    n = v.size
    if n <= 30:
        raise c.ProtocolViolation(f"D.2 applies to n > 30; got n={n} (D.4 covers n<=30)")
    z = Z[ci_level]
    if n % 2 == 1:
        j = math.ceil(z * math.sqrt(n) / 2)
        m = (n + 1) // 2
        lo_rank, hi_rank = m - j, m + j
    else:
        j = math.ceil(z * math.sqrt(n) / 2 + 0.5)
        lo_rank, hi_rank = n // 2 + 1 - j, n // 2 + j
    clamped = bool(lo_rank < 1 or hi_rank > n)
    lo_rank, hi_rank = max(1, lo_rank), min(n, hi_rank)
    return {"lo": float(v[lo_rank - 1]), "hi": float(v[hi_rank - 1]), "j": int(j),
            "lo_rank": int(lo_rank), "hi_rank": int(hi_rank), "clamped": clamped,
            "n": int(n), "parity": "odd" if n % 2 else "even"}


# ------------------------------------------------------- ED2 App. E grouping
def vei_groups(assessed: np.ndarray, sale_price: np.ndarray):
    """Reproduce utils.motivation_utils.vei's grouping EXACTLY (it produced the
    frozen VEI values): equal-weight proxy, mergesort, numpy.array_split."""
    a = np.asarray(assessed, float); s = np.asarray(sale_price, float)
    m = np.isfinite(a) & np.isfinite(s) & (a > 0) & (s > 0)
    a, s = a[m], s[m]
    n = a.size
    if n < 20:
        return None
    k = 2 if n <= 50 else (4 if n <= 500 else 10)
    ratio = a / s
    ratio = ratio[np.isfinite(ratio)]
    med = float(np.median(ratio))
    if not np.isfinite(med) or med == 0:
        return None
    # ED2 Step 2 PRINTED formula omits the 0.50 on AV/Median; its prose says equal
    # weight. The equal-weight (2013 Standard) form is implemented -- see
    # configs/ed2_vei_procedure.json -> appendix_E_vei.step_2.
    proxy = 0.5 * s + 0.5 * (a / med)
    order = np.argsort(proxy, kind="mergesort")
    chunks = np.array_split(np.arange(n), k)
    groups = [(a[order[ch]] / s[order[ch]]) for ch in chunks]
    return {"k": k, "n": n, "sample_median": med, "groups": groups}


def _bootstrap_group_ci(vals: np.ndarray, n_boot=1000, ci=0.90, seed=2025):
    """The existing deterministic percentile bootstrap -- SENSITIVITY ONLY."""
    rng = np.random.default_rng(int(seed))
    v = np.asarray(vals, float)
    meds = np.empty(n_boot)
    for b in range(n_boot):
        meds[b] = np.median(v[rng.integers(0, v.size, v.size)])
    a = (1 - ci) / 2
    return float(np.quantile(meds, a)), float(np.quantile(meds, 1 - a))


def evaluate_ed2(assessed, sale_price, with_bootstrap=False) -> dict | None:
    P = c.ed2_procedure()
    DM = P["decision_mapping"]
    g = vei_groups(assessed, sale_price)
    if g is None:
        return None
    med = g["sample_median"]
    first, last = g["groups"][0], g["groups"][-1]
    if min(first.size, last.size) < 10:
        return None
    m_first, m_last = float(np.median(first)), float(np.median(last))
    vei = 100.0 * (m_last - m_first) / med

    ci_first = ed2_median_ci(first)
    ci_last = ed2_median_ci(last)

    out = {"n": g["n"], "n_groups": g["k"], "sample_median_ratio": med,
           "VEI_step5": vei,
           "first_pg_n": int(first.size), "last_pg_n": int(last.size),
           "first_pg_median": m_first, "last_pg_median": m_last,
           "first_pg_ci_lo": ci_first["lo"], "first_pg_ci_hi": ci_first["hi"],
           "last_pg_ci_lo": ci_last["lo"], "last_pg_ci_hi": ci_last["hi"],
           "first_pg_ci_j": ci_first["j"], "last_pg_ci_j": ci_last["j"],
           "first_pg_ci_ranks": f"{ci_first['lo_rank']}|{ci_first['hi_rank']}",
           "last_pg_ci_ranks": f"{ci_last['lo_rank']}|{ci_last['hi_rank']}",
           "first_pg_n_parity": ci_first["parity"], "last_pg_n_parity": ci_last["parity"],
           "ci_rank_clamped": bool(ci_first["clamped"] or ci_last["clamped"]),
           "ci_method": "ED2 App. D.2 rank-based order statistic", "ci_level": CI_LEVEL,
           "ed2_band_pct": BAND}

    # ---- Step 5 gate, evaluated from THIS cell's own computed VEI
    if abs(vei) <= BAND:
        st5 = DM["step5_gate"]["within_band"]
        out.update({"step5_gate": st5["code"], "step5_verdict": st5["ed2_verdict"],
                    "step6_result": DM["step6_result"]["not_run"]["code"],
                    "step6_verdict": DM["step6_result"]["not_run"]["ed2_verdict"],
                    "VEI_significance_step7": None,
                    "step7_outcome": DM["step7_outcome"]["not_run"]["code"],
                    "step7_verdict": DM["step7_outcome"]["not_run"]["ed2_verdict"],
                    "ed2_verdict": st5["ed2_verdict"], "ed2_boundary_exact": False})
        return out
    st5 = DM["step5_gate"]["outside_band"]
    out.update({"step5_gate": st5["code"], "step5_verdict": st5["ed2_verdict"]})

    # ---- Step 6: do the first/last PG 90% median CIs overlap?
    overlap = not (out["first_pg_ci_hi"] < out["last_pg_ci_lo"]
                   or out["last_pg_ci_hi"] < out["first_pg_ci_lo"])
    if overlap:
        s6 = DM["step6_result"]["overlap"]
        out.update({"step6_result": s6["code"], "step6_verdict": s6["ed2_verdict"],
                    "VEI_significance_step7": None,
                    "step7_outcome": DM["step7_outcome"]["not_run"]["code"],
                    "step7_verdict": DM["step7_outcome"]["not_run"]["ed2_verdict"],
                    "ed2_verdict": s6["ed2_verdict"], "ed2_boundary_exact": False})
        return out
    s6 = DM["step6_result"]["no_overlap"]
    out.update({"step6_result": s6["code"], "step6_verdict": s6["ed2_verdict"]})

    # ---- Step 7 (interpretation ED2-I-2: the two groups carried from Step 6)
    if m_first >= m_last:
        lo_of_highest, hi_of_lowest = out["first_pg_ci_lo"], out["last_pg_ci_hi"]
    else:
        lo_of_highest, hi_of_lowest = out["last_pg_ci_lo"], out["first_pg_ci_hi"]
    sig = 100.0 * (lo_of_highest - hi_of_lowest) / med
    if sig < 0:
        raise c.ProtocolViolation(
            f"VEI Significance negative ({sig}) despite non-overlapping CIs at Step 6")
    key = "gt_10" if sig > BAND else "le_10"       # ED2-I-3: exactly 10 -> fail to reject
    s7 = DM["step7_outcome"][key]
    out.update({"VEI_significance_step7": sig, "step7_outcome": s7["code"],
                "step7_verdict": s7["ed2_verdict"], "ed2_verdict": s7["ed2_verdict"],
                "ed2_boundary_exact": bool(sig == BAND)})

    if with_bootstrap:
        bl, bh = _bootstrap_group_ci(first)
        cl, ch = _bootstrap_group_ci(last)
        out.update({"first_pg_bootstrap_ci_lo_sensitivity": bl,
                    "first_pg_bootstrap_ci_hi_sensitivity": bh,
                    "last_pg_bootstrap_ci_lo_sensitivity": cl,
                    "last_pg_bootstrap_ci_hi_sensitivity": ch})
    return out


ROLE_FOR_EVAL = {**{f"fold_{k}": "descriptive" for k in range(1, 8)},
                 "pooled_oof": "not_applicable", "heldout": "standards_facing",
                 "forward_2025": "standards_facing"}

# The ED2 App. D.2 confidence interval is a RANK-BASED ORDER STATISTIC on a sample of
# distinct observations.  It has no row-balanced / weighted analogue, and the pooled-OOF
# sample counts 20,988 unique rows twice (fold_6 & fold_7 overlap).  Rather than invent a
# weighted variant the draft does not define, pooled_oof is carried as an explicit
# not-applicable row with the reason recorded.
POOLED_OOF_REASON = (
    "ED2 App. D.2 is a rank-based order-statistic CI on distinct observations and has no "
    "row-balanced analogue; the pooled-OOF sample contains 20,988 unique rows twice, so the "
    "ED2 procedure is not applied there. No weighted variant was invented.")


def _frozen_vei(v4: pd.DataFrame, entry: dict, ev: str):
    if entry["kind"] != "fitted" or ev == "pooled_oof":
        return None
    fam = {"Direct": "Direct", "Surrogate": "Surrogate", "A-native": "LightGBM"}.get(entry["family"])
    if fam is None:
        return None
    s = v4[v4.family == fam] if fam == "LightGBM" else v4[
        (v4.family == fam) & (v4.rho.astype(float) - float(entry["rho"])).abs().le(1e-12)]
    if len(s) != 1:
        return None
    col = f"VEI__{ev}"
    if col not in v4.columns:
        return None
    v = s.iloc[0][col]
    return None if pd.isna(v) else float(v)


def main() -> int:
    from utils.motivation_utils import vei as canonical_vei
    M = c.ed2_source_manifest()            # refuses to run without a verified ED2 source
    P = c.ed2_procedure()
    print(f"[ed2] source verified sha256={M['source']['sha256'][:16]}... "
          f"title='{M['document_identity']['internal_title_as_printed']}'")
    c.assert_d3_multiplicity_identities()
    D = c.display_set()
    v4 = pd.read_csv(c.V12 / "analysis" / "data_id=d4929d43ec19badf"
                     / "split_id=3d464d4a611b131b" / "penalty_path_analysis"
                     / "transition_regions_paper_assets_v4_delta_nl_bends" / "tables"
                     / "combined_path_table_v4_analysis_view.csv")

    by_real = {}
    for e in D["entries"]:
        by_real.setdefault(e["realization_key"], []).append(e)

    STATUS = P["guidance_status"]
    rows, max_recon = [], 0.0
    for rk, group in by_real.items():
        for ev in c.ALL_EVALS:
            base_common = {"realization_key": rk, "evaluation": ev,
                           "evaluation_role": ROLE_FOR_EVAL[ev],
                           "guidance_status": STATUS,
                           "adopted_reference": P["adopted_reference"],
                           "ed2_source_sha256": M["source"]["sha256"]}
            if rk == "NOT_ATTAINED" or ev == "pooled_oof":
                res = None
                note = ("NOT_ATTAINED within the frozen design" if rk == "NOT_ATTAINED"
                        else POOLED_OOF_REASON)
            else:
                head = group[0]
                d = c.load_observations(head, ev)
                assessed = np.exp(d.y_pred_log.to_numpy())
                sale_price = np.exp(d.y_true_log.to_numpy())
                res = evaluate_ed2(assessed, sale_price,
                                   with_bootstrap=(ev in BOOTSTRAP_SCOPE))
                canon = float(canonical_vei(assessed, sale_price, na_rm=True))
                delta = abs(res["VEI_step5"] - canon)
                max_recon = max(max_recon, delta)
                if delta > 1e-9:
                    raise c.ProtocolViolation(
                        f"VEI Step 5 does not reproduce canonical vei() for {rk}/{ev}: {delta:.3e}")
                res["canonical_vei_fn"] = canon
                res["step5_minus_canonical"] = res["VEI_step5"] - canon
                fz = _frozen_vei(v4, head, ev)
                res["frozen_VEI_value"] = fz
                res["frozen_minus_recomputed"] = None if fz is None else fz - canon
                note = head["artifacts"].get(ev)
            for e in group:
                row = {**base_common,
                       "display_kind": e["display_kind"], "reference_cell": e["reference_cell"],
                       "family": e["family"], "role_label": e["role"], "j": e["j"],
                       "ext_target": e["ext_target"], "target": e["target"],
                       "rho": e["rho"], "b": e["b"], "attained": bool(e["attained"]),
                       "match_mode": e["match_mode"],
                       "max_achieved_dev_correction": e["max_achieved_dev_correction"],
                       "config_id": e.get("config_id"), "source": note}
                row.update(res if res else {})
                rows.append(row)
        print(f"  done {rk}  ({len(rows)} rows)", flush=True)

    df = pd.DataFrame(rows)
    c.write_table(df, c.TABLES / "vei_significance.csv")

    ap = df[df.evaluation_role.isin(["standards_facing", "descriptive"])]
    ap = ap[ap.VEI_step5.notna()]
    fz = df.frozen_minus_recomputed.dropna().abs() if "frozen_minus_recomputed" in df else pd.Series([], dtype=float)
    summary = {
        "guidance_status": STATUS,
        "adopted_reference": P["adopted_reference"],
        "not_a_compliance_determination": True,
        "ed2_source": {"sha256": M["source"]["sha256"], "bytes": M["source"]["bytes"],
                       "url": M["source"]["pdf_url"],
                       "title_as_printed": M["document_identity"]["internal_title_as_printed"],
                       "appendix_D2_page": M["page_references"]["appendix_D_2_median_ci"],
                       "appendix_E_pages": "78-81"},
        "ci_method": "ED2 App. D.2 rank-based order statistic (NOT the bootstrap)",
        "ci_level": CI_LEVEL,
        "bootstrap_role": "separately labelled SENSITIVITY columns only",
        "bootstrap_scope": list(BOOTSTRAP_SCOPE),
        "pooled_oof": POOLED_OOF_REASON,
        "step5_gate_note": ("evaluated independently for every display cell from its own computed "
                            "VEI; no cell's outcome was anticipated from any existing reported value"),
        "rows": int(len(df)),
        "rows_evaluated": int(ap.shape[0]),
        "rows_not_attained": int((~df.attained.astype(bool)).sum()),
        "max_abs_step5_minus_canonical_vei": float(max_recon),
        "frozen_reconciliation": {
            "n_compared": int(fz.size),
            "max_abs_frozen_minus_recomputed": float(fz.max()) if fz.size else None},
        "step5_gate_counts": {str(k): int(v) for k, v in
                              df.step5_gate.value_counts(dropna=False).items()},
        "step6_counts": {str(k): int(v) for k, v in
                         df.step6_result.value_counts(dropna=False).items()},
        "step7_counts": {str(k): int(v) for k, v in
                         df.step7_outcome.value_counts(dropna=False).items()},
        "ci_rank_clamped_any": bool(df.get("ci_rank_clamped", pd.Series([False])).fillna(False).any()),
        "provenance": c.preflight_block(),
    }
    c.write_json(c.TABLES / "vei_significance_summary.json", summary)
    print("\nrows:", len(df), "| evaluated:", summary["rows_evaluated"])
    print("max |Step5 - canonical vei()|:", f"{max_recon:.3e}")
    print("frozen reconciliation        :", summary["frozen_reconciliation"])
    print("step5 gate :", summary["step5_gate_counts"])
    print("step6      :", summary["step6_counts"])
    print("step7      :", summary["step7_counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
