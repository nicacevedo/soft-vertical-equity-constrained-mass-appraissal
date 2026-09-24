#!/usr/bin/env python3
"""Shared helpers for the ordinary-linear baseline reproduction (LB stage).

Scope and binding rules
-----------------------
* READ-ONLY outside this stage.  Nothing under ``paper/``, ``utils/``,
  ``soft_constrained_models/``, ``scripts/``, ``analysis/p0_major_revision_validation/``,
  ``analysis/p1_inferential_reporting/``, ``analysis/final_manuscript_evidence/``,
  ``output/paper_v6_preselection*`` or ``output/paper_v12_*`` is ever written.
  Writes are confined to ``analysis/linear_baseline_reproduction_v1/`` and
  ``output/linear_baseline_reproduction_v1/`` and are refused elsewhere by assertion.
* Canonical machinery is IMPORTED, never re-implemented: the loader/split code
  (``run_temporal_cv._load_and_split_data`` via ``p0_common.load_canonical_splits``),
  the rolling-origin protocol builder, the IAAO/paper metric functions in
  ``utils.motivation_utils``, and the frozen ``utils.delta_nl`` estimator.
* This stage fits NOTHING by default.  Every number is derived from cached
  row-level prediction artifacts resolved through the frozen P0 config maps.
* Restricted data rule: no sale-level record leaves this stage.  Row-level
  predictions are identified by content hash only.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

LB_DIR = Path(__file__).resolve().parents[1]          # analysis/linear_baseline_reproduction_v1
REPO = LB_DIR.parents[1]                              # repository root (this worktree)
P0_DIR = REPO / "analysis" / "p0_major_revision_validation"
P1_DIR = REPO / "analysis" / "p1_inferential_reporting"

sys.path.insert(0, str(P0_DIR / "code"))
sys.path.insert(0, str(REPO))

import p0_common as p0                                                  # noqa: E402
from p0_common import (                                                 # noqa: E402,F401
    ProtocolViolation,
    sha256_file, sha256_obj, git_state, package_versions, thread_env,
    load_canonical_splits, load_archived_folds, rebuild_folds, frozen_lgbm_config,
    N_DEVELOPMENT, N_HELDOUT, N_2025, N_PRODUCTION,
    TARGET_COL, DATE_COL, DATA_PATH, V6, V12,
    LINEAR_CONFIG_ID, NATIVE_CONFIG_ID, EXPECTED_LGBM_PARAMS_SHA256,
)

TABLES = LB_DIR / "tables"
FIGURES = LB_DIR / "figures"
REPORTS = LB_DIR / "reports"
PROVENANCE = LB_DIR / "provenance"
SNIPPETS = LB_DIR / "snippets"
LOGS = LB_DIR / "logs"
LB_OUTPUT_ROOT = REPO / "output" / "linear_baseline_reproduction_v1"

P0_CONFIGS = P0_DIR / "configs"
P0_TABLES = P0_DIR / "tables"
FROZEN_CONFIG_MAP = P0_CONFIGS / "frozen_config_map.csv"
FROZEN_CV_RUN_MAP = P0_CONFIGS / "frozen_cv_run_map.csv"
ZERO_CONTROL_FULL = P0_TABLES / "zero_control_full.csv"

ZREF = REPO / "output" / "p0_major_revision_validation" / "zero_reference_fits"
BASELINE_REPORTING = (V6 / "baseline_reporting" / "analysis"
                      / "data_id=d4929d43ec19badf" / "split_id=3d464d4a611b131b")

# Recorded v20 extract identity (paper/paper_v20.tex, Section "Application Design").
V20_EXTRACT_SHA256 = "b1fc00b514041af5aa7135d85ed59a028afe40569bba1014c4dcf3ad2f3a7b51"
V20_EXTRACT_BYTES = 215_400_916
V20_N_PREDICTORS = 95
V20_N_CATEGORICAL = 23

# Model display names used in every table this stage writes.
LINEAR_NAME = "Ordinary linear regression (log sale price)"
NATIVE_NAME = "Ordinary LightGBM (standard raw-label native)"

# Guidance status, carried verbatim so no table can silently promote ED2 to adopted.
ADOPTED_GUIDANCE = "IAAO Standard on Ratio Studies (2013): adopted/current guidance."
ED2_GUIDANCE = "May-2026 IAAO Exposure Draft: proposed guidance; NOT adopted."
GUIDANCE_STATUS = {
    "median_ratio": ADOPTED_GUIDANCE, "mean_ratio": ADOPTED_GUIDANCE,
    "weighted_mean_ratio": ADOPTED_GUIDANCE, "COD": ADOPTED_GUIDANCE,
    "COV": "no assessor reference range", "PRD": ADOPTED_GUIDANCE,
    "PRB": ADOPTED_GUIDANCE, "MKI": ED2_GUIDANCE, "VEI": ED2_GUIDANCE,
    "beta_log": "no assessor reference range (paper mechanism diagnostic)",
    "Delta_NL": "no assessor reference range (residual-structure diagnostic)",
    "dCor_e_y": "no assessor reference range (residual-structure diagnostic)",
    "R2_price": "not an assessor-standard measure",
    "MAE_price": "not an assessor-standard measure",
    "MAPE": "not an assessor-standard measure",
    "RMSE_log": "not an assessor-standard measure",
}

# ---------------------------------------------------------------------------
# Write isolation
# ---------------------------------------------------------------------------
_ALLOWED_WRITE_ROOTS = (LB_DIR.resolve(), LB_OUTPUT_ROOT.resolve())
_PROTECTED = (
    REPO / "paper", REPO / "utils", REPO / "soft_constrained_models", REPO / "scripts",
    REPO / "preprocessing", REPO / "run_temporal_cv.py", REPO / "params.yaml",
    P0_DIR, P1_DIR, REPO / "analysis" / "final_manuscript_evidence",
    REPO / "output" / "p0_major_revision_validation",
    REPO / "output" / "paper_v6_preselection", REPO / "output" / "paper_v6_preselection_994",
    REPO / "output" / "paper_v12_lower_rho_extension_994_v2",
    REPO / "data",
)


def _assert_write_allowed(path: Path) -> None:
    rp = path.resolve()
    for prot in _PROTECTED:
        pr = prot.resolve()
        if str(rp) == str(pr) or str(rp).startswith(str(pr) + "/"):
            raise ProtocolViolation(f"Refusing to write to a PROTECTED path: {rp}")
    if not any(str(rp) == str(r) or str(rp).startswith(str(r) + "/")
               for r in _ALLOWED_WRITE_ROOTS):
        raise ProtocolViolation(f"Refusing to write outside the LB stage: {rp}")


def write_json(path: Path, payload: Any) -> Path:
    _assert_write_allowed(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(path)
    return path


def write_table(df: pd.DataFrame, path: Path, *, full_precision: bool = True) -> Path:
    """csv (+ parquet twin) at full float64 repr precision (house convention)."""
    _assert_write_allowed(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False, float_format="%.17g" if full_precision else None)
    tmp.replace(path)
    try:
        df.to_parquet(path.with_suffix(".parquet"), index=False)
    except Exception:
        pass
    return path


def write_text(path: Path, text: str) -> Path:
    _assert_write_allowed(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)
    return path


# ---------------------------------------------------------------------------
# Hashing (identical convention to p0_zero_reference_fits._hash_arr)
# ---------------------------------------------------------------------------
def hash_f64(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()


def hash_i64(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.int64).tobytes()).hexdigest()


def eval_index_hash(row_id, kind: str) -> str:
    """Reproduce whichever index-hash convention the frozen block actually used.

    The two out-of-time blocks were hashed by p0_zero_reference_fits as
    sha256 over int64 bytes; the seven CV blocks carry the archived rolling-origin
    protocol's own ``val_index_hash``, which is the 16-hex
    ``motivation_utils._stable_hash({"idx": [...]})``.  Comparing one convention
    against the other is meaningless, so the convention is selected by block kind
    and recorded next to the value.
    """
    if kind == "out_of_time":
        return hash_i64(row_id)
    from utils.motivation_utils import _stable_hash
    return _stable_hash({"idx": [int(v) for v in np.asarray(row_id).reshape(-1)]})


def archived_fold_val_index_hash(fold_1based: int) -> str:
    arch = load_archived_folds()
    return str(arch["folds"][fold_1based - 1]["val_index_hash"])


# ---------------------------------------------------------------------------
# Frozen artifact resolution
# ---------------------------------------------------------------------------
def frozen_oos_prediction_path(config_id: str, stage: str) -> Path:
    """stage in {'heldout','forward_2025'}; resolved through the frozen P0 config map."""
    m = pd.read_csv(FROZEN_CONFIG_MAP)
    sel = m[(m.stage == stage) & (m.config_id == config_id) & (m.root == "baseline_reporting")]
    if len(sel) != 1:
        raise ProtocolViolation(
            f"frozen_config_map does not resolve a unique baseline_reporting row for "
            f"config_id={config_id} stage={stage} (got {len(sel)})")
    p = REPO / str(sel.iloc[0]["pred_file"])
    if not p.exists():
        raise ProtocolViolation(f"frozen prediction artifact missing: {p}")
    return p


def frozen_fold_prediction_path(config_id: str, fold_1based: int) -> Path:
    m = pd.read_csv(FROZEN_CV_RUN_MAP)
    sel = m[(m.config_id == config_id) & (m.fold_1based == fold_1based)
            & (m.root == "paper_v6_preselection_994")]
    if config_id == NATIVE_CONFIG_ID:
        sel = sel[sel.model_name == "LGBMRegressor"]
    if len(sel) != 1:
        raise ProtocolViolation(
            f"frozen_cv_run_map does not resolve a unique row for config_id={config_id} "
            f"fold={fold_1based} (got {len(sel)})")
    p = REPO / str(sel.iloc[0]["pred_file"])
    if not p.exists():
        raise ProtocolViolation(f"frozen fold prediction artifact missing: {p}")
    return p


def zref_meta(cell: str, block_id: str) -> dict:
    return json.loads((ZREF / f"cell={cell}" / f"block={block_id}" / "fit_meta.json").read_text())


def zref_predictions(cell: str, block_id: str, kind: str) -> pd.DataFrame:
    assert kind in {"train", "eval"}
    return pd.read_parquet(ZREF / f"cell={cell}" / f"block={block_id}" / f"{kind}_predictions.parquet")


# Nine canonical blocks: the paired (fitting block, evaluation) design of the frozen study.
BLOCKS = (
    [{"eval": f"fold_{k}", "block_id": f"fold_{k}_train", "fold_1based": k,
      "kind": "cv_validation"} for k in range(1, 8)]
    + [{"eval": "heldout", "block_id": "development_pool", "fold_1based": None,
        "kind": "out_of_time"},
       {"eval": "forward_2025", "block_id": "production_2016_2024", "fold_1based": None,
        "kind": "out_of_time"}]
)


def load_pair(eval_name: str) -> Dict[str, Any]:
    """Row-level (linear, native LightGBM) predictions plus the fitting block's log target.

    Returns bit-exactness facts alongside the arrays so no caller can silently
    compare two differently-ordered samples.
    """
    blk = next(b for b in BLOCKS if b["eval"] == eval_name)
    if blk["kind"] == "out_of_time":
        stage = "heldout" if eval_name == "heldout" else "forward_2025"
        lin = pd.read_parquet(frozen_oos_prediction_path(LINEAR_CONFIG_ID, stage))
        nat = pd.read_parquet(frozen_oos_prediction_path(NATIVE_CONFIG_ID, stage))
    else:
        lin = pd.read_parquet(frozen_fold_prediction_path(LINEAR_CONFIG_ID, blk["fold_1based"]))
        nat = pd.read_parquet(frozen_fold_prediction_path(NATIVE_CONFIG_ID, blk["fold_1based"]))
    # The frozen benchmark artifact's own row order is canonical here: the archived
    # index and array hashes were taken in that order, so re-sorting would break the
    # hash comparison without changing any (order-invariant) metric.
    ref = zref_predictions("A", blk["block_id"], "eval")
    tr = zref_predictions("A", blk["block_id"], "train")
    meta = zref_meta("A", blk["block_id"])

    if not (len(lin) == len(nat) == len(ref)):
        raise ProtocolViolation(f"{eval_name}: n mismatch {len(lin)}/{len(nat)}/{len(ref)}")
    order = pd.Index(ref.row_id.to_numpy())
    if order.has_duplicates:
        raise ProtocolViolation(f"{eval_name}: frozen benchmark row_id is not unique")
    lin = lin.set_index("row_id").reindex(order)
    nat = nat.set_index("row_id").reindex(order)
    for nm, d in (("linear", lin), ("native", nat)):
        if d.y_pred_log.isna().any():
            raise ProtocolViolation(f"{eval_name}/{nm}: row_id set differs from the frozen benchmark")
        if not np.array_equal(d.y_true_log.to_numpy(), ref.y_true_log.to_numpy()):
            raise ProtocolViolation(f"{eval_name}/{nm}: y_true_log is not bitwise equal to the benchmark")
    lin = lin.reset_index()
    nat = nat.reset_index()
    return {
        "eval": eval_name, "block_id": blk["block_id"], "fold_1based": blk["fold_1based"],
        "kind": blk["kind"], "n": int(len(ref)),
        "row_id": ref.row_id.to_numpy(),
        "y_true_log": ref.y_true_log.to_numpy(),
        "y_train_log": tr.y_true_log.to_numpy(),
        "n_train": int(len(tr)),
        "pred": {LINEAR_NAME: lin.y_pred_log.to_numpy(), NATIVE_NAME: nat.y_pred_log.to_numpy()},
        "pred_path": {LINEAR_NAME: str(lin.attrs.get("path", "")),
                      NATIVE_NAME: str(nat.attrs.get("path", ""))},
        "native_pred_matches_frozen_bitwise": bool(
            np.array_equal(nat.y_pred_log.to_numpy(), ref.y_pred_log.to_numpy())),
        "native_pred_max_abs_delta": float(
            np.max(np.abs(nat.y_pred_log.to_numpy() - ref.y_pred_log.to_numpy()))),
        "frozen_eval_pred_sha256": meta["eval_pred_sha256"],
        "frozen_y_eval_log_sha256": meta["y_eval_log_sha256"],
        "frozen_eval_index_hash": meta["eval_index_hash"],
        "archived_val_index_hash": (archived_fold_val_index_hash(blk["fold_1based"])
                                    if blk["fold_1based"] else None),
        "frozen_train_index_hash": meta["train_index_hash"],
        "frozen_config_hash": meta["lgbm_params_sha256"],
        "sale_date_min": str(lin.sale_date.min()) if "sale_date" in lin else None,
        "sale_date_max": str(lin.sale_date.max()) if "sale_date" in lin else None,
    }


# ---------------------------------------------------------------------------
# Executed metric code -- imported, never re-implemented
# ---------------------------------------------------------------------------
METRIC_KEYS = ["R2_price", "MAE_price", "MAPE", "RMSE_log", "Median ratio", "Mean ratio",
               "W. Mean ratio", "COD", "COV_IAAO", "PRD", "PRB", "MKI", "VEI"]
RENAME = {"Median ratio": "median_ratio", "Mean ratio": "mean_ratio",
          "W. Mean ratio": "weighted_mean_ratio", "COV_IAAO": "COV"}


def metrics_from(y_log, p_log, y_train_log, row_ids) -> dict:
    """Byte-for-byte the p0_zero_control_assemble.metrics_from convention."""
    from utils.motivation_utils import _compute_extended_metrics, paper_mechanism_metrics
    from utils.delta_nl import estimate_delta_nl
    m = _compute_extended_metrics(y_true_log=np.asarray(y_log, float),
                                  y_pred_log=np.asarray(p_log, float),
                                  y_train_log=np.asarray(y_train_log, float),
                                  ratio_mode="diff")
    out = {}
    for k in METRIC_KEYS:
        if k in m:
            try:
                out[RENAME.get(k, k)] = float(m[k])
            except (TypeError, ValueError):
                pass
    mech = paper_mechanism_metrics(np.asarray(y_log, float), np.asarray(p_log, float))
    out["beta_log"] = float(mech["Beta_log"])
    out["Cov_log_residual_log_price"] = float(mech["Cov_log_residual_log_price"])
    out["dCor_e_y"] = float(mech["dCor_e_y"])
    if "RMSE_log" not in out:
        out["RMSE_log"] = float(np.sqrt(np.mean(
            (np.asarray(p_log, float) - np.asarray(y_log, float)) ** 2)))
    dn = estimate_delta_nl(np.asarray(y_log, float), np.asarray(p_log, float), row_ids)
    out["Delta_NL"] = float(dn["Delta_NL"])
    out["Delta_NL_raw"] = float(dn["Delta_NL_raw"])
    return out


def value_proxy_summary(y_log, p_log) -> dict:
    """The equal-weight PRB/VEI value proxy, recomputed per model and sample.

    Both the adopted PRB and the exposure-draft VEI use the same equal-weight
    combination of sale price and median-normalised valuation
        V_proxy_i = 0.5 * SP_i + 0.5 * (AV_i / median_ratio),
    so the proxy is MODEL-SPECIFIC: it embeds that model's own valuations and its
    own sample median ratio.  The two models are therefore not ranked against a
    shared value axis, which is why this summary is reported next to PRB and VEI.
    """
    y = np.exp(np.asarray(y_log, float))
    av = np.exp(np.asarray(p_log, float))
    ratio = av / y
    med = float(np.median(ratio))
    proxy = 0.5 * y + 0.5 * (av / med)
    return {
        "proxy_definition": "V_proxy = 0.5*SP + 0.5*(AV/median_ratio)  [equal weight]",
        "median_ratio_used": med,
        "proxy_median": float(np.median(proxy)),
        "proxy_mean": float(np.mean(proxy)),
        "proxy_q05": float(np.quantile(proxy, 0.05)),
        "proxy_q95": float(np.quantile(proxy, 0.95)),
        "proxy_is_model_specific": True,
        "prb_rhs": "log2(V_proxy)  (utils.motivation_utils.prb)",
        "vei_grouping": "deciles of V_proxy (n >= 501)  (utils.motivation_utils.vei)",
    }


def preflight() -> Dict[str, Any]:
    return {
        "git": git_state(),
        "versions": package_versions(),
        "thread_env": thread_env(),
        "metric_code": {
            "utils/motivation_utils.py": sha256_file(REPO / "utils" / "motivation_utils.py"),
            "utils/delta_nl.py": sha256_file(REPO / "utils" / "delta_nl.py"),
            "preprocessing/recipes_pipelined.py": sha256_file(
                REPO / "preprocessing" / "recipes_pipelined.py"),
            "run_temporal_cv.py": sha256_file(REPO / "run_temporal_cv.py"),
            "params.yaml": sha256_file(REPO / "params.yaml"),
        },
        "stage_code": {
            p.name: sha256_file(p) for p in sorted((LB_DIR / "code").glob("*.py"))
        },
    }
