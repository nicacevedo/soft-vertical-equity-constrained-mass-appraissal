#!/usr/bin/env python3
"""Frozen 2025 forward path: one jurisdiction × one family.

Train on the full frozen development sample (sale_date < 2025-01-01).
Evaluate only on calendar 2025. raw_rho = rho_tilde / Var_train(log price),
ddof=0, computed once from the 2016-2024 training block. Never recalibrated
from 2025. Never writes a candidate region.

Also stores CV-fold OOF predictions at predeclared frozen anchors only, for
ratio-profile panels. Those fits still use 2016-2024 data only.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMRegressor

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from soft_constrained_models.boosting_models import LGBCovPenalty, LGBSmoothPenalty  # noqa: E402
from analysis.external_jurisdiction_benchmark_v1.scripts.run_baseline_cv import (  # noqa: E402
    build_folds, enrich, v1_features,
)
from analysis.external_jurisdiction_benchmark_v1.scripts.run_normalized_direct_cv import (  # noqa: E402
    diverged_outside_training_support, screening_metrics_finite, training_support_window,
)
from analysis.external_jurisdiction_benchmark_v1.scripts.forward_common import (  # noqa: E402
    ANALYSIS, CANONICAL_METRIC_MAP, DIRECT_COMMON_INTERVAL, EXPECTED_FORWARD_FREEZE_SHA256,
    FORWARD_LOCK_DATE, OUTPUT, add_derived, canonicalize_metrics, frozen_anchor_points,
    frozen_grid_rho_tilde, preforward_identity, verify_forward_freeze,
)
from analysis.external_jurisdiction_benchmark_v1.scripts.v1_common import (  # noqa: E402
    DIRECT_CANONICAL_KWARGS, LGBM_CONFIG_PATH, SURROGATE_CANONICAL_KWARGS,
    population_variance, write_json,
)

FORWARD_END = pd.Timestamp("2026-01-01")


def _near(a: float, b: float) -> bool:
    return bool(np.isclose(float(a), float(b), rtol=0.0, atol=1e-12))


def _on_grid(rho: float, grid: np.ndarray) -> bool:
    return any(_near(rho, g) for g in grid)


def load_params(key: str, freeze: dict, threads: int) -> tuple[dict, str]:
    cfg_path = ANALYSIS / "baseline" / f"{key}_baseline_config.json"
    cfg = json.loads(cfg_path.read_text())
    selected = cfg["selected_lgbm_config"]
    if selected != freeze["baseline_configs"][key]:
        raise RuntimeError(f"{key}: config {selected} != freeze {freeze['baseline_configs'][key]}")
    grid = yaml.safe_load(LGBM_CONFIG_PATH.read_text())["lgbm_grid"]
    params = dict(grid[selected])
    params["n_jobs"] = threads
    return params, selected


def fit_pred_log(family: str, raw_rho: float, features: pd.DataFrame, y_log_train: pd.Series,
                 n_train: int, params: dict, cats: list[str]) -> np.ndarray:
    X_tr = features.iloc[:n_train]
    X_ev = features.iloc[n_train:]
    if family == "direct":
        model = LGBCovPenalty(
            rho=float(raw_rho), early_stopping_rounds=None, zero_grad_tol=1e-12,
            lgbm_params=dict(params), verbose=False, **DIRECT_CANONICAL_KWARGS,
        )
        model.fit(X_tr, y_log_train)
        return np.asarray(model.predict(X_ev), dtype=float)
    if float(raw_rho) == 0.0:
        model = LGBMRegressor(**dict(params))
        model.fit(X_tr, y_log_train, categorical_feature=cats)
        return np.asarray(model.predict(X_ev), dtype=float)
    model = LGBSmoothPenalty(
        rho=float(raw_rho), early_stopping_rounds=None, lgbm_params=dict(params),
        verbose=False, **SURROGATE_CANONICAL_KWARGS,
    )
    model.fit(X_tr, y_log_train)
    return np.asarray(model.predict(X_ev), dtype=float)


def score_split(family: str, raw_rho: float, rho_tilde: float, features, y_log_train, n_train,
                params, cats, eval_price, train_price, support_lo, support_hi) -> tuple[dict, np.ndarray | None]:
    row = {"fit_status": "OK"}
    try:
        pred_log = fit_pred_log(family, raw_rho, features, y_log_train, n_train, params, cats)
        if not np.all(np.isfinite(pred_log)):
            raise FloatingPointError("non-finite predicted log price")
        if family == "direct":
            divergence = diverged_outside_training_support(pred_log, support_lo, support_hi)
            if divergence is not None:
                row["fit_status"] = "DIVERGED_OUTSIDE_TRAINING_SUPPORT"
                row["fit_error"] = (
                    "pred_log in [{pred_log_min:.4g},{pred_log_max:.4g}] left training-support "
                    "window [{support_lo:.4g},{support_hi:.4g}]".format(**divergence)
                )
                row.update(divergence)
                return row, None
        pred_price = np.exp(pred_log)
        if not np.all(np.isfinite(pred_price)):
            raise FloatingPointError("non-finite predicted price (exp overflow)")
        metrics = enrich(eval_price, pred_price, train_price)
        if family == "direct":
            bad = screening_metrics_finite(metrics)
            if bad:
                raise FloatingPointError("non-finite screening metric(s): " + ",".join(bad))
        row.update(metrics)
        return row, pred_price
    except (FloatingPointError, ValueError) as exc:
        row["fit_status"] = "NUMERICALLY_UNSTABLE_RHO"
        row["fit_error"] = str(exc)
        return row, None
    except Exception as exc:
        row["fit_status"] = "FIT_FAILURE"
        row["fit_error"] = f"{type(exc).__name__}: {exc}"
        return row, None


def pred_frame(eval_df: pd.DataFrame, pred_price: np.ndarray, rho_tilde: float, roles: str,
               on_grid: bool, split: str, fold: int | None) -> pd.DataFrame:
    cols = [c for c in ("TRANSACTIONID", "ATTOMID", "sale_date", "sale_price") if c in eval_df.columns]
    out = eval_df.loc[:, cols].copy()
    out["pred_price"] = pred_price
    out["rho_tilde"] = float(rho_tilde)
    out["anchor_roles"] = roles
    out["on_frozen_grid"] = bool(on_grid)
    out["split"] = split
    out["fold"] = fold
    return out


def metric_row(key, family, rho_tilde, raw_rho, vy, n_train, n_eval, split, fold, roles,
               on_grid, raw_metrics, baseline_canon) -> dict:
    canon = canonicalize_metrics(raw_metrics)
    derived = add_derived(canon, baseline_canon)
    rec = {
        "county_key": key, "family": family, "rho_tilde": float(rho_tilde),
        "raw_rho": float(raw_rho) if np.isfinite(raw_rho) else float("nan"),
        "Var_train_y": float(vy), "n_train": int(n_train), "n_eval": int(n_eval),
        "evaluation_layer": split, "fold": fold, "anchor_roles": roles,
        "on_frozen_grid": bool(on_grid),
        "fit_status": raw_metrics.get("fit_status", "OK"),
        "fit_error": raw_metrics.get("fit_error"),
    }
    rec.update(derived)
    return rec


def unique_forward_rhos(family: str, key: str) -> list[dict]:
    grid = frozen_grid_rho_tilde(family)
    items = []
    if family == "surrogate":
        items.append({"rho_tilde": 0.0, "roles": ["native_lgbm_baseline"], "on_frozen_grid": False})
    for r in grid:
        items.append({"rho_tilde": float(r), "roles": ["grid"], "on_frozen_grid": True})
    for p in frozen_anchor_points(key, family):
        rho = float(p["rho_tilde"])
        matched = next((it for it in items if _near(it["rho_tilde"], rho)), None)
        if matched:
            if p["role"] not in matched["roles"]:
                matched["roles"].append(p["role"])
        else:
            items.append({"rho_tilde": rho, "roles": [p["role"]], "on_frozen_grid": False})
    return items


def run_forward(key: str, family: str, threads: int, skip_oof: bool) -> dict:
    freeze = verify_forward_freeze()
    if freeze["jurisdiction_roles"].get(key) != "PRIMARY_FULL_7_FOLD":
        raise RuntimeError(f"{key} is not PRIMARY_FULL_7_FOLD")
    params, selected = load_params(key, freeze, threads)

    full_path = OUTPUT / "modeling_tables" / key / "history_market_core_full.parquet"
    if not full_path.exists():
        raise RuntimeError(f"{key}: full modeling table missing; run build_modeling_tables --mode forward")
    data = pd.read_parquet(full_path).sort_values("sale_date").reset_index(drop=True)
    data["sale_date"] = pd.to_datetime(data["sale_date"])
    identity = preforward_identity(key, data)

    train = data.loc[data["sale_date"] < FORWARD_LOCK_DATE].reset_index(drop=True)
    ev = data.loc[(data["sale_date"] >= FORWARD_LOCK_DATE) & (data["sale_date"] < FORWARD_END)].reset_index(drop=True)
    if len(train) < 500 or len(ev) < 50:
        raise RuntimeError(f"{key}: n_train={len(train)} n_eval={len(ev)} too small")
    if train["sale_date"].max() >= FORWARD_LOCK_DATE:
        raise RuntimeError(f"{key}: 2025 leakage into forward training")
    if ev["sale_date"].min() < FORWARD_LOCK_DATE or ev["sale_date"].max() >= FORWARD_END:
        raise RuntimeError(f"{key}: forward eval is not calendar 2025")
    if (ev["sale_date"].dt.year != 2025).any():
        raise RuntimeError(f"{key}: forward eval contains non-2025 rows")

    combined = pd.concat([train, ev], ignore_index=True)
    features, cats = v1_features(combined, len(train), True)
    y_log_train = np.log(train["sale_price"].astype(float))
    vy = population_variance(y_log_train)
    if not np.isfinite(vy) or vy <= 0:
        raise RuntimeError(f"{key}: invalid Var_train(log price)={vy}")
    train_price = train["sale_price"].to_numpy()
    eval_price = ev["sale_price"].to_numpy()
    support_lo, support_hi = training_support_window(np.asarray(y_log_train))
    rho_items = unique_forward_rhos(family, key)

    pred_parts = []
    metric_rows = []
    baseline_canon = None
    for i, item in enumerate(rho_items):
        rho_tilde = float(item["rho_tilde"])
        raw_rho = 0.0 if rho_tilde == 0.0 else float(rho_tilde) / vy
        roles = ",".join(item["roles"])
        raw, pred_price = score_split(
            family, raw_rho, rho_tilde, features, y_log_train, len(train),
            params, cats, eval_price, train_price, support_lo, support_hi,
        )
        if _near(rho_tilde, 0.0) and raw.get("fit_status") == "OK":
            baseline_canon = canonicalize_metrics(raw)
        rec = metric_row(key, family, rho_tilde, raw_rho, vy, len(train), len(ev),
                         "FORWARD_2025", None, roles, item["on_frozen_grid"], raw, baseline_canon)
        rec["raw_rho_reproduced"] = bool(_near(raw_rho * vy if rho_tilde else 0.0, rho_tilde) or rho_tilde == 0.0)
        metric_rows.append(rec)
        if pred_price is not None:
            pred_parts.append(pred_frame(ev, pred_price, rho_tilde, roles, item["on_frozen_grid"],
                                         "FORWARD_2025", None))
        print(f"{key} {family} forward {i+1}/{len(rho_items)} rho_tilde={rho_tilde:.6g} "
              f"status={raw.get('fit_status')}", flush=True)

    # Second pass: attach Delta_* now that rho=0 baseline is known even if it
    # was not first (it is first by construction).
    if baseline_canon is None:
        raise RuntimeError(f"{key}/{family}: rho=0 baseline fit did not succeed")
    for rec in metric_rows:
        if rec.get("Delta_NMSE") is None or not np.isfinite(rec.get("Delta_NMSE", np.nan)):
            rec.update(add_derived({k: rec[k] for k in rec if k in (
                "R2_price", "R2_log", "NMSE", "RMSE_log", "MAE", "MAPE",
                "median_ratio", "mean_ratio", "weighted_mean_ratio", "COD", "COV",
                "PRD", "PRB", "MKI", "VEI", "beta_log", "Delta_NL", "dCor", "n_eval",
            )}, baseline_canon))

    oof_parts = []
    oof_metric_rows = []
    if not skip_oof:
        pre = data.loc[data["sale_date"] < FORWARD_LOCK_DATE].reset_index(drop=True)
        anchor_items = [it for it in rho_items if any(
            r in it["roles"] for r in (
                "baseline_rho0", "native_lgbm_baseline", "activity", "guardrail",
                "A_beta_0.25", "A_beta_0.5", "A_beta_0.75", "A_beta_0.9",
                "direct_common_lo", "direct_common_hi",
            )
        ) or _near(it["rho_tilde"], 0.0)]
        for fold_idx, (_ds, train_end, val_end) in enumerate(build_folds(pre["sale_date"]), start=1):
            tr_mask = pre["sale_date"] < train_end
            va_mask = (pre["sale_date"] >= train_end) & (pre["sale_date"] < val_end)
            tr = pre.loc[tr_mask].reset_index(drop=True)
            va = pre.loc[va_mask].reset_index(drop=True)
            if len(tr) < 500 or len(va) < 50:
                continue
            if va["sale_date"].max() >= FORWARD_LOCK_DATE:
                raise RuntimeError(f"{key}: OOF fold {fold_idx} leaked 2025")
            comb = pd.concat([tr, va], ignore_index=True)
            feats, cats_f = v1_features(comb, len(tr), True)
            ytr = np.log(tr["sale_price"].astype(float))
            vy_f = population_variance(ytr)
            slo, shi = training_support_window(np.asarray(ytr))
            fold_base = None
            for item in anchor_items:
                rho_tilde = float(item["rho_tilde"])
                raw_rho = 0.0 if rho_tilde == 0.0 else float(rho_tilde) / vy_f
                roles = ",".join(item["roles"])
                raw, pred_price = score_split(
                    family, raw_rho, rho_tilde, feats, ytr, len(tr), params, cats_f,
                    va["sale_price"].to_numpy(), tr["sale_price"].to_numpy(), slo, shi,
                )
                if _near(rho_tilde, 0.0) and raw.get("fit_status") == "OK":
                    fold_base = canonicalize_metrics(raw)
                rec = metric_row(key, family, rho_tilde, raw_rho, vy_f, len(tr), len(va),
                                 "CV_OOF", fold_idx, roles, item["on_frozen_grid"], raw, fold_base)
                rec["validation_year"] = int(train_end.year)
                oof_metric_rows.append(rec)
                if pred_price is not None:
                    oof_parts.append(pred_frame(va, pred_price, rho_tilde, roles, item["on_frozen_grid"],
                                                "CV_OOF", fold_idx))
            print(f"{key} {family} OOF fold {fold_idx} done", flush=True)

    out_pred = OUTPUT / "forward_2025" / "predictions"
    out_boot = OUTPUT / "forward_2025" / "bootstrap"
    out_pred.mkdir(parents=True, exist_ok=True)
    out_boot.mkdir(parents=True, exist_ok=True)
    metrics_dir = ANALYSIS / "forward_2025" / "metrics" / "partial"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    if pred_parts:
        pd.concat(pred_parts, ignore_index=True).to_parquet(
            out_pred / f"{key}_{family}_forward_preds.parquet", index=False
        )
    if oof_parts:
        pd.concat(oof_parts, ignore_index=True).to_parquet(
            out_pred / f"{key}_{family}_oof_anchor_preds.parquet", index=False
        )
    mdf = pd.DataFrame(metric_rows)
    mdf.to_csv(metrics_dir / f"{key}_{family}_forward_path.csv", index=False)
    if oof_metric_rows:
        pd.DataFrame(oof_metric_rows).to_csv(metrics_dir / f"{key}_{family}_oof_anchor_metrics.csv", index=False)

    n_grid = int(sum(1 for it in rho_items if it["on_frozen_grid"]))
    expected_grid = 34 if family == "direct" else 33
    meta = {
        "county_key": key, "family": family,
        "selected_lgbm_config": selected,
        "forward_freeze_sha256": EXPECTED_FORWARD_FREEZE_SHA256,
        "Var_train_y": float(vy),
        "n_train": int(len(train)), "n_eval": int(len(ev)),
        "n_forward_rho_attempts": len(rho_items),
        "n_frozen_grid_points": n_grid,
        "expected_frozen_grid_points": expected_grid,
        "direct_common_interval": list(DIRECT_COMMON_INTERVAL),
        "identity": identity,
        "no_2025_in_training": True,
        "eval_year_min": int(ev["sale_date"].dt.year.min()),
        "eval_year_max": int(ev["sale_date"].dt.year.max()),
        "wrote_candidate_region": False,
        "surrogate_branch_recalibrated_from_2025": False,
        "written_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    write_json(metrics_dir / f"{key}_{family}_forward_meta.json", meta)
    print(json.dumps({k: meta[k] for k in (
        "county_key", "family", "n_train", "n_eval", "n_frozen_grid_points", "Var_train_y"
    )}), flush=True)
    if n_grid != expected_grid:
        raise RuntimeError(f"{key}/{family}: grid inventory {n_grid} != {expected_grid}")
    return meta


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--county-key", required=True)
    parser.add_argument("--family", required=True, choices=("direct", "surrogate"))
    parser.add_argument("--lgbm-threads", type=int, default=8)
    parser.add_argument("--skip-oof-anchors", action="store_true")
    args = parser.parse_args()
    run_forward(args.county_key, args.family, args.lgbm_threads, args.skip_oof_anchors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
