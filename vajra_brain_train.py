#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vajra_brain_train.py — Advanced Trainer (v8.0 - XGBoost Institutional Paradigm)
====================================================================
LEVEL 30 UPGRADE:
- ALGORITHM SWAP: Replaced GradientBoostingClassifier with XGBClassifier.
- INSTITUTIONAL NOISE CANCELLATION: Native L1 (reg_alpha) and L2 (reg_lambda).
- TOXIC FEATURE BANLIST: Purged absolute prices/derivatives to prevent Time-Travel OOD Collapse.
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import time
from pathlib import Path
from typing import List, Tuple, Optional, Set

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV

log = logging.getLogger("vajra.train.v8")
if not log.handlers:
    log.setLevel(logging.INFO)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
    log.addHandler(h)

def _open_any(p: str):
    path = Path(p)
    if path.suffix == ".gz": return gzip.open(path, "rt", encoding="utf-8")
    else: return open(path, "r", encoding="utf-8")

def _parse_extras(s: Optional[str]) -> List[str]:
    if not s: return []
    return [x.strip() for x in s.split(",") if x.strip()]

def is_feature_column(df: pd.DataFrame, col: str, extra_exclude: Set[str]) -> bool:
    # ========================================================================
    # THE TOXIC FEATURE BANLIST (PREVENTS OOD COLLAPSE & TIME TRAVEL)
    # ========================================================================
    BASE_EXCLUDE_COLS = {
        "timestamp", "symbol", "entry_ts", "exit_ts", "key", "features", "label", "entry_ts_dt",
        "pnl_r", "roi", "outcome", "reason", "exit_reason", "meta_label",
        "entry_price", "exit_price", "price", "stop_loss", "sl", "tp", "take_profit",
        "side", "risk_factor", "rr", "prob", "atr_htf", "atr_ltf", "entry",
        "hour_of_day", "day_of_week",
        
        # --- ABSOLUTE PRICES (Forces AI to memorize the year) ---
        "last_swing_high", "last_swing_low", "ob_bull_price", "ob_bear_price",
        "avwap_bull", "avwap_bear", "poc", "vah", "val", "asian_high", "asian_low",
        "dc_high_20", "dc_low_20", "htf_ema50",
        "htf_swing_high", "htf_swing_low", "mtf_swing_high", "mtf_swing_low",
        "ema20_L", "ema50_L", "ema100_L", "mtf_ema200_arr", "bb_upper", "bb_lower",
        "kc_upper", "kc_lower", "kalman_price", 
        "fib_786_long", "fib_886_long", "fib_786_short", "fib_886_short",
        
        # --- ABSOLUTE DERIVATIVES (Scale massively as BTC price rises) ---
        "kalman_vel", "atr14_L", "atr7_L", "cvd"
    }
    EXCLUDE_SUFFIXES = {"_dt", "_ts"}
    
    if col in BASE_EXCLUDE_COLS or col in extra_exclude: return False
    if any(col.endswith(suf) for suf in EXCLUDE_SUFFIXES): return False
    if col not in df.columns: return False
    return pd.api.types.is_numeric_dtype(df[col])

def _enforce_causality_drop(df: pd.DataFrame, lookahead: int = 5) -> pd.DataFrame:
    if len(df) > lookahead: return df.iloc[:-lookahead]
    return df

def _calculate_recency_and_pnl_weights(df: pd.DataFrame) -> np.ndarray:
    n = len(df)
    pnl_weights = np.clip(np.abs(df['pnl_r'].values), 0, 5)
    weights = pnl_weights
    if weights.sum() > 0:
        weights = weights * (n / weights.sum())
    else:
        weights = np.ones(n)
    return weights

def load_events_df(paths: List[str], min_win_r: float, filter_side: str, extra_exclude: Set[str]) -> Tuple[pd.DataFrame, List[str]]:
    log.info(f"Loading events from {len(paths)} file(s)...")
    rows = []
    for p in paths:
        try:
            with _open_any(p) as f:
                for line in f:
                    if line.strip(): rows.append(json.loads(line))
        except Exception as e: log.error(f"Error {p}: {e}")
    
    if not rows: raise ValueError("No valid events found in files.")
    df = pd.DataFrame(rows)
    
    if filter_side in ("long", "short") and "side" in df.columns:
        sv = 1.0 if filter_side == "long" else 0.0
        if pd.api.types.is_numeric_dtype(df["side"]): df = df[df["side"] == sv].copy()
        else: df = df[df["side"] == filter_side].copy()
        
    df["pnl_r"] = pd.to_numeric(df["pnl_r"], errors='coerce')
    df.dropna(subset=["pnl_r"], inplace=True)
    
    if "meta_label" in df.columns:
        df["label"] = df["meta_label"].astype(np.uint8)
        log.info("🎯 Meta-Labeling detected. Training exclusively as Veto Meta-Model.")
    else:
        log.warning("⚠️ No Meta-Label detected. Falling back to PnL heuristic.")
        df["label"] = (df["pnl_r"] >= min_win_r).astype(np.uint8)
        
    df["entry_ts"] = pd.to_numeric(df["entry_ts"], errors='coerce')
    df.dropna(subset=["entry_ts"], inplace=True)
    
    df.sort_values("entry_ts", inplace=True)
    
    initial_len = len(df)
    df = _enforce_causality_drop(df, lookahead=5)
    log.info(f"Causality Check: Dropped {initial_len - len(df)} rows.")
    
    df["entry_ts_dt"] = pd.to_datetime(df["entry_ts"], unit="ms", utc=True)
    
    candidates = [c for c in df.columns if is_feature_column(df, c, extra_exclude)]
    keep = []
    for c in candidates:
        s = pd.to_numeric(df[c], errors='coerce')
        if s.isnull().all() or s.dropna().nunique() <= 1: continue
        keep.append(c)
        df[c] = s
        
    log.info(f"Selected {len(keep)} Normalized Features. Total Samples: {len(df)}")
    return df[keep + ["label", "entry_ts_dt", "pnl_r"]].copy(), sorted(keep)

def _sanitize_data(X: np.ndarray) -> np.ndarray:
    X = X.copy()
    X[np.isinf(X)] = np.nan
    limit = 1e12
    X[np.abs(X) > limit] = np.nan
    return X.astype(np.float32)

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-win-r", type=float, default=0.0)
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--reg-alpha", type=float, default=0.1)
    ap.add_argument("--reg-lambda", type=float, default=1.0)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--filter-side", choices=["all","long","short"], default="all")
    ap.add_argument("--exclude-cols", type=str, default="")
    ap.add_argument("--weight-decay", type=float, default=0.999)
    ap.add_argument("--wfa-folds", type=int, default=5)
    
    args = ap.parse_args(argv)

    try:
        df, feature_names = load_events_df(args.events, args.min_win_r, args.filter_side, set(_parse_extras(args.exclude_cols)))
    except Exception as e: 
        log.error(f"Data load failed: {e}")
        return

    X_all = df[feature_names].values
    y_all = df["label"].values
    
    sample_weights = _calculate_recency_and_pnl_weights(df)
    log.info(f"Applying PnL Weights. Head (Old)={sample_weights[0]:.4f}, Tail (New)={sample_weights[-1]:.4f}")

    X_all = _sanitize_data(X_all)

    log.info(f"Running Walk-Forward Analysis ({args.wfa_folds} folds) on ALL Features...")
    tscv = TimeSeriesSplit(n_splits=args.wfa_folds)
    
    auc_scores = []
    brier_scores = []
    
    for i, (train_idx, test_idx) in enumerate(tscv.split(X_all)):
        X_tr, y_tr = X_all[train_idx], y_all[train_idx]
        X_te, y_te = X_all[test_idx], y_all[test_idx]
        w_tr = sample_weights[train_idx]
        
        if w_tr.sum() > 0:
            w_tr = w_tr * (len(w_tr) / w_tr.sum())

        num_pos = int(np.sum(y_tr.astype(int)))
        num_neg = int(len(y_tr)) - num_pos
        spw = min(num_neg / max(num_pos, 1), 5.0)
        
        clf = xgb.XGBClassifier(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=3,
            reg_alpha=args.reg_alpha,
            reg_lambda=args.reg_lambda,
            colsample_bytree=0.7,
            subsample=0.8,
            scale_pos_weight=spw,
            random_state=42,
            eval_metric='logloss'
        )
            
        clf.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_te, y_te)], verbose=False)
        
        if args.calibrate:
            cal_clf = CalibratedClassifierCV(clf, method='isotonic', cv=3) 
            try: cal_clf.fit(X_tr, y_tr, sample_weight=w_tr)
            except: cal_clf.fit(X_tr, y_tr) 
            model = cal_clf
        else:
            model = clf
            
        probs = model.predict_proba(X_te)[:, 1]
        
        if len(np.unique(probs)) < 2:
            log.warning(f"  Fold {i+1}: Model Collapse Detected (Constant output).")
            auc = 0.5
            brier = 0.25
        else:
            try:
                auc = roc_auc_score(y_te, probs)
                brier = brier_score_loss(y_te, probs)
            except ValueError:
                auc = 0.5
                brier = 0.25
        
        auc_scores.append(auc)
        brier_scores.append(brier)
        log.info(f"  Fold {i+1}: AUC={auc:.4f} | Brier={brier:.4f}")

    avg_auc = np.mean(auc_scores)
    avg_brier = np.mean(brier_scores)
    log.info(f"✅ WFA RESULTS: Avg AUC = {avg_auc:.4f} | Avg Brier = {avg_brier:.4f}")

    log.info("Feature Pruning: Evaluating feature importance on FULL dataset...")

    # FIX MINORITY CLASS STARVATION (FULL DATASET)
    num_pos_all = int(np.sum(y_all.astype(int)))
    num_neg_all = int(len(y_all)) - num_pos_all
    spw_all = min(num_neg_all / max(num_pos_all, 1), 5.0)

    feature_eval_model = xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=3,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        colsample_bytree=0.7,
        subsample=0.8,
        scale_pos_weight=spw_all,
        random_state=42,
        eval_metric='logloss'
    )
    feature_eval_model.fit(X_all, y_all, sample_weight=sample_weights)

    importances = feature_eval_model.feature_importances_

    # Select top 30 features
    num_top_features = min(30, len(feature_names))
    top_indices = np.argsort(importances)[::-1][:num_top_features]

    top_feature_names = [feature_names[i] for i in top_indices]
    log.info(f"Selected Top {num_top_features} Features: {top_feature_names}")

    X_top = X_all[:, top_indices]

    final_base = xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=3,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        colsample_bytree=0.7,
        subsample=0.8,
        scale_pos_weight=spw_all,
        random_state=42,
        eval_metric='logloss'
    )
        
    final_model = final_base
    
    log.info("Training Final Production Model on FULL DATASET (Weighted) with Top Features...")
    if args.calibrate:
        final_cal = CalibratedClassifierCV(final_base, method='isotonic', cv=5)
        try: final_cal.fit(X_top, y_all, sample_weight=sample_weights)
        except: final_cal.fit(X_top, y_all)
        final_model = final_cal
    else:
        final_model.fit(X_top, y_all, sample_weight=sample_weights)

    pipeline = {
        "classifier": final_model, 
        "feature_names": top_feature_names,
        "training_args": vars(args), 
        "invert_prob": False, 
        "model": "xgboost",
        "wfa_auc": avg_auc,
        "wfa_brier": avg_brier
    }
    
    joblib.dump(pipeline, args.out)
    log.info(f"Saved Apex Predator XGBoost Brain to {args.out}")

if __name__ == "__main__":
    main()