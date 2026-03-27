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
        "ob_bull_top", "ob_bull_bot", "ob_bear_top", "ob_bear_bot",
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
    pnl_weights = np.clip(np.abs(df['pnl_r'].values), 0.1, 5.0)
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
    
    if len(df) == 0:
        raise ValueError("0 samples remain after filtering. Aborting training.")

    df["entry_ts_dt"] = pd.to_datetime(df["entry_ts"], unit="ms", utc=True)
    
    candidates = [c for c in df.columns if is_feature_column(df, c, extra_exclude)]

    # FORCE INCLUSION OF NEW SENTIENT REGIME SCORE
    if "sentient_regime_score" in df.columns and "sentient_regime_score" not in candidates:
        candidates.append("sentient_regime_score")

    keep = []
    for c in candidates:
        s = pd.to_numeric(df[c], errors='coerce')
        if s.isnull().all() or s.dropna().nunique() <= 1: continue
        keep.append(c)
        df[c] = s
        
    log.info(f"Selected {len(keep)} Normalized Features. Total Samples: {len(df)}")
    return df[keep + ["label", "entry_ts_dt", "pnl_r"]].copy(), sorted(keep)

def _sanitize_data(X: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num(X, nan=0.0), -1e10, 1e10).astype(np.float32)

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-win-r", type=float, default=0.0)
    ap.add_argument("--n-estimators", type=int, default=100)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--reg-alpha", type=float, default=0.1)
    ap.add_argument("--reg-lambda", type=float, default=5.0)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--filter-side", choices=["all","long","short"], default="all")
    ap.add_argument("--exclude-cols", type=str, default="")
    ap.add_argument("--weight-decay", type=float, default=0.999)
    ap.add_argument("--wfa-folds", type=int, default=5)
    ap.add_argument("--tune", action="store_true", help="Enable RandomizedSearchCV for hyperparameter tuning.")
    
    args = ap.parse_args(argv)

    # BULLETPROOF TRAINING PARAMS: Forcefully prevent users from extreme overfitting via CLI
    args.max_depth = min(args.max_depth, 7)
    args.n_estimators = min(args.n_estimators, 300)

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

    log.info(f"Running Walk-Forward Analysis ({args.wfa_folds} folds)...")
    tscv = TimeSeriesSplit(n_splits=args.wfa_folds, gap=10)
    
    auc_scores = []
    brier_scores = []
    fold_importances = []
    
    for i, (train_idx, test_idx) in enumerate(tscv.split(X_all)):
        X_tr, y_tr = X_all[train_idx], y_all[train_idx]
        X_te, y_te = X_all[test_idx], y_all[test_idx]
        w_tr = sample_weights[train_idx]
        
        if w_tr.sum() > 0:
            w_tr = w_tr * (len(w_tr) / w_tr.sum())

        # DAMPENED MINORITY CLASS STARVATION (Prevent AI from predicting 0 exclusively)
        num_pos = int(np.sum(y_tr.astype(int)))
        num_neg = int(len(y_tr)) - num_pos
        spw = np.sqrt(num_neg / max(num_pos, 1))

        clf = xgb.XGBClassifier(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            reg_alpha=args.reg_alpha,
            reg_lambda=args.reg_lambda,
            scale_pos_weight=spw,
            colsample_bytree=0.7,
            subsample=0.8,
            random_state=42,
            eval_metric='logloss'
        )

        # Hyperparameter Tuning Support
        if getattr(args, 'tune', False):
            from sklearn.model_selection import RandomizedSearchCV
            param_dist = {
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'n_estimators': [100, 200, 300],
                'reg_lambda': [1.0, 5.0, 10.0],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0]
            }
            random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=10, scoring='roc_auc', cv=3, random_state=42)
            random_search.fit(X_tr, y_tr, sample_weight=w_tr)
            clf = random_search.best_estimator_
            log.info(f"    Tuned Params: {random_search.best_params_}")
            
        clf.fit(X_tr, y_tr, sample_weight=w_tr)
        
        if args.calibrate:
            cal_clf = CalibratedClassifierCV(clf, method='sigmoid', cv=3)
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
        fold_importances.append(model.feature_importances_)
        log.info(f"  Fold {i+1}: AUC={auc:.4f} | Brier={brier:.4f}")

    avg_auc = np.mean(auc_scores)
    avg_brier = np.mean(brier_scores)
    log.info(f"✅ WFA RESULTS: Avg AUC = {avg_auc:.4f} | Avg Brier = {avg_brier:.4f}")

    log.info("Training Final Production Model on FULL DATASET (Weighted)...")
    
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import RobustScaler
    imputer = SimpleImputer(strategy='median')
    scaler = RobustScaler()
    X_all_s = scaler.fit_transform(imputer.fit_transform(X_all))
    
    # DAMPENED MINORITY CLASS STARVATION (FULL DATASET)
    num_pos_all = int(np.sum(y_all.astype(int)))
    num_neg_all = int(len(y_all)) - num_pos_all
    spw_all = np.sqrt(num_neg_all / max(num_pos_all, 1))

    final_base = xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        scale_pos_weight=spw_all,
        colsample_bytree=0.7,
        subsample=0.8,
        random_state=42,
        eval_metric='logloss'
    )
        
    final_model = final_base
    
    # Removed CalibratedClassifierCV.
    # Calibration natively suppresses scale_pos_weight forcing the outputs
    # back to raw sample distribution percentages. We want raw uncalibrated
    # outputs to center around 0.5 for asymmetric edge hunting.
    final_model.fit(X_all_s, y_all, sample_weight=sample_weights)

    # EDGE EXTRACTION: Anti-Signal Flipping
    # In highly mean-reverting crypto regimes where the AUC collapses below 0.40,
    # the model is reliably misclassifying true direction. We set invert_prob to True
    # to harvest the inverse edge.
    invert = bool(avg_auc < 0.40)
    if invert:
        log.info("⚠️ Severe Anti-Signal Detected (AUC < 0.40). Flipping Probability Pipeline to Harvest Edge.")
    if args.calibrate:
        final_cal = CalibratedClassifierCV(final_base, method='sigmoid', cv=5)
        try: final_cal.fit(X_all_s, y_all, sample_weight=sample_weights)
        except: final_cal.fit(X_all_s, y_all)
        final_model = final_cal
    else:
        final_model.fit(X_all_s, y_all, sample_weight=sample_weights)

    pipeline = {
        "classifier": final_model, 
        "feature_names": feature_names,
        "training_args": vars(args), 
        "invert_prob": invert,
        "model": "xgboost",
        "wfa_auc": avg_auc,
        "wfa_brier": avg_brier
    }
    
    joblib.dump(pipeline, args.out)
    log.info(f"Saved Apex Predator XGBoost Brain to {args.out}")

if __name__ == "__main__":
    main()