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
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import RFE
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
    
    if "meta_label" in df.columns and "rr" in df.columns:
        # Target is 1 ONLY if the maximum favorable excursion reached the planned RR
        df["label"] = (df["meta_label"] >= df["rr"]).astype(int)
        log.info("🎯 Meta-Labeling detected. Target dynamically mapped to structural RR (MFE >= RR).")
    elif "meta_label" in df.columns:
        df["label"] = (df["meta_label"] >= 1.0).astype(int)
        log.info("🎯 Meta-Labeling detected. Target dynamically mapped to structural RR (MFE >= RR).")
    else:
        log.warning("⚠️ No Meta-Label detected. Falling back to PnL for binary label.")
        df["label"] = (df["pnl_r"] > 0).astype(int)
        
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
    meta_cols = [c for c in ["side", "strategy"] if c in df.columns]
    return df[keep + ["label", "entry_ts_dt", "pnl_r"] + meta_cols].copy(), sorted(keep)

def _sanitize_data(X: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num(X, nan=0.0), -1e10, 1e10).astype(np.float32)

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", nargs="+", required=True)
    ap.add_argument("--brains-dir", required=True, help="Directory to save individual strategy brains")
    ap.add_argument("--min-win-r", type=float, default=0.0)
    ap.add_argument("--n-estimators", type=int, default=100)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--max-depth", type=int, default=5)
    ap.add_argument("--reg-alpha", type=float, default=0.1)
    ap.add_argument("--reg-lambda", type=float, default=5.0)
    ap.add_argument("--max-features", type=int, default=20, help="Number of features to keep after RFE")
    ap.add_argument("--exclude-cols", type=str, default="")
    ap.add_argument("--weight-decay", type=float, default=0.999)
    ap.add_argument("--wfa-folds", type=int, default=5)
    ap.add_argument("--tune", action="store_true", help="Enable RandomizedSearchCV for hyperparameter tuning.")
    
    args = ap.parse_args(argv)

    # BULLETPROOF TRAINING PARAMS: Forcefully prevent users from extreme overfitting via CLI
    args.max_depth = min(args.max_depth, 9)
    args.n_estimators = min(args.n_estimators, 500)

    try:
        # Load all sides, we will filter manually
        df, base_feature_names = load_events_df(args.events, args.min_win_r, "all", set(_parse_extras(args.exclude_cols)))
    except Exception as e: 
        log.error(f"Data load failed: {e}")
        return

    out_dir = Path(args.brains_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all strategies
    if "strategy" not in df.columns:
        log.warning("No 'strategy' column found in exported events! Defaulting to 'UNKNOWN'.")
        df["strategy"] = "UNKNOWN"

    # Extract all distinct setups
    base_strategies = list(set([s.split('_')[0] for s in df["strategy"].unique()]))
    for strat_clean in base_strategies:
        # Strat names often have _LONG / _SHORT appended in the strategy string (e.g. ALPHA_LONG)
        # OR side is saved separately
        for side in ["long", "short"]:
            side_val = 1.0 if side == "long" else 0.0
            mask = (df["side"] == side_val) | (df["side"] == side) | (df["strategy"].str.endswith(side.upper()))
            subset = df[(df["strategy"].str.startswith(strat_clean)) & mask].copy()

            if len(subset) < 30:
                log.info(f"Skipping {strat_clean}_{side} - Not enough samples ({len(subset)})")
                continue

            log.info(f"--- Training Isolated Brain: {strat_clean}_{side} ({len(subset)} samples) ---")

            X_all = subset[base_feature_names].values
            y_all = subset["label"].values

            sample_weights = _calculate_recency_and_pnl_weights(subset)
            X_all = _sanitize_data(X_all)

            pos_cases = np.sum(y_all == 1)
            neg_cases = np.sum(y_all == 0)
            scale_weight = float(neg_cases) / max(1.0, float(pos_cases))
            scale_weight = min(scale_weight, 10.0) # Cap extreme weights

            # --- Feature Selection (RFE) ---
            log.info("Running RFE Feature Selection...")
            estimator = xgb.XGBClassifier(n_estimators=25, max_depth=2, random_state=42, objective='binary:logistic', eval_metric='logloss', scale_pos_weight=scale_weight)
            selector = RFE(estimator, n_features_to_select=min(args.max_features, len(base_feature_names)), step=1)
            selector = selector.fit(X_all, y_all)
            selected_features = [f for f, s in zip(base_feature_names, selector.support_) if s]

            X_all_sel = X_all[:, selector.support_]
            log.info(f"RFE selected {len(selected_features)} features.")

            n_samples = len(X_all_sel)
            actual_folds = min(args.wfa_folds, max(2, n_samples // 15))
            actual_gap = min(10, n_samples // 10)
            log.info(f"Running Walk-Forward Analysis ({actual_folds} folds)...")
            tscv = TimeSeriesSplit(n_splits=actual_folds, gap=actual_gap)
            
            acc_scores = []
            prec_scores = []
            roc_auc_scores = []
            f1_scores = []
            
            for i, (train_idx, test_idx) in enumerate(tscv.split(X_all_sel)):
                X_tr, y_tr = X_all_sel[train_idx], y_all[train_idx]
                X_te, y_te = X_all_sel[test_idx], y_all[test_idx]
                w_tr = sample_weights[train_idx]

                if w_tr.sum() > 0:
                    w_tr = w_tr * (len(w_tr) / w_tr.sum())

                clf = xgb.XGBClassifier(
                    n_estimators=args.n_estimators,
                    learning_rate=args.learning_rate,
                    max_depth=args.max_depth,
                    reg_alpha=args.reg_alpha,
                    reg_lambda=args.reg_lambda,
                    colsample_bytree=0.7,
                    subsample=0.8,
                    random_state=42,
                    objective='binary:logistic',
                    eval_metric='logloss',
                    scale_pos_weight=scale_weight
                )

                # Hyperparameter Tuning Support
                if getattr(args, 'tune', False):
                    from sklearn.model_selection import RandomizedSearchCV
                    param_dist = {
                        'learning_rate': [0.01, 0.05, 0.1, 0.2],
                        'max_depth': [1, 2, 3, 4],
                        'n_estimators': [50, 100, 150],
                        'reg_alpha': [0.1, 1.0, 5.0],
                        'reg_lambda': [0.1, 1.0, 5.0],
                        'subsample': [0.6, 0.8, 1.0],
                        'colsample_bytree': [0.6, 0.8, 1.0]
                    }
                    cv_folds = min(3, max(2, len(X_tr) // 15))
                    random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=10, scoring='roc_auc', cv=cv_folds, random_state=42)
                    random_search.fit(X_tr, y_tr, sample_weight=w_tr)
                    clf = random_search.best_estimator_
                    log.info(f"    Tuned Params: {random_search.best_params_}")

                clf.fit(X_tr, y_tr, sample_weight=w_tr)

                preds = clf.predict(X_te)
                probs = clf.predict_proba(X_te)[:, 1] if len(np.unique(y_te)) > 1 else np.zeros_like(preds)

                acc = accuracy_score(y_te, preds)
                prec = precision_score(y_te, preds, zero_division=0)
                roc = roc_auc_score(y_te, probs) if len(np.unique(y_te)) > 1 else 0.5
                f1 = f1_score(y_te, preds, zero_division=0)

                acc_scores.append(acc)
                prec_scores.append(prec)
                roc_auc_scores.append(roc)
                f1_scores.append(f1)

                log.info(f"  Fold {i+1}: Acc={acc:.4f} | Prec={prec:.4f} | ROC={roc:.4f} | F1={f1:.4f}")

            avg_acc = np.mean(acc_scores)
            avg_prec = np.mean(prec_scores)
            avg_roc = np.mean(roc_auc_scores)
            avg_f1 = np.mean(f1_scores)

            log.info(f"✅ WFA RESULTS: Avg Acc = {avg_acc:.4f} | Avg Prec = {avg_prec:.4f} | Avg ROC-AUC = {avg_roc:.4f} | Avg F1 = {avg_f1:.4f}")

            log.info("Calibrating Final Production Model on final fold...")

            final_model = xgb.XGBClassifier(
                n_estimators=args.n_estimators,
                learning_rate=args.learning_rate,
                max_depth=args.max_depth,
                reg_alpha=args.reg_alpha,
                reg_lambda=args.reg_lambda,
                colsample_bytree=0.7,
                subsample=0.8,
                random_state=42,
                objective='binary:logistic',
                eval_metric='logloss',
                scale_pos_weight=scale_weight
            )

            # Fit the final model on the training set of the last fold
            final_model.fit(X_tr, y_tr, sample_weight=w_tr)

            # Calibrate probabilities to fix distortion from class weights
            calibrated_model = CalibratedClassifierCV(estimator=final_model, method='sigmoid', cv='prefit')
            calibrated_model.fit(X_te, y_te) # Fit on the validation set of the last fold

            pipeline = {
                "classifier": calibrated_model,
                "feature_names": selected_features,
                "training_args": vars(args),
                "model": "xgboost_calibrated",
                "wfa_acc": avg_acc,
                "wfa_prec": avg_prec,
                "wfa_roc_auc": avg_roc,
                "wfa_f1": avg_f1
            }

            out_file = out_dir / f"brain_{strat_clean}_{side}.joblib"
            joblib.dump(pipeline, out_file)
            log.info(f"Saved Apex Predator XGBoost Brain to {out_file}\n")

if __name__ == "__main__":
    main()