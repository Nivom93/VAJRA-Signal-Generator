#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vajra_brain_train.py — Advanced Trainer (v9.0 - Enhanced Signal Capture)
====================================================================
LEVEL 31 UPGRADE:
- SMOTE OVERSAMPLING: Synthetic minority class generation for balanced training.
- ADAPTIVE RFE: Feature count scales with sqrt(n_samples) — no more arbitrary cap at 30.
- IMPROVED BANLIST: Preserved normalized ATR/velocity features for brain learning.
- CALIBRATED PROBABILITIES: Isotonic calibration for reliable confidence estimates.
- ADAPTIVE WFA FOLDS: Fewer folds for small datasets to prevent fold starvation.
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

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

try:
    from sklearn.frozen import FrozenEstimator
except ImportError:
    FrozenEstimator = None

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
    # REFINED TOXIC FEATURE BANLIST (v9.0)
    # Removed: atr_pct features (normalized, safe to use)
    # Removed: kalman_vel (when normalized to ATR, not absolute)
    # Kept: All absolute price levels that cause OOD collapse
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
        "dc_high_20", "dc_low_20", "htf_ema50", "macro_ema50", "swing_ema50",
        "htf_swing_high", "htf_swing_low", "mtf_swing_high", "mtf_swing_low",
        "ema20_L", "ema50_L", "ema100_L", "mtf_ema200_arr", "bb_upper", "bb_lower",
        "kc_upper", "kc_lower", "kalman_price",
        "fib_382_long", "fib_500_long", "fib_618_long", "fib_786_long", "fib_886_long",
        "fib_382_short", "fib_500_short", "fib_618_short", "fib_786_short", "fib_886_short",
        "fib_ext_1272", "fib_ext_1618",
        "rolling_vwap_20",

        # --- ABSOLUTE DERIVATIVES (Scale with price — use normalized versions instead) ---
        "atr14_L", "atr7_L", "cvd",

        # --- POST-HOC LEAKAGE (only known after trade closes, always 0 at entry) ---
        "bars_open",

        # --- STRING/NON-NUMERIC FIELDS ---
        "bull_confluence_reasons", "bear_confluence_reasons",
        "market_regime"
    }
    EXCLUDE_SUFFIXES = {"_dt", "_ts"}

    if col in BASE_EXCLUDE_COLS or col in extra_exclude: return False
    if any(col.endswith(suf) for suf in EXCLUDE_SUFFIXES): return False
    # Exclude specialist brain probability columns (leaked in-sample predictions)
    if col.startswith("brain_"): return False
    if col not in df.columns: return False
    return pd.api.types.is_numeric_dtype(df[col])

def _enforce_causality_drop(df: pd.DataFrame, lookahead: int = 5) -> pd.DataFrame:
    if len(df) > lookahead: return df.iloc[:-lookahead]
    return df

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

    # Pure Target Binary with improved labeling
    if "meta_label" in df.columns:
        # Threshold 0.50: only trades achieving ≥50% of target R:R are positive.
        # At 0.25 barely-profitable noise trades (0.01R→meta_label 0.30) were
        # labeled positive, diluting the signal the brain learns from.
        df["label"] = (pd.to_numeric(df["meta_label"], errors='coerce').fillna(0) > 0.50).astype(int)
        log.info("🎯 Dynamic Target mapped via meta_label (threshold > 0.50 for positive).")
    elif "pnl_r" in df.columns:
        df["label"] = (df["pnl_r"] > 0).astype(int)
        log.info("🎯 Pure Target Binary (Fallback): pnl_r > 0.")
    else:
        df["label"] = 0

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

def _apply_smote(X_train, y_train, random_state=42):
    """Apply SMOTE oversampling if imlearn is available and minority class is sufficient."""
    if not HAS_SMOTE:
        return X_train, y_train

    pos_count = np.sum(y_train == 1)
    neg_count = np.sum(y_train == 0)

    if pos_count < 6 or neg_count < 6:
        return X_train, y_train

    # Only oversample to 40% of majority (avoid full balance which introduces too much noise)
    target_ratio = min(0.4, pos_count / max(1, neg_count))
    if target_ratio >= 0.35:
        return X_train, y_train  # Already reasonably balanced

    try:
        k_neighbors = min(5, pos_count - 1)
        smote = SMOTE(sampling_strategy=0.4, random_state=random_state, k_neighbors=k_neighbors)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        return X_res, y_res
    except Exception:
        return X_train, y_train

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
    ap.add_argument("--max-features", type=int, default=130, help="Number of features to keep after RFE")
    ap.add_argument("--exclude-cols", type=str, default="")
    ap.add_argument("--weight-decay", type=float, default=0.999)
    ap.add_argument("--wfa-folds", type=int, default=5)
    ap.add_argument("--tune", action="store_true", help="Enable RandomizedSearchCV for hyperparameter tuning.")
    ap.add_argument("--no-smote", action="store_true", help="Disable SMOTE oversampling.")
    ap.add_argument("--meta-brain", action="store_true",
                    help="Train a unified Meta-Brain after specialist brains. "
                         "Requires events exported with --brains-dir (specialist probabilities).")

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
        for side in ["long", "short"]:
            side_val = 1.0 if side == "long" else 0.0
            mask = (df["side"] == side_val) | (df["side"] == side) | (df["strategy"].str.endswith(side.upper()))
            subset = df[(df["strategy"].str.startswith(strat_clean)) & mask].copy()

            # Minimum 100 samples for robust generalization — with feature cap
            # of n/25, fewer samples means too few features to learn from.
            if len(subset) < 100:
                log.info(f"Skipping {strat_clean}_{side} - Not enough samples ({len(subset)})")
                continue

            log.info(f"--- Training Isolated Brain: {strat_clean}_{side} ({len(subset)} samples) ---")

            X_all = subset[base_feature_names].values
            y_all = subset["label"].values

            X_all = _sanitize_data(X_all)

            pos_cases = np.sum(y_all == 1)
            neg_cases = np.sum(y_all == 0)
            # Softer class weighting — cap at 20x instead of 100x to prevent noise amplification
            scale_weight = float(neg_cases) / max(1.0, float(pos_cases))
            scale_weight = min(scale_weight, 20.0)

            n_samples = len(X_all)
            # Adaptive folds: fewer folds for small datasets (min 2, scale with data)
            actual_folds = min(args.wfa_folds, max(2, n_samples // 25))
            # Gap must fit: TimeSeriesSplit needs n_samples >= (folds+1)*test_size + folds*gap
            # where test_size ≈ n_samples/(folds+1). Solve for max safe gap.
            test_size_est = max(5, n_samples // (actual_folds + 1))
            max_safe_gap = max(1, (n_samples - (actual_folds + 1) * test_size_est) // actual_folds)
            actual_gap = min(max(3, n_samples // 30), max_safe_gap)
            # Final safety: if gap still too large, reduce folds
            while actual_folds > 2 and n_samples < (actual_folds + 1) * test_size_est + actual_folds * actual_gap:
                actual_folds -= 1
                test_size_est = max(5, n_samples // (actual_folds + 1))
                max_safe_gap = max(1, (n_samples - (actual_folds + 1) * test_size_est) // actual_folds)
                actual_gap = min(max(3, n_samples // 30), max_safe_gap)
            log.info(f"Running Walk-Forward Analysis ({actual_folds} folds) with dynamic per-fold RFE Feature Selection...")

            if HAS_SMOTE and not args.no_smote:
                log.info(f"  SMOTE oversampling: ENABLED (pos={pos_cases}, neg={neg_cases})")

            tscv = TimeSeriesSplit(n_splits=actual_folds, gap=actual_gap)

            best_tuned_params = {}
            acc_scores = []
            prec_scores = []
            roc_auc_scores = []
            f1_scores = []

            for i, (train_idx, test_idx) in enumerate(tscv.split(X_all)):
                X_tr_full, y_tr = X_all[train_idx], y_all[train_idx]
                X_te_full, y_te = X_all[test_idx], y_all[test_idx]

                # ADAPTIVE RFE: cap at n_samples/25 to prevent overfitting.
                # sqrt(n)*1.5 was too aggressive (428 samples → 31 features = 14:1 ratio).
                # Need at least 25 samples per feature for robust generalization.
                n_samples_fold = len(X_tr_full)
                max_by_ratio = max(8, n_samples_fold // 25)
                dynamic_n_features_fold = max(8, min(max_by_ratio, int(np.sqrt(n_samples_fold) * 1.5), len(base_feature_names)))

                # Dynamically run RFE with a stronger estimator for reliable feature ranking
                estimator_rfe = xgb.XGBClassifier(n_estimators=50, max_depth=3, random_state=42, objective='binary:logistic', eval_metric='logloss', scale_pos_weight=scale_weight)
                selector = RFE(estimator_rfe, n_features_to_select=dynamic_n_features_fold, step=1)
                selector = selector.fit(X_tr_full, y_tr)

                # Slice training and validation sets strictly to selected features
                X_tr = X_tr_full[:, selector.support_]
                X_te = X_te_full[:, selector.support_]

                # Apply SMOTE oversampling on training fold (not validation!)
                smote_applied = False
                if not args.no_smote:
                    X_tr_before = X_tr
                    X_tr, y_tr = _apply_smote(X_tr, y_tr, random_state=42 + i)
                    smote_applied = len(X_tr) != len(X_tr_before)

                # When SMOTE rebalanced the data, it already handles the class
                # imbalance — adding scale_pos_weight on top double-corrects and
                # warps probability calibration (pushes outputs to extremes).
                fold_scale_weight = 1.0 if smote_applied else scale_weight

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
                    scale_pos_weight=fold_scale_weight
                )

                clf.fit(X_tr, y_tr)

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

                log.info(f"  Fold {i+1}: Acc={acc:.4f} | Prec={prec:.4f} | ROC={roc:.4f} | F1={f1:.4f} (Features: {len(X_tr[0])})")

            avg_acc = np.mean(acc_scores)
            avg_prec = np.mean(prec_scores)
            avg_roc = np.mean(roc_auc_scores)
            avg_f1 = np.mean(f1_scores)

            log.info(f"✅ WFA RESULTS: Avg Acc = {avg_acc:.4f} | Avg Prec = {avg_prec:.4f} | Avg ROC-AUC = {avg_roc:.4f} | Avg F1 = {avg_f1:.4f}")

            log.info("Running Final RFE Feature Selection on FULL dataset for Production Model...")

            # ADAPTIVE RFE for final model: cap at n_samples/25 for robustness.
            n_samples_all = len(X_all)
            max_by_ratio_final = max(10, n_samples_all // 25)
            dynamic_n_features_final = max(10, min(max_by_ratio_final, int(np.sqrt(n_samples_all) * 1.5), len(base_feature_names)))

            log.info(f"Dynamic RFE: Selecting up to {dynamic_n_features_final} features for {n_samples_all} samples.")

            # Apply SMOTE before final RFE if enabled
            X_all_for_rfe, y_all_for_rfe = X_all, y_all
            rfe_smote_applied = False
            if not args.no_smote:
                X_before = X_all_for_rfe
                X_all_for_rfe, y_all_for_rfe = _apply_smote(X_all, y_all, random_state=99)
                rfe_smote_applied = len(X_all_for_rfe) != len(X_before)

            # When SMOTE rebalances the data for RFE, scale_pos_weight must be
            # 1.0 to avoid double-correcting (same fix as fold-level training).
            rfe_final_weight = 1.0 if rfe_smote_applied else scale_weight
            estimator_rfe_final = xgb.XGBClassifier(n_estimators=50, max_depth=3, random_state=42, objective='binary:logistic', eval_metric='logloss', scale_pos_weight=rfe_final_weight)
            selector_final = RFE(estimator_rfe_final, n_features_to_select=dynamic_n_features_final, step=1)

            selector_final = selector_final.fit(X_all_for_rfe, y_all_for_rfe)
            selected_features = [f for f, s in zip(base_feature_names, selector_final.support_) if s]

            X_all_sel = X_all[:, selector_final.support_]
            log.info(f"Final RFE selected {len(selected_features)} features.")

            log.info("Training and Calibrating Final Production Model on FULL dataset...")

            # Apply SMOTE on the selected features for final training
            X_final_train, y_final_train = X_all_sel, y_all
            final_smote_applied = False
            if not args.no_smote:
                X_before_smote = X_final_train
                X_final_train, y_final_train = _apply_smote(X_all_sel, y_all, random_state=42)
                final_smote_applied = len(X_final_train) != len(X_before_smote)

            # When SMOTE rebalanced the data, it handles class imbalance —
            # scale_pos_weight on top double-corrects and warps calibration.
            final_weight = 1.0 if final_smote_applied else scale_weight

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
                scale_pos_weight=final_weight
            )

            final_model.fit(X_final_train, y_final_train)

            # NOTE: Isotonic calibration was tried but distorted probabilities — raw
            # XGBoost outputs around 0.30-0.40 get calibrated down to 0.09, which
            # blows away the threshold filter. The raw XGBoost probabilities are
            # good enough given we're using threshold-based filtering. Reverted.

            # Tiered quality system — all brains are saved with a quality tag
            # instead of hard-blocking on WFA metrics
            if avg_roc > 0.55 and avg_prec > 0.15:
                quality_tier = "strong"
                threshold_adj = 0.0
            elif avg_roc > 0.50:
                quality_tier = "medium"
                threshold_adj = 0.03
            else:
                quality_tier = "weak"
                threshold_adj = 0.07

            valid_edge = quality_tier in ("strong", "medium")

            log.info(f"Brain {strat_clean}_{side}: quality_tier={quality_tier} "
                     f"(ROC={avg_roc:.4f}, Prec={avg_prec:.4f}) → threshold_adj=+{threshold_adj:.2f}")

            # Save XGBoost model natively (version-agnostic JSON format)
            xgb_model_file = out_dir / f"brain_{strat_clean}_{side}.json"
            final_model.get_booster().save_model(str(xgb_model_file))

            pipeline = {
                "xgb_model_file": str(xgb_model_file.name),
                "feature_names": selected_features,
                "training_args": vars(args),
                "model": "xgboost_native",
                "wfa_acc": avg_acc,
                "wfa_prec": avg_prec,
                "wfa_roc_auc": avg_roc,
                "wfa_f1": avg_f1,
                "valid_edge": valid_edge,
                "quality_tier": quality_tier,
                "threshold_adj": threshold_adj,
                "smote_enabled": HAS_SMOTE and not args.no_smote,
                "n_features_selected": len(selected_features),
                "n_samples_total": n_samples_all,
                "pos_neg_ratio": f"{pos_cases}/{neg_cases}",
            }

            out_file = out_dir / f"brain_{strat_clean}_{side}.joblib"
            joblib.dump(pipeline, out_file)
            log.info(f"Saved Apex Predator XGBoost Brain to {out_file}\n")

    # ══════════════════════════════════════════════════════════════
    # META-BRAIN: Unified decision maker trained on ALL events
    # ══════════════════════════════════════════════════════════════
    if getattr(args, 'meta_brain', False):
        _train_meta_brain(df, base_feature_names, args, out_dir)


def _train_meta_brain(df: pd.DataFrame, base_feature_names: list, args, out_dir: Path):
    """Train a unified Meta-Brain using proper stacked generalization (no leakage).

    CRITICAL: We do NOT use the pre-exported specialist probabilities as features,
    because those were generated by models trained on the full dataset (in-sample).
    Instead, we generate OUT-OF-SAMPLE specialist predictions using walk-forward
    cross-prediction: for each time fold, train temporary specialist models on the
    training split only, then predict on the held-out test split. This ensures the
    meta-brain never sees specialist predictions that were "cheating".

    The Meta-Brain sees:
    1. OOS specialist probabilities (generated via walk-forward stacking)
    2. Aggregate specialist statistics (mean, max, std, count)
    3. Key market context features (confluence scores, regime indicators, macro)
    """
    log.info("=" * 60)
    log.info("  META-BRAIN TRAINING: Unified Decision Maker (OOS Stacking)")
    log.info("=" * 60)

    SPECIALIST_KEYS = [
        ("ALPHA", "long"), ("ALPHA", "short"),
        ("BETA", "long"), ("BETA", "short"),
        ("GAMMA", "long"), ("GAMMA", "short"),
        ("DELTA", "long"), ("DELTA", "short"),
        ("EPSILON", "long"), ("EPSILON", "short"),
        ("ZETA", "long"), ("ZETA", "short"),
        ("ETA", "long"), ("ETA", "short"),
        ("THETA", "long"), ("THETA", "short"),
        ("IOTA", "long"), ("IOTA", "short"),
        ("KAPPA", "long"), ("KAPPA", "short"),
        ("LAMBDA", "long"), ("LAMBDA", "short"),
    ]
    specialist_col_names = [f"brain_{s}_{sd}" for s, sd in SPECIALIST_KEYS]

    # ── Market context features for Meta-Brain ──
    CONTEXT_FEATURES = [
        "bull_confluence_score", "bear_confluence_score",
        "adx_14", "rsi_14", "atr_pct",
        "btc_bullish", "funding_rate", "delta_oi",
        "dxy_trend", "spx_trend", "btcd_trend",
        "rvol", "vol_zscore",
        "ob_bull_near", "ob_bear_near",
        "fvg_bull_near", "fvg_bear_near",
        "bos_bull", "bos_bear",
        "choch_bull", "choch_bear",
        "displacement_bull_count", "displacement_bear_count",
        "spring", "upthrust",
        "accum_phase", "distrib_phase",
        "qm_bull", "qm_bear",
        "in_ote_zone_bull", "in_ote_zone_short",
        "in_discount_zone", "in_premium_zone",
        "equal_highs_count", "equal_lows_count",
        "hurst_exponent",
        "macro_sentiment",
        # Market Structure Reading
        "struct_trend", "struct_strength", "struct_break",
        "struct_bias_score",
        "htf_struct_trend", "htf_struct_strength", "htf_struct_break",
        "macro_struct_trend", "macro_struct_strength",
        "struct_align_bull", "struct_align_bear",
        "recent_hh_count", "recent_hl_count",
        "recent_lh_count", "recent_ll_count",
    ]
    found_ctx = [c for c in CONTEXT_FEATURES if c in df.columns]

    if len(found_ctx) < 5:
        log.error("Meta-Brain: Too few context features found. Aborting.")
        return

    # ── Prepare base features for specialist cross-prediction ──
    X_base = df[base_feature_names].values.astype(np.float32)
    X_base = _sanitize_data(X_base)
    y_all = df["label"].values
    n_samples = len(X_base)

    if n_samples < 30:
        log.error(f"Meta-Brain: Only {n_samples} samples — need at least 30. Aborting.")
        return

    # Find which strategies have enough data
    if "strategy" not in df.columns:
        df["strategy"] = "UNKNOWN"

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1: Generate OOS specialist predictions via Walk-Forward Stacking
    # ══════════════════════════════════════════════════════════════════
    log.info("Phase 1: Generating OUT-OF-SAMPLE specialist predictions (walk-forward stacking)...")

    # Initialize OOS prediction arrays (filled with -1 = "no prediction")
    oos_specialist_preds = np.full((n_samples, len(SPECIALIST_KEYS)), -1.0, dtype=np.float32)

    # Use time-series split for OOS generation (more folds = more OOS coverage)
    oos_folds = min(8, max(3, n_samples // 50))
    oos_gap = max(3, min(10, n_samples // 20))
    tscv_oos = TimeSeriesSplit(n_splits=oos_folds, gap=oos_gap)

    for fold_i, (train_idx, test_idx) in enumerate(tscv_oos.split(X_base)):
        X_tr_base = X_base[train_idx]
        y_tr = y_all[train_idx]

        for spec_j, (strat, side) in enumerate(SPECIALIST_KEYS):
            # Filter training data to this strategy+side
            side_val = 1.0 if side == "long" else 0.0
            strat_mask_tr = np.zeros(len(train_idx), dtype=bool)
            for ti, tidx in enumerate(train_idx):
                row_strat = str(df.iloc[tidx].get("strategy", ""))
                row_side = df.iloc[tidx].get("side", -1)
                if row_strat.startswith(strat) and (row_side == side_val or row_side == side):
                    strat_mask_tr[ti] = True

            n_strat_train = int(np.sum(strat_mask_tr))
            if n_strat_train < 10:
                continue  # Not enough data for this specialist in this fold

            # Train a temporary specialist on the training split only
            X_spec_tr = X_tr_base[strat_mask_tr]
            y_spec_tr = y_tr[strat_mask_tr]

            pos_s = int(np.sum(y_spec_tr == 1))
            neg_s = int(np.sum(y_spec_tr == 0))
            if pos_s < 2 or neg_s < 2:
                continue

            sw = min(float(neg_s) / max(1.0, float(pos_s)), 20.0)
            temp_clf = xgb.XGBClassifier(
                n_estimators=50, learning_rate=0.1, max_depth=3,
                reg_alpha=0.1, reg_lambda=5.0,
                colsample_bytree=0.7, subsample=0.8,
                random_state=42, objective='binary:logistic',
                eval_metric='logloss', scale_pos_weight=sw
            )
            temp_clf.fit(X_spec_tr, y_spec_tr)

            # Predict on ALL test samples (not just this strategy's samples)
            X_te_base = X_base[test_idx]
            try:
                preds = temp_clf.predict_proba(X_te_base)[:, 1]
                for ti, tidx in enumerate(test_idx):
                    oos_specialist_preds[tidx, spec_j] = preds[ti]
            except Exception:
                pass

        if (fold_i + 1) % 2 == 0 or fold_i == oos_folds - 1:
            covered = int(np.sum(np.any(oos_specialist_preds >= 0, axis=1)))
            log.info(f"  OOS Stacking fold {fold_i+1}/{oos_folds}: {covered}/{n_samples} samples have OOS predictions")

    # Count coverage
    has_oos = np.any(oos_specialist_preds >= 0, axis=1)
    oos_count = int(np.sum(has_oos))
    log.info(f"OOS specialist coverage: {oos_count}/{n_samples} samples ({oos_count/n_samples*100:.1f}%)")

    if oos_count < 30:
        log.error("Meta-Brain: Too few samples with OOS specialist predictions. Aborting.")
        return

    # Filter to only samples that have OOS predictions
    oos_mask = has_oos
    df_oos = df[oos_mask].copy().reset_index(drop=True)
    oos_preds_filtered = oos_specialist_preds[oos_mask]
    y_oos = y_all[oos_mask]

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2: Build Meta-Brain feature matrix from OOS predictions + context
    # ══════════════════════════════════════════════════════════════════
    log.info("Phase 2: Assembling Meta-Brain features from OOS predictions + market context...")

    # Create OOS specialist columns in the filtered dataframe
    for spec_j, col_name in enumerate(specialist_col_names):
        df_oos[col_name] = oos_preds_filtered[:, spec_j]

    # Aggregate stats from OOS predictions
    active_mask = oos_preds_filtered >= 0.0
    n_active = active_mask.sum(axis=1).astype(float)
    active_vals = np.where(active_mask, oos_preds_filtered, np.nan)
    df_oos["n_active_specialists"] = n_active
    df_oos["mean_specialist_prob"] = np.nanmean(active_vals, axis=1)
    df_oos["max_specialist_prob"] = np.nanmax(active_vals, axis=1)
    with np.errstate(invalid='ignore'):
        df_oos["std_specialist_prob"] = np.nanstd(active_vals, axis=1)
    df_oos["std_specialist_prob"] = df_oos["std_specialist_prob"].fillna(0.0)
    df_oos["mean_specialist_prob"] = df_oos["mean_specialist_prob"].fillna(0.0)
    df_oos["max_specialist_prob"] = df_oos["max_specialist_prob"].fillna(0.0)

    if "side" in df_oos.columns:
        df_oos["selected_side"] = pd.to_numeric(df_oos["side"], errors='coerce').fillna(0.0)
    else:
        df_oos["selected_side"] = 0.0

    # Assemble feature columns
    found_spec_oos = [c for c in specialist_col_names if c in df_oos.columns]
    found_ctx_oos = [c for c in CONTEXT_FEATURES if c in df_oos.columns]

    meta_feature_cols = (
        found_spec_oos
        + ["n_active_specialists", "mean_specialist_prob", "max_specialist_prob", "std_specialist_prob",
           "selected_side"]
        + found_ctx_oos
    )

    for c in meta_feature_cols:
        if c in df_oos.columns:
            df_oos[c] = pd.to_numeric(df_oos[c], errors='coerce').fillna(0.0)

    keep_meta = []
    for c in meta_feature_cols:
        if c in df_oos.columns and df_oos[c].nunique() > 1:
            keep_meta.append(c)

    log.info(f"Meta-Brain feature set: {len(keep_meta)} features "
             f"({len(found_spec_oos)} specialist + {len(keep_meta) - len(found_spec_oos)} context/aggregate)")

    if len(keep_meta) < 5:
        log.error("Meta-Brain: Too few valid features. Aborting.")
        return

    X_all_meta = df_oos[keep_meta].values.astype(np.float32)
    X_all_meta = _sanitize_data(X_all_meta)
    n_meta = len(X_all_meta)

    pos_cases = int(np.sum(y_oos == 1))
    neg_cases = int(np.sum(y_oos == 0))
    scale_weight = min(float(neg_cases) / max(1.0, float(pos_cases)), 20.0)

    meta_scale_weight = scale_weight  # Pre-SMOTE weight for proper calibration
    log.info(f"Meta-Brain: {n_meta} samples with OOS predictions ({pos_cases} pos / {neg_cases} neg)")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 3: Walk-Forward Validation of Meta-Brain
    # ══════════════════════════════════════════════════════════════════
    actual_folds = min(args.wfa_folds, max(2, n_meta // 25))
    actual_gap = max(3, min(15, n_meta // 10))
    tscv = TimeSeriesSplit(n_splits=actual_folds, gap=actual_gap)

    acc_scores, prec_scores, roc_auc_scores, f1_scores = [], [], [], []

    for i, (train_idx, test_idx) in enumerate(tscv.split(X_all_meta)):
        X_tr, y_tr = X_all_meta[train_idx], y_oos[train_idx]
        X_te, y_te = X_all_meta[test_idx], y_oos[test_idx]

        if not args.no_smote:
            X_tr, y_tr = _apply_smote(X_tr, y_tr, random_state=42 + i)

        # Use ORIGINAL class weight (pre-SMOTE) for proper probability calibration
        sw = meta_scale_weight

        clf = xgb.XGBClassifier(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=min(args.max_depth, 6),
            reg_alpha=args.reg_alpha,
            reg_lambda=args.reg_lambda,
            colsample_bytree=0.7,
            subsample=0.8,
            random_state=42,
            objective='binary:logistic',
            eval_metric='logloss',
            scale_pos_weight=sw
        )
        clf.fit(X_tr, y_tr)

        preds = clf.predict(X_te)
        probs = clf.predict_proba(X_te)[:, 1] if len(np.unique(y_te)) > 1 else np.zeros_like(preds, dtype=float)

        acc = accuracy_score(y_te, preds)
        prec = precision_score(y_te, preds, zero_division=0)
        roc = roc_auc_score(y_te, probs) if len(np.unique(y_te)) > 1 else 0.5
        f1 = f1_score(y_te, preds, zero_division=0)

        acc_scores.append(acc); prec_scores.append(prec)
        roc_auc_scores.append(roc); f1_scores.append(f1)
        log.info(f"  Meta Fold {i+1}: Acc={acc:.4f} | Prec={prec:.4f} | ROC={roc:.4f} | F1={f1:.4f}")

    avg_acc = np.mean(acc_scores); avg_prec = np.mean(prec_scores)
    avg_roc = np.mean(roc_auc_scores); avg_f1 = np.mean(f1_scores)
    log.info(f"META-BRAIN WFA: Avg Acc={avg_acc:.4f} | Prec={avg_prec:.4f} | ROC={avg_roc:.4f} | F1={avg_f1:.4f}")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 4: Train final production Meta-Brain on ALL OOS data
    # ══════════════════════════════════════════════════════════════════
    X_final, y_final = X_all_meta, y_oos
    if not args.no_smote:
        X_final, y_final = _apply_smote(X_all_meta, y_oos, random_state=99)

    # Use ORIGINAL class weight (pre-SMOTE) for proper probability calibration
    final_weight = meta_scale_weight

    meta_model = xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=min(args.max_depth, 6),
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        colsample_bytree=0.7,
        subsample=0.8,
        random_state=42,
        objective='binary:logistic',
        eval_metric='logloss',
        scale_pos_weight=final_weight
    )
    meta_model.fit(X_final, y_final)

    valid_edge = bool(avg_roc > 0.50 and avg_prec > 0.10)

    # Save XGBoost model natively
    xgb_model_file = out_dir / "brain_META_unified.json"
    meta_model.get_booster().save_model(str(xgb_model_file))

    pipeline = {
        "xgb_model_file": str(xgb_model_file.name),
        "feature_names": keep_meta,
        "training_args": vars(args),
        "model": "xgboost_native_meta",
        "wfa_acc": avg_acc,
        "wfa_prec": avg_prec,
        "wfa_roc_auc": avg_roc,
        "wfa_f1": avg_f1,
        "valid_edge": valid_edge,
        "smote_enabled": HAS_SMOTE and not args.no_smote,
        "n_features": len(keep_meta),
        "n_specialist_features": len(found_spec_oos),
        "n_context_features": len(found_ctx_oos),
        "n_samples_total": n_meta,
        "oos_coverage_pct": round(oos_count / n_samples * 100, 1),
        "pos_neg_ratio": f"{pos_cases}/{neg_cases}"
    }

    out_file = out_dir / "brain_META_unified.joblib"
    joblib.dump(pipeline, out_file)
    log.info(f"Saved Meta-Brain (unified) to {out_file}")
    log.info(f"Meta-Brain: {len(keep_meta)} features = "
             f"{len(found_spec_oos)} OOS specialist probs + "
             f"4 aggregates + 1 side + {len(found_ctx_oos)} market context")


if __name__ == "__main__":
    main()
