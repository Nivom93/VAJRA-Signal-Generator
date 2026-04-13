#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""brain_audit.py — pre-flight audit of trained brains.

Reads the SAME tier/quality fields written by vajra_brain_train.py.
Does NOT recompute tier — single source of truth.

Usage:
    python brain_audit.py --brains-dir ./brains
    python brain_audit.py --brains-dir ./brains --trades-csv backtest_trades.csv
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd


# ==============================================================================
# 1. BRAIN AUDIT TABLE — reads quality_tier directly from trainer output
# ==============================================================================

def audit_brains(brains_dir: str) -> bool:
    p = Path(brains_dir)
    if not p.is_dir():
        print(f"ERROR: directory not found: {brains_dir}")
        return False

    print("-" * 90)
    print("VAJRA BRAIN PRE-FLIGHT AUDIT")
    print("-" * 90)
    print(f"{'Brain':<28} {'Tier':<10} {'ROC':>6} {'Prec':>6} {'F1':>6} "
          f"{'#Feat':>6} {'Pos%':>6} {'ThrAdj':>7} {'Deploy?':>8}")
    print("-" * 90)

    files = sorted(p.glob("brain_*.joblib"))
    deployable = 0
    total = 0
    no_deploy_strategies: set = set()

    for f in files:
        if "META" in f.stem or "calibrator" in f.stem:
            continue
        try:
            b = joblib.load(f)
            # Read quality_tier directly — NEVER recompute.
            # This is the single source of truth written by vajra_brain_train.py.
            tier = b.get("quality_tier", "unknown")
            roc = b.get("wfa_roc_auc", 0.0)
            prec = b.get("wfa_prec", 0.0)
            f1 = b.get("wfa_f1", 0.0)
            n_feat = b.get("n_features_selected", len(b.get("feature_names", [])))
            pos_rate = b.get("positive_class_rate", 0.0) * 100
            thr_adj = b.get("threshold_adj", 0.0)
            # Deploy if tier is strong or medium — matching the trainer's definition
            deploy = tier in ("strong", "medium")
            print(f"{f.stem:<28} {tier:<10} {roc:>6.3f} {prec:>6.3f} {f1:>6.3f} "
                  f"{n_feat:>6} {pos_rate:>5.1f}% {'+' + str(thr_adj):>7} "
                  f"{'YES' if deploy else 'NO':>8}")
            total += 1
            if deploy:
                deployable += 1
            else:
                # Extract strategy name (BOS, CHOCH, etc.)
                parts = f.stem.split("_")
                if len(parts) >= 2:
                    no_deploy_strategies.add(parts[1])
        except Exception as e:
            print(f"  ERROR reading {f.name}: {e}")

    print("-" * 90)
    pct = (100.0 * deployable / total) if total > 0 else 0.0
    print(f"\nSUMMARY: {deployable}/{total} brains deployable ({pct:.0f}%)")
    if no_deploy_strategies:
        print(f"STRATEGIES WITH NO DEPLOYABLE BRAIN: {', '.join(sorted(no_deploy_strategies))}")

    # CI gate
    if pct >= 50.0:
        print(f"\nCI GATE PASSED: {deployable}/{total} brains deployable ({pct:.0f}% >= 50%)")
        return True
    else:
        print(f"\nCI GATE FAILED: only {deployable}/{total} deployable ({pct:.0f}% < 50%)")
        return False


# ==============================================================================
# 2. FEATURE OVERLAP ANALYSIS
# ==============================================================================

def print_feature_overlap(brains_dir: str) -> None:
    """Analyse feature usage across brains and flag fragile/universal features."""
    p = Path(brains_dir)
    files = sorted(p.glob("brain_*.joblib"))
    brains = []
    for f in files:
        if "META" in f.stem or "calibrator" in f.stem:
            continue
        try:
            brains.append(joblib.load(f))
        except Exception:
            pass

    if not brains:
        return

    feature_counter: Counter = Counter()
    n_brains = len(brains)

    for b in brains:
        for feat in b.get("feature_names", []):
            feature_counter[feat] += 1

    fragile = sorted(f for f, c in feature_counter.items() if c == 1)
    universal = sorted(f for f, c in feature_counter.items() if c == n_brains)

    sep = "-" * 60
    print(f"\n{sep}")
    print("FEATURE OVERLAP ANALYSIS")
    print(sep)
    print(f"Total unique features across all brains: {len(feature_counter)}")
    print(f"Brains analysed: {n_brains}")

    if universal:
        print(f"\nUniversal features (used by ALL {n_brains} brains):")
        for f in universal:
            print(f"  + {f}")
    else:
        print("\nNo universal features found (none used by every brain).")

    if fragile:
        print(f"\nFragile features (used by only 1 brain):")
        for f in fragile:
            print(f"  ! {f}")
    else:
        print("\nNo fragile features (all features used by 2+ brains).")
    print(sep)


# ==============================================================================
# 3. CALIBRATION REPORT
# ==============================================================================

def calibration_report(trades_csv_path: str) -> None:
    """
    Read backtest trades CSV, group by strategy, compute ECE per strategy,
    and flag overconfident strategies.
    """
    csv_path = Path(trades_csv_path)
    if not csv_path.exists():
        print(f"\nERROR: Trades CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # Identify required columns — be flexible with naming
    prob_col = None
    for candidate in ("prob", "probability", "model_prob", "specialist_prob"):
        if candidate in df.columns:
            prob_col = candidate
            break

    outcome_col = None
    for candidate in ("outcome", "label", "win", "meta_label"):
        if candidate in df.columns:
            outcome_col = candidate
            break

    # Fallback: derive a binary _win column from pnl_r or exit_reason
    if outcome_col is None:
        if "pnl_r" in df.columns:
            df["_derived_win"] = (pd.to_numeric(df["pnl_r"], errors="coerce") > 0).astype(int)
            outcome_col = "_derived_win"
        elif "exit_reason" in df.columns:
            df["_derived_win"] = (df["exit_reason"].astype(str).str.lower() == "tp").astype(int)
            outcome_col = "_derived_win"

    strategy_col = None
    for candidate in ("strategy", "strat", "reason", "entry_kind"):
        if candidate in df.columns:
            strategy_col = candidate
            break

    if prob_col is None or outcome_col is None:
        print(f"\nERROR: Trades CSV must contain a probability column "
              f"(tried: prob, probability, model_prob, specialist_prob) "
              f"and an outcome column (tried: outcome, label, win, meta_label).")
        print(f"  Found columns: {list(df.columns)}")
        return

    # Convert outcome to binary if needed (positive pnl_r -> win)
    if outcome_col == "outcome":
        df["_win"] = df[outcome_col].apply(lambda x: 1 if str(x).lower() in ("1", "win", "true", "tp") else 0)
    else:
        df["_win"] = pd.to_numeric(df[outcome_col], errors="coerce").fillna(0).astype(int)

    df["_prob"] = pd.to_numeric(df[prob_col], errors="coerce")
    df = df.dropna(subset=["_prob"])

    if strategy_col is None:
        df["_strategy"] = "ALL"
        strategy_col = "_strategy"
    else:
        df["_strategy"] = df[strategy_col]

    sep = "-" * 72
    print(f"\n{sep}")
    print("CALIBRATION REPORT (Model Probability vs Actual Win Rate)")
    print(sep)

    strategies = sorted(df["_strategy"].unique())
    overconfident_flags = []

    for strat in strategies:
        sdf = df[df["_strategy"] == strat].copy()
        if len(sdf) < 10:
            print(f"\n  {strat}: too few trades ({len(sdf)}) for calibration analysis")
            continue

        sdf["_decile"] = pd.qcut(sdf["_prob"], q=10, duplicates="drop")

        print(f"\n  Strategy: {strat} (n={len(sdf)})")
        print(f"  {'Decile':<24} {'Avg Prob':>9} {'Win Rate':>9} {'Gap':>8} {'N':>5}")

        ece = 0.0
        n_total = len(sdf)
        max_overconf = 0.0

        for decile, group in sdf.groupby("_decile", observed=True):
            avg_prob = group["_prob"].mean()
            win_rate = group["_win"].mean()
            gap = avg_prob - win_rate
            n = len(group)
            ece += (n / n_total) * abs(gap)
            max_overconf = max(max_overconf, gap)
            print(f"  {str(decile):<24} {avg_prob:>8.1%} {win_rate:>8.1%} {gap:>+7.1%} {n:>5}")

        print(f"  ECE = {ece:.4f}")
        if max_overconf > 0.15:
            overconfident_flags.append((strat, max_overconf))
            print(f"  ** OVERCONFIDENT: model exceeds actual by {max_overconf:.1%} in worst decile")

    if overconfident_flags:
        print(f"\n{sep}")
        print("OVERCONFIDENCE WARNINGS (model prob > actual win rate by >15pp):")
        for strat, gap in overconfident_flags:
            print(f"  {strat}: worst decile gap = {gap:.1%}")
    else:
        print(f"\nNo strategies flagged for overconfidence (all decile gaps <= 15pp).")
    print(sep)


# ==============================================================================
# CLI ENTRYPOINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Vajra Brain Pre-Flight Audit — reads quality_tier from trainer (single source of truth)"
    )
    parser.add_argument(
        "--brains-dir", required=True,
        help="Path to directory containing brain_*.joblib files"
    )
    parser.add_argument(
        "--trades-csv", default=None,
        help="Path to backtest trades CSV for calibration report (optional)"
    )
    args = parser.parse_args()

    # 1. Audit table (reads quality_tier directly — no recomputation)
    ok = audit_brains(args.brains_dir)

    # 2. Feature overlap
    print_feature_overlap(args.brains_dir)

    # 3. Calibration report (if trades CSV provided)
    if args.trades_csv:
        calibration_report(args.trades_csv)

    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
