#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
brain_audit.py — Pre-flight Brain Quality Checker & Calibration Diagnostics
==========================================================================
Loads all specialist brain_*.joblib files, prints a formatted quality table,
checks feature overlap, and optionally computes calibration error from a
backtest trades CSV.

Usage:
    python brain_audit.py --brains-dir ./brains
    python brain_audit.py --brains-dir ./brains --trades-csv backtest_trades.csv
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

# ── Deployability thresholds ──
MIN_ROC = 0.52
MIN_PRECISION = 0.20
MIN_FEATURES = 8

# ── Protected features (must match vajra_brain_train.py) ──
PROTECTED_FEATURES = {
    "is_ttm_squeeze",
    "atr_expansion_rate",
    "cvd_acceleration",
    "displacement_bos_bull",
}


# ==============================================================================
# 1. BRAIN AUDIT TABLE
# ==============================================================================

def _tier_label(roc: float) -> str:
    if roc > 0.55:
        return "strong"
    elif roc > 0.50:
        return "medium"
    else:
        return "noise"


def _is_deployable(roc: float, precision: float, n_features: int) -> bool:
    return roc > MIN_ROC and precision > MIN_PRECISION and n_features >= MIN_FEATURES


def load_brains(brains_dir: Path) -> List[dict]:
    """Load all specialist brain joblib files, excluding META and calibrator files."""
    brains = []
    for f in sorted(brains_dir.glob("brain_*.joblib")):
        stem = f.stem
        parts = stem.split("_")
        # Skip META brains and calibrators
        if len(parts) >= 2 and parts[1] == "META":
            continue
        if "calibrator" in stem:
            continue
        try:
            data = joblib.load(str(f))
            data["_filename"] = f.name
            # Parse strategy and side from filename: brain_{STRATEGY}_{SIDE}.joblib
            if len(parts) >= 3:
                data["_strategy"] = parts[1]
                data["_side"] = parts[2]
            brains.append(data)
        except Exception as e:
            print(f"  WARNING: Failed to load {f.name}: {e}")
    return brains


def print_audit_table(brains: List[dict]) -> Tuple[int, int, List[str]]:
    """Print formatted audit table. Returns (total, deployable_count, strategies_with_no_deployable)."""
    header = (
        f"{'Brain':<28} {'Tier':<8} {'ROC':>6} {'Prec':>6} {'F1':>6} "
        f"{'#Feat':>6} {'Pos%':>7} {'ThrAdj':>7} {'Deploy?':>8}"
    )
    sep = "-" * len(header)

    print("\n" + sep)
    print("VAJRA BRAIN PRE-FLIGHT AUDIT")
    print(sep)
    print(header)
    print(sep)

    total = 0
    deployable = 0
    strategy_deploy_map: Dict[str, bool] = {}

    for b in brains:
        name = b.get("_filename", "unknown").replace(".joblib", "")
        roc = b.get("wfa_roc_auc", 0.0)
        prec = b.get("wfa_prec", 0.0)
        f1 = b.get("wfa_f1", 0.0)
        n_feat = b.get("n_features_selected", len(b.get("feature_names", [])))
        pos_rate = b.get("positive_class_rate", 0.0)
        thresh_adj = b.get("threshold_adj", 0.0)
        tier = _tier_label(roc)
        dep = _is_deployable(roc, prec, n_feat)

        strat = b.get("_strategy", "?")
        if strat not in strategy_deploy_map:
            strategy_deploy_map[strat] = False
        if dep:
            strategy_deploy_map[strat] = True
            deployable += 1
        total += 1

        dep_str = "YES" if dep else "NO"
        print(
            f"{name:<28} {tier:<8} {roc:>6.3f} {prec:>6.3f} {f1:>6.3f} "
            f"{n_feat:>6} {pos_rate:>6.1%} {thresh_adj:>+7.2f} {dep_str:>8}"
        )

    print(sep)

    no_deploy_strats = sorted(s for s, has in strategy_deploy_map.items() if not has)
    pct = (deployable / total * 100) if total else 0

    print(f"\nSUMMARY: {deployable}/{total} brains deployable ({pct:.0f}%)")
    if no_deploy_strats:
        print(f"STRATEGIES WITH NO DEPLOYABLE BRAIN: {', '.join(no_deploy_strats)}")
    else:
        print("All strategies have at least one deployable brain.")

    return total, deployable, no_deploy_strats


# ==============================================================================
# 2. FEATURE OVERLAP ANALYSIS
# ==============================================================================

def print_feature_overlap(brains: List[dict]) -> None:
    """Analyse feature usage across brains and flag fragile/universal features."""
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

    # Protected feature coverage check
    missing_threshold = n_brains / 2
    print(f"\nProtected feature coverage (warn if missing from >{int(missing_threshold)} brains):")
    any_warning = False
    for pf in PROTECTED_FEATURES:
        count = feature_counter.get(pf, 0)
        missing = n_brains - count
        status = "OK" if missing <= missing_threshold else "WARNING"
        if status == "WARNING":
            any_warning = True
        print(f"  {pf:<30} used by {count}/{n_brains} brains  [{status}]")
    if not any_warning:
        print("  All protected features have adequate coverage.")
    print(sep)


# ==============================================================================
# 3. CALIBRATION REPORT
# ==============================================================================

def calibration_report(trades_csv_path: str, brains_dir: str) -> None:
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

    # Convert outcome to binary if needed (positive pnl_r → win)
    if outcome_col == "outcome":
        df["_win"] = df[outcome_col].apply(lambda x: 1 if str(x).lower() in ("1", "win", "true", "tp") else 0)
    else:
        df["_win"] = pd.to_numeric(df[outcome_col], errors="coerce").fillna(0).astype(int)

    df["_prob"] = pd.to_numeric(df[prob_col], errors="coerce")
    df = df.dropna(subset=["_prob"])

    if strategy_col is None:
        # If no strategy column, treat all as one group
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

        # Create decile bins based on predicted probability
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
        description="Vajra Brain Pre-Flight Audit — checks brain quality before deployment"
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

    brains_dir = Path(args.brains_dir)
    if not brains_dir.is_dir():
        print(f"ERROR: brains directory not found: {brains_dir}")
        sys.exit(1)

    brains = load_brains(brains_dir)
    if not brains:
        print("ERROR: No specialist brains found in directory.")
        sys.exit(1)

    # 1. Audit table
    total, deployable, no_deploy_strats = print_audit_table(brains)

    # 2. Feature overlap
    print_feature_overlap(brains)

    # 3. Calibration report (if trades CSV provided)
    if args.trades_csv:
        calibration_report(args.trades_csv, args.brains_dir)

    # CI gate: exit 1 if fewer than 50% deployable
    if total > 0 and (deployable / total) < 0.50:
        print(f"\nCI GATE FAILED: only {deployable}/{total} brains deployable "
              f"({deployable / total:.0%} < 50% required)")
        sys.exit(1)
    else:
        print(f"\nCI GATE PASSED: {deployable}/{total} brains deployable "
              f"({deployable / total:.0%} >= 50%)")
        sys.exit(0)


if __name__ == "__main__":
    main()
