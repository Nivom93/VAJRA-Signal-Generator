#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""vajra_feature_drift_check.py — detect train/serve feature-distribution drift.

Compares the per-feature distribution in an exported events JSONL (training
distribution) against an inference-time feature dump (backtest or live trace).

Catches issues like:
- A feature hardcoded to a constant in export but variable in inference (OOD)
- A feature whose mean shifts by >2 sigmas between train and inference
- A feature present in one path but absent in the other

Usage:
    python vajra_feature_drift_check.py \\
        --train-events events_export.jsonl.gz \\
        --infer-features backtest_inference_features.jsonl \\
        --sigma-threshold 2.0
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_jsonl(path: str) -> pd.DataFrame:
    p = Path(path)
    rows = []
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def compare(train_df: pd.DataFrame, infer_df: pd.DataFrame, sigma_threshold: float = 2.0):
    """Compare per-feature mean/std between training and inference."""
    train_cols = set(train_df.select_dtypes(include=[np.number]).columns)
    infer_cols = set(infer_df.select_dtypes(include=[np.number]).columns)

    only_train = train_cols - infer_cols
    only_infer = infer_cols - train_cols
    both = train_cols & infer_cols

    print("=" * 80)
    print("FEATURE DRIFT REPORT")
    print("=" * 80)

    if only_train:
        print(f"\n[WARN] Features in training but NOT in inference ({len(only_train)}):")
        for c in sorted(only_train)[:20]:
            print(f"  - {c}")

    if only_infer:
        print(f"\n[WARN] Features in inference but NOT in training ({len(only_infer)}):")
        for c in sorted(only_infer)[:20]:
            print(f"  - {c}")

    print(f"\n[INFO] Common features ({len(both)}); checking for distribution drift...\n")
    drifted = []
    constants_in_train = []

    for c in sorted(both):
        t = pd.to_numeric(train_df[c], errors="coerce").dropna()
        i = pd.to_numeric(infer_df[c], errors="coerce").dropna()
        if len(t) < 10 or len(i) < 10:
            continue

        t_mean, t_std = t.mean(), t.std()
        i_mean, i_std = i.mean(), i.std()

        # Constant in training but variable in inference = OOD risk
        if t_std < 1e-9 and i_std > 1e-6:
            constants_in_train.append((c, t_mean, i_mean, i_std))
            continue

        # Mean shift in sigmas (using training std as denominator)
        if t_std > 1e-9:
            shift_sigmas = abs(i_mean - t_mean) / t_std
            if shift_sigmas > sigma_threshold:
                drifted.append((c, t_mean, t_std, i_mean, i_std, shift_sigmas))

    if constants_in_train:
        print("[CRITICAL] Features CONSTANT in training but variable in inference:")
        print("           -> These features will produce out-of-distribution outputs.")
        print("           -> Either backfill them in training, or REMOVE from feature set.\n")
        print(f"  {'Feature':<30} {'Train val':>12} {'Infer mean':>12} {'Infer std':>12}")
        for c, tm, im, isd in constants_in_train:
            print(f"  {c:<30} {tm:>12.4f} {im:>12.4f} {isd:>12.4f}")
        print()

    if drifted:
        print(f"[WARN] Features with >{sigma_threshold}σ mean shift "
              f"({len(drifted)} features):")
        print(f"  {'Feature':<30} {'TrainMean':>10} {'TrainStd':>10} "
              f"{'InferMean':>10} {'InferStd':>10} {'σ-shift':>8}")
        for c, tm, ts, im, isd, sh in sorted(drifted, key=lambda x: -x[-1])[:30]:
            print(f"  {c:<30} {tm:>10.4f} {ts:>10.4f} {im:>10.4f} {isd:>10.4f} {sh:>8.2f}")
        print()

    if not constants_in_train and not drifted:
        print("[OK] No critical drift detected. Feature distributions match.")

    return constants_in_train, drifted


def main():
    p = argparse.ArgumentParser(
        description="Detect train/serve feature-distribution drift for VAJRA brains"
    )
    p.add_argument("--train-events", required=True,
                   help="JSONL exported by vajra_export_events.py (training distribution)")
    p.add_argument("--infer-features", required=True,
                   help="JSONL or CSV with inference-time feature snapshots")
    p.add_argument("--sigma-threshold", type=float, default=2.0,
                   help="Mean shift in sigmas to flag as drift (default 2.0)")
    args = p.parse_args()

    train_df = load_jsonl(args.train_events)
    if args.infer_features.endswith(".csv"):
        infer_df = pd.read_csv(args.infer_features)
    else:
        infer_df = load_jsonl(args.infer_features)

    constants, drifted = compare(train_df, infer_df, args.sigma_threshold)

    if constants:
        raise SystemExit(2)  # CI failure for OOD constants
    if drifted:
        raise SystemExit(1)  # warning for distribution shift


if __name__ == "__main__":
    main()
