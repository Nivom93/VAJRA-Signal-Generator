#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vajra_backtest_brain.py — Centralized Brain Backtester
=====================================================
FIX (v11.8 - LEVEL 30 PARITY):
- FEATURE PARITY: Added missing bid_ask_imbalance and btcd_trend stubs to avoid Spoofing Defense rejections.
"""
from __future__ import annotations

import argparse, time, json, logging
from pathlib import Path
import pandas as pd
import numpy as np
from vajra_export_events import fetch_macro_trend, fetch_delta_oi

# IMPORT FROM ENGINE
from vajra_engine_ultra_v6_final import (
    AadhiraayanEngineConfig as EngineConfig,
    ExchangeWrapper, MemoryManager, TradeManager, Precomp,
    confluence_features, plan_trade_with_brain,
    precompute_v6_features,
    BrainLearningManager,
    _ema_np
)
import vajra_backtest_optimized as bt_helpers

try:
    from vajra_overrides import _strategy_overrides
except ImportError:
    def _strategy_overrides(cfg): pass

log = logging.getLogger("vajra.bt.brain")
if not log.handlers:
    log.setLevel(logging.INFO)
    log.addHandler(logging.StreamHandler())

def run_backtest_with_brain(args, preloaded=None):
    start_t = time.perf_counter()
    cfg = bt_helpers.args_to_cfg(args)
    
    # 1. Apply Strategy File Overrides
    try: _strategy_overrides(cfg)
    except Exception as e: log.warning(f"Failed to apply overrides: {e}")

    # 2. Apply CLI args
    if args.min_prob_long > 0: 
        cfg.min_prob_long = args.min_prob_long
    if args.min_prob_short > 0:
        cfg.min_prob_short = args.min_prob_short
        
    cfg.symbol = args.symbol 
    
    log.info(f"Strategy Loaded: {cfg.symbol} | Long Gate: {cfg.min_prob_long:.2f} | Short Gate: {cfg.min_prob_short:.2f}")
    
    # Load Data (Including BTC Context)
    if preloaded is None: pre = bt_helpers._preload_or_fetch(args)
    else: pre = preloaded
    htf, mtf, ltf = pre.htf, pre.mtf, pre.ltf
    btc = pre.btc if pre.btc is not None else htf.copy()

    # --- CALC BTC TREND (Aligned) ---
    # Safe check in case btc is empty or incorrectly shaped
    if btc is not None and not btc.empty:
        btc.sort_values("timestamp", inplace=True, ignore_index=True)
        btc_c = btc["close"].values
    else:
        btc = pd.DataFrame({'timestamp': ltf['timestamp'], 'close': ltf['close']})
        btc_c = btc["close"].values
    
    # Use 100 EMA for 4h context
    btc_ema_trend = _ema_np(btc_c, 100)
    btc_bull_arr = (btc_c > btc_ema_trend).astype(float)
    
    # Align to LTF (Shift 1 for Lag)
    btc_s = pd.Series(btc_bull_arr, index=btc["timestamp"]).shift(1)
    btc_aligned = btc_s.reindex(ltf["timestamp"], method='ffill').fillna(1.0).values

    # Load Brains
    brain = BrainLearningManager(cfg, args.brain_long_path, args.brain_short_path)
    
    log.info("Generating V7 FEATURES via Engine...")
    pre_h, pre_m, pre_l = Precomp(htf), Precomp(mtf), Precomp(ltf)
    pre_map = {"htf": pre_h, "mtf": pre_m, "ltf": pre_l}
    
    btc_val_arr = btc["close"].values
    btc_s_close = pd.Series(btc_val_arr, index=btc["timestamp"]).shift(1).reindex(ltf["timestamp"], method='ffill').bfill().values
    
    adv_features = precompute_v6_features(
        pre_h, pre_m, pre_l, htf, mtf, ltf, 
        btc_close_arr=btc_s_close
    )
    
    tm = TradeManager(cfg, ExchangeWrapper(cfg), MemoryManager(cfg), brain)
    all_closed = []
    
    since_ms = bt_helpers.parse_time(args.since)
    until_ms = bt_helpers.parse_time(args.until)
    ltf_iter = ltf[(ltf["timestamp"] >= since_ms) & (ltf["timestamp"] < until_ms)]

    log.info("Downloading Macro Context for Backtester...")
    dxy_aligned = fetch_macro_trend("DX-Y.NYB", ltf["timestamp"]) if getattr(cfg, 'use_macro_data', False) else np.zeros(len(ltf))
    spx_aligned = fetch_macro_trend("^GSPC", ltf["timestamp"]) if getattr(cfg, 'use_macro_data', False) else np.zeros(len(ltf))
    oi_aligned = fetch_delta_oi(ExchangeWrapper(cfg), cfg.symbol, cfg.ltf, ltf["timestamp"]) if getattr(cfg, 'use_macro_data', False) else np.zeros(len(ltf))

    for original_idx, row in ltf_iter.iterrows():
        ts = int(row["timestamp"])
        iH = np.searchsorted(htf['timestamp'].values, ts, side='right') - 1
        iM = np.searchsorted(mtf['timestamp'].values, ts, side='right') - 1
        iL = int(original_idx)

        if iH < 0 or iM < 0: continue

        bar = {"o": row["open"], "h": row["high"], "l": row["low"], "c": row["close"]}
        closed = tm.step_bar(float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"]), ts=ts)
        all_closed.extend(closed)

        # Inject Context (Regime Filter & Macro)
        btc_val = btc_aligned[iL] if iL < len(btc_aligned) else 1.0
        
        extras = {
            "btc_bullish": btc_val, 
            "funding_rate": 0.0,
            "btcd_trend": 0.0,
            "dxy_trend": float(dxy_aligned[iL]) if getattr(cfg, 'use_macro_data', False) and iL < len(dxy_aligned) else 0.0,
            "spx_trend": float(spx_aligned[iL]) if getattr(cfg, 'use_macro_data', False) and iL < len(spx_aligned) else 0.0,
            "delta_oi": float(oi_aligned[iL]) if getattr(cfg, 'use_macro_data', False) and iL < len(oi_aligned) else 0.0,
            "bid_ask_imbalance": 0.5,
            "macro_sentiment": 0.0
        }

        base = confluence_features(cfg, htf, mtf, ltf, iH, iM, iL, pre_map, extras=extras)
        base["timestamp"] = ts 

        plan = plan_trade_with_brain(cfg, brain, base, adv_features, iH, iM, iL, pre_l)
        if plan: tm.submit_plan(plan, bar)

    dur = time.perf_counter() - start_t
    wins = sum(1 for t in all_closed if t["pnl_r"] > 0)
    summary = {
        "trades": len(all_closed),
        "winrate": (wins/len(all_closed)) if all_closed else 0.0,
        "total_r": sum(t["pnl_r"] for t in all_closed),
        "net_profit_est": sum(t["pnl_r"] for t in all_closed) * cfg.account_notional * cfg.risk_per_trade,
        "duration": dur
    }

    # --- EXCEL/CSV DETAILED TRADE REPORT ---
    if all_closed:
        try:
            df_export = pd.DataFrame(all_closed)
            if 'created_ts' in df_export.columns:
                df_export['created_date'] = pd.to_datetime(df_export['created_ts'], unit='ms')
            if 'fill_ts' in df_export.columns:
                df_export['fill_date'] = pd.to_datetime(df_export['fill_ts'], unit='s')
            if 'exit_ts' in df_export.columns:
                df_export['exit_date'] = pd.to_datetime(df_export['exit_ts'], unit='s')

            desired_cols = [
                'fill_date', 'exit_date', 'side', 'strategy', 'prob', 'rr',
                'entry', 'avg_price', 'exit_price', 'sl', 'tp',
                'pnl_r', 'exit_reason', 'bars_open', 'initial_risk_unit'
            ]
            export_cols = [c for c in desired_cols if c in df_export.columns]

            report_path = "backtest_trades_report.csv"
            df_export[export_cols].to_csv(report_path, index=False)
            log.info(f"📊 Saved detailed trade report to: {report_path}")
        except Exception as e:
            log.error(f"Failed to export CSV trade report: {e}")

    return summary, all_closed

def main():
    p = bt_helpers.build_arg_parser()
    p.add_argument("--brain-long-path", required=True)
    p.add_argument("--brain-short-path", required=True)
    p.add_argument("--min-prob-long", type=float, default=-1.0)
    p.add_argument("--min-prob-short", type=float, default=-1.0)
    args = p.parse_args()
    
    summary, trades = run_backtest_with_brain(args)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()