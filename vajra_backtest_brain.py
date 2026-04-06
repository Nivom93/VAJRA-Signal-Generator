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
    macro_tf, swing_tf, htf, exec_tf = pre.macro_tf, pre.swing_tf, pre.htf, pre.exec_tf

    for df in (macro_tf, swing_tf, htf, exec_tf):
        df.sort_values("timestamp", inplace=True, ignore_index=True)

    btc = pre.btc if pre.btc is not None else macro_tf.copy()

    # --- CALC BTC TREND (Aligned) ---
    # Safe check in case btc is empty or incorrectly shaped
    if btc is not None and not btc.empty:
        btc.sort_values("timestamp", inplace=True, ignore_index=True)
        btc_c = btc["close"].values
    else:
        btc = pd.DataFrame({'timestamp': exec_tf['timestamp'], 'close': exec_tf['close']})
        btc_c = btc["close"].values

    # Use 100 EMA for 4h context
    btc_ema_trend = _ema_np(btc_c, 100)
    btc_bull_arr = (btc_c > btc_ema_trend).astype(float)

    # Align to exec_tf using searchsorted for reliable alignment (no shift/reindex gaps)
    btc_ts = btc["timestamp"].values
    exec_ts = exec_tf["timestamp"].values
    btc_idx = np.searchsorted(btc_ts, exec_ts, side='right') - 1
    # Use idx-1 to enforce lag (only fully closed BTC candles)
    btc_idx_lagged = np.clip(btc_idx - 1, 0, len(btc_bull_arr) - 1)
    btc_aligned = btc_bull_arr[btc_idx_lagged]

    # Load Brains (pass directory containing brain_*.joblib files)
    brain = BrainLearningManager(cfg, args.brain_long_path)
    
    log.info("Generating V7 FEATURES via Engine...")
    pMacro, pSwing, pHtf, pExec = Precomp(macro_tf), Precomp(swing_tf), Precomp(htf), Precomp(exec_tf)
    pre_map = {"macro_tf": pMacro, "swing_tf": pSwing, "htf": pHtf, "exec_tf": pExec}

    btc_val_arr = btc["close"].values
    btc_idx_close = np.searchsorted(btc_ts, exec_ts, side='right') - 1
    btc_idx_close_lagged = np.clip(btc_idx_close - 1, 0, len(btc_val_arr) - 1)
    btc_s_close = btc_val_arr[btc_idx_close_lagged]

    adv_features = precompute_v6_features(
        pMacro, pSwing, pExec, macro_tf, swing_tf, exec_tf,
        btc_close_arr=btc_s_close
    )
    
    tm = TradeManager(cfg, ExchangeWrapper(cfg), MemoryManager(cfg), brain)
    all_closed = []
    daily_pnl_r = 0.0
    daily_reset_ts = 0
    max_daily_loss_r = getattr(cfg, 'max_daily_loss_r', 5.0)
    MS_PER_DAY = 86_400_000

    since_ms = bt_helpers.parse_time(args.since)
    until_ms = bt_helpers.parse_time(args.until)
    exec_iter = exec_tf[(exec_tf["timestamp"] >= since_ms) & (exec_tf["timestamp"] < until_ms)]

    log.info("Downloading Macro Context for Backtester...")
    dxy_aligned = fetch_macro_trend("DX=F", exec_tf["timestamp"]) if getattr(cfg, 'use_macro_data', False) else np.zeros(len(exec_tf))
    spx_aligned = fetch_macro_trend("ES=F", exec_tf["timestamp"]) if getattr(cfg, 'use_macro_data', False) else np.zeros(len(exec_tf))
    oi_aligned = fetch_delta_oi(ExchangeWrapper(cfg), cfg.symbol, cfg.exec_tf, exec_tf["timestamp"]) if getattr(cfg, 'use_macro_data', False) else np.zeros(len(exec_tf))

    for row in exec_iter.itertuples():
        ts = int(row.timestamp)
        iMacro = np.searchsorted(macro_tf['timestamp'].values, ts, side='right') - 1
        iSwing = np.searchsorted(swing_tf['timestamp'].values, ts, side='right') - 1
        iHtf = np.searchsorted(htf['timestamp'].values, ts, side='right') - 1
        iExec = int(row.Index)

        if iMacro < 0 or iSwing < 0 or iHtf < 0: continue

        bar = {"o": row.open, "h": row.high, "l": row.low, "c": row.close}
        closed = tm.step_bar(cfg.symbol, float(row.open), float(row.high), float(row.low), float(row.close), ts=ts)
        if closed:
            all_closed.extend(closed)
            for t in closed:
                daily_pnl_r += t.get('pnl_r', 0.0)

        # Reset daily PnL counter at day boundary
        if daily_reset_ts == 0:
            daily_reset_ts = ts
        if ts - daily_reset_ts >= MS_PER_DAY:
            daily_pnl_r = 0.0
            daily_reset_ts = ts

        # Daily loss circuit breaker
        if daily_pnl_r <= -max_daily_loss_r:
            continue

        btc_val = btc_aligned[iExec] if iExec < len(btc_aligned) else 1.0

        extras = {
            "btc_bullish": btc_val,
            "funding_rate": 0.0,
            "btcd_trend": 0.0,
            "dxy_trend": float(dxy_aligned[iExec]) if getattr(cfg, 'use_macro_data', False) and iExec < len(dxy_aligned) else 0.0,
            "spx_trend": float(spx_aligned[iExec]) if getattr(cfg, 'use_macro_data', False) and iExec < len(spx_aligned) else 0.0,
            "delta_oi": float(oi_aligned[iExec]) if getattr(cfg, 'use_macro_data', False) and iExec < len(oi_aligned) else 0.0,
            "bid_ask_imbalance": 0.5,
            "macro_sentiment": 0.0
        }

        base = confluence_features(cfg, macro_tf, swing_tf, htf, exec_tf, max(0, iMacro-1), max(0, iSwing-1), max(0, iHtf-1), iExec, precomp=pre_map, extras=extras)
        base["timestamp"] = ts
        base["symbol"] = cfg.symbol

        plan = plan_trade_with_brain(cfg, brain, base, adv_features, iExec, pExec)
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
                df_export['fill_date'] = pd.to_datetime(df_export['fill_ts'], unit='ms')
            if 'exit_ts' in df_export.columns:
                df_export['exit_date'] = pd.to_datetime(df_export['exit_ts'], unit='ms')

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