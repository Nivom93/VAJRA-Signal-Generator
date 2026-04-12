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
from vajra_export_events import fetch_macro_trend, fetch_delta_oi, fetch_historical_funding_rates

# IMPORT FROM ENGINE
from vajra_engine_ultra_v6_final import (
    AadhiraayanEngineConfig as EngineConfig,
    ExchangeWrapper, MemoryManager, TradeManager, Precomp,
    confluence_features, plan_trade_with_brain,
    precompute_v6_features,
    BrainLearningManager,
    _ema_np,
    reset_p6_counters, log_p6_summary
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
    reset_p6_counters()
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
    brain = BrainLearningManager(cfg, args.brains_dir)
    
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

    # Fetch ALL macro context unconditionally — must match export pipeline for feature parity.
    # The export (training data) always fetches these; if backtest doesn't, the model sees
    # all-zeros for features it trained on real data, producing ~3% probability output.
    log.info("Downloading Macro Context for Backtester...")
    exw_bt = ExchangeWrapper(cfg)
    dxy_aligned = fetch_macro_trend("DX=F", exec_tf["timestamp"])
    spx_aligned = fetch_macro_trend("ES=F", exec_tf["timestamp"])
    oi_aligned = fetch_delta_oi(exw_bt, cfg.symbol, cfg.exec_tf, exec_tf["timestamp"])
    funding_aligned = fetch_historical_funding_rates(exw_bt, cfg.symbol, exec_tf["timestamp"])

    # BTC Dominance trend (same logic as export pipeline)
    try:
        import ccxt
        binance_ex = ccxt.binance({'options': {'defaultType': 'swap'}})
        btcd_raw = binance_ex.fetch_ohlcv("BTCDOM/USDT:USDT", timeframe="4h", limit=1000)
        btcd_df = pd.DataFrame(btcd_raw, columns=["timestamp","open","high","low","close","volume"])
        if not btcd_df.empty:
            btcd_c = btcd_df['close'].values
            btcd_trend = np.zeros_like(btcd_c)
            for i in range(5, len(btcd_c)):
                slope = (btcd_c[i] - btcd_c[i-5]) / btcd_c[i-5] * 100.0
                if slope > 0.5: btcd_trend[i] = 1.0
                elif slope < -0.5: btcd_trend[i] = -1.0
            btcd_series = pd.Series(btcd_trend, index=pd.to_datetime(btcd_df['timestamp'], unit='ms', utc=True))
            btcd_aligned = btcd_series.shift(1).reindex(pd.to_datetime(exec_tf["timestamp"], unit='ms', utc=True), method='ffill').fillna(0.0).values
        else:
            btcd_aligned = np.zeros(len(exec_tf))
    except Exception as e:
        log.warning(f"Failed to fetch BTCDOM: {e}")
        btcd_aligned = np.zeros(len(exec_tf))

    # ── DIAGNOSTIC: Feature parity assertion block ──────────────────────
    # On the first qualifying bar, build the feature dict via BOTH the
    # export path (_build_full_features) and the inference path
    # (_build_vec), then log a side-by-side comparison for the 6
    # historically-zero features.  This catches training/inference drift.
    _PARITY_TARGET_FEATURES = [
        "atr_pct_htf_aligned", "bars_since_ob_bull", "is_london_session",
        "is_ttm_squeeze", "macro_struct_strength", "rel_strength_divergence",
    ]
    _parity_checked = False

    def _run_parity_diagnostic(ts, iMacro, iSwing, iHtf, iExec, base, adv_features, pExec, brain):
        """Log side-by-side: export-path value vs inference-path value."""
        nonlocal _parity_checked
        if _parity_checked:
            return
        _parity_checked = True

        # --- Export-path reconstruction (mirrors _build_full_features) ---
        from vajra_export_events import _build_full_features
        export_feats = _build_full_features(
            base, adv_features, iMacro, iSwing, iHtf, iExec,
            len(macro_tf), len(swing_tf), len(htf), len(exec_tf), ts, pExec
        )

        # --- Inference-path reconstruction (mirrors _build_vec) ---
        infer_d = {**base}
        for k, v in adv_features.items():
            if hasattr(v, '__getitem__') and not isinstance(v, (str, bytes)):
                try:
                    vlen = len(v)
                except TypeError:
                    infer_d[k] = float(v) if np.isfinite(float(v)) else 0.0
                    continue
                if vlen > iExec:
                    val = v[iExec]
                    infer_d[k] = float(val) if np.isfinite(val) else 0.0
                else:
                    if k not in base:
                        infer_d[k] = 0.0
            elif isinstance(v, (int, float, np.integer, np.floating)):
                infer_d[k] = float(v) if np.isfinite(float(v)) else 0.0
            else:
                if k not in base:
                    infer_d[k] = 0.0
        # Session override (parity with _build_full_features)
        try:
            _dt = pd.to_datetime(int(ts), unit="ms", utc=True)
            infer_d["is_london_session"] = 1.0 if 7 <= _dt.hour <= 16 else 0.0
            infer_d["is_ny_session"] = 1.0 if 13 <= _dt.hour <= 22 else 0.0
        except Exception:
            pass

        log.info("=" * 72)
        log.info("FEATURE PARITY DIAGNOSTIC (bar ts=%d, iExec=%d)", ts, iExec)
        log.info("%-30s  %12s  %12s  %s", "Feature", "Export", "Inference", "Match?")
        log.info("-" * 72)
        any_mismatch = False
        for feat in _PARITY_TARGET_FEATURES:
            exp_val = export_feats.get(feat, "MISSING")
            inf_val = infer_d.get(feat, "MISSING")
            if isinstance(exp_val, float) and isinstance(inf_val, float):
                match = "OK" if abs(exp_val - inf_val) < 1e-9 else "MISMATCH"
            else:
                match = "OK" if exp_val == inf_val else "MISMATCH"
            if match != "OK":
                any_mismatch = True
            log.info("%-30s  %12s  %12s  %s", feat,
                     f"{exp_val:.6f}" if isinstance(exp_val, float) else str(exp_val),
                     f"{inf_val:.6f}" if isinstance(inf_val, float) else str(inf_val),
                     match)
        if not any_mismatch:
            log.info("ALL 6 TARGET FEATURES MATCH between export and inference paths.")
        else:
            log.warning("PARITY VIOLATION detected — see MISMATCH rows above.")
        log.info("=" * 72)

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
            "funding_rate": float(funding_aligned[iExec]) if iExec < len(funding_aligned) else 0.0,
            "btcd_trend": float(btcd_aligned[iExec]) if iExec < len(btcd_aligned) else 0.0,
            "dxy_trend": float(dxy_aligned[iExec]) if iExec < len(dxy_aligned) else 0.0,
            "spx_trend": float(spx_aligned[iExec]) if iExec < len(spx_aligned) else 0.0,
            "delta_oi": float(oi_aligned[iExec]) if iExec < len(oi_aligned) else 0.0,
            "bid_ask_imbalance": 0.5,
            "macro_sentiment": 0.0
        }

        base = confluence_features(cfg, macro_tf, swing_tf, htf, exec_tf, max(0, iMacro-1), max(0, iSwing-1), max(0, iHtf-1), iExec, precomp=pre_map, extras=extras)
        # PARITY FIX: timestamp MUST be set before plan_trade_with_brain so
        # that _build_vec can recompute session flags (is_london_session,
        # is_ny_session) via pd.to_datetime — matching _build_full_features.
        base["timestamp"] = ts
        base["symbol"] = cfg.symbol

        # Run parity diagnostic on first qualifying bar
        _run_parity_diagnostic(ts, iMacro, iSwing, iHtf, iExec, base, adv_features, pExec, brain)

        plan = plan_trade_with_brain(cfg, brain, base, adv_features, iExec, pExec)
        if plan: tm.submit_plan(plan, bar)

    dur = time.perf_counter() - start_t
    log_p6_summary()
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
    p.add_argument("--brain-long-path", required=False, default=None, help="(Deprecated) Use --brains-dir instead")
    p.add_argument("--brain-short-path", required=False, default=None, help="(Deprecated) Use --brains-dir instead")
    p.add_argument("--min-prob-long", type=float, default=-1.0)
    p.add_argument("--min-prob-short", type=float, default=-1.0)
    args = p.parse_args()
    
    summary, trades = run_backtest_with_brain(args)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()