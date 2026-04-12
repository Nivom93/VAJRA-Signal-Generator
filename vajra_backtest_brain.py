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
    reset_p6_counters, log_p6_summary,
    _p6_counters
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

def _bar_imbalance_proxy(bar_o, bar_h, bar_l, bar_c, bar_v):
    """
    Approximate bid/ask imbalance from a single bar.
    Bullish bar with high volume -> high imbalance (bids dominant).
    Bearish bar with high volume -> low imbalance (asks dominant).
    Returns value in [0, 1].
    """
    bar_range = bar_h - bar_l
    if bar_range < 1e-9:
        return 0.5
    body = bar_c - bar_o
    body_pct = body / bar_range  # in [-1, 1]
    # Map [-1, 1] -> [0.2, 0.8] (avoid extremes — single-bar inference is noisy)
    imb = 0.5 + 0.3 * body_pct
    return float(np.clip(imb, 0.2, 0.8))

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
    tm.set_funding_lookup(exec_tf["timestamp"].values, funding_aligned)
    log.info(f"Loaded {len(funding_aligned)} funding rate samples into TradeManager")

    # BTC Dominance — use real dominance cache instead of BTCDOM perpetual proxy
    from vajra_macro_data import fetch_btc_dominance_series
    btcd_aligned = fetch_btc_dominance_series(
        start_ms=int(exec_tf["timestamp"].iloc[0]),
        end_ms=int(exec_tf["timestamp"].iloc[-1]),
        exec_timestamps=exec_tf["timestamp"],
    )

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
        # Session override (parity with _build_full_features) — timezone-correct
        try:
            from vajra_macro_data import compute_session_flags
            infer_d.update(compute_session_flags(int(ts)))
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
            "bid_ask_imbalance": _bar_imbalance_proxy(row.open, row.high, row.low, row.close,
                                                       row.volume if hasattr(row, 'volume') else 0),
            "macro_sentiment": 0.0
        }

        base = confluence_features(cfg, macro_tf, swing_tf, htf, exec_tf, max(0, iMacro-1), max(0, iSwing-1), max(0, iHtf-1), iExec, precomp=pre_map, extras=extras)
        # PARITY FIX: timestamp MUST be set before plan_trade_with_brain so
        # that _build_vec can recompute session flags (is_london_session,
        # is_ny_session) via pd.to_datetime — matching _build_full_features.
        base["timestamp"] = ts
        base["symbol"] = cfg.symbol

        # Emit a sample of inference-time feature snapshots for drift detection.
        # Only sample the bars where a plan would actually be considered (post-filter).
        if not hasattr(run_backtest_with_brain, '_drift_snapshot'):
            run_backtest_with_brain._drift_snapshot = []
            run_backtest_with_brain._drift_count = 0
        if run_backtest_with_brain._drift_count < 1000:
            snap = {**base}
            for k, v in adv_features.items():
                if hasattr(v, '__getitem__') and not isinstance(v, (str, bytes)):
                    try:
                        if len(v) > iExec:
                            val = v[iExec]
                            if hasattr(val, 'item'):
                                val = val.item()
                            snap[k] = float(val) if np.isfinite(val) else 0.0
                    except (TypeError, ValueError):
                        pass
            snap['timestamp'] = ts
            run_backtest_with_brain._drift_snapshot.append(snap)
            run_backtest_with_brain._drift_count += 1

        # Run parity diagnostic on first qualifying bar
        _run_parity_diagnostic(ts, iMacro, iSwing, iHtf, iExec, base, adv_features, pExec, brain)

        plan = plan_trade_with_brain(cfg, brain, base, adv_features, iExec, pExec)

        # SPOOFING DEFENSE PARITY: Apply the same spoofing rejection the live bot uses.
        # Without this, backtest takes trades that live would block, inflating expectations.
        if plan and getattr(cfg, 'spoofing_defense_enabled', True):
            bid_ask_imb = base.get('bid_ask_imbalance', 0.5)
            plan_type = plan.get('type', getattr(cfg, 'execution_style', 'limit'))
            if plan_type == 'limit':
                if plan['side'] == 'long' and bid_ask_imb < 0.25:
                    _p6_counters.setdefault("spoofing_rejected", 0)
                    _p6_counters["spoofing_rejected"] += 1
                    plan = None
                elif plan['side'] == 'short' and bid_ask_imb > 0.75:
                    _p6_counters.setdefault("spoofing_rejected", 0)
                    _p6_counters["spoofing_rejected"] += 1
                    plan = None

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

    # Write inference feature snapshots for drift detection
    if hasattr(run_backtest_with_brain, '_drift_snapshot') and run_backtest_with_brain._drift_snapshot:
        drift_path = "backtest_inference_features.jsonl"
        with open(drift_path, "w", encoding="utf-8") as f:
            for snap in run_backtest_with_brain._drift_snapshot:
                f.write(json.dumps({k: v for k, v in snap.items() if isinstance(v, (int, float, str, bool))}) + "\n")
        log.info(f"Wrote {len(run_backtest_with_brain._drift_snapshot)} inference feature snapshots to: {drift_path}")
        # reset for next call
        delattr(run_backtest_with_brain, '_drift_snapshot')
        delattr(run_backtest_with_brain, '_drift_count')

    return summary, all_closed

def strategy_concentration_report(trades_csv_path: str) -> None:
    """Read a backtest CSV and print per-strategy concentration diagnostics.

    Outputs:
        - Trade count per strategy
        - PnL (R) per strategy
        - Herfindahl-Hirschman Index (HHI) of trade-count concentration
        - Warning if any single strategy exceeds 25% of total trades
    """
    df = pd.read_csv(trades_csv_path)
    if "strategy" not in df.columns or "pnl_r" not in df.columns:
        print("ERROR: CSV must contain 'strategy' and 'pnl_r' columns.")
        return

    total_trades = len(df)
    if total_trades == 0:
        print("No trades found in CSV.")
        return

    trades_per = df.groupby("strategy").size().sort_values(ascending=False)
    pnl_per = df.groupby("strategy")["pnl_r"].sum().sort_values()

    print("=" * 64)
    print("STRATEGY CONCENTRATION REPORT")
    print("=" * 64)
    print(f"{'Strategy':<25} {'Trades':>8} {'%':>7} {'PnL (R)':>10}")
    print("-" * 64)
    for strat in trades_per.index:
        n = trades_per[strat]
        pct = 100.0 * n / total_trades
        pnl = pnl_per.get(strat, 0.0)
        print(f"{strat:<25} {n:>8d} {pct:>6.1f}% {pnl:>+10.2f}")
    print("-" * 64)
    print(f"{'TOTAL':<25} {total_trades:>8d} {'100.0%':>7} {df['pnl_r'].sum():>+10.2f}")

    # HHI: sum of squared market shares (0-10000 scale)
    shares = trades_per.values / total_trades
    hhi = float(np.sum(shares ** 2) * 10000)
    print(f"\nHerfindahl-Hirschman Index (HHI): {hhi:.0f}")
    if hhi > 2500:
        print("  >> HIGH concentration (HHI > 2500) — portfolio dominated by few strategies")
    elif hhi > 1500:
        print("  >> MODERATE concentration (HHI 1500-2500)")
    else:
        print("  >> LOW concentration (HHI < 1500)")

    # Single-strategy dominance warning
    for strat in trades_per.index:
        pct = trades_per[strat] / total_trades
        if pct > 0.25:
            print(f"\n  WARNING: {strat} accounts for {pct*100:.1f}% of trades (>{25}% threshold)")

    print("=" * 64)


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