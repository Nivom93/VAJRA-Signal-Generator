#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vajra_backtest_optimized.py — Pointer Slicer + Precomputed Indicators
====================================================================
UPGRADE (v11.8 - REALITY CHECK v34.0 & V6 PARITY):
- FEATURE PARITY: Uses precompute_v6_features and plan_trade_with_brain for true historical testing without brain models.
- QUADRATIC IMPACT: Syncs with Engine's quadratic slippage logic.
- DATA FETCH: BTC Context is strictly hardcoded to '4h'.
"""
from __future__ import annotations

import argparse, os, time, json, logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

try:
    import pandas as pd
    import numpy as np
except Exception:
    pd = None
    np = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from vajra_engine_ultra_v6_final import (
    AadhiraayanEngineConfig as EngineConfig,
    BrainLearningManager, 
    ExchangeWrapper,
    MemoryManager,
    TradeManager,
    Precomp,
    confluence_features,
    precompute_v6_features,
    plan_trade_with_brain,
    _ema_np
)

try:
    from vajra_overrides import _strategy_overrides
except ImportError:
    def _strategy_overrides(cfg): pass

log = logging.getLogger("vajra.bt")
if not log.handlers:
    log.setLevel(logging.INFO)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
    log.addHandler(h)

_MS_PER_UNIT = {"m": 60_000, "h": 3_600_000, "d": 86_400_000, "w": 7 * 86_400_000}

def timeframe_to_ms(tf: str) -> int:
    tf = tf.strip().lower()
    if tf.endswith("ms"):
        return int(tf[:-2])
    num = int("".join(ch for ch in tf if ch.isdigit()))
    unit = "".join(ch for ch in tf if ch.isalpha())
    if unit not in _MS_PER_UNIT:
        raise ValueError(f"Unsupported timeframe: {tf}")
    return num * _MS_PER_UNIT[unit]

def parse_time(s: str) -> int:
    s = str(s).strip()
    if s.isdigit():
        v = int(s)
        return v if v > 10_000_000 else v * 1000  # allow seconds
    from datetime import datetime, timezone
    dt = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)

def _ensure_ms_int(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        raise ValueError("OHLCV df missing 'timestamp'")
    ts = df["timestamp"]
    if pd.api.types.is_numeric_dtype(ts):
        df["timestamp"] = pd.to_numeric(ts, errors="coerce").astype("int64")
        return df
    if pd.api.types.is_datetime64_any_dtype(ts):
        df["timestamp"] = (ts.view("int64") // 1_000_000).astype("int64")
        return df
    df["timestamp"] = pd.to_datetime(ts, utc=True, errors="coerce").view("int64") // 1_000_000
    df["timestamp"] = df["timestamp"].astype("int64")
    return df

# --------- IO ---------

def _fetch_ohlcv_paged(exw: ExchangeWrapper, symbol: str, timeframe: str, since_ms: int, until_ms: int) -> pd.DataFrame:
    tf_ms = timeframe_to_ms(timeframe)
    frames: List[pd.DataFrame] = []
    cur = since_ms; last_guard = 0
    page = 0
    while cur < until_ms:
        # Rate-limit: Bybit allows ~5 req/s on public endpoints.
        # Sleep every request to stay safely under the limit.
        if page > 0:
            time.sleep(0.25)
        page += 1
        try:
            df = exw.fetch_ohlcv_df(symbol, timeframe, limit=1000, since=cur)
        except Exception as e:
            err_str = str(e).lower()
            if "rate" in err_str or "10006" in err_str or "too many" in err_str:
                log.warning(f"Rate limited on page {page}, backing off 5s...")
                time.sleep(5)
                continue  # retry same cursor
            raise
        if df is None or len(df) == 0:
            break
        df = _ensure_ms_int(df)
        df = df[(df["timestamp"] >= since_ms) & (df["timestamp"] < until_ms)]
        if len(df):
            frames.append(df)
            new_last = int(df["timestamp"].iloc[-1])
            if new_last <= last_guard:
                cur = last_guard + tf_ms
            else:
                cur = new_last + tf_ms
            last_guard = new_last
        else:
            cur += tf_ms
    if not frames:
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])
    out = pd.concat(frames, ignore_index=True)
    out.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
    out.sort_values("timestamp", inplace=True, ignore_index=True)
    return out

@dataclass
class Preloaded:
    macro_tf: pd.DataFrame
    swing_tf: pd.DataFrame
    htf: pd.DataFrame
    exec_tf: pd.DataFrame
    btc: Optional[pd.DataFrame] = None

def _preload_or_fetch(args) -> Preloaded:
    if pd is None:
        raise ImportError("pandas is required for backtesting.")
    cache_dir = Path(getattr(args, "cache_dir", "./data_cache"))
    use_cache = bool(getattr(args, "use_cache", True))
    cache_dir.mkdir(parents=True, exist_ok=True)

    cfg = args_to_cfg(args)
    exw = ExchangeWrapper(cfg)

    since_ms = parse_time(args.since)
    until_ms = parse_time(args.until)

    def one(sym: str, tf: str) -> pd.DataFrame:
        fname = cache_dir / f"{args.exchange_id}_{sym.replace('/','-')}_{tf}_{since_ms}_{until_ms}.csv"
        if use_cache and fname.exists():
            df = pd.read_csv(fname)
            df = _ensure_ms_int(df)
            for c in ("open","high","low","close","volume"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df.dropna(inplace=True)
            return df
        df = _fetch_ohlcv_paged(exw, sym, tf, since_ms, until_ms)
        if use_cache and not df.empty:
            df.to_csv(fname, index=False)
        return df

    macro_df = one(args.symbol, args.macro_tf)
    swing_df = one(args.symbol, args.swing_tf)
    htf_df = one(args.symbol, args.htf)
    exec_df = one(args.symbol, args.exec_tf)
    
    # FETCH BTC Context (Regime Filter)
    btc_df = None
    if args.symbol != "BTC/USDT":
        log.info("Fetching BTC/USDT (4h) context data...")
        btc_df = one("BTC/USDT", "4h") 
    else:
        btc_df = one("BTC/USDT", "4h") 

    return Preloaded(macro_tf=macro_df, swing_tf=swing_df, htf=htf_df, exec_tf=exec_df, btc=btc_df)

# --------- Config mapping ---------

def args_to_cfg(args) -> EngineConfig:
    cfg = EngineConfig()
    cfg.exchange_id = args.exchange_id
    cfg.market_type = args.market_type
    cfg.symbol = args.symbol
    cfg.macro_tf = args.macro_tf
    cfg.swing_tf = args.swing_tf
    cfg.htf = args.htf
    cfg.exec_tf = args.exec_tf
    for k in ["min_rr","rr","atr_mult_sl","atr_mult_tp","scalper_rr","risk_per_trade",
              "min_conf_long","min_conf_short","min_prob","max_concurrent",
              "max_concurrent_buy","max_concurrent_sell"]:
        if hasattr(args, k) and getattr(args, k) is not None:
            setattr(cfg, k, getattr(args, k))
    cfg.account_notional = getattr(args, "account_notional", 1000.0)
    return cfg

# --------- Backtest ---------

def run_backtest(args, preloaded: Optional[Preloaded]=None, markets_data=None):
    if pd is None:
        raise ImportError("pandas is required for backtesting.")
    start_t = time.perf_counter()

    cfg = args_to_cfg(args)
    exw = ExchangeWrapper(cfg, markets_data=markets_data)

    try:
        _strategy_overrides(cfg)
        log.info("Applied strategy overrides to Backtester.")
    except Exception as e:
        log.warning(f"Failed to apply overrides: {e}")
    if hasattr(args, 'rr') and args.rr is not None:
        cfg.rr = args.rr

    if preloaded is None:
        pre = _preload_or_fetch(args)
        macro_tf, swing_tf, htf, exec_tf, btc = pre.macro_tf, pre.swing_tf, pre.htf, pre.exec_tf, pre.btc
    else:
        macro_tf, swing_tf, htf, exec_tf = preloaded.macro_tf.copy(), preloaded.swing_tf.copy(), preloaded.htf.copy(), preloaded.exec_tf.copy()
        btc = preloaded.btc.copy() if preloaded.btc is not None else macro_tf.copy()

    # Calculate BTC Trend
    btc.sort_values("timestamp", inplace=True, ignore_index=True)
    btc_c = btc["close"].values
    
    btc_ema_trend = _ema_np(btc_c, 100)
    btc_bull_arr = (btc_c > btc_ema_trend).astype(float)
    
    # Align BTC trend to exec_tf using searchsorted for reliable alignment
    btc_ts = btc["timestamp"].values
    exec_ts_arr = exec_tf["timestamp"].values
    btc_idx = np.searchsorted(btc_ts, exec_ts_arr, side='right') - 1
    # Use idx-1 to enforce lag (only fully closed BTC candles)
    btc_idx_lagged = np.clip(btc_idx - 1, 0, len(btc_bull_arr) - 1)
    btc_aligned = btc_bull_arr[btc_idx_lagged]

    # Align BTC Close for relative strength features
    btc_idx_close_lagged = np.clip(btc_idx - 1, 0, len(btc_c) - 1)
    btc_s_close = btc_c[btc_idx_close_lagged]

    brain = BrainLearningManager(cfg, getattr(args, 'brains_dir', None))
    mem = MemoryManager(cfg, db=None)
    tm = TradeManager(cfg, exw, mem, brain, narrative_map=None)

    for df in (macro_tf, swing_tf, htf, exec_tf):
        df.sort_values("timestamp", inplace=True, ignore_index=True)

    since_ms = parse_time(args.since)
    until_ms = parse_time(args.until)
    exec_iter = exec_tf[(exec_tf["timestamp"] >= since_ms) & (exec_tf["timestamp"] < until_ms)]

    log.info("Downloading Macro Context for Backtester...")
    from vajra_export_events import fetch_macro_trend, fetch_delta_oi
    dxy_aligned = fetch_macro_trend("DX=F", exec_tf["timestamp"])
    spx_aligned = fetch_macro_trend("^GSPC", exec_tf["timestamp"])
    oi_aligned = fetch_delta_oi(exw, cfg.symbol, cfg.exec_tf, exec_tf["timestamp"])

    log.info("Generating Engine V7 Precomputed features for base engine parity...")
    pMacro = Precomp(macro_tf)
    pSwing = Precomp(swing_tf)
    pHtf = Precomp(htf)
    pExec = Precomp(exec_tf)
    pre_map = {"macro_tf": pMacro, "swing_tf": pSwing, "htf": pHtf, "exec_tf": pExec}

    adv_features = precompute_v6_features(
        pMacro, pSwing, pExec, macro_tf, swing_tf, exec_tf,
        btc_close_arr=btc_s_close
    )

    iMacro = 0; iSwing = 0; iHtf = 0
    all_closed: List[Dict[str, Any]] = []
    daily_pnl_r = 0.0
    daily_reset_ts = since_ms
    max_daily_loss_r = getattr(cfg, 'max_daily_loss_r', 5.0)
    MS_PER_DAY = 86_400_000
    
    # OPTIMIZED PROGRESS BAR (Less I/O)
    progress = None
    if tqdm and getattr(args, "progress", "auto") != "off":
        progress = tqdm(total=len(exec_iter), desc="Backtest", mininterval=1.0)

    for rowL in exec_iter.itertuples():
        ts = int(rowL.timestamp)

        while iMacro+1 < len(macro_tf) and int(macro_tf["timestamp"].iloc[iMacro+1]) <= ts: iMacro += 1
        while iSwing+1 < len(swing_tf) and int(swing_tf["timestamp"].iloc[iSwing+1]) <= ts: iSwing += 1
        while iHtf+1 < len(htf) and int(htf["timestamp"].iloc[iHtf+1]) <= ts: iHtf += 1
        iExec = int(exec_tf.index.get_loc(rowL.Index))

        closed = tm.step_bar(cfg.symbol, float(rowL.open), float(rowL.high), float(rowL.low), float(rowL.close), ts=ts)
        if closed:
            all_closed.extend(closed)
            for t in closed:
                daily_pnl_r += t.get('pnl_r', 0.0)

        # Reset daily PnL counter at day boundary
        if ts - daily_reset_ts >= MS_PER_DAY:
            daily_pnl_r = 0.0
            daily_reset_ts = ts

        # Daily loss circuit breaker
        if daily_pnl_r <= -max_daily_loss_r:
            if progress: progress.update(1)
            continue

        # Inject BTC Context & Orderbook Stub
        btc_bull_val = btc_aligned[iExec] if iExec < len(btc_aligned) else 1.0
        extras = {
            "btc_bullish": btc_bull_val, 
            "funding_rate": 0.0,
            "btcd_trend": 0.0,
            "dxy_trend": float(dxy_aligned[iExec]) if iExec < len(dxy_aligned) else 0.0,
            "spx_trend": float(spx_aligned[iExec]) if iExec < len(spx_aligned) else 0.0,
            "delta_oi": float(oi_aligned[iExec]) if iExec < len(oi_aligned) else 0.0,
            "bid_ask_imbalance": 0.5,
            "macro_sentiment": 0.0
        }

        feats = confluence_features(cfg, macro_tf, swing_tf, htf, exec_tf, max(0, iMacro-1), max(0, iSwing-1), max(0, iHtf-1), iExec, precomp=pre_map, extras=extras)
        feats["symbol"] = cfg.symbol
        
        plan = plan_trade_with_brain(cfg, brain, feats, adv_features, iExec, pExec)
        
        if plan:
            tm.submit_plan(plan, {"o": float(rowL.open), "h": float(rowL.high), "l": float(rowL.low), "c": float(rowL.close)})

        if progress: progress.update(1)

    if progress: progress.close()

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
                'pnl_r', 'exit_reason', 'bars_open', 'initial_risk_unit', 'analysis'
            ]
            export_cols = [c for c in desired_cols if c in df_export.columns]
            df_export[export_cols].to_csv("backtest_trades_report.csv", index=False)
            log.info("📊 Saved detailed trade report to: backtest_trades_report.csv")
        except Exception as e:
            log.error(f"Failed to export CSV trade report: {e}")

    wins = sum(1 for t in all_closed if t["pnl_r"] > 0)
    total_r = float(sum(t["pnl_r"] for t in all_closed))
    winrate = (wins / max(1, len(all_closed))) * 100.0
    dur_s = time.perf_counter() - start_t
    summary = {
        "symbol": cfg.symbol,
        "bars": int(len(exec_iter)),
        "trades": int(len(all_closed)),
        "winrate_pct": float(winrate),
        "total_r": float(total_r),
        "net_profit_est": float(total_r * cfg.account_notional * cfg.risk_per_trade),
        "total_duration_s": float(dur_s)
    }
    return summary, all_closed

def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--exchange", dest="exchange_id", default="bybit")
    p.add_argument("--market", dest="market_type", choices=["spot","swap","future"], default="swap")
    p.add_argument("--symbol", default="BTC/USDT")
    p.add_argument("--since", required=True)
    p.add_argument("--until", required=True)
    p.add_argument("--macro-tf", default="1d")
    p.add_argument("--swing-tf", default="4h")
    p.add_argument("--htf", default="1h")
    p.add_argument("--exec-tf", default="15m")
    p.add_argument("--brains-dir", default=None, help="Directory containing localized strategy brains")
    p.add_argument("--account-notional", type=float, default=1000.0)
    p.add_argument("--min-rr", type=float, default=2.5)
    p.add_argument("--rr", type=float, default=2.0)
    p.add_argument("--atr-mult-sl", type=float, default=1.0)
    p.add_argument("--atr-mult-tp", type=float, default=2.0)
    p.add_argument("--scalper-rr", type=float, default=2.0)
    p.add_argument("--risk-per-trade", type=float, default=0.01)
    p.add_argument("--min-prob", type=float, default=0.55)
    p.add_argument("--min-conf-long", type=float, default=3.0)
    p.add_argument("--min-conf-short", type=float, default=3.0)
    p.add_argument("--max-concurrent", type=int, default=3)
    p.add_argument("--max-concurrent-buy", type=int, default=0)
    p.add_argument("--max-concurrent-sell", type=int, default=0)
    p.add_argument("--progress", choices=["on","off","auto"], default="auto")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--cache-dir", default="./data_cache")
    p.add_argument("--use-cache", action=argparse.BooleanOptionalAction, default=True)
    return p

def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    summary, _ = run_backtest(args, preloaded=None)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()