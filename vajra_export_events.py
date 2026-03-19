#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vajra_export_events.py — Cleaned & Centralized Exporter (Production Ready)
========================================================================
LEVEL 30 (THE META-LABEL GENERATOR):
- EXTRAS SYNC: Orderbook and BTCD stubs synchronized for true extraction parity.
- LOOKAHEAD PATCHED: MTF/HTF unclosed candle index leaks sealed.
- MACRO PATCHED: 24-hour delay added to yfinance daily closes to prevent future leakage.
"""
from __future__ import annotations

import argparse, gzip, json, logging, traceback
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd

# Import from the Central Engine
from vajra_engine_ultra_v6_final import (
    AadhiraayanEngineConfig as EngineConfig,
    ExchangeWrapper, MemoryManager, Precomp, TradeManager,
    confluence_features,
    precompute_v6_features,
    plan_trade_with_brain,  
    _ema_np,
    _roc
)
import vajra_backtest_optimized as bt

try:
    from vajra_overrides import _strategy_overrides
except ImportError:
    def _strategy_overrides(cfg): pass

log = logging.getLogger("vajra.export")
if not log.handlers:
    log.setLevel(logging.INFO)
    log.addHandler(logging.StreamHandler())

def _open_out(path: str):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    return gzip.open(p, "wt", encoding="utf-8") if p.suffix == ".gz" else open(p, "w", encoding="utf-8")

def _parse_date_or_ms(s):
    if str(s).isdigit(): return int(s)
    return int(pd.Timestamp(s, tz="UTC").timestamp() * 1000)

def fetch_macro_trend(ticker_symbol: str, ltf_timestamps: pd.Series) -> np.ndarray:
    try:
        import yfinance as yf
        start_dt = pd.to_datetime(ltf_timestamps.iloc[0], unit='ms')
        end_dt = pd.to_datetime(ltf_timestamps.iloc[-1], unit='ms')
        start_dt_padded = start_dt - pd.Timedelta(days=30)
        df = yf.download(ticker_symbol, start=start_dt_padded, end=end_dt, interval="1d", progress=False, auto_adjust=True)
        
        if df.empty: return np.zeros(len(ltf_timestamps))
        
        if isinstance(df.columns, pd.MultiIndex):
            close_s = df['Close'].iloc[:, 0]
        else:
            close_s = df['Close']
            
        close_np = close_s.ffill().values
        ma20 = _ema_np(close_np, 20)
        trend = np.zeros_like(close_np)
        for i in range(len(close_np)):
            if ma20[i] > 1e-12:
                trend[i] = (close_np[i] - ma20[i]) / ma20[i] * 100.0
        
        # MACRO TIME LEAK FIX: Shift the daily close by 24h to ensure it is fully closed!
        trend_series = pd.Series(trend, index=df.index.view('int64') // 10**6)

        # FIX: Strip duplicate indices to prevent reindex crashes
        trend_series = trend_series[~trend_series.index.duplicated(keep='last')]

        trend_series.index = trend_series.index + 86400000 
        
        aligned = trend_series.shift(1).reindex(ltf_timestamps, method='ffill').fillna(0.0).values
        return aligned
    except Exception as e:
        log.warning(f"yfinance failed for {ticker_symbol}: {e}")
        return np.zeros(len(ltf_timestamps))

def fetch_historical_funding_rates(exw: ExchangeWrapper, symbol: str, ltf_timestamps: pd.Series) -> np.ndarray:
    try:
        if exw.client.has.get('fetchFundingRateHistory'):
            # Fetch a larger history if possible, but CCXT often limits to 200/1000
            funding = exw.client.fetch_funding_rate_history(symbol, limit=1000, params={'category': 'linear'})
            if not funding: return np.zeros(len(ltf_timestamps))

            df = pd.DataFrame(funding)
            if 'fundingRate' not in df.columns or 'timestamp' not in df.columns:
                return np.zeros(len(ltf_timestamps))

            df['fundingRate'] = pd.to_numeric(df['fundingRate'], errors='coerce')
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)

            aligned = df['fundingRate'].shift(1).reindex(pd.to_datetime(ltf_timestamps, unit='ms', utc=True), method='ffill').fillna(0.0)
            return aligned.values
    except Exception as e:
        log.warning(f"CCXT fetch_funding failed for {symbol}: {e}")
    return np.zeros(len(ltf_timestamps))

def fetch_delta_oi(exw: ExchangeWrapper, symbol: str, timeframe: str, ltf_timestamps: pd.Series) -> np.ndarray:
    try:
        target_symbol = symbol
        if exw.client.id == 'bybit' and ':' not in symbol and '/' in symbol:
            quote = symbol.split('/')[1]
            target_symbol = f"{symbol}:{quote}"
            
        if exw.client.has.get('fetchOpenInterestHistory'):
            oi = exw.client.fetch_open_interest_history(target_symbol, timeframe, limit=1000)
            if not oi: return np.zeros(len(ltf_timestamps))
            
            oi = [x for x in oi if x.get('timestamp') is not None]
            if not oi: return np.zeros(len(ltf_timestamps))
            
            df = pd.DataFrame(oi)
            if 'openInterestValue' not in df.columns:
                return np.zeros(len(ltf_timestamps))
            
            df['openInterestValue'] = pd.to_numeric(df['openInterestValue'], errors='coerce')
            vals = df['openInterestValue'].ffill().values
            
            if len(vals) > 1:
                delta = np.zeros_like(vals)
                delta[1:] = (vals[1:] - vals[:-1]) / (vals[:-1] + 1e-9) * 100.0
                df['delta_oi'] = delta
            else:
                df['delta_oi'] = 0.0
                
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
            
            aligned = df['delta_oi'].shift(1).reindex(pd.to_datetime(ltf_timestamps, unit='ms', utc=True), method='ffill').fillna(0.0)
            return aligned.values
    except Exception as e:
        log.warning(f"CCXT fetch_oi failed for {symbol}: {e}")
    return np.zeros(len(ltf_timestamps))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exchange", dest="exchange_id", default="bybit")
    p.add_argument("--market", dest="market_type", default="swap")
    p.add_argument("--symbol", default="BTC/USDT")
    p.add_argument("--since", required=True)
    p.add_argument("--until", required=True)
    p.add_argument("--htf", default="12h"); p.add_argument("--mtf", default="1h"); p.add_argument("--ltf", default="15m")
    p.add_argument("--out", required=True)
    p.add_argument("--min-rr", type=float, default=-1.0)
    
    args = p.parse_args()

    cfg = EngineConfig()
    try: _strategy_overrides(cfg); log.info("Applied strategy overrides.")
    except Exception as e: log.warning(f"Could not apply overrides: {e}")

    # --- FORCE EXPORT TO SEE EVERYTHING (DO NOT STARVE THE AI) ---
    cfg.filter_htf_trend = False
    cfg.filter_btc_regime = False
    cfg.filter_rvol_breakout = False
    cfg.filter_adx_chop = False

    cfg.exchange_id = args.exchange_id; cfg.market_type = args.market_type; cfg.symbol = args.symbol
    cfg.htf, cfg.mtf, cfg.ltf = args.htf, args.mtf, args.ltf
    if args.min_rr >= 0: cfg.min_rr = args.min_rr
    cfg.max_concurrent = 0  

    if getattr(cfg, "use_meta_labeling", False):
        log.info("META-LABEL MODE ACTIVE. Obeying Overrides Geometry.")

    log.info("="*60)
    log.info(f" EXPORT MODE: META-LABEL GENERATOR")
    log.info(f" Logic Source: vajra_engine_ultra_v6_final.plan_trade")
    log.info("="*60)

    preloaded = bt._preload_or_fetch(args)
    htf, mtf, ltf = preloaded.htf, preloaded.mtf, preloaded.ltf
    
    btc = preloaded.btc if preloaded.btc is not None else htf.copy()
    btc.sort_values("timestamp", inplace=True, ignore_index=True)
    btc_c = btc["close"].values
    
    btc_ema_trend = _ema_np(btc_c, 100) 
    btc_bull_arr = (btc_c > btc_ema_trend).astype(float)
    
    btc_s = pd.Series(btc_bull_arr, index=btc["timestamp"]).shift(1)
    btc_aligned = btc_s.reindex(ltf["timestamp"], method='ffill').fillna(1.0).values

    htf.sort_values("timestamp", inplace=True); htf.reset_index(drop=True, inplace=True)
    mtf.sort_values("timestamp", inplace=True); mtf.reset_index(drop=True, inplace=True)
    ltf.sort_values("timestamp", inplace=True); ltf.reset_index(drop=True, inplace=True)

    log.info("Downloading Macro & Micro-Structure contextual data...")
    exw = ExchangeWrapper(cfg)
    dxy_aligned = fetch_macro_trend("DX=F", ltf["timestamp"])
    spx_aligned = fetch_macro_trend("ES=F", ltf["timestamp"])
    oi_aligned = fetch_delta_oi(exw, cfg.symbol, cfg.ltf, ltf["timestamp"])
    funding_aligned = fetch_historical_funding_rates(exw, cfg.symbol, ltf["timestamp"])

    # Historical BTC.D Fetch
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
            btcd_aligned = btcd_series.shift(1).reindex(pd.to_datetime(ltf["timestamp"], unit='ms', utc=True), method='ffill').fillna(0.0).values
        else:
            btcd_aligned = np.zeros(len(ltf))
    except Exception as e:
        log.warning(f"Failed to fetch BTCDOM: {e}")
        btcd_aligned = np.zeros(len(ltf))

    pre_h, pre_m, pre_l = Precomp(htf), Precomp(mtf), Precomp(ltf)
    pre_map = {"htf": pre_h, "mtf": pre_m, "ltf": pre_l}

    # Synthetic Bid-Ask Imbalance Proxy (Using wicks vs body size as order flow proxy since historical L2 is unavailable)
    range_l = np.maximum(pre_l.h - pre_l.l, 1e-9)
    body_top = np.maximum(pre_l.c, pre_l.o)
    body_bot = np.minimum(pre_l.c, pre_l.o)
    upper_wick = pre_l.h - body_top
    lower_wick = body_bot - pre_l.l
    bid_ask_proxy = 0.5 + (lower_wick - upper_wick) / (2 * range_l)
    bid_ask_aligned = pd.Series(bid_ask_proxy).shift(1).fillna(0.5).values

    btc_s_close = pd.Series(btc_c, index=btc["timestamp"]).shift(1).reindex(ltf["timestamp"], method='ffill').bfill().values
    if len(btc_s_close) != len(ltf):
        btc_s_close = np.resize(btc_s_close, len(ltf))

    log.info("Generating V7 features...")
    adv_features = precompute_v6_features(
        pre_h, pre_m, pre_l, htf, mtf, ltf, 
        btc_close_arr=btc_s_close
    )
    
    mem = MemoryManager(cfg, db=None)
    tm = TradeManager(cfg, exw, mem, brain=None)
    
    events = []; open_meta = []
    
    since_ms, until_ms = _parse_date_or_ms(args.since), _parse_date_or_ms(args.until)
    ltf_iter = ltf[(ltf["timestamp"] >= since_ms) & (ltf["timestamp"] < until_ms)]

    log.info(f"Processing {len(ltf_iter)} bars...")
    
    len_ltf = len(ltf); len_mtf = len(mtf); len_htf = len(htf)
    
    for row in ltf_iter.itertuples():
        ts = int(row.timestamp)
        
        iH = np.searchsorted(htf['timestamp'].values, ts, side='right') - 1
        iM = np.searchsorted(mtf['timestamp'].values, ts, side='right') - 1
        iL = int(row.Index)
        
        if iH < 0 or iM < 0: continue

        bar = {"o": row.open, "h": row.high, "l": row.low, "c": row.close}
        exits = tm.step_bar(cfg.symbol, bar["o"], bar["h"], bar["l"], bar["c"], ts=ts)
        
        for cl in exits:
            meta = next((m for m in open_meta if m["key"] == cl.get("key")), None)
            if meta:
                open_meta.remove(meta)
                # PURE SIGNAL EDGE: 1.0 strictly if it hits structural TP.
                meta_label = 1.0 if cl.get("pnl_r", 0.0) >= 0.5 else 0.0
                
                if -50 < cl["pnl_r"] < 50:
                    events.append({
                        "symbol": cfg.symbol, 
                        "entry_ts": meta["entry_ts"], 
                        "exit_ts": ts,
                        "pnl_r": cl["pnl_r"], 
                        "rr": meta["rr"], 
                        "side": cl.get("side"),
                        "entry_price": cl.get("entry"),
                        "exit_price": cl.get("exit_price"),
                        "stop_loss": cl.get("sl"),
                        "tp": cl.get("tp"),
                        "reason": cl.get("exit_reason"),
                        "meta_label": meta_label,
                        **meta["features"]
                    })

        btc_val = btc_aligned[iL] if iL < len(btc_aligned) else 1.0
        
        extras = {
            "btc_bullish": btc_val,
            "funding_rate": float(funding_aligned[iL]) if iL < len(funding_aligned) else 0.0,
            "btcd_trend": float(btcd_aligned[iL]) if iL < len(btcd_aligned) else 0.0,
            "dxy_trend": float(dxy_aligned[iL]) if iL < len(dxy_aligned) else 0.0,
            "spx_trend": float(spx_aligned[iL]) if iL < len(spx_aligned) else 0.0,
            "delta_oi": float(oi_aligned[iL]) if iL < len(oi_aligned) else 0.0,
            "bid_ask_imbalance": float(bid_ask_aligned[iL]) if iL < len(bid_ask_aligned) else 0.5,
            "macro_sentiment": 0.0
        }

        base = confluence_features(cfg, htf, mtf, ltf, iH, iM, iL, pre_map, extras=extras)
        base["symbol"] = cfg.symbol
        plan = plan_trade_with_brain(cfg, None, base, adv_features, iH, iM, iL, pre_l)
        
        if plan:
            plan['key'] = f"{plan['side']}-{ts}"
            full_feats = {**base}
            
            for k, v_arr in adv_features.items():
                 if not hasattr(v_arr, '__getitem__') or isinstance(v_arr, (str, float, int)):
                     full_feats[k] = v_arr; continue
                 arr_len = len(v_arr)
                 idx = -1
                 if arr_len == len_ltf: idx = iL
                 # UNCLOSED CANDLE LEAK PATCHED: Force historical 1-shift 
                 elif arr_len == len_mtf: idx = max(0, iM - 1)
                 elif arr_len == len_htf: idx = max(0, iH - 1)
                 else: continue
                 if 0 <= idx < arr_len:
                     val = v_arr[idx]
                     full_feats[k] = float(val) if np.isfinite(val) else 0.0

            px = bar["c"]
            full_feats["rsi_14"] = float(pre_l.rsi14[iL]) if iL < len(pre_l.rsi14) else 50.0
            
            entry_dt = pd.to_datetime(ts, unit="ms", utc=True)
            full_feats["hour_of_day"] = entry_dt.hour
            full_feats["day_of_week"] = entry_dt.dayofweek
            full_feats["side"] = 1.0 if plan["side"] == "long" else 0.0

            if tm.submit_plan(plan, bar):
                open_meta.append({
                    "entry_ts": ts, 
                    "features": full_feats, 
                    "rr": plan["rr"], 
                    "key": plan.get("key")
                })

    with _open_out(args.out) as f:
        for ev in events:
            clean_ev = {k: (float(v) if isinstance(v, (float, np.floating)) and np.isfinite(v) else v) for k,v in ev.items()}
            f.write(json.dumps(clean_ev) + "\n")
            
    log.info(f"Exported {len(events)} sanitized Meta-Label events.")

if __name__ == "__main__":
    main()