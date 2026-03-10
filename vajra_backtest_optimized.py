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
    while cur < until_ms:
        df = exw.fetch_ohlcv_df(symbol, timeframe, limit=1000, since=cur)
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
    htf: pd.DataFrame
    mtf: pd.DataFrame
    ltf: pd.DataFrame
    btc: Optional[pd.DataFrame] = None # Added BTC Dataframe

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

    htf = one(args.symbol, args.htf)
    mtf = one(args.symbol, args.mtf)
    ltf = one(args.symbol, args.ltf)
    
    # FETCH BTC Context (Regime Filter)
    # STRICTLY USE 4H FOR REGIME, IGNORING STRATEGY HTF
    btc_df = None
    if args.symbol != "BTC/USDT":
        log.info("Fetching BTC/USDT (4h) context data...")
        btc_df = one("BTC/USDT", "4h") 
    else:
        btc_df = one("BTC/USDT", "4h") 

    return Preloaded(htf=htf, mtf=mtf, ltf=ltf, btc=btc_df)

# --------- Config mapping ---------

def args_to_cfg(args) -> EngineConfig:
    cfg = EngineConfig()
    cfg.exchange_id = args.exchange_id
    cfg.market_type = args.market_type
    cfg.symbol = args.symbol
    cfg.htf = args.htf; cfg.mtf = args.mtf; cfg.ltf = args.ltf
    cfg.scalper_tf = getattr(args, "scalper_tf", "1m")
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

    if preloaded is None:
        pre = _preload_or_fetch(args)
        htf, mtf, ltf, btc = pre.htf, pre.mtf, pre.ltf, pre.btc
    else:
        htf, mtf, ltf = preloaded.htf.copy(), preloaded.mtf.copy(), preloaded.ltf.copy()
        btc = preloaded.btc.copy() if preloaded.btc is not None else htf.copy()

    # Calculate BTC Trend
    btc.sort_values("timestamp", inplace=True, ignore_index=True)
    btc_c = btc["close"].values
    
    btc_ema_trend = _ema_np(btc_c, 100)
    btc_bull_arr = (btc_c > btc_ema_trend).astype(float)
    
    btc_s = pd.Series(btc_bull_arr, index=btc["timestamp"]).shift(1) 
    btc_aligned = btc_s.reindex(ltf["timestamp"], method='ffill').fillna(1.0).values 

    # Align BTC Close for relative strength features
    btc_s_close = pd.Series(btc_c, index=btc["timestamp"]).shift(1).reindex(ltf["timestamp"], method='ffill').fillna(method='bfill').values
    if len(btc_s_close) != len(ltf):
        btc_s_close = np.resize(btc_s_close, len(ltf))

    brain = BrainLearningManager(cfg) 
    mem = MemoryManager(cfg, db=None)
    tm = TradeManager(cfg, exw, mem, brain, narrative_map=None)

    for df in (htf, mtf, ltf):
        df.sort_values("timestamp", inplace=True, ignore_index=True)

    since_ms = parse_time(args.since)
    until_ms = parse_time(args.until)
    ltf_iter = ltf[(ltf["timestamp"] >= since_ms) & (ltf["timestamp"] < until_ms)]

    log.info("Generating Engine V7 Precomputed features for base engine parity...")
    pre_htf = Precomp(htf)
    pre_mtf = Precomp(mtf)
    pre_ltf = Precomp(ltf)
    pre_map = {"htf": pre_htf, "mtf": pre_mtf, "ltf": pre_ltf}

    adv_features = precompute_v6_features(
        pre_htf, pre_mtf, pre_ltf, htf, mtf, ltf, 
        btc_close_arr=btc_s_close
    )

    iH = 0; iM = 0
    all_closed: List[Dict[str, Any]] = []
    
    # OPTIMIZED PROGRESS BAR (Less I/O)
    progress = None
    if tqdm and getattr(args, "progress", "auto") != "off":
        progress = tqdm(total=len(ltf_iter), desc="Backtest", mininterval=1.0) 

    for idx, (_, rowL) in enumerate(ltf_iter.iterrows()):
        ts = int(rowL["timestamp"])

        while iH+1 < len(htf) and int(htf["timestamp"].iloc[iH+1]) <= ts: iH += 1
        while iM+1 < len(mtf) and int(mtf["timestamp"].iloc[iM+1]) <= ts: iM += 1
        iL = int(ltf.index.get_loc(rowL.name))

        closed = tm.step_bar(cfg.symbol, float(rowL["open"]), float(rowL["high"]), float(rowL["low"]), float(rowL["close"]), ts=ts)
        if closed: all_closed.extend(closed)

        # Inject BTC Context & Orderbook Stub
        btc_bull_val = btc_aligned[iL] if iL < len(btc_aligned) else 1.0
        extras = {
            "btc_bullish": btc_bull_val, 
            "funding_rate": 0.0,
            "btcd_trend": 0.0,
            "bid_ask_imbalance": 0.5,
            "macro_sentiment": 0.0
        }

        feats = confluence_features(cfg, htf, mtf, ltf, iH=iH, iM=iM, iL=iL, precomp=pre_map, extras=extras)
        feats["symbol"] = cfg.symbol
        
        # Test baseline engine with NO brain
        plan = plan_trade_with_brain(cfg, None, feats, adv_features, iH, iM, iL, pre_ltf)
        
        if plan:
            tm.submit_plan(plan, {"o": float(rowL["open"]), "h": float(rowL["high"]), "l": float(rowL["low"]), "c": float(rowL["close"])})

        if progress: progress.update(1)

    if progress: progress.close()

    wins = sum(1 for t in all_closed if t["pnl_r"] > 0)
    total_r = float(sum(t["pnl_r"] for t in all_closed))
    winrate = (wins / max(1, len(all_closed))) * 100.0
    dur_s = time.perf_counter() - start_t
    summary = {
        "symbol": cfg.symbol,
        "bars": int(len(ltf_iter)),
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
    p.add_argument("--htf", default="12h"); p.add_argument("--mtf", default="1h"); p.add_argument("--ltf", default="15m")
    p.add_argument("--scalper-tf", default="1m")
    p.add_argument("--account-notional", type=float, default=1000.0)
    p.add_argument("--min-rr", type=float, default=2.5)
    p.add_argument("--rr", type=float, default=3.0)
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