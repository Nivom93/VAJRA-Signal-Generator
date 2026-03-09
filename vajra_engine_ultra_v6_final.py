#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vajra_engine_ultra_v6_final.py — Core Engine & Central Logic Hub
==============================================================
LEVEL 30 UPGRADE (THE INSTITUTIONAL MASTERPIECE):
- SYNTHETIC MICRO-STRUCTURE: Anchored VWAP, Keltner Channels, CVD Divergence, Volume Profile.
- TRADE MANAGEMENT: Auto-Breakeven, Structural Trailing SL, Time-in-Force Decay.
- STRATEGY EXPANSION: ICT Killzones, Auction Market Theory (Zeta), Fractal Liquidity (Epsilon).
- DYNAMIC RISK: Liquidation Squeeze Multipliers, Dynamic Take Profits based on BB Expansion.
"""
from __future__ import annotations

import os, json, time, math, threading, sqlite3, random, csv, traceback
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# --- BULLETPROOF IMPORTS ---
try:
    import ccxt; import joblib; from sklearn.preprocessing import StandardScaler
except:
    pass

try:
    from numba import njit
except Exception:
    def njit(*args, **kwargs):
        def decorator(func): return func
        if len(args) == 1 and callable(args[0]): return args[0]
        return decorator

# ==============================================================================
# 1. BASE MATHEMATICAL HELPERS (JIT COMPILED)
# ==============================================================================

@njit(cache=True)
def _safe_divide(a, b, default=0.0):
    if abs(b) < 1e-12: return default
    return a / b

def _ema_np(x: np.ndarray, span: int) -> np.ndarray:
    if span <= 1 or x.size == 0: return x.astype(np.float64, copy=True)
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(x, dtype=np.float64)
    out[0] = x[0]
    _ema_loop(x, out, alpha)
    return out

@njit(cache=True)
def _ema_loop(x, out, alpha):
    for i in range(1, x.size):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i-1]

def _rolling_mean(arr, window, min_periods=1):
    if window <= 0: return arr.astype(np.float64)
    return pd.Series(arr).rolling(window, min_periods=min_periods).mean().to_numpy(dtype=np.float64)

def _rolling_std(arr, window, min_periods=2):
    if window <= 0: return np.zeros_like(arr, dtype=np.float64)
    return pd.Series(arr).rolling(window, min_periods=max(min_periods,2)).std(ddof=0).to_numpy(dtype=np.float64)

def _roc(arr, k):
    out = np.zeros_like(arr, dtype=np.float64)
    if k <= 0 or arr.size <= k: return out
    shifted = np.full_like(arr, np.nan); shifted[k:] = arr[:-k]
    with np.errstate(divide='ignore', invalid='ignore'): out = (arr / shifted) - 1.0
    return np.nan_to_num(out)

def _atr_np(high, low, close, n=14):
    if n <= 0 or high.size < 1: return np.zeros_like(close)
    prev_c = np.roll(close, 1); prev_c[0] = close[0]
    tr = np.maximum.reduce([np.abs(high-low), np.abs(high-prev_c), np.abs(low-prev_c)])
    return _ema_np(np.nan_to_num(tr), n*2-1)

# ==============================================================================
# 2. ADVANCED INDICATORS & SYNTHETIC MICRO-STRUCTURE
# ==============================================================================

@njit(cache=True)
def _synthetic_cvd_np(open_p, high, low, close, volume):
    n = len(close)
    cvd = np.zeros(n, dtype=np.float64)
    if n == 0: return cvd
    for i in range(1, n):
        rng = high[i] - low[i]
        if rng > 1e-12:
            bp_ratio = (close[i] - low[i]) / rng
        else:
            bp_ratio = 0.5
        bp = volume[i] * bp_ratio
        sp = volume[i] - bp
        cvd[i] = cvd[i-1] + (bp - sp)
    return cvd

def _keltner_channels_np(close, high, low, n=20, k=1.5):
    atr = _atr_np(high, low, close, n)
    ema_c = _ema_np(close, n)
    upper = ema_c + (k * atr)
    lower = ema_c - (k * atr)
    return upper, lower

@njit(cache=True)
def _avwap_np(typical_price, volume, anchors):
    n = len(typical_price)
    avwap = np.zeros(n, dtype=np.float64)
    cum_vol = 0.0
    cum_pv = 0.0
    for i in range(n):
        if anchors[i] == 1:
            cum_vol = 0.0
            cum_pv = 0.0
        cum_vol += volume[i]
        cum_pv += typical_price[i] * volume[i]
        if cum_vol > 1e-12:
            avwap[i] = cum_pv / cum_vol
        else:
            avwap[i] = typical_price[i]
    return avwap

@njit(cache=True)
def _volume_profile_np(close, volume, window=100, bins=20):
    n = len(close)
    poc = np.zeros(n, dtype=np.float64)
    vah = np.zeros(n, dtype=np.float64)
    val = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if i < 2:
            poc[i] = vah[i] = val[i] = close[i]
            continue
        start_idx = max(0, i - window + 1)
        chunk_c = close[start_idx:i+1]
        chunk_v = volume[start_idx:i+1]
        min_c = np.min(chunk_c)
        max_c = np.max(chunk_c)
        if max_c - min_c < 1e-12:
            poc[i] = vah[i] = val[i] = close[i]
            continue
        bin_size = (max_c - min_c) / bins
        vol_bins = np.zeros(bins, dtype=np.float64)
        for j in range(len(chunk_c)):
            idx = int((chunk_c[j] - min_c) / bin_size)
            if idx >= bins: idx = bins - 1
            vol_bins[idx] += chunk_v[j]
            
        max_bin = np.argmax(vol_bins)
        poc[i] = min_c + (max_bin + 0.5) * bin_size
        
        target_vol = np.sum(vol_bins) * 0.70
        curr_vol = vol_bins[max_bin]
        up_idx = max_bin
        dn_idx = max_bin
        
        while curr_vol < target_vol and (up_idx < bins - 1 or dn_idx > 0):
            v_up = vol_bins[up_idx + 1] if up_idx < bins - 1 else -1
            v_dn = vol_bins[dn_idx - 1] if dn_idx > 0 else -1
            if v_up > v_dn:
                up_idx += 1
                curr_vol += v_up
            else:
                dn_idx -= 1
                curr_vol += v_dn
                
        vah[i] = min_c + (up_idx + 1) * bin_size
        val[i] = min_c + dn_idx * bin_size
    return poc, vah, val

@njit(cache=True)
def _cvd_divergence_np(close, cvd, last_sl, last_sh, last_sl_cvd, last_sh_cvd):
    n = len(close)
    div_bull = np.zeros(n, dtype=np.float64)
    div_bear = np.zeros(n, dtype=np.float64)
    for i in range(10, n):
        # Bullish Divergence: Price makes a Lower Low (relative to last swing low)
        # BUT CVD makes a Higher Low (Current CVD > CVD at the time of that swing low)
        if close[i] < last_sl[i]:
            if cvd[i] > last_sl_cvd[i]:
                div_bull[i] = 1.0

        # Bearish Divergence: Price makes a Higher High (relative to last swing high)
        # BUT CVD makes a Lower High (Current CVD < CVD at the time of that swing high)
        if close[i] > last_sh[i]:
            if cvd[i] < last_sh_cvd[i]:
                div_bear[i] = 1.0
    return div_bull, div_bear

@njit(cache=True)
def _asian_range_np(high, low, hours, days):
    n = len(high)
    asian_high = np.zeros(n, dtype=np.float64)
    asian_low = np.zeros(n, dtype=np.float64)
    if n == 0: return asian_high, asian_low
    curr_h = high[0]; curr_l = low[0]
    for i in range(n):
        if 0 <= hours[i] < 6:
            if i > 0 and (hours[i-1] >= 6 or days[i] != days[i-1]):
                curr_h = high[i]; curr_l = low[i]
            else:
                curr_h = max(curr_h, high[i]); curr_l = min(curr_l, low[i])
        asian_high[i] = curr_h; asian_low[i] = curr_l
    return asian_high, asian_low

@njit(cache=True)
def _rvol_np(vol, n=20):
    sz = len(vol)
    if sz < n: return np.zeros(sz)
    rvol = np.zeros(sz)
    current_sum = 0.0
    for i in range(n): current_sum += vol[i]
    avg = current_sum / n
    if avg > 0: rvol[n-1] = vol[n-1] / avg
    for i in range(n, sz):
        current_sum = current_sum - vol[i-n] + vol[i] 
        avg = current_sum / n
        rvol[i] = vol[i] / avg if avg > 1e-9 else 0.0
    return rvol

@njit(cache=True)
def _adx_np(high, low, close, n=14):
    sz = len(close)
    if sz < n * 2: return np.zeros(sz)
    tr = np.zeros(sz); dm_plus = np.zeros(sz); dm_minus = np.zeros(sz)
    for i in range(1, sz):
        hl = high[i] - low[i]; hc = abs(high[i] - close[i-1]); lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, max(hc, lc))
        up = high[i] - high[i-1]; down = low[i-1] - low[i]
        if (up > down) and (up > 0): dm_plus[i] = up
        else: dm_plus[i] = 0.0
        if (down > up) and (down > 0): dm_minus[i] = down
        else: dm_minus[i] = 0.0
    alpha = 1.0 / n 
    tr_s = np.zeros(sz); dm_plus_s = np.zeros(sz); dm_minus_s = np.zeros(sz)
    tr_s[0]=tr[0]; dm_plus_s[0]=dm_plus[0]; dm_minus_s[0]=dm_minus[0]
    for i in range(1, sz):
        tr_s[i] = alpha * tr[i] + (1 - alpha) * tr_s[i-1]
        dm_plus_s[i] = alpha * dm_plus[i] + (1 - alpha) * dm_plus_s[i-1]
        dm_minus_s[i] = alpha * dm_minus[i] + (1 - alpha) * dm_minus_s[i-1]
    dx = np.zeros(sz)
    for i in range(sz):
        di_plus = 100 * _safe_divide(dm_plus_s[i], tr_s[i])
        di_minus = 100 * _safe_divide(dm_minus_s[i], tr_s[i])
        sum_di = di_plus + di_minus
        dx[i] = 100 * _safe_divide(abs(di_plus - di_minus), sum_di)
    adx = np.zeros(sz); adx[0] = dx[0]
    for i in range(1, sz):
        adx[i] = alpha * dx[i] + (1 - alpha) * adx[i-1]
    return adx

@njit(cache=True)
def _bb_bands_np(close, n=20, k=2.0):
    sz = len(close)
    if sz < n: return np.zeros(sz), np.zeros(sz), np.zeros(sz)
    upper = np.zeros(sz); lower = np.zeros(sz); width = np.zeros(sz)
    for i in range(n, sz):
        sl = close[i-n+1 : i+1]
        mu = np.mean(sl)
        sigma = np.std(sl)
        u = mu + (k * sigma); l = mu - (k * sigma)
        upper[i] = u; lower[i] = l
        width[i] = _safe_divide((u - l), mu) * 100.0
    return upper, lower, width

# ==============================================================================
# 3. LEVEL 9 PATTERN RECOGNITION
# ==============================================================================

@njit(cache=True)
def _detect_patterns(open_p, high, low, close, swing_hi, swing_lo, last_sh, last_sl):
    n = len(close)
    w_pattern = np.zeros(n); m_pattern = np.zeros(n)
    fvg_bull = np.zeros(n); fvg_bear = np.zeros(n)
    sweep_bull = np.zeros(n); sweep_bear = np.zeros(n)
    
    for i in range(5, n):
        # 1. Double Bottom (W Pattern)
        if swing_lo[i-1] == 0 and low[i] > low[i-1]: 
            prev_sl = last_sl[i-5] 
            if prev_sl > 0:
                dist = abs(low[i] - prev_sl) / prev_sl
                if dist < 0.005: w_pattern[i] = 1.0

        # 2. Double Top (M Pattern)
        if swing_hi[i-1] == 0 and high[i] < high[i-1]:
            prev_sh = last_sh[i-5]
            if prev_sh > 0:
                dist = abs(high[i] - prev_sh) / prev_sh
                if dist < 0.005: m_pattern[i] = 1.0

        # 3. Fair Value Gaps (FVG)
        if low[i] > high[i-2]: fvg_bull[i] = low[i]
        if high[i] < low[i-2]: fvg_bear[i] = high[i]

        # 4. Liquidity Sweeps (False Breakout)
        curr_sl = last_sl[i]
        if low[i] < curr_sl and close[i] > curr_sl: sweep_bull[i] = 1.0
            
        curr_sh = last_sh[i]
        if high[i] > curr_sh and close[i] < curr_sh: sweep_bear[i] = 1.0
            
    return w_pattern, m_pattern, fvg_bull, fvg_bear, sweep_bull, sweep_bear

# ==============================================================================
# 4. COMPLEX INDICATORS & STRUCTURE
# ==============================================================================

@njit(cache=True)
def _kalman_smooth(data, n_iter=1):
    sz = len(data)
    if sz < 2: return data, np.zeros(sz)
    if not np.isfinite(data[0]): return data, np.zeros(sz)
    xhat = np.zeros(sz); P = np.zeros(sz); xhatminus = np.zeros(sz); Pminus = np.zeros(sz); K = np.zeros(sz); vel = np.zeros(sz)
    Q = 1e-5
    # DYNAMIC NORMALIZATION FIX:
    if data[0] > 1e-9: R = (data[0] * 0.001) ** 2
    else: R = 0.01 ** 2
        
    xhat[0] = data[0]; P[0] = 1.0
    for k in range(1, sz):
        if not np.isfinite(data[k]): xhat[k] = xhat[k-1]; vel[k] = 0.0; continue
        xhatminus[k] = xhat[k-1]; Pminus[k] = P[k-1] + Q
        K[k] = _safe_divide(Pminus[k], (Pminus[k] + R))
        xhat[k] = xhatminus[k] + K[k] * (data[k] - xhatminus[k])
        P[k] = (1 - K[k]) * Pminus[k]
        vel[k] = xhat[k] - xhat[k-1]
    return xhat, vel

@njit(cache=True)
def _calc_hurst(prices, window=100):
    sz = len(prices); hurst = np.full(sz, 0.5)
    if sz < window: return hurst
    for i in range(window, sz):
        chunk = prices[i-window:i]
        diffs = np.diff(np.log(np.maximum(chunk, 1e-9))) 
        if np.all(diffs == 0): continue
        mean_diff = np.mean(diffs); cum_dev = np.cumsum(diffs - mean_diff)
        r_val = np.max(cum_dev) - np.min(cum_dev); s_val = np.std(diffs)
        if s_val < 1e-9: continue
        rs = r_val / s_val; h = np.log(rs) / np.log(window)
        if h < 0: h = 0.0; 
        if h > 1: h = 1.0
        hurst[i] = h
    return hurst

def _rsi14_np(close):
    if close.size < 2: return np.full_like(close, 50.0)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta>0, delta, 0.0); loss = np.where(delta<0, -delta, 0.0)
    avg_gain = _ema_np(gain, 27); avg_loss = _ema_np(loss, 27)
    with np.errstate(divide='ignore', invalid='ignore'): rsi = 100.0 - (100.0 / (1.0 + (avg_gain/(avg_loss+1e-12))))
    return np.nan_to_num(rsi, nan=50.0)

def _macd_np(close, f=12, s=26, sig=9):
    if close.size < s: return np.zeros_like(close), np.zeros_like(close), np.zeros_like(close)
    f_ema = _ema_np(close, f); s_ema = _ema_np(close, s)
    macd = f_ema - s_ema; sign = _ema_np(macd, sig)
    return macd, sign, macd - sign

@njit(cache=True)
def _efficiency_ratio(close, n=10):
    sz = close.size; er = np.zeros(sz, dtype=np.float64)
    if sz <= n: return er
    diff = np.diff(close); abs_diff = np.abs(diff)
    for i in range(n, sz):
        net = np.abs(close[i] - close[i-n])
        noise = np.sum(abs_diff[i-n:i])
        er[i] = net/noise if noise > 1e-9 else (1.0 if net>0 else 0.0)
    return er

@njit(cache=True)
def _detect_divergence(close, rsi, lsl, lsh):
    n = close.size; db = np.zeros(n, dtype=np.float64); dbr = np.zeros(n, dtype=np.float64)
    for i in range(10, n):
        if close[i] < lsl[i]:
            pm = close[i] - close[i-5]; rm = rsi[i] - rsi[i-5]
            if pm < 0 and rm > 0 and rsi[i] < 40: db[i] = 1.0
        if close[i] > lsh[i]:
            pm = close[i] - close[i-5]; rm = rsi[i] - rsi[i-5]
            if pm > 0 and rm < 0 and rsi[i] > 60: dbr[i] = 1.0
    return db, dbr

@njit(cache=True)
def _bars_since_change(arr):
    n = arr.size; out = np.zeros(n, dtype=np.int64)
    if n == 0: return out
    last = arr[0]; count = 0
    for i in range(1, n):
        if arr[i] == last: count += 1
        else: last = arr[i]; count = 0
        out[i] = count
    return out

@njit(cache=True)
def _rolling_percentile_np(arr, window=100):
    n = len(arr)
    out = np.full(n, 0.5, dtype=np.float64)
    if n == 0: return out
    for i in range(n):
        start_idx = max(0, i - window + 1)
        chunk = arr[start_idx:i+1]
        val = arr[i]
        count_less = 0
        for v in chunk:
            if v <= val:
                count_less += 1
        out[i] = count_less / len(chunk)
    return out

@njit(cache=True)
def _rolling_vwap_np(typical_price, volume, window=20):
    n = len(typical_price)
    vwap = np.zeros(n, dtype=np.float64)
    if n == 0: return vwap
    for i in range(n):
        start_idx = max(0, i - window + 1)
        chunk_tp = typical_price[start_idx:i+1]
        chunk_v = volume[start_idx:i+1]
        cum_vol = np.sum(chunk_v)
        cum_pv = np.sum(chunk_tp * chunk_v)
        if cum_vol > 1e-12:
            vwap[i] = cum_pv / cum_vol
        else:
            vwap[i] = typical_price[i]
    return vwap

@njit(cache=True)
def _donchian_channels_np(high, low, window=20):
    n = len(high)
    dc_high = np.zeros(n, dtype=np.float64)
    dc_low = np.zeros(n, dtype=np.float64)
    for i in range(n):
        start_idx = max(0, i - window + 1)
        dc_high[i] = np.max(high[start_idx:i+1])
        dc_low[i] = np.min(low[start_idx:i+1])
    return dc_high, dc_low

# ==============================================================================
# 5. CONFIG & CLASSES
# ==============================================================================

@dataclass
class AadhiraayanEngineConfig:
    exchange_id: str = "bybit"; market_type: str = "swap"; symbol: str = "BTC/USDT"
    htf: str = "12h"; mtf: str = "1h"; ltf: str = "15m"; scalper_tf: str = "1m"
    risk_per_trade: float = 0.01; min_rr: float = 1.0; rr: float = 2.0
    atr_mult_sl: float = 1.0; atr_mult_tp: float = 2.0; scalper_rr: float = 2.0
    use_dca: bool = False; dca_max_safety_orders: int = 1
    dca_step_scale: float = 1.0; dca_volume_scale: float = 2.0; dca_tp_scale: float = 1.5
    be_trigger_r: float = 0.0; trailing_stop_trigger_r: float = 0.0; trailing_dist_r: float = 0.0
    filter_htf_trend: bool = False; filter_volatility_expansion: bool = False
    filter_funding_check: bool = True; filter_btc_regime: bool = True; filter_fib_entry: bool = False
    filter_hurst_chop: bool = True; filter_kalman_velocity: bool = True
    
    filter_adx_chop: bool = True       
    filter_btcd_regime: bool = True    
    filter_rvol_breakout: bool = True
    
    filter_time_of_day: bool = False
    filter_hurst_strict: bool = False
    use_macro_data: bool = False
    use_meta_labeling: bool = False

    allow_mean_reversion: bool = True
    entry_style: str = "limit"
    
    execution_style: str = "limit"  
    pullback_atr_mult: float = 0.0     
    
    strat_alpha_enabled: bool = True
    strat_omega_enabled: bool = True
    strat_gamma_enabled: bool = True
    strat_delta_enabled: bool = True
    strat_epsilon_enabled: bool = True
    strat_zeta_enabled: bool = True
    
    dynamic_tp_enabled: bool = True
    time_in_force_decay: int = 8
    
    min_prob: float = 0.65; min_conf_long: float = 3.0; min_conf_short: float = 3.0
    min_prob_long: float = 0.65; min_prob_short: float = 0.65
    max_concurrent: int = 3; max_concurrent_buy: int = 0; max_concurrent_sell: int = 0
    maker_fee_bps: float = 1.0; taker_fee_bps: float = 5.0; slippage_bps: float = 2.0
    account_notional: float = 1000.0; db_path: str = "vajra.sqlite"
    verbose: bool = True; dynamic_confluence: bool = True; dvol_enable: bool = False
    skip_log_throttle: int = 50; paper_mode: bool = True

EngineConfig = AadhiraayanEngineConfig

class VajraDB:
    def __init__(self, path):
        self.conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        with self.conn:
            self.conn.execute("CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY, ts REAL, event TEXT, symbol TEXT, tf TEXT, side TEXT, price REAL, size REAL, sl REAL, tp REAL, mode TEXT, reason TEXT, pnl_r REAL, key TEXT, entry_kind TEXT)")
    def append_trade(self, r):
        cols = ["ts","event","symbol","tf","side","price","size","sl","tp","mode","reason","pnl_r","key","entry_kind"]
        self.conn.execute(f"INSERT INTO trades ({','.join(cols)}) VALUES ({','.join(['?']*len(cols))})", tuple(r.get(c) for c in cols))

class MemoryManager:
    def __init__(self, cfg, db=None): self.cfg=cfg; self._db=db; self._live=db is not None; self._seen=set() if db else None
    def seen(self, k): 
        if not self._live: return False
        if k in self._seen: return True
        self._seen.add(k); return False
    def record_signal(self, r): 
        if self._live: self._db.append_trade(r)
    def record_fill(self, r): 
        if self._live: self._db.append_trade(r)
    def record_exit(self, r): 
        if self._live: self._db.append_trade(r)

class ExchangeWrapper:
    def __init__(self, cfg, markets_data=None):
        if ccxt is None: raise RuntimeError("ccxt missing")
        self.client = getattr(ccxt, cfg.exchange_id)({
            "enableRateLimit": True, "options": {"defaultType": cfg.market_type}, "timeout": 30000
        })
        if markets_data: self.client.markets = markets_data
        else: 
            for attempt in range(3):
                try: self.client.load_markets(); break
                except Exception as e:
                    if attempt == 2: raise e
                    print(f"Connection warning: {e}. Retrying..."); time.sleep(2)
    def fetch_ohlcv_df(self, sym, tf, limit=1000, since=None):
        r = self.client.fetch_ohlcv(sym, timeframe=tf, limit=min(1000,limit), since=since)
        return pd.DataFrame(r, columns=["timestamp","open","high","low","close","volume"])
    def fetch_funding_rate(self, symbol):
        try: funding = self.client.fetch_funding_rate(symbol); return float(funding.get('fundingRate', 0.0))
        except: return 0.0

# ==============================================================================
# 6. MARKET STRUCTURE & FEATURE ENGINEERING
# ==============================================================================

@njit(cache=True)
def _swing_points_strict(high, low, left=3, right=3):
    n = high.size; sh_conf = np.zeros(n, dtype=np.uint8); sl_conf = np.zeros(n, dtype=np.uint8)
    for i in range(left, n - right):
        window_h = high[i-left : i+right+1]; window_l = low[i-left : i+right+1]
        if high[i] == np.max(window_h):
            if i + right < n: sh_conf[i + right] = 1
        if low[i] == np.min(window_l):
            if i + right < n: sl_conf[i + right] = 1
    return sh_conf, sl_conf

@njit(cache=True)
def _last_swing_prices_strict(high, low, cvd, sh_conf, sl_conf, right=3):
    n = high.size
    lsh = np.zeros(n); lsl = np.zeros(n)
    lsh_cvd = np.zeros(n); lsl_cvd = np.zeros(n)

    curr_sh = high[0]; curr_sl = low[0]
    curr_sh_cvd = cvd[0]; curr_sl_cvd = cvd[0]

    for i in range(n):
        if sh_conf[i] == 1:
            curr_sh = high[i - right]
            curr_sh_cvd = cvd[i - right]
        if sl_conf[i] == 1:
            curr_sl = low[i - right]
            curr_sl_cvd = cvd[i - right]

        lsh[i] = curr_sh; lsl[i] = curr_sl
        lsh_cvd[i] = curr_sh_cvd; lsl_cvd[i] = curr_sl_cvd

    return lsh, lsl, lsh_cvd, lsl_cvd

@njit(cache=True)
def _bos_flags(close, high, low, lsh, lsl):
    n=close.size; bos_up=np.zeros(n,dtype=np.uint8); bos_dn=np.zeros(n,dtype=np.uint8)
    sweep_up=np.zeros(n,dtype=np.uint8); sweep_dn=np.zeros(n,dtype=np.uint8)
    for i in range(n):
        if close[i] > lsh[i]: bos_up[i]=1
        if close[i] < lsl[i]: bos_dn[i]=1
        if high[i] > lsh[i] and close[i] < lsh[i]: sweep_up[i]=1
        if low[i] < lsl[i] and close[i] > lsl[i]: sweep_dn[i]=1
    return bos_up, bos_dn, sweep_up, sweep_dn

@njit(cache=True)
def _inside_engulf_pin(o, h, l, c):
    n=c.size; eb=np.zeros(n,dtype=np.uint8); ebr=np.zeros(n,dtype=np.uint8); pb=np.zeros(n,dtype=np.uint8); pbr=np.zeros(n,dtype=np.uint8); ib=np.zeros(n,dtype=np.uint8)
    for i in range(1, n):
        if c[i-1]<o[i-1] and c[i]>o[i] and c[i]>=o[i-1] and o[i]<=c[i-1]: eb[i]=1
        if c[i-1]>o[i-1] and c[i]<o[i] and o[i]>=c[i-1] and c[i]<=o[i-1]: ebr[i]=1
        body=abs(c[i]-o[i]); rng=h[i]-l[i]+1e-12
        uw=h[i]-max(c[i],o[i]); lw=min(c[i],o[i])-l[i]
        if uw>2*body and uw>0.6*rng: pbr[i]=1
        if lw>2*body and lw>0.6*rng: pb[i]=1
        if h[i]<=h[i-1] and l[i]>=l[i-1]: ib[i]=1
    return eb, ebr, pb, pbr, ib

@njit(cache=True)
def _fvg_flags(h, l):
    n=h.size; fu=np.zeros(n,dtype=np.uint8); fd=np.zeros(n,dtype=np.uint8)
    for i in range(2, n):
        if l[i]>h[i-2]: fu[i]=1
        if h[i]<l[i-2]: fd[i]=1
    return fu, fd

@njit(cache=True)
def _find_ob_zones_strict(o, c, h, l, bos_up, bos_dn):
    n = len(c); obt = np.zeros(n); obb = np.zeros(n)
    bars_since_ob_bull = np.zeros(n); bars_since_ob_bear = np.zeros(n)
    lbu_top = 0.0; lbu_bot = 0.0
    lbe_bot = 0.0; lbe_top = 0.0
    bull_age = 0; bear_age = 0

    for i in range(2, n - 1):
        if bos_up[i] == 1 and bos_up[i-1] == 0:
            for j in range(i-1, max(0, i-10), -1):
                if c[j] < o[j]: 
                    if j+1 < n:
                        body_ob = o[j] - c[j]; next_body = c[j+1] - o[j+1]
                        if c[j+1] > o[j+1] and next_body > 1.5 * body_ob:
                            lbu_top = float(h[j])
                            lbu_bot = float(l[j])
                            bull_age = 0
                    break
        if bos_dn[i] == 1 and bos_dn[i-1] == 0:
            for j in range(i-1, max(0, i-10), -1):
                if c[j] > o[j]: 
                    if j+1 < n:
                        body_ob = c[j] - o[j]; next_body = o[j+1] - c[j+1]
                        if c[j+1] < o[j+1] and next_body > 1.5 * body_ob:
                            lbe_bot = float(l[j])
                            lbe_top = float(h[j])
                            bear_age = 0
                    break

        if lbu_top > 0: bull_age += 1
        if lbe_bot > 0: bear_age += 1

        # Mitigation / Invalidation logic to prevent stale limit orders
        if lbu_top > 0 and c[i] < lbu_bot:
            lbu_top = 0.0; lbu_bot = 0.0; bull_age = 0
        if lbe_bot > 0 and c[i] > lbe_top:
            lbe_bot = 0.0; lbe_top = 0.0; bear_age = 0

        obt[i] = lbu_top; obb[i] = lbe_bot
        bars_since_ob_bull[i] = bull_age
        bars_since_ob_bear[i] = bear_age

    return obt, obb, bars_since_ob_bull, bars_since_ob_bear

class Precomp:
    def __init__(self, df):
        self.o=df.open.values; self.h=df.high.values; self.l=df.low.values; self.c=df.close.values; self.v=df.volume.values
        self.open=self.o; self.high=self.h; self.low=self.l; self.close=self.c
        
        self.cvd = _synthetic_cvd_np(self.o, self.h, self.l, self.c, self.v) 
        
        sh_conf, sl_conf = _swing_points_strict(self.h, self.l, 3, 3)
        self.swing_hi = sh_conf; self.swing_lo = sl_conf
        self.last_sh, self.last_sl, self.last_sh_cvd, self.last_sl_cvd = _last_swing_prices_strict(self.h, self.l, self.cvd, sh_conf, sl_conf, 3)
        self.bos_up, self.bos_down, self.sweep_up, self.sweep_dn = _bos_flags(self.c, self.h, self.l, self.last_sh, self.last_sl)
        self.engulf_bull, self.engulf_bear, self.pin_bull, self.pin_bear, self.inside_bar = _inside_engulf_pin(self.o, self.h, self.l, self.c)
        self.fvg_up, self.fvg_dn = _fvg_flags(self.h, self.l)
        self.ob_bull, self.ob_bear, self.bars_since_ob_bull, self.bars_since_ob_bear = _find_ob_zones_strict(self.o, self.c, self.h, self.l, self.bos_up, self.bos_down)
        
        self.ema50 = _ema_np(self.c, 50); self.ema200 = _ema_np(self.c, 200)
        self.atr14 = _atr_np(self.h, self.l, self.c, 14); self.rsi14 = _rsi14_np(self.c)
        self.macd, self.macd_sig, self.macd_hist = _macd_np(self.c)
        self.eff_ratio = _efficiency_ratio(self.c, 10)
        self.div_bull, self.div_bear = _detect_divergence(self.c, self.rsi14, self.last_sl, self.last_sh)
        self.vol_ma = _rolling_mean(self.v, 20); self.vol_spike = (self.v > self.vol_ma * 1.5).astype(np.float64)
        self.kalman_price, self.kalman_vel = _kalman_smooth(self.c)
        self.hurst = _calc_hurst(self.c, window=100)
        self.ema20 = _ema_np(self.c, 20)

        self.adx = _adx_np(self.h, self.l, self.c, 14)
        self.rvol = _rvol_np(self.v, 20) 
        self.bb_upper, self.bb_lower, self.bb_width = _bb_bands_np(self.c, 20, 2.0)
        
        self.w_pattern, self.m_pattern, self.fvg_bull_p, self.fvg_bear_p, self.sweep_bull_p, self.sweep_bear_p = _detect_patterns(
            self.o, self.h, self.l, self.c, self.swing_hi, self.swing_lo, self.last_sh, self.last_sl
        )

        # SYNTHETIC MICRO-STRUCTURE UPDATES
        self.kc_upper, self.kc_lower = _keltner_channels_np(self.c, self.h, self.l, 20, 1.5)
        self.is_squeezed = (self.bb_upper < self.kc_upper) & (self.bb_lower > self.kc_lower)
        self.squeeze_fired = np.zeros(len(self.c), dtype=np.float64)
        if len(self.c) > 1:
            self.squeeze_fired[1:] = (~self.is_squeezed[1:]) & self.is_squeezed[:-1]

        typical_price = (self.h + self.l + self.c) / 3.0
        self.avwap_bull = _avwap_np(typical_price, self.v, self.swing_lo)
        self.avwap_bear = _avwap_np(typical_price, self.v, self.swing_hi)
        self.poc, self.vah, self.val = _volume_profile_np(self.c, self.v, 100, 20)
        self.cvd_div_bull, self.cvd_div_bear = _cvd_divergence_np(self.c, self.cvd, self.last_sl, self.last_sh, self.last_sl_cvd, self.last_sh_cvd)
        self.cvd_roc = _roc(self.cvd, 5)
        self.cvd_acceleration = _roc(self.cvd_roc, 5)

        hours = pd.to_datetime(df.timestamp, unit='ms', utc=True).dt.hour.values
        days = pd.to_datetime(df.timestamp, unit='ms', utc=True).dt.dayofyear.values
        self.asian_high, self.asian_low = _asian_range_np(self.h, self.l, hours, days)

        self.atr_percentile_100 = _rolling_percentile_np(self.atr14, 100)
        self.rolling_vwap_20 = _rolling_vwap_np((self.h + self.l + self.c) / 3.0, self.v, 20)
        self.dc_high_20, self.dc_low_20 = _donchian_channels_np(self.h, self.l, 20)

def _trend_flags(e50, e200): return (1.0, 0.0) if e50>e200 else ((0.0, 1.0) if e50<e200 else (0.0, 0.0))
def _ensure_precomp(df, pre): return pre if isinstance(pre, Precomp) else Precomp(df)

def confluence_features(cfg, htf, mtf, ltf, iH, iM, iL, precomp=None, extras=None):
    if iH is None: iH=len(htf)-1; iM=len(mtf)-1; iL=len(ltf)-1
    
    idx_H = max(0, iH - 1)
    idx_M = max(0, iM - 1)
    idx_L = iL 
    
    ph=_ensure_precomp(htf, precomp.get("htf") if precomp else None)
    pm=_ensure_precomp(mtf, precomp.get("mtf") if precomp else None)
    pl=_ensure_precomp(ltf, precomp.get("ltf") if precomp else None)
    
    f={}
    f["htf_up"], f["htf_down"] = _trend_flags(ph.ema50[idx_H], ph.ema200[idx_H])
    f["mtf_up"], f["mtf_down"] = _trend_flags(pm.ema50[idx_M], pm.ema200[idx_M])
    f["htf_ema50"] = float(ph.ema50[idx_H])
    
    px = pl.c[idx_L]; f["price"]=float(px); 
    
    f["atr_htf_pct"]=float(ph.atr14[idx_H]/max(px,1e-9)*100.0)
    f["atr_mtf_pct"]=float(pm.atr14[idx_M]/max(px,1e-9)*100.0)
    f["atr_ltf_pct"]=float(pl.atr14[idx_L]/max(px,1e-9)*100.0)
    
    f["last_swing_high"] = float(pl.last_sh[idx_L])
    f["last_swing_low"] = float(pl.last_sl[idx_L])
    f["ob_bull_price"] = float(pl.ob_bull[idx_L])
    f["ob_bear_price"] = float(pl.ob_bear[idx_L])
    
    px_safe = px if px > 1e-12 else 1.0
    f["dist_last_sh_pct"] = (px - f["last_swing_high"]) / px_safe * 100
    f["dist_last_sl_pct"] = (px - f["last_swing_low"]) / px_safe * 100
    f["dist_ob_bull_pct"] = (px - f["ob_bull_price"]) / px_safe * 100
    f["dist_ob_bear_pct"] = (px - f["ob_bear_price"]) / px_safe * 100

    atr_abs = ph.atr14[idx_H]
    if atr_abs < 1e-9: atr_abs = px * 0.01
    current_range = pl.h[idx_L] - pl.l[idx_L]
    f["candle_range_atr"] = float(current_range / atr_abs)
    
    for k in ["bos_up","bos_down","fvg_up","fvg_dn","engulf_bull","engulf_bear","pin_bull","pin_bear","inside_bar"]:
        f[k]=float(getattr(pl,k)[idx_L])
    f["sweep_high"] = float(pl.sweep_up[idx_L])
    f["sweep_low"] = float(pl.sweep_dn[idx_L])
    
    f["choch_up"]=float(1.0 if pl.bos_up[idx_L] and idx_L>=5 and pl.bos_down[idx_L-5] else 0.0)
    f["choch_down"]=float(1.0 if pl.bos_down[idx_L] and idx_L>=5 and pl.bos_up[idx_L-5] else 0.0)
    f["ob_bull_dist"] = (px - pl.ob_bull[idx_L])/px*100 if pl.ob_bull[idx_L]>0 else 0.0
    f["ob_bear_dist"] = (pl.ob_bear[idx_L] - px)/px*100 if pl.ob_bear[idx_L]>0 else 0.0
    f["bars_since_ob_bull"] = float(pl.bars_since_ob_bull[idx_L])
    f["bars_since_ob_bear"] = float(pl.bars_since_ob_bear[idx_L])

    # Safely pull velocity_atr_3 if precomputed, else 0
    if hasattr(pl, 'velocity_atr_3'):
        f["velocity_atr_3"] = float(pl.velocity_atr_3[idx_L])

    f["vol_spike"] = float(pl.vol_spike[idx_L])
    f["kalman_vel"] = float(pl.kalman_vel[idx_L]); f["hurst"] = float(pl.hurst[idx_L])
    
    f["cvd"] = float(pl.cvd[idx_L])
    f["adx"] = float(pl.adx[idx_L])
    f["bb_width"] = float(pl.bb_width[idx_L])
    f["rvol"] = float(pl.rvol[idx_L]) 
    f["bb_upper"] = float(pl.bb_upper[idx_L])
    f["bb_lower"] = float(pl.bb_lower[idx_L])
    f["kc_upper"] = float(pl.kc_upper[idx_L])
    f["kc_lower"] = float(pl.kc_lower[idx_L])

    f["dist_bb_upper_pct"] = (px - f["bb_upper"]) / px_safe * 100
    f["dist_bb_lower_pct"] = (px - f["bb_lower"]) / px_safe * 100
    f["rsi"] = float(pl.rsi14[idx_L])
    
    f["w_pattern"] = float(pl.w_pattern[idx_L])
    f["m_pattern"] = float(pl.m_pattern[idx_L])
    f["fvg_bull"] = float(pl.fvg_bull_p[idx_L])
    f["fvg_bear"] = float(pl.fvg_bear_p[idx_L])
    f["sweep_bull_p"] = float(pl.sweep_bull_p[idx_L])
    f["sweep_bear_p"] = float(pl.sweep_bear_p[idx_L])

    # MICRO-STRUCTURE EXTRACTION
    f["avwap_bull"] = float(pl.avwap_bull[idx_L])
    f["avwap_bear"] = float(pl.avwap_bear[idx_L])
    f["poc"] = float(pl.poc[idx_L])
    f["vah"] = float(pl.vah[idx_L])
    f["val"] = float(pl.val[idx_L])

    f["dist_avwap_bull_pct"] = (px - f["avwap_bull"]) / px_safe * 100
    f["dist_avwap_bear_pct"] = (px - f["avwap_bear"]) / px_safe * 100
    f["dist_poc_pct"] = (px - f["poc"]) / px_safe * 100
    f["dist_vah_pct"] = (px - f["vah"]) / px_safe * 100
    f["dist_val_pct"] = (px - f["val"]) / px_safe * 100

    f["is_squeezed"] = float(pl.is_squeezed[idx_L])
    f["squeeze_fired"] = float(pl.squeeze_fired[idx_L])
    f["mtf_is_squeezed"] = float(pm.is_squeezed[idx_M])
    f["mtf_squeeze_fired"] = float(pm.squeeze_fired[idx_M])
    f["cvd_div_bull"] = float(pl.cvd_div_bull[idx_L])
    f["cvd_div_bear"] = float(pl.cvd_div_bear[idx_L])
    f["cvd_roc"] = float(pl.cvd_roc[idx_L])
    f["cvd_acceleration"] = float(pl.cvd_acceleration[idx_L])

    f["asian_high"] = float(pl.asian_high[idx_L])
    f["asian_low"] = float(pl.asian_low[idx_L])
    f["dc_high_20"] = float(pl.dc_high_20[idx_L])
    f["dc_low_20"] = float(pl.dc_low_20[idx_L])
    
    f["dist_asian_high_pct"] = (px - f["asian_high"]) / px_safe * 100
    f["dist_asian_low_pct"] = (px - f["asian_low"]) / px_safe * 100

    # SMC & Fibonacci Deep Wick Geometry
    fractal_range = f["last_swing_high"] - f["last_swing_low"]
    if fractal_range > 0:
        f["fib_786_long"] = f["last_swing_low"] + (fractal_range * 0.214)
        f["fib_886_long"] = f["last_swing_low"] + (fractal_range * 0.114)
        f["fib_786_short"] = f["last_swing_high"] - (fractal_range * 0.214)
        f["fib_886_short"] = f["last_swing_high"] - (fractal_range * 0.114)
    else:
        f["fib_786_long"] = f["fib_886_long"] = f["fib_786_short"] = f["fib_886_short"] = px

    f["dist_fib_786_long_pct"] = (px - f["fib_786_long"]) / px_safe * 100
    f["dist_fib_886_long_pct"] = (px - f["fib_886_long"]) / px_safe * 100
    f["dist_fib_786_short_pct"] = (px - f["fib_786_short"]) / px_safe * 100
    f["dist_fib_886_short_pct"] = (px - f["fib_886_short"]) / px_safe * 100

    # Check if Asian Range was just swept
    f["asian_range_swept_up"] = 1.0 if pl.h[idx_L] > pl.asian_high[idx_L] and pl.c[idx_L] < pl.asian_high[idx_L] else 0.0
    f["asian_range_swept_dn"] = 1.0 if pl.l[idx_L] < pl.asian_low[idx_L] and pl.c[idx_L] > pl.asian_low[idx_L] else 0.0
    
    ts = ltf.timestamp.iloc[idx_L]
    f["hour_of_day"] = float((int(ts) // 3600000) % 24)
    
    if extras:
        f["funding_rate"] = float(extras.get("funding_rate", 0.0))
        f["macro_sentiment"] = float(extras.get("macro_sentiment", 0.0))
        f["btc_bullish"] = float(extras.get("btc_bullish", 1.0))
        f["btcd_trend"] = float(extras.get("btcd_trend", 0.0))
        f["dxy_trend"] = float(extras.get("dxy_trend", 0.0))
        f["spx_trend"] = float(extras.get("spx_trend", 0.0))
        f["delta_oi"] = float(extras.get("delta_oi", 0.0))
        f["bid_ask_imbalance"] = float(extras.get("bid_ask_imbalance", 0.5))
    else:
        f["funding_rate"] = 0.0; f["macro_sentiment"] = 0.0; f["btc_bullish"] = 1.0; f["btcd_trend"] = 0.0
        f["dxy_trend"] = 0.0; f["spx_trend"] = 0.0; f["delta_oi"] = 0.0
        f["bid_ask_imbalance"] = 0.5

    # FIX FEATURE LEAK: Generate Interaction features in the core dictionary so they export correctly
    f["inter_htf_bos_up"] = f.get("htf_up",0) * f.get("bos_up",0)
    f["inter_mtf_down_sweep_high"] = f.get("mtf_down",0) * f.get("sweep_high",0)
    ema20_gt = 1.0 if (pl.ema20[idx_L] > _ema_np(pl.c, 100)[idx_L] if hasattr(pl, 'ema20') else 0) else 0.0
    f["trend_align_up_3tf"] = f.get("htf_up",0) + f.get("mtf_up",0) + ema20_gt
    f["trend_align_down_3tf"] = f.get("htf_down",0) + f.get("mtf_down",0) + (1.0 - ema20_gt)

    # ---------------------------------------------------------
    # DIRECTIVE 1: THE REGIME DETECTION MATRIX
    # ---------------------------------------------------------
    adx_val = f["adx"]
    atr_p = pl.atr_percentile_100[idx_L] * 100.0
    htf_trend = "UP" if f["htf_up"] > 0 else ("DOWN" if f["htf_down"] > 0 else "NEUTRAL")

    is_bb_outside = (px > f["bb_upper"]) or (px < f["bb_lower"])
    has_cvd_div = (f["cvd_div_bull"] > 0) or (f["cvd_div_bear"] > 0)

    market_regime = "CONSOLIDATION"

    if is_bb_outside and has_cvd_div:
        market_regime = "REVERSAL_WARNING"
    elif htf_trend == "UP" and f["price"] < pl.rolling_vwap_20[idx_L]:
        market_regime = "RETRACEMENT"
    elif htf_trend == "DOWN" and f["price"] > pl.rolling_vwap_20[idx_L]:
        market_regime = "RETRACEMENT"
    elif adx_val >= 25 and atr_p >= 50:
        market_regime = "EXPANSION"
    elif adx_val < 20 and atr_p < 40:
        market_regime = "CONSOLIDATION"
    else:
        market_regime = "CONSOLIDATION" # Default

    f["market_regime"] = market_regime

    return f

def precompute_v6_features(ph, pm, pl, htf, mtf, ltf, btc_close_arr=None):
    cl=pl.c; f={}
    if cl.size==0: return {}
    eps=1e-12
    f['ema20_L']=_ema_np(cl,20)
    f['ema50_L']=pl.ema50
    f['ema100_L']=_ema_np(cl,100)

    f['dist_ema20_pct']=(cl - f['ema20_L']) / (np.abs(cl)+eps) * 100
    f['dist_ema50_pct']=(cl - f['ema50_L']) / (np.abs(cl)+eps) * 100
    f['dist_ema100_pct']=(cl - f['ema100_L']) / (np.abs(cl)+eps) * 100

    f['ts_50_200']=(pl.ema50-pl.ema200)/(np.abs(cl)+eps)
    f['ts_20_100']=(_ema_np(cl,20)-_ema_np(cl,100))/(np.abs(cl)+eps)
    f['ts_roll_mean_10']=_rolling_mean(f['ts_50_200'],10)
    f['ts_roll_std_10']=_rolling_std(f['ts_50_200'],10)
    f['ts_roc_5']=_roc(f['ts_50_200'],5)
    f['ema20_gt_ema100']=(_ema_np(cl,20)>_ema_np(cl,100)).astype(float)
    f['bb_width_pct_20']=pl.bb_width 
    f['atr14_L']=pl.atr14
    f['atr_pct_ltf']=pl.atr14/(np.abs(cl)+eps)*100
    f['rsi14_ltf']=pl.rsi14
    f['vol_ratio_200']=f['atr_pct_ltf']/_rolling_mean(f['atr_pct_ltf'],200,20)
    f['vol_ratio_200']=np.nan_to_num(f['vol_ratio_200'],nan=1.0)
    mtf_ema200_s = pd.Series(pm.ema200, index=mtf.timestamp).shift(1).reindex(ltf.timestamp, method='ffill').fillna(0).values
    f['mtf_ema200_arr']=mtf_ema200_s
    f['dist_mtf_ema200_pct']=np.where(cl>0, (cl - mtf_ema200_s) / cl * 100, 0.0)
    cm=pm.c
    f['trend_strength_mtf_arr']=(pm.ema50-pm.ema200)/(np.abs(cm)+eps)
    f['atr_pct_ltf_roc_5_arr']=_roc(f['atr_pct_ltf'],5)
    f['vol_ltf_accel_5_arr']=_roc(f['atr_pct_ltf_roc_5_arr'],5)
    f['trend_strength_ltf_ema20_50_arr']=(_ema_np(cl,20)-pl.ema50)/(np.abs(cl)+eps)
    
    htf_s = pd.Series(ph.atr14/ph.c*100, index=htf.timestamp).shift(1).reindex(ltf.timestamp, method='ffill').fillna(0).values
    f['atr_pct_htf_aligned']=htf_s
    mtf_s = pd.Series(f['trend_strength_mtf_arr'], index=mtf.timestamp).shift(1).reindex(ltf.timestamp, method='ffill').fillna(0).values
    f['trend_strength_mtf_aligned']=mtf_s
    
    # INJECT ALIGNED HTF/MTF FRACTAL SWINGS
    f['htf_swing_high'] = pd.Series(ph.last_sh, index=htf.timestamp).shift(1).reindex(ltf.timestamp, method='ffill').fillna(0.0).values
    f['htf_swing_low'] = pd.Series(ph.last_sl, index=htf.timestamp).shift(1).reindex(ltf.timestamp, method='ffill').fillna(0.0).values
    f['mtf_swing_high'] = pd.Series(pm.last_sh, index=mtf.timestamp).shift(1).reindex(ltf.timestamp, method='ffill').fillna(0.0).values
    f['mtf_swing_low'] = pd.Series(pm.last_sl, index=mtf.timestamp).shift(1).reindex(ltf.timestamp, method='ffill').fillna(0.0).values

    htf_tr = np.where(ph.ema50>ph.ema200,1,np.where(ph.ema50<ph.ema200,-1,0))
    htf_tr_s = pd.Series(htf_tr, index=htf.timestamp).shift(1).reindex(ltf.timestamp, method='ffill').fillna(0).values.astype(int)
    f['bars_since_htf_trend_flip_arr']=_bars_since_change(htf_tr_s)
    f['atr7_L']=_atr_np(pl.h,pl.l,pl.c,7)
    f['atr_pct_ltf_7_arr']=f['atr7_L']/(np.abs(cl)+eps)*100
    
    f['dist_to_bull_ob_arr'] = np.where(pl.ob_bull>0, (cl-pl.ob_bull)/cl*100, 0.0)
    f['dist_to_bear_ob_arr'] = np.where(pl.ob_bear>0, (pl.ob_bear-cl)/cl*100, 0.0)
    f['is_sweep_high_arr'] = pl.sweep_up.astype(float)
    f['is_sweep_low_arr'] = pl.sweep_dn.astype(float)
    
    l_mom = np.where(pl.rsi14>50,0.5,-0.5)+np.where(pl.macd>pl.macd_sig,0.5,-0.5)
    m_mom_s = pd.Series(np.where(pm.rsi14>50,0.5,-0.5)+np.where(pm.macd>pm.macd_sig,0.5,-0.5), index=mtf.timestamp).shift(1).reindex(ltf.timestamp, method='ffill').fillna(0).values
    h_mom_s = pd.Series(np.where(ph.rsi14>50,0.5,-0.5)+np.where(ph.macd>ph.macd_sig,0.5,-0.5), index=htf.timestamp).shift(1).reindex(ltf.timestamp, method='ffill').fillna(0).values
    f['momentum_alignment_score_arr'] = l_mom + m_mom_s + h_mom_s
    
    f['market_efficiency_ratio'] = pl.eff_ratio
    rng = pl.last_sh - pl.last_sl
    safe_rng = np.where(rng == 0, 1.0, rng)
    f['premium_discount_index'] = np.where(rng > 0, (cl - pl.last_sl) / safe_rng, 0.5)

    f['div_bull_arr'] = pl.div_bull
    f['div_bear_arr'] = pl.div_bear
    
    if btc_close_arr is not None and len(btc_close_arr) == len(cl):
        safe_btc = np.where(btc_close_arr == 0, 1.0, btc_close_arr)
        eth_btc_ratio = cl / safe_btc
        f['eth_btc_rsi'] = _rsi14_np(eth_btc_ratio)
        ema20_ratio = _ema_np(eth_btc_ratio, 20)
        ema100_ratio = _ema_np(eth_btc_ratio, 100)
        f['eth_btc_trend_score'] = (ema20_ratio - ema100_ratio) / (ema100_ratio + eps) * 1000.0 
        price_weak = cl < f['ema50_L']
        ratio_strong = eth_btc_ratio > _ema_np(eth_btc_ratio, 50)
        f['rel_strength_divergence'] = np.where(price_weak & ratio_strong, 1.0, 0.0)
    else:
        f['eth_btc_rsi'] = np.full(len(cl), 50.0)
        f['eth_btc_trend_score'] = np.zeros(len(cl))
        f['rel_strength_divergence'] = np.zeros(cl)

    # ---------------------------------------------------------
    # DIRECTIVE 1: MULTI-TIMEFRAME VOLATILITY SQUEEZE
    # ---------------------------------------------------------
    bbw_arr = pl.bb_width
    kc_upper_arr, kc_lower_arr = _keltner_channels_np(cl, pl.h, pl.l, 20, 1.5)
    ema20_arr = _ema_np(cl, 20)
    kcw_arr = np.where(ema20_arr > 0, (kc_upper_arr - kc_lower_arr) / ema20_arr * 100.0, 0.0)

    f['is_ttm_squeeze'] = np.where(bbw_arr < kcw_arr, 1.0, 0.0)

    # Linear regression Z-score (squeeze_momentum)
    # y = mx + c -> regression over 20 periods
    n = len(cl)
    squeeze_mom = np.zeros(n)
    for i in range(20, n):
        chunk = cl[i-20+1:i+1]
        x = np.arange(20)
        slope, intercept = np.polyfit(x, chunk, 1)
        pred = slope * 19 + intercept
        std_err = np.std(chunk - (slope * x + intercept))
        if std_err > 1e-9:
            squeeze_mom[i] = (cl[i] - pred) / std_err
        else:
            squeeze_mom[i] = 0.0
    f['squeeze_momentum'] = squeeze_mom

    # Volatility Expansion Filter: ATR Percentile (100-period)
    f['atr_percentile_100'] = pl.atr_percentile_100 * 100.0

    # ---------------------------------------------------------
    # DIRECTIVE 2: ICT & MICROSTRUCTURE PROXIES
    # ---------------------------------------------------------
    # rolling 20-bar Swing High/Low
    rolling_sh = pd.Series(pl.h).rolling(20, min_periods=1).max().to_numpy()
    rolling_sl = pd.Series(pl.l).rolling(20, min_periods=1).min().to_numpy()
    atr_safe = np.where(pl.atr14 > 1e-9, pl.atr14, cl * 0.01)

    f['dist_to_rolling_sh_atr'] = (rolling_sh - cl) / atr_safe
    f['dist_to_rolling_sl_atr'] = (cl - rolling_sl) / atr_safe

    # Wick Rejection Ratio
    total_range = pl.h - pl.l
    total_range_safe = np.where(total_range > 1e-9, total_range, 1e-9)
    body_top = np.maximum(pl.o, cl)
    body_bottom = np.minimum(pl.o, cl)
    f['upper_wick_pct'] = (pl.h - body_top) / total_range_safe
    f['lower_wick_pct'] = (body_bottom - pl.l) / total_range_safe

    # Killzone Session Flags
    hours = pd.to_datetime(ltf.timestamp, unit='ms', utc=True).dt.hour.values
    f['is_london_killzone'] = np.where((hours >= 7) & (hours <= 10), 1.0, 0.0)
    f['is_ny_killzone'] = np.where((hours >= 13) & (hours <= 16), 1.0, 0.0)
    f['is_asian_range'] = np.where((hours >= 0) & (hours <= 6), 1.0, 0.0)

    # Wyckoff Volume Absorption (Directive 3)
    candle_spread = pl.h - pl.l
    body_size = np.abs(cl - pl.o)
    f['vol_absorption'] = np.where((pl.rvol > 1.5) & ((body_size / (candle_spread + 1e-9)) < 0.3), 1.0, 0.0)

    # Time at Mode (Liquidity Consumption proxy)
    # Measures how many bars in the last 20 were within 0.5 ATR of the VWAP
    diff_from_vwap = np.abs(cl - pl.rolling_vwap_20)
    is_near_vwap = np.where(diff_from_vwap <= atr_safe * 0.5, 1.0, 0.0)
    f['time_at_mode'] = pd.Series(is_near_vwap).rolling(20, min_periods=1).sum().to_numpy()

    # ---------------------------------------------------------
    # DIRECTIVE 3: VWAP & MEAN REVERSION EXTREMES
    # ---------------------------------------------------------
    f['vwap_z_score'] = (cl - pl.rolling_vwap_20) / atr_safe

    # Liquidation Cascade Velocity Filter (Anti-Falling Knife)
    # Measures the speed of the price drop/rally over the last 3 bars
    cl_shifted_3 = pd.Series(cl).shift(3).fillna(cl[0]).to_numpy()
    velocity_atr_3_arr = np.abs(cl - cl_shifted_3) / atr_safe
    f['velocity_atr_3'] = velocity_atr_3_arr
    pl.velocity_atr_3 = velocity_atr_3_arr

    # ---------------------------------------------------------
    # DIRECTIVE 4: STRIP COLLINEAR NOISE
    # ---------------------------------------------------------
    keys_to_drop = [
        'ema20_L', 'ema50_L', 'ema100_L',
        'mtf_ema200_arr', 'atr14_L', 'atr7_L',
        'htf_swing_high', 'htf_swing_low',
        'mtf_swing_high', 'mtf_swing_low'
    ]
    for k in keys_to_drop:
        if k in f:
            del f[k]

    return f

class BrainLearningManager:
    def __init__(self, cfg, long_p=None, short_p=None):
        self.cfg=cfg; self.brains={'long':None, 'short':None}
        if not joblib: return
        self.load_brains(long_p, short_p)

    def load_brains(self, long_p, short_p):
        for s, p in [('long',long_p), ('short',short_p)]:
            if p:
                try: self.brains[s] = joblib.load(p)
                except Exception as e: print(f"Load error {s}: {e}")

    def predict_prob(self, side, base, adv, iH, iM, iL, px, pre_l):
        b = self.brains.get(side)
        if not b: return 0.0
        try:
            vec = self._build_vec(side, base, adv, iL, iM, px, pre_l, b['feature_names'])
            # CRITICAL FIX: Clip all array values before casting to prevent float32 memory overflow
            vec = np.clip(np.nan_to_num(vec, nan=0.0), -1e10, 1e10).astype(np.float32)
            if 'imputer' in b: vec = b['imputer'].transform(vec)
            vec_s = b['scaler'].transform(vec)
            p = b['classifier'].predict_proba(vec_s)[0][1]
            return 1.0-p if b.get('invert_prob') else p
        except: return 0.0

    def _build_vec(self, side, b, a, iL, iM, px, pl, fnames):
        d = {**b} 
        for k, v in a.items():
            if hasattr(v, '__getitem__') and len(v) > iL:
                val = v[iL]
                d[k] = float(val) if np.isfinite(val) else 0.0
                if d[k] > 1e6: d[k] = 1e6
                elif d[k] < -1e6: d[k] = -1e6
            else: d[k] = 0.0
        d.update({
            "pos_vs_swing_h": (px-pl.last_sh[iL])/px if px else 0.0, 
            "pos_vs_swing_l": (px-pl.last_sl[iL])/px if px else 0.0,
            "dist_to_mtf_ema200_pct": (px-a['mtf_ema200_arr'][iM])/px*100 if px and 'mtf_ema200_arr' in a else 0.0,
            "side": 1.0 if side=="long" else 0.0
        })
        return np.array([d.get(n, 0.0) for n in fnames]).reshape(1,-1)

class TradeManager:
    def __init__(self, cfg, exw, mem, brain, narrative_map=None):
        self.cfg = cfg
        self.exw = exw
        self.mem = mem
        self.brain = brain
        self.open_trades = []
        self.pending_orders = []
        self.last_signal_time = {}
        self.current_bar_index = 0

    def can_open(self, side):
        total_active = len(self.open_trades) + len(self.pending_orders)
        if self.cfg.max_concurrent > 0 and total_active >= self.cfg.max_concurrent: return False
        return True

    def submit_plan(self, plan, bar):
        if not plan or not self.can_open(plan['side']): return None
        
        # Signal cooldown to prevent trade stacking
        signal_key = f"{plan.get('strat', plan.get('strategy', 'unknown'))}_{plan['side']}"
        last_signal = self.last_signal_time.get(signal_key, -999)

        # Reject identical signals for the next 10 bars
        if self.current_bar_index - last_signal < 10:
            return None

        self.last_signal_time[signal_key] = self.current_bar_index

        adx = plan.get('features', {}).get('adx', 25.0)
        ttl = 12 
        
        if adx > 40: ttl = 6      
        elif adx < 20: ttl = 16   
        
        order = {
            **plan,
            "dca_level": 0,
            "risk_factor": plan.get('risk_factor', 1.0),
            "created_ts": bar.get("timestamp", 0),
            "ttl": ttl, 
            "status": "PENDING" 
        }
        
        self.pending_orders.append(order)
        self.mem.record_signal(order)
        return order

    def step_bar(self, o, h, l, c, ts=None, swing_high=0.0, swing_low=0.0):
        self.current_bar_index += 1
        closed = []
        still_open = []
        still_pending = []
        
        # Determine the current bar's timestamp, fallback to 0 if not provided
        current_ts = ts if ts is not None else 0

        fee = (self.cfg.taker_fee_bps + self.cfg.slippage_bps) / 10000.0
        trail_trig = self.cfg.trailing_stop_trigger_r
        trail_dist = self.cfg.trailing_dist_r
        
        for order in self.pending_orders:
            triggered = False
            fill_px = 0.0
            
            order['ttl'] -= 1
            if order['ttl'] < 0:
                continue 
            
            side = order['side']
            entry_target = order['entry']
            # PATCH: DYNAMIC EXECUTION STYLE INHERITANCE
            order_type = order.get('type', self.cfg.execution_style)
            
            if order_type == 'limit':
                if side == 'long':
                    if l <= entry_target:
                        triggered = True
                        fill_px = min(o, entry_target)
                elif side == 'short':
                    if h >= entry_target:
                        triggered = True
                        fill_px = max(o, entry_target)
            else:
                if side == 'long':
                    if h >= entry_target:
                        triggered = True
                        fill_px = max(o, entry_target)
                elif side == 'short':
                    if l <= entry_target:
                        triggered = True
                        fill_px = min(o, entry_target)
            
            if triggered:
                planned_risk = abs(order['entry'] - order['sl'])
                if planned_risk < 1e-9:
                    planned_risk = order['entry'] * 0.01

                initial_risk = planned_risk
                if self.cfg.use_dca: initial_risk /= (self.cfg.dca_max_safety_orders + 1.5)
                
                trade = order.copy()
                trade['avg_price'] = fill_px
                trade['total_size'] = order.get('total_size', 1.0 * order.get('risk_factor',1.0))
                trade['initial_risk_unit'] = initial_risk
                trade['status'] = 'OPEN'
                trade['fill_ts'] = current_ts
                trade['bars_open'] = 0
                
                if self.cfg.use_dca:
                    dist = abs(fill_px - trade['sl'])
                    if side == 'long': 
                        trade['next_safety_price'] = fill_px - dist
                        trade['sl'] -= dist * 1.5
                    else: 
                        trade['next_safety_price'] = fill_px + dist
                        trade['sl'] += dist * 1.5
                else:
                    trade['next_safety_price'] = 0.0

                # PREVENT INTRA-BAR LEAKAGE: A trade cannot hit TP on the same bar it triggers
                # because we do not know if the high/low occurred before or after the trigger.
                # However, it CAN hit SL (pessimistic outcome).
                trade['can_tp_this_bar'] = False

                self.open_trades.append(trade)
                self.mem.record_fill(trade)
            else:
                still_pending.append(order)
        
        self.pending_orders = still_pending

        for t in self.open_trades:
            t['bars_open'] = t.get('bars_open', 0) + 1
            side = t['side']
            entry = t['avg_price']
            sl = t['sl']
            tp = t['tp']
            
            can_tp = t.get('can_tp_this_bar', True)
            if not can_tp:
                # Reset for the next bar
                t['can_tp_this_bar'] = True

            # Real-time PnL calc for Management
            risk = t.get('initial_risk_unit', abs(t['entry'] - t['sl'])) 
            if risk == 0: risk = entry * 0.01
            curr_pnl = (c - entry) * t['total_size'] if side == 'long' else (entry - c) * t['total_size']
            cost = (entry * t['total_size'] + c * t['total_size']) * fee
            t['pnl_r'] = (curr_pnl - cost) / risk
            
            hit_sl = False
            hit_tp = False
            exit_reason = ''
            
            if side == 'long':
                if l <= sl: hit_sl = True; exit_reason = 'sl'
                elif h >= tp and can_tp: hit_tp = True; exit_reason = 'tp'
            else:
                if h >= sl: hit_sl = True; exit_reason = 'sl'
                elif l <= tp and can_tp: hit_tp = True; exit_reason = 'tp'

            # TIME-IN-FORCE DECAY (Institutional Capital Velocity)
            # DIRECTIVE 4: Disabled to allow macro structures to play out without arbitrary time limits.
            # if not hit_sl and not hit_tp:
            #     decay_limit = getattr(self.cfg, 'time_in_force_decay', 8)
            #     if t['bars_open'] > decay_limit and t['pnl_r'] <= 0:
            #         hit_sl = True
            #         t['sl'] = c  # Force execution at close
            #         exit_reason = 'time_decay'

            if hit_sl or hit_tp:
                if hit_sl:
                    if side == 'long': exit_px = min(o, t['sl']) if o < t['sl'] else t['sl']
                    else: exit_px = max(o, t['sl']) if o > t['sl'] else t['sl']
                else:
                    if side == 'long': exit_px = max(o, tp) if o > tp else tp
                    else: exit_px = min(o, tp) if o < tp else tp

                pnl = (exit_px - entry) * t['total_size'] if side == 'long' else (entry - exit_px) * t['total_size']
                cost = (entry * t['total_size'] + exit_px * t['total_size']) * fee
                t['pnl_r'] = (pnl - cost) / risk 
                t['exit_price'] = exit_px
                t['exit_reason'] = exit_reason
                t['exit_ts'] = current_ts
                
                closed.append(t)
                self.mem.record_exit(t)
                continue

            # AUTO-BREAKEVEN
            #if not t.get('be_locked', False) and t['pnl_r'] >= 1.0:
                #if side == 'long': t['sl'] = entry + (entry * fee * 2) 
                #else: t['sl'] = entry - (entry * fee * 2)
                #t['be_locked'] = True
                
            # STRUCTURAL TRAILING
            #if side == 'long' and swing_low > 0 and swing_low > t['sl'] and swing_low < c:
            #    #t['sl'] = swing_low
            #elif side == 'short' and swing_high > 0 and swing_high < t['sl'] and swing_high > c:
            #    t['sl'] = swing_high

            if self.cfg.use_dca and t['dca_level'] < self.cfg.dca_max_safety_orders:
                hit_safe = (side == 'long' and l <= t['next_safety_price']) or (side == 'short' and h >= t['next_safety_price'])
                if hit_safe:
                    fill = t['next_safety_price']
                    add = 1.0 * (self.cfg.dca_volume_scale ** (t['dca_level'] + 1))
                    new_sz = t['total_size'] + add
                    t['avg_price'] = (t['avg_price'] * t['total_size'] + fill * add) / new_sz
                    t['total_size'] = new_sz
                    t['dca_level'] += 1
                    
                    base_risk = abs(t['entry'] - t['sl']) / (self.cfg.dca_max_safety_orders + 1.5)
                    if side == 'long': 
                        t['tp'] = t['avg_price'] + base_risk * self.cfg.dca_tp_scale
                        t['next_safety_price'] -= base_risk
                    else: 
                        t['tp'] = t['avg_price'] - base_risk * self.cfg.dca_tp_scale
                        t['next_safety_price'] += base_risk

            if trail_trig > 0:
                ru = t.get('initial_risk_unit', 0.0)
                if ru == 0: ru = abs(t['entry'] - t['sl'])
                if ru > 0:
                    if side == 'long':
                        curr_r = (h - entry) / ru
                        if curr_r >= trail_trig:
                            new_sl = h - (trail_dist * ru)
                            if new_sl > t['sl']: t['sl'] = new_sl
                    else:
                        curr_r = (entry - l) / ru
                        if curr_r >= trail_trig:
                            new_sl = l + (trail_dist * ru)
                            if new_sl < t['sl']: t['sl'] = new_sl
            
            still_open.append(t)
        
        self.open_trades = still_open
        return closed
def _score_side(f, s):
    if s=='long': return f.get("bos_up",0)+f.get("engulf_bull",0)+f.get("pin_bull",0)+0.5*f.get("htf_up",0)
    else: return f.get("bos_down",0)+f.get("engulf_bear",0)+f.get("pin_bear",0)+0.5*f.get("htf_down",0)

def plan_trade(cfg, f):
    # Backward compatibility stub
    return None

def plan_trade_with_brain(cfg, brain, base, adv, iH, iM, iL, pre):
    if cfg.filter_hurst_strict and base.get("hurst", 0.5) < 0.45:
        return None

    ts = base.get("timestamp", 0)
    hour = base.get("hour_of_day", 0)
    
    if cfg.filter_time_of_day and ts > 0:
        if 2 <= hour <= 6:
            return None

    px = base.get("price")
    if not px: return None
    
    current_atr = pre.atr14[iL] if pre and iL < len(pre.atr14) else (px * 0.01)
    sl_atr = base.get("atr_ltf_pct",0)*0.01*px*cfg.atr_mult_sl
    
    # DYNAMIC TP SCALING
    bb_width = base.get("bb_width", 0.0)
    if getattr(cfg, 'dynamic_tp_enabled', True):
        if bb_width > 5.0: tp_atr_dist = current_atr * 2.5
        else: tp_atr_dist = current_atr * 1.5
    else:
        tp_atr_dist = base.get("atr_ltf_pct",0)*0.01*px*cfg.atr_mult_tp
        
    structure_buffer = 0.5 * current_atr
    
    funding = base.get("funding_rate", 0.0)
    delta_oi = base.get("delta_oi", 0.0) # Passed from CCXT real-time
    btc_bullish = base.get("btc_bullish", 1.0) 
    btcd_trend = base.get("btcd_trend", 0.0) 
    dxy_trend = base.get("dxy_trend", 0.0)
    spx_trend = base.get("spx_trend", 0.0)
    bid_ask_imbalance = base.get("bid_ask_imbalance", 0.5)
    
    adx_val = base.get("adx", 25.0)
    rvol = base.get("rvol", 1.0)
    rsi = base.get("rsi", 50.0)
    cvd_roc = base.get("cvd_roc", 0.0)
    poc = base.get("poc", px)
    vah = base.get("vah", px)
    val = base.get("val", px)
    
    if base.get("candle_range_atr", 0.0) > 1.5: return None
        
    can_long = True
    can_short = True
    if rsi > 70: can_long = False
    if rsi < 30: can_short = False
    
    # SPOOFING PROTECTION
    if bid_ask_imbalance > 0.80: can_long = False
    if bid_ask_imbalance < 0.20: can_short = False
    
    w_pattern = base.get("w_pattern", 0.0); m_pattern = base.get("m_pattern", 0.0)
    fvg_bull = base.get("fvg_bull", 0.0); fvg_bear = base.get("fvg_bear", 0.0)
    sweep_bull = base.get("sweep_bull_p", 0.0); sweep_bear = base.get("sweep_bear_p", 0.0)
    
    swing_high = pre.last_sh[iL] if pre else 0.0
    swing_low = pre.last_sl[iL] if pre else 0.0
    
    htf_sh = adv.get('htf_swing_high', [0.0])[iL] if 'htf_swing_high' in adv else 0.0
    htf_sl = adv.get('htf_swing_low', [0.0])[iL] if 'htf_swing_low' in adv else 0.0
    mtf_sh = adv.get('mtf_swing_high', [0.0])[iL] if 'mtf_swing_high' in adv else 0.0
    mtf_sl = adv.get('mtf_swing_low', [0.0])[iL] if 'mtf_swing_low' in adv else 0.0

    reversion_penalty = 1.0
    if adx_val > 30: reversion_penalty = 0.5 
    
    candidates = []

    # ==========================================================
    # THE ULTIMATE CRYPTO PIVOT PLAN (3-MODEL MATRIX)
    # ==========================================================

    htf_ema50 = base.get("htf_ema50", px)
    velocity_atr_3 = base.get("velocity_atr_3", 0.0)

    # ----------------------------------------------------------
    # MODEL A: The Volatility Squeeze Breakout (The 1:3 Momentum Engine)
    # ----------------------------------------------------------
    is_squeezed = base.get("is_squeezed", 0)

    if is_squeezed > 0:
        dc_high = base.get("dc_high_20", px + current_atr)
        dc_low = base.get("dc_low_20", px - current_atr)

        if can_long and px > htf_ema50:
            candidates.append({
                "strat": "SQUEEZE_BREAKOUT_LONG",
                "priority": 3.0,
                "side": "long",
                "entry": dc_high,
                "sl_override": dc_low,
                "risk_mult": 1.0,
                "type": "breakout"
            })

        if can_short and px < htf_ema50:
            candidates.append({
                "strat": "SQUEEZE_BREAKOUT_SHORT",
                "priority": 3.0,
                "side": "short",
                "entry": dc_low,
                "sl_override": dc_high,
                "risk_mult": 1.0,
                "type": "breakout"
            })

    # ----------------------------------------------------------
    # MODEL B: The V-Shape Mean Reversion (The Panic Buy/Sell)
    # ----------------------------------------------------------
    bb_lower_25 = base.get("bb_lower_25", px - current_atr * 2.5)
    bb_upper_25 = base.get("bb_upper_25", px + current_atr * 2.5)

    # We only fade extremes if velocity is insanely high (algorithmic cascade)
    if velocity_atr_3 > 1.5:
        if can_long and rsi < 30 and px < bb_lower_25:
            candidates.append({
                "strat": "PANIC_REVERSION_LONG",
                "priority": 2.5,
                "side": "long",
                "entry": bb_lower_25,
                "sl_dist_atr": 1.0,
                "risk_mult": 1.5,
                "type": "limit"
            })

        if can_short and rsi > 70 and px > bb_upper_25:
            candidates.append({
                "strat": "PANIC_REVERSION_SHORT",
                "priority": 2.5,
                "side": "short",
                "entry": bb_upper_25,
                "sl_dist_atr": 1.0,
                "risk_mult": 1.5,
                "type": "limit"
            })

    # ----------------------------------------------------------
    # MODEL C: The EMA20 Trend Continuation
    # ----------------------------------------------------------
    ema20 = adv.get('ema20_L', [px])[iL] if 'ema20_L' in adv else px

    if adx_val > 35:
        if can_long and px > htf_ema50 and px > ema20:
            candidates.append({
                "strat": "EMA_TREND_RIDE_LONG",
                "priority": 2.0,
                "side": "long",
                "entry": ema20,
                "sl_dist_atr": 1.5,
                "risk_mult": 1.0,
                "type": "limit"
            })

        if can_short and px < htf_ema50 and px < ema20:
            candidates.append({
                "strat": "EMA_TREND_RIDE_SHORT",
                "priority": 2.0,
                "side": "short",
                "entry": ema20,
                "sl_dist_atr": 1.5,
                "risk_mult": 1.0,
                "type": "limit"
            })

    # EVALUATE ALL CANDIDATES
    best_plan = None
    best_score = -999.0
    
    for cand in candidates:
        side = cand['side']
        order_type = cand.get('type', cfg.execution_style)
        
        entry = cand['entry']


        # DEFAULT BASELINE VARIABLES (If no brain is present)
        prob = 1.0
        risk_factor = cand.get('risk_mult', 1.0)
        score = cand['priority']

        if brain:
            prob = brain.predict_prob(side, base, adv, iH, iM, iL, px, pre)
            min_prob = cfg.min_prob_long if side == 'long' else cfg.min_prob_short
            
            if prob < min_prob:
                continue 
            
            edge = prob - min_prob
            base_risk = 1.0 + (edge * 15.0) 
            risk_factor = min(base_risk * cand.get('risk_mult', 1.0), getattr(cfg, 'max_risk_factor', 2.0))
            score = prob * cand['priority']
            
        # CVD DIVERGENCE BOOST
        if side == 'long' and base.get("cvd_div_bull", 0) > 0 and "GAMMA" in cand['strat']:
            score += 0.2
            risk_factor *= 1.2
        elif side == 'short' and base.get("cvd_div_bear", 0) > 0 and "GAMMA" in cand['strat']:
            score += 0.2
            risk_factor *= 1.2

        # LIQUIDATION SQUEEZE CASCADE
        if delta_oi > 0.05 and funding < -0.01: # 5% jump + high negative funding
            if side == 'long' and "GAMMA" in cand['strat']:
                risk_factor *= 1.5
                
        if getattr(cfg, 'filter_htf_trend', False):
            is_counter = (side == 'short' and btc_bullish > 0.5) or (side == 'long' and btc_bullish < 0.5)
            if is_counter:
                score *= 0.5
                risk_factor *= 0.5

        # STAGE 1: Convert Hard Gates to "Soft" Features
        # Derivatives, Volume Profile Confluence are now handled entirely by XGBoost features
        # (funding_rate, time_at_mode, dist_poc_pct) instead of hard rejections to ensure trade quantity.

        # Relaxed UPGRADE 3: MTF Premium/Discount Filtering (Value Area Logic)
        # Only reject absolute extremes (e.g., > 0.8 for longs, < 0.2 for shorts) instead of the 0.5 midpoint.
        mtf_range = mtf_sh - mtf_sl
        if mtf_range > 0:
            mtf_premium_discount = (px - mtf_sl) / mtf_range
            if side == 'long' and mtf_premium_discount > 0.80:
                continue
            if side == 'short' and mtf_premium_discount < 0.20:
                continue

        # UPGRADE 1: Liquidation Cascade Velocity Filter (Only for Limits, Breakouts ride the cascade)
        # Do not catch a falling knife if it's crashing > 1.5 ATR per bar into our limit
        velocity_atr_3 = base.get("velocity_atr_3", 0.0)
        if order_type == 'limit' and velocity_atr_3 > 1.5:
            if cand['strat'] != 'PANIC_REVERSION_LONG' and cand['strat'] != 'PANIC_REVERSION_SHORT':
                continue # We WANT high velocity for panic reversion, but reject for trend rides

        # ==========================================================
        # MULTI-MODEL RISK & REWARD GEOMETRY
        # ==========================================================

        # 1. Stop Loss Assignment
        if 'sl_override' in cand:
            sl = cand['sl_override']
            # Sanity clamp to prevent mathematically impossible physics
            if side == 'long' and sl >= entry: sl = entry - (current_atr * 1.5)
            if side == 'short' and sl <= entry: sl = entry + (current_atr * 1.5)
        else:
            # Fallback to dynamic ATR distance provided by the model
            sl_dist_atr = cand.get('sl_dist_atr', 1.5)
            if side == 'long':
                sl = entry - (current_atr * sl_dist_atr)
            else:
                sl = entry + (current_atr * sl_dist_atr)

        dynamic_risk = abs(entry - sl)
        if dynamic_risk < 1e-9: continue

        # 2. Strict 1:3 Take Profit Lock
        # Since these models are purely momentum/mean-reversion and not structural pullbacks,
        # we lock the Take Profit to EXACTLY 3.0x the calculated risk distance.
        implied_rr = 3.0

        if side == 'long':
            tp = entry + (3.0 * dynamic_risk)
        else:
            tp = entry - (3.0 * dynamic_risk)

        if score > best_score:
            best_score = score
            
            if implied_rr >= cfg.min_rr:
                best_plan = {
                    "side": side,
                    "entry": entry,
                    "sl": sl,
                    "tp": tp,
                    "rr": implied_rr,
                    "prob": prob,
                    "key": f"{cand['strat']}_{side}",
                    "features": base,
                    "risk_factor": risk_factor,
                    "strategy": cand['strat'],
                    "type": order_type
                }

    return best_plan