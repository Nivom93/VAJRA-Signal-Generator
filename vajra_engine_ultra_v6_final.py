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

import os, time, math, threading, sqlite3, random, csv, traceback
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger("vajra.engine")
if not log.handlers:
    log.setLevel(logging.INFO)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
    log.addHandler(h)

# --- BULLETPROOF IMPORTS ---
try:
    import ccxt; import joblib
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
    macro_tf: str = "1d"; swing_tf: str = "4h"; htf: str = "1h"; exec_tf: str = "15m"
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
    maker_fee_bps: float = 0.0; taker_fee_bps: float = 0.0; slippage_bps: float = 0.0
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
        if ccxt is None:
            raise RuntimeError("ccxt missing")

        self.client = getattr(ccxt, cfg.exchange_id)({
            "enableRateLimit": True,
            "options": {"defaultType": cfg.market_type},
            "timeout": 30000
        })

        if markets_data:
            self.client.markets = markets_data
        else:
            for attempt in range(3):
                try:
                    self.client.load_markets()
                    break
                except Exception as e:
                    if attempt == 2:
                        raise e
                    print(f"Connection warning: {e}. Retrying...")
                    time.sleep(2)

    def fetch_ohlcv_df(self, sym, tf, limit=1000, since=None):
        r = self.client.fetch_ohlcv(sym, timeframe=tf, limit=min(1000, limit), since=since)
        return pd.DataFrame(r, columns=["timestamp", "open", "high", "low", "close", "volume"])

    def fetch_funding_rate(self, symbol):
        try:
            funding = self.client.fetch_funding_rate(symbol)
            return float(funding.get('fundingRate', 0.0))
        except:
            return 0.0

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
    n = len(c)
    ob_bull_top = np.zeros(n)
    ob_bull_bot = np.zeros(n)
    ob_bear_top = np.zeros(n)
    ob_bear_bot = np.zeros(n)
    bars_since_ob_bull = np.zeros(n)
    bars_since_ob_bear = np.zeros(n)

    active_bull_top = 0.0
    active_bull_bot = 0.0
    active_bear_top = 0.0
    active_bear_bot = 0.0

    active_bull_idx = -1
    active_bear_idx = -1

    for i in range(2, n):
        # Carry forward unmitigated zones
        ob_bull_top[i] = active_bull_top
        ob_bull_bot[i] = active_bull_bot
        ob_bear_top[i] = active_bear_top
        ob_bear_bot[i] = active_bear_bot

        if active_bull_idx != -1:
            bars_since_ob_bull[i] = float(i - active_bull_idx)
        if active_bear_idx != -1:
            bars_since_ob_bear[i] = float(i - active_bear_idx)

        # Advanced Mitigation: >50% penetration or close beyond
        if active_bull_top > 0:
            midpoint = active_bull_top - ((active_bull_top - active_bull_bot) * 0.5)
            if l[i] < midpoint or c[i] < active_bull_bot:
                active_bull_top = 0.0
                active_bull_bot = 0.0
                active_bull_idx = -1
                ob_bull_top[i] = 0.0
                ob_bull_bot[i] = 0.0
                bars_since_ob_bull[i] = 0.0

        if active_bear_bot > 0:
            midpoint = active_bear_bot + ((active_bear_top - active_bear_bot) * 0.5)
            if h[i] > midpoint or c[i] > active_bear_top:
                active_bear_bot = 0.0
                active_bear_top = 0.0
                active_bear_idx = -1
                ob_bear_bot[i] = 0.0
                ob_bear_top[i] = 0.0
                bars_since_ob_bear[i] = 0.0

        # Scan for new accumulation/distribution zones
        if bos_up[i] == 1 and bos_up[i-1] == 0:
            for j in range(i-1, max(0, i-50), -1):
                if c[j] < o[j]: 
                    if j+1 < n:
                        body_ob = o[j] - c[j]; next_body = c[j+1] - o[j+1]
                        if c[j+1] > o[j+1] and next_body > 1.5 * body_ob:
                            active_bull_top = float(h[j])
                            active_bull_bot = float(l[j])
                            active_bull_idx = j
                            ob_bull_top[i] = active_bull_top
                            ob_bull_bot[i] = active_bull_bot
                            bars_since_ob_bull[i] = float(i - j)
                    break
        if bos_dn[i] == 1 and bos_dn[i-1] == 0:
            for j in range(i-1, max(0, i-50), -1):
                if c[j] > o[j]: 
                    if j+1 < n:
                        body_ob = c[j] - o[j]; next_body = o[j+1] - c[j+1]
                        if c[j+1] < o[j+1] and next_body > 1.5 * body_ob:
                            active_bear_bot = float(l[j])
                            active_bear_top = float(h[j])
                            active_bear_idx = j
                            ob_bear_bot[i] = active_bear_bot
                            ob_bear_top[i] = active_bear_top
                            bars_since_ob_bear[i] = float(i - j)
                    break
    return ob_bull_top, ob_bull_bot, ob_bear_bot, ob_bear_top, bars_since_ob_bull, bars_since_ob_bear

@njit(cache=True)
def _detect_qm(h, l, c, swing_hi, swing_lo):
    n = len(c)
    qm_bull = np.zeros(n)
    qm_bear = np.zeros(n)
    # A Bullish QM: Low, High, Lower Low (sweeps Low), Higher High (breaks High).
    last_l = 0.0; last_h = 0.0; prev_l = 0.0; prev_h = 0.0
    for i in range(10, n):
        if swing_lo[i-1]:
            prev_l = last_l
            last_l = l[i-1]
        if swing_hi[i-1]:
            prev_h = last_h
            last_h = h[i-1]

        # Bullish QM active if last_l < prev_l (Lower Low) and c[i] > prev_h (Higher High)
        if prev_l > 0 and last_l < prev_l and c[i] > prev_h and prev_h > 0:
            qm_bull[i] = prev_l # The left shoulder to bid

        # Bearish QM active if last_h > prev_h (Higher High) and c[i] < prev_l (Lower Low)
        if prev_h > 0 and last_h > prev_h and c[i] < prev_l and prev_l > 0:
            qm_bear[i] = prev_h # The left shoulder to offer
    return qm_bull, qm_bear

@njit(cache=True)
def _detect_wyckoff(h, l, c, vol_spike, cvd_div_bull, cvd_div_bear, last_sl, last_sh):
    n = len(c)
    spring = np.zeros(n)
    upthrust = np.zeros(n)
    for i in range(5, n):
        # Spring: Sweeps last_sl, but closes back above it, with vol_spike and cvd_div
        curr_sl = last_sl[i-1]
        if curr_sl > 0 and l[i] < curr_sl and c[i] > curr_sl:
            if vol_spike[i] > 0 and cvd_div_bull[i] > 0:
                spring[i] = 1.0

        # Upthrust: Sweeps last_sh, but closes back below it, with vol_spike and cvd_div
        curr_sh = last_sh[i-1]
        if curr_sh > 0 and h[i] > curr_sh and c[i] < curr_sh:
            if vol_spike[i] > 0 and cvd_div_bear[i] > 0:
                upthrust[i] = 1.0
    return spring, upthrust

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
        self.ob_bull_top, self.ob_bull_bot, self.ob_bear_bot, self.ob_bear_top, self.bars_since_ob_bull, self.bars_since_ob_bear = _find_ob_zones_strict(self.o, self.c, self.h, self.l, self.bos_up, self.bos_down)
        
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

        # ADVANCED STRUCTURE PATTERNS
        self.qm_bull, self.qm_bear = _detect_qm(self.h, self.l, self.c, self.swing_hi, self.swing_lo)
        self.spring, self.upthrust = _detect_wyckoff(self.h, self.l, self.c, self.vol_spike, self.cvd_div_bull, self.cvd_div_bear, self.last_sl, self.last_sh)

        hours = pd.to_datetime(df.timestamp, unit='ms', utc=True).dt.hour.values
        days = pd.to_datetime(df.timestamp, unit='ms', utc=True).dt.dayofyear.values
        self.asian_high, self.asian_low = _asian_range_np(self.h, self.l, hours, days)

        self.atr_percentile_100 = _rolling_percentile_np(self.atr14, 100)
        self.rolling_vwap_20 = _rolling_vwap_np((self.h + self.l + self.c) / 3.0, self.v, 20)
        self.dc_high_20, self.dc_low_20 = _donchian_channels_np(self.h, self.l, 20)

def _trend_flags(e50, e200): return (1.0, 0.0) if e50>e200 else ((0.0, 1.0) if e50<e200 else (0.0, 0.0))
def _ensure_precomp(df, pre): return pre if isinstance(pre, Precomp) else Precomp(df)

def confluence_features(cfg, macro_tf, swing_tf, htf, exec_tf, iMacro, iSwing, iHtf, iExec, precomp=None, extras=None):
    if iMacro is None: iMacro=len(macro_tf)-1; iSwing=len(swing_tf)-1; iHtf=len(htf)-1; iExec=len(exec_tf)-1
    
    # Strictly aligned pointers for fully closed candles to prevent lookahead
    idx_Macro = max(0, iMacro)
    idx_Swing = max(0, iSwing)
    idx_Htf = max(0, iHtf)
    idx_Exec = iExec
    
    pMacro=_ensure_precomp(macro_tf, precomp.get("macro_tf") if precomp else None)
    pSwing=_ensure_precomp(swing_tf, precomp.get("swing_tf") if precomp else None)
    pHtf=_ensure_precomp(htf, precomp.get("htf") if precomp else None)
    pExec=_ensure_precomp(exec_tf, precomp.get("exec_tf") if precomp else None)
    
    f={}
    f["macro_up"], f["macro_down"] = _trend_flags(pMacro.ema50[idx_Macro], pMacro.ema200[idx_Macro])
    f["swing_up"], f["swing_down"] = _trend_flags(pSwing.ema50[idx_Swing], pSwing.ema200[idx_Swing])
    f["htf_up"], f["htf_down"] = _trend_flags(pHtf.ema50[idx_Htf], pHtf.ema200[idx_Htf])
    f["macro_ema50"] = float(pMacro.ema50[idx_Macro])
    f["swing_ema50"] = float(pSwing.ema50[idx_Swing])
    f["htf_ema50"] = float(pHtf.ema50[idx_Htf])
    
    px = pExec.c[idx_Exec]; f["price"]=float(px);
    
    f["atr_macro_pct"]=float(pMacro.atr14[idx_Macro]/max(px,1e-9)*100.0)
    f["atr_swing_pct"]=float(pSwing.atr14[idx_Swing]/max(px,1e-9)*100.0)
    f["atr_htf_pct"]=float(pHtf.atr14[idx_Htf]/max(px,1e-9)*100.0)
    f["atr_exec_pct"]=float(pExec.atr14[idx_Exec]/max(px,1e-9)*100.0)
    f["atr_ltf_pct"]=f["atr_exec_pct"] # Backwards compatibility alias
    
    f["last_swing_high"] = float(pExec.last_sh[idx_Exec])
    f["last_swing_low"] = float(pExec.last_sl[idx_Exec])
    f["ob_bull_top"] = float(pExec.ob_bull_top[idx_Exec])
    f["ob_bull_bot"] = float(pExec.ob_bull_bot[idx_Exec])
    f["ob_bear_bot"] = float(pExec.ob_bear_bot[idx_Exec])
    f["ob_bear_top"] = float(pExec.ob_bear_top[idx_Exec])
    
    px_safe = px if px > 1e-12 else 1.0
    f["dist_last_sh_pct"] = (px - f["last_swing_high"]) / px_safe * 100
    f["dist_last_sl_pct"] = (px - f["last_swing_low"]) / px_safe * 100
    f["dist_ob_bull_pct"] = (px - f["ob_bull_top"]) / px_safe * 100
    f["dist_ob_bear_pct"] = (f["ob_bear_bot"] - px) / px_safe * 100

    atr_abs = pHtf.atr14[idx_Htf]
    if atr_abs < 1e-9: atr_abs = px * 0.01
    current_range = pExec.h[idx_Exec] - pExec.l[idx_Exec]
    f["candle_range_atr"] = float(current_range / atr_abs)
    
    for k in ["bos_up","bos_down","fvg_up","fvg_dn","engulf_bull","engulf_bear","pin_bull","pin_bear","inside_bar"]:
        f[k]=float(getattr(pExec,k)[idx_Exec])
    f["sweep_high"] = float(pExec.sweep_up[idx_Exec])
    f["sweep_low"] = float(pExec.sweep_dn[idx_Exec])
    
    f["choch_up"]=float(1.0 if pExec.bos_up[idx_Exec] and idx_Exec>=5 and pExec.bos_down[idx_Exec-5] else 0.0)
    f["choch_down"]=float(1.0 if pExec.bos_down[idx_Exec] and idx_Exec>=5 and pExec.bos_up[idx_Exec-5] else 0.0)
    f["bars_since_ob_bull"] = float(pExec.bars_since_ob_bull[idx_Exec])
    f["bars_since_ob_bear"] = float(pExec.bars_since_ob_bear[idx_Exec])

    # Safely pull velocity_atr_3 if precomputed, else 0
    if hasattr(pExec, 'velocity_atr_3'):
        f["velocity_atr_3"] = float(pExec.velocity_atr_3[idx_Exec])

    f["vol_spike"] = float(pExec.vol_spike[idx_Exec])
    f["kalman_vel"] = float(pExec.kalman_vel[idx_Exec]); f["hurst"] = float(pExec.hurst[idx_Exec])
    
    f["cvd"] = float(pExec.cvd[idx_Exec])
    f["adx"] = float(pExec.adx[idx_Exec])
    f["bb_width"] = float(pExec.bb_width[idx_Exec])
    f["rvol"] = float(pExec.rvol[idx_Exec])
    f["bb_upper"] = float(pExec.bb_upper[idx_Exec])
    f["bb_lower"] = float(pExec.bb_lower[idx_Exec])
    f["kc_upper"] = float(pExec.kc_upper[idx_Exec])
    f["kc_lower"] = float(pExec.kc_lower[idx_Exec])

    f["dist_bb_upper_pct"] = (px - f["bb_upper"]) / px_safe * 100
    f["dist_bb_lower_pct"] = (px - f["bb_lower"]) / px_safe * 100
    f["rsi_14"] = float(pExec.rsi14[idx_Exec])
    
    f["w_pattern"] = float(pExec.w_pattern[idx_Exec])
    f["m_pattern"] = float(pExec.m_pattern[idx_Exec])
    f["fvg_bull"] = float(pExec.fvg_bull_p[idx_Exec])
    f["fvg_bear"] = float(pExec.fvg_bear_p[idx_Exec])
    f["sweep_bull_p"] = float(pExec.sweep_bull_p[idx_Exec])
    f["sweep_bear_p"] = float(pExec.sweep_bear_p[idx_Exec])

    # MICRO-STRUCTURE EXTRACTION
    f["avwap_bull"] = float(pExec.avwap_bull[idx_Exec])
    f["avwap_bear"] = float(pExec.avwap_bear[idx_Exec])
    f["poc"] = float(pExec.poc[idx_Exec])
    f["vah"] = float(pExec.vah[idx_Exec])
    f["val"] = float(pExec.val[idx_Exec])

    f["dist_avwap_bull_pct"] = (px - f["avwap_bull"]) / px_safe * 100
    f["dist_avwap_bear_pct"] = (px - f["avwap_bear"]) / px_safe * 100
    f["dist_poc_pct"] = (px - f["poc"]) / px_safe * 100
    f["dist_vah_pct"] = (px - f["vah"]) / px_safe * 100
    f["dist_val_pct"] = (px - f["val"]) / px_safe * 100

    f["is_squeezed"] = float(pExec.is_squeezed[idx_Exec])
    f["squeeze_fired"] = float(pExec.squeeze_fired[idx_Exec])
    f["swing_is_squeezed"] = float(pSwing.is_squeezed[idx_Swing])
    f["swing_squeeze_fired"] = float(pSwing.squeeze_fired[idx_Swing])
    f["cvd_div_bull"] = float(pExec.cvd_div_bull[idx_Exec])
    f["cvd_div_bear"] = float(pExec.cvd_div_bear[idx_Exec])
    f["cvd_roc"] = float(pExec.cvd_roc[idx_Exec])
    f["cvd_acceleration"] = float(pExec.cvd_acceleration[idx_Exec])

    f["qm_bull"] = float(pExec.qm_bull[idx_Exec])
    f["qm_bear"] = float(pExec.qm_bear[idx_Exec])
    f["wyckoff_spring"] = float(pExec.spring[idx_Exec])
    f["wyckoff_upthrust"] = float(pExec.upthrust[idx_Exec])

    f["asian_high"] = float(pExec.asian_high[idx_Exec])
    f["asian_low"] = float(pExec.asian_low[idx_Exec])
    f["dc_high_20"] = float(pExec.dc_high_20[idx_Exec])
    f["dc_low_20"] = float(pExec.dc_low_20[idx_Exec])
    
    f["dist_asian_high_pct"] = (px - f["asian_high"]) / px_safe * 100
    f["dist_asian_low_pct"] = (px - f["asian_low"]) / px_safe * 100

    # SMC & Fibonacci Deep Wick Geometry
    macro_high = max(f["last_swing_high"], f["last_swing_low"])
    macro_low = min(f["last_swing_high"], f["last_swing_low"])
    fractal_range = macro_high - macro_low

    if fractal_range > 0:
        f["fib_786_long"] = macro_low + (fractal_range * 0.214)
        f["fib_886_long"] = macro_low + (fractal_range * 0.114)
        f["fib_786_short"] = macro_high - (fractal_range * 0.214)
        f["fib_886_short"] = macro_high - (fractal_range * 0.114)
    else:
        f["fib_786_long"] = f["fib_886_long"] = f["fib_786_short"] = f["fib_886_short"] = px

    f["dist_fib_786_long_pct"] = (px - f["fib_786_long"]) / px_safe * 100
    f["dist_fib_886_long_pct"] = (px - f["fib_886_long"]) / px_safe * 100
    f["dist_fib_786_short_pct"] = (px - f["fib_786_short"]) / px_safe * 100
    f["dist_fib_886_short_pct"] = (px - f["fib_886_short"]) / px_safe * 100

    # Check if Asian Range was just swept
    f["asian_range_swept_up"] = 1.0 if pExec.h[idx_Exec] > pExec.asian_high[idx_Exec] and pExec.c[idx_Exec] < pExec.asian_high[idx_Exec] else 0.0
    f["asian_range_swept_dn"] = 1.0 if pExec.l[idx_Exec] < pExec.asian_low[idx_Exec] and pExec.c[idx_Exec] > pExec.asian_low[idx_Exec] else 0.0
    
    ts = exec_tf.timestamp.iloc[idx_Exec]
    f["hour_of_day"] = float((int(ts) // 3600000) % 24)
    
    dt = pd.to_datetime(ts, unit='ms', utc=True)
    f["day_of_week"] = float(dt.dayofweek)

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
    f["inter_swing_down_sweep_high"] = f.get("swing_down",0) * f.get("sweep_high",0)
    ema20_gt = 1.0 if (pExec.ema20[idx_Exec] > _ema_np(pExec.c, 100)[idx_Exec] if hasattr(pExec, 'ema20') else 0) else 0.0
    f["trend_align_up_3tf"] = f.get("macro_up",0) + f.get("swing_up",0) + f.get("htf_up",0) + ema20_gt
    f["trend_align_down_3tf"] = f.get("macro_down",0) + f.get("swing_down",0) + f.get("htf_down",0) + (1.0 - ema20_gt)

    # ---------------------------------------------------------
    # DIRECTIVE 1: THE REGIME DETECTION MATRIX
    # ---------------------------------------------------------
    adx_val = f["adx"]
    atr_p = pExec.atr_percentile_100[idx_Exec] * 100.0
    htf_trend = "UP" if f["htf_up"] > 0 else ("DOWN" if f["htf_down"] > 0 else "NEUTRAL")

    is_bb_outside = (px > f["bb_upper"]) or (px < f["bb_lower"])
    has_cvd_div = (f["cvd_div_bull"] > 0) or (f["cvd_div_bear"] > 0)

    market_regime = "CONSOLIDATION"

    if is_bb_outside and has_cvd_div:
        market_regime = "REVERSAL_WARNING"
    elif htf_trend == "UP" and f["price"] < pExec.rolling_vwap_20[idx_Exec]:
        market_regime = "RETRACEMENT"
    elif htf_trend == "DOWN" and f["price"] > pExec.rolling_vwap_20[idx_Exec]:
        market_regime = "RETRACEMENT"
    elif adx_val >= 25 and atr_p >= 50:
        market_regime = "EXPANSION"
    elif adx_val < 20 and atr_p < 40:
        market_regime = "CONSOLIDATION"
    else:
        market_regime = "CONSOLIDATION" # Default

    f["market_regime"] = market_regime
    # ==========================================================
    # TREND TENSOR: MULTIDIMENSIONAL MARKET PERCEPTION
    # ==========================================================
    regime_raw = (
        0.4 * f.get("trend_align_up_3tf", 0.0) +
        0.3 * f.get("cvd_roc", 0.0) +
        0.2 * f.get("market_efficiency_ratio", 0.0) +
        0.1 * f.get("macro_sentiment", 0.0)
    )
    f["sentient_regime_score"] = float(np.tanh(regime_raw))

    return f

def precompute_v6_features(pMacro, pSwing, pExec, macro_tf, swing_tf, exec_tf, btc_close_arr=None):
    cl=pExec.c; f={}
    if cl.size==0: return {}
    eps=1e-12
    f['ema20_L']=_ema_np(cl,20)
    f['ema50_L']=pExec.ema50
    f['ema100_L']=_ema_np(cl,100)

    f['dist_ema20_pct']=(cl - f['ema20_L']) / (np.abs(cl)+eps) * 100
    f['dist_ema50_pct']=(cl - f['ema50_L']) / (np.abs(cl)+eps) * 100
    f['dist_ema100_pct']=(cl - f['ema100_L']) / (np.abs(cl)+eps) * 100

    f['ts_50_200']=(pExec.ema50-pExec.ema200)/(np.abs(cl)+eps)
    f['ts_20_100']=(_ema_np(cl,20)-_ema_np(cl,100))/(np.abs(cl)+eps)
    f['ts_roll_mean_10']=_rolling_mean(f['ts_50_200'],10)
    f['ts_roll_std_10']=_rolling_std(f['ts_50_200'],10)
    f['ts_roc_5']=_roc(f['ts_50_200'],5)
    f['ema20_gt_ema100']=(_ema_np(cl,20)>_ema_np(cl,100)).astype(float)
    f['bb_width_pct_20']=pExec.bb_width
    f['atr14_L']=pExec.atr14
    f['atr_pct_ltf']=pExec.atr14/(np.abs(cl)+eps)*100
    f['rsi14_ltf']=pExec.rsi14
    f['vol_ratio_200']=f['atr_pct_ltf']/_rolling_mean(f['atr_pct_ltf'],200,20)
    f['vol_ratio_200']=np.nan_to_num(f['vol_ratio_200'],nan=1.0)
    mtf_ema200_s = pd.Series(pSwing.ema200, index=swing_tf.timestamp).shift(1).reindex(exec_tf.timestamp, method='ffill').fillna(0).values
    f['mtf_ema200_arr']=mtf_ema200_s
    f['dist_mtf_ema200_pct']=np.where(cl>0, (cl - mtf_ema200_s) / cl * 100, 0.0)
    cm=pSwing.c
    f['trend_strength_mtf_arr']=(pSwing.ema50-pSwing.ema200)/(np.abs(cm)+eps)
    f['atr_pct_ltf_roc_5_arr']=_roc(f['atr_pct_ltf'],5)
    f['vol_ltf_accel_5_arr']=_roc(f['atr_pct_ltf_roc_5_arr'],5)
    f['trend_strength_ltf_ema20_50_arr']=(_ema_np(cl,20)-pExec.ema50)/(np.abs(cl)+eps)
    
    htf_s = pd.Series(pMacro.atr14/pMacro.c*100, index=macro_tf.timestamp).shift(1).reindex(exec_tf.timestamp, method='ffill').fillna(0).values
    f['atr_pct_htf_aligned']=htf_s
    mtf_s = pd.Series(f['trend_strength_mtf_arr'], index=swing_tf.timestamp).shift(1).reindex(exec_tf.timestamp, method='ffill').fillna(0).values
    f['trend_strength_mtf_aligned']=mtf_s
    
    # INJECT ALIGNED HTF/MTF FRACTAL SWINGS
    f['htf_swing_high'] = pd.Series(pMacro.last_sh, index=macro_tf.timestamp).shift(1).reindex(exec_tf.timestamp, method='ffill').fillna(0.0).values
    f['htf_swing_low'] = pd.Series(pMacro.last_sl, index=macro_tf.timestamp).shift(1).reindex(exec_tf.timestamp, method='ffill').fillna(0.0).values
    f['mtf_swing_high'] = pd.Series(pSwing.last_sh, index=swing_tf.timestamp).shift(1).reindex(exec_tf.timestamp, method='ffill').fillna(0.0).values
    f['mtf_swing_low'] = pd.Series(pSwing.last_sl, index=swing_tf.timestamp).shift(1).reindex(exec_tf.timestamp, method='ffill').fillna(0.0).values

    htf_tr = np.where(pMacro.ema50>pMacro.ema200,1,np.where(pMacro.ema50<pMacro.ema200,-1,0))
    htf_tr_s = pd.Series(htf_tr, index=macro_tf.timestamp).shift(1).reindex(exec_tf.timestamp, method='ffill').fillna(0).values.astype(int)
    f['bars_since_htf_trend_flip_arr']=_bars_since_change(htf_tr_s)
    f['atr7_L']=_atr_np(pExec.h,pExec.l,pExec.c,7)
    f['atr_pct_ltf_7_arr']=f['atr7_L']/(np.abs(cl)+eps)*100
    
    f['dist_to_bull_ob_arr'] = np.where(pExec.ob_bull_top>0, (cl-pExec.ob_bull_top)/cl*100, 0.0)
    f['dist_to_bear_ob_arr'] = np.where(pExec.ob_bear_bot>0, (pExec.ob_bear_bot-cl)/cl*100, 0.0)
    f['is_sweep_high_arr'] = pExec.sweep_up.astype(float)
    f['is_sweep_low_arr'] = pExec.sweep_dn.astype(float)
    
    l_mom = np.where(pExec.rsi14>50,0.5,-0.5)+np.where(pExec.macd>pExec.macd_sig,0.5,-0.5)
    m_mom_s = pd.Series(np.where(pSwing.rsi14>50,0.5,-0.5)+np.where(pSwing.macd>pSwing.macd_sig,0.5,-0.5), index=swing_tf.timestamp).shift(1).reindex(exec_tf.timestamp, method='ffill').fillna(0).values
    h_mom_s = pd.Series(np.where(pMacro.rsi14>50,0.5,-0.5)+np.where(pMacro.macd>pMacro.macd_sig,0.5,-0.5), index=macro_tf.timestamp).shift(1).reindex(exec_tf.timestamp, method='ffill').fillna(0).values
    f['momentum_alignment_score_arr'] = l_mom + m_mom_s + h_mom_s
    
    f['market_efficiency_ratio'] = pExec.eff_ratio
    rng = pExec.last_sh - pExec.last_sl
    safe_rng = np.where(rng == 0, 1.0, rng)
    f['premium_discount_index'] = np.where(rng > 0, (cl - pExec.last_sl) / safe_rng, 0.5)

    f['div_bull_arr'] = pExec.div_bull
    f['div_bear_arr'] = pExec.div_bear
    
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
    bbw_arr = pExec.bb_width
    kc_upper_arr, kc_lower_arr = _keltner_channels_np(cl, pExec.h, pExec.l, 20, 1.5)
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
    f['atr_percentile_100'] = pExec.atr_percentile_100 * 100.0

    # ---------------------------------------------------------
    # DIRECTIVE 2: ICT & MICROSTRUCTURE PROXIES
    # ---------------------------------------------------------
    # rolling 20-bar Swing High/Low
    rolling_sh = pd.Series(pExec.h).rolling(20, min_periods=1).max().to_numpy()
    rolling_sl = pd.Series(pExec.l).rolling(20, min_periods=1).min().to_numpy()
    atr_safe = np.where(pExec.atr14 > 1e-9, pExec.atr14, cl * 0.01)

    f['dist_to_rolling_sh_atr'] = (rolling_sh - cl) / atr_safe
    f['dist_to_rolling_sl_atr'] = (cl - rolling_sl) / atr_safe

    # Wick Rejection Ratio
    total_range = pExec.h - pExec.l
    total_range_safe = np.where(total_range > 1e-9, total_range, 1e-9)
    body_top = np.maximum(pExec.o, cl)
    body_bottom = np.minimum(pExec.o, cl)
    f['upper_wick_pct'] = (pExec.h - body_top) / total_range_safe
    f['lower_wick_pct'] = (body_bottom - pExec.l) / total_range_safe

    # Killzone Session Flags
    hours = pd.to_datetime(exec_tf.timestamp, unit='ms', utc=True).dt.hour.values
    f['is_london_killzone'] = np.where((hours >= 7) & (hours <= 10), 1.0, 0.0)
    f['is_ny_killzone'] = np.where((hours >= 13) & (hours <= 16), 1.0, 0.0)
    f['is_asian_range'] = np.where((hours >= 0) & (hours <= 6), 1.0, 0.0)

    # Wyckoff Volume Absorption (Directive 3)
    candle_spread = pExec.h - pExec.l
    body_size = np.abs(cl - pExec.o)
    f['vol_absorption'] = np.where((pExec.rvol > 1.5) & ((body_size / (candle_spread + 1e-9)) < 0.3), 1.0, 0.0)

    # Time at Mode (Liquidity Consumption proxy)
    # Measures how many bars in the last 20 were within 0.5 ATR of the VWAP
    diff_from_vwap = np.abs(cl - pExec.rolling_vwap_20)
    is_near_vwap = np.where(diff_from_vwap <= atr_safe * 0.5, 1.0, 0.0)
    f['time_at_mode'] = pd.Series(is_near_vwap).rolling(20, min_periods=1).sum().to_numpy()

    # ---------------------------------------------------------
    # DIRECTIVE 3: VWAP & MEAN REVERSION EXTREMES
    # ---------------------------------------------------------
    f['vwap_z_score'] = (cl - pExec.rolling_vwap_20) / atr_safe

    # Liquidation Cascade Velocity Filter (Anti-Falling Knife)
    # Measures the speed of the price drop/rally over the last 3 bars
    cl_shifted_3 = pd.Series(cl).shift(3).fillna(cl[0]).to_numpy()
    velocity_atr_3_arr = np.abs(cl - cl_shifted_3) / atr_safe
    f['velocity_atr_3'] = velocity_atr_3_arr
    pExec.velocity_atr_3 = velocity_atr_3_arr

    # ---------------------------------------------------------
    # DIRECTIVE 4: STRIP COLLINEAR NOISE
    # ---------------------------------------------------------
    keys_to_drop = [
        'ema20_L', 'ema50_L', 'ema100_L',
        'mtf_ema200_arr', 'atr14_L', 'atr7_L',
        'trend_strength_mtf_arr'
    ]
    for k in keys_to_drop:
        if k in f:
            del f[k]

    return f

class BrainLearningManager:
    def __init__(self, cfg, brains_dir=None):
        global log
        if 'log' not in globals() or not log:
            log = logging.getLogger("vajra.engine")
            if not log.handlers:
                log.setLevel(logging.INFO)
                h = logging.StreamHandler()
                h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
                log.addHandler(h)
        self.cfg = cfg
        self.brains = {} # Format: {(strategy, side): model}
        if not joblib: return
        self.load_brains(brains_dir)

    def load_brains(self, brains_dir):
        if not brains_dir: return
        p = Path(brains_dir)
        if not p.is_dir(): return

        loaded = 0
        for model_file in p.glob("brain_*.joblib"):
            try:
                # brain_{STRATEGY}_{SIDE}.joblib
                parts = model_file.stem.split("_")
                if len(parts) >= 3:
                    strat = parts[1]
                    side = parts[2]
                    self.brains[(strat, side)] = joblib.load(str(model_file))
                    loaded += 1
            except Exception as e:
                log.error(f"Load error {model_file.name}: {e}")
        if loaded > 0:
            log.info(f"Loaded {loaded} localized brain models from {brains_dir}")

    def predict_probability(self, strategy, side, base, adv, iExec, px, pExec):
        b = self.brains.get((strategy, side))
        if not b: return None # Strict fallback block
        try:
            vec = self._build_vec(side, base, adv, iExec, px, pExec, b['feature_names'])
            vec = np.clip(np.nan_to_num(vec, nan=0.0), -1e10, 1e10).astype(np.float32)
            # Classifier predict_proba logic
            prob = b['classifier'].predict_proba(vec)[0][1]
            return prob
        except Exception as e:
            log.error(f"Brain Prediction Failed: {e}\n{traceback.format_exc()}")
            return None

    def _build_vec(self, side, b, a, iExec, px, pExec, fnames):
        d = {**b} 
        for k, v in a.items():
            if hasattr(v, '__getitem__') and len(v) > iExec:
                val = v[iExec]
                d[k] = float(val) if np.isfinite(val) else 0.0
                if d[k] > 1e10: d[k] = 1e10
                elif d[k] < -1e10: d[k] = -1e10
            else: d[k] = 0.0

        # ALIGNMENT FIX
        d.update({
            "pos_vs_swing_h": (px-pExec.last_sh[iExec])/px if px else 0.0,
            "pos_vs_swing_l": (px-pExec.last_sl[iExec])/px if px else 0.0,
            "dist_to_mtf_ema200_pct": (px-a['mtf_ema200_arr'][iExec])/px*100 if px and 'mtf_ema200_arr' in a else 0.0,
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
        self.current_bar_index = {}
        self.last_processed_ts = {}

    def can_open(self, side):
        total_active = len(self.open_trades) + len(self.pending_orders)
        if self.cfg.max_concurrent > 0 and total_active >= self.cfg.max_concurrent: return False
        return True

    def submit_plan(self, plan, bar, force_open=False, fill_price=None, sl_order_id=None):
        if not plan or not self.can_open(plan['side']): return None
        
        # Signal cooldown to prevent trade stacking isolated by symbol
        symbol = plan.get('features', {}).get('symbol', 'UNKNOWN')
        signal_key = f"{symbol}_{plan.get('strat', plan.get('strategy', 'unknown'))}_{plan['side']}"
        last_signal = self.last_signal_time.get(signal_key, -999)

        # Reject identical signals for the next 10 bars
        curr_idx = self.current_bar_index.get(symbol, 0)
        if curr_idx - last_signal < 10:
            return None

        self.last_signal_time[signal_key] = curr_idx

        adx = plan.get('features', {}).get('adx', 25.0)
        ttl = 4
        
        order = {
            **plan,
            "symbol": plan.get("features", {}).get("symbol", self.cfg.symbol),
            "dca_level": 0,
            "risk_factor": plan.get('risk_factor', 1.0),
            "created_ts": bar.get("timestamp", 0),
            "ttl": ttl, 
            "status": "PENDING" 
        }
        
        if force_open and fill_price is not None:
            # Bypass PENDING and go straight to OPEN (Execution Desync Fix)
            planned_risk = abs(order['entry'] - order['sl'])
            if planned_risk < 1e-9: planned_risk = order['entry'] * 0.01

            trade = order.copy()
            trade['avg_price'] = fill_price
            trade['total_size'] = order.get('total_size', 1.0 * order.get('risk_factor', 1.0))
            trade['initial_risk_unit'] = planned_risk
            trade['status'] = 'OPEN'
            trade['fill_ts'] = time.time()
            trade['bars_open'] = 0
            trade['next_safety_price'] = 0.0
            trade['sl_order_id'] = sl_order_id
            trade['can_tp_this_bar'] = False  # <--- CRITICAL FIX

            self.open_trades.append(trade)
            self.mem.record_fill(trade)
            return trade

        self.pending_orders.append(order)
        self.mem.record_signal(order)
        return order

    def step_bar(self, symbol, o, h, l, c, ts=None, swing_high=0.0, swing_low=0.0):
        if ts is not None and ts == self.last_processed_ts.get(symbol): return []
        self.last_processed_ts[symbol] = ts
        self.current_bar_index[symbol] = self.current_bar_index.get(symbol, 0) + 1

        closed = []
        still_open = []
        still_pending = []
        
        # Determine the current bar's timestamp, fallback to 0 if not provided
        current_ts = ts if ts is not None else 0

        fee = (self.cfg.taker_fee_bps + self.cfg.slippage_bps) / 10000.0
        
        for order in self.pending_orders:
            if order.get('symbol', self.cfg.symbol) != symbol:
                still_pending.append(order)
                continue

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
                    if o <= order.get('sl', -1.0):
                        order['ttl'] = -1
                        continue
                    if l <= entry_target:
                        triggered = True
                        fill_px = min(o, entry_target)
                elif side == 'short':
                    if o >= order.get('sl', 99999999.0):
                        order['ttl'] = -1
                        continue
                    if h >= entry_target:
                        triggered = True
                        fill_px = max(o, entry_target)
            elif order_type == 'market':
                triggered = True
                fill_px = entry_target
            else: # breakout
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
                
                trade = order.copy()
                trade['avg_price'] = fill_px
                trade['total_size'] = order.get('total_size', 1.0 * order.get('risk_factor',1.0))
                trade['initial_risk_unit'] = initial_risk
                trade['status'] = 'OPEN'
                trade['fill_ts'] = current_ts
                trade['bars_open'] = 0

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
            if t.get('symbol', self.cfg.symbol) != symbol:
                still_open.append(t)
                continue

            t['bars_open'] = t.get('bars_open', 0) + 1
            side = t['side']
            entry = t['avg_price']
            sl = t['sl']
            tp = t['tp']
            
            can_tp = t.get('can_tp_this_bar', True)
            if not can_tp:
                t['can_tp_this_bar'] = True

            raw_risk = max(t.get('initial_risk_unit', abs(t['entry'] - t['sl'])), 1e-9)

            curr_pnl = (c - entry) if side == 'long' else (entry - c)
            t['pnl_r'] = (curr_pnl / raw_risk)

            intra_max_pnl = (h - entry) if side == 'long' else (entry - l)
            intra_max_pnl_r = (intra_max_pnl / raw_risk)

            t['max_unrealized_pnl_r'] = max(t.get('max_unrealized_pnl_r', -999.0), t['pnl_r'], intra_max_pnl_r)
            t['max_pnl_r'] = max(t.get('max_pnl_r', -99.0), intra_max_pnl_r)

            hit_sl = False
            hit_tp = False
            exit_reason = ''
            
            # Pessimistic Check: SL evaluated first!
            if side == 'long':
                if l <= sl: hit_sl = True; exit_reason = 'sl'
                elif h >= tp and can_tp: hit_tp = True; exit_reason = 'tp'
            else:
                if h >= sl: hit_sl = True; exit_reason = 'sl'
                elif l <= tp and can_tp: hit_tp = True; exit_reason = 'tp'

            if hit_sl or hit_tp:
                if hit_sl:
                    if side == 'long': exit_px = min(o, t['sl']) if o < t['sl'] else t['sl']
                    else: exit_px = max(o, t['sl']) if o > t['sl'] else t['sl']
                else:
                    if side == 'long': exit_px = max(o, tp) if o > tp else tp
                    else: exit_px = min(o, tp) if o < tp else tp

                raw_pnl = (exit_px - entry) if side == 'long' else (entry - exit_px)
                t['pnl_r'] = (raw_pnl / raw_risk)
                t['exit_price'] = exit_px
                t['exit_reason'] = exit_reason
                t['exit_ts'] = current_ts
                
                closed.append(t)
                self.mem.record_exit(t)
                continue

            still_open.append(t)
        
        self.open_trades = still_open
        return closed
def _score_side(f, s):
    if s=='long': return f.get("bos_up",0)+f.get("engulf_bull",0)+f.get("pin_bull",0)+0.5*f.get("htf_up",0)
    else: return f.get("bos_down",0)+f.get("engulf_bear",0)+f.get("pin_bear",0)+0.5*f.get("htf_down",0)

def plan_trade(cfg, f):
    # Backward compatibility stub
    return None

def plan_trade_with_brain(cfg, brain, base, adv, iExec, pExec):
    px = base.get("price")
    if not px: return None
    
    current_atr = pExec.atr14[iExec] if pExec and iExec < len(pExec.atr14) else (px * 0.01)
    sl_atr = base.get("atr_exec_pct",0)*0.01*px*cfg.atr_mult_sl
    
    # DYNAMIC TP SCALING
    bb_width = base.get("bb_width", 0.0)
    if getattr(cfg, 'dynamic_tp_enabled', True):
        # Scale structurally relative to Bollinger Band volatility expansion
        bb_mult = min(max(bb_width / 2.0, 1.5), 5.0)
        tp_atr_dist = current_atr * bb_mult
    else:
        tp_atr_dist = base.get("atr_exec_pct",0)*0.01*px*cfg.atr_mult_tp
        
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
    
    can_long = True
    can_short = True
    
    # SPOOFING PROTECTION
    if bid_ask_imbalance > 0.80: can_long = False
    if bid_ask_imbalance < 0.20: can_short = False
    
    w_pattern = base.get("w_pattern", 0.0); m_pattern = base.get("m_pattern", 0.0)
    fvg_bull = base.get("fvg_bull", 0.0); fvg_bear = base.get("fvg_bear", 0.0)
    sweep_bull = base.get("sweep_bull_p", 0.0); sweep_bear = base.get("sweep_bear_p", 0.0)
    
    swing_high = pExec.last_sh[iExec] if pExec else 0.0
    swing_low = pExec.last_sl[iExec] if pExec else 0.0
    
    htf_sh = adv.get('htf_swing_high', [0.0])[iExec] if 'htf_swing_high' in adv else 0.0
    htf_sl = adv.get('htf_swing_low', [0.0])[iExec] if 'htf_swing_low' in adv else 0.0
    mtf_sh = adv.get('mtf_swing_high', [0.0])[iExec] if 'mtf_swing_high' in adv else 0.0
    mtf_sl = adv.get('mtf_swing_low', [0.0])[iExec] if 'mtf_swing_low' in adv else 0.0

    reversion_penalty = 1.0
    if adx_val > 30: reversion_penalty = 0.5 
    
    # ==========================================================
    # FUZZY LOGIC & MASSIVE STRATEGY EXPANSION
    # ==========================================================
    setup_type = None
    logic_desc = ""
    entry_target = px
    side = None

    # Pre-fetch variables for fuzzy logic
    fvg_bull = base.get("fvg_bull", 0); fvg_bear = base.get("fvg_bear", 0)
    ob_bull = base.get("ob_bull_top", 0); ob_bear = base.get("ob_bear_bot", 0)
    fib_786_l = base.get("fib_786_long", 0); fib_786_s = base.get("fib_786_short", 0)
    macro_high = max(base.get("last_swing_high", px), base.get("last_swing_low", px))
    macro_low = min(base.get("last_swing_high", px), base.get("last_swing_low", px))
    fib_618_l = macro_low + ((macro_high - macro_low) * 0.382)
    fib_618_s = macro_high - ((macro_high - macro_low) * 0.382)
    qm_bull = base.get("qm_bull", 0); qm_bear = base.get("qm_bear", 0)
    spring = base.get("wyckoff_spring", 0); upthrust = base.get("wyckoff_upthrust", 0)

    # Fuzzy Proximity Helper (Phantom Fill Fix)
    def is_tapped(level, buffer_atr_mult=0.50):
        if level <= 0: return False
        buffer = current_atr * buffer_atr_mult
        return (abs(px - level) < buffer) or ((pExec.l[iExec] - buffer) <= level <= (pExec.h[iExec] + buffer))

    # Regime Filtering (Hurst Exponent)
    hurst_val = base.get("hurst", 0.5)
    is_trending_regime = hurst_val > 0.50
    is_ranging_regime = hurst_val <= 0.50

    # Direct Wick Geometry Calculation (Prevents Dictionary Misses)
    total_range = pExec.h[iExec] - pExec.l[iExec]
    total_range_safe = total_range if total_range > 1e-9 else 1e-9
    body_top = max(pExec.o[iExec], pExec.c[iExec])
    body_bottom = min(pExec.o[iExec], pExec.c[iExec])

    curr_lower_wick_pct = (body_bottom - pExec.l[iExec]) / total_range_safe
    curr_upper_wick_pct = (pExec.h[iExec] - body_top) / total_range_safe

    # Dynamic Volatility & Momentum Thresholds
    atr_p_decimal = base.get("atr_percentile_100", 50.0) / 100.0
    dyn_adx_thresh = 20.0 + (atr_p_decimal * 10.0)
    dyn_rvol_thresh = 1.0 + (atr_p_decimal * 0.5)

    # Evaluate Longs
    if can_long:
        side = 'long'
        # Relaxed: Lower wick is at least 20% of the candle range
        is_bull_rejection = curr_lower_wick_pct > 0.20

        # Strat Alpha (Trend Pullbacks) - Prioritize in Trending Regime
        if ((fib_786_l <= px <= fib_618_l) or (ob_bull > 0 and is_tapped(ob_bull))) and (is_bull_rejection or brain is None):
            setup_type = "ALPHA_LONG"
            entry_target = ob_bull if ob_bull > 0 else px
            logic_desc = "Trend Pullback: 3-TF alignment. Structural bounce confirmed on 0.618-0.786 Fib or active OB."
        # Strat Beta (Momentum Breakouts) - Prioritize in Trending Regime
        elif base.get("squeeze_fired", 0) > 0 and base.get("vol_spike", 0) > 0 and base.get("bos_up", 0) > 0:
            setup_type = "BETA_LONG"
            entry_target = px
            logic_desc = "Momentum Breakout: Squeeze fired with volume spike and Structural BOS. Entering momentum explosion."
        # Strat Gamma (Liquidity Traps) - Prioritize in Ranging/Reversion
        elif (sweep_bull > 0 or base.get("sweep_low", 0) > 0) and (is_bull_rejection or brain is None) and base.get("cvd_div_bull", 0) > 0:
            setup_type = "GAMMA_LONG"
            entry_target = base.get("last_swing_low", px)
            logic_desc = "Liquidity Trap: Retail stops hunted with bullish rejection and CVD absorption divergence."
        # Strat Delta (Fractal Mitigation)
        elif fvg_bull > 0 and ob_bull > 0 and (is_tapped(ob_bull) or px <= ob_bull):
            setup_type = "DELTA_LONG"
            entry_target = ob_bull
            logic_desc = "Fractal Mitigation: Fair Value Gap perfectly aligns with institutional Order Block."
        # Strat Epsilon (Quasimodo)
        elif qm_bull > 0 and is_tapped(qm_bull) and (is_bull_rejection or brain is None):
            setup_type = "EPSILON_LONG"
            entry_target = qm_bull
            logic_desc = "Quasimodo Structure: Structural bounce confirmed on the QM left shoulder retest."
        # Strat Zeta (Wyckoff Spring / Judas Swing)
        elif spring > 0 or (base.get("asian_range_swept_dn", 0) > 0 and (is_bull_rejection or brain is None)):
            setup_type = "ZETA_LONG"
            entry_target = px
            logic_desc = "Wyckoff/Judas Spring: HTF/Asian swing swept with immediate volume/CVD reclaim."
        # Strat Omega (Auction Market Theory) - Prioritize in Ranging Regime
        elif (is_bull_rejection or brain is None) and is_tapped(val) and base.get("poc", 0) > entry_target:
            setup_type = "OMEGA_LONG"
            entry_target = val
            logic_desc = "Auction Market Theory: Ranging environment. Bullish rejection confirmed at Value Area Low (VAL)."

    # Evaluate Shorts
    if not setup_type and can_short:
        side = 'short'
        # Relaxed: Upper wick is at least 20% of the candle range
        is_bear_rejection = curr_upper_wick_pct > 0.20

        # Strat Alpha (Trend Pullbacks)
        if ((fib_618_s <= px <= fib_786_s) or (ob_bear > 0 and is_tapped(ob_bear))) and (is_bear_rejection or brain is None):
            setup_type = "ALPHA_SHORT"
            entry_target = ob_bear if ob_bear > 0 else px
            logic_desc = "Trend Pullback: 3-TF alignment. Structural rejection confirmed on 0.618-0.786 Fib or active OB."
        # Strat Beta (Momentum Breakouts)
        elif base.get("squeeze_fired", 0) > 0 and base.get("vol_spike", 0) > 0 and base.get("bos_down", 0) > 0:
            setup_type = "BETA_SHORT"
            entry_target = px
            logic_desc = "Momentum Breakout: Squeeze fired with volume spike and Structural BOS. Entering momentum explosion."
        # Strat Gamma (Liquidity Traps)
        elif (sweep_bear > 0 or base.get("sweep_high", 0) > 0) and (is_bear_rejection or brain is None) and base.get("cvd_div_bear", 0) > 0:
            setup_type = "GAMMA_SHORT"
            entry_target = base.get("last_swing_high", px)
            logic_desc = "Liquidity Trap: Retail stops hunted with bearish rejection and CVD absorption divergence."
        # Strat Delta (Fractal Mitigation)
        elif fvg_bear > 0 and ob_bear > 0 and (is_tapped(ob_bear) or px >= ob_bear):
            setup_type = "DELTA_SHORT"
            entry_target = ob_bear
            logic_desc = "Fractal Mitigation: Fair Value Gap perfectly aligns with institutional Order Block."
        # Strat Epsilon (Quasimodo)
        elif qm_bear > 0 and is_tapped(qm_bear) and (is_bear_rejection or brain is None):
            setup_type = "EPSILON_SHORT"
            entry_target = qm_bear
            logic_desc = "Quasimodo Structure: Structural rejection confirmed on the QM left shoulder retest."
        # Strat Zeta (Wyckoff Upthrust / Judas Swing)
        elif upthrust > 0 or (base.get("asian_range_swept_up", 0) > 0 and (is_bear_rejection or brain is None)):
            setup_type = "ZETA_SHORT"
            entry_target = px
            logic_desc = "Wyckoff/Judas Upthrust: HTF/Asian swing swept with immediate volume/CVD reclaim."
        # Strat Omega (Auction Market Theory)
        elif (is_bear_rejection or brain is None) and is_tapped(vah) and base.get("poc", px) < entry_target:
            setup_type = "OMEGA_SHORT"
            entry_target = vah
            logic_desc = "Auction Market Theory: Ranging environment. Bearish rejection confirmed at Value Area High (VAH)."

    if not setup_type:
        return None

    # ==========================================================
    # RELAX THE MATHEMATICAL STRAITJACKET & DYNAMIC ESCALATION
    # ==========================================================
    # 1. STRICT SNIPER TP & SL LOGIC
    if side == 'long':
        sl = base.get("last_swing_low", entry_target - current_atr) - (current_atr * 0.2)

        # Find the CLOSEST valid resistance for TP (Excluding Infinity)
        bear_ob = base.get("ob_bear_bot", float('inf'))
        vah = base.get("vah", float('inf'))

        valid_targets = [t for t in (bear_ob, vah) if t > entry_target and t != float('inf')]
        tp = min(valid_targets) if valid_targets else entry_target + (current_atr * 1.5)

    else:
        sl = base.get("last_swing_high", entry_target + current_atr) + (current_atr * 0.2)

        # Find the CLOSEST valid support for TP
        bull_ob = base.get("ob_bull_top", 0.0)
        val = base.get("val", 0.0)

        valid_targets = [t for t in (bull_ob, val) if t > 0 and t < entry_target and t != float('inf')]
        tp = max(valid_targets) if valid_targets else entry_target - (current_atr * 1.5)

    # 2. RISK, REWARD, AND RR CALCULATION
    dynamic_risk = abs(entry_target - sl)
    dynamic_reward = abs(tp - entry_target)
    rr = dynamic_reward / max(1e-12, dynamic_risk)

    # 3. THE EV HACK PREVENTION (RR CAP)
    if rr > 2.5:
        rr = 2.5
        if side == 'long':
            tp = entry_target + (dynamic_risk * 2.5)
        else:
            tp = entry_target - (dynamic_risk * 2.5)

    # 4. RR GATES (Live vs Exporter)
    if rr < 1.2 and brain is not None:
        return None # Live Bot: Require reward to be greater than risk
    elif rr < 0.5 and brain is None:
        return None # Exporter: Allow base hits and failed setups

    # ==========================================================
    # THE COMPREHENSIVE ANALYST RATIONALE
    # ==========================================================
    analysis_str = "No AI active."

    # ==========================================================
    # DYNAMIC EV GATE
    # ==========================================================
    ev = rr
    risk_factor = 1.0
    parts = setup_type.split("_")
    strat = parts[0]

    if brain:
        win_prob = brain.predict_probability(strat, side, base, adv, iExec, px, pExec)
        if win_prob is None: return None # Strict Fallback block! Needs model

        min_p = getattr(cfg, 'min_prob_long', 0.51) if side == 'long' else getattr(cfg, 'min_prob_short', 0.51)
        if win_prob < min_p: return None

        # --- MACRO ORACLE VETO ---
        # Sentiment Score: 0 (Extreme Fear) to 100 (Extreme Greed). Default is 50 (Neutral).
        macro_sentiment = base.get("sentiment_score", base.get("macro_sentiment", 50.0))

        if side == 'long' and macro_sentiment < 40:
            win_prob *= 0.8  # 20% penalty for longing into Fear
            logic_desc += " [ORACLE PENALTY: Longing into Bearish Macro]"
        elif side == 'short' and macro_sentiment > 60:
            win_prob *= 0.8  # 20% penalty for shorting into Greed
            logic_desc += " [ORACLE PENALTY: Shorting into Bullish Macro]"

        # Calculate true Expected Value (EV) = (Win% * Reward) - (Loss% * Risk)
        ev = (win_prob * rr) - ((1.0 - win_prob) * 1.0)
        if ev <= 0.1: return None # Only take trades with a positive mathematical edge

        edge = ev
        if getattr(cfg, 'dynamic_risk_scaling', True):
            # Exponential scaling: higher EV pushes risk towards max_risk_factor
            base_risk = 1.0 + (edge ** 1.5)
            risk_factor = min(base_risk, getattr(cfg, 'max_risk_factor', 2.5))
        else:
            risk_factor = 1.0

        # Calculate dynamic reversal warning level (Structural failure point before SL)
        rev_warn = base.get("ema50_L", px)

        prob_desc = "HIGH CONFIDENCE" if win_prob > 0.55 else "EDGE"
        analysis_str = (
            f"SETUP: {setup_type}. LOGIC: {logic_desc} "
            f"AI PROB: {win_prob*100:.1f}% ({prob_desc} - EV: {ev:.2f}R). "
            f"ENTRY: {entry_target:.2f}. TARGET: {tp:.2f} ({rr:.2f} RR). "
            f"REVERSAL WARNING: Consider manual trim if 1H trend violates {rev_warn:.2f}."
        )

    best_plan = {
        "side": side, "entry": entry_target, "sl": sl, "tp": tp, "rr": rr,
        "prob": win_prob if brain else 0.5, # Compatibility
        "key": f"{setup_type}_{side}", "features": base,
        "risk_factor": risk_factor, "strategy": setup_type, "type": "limit",
        "analysis": analysis_str
    }
    return best_plan