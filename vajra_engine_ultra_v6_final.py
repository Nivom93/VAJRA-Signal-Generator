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

@njit(cache=True)
def _rolling_mean(arr, window, min_periods=1):
    n = len(arr)
    out = np.empty(n, dtype=np.float64)
    if window <= 0: return arr.astype(np.float64)
    for i in range(n):
        start_idx = max(0, i - window + 1)
        count = i - start_idx + 1
        if count < min_periods:
            out[i] = np.nan
        else:
            s = 0.0
            for j in range(start_idx, i + 1):
                s += arr[j]
            out[i] = s / count
    return out

@njit(cache=True)
def _rolling_std(arr, window, min_periods=2):
    n = len(arr)
    out = np.empty(n, dtype=np.float64)
    if window <= 0: return np.zeros_like(arr, dtype=np.float64)
    min_p = max(min_periods, 2)
    for i in range(n):
        start_idx = max(0, i - window + 1)
        count = i - start_idx + 1
        if count < min_p:
            out[i] = np.nan
        else:
            s = 0.0
            for j in range(start_idx, i + 1):
                s += arr[j]
            mean = s / count

            var_sum = 0.0
            for j in range(start_idx, i + 1):
                diff = arr[j] - mean
                var_sum += diff * diff
            out[i] = np.sqrt(var_sum / count)
    return out

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

@njit(cache=True)
def _obv_np(close, volume):
    n = len(close)
    obv = np.zeros(n, dtype=np.float64)
    if n == 0: return obv
    for i in range(1, n):
        if close[i] > close[i-1]: obv[i] = obv[i-1] + volume[i]
        elif close[i] < close[i-1]: obv[i] = obv[i-1] - volume[i]
        else: obv[i] = obv[i-1]
    return obv

@njit(cache=True)
def _ewo_np(close):
    # Elliott Wave Oscillator: SMA(5) - SMA(35)
    n = len(close)
    ewo = np.zeros(n, dtype=np.float64)
    if n < 35: return ewo
    sma5 = _rolling_mean(close, 5)
    sma35 = _rolling_mean(close, 35)
    for i in range(n): ewo[i] = sma5[i] - sma35[i]
    return ewo

@njit(cache=True)
def _strict_divergence_np(close, indicator, last_sl, last_sh):
    n = len(close)
    div_bull = np.zeros(n, dtype=np.float64)
    div_bear = np.zeros(n, dtype=np.float64)
    # Track indicator value at the most recent swing low/high
    ind_at_sl = np.zeros(n, dtype=np.float64)
    ind_at_sh = np.zeros(n, dtype=np.float64)
    prev_sl_price = 0.0
    prev_sh_price = 0.0
    for i in range(1, n):
        # Detect when swing low changes (new swing low confirmed)
        if last_sl[i] != last_sl[i-1] and last_sl[i] > 0:
            ind_at_sl[i] = indicator[i]
            prev_sl_price = last_sl[i]
        else:
            ind_at_sl[i] = ind_at_sl[i-1]
        # Detect when swing high changes
        if last_sh[i] != last_sh[i-1] and last_sh[i] > 0:
            ind_at_sh[i] = indicator[i]
            prev_sh_price = last_sh[i]
        else:
            ind_at_sh[i] = ind_at_sh[i-1]

    for i in range(10, n):
        # Bullish Div: Price makes lower low vs last swing, indicator makes higher low
        if close[i] < last_sl[i] and ind_at_sl[i] > 0 and indicator[i] > ind_at_sl[i]:
            div_bull[i] = 1.0
        # Bearish Div: Price makes higher high vs last swing, indicator makes lower high
        if close[i] > last_sh[i] and ind_at_sh[i] != 0 and indicator[i] < ind_at_sh[i]:
            div_bear[i] = 1.0
    return div_bull, div_bear

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
        # 1. Double Bottom (W Pattern) — with tolerance zone
        if swing_lo[i-1] == 0 and low[i] > low[i-1]:
            prev_sl = last_sl[i-5]
            if prev_sl > 0:
                dist = abs(low[i] - prev_sl) / prev_sl
                if dist < 0.008: w_pattern[i] = 1.0

        # 2. Double Top (M Pattern) — with tolerance zone
        if swing_hi[i-1] == 0 and high[i] < high[i-1]:
            prev_sh = last_sh[i-5]
            if prev_sh > 0:
                dist = abs(high[i] - prev_sh) / prev_sh
                if dist < 0.008: m_pattern[i] = 1.0

        # 3. Fair Value Gaps — include near-gaps (gap within 0.15% of closing = near-FVG)
        exact_gap_bull = low[i] - high[i-2]
        if exact_gap_bull > 0:
            fvg_bull[i] = high[i-2] + exact_gap_bull * 0.5  # CE of the gap
        elif close[i] > 0:
            near_gap_pct = (high[i-2] - low[i]) / close[i]
            if near_gap_pct < 0.0015 and near_gap_pct >= 0:
                fvg_bull[i] = (low[i] + high[i-2]) * 0.5

        exact_gap_bear = low[i-2] - high[i]
        if exact_gap_bear > 0:
            fvg_bear[i] = high[i] + exact_gap_bear * 0.5  # CE of the gap
        elif close[i] > 0:
            near_gap_pct = (high[i] - low[i-2]) / close[i]
            if near_gap_pct < 0.0015 and near_gap_pct >= 0:
                fvg_bear[i] = (high[i] + low[i-2]) * 0.5

        # 4. Liquidity Sweeps — require rejection wick (close back inside structure zone)
        curr_sl = last_sl[i]
        if curr_sl > 0 and low[i] < curr_sl and close[i] > curr_sl:
            # Validate: lower wick must be significant (swept + rejected)
            rng = high[i] - low[i]
            if rng > 1e-9:
                lower_wick = (min(open_p[i], close[i]) - low[i]) / rng
                if lower_wick > 0.15:
                    sweep_bull[i] = 1.0

        curr_sh = last_sh[i]
        if curr_sh > 0 and high[i] > curr_sh and close[i] < curr_sh:
            rng = high[i] - low[i]
            if rng > 1e-9:
                upper_wick = (high[i] - max(open_p[i], close[i])) / rng
                if upper_wick > 0.15:
                    sweep_bear[i] = 1.0

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
    atr_mult_sl: float = 0.0; atr_mult_tp: float = 2.0; scalper_rr: float = 2.0
    pullback_atr_mult: float = 0.0
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
    min_prob_long: float = 0.51; min_prob_short: float = 0.55
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
    for i in range(1, n):
        rng = high[i] - low[i]
        if rng < 1e-12:
            continue
        # BOS requires displacement: candle body must be >50% of range (strong commitment)
        body_pct = abs(close[i] - close[i-1]) / rng if rng > 0 else 0.0
        if close[i] > lsh[i] and body_pct > 0.4: bos_up[i]=1
        if close[i] < lsl[i] and body_pct > 0.4: bos_dn[i]=1
        # Sweeps: wick beyond structure but close back inside (liquidity grab)
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
        # Spring: Sweeps last_sl, closes back above it
        # Requires: volume confirmation OR CVD divergence (within 3-bar window)
        curr_sl = last_sl[i-1]
        if curr_sl > 0 and l[i] < curr_sl and c[i] > curr_sl:
            has_vol = False
            has_cvd = False
            for k in range(max(0, i-2), i+1):
                if vol_spike[k] > 0: has_vol = True
                if cvd_div_bull[k] > 0: has_cvd = True
            if has_vol or has_cvd:
                spring[i] = 1.0

        # Upthrust: Sweeps last_sh, closes back below it
        curr_sh = last_sh[i-1]
        if curr_sh > 0 and h[i] > curr_sh and c[i] < curr_sh:
            has_vol = False
            has_cvd = False
            for k in range(max(0, i-2), i+1):
                if vol_spike[k] > 0: has_vol = True
                if cvd_div_bear[k] > 0: has_cvd = True
            if has_vol or has_cvd:
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

        # GOD-TIER METRICS
        self.obv = _obv_np(self.c, self.v)
        self.ewo = _ewo_np(self.c)
        self.obv_div_bull, self.obv_div_bear = _strict_divergence_np(self.c, self.obv, self.last_sl, self.last_sh)
        self.ewo_div_bull, self.ewo_div_bear = _strict_divergence_np(self.c, self.ewo, self.last_sl, self.last_sh)

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

    # Respect the passed-in indices (callers pass iMacro-1 etc. to prevent lookahead)
    idx_Macro = max(0, min(iMacro, len(macro_tf)-1))
    idx_Swing = max(0, min(iSwing, len(swing_tf)-1))
    idx_Htf = max(0, min(iHtf, len(htf)-1))
    idx_Exec = max(0, min(iExec, len(exec_tf)-1))
    
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
    
    # STRICT GOD-TIER FEATURES
    f["obv_div_bull"] = float(pExec.obv_div_bull[idx_Exec]) if hasattr(pExec, 'obv_div_bull') else 0.0
    f["obv_div_bear"] = float(pExec.obv_div_bear[idx_Exec]) if hasattr(pExec, 'obv_div_bear') else 0.0
    f["ewo_div_bull"] = float(pExec.ewo_div_bull[idx_Exec]) if hasattr(pExec, 'ewo_div_bull') else 0.0
    f["ewo_div_bear"] = float(pExec.ewo_div_bear[idx_Exec]) if hasattr(pExec, 'ewo_div_bear') else 0.0

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
    # Killzone session flags (generalize better than raw hour/day features)
    hour = (int(ts) // 3600000) % 24
    f["is_london_session"] = 1.0 if 7 <= hour <= 16 else 0.0
    f["is_ny_session"] = 1.0 if 13 <= hour <= 22 else 0.0

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
        f['rel_strength_divergence'] = np.zeros(len(cl))

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
    # GOD TIER KNOWLEDGE: RSI DIVERGENCE & KELTNER EXTREMES
    # ---------------------------------------------------------
    # Track RSI bullish divergence (Price makes lower low, RSI makes higher low)
    rsi = pExec.rsi14
    n = len(cl)
    bull_div_rsi = np.zeros(n)
    bear_div_rsi = np.zeros(n)
    for i in range(15, n):
        # Lookback window of 15 PRIOR bars (exclude current bar to avoid tautology)
        window_cl = cl[i-15:i]
        window_rsi = rsi[i-15:i]

        # Bullish Divergence: price makes lower low vs prior window low, RSI makes higher low
        min_idx = np.argmin(window_cl)
        if cl[i] < window_cl[min_idx] and rsi[i] > window_rsi[min_idx]:
            bull_div_rsi[i] = 1.0

        # Bearish Divergence: price makes higher high vs prior window high, RSI makes lower high
        max_idx = np.argmax(window_cl)
        if cl[i] > window_cl[max_idx] and rsi[i] < window_rsi[max_idx]:
            bear_div_rsi[i] = 1.0

    f['bull_div_rsi'] = bull_div_rsi
    f['bear_div_rsi'] = bear_div_rsi

    f['dist_to_kc_upper_pct'] = np.where(kc_upper_arr > 0, (cl - kc_upper_arr) / kc_upper_arr * 100.0, 0.0)
    f['dist_to_kc_lower_pct'] = np.where(kc_lower_arr > 0, (cl - kc_lower_arr) / kc_lower_arr * 100.0, 0.0)

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
                    brain_data = joblib.load(str(model_file))

                    self.brains[(strat, side)] = brain_data
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

        # Reject identical signals for the next 3 bars
        curr_idx = self.current_bar_index.get(symbol, 0)
        if curr_idx - last_signal < 3:
            return None

        self.last_signal_time[signal_key] = curr_idx

        adx = plan.get('features', {}).get('adx', 25.0)

        # USE DYNAMIC TIME-IN-FORCE (Defaults to 24 bars / 6 hours to allow pullbacks)
        ttl = getattr(self.cfg, 'time_in_force_decay', 24)
        if ttl <= 0: ttl = 24
        
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
    if not px or not pExec: return None

    current_atr = pExec.atr14[iExec] if iExec < len(pExec.atr14) else (px * 0.01)
    if current_atr < 1e-12: current_atr = px * 0.01
    sl_buffer = current_atr * getattr(cfg, 'atr_mult_sl', 0.15)

    # ==========================================================
    # PHASE 1: MULTI-TIMEFRAME REGIME DETECTION
    # ==========================================================
    macro_up = base.get("macro_up", 0)
    macro_down = base.get("macro_down", 0)
    swing_up = base.get("swing_up", 0)
    swing_down = base.get("swing_down", 0)
    htf_up = base.get("htf_up", 0)
    htf_down = base.get("htf_down", 0)

    bullish_tf_count = int(macro_up > 0) + int(swing_up > 0) + int(htf_up > 0)
    bearish_tf_count = int(macro_down > 0) + int(swing_down > 0) + int(htf_down > 0)

    # HARD REGIME GATE: Block counter-trend trades in strong unidirectional trends
    can_long = True
    can_short = True
    is_reversal_context = False

    if bearish_tf_count == 3:
        # All 3 higher TFs bearish — block longs unless reversal evidence
        can_long = False
    if bullish_tf_count == 3:
        # All 3 higher TFs bullish — block shorts unless reversal evidence
        can_short = False

    # ==========================================================
    # PHASE 2: REVERSAL DETECTION (Override regime gate when evidence is overwhelming)
    # ==========================================================
    # Divergence detection (5-bar temporal lookback)
    bull_obv = any(pExec.obv_div_bull[max(0, iExec-5):iExec+1] > 0) if hasattr(pExec, 'obv_div_bull') else False
    bear_obv = any(pExec.obv_div_bear[max(0, iExec-5):iExec+1] > 0) if hasattr(pExec, 'obv_div_bear') else False
    bull_ewo = any(pExec.ewo_div_bull[max(0, iExec-5):iExec+1] > 0) if hasattr(pExec, 'ewo_div_bull') else False
    bear_ewo = any(pExec.ewo_div_bear[max(0, iExec-5):iExec+1] > 0) if hasattr(pExec, 'ewo_div_bear') else False
    bull_cvd = any(pExec.cvd_div_bull[max(0, iExec-5):iExec+1] > 0) if hasattr(pExec, 'cvd_div_bull') else False
    bear_cvd = any(pExec.cvd_div_bear[max(0, iExec-5):iExec+1] > 0) if hasattr(pExec, 'cvd_div_bear') else False
    bull_rsi_div = any(pExec.div_bull[max(0, iExec-5):iExec+1] > 0) if hasattr(pExec, 'div_bull') else False
    bear_rsi_div = any(pExec.div_bear[max(0, iExec-5):iExec+1] > 0) if hasattr(pExec, 'div_bear') else False

    # Wyckoff spring/upthrust (3-bar lookback)
    has_spring = any(pExec.spring[max(0, iExec-2):iExec+1] > 0) if hasattr(pExec, 'spring') else False
    has_upthrust = any(pExec.upthrust[max(0, iExec-2):iExec+1] > 0) if hasattr(pExec, 'upthrust') else False

    # Count bullish reversal evidence
    bull_reversal_evidence = int(bull_obv) + int(bull_ewo) + int(bull_cvd) + int(bull_rsi_div) + int(has_spring)
    bear_reversal_evidence = int(bear_obv) + int(bear_ewo) + int(bear_cvd) + int(bear_rsi_div) + int(has_upthrust)

    # Unlock counter-trend if 3+ reversal signals converge
    if not can_long and bull_reversal_evidence >= 3:
        can_long = True
        is_reversal_context = True
    if not can_short and bear_reversal_evidence >= 3:
        can_short = True
        is_reversal_context = True

    # ==========================================================
    # PHASE 3: ORACLE MACRO GATE
    # ==========================================================
    oracle_sentiment = base.get("macro_sentiment", 0.0)
    btc_bullish = base.get("btc_bullish", 1.0)
    bid_ask_imbalance = base.get("bid_ask_imbalance", 0.5)

    # Oracle hard veto: deeply negative sentiment blocks longs, deeply positive blocks shorts
    if oracle_sentiment < -0.6 and not is_reversal_context:
        can_long = False
    if oracle_sentiment > 0.6 and not is_reversal_context:
        can_short = False

    # BTC regime: if BTC is bearish and this isn't BTC, reduce long conviction
    if btc_bullish < 0.5 and not is_reversal_context:
        if bearish_tf_count >= 2:
            can_long = False

    # Spoofing defense
    if bid_ask_imbalance > 0.80: can_long = False
    if bid_ask_imbalance < 0.20: can_short = False

    if not can_long and not can_short: return None

    # ==========================================================
    # PHASE 4: PRECISE STRUCTURAL ENTRY DETECTION
    # ==========================================================
    px = pExec.c[iExec]
    low = pExec.l[iExec]
    high = pExec.h[iExec]
    adx_val = base.get("adx", 25.0)

    ob_bull_bot = base.get("ob_bull_bot", 0)
    ob_bull_top = base.get("ob_bull_top", 0)
    ob_bear_bot = base.get("ob_bear_bot", 0)
    ob_bear_top = base.get("ob_bear_top", 0)
    ob_bull_ce = (ob_bull_top + ob_bull_bot) / 2 if ob_bull_top > 0 else 0
    ob_bear_ce = (ob_bear_top + ob_bear_bot) / 2 if ob_bear_bot > 0 else 0

    ob_bull_fresh = 0 < base.get("bars_since_ob_bull", 999) <= 20
    ob_bear_fresh = 0 < base.get("bars_since_ob_bear", 999) <= 20

    fvg_bull = base.get("fvg_bull", 0)
    fvg_bear = base.get("fvg_bear", 0)
    fvg_tol = px * 0.001
    qm_zone = current_atr * 0.3

    sweep_bull = base.get("sweep_bull_p", 0.0)
    sweep_bear = base.get("sweep_bear_p", 0.0)

    # Wick rejection detection (3-bar window)
    is_bull_rejection = False
    is_bear_rejection = False
    for idx in range(max(0, iExec - 2), iExec + 1):
        if idx >= len(pExec.h): continue
        c_range = pExec.h[idx] - pExec.l[idx]
        if c_range > 1e-9:
            body_top = max(pExec.o[idx], pExec.c[idx])
            body_bot = min(pExec.o[idx], pExec.c[idx])
            if (body_bot - pExec.l[idx]) / c_range > 0.25: is_bull_rejection = True
            if (pExec.h[idx] - body_top) / c_range > 0.25: is_bear_rejection = True

    # Volume confirmation (current or recent bar)
    has_volume = any(pExec.vol_spike[max(0, iExec-2):iExec+1] > 0) if hasattr(pExec, 'vol_spike') else False
    rvol = base.get("rvol", 1.0)
    has_elevated_vol = rvol > 1.2

    setup_type = None
    logic_desc = ""
    side = None

    # ---- LONG SETUPS ----
    if can_long:
        side = 'long'

        # ALPHA_LONG: OB Consequent Encroachment + Rejection Wick
        if (ob_bull_ce > 0 and ob_bull_fresh and low <= ob_bull_ce and px > ob_bull_bot
                and is_bull_rejection):
            setup_type = "ALPHA_LONG"
            logic_desc = "OB CE mitigation with wick rejection confirmation."

        # BETA_LONG: Multi-divergence with structural BOS + momentum exhaustion
        elif ((bull_ewo or bull_obv or bull_cvd) and base.get("bos_up", 0) > 0
                and base.get("rsi_14", 50) < 45):
            setup_type = "BETA_LONG"
            logic_desc = "Divergence confluence + BOS + RSI exhaustion."

        # GAMMA_LONG: Quasimodo retest with volume
        elif (base.get("qm_bull", 0) > 0
                and low <= (base.get("qm_bull", 0) + qm_zone) and px > base.get("qm_bull", 0)
                and (has_volume or has_elevated_vol)):
            setup_type = "GAMMA_LONG"
            logic_desc = "QM structural retest with volume confirmation."

        # DELTA_LONG: FVG mitigation
        elif (fvg_bull > 0 and low <= (fvg_bull + fvg_tol) and px > (fvg_bull - fvg_tol)
                and is_bull_rejection):
            setup_type = "DELTA_LONG"
            logic_desc = "FVG CE mitigation with rejection."

        # EPSILON_LONG: Wyckoff Spring (sweep + reclaim + volume)
        elif has_spring and sweep_bull > 0:
            setup_type = "EPSILON_LONG"
            logic_desc = "Wyckoff spring: liquidity sweep + reclaim."

    # ---- SHORT SETUPS ----
    if not setup_type and can_short:
        side = 'short'

        # ALPHA_SHORT: OB CE + Rejection Wick
        if (ob_bear_ce > 0 and ob_bear_fresh and high >= ob_bear_ce and px < ob_bear_top
                and is_bear_rejection):
            setup_type = "ALPHA_SHORT"
            logic_desc = "OB CE mitigation with wick rejection confirmation."

        # BETA_SHORT: Multi-divergence with BOS + momentum exhaustion
        elif ((bear_ewo or bear_obv or bear_cvd) and base.get("bos_down", 0) > 0
                and base.get("rsi_14", 50) > 55):
            setup_type = "BETA_SHORT"
            logic_desc = "Divergence confluence + BOS + RSI exhaustion."

        # GAMMA_SHORT: QM retest with volume
        elif (base.get("qm_bear", 0) > 0
                and high >= (base.get("qm_bear", 0) - qm_zone) and px < base.get("qm_bear", 0)
                and (has_volume or has_elevated_vol)):
            setup_type = "GAMMA_SHORT"
            logic_desc = "QM structural retest with volume confirmation."

        # DELTA_SHORT: FVG mitigation
        elif (fvg_bear > 0 and low <= (fvg_bear + fvg_tol) and px < (fvg_bear + fvg_tol)
                and is_bear_rejection):
            setup_type = "DELTA_SHORT"
            logic_desc = "FVG CE mitigation with rejection."

        # EPSILON_SHORT: Wyckoff Upthrust
        elif has_upthrust and sweep_bear > 0:
            setup_type = "EPSILON_SHORT"
            logic_desc = "Wyckoff upthrust: liquidity sweep + reclaim."

    if not setup_type: return None

    # ==========================================================
    # PHASE 5: SETUP-SPECIFIC PRECISE SL (Invalidation Point)
    # ==========================================================
    pullback_dist = current_atr * getattr(cfg, 'pullback_atr_mult', 0.0)
    entry_target = px - pullback_dist if side == 'long' else px + pullback_dist

    swing_high = pExec.last_sh[iExec]
    swing_low = pExec.last_sl[iExec]

    htf_sh = adv.get('htf_swing_high', np.zeros(1))
    htf_sl = adv.get('htf_swing_low', np.zeros(1))
    mtf_sh = adv.get('mtf_swing_high', np.zeros(1))
    mtf_sl = adv.get('mtf_swing_low', np.zeros(1))
    htf_sh_val = float(htf_sh[iExec]) if hasattr(htf_sh, '__getitem__') and iExec < len(htf_sh) else 0.0
    htf_sl_val = float(htf_sl[iExec]) if hasattr(htf_sl, '__getitem__') and iExec < len(htf_sl) else 0.0
    mtf_sh_val = float(mtf_sh[iExec]) if hasattr(mtf_sh, '__getitem__') and iExec < len(mtf_sh) else 0.0
    mtf_sl_val = float(mtf_sl[iExec]) if hasattr(mtf_sl, '__getitem__') and iExec < len(mtf_sl) else 0.0

    strat_base = setup_type.split("_")[0]

    if side == 'long':
        # Setup-specific SL: use the tightest structural invalidation
        if strat_base == "ALPHA" and ob_bull_bot > 0:
            sl = ob_bull_bot - sl_buffer  # Below OB body = OB invalidated
        elif strat_base == "GAMMA" and base.get("qm_bull", 0) > 0:
            sl = base.get("qm_bull", 0) - sl_buffer  # Below QM pivot
        elif strat_base == "DELTA" and fvg_bull > 0:
            # Below FVG low boundary
            fvg_low = fvg_bull - fvg_tol
            sl = fvg_low - sl_buffer
        else:
            sl = min(swing_low, low) - sl_buffer

        if sl >= entry_target: return None
        risk_distance = entry_target - sl

        # Structural TP targets
        possible_tps = [t for t in [ob_bear_bot, base.get("vah", 0), htf_sh_val, mtf_sh_val] if t > entry_target]
        possible_tps.sort()
        if not possible_tps:
            possible_tps = [entry_target + (risk_distance * getattr(cfg, 'atr_mult_tp', 3.0))]

        selected_tp = None
        for t in possible_tps:
            curr_rr = (t - entry_target) / risk_distance
            if curr_rr >= getattr(cfg, 'min_rr', 2.0):
                if curr_rr > getattr(cfg, 'atr_mult_tp', 3.5):
                    selected_tp = entry_target + (risk_distance * getattr(cfg, 'atr_mult_tp', 3.5))
                else:
                    selected_tp = t
                break

        if selected_tp is None: return None
        tp = selected_tp
        rr = (tp - entry_target) / risk_distance

    else:  # short
        if strat_base == "ALPHA" and ob_bear_top > 0:
            sl = ob_bear_top + sl_buffer
        elif strat_base == "GAMMA" and base.get("qm_bear", 0) > 0:
            sl = base.get("qm_bear", 0) + sl_buffer
        elif strat_base == "DELTA" and fvg_bear > 0:
            fvg_high = fvg_bear + fvg_tol
            sl = fvg_high + sl_buffer
        else:
            sl = max(swing_high, high) + sl_buffer

        if sl <= entry_target: return None
        risk_distance = sl - entry_target

        possible_tps = [t for t in [ob_bull_top, base.get("val", 0), htf_sl_val, mtf_sl_val] if 0 < t < entry_target]
        possible_tps.sort(reverse=True)
        if not possible_tps:
            possible_tps = [entry_target - (risk_distance * getattr(cfg, 'atr_mult_tp', 3.5))]

        selected_tp = None
        for t in possible_tps:
            curr_rr = (entry_target - t) / risk_distance
            if curr_rr >= getattr(cfg, 'min_rr', 2.0):
                if curr_rr > getattr(cfg, 'atr_mult_tp', 3.5):
                    selected_tp = entry_target - (risk_distance * getattr(cfg, 'atr_mult_tp', 3.5))
                else:
                    selected_tp = t
                break

        if selected_tp is None: return None
        tp = selected_tp
        rr = (entry_target - tp) / risk_distance

    # Minimum target distance gate
    if abs(tp - entry_target) / entry_target * 100.0 < getattr(cfg, 'min_target_dist_pct', 0.15):
        return None

    # ==========================================================
    # PHASE 6: AI PROBABILITY GATE & EV CALCULATION
    # ==========================================================
    ev = rr
    risk_factor = 1.0
    strat = strat_base
    win_prob = 0.5

    rev_warn = swing_low if side == 'long' else swing_high
    analysis_str = f"SETUP: {setup_type}. {logic_desc}"

    if brain:
        prob = brain.predict_probability(strat, side, base, adv, iExec, px, pExec)
        if prob is None: return None
        win_prob = prob

        min_prob = getattr(cfg, 'min_prob_long', 0.50) if side == 'long' else getattr(cfg, 'min_prob_short', 0.50)
        if win_prob < min_prob:
            return None

        ev = (win_prob * rr) - ((1.0 - win_prob) * 1.0)
        if ev <= getattr(cfg, 'min_ev', 0.0): return None

        if getattr(cfg, 'dynamic_risk_scaling', True):
            risk_factor = min(1.0 + (ev ** 1.5), getattr(cfg, 'max_risk_factor', 2.5))

        regime = "REVERSAL" if is_reversal_context else "TREND"
        confidence = "HIGH" if win_prob > 0.60 else "MODERATE"
        analysis_str = (
            f"SETUP: {setup_type} [{regime}]. {logic_desc} "
            f"AI: {win_prob*100:.1f}% ({confidence}, EV: {ev:.2f}R). "
            f"R:R {rr:.1f}. TF alignment: {bullish_tf_count}B/{bearish_tf_count}S. "
            f"Invalidation: {rev_warn:.2f}."
        )

    return {
        "side": side, "entry": entry_target, "sl": sl, "tp": tp, "rr": rr,
        "prob": win_prob, "key": f"{setup_type}_{side}", "features": base,
        "risk_factor": risk_factor, "strategy": setup_type, "type": "market",
        "analysis": analysis_str
    }