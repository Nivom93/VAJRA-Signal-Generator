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
# PHASE 6: CORRELATION-AWARE POSITION SIZING
# ==============================================================================
# Approximate 4h correlation matrix for major crypto perps (2023-2024 average).
# Conservative estimates — refresh quarterly from your own analysis.
# Used to scale risk_factor when multiple correlated positions are open simultaneously.
_CORRELATION_MATRIX = {
    ("BTC", "ETH"): 0.85,
    ("BTC", "SOL"): 0.78,
    ("BTC", "AVAX"): 0.75,
    ("BTC", "MATIC"): 0.72,
    ("ETH", "SOL"): 0.82,
    ("ETH", "AVAX"): 0.80,
    ("ETH", "MATIC"): 0.85,
    ("SOL", "AVAX"): 0.85,
    # Add more pairs as you trade them
}

def _get_correlation(sym_a: str, sym_b: str) -> float:
    """Symmetric correlation lookup. Returns 0.7 default for unknown pairs (conservative)."""
    if sym_a == sym_b:
        return 1.0
    a = sym_a.split('/')[0].upper()
    b = sym_b.split('/')[0].upper()
    return _CORRELATION_MATRIX.get((a, b), _CORRELATION_MATRIX.get((b, a), 0.7))

# ==============================================================================
# PHASE 6 DIAGNOSTIC COUNTERS — Track why trades are rejected
# ==============================================================================
_p6_counters = {
    "reached_phase6": 0,
    "no_brain": 0,
    "specialist_none": 0,
    "specialist_below_thresh": 0,
    "meta_below_thresh": 0,
    "ev_too_low": 0,
    "risk_dist_too_small": 0,
    "strategy_cap_rejected": 0,
    "cooldown_rejected": 0,
    "confluence_too_low": 0,
    "funding_window_rejected": 0,
    "passed": 0,
    "filled": 0,                  # plan was triggered and entered a position
    "expired_unfilled": 0,        # plan TTL expired with no fill
    "stopped_pre_entry": 0,       # SL was breached before entry filled (limit-order race)
    "specialist_prob_sum": 0.0,
    "specialist_prob_count": 0,
    "meta_prob_sum": 0.0,
    "meta_prob_count": 0,
    "per_strategy_passed": {},    # {strategy_base: count}
    "per_strategy_filled": {},    # {strategy_base: count}
}

def reset_p6_counters():
    """Reset diagnostic counters (call before each backtest run)."""
    for k in _p6_counters:
        if isinstance(_p6_counters[k], dict):
            _p6_counters[k] = {}
        elif isinstance(_p6_counters[k], int):
            _p6_counters[k] = 0
        else:
            _p6_counters[k] = 0.0

def log_p6_summary():
    """Print a summary of Phase 6 gate rejections and fill-rate honesty."""
    c = _p6_counters
    log.info("=" * 60)
    log.info("PHASE 6 AI GATE DIAGNOSTIC SUMMARY")
    log.info("=" * 60)
    log.info(f"  Setups reaching Phase 6:       {c['reached_phase6']}")
    log.info(f"  No brain loaded for setup:     {c['no_brain']}")
    log.info(f"  Specialist returned None:      {c['specialist_none']}")
    log.info(f"  Specialist below threshold:    {c['specialist_below_thresh']}")
    log.info(f"  Meta-Brain below threshold:    {c['meta_below_thresh']}")
    log.info(f"  EV too low:                    {c['ev_too_low']}")
    log.info(f"  Risk distance too small:       {c['risk_dist_too_small']}")
    log.info(f"  Strategy cap rejected:         {c['strategy_cap_rejected']}")
    log.info(f"  Cooldown rejected:             {c['cooldown_rejected']}")
    log.info(f"  Confluence too low (<2.0):      {c['confluence_too_low']}")
    log.info(f"  Funding-window rejected:       {c['funding_window_rejected']}")
    log.info(f"  PLANS GENERATED (Phase 6 pass): {c['passed']}")
    log.info(f"  --- of which ---")
    log.info(f"  Plans FILLED (entered position): {c['filled']}")
    log.info(f"  Plans EXPIRED unfilled:          {c['expired_unfilled']}")
    log.info(f"  Plans STOPPED pre-entry:         {c['stopped_pre_entry']}")
    if c['passed'] > 0:
        fill_rate = 100.0 * c['filled'] / c['passed']
        log.info(f"  → FILL RATE: {fill_rate:.1f}%  (anything below 30% indicates limit-order starvation)")
    if c['specialist_prob_count'] > 0:
        log.info(f"  Avg specialist probability: {c['specialist_prob_sum']/c['specialist_prob_count']:.4f}")
    if c['meta_prob_count'] > 0:
        log.info(f"  Avg meta probability:       {c['meta_prob_sum']/c['meta_prob_count']:.4f}")
    # Per-strategy fill-rate breakdown
    log.info("  --- PER-STRATEGY FILL RATE ---")
    all_strats = set(c['per_strategy_passed'].keys()) | set(c['per_strategy_filled'].keys())
    for s in sorted(all_strats):
        p = c['per_strategy_passed'].get(s, 0)
        f = c['per_strategy_filled'].get(s, 0)
        rate = (100.0 * f / p) if p > 0 else 0.0
        log.info(f"    {s:<20} passed={p:>5}  filled={f:>5}  rate={rate:>5.1f}%")
    log.info("=" * 60)


_TF_MS_UNIT = {"m": 60_000, "h": 3_600_000, "d": 86_400_000, "w": 7 * 86_400_000}

def _tf_to_ms(tf: str) -> int:
    """Convert a timeframe string like '15m' or '4h' to milliseconds."""
    tf = tf.strip().lower()
    if tf.endswith("ms"):
        return int(tf[:-2])
    num = int("".join(ch for ch in tf if ch.isdigit()))
    unit = "".join(ch for ch in tf if ch.isalpha())
    return num * _TF_MS_UNIT.get(unit, 60_000)


def compute_min_risk_distance(cfg, entry_price: float) -> float:
    """Return the minimum allowable stop distance (in price units) such that
    round-trip friction never exceeds ``cfg.max_friction_pct`` of 1R.

    Formula:  min_risk = (entry_price × round_trip_bps / 10_000) / max_friction_pct

    If ``cfg.min_risk_distance_usd > 0`` it acts as a hard override, bypassing
    the dynamic calculation entirely.
    """
    hard_floor = getattr(cfg, "min_risk_distance_usd", 0.0)
    if hard_floor > 0:
        return hard_floor

    maker_bps = getattr(cfg, "maker_fee_bps", 0.0)
    taker_bps = getattr(cfg, "taker_fee_bps", 0.0)
    slip_bps = getattr(cfg, "slippage_bps", 0.0)
    # Round-trip: maker entry + taker exit + slippage both legs
    round_trip_bps = maker_bps + taker_bps + 2.0 * slip_bps
    max_fric = getattr(cfg, "max_friction_pct", 0.20)
    if max_fric <= 0:
        return 0.0
    return (entry_price * round_trip_bps / 10_000.0) / max_fric


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
    """Fixed divergence detection: captures indicator value at the bar where
    swing was confirmed (accounting for right=3 confirmation delay), and
    tracks previous swing indicator value for proper comparison."""
    n = len(close)
    div_bull = np.zeros(n, dtype=np.float64)
    div_bear = np.zeros(n, dtype=np.float64)
    # Track indicator value at swing confirmation points
    ind_at_sl = np.zeros(n, dtype=np.float64)
    ind_at_sh = np.zeros(n, dtype=np.float64)
    # Track PREVIOUS swing indicator values (for comparing two swing points)
    prev_ind_at_sl = 0.0
    prev_ind_at_sh = 0.0
    prev_sl_price = 0.0
    prev_sh_price = 0.0

    for i in range(1, n):
        # Detect when swing low changes (new swing low confirmed)
        if last_sl[i] != last_sl[i-1] and last_sl[i] > 0:
            # Save previous swing's indicator value before overwriting
            prev_ind_at_sl = ind_at_sl[i-1]
            prev_sl_price = last_sl[i-1]
            # Capture indicator at the actual swing low bar (i-3 due to right=3 confirmation)
            swing_bar = max(0, i - 3)
            ind_at_sl[i] = indicator[swing_bar]
        else:
            ind_at_sl[i] = ind_at_sl[i-1]
        # Detect when swing high changes
        if last_sh[i] != last_sh[i-1] and last_sh[i] > 0:
            prev_ind_at_sh = ind_at_sh[i-1]
            prev_sh_price = last_sh[i-1]
            swing_bar = max(0, i - 3)
            ind_at_sh[i] = indicator[swing_bar]
        else:
            ind_at_sh[i] = ind_at_sh[i-1]

    for i in range(10, n):
        # Bullish Div: Price makes lower low, indicator makes higher low
        # Compare current swing low vs previous swing low
        if close[i] < last_sl[i] and ind_at_sl[i] > 0 and indicator[i] > ind_at_sl[i]:
            div_bull[i] = 1.0
        # Also check proximity: price within 0.5% of swing low with indicator divergence
        elif last_sl[i] > 0 and close[i] > 0:
            dist_pct = abs(close[i] - last_sl[i]) / close[i]
            if dist_pct < 0.005 and ind_at_sl[i] > 0 and indicator[i] > ind_at_sl[i] * 1.05:
                div_bull[i] = 1.0

        # Bearish Div: Price makes higher high, indicator makes lower high
        if close[i] > last_sh[i] and ind_at_sh[i] != 0 and indicator[i] < ind_at_sh[i]:
            div_bear[i] = 1.0
        elif last_sh[i] > 0 and close[i] > 0:
            dist_pct = abs(close[i] - last_sh[i]) / close[i]
            if dist_pct < 0.005 and ind_at_sh[i] != 0 and indicator[i] < ind_at_sh[i] * 0.95:
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
    """ADX using proper Wilder's smoothing: smoothed = (prev * (n-1) + current) / n"""
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
    # Wilder's smoothing: first n periods = simple sum, then recursive
    tr_s = np.zeros(sz); dm_plus_s = np.zeros(sz); dm_minus_s = np.zeros(sz)
    # Seed: sum of first n periods
    for i in range(1, min(n + 1, sz)):
        tr_s[n] += tr[i]
        dm_plus_s[n] += dm_plus[i]
        dm_minus_s[n] += dm_minus[i]
    # Wilder's recursive: smoothed = (prev * (n-1) + current) / n
    for i in range(n + 1, sz):
        tr_s[i] = (tr_s[i-1] * (n - 1) + tr[i]) / n
        dm_plus_s[i] = (dm_plus_s[i-1] * (n - 1) + dm_plus[i]) / n
        dm_minus_s[i] = (dm_minus_s[i-1] * (n - 1) + dm_minus[i]) / n
    dx = np.zeros(sz)
    for i in range(n, sz):
        di_plus = 100.0 * _safe_divide(dm_plus_s[i], tr_s[i])
        di_minus = 100.0 * _safe_divide(dm_minus_s[i], tr_s[i])
        sum_di = di_plus + di_minus
        dx[i] = 100.0 * _safe_divide(abs(di_plus - di_minus), sum_di)
    # ADX = Wilder's smoothed DX
    adx = np.zeros(sz)
    # Seed ADX: average of first n DX values after n
    if sz > 2 * n:
        dx_sum = 0.0
        for i in range(n, 2 * n):
            dx_sum += dx[i]
        adx[2 * n - 1] = dx_sum / n
        for i in range(2 * n, sz):
            adx[i] = (adx[i-1] * (n - 1) + dx[i]) / n
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
    avg_gain = _ema_np(gain, 14); avg_loss = _ema_np(loss, 14)
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
    risk_per_trade: float = 0.01; min_rr: float = 2.5; rr: float = 2.0
    atr_mult_sl: float = 0.0; atr_mult_tp: float = 3.0; scalper_rr: float = 2.0
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
    
    # Strict Maker-only execution on entries: saves ~0.05% per trade vs Taker fees.
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
    max_friction_pct: float = 0.20
    min_risk_distance_usd: float = 0.0
    account_notional: float = 1000.0; db_path: str = "vajra.sqlite"
    verbose: bool = True; dynamic_confluence: bool = True; dvol_enable: bool = False
    skip_log_throttle: int = 50; paper_mode: bool = True
    max_trades_per_strategy: int = 0    # 0 = unlimited
    strategy_cooldown_bars: int = 0     # 0 = no cooldown
    strict_brain_validation: bool = False  # If True, raise RuntimeError when ALL brains are noise-tier

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
    # Accumulation/Distribution phase detection
    accum_phase = np.zeros(n)
    distrib_phase = np.zeros(n)

    for i in range(10, n):
        # Extended lookback (10 bars) for volume/CVD confirmation
        vol_lookback = 7

        # Spring: Sweeps last_sl, closes back above it
        curr_sl = last_sl[i-1]
        if curr_sl > 0 and l[i] < curr_sl and c[i] > curr_sl:
            has_vol = False
            has_cvd = False
            for k in range(max(0, i - vol_lookback), i + 1):
                if vol_spike[k] > 0: has_vol = True
                if cvd_div_bull[k] > 0: has_cvd = True
            if has_vol or has_cvd:
                spring[i] = 1.0

        # Upthrust: Sweeps last_sh, closes back below it
        curr_sh = last_sh[i-1]
        if curr_sh > 0 and h[i] > curr_sh and c[i] < curr_sh:
            has_vol = False
            has_cvd = False
            for k in range(max(0, i - vol_lookback), i + 1):
                if vol_spike[k] > 0: has_vol = True
                if cvd_div_bear[k] > 0: has_cvd = True
            if has_vol or has_cvd:
                upthrust[i] = 1.0

        # ---- ACCUMULATION PHASE DETECTION (Wyckoff Phase C/D) ----
        # Accumulation = price consolidates near support + volume absorption
        # (small bodies, high volume = selling being absorbed by smart money)
        if i >= 20:
            # Check if price has been ranging near swing low for 10+ bars
            range_count = 0
            absorption_count = 0
            atr_local = 0.0
            for ab in range(i - 20, i):
                atr_local += (h[ab] - l[ab])
            atr_local /= 20.0
            if atr_local < 1e-9:
                atr_local = 1e-9

            for ab in range(i - 10, i):
                dist_to_sl = abs(c[ab] - curr_sl) / atr_local if curr_sl > 0 else 99.0
                if dist_to_sl < 2.0:
                    range_count += 1
                # Absorption: high volume + small body (selling absorbed)
                body = abs(c[ab] - c[ab - 1]) if ab > 0 else 0.0
                rng_bar = h[ab] - l[ab]
                if rng_bar > 1e-9 and vol_spike[ab] > 0 and (body / rng_bar) < 0.3:
                    absorption_count += 1

            if range_count >= 6 and absorption_count >= 1:
                accum_phase[i] = 1.0

        # ---- DISTRIBUTION PHASE DETECTION ----
        if i >= 20:
            range_count = 0
            absorption_count = 0
            atr_local_d = 0.0
            for db in range(i - 20, i):
                atr_local_d += (h[db] - l[db])
            atr_local_d /= 20.0
            if atr_local_d < 1e-9:
                atr_local_d = 1e-9

            for db in range(i - 10, i):
                dist_to_sh = abs(c[db] - curr_sh) / atr_local_d if curr_sh > 0 else 99.0
                if dist_to_sh < 2.0:
                    range_count += 1
                body = abs(c[db] - c[db - 1]) if db > 0 else 0.0
                rng_bar = h[db] - l[db]
                if rng_bar > 1e-9 and vol_spike[db] > 0 and (body / rng_bar) < 0.3:
                    absorption_count += 1

            if range_count >= 6 and absorption_count >= 1:
                distrib_phase[i] = 1.0

    return spring, upthrust, accum_phase, distrib_phase

@njit(cache=True)
def _multi_swing_fractal_range(high, low, swing_hi, swing_lo, lookback_swings=3):
    """Track the highest/lowest from the last N confirmed swing points.
    This gives a proper major fractal range for Fibonacci anchoring instead of
    just using the last single swing high/low."""
    n = len(high)
    frac_high = np.zeros(n)
    frac_low = np.zeros(n)

    # Circular buffers for recent swing highs and lows
    recent_sh = np.zeros(lookback_swings)
    recent_sl = np.zeros(lookback_swings)
    sh_idx = 0; sl_idx = 0
    sh_count = 0; sl_count = 0

    for i in range(n):
        if swing_hi[i] == 1 and i >= 3:
            recent_sh[sh_idx % lookback_swings] = high[i - 3]  # right=3 confirmation
            sh_idx += 1
            sh_count = min(sh_count + 1, lookback_swings)

        if swing_lo[i] == 1 and i >= 3:
            recent_sl[sl_idx % lookback_swings] = low[i - 3]
            sl_idx += 1
            sl_count = min(sl_count + 1, lookback_swings)

        # Find max of recent swing highs and min of recent swing lows
        if sh_count > 0:
            max_val = recent_sh[0]
            for j in range(1, sh_count):
                if recent_sh[j] > max_val:
                    max_val = recent_sh[j]
            frac_high[i] = max_val
        else:
            frac_high[i] = high[i]

        if sl_count > 0:
            min_val = recent_sl[0]
            for j in range(1, sl_count):
                if recent_sl[j] < min_val and recent_sl[j] > 0:
                    min_val = recent_sl[j]
            frac_low[i] = min_val
        else:
            frac_low[i] = low[i]

    return frac_high, frac_low


@njit(cache=True)
def _market_structure_np(high, low, sh_conf, sl_conf, right=3):
    """Classify market structure by comparing consecutive swing highs and swing lows.

    For each bar, tracks:
    - Whether each swing high is HH or LH relative to the previous swing high
    - Whether each swing low is HL or LL relative to the previous swing low
    - Structural trend state: +1 (bullish: HH+HL), -1 (bearish: LH+LL), 0 (ranging/mixed)
    - Trend strength: count of consecutive structure points confirming the trend
    - Structure break flag: 1 when the current bar breaks the prevailing structure

    Returns 7 arrays:
      is_hh, is_hl, is_lh, is_ll — rolling flags at confirmation bar
      struct_trend — +1 bullish, -1 bearish, 0 ranging
      struct_strength — consecutive confirmations (0-10 capped)
      struct_break — 1 on the bar where structure first invalidates
    """
    n = high.size
    is_hh = np.zeros(n, dtype=np.float64)
    is_hl = np.zeros(n, dtype=np.float64)
    is_lh = np.zeros(n, dtype=np.float64)
    is_ll = np.zeros(n, dtype=np.float64)
    struct_trend = np.zeros(n, dtype=np.float64)
    struct_strength = np.zeros(n, dtype=np.float64)
    struct_break = np.zeros(n, dtype=np.float64)

    # Track the last two swing highs and swing lows for comparison
    prev_sh = 0.0  # previous confirmed swing high price
    curr_sh = 0.0  # most recent confirmed swing high price
    prev_sl = 0.0
    curr_sl = 0.0
    sh_count = 0
    sl_count = 0

    # Running structural state
    last_sh_type = 0   # +1 = HH, -1 = LH
    last_sl_type = 0   # +1 = HL, -1 = LL
    trend = 0          # +1 bullish, -1 bearish, 0 mixed
    strength = 0       # consecutive confirmations

    for i in range(n):
        # Check for new confirmed swing high
        if sh_conf[i] == 1 and i >= right:
            swing_price = high[i - right]
            if sh_count >= 1 and curr_sh > 0:
                prev_sh = curr_sh
                curr_sh = swing_price
                if curr_sh > prev_sh:
                    is_hh[i] = 1.0
                    last_sh_type = 1  # HH
                elif curr_sh < prev_sh:
                    is_lh[i] = 1.0
                    last_sh_type = -1  # LH
                # else equal — no change
            else:
                curr_sh = swing_price
            sh_count += 1

        # Check for new confirmed swing low
        if sl_conf[i] == 1 and i >= right:
            swing_price = low[i - right]
            if sl_count >= 1 and curr_sl > 0:
                prev_sl = curr_sl
                curr_sl = swing_price
                if curr_sl > prev_sl:
                    is_hl[i] = 1.0
                    last_sl_type = 1  # HL
                elif curr_sl < prev_sl:
                    is_ll[i] = 1.0
                    last_sl_type = -1  # LL
            else:
                curr_sl = swing_price
            sl_count += 1

        # Determine structural trend from latest swing classifications
        old_trend = trend
        if last_sh_type == 1 and last_sl_type == 1:
            # HH + HL = bullish structure
            trend = 1
        elif last_sh_type == -1 and last_sl_type == -1:
            # LH + LL = bearish structure
            trend = -1
        elif last_sh_type != 0 and last_sl_type != 0:
            # Mixed (HH + LL or LH + HL) = ranging / transitional
            trend = 0
        # else: not enough data yet, trend stays 0

        # Track strength: consecutive confirmations in the same direction
        if trend == old_trend and trend != 0:
            if sh_conf[i] == 1 or sl_conf[i] == 1:
                strength = min(strength + 1, 10)
        elif trend != old_trend:
            strength = 1  # Reset on change

        # Structure break detection:
        # A break occurs when trend changes from established (+1 or -1) to something else
        if old_trend != 0 and trend != old_trend and (sh_conf[i] == 1 or sl_conf[i] == 1):
            struct_break[i] = 1.0

        struct_trend[i] = float(trend)
        struct_strength[i] = float(strength)

    return is_hh, is_hl, is_lh, is_ll, struct_trend, struct_strength, struct_break


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

        # Track last UNFILLED FVG — carry forward until price fills it
        n = len(self.c)
        last_fvg_bull = np.zeros(n)
        last_fvg_bear = np.zeros(n)
        curr_bull = 0.0
        curr_bear = 0.0
        for i in range(n):
            if self.fvg_bull_p[i] > 0:
                curr_bull = self.fvg_bull_p[i]
            if curr_bull > 0 and self.c[i] > curr_bull:
                curr_bull = 0.0  # filled — invalidate
            last_fvg_bull[i] = curr_bull

            if self.fvg_bear_p[i] > 0:
                curr_bear = self.fvg_bear_p[i]
            if curr_bear > 0 and self.c[i] < curr_bear:
                curr_bear = 0.0  # filled — invalidate
            last_fvg_bear[i] = curr_bear
        self.last_fvg_bull = last_fvg_bull
        self.last_fvg_bear = last_fvg_bear

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
        self.spring, self.upthrust, self.accum_phase, self.distrib_phase = _detect_wyckoff(self.h, self.l, self.c, self.vol_spike, self.cvd_div_bull, self.cvd_div_bear, self.last_sl, self.last_sh)

        hours = pd.to_datetime(df.timestamp, unit='ms', utc=True).dt.hour.values
        days = pd.to_datetime(df.timestamp, unit='ms', utc=True).dt.dayofyear.values
        self.asian_high, self.asian_low = _asian_range_np(self.h, self.l, hours, days)

        self.atr_percentile_100 = _rolling_percentile_np(self.atr14, 100)
        self.rolling_vwap_20 = _rolling_vwap_np((self.h + self.l + self.c) / 3.0, self.v, 20)
        self.dc_high_20, self.dc_low_20 = _donchian_channels_np(self.h, self.l, 20)

        # Cache EMA100 for performance (avoid recalculation in confluence_features)
        self.ema100 = _ema_np(self.c, 100)

        # Multi-swing fractal range: track the highest high and lowest low
        # from the last 3 confirmed swing points (major structure)
        self.fractal_high, self.fractal_low = _multi_swing_fractal_range(
            self.h, self.l, self.swing_hi, self.swing_lo, lookback_swings=3
        )

        # Market Structure Reading: HH/HL/LH/LL classification + trend state
        (self.is_hh, self.is_hl, self.is_lh, self.is_ll,
         self.struct_trend, self.struct_strength, self.struct_break
        ) = _market_structure_np(self.h, self.l, self.swing_hi, self.swing_lo, right=3)

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
    
    # ---- PROPER ChoCH DETECTION (Change of Character) ----
    # ChoCH = BOS in current direction + prior established trend in opposite direction
    # Scan a 20-bar window: if prior trend had 2+ BOS in one direction,
    # and now we get a BOS in the opposite direction = genuine character change
    choch_up_val = 0.0
    choch_down_val = 0.0
    if pExec.bos_up[idx_Exec]:
        prior_bear_count = 0
        for cb in range(max(0, idx_Exec - 20), idx_Exec):
            if pExec.bos_down[cb]: prior_bear_count += 1
        # Need at least 2 prior bearish BOS = established downtrend, then bullish BOS = ChoCH
        if prior_bear_count >= 2:
            choch_up_val = 1.0
    if pExec.bos_down[idx_Exec]:
        prior_bull_count = 0
        for cb in range(max(0, idx_Exec - 20), idx_Exec):
            if pExec.bos_up[cb]: prior_bull_count += 1
        if prior_bull_count >= 2:
            choch_down_val = 1.0
    f["choch_up"] = choch_up_val
    f["choch_down"] = choch_down_val

    # ---- MARKET STRUCTURE READING (HH/HL/LH/LL + Trend State) ----
    # Structural trend from ACTUAL price action, not lagging EMA crossovers
    f["struct_trend"] = float(pExec.struct_trend[idx_Exec])         # +1 bull, -1 bear, 0 ranging
    f["struct_strength"] = float(pExec.struct_strength[idx_Exec])   # consecutive confirmations (0-10)
    f["struct_break"] = float(pExec.struct_break[idx_Exec])         # 1.0 on structure break bar

    # Recent HH/HL/LH/LL counts in lookback window (structure quality)
    struct_lookback = min(30, idx_Exec)
    recent_hh = 0.0; recent_hl = 0.0; recent_lh = 0.0; recent_ll = 0.0
    for sb in range(max(0, idx_Exec - struct_lookback), idx_Exec + 1):
        recent_hh += pExec.is_hh[sb]
        recent_hl += pExec.is_hl[sb]
        recent_lh += pExec.is_lh[sb]
        recent_ll += pExec.is_ll[sb]
    f["recent_hh_count"] = recent_hh
    f["recent_hl_count"] = recent_hl
    f["recent_lh_count"] = recent_lh
    f["recent_ll_count"] = recent_ll

    # Structure bias score: +1 = fully bullish structure, -1 = fully bearish
    bull_struct_pts = recent_hh + recent_hl
    bear_struct_pts = recent_lh + recent_ll
    total_struct_pts = bull_struct_pts + bear_struct_pts
    f["struct_bias_score"] = float((bull_struct_pts - bear_struct_pts) / max(total_struct_pts, 1.0))

    # HTF structure reading (swing timeframe gives higher-level structure)
    f["htf_struct_trend"] = float(pHtf.struct_trend[idx_Htf])
    f["htf_struct_strength"] = float(pHtf.struct_strength[idx_Htf])
    f["htf_struct_break"] = float(pHtf.struct_break[idx_Htf])

    # Macro structure (daily)
    f["macro_struct_trend"] = float(pMacro.struct_trend[idx_Macro])
    # FIX: Clamp to finite float — pMacro.struct_strength can be np.int64 which
    # np.isfinite handles but float() conversion must be explicit to survive
    # the _build_vec pipeline (which checks np.isfinite on the float cast).
    _mss = pMacro.struct_strength[idx_Macro]
    f["macro_struct_strength"] = float(_mss) if np.isfinite(float(_mss)) else 0.0

    # Multi-TF structure alignment (price-action based, not EMA-based)
    f["struct_align_bull"] = float(
        (1.0 if pMacro.struct_trend[idx_Macro] > 0 else 0.0) +
        (1.0 if pHtf.struct_trend[idx_Htf] > 0 else 0.0) +
        (1.0 if pExec.struct_trend[idx_Exec] > 0 else 0.0)
    )
    f["struct_align_bear"] = float(
        (1.0 if pMacro.struct_trend[idx_Macro] < 0 else 0.0) +
        (1.0 if pHtf.struct_trend[idx_Htf] < 0 else 0.0) +
        (1.0 if pExec.struct_trend[idx_Exec] < 0 else 0.0)
    )

    # ---- DISPLACEMENT SEQUENCE DETECTION ----
    # Displacement = large body candle (>1.5x ATR) that breaks structure
    curr_atr_cf = pExec.atr14[idx_Exec] if idx_Exec < len(pExec.atr14) else px * 0.01
    # Count displacement candles in recent window (institutional activity)
    disp_bull_count = 0
    disp_bear_count = 0
    for db in range(max(0, idx_Exec - 10), idx_Exec + 1):
        if db < len(pExec.c):
            db_body = abs(pExec.c[db] - pExec.o[db])
            db_atr = pExec.atr14[db] if db < len(pExec.atr14) else px * 0.01
            if db_body > db_atr * 1.5:
                if pExec.c[db] > pExec.o[db]: disp_bull_count += 1
                else: disp_bear_count += 1
    f["displacement_bull_count"] = float(disp_bull_count)
    f["displacement_bear_count"] = float(disp_bear_count)
    # Displacement + BOS = institutional intent confirmed
    f["displacement_bos_bull"] = 1.0 if disp_bull_count > 0 and pExec.bos_up[idx_Exec] else 0.0
    f["displacement_bos_bear"] = 1.0 if disp_bear_count > 0 and pExec.bos_down[idx_Exec] else 0.0

    f["bars_since_ob_bull"] = float(pExec.bars_since_ob_bull[idx_Exec])
    f["bars_since_ob_bear"] = float(pExec.bars_since_ob_bear[idx_Exec])

    # Safely pull velocity_atr_3 if precomputed, else 0
    if hasattr(pExec, 'velocity_atr_3'):
        f["velocity_atr_3"] = float(pExec.velocity_atr_3[idx_Exec])

    f["vol_spike"] = float(pExec.vol_spike[idx_Exec])

    # ── SESSION TIMING (high-liquidity windows produce 70% of directional moves) ──
    try:
        _bar_ts_ms = int(exec_tf.iloc[idx_Exec]['timestamp'])
        _bar_dt = pd.to_datetime(_bar_ts_ms, unit='ms', utc=True)
        _bar_hour = _bar_dt.hour
    except Exception:
        _bar_hour = 12  # neutral fallback
    f["is_london_open"]    = 1.0 if 7  <= _bar_hour <= 10 else 0.0
    f["is_ny_open"]        = 1.0 if 13 <= _bar_hour <= 16 else 0.0
    f["is_high_liquidity"] = 1.0 if 7  <= _bar_hour <= 17 else 0.0
    f["is_dead_zone"]      = 1.0 if (_bar_hour >= 22 or _bar_hour <= 5) else 0.0
    f["is_asian_session"]  = 1.0 if 0  <= _bar_hour <= 6  else 0.0

    # ── PRIOR CANDLE MOMENTUM (direction persistence feature) ──
    _prev_idx = max(0, idx_Exec - 1)
    _prev_c = float(pExec.c[_prev_idx])
    _prev_o = float(pExec.o[_prev_idx])
    _curr_atr_cf = float(pExec.atr14[idx_Exec]) if idx_Exec < len(pExec.atr14) else px * 0.003
    f["prev_candle_bull"]      = 1.0 if _prev_c > _prev_o else 0.0
    f["prev_candle_bear"]      = 1.0 if _prev_c < _prev_o else 0.0
    f["prev_candle_size_atr"]  = abs(_prev_c - _prev_o) / max(_curr_atr_cf, 1e-9)
    # Consecutive same-direction candles (momentum persistence)
    _prev2_idx = max(0, idx_Exec - 2)
    _prev2_c = float(pExec.c[_prev2_idx])
    _prev2_o = float(pExec.o[_prev2_idx])
    f["two_bull_consecutive"]  = 1.0 if (_prev_c > _prev_o and _prev2_c > _prev2_o) else 0.0
    f["two_bear_consecutive"]  = 1.0 if (_prev_c < _prev_o and _prev2_c < _prev2_o) else 0.0

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

    f["vwap_z_score"] = float((px - pExec.rolling_vwap_20[idx_Exec]) / max(pExec.atr14[idx_Exec], 1e-9))
    kc_up = pExec.kc_upper[idx_Exec]
    kc_lo = pExec.kc_lower[idx_Exec]
    f["dist_to_kc_upper_pct"] = float((px - kc_up) / kc_up * 100.0) if kc_up > 0 else 0.0
    f["dist_to_kc_lower_pct"] = float((px - kc_lo) / kc_lo * 100.0) if kc_lo > 0 else 0.0

    f["rsi_14"] = float(pExec.rsi14[idx_Exec])
    
    # STRICT GOD-TIER FEATURES
    f["obv_div_bull"] = float(pExec.obv_div_bull[idx_Exec]) if hasattr(pExec, 'obv_div_bull') else 0.0
    f["obv_div_bear"] = float(pExec.obv_div_bear[idx_Exec]) if hasattr(pExec, 'obv_div_bear') else 0.0
    f["ewo_div_bull"] = float(pExec.ewo_div_bull[idx_Exec]) if hasattr(pExec, 'ewo_div_bull') else 0.0
    f["ewo_div_bear"] = float(pExec.ewo_div_bear[idx_Exec]) if hasattr(pExec, 'ewo_div_bear') else 0.0

    f["w_pattern"] = float(pExec.w_pattern[idx_Exec])
    f["m_pattern"] = float(pExec.m_pattern[idx_Exec])
    f["fvg_bull"] = float(pExec.last_fvg_bull[idx_Exec])
    f["fvg_bear"] = float(pExec.last_fvg_bear[idx_Exec])
    # Carry sweep signal 3 bars forward — entry confirmation takes 1-2 bars
    f["sweep_bull_p"] = float(np.any(pExec.sweep_bull_p[max(0, idx_Exec-2):idx_Exec+1] > 0))
    f["sweep_bear_p"] = float(np.any(pExec.sweep_bear_p[max(0, idx_Exec-2):idx_Exec+1] > 0))

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
    f["wyckoff_accum"] = float(pExec.accum_phase[idx_Exec])
    f["wyckoff_distrib"] = float(pExec.distrib_phase[idx_Exec])

    # QM quality scoring — reward QMs with displacement confirmation
    qm_bull_quality = 0.0
    if f["qm_bull"] > 0:
        qm_bull_quality = 1.0
        # Bonus: if displacement BOS preceded the QM setup
        if f.get("displacement_bos_bull", 0) > 0: qm_bull_quality = 2.0
        # Bonus: if accumulation phase detected near QM level
        if f["wyckoff_accum"] > 0: qm_bull_quality += 1.0
    qm_bear_quality = 0.0
    if f["qm_bear"] > 0:
        qm_bear_quality = 1.0
        if f.get("displacement_bos_bear", 0) > 0: qm_bear_quality = 2.0
        if f["wyckoff_distrib"] > 0: qm_bear_quality += 1.0
    f["qm_bull_quality"] = qm_bull_quality
    f["qm_bear_quality"] = qm_bear_quality

    # ---- LIQUIDITY POOL TRACKING ----
    # Equal highs/lows = stacked liquidity (stop losses clustered)
    eq_high_count = 0
    eq_low_count = 0
    if idx_Exec >= 20:
        tol_liq = curr_atr_cf * 0.15  # Tolerance = 15% of ATR
        for lp in range(max(0, idx_Exec - 20), idx_Exec):
            if lp < len(pExec.last_sh):
                # Equal highs within tolerance
                if abs(pExec.last_sh[lp] - pExec.last_sh[idx_Exec]) < tol_liq and pExec.last_sh[lp] > 0:
                    eq_high_count += 1
                if abs(pExec.last_sl[lp] - pExec.last_sl[idx_Exec]) < tol_liq and pExec.last_sl[lp] > 0:
                    eq_low_count += 1
    f["equal_highs_count"] = float(eq_high_count)
    f["equal_lows_count"] = float(eq_low_count)
    # Distance to nearest liquidity pool
    f["dist_to_liq_high_atr"] = (pExec.last_sh[idx_Exec] - px) / (curr_atr_cf + 1e-12) if pExec.last_sh[idx_Exec] > px else 0.0
    f["dist_to_liq_low_atr"] = (px - pExec.last_sl[idx_Exec]) / (curr_atr_cf + 1e-12) if pExec.last_sl[idx_Exec] < px else 0.0

    f["asian_high"] = float(pExec.asian_high[idx_Exec])
    f["asian_low"] = float(pExec.asian_low[idx_Exec])
    f["dc_high_20"] = float(pExec.dc_high_20[idx_Exec])
    f["dc_low_20"] = float(pExec.dc_low_20[idx_Exec])
    
    f["dist_asian_high_pct"] = (px - f["asian_high"]) / px_safe * 100
    f["dist_asian_low_pct"] = (px - f["asian_low"]) / px_safe * 100

    # Rolling VWAP for base dict access (used by TP targeting in plan_trade_with_brain)
    f["rolling_vwap_20"] = float(pExec.rolling_vwap_20[idx_Exec])

    # SMC & Fibonacci Deep Wick Geometry (EXPANDED: Full Fib Suite + OTE Zone)
    # Use multi-swing fractal range for proper major structure Fibonacci anchoring
    macro_high = float(pExec.fractal_high[idx_Exec])
    macro_low = float(pExec.fractal_low[idx_Exec])
    # Fallback to single swing if fractal range is invalid
    if macro_high <= macro_low or macro_high <= 0:
        macro_high = max(f["last_swing_high"], f["last_swing_low"])
        macro_low = min(f["last_swing_high"], f["last_swing_low"])
    fractal_range = macro_high - macro_low

    if fractal_range > 0:
        # Standard Fibonacci retracements
        f["fib_382_long"] = macro_low + (fractal_range * 0.618)    # 38.2% retrace for longs
        f["fib_500_long"] = macro_low + (fractal_range * 0.500)    # 50% retrace (equilibrium)
        f["fib_618_long"] = macro_low + (fractal_range * 0.382)    # 61.8% retrace (golden ratio)
        f["fib_786_long"] = macro_low + (fractal_range * 0.214)    # 78.6% retrace (deep)
        f["fib_886_long"] = macro_low + (fractal_range * 0.114)    # 88.6% retrace (extreme)
        f["fib_382_short"] = macro_high - (fractal_range * 0.618)
        f["fib_500_short"] = macro_high - (fractal_range * 0.500)
        f["fib_618_short"] = macro_high - (fractal_range * 0.382)
        f["fib_786_short"] = macro_high - (fractal_range * 0.214)
        f["fib_886_short"] = macro_high - (fractal_range * 0.114)

        # ICT Optimal Trade Entry (OTE) Zone: 62-79% retracement
        ote_top_long = macro_low + (fractal_range * 0.382)   # 61.8% retrace
        ote_bot_long = macro_low + (fractal_range * 0.214)   # 78.6% retrace
        f["in_ote_zone_long"] = 1.0 if ote_bot_long <= px <= ote_top_long else 0.0
        ote_top_short = macro_high - (fractal_range * 0.214)
        ote_bot_short = macro_high - (fractal_range * 0.382)
        f["in_ote_zone_short"] = 1.0 if ote_bot_short <= px <= ote_top_short else 0.0

        # Premium/Discount zone (above/below 50% = premium/discount)
        f["fib_position_pct"] = (px - macro_low) / fractal_range * 100.0  # 0-100 scale
        f["is_discount_zone"] = 1.0 if px < (macro_low + fractal_range * 0.5) else 0.0
        f["is_premium_zone"] = 1.0 if px > (macro_low + fractal_range * 0.5) else 0.0

        # Fibonacci extensions (for TP targeting)
        f["fib_ext_1272"] = macro_high + (fractal_range * 0.272)
        f["fib_ext_1618"] = macro_high + (fractal_range * 0.618)
    else:
        f["fib_382_long"] = f["fib_500_long"] = f["fib_618_long"] = px
        f["fib_786_long"] = f["fib_886_long"] = px
        f["fib_382_short"] = f["fib_500_short"] = f["fib_618_short"] = px
        f["fib_786_short"] = f["fib_886_short"] = px
        f["in_ote_zone_long"] = 0.0; f["in_ote_zone_short"] = 0.0
        f["fib_position_pct"] = 50.0
        f["is_discount_zone"] = 0.0; f["is_premium_zone"] = 0.0
        f["fib_ext_1272"] = f["fib_ext_1618"] = px

    f["dist_fib_618_long_pct"] = (px - f["fib_618_long"]) / px_safe * 100
    f["dist_fib_786_long_pct"] = (px - f["fib_786_long"]) / px_safe * 100
    f["dist_fib_886_long_pct"] = (px - f["fib_886_long"]) / px_safe * 100
    f["dist_fib_618_short_pct"] = (px - f["fib_618_short"]) / px_safe * 100
    f["dist_fib_786_short_pct"] = (px - f["fib_786_short"]) / px_safe * 100
    f["dist_fib_886_short_pct"] = (px - f["fib_886_short"]) / px_safe * 100

    # Displacement detection: large body candle (>2x ATR) = institutional commitment
    curr_atr = pExec.atr14[idx_Exec] if idx_Exec < len(pExec.atr14) else px * 0.01
    curr_body = abs(pExec.c[idx_Exec] - pExec.o[idx_Exec])
    f["is_displacement"] = 1.0 if curr_body > curr_atr * 1.5 else 0.0
    f["displacement_ratio"] = curr_body / (curr_atr + 1e-12)

    # Recent displacement lookback (any displacement in last 5 bars)
    recent_disp_bull = 0.0
    recent_disp_bear = 0.0
    for rb in range(max(0, idx_Exec - 4), idx_Exec + 1):
        if rb < len(pExec.c):
            rb_body = abs(pExec.c[rb] - pExec.o[rb])
            rb_atr = pExec.atr14[rb] if rb < len(pExec.atr14) else px * 0.01
            if rb_body > rb_atr * 1.5:
                if pExec.c[rb] > pExec.o[rb]: recent_disp_bull = 1.0
                else: recent_disp_bear = 1.0
    f["recent_displacement_bull"] = recent_disp_bull
    f["recent_displacement_bear"] = recent_disp_bear

    # Liquidity void detection: gap between current bar low and prior bar high (or vice versa)
    if idx_Exec > 0:
        gap_up = pExec.l[idx_Exec] - pExec.h[idx_Exec - 1]
        gap_dn = pExec.l[idx_Exec - 1] - pExec.h[idx_Exec]
        f["liquidity_void_up"] = max(0, gap_up) / (curr_atr + 1e-12)
        f["liquidity_void_dn"] = max(0, gap_dn) / (curr_atr + 1e-12)
    else:
        f["liquidity_void_up"] = 0.0
        f["liquidity_void_dn"] = 0.0

    # Check if Asian Range was just swept
    f["asian_range_swept_up"] = 1.0 if pExec.h[idx_Exec] > pExec.asian_high[idx_Exec] and pExec.c[idx_Exec] < pExec.asian_high[idx_Exec] else 0.0
    f["asian_range_swept_dn"] = 1.0 if pExec.l[idx_Exec] < pExec.asian_low[idx_Exec] and pExec.c[idx_Exec] > pExec.asian_low[idx_Exec] else 0.0
    
    ts = exec_tf.timestamp.iloc[idx_Exec]
    # Killzone session flags — timezone-correct (handles BST/GMT and EST/EDT)
    from vajra_macro_data import compute_session_flags
    _sess = compute_session_flags(int(ts))
    f["is_london_session"] = _sess["is_london_session"]
    f["is_ny_session"] = _sess["is_ny_session"]

    # ORB breakout flags — scalar for plan_trade_with_brain (LAMBDA strategy)
    asian_h = float(pExec.asian_high[idx_Exec])
    asian_l = float(pExec.asian_low[idx_Exec])
    prev_c = float(pExec.c[max(0, idx_Exec - 1)])
    _is_active = 7 <= hour <= 16
    f["orb_bull"] = 1.0 if (_is_active and asian_h > 0 and px > asian_h and prev_c <= asian_h) else 0.0
    f["orb_bear"] = 1.0 if (_is_active and asian_l > 0 and px < asian_l and prev_c >= asian_l) else 0.0

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
    ema20_gt = 1.0 if (pExec.ema20[idx_Exec] > pExec.ema100[idx_Exec] if hasattr(pExec, 'ema100') else 0) else 0.0
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
    
    # FIX: Use searchsorted for robust HTF→LTF alignment (prevents all-zero
    # arrays when pd.reindex ffill fails on non-overlapping timestamp indices).
    _macro_atr_pct = pMacro.atr14 / (np.abs(pMacro.c) + eps) * 100
    _macro_ts = macro_tf.timestamp.values
    _exec_ts = exec_tf.timestamp.values
    _macro_idx = np.searchsorted(_macro_ts, _exec_ts, side='right') - 1
    # shift(1) equivalent: use idx-1 to enforce one-candle lag (closed candle only)
    _macro_idx_lagged = np.clip(_macro_idx - 1, 0, len(_macro_atr_pct) - 1)
    htf_s = np.where(_macro_idx >= 1, _macro_atr_pct[_macro_idx_lagged], 0.0)
    f['atr_pct_htf_aligned']=htf_s

    # FIX: Same searchsorted alignment for MTF trend strength
    _swing_ts = swing_tf.timestamp.values
    _swing_idx = np.searchsorted(_swing_ts, _exec_ts, side='right') - 1
    _swing_idx_lagged = np.clip(_swing_idx - 1, 0, len(f['trend_strength_mtf_arr']) - 1)
    mtf_s = np.where(_swing_idx >= 1, f['trend_strength_mtf_arr'][_swing_idx_lagged], 0.0)
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

    # Market Structure arrays (HH/HL/LH/LL + trend state)
    f['struct_trend_arr'] = pExec.struct_trend
    f['struct_strength_arr'] = pExec.struct_strength
    f['struct_break_arr'] = pExec.struct_break

    f['div_bull_arr'] = pExec.div_bull
    f['div_bear_arr'] = pExec.div_bear
    
    # FIX: Flatten btc_close_arr to 1-D and coerce length to guard against
    # shape mismatches (e.g. (n,1) from DataFrame slicing) that silently
    # pushed execution into the all-zeros else branch.
    _btc_ok = False
    if btc_close_arr is not None:
        btc_close_arr = np.asarray(btc_close_arr).ravel()
        if len(btc_close_arr) != len(cl):
            # Trim or pad to exact exec_tf length so the branch always fires
            if len(btc_close_arr) > len(cl):
                btc_close_arr = btc_close_arr[:len(cl)]
            else:
                btc_close_arr = np.pad(btc_close_arr, (0, len(cl) - len(btc_close_arr)),
                                       mode='edge')
        _btc_ok = True

    if _btc_ok:
        safe_btc = np.where(btc_close_arr == 0, 1.0, btc_close_arr)
        eth_btc_ratio = cl / safe_btc
        f['eth_btc_rsi'] = _rsi14_np(eth_btc_ratio)
        ema20_ratio = _ema_np(eth_btc_ratio, 20)
        ema100_ratio = _ema_np(eth_btc_ratio, 100)
        f['eth_btc_trend_score'] = (ema20_ratio - ema100_ratio) / (ema100_ratio + eps) * 1000.0
        price_weak = cl < pExec.ema50   # use pExec directly (ema50_L may be dropped later)
        ratio_strong = eth_btc_ratio > _ema_np(eth_btc_ratio, 50)
        f['rel_strength_divergence'] = np.where(price_weak & ratio_strong, 1.0, 0.0)
    else:
        f['eth_btc_rsi'] = np.full(len(cl), 50.0)
        f['eth_btc_trend_score'] = np.zeros(len(cl))
        f['rel_strength_divergence'] = np.zeros(len(cl))

    # ---------------------------------------------------------
    # DIRECTIVE 1: MULTI-TIMEFRAME VOLATILITY SQUEEZE
    # ---------------------------------------------------------
    # FIX: pExec.bb_width is normalised by SMA (from _bb_bands_np), but the
    # old kcw_arr was normalised by EMA.  The denominator mismatch (SMA vs
    # EMA) caused the percentage-width comparison to diverge from the true
    # TTM squeeze condition, producing all-zero is_ttm_squeeze arrays when
    # the SMA-normalised BB width was marginally wider than the
    # EMA-normalised KC width.  Fix: compare ABSOLUTE widths directly,
    # eliminating the normalisation entirely.  This matches the canonical
    # TTM squeeze definition (BB bands inside KC bands).
    kc_upper_arr, kc_lower_arr = _keltner_channels_np(cl, pExec.h, pExec.l, 20, 1.5)
    bb_abs_width = pExec.bb_upper - pExec.bb_lower
    kc_abs_width = kc_upper_arr - kc_lower_arr
    # Guard warmup: first 20 bars have bb_upper/bb_lower = 0 (not yet computed)
    _bb_valid = np.arange(len(cl)) >= 20
    f['is_ttm_squeeze'] = np.where(_bb_valid & (bb_abs_width < kc_abs_width), 1.0, 0.0)

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

    # Killzone Session Flags (EXPANDED)
    hours = pd.to_datetime(exec_tf.timestamp, unit='ms', utc=True).dt.hour.values
    f['is_london_killzone'] = np.where((hours >= 7) & (hours <= 10), 1.0, 0.0)
    f['is_ny_killzone'] = np.where((hours >= 13) & (hours <= 16), 1.0, 0.0)
    f['is_asian_range'] = np.where((hours >= 0) & (hours <= 6), 1.0, 0.0)
    # London-NY overlap = highest liquidity period (13:00-16:00 UTC)
    f['is_london_ny_overlap'] = np.where((hours >= 13) & (hours <= 16), 1.0, 0.0)
    # Pre-market / post-market dead zones (low liquidity = bad for entries)
    f['is_low_liquidity'] = np.where((hours >= 22) | (hours <= 1), 1.0, 0.0)

    # Opening Range Breakout (ORB) — detect break of Asian range high/low
    asian_high_arr = pExec.asian_high
    asian_low_arr = pExec.asian_low
    # ORB signals: first break of Asian range during London/NY sessions
    orb_bull = np.zeros(n)
    orb_bear = np.zeros(n)
    for orb_i in range(1, n):
        if hours[orb_i] >= 7 and hours[orb_i] <= 16:  # Active sessions only
            if asian_high_arr[orb_i] > 0 and cl[orb_i] > asian_high_arr[orb_i]:
                # Confirm: was below Asian high in previous bar
                if orb_i > 0 and cl[orb_i - 1] <= asian_high_arr[orb_i]:
                    orb_bull[orb_i] = 1.0
            if asian_low_arr[orb_i] > 0 and cl[orb_i] < asian_low_arr[orb_i]:
                if orb_i > 0 and cl[orb_i - 1] >= asian_low_arr[orb_i]:
                    orb_bear[orb_i] = 1.0
    f['orb_bull'] = orb_bull
    f['orb_bear'] = orb_bear

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
    # DIRECTIVE 5: MOMENTUM PERSISTENCE & ACCELERATION FEATURES
    # ---------------------------------------------------------
    # RSI Rate of Change — captures momentum acceleration/deceleration
    f['rsi_roc_5'] = _roc(pExec.rsi14, 5)
    f['rsi_roc_10'] = _roc(pExec.rsi14, 10)

    # MACD Histogram slope — momentum persistence indicator
    macd_hist = pExec.macd - pExec.macd_sig
    f['macd_hist_slope_3'] = _roc(macd_hist, 3)
    f['macd_hist_slope_5'] = _roc(macd_hist, 5)

    # Stochastic %K/%D proxy via RSI normalization
    rsi_min_14 = pd.Series(pExec.rsi14).rolling(14, min_periods=1).min().to_numpy()
    rsi_max_14 = pd.Series(pExec.rsi14).rolling(14, min_periods=1).max().to_numpy()
    rsi_range = rsi_max_14 - rsi_min_14
    safe_rsi_range = np.where(rsi_range > 1e-9, rsi_range, 1.0)  # avoid divide-by-zero warning
    f['stoch_rsi_k'] = np.where(rsi_range > 1e-9, (pExec.rsi14 - rsi_min_14) / safe_rsi_range * 100.0, 50.0)
    f['stoch_rsi_d'] = _ema_np(f['stoch_rsi_k'], 3)

    # ---------------------------------------------------------
    # DIRECTIVE 6: CANDLE PATTERN FEATURES (numeric for brain)
    # ---------------------------------------------------------
    body_size_arr = np.abs(cl - pExec.o)
    total_range_arr = pExec.h - pExec.l
    safe_range = np.where(total_range_arr > 1e-9, total_range_arr, 1e-9)

    # Engulfing pattern detection
    prev_body = np.abs(np.roll(cl, 1) - np.roll(pExec.o, 1))
    curr_body = body_size_arr
    bullish_close = cl > pExec.o
    bearish_close = cl < pExec.o
    prev_bearish = np.roll(cl, 1) < np.roll(pExec.o, 1)
    prev_bullish = np.roll(cl, 1) > np.roll(pExec.o, 1)
    f['engulf_bull_signal'] = np.where(bullish_close & prev_bearish & (curr_body > prev_body * 1.1), 1.0, 0.0)
    f['engulf_bear_signal'] = np.where(bearish_close & prev_bullish & (curr_body > prev_body * 1.1), 1.0, 0.0)
    f['engulf_bull_signal'][0] = 0.0
    f['engulf_bear_signal'][0] = 0.0

    # Pin bar strength (normalized wick dominance)
    f['pin_bull_strength'] = (np.minimum(pExec.o, cl) - pExec.l) / safe_range
    f['pin_bear_strength'] = (pExec.h - np.maximum(pExec.o, cl)) / safe_range

    # Inside bar (contraction) — prior bar engulfs current
    prev_high = np.roll(pExec.h, 1)
    prev_low = np.roll(pExec.l, 1)
    f['is_inside_bar'] = np.where((pExec.h <= prev_high) & (pExec.l >= prev_low), 1.0, 0.0)
    f['is_inside_bar'][0] = 0.0

    # Body-to-range ratio (doji detection when low)
    f['body_range_ratio'] = body_size_arr / safe_range

    # ---------------------------------------------------------
    # DIRECTIVE 7: ROLLING HURST EXPONENT (Mean Reversion vs Trend)
    # ---------------------------------------------------------
    n = len(cl)
    hurst_arr = np.full(n, 0.5)
    hurst_window = 100
    for i in range(hurst_window, n):
        chunk = cl[i - hurst_window:i]
        if np.std(chunk) < 1e-9:
            continue
        # Simplified R/S method
        mean_c = np.mean(chunk)
        deviations = np.cumsum(chunk - mean_c)
        r_val = np.max(deviations) - np.min(deviations)
        s_val = np.std(chunk, ddof=1)
        if s_val > 1e-9 and r_val > 1e-9:
            # H = log(R/S) / log(n)
            hurst_arr[i] = np.log(r_val / s_val) / np.log(hurst_window)
    f['rolling_hurst'] = np.clip(hurst_arr, 0.0, 1.0)

    # ---------------------------------------------------------
    # DIRECTIVE 8: INTER-TIMEFRAME DIVERGENCE SCORING
    # ---------------------------------------------------------
    # HTF bullish + LTF bearish = bullish divergence setup, and vice versa
    ltf_momentum = np.where(pExec.rsi14 > 50, 1.0, -1.0)
    mtf_rsi_s = pd.Series(pSwing.rsi14, index=swing_tf.timestamp).shift(1).reindex(exec_tf.timestamp, method='ffill').fillna(50).values
    htf_rsi_s = pd.Series(pMacro.rsi14, index=macro_tf.timestamp).shift(1).reindex(exec_tf.timestamp, method='ffill').fillna(50).values
    mtf_momentum = np.where(mtf_rsi_s > 50, 1.0, -1.0)
    htf_momentum = np.where(htf_rsi_s > 50, 1.0, -1.0)

    # Cross-TF divergence: when LTF diverges from HTF, potential reversal
    f['ltf_htf_momentum_div'] = ltf_momentum - htf_momentum  # +2 = LTF bull/HTF bear, -2 = opposite
    f['ltf_mtf_momentum_div'] = ltf_momentum - mtf_momentum

    # ---------------------------------------------------------
    # DIRECTIVE 9: ADAPTIVE VOLATILITY FEATURES
    # ---------------------------------------------------------
    # Volatility of volatility (vol-of-vol) — regime change detector
    atr_pct = pExec.atr14 / (np.abs(cl) + 1e-12) * 100
    f['vol_of_vol_20'] = _rolling_std(atr_pct, 20)

    # ATR expansion/contraction rate
    f['atr_expansion_rate'] = _roc(atr_pct, 10)

    # Normalized ATR percentile change (acceleration of volatility)
    f['atr_pctl_roc_10'] = _roc(pExec.atr_percentile_100 * 100.0, 10)

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

    @property
    def ALL_SPECIALIST_KEYS(self):
        """Derive specialist keys dynamically from meta-brain metadata or loaded brains.

        Priority: meta-brain saved keys (exact training alignment) > loaded brain keys.
        """
        if not hasattr(self, '_specialist_keys_cache'):
            self._specialist_keys_cache = None
        if self._specialist_keys_cache is not None:
            return self._specialist_keys_cache

        # 1. Prefer specialist_keys saved by the meta-brain trainer
        if self.meta_brain and "specialist_keys" in self.meta_brain:
            self._specialist_keys_cache = self.meta_brain["specialist_keys"]
            return self._specialist_keys_cache

        # 2. Fallback: derive from actually loaded specialist brains
        if self.brains:
            self._specialist_keys_cache = sorted(self.brains.keys())
            return self._specialist_keys_cache

        # 3. Empty — no brains loaded at all
        return []

    def _invalidate_specialist_keys_cache(self):
        self._specialist_keys_cache = None

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
        self.meta_brain = None  # Unified Meta-Brain
        self._specialist_keys_cache = None
        if not joblib: return
        self.load_brains(brains_dir)

    def load_brains(self, brains_dir):
        if not brains_dir: return
        p = Path(brains_dir)
        if not p.is_dir(): return

        loaded = 0
        for model_file in p.glob("brain_*.joblib"):
            try:
                parts = model_file.stem.split("_")
                # Skip the meta-brain file and calibrator files during specialist loading
                if len(parts) >= 3 and parts[1] == "META":
                    continue
                if "calibrator" in model_file.stem:
                    continue
                # brain_{STRATEGY}_{SIDE}.joblib
                if len(parts) >= 3:
                    strat = parts[1]
                    side = parts[2]
                    brain_data = joblib.load(str(model_file))

                    # Load all brains regardless of valid_edge — quality tier
                    # adjusts threshold at inference time instead of blocking
                    quality_tier = brain_data.get("quality_tier", "strong")
                    threshold_adj = brain_data.get("threshold_adj", 0.0)
                    wfa_roc = brain_data.get("wfa_roc_auc", 0)
                    log.info(f"Loading {model_file.stem}: tier={quality_tier}, ROC={wfa_roc:.3f}, threshold_adj=+{threshold_adj:.2f}")

                    # Load XGBoost model natively if saved in new format
                    if "xgb_model_file" in brain_data:
                        import xgboost as xgb
                        booster = xgb.Booster()
                        xgb_path = p / brain_data["xgb_model_file"]
                        # Fallback: .xgb -> .json for renamed format
                        if not xgb_path.exists():
                            alt = xgb_path.with_suffix(".json") if xgb_path.suffix == ".xgb" else xgb_path.with_suffix(".xgb")
                            if alt.exists():
                                xgb_path = alt
                        booster.load_model(str(xgb_path))
                        brain_data["booster"] = booster

                        # Load ensemble models if available (seed ensembling)
                        ensemble_boosters = []
                        for ens_file in brain_data.get("ensemble_files", []) or []:
                            ens_path = p / ens_file
                            if ens_path.exists():
                                try:
                                    ens_booster = xgb.Booster()
                                    ens_booster.load_model(str(ens_path))
                                    ensemble_boosters.append(ens_booster)
                                except Exception as e:
                                    log.warning(f"Failed to load ensemble model {ens_file}: {e}")
                        if ensemble_boosters:
                            brain_data["ensemble_boosters"] = ensemble_boosters
                            log.info(f"  Loaded {len(ensemble_boosters)+1} ensemble models for {strat}_{side}")

                    self.brains[(strat, side)] = brain_data
                    loaded += 1
            except Exception as e:
                log.error(f"Load error {model_file.name}: {e}")
        if loaded > 0:
            log.info(f"Loaded {loaded} localized brain models from {brains_dir}")

        # ── Load Meta-Brain (unified decision maker) ──
        meta_file = p / "brain_META_unified.joblib"
        if meta_file.exists():
            try:
                meta_data = joblib.load(str(meta_file))
                if "xgb_model_file" in meta_data:
                    import xgboost as xgb
                    booster = xgb.Booster()
                    xgb_path = p / meta_data["xgb_model_file"]
                    if not xgb_path.exists():
                        alt = xgb_path.with_suffix(".json") if xgb_path.suffix == ".xgb" else xgb_path.with_suffix(".xgb")
                        if alt.exists(): xgb_path = alt
                    booster.load_model(str(xgb_path))
                    meta_data["booster"] = booster
                self.meta_brain = meta_data
                self._invalidate_specialist_keys_cache()
                log.info(f"Loaded Meta-Brain (unified) with {len(meta_data.get('feature_names', []))} features")
            except Exception as e:
                log.error(f"Meta-Brain load error: {e}")

        # ── Post-load deployment readiness check ──
        self.validate_deployment_readiness()

    def validate_deployment_readiness(self):
        """Check brain quality distribution and warn/block if too many are noise-tier."""
        if not self.brains:
            return

        total = len(self.brains)
        below_050 = sum(
            1 for b in self.brains.values()
            if b.get("wfa_roc_auc", 0) < 0.50
        )
        all_noise = all(
            b.get("wfa_roc_auc", 0) <= 0.50 for b in self.brains.values()
        )

        if below_050 > total * 0.50:
            log.warning(
                f"DEPLOYMENT READINESS: {below_050}/{total} brains have ROC < 0.50 "
                f"(noise-tier). Recommend retraining with more data or relaxed "
                f"feature selection before live deployment."
            )

        if all_noise and getattr(self.cfg, "strict_brain_validation", False):
            raise RuntimeError(
                f"strict_brain_validation is enabled and ALL {total} brains are "
                f"noise-tier (ROC <= 0.50). Refusing to deploy. Retrain brains "
                f"or set strict_brain_validation=False to override."
            )

    def get_threshold_adj(self, strategy, side):
        """Return the threshold adjustment for a brain based on its quality tier."""
        b = self.brains.get((strategy, side))
        if not b:
            return 0.0
        return b.get("threshold_adj", 0.0)

    def predict_probability(self, strategy, side, base, adv, iExec, px, pExec):
        b = self.brains.get((strategy, side))
        if not b: return None # Strict fallback block
        try:
            vec = self._build_vec(side, base, adv, iExec, px, pExec, b['feature_names'])
            vec = np.clip(np.nan_to_num(vec, nan=0.0), -1e10, 1e10).astype(np.float32)

            # One-time feature match diagnostic
            if not getattr(self, '_feature_diag_done', False):
                self._feature_diag_done = True
                d = {**base}
                for k, v in adv.items():
                    if hasattr(v, '__getitem__') and len(v) > iExec:
                        d[k] = float(v[iExec])
                d["side"] = 1.0
                d["pos_vs_swing_h"] = 0.0
                d["pos_vs_swing_l"] = 0.0
                d["dist_to_mtf_ema200_pct"] = 0.0
                fnames = b['feature_names']
                found = sum(1 for n in fnames if n in d and d[n] != 0.0)
                missing = [n for n in fnames if n not in d]
                zeros = [n for n in fnames if n in d and d[n] == 0.0]
                log.info(f"FEATURE DIAGNOSTIC ({strategy}_{side}): {found}/{len(fnames)} features have non-zero values")
                if missing:
                    log.info(f"  MISSING features (defaulting to 0.0): {missing[:20]}")
                if zeros:
                    log.info(f"  ZERO-valued features ({len(zeros)}): {zeros[:20]}")
                non_zero_vals = [(n, float(vec[0][i])) for i, n in enumerate(fnames) if abs(vec[0][i]) > 1e-9]
                log.info(f"  Non-zero feature values (first 10): {non_zero_vals[:10]}")

            if "booster" in b:
                import xgboost as xgb
                dmat = xgb.DMatrix(vec, feature_names=b['feature_names'])
                # Always get raw log-odds and apply sigmoid manually.
                # Booster.predict() behavior varies across XGBoost versions —
                # output_margin=True guarantees raw log-odds in ALL versions.
                raw = float(b['booster'].predict(dmat, output_margin=True)[0])

                # Seed ensembling: average raw log-odds across all ensemble models,
                # then apply sigmoid. This reduces training variance significantly.
                ensemble_boosters = b.get("ensemble_boosters", [])
                if ensemble_boosters:
                    raws = [raw]
                    for eb in ensemble_boosters:
                        raws.append(float(eb.predict(dmat, output_margin=True)[0]))
                    raw = sum(raws) / len(raws)

                prob = 1.0 / (1.0 + math.exp(-raw))

                # ── Apply isotonic calibration if available ──
                calibrator = b.get("calibrator")
                if calibrator is not None:
                    try:
                        prob = float(calibrator.predict([prob])[0])
                    except Exception as e:
                        log.warning(f"Calibrator failed for {strategy}_{side}: {e}. Using raw prob.")

                # One-time raw margin diagnostic
                if not getattr(self, '_raw_diag_done', False):
                    self._raw_diag_done = True
                    ens_note = f" [+{len(ensemble_boosters)} ensemble models]" if ensemble_boosters else ""
                    log.info(f"BOOSTER DIAGNOSTIC: ensemble_avg_raw={raw:.6f}, sigmoid={prob:.6f}{ens_note}")
            else:
                prob = b['classifier'].predict_proba(vec)[0][1]
            return prob
        except Exception as e:
            log.error(f"Brain Prediction Failed: {e}\n{traceback.format_exc()}")
            return None

    def predict_meta_probability(self, strategy, side, base, adv, iExec, px, pExec):
        """Query ALL specialist brains + market context → Meta-Brain final decision.

        Returns (meta_prob, specialist_prob) tuple.
        meta_prob is the unified decision; specialist_prob is the setup-specific one.
        If no meta-brain is loaded, falls back to specialist-only.
        """
        # Step 1: Get the specialist probability for the selected setup
        specialist_prob = self.predict_probability(strategy, side, base, adv, iExec, px, pExec)
        if specialist_prob is None:
            return None, None

        # If no meta-brain loaded, fall back to specialist-only
        if self.meta_brain is None:
            return specialist_prob, specialist_prob

        try:
            # Step 2: Query ALL specialist brains to build the full probability landscape
            specialist_probs = {}
            for (strat_key, side_key) in self.ALL_SPECIALIST_KEYS:
                prob = self.predict_probability(strat_key, side_key, base, adv, iExec, px, pExec)
                # Use NaN for missing (matches meta-brain training distribution)
                specialist_probs[f"brain_{strat_key}_{side_key}"] = prob if prob is not None else float('nan')

            # Step 3: Build meta-brain feature vector
            meta_features = {}

            # 3a: All specialist probabilities
            meta_features.update(specialist_probs)

            # 3b: Count how many specialists are active (have loaded brains)
            active_probs = [v for v in specialist_probs.values() if not (isinstance(v, float) and math.isnan(v))]
            meta_features["n_active_specialists"] = float(len(active_probs))
            meta_features["mean_specialist_prob"] = float(np.mean(active_probs)) if active_probs else 0.0
            meta_features["max_specialist_prob"] = float(np.max(active_probs)) if active_probs else 0.0
            meta_features["std_specialist_prob"] = float(np.std(active_probs)) if len(active_probs) > 1 else 0.0

            # 3c: The selected setup's identity
            meta_features["selected_specialist_prob"] = specialist_prob
            meta_features["selected_side"] = 1.0 if side == "long" else 0.0

            # 3d: Market context from confluence_features (base dict)
            context_keys = [
                "bull_confluence_score", "bear_confluence_score",
                "adx_14", "rsi_14", "atr_pct",
                "btc_bullish", "funding_rate", "delta_oi",
                "dxy_trend", "spx_trend", "btcd_trend",
                "rvol", "vol_zscore",
                "ob_bull_near", "ob_bear_near",
                "fvg_bull_near", "fvg_bear_near",
                "bos_bull", "bos_bear",
                "choch_bull", "choch_bear",
                "displacement_bull_count", "displacement_bear_count",
                "spring", "upthrust",
                "accum_phase", "distrib_phase",
                "qm_bull", "qm_bear",
                "in_ote_zone_bull", "in_ote_zone_short",
                "in_discount_zone", "in_premium_zone",
                "equal_highs_count", "equal_lows_count",
                "hurst_exponent",
                "macro_sentiment",
                # Market Structure Reading
                "struct_trend", "struct_strength", "struct_break",
                "struct_bias_score",
                "htf_struct_trend", "htf_struct_strength", "htf_struct_break",
                "macro_struct_trend", "macro_struct_strength",
                "struct_align_bull", "struct_align_bear",
                "recent_hh_count", "recent_hl_count",
                "recent_lh_count", "recent_ll_count",
            ]
            for k in context_keys:
                val = base.get(k, 0.0)
                meta_features[k] = float(val) if isinstance(val, (int, float, np.floating, np.integer)) and np.isfinite(val) else 0.0

            # Step 4: Build vector and predict
            meta_fnames = self.meta_brain['feature_names']
            # Preserve NaN for specialist columns; XGBoost will route via missing path.
            # Other features use 0.0 default if absent.
            specialist_set = {f"brain_{s}_{sd}" for s, sd in self.ALL_SPECIALIST_KEYS}
            vec_vals = []
            for n in meta_fnames:
                val = meta_features.get(n, np.nan if n in specialist_set else 0.0)
                vec_vals.append(val)
            vec = np.array(vec_vals).reshape(1, -1).astype(np.float32)
            # Clip extreme numeric values but PRESERVE NaN
            vec = np.where(np.isnan(vec), vec, np.clip(vec, -1e10, 1e10))

            if "booster" in self.meta_brain:
                import xgboost as xgb
                dmat = xgb.DMatrix(vec, feature_names=meta_fnames, missing=np.nan)
                raw = float(self.meta_brain['booster'].predict(dmat, output_margin=True)[0])
                meta_prob = 1.0 / (1.0 + math.exp(-raw))
            else:
                raw = None
                meta_prob = self.meta_brain['classifier'].predict_proba(vec)[0][1]

            # One-time meta-brain diagnostic (guarded by _meta_diag_done flag)
            if not getattr(self, '_meta_diag_done', False):
                self._meta_diag_done = True
                n_loaded = sum(1 for k in self.ALL_SPECIALIST_KEYS if k in self.brains)
                n_expected = len(self.ALL_SPECIALIST_KEYS)
                log.info(f"META-BRAIN DIAGNOSTIC: {n_loaded} specialist brains loaded vs "
                         f"{n_expected} expected slots")
                log.info(f"  Meta feature vector length: {len(meta_fnames)}")
                if raw is not None:
                    log.info(f"  Meta raw log-odds: {raw:.6f}, sigmoid output: {meta_prob:.6f}")
                else:
                    log.info(f"  Meta probability output: {meta_prob:.6f}")

            return meta_prob, specialist_prob

        except Exception as e:
            log.error(f"Meta-Brain prediction failed: {e}\n{traceback.format_exc()}")
            # Fallback to specialist-only
            return specialist_prob, specialist_prob

    def get_all_specialist_probs(self, side, base, adv, iExec, px, pExec):
        """Get probabilities from ALL specialist brains. Used during export for meta-brain training."""
        probs = {}
        for (strat_key, side_key) in self.ALL_SPECIALIST_KEYS:
            prob = self.predict_probability(strat_key, side_key, base, adv, iExec, px, pExec)
            probs[f"brain_{strat_key}_{side_key}"] = prob if prob is not None else -1.0
        return probs

    # Features that leaked post-hoc trade info into training data.
    # Old models still list them — force a neutral value so they don't
    # push every prediction into the negative branch (bars_open was always
    # >0 in training but 0/missing at inference → OOD → near-zero probs).
    _LEAKED_FEATURE_DEFAULTS = {"bars_open": -1.0}

    def _build_vec(self, side, b, a, iExec, px, pExec, fnames):
        d = {**b}
        # FIX: Match _build_full_features semantics — do NOT blindly overwrite
        # base (confluence_features) values with 0.0 when an adv_features entry
        # is not a valid array or is shorter than iExec.  The training export
        # path uses `else: continue` which preserves base values; the old
        # `else: d[k] = 0.0` destroyed base features like bars_since_ob_bull,
        # macro_struct_strength, and any other key that happened to collide or
        # that the brain expected from the base dict.
        for k, v in a.items():
            if hasattr(v, '__getitem__') and not isinstance(v, (str, bytes)):
                try:
                    vlen = len(v)
                except TypeError:
                    # numpy scalar — treat as direct value
                    d[k] = float(v) if np.isfinite(float(v)) else 0.0
                    continue
                if vlen > iExec:
                    val = v[iExec]
                    d[k] = float(val) if np.isfinite(val) else 0.0
                    if d[k] > 1e10: d[k] = 1e10
                    elif d[k] < -1e10: d[k] = -1e10
                else:
                    # Array too short — only set if key is NOT already populated
                    # from the base dict (preserve confluence_features values).
                    if k not in b:
                        d[k] = 0.0
            elif isinstance(v, (int, float, np.integer, np.floating)):
                d[k] = float(v) if np.isfinite(float(v)) else 0.0
            else:
                # Non-numeric / non-array — skip, preserving base value
                if k not in b:
                    d[k] = 0.0

        # ALIGNMENT FIX
        d.update({
            "pos_vs_swing_h": (px-pExec.last_sh[iExec])/px if px else 0.0,
            "pos_vs_swing_l": (px-pExec.last_sl[iExec])/px if px else 0.0,
            "dist_to_mtf_ema200_pct": (px-a['mtf_ema200_arr'][iExec])/px*100 if px and 'mtf_ema200_arr' in a else 0.0,
            "side": 1.0 if side=="long" else 0.0
        })

        # SESSION FLAG PARITY FIX: Use the same timezone-correct helper as
        # the training export (_build_full_features) to avoid training/inference drift.
        _ts = b.get("timestamp")
        if _ts is not None:
            try:
                from vajra_macro_data import compute_session_flags
                d.update(compute_session_flags(int(_ts)))
            except Exception:
                pass  # keep confluence_features values

        # Neutralize leaked features in old models (use XGBoost missing-value
        # sentinel so trees route via the default direction, not the 0-branch).
        for lf, default_val in self._LEAKED_FEATURE_DEFAULTS.items():
            if lf in fnames and lf not in d:
                d[lf] = default_val

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
        self.strategy_trade_counts: Dict[str, int] = {}
        self.strategy_last_exit_ts: Dict[str, int] = {}
        self.funding_lookup: Dict[int, float] = {}  # {timestamp_ms: funding_rate}
        self.exec_tf_minutes: int = self._exec_tf_minutes_from_cfg(cfg)

    def _exec_tf_minutes_from_cfg(self, cfg) -> int:
        """Parse cfg.exec_tf string ('15m', '1h', etc.) -> minutes."""
        tf = getattr(cfg, 'exec_tf', '15m').strip().lower()
        if tf.endswith('m'):
            return int(tf[:-1])
        if tf.endswith('h'):
            return int(tf[:-1]) * 60
        if tf.endswith('d'):
            return int(tf[:-1]) * 1440
        return 15  # safe default

    def set_funding_lookup(self, ts_array, funding_array):
        """Populate the funding lookup table from aligned arrays (called by backtester)."""
        self.funding_lookup = dict(zip(
            np.asarray(ts_array, dtype=np.int64).tolist(),
            np.asarray(funding_array, dtype=float).tolist(),
        ))

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

        # ── Per-strategy trade cap & cooldown gate ────────────────────────
        strategy_base = plan.get('strategy', '').split('_')[0]
        if strategy_base:
            max_per_strat = getattr(self.cfg, 'max_trades_per_strategy', 0)
            if max_per_strat > 0:
                strat_count = sum(
                    1 for t in self.open_trades
                    if t.get('strategy', '').split('_')[0] == strategy_base
                ) + sum(
                    1 for p in self.pending_orders
                    if p.get('strategy', '').split('_')[0] == strategy_base
                )
                if strat_count >= max_per_strat:
                    _p6_counters["strategy_cap_rejected"] += 1
                    return None

            cooldown_bars = getattr(self.cfg, 'strategy_cooldown_bars', 0)
            if cooldown_bars > 0:
                exec_tf_ms = _tf_to_ms(getattr(self.cfg, 'exec_tf', '15m'))
                cooldown_ms = cooldown_bars * exec_tf_ms
                current_ts = bar.get('timestamp', 0) if isinstance(bar, dict) else 0
                last_exit = self.strategy_last_exit_ts.get(strategy_base, 0)
                if last_exit > 0 and (current_ts - last_exit) < cooldown_ms:
                    _p6_counters["cooldown_rejected"] += 1
                    return None

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

        # ── Correlation-adjusted risk factor ──────────────────────────────
        # If you already have N open positions correlated with this new one, the
        # effective leverage is sum(corr_i). To keep total portfolio risk constant,
        # scale this trade's risk by 1/sqrt(1 + sum(corr_i)).
        new_symbol = order.get('symbol', self.cfg.symbol)
        new_side = plan['side']
        correlated_exposure = 0.0
        for t in self.open_trades + self.pending_orders:
            other_sym = t.get('symbol', '')
            other_side = t.get('side', '')
            if not other_sym or not other_side:
                continue
            corr = _get_correlation(new_symbol, other_sym)
            # Same-side correlation adds risk; opposite-side correlation hedges it
            if other_side == new_side:
                correlated_exposure += corr
            else:
                correlated_exposure -= 0.5 * corr   # half-credit hedge

        correlated_exposure = max(0.0, correlated_exposure)
        correlation_scalar = 1.0 / math.sqrt(1.0 + correlated_exposure)

        # Apply scalar to the order's risk_factor BEFORE order creation
        original_rf = order.get('risk_factor', 1.0)
        order['risk_factor'] = original_rf * correlation_scalar

        if correlation_scalar < 0.95:
            log.info(
                f"[{new_symbol}] Correlation-adjusted risk: {original_rf:.2f} -> "
                f"{order['risk_factor']:.2f} (scalar={correlation_scalar:.2f}, "
                f"correlated_exposure={correlated_exposure:.2f})"
            )

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

    # ------------------------------------------------------------------
    # FRICTION PHYSICS (Directive 1 — Authentic PnL)
    # ------------------------------------------------------------------
    # Round-trip monetary cost decomposition (per unit of contract size):
    #
    #     entry_fee_cost  = entry_fill_px * entry_fee_rate
    #     entry_slip_cost = entry_fill_px * slip_rate
    #     exit_fee_cost   = exit_px       * taker_fee_rate   (always taker on exit)
    #     exit_slip_cost  = exit_px       * slip_rate
    #     total_friction  = entry_fee_cost + entry_slip_cost
    #                     + exit_fee_cost  + exit_slip_cost
    #
    # ``total_friction`` is an ALWAYS-POSITIVE dollar drag and is subtracted
    # from the directional raw_delta BEFORE we divide by the per-unit risk
    # distance. The resulting ``pnl_r`` is therefore the TRUE NET yield in
    # R-multiples — identical to how an institutional post-trade TCA stack
    # marks a fill after the clearing house takes its cut.
    #
    # The ENTRY leg is maker-priced only when the live router actually used
    # an unmarketable limit order; every other execution style crosses the
    # book and pays taker. The EXIT leg is ALWAYS taker because SL, TP, and
    # TIF market-out all hit the book.
    # ------------------------------------------------------------------
    def _fee_slip_rates(self, order_type: str):
        """Resolve the per-leg fee and slippage rates from cfg (deterministic)."""
        taker_rate = float(getattr(self.cfg, 'taker_fee_bps', 0.0)) / 10000.0
        maker_rate = float(getattr(self.cfg, 'maker_fee_bps', 0.0)) / 10000.0
        slip_rate  = float(getattr(self.cfg, 'slippage_bps',  0.0)) / 10000.0
        entry_fee_rate = maker_rate if order_type == 'limit' else taker_rate
        exit_fee_rate  = taker_rate
        return entry_fee_rate, exit_fee_rate, slip_rate

    def _friction_breakdown(self, entry_fill_px: float, exit_px: float,
                            order_type: str) -> Dict[str, float]:
        """Explicit per-unit monetary friction decomposition for audit + PnL."""
        entry_fee_rate, exit_fee_rate, slip_rate = self._fee_slip_rates(order_type)

        ep = float(entry_fill_px)
        xp = float(exit_px)

        entry_fee_cost  = ep * entry_fee_rate
        entry_slip_cost = ep * slip_rate
        exit_fee_cost   = xp * exit_fee_rate
        exit_slip_cost  = xp * slip_rate

        entry_friction = entry_fee_cost + entry_slip_cost
        exit_friction  = exit_fee_cost  + exit_slip_cost
        total_friction = entry_friction + exit_friction

        return {
            "entry_fee_cost":  entry_fee_cost,
            "entry_slip_cost": entry_slip_cost,
            "exit_fee_cost":   exit_fee_cost,
            "exit_slip_cost":  exit_slip_cost,
            "entry_friction":  entry_friction,
            "exit_friction":   exit_friction,
            "total_friction_cost": total_friction,
        }

    def _compute_pnl_r(self, side: str, entry_fill_px: float, exit_px: float,
                       raw_risk: float, order_type: str,
                       bars_held: int = 0, avg_funding_rate: float = 0.0,
                       exec_tf_minutes: int = 15) -> Tuple[float, float, Dict[str, float]]:
        """Return ``(net_pnl_r, gross_pnl_r, friction_breakdown)``.

        The monetary friction is deducted from the raw directional delta
        BEFORE the R-multiple is computed so ``net_pnl_r`` is the authentic
        yield after trading friction.

        Now also deducts funding cost.  Bybit funding settles every 8h
        (00:00, 08:00, 16:00 UTC).  Per-cycle cost is
        funding_rate * notional.  Longs pay positive funding, shorts pay
        negative funding.

        Args:
            bars_held: number of execution-tf bars the position was held
            avg_funding_rate: average funding rate (decimal, e.g. 0.0001 = 0.01%)
            exec_tf_minutes: minutes per execution bar (default 15)
        """
        raw_delta = (exit_px - entry_fill_px) if side == 'long' else (entry_fill_px - exit_px)
        fb = self._friction_breakdown(entry_fill_px, exit_px, order_type)
        total_friction = fb["total_friction_cost"]

        # ── Funding accrual ──
        # Number of 8h funding cycles crossed (rounded up — partial cycles still pay)
        minutes_held = bars_held * exec_tf_minutes
        n_funding_cycles = max(1, int(np.ceil(minutes_held / (8 * 60)))) if minutes_held > 0 else 0
        # Long pays positive funding, short pays negative funding
        sign = 1.0 if side == "long" else -1.0
        funding_cost = avg_funding_rate * entry_fill_px * n_funding_cycles * sign
        fb["funding_cost"] = funding_cost
        fb["n_funding_cycles"] = n_funding_cycles
        fb["avg_funding_rate"] = avg_funding_rate

        net_delta = raw_delta - total_friction - funding_cost

        denom = raw_risk if raw_risk > 1e-12 else 1e-12
        gross_pnl_r = float(raw_delta / denom)
        net_pnl_r   = float(net_delta / denom)
        fb["funding_drag_r"] = float(funding_cost / denom)
        return net_pnl_r, gross_pnl_r, fb

    def _compute_net_pnl_r(self, side: str, entry_fill_px: float, exit_px: float,
                           raw_risk: float, order_type: str,
                           bars_held: int = 0, avg_funding_rate: float = 0.0,
                           exec_tf_minutes: int = 15) -> float:
        """Back-compat thin wrapper that only returns the net R-multiple."""
        net_pnl_r, _gross, _fb = self._compute_pnl_r(
            side, entry_fill_px, exit_px, raw_risk, order_type,
            bars_held, avg_funding_rate, exec_tf_minutes
        )
        return net_pnl_r

    def step_bar(self, symbol, o, h, l, c, ts=None, swing_high=0.0, swing_low=0.0):
        if ts is not None and ts == self.last_processed_ts.get(symbol): return []
        self.last_processed_ts[symbol] = ts
        self.current_bar_index[symbol] = self.current_bar_index.get(symbol, 0) + 1

        closed = []
        still_open = []
        still_pending = []

        # Determine the current bar's timestamp, fallback to 0 if not provided
        current_ts = ts if ts is not None else 0

        # Strict Time-In-Force horizon (bars). Directive 4: a 15m structural
        # trigger must resolve inside 12h (48 bars) or be force-exited.
        tif_decay_global = int(getattr(self.cfg, 'time_in_force_decay', 0) or 0)

        for order in self.pending_orders:
            if order.get('symbol', self.cfg.symbol) != symbol:
                still_pending.append(order)
                continue

            triggered = False
            fill_px = 0.0

            order['ttl'] -= 1
            if order['ttl'] < 0:
                _p6_counters["expired_unfilled"] += 1
                continue

            side = order['side']
            entry_target = order['entry']
            # PATCH: DYNAMIC EXECUTION STYLE INHERITANCE
            order_type = order.get('type', getattr(self.cfg, 'execution_style', 'limit'))

            if order_type == 'limit':
                # Realistic fill: require the bar to CROSS the limit by a buffer,
                # not just touch it. This proxies queue-position and wick-depth
                # effects. The buffer is configurable via cfg.limit_fill_buffer_atr
                # (default 0.10 ATR).
                fill_buffer_pct = getattr(self.cfg, 'limit_fill_buffer_pct', 0.001)  # 0.1% of price
                fill_buffer_atr_mult = getattr(self.cfg, 'limit_fill_buffer_atr', 0.10)
                # Use ATR if the order carried it; else fall back to pct of entry
                order_atr = order.get('atr_at_signal', 0.0)
                if order_atr > 0:
                    fill_buffer = order_atr * fill_buffer_atr_mult
                else:
                    fill_buffer = entry_target * fill_buffer_pct

                if side == 'long':
                    if o <= order.get('sl', -1.0):
                        order['ttl'] = -1
                        _p6_counters["stopped_pre_entry"] += 1
                        continue
                    # Long limit: bar low must penetrate BELOW (target - buffer)
                    if l <= (entry_target - fill_buffer):
                        triggered = True
                        fill_px = min(o, entry_target)
                elif side == 'short':
                    if o >= order.get('sl', 99999999.0):
                        order['ttl'] = -1
                        _p6_counters["stopped_pre_entry"] += 1
                        continue
                    # Short limit: bar high must penetrate ABOVE (target + buffer)
                    if h >= (entry_target + fill_buffer):
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
                _p6_counters["filled"] += 1
                strat_base = order.get('strategy', 'UNKNOWN').split('_')[0]
                _p6_counters["per_strategy_filled"][strat_base] = \
                    _p6_counters["per_strategy_filled"].get(strat_base, 0) + 1

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
                # Lock in the order type used for the fill so exit math knows
                # whether the entry leg was maker or taker.
                trade['type'] = order_type

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
            order_type = t.get('type', getattr(self.cfg, 'execution_style', 'limit'))

            can_tp = t.get('can_tp_this_bar', True)
            if not can_tp:
                t['can_tp_this_bar'] = True

            raw_risk = max(t.get('initial_risk_unit', abs(t['entry'] - t['sl'])), 1e-9)

            # Live unrealized PnL tracked as NET of friction (fair apples-to-apples).
            t['pnl_r'] = self._compute_net_pnl_r(side, entry, c, raw_risk, order_type)

            intra_best_px = h if side == 'long' else l
            intra_max_pnl_r = self._compute_net_pnl_r(side, entry, intra_best_px, raw_risk, order_type)

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

            # Directive 4B: forced TIF exit at current bar close if the window
            # has been fully consumed without an SL/TP resolution.
            force_tif = (
                not hit_sl
                and not hit_tp
                and tif_decay_global > 0
                and t['bars_open'] >= tif_decay_global
            )

            if hit_sl or hit_tp or force_tif:
                if hit_sl:
                    if side == 'long': exit_px = min(o, t['sl']) if o < t['sl'] else t['sl']
                    else:              exit_px = max(o, t['sl']) if o > t['sl'] else t['sl']
                elif hit_tp:
                    if side == 'long': exit_px = max(o, tp) if o > tp else tp
                    else:              exit_px = min(o, tp) if o < tp else tp
                else:
                    # Time-based market exit at the bar close.
                    exit_px = c
                    exit_reason = 'tif'

                # Directive 1: deduct full round-trip friction from pnl_r.
                # Stamp the full audit trail so the backtester, exporter,
                # and meta-brain labeller all see identical numbers.

                # ── Funding context for accrual ──
                fill_ts = t.get('fill_ts', current_ts)
                bars_held = t.get('bars_open', 0)
                # Average funding over the hold window.  If lookup is empty
                # (live mode w/o pre-loaded funding), default to 0 — the live
                # executor handles funding via the exchange's actual debits.
                avg_funding = 0.0
                if self.funding_lookup:
                    fill_ts_int = int(fill_ts) if fill_ts else current_ts
                    exit_ts_int = int(current_ts)
                    fr_samples = [v for ts_k, v in self.funding_lookup.items()
                                  if fill_ts_int <= ts_k <= exit_ts_int]
                    if fr_samples:
                        avg_funding = float(np.mean(fr_samples))

                net_pnl_r, gross_pnl_r, fb = self._compute_pnl_r(
                    side, entry, exit_px, raw_risk, order_type,
                    bars_held=bars_held,
                    avg_funding_rate=avg_funding,
                    exec_tf_minutes=self.exec_tf_minutes,
                )
                t['pnl_r']                = net_pnl_r
                t['gross_pnl_r']          = gross_pnl_r
                t['entry_friction']       = fb["entry_friction"]
                t['exit_friction']        = fb["exit_friction"]
                t['total_friction_cost']  = fb["total_friction_cost"]
                t['friction_drag_r']      = gross_pnl_r - net_pnl_r
                t['funding_drag_r']       = fb.get("funding_drag_r", 0.0)
                t['n_funding_cycles']     = fb.get("n_funding_cycles", 0)
                t['avg_funding_rate']     = fb.get("avg_funding_rate", 0.0)
                t['exit_price']           = exit_px
                t['exit_reason']          = exit_reason
                t['exit_ts']              = current_ts

                closed.append(t)
                self.mem.record_exit(t)
                # Track last exit timestamp per strategy base for cooldown gate
                exit_strat_base = t.get('strategy', '').split('_')[0]
                if exit_strat_base:
                    self.strategy_last_exit_ts[exit_strat_base] = current_ts
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

    # Price-action structural alignment (HH/HL/LH/LL based, not EMA lag)
    struct_align_bull = base.get("struct_align_bull", 0)
    struct_align_bear = base.get("struct_align_bear", 0)

    # REGIME GATE: Block counter-trend trades when strong unidirectional trend
    can_long = True
    can_short = True
    is_reversal_context = False

    # Relaxed gate: block only when ALL 3 TFs agree AND structure confirms
    # EMAs say bearish AND price structure is bearish = high conviction block
    if bearish_tf_count == 3 and struct_align_bear >= 2:
        can_long = False
    elif bearish_tf_count == 3:
        # EMAs say bearish but structure disagrees — allow with weak flag
        can_long = True
    if bullish_tf_count == 3 and struct_align_bull >= 2:
        can_short = False
    elif bullish_tf_count == 3:
        can_short = True

    # Partial regime bias: when 2/3 TFs agree, allow but flag as weaker conviction
    is_weak_long = (bearish_tf_count >= 2 and bullish_tf_count == 0 and struct_align_bear >= 1)
    is_weak_short = (bullish_tf_count >= 2 and bearish_tf_count == 0 and struct_align_bull >= 1)

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

    # Unlock counter-trend when reversal evidence converges (configurable threshold)
    reversal_min = getattr(cfg, 'reversal_evidence_min', 2)
    if not can_long and bull_reversal_evidence >= reversal_min:
        can_long = True
        is_reversal_context = True
    if not can_short and bear_reversal_evidence >= reversal_min:
        can_short = True
        is_reversal_context = True
    # Also unlock on strong single-signal reversals (Wyckoff spring/upthrust are high-conviction)
    if not can_long and has_spring and bull_reversal_evidence >= 1:
        can_long = True
        is_reversal_context = True
    if not can_short and has_upthrust and bear_reversal_evidence >= 1:
        can_short = True
        is_reversal_context = True

    # ==========================================================
    # PHASE 3: ORACLE MACRO GATE
    # ==========================================================
    oracle_sentiment = base.get("macro_sentiment", 0.0)
    btc_bullish = base.get("btc_bullish", 1.0)
    bid_ask_imbalance = base.get("bid_ask_imbalance", 0.5)

    # Oracle hard veto: only block on extreme sentiment (widened from ±0.6 to ±0.8)
    if oracle_sentiment < -0.8 and not is_reversal_context:
        can_long = False
    if oracle_sentiment > 0.8 and not is_reversal_context:
        can_short = False

    # BTC regime: only block longs when BTC is deeply bearish AND all TFs confirm
    if btc_bullish < 0.3 and not is_reversal_context:
        if bearish_tf_count >= 3:
            can_long = False

    # Spoofing defense (widened from 0.80/0.20 to 0.90/0.10 — less false positives)
    if bid_ask_imbalance > 0.90: can_long = False
    if bid_ask_imbalance < 0.10: can_short = False

    if not can_long and not can_short: return None

    # ==========================================================
    # PHASE 3B: MARKET QUALITY FILTERS
    # ==========================================================
    adx_val = base.get("adx", 25.0)

    # Choppy market filter — ADX < 15 means price is stuck in a range with no
    # directional edge. Signals in this regime have poor intrinsic win rate, so
    # filtering them at the raw level gives the brain a cleaner training distribution.
    adx_threshold = getattr(cfg, 'adx_chop_threshold', 15.0)
    if getattr(cfg, 'filter_adx_chop', True) and adx_val < adx_threshold and not is_reversal_context:
        return None

    # Multi-TF alignment: configurable minimum (default 1 TF agreement)
    mtf_min = getattr(cfg, 'mtf_alignment_min', 1)
    if not is_reversal_context:
        if can_long and not can_short and bullish_tf_count < mtf_min:
            return None
        if can_short and not can_long and bearish_tf_count < mtf_min:
            return None

    # ==========================================================
    # PHASE 4: PRECISE STRUCTURAL ENTRY DETECTION
    # ==========================================================
    px = pExec.c[iExec]
    low = pExec.l[iExec]
    high = pExec.h[iExec]

    ob_bull_bot = base.get("ob_bull_bot", 0)
    ob_bull_top = base.get("ob_bull_top", 0)
    ob_bear_bot = base.get("ob_bear_bot", 0)
    ob_bear_top = base.get("ob_bear_top", 0)
    ob_bull_ce = (ob_bull_top + ob_bull_bot) / 2 if ob_bull_top > 0 else 0
    ob_bear_ce = (ob_bear_top + ob_bear_bot) / 2 if ob_bear_bot > 0 else 0

    ob_max_age = getattr(cfg, 'ob_freshness_bars', 192)
    ob_bull_fresh = 0 < base.get("bars_since_ob_bull", 999) <= ob_max_age
    ob_bear_fresh = 0 < base.get("bars_since_ob_bear", 999) <= ob_max_age

    fvg_bull = base.get("fvg_bull", 0)
    fvg_bear = base.get("fvg_bear", 0)
    fvg_tol_atr = getattr(cfg, 'fvg_tolerance_atr', 0.5)
    fvg_tol = current_atr * fvg_tol_atr  # FVG zone = 0.5 ATR (was 0.2% of price)
    qm_zone = current_atr * 0.75  # Widen QM zone to 0.75 ATR (was 0.5)

    sweep_bull = base.get("sweep_bull_p", 0.0)
    sweep_bear = base.get("sweep_bear_p", 0.0)

    # Wick rejection detection (5-bar window, configurable threshold)
    wick_threshold = getattr(cfg, 'wick_rejection_pct', 0.15)
    is_bull_rejection = False
    is_bear_rejection = False
    for idx in range(max(0, iExec - 4), iExec + 1):
        if idx >= len(pExec.h): continue
        c_range = pExec.h[idx] - pExec.l[idx]
        if c_range > 1e-9:
            body_top = max(pExec.o[idx], pExec.c[idx])
            body_bot = min(pExec.o[idx], pExec.c[idx])
            if (body_bot - pExec.l[idx]) / c_range > wick_threshold: is_bull_rejection = True
            if (pExec.h[idx] - body_top) / c_range > wick_threshold: is_bear_rejection = True

    # Volume confirmation (configurable lookback and threshold)
    vol_lookback = getattr(cfg, 'vol_spike_lookback', 5)
    has_volume = any(pExec.vol_spike[max(0, iExec-vol_lookback+1):iExec+1] > 0) if hasattr(pExec, 'vol_spike') else False
    rvol = base.get("rvol", 1.0)
    rvol_threshold = getattr(cfg, 'vol_confirm_rvol', 0.8)
    has_elevated_vol = rvol > rvol_threshold

    setup_type = None
    logic_desc = ""
    side = None

    # Volume confirmation gate — require EITHER a recent volume spike OR elevated
    # relative volume.  The AND gate (has_volume and has_elevated_vol) starved
    # signal generation to ~0.01% of bars because both conditions rarely coincide
    # within the same 5-bar window on 15m crypto data.  An OR gate restores
    # volume-flexible signal flow while the confluence floor (>=2.0) and the
    # downstream EV / risk-distance gates still filter low-quality setups.
    has_vol_confirm = has_volume or has_elevated_vol

    # ==========================================================
    # PHASE 4A: STRUCTURAL CONFLUENCE SCORING
    # ==========================================================
    # Count how many independent structural signals are active RIGHT NOW.
    # This score is passed to the brain as a feature AND used to pick the
    # best setup when multiple could fire on the same bar.

    # ---- Bullish confluence ----
    bull_struct_score = 0.0
    bull_struct_reasons = []
    if ob_bull_ce > 0 and ob_bull_fresh and low <= ob_bull_ce:
        bull_struct_score += 1.0; bull_struct_reasons.append("OB")
    if fvg_bull > 0 and low <= (fvg_bull + fvg_tol) and px > (fvg_bull - fvg_tol):
        bull_struct_score += 1.0; bull_struct_reasons.append("FVG")
    if base.get("qm_bull", 0) > 0 and low <= (base.get("qm_bull", 0) + qm_zone):
        bull_struct_score += 1.0; bull_struct_reasons.append("QM")
    if has_spring:
        bull_struct_score += 1.0; bull_struct_reasons.append("SPRING")
    if sweep_bull > 0:
        bull_struct_score += 1.0; bull_struct_reasons.append("SWEEP")
    if base.get("in_ote_zone_long", 0) > 0:
        bull_struct_score += 1.0; bull_struct_reasons.append("OTE")
    if base.get("is_discount_zone", 0) > 0:
        bull_struct_score += 0.5; bull_struct_reasons.append("DISCOUNT")
    if base.get("wyckoff_accum", 0) > 0:
        bull_struct_score += 1.0; bull_struct_reasons.append("ACCUM")
    if base.get("choch_up", 0) > 0:
        bull_struct_score += 1.0; bull_struct_reasons.append("CHOCH")
    if base.get("displacement_bos_bull", 0) > 0:
        bull_struct_score += 1.0; bull_struct_reasons.append("DISP+BOS")
    div_count_bull = int(bull_ewo) + int(bull_obv) + int(bull_cvd) + int(bull_rsi_div)
    if div_count_bull >= 2:
        bull_struct_score += 1.0; bull_struct_reasons.append(f"DIV×{div_count_bull}")
    if is_bull_rejection:
        bull_struct_score += 0.5; bull_struct_reasons.append("WICK")
    # Structure alignment: bullish HH/HL structure on exec TF
    if base.get("struct_trend", 0) > 0:
        bull_struct_score += 1.0; bull_struct_reasons.append("STRUCT_BULL")
    # HTF structure confirmation: higher timeframe also bullish structure
    if base.get("htf_struct_trend", 0) > 0:
        bull_struct_score += 0.5; bull_struct_reasons.append("HTF_STRUCT")
    # Structure break of bearish trend = bullish reversal signal
    if base.get("struct_break", 0) > 0 and base.get("struct_trend", 0) >= 0:
        bull_struct_score += 1.0; bull_struct_reasons.append("STRUCT_BRK")

    # ---- Bearish confluence ----
    bear_struct_score = 0.0
    bear_struct_reasons = []
    if ob_bear_ce > 0 and ob_bear_fresh and high >= ob_bear_ce:
        bear_struct_score += 1.0; bear_struct_reasons.append("OB")
    if fvg_bear > 0 and high >= (fvg_bear - fvg_tol) and px < (fvg_bear + fvg_tol):
        bear_struct_score += 1.0; bear_struct_reasons.append("FVG")
    if base.get("qm_bear", 0) > 0 and high >= (base.get("qm_bear", 0) - qm_zone):
        bear_struct_score += 1.0; bear_struct_reasons.append("QM")
    if has_upthrust:
        bear_struct_score += 1.0; bear_struct_reasons.append("UPTHRUST")
    if sweep_bear > 0:
        bear_struct_score += 1.0; bear_struct_reasons.append("SWEEP")
    if base.get("in_ote_zone_short", 0) > 0:
        bear_struct_score += 1.0; bear_struct_reasons.append("OTE")
    if base.get("is_premium_zone", 0) > 0:
        bear_struct_score += 0.5; bear_struct_reasons.append("PREMIUM")
    if base.get("wyckoff_distrib", 0) > 0:
        bear_struct_score += 1.0; bear_struct_reasons.append("DISTRIB")
    if base.get("choch_down", 0) > 0:
        bear_struct_score += 1.0; bear_struct_reasons.append("CHOCH")
    if base.get("displacement_bos_bear", 0) > 0:
        bear_struct_score += 1.0; bear_struct_reasons.append("DISP+BOS")
    div_count_bear = int(bear_ewo) + int(bear_obv) + int(bear_cvd) + int(bear_rsi_div)
    if div_count_bear >= 2:
        bear_struct_score += 1.0; bear_struct_reasons.append(f"DIV×{div_count_bear}")
    if is_bear_rejection:
        bear_struct_score += 0.5; bear_struct_reasons.append("WICK")
    # Structure alignment: bearish LH/LL structure on exec TF
    if base.get("struct_trend", 0) < 0:
        bear_struct_score += 1.0; bear_struct_reasons.append("STRUCT_BEAR")
    # HTF structure confirmation: higher timeframe also bearish structure
    if base.get("htf_struct_trend", 0) < 0:
        bear_struct_score += 0.5; bear_struct_reasons.append("HTF_STRUCT")
    # Structure break of bullish trend = bearish reversal signal
    if base.get("struct_break", 0) > 0 and base.get("struct_trend", 0) <= 0:
        bear_struct_score += 1.0; bear_struct_reasons.append("STRUCT_BRK")

    # Inject confluence scores into base features so the brain can see them
    base["bull_confluence_score"] = bull_struct_score
    base["bear_confluence_score"] = bear_struct_score
    base["bull_confluence_reasons"] = "+".join(bull_struct_reasons) if bull_struct_reasons else ""
    base["bear_confluence_reasons"] = "+".join(bear_struct_reasons) if bear_struct_reasons else ""

    # ==========================================================
    # PHASE 4B: STRUCTURE-FIRST SETUP DETECTION
    # ==========================================================
    # Philosophy: Market structure events ARE the signal. Geometry (OBs, FVGs)
    # is used downstream in Phase 5 for stop placement, NOT as a trigger here.
    # Each setup computes its own confluence score; Phase 4C picks the best.
    candidates = []

    # ── Session and volatility context for confluence bonuses ──
    _ts_base = base.get("timestamp", 0)
    try:
        _session_hour = pd.to_datetime(int(_ts_base), unit="ms", utc=True).hour if _ts_base else 12
    except Exception:
        _session_hour = 12
    _in_prime_session = 1.0 if (7 <= _session_hour <= 10 or 13 <= _session_hour <= 16) else 0.0
    _in_high_liq = 1.0 if 7 <= _session_hour <= 17 else 0.0
    _is_squeeze = base.get("is_ttm_squeeze", 0)
    _squeeze_fired = base.get("squeeze_fired", 0)
    _orb_bull = base.get("orb_bull", 0)
    _orb_bear = base.get("orb_bear", 0)

    # Pre-fetch structure features used across multiple setups
    _struct_trend = base.get("struct_trend", 0)
    _struct_strength = base.get("struct_strength", 0)
    _htf_struct_trend = base.get("htf_struct_trend", 0)
    _is_discount = base.get("is_discount_zone", 0)
    _is_premium = base.get("is_premium_zone", 0)
    _choch_up = base.get("choch_up", 0)
    _choch_down = base.get("choch_down", 0)
    _disp_bos_bull = base.get("displacement_bos_bull", 0)
    _disp_bos_bear = base.get("displacement_bos_bear", 0)
    _wyckoff_spring = base.get("wyckoff_spring", 0)
    _wyckoff_upthrust = base.get("wyckoff_upthrust", 0)
    _adx = base.get("adx", 0)

    # BOS lookback: check if BOS occurred within last 8 bars
    _bos_up_recent = any(pExec.bos_up[max(0, iExec - 7):iExec + 1] > 0) if hasattr(pExec, 'bos_up') else False
    _bos_down_recent = any(pExec.bos_down[max(0, iExec - 7):iExec + 1] > 0) if hasattr(pExec, 'bos_down') else False

    # Pullback detection for BOS setups: price retraced toward BOS area
    # (price is in lower half of recent range = pulled back after bullish BOS)
    _fib_pos = base.get("fib_position_pct", 50.0)
    _bos_pullback_long = _fib_pos < 50.0   # price in lower half = pullback after up-BOS
    _bos_pullback_short = _fib_pos > 50.0  # price in upper half = pullback after down-BOS

    # ---- SETUP 1: STRUCTURE_LONG (Bullish structure + confluence) ----
    if can_long and _struct_trend > 0 and _struct_strength >= 1:
        if _htf_struct_trend >= 0 and has_vol_confirm:
            sc = 1.0          # base score: 15m structural trend is confirmed
            desc_parts = ["STRUCT_BULL"]
            if _struct_strength >= 2:
                sc += 0.5; desc_parts.append("STRUCT_STR")
            if macro_up > 0:
                sc += 1.0; desc_parts.append("MACRO_UP")
            if swing_up > 0:
                sc += 1.0; desc_parts.append("SWING_UP")
            if htf_up > 0:
                sc += 1.0; desc_parts.append("HTF_UP")
            if struct_align_bull >= 2:
                sc += 1.0; desc_parts.append("STRUCT_ALIGN")
            if _is_discount > 0:
                sc += 1.0; desc_parts.append("DISCOUNT")
            if rvol > 1.5:
                sc += 0.5; desc_parts.append("RVOL")
            if _in_prime_session > 0:
                sc += 1.0; desc_parts.append("SESSION")
            if _is_squeeze > 0 or _squeeze_fired > 0:
                sc += 1.0; desc_parts.append("SQUEEZE")
            candidates.append(("STRUCTURE_LONG", "long",
                f"Bullish HH/HL structure (str={_struct_strength:.0f}) + vol. [{'+'.join(desc_parts)}]", sc))

    # ---- SETUP 2: STRUCTURE_SHORT (Bearish structure + confluence) ----
    if can_short and _struct_trend < 0 and _struct_strength >= 1:
        if _htf_struct_trend <= 0 and has_vol_confirm:
            sc = 1.0
            desc_parts = ["STRUCT_BEAR"]
            if _struct_strength >= 2:
                sc += 0.5; desc_parts.append("STRUCT_STR")
            if macro_down > 0:
                sc += 1.0; desc_parts.append("MACRO_DN")
            if swing_down > 0:
                sc += 1.0; desc_parts.append("SWING_DN")
            if htf_down > 0:
                sc += 1.0; desc_parts.append("HTF_DN")
            if struct_align_bear >= 2:
                sc += 1.0; desc_parts.append("STRUCT_ALIGN")
            if _is_premium > 0:
                sc += 1.0; desc_parts.append("PREMIUM")
            if rvol > 1.5:
                sc += 0.5; desc_parts.append("RVOL")
            if _in_prime_session > 0:
                sc += 1.0; desc_parts.append("SESSION")
            if _is_squeeze > 0 or _squeeze_fired > 0:
                sc += 1.0; desc_parts.append("SQUEEZE")
            candidates.append(("STRUCTURE_SHORT", "short",
                f"Bearish LH/LL structure (str={_struct_strength:.0f}) + vol. [{'+'.join(desc_parts)}]", sc))

    # ---- SETUP 3: BOS_PULLBACK_LONG (BOS up within 8 bars) ----
    if can_long and _bos_up_recent and _struct_trend >= 0 and has_vol_confirm:
        sc = 1.5          # base: confirmed BOS up (strong structural event)
        desc_parts = ["BOS_UP"]
        if _struct_trend > 0:
            sc += 0.5; desc_parts.append("STRUCT_BULL")
        if _bos_pullback_long:
            sc += 0.5; desc_parts.append("PULLBACK")
        if htf_up > 0:
            sc += 1.0; desc_parts.append("HTF_UP")
        if macro_up > 0:
            sc += 1.0; desc_parts.append("MACRO_UP")
        if _disp_bos_bull > 0:
            sc += 1.0; desc_parts.append("DISP+BOS")
        if rvol > 1.5:
            sc += 0.5; desc_parts.append("RVOL")
        if _is_discount > 0:
            sc += 0.5; desc_parts.append("DISCOUNT")
        if _in_prime_session > 0:
            sc += 1.0; desc_parts.append("SESSION")
        if _is_squeeze > 0 or _squeeze_fired > 0:
            sc += 1.0; desc_parts.append("SQUEEZE")
        if _orb_bull > 0:
            sc += 0.5; desc_parts.append("ORB")
        candidates.append(("BOS_PULLBACK_LONG", "long",
            f"BOS up (last 8 bars) + vol. [{'+'.join(desc_parts)}]", sc))

    # ---- SETUP 4: BOS_PULLBACK_SHORT (BOS down within 8 bars) ----
    if can_short and _bos_down_recent and _struct_trend <= 0 and has_vol_confirm:
        sc = 1.5
        desc_parts = ["BOS_DN"]
        if _struct_trend < 0:
            sc += 0.5; desc_parts.append("STRUCT_BEAR")
        if _bos_pullback_short:
            sc += 0.5; desc_parts.append("PULLBACK")
        if htf_down > 0:
            sc += 1.0; desc_parts.append("HTF_DN")
        if macro_down > 0:
            sc += 1.0; desc_parts.append("MACRO_DN")
        if _disp_bos_bear > 0:
            sc += 1.0; desc_parts.append("DISP+BOS")
        if rvol > 1.5:
            sc += 0.5; desc_parts.append("RVOL")
        if _is_premium > 0:
            sc += 0.5; desc_parts.append("PREMIUM")
        if _in_prime_session > 0:
            sc += 1.0; desc_parts.append("SESSION")
        if _is_squeeze > 0 or _squeeze_fired > 0:
            sc += 1.0; desc_parts.append("SQUEEZE")
        if _orb_bear > 0:
            sc += 0.5; desc_parts.append("ORB")
        candidates.append(("BOS_PULLBACK_SHORT", "short",
            f"BOS down (last 8 bars) + vol. [{'+'.join(desc_parts)}]", sc))

    # ---- SETUP 5: CHOCH_LONG (Change of Character — reversal) ----
    # Exempt from regime gate: ChoCH is inherently counter-trend.
    if _choch_up > 0 and bull_reversal_evidence >= 1 and has_vol_confirm:
        _choch_long_allowed = can_long or (bull_reversal_evidence >= 1)  # bypass regime gate for reversals
        if _choch_long_allowed:
            is_reversal_context = True
            sc = 1.0  # base: ChoCH confirmed
            desc_parts = ["CHOCH_UP"]
            if bull_ewo:
                sc += 1.0; desc_parts.append("EWO_DIV")
            if bull_obv:
                sc += 1.0; desc_parts.append("OBV_DIV")
            if bull_cvd:
                sc += 1.0; desc_parts.append("CVD_DIV")
            if bull_rsi_div:
                sc += 1.0; desc_parts.append("RSI_DIV")
            if _wyckoff_spring > 0 or has_spring:
                sc += 1.0; desc_parts.append("SPRING")
            if _is_discount > 0:
                sc += 1.0; desc_parts.append("DISCOUNT")
            if _in_prime_session > 0:
                sc += 1.0; desc_parts.append("SESSION")
            if _is_squeeze > 0 or _squeeze_fired > 0:
                sc += 1.0; desc_parts.append("SQUEEZE")
            candidates.append(("CHOCH_LONG", "long",
                f"Change of Character bullish + {bull_reversal_evidence} reversal signals + vol. [{'+'.join(desc_parts)}]", sc))

    # ---- SETUP 6: CHOCH_SHORT (Change of Character — reversal) ----
    if _choch_down > 0 and bear_reversal_evidence >= 1 and has_vol_confirm:
        _choch_short_allowed = can_short or (bear_reversal_evidence >= 1)
        if _choch_short_allowed:
            is_reversal_context = True
            sc = 1.0
            desc_parts = ["CHOCH_DN"]
            if bear_ewo:
                sc += 1.0; desc_parts.append("EWO_DIV")
            if bear_obv:
                sc += 1.0; desc_parts.append("OBV_DIV")
            if bear_cvd:
                sc += 1.0; desc_parts.append("CVD_DIV")
            if bear_rsi_div:
                sc += 1.0; desc_parts.append("RSI_DIV")
            if _wyckoff_upthrust > 0 or has_upthrust:
                sc += 1.0; desc_parts.append("UPTHRUST")
            if _is_premium > 0:
                sc += 1.0; desc_parts.append("PREMIUM")
            if _in_prime_session > 0:
                sc += 1.0; desc_parts.append("SESSION")
            if _is_squeeze > 0 or _squeeze_fired > 0:
                sc += 1.0; desc_parts.append("SQUEEZE")
            candidates.append(("CHOCH_SHORT", "short",
                f"Change of Character bearish + {bear_reversal_evidence} reversal signals + vol. [{'+'.join(desc_parts)}]", sc))

    # ---- SETUP 7: DISPLACEMENT_CONTINUATION_LONG ----
    if can_long and _disp_bos_bull > 0 and _struct_trend >= 0 and bullish_tf_count >= 1 and has_vol_confirm:
        sc = 1.5  # base: displacement + BOS
        desc_parts = ["DISP+BOS"]
        if macro_up > 0:
            sc += 1.0; desc_parts.append("MACRO_UP")
        if swing_up > 0:
            sc += 1.0; desc_parts.append("SWING_UP")
        if htf_up > 0:
            sc += 1.0; desc_parts.append("HTF_UP")
        if struct_align_bull >= 2:
            sc += 1.0; desc_parts.append("STRUCT_ALIGN")
        if _adx > 25:
            sc += 0.5; desc_parts.append("ADX")
        if _in_prime_session > 0:
            sc += 1.0; desc_parts.append("SESSION")
        if _is_squeeze > 0 or _squeeze_fired > 0:
            sc += 1.0; desc_parts.append("SQUEEZE")
        candidates.append(("DISPLACEMENT_CONT_LONG", "long",
            f"Displacement candle + BOS bullish + {bullish_tf_count}TF trend + vol. [{'+'.join(desc_parts)}]", sc))

    # ---- SETUP 8: DISPLACEMENT_CONTINUATION_SHORT ----
    if can_short and _disp_bos_bear > 0 and _struct_trend <= 0 and bearish_tf_count >= 1 and has_vol_confirm:
        sc = 1.5
        desc_parts = ["DISP+BOS"]
        if macro_down > 0:
            sc += 1.0; desc_parts.append("MACRO_DN")
        if swing_down > 0:
            sc += 1.0; desc_parts.append("SWING_DN")
        if htf_down > 0:
            sc += 1.0; desc_parts.append("HTF_DN")
        if struct_align_bear >= 2:
            sc += 1.0; desc_parts.append("STRUCT_ALIGN")
        if _adx > 25:
            sc += 0.5; desc_parts.append("ADX")
        if _in_prime_session > 0:
            sc += 1.0; desc_parts.append("SESSION")
        if _is_squeeze > 0 or _squeeze_fired > 0:
            sc += 1.0; desc_parts.append("SQUEEZE")
        candidates.append(("DISPLACEMENT_CONT_SHORT", "short",
            f"Displacement candle + BOS bearish + {bearish_tf_count}TF trend + vol. [{'+'.join(desc_parts)}]", sc))

    if not candidates: return None

    # ==========================================================
    # PHASE 4C: HIGH-CONFIDENCE SIGNAL GATE
    # ==========================================================
    # As a signal generator, quality > quantity. Only emit signals when multiple
    # structural factors converge. Single-factor setups are noise.
    struct_trend = base.get("struct_trend", 0)
    htf_struct = base.get("htf_struct_trend", 0)

    # Pick the candidate with the highest confluence score.
    _struct_priority = {"CHOCH": 10, "DISPLACEMENT": 9, "BOS": 8, "STRUCTURE": 7}
    candidates.sort(key=lambda c: (c[3], _struct_priority.get(c[0].split("_")[0], 0)), reverse=True)
    setup_type, side, logic_desc, confluence = candidates[0]

    # HARD CONFLUENCE GATE: Only emit structurally pristine setups.
    # Post-friction physics make marginal 1-factor setups unprofitable. We require
    # the winning candidate to stack at least 2 structural factors (e.g. Order Block
    # + Sweep + Displacement) before the trade is even considered by the brain.
    if confluence < 2.0:
        _p6_counters["confluence_too_low"] += 1
        return None

    # Enrich the logic description with structure context
    conf_reasons = bull_struct_reasons if side == "long" else bear_struct_reasons
    struct_label = "BULLISH" if struct_trend > 0 else ("BEARISH" if struct_trend < 0 else "RANGING")
    htf_label = "HTF_BULL" if htf_struct > 0 else ("HTF_BEAR" if htf_struct < 0 else "HTF_FLAT")
    logic_desc += f" STRUCT={struct_label} {htf_label}."
    if confluence >= 2:
        logic_desc += f" CONFLUENCE {confluence:.0f}× [{'+'.join(conf_reasons)}]."

    if not setup_type: return None

    # ==========================================================
    # PHASE 5: GEOMETRIC ENTRY REFINEMENT + PRECISE SL
    # ==========================================================
    # Structure triggered the signal in Phase 4B. Now use geometric
    # levels (OBs, FVGs, OTE) for two things:
    #   1. Entry precision — snap entry_target to the nearest geometric
    #      level that improves fill price (within 1 ATR of current price).
    #   2. Stop placement — anchor SL at the structural invalidation point.

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

    # ---- Geometric entry refinement ----
    # Collect nearby geometric levels that improve fill price.
    # For longs: lower price = better fill. For shorts: higher = better.
    # Only consider levels within 1 ATR of current price (reachable on
    # a pullback within the next few bars).
    geo_snap_range = current_atr * getattr(cfg, 'geo_entry_snap_atr', 1.0)

    if side == 'long':
        # Candidate entry levels below current price (better fills for longs)
        geo_entries = []
        if ob_bull_ce > 0 and ob_bull_fresh and (px - geo_snap_range) <= ob_bull_ce <= px:
            geo_entries.append(ob_bull_ce)
        if fvg_bull > 0 and (px - geo_snap_range) <= fvg_bull <= px:
            geo_entries.append(fvg_bull)
        if base.get("qm_bull", 0) > 0 and (px - geo_snap_range) <= base["qm_bull"] <= px:
            geo_entries.append(base["qm_bull"])
        # OTE midpoint (70.5% retrace) as entry magnet
        ote_mid = base.get("ote_mid_long", 0)
        if ote_mid > 0 and (px - geo_snap_range) <= ote_mid <= px:
            geo_entries.append(ote_mid)

        if geo_entries:
            # Pick the geometric level closest to current price (most likely to fill)
            entry_target = max(geo_entries)  # highest of the below-price levels = tightest
            logic_desc += f" GEO_ENTRY={entry_target:.1f}."
        else:
            pullback_dist = current_atr * getattr(cfg, 'pullback_atr_mult', 0.0)
            entry_target = px - pullback_dist

    else:  # short
        geo_entries = []
        if ob_bear_ce > 0 and ob_bear_fresh and px <= ob_bear_ce <= (px + geo_snap_range):
            geo_entries.append(ob_bear_ce)
        if fvg_bear > 0 and px <= fvg_bear <= (px + geo_snap_range):
            geo_entries.append(fvg_bear)
        if base.get("qm_bear", 0) > 0 and px <= base["qm_bear"] <= (px + geo_snap_range):
            geo_entries.append(base["qm_bear"])
        ote_mid = base.get("ote_mid_short", 0)
        if ote_mid > 0 and px <= ote_mid <= (px + geo_snap_range):
            geo_entries.append(ote_mid)

        if geo_entries:
            entry_target = min(geo_entries)  # lowest of the above-price levels = tightest
            logic_desc += f" GEO_ENTRY={entry_target:.1f}."
        else:
            pullback_dist = current_atr * getattr(cfg, 'pullback_atr_mult', 0.0)
            entry_target = px + pullback_dist

    # ---- Geometric stop placement ----
    # Use the best available structural invalidation point.
    # Priority: OB edge > FVG edge > QM level > swing level > bar extreme.
    if side == 'long':
        sl_candidates = []
        if ob_bull_bot > 0 and ob_bull_fresh:
            sl_candidates.append(ob_bull_bot - sl_buffer)
        if fvg_bull > 0:
            sl_candidates.append((fvg_bull - fvg_tol) - sl_buffer)
        if base.get("qm_bull", 0) > 0:
            sl_candidates.append(base["qm_bull"] - sl_buffer)
        # Always include swing low as fallback
        sl_candidates.append(min(swing_low, low) - sl_buffer)

        # Pick the tightest SL that is still below entry (maximizes R:R)
        valid_sls = [s for s in sl_candidates if s < entry_target]
        sl = max(valid_sls) if valid_sls else min(swing_low, low) - sl_buffer

        if sl >= entry_target: return None
        risk_distance = entry_target - sl

        # Structural TP targets — proven 6-level approach + ATR fallback
        vwap_val = base.get("rolling_vwap_20", 0)
        poc_val = base.get("poc", 0)
        possible_tps = [t for t in [ob_bear_bot, base.get("vah", 0), htf_sh_val, mtf_sh_val, vwap_val, poc_val] if t > entry_target]
        possible_tps.sort()
        if not possible_tps:
            possible_tps = [entry_target + (risk_distance * getattr(cfg, 'atr_mult_tp', 4.0))]

        selected_tp = None
        for t in possible_tps:
            curr_rr = (t - entry_target) / risk_distance
            if curr_rr >= getattr(cfg, 'min_rr', 2.0):
                if curr_rr > getattr(cfg, 'atr_mult_tp', 3.5):
                    selected_tp = entry_target + (risk_distance * getattr(cfg, 'atr_mult_tp', 3.5))
                else:
                    selected_tp = t
                break

        # ATR fallback when structural TPs exist but none met min_rr
        if selected_tp is None and risk_distance > 0:
            atr_tp_mult = getattr(cfg, 'atr_mult_tp', 4.0)
            selected_tp = entry_target + (risk_distance * atr_tp_mult)

        if selected_tp is None: return None
        tp = selected_tp
        rr = (tp - entry_target) / risk_distance

    else:  # short
        sl_candidates = []
        if ob_bear_top > 0 and ob_bear_fresh:
            sl_candidates.append(ob_bear_top + sl_buffer)
        if fvg_bear > 0:
            sl_candidates.append((fvg_bear + fvg_tol) + sl_buffer)
        if base.get("qm_bear", 0) > 0:
            sl_candidates.append(base["qm_bear"] + sl_buffer)
        sl_candidates.append(max(swing_high, high) + sl_buffer)

        valid_sls = [s for s in sl_candidates if s > entry_target]
        sl = min(valid_sls) if valid_sls else max(swing_high, high) + sl_buffer

        if sl <= entry_target: return None
        risk_distance = sl - entry_target

        vwap_val = base.get("rolling_vwap_20", 0)
        poc_val = base.get("poc", 0)
        possible_tps = [t for t in [ob_bull_top, base.get("val", 0), htf_sl_val, mtf_sl_val, vwap_val, poc_val] if 0 < t < entry_target]
        possible_tps.sort(reverse=True)
        if not possible_tps:
            possible_tps = [entry_target - (risk_distance * getattr(cfg, 'atr_mult_tp', 4.0))]

        selected_tp = None
        for t in possible_tps:
            curr_rr = (entry_target - t) / risk_distance
            if curr_rr >= getattr(cfg, 'min_rr', 2.0):
                if curr_rr > getattr(cfg, 'atr_mult_tp', 3.5):
                    selected_tp = entry_target - (risk_distance * getattr(cfg, 'atr_mult_tp', 3.5))
                else:
                    selected_tp = t
                break

        # ATR fallback when structural TPs exist but none met min_rr
        if selected_tp is None and risk_distance > 0:
            atr_tp_mult = getattr(cfg, 'atr_mult_tp', 4.0)
            selected_tp = entry_target - (risk_distance * atr_tp_mult)

        if selected_tp is None: return None
        tp = selected_tp
        rr = (entry_target - tp) / risk_distance

    # Minimum target distance gate
    if abs(tp - entry_target) / entry_target * 100.0 < getattr(cfg, 'min_target_dist_pct', 0.15):
        return None

    # ----------------------------------------------------------
    # MINIMUM RISK DISTANCE FILTER (friction-dominance guard)
    # Reject setups where round-trip friction would consume an
    # unacceptable fraction of 1R, making the trade structurally
    # unprofitable regardless of directional accuracy.
    # ----------------------------------------------------------
    min_risk = compute_min_risk_distance(cfg, entry_target)
    if risk_distance < min_risk:
        log.debug(
            "SKIP %s %s: risk_distance %.2f < min_risk %.2f "
            "(friction would exceed %.0f%% of 1R)",
            setup_type, side, risk_distance, min_risk,
            getattr(cfg, "max_friction_pct", 0.20) * 100,
        )
        _p6_counters["risk_dist_too_small"] += 1
        return None

    # ── Funding-window filter ──
    # Bybit funding settles at 00:00, 08:00, 16:00 UTC. Entering within
    # `funding_buffer_minutes` of a settlement means paying funding almost
    # immediately. Skip if too close.
    funding_buffer_minutes = getattr(cfg, 'funding_buffer_minutes', 30)
    if funding_buffer_minutes > 0:
        ts_now = base.get("timestamp", 0)
        if ts_now > 0:
            from datetime import datetime, timezone
            dt_now = datetime.fromtimestamp(ts_now / 1000, tz=timezone.utc)
            hour = dt_now.hour
            minute = dt_now.minute
            # Minutes to next funding settlement (00, 08, 16 UTC)
            next_funding_hour = ((hour // 8) + 1) * 8
            if next_funding_hour >= 24:
                next_funding_hour = 24  # wraps to 00:00 next day
            minutes_to_funding = (next_funding_hour - hour) * 60 - minute
            if 0 < minutes_to_funding < funding_buffer_minutes:
                _p6_counters["funding_window_rejected"] += 1
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
        _p6_counters["reached_phase6"] += 1

        # ── Meta-Brain: Two-tier decision system ──
        meta_prob, specialist_prob = brain.predict_meta_probability(
            strat, side, base, adv, iExec, px, pExec
        )
        if meta_prob is None:
            _p6_counters["specialist_none"] += 1
            return None

        # Specialist threshold — dynamically adjusted by brain quality tier.
        # Weak brains need higher confidence to pass, but still provide filtering.
        base_min_prob = getattr(cfg, 'min_prob_long', 0.50) if side == 'long' else getattr(cfg, 'min_prob_short', 0.50)
        threshold_adj = brain.get_threshold_adj(strat, side)
        min_prob = base_min_prob + threshold_adj

        if brain.meta_brain is not None and specialist_prob is not None:
            # Track probability distributions
            _p6_counters["specialist_prob_sum"] += specialist_prob
            _p6_counters["specialist_prob_count"] += 1
            _p6_counters["meta_prob_sum"] += meta_prob
            _p6_counters["meta_prob_count"] += 1

            # Two-gate system: specialist must clear its threshold AND meta must clear its own
            if specialist_prob < min_prob:
                _p6_counters["specialist_below_thresh"] += 1
                return None
            meta_threshold = getattr(cfg, 'min_meta_prob', 0.22)
            if meta_prob < meta_threshold:
                _p6_counters["meta_below_thresh"] += 1
                return None
            # Blend: specialist drives the probability estimate, meta scales confidence
            # meta_prob > base_rate means meta agrees; < base_rate means meta disagrees
            meta_base_rate = 0.25  # approximate positive rate in training
            meta_confidence = meta_prob / max(meta_base_rate, 0.01)  # >1 = agree, <1 = disagree
            win_prob = specialist_prob * min(meta_confidence, 1.5)  # Cap the boost at 1.5x
            win_prob = min(win_prob, 0.95)  # Safety cap
        else:
            # Specialist-only mode (no meta-brain loaded)
            _p6_counters["specialist_prob_sum"] += meta_prob
            _p6_counters["specialist_prob_count"] += 1
            win_prob = meta_prob  # meta_prob == specialist_prob in fallback mode
            if win_prob < min_prob:
                _p6_counters["specialist_below_thresh"] += 1
                return None

        # ── EV computation: win_prob is now isotonically calibrated at source (predict_probability).
        # No more invented shrinkage formula. EV is computed directly from the calibrated probability.
        ev = (win_prob * rr) - ((1.0 - win_prob) * 1.0)

        # Optional: log the calibration provenance so we can audit
        brain_data = brain.brains.get((strat, side)) or {}
        calib_method = brain_data.get("calibration_method", "none")
        min_ev = getattr(cfg, 'min_ev', 0.0)
        log.debug(
            f"EV computation: prob={win_prob:.3f} (calibration={calib_method}) "
            f"rr={rr:.2f} ev={ev:.3f}"
        )
        if ev <= min_ev:
            _p6_counters["ev_too_low"] += 1
            return None

        if getattr(cfg, 'dynamic_risk_scaling', True):
            risk_factor = min(1.0 + (ev ** 1.5), getattr(cfg, 'max_risk_factor', 2.5))

        regime = "REVERSAL" if is_reversal_context else "TREND"
        confidence = "HIGH" if win_prob > 0.60 else "MODERATE"
        meta_tag = ""
        if brain.meta_brain is not None and specialist_prob is not None:
            meta_tag = f" Meta: {win_prob*100:.1f}% | Specialist: {specialist_prob*100:.1f}%."
        analysis_str = (
            f"SETUP: {setup_type} [{regime}]. {logic_desc} "
            f"AI: {win_prob*100:.1f}% ({confidence}, EV: {ev:.2f}R).{meta_tag} "
            f"R:R {rr:.1f}. TF alignment: {bullish_tf_count}B/{bearish_tf_count}S. "
            f"Invalidation: {rev_warn:.2f}."
        )
    else:
        _p6_counters["no_brain"] += 1

    _p6_counters["passed"] += 1
    strat_base_key = setup_type.split("_")[0] if setup_type else "UNKNOWN"
    _p6_counters["per_strategy_passed"][strat_base_key] = \
        _p6_counters["per_strategy_passed"].get(strat_base_key, 0) + 1
    # DIRECTIVE 2 — EXECUTION ROUTER FIX:
    # The order type MUST be dynamically inherited from the global config's
    # ``execution_style`` property via a single safe ``getattr`` fallback.
    # NEVER hardcode "market" / "limit" in the return dict — a literal here
    # nullifies the live L2 order-book spoofing defense which relies on the
    # limit-order path. The ``getattr`` fallback of ``'limit'`` is the only
    # tolerated safety default, and the TradeManager carries an identical
    # secondary safety net at fill time.
    return {
        "side": side, "entry": entry_target, "sl": sl, "tp": tp, "rr": rr,
        "prob": win_prob, "key": f"{setup_type}_{side}", "features": base,
        "risk_factor": risk_factor, "strategy": setup_type,
        "type": getattr(cfg, 'execution_style', 'limit'),
        "analysis": analysis_str,
        "atr_at_signal": float(current_atr),
    }