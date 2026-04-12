#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""vajra_macro_data.py — Macro and dominance data fetchers.

Replaces the BTCDOM perpetual proxy with a true BTC-dominance time series
sourced from a local cache (populated offline from CoinGecko or CryptoCompare).

Also provides a shared compute_session_flags() helper used by both the training
export path and the inference path, ensuring a single source of truth for
timezone-correct session flags.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

log = logging.getLogger("vajra.macro_data")
if not log.handlers:
    log.setLevel(logging.INFO)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    ))
    log.addHandler(h)


CACHE_DIR = Path("data_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_TZ_LONDON = ZoneInfo("Europe/London")
_TZ_NY = ZoneInfo("America/New_York")


# ─── Session flags (single source of truth) ──────────────────────────────

def compute_session_flags(ts_ms: int) -> dict:
    """Return timezone-correct London and NY session flags for a given timestamp.

    London FX session:  08:00–17:00 Europe/London (handles BST/GMT automatically)
    NY equity session:  09:30–16:00 America/New_York (handles EST/EDT automatically)
    """
    dt = pd.to_datetime(ts_ms, unit="ms", utc=True)
    london = dt.astimezone(_TZ_LONDON)
    ny = dt.astimezone(_TZ_NY)
    ny_minutes = ny.hour * 60 + ny.minute
    return {
        "is_london_session": 1.0 if 8 <= london.hour < 17 else 0.0,
        "is_ny_session": 1.0 if 570 <= ny_minutes < 960 else 0.0,  # 9:30 → 16:00
    }


# ─── BTC Dominance ───────────────────────────────────────────────────────

def fetch_btc_dominance_series(
    start_ms: int, end_ms: int, exec_timestamps: pd.Series
) -> np.ndarray:
    """Return a per-bar BTC dominance trend signal aligned to exec_timestamps.

    The signal is the 5-day rate-of-change of BTC dominance, mapped to {-1, 0, +1}:
      slope > 0.5%  -> +1 (BTC outperforming alts)
      slope < -0.5% -> -1 (alts outperforming BTC)
      else          ->  0

    Source: a pre-populated CSV cache at data_cache/btc_dominance_history.csv.
    If the cache is missing or stale, returns zeros and logs a clear warning.
    """
    cache_file = CACHE_DIR / "btc_dominance_history.csv"

    if cache_file.exists():
        try:
            df = pd.read_csv(cache_file)
            df["timestamp"] = pd.to_numeric(df["timestamp"])
            df_in_range = df[
                (df["timestamp"] >= start_ms - 30 * 86400000)
                & (df["timestamp"] <= end_ms + 86400000)
            ]
            if len(df_in_range) > 30:
                return _compute_dominance_trend(df_in_range, exec_timestamps)
        except Exception as e:
            log.warning(f"BTC dominance cache load failed: {e}")

    log.warning(
        "BTC dominance cache file not found or out of range. "
        f"Create {cache_file} with columns: timestamp,btc_dominance_pct. "
        "Returning zeros (signal disabled). See vajra_macro_data.py docstring."
    )
    return np.zeros(len(exec_timestamps))


def _compute_dominance_trend(
    df: pd.DataFrame, exec_timestamps: pd.Series
) -> np.ndarray:
    """Map BTC dominance pct -> {-1, 0, +1} trend signal at each exec timestamp."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    btcd = df["btc_dominance_pct"].values

    # 5-bar rate of change (daily samples)
    trend = np.zeros_like(btcd)
    for i in range(5, len(btcd)):
        if btcd[i - 5] > 0:
            slope = (btcd[i] - btcd[i - 5]) / btcd[i - 5] * 100.0
            if slope > 0.5:
                trend[i] = 1.0
            elif slope < -0.5:
                trend[i] = -1.0

    # Align onto exec timestamps via forward-fill (with 1-bar lag for causality)
    ts_index = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    series = pd.Series(trend, index=ts_index)
    aligned = (
        series.shift(1)
        .reindex(pd.to_datetime(exec_timestamps, unit="ms", utc=True), method="ffill")
        .fillna(0.0)
        .values
    )
    return aligned


def populate_dominance_cache(api_key: str | None = None):
    """Background utility: populate data_cache/btc_dominance_history.csv.

    Run this script offline before backtesting:
        python -c "from vajra_macro_data import populate_dominance_cache; populate_dominance_cache()"

    For production, set up a daily cron that calls the CoinGecko /global endpoint
    and appends to the CSV:
        timestamp,btc_dominance_pct
        1577836800000,67.5
        ...
    """
    log.info("Attempting to populate BTC dominance cache...")
    try:
        url = "https://api.coingecko.com/api/v3/global"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode())

        btc_dom = data.get("data", {}).get("market_cap_percentage", {}).get("btc")
        if btc_dom is not None:
            cache_file = CACHE_DIR / "btc_dominance_history.csv"
            ts_ms = int(time.time() * 1000)
            row = f"{ts_ms},{btc_dom:.2f}\n"

            if cache_file.exists():
                with open(cache_file, "a") as f:
                    f.write(row)
            else:
                with open(cache_file, "w") as f:
                    f.write("timestamp,btc_dominance_pct\n")
                    f.write(row)
            log.info(f"Appended BTC dominance snapshot: {btc_dom:.2f}% at {ts_ms}")
        else:
            log.warning("CoinGecko /global response missing btc market_cap_percentage")
    except Exception as e:
        log.error(f"Cache population failed: {e}")
        log.warning(
            "Manual cache population required. Run a daily scheduled script that calls "
            "https://api.coingecko.com/api/v3/global and appends to "
            "data_cache/btc_dominance_history.csv with columns: timestamp,btc_dominance_pct"
        )


# ─── Macro data sanity check ─────────────────────────────────────────────

def sanity_check_macro_caches() -> bool:
    """Verify all required macro data sources are present and fresh."""
    issues = []

    # BTC dominance cache
    btcd_cache = CACHE_DIR / "btc_dominance_history.csv"
    if not btcd_cache.exists():
        issues.append(f"MISSING: {btcd_cache}")
    else:
        df = pd.read_csv(btcd_cache)
        if len(df) < 100:
            issues.append(f"STALE: {btcd_cache} has only {len(df)} rows")
        else:
            latest_ts = pd.to_numeric(df["timestamp"]).max()
            age_days = (time.time() * 1000 - latest_ts) / (86400 * 1000)
            if age_days > 7:
                issues.append(
                    f"STALE: {btcd_cache} latest sample is {age_days:.1f} days old"
                )

    # Oracle sentiment cache
    sentiment_cache = CACHE_DIR / "oracle_sentiment.json"
    if sentiment_cache.exists():
        with open(sentiment_cache) as f:
            data = json.load(f)
        age_minutes = (time.time() * 1000 - data.get("timestamp", 0)) / (60 * 1000)
        if age_minutes > 60:
            issues.append(
                f"STALE: {sentiment_cache} is {age_minutes:.0f} min old"
            )

    if issues:
        print("MACRO DATA SANITY CHECK: ISSUES FOUND")
        for i in issues:
            print(f"  - {i}")
        return False
    else:
        print("MACRO DATA SANITY CHECK: ALL GOOD")
        return True


if __name__ == "__main__":
    import sys

    ok = sanity_check_macro_caches()
    sys.exit(0 if ok else 1)
