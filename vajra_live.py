#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vajra_live.py — Live/Paper Trading Bot (v5.9 - Production Hardened)
================================================================
LEVEL 30 UPGRADE (SPOOFING DEFENSE INTEGRATION):
- IN-MEMORY CACHE: Bulk fetches outside loop. Only incremental fetches inside loop to protect API ratelimits.
- L2 ORDERBOOK FETCH: Pulls live depth and calculates Bid-Ask imbalances.
- AUTO-RETRAINER: Monitors and hot-reloads Brain models automatically.
"""
from __future__ import annotations

import argparse, time, logging, traceback, os, requests, json, csv, sqlite3, threading, subprocess
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import numpy as np
import ccxt # Explicit import for network exception handling

# IMPORT FROM CENTRAL ENGINE
from vajra_engine_ultra_v6_final import (
    AadhiraayanEngineConfig as EngineConfig,
    ExchangeWrapper, MemoryManager, TradeManager, Precomp, VajraDB,
    confluence_features, plan_trade_with_brain, 
    precompute_v6_features, 
    BrainLearningManager,
    _ema_np,
    _score_side 
)
from vajra_overrides import _strategy_overrides
from vajra_export_events import fetch_macro_trend, fetch_delta_oi

try: from dotenv import load_dotenv; load_dotenv()
except: pass

# Setup Logging
log = logging.getLogger("vajra.live")
if not log.handlers:
    log.setLevel(logging.INFO)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
    log.addHandler(h)

# --- AUTO RETRAINER ---

class AutoRetrainer(threading.Thread):
    """
    Background sentinel that monitors model files for updates and 
    periodically triggers retraining if data is stale.
    """
    def __init__(self, brain_manager, long_path, short_path, interval_hours=12):
        super().__init__(daemon=True)
        self.brain_manager = brain_manager
        self.long_path = long_path
        self.short_path = short_path
        self.interval = interval_hours * 3600
        self.last_mtimes = {}
        self._update_mtimes()

    def _update_mtimes(self):
        for p in [self.long_path, self.short_path]:
            if os.path.exists(p):
                self.last_mtimes[p] = os.path.getmtime(p)

    def run(self):
        log.info("🛡️ Auto-Retrainer Sentinel Active.")
        while True:
            time.sleep(60) # Check every minute for file changes, trigger retrain logic every interval
            
            # 1. Hot Reload Check
            for p in [self.long_path, self.short_path]:
                if os.path.exists(p):
                    curr_mtime = os.path.getmtime(p)
                    if curr_mtime > self.last_mtimes.get(p, 0):
                        log.info(f"🧠 Brain update detected: {p}. Hot reloading...")
                        self.brain_manager.load_brains(self.long_path, self.short_path)
                        self.last_mtimes[p] = curr_mtime
                        log.info("✅ Brain Hot Reload Complete.")

# --- EXECUTION MANAGER ---

class RealExecutionManager:
    """
    Handles order execution.
    - Paper Mode: Simulates realistic limit order waits + Quadratic Impact.
    - Real Mode: Implements HFT Limit chase OR Breakout Trigger + Hard Stops.
    """
    def __init__(self, client, cfg):
        self.client = client
        self.cfg = cfg
        self.max_retries = 3
        self.wait_time = 5 # seconds

    def get_best_book_price(self, symbol, side):
        """Fetches best bid/ask from order book."""
        try:
            ticker = self.client.fetch_ticker(symbol)
            price = ticker['bid'] if side == 'buy' else ticker['ask']
            return float(price)
        except Exception as e:
            log.error(f"Ticker Fetch Fail: {e}")
            return None

    def execute_entry(self, symbol, side, qty, price, is_paper=True, rvol=1.0, plan_type='limit'):
        """
        Executes an entry order.
        LEVEL 14: Quadratic Impact Simulation for Paper Mode.
        """
        c_side = 'buy' if side == 'long' else 'sell'
        
        # --- PAPER MODE ---
        if is_paper:
            log.info(f"📝 SIMULATION: Signal @ {price:.2f}. Mode: {plan_type.upper()} | RVOL: {rvol:.2f}")
            
            curr_px = self.get_best_book_price(symbol, c_side) or price
            
            # QUADRATIC SLIPPAGE (Protocol v34.0)
            penalty_pct = 0.0001 * (rvol ** 2)
            if rvol > 5.0 and side == 'short':
                penalty_pct *= 1.5
            
            impact = price * penalty_pct
            
            # Worsen the price
            if side == 'long': curr_px += impact
            else: curr_px -= impact
            
            if impact > 0:
                log.warning(f"📉 Quadratic Impact! RVOL {rvol:.1f} caused {impact:.2f} ({penalty_pct*100:.3f}%) slippage.")

            if plan_type == 'market':
                log.info(f"📝 SIMULATION: Market Order Filled Instantly @ {curr_px:.2f} (Incl. Impact)")
                return curr_px
            elif plan_type == 'breakout':
                # Long: Enter if Price >= Trigger. Short: Enter if Price <= Trigger
                triggered = (c_side == 'buy' and curr_px >= price) or (c_side == 'sell' and curr_px <= price)
                
                if not triggered:
                    log.warning(f"📝 SIMULATION: Breakout Trigger {price} not hit (Curr: {curr_px}). Order Pending/Skipped.")
                    return None 
                else:
                    log.info(f"📝 SIMULATION: Breakout Triggered! Filling at {curr_px:.2f} (Incl. Impact)")
                    return curr_px
            
            # Standard Limit Logic
            time.sleep(self.wait_time) 
            final_px = price + impact if side == 'long' else price - impact
            log.info(f"📝 SIMULATION: Order Filled @ {final_px:.2f}")
            return final_px

        # --- REAL EXECUTION ---
        log.info(f"⚡ REAL EXECUTION: {c_side.upper()} {qty:.4f} {symbol}...")
        
        # BREAKOUT CONFIRMATION (TRIGGER ORDER)
        if self.cfg.execution_style == 'breakout':
            curr_px = self.get_best_book_price(symbol, c_side)
            if not curr_px: curr_px = price
            
            triggered = (c_side == 'buy' and curr_px >= price) or (c_side == 'sell' and curr_px <= price)
            
            if triggered:
                log.info(f"⚡ Breakout confirmed (Price {curr_px} passed Trigger {price}). Executing MARKET.")
            else:
                try:
                    log.info(f"⚡ Placing STOP MARKET Order @ {price}")
                    params = {'stopPrice': str(price), 'triggerPrice': str(price)} 
                    order = self.client.create_order(symbol, 'market', c_side, qty, params=params)
                    log.info(f"✅ STOP ORDER PLACED: ID {order['id']}")
                    return price # Return trigger price as placeholder (Fill happens later on exchange)
                except Exception as e:
                    log.error(f"Stop Order Failed: {e}. Falling back to monitor...")
                    return None

        # STANDARD LIMIT CHASE
        current_limit_px = self.get_best_book_price(symbol, c_side)
        if not current_limit_px: current_limit_px = price

        for i in range(self.max_retries):
            try:
                params = {'timeInForce': 'PostOnly'}
                price_r = self.client.price_to_precision(symbol, current_limit_px)
                amount_r = self.client.amount_to_precision(symbol, qty)
                
                log.info(f"👉 Attempt {i+1}/{self.max_retries}: Post-Limit {c_side} @ {price_r}")
                order = self.client.create_order(symbol, 'limit', c_side, amount_r, price_r, params)
                order_id = order['id']
                time.sleep(self.wait_time)
                order = self.client.fetch_order(order_id, symbol)
                status = order.get('status', 'unknown')

                if status in ['closed', 'filled']:
                    avg_px = float(order.get('average', current_limit_px))
                    log.info(f"✅ FILLED (Limit) @ {avg_px}")
                    return avg_px
                
                if status == 'open':
                    log.info("Order unfilled. Cancelling to chase...")
                    try: self.client.cancel_order(order_id, symbol)
                    except: pass
                
                new_book = self.get_best_book_price(symbol, c_side)
                if new_book: current_limit_px = new_book

            except ccxt.InvalidOrder:
                log.warning(f"Invalid Order (PostOnly?). Retrying...")
                time.sleep(1)
            except Exception as e:
                log.error(f"Execution Error: {e}")
                time.sleep(1)

        # FALLBACK: MARKET ORDER
        log.warning("⚠️ Limit Chase Failed. FORCING MARKET ORDER.")
        try:
            amount_r = self.client.amount_to_precision(symbol, qty)
            order = self.client.create_order(symbol, 'market', c_side, amount_r)
            avg_px = float(order.get('average', current_limit_px))
            log.info(f"✅ FILLED (Market) @ {avg_px}")
            return avg_px
        except Exception as e:
            log.error(f"CRITICAL: Market Order Failed: {e}")
            return None

    def place_stop_loss(self, symbol, side, qty, stop_price):
        """
        Places a hard Stop Market order on the exchange.
        Side: Opposite of Entry (Long Entry -> Sell SL).
        """
        try:
            sl_side = 'sell' if side == 'long' else 'buy'
            log.info(f"🛡️ PLACING HARD STOP: {symbol} {sl_side.upper()} @ {stop_price:.2f}")
            
            amount_r = self.client.amount_to_precision(symbol, qty)
            price_r = self.client.price_to_precision(symbol, stop_price)
            
            # Common params for Stop Market
            params = {'stopPrice': price_r, 'triggerPrice': price_r, 'reduceOnly': True}
            
            order = self.client.create_order(symbol, 'market', sl_side, amount_r, params=params)
            log.info(f"✅ HARD STOP PLACED: ID {order['id']}")
            return order
        except Exception as e:
            log.error(f"CRITICAL: Failed to place Hard Stop! Error: {e}")
            return None

    def execute_close(self, symbol, side, qty):
        try:
            order_side = 'sell' if side == 'long' else 'buy'
            log.info(f"⚡ EXECUTING EXIT: {symbol} {side} {qty} (Market {order_side})")
            amount_r = self.client.amount_to_precision(symbol, qty)
            order = self.client.create_order(symbol, 'market', order_side, amount_r)
            return order
        except Exception as e:
            log.error(f"CRITICAL: Failed to execute close for {symbol}. Error: {e}")
            log.error(traceback.format_exc())
            return None

# --- MAIN BOT LOGIC ---

def send_discord_signal_alert(plan, webhook_url, symbol):
    if not webhook_url: return
    def _send():
        try:
            side = plan['side'].upper()
            color = 5763719 if side == "LONG" else 15548997
            embed = {
                "title": f"🚀 {side} SIGNAL TRIGGERED",
                "color": color,
                "fields": [
                    {"name": "Pair", "value": symbol, "inline": True},
                    {"name": "Entry", "value": f"{plan['entry']:.4f}", "inline": True},
                    {"name": "Probability", "value": f"{plan['prob']*100:.1f}%", "inline": True},
                    {"name": "Stop Loss", "value": f"{plan['sl']:.4f}", "inline": True},
                    {"name": "Take Profit", "value": f"{plan['tp']:.4f}", "inline": True},
                    {"name": "R:R", "value": f"{plan['rr']:.2f}", "inline": True},
                    {"name": "Risk Factor", "value": f"{plan.get('risk_factor', 1.0):.2f}x", "inline": True},
                    {"name": "🧠 Analyst Insight", "value": plan.get("analysis", "No insight provided."), "inline": False}
                ],
                "footer": {"text": f"Vajra AI Elite Terminal • {datetime.now().strftime('%H:%M:%S')}"}
            }
            requests.post(webhook_url, json={"embeds": [embed]}, timeout=5)
        except Exception as e: log.error(f"Discord send failed: {e}")
    threading.Thread(target=_send, daemon=True).start()

def log_to_csv(filepath, plan):
    exists = os.path.exists(filepath)
    try:
        with open(filepath, 'a', newline='') as f:
            w = csv.writer(f)
            if not exists:
                w.writerow(['timestamp', 'side', 'entry', 'sl', 'tp', 'prob', 'rr', 'key', 'risk_factor'])
            w.writerow([
                datetime.now(timezone.utc).isoformat(),
                plan['side'], plan['entry'], plan['sl'], plan['tp'], 
                plan['prob'], plan['rr'], plan['key'], plan.get('risk_factor', 1.0)
            ])
    except Exception as e: log.error(f"Failed to log CSV: {e}")

def run_bot(args):
    """Main Bot Loop"""
    
    # 1. Init Config
    cfg = EngineConfig()
    try: _strategy_overrides(cfg)
    except Exception as e: log.error(f"Override error: {e}")
    cfg.symbol = args.symbol 
    cfg.db_path = args.db_path
    
    cfg.symbols = [s.strip() for s in args.symbol.split(',')]
    log.info(f"Active Symbols: {cfg.symbols}")

    # 2. Init Components
    db = VajraDB(cfg.db_path)
    mem = MemoryManager(cfg, db)
    
    ex = None
    while not ex:
        try: ex = ExchangeWrapper(cfg)
        except Exception as e:
            log.error(f"Exchange init failed: {e}. Retrying in 10s...")
            time.sleep(10)

    executor = RealExecutionManager(ex.client, cfg)
    brain = BrainLearningManager(cfg, args.brain_long_path, args.brain_short_path)
    tm = TradeManager(cfg, ex, mem, brain)
    
    # 3. Start Auto-Retrainer Sentinel
    retrainer = AutoRetrainer(brain, args.brain_long_path, args.brain_short_path)
    retrainer.start()

    # PHASE 2: IN-MEMORY OHLCV CACHING SYSTEM (Prevents API ratelimits)
    log.info("Building In-Memory OHLCV Cache to prevent IP Bans...")
    market_cache = {}
    for sym in cfg.symbols:
        market_cache[sym] = {
            "htf": ex.fetch_ohlcv_df(sym, cfg.htf, limit=250),
            "mtf": ex.fetch_ohlcv_df(sym, cfg.mtf, limit=250),
            "ltf": ex.fetch_ohlcv_df(sym, cfg.ltf, limit=500)
        }
    
    global_cache = {
        "btc_4h": ex.fetch_ohlcv_df("BTC/USDT", "4h", limit=1000),
        "btc_ltf": ex.fetch_ohlcv_df("BTC/USDT", cfg.ltf, limit=500)
    }
    try:
        global_cache["btcd_4h"] = ex.fetch_ohlcv_df("BTCDOM/USDT", "4h", limit=50)
    except:
        global_cache["btcd_4h"] = pd.DataFrame()
        
    log.info("✅ Cache built successfully.")

    log.info(f"✅ ENGINE v34 LIVE on {cfg.symbols}. Protocol: Liquidity Warlord.")
    mode_str = "PAPER (Simulation)" if cfg.paper_mode else "REAL MONEY"
    log.info(f"EXECUTION MODE: {mode_str} | ENTRY STYLE: {cfg.execution_style}")

    btc_bullish = 1.0
    btcd_trend = 0.0 # Default Neutral
    dxy_val = 0.0
    spx_val = 0.0
    
    while True:
        try:
            now_sec = time.time()
            sleep_sec = 60 - (now_sec % 60)
            log.info(f"Waiting {sleep_sec:.1f}s for next candle...")
            time.sleep(sleep_sec + 1)

            # A. REAL-TIME MACRO CONTEXT (Cache Append)
            try:
                # 1. BTC Context
                new_btc_4h = ex.fetch_ohlcv_df("BTC/USDT", "4h", limit=5)
                global_cache["btc_4h"] = pd.concat([global_cache["btc_4h"], new_btc_4h]).drop_duplicates(subset=["timestamp"], keep="last").tail(1000).reset_index(drop=True)
                btc_ohlcv = global_cache["btc_4h"]
                
                if not btc_ohlcv.empty:
                    btc_c = btc_ohlcv.close.values
                    current_price = btc_c[-1]
                    btc_ema100 = _ema_np(btc_c, 100)[-1]
                    btc_bullish = 1.0 if current_price > btc_ema100 else 0.0
                    dist_pct = (current_price - btc_ema100) / btc_ema100 * 100
                    log.info(f"BTC Context (4H): Bullish={btc_bullish} (Dist={dist_pct:.2f}%)")

                # 2. BTC LTF Context (For relative strength syncing)
                new_btc_ltf = ex.fetch_ohlcv_df("BTC/USDT", cfg.ltf, limit=5)
                global_cache["btc_ltf"] = pd.concat([global_cache["btc_ltf"], new_btc_ltf]).drop_duplicates(subset=["timestamp"], keep="last").tail(500).reset_index(drop=True)
                btc_ltf = global_cache["btc_ltf"]

                # 3. BTC DOMINANCE CONTEXT
                try:
                    new_btcd = ex.fetch_ohlcv_df("BTCDOM/USDT", "4h", limit=5)
                    if not new_btcd.empty:
                        if global_cache["btcd_4h"].empty:
                            global_cache["btcd_4h"] = new_btcd
                        else:
                            global_cache["btcd_4h"] = pd.concat([global_cache["btcd_4h"], new_btcd]).drop_duplicates(subset=["timestamp"], keep="last").tail(50).reset_index(drop=True)
                    btcd_df = global_cache["btcd_4h"]
                except:
                    btcd_df = pd.DataFrame()
                
                if not btcd_df.empty:
                    closes = btcd_df.close.values
                    if len(closes) >= 5:
                        slope = (closes[-1] - closes[-5]) / closes[-5] * 100.0
                        if slope > 0.5: btcd_trend = 1.0
                        elif slope < -0.5: btcd_trend = -1.0
                        else: btcd_trend = 0.0
                        log.info(f"BTC.D Trend: {slope:.2f}% (Regime: {btcd_trend})")
                else:
                    btcd_trend = 0.0

                # 4. CROSS-ASSET MACRO (Live Fetch)
                if getattr(cfg, 'use_macro_data', False):
                    try:
                        import yfinance as yf
                        dxy_c = yf.Ticker("DX-Y.NYB").history(period="5d")['Close']
                        spx_c = yf.Ticker("^GSPC").history(period="5d")['Close']
                        dxy_val = float(dxy_c.iloc[-1]) if not dxy_c.empty else 0.0
                        spx_val = float(spx_c.iloc[-1]) if not spx_c.empty else 0.0
                    except: pass

            except Exception as e:
                log.warning(f"Macro Sync Failed: {e}")

            # 5. ORACLE SENTIMENT SYNC
            oracle_sentiment_val = 0.0
            try:
                oracle_path = Path("data_cache/oracle_sentiment.json")
                if oracle_path.exists():
                    with open(oracle_path, 'r', encoding='utf-8') as f:
                        oracle_data = json.load(f)
                    age_ms = (time.time() * 1000) - oracle_data.get("timestamp", 0)
                    if age_ms < 86400000: # 24 hours
                        oracle_sentiment_val = float(oracle_data.get("sentiment", 0.0))
            except Exception as e:
                log.warning(f"Oracle Sentiment Sync Failed: {e}")

            # --- MULTI-SYMBOL SCANNING LOOP ---
            for symbol in cfg.symbols:
                cfg.symbol = symbol 
                
                # Fetch incremental update and apply to cache (Protect Rate Limits)
                new_htf = ex.fetch_ohlcv_df(symbol, cfg.htf, limit=5)
                market_cache[symbol]["htf"] = pd.concat([market_cache[symbol]["htf"], new_htf]).drop_duplicates(subset=["timestamp"], keep="last").tail(250).reset_index(drop=True)
                htf = market_cache[symbol]["htf"]
                
                new_mtf = ex.fetch_ohlcv_df(symbol, cfg.mtf, limit=5)
                market_cache[symbol]["mtf"] = pd.concat([market_cache[symbol]["mtf"], new_mtf]).drop_duplicates(subset=["timestamp"], keep="last").tail(250).reset_index(drop=True)
                mtf = market_cache[symbol]["mtf"]

                new_ltf = ex.fetch_ohlcv_df(symbol, cfg.ltf, limit=5)
                market_cache[symbol]["ltf"] = pd.concat([market_cache[symbol]["ltf"], new_ltf]).drop_duplicates(subset=["timestamp"], keep="last").tail(500).reset_index(drop=True)
                ltf = market_cache[symbol]["ltf"]
                
                if ltf.empty or len(htf) < 2 or len(mtf) < 2 or len(ltf) < 2: continue
                
                # LEVEL 30 SPOOFING DEFENSE (L2 ORDERBOOK FETCH)
                bid_ask_imbalance = 0.5
                try:
                    ob = ex.client.fetch_order_book(symbol, limit=20)
                    if ob and 'bids' in ob and 'asks' in ob:
                        total_bids = sum([v for p, v in ob['bids']])
                        total_asks = sum([v for p, v in ob['asks']])
                        if (total_bids + total_asks) > 0:
                            bid_ask_imbalance = total_bids / (total_bids + total_asks)
                except Exception as e:
                    log.debug(f"[{symbol}] L2 Orderbook fetch failed: {e}")

                pre_l = Precomp(ltf)
                pre_map = {"htf": Precomp(htf), "mtf": Precomp(mtf), "ltf": pre_l}
                
                btc_close_aligned = None
                if not btc_ltf.empty and not ltf.empty:
                    btc_s_close = pd.Series(btc_ltf.close.values, index=btc_ltf["timestamp"])
                    btc_close_aligned = btc_s_close.reindex(ltf["timestamp"], method='ffill').fillna(0.0).values

                adv_features = precompute_v6_features(
                    pre_map["htf"], pre_map["mtf"], pre_l, htf, mtf, ltf, 
                    btc_close_arr=btc_close_aligned
                )

                oi_val = 0.0
                if getattr(cfg, 'use_macro_data', False):
                    try: 
                        oi_arr = fetch_delta_oi(ex, symbol, cfg.ltf, ltf["timestamp"])
                        oi_val = float(oi_arr[-2]) if len(oi_arr) > 1 else 0.0
                    except: pass

                funding_rate = ex.fetch_funding_rate(symbol)
                extras = {
                    "btc_bullish": btc_bullish,
                    "funding_rate": funding_rate,
                    "btcd_trend": btcd_trend,
                    "dxy_trend": dxy_val,
                    "spx_trend": spx_val,
                    "delta_oi": oi_val,
                    "bid_ask_imbalance": bid_ask_imbalance,
                    "macro_sentiment": oracle_sentiment_val
                }
                
                # PHASE 1: PREVENT FEATURE REPAINTING (Strict evaluation on fully closed candle)
                iH, iM, iL = len(htf)-2, len(mtf)-2, len(ltf)-2
                base = confluence_features(cfg, htf, mtf, ltf, iH, iM, iL, pre_map, extras=extras)
                base["timestamp"] = ltf.timestamp.iloc[-2]
                
                curr_bar = {"o":ltf.open.iloc[-2], "h":ltf.high.iloc[-2], "l":ltf.low.iloc[-2], "c":ltf.close.iloc[-2]}
                
                # Step 1: Manage Existing
                closed = tm.step_bar(curr_bar["o"], curr_bar["h"], curr_bar["l"], curr_bar["c"])
                if closed:
                    for t in closed:
                        log.info(f"[{symbol}] ⛔ Trade Closed: {t['side']} PnL: {t['pnl_r']:.2f}R")
                        if not cfg.paper_mode:
                            try:
                                sym = t.get('symbol', symbol)
                                executor.execute_close(sym, t['side'], t['total_size'])
                            except Exception as e:
                                log.error(f"[{symbol}] Execution Close Failed: {e}")

                # Step 2: Look for New Entries
                plan = plan_trade_with_brain(cfg, brain, base, adv_features, iH, iM, iL, pre_l)
                
                if plan:
                    unique_id = f"{plan['key']}_{base['timestamp']}_{symbol}"
                    if not mem.seen(unique_id):
                        # PATCH: DYNAMIC EXECUTION TYPE LOGGING
                        log.info(f"[{symbol}] 🚀 SIGNAL: {plan['side'].upper()} | Entry: {plan['entry']:.4f} ({plan.get('type', cfg.execution_style).upper()}) | RiskFactor={plan.get('risk_factor',1.0)}")
                        
                        send_discord_signal_alert(plan, args.discord_webhook, symbol)
                        log_to_csv(args.csv_log_path, plan)
                        
                        risk_amt = cfg.account_notional * cfg.risk_per_trade * plan.get('risk_factor', 1.0)
                        dist = abs(plan['entry'] - plan['sl'])
                        
                        if dist > 0:
                            qty = risk_amt / dist
                            
                            # Extract RVOL for Impact Calculation
                            rvol = plan['features'].get('rvol', 1.0)
                            plan_type = plan.get('type', cfg.execution_style)
                            
                            fill_price = executor.execute_entry(symbol, plan['side'], qty, plan['entry'], is_paper=cfg.paper_mode, rvol=rvol, plan_type=plan_type)
                            
                            if fill_price:
                                plan['entry'] = fill_price 
                                tm.submit_plan(plan, curr_bar)
                                
                                # IMMEDIATE STOP LOSS PLACEMENT (Real Mode Only)
                                if not cfg.paper_mode:
                                    executor.place_stop_loss(symbol, plan['side'], qty, plan['sl'])
                        else:
                            log.warning(f"[{symbol}] Risk Distance is 0. Skipping.")
                    else:
                        log.debug(f"[{symbol}] Skipping duplicate {unique_id}")
                
                else:
                    # Near Miss Logging
                    ls = _score_side(base, "long")
                    ss = _score_side(base, "short")
                    if ls >= 0.1 or ss >= 0.1:
                         adx = base.get('adx', 0); bbw = base.get('bb_width', 99)
                         if adx < 20 and bbw < 5.0:
                             log.debug(f"[{symbol}] ⚠️  CHOP FILTER ACTIVE (ADX={adx:.1f}, BBW={bbw:.1f}). Sleeping.")

        except KeyboardInterrupt:
            log.info("Manual Stop triggered.")
            return 
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            log.warning(f"⚠️ Network glitch: {e}. Sleeping 5s...")
            time.sleep(5)
        except Exception as e:
            log.error(f"CRITICAL LOOP ERROR: {e}")
            traceback.print_exc()
            time.sleep(30) 

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="ETH/USDT") 
    p.add_argument("--db-path", default="vajra.sqlite")
    p.add_argument("--csv-log-path", default="live_trades_log.csv")
    p.add_argument("--brain-long-path", required=True)
    p.add_argument("--brain-short-path", required=True)
    p.add_argument("--htf", default="12h"); p.add_argument("--mtf", default="1h"); p.add_argument("--ltf", default="15m")
    p.add_argument("--discord-webhook", default=os.getenv("DISCORD_WEBHOOK_URL"))
    args = p.parse_args()
    while True:
        try:
            run_bot(args); break 
        except Exception as e:
            log.critical(f"FATAL CRASH: {e}. Restarting...")
            time.sleep(10)