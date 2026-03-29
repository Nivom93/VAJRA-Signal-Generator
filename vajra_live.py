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

class MacroFetcher(threading.Thread):
    """
    Background daemon that asynchronously fetches slow macro data (yfinance)
    to prevent main execution loop freezing.
    """
    def __init__(self, cfg):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.state = {"dxy_val": 0.0, "spx_val": 0.0}

    def run(self):
        log.info("🌐 Macro Fetcher Daemon Active.")
        while True:
            if getattr(self.cfg, 'use_macro_data', False):
                try:
                    import yfinance as yf
                    for t in ["DX-Y.NYB", "UUP", "DX=F"]:
                        try:
                            dxy_c = yf.Ticker(t).history(period="5d")['Close']
                            if not dxy_c.empty:
                                self.state["dxy_val"] = float(dxy_c.iloc[-2]) if len(dxy_c) > 1 else 0.0
                                break
                        except Exception:
                            continue

                    for t in ["^GSPC", "SPY", "ES=F"]:
                        try:
                            spx_c = yf.Ticker(t).history(period="5d")['Close']
                            if not spx_c.empty:
                                self.state["spx_val"] = float(spx_c.iloc[-2]) if len(spx_c) > 1 else 0.0
                                break
                        except Exception:
                            continue
                except Exception as e:
                    log.debug(f"Macro Fetcher silent fail: {e}")
            time.sleep(300) # Update every 5 minutes

class AutoRetrainer(threading.Thread):
    """
    Background sentinel that monitors model files for updates and 
    periodically triggers retraining if data is stale.
    """
    def __init__(self, brain_manager, brains_dir, interval_hours=12):
        super().__init__(daemon=True)
        self.brain_manager = brain_manager
        self.brains_dir = brains_dir
        self.interval = interval_hours * 3600
        self.last_mtimes = {}
        self._update_mtimes()

    def _update_mtimes(self):
        if not self.brains_dir: return
        p = Path(self.brains_dir)
        if p.is_dir():
            for f in p.glob("brain_*.joblib"):
                self.last_mtimes[str(f)] = os.path.getmtime(f)

    def run(self):
        log.info("🛡️ Auto-Retrainer Sentinel Active.")
        while True:
            time.sleep(60) # Check every minute for file changes, trigger retrain logic every interval
            
            if not self.brains_dir: continue

            # 1. Hot Reload Check
            p = Path(self.brains_dir)
            needs_reload = False
            if p.is_dir():
                for f in p.glob("brain_*.joblib"):
                    curr_mtime = os.path.getmtime(f)
                    if curr_mtime > self.last_mtimes.get(str(f), 0):
                        log.info(f"🧠 Brain update detected: {f.name}. Hot reloading...")
                        self.last_mtimes[str(f)] = curr_mtime
                        needs_reload = True

            if needs_reload:
                self.brain_manager.load_brains(self.brains_dir)
                log.info("✅ Brain Hot Reload Complete.")

# --- EXECUTION MANAGER ---

class RealExecutionManager:
    """
    Handles PAPER execution only for pure Signal Generation.
    Implements Quadratic Impact simulation.
    """
    def __init__(self, client, cfg):
        self.client = client
        self.cfg = cfg
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

    def execute_entry(self, symbol, side, qty, price, is_paper=True, rvol=1.0, plan_type='limit', sl=None, tp=None):
        """
        Executes an entry order.
        LEVEL 14: Quadratic Impact Simulation for Paper Mode.
        """
        c_side = 'buy' if side == 'long' else 'sell'
        
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

def _fetch_macro_context(ex, global_cache, cfg):
    btc_bullish = 1.0
    btcd_trend = 0.0
    oracle_sentiment_val = 0.0

    try:
        new_btc_4h = ex.fetch_ohlcv_df("BTC/USDT", "4h", limit=5)
        if global_cache["btc_4h"].empty:
            global_cache["btc_4h"] = new_btc_4h
        else:
            global_cache["btc_4h"] = pd.concat([global_cache["btc_4h"], new_btc_4h]).drop_duplicates(subset=["timestamp"], keep="last").tail(1000).reset_index(drop=True)
        btc_ohlcv = global_cache["btc_4h"]

        if not btc_ohlcv.empty:
            btc_c = btc_ohlcv.close.values
            current_price = btc_c[-1]
            btc_ema100 = _ema_np(btc_c, 100)[-1]
            btc_bullish = 1.0 if current_price > btc_ema100 else 0.0
            dist_pct = (current_price - btc_ema100) / btc_ema100 * 100
            log.info(f"BTC Context (4H): Bullish={btc_bullish} (Dist={dist_pct:.2f}%)")

        new_btc_exec = ex.fetch_ohlcv_df("BTC/USDT", cfg.exec_tf, limit=5)
        if global_cache["btc_exec_tf"].empty:
            global_cache["btc_exec_tf"] = new_btc_exec
        else:
            global_cache["btc_exec_tf"] = pd.concat([global_cache["btc_exec_tf"], new_btc_exec]).drop_duplicates(subset=["timestamp"], keep="last").tail(500).reset_index(drop=True)

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

    except Exception as e:
        log.warning(f"Macro Sync Failed: {e}")

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

    return btc_bullish, btcd_trend, 0.0, 0.0, oracle_sentiment_val

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
    brain = BrainLearningManager(cfg, args.brains_dir)
    tm = TradeManager(cfg, ex, mem, brain)
    
    # 3. Start Auto-Retrainer Sentinel
    retrainer = AutoRetrainer(brain, args.brains_dir)
    retrainer.start()

    # 4. Start Macro Fetcher Daemon
    macro_fetcher = MacroFetcher(cfg)
    macro_fetcher.start()

    # PHASE 2: IN-MEMORY OHLCV CACHING SYSTEM (Prevents API ratelimits)
    log.info("Building In-Memory OHLCV Cache to prevent IP Bans...")
    market_cache = {}
    for sym in cfg.symbols:
        market_cache[sym] = {
            "macro_tf": ex.fetch_ohlcv_df(sym, cfg.macro_tf, limit=250),
            "swing_tf": ex.fetch_ohlcv_df(sym, cfg.swing_tf, limit=250),
            "htf": ex.fetch_ohlcv_df(sym, cfg.htf, limit=250),
            "exec_tf": ex.fetch_ohlcv_df(sym, cfg.exec_tf, limit=500)
        }
    
    global_cache = {
        "btc_4h": ex.fetch_ohlcv_df("BTC/USDT", "4h", limit=1000),
        "btc_exec_tf": ex.fetch_ohlcv_df("BTC/USDT", cfg.exec_tf, limit=500)
    }
    try:
        global_cache["btcd_4h"] = ex.fetch_ohlcv_df("BTCDOM/USDT", "4h", limit=50)
    except:
        global_cache["btcd_4h"] = pd.DataFrame()
        
    log.info("✅ Cache built successfully.")

    log.info(f"✅ ENGINE v35 LIVE on {cfg.symbols}. Protocol: Liquidity Warlord.")
    mode_str = "PAPER (Simulation)" if cfg.paper_mode else "REAL MONEY"
    log.info(f"EXECUTION MODE: {mode_str} | ENTRY STYLE: {cfg.execution_style}")

    btc_bullish = 1.0
    btcd_trend = 0.0 # Default Neutral
    
    last_processed_ts = {}

    while True:
        try:
            now_sec = time.time()
            sleep_sec = 60 - (now_sec % 60)
            log.info(f"Waiting {sleep_sec:.1f}s for next candle...")
            time.sleep(sleep_sec + 1)

            btc_bullish, btcd_trend, _, _, oracle_sentiment_val = _fetch_macro_context(ex, global_cache, cfg)
            dxy_val = macro_fetcher.state["dxy_val"]
            spx_val = macro_fetcher.state["spx_val"]
            btc_exec_tf = global_cache.get("btc_exec_tf", pd.DataFrame())

            # --- MULTI-SYMBOL SCANNING LOOP ---
            for symbol in cfg.symbols:
                cfg.symbol = symbol 
                
                # Fetch incremental update and apply to cache (Protect Rate Limits)
                new_macro = ex.fetch_ohlcv_df(symbol, cfg.macro_tf, limit=5)
                market_cache[symbol]["macro_tf"] = pd.concat([market_cache[symbol]["macro_tf"], new_macro]).drop_duplicates(subset=["timestamp"], keep="last").tail(250).reset_index(drop=True)
                macro_tf = market_cache[symbol]["macro_tf"]

                new_swing = ex.fetch_ohlcv_df(symbol, cfg.swing_tf, limit=5)
                market_cache[symbol]["swing_tf"] = pd.concat([market_cache[symbol]["swing_tf"], new_swing]).drop_duplicates(subset=["timestamp"], keep="last").tail(250).reset_index(drop=True)
                swing_tf = market_cache[symbol]["swing_tf"]

                new_htf = ex.fetch_ohlcv_df(symbol, cfg.htf, limit=5)
                market_cache[symbol]["htf"] = pd.concat([market_cache[symbol]["htf"], new_htf]).drop_duplicates(subset=["timestamp"], keep="last").tail(250).reset_index(drop=True)
                htf = market_cache[symbol]["htf"]

                new_exec = ex.fetch_ohlcv_df(symbol, cfg.exec_tf, limit=5)
                market_cache[symbol]["exec_tf"] = pd.concat([market_cache[symbol]["exec_tf"], new_exec]).drop_duplicates(subset=["timestamp"], keep="last").tail(500).reset_index(drop=True)
                exec_tf = market_cache[symbol]["exec_tf"]
                
                if exec_tf.empty or len(macro_tf) < 2 or len(swing_tf) < 2 or len(htf) < 2 or len(exec_tf) < 2: continue
                
                # FIX 1: THE GHOST CANDLE TIME-DILATION BUG
                curr_ts = int(exec_tf.timestamp.iloc[-2])
                if curr_ts <= last_processed_ts.get(symbol, 0):
                    continue
                last_processed_ts[symbol] = curr_ts

                # LEVEL 30 SPOOFING DEFENSE (L2 ORDERBOOK FETCH)
                bid_ask_imbalance = 0.5
                try:
                    ob = ex.client.fetch_order_book(symbol, limit=20)
                    if ob and 'bids' in ob and 'asks' in ob:
                        total_bids = sum((v for p, v in ob['bids']))
                        total_asks = sum((v for p, v in ob['asks']))
                        if (total_bids + total_asks) > 0:
                            bid_ask_imbalance = total_bids / (total_bids + total_asks)
                except Exception as e:
                    log.debug(f"[{symbol}] L2 Orderbook fetch failed: {e}")

                pMacro, pSwing, pHtf, pExec = Precomp(macro_tf), Precomp(swing_tf), Precomp(htf), Precomp(exec_tf)
                pre_map = {"macro_tf": pMacro, "swing_tf": pSwing, "htf": pHtf, "exec_tf": pExec}
                
                btc_close_aligned = None
                if not btc_exec_tf.empty and not exec_tf.empty:
                    btc_s_close = pd.Series(btc_exec_tf.close.values, index=btc_exec_tf["timestamp"])
                    btc_close_aligned = btc_s_close.reindex(exec_tf["timestamp"], method='ffill').fillna(0.0).values

                adv_features = precompute_v6_features(
                    pMacro, pSwing, pExec, macro_tf, swing_tf, exec_tf,
                    btc_close_arr=btc_close_aligned
                )

                oi_val = 0.0
                if getattr(cfg, 'use_macro_data', False):
                    try: 
                        oi_arr = fetch_delta_oi(ex, symbol, cfg.exec_tf, exec_tf["timestamp"])
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
                # For Live Bot, [-1] is the open unclosed candle, [-2] is the completely closed one.
                # So the completely closed timestamp is ts of [-2].
                ts_closed = int(exec_tf.timestamp.iloc[-2])

                iMacro = np.searchsorted(macro_tf['timestamp'].values, ts_closed, side='right') - 1
                iSwing = np.searchsorted(swing_tf['timestamp'].values, ts_closed, side='right') - 1
                iHtf = np.searchsorted(htf['timestamp'].values, ts_closed, side='right') - 1
                iExec = len(exec_tf)-2

                base = confluence_features(cfg, macro_tf, swing_tf, htf, exec_tf, max(0, iMacro-1), max(0, iSwing-1), max(0, iHtf-1), iExec, pre_map, extras=extras)
                base["timestamp"] = exec_tf.timestamp.iloc[-2]
                base["symbol"] = symbol
                
                curr_bar = {"o":exec_tf.open.iloc[-2], "h":exec_tf.high.iloc[-2], "l":exec_tf.low.iloc[-2], "c":exec_tf.close.iloc[-2]}
                
                # Step 1: Manage Existing
                # FIX 7: STRUCTURAL TRAILING STOP PARALYSIS (Pass swing_high/swing_low)
                sh = pExec.last_sh[iExec] if iExec < len(pExec.last_sh) else 0.0
                sl = pExec.last_sl[iExec] if iExec < len(pExec.last_sl) else 0.0
                closed = tm.step_bar(symbol, curr_bar["o"], curr_bar["h"], curr_bar["l"], curr_bar["c"], ts=curr_ts, swing_high=sh, swing_low=sl)

                if closed:
                    for t in closed:
                        log.info(f"[{symbol}] ⛔ Trade Closed: {t['side']} PnL: {t['pnl_r']:.2f}R")

                # Step 2: Look for New Entries
                plan = plan_trade_with_brain(cfg, brain, base, adv_features, iExec, pExec)
                
                if plan:
                    # SPOOFING DEFENSE: Block limit orders if facing a massive spoofed wall
                    if plan.get("type", cfg.execution_style) == 'limit':
                        if plan['side'] == 'long' and bid_ask_imbalance < 0.25:
                            log.warning(f"[{symbol}] 🛡️ SPOOFING DEFENSE: Massive Ask Wall detected (Imbalance {bid_ask_imbalance:.2f}). Blocking LONG limit order.")
                            continue
                        elif plan['side'] == 'short' and bid_ask_imbalance > 0.75:
                            log.warning(f"[{symbol}] 🛡️ SPOOFING DEFENSE: Massive Bid Wall detected (Imbalance {bid_ask_imbalance:.2f}). Blocking SHORT limit order.")
                            continue

                    unique_id = f"{plan['key']}_{base['timestamp']}_{symbol}"
                    if not mem.seen(unique_id):
                        # PATCH: DYNAMIC EXECUTION TYPE LOGGING
                        log.info(f"[{symbol}] 🚀 SIGNAL: {plan['side'].upper()} | Entry: {plan['entry']:.4f} ({plan.get('type', cfg.execution_style).upper()}) | RiskFactor={plan.get('risk_factor',1.0)}")
                        
                        send_discord_signal_alert(plan, args.discord_webhook, symbol)
                        log_to_csv(args.csv_log_path, plan)
                        
                        risk_amt = cfg.account_notional * cfg.risk_per_trade * plan.get('risk_factor', 1.0)
                        dist = abs(plan['entry'] - plan['sl'])
                        
                        if dist > 0:
                            max_notional = cfg.account_notional * 10.0 # Hard 10x leverage cap
                            raw_qty = risk_amt / dist
                            qty = min(raw_qty, max_notional / plan['entry'])
                            
                            # Extract RVOL for Impact Calculation
                            rvol = plan['features'].get('rvol', 1.0)
                            plan_type = plan.get('type', cfg.execution_style)
                            
                            fill_price = executor.execute_entry(symbol, plan['side'], qty, plan['entry'], is_paper=cfg.paper_mode, rvol=rvol, plan_type=plan_type)
                            
                            if fill_price:
                                plan['entry'] = fill_price 

                                # Force Open instantly using the updated TradeManager args
                                tm.submit_plan(plan, curr_bar, force_open=True, fill_price=fill_price, sl_order_id=None)
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
    p.add_argument("--brains-dir", required=True, help="Directory containing localized strategy brains")
    p.add_argument("--macro-tf", default="1d")
    p.add_argument("--swing-tf", default="4h")
    p.add_argument("--htf", default="1h")
    p.add_argument("--exec-tf", default="15m")
    p.add_argument("--discord-webhook", default=os.getenv("DISCORD_WEBHOOK_URL"))
    args = p.parse_args()
    while True:
        try:
            run_bot(args); break 
        except Exception as e:
            log.critical(f"FATAL CRASH: {e}. Restarting...")
            time.sleep(10)