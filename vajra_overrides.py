#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

def _strategy_overrides(cfg):
    # ==========================================================
    # 1. GLOBAL BASE SETTINGS 
    # ==========================================================
    cfg.exchange_id = "bybit"
    cfg.market_type = "swap"
    cfg.htf = "1d"
    cfg.mtf = "4h"
    cfg.ltf = "1h"
    cfg.paper_mode = True
    
    # --- EXECUTION DEFAULTS (PURE LIMIT MODE) ---
    cfg.execution_style = 'limit'     
    cfg.pullback_atr_mult = 0.05        
    cfg.slippage_bps = 0.0             
    cfg.maker_fee_bps = 0.0            
    cfg.taker_fee_bps = 0.0            

    # --- DEFENSE MECHANISMS (PURE SIGNAL EDGE) ---
    cfg.be_trigger_r = 0.0             # Do not use BE tricks. Let structural TP or SL hit naturally.
    cfg.trailing_stop_trigger_r = 0.0
    cfg.trailing_dist_r = 0.0
    cfg.dynamic_tp_enabled = False
    cfg.time_in_force_decay = 72

    # --- GLOBAL RISK DEFAULTS ---
    cfg.risk_per_trade = 0.01          
    cfg.max_concurrent = 3             
    cfg.dynamic_risk_scaling = False   
    cfg.max_risk_factor = 1.0          
    
    # --- FILTERS (LET THE AI DECIDE) ---
    cfg.filter_htf_trend = True        
    cfg.filter_btc_regime = True       
    cfg.filter_funding_check = False    
    cfg.filter_btcd_regime = False      
    cfg.filter_rvol_breakout = False   # Disabled: Let ML score it
    cfg.filter_adx_chop = False        # Disabled: Let ML score it
    cfg.filter_time_of_day = False
    cfg.filter_hurst_strict = False    # Disabled: Let ML score it
    cfg.use_macro_data = True
    cfg.use_meta_labeling = True

    # --- STRATEGY ARSENAL (TREND & MOMENTUM ONLY) ---
    cfg.strat_alpha_enabled = True 
    cfg.strat_gamma_enabled = True  
    cfg.strat_delta_enabled = True 
    cfg.strat_epsilon_enabled = True  
    cfg.strat_omega_enabled = True     # Ranging environments provide the deepest structure limits for massive RR
    cfg.strat_zeta_enabled = False     

   # ==========================================================
    # 2. COIN-SPECIFIC MATRIX (Max Win Rate)
    # ==========================================================
    symbol = cfg.symbol.upper()
    cfg.use_dca = False
    cfg.dca_max_safety_orders = 0

    # --- STRUCTURAL SNIPER GEOMETRY ---
    # Target >50% WR with >2.2 RR natively via structural alignment (Order Blocks/Swings)
    cfg.min_rr = 2.2                # Rigid minimum RR enforcement
    cfg.atr_mult_sl = 1.0           # Mathematical fallback for very tight snipes
    cfg.atr_mult_tp = 2.2           # Mathematical fallback to achieve >= 2.2 RR baseline

    if "BTC" in symbol or "ETH" in symbol:
        cfg.min_conf_long = 1.0
        cfg.min_conf_short = 1.0
        # Optimal probability threshold to find ~250+ trades with highest structural confidence
        cfg.min_prob_long = 0.55
        cfg.min_prob_short = 0.55
    else:
        cfg.min_prob_long = 0.55
        cfg.min_prob_short = 0.55