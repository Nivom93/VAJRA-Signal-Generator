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

    # --- DEFENSE MECHANISMS (ACTIVATED) ---
    cfg.be_trigger_r = 1.0             # Move SL to breakeven at +1.0R
    cfg.trailing_stop_trigger_r = 1.5  # Start standard trailing at +1.5R
    cfg.trailing_dist_r = 0.5          # Trail price by 0.5R
    cfg.dynamic_tp_enabled = True      # Let the engine take profit at BB expansions
    cfg.time_in_force_decay = 24       # Kill stagnant trades after 24 bars

    # --- GLOBAL RISK DEFAULTS ---
    cfg.risk_per_trade = 0.01          
    cfg.max_concurrent = 3             
    cfg.dynamic_risk_scaling = False   
    cfg.max_risk_factor = 3.0
    
    # --- FILTERS (LET THE AI DECIDE) ---
    cfg.filter_htf_trend = True        
    cfg.filter_btc_regime = True       
    cfg.filter_funding_check = False    
    cfg.filter_btcd_regime = False      
    cfg.filter_rvol_breakout = False
    cfg.filter_adx_chop = False
    cfg.filter_time_of_day = False
    cfg.filter_hurst_strict = False
    cfg.use_macro_data = True
    cfg.use_meta_labeling = True

    # --- STRATEGY ARSENAL (TREND & MOMENTUM ONLY) ---
    cfg.strat_alpha_enabled = True 
    cfg.strat_gamma_enabled = True  
    cfg.strat_delta_enabled = True 
    cfg.strat_epsilon_enabled = True  
    cfg.strat_omega_enabled = True
    cfg.strat_zeta_enabled = True

   # ==========================================================
    # 2. COIN-SPECIFIC MATRIX (Max Win Rate)
    # ==========================================================
    symbol = cfg.symbol.upper()
    cfg.use_dca = False
    cfg.dca_max_safety_orders = 0

    # --- SURVIVAL GEOMETRY ---
    cfg.min_rr = 1.5                # Lower math constraint to prevent target stretching
    cfg.atr_mult_sl = 1.5           # Widen stops to survive MM sweeps
    cfg.atr_mult_tp = 2.5           # Natural target distance

    # --- STRICT AI GATES ---
    if "BTC" in symbol or "ETH" in symbol:
        cfg.min_conf_long = 1.0
        cfg.min_conf_short = 1.0
        cfg.min_prob_long = 0.52    # Demand true ML confidence
        cfg.min_prob_short = 0.52   # Demand true ML confidence
    else:
        cfg.min_prob_long = 0.52
        cfg.min_prob_short = 0.52
