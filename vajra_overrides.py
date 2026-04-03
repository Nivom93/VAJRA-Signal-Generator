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
    cfg.use_dca = False
    cfg.dca_max_safety_orders = 0
    
    # --- EXECUTION DEFAULTS (PURE LIMIT MODE) ---
    cfg.execution_style = 'market'
    cfg.pullback_atr_mult = 0.05        
    cfg.slippage_bps = 0.0
    cfg.maker_fee_bps = 0.0
    cfg.taker_fee_bps = 0.0

    # --- DEFENSE MECHANISMS (WIN-RATE ENHANCEMENTS) ---
    cfg.be_trigger_r = 0.0
    cfg.trailing_stop_trigger_r = 0.0
    cfg.trailing_dist_r = 0.0
    cfg.dynamic_tp_enabled = True  # <--- MUST BE TRUE
    cfg.time_in_force_decay = 0
    cfg.maker_fee_bps = 0.0
    cfg.taker_fee_bps = 0.0

    # --- GLOBAL RISK DEFAULTS ---
    cfg.risk_per_trade = 0.01          
    cfg.max_concurrent = 6
    cfg.dynamic_risk_scaling = True
    cfg.max_risk_factor = 2.5
    cfg.min_ev = 0.08
    cfg.min_target_dist_pct = 0.15
    
    # --- FILTERS (LET THE AI DECIDE) ---
    cfg.filter_htf_trend = False
    cfg.filter_btc_regime = False
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
    cfg.strat_omega_enabled = True     # Ranging environments provide the deepest structure limits for massive RR
    cfg.strat_zeta_enabled = True

   # ==========================================================
    # 2. COIN-SPECIFIC MATRIX (Max Win Rate)
    # ==========================================================
    symbol = cfg.symbol.upper()
    cfg.use_dca = False
    cfg.dca_max_safety_orders = 0

    # --- STRUCTURAL SNIPER GEOMETRY ---
    cfg.min_rr = 2.0          # Engine will reject ANY structural target below 2.0R to align with market structure
    cfg.atr_mult_sl = 1.0     # Anchors risk exactly 1.0 ATR beyond the structural swing
    cfg.atr_mult_tp = 2.5     # Hard cap greed at exactly 3.0R to protect win-rate

    # --- STRICT AI GATES ---
    pass
