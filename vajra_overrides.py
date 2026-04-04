#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

def _strategy_overrides(cfg):
    cfg.exchange_id = "bybit"
    cfg.market_type = "swap"
    cfg.paper_mode = True
    cfg.use_dca = False
    cfg.dca_max_safety_orders = 0
    
    # Pure Signal Generation (No Mid-Trade Alterations)
    cfg.execution_style = 'market'
    cfg.pullback_atr_mult = 0.0
    cfg.slippage_bps = 0.0
    cfg.be_trigger_r = 0.0                 # STRICTLY DISABLED
    cfg.trailing_stop_trigger_r = 0.0      # STRICTLY DISABLED
    cfg.dynamic_tp_enabled = False         # Enforce strict structural targets
    cfg.time_in_force_decay = 0

    # Strict Structural Geometry
    cfg.min_rr = 1.8          # Engine WILL REJECT trades with < 1.8R structural distance
    cfg.atr_mult_sl = 0.0     # SL is 100% pure structure, no ATR offset
    cfg.atr_mult_tp = 3.0     # Max TP cap at 3.0R

    # Risk & Concurrency
    cfg.risk_per_trade = 0.01          
    cfg.max_concurrent = 6
    cfg.min_target_dist_pct = 0.15
    cfg.use_macro_data = True
    cfg.use_meta_labeling = True

    # AI Gates
    cfg.min_ev = 0.02
    cfg.min_prob_long = 0.35
    cfg.min_prob_short = 0.35
