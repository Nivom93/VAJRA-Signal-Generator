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
    cfg.be_trigger_r = 0.0                 # STRICTLY DISABLED (signal generator)
    cfg.trailing_stop_trigger_r = 0.0      # STRICTLY DISABLED (signal generator)
    cfg.dynamic_tp_enabled = False         # Enforce strict structural targets
    cfg.time_in_force_decay = 24           # 24 bars (6h) for order TTL

    # Structural Geometry (relaxed for higher trade frequency)
    cfg.min_rr = 1.5           # Accept 1.5R+ setups (captures more movements)
    cfg.atr_mult_sl = 0.05     # Tight 0.05 ATR micro-buffer (setup invalidation + whisker)
    cfg.atr_mult_tp = 3.5      # Max TP cap at 3.5R

    # Risk & Concurrency
    cfg.risk_per_trade = 0.01
    cfg.max_concurrent = 6             # Allow 6 concurrent trades (capture all movements)
    cfg.min_target_dist_pct = 0.10     # Lower min target distance (more setups qualify)
    cfg.use_macro_data = True
    cfg.use_meta_labeling = True

    # Risk Controls
    cfg.max_daily_loss_r = 4.0         # Halt trading after 4R daily loss
    cfg.slippage_bps = 3.0             # Realistic 3bps slippage
    cfg.max_risk_factor = 2.0          # Cap dynamic risk scaling

    # AI Gates (relaxed — let structural logic do the filtering)
    cfg.min_ev = 0.0                   # No EV gate (structural setups pre-filtered)
    cfg.min_prob_long = 0.20           # Soft 20% AI gate for longs
    cfg.min_prob_short = 0.20          # Soft 20% AI gate for shorts
    cfg.dynamic_risk_scaling = True
