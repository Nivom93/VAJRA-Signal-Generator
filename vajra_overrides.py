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

    # Strict Structural Geometry
    cfg.min_rr = 2.0           # Minimum 2R structural distance (god-tier selectivity)
    cfg.atr_mult_sl = 0.05     # Tight 0.05 ATR micro-buffer (setup invalidation + whisker)
    cfg.atr_mult_tp = 3.5      # Max TP cap at 3.5R

    # Risk & Concurrency
    cfg.risk_per_trade = 0.01
    cfg.max_concurrent = 4             # Reduce from 6 to 4 (quality over quantity)
    cfg.min_target_dist_pct = 0.15
    cfg.use_macro_data = True
    cfg.use_meta_labeling = True

    # Risk Controls
    cfg.max_daily_loss_r = 4.0         # Halt trading after 4R daily loss
    cfg.slippage_bps = 3.0             # Realistic 3bps slippage
    cfg.max_risk_factor = 2.0          # Cap dynamic risk scaling

    # AI Gates (higher selectivity)
    cfg.min_ev = 0.10                  # Minimum 0.10R expected value
    cfg.min_prob_long = 0.35           # Require 35%+ AI probability for longs
    cfg.min_prob_short = 0.35          # Require 35%+ AI probability for shorts
    cfg.dynamic_risk_scaling = True
