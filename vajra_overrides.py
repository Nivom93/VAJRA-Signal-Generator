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

    # Structural Geometry (precision over frequency — win rate priority)
    cfg.min_rr = 2.2           # SL must be 2.2-3x smaller than R (high R:R with tight SL)
    cfg.atr_mult_sl = 0.05     # Tight 0.05 ATR micro-buffer (setup invalidation + whisker)
    cfg.atr_mult_tp = 3.0      # Max TP cap at 3.0R (closer targets hit more often)

    # Risk & Concurrency
    cfg.risk_per_trade = 0.01
    cfg.max_concurrent = 4             # Fewer, higher-quality trades only
    cfg.min_target_dist_pct = 0.15     # Filter out micro-targets that add noise
    cfg.use_macro_data = True
    cfg.use_meta_labeling = True

    # Risk Controls
    cfg.max_daily_loss_r = 4.0         # Halt trading after 4R daily loss
    cfg.slippage_bps = 3.0             # Realistic 3bps slippage
    cfg.max_risk_factor = 2.0          # Cap dynamic risk scaling

    # AI Gates (strict — only high-conviction setups pass)
    cfg.min_ev = 0.10                  # Require meaningful positive EV
    cfg.min_prob_long = 0.45           # 45% minimum AI confidence for longs
    cfg.min_prob_short = 0.45          # 45% minimum AI confidence for shorts
    cfg.dynamic_risk_scaling = True
