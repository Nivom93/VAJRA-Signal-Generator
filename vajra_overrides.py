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
    cfg.be_trigger_r = 1.0                 # Break-even at 1R to protect capital
    cfg.trailing_stop_trigger_r = 1.5      # Trail after 1.5R to lock profits
    cfg.dynamic_tp_enabled = False         # Enforce strict structural targets
    cfg.time_in_force_decay = 96           # 96 bars (24h on 15m) — setups need time to develop

    # Structural Geometry (balanced precision + frequency)
    cfg.min_rr = 1.5            # 1.5R minimum — captures many valid setups missed at 2.2
    cfg.atr_mult_sl = 0.20      # 0.20 ATR buffer — enough room for normal wicks
    cfg.atr_mult_tp = 4.0       # Max TP cap at 4.0R — let winners run further

    # Risk & Concurrency
    cfg.risk_per_trade = 0.01
    cfg.max_concurrent = 6              # Allow more concurrent trades with diversified setups
    cfg.min_target_dist_pct = 0.10      # Slightly lower noise filter (was 0.15)
    cfg.use_macro_data = True
    cfg.use_meta_labeling = True

    # Risk Controls
    cfg.max_daily_loss_r = 4.0          # Halt trading after 4R daily loss
    cfg.slippage_bps = 3.0              # Realistic 3bps slippage
    cfg.max_risk_factor = 2.0           # Cap dynamic risk scaling

    # AI Gates (relaxed — let the brain see more setups to learn patterns)
    cfg.min_ev = 0.05                   # Lower EV floor (was 0.10)
    cfg.min_prob_long = 0.40            # 40% minimum AI confidence for longs (was 0.45)
    cfg.min_prob_short = 0.40           # 40% minimum AI confidence for shorts (was 0.45)
    cfg.dynamic_risk_scaling = True

    # ── NEW: Engine Gate Controls (override hardcoded gates) ──
    cfg.filter_adx_chop = True          # Enable ADX filter but with lower threshold
    cfg.adx_chop_threshold = 10.0       # ADX < 10 = no direction (was hardcoded 15)
    cfg.reversal_evidence_min = 2       # 2/5 reversal signals to unlock counter-trend (was 4)
    cfg.wick_rejection_pct = 0.15       # 15% wick ratio threshold (was hardcoded 0.25)
    cfg.vol_confirm_rvol = 0.8          # rvol > 0.8 for volume confirmation (was 1.2)
    cfg.vol_spike_lookback = 5          # 5-bar lookback for vol spikes (was 3)
    cfg.ob_freshness_bars = 192         # OB valid for 192 bars / 48h on 15m (was 40)
    cfg.fvg_tolerance_atr = 0.5         # FVG zone tolerance = 0.5 ATR (was 0.2% of price)
    cfg.mtf_alignment_min = 1           # Only 1 TF agreement needed for trend trades (was 2)

    # Meta-Brain threshold (separate from specialist min_prob)
    # Meta-Brain outputs lower probabilities due to unbalanced base rate (~22% positive).
    # This threshold is its go/no-go gate — trades must also clear specialist min_prob above.
    cfg.min_meta_prob = 0.22
