#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

def _strategy_overrides(cfg):
    cfg.exchange_id = "bybit"
    cfg.market_type = "swap"
    cfg.paper_mode = True
    cfg.use_dca = False
    cfg.dca_max_safety_orders = 0

    # Pure Signal Generation — raw TP/SL outcome only.
    # No trade management overlays (BE, trailing, time decay) so we measure
    # true signal quality.  MUST match vajra_export_events.py for train/test parity.
    cfg.execution_style = 'market'
    cfg.pullback_atr_mult = 0.0
    cfg.be_trigger_r = 0.0                 # No break-even — raw signal outcome
    cfg.trailing_stop_trigger_r = 0.0      # No trailing — raw signal outcome
    cfg.dynamic_tp_enabled = False         # Enforce strict structural targets
    cfg.time_in_force_decay = 0            # No time decay — let signal resolve to TP or SL

    # Structural Geometry — calibrated for BTC 15m volatility
    cfg.min_rr = 2.0            # 2.0R minimum — balanced between frequency and expectancy
    cfg.atr_mult_sl = 0.40      # 0.40 ATR buffer — crypto wicks routinely exceed 0.20
    cfg.atr_mult_tp = 6.0       # Max TP cap at 6.0R — allow realistic structural targets

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

    # AI Gates — calibrated for SMOTE(0.4) + scale_pos_weight training
    # SMOTE at 0.4 ratio only enriches minority to ~29%, so specialist outputs
    # cluster in 0.15-0.45 range. Thresholds must match this distribution.
    cfg.min_ev = 0.01                   # Low EV floor — let blended probs through
    cfg.min_prob_long = 0.30            # 30% specialist threshold for longs
    cfg.min_prob_short = 0.30           # 30% specialist threshold for shorts
    cfg.dynamic_risk_scaling = True

    # ── NEW: Engine Gate Controls (override hardcoded gates) ──
    cfg.filter_adx_chop = True          # Enable ADX filter but with lower threshold
    cfg.adx_chop_threshold = 10.0       # ADX < 10 = no direction (was hardcoded 15)
    cfg.reversal_evidence_min = 2       # 2/5 reversal signals to unlock counter-trend (was 4)
    cfg.wick_rejection_pct = 0.30       # 30% wick ratio — real rejections, not noise (was 0.15)
    cfg.vol_confirm_rvol = 1.2          # rvol > 1.2 for volume confirmation (was 0.8)
    cfg.vol_spike_lookback = 5          # 5-bar lookback for vol spikes (was 3)
    cfg.ob_freshness_bars = 64          # OB valid for 64 bars / 16h on 15m (was 192 / 48h)
    cfg.fvg_tolerance_atr = 0.5         # FVG zone tolerance = 0.5 ATR (was 0.2% of price)
    cfg.mtf_alignment_min = 1           # Only 1 TF agreement needed for trend trades (was 2)

    # Meta-Brain threshold (separate from specialist min_prob)
    # Meta-Brain outputs lower probabilities due to unbalanced base rate (~22% positive).
    # This threshold is its go/no-go gate — trades must also clear specialist min_prob above.
    cfg.min_meta_prob = 0.22
