#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vajra_overrides.py — Institutional Strategy Parameters (v10.0)
==============================================================
Authentic live-market calibration for Vajra. All four execution frictions are
now active and mathematically enforced inside the engine's TradeManager:

    1. Maker / Taker fees            (bps — Bybit linear perps)
    2. Slippage                      (bps — conservative blended VWAP impact)
    3. Time-in-Force decay           (bars — forces market exit on stale setups)
    4. Widened structural stop       (ATR — survives BTC Brownian motion)

These settings MUST be identical across the backtester, the event exporter
and the live bot so that training distribution == inference distribution.
"""
from __future__ import annotations


def _strategy_overrides(cfg):
    # ── Exchange / account wiring ──────────────────────────────────────
    cfg.exchange_id = "bybit"
    cfg.market_type = "swap"
    cfg.paper_mode = True
    cfg.use_dca = False
    cfg.dca_max_safety_orders = 0

    # ── Execution physics ──────────────────────────────────────────────
    # Limit-order entries engage the live bot's L2 spoofing defense (see
    # `plan.get('type', cfg.execution_style)` in vajra_live.py). The engine
    # now dynamically inherits this value in plan_trade_with_brain, so the
    # same setting drives backtest, export and live execution.
    cfg.execution_style = "limit"
    cfg.pullback_atr_mult = 0.0

    # REAL friction — previously 0 (illusion of profitability).
    # Bybit linear perps published schedule:
    #   maker = 0.02% = 2 bps   |   taker = 0.055% ≈ 5.5 bps
    # Plus a conservative 3 bps slippage envelope per leg to account for
    # hostile book depth during news events and funding resets.
    cfg.maker_fee_bps = 2.0
    cfg.taker_fee_bps = 5.5
    cfg.slippage_bps = 3.0

    # ── Directive 4: Structural SL & Time-In-Force ─────────────────────
    # 0.40 ATR was a liquidity sweep target — widened to 1.5 ATR so the
    # stop sits beyond the typical Brownian noise envelope of BTC 15m.
    cfg.atr_mult_sl = 1.5
    cfg.atr_mult_tp = 6.0                # structural TP cap (R-multiples)
    cfg.min_rr = 2.0                     # minimum target R:R per setup

    # Hard 12h horizon on 15m (= 48 bars). If a setup has not resolved to
    # TP or SL within this window the TradeManager force-exits at the
    # current bar close with fees applied — preventing infinite alpha decay.
    cfg.time_in_force_decay = 48

    # Keep raw TP/SL semantics: no break-even, no trailing overlays so that
    # the meta-labels reflect the true signal outcome (net of fees + TIF).
    cfg.be_trigger_r = 0.0
    cfg.trailing_stop_trigger_r = 0.0
    cfg.dynamic_tp_enabled = False

    # ── Risk & Concurrency ─────────────────────────────────────────────
    cfg.risk_per_trade = 0.01
    cfg.max_concurrent = 6
    cfg.min_target_dist_pct = 0.10
    cfg.use_macro_data = True
    cfg.use_meta_labeling = True

    # ── Portfolio controls ─────────────────────────────────────────────
    cfg.max_daily_loss_r = 4.0
    cfg.max_risk_factor = 2.0

    # ── AI Gates ───────────────────────────────────────────────────────
    # SMOTE has been eradicated from the trainer. Specialist brains now
    # train on the native class balance with scale_pos_weight only, so
    # raw XGBoost outputs are properly calibrated (no extreme squashing).
    cfg.min_ev = 0.01
    cfg.min_prob_long = 0.30
    cfg.min_prob_short = 0.30
    cfg.dynamic_risk_scaling = True

    # ── Engine structural gate controls ────────────────────────────────
    cfg.filter_adx_chop = True
    cfg.adx_chop_threshold = 10.0
    cfg.reversal_evidence_min = 2
    cfg.wick_rejection_pct = 0.30
    cfg.vol_confirm_rvol = 1.2
    cfg.vol_spike_lookback = 5
    cfg.ob_freshness_bars = 64
    cfg.fvg_tolerance_atr = 0.5
    cfg.mtf_alignment_min = 1

    # ── Meta-Brain threshold (two-gate with specialist) ────────────────
    cfg.min_meta_prob = 0.22
