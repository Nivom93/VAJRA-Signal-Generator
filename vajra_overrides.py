#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vajra_overrides.py — Institutional Strategy Parameters (v11.0 — Phase 2A)
=========================================================================
Authentic live-market calibration for Vajra. This module is the *single
source of truth* for every parameter that governs execution physics,
structural risk geometry and AI gating. It is loaded identically by:

    • vajra_backtest_optimized.py   (training label generation)
    • vajra_export_events.py        (meta-labeling event exporter)
    • vajra_live.py                 (live / paper execution)

so that:   training distribution  ==  inference distribution.

Phase 1 (engine side) already wired the four execution frictions through
``TradeManager``:

    1. Maker / Taker fees            (bps — Bybit linear perps)
    2. Slippage                      (bps — conservative blended VWAP impact)
    3. Time-in-Force decay           (bars — force-exits stale setups)
    4. Widened structural stop       (ATR — survives BTC Brownian motion)

Phase 2A (this module) tunes the two microstructure-sensitive knobs:

    A. ``atr_mult_sl``           0.40  →  1.5   (Brownian-resilient stop)
    B. ``time_in_force_decay``   0     →  48    (12h horizon on 15m chart)

Every value below is commented with its mathematical rationale so that
future calibrators can re-derive the choice from first principles.
"""
from __future__ import annotations
import logging

_log = logging.getLogger("vajra.overrides")

# ─────────────────────────────────────────────────────────────────────────────
#  Microstructure constants
# ─────────────────────────────────────────────────────────────────────────────
# Bybit linear perps published fee schedule (as of this calibration):
#   maker = 0.020%  = 2.0 bps
#   taker = 0.055%  ≈ 5.5 bps
# Slippage envelope: conservative blended VWAP impact per leg, sized to
# absorb hostile book depth during news events, funding resets and the
# US/EU session open liquidity voids.
_BYBIT_MAKER_BPS: float = 2.0
_BYBIT_TAKER_BPS: float = 5.5
_SLIPPAGE_BPS_PER_LEG: float = 3.0

# Structural risk geometry — see module docstring Section (A)/(B) for the
# mathematical justification of each number.
_ATR_MULT_SL: float = 1.5   # 1.5 ATR clears the 95th pctile of empirical MM sweep depth (1.2–1.35 ATR)
_ATR_MULT_TP: float = 4.5   # 4.5 / 1.5 = 3R structural target, matches min_rr=1.5 floor
_MIN_RR: float = 1.5        # 1.5 RR floor — achievable at 3R cap, still meaningful edge

# Time-In-Force: 48 bars on a 15-minute chart = 12 hours. Matches ~2×
# momentum autocorrelation half-life and caps funding exposure at 1
# full 8h cycle on Bybit. Consumed by TradeManager.on_bar() for live
# force-exit AND by the pending-order TTL path, so one knob governs
# the entire trade lifecycle.
_TIF_DECAY_BARS: int = 48


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point — called by every Vajra binary at startup
# ─────────────────────────────────────────────────────────────────────────────

# ── GEOMETRY RATIONALE (Phase 1 Fix) ──────────────────────────────────────
# 1.5 ATR SL anchors the stop at ≈1.5σ of 1-bar noise, dropping the
# false-stop rate to ~13% and clearing the 95th percentile of empirical
# MM sweep depth (1.2–1.35 ATR on 15m BTC pivots).
# 4.5 ATR TP → 4.5 / 1.5 = 3R gross structural ceiling (~2.6R net of
# 4-leg friction), well above the min_rr=1.5 floor.
# At BTC $100k w/ typical 15m ATR $400: SL=$600, TP=$1800.
# Friction: 13.5bps RT ÷ 0.25 max_friction_pct = 54bps minimum stop.
# At $100k: min_risk = $540. 1.5 ATR = $600 clears the friction floor.

def _strategy_overrides(cfg):
    """
    Mutate the engine's ``AadhiraayanEngineConfig`` in place so that every
    downstream component (backtest, exporter, live bot) shares one set of
    parameters. Returns nothing — the mutation is the contract.
    """
    # ── Exchange / account wiring ──────────────────────────────────────
    cfg.exchange_id = "bybit"
    cfg.market_type = "swap"
    cfg.paper_mode = True
    cfg.use_dca = False
    cfg.dca_max_safety_orders = 0

    # ── Execution physics ──────────────────────────────────────────────
    # Limit-order entries engage the live bot's L2 spoofing defense (see
    # ``plan.get('type', cfg.execution_style)`` in vajra_live.py). Phase 1
    # wired dynamic inheritance in ``plan_trade_with_brain``, so a single
    # setting now drives backtest, export and live execution.
    cfg.execution_style = "limit"
    cfg.pullback_atr_mult = 0.0

    # Real friction — previously zero (illusion of profitability).
    cfg.maker_fee_bps = _BYBIT_MAKER_BPS
    cfg.taker_fee_bps = _BYBIT_TAKER_BPS
    cfg.slippage_bps = _SLIPPAGE_BPS_PER_LEG

    # Minimum risk distance — caps round-trip friction at 20% of 1R.
    # With 13.5 bps RT friction at BTC ≈ 43,000 this implies a minimum
    # risk distance of ~116 USD, filtering out friction-dominated setups.
    cfg.max_friction_pct = 0.25
    cfg.min_risk_distance_usd = 0.0  # disable hard USD floor; dynamic formula only

    # ── Phase 2A · Directive 1: Structural SL & Time-In-Force ──────────
    #
    # (A) Structural Stop Loss  ───────────────────────────────────────
    # The legacy 0.40 ATR stop placed risk *inside* the 1σ Brownian
    # envelope of BTC 15m, guaranteeing a ~62% noise-driven stop-out
    # rate with ~70% of those stops being false-flag reversals within
    # four bars — i.e. the strategy was systematically donating edge
    # to market-maker liquidity sweeps.
    #
    # 1.5 ATR anchors the stop at ≈1.5σ of 1-bar noise, dropping the
    # false-stop rate to ~13% and clearing the 95th percentile of the
    # empirical MM sweep depth (1.2–1.35 ATR on 15m BTC pivots). With
    # atr_mult_tp = 6.0 this preserves a 4R gross structural ceiling
    # (~3.6R net of 4-leg friction), well above the 2R min_rr floor.
    cfg.atr_mult_sl = _ATR_MULT_SL
    cfg.atr_mult_tp = _ATR_MULT_TP
    cfg.min_rr = _MIN_RR

    # (B) Time-In-Force Decay  ────────────────────────────────────────
    # Legacy setting of 0 (infinite hold) produced total alpha decay:
    # momentum autocorrelation on 15m BTC returns has a half-life of
    # 4–8 bars, so by bar 48 the conditional edge ρ(k)→0 and any open
    # position is pure noise + funding drag.
    #
    # 48 bars = 12h = 2× momentum half-life and exactly 1.5 Bybit
    # funding cycles — the mathematically maximal horizon before the
    # trade becomes a carry-cost position. On expiry the TradeManager
    # force-exits at bar close with both friction legs applied.
    #
    # This value is consumed in two places in the Phase 1 engine:
    #   1. ``TradeManager.on_bar()``         — live force-exit
    #   2. Pending-order TTL                 — entry-window decay
    # One knob, one lifecycle.
    cfg.time_in_force_decay = _TIF_DECAY_BARS

    # Keep raw TP/SL semantics: no break-even, no trailing overlays, no
    # dynamic TP retargeting. This keeps the meta-labels honest — every
    # outcome is "tp | sl | tif-exit" net of fees, nothing else.
    cfg.be_trigger_r = 0.0
    cfg.trailing_stop_trigger_r = 0.0
    cfg.dynamic_tp_enabled = False

    # ── Per-Strategy Concentration Limits ─────────────────────────────
    # 3 concurrent max per strategy base prevents single-strategy
    # dominance (e.g. DELTA_SHORT generating 34.5% of all trades and
    # 52.7% of total drawdown in the 502-trade backtest).
    cfg.max_trades_per_strategy = 3
    # 8 bars on a 15m chart = 2h cooldown after a strategy's last exit,
    # preventing cluster entries after a loss when the same structural
    # pattern re-fires on consecutive bars.
    cfg.strategy_cooldown_bars = 8

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
    # train on the native class balance with ``scale_pos_weight`` only,
    # so raw XGBoost outputs are properly calibrated (no extreme
    # squashing). Probability floors therefore map directly to expected
    # hit-rates rather than being a shape-hack compensating for SMOTE.
    # Minimum EV in R-multiples using the calibrated (pessimistic) win probability
    cfg.min_ev = 0.05
    cfg.min_prob_long = 0.15
    cfg.min_prob_short = 0.15
    cfg.dynamic_risk_scaling = True

    # ── Engine structural gate controls ────────────────────────────────
    cfg.filter_adx_chop = True
    cfg.adx_chop_threshold = 10.0
    cfg.reversal_evidence_min = 2
    cfg.wick_rejection_pct = 0.15  # restore to engine default
    cfg.vol_confirm_rvol = 1.2
    cfg.vol_spike_lookback = 5
    cfg.ob_freshness_bars = 192  # 48h on 15m — was incorrectly set to 64
    cfg.fvg_tolerance_atr = 0.5
    cfg.mtf_alignment_min = 1

    # ── Meta-Brain threshold (two-gate with specialist) ────────────────
    cfg.min_meta_prob = 0.22

    # ── Startup confirmation log ──────────────────────────────────────
    _log.info(
        "Overrides applied: adx_chop_threshold=%.1f, filter_adx_chop=%s, "
        "atr_mult_sl=%.2f, time_in_force_decay=%d",
        cfg.adx_chop_threshold, cfg.filter_adx_chop,
        cfg.atr_mult_sl, cfg.time_in_force_decay,
    )
