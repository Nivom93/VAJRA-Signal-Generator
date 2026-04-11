#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vajra_overrides.py — Institutional Strategy Parameters (v11.0)
==============================================================
PHASE 2 LOCK-IN:  Market-microstructure-realistic stops + strict TIF decay.

Every parameter below is wired into the Phase 1 friction physics that live
inside ``vajra_engine_ultra_v6_final.TradeManager``. The engine deducts the
round-trip maker/taker fee + slippage from the raw directional delta BEFORE
dividing by the per-unit risk distance, so the ``pnl_r`` reported by the
backtester, exporter and live bot is the TRUE NET R-multiple.

The four market frictions that must be in perfect alignment between the
training distribution and the inference distribution:

    1. Maker / Taker fees           (bps — Bybit linear perps)
    2. Slippage                     (bps — conservative blended VWAP impact)
    3. Time-In-Force decay          (bars — forces market exit on stale setups)
    4. Widened structural stop      (ATR — survives BTC Brownian noise envelope)

If ANY of the above drifts between the exporter, backtester and live bot the
meta-brain is trained on a distribution it will never see in production and
the whole edge collapses. This file is the single source of truth.
"""
from __future__ import annotations


def _strategy_overrides(cfg):
    # ══════════════════════════════════════════════════════════════════
    # EXCHANGE / ACCOUNT WIRING
    # ══════════════════════════════════════════════════════════════════
    cfg.exchange_id = "bybit"
    cfg.market_type = "swap"
    cfg.paper_mode = True
    cfg.use_dca = False
    cfg.dca_max_safety_orders = 0

    # ══════════════════════════════════════════════════════════════════
    # EXECUTION PHYSICS — ROUTER + FEE + SLIPPAGE
    # ══════════════════════════════════════════════════════════════════
    # Limit-order entries engage the live bot's L2 spoofing defense
    # (vajra_live.py:593/604/619/625 all read plan.get('type', cfg.execution_style)).
    # Phase 1 Directive 2 guarantees plan_trade_with_brain inherits this via a
    # single safe getattr() so the same setting drives exporter, backtester AND
    # live execution with zero divergence.
    cfg.execution_style = "limit"
    cfg.pullback_atr_mult = 0.0

    # REAL friction — zero would be an illusion of profitability.
    # Bybit linear perps published schedule:
    #   maker = 0.02% = 2 bps     taker = 0.055% ≈ 5.5 bps
    # Plus a conservative 3 bps slippage envelope per leg to account for
    # hostile book depth during news events and funding resets.
    cfg.maker_fee_bps = 2.0
    cfg.taker_fee_bps = 5.5
    cfg.slippage_bps  = 3.0

    # ══════════════════════════════════════════════════════════════════
    # DIRECTIVE 1 — STRUCTURAL STOP LOSS & TIME-IN-FORCE
    # ══════════════════════════════════════════════════════════════════
    # 0.40 ATR was a liquidity-sweep target that sat INSIDE the typical
    # BTC 15m Brownian noise envelope (~0.6-0.8 ATR one-sigma). That meant
    # ~55% of trades were being stopped out by random walk noise, not by
    # genuine structural invalidation.
    #
    # 1.5 ATR places the stop beyond ~95% of random walk excursions, so
    # only a real break of the setup's invalidation level can trigger SL.
    # This widens the per-unit risk distance (raw_risk) which correspondingly
    # shrinks the per-trade R-multiple variance and gives the meta-brain a
    # cleaner signal on which setups actually survive hostile microstructure.
    cfg.atr_mult_sl = 1.5

    # Outer cap on structural TP targets. When Phase 5 projects a TP via
    # volume profile / Fib extensions / HTF swings, this cap prevents
    # selecting targets that imply >6R — targets that far out have negligible
    # realized hit rate in backtests and distort the EV calculation.
    cfg.atr_mult_tp = 6.0

    # Minimum GROSS risk/reward that a setup must promise before the Phase 6
    # AI gate evaluates. After friction deduction (maker-limit entry + taker
    # exit + 3 bps slippage per leg at typical BTC price ≈ 17-20 bps total)
    # a 2.0R gross target collapses to roughly 1.82-1.85R net — still a
    # healthy positive-expectancy cushion.
    cfg.min_rr = 2.0

    # Hard 12h horizon on 15m (= 48 bars). If a structural trigger has NOT
    # resolved to TP or SL within this window, TradeManager force-exits at
    # the current bar close with full friction applied. Previously 0, which
    # meant setups held indefinitely and bled alpha on every candle they
    # stayed open — now bounded to a finite, measurable R outcome.
    #
    # This must match the exporter's TIF horizon EXACTLY so that the meta-
    # brain's training labels reflect the same exit distribution that the
    # live bot will produce. vajra_export_events.py intentionally does NOT
    # override this value.
    cfg.time_in_force_decay = 48

    # Keep raw TP/SL/TIF semantics: NO break-even, NO trailing overlays, NO
    # dynamic TP scaling. These features decouple the realized R-multiple
    # from the setup's intrinsic signal and pollute the training labels.
    cfg.be_trigger_r            = 0.0
    cfg.trailing_stop_trigger_r = 0.0
    cfg.trailing_dist_r         = 0.0
    cfg.dynamic_tp_enabled      = False

    # ══════════════════════════════════════════════════════════════════
    # RISK & CONCURRENCY
    # ══════════════════════════════════════════════════════════════════
    cfg.risk_per_trade      = 0.01
    cfg.max_concurrent      = 6
    cfg.min_target_dist_pct = 0.10
    cfg.use_macro_data      = True
    cfg.use_meta_labeling   = True

    # ══════════════════════════════════════════════════════════════════
    # PORTFOLIO CONTROLS
    # ══════════════════════════════════════════════════════════════════
    cfg.max_daily_loss_r = 4.0
    cfg.max_risk_factor  = 2.0

    # ══════════════════════════════════════════════════════════════════
    # AI GATES
    # ══════════════════════════════════════════════════════════════════
    # SMOTE has been eradicated from the trainer. Specialist brains now
    # train on the native class balance with scale_pos_weight only, so
    # raw XGBoost outputs are properly calibrated (no extreme squashing).
    cfg.min_ev             = 0.01
    cfg.min_prob_long      = 0.30
    cfg.min_prob_short     = 0.30
    cfg.dynamic_risk_scaling = True

    # ══════════════════════════════════════════════════════════════════
    # ENGINE STRUCTURAL GATE CONTROLS
    # ══════════════════════════════════════════════════════════════════
    cfg.filter_adx_chop      = True
    cfg.adx_chop_threshold   = 10.0
    cfg.reversal_evidence_min = 2
    cfg.wick_rejection_pct   = 0.30
    cfg.vol_confirm_rvol     = 1.2
    cfg.vol_spike_lookback   = 5
    cfg.ob_freshness_bars    = 64
    cfg.fvg_tolerance_atr    = 0.5
    cfg.mtf_alignment_min    = 1

    # ══════════════════════════════════════════════════════════════════
    # META-BRAIN THRESHOLD (two-gate w/ specialist)
    # ══════════════════════════════════════════════════════════════════
    cfg.min_meta_prob = 0.22
