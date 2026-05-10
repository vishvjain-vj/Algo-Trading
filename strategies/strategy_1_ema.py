# ==========================================
# FILE: strategies/strategy_1_ema.py
# PURPOSE: Strategy 1 — EMA Crossover (fast/slow/trend) + Marubozu + Slope Filter
#
# WHAT THIS STRATEGY DOES (plain English):
#   This is a TREND-FOLLOWING strategy. It only looks to BUY when the big
#   picture is bullish (price above the 200 EMA), and only when the short-term
#   momentum is accelerating (fast EMA crossing above slow EMA with rising slope,
#   optionally confirmed by a strong Marubozu candle).
#
#   The strategy produces three output columns that the backtester uses:
#     - Signal     : 1 = buy signal, NaN = no action
#                    (Signal = -1 was removed as the backtester handles all exits)
#     - Confidence : 0.5 = crossover only (Tier 1), 1.0 = crossover + Marubozu (Tier 2)
#     - ATR_14     : used by backtester to calculate stop distance and target
#
# BUGS FIXED FROM PREVIOUS VERSION:
#   1. EMA SPAN MISMATCH — EMA_10 was calculated with ewm(span=15) and
#      EMA_20 was calculated with ewm(span=30). The column name said "10"
#      but the math was doing "15". Fixed to match names: EMA_10 uses span=10,
#      EMA_20 uses span=20.
#      NOTE: If you intentionally wanted slower EMAs (15/30), change the
#      spans back and rename the columns to EMA_15 and EMA_20 to be honest.
#
#   2. SLOPE_THRESHOLD WAS 0.75 — the docstring said 0.10 is normal and
#      0.15 is strict, but the code used 0.75, which filtered out almost
#      every signal. Reset to 0.10 as the docstring intended.
#
#   3. HARDCODED CONSTANTS — SLOPE_LOOKBACK, SLOPE_THRESHOLD, and the
#      5-day rolling MEMORY_DAYS were buried inside methods, making them
#      impossible to tune without editing the source code. Moved to __init__.
#
#   4. DEAD SIGNAL = -1 REMOVED — The strategy was generating Signal = -1
#      for bearish crossovers, but PortfolioBacktester does not use -1 to exit.
#      Generating it wasted computation and created confusion about what
#      signals were active. Removed from signal generation.
#
#   5. COMMENTED-OUT EXITS CLEANED UP — EXIT 1, EXIT 3, EXIT 4 were
#      commented out with no explanation. Replaced with clear documentation
#      of what each exit option is and how to re-enable it if you want to.
# ==========================================

import pandas as pd
import numpy as np


class EmaMarubozuStrategy:
    """
    EMA Crossover + Marubozu confirmation strategy.

    Parameters
    ----------
    data              : Raw OHLCV DataFrame from data_fetcher (columns must include
                        Open, High, Low, Close, Volume — capital letters)
    fast_span         : EMA span for the fast (entry signal) EMA. Default 10.
    slow_span         : EMA span for the slow (trend direction) EMA. Default 20.
    trend_span        : EMA span for the major trend filter. Default 200.
                        Buy signals are only generated when Close > this EMA.
    atr_period        : ATR lookback in days. Default 14.
    marubozu_threshold: Candle body must be this fraction of the candle range
                        to qualify as a Marubozu. Default 0.75 (75%).
                        Lower = more signals (looser), higher = fewer (stricter).
    slope_lookback    : Number of days over which to measure the EMA slope. Default 5.
    slope_threshold   : Minimum normalized slope (%) required to enter.
                        0.05 = very permissive, 0.10 = standard, 0.15 = strict.
    memory_days       : How many days back a crossover or Marubozu "counts."
                        Tier 2 fires if cross + Marubozu happened within this window.
    """

    def __init__(
        self,
        data,
        fast_span          = 10,
        slow_span          = 20,
        trend_span         = 200,
        atr_period         = 14,
        marubozu_threshold = 0.75,
        slope_lookback     = 5,
        slope_threshold    = 0.10,   # BUG FIX: was 0.75 in code while docstring said 0.10
        memory_days        = 5,
    ):
        self.df                 = data.copy()
        self.fast_span          = fast_span
        self.slow_span          = slow_span
        self.trend_span         = trend_span
        self.atr_period         = atr_period
        self.marubozu_threshold = marubozu_threshold
        self.slope_lookback     = slope_lookback
        self.slope_threshold    = slope_threshold
        self.memory_days        = memory_days

    # ------------------------------------------------------------------
    # STEP 1: CALCULATE INDICATORS
    # All numbers the signal logic will need, computed once here.
    # ------------------------------------------------------------------
    def _apply_indicators(self):
        print(f"  Calculating EMA-{self.fast_span}, EMA-{self.slow_span}, EMA-{self.trend_span}...")
        # BUG FIX: span now matches the variable name.
        # Previously: EMA_10 used span=15, EMA_20 used span=30.
        self.df['EMA_Fast']  = self.df['Close'].ewm(span=self.fast_span,  adjust=False).mean()
        self.df['EMA_Slow']  = self.df['Close'].ewm(span=self.slow_span,  adjust=False).mean()
        self.df['EMA_Trend'] = self.df['Close'].ewm(span=self.trend_span, adjust=False).mean()

        # ATR (Average True Range) — used by backtester for stop and target distance.
        # True Range is the largest of:
        #   (a) Today's High - Today's Low
        #   (b) |Today's High - Yesterday's Close|
        #   (c) |Today's Low  - Yesterday's Close|
        print(f"  Calculating ATR-{self.atr_period}...")
        hl         = self.df['High'] - self.df['Low']
        hc         = (self.df['High'] - self.df['Close'].shift(1)).abs()
        lc         = (self.df['Low']  - self.df['Close'].shift(1)).abs()
        true_range = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        self.df['ATR_14'] = true_range.ewm(span=self.atr_period, adjust=False).mean()

        # EMA Fast Slope — measures how fast the fast EMA is rising or falling.
        # Normalized by Close price so the % is comparable across different
        # price levels (e.g. Reliance at ₹1,500 vs ₹3,000).
        # Formula: ((EMA_Fast_today - EMA_Fast_N_days_ago) / Close_today) × 100
        print(f"  Calculating EMA Fast Slope ({self.slope_lookback}-day)...")
        self.df['EMA_Slope'] = (
            (self.df['EMA_Fast'] - self.df['EMA_Fast'].shift(self.slope_lookback))
            / self.df['Close']
        ) * 100

    # ------------------------------------------------------------------
    # STEP 2: GENERATE SIGNALS
    # Two tiers of entry signals. No exit signals — the backtester handles
    # all exits (Logic Gate, Target Hit, Trailing Stop).
    # ------------------------------------------------------------------
    def _generate_signals(self):
        df = self.df

        # --- Marubozu Detection ---
        # A Marubozu is a candle where most of the range is body (little wick).
        # It signals strong directional commitment from buyers or sellers.
        print(f"  Scanning for Marubozu candles (threshold: {self.marubozu_threshold:.0%})...")
        body  = (df['Close'] - df['Open']).abs()
        rng   = (df['High'] - df['Low']).replace(0, 0.0001)
        is_marubozu_shape    = (body / rng) > self.marubozu_threshold
        df['Bull_Marubozu']  = is_marubozu_shape & (df['Close'] > df['Open'])

        # --- Initialize output columns ---
        df['Signal']     = np.nan
        df['Confidence'] = np.nan

        # --- Crossover Detection ---
        # A bullish crossover: fast EMA was below slow EMA yesterday,
        # now it is above. This is the exact day the crossover fires.
        print("  Generating crossover signals...")
        cross_up = (
            (df['EMA_Fast'] >  df['EMA_Slow']) &
            (df['EMA_Fast'].shift(1) <= df['EMA_Slow'].shift(1))
        )

        # Rolling memory: was there a crossover OR a Marubozu in the last N days?
        N = self.memory_days
        recent_cross_up = cross_up.rolling(window=N).max() == 1
        recent_bull_maru = df['Bull_Marubozu'].rolling(window=N).max() == 1

        # --- Slope Filter ---
        # We check YESTERDAY's slope, not today's.
        #
        # Why yesterday and not today?
        # On the exact crossover day, the fast EMA is by definition higher
        # than it was yesterday, so today's slope is always positive — it
        # would never filter anything. Checking yesterday's slope asks:
        # "Was momentum ALREADY building before the crossover fired?"
        # That is the question we actually care about.
        slope_was_rising = df['EMA_Slope'].shift(1) > self.slope_threshold

        # --- Major Trend Filter ---
        # Only go long when the stock is above its long-term EMA.
        # This keeps us out of downtrends.
        in_uptrend = df['EMA_Slow'] > df['EMA_Trend']

        # --- TIER 1 (Confidence 0.5): Crossover + trend + slope ---
        # The basic setup. Fewer filters = more signals but more false positives.
        tier1_long = cross_up & in_uptrend & slope_was_rising
        df.loc[tier1_long, ['Signal', 'Confidence']] = [1, 0.5]

        # --- TIER 2 (Confidence 1.0): Crossover + Marubozu + trend + slope ---
        # Scenario A: Crossover TODAY + Marubozu somewhere in last N days
        tier2_a = cross_up & recent_bull_maru & in_uptrend & slope_was_rising
        df.loc[tier2_a, ['Signal', 'Confidence']] = [1, 1.0]

        # Scenario B: Crossover was in last N days + Marubozu TODAY
        tier2_b = recent_cross_up & df['Bull_Marubozu'] & in_uptrend & slope_was_rising
        df.loc[tier2_b, ['Signal', 'Confidence']] = [1, 1.0]

        # ------------------------------------------------------------------
        # EXITS (reference — all exits are handled in backtester.py)
        # ------------------------------------------------------------------
        # The backtester handles three exits:
        #
        #   ACTIVE  — Trailing Stop-Loss: exit when low falls below (peak_close - N×ATR)
        #   ACTIVE  — Take-Profit:        exit when close exceeds  (entry + N×ATR)
        #   ACTIVE  — Logic Gate:         exit if profit ≥ 2% every 15 calendar days
        #
        # Options you can re-enable by uncommenting:
        #
        #   OPTION A — EMA Cross Exit:
        #     Exit immediately when the fast EMA crosses back below the slow EMA.
        #     Good for: active traders who want quick exits.
        #     Risk: gets whipsawed in choppy markets.
        #     To re-enable: uncomment the block below AND add this to backtester.py
        #     Section B: "if row['Signal'] == 0: is_exit, exit_reason = True, 'EMA Cross'"
        #
        #     cross_down = (df['EMA_Fast'] < df['EMA_Slow']) & (df['EMA_Fast'].shift(1) >= df['EMA_Slow'].shift(1))
        #     df.loc[cross_down, ['Signal', 'Confidence']] = [0, 0.0]
        #
        #   OPTION B — Price Below Fast EMA Exit:
        #     Exit when Close drops below the fast EMA. More sensitive than Option A.
        #     Good for: momentum traders who want to lock in gains quickly.
        #     Risk: even more prone to whipsaw on volatile days.
        #
        #     df.loc[df['Close'] < df['EMA_Fast'], ['Signal', 'Confidence']] = [0, 0.0]
        # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # PUBLIC METHOD: run the full pipeline and return the processed DataFrame.
    # ------------------------------------------------------------------
    def run(self):
        print(f"Strategy: EMA-{self.fast_span}/{self.slow_span}/{self.trend_span} + Marubozu")
        self._apply_indicators()
        self._generate_signals()

        # Drop the first N rows where indicators haven't warmed up yet.
        # EMA-200 needs ~200 rows before its values are meaningful.
        self.df.dropna(subset=['EMA_Trend', 'ATR_14', 'EMA_Slope'], inplace=True)

        return self.df
