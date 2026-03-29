# ==========================================
# FILE: strategies/strategy_1_ema.py
# PURPOSE: Strategy 1 - 10/20/200 EMA + Marubozu
# CHANGES FROM PREVIOUS VERSION:
#   1. Marubozu threshold relaxed from 0.90 → 0.75 (standard definition)
#   2. Removed duplicate Marubozu calculation in _apply_indicators
#   3. Added ATR (14-day) column for use by the backtester's stop-loss logic
#   4. _generate_signals() is cleaner — no duplicate code
# ==========================================
import pandas as pd
import numpy as np


class EmaMarubozuStrategy:
    def __init__(self, data):
        self.df = data.copy()

    def _apply_indicators(self):
        """Calculates EMAs and ATR. Marubozu is done once in _generate_signals."""

        print("Strategy 1: Calculating 10, 20, 200 EMAs...")
        self.df['EMA_10'] = self.df['Close'].ewm(span=15, adjust=False).mean()
        self.df['EMA_20'] = self.df['Close'].ewm(span=30, adjust=False).mean()
        self.df['EMA_200'] = self.df['Close'].ewm(span=200, adjust=False).mean()

        # ------------------------------------------------------------------
        # ATR (Average True Range) — 14-day
        # This is used by the backtester for the stop-loss calculation.
        # True Range = max of:
        #   (High - Low),
        #   abs(High - previous Close),
        #   abs(Low  - previous Close)
        # ------------------------------------------------------------------
        print("Strategy 1: Calculating ATR (14-day)...")
        high_low   = self.df['High'] - self.df['Low']
        high_close = abs(self.df['High'] - self.df['Close'].shift(1))
        low_close  = abs(self.df['Low']  - self.df['Close'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        self.df['ATR_14'] = true_range.ewm(span=14, adjust=False).mean()

    def _generate_signals(self):
        """
        Signal logic with 3 tiers.

        FIXED: Marubozu threshold changed from 0.90 to 0.75.
        At 0.90 (90% of candle range must be body), almost no real candles qualify.
        At 0.75 (75%), you catch strong momentum candles — the standard definition
        used by most Indian technical analysts for daily charts.
        """

        print("Strategy 1: Scanning for Marubozu candles (threshold: 75%)...")
        candle_body  = abs(self.df['Close'] - self.df['Open'])
        candle_range = (self.df['High'] - self.df['Low']).replace(0, 0.0001)

        # CHANGED: was 0.90, now 0.75
        is_marubozu_shape = (candle_body / candle_range) > 0.75

        self.df['Bullish_Marubozu'] = is_marubozu_shape & (self.df['Close'] > self.df['Open'])
        self.df['Bearish_Marubozu'] = is_marubozu_shape & (self.df['Close'] < self.df['Open'])

        # ------------------------------------------------------------------
        # Signal and Confidence columns
        # Signal    : 1 = enter long, -1 = enter short/exit long, 0 = exit
        # Confidence: 0.5 = half capital (Tier 1), 1.0 = full capital (Tier 2)
        # ------------------------------------------------------------------
        self.df['Signal']     = np.nan
        self.df['Confidence'] = np.nan

        print("Strategy 1: Generating crossover signals...")

        # Detect the exact day of crossovers
        cross_up   = (self.df['EMA_10'] > self.df['EMA_20']) & \
                     (self.df['EMA_10'].shift(1) <= self.df['EMA_20'].shift(1))
        cross_down = (self.df['EMA_10'] < self.df['EMA_20']) & \
                     (self.df['EMA_10'].shift(1) >= self.df['EMA_20'].shift(1))

        # 5-day rolling memory for crossovers and Marubozus
        LIMIT = 5
        recent_cross_up    = cross_up.rolling(window=LIMIT).max()  == 1
        recent_cross_down  = cross_down.rolling(window=LIMIT).max() == 1
        recent_marubozu_bull = self.df['Bullish_Marubozu'].rolling(window=LIMIT).max() == 1
        recent_marubozu_bear = self.df['Bearish_Marubozu'].rolling(window=LIMIT).max() == 1

        # --- TIER 1: Confidence 0.5 (50% capital) ---
        # Plain crossover in the direction of the major trend. No Marubozu needed.
        basic_long  = cross_up   & (self.df['EMA_20'] > self.df['EMA_200'])
        basic_short = cross_down & (self.df['EMA_20'] < self.df['EMA_200'])
        self.df.loc[basic_long,  ['Signal', 'Confidence']] = [1,  0.5]
        self.df.loc[basic_short, ['Signal', 'Confidence']] = [-1, 0.5]

        # --- TIER 2: Confidence 1.0 (100% capital) ---
        # Scenario A: Cross TODAY + Marubozu in last 5 days
        strong_long_a  = cross_up   & recent_marubozu_bull & (self.df['EMA_20'] > self.df['EMA_200'])
        strong_short_a = cross_down & recent_marubozu_bear & (self.df['EMA_20'] < self.df['EMA_200'])
        self.df.loc[strong_long_a,  ['Signal', 'Confidence']] = [1,  1.0]
        self.df.loc[strong_short_a, ['Signal', 'Confidence']] = [-1, 1.0]

        # Scenario B: Cross was in last 5 days + Marubozu TODAY → upgrade position
        strong_long_b  = recent_cross_up   & self.df['Bullish_Marubozu'] & \
                         (self.df['EMA_20'] > self.df['EMA_200'])
        strong_short_b = recent_cross_down & self.df['Bearish_Marubozu'] & \
                         (self.df['EMA_20'] < self.df['EMA_200'])
        self.df.loc[strong_long_b,  ['Signal', 'Confidence']] = [1,  1.0]
        self.df.loc[strong_short_b, ['Signal', 'Confidence']] = [-1, 1.0]

        # --- EXITS ---

        # --- EXIT 1 ----

        # Counter-crossover while we are IN a long trade (EMA_20 still above EMA_200)
        # Counter-crossover while we are IN a short trade (EMA_20 still below EMA_200)

        #self.df.loc[cross_down & (self.df['EMA_20'] > self.df['EMA_200']),
        #            ['Signal', 'Confidence']] = [0, 0.0]
       # self.df.loc[cross_up   & (self.df['EMA_20'] < self.df['EMA_200']),
       #             ['Signal', 'Confidence']] = [0, 0.0] 

        # --- EXIT 2 ----
        # Exit via backtester using Trailing loss principle i.e.
        # the stop loss value changes as peak changes (peak is monitored) and as price comes to highest_peak - 2 * ATR => EXIT..
        # Good for traders who want to hold the stocks for 6 or more months. 
        
        # --- EXIT 3 (Idea 2: Active Trader Price-to-EMA) ---
        # Exit the trade immediately if the daily Close drops below the 10 EMA
        
        #price_drops_below_10 = self.df['Close'] < self.df['EMA_10']
        #self.df.loc[price_drops_below_10, ['Signal', 'Confidence']] = [0, 0.0]
        
        # --- EXITS 4 Risk to Reward 1 : 2



    def run(self):
        self._apply_indicators()
        self._generate_signals()
        # Drop the first 200 rows — EMA_200 needs 200 candles to be accurate
        self.df.dropna(inplace=True)
        return self.df