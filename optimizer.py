# ==========================================
# FILE: optimizer.py
# PURPOSE: Sensitivity / parameter optimization
#
# BUGS FIXED FROM ORIGINAL:
#   1. backtester.run_backtest() was never called — generate_trade_log()
#      reads self.df which only gets populated AFTER run_backtest().
#      Without it, the loop iterates over raw unprocessed data → no trades.
#
#   2. The optimizer bypassed the Strategy class entirely — it wrote raw
#      Signal = 1/-1 as a continuous block (wherever Fast > Slow),
#      but the backtester's generate_trade_log() looks for Signal transitions
#      (entry when sig == 1, exit when sig == 0 or -1). A continuous block
#      of Signal=1 with no transition = zero trades detected.
#      Fix: generate crossover signals (only the day of the cross).
#
#   3. No ATR_14 column was being calculated, so the stop-loss silently
#      did nothing (the 'ATR_14' in df.columns check in backtester was False).
#
#   4. No Confidence column → backtester threw KeyError on
#      df.loc[df['Signal']==1, 'Confidence']. Fixed by adding Confidence=1.0.
#
#   5. No EMA_200 filter — optimizer was testing ALL crossovers regardless
#      of major trend direction. Added as optional (default: enabled).
#
#   6. Final capital calculation was wrong: summing raw P&L ₹ values from
#      generate_trade_log() doesn't compound correctly when position sizing
#      varies. Fixed by using Portfolio_Value from run_backtest() directly.
#
#   7. Added proper metrics beyond just Return%: Win Rate, Sharpe Ratio,
#      Max Drawdown, Num Trades — so you can pick a robust combination,
#      not just the highest return (which is often overfitted noise).
# ==========================================

import pandas as pd
import numpy as np
import itertools

from core.backtester import SimpleBacktester
from core.data_fetcher import fetch_daily_data


# ------------------------------------------------------------------
# Helper: calculate ATR (14-day) — same formula as strategy_1_ema.py
# ------------------------------------------------------------------
def _calc_atr(df, period=14):
    high_low   = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift(1))
    low_close  = abs(df['Low']  - df['Close'].shift(1))
    tr         = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


# ------------------------------------------------------------------
# Helper: build signals for a given fast/slow EMA pair
#         Returns a copy of df with Signal, Confidence, ATR_14 added.
# ------------------------------------------------------------------
def _build_signals(df, fast, slow, use_ema200_filter=True):
    out = df.copy()

    # 1. EMAs and ATR
    out['EMA_Fast'] = out['Close'].ewm(span=fast,  adjust=False).mean()
    out['EMA_Slow'] = out['Close'].ewm(span=slow,  adjust=False).mean()
    out['EMA_200']  = out['Close'].ewm(span=200,   adjust=False).mean()
    out['ATR_14']   = _calc_atr(out, period=14)

    # ------------------------------------------------------------------
    # 2. THE MARUBOZU FILTER (Exact match to strategy_1_ema.py)
    # ------------------------------------------------------------------
    candle_body  = abs(out['Close'] - out['Open'])
    candle_range = (out['High'] - out['Low']).replace(0, 0.0001)

    # 75% threshold
    is_marubozu_shape = (candle_body / candle_range) > 0.75

    out['Bullish_Marubozu'] = is_marubozu_shape & (out['Close'] > out['Open'])
    out['Bearish_Marubozu'] = is_marubozu_shape & (out['Close'] < out['Open'])

    # ------------------------------------------------------------------
    # 3. CROSSOVERS & 5-DAY MEMORY
    # ------------------------------------------------------------------
    cross_up   = (out['EMA_Fast'] > out['EMA_Slow']) & \
                 (out['EMA_Fast'].shift(1) <= out['EMA_Slow'].shift(1))
    cross_down = (out['EMA_Fast'] < out['EMA_Slow']) & \
                 (out['EMA_Fast'].shift(1) >= out['EMA_Slow'].shift(1))

    LIMIT = 5
    recent_cross_up      = cross_up.rolling(window=LIMIT).max() == 1
    recent_cross_down    = cross_down.rolling(window=LIMIT).max() == 1
    recent_marubozu_bull = out['Bullish_Marubozu'].rolling(window=LIMIT).max() == 1
    recent_marubozu_bear = out['Bearish_Marubozu'].rolling(window=LIMIT).max() == 1

    out['Signal']     = np.nan
    out['Confidence'] = np.nan

    # Trend filter matching your code (using Slow EMA vs 200 EMA)
    if use_ema200_filter:
        bull_trend = out['EMA_Slow'] > out['EMA_200']
        bear_trend = out['EMA_Slow'] < out['EMA_200']
    else:
        bull_trend = True
        bear_trend = True

    # ------------------------------------------------------------------
    # 4. TIER 1: Confidence 0.5 (Basic Crossover)
    # ------------------------------------------------------------------
    basic_long  = cross_up   & bull_trend
    basic_short = cross_down & bear_trend
    out.loc[basic_long,  ['Signal', 'Confidence']] = [1,  0.5]
    out.loc[basic_short, ['Signal', 'Confidence']] = [-1, 0.5]

    # ------------------------------------------------------------------
    # 5. TIER 2: Confidence 1.0 (Crossover + Marubozu Combo)
    # ------------------------------------------------------------------
    # Scenario A: Cross TODAY + Marubozu recently
    strong_long_a  = cross_up   & recent_marubozu_bull & bull_trend
    strong_short_a = cross_down & recent_marubozu_bear & bear_trend
    out.loc[strong_long_a,  ['Signal', 'Confidence']] = [1,  1.0]
    out.loc[strong_short_a, ['Signal', 'Confidence']] = [-1, 1.0]

    # Scenario B: Cross recently + Marubozu TODAY
    strong_long_b  = recent_cross_up   & out['Bullish_Marubozu'] & bull_trend
    strong_short_b = recent_cross_down & out['Bearish_Marubozu'] & bear_trend
    out.loc[strong_long_b,  ['Signal', 'Confidence']] = [1,  1.0]
    out.loc[strong_short_b, ['Signal', 'Confidence']] = [-1, 1.0]

    # Note: As requested, standard EMA exits are disabled here just like 
    # they are commented out in your `strategy_1_ema.py`. 
    # The backtester's Trailing Stop and Take-Profit will handle all exits!

    # Drop rows where indicators aren't warm yet
    out.dropna(subset=['EMA_200', 'ATR_14'], inplace=True)
    return out


# ------------------------------------------------------------------
# Helper: run one backtest and return a metrics dict
# ------------------------------------------------------------------
def _run_one(df_signals, initial_capital, atr_mult):
    bt = SimpleBacktester(
        df_signals,
        initial_capital=initial_capital,
        atr_stop_multiplier=atr_mult
    )
    bt.run_backtest()          # <-- MUST be called before generate_trade_log()
    trades = bt.generate_trade_log()

    # --- Final return ---
    final_val     = bt.df['Portfolio_Value'].dropna().iloc[-1] \
                    if 'Portfolio_Value' in bt.df.columns and not bt.df['Portfolio_Value'].dropna().empty \
                    else initial_capital
    total_ret_pct = ((final_val - initial_capital) / initial_capital) * 100

    if trades.empty:
        return {
            'Return_%'      : round(total_ret_pct, 2),
            'Num_Trades'    : 0,
            'Win_Rate_%'    : 0.0,
            'Avg_Win_%'     : 0.0,
            'Avg_Loss_%'    : 0.0,
            'Max_DD_%'      : 0.0,
            'Sharpe'        : 0.0,
        }

    wins   = trades[trades['Return %'] > 0]
    losses = trades[trades['Return %'] <= 0]

    # --- Max portfolio drawdown (peak-to-trough on Portfolio_Value) ---
    pv        = bt.df['Portfolio_Value'].dropna()
    roll_max  = pv.cummax()
    drawdowns = (pv - roll_max) / roll_max * 100
    max_dd    = drawdowns.min()   # most negative value

    # --- Simplified Sharpe (daily strategy returns, annualised) ---
    daily_ret = bt.df['Strategy_Returns'].dropna()
    sharpe    = 0.0
    if daily_ret.std() > 0:
        sharpe = round((daily_ret.mean() / daily_ret.std()) * np.sqrt(252), 2)

    return {
        'Return_%'   : round(total_ret_pct, 2),
        'Num_Trades' : len(trades),
        'Win_Rate_%' : round(len(wins) / len(trades) * 100, 1),
        'Avg_Win_%'  : round(wins['Return %'].mean(), 2)   if not wins.empty   else 0.0,
        'Avg_Loss_%' : round(losses['Return %'].mean(), 2) if not losses.empty else 0.0,
        'Max_DD_%'   : round(max_dd, 2),
        'Sharpe'     : sharpe,
    }


# ------------------------------------------------------------------
# MAIN OPTIMIZER
# ------------------------------------------------------------------
def run_optimizer():
    # ---- Config ----
    TICKER          = "RELIANCE.NS"
    START           = "2019-01-01"
    END             = "2026-01-01"
    INITIAL_CAPITAL = 100_000
    USE_EMA200      = True          # set False to test without trend filter

    # ---- Parameter grid ----
    fast_emas = [5, 8, 10, 12, 15]
    slow_emas = [15, 20, 25, 30]
    atr_stops = [0.5, 1.0, 1.5, 2.0]

    print(f"Loading data for {TICKER} ({START} → {END})...")
    raw_df = fetch_daily_data(TICKER, START, END)
    if raw_df is None:
        print("Data fetch failed. Exiting.")
        return

    combos = [
        (f, s, a)
        for f, s, a in itertools.product(fast_emas, slow_emas, atr_stops)
        if f < s   # fast must be strictly less than slow
    ]
    print(f"Running {len(combos)} combinations...\n")

    leaderboard = []

    for i, (fast, slow, atr) in enumerate(combos, 1):
        try:
            signals_df = _build_signals(raw_df, fast, slow,
                                        use_ema200_filter=USE_EMA200)
            metrics    = _run_one(signals_df, INITIAL_CAPITAL, atr)
        except Exception as e:
            print(f"  Skipped Fast={fast} Slow={slow} ATR={atr}: {e}")
            continue

        leaderboard.append({
            'Fast_EMA'  : fast,
            'Slow_EMA'  : slow,
            'ATR_Stop'  : atr,
            **metrics
        })

        # Progress every 10 combos
        if i % 10 == 0:
            print(f"  [{i}/{len(combos)}] done...")

    if not leaderboard:
        print("No results. Check data and parameters.")
        return

    lb_df = pd.DataFrame(leaderboard)

    # ---- Print top 10 by Return ----
    print("\n" + "=" * 80)
    print("  TOP 10 BY TOTAL RETURN")
    print("=" * 80)
    top_return = lb_df.sort_values('Return_%', ascending=False).head(10)
    _print_table(top_return)

    # ---- Print top 10 by Sharpe (risk-adjusted) ----
    print("\n" + "=" * 80)
    print("  TOP 10 BY SHARPE RATIO  (best risk-adjusted, more reliable)")
    print("=" * 80)
    top_sharpe = lb_df[lb_df['Num_Trades'] >= 3] \
                     .sort_values('Sharpe', ascending=False).head(10)
    _print_table(top_sharpe)

    # ---- Print top 10 by Win Rate (min 4 trades) ----
    print("\n" + "=" * 80)
    print("  TOP 10 BY WIN RATE  (min 4 trades)")
    print("=" * 80)
    top_wr = lb_df[lb_df['Num_Trades'] >= 4] \
                 .sort_values('Win_Rate_%', ascending=False).head(10)
    _print_table(top_wr)

    # ---- Save full results ----
    out_path = "optimizer_results.csv"
    lb_df.sort_values('Return_%', ascending=False).to_csv(out_path, index=False)
    print(f"\nFull results saved to {out_path}")


def _print_table(df):
    header = (
        f"{'#':<3} {'Fast':>5} {'Slow':>5} {'ATR':>5} "
        f"{'Return%':>9} {'Trades':>7} {'WinRate%':>9} "
        f"{'AvgWin%':>8} {'AvgLoss%':>9} {'MaxDD%':>8} {'Sharpe':>7}"
    )
    print(header)
    print("-" * len(header))
    for rank, (_, row) in enumerate(df.iterrows(), 1):
        print(
            f"{rank:<3} {int(row['Fast_EMA']):>5} {int(row['Slow_EMA']):>5} "
            f"{row['ATR_Stop']:>5.1f} "
            f"{row['Return_%']:>+9.2f}% {int(row['Num_Trades']):>7} "
            f"{row['Win_Rate_%']:>9.1f}% "
            f"{row['Avg_Win_%']:>+8.2f}% {row['Avg_Loss_%']:>+9.2f}% "
            f"{row['Max_DD_%']:>+8.2f}% {row['Sharpe']:>7.2f}"
        )


if __name__ == "__main__":
    run_optimizer()