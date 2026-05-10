# ==========================================
# FILE: main_multiTicker.py
# PURPOSE: Entry point — fetches data, runs strategy, runs backtest, prints report.
#
# HOW IT WORKS:
#   1. Download daily OHLCV data for each ticker in the watchlist
#   2. Run EmaMarubozuStrategy on each → adds Signal, ATR_14, Confidence columns
#   3. Hand all processed data to PortfolioBacktester
#   4. PortfolioBacktester walks day-by-day, manages shared cash pool,
#      enters on Signal=1, exits via Trailing Stop / Take-Profit / Logic Gate
#   5. Print the final trade log and summary statistics
#
# TUNING GUIDE (all parameters are in one place below):
#   - WATCHLIST       : add/remove tickers. More tickers = more signals = more trades
#                       = more statistically meaningful backtest.
#   - INITIAL_CAPITAL : starting cash in rupees
#   - FAST_SPAN/SLOW_SPAN : EMA crossover periods. Smaller = more signals (noisy).
#                            Larger = fewer signals (robust).
#   - ATR_STOP_MULT   : stop distance in ATR units. Smaller = tighter stop = more exits.
#   - ATR_TARGET_MULT : target distance in ATR units. Must be > ATR_STOP_MULT for
#                       positive risk-reward. E.g. stop=2×ATR, target=4×ATR = 1:2 RR.
#   - SLOPE_THRESHOLD : how steep the fast EMA must be rising before entry.
#                       0.05 = loose, 0.10 = balanced, 0.15 = strict.
# ==========================================

import os
import sys
import pandas as pd
from colorama import Fore, Back, Style, init
init(autoreset=True) # Automatically resets color after each print
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.data_fetcher import fetch_daily_data
from strategies.strategy_1_ema import EmaMarubozuStrategy
from core.backtester import PortfolioBacktester


# -----------------------------------------------------------------------
# ALL TUNABLE SETTINGS — only change things here, not inside the classes
# -----------------------------------------------------------------------
WATCHLIST        = ['BEL.NS', 'BSE.NS', 'TRENT.NS', 'ADANIPOWER.NS',
                    'APARINDS.NS', 'SOLARINDS.NS', 'SUNPHARMA.NS']
INITIAL_CAPITAL  = 100_000          # ₹1,00,000

START            = "2019-01-01"
END              = "2026-01-01"

# Strategy parameters
FAST_SPAN        = 10               # Fast EMA period
SLOW_SPAN        = 20               # Slow EMA period
SLOPE_THRESHOLD  = 0.10             # BUG FIX: was 0.75 — now 0.10 as intended

# Backtester parameters
ATR_STOP_MULT    = 3.0              # Stop = entry - 2×ATR
ATR_TARGET_MULT  = 6.0              # Target = entry + 4×ATR  → 1:2 risk-reward
MAX_POSITIONS    = len(WATCHLIST)   # Max simultaneous open positions
GATE_DAYS        = 15               # Logic Gate check every 15 calendar days
MIN_GATE_PROFIT  = 0.02             # Logic Gate fires if profit ≥ 2%


def run_portfolio_report():
    print(f"\n{'='*70}")
    print(f"  ALGO TRADING BACKTEST: {', '.join(WATCHLIST)}")
    print(f"  Period: {START} → {END}  |  Capital: ₹{INITIAL_CAPITAL:,.0f}")
    print(f"{'='*70}\n")

    # Step 1: Download data and apply strategy indicators/signals
    processed_data = {}

    for ticker in st.session_state.watchlist:
        print(f"\n[{ticker}] Fetching and processing...")
        try:
            data = fetch_daily_data(ticker, start_date=START, end_date=END)
            if data is None or data.empty:
                print(f"  ⚠ No data returned for {ticker}, skipping.")
                continue

            strategy = EmaMarubozuStrategy(
                data,
                fast_span       = FAST_SPAN,
                slow_span       = SLOW_SPAN,
                slope_threshold = SLOPE_THRESHOLD,
            )
            processed_data[ticker] = strategy.run()
            n_signals = (processed_data[ticker]['Signal'] == 1).sum()
            print(f"  ✓ Done. Buy signals found: {n_signals}")

        except Exception as e:
            print(f"  ✗ Error processing {ticker}: {e}")

    if not processed_data:
        print("\nNo data was processed. Check your internet connection or ticker symbols.")
        return

    # Step 2: Run the portfolio backtest
    print(f"\n{'='*70}")
    print("  RUNNING PORTFOLIO BACKTEST")
    print(f"{'='*70}")

    backtester = PortfolioBacktester(
        processed_data,
        initial_capital  = INITIAL_CAPITAL,
        atr_stop_mult    = ATR_STOP_MULT,
        atr_target_mult  = ATR_TARGET_MULT,
        max_positions    = MAX_POSITIONS,
        gate_days        = GATE_DAYS,
        min_gate_profit  = MIN_GATE_PROFIT,
    )

    master_log = backtester.run_portfolio_backtest()

# Step 3: Print the full trade log
    if not master_log.empty:
        # Sort by entry date
        master_log['_sort'] = pd.to_datetime(master_log['Entry Date'], format='%d %b %Y')
        master_log = master_log.sort_values('_sort').drop(columns=['_sort'])
        master_log = master_log.reset_index(drop=True)

        # --- ADD COLOR LOGIC HERE ---
        def color_reason(reason):
            if reason == "Stop-loss":
                return f"{Back.RED}{Fore.WHITE} {reason} {Style.RESET_ALL}"
            elif reason == "Target Hit":
                return f"{Back.GREEN}{Fore.WHITE} {reason} {Style.RESET_ALL}"
            elif reason == "Logic Gate":
                return f"{Back.CYAN}{Fore.BLACK} {reason} {Style.RESET_ALL}"
            return reason
            
        def color_return(ret):
            # Using Bright Text colors for better readability on black terminals
            if ret > 0:
                return f"{Fore.GREEN}{Style.BRIGHT}{ret:>+7.2f}%{Style.RESET_ALL}"
            elif ret < 0:
                return f"{Fore.RED}{Style.BRIGHT}{ret:>+7.2f}%{Style.RESET_ALL}"
            return f"{ret:>7.2f}%"
            
        # Create a display copy so we don't mess up the actual data
        display_log = master_log.copy()
        display_log['Exit Reason'] = display_log['Exit Reason'].apply(color_reason)
        display_log['Return %'] =  display_log['Return %'].apply(color_return)
        cols = [
            'Ticker', 'Entry Date', 'Entry Price ₹', 'Exit Date', 'Exit Price ₹',
            'Confidence', 'Capital Used ₹', 'Return %', 'P&L ₹',
            'Max Drawdown %', 'Days Held', 'Exit Reason'
        ]

        print("\n" + "=" * 135)
        print(f"{'TRADE LOG':^135}")
        print("=" * 135)
        # Use display_log here instead of master_log
        print(display_log[cols].to_string(index=False))

        # Step 4: Print summary statistics (pass the original master_log)
        backtester.print_summary(master_log)

    else:
        print("\nNo trades were completed in the simulation.")
        print("Possible causes:")
        print("  - SLOPE_THRESHOLD is too high (try lowering it)")
        print("  - Watchlist stocks were in a downtrend for most of the period")
        print("  - START date is too recent (not enough warm-up data for EMA-200)")


if __name__ == "__main__":
    run_portfolio_report()
