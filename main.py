# ==========================================
# FILE: main.py
# PURPOSE: System dispatcher
# CHANGES: Uses print_trade_report() instead of print_metrics()
# ==========================================
from core.data_fetcher import fetch_daily_data
from core.backtester import SimpleBacktester
from strategies.strategy_1_ema import EmaMarubozuStrategy


def main():
    # ------------------------------------------------------------------
    # 1. Configuration
    # ------------------------------------------------------------------
    TICKER          = "RELIANCE.NS"
    START           = "2019-01-01"
    END             = "2026-01-01"
    INITIAL_CAPITAL = 100_000      # ₹1,00,000

    # Stop-loss: 2 × ATR below entry price.
    # Increase to 3.0 for more room, decrease to 1.5 for tighter stops.
    # Set to None to disable stop-loss entirely.
    ATR_STOP = 2.0

    # ------------------------------------------------------------------
    # 2. Fetch Data
    # ------------------------------------------------------------------
    raw_data = fetch_daily_data(TICKER, START, END)

    if raw_data is None:
        print("Data fetch failed. Exiting.")
        return

    # ------------------------------------------------------------------
    # 3. Run Strategy
    # ------------------------------------------------------------------
    print(f"\n--- Running Strategy 1: EMA & Marubozu ---")
    strategy      = EmaMarubozuStrategy(raw_data)
    processed_data = strategy.run()

    # Quick diagnostic: how many Marubozu candles were found?
    bull_m = processed_data['Bullish_Marubozu'].sum()
    bear_m = processed_data['Bearish_Marubozu'].sum()
    print(f"  Bullish Marubozu candles found : {bull_m}")
    print(f"  Bearish Marubozu candles found : {bear_m}")

    # ------------------------------------------------------------------
    # 4. Run Backtest
    # ----------------------------------------------------------------
    backtester = SimpleBacktester(
        processed_data,
        initial_capital=INITIAL_CAPITAL,
        atr_stop_multiplier=1.0
    )
    results_df = backtester.run_backtest()

    # ------------------------------------------------------------------
    # 5. Print Trade Report (per-trade, no buy-and-hold comparison)
    # ------------------------------------------------------------------
    backtester.print_trade_report()


if __name__ == "__main__":
    main()
