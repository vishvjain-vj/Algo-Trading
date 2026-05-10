# ==========================================
# FILE: core/backtester.py
# PURPOSE: Day-by-day portfolio simulation across multiple tickers.
#
# WHAT THIS FILE DOES (plain English):
#   - Takes a dictionary of {ticker: processed_dataframe} from the strategy
#   - Walks through every trading day in chronological order
#   - On each day: checks if any open position should exit, then checks if any
#     new position should be entered
#   - Produces a trade log DataFrame at the end
#
# BUGS FIXED FROM PREVIOUS VERSION:
#   1. STOP-LOSS EXIT PRICE — was using close_price when stop triggered.
#      A stop order fires at the stop price, not wherever the stock closes.
#      Fixed: exit at stop_price when stop is triggered. For gap-down days
#      (open far below stop) this is still optimistic vs reality, but it is
#      the standard approach with daily OHLC data.
#
#   2. LOGIC GATE MISSING GUARD — PortfolioBacktester was missing the
#      "close > stop_price" check that SimpleBacktester had. Without it, the
#      gate could exit a trade at a profit even on a day when the price was
#      already below the stop level. Fixed: added the guard condition.
#
#   3. DEAD CODE REMOVED — SimpleBacktester was never called by
#      main_multiTicker.py. Keeping it caused confusion about which class
#      was actually running. Removed entirely. If you need single-stock
#      backtesting in future, build it as a wrapper around PortfolioBacktester.
#
# NOTE ON EXITS (important to understand):
#   The strategy (strategy_1_ema.py) generates Signal = -1 for bearish
#   crossovers. PortfolioBacktester does NOT use that signal as an exit.
#   The only exits are: Logic Gate, Target Hit, and Trailing Stop-loss.
#   The -1 signal from the strategy is intentionally ignored here.
#   If you want to re-enable EMA-cross exits, add this block in section B:
#       if 'Signal' in df.columns and row['Signal'] == -1:
#           is_exit, exit_reason = True, "EMA Cross"
# ==========================================

import pandas as pd
import numpy as np


class PortfolioBacktester:
    """
    Simulates a shared-capital portfolio across multiple tickers, day by day.

    Parameters
    ----------
    data_dict         : dict of {ticker_string: processed_DataFrame}
                        Each DataFrame must have columns:
                        Close, Low, ATR_14, Signal, Confidence
    initial_capital   : Starting cash in rupees (default ₹1,00,000)
    atr_stop_mult     : Trailing stop = peak_close - (atr_stop_mult × ATR_14)
                        Default 2.0 means stop sits 2 ATRs below the peak.
    atr_target_mult   : Take-profit = entry + (atr_target_mult × ATR_14)
                        Default 4.0 gives a 1:2 risk-to-reward ratio
                        (stop is 2×ATR away, target is 4×ATR away).
    max_positions     : Max number of stocks held simultaneously (default 6)
    gate_days         : Logic Gate check frequency in calendar days (default 15)
    min_gate_profit   : Minimum return % to trigger a Logic Gate exit (default 0.02 = 2%)
    """

    def __init__(
        self,
        data_dict,
        initial_capital   = 100_000,
        atr_stop_mult     = 2.0,
        atr_target_mult   = 4.0,   # FIX: was hardcoded as 2.0 inside the loop (1:1 RR). Now 4.0 = 1:2 RR.
        max_positions     = 6,
        gate_days         = 15,
        min_gate_profit   = 0.02,
    ):
        self.data_dict       = data_dict
        self.initial_capital = initial_capital
        self.available_cash  = float(initial_capital)
        self.atr_stop_mult   = atr_stop_mult
        self.atr_target_mult = atr_target_mult
        self.max_positions   = max_positions
        self.gate_days       = gate_days
        self.min_gate_profit = min_gate_profit

    # ------------------------------------------------------------------
    # INTERNAL HELPER: build a sorted list of every unique trading date
    # across all tickers. This is the "master calendar" for the simulation.
    # ------------------------------------------------------------------
    def _build_master_calendar(self):
        all_dates = set()
        for df in self.data_dict.values():
            all_dates.update(df.index)
        return sorted(all_dates)

    # ------------------------------------------------------------------
    # INTERNAL HELPER: decide the correct exit price depending on the
    # reason for exit.
    #
    # WHY THIS MATTERS (the bug this fixes):
    #   Previously the code always used close_price as the exit price,
    #   even for stop-loss exits. But if the day's Low touched the stop,
    #   the actual exit in real trading happens AT the stop price
    #   (your broker's stop order fills there), not at the day's close.
    #
    #   Example: stop = ₹140, day's low = ₹135, day's close = ₹145.
    #   Old code: exit at ₹145 → stop didn't actually protect you.
    #   Fixed code: exit at ₹140 → stop did its job.
    #
    #   For gap-down scenarios (stock opens below stop), this is still
    #   slightly optimistic (real fill might be at the open, not stop price),
    #   but daily OHLC data doesn't let us do better.
    # ------------------------------------------------------------------
    def _get_exit_price(self, exit_reason, close_price, stop_price):
        if exit_reason == "Stop-loss":
            # Exit at the stop level, not wherever the stock happens to close.
            return stop_price
        # For Logic Gate, Target Hit: close price is a fair approximation.
        return close_price

    # ------------------------------------------------------------------
    # MAIN SIMULATION
    # ------------------------------------------------------------------
    def run_portfolio_backtest(self):
        print(f"\n[Portfolio Engine] Starting simulation | Capital: ₹{self.initial_capital:,.0f} | "
              f"Max positions: {self.max_positions} | Stop: {self.atr_stop_mult}×ATR | "
              f"Target: {self.atr_target_mult}×ATR")

        master_calendar = self._build_master_calendar()
        active_trades   = {}   # { ticker: trade_dict }
        trade_log       = []

        for current_date in master_calendar:

            # -------------------------------------------------------
            # A. UPDATE OPEN POSITION VALUES (for equity calculation)
            # -------------------------------------------------------
            open_value = 0.0
            for ticker, trade in active_trades.items():
                df = self.data_dict[ticker]
                if current_date in df.index:
                    price = df.loc[current_date, 'Close']
                    trade['last_price'] = price
                else:
                    price = trade['last_price']   # use last known price on non-trading days
                open_value += trade['shares'] * price

            total_equity = self.available_cash + open_value

            # -------------------------------------------------------
            # B. CHECK FOR EXITS (process before entries on same day)
            # -------------------------------------------------------
            exited = []

            for ticker, trade in active_trades.items():
                df = self.data_dict[ticker]
                if current_date not in df.index:
                    continue

                row         = df.loc[current_date]
                close_price = row['Close']
                low_price   = row['Low']
                atr         = row['ATR_14']
                days_held   = (current_date - trade['entry_date']).days

                # Update trailing stop: only move the stop UP, never down.
                # The stop follows the highest closing price the stock reached.
                if close_price > trade['peak_close']:
                    trade['peak_close'] = close_price
                    new_stop = close_price - (self.atr_stop_mult * atr)
                    if new_stop > trade['stop_price']:
                        trade['stop_price'] = new_stop

                trade['price_history'].append(close_price)

                is_exit     = False
                exit_reason = ""

                # --- EXIT 1: Logic Gate ("Bird in Hand" rule) ---
                # Every `gate_days` calendar days, check if we are sitting on
                # a decent profit. If yes, bank it rather than risk giving it back.
                #
                # GUARD ADDED (Bug 5 fix): also require close > stop_price,
                # so we don't gate-exit on a day where the stop was also nearly hit.
                if days_held > 0 and days_held % self.gate_days == 0:
                    current_return = (close_price - trade['entry_price']) / trade['entry_price']
                    if current_return >= self.min_gate_profit and close_price > trade['stop_price']:
                        is_exit, exit_reason = True, "Logic Gate"

                # --- EXIT 2: Take-Profit ---
                if not is_exit and close_price >= trade['take_profit_price']:
                    is_exit, exit_reason = True, "Target Hit"

                # --- EXIT 3: Trailing Stop-Loss ---
                # Triggered when the day's LOW breaches the stop level.
                # Exit price is the STOP PRICE, not the close. (Bug 1 fix)
                if not is_exit and low_price <= trade['stop_price']:
                    is_exit, exit_reason = True, "Stop-loss"

                if is_exit:
                    # BUG 1 FIX: use stop_price as exit when stop triggered,
                    # not close_price.
                    exit_price  = self._get_exit_price(exit_reason, close_price, trade['stop_price'])
                    exit_value  = trade['shares'] * exit_price
                    pnl_inr     = exit_value - trade['capital_deployed']
                    return_pct  = (pnl_inr / trade['capital_deployed']) * 100

                    # Max drawdown for this specific trade
                    max_dd_pct  = self._calc_max_drawdown(trade['price_history'], trade['entry_price'])

                    self.available_cash += exit_value

                    trade_log.append({
                        'Ticker'        : ticker,
                        'Entry Date'    : trade['entry_date'].strftime('%d %b %Y'),
                        'Entry Price ₹' : round(trade['entry_price'], 2),
                        'Exit Date'     : current_date.strftime('%d %b %Y'),
                        'Exit Price ₹'  : round(exit_price, 2),
                        'Confidence'    : trade['confidence'],
                        'Capital Used ₹': round(trade['capital_deployed'], 0),
                        'Return %'      : round(return_pct, 2),
                        'P&L ₹'        : round(pnl_inr, 2),
                        'Max Drawdown %': round(max_dd_pct, 2),
                        'Days Held'     : days_held,
                        'Exit Reason'   : exit_reason,
                    })
                    exited.append(ticker)

            for t in exited:
                del active_trades[t]

            # -------------------------------------------------------
            # C. CHECK FOR NEW ENTRIES
            # -------------------------------------------------------
            if len(active_trades) >= self.max_positions:
                continue   # portfolio is full, skip entry scan

            for ticker, df in self.data_dict.items():
                if ticker in active_trades:
                    continue
                if current_date not in df.index:
                    continue

                row = df.loc[current_date]
                if row['Signal'] != 1:
                    continue

                entry_price = row['Close']
                atr         = row['ATR_14'] if not pd.isna(row['ATR_14']) else entry_price * 0.01
                stop_dist   = self.atr_stop_mult * atr

                # Position sizing: risk 2% of total equity per trade,
                # but cap at an equal share of total equity across max_positions.
                max_allocation  = total_equity / self.max_positions
                risk_budget     = total_equity * 0.02
                shares          = risk_budget / stop_dist
                capital_needed  = min(shares * entry_price, max_allocation)

                if self.available_cash < capital_needed or capital_needed <= 0:
                    continue

                actual_shares        = capital_needed / entry_price
                self.available_cash -= capital_needed

                active_trades[ticker] = {
                    'entry_date'      : current_date,
                    'entry_price'     : entry_price,
                    'confidence'      : row.get('Confidence', 1.0),
                    'shares'          : actual_shares,
                    'capital_deployed': capital_needed,
                    'stop_price'      : entry_price - stop_dist,
                    'take_profit_price': entry_price + (self.atr_target_mult * atr),
                    'peak_close'      : entry_price,
                    'price_history'   : [entry_price],
                    'last_price'      : entry_price,
                }

                if len(active_trades) >= self.max_positions:
                    break   # filled up, stop scanning more tickers

        return pd.DataFrame(trade_log)

    # ------------------------------------------------------------------
    # INTERNAL HELPER: calculate the maximum intra-trade drawdown.
    # "How far did the position fall from its peak at any point while we held it?"
    # This is expressed as a % of the entry price (not portfolio equity).
    # ------------------------------------------------------------------
    def _calc_max_drawdown(self, price_history, entry_price):
        max_dd  = 0.0
        peak    = entry_price
        for price in price_history:
            if price > peak:
                peak = price
            dd = (price - peak) / peak * 100   # negative number
            if dd < max_dd:
                max_dd = dd
        return max_dd   # negative percentage, e.g. -12.5 means 12.5% drawdown

    # ------------------------------------------------------------------
    # PRINT SUMMARY — call this after run_portfolio_backtest()
    # ------------------------------------------------------------------
    def print_summary(self, trade_log_df):
        if trade_log_df.empty:
            print("\nNo trades were completed.")
            return

        final_value  = self.initial_capital + trade_log_df['P&L ₹'].sum()
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100
        wins         = trade_log_df[trade_log_df['Return %'] > 0]
        losses       = trade_log_df[trade_log_df['Return %'] <= 0]
        win_rate     = len(wins) / len(trade_log_df) * 100

        print("\n" + "=" * 65)
        print("                  PORTFOLIO PERFORMANCE SUMMARY")
        print("=" * 65)
        print(f"  Initial Capital  : ₹{self.initial_capital:>12,.2f}")
        print(f"  Final Value      : ₹{final_value:>12,.2f}")
        print(f"  Total Return     : {total_return:>+.2f}%")
        print("-" * 65)
        print(f"  Total Trades     : {len(trade_log_df)}")
        print(f"  Winning Trades   : {len(wins)}  ({win_rate:.1f}%)")
        print(f"  Losing Trades    : {len(losses)}  ({100 - win_rate:.1f}%)")
        if not wins.empty:
            print(f"  Avg Win          : {wins['Return %'].mean():.2f}%")
        if not losses.empty:
            print(f"  Avg Loss         : {losses['Return %'].mean():.2f}%")
        print(f"  Best Trade       : {trade_log_df['Return %'].max():.2f}%")
        print(f"  Worst Trade      : {trade_log_df['Return %'].min():.2f}%")
        print(f"  Avg Days Held    : {trade_log_df['Days Held'].mean():.0f} days")
        print(f"  Stop-loss exits  : {(trade_log_df['Exit Reason'] == 'Stop-loss').sum()}")
        print(f"  Logic Gate exits : {(trade_log_df['Exit Reason'] == 'Logic Gate').sum()}")
        print(f"  Target Hit exits : {(trade_log_df['Exit Reason'] == 'Target Hit').sum()}")
        print("=" * 65)
