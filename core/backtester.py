# ==========================================
# FILE: core/backtester.py
# PURPOSE: Simulate trading and calculate PnL
#
# CHANGES FROM PREVIOUS VERSION:
#   1. Position sizing now uses Confidence column (fixes the 0.5/1.0 bug)
#   2. Signal_Display column added: 1 on entry day, 0 while holding, -1 on exit day
#   3. print_metrics() replaced with generate_trade_log() — per-trade table
#   4. ATR-based stop-loss added (configurable, default 2x ATR)
#   5. Max drawdown tracked per trade
#   6. Buy & Hold comparison REMOVED from output (per your request)
# ==========================================
import pandas as pd
import numpy as np


class SimpleBacktester:
    def __init__(self, data, initial_capital=100_000, atr_stop_multiplier=1.5):
        """
        Parameters
        ----------
        data                 : DataFrame from strategy.run()
        initial_capital      : Starting money in ₹
        atr_stop_multiplier  : Stop-loss = entry_price - (multiplier × ATR_14)
                               Default 2.0 means stop is 2 ATRs below entry.
                               Set to None to disable ATR stop-loss.
        """
        self.df                  = data.copy()
        self.initial_capital     = initial_capital
        self.atr_stop_multiplier = atr_stop_multiplier

    # ------------------------------------------------------------------
    # MAIN BACKTEST
    # ------------------------------------------------------------------
    def run_backtest(self):
        print("\nRunning Backtest Simulator...")

        df = self.df

        # ------------------------------------------------------------------
        # 1. Position column (forward-filled — used for PnL math)
        #    Uses Confidence so 0.5 = half capital, 1.0 = full capital.
        #    THIS IS THE FIX for the position sizing bug.
        # ------------------------------------------------------------------
        df['Position'] = np.nan
        df.loc[df['Signal'] == 1,  'Position'] = df.loc[df['Signal'] == 1,  'Confidence']
        df.loc[df['Signal'] == -1, 'Position'] = 0.0
        df.loc[df['Signal'] == 0,  'Position'] = 0.0
        df['Position'] = df['Position'].ffill().fillna(0.0)

        # ------------------------------------------------------------------
        # 2. Signal_Display: 1 on entry candle only, 0 while holding, -1 on exit
        #    This is for display/reporting only — NOT used for PnL.
        # ------------------------------------------------------------------
        df['Signal_Display'] = 0
        # Entry candles: where Signal == 1
        df.loc[df['Signal'] == 1, 'Signal_Display'] = 1
        # Exit candles: where Signal is 0 or -1 AND previous position was > 0
        prev_position = df['Position'].shift(1)
        exit_mask = ((df['Signal'] == 0) | (df['Signal'] == -1)) & (prev_position > 0)
        df.loc[exit_mask, 'Signal_Display'] = -1

        # ------------------------------------------------------------------
        # 3. ATR-based stop-loss (optional)
        #    Each day while in a trade: check if Close dropped below stop price.
        #    If yes: override Position to 0 (exit).
        # ------------------------------------------------------------------
        df['Stop_Price']    = np.nan
        df['Stop_Triggered'] = False

        # if self.atr_stop_multiplier is not None and 'ATR_14' in df.columns:
        #     print(f"  Applying ATR stop-loss ({self.atr_stop_multiplier}x ATR)...")
        #     in_trade    = False
        #     stop_price  = None
        #     entry_conf  = None

        #     for idx in df.index:
        #         sig   = df.loc[idx, 'Signal']
        #         close = df.loc[idx, 'Close']
        #         atr   = df.loc[idx, 'ATR_14']

        #         if not in_trade and sig == 1:
        #             in_trade   = True
        #             stop_price = close - (self.atr_stop_multiplier * atr)
        #             entry_conf = df.loc[idx, 'Confidence']
        #             df.loc[idx, 'Stop_Price'] = stop_price

        #         elif in_trade:
        #             df.loc[idx, 'Stop_Price'] = stop_price  # carry forward for display

        #             # Check: did the close hit the stop?
        #             if close <= stop_price:
        #                 df.loc[idx, 'Position']       = 0.0
        #                 df.loc[idx, 'Signal_Display'] = -1  # mark as exit
        #                 df.loc[idx, 'Stop_Triggered'] = True
        #                 in_trade   = False
        #                 stop_price = None
        #                 entry_conf = None
        #                 print(f"  Stop-loss triggered on {idx.date()} at ₹{close:.2f}")

        #             # Normal EMA exit
        #             elif (sig == 0) or (sig == -1):
        #                 in_trade   = False
        #                 stop_price = None
        #                 entry_conf = None


        # -------------------------------
        #  ---- TRAILING LOSS EXIT  ||  Risk : Reward Exit  -----
        # -------------------------------
        df['Stop_Price']     = np.nan
        df['Stop_Triggered'] = False
        df['TP_Triggered']   = False  # NEW: Track if we hit the target

        if self.atr_stop_multiplier is not None and 'ATR_14' in df.columns:
            print(f"  Applying Hard Take-Profit (4x ATR) & Stop-Loss ({self.atr_stop_multiplier}x ATR)...")
            in_trade          = False
            stop_price        = None
            take_profit_price = None

            for idx in df.index:
                sig   = df.loc[idx, 'Signal']
                close = df.loc[idx, 'Close']
                atr   = df.loc[idx, 'ATR_14']
                low   = df.loc[idx, 'Low']
                # --- 1. ENTRY ---
                if not in_trade and sig == 1:
                    in_trade          = True
                    highest_close     = close
                    stop_price        = highest_close - (self.atr_stop_multiplier * atr)
                    take_profit_price = close + (2.0* atr)  # 1:2 Risk/Reward Target
                    df.loc[idx, 'Stop_Price'] = stop_price

                # --- 2. HOLDING THE TRADE ---
                elif in_trade:
                    # INTRODUCING trailing stop (it moves upwards)
                   #  new_stop = close - (self.atr_stop_multiplier * atr)  ==> ONLY SEES EACH DAY CLOSE=> WRONG
                   
                   #  THIS NEW SYSTEM SEES HIGHEST CLOSE SEEN YET AND SELECTS BEST STOP TARGET HIT IN WORSE COND
                    highest_close = max(highest_close, close)
                    new_stop = highest_close - (self.atr_stop_multiplier * atr)
                    if new_stop > stop_price:
                        stop_price = new_stop

                    df.loc[idx, 'Stop_Price'] = stop_price  

                    # --- 3a. CHECK FOR TAKE-PROFIT EXIT (Target Hit!) ---
                    if close >= take_profit_price:
                        df.loc[idx, 'Position']       = 0.0
                        df.loc[idx, 'Signal_Display'] = -1  
                        df.loc[idx, 'TP_Triggered']   = True
                        in_trade          = False
                        stop_price        = None
                        take_profit_price = None
                        highest_close = None
                        print(f"  🎯 Target Hit! Take-profit triggered on {idx.date()} at ₹{close:.2f}")

                    # --- 3b. CHECK FOR STOP-LOSS EXIT ---
                    elif low <= stop_price:
                        df.loc[idx, 'Position']       = 0.0
                        df.loc[idx, 'Signal_Display'] = -1  
                        df.loc[idx, 'Stop_Triggered'] = True
                        in_trade          = False
                        stop_price        = None
                        take_profit_price = None
                        highest_close = None
                        print(f"  🛑 Stop-loss triggered on {idx.date()} at ₹{close:.2f}")

                    # --- 4. NORMAL EMA EXIT ---
                    elif (sig == 0) or (sig == -1):
                        in_trade          = False
                        stop_price        = None
                        take_profit_price = None
                        highest_close = None

        # ------------------------------------------------------------------
        # 4. Daily returns
        # ------------------------------------------------------------------
        df['Market_Returns']   = df['Close'].pct_change()
        df['Strategy_Returns'] = df['Market_Returns'] * df['Position'].shift(1)

        # Cumulative portfolio value
        df['Portfolio_Value']  = self.initial_capital * \
                                 (1 + df['Strategy_Returns']).cumprod()

        self.df = df
        return df

    # ------------------------------------------------------------------
    # TRADE LOG (replaces print_metrics)
    # ------------------------------------------------------------------
    def generate_trade_log(self):
        """
        Builds a per-trade table with:
          Entry Date, Entry Price, Exit Date, Exit Price,
          Confidence, Return %, Profit/Loss ₹, Max Drawdown %, Exit Reason, Days Held
        """
        df      = self.df
        capital = self.initial_capital
        trades  = []

        in_trade        = False
        entry_date      = None
        entry_price     = None
        entry_conf      = None
        capital_deployed = None
        daily_lows      = []   # track intra-trade drawdown

        for idx, row in df.iterrows():
            sig   = row['Signal']
            close = row['Close']
            stop  = row.get('Stop_Triggered', False)
            tp    = row.get('TP_Triggered', False)

           # --- ENTRY ---
            if not in_trade and sig == 1:
                in_trade         = True
                entry_date       = idx
                entry_price      = close
                entry_conf       = row['Confidence'] if not pd.isna(row['Confidence']) else 0.5
                
                # --- RISK-BASED POSITION SIZING ---
                # 1. Find the ATR (default to 1% of price if ATR is warming up)
                atr = row.get('ATR_14', close * 0.01)
                if pd.isna(atr) or atr == 0: 
                    atr = close * 0.01
                
                # 2. Calculate Stop-Loss Distance in ₹
                stop_distance = atr * self.atr_stop_multiplier
                
                # 3. We are willing to risk exactly 1% of our CURRENT capital
                risk_allowed = capital * 0.05
                
                # 4. How many shares can we safely buy?
                shares = risk_allowed / stop_distance
                
                # 5. How much cash does that cost?
                capital_deployed = shares * entry_price
                
                # 6. Safety Check: No margin trading. Can't spend more than we have!
                if capital_deployed > capital:
                    capital_deployed = capital
                    
                # 7. Apply Confidence Tier (0.5 confidence = risk half as much)
                capital_deployed = capital_deployed * entry_conf
                
                daily_lows       = [close]

            # --- WHILE IN TRADE ---
            elif in_trade:
                daily_lows.append(close)

                # --- EXIT (EMA cross, stop-loss, or signal -1) ---
                is_exit = (sig == 0) or (sig == -1) or stop or tp
                if is_exit:
                    exit_date  = idx
                    exit_price = close

                    # 1. Calculate actual money made/lost (INR) based on the stock's move
                    stock_pnl_pct = (exit_price - entry_price) / entry_price
                    pnl_inr       = capital_deployed * stock_pnl_pct
                    
                    # 2. PORTFOLIO Return % (How much did our ENTIRE account change?)
                    pnl_pct       = (pnl_inr / capital) * 100
                    
                    days_held     = (exit_date - entry_date).days

                    # 3. PORTFOLIO Max Drawdown % (How much of the total account was at risk?)
                    max_dd_pct = 0.0
                    running_max = entry_price
                    for p in daily_lows:
                        if p > running_max:
                            running_max = p
                        
                        # Calculate the stock's drop
                        stock_dd = (p - running_max) / running_max
                        # Convert that to actual rupees lost at that moment
                        money_lost = capital_deployed * stock_dd 
                        # Calculate what percentage of the total account that represents
                        account_dd_pct = (money_lost / capital) * 100
                        
                        if account_dd_pct < max_dd_pct:
                            max_dd_pct = account_dd_pct

                    exit_reason = "Target Hit" if tp else ("Stop-loss" if stop else ("EMA cross" if sig in [0, -1] else "Signal"))

                    trades.append({
                        'Entry Date'     : entry_date.strftime('%d %b %Y') if hasattr(entry_date, 'strftime') else entry_date,
                        'Entry Price ₹'  : round(entry_price, 2),
                        'Exit Date'      : exit_date.strftime('%d %b %Y') if hasattr(exit_date, 'strftime') else exit_date,
                        'Exit Price ₹'   : round(exit_price, 2),
                        'Confidence'     : entry_conf,
                        'Capital Used ₹' : round(capital_deployed, 0),
                        'Return %'       : round(pnl_pct, 2),       # <--- Now reports Portfolio % !
                        'P&L ₹'          : round(pnl_inr, 2),
                        'Max Drawdown %' : round(max_dd_pct, 2),    # <--- Now reports Portfolio DD% !
                        'Days Held'      : days_held,
                        'Exit Reason'    : exit_reason,
                    })

                    # --- COMPOUNDING ---
                    # Update our total capital with the profit or loss
                    capital += pnl_inr
                    
                    in_trade = False

        trades_df = pd.DataFrame(trades)
        return trades_df

    def print_trade_report(self):
        """Prints the full trade log + summary statistics."""
        trades_df = self.generate_trade_log()

        if trades_df.empty:
            print("No completed trades found.")
            return

        # --- FIX: Calculate final value by adding up all actual P&L from the log ---
        final_value   = self.initial_capital + trades_df['P&L ₹'].sum()
        total_return  = ((final_value - self.initial_capital) / self.initial_capital) * 100
        wins          = trades_df[trades_df['Return %'] > 0]
        losses        = trades_df[trades_df['Return %'] <= 0]
        win_rate      = (len(wins) / len(trades_df)) * 100

        print("\n" + "=" * 60)
        print("              TRADE REPORT")
        print("=" * 60)
        print(f"Initial Capital :  ₹{self.initial_capital:>12,.2f}")
        print(f"Final Value     :  ₹{final_value:>12,.2f}")
        print(f"Total Return    :  {total_return:>+.2f}%")
        print("-" * 60)
        print(f"Total Trades    :  {len(trades_df)}")
        print(f"Winning Trades  :  {len(wins)}  ({win_rate:.1f}%)")
        print(f"Losing Trades   :  {len(losses)}  ({100 - win_rate:.1f}%)")
        print(f"Avg Win         :  {wins['Return %'].mean():.2f}%")
        print(f"Avg Loss        :  {losses['Return %'].mean():.2f}%")
        print(f"Best Trade      :  {trades_df['Return %'].max():.2f}%")
        print(f"Worst Trade     :  {trades_df['Return %'].min():.2f}%")
        print(f"Avg Days Held   :  {trades_df['Days Held'].mean():.0f} days")
        print(f"Stop-loss exits :  {(trades_df['Exit Reason'] == 'Stop-loss').sum()}")
        print("=" * 60)
        print("\nPer-trade breakdown:\n")
        print(trades_df.to_string(index=False))
        print("=" * 60)

        return trades_df