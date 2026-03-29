# ==========================================
# FILE: core/data_fetcher.py
# PURPOSE: Universal data acquisition
# ==========================================
import yfinance as yf
import pandas as pd  # Make sure pandas is imported here

def fetch_daily_data(ticker, start_date, end_date):
    print(f"Fetching data for {ticker}...")
    data = yf.download(ticker, start=start_date, end=end_date)
    
    if data.empty:
        print(f"Warning: No data found for {ticker}.")
        return None
        
    # --- THE FIX IS HERE ---
    # If yfinance returns multi-level columns, flatten them to just the top level (Open, High, Low, Close, Volume)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    # -----------------------

    data.dropna(inplace=True)
    return data