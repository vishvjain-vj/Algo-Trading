# nse_stock_updater.py
#
# PURPOSE:
#   Fetches the official NSE EQUITY_L.csv every 15 days and merges
#   new listings into nse_stocks_dynamic.json (used by search).
#   Falls back to the hardcoded nse_stocks.py list if fetch fails.
#
# SOURCE:
#   https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv
#   This is NSE's official complete listing file — updated daily by NSE.
#   Contains every actively listed equity including fresh IPOs.

import json
import os
import requests
import pandas as pd
import io
from datetime import datetime, timedelta

# ── Config ────────────────────────────────────────────────────────
NSE_EQUITY_URL  = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
DYNAMIC_FILE    = "nse_stocks_dynamic.json"   # auto-generated, lives in repo root
REFRESH_DAYS    = 15                           # how often to re-fetch from NSE

# Nifty50 symbols — these get popularity rank 1 (shown first in search)
# Keep this list updated if index composition changes
NIFTY50 = {
    "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","HINDUNILVR","SBIN",
    "BAJFINANCE","BHARTIARTL","KOTAKBANK","LT","AXISBANK","ASIANPAINT",
    "MARUTI","SUNPHARMA","TITAN","WIPRO","ULTRACEMCO","ONGC","NESTLEIND",
    "POWERGRID","NTPC","TATAMOTORS","TATASTEEL","HCLTECH","JSWSTEEL",
    "BAJAJFINSV","TECHM","INDUSINDBK","COALINDIA","DIVISLAB","DRREDDY",
    "EICHERMOT","GRASIM","HEROMOTOCO","HINDALCO","CIPLA","BPCL",
    "TATACONSUM","APOLLOHOSP","BRITANNIA","ADANIENT","ADANIPORTS",
    "BAJAJ-AUTO","SBILIFE","HDFCLIFE","SHRIRAMFIN","M&M","ITC","TRENT"
}

# Nifty Next 50 — popularity rank 2
NIFTY_NEXT50 = {
    "ZOMATO","IRCTC","DMART","PIDILITIND","SIEMENS","HAVELLS","DABUR",
    "MARICO","BERGEPAINT","COLPAL","GODREJCP","MUTHOOTFIN","CHOLAFIN",
    "PAGEIND","BANDHANBNK","FEDERALBNK","IDFCFIRSTB","PNB","BANKBARODA",
    "CANBK","SAIL","NMDC","VEDL","HINDZINC","RECLTD","PFC","IRFC",
    "MOTHERSON","BALKRISIND","APOLLOTYRE","MRF","BOSCHLTD","TVSMOTOR",
    "ASHOKLEY","ZYDUSLIFE","TORNTPHARM","LUPIN","AUROPHARMA","BIOCON",
    "LICHSGFIN","GAIL","IGL","PETRONET","IOC","HINDPETRO","TATAPOWER",
    "ADANIGREEN","ADANIPOWER","DLF","GODREJPROP","JUBLFOOD","DIXON",
    "POLYCAB","LTIM","LTTS","PERSISTENT","COFORGE","MPHASIS","ANGELONE",
    "MCX","CDSL","BSE","HDFCAMC","NIPPONLIFE","CAMS"
}


def _get_popularity(symbol: str) -> int:
    """Return popularity rank: 1=Nifty50, 2=NextNifty50, 3=others."""
    s = symbol.upper()
    if s in NIFTY50:
        return 1
    if s in NIFTY_NEXT50:
        return 2
    return 3


def _load_dynamic_file() -> dict:
    """Load the dynamic JSON file if it exists."""
    if os.path.exists(DYNAMIC_FILE):
        try:
            with open(DYNAMIC_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"last_updated": None, "stocks": []}


def _save_dynamic_file(data: dict):
    """Save the dynamic JSON file."""
    with open(DYNAMIC_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _fetch_nse_list() -> list:
    """
    Download EQUITY_L.csv from NSE archives.
    Returns list of (symbol, company_name, popularity_rank).
    Returns [] if fetch fails — caller handles fallback.
    """
    headers = {
        # NSE blocks plain Python requests; a browser User-Agent works
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://www.nseindia.com/",
    }

    try:
        session = requests.Session()
        # First hit the main site to get cookies (NSE requires this)
        session.get("https://www.nseindia.com", headers=headers, timeout=10)

        # Now fetch the CSV
        response = session.get(NSE_EQUITY_URL, headers=headers, timeout=15)
        response.raise_for_status()

        df = pd.read_csv(io.BytesIO(response.content))

        # EQUITY_L.csv columns: SYMBOL, NAME OF COMPANY, SERIES, ...
        # We only want EQ series (regular equity, not BE/SM/etc.)
        df.columns = [c.strip() for c in df.columns]
        df = df[df['SERIES'].str.strip() == 'EQ']

        stocks = []
        for _, row in df.iterrows():
            symbol   = str(row['SYMBOL']).strip()
            name     = str(row['NAME OF COMPANY']).strip()
            popularity = _get_popularity(symbol)
            stocks.append({
                "symbol":     symbol + ".NS",
                "name":       name,
                "popularity": popularity
            })

        return stocks

    except Exception as e:
        print(f"[NSE Updater] Fetch failed: {e}")
        return []


def needs_refresh() -> bool:
    """Returns True if 15+ days have passed since last update."""
    data = _load_dynamic_file()
    last = data.get("last_updated")
    if not last:
        return True
    try:
        last_dt = datetime.fromisoformat(last)
        return datetime.now() - last_dt > timedelta(days=REFRESH_DAYS)
    except Exception:
        return True


def refresh_stock_list(force: bool = False) -> bool:
    """
    Refresh the dynamic stock list from NSE if 15 days have passed.
    
    Args:
        force: if True, refresh regardless of time elapsed
    
    Returns:
        True if refresh succeeded, False if it failed (fallback used)
    """
    if not force and not needs_refresh():
        return True  # no refresh needed

    print("[NSE Updater] Fetching latest stock list from NSE...")
    stocks = _fetch_nse_list()

    if not stocks:
        print("[NSE Updater] Fetch failed. Using existing/fallback list.")
        return False

    data = {
        "last_updated": datetime.now().isoformat(),
        "stock_count":  len(stocks),
        "stocks":       stocks
    }
    _save_dynamic_file(data)
    print(f"[NSE Updater] ✅ Updated: {len(stocks)} stocks saved to {DYNAMIC_FILE}")
    return True


def get_all_stocks() -> list:
    """
    Get the full stock list for search.
    Priority: dynamic JSON → fallback to hardcoded nse_stocks.py
    
    Returns:
        List of (symbol, name, popularity_rank) tuples
    """
    data = _load_dynamic_file()
    stocks = data.get("stocks", [])

    if stocks:
        return [(s["symbol"], s["name"], s["popularity"]) for s in stocks]

    # Fallback to hardcoded list if dynamic file doesn't exist yet
    try:
        from nse_stocks import NSE_STOCKS
        return NSE_STOCKS
    except ImportError:
        return []


def get_last_updated() -> str:
    """Returns human-readable last update time."""
    data = _load_dynamic_file()
    last = data.get("last_updated")
    if not last:
        return "Never (using built-in list)"
    try:
        dt = datetime.fromisoformat(last)
        return dt.strftime("%d %b %Y, %I:%M %p")
    except Exception:
        return "Unknown"


def get_stock_count() -> int:
    """Returns number of stocks in dynamic list."""
    data = _load_dynamic_file()
    return data.get("stock_count", 0)
