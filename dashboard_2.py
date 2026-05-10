import streamlit as st
import pandas as pd
import requests
from main_multiTicker import WATCHLIST, INITIAL_CAPITAL, START, END, FAST_SPAN, SLOW_SPAN, SLOPE_THRESHOLD, ATR_STOP_MULT, ATR_TARGET_MULT, MAX_POSITIONS, GATE_DAYS, MIN_GATE_PROFIT
from core.data_fetcher import fetch_daily_data
from strategies.strategy_1_ema import EmaMarubozuStrategy
from core.backtester import PortfolioBacktester

st.set_page_config(page_title="Algo Trading Report", layout="wide")
@st.cache_data(ttl=300)
def search_yahoo(query):
    url = "https://query1.finance.yahoo.com/v1/finance/search"
    params = {"q": query, "quotesCount": 8, "newsCount": 0}

    try:
        res = requests.get(url, params=params, timeout=3)
        data = res.json()

        results = []
        for item in data.get("quotes", []):
            symbol = item.get("symbol", "")
            name = item.get("shortname", "")

            # IMPORTANT: only NSE stocks (your system expects this)
            if symbol.endswith(".NS"):
                results.append((symbol, name))

        return results

    except:
        return []
        
        
def style_exit_reason(val):
    color_map = {
        'Stop-loss': 'background-color: #701c1c; color: white',
        'Target Hit': 'background-color: #1e4d2b; color: white',
        'Logic Gate': 'background-color: #1c4e70; color: white'
    }
    return color_map.get(val, '')

st.title("📈 Multi-Ticker Backtest Report")
st.sidebar.header("Strategy Parameters")
st.sidebar.write(f"Capital: ₹{INITIAL_CAPITAL:,}")
st.sidebar.write(f"Stop: {ATR_STOP_MULT}x ATR")
st.sidebar.write(f"Target: {ATR_TARGET_MULT}x ATR")

# ---------------- WATCHLIST UI (SAFE INTEGRATION) ----------------
st.sidebar.subheader("📊 Watchlist")

# Initialize once (uses your original WATCHLIST)
if "watchlist" not in st.session_state:
    st.session_state.watchlist = WATCHLIST.copy()

# Show current list
st.sidebar.write("Current:")
for s in st.session_state.watchlist:
    st.sidebar.write(f"• {s}")

# --- SEARCH ---
query = st.sidebar.text_input("🔍 Search (e.g. manaksteel)")

if query:
    results = search_yahoo(query)

    if results:
        options = [f"{sym} | {name}" for sym, name in results]
        selected = st.sidebar.selectbox("Results", options)

        if st.sidebar.button("Add"):
            ticker = selected.split(" | ")[0]

            if ticker not in st.session_state.watchlist:
                st.session_state.watchlist.append(ticker)
                st.sidebar.success(f"Added {ticker}")
            else:
                st.sidebar.warning("Already exists")
    else:
        st.sidebar.info("No NSE results found")

# --- REMOVE ---
if st.session_state.watchlist:
    rem = st.sidebar.selectbox("Remove", st.session_state.watchlist)
    if st.sidebar.button("Remove"):
        st.session_state.watchlist.remove(rem)
        
if st.button("Run Backtest"):
    processed_data = {}
    
    with st.status("Processing Tickers...", expanded=True) as status:
        for ticker in WATCHLIST:
            st.write(f"Fetching {ticker}...")
            data = fetch_daily_data(ticker, start_date=START, end_date=END)
            if data is not None and not data.empty:
# Use keyword arguments to ensure SLOPE_THRESHOLD doesn't overwrite the Trend Span
                strategy = EmaMarubozuStrategy(
                    data,
                    fast_span       = FAST_SPAN,
                    slow_span       = SLOW_SPAN,
                    slope_threshold = SLOPE_THRESHOLD,
                )
                processed_data[ticker] = strategy.run()
        status.update(label="Backtest Complete!", state="complete")

    backtester = PortfolioBacktester(
        processed_data, INITIAL_CAPITAL, ATR_STOP_MULT, 
        ATR_TARGET_MULT, MAX_POSITIONS, GATE_DAYS, MIN_GATE_PROFIT
    )
    
    master_log = backtester.run_portfolio_backtest()

    if not master_log.empty:
        # Metrics Row
        total_pnl = master_log['P&L ₹'].sum()
        win_rate = (master_log['Return %'] > 0).mean() * 100
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Total P&L", f"₹{total_pnl:,.2f}", delta=f"{total_pnl/INITIAL_CAPITAL*100:.2f}%")
        m2.metric("Win Rate", f"{win_rate:.1f}%")
        m3.metric("Total Trades", len(master_log))

        # Styled Table
        st.subheader("Trade Log")
        styled_log = master_log.style.map(style_exit_reason, subset=['Exit Reason'])
        
        st.dataframe(styled_log, use_container_width=True, height=500)
    else:
        st.error("No trades executed. Check Strategy filters.")
        

