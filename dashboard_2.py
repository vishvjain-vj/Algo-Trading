import streamlit as st
import pandas as pd
from nse_stocks import search_nse
from nse_stock_updater import refresh_stock_list, get_last_updated, get_stock_count, needs_refresh
from main_multiTicker import (WATCHLIST, INITIAL_CAPITAL, START, END,
                               FAST_SPAN, SLOW_SPAN, SLOPE_THRESHOLD,
                               ATR_STOP_MULT, ATR_TARGET_MULT,
                               MAX_POSITIONS, GATE_DAYS, MIN_GATE_PROFIT)
from core.data_fetcher import fetch_daily_data
from strategies.strategy_1_ema import EmaMarubozuStrategy
from core.backtester import PortfolioBacktester

st.set_page_config(page_title="Algo Trading Report", layout="wide")

# ------------------------------------------------------------------
# AUTO REFRESH NSE STOCK LIST (runs silently every 15 days)
# Only triggers if 15 days have passed — zero cost otherwise
# ------------------------------------------------------------------
@st.cache_data(ttl=60 * 60 * 24)   # check once per day max
def _trigger_nse_refresh():
    """Runs in background on app load. Silent if no refresh needed."""
    if needs_refresh():
        success = refresh_stock_list()
        return success
    return True

_trigger_nse_refresh()


def style_exit_reason(val):
    color_map = {
        'Stop-loss':  'background-color: #701c1c; color: white',
        'Target Hit': 'background-color: #1e4d2b; color: white',
        'Logic Gate': 'background-color: #1c4e70; color: white'
    }
    return color_map.get(val, '')


# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
st.title("📈 Multi-Ticker Backtest Report")
st.sidebar.header("Strategy Parameters")
st.sidebar.write(f"Capital: ₹{INITIAL_CAPITAL:,}")
st.sidebar.write(f"Stop: {ATR_STOP_MULT}x ATR")
st.sidebar.write(f"Target: {ATR_TARGET_MULT}x ATR")

# ── Stock List Status (shows last update + count) ─────────────────
st.sidebar.markdown("---")
count    = get_stock_count()
updated  = get_last_updated()

if count > 0:
    st.sidebar.caption(f"📋 Stock list: **{count:,} NSE stocks**")
    st.sidebar.caption(f"🔄 Last updated: {updated}")
else:
    st.sidebar.caption("📋 Using built-in stock list")

# Manual refresh button (for you/colleague to force update anytime)
if st.sidebar.button("🔄 Force Refresh Stock List"):
    with st.sidebar:
        with st.spinner("Fetching latest NSE list..."):
            ok = refresh_stock_list(force=True)
    if ok:
        st.sidebar.success(f"✅ Updated: {get_stock_count():,} stocks")
        st.cache_data.clear()
        st.rerun()
    else:
        st.sidebar.error("❌ Fetch failed. NSE site may be down. Try later.")

# ── WATCHLIST ──────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Watchlist")

if "watchlist" not in st.session_state:
    st.session_state.watchlist = WATCHLIST.copy()

st.sidebar.write("Current:")
for s in st.session_state.watchlist:
    st.sidebar.write(f"• {s}")

st.sidebar.markdown("---")

# ── SEARCH ────────────────────────────────────────────────────────
query = st.sidebar.text_input(
    "🔍 Search NSE Stock",
    placeholder="Type symbol or name e.g. rel, tata, bank"
)

if query and len(query) >= 1:
    # search_nse handles case, partial match, ranking automatically
    results = search_nse(query, max_results=10)

    if results:
        options = [f"{sym.replace('.NS', '')} — {name}" for sym, name in results]
        selected = st.sidebar.selectbox("Results (most popular first)", options)

        if st.sidebar.button("➕ Add to Watchlist"):
            raw    = selected.split(" — ")[0].strip()
            ticker = raw if raw.endswith(".NS") else raw + ".NS"

            if ticker not in st.session_state.watchlist:
                st.session_state.watchlist.append(ticker)
                st.sidebar.success(f"✅ Added {ticker}")
                st.rerun()
            else:
                st.sidebar.warning("⚠️ Already in watchlist")
    else:
        st.sidebar.info(
            f"No match for **'{query}'**\n\n"
            "Tips:\n"
            "• Try the NSE symbol: `TATASTEEL`, `HDFCBANK`\n"
            "• Partial works: `rel` finds RELIANCE\n"
            "• If it's a new IPO, click **Force Refresh Stock List** above"
        )

# ── REMOVE ────────────────────────────────────────────────────────
st.sidebar.markdown("---")
if st.session_state.watchlist:
    rem = st.sidebar.selectbox("🗑️ Remove ticker", st.session_state.watchlist)
    if st.sidebar.button("Remove"):
        st.session_state.watchlist.remove(rem)
        st.sidebar.success(f"Removed {rem}")
        st.rerun()

# ------------------------------------------------------------------
# MAIN AREA
# ------------------------------------------------------------------
st.markdown("---")
st.info(f"**Running backtest on:** {', '.join(st.session_state.watchlist)}")

if st.button("🚀 Run Backtest", type="primary"):

    if not st.session_state.watchlist:
        st.error("Watchlist is empty. Add at least one ticker.")
    else:
        processed_data = {}

        with st.status("Processing Tickers...", expanded=True) as status:
            for ticker in st.session_state.watchlist:
                st.write(f"📥 Fetching {ticker}...")
                try:
                    data = fetch_daily_data(ticker, start_date=START, end_date=END)
                    if data is not None and not data.empty:
                        strategy = EmaMarubozuStrategy(
                            data,
                            fast_span=FAST_SPAN,
                            slow_span=SLOW_SPAN,
                            slope_threshold=SLOPE_THRESHOLD,
                        )
                        processed_data[ticker] = strategy.run()
                        st.write(f"✅ {ticker} done")
                    else:
                        st.write(f"⚠️ {ticker} — no data, skipping")
                except Exception as e:
                    st.write(f"❌ {ticker} failed: {e}")
                    continue

            status.update(label="Backtest Complete!", state="complete")

        if not processed_data:
            st.error("No data returned. Check ticker symbols or date range.")
        else:
            backtester = PortfolioBacktester(
                processed_data, INITIAL_CAPITAL, ATR_STOP_MULT,
                ATR_TARGET_MULT, MAX_POSITIONS, GATE_DAYS, MIN_GATE_PROFIT
            )
            master_log = backtester.run_portfolio_backtest()

            if not master_log.empty:
                total_pnl = master_log['P&L ₹'].sum()
                win_rate  = (master_log['Return %'] > 0).mean() * 100

                m1, m2, m3 = st.columns(3)
                m1.metric("Total P&L",    f"₹{total_pnl:,.2f}",
                          delta=f"{total_pnl / INITIAL_CAPITAL * 100:.2f}%")
                m2.metric("Win Rate",     f"{win_rate:.1f}%")
                m3.metric("Total Trades", len(master_log))

                st.subheader("Trade Log")
                styled_log = master_log.style.map(
                    style_exit_reason, subset=['Exit Reason']
                )
                st.dataframe(styled_log, use_container_width=True, height=500)
            else:
                st.error("No trades executed. Check strategy filters or date range.")
