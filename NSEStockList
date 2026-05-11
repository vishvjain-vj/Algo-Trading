# nse_stocks.py
#
# PURPOSE:
#   1. Hardcoded fallback list (~200 popular NSE stocks) used when
#      the dynamic list hasn't been fetched yet.
#   2. search_nse() — searches whichever list is available,
#      sorted by relevance + popularity (like a real search engine).

# ── Hardcoded fallback list ───────────────────────────────────────
# Format: (SYMBOL.NS, Company Name, popularity_rank)
# popularity_rank: 1=Nifty50, 2=NiftyNext50, 3=others
NSE_STOCKS = [
    # ── Nifty 50 ──────────────────────────────────────────────────
    ("RELIANCE.NS",    "Reliance Industries Ltd",            1),
    ("TCS.NS",         "Tata Consultancy Services Ltd",      1),
    ("HDFCBANK.NS",    "HDFC Bank Ltd",                      1),
    ("INFY.NS",        "Infosys Ltd",                        1),
    ("ICICIBANK.NS",   "ICICI Bank Ltd",                     1),
    ("HINDUNILVR.NS",  "Hindustan Unilever Ltd",             1),
    ("SBIN.NS",        "State Bank of India",                1),
    ("BAJFINANCE.NS",  "Bajaj Finance Ltd",                  1),
    ("BHARTIARTL.NS",  "Bharti Airtel Ltd",                  1),
    ("KOTAKBANK.NS",   "Kotak Mahindra Bank Ltd",            1),
    ("LT.NS",          "Larsen & Toubro Ltd",                1),
    ("AXISBANK.NS",    "Axis Bank Ltd",                      1),
    ("ASIANPAINT.NS",  "Asian Paints Ltd",                   1),
    ("MARUTI.NS",      "Maruti Suzuki India Ltd",            1),
    ("SUNPHARMA.NS",   "Sun Pharmaceutical Industries Ltd",  1),
    ("TITAN.NS",       "Titan Company Ltd",                  1),
    ("WIPRO.NS",       "Wipro Ltd",                          1),
    ("ULTRACEMCO.NS",  "UltraTech Cement Ltd",               1),
    ("ONGC.NS",        "Oil & Natural Gas Corporation Ltd",  1),
    ("NESTLEIND.NS",   "Nestle India Ltd",                   1),
    ("POWERGRID.NS",   "Power Grid Corporation of India",    1),
    ("NTPC.NS",        "NTPC Ltd",                           1),
    ("TATAMOTORS.NS",  "Tata Motors Ltd",                    1),
    ("TATASTEEL.NS",   "Tata Steel Ltd",                     1),
    ("HCLTECH.NS",     "HCL Technologies Ltd",               1),
    ("JSWSTEEL.NS",    "JSW Steel Ltd",                      1),
    ("BAJAJFINSV.NS",  "Bajaj Finserv Ltd",                  1),
    ("TECHM.NS",       "Tech Mahindra Ltd",                  1),
    ("INDUSINDBK.NS",  "IndusInd Bank Ltd",                  1),
    ("COALINDIA.NS",   "Coal India Ltd",                     1),
    ("DIVISLAB.NS",    "Divi's Laboratories Ltd",            1),
    ("DRREDDY.NS",     "Dr Reddy's Laboratories Ltd",        1),
    ("EICHERMOT.NS",   "Eicher Motors Ltd",                  1),
    ("GRASIM.NS",      "Grasim Industries Ltd",              1),
    ("HEROMOTOCO.NS",  "Hero MotoCorp Ltd",                  1),
    ("HINDALCO.NS",    "Hindalco Industries Ltd",            1),
    ("CIPLA.NS",       "Cipla Ltd",                          1),
    ("BPCL.NS",        "Bharat Petroleum Corporation Ltd",   1),
    ("TATACONSUM.NS",  "Tata Consumer Products Ltd",         1),
    ("APOLLOHOSP.NS",  "Apollo Hospitals Enterprise Ltd",    1),
    ("BRITANNIA.NS",   "Britannia Industries Ltd",           1),
    ("ADANIENT.NS",    "Adani Enterprises Ltd",              1),
    ("ADANIPORTS.NS",  "Adani Ports & SEZ Ltd",              1),
    ("BAJAJ-AUTO.NS",  "Bajaj Auto Ltd",                     1),
    ("SBILIFE.NS",     "SBI Life Insurance Company Ltd",     1),
    ("HDFCLIFE.NS",    "HDFC Life Insurance Company Ltd",    1),
    ("SHRIRAMFIN.NS",  "Shriram Finance Ltd",                1),
    ("M&M.NS",         "Mahindra & Mahindra Ltd",            1),
    ("ITC.NS",         "ITC Ltd",                            1),
    ("TRENT.NS",       "Trent Ltd",                          1),

    # ── Nifty Next 50 / Popular Midcap ────────────────────────────
    ("ZOMATO.NS",      "Zomato Ltd",                         2),
    ("IRCTC.NS",       "Indian Railway Catering & Tourism",  2),
    ("DMART.NS",       "Avenue Supermarts Ltd",              2),
    ("PIDILITIND.NS",  "Pidilite Industries Ltd",            2),
    ("SIEMENS.NS",     "Siemens Ltd",                        2),
    ("HAVELLS.NS",     "Havells India Ltd",                  2),
    ("DABUR.NS",       "Dabur India Ltd",                    2),
    ("MARICO.NS",      "Marico Ltd",                         2),
    ("BERGEPAINT.NS",  "Berger Paints India Ltd",            2),
    ("COLPAL.NS",      "Colgate-Palmolive (India) Ltd",      2),
    ("GODREJCP.NS",    "Godrej Consumer Products Ltd",       2),
    ("MUTHOOTFIN.NS",  "Muthoot Finance Ltd",                2),
    ("CHOLAFIN.NS",    "Cholamandalam Investment",           2),
    ("PAGEIND.NS",     "Page Industries Ltd",                2),
    ("BANDHANBNK.NS",  "Bandhan Bank Ltd",                   2),
    ("FEDERALBNK.NS",  "The Federal Bank Ltd",               2),
    ("IDFCFIRSTB.NS",  "IDFC First Bank Ltd",                2),
    ("PNB.NS",         "Punjab National Bank",               2),
    ("BANKBARODA.NS",  "Bank of Baroda",                     2),
    ("CANBK.NS",       "Canara Bank",                        2),
    ("SAIL.NS",        "Steel Authority of India Ltd",       2),
    ("NMDC.NS",        "NMDC Ltd",                           2),
    ("VEDL.NS",        "Vedanta Ltd",                        2),
    ("HINDZINC.NS",    "Hindustan Zinc Ltd",                 2),
    ("RECLTD.NS",      "REC Ltd",                            2),
    ("PFC.NS",         "Power Finance Corporation Ltd",      2),
    ("IRFC.NS",        "Indian Railway Finance Corporation", 2),
    ("TATAPOWER.NS",   "Tata Power Company Ltd",             2),
    ("ADANIGREEN.NS",  "Adani Green Energy Ltd",             2),
    ("DLF.NS",         "DLF Ltd",                            2),
    ("GODREJPROP.NS",  "Godrej Properties Ltd",              2),
    ("JUBLFOOD.NS",    "Jubilant Foodworks Ltd",             2),
    ("DIXON.NS",       "Dixon Technologies Ltd",             2),
    ("POLYCAB.NS",     "Polycab India Ltd",                  2),
    ("LTIM.NS",        "LTIMindtree Ltd",                    2),
    ("LTTS.NS",        "L&T Technology Services Ltd",        2),
    ("PERSISTENT.NS",  "Persistent Systems Ltd",             2),
    ("COFORGE.NS",     "Coforge Ltd",                        2),
    ("MPHASIS.NS",     "Mphasis Ltd",                        2),
    ("ANGELONE.NS",    "Angel One Ltd",                      2),
    ("MCX.NS",         "Multi Commodity Exchange of India",  2),
    ("CDSL.NS",        "Central Depository Services Ltd",    2),
    ("BSE.NS",         "BSE Ltd",                            2),
    ("HDFCAMC.NS",     "HDFC Asset Management Company",      2),
    ("GAIL.NS",        "GAIL (India) Ltd",                   2),
    ("IGL.NS",         "Indraprastha Gas Ltd",               2),
    ("IOC.NS",         "Indian Oil Corporation Ltd",         2),
    ("TVSMOTOR.NS",    "TVS Motor Company Ltd",              2),
    ("ASHOKLEY.NS",    "Ashok Leyland Ltd",                  2),
    ("LUPIN.NS",       "Lupin Ltd",                          2),
    ("AUROPHARMA.NS",  "Aurobindo Pharma Ltd",               2),
    ("TORNTPHARM.NS",  "Torrent Pharmaceuticals Ltd",        2),
    ("MRF.NS",         "MRF Ltd",                            2),
    ("SUZLON.NS",      "Suzlon Energy Ltd",                  2),
    ("MANAKSTEEL.NS",  "Man Industries (India) Ltd",         3),
    ("RAYMOND.NS",     "Raymond Ltd",                        3),
    ("TRIDENT.NS",     "Trident Ltd",                        3),
    ("DEEPAKFERT.NS",  "Deepak Fertilisers & Petrochemicals",3),
    ("DEEPAKNTR.NS",   "Deepak Nitrite Ltd",                 3),
    ("AARTIIND.NS",    "Aarti Industries Ltd",               3),
    ("NAVINFLUOR.NS",  "Navin Fluorine International",       3),
    ("VINATIORGA.NS",  "Vinati Organics Ltd",                3),
    ("TATAELXSI.NS",   "Tata Elxsi Ltd",                     3),
    ("KPITTECH.NS",    "KPIT Technologies Ltd",              3),
    ("CYIENT.NS",      "Cyient Ltd",                         3),
    ("VOLTAS.NS",      "Voltas Ltd",                         3),
    ("CROMPTON.NS",    "Crompton Greaves Consumer",          3),
    ("KEI.NS",         "KEI Industries Ltd",                 3),
    ("APLAPOLLO.NS",   "APL Apollo Tubes Ltd",               3),
    ("BALKRISIND.NS",  "Balkrishna Industries Ltd",          3),
    ("APOLLOTYRE.NS",  "Apollo Tyres Ltd",                   3),
    ("INDHOTEL.NS",    "The Indian Hotels Company Ltd",      3),
    ("PRESTIGE.NS",    "Prestige Estates Projects Ltd",      3),
    ("OBEROIRLTY.NS",  "Oberoi Realty Ltd",                  3),
]


def search_nse(query: str, max_results: int = 10) -> list:
    """
    Search NSE stocks — case insensitive, works from 1st character.
    Uses dynamic list (auto-updated every 15 days) if available,
    falls back to hardcoded list above.

    Ranking:
        Tier 0 — exact symbol match       (highest priority)
        Tier 1 — symbol starts with query
        Tier 2 — name starts with query
        Tier 3 — symbol contains query
        Tier 4 — name contains query
    Within each tier, sorted by popularity rank (Nifty50 first).

    Args:
        query:       user input (any case, partial ok)
        max_results: max results to return

    Returns:
        List of (symbol, company_name) tuples
    """
    q = query.strip().upper()
    if not q:
        return []

    # Use dynamic list if available, else hardcoded fallback
    try:
        from NSEStockUpdater import get_all_stocks
        stock_list = get_all_stocks()
    except ImportError:
        stock_list = NSE_STOCKS

    if not stock_list:
        stock_list = NSE_STOCKS

    scored   = []
    seen     = set()

    for item in stock_list:
        symbol, name, popularity = item[0], item[1], item[2]

        sym_clean  = symbol.replace(".NS", "").upper()
        name_upper = name.upper()

        # Skip duplicates (dynamic list may have repeats)
        if sym_clean in seen:
            continue

        # ── Tier scoring ──────────────────────────────────────────
        if sym_clean == q:
            tier = 0                          # exact symbol match
        elif sym_clean.startswith(q):
            tier = 1                          # symbol starts with
        elif name_upper.startswith(q):
            tier = 2                          # name starts with
        elif q in sym_clean:
            tier = 3                          # symbol contains
        elif q in name_upper:
            tier = 4                          # name contains
        else:
            continue                          # no match — skip

        seen.add(sym_clean)
        scored.append((tier, popularity, symbol, name))

    # Sort: tier first (lower = better), then popularity (lower = better)
    scored.sort(key=lambda x: (x[0], x[1]))

    return [(sym, name) for _, _, sym, name in scored[:max_results]]
