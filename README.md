# Algorithmic Trading Backtesting & Optimisation Engine

A fully modular, end-to-end algorithmic trading research framework built in Python. Takes raw historical market data, applies a multi-confirmation trend-following strategy, manages risk institutionally, and outputs a professional performance report — all from scratch, no third-party strategy libraries.

> **Status:** Phase 1 Complete — EOD backtesting engine fully operational. Phase 2 (intraday + advanced exits) in progress.

---

## What This Project Does

Most retail trading is guesswork. This system replaces guesswork with a strict, rule-based pipeline:

```
Raw OHLCV Data → Strategy Signals → Risk-Sized Position → Backtest Simulation → Performance Report
```

Every decision — when to enter, how much capital to deploy, where to place the stop, when to exit — is encoded as an explicit mathematical rule and tested against 7 years of real NSE market data.

---

## Architecture Overview

```
├── data_fetcher.py       # Data ingestion via yfinance (NSE/BSE tickers)
├── strategy_1_ema.py     # Signal engine: EMA crossover + Marubozu filter
├── backtester.py         # Simulation core: compounding equity, stops, trade log
├── optimizer.py          # Grid search: 80+ parameter combos ranked by Sharpe
└── main.py               # Entry point / dispatcher
```

---

## Phase 1 — What's Built

### 1. Strategy Engine (Entry Logic)

A multi-layer filter designed to only enter **high-probability trades in the direction of the dominant trend**.

**Layer 1 — Macro Trend Filter (200 EMA)**
- Acts as a hard gate. Long entries are strictly forbidden if price is below the 200 EMA.
- Eliminates counter-trend trades that look good short-term but fail against the larger cycle.

**Layer 2 — Momentum Trigger (Fast/Slow EMA Crossover)**
- Detects the exact day a new short-term wave begins *within* the macro trend.
- Default: 10 EMA crossing above 20 EMA. Fully parameterisable.
- Signal fires only on the **crossover day**, not continuously — prevents false holding.

**Layer 3 — Volume/Conviction Confirmation (Marubozu Candle)**
- Requires the entry candle body to be ≥ 75% of the day's High–Low range.
- Filters out weak crossovers not backed by real institutional buying pressure.
- Result: 3 independent conditions must align simultaneously to enter a trade.

**Signal Confidence Tiers**

| Condition | Capital Deployed |
|-----------|-----------------|
| EMA crossover only (trend-aligned) | 50% |
| Crossover + Marubozu within 5 days | 100% |

---

### 2. Risk Management System

**ATR Trailing Stop-Loss**
- Stop placed at `Entry Price − (N × ATR_14)` — sized to the stock's actual daily volatility, not an arbitrary percentage.
- Trails upward as the trade profits, never downward — locks in gains automatically.
- Built on a **peak-tracking mechanism**: recalculates from the highest close seen since entry.

**3:1 Risk-Reward Take-Profit**
- Hard target at `Entry Price + (3 × ATR)`.
- Ensures long-term profitability even at a 40% win rate.

**Confidence-Weighted Position Sizing**
- Capital deployed scales with signal conviction tier (50% or 100%).
- Prevents maximum capital exposure on weak signals.

**Result:** Max portfolio drawdown reduced from **−44% → −4.5%** purely through position sizing discipline.

---

### 3. Backtesting Core

A custom time-series simulation loop — no black-box strategy frameworks.

- **Compounding equity curve**: starts at ₹1,00,000 and compounds trade-by-trade.
- **Accurate P&L math**: portfolio return derived from the equity curve directly, not by summing raw trade P&L — prevents the common compounding error.
- **Per-trade log output**:

| Field | Description |
|-------|-------------|
| Entry / Exit Date | Exact dates |
| Entry / Exit Price ₹ | Execution prices |
| Capital Deployed ₹ | Sized per confidence tier |
| Return % | Per-trade percentage |
| P&L ₹ | Absolute rupee gain/loss |
| Max Drawdown % | Worst intra-trade dip |
| Days Held | Trade duration |
| Exit Reason | EMA cross / Stop-loss / Target Hit |

---

### 4. Parameter Optimiser

Brute-force grid search across the full parameter space — no hand-tuning.

**Parameter Grid**
```
Fast EMA  : [5, 8, 10, 12, 15]
Slow EMA  : [15, 20, 25, 30]
ATR Stop  : [0.5×, 1.0×, 1.5×, 2.0×]
→ 80+ valid combinations tested per run
```

**Ranking**
Results ranked by **Sharpe Ratio and Max Drawdown**, not raw return. The highest-return combo in a backtest is almost always the most overfitted. Optimising for risk-adjusted metrics selects parameters more likely to hold out-of-sample.

Output tables:
- Top 10 by Total Return
- Top 10 by Sharpe Ratio *(recommended selection criterion)*
- Top 10 by Win Rate (min 4 trades)

---

## How to Run

**1. Install dependencies**
```bash
pip install yfinance pandas numpy
```

**2. Run the main backtest**
```bash
python main.py
```

**3. Run the parameter optimiser**
```bash
python optimizer.py
```

**4. Configuration** (inside `main.py` / `optimizer.py`)
```python
TICKER          = "RELIANCE.NS"   # Any NSE/BSE ticker
START           = "2019-01-01"
END             = "2026-01-01"
INITIAL_CAPITAL = 100_000         # ₹
ATR_STOP        = 2.0             # ATR multiplier for stop-loss
```

---

## What's Next — Phase 2

- **Time Stops**: Exit trades after N days regardless of signal — frees dead capital in sideways moves.
- **Slope-Based Exits**: Exit when Fast EMA slope flattens — captures momentum exhaustion earlier than a crossover.
- **Intraday Scaling**: Migrate from EOD candles to 15-min / 1-hour for more trade opportunities.

---

## Tech Stack

| Library | Use |
|---------|-----|
| `yfinance` | Historical OHLCV data (NSE/BSE) |
| `pandas` | Time-series manipulation, signal logic |
| `NumPy` | ATR calculation, vectorised operations |
| `itertools` | Parameter grid generation |

---

## Disclaimer

Built for **educational and research purposes only**. Past backtested performance does not guarantee future results. Not financial advice.
