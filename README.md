# Long/Short Z-Score Mean Reversion Strategy

A quantitative statistical arbitrage strategy applied to European large-cap equities using rolling Z-score signals to identify mean reversion opportunities.

## Strategy Overview

**Investment Thesis:** Stock prices tend to revert to their historical mean after periods of overextension. This strategy systematically exploits these mean reversion patterns using statistical signals.

**Core Methodology:**
- **Universe:** 40+ liquid European stocks (CAC 40, DAX, STOXX 600 constituents)
- **Signal:** 252-day rolling Z-scores identify statistical deviations
- **Entry Conditions:**
  - **Long:** Z-score < -1.5 (undervalued relative to history)
  - **Short:** Z-score > +1.5 (overvalued relative to history)
- **Exit Condition:** Z-score reverts to neutral band [-0.5, +0.5]
- **Rebalancing:** Monthly (first business day)
- **Position Sizing:** Equal-weighted across all active signals
- **Transaction Costs:** 10 basis points per trade

**Backtest Period:** January 2019 - January 2024 (5 years)

## Key Features

- Production-ready code with robust error handling and retry logic
- Real market data via Yahoo Finance API integration
- Comprehensive performance metrics: Sharpe, Sortino, Calmar, Win Rate, Profit Factor
- Professional visualizations: equity curves, drawdown, heatmaps, distributions
- Complete audit trail with trade log including prices and Z-scores
- Suitable for MFE and Quantitative Finance portfolios

## Project Structure
```
longshort-zscore-strategy/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git exclusions
├── longshort_zscore.py          # Main strategy implementation
├── test_installation.py         # Environment verification script
├── data/
│   └── cleaned_prices.csv       # Historical price data (generated)
├── results/
│   ├── equity_curve.csv         # Daily portfolio values
│   ├── positions.csv            # Position matrix (1/-1/0)
│   ├── trade_log.csv            # Complete trade history
│   ├── metrics.txt              # Performance summary
│   └── dashboard.png            # Visualization dashboard
└── notebooks/
    └── analysis.ipynb           # (Optional) Jupyter analysis
```

## Installation

### Prerequisites

- **Python:** 3.9 or 3.10 (3.11+ may have compatibility issues)
- **Operating System:** Windows, macOS, or Linux
- **Internet Connection:** Required for Yahoo Finance data download

### Setup Instructions

1. **Clone Repository:**
```bash
   git clone https://github.com/your-username/longshort-zscore-strategy.git
   cd longshort-zscore-strategy
```

2. **Create Virtual Environment:**
```bash
   python -m venv venv
   
   # Activate (choose your OS):
   source venv/bin/activate          # macOS/Linux
   venv\Scripts\activate             # Windows
```

3. **Install Dependencies:**
```bash
   pip install --upgrade pip
   pip install -r requirements.txt
```

4. **Verify Installation:**
```bash
   python test_installation.py
```
   
   Expected output: All checks should show [OK] or [PASS]

## Usage

### Run Strategy
```bash
python longshort_zscore.py
```

**Expected Runtime:** 2-5 minutes (depending on network speed)

**Console Output:**
```
============================================================
Z-SCORE LONG/SHORT STRATEGY INITIALIZATION
============================================================
Universe:       40 European stocks
Period:         2019-01-01 to 2024-01-01
Initial Capital: EUR 100,000
============================================================

[STEP 1/6] Downloading Historical Data...
[STEP 2/6] Calculating Z-Score Indicators...
[STEP 3/6] Generating Trading Signals...
[STEP 4/6] Running Backtest Engine...
[STEP 5/6] Computing Performance Metrics...
[STEP 6/6] Generating Visualizations...

EXECUTION COMPLETE
```

### View Results

After execution, check the `results/` directory:

- **`dashboard.png`:** Visual performance summary
- **`metrics.txt`:** Quantitative performance metrics
- **`equity_curve.csv`:** Daily portfolio values for external analysis
- **`trade_log.csv`:** Individual trade details

### Customize Parameters

Edit the strategy configuration in `longshort_zscore.py`:
```python
class ZScoreStrategy:
    def __init__(self, ...):
        self.lookback_window = 252       # Rolling window (days)
        self.entry_threshold = 1.5       # Entry Z-score threshold
        self.exit_threshold = 0.5        # Exit Z-score threshold
        self.transaction_cost = 0.0010   # Transaction cost (10 bps)
        self.rebalance_freq = 'MS'       # Rebalance frequency
```

## Expected Results

Based on 2019-2024 backtest (results may vary with market conditions):

| Metric | Typical Range |
|--------|--------------|
| **Total Return** | 15% - 35% |
| **CAGR** | 3% - 7% |
| **Sharpe Ratio** | 0.5 - 1.2 |
| **Max Drawdown** | -15% to -30% |
| **Win Rate** | 50% - 55% |

Note: Past performance does not guarantee future results.

## Troubleshooting

### Common Issues

**1. Package Installation Errors**
```bash
# Issue: pip fails to install packages
# Solution: Upgrade pip first
python -m pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

**2. Yahoo Finance Connection Timeout**
```bash
# Issue: "CRITICAL: No valid data could be retrieved"
# Solutions:
# - Check internet connection
# - Try again (Yahoo may rate-limit)
# - Reduce ticker list temporarily
```

**3. Empty Dataframe Errors**
```python
# Issue: "No data available" for specific tickers
# Solution: Tickers may be delisted or have data gaps
# The script automatically skips problematic tickers
# Ensure at least 10 tickers download successfully
```

**4. Memory Issues**
```bash
# Issue: Script crashes with large datasets
# Solution: Reduce ticker count or date range
# Edit main() function in longshort_zscore.py
```

### Data Quality Checks

Run these checks if results seem unusual:
```python
# In Python REPL after running strategy:
import pandas as pd

# Check price data
prices = pd.read_csv('data/cleaned_prices.csv', index_col=0, parse_dates=True)
print(prices.describe())
print(f"Missing values: {prices.isna().sum().sum()}")

# Check positions
positions = pd.read_csv('results/positions.csv', index_col=0, parse_dates=True)
print(f"Long positions: {(positions == 1).sum().sum()}")
print(f"Short positions: {(positions == -1).sum().sum()}")
```

## Academic Context

### Theoretical Foundation

This strategy implements concepts from:
- **Fama & French (1992):** Mean reversion in equity markets
- **Jegadeesh & Titman (1993):** Momentum and reversal effects
- **Lo & MacKinlay (1988):** Contrarian investment strategies

### Key Design Rationale

| Parameter | Value | Justification |
|-----------|-------|---------------|
| **Lookback Window** | 252 days | One trading year balances sensitivity vs. stability |
| **Entry Threshold** | ±1.5σ | Statistically significant deviation (93rd percentile) |
| **Exit Band** | ±0.5σ | Allows profit capture before full mean reversion |
| **Rebalancing** | Monthly | Reduces transaction costs vs. daily rebalancing |

### Suitable For

- Master in Financial Engineering (MFE) portfolios
- Quantitative Finance graduate applications
- Financial Data Science projects
- Algorithmic Trading course assignments

### Extensions for Research

Potential improvements for advanced projects:

- Sector Neutrality: Control for industry exposure
- Dynamic Thresholds: Optimize entry/exit based on volatility regime
- Machine Learning: Predict mean reversion probability
- Pair Trading: Apply Z-score logic to stock pairs
- Risk Parity: Volatility-weighted position sizing
- Regime Detection: Adapt strategy to market conditions

## Disclaimer

**For Educational Purposes Only**

This project is designed for academic demonstration and portfolio presentation. It is not intended for live trading or investment advice.

Key considerations:
- Historical performance does not guarantee future results
- Real-world trading involves slippage, market impact, and liquidity constraints
- Transaction costs may be higher in practice
- Regulatory and tax implications not modeled
- No warranty or guarantee of profitability

Always consult a licensed financial advisor before making investment decisions.

## License

This project is released under the MIT License. See LICENSE file for details.

## Contributing

Contributions are welcome. Areas for improvement:

- Additional risk metrics (VaR, CVaR)
- Alternative data sources (Quandl, Alpha Vantage)
- Portfolio optimization techniques
- Interactive dashboard (Plotly/Dash)

## Contact

Edouard Lavalard
