# Long/Short Z-Score Screening Strategy

A quantitative statistical arbitrage strategy applied to European large-cap equities using rank-based Z-score screening to identify mean reversion opportunities.

## Strategy Overview

**Investment Thesis:** Stock prices exhibit temporary deviations from their historical mean. This strategy systematically exploits these mean reversion patterns using statistical ranking methodology.

**Core Methodology:**
- **Universe:** 40+ liquid European stocks (CAC 40, DAX, STOXX 600 constituents)
- **Signal:** 252-day rolling Z-scores measure statistical deviations
- **Screening Process:**
  1. Calculate Z-scores for all securities at rebalancing date
  2. **Rank** all securities from most undervalued to most overvalued
  3. **Select** top 10 undervalued stocks for long positions
  4. **Select** top 10 overvalued stocks for short positions
- **Position Sizing:** Equal-weighted allocation (50% long side, 50% short side)
- **Rebalancing:** Monthly (first business day)
- **Transaction Costs:** 10 basis points per trade

**Backtest Period:** January 2019 - January 2024 (5 years)

## Key Features

- Production-ready code with robust error handling and retry logic
- Real market data via Yahoo Finance API integration
- Rank-based screening methodology (not threshold-based)
- Comprehensive statistical validation: stationarity, normality, correlation
- Performance metrics: Sharpe, Sortino, Calmar, Win Rate, Profit Factor
- Professional visualizations including correlation heatmaps
- Complete audit trail with detailed trade logs
- Academic rigor suitable for MFE portfolios

## Project Structurelongshort-zscore-strategy/
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git exclusions
├── longshort_zscore.py             # Main strategy implementation
├── test_installation.py            # Environment verification script
├── data/
│   └── cleaned_prices.csv          # Historical price data (generated)
├── results/
│   ├── equity_curve.csv            # Daily portfolio values
│   ├── positions.csv               # Position matrix (1/-1/0)
│   ├── trade_log.csv               # Complete trade history
│   ├── metrics.txt                 # Performance summary
│   ├── statistical_tests.txt       # Statistical validation results
│   └── dashboard.png               # Visualization dashboard
└── notebooks/

**For Educational and Research Purposes Only**

This project is designed for academic demonstration, portfolio presentation, and quantitative research. It is not intended as investment advice or for live trading without significant additional development.

**Important Considerations:**
- Historical performance does not guarantee future results
- Real-world trading involves additional frictions not fully modeled:
  - Market impact and slippage
  - Borrowing costs for short positions
  - Margin requirements and collateral management
  - Regulatory constraints and compliance costs
- Statistical relationships may degrade due to:
  - Strategy capacity constraints
  - Increased market efficiency
  - Structural regime changes
- No warranty or guarantee of profitability
- Not financial advice - consult licensed professionals

**Regulatory Notice:**
This strategy involves short-selling, which is subject to regulatory restrictions in various jurisdictions. Ensure compliance with local regulations before any implementation.

## License

This project is released under the MIT License. See LICENSE file for details.

## Author

**Edouard Lavalard** 
