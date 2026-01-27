# Z-Score Mean Reversion Strategy

Statistical arbitrage strategy using Z-score based mean reversion on S&P 500 stocks.

## Overview

This strategy identifies overvalued and undervalued stocks using rolling Z-scores, entering long positions on undervalued stocks (Z < -2.0) and short positions on overvalued stocks (Z > +2.0). Positions are closed when prices revert to the mean (|Z| < 0.2).

## Strategy Logic

- **Entry**: |Z-score| > 2.0 (extreme deviation from mean)
- **Exit**: |Z-score| < 0.2 (reversion to mean)
- **Lookback**: 20-day rolling window
- **Portfolio**: 5 long + 5 short positions
- **Rebalancing**: Monthly (first week of month)

## Usage
```python
from zscore_mean_reversion import ZScoreMeanReversionStrategy

strategy = ZScoreMeanReversionStrategy(
    tickers=['AAPL', 'MSFT', 'GOOGL', ...],
    start_date='2019-01-01',
    end_date='2021-12-31',
    lookback=20,
    n_long=5,
    n_short=5
)

metrics = strategy.run('Stock_Data_History.xlsx')
```

## Data Format

Excel file with one sheet per ticker containing:
- **Date** column (DD/MM/YYYY format)
- **Close** column (price data)

## Output

- **Metrics**: Returns, Sharpe, Sortino, drawdown, win rate
- **Visualization**: Equity curve, drawdown, Z-score distribution, returns histogram

## Requirements
```bash
pip install -r requirements.txt
```

## License

MIT
