# Z-Score Mean Reversion Strategy with VIX Filter Analysis

Long/short equity strategy testing statistical mean reversion on S&P 500 stocks with volatility regime filtering.

## Overview

This project implements and analyzes a market-neutral mean reversion strategy:
- Calculate rolling Z-scores to identify statistical mispricings
- Go long undervalued stocks (Z-score < -2.0)
- Go short overvalued stocks (Z-score > +2.0)
- Exit when prices revert to mean (|Z-score| < 0.2)

The strategy is tested with and without VIX-based regime filters to evaluate whether volatility filtering improves performance.

**Key Finding:** Neither the base strategy nor VIX filtering produces positive risk-adjusted returns across the tested periods.

## Strategy Details

**Core Parameters:**
- Universe: Top 50 large-cap S&P 500 stocks
- Lookback window: 20 days
- Entry threshold: |Z-score| > 2.0
- Exit threshold: |Z-score| < 0.2
- Portfolio: 5 long + 5 short positions (50/50 allocation)
- Rebalancing: Monthly (first week)
- Transaction costs: 1 basis point

**VIX Filter Thresholds Tested:**
- No filter (baseline)
- VIX < 20 (strict - low volatility only)
- VIX < 25 (moderate)
- VIX < 30 (permissive)

## Results

### Full Period (2019-2024)

| Strategy | Return | CAGR | Sharpe | Max DD | Volatility |
|----------|--------|------|--------|--------|------------|
| No Filter | -28.19% | -6.20% | -0.42 | -49.52% | 14.86% |
| VIX < 20 | -40.44% | -9.53% | -0.70 | -51.68% | 13.54% |
| VIX < 25 | -40.43% | -9.52% | -0.66 | -51.05% | 14.35% |
| VIX < 30 | -28.71% | -6.33% | -0.44 | -49.39% | 14.47% |

**Result:** All VIX filters worsen performance. VIX < 20 and VIX < 25 increase losses by over 40%.

### Training Period (2019-2020)

| Strategy | Return | CAGR | Sharpe | Max DD | Volatility |
|----------|--------|------|--------|--------|------------|
| No Filter | +0.06% | 0.34% | 0.05 | -4.53% | 7.47% |
| VIX < 20 | 0.00% | 0.00% | 0.00 | 0.00% | 0.00% |
| VIX < 25 | -9.53% | -43.05% | -2.53 | -15.43% | 17.04% |
| VIX < 30 | -1.00% | -5.47% | -0.81 | -4.69% | 6.71% |

**Result:** VIX < 20 stayed entirely in cash. Looser filters generated losses. Base strategy broke even.

### Bull Market Period (2019-2021)

| Strategy | Return | CAGR | Sharpe | Max DD | Volatility |
|----------|--------|------|--------|--------|------------|
| No Filter | +26.19% | 21.85% | 1.82 | -5.07% | 12.00% |
| VIX < 20 | +10.18% | 8.58% | 0.86 | -6.44% | 9.93% |
| VIX < 25 | +4.16% | 3.52% | 0.26 | -15.43% | 13.69% |
| VIX < 30 | +22.85% | 19.11% | 1.62 | -5.23% | 11.77% |

**Result:** Only period where base strategy succeeds (Sharpe 1.82). VIX filters reduce returns by 61-84%.

### Out-of-Sample Period (2021-2024)

| Strategy | Return | CAGR | Sharpe | Max DD | Volatility |
|----------|--------|------|--------|--------|------------|
| No Filter | -30.63% | -7.17% | -0.48 | -49.52% | 15.06% |
| VIX < 20 | -40.44% | -10.01% | -0.72 | -51.68% | 13.90% |
| VIX < 25 | -35.66% | -8.59% | -0.60 | -51.05% | 14.20% |
| VIX < 30 | -31.11% | -7.31% | -0.50 | -49.39% | 14.66% |

**Result:** Strategy fails. VIX filters worsen losses by 16-32%.

## Analysis

### Why the Strategy Fails

**1. Regime Mismatch**

Mean reversion requires prices to oscillate around stable means. This assumption breaks in trending markets:

- **2019-2021:** Low volatility bull market favors mean reversion (Sharpe 1.82)
- **2020:** COVID crash creates false mean reversion signals
- **2021:** Momentum rally invalidates reversion assumptions
- **2022:** Bear market trending down, no oscillation
- **2023-2024:** Tech concentration and AI momentum dominate

**2. Out-of-Sample Collapse**

Strong in-sample performance (2019-2021: Sharpe 1.82) completely fails out-of-sample (2021-2024: Sharpe -0.48). Classic overfitting pattern.

**3. VIX Filter Failure**

VIX filtering fails across all thresholds and periods:

| Period | Best VIX Threshold | vs Baseline | Conclusion |
|--------|-------------------|-------------|------------|
| 2019-2020 | None (baseline) | All filters worse | Filters add losses |
| 2019-2021 | VIX < 30 | -12.7% return | Still underperforms |
| 2021-2024 | VIX < 30 | -1.6% return | Marginal, still negative |
| 2019-2024 | VIX < 30 | -1.9% return | No improvement |

**Why VIX Filtering Fails:**

1. **Wrong diagnosis:** Problem is not volatility level but market structure (momentum vs. mean reversion)
2. **Timing issues:** VIX is reactive, not predictive. Rises after crashes begin, falls after recoveries start
3. **Misses recoveries:** Strict filters (VIX < 20, 25) keep strategy in cash during high-volatility rallies
4. **No alpha generation:** Filters can't fix a fundamentally broken signal

## Technical Implementation

**Requirements:**
```
pandas==2.2.2
numpy>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
openpyxl>=3.1.0
```

**Usage:**
```python
from longshort_zscore import ZScoreMeanReversionStrategy

strategy = ZScoreMeanReversionStrategy(
    tickers=['AAPL', 'MSFT', ...],
    start_date='2019-01-01',
    end_date='2024-12-31',
    lookback=20,
    n_long=5,
    n_short=5,
    entry_threshold=2.0,
    exit_threshold=0.2,
    vix_threshold=25.0  # or None for no filter
)

metrics = strategy.run('Stock_Data_History.xlsx', vix_filepath='VIXCLS.xlsx')
```

**Run full sensitivity analysis:**
```bash
python longshort_zscore.py
```

## Repository Structure
```
Long-Short-Z-Score-Strategy/
├── longshort_zscore.py           # Main strategy implementation
├── Stock_Data_History.xlsx       # Historical price data (50 tickers)
├── VIXCLS.xlsx                   # VIX historical data
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── results_2019-2021/            # Charts for bull market period
├── Results per period/           # Graphs for every tested period
└── LICENSE                       # MIT License
```

## Data Format

**Stock_Data_History.xlsx:**
- One sheet per ticker
- Column D: Date (DD/MM/YYYY)
- Column E: Close price

**VIXCLS.xlsx:** (Source: FRED)
- Column: observation_date (datetime)
- Column: VIXCLS (VIX close value)

**Tickers (50 stocks):**
Tech: NVDA, AAPL, MSFT, AMZN, GOOG, META, AVGO, TSLA, ORCL, AMD, PLTR, NFLX, CSCO, IBM, CRM, INTC, MU, LRCX, AMAT, KLAC
Financials: BRK.B, JPM, V, MA, BAC, MS, GS, WFC, AXP, C
Healthcare: LLY, JNJ, ABBV, UNH, MRK, TMO
Consumer: WMT, COST, HD, PG, KO, MCD, PM
Energy/Industrial: XOM, CVX, GE, CAT, RTX, LIN, TMUS

## Conclusion

This project demonstrates rigorous quantitative analysis:

1. Strategy implemented with proper transaction costs and realistic constraints
2. Multiple time periods tested (in-sample, out-of-sample, bull, bear, full)
3. Sensitivity analysis on key parameters (VIX thresholds)
4. Honest assessment of failure and understanding of root causes
5. Actionable recommendations for improvement

The strategy fails to generate positive risk-adjusted returns, and VIX filtering does not solve the underlying problem. However, the systematic analysis provides valuable insights into why mean reversion strategies struggle in momentum-driven markets and what characteristics a viable strategy would require.

## License

MIT

## Author

Edouard Lavalard  
