"""
Long/Short Z-Score Screening Strategy
European Large-Cap Equity Universe (2019-2024)

Strategy Logic:
- Calculate 252-day rolling Z-scores for each stock
- RANK all stocks by Z-score at each rebalancing date
- Long top N stocks with lowest Z-scores (most undervalued)
- Short top N stocks with highest Z-scores (most overvalued)
- Rebalancing: Monthly (first business day)
- Transaction costs: 10 basis points per trade

Statistical Validation:
- Stationarity tests (Augmented Dickey-Fuller)
- Normality tests (Jarque-Bera)
- Correlation analysis
"""

import os
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller

# Configuration
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

class ZScoreScreeningStrategy:
    """
    Statistical arbitrage strategy using Z-score ranking for security selection.
    Implements proper screening methodology with ranking and top-N selection.
    """
    
    def __init__(self, tickers, start_date, end_date, initial_capital=100000, 
                 n_long=10, n_short=10):
        """
        Initialize strategy parameters.
        
        Args:
            tickers: List of ticker symbols (Yahoo Finance format)
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)
            initial_capital: Starting portfolio value in EUR
            n_long: Number of long positions in portfolio
            n_short: Number of short positions in portfolio
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.n_long = n_long
        self.n_short = n_short
        
        # Strategy Parameters
        self.lookback_window = 252      # Rolling window for Z-score (1 year)
        self.transaction_cost = 0.0010  # 10 basis points
        self.rebalance_freq = 'MS'      # Monthly start frequency
        
        # Data containers
        self.data = pd.DataFrame()
        self.z_scores = pd.DataFrame()
        self.positions = pd.DataFrame()
        self.portfolio_value = pd.Series(dtype=float)
        self.daily_returns = pd.Series(dtype=float)
        self.trades = pd.DataFrame()
        self.metrics = {}
        self.statistical_tests = {}
        
        # Create output directories
        os.makedirs('data', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"Z-SCORE SCREENING STRATEGY INITIALIZATION")
        print(f"{'='*70}")
        print(f"Universe:         {len(tickers)} European stocks")
        print(f"Period:           {start_date} to {end_date}")
        print(f"Initial Capital:  EUR {initial_capital:,.0f}")
        print(f"Portfolio Size:   {n_long} Long + {n_short} Short = {n_long + n_short} positions")
        print(f"Screening Method: Rank-based selection (top/bottom N by Z-score)")
        print(f"{'='*70}\n")
    
    def download_data(self, max_retries=3, retry_delay=2):
        """
        Download historical adjusted close prices with robust error handling.
        Uses Ticker.history() method for better reliability.
        """
        print("[STEP 1/7] Downloading Historical Data...")
        print("-" * 70)
        print("  This may take 5-10 minutes, please wait...")
        
        valid_data = {}
        failed_tickers = []
        
        # Calculate minimum required data points
        date_range = pd.date_range(self.start_date, self.end_date, freq='B')
        min_data_points = int(len(date_range) * 0.70)
        
        for i, ticker in enumerate(self.tickers, 1):
            success = False
            
            for attempt in range(1, max_retries + 1):
                try:
                    # Use Ticker.history() method instead of yf.download()
                    stock = yf.Ticker(ticker)
                    df = stock.history(start=self.start_date, end=self.end_date, auto_adjust=True)
                    
                    if df.empty:
                        print(f"  [{i:2d}/{len(self.tickers)}] {ticker:12} - No data available")
                        break
                    
                    if len(df) < min_data_points:
                        print(f"  [{i:2d}/{len(self.tickers)}] {ticker:12} - Insufficient data ({len(df)} points)")
                        break
                    
                    # Use 'Close' instead of 'Adj Close' because auto_adjust=True
                    valid_data[ticker] = df['Close']
                    print(f"  [{i:2d}/{len(self.tickers)}] {ticker:12} - OK ({len(df)} points)")
                    success = True
                    break
                    
                except Exception as e:
                    if attempt < max_retries:
                        print(f"  [{i:2d}/{len(self.tickers)}] {ticker:12} - Retry {attempt}/{max_retries}")
                        time.sleep(retry_delay)
                    else:
                        print(f"  [{i:2d}/{len(self.tickers)}] {ticker:12} - FAILED")
            
            if not success:
                failed_tickers.append(ticker)
            
            # Pause every 10 tickers to avoid rate limiting
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(self.tickers)} tickers processed")
                time.sleep(1)
        
        # Validate minimum universe size
        min_universe = max(self.n_long + self.n_short + 5, 15)
        if len(valid_data) < min_universe:
            raise RuntimeError(
                f"CRITICAL: Only {len(valid_data)} tickers retrieved. "
                f"Minimum {min_universe} required for {self.n_long}L/{self.n_short}S portfolio."
            )
        
        # Construct price matrix
        self.data = pd.DataFrame(valid_data)
        
        # Data cleaning
        missing_before = self.data.isna().sum().sum()
        self.data = self.data.interpolate(method='linear', limit=5, limit_direction='both')
        self.data.dropna(inplace=True)
        
        print(f"\n  Data Cleaning:")
        print(f"    - Missing values filled: {missing_before}")
        print(f"    - Final universe: {self.data.shape[1]} stocks, {self.data.shape[0]} trading days")
        
        if failed_tickers:
            print(f"    - Failed tickers: {len(failed_tickers)}")
        
        self.data.to_csv('data/cleaned_prices.csv')
        print(f"  Saved to data/cleaned_prices.csv\n")
    
    def calculate_indicators(self):
        """Calculate rolling Z-scores for screening."""
        print("[STEP 2/7] Calculating Z-Score Indicators...")
        print("-" * 70)
        
        # Rolling statistics
        rolling_mean = self.data.rolling(
            window=self.lookback_window, 
            min_periods=self.lookback_window
        ).mean()
        
        rolling_std = self.data.rolling(
            window=self.lookback_window, 
            min_periods=self.lookback_window
        ).std()
        
        # Z-score: (Price - Mean) / StdDev
        self.z_scores = (self.data - rolling_mean) / rolling_std
        
        # Remove warm-up period
        self.z_scores = self.z_scores.dropna()
        self.data = self.data.loc[self.z_scores.index]
        
        # Summary statistics
        print(f"  Z-Score Statistics:")
        print(f"    Range:      [{self.z_scores.min().min():.2f}, {self.z_scores.max().max():.2f}]")
        print(f"    Mean:       {self.z_scores.mean().mean():.3f}")
        print(f"    Std Dev:    {self.z_scores.std().mean():.3f}")
        print(f"    Data points: {len(self.z_scores)}")
        print(f"  Indicators calculated\n")
    
    def run_statistical_tests(self):
        """
        Perform statistical validation of Z-scores.
        Tests for stationarity, normality, and correlation structure.
        """
        print("[STEP 3/7] Running Statistical Tests...")
        print("-" * 70)
        
        results = {
            'stationarity': {},
            'normality': {},
            'correlation': {}
        }
        
        # 1. Stationarity Test (Augmented Dickey-Fuller)
        print("  [1/3] Testing Stationarity (ADF Test)...")
        stationary_count = 0
        
        for ticker in self.z_scores.columns:
            z_series = self.z_scores[ticker].dropna()
            
            try:
                adf_result = adfuller(z_series, maxlag=20, regression='c')
                
                results['stationarity'][ticker] = {
                    'adf_statistic': adf_result[0],
                    'p_value': adf_result[1],
                    'critical_values': adf_result[4],
                    'is_stationary': adf_result[1] < 0.05
                }
                
                if adf_result[1] < 0.05:
                    stationary_count += 1
                    
            except Exception as e:
                results['stationarity'][ticker] = {'error': str(e)}
        
        stationarity_pct = (stationary_count / len(self.z_scores.columns)) * 100
        print(f"        Result: {stationary_count}/{len(self.z_scores.columns)} "
              f"({stationarity_pct:.1f}%) series are stationary (p < 0.05)")
        
        # 2. Normality Test (Jarque-Bera)
        print("  [2/3] Testing Normality (Jarque-Bera Test)...")
        normal_count = 0
        
        for ticker in self.z_scores.columns:
            z_series = self.z_scores[ticker].dropna()
            
            try:
                jb_stat, p_value = stats.jarque_bera(z_series)
                
                results['normality'][ticker] = {
                    'jb_statistic': jb_stat,
                    'p_value': p_value,
                    'skewness': stats.skew(z_series),
                    'kurtosis': stats.kurtosis(z_series),
                    'is_normal': p_value > 0.05
                }
                
                if p_value > 0.05:
                    normal_count += 1
                    
            except Exception as e:
                results['normality'][ticker] = {'error': str(e)}
        
        normality_pct = (normal_count / len(self.z_scores.columns)) * 100
        print(f"        Result: {normal_count}/{len(self.z_scores.columns)} "
              f"({normality_pct:.1f}%) series are approximately normal (p > 0.05)")
        
        # Calculate aggregate skewness and kurtosis
        all_skewness = [results['normality'][t]['skewness'] 
                       for t in self.z_scores.columns 
                       if 'skewness' in results['normality'][t]]
        all_kurtosis = [results['normality'][t]['kurtosis'] 
                       for t in self.z_scores.columns 
                       if 'kurtosis' in results['normality'][t]]
        
        avg_skew = np.mean(all_skewness)
        avg_kurt = np.mean(all_kurtosis)
        
        print(f"        Average Skewness: {avg_skew:.3f}")
        print(f"        Average Kurtosis: {avg_kurt:.3f}")
        
        # 3. Correlation Analysis
        print("  [3/3] Analyzing Correlation Structure...")
        corr_matrix = self.z_scores.corr()
        
        # Remove diagonal (self-correlation = 1)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        correlations = corr_matrix.where(mask).stack().values
        
        results['correlation'] = {
            'matrix': corr_matrix,
            'mean_correlation': np.mean(correlations),
            'median_correlation': np.median(correlations),
            'max_correlation': np.max(correlations),
            'min_correlation': np.min(correlations),
            'std_correlation': np.std(correlations)
        }
        
        print(f"        Mean pairwise correlation: {results['correlation']['mean_correlation']:.3f}")
        print(f"        Median correlation:        {results['correlation']['median_correlation']:.3f}")
        print(f"        Max correlation:           {results['correlation']['max_correlation']:.3f}")
        
        self.statistical_tests = results
        
        # Save detailed results
        self._save_statistical_tests()
        
        print(f"  Statistical tests complete\n")
    
    def _save_statistical_tests(self):
        """Save statistical test results to file."""
        with open('results/statistical_tests.txt', 'w') as f:
            f.write("Z-SCORE SCREENING STRATEGY - STATISTICAL VALIDATION\n")
            f.write("="*70 + "\n\n")
            
            # Stationarity Summary
            f.write("1. STATIONARITY TESTS (Augmented Dickey-Fuller)\n")
            f.write("-"*70 + "\n")
            
            stationary = [t for t, r in self.statistical_tests['stationarity'].items() 
                         if r.get('is_stationary', False)]
            f.write(f"Stationary series: {len(stationary)}/{len(self.statistical_tests['stationarity'])} "
                   f"({len(stationary)/len(self.statistical_tests['stationarity'])*100:.1f}%)\n\n")
            
            f.write("Sample Results (first 5 tickers):\n")
            for ticker in list(self.statistical_tests['stationarity'].keys())[:5]:
                result = self.statistical_tests['stationarity'][ticker]
                if 'p_value' in result:
                    f.write(f"  {ticker:12} ADF={result['adf_statistic']:7.3f}  "
                           f"p-value={result['p_value']:.4f}  "
                           f"{'STATIONARY' if result['is_stationary'] else 'NON-STATIONARY'}\n")
            
            # Normality Summary
            f.write("\n2. NORMALITY TESTS (Jarque-Bera)\n")
            f.write("-"*70 + "\n")
            
            normal = [t for t, r in self.statistical_tests['normality'].items() 
                     if r.get('is_normal', False)]
            f.write(f"Normal distributions: {len(normal)}/{len(self.statistical_tests['normality'])} "
                   f"({len(normal)/len(self.statistical_tests['normality'])*100:.1f}%)\n\n")
            
            # Correlation Summary
            f.write("\n3. CORRELATION ANALYSIS\n")
            f.write("-"*70 + "\n")
            corr = self.statistical_tests['correlation']
            f.write(f"Mean pairwise correlation:   {corr['mean_correlation']:7.3f}\n")
            f.write(f"Median correlation:          {corr['median_correlation']:7.3f}\n")
            f.write(f"Range:                       [{corr['min_correlation']:.3f}, {corr['max_correlation']:.3f}]\n")
            f.write(f"Standard deviation:          {corr['std_correlation']:7.3f}\n")
            
            # Interpretation
            f.write("\n4. INTERPRETATION\n")
            f.write("-"*70 + "\n")
            
            stat_pct = len(stationary)/len(self.statistical_tests['stationarity'])*100
            if stat_pct > 80:
                f.write("Stationarity: EXCELLENT - Z-scores exhibit strong mean reversion\n")
            elif stat_pct > 60:
                f.write("Stationarity: GOOD - Majority of Z-scores are mean-reverting\n")
            else:
                f.write("Stationarity: POOR - Many series lack mean reversion properties\n")
            
            if abs(corr['mean_correlation']) < 0.3:
                f.write("Correlation: LOW - Good diversification across signals\n")
            elif abs(corr['mean_correlation']) < 0.5:
                f.write("Correlation: MODERATE - Acceptable diversification\n")
            else:
                f.write("Correlation: HIGH - Limited diversification benefits\n")
    
    def generate_signals(self):
        """
        Generate trading signals using RANKING-BASED SCREENING.
        At each rebalancing date:
        1. Rank all stocks by Z-score
        2. Long top N stocks with lowest Z-scores (most undervalued)
        3. Short top N stocks with highest Z-scores (most overvalued)
        """
        print("[STEP 4/7] Generating Screening Signals (Rank-Based)...")
        print("-" * 70)
        
        # Initialize signals
        signals = pd.DataFrame(0, index=self.z_scores.index, columns=self.z_scores.columns)
        
        # Get rebalancing dates (monthly)
        rebalance_dates = self.z_scores.resample(self.rebalance_freq).first().index
        
        long_selections = []
        short_selections = []
        
        # Generate signals at each rebalancing date
        for date in rebalance_dates:
            if date not in self.z_scores.index:
                continue
            
            # Get Z-scores for this date
            z_row = self.z_scores.loc[date].dropna()
            
            if len(z_row) < self.n_long + self.n_short:
                print(f"  Warning: Insufficient stocks at {date.date()} - skipping")
                continue
            
            # SCREENING: Rank by Z-score
            sorted_zscores = z_row.sort_values()
            
            # Select TOP N most undervalued (lowest/most negative Z-scores)
            long_tickers = sorted_zscores.head(self.n_long).index.tolist()
            signals.loc[date, long_tickers] = 1
            long_selections.extend(long_tickers)
            
            # Select TOP N most overvalued (highest/most positive Z-scores)
            short_tickers = sorted_zscores.tail(self.n_short).index.tolist()
            signals.loc[date, short_tickers] = -1
            short_selections.extend(short_tickers)
        
        # Forward fill positions between rebalances
        monthly_signals = signals.loc[rebalance_dates]
        self.positions = monthly_signals.reindex(self.z_scores.index, method='ffill').fillna(0)
        
        # Calculate selection statistics
        from collections import Counter
        long_counter = Counter(long_selections)
        short_counter = Counter(short_selections)
        
        print(f"  Screening Configuration:")
        print(f"    Rebalancing dates: {len(rebalance_dates)}")
        print(f"    Long positions:    {self.n_long} stocks per rebalance")
        print(f"    Short positions:   {self.n_short} stocks per rebalance")
        
        print(f"\n  Selection Statistics:")
        print(f"    Total long selections:  {len(long_selections)}")
        print(f"    Total short selections: {len(short_selections)}")
        print(f"    Unique stocks traded:   {len(set(long_selections + short_selections))}")
        
        # Most frequently selected stocks
        print(f"\n  Most Frequently Selected (Long):")
        for ticker, count in long_counter.most_common(5):
            pct = (count / len(rebalance_dates)) * 100
            print(f"    {ticker:12} {count:3d} times ({pct:5.1f}%)")
        
        print(f"\n  Most Frequently Selected (Short):")
        for ticker, count in short_counter.most_common(5):
            pct = (count / len(rebalance_dates)) * 100
            print(f"    {ticker:12} {count:3d} times ({pct:5.1f}%)")
        
        # Save positions
        self.positions.to_csv('results/positions.csv')
        print(f"\n  Saved to results/positions.csv\n")
    
    def run_backtest(self):
        """Execute backtest with equal-weight allocation."""
        print("[STEP 5/7] Running Backtest Engine...")
        print("-" * 70)
        
        # Daily returns
        returns = self.data.pct_change().fillna(0)
        
        # Lag positions (trade on T+1)
        lagged_positions = self.positions.shift(1).fillna(0)
        
        # Position changes for transaction costs
        position_changes = lagged_positions.diff().abs().fillna(0)
        
        # Equal-weight allocation
        weights = pd.DataFrame(0.0, index=lagged_positions.index, columns=lagged_positions.columns)
        
        for date in weights.index:
            long_mask = lagged_positions.loc[date] == 1
            short_mask = lagged_positions.loc[date] == -1
            
            n_long_active = long_mask.sum()
            n_short_active = short_mask.sum()
            
            if n_long_active > 0:
                weights.loc[date, long_mask] = 0.5 / n_long_active
            
            if n_short_active > 0:
                weights.loc[date, short_mask] = -0.5 / n_short_active
        
        # Strategy returns
        strategy_returns_gross = (weights * returns).sum(axis=1)
        
        # Transaction costs
        turnover = position_changes.sum(axis=1)
        daily_costs = turnover * self.transaction_cost
        
        # Net returns
        self.daily_returns = strategy_returns_gross - daily_costs
        
        # Portfolio value
        self.portfolio_value = self.initial_capital * (1 + self.daily_returns).cumprod()
        
        # Generate trade log
        self._generate_trade_log(position_changes)
        
        # Summary
        total_trades = position_changes.sum().sum()
        total_costs = daily_costs.sum() * self.initial_capital
        avg_turnover = turnover.mean()
        
        print(f"  Backtest Results:")
        print(f"    Total trades:       {int(total_trades)}")
        print(f"    Transaction costs:  EUR {total_costs:,.0f}")
        print(f"    Average turnover:   {avg_turnover:.2f} positions/day")
        print(f"  Backtest complete\n")
        
        self.portfolio_value.to_csv('results/equity_curve.csv')
    
    def _generate_trade_log(self, position_changes):
        """Generate detailed trade log."""
        trades = []
        
        for date in position_changes.index:
            for ticker in position_changes.columns:
                change = position_changes.loc[date, ticker]
                
                if change != 0:
                    trades.append({
                        'Date': date,
                        'Ticker': ticker,
                        'Change': change,
                        'New_Position': self.positions.loc[date, ticker],
                        'Price': self.data.loc[date, ticker],
                        'Z_Score': self.z_scores.loc[date, ticker]
                    })
        
        self.trades = pd.DataFrame(trades)
        
        if not self.trades.empty:
            self.trades.to_csv('results/trade_log.csv', index=False)
    
    def calculate_metrics(self):
        """Calculate comprehensive performance metrics."""
        print("[STEP 6/7] Computing Performance Metrics...")
        print("-" * 70)
        
        # Time metrics
        days = (self.portfolio_value.index[-1] - self.portfolio_value.index[0]).days
        years = days / 365.25
        
        # Return metrics
        total_return = (self.portfolio_value.iloc[-1] / self.initial_capital) - 1
        cagr = (self.portfolio_value.iloc[-1] / self.initial_capital) ** (1 / years) - 1
        
        # Risk metrics
        volatility = self.daily_returns.std() * np.sqrt(252)
        downside_returns = self.daily_returns[self.daily_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252)
        
        # Ratios
        sharpe_ratio = cagr / volatility if volatility > 0 else 0
        sortino_ratio = cagr / downside_deviation if downside_deviation > 0 else 0
        
        # Drawdown
        cumulative = (1 + self.daily_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win metrics
        winning_days = (self.daily_returns > 0).sum()
        losing_days = (self.daily_returns < 0).sum()
        win_rate = winning_days / len(self.daily_returns) if len(self.daily_returns) > 0 else 0
        
        avg_win = self.daily_returns[self.daily_returns > 0].mean()
        avg_loss = self.daily_returns[self.daily_returns < 0].mean()
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
        
        self.metrics = {
            'Total Return': total_return,
            'CAGR': cagr,
            'Volatility (Ann.)': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Max Drawdown': max_drawdown,
            'Calmar Ratio': calmar_ratio,
            'Win Rate': win_rate,
            'Profit Factor': profit_factor,
            'Best Day': self.daily_returns.max(),
            'Worst Day': self.daily_returns.min(),
            'Avg Daily Return': self.daily_returns.mean()
        }
        
        # Display
        print(f"\n  {'='*66}")
        print(f"  {'PERFORMANCE SUMMARY':^66}")
        print(f"  {'='*66}")
        print(f"  Total Return:        {total_return:>10.2%}")
        print(f"  CAGR:                {cagr:>10.2%}")
        print(f"  Volatility:          {volatility:>10.2%}")
        print(f"  {'='*66}")
        print(f"  Sharpe Ratio:        {sharpe_ratio:>10.2f}")
        print(f"  Sortino Ratio:       {sortino_ratio:>10.2f}")
        print(f"  Calmar Ratio:        {calmar_ratio:>10.2f}")
        print(f"  {'='*66}")
        print(f"  Max Drawdown:        {max_drawdown:>10.2%}")
        print(f"  Win Rate:            {win_rate:>10.2%}")
        print(f"  Profit Factor:       {profit_factor:>10.2f}")
        print(f"  {'='*66}\n")
        
        # Save
        with open('results/metrics.txt', 'w') as f:
            f.write("Z-SCORE SCREENING STRATEGY - PERFORMANCE METRICS\n")
            f.write("="*70 + "\n\n")
            
            f.write("PORTFOLIO CONFIGURATION\n")
            f.write(f"Long positions:  {self.n_long}\n")
            f.write(f"Short positions: {self.n_short}\n")
            f.write(f"Total positions: {self.n_long + self.n_short}\n\n")
            
            f.write("PERFORMANCE METRICS\n")
            for key, value in self.metrics.items():
                if isinstance(value, float):
                    if abs(value) < 0.1:
                        f.write(f"{key:.<40} {value:>12.2%}\n")
                    else:
                        f.write(f"{key:.<40} {value:>12.4f}\n")
        
        print(f"  Saved to results/metrics.txt\n")
    
    def generate_visualizations(self):
        """Create comprehensive performance dashboard."""
        print("[STEP 7/7] Generating Visualizations...")
        print("-" * 70)
        
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)
        
        # 1. Equity Curve
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(self.portfolio_value, linewidth=2, color='steelblue', label='Strategy')
        ax1.axhline(self.initial_capital, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax1.set_title('Portfolio Equity Curve', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Portfolio Value (EUR)', fontsize=11)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'EUR {x/1000:.0f}K'))
        
        # 2. Drawdown
        cumulative = (1 + self.daily_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        
        ax2 = fig.add_subplot(gs[1, :2])
        ax2.fill_between(drawdown.index, drawdown * 100, 0, color='crimson', alpha=0.3)
        ax2.plot(drawdown.index, drawdown * 100, color='darkred', linewidth=1.5)
        ax2.set_title('Drawdown Profile', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Drawdown (%)', fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # 3. Monthly Returns Heatmap
        monthly_ret = self.daily_returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        monthly_ret_df = pd.DataFrame({
            'Year': monthly_ret.index.year,
            'Month': monthly_ret.index.month,
            'Return': monthly_ret.values
        })
        
        if not monthly_ret_df.empty:
            pivot = monthly_ret_df.pivot(index='Year', columns='Month', values='Return')
            
            ax3 = fig.add_subplot(gs[2, :2])
            sns.heatmap(
                pivot * 100,
                annot=True,
                fmt='.1f',
                cmap='RdYlGn',
                center=0,
                ax=ax3,
                cbar_kws={'label': 'Return (%)'},
                linewidths=0.5
            )
            ax3.set_title('Monthly Returns Heatmap (%)', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Month', fontsize=11)
            ax3.set_ylabel('Year', fontsize=11)
        
        # 4. Z-Score Distribution
        ax4 = fig.add_subplot(gs[0, 2])
        current_zscores = self.z_scores.iloc[-1].dropna()
        ax4.hist(current_zscores, bins=30, color='navy', alpha=0.6, edgecolor='black')
        ax4.axvline(0, color='orange', linestyle='-', linewidth=2, label='Mean')
        ax4.set_title('Current Z-Score Distribution', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Z-Score', fontsize=10)
        ax4.set_ylabel('Frequency', fontsize=10)
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # 5. Rolling Sharpe
        ax5 = fig.add_subplot(gs[1, 2])
        rolling_sharpe = (
            self.daily_returns.rolling(window=252).mean() / 
            self.daily_returns.rolling(window=252).std()
        ) * np.sqrt(252)
        ax5.plot(rolling_sharpe, linewidth=1.5, color='purple')
        ax5.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax5.set_title('Rolling 1Y Sharpe Ratio', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Sharpe Ratio', fontsize=10)
        ax5.grid(True, alpha=0.3)
        
        # 6. Metrics Table
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.axis('off')
        
        metrics_text = [
            ['Metric', 'Value'],
            ['Total Return', f"{self.metrics['Total Return']:.2%}"],
            ['CAGR', f"{self.metrics['CAGR']:.2%}"],
            ['Volatility', f"{self.metrics['Volatility (Ann.)']:.2%}"],
            ['Sharpe', f"{self.metrics['Sharpe Ratio']:.2f}"],
            ['Sortino', f"{self.metrics['Sortino Ratio']:.2f}"],
            ['Max DD', f"{self.metrics['Max Drawdown']:.2%}"],
            ['Win Rate', f"{self.metrics['Win Rate']:.2%}"]
        ]
        
        table = ax6.table(
            cellText=metrics_text,
            loc='center',
            cellLoc='left',
            colWidths=[0.6, 0.4]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        for i in range(2):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # 7. Correlation Heatmap
        ax7 = fig.add_subplot(gs[3, :2])
        
        # Sample 20 stocks for readability
        sample_tickers = self.z_scores.columns[:20]
        corr_sample = self.statistical_tests['correlation']['matrix'].loc[sample_tickers, sample_tickers]
        
        sns.heatmap(
            corr_sample,
            cmap='coolwarm',
            center=0,
            ax=ax7,
            cbar_kws={'label': 'Correlation'},
            square=True,
            linewidths=0.5,
            vmin=-1,
            vmax=1
        )
        ax7.set_title('Z-Score Correlation Matrix (Sample)', fontsize=12, fontweight='bold')
        ax7.tick_params(labelsize=8)
        
        # 8. Statistical Tests Summary
        ax8 = fig.add_subplot(gs[3, 2])
        ax8.axis('off')
        
        stat_results = self.statistical_tests
        stationary_count = sum(1 for r in stat_results['stationarity'].values() 
                              if r.get('is_stationary', False))
        normal_count = sum(1 for r in stat_results['normality'].values() 
                          if r.get('is_normal', False))
        
        stat_text = [
            ['Statistical Tests', ''],
            ['', ''],
            ['Stationarity (ADF)', f"{stationary_count}/{len(stat_results['stationarity'])}"],
            ['Mean Reverting', f"{stationary_count/len(stat_results['stationarity'])*100:.0f}%"],
            ['', ''],
            ['Normality (JB)', f"{normal_count}/{len(stat_results['normality'])}"],
            ['Normal Dist.', f"{normal_count/len(stat_results['normality'])*100:.0f}%"],
            ['', ''],
            ['Avg Correlation', f"{stat_results['correlation']['mean_correlation']:.3f}"]
        ]
        
        stat_table = ax8.table(
            cellText=stat_text,
            loc='center',
            cellLoc='left',
            colWidths=[0.7, 0.3]
        )
        stat_table.auto_set_font_size(False)
        stat_table.set_fontsize(9)
        stat_table.scale(1, 1.8)
        
        stat_table[(0, 0)].set_facecolor('#4472C4')
        stat_table[(0, 0)].set_text_props(weight='bold', color='white')
        stat_table[(0, 1)].set_facecolor('#4472C4')
        
        plt.savefig('results/dashboard.png', dpi=150, bbox_inches='tight')
        print(f"  Dashboard saved to results/dashboard.png\n")
        
        plt.close()
    
    def run(self):
        """Execute complete strategy workflow."""
        start_time = datetime.now()
        
        try:
            self.download_data()
            self.calculate_indicators()
            self.run_statistical_tests()
            self.generate_signals()
            self.run_backtest()
            self.calculate_metrics()
            self.generate_visualizations()
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            print(f"{'='*70}")
            print(f"EXECUTION COMPLETE")
            print(f"{'='*70}")
            print(f"Total Runtime: {elapsed:.1f} seconds")
            print(f"Results saved to: results/")
            print(f"{'='*70}\n")
            
        except Exception as e:
            print(f"\n{'='*70}")
            print(f"CRITICAL ERROR")
            print(f"{'='*70}")
            print(f"Execution failed: {str(e)}")
            print(f"{'='*70}\n")
            raise

def main():
    """Main execution function."""
    
    # European Large-Cap Universe (avec les bons tickers)
    EUROPEAN_TICKERS = [
        # CAC 40 - France
        'AI.PA', 'AIR.PA', 'ALO.PA', 'MT.AS', 'CS.PA', 'BNP.PA',
        'EN.PA', 'CAP.PA', 'CA.PA', 'ACA.PA', 'BN.PA', 'ENGI.PA',
        'EL.PA', 'RMS.PA', 'KER.PA', 'OR.PA', 'LR.PA', 'MC.PA',
        'ML.PA', 'ORA.PA', 'RI.PA', 'PUB.PA', 'RNO.PA', 'SAF.PA',
        'SGO.PA', 'SAN.PA', 'SU.PA', 'GLE.PA', 'STLAP.PA', 'STMPA.PA',
        'TEP.PA', 'HO.PA', 'TTE.PA', 'URW.PA', 'VIE.PA', 'DG.PA', 'VIV.PA',
        
        # DAX 30 - Germany
        'ADS.DE', 'ALV.DE', 'BAS.DE', 'BAYN.DE', 'BMW.DE', 'CON.DE',
        'MBG.DE', 'DB1.DE', 'DBK.DE', 'DHL.DE', 'DTE.DE',
        'EOAN.DE', 'FRE.DE', 'HEI.DE', 'HEN3.DE', 'IFX.DE', 'LIN.DE',
        'MRK.DE', 'MTX.DE', 'MUV2.DE', 'RWE.DE', 'SAP.DE', 'SIE.DE',
        'VOW3.DE', 'VNA.DE'
    ]
    
    # Strategy Configuration
    CONFIG = {
        'tickers': EUROPEAN_TICKERS,
        'start_date': '2019-01-01',
        'end_date': '2024-01-01',
        'initial_capital': 100000,
        'n_long': 10,
        'n_short': 10
    }
    
    # Execute Strategy
    strategy = ZScoreScreeningStrategy(**CONFIG)
    strategy.run()

if __name__ == "__main__":
    main()
