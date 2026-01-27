"""
Long/Short Z-Score Screening Strategy
S&P 500 Top 50 Universe (2019-2024)

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
from collections import Counter

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
            initial_capital: Starting portfolio value in USD
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
        print(f"Universe:         {len(tickers)} S&P 500 stocks")
        print(f"Period:           {start_date} to {end_date}")
        print(f"Initial Capital:  USD ${initial_capital:,.0f}")
        print(f"Portfolio Size:   {n_long} Long + {n_short} Short = {n_long + n_short} positions")
        print(f"Screening Method: Rank-based selection (top/bottom N by Z-score)")
        print(f"{'='*70}\n")
    
    def download_data(self):
        """
        Download historical price data using the EXACT method from working momentum strategy.
        """
        print("[STEP 1/7] Downloading Historical Data...")
        print("-" * 70)
        print("  Downloading stocks individually (this may take 10-15 minutes)...")
        
        # CRITICAL: Keep dates as strings (like momentum code)
        start_date = self.start_date  # Keep as string
        end_date = self.end_date      # Keep as string
        
        all_data = {}
        success_count = 0
        
        for i, ticker in enumerate(self.tickers, 1):
            try:
                # Use EXACT same method as momentum strategy
                stock = yf.Ticker(ticker)
                hist = stock.history(start=start_date, end=end_date, auto_adjust=True)
                
                if not hist.empty and len(hist) > 100:  # At least 100 days of data
                    all_data[ticker] = hist['Close']
                    success_count += 1
                    if success_count % 10 == 0:
                        print(f"  Progress: {success_count}/{len(self.tickers)} stocks downloaded")
            except Exception as e:
                continue
        
        if len(all_data) == 0:
            raise RuntimeError("No price data available")
        
        self.data = pd.DataFrame(all_data)
        
        # Clean data: remove stocks with insufficient history
        initial_stocks = self.data.shape[1]
        self.data = self.data.dropna(thresh=len(self.data)*0.7, axis=1)
        removed = initial_stocks - self.data.shape[1]
        
        if removed > 0:
            print(f"  Removed {removed} stocks due to insufficient data")
        
        if self.data.empty or self.data.shape[1] == 0:
            raise RuntimeError("No price data available")
        
        print(f"  Final dataset: {self.data.shape[1]} stocks, {len(self.data)} days")
        
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
            
            z_row = self.z_scores.loc[date].dropna()
            
            if len(z_row) < self.n_long + self.n_short:
                continue
            
            # SCREENING: Rank by Z-score
            sorted_zscores = z_row.sort_values()
            
            long_tickers = sorted_zscores.head(self.n_long).index.tolist()
            signals.loc[date, long_tickers] = 1
            long_selections.extend(long_tickers)
            
            short_tickers = sorted_zscores.tail(self.n_short).index.tolist()
            signals.loc[date, short_tickers] = -1
            short_selections.extend(short_tickers)
        
        # Forward fill positions between rebalances
        monthly_signals = signals.loc[rebalance_dates]
        self.positions = monthly_signals.reindex(self.z_scores.index, method='ffill').fillna(0)
        
        print(f"  Generated signals for {len(rebalance_dates)} rebalancing dates")
        self.positions.to_csv('results/positions.csv')
        print(f"  Saved to results/positions.csv\n")
    
    def run_backtest(self):
        """Execute backtest."""
        print("[STEP 5/7] Running Backtest...")
        print("-" * 70)
        
        returns = self.data.pct_change().fillna(0)
        lagged_positions = self.positions.shift(1).fillna(0)
        position_changes = lagged_positions.diff().abs().fillna(0)
        
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
        
        strategy_returns_gross = (weights * returns).sum(axis=1)
        turnover = position_changes.sum(axis=1)
        daily_costs = turnover * self.transaction_cost
        self.daily_returns = strategy_returns_gross - daily_costs
        self.portfolio_value = self.initial_capital * (1 + self.daily_returns).cumprod()
        
        print(f"  Backtest complete\n")
        self.portfolio_value.to_csv('results/equity_curve.csv')
    
    def calculate_metrics(self):
        """Calculate performance metrics."""
        print("[STEP 6/7] Computing Metrics...")
        print("-" * 70)
        
        days = (self.portfolio_value.index[-1] - self.portfolio_value.index[0]).days
        years = days / 365.25
        
        total_return = (self.portfolio_value.iloc[-1] / self.initial_capital) - 1
        cagr = (self.portfolio_value.iloc[-1] / self.initial_capital) ** (1 / years) - 1
        volatility = self.daily_returns.std() * np.sqrt(252)
        sharpe_ratio = cagr / volatility if volatility > 0 else 0
        
        cumulative = (1 + self.daily_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        self.metrics = {
            'Total Return': total_return,
            'CAGR': cagr,
            'Volatility': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown
        }
        
        print(f"  CAGR: {cagr:.2%}, Sharpe: {sharpe_ratio:.2f}\n")
        
        with open('results/metrics.txt', 'w') as f:
            f.write("PERFORMANCE METRICS\n")
            f.write("="*50 + "\n")
            for key, value in self.metrics.items():
                if abs(value) < 0.1:
                    f.write(f"{key:.<30} {value:>10.2%}\n")
                else:
                    f.write(f"{key:.<30} {value:>10.4f}\n")
    
    def generate_visualizations(self):
        """Create dashboard."""
        print("[STEP 7/7] Generating Visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Equity Curve
        axes[0,0].plot(self.portfolio_value, color='steelblue', linewidth=2)
        axes[0,0].set_title('Equity Curve', fontweight='bold')
        axes[0,0].set_ylabel('Portfolio Value (USD)')
        axes[0,0].grid(True, alpha=0.3)
        
        # Drawdown
        cumulative = (1 + self.daily_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        axes[0,1].fill_between(drawdown.index, drawdown*100, 0, color='red', alpha=0.3)
        axes[0,1].set_title('Drawdown', fontweight='bold')
        axes[0,1].set_ylabel('Drawdown (%)')
        axes[0,1].grid(True, alpha=0.3)
        
        # Z-Score Distribution
        current_zscores = self.z_scores.iloc[-1].dropna()
        axes[1,0].hist(current_zscores, bins=30, color='navy', alpha=0.6)
        axes[1,0].axvline(0, color='red', linestyle='--')
        axes[1,0].set_title('Z-Score Distribution', fontweight='bold')
        axes[1,0].set_xlabel('Z-Score')
        axes[1,0].grid(True, alpha=0.3)
        
        # Returns Distribution
        axes[1,1].hist(self.daily_returns*100, bins=50, color='green', alpha=0.6)
        axes[1,1].axvline(0, color='red', linestyle='--')
        axes[1,1].set_title('Daily Returns Distribution', fontweight='bold')
        axes[1,1].set_xlabel('Return (%)')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/dashboard.png', dpi=150, bbox_inches='tight')
        print("  Dashboard saved\n")
        plt.close()
    
    def run(self):
        """Execute strategy."""
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
            print(f"COMPLETE - Runtime: {elapsed:.1f}s")
            print(f"{'='*70}\n")
            
        except Exception as e:
            print(f"\nERROR: {str(e)}\n")
            raise


def main():
    """Main execution."""
    
    SP500_TOP50 = [
        'NVDA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG', 'META', 'AVGO', 'TSLA', 'ORCL',
        'AMD', 'PLTR', 'NFLX', 'CSCO', 'IBM', 'CRM', 'INTC', 'MU', 'LRCX', 'AMAT', 'KLAC',
        'BRK.B', 'JPM', 'V', 'MA', 'BAC', 'MS', 'GS', 'WFC', 'AXP',
        'LLY', 'JNJ', 'ABBV', 'UNH', 'MRK', 'TMO',
        'WMT', 'COST', 'HD', 'PG', 'KO', 'MCD', 'PM',
        'XOM', 'CVX', 'GE', 'CAT', 'RTX', 'LIN', 'TMUS'
    ]
    
    CONFIG = {
        'tickers': SP500_TOP50,
        'start_date': '2019-01-01',
        'end_date': '2024-01-01',
        'initial_capital': 100000,
        'n_long': 10,
        'n_short': 10
    }
    
    strategy = ZScoreScreeningStrategy(**CONFIG)
    strategy.run()

if __name__ == "__main__":
    main()
