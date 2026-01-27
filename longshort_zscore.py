"""
Z-Score Mean Reversion Strategy
Long/Short equity strategy based on statistical mean reversion
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class ZScoreMeanReversionStrategy:
    """
    Long/Short mean reversion strategy using Z-score thresholds.
    
    Parameters:
    -----------
    tickers : list
        List of stock tickers to trade
    start_date : str
        Start date for backtesting (YYYY-MM-DD)
    end_date : str
        End date for backtesting (YYYY-MM-DD)
    lookback : int
        Rolling window for Z-score calculation (default: 20 days)
    n_long : int
        Number of long positions (default: 5)
    n_short : int
        Number of short positions (default: 5)
    entry_threshold : float
        Z-score threshold for entry (default: 2.0)
    exit_threshold : float
        Z-score threshold for exit (default: 0.2)
    initial_capital : float
        Starting portfolio value (default: 1,000,000)
    transaction_cost : float
        Transaction cost per trade (default: 0.0001)
    """
    
    def __init__(self, tickers, start_date='2019-01-01', end_date='2021-12-31',
                 lookback=20, n_long=5, n_short=5, entry_threshold=2.0,
                 exit_threshold=0.2, initial_capital=1000000, transaction_cost=0.0001):
        
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.lookback_window = lookback
        self.n_long = n_long
        self.n_short = n_short
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        
        self.price_data = None
        self.z_scores = None
        self.positions = None
        self.portfolio_value = None
        self.daily_returns = None
        self.metrics = {}
    
    def load_data(self, filepath):
        """Load price data from Excel file with multiple sheets."""
        excel_data = pd.read_excel(filepath, sheet_name=None)
        
        all_data = {}
        
        for ticker in self.tickers:
            for sheet_name in [ticker, ticker.replace('.', '_'), ticker.replace('.', '')]:
                if sheet_name not in excel_data:
                    continue
                
                df = excel_data[sheet_name]
                
                date_col = next((col for col in df.columns if 'date' in str(col).lower()), None)
                price_col = next((col for col in df.columns if 'close' in str(col).lower()), None)
                
                if not date_col or not price_col:
                    continue
                
                df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
                
                if df[price_col].dtype == 'object':
                    df[price_col] = (df[price_col].astype(str)
                                     .str.replace('$', '').str.replace(',', '.')
                                     .str.strip())
                
                df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
                
                clean_df = df[[date_col, price_col]].dropna()
                clean_df = clean_df.drop_duplicates(subset=[date_col])
                clean_df = clean_df.set_index(date_col)
                
                if len(clean_df) > 100:
                    all_data[ticker] = clean_df[price_col]
                    break
        
        self.price_data = pd.DataFrame(all_data)
        self.price_data = self.price_data[~self.price_data.index.duplicated()]
        self.price_data = self.price_data[(self.price_data.index >= self.start_date) & 
                                          (self.price_data.index <= self.end_date)]
        self.price_data = self.price_data.sort_index()
        
        return self
    
    def calculate_zscore(self):
        """Calculate rolling Z-scores for all assets."""
        rolling_mean = self.price_data.rolling(window=self.lookback_window,
                                                min_periods=self.lookback_window).mean()
        rolling_std = self.price_data.rolling(window=self.lookback_window,
                                               min_periods=self.lookback_window).std()
        
        self.z_scores = (self.price_data - rolling_mean) / rolling_std
        self.z_scores = self.z_scores.dropna()
        self.price_data = self.price_data.loc[self.z_scores.index]
        
        return self
    
    def generate_signals(self):
        """Generate mean reversion signals based on Z-score thresholds."""
        signals = pd.DataFrame(0, index=self.z_scores.index, columns=self.z_scores.columns)
        
        for i, date in enumerate(self.z_scores.index):
            z_row = self.z_scores.loc[date].dropna()
            
            if i > 0:
                prev_signals = signals.iloc[i-1]
            else:
                prev_signals = pd.Series(0, index=z_row.index)
            
            signals.loc[date] = prev_signals
            
            # Exit positions on mean reversion
            for ticker in z_row.index:
                z_val = z_row[ticker]
                current_pos = signals.loc[date, ticker]
                
                if current_pos == 1 and z_val > -self.exit_threshold:
                    signals.loc[date, ticker] = 0
                elif current_pos == -1 and z_val < self.exit_threshold:
                    signals.loc[date, ticker] = 0
            
            # Enter positions on extremes (monthly rebalance)
            if date.day <= 7:
                undervalued = z_row[z_row < -self.entry_threshold]
                overvalued = z_row[z_row > self.entry_threshold]
                
                current_longs = (signals.loc[date] == 1).sum()
                current_shorts = (signals.loc[date] == -1).sum()
                
                if len(undervalued) > 0 and current_longs < self.n_long:
                    n_add = min(self.n_long - current_longs, len(undervalued))
                    long_tickers = undervalued.nsmallest(n_add).index
                    for ticker in long_tickers:
                        if signals.loc[date, ticker] == 0:
                            signals.loc[date, ticker] = 1
                
                if len(overvalued) > 0 and current_shorts < self.n_short:
                    n_add = min(self.n_short - current_shorts, len(overvalued))
                    short_tickers = overvalued.nlargest(n_add).index
                    for ticker in short_tickers:
                        if signals.loc[date, ticker] == 0:
                            signals.loc[date, ticker] = -1
        
        self.positions = signals
        return self
    
    def backtest(self):
        """Execute backtest and calculate portfolio returns."""
        returns = self.price_data.pct_change().fillna(0)
        lagged_positions = self.positions.shift(1).fillna(0)
        position_changes = lagged_positions.diff().abs().fillna(0)
        
        weights = pd.DataFrame(0.0, index=lagged_positions.index, 
                             columns=lagged_positions.columns)
        
        for date in weights.index:
            long_mask = lagged_positions.loc[date] == 1
            short_mask = lagged_positions.loc[date] == -1
            
            n_long = long_mask.sum()
            n_short = short_mask.sum()
            
            if n_long > 0:
                weights.loc[date, long_mask] = 0.5 / n_long
            if n_short > 0:
                weights.loc[date, short_mask] = -0.5 / n_short
        
        strategy_returns_gross = (weights * returns).sum(axis=1)
        turnover = position_changes.sum(axis=1)
        transaction_costs = turnover * self.transaction_cost
        
        self.daily_returns = strategy_returns_gross - transaction_costs
        self.portfolio_value = self.initial_capital * (1 + self.daily_returns).cumprod()
        
        return self
    
    def calculate_metrics(self):
        """Calculate performance metrics."""
        days = (self.portfolio_value.index[-1] - self.portfolio_value.index[0]).days
        years = days / 365.25
        
        total_return = (self.portfolio_value.iloc[-1] / self.initial_capital) - 1
        cagr = (self.portfolio_value.iloc[-1] / self.initial_capital) ** (1/years) - 1
        volatility = self.daily_returns.std() * np.sqrt(252)
        sharpe = cagr / volatility if volatility > 0 else 0
        
        downside_returns = self.daily_returns[self.daily_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252)
        sortino = cagr / downside_deviation if downside_deviation > 0 else 0
        
        cumulative = (1 + self.daily_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        win_rate = (self.daily_returns > 0).sum() / len(self.daily_returns)
        
        self.metrics = {
            'Total Return': total_return,
            'CAGR': cagr,
            'Volatility': volatility,
            'Sharpe Ratio': sharpe,
            'Sortino Ratio': sortino,
            'Max Drawdown': max_dd,
            'Calmar Ratio': calmar,
            'Win Rate': win_rate
        }
        
        return self.metrics
    
    def plot_results(self, save_path='results.png'):
        """Generate performance visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Equity curve
        axes[0,0].plot(self.portfolio_value, linewidth=2, color='steelblue')
        axes[0,0].axhline(self.initial_capital, linestyle='--', alpha=0.5, color='gray')
        axes[0,0].set_title('Portfolio Value', fontweight='bold')
        axes[0,0].grid(alpha=0.3)
        
        # Drawdown
        cumulative = (1 + self.daily_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        
        axes[0,1].fill_between(drawdown.index, drawdown*100, 0, color='red', alpha=0.3)
        axes[0,1].set_title('Drawdown (%)', fontweight='bold')
        axes[0,1].grid(alpha=0.3)
        
        # Z-score distribution
        current_z = self.z_scores.iloc[-1].dropna()
        axes[1,0].hist(current_z, bins=30, color='navy', alpha=0.6, edgecolor='black')
        axes[1,0].axvline(0, color='orange', linestyle='--', linewidth=2)
        axes[1,0].axvline(-self.entry_threshold, color='green', linestyle='--', alpha=0.7)
        axes[1,0].axvline(self.entry_threshold, color='red', linestyle='--', alpha=0.7)
        axes[1,0].set_title('Z-Score Distribution', fontweight='bold')
        axes[1,0].grid(alpha=0.3)
        
        # Returns distribution
        axes[1,1].hist(self.daily_returns*100, bins=50, color='green', alpha=0.6, edgecolor='black')
        axes[1,1].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[1,1].set_title('Daily Returns (%)', fontweight='bold')
        axes[1,1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def run(self, filepath):
        """Execute complete strategy pipeline."""
        self.load_data(filepath)
        self.calculate_zscore()
        self.generate_signals()
        self.backtest()
        self.calculate_metrics()
        self.plot_results()
        
        return self.metrics


if __name__ == '__main__':
    # Example usage
    TICKERS = [
        'NVDA', 'AAPL', 'MSFT', 'AMZN', 'GOOG', 'META', 'AVGO', 'TSLA', 'ORCL',
        'AMD', 'PLTR', 'NFLX', 'CSCO', 'IBM', 'CRM', 'INTC', 'MU', 'LRCX', 'AMAT',
        'BRK.B', 'JPM', 'V', 'MA', 'BAC', 'MS', 'GS', 'WFC', 'AXP',
        'LLY', 'JNJ', 'ABBV', 'UNH', 'MRK', 'TMO',
        'WMT', 'COST', 'HD', 'PG', 'KO', 'MCD', 'PM',
        'XOM', 'CVX', 'GE', 'CAT', 'RTX', 'LIN', 'TMUS'
    ]
    
    strategy = ZScoreMeanReversionStrategy(
        tickers=TICKERS,
        start_date='2019-01-01',
        end_date='2021-12-31',
        lookback=20,
        n_long=5,
        n_short=5,
        entry_threshold=2.0,
        exit_threshold=0.2
    )
    
    metrics = strategy.run('Stock_Data_History.xlsx')
    
    print("\nPerformance Metrics:")
    print("=" * 50)
    for key, value in metrics.items():
        if isinstance(value, float):
            if 'Return' in key or 'Drawdown' in key or 'Rate' in key:
                print(f"{key:.<30} {value:>10.2%}")
            else:
                print(f"{key:.<30} {value:>10.2f}")
    print("=" * 50)
