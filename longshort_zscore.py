"""
Z-Score Mean Reversion Strategy with VIX Filter Sensitivity Analysis
Long/Short equity strategy testing multiple VIX thresholds
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
    Long/Short mean reversion strategy using Z-score thresholds with optional VIX filter.
    
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
    vix_threshold : float
        VIX threshold for regime filter (default: None for no filter)
    """
    
    def __init__(self, tickers, start_date='2019-01-01', end_date='2021-12-31',
                 lookback=20, n_long=5, n_short=5, entry_threshold=2.0,
                 exit_threshold=0.2, initial_capital=1000000, transaction_cost=0.0001,
                 vix_threshold=None):
        
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
        self.vix_threshold = vix_threshold
        
        self.price_data = None
        self.vix_data = None
        self.z_scores = None
        self.positions = None
        self.portfolio_value = None
        self.daily_returns = None
        self.metrics = {}
        self.vix_enabled = (vix_threshold is not None)
    
    def load_data(self, filepath):
        """Load price data from Excel file with multiple sheets."""
        try:
            excel_data = pd.read_excel(filepath, sheet_name=None)
        except Exception as e:
            raise RuntimeError(f"Error reading Excel: {e}")
        
        all_data = {}
        
        for ticker in self.tickers:
            possible_names = [ticker, ticker.replace('.', '_'), ticker.replace('.', '')]
            
            found = False
            for sheet_name in possible_names:
                if sheet_name not in excel_data:
                    continue
                
                try:
                    df = excel_data[sheet_name]
                    
                    if df.shape[1] < 5:
                        continue
                    
                    date_col = df.columns[3]
                    price_col = df.columns[4]
                    
                    dates_raw = df[date_col]
                    
                    if pd.api.types.is_numeric_dtype(dates_raw):
                        dates_converted = pd.to_datetime(dates_raw, unit='D', origin='1899-12-30', errors='coerce')
                    else:
                        dates_converted = pd.to_datetime(dates_raw, dayfirst=True, errors='coerce')
                        
                        if dates_converted.isna().all():
                            dates_converted = pd.to_datetime(dates_raw, format='%d/%m/%Y', errors='coerce')
                        
                        if dates_converted.isna().all():
                            dates_converted = pd.to_datetime(dates_raw, format='%m/%d/%Y', errors='coerce')
                        
                        if dates_converted.isna().all():
                            dates_converted = pd.to_datetime(dates_raw, errors='coerce')
                    
                    prices_raw = df[price_col]
                    
                    if prices_raw.dtype == 'object':
                        prices_clean = (prices_raw.astype(str)
                                       .str.replace('$', '', regex=False)
                                       .str.replace('€', '', regex=False)
                                       .str.replace(' ', '', regex=False)
                                       .str.replace(',', '.', regex=False)
                                       .str.strip())
                        prices_converted = pd.to_numeric(prices_clean, errors='coerce')
                    else:
                        prices_converted = pd.to_numeric(prices_raw, errors='coerce')
                    
                    clean_df = pd.DataFrame({
                        'Date': dates_converted,
                        'Close': prices_converted
                    })
                    
                    clean_df = clean_df.dropna()
                    clean_df = clean_df.drop_duplicates(subset=['Date'], keep='first')
                    clean_df = clean_df.set_index('Date')
                    
                    if len(clean_df) > 100:
                        all_data[ticker] = clean_df['Close']
                        found = True
                        break
                        
                except Exception:
                    continue
        
        if len(all_data) == 0:
            raise RuntimeError("No data loaded from Excel")
        
        self.price_data = pd.DataFrame(all_data)
        self.price_data = self.price_data[self.price_data.index.notna()]
        self.price_data = self.price_data[~self.price_data.index.duplicated(keep='first')]
        
        self.price_data = self.price_data[
            (self.price_data.index >= self.start_date) & 
            (self.price_data.index <= self.end_date)
        ]
        
        self.price_data = self.price_data.sort_index()
        
        if len(self.price_data) == 0:
            raise RuntimeError(f"No data in date range {self.start_date} to {self.end_date}")
        
        return self
    
    def load_vix_data(self, filepath):
        """Load VIX data from Excel file."""
        if not self.vix_enabled:
            return self
            
        try:
            df = pd.read_excel(filepath)
            
            if 'observation_date' in df.columns and 'VIXCLS' in df.columns:
                df['observation_date'] = pd.to_datetime(df['observation_date'])
                df = df.set_index('observation_date')
                self.vix_data = df['VIXCLS']
            else:
                raise ValueError("Expected columns 'observation_date' and 'VIXCLS'")
            
            self.vix_data = self.vix_data[
                (self.vix_data.index >= self.start_date) & 
                (self.vix_data.index <= self.end_date)
            ]
            
            vix_low = (self.vix_data < self.vix_threshold).sum()
            vix_high = (self.vix_data >= self.vix_threshold).sum()
            total = len(self.vix_data)
            
            print(f"  VIX Filter: VIX < {self.vix_threshold}")
            print(f"  Low vol days: {vix_low} ({vix_low/total*100:.1f}%)")
            print(f"  High vol days: {vix_high} ({vix_high/total*100:.1f}%)")
            
            return self
            
        except Exception as e:
            print(f"  Warning: Could not load VIX data: {e}")
            print("  Running without VIX filter")
            self.vix_data = None
            self.vix_enabled = False
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
        """Generate mean reversion signals with optional VIX filter."""
        signals = pd.DataFrame(0, index=self.z_scores.index, columns=self.z_scores.columns)
        
        entries_blocked = 0
        
        for i, date in enumerate(self.z_scores.index):
            z_row = self.z_scores.loc[date].dropna()
            
            vix_today = None
            if self.vix_enabled and self.vix_data is not None:
                if date in self.vix_data.index:
                    vix_today = self.vix_data.loc[date]
                else:
                    prior_dates = self.vix_data.index[self.vix_data.index < date]
                    if len(prior_dates) > 0:
                        vix_today = self.vix_data.loc[prior_dates[-1]]
            
            if i > 0:
                prev_signals = signals.iloc[i-1]
            else:
                prev_signals = pd.Series(0, index=z_row.index)
            
            signals.loc[date] = prev_signals
            
            for ticker in z_row.index:
                z_val = z_row[ticker]
                current_pos = signals.loc[date, ticker]
                
                if current_pos == 1 and z_val > -self.exit_threshold:
                    signals.loc[date, ticker] = 0
                elif current_pos == -1 and z_val < self.exit_threshold:
                    signals.loc[date, ticker] = 0
            
            if date.day <= 7:
                vix_allows_entry = True
                if self.vix_enabled and vix_today is not None:
                    vix_allows_entry = (vix_today < self.vix_threshold)
                    if not vix_allows_entry:
                        entries_blocked += 1
                
                if vix_allows_entry:
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
        
        if self.vix_enabled and entries_blocked > 0:
            print(f"  Entries blocked: {entries_blocked} rebalance dates")
        
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
        if self.portfolio_value is None or len(self.portfolio_value) == 0:
            raise RuntimeError("Portfolio value is empty")
        
        if len(self.portfolio_value) < 2:
            raise RuntimeError(f"Not enough data points: {len(self.portfolio_value)} days")
        
        days = (self.portfolio_value.index[-1] - self.portfolio_value.index[0]).days
        years = days / 365.25
        
        if years == 0:
            raise RuntimeError("Time period is too short")
        
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
        
        axes[0,0].plot(self.portfolio_value, linewidth=2, color='steelblue')
        axes[0,0].axhline(self.initial_capital, linestyle='--', alpha=0.5, color='gray')
        title = 'Portfolio Value'
        if self.vix_enabled:
            title += f' (VIX < {self.vix_threshold})'
        axes[0,0].set_title(title, fontweight='bold')
        axes[0,0].grid(alpha=0.3)
        
        cumulative = (1 + self.daily_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        
        axes[0,1].fill_between(drawdown.index, drawdown*100, 0, color='red', alpha=0.3)
        axes[0,1].set_title('Drawdown (%)', fontweight='bold')
        axes[0,1].grid(alpha=0.3)
        
        current_z = self.z_scores.iloc[-1].dropna()
        axes[1,0].hist(current_z, bins=30, color='navy', alpha=0.6, edgecolor='black')
        axes[1,0].axvline(0, color='orange', linestyle='--', linewidth=2)
        axes[1,0].axvline(-self.entry_threshold, color='green', linestyle='--', alpha=0.7)
        axes[1,0].axvline(self.entry_threshold, color='red', linestyle='--', alpha=0.7)
        axes[1,0].set_title('Z-Score Distribution', fontweight='bold')
        axes[1,0].grid(alpha=0.3)
        
        axes[1,1].hist(self.daily_returns*100, bins=50, color='green', alpha=0.6, edgecolor='black')
        axes[1,1].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[1,1].set_title('Daily Returns (%)', fontweight='bold')
        axes[1,1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def run(self, filepath, vix_filepath=None):
        """Execute complete strategy pipeline."""
        self.load_data(filepath)
        
        if vix_filepath and self.vix_enabled:
            self.load_vix_data(vix_filepath)
        
        self.calculate_zscore()
        self.generate_signals()
        self.backtest()
        
        if len(self.portfolio_value) == 0:
            raise RuntimeError("Backtest produced no results")
        
        self.calculate_metrics()
        
        vix_suffix = f'_vix{int(self.vix_threshold)}' if self.vix_enabled else '_no_vix'
        plot_name = f'results_{self.start_date}_{self.end_date}{vix_suffix}.png'
        self.plot_results(save_path=plot_name)
        
        return self.metrics


def run_vix_sensitivity_analysis(tickers, start_date, end_date, stock_filepath, vix_filepath):
    """Run complete VIX sensitivity analysis."""
    
    print("="*80)
    print(f"VIX SENSITIVITY ANALYSIS: {start_date} to {end_date}")
    print("="*80)
    
    results = {}
    
    print("\n[1/4] WITHOUT VIX FILTER")
    print("-" * 80)
    strategy_no_vix = ZScoreMeanReversionStrategy(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        lookback=20,
        n_long=5,
        n_short=5,
        entry_threshold=2.0,
        exit_threshold=0.2,
        vix_threshold=None
    )
    results['No Filter'] = strategy_no_vix.run(stock_filepath)
    print(f"  Return: {results['No Filter']['Total Return']:.2%}")
    print(f"  Sharpe: {results['No Filter']['Sharpe Ratio']:.2f}")
    
    print("\n[2/4] WITH VIX < 20")
    print("-" * 80)
    strategy_vix20 = ZScoreMeanReversionStrategy(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        lookback=20,
        n_long=5,
        n_short=5,
        entry_threshold=2.0,
        exit_threshold=0.2,
        vix_threshold=20.0
    )
    results['VIX < 20'] = strategy_vix20.run(stock_filepath, vix_filepath)
    print(f"  Return: {results['VIX < 20']['Total Return']:.2%}")
    print(f"  Sharpe: {results['VIX < 20']['Sharpe Ratio']:.2f}")
    
    print("\n[3/4] WITH VIX < 25")
    print("-" * 80)
    strategy_vix25 = ZScoreMeanReversionStrategy(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        lookback=20,
        n_long=5,
        n_short=5,
        entry_threshold=2.0,
        exit_threshold=0.2,
        vix_threshold=25.0
    )
    results['VIX < 25'] = strategy_vix25.run(stock_filepath, vix_filepath)
    print(f"  Return: {results['VIX < 25']['Total Return']:.2%}")
    print(f"  Sharpe: {results['VIX < 25']['Sharpe Ratio']:.2f}")
    
    print("\n[4/4] WITH VIX < 30")
    print("-" * 80)
    strategy_vix30 = ZScoreMeanReversionStrategy(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        lookback=20,
        n_long=5,
        n_short=5,
        entry_threshold=2.0,
        exit_threshold=0.2,
        vix_threshold=30.0
    )
    results['VIX < 30'] = strategy_vix30.run(stock_filepath, vix_filepath)
    print(f"  Return: {results['VIX < 30']['Total Return']:.2%}")
    print(f"  Sharpe: {results['VIX < 30']['Sharpe Ratio']:.2f}")
    
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    print(f"{'Strategy':<15} {'Return':>12} {'CAGR':>10} {'Sharpe':>10} {'Sortino':>10} {'Max DD':>10} {'Vol':>10}")
    print("-"*80)
    
    for name, metrics in results.items():
        print(f"{name:<15} "
              f"{metrics['Total Return']:>11.2%} "
              f"{metrics['CAGR']:>9.2%} "
              f"{metrics['Sharpe Ratio']:>10.2f} "
              f"{metrics['Sortino Ratio']:>10.2f} "
              f"{metrics['Max Drawdown']:>10.2%} "
              f"{metrics['Volatility']:>10.2%}")
    
    print("="*80)
    
    print("\n" + "="*80)
    print("IMPROVEMENTS vs NO FILTER")
    print("="*80)
    print(f"{'Strategy':<15} {'Return Improvement':>20} {'Sharpe Improvement':>20} {'Max DD Improvement':>20}")
    print("-"*80)
    
    baseline = results['No Filter']
    for name, metrics in results.items():
        if name == 'No Filter':
            continue
        
        ret_improvement = ((metrics['Total Return'] - baseline['Total Return']) / 
                          abs(baseline['Total Return']) * 100) if baseline['Total Return'] != 0 else 0
        sharpe_improvement = ((metrics['Sharpe Ratio'] - baseline['Sharpe Ratio']) / 
                             abs(baseline['Sharpe Ratio']) * 100) if baseline['Sharpe Ratio'] != 0 else 0
        dd_improvement = ((baseline['Max Drawdown'] - metrics['Max Drawdown']) / 
                         abs(baseline['Max Drawdown']) * 100) if baseline['Max Drawdown'] != 0 else 0
        
        print(f"{name:<15} {ret_improvement:>19.1f}% {sharpe_improvement:>19.1f}% {dd_improvement:>19.1f}%")
    
    print("="*80)
    
    return results


if __name__ == '__main__':
    
    START_DATE = '2019-01-01'
    END_DATE = '2021-12-31'
    
    TICKERS = [
        'NVDA', 'AAPL', 'MSFT', 'AMZN', 'GOOG', 'META', 'AVGO', 'TSLA', 'ORCL',
        'AMD', 'PLTR', 'NFLX', 'CSCO', 'IBM', 'CRM', 'INTC', 'MU', 'LRCX', 'AMAT',
        'BRK.B', 'JPM', 'V', 'MA', 'BAC', 'MS', 'GS', 'WFC', 'AXP', 'C',
        'LLY', 'JNJ', 'ABBV', 'UNH', 'MRK', 'TMO',
        'WMT', 'COST', 'HD', 'PG', 'KO', 'MCD', 'PM',
        'XOM', 'CVX', 'GE', 'CAT', 'RTX', 'LIN', 'TMUS'
    ]
    
    results = run_vix_sensitivity_analysis(
        tickers=TICKERS,
        start_date=START_DATE,
        end_date=END_DATE,
        stock_filepath='Stock_Data_History.xlsx',
        vix_filepath='VIXCLS.xlsx'
    )
    
    print("\nAnalysis complete.")
    print(f"Generated 4 result plots for {START_DATE} to {END_DATE}")
