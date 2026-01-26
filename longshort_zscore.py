"""
Long/Short Z-Score Mean Reversion Strategy
European Large-Cap Equity Universe (2019-2024)

Strategy Logic:
- Calculate 252-day rolling Z-scores for each stock
- Long positions: Z-score < -1.5 (undervalued)
- Short positions: Z-score > 1.5 (overvalued)
- Exit: Z-score returns to [-0.5, 0.5] (mean reversion)
- Rebalancing: Monthly (first business day)
- Transaction costs: 10 basis points per trade
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

# Configuration
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

class ZScoreStrategy:
    """
    Statistical arbitrage strategy using Z-score mean reversion signals.
    Implements monthly rebalancing with robust data handling.
    """
    
    def __init__(self, tickers, start_date, end_date, initial_capital=100000):
        """
        Initialize strategy parameters.
        
        Args:
            tickers: List of ticker symbols (Yahoo Finance format)
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)
            initial_capital: Starting portfolio value in EUR
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        
        # Strategy Parameters
        self.lookback_window = 252      # Rolling window for Z-score (1 year)
        self.entry_threshold = 1.5      # |Z-score| > 1.5 triggers entry
        self.exit_threshold = 0.5       # |Z-score| < 0.5 triggers exit
        self.transaction_cost = 0.0010  # 10 basis points
        self.rebalance_freq = 'MS'      # Monthly start frequency
        
        # Data containers
        self.data = pd.DataFrame()
        self.z_scores = pd.DataFrame()
        self.signals = pd.DataFrame()
        self.positions = pd.DataFrame()
        self.portfolio_value = pd.Series(dtype=float)
        self.daily_returns = pd.Series(dtype=float)
        self.trades = pd.DataFrame()
        self.metrics = {}
        
        # Create output directories
        os.makedirs('data', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Z-SCORE LONG/SHORT STRATEGY INITIALIZATION")
        print(f"{'='*60}")
        print(f"Universe:       {len(tickers)} European stocks")
        print(f"Period:         {start_date} to {end_date}")
        print(f"Initial Capital: EUR {initial_capital:,.0f}")
        print(f"{'='*60}\n")
    
    def download_data(self, max_retries=3, retry_delay=2):
        """
        Download historical adjusted close prices with robust error handling.
        
        Args:
            max_retries: Maximum number of retry attempts per ticker
            retry_delay: Seconds to wait between retries
        """
        print("[STEP 1/6] Downloading Historical Data...")
        print("-" * 60)
        
        valid_data = {}
        failed_tickers = []
        
        # Calculate minimum required data points (70% of expected trading days)
        date_range = pd.date_range(self.start_date, self.end_date, freq='B')
        min_data_points = int(len(date_range) * 0.70)
        
        for i, ticker in enumerate(self.tickers, 1):
            success = False
            
            for attempt in range(1, max_retries + 1):
                try:
                    # Download with progress suppression
                    df = yf.download(
                        ticker,
                        start=self.start_date,
                        end=self.end_date,
                        progress=False,
                        show_errors=False
                    )
                    
                    # Validation checks
                    if df.empty:
                        print(f"  [{i:2d}/{len(self.tickers)}] {ticker:12} - No data available")
                        break
                    
                    if len(df) < min_data_points:
                        print(f"  [{i:2d}/{len(self.tickers)}] {ticker:12} - Insufficient data ({len(df)} points)")
                        break
                    
                    # Success
                    valid_data[ticker] = df['Adj Close']
                    print(f"  [{i:2d}/{len(self.tickers)}] {ticker:12} - OK ({len(df)} points)")
                    success = True
                    break
                    
                except Exception as e:
                    if attempt < max_retries:
                        print(f"  [{i:2d}/{len(self.tickers)}] {ticker:12} - Retry {attempt}/{max_retries}")
                        time.sleep(retry_delay)
                    else:
                        print(f"  [{i:2d}/{len(self.tickers)}] {ticker:12} - FAILED after {max_retries} attempts")
            
            if not success:
                failed_tickers.append(ticker)
        
        # Check minimum universe size
        if len(valid_data) < 10:
            raise RuntimeError(
                f"CRITICAL: Only {len(valid_data)} tickers retrieved. "
                f"Minimum 10 required for diversification."
            )
        
        # Construct price matrix
        self.data = pd.DataFrame(valid_data)
        
        # Handle missing values
        missing_before = self.data.isna().sum().sum()
        self.data = self.data.interpolate(method='linear', limit=5, limit_direction='both')
        self.data.dropna(inplace=True)
        missing_after = self.data.isna().sum().sum()
        
        print(f"\n  Data Cleaning:")
        print(f"    - Missing values filled: {missing_before}")
        print(f"    - Remaining NaNs dropped: {missing_after}")
        print(f"\n  Final Universe: {self.data.shape[1]} stocks, {self.data.shape[0]} trading days")
        
        if failed_tickers:
            print(f"  Failed Tickers ({len(failed_tickers)}): {', '.join(failed_tickers[:5])}" + 
                  ("..." if len(failed_tickers) > 5 else ""))
        
        # Save cleaned data
        self.data.to_csv('data/cleaned_prices.csv')
        print(f"  Saved to data/cleaned_prices.csv\n")
    
    def calculate_indicators(self):
        """Calculate rolling Z-scores for mean reversion signals."""
        print("[STEP 2/6] Calculating Z-Score Indicators...")
        print("-" * 60)
        
        # Rolling statistics
        rolling_mean = self.data.rolling(window=self.lookback_window, min_periods=self.lookback_window).mean()
        rolling_std = self.data.rolling(window=self.lookback_window, min_periods=self.lookback_window).std()
        
        # Z-score calculation: (Price - Mean) / StdDev
        self.z_scores = (self.data - rolling_mean) / rolling_std
        
        # Remove initial warm-up period
        self.z_scores = self.z_scores.dropna()
        self.data = self.data.loc[self.z_scores.index]
        
        # Summary statistics
        print(f"  Z-Score Range: [{self.z_scores.min().min():.2f}, {self.z_scores.max().max():.2f}]")
        print(f"  Mean: {self.z_scores.mean().mean():.3f}")
        print(f"  Std Dev: {self.z_scores.std().mean():.3f}")
        print(f"  Valid data points: {len(self.z_scores)}")
        print(f"  Indicators calculated\n")
    
    def generate_signals(self):
        """
        Generate trading signals based on Z-score thresholds.
        Monthly rebalancing with position flip capability.
        """
        print("[STEP 3/6] Generating Trading Signals...")
        print("-" * 60)
        
        # Generate daily signals
        signals = pd.DataFrame(0, index=self.z_scores.index, columns=self.z_scores.columns)
        
        # Long signal: Z-score < -entry_threshold (undervalued)
        signals[self.z_scores < -self.entry_threshold] = 1
        
        # Short signal: Z-score > +entry_threshold (overvalued)
        signals[self.z_scores > self.entry_threshold] = -1
        
        # Exit signal: Z-score returns to neutral band
        neutral_band = (self.z_scores > -self.exit_threshold) & (self.z_scores < self.exit_threshold)
        signals[neutral_band] = 0
        
        # Apply monthly rebalancing frequency
        # Keep positions only on rebalance dates, forward fill between
        rebalance_dates = signals.resample(self.rebalance_freq).first().index
        monthly_signals = signals.loc[rebalance_dates]
        
        # Forward fill to maintain positions between rebalances
        self.positions = monthly_signals.reindex(signals.index, method='ffill').fillna(0)
        
        # Calculate signal statistics
        long_signals = (self.positions == 1).sum().sum()
        short_signals = (self.positions == -1).sum().sum()
        flat_signals = (self.positions == 0).sum().sum()
        total_signals = long_signals + short_signals + flat_signals
        
        print(f"  Rebalance Frequency: Monthly ({len(rebalance_dates)} rebalances)")
        print(f"  Signal Distribution:")
        print(f"    Long:  {long_signals:6d} ({long_signals/total_signals*100:5.1f}%)")
        print(f"    Short: {short_signals:6d} ({short_signals/total_signals*100:5.1f}%)")
        print(f"    Flat:  {flat_signals:6d} ({flat_signals/total_signals*100:5.1f}%)")
        
        # Save positions matrix
        self.positions.to_csv('results/positions.csv')
        print(f"  Saved to results/positions.csv\n")
    
    def run_backtest(self):
        """
        Execute backtest with transaction costs and equal-weight allocation.
        """
        print("[STEP 4/6] Running Backtest Engine...")
        print("-" * 60)
        
        # Calculate daily returns
        returns = self.data.pct_change().fillna(0)
        
        # Lag positions by 1 day (trade on T+1 after signal on T)
        lagged_positions = self.positions.shift(1).fillna(0)
        
        # Position changes for transaction costs
        position_changes = lagged_positions.diff().abs().fillna(0)
        
        # Count active positions per day for equal weighting
        active_positions = lagged_positions.abs().sum(axis=1)
        active_positions = active_positions.replace(0, 1)  # Avoid division by zero
        
        # Daily strategy returns (equal-weighted across active positions)
        strategy_returns_gross = (lagged_positions * returns).sum(axis=1) / active_positions
        
        # Transaction costs (10 bps per trade)
        daily_costs = (position_changes.sum(axis=1) / active_positions) * self.transaction_cost
        
        # Net daily returns
        self.daily_returns = strategy_returns_gross - daily_costs
        
        # Cumulative portfolio value
        self.portfolio_value = self.initial_capital * (1 + self.daily_returns).cumprod()
        
        # Generate trade log
        self._generate_trade_log(position_changes)
        
        # Summary
        total_trades = position_changes.sum().sum()
        total_costs = daily_costs.sum() * self.initial_capital
        
        print(f"  Total Trades: {int(total_trades)}")
        print(f"  Transaction Costs: EUR {total_costs:,.0f}")
        print(f"  Average Active Positions: {active_positions.mean():.1f}")
        print(f"  Backtest complete\n")
        
        # Save equity curve
        self.portfolio_value.to_csv('results/equity_curve.csv')
    
    def _generate_trade_log(self, position_changes):
        """Generate detailed trade log for audit trail."""
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
        print("[STEP 5/6] Computing Performance Metrics...")
        print("-" * 60)
        
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
        
        # Sharpe and Sortino Ratios (risk-free rate = 0%)
        sharpe_ratio = cagr / volatility if volatility > 0 else 0
        sortino_ratio = cagr / downside_deviation if downside_deviation > 0 else 0
        
        # Drawdown analysis
        cumulative = (1 + self.daily_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar Ratio
        calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win rate and profit metrics
        winning_days = (self.daily_returns > 0).sum()
        losing_days = (self.daily_returns < 0).sum()
        win_rate = winning_days / len(self.daily_returns) if len(self.daily_returns) > 0 else 0
        
        avg_win = self.daily_returns[self.daily_returns > 0].mean()
        avg_loss = self.daily_returns[self.daily_returns < 0].mean()
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
        
        # Store metrics
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
        
        # Display metrics
        print(f"\n  {'='*56}")
        print(f"  {'PERFORMANCE SUMMARY':^56}")
        print(f"  {'='*56}")
        print(f"  Total Return:        {total_return:>10.2%}")
        print(f"  CAGR:                {cagr:>10.2%}")
        print(f"  Volatility:          {volatility:>10.2%}")
        print(f"  {'='*56}")
        print(f"  Sharpe Ratio:        {sharpe_ratio:>10.2f}")
        print(f"  Sortino Ratio:       {sortino_ratio:>10.2f}")
        print(f"  Calmar Ratio:        {calmar_ratio:>10.2f}")
        print(f"  {'='*56}")
        print(f"  Max Drawdown:        {max_drawdown:>10.2%}")
        print(f"  Win Rate:            {win_rate:>10.2%}")
        print(f"  Profit Factor:       {profit_factor:>10.2f}")
        print(f"  {'='*56}\n")
        
        # Save metrics
        with open('results/metrics.txt', 'w') as f:
            f.write("LONG/SHORT Z-SCORE STRATEGY - PERFORMANCE METRICS\n")
            f.write("="*60 + "\n\n")
            for key, value in self.metrics.items():
                if isinstance(value, float):
                    if abs(value) < 0.1:  # Percentage metrics
                        f.write(f"{key:.<40} {value:>12.2%}\n")
                    else:  # Ratio metrics
                        f.write(f"{key:.<40} {value:>12.4f}\n")
        
        print(f"  Saved to results/metrics.txt\n")
    
    def generate_visualizations(self):
        """Create comprehensive performance dashboard."""
        print("[STEP 6/6] Generating Visualizations...")
        print("-" * 60)
        
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
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
        
        # 4. Z-Score Distribution (Current)
        ax4 = fig.add_subplot(gs[0, 2])
        current_zscores = self.z_scores.iloc[-1].dropna()
        ax4.hist(current_zscores, bins=30, color='navy', alpha=0.6, edgecolor='black')
        ax4.axvline(-self.entry_threshold, color='green', linestyle='--', linewidth=2, label='Long Threshold')
        ax4.axvline(self.entry_threshold, color='red', linestyle='--', linewidth=2, label='Short Threshold')
        ax4.axvline(-self.exit_threshold, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
        ax4.axvline(self.exit_threshold, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label='Exit Band')
        ax4.set_title('Current Z-Score Distribution', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Z-Score', fontsize=10)
        ax4.set_ylabel('Frequency', fontsize=10)
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # 5. Rolling Sharpe Ratio
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
        
        # 6. Performance Metrics Table
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
        
        # Style header row
        for i in range(2):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Save figure
        plt.savefig('results/dashboard.png', dpi=150, bbox_inches='tight')
        print(f"  Dashboard saved to results/dashboard.png\n")
        
        plt.close()
    
    def run(self):
        """Execute complete strategy workflow."""
        start_time = datetime.now()
        
        try:
            self.download_data()
            self.calculate_indicators()
            self.generate_signals()
            self.run_backtest()
            self.calculate_metrics()
            self.generate_visualizations()
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            print(f"{'='*60}")
            print(f"EXECUTION COMPLETE")
            print(f"{'='*60}")
            print(f"Total Runtime: {elapsed:.1f} seconds")
            print(f"Results saved to: results/")
            print(f"{'='*60}\n")
            
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"CRITICAL ERROR")
            print(f"{'='*60}")
            print(f"Execution failed: {str(e)}")
            print(f"{'='*60}\n")
            raise

def main():
    """Main execution function."""
    
    # European Large-Cap Universe (Validated Tickers)
    EUROPEAN_TICKERS = [
        # France (CAC 40)
        "AIR.PA",   # Airbus
        "MC.PA",    # LVMH
        "OR.PA",    # L'Oréal
        "SAN.PA",   # Sanofi
        "TTE.PA",   # TotalEnergies
        "BNP.PA",   # BNP Paribas
        "SU.PA",    # Schneider Electric
        "KER.PA",   # Kering
        "AI.PA",    # Air Liquide
        "RMS.PA",   # Hermès
        "CS.PA",    # AXA
        "DG.PA",    # Vinci
        "VIE.PA",   # Veolia
        "BN.PA",    # Danone
        
        # Germany (DAX)
        "SAP.DE",   # SAP
        "SIE.DE",   # Siemens
        "ALV.DE",   # Allianz
        "DTE.DE",   # Deutsche Telekom
        "BMW.DE",   # BMW
        "VOW3.DE",  # Volkswagen
        "BAS.DE",   # BASF
        "ADS.DE",   # Adidas
        "MBG.DE",   # Mercedes-Benz
        "DB1.DE",   # Deutsche Börse
        
        # Netherlands
        "ASML.AS",  # ASML
        "INGA.AS",  # ING
        "ADYEN.AS", # Adyen
        "UNA.AS",   # Unilever
        "PHIA.AS",  # Philips
        
        # Spain (IBEX 35)
        "SAN.MC",   # Banco Santander
        "ITX.MC",   # Inditex
        "BBVA.MC",  # BBVA
        "IBE.MC",   # Iberdrola
        
        # Italy (FTSE MIB)
        "ISP.MI",   # Intesa Sanpaolo
        "ENEL.MI",  # Enel
        "ENI.MI",   # Eni
        "UCG.MI",   # UniCredit
    ]
    
    # Strategy Configuration
    CONFIG = {
        'tickers': EUROPEAN_TICKERS,
        'start_date': '2019-01-01',
        'end_date': '2024-01-01',
        'initial_capital': 100000  # EUR
    }
    
    # Execute Strategy
    strategy = ZScoreStrategy(**CONFIG)
    strategy.run()

if __name__ == "__main__":
    main()
