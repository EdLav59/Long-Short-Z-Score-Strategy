"""
Environment verification script for Long/Short Z-Score Screening Strategy.
Tests package installations, Yahoo Finance connectivity, and statistical libraries.
"""

import sys
import importlib
from datetime import datetime, timedelta

REQUIRED_PACKAGES = {
    'yfinance': '0.2.48',
    'pandas': '2.2.3',
    'numpy': '1.26.4',
    'matplotlib': '3.9.3',
    'seaborn': '0.13.2',
    'scipy': '1.14.1',
    'statsmodels': '0.14.4'
}

def check_python_version():
    """Verify Python version compatibility."""
    version = sys.version_info
    print(f"Python Version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 9 and version.minor <= 10:
        print("[OK] Python version is compatible (3.9 or 3.10 recommended)\n")
        return True
    elif version.major == 3 and version.minor >= 9:
        print("[WARN] Python 3.11+ may have compatibility issues with statsmodels\n")
        return True
    else:
        print("[ERROR] Python 3.9+ required\n")
        return False

def check_packages():
    """Verify all required packages are installed with correct versions."""
    print("Checking Package Installations...")
    print("-" * 50)
    
    missing = []
    version_mismatches = []
    
    for pkg, expected_version in REQUIRED_PACKAGES.items():
        try:
            lib = importlib.import_module(pkg)
            installed_version = lib.__version__
            
            # Check major.minor version match (ignore patch)
            expected_major_minor = '.'.join(expected_version.split('.')[:2])
            installed_major_minor = '.'.join(installed_version.split('.')[:2])
            
            if expected_major_minor == installed_major_minor:
                print(f"[OK] {pkg:12} {installed_version:10} (expected {expected_version})")
            else:
                print(f"[WARN] {pkg:12} {installed_version:10} (expected {expected_version})")
                version_mismatches.append(pkg)
                
        except ImportError:
            print(f"[MISSING] {pkg}")
            missing.append(pkg)
    
    print()
    
    if missing:
        print(f"[ACTION REQUIRED] Install missing packages:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    if version_mismatches:
        print(f"[WARNING] Version mismatches detected for: {', '.join(version_mismatches)}")
        print("Consider running: pip install -r requirements.txt --upgrade")
    
    return True

def test_yahoo_finance():
    """Test Yahoo Finance API connectivity with European ticker."""
    print("\nTesting Yahoo Finance API...")
    print("-" * 50)
    
    try:
        import yfinance as yf
        
        # Test with a highly liquid European ticker
        test_ticker = "AIR.PA"  # Airbus
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        print(f"Downloading {test_ticker} (last 30 days)...")
        df = yf.download(
            test_ticker,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            progress=False
        )
        
        if df.empty:
            print(f"[ERROR] API returned empty data for {test_ticker}")
            return False
        
        print(f"[OK] Successfully retrieved {len(df)} data points")
        print(f"     Date range: {df.index[0].date()} to {df.index[-1].date()}")
        print(f"     Last close: EUR {df['Adj Close'].iloc[-1]:.2f}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Yahoo Finance test failed: {str(e)}")
        return False

def test_statistical_libraries():
    """Test statistical analysis capabilities (statsmodels)."""
    print("\nTesting Statistical Libraries...")
    print("-" * 50)
    
    try:
        import pandas as pd
        import numpy as np
        from statsmodels.tsa.stattools import adfuller
        from scipy import stats
        
        # Create synthetic mean-reverting series
        np.random.seed(42)
        n = 500
        
        # AR(1) process with mean reversion
        ar_param = 0.7  # Autoregressive parameter
        series = np.zeros(n)
        for t in range(1, n):
            series[t] = ar_param * series[t-1] + np.random.randn()
        
        # Test ADF (should detect stationarity)
        adf_result = adfuller(series, maxlag=20)
        
        print(f"[OK] Statsmodels ADF test executed")
        print(f"     ADF Statistic: {adf_result[0]:.4f}")
        print(f"     P-value: {adf_result[1]:.4f}")
        print(f"     Result: {'Stationary' if adf_result[1] < 0.05 else 'Non-stationary'}")
        
        # Test Jarque-Bera
        jb_stat, jb_pvalue = stats.jarque_bera(series)
        print(f"\n[OK] Scipy Jarque-Bera test executed")
        print(f"     JB Statistic: {jb_stat:.4f}")
        print(f"     P-value: {jb_pvalue:.4f}")
        
        return True
        
    except ImportError as e:
        print(f"[ERROR] Cannot import statistical libraries: {str(e)}")
        print("     Install with: pip install statsmodels scipy")
        return False
    except Exception as e:
        print(f"[ERROR] Statistical test failed: {str(e)}")
        return False

def test_data_processing():
    """Test basic data processing capabilities."""
    print("\nTesting Data Processing...")
    print("-" * 50)
    
    try:
        import pandas as pd
        import numpy as np
        
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100)
        prices = pd.DataFrame({
            'Stock_A': np.random.randn(100).cumsum() + 100,
            'Stock_B': np.random.randn(100).cumsum() + 100
        }, index=dates)
        
        # Test rolling calculations
        rolling_mean = prices.rolling(window=20).mean()
        rolling_std = prices.rolling(window=20).std()
        z_scores = (prices - rolling_mean) / rolling_std
        
        print(f"[OK] Created sample dataset: {prices.shape}")
        print(f"[OK] Calculated rolling statistics")
        print(f"[OK] Z-scores computed successfully")
        print(f"     Z-score range: [{z_scores.min().min():.2f}, {z_scores.max().max():.2f}]")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Data processing test failed: {str(e)}")
        return False

def main():
    """Run all verification tests."""
    print("=" * 50)
    print("ENVIRONMENT VERIFICATION")
    print("Z-Score Screening Strategy")
    print("=" * 50)
    print()
    
    results = {
        'Python Version': check_python_version(),
        'Package Installation': check_packages(),
        'Yahoo Finance API': test_yahoo_finance(),
        'Statistical Libraries': test_statistical_libraries(),
        'Data Processing': test_data_processing()
    }
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    for test_name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {test_name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\nAll tests passed. Ready to run the strategy.")
        print("\nNext steps:")
        print("  1. Review configuration in longshort_zscore.py")
        print("  2. Execute: python longshort_zscore.py")
        print("  3. Results will be saved to results/ directory")
    else:
        print("\nSome tests failed. Resolve issues before running strategy.")
        print("\nCommon solutions:")
        print("  - Package errors: pip install -r requirements.txt")
        print("  - Network errors: Check internet connection")
        print("  - Python version: Use Python 3.9 or 3.10")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
