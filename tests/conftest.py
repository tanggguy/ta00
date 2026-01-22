"""
Tests configuration and shared fixtures.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n = 500
    
    dates = pd.date_range('2024-01-01', periods=n, freq='4h')
    
    # Generate realistic price data with trend
    base_price = 42000
    returns = np.random.normal(0.0001, 0.02, n)
    prices = base_price * np.cumprod(1 + returns)
    
    # Generate OHLC from prices
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.uniform(-0.005, 0.005, n)),
        'high': prices * (1 + np.random.uniform(0.001, 0.02, n)),
        'low': prices * (1 - np.random.uniform(0.001, 0.02, n)),
        'close': prices,
        'volume': np.random.uniform(100, 10000, n)
    })
    
    # Ensure high >= low and high >= open/close
    df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
    df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
    
    return df


@pytest.fixture
def trending_up_data():
    """Generate upward trending data for testing."""
    n = 200
    dates = pd.date_range('2024-01-01', periods=n, freq='4h')
    
    # Strong upward trend
    prices = 40000 + np.linspace(0, 10000, n) + np.random.normal(0, 200, n)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices * 0.998,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': np.random.uniform(100, 1000, n)
    })
    
    return df


@pytest.fixture
def trending_down_data():
    """Generate downward trending data for testing."""
    n = 200
    dates = pd.date_range('2024-01-01', periods=n, freq='4h')
    
    # Strong downward trend
    prices = 50000 - np.linspace(0, 10000, n) + np.random.normal(0, 200, n)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices * 1.002,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': np.random.uniform(100, 1000, n)
    })
    
    return df


@pytest.fixture
def sideways_data():
    """Generate sideways/ranging data for testing."""
    n = 200
    dates = pd.date_range('2024-01-01', periods=n, freq='4h')
    
    # Sideways movement
    prices = 45000 + np.random.normal(0, 500, n)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices * 0.999,
        'high': prices * 1.005,
        'low': prices * 0.995,
        'close': prices,
        'volume': np.random.uniform(100, 1000, n)
    })
    
    return df


@pytest.fixture
def empty_dataframe():
    """Empty DataFrame for error testing."""
    return pd.DataFrame()


@pytest.fixture
def minimal_dataframe():
    """Minimal valid DataFrame."""
    return pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=10, freq='4h'),
        'close': [100, 101, 102, 101, 100, 99, 98, 99, 100, 101]
    })
