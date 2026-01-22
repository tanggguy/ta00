"""Tests for trading strategies."""

import pytest
import pandas as pd
import numpy as np


class TestTrendFollowing:
    """Tests for Trend Following strategy."""
    
    def test_initialization_default_params(self):
        """Test TrendFollowing initializes with default parameters."""
        from src.strategies.trend_following import TrendFollowing
        
        strategy = TrendFollowing()
        
        assert strategy.name == 'trend_following'
        assert strategy.sma_short == 20
        assert strategy.sma_long == 50
    
    def test_initialization_custom_params(self):
        """Test TrendFollowing with custom parameters."""
        from src.strategies.trend_following import TrendFollowing
        
        strategy = TrendFollowing(sma_short=10, sma_long=30)
        
        assert strategy.sma_short == 10
        assert strategy.sma_long == 30
    
    def test_invalid_params_raises_error(self):
        """Test invalid parameters raise ValueError."""
        from src.strategies.trend_following import TrendFollowing
        
        with pytest.raises(ValueError):
            TrendFollowing(sma_short=50, sma_long=20)
    
    def test_generate_signals_output_shape(self, sample_ohlcv_data):
        """Test generate_signals returns correct shape."""
        from src.strategies.trend_following import TrendFollowing
        
        strategy = TrendFollowing()
        result = strategy.generate_signals(sample_ohlcv_data)
        
        assert len(result) == len(sample_ohlcv_data)
    
    def test_generate_signals_has_signal_column(self, sample_ohlcv_data):
        """Test generate_signals adds signal column."""
        from src.strategies.trend_following import TrendFollowing
        
        strategy = TrendFollowing()
        result = strategy.generate_signals(sample_ohlcv_data)
        
        assert 'signal' in result.columns
        assert 'sma_short' in result.columns
        assert 'sma_long' in result.columns
    
    def test_generate_signals_valid_values(self, sample_ohlcv_data):
        """Test signal values are valid (-1 or 1)."""
        from src.strategies.trend_following import TrendFollowing
        
        strategy = TrendFollowing()
        result = strategy.generate_signals(sample_ohlcv_data)
        
        # Skip NaN values from SMA calculation
        valid_signals = result['signal'].dropna()
        assert set(valid_signals.unique()).issubset({-1, 1})
    
    def test_uptrend_generates_buy_signals(self, trending_up_data):
        """Test uptrend generates predominantly buy signals."""
        from src.strategies.trend_following import TrendFollowing
        
        strategy = TrendFollowing(sma_short=10, sma_long=30)
        result = strategy.generate_signals(trending_up_data)
        
        # After SMAs stabilize, should see more buys than sells
        signals = result['signal'].iloc[50:]
        buy_count = (signals == 1).sum()
        sell_count = (signals == -1).sum()
        
        assert buy_count > sell_count
    
    def test_missing_close_column_raises_error(self):
        """Test missing close column raises ValueError."""
        from src.strategies.trend_following import TrendFollowing
        
        strategy = TrendFollowing()
        bad_df = pd.DataFrame({'open': [1, 2, 3], 'high': [2, 3, 4]})
        
        with pytest.raises(ValueError):
            strategy.generate_signals(bad_df)
    
    def test_empty_dataframe_raises_error(self, empty_dataframe):
        """Test empty DataFrame raises ValueError."""
        from src.strategies.trend_following import TrendFollowing
        
        strategy = TrendFollowing()
        
        with pytest.raises(ValueError):
            strategy.generate_signals(empty_dataframe)


class TestMeanReversion:
    """Tests for Mean Reversion strategy."""
    
    def test_initialization_default_params(self):
        """Test MeanReversion initializes with default parameters."""
        from src.strategies.mean_reversion import MeanReversion
        
        strategy = MeanReversion()
        
        assert strategy.name == 'mean_reversion'
        assert strategy.rsi_period == 14
        assert strategy.oversold == 30
        assert strategy.overbought == 70
    
    def test_initialization_custom_params(self):
        """Test MeanReversion with custom parameters."""
        from src.strategies.mean_reversion import MeanReversion
        
        strategy = MeanReversion(rsi_period=10, oversold=20, overbought=80)
        
        assert strategy.rsi_period == 10
        assert strategy.oversold == 20
        assert strategy.overbought == 80
    
    def test_invalid_thresholds_raises_error(self):
        """Test invalid thresholds raise ValueError."""
        from src.strategies.mean_reversion import MeanReversion
        
        with pytest.raises(ValueError):
            MeanReversion(oversold=70, overbought=30)
    
    def test_generate_signals_output_shape(self, sample_ohlcv_data):
        """Test generate_signals returns correct shape."""
        from src.strategies.mean_reversion import MeanReversion
        
        strategy = MeanReversion()
        result = strategy.generate_signals(sample_ohlcv_data)
        
        assert len(result) == len(sample_ohlcv_data)
    
    def test_generate_signals_has_rsi_column(self, sample_ohlcv_data):
        """Test generate_signals adds RSI column."""
        from src.strategies.mean_reversion import MeanReversion
        
        strategy = MeanReversion()
        result = strategy.generate_signals(sample_ohlcv_data)
        
        assert 'rsi' in result.columns
        assert 'signal' in result.columns
    
    def test_generate_signals_valid_values(self, sample_ohlcv_data):
        """Test signal values are valid (-1, 0, or 1)."""
        from src.strategies.mean_reversion import MeanReversion
        
        strategy = MeanReversion()
        result = strategy.generate_signals(sample_ohlcv_data)
        
        valid_signals = result['signal'].dropna()
        assert set(valid_signals.unique()).issubset({-1, 0, 1})
    
    def test_rsi_range(self, sample_ohlcv_data):
        """Test RSI values are in 0-100 range."""
        from src.strategies.mean_reversion import MeanReversion
        
        strategy = MeanReversion()
        result = strategy.generate_signals(sample_ohlcv_data)
        
        rsi_values = result['rsi'].dropna()
        assert (rsi_values >= 0).all()
        assert (rsi_values <= 100).all()


class TestMomentum:
    """Tests for Momentum strategy."""
    
    def test_initialization_default_params(self):
        """Test Momentum initializes with default parameters."""
        from src.strategies.momentum import Momentum
        
        strategy = Momentum()
        
        assert strategy.name == 'momentum'
        assert strategy.macd_fast == 12
        assert strategy.macd_slow == 26
        assert strategy.macd_signal == 9
    
    def test_initialization_custom_params(self):
        """Test Momentum with custom parameters."""
        from src.strategies.momentum import Momentum
        
        strategy = Momentum(macd_fast=8, macd_slow=21, macd_signal=5)
        
        assert strategy.macd_fast == 8
        assert strategy.macd_slow == 21
        assert strategy.macd_signal == 5
    
    def test_invalid_params_raises_error(self):
        """Test invalid parameters raise ValueError."""
        from src.strategies.momentum import Momentum
        
        with pytest.raises(ValueError):
            Momentum(macd_fast=30, macd_slow=20)
    
    def test_generate_signals_output_shape(self, sample_ohlcv_data):
        """Test generate_signals returns correct shape."""
        from src.strategies.momentum import Momentum
        
        strategy = Momentum()
        result = strategy.generate_signals(sample_ohlcv_data)
        
        assert len(result) == len(sample_ohlcv_data)
    
    def test_generate_signals_has_macd_columns(self, sample_ohlcv_data):
        """Test generate_signals adds MACD columns."""
        from src.strategies.momentum import Momentum
        
        strategy = Momentum()
        result = strategy.generate_signals(sample_ohlcv_data)
        
        assert 'macd' in result.columns
        assert 'macd_signal_line' in result.columns
        assert 'macd_histogram' in result.columns
        assert 'signal' in result.columns
    
    def test_generate_signals_valid_values(self, sample_ohlcv_data):
        """Test signal values are valid (-1 or 1)."""
        from src.strategies.momentum import Momentum
        
        strategy = Momentum()
        result = strategy.generate_signals(sample_ohlcv_data)
        
        valid_signals = result['signal'].dropna()
        assert set(valid_signals.unique()).issubset({-1, 1})


class TestStrategyRegistry:
    """Tests for strategy registry."""
    
    def test_list_strategies(self):
        """Test listing available strategies."""
        from src.strategies import list_strategies
        
        strategies = list_strategies()
        
        assert 'trend_following' in strategies
        assert 'mean_reversion' in strategies
        assert 'momentum' in strategies
    
    def test_get_strategy_by_name(self):
        """Test getting strategy by name."""
        from src.strategies import get_strategy
        
        strategy = get_strategy('trend_following')
        
        assert strategy.name == 'trend_following'
    
    def test_get_unknown_strategy_raises_error(self):
        """Test getting unknown strategy raises ValueError."""
        from src.strategies import get_strategy
        
        with pytest.raises(ValueError):
            get_strategy('unknown_strategy')
