"""
Tests for src/data_trends.py
Tests Google Trends analysis functionality
"""

import pytest
from unittest.mock import patch, Mock, MagicMock
from datetime import datetime
import pandas as pd
import numpy as np

from src.data_trends import GoogleTrendsAnalyzer, check_momentum_confirmation


class TestGoogleTrendsAnalyzer:
    """Test cases for GoogleTrendsAnalyzer class."""
    
    @pytest.fixture
    def analyzer(self):
        """Create an analyzer with test config."""
        config = {
            'enabled': True,
            'keyword': 'Bitcoin',
            'timeframe': 'now 7-d',
            'geo': '',
            'surge_threshold_pct': 150,
            'use_as_momentum_confirmation': True,
            'cache_duration_seconds': 14400
        }
        
        with patch('src.data_trends.TrendReq'):
            analyzer = GoogleTrendsAnalyzer(config=config)
        
        return analyzer
    
    @pytest.fixture
    def mock_trends_data(self):
        """Mock Google Trends DataFrame."""
        # Create a DataFrame with 7 days of data
        dates = pd.date_range(end=datetime.now(), periods=7, freq='D')
        # Values: avg=50, current=85 -> surge ratio = 1.7 (170%)
        values = [45, 50, 48, 55, 52, 45, 85]
        
        df = pd.DataFrame({
            'Bitcoin': values,
            'isPartial': [False] * 7
        }, index=dates)
        
        return df
    
    def test_init_with_config(self, analyzer):
        """Test analyzer initialization with config."""
        assert analyzer.config['keyword'] == 'Bitcoin'
        assert analyzer.config['surge_threshold_pct'] == 150
    
    @patch('src.data_trends.TrendReq')
    def test_get_interest_data_success(self, mock_trendreq, mock_trends_data):
        """Test successful trends data fetch."""
        # Setup mock
        mock_pytrends = MagicMock()
        mock_pytrends.interest_over_time.return_value = mock_trends_data
        mock_trendreq.return_value = mock_pytrends
        
        config = {
            'enabled': True,
            'keyword': 'Bitcoin',
            'surge_threshold_pct': 150
        }
        analyzer = GoogleTrendsAnalyzer(config=config)
        
        data = analyzer.get_interest_data()
        
        # Current = 85, avg ~= 54.3, surge ratio ~= 1.57 (157%)
        assert data['error'] is False
        assert data['current_interest'] == 85
        assert data['is_surging'] is True  # 157% > 150%
    
    @patch('src.data_trends.TrendReq')
    def test_get_interest_data_no_surge(self, mock_trendreq):
        """Test when there's no surge in interest."""
        # Create data with no surge
        dates = pd.date_range(end=datetime.now(), periods=7, freq='D')
        values = [50, 52, 48, 50, 51, 49, 52]  # avg ~= 50, current = 52
        
        mock_df = pd.DataFrame({
            'Bitcoin': values,
            'isPartial': [False] * 7
        }, index=dates)
        
        mock_pytrends = MagicMock()
        mock_pytrends.interest_over_time.return_value = mock_df
        mock_trendreq.return_value = mock_pytrends
        
        config = {'surge_threshold_pct': 150}
        analyzer = GoogleTrendsAnalyzer(config=config)
        
        data = analyzer.get_interest_data()
        
        # surge ratio ~= 1.04 (104%) < 150%
        assert data['error'] is False
        assert data['is_surging'] is False
    
    @patch('src.data_trends.TrendReq')
    def test_get_interest_data_api_error(self, mock_trendreq):
        """Test handling of API errors."""
        mock_pytrends = MagicMock()
        mock_pytrends.interest_over_time.side_effect = Exception("API Error")
        mock_trendreq.return_value = mock_pytrends
        
        analyzer = GoogleTrendsAnalyzer(config={})
        data = analyzer.get_interest_data()
        
        assert data['error'] is True
        assert data['is_surging'] is False
    
    @patch('src.data_trends.TrendReq')
    def test_is_momentum_confirmed_true(self, mock_trendreq, mock_trends_data):
        """Test momentum confirmation when surging."""
        mock_pytrends = MagicMock()
        mock_pytrends.interest_over_time.return_value = mock_trends_data
        mock_trendreq.return_value = mock_pytrends
        
        config = {
            'enabled': True,
            'use_as_momentum_confirmation': True,
            'surge_threshold_pct': 150
        }
        analyzer = GoogleTrendsAnalyzer(config=config)
        
        confirmed, reason = analyzer.is_momentum_confirmed()
        
        assert confirmed is True
        assert "CONFIRMED" in reason
    
    @patch('src.data_trends.TrendReq')
    def test_is_momentum_confirmed_false(self, mock_trendreq):
        """Test momentum not confirmed when no surge."""
        dates = pd.date_range(end=datetime.now(), periods=7, freq='D')
        values = [50, 50, 50, 50, 50, 50, 50]
        
        mock_df = pd.DataFrame({
            'Bitcoin': values,
            'isPartial': [False] * 7
        }, index=dates)
        
        mock_pytrends = MagicMock()
        mock_pytrends.interest_over_time.return_value = mock_df
        mock_trendreq.return_value = mock_pytrends
        
        config = {
            'enabled': True,
            'use_as_momentum_confirmation': True,
            'surge_threshold_pct': 150
        }
        analyzer = GoogleTrendsAnalyzer(config=config)
        
        confirmed, reason = analyzer.is_momentum_confirmed()
        
        assert confirmed is False
        assert "No surge" in reason
    
    def test_is_momentum_disabled(self):
        """Test when momentum confirmation is disabled."""
        config = {
            'enabled': True,
            'use_as_momentum_confirmation': False
        }
        
        with patch('src.data_trends.TrendReq'):
            analyzer = GoogleTrendsAnalyzer(config=config)
        
        confirmed, reason = analyzer.is_momentum_confirmed()
        
        assert confirmed is True
        assert "disabled" in reason.lower()
    
    def test_trends_disabled(self):
        """Test when trends filtering is fully disabled."""
        config = {'enabled': False}
        
        with patch('src.data_trends.TrendReq'):
            analyzer = GoogleTrendsAnalyzer(config=config)
        
        confirmed, reason = analyzer.is_momentum_confirmed()
        
        assert confirmed is True
        assert "disabled" in reason.lower()
    
    def test_cache_mechanism(self, analyzer):
        """Test caching works correctly."""
        import time
        
        # Simulate cached data
        analyzer._cache = {
            'keyword': 'Bitcoin',
            'current_interest': 75,
            'avg_7d': 50.0,
            'surge_ratio': 1.5,
            'surge_ratio_pct': 150.0,
            'is_surging': True,
            'error': False,
            'timestamp': datetime.now().isoformat()
        }
        analyzer._cache_time = time.time()
        
        # Should return cached data
        data = analyzer.get_interest_data()
        
        assert data['current_interest'] == 75
        assert data['is_surging'] is True
    
    def test_clear_cache(self, analyzer):
        """Test cache clearing."""
        import time
        analyzer._cache = {'test': 'data'}
        analyzer._cache_time = time.time()
        
        analyzer.clear_cache()
        
        assert analyzer._cache is None
        assert analyzer._cache_time is None


class TestCheckMomentumConfirmationFunction:
    """Test the convenience function."""
    
    @patch.object(GoogleTrendsAnalyzer, 'is_momentum_confirmed')
    @patch('src.data_trends.TrendReq')
    def test_check_momentum_confirmation(self, mock_trendreq, mock_is_confirmed):
        """Test the convenience function."""
        mock_is_confirmed.return_value = (True, "Momentum confirmed")
        
        confirmed, reason = check_momentum_confirmation("Bitcoin")
        
        assert confirmed is True
