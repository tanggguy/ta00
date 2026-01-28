"""
Tests for src/data_sentiment.py
Tests CryptoPanic sentiment analysis functionality
"""

import os
import pytest
from unittest.mock import patch, Mock
from datetime import datetime

from src.data_sentiment import CryptoPanicClient, check_sentiment


class TestCryptoPanicClient:
    """Test cases for CryptoPanicClient class."""
    
    @pytest.fixture
    def client(self):
        """Create a client with test config."""
        config = {
            'enabled': True,
            'bearish_threshold_pct': 70,
            'lookback_hours': 24,
            'block_on_error': True,
            'cache_duration_seconds': 14400
        }
        return CryptoPanicClient(api_key="test_key", config=config)
    
    @pytest.fixture
    def mock_posts_response(self):
        """Mock API response with posts data."""
        return {
            'results': [
                {
                    'title': 'Bitcoin pumps',
                    'votes': {'positive': 100, 'negative': 30, 'neutral': 20}
                },
                {
                    'title': 'Market update',
                    'votes': {'positive': 50, 'negative': 20, 'neutral': 10}
                },
                {
                    'title': 'BTC analysis',
                    'votes': {'positive': 30, 'negative': 60, 'neutral': 10}
                }
            ]
        }
    
    def test_init_without_api_key(self):
        """Test client initialization without API key."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove the env var if it exists
            if 'CRYPTOPANIC_API_KEY' in os.environ:
                del os.environ['CRYPTOPANIC_API_KEY']
            
            # Should initialize but log warning
            client = CryptoPanicClient(api_key=None, config={})
            assert client.api_key is None
    
    def test_init_with_api_key(self, client):
        """Test client initialization with API key."""
        assert client.api_key == "test_key"
        assert client.config['bearish_threshold_pct'] == 70
    
    @patch('src.data_sentiment.requests.get')
    def test_get_sentiment_success(self, mock_get, client, mock_posts_response):
        """Test successful sentiment calculation."""
        mock_get.return_value = Mock(
            status_code=200,
            json=lambda: mock_posts_response
        )
        mock_get.return_value.raise_for_status = Mock()
        
        sentiment = client.get_sentiment("BTC")
        
        # Total votes: 180 bullish, 110 bearish, 40 neutral = 330 total
        # Bearish pct = 110/330 * 100 = 33.33%
        assert sentiment['error'] is False
        assert sentiment['total_votes'] == 330
        assert 33 < sentiment['bearish_pct'] < 34  # ~33.33%
        assert sentiment['is_bearish_high'] is False  # < 70%
        assert sentiment['can_buy'] is True
    
    @patch('src.data_sentiment.requests.get')
    def test_get_sentiment_high_bearish(self, mock_get, client):
        """Test sentiment when bearish is above threshold."""
        mock_response = {
            'results': [
                {'votes': {'positive': 10, 'negative': 80, 'neutral': 10}}
            ]
        }
        mock_get.return_value = Mock(
            status_code=200,
            json=lambda: mock_response
        )
        mock_get.return_value.raise_for_status = Mock()
        
        sentiment = client.get_sentiment("BTC")
        
        # 80% bearish > 70% threshold
        assert sentiment['bearish_pct'] == 80.0
        assert sentiment['is_bearish_high'] is True
        assert sentiment['can_buy'] is False
    
    @patch('src.data_sentiment.requests.get')
    def test_get_sentiment_api_error(self, mock_get, client):
        """Test sentiment when API fails."""
        mock_get.side_effect = Exception("API Error")
        
        sentiment = client.get_sentiment("BTC")
        
        assert sentiment['error'] is True
        assert sentiment['can_buy'] is False  # block_on_error=True
    
    @patch('src.data_sentiment.requests.get')
    def test_is_buy_allowed_success(self, mock_get, client, mock_posts_response):
        """Test is_buy_allowed with bullish sentiment."""
        mock_get.return_value = Mock(
            status_code=200,
            json=lambda: mock_posts_response
        )
        mock_get.return_value.raise_for_status = Mock()
        
        allowed, reason = client.is_buy_allowed("BTC")
        
        assert allowed is True
        assert "Sentiment OK" in reason
    
    @patch('src.data_sentiment.requests.get')
    def test_is_buy_allowed_blocked(self, mock_get, client):
        """Test is_buy_allowed with bearish sentiment."""
        mock_response = {
            'results': [
                {'votes': {'positive': 5, 'negative': 95, 'neutral': 0}}
            ]
        }
        mock_get.return_value = Mock(
            status_code=200,
            json=lambda: mock_response
        )
        mock_get.return_value.raise_for_status = Mock()
        
        allowed, reason = client.is_buy_allowed("BTC")
        
        assert allowed is False
        assert "BLOCKED" in reason
    
    def test_cache_mechanism(self, client):
        """Test that caching works correctly."""
        # Simulate cached data
        client._cache = {
            'bullish_pct': 60.0,
            'bearish_pct': 30.0,
            'neutral_pct': 10.0,
            'total_votes': 100,
            'is_bearish_high': False,
            'can_buy': True,
            'error': False,
            'timestamp': datetime.now().isoformat()
        }
        import time
        client._cache_time = time.time()
        
        # Should return cached data without API call
        sentiment = client.get_sentiment("BTC")
        
        assert sentiment['bullish_pct'] == 60.0
        assert sentiment['bearish_pct'] == 30.0
    
    def test_clear_cache(self, client):
        """Test cache clearing."""
        import time
        client._cache = {'test': 'data'}
        client._cache_time = time.time()
        
        client.clear_cache()
        
        assert client._cache is None
        assert client._cache_time is None
    
    def test_disabled_sentiment(self):
        """Test when sentiment filtering is disabled."""
        config = {'enabled': False}
        client = CryptoPanicClient(api_key="test", config=config)
        
        allowed, reason = client.is_buy_allowed("BTC")
        
        assert allowed is True
        assert "disabled" in reason.lower()


class TestCheckSentimentFunction:
    """Test the convenience function."""
    
    @patch.object(CryptoPanicClient, 'is_buy_allowed')
    def test_check_sentiment(self, mock_is_buy_allowed):
        """Test the check_sentiment convenience function."""
        mock_is_buy_allowed.return_value = (True, "Test reason")
        
        allowed, reason = check_sentiment("BTC")
        
        assert allowed is True
        assert reason == "Test reason"
