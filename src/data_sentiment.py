"""
Module: src/data_sentiment.py
Description: Fetch and analyze market sentiment from CryptoPanic API
Author: Trading Bot
Date: 2026-01-28
Version: 1.0

Provides sentiment analysis based on CryptoPanic news votes.
Used to filter BUY signals when market sentiment is too bearish.
"""

import os
import time
from datetime import datetime
from typing import Dict, Optional, Tuple

import requests

from src.config_loader import get_config
from src.logger import get_logger

logger = get_logger(__name__)


class CryptoPanicClient:
    """
    Fetches sentiment data from CryptoPanic news API.
    
    CryptoPanic aggregates crypto news and provides vote data
    (bullish/bearish/neutral) for sentiment analysis.
    
    Attributes:
        api_key (str): CryptoPanic API authentication key
        config (dict): Sentiment configuration from config/sentiment.json
    """
    
    BASE_URL = "https://cryptopanic.com/api/v1"
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[dict] = None):
        """
        Initialize CryptoPanic client.
        
        Args:
            api_key: API key (defaults to env variable CRYPTOPANIC_API_KEY)
            config: Configuration dict (defaults to config/sentiment.json)
        """
        self.api_key = api_key or os.getenv("CRYPTOPANIC_API_KEY")
        
        if not self.api_key:
            logger.warning("CRYPTOPANIC_API_KEY not set - sentiment filtering disabled")
        
        # Load config
        if config is None:
            try:
                full_config = get_config()
                # Use the sentiment property which returns the full sentiment dict
                sentiment_config = full_config.sentiment
                self.config = sentiment_config.get('cryptopanic', {}) if sentiment_config else {}
            except Exception as e:
                logger.warning(f"Failed to load sentiment config: {e}")
                self.config = {}
        else:
            self.config = config
        
        # Cache for API responses
        self._cache: Optional[Dict] = None
        self._cache_time: Optional[float] = None
        self._cache_duration = self.config.get('cache_duration_seconds', 14400)  # 4h default
        
        logger.info("CryptoPanicClient initialized")
    
    def _is_cache_valid(self) -> bool:
        """Check if cached data is still valid."""
        if self._cache is None or self._cache_time is None:
            return False
        return (time.time() - self._cache_time) < self._cache_duration
    
    def _fetch_posts(self, currency: str = "BTC") -> Optional[Dict]:
        """
        Fetch latest posts from CryptoPanic API.
        
        Args:
            currency: Currency filter (e.g., 'BTC', 'ETH')
        
        Returns:
            API response dict or None on error
        """
        if not self.api_key:
            logger.error("Cannot fetch posts - API key not configured")
            return None
        
        try:
            url = f"{self.BASE_URL}/posts/"
            params = {
                "auth_token": self.api_key,
                "currencies": currency,
                "filter": "rising",
                "public": "true"
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            logger.debug(f"Fetched {len(data.get('results', []))} posts for {currency}")
            return data
            
        except requests.Timeout:
            logger.error("CryptoPanic API timeout")
            return None
        except requests.RequestException as e:
            logger.error(f"CryptoPanic API error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching CryptoPanic data: {e}")
            return None
    
    def get_sentiment(self, currency: str = "BTC") -> Dict:
        """
        Calculate sentiment from recent news votes.
        
        Args:
            currency: Currency to analyze (e.g., 'BTC', 'ETH')
        
        Returns:
            dict: {
                'bullish_pct': float (0-100),
                'bearish_pct': float (0-100),
                'neutral_pct': float (0-100),
                'total_votes': int,
                'is_bearish_high': bool,
                'can_buy': bool,
                'error': bool,
                'timestamp': str
            }
        """
        # Check cache
        if self._is_cache_valid() and self._cache:
            logger.debug("Using cached sentiment data")
            return self._cache
        
        # Default response for errors
        error_response = {
            'bullish_pct': 0.0,
            'bearish_pct': 0.0,
            'neutral_pct': 0.0,
            'total_votes': 0,
            'is_bearish_high': False,
            'can_buy': not self.config.get('block_on_error', True),
            'error': True,
            'timestamp': datetime.now().isoformat()
        }
        
        # Fetch data
        data = self._fetch_posts(currency)
        if data is None:
            logger.warning("Failed to fetch sentiment - returning error response")
            return error_response
        
        # Count votes from posts
        total_bullish = 0
        total_bearish = 0
        total_neutral = 0
        
        posts = data.get('results', [])
        for post in posts:
            votes = post.get('votes', {})
            total_bullish += votes.get('positive', 0)
            total_bearish += votes.get('negative', 0)
            total_neutral += votes.get('neutral', 0)  # Some may not have this
        
        total_votes = total_bullish + total_bearish + total_neutral
        
        if total_votes == 0:
            logger.warning("No votes found in recent posts")
            return {
                **error_response,
                'error': False,
                'can_buy': True  # No data = allow trade
            }
        
        # Calculate percentages
        bullish_pct = (total_bullish / total_votes) * 100
        bearish_pct = (total_bearish / total_votes) * 100
        neutral_pct = (total_neutral / total_votes) * 100
        
        # Check threshold
        threshold = self.config.get('bearish_threshold_pct', 70)
        is_bearish_high = bearish_pct > threshold
        
        result = {
            'bullish_pct': round(bullish_pct, 2),
            'bearish_pct': round(bearish_pct, 2),
            'neutral_pct': round(neutral_pct, 2),
            'total_votes': total_votes,
            'is_bearish_high': is_bearish_high,
            'can_buy': not is_bearish_high,
            'error': False,
            'timestamp': datetime.now().isoformat()
        }
        
        # Update cache
        self._cache = result
        self._cache_time = time.time()
        
        logger.info(
            f"Sentiment for {currency}: "
            f"Bullish={bullish_pct:.1f}%, Bearish={bearish_pct:.1f}% "
            f"(threshold={threshold}%)"
        )
        
        return result
    
    def is_buy_allowed(self, currency: str = "BTC") -> Tuple[bool, str]:
        """
        Check if BUY signal is allowed based on sentiment.
        
        This is the main method used by the trading bot to filter signals.
        
        Args:
            currency: Currency to check
        
        Returns:
            tuple: (allowed: bool, reason: str)
        
        Examples:
            >>> client.is_buy_allowed("BTC")
            (True, "Sentiment OK: 45.2% bearish (threshold: 70%)")
            
            >>> client.is_buy_allowed("BTC")
            (False, "BLOCKED: 75.3% bearish > 70% threshold")
        """
        # Check if sentiment filtering is enabled
        if not self.config.get('enabled', True):
            return True, "Sentiment filtering disabled"
        
        sentiment = self.get_sentiment(currency)
        
        if sentiment['error']:
            if self.config.get('block_on_error', True):
                return False, "BLOCKED: Sentiment API unavailable (block_on_error=True)"
            else:
                return True, "Sentiment API unavailable - proceeding without filter"
        
        threshold = self.config.get('bearish_threshold_pct', 70)
        
        if sentiment['is_bearish_high']:
            return False, f"BLOCKED: {sentiment['bearish_pct']:.1f}% bearish > {threshold}% threshold"
        
        return True, f"Sentiment OK: {sentiment['bearish_pct']:.1f}% bearish (threshold: {threshold}%)"
    
    def clear_cache(self) -> None:
        """Clear the sentiment cache."""
        self._cache = None
        self._cache_time = None
        logger.debug("Sentiment cache cleared")


# Convenience function for quick access
def check_sentiment(currency: str = "BTC") -> Tuple[bool, str]:
    """
    Quick check if BUY is allowed for a currency.
    
    Args:
        currency: Currency to check (e.g., 'BTC', 'ETH')
    
    Returns:
        tuple: (allowed: bool, reason: str)
    """
    client = CryptoPanicClient()
    return client.is_buy_allowed(currency)
