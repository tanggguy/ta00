"""
Module: src/data_trends.py
Description: Analyze Google Trends data for momentum confirmation
Author: Trading Bot
Date: 2026-01-28
Version: 1.0

Uses pytrends library to fetch Google Trends interest data.
A surge in search interest confirms momentum trading signals.
"""

import time
from datetime import datetime
from typing import Dict, Optional, Tuple

from pytrends.request import TrendReq

from src.config_loader import get_config
from src.logger import get_logger

logger = get_logger(__name__)


class GoogleTrendsAnalyzer:
    """
    Analyzes Google Trends data for momentum confirmation.
    
    A sudden rise in interest (current > 150% of 7-day average)
    serves as confirmation for momentum trading strategies.
    
    Attributes:
        config (dict): Trends configuration from config/sentiment.json
        pytrends (TrendReq): Pytrends API client
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize Google Trends analyzer.
        
        Args:
            config: Configuration dict (defaults to config/sentiment.json)
        """
        # Load config
        if config is None:
            try:
                full_config = get_config()
                # Use the sentiment property which returns the full sentiment dict
                sentiment_config = full_config.sentiment
                self.config = sentiment_config.get('google_trends', {}) if sentiment_config else {}
            except Exception as e:
                logger.warning(f"Failed to load trends config: {e}")
                self.config = {}
        else:
            self.config = config
        
        # Initialize pytrends client
        try:
            self.pytrends = TrendReq(
                hl='en-US',
                tz=0,  # UTC
                timeout=(10, 25),  # Connect timeout, read timeout
                retries=2,
                backoff_factor=0.5
            )
            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize pytrends: {e}")
            self._initialized = False
        
        # Cache for API responses
        self._cache: Optional[Dict] = None
        self._cache_time: Optional[float] = None
        self._cache_duration = self.config.get('cache_duration_seconds', 14400)  # 4h default
        
        logger.info("GoogleTrendsAnalyzer initialized")
    
    def _is_cache_valid(self) -> bool:
        """Check if cached data is still valid."""
        if self._cache is None or self._cache_time is None:
            return False
        return (time.time() - self._cache_time) < self._cache_duration
    
    def get_interest_data(self, keyword: Optional[str] = None) -> Dict:
        """
        Fetch Google Trends interest data for a keyword.
        
        Args:
            keyword: Search term (defaults to config keyword, usually "Bitcoin")
        
        Returns:
            dict: {
                'current_interest': int (0-100 scale),
                'avg_7d': float,
                'max_7d': int,
                'min_7d': int,
                'surge_ratio': float (current / avg),
                'is_surging': bool (ratio > threshold),
                'error': bool,
                'timestamp': str
            }
        """
        # Check cache
        if self._is_cache_valid() and self._cache:
            logger.debug("Using cached trends data")
            return self._cache
        
        keyword = keyword or self.config.get('keyword', 'Bitcoin')
        timeframe = self.config.get('timeframe', 'now 7-d')
        geo = self.config.get('geo', '')  # Empty = worldwide
        
        # Default error response
        error_response = {
            'keyword': keyword,
            'current_interest': 0,
            'avg_7d': 0.0,
            'max_7d': 0,
            'min_7d': 0,
            'surge_ratio': 0.0,
            'is_surging': False,
            'error': True,
            'timestamp': datetime.now().isoformat()
        }
        
        if not self._initialized:
            logger.error("Google Trends client not initialized")
            return error_response
        
        try:
            # Build payload
            self.pytrends.build_payload(
                kw_list=[keyword],
                timeframe=timeframe,
                geo=geo
            )
            
            # Get interest over time
            interest_df = self.pytrends.interest_over_time()
            
            if interest_df.empty:
                logger.warning(f"No trends data returned for '{keyword}'")
                return error_response
            
            # Extract values
            values = interest_df[keyword].values
            current_interest = int(values[-1])  # Latest value
            avg_7d = float(values.mean())
            max_7d = int(values.max())
            min_7d = int(values.min())
            
            # Calculate surge ratio
            surge_ratio = current_interest / avg_7d if avg_7d > 0 else 0.0
            
            # Check threshold (default 150% = 1.5)
            threshold = self.config.get('surge_threshold_pct', 150) / 100.0
            is_surging = surge_ratio >= threshold
            
            result = {
                'keyword': keyword,
                'current_interest': current_interest,
                'avg_7d': round(avg_7d, 2),
                'max_7d': max_7d,
                'min_7d': min_7d,
                'surge_ratio': round(surge_ratio, 2),
                'surge_ratio_pct': round(surge_ratio * 100, 1),
                'is_surging': is_surging,
                'error': False,
                'timestamp': datetime.now().isoformat()
            }
            
            # Update cache
            self._cache = result
            self._cache_time = time.time()
            
            logger.info(
                f"Trends for '{keyword}': "
                f"Current={current_interest}, Avg={avg_7d:.1f}, "
                f"Surge={surge_ratio:.2f}x (threshold={threshold:.2f}x)"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to fetch Google Trends data: {e}")
            return error_response
    
    def is_momentum_confirmed(self, keyword: Optional[str] = None) -> Tuple[bool, str]:
        """
        Check if momentum is confirmed by a surge in search interest.
        
        This method is used to confirm momentum trading signals.
        A surge is defined as current interest > threshold% of 7-day average.
        
        Args:
            keyword: Search term (defaults to config keyword)
        
        Returns:
            tuple: (confirmed: bool, reason: str)
        
        Examples:
            >>> analyzer.is_momentum_confirmed()
            (True, "Momentum CONFIRMED: 175% surge (current=85, avg=48.6)")
            
            >>> analyzer.is_momentum_confirmed()
            (False, "No surge detected: 95% (current=45, avg=47.3)")
        """
        # Check if trends filtering is enabled
        if not self.config.get('enabled', True):
            return True, "Google Trends filtering disabled"
        
        if not self.config.get('use_as_momentum_confirmation', True):
            return True, "Momentum confirmation via trends disabled"
        
        data = self.get_interest_data(keyword)
        threshold_pct = self.config.get('surge_threshold_pct', 150)
        
        if data['error']:
            # On error, don't block the trade but log warning
            return True, "Trends API unavailable - proceeding without confirmation"
        
        if data['is_surging']:
            return True, (
                f"Momentum CONFIRMED: {data['surge_ratio_pct']:.0f}% surge "
                f"(current={data['current_interest']}, avg={data['avg_7d']:.1f}, "
                f"threshold={threshold_pct}%)"
            )
        
        return False, (
            f"No surge detected: {data['surge_ratio_pct']:.0f}% "
            f"(current={data['current_interest']}, avg={data['avg_7d']:.1f}, "
            f"need >{threshold_pct}%)"
        )
    
    def clear_cache(self) -> None:
        """Clear the trends cache."""
        self._cache = None
        self._cache_time = None
        logger.debug("Trends cache cleared")


# Convenience function for quick access
def check_momentum_confirmation(keyword: str = "Bitcoin") -> Tuple[bool, str]:
    """
    Quick check if momentum is confirmed by Google Trends.
    
    Args:
        keyword: Search term to analyze
    
    Returns:
        tuple: (confirmed: bool, reason: str)
    """
    analyzer = GoogleTrendsAnalyzer()
    return analyzer.is_momentum_confirmed(keyword)
