"""
Module: src/strategies/trend_following.py
Description: Trend Following strategy using SMA crossover
Author: Trading Bot
Date: 2025-01-22
Version: 1.0
"""

from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import ta

from src.strategies.base import Strategy
from src.config_loader import get_config
from src.logger import get_logger

logger = get_logger(__name__)


class TrendFollowing(Strategy):
    """
    Trend Following Strategy using Simple Moving Average (SMA) crossover.
    
    Generates BUY signals when short SMA crosses above long SMA.
    Generates SELL signals when short SMA crosses below long SMA.
    
    Parameters (from config/strategies.json or constructor):
        sma_short (int): Short SMA period (default: 20)
        sma_long (int): Long SMA period (default: 50)
    
    Example:
        >>> strategy = TrendFollowing()
        >>> df_with_signals = strategy.generate_signals(ohlcv_df)
    """
    
    def __init__(
        self,
        sma_short: Optional[int] = None,
        sma_long: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize Trend Following Strategy.
        
        Args:
            sma_short (Optional[int]): Short SMA period
            sma_long (Optional[int]): Long SMA period
            **kwargs: Additional parameters
        
        Raises:
            ValueError: If sma_short >= sma_long
        """
        # Load defaults from config
        config = get_config()
        strategy_config = config.get('strategies', 'trend_following', {})
        
        self.sma_short = sma_short or strategy_config.get('sma_short', 20)
        self.sma_long = sma_long or strategy_config.get('sma_long', 50)
        
        # Validate parameters
        if self.sma_short >= self.sma_long:
            raise ValueError(
                f"sma_short ({self.sma_short}) must be less than "
                f"sma_long ({self.sma_long})"
            )
        
        params = {
            'sma_short': self.sma_short,
            'sma_long': self.sma_long,
            **kwargs
        }
        
        super().__init__(name='trend_following', params=params)
        
        logger.info(
            f"TrendFollowing initialized: SMA({self.sma_short}/{self.sma_long})"
        )
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trend following signals using SMA crossover.
        
        Args:
            df (pd.DataFrame): OHLCV DataFrame with 'close' column
        
        Returns:
            pd.DataFrame: DataFrame with added columns:
                - sma_short: Short SMA values
                - sma_long: Long SMA values
                - signal: Trading signal (1=BUY, -1=SELL)
        
        Raises:
            ValueError: If close column is missing
        """
        self._validate_dataframe(df)
        
        df = df.copy()
        
        # Calculate SMAs
        df['sma_short'] = ta.trend.sma_indicator(df['close'], self.sma_short)
        df['sma_long'] = ta.trend.sma_indicator(df['close'], self.sma_long)
        
        # Generate signals
        # BUY (1) when short SMA > long SMA (bullish trend)
        # SELL (-1) when short SMA < long SMA (bearish trend)
        df['signal'] = np.where(
            df['sma_short'] > df['sma_long'],
            1,   # BUY
            -1   # SELL
        )
        
        # Log signal statistics
        buy_count = (df['signal'] == 1).sum()
        sell_count = (df['signal'] == -1).sum()
        logger.debug(
            f"TrendFollowing signals: {buy_count} BUY, {sell_count} SELL "
            f"(total {len(df)} rows)"
        )
        
        return df
    
    def get_indicators(self) -> List[str]:
        """Get list of indicators added by this strategy."""
        return ['sma_short', 'sma_long']
