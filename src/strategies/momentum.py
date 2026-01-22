"""
Module: src/strategies/momentum.py
Description: Momentum strategy using MACD
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


class Momentum(Strategy):
    """
    Momentum Strategy using Moving Average Convergence Divergence (MACD).
    
    Generates BUY signals when MACD line crosses above signal line.
    Generates SELL signals when MACD line crosses below signal line.
    
    Parameters (from config/strategies.json or constructor):
        macd_fast (int): Fast EMA period (default: 12)
        macd_slow (int): Slow EMA period (default: 26)
        macd_signal (int): Signal line period (default: 9)
    
    Example:
        >>> strategy = Momentum()
        >>> df_with_signals = strategy.generate_signals(ohlcv_df)
    """
    
    def __init__(
        self,
        macd_fast: Optional[int] = None,
        macd_slow: Optional[int] = None,
        macd_signal: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize Momentum Strategy.
        
        Args:
            macd_fast (Optional[int]): Fast EMA period
            macd_slow (Optional[int]): Slow EMA period
            macd_signal (Optional[int]): Signal line period
            **kwargs: Additional parameters
        
        Raises:
            ValueError: If macd_fast >= macd_slow
        """
        # Load defaults from config
        config = get_config()
        strategy_config = config.get('strategies', 'momentum', {})
        
        self.macd_fast = macd_fast or strategy_config.get('macd_fast', 12)
        self.macd_slow = macd_slow or strategy_config.get('macd_slow', 26)
        self.macd_signal = macd_signal or strategy_config.get('macd_signal', 9)
        
        # Validate parameters
        if self.macd_fast >= self.macd_slow:
            raise ValueError(
                f"macd_fast ({self.macd_fast}) must be less than "
                f"macd_slow ({self.macd_slow})"
            )
        
        params = {
            'macd_fast': self.macd_fast,
            'macd_slow': self.macd_slow,
            'macd_signal': self.macd_signal,
            **kwargs
        }
        
        super().__init__(name='momentum', params=params)
        
        logger.info(
            f"Momentum initialized: MACD({self.macd_fast}/{self.macd_slow}/{self.macd_signal})"
        )
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate momentum signals using MACD.
        
        Args:
            df (pd.DataFrame): OHLCV DataFrame with 'close' column
        
        Returns:
            pd.DataFrame: DataFrame with added columns:
                - macd: MACD line values
                - macd_signal_line: Signal line values
                - macd_histogram: MACD histogram
                - signal: Trading signal (1=BUY, -1=SELL)
        
        Raises:
            ValueError: If close column is missing
        """
        self._validate_dataframe(df)
        
        df = df.copy()
        
        # Calculate MACD components
        macd_indicator = ta.trend.MACD(
            df['close'],
            window_slow=self.macd_slow,
            window_fast=self.macd_fast,
            window_sign=self.macd_signal
        )
        
        df['macd'] = macd_indicator.macd()
        df['macd_signal_line'] = macd_indicator.macd_signal()
        df['macd_histogram'] = macd_indicator.macd_diff()
        
        # Generate signals
        # BUY (1) when MACD > Signal Line (bullish momentum)
        # SELL (-1) when MACD < Signal Line (bearish momentum)
        df['signal'] = np.where(
            df['macd'] > df['macd_signal_line'],
            1,   # BUY
            -1   # SELL
        )
        
        # Log signal statistics
        buy_count = (df['signal'] == 1).sum()
        sell_count = (df['signal'] == -1).sum()
        logger.debug(
            f"Momentum signals: {buy_count} BUY, {sell_count} SELL "
            f"(total {len(df)} rows)"
        )
        
        return df
    
    def get_indicators(self) -> List[str]:
        """Get list of indicators added by this strategy."""
        return ['macd', 'macd_signal_line', 'macd_histogram']
