"""
Module: src/strategies/mean_reversion.py
Description: Mean Reversion strategy using RSI
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


class MeanReversion(Strategy):
    """
    Mean Reversion Strategy using Relative Strength Index (RSI).
    
    Generates BUY signals when RSI indicates oversold conditions.
    Generates SELL signals when RSI indicates overbought conditions.
    
    Parameters (from config/strategies.json or constructor):
        rsi_period (int): RSI calculation period (default: 14)
        oversold (int): Oversold threshold (default: 30)
        overbought (int): Overbought threshold (default: 70)
    
    Example:
        >>> strategy = MeanReversion()
        >>> df_with_signals = strategy.generate_signals(ohlcv_df)
    """
    
    def __init__(
        self,
        rsi_period: Optional[int] = None,
        oversold: Optional[int] = None,
        overbought: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize Mean Reversion Strategy.
        
        Args:
            rsi_period (Optional[int]): RSI calculation period
            oversold (Optional[int]): Oversold threshold (BUY when RSI < this)
            overbought (Optional[int]): Overbought threshold (SELL when RSI > this)
            **kwargs: Additional parameters
        
        Raises:
            ValueError: If oversold >= overbought
        """
        # Load defaults from config
        config = get_config()
        strategy_config = config.get('strategies', 'mean_reversion', {})
        
        self.rsi_period = rsi_period or strategy_config.get('rsi_period', 14)
        self.oversold = oversold or strategy_config.get('oversold', 30)
        self.overbought = overbought or strategy_config.get('overbought', 70)
        
        # Validate parameters
        if self.oversold >= self.overbought:
            raise ValueError(
                f"oversold ({self.oversold}) must be less than "
                f"overbought ({self.overbought})"
            )
        
        if not (0 <= self.oversold <= 100) or not (0 <= self.overbought <= 100):
            raise ValueError("RSI thresholds must be between 0 and 100")
        
        params = {
            'rsi_period': self.rsi_period,
            'oversold': self.oversold,
            'overbought': self.overbought,
            **kwargs
        }
        
        super().__init__(name='mean_reversion', params=params)
        
        logger.info(
            f"MeanReversion initialized: RSI({self.rsi_period}) "
            f"oversold={self.oversold}, overbought={self.overbought}"
        )
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate mean reversion signals using RSI.
        
        Args:
            df (pd.DataFrame): OHLCV DataFrame with 'close' column
        
        Returns:
            pd.DataFrame: DataFrame with added columns:
                - rsi: RSI values
                - signal: Trading signal (1=BUY, -1=SELL, 0=HOLD)
        
        Raises:
            ValueError: If close column is missing
        """
        self._validate_dataframe(df)
        
        df = df.copy()
        
        # Calculate RSI
        df['rsi'] = ta.momentum.rsi(df['close'], self.rsi_period)
        
        # Generate signals
        # BUY (1) when oversold (RSI < 30)
        # SELL (-1) when overbought (RSI > 70)
        # HOLD (0) otherwise
        df['signal'] = np.where(
            df['rsi'] < self.oversold,
            1,   # BUY (oversold)
            np.where(
                df['rsi'] > self.overbought,
                -1,  # SELL (overbought)
                0    # HOLD
            )
        )
        
        # Log signal statistics
        buy_count = (df['signal'] == 1).sum()
        sell_count = (df['signal'] == -1).sum()
        hold_count = (df['signal'] == 0).sum()
        logger.debug(
            f"MeanReversion signals: {buy_count} BUY, {sell_count} SELL, "
            f"{hold_count} HOLD (total {len(df)} rows)"
        )
        
        return df
    
    def get_indicators(self) -> List[str]:
        """Get list of indicators added by this strategy."""
        return ['rsi']
