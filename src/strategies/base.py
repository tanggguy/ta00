"""
Module: src/strategies/base.py
Description: Abstract base class for all trading strategies
Author: Trading Bot
Date: 2025-01-22
Version: 1.0
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)


class Strategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    All strategies must implement the generate_signals method
    which adds a 'signal' column to the DataFrame:
    - 1 = BUY signal
    - -1 = SELL signal
    - 0 = HOLD (no action)
    
    Attributes:
        name (str): Strategy name
        params (Dict[str, Any]): Strategy parameters
    """
    
    REQUIRED_COLUMNS = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        """
        Initialize Strategy.
        
        Args:
            name (str): Strategy name
            params (Optional[Dict[str, Any]]): Strategy parameters
        """
        self.name = name
        self.params = params or {}
        logger.info(f"Initialized strategy: {name} with params: {self.params}")
    
    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """
        Validate input DataFrame has required columns.
        
        Args:
            df (pd.DataFrame): Input DataFrame
        
        Raises:
            ValueError: If required columns are missing
        """
        if df is None or df.empty:
            raise ValueError("DataFrame is empty or None")
        
        missing_cols = set(['close']) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals.
        
        Args:
            df (pd.DataFrame): OHLCV DataFrame with at least 'close' column
        
        Returns:
            pd.DataFrame: DataFrame with added 'signal' column
                - 1 = BUY
                - -1 = SELL
                - 0 = HOLD
        
        Raises:
            ValueError: If input DataFrame is invalid
        """
        pass
    
    def get_param(self, key: str, default: Any = None) -> Any:
        """
        Get strategy parameter.
        
        Args:
            key (str): Parameter name
            default (Any): Default value if not found
        
        Returns:
            Any: Parameter value
        """
        return self.params.get(key, default)
    
    def set_param(self, key: str, value: Any) -> None:
        """
        Set strategy parameter.
        
        Args:
            key (str): Parameter name
            value (Any): Parameter value
        """
        self.params[key] = value
        logger.debug(f"Set {self.name} param {key} = {value}")
    
    def get_indicators(self) -> List[str]:
        """
        Get list of indicators added by this strategy.
        
        Returns:
            List[str]: List of column names added to DataFrame
        """
        return []
    
    def describe(self) -> Dict[str, Any]:
        """
        Get strategy description.
        
        Returns:
            Dict[str, Any]: Strategy metadata
        """
        return {
            'name': self.name,
            'params': self.params,
            'indicators': self.get_indicators()
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', params={self.params})"
