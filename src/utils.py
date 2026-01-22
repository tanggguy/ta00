"""
Module: src/utils.py
Description: Utility functions for the trading bot
Author: Trading Bot
Date: 2025-01-22
Version: 1.0
"""

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)


def ensure_directory(path: str) -> Path:
    """
    Ensure a directory exists, create if necessary.
    
    Args:
        path (str): Directory path
    
    Returns:
        Path: Path object for the directory
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def format_pair(pair: str, separator: str = "/") -> str:
    """
    Format trading pair string.
    
    Args:
        pair (str): Trading pair (e.g., 'BTCUSDT' or 'BTC/USDT')
        separator (str): Desired separator
    
    Returns:
        str: Formatted pair string
    
    Examples:
        >>> format_pair('BTCUSDT', '/')
        'BTC/USDT'
        >>> format_pair('BTC/USDT', '')
        'BTCUSDT'
    """
    # Remove existing separators
    clean_pair = pair.replace("/", "").replace("-", "").replace("_", "")
    
    # Common quote currencies
    quote_currencies = ['USDT', 'USDC', 'BUSD', 'EUR', 'USD', 'BTC', 'ETH']
    
    for quote in quote_currencies:
        if clean_pair.endswith(quote):
            base = clean_pair[:-len(quote)]
            if separator:
                return f"{base}{separator}{quote}"
            return clean_pair
    
    return pair


def parse_date(date_str: str) -> datetime:
    """
    Parse date string to datetime.
    
    Args:
        date_str (str): Date string (YYYY-MM-DD or similar formats)
    
    Returns:
        datetime: Parsed datetime object
    
    Raises:
        ValueError: If date format is invalid
    """
    formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%Y-%m-%d %H:%M:%S"
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    raise ValueError(f"Invalid date format: {date_str}")


def timeframe_to_seconds(timeframe: str) -> int:
    """
    Convert timeframe string to seconds.
    
    Args:
        timeframe (str): Timeframe (e.g., '1m', '5m', '1h', '4h', '1d')
    
    Returns:
        int: Number of seconds
    
    Raises:
        ValueError: If timeframe format is invalid
    """
    multipliers = {
        'm': 60,
        'h': 3600,
        'd': 86400,
        'w': 604800
    }
    
    if len(timeframe) < 2:
        raise ValueError(f"Invalid timeframe: {timeframe}")
    
    unit = timeframe[-1].lower()
    try:
        value = int(timeframe[:-1])
    except ValueError:
        raise ValueError(f"Invalid timeframe: {timeframe}")
    
    if unit not in multipliers:
        raise ValueError(f"Unknown time unit: {unit}")
    
    return value * multipliers[unit]


def calculate_date_range(
    start_date: str,
    end_date: str,
    timeframe: str
) -> Tuple[datetime, datetime, int]:
    """
    Calculate expected number of candles for a date range.
    
    Args:
        start_date (str): Start date (YYYY-MM-DD)
        end_date (str): End date (YYYY-MM-DD)
        timeframe (str): Candle timeframe
    
    Returns:
        Tuple[datetime, datetime, int]: (start_dt, end_dt, expected_candles)
    """
    start_dt = parse_date(start_date)
    end_dt = parse_date(end_date)
    
    total_seconds = (end_dt - start_dt).total_seconds()
    candle_seconds = timeframe_to_seconds(timeframe)
    expected_candles = int(total_seconds / candle_seconds)
    
    return start_dt, end_dt, expected_candles


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: List[str]
) -> bool:
    """
    Validate DataFrame has required columns and no null values.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        required_columns (List[str]): List of required column names
    
    Returns:
        bool: True if valid, False otherwise
    """
    if df is None or df.empty:
        logger.error("DataFrame is empty or None")
        return False
    
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        logger.error(f"Missing columns: {missing_cols}")
        return False
    
    null_counts = df[required_columns].isnull().sum()
    if null_counts.any():
        logger.warning(f"Null values found: {null_counts[null_counts > 0].to_dict()}")
    
    return True


def calculate_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate percentage returns from prices.
    
    Args:
        prices (pd.Series): Series of prices
    
    Returns:
        pd.Series: Percentage returns
    """
    return prices.pct_change() * 100


def generate_filename(
    prefix: str,
    pair: str,
    start_date: str,
    end_date: str,
    extension: str = "csv"
) -> str:
    """
    Generate a standardized filename.
    
    Args:
        prefix (str): File prefix (e.g., 'backtest', 'ohlcv')
        pair (str): Trading pair
        start_date (str): Start date
        end_date (str): End date
        extension (str): File extension
    
    Returns:
        str: Formatted filename
    
    Examples:
        >>> generate_filename('backtest', 'BTCUSDT', '2024-01-01', '2024-12-31')
        'backtest_BTCUSDT_20240101_20241231.csv'
    """
    pair_clean = format_pair(pair, separator="")
    start_clean = start_date.replace("-", "").replace("/", "")
    end_clean = end_date.replace("-", "").replace("/", "")
    
    return f"{prefix}_{pair_clean}_{start_clean}_{end_clean}.{extension}"
