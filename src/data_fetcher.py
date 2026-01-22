"""
Module: src/data_fetcher.py
Description: Fetch OHLCV data from Binance via CCXT
Author: Trading Bot
Date: 2025-01-22
Version: 1.1
"""

import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import ccxt
import pandas as pd

from src.config_loader import get_config
from src.logger import get_logger
from src.utils import ensure_directory, format_pair, generate_filename

logger = get_logger(__name__)


class BinanceDataFetcher:
    """
    Fetches OHLCV (Open, High, Low, Close, Volume) data from Binance.
    
    Uses CCXT library with rate limiting to avoid API blocks.
    Configuration loaded from config/pairs.json and config/settings.json.
    """
    
    def __init__(self, testnet: bool = False, force_mainnet: bool = False):
        """
        Initialize Binance Data Fetcher.
        
        Args:
            testnet (bool): Use Binance testnet (default: False)
            force_mainnet (bool): Force Mainnet connection ignoring config (default: False)
        """
        config = get_config()
        
        # Logic: Use testnet if (requested OR config says so) AND NOT forced to mainnet
        should_use_testnet = (testnet or config.is_testnet) and not force_mainnet
        
        if should_use_testnet:
            self.exchange = ccxt.binance({
                'sandbox': True,
                'enableRateLimit': True,
                'rateLimit': 100  # ms between requests
            })
            logger.info("Initialized Binance Data Fetcher (TESTNET)")
        else:
            self.exchange = ccxt.binance({
                'enableRateLimit': True,
                'rateLimit': 100
            })
            logger.info("Initialized Binance Data Fetcher (MAINNET)")
        
        # Load markets
        try:
            self.exchange.load_markets()
            logger.debug(f"Loaded {len(self.exchange.markets)} markets")
        except Exception as e:
            logger.error(f"Failed to load markets: {e}")
            raise
    
    def fetch_ohlcv(
        self,
        pair: str,
        timeframe: str = '4h',
        start_date: str = '2023-01-01',
        end_date: Optional[str] = None,
        limit_per_request: int = 500
    ) -> pd.DataFrame:
        """
        Fetch OHLCV historical data for a trading pair.
        
        Args:
            pair (str): Trading pair (e.g., 'BTC/USDT')
            timeframe (str): Candle timeframe ('1m', '5m', '1h', '4h', '1d')
            start_date (str): Start date (YYYY-MM-DD)
            end_date (Optional[str]): End date (YYYY-MM-DD), defaults to today
            limit_per_request (int): Max candles per API request (default: 500)
        
        Returns:
            pd.DataFrame: DataFrame with columns [timestamp, open, high, low, close, volume]
        
        Raises:
            ValueError: If pair is not supported
            Exception: If API request fails
        """
        # Format pair for CCXT
        formatted_pair = format_pair(pair, separator="/")
        
        # Check if market exists (safeguard)
        if formatted_pair not in self.exchange.markets:
            # Try to reload markets once if pair not found
            self.exchange.load_markets()
            if formatted_pair not in self.exchange.markets:
                raise ValueError(f"Pair not supported: {formatted_pair}")
        
        # Parse dates
        since = self.exchange.parse8601(f"{start_date}T00:00:00Z")
        
        if end_date:
            until = self.exchange.parse8601(f"{end_date}T23:59:59Z")
        else:
            until = self.exchange.milliseconds()
        
        logger.info(
            f"Fetching {formatted_pair} {timeframe} from {start_date} to "
            f"{end_date or 'now'}"
        )
        
        all_candles: List[List] = []
        request_count = 0
        
        while since < until:
            try:
                candles = self.exchange.fetch_ohlcv(
                    formatted_pair,
                    timeframe,
                    since,
                    limit=limit_per_request
                )
                
                if not candles:
                    logger.debug("No more candles to fetch")
                    break
                
                all_candles.extend(candles)
                request_count += 1
                
                # Update since to last candle timestamp + 1ms
                since = candles[-1][0] + 1
                
                # Progress logging every 10 requests
                if request_count % 10 == 0:
                    current_date = datetime.fromtimestamp(since / 1000)
                    logger.debug(
                        f"Progress: {len(all_candles)} candles fetched, "
                        f"current date: {current_date.date()}"
                    )
                
                # Extra delay to respect rate limits
                time.sleep(0.1)
                
            except ccxt.RateLimitExceeded as e:
                logger.warning(f"Rate limit hit, waiting 60s: {e}")
                time.sleep(60)
            except ccxt.NetworkError as e:
                logger.error(f"Network error, retrying in 5s: {e}")
                time.sleep(5)
            except Exception as e:
                logger.error(f"Unexpected error fetching data: {e}")
                raise
        
        if not all_candles:
            logger.warning(f"No data fetched for {formatted_pair}")
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(
            all_candles,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Remove duplicates and sort
        df = df.drop_duplicates(subset=['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Filter to end date
        if end_date:
            end_datetime = pd.to_datetime(end_date)
            df = df[df['timestamp'] <= end_datetime]
        
        logger.info(
            f"✅ Fetched {len(df)} candles for {formatted_pair} "
            f"({request_count} API requests)"
        )
        
        return df
    
    def validate_data(self, df: pd.DataFrame, pair: str) -> bool:
        """
        Validate that fetched data is complete and correct.
        """
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        if df is None or df.empty:
            logger.error(f"Empty data for {pair}")
            return False
        
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            logger.error(f"Missing columns for {pair}: {missing_cols}")
            return False
        
        # Check for null values
        null_counts = df[required_cols].isnull().sum()
        if null_counts.any():
            logger.warning(
                f"Null values in {pair}: {null_counts[null_counts > 0].to_dict()}"
            )
        
        # Check for negative values
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if (df[col] < 0).any():
                logger.error(f"Negative values in {col} for {pair}")
                return False
        
        # Check OHLC consistency (high >= low)
        if (df['high'] < df['low']).any():
            logger.error(f"Invalid OHLC: high < low for {pair}")
            return False
        
        logger.info(f"✅ Data validated for {pair}: {len(df)} rows")
        return True
    
    def save_to_csv(
        self,
        df: pd.DataFrame,
        pair: str,
        output_dir: str = "data/raw/",
        start_date: str = "",
        end_date: str = ""
    ) -> str:
        """Save DataFrame to CSV file."""
        ensure_directory(output_dir)
        
        pair_clean = format_pair(pair, separator="")
        
        if start_date and end_date:
            filename = generate_filename("ohlcv", pair, start_date, end_date)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ohlcv_{pair_clean}_{timestamp}.csv"
        
        filepath = Path(output_dir) / filename
        df.to_csv(filepath, index=False)
        
        logger.info(f"✅ Saved data to {filepath}")
        return str(filepath)
    
    def load_from_csv(self, filepath: str) -> pd.DataFrame:
        """Load OHLCV data from CSV file."""
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        logger.info(f"✅ Loaded {len(df)} rows from {filepath}")
        return df
    
    def fetch_multiple_pairs(
        self,
        pairs: Optional[List[str]] = None,
        timeframe: str = '4h',
        start_date: str = '2023-01-01',
        end_date: Optional[str] = None,
        save: bool = True,
        output_dir: str = "data/raw/"
    ) -> dict:
        """Fetch data for multiple trading pairs."""
        if pairs is None:
            config = get_config()
            pairs = config.get('pairs', 'pairs', ['BTC/USDT', 'ETH/USDT'])
        
        logger.info(f"Fetching data for {len(pairs)} pairs: {pairs}")
        
        results = {}
        
        for pair in pairs:
            try:
                df = self.fetch_ohlcv(pair, timeframe, start_date, end_date)
                
                if self.validate_data(df, pair):
                    results[pair] = df
                    
                    if save:
                        self.save_to_csv(
                            df, pair, output_dir, start_date,
                            end_date or datetime.now().strftime("%Y-%m-%d")
                        )
                else:
                    logger.warning(f"Skipping {pair} due to validation errors")
                    
            except Exception as e:
                logger.error(f"Failed to fetch {pair}: {e}")
                continue
        
        logger.info(f"✅ Successfully fetched {len(results)}/{len(pairs)} pairs")
        return results