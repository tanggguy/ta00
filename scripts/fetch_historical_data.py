"""
Script: scripts/fetch_historical_data.py
Description: Fetch historical OHLCV data from Binance
Author: Trading Bot
Date: 2025-01-22
Version: 1.1

Usage:
    python scripts/fetch_historical_data.py
    python scripts/fetch_historical_data.py --pairs BTC/USDT ETH/USDT
    python scripts/fetch_historical_data.py --start 2023-01-01 --end 2024-12-31
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import get_config
from src.data_fetcher import BinanceDataFetcher
from src.logger import setup_logging, get_logger

# Initialize logging
setup_logging()
logger = get_logger(__name__)


def main():
    """Main entry point for data fetching."""
    parser = argparse.ArgumentParser(
        description="Fetch historical OHLCV data from Binance"
    )
    parser.add_argument(
        '--pairs',
        nargs='+',
        help="Trading pairs to fetch (e.g., BTC/USDT ETH/USDT)"
    )
    parser.add_argument(
        '--start',
        type=str,
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        '--end',
        type=str,
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        '--timeframe',
        type=str,
        default='4h',
        help="Timeframe (1m, 5m, 1h, 4h, 1d)"
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/raw/',
        help="Output directory for CSV files"
    )
    parser.add_argument(
        '--testnet',
        action='store_true',
        help="Use Binance testnet (WARNING: No historical data available on testnet)"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = get_config()
    backtest_config = config.backtest
    pairs_config = config.pairs
    
    # Use args or config defaults
    pairs = args.pairs or pairs_config.get('pairs', ['BTC/USDT', 'ETH/USDT'])
    start_date = args.start or backtest_config.get('start_date', '2023-01-01')
    end_date = args.end or backtest_config.get('end_date', '2025-01-22')
    timeframe = args.timeframe or backtest_config.get('timeframe', '4h')
    
    logger.info("=" * 60)
    logger.info("FETCHING HISTORICAL DATA")
    logger.info("=" * 60)
    logger.info(f"Pairs: {pairs}")
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info(f"Timeframe: {timeframe}")
    logger.info(f"Output: {args.output}")
    logger.info("=" * 60)
    
    try:
        # CORRECTION IMPORTANTE :
        # On force le Mainnet SAUF si l'utilisateur demande explicitement le testnet via l'argument CLI.
        # Cela permet de télécharger les vraies données historiques même si settings.json est en mode testnet.
        fetcher = BinanceDataFetcher(
            testnet=args.testnet,
            force_mainnet=not args.testnet
        )
        
        # Fetch data for all pairs
        results = fetcher.fetch_multiple_pairs(
            pairs=pairs,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            save=True,
            output_dir=args.output
        )
        
        # Summary
        logger.info("=" * 60)
        logger.info("FETCH COMPLETE")
        logger.info("=" * 60)
        
        for pair, df in results.items():
            logger.info(f"  {pair}: {len(df)} candles")
        
        logger.info(f"Total pairs: {len(results)}/{len(pairs)}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Fetch failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()