"""
Script: scripts/run_backtest.py
Description: Run backtest with configured strategies
Author: Trading Bot
Date: 2025-01-22
Version: 1.1
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import get_config
from src.data_fetcher import BinanceDataFetcher
from src.backtest_engine import BacktestEngine
from src.results_exporter import ResultsExporter
from src.strategies import (
    get_strategy,
    list_strategies,
    TrendFollowing,
    MeanReversion,
    Momentum
)
from src.logger import setup_logging, get_logger

# Initialize logging
setup_logging()
logger = get_logger(__name__)


def load_data(pair: str, data_dir: str = "data/raw/"):
    """
    Load data from CSV or fetch from Binance.
    """
    data_path = Path(data_dir)
    pair_clean = pair.replace("/", "")
    
    # Look for existing CSV files
    csv_files = list(data_path.glob(f"*{pair_clean}*.csv"))
    
    if csv_files:
        # Use most recent file
        csv_file = sorted(csv_files)[-1]
        logger.info(f"Loading data from {csv_file}")
        # Use mainnet fetcher just for loading CSV (safe default)
        fetcher = BinanceDataFetcher(force_mainnet=True)
        return fetcher.load_from_csv(str(csv_file))
    else:
        logger.warning(f"No CSV found for {pair}, fetching from Binance...")
        config = get_config()
        backtest_config = config.backtest
        
        # CRITICAL FIX: Force mainnet for historical data
        fetcher = BinanceDataFetcher(force_mainnet=True)
        return fetcher.fetch_ohlcv(
            pair,
            timeframe=backtest_config.get('timeframe', '4h'),
            start_date=backtest_config.get('start_date', '2023-01-01'),
            end_date=backtest_config.get('end_date', '2025-01-22')
        )


def run_single_strategy(
    strategy_name: str,
    pair: str,
    engine: BacktestEngine,
    exporter: ResultsExporter
):
    """Run backtest for a single strategy."""
    logger.info(f"Running {strategy_name} on {pair}")
    
    # Load data
    df = load_data(pair)
    
    if df is None or df.empty:
        logger.error(f"No data for {pair}")
        return None
    
    # Get strategy
    strategy = get_strategy(strategy_name)
    
    # Run backtest
    results = engine.run(df, strategy, pair)
    
    # Export results
    exporter.export_all(results)
    
    return results


def run_all_strategies(pair: str, engine: BacktestEngine, exporter: ResultsExporter):
    """Run backtest for all enabled strategies."""
    config = get_config()
    strategies_config = config.strategies
    
    all_results = []
    
    for strategy_name in list_strategies():
        strategy_config = strategies_config.get(strategy_name, {})
        
        if not strategy_config.get('enabled', True):
            logger.info(f"Skipping disabled strategy: {strategy_name}")
            continue
        
        result = run_single_strategy(strategy_name, pair, engine, exporter)
        if result:
            all_results.append(result)
    
    return all_results


def main():
    """Main entry point for backtest."""
    parser = argparse.ArgumentParser(
        description="Run backtest with configured strategies"
    )
    parser.add_argument(
        '--strategy',
        type=str,
        default='all',
        help="Strategy to run (trend_following, mean_reversion, momentum, all)"
    )
    parser.add_argument(
        '--pair',
        type=str,
        help="Trading pair (e.g., BTC/USDT)"
    )
    parser.add_argument(
        '--all-pairs',
        action='store_true',
        help="Run on all configured pairs"
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/backtest_results/',
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = get_config()
    pairs_config = config.pairs
    
    # Determine pairs to backtest
    if args.all_pairs:
        pairs = pairs_config.get('pairs', ['BTC/USDT', 'ETH/USDT'])
    elif args.pair:
        pairs = [args.pair]
    else:
        pairs = pairs_config.get('pairs', ['BTC/USDT', 'ETH/USDT'])
    
    logger.info("=" * 60)
    logger.info("BACKTEST ENGINE")
    logger.info("=" * 60)
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Pairs: {pairs}")
    logger.info(f"Output: {args.output}")
    logger.info("=" * 60)
    
    # Initialize engine and exporter
    engine = BacktestEngine()
    exporter = ResultsExporter(output_dir=args.output)
    
    all_results = []
    
    for pair in pairs:
        logger.info(f"\n{'='*40}")
        logger.info(f"PROCESSING: {pair}")
        logger.info(f"{'='*40}")
        
        try:
            if args.strategy == 'all':
                results = run_all_strategies(pair, engine, exporter)
                all_results.extend(results)
            else:
                result = run_single_strategy(
                    args.strategy, pair, engine, exporter
                )
                if result:
                    all_results.append(result)
                    
        except Exception as e:
            logger.error(f"Backtest failed for {pair}: {e}")
            continue
    
    # Generate comparison if multiple results
    if len(all_results) > 1:
        comparison = engine.compare_results(all_results)
        exporter.export_comparison(comparison)
        
        logger.info("\n" + "=" * 60)
        logger.info("RESULTS COMPARISON")
        logger.info("=" * 60)
        print(comparison.to_string(index=False))
        logger.info("=" * 60)
    
    logger.info("\n✅ Backtest complete!")


if __name__ == "__main__":
    main()