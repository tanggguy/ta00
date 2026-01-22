import argparse
import json
import logging
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from datetime import datetime
from src.data_fetcher import BinanceDataFetcher
from src.optimizer import walk_forward_validation
from src.strategies import TrendFollowing, MeanReversion, Momentum
from src.config_loader import get_config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Optimizer")

STRATEGIES = {
    "trend_following": TrendFollowing,
    "mean_reversion": MeanReversion,
    "momentum": Momentum
}

def main():
    parser = argparse.ArgumentParser(description="Run Strategy Optimization")
    parser.add_argument("--strategy", type=str, required=True, help="Strategy name (trend_following, mean_reversion, momentum)")
    parser.add_argument("--pair", type=str, required=True, help="Trading pair (e.g., BTCUSDT)")
    parser.add_argument("--start_date", type=str, default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default="2024-01-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--timeframe", type=str, default="4h", help="Timeframe (4h, 1d)")
    
    args = parser.parse_args()
    
    # 1. Load Configs
    try:
        config_loader = get_config()
        # optimization.json key is 'optimization' (filename without extension)
        opt_config = config_loader.get('optimization')
        
        if not opt_config:
             logger.error("Optimization config not found or empty.")
             return

        if args.strategy not in opt_config:
            logger.error(f"No optimization config found for {args.strategy}")
            return
        param_grid = opt_config[args.strategy]
    except Exception as e:
        logger.error(f"Failed to load optimization config: {e}")
        return

    # 2. Fetch Data
    logger.info(f"Fetching data for {args.pair}...")
    fetcher = BinanceDataFetcher(force_mainnet=True)
    try:
        df = fetcher.fetch_ohlcv(args.pair, args.timeframe, args.start_date, args.end_date)
        if df.empty:
            logger.error("No data fetched.")
            return
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return

    # 3. Get Strategy Class
    strategy_class = STRATEGIES.get(args.strategy)
    if not strategy_class:
        logger.error(f"Strategy class {args.strategy} not found.")
        return

    # 4. Run Optimization
    logger.info(f"Starting optimization for {args.strategy} on {args.pair}...")
    results = walk_forward_validation(
        data=df,
        strategy_class=strategy_class,
        param_grid=param_grid,
        pair=args.pair,
        # Default split 70/30
        train_size=0.7 
    )

    if 'error' in results:
        logger.error("Optimization failed.")
        return

    # 5. Save Results
    output_dir = "data/optimization_results"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/{args.pair}_{args.strategy}_opt_{timestamp}.json"
    
    # Convert numpy types to native python types for JSON serialization
    def convert(o):
        if isinstance(o, (pd.Int64Dtype, pd.Float64Dtype, np.integer, np.int64, np.int32)): 
            return int(o)
        if isinstance(o, (np.floating, np.float64, np.float32)):
            return float(o)
        if isinstance(o, (np.bool_)):
            return bool(o)
        return str(o)

    with open(filename, 'w') as f:
        json.dump(results, f, indent=4, default=convert)
        
    logger.info(f"✅ Optimization results saved to {filename}")
    logger.info(f"Best Params: {results['best_params']}")
    logger.info(f"Overfitting Detected: {results['is_overfitting']} (Diff: {results['overfitting_detected']:.2f})")

if __name__ == "__main__":
    main()
