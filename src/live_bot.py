"""
Module: src/live_bot.py
Description: Live/Paper trading bot with sentiment and trends filtering
Author: Trading Bot
Date: 2026-01-28
Version: 1.0

Main trading bot for paper and live trading.
Integrates market context filters (sentiment + trends) to validate signals.
"""

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ccxt
import pandas as pd

from src.config_loader import get_config
from src.data_fetcher import BinanceDataFetcher
from src.data_sentiment import CryptoPanicClient
from src.data_trends import GoogleTrendsAnalyzer
from src.logger import get_logger
from src.risk_manager import RiskManager
from src.strategies import get_strategy, STRATEGY_REGISTRY

logger = get_logger(__name__)


class TradeLogger:
    """Simple trade logger for CSV logging."""
    
    def __init__(self, log_dir: str = 'data/live_trading'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def log_trade(
        self,
        pair: str,
        signal: str,
        entry_price: float,
        exit_price: Optional[float] = None,
        pnl: Optional[float] = None,
        notes: str = ""
    ) -> None:
        """Log a trade to CSV file."""
        timestamp = datetime.now()
        filename = self.log_dir / f"{pair.replace('/', '_')}_trades_{timestamp.date()}.csv"
        
        trade_data = {
            'timestamp': timestamp.isoformat(),
            'pair': pair,
            'signal': signal,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_usdt': pnl,
            'pnl_percent': (pnl / entry_price * 100) if pnl and entry_price else None,
            'status': 'OPEN' if exit_price is None else 'CLOSED',
            'notes': notes
        }
        
        df_new = pd.DataFrame([trade_data])
        
        if filename.exists():
            df_existing = pd.read_csv(filename)
            df = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df = df_new
        
        df.to_csv(filename, index=False)
        logger.debug(f"Trade logged to {filename}")


class MarketContextFilter:
    """
    Combines sentiment and trends filters for trade validation.
    
    This class acts as a gatekeeper for trading signals, ensuring
    that market context supports the proposed trade direction.
    
    Rules:
    1. If sentiment > 70% bearish → BLOCK BUY signals
    2. If momentum strategy AND trends surging → CONFIRM signal
    """
    
    def __init__(self):
        """Initialize market context filter with sentiment and trends clients."""
        self.sentiment_client = CryptoPanicClient()
        self.trends_analyzer = GoogleTrendsAnalyzer()
        
        # Load config
        try:
            config = get_config()
            # Use the sentiment property which returns the full sentiment dict
            sentiment_config = config.sentiment
            self.sentiment_config = sentiment_config.get('cryptopanic', {}) if sentiment_config else {}
            self.trends_config = sentiment_config.get('google_trends', {}) if sentiment_config else {}
        except Exception as e:
            logger.warning(f"Failed to load market context config: {e}")
            self.sentiment_config = {}
            self.trends_config = {}
        
        logger.info("MarketContextFilter initialized")
    
    def validate_buy_signal(
        self, 
        pair: str, 
        strategy_name: str
    ) -> Tuple[bool, str, Dict]:
        """
        Validate if a BUY signal should be executed.
        
        Args:
            pair: Trading pair (e.g., 'BTC/USDT')
            strategy_name: Name of the strategy generating the signal
        
        Returns:
            tuple: (
                allowed: bool,
                reason: str,
                details: dict with sentiment and trends data
            )
        """
        details = {
            'sentiment': None,
            'trends': None,
            'filters_applied': []
        }
        
        # Extract currency from pair (e.g., 'BTC/USDT' -> 'BTC')
        currency = pair.split('/')[0] if '/' in pair else pair.replace('USDT', '')
        
        # 1. Check sentiment filter
        if self.sentiment_config.get('enabled', True):
            sentiment_allowed, sentiment_reason = self.sentiment_client.is_buy_allowed(currency)
            details['sentiment'] = self.sentiment_client.get_sentiment(currency)
            details['filters_applied'].append('sentiment')
            
            if not sentiment_allowed:
                logger.warning(f"BUY signal BLOCKED for {pair}: {sentiment_reason}")
                return False, sentiment_reason, details
        
        # 2. Check trends confirmation for momentum strategy
        if strategy_name == 'momentum' and self.trends_config.get('use_as_momentum_confirmation', True):
            trends_confirmed, trends_reason = self.trends_analyzer.is_momentum_confirmed()
            details['trends'] = self.trends_analyzer.get_interest_data()
            details['filters_applied'].append('trends')
            
            if not trends_confirmed:
                logger.info(f"Momentum signal for {pair} lacks trends confirmation: {trends_reason}")
                # Note: We don't block, just log. The signal proceeds but without confirmation.
                return True, f"Signal allowed (no trends confirmation): {trends_reason}", details
            else:
                logger.info(f"Momentum signal for {pair} CONFIRMED by trends: {trends_reason}")
                return True, f"Signal CONFIRMED: {trends_reason}", details
        
        return True, "Signal allowed - all filters passed", details
    
    def validate_sell_signal(
        self, 
        pair: str, 
        strategy_name: str
    ) -> Tuple[bool, str, Dict]:
        """
        Validate if a SELL signal should be executed.
        
        Currently, SELL signals are not filtered by sentiment/trends.
        This method exists for future extensibility.
        
        Args:
            pair: Trading pair
            strategy_name: Strategy name
        
        Returns:
            tuple: (allowed: bool, reason: str, details: dict)
        """
        # SELL signals are always allowed (no sentiment filter for exits)
        return True, "SELL signals not filtered", {'filters_applied': []}
    
    def get_market_context(self, currency: str = "BTC") -> Dict:
        """
        Get current market context summary.
        
        Args:
            currency: Currency to analyze
        
        Returns:
            dict: Combined sentiment and trends data
        """
        return {
            'currency': currency,
            'sentiment': self.sentiment_client.get_sentiment(currency),
            'trends': self.trends_analyzer.get_interest_data(),
            'timestamp': datetime.now().isoformat()
        }


class LiveBot:
    """
    Main bot for paper/live trading with market context filtering.
    
    This bot:
    1. Fetches latest market data
    2. Generates signals using configured strategies
    3. Validates signals through market context filters
    4. Executes (or simulates) trades
    5. Logs all activity
    
    Attributes:
        paper_trading (bool): If True, simulate trades without real execution
        pairs (list): Trading pairs to monitor
        strategies (dict): Strategy instances by name
    """
    
    def __init__(
        self,
        pairs: Optional[List[str]] = None,
        strategies: Optional[List[str]] = None,
        paper_trading: bool = True,
        timeframe: str = '4h'
    ):
        """
        Initialize Live Bot.
        
        Args:
            pairs: Trading pairs (defaults to config)
            strategies: Strategy names (defaults to config)
            paper_trading: Simulate trades if True
            timeframe: Candle timeframe for signals
        """
        self.paper_trading = paper_trading
        self.timeframe = timeframe
        
        # Load config
        config = get_config()
        
        # Set pairs - use pairs property
        if pairs is None:
            pairs_config = config.pairs
            self.pairs = pairs_config.get('active', ['BTC/USDT', 'ETH/USDT']) if pairs_config else ['BTC/USDT', 'ETH/USDT']
        else:
            self.pairs = pairs
        
        # Set strategies - use strategies property
        if strategies is None:
            strategies_config = config.strategies
            if strategies_config:
                self.strategy_names = [
                    name for name, cfg in strategies_config.items()
                    if isinstance(cfg, dict) and cfg.get('enabled', True)
                ]
            else:
                self.strategy_names = []
        else:
            self.strategy_names = strategies
        
        # Initialize components
        self.data_fetcher = BinanceDataFetcher(testnet=paper_trading)
        self.market_filter = MarketContextFilter()
        self.risk_manager = RiskManager()
        self.trade_logger = TradeLogger(log_dir='data/live_trading')
        
        # Initialize strategies
        self.strategies = {}
        strategies_config = config.strategies or {}
        for name in self.strategy_names:
            try:
                strategy_params = strategies_config.get(name, {})
                # Filter out non-param keys like 'enabled'
                if isinstance(strategy_params, dict):
                    params = {k: v for k, v in strategy_params.items() if k != 'enabled'}
                else:
                    params = {}
                self.strategies[name] = get_strategy(name, **params)
                logger.info(f"Loaded strategy: {name}")
            except Exception as e:
                logger.error(f"Failed to load strategy {name}: {e}")
        
        # Track open positions
        self.positions: Dict[str, Dict] = {}
        
        # Initialize exchange connection
        self._init_exchange()
        
        logger.info(
            f"LiveBot initialized: "
            f"pairs={self.pairs}, strategies={list(self.strategies.keys())}, "
            f"paper_trading={paper_trading}"
        )
    
    def _init_exchange(self) -> None:
        """Initialize exchange connection."""
        try:
            if self.paper_trading:
                # Use testnet or simulate
                self.exchange = ccxt.binance({
                    'enableRateLimit': True,
                    'options': {'defaultType': 'spot'}
                })
                logger.info("Exchange initialized (paper trading mode)")
            else:
                # Real trading - use API keys
                api_key = os.getenv('BINANCE_API_KEY')
                api_secret = os.getenv('BINANCE_API_SECRET')
                
                if not api_key or not api_secret:
                    raise ValueError("API keys not configured for live trading")
                
                self.exchange = ccxt.binance({
                    'apiKey': api_key,
                    'secret': api_secret,
                    'enableRateLimit': True,
                    'options': {'defaultType': 'spot'}
                })
                logger.info("Exchange initialized (LIVE trading mode)")
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")
            self.exchange = None
    
    def fetch_latest_data(self, pair: str, limit: int = 200) -> Optional[pd.DataFrame]:
        """
        Fetch latest OHLCV data for a pair.
        
        Args:
            pair: Trading pair
            limit: Number of candles to fetch
        
        Returns:
            DataFrame with OHLCV data or None on error
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(pair, self.timeframe, limit=limit)
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            logger.error(f"Failed to fetch data for {pair}: {e}")
            return None
    
    def generate_signal(
        self, 
        df: pd.DataFrame, 
        strategy_name: str
    ) -> Tuple[int, pd.DataFrame]:
        """
        Generate trading signal using a strategy.
        
        Args:
            df: OHLCV DataFrame
            strategy_name: Strategy to use
        
        Returns:
            tuple: (signal: int, df_with_indicators: DataFrame)
            signal: 1 (BUY), -1 (SELL), 0 (HOLD)
        """
        if strategy_name not in self.strategies:
            logger.error(f"Strategy not loaded: {strategy_name}")
            return 0, df
        
        strategy = self.strategies[strategy_name]
        df_signals = strategy.generate_signals(df.copy())
        
        signal = df_signals['signal'].iloc[-1]
        return signal, df_signals
    
    def execute_buy(
        self, 
        pair: str, 
        price: float, 
        strategy_name: str,
        context_details: Dict
    ) -> bool:
        """
        Execute a BUY order (or simulate in paper mode).
        
        Args:
            pair: Trading pair
            price: Entry price
            strategy_name: Strategy that generated the signal
            context_details: Market context filter details
        
        Returns:
            bool: True if order executed successfully
        """
        if self.paper_trading:
            logger.info(f"[PAPER] BUY {pair} @ {price} (strategy: {strategy_name})")
            self.positions[pair] = {
                'entry_price': price,
                'entry_time': datetime.now(),
                'strategy': strategy_name,
                'context': context_details
            }
            
            # Log trade
            self.trade_logger.log_trade(
                pair=pair,
                signal='BUY',
                entry_price=price,
                notes=f"Strategy: {strategy_name}"
            )
            return True
        else:
            # Real order execution
            try:
                # Calculate position size (simplified)
                balance = self.exchange.fetch_balance()
                usdt_available = balance['USDT']['free']
                position_size = self.risk_manager.calculate_position_size(
                    capital=usdt_available,
                    entry_price=price,
                    stop_loss_price=price * 0.98  # Simplified 2% stop
                )
                
                order = self.exchange.create_market_buy_order(pair, position_size)
                logger.info(f"[LIVE] BUY {pair}: {order}")
                
                self.positions[pair] = {
                    'entry_price': order['average'],
                    'entry_time': datetime.now(),
                    'strategy': strategy_name,
                    'order_id': order['id'],
                    'amount': position_size
                }
                return True
            except Exception as e:
                logger.error(f"Failed to execute BUY for {pair}: {e}")
                return False
    
    def execute_sell(
        self, 
        pair: str, 
        price: float, 
        strategy_name: str
    ) -> bool:
        """
        Execute a SELL order (or simulate in paper mode).
        
        Args:
            pair: Trading pair
            price: Exit price
            strategy_name: Strategy that generated the signal
        
        Returns:
            bool: True if order executed successfully
        """
        if pair not in self.positions:
            logger.warning(f"No position to sell for {pair}")
            return False
        
        position = self.positions[pair]
        entry_price = position['entry_price']
        pnl_pct = ((price - entry_price) / entry_price) * 100
        
        if self.paper_trading:
            logger.info(
                f"[PAPER] SELL {pair} @ {price} "
                f"(entry: {entry_price}, PnL: {pnl_pct:.2f}%)"
            )
            
            # Log trade
            self.trade_logger.log_trade(
                pair=pair,
                signal='SELL',
                entry_price=entry_price,
                exit_price=price,
                pnl=(price - entry_price),
                notes=f"Strategy: {strategy_name}, PnL: {pnl_pct:.2f}%"
            )
        else:
            # Real order execution
            try:
                amount = position.get('amount')
                order = self.exchange.create_market_sell_order(pair, amount)
                logger.info(f"[LIVE] SELL {pair}: {order}")
            except Exception as e:
                logger.error(f"Failed to execute SELL for {pair}: {e}")
                return False
        
        del self.positions[pair]
        return True
    
    def run_once(self) -> Dict:
        """
        Run one iteration of the trading loop.
        
        Checks all pairs with all strategies, validates signals,
        and executes trades.
        
        Returns:
            dict: Summary of actions taken
        """
        summary = {
            'timestamp': datetime.now().isoformat(),
            'signals_generated': 0,
            'signals_blocked': 0,
            'trades_executed': 0,
            'details': []
        }
        
        for pair in self.pairs:
            # Fetch latest data
            df = self.fetch_latest_data(pair)
            if df is None:
                continue
            
            for strategy_name in self.strategies:
                # Generate signal
                signal, df_signals = self.generate_signal(df, strategy_name)
                
                if signal == 0:
                    continue  # HOLD - no action
                
                summary['signals_generated'] += 1
                current_price = df['close'].iloc[-1]
                
                action_detail = {
                    'pair': pair,
                    'strategy': strategy_name,
                    'signal': 'BUY' if signal == 1 else 'SELL',
                    'price': current_price
                }
                
                if signal == 1:  # BUY
                    # Check if we already have a position
                    if pair in self.positions:
                        action_detail['result'] = 'SKIPPED - position exists'
                        summary['details'].append(action_detail)
                        continue
                    
                    # Validate through market context filter
                    allowed, reason, context = self.market_filter.validate_buy_signal(
                        pair, strategy_name
                    )
                    
                    action_detail['filter_result'] = reason
                    action_detail['context'] = context
                    
                    if not allowed:
                        summary['signals_blocked'] += 1
                        action_detail['result'] = 'BLOCKED'
                        logger.warning(f"Signal blocked: {pair} {strategy_name} - {reason}")
                    else:
                        # Execute trade
                        success = self.execute_buy(pair, current_price, strategy_name, context)
                        action_detail['result'] = 'EXECUTED' if success else 'FAILED'
                        if success:
                            summary['trades_executed'] += 1
                
                elif signal == -1:  # SELL
                    # Validate (currently no filter for SELL)
                    allowed, reason, context = self.market_filter.validate_sell_signal(
                        pair, strategy_name
                    )
                    
                    action_detail['filter_result'] = reason
                    
                    if allowed and pair in self.positions:
                        success = self.execute_sell(pair, current_price, strategy_name)
                        action_detail['result'] = 'EXECUTED' if success else 'FAILED'
                        if success:
                            summary['trades_executed'] += 1
                    else:
                        action_detail['result'] = 'SKIPPED - no position'
                
                summary['details'].append(action_detail)
        
        logger.info(
            f"Run complete: {summary['signals_generated']} signals, "
            f"{summary['signals_blocked']} blocked, "
            f"{summary['trades_executed']} trades"
        )
        
        return summary
    
    def run(self, interval_hours: float = 4.0) -> None:
        """
        Run the trading bot in a continuous loop.
        
        Args:
            interval_hours: Hours between iterations (default: 4h for swing)
        """
        logger.info(f"Starting LiveBot loop (interval: {interval_hours}h)")
        
        while True:
            try:
                summary = self.run_once()
                logger.info(f"Iteration complete: {summary}")
                
                # Sleep until next iteration
                sleep_seconds = interval_hours * 3600
                logger.info(f"Sleeping for {interval_hours} hours...")
                time.sleep(sleep_seconds)
                
            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                # Sleep briefly before retrying
                time.sleep(60)


# CLI entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Crypto Swing Trading Bot")
    parser.add_argument('--live', action='store_true', help="Enable live trading (default: paper)")
    parser.add_argument('--pairs', nargs='+', help="Trading pairs (e.g., BTC/USDT ETH/USDT)")
    parser.add_argument('--strategies', nargs='+', help="Strategies to use")
    parser.add_argument('--interval', type=float, default=4.0, help="Hours between checks")
    parser.add_argument('--once', action='store_true', help="Run once and exit")
    
    args = parser.parse_args()
    
    bot = LiveBot(
        pairs=args.pairs,
        strategies=args.strategies,
        paper_trading=not args.live
    )
    
    if args.once:
        result = bot.run_once()
        print(f"Result: {result}")
    else:
        bot.run(interval_hours=args.interval)
