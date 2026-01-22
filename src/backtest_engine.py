"""
Module: src/backtest_engine.py
Description: Vectorized backtest engine using VectorBT
Author: Trading Bot
Date: 2025-01-22
Version: 1.0
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import pandas as pd
import numpy as np

try:
    import vectorbt as vbt
    HAS_VECTORBT = True
except ImportError:
    HAS_VECTORBT = False

from src.config_loader import get_config
from src.logger import get_logger
from src.strategies.base import Strategy

logger = get_logger(__name__)


class BacktestEngine:
    """
    Vectorized backtest engine using VectorBT.
    
    Simulates trading strategy performance on historical data
    with configurable capital, fees, and slippage.
    
    Configuration loaded from config/backtest.json
    """
    
    def __init__(
        self,
        initial_capital: Optional[float] = None,
        fees: Optional[float] = None,
        slippage: Optional[float] = None
    ):
        """
        Initialize Backtest Engine.
        
        Args:
            initial_capital (Optional[float]): Starting capital in EUR/USDT
            fees (Optional[float]): Trading fees as decimal (0.001 = 0.1%)
            slippage (Optional[float]): Slippage as decimal (0.0005 = 0.05%)
        """
        if not HAS_VECTORBT:
            logger.warning(
                "VectorBT not installed. Using simple backtest implementation."
            )
        
        config = get_config()
        backtest_config = config.backtest
        
        self.initial_capital = initial_capital or backtest_config.get(
            'initial_capital', 2000
        )
        self.fees = fees or backtest_config.get('fees', 0.001)
        self.slippage = slippage or backtest_config.get('slippage', 0.0005)
        
        logger.info(
            f"BacktestEngine initialized: capital={self.initial_capital}, "
            f"fees={self.fees*100:.2f}%, slippage={self.slippage*100:.3f}%"
        )
    
    def run(
        self,
        df: pd.DataFrame,
        strategy: Strategy,
        pair: str = "UNKNOWN"
    ) -> Dict[str, Any]:
        """
        Run backtest on historical data with given strategy.
        
        Args:
            df (pd.DataFrame): OHLCV DataFrame with 'close' column
            strategy (Strategy): Trading strategy instance
            pair (str): Trading pair name (for logging)
        
        Returns:
            Dict[str, Any]: Backtest results containing:
                - strategy: Strategy name
                - pair: Trading pair
                - total_return: Total return percentage
                - sharpe_ratio: Sharpe ratio
                - max_drawdown: Maximum drawdown percentage
                - win_rate: Win rate percentage
                - num_trades: Number of trades
                - profit_factor: Profit factor
                - equity_curve: Portfolio value over time
                - trades: Trade details DataFrame
        
        Raises:
            ValueError: If DataFrame is invalid
        """
        if df is None or df.empty:
            raise ValueError("DataFrame is empty")
        
        if 'close' not in df.columns:
            raise ValueError("DataFrame must have 'close' column")
        
        logger.info(
            f"Running backtest: {strategy.name} on {pair} "
            f"({len(df)} candles)"
        )
        
        # Generate signals
        df_signals = strategy.generate_signals(df)
        
        if 'signal' not in df_signals.columns:
            raise ValueError("Strategy did not generate 'signal' column")
        
        if HAS_VECTORBT:
            return self._run_vectorbt(df_signals, strategy, pair)
        else:
            return self._run_simple(df_signals, strategy, pair)
    
    def _run_vectorbt(
        self,
        df: pd.DataFrame,
        strategy: Strategy,
        pair: str
    ) -> Dict[str, Any]:
        """Run backtest using VectorBT."""
        # Define entry and exit signals
        entries = df['signal'] == 1
        exits = df['signal'] == -1
        
        # Handle timezone-aware timestamps
        close = df['close'].copy()
        if isinstance(df.index, pd.DatetimeIndex):
            close.index = df.index
        elif 'timestamp' in df.columns:
            close.index = pd.to_datetime(df['timestamp'])
        
        # Create portfolio from signals
        portfolio = vbt.Portfolio.from_signals(
            close=close,
            entries=entries,
            exits=exits,
            init_cash=self.initial_capital,
            fees=self.fees,
            slippage=self.slippage,
            freq='4h'
        )
        
        # Extract metrics
        total_return = float(portfolio.total_return()) * 100
        sharpe_ratio = float(portfolio.sharpe_ratio()) if not np.isnan(
            portfolio.sharpe_ratio()
        ) else 0.0
        max_drawdown = float(portfolio.max_drawdown()) * 100
        
        # Get trades info
        trades = portfolio.trades.records_readable
        num_trades = len(trades) if trades is not None else 0
        
        # Calculate win rate
        if num_trades > 0:
            winning_trades = trades[trades['PnL'] > 0]
            win_rate = len(winning_trades) / num_trades * 100
            
            # Profit factor
            gross_profit = trades[trades['PnL'] > 0]['PnL'].sum()
            gross_loss = abs(trades[trades['PnL'] < 0]['PnL'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        else:
            win_rate = 0.0
            profit_factor = 0.0
        
        results = {
            'strategy': strategy.name,
            'pair': pair,
            'total_return': round(total_return, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'max_drawdown': round(max_drawdown, 2),
            'win_rate': round(win_rate, 2),
            'num_trades': num_trades,
            'profit_factor': round(profit_factor, 2),
            'initial_capital': self.initial_capital,
            'final_capital': round(
                self.initial_capital * (1 + total_return / 100), 2
            ),
            'equity_curve': portfolio.value(),
            'trades': trades,
            'portfolio': portfolio
        }
        
        logger.info(
            f"✅ Backtest complete: {strategy.name} ({pair}) - "
            f"Return: {total_return:.2f}%, Sharpe: {sharpe_ratio:.2f}, "
            f"MaxDD: {max_drawdown:.2f}%, WinRate: {win_rate:.1f}%"
        )
        
        return results
    
    def _run_simple(
        self,
        df: pd.DataFrame,
        strategy: Strategy,
        pair: str
    ) -> Dict[str, Any]:
        """
        Simple backtest implementation without VectorBT.
        
        Used as fallback when VectorBT is not installed.
        """
        logger.warning("Using simple backtest (VectorBT not available)")
        
        capital = self.initial_capital
        position = 0
        entry_price = 0
        trades: List[Dict] = []
        equity_curve = []
        
        for i, row in df.iterrows():
            price = row['close']
            signal = row['signal']
            
            # Calculate current portfolio value
            if position > 0:
                portfolio_value = capital + (position * price)
            else:
                portfolio_value = capital
            equity_curve.append(portfolio_value)
            
            # Buy signal
            if signal == 1 and position == 0:
                # Apply slippage
                buy_price = price * (1 + self.slippage)
                # Calculate position size (use 95% of capital for fees)
                available = capital * 0.95
                position = available / buy_price
                capital -= position * buy_price * (1 + self.fees)
                entry_price = buy_price
                
            # Sell signal
            elif signal == -1 and position > 0:
                # Apply slippage
                sell_price = price * (1 - self.slippage)
                proceeds = position * sell_price * (1 - self.fees)
                capital += proceeds
                
                pnl = (sell_price - entry_price) / entry_price * 100
                trades.append({
                    'entry_price': entry_price,
                    'exit_price': sell_price,
                    'pnl_percent': pnl
                })
                
                position = 0
                entry_price = 0
        
        # Calculate final portfolio value
        if position > 0:
            final_value = capital + (position * df['close'].iloc[-1])
        else:
            final_value = capital
        
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100
        
        # Calculate metrics
        num_trades = len(trades)
        if num_trades > 0:
            winning_trades = [t for t in trades if t['pnl_percent'] > 0]
            win_rate = len(winning_trades) / num_trades * 100
        else:
            win_rate = 0.0
        
        # Simple Sharpe approximation
        equity_series = pd.Series(equity_curve)
        returns = equity_series.pct_change().dropna()
        sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(252 * 6)
                       if returns.std() > 0 else 0.0)
        
        # Max drawdown
        equity_series = pd.Series(equity_curve)
        rolling_max = equity_series.expanding().max()
        drawdowns = (equity_series - rolling_max) / rolling_max * 100
        max_drawdown = drawdowns.min()
        
        results = {
            'strategy': strategy.name,
            'pair': pair,
            'total_return': round(total_return, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'max_drawdown': round(max_drawdown, 2),
            'win_rate': round(win_rate, 2),
            'num_trades': num_trades,
            'profit_factor': 0.0,  # Not calculated in simple version
            'initial_capital': self.initial_capital,
            'final_capital': round(final_value, 2),
            'equity_curve': pd.Series(equity_curve),
            'trades': pd.DataFrame(trades),
            'portfolio': None
        }
        
        logger.info(
            f"✅ Simple backtest complete: {strategy.name} ({pair}) - "
            f"Return: {total_return:.2f}%"
        )
        
        return results
    
    def run_multiple_strategies(
        self,
        df: pd.DataFrame,
        strategies: List[Strategy],
        pair: str = "UNKNOWN"
    ) -> List[Dict[str, Any]]:
        """
        Run backtest for multiple strategies on same data.
        
        Args:
            df (pd.DataFrame): OHLCV DataFrame
            strategies (List[Strategy]): List of strategy instances
            pair (str): Trading pair name
        
        Returns:
            List[Dict[str, Any]]: List of backtest results
        """
        results = []
        
        for strategy in strategies:
            try:
                result = self.run(df, strategy, pair)
                results.append(result)
            except Exception as e:
                logger.error(f"Backtest failed for {strategy.name}: {e}")
                continue
        
        return results
    
    def compare_results(
        self,
        results: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Compare multiple backtest results.
        
        Args:
            results (List[Dict[str, Any]]): List of backtest results
        
        Returns:
            pd.DataFrame: Comparison table
        """
        comparison_data = []
        
        for r in results:
            comparison_data.append({
                'Strategy': r['strategy'],
                'Pair': r['pair'],
                'Return (%)': r['total_return'],
                'Sharpe': r['sharpe_ratio'],
                'Max DD (%)': r['max_drawdown'],
                'Win Rate (%)': r['win_rate'],
                'Trades': r['num_trades'],
                'Final Capital': r['final_capital']
            })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('Return (%)', ascending=False)
        
        return df
