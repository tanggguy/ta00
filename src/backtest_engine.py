"""
Module: src/backtest_engine.py
Description: Vectorized backtest engine using VectorBT
Author: Trading Bot
Date: 2025-01-22
Version: 1.1
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
from src.risk_manager import RiskManager

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
        
        self.risk_manager = RiskManager()
        
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
            Dict[str, Any]: Backtest results
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
        
        # --- FIX: Ensure index is DatetimeIndex for all series ---
        df_vbt = df.copy()
        if 'timestamp' in df_vbt.columns:
            # Convert timestamp to datetime if not already
            df_vbt['timestamp'] = pd.to_datetime(df_vbt['timestamp'])
            df_vbt.set_index('timestamp', inplace=True)
        elif not isinstance(df_vbt.index, pd.DatetimeIndex):
            # Fallback if no timestamp col and index is not datetime
            logger.warning("No timestamp column found and index is not DatetimeIndex. Using integer index.")
        # ---------------------------------------------------------

        # Define entry and exit signals (now aligned with DatetimeIndex)
        entries = df_vbt['signal'] == 1
        exits = df_vbt['signal'] == -1
        close = df_vbt['close']
        
        # --- Risk Management (Vectorized) ---
        # 1. Get Params for this strategy
        risk_pct = self.risk_manager._get_param('risk_per_trade_pct', strategy.name) / 100.0
        max_size_pct = self.risk_manager._get_param('max_position_size_pct', strategy.name) / 100.0
        atr_period = int(self.risk_manager._get_param('atr_period', strategy.name) or 14)
        mult_sl = self.risk_manager._get_param('atr_multiplier_sl', strategy.name)
        mult_tp = self.risk_manager._get_param('atr_multiplier_tp', strategy.name)
        trailing_enabled = self.risk_manager._get_param('trailing_stop_enabled', strategy.name)
        
        # 2. Calculate ATR and Stop Distances
        atr = self.risk_manager.calculate_atr(df_vbt, period=atr_period)
        
        # SL/TP percentages (distance from close)
        # Note: VectorBT expects positive percentages for sl_stop/tp_stop
        sl_stop_pct = (atr * mult_sl) / df_vbt['close']
        tp_stop_pct = (atr * mult_tp) / df_vbt['close']
        
        # 3. Calculate Position Size % (Risk Based)
        # Size % = Risk % / SL Dist %
        # Avoid division by zero
        # Make sure sl_stop_pct is not 0
        sl_stop_pct = sl_stop_pct.replace(0, np.nan).fillna(0.01) # fallback
        
        size_pct = risk_pct / sl_stop_pct
        
        # Cap at max position size
        size_pct = size_pct.clip(upper=max_size_pct)
        
        # Clean up NaNs
        size_pct = size_pct.fillna(0.0)
        sl_stop_pct = sl_stop_pct.fillna(0.0)
        tp_stop_pct = tp_stop_pct.fillna(0.0)

        # Create portfolio from signals with risk parameters
        portfolio = vbt.Portfolio.from_signals(
            close=close,
            entries=entries,
            exits=exits,
            init_cash=self.initial_capital,
            fees=self.fees,
            slippage=self.slippage,
            freq='4h',
            # Risk Parameters
            size=size_pct,
            size_type='percent', # Size is % of current equity (or init_cash if compounded=False?) 
                                 # 'percent' in vbt typically means % of current value (compounded)
            sl_stop=sl_stop_pct,
            tp_stop=tp_stop_pct,
            sl_trail=trailing_enabled
        )
        
        # Extract metrics
        total_return = float(portfolio.total_return()) * 100
        sharpe_ratio = float(portfolio.sharpe_ratio())
        if np.isnan(sharpe_ratio) or np.isinf(sharpe_ratio):
            sharpe_ratio = 0.0
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
        
        # ATR Calculation
        atr_period = int(self.risk_manager._get_param('atr_period', strategy.name) or 14)
        df_simple = df.copy()
        df_simple['atr'] = self.risk_manager.calculate_atr(df_simple, period=atr_period)
        
        # State variables
        stop_loss = None
        take_profit = None
        
        for i, row in df_simple.iterrows():
            price = row['close']
            high = row['high'] if 'high' in row else price
            low = row['low'] if 'low' in row else price
            signal = row['signal']
            atr = row['atr']
            
            # Calculate current portfolio value
            if position > 0:
                portfolio_value = capital + (position * price)
            else:
                portfolio_value = capital
            equity_curve.append(portfolio_value)
            
            # --- Check Exits if in position ---
            exit_signal = False
            exit_price_exec = price
            exit_reason = ""
            
            if position > 0:
                # 1. Trailing Stop Update
                stop_loss = self.risk_manager.check_trailing_stop(
                    current_price=price,
                    current_stop=stop_loss,
                    highest_price=high,
                    lowest_price=low,
                    strategy_name=strategy.name,
                    direction=1
                )
                
                # 2. Check Exits (Order of precedence: SL, TP, Signal)
                # Assuming Low happens before High? We don't know within the bar.
                # Worst case: Check SL first.
                
                if low <= stop_loss:
                    exit_signal = True
                    exit_price_exec = stop_loss
                    exit_reason = "Stop Loss"
                elif high >= take_profit:
                    exit_signal = True
                    exit_price_exec = take_profit
                    exit_reason = "Take Profit"
                elif signal == -1:
                    exit_signal = True
                    exit_price_exec = price
                    exit_reason = "Signal"
            
            # Execute Exit
            if exit_signal and position > 0:
                # Apply slippage
                sell_price = exit_price_exec * (1 - self.slippage)
                proceeds = position * sell_price * (1 - self.fees)
                capital += proceeds
                
                pnl = (sell_price - entry_price) / entry_price * 100
                trades.append({
                    'entry_price': entry_price,
                    'exit_price': sell_price,
                    'pnl_percent': pnl,
                    'reason': exit_reason
                })
                
                position = 0
                entry_price = 0
                stop_loss = None
                take_profit = None
                
            
            # --- Check Entry ---
            if signal == 1 and position == 0 and not exit_signal:
                # Calculate Stops
                sl, tp = self.risk_manager.calculate_stops(
                    entry_price=price,
                    atr_value=atr,
                    strategy_name=strategy.name
                )
                stop_loss = sl
                take_profit = tp
                
                # Calculate Size
                size_to_buy = self.risk_manager.calculate_position_size(
                    capital=capital,
                    entry_price=price,
                    stop_loss_price=stop_loss,
                    strategy_name=strategy.name
                )
                
                if size_to_buy > 0:
                    buy_price = price * (1 + self.slippage)
                    cost = size_to_buy * buy_price * (1 + self.fees)
                    
                    if cost <= capital:
                        capital -= cost
                        position = size_to_buy
                        entry_price = buy_price
                    else:
                        # Not enough cash (should be handled by calc, but rounding errs)
                        pass
        
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
        # Prevent division by zero if std is 0
        if returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252 * 6)
        else:
            sharpe_ratio = 0.0
        
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