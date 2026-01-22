"""
Module: src/risk_manager.py
Description: Risk management system handling position sizing and stop/limit orders.
Author: Trading Bot
Date: 2025-01-22
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import ta
from src.config_loader import get_config
from src.logger import get_logger

logger = get_logger(__name__)

class RiskManager:
    """
    Manages risk for trading strategies including:
    - Position Sizing (Risk-based)
    - Stop Loss & Take Profit (ATR-based)
    - Trailing Stops
    """
    
    def __init__(self):
        self.config = get_config()
        self.risk_config = self.config.risk if hasattr(self.config, 'risk') else {}
        if not self.risk_config:
            # Fallback if config not loaded correctly yet
            # In a real scenario, we might reload or warn.
            logger.warning("Risk configuration empty or not found in global config.")
            self.defaults = {}
        else:
            self.defaults = self.risk_config.get('global_defaults', {})
            
    def _get_param(self, param_name: str, strategy_name: str) -> float:
        """
        Get parameter value, prioritizing strategy-specific config over global defaults.
        """
        strategy_config = self.risk_config.get('strategies', {}).get(strategy_name, {})
        if param_name in strategy_config:
            return strategy_config[param_name]
        return self.defaults.get(param_name, 0.0)

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        """
        high = df['high'] if 'high' in df.columns else df['close'] * 1.001 # Fallback
        low = df['low'] if 'low' in df.columns else df['close'] * 0.999   # Fallback
        close = df['close']
        
        return ta.volatility.average_true_range(high, low, close, window=period)

    def calculate_stops(self, 
                       entry_price: float, 
                       atr_value: float, 
                       strategy_name: str, 
                       direction: int = 1) -> Tuple[float, float]:
        """
        Calculate Stop Loss and Take Profit levels based on ATR.
        
        Args:
            entry_price: Price of entry
            atr_value: Current ATR value
            strategy_name: Name of strategy to look up params
            direction: 1 for Long, -1 for Short
            
        Returns:
            (stop_loss_price, take_profit_price)
        """
        atr_mult_sl = self._get_param('atr_multiplier_sl', strategy_name)
        atr_mult_tp = self._get_param('atr_multiplier_tp', strategy_name)
        
        if direction == 1: # LONG
            sl = entry_price - (atr_value * atr_mult_sl)
            tp = entry_price + (atr_value * atr_mult_tp)
        else: # SHORT
            sl = entry_price + (atr_value * atr_mult_sl)
            tp = entry_price - (atr_value * atr_mult_tp)
            
        return sl, tp

    def calculate_position_size(self, 
                              capital: float, 
                              entry_price: float, 
                              stop_loss_price: float, 
                              strategy_name: str) -> float:
        """
        Calculate position size based on risk percentage.
        
        Formula:
        Position Size = (Capital * Risk_Per_Trade_Pct) / |Entry - SL|
        
        Also capped by Max_Position_Size_Pct.
        """
        if entry_price == stop_loss_price:
            return 0.0
            
        risk_per_trade_pct = self._get_param('risk_per_trade_pct', strategy_name) / 100.0
        max_pos_size_pct = self._get_param('max_position_size_pct', strategy_name) / 100.0
        
        risk_amount = capital * risk_per_trade_pct
        price_diff = abs(entry_price - stop_loss_price)
        
        # Theoretical size to risk exactly 'risk_amount'
        size_based_on_risk = risk_amount / price_diff
        
        # Cap size by absolute capital percentage (e.g., max 20% of portfolio)
        max_size_based_on_capital = (capital * max_pos_size_pct) / entry_price
        
        final_size = min(size_based_on_risk, max_size_based_on_capital)
        
        return final_size

    def check_trailing_stop(self, 
                          current_price: float, 
                          current_stop: float, 
                          highest_price: float, # For Longs
                          lowest_price: float,  # For Shorts
                          strategy_name: str,
                          direction: int = 1) -> float:
        """
        Calculate new trailing stop price if applicable.
        
        Returns:
            New stop price (or current one if no change)
        """
        if not self._get_param('trailing_stop_enabled', strategy_name):
            return current_stop
            
        # Implementation of a simple chandelier-like or percent-based trailing stop
        # For simplicity here, we'll use a callback percentage from the peak
        # But commonly we might want to just move SL up if price moves up
        
        # Let's use ATR based trailing if available or fallback to defaults? 
        # The config had 'trailing_stop_callback_pct'.
        
        callback_pct = self._get_param('trailing_stop_callback_pct', strategy_name) / 100.0
        
        new_stop = current_stop
        
        if direction == 1: # LONG
            # If price moved up, potential new stop is high - callback
            potential_stop = highest_price * (1 - callback_pct)
            if potential_stop > current_stop:
                new_stop = potential_stop
        else: # SHORT
            potential_stop = lowest_price * (1 + callback_pct)
            if potential_stop < current_stop:
                new_stop = potential_stop
                
        return new_stop
