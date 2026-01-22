"""
Module: src/strategies/__init__.py
Description: Strategy module initialization and registry
Author: Trading Bot
Date: 2025-01-22
Version: 1.0
"""

from src.strategies.base import Strategy
from src.strategies.trend_following import TrendFollowing
from src.strategies.mean_reversion import MeanReversion
from src.strategies.momentum import Momentum

# Strategy registry for dynamic loading
STRATEGY_REGISTRY = {
    'trend_following': TrendFollowing,
    'mean_reversion': MeanReversion,
    'momentum': Momentum
}


def get_strategy(name: str, **kwargs) -> Strategy:
    """
    Get strategy instance by name.
    
    Args:
        name (str): Strategy name (trend_following, mean_reversion, momentum)
        **kwargs: Strategy-specific parameters
    
    Returns:
        Strategy: Configured strategy instance
    
    Raises:
        ValueError: If strategy name is not recognized
    """
    if name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy: {name}. "
            f"Available: {list(STRATEGY_REGISTRY.keys())}"
        )
    
    return STRATEGY_REGISTRY[name](**kwargs)


def list_strategies() -> list:
    """List all available strategy names."""
    return list(STRATEGY_REGISTRY.keys())


__all__ = [
    'Strategy',
    'TrendFollowing',
    'MeanReversion',
    'Momentum',
    'get_strategy',
    'list_strategies',
    'STRATEGY_REGISTRY'
]
