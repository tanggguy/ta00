"""Tests for configuration loader."""

import pytest
import json
import os
from pathlib import Path
from unittest.mock import patch


class TestConfigLoader:
    """Tests for ConfigLoader class."""
    
    def test_config_loader_initialization(self):
        """Test ConfigLoader initializes correctly."""
        from src.config_loader import ConfigLoader
        
        config = ConfigLoader(config_dir="config/")
        assert config is not None
        assert config.config_dir == Path("config/")
    
    def test_config_loader_missing_directory(self):
        """Test ConfigLoader raises error for missing directory."""
        from src.config_loader import ConfigLoader
        
        with pytest.raises(FileNotFoundError):
            ConfigLoader(config_dir="nonexistent_dir/")
    
    def test_get_settings(self):
        """Test getting settings configuration."""
        from src.config_loader import get_config
        
        config = get_config()
        settings = config.settings
        
        assert 'initial_capital' in settings
        assert settings['initial_capital'] == 2000
    
    def test_get_pairs(self):
        """Test getting pairs configuration."""
        from src.config_loader import get_config
        
        config = get_config()
        pairs = config.pairs
        
        assert 'pairs' in pairs
        assert 'BTC/USDT' in pairs['pairs']
        assert 'ETH/USDT' in pairs['pairs']
    
    def test_get_strategies_config(self):
        """Test getting strategies configuration."""
        from src.config_loader import get_config
        
        config = get_config()
        strategies = config.strategies
        
        assert 'trend_following' in strategies
        assert 'mean_reversion' in strategies
        assert 'momentum' in strategies
    
    def test_get_specific_value(self):
        """Test getting specific configuration value."""
        from src.config_loader import get_config
        
        config = get_config()
        capital = config.get('settings', 'initial_capital')
        
        assert capital == 2000
    
    def test_get_with_default(self):
        """Test getting value with default fallback."""
        from src.config_loader import get_config
        
        config = get_config()
        value = config.get('settings', 'nonexistent_key', default='default_value')
        
        assert value == 'default_value'
    
    def test_backtest_config(self):
        """Test getting backtest configuration."""
        from src.config_loader import get_config
        
        config = get_config()
        backtest = config.backtest
        
        assert 'start_date' in backtest
        assert 'end_date' in backtest
        assert 'timeframe' in backtest
        assert backtest['timeframe'] == '4h'
    
    def test_logging_config(self):
        """Test getting logging configuration."""
        from src.config_loader import get_config
        
        config = get_config()
        logging = config.logging_config
        
        assert 'log_level' in logging
        assert 'module_levels' in logging
