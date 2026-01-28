"""
Module: src/config_loader.py
Description: Load and validate configuration files from config/ directory
Author: Trading Bot
Date: 2025-01-22
Version: 1.0
"""

import json
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Loads and manages configuration files from the config/ directory.
    
    Provides centralized access to all configuration settings with validation
    and environment variable support.
    """
    
    def __init__(self, config_dir: str = "config/"):
        """
        Initialize ConfigLoader.
        
        Args:
            config_dir (str): Path to configuration directory (default: 'config/')
        
        Raises:
            FileNotFoundError: If config directory doesn't exist
        """
        self.config_dir = Path(config_dir)
        
        if not self.config_dir.exists():
            raise FileNotFoundError(f"Config directory not found: {config_dir}")
        
        # Load environment variables
        load_dotenv()
        
        # Configuration cache
        self._cache: Dict[str, Any] = {}
        
        # Load all configurations
        self._load_all_configs()
        
        logger.info(f"ConfigLoader initialized from {config_dir}")
    
    def _load_json(self, filename: str) -> Dict[str, Any]:
        """
        Load a JSON configuration file.
        
        Args:
            filename (str): Name of the JSON file (without path)
        
        Returns:
            Dict[str, Any]: Parsed JSON content
        
        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If JSON is invalid
        """
        filepath = self.config_dir / filename
        
        if not filepath.exists():
            logger.warning(f"Config file not found: {filename}")
            return {}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.debug(f"Loaded config: {filename}")
            return data
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {filename}: {e}")
            raise
    
    def _load_all_configs(self) -> None:
        """Load all configuration files into cache."""
        config_files = [
            'settings.json',
            'pairs.json',
            'strategies.json',
            'logging.json',
            'backtest.json',
            'optimization.json',
            'risk.json',
            'sentiment.json'
        ]
        
        for filename in config_files:
            key = filename.replace('.json', '')
            self._cache[key] = self._load_json(filename)
        
        logger.info(f"Loaded {len(self._cache)} configuration files")
    
    def get(self, config_name: str, key: Optional[str] = None, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            config_name (str): Name of config file (without .json)
            key (Optional[str]): Specific key to retrieve (None = entire config)
            default (Any): Default value if key not found
        
        Returns:
            Any: Configuration value or default
        
        Examples:
            >>> config.get('settings', 'initial_capital')
            2000
            >>> config.get('strategies', 'trend_following')
            {'enabled': True, 'sma_short': 20, 'sma_long': 50}
        """
        if config_name not in self._cache:
            logger.warning(f"Config not found: {config_name}")
            return default
        
        if key is None:
            return self._cache[config_name]
        
        return self._cache[config_name].get(key, default)
    
    def reload(self, config_name: Optional[str] = None) -> None:
        """
        Reload configuration file(s).
        
        Args:
            config_name (Optional[str]): Specific config to reload (None = all)
        """
        if config_name:
            self._cache[config_name] = self._load_json(f"{config_name}.json")
            logger.info(f"Reloaded config: {config_name}")
        else:
            self._load_all_configs()
            logger.info("Reloaded all configurations")
    
    # Convenience properties
    @property
    def settings(self) -> Dict[str, Any]:
        """Get settings configuration."""
        return self._cache.get('settings', {})
    
    @property
    def pairs(self) -> Dict[str, Any]:
        """Get pairs configuration."""
        return self._cache.get('pairs', {})
    
    @property
    def strategies(self) -> Dict[str, Any]:
        """Get strategies configuration."""
        return self._cache.get('strategies', {})
        
    @property
    def risk(self) -> Dict[str, Any]:
        """Get risk configuration."""
        return self._cache.get('risk', {})
    
    @property
    def logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self._cache.get('logging', {})
    
    @property
    def backtest(self) -> Dict[str, Any]:
        """Get backtest configuration."""
        return self._cache.get('backtest', {})
    
    @property
    def sentiment(self) -> Dict[str, Any]:
        """Get sentiment configuration."""
        return self._cache.get('sentiment', {})
    
    # Environment variables
    def get_env(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get environment variable.
        
        Args:
            key (str): Environment variable name
            default (Optional[str]): Default value if not set
        
        Returns:
            Optional[str]: Environment variable value
        """
        return os.getenv(key, default)
    
    @property
    def binance_api_key(self) -> Optional[str]:
        """Get Binance API key from environment."""
        return self.get_env('BINANCE_API_KEY')
    
    @property
    def binance_api_secret(self) -> Optional[str]:
        """Get Binance API secret from environment."""
        return self.get_env('BINANCE_API_SECRET')
    
    @property
    def is_testnet(self) -> bool:
        """Check if testnet mode is enabled."""
        env_testnet = self.get_env('BINANCE_TESTNET_ENABLED', 'false').lower()
        config_testnet = self.settings.get('testnet_enabled', True)
        return env_testnet == 'true' or config_testnet


# Global config instance (lazy loaded)
_config_instance: Optional[ConfigLoader] = None


def get_config(config_dir: str = "config/") -> ConfigLoader:
    """
    Get or create the global config instance.
    
    Args:
        config_dir (str): Path to configuration directory
    
    Returns:
        ConfigLoader: Global configuration instance
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = ConfigLoader(config_dir)
    
    return _config_instance
