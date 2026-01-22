"""
Module: src/logger.py
Description: Configurable logging system with per-module log levels
Author: Trading Bot
Date: 2025-01-22
Version: 1.0
"""

import logging
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any
from logging.handlers import RotatingFileHandler


class LoggerFactory:
    """
    Factory class for creating configured loggers.
    
    Reads configuration from config/logging.json and provides
    consistent logging setup across all modules.
    """
    
    _initialized: bool = False
    _config: Dict[str, Any] = {}
    _log_dir: str = "logs/"
    
    @classmethod
    def _load_config(cls) -> Dict[str, Any]:
        """Load logging configuration from JSON file."""
        config_path = Path("config/logging.json")
        
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Invalid logging.json, using defaults")
        
        # Default configuration
        return {
            "log_level": "INFO",
            "log_to_file": True,
            "log_to_console": True,
            "log_dir": "logs/",
            "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "date_format": "%Y-%m-%d %H:%M:%S",
            "module_levels": {},
            "rotation": {
                "enabled": True,
                "max_bytes": 10485760,
                "backup_count": 5
            }
        }
    
    @classmethod
    def initialize(cls, config_path: Optional[str] = None) -> None:
        """
        Initialize the logging system.
        
        Args:
            config_path (Optional[str]): Path to logging config file
        """
        if cls._initialized:
            return
        
        cls._config = cls._load_config()
        cls._log_dir = cls._config.get("log_dir", "logs/")
        
        # Create log directory
        os.makedirs(cls._log_dir, exist_ok=True)
        
        cls._initialized = True
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get a configured logger for a module.
        
        Args:
            name (str): Logger name (typically __name__)
        
        Returns:
            logging.Logger: Configured logger instance
        
        Examples:
            >>> logger = LoggerFactory.get_logger(__name__)
            >>> logger.info("Application started")
        """
        if not cls._initialized:
            cls.initialize()
        
        logger = logging.getLogger(name)
        
        # Prevent duplicate handlers
        if logger.handlers:
            return logger
        
        # Determine log level for this module
        module_levels = cls._config.get("module_levels", {})
        default_level = cls._config.get("log_level", "INFO")
        level_name = module_levels.get(name, default_level)
        level = getattr(logging, level_name.upper(), logging.INFO)
        
        logger.setLevel(level)
        
        # Create formatters
        log_format = cls._config.get(
            "log_format",
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        date_format = cls._config.get("date_format", "%Y-%m-%d %H:%M:%S")
        formatter = logging.Formatter(log_format, datefmt=date_format)
        
        # Console handler
        if cls._config.get("log_to_console", True):
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # File handler
        if cls._config.get("log_to_file", True):
            timestamp = datetime.now().strftime("%Y%m%d")
            log_file = Path(cls._log_dir) / f"trading_{timestamp}.log"
            
            rotation_config = cls._config.get("rotation", {})
            if rotation_config.get("enabled", True):
                file_handler = RotatingFileHandler(
                    log_file,
                    maxBytes=rotation_config.get("max_bytes", 10485760),
                    backupCount=rotation_config.get("backup_count", 5),
                    encoding='utf-8'
                )
            else:
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
            
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    @classmethod
    def set_level(cls, name: str, level: str) -> None:
        """
        Dynamically change log level for a logger.
        
        Args:
            name (str): Logger name
            level (str): Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        logger = logging.getLogger(name)
        log_level = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(log_level)
        
        for handler in logger.handlers:
            handler.setLevel(log_level)


def get_logger(name: str) -> logging.Logger:
    """
    Convenience function to get a logger.
    
    Args:
        name (str): Logger name (typically __name__)
    
    Returns:
        logging.Logger: Configured logger instance
    
    Examples:
        >>> from src.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Data fetching started")
        >>> logger.warning("Rate limit approaching")
        >>> logger.error("API connection failed")
    """
    return LoggerFactory.get_logger(name)


def setup_logging(config_path: Optional[str] = None) -> None:
    """
    Initialize the logging system.
    
    Args:
        config_path (Optional[str]): Path to logging config file
    """
    LoggerFactory.initialize(config_path)
