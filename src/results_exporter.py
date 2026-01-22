"""
Module: src/results_exporter.py
Description: Export backtest results to CSV and JSON
Author: Trading Bot
Date: 2025-01-22
Version: 1.0
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd

from src.config_loader import get_config
from src.logger import get_logger
from src.utils import ensure_directory, format_pair

logger = get_logger(__name__)


class ResultsExporter:
    """
    Export backtest results to various formats.
    
    Supports CSV and JSON exports with configurable output directory.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize Results Exporter.
        
        Args:
            output_dir (Optional[str]): Output directory (default from config)
        """
        config = get_config()
        backtest_config = config.backtest.get('output', {})
        
        self.output_dir = output_dir or backtest_config.get(
            'results_dir', 'data/backtest_results/'
        )
        
        ensure_directory(self.output_dir)
        
        logger.info(f"ResultsExporter initialized: output_dir={self.output_dir}")
    
    def export_summary(
        self,
        results: Dict[str, Any],
        filename: Optional[str] = None
    ) -> str:
        """
        Export backtest summary to CSV.
        
        Args:
            results (Dict[str, Any]): Backtest results from BacktestEngine
            filename (Optional[str]): Custom filename
        
        Returns:
            str: Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pair_clean = format_pair(results['pair'], separator="")
            filename = f"backtest_{results['strategy']}_{pair_clean}_{timestamp}.csv"
        
        # Create summary DataFrame
        summary_data = {
            'strategy': [results['strategy']],
            'pair': [results['pair']],
            'total_return_pct': [results['total_return']],
            'sharpe_ratio': [results['sharpe_ratio']],
            'max_drawdown_pct': [results['max_drawdown']],
            'win_rate_pct': [results['win_rate']],
            'num_trades': [results['num_trades']],
            'profit_factor': [results.get('profit_factor', 0)],
            'initial_capital': [results['initial_capital']],
            'final_capital': [results['final_capital']],
            'timestamp': [datetime.now().isoformat()]
        }
        
        df = pd.DataFrame(summary_data)
        filepath = Path(self.output_dir) / filename
        df.to_csv(filepath, index=False)
        
        logger.info(f"✅ Exported summary to {filepath}")
        return str(filepath)
    
    def export_trades(
        self,
        results: Dict[str, Any],
        filename: Optional[str] = None
    ) -> Optional[str]:
        """
        Export trade details to CSV.
        
        Args:
            results (Dict[str, Any]): Backtest results
            filename (Optional[str]): Custom filename
        
        Returns:
            Optional[str]: Path to exported file or None if no trades
        """
        trades = results.get('trades')
        
        if trades is None or (isinstance(trades, pd.DataFrame) and trades.empty):
            logger.warning("No trades to export")
            return None
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pair_clean = format_pair(results['pair'], separator="")
            filename = f"trades_{results['strategy']}_{pair_clean}_{timestamp}.csv"
        
        filepath = Path(self.output_dir) / filename
        
        if isinstance(trades, pd.DataFrame):
            trades.to_csv(filepath, index=False)
        else:
            pd.DataFrame(trades).to_csv(filepath, index=False)
        
        logger.info(f"✅ Exported trades to {filepath}")
        return str(filepath)
    
    def export_equity_curve(
        self,
        results: Dict[str, Any],
        filename: Optional[str] = None
    ) -> Optional[str]:
        """
        Export equity curve to CSV.
        
        Args:
            results (Dict[str, Any]): Backtest results
            filename (Optional[str]): Custom filename
        
        Returns:
            Optional[str]: Path to exported file or None
        """
        equity_curve = results.get('equity_curve')
        
        if equity_curve is None:
            logger.warning("No equity curve to export")
            return None
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pair_clean = format_pair(results['pair'], separator="")
            filename = f"equity_{results['strategy']}_{pair_clean}_{timestamp}.csv"
        
        filepath = Path(self.output_dir) / filename
        
        if isinstance(equity_curve, pd.Series):
            equity_curve.to_csv(filepath, header=['portfolio_value'])
        else:
            pd.Series(equity_curve).to_csv(filepath, header=['portfolio_value'])
        
        logger.info(f"✅ Exported equity curve to {filepath}")
        return str(filepath)
    
    def export_all(
        self,
        results: Dict[str, Any],
        prefix: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Export all backtest data (summary, trades, equity curve).
        
        Args:
            results (Dict[str, Any]): Backtest results
            prefix (Optional[str]): Filename prefix
        
        Returns:
            Dict[str, str]: Dictionary of {type: filepath}
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pair_clean = format_pair(results['pair'], separator="")
        
        if prefix is None:
            prefix = f"{results['strategy']}_{pair_clean}_{timestamp}"
        
        exported = {}
        
        # Summary
        summary_file = self.export_summary(
            results, f"backtest_{prefix}.csv"
        )
        exported['summary'] = summary_file
        
        # Trades
        trades_file = self.export_trades(
            results, f"trades_{prefix}.csv"
        )
        if trades_file:
            exported['trades'] = trades_file
        
        # Equity curve
        equity_file = self.export_equity_curve(
            results, f"equity_{prefix}.csv"
        )
        if equity_file:
            exported['equity_curve'] = equity_file
        
        logger.info(f"✅ Exported all results: {list(exported.keys())}")
        return exported
    
    def export_comparison(
        self,
        comparison_df: pd.DataFrame,
        filename: Optional[str] = None
    ) -> str:
        """
        Export strategy comparison table.
        
        Args:
            comparison_df (pd.DataFrame): Comparison DataFrame
            filename (Optional[str]): Custom filename
        
        Returns:
            str: Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comparison_{timestamp}.csv"
        
        filepath = Path(self.output_dir) / filename
        comparison_df.to_csv(filepath, index=False)
        
        logger.info(f"✅ Exported comparison to {filepath}")
        return str(filepath)
    
    def export_json(
        self,
        results: Dict[str, Any],
        filename: Optional[str] = None
    ) -> str:
        """
        Export results to JSON (without non-serializable objects).
        
        Args:
            results (Dict[str, Any]): Backtest results
            filename (Optional[str]): Custom filename
        
        Returns:
            str: Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pair_clean = format_pair(results['pair'], separator="")
            filename = f"results_{results['strategy']}_{pair_clean}_{timestamp}.json"
        
        # Create JSON-serializable copy
        json_results = {
            'strategy': results['strategy'],
            'pair': results['pair'],
            'total_return': results['total_return'],
            'sharpe_ratio': results['sharpe_ratio'],
            'max_drawdown': results['max_drawdown'],
            'win_rate': results['win_rate'],
            'num_trades': results['num_trades'],
            'profit_factor': results.get('profit_factor', 0),
            'initial_capital': results['initial_capital'],
            'final_capital': results['final_capital'],
            'timestamp': datetime.now().isoformat()
        }
        
        filepath = Path(self.output_dir) / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"✅ Exported JSON to {filepath}")
        return str(filepath)
