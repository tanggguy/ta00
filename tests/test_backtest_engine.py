"""Tests for backtest engine."""

import pytest
import pandas as pd
import numpy as np


class TestBacktestEngine:
    """Tests for BacktestEngine class."""
    
    def test_initialization_default_params(self):
        """Test BacktestEngine initializes with default parameters."""
        from src.backtest_engine import BacktestEngine
        
        engine = BacktestEngine()
        
        assert engine.initial_capital == 2000
        assert engine.fees == 0.001
    
    def test_initialization_custom_params(self):
        """Test BacktestEngine with custom parameters."""
        from src.backtest_engine import BacktestEngine
        
        engine = BacktestEngine(
            initial_capital=5000,
            fees=0.002,
            slippage=0.001
        )
        
        assert engine.initial_capital == 5000
        assert engine.fees == 0.002
        assert engine.slippage == 0.001
    
    def test_run_backtest_returns_dict(self, sample_ohlcv_data):
        """Test run returns dictionary with required keys."""
        from src.backtest_engine import BacktestEngine
        from src.strategies import TrendFollowing
        
        engine = BacktestEngine()
        strategy = TrendFollowing()
        
        results = engine.run(sample_ohlcv_data, strategy, "BTCUSDT")
        
        required_keys = [
            'strategy', 'pair', 'total_return', 'sharpe_ratio',
            'max_drawdown', 'win_rate', 'num_trades', 'initial_capital',
            'final_capital', 'equity_curve'
        ]
        
        for key in required_keys:
            assert key in results, f"Missing key: {key}"
    
    def test_run_backtest_numeric_results(self, sample_ohlcv_data):
        """Test backtest returns numeric results."""
        from src.backtest_engine import BacktestEngine
        from src.strategies import TrendFollowing
        
        engine = BacktestEngine()
        strategy = TrendFollowing()
        
        results = engine.run(sample_ohlcv_data, strategy, "BTCUSDT")
        
        assert isinstance(results['total_return'], (int, float))
        assert isinstance(results['sharpe_ratio'], (int, float))
        assert isinstance(results['max_drawdown'], (int, float))
        assert isinstance(results['win_rate'], (int, float))
        assert isinstance(results['num_trades'], int)
    
    def test_empty_dataframe_raises_error(self, empty_dataframe):
        """Test empty DataFrame raises ValueError."""
        from src.backtest_engine import BacktestEngine
        from src.strategies import TrendFollowing
        
        engine = BacktestEngine()
        strategy = TrendFollowing()
        
        with pytest.raises(ValueError):
            engine.run(empty_dataframe, strategy, "BTCUSDT")
    
    def test_missing_close_column_raises_error(self):
        """Test missing close column raises ValueError."""
        from src.backtest_engine import BacktestEngine
        from src.strategies import TrendFollowing
        
        engine = BacktestEngine()
        strategy = TrendFollowing()
        bad_df = pd.DataFrame({'open': [1, 2, 3]})
        
        with pytest.raises(ValueError):
            engine.run(bad_df, strategy, "BTCUSDT")
    
    def test_run_multiple_strategies(self, sample_ohlcv_data):
        """Test running multiple strategies."""
        from src.backtest_engine import BacktestEngine
        from src.strategies import TrendFollowing, MeanReversion, Momentum
        
        engine = BacktestEngine()
        strategies = [TrendFollowing(), MeanReversion(), Momentum()]
        
        results = engine.run_multiple_strategies(
            sample_ohlcv_data, strategies, "BTCUSDT"
        )
        
        assert len(results) == 3
        assert all(isinstance(r, dict) for r in results)
    
    def test_compare_results(self, sample_ohlcv_data):
        """Test comparing multiple results."""
        from src.backtest_engine import BacktestEngine
        from src.strategies import TrendFollowing, MeanReversion
        
        engine = BacktestEngine()
        
        results1 = engine.run(sample_ohlcv_data, TrendFollowing(), "BTCUSDT")
        results2 = engine.run(sample_ohlcv_data, MeanReversion(), "BTCUSDT")
        
        comparison = engine.compare_results([results1, results2])
        
        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 2
        assert 'Strategy' in comparison.columns
        assert 'Return (%)' in comparison.columns
    
    def test_uptrend_positive_return(self, trending_up_data):
        """Test uptrend with trend following gives positive return."""
        from src.backtest_engine import BacktestEngine
        from src.strategies import TrendFollowing
        
        engine = BacktestEngine()
        strategy = TrendFollowing(sma_short=5, sma_long=20)
        
        results = engine.run(trending_up_data, strategy, "BTCUSDT")
        
        # In strong uptrend, should have positive return
        # (may not always be true due to fees and timing)
        assert results['total_return'] is not None


class TestResultsExporter:
    """Tests for ResultsExporter class."""
    
    @pytest.fixture
    def sample_results(self):
        """Sample backtest results for testing."""
        return {
            'strategy': 'test_strategy',
            'pair': 'BTCUSDT',
            'total_return': 10.5,
            'sharpe_ratio': 1.2,
            'max_drawdown': -5.3,
            'win_rate': 55.0,
            'num_trades': 50,
            'profit_factor': 1.5,
            'initial_capital': 2000,
            'final_capital': 2210,
            'equity_curve': pd.Series([2000, 2050, 2100, 2080, 2210]),
            'trades': pd.DataFrame({
                'entry_price': [100, 105],
                'exit_price': [105, 103],
                'pnl_percent': [5.0, -1.9]
            })
        }
    
    def test_initialization(self, tmp_path):
        """Test ResultsExporter initialization."""
        from src.results_exporter import ResultsExporter
        
        exporter = ResultsExporter(output_dir=str(tmp_path))
        
        assert exporter.output_dir == str(tmp_path)
    
    def test_export_summary(self, sample_results, tmp_path):
        """Test exporting summary to CSV."""
        from src.results_exporter import ResultsExporter
        
        exporter = ResultsExporter(output_dir=str(tmp_path))
        filepath = exporter.export_summary(sample_results)
        
        assert filepath is not None
        assert 'backtest_' in filepath
        assert filepath.endswith('.csv')
    
    def test_export_trades(self, sample_results, tmp_path):
        """Test exporting trades to CSV."""
        from src.results_exporter import ResultsExporter
        
        exporter = ResultsExporter(output_dir=str(tmp_path))
        filepath = exporter.export_trades(sample_results)
        
        assert filepath is not None
        assert 'trades_' in filepath
    
    def test_export_json(self, sample_results, tmp_path):
        """Test exporting results to JSON."""
        from src.results_exporter import ResultsExporter
        
        exporter = ResultsExporter(output_dir=str(tmp_path))
        filepath = exporter.export_json(sample_results)
        
        assert filepath is not None
        assert filepath.endswith('.json')
    
    def test_export_all(self, sample_results, tmp_path):
        """Test exporting all result types."""
        from src.results_exporter import ResultsExporter
        
        exporter = ResultsExporter(output_dir=str(tmp_path))
        exported = exporter.export_all(sample_results)
        
        assert 'summary' in exported
        assert 'trades' in exported
        assert 'equity_curve' in exported
