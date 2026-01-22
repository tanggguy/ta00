"""
Module: dashboard/pages/01_Backtest_Results.py
Description: Backtest analysis page with equity curves, stats and HTML export
Author: Trading Bot
Date: 2026-01-22
Version: 1.0
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime
import sys
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.report_generator import ReportGenerator


def load_known_strategies_and_pairs() -> tuple[list[str], list[str]]:
    """
    Load known strategies and pairs from config files.
    
    Returns:
        (strategies, pairs) where pairs are formatted for filenames (e.g., BTCUSDT)
    """
    config_dir = PROJECT_ROOT / "config"
    
    # Load strategies
    strategies = []
    strategies_file = config_dir / "strategies.json"
    if strategies_file.exists():
        with open(strategies_file, 'r') as f:
            strategies_config = json.load(f)
            strategies = list(strategies_config.keys())
    
    # Load pairs (convert BTC/USDT -> BTCUSDT for filename matching)
    pairs = []
    pairs_file = config_dir / "pairs.json"
    if pairs_file.exists():
        with open(pairs_file, 'r') as f:
            pairs_config = json.load(f)
            pairs = [p.replace("/", "") for p in pairs_config.get("pairs", [])]
    
    return strategies, pairs


def parse_backtest_filename(filename: str, known_strategies: list[str], known_pairs: list[str]) -> tuple[str, str, str] | None:
    """
    Intelligently parse backtest filename using known strategies and pairs.
    
    Args:
        filename: Stem of the file (without extension), e.g., "backtest_mean_reversion_BTCUSDT_20260122_193851"
        known_strategies: List of known strategy names
        known_pairs: List of known pairs (formatted for filenames)
    
    Returns:
        (strategy, pair, timestamp) or None if parsing fails
    """
    # Remove "backtest_" prefix
    if not filename.startswith("backtest_"):
        return None
    
    remaining = filename[9:]  # After "backtest_"
    
    # Try to match known strategies (sorted by length descending to match longest first)
    matched_strategy = None
    for strategy in sorted(known_strategies, key=len, reverse=True):
        if remaining.startswith(strategy + "_"):
            matched_strategy = strategy
            remaining = remaining[len(strategy) + 1:]  # +1 for the underscore
            break
    
    if not matched_strategy:
        return None
    
    # Try to match known pairs
    matched_pair = None
    for pair in sorted(known_pairs, key=len, reverse=True):
        if remaining.startswith(pair + "_"):
            matched_pair = pair
            remaining = remaining[len(pair) + 1:]
            break
    
    if not matched_pair:
        return None
    
    # Remaining is the timestamp
    timestamp = remaining
    
    return matched_strategy, matched_pair, timestamp


def load_available_backtests(results_dir: Path) -> dict:
    """
    Scan directory and group files by strategy/pair.
    Uses intelligent parsing based on known strategies and pairs from config.
    
    Returns dict: {(strategy, pair, timestamp): {'backtest': path, 'equity': path, 'trades': path}}
    """
    backtests = {}
    
    # Load known strategies and pairs from config
    known_strategies, known_pairs = load_known_strategies_and_pairs()
    
    for f in results_dir.glob("backtest_*.csv"):
        parsed = parse_backtest_filename(f.stem, known_strategies, known_pairs)
        
        if parsed:
            strategy, pair, timestamp = parsed
            key = (strategy, pair, timestamp)
            
            if key not in backtests:
                backtests[key] = {}
            backtests[key]['backtest'] = f
            backtests[key]['timestamp'] = timestamp
    
    # Match equity and trades files
    for key in backtests:
        strategy, pair, timestamp = key
        
        equity_pattern = f"equity_{strategy}_{pair}_{timestamp}.csv"
        trades_pattern = f"trades_{strategy}_{pair}_{timestamp}.csv"
        
        equity_file = results_dir / equity_pattern
        trades_file = results_dir / trades_pattern
        
        if equity_file.exists():
            backtests[key]['equity'] = equity_file
        if trades_file.exists():
            backtests[key]['trades'] = trades_file
    
    return backtests


def load_backtest_data(files: dict) -> dict:
    """Load all CSV files for a backtest."""
    data = {}
    
    if 'backtest' in files:
        data['stats'] = pd.read_csv(files['backtest'])
    
    if 'equity' in files:
        df = pd.read_csv(files['equity'])
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        data['equity'] = df
    
    if 'trades' in files:
        data['trades'] = pd.read_csv(files['trades'])
    
    return data


def create_equity_chart(equity_df: pd.DataFrame) -> go.Figure:
    """Create equity curve chart with drawdown subplot."""
    
    # Calculate drawdown
    equity = equity_df['portfolio_value'] if 'portfolio_value' in equity_df.columns else equity_df.iloc[:, 0]
    rolling_max = equity.expanding().max()
    drawdown = (equity - rolling_max) / rolling_max * 100
    
    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=("Portfolio Value", "Drawdown (%)")
    )
    
    # Equity curve
    fig.add_trace(
        go.Scatter(
            x=equity_df.index,
            y=equity,
            mode='lines',
            name='Equity',
            line=dict(color='#00d4aa', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 212, 170, 0.1)'
        ),
        row=1, col=1
    )
    
    # Drawdown
    fig.add_trace(
        go.Scatter(
            x=equity_df.index,
            y=drawdown,
            mode='lines',
            name='Drawdown',
            line=dict(color='#ff6b6b', width=1.5),
            fill='tozeroy',
            fillcolor='rgba(255, 107, 107, 0.2)'
        ),
        row=2, col=1
    )
    
    # Layout
    fig.update_layout(
        height=600,
        template='plotly_dark',
        showlegend=False,
        hovermode='x unified',
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="DD %", row=2, col=1)
    
    return fig


def display_stats(stats_df: pd.DataFrame):
    """Display statistics as metrics."""
    
    if stats_df.empty:
        st.warning("No statistics available")
        return
    
    row = stats_df.iloc[0]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_return = row.get('total_return_pct', 0)
        delta_color = "normal" if total_return >= 0 else "inverse"
        st.metric(
            "Total Return",
            f"{total_return:.2f}%",
            delta=f"${row.get('final_capital', 0) - row.get('initial_capital', 0):.2f}",
            delta_color=delta_color
        )
    
    with col2:
        st.metric(
            "Sharpe Ratio",
            f"{row.get('sharpe_ratio', 0):.2f}"
        )
    
    with col3:
        max_dd = row.get('max_drawdown_pct', 0)
        st.metric(
            "Max Drawdown",
            f"{max_dd:.2f}%"
        )
    
    with col4:
        st.metric(
            "Win Rate",
            f"{row.get('win_rate_pct', 0):.1f}%"
        )
    
    # Second row
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric(
            "Trades",
            f"{int(row.get('num_trades', 0))}"
        )
    
    with col6:
        st.metric(
            "Profit Factor",
            f"{row.get('profit_factor', 0):.2f}"
        )
    
    with col7:
        st.metric(
            "Initial Capital",
            f"${row.get('initial_capital', 0):,.2f}"
        )
    
    with col8:
        st.metric(
            "Final Capital",
            f"${row.get('final_capital', 0):,.2f}"
        )


def display_trades_table(trades_df: pd.DataFrame):
    """Display trades in an interactive table."""
    
    if trades_df.empty:
        st.info("No trades recorded")
        return
    
    # Select key columns
    display_cols = [
        'Entry Timestamp', 'Exit Timestamp', 'Avg Entry Price', 
        'Avg Exit Price', 'PnL', 'Return', 'Direction', 'Status'
    ]
    
    available_cols = [c for c in display_cols if c in trades_df.columns]
    
    if available_cols:
        display_df = trades_df[available_cols].copy()
        
        # Format columns
        if 'PnL' in display_df.columns:
            display_df['PnL'] = display_df['PnL'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "-")
        if 'Return' in display_df.columns:
            display_df['Return'] = display_df['Return'].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "-")
        if 'Avg Entry Price' in display_df.columns:
            display_df['Avg Entry Price'] = display_df['Avg Entry Price'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "-")
        if 'Avg Exit Price' in display_df.columns:
            display_df['Avg Exit Price'] = display_df['Avg Exit Price'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "-")
        
        st.dataframe(display_df, use_container_width=True, height=400)
    else:
        st.dataframe(trades_df, use_container_width=True, height=400)


def main():
    """Main page content."""
    
    st.title("Backtest Results")
    st.markdown("Analyze strategy performance, view equity curves, and export reports.")
    
    # Directory setup
    results_dir = Path("data/backtest_results")
    
    if not results_dir.exists():
        st.error("No backtest results directory found!")
        st.info("Run a backtest first: `python scripts/run_backtest.py`")
        return
    
    # Load available backtests
    backtests = load_available_backtests(results_dir)
    
    if not backtests:
        st.warning("No backtest results found in data/backtest_results/")
        return
    
    # Sidebar selectors
    st.sidebar.header("Select Backtest")
    
    # Get unique strategies and pairs
    strategies = sorted(set(k[0] for k in backtests.keys()))
    pairs = sorted(set(k[1] for k in backtests.keys()))
    
    selected_strategy = st.sidebar.selectbox("Strategy", strategies)
    selected_pair = st.sidebar.selectbox("Pair", pairs)
    
    # Filter matching backtests
    matching = {k: v for k, v in backtests.items() 
                if k[0] == selected_strategy and k[1] == selected_pair}
    
    if not matching:
        st.warning(f"No backtest found for {selected_strategy} / {selected_pair}")
        return
    
    # Select timestamp if multiple
    timestamps = sorted(matching.keys(), key=lambda x: x[2], reverse=True)
    selected_key = timestamps[0]  # Most recent
    
    if len(timestamps) > 1:
        timestamp_options = [k[2] for k in timestamps]
        selected_timestamp = st.sidebar.selectbox("Timestamp", timestamp_options)
        selected_key = next(k for k in timestamps if k[2] == selected_timestamp)
    
    # Load data
    files = matching[selected_key]
    data = load_backtest_data(files)
    
    # Display header
    st.markdown(f"""
    **Strategy:** `{selected_strategy}` | **Pair:** `{selected_pair}` | 
    **Date:** `{selected_key[2]}`
    """)
    
    st.divider()
    
    # Statistics section
    st.subheader("Performance Metrics")
    if 'stats' in data:
        display_stats(data['stats'])
    
    st.divider()
    
    # Charts section
    st.subheader("Equity Curve & Drawdown")
    if 'equity' in data:
        fig = create_equity_chart(data['equity'])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No equity curve data available")
    
    st.divider()
    
    # Trades section
    st.subheader("Trades History")
    if 'trades' in data:
        display_trades_table(data['trades'])
    else:
        st.info("No trades data available")
    
    st.divider()
    
    # Export section
    st.subheader("Export Report")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button("Generate HTML Report", type="primary", use_container_width=True):
            with st.spinner("Generating report..."):
                try:
                    generator = ReportGenerator()
                    
                    report_path = generator.generate_html_report(
                        strategy=selected_strategy,
                        pair=selected_pair,
                        stats_df=data.get('stats'),
                        equity_df=data.get('equity'),
                        trades_df=data.get('trades')
                    )
                    
                    st.success(f"Report generated!")
                    st.code(report_path)
                    
                    # Offer download
                    with open(report_path, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    
                    st.download_button(
                        label="Download HTML",
                        data=html_content,
                        file_name=Path(report_path).name,
                        mime="text/html"
                    )
                    
                except Exception as e:
                    st.error(f"Error generating report: {e}")
    
    with col2:
        st.info("The HTML report contains interactive charts and can be opened in any browser.")


if __name__ == "__main__":
    main()
