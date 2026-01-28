"""
Module: dashboard/pages/06_Live_Monitoring.py
Description: Real-time paper trading monitoring with sentiment and trends status
Author: Trading Bot
Date: 2026-01-28
Version: 1.0

Displays:
- Current sentiment and Google Trends status
- Equity curve from paper trading
- Performance metrics (Sharpe, Drawdown, Win Rate)
- Trade history with CSV download
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime, timedelta
import sys
import time

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Page config
st.set_page_config(
    page_title="Live Monitoring",
    page_icon="",
    layout="wide"
)

# Auto-refresh every 60 seconds
st.markdown(
    """
    <meta http-equiv="refresh" content="60">
    """,
    unsafe_allow_html=True
)


def load_config():
    """Load configuration from settings.json."""
    import json
    config_path = PROJECT_ROOT / "config" / "settings.json"
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    
    return {"initial_capital": 2000, "currency": "USDT"}


def load_all_trades():
    """
    Load all paper trading trades from CSV files.
    
    Returns:
        DataFrame with all trades sorted by timestamp
    """
    trades_dir = PROJECT_ROOT / "data" / "live_trading"
    
    if not trades_dir.exists():
        return pd.DataFrame()
    
    all_trades = []
    
    for csv_file in trades_dir.glob("*_trades_*.csv"):
        try:
            df = pd.read_csv(csv_file)
            all_trades.append(df)
        except Exception as e:
            st.warning(f"Error reading {csv_file.name}: {e}")
    
    if not all_trades:
        return pd.DataFrame()
    
    # Combine all trades
    combined = pd.concat(all_trades, ignore_index=True)
    
    # Parse timestamp and sort
    if 'timestamp' in combined.columns:
        combined['timestamp'] = pd.to_datetime(combined['timestamp'])
        combined = combined.sort_values('timestamp')
    
    return combined


def calculate_equity_curve(trades_df: pd.DataFrame, initial_capital: float):
    """
    Calculate equity curve from trades.
    
    Args:
        trades_df: DataFrame with trades
        initial_capital: Starting capital
    
    Returns:
        DataFrame with equity curve data
    """
    if trades_df.empty:
        # Return initial equity point
        return pd.DataFrame({
            'timestamp': [datetime.now()],
            'equity': [initial_capital],
            'pnl_cumulative': [0.0],
            'drawdown_pct': [0.0]
        })
    
    # Filter closed trades only
    closed_trades = trades_df[trades_df['status'] == 'CLOSED'].copy()
    
    if closed_trades.empty:
        return pd.DataFrame({
            'timestamp': [datetime.now()],
            'equity': [initial_capital],
            'pnl_cumulative': [0.0],
            'drawdown_pct': [0.0]
        })
    
    # Calculate cumulative PnL
    closed_trades = closed_trades.sort_values('timestamp')
    closed_trades['pnl_cumulative'] = closed_trades['pnl_usdt'].cumsum()
    closed_trades['equity'] = initial_capital + closed_trades['pnl_cumulative']
    
    # Calculate drawdown
    closed_trades['peak'] = closed_trades['equity'].cummax()
    closed_trades['drawdown_pct'] = (
        (closed_trades['equity'] - closed_trades['peak']) / closed_trades['peak'] * 100
    )
    
    return closed_trades[['timestamp', 'equity', 'pnl_cumulative', 'drawdown_pct']]


def calculate_metrics(trades_df: pd.DataFrame, initial_capital: float):
    """
    Calculate performance metrics.
    
    Returns:
        dict with metrics
    """
    metrics = {
        'total_trades': 0,
        'winning_trades': 0,
        'losing_trades': 0,
        'win_rate': 0.0,
        'total_pnl': 0.0,
        'current_equity': initial_capital,
        'max_drawdown': 0.0,
        'sharpe_ratio': 0.0,
        'profit_factor': 0.0,
        'avg_win': 0.0,
        'avg_loss': 0.0
    }
    
    if trades_df.empty:
        return metrics
    
    closed_trades = trades_df[trades_df['status'] == 'CLOSED'].copy()
    
    if closed_trades.empty:
        return metrics
    
    # Basic counts
    metrics['total_trades'] = len(closed_trades)
    
    if 'pnl_usdt' in closed_trades.columns:
        wins = closed_trades[closed_trades['pnl_usdt'] > 0]
        losses = closed_trades[closed_trades['pnl_usdt'] < 0]
        
        metrics['winning_trades'] = len(wins)
        metrics['losing_trades'] = len(losses)
        metrics['win_rate'] = (metrics['winning_trades'] / metrics['total_trades'] * 100) if metrics['total_trades'] > 0 else 0
        
        metrics['total_pnl'] = closed_trades['pnl_usdt'].sum()
        metrics['current_equity'] = initial_capital + metrics['total_pnl']
        
        # Average win/loss
        metrics['avg_win'] = wins['pnl_usdt'].mean() if len(wins) > 0 else 0
        metrics['avg_loss'] = losses['pnl_usdt'].mean() if len(losses) > 0 else 0
        
        # Profit factor
        total_wins = wins['pnl_usdt'].sum() if len(wins) > 0 else 0
        total_losses = abs(losses['pnl_usdt'].sum()) if len(losses) > 0 else 1
        metrics['profit_factor'] = total_wins / total_losses if total_losses > 0 else total_wins
        
        # Calculate equity curve for drawdown
        equity_curve = calculate_equity_curve(trades_df, initial_capital)
        if not equity_curve.empty:
            metrics['max_drawdown'] = equity_curve['drawdown_pct'].min()
        
        # Sharpe Ratio (simplified)
        if len(closed_trades) > 1:
            returns = closed_trades['pnl_usdt'] / initial_capital
            if returns.std() > 0:
                # Annualized assuming 4h timeframe = 6 trades per day = ~1500 per year
                metrics['sharpe_ratio'] = (returns.mean() / returns.std()) * np.sqrt(365)
    
    return metrics


def get_sentiment_status():
    """
    Get current sentiment status from CryptoPanic.
    
    Returns:
        dict with sentiment data
    """
    try:
        from src.data_sentiment import CryptoPanicClient
        client = CryptoPanicClient()
        sentiment = client.get_sentiment("BTC")
        allowed, reason = client.is_buy_allowed("BTC")
        
        return {
            'bullish_pct': sentiment.get('bullish_pct', 0),
            'bearish_pct': sentiment.get('bearish_pct', 0),
            'can_buy': allowed,
            'reason': reason,
            'error': sentiment.get('error', False),
            'timestamp': sentiment.get('timestamp', '')
        }
    except Exception as e:
        return {
            'bullish_pct': 0,
            'bearish_pct': 0,
            'can_buy': False,
            'reason': f"Error: {str(e)}",
            'error': True,
            'timestamp': ''
        }


def get_trends_status():
    """
    Get current Google Trends status.
    
    Returns:
        dict with trends data
    """
    try:
        from src.data_trends import GoogleTrendsAnalyzer
        analyzer = GoogleTrendsAnalyzer()
        data = analyzer.get_interest_data()
        confirmed, reason = analyzer.is_momentum_confirmed()
        
        return {
            'current_interest': data.get('current_interest', 0),
            'avg_7d': data.get('avg_7d', 0),
            'surge_ratio_pct': data.get('surge_ratio_pct', 0),
            'is_surging': data.get('is_surging', False),
            'confirmed': confirmed,
            'reason': reason,
            'error': data.get('error', False)
        }
    except Exception as e:
        return {
            'current_interest': 0,
            'avg_7d': 0,
            'surge_ratio_pct': 0,
            'is_surging': False,
            'confirmed': False,
            'reason': f"Error: {str(e)}",
            'error': True
        }


def create_equity_chart(equity_df: pd.DataFrame):
    """Create equity curve chart with drawdown."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=("Courbe d'Équité", "Drawdown %")
    )
    
    # Equity curve
    fig.add_trace(
        go.Scatter(
            x=equity_df['timestamp'],
            y=equity_df['equity'],
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
            x=equity_df['timestamp'],
            y=equity_df['drawdown_pct'],
            name='Drawdown',
            line=dict(color='#ff6b6b', width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 107, 107, 0.2)'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=500,
        showlegend=True,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    
    return fig


def display_sentiment_status(sentiment: dict):
    """Display sentiment status in a card."""
    st.subheader("Sentiment CryptoPanic")
    
    if sentiment['error']:
        st.error("Sentiment indisponible")
        st.caption(sentiment['reason'])
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bearish percentage with color coding
        bearish_pct = sentiment['bearish_pct']
        delta_color = "inverse" if bearish_pct > 70 else "normal"
        st.metric(
            "% Bearish",
            f"{bearish_pct:.1f}%",
            delta=f"{bearish_pct - 50:.1f}% vs neutral",
            delta_color=delta_color
        )
    
    with col2:
        # Buy status
        if sentiment['can_buy']:
            st.success("BUY Autorisé")
        else:
            st.error("BUY Bloqué")
    
    st.caption(sentiment['reason'])


def display_trends_status(trends: dict):
    """Display Google Trends status in a card."""
    st.subheader("Google Trends (Bitcoin)")
    
    if trends['error']:
        st.warning("Trends indisponible")
        st.caption(trends['reason'])
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        surge_pct = trends['surge_ratio_pct']
        st.metric(
            "Ratio Surge",
            f"{surge_pct:.0f}%",
            delta=f"{surge_pct - 100:.0f}% vs moyenne",
            delta_color="normal"
        )
    
    with col2:
        if trends['is_surging']:
            st.success("Surge détecté!")
        else:
            st.info("Pas de surge")
    
    st.caption(f"Actuel: {trends['current_interest']} | Moy 7j: {trends['avg_7d']:.1f}")


def main():
    """Main page content."""
    st.title("Live Monitoring - Paper Trading")
    
    # Last refresh time
    st.caption(f"Dernière mise à jour: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (auto-refresh 60s)")
    
    # Load config and data
    config = load_config()
    initial_capital = config.get('initial_capital', 2000)
    currency = config.get('currency', 'USDT')
    
    trades_df = load_all_trades()
    
    # ============================================
    # Section 1: Market Context Status
    # ============================================
    st.markdown("---")
    st.header("Contexte Marché")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container():
            sentiment = get_sentiment_status()
            display_sentiment_status(sentiment)
    
    with col2:
        with st.container():
            trends = get_trends_status()
            display_trends_status(trends)
    
    # ============================================
    # Section 2: Performance Metrics
    # ============================================
    st.markdown("---")
    st.header("Performance")
    
    metrics = calculate_metrics(trades_df, initial_capital)
    
    # Main metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        pnl_delta = f"+{metrics['total_pnl']:.2f}" if metrics['total_pnl'] >= 0 else f"{metrics['total_pnl']:.2f}"
        st.metric(
            f"Équité ({currency})",
            f"{metrics['current_equity']:,.2f}",
            delta=pnl_delta,
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            "Max Drawdown",
            f"{metrics['max_drawdown']:.2f}%",
            delta=None
        )
    
    with col3:
        st.metric(
            "Sharpe Ratio",
            f"{metrics['sharpe_ratio']:.2f}",
            delta=None
        )
    
    with col4:
        st.metric(
            "Win Rate",
            f"{metrics['win_rate']:.1f}%",
            delta=f"{metrics['winning_trades']}/{metrics['total_trades']} trades"
        )
    
    # Secondary metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total PnL", f"{metrics['total_pnl']:.2f} {currency}")
    
    with col2:
        st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
    
    with col3:
        st.metric("Avg Win", f"{metrics['avg_win']:.2f} {currency}")
    
    with col4:
        st.metric("Avg Loss", f"{metrics['avg_loss']:.2f} {currency}")
    
    # ============================================
    # Section 3: Equity Curve
    # ============================================
    st.markdown("---")
    st.header("Courbe d'Équité")
    
    equity_df = calculate_equity_curve(trades_df, initial_capital)
    
    if len(equity_df) > 1:
        fig = create_equity_chart(equity_df)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📭 Pas encore de trades fermés. Lancez le bot en paper trading pour commencer!")
        st.code("python -m src.live_bot --pairs BTC/USDT --once", language="bash")
    
    # ============================================
    # Section 4: Trade History
    # ============================================
    st.markdown("---")
    st.header("Historique des Trades")
    
    if trades_df.empty:
        st.info("Aucun trade enregistré. Le bot doit d'abord être exécuté.")
    else:
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pairs = ['Tous'] + list(trades_df['pair'].unique()) if 'pair' in trades_df.columns else ['Tous']
            selected_pair = st.selectbox("Paire", pairs)
        
        with col2:
            statuses = ['Tous', 'OPEN', 'CLOSED']
            selected_status = st.selectbox("Status", statuses)
        
        with col3:
            signals = ['Tous', 'BUY', 'SELL']
            selected_signal = st.selectbox("Signal", signals)
        
        # Apply filters
        filtered_df = trades_df.copy()
        
        if selected_pair != 'Tous':
            filtered_df = filtered_df[filtered_df['pair'] == selected_pair]
        
        if selected_status != 'Tous':
            filtered_df = filtered_df[filtered_df['status'] == selected_status]
        
        if selected_signal != 'Tous':
            filtered_df = filtered_df[filtered_df['signal'] == selected_signal]
        
        # Display table
        st.dataframe(
            filtered_df.sort_values('timestamp', ascending=False),
            use_container_width=True,
            height=400
        )
        
        # Download button
        st.download_button(
            label="Télécharger CSV",
            data=filtered_df.to_csv(index=False),
            file_name=f"paper_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # ============================================
    # Section 5: Open Positions
    # ============================================
    if not trades_df.empty and 'status' in trades_df.columns:
        open_positions = trades_df[trades_df['status'] == 'OPEN']
        
        if not open_positions.empty:
            st.markdown("---")
            st.header("Positions Ouvertes")
            
            st.dataframe(open_positions, use_container_width=True)


if __name__ == "__main__":
    main()
