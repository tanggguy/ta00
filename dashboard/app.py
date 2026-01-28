"""
Module: dashboard/app.py
Description: Main Streamlit dashboard application
Author: Trading Bot
Date: 2026-01-22
Version: 1.0

Multi-page Streamlit app for crypto trading analysis.
Pages are located in dashboard/pages/ folder.
"""

import streamlit as st

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Crypto Swing Trading",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for consistent styling across all pages
st.markdown("""
<style>
    /* Main styling */
    .main > div {
        padding-top: 2rem;
    }
    
    /* Metrics styling */
    [data-testid="stMetric"] {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0e1117;
    }
    
    /* Table styling */
    .dataframe {
        font-size: 0.85rem;
    }
    
    /* Success/Error messages */
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #0d3320;
        border: 1px solid #28a745;
    }
    
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #3d1a1a;
        border: 1px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main dashboard home page."""
    
    st.title("Crypto Swing Trading Dashboard")
    
    st.markdown("""
    Welcome to your trading analysis dashboard. Use the sidebar to navigate between pages.
    
    ---
    
    ### Available Pages
    
    | Page | Description | Status |
    |------|-------------|--------|
    | ** Backtest Results** | Analyze strategy performance, equity curves, exports | ✅ Active |
    | **Live Monitoring** | Real-time paper/live trading monitoring | ✅ Active |
    | **CSV Explorer** | Browse and filter trading data files | 🔜 Coming soon |
    | **Risk Analysis** | Correlation, drawdown analysis | 🔜 Phase 2 |
    
    ---
    
    ### Quick Start
    
    1. Navigate to **📈 Backtest Results** in the sidebar
    2. Select a strategy and trading pair
    3. Analyze performance metrics and charts
    4. Export reports as interactive HTML
    
    ---
    """)
    
    # Show available data summary
    st.subheader("📁 Data Status")
    
    from pathlib import Path
    import pandas as pd
    
    results_dir = Path("data/backtest_results")
    
    if results_dir.exists():
        backtest_files = list(results_dir.glob("backtest_*.csv"))
        equity_files = list(results_dir.glob("equity_*.csv"))
        trades_files = list(results_dir.glob("trades_*.csv"))
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Backtest Results", len(backtest_files))
        
        with col2:
            st.metric("Equity Curves", len(equity_files))
        
        with col3:
            st.metric("Trade Records", len(trades_files))
        
        # Show comparison if available
        comparison_files = list(results_dir.glob("comparison_*.csv"))
        if comparison_files:
            st.subheader("📊 Latest Comparison")
            latest_comparison = sorted(comparison_files)[-1]
            df = pd.read_csv(latest_comparison)
            st.dataframe(df, use_container_width=True)
    else:
        st.warning("No backtest data found. Run a backtest first!")
        st.code("python scripts/run_backtest.py", language="bash")


if __name__ == "__main__":
    main()
