import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Strategy Comparison",
    page_icon="",
    layout="wide"
)

st.title("Strategy Comparison")

# --- Data Loading ---
results_dir = Path("data/backtest_results")

if not results_dir.exists():
    st.error("No data directory found.")
    st.stop()

# Find comparison files
comparison_files = list(results_dir.glob("comparison_*.csv"))

if not comparison_files:
    st.warning("No comparison files found. Run multiple backtests to generate a comparison.")
    st.stop()

# Sort files by internal creation date (filename usually has timestamp, but we can also use os.path.getmtime)
# Assuming filename format: comparison_YYYYMMDD_HHMMSS.csv
comparison_files = sorted(comparison_files, reverse=True)
file_options = {f.name: f for f in comparison_files}

# Sidebar Selection
st.sidebar.header("Data Source")
selected_file_name = st.sidebar.selectbox(
    "Select Comparison File",
    options=list(file_options.keys()),
    index=0
)
selected_file_path = file_options[selected_file_name]

# Load Data
try:
    df = pd.read_csv(selected_file_path)
except Exception as e:
    st.error(f"Error loading file: {e}")
    st.stop()

# --- Main Content ---

# Top level metrics overview
st.markdown("### Top Performers")
col1, col2, col3 = st.columns(3)

if not df.empty:
    best_return = df.loc[df['Return (%)'].idxmax()]
    best_sharpe = df.loc[df['Sharpe'].idxmax()]
    lowest_dd = df.loc[df['Max DD (%)'].idxmax()] # Max DD is negative, so max value is closest to 0 (smallest drawdown)

    with col1:
        st.metric("Highest Return", f"{best_return['Return (%)']:.2f}%", f"{best_return['Strategy']} ({best_return['Pair']})")
    
    with col2:
        st.metric("Best Sharpe", f"{best_sharpe['Sharpe']:.2f}", f"{best_sharpe['Strategy']} ({best_sharpe['Pair']})")
        
    with col3:
        st.metric("Lowest Drawdown", f"{lowest_dd['Max DD (%)']:.2f}%", f"{lowest_dd['Strategy']} ({lowest_dd['Pair']})")

st.divider()

# --- Visualizations ---

col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.subheader("Performance Comparison")
    # Grouped Bar Chart for Return
    fig_perf = px.bar(
        df, 
        x="Pair", 
        y="Return (%)", 
        color="Strategy", 
        barmode="group",
        title="Total Return by Pair & Strategy",
        hover_data=["Win Rate (%)", "Trades", "Max DD (%)"]
    )
    st.plotly_chart(fig_perf, use_container_width=True)

with col_chart2:
    st.subheader("Risk vs Reward")
    # Scatter Plot: Return vs Max DD
    # Max DD is usually negative. We might want to invert axis or just plot as is.
    fig_risk = px.scatter(
        df, 
        x="Max DD (%)", 
        y="Return (%)", 
        color="Strategy", 
        symbol="Pair",
        size="Trades", # Bubble size by number of trades
        title="Return vs Max Drawdown (Size = No. of Trades)",
        hover_data=["Sharpe", "Final Capital"]
    )
    # Reverse X axis if users prefer seeing 0 on the right (since drawdowns are negative)
    # fig_risk.update_xaxes(autorange="reversed") 
    st.plotly_chart(fig_risk, use_container_width=True)


# --- Detailed Metrics Comparison ---
st.subheader("Detailed Metrics Analysis")

metric_to_plot = st.selectbox(
    "Select Metric to Compare",
    options=["Sharpe", "Win Rate (%)", "Trades", "Final Capital"],
    index=0
)

fig_metric = px.bar(
    df,
    x="Strategy",
    y=metric_to_plot,
    color="Pair",
    barmode="group",
    title=f"{metric_to_plot} by Strategy & Pair"
)
st.plotly_chart(fig_metric, use_container_width=True)

# --- Raw Data ---
with st.expander("View Raw Data", expanded=True):
    st.dataframe(
        df.style.highlight_max(axis=0, color='#1f77b4'), 
        use_container_width=True
    )
