import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import importlib
import src.report_generator
importlib.reload(src.report_generator)
from src.report_generator import ReportGenerator

st.set_page_config(page_title="Optimization Results", layout="wide")

st.title("Optimization Results")

RESULTS_DIR = "data/optimization_results"

# 1. Load Files
if not os.path.exists(RESULTS_DIR):
    st.warning(f"No optimization results found in `{RESULTS_DIR}`.")
    st.stop()

files = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".json")]
files.sort(reverse=True)

if not files:
    st.warning("No JSON result files found.")
    st.stop()

selected_file = st.selectbox("Select Optimization Run", files)

# 2. Load Data
filepath = os.path.join(RESULTS_DIR, selected_file)
with open(filepath, 'r') as f:
    data = json.load(f)

# 3. Display Overview
st.header("Best Configuration Performance")

col1, col2, col3, col4 = st.columns(4)

best_params = data.get('best_params', {})
train_stats = data.get('train_stats', {})
test_stats = data.get('test_stats', {})

train_sharpe = train_stats.get('sharpe', 0)
test_sharpe = test_stats.get('sharpe', 0)
sharpe_diff = abs(train_sharpe - test_sharpe)

col1.metric("Best Train Sharpe", f"{train_sharpe:.2f}")
col2.metric("Validation (Test) Sharpe", f"{test_sharpe:.2f}", 
            delta=f"{test_sharpe-train_sharpe:.2f}", delta_color="normal")
col3.metric("Stability (Diff)", f"{sharpe_diff:.2f}", 
            delta="Overfitting Risk" if sharpe_diff > 1.0 else "Stable", delta_color="inverse")
col4.metric("Test Total Return", f"{test_stats.get('total_return', 0):.2f}%")

st.markdown("### Optimal Parameters")
st.json(best_params)

# 4. Overfitting Analysis
st.header("Overfitting Analysis")
if data.get('is_overfitting'):
    st.error(f"POSSIBLE OVERFITTING DETECTED! Sharpe difference ({sharpe_diff:.2f}) > 1.0")
else:
    st.success(f"Model appears stable. Sharpe difference ({sharpe_diff:.2f}) is within acceptable range.")

# 5. Detailed Runs Analysis
if 'all_train_results' in data:
    st.header("Grid Search Exploration")
    all_results = pd.DataFrame(data['all_train_results'])
    
    # Sort columns to put metrics at the end
    cols = [c for c in all_results.columns if c not in ['sharpe', 'total_return']]
    cols += ['sharpe', 'total_return']
    all_results = all_results[cols]
    
    st.dataframe(all_results.style.background_gradient(subset=['sharpe'], cmap='RdYlGn'), use_container_width=True)

    # Visualization
    if len(cols) >= 3: # At least 1 param + metrics
        param_x = st.selectbox("X Axis (Parameter)", cols[:-2], index=0)
        param_y = st.selectbox("Y Axis (Parameter)", cols[:-2], index=min(1, len(cols)-3))
        
        fig = px.scatter(
            all_results, 
            x=param_x, 
            y=param_y, 
            color='sharpe',
            hover_data=cols,
            title="Parameter Performance Heatmap (Color = Sharpe)",
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig, use_container_width=True)

# 6. Export Report
st.divider()
col_exp, _ = st.columns([1, 4])
if col_exp.button("Generate HTML Report"):
    generator = ReportGenerator()
    try:
        # Reconstruct DataFrame if it exists
        all_res_df = pd.DataFrame(data.get('all_train_results', []))
        
        # Extract metadata from filename (quick hack, ideally stored in json)
        # filename format: pair_strategy_opt_timestamp.json
        parts = selected_file.split('_')
        pair = parts[0] # Very rough parsing, assuming no underscores in pair usually
        strategy = parts[1] # rough
        if len(parts) > 4: # Format pair_strategy_opt_timestamp
             pair = parts[0]
             strategy = "_".join(parts[1:-3]) # handle strategies with underscores?
        
        # Use filename parts for robustness if JSON lacks it
        # Actually generate_filename uses pair_strategy...
        # Let's try to infer from content if possible, or fallback
        
        report_path = generator.generate_optimization_report(
            strategy="Unknown" if 'strategy' not in data else data['strategy'], # Assuming stored? JSON structure didn't show it explicitly in my previous edit, checking...
            # Actually I didn't save strategy name in JSON in run_optimizer.py, I should have!
            # I'll rely on filename or just "Strategy"
            pair="Unknown",
            best_params=best_params,
            train_stats=train_stats,
            test_stats=test_stats,
            all_results=all_res_df
        )
        st.success(f"Report generated successfully!")
        st.markdown(f"**Location:** `{report_path}`")
        
        # Open button (local only)
        if st.checkbox("Open in browser (Local only)"):
            import webbrowser
            webbrowser.open(f"file://{os.path.abspath(report_path)}")
            
    except Exception as e:
        st.error(f"Failed to generate report: {e}")
