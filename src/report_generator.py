"""
Module: src/report_generator.py
Description: Generate standalone HTML reports for backtest analysis
Author: Trading Bot
Date: 2026-01-22
Version: 1.0
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

from src.config_loader import get_config
from src.logger import get_logger
from src.utils import ensure_directory

logger = get_logger(__name__)


class ReportGenerator:
    """
    Generate standalone HTML reports for backtest analysis.
    
    Reports include:
    - Performance statistics table
    - Interactive equity curve chart
    - Drawdown visualization
    - Trades history table
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize Report Generator.
        
        Args:
            output_dir (Optional[str]): Output directory for reports
        """
        self.output_dir = output_dir or "data/reports"
        ensure_directory(self.output_dir)
        
        logger.info(f"ReportGenerator initialized: output_dir={self.output_dir}")
    
    def _create_equity_chart(self, equity_df: pd.DataFrame) -> go.Figure:
        """Create equity curve with drawdown subplot."""
        
        if equity_df is None or equity_df.empty:
            return None
        
        # Get equity column
        equity = equity_df['portfolio_value'] if 'portfolio_value' in equity_df.columns else equity_df.iloc[:, 0]
        
        # Calculate drawdown
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max * 100
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3],
            subplot_titles=("Portfolio Value ($)", "Drawdown (%)")
        )
        
        # Equity curve
        fig.add_trace(
            go.Scatter(
                x=equity_df.index if isinstance(equity_df.index, pd.DatetimeIndex) else list(range(len(equity))),
                y=equity,
                mode='lines',
                name='Equity',
                line=dict(color='#00d4aa', width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 212, 170, 0.15)'
            ),
            row=1, col=1
        )
        
        # Drawdown
        fig.add_trace(
            go.Scatter(
                x=equity_df.index if isinstance(equity_df.index, pd.DatetimeIndex) else list(range(len(drawdown))),
                y=drawdown,
                mode='lines',
                name='Drawdown',
                line=dict(color='#ff6b6b', width=1.5),
                fill='tozeroy',
                fillcolor='rgba(255, 107, 107, 0.2)'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=500,
            template='plotly_dark',
            showlegend=False,
            hovermode='x unified',
            margin=dict(l=60, r=40, t=60, b=40),
            paper_bgcolor='#1a1a2e',
            plot_bgcolor='#16213e'
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1, gridcolor='#2d3a4f')
        fig.update_yaxes(title_text="$", row=1, col=1, gridcolor='#2d3a4f')
        fig.update_yaxes(title_text="%", row=2, col=1, gridcolor='#2d3a4f')
        
        return fig
    
    def _generate_stats_html(self, stats_df: pd.DataFrame, strategy: str, pair: str) -> str:
        """Generate HTML for statistics section."""
        
        if stats_df is None or stats_df.empty:
            return "<p>No statistics available</p>"
        
        row = stats_df.iloc[0]
        
        total_return = row.get('total_return_pct', 0)
        return_class = 'positive' if total_return >= 0 else 'negative'
        
        html = f"""
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Total Return</div>
                <div class="stat-value {return_class}">{total_return:.2f}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Sharpe Ratio</div>
                <div class="stat-value">{row.get('sharpe_ratio', 0):.2f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Max Drawdown</div>
                <div class="stat-value negative">{row.get('max_drawdown_pct', 0):.2f}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Win Rate</div>
                <div class="stat-value">{row.get('win_rate_pct', 0):.1f}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Number of Trades</div>
                <div class="stat-value">{int(row.get('num_trades', 0))}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Profit Factor</div>
                <div class="stat-value">{row.get('profit_factor', 0):.2f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Initial Capital</div>
                <div class="stat-value">${row.get('initial_capital', 0):,.2f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Final Capital</div>
                <div class="stat-value">${row.get('final_capital', 0):,.2f}</div>
            </div>
        </div>
        """
        
        return html
    
    def _generate_trades_html(self, trades_df: pd.DataFrame) -> str:
        """Generate HTML table for trades."""
        
        if trades_df is None or trades_df.empty:
            return "<p>No trades recorded</p>"
        
        # Select key columns
        display_cols = [
            'Entry Timestamp', 'Exit Timestamp', 'Avg Entry Price',
            'Avg Exit Price', 'PnL', 'Return', 'Direction', 'Status'
        ]
        
        available_cols = [c for c in display_cols if c in trades_df.columns]
        
        if not available_cols:
            available_cols = trades_df.columns.tolist()[:8]
        
        df = trades_df[available_cols].copy()
        
        # Generate table HTML
        html = '<div class="trades-table-container"><table class="trades-table">'
        
        # Header
        html += '<thead><tr>'
        for col in available_cols:
            html += f'<th>{col}</th>'
        html += '</tr></thead>'
        
        # Body (limit to first 50 trades)
        html += '<tbody>'
        for _, row in df.head(50).iterrows():
            html += '<tr>'
            for col in available_cols:
                val = row[col]
                if col == 'PnL' and pd.notna(val):
                    css_class = 'positive' if val > 0 else 'negative'
                    html += f'<td class="{css_class}">${val:,.2f}</td>'
                elif col == 'Return' and pd.notna(val):
                    css_class = 'positive' if val > 0 else 'negative'
                    html += f'<td class="{css_class}">{val*100:.2f}%</td>'
                elif 'Price' in col and pd.notna(val):
                    html += f'<td>${val:,.2f}</td>'
                else:
                    html += f'<td>{val}</td>'
            html += '</tr>'
        html += '</tbody></table>'
        
        if len(df) > 50:
            html += f'<p class="note">Showing 50 of {len(df)} trades</p>'
        
        html += '</div>'
        
        return html
    
    def generate_html_report(
        self,
        strategy: str,
        pair: str,
        stats_df: Optional[pd.DataFrame] = None,
        equity_df: Optional[pd.DataFrame] = None,
        trades_df: Optional[pd.DataFrame] = None
    ) -> str:
        """
        Generate a standalone HTML report.
        
        Args:
            strategy: Strategy name
            pair: Trading pair
            stats_df: Statistics DataFrame
            equity_df: Equity curve DataFrame
            trades_df: Trades DataFrame
        
        Returns:
            str: Path to generated HTML file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pair_clean = pair.replace("/", "")
        filename = f"report_{strategy}_{pair_clean}_{timestamp}.html"
        filepath = Path(self.output_dir) / filename
        
        # Generate chart
        chart_html = ""
        if equity_df is not None and not equity_df.empty:
            fig = self._create_equity_chart(equity_df)
            if fig:
                chart_html = pio.to_html(fig, include_plotlyjs='cdn', full_html=False)
        
        # Generate stats
        stats_html = self._generate_stats_html(stats_df, strategy, pair)
        
        # Generate trades table
        trades_html = self._generate_trades_html(trades_df)
        
        # Full HTML template
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Backtest Report - {strategy} | {pair}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e0e0e0;
            min-height: 100vh;
            padding: 2rem;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 2rem;
            padding: 2rem;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        .header h1 {{
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(90deg, #00d4aa, #7c3aed);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        .header .meta {{
            color: #888;
            font-size: 1rem;
        }}
        
        .section {{
            background: rgba(255, 255, 255, 0.03);
            border-radius: 15px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.08);
        }}
        
        .section h2 {{
            font-size: 1.3rem;
            margin-bottom: 1rem;
            color: #fff;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem;
        }}
        
        .stat-card {{
            background: rgba(255, 255, 255, 0.05);
            padding: 1.2rem;
            border-radius: 10px;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        .stat-label {{
            font-size: 0.85rem;
            color: #888;
            margin-bottom: 0.5rem;
        }}
        
        .stat-value {{
            font-size: 1.5rem;
            font-weight: 600;
            color: #fff;
        }}
        
        .stat-value.positive {{
            color: #00d4aa;
        }}
        
        .stat-value.negative {{
            color: #ff6b6b;
        }}
        
        .trades-table-container {{
            overflow-x: auto;
            max-height: 500px;
            overflow-y: auto;
        }}
        
        .trades-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
        }}
        
        .trades-table th,
        .trades-table td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        .trades-table th {{
            background: rgba(255, 255, 255, 0.05);
            color: #888;
            font-weight: 500;
            position: sticky;
            top: 0;
        }}
        
        .trades-table tbody tr:hover {{
            background: rgba(255, 255, 255, 0.03);
        }}
        
        .positive {{
            color: #00d4aa;
        }}
        
        .negative {{
            color: #ff6b6b;
        }}
        
        .note {{
            color: #666;
            font-size: 0.85rem;
            margin-top: 1rem;
            text-align: center;
        }}
        
        .footer {{
            text-align: center;
            padding: 1.5rem;
            color: #555;
            font-size: 0.85rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Backtest Report</h1>
            <div class="meta">
                <strong>{strategy}</strong> | {pair} | Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}
            </div>
        </div>
        
        <div class="section">
            <h2>📈 Performance Metrics</h2>
            {stats_html}
        </div>
        
        <div class="section">
            <h2>💰 Equity Curve & Drawdown</h2>
            {chart_html if chart_html else '<p>No chart data available</p>'}
        </div>
        
        <div class="section">
            <h2>📋 Trade History</h2>
            {trades_html}
        </div>
        
        <div class="footer">
            Generated by Crypto Swing Trading Bot • {datetime.now().year}
        </div>
    </div>
</body>
</html>
"""
        
        # Write file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"✅ HTML report generated: {filepath}")
        
        return str(filepath)
    def generate_optimization_report(
        self,
        strategy: str,
        pair: str,
        best_params: dict,
        train_stats: dict,
        test_stats: dict,
        all_results: pd.DataFrame
    ) -> str:
        """
        Generate a detailed Optimization HTML report.
        
        Args:
            strategy: Strategy name
            pair: Trading pair
            best_params: Best parameters found (dict)
            train_stats: Stats on Train set (dict)
            test_stats: Stats on Test set (dict)
            all_results: DataFrame of all parameter combinations run
            
        Returns:
            str: Path to generated HTML file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"opt_report_{strategy}_{pair.replace('/','')}_{timestamp}.html"
        filepath = Path(self.output_dir) / filename
        
        # 1. Overview Section
        train_sharpe = train_stats.get('sharpe', 0)
        test_sharpe = test_stats.get('sharpe', 0)
        sharpe_diff = trn_diff = abs(train_sharpe - test_sharpe)
        stability_class = 'positive' if sharpe_diff < 0.5 else ('negative' if sharpe_diff > 1.0 else 'warning')
        
        overview_html = f"""
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Best Train Sharpe</div>
                <div class="stat-value">{train_sharpe:.2f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Test Sharpe (Validation)</div>
                <div class="stat-value">{test_sharpe:.2f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Stability (Sharpe Diff)</div>
                <div class="stat-value {stability_class}">{sharpe_diff:.2f}</div>
            </div>
             <div class="stat-card">
                <div class="stat-label">Test Total Return</div>
                <div class="stat-value {'positive' if test_stats.get('total_return', 0) > 0 else 'negative'}">{test_stats.get('total_return', 0):.2f}%</div>
            </div>
        </div>
        """
        
        # 2. Parameters Table
        params_html = '<table class="trades-table"><thead><tr><th>Parameter</th><th>Value</th></tr></thead><tbody>'
        for k, v in best_params.items():
            params_html += f"<tr><td>{k}</td><td>{v}</td></tr>"
        params_html += "</tbody></table>"
        
        # 3. Top 10 Runs Table
        if all_results is not None and not all_results.empty:
            top_10 = all_results.sort_values('sharpe', ascending=False).head(10)
            top_10_html = '<table class="trades-table"><thead><tr>'
            for col in top_10.columns:
                top_10_html += f'<th>{col}</th>'
            top_10_html += '</tr></thead><tbody>'
            for _, row in top_10.iterrows():
                top_10_html += '<tr>'
                for val in row:
                    if isinstance(val, float):
                        top_10_html += f'<td>{val:.2f}</td>'
                    else:
                        top_10_html += f'<td>{val}</td>'
                top_10_html += '</tr>'
            top_10_html += '</tbody></table>'
        else:
            top_10_html = "<p>No detailed results available.</p>"

        # Full HTML
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Optimization Report - {strategy} | {pair}</title>
    <style>
        /* Reusing simpler styles for consistency */
        body {{ font-family: sans-serif; background: #1a1a2e; color: #e0e0e0; padding: 2rem; }}
        .header {{ text-align: center; margin-bottom: 2rem; padding: 1rem; background: rgba(255,255,255,0.05); border-radius: 10px; }}
        h1 {{ color: #00d4aa; }}
        .section {{ background: rgba(255,255,255,0.03); padding: 1.5rem; border-radius: 10px; margin-bottom: 1.5rem; }}
        h2 {{ border-bottom: 1px solid #444; padding-bottom: 0.5rem; margin-bottom: 1rem; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; }}
        .stat-card {{ background: rgba(255,255,255,0.05); padding: 1rem; border-radius: 8px; text-align: center; }}
        .stat-value {{ font-size: 1.5rem; font-weight: bold; }}
        .positive {{ color: #00d4aa; }}
        .negative {{ color: #ff6b6b; }}
        .warning {{ color: #f1c40f; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; }}
        th, td {{ padding: 0.8rem; text-align: left; border-bottom: 1px solid #444; }}
        th {{ background: rgba(255,255,255,0.05); }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧬 Optimization Report</h1>
        <p>{strategy} | {pair} | {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
    </div>
    
    <div class="section">
        <h2>🏆 Best Run Review</h2>
        {overview_html}
    </div>
    
    <div class="section">
        <h2>⚙️ Optimal Parameters</h2>
        {params_html}
    </div>
    
    <div class="section">
        <h2>🔝 Top 10 Configurations (Train Phase)</h2>
        <div style="overflow-x: auto;">
            {top_10_html}
        </div>
    </div>
    
    <div class="header" style="font-size: 0.8rem; color: #666;">
        Generated by Crypto Swing Trading Bot
    </div>
</body>
</html>
"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        logger.info(f"✅ Optimization report generated: {filepath}")
        return str(filepath)
