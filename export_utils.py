"""
Export Utilities Module
Generate CSV, JSON, and HTML report exports
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import io
import warnings

warnings.filterwarnings("ignore")


class ExportManager:
    """Manage data exports for screener results, backtests, and reports"""

    @staticmethod
    def screener_to_csv(results_df):
        """Export screener results to CSV string"""
        if results_df is None or results_df.empty:
            return ""
        return results_df.to_csv(index=False)

    @staticmethod
    def backtest_to_csv(backtest_result):
        """Export backtest results to CSV string"""
        if not backtest_result:
            return ""

        # Summary metrics
        summary = {k: v for k, v in backtest_result.items()
                   if not isinstance(v, (pd.DataFrame, pd.Series))}
        summary_df = pd.DataFrame([summary])

        output = io.StringIO()
        output.write("=== BACKTEST SUMMARY ===\n")
        summary_df.to_csv(output, index=False)

        if "trades" in backtest_result and not backtest_result["trades"].empty:
            output.write("\n=== TRADE LOG ===\n")
            backtest_result["trades"].to_csv(output, index=False)

        return output.getvalue()

    @staticmethod
    def risk_metrics_to_csv(metrics):
        """Export risk metrics to CSV string"""
        if not metrics:
            return ""

        # Filter out non-scalar values
        clean = {k: v for k, v in metrics.items()
                 if isinstance(v, (int, float, str, np.floating, np.integer))}
        df = pd.DataFrame([clean])
        return df.to_csv(index=False)

    @staticmethod
    def fundamentals_to_csv(fundamentals):
        """Export fundamental analysis to CSV string"""
        if not fundamentals:
            return ""

        flat = {}
        for category, data in fundamentals.items():
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, (int, float, str, type(None))):
                        flat[f"{category}_{k}"] = v
            elif isinstance(data, (int, float, str)):
                flat[category] = data

        df = pd.DataFrame([flat])
        return df.to_csv(index=False)

    @staticmethod
    def comparison_to_csv(comparison_df):
        """Export comparison DataFrame to CSV"""
        if comparison_df is None or comparison_df.empty:
            return ""
        return comparison_df.to_csv()

    @staticmethod
    def generate_stock_report_html(symbol, analysis_data, fundamentals=None,
                                    technical=None, risk_metrics=None):
        """Generate a comprehensive HTML report for a stock"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Stock Report - {symbol}</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 2rem; background: #0e1117; color: #e0e0e0; }}
        h1 {{ color: #667eea; border-bottom: 2px solid #667eea; padding-bottom: 10px; }}
        h2 {{ color: #764ba2; margin-top: 2rem; }}
        h3 {{ color: #88a0d0; }}
        table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
        th, td {{ border: 1px solid #333; padding: 8px 12px; text-align: left; }}
        th {{ background: #1e2130; color: #667eea; }}
        tr:nth-child(even) {{ background: #161b22; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin: 1rem 0; }}
        .metric-card {{ background: #1e2130; padding: 1rem; border-radius: 8px; text-align: center; }}
        .metric-value {{ font-size: 1.4em; font-weight: bold; color: #667eea; }}
        .metric-label {{ color: #888; font-size: 0.85em; }}
        .bullish {{ color: #2ecc71; }} .bearish {{ color: #e74c3c; }} .neutral {{ color: #f39c12; }}
        .footer {{ text-align: center; margin-top: 3rem; color: #666; font-size: 0.85em; }}
        .badge {{ display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 0.85em; }}
        .badge-buy {{ background: #1a4731; color: #2ecc71; }}
        .badge-sell {{ background: #4a1a1a; color: #e74c3c; }}
        .badge-hold {{ background: #4a3a1a; color: #f39c12; }}
    </style>
</head>
<body>
    <h1>📊 Stock Report: {symbol}</h1>
    <p><strong>Generated:</strong> {timestamp} | <strong>Source:</strong> Artha Drishti v4.0</p>
"""

        # Analysis section
        if analysis_data:
            html += f"""
    <h2>📈 Price & Technical Overview</h2>
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-label">Current Price</div>
            <div class="metric-value">₹{analysis_data.get('current_price', 'N/A')}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Market Cap</div>
            <div class="metric-value">{analysis_data.get('market_cap', 'N/A')}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">P/E Ratio</div>
            <div class="metric-value">{analysis_data.get('pe_ratio', 'N/A')}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">RSI</div>
            <div class="metric-value">{analysis_data.get('rsi', 'N/A')}</div>
        </div>
    </div>
"""

        # Fundamentals section
        if fundamentals:
            val = fundamentals.get("valuation", {})
            prof = fundamentals.get("profitability", {})
            health = fundamentals.get("financial_health", {})
            html += """
    <h2>📋 Fundamental Analysis</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
"""
            for label, key, src in [
                ("P/E Ratio", "pe_ratio", val),
                ("P/B Ratio", "pb_ratio", val),
                ("EV/EBITDA", "ev_ebitda", val),
                ("ROE", "roe", prof),
                ("Profit Margin", "profit_margin", prof),
                ("Debt/Equity", "debt_to_equity", health),
                ("Current Ratio", "current_ratio", health),
            ]:
                v = src.get(key)
                display = f"{v:.2f}" if isinstance(v, (int, float)) else "N/A"
                html += f"        <tr><td>{label}</td><td>{display}</td></tr>\n"
            html += "    </table>\n"

        # Technical patterns
        if technical:
            html += "    <h2>🔍 Technical Patterns</h2>\n"
            for p in technical.get("chart_patterns", []):
                signal_class = "bullish" if p["signal"] == "Bullish" else ("bearish" if p["signal"] == "Bearish" else "neutral")
                html += f'    <p><strong>{p["pattern"]}</strong> - <span class="{signal_class}">{p["signal"]}</span> (Confidence: {p["confidence"]}%)</p>\n'
            for p in technical.get("candlestick_patterns", []):
                signal_class = "bullish" if "Bullish" in p["signal"] else ("bearish" if "Bearish" in p["signal"] else "neutral")
                html += f'    <p><strong>{p["pattern"]}</strong> - <span class="{signal_class}">{p["signal"]}</span></p>\n'

        # Risk metrics
        if risk_metrics:
            html += """
    <h2>🛡️ Risk Metrics</h2>
    <div class="metric-grid">
"""
            for label, key in [("Sharpe Ratio", "sharpe_ratio"), ("Sortino Ratio", "sortino_ratio"),
                               ("Max Drawdown", "max_drawdown"), ("VaR (95%)", "var_95")]:
                v = risk_metrics.get(key)
                display = f"{v:.3f}" if isinstance(v, (int, float)) else "N/A"
                html += f'        <div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{display}</div></div>\n'
            html += "    </div>\n"

        html += f"""
    <div class="footer">
        <p>⚠️ For educational purposes only. Not financial advice.</p>
        <p>Generated by Artha Drishti v4.0 | {timestamp}</p>
    </div>
</body>
</html>
"""
        return html

    @staticmethod
    def portfolio_to_csv(portfolio_data):
        """Export portfolio holdings and analysis to CSV"""
        if not portfolio_data:
            return ""
        df = pd.DataFrame(portfolio_data)
        return df.to_csv(index=False)

    @staticmethod
    def watchlist_to_csv(watchlist_data):
        """Export watchlist to CSV"""
        if not watchlist_data:
            return ""
        df = pd.DataFrame(watchlist_data)
        return df.to_csv(index=False)
