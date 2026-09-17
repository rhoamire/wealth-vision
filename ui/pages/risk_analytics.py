"""Page: Risk Analytics"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import json

from core.utils import *
from core.strategy_engine import StrategyEngine
from core.ml_predictor import EnhancedMLPredictor
from core.portfolio_manager import PortfolioManager
from core.stock_analyzer import StockAnalyzer
from core.data_loader import *
from ui.shared import display_stock_analysis
from core.auth import save_user_watchlist, save_user_portfolio, save_user_alerts, save_user_settings, save_user_trade_journal, change_password, get_user_data
from core.export_utils import ExportManager
from core.multi_exchange import MultiExchangeHandler, MarketOverview, CrossExchangeComparator, EXCHANGE_CONFIG
from core.advanced_strategies import AdvancedStrategyEngine, SectorRotationDetector, compute_full_indicators
from core.risk_analytics import RiskAnalytics, PortfolioRiskAnalyzer, StockComparator
from core.backtester import Backtester, BACKTEST_STRATEGIES
from core.technical_patterns import TechnicalAnalyzer, SupportResistanceDetector, ChartPatternDetector
from core.fundamental_analysis import FundamentalAnalyzer
from core.news_sentiment import NewsSentimentAnalyzer
from core.price_predictor import PricePredictor
from config import NEWS_API_KEY

def render_risk_analytics_page():
    st.markdown("## 🛡️ Risk Analytics")
    st.info("Comprehensive risk analysis with VaR, drawdown, Sharpe, Sortino, and more")

    tab1, tab2, tab3 = st.tabs(["Single Stock Risk", "Stock Comparison", "Portfolio Risk"])

    with tab1:
        st.markdown("### Stock Risk Analysis")

        col1, col2, col3 = st.columns(3)
        with col1:
            ra_exchange = st.selectbox("Exchange", st.session_state.exchange_handler.get_supported_exchanges(), key="ra_ex")
            ra_lists = st.session_state.exchange_handler.get_stock_lists(ra_exchange)
            if ra_lists:
                ra_list_name = st.selectbox("Stock List", list(ra_lists.keys()), key="ra_list")
                ra_stocks = ra_lists[ra_list_name]
            else:
                ra_stocks = st.session_state.nse_stocks
            ra_symbol = st.selectbox("Stock", ra_stocks, key="ra_sym")

        with col2:
            ra_period = st.selectbox("Period", ["1y", "2y", "3y", "5y"], index=1, key="ra_period")

        with col3:
            ra_benchmark = st.checkbox("Include Benchmark Comparison", value=True, key="ra_bench")

        if st.button("🔍 Analyze Risk", type="primary", key="ra_run"):
            with st.spinner(f"Computing risk metrics for {ra_symbol}..."):
                df = st.session_state.exchange_handler.get_stock_data(ra_symbol, ra_exchange, period=ra_period)

                benchmark_df = None
                if ra_benchmark:
                    try:
                        from multi_exchange import EXCHANGE_CONFIG
                        idx_symbol = EXCHANGE_CONFIG.get(ra_exchange, {}).get('index', '^NSEI')
                        benchmark_df = yf.download(idx_symbol, period=ra_period, progress=False)
                        # Flatten multi-level columns from yf.download()
                        if benchmark_df is not None and isinstance(benchmark_df.columns, pd.MultiIndex):
                            benchmark_df.columns = benchmark_df.columns.get_level_values(0)
                    except:
                        pass

                if df is not None and len(df) >= 30:
                    metrics = st.session_state.risk_analytics.compute_all_metrics(df, benchmark_df)

                    if metrics:
                        # Return & Volatility
                        st.markdown("### Return & Volatility Metrics")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("CAGR", f"{metrics.get('cagr', 0):.2f}%")
                        with col2:
                            st.metric("Annual Volatility", f"{metrics.get('annualized_volatility', 0):.2f}%")
                        with col3:
                            st.metric("Cumulative Return", f"{metrics.get('cumulative_return', 0):.2f}%")
                        with col4:
                            st.metric("Daily Avg Return", f"{metrics.get('daily_mean_return', 0):.4f}%")

                        st.markdown("---")

                        # Risk-Adjusted
                        st.markdown("### Risk-Adjusted Metrics")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            sharpe = metrics.get('sharpe_ratio', 0)
                            sharpe_emoji = "🟢" if sharpe > 1 else ("🟡" if sharpe > 0 else "🔴")
                            st.metric("Sharpe Ratio", f"{sharpe:.3f} {sharpe_emoji}")
                        with col2:
                            sortino = metrics.get('sortino_ratio', 0)
                            st.metric("Sortino Ratio", f"{sortino:.3f}")
                        with col3:
                            calmar = metrics.get('calmar_ratio', 0)
                            st.metric("Calmar Ratio", f"{calmar:.3f}")
                        with col4:
                            info_r = metrics.get('information_ratio', 'N/A')
                            if isinstance(info_r, (int, float)):
                                st.metric("Information Ratio", f"{info_r:.3f}")
                            else:
                                st.metric("Information Ratio", "N/A")

                        st.markdown("---")

                        # Drawdown
                        st.markdown("### Drawdown Analysis")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2f}%")
                        with col2:
                            st.metric("Current Drawdown", f"{metrics.get('current_drawdown', 0):.2f}%")
                        with col3:
                            st.metric("Avg Drawdown", f"{metrics.get('avg_drawdown', 0):.2f}%")

                        # Drawdown chart
                        prices = df['Close']
                        peak = prices.expanding(min_periods=1).max()
                        dd_series = (prices - peak) / peak * 100
                        fig_dd = go.Figure()
                        fig_dd.add_trace(go.Scatter(
                            x=df.index, y=dd_series, mode='lines', fill='tozeroy',
                            name='Drawdown', line=dict(color='#e74c3c', width=1)
                        ))
                        fig_dd.update_layout(title="Drawdown Over Time", yaxis_title="Drawdown %", height=300)
                        st.plotly_chart(fig_dd, use_container_width=True)

                        st.markdown("---")

                        # VaR
                        st.markdown("### Value at Risk (VaR)")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("VaR (95%)", f"{metrics.get('var_95', 0):.4f}")
                        with col2:
                            st.metric("VaR (99%)", f"{metrics.get('var_99', 0):.4f}")
                        with col3:
                            st.metric("CVaR (95%)", f"{metrics.get('cvar_95', 0):.4f}")

                        st.markdown("---")

                        # Benchmark relative metrics
                        if ra_benchmark and 'beta' in metrics:
                            st.markdown("### Benchmark-Relative Metrics")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Beta", f"{metrics.get('beta', 0):.3f}")
                            with col2:
                                st.metric("Alpha (Annual)", f"{metrics.get('alpha_annual', 0):.3f}")
                            with col3:
                                st.metric("Treynor Ratio", f"{metrics.get('treynor_ratio', 0):.4f}")
                            with col4:
                                st.metric("Correlation", f"{metrics.get('correlation', 0):.3f}")

                        # Tail Risk
                        st.markdown("### Tail Risk & Distribution")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Skewness", f"{metrics.get('skewness', 0):.3f}")
                        with col2:
                            st.metric("Kurtosis", f"{metrics.get('kurtosis', 0):.3f}")
                        with col3:
                            st.metric("Worst Day", f"{metrics.get('worst_day', 0):.2f}%")
                        with col4:
                            st.metric("Best Day", f"{metrics.get('best_day', 0):.2f}%")

                        # Returns distribution
                        returns = df['Close'].pct_change().dropna() * 100
                        fig_hist = go.Figure()
                        fig_hist.add_trace(go.Histogram(
                            x=returns, nbinsx=50, name='Daily Returns',
                            marker_color='#667eea', opacity=0.75
                        ))
                        fig_hist.update_layout(title="Daily Returns Distribution",
                                               xaxis_title="Return %", yaxis_title="Frequency", height=350)
                        st.plotly_chart(fig_hist, use_container_width=True)

                        # Export risk metrics
                        try:
                            risk_csv = st.session_state.export_manager.risk_metrics_to_csv(metrics)
                            st.download_button("📥 Download Risk Metrics (CSV)", risk_csv,
                                               f"risk_{ra_symbol}_{ra_period}.csv", "text/csv",
                                               use_container_width=True, key="ra_export_single")
                        except Exception:
                            pass
                    else:
                        st.error("Could not compute risk metrics.")
                else:
                    st.error("Could not fetch stock data or insufficient data.")

    with tab2:
        st.markdown("### Compare Multiple Stocks")

        cmp_exchange_r = st.selectbox("Exchange", st.session_state.exchange_handler.get_supported_exchanges(), key="ra_cmp_ex")
        cmp_lists_r = st.session_state.exchange_handler.get_stock_lists(cmp_exchange_r)
        if cmp_lists_r:
            cmp_list_r = st.selectbox("Stock List", list(cmp_lists_r.keys()), key="ra_cmp_list")
            cmp_stocks_r = cmp_lists_r[cmp_list_r]
        else:
            cmp_stocks_r = st.session_state.nse_stocks

        selected_stocks = st.multiselect("Select Stocks to Compare (2-6)", cmp_stocks_r, default=cmp_stocks_r[:3] if len(cmp_stocks_r) >= 3 else cmp_stocks_r[:2], key="ra_cmp_stocks")
        cmp_period_r = st.selectbox("Period", ["1y", "2y", "3y"], index=0, key="ra_cmp_period")

        if st.button("📊 Compare Risk Profiles", type="primary", key="ra_cmp_run") and len(selected_stocks) >= 2:
            with st.spinner("Fetching data and computing metrics..."):
                stock_data = {}
                for sym in selected_stocks[:6]:
                    data = st.session_state.exchange_handler.get_stock_data(sym, cmp_exchange_r, period=cmp_period_r)
                    if data is not None and len(data) > 30:
                        stock_data[sym] = data

                if len(stock_data) >= 2:
                    comparison_result = st.session_state.stock_comparator.compare(stock_data)
                    if not comparison_result.empty:
                        st.markdown("### Comparison Table")
                        st.dataframe(comparison_result.round(3), use_container_width=True)

                        # Normalized price chart
                        st.markdown("### Normalized Performance")
                        fig_norm = go.Figure()
                        for sym, data in stock_data.items():
                            normalized = data['Close'] / data['Close'].iloc[0] * 100
                            fig_norm.add_trace(go.Scatter(
                                x=data.index, y=normalized, mode='lines', name=sym
                            ))
                        fig_norm.update_layout(title="Normalized Performance (Base=100)",
                                               yaxis_title="Value", height=400)
                        st.plotly_chart(fig_norm, use_container_width=True)

                        # Correlation matrix
                        st.markdown("### Correlation Matrix")
                        returns_dict = {}
                        for sym, data in stock_data.items():
                            ret_s = data['Close'].pct_change().dropna()
                            if hasattr(ret_s.index, 'tz') and ret_s.index.tz is not None:
                                ret_s = ret_s.copy()
                                ret_s.index = ret_s.index.tz_localize(None)
                            returns_dict[sym] = ret_s
                        corr_df = pd.DataFrame(returns_dict).corr()
                        fig_corr = go.Figure(data=go.Heatmap(
                            z=corr_df.values, x=corr_df.columns, y=corr_df.index,
                            colorscale='RdYlGn', zmin=-1, zmax=1,
                            text=corr_df.round(2).values, texttemplate="%{text}"
                        ))
                        fig_corr.update_layout(title="Return Correlation Matrix", height=400)
                        st.plotly_chart(fig_corr, use_container_width=True)

                        # Export comparison
                        try:
                            cmp_csv = st.session_state.export_manager.comparison_to_csv(comparison_result)
                            st.download_button("📥 Download Comparison (CSV)", cmp_csv,
                                               f"risk_comparison_{cmp_period_r}.csv", "text/csv",
                                               use_container_width=True, key="ra_export_cmp")
                        except Exception:
                            pass
                else:
                    st.error("Need at least 2 stocks with valid data.")

    with tab3:
        st.markdown("### Portfolio Risk Analysis")
        st.markdown("Add stocks and weights to analyze portfolio-level risk metrics")

        pf_exchange = st.selectbox("Exchange", st.session_state.exchange_handler.get_supported_exchanges(), key="ra_pf_ex")
        pf_lists = st.session_state.exchange_handler.get_stock_lists(pf_exchange)
        if pf_lists:
            pf_list_name = st.selectbox("Stock List", list(pf_lists.keys()), key="ra_pf_list")
            pf_available = pf_lists[pf_list_name]
        else:
            pf_available = st.session_state.nse_stocks

        pf_stocks = st.multiselect("Select Portfolio Stocks", pf_available,
                                    default=pf_available[:4] if len(pf_available) >= 4 else pf_available[:2],
                                    key="ra_pf_stocks")
        pf_period = st.selectbox("Period", ["1y", "2y", "3y"], index=0, key="ra_pf_period")

        if pf_stocks:
            st.markdown("**Set Weights (must sum to 1.0):**")
            weights = []
            equal_w = 1.0 / len(pf_stocks) if pf_stocks else 0.25
            cols = st.columns(min(len(pf_stocks), 4))
            for i, sym in enumerate(pf_stocks):
                with cols[i % 4]:
                    w = st.number_input(f"{sym}", 0.0, 1.0, round(equal_w, 2), 0.05, key=f"pf_w_{sym}")
                    weights.append(w)

            if st.button("📈 Analyze Portfolio Risk", type="primary", key="ra_pf_run"):
                total_w = sum(weights)
                if abs(total_w - 1.0) > 0.05:
                    st.warning(f"Weights sum to {total_w:.2f}. Normalizing to 1.0.")
                    weights = [w / total_w for w in weights]

                with st.spinner("Computing portfolio risk..."):
                    stock_data = {}
                    for sym in pf_stocks:
                        data = st.session_state.exchange_handler.get_stock_data(sym, pf_exchange, period=pf_period)
                        if data is not None and len(data) > 30:
                            stock_data[sym] = data

                    if len(stock_data) >= 2:
                        adj_weights = [weights[i] for i, s in enumerate(pf_stocks) if s in stock_data]
                        total_aw = sum(adj_weights)
                        adj_weights = [w / total_aw for w in adj_weights]

                        pf_result = st.session_state.portfolio_risk.analyze_portfolio_risk(stock_data, adj_weights)
                        if pf_result:
                            st.markdown("### Portfolio Metrics")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Portfolio Return", f"{pf_result.get('portfolio_return', 0)*100:.2f}%")
                            with col2:
                                st.metric("Portfolio Volatility", f"{pf_result.get('portfolio_volatility', 0)*100:.2f}%")
                            with col3:
                                st.metric("Portfolio Sharpe", f"{pf_result.get('portfolio_sharpe', 0):.3f}")
                            with col4:
                                st.metric("Diversification Ratio", f"{pf_result.get('diversification_ratio', 0):.3f}")

                            # Weights pie chart
                            st.markdown("### Portfolio Allocation")
                            stock_names = list(stock_data.keys())
                            fig_pie = go.Figure(data=[go.Pie(
                                labels=stock_names, values=adj_weights,
                                hole=0.4, textinfo='label+percent'
                            )])
                            fig_pie.update_layout(title="Portfolio Weights", height=350)
                            st.plotly_chart(fig_pie, use_container_width=True)
                    else:
                        st.error("Need at least 2 stocks with data for portfolio analysis.")

