"""Page: Backtesting"""
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

def render_backtesting_page():
    st.markdown("## 📈 Strategy Backtesting")
    st.info("Backtest trading strategies against historical data to evaluate their performance")

    tab1, tab2 = st.tabs(["Single Strategy Backtest", "Strategy Comparison"])

    with tab1:
        st.markdown("### Backtest a Strategy")

        col1, col2, col3 = st.columns(3)
        with col1:
            bt_exchange = st.selectbox("Exchange", st.session_state.exchange_handler.get_supported_exchanges(), key="bt_ex")
            bt_lists = st.session_state.exchange_handler.get_stock_lists(bt_exchange)
            if bt_lists:
                bt_list_name = st.selectbox("Stock List", list(bt_lists.keys()), key="bt_list")
                bt_stocks = bt_lists[bt_list_name]
            else:
                bt_stocks = st.session_state.nse_stocks
            bt_symbol = st.selectbox("Stock", bt_stocks, key="bt_sym")

        with col2:
            bt_strategy = st.selectbox("Strategy", list(BACKTEST_STRATEGIES.keys()), key="bt_strat")
            bt_period = st.selectbox("Backtest Period", ["1y", "2y", "3y", "5y"], index=1, key="bt_period")

        with col3:
            bt_capital = st.number_input("Initial Capital", 10000, 10000000, 100000, step=10000, key="bt_cap")
            bt_commission = st.number_input("Commission %", 0.0, 1.0, 0.1, 0.01, key="bt_comm")

        if st.button("▶️ Run Backtest", type="primary", key="bt_run"):
            with st.spinner(f"Backtesting {bt_strategy} on {bt_symbol}..."):
                df = st.session_state.exchange_handler.get_stock_data(bt_symbol, bt_exchange, period=bt_period)

                if df is not None and len(df) >= 50:
                    bt = Backtester(
                        initial_capital=bt_capital,
                        commission_pct=bt_commission / 100
                    )
                    result = bt.run_backtest(df, BACKTEST_STRATEGIES[bt_strategy], strategy_name=bt_strategy)

                    if result and 'error' not in result:
                        # Performance metrics
                        st.markdown("### Performance Summary")
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.metric("Total Return", f"{result['total_return']:.2f}%")
                        with col2:
                            st.metric("CAGR", f"{result['cagr']:.2f}%")
                        with col3:
                            st.metric("Sharpe Ratio", f"{result['sharpe_ratio']:.2f}")
                        with col4:
                            st.metric("Max Drawdown", f"{result['max_drawdown']:.2f}%")
                        with col5:
                            st.metric("Win Rate", f"{result['win_rate']:.1f}%")

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Final Equity", f"{result['final_equity']:,.0f}")
                        with col2:
                            st.metric("Total Trades", result['total_trades'])
                        with col3:
                            st.metric("Profit Factor", f"{result['profit_factor']:.2f}")
                        with col4:
                            st.metric("Sortino Ratio", f"{result['sortino_ratio']:.2f}")

                        st.markdown("---")

                        # Equity curve
                        if 'equity_curve' in result and not result['equity_curve'].empty:
                            st.markdown("### Equity Curve")
                            equity_df = result['equity_curve']
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=equity_df.index, y=equity_df['equity'],
                                mode='lines', name='Portfolio Value',
                                line=dict(color='#667eea', width=2)
                            ))
                            fig.add_hline(y=bt_capital, line_dash="dash", line_color="gray",
                                          annotation_text="Initial Capital")
                            fig.update_layout(
                                title=f"Equity Curve - {bt_strategy} on {bt_symbol}",
                                yaxis_title="Portfolio Value", height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            # Drawdown chart
                            peak = equity_df['equity'].expanding(min_periods=1).max()
                            drawdown = (equity_df['equity'] - peak) / peak * 100
                            fig_dd = go.Figure()
                            fig_dd.add_trace(go.Scatter(
                                x=equity_df.index, y=drawdown,
                                mode='lines', fill='tozeroy', name='Drawdown',
                                line=dict(color='#e74c3c', width=1)
                            ))
                            fig_dd.update_layout(title="Drawdown", yaxis_title="Drawdown %", height=250)
                            st.plotly_chart(fig_dd, use_container_width=True)

                        # Trade log
                        if 'trades' in result and not result['trades'].empty:
                            st.markdown("### Trade Log")
                            trades_df = result['trades'].copy()
                            display_cols = [c for c in ['date', 'type', 'price', 'shares', 'pnl', 'pnl_pct'] if c in trades_df.columns]
                            if display_cols:
                                st.dataframe(trades_df[display_cols].tail(20), use_container_width=True, hide_index=True)

                        # Additional metrics
                        st.markdown("### Detailed Metrics")
                        detail_data = {
                            "Metric": ["Annualized Volatility", "Avg Win %", "Avg Loss %", "Max Win %", "Max Loss %", "Total Commission"],
                            "Value": [
                                f"{result['annualized_volatility']:.2f}%",
                                f"{result['avg_win_pct']:.2f}%",
                                f"{result['avg_loss_pct']:.2f}%",
                                f"{result['max_win_pct']:.2f}%",
                                f"{result['max_loss_pct']:.2f}%",
                                f"{result['total_commission']:,.2f}"
                            ]
                        }
                        st.dataframe(pd.DataFrame(detail_data), use_container_width=True, hide_index=True)

                        # Export backtest results
                        try:
                            bt_csv = st.session_state.export_manager.backtest_to_csv(result, bt_strategy)
                            st.download_button("📥 Download Backtest Results (CSV)", bt_csv,
                                               f"backtest_{bt_symbol}_{bt_strategy}_{bt_period}.csv", "text/csv",
                                               use_container_width=True, key="bt_export_single")
                        except Exception:
                            pass
                    else:
                        st.warning("Backtest produced no results. The strategy may not have generated any signals for this stock/period.")
                else:
                    st.error("Could not fetch data or insufficient data points.")

    with tab2:
        st.markdown("### Compare All Strategies")

        col1, col2, col3 = st.columns(3)
        with col1:
            cmp_exchange = st.selectbox("Exchange", st.session_state.exchange_handler.get_supported_exchanges(), key="cmp_ex")
            cmp_lists = st.session_state.exchange_handler.get_stock_lists(cmp_exchange)
            if cmp_lists:
                cmp_list_name = st.selectbox("Stock List", list(cmp_lists.keys()), key="cmp_list")
                cmp_stocks = cmp_lists[cmp_list_name]
            else:
                cmp_stocks = st.session_state.nse_stocks
            cmp_symbol = st.selectbox("Stock", cmp_stocks, key="cmp_sym")

        with col2:
            cmp_period = st.selectbox("Period", ["1y", "2y", "3y", "5y"], index=1, key="cmp_period")

        with col3:
            cmp_capital = st.number_input("Capital", 10000, 10000000, 100000, 10000, key="cmp_cap")

        if st.button("📊 Compare All Strategies", type="primary", key="cmp_run"):
            with st.spinner("Running all strategies..."):
                df = st.session_state.exchange_handler.get_stock_data(cmp_symbol, cmp_exchange, period=cmp_period)

                if df is not None and len(df) >= 50:
                    bt = Backtester(initial_capital=cmp_capital)
                    comparison = bt.compare_strategies(df, BACKTEST_STRATEGIES)

                    if comparison:
                        # Comparison table
                        cmp_data = []
                        for name, res in comparison.items():
                            cmp_data.append({
                                'Strategy': name,
                                'Total Return %': f"{res['total_return']:.2f}",
                                'CAGR %': f"{res['cagr']:.2f}",
                                'Sharpe': f"{res['sharpe_ratio']:.2f}",
                                'Sortino': f"{res['sortino_ratio']:.2f}",
                                'Max DD %': f"{res['max_drawdown']:.2f}",
                                'Win Rate %': f"{res['win_rate']:.1f}",
                                'Trades': res['total_trades'],
                                'Profit Factor': f"{res['profit_factor']:.2f}",
                            })

                        st.dataframe(pd.DataFrame(cmp_data), use_container_width=True, hide_index=True)

                        # Returns bar chart
                        fig = go.Figure()
                        strats = list(comparison.keys())
                        returns_vals = [comparison[s]['total_return'] for s in strats]
                        colors = ['#2ecc71' if r > 0 else '#e74c3c' for r in returns_vals]
                        fig.add_trace(go.Bar(x=strats, y=returns_vals, marker_color=colors,
                                             text=[f"{r:.1f}%" for r in returns_vals], textposition='outside'))
                        fig.update_layout(title="Total Return by Strategy", yaxis_title="Return %",
                                          height=400, xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)

                        # Equity curves overlay
                        st.markdown("### Equity Curves Comparison")
                        fig_eq = go.Figure()
                        for name, res in comparison.items():
                            if 'equity_curve' in res and not res['equity_curve'].empty:
                                fig_eq.add_trace(go.Scatter(
                                    x=res['equity_curve'].index,
                                    y=res['equity_curve']['equity'],
                                    mode='lines', name=name
                                ))
                        fig_eq.add_hline(y=cmp_capital, line_dash="dash", line_color="gray")
                        fig_eq.update_layout(title=f"Equity Curves - {cmp_symbol}",
                                             yaxis_title="Portfolio Value", height=500)
                        st.plotly_chart(fig_eq, use_container_width=True)

                        # Export comparison
                        try:
                            cmp_csv_data = pd.DataFrame(cmp_data).to_csv(index=False)
                            st.download_button("📥 Download Strategy Comparison (CSV)", cmp_csv_data,
                                               f"strategy_comparison_{cmp_symbol}_{cmp_period}.csv", "text/csv",
                                               use_container_width=True, key="bt_cmp_export")
                        except Exception:
                            pass
                    else:
                        st.warning("No strategies produced results for this stock.")
                else:
                    st.error("Could not fetch data or insufficient data.")

