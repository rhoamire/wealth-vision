"""Page: Market Overview"""
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

def render_market_overview_page():
    st.markdown("## 🌍 Market Overview")
    st.info("Global market indices, sector performance, and market breadth analysis")

    tab1, tab2, tab3, tab4 = st.tabs(["Global Indices", "Sector Performance", "Sector Rotation", "Correlation Finder"])

    with tab1:
        st.markdown("### Global Market Indices")

        if st.button("🔄 Refresh Indices", key="mo_refresh"):
            st.rerun()

        with st.spinner("Loading global indices..."):
            indices = st.session_state.market_overview.get_all_indices()
            st.caption(f"Live snapshot refreshed: {datetime.now().strftime('%H:%M:%S')}")

            if indices:
                cols = st.columns(len(indices))
                for i, (exchange, data) in enumerate(indices.items()):
                    with cols[i]:
                        change_color = "normal" if data['change_pct'] >= 0 else "inverse"
                        st.metric(
                            f"{data['name']}",
                            f"{data['currency']}{data['value']:,.2f}",
                            delta=f"{data['change_pct']:.2f}%",
                            delta_color=change_color
                        )

                st.markdown("---")

                # Index comparison chart
                st.markdown("### Index Performance (1 Month)")
                fig_idx = go.Figure()
                for exchange_name, data in indices.items():
                    fig_idx.add_trace(go.Bar(
                        x=[data['name']], y=[data['change_pct']],
                        name=data['name'],
                        marker_color='#2ecc71' if data['change_pct'] >= 0 else '#e74c3c',
                        text=f"{data['change_pct']:.2f}%", textposition='outside'
                    ))
                fig_idx.update_layout(title="Daily Change (%)", yaxis_title="Change %",
                                      height=400, showlegend=False)
                st.plotly_chart(fig_idx, use_container_width=True)
            else:
                st.warning("Could not fetch market indices.")

    with tab2:
        st.markdown("### Sector Performance")

        sp_exchange = st.selectbox("Exchange", ["NSE", "NYSE"], key="mo_sp_ex")

        if st.button("📊 Load Sector Data", type="primary", key="mo_sp_run"):
            with st.spinner(f"Loading sector performance for {sp_exchange}..."):
                sector_data = st.session_state.market_overview.get_sector_performance(sp_exchange)

                if sector_data:
                    # Sector bar chart
                    sectors = list(sector_data.keys())
                    avg_returns = [sector_data[s]['avg_return'] for s in sectors]
                    colors = ['#2ecc71' if r >= 0 else '#e74c3c' for r in avg_returns]

                    fig_sec = go.Figure()
                    fig_sec.add_trace(go.Bar(
                        x=sectors, y=avg_returns, marker_color=colors,
                        text=[f"{r:.2f}%" for r in avg_returns], textposition='outside'
                    ))
                    fig_sec.update_layout(title=f"Sector Performance (1 Month) - {sp_exchange}",
                                          yaxis_title="Avg Return %", height=400)
                    st.plotly_chart(fig_sec, use_container_width=True)

                    # Sector details
                    st.markdown("### Sector Details")
                    for sector, data in sector_data.items():
                        with st.expander(f"📊 {sector}"):
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Avg Return", f"{data['avg_return']:.2f}%")
                            with col2:
                                st.metric("Best Stock", f"{data['best']:.2f}%")
                            with col3:
                                st.metric("Worst Stock", f"{data['worst']:.2f}%")
                            with col4:
                                st.metric("Stocks Analyzed", data['stocks_count'])
                else:
                    st.warning("No sector data available.")

    with tab3:
        st.markdown("### Sector Rotation Analysis")
        st.markdown("Identifies which sectors are gaining/losing momentum across different timeframes")

        sr_exchange = st.selectbox("Exchange", ["NSE", "NYSE"], key="mo_sr_ex")

        if st.button("🔄 Analyze Rotation", type="primary", key="mo_sr_run"):
            with st.spinner(f"Analyzing sector rotation for {sr_exchange}..."):
                rotation = st.session_state.sector_rotation.analyze_rotation(sr_exchange)

                if rotation:
                    # Phase summary
                    st.markdown("### Sector Phases")
                    phase_cols = st.columns(min(len(rotation), 4))
                    for i, (sector, data) in enumerate(rotation.items()):
                        with phase_cols[i % 4]:
                            phase = data['phase']
                            phase_emojis = {
                                "Strong Uptrend": "🟢",
                                "Early Uptrend": "🔵",
                                "Recovery": "🟡",
                                "Topping Out": "🟠",
                                "Downtrend": "🔴",
                                "Consolidation": "⚪"
                            }
                            emoji = phase_emojis.get(phase, "⚪")
                            st.metric(f"{emoji} {sector}", phase,
                                      delta=f"Score: {data['momentum_score']:.2f}")

                    st.markdown("---")

                    # Rotation heatmap
                    st.markdown("### Return Heatmap")
                    sectors_list = list(rotation.keys())
                    periods_list = list(next(iter(rotation.values()))['returns'].keys()) if rotation else []

                    if periods_list:
                        z_data = []
                        for sector in sectors_list:
                            row = [rotation[sector]['returns'].get(p, 0) for p in periods_list]
                            z_data.append(row)

                        fig_hm = go.Figure(data=go.Heatmap(
                            z=z_data, x=periods_list, y=sectors_list,
                            colorscale='RdYlGn', zmid=0,
                            text=[[f"{v:.2f}%" for v in row] for row in z_data],
                            texttemplate="%{text}", textfont={"size": 12}
                        ))
                        fig_hm.update_layout(title=f"Sector Returns Heatmap - {sr_exchange}", height=400)
                        st.plotly_chart(fig_hm, use_container_width=True)

                    # Momentum ranking
                    st.markdown("### Momentum Ranking")
                    ranked = sorted(rotation.items(), key=lambda x: x[1]['momentum_score'], reverse=True)
                    rank_data = []
                    for rank, (sector, data) in enumerate(ranked, 1):
                        rank_data.append({
                            'Rank': rank,
                            'Sector': sector,
                            'Phase': data['phase'],
                            'Momentum Score': f"{data['momentum_score']:.2f}",
                            '1W Return': f"{data['returns'].get('1W', 0):.2f}%",
                            '1M Return': f"{data['returns'].get('1M', 0):.2f}%",
                            '3M Return': f"{data['returns'].get('3M', 0):.2f}%",
                        })
                    st.dataframe(pd.DataFrame(rank_data), use_container_width=True, hide_index=True)
                else:
                    st.warning("No rotation data available.")

    with tab4:
        st.markdown("### 🔗 Correlation Finder")
        st.markdown("Discover correlated and uncorrelated stocks for diversification or pair trading")

        col1, col2, col3 = st.columns(3)
        with col1:
            corr_exchange = st.selectbox("Exchange", st.session_state.exchange_handler.get_supported_exchanges(), key="corr_ex")
            corr_lists = st.session_state.exchange_handler.get_stock_lists(corr_exchange)
            if corr_lists:
                corr_list_name = st.selectbox("Stock List", list(corr_lists.keys()), key="corr_list")
                corr_available = corr_lists[corr_list_name]
            else:
                corr_available = st.session_state.nse_stocks

        with col2:
            corr_stocks = st.multiselect(
                "Select Stocks (3-15)", corr_available,
                default=corr_available[:6] if len(corr_available) >= 6 else corr_available[:3],
                key="corr_stocks"
            )

        with col3:
            corr_period = st.selectbox("Period", ["3mo", "6mo", "1y", "2y"], index=2, key="corr_period")
            corr_method = st.selectbox("Method", ["pearson", "spearman", "kendall"], key="corr_method")

        if st.button("🔍 Find Correlations", type="primary", key="corr_run") and len(corr_stocks) >= 3:
            with st.spinner("Computing correlations..."):
                returns_dict = {}
                for sym in corr_stocks[:15]:
                    try:
                        data = st.session_state.exchange_handler.get_stock_data(sym, corr_exchange, period=corr_period)
                        if data is not None and len(data) > 20:
                            rets = data['Close'].pct_change().dropna()
                            if isinstance(rets, pd.DataFrame):
                                rets = rets.iloc[:, 0]
                            if hasattr(rets.index, 'tz') and rets.index.tz is not None:
                                rets = rets.copy()
                                rets.index = rets.index.tz_localize(None)
                            returns_dict[sym] = rets
                    except:
                        pass

                if len(returns_dict) >= 3:
                    returns_df = pd.DataFrame(returns_dict).dropna()
                    corr_matrix = returns_df.corr(method=corr_method)

                    # Correlation heatmap
                    st.markdown("### Correlation Matrix")
                    fig_corr = go.Figure(data=go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.index,
                        colorscale='RdYlGn', zmin=-1, zmax=1,
                        text=corr_matrix.round(2).values,
                        texttemplate="%{text}",
                        textfont={"size": 11}
                    ))
                    fig_corr.update_layout(title=f"Correlation Matrix ({corr_method.title()})", height=500)
                    st.plotly_chart(fig_corr, use_container_width=True)

                    # Top correlated and uncorrelated pairs
                    st.markdown("### Pair Analysis")
                    pairs = []
                    symbols = list(corr_matrix.columns)
                    for i in range(len(symbols)):
                        for j in range(i + 1, len(symbols)):
                            pairs.append({
                                'Stock A': symbols[i],
                                'Stock B': symbols[j],
                                'Correlation': corr_matrix.iloc[i, j]
                            })

                    pairs_df = pd.DataFrame(pairs).sort_values('Correlation', ascending=False)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### 🟢 Most Correlated (move together)")
                        top_corr = pairs_df.head(5).copy()
                        top_corr['Correlation'] = top_corr['Correlation'].apply(lambda x: f"{x:.3f}")
                        st.dataframe(top_corr, use_container_width=True, hide_index=True)

                    with col2:
                        st.markdown("#### 🔵 Least Correlated (diversification)")
                        low_corr = pairs_df.sort_values('Correlation', key=abs).head(5).copy()
                        low_corr['Correlation'] = low_corr['Correlation'].apply(lambda x: f"{x:.3f}")
                        st.dataframe(low_corr, use_container_width=True, hide_index=True)

                    st.markdown("#### 🔴 Most Negatively Correlated (hedge)")
                    neg_corr = pairs_df.tail(5).copy()
                    neg_corr['Correlation'] = neg_corr['Correlation'].apply(lambda x: f"{x:.3f}")
                    st.dataframe(neg_corr, use_container_width=True, hide_index=True)

                    # Scatter plot for top pair
                    if len(pairs_df) > 0:
                        st.markdown("### Scatter Plot - Top Correlated Pair")
                        top_pair = pairs_df.iloc[0]
                        sym_a, sym_b = top_pair['Stock A'], top_pair['Stock B']
                        fig_scatter = go.Figure()
                        fig_scatter.add_trace(go.Scatter(
                            x=returns_df[sym_a] * 100,
                            y=returns_df[sym_b] * 100,
                            mode='markers',
                            marker=dict(color='#667eea', size=4, opacity=0.5),
                            name=f"{sym_a} vs {sym_b}"
                        ))
                        fig_scatter.update_layout(
                            title=f"{sym_a} vs {sym_b} Daily Returns (Corr: {float(top_pair['Correlation']):.3f})",
                            xaxis_title=f"{sym_a} Return %",
                            yaxis_title=f"{sym_b} Return %",
                            height=400
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)

                    # Export
                    try:
                        corr_csv = corr_matrix.to_csv()
                        st.download_button("📥 Download Correlation Matrix (CSV)", corr_csv,
                                           f"correlation_{corr_exchange}_{corr_period}.csv", "text/csv",
                                           use_container_width=True, key="corr_export")
                    except Exception:
                        pass
                else:
                    st.error("Need at least 3 stocks with valid data.")

