"""Page: Multi-Strategy"""
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

def render_multi_strategy_page():
    st.markdown("## 🎯 Multi-Strategy Screener")
    st.info("Evaluate stocks against 16 advanced strategies across multiple categories")

    tab1, tab2, tab3 = st.tabs(["Single Stock Analysis", "Batch Screening", "Strategy Explorer"])

    with tab1:
        st.markdown("### Analyze a Stock with All Strategies")

        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            exchange = st.selectbox("Exchange", st.session_state.exchange_handler.get_supported_exchanges(), key="ms_exchange")
            stock_lists = st.session_state.exchange_handler.get_stock_lists(exchange)
            if stock_lists:
                list_name = st.selectbox("Stock List", list(stock_lists.keys()), key="ms_list")
                stocks = stock_lists[list_name]
            else:
                stocks = st.session_state.nse_stocks
            symbol = st.selectbox("Stock", stocks, key="ms_symbol")

        with col2:
            min_confidence = st.slider("Minimum Confidence %", 30, 90, 60, 5, key="ms_conf")

        with col3:
            category_filter = st.selectbox("Category Filter",
                ["All"] + st.session_state.advanced_strategy_engine.get_categories(),
                key="ms_cat")

        if st.button("🔍 Run Multi-Strategy Analysis", type="primary", key="ms_run"):
            with st.spinner(f"Evaluating {symbol} against all strategies..."):
                df = st.session_state.exchange_handler.get_stock_data(symbol, exchange)

                if df is not None and len(df) >= 50:
                    screen_result = st.session_state.advanced_strategy_engine.multi_strategy_screen(
                        df, min_strategies_passing=2, min_confidence=min_confidence
                    )

                    if screen_result:
                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Strategies Passing", f"{screen_result['strategies_passing']}/{screen_result['strategies_evaluated']}")
                        with col2:
                            st.metric("Best Strategy", screen_result['best_strategy'])
                        with col3:
                            st.metric("Max Confidence", f"{screen_result['max_confidence']:.1f}%")
                        with col4:
                            qualifies_text = "✅ QUALIFIES" if screen_result['qualifies'] else "❌ DOES NOT QUALIFY"
                            st.metric("Verdict", qualifies_text)

                        st.markdown("---")

                        # Results table
                        all_results = screen_result['all_results']
                        if category_filter != "All":
                            all_results = [r for r in all_results if r['category'] == category_filter]

                        if all_results:
                            df_results = pd.DataFrame(all_results)
                            df_results = df_results[['strategy', 'category', 'confidence', 'conditions_met', 'total_conditions', 'risk_level']]
                            df_results.columns = ['Strategy', 'Category', 'Confidence %', 'Met', 'Total', 'Risk Level']

                            # Color code confidence
                            st.dataframe(
                                df_results.style.background_gradient(
                                    subset=['Confidence %'], cmap='RdYlGn', vmin=0, vmax=100
                                ),
                                use_container_width=True,
                                hide_index=True
                            )

                            # Bar chart
                            fig = go.Figure()
                            colors = ['#2ecc71' if r['confidence'] >= min_confidence else '#e74c3c' for r in all_results]
                            fig.add_trace(go.Bar(
                                x=[r['strategy'] for r in all_results],
                                y=[r['confidence'] for r in all_results],
                                marker_color=colors,
                                text=[f"{r['confidence']:.0f}%" for r in all_results],
                                textposition='outside'
                            ))
                            fig.add_hline(y=min_confidence, line_dash="dash", line_color="orange",
                                          annotation_text=f"Min Confidence: {min_confidence}%")
                            fig.update_layout(
                                title="Strategy Confidence Scores",
                                xaxis_title="Strategy", yaxis_title="Confidence %",
                                height=500, xaxis_tickangle=-45
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            # Category summary
                            st.markdown("#### Category Summary")
                            categories = {}
                            for r in screen_result['all_results']:
                                cat = r['category']
                                if cat not in categories:
                                    categories[cat] = []
                                categories[cat].append(r['confidence'])

                            cat_cols = st.columns(len(categories))
                            for i, (cat, confs) in enumerate(categories.items()):
                                with cat_cols[i]:
                                    avg_conf = np.mean(confs)
                                    st.metric(cat, f"{avg_conf:.1f}%", delta=f"{len([c for c in confs if c >= min_confidence])} passing")
                else:
                    st.error("Could not fetch stock data or insufficient data points.")

    with tab2:
        st.markdown("### Batch Screen Stocks")

        col1, col2 = st.columns(2)
        with col1:
            batch_exchange = st.selectbox("Exchange", st.session_state.exchange_handler.get_supported_exchanges(), key="ms_batch_ex")
            batch_lists = st.session_state.exchange_handler.get_stock_lists(batch_exchange)
            if batch_lists:
                batch_list_options = list(batch_lists.keys()) + [f"All {batch_exchange} Stocks"]
                batch_list_name = st.selectbox("Stock List", batch_list_options, key="ms_batch_list")
                if batch_list_name.startswith("All "):
                    all_stocks = st.session_state.exchange_handler.load_exchange_stocks(batch_exchange)
                    if all_stocks:
                        batch_stocks = all_stocks
                    else:
                        merged = []
                        for syms in batch_lists.values():
                            merged.extend(syms)
                        batch_stocks = list(dict.fromkeys(merged))
                else:
                    batch_stocks = batch_lists[batch_list_name]
            else:
                batch_stocks = st.session_state.nse_stocks
        with col2:
            max_stocks = st.slider("Max Stocks to Screen", 5, len(batch_stocks), min(15, len(batch_stocks)), key="ms_batch_max")
            batch_min_conf = st.slider("Min Confidence %", 30, 90, 60, 5, key="ms_batch_conf")

        if st.button("🚀 Run Batch Screening", type="primary", key="ms_batch_run"):
            results_list = []
            progress = st.progress(0)

            symbols_to_screen = batch_stocks[:max_stocks]
            progress.progress(0.05, text=f"Fetching realtime data for {len(symbols_to_screen)} stocks...")

            data_map = st.session_state.exchange_handler.get_bulk_stock_data(
                symbols_to_screen,
                batch_exchange,
                period="2y",
                interval="1d",
                include_live_quote=True,
            )
            bulk_meta = st.session_state.exchange_handler.get_last_bulk_meta()
            if bulk_meta:
                latency_ema = bulk_meta.get('latency_ema')
                latency_ema_text = f"{float(latency_ema):.3f}s" if latency_ema is not None else "N/A"
                st.caption(
                    f"Realtime fetch: {bulk_meta.get('fetched', 0)}/{bulk_meta.get('requested', len(symbols_to_screen))} symbols | "
                    f"Workers: {bulk_meta.get('workers_min', 1)}-{bulk_meta.get('workers_max', 1)} | "
                    f"Adaptive latency EMA: {latency_ema_text}"
                )

            for idx, sym in enumerate(symbols_to_screen):
                progress.progress(0.05 + ((idx + 1) / max(1, len(symbols_to_screen))) * 0.95, text=f"Screening {sym}...")
                try:
                    df = data_map.get(str(sym).strip().upper())
                    if df is not None and len(df) >= 50:
                        screen = st.session_state.advanced_strategy_engine.multi_strategy_screen(
                            df, min_confidence=batch_min_conf
                        )
                        if screen:
                            curr_price = df['Close'].iloc[-1]
                            results_list.append({
                                'Symbol': sym,
                                'Exchange': batch_exchange,
                                'Price': f"{curr_price:.2f}",
                                'Strategies Passing': screen['strategies_passing'],
                                'Best Strategy': screen['best_strategy'],
                                'Max Confidence': f"{screen['max_confidence']:.1f}%",
                                'Avg Confidence': f"{screen['avg_confidence']:.1f}%",
                                'Qualifies': "✅" if screen['qualifies'] else "❌"
                            })
                except Exception:
                    pass

            progress.empty()

            if results_list:
                df_batch = pd.DataFrame(results_list)
                df_batch = df_batch.sort_values('Strategies Passing', ascending=False)
                st.dataframe(df_batch, use_container_width=True, hide_index=True)
                st.success(f"Screened {len(results_list)} stocks. {len([r for r in results_list if r['Qualifies'] == '✅'])} qualified.")
            else:
                st.warning("No results found.")

    with tab3:
        st.markdown("### Strategy Explorer")
        st.markdown("Browse all 16 advanced strategies and their details")

        categories = st.session_state.advanced_strategy_engine.get_categories()
        for cat in categories:
            st.markdown(f"#### {cat}")
            strategies = st.session_state.advanced_strategy_engine.get_strategies_by_category(cat)
            for name, details in strategies.items():
                with st.expander(f"📋 {name} ({details.get('risk_level', 'N/A')} Risk)"):
                    st.write(f"**Description:** {details.get('description', 'N/A')}")
                    st.write(f"**Timeframe:** {details.get('timeframe', 'N/A')}")
                    st.write(f"**Risk Level:** {details.get('risk_level', 'N/A')}")
                    st.write(f"**Number of Rules:** {len(details.get('rules', []))}")
                    st.markdown("**Rules:**")
                    for i, rule in enumerate(details.get('rules', []), 1):
                        st.write(f"  {i}. `{rule['indicator']}` → {rule['condition']} (weight: {rule.get('weight', 1.0)})")

