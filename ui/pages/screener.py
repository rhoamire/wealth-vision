"""Page: Stock Screener"""
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

def render_screener_page():
    st.markdown("## Multi-Exchange Stock Screener")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        selected_exchange = st.selectbox(
            "Select Exchange",
            st.session_state.exchange_handler.get_supported_exchanges(),
            index=0,
            key="screener_exchange"
        )
        st.session_state.current_exchange = selected_exchange
    
    with col2:
        # Get available stock lists for the exchange
        exchange_lists = st.session_state.exchange_handler.get_stock_lists(selected_exchange)
        list_options = list(exchange_lists.keys())
        if selected_exchange == "NSE":
            list_options.append("All NSE Stocks")
        # Always add "All Exchange Stocks" option
        list_options.append(f"All {selected_exchange} Stocks")
        stock_list_option = st.selectbox("Stock Universe", list_options) if list_options else "Default"
    
    with col3:
        strategy_name = st.selectbox(
            "Select Strategy",
            list(st.session_state.advanced_strategy_engine.strategies.keys()),
            key="screener_strategy"
        )
    
    with col4:
        min_confidence = st.slider("Min Confidence %", 30, 90, 60)
    
    # Display strategy info
    strat = st.session_state.advanced_strategy_engine.strategies[strategy_name]
    exchange_info = st.session_state.exchange_handler.get_exchange_info(selected_exchange)
    st.info(f"**Strategy:** {strat['description']} | **Risk:** {strat.get('risk_level', 'N/A')} | "
            f"**Timeframe:** {strat.get('timeframe', 'N/A')} | **Exchange:** {exchange_info.get('name', selected_exchange)}")

    stocks_to_scan = resolve_stock_universe(selected_exchange, stock_list_option, exchange_lists)
    
    if st.button("Start Screening", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        if not stocks_to_scan:
            progress_bar.empty()
            status_text.empty()
            st.warning("No stocks available in this universe. Try another list or exchange.")
        else:
            results = []

            total_symbols = len(stocks_to_scan)
            status_text.text(f"Fetching realtime data for {total_symbols} symbols...")
            progress_bar.progress(0.05)

            data_map = st.session_state.exchange_handler.get_bulk_stock_data(
                stocks_to_scan,
                selected_exchange,
                period="2y",
                interval="1d",
                include_live_quote=True,
            )
            bulk_meta = st.session_state.exchange_handler.get_last_bulk_meta()
            if bulk_meta:
                status_text.text(
                    f"Fetched {bulk_meta.get('fetched', 0)}/{bulk_meta.get('requested', total_symbols)} symbols "
                    f"using {bulk_meta.get('workers_min', 1)}-{bulk_meta.get('workers_max', 1)} workers"
                )

            for i, symbol in enumerate(stocks_to_scan):
                status_text.text(f"Scoring {symbol}... ({i+1}/{total_symbols})")

                try:
                    df = data_map.get(str(symbol).strip().upper())
                    if df is None:
                        continue

                    result = st.session_state.advanced_strategy_engine.evaluate_strategy(df, strategy_name)

                    if result and result['confidence'] >= min_confidence:
                        current_price = float(df['Close'].iloc[-1])

                        # Quick price levels for shortlist display.
                        try:
                            prediction = st.session_state.price_predictor.predict_target_price(
                                df, sentiment_score=0, fundamental_score=50
                            )
                            buy_price = float(prediction['buy_price']) if prediction else current_price * 0.98
                            target_price = float(prediction['target_price']) if prediction else current_price * 1.05
                            stop_loss = float(prediction['stop_loss']) if prediction else current_price * 0.97
                            expected_return = float(prediction['expected_return']) if prediction else 5.0
                            pred_confidence = float(prediction['confidence']) if prediction else 50.0
                        except Exception:
                            buy_price = current_price * 0.98
                            target_price = current_price * 1.05
                            stop_loss = current_price * 0.97
                            expected_return = 5.0
                            pred_confidence = 50.0

                        results.append({
                            'Symbol': symbol,
                            'Exchange': selected_exchange,
                            'Current Price': current_price,
                            'Buy Price': buy_price,
                            'Target Price': target_price,
                            'Stop Loss': stop_loss,
                            'Expected Return %': expected_return,
                            'Strategy Confidence': result['confidence'],
                            'Prediction Confidence': pred_confidence,
                            'Conditions Met': result['conditions_met'],
                            'Total Conditions': result['total_conditions'],
                            'Category': result.get('category', 'N/A'),
                            'Risk Level': result.get('risk_level', 'N/A'),
                        })
                except Exception:
                    pass

                progress_bar.progress(0.05 + ((i + 1) / max(1, total_symbols)) * 0.95)

            status_text.text("Screening Complete!")

            if results:
                st.session_state.screened_stocks = pd.DataFrame(results).sort_values(
                    'Strategy Confidence', ascending=False
                )
                st.success(f"Found {len(results)} stocks matching criteria!")
            else:
                st.warning("No stocks found. Try adjusting filters.")

    st.markdown("---")
    st.markdown("## Best Possible Buys (By Exchange)")
    st.caption("Ranks opportunities using strategy confidence, expected upside, risk/reward, and prediction confidence.")

    best_use_all_exchange = st.checkbox(
        f"Use all stocks from {selected_exchange} for best buys",
        value=stock_list_option.startswith("All "),
        key="best_buy_use_all_exchange",
    )

    if best_use_all_exchange:
        best_buy_universe = resolve_stock_universe(
            selected_exchange,
            f"All {selected_exchange} Stocks",
            exchange_lists,
        )
        best_universe_label = f"All {selected_exchange} Stocks"
    else:
        best_buy_universe = stocks_to_scan
        best_universe_label = stock_list_option

    universe_count = len(best_buy_universe)
    if universe_count > 0:
        st.caption(f"Universe selected: {best_universe_label} | Available symbols: {universe_count}")
    else:
        st.info("No symbols available for this exchange/list combination.")

    scan_limit_max = max(1, universe_count)
    default_scan_limit = min(75, scan_limit_max)

    bb_col1, bb_col2, bb_col3, bb_col4 = st.columns(4)
    with bb_col1:
        best_scan_all = st.checkbox(
            "Evaluate complete universe",
            value=universe_count <= 75 and universe_count > 0,
            disabled=universe_count == 0,
            key="best_buy_scan_all",
        )
    with bb_col2:
        best_scan_limit = int(
            st.number_input(
                "Stocks to evaluate",
                min_value=1,
                max_value=scan_limit_max,
                value=default_scan_limit,
                step=1,
                disabled=best_scan_all,
                key="best_buy_scan_limit",
            )
        )
    if best_scan_all and universe_count > 0:
        best_scan_limit = universe_count

    with bb_col3:
        best_top_n = int(
            st.number_input(
                "Top buy ideas",
                min_value=1,
                max_value=max(1, best_scan_limit),
                value=min(10, best_scan_limit),
                step=1,
                key="best_buy_top_n",
            )
        )
    with bb_col4:
        best_min_conf = st.slider(
            "Min confidence for best buys",
            30,
            90,
            max(55, min_confidence),
            key="best_buy_min_conf",
        )

    if st.button(
        "Find Best Possible Buys",
        type="primary",
        use_container_width=True,
        key="best_buy_scan_btn",
        disabled=universe_count == 0,
    ):
        if universe_count == 0:
            st.warning("No symbols available to evaluate.")
        else:
            symbols_to_evaluate = best_buy_universe[:best_scan_limit]
            progress_bar = st.progress(0)
            status_text = st.empty()
            best_results = []

            total_symbols = len(symbols_to_evaluate)
            status_text.text(f"Fetching realtime data for {total_symbols} symbols...")
            progress_bar.progress(0.05)

            data_map = st.session_state.exchange_handler.get_bulk_stock_data(
                symbols_to_evaluate,
                selected_exchange,
                period="2y",
                interval="1d",
                include_live_quote=True,
            )
            bulk_meta = st.session_state.exchange_handler.get_last_bulk_meta()
            if bulk_meta:
                status_text.text(
                    f"Fetched {bulk_meta.get('fetched', 0)}/{bulk_meta.get('requested', total_symbols)} symbols "
                    f"using {bulk_meta.get('workers_min', 1)}-{bulk_meta.get('workers_max', 1)} workers"
                )

            for i, symbol in enumerate(symbols_to_evaluate):
                status_text.text(f"Evaluating {symbol}... ({i+1}/{total_symbols})")

                try:
                    df = data_map.get(str(symbol).strip().upper())
                    if df is None or len(df) < 50:
                        continue

                    result = st.session_state.advanced_strategy_engine.evaluate_strategy(df, strategy_name)
                    if not result or result.get('confidence', 0) < best_min_conf:
                        continue

                    current_price = float(df['Close'].iloc[-1])
                    try:
                        prediction = st.session_state.price_predictor.predict_target_price(
                            df, sentiment_score=0, fundamental_score=50
                        )
                    except Exception:
                        prediction = None

                    buy_price = float(prediction['buy_price']) if prediction else current_price * 0.98
                    target_price = float(prediction['target_price']) if prediction else current_price * 1.05
                    stop_loss = float(prediction['stop_loss']) if prediction else current_price * 0.97
                    expected_return = float(prediction['expected_return']) if prediction else ((target_price / current_price) - 1) * 100
                    pred_confidence = float(prediction['confidence']) if prediction else 50.0

                    risk_pct = ((current_price - stop_loss) / current_price) * 100 if current_price > 0 else 0
                    risk_pct = max(risk_pct, 0.01)
                    risk_reward = expected_return / risk_pct
                    buy_score = compute_buy_opportunity_score(
                        result.get('confidence', 0),
                        expected_return,
                        pred_confidence,
                        risk_reward,
                    )

                    best_results.append({
                        'Symbol': symbol,
                        'Exchange': selected_exchange,
                        'Current Price': current_price,
                        'Buy Price': buy_price,
                        'Target Price': target_price,
                        'Stop Loss': stop_loss,
                        'Expected Return %': expected_return,
                        'Risk %': risk_pct,
                        'Risk/Reward': risk_reward,
                        'Strategy Confidence': float(result.get('confidence', 0)),
                        'Prediction Confidence': pred_confidence,
                        'Buy Score': float(buy_score),
                        'Category': result.get('category', 'N/A'),
                        'Risk Level': result.get('risk_level', 'N/A'),
                    })
                except Exception:
                    pass

                progress_bar.progress(0.05 + ((i + 1) / max(1, total_symbols)) * 0.95)

            status_text.text("Best-buy scan complete!")

            if best_results:
                best_df = pd.DataFrame(best_results).sort_values(
                    ['Buy Score', 'Strategy Confidence', 'Expected Return %'],
                    ascending=[False, False, False],
                ).head(best_top_n).reset_index(drop=True)
                best_df.insert(0, 'Rank', np.arange(1, len(best_df) + 1))

                st.session_state.best_buy_opportunities = best_df
                st.session_state.best_buy_context = {
                    'exchange': selected_exchange,
                    'stock_list_option': best_universe_label,
                    'strategy': strategy_name,
                    'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'symbols_evaluated': len(symbols_to_evaluate),
                }
                st.success(f"Found {len(best_df)} high-potential buy opportunities.")
            else:
                st.session_state.best_buy_opportunities = None
                st.warning("No best-buy opportunities found. Try lowering confidence or scanning more stocks.")

    if st.session_state.best_buy_opportunities is not None:
        best_ctx = st.session_state.get('best_buy_context', {})
        best_df = st.session_state.best_buy_opportunities.copy()

        st.markdown("### Top Buy Opportunities")
        st.caption(
            f"Generated: {best_ctx.get('generated_at', 'N/A')} | "
            f"Exchange: {best_ctx.get('exchange', 'N/A')} | "
            f"Strategy: {best_ctx.get('strategy', 'N/A')} | "
            f"Symbols Evaluated: {best_ctx.get('symbols_evaluated', 0)}"
        )

        met_col1, met_col2, met_col3, met_col4 = st.columns(4)
        with met_col1:
            st.metric("Top Ideas", len(best_df))
        with met_col2:
            st.metric("Avg Buy Score", f"{best_df['Buy Score'].mean():.1f}")
        with met_col3:
            st.metric("Avg Expected Return", f"{best_df['Expected Return %'].mean():.2f}%")
        with met_col4:
            st.metric("Best Symbol", best_df.iloc[0]['Symbol'])

        best_currency = st.session_state.exchange_handler.get_currency_symbol(
            best_ctx.get('exchange', selected_exchange)
        )
        display_best_df = best_df.copy()
        display_best_df['Current Price'] = display_best_df['Current Price'].map(lambda x: f"{best_currency}{x:.2f}")
        display_best_df['Buy Price'] = display_best_df['Buy Price'].map(lambda x: f"{best_currency}{x:.2f}")
        display_best_df['Target Price'] = display_best_df['Target Price'].map(lambda x: f"{best_currency}{x:.2f}")
        display_best_df['Stop Loss'] = display_best_df['Stop Loss'].map(lambda x: f"{best_currency}{x:.2f}")
        display_best_df['Expected Return %'] = display_best_df['Expected Return %'].map(lambda x: f"{x:.2f}%")
        display_best_df['Risk %'] = display_best_df['Risk %'].map(lambda x: f"{x:.2f}%")
        display_best_df['Risk/Reward'] = display_best_df['Risk/Reward'].map(lambda x: f"{x:.2f}")
        display_best_df['Strategy Confidence'] = display_best_df['Strategy Confidence'].map(lambda x: f"{x:.1f}%")
        display_best_df['Prediction Confidence'] = display_best_df['Prediction Confidence'].map(lambda x: f"{x:.1f}%")
        display_best_df['Buy Score'] = display_best_df['Buy Score'].map(lambda x: f"{x:.1f}")

        st.dataframe(display_best_df, use_container_width=True, height=320)

        act_col1, act_col2 = st.columns(2)
        with act_col1:
            best_csv = best_df.to_csv(index=False)
            st.download_button(
                "📥 Download Best Buys",
                best_csv,
                f"best_buys_{best_ctx.get('exchange', selected_exchange)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True,
                key="best_buys_download",
            )

        with act_col2:
            selected_best_symbol = st.selectbox(
                "Select best buy for detailed analysis",
                best_df['Symbol'].tolist(),
                key="best_buy_detail_select",
            )

        if st.button("🔍 Analyze Selected Best Buy", type="primary", use_container_width=True, key="best_buy_analyze"):
            with st.spinner(f"Analyzing {selected_best_symbol}..."):
                analysis, news_data, prediction = run_detailed_stock_analysis(selected_best_symbol)

                if analysis:
                    st.success(f"Analysis complete! Scroll down to view detailed analysis of {selected_best_symbol}")
                    st.markdown("---")
                    st.markdown(f"## Detailed Analysis: {selected_best_symbol}")
                    _display_stock_analysis(analysis, news_data, prediction)
                else:
                    st.error("Failed to analyze stock")
    
    # Display results
    if st.session_state.screened_stocks is not None:
        st.markdown("---")
        st.markdown("## Results")
        
        df = st.session_state.screened_stocks
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Stocks", len(df))
        with col2:
            st.metric("Avg Strategy Confidence", f"{df['Strategy Confidence'].mean():.1f}%")
        with col3:
            high = len(df[df['Strategy Confidence'] >= 75])
            st.metric("High Confidence", high)
        with col4:
            st.metric("Exchange", st.session_state.current_exchange)
        
        # Format display columns
        display_df = df.copy()
        curr = st.session_state.exchange_handler.get_currency_symbol(st.session_state.current_exchange)
        display_df['Current Price'] = display_df['Current Price'].apply(lambda x: f"{curr}{x:.2f}")
        display_df['Buy Price'] = display_df['Buy Price'].apply(lambda x: f"{curr}{x:.2f}")
        display_df['Target Price'] = display_df['Target Price'].apply(lambda x: f"{curr}{x:.2f}")
        display_df['Stop Loss'] = display_df['Stop Loss'].apply(lambda x: f"{curr}{x:.2f}")
        display_df['Expected Return %'] = display_df['Expected Return %'].apply(lambda x: f"{x:.2f}%")
        display_df['Strategy Confidence'] = display_df['Strategy Confidence'].apply(lambda x: f"{x:.1f}%")
        display_df['Prediction Confidence'] = display_df['Prediction Confidence'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(display_df, use_container_width=True, height=400)
        
        # Action buttons
        col1, col2 = st.columns(2)
        
        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download Results",
                csv,
                f"screening_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col2:
            # Quick analyze option
            selected_symbol = st.selectbox(
                "Select stock to analyze in detail",
                df['Symbol'].tolist(),
                key="quick_analyze_select"
            )
        
        if st.button("🔍 Analyze Selected Stock", type="primary", use_container_width=True):
            with st.spinner(f"Analyzing {selected_symbol}..."):
                analysis, news_data, prediction = run_detailed_stock_analysis(selected_symbol)

                if analysis:
                    st.success(f"Analysis complete! Scroll down to view detailed analysis of {selected_symbol}")
                    st.markdown("---")
                    st.markdown(f"## Detailed Analysis: {selected_symbol}")
                    _display_stock_analysis(analysis, news_data, prediction)
                else:
                    st.error("Failed to analyze stock")

