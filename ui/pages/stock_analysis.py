"""Page: Stock Analysis"""
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

def render_stock_analysis_page():
    st.markdown("## Detailed Stock Analysis")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        symbol = st.selectbox("Select Stock", st.session_state.nse_stocks)
    
    with col2:
        if st.button("🔍 Analyze", type="primary", use_container_width=True):
            with st.spinner(f"Analyzing {symbol}..."):
                analysis, _, _ = run_detailed_stock_analysis(symbol)
                if not analysis:
                    st.error("Failed to analyze stock")

    history_entries = st.session_state.get("analysis_history", [])
    if history_entries:
        st.markdown("### Recent Analysis History")

        preview_rows = list(reversed(history_entries[-12:]))
        history_df = pd.DataFrame(preview_rows)
        display_history_df = history_df.reindex(columns=[
            "timestamp",
            "symbol",
            "recommendation",
            "price",
            "prediction_confidence",
            "expected_return",
            "news_sentiment",
        ]).copy()
        display_history_df["price"] = display_history_df["price"].map(lambda x: f"₹{float(x):.2f}")
        display_history_df["prediction_confidence"] = display_history_df["prediction_confidence"].map(
            lambda x: f"{float(x):.1f}%"
        )
        display_history_df["expected_return"] = display_history_df["expected_return"].map(
            lambda x: f"{float(x):.2f}%"
        )
        display_history_df = display_history_df.rename(columns={
            "timestamp": "Time",
            "symbol": "Symbol",
            "recommendation": "Call",
            "price": "Price",
            "prediction_confidence": "Prediction Confidence",
            "expected_return": "Expected Return",
            "news_sentiment": "News Sentiment",
        })
        st.dataframe(display_history_df, use_container_width=True, hide_index=True, height=260)

        history_symbols = []
        for row in preview_rows:
            sym = row.get("symbol")
            if sym and sym not in history_symbols:
                history_symbols.append(sym)

        hs_col1, hs_col2, hs_col3 = st.columns([2, 1, 1])
        with hs_col3:
            history_csv = pd.DataFrame(preview_rows).to_csv(index=False)
            st.download_button(
                "Download History",
                history_csv,
                f"analysis_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True,
                key="analysis_history_export",
            )

        if history_symbols:
            with hs_col1:
                recent_symbol = st.selectbox(
                    "Quick Re-Analyze",
                    history_symbols,
                    key="recent_symbol_reanalyze",
                )
            with hs_col2:
                if st.button("Re-Analyze", use_container_width=True, key="recent_symbol_reanalyze_btn"):
                    with st.spinner(f"Re-analyzing {recent_symbol}..."):
                        analysis, _, _ = run_detailed_stock_analysis(recent_symbol)
                        if not analysis:
                            st.error("Could not run analysis for selected symbol.")
        else:
            st.info("History is available, but no valid symbols were found for quick re-analysis.")

        st.markdown("---")
    
    if 'current_analysis' in st.session_state:
        analysis = st.session_state.current_analysis
        news_data = st.session_state.get('current_news', {})
        prediction = st.session_state.get('current_prediction', None)
        
        # Header info
        st.markdown(f"### {analysis['company_name']} ({analysis['symbol']})")
        st.markdown(f"**Sector:** {analysis['sector']} | **Industry:** {analysis['industry']}")

        # Quick action buttons
        qa_col1, qa_col2, qa_col3 = st.columns([1, 1, 2])
        with qa_col1:
            if st.button("⭐ Add to Watchlist", key="qa_add_wl", use_container_width=True):
                entry = {'symbol': analysis['symbol'], 'exchange': st.session_state.current_exchange}
                existing = [(item.get('symbol') if isinstance(item, dict) else item)
                            for item in st.session_state.watchlist]
                if analysis['symbol'] not in existing:
                    st.session_state.watchlist.append(entry)
                    if st.session_state.auth_username and st.session_state.auth_username != "__guest__":
                        try: save_user_watchlist(st.session_state.auth_username, st.session_state.watchlist)
                        except: pass
                    st.success(f"Added {analysis['symbol']} to watchlist!")
                else:
                    st.info("Already in watchlist.")
        with qa_col2:
            if st.button("🔔 Set Alert", key="qa_set_alert", use_container_width=True):
                st.session_state.current_page = "Price Alerts"
                st.rerun()
        
        # AI Recommendation
        rec = analysis['recommendation']
        st.markdown(f"<div class='recommendation-{rec['action'].lower().replace(' ', '-')}'>"
                   f"AI RECOMMENDATION: {rec['action']} (Score: {rec['score']})</div>",
                   unsafe_allow_html=True)
        
        st.markdown("**Reasons:**")
        for reason in rec['reasons']:
            st.markdown(f"- {reason}")
        
        st.markdown("---")
        
        # ============ PRICE PREDICTION SECTION ============
        if prediction:
            st.markdown("## Price Prediction & Trading Levels")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"""
                    <div class='prediction-card'>
                        <h3 style='margin-top: 0;'>Target Price Prediction</h3>
                        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;'>
                            <div>
                                <p style='margin: 0; opacity: 0.8;'>Conservative Target</p>
                                <h2 style='margin: 0.5rem 0;'>₹{prediction['conservative_target']:.2f}</h2>
                            </div>
                            <div>
                                <p style='margin: 0; opacity: 0.8;'>Primary Target</p>
                                <h2 style='margin: 0.5rem 0;'>₹{prediction['target_price']:.2f}</h2>
                            </div>
                            <div>
                                <p style='margin: 0; opacity: 0.8;'>Aggressive Target</p>
                                <h2 style='margin: 0.5rem 0;'>₹{prediction['aggressive_target']:.2f}</h2>
                            </div>
                            <div>
                                <p style='margin: 0; opacity: 0.8;'>Expected Return</p>
                                <h2 style='margin: 0.5rem 0;'>{prediction['expected_return']:.2f}%</h2>
                            </div>
                        </div>
                        <p style='margin-top: 1rem; opacity: 0.9;'>
                            <strong>Confidence:</strong> {prediction['confidence']:.1f}% | 
                            <strong>Time Horizon:</strong> {prediction['time_horizon']} days
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Trading levels
                st.markdown("### Key Levels")
                st.metric("Current Price", f"₹{prediction['current_price']:.2f}")
                st.metric("Buy Price", f"₹{prediction['buy_price']:.2f}", 
                         delta=f"{((prediction['buy_price']/prediction['current_price']-1)*100):.2f}%")
                st.metric("Sell Price", f"₹{prediction['sell_price']:.2f}",
                         delta=f"{((prediction['sell_price']/prediction['current_price']-1)*100):.2f}%")
                st.metric("Stop Loss", f"₹{prediction['stop_loss']:.2f}",
                         delta=f"{((prediction['stop_loss']/prediction['current_price']-1)*100):.2f}%")
            
            # Detailed prediction metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Risk/Reward", f"{prediction['risk_reward']:.2f}")
            with col2:
                st.metric("Support", f"₹{prediction['support']:.2f}")
            with col3:
                st.metric("Resistance", f"₹{prediction['resistance']:.2f}")
            with col4:
                st.metric("Pivot Point", f"₹{prediction['pivot_points']['pivot']:.2f}")

            st.markdown("#### Trade Planner (Risk-Based Position Sizing)")
            tp_col1, tp_col2, tp_col3 = st.columns(3)
            with tp_col1:
                plan_capital = st.number_input(
                    "Account Capital (₹)",
                    min_value=1000.0,
                    value=100000.0,
                    step=5000.0,
                    key=f"tp_cap_{analysis['symbol']}",
                )
            with tp_col2:
                plan_risk_pct = st.slider(
                    "Risk Per Trade (%)",
                    min_value=0.25,
                    max_value=5.0,
                    value=1.0,
                    step=0.25,
                    key=f"tp_risk_{analysis['symbol']}",
                )
            with tp_col3:
                entry_basis = st.radio(
                    "Entry Basis",
                    ["Buy Price", "Current Price"],
                    horizontal=True,
                    key=f"tp_entry_{analysis['symbol']}",
                )

            entry_price = prediction['buy_price'] if entry_basis == "Buy Price" else prediction['current_price']
            trade_plan = build_trade_plan(
                account_capital=plan_capital,
                risk_pct=plan_risk_pct,
                entry_price=entry_price,
                stop_loss=prediction['stop_loss'],
                target_price=prediction['target_price'],
            )

            if trade_plan['valid']:
                pc1, pc2, pc3, pc4 = st.columns(4)
                with pc1:
                    st.metric("Risk Budget", f"₹{trade_plan['risk_budget']:.2f}")
                with pc2:
                    st.metric("Position Size", f"{trade_plan['shares']} shares")
                with pc3:
                    st.metric("Position Value", f"₹{trade_plan['position_value']:.2f}")
                with pc4:
                    st.metric("Max Loss", f"₹{trade_plan['potential_loss']:.2f}")

                pc5, pc6, pc7, pc8 = st.columns(4)
                with pc5:
                    st.metric("Potential Profit", f"₹{trade_plan['potential_profit']:.2f}")
                with pc6:
                    st.metric("Planned R:R", f"{trade_plan['risk_reward']:.2f}")
                with pc7:
                    st.metric("Capital Used", f"{trade_plan['capital_utilization_pct']:.1f}%")
                with pc8:
                    st.metric("Risk Used", f"{trade_plan['risk_utilization_pct']:.1f}%")

                if trade_plan['capital_limited']:
                    st.info("Position size is capital-limited before full risk budget could be deployed.")
                if trade_plan['potential_profit'] <= 0:
                    st.warning("Target price is not above entry; expected reward is not favorable for a long setup.")
            else:
                st.warning(trade_plan['error'])

            st.markdown("#### Scenario Simulator (What-If)")
            sc_col1, sc_col2, sc_col3, sc_col4 = st.columns(4)
            with sc_col1:
                scenario_entry = st.number_input(
                    "Entry Price",
                    min_value=0.01,
                    value=float(prediction['buy_price']),
                    step=1.0,
                    key=f"scenario_entry_{analysis['symbol']}",
                )
            with sc_col2:
                scenario_stop = st.number_input(
                    "Stop Loss",
                    min_value=0.01,
                    value=float(prediction['stop_loss']),
                    step=1.0,
                    key=f"scenario_stop_{analysis['symbol']}",
                )
            with sc_col3:
                scenario_target = st.number_input(
                    "Target Price",
                    min_value=0.01,
                    value=float(prediction['target_price']),
                    step=1.0,
                    key=f"scenario_target_{analysis['symbol']}",
                )
            with sc_col4:
                scenario_win_prob = st.slider(
                    "Win Probability (%)",
                    min_value=5,
                    max_value=95,
                    value=int(min(max(prediction.get('confidence', 50), 5), 95)),
                    key=f"scenario_win_prob_{analysis['symbol']}",
                )

            scenario_capital = st.number_input(
                "Scenario Capital (₹)",
                min_value=1000.0,
                value=float(plan_capital),
                step=5000.0,
                key=f"scenario_capital_{analysis['symbol']}",
            )

            scenario = compute_trade_scenario(
                entry_price=float(scenario_entry),
                stop_loss=float(scenario_stop),
                target_price=float(scenario_target),
                win_probability_pct=float(scenario_win_prob),
                capital=float(scenario_capital),
            )

            if scenario['valid']:
                sm1, sm2, sm3, sm4 = st.columns(4)
                with sm1:
                    st.metric("Shares", f"{scenario['shares']}")
                with sm2:
                    st.metric("R:R", f"{scenario['risk_reward']:.2f}")
                with sm3:
                    st.metric("Expected Value", f"₹{scenario['expected_value']:.2f}")
                with sm4:
                    st.metric("Breakeven Win %", f"{scenario['breakeven_win_rate']:.1f}%")

                sm5, sm6, sm7 = st.columns(3)
                with sm5:
                    st.metric("Win Outcome", f"₹{scenario['pnl_if_win']:.2f}")
                with sm6:
                    st.metric("Loss Outcome", f"-₹{scenario['pnl_if_loss']:.2f}")
                with sm7:
                    st.metric("Expectancy (R)", f"{scenario['expectancy_r']:.2f}")

                scenario_fig = go.Figure()
                scenario_fig.add_trace(
                    go.Bar(
                        x=["Win Case", "Loss Case", "Expected"],
                        y=[
                            scenario['pnl_if_win'],
                            -scenario['pnl_if_loss'],
                            scenario['expected_value'],
                        ],
                        marker_color=["#2ecc71", "#e74c3c", "#3498db"],
                    )
                )
                scenario_fig.update_layout(
                    title="Scenario Payoff Profile",
                    height=280,
                    margin=dict(l=10, r=10, t=40, b=10),
                    yaxis_title="PnL (₹)",
                )
                st.plotly_chart(scenario_fig, use_container_width=True)
            else:
                st.warning(scenario['error'])
            
            # Score breakdown
            st.markdown("#### Prediction Score Breakdown")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                tech_score = prediction['technical_score']
                st.metric("Technical Score", f"{tech_score:.1f}/100")
                st.progress(min(abs(tech_score)/100, 1.0))
            
            with col2:
                sent_score = prediction['sentiment_score']
                st.metric("Sentiment Score", f"{sent_score:.1f}/100")
                st.progress(min(abs(sent_score)/100, 1.0))
            
            with col3:
                fund_score = prediction['fundamental_score']
                st.metric("Fundamental Score", f"{fund_score:.1f}/100")
                st.progress(min(abs(fund_score)/100, 1.0))
            
            st.markdown("---")
        
        # ============ NEWS SENTIMENT SECTION ============
        if news_data and news_data.get('news_count', 0) > 0:
            st.markdown("## News Sentiment Analysis")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                sentiment = news_data['overall_sentiment']
                sentiment_class = f"sentiment-{sentiment.lower()}"
                st.markdown(f"**Overall Sentiment**")
                st.markdown(f"<p class='{sentiment_class}'>{sentiment}</p>", unsafe_allow_html=True)
            
            with col2:
                st.metric("Total News", news_data['news_count'])
            
            with col3:
                st.metric("Positive", news_data['positive_count'], 
                         delta=f"{(news_data['positive_count']/news_data['news_count']*100):.0f}%")
            
            with col4:
                st.metric("Negative", news_data['negative_count'],
                         delta=f"{(news_data['negative_count']/news_data['news_count']*100):.0f}%",
                         delta_color="inverse")
            
            # Sentiment score gauge
            score = news_data['sentiment_score']
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Sentiment Score"},
                delta={'reference': 0},
                gauge={
                    'axis': {'range': [-1, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [-1, -0.3], 'color': "lightcoral"},
                        {'range': [-0.3, 0.3], 'color': "lightyellow"},
                        {'range': [0.3, 1], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0
                    }
                }
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Recent news articles
            st.markdown("### 📑 Recent News")
            
            for i, article in enumerate(news_data['articles'][:5], 1):
                sentiment_class = f"sentiment-{article['sentiment'].lower()}"
                
                with st.expander(f"{i}. {article['title'][:100]}..."):
                    st.markdown(f"**Sentiment:** <span class='{sentiment_class}'>{article['sentiment']}</span> "
                              f"(Score: {article['score']:.2f})", unsafe_allow_html=True)
                    st.markdown(f"**Published:** {article['publishedAt'][:10]}")
                    st.markdown(f"**Description:** {article['description']}")
                    st.markdown(f"[Read more]({article['url']})")
            
            st.markdown("---")
        elif news_data and news_data.get('news_count', 0) == 0:
            st.info("No recent news found for this stock")
            st.markdown("---")
        
        # Key Metrics
        st.markdown("### Key Metrics")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Current Price", f"₹{analysis['current_price']:.2f}")
        with col2:
            st.metric("Market Cap", format_number(analysis['market_cap']))
        with col3:
            st.metric("P/E Ratio", f"{analysis['pe_ratio']:.2f}")
        with col4:
            st.metric("P/B Ratio", f"{analysis['pb_ratio']:.2f}")
        with col5:
            st.metric("Div Yield", f"{analysis['dividend_yield']:.2f}%")
        
        st.markdown("---")
        
        # Performance Metrics
        st.markdown("### Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("1 Month", f"{analysis['returns_1m']:.2f}%")
        with col2:
            st.metric("3 Months", f"{analysis['returns_3m']:.2f}%")
        with col3:
            st.metric("6 Months", f"{analysis['returns_6m']:.2f}%")
        with col4:
            st.metric("1 Year", f"{analysis['returns_1y']:.2f}%")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("52W High", f"₹{analysis['week_52_high']:.2f}")
        with col2:
            st.metric("52W Low", f"₹{analysis['week_52_low']:.2f}")
        with col3:
            st.metric("Volatility", f"{analysis['volatility']:.2f}%")
        
        st.markdown("---")
        
        # Technical Indicators
        st.markdown("### Technical Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            rsi = analysis['rsi']
            rsi_color = "🟢" if 30 < rsi < 70 else "🔴"
            st.metric("RSI", f"{rsi:.2f} {rsi_color}")
        
        with col2:
            macd = analysis['macd']
            macd_color = "🟢" if macd > analysis['macd_signal'] else "🔴"
            st.metric("MACD", f"{macd:.2f} {macd_color}")
        
        with col3:
            adx = analysis['adx']
            adx_color = "🟢" if adx > 25 else "🟡"
            st.metric("ADX", f"{adx:.2f} {adx_color}")
        
        with col4:
            vol_ratio = analysis['volume_ratio']
            vol_color = "🟢" if vol_ratio > 1 else "🔴"
            st.metric("Volume Ratio", f"{vol_ratio:.2f} {vol_color}")
        
        st.markdown("---")
        
        # Charts
        st.markdown("### Charts")
        
        df = analysis['df']
        
        # Price chart with indicators
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=('Price & Bollinger Bands', 'RSI', 'MACD')
        )
        
        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Bollinger Bands
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], 
                                line=dict(dash='dash', color='gray', width=1),
                                name='BB Upper'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], 
                                line=dict(dash='dash', color='gray', width=1),
                                name='BB Lower'), row=1, col=1)
        
        # RSI
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], 
                                line=dict(color='purple'),
                                name='RSI'), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # MACD
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], 
                                line=dict(color='blue'),
                                name='MACD'), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], 
                                line=dict(color='red'),
                                name='Signal'), row=3, col=1)
        
        fig.update_layout(height=800, showlegend=False, xaxis_rangeslider_visible=False)
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Volume chart
        st.markdown("### Volume Analysis")
        
        fig_vol = go.Figure()
        
        colors = ['red' if df['Close'].iloc[i] < df['Open'].iloc[i] else 'green' 
                 for i in range(len(df))]
        
        fig_vol.add_trace(go.Bar(
            x=df.index,
            y=df['Volume'],
            marker_color=colors,
            name='Volume'
        ))
        
        fig_vol.add_trace(go.Scatter(
            x=df.index,
            y=df['Vol_MA'],
            line=dict(color='orange', width=2),
            name='Avg Volume'
        ))
        
        fig_vol.update_layout(height=300)
        st.plotly_chart(fig_vol, use_container_width=True)

        st.markdown("---")

        # ============ TECHNICAL PATTERNS SECTION ============
        st.markdown("### 🔍 Chart Pattern Detection")

        try:
            tech_result = st.session_state.technical_analyzer.full_analysis(df)

            if tech_result:
                # Overall sentiment
                sentiment = tech_result.get("sentiment", "Neutral")
                sentiment_score = tech_result.get("sentiment_score", 0)
                s_color = "#2ecc71" if sentiment == "Bullish" else ("#e74c3c" if sentiment == "Bearish" else "#f39c12")
                st.markdown(f"**Technical Sentiment:** <span style='color:{s_color}; font-weight:bold;'>"
                           f"{sentiment} ({sentiment_score:+.0f})</span> | "
                           f"Patterns Found: {tech_result.get('total_patterns_found', 0)}",
                           unsafe_allow_html=True)

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### Chart Patterns")
                    if tech_result.get("chart_patterns"):
                        for p in tech_result["chart_patterns"]:
                            sig_color = "#2ecc71" if p["signal"] == "Bullish" else ("#e74c3c" if p["signal"] == "Bearish" else "#f39c12")
                            st.markdown(f"- **{p['pattern']}** → <span style='color:{sig_color}'>{p['signal']}</span> "
                                       f"({p['confidence']}%) — {p['detail']}", unsafe_allow_html=True)
                    else:
                        st.write("No chart patterns detected")

                with col2:
                    st.markdown("#### Candlestick Patterns")
                    if tech_result.get("candlestick_patterns"):
                        for p in tech_result["candlestick_patterns"]:
                            sig_color = "#2ecc71" if "Bullish" in p["signal"] else ("#e74c3c" if "Bearish" in p["signal"] else "#f39c12")
                            st.markdown(f"- **{p['pattern']}** → <span style='color:{sig_color}'>{p['signal']}</span> "
                                       f"({p['confidence']}%)", unsafe_allow_html=True)
                    else:
                        st.write("No candlestick patterns detected")

                # Divergences
                if tech_result.get("divergences"):
                    st.markdown("#### Divergences")
                    for d in tech_result["divergences"]:
                        d_color = "#2ecc71" if d["signal"] == "Bullish" else "#e74c3c"
                        st.markdown(f"- **{d['type']}** → <span style='color:{d_color}'>{d['signal']}</span> "
                                   f"({d['confidence']}%) — {d['detail']}", unsafe_allow_html=True)

                st.markdown("---")

                # ============ SUPPORT & RESISTANCE ============
                sr = tech_result.get("support_resistance", {})
                if sr:
                    st.markdown("### 📏 Support & Resistance Levels")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown("**Support Levels**")
                        for i, level in enumerate(sr.get("support", [])[:5], 1):
                            dist_pct = (level / sr['current_price'] - 1) * 100
                            st.write(f"S{i}: ₹{level:.2f} ({dist_pct:+.1f}%)")

                    with col2:
                        st.markdown("**Resistance Levels**")
                        for i, level in enumerate(sr.get("resistance", [])[:5], 1):
                            dist_pct = (level / sr['current_price'] - 1) * 100
                            st.write(f"R{i}: ₹{level:.2f} ({dist_pct:+.1f}%)")

                    with col3:
                        st.markdown("**Fibonacci Retracement**")
                        fib = sr.get("fibonacci", {})
                        for label, val in fib.items():
                            pct = label.replace("fib_", "").replace("00", "0")
                            st.write(f"{pct}%: ₹{val:.2f}")

                    # S/R on price chart
                    st.markdown("#### Support & Resistance Chart")
                    fig_sr = go.Figure()
                    fig_sr.add_trace(go.Scatter(
                        x=df.index[-120:], y=df['Close'][-120:],
                        mode='lines', name='Price', line=dict(color='#667eea', width=2)
                    ))
                    for i, level in enumerate(sr.get("support", [])[:3]):
                        fig_sr.add_hline(y=level, line_dash="dash", line_color="#2ecc71",
                                         annotation_text=f"S{i+1}: ₹{level:.2f}")
                    for i, level in enumerate(sr.get("resistance", [])[:3]):
                        fig_sr.add_hline(y=level, line_dash="dash", line_color="#e74c3c",
                                         annotation_text=f"R{i+1}: ₹{level:.2f}")
                    # Pivot point
                    pivot = sr.get("pivot_points", {}).get("pivot")
                    if pivot:
                        fig_sr.add_hline(y=pivot, line_dash="dot", line_color="#f39c12",
                                         annotation_text=f"Pivot: ₹{pivot:.2f}")
                    fig_sr.update_layout(title="Price with Support & Resistance", height=400)
                    st.plotly_chart(fig_sr, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not run pattern detection: {e}")

        st.markdown("---")

        # ============ FUNDAMENTAL ANALYSIS SECTION ============
        st.markdown("### 📋 Fundamental Analysis")

        fundamental_overall_score = None

        try:
            fund_data = st.session_state.fundamental_analyzer.get_fundamentals(
                analysis['symbol'].replace('.NS', ''), "NSE"
            )

            if fund_data:
                score_result = st.session_state.fundamental_analyzer.score_fundamentals(fund_data)

                if score_result:
                    fundamental_overall_score = float(score_result.get("overall_score", 50))
                    rating = score_result["rating"]
                    rating_color = "#2ecc71" if "Buy" in rating else ("#e74c3c" if "Sell" in rating else "#f39c12")
                    st.markdown(f"**Fundamental Rating:** <span style='color:{rating_color}; font-weight:bold; font-size:1.2em;'>"
                               f"{rating} ({score_result['overall_score']:.0f}/100)</span>", unsafe_allow_html=True)

                    # Category scores
                    cat_scores = score_result.get("category_scores", {})
                    score_cols = st.columns(5)
                    for i, (cat, score) in enumerate(cat_scores.items()):
                        with score_cols[i]:
                            st.metric(cat.replace("_", " ").title(), f"{score:.0f}")

                st.markdown("---")

                # Valuation
                val = fund_data.get("valuation", {})
                prof = fund_data.get("profitability", {})
                health = fund_data.get("financial_health", {})
                growth_data = fund_data.get("growth", {})
                div_data = fund_data.get("dividends", {})

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("**Valuation**")
                    for label, key in [("P/E Ratio", "pe_ratio"), ("Forward P/E", "forward_pe"),
                                       ("P/B Ratio", "pb_ratio"), ("P/S Ratio", "ps_ratio"),
                                       ("PEG Ratio", "peg_ratio"), ("EV/EBITDA", "ev_ebitda")]:
                        v = val.get(key)
                        st.write(f"{label}: **{v:.2f}**" if isinstance(v, (int, float)) else f"{label}: N/A")

                with col2:
                    st.markdown("**Profitability & Growth**")
                    for label, key, src in [
                        ("ROE", "roe", prof), ("ROA", "roa", prof),
                        ("Profit Margin", "profit_margin", prof),
                        ("Operating Margin", "operating_margin", prof),
                        ("Revenue Growth", "revenue_growth", growth_data),
                        ("Earnings Growth", "earnings_growth", growth_data),
                    ]:
                        v = src.get(key)
                        if isinstance(v, (int, float)):
                            pct = v * 100 if abs(v) < 5 else v
                            st.write(f"{label}: **{pct:.2f}%**")
                        else:
                            st.write(f"{label}: N/A")

                with col3:
                    st.markdown("**Financial Health & Dividends**")
                    for label, key in [("Debt/Equity", "debt_to_equity"), ("Current Ratio", "current_ratio"),
                                       ("Quick Ratio", "quick_ratio")]:
                        v = health.get(key)
                        st.write(f"{label}: **{v:.2f}**" if isinstance(v, (int, float)) else f"{label}: N/A")

                    dy = div_data.get("dividend_yield")
                    if isinstance(dy, (int, float)):
                        st.write(f"Dividend Yield: **{dy*100:.2f}%**")
                    else:
                        st.write("Dividend Yield: N/A")

                    pr = div_data.get("payout_ratio")
                    if isinstance(pr, (int, float)):
                        st.write(f"Payout Ratio: **{pr*100:.1f}%**")
                    else:
                        st.write("Payout Ratio: N/A")

                    fcf = health.get("free_cash_flow")
                    if isinstance(fcf, (int, float)):
                        st.write(f"Free Cash Flow: **₹{fcf/10000000:.2f} Cr**")
                    else:
                        st.write("Free Cash Flow: N/A")

                # Store for export
                st.session_state['_last_fundamentals'] = fund_data
            else:
                st.info("Fundamental data not available for this stock.")
        except Exception as e:
            st.warning(f"Could not fetch fundamental data: {e}")

        st.markdown("---")

        # ============ CONFLUENCE DASHBOARD ============
        st.markdown("### Signal Confluence Dashboard")
        confluence = compute_analysis_confluence(
            analysis,
            prediction=prediction,
            news_data=news_data,
            fundamental_score=fundamental_overall_score,
        )

        overall_score = confluence["overall_score"]
        if overall_score >= 75:
            badge_color = "#2ecc71"
        elif overall_score >= 60:
            badge_color = "#27ae60"
        elif overall_score >= 45:
            badge_color = "#f39c12"
        elif overall_score >= 30:
            badge_color = "#e67e22"
        else:
            badge_color = "#e74c3c"

        st.markdown(
            f"<div style='padding:0.8rem 1rem; border-radius:10px; border:1px solid {badge_color}; "
            f"background: rgba(255,255,255,0.02);'>"
            f"<strong>Overall Confluence:</strong> {overall_score:.1f}/100 &nbsp; | &nbsp; "
            f"<strong>Bias:</strong> <span style='color:{badge_color}; font-weight:700;'>{confluence['label']}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        comp = confluence["components"]
        cc1, cc2, cc3, cc4, cc5 = st.columns(5)
        with cc1:
            st.metric("Technical", f"{comp['technical']:.1f}")
        with cc2:
            st.metric("Prediction", f"{comp['prediction']:.1f}")
        with cc3:
            st.metric("Sentiment", f"{comp['sentiment']:.1f}")
        with cc4:
            st.metric("Fundamental", f"{comp['fundamental']:.1f}")
        with cc5:
            st.metric("Recommendation", f"{comp['recommendation']:.1f}")

        confluence_df = pd.DataFrame(
            [
                {"Component": "Technical", "Score": comp["technical"]},
                {"Component": "Prediction", "Score": comp["prediction"]},
                {"Component": "Sentiment", "Score": comp["sentiment"]},
                {"Component": "Fundamental", "Score": comp["fundamental"]},
                {"Component": "Recommendation", "Score": comp["recommendation"]},
            ]
        )
        fig_confluence = px.bar(
            confluence_df,
            x="Component",
            y="Score",
            color="Component",
            text=confluence_df["Score"].map(lambda x: f"{x:.1f}"),
            range_y=[0, 100],
            title="Confluence Components (0-100)",
        )
        fig_confluence.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig_confluence, use_container_width=True)

        st.markdown("---")

        # ============ EXPORT SECTION ============
        st.markdown("### 📥 Export Report")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            # CSV export of key metrics
            try:
                export_data = {
                    'Symbol': analysis['symbol'],
                    'Company': analysis['company_name'],
                    'Price': analysis['current_price'],
                    'P/E': analysis.get('pe_ratio', 'N/A'),
                    'P/B': analysis.get('pb_ratio', 'N/A'),
                    'RSI': analysis.get('rsi', 'N/A'),
                    'MACD': analysis.get('macd', 'N/A'),
                    'ADX': analysis.get('adx', 'N/A'),
                    'Recommendation': analysis['recommendation']['action'],
                    '1M Return': analysis.get('returns_1m', 'N/A'),
                    '3M Return': analysis.get('returns_3m', 'N/A'),
                    '1Y Return': analysis.get('returns_1y', 'N/A'),
                }
                csv_str = pd.DataFrame([export_data]).to_csv(index=False)
                st.download_button("📊 Download CSV", csv_str,
                                   f"{analysis['symbol']}_analysis.csv", "text/csv",
                                   use_container_width=True)
            except Exception:
                pass

        with col2:
            # HTML report export
            try:
                fund_for_export = st.session_state.get('_last_fundamentals', None)
                tech_for_export = tech_result if 'tech_result' in dir() else None
                html_report = st.session_state.export_manager.generate_stock_report_html(
                    analysis['symbol'], analysis, fund_for_export, tech_for_export
                )
                st.download_button("📄 Download HTML Report", html_report,
                                   f"{analysis['symbol']}_report.html", "text/html",
                                   use_container_width=True)
            except Exception:
                pass

        with col3:
            # Risk metrics export
            try:
                risk_m = st.session_state.risk_analytics.compute_all_metrics(df)
                if risk_m:
                    risk_csv = st.session_state.export_manager.risk_metrics_to_csv(risk_m)
                    st.download_button("🛡️ Download Risk Metrics", risk_csv,
                                       f"{analysis['symbol']}_risk.csv", "text/csv",
                                       use_container_width=True)
            except Exception:
                pass

        with col4:
            # Confluence snapshot export
            try:
                confluence_export = {
                    "symbol": analysis.get("symbol"),
                    "company": analysis.get("company_name"),
                    "timestamp": datetime.now().isoformat(),
                    "overall_score": float(confluence.get("overall_score", 0)),
                    "label": confluence.get("label", "N/A"),
                    "components": {
                        k: float(v) for k, v in confluence.get("components", {}).items()
                    },
                }
                confluence_json = json.dumps(confluence_export, indent=2)
                st.download_button(
                    "📦 Download Confluence",
                    confluence_json,
                    f"{analysis['symbol']}_confluence.json",
                    "application/json",
                    use_container_width=True,
                    key="confluence_export",
                )
            except Exception:
                pass

