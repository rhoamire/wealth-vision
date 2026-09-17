"""Shared display helper used by several pages."""
from plotly.subplots import make_subplots

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from core.utils import format_number, _apply_hover_defaults_to_figure, _merge_plotly_config


def display_stock_analysis(analysis, news_data, prediction):
    """Helper function to display stock analysis"""
    
    # Header info
    st.markdown(f"### {analysis['company_name']} ({analysis['symbol']})")
    st.markdown(f"**Sector:** {analysis['sector']} | **Industry:** {analysis['industry']}")
    
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


