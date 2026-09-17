"""Page: Portfolio"""
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

def render_portfolio_page():
    st.markdown("## Portfolio Management")
    
    tab1, tab2, tab3, tab4 = st.tabs(["My Portfolio", "Add Stock", "Analysis", "Allocation & Rebalance"])
    
    with tab1:
        st.markdown("### My Portfolio")
        
        if st.session_state.portfolio_manager.portfolio:
            analysis = st.session_state.portfolio_manager.analyze_portfolio()
            
            if analysis:
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Invested", format_number(analysis['total_invested']))
                with col2:
                    st.metric("Current Value", format_number(analysis['total_current']))
                with col3:
                    pnl = analysis['total_pnl']
                    st.metric("P&L", format_number(pnl), delta=f"{analysis['total_pnl_pct']:.2f}%")
                with col4:
                    returns = analysis['total_pnl_pct']
                    st.metric("Returns", f"{returns:.2f}%")
                
                # Portfolio table
                st.markdown("---")
                df_portfolio = pd.DataFrame(analysis['stocks'])
                st.dataframe(df_portfolio, use_container_width=True)
                
                # Remove stock
                st.markdown("---")
                stock_to_remove = st.selectbox(
                    "Remove Stock",
                    [s['symbol'] for s in st.session_state.portfolio_manager.portfolio]
                )
                if st.button("Remove"):
                    st.session_state.portfolio_manager.remove_stock(stock_to_remove)
                    if st.session_state.auth_username and st.session_state.auth_username != "__guest__":
                        try:
                            save_user_portfolio(
                                st.session_state.auth_username,
                                st.session_state.portfolio_manager.portfolio,
                            )
                        except Exception:
                            pass
                    st.rerun()
        else:
            st.info("📊 Your portfolio is empty. Add stocks to get started!")
    
    with tab2:
        st.markdown("### Add Stock to Portfolio")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            symbol = st.selectbox("Stock Symbol", st.session_state.nse_stocks)
        
        with col2:
            quantity = st.number_input("Quantity", min_value=1, value=100)
        
        with col3:
            buy_price = st.number_input("Buy Price (₹)", min_value=0.01, value=100.0)
        
        if st.button("Add to Portfolio", type="primary"):
            st.session_state.portfolio_manager.add_stock(symbol, quantity, buy_price)
            if st.session_state.auth_username and st.session_state.auth_username != "__guest__":
                try:
                    save_user_portfolio(
                        st.session_state.auth_username,
                        st.session_state.portfolio_manager.portfolio,
                    )
                except Exception:
                    pass
            st.success(f"Added {quantity} shares of {symbol} to portfolio!")
            st.rerun()
    
    with tab3:
        st.markdown("### Portfolio Analysis")
        
        if st.session_state.portfolio_manager.portfolio:
            st.markdown("#### Individual Stock Performance")
            
            for stock in st.session_state.portfolio_manager.portfolio:
                symbol = stock['symbol']
                
                with st.expander(f"📊 {symbol}"):
                    analysis = st.session_state.stock_analyzer.analyze_stock(symbol)
                    
                    if analysis:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Current Price", f"₹{analysis['current_price']:.2f}")
                        with col2:
                            st.metric("1M Return", f"{analysis['returns_1m']:.2f}%")
                        with col3:
                            rec = analysis['recommendation']['action']
                            st.markdown(f"<div class='recommendation-{rec.lower().replace(' ', '-')}'>{rec}</div>", 
                                      unsafe_allow_html=True)
                        
                        # Mini chart
                        if analysis['df'] is not None:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=analysis['df'].index[-60:],
                                y=analysis['df']['Close'][-60:],
                                mode='lines',
                                name='Price',
                                line=dict(color='#667eea', width=2)
                            ))
                            fig.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0))
                            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Add stocks to your portfolio first!")

    with tab4:
        st.markdown("### Allocation & Rebalance Assistant")
        st.caption("Detect drift versus target allocation, position-size risk breaches, and sector concentration.")

        if st.session_state.portfolio_manager.portfolio:
            pf_summary = st.session_state.portfolio_manager.analyze_portfolio()
            if pf_summary and pf_summary.get("stocks"):
                rebalance_df = pd.DataFrame(pf_summary["stocks"]).copy()
                rebalance_df["Current Value"] = pd.to_numeric(rebalance_df["Current Value"], errors="coerce").fillna(0.0)
                total_current_value = float(rebalance_df["Current Value"].sum())

                if total_current_value <= 0:
                    st.warning("Portfolio current value is unavailable for rebalance calculations.")
                else:
                    rb_col1, rb_col2, rb_col3 = st.columns(3)
                    with rb_col1:
                        use_equal_weight = st.checkbox("Use equal-weight targets", value=True, key="rb_equal_weight")
                    with rb_col2:
                        max_position_pct = st.slider("Max position weight (%)", 5, 60, 25, key="rb_max_pos")
                    with rb_col3:
                        max_sector_pct = st.slider("Max sector exposure (%)", 10, 90, 40, key="rb_max_sector")

                    symbols = rebalance_df["Symbol"].tolist()
                    target_inputs = {}
                    if use_equal_weight:
                        equal_target = 100 / len(symbols)
                        target_inputs = {s: equal_target for s in symbols}
                        st.info(f"Equal-weight target applied: {equal_target:.2f}% per stock")
                    else:
                        st.markdown("#### Target Weights")
                        target_cols = st.columns(min(4, len(symbols)))
                        for i, sym in enumerate(symbols):
                            with target_cols[i % len(target_cols)]:
                                target_inputs[sym] = st.number_input(
                                    f"{sym} (%)",
                                    min_value=0.0,
                                    max_value=100.0,
                                    value=round(100 / len(symbols), 2),
                                    step=0.5,
                                    key=f"rb_target_{sym}",
                                )
                        st.caption(f"Target sum: {sum(target_inputs.values()):.2f}%")

                    target_total = sum(target_inputs.values())
                    if target_total <= 0:
                        st.warning("Target allocation sum must be greater than 0.")
                    else:
                        plan_rows = []
                        for _, row in rebalance_df.iterrows():
                            sym = row["Symbol"]
                            current_value = float(row["Current Value"])
                            current_weight = (current_value / total_current_value) * 100 if total_current_value > 0 else 0
                            target_weight = (target_inputs.get(sym, 0) / target_total) * 100
                            target_value = total_current_value * (target_weight / 100)
                            rebalance_amount = target_value - current_value

                            if rebalance_amount > 0.01:
                                action = "BUY"
                            elif rebalance_amount < -0.01:
                                action = "SELL"
                            else:
                                action = "HOLD"

                            plan_rows.append({
                                "Symbol": sym,
                                "Current Weight %": round(current_weight, 2),
                                "Target Weight %": round(target_weight, 2),
                                "Drift %": round(current_weight - target_weight, 2),
                                "Current Value": round(current_value, 2),
                                "Target Value": round(target_value, 2),
                                "Action": action,
                                "Rebalance Amount": round(abs(rebalance_amount), 2),
                                "Risk Breach": "Yes" if current_weight > max_position_pct else "No",
                            })

                        plan_df = pd.DataFrame(plan_rows)
                        st.markdown("#### Rebalance Plan")
                        st.dataframe(plan_df, use_container_width=True, hide_index=True)

                        risk_breaches = plan_df[plan_df["Risk Breach"] == "Yes"]
                        if not risk_breaches.empty:
                            st.warning(f"{len(risk_breaches)} position(s) exceed max position weight of {max_position_pct}%.")
                        else:
                            st.success("No single-position risk breaches detected.")

                        # Sector concentration analysis
                        if "portfolio_sector_cache" not in st.session_state:
                            st.session_state.portfolio_sector_cache = {}

                        if st.button("🔄 Refresh Sector Mapping", key="rb_refresh_sector"):
                            st.session_state.portfolio_sector_cache = {}

                        sector_totals = {}
                        for _, row in rebalance_df.iterrows():
                            sym = row["Symbol"]
                            current_value = float(row["Current Value"])

                            if sym in st.session_state.portfolio_sector_cache:
                                sector = st.session_state.portfolio_sector_cache[sym]
                            else:
                                try:
                                    sym_analysis = st.session_state.stock_analyzer.analyze_stock(sym)
                                    sector = sym_analysis.get("sector", "Unknown") if sym_analysis else "Unknown"
                                except Exception:
                                    sector = "Unknown"
                                st.session_state.portfolio_sector_cache[sym] = sector

                            sector_totals[sector] = sector_totals.get(sector, 0.0) + current_value

                        sector_rows = []
                        for sector, value in sector_totals.items():
                            weight = (value / total_current_value) * 100 if total_current_value > 0 else 0
                            sector_rows.append({
                                "Sector": sector,
                                "Value": round(value, 2),
                                "Weight %": round(weight, 2),
                                "Breach": "Yes" if weight > max_sector_pct else "No",
                            })

                        sector_df = pd.DataFrame(sector_rows).sort_values("Weight %", ascending=False)
                        st.markdown("#### Sector Concentration")
                        st.dataframe(sector_df, use_container_width=True, hide_index=True)

                        sector_breaches = sector_df[sector_df["Breach"] == "Yes"]
                        if not sector_breaches.empty:
                            st.warning(f"{len(sector_breaches)} sector(s) exceed max sector exposure of {max_sector_pct}%.")
                        else:
                            st.success("Sector exposure is within configured limits.")

                        rebalance_export = plan_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Rebalance Plan (CSV)",
                            rebalance_export,
                            f"rebalance_plan_{datetime.now().strftime('%Y%m%d')}.csv",
                            "text/csv",
                            use_container_width=True,
                            key="rb_export_plan",
                        )
            else:
                st.info("Unable to compute portfolio summary for rebalance.")
        else:
            st.info("Add stocks to your portfolio to use the rebalance assistant.")

