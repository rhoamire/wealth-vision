"""Page: Watchlist"""
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

def render_watchlist_page():
    st.markdown("## ⭐ Watchlist")
    st.info("Track your favorite stocks with live prices and quick analysis")

    tab1, tab2 = st.tabs(["My Watchlist", "Manage Watchlist"])

    with tab1:
        st.markdown("### Watchlist Stocks")

        if st.session_state.watchlist:
            if st.button("🔄 Refresh Prices", key="wl_refresh"):
                st.session_state.pop('watchlist_data', None)

            with st.spinner("Loading watchlist data..."):
                watchlist_data = []
                current_exchange = st.session_state.get('current_exchange', 'NSE')

                for item in st.session_state.watchlist:
                    sym = item if isinstance(item, str) else item.get('symbol', '')
                    exc = 'NSE' if isinstance(item, str) else item.get('exchange', current_exchange)

                    try:
                        df = st.session_state.exchange_handler.get_stock_data(sym, exc, period="1mo")
                        if df is not None and len(df) >= 2:
                            curr = df['Close'].iloc[-1]
                            prev = df['Close'].iloc[-2]
                            change = curr - prev
                            change_pct = (change / prev) * 100
                            high_52w = df['High'].max() if len(df) > 20 else curr
                            low_52w = df['Low'].min() if len(df) > 20 else curr
                            vol = df['Volume'].iloc[-1]

                            watchlist_data.append({
                                'Symbol': sym,
                                'Exchange': exc,
                                'Price': f"{curr:.2f}",
                                'Change': f"{change:.2f}",
                                'Change %': f"{change_pct:.2f}%",
                                'High (Period)': f"{high_52w:.2f}",
                                'Low (Period)': f"{low_52w:.2f}",
                                'Volume': f"{vol:,.0f}",
                            })
                    except:
                        watchlist_data.append({
                            'Symbol': sym, 'Exchange': exc,
                            'Price': 'N/A', 'Change': 'N/A', 'Change %': 'N/A',
                            'High (Period)': 'N/A', 'Low (Period)': 'N/A', 'Volume': 'N/A'
                        })

                if watchlist_data:
                    wl_df = pd.DataFrame(watchlist_data)
                    st.dataframe(wl_df, use_container_width=True, hide_index=True)

                    # Export watchlist
                    try:
                        wl_csv = st.session_state.export_manager.watchlist_to_csv(watchlist_data)
                        st.download_button("📥 Download Watchlist (CSV)", wl_csv,
                                           f"watchlist_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv",
                                           use_container_width=True, key="wl_export")
                    except Exception:
                        pass

                    # Mini charts
                    st.markdown("### Price Charts (1 Month)")
                    chart_cols = st.columns(min(len(st.session_state.watchlist), 3))
                    for i, item in enumerate(st.session_state.watchlist[:6]):
                        sym = item if isinstance(item, str) else item.get('symbol', '')
                        exc = 'NSE' if isinstance(item, str) else item.get('exchange', 'NSE')
                        with chart_cols[i % 3]:
                            try:
                                df = st.session_state.exchange_handler.get_stock_data(sym, exc, period="1mo")
                                if df is not None and len(df) > 1:
                                    fig = go.Figure()
                                    color = '#2ecc71' if df['Close'].iloc[-1] >= df['Close'].iloc[0] else '#e74c3c'
                                    fig.add_trace(go.Scatter(
                                        x=df.index, y=df['Close'], mode='lines',
                                        line=dict(color=color, width=2), name=sym
                                    ))
                                    fig.update_layout(title=sym, height=200,
                                                      margin=dict(l=0, r=0, t=30, b=0),
                                                      showlegend=False)
                                    st.plotly_chart(fig, use_container_width=True)
                            except:
                                st.write(f"{sym}: Chart unavailable")
                else:
                    st.info("Could not load watchlist data.")
        else:
            st.info("⭐ Your watchlist is empty. Add stocks from the Manage tab!")

    with tab2:
        st.markdown("### Add / Remove Stocks")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Add Stock")
            wl_exchange = st.selectbox("Exchange", st.session_state.exchange_handler.get_supported_exchanges(), key="wl_ex")
            wl_lists = st.session_state.exchange_handler.get_stock_lists(wl_exchange)
            if wl_lists:
                wl_list_name = st.selectbox("Stock List", list(wl_lists.keys()), key="wl_list")
                wl_stocks = wl_lists[wl_list_name]
            else:
                wl_stocks = st.session_state.nse_stocks
            wl_symbol = st.selectbox("Stock", wl_stocks, key="wl_sym")

            if st.button("➕ Add to Watchlist", type="primary", key="wl_add"):
                entry = {'symbol': wl_symbol, 'exchange': wl_exchange}
                # Avoid duplicates
                existing_syms = [
                    (item.get('symbol') if isinstance(item, dict) else item)
                    for item in st.session_state.watchlist
                ]
                if wl_symbol not in existing_syms:
                    st.session_state.watchlist.append(entry)
                    # Auto-save for logged-in users
                    if st.session_state.auth_username and st.session_state.auth_username != "__guest__":
                        try: save_user_watchlist(st.session_state.auth_username, st.session_state.watchlist)
                        except: pass
                    st.success(f"Added {wl_symbol} ({wl_exchange}) to watchlist!")
                    st.rerun()
                else:
                    st.warning(f"{wl_symbol} is already in your watchlist.")

            st.markdown("##### Bulk Add")
            wl_bulk_symbols = st.text_area(
                "Paste symbols (comma/space/newline separated)",
                height=90,
                key="wl_bulk_symbols",
            )
            if st.button("➕ Add Symbols in Bulk", key="wl_add_bulk"):
                parsed_symbols = parse_symbol_list(wl_bulk_symbols)
                if not parsed_symbols:
                    st.warning("Please provide at least one valid symbol.")
                else:
                    allowed_symbols = {normalize_ticker_symbol(s) for s in wl_stocks}
                    existing_symbols = {
                        normalize_ticker_symbol(item.get('symbol') if isinstance(item, dict) else item)
                        for item in st.session_state.watchlist
                    }

                    added = []
                    duplicates = []
                    invalid = []

                    for sym in parsed_symbols:
                        if sym in existing_symbols:
                            duplicates.append(sym)
                            continue
                        if sym not in allowed_symbols:
                            invalid.append(sym)
                            continue

                        st.session_state.watchlist.append({'symbol': sym, 'exchange': wl_exchange})
                        existing_symbols.add(sym)
                        added.append(sym)

                    if added and st.session_state.auth_username and st.session_state.auth_username != "__guest__":
                        try:
                            save_user_watchlist(st.session_state.auth_username, st.session_state.watchlist)
                        except Exception:
                            pass

                    if added:
                        st.success(f"Added {len(added)} symbol(s): {', '.join(added[:8])}{'...' if len(added) > 8 else ''}")
                    if duplicates:
                        st.info(f"Skipped {len(duplicates)} duplicate symbol(s).")
                    if invalid:
                        st.warning(f"Skipped {len(invalid)} symbol(s) not in selected universe: {', '.join(invalid[:8])}{'...' if len(invalid) > 8 else ''}")

        with col2:
            st.markdown("#### Remove Stock")
            if st.session_state.watchlist:
                remove_options = [
                    f"{(item.get('symbol') if isinstance(item, dict) else item)} ({(item.get('exchange', 'NSE') if isinstance(item, dict) else 'NSE')})"
                    for item in st.session_state.watchlist
                ]
                to_remove = st.selectbox("Select stock to remove", remove_options, key="wl_remove_sel")

                if st.button("🗑️ Remove from Watchlist", key="wl_remove"):
                    idx = remove_options.index(to_remove)
                    st.session_state.watchlist.pop(idx)
                    if st.session_state.auth_username and st.session_state.auth_username != "__guest__":
                        try: save_user_watchlist(st.session_state.auth_username, st.session_state.watchlist)
                        except: pass
                    st.success(f"Removed {to_remove} from watchlist!")
                    st.rerun()

                if st.button("🗑️ Clear Entire Watchlist", key="wl_clear"):
                    st.session_state.watchlist = []
                    if st.session_state.auth_username and st.session_state.auth_username != "__guest__":
                        try: save_user_watchlist(st.session_state.auth_username, st.session_state.watchlist)
                        except: pass
                    st.success("Watchlist cleared!")
                    st.rerun()
            else:
                st.info("No stocks to remove.")

