"""Page: Price Alerts"""
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

def render_price_alerts_page():
    st.markdown("## 🔔 Price Alerts")
    st.info("Set threshold alerts with auto-checking, repeat cooldowns, expiry, and webhook/email delivery support.")

    tab1, tab2, tab3 = st.tabs(["Active Alerts", "Create Alert", "Alert History"])

    with tab1:
        st.markdown("### Active Alerts")

        top_col1, top_col2 = st.columns([1, 2])
        with top_col1:
            if st.button("🔄 Check Alerts Now", key="alert_check_now", use_container_width=True):
                manual_result = process_price_alerts(force=True)
                if manual_result.get("triggered"):
                    st.success(f"Triggered {len(manual_result['triggered'])} alert(s).")
                if manual_result.get("expired"):
                    st.warning(f"Expired {len(manual_result['expired'])} alert(s).")
                if not manual_result.get("triggered") and not manual_result.get("expired"):
                    st.info("No alert state changes found.")
        with top_col2:
            st.caption(
                f"Auto-check interval: {int(st.session_state.alert_check_interval_sec)} sec | "
                f"Repeat cooldown: {int(st.session_state.alert_repeat_cooldown_min)} min"
            )

        recent_notifications = st.session_state.get("recent_alert_notifications", [])
        if recent_notifications:
            with st.expander("Recent Notifications"):
                for n in reversed(recent_notifications[-10:]):
                    channels = ", ".join(n.get("channels", []))
                    st.write(f"{n.get('timestamp', '')[:19]} | {n.get('message', '')} | Channels: {channels}")

        if st.session_state.price_alerts:
            alert_data = []
            for a in st.session_state.price_alerts:
                try:
                    data = st.session_state.exchange_handler.get_stock_data(
                        a['symbol'], a.get('exchange', 'NSE'), period="5d"
                    )
                    if data is not None and len(data) > 0:
                        a['current_price'] = float(data['Close'].iloc[-1])
                except Exception:
                    pass

                target_price = float(a.get('target_price', 0) or 0)
                current_price = float(a.get('current_price', 0) or 0)
                distance = ((current_price / target_price) - 1) * 100 if target_price > 0 else 0
                expires_at = safe_parse_datetime(a.get('expires_at'))
                alert_data.append({
                    'Symbol': a['symbol'],
                    'Exchange': a.get('exchange', 'NSE'),
                    'Condition': f"Price {a['condition']}",
                    'Target': f"₹{target_price:.2f}",
                    'Current': f"₹{current_price:.2f}",
                    'Distance': f"{distance:+.2f}%",
                    'Repeat': "Yes" if a.get('repeat', False) else "No",
                    'Cooldown (min)': int(a.get('cooldown_minutes', st.session_state.alert_repeat_cooldown_min)),
                    'Expires': expires_at.strftime('%Y-%m-%d') if expires_at else "Never",
                    'Note': a.get('note', ''),
                })

            st.dataframe(pd.DataFrame(alert_data), use_container_width=True, hide_index=True)

            del_idx = st.selectbox(
                "Select alert to delete",
                range(len(st.session_state.price_alerts)),
                format_func=lambda i: (
                    f"{st.session_state.price_alerts[i]['symbol']} - "
                    f"{st.session_state.price_alerts[i]['condition']} "
                    f"₹{float(st.session_state.price_alerts[i]['target_price']):.2f}"
                ),
                key="alert_del_sel",
            )
            if st.button("🗑️ Delete Selected Alert", key="alert_del"):
                st.session_state.price_alerts.pop(del_idx)
                if st.session_state.auth_username and st.session_state.auth_username != "__guest__":
                    try:
                        save_user_alerts(st.session_state.auth_username, st.session_state.price_alerts)
                    except Exception:
                        pass
                st.success("Alert deleted.")
                st.rerun()
        else:
            st.info("No active alerts. Create one from the 'Create Alert' tab.")

    with tab2:
        st.markdown("### Create New Alert")

        col1, col2 = st.columns(2)
        with col1:
            alert_exchange = st.selectbox("Exchange", st.session_state.exchange_handler.get_supported_exchanges(), key="alert_ex")
            alert_lists = st.session_state.exchange_handler.get_stock_lists(alert_exchange)
            if alert_lists:
                alert_list_name = st.selectbox("Stock List", list(alert_lists.keys()), key="alert_list")
                alert_available = alert_lists[alert_list_name]
            else:
                alert_available = st.session_state.nse_stocks
            alert_symbol = st.selectbox("Stock", alert_available, key="alert_sym")

        with col2:
            alert_condition = st.selectbox("Condition", ["above", "below"], key="alert_cond",
                                           format_func=lambda x: f"Price goes {x}")
            alert_price = st.number_input("Target Price (₹)", min_value=0.01, value=100.0, step=1.0, key="alert_price")
            alert_note = st.text_input("Note (optional)", key="alert_note")
            alert_repeat = st.checkbox("Repeat after trigger", value=False, key="alert_repeat")
            alert_cooldown_min = st.number_input(
                "Repeat Cooldown (minutes)",
                min_value=1,
                max_value=1440,
                value=int(st.session_state.alert_repeat_cooldown_min),
                step=5,
                key="alert_cooldown_min",
            )
            alert_expiry_days = st.number_input(
                "Auto-expire after (days)",
                min_value=1,
                max_value=365,
                value=30,
                step=1,
                key="alert_expiry_days",
            )

        # Show current price for reference
        try:
            ref_data = st.session_state.exchange_handler.get_stock_data(alert_symbol, alert_exchange, period="5d")
            if ref_data is not None and len(ref_data) > 0:
                ref_price = float(ref_data['Close'].iloc[-1])
                st.markdown(f"**Current price of {alert_symbol}:** ₹{ref_price:.2f}")
        except Exception:
            pass

        if st.button("🔔 Create Alert", type="primary", use_container_width=True, key="alert_create"):
            created_at = datetime.now()
            new_alert = {
                'symbol': alert_symbol,
                'exchange': alert_exchange,
                'condition': alert_condition,
                'target_price': alert_price,
                'note': alert_note,
                'repeat': bool(alert_repeat),
                'cooldown_minutes': int(alert_cooldown_min),
                'created_at': created_at.isoformat(),
                'expires_at': (created_at + timedelta(days=int(alert_expiry_days))).isoformat(),
            }
            st.session_state.price_alerts.append(new_alert)
            # Persist for logged-in users
            if st.session_state.auth_username and st.session_state.auth_username != "__guest__":
                try:
                    save_user_alerts(st.session_state.auth_username, st.session_state.price_alerts)
                except Exception:
                    pass
            st.success(
                f"Alert created: {alert_symbol} {alert_condition} ₹{alert_price:.2f} | "
                f"Repeat: {'Yes' if alert_repeat else 'No'} | "
                f"Cooldown: {int(alert_cooldown_min)} min | "
                f"Expires in: {int(alert_expiry_days)} day(s)"
            )
            st.rerun()

    with tab3:
        st.markdown("### Alert History")
        if st.session_state.alert_history:
            hist_data = []
            for h in reversed(st.session_state.alert_history[-50:]):
                expires_at = safe_parse_datetime(h.get('expires_at'))
                hist_data.append({
                    'Symbol': h['symbol'],
                    'Exchange': h.get('exchange', 'NSE'),
                    'Condition': f"Price {h['condition']} ₹{h['target_price']:.2f}",
                    'Status': h.get('status', 'triggered').title(),
                    'Channels': h.get('notification_channels', 'N/A'),
                    'Triggered Price': f"₹{h.get('triggered_price', 0):.2f}",
                    'Triggered At': h.get('triggered_at', 'N/A'),
                    'Expires': expires_at.strftime('%Y-%m-%d') if expires_at else 'N/A',
                    'Note': h.get('note', ''),
                })
            st.dataframe(pd.DataFrame(hist_data), use_container_width=True, hide_index=True)

            if st.button("🗑️ Clear History", key="alert_hist_clear"):
                st.session_state.alert_history = []
                st.rerun()
        else:
            st.info("No alert history yet.")


