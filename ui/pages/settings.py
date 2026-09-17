"""Page: Settings"""
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

def render_settings_page():
    st.markdown("## ⚙️ Settings")

    tab1, tab2, tab3 = st.tabs(["Profile", "Preferences", "Data Management"])

    with tab1:
        st.markdown("### User Profile")

        if st.session_state.auth_username and st.session_state.auth_username != "__guest__":
            user_data = get_user_data(st.session_state.auth_username)
            st.markdown(f"**Username:** {st.session_state.auth_username}")
            st.markdown(f"**Full Name:** {user_data.get('full_name', 'N/A')}")
            st.markdown(f"**Email:** {user_data.get('email', 'N/A')}")
            st.markdown(f"**Member Since:** {user_data.get('created_at', 'N/A')[:10]}")
            st.markdown(f"**Last Login:** {user_data.get('last_login', 'N/A')[:19] if user_data.get('last_login') else 'N/A'}")

            st.markdown("---")
            st.markdown("### Change Password")
            with st.form("change_pw_form"):
                old_pw = st.text_input("Current Password", type="password", key="old_pw")
                new_pw = st.text_input("New Password", type="password", key="new_pw")
                new_pw2 = st.text_input("Confirm New Password", type="password", key="new_pw2")
                pw_submit = st.form_submit_button("Change Password", use_container_width=True)
                if pw_submit:
                    if new_pw != new_pw2:
                        st.error("New passwords do not match.")
                    else:
                        ok, msg = change_password(st.session_state.auth_username, old_pw, new_pw)
                        if ok:
                            st.success(msg)
                        else:
                            st.error(msg)
        else:
            st.info("You are in guest mode. Create an account to save your data across sessions.")

    with tab2:
        st.markdown("### Preferences")

        pref_exchange = st.selectbox("Default Exchange",
                                      st.session_state.exchange_handler.get_supported_exchanges(),
                                      index=st.session_state.exchange_handler.get_supported_exchanges().index(
                                          st.session_state.current_exchange
                                      ) if st.session_state.current_exchange in st.session_state.exchange_handler.get_supported_exchanges() else 0,
                                      key="pref_exchange")

        pref_risk_free = st.number_input("Risk-Free Rate (%)", 0.0, 20.0,
                                          st.session_state.risk_analytics.risk_free_rate * 100, 0.5,
                                          key="pref_rfr")

        st.markdown("#### Alert Delivery")
        pref_alert_webhook = st.text_input(
            "Webhook URL (optional)",
            value=st.session_state.alert_webhook_url,
            placeholder="https://example.com/alerts",
            key="pref_alert_webhook",
        )
        pref_email_alerts = st.checkbox(
            "Enable Email Alerts (requires SMTP environment variables)",
            value=st.session_state.enable_email_alerts,
            key="pref_email_alerts",
        )

        c_pref1, c_pref2 = st.columns(2)
        with c_pref1:
            pref_alert_interval = st.slider(
                "Auto-check Interval (seconds)",
                min_value=15,
                max_value=300,
                value=int(st.session_state.alert_check_interval_sec),
                step=15,
                key="pref_alert_interval",
            )
        with c_pref2:
            pref_repeat_cooldown = st.number_input(
                "Default Repeat Cooldown (minutes)",
                min_value=1,
                max_value=1440,
                value=int(st.session_state.alert_repeat_cooldown_min),
                step=5,
                key="pref_repeat_cooldown",
            )

        if st.button("💾 Save Preferences", type="primary", use_container_width=True, key="pref_save"):
            st.session_state.current_exchange = pref_exchange
            st.session_state.risk_analytics.risk_free_rate = pref_risk_free / 100
            st.session_state.portfolio_risk.risk_analytics.risk_free_rate = pref_risk_free / 100
            st.session_state.alert_webhook_url = pref_alert_webhook.strip()
            st.session_state.enable_email_alerts = bool(pref_email_alerts)
            st.session_state.alert_check_interval_sec = int(pref_alert_interval)
            st.session_state.alert_repeat_cooldown_min = int(pref_repeat_cooldown)
            if st.session_state.auth_username and st.session_state.auth_username != "__guest__":
                save_user_settings(st.session_state.auth_username, {
                    "default_exchange": pref_exchange,
                    "risk_free_rate": pref_risk_free,
                    "alert_webhook_url": st.session_state.alert_webhook_url,
                    "enable_email_alerts": st.session_state.enable_email_alerts,
                    "alert_check_interval_sec": st.session_state.alert_check_interval_sec,
                    "alert_repeat_cooldown_min": st.session_state.alert_repeat_cooldown_min,
                })
            st.success("Preferences saved!")

    with tab3:
        st.markdown("### Data Management")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Export All Data")
            try:
                all_data = {
                    "watchlist": st.session_state.get('watchlist', []),
                    "portfolio": st.session_state.portfolio_manager.portfolio,
                    "alerts": st.session_state.get('price_alerts', []),
                    "alert_history": st.session_state.get('alert_history', []),
                    "trade_journal": st.session_state.get('trade_journal', []),
                }
                export_json = json.dumps(all_data, indent=2, default=str)
                st.download_button("📥 Export Data (JSON)", export_json,
                                   f"artha_drishti_data_{datetime.now().strftime('%Y%m%d')}.json",
                                   "application/json", use_container_width=True, key="export_all_data")
            except Exception:
                st.warning("Could not prepare data export.")

        with col2:
            st.markdown("#### Import Data")
            uploaded = st.file_uploader("Upload JSON data file", type=["json"], key="import_data_file")
            if uploaded is not None:
                try:
                    imported = json.loads(uploaded.read().decode())
                    if st.button("📤 Import Data", type="primary", use_container_width=True, key="import_data_btn"):
                        if "watchlist" in imported:
                            st.session_state.watchlist = imported["watchlist"]
                        if "portfolio" in imported and isinstance(imported["portfolio"], list):
                            st.session_state.portfolio_manager.portfolio = imported["portfolio"]
                        if "alerts" in imported:
                            st.session_state.price_alerts = imported["alerts"]
                        if "alert_history" in imported:
                            st.session_state.alert_history = imported["alert_history"]
                        if "trade_journal" in imported and isinstance(imported["trade_journal"], list):
                            st.session_state.trade_journal = imported["trade_journal"]

                        if st.session_state.auth_username and st.session_state.auth_username != "__guest__":
                            try:
                                save_user_watchlist(st.session_state.auth_username, st.session_state.watchlist)
                            except Exception:
                                pass
                            try:
                                save_user_portfolio(st.session_state.auth_username,
                                                    st.session_state.portfolio_manager.portfolio)
                            except Exception:
                                pass
                            try:
                                save_user_alerts(st.session_state.auth_username, st.session_state.price_alerts)
                            except Exception:
                                pass
                            try:
                                save_user_trade_journal(
                                    st.session_state.auth_username,
                                    st.session_state.trade_journal,
                                )
                            except Exception:
                                pass

                        st.success("Data imported successfully!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Invalid JSON file: {e}")

        st.markdown("---")
        st.markdown("#### Clear Cache")
        if st.button("🗑️ Clear All Cached Data", key="clear_cache"):
            try:
                st.session_state.exchange_handler.clear_cache()
            except Exception:
                st.session_state.exchange_handler._cache.clear()
            st.cache_data.clear()
            st.success("Cache cleared! Realtime data will be fetched on the next action.")

        st.markdown("#### Reset Application")
        if st.button("⚠️ Reset All Session Data", key="reset_all"):
            for key in list(st.session_state.keys()):
                if key not in ['authenticated', 'auth_username', 'auth_user_data']:
                    del st.session_state[key]
            st.success("Session reset. Refreshing...")
            st.rerun()

