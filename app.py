"""
Artha Drishti - ML Based Advanced Stock Screener v4.0
Features: Multi-Exchange, Multi-Strategy, Backtesting, Risk Analytics, Portfolio, News Sentiment, Price Predictions
Run with: streamlit run app.py
"""

import streamlit as st
import warnings
from datetime import datetime

from config import NEWS_API_KEY

from core.multi_exchange import MultiExchangeHandler, MarketOverview, CrossExchangeComparator
from core.advanced_strategies import AdvancedStrategyEngine, SectorRotationDetector
from core.risk_analytics import RiskAnalytics, PortfolioRiskAnalyzer, StockComparator
from core.backtester import Backtester
from core.technical_patterns import TechnicalAnalyzer
from core.fundamental_analysis import FundamentalAnalyzer
from core.export_utils import ExportManager
from core.strategy_engine import StrategyEngine
from core.ml_predictor import EnhancedMLPredictor
from core.portfolio_manager import PortfolioManager
from core.stock_analyzer import StockAnalyzer
from core.news_sentiment import NewsSentimentAnalyzer
from core.price_predictor import PricePredictor
from core.data_loader import load_nse_stocks
from core.utils import normalize_ticker_symbol, process_price_alerts
from core.auth import save_user_watchlist, save_user_portfolio, save_user_alerts, save_user_trade_journal

from ui.auth_page import show_auth_page
from ui.pages.screener import render_screener_page
from ui.pages.multi_strategy import render_multi_strategy_page
from ui.pages.backtesting import render_backtesting_page
from ui.pages.risk_analytics import render_risk_analytics_page
from ui.pages.market_overview import render_market_overview_page
from ui.pages.portfolio import render_portfolio_page
from ui.pages.stock_analysis import render_stock_analysis_page
from ui.pages.watchlist import render_watchlist_page
from ui.pages.price_alerts import render_price_alerts_page
from ui.pages.trade_journal import render_trade_journal_page
from ui.pages.settings import render_settings_page

warnings.filterwarnings("ignore")

# ============================================================================
# PAGE CONFIG + CSS
# ============================================================================


st.set_page_config(
    page_title="Wealth Vision - ML Stock Screener",
    page_icon="assets/logo_512.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS with logo and background
st.markdown("""
    <style>
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes float {
        0%, 100% {
            transform: translateY(0px);
        }
        50% {
            transform: translateY(-10px);
        }
    }
    
    @keyframes glow {
        0%, 100% {
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.2);
        }
        50% {
            box-shadow: 0 15px 50px rgba(102, 126, 234, 0.4);
        }
    }
    
    * {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 25%, #16213e 50%, #0f3460 75%, #0a0e27 100%);
        min-height: 100vh;
        background-attachment: fixed;
        position: relative;
    }
    
    [data-testid="stAppViewContainer"]::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: radial-gradient(circle at 20% 50%, rgba(102, 126, 234, 0.05) 0%, transparent 50%),
                    radial-gradient(circle at 80% 80%, rgba(118, 75, 162, 0.05) 0%, transparent 50%);
        pointer-events: none;
        z-index: 0;
    }
    
    [data-testid="stMainBlockContainer"] {
        background: transparent;
        animation: fadeInUp 0.6s ease-out;
    }
    
    .stMarkdown {
        transition: all 0.3s ease;
    }
    
    p, span, label {
        color: #ffffff !important;
    }
    
    .stSelectbox label, .stNumberInput label, .stSlider label {
        color: #e0e0ff !important;
        font-weight: 500;
    }
    
    input, textarea, select {
        background: rgba(102, 126, 234, 0.08) !important;
        border: 1px solid rgba(102, 126, 234, 0.2) !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
    }
    
    input:hover, textarea:hover, select:hover {
        border-color: rgba(102, 126, 234, 0.4) !important;
        background: rgba(102, 126, 234, 0.12) !important;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.1) !important;
    }
    
    input:focus, textarea:focus, select:focus {
        border-color: #667eea !important;
        background: rgba(102, 126, 234, 0.15) !important;
        box-shadow: 0 0 30px rgba(102, 126, 234, 0.2) !important;
        outline: none !important;
    }
    
    .header-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 2rem 2rem;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.12) 0%, rgba(118, 75, 162, 0.12) 100%);
        border: 1px solid rgba(102, 126, 234, 0.25);
        border-radius: 20px;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2),
                    inset 0 1px 1px 0 rgba(255, 255, 255, 0.05);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeInUp 0.6s ease-out;
    }
    
    .header-container:hover {
        box-shadow: 0 15px 50px rgba(102, 126, 234, 0.3),
                    inset 0 1px 1px 0 rgba(255, 255, 255, 0.1);
        transform: translateY(-4px);
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.18) 0%, rgba(118, 75, 162, 0.18) 100%);
        border-color: rgba(102, 126, 234, 0.4);
    }
    
    .logo-image {
        max-width: 280px;
        height: auto;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        filter: drop-shadow(0 12px 30px rgba(102, 126, 234, 0.35)) 
                brightness(1.05);
        animation: float 3s ease-in-out infinite;
    }
    
    .logo-image:hover {
        transform: scale(1.12) rotate(2deg);
        filter: drop-shadow(0 20px 40px rgba(102, 126, 234, 0.5)) 
                brightness(1.15);
        animation: glow 2s ease-in-out infinite;
    }
    
    img {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .header-text {
        flex: 1;
        margin-left: 2rem;
        color: white;
        transition: all 0.3s ease;
    }
    .header-text h1 {
        margin: 0;
        font-size: 2.8rem;
        font-weight: 900;
        letter-spacing: -1px;
        transition: all 0.3s ease;
    }
    .header-text p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.95;
        transition: opacity 0.3s ease;
        font-weight: 500;
    }
    
    .nav-menu {
        display: flex;
        gap: 1rem;
        margin-bottom: 2rem;
        flex-wrap: wrap;
    }
    
    .nav-button {
        flex: 1;
        min-width: 180px;
        padding: 14px 24px;
        border: 1.5px solid rgba(102, 126, 234, 0.25);
        border-radius: 12px;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.08) 0%, rgba(118, 75, 162, 0.08) 100%);
        color: #e0e0ff;
        font-size: 1rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        cursor: pointer;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(5px);
        position: relative;
        overflow: hidden;
    }
    
    .nav-button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transition: left 0.5s ease;
    }
    
    .nav-button:hover::before {
        left: 100%;
    }
    
    .nav-button:hover {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
        transform: translateX(6px) translateY(-2px);
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.25),
                    inset 0 1px 1px rgba(255, 255, 255, 0.05);
        border-color: rgba(102, 126, 234, 0.4);
    }
    
    .nav-button.active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #ffffff;
        border-color: #667eea;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.5),
                    inset 0 1px 1px rgba(255, 255, 255, 0.1);
        transform: scale(1.02);
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .logo-container {
        text-align: center;
        padding: 2rem 0;
    }
    .tagline {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-top: -1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.12) 0%, rgba(118, 75, 162, 0.08) 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(102, 126, 234, 0.25);
        backdrop-filter: blur(10px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1),
                    inset 0 1px 1px 0 rgba(255, 255, 255, 0.05);
        animation: fadeInUp 0.6s ease-out;
    }
    
    .metric-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 15px 50px rgba(102, 126, 234, 0.3),
                    inset 0 1px 1px rgba(255, 255, 255, 0.1);
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.18) 0%, rgba(118, 75, 162, 0.14) 100%);
        border-color: rgba(102, 126, 234, 0.4);
    }
    
    .recommendation-strong-buy { 
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white; 
        padding: 0.75rem 1.25rem; 
        border-radius: 0.8rem; 
        font-weight: bold;
        display: inline-block;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    }
    
    .recommendation-strong-buy:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(16, 185, 129, 0.4);
    }
    
    .recommendation-buy { 
        background: linear-gradient(135deg, #34d399 0%, #10b981 100%);
        color: white; 
        padding: 0.75rem 1.25rem; 
        border-radius: 0.8rem; 
        font-weight: bold;
        display: inline-block;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(52, 211, 153, 0.3);
    }
    
    .recommendation-buy:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(52, 211, 153, 0.4);
    }
    
    .recommendation-hold { 
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
        color: white; 
        padding: 0.75rem 1.25rem; 
        border-radius: 0.8rem; 
        font-weight: bold;
        display: inline-block;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(251, 191, 36, 0.3);
    }
    
    .recommendation-hold:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(251, 191, 36, 0.4);
    }
    
    .recommendation-sell { 
        background: linear-gradient(135deg, #f97316 0%, #ea580c 100%);
        color: white; 
        padding: 0.75rem 1.25rem; 
        border-radius: 0.8rem; 
        font-weight: bold;
        display: inline-block;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(249, 115, 22, 0.3);
    }
    
    .recommendation-sell:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(249, 115, 22, 0.4);
    }
    
    .recommendation-strong-sell { 
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white; 
        padding: 0.75rem 1.25rem; 
        border-radius: 0.8rem; 
        font-weight: bold;
        display: inline-block;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);
    }
    
    .recommendation-strong-sell:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(239, 68, 68, 0.4);
    }
    
    .prediction-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.25) 0%, rgba(118, 75, 162, 0.25) 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 1px solid rgba(102, 126, 234, 0.3);
        backdrop-filter: blur(15px);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2),
                    inset 0 1px 1px 0 rgba(255, 255, 255, 0.1);
        animation: fadeInUp 0.6s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    .prediction-card::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.1) 0%, transparent 70%);
        transition: all 0.5s ease;
    }
    
    .prediction-card:hover {
        transform: translateY(-12px);
        box-shadow: 0 20px 50px rgba(102, 126, 234, 0.4),
                    inset 0 1px 1px rgba(255, 255, 255, 0.15);
        border-color: rgba(102, 126, 234, 0.5);
    }
    
    .prediction-card:hover::before {
        top: -20%;
        right: -20%;
    }
    .sentiment-positive {
        color: #10b981;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .sentiment-positive:hover {
        transform: scale(1.05);
    }
    
    .sentiment-negative {
        color: #ef4444;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .sentiment-negative:hover {
        transform: scale(1.05);
    }
    
    .sentiment-neutral {
        color: #f59e0b;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .sentiment-neutral:hover {
        transform: scale(1.05);
    }

    .js-plotly-plot .plotly .hoverlayer .hovertext {
        transition: transform 120ms ease-out, opacity 120ms ease-out;
    }

    .js-plotly-plot .plotly .hoverlayer .hovertext text {
        fill: #f4f8ff !important;
        font-weight: 700 !important;
        letter-spacing: 0.2px;
    }

    .js-plotly-plot .plotly .hoverlayer .hovertext rect {
        fill: rgba(8, 14, 30, 0.96) !important;
        stroke: rgba(104, 180, 255, 0.85) !important;
        stroke-width: 1.2px !important;
        filter: drop-shadow(0 8px 20px rgba(0, 0, 0, 0.45));
    }

    .js-plotly-plot .plotly .spikeline {
        stroke: rgba(104, 180, 255, 0.62) !important;
        stroke-width: 1.2px !important;
    }
    
    button {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    button:hover {
        transform: translateY(-2px) !important;
    }
    
    [role="tablist"] button {
        transition: all 0.3s ease;
    }
    
    [role="tab"]:hover {
        transform: translateY(-2px);
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================================
# AUTHENTICATION GATE
# ============================================================================

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'auth_username' not in st.session_state:
    st.session_state.auth_username = None
if 'auth_user_data' not in st.session_state:
    st.session_state.auth_user_data = {}

if not st.session_state.authenticated:
    show_auth_page()
    st.stop()

# Initialize session state
if 'nse_stocks' not in st.session_state:
    st.session_state.nse_stocks = None
if 'screened_stocks' not in st.session_state:
    st.session_state.screened_stocks = None
if 'best_buy_opportunities' not in st.session_state:
    st.session_state.best_buy_opportunities = None
if 'best_buy_context' not in st.session_state:
    st.session_state.best_buy_context = {}
if 'strategy_engine' not in st.session_state:
    st.session_state.strategy_engine = StrategyEngine()
if 'ml_predictor' not in st.session_state:
    st.session_state.ml_predictor = EnhancedMLPredictor()
if 'portfolio_manager' not in st.session_state:
    st.session_state.portfolio_manager = PortfolioManager()
if 'stock_analyzer' not in st.session_state:
    st.session_state.stock_analyzer = StockAnalyzer()
if 'news_analyzer' not in st.session_state:
    st.session_state.news_analyzer = NewsSentimentAnalyzer(NEWS_API_KEY)
if 'price_predictor' not in st.session_state:
    st.session_state.price_predictor = PricePredictor()
# --- NEW: Advanced module session state ---
if 'exchange_handler' not in st.session_state:
    st.session_state.exchange_handler = MultiExchangeHandler()
try:
    st.session_state.exchange_handler.set_realtime_mode(True)
except Exception:
    pass
if 'advanced_strategy_engine' not in st.session_state:
    st.session_state.advanced_strategy_engine = AdvancedStrategyEngine()
if 'market_overview' not in st.session_state:
    st.session_state.market_overview = MarketOverview(st.session_state.exchange_handler)
if 'cross_comparator' not in st.session_state:
    st.session_state.cross_comparator = CrossExchangeComparator(st.session_state.exchange_handler)
if 'risk_analytics' not in st.session_state:
    st.session_state.risk_analytics = RiskAnalytics()
if 'portfolio_risk' not in st.session_state:
    st.session_state.portfolio_risk = PortfolioRiskAnalyzer()
if 'stock_comparator' not in st.session_state:
    st.session_state.stock_comparator = StockComparator()
if 'backtester' not in st.session_state:
    st.session_state.backtester = Backtester()
if 'sector_rotation' not in st.session_state:
    st.session_state.sector_rotation = SectorRotationDetector(st.session_state.exchange_handler)
if 'current_exchange' not in st.session_state:
    st.session_state.current_exchange = "NSE"
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []
if 'technical_analyzer' not in st.session_state:
    st.session_state.technical_analyzer = TechnicalAnalyzer()
if 'fundamental_analyzer' not in st.session_state:
    st.session_state.fundamental_analyzer = FundamentalAnalyzer()
if 'export_manager' not in st.session_state:
    st.session_state.export_manager = ExportManager()
if 'price_alerts' not in st.session_state:
    st.session_state.price_alerts = []
if 'alert_history' not in st.session_state:
    st.session_state.alert_history = []
if 'trade_journal' not in st.session_state:
    st.session_state.trade_journal = []
if 'alert_webhook_url' not in st.session_state:
    st.session_state.alert_webhook_url = ""
if 'enable_email_alerts' not in st.session_state:
    st.session_state.enable_email_alerts = False
if 'alert_check_interval_sec' not in st.session_state:
    st.session_state.alert_check_interval_sec = 60
if 'alert_repeat_cooldown_min' not in st.session_state:
    st.session_state.alert_repeat_cooldown_min = 60
if 'last_alert_check_ts' not in st.session_state:
    st.session_state.last_alert_check_ts = 0.0
if 'recent_alert_notifications' not in st.session_state:
    st.session_state.recent_alert_notifications = []
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

# Restore saved portfolio after login when available.
if '_saved_portfolio' in st.session_state and st.session_state.get('_saved_portfolio'):
    if not st.session_state.portfolio_manager.portfolio:
        restored_portfolio = []
        for item in st.session_state.get('_saved_portfolio', []):
            if not isinstance(item, dict) or not item.get('symbol'):
                continue
            try:
                quantity = int(item.get('quantity', 0))
                buy_price = float(item.get('buy_price', 0))
                if quantity <= 0 or buy_price <= 0:
                    continue
                restored_portfolio.append({
                    'symbol': normalize_ticker_symbol(item.get('symbol')),
                    'quantity': quantity,
                    'buy_price': buy_price,
                    'buy_date': str(item.get('buy_date', datetime.now().strftime('%Y-%m-%d'))),
                })
            except Exception:
                continue
        if restored_portfolio:
            st.session_state.portfolio_manager.portfolio = restored_portfolio
    st.session_state.pop('_saved_portfolio', None)

# Apply saved profile settings once per authenticated session.
if 'user_settings_applied' not in st.session_state:
    st.session_state.user_settings_applied = False

if not st.session_state.user_settings_applied:
    saved_settings = (st.session_state.get('auth_user_data') or {}).get('settings', {})
    if isinstance(saved_settings, dict) and saved_settings:
        st.session_state.current_exchange = saved_settings.get('default_exchange', st.session_state.current_exchange)
        st.session_state.alert_webhook_url = saved_settings.get('alert_webhook_url', st.session_state.alert_webhook_url)
        st.session_state.enable_email_alerts = bool(saved_settings.get('enable_email_alerts', st.session_state.enable_email_alerts))
        st.session_state.alert_check_interval_sec = int(saved_settings.get('alert_check_interval_sec', st.session_state.alert_check_interval_sec))
        st.session_state.alert_repeat_cooldown_min = int(saved_settings.get('alert_repeat_cooldown_min', st.session_state.alert_repeat_cooldown_min))

        if 'risk_free_rate' in saved_settings:
            try:
                loaded_rfr = float(saved_settings.get('risk_free_rate'))
                st.session_state.risk_analytics.risk_free_rate = loaded_rfr / 100
                st.session_state.portfolio_risk.risk_analytics.risk_free_rate = loaded_rfr / 100
            except Exception:
                pass

    st.session_state.user_settings_applied = True

# Load NSE stocks
if st.session_state.nse_stocks is None:
    with st.spinner("Loading NSE stocks..."):
        st.session_state.nse_stocks = load_nse_stocks()

# Background-style alert polling on each rerun with interval throttling.
poll_result = process_price_alerts(force=False)
if poll_result.get("triggered"):
    for event in poll_result["triggered"][:3]:
        st.toast(
            f"Alert: {event['symbol']} {event['condition']} ₹{event['target_price']:.2f} (Now ₹{event.get('triggered_price', 0):.2f})",
            icon="🔔",
        )
    if len(poll_result["triggered"]) > 3:
        st.toast(f"{len(poll_result['triggered']) - 3} more alert(s) triggered.", icon="ℹ️")

if poll_result.get("expired"):
    st.toast(f"{len(poll_result['expired'])} alert(s) expired.", icon="⏱️")

# Header with centered logo
st.markdown("""
    <style>
    .header-logo-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem 0;
    }
    </style>
    <div class="header-logo-container">
""", unsafe_allow_html=True)

header_col1, header_col2, header_col3 = st.columns([1, 1, 1])

with header_col1:
    # User info
    if st.session_state.auth_username and st.session_state.auth_username != "__guest__":
        display_name = st.session_state.auth_user_data.get('full_name') or st.session_state.auth_username
        st.markdown(f"<p style='margin-top:2rem; color:#a0aec0;'>👤 <strong style='color:#667eea;'>{display_name}</strong></p>",
                    unsafe_allow_html=True)
    else:
        st.markdown("<p style='margin-top:2rem; color:#a0aec0;'>👤 Guest Mode</p>", unsafe_allow_html=True)

with header_col2:
    st.image("assets/logo_512.png", 
             width=280)

with header_col3:
    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
    lc1, lc2 = st.columns(2)
    with lc1:
        if st.button("⚙️ Settings", key="btn_settings_hdr", use_container_width=True):
            st.session_state.current_page = "Settings"
            st.rerun()
    with lc2:
        if st.button("🚪 Logout", key="btn_logout", use_container_width=True):
            # Save user data before logout
            if st.session_state.auth_username and st.session_state.auth_username != "__guest__":
                try:
                    save_user_watchlist(st.session_state.auth_username,
                                        st.session_state.get('watchlist', []))
                except Exception:
                    pass
                try:
                    save_user_portfolio(st.session_state.auth_username,
                                        st.session_state.portfolio_manager.portfolio)
                except Exception:
                    pass
                try:
                    save_user_alerts(st.session_state.auth_username,
                                     st.session_state.get('price_alerts', []))
                except Exception:
                    pass
                try:
                    save_user_trade_journal(st.session_state.auth_username,
                                            st.session_state.get('trade_journal', []))
                except Exception:
                    pass
            for k in ['authenticated', 'auth_username', 'auth_user_data']:
                st.session_state[k] = False if k == 'authenticated' else None
            st.session_state.user_settings_applied = False
            st.rerun()

st.markdown("""
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    .divider {
        border-top: 2px solid rgba(102, 126, 234, 0.3);
        margin: 20px 0;
    }
    </style>
    <div class="divider"></div>
""", unsafe_allow_html=True)

# Initialize page session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Stock Screener"

# Enhanced Navigation Menu with better styling
st.markdown("""
    <style>
    .nav-container {
        display: flex;
        gap: 10px;
        margin-bottom: 20px;
        flex-wrap: wrap;
    }
    .nav-item {
        flex: 1;
        min-width: 180px;
    }
    </style>
""", unsafe_allow_html=True)

nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4, gap="small")

nav_buttons_row1 = [
    ("Stock Screener", "btn_screener", nav_col1),
    ("Multi-Strategy", "btn_multi_strategy", nav_col2),
    ("Backtesting", "btn_backtest", nav_col3),
    ("Risk Analytics", "btn_risk", nav_col4),
]

for btn_text, btn_key, col in nav_buttons_row1:
    with col:
        is_active = st.session_state.current_page == btn_text
        btn_type = "primary" if is_active else "secondary"
        if st.button(btn_text, use_container_width=True, key=btn_key, type=btn_type):
            st.session_state.current_page = btn_text
            st.rerun()

nav_col5, nav_col6, nav_col7, nav_col8 = st.columns(4, gap="small")

nav_buttons_row2 = [
    ("Market Overview", "btn_market", nav_col5),
    ("Portfolio", "btn_portfolio", nav_col6),
    ("Stock Analysis", "btn_analysis", nav_col7),
    ("Watchlist", "btn_watchlist", nav_col8),
]

for btn_text, btn_key, col in nav_buttons_row2:
    with col:
        is_active = st.session_state.current_page == btn_text
        btn_type = "primary" if is_active else "secondary"
        if st.button(btn_text, use_container_width=True, key=btn_key, type=btn_type):
            st.session_state.current_page = btn_text
            st.rerun()

nav_col9, nav_col10, nav_col11, nav_col12 = st.columns(4, gap="small")

nav_buttons_row3 = [
    ("Price Alerts", "btn_alerts", nav_col9),
    ("Trade Journal", "btn_journal", nav_col10),
    ("Settings", "btn_settings", nav_col11),
]

for btn_text, btn_key, col in nav_buttons_row3:
    with col:
        is_active = st.session_state.current_page == btn_text
        btn_type = "primary" if is_active else "secondary"
        if st.button(btn_text, use_container_width=True, key=btn_key, type=btn_type):
            st.session_state.current_page = btn_text
            st.rerun()

st.markdown("""
    <style>
    .info-bar {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.4) 0%, rgba(118, 75, 162, 0.4) 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        font-weight: 600;
        margin-bottom: 2rem;
        border: 1px solid rgba(102, 126, 234, 0.3);
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2),
                    inset 0 1px 1px rgba(255, 255, 255, 0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeInUp 0.6s ease-out;
        letter-spacing: 0.5px;
    }
    
    .info-bar:hover {
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.4),
                    inset 0 1px 1px rgba(255, 255, 255, 0.15);
        transform: translateY(-4px);
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.5) 0%, rgba(118, 75, 162, 0.5) 100%);
        border-color: rgba(102, 126, 234, 0.5);
    }
    
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(102, 126, 234, 0.05);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #7b8ff5 0%, #8659b5 100%);
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.5);
    }
    </style>
    <div class="info-bar">
        Exchange: """ + st.session_state.current_exchange + """ | Stocks Loaded: """ + str(len(st.session_state.nse_stocks)) + """ | Strategies: """ + str(len(st.session_state.advanced_strategy_engine.strategies)) + """ | Exchanges: 5 (NSE, BSE, NYSE, NASDAQ, LSE) | v5.0
    </div>
""", unsafe_allow_html=True)

page = st.session_state.current_page


# ============================================================================
# PAGE DISPATCH
# ============================================================================

if page == "Stock Screener":
    render_screener_page()
elif page == "Multi-Strategy":
    render_multi_strategy_page()
elif page == "Backtesting":
    render_backtesting_page()
elif page == "Risk Analytics":
    render_risk_analytics_page()
elif page == "Market Overview":
    render_market_overview_page()
elif page == "Portfolio":
    render_portfolio_page()
elif page == "Stock Analysis":
    render_stock_analysis_page()
elif page == "Watchlist":
    render_watchlist_page()
elif page == "Price Alerts":
    render_price_alerts_page()
elif page == "Trade Journal":
    render_trade_journal_page()
elif page == "Settings":
    render_settings_page()

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <h3 style='color: #667eea;'>👁️ ARTHA DRISHTI</h3>
        <p><strong>ML Based Advanced Stock Screener v5.0</strong></p>
        <p>Multi-Exchange | 16 Strategies | Backtesting | Risk Analytics | Chart Patterns | Fundamentals</p>
        <p>⚠️ For educational purposes only. Not financial advice.</p>
        <p>Data: Yahoo Finance | News: NewsAPI | Built with Streamlit & ML</p>
    </div>
""", unsafe_allow_html=True)
