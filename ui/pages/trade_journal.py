"""Page: Trade Journal"""
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

def render_trade_journal_page():
    st.markdown("## 📘 Trade Journal")
    st.info("Log trades with outcome tags and attachments, then analyze your execution quality over time.")

    tab1, tab2, tab3 = st.tabs(["Log Trade", "Journal Entries", "Analytics"])

    with tab1:
        st.markdown("### Add Trade Entry")

        default_symbol = st.session_state.get("current_analysis", {}).get("symbol") if isinstance(st.session_state.get("current_analysis"), dict) else None
        symbol_index = st.session_state.nse_stocks.index(default_symbol) if default_symbol in st.session_state.nse_stocks else 0

        with st.form("journal_add_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                j_symbol = st.selectbox("Symbol", st.session_state.nse_stocks, index=symbol_index, key="j_symbol")
                j_side = st.selectbox("Side", ["Long", "Short"], key="j_side")
                j_outcome = st.selectbox("Outcome", ["Open", "Win", "Loss", "Breakeven"], key="j_outcome")

            with c2:
                j_entry_date = st.date_input("Entry Date", value=datetime.now().date(), key="j_entry_date")
                j_exit_date = st.date_input("Exit Date", value=datetime.now().date(), key="j_exit_date")
                j_quantity = st.number_input("Quantity", min_value=1, value=1, step=1, key="j_quantity")

            with c3:
                j_entry_price = st.number_input("Entry Price (₹)", min_value=0.01, value=100.0, step=0.5, key="j_entry_price")
                j_exit_price = st.number_input("Exit Price (₹)", min_value=0.01, value=100.0, step=0.5, key="j_exit_price")
                j_strategy = st.text_input("Strategy", value="", placeholder="e.g., Breakout, Mean Reversion", key="j_strategy")

            j_tags = st.multiselect(
                "Outcome Tags",
                ["Breakout", "Pullback", "Reversal", "News", "Swing", "Intraday", "FOMO", "Disciplined", "Late Entry", "Good Risk Control"],
                key="j_tags",
            )
            j_custom_tags = st.text_input("Custom Tags (comma separated)", key="j_custom_tags")
            j_notes = st.text_area("Notes", height=110, key="j_notes")
            j_attachment = st.file_uploader("Attach chart screenshot (optional)", type=["png", "jpg", "jpeg", "webp"], key="j_attachment")

            add_trade_submit = st.form_submit_button("➕ Save Trade", type="primary", use_container_width=True)

            if add_trade_submit:
                if j_exit_date < j_entry_date:
                    st.error("Exit date cannot be earlier than entry date.")
                else:
                    entry_val = float(j_entry_price) * int(j_quantity)
                    if j_outcome == "Open":
                        pnl = 0.0
                        pnl_pct = 0.0
                    else:
                        if j_side == "Long":
                            pnl = (float(j_exit_price) - float(j_entry_price)) * int(j_quantity)
                        else:
                            pnl = (float(j_entry_price) - float(j_exit_price)) * int(j_quantity)
                        pnl_pct = (pnl / entry_val) * 100 if entry_val > 0 else 0.0

                    custom_tags = [t.strip() for t in str(j_custom_tags).split(",") if t.strip()]
                    all_tags = list(dict.fromkeys(list(j_tags) + custom_tags))
                    attachment_name = save_journal_attachment(j_attachment) if j_attachment is not None else ""

                    new_trade = {
                        "id": datetime.now().strftime("%Y%m%d%H%M%S%f"),
                        "symbol": j_symbol,
                        "side": j_side,
                        "entry_date": str(j_entry_date),
                        "exit_date": str(j_exit_date),
                        "quantity": int(j_quantity),
                        "entry_price": float(j_entry_price),
                        "exit_price": float(j_exit_price),
                        "strategy": j_strategy.strip(),
                        "outcome": j_outcome,
                        "tags": all_tags,
                        "notes": j_notes.strip(),
                        "attachment": attachment_name,
                        "pnl": float(round(pnl, 2)),
                        "pnl_pct": float(round(pnl_pct, 2)),
                        "created_at": datetime.now().isoformat(),
                    }

                    st.session_state.trade_journal.append(new_trade)
                    if st.session_state.auth_username and st.session_state.auth_username != "__guest__":
                        try:
                            save_user_trade_journal(st.session_state.auth_username, st.session_state.trade_journal)
                        except Exception:
                            pass

                    st.success(f"Trade saved for {j_symbol}. P&L: ₹{pnl:.2f} ({pnl_pct:.2f}%).")
                    st.rerun()

    with tab2:
        st.markdown("### Journal Entries")
        entries = st.session_state.get("trade_journal", [])

        if entries:
            journal_df = pd.DataFrame(entries)
            if "tags" in journal_df.columns:
                journal_df["tags"] = journal_df["tags"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x or ""))

            filt_col1, filt_col2 = st.columns(2)
            with filt_col1:
                outcome_filter = st.multiselect(
                    "Filter by Outcome",
                    options=["Open", "Win", "Loss", "Breakeven"],
                    default=["Open", "Win", "Loss", "Breakeven"],
                    key="journal_outcome_filter",
                )
            with filt_col2:
                strategy_query = st.text_input("Filter by Strategy", key="journal_strategy_filter")

            filtered_df = journal_df[journal_df["outcome"].isin(outcome_filter)] if outcome_filter else journal_df.copy()
            if strategy_query.strip() and "strategy" in filtered_df.columns:
                filtered_df = filtered_df[
                    filtered_df["strategy"].astype(str).str.contains(strategy_query.strip(), case=False, na=False)
                ]

            display_cols = [
                "symbol", "side", "entry_date", "exit_date", "quantity", "entry_price", "exit_price",
                "outcome", "strategy", "tags", "pnl", "pnl_pct", "attachment"
            ]
            display_cols = [c for c in display_cols if c in filtered_df.columns]
            st.dataframe(filtered_df[display_cols], use_container_width=True, hide_index=True)

            del_col1, del_col2 = st.columns([3, 1])
            with del_col1:
                selected_trade_id = st.selectbox(
                    "Select Entry",
                    options=[e["id"] for e in entries],
                    format_func=lambda tid: next(
                        (
                            f"{e['entry_date']} | {e['symbol']} | {e['outcome']} | ₹{e.get('pnl', 0):.2f}"
                            for e in entries if e["id"] == tid
                        ),
                        tid,
                    ),
                    key="journal_delete_select",
                )
            with del_col2:
                st.markdown("<div style='height: 1.8rem;'></div>", unsafe_allow_html=True)
                if st.button("🗑️ Delete Entry", key="journal_delete_btn", use_container_width=True):
                    st.session_state.trade_journal = [e for e in entries if e.get("id") != selected_trade_id]
                    if st.session_state.auth_username and st.session_state.auth_username != "__guest__":
                        try:
                            save_user_trade_journal(st.session_state.auth_username, st.session_state.trade_journal)
                        except Exception:
                            pass
                    st.success("Trade entry deleted.")
                    st.rerun()

            selected_entry = next((e for e in entries if e.get("id") == selected_trade_id), None)
            attachment_name = selected_entry.get("attachment") if isinstance(selected_entry, dict) else ""
            if attachment_name:
                attachment_path = Path(__file__).parent / ".journal_attachments" / attachment_name
                if attachment_path.exists():
                    st.markdown("#### Attached Screenshot")
                    st.image(str(attachment_path), use_container_width=True)

            st.download_button(
                "📥 Download Journal (CSV)",
                filtered_df.to_csv(index=False),
                f"trade_journal_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv",
                use_container_width=True,
                key="journal_export_csv",
            )
        else:
            st.info("No journal entries yet. Log your first trade in the first tab.")

    with tab3:
        st.markdown("### Journal Analytics")
        entries = st.session_state.get("trade_journal", [])

        if entries:
            analytics_df = pd.DataFrame(entries)
            closed_df = analytics_df[analytics_df["outcome"] != "Open"].copy() if "outcome" in analytics_df.columns else pd.DataFrame()

            if not closed_df.empty:
                closed_df["pnl"] = pd.to_numeric(closed_df["pnl"], errors="coerce").fillna(0.0)
                closed_df["pnl_pct"] = pd.to_numeric(closed_df["pnl_pct"], errors="coerce").fillna(0.0)
                total_closed = len(closed_df)
                wins_df = closed_df[closed_df["pnl"] > 0]
                losses_df = closed_df[closed_df["pnl"] < 0]

                win_rate = (len(wins_df) / total_closed) * 100 if total_closed > 0 else 0
                avg_win = wins_df["pnl"].mean() if len(wins_df) > 0 else 0.0
                avg_loss = losses_df["pnl"].mean() if len(losses_df) > 0 else 0.0
                expectancy = (win_rate / 100) * avg_win + (1 - win_rate / 100) * avg_loss

                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Closed Trades", total_closed)
                with m2:
                    st.metric("Win Rate", f"{win_rate:.2f}%")
                with m3:
                    st.metric("Net P&L", f"₹{closed_df['pnl'].sum():.2f}")
                with m4:
                    st.metric("Expectancy", f"₹{expectancy:.2f} / trade")

                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Average Win", f"₹{avg_win:.2f}")
                with c2:
                    st.metric("Average Loss", f"₹{avg_loss:.2f}")

                closed_df["exit_date"] = pd.to_datetime(closed_df["exit_date"], errors="coerce")
                closed_df = closed_df.sort_values("exit_date")
                closed_df["cumulative_pnl"] = closed_df["pnl"].cumsum()

                fig_pnl = go.Figure()
                fig_pnl.add_trace(go.Scatter(
                    x=closed_df["exit_date"],
                    y=closed_df["cumulative_pnl"],
                    mode="lines+markers",
                    name="Cumulative P&L",
                    line=dict(color="#667eea", width=2),
                ))
                fig_pnl.update_layout(title="Cumulative P&L Curve", height=320, yaxis_title="₹")
                st.plotly_chart(fig_pnl, use_container_width=True)

                outcome_counts = closed_df["outcome"].value_counts().reset_index()
                outcome_counts.columns = ["Outcome", "Count"]
                fig_outcome = px.pie(outcome_counts, names="Outcome", values="Count", title="Outcome Distribution")
                st.plotly_chart(fig_outcome, use_container_width=True)
            else:
                st.info("No closed trades yet. Close at least one trade for analytics.")
        else:
            st.info("No journal entries yet.")


