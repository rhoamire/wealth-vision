"""Extracted from app.py"""
import os
import ta

from core.auth import save_user_alerts
import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import re
from datetime import datetime, timedelta
from pathlib import Path
import smtplib
from email.message import EmailMessage
import requests


def compute_indicators(df):
    """Compute technical indicators"""
    if df is None or len(df) < 30:
        return df
    
    try:
        bb = ta.volatility.BollingerBands(df['Close'], window=20)
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_lower'] = bb.bollinger_lband()
        
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        
        adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14)
        df['+DMI'] = adx.adx_pos()
        df['-DMI'] = adx.adx_neg()
        df['ADX'] = adx.adx()
        
        df['Vol_MA'] = df['Volume'].rolling(20).mean()
    except:
        pass
    
    return df

def format_number(num):
    """Format large numbers"""
    if num >= 1e12:
        return f"₹{num/1e12:.2f}T"
    elif num >= 1e9:
        return f"₹{num/1e9:.2f}B"
    elif num >= 1e7:
        return f"₹{num/1e7:.2f}Cr"
    elif num >= 1e5:
        return f"₹{num/1e5:.2f}L"
    else:
        return f"₹{num:.2f}"


def _apply_hover_defaults_to_figure(fig):
    """Ensure hover labels remain enabled on Plotly figures."""
    if fig is None or not hasattr(fig, "update_layout"):
        return

    try:
        current_hovermode = getattr(getattr(fig, "layout", None), "hovermode", None)
        if current_hovermode in (None, "", False):
            fig.update_layout(hovermode="x unified")

        fig.update_layout(
            hoverdistance=30,
            spikedistance=30,
            hoverlabel=dict(
                namelength=-1,
                bgcolor="rgba(8, 14, 30, 0.96)",
                bordercolor="rgba(104, 180, 255, 0.85)",
                font=dict(color="#F4F8FF", size=13, family="Segoe UI, Inter, sans-serif"),
                align="left",
            ),
            transition=dict(duration=180, easing="cubic-in-out"),
        )

        fig.update_xaxes(
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikethickness=1,
            spikecolor="rgba(104, 180, 255, 0.65)",
        )
        fig.update_yaxes(
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikethickness=1,
            spikecolor="rgba(104, 180, 255, 0.45)",
        )
    except Exception:
        pass

    for trace in getattr(fig, "data", []):
        try:
            if getattr(trace, "hoverinfo", None) in ("skip", "none"):
                trace.hoverinfo = "x+y+name"

            trace_type = getattr(trace, "type", "")
            has_template = bool(getattr(trace, "hovertemplate", None))

            if trace_type in ("candlestick", "ohlc") and not has_template:
                trace.hovertemplate = (
                    "<b>%{x|%d %b %Y %H:%M}</b><br>"
                    "Open: %{open:,.2f}<br>"
                    "High: %{high:,.2f}<br>"
                    "Low: %{low:,.2f}<br>"
                    "Close: %{close:,.2f}"
                    "<extra>%{fullData.name}</extra>"
                )

            if trace_type in ("scatter", "scattergl", "bar") and not has_template:
                if hasattr(trace, "x") and hasattr(trace, "y"):
                    trace.hovertemplate = (
                        "<b>%{x|%d %b %Y %H:%M}</b><br>"
                        "Value: %{y:,.2f}"
                        "<extra>%{fullData.name}</extra>"
                    )
        except Exception:
            continue


def _merge_plotly_config(user_config=None):
    """Merge app-level Plotly defaults with per-chart config overrides."""
    merged = {
        "displayModeBar": True,
        "displaylogo": False,
        "scrollZoom": True,
        "doubleClick": "reset",
        "responsive": True,
        "staticPlot": False,
    }
    if isinstance(user_config, dict):
        merged.update(user_config)
    merged["staticPlot"] = False
    merged["displayModeBar"] = True
    return merged


if not getattr(st.plotly_chart, "_artha_hover_patch", False):
    _original_plotly_chart = st.plotly_chart

    def _plotly_chart_with_hover(fig, *args, **kwargs):
        _apply_hover_defaults_to_figure(fig)
        user_config = kwargs.pop("config", None)
        kwargs["config"] = _merge_plotly_config(user_config)
        return _original_plotly_chart(fig, *args, **kwargs)

    _plotly_chart_with_hover._artha_hover_patch = True
    st.plotly_chart = _plotly_chart_with_hover


def normalize_ticker_symbol(symbol):
    """Normalize symbol text for reliable comparisons across UI inputs."""
    cleaned = str(symbol).strip().upper()
    for suffix in [".NS", ".BO", ".L"]:
        if cleaned.endswith(suffix):
            cleaned = cleaned[:-len(suffix)]
    return cleaned


def parse_symbol_list(raw_text):
    """Parse comma/space/newline separated symbols and deduplicate them."""
    if not raw_text:
        return []

    symbols = []
    for token in re.split(r"[,;\s]+", raw_text):
        symbol = normalize_ticker_symbol(token)
        if symbol and symbol not in symbols:
            symbols.append(symbol)
    return symbols


def safe_parse_datetime(value):
    """Parse ISO datetime safely, returning None for invalid values."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def build_trade_plan(account_capital, risk_pct, entry_price, stop_loss, target_price):
    """Build a simple long-trade position sizing plan."""
    if account_capital <= 0 or risk_pct <= 0 or entry_price <= 0 or stop_loss <= 0:
        return {
            "valid": False,
            "error": "Capital, risk %, entry, and stop loss must be greater than zero.",
        }

    risk_per_share = entry_price - stop_loss
    if risk_per_share <= 0:
        return {
            "valid": False,
            "error": "Stop loss should be below entry price for a long setup.",
        }

    risk_budget = account_capital * (risk_pct / 100)
    shares_by_risk = int(risk_budget / risk_per_share)
    shares_by_capital = int(account_capital / entry_price)
    shares = min(shares_by_risk, shares_by_capital)

    if shares <= 0:
        return {
            "valid": False,
            "error": "Risk budget is too small for this entry/stop configuration.",
        }

    reward_per_share = target_price - entry_price
    position_value = shares * entry_price
    potential_loss = shares * risk_per_share
    potential_profit = shares * reward_per_share

    return {
        "valid": True,
        "risk_budget": risk_budget,
        "shares": shares,
        "position_value": position_value,
        "potential_loss": potential_loss,
        "potential_profit": potential_profit,
        "risk_reward": (potential_profit / potential_loss) if potential_loss > 0 else 0,
        "capital_utilization_pct": (position_value / account_capital) * 100,
        "risk_utilization_pct": (potential_loss / risk_budget) * 100 if risk_budget > 0 else 0,
        "capital_limited": shares_by_capital < shares_by_risk,
    }


def compute_trade_scenario(entry_price, stop_loss, target_price, win_probability_pct, capital):
    """Compute payoff and expectancy metrics for a custom long-trade scenario."""
    if entry_price <= 0 or stop_loss <= 0 or target_price <= 0 or capital <= 0:
        return {
            "valid": False,
            "error": "Entry, stop, target, and capital must be greater than zero.",
        }

    risk_per_share = entry_price - stop_loss
    reward_per_share = target_price - entry_price

    if risk_per_share <= 0:
        return {
            "valid": False,
            "error": "Stop loss should be below entry for a long scenario.",
        }

    if reward_per_share <= 0:
        return {
            "valid": False,
            "error": "Target should be above entry for a long scenario.",
        }

    win_probability = min(max(float(win_probability_pct), 0.0), 100.0) / 100.0
    loss_probability = 1.0 - win_probability

    shares = int(capital / entry_price)
    if shares <= 0:
        return {
            "valid": False,
            "error": "Capital is too small for even one share at this entry.",
        }

    pnl_if_win = shares * reward_per_share
    pnl_if_loss = shares * risk_per_share
    expected_value = (win_probability * pnl_if_win) - (loss_probability * pnl_if_loss)
    breakeven_win_rate = (risk_per_share / (risk_per_share + reward_per_share)) * 100
    payoff_ratio = reward_per_share / risk_per_share

    return {
        "valid": True,
        "shares": shares,
        "capital_used": shares * entry_price,
        "risk_per_share": risk_per_share,
        "reward_per_share": reward_per_share,
        "risk_reward": payoff_ratio,
        "pnl_if_win": pnl_if_win,
        "pnl_if_loss": pnl_if_loss,
        "expected_value": expected_value,
        "expected_value_pct": (expected_value / (shares * entry_price)) * 100,
        "breakeven_win_rate": breakeven_win_rate,
        "expectancy_r": (win_probability * payoff_ratio) - loss_probability,
    }


def compute_analysis_confluence(analysis, prediction=None, news_data=None, fundamental_score=None):
    """Create a weighted 0-100 confluence score from technical, ML, news, and fundamentals."""
    rec_action = str((analysis.get("recommendation") or {}).get("action", "HOLD")).upper()
    rec_score_map = {
        "STRONG BUY": 92,
        "BUY": 78,
        "HOLD": 55,
        "SELL": 32,
        "STRONG SELL": 15,
    }
    recommendation_score = rec_score_map.get(rec_action, 50)

    try:
        rsi = float(analysis.get("rsi", 50) or 50)
    except Exception:
        rsi = 50
    try:
        adx = float(analysis.get("adx", 20) or 20)
    except Exception:
        adx = 20
    try:
        volume_ratio = float(analysis.get("volume_ratio", 1) or 1)
    except Exception:
        volume_ratio = 1

    try:
        macd = float(analysis.get("macd", 0) or 0)
        macd_signal = float(analysis.get("macd_signal", 0) or 0)
    except Exception:
        macd = 0
        macd_signal = 0

    rsi_score = max(0.0, min(100.0, 100.0 - abs(rsi - 55.0) * 2.0))
    adx_score = max(0.0, min(100.0, (adx / 40.0) * 100.0))
    volume_score = max(0.0, min(100.0, volume_ratio * 50.0))
    macd_score = 70.0 if macd > macd_signal else 35.0

    technical_score = (
        0.35 * rsi_score
        + 0.30 * adx_score
        + 0.20 * volume_score
        + 0.15 * macd_score
    )

    prediction_confidence = float((prediction or {}).get("confidence", 50) or 50)
    expected_return = float((prediction or {}).get("expected_return", 0) or 0)
    expected_return_score = max(0.0, min(100.0, (expected_return + 10.0) * 2.5))
    prediction_score = max(0.0, min(100.0, 0.7 * prediction_confidence + 0.3 * expected_return_score))

    sentiment_raw = float((news_data or {}).get("sentiment_score", 0) or 0)
    sentiment_score = max(0.0, min(100.0, (sentiment_raw + 1.0) * 50.0))

    if fundamental_score is None:
        fundamental_score = 50.0
    fundamental_score = max(0.0, min(100.0, float(fundamental_score)))

    overall_score = (
        0.35 * technical_score
        + 0.25 * prediction_score
        + 0.15 * sentiment_score
        + 0.15 * fundamental_score
        + 0.10 * recommendation_score
    )

    if overall_score >= 75:
        label = "High Conviction Bullish"
    elif overall_score >= 60:
        label = "Moderately Bullish"
    elif overall_score >= 45:
        label = "Neutral"
    elif overall_score >= 30:
        label = "Cautious"
    else:
        label = "High Risk / Bearish"

    return {
        "overall_score": overall_score,
        "label": label,
        "components": {
            "technical": technical_score,
            "prediction": prediction_score,
            "sentiment": sentiment_score,
            "fundamental": fundamental_score,
            "recommendation": recommendation_score,
        },
    }


def record_analysis_history(analysis, prediction=None, news_data=None):
    """Append a compact stock-analysis snapshot to session history."""
    if not analysis:
        return

    st.session_state.setdefault("analysis_history", [])

    rec = analysis.get("recommendation", {}) if isinstance(analysis, dict) else {}
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": str(analysis.get("symbol", "")),
        "company": str(analysis.get("company_name", "")),
        "price": float(analysis.get("current_price", 0) or 0),
        "recommendation": str(rec.get("action", "HOLD")),
        "rec_score": float(rec.get("score", 0) or 0),
        "prediction_confidence": float((prediction or {}).get("confidence", 0) or 0),
        "expected_return": float((prediction or {}).get("expected_return", 0) or 0),
        "news_sentiment": str((news_data or {}).get("overall_sentiment", "Neutral")),
    }

    st.session_state.analysis_history.append(record)
    st.session_state.analysis_history = st.session_state.analysis_history[-100:]


def resolve_stock_universe(selected_exchange, stock_list_option, exchange_lists):
    """Resolve a screener stock-universe selection into concrete symbols."""
    if stock_list_option == "All NSE Stocks":
        return st.session_state.nse_stocks or []

    if stock_list_option.startswith("All ") and stock_list_option.endswith(" Stocks"):
        all_exchange_stocks = st.session_state.exchange_handler.load_exchange_stocks(selected_exchange)
        if all_exchange_stocks:
            return all_exchange_stocks

        merged = []
        for syms in exchange_lists.values():
            merged.extend(syms)
        return list(dict.fromkeys(merged))

    if stock_list_option in exchange_lists:
        return exchange_lists[stock_list_option]

    fallback = st.session_state.exchange_handler.load_exchange_stocks(selected_exchange)
    return fallback or []


def compute_buy_opportunity_score(strategy_confidence, expected_return_pct, prediction_confidence, risk_reward):
    """Compute a blended buy-opportunity score on a 0-100 scale."""
    upside_score = min(max(expected_return_pct, 0), 25) * 4
    risk_reward_score = min(max(risk_reward, 0), 4) * 25
    return (
        (0.50 * float(strategy_confidence))
        + (0.25 * float(upside_score))
        + (0.15 * float(prediction_confidence))
        + (0.10 * float(risk_reward_score))
    )


def run_detailed_stock_analysis(symbol):
    """Run stock, news, and prediction analysis and store it in session state."""
    analysis = st.session_state.stock_analyzer.analyze_stock(symbol)
    if not analysis:
        return None, {}, None

    st.session_state.current_analysis = analysis

    try:
        news_data = st.session_state.news_analyzer.analyze_stock_news(
            symbol,
            analysis.get("company_name", symbol),
        )
    except Exception:
        news_data = {}
    st.session_state.current_news = news_data

    try:
        prediction = st.session_state.price_predictor.predict_target_price(
            analysis["df"],
            sentiment_score=news_data.get("sentiment_score", 0),
            fundamental_score=50,
        )
    except Exception:
        prediction = None
    st.session_state.current_prediction = prediction

    record_analysis_history(analysis, prediction=prediction, news_data=news_data)

    return analysis, news_data, prediction


def save_journal_attachment(uploaded_file):
    """Save uploaded journal attachment and return local filename."""
    if uploaded_file is None:
        return ""

    attachments_dir = Path(__file__).parent / ".journal_attachments"
    attachments_dir.mkdir(exist_ok=True)

    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", uploaded_file.name)
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{safe_name}"
    path = attachments_dir / filename

    with path.open("wb") as f:
        f.write(uploaded_file.getbuffer())

    return filename


def _build_alert_payload(alert, current_price, status="triggered"):
    """Build a consistent alert payload for downstream notification channels."""
    target_price = float(alert.get("target_price", 0) or 0)
    distance_pct = ((current_price / target_price) - 1) * 100 if target_price > 0 else 0
    return {
        "symbol": alert.get("symbol"),
        "exchange": alert.get("exchange", "NSE"),
        "condition": alert.get("condition"),
        "target_price": target_price,
        "current_price": current_price,
        "distance_pct": distance_pct,
        "status": status,
        "note": alert.get("note", ""),
        "timestamp": datetime.now().isoformat(),
    }


def _send_webhook_alert(payload):
    """Send alert payload to configured webhook endpoint."""
    webhook_url = st.session_state.get("alert_webhook_url", "").strip()
    if not webhook_url:
        return False, "Webhook not configured"

    try:
        response = requests.post(webhook_url, json=payload, timeout=8)
        if 200 <= response.status_code < 300:
            return True, f"Webhook delivered ({response.status_code})"
        return False, f"Webhook failed ({response.status_code})"
    except Exception as e:
        return False, f"Webhook error: {e}"


def _send_email_alert(payload):
    """Send alert email via SMTP environment configuration."""
    if not st.session_state.get("enable_email_alerts", False):
        return False, "Email alerts disabled"

    recipient = ((st.session_state.get("auth_user_data") or {}).get("email") or "").strip()
    if not recipient:
        return False, "No recipient email in user profile"

    smtp_host = os.getenv("ALERT_SMTP_HOST", "").strip()
    smtp_user = os.getenv("ALERT_SMTP_USER", "").strip()
    smtp_pass = os.getenv("ALERT_SMTP_PASS", "").strip()
    smtp_from = os.getenv("ALERT_SMTP_FROM", smtp_user).strip()
    smtp_port = int(os.getenv("ALERT_SMTP_PORT", "587"))
    use_tls = os.getenv("ALERT_SMTP_TLS", "true").strip().lower() in {"1", "true", "yes", "on"}

    if not (smtp_host and smtp_user and smtp_pass and smtp_from):
        return False, "SMTP env vars missing"

    try:
        msg = EmailMessage()
        msg["Subject"] = (
            f"[Artha Drishti] Alert {payload['symbol']} {payload['condition']} "
            f"₹{payload['target_price']:.2f}"
        )
        msg["From"] = smtp_from
        msg["To"] = recipient
        msg.set_content(
            "\n".join([
                "Artha Drishti Price Alert",
                f"Symbol: {payload['symbol']} ({payload['exchange']})",
                f"Condition: price {payload['condition']} ₹{payload['target_price']:.2f}",
                f"Current: ₹{payload['current_price']:.2f}",
                f"Distance: {payload['distance_pct']:+.2f}%",
                f"Status: {payload['status']}",
                f"Time: {payload['timestamp']}",
                f"Note: {payload.get('note', '')}",
            ])
        )

        with smtplib.SMTP(smtp_host, smtp_port, timeout=12) as server:
            if use_tls:
                server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)

        return True, "Email delivered"
    except Exception as e:
        return False, f"Email error: {e}"


def dispatch_alert_notification(alert, current_price, status="triggered"):
    """Dispatch alert notifications across configured channels."""
    payload = _build_alert_payload(alert, current_price, status=status)
    webhook_ok, webhook_msg = _send_webhook_alert(payload)
    email_ok, email_msg = _send_email_alert(payload)

    channels = []
    if webhook_ok:
        channels.append("webhook")
    if email_ok:
        channels.append("email")
    if not channels:
        channels.append("in-app")

    st.session_state.setdefault("recent_alert_notifications", [])
    st.session_state.recent_alert_notifications.append({
        "message": (
            f"{payload['symbol']} {payload['condition']} ₹{payload['target_price']:.2f} "
            f"(Now ₹{payload['current_price']:.2f})"
        ),
        "channels": channels,
        "timestamp": payload["timestamp"],
    })
    st.session_state.recent_alert_notifications = st.session_state.recent_alert_notifications[-30:]

    return {
        "payload": payload,
        "channels": channels,
        "webhook_message": webhook_msg,
        "email_message": email_msg,
    }


def process_price_alerts(force=False):
    """Check active alerts, move expired alerts, and notify on triggered alerts."""
    active_alerts = st.session_state.get("price_alerts", [])
    if not active_alerts:
        return {"checked": False, "triggered": [], "expired": []}

    check_interval = int(st.session_state.get("alert_check_interval_sec", 60))
    now_ts = time.time()
    last_check = float(st.session_state.get("last_alert_check_ts", 0))

    if not force and (now_ts - last_check) < max(10, check_interval):
        return {"checked": False, "triggered": [], "expired": []}

    st.session_state.last_alert_check_ts = now_ts

    kept_alerts = []
    triggered_events = []
    expired_events = []
    now_dt = datetime.now()
    default_cooldown = int(st.session_state.get("alert_repeat_cooldown_min", 60))

    for alert in active_alerts:
        try:
            expires_at = safe_parse_datetime(alert.get("expires_at"))
            if expires_at and now_dt > expires_at:
                expired_event = {
                    **alert,
                    "status": "expired",
                    "triggered_at": now_dt.isoformat(),
                    "triggered_price": alert.get("current_price"),
                    "notification_channels": "none",
                }
                st.session_state.alert_history.append(expired_event)
                expired_events.append(expired_event)
                continue

            data = st.session_state.exchange_handler.get_stock_data(
                alert["symbol"], alert.get("exchange", "NSE"), period="5d"
            )
            if data is None or len(data) == 0:
                kept_alerts.append(alert)
                continue

            current_price = float(data["Close"].iloc[-1])
            alert["current_price"] = current_price

            condition_met = False
            if alert.get("condition") == "above" and current_price >= float(alert.get("target_price", 0)):
                condition_met = True
            elif alert.get("condition") == "below" and current_price <= float(alert.get("target_price", 0)):
                condition_met = True

            if not condition_met:
                kept_alerts.append(alert)
                continue

            repeat_enabled = bool(alert.get("repeat", False))
            cooldown_minutes = max(1, int(alert.get("cooldown_minutes", default_cooldown)))
            last_triggered = safe_parse_datetime(alert.get("last_triggered_at"))

            if repeat_enabled and last_triggered:
                elapsed_seconds = (now_dt - last_triggered).total_seconds()
                if elapsed_seconds < (cooldown_minutes * 60):
                    kept_alerts.append(alert)
                    continue

            notify_meta = dispatch_alert_notification(alert, current_price, status="triggered")
            event = {
                **alert,
                "status": "triggered",
                "triggered_at": now_dt.isoformat(),
                "triggered_price": current_price,
                "notification_channels": ", ".join(notify_meta["channels"]),
            }
            st.session_state.alert_history.append(event)
            triggered_events.append(event)

            if repeat_enabled:
                alert["last_triggered_at"] = now_dt.isoformat()
                kept_alerts.append(alert)

        except Exception:
            kept_alerts.append(alert)

    st.session_state.price_alerts = kept_alerts

    if (triggered_events or expired_events) and st.session_state.auth_username and st.session_state.auth_username != "__guest__":
        try:
            save_user_alerts(st.session_state.auth_username, st.session_state.price_alerts)
        except Exception:
            pass

    return {"checked": True, "triggered": triggered_events, "expired": expired_events}

