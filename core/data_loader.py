"""Extracted from app.py"""
import yfinance as yf

import streamlit as st
import pandas as pd
import requests
from io import StringIO


@st.cache_data(ttl=86400)  # Cache for 24 hours
def load_nse_stocks():
    """Load all NSE stock symbols"""
    try:
        url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        df = pd.read_csv(StringIO(response.text))
        symbols = df['SYMBOL'].str.strip().tolist()
        return sorted(symbols)
    except Exception as e:
        st.error(f"Failed to load NSE stocks: {e}")
        # Fallback to sample stocks
        return ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", 
                "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK",
                "LT", "AXISBANK", "BAJFINANCE", "ASIANPAINT", "MARUTI"]

@st.cache_data(ttl=86400)
def load_nifty50():
    """Load NIFTY50 stocks"""
    nifty50 = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "HINDUNILVR", 
               "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK", "LT", "AXISBANK", 
               "BAJFINANCE", "ASIANPAINT", "MARUTI", "SUNPHARMA", "WIPRO", "POWERINDIA",
               "JSWSTEEL", "BAJAJFINSV", "HDFC", "LUPIN", "TECHM", "ULTRACEMCO",
               "DRREDDY", "ADANIPORTS", "ADANIGREEN", "ADANITRANS", "TITAN", "HCLTECH",
               "NESTLEIND", "BRITANNIA", "SIEMENS", "CUMMINSIND", "INDIGO", "DIVISLAB",
               "SHREECEM", "EICHERMOT", "HEROMOTOCO", "SBICARD", "MUTHOOTFIN", "LTIM",
               "PAGEIND", "TATASTEEL", "M&M", "BOSCHLTD", "GAIL", "IOC"]
    return sorted(nifty50)

@st.cache_data(ttl=86400)
def load_nifty500():
    """Load NIFTY500 stocks (top 500 by market cap)"""
    try:
        # Try to fetch from NSE, filter to NIFTY500
        url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        df = pd.read_csv(StringIO(response.text))
        symbols = df['SYMBOL'].str.strip().tolist()
        return sorted(symbols[:500])  # Return top 500
    except:
        # Fallback to NIFTY50 + additional stocks
        nifty50 = load_nifty50()
        additional = ["AARTI", "ABB", "ACRYSIL", "ADORWELD", "ADVANIHOTELS", 
                     "AEGISCHEM", "AGRITECH", "AMBUJACEM", "ANIKINDUSTRY", "APARINDS",
                     "APOLLOHOSP", "APOLLOTYRE", "ARVINDFARM", "ARVIND", "ASAHISONG"]
        return sorted(nifty50 + additional)


def fetch_realtime_nse_history(symbol, period="2y", interval="1d"):
    """Fetch fresh NSE history and patch latest quote into the final candle."""
    try:
        cleaned_symbol = str(symbol).strip().upper()
        full_symbol = cleaned_symbol if cleaned_symbol.endswith(".NS") else f"{cleaned_symbol}.NS"

        ticker = yf.Ticker(full_symbol)
        df = ticker.history(period=period, interval=interval, auto_adjust=False, prepost=True)
        if df is None or df.empty:
            return None

        df = df.copy()

        live_price = None
        live_volume = None
        try:
            fast_info = getattr(ticker, "fast_info", {}) or {}
            live_price = fast_info.get("lastPrice") or fast_info.get("regularMarketPrice")
            live_volume = fast_info.get("lastVolume") or fast_info.get("regularMarketVolume")
        except Exception:
            pass

        if live_price is None:
            try:
                info = ticker.info or {}
                live_price = info.get("regularMarketPrice") or info.get("currentPrice")
                live_volume = live_volume or info.get("regularMarketVolume")
            except Exception:
                pass

        if live_price is not None and len(df) > 0:
            last_idx = df.index[-1]
            live_price = float(live_price)
            df.at[last_idx, "Close"] = live_price

            if "High" in df.columns:
                try:
                    df.at[last_idx, "High"] = max(float(df.at[last_idx, "High"]), live_price)
                except Exception:
                    df.at[last_idx, "High"] = live_price

            if "Low" in df.columns:
                try:
                    df.at[last_idx, "Low"] = min(float(df.at[last_idx, "Low"]), live_price)
                except Exception:
                    df.at[last_idx, "Low"] = live_price

            if "Volume" in df.columns and live_volume is not None:
                try:
                    df.at[last_idx, "Volume"] = max(float(df.at[last_idx, "Volume"]), float(live_volume))
                except Exception:
                    pass

        return df
    except Exception:
        return None


def get_stock_data_cached(symbol, period="2y"):
    """Realtime stock data retrieval (legacy function name kept for compatibility)."""
    return fetch_realtime_nse_history(symbol, period=period, interval="1d")

