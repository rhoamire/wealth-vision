"""
Artha Drishti - ML Based Advanced Stock Screener v4.0
Features: Multi-Exchange, Multi-Strategy, Backtesting, Risk Analytics, Portfolio, News Sentiment, Price Predictions
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
import ta
from datetime import datetime, timedelta
import json
import time
import warnings
import os
import smtplib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
import requests
from io import StringIO
import concurrent.futures
from textblob import TextBlob
import re
from scipy import stats
from pathlib import Path
from email.message import EmailMessage

# New advanced modules
from multi_exchange import MultiExchangeHandler, MarketOverview, CrossExchangeComparator, EXCHANGE_CONFIG
from advanced_strategies import AdvancedStrategyEngine, SectorRotationDetector, compute_full_indicators
from risk_analytics import RiskAnalytics, PortfolioRiskAnalyzer, StockComparator
from backtester import Backtester, BACKTEST_STRATEGIES
from technical_patterns import TechnicalAnalyzer, SupportResistanceDetector, ChartPatternDetector
from fundamental_analysis import FundamentalAnalyzer
from export_utils import ExportManager
from auth import (register_user, authenticate, get_user_data,
                  save_user_watchlist, save_user_portfolio, save_user_alerts,
                  save_user_settings, save_user_trade_journal, change_password)

warnings.filterwarnings("ignore")

# NewsAPI Configuration
NEWS_API_KEY = "fe303478-aba4-479e-8bdb-b2b8feb2db64"  # Replace with your NewsAPI key from newsapi.org
NEWS_API_URL = "https://newsapi.org/v2/everything"

# ============================================================================
# NEWS SENTIMENT ANALYZER
# ============================================================================

class NewsSentimentAnalyzer:
    """Analyze news sentiment for stocks"""
    
    def __init__(self, api_key):
        self.api_key = api_key
    
    def get_news(self, symbol, company_name, days=7):
        """Fetch news for a stock"""
        try:
            # Search query
            query = f"{company_name} OR {symbol}"
            from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            params = {
                'q': query,
                'from': from_date,
                'sortBy': 'relevancy',
                'language': 'en',
                'pageSize': 10,
                'apiKey': self.api_key
            }
            
            response = requests.get(NEWS_API_URL, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])
                return articles
            else:
                return []
        except:
            return []
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of text"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                sentiment = "Positive"
                score = polarity
            elif polarity < -0.1:
                sentiment = "Negative"
                score = polarity
            else:
                sentiment = "Neutral"
                score = polarity
            
            return sentiment, score
        except:
            return "Neutral", 0
    
    def analyze_stock_news(self, symbol, company_name):
        """Complete news sentiment analysis"""
        articles = self.get_news(symbol, company_name)
        
        if not articles:
            return {
                'overall_sentiment': 'Neutral',
                'sentiment_score': 0,
                'news_count': 0,
                'articles': [],
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0
            }
        
        sentiments = []
        article_data = []
        
        for article in articles:
            title = article.get('title', '')
            description = article.get('description', '')
            text = f"{title} {description}"
            
            sentiment, score = self.analyze_sentiment(text)
            sentiments.append(score)
            
            article_data.append({
                'title': title,
                'description': description,
                'sentiment': sentiment,
                'score': score,
                'url': article.get('url', ''),
                'publishedAt': article.get('publishedAt', '')
            })
        
        # Calculate overall sentiment
        avg_sentiment = np.mean(sentiments) if sentiments else 0
        
        if avg_sentiment > 0.1:
            overall = "Positive"
        elif avg_sentiment < -0.1:
            overall = "Negative"
        else:
            overall = "Neutral"
        
        positive_count = sum(1 for s in sentiments if s > 0.1)
        negative_count = sum(1 for s in sentiments if s < -0.1)
        neutral_count = len(sentiments) - positive_count - negative_count
        
        return {
            'overall_sentiment': overall,
            'sentiment_score': avg_sentiment,
            'news_count': len(articles),
            'articles': article_data,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count
        }

# ============================================================================
# ADVANCED ML-BASED PRICE PREDICTOR
# ============================================================================

class PricePredictor:
    """Advanced ML-based price prediction with target, buy, sell, and stop loss using ensemble methods"""
    
    def __init__(self):
        self.model_price = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, 
                                                     max_depth=5, random_state=42)
        self.model_buy = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
        self.model_stop = GradientBoostingRegressor(n_estimators=80, learning_rate=0.08, 
                                                    max_depth=5, random_state=42)
        self.model_direction = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        self.scaler = StandardScaler()
        self.models_trained = False
    
    def _extract_comprehensive_features(self, df, sentiment_score=0):
        """Extract comprehensive features from price data"""
        if len(df) < 60:
            return None
        
        features = pd.DataFrame(index=df.index)
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        
        try:
            # Price-based features
            features['price_normalized'] = close / close.rolling(50).mean()
            features['returns_1d'] = close.pct_change(1)
            features['returns_5d'] = close.pct_change(5)
            features['returns_10d'] = close.pct_change(10)
            features['returns_20d'] = close.pct_change(20)
            
            # Volatility features
            features['volatility_10'] = close.pct_change().rolling(10).std()
            features['volatility_20'] = close.pct_change().rolling(20).std()
            features['volatility_30'] = close.pct_change().rolling(30).std()
            
            # Momentum indicators
            features['rsi_14'] = ta.momentum.RSIIndicator(close, window=14).rsi()
            features['rsi_21'] = ta.momentum.RSIIndicator(close, window=21).rsi()
            features['rsi_slope'] = features['rsi_14'].diff(5)
            
            # Stochastic
            stoch = ta.momentum.StochasticOscillator(high, low, close)
            features['stoch_k'] = stoch.stoch()
            features['stoch_d'] = stoch.stoch_signal()
            
            # MACD
            macd = ta.trend.MACD(close)
            features['macd'] = macd.macd()
            features['macd_signal'] = macd.macd_signal()
            features['macd_diff'] = macd.macd_diff()
            features['macd_slope'] = features['macd'].diff(3)
            
            # ADX
            adx = ta.trend.ADXIndicator(high, low, close, window=14)
            features['adx'] = adx.adx()
            features['adx_pos'] = adx.adx_pos()
            features['adx_neg'] = adx.adx_neg()
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
            features['bb_upper'] = bb.bollinger_hband()
            features['bb_lower'] = bb.bollinger_lband()
            features['bb_position'] = (close - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
            
            # ATR
            atr = ta.volatility.AverageTrueRange(high, low, close, window=14)
            features['atr'] = atr.average_true_range()
            features['atr_pct'] = features['atr'] / close
            
            # Volume features
            features['volume_ratio'] = volume / volume.rolling(20).mean()
            features['volume_trend'] = volume.rolling(5).mean() / volume.rolling(20).mean()
            features['volume_change'] = volume.pct_change()
            
            # Trend features
            sma_12 = close.rolling(12).mean()
            sma_26 = close.rolling(26).mean()
            sma_50 = close.rolling(50).mean()
            sma_200 = close.rolling(200).mean()
            
            features['price_sma12'] = close / sma_12
            features['price_sma26'] = close / sma_26
            features['price_sma50'] = close / sma_50
            features['price_sma200'] = close / sma_200
            features['sma12_sma26'] = sma_12 / sma_26
            features['sma50_sma200'] = sma_50 / sma_200
            
            # Support and Resistance
            features['support'] = low.rolling(20).min()
            features['resistance'] = high.rolling(20).max()
            features['distance_support'] = (close - features['support']) / features['support']
            features['distance_resistance'] = (features['resistance'] - close) / features['resistance']
            
            # Pattern detection - Higher High/Low patterns
            features['hh_ll'] = ((high > high.shift(1)) & (low > low.shift(1))).astype(int)
            features['lh_ll'] = ((high < high.shift(1)) & (low > low.shift(1))).astype(int)
            
            # Range and volatility patterns
            high_low_range = high - low
            features['candle_range'] = high_low_range / close
            features['candle_body'] = abs(close - df['Open']) / close
            
            # CCI (Commodity Channel Index)
            cci = ta.trend.CCIIndicator(high, low, close, window=20)
            features['cci'] = cci.cci()
            
            # Sentiment integration
            features['sentiment_factor'] = sentiment_score
            
            # Rate of Change
            features['roc'] = ta.momentum.ROCIndicator(close, window=12).roc()
            
            # Williams %R
            williams = ta.momentum.WilliamsRIndicator(high, low, close, lperiod=14)
            features['williams_r'] = williams.williams_r()
            
            # Fill NaN values
            features = features.fillna(method='bfill').fillna(method='ffill').fillna(0)
            
            return features
        
        except Exception as e:
            return None
    
    def _create_labels(self, df, lookahead=5):
        """Create labels for price targets"""
        labels_price = np.zeros(len(df))
        labels_buy = np.zeros(len(df))
        labels_stop = np.zeros(len(df))
        labels_direction = np.zeros(len(df))
        
        for i in range(len(df) - lookahead):
            current_price = df['Close'].iloc[i]
            future_high = df['High'].iloc[i:i+lookahead].max()
            future_low = df['Low'].iloc[i:i+lookahead].min()
            future_close = df['Close'].iloc[i+lookahead]
            
            labels_price[i] = future_high
            labels_buy[i] = future_low * 0.98
            labels_stop[i] = future_low * 0.95
            labels_direction[i] = 1 if future_close > current_price else 0
        
        return labels_price, labels_buy, labels_stop, labels_direction
    
    def _train_models(self, df, sentiment_score=0):
        """Train ensemble models on historical data"""
        try:
            if len(df) < 100:
                return False
            
            features = self._extract_comprehensive_features(df, sentiment_score)
            if features is None or features.empty:
                return False
            
            # Create labels
            labels_price, labels_buy, labels_stop, labels_direction = self._create_labels(df, lookahead=5)
            
            # Prepare training data
            valid_idx = ~(features.isna().any(axis=1) | pd.isna(labels_price))
            features_train = features[valid_idx].iloc[:-5]
            labels_price_train = labels_price[valid_idx][:-5]
            labels_buy_train = labels_buy[valid_idx][:-5]
            labels_stop_train = labels_stop[valid_idx][:-5]
            labels_direction_train = labels_direction[valid_idx][:-5]
            
            if len(features_train) < 50:
                return False
            
            # Scale features
            X_scaled = self.scaler.fit_transform(features_train)
            
            # Train models with Time Series Split
            tscv = TimeSeriesSplit(n_splits=3)
            
            for train_idx, val_idx in tscv.split(X_scaled):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_price_train, y_price_val = labels_price_train.iloc[train_idx], labels_price_train.iloc[val_idx]
                y_buy_train, y_buy_val = labels_buy_train.iloc[train_idx], labels_buy_train.iloc[val_idx]
                y_stop_train, y_stop_val = labels_stop_train.iloc[train_idx], labels_stop_train.iloc[val_idx]
                y_dir_train, y_dir_val = labels_direction_train.iloc[train_idx], labels_direction_train.iloc[val_idx]
                
                self.model_price.fit(X_train, y_price_train)
                self.model_buy.fit(X_train, y_buy_train)
                self.model_stop.fit(X_train, y_stop_train)
                self.model_direction.fit(X_train, y_dir_train)
            
            self.models_trained = True
            return True
        
        except Exception as e:
            return False
    
    def calculate_support_resistance(self, df, window=20):
        """Calculate support and resistance levels using multiple windows"""
        highs = df['High'].rolling(window).max()
        lows = df['Low'].rolling(window).min()
        
        resistance = highs.iloc[-1]
        support = lows.iloc[-1]
        
        return support, resistance
    
    def calculate_pivot_points(self, df):
        """Calculate pivot points"""
        high = df['High'].iloc[-1]
        low = df['Low'].iloc[-1]
        close = df['Close'].iloc[-1]
        
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        r2 = pivot + (high - low)
        s1 = 2 * pivot - high
        s2 = pivot - (high - low)
        
        return {
            'pivot': pivot,
            'r1': r1,
            'r2': r2,
            's1': s1,
            's2': s2
        }
    
    def calculate_atr_stops(self, df, multiplier=2):
        """Calculate ATR-based stop loss"""
        atr = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        current_price = df['Close'].iloc[-1]
        atr_value = atr.iloc[-1]
        
        stop_loss = current_price - (multiplier * atr_value)
        take_profit = current_price + (multiplier * atr_value * 1.5)
        
        return stop_loss, take_profit
    
    def predict_target_price(self, df, sentiment_score=0, fundamental_score=50):
        """Predict target price with ML models + technical analysis"""
        try:
            current_price = df['Close'].iloc[-1]
            
            if len(df) < 60:
                return None
            
            # Train models on historical data
            if not self.models_trained:
                self._train_models(df, sentiment_score)
            
            # Extract features for current state
            features = self._extract_comprehensive_features(df, sentiment_score)
            if features is None or features.empty:
                return None
            
            latest_features = features.iloc[-1:].values
            X_scaled = self.scaler.transform(latest_features)
            
            # Make predictions
            if self.models_trained:
                ml_target_price = self.model_price.predict(X_scaled)[0]
                ml_buy_price = self.model_buy.predict(X_scaled)[0]
                ml_stop_loss = self.model_stop.predict(X_scaled)[0]
                direction_pred = self.model_direction.predict(X_scaled)[0]
                direction_prob = self.model_direction.predict_proba(X_scaled)[0]
            else:
                ml_target_price = current_price * 1.05
                ml_buy_price = current_price * 0.98
                ml_stop_loss = current_price * 0.95
                direction_pred = 1
                direction_prob = [0.5, 0.5]
            
            # Technical Analysis Component
            sma_20 = df['Close'].rolling(20).mean().iloc[-1]
            sma_50 = df['Close'].rolling(50).mean().iloc[-1]
            trend_strength = (current_price - sma_50) / sma_50 * 100
            
            rsi = ta.momentum.RSIIndicator(df['Close']).rsi().iloc[-1]
            rsi_score = (rsi - 50) / 50
            
            macd = ta.trend.MACD(df['Close'])
            macd_value = macd.macd().iloc[-1]
            macd_signal = macd.macd_signal().iloc[-1]
            macd_bullish = 1 if macd_value > macd_signal else -1
            
            volatility = df['Close'].pct_change().std() * np.sqrt(252)
            
            support, resistance = self.calculate_support_resistance(df)
            distance_to_resistance = (resistance - current_price) / current_price * 100
            
            pivots = self.calculate_pivot_points(df)
            atr_stop, atr_target = self.calculate_atr_stops(df)
            
            # Technical score
            technical_score = (
                (trend_strength / 10) * 0.3 +
                rsi_score * 0.2 +
                macd_bullish * 0.3 +
                (distance_to_resistance / 10) * 0.2
            ) * 100
            
            # Sentiment Component
            sentiment_component = sentiment_score * 100
            
            # Fundamental Component
            fundamental_component = (fundamental_score - 50) / 50 * 100
            
            # Combined Score
            combined_score = (
                technical_score * 0.35 +
                sentiment_component * 0.25 +
                fundamental_component * 0.25 +
                direction_prob[1] * 40
            )
            
            # Blend ML and technical predictions
            target_price = (ml_target_price * 0.6 + (current_price * (1 + combined_score / 100 * volatility)) * 0.4)
            conservative_target = target_price * 0.85
            aggressive_target = target_price * 1.15
            
            buy_price = (ml_buy_price * 0.5 + min(current_price * 0.98, pivots['s1']) * 0.5)
            sell_price = max(target_price * 1.05, pivots['r1'])
            
            stop_loss = (ml_stop_loss * 0.5 + max(atr_stop, buy_price * 0.97) * 0.5)
            
            # Calculate confidence
            ml_confidence = direction_prob[1] * 70 + 15
            
            technical_confidence = 0
            if 40 < rsi < 60:
                technical_confidence += 20
            elif 30 < rsi < 70:
                technical_confidence += 15
            else:
                technical_confidence += 5
            
            if trend_strength > 5:
                technical_confidence += 20
            elif trend_strength > 0:
                technical_confidence += 10
            
            if macd_bullish > 0:
                technical_confidence += 15
            else:
                technical_confidence += 5
            
            volume_ratio = df['Volume'].iloc[-5:].mean() / df['Volume'].mean()
            if volume_ratio > 1.2:
                technical_confidence += 10
            elif volume_ratio > 0.8:
                technical_confidence += 5
            
            confidence = (ml_confidence * 0.6 + technical_confidence * 0.4)
            
            if volatility > 0.5:
                confidence -= 10
            
            confidence = min(max(confidence, 20), 95)
            
            time_horizon = int(30 / (volatility + 0.1))
            time_horizon = min(max(time_horizon, 7), 90)
            
            expected_return = ((target_price - current_price) / current_price) * 100
            
            return {
                'current_price': current_price,
                'target_price': target_price,
                'conservative_target': conservative_target,
                'aggressive_target': aggressive_target,
                'buy_price': buy_price,
                'sell_price': sell_price,
                'stop_loss': stop_loss,
                'confidence': confidence,
                'time_horizon': time_horizon,
                'expected_return': expected_return,
                'technical_score': technical_score,
                'sentiment_score': sentiment_component,
                'fundamental_score': fundamental_component,
                'support': support,
                'resistance': resistance,
                'pivot_points': pivots,
                'atr_stop': atr_stop,
                'atr_target': atr_target,
                'risk_reward': abs((target_price - buy_price) / (buy_price - stop_loss)) if (buy_price - stop_loss) > 0 else 0
            }
        
        except Exception as e:
            return None

# ============================================================================
# DATA LOADER
# ============================================================================

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

# ============================================================================
# STRATEGY ENGINE
# ============================================================================

class StrategyEngine:
    """Custom screening strategy builder"""
    
    AVAILABLE_INDICATORS = {
        'MACD': {'type': 'trend', 'params': ['fast', 'slow', 'signal']},
        'RSI': {'type': 'momentum', 'params': ['period', 'overbought', 'oversold']},
        'ADX': {'type': 'trend', 'params': ['period', 'threshold']},
        'Bollinger': {'type': 'volatility', 'params': ['period', 'std']},
        'Volume': {'type': 'volume', 'params': ['period', 'multiplier']},
        'SMA_Cross': {'type': 'trend', 'params': ['fast', 'slow']},
        'EMA_Cross': {'type': 'trend', 'params': ['fast', 'slow']},
        'Stochastic': {'type': 'momentum', 'params': ['k', 'd', 'smooth']},
        'CCI': {'type': 'momentum', 'params': ['period', 'threshold']},
        'ATR': {'type': 'volatility', 'params': ['period']},
    }
    
    def __init__(self):
        self.strategies = self.load_default_strategies()
    
    def load_default_strategies(self):
        """Load default strategies"""
        return {
            'Swing Trader': {
                'description': 'Medium-term momentum with trend confirmation',
                'rules': [
                    {'indicator': 'MACD', 'condition': 'bullish_cross', 'weight': 2.0},
                    {'indicator': 'RSI', 'condition': 'between', 'min': 50, 'max': 70, 'weight': 1.5},
                    {'indicator': 'ADX', 'condition': 'above', 'threshold': 25, 'weight': 1.5},
                    {'indicator': 'Volume', 'condition': 'above_avg', 'multiplier': 1.2, 'weight': 1.0},
                ]
            },
            'Momentum Hunter': {
                'description': 'High momentum stocks breaking out',
                'rules': [
                    {'indicator': 'RSI', 'condition': 'above', 'threshold': 60, 'weight': 2.0},
                    {'indicator': 'MACD', 'condition': 'bullish', 'weight': 1.5},
                    {'indicator': 'Volume', 'condition': 'above_avg', 'multiplier': 1.5, 'weight': 2.0},
                    {'indicator': 'SMA_Cross', 'condition': 'golden', 'fast': 20, 'slow': 50, 'weight': 1.5},
                ]
            },
            'Value Bounce': {
                'description': 'Oversold stocks with reversal potential',
                'rules': [
                    {'indicator': 'RSI', 'condition': 'below', 'threshold': 40, 'weight': 2.0},
                    {'indicator': 'Bollinger', 'condition': 'near_lower', 'weight': 1.5},
                    {'indicator': 'MACD', 'condition': 'turning_up', 'weight': 1.5},
                    {'indicator': 'Volume', 'condition': 'above_avg', 'multiplier': 1.0, 'weight': 1.0},
                ]
            },
            'Trend Rider': {
                'description': 'Strong trending stocks',
                'rules': [
                    {'indicator': 'ADX', 'condition': 'above', 'threshold': 30, 'weight': 2.5},
                    {'indicator': 'EMA_Cross', 'condition': 'bullish', 'fast': 12, 'slow': 26, 'weight': 2.0},
                    {'indicator': 'RSI', 'condition': 'above', 'threshold': 55, 'weight': 1.5},
                    {'indicator': 'Volume', 'condition': 'trending_up', 'weight': 1.0},
                ]
            }
        }
    
    def evaluate_strategy(self, df, strategy_name):
        """Evaluate a strategy on stock data"""
        if strategy_name not in self.strategies:
            return None
        
        strategy = self.strategies[strategy_name]
        df = compute_indicators(df)
        
        if df is None or len(df) < 50:
            return None
        
        total_score = 0
        total_weight = 0
        conditions_met = []
        
        for rule in strategy['rules']:
            indicator = rule['indicator']
            condition = rule['condition']
            weight = rule.get('weight', 1.0)
            
            met = self._evaluate_rule(df, rule)
            
            if met:
                total_score += weight
                conditions_met.append(f"{indicator}:{condition}")
            
            total_weight += weight
        
        confidence = (total_score / total_weight * 100) if total_weight > 0 else 0
        
        return {
            'confidence': confidence,
            'conditions_met': len(conditions_met),
            'total_conditions': len(strategy['rules']),
            'details': conditions_met
        }
    
    def _evaluate_rule(self, df, rule):
        """Evaluate a single rule"""
        try:
            indicator = rule['indicator']
            condition = rule['condition']
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            
            if indicator == 'MACD':
                if condition == 'bullish_cross':
                    return latest['MACD'] > latest['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal']
                elif condition == 'bullish':
                    return latest['MACD'] > 0
                elif condition == 'turning_up':
                    return latest['MACD'] > prev['MACD']
            
            elif indicator == 'RSI':
                if condition == 'above':
                    return latest['RSI'] > rule.get('threshold', 50)
                elif condition == 'below':
                    return latest['RSI'] < rule.get('threshold', 50)
                elif condition == 'between':
                    return rule.get('min', 0) < latest['RSI'] < rule.get('max', 100)
            
            elif indicator == 'ADX':
                if condition == 'above':
                    return latest['ADX'] > rule.get('threshold', 25)
            
            elif indicator == 'Bollinger':
                if condition == 'near_lower':
                    distance = (latest['Close'] - latest['BB_lower']) / (latest['BB_upper'] - latest['BB_lower'])
                    return distance < 0.2
            
            elif indicator == 'Volume':
                if condition == 'above_avg':
                    return latest['Volume'] > latest['Vol_MA'] * rule.get('multiplier', 1.0)
                elif condition == 'trending_up':
                    vol_5 = df['Volume'].iloc[-5:].mean()
                    vol_20 = df['Volume'].iloc[-20:].mean()
                    return vol_5 > vol_20 * 1.1
            
            elif indicator == 'SMA_Cross':
                if condition == 'golden':
                    fast = rule.get('fast', 20)
                    slow = rule.get('slow', 50)
                    sma_fast = df['Close'].rolling(fast).mean()
                    sma_slow = df['Close'].rolling(slow).mean()
                    return sma_fast.iloc[-1] > sma_slow.iloc[-1] and sma_fast.iloc[-2] <= sma_slow.iloc[-2]
            
            elif indicator == 'EMA_Cross':
                if condition == 'bullish':
                    fast = rule.get('fast', 12)
                    slow = rule.get('slow', 26)
                    ema_fast = df['Close'].ewm(span=fast).mean()
                    ema_slow = df['Close'].ewm(span=slow).mean()
                    return ema_fast.iloc[-1] > ema_slow.iloc[-1]
            
            return False
        except:
            return False

# ============================================================================
# ML PREDICTOR (Enhanced)
# ============================================================================

class EnhancedMLPredictor:
    """Enhanced ML predictor with ensemble methods"""
    
    def __init__(self):
        self.models = {
            'rf': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1),
            'gb': GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, random_state=42)
        }
        self.scaler = StandardScaler()
    
    def create_advanced_features(self, df):
        """Create advanced features"""
        features = pd.DataFrame(index=df.index)
        
        try:
            features['returns_1d'] = df['Close'].pct_change(1)
            features['returns_5d'] = df['Close'].pct_change(5)
            features['returns_20d'] = df['Close'].pct_change(20)
            
            features['volatility_10d'] = df['Close'].pct_change().rolling(10).std()
            features['volatility_30d'] = df['Close'].pct_change().rolling(30).std()
            
            features['rsi'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
            features['rsi_slope'] = features['rsi'].diff(5)
            
            macd = ta.trend.MACD(df['Close'])
            features['macd'] = macd.macd()
            features['macd_signal'] = macd.macd_signal()
            features['macd_diff'] = macd.macd_diff()
            
            adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'])
            features['adx'] = adx.adx()
            
            features['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
            
            return features.fillna(method='ffill').fillna(0)
        except:
            return pd.DataFrame()
    
    def predict(self, symbol):
        """Make prediction for a symbol"""
        try:
            df = get_stock_data_cached(symbol)
            if df is None or len(df) < 150:
                return None
            
            features = self.create_advanced_features(df)
            if features.empty:
                return None
            
            # Simple prediction based on recent trend
            recent_return = (df['Close'].iloc[-1] / df['Close'].iloc[-10] - 1)
            
            if recent_return > 0.05:
                prediction = 2
                confidence = min(abs(recent_return) * 1000, 85)
            elif recent_return > 0.02:
                prediction = 1
                confidence = min(abs(recent_return) * 800, 70)
            else:
                prediction = 0
                confidence = 50
            
            return {
                'prediction': prediction,
                'confidence': confidence
            }
        except:
            return None

# ============================================================================
# PORTFOLIO MANAGER
# ============================================================================

class PortfolioManager:
    """Portfolio analysis and management"""
    
    def __init__(self):
        self.portfolio = []
    
    def add_stock(self, symbol, quantity, buy_price):
        """Add stock to portfolio"""
        self.portfolio.append({
            'symbol': symbol,
            'quantity': quantity,
            'buy_price': buy_price,
            'buy_date': datetime.now().strftime('%Y-%m-%d')
        })
    
    def remove_stock(self, symbol):
        """Remove stock from portfolio"""
        self.portfolio = [s for s in self.portfolio if s['symbol'] != symbol]
    
    def analyze_portfolio(self):
        """Analyze entire portfolio"""
        if not self.portfolio:
            return None
        
        results = []
        total_invested = 0
        total_current = 0
        
        for stock in self.portfolio:
            symbol = stock['symbol']
            quantity = stock['quantity']
            buy_price = stock['buy_price']
            
            df = get_stock_data_cached(symbol)
            if df is None:
                continue
            
            current_price = df['Close'].iloc[-1]
            invested = quantity * buy_price
            current_value = quantity * current_price
            pnl = current_value - invested
            pnl_pct = (pnl / invested) * 100
            
            total_invested += invested
            total_current += current_value
            
            results.append({
                'Symbol': symbol,
                'Quantity': quantity,
                'Buy Price': buy_price,
                'Current Price': current_price,
                'Invested': invested,
                'Current Value': current_value,
                'P&L': pnl,
                'P&L %': pnl_pct,
                'Buy Date': stock['buy_date']
            })
        
        summary = {
            'total_invested': total_invested,
            'total_current': total_current,
            'total_pnl': total_current - total_invested,
            'total_pnl_pct': ((total_current - total_invested) / total_invested * 100) if total_invested > 0 else 0,
            'stocks': results
        }
        
        return summary

# ============================================================================
# STOCK ANALYZER
# ============================================================================

class StockAnalyzer:
    """Comprehensive stock analysis"""
    
    def analyze_stock(self, symbol):
        """Comprehensive stock analysis"""
        try:
            df = get_stock_data_cached(symbol, period="2y")
            
            if df is None or df.empty:
                return None

            cleaned_symbol = str(symbol).strip().upper()
            full_symbol = cleaned_symbol if cleaned_symbol.endswith(".NS") else f"{cleaned_symbol}.NS"
            ticker = yf.Ticker(full_symbol)
            
            df = compute_indicators(df)
            info = ticker.info
            
            # Calculate metrics
            latest = df.iloc[-1]
            
            # Price metrics
            week_52_high = df['High'].rolling(252).max().iloc[-1]
            week_52_low = df['Low'].rolling(252).min().iloc[-1]
            current_price = latest['Close']
            
            # Returns
            returns_1m = (current_price / df['Close'].iloc[-21] - 1) * 100 if len(df) > 21 else 0
            returns_3m = (current_price / df['Close'].iloc[-63] - 1) * 100 if len(df) > 63 else 0
            returns_6m = (current_price / df['Close'].iloc[-126] - 1) * 100 if len(df) > 126 else 0
            returns_1y = (current_price / df['Close'].iloc[-252] - 1) * 100 if len(df) > 252 else 0
            
            # Volatility
            volatility = df['Close'].pct_change().std() * np.sqrt(252) * 100
            
            # Volume
            avg_volume = df['Volume'].mean()
            current_volume = latest['Volume']
            volume_ratio = current_volume / avg_volume
            
            # Technical signals
            rsi = latest['RSI']
            macd = latest['MACD']
            macd_signal = latest['MACD_Signal']
            adx = latest['ADX']
            
            # AI Recommendation
            recommendation = self.generate_recommendation(df, info)
            
            analysis = {
                'symbol': symbol,
                'company_name': info.get('longName', symbol),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'current_price': current_price,
                'week_52_high': week_52_high,
                'week_52_low': week_52_low,
                'pe_ratio': info.get('trailingPE', 0),
                'pb_ratio': info.get('priceToBook', 0),
                'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
                'returns_1m': returns_1m,
                'returns_3m': returns_3m,
                'returns_6m': returns_6m,
                'returns_1y': returns_1y,
                'volatility': volatility,
                'avg_volume': avg_volume,
                'current_volume': current_volume,
                'volume_ratio': volume_ratio,
                'rsi': rsi,
                'macd': macd,
                'macd_signal': macd_signal,
                'adx': adx,
                'recommendation': recommendation,
                'df': df
            }
            
            return analysis
            
        except Exception as e:
            st.error(f"Error analyzing {symbol}: {e}")
            return None
    
    def generate_recommendation(self, df, info):
        """Generate AI recommendation"""
        try:
            latest = df.iloc[-1]
            
            score = 0
            reasons = []
            
            # RSI analysis
            if latest['RSI'] > 70:
                score -= 2
                reasons.append("Overbought (RSI > 70)")
            elif latest['RSI'] < 30:
                score += 2
                reasons.append("Oversold (RSI < 30)")
            elif 50 < latest['RSI'] < 60:
                score += 1
                reasons.append("Healthy RSI")
            
            # MACD analysis
            if latest['MACD'] > latest['MACD_Signal']:
                score += 2
                reasons.append("Bullish MACD")
            else:
                score -= 1
                reasons.append("Bearish MACD")
            
            # ADX analysis
            if latest['ADX'] > 25:
                score += 1
                reasons.append("Strong trend")
            
            # Volume analysis
            volume_ratio = latest['Volume'] / df['Volume'].mean()
            if volume_ratio > 1.5:
                score += 1
                reasons.append("High volume")
            
            # Price vs SMA
            sma_50 = df['Close'].rolling(50).mean().iloc[-1]
            if latest['Close'] > sma_50:
                score += 1
                reasons.append("Above 50 SMA")
            else:
                score -= 1
                reasons.append("Below 50 SMA")
            
            # Generate recommendation
            if score >= 5:
                rec = "STRONG BUY"
                color = "success"
            elif score >= 3:
                rec = "BUY"
                color = "success"
            elif score >= 1:
                rec = "HOLD"
                color = "warning"
            elif score >= -2:
                rec = "SELL"
                color = "warning"
            else:
                rec = "STRONG SELL"
                color = "error"
            
            return {
                'action': rec,
                'score': score,
                'reasons': reasons,
                'color': color
            }
            
        except:
            return {
                'action': 'HOLD',
                'score': 0,
                'reasons': ['Insufficient data'],
                'color': 'warning'
            }

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

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

# ============================================================================
# STREAMLIT APP
# ============================================================================

st.set_page_config(
    page_title="Artha Drishti - ML Stock Screener",
    page_icon="C:\\Users\\Lenovo\\Documents\\VS CODE codes(files)\\helloworld\\BTP SEM-5\\arthadrishti_logo.png",
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
# AUTHENTICATION
# ============================================================================

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'auth_username' not in st.session_state:
    st.session_state.auth_username = None
if 'auth_user_data' not in st.session_state:
    st.session_state.auth_user_data = {}

def _show_auth_page():
    """Display login / registration page."""

    st.markdown("""
        <div style='text-align:center; padding:2rem 0;'>
            <h1 style='background: linear-gradient(90deg, #667eea, #764ba2);
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                        font-size:3rem; font-weight:900;'>👁️ ARTHA DRISHTI</h1>
            <p style='color:#a0aec0; font-size:1.15rem;'>ML-Based Advanced Stock Screener &amp; Analyzer</p>
        </div>
    """, unsafe_allow_html=True)

    auth_tab1, auth_tab2, auth_tab3 = st.tabs(["🔑 Login", "📝 Sign Up", "👤 Guest Access"])

    with auth_tab1:
        st.markdown("### Login to Your Account")
        with st.form("login_form"):
            login_user = st.text_input("Username", key="login_user")
            login_pass = st.text_input("Password", type="password", key="login_pass")
            login_submit = st.form_submit_button("Login", type="primary", use_container_width=True)

            if login_submit:
                if not login_user or not login_pass:
                    st.error("Please enter both username and password.")
                else:
                    success, result = authenticate(login_user, login_pass)
                    if success:
                        st.session_state.authenticated = True
                        st.session_state.auth_username = login_user.strip().lower()
                        st.session_state.auth_user_data = result
                        # Restore saved watchlist / portfolio
                        st.session_state.watchlist = result.get("watchlist", [])
                        st.session_state['_saved_portfolio'] = result.get("portfolio", [])
                        st.session_state['price_alerts'] = result.get("alerts", [])
                        st.session_state['trade_journal'] = result.get("trade_journal", [])

                        saved_settings = result.get("settings", {}) if isinstance(result.get("settings", {}), dict) else {}
                        if saved_settings:
                            st.session_state.current_exchange = saved_settings.get("default_exchange", st.session_state.get("current_exchange", "NSE"))
                            st.session_state.alert_webhook_url = saved_settings.get("alert_webhook_url", "")
                            st.session_state.enable_email_alerts = bool(saved_settings.get("enable_email_alerts", False))
                            st.session_state.alert_check_interval_sec = int(saved_settings.get("alert_check_interval_sec", 60))
                            st.session_state.alert_repeat_cooldown_min = int(saved_settings.get("alert_repeat_cooldown_min", 60))
                        st.session_state.user_settings_applied = False
                        st.success(f"Welcome back, {result.get('full_name') or login_user}!")
                        st.rerun()
                    else:
                        st.error(result)

    with auth_tab2:
        st.markdown("### Create a New Account")
        with st.form("register_form"):
            reg_name = st.text_input("Full Name", key="reg_name")
            reg_email = st.text_input("Email (optional)", key="reg_email")
            reg_user = st.text_input("Username (min 3 chars)", key="reg_user")
            reg_pass = st.text_input("Password (min 6 chars)", type="password", key="reg_pass")
            reg_pass2 = st.text_input("Confirm Password", type="password", key="reg_pass2")
            reg_submit = st.form_submit_button("Create Account", type="primary", use_container_width=True)

            if reg_submit:
                if reg_pass != reg_pass2:
                    st.error("Passwords do not match.")
                else:
                    success, msg = register_user(reg_user, reg_pass, reg_name, reg_email)
                    if success:
                        st.success(msg + " You can now log in.")
                    else:
                        st.error(msg)

    with auth_tab3:
        st.markdown("### Continue as Guest")
        st.info("Guest mode gives full access but your watchlist, portfolio, alerts, and trade journal won't be saved between sessions.")
        if st.button("🚀 Continue as Guest", type="primary", use_container_width=True, key="guest_btn"):
            st.session_state.authenticated = True
            st.session_state.auth_username = "__guest__"
            st.session_state.auth_user_data = {}
            st.session_state.watchlist = []
            st.session_state.price_alerts = []
            st.session_state.trade_journal = []
            st.session_state._saved_portfolio = []
            st.session_state.user_settings_applied = False
            st.rerun()


if not st.session_state.authenticated:
    _show_auth_page()
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
    st.image("C:\\Users\\Lenovo\\Documents\\VS CODE codes(files)\\helloworld\\BTP SEM-5\\arthadrishti_logo.png", 
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
# HELPER FUNCTION FOR STOCK ANALYSIS DISPLAY
# ============================================================================

def _display_stock_analysis(analysis, news_data, prediction):
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


# ============================================================================
# PAGE 1: STOCK SCREENER (with Multi-Exchange Support)
# ============================================================================

if page == "Stock Screener":
    st.markdown("## Multi-Exchange Stock Screener")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        selected_exchange = st.selectbox(
            "Select Exchange",
            st.session_state.exchange_handler.get_supported_exchanges(),
            index=0,
            key="screener_exchange"
        )
        st.session_state.current_exchange = selected_exchange
    
    with col2:
        # Get available stock lists for the exchange
        exchange_lists = st.session_state.exchange_handler.get_stock_lists(selected_exchange)
        list_options = list(exchange_lists.keys())
        if selected_exchange == "NSE":
            list_options.append("All NSE Stocks")
        # Always add "All Exchange Stocks" option
        list_options.append(f"All {selected_exchange} Stocks")
        stock_list_option = st.selectbox("Stock Universe", list_options) if list_options else "Default"
    
    with col3:
        strategy_name = st.selectbox(
            "Select Strategy",
            list(st.session_state.advanced_strategy_engine.strategies.keys()),
            key="screener_strategy"
        )
    
    with col4:
        min_confidence = st.slider("Min Confidence %", 30, 90, 60)
    
    # Display strategy info
    strat = st.session_state.advanced_strategy_engine.strategies[strategy_name]
    exchange_info = st.session_state.exchange_handler.get_exchange_info(selected_exchange)
    st.info(f"**Strategy:** {strat['description']} | **Risk:** {strat.get('risk_level', 'N/A')} | "
            f"**Timeframe:** {strat.get('timeframe', 'N/A')} | **Exchange:** {exchange_info.get('name', selected_exchange)}")

    stocks_to_scan = resolve_stock_universe(selected_exchange, stock_list_option, exchange_lists)
    
    if st.button("Start Screening", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        if not stocks_to_scan:
            progress_bar.empty()
            status_text.empty()
            st.warning("No stocks available in this universe. Try another list or exchange.")
        else:
            results = []

            total_symbols = len(stocks_to_scan)
            status_text.text(f"Fetching realtime data for {total_symbols} symbols...")
            progress_bar.progress(0.05)

            data_map = st.session_state.exchange_handler.get_bulk_stock_data(
                stocks_to_scan,
                selected_exchange,
                period="2y",
                interval="1d",
                include_live_quote=True,
            )
            bulk_meta = st.session_state.exchange_handler.get_last_bulk_meta()
            if bulk_meta:
                status_text.text(
                    f"Fetched {bulk_meta.get('fetched', 0)}/{bulk_meta.get('requested', total_symbols)} symbols "
                    f"using {bulk_meta.get('workers_min', 1)}-{bulk_meta.get('workers_max', 1)} workers"
                )

            for i, symbol in enumerate(stocks_to_scan):
                status_text.text(f"Scoring {symbol}... ({i+1}/{total_symbols})")

                try:
                    df = data_map.get(str(symbol).strip().upper())
                    if df is None:
                        continue

                    result = st.session_state.advanced_strategy_engine.evaluate_strategy(df, strategy_name)

                    if result and result['confidence'] >= min_confidence:
                        current_price = float(df['Close'].iloc[-1])

                        # Quick price levels for shortlist display.
                        try:
                            prediction = st.session_state.price_predictor.predict_target_price(
                                df, sentiment_score=0, fundamental_score=50
                            )
                            buy_price = float(prediction['buy_price']) if prediction else current_price * 0.98
                            target_price = float(prediction['target_price']) if prediction else current_price * 1.05
                            stop_loss = float(prediction['stop_loss']) if prediction else current_price * 0.97
                            expected_return = float(prediction['expected_return']) if prediction else 5.0
                            pred_confidence = float(prediction['confidence']) if prediction else 50.0
                        except Exception:
                            buy_price = current_price * 0.98
                            target_price = current_price * 1.05
                            stop_loss = current_price * 0.97
                            expected_return = 5.0
                            pred_confidence = 50.0

                        results.append({
                            'Symbol': symbol,
                            'Exchange': selected_exchange,
                            'Current Price': current_price,
                            'Buy Price': buy_price,
                            'Target Price': target_price,
                            'Stop Loss': stop_loss,
                            'Expected Return %': expected_return,
                            'Strategy Confidence': result['confidence'],
                            'Prediction Confidence': pred_confidence,
                            'Conditions Met': result['conditions_met'],
                            'Total Conditions': result['total_conditions'],
                            'Category': result.get('category', 'N/A'),
                            'Risk Level': result.get('risk_level', 'N/A'),
                        })
                except Exception:
                    pass

                progress_bar.progress(0.05 + ((i + 1) / max(1, total_symbols)) * 0.95)

            status_text.text("Screening Complete!")

            if results:
                st.session_state.screened_stocks = pd.DataFrame(results).sort_values(
                    'Strategy Confidence', ascending=False
                )
                st.success(f"Found {len(results)} stocks matching criteria!")
            else:
                st.warning("No stocks found. Try adjusting filters.")

    st.markdown("---")
    st.markdown("## Best Possible Buys (By Exchange)")
    st.caption("Ranks opportunities using strategy confidence, expected upside, risk/reward, and prediction confidence.")

    best_use_all_exchange = st.checkbox(
        f"Use all stocks from {selected_exchange} for best buys",
        value=stock_list_option.startswith("All "),
        key="best_buy_use_all_exchange",
    )

    if best_use_all_exchange:
        best_buy_universe = resolve_stock_universe(
            selected_exchange,
            f"All {selected_exchange} Stocks",
            exchange_lists,
        )
        best_universe_label = f"All {selected_exchange} Stocks"
    else:
        best_buy_universe = stocks_to_scan
        best_universe_label = stock_list_option

    universe_count = len(best_buy_universe)
    if universe_count > 0:
        st.caption(f"Universe selected: {best_universe_label} | Available symbols: {universe_count}")
    else:
        st.info("No symbols available for this exchange/list combination.")

    scan_limit_max = max(1, universe_count)
    default_scan_limit = min(75, scan_limit_max)

    bb_col1, bb_col2, bb_col3, bb_col4 = st.columns(4)
    with bb_col1:
        best_scan_all = st.checkbox(
            "Evaluate complete universe",
            value=universe_count <= 75 and universe_count > 0,
            disabled=universe_count == 0,
            key="best_buy_scan_all",
        )
    with bb_col2:
        best_scan_limit = int(
            st.number_input(
                "Stocks to evaluate",
                min_value=1,
                max_value=scan_limit_max,
                value=default_scan_limit,
                step=1,
                disabled=best_scan_all,
                key="best_buy_scan_limit",
            )
        )
    if best_scan_all and universe_count > 0:
        best_scan_limit = universe_count

    with bb_col3:
        best_top_n = int(
            st.number_input(
                "Top buy ideas",
                min_value=1,
                max_value=max(1, best_scan_limit),
                value=min(10, best_scan_limit),
                step=1,
                key="best_buy_top_n",
            )
        )
    with bb_col4:
        best_min_conf = st.slider(
            "Min confidence for best buys",
            30,
            90,
            max(55, min_confidence),
            key="best_buy_min_conf",
        )

    if st.button(
        "Find Best Possible Buys",
        type="primary",
        use_container_width=True,
        key="best_buy_scan_btn",
        disabled=universe_count == 0,
    ):
        if universe_count == 0:
            st.warning("No symbols available to evaluate.")
        else:
            symbols_to_evaluate = best_buy_universe[:best_scan_limit]
            progress_bar = st.progress(0)
            status_text = st.empty()
            best_results = []

            total_symbols = len(symbols_to_evaluate)
            status_text.text(f"Fetching realtime data for {total_symbols} symbols...")
            progress_bar.progress(0.05)

            data_map = st.session_state.exchange_handler.get_bulk_stock_data(
                symbols_to_evaluate,
                selected_exchange,
                period="2y",
                interval="1d",
                include_live_quote=True,
            )
            bulk_meta = st.session_state.exchange_handler.get_last_bulk_meta()
            if bulk_meta:
                status_text.text(
                    f"Fetched {bulk_meta.get('fetched', 0)}/{bulk_meta.get('requested', total_symbols)} symbols "
                    f"using {bulk_meta.get('workers_min', 1)}-{bulk_meta.get('workers_max', 1)} workers"
                )

            for i, symbol in enumerate(symbols_to_evaluate):
                status_text.text(f"Evaluating {symbol}... ({i+1}/{total_symbols})")

                try:
                    df = data_map.get(str(symbol).strip().upper())
                    if df is None or len(df) < 50:
                        continue

                    result = st.session_state.advanced_strategy_engine.evaluate_strategy(df, strategy_name)
                    if not result or result.get('confidence', 0) < best_min_conf:
                        continue

                    current_price = float(df['Close'].iloc[-1])
                    try:
                        prediction = st.session_state.price_predictor.predict_target_price(
                            df, sentiment_score=0, fundamental_score=50
                        )
                    except Exception:
                        prediction = None

                    buy_price = float(prediction['buy_price']) if prediction else current_price * 0.98
                    target_price = float(prediction['target_price']) if prediction else current_price * 1.05
                    stop_loss = float(prediction['stop_loss']) if prediction else current_price * 0.97
                    expected_return = float(prediction['expected_return']) if prediction else ((target_price / current_price) - 1) * 100
                    pred_confidence = float(prediction['confidence']) if prediction else 50.0

                    risk_pct = ((current_price - stop_loss) / current_price) * 100 if current_price > 0 else 0
                    risk_pct = max(risk_pct, 0.01)
                    risk_reward = expected_return / risk_pct
                    buy_score = compute_buy_opportunity_score(
                        result.get('confidence', 0),
                        expected_return,
                        pred_confidence,
                        risk_reward,
                    )

                    best_results.append({
                        'Symbol': symbol,
                        'Exchange': selected_exchange,
                        'Current Price': current_price,
                        'Buy Price': buy_price,
                        'Target Price': target_price,
                        'Stop Loss': stop_loss,
                        'Expected Return %': expected_return,
                        'Risk %': risk_pct,
                        'Risk/Reward': risk_reward,
                        'Strategy Confidence': float(result.get('confidence', 0)),
                        'Prediction Confidence': pred_confidence,
                        'Buy Score': float(buy_score),
                        'Category': result.get('category', 'N/A'),
                        'Risk Level': result.get('risk_level', 'N/A'),
                    })
                except Exception:
                    pass

                progress_bar.progress(0.05 + ((i + 1) / max(1, total_symbols)) * 0.95)

            status_text.text("Best-buy scan complete!")

            if best_results:
                best_df = pd.DataFrame(best_results).sort_values(
                    ['Buy Score', 'Strategy Confidence', 'Expected Return %'],
                    ascending=[False, False, False],
                ).head(best_top_n).reset_index(drop=True)
                best_df.insert(0, 'Rank', np.arange(1, len(best_df) + 1))

                st.session_state.best_buy_opportunities = best_df
                st.session_state.best_buy_context = {
                    'exchange': selected_exchange,
                    'stock_list_option': best_universe_label,
                    'strategy': strategy_name,
                    'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'symbols_evaluated': len(symbols_to_evaluate),
                }
                st.success(f"Found {len(best_df)} high-potential buy opportunities.")
            else:
                st.session_state.best_buy_opportunities = None
                st.warning("No best-buy opportunities found. Try lowering confidence or scanning more stocks.")

    if st.session_state.best_buy_opportunities is not None:
        best_ctx = st.session_state.get('best_buy_context', {})
        best_df = st.session_state.best_buy_opportunities.copy()

        st.markdown("### Top Buy Opportunities")
        st.caption(
            f"Generated: {best_ctx.get('generated_at', 'N/A')} | "
            f"Exchange: {best_ctx.get('exchange', 'N/A')} | "
            f"Strategy: {best_ctx.get('strategy', 'N/A')} | "
            f"Symbols Evaluated: {best_ctx.get('symbols_evaluated', 0)}"
        )

        met_col1, met_col2, met_col3, met_col4 = st.columns(4)
        with met_col1:
            st.metric("Top Ideas", len(best_df))
        with met_col2:
            st.metric("Avg Buy Score", f"{best_df['Buy Score'].mean():.1f}")
        with met_col3:
            st.metric("Avg Expected Return", f"{best_df['Expected Return %'].mean():.2f}%")
        with met_col4:
            st.metric("Best Symbol", best_df.iloc[0]['Symbol'])

        best_currency = st.session_state.exchange_handler.get_currency_symbol(
            best_ctx.get('exchange', selected_exchange)
        )
        display_best_df = best_df.copy()
        display_best_df['Current Price'] = display_best_df['Current Price'].map(lambda x: f"{best_currency}{x:.2f}")
        display_best_df['Buy Price'] = display_best_df['Buy Price'].map(lambda x: f"{best_currency}{x:.2f}")
        display_best_df['Target Price'] = display_best_df['Target Price'].map(lambda x: f"{best_currency}{x:.2f}")
        display_best_df['Stop Loss'] = display_best_df['Stop Loss'].map(lambda x: f"{best_currency}{x:.2f}")
        display_best_df['Expected Return %'] = display_best_df['Expected Return %'].map(lambda x: f"{x:.2f}%")
        display_best_df['Risk %'] = display_best_df['Risk %'].map(lambda x: f"{x:.2f}%")
        display_best_df['Risk/Reward'] = display_best_df['Risk/Reward'].map(lambda x: f"{x:.2f}")
        display_best_df['Strategy Confidence'] = display_best_df['Strategy Confidence'].map(lambda x: f"{x:.1f}%")
        display_best_df['Prediction Confidence'] = display_best_df['Prediction Confidence'].map(lambda x: f"{x:.1f}%")
        display_best_df['Buy Score'] = display_best_df['Buy Score'].map(lambda x: f"{x:.1f}")

        st.dataframe(display_best_df, use_container_width=True, height=320)

        act_col1, act_col2 = st.columns(2)
        with act_col1:
            best_csv = best_df.to_csv(index=False)
            st.download_button(
                "📥 Download Best Buys",
                best_csv,
                f"best_buys_{best_ctx.get('exchange', selected_exchange)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True,
                key="best_buys_download",
            )

        with act_col2:
            selected_best_symbol = st.selectbox(
                "Select best buy for detailed analysis",
                best_df['Symbol'].tolist(),
                key="best_buy_detail_select",
            )

        if st.button("🔍 Analyze Selected Best Buy", type="primary", use_container_width=True, key="best_buy_analyze"):
            with st.spinner(f"Analyzing {selected_best_symbol}..."):
                analysis, news_data, prediction = run_detailed_stock_analysis(selected_best_symbol)

                if analysis:
                    st.success(f"Analysis complete! Scroll down to view detailed analysis of {selected_best_symbol}")
                    st.markdown("---")
                    st.markdown(f"## Detailed Analysis: {selected_best_symbol}")
                    _display_stock_analysis(analysis, news_data, prediction)
                else:
                    st.error("Failed to analyze stock")
    
    # Display results
    if st.session_state.screened_stocks is not None:
        st.markdown("---")
        st.markdown("## Results")
        
        df = st.session_state.screened_stocks
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Stocks", len(df))
        with col2:
            st.metric("Avg Strategy Confidence", f"{df['Strategy Confidence'].mean():.1f}%")
        with col3:
            high = len(df[df['Strategy Confidence'] >= 75])
            st.metric("High Confidence", high)
        with col4:
            st.metric("Exchange", st.session_state.current_exchange)
        
        # Format display columns
        display_df = df.copy()
        curr = st.session_state.exchange_handler.get_currency_symbol(st.session_state.current_exchange)
        display_df['Current Price'] = display_df['Current Price'].apply(lambda x: f"{curr}{x:.2f}")
        display_df['Buy Price'] = display_df['Buy Price'].apply(lambda x: f"{curr}{x:.2f}")
        display_df['Target Price'] = display_df['Target Price'].apply(lambda x: f"{curr}{x:.2f}")
        display_df['Stop Loss'] = display_df['Stop Loss'].apply(lambda x: f"{curr}{x:.2f}")
        display_df['Expected Return %'] = display_df['Expected Return %'].apply(lambda x: f"{x:.2f}%")
        display_df['Strategy Confidence'] = display_df['Strategy Confidence'].apply(lambda x: f"{x:.1f}%")
        display_df['Prediction Confidence'] = display_df['Prediction Confidence'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(display_df, use_container_width=True, height=400)
        
        # Action buttons
        col1, col2 = st.columns(2)
        
        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download Results",
                csv,
                f"screening_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col2:
            # Quick analyze option
            selected_symbol = st.selectbox(
                "Select stock to analyze in detail",
                df['Symbol'].tolist(),
                key="quick_analyze_select"
            )
        
        if st.button("🔍 Analyze Selected Stock", type="primary", use_container_width=True):
            with st.spinner(f"Analyzing {selected_symbol}..."):
                analysis, news_data, prediction = run_detailed_stock_analysis(selected_symbol)

                if analysis:
                    st.success(f"Analysis complete! Scroll down to view detailed analysis of {selected_symbol}")
                    st.markdown("---")
                    st.markdown(f"## Detailed Analysis: {selected_symbol}")
                    _display_stock_analysis(analysis, news_data, prediction)
                else:
                    st.error("Failed to analyze stock")

# ============================================================================
# PAGE 2: MULTI-STRATEGY SCREENER
# ============================================================================

elif page == "Multi-Strategy":
    st.markdown("## 🎯 Multi-Strategy Screener")
    st.info("Evaluate stocks against 16 advanced strategies across multiple categories")

    tab1, tab2, tab3 = st.tabs(["Single Stock Analysis", "Batch Screening", "Strategy Explorer"])

    with tab1:
        st.markdown("### Analyze a Stock with All Strategies")

        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            exchange = st.selectbox("Exchange", st.session_state.exchange_handler.get_supported_exchanges(), key="ms_exchange")
            stock_lists = st.session_state.exchange_handler.get_stock_lists(exchange)
            if stock_lists:
                list_name = st.selectbox("Stock List", list(stock_lists.keys()), key="ms_list")
                stocks = stock_lists[list_name]
            else:
                stocks = st.session_state.nse_stocks
            symbol = st.selectbox("Stock", stocks, key="ms_symbol")

        with col2:
            min_confidence = st.slider("Minimum Confidence %", 30, 90, 60, 5, key="ms_conf")

        with col3:
            category_filter = st.selectbox("Category Filter",
                ["All"] + st.session_state.advanced_strategy_engine.get_categories(),
                key="ms_cat")

        if st.button("🔍 Run Multi-Strategy Analysis", type="primary", key="ms_run"):
            with st.spinner(f"Evaluating {symbol} against all strategies..."):
                df = st.session_state.exchange_handler.get_stock_data(symbol, exchange)

                if df is not None and len(df) >= 50:
                    screen_result = st.session_state.advanced_strategy_engine.multi_strategy_screen(
                        df, min_strategies_passing=2, min_confidence=min_confidence
                    )

                    if screen_result:
                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Strategies Passing", f"{screen_result['strategies_passing']}/{screen_result['strategies_evaluated']}")
                        with col2:
                            st.metric("Best Strategy", screen_result['best_strategy'])
                        with col3:
                            st.metric("Max Confidence", f"{screen_result['max_confidence']:.1f}%")
                        with col4:
                            qualifies_text = "✅ QUALIFIES" if screen_result['qualifies'] else "❌ DOES NOT QUALIFY"
                            st.metric("Verdict", qualifies_text)

                        st.markdown("---")

                        # Results table
                        all_results = screen_result['all_results']
                        if category_filter != "All":
                            all_results = [r for r in all_results if r['category'] == category_filter]

                        if all_results:
                            df_results = pd.DataFrame(all_results)
                            df_results = df_results[['strategy', 'category', 'confidence', 'conditions_met', 'total_conditions', 'risk_level']]
                            df_results.columns = ['Strategy', 'Category', 'Confidence %', 'Met', 'Total', 'Risk Level']

                            # Color code confidence
                            st.dataframe(
                                df_results.style.background_gradient(
                                    subset=['Confidence %'], cmap='RdYlGn', vmin=0, vmax=100
                                ),
                                use_container_width=True,
                                hide_index=True
                            )

                            # Bar chart
                            fig = go.Figure()
                            colors = ['#2ecc71' if r['confidence'] >= min_confidence else '#e74c3c' for r in all_results]
                            fig.add_trace(go.Bar(
                                x=[r['strategy'] for r in all_results],
                                y=[r['confidence'] for r in all_results],
                                marker_color=colors,
                                text=[f"{r['confidence']:.0f}%" for r in all_results],
                                textposition='outside'
                            ))
                            fig.add_hline(y=min_confidence, line_dash="dash", line_color="orange",
                                          annotation_text=f"Min Confidence: {min_confidence}%")
                            fig.update_layout(
                                title="Strategy Confidence Scores",
                                xaxis_title="Strategy", yaxis_title="Confidence %",
                                height=500, xaxis_tickangle=-45
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            # Category summary
                            st.markdown("#### Category Summary")
                            categories = {}
                            for r in screen_result['all_results']:
                                cat = r['category']
                                if cat not in categories:
                                    categories[cat] = []
                                categories[cat].append(r['confidence'])

                            cat_cols = st.columns(len(categories))
                            for i, (cat, confs) in enumerate(categories.items()):
                                with cat_cols[i]:
                                    avg_conf = np.mean(confs)
                                    st.metric(cat, f"{avg_conf:.1f}%", delta=f"{len([c for c in confs if c >= min_confidence])} passing")
                else:
                    st.error("Could not fetch stock data or insufficient data points.")

    with tab2:
        st.markdown("### Batch Screen Stocks")

        col1, col2 = st.columns(2)
        with col1:
            batch_exchange = st.selectbox("Exchange", st.session_state.exchange_handler.get_supported_exchanges(), key="ms_batch_ex")
            batch_lists = st.session_state.exchange_handler.get_stock_lists(batch_exchange)
            if batch_lists:
                batch_list_options = list(batch_lists.keys()) + [f"All {batch_exchange} Stocks"]
                batch_list_name = st.selectbox("Stock List", batch_list_options, key="ms_batch_list")
                if batch_list_name.startswith("All "):
                    all_stocks = st.session_state.exchange_handler.load_exchange_stocks(batch_exchange)
                    if all_stocks:
                        batch_stocks = all_stocks
                    else:
                        merged = []
                        for syms in batch_lists.values():
                            merged.extend(syms)
                        batch_stocks = list(dict.fromkeys(merged))
                else:
                    batch_stocks = batch_lists[batch_list_name]
            else:
                batch_stocks = st.session_state.nse_stocks
        with col2:
            max_stocks = st.slider("Max Stocks to Screen", 5, len(batch_stocks), min(15, len(batch_stocks)), key="ms_batch_max")
            batch_min_conf = st.slider("Min Confidence %", 30, 90, 60, 5, key="ms_batch_conf")

        if st.button("🚀 Run Batch Screening", type="primary", key="ms_batch_run"):
            results_list = []
            progress = st.progress(0)

            symbols_to_screen = batch_stocks[:max_stocks]
            progress.progress(0.05, text=f"Fetching realtime data for {len(symbols_to_screen)} stocks...")

            data_map = st.session_state.exchange_handler.get_bulk_stock_data(
                symbols_to_screen,
                batch_exchange,
                period="2y",
                interval="1d",
                include_live_quote=True,
            )
            bulk_meta = st.session_state.exchange_handler.get_last_bulk_meta()
            if bulk_meta:
                latency_ema = bulk_meta.get('latency_ema')
                latency_ema_text = f"{float(latency_ema):.3f}s" if latency_ema is not None else "N/A"
                st.caption(
                    f"Realtime fetch: {bulk_meta.get('fetched', 0)}/{bulk_meta.get('requested', len(symbols_to_screen))} symbols | "
                    f"Workers: {bulk_meta.get('workers_min', 1)}-{bulk_meta.get('workers_max', 1)} | "
                    f"Adaptive latency EMA: {latency_ema_text}"
                )

            for idx, sym in enumerate(symbols_to_screen):
                progress.progress(0.05 + ((idx + 1) / max(1, len(symbols_to_screen))) * 0.95, text=f"Screening {sym}...")
                try:
                    df = data_map.get(str(sym).strip().upper())
                    if df is not None and len(df) >= 50:
                        screen = st.session_state.advanced_strategy_engine.multi_strategy_screen(
                            df, min_confidence=batch_min_conf
                        )
                        if screen:
                            curr_price = df['Close'].iloc[-1]
                            results_list.append({
                                'Symbol': sym,
                                'Exchange': batch_exchange,
                                'Price': f"{curr_price:.2f}",
                                'Strategies Passing': screen['strategies_passing'],
                                'Best Strategy': screen['best_strategy'],
                                'Max Confidence': f"{screen['max_confidence']:.1f}%",
                                'Avg Confidence': f"{screen['avg_confidence']:.1f}%",
                                'Qualifies': "✅" if screen['qualifies'] else "❌"
                            })
                except Exception:
                    pass

            progress.empty()

            if results_list:
                df_batch = pd.DataFrame(results_list)
                df_batch = df_batch.sort_values('Strategies Passing', ascending=False)
                st.dataframe(df_batch, use_container_width=True, hide_index=True)
                st.success(f"Screened {len(results_list)} stocks. {len([r for r in results_list if r['Qualifies'] == '✅'])} qualified.")
            else:
                st.warning("No results found.")

    with tab3:
        st.markdown("### Strategy Explorer")
        st.markdown("Browse all 16 advanced strategies and their details")

        categories = st.session_state.advanced_strategy_engine.get_categories()
        for cat in categories:
            st.markdown(f"#### {cat}")
            strategies = st.session_state.advanced_strategy_engine.get_strategies_by_category(cat)
            for name, details in strategies.items():
                with st.expander(f"📋 {name} ({details.get('risk_level', 'N/A')} Risk)"):
                    st.write(f"**Description:** {details.get('description', 'N/A')}")
                    st.write(f"**Timeframe:** {details.get('timeframe', 'N/A')}")
                    st.write(f"**Risk Level:** {details.get('risk_level', 'N/A')}")
                    st.write(f"**Number of Rules:** {len(details.get('rules', []))}")
                    st.markdown("**Rules:**")
                    for i, rule in enumerate(details.get('rules', []), 1):
                        st.write(f"  {i}. `{rule['indicator']}` → {rule['condition']} (weight: {rule.get('weight', 1.0)})")

# ============================================================================
# PAGE 3: BACKTESTING
# ============================================================================

elif page == "Backtesting":
    st.markdown("## 📈 Strategy Backtesting")
    st.info("Backtest trading strategies against historical data to evaluate their performance")

    tab1, tab2 = st.tabs(["Single Strategy Backtest", "Strategy Comparison"])

    with tab1:
        st.markdown("### Backtest a Strategy")

        col1, col2, col3 = st.columns(3)
        with col1:
            bt_exchange = st.selectbox("Exchange", st.session_state.exchange_handler.get_supported_exchanges(), key="bt_ex")
            bt_lists = st.session_state.exchange_handler.get_stock_lists(bt_exchange)
            if bt_lists:
                bt_list_name = st.selectbox("Stock List", list(bt_lists.keys()), key="bt_list")
                bt_stocks = bt_lists[bt_list_name]
            else:
                bt_stocks = st.session_state.nse_stocks
            bt_symbol = st.selectbox("Stock", bt_stocks, key="bt_sym")

        with col2:
            bt_strategy = st.selectbox("Strategy", list(BACKTEST_STRATEGIES.keys()), key="bt_strat")
            bt_period = st.selectbox("Backtest Period", ["1y", "2y", "3y", "5y"], index=1, key="bt_period")

        with col3:
            bt_capital = st.number_input("Initial Capital", 10000, 10000000, 100000, step=10000, key="bt_cap")
            bt_commission = st.number_input("Commission %", 0.0, 1.0, 0.1, 0.01, key="bt_comm")

        if st.button("▶️ Run Backtest", type="primary", key="bt_run"):
            with st.spinner(f"Backtesting {bt_strategy} on {bt_symbol}..."):
                df = st.session_state.exchange_handler.get_stock_data(bt_symbol, bt_exchange, period=bt_period)

                if df is not None and len(df) >= 50:
                    bt = Backtester(
                        initial_capital=bt_capital,
                        commission_pct=bt_commission / 100
                    )
                    result = bt.run_backtest(df, BACKTEST_STRATEGIES[bt_strategy], strategy_name=bt_strategy)

                    if result and 'error' not in result:
                        # Performance metrics
                        st.markdown("### Performance Summary")
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.metric("Total Return", f"{result['total_return']:.2f}%")
                        with col2:
                            st.metric("CAGR", f"{result['cagr']:.2f}%")
                        with col3:
                            st.metric("Sharpe Ratio", f"{result['sharpe_ratio']:.2f}")
                        with col4:
                            st.metric("Max Drawdown", f"{result['max_drawdown']:.2f}%")
                        with col5:
                            st.metric("Win Rate", f"{result['win_rate']:.1f}%")

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Final Equity", f"{result['final_equity']:,.0f}")
                        with col2:
                            st.metric("Total Trades", result['total_trades'])
                        with col3:
                            st.metric("Profit Factor", f"{result['profit_factor']:.2f}")
                        with col4:
                            st.metric("Sortino Ratio", f"{result['sortino_ratio']:.2f}")

                        st.markdown("---")

                        # Equity curve
                        if 'equity_curve' in result and not result['equity_curve'].empty:
                            st.markdown("### Equity Curve")
                            equity_df = result['equity_curve']
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=equity_df.index, y=equity_df['equity'],
                                mode='lines', name='Portfolio Value',
                                line=dict(color='#667eea', width=2)
                            ))
                            fig.add_hline(y=bt_capital, line_dash="dash", line_color="gray",
                                          annotation_text="Initial Capital")
                            fig.update_layout(
                                title=f"Equity Curve - {bt_strategy} on {bt_symbol}",
                                yaxis_title="Portfolio Value", height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            # Drawdown chart
                            peak = equity_df['equity'].expanding(min_periods=1).max()
                            drawdown = (equity_df['equity'] - peak) / peak * 100
                            fig_dd = go.Figure()
                            fig_dd.add_trace(go.Scatter(
                                x=equity_df.index, y=drawdown,
                                mode='lines', fill='tozeroy', name='Drawdown',
                                line=dict(color='#e74c3c', width=1)
                            ))
                            fig_dd.update_layout(title="Drawdown", yaxis_title="Drawdown %", height=250)
                            st.plotly_chart(fig_dd, use_container_width=True)

                        # Trade log
                        if 'trades' in result and not result['trades'].empty:
                            st.markdown("### Trade Log")
                            trades_df = result['trades'].copy()
                            display_cols = [c for c in ['date', 'type', 'price', 'shares', 'pnl', 'pnl_pct'] if c in trades_df.columns]
                            if display_cols:
                                st.dataframe(trades_df[display_cols].tail(20), use_container_width=True, hide_index=True)

                        # Additional metrics
                        st.markdown("### Detailed Metrics")
                        detail_data = {
                            "Metric": ["Annualized Volatility", "Avg Win %", "Avg Loss %", "Max Win %", "Max Loss %", "Total Commission"],
                            "Value": [
                                f"{result['annualized_volatility']:.2f}%",
                                f"{result['avg_win_pct']:.2f}%",
                                f"{result['avg_loss_pct']:.2f}%",
                                f"{result['max_win_pct']:.2f}%",
                                f"{result['max_loss_pct']:.2f}%",
                                f"{result['total_commission']:,.2f}"
                            ]
                        }
                        st.dataframe(pd.DataFrame(detail_data), use_container_width=True, hide_index=True)

                        # Export backtest results
                        try:
                            bt_csv = st.session_state.export_manager.backtest_to_csv(result, bt_strategy)
                            st.download_button("📥 Download Backtest Results (CSV)", bt_csv,
                                               f"backtest_{bt_symbol}_{bt_strategy}_{bt_period}.csv", "text/csv",
                                               use_container_width=True, key="bt_export_single")
                        except Exception:
                            pass
                    else:
                        st.warning("Backtest produced no results. The strategy may not have generated any signals for this stock/period.")
                else:
                    st.error("Could not fetch data or insufficient data points.")

    with tab2:
        st.markdown("### Compare All Strategies")

        col1, col2, col3 = st.columns(3)
        with col1:
            cmp_exchange = st.selectbox("Exchange", st.session_state.exchange_handler.get_supported_exchanges(), key="cmp_ex")
            cmp_lists = st.session_state.exchange_handler.get_stock_lists(cmp_exchange)
            if cmp_lists:
                cmp_list_name = st.selectbox("Stock List", list(cmp_lists.keys()), key="cmp_list")
                cmp_stocks = cmp_lists[cmp_list_name]
            else:
                cmp_stocks = st.session_state.nse_stocks
            cmp_symbol = st.selectbox("Stock", cmp_stocks, key="cmp_sym")

        with col2:
            cmp_period = st.selectbox("Period", ["1y", "2y", "3y", "5y"], index=1, key="cmp_period")

        with col3:
            cmp_capital = st.number_input("Capital", 10000, 10000000, 100000, 10000, key="cmp_cap")

        if st.button("📊 Compare All Strategies", type="primary", key="cmp_run"):
            with st.spinner("Running all strategies..."):
                df = st.session_state.exchange_handler.get_stock_data(cmp_symbol, cmp_exchange, period=cmp_period)

                if df is not None and len(df) >= 50:
                    bt = Backtester(initial_capital=cmp_capital)
                    comparison = bt.compare_strategies(df, BACKTEST_STRATEGIES)

                    if comparison:
                        # Comparison table
                        cmp_data = []
                        for name, res in comparison.items():
                            cmp_data.append({
                                'Strategy': name,
                                'Total Return %': f"{res['total_return']:.2f}",
                                'CAGR %': f"{res['cagr']:.2f}",
                                'Sharpe': f"{res['sharpe_ratio']:.2f}",
                                'Sortino': f"{res['sortino_ratio']:.2f}",
                                'Max DD %': f"{res['max_drawdown']:.2f}",
                                'Win Rate %': f"{res['win_rate']:.1f}",
                                'Trades': res['total_trades'],
                                'Profit Factor': f"{res['profit_factor']:.2f}",
                            })

                        st.dataframe(pd.DataFrame(cmp_data), use_container_width=True, hide_index=True)

                        # Returns bar chart
                        fig = go.Figure()
                        strats = list(comparison.keys())
                        returns_vals = [comparison[s]['total_return'] for s in strats]
                        colors = ['#2ecc71' if r > 0 else '#e74c3c' for r in returns_vals]
                        fig.add_trace(go.Bar(x=strats, y=returns_vals, marker_color=colors,
                                             text=[f"{r:.1f}%" for r in returns_vals], textposition='outside'))
                        fig.update_layout(title="Total Return by Strategy", yaxis_title="Return %",
                                          height=400, xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)

                        # Equity curves overlay
                        st.markdown("### Equity Curves Comparison")
                        fig_eq = go.Figure()
                        for name, res in comparison.items():
                            if 'equity_curve' in res and not res['equity_curve'].empty:
                                fig_eq.add_trace(go.Scatter(
                                    x=res['equity_curve'].index,
                                    y=res['equity_curve']['equity'],
                                    mode='lines', name=name
                                ))
                        fig_eq.add_hline(y=cmp_capital, line_dash="dash", line_color="gray")
                        fig_eq.update_layout(title=f"Equity Curves - {cmp_symbol}",
                                             yaxis_title="Portfolio Value", height=500)
                        st.plotly_chart(fig_eq, use_container_width=True)

                        # Export comparison
                        try:
                            cmp_csv_data = pd.DataFrame(cmp_data).to_csv(index=False)
                            st.download_button("📥 Download Strategy Comparison (CSV)", cmp_csv_data,
                                               f"strategy_comparison_{cmp_symbol}_{cmp_period}.csv", "text/csv",
                                               use_container_width=True, key="bt_cmp_export")
                        except Exception:
                            pass
                    else:
                        st.warning("No strategies produced results for this stock.")
                else:
                    st.error("Could not fetch data or insufficient data.")

# ============================================================================
# PAGE 4: RISK ANALYTICS
# ============================================================================

elif page == "Risk Analytics":
    st.markdown("## 🛡️ Risk Analytics")
    st.info("Comprehensive risk analysis with VaR, drawdown, Sharpe, Sortino, and more")

    tab1, tab2, tab3 = st.tabs(["Single Stock Risk", "Stock Comparison", "Portfolio Risk"])

    with tab1:
        st.markdown("### Stock Risk Analysis")

        col1, col2, col3 = st.columns(3)
        with col1:
            ra_exchange = st.selectbox("Exchange", st.session_state.exchange_handler.get_supported_exchanges(), key="ra_ex")
            ra_lists = st.session_state.exchange_handler.get_stock_lists(ra_exchange)
            if ra_lists:
                ra_list_name = st.selectbox("Stock List", list(ra_lists.keys()), key="ra_list")
                ra_stocks = ra_lists[ra_list_name]
            else:
                ra_stocks = st.session_state.nse_stocks
            ra_symbol = st.selectbox("Stock", ra_stocks, key="ra_sym")

        with col2:
            ra_period = st.selectbox("Period", ["1y", "2y", "3y", "5y"], index=1, key="ra_period")

        with col3:
            ra_benchmark = st.checkbox("Include Benchmark Comparison", value=True, key="ra_bench")

        if st.button("🔍 Analyze Risk", type="primary", key="ra_run"):
            with st.spinner(f"Computing risk metrics for {ra_symbol}..."):
                df = st.session_state.exchange_handler.get_stock_data(ra_symbol, ra_exchange, period=ra_period)

                benchmark_df = None
                if ra_benchmark:
                    try:
                        from multi_exchange import EXCHANGE_CONFIG
                        idx_symbol = EXCHANGE_CONFIG.get(ra_exchange, {}).get('index', '^NSEI')
                        benchmark_df = yf.download(idx_symbol, period=ra_period, progress=False)
                        # Flatten multi-level columns from yf.download()
                        if benchmark_df is not None and isinstance(benchmark_df.columns, pd.MultiIndex):
                            benchmark_df.columns = benchmark_df.columns.get_level_values(0)
                    except:
                        pass

                if df is not None and len(df) >= 30:
                    metrics = st.session_state.risk_analytics.compute_all_metrics(df, benchmark_df)

                    if metrics:
                        # Return & Volatility
                        st.markdown("### Return & Volatility Metrics")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("CAGR", f"{metrics.get('cagr', 0):.2f}%")
                        with col2:
                            st.metric("Annual Volatility", f"{metrics.get('annualized_volatility', 0):.2f}%")
                        with col3:
                            st.metric("Cumulative Return", f"{metrics.get('cumulative_return', 0):.2f}%")
                        with col4:
                            st.metric("Daily Avg Return", f"{metrics.get('daily_mean_return', 0):.4f}%")

                        st.markdown("---")

                        # Risk-Adjusted
                        st.markdown("### Risk-Adjusted Metrics")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            sharpe = metrics.get('sharpe_ratio', 0)
                            sharpe_emoji = "🟢" if sharpe > 1 else ("🟡" if sharpe > 0 else "🔴")
                            st.metric("Sharpe Ratio", f"{sharpe:.3f} {sharpe_emoji}")
                        with col2:
                            sortino = metrics.get('sortino_ratio', 0)
                            st.metric("Sortino Ratio", f"{sortino:.3f}")
                        with col3:
                            calmar = metrics.get('calmar_ratio', 0)
                            st.metric("Calmar Ratio", f"{calmar:.3f}")
                        with col4:
                            info_r = metrics.get('information_ratio', 'N/A')
                            if isinstance(info_r, (int, float)):
                                st.metric("Information Ratio", f"{info_r:.3f}")
                            else:
                                st.metric("Information Ratio", "N/A")

                        st.markdown("---")

                        # Drawdown
                        st.markdown("### Drawdown Analysis")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2f}%")
                        with col2:
                            st.metric("Current Drawdown", f"{metrics.get('current_drawdown', 0):.2f}%")
                        with col3:
                            st.metric("Avg Drawdown", f"{metrics.get('avg_drawdown', 0):.2f}%")

                        # Drawdown chart
                        prices = df['Close']
                        peak = prices.expanding(min_periods=1).max()
                        dd_series = (prices - peak) / peak * 100
                        fig_dd = go.Figure()
                        fig_dd.add_trace(go.Scatter(
                            x=df.index, y=dd_series, mode='lines', fill='tozeroy',
                            name='Drawdown', line=dict(color='#e74c3c', width=1)
                        ))
                        fig_dd.update_layout(title="Drawdown Over Time", yaxis_title="Drawdown %", height=300)
                        st.plotly_chart(fig_dd, use_container_width=True)

                        st.markdown("---")

                        # VaR
                        st.markdown("### Value at Risk (VaR)")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("VaR (95%)", f"{metrics.get('var_95', 0):.4f}")
                        with col2:
                            st.metric("VaR (99%)", f"{metrics.get('var_99', 0):.4f}")
                        with col3:
                            st.metric("CVaR (95%)", f"{metrics.get('cvar_95', 0):.4f}")

                        st.markdown("---")

                        # Benchmark relative metrics
                        if ra_benchmark and 'beta' in metrics:
                            st.markdown("### Benchmark-Relative Metrics")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Beta", f"{metrics.get('beta', 0):.3f}")
                            with col2:
                                st.metric("Alpha (Annual)", f"{metrics.get('alpha_annual', 0):.3f}")
                            with col3:
                                st.metric("Treynor Ratio", f"{metrics.get('treynor_ratio', 0):.4f}")
                            with col4:
                                st.metric("Correlation", f"{metrics.get('correlation', 0):.3f}")

                        # Tail Risk
                        st.markdown("### Tail Risk & Distribution")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Skewness", f"{metrics.get('skewness', 0):.3f}")
                        with col2:
                            st.metric("Kurtosis", f"{metrics.get('kurtosis', 0):.3f}")
                        with col3:
                            st.metric("Worst Day", f"{metrics.get('worst_day', 0):.2f}%")
                        with col4:
                            st.metric("Best Day", f"{metrics.get('best_day', 0):.2f}%")

                        # Returns distribution
                        returns = df['Close'].pct_change().dropna() * 100
                        fig_hist = go.Figure()
                        fig_hist.add_trace(go.Histogram(
                            x=returns, nbinsx=50, name='Daily Returns',
                            marker_color='#667eea', opacity=0.75
                        ))
                        fig_hist.update_layout(title="Daily Returns Distribution",
                                               xaxis_title="Return %", yaxis_title="Frequency", height=350)
                        st.plotly_chart(fig_hist, use_container_width=True)

                        # Export risk metrics
                        try:
                            risk_csv = st.session_state.export_manager.risk_metrics_to_csv(metrics)
                            st.download_button("📥 Download Risk Metrics (CSV)", risk_csv,
                                               f"risk_{ra_symbol}_{ra_period}.csv", "text/csv",
                                               use_container_width=True, key="ra_export_single")
                        except Exception:
                            pass
                    else:
                        st.error("Could not compute risk metrics.")
                else:
                    st.error("Could not fetch stock data or insufficient data.")

    with tab2:
        st.markdown("### Compare Multiple Stocks")

        cmp_exchange_r = st.selectbox("Exchange", st.session_state.exchange_handler.get_supported_exchanges(), key="ra_cmp_ex")
        cmp_lists_r = st.session_state.exchange_handler.get_stock_lists(cmp_exchange_r)
        if cmp_lists_r:
            cmp_list_r = st.selectbox("Stock List", list(cmp_lists_r.keys()), key="ra_cmp_list")
            cmp_stocks_r = cmp_lists_r[cmp_list_r]
        else:
            cmp_stocks_r = st.session_state.nse_stocks

        selected_stocks = st.multiselect("Select Stocks to Compare (2-6)", cmp_stocks_r, default=cmp_stocks_r[:3] if len(cmp_stocks_r) >= 3 else cmp_stocks_r[:2], key="ra_cmp_stocks")
        cmp_period_r = st.selectbox("Period", ["1y", "2y", "3y"], index=0, key="ra_cmp_period")

        if st.button("📊 Compare Risk Profiles", type="primary", key="ra_cmp_run") and len(selected_stocks) >= 2:
            with st.spinner("Fetching data and computing metrics..."):
                stock_data = {}
                for sym in selected_stocks[:6]:
                    data = st.session_state.exchange_handler.get_stock_data(sym, cmp_exchange_r, period=cmp_period_r)
                    if data is not None and len(data) > 30:
                        stock_data[sym] = data

                if len(stock_data) >= 2:
                    comparison_result = st.session_state.stock_comparator.compare(stock_data)
                    if not comparison_result.empty:
                        st.markdown("### Comparison Table")
                        st.dataframe(comparison_result.round(3), use_container_width=True)

                        # Normalized price chart
                        st.markdown("### Normalized Performance")
                        fig_norm = go.Figure()
                        for sym, data in stock_data.items():
                            normalized = data['Close'] / data['Close'].iloc[0] * 100
                            fig_norm.add_trace(go.Scatter(
                                x=data.index, y=normalized, mode='lines', name=sym
                            ))
                        fig_norm.update_layout(title="Normalized Performance (Base=100)",
                                               yaxis_title="Value", height=400)
                        st.plotly_chart(fig_norm, use_container_width=True)

                        # Correlation matrix
                        st.markdown("### Correlation Matrix")
                        returns_dict = {}
                        for sym, data in stock_data.items():
                            ret_s = data['Close'].pct_change().dropna()
                            if hasattr(ret_s.index, 'tz') and ret_s.index.tz is not None:
                                ret_s = ret_s.copy()
                                ret_s.index = ret_s.index.tz_localize(None)
                            returns_dict[sym] = ret_s
                        corr_df = pd.DataFrame(returns_dict).corr()
                        fig_corr = go.Figure(data=go.Heatmap(
                            z=corr_df.values, x=corr_df.columns, y=corr_df.index,
                            colorscale='RdYlGn', zmin=-1, zmax=1,
                            text=corr_df.round(2).values, texttemplate="%{text}"
                        ))
                        fig_corr.update_layout(title="Return Correlation Matrix", height=400)
                        st.plotly_chart(fig_corr, use_container_width=True)

                        # Export comparison
                        try:
                            cmp_csv = st.session_state.export_manager.comparison_to_csv(comparison_result)
                            st.download_button("📥 Download Comparison (CSV)", cmp_csv,
                                               f"risk_comparison_{cmp_period_r}.csv", "text/csv",
                                               use_container_width=True, key="ra_export_cmp")
                        except Exception:
                            pass
                else:
                    st.error("Need at least 2 stocks with valid data.")

    with tab3:
        st.markdown("### Portfolio Risk Analysis")
        st.markdown("Add stocks and weights to analyze portfolio-level risk metrics")

        pf_exchange = st.selectbox("Exchange", st.session_state.exchange_handler.get_supported_exchanges(), key="ra_pf_ex")
        pf_lists = st.session_state.exchange_handler.get_stock_lists(pf_exchange)
        if pf_lists:
            pf_list_name = st.selectbox("Stock List", list(pf_lists.keys()), key="ra_pf_list")
            pf_available = pf_lists[pf_list_name]
        else:
            pf_available = st.session_state.nse_stocks

        pf_stocks = st.multiselect("Select Portfolio Stocks", pf_available,
                                    default=pf_available[:4] if len(pf_available) >= 4 else pf_available[:2],
                                    key="ra_pf_stocks")
        pf_period = st.selectbox("Period", ["1y", "2y", "3y"], index=0, key="ra_pf_period")

        if pf_stocks:
            st.markdown("**Set Weights (must sum to 1.0):**")
            weights = []
            equal_w = 1.0 / len(pf_stocks) if pf_stocks else 0.25
            cols = st.columns(min(len(pf_stocks), 4))
            for i, sym in enumerate(pf_stocks):
                with cols[i % 4]:
                    w = st.number_input(f"{sym}", 0.0, 1.0, round(equal_w, 2), 0.05, key=f"pf_w_{sym}")
                    weights.append(w)

            if st.button("📈 Analyze Portfolio Risk", type="primary", key="ra_pf_run"):
                total_w = sum(weights)
                if abs(total_w - 1.0) > 0.05:
                    st.warning(f"Weights sum to {total_w:.2f}. Normalizing to 1.0.")
                    weights = [w / total_w for w in weights]

                with st.spinner("Computing portfolio risk..."):
                    stock_data = {}
                    for sym in pf_stocks:
                        data = st.session_state.exchange_handler.get_stock_data(sym, pf_exchange, period=pf_period)
                        if data is not None and len(data) > 30:
                            stock_data[sym] = data

                    if len(stock_data) >= 2:
                        adj_weights = [weights[i] for i, s in enumerate(pf_stocks) if s in stock_data]
                        total_aw = sum(adj_weights)
                        adj_weights = [w / total_aw for w in adj_weights]

                        pf_result = st.session_state.portfolio_risk.analyze_portfolio_risk(stock_data, adj_weights)
                        if pf_result:
                            st.markdown("### Portfolio Metrics")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Portfolio Return", f"{pf_result.get('portfolio_return', 0)*100:.2f}%")
                            with col2:
                                st.metric("Portfolio Volatility", f"{pf_result.get('portfolio_volatility', 0)*100:.2f}%")
                            with col3:
                                st.metric("Portfolio Sharpe", f"{pf_result.get('portfolio_sharpe', 0):.3f}")
                            with col4:
                                st.metric("Diversification Ratio", f"{pf_result.get('diversification_ratio', 0):.3f}")

                            # Weights pie chart
                            st.markdown("### Portfolio Allocation")
                            stock_names = list(stock_data.keys())
                            fig_pie = go.Figure(data=[go.Pie(
                                labels=stock_names, values=adj_weights,
                                hole=0.4, textinfo='label+percent'
                            )])
                            fig_pie.update_layout(title="Portfolio Weights", height=350)
                            st.plotly_chart(fig_pie, use_container_width=True)
                    else:
                        st.error("Need at least 2 stocks with data for portfolio analysis.")

# ============================================================================
# PAGE 5: MARKET OVERVIEW
# ============================================================================

elif page == "Market Overview":
    st.markdown("## 🌍 Market Overview")
    st.info("Global market indices, sector performance, and market breadth analysis")

    tab1, tab2, tab3, tab4 = st.tabs(["Global Indices", "Sector Performance", "Sector Rotation", "Correlation Finder"])

    with tab1:
        st.markdown("### Global Market Indices")

        if st.button("🔄 Refresh Indices", key="mo_refresh"):
            st.rerun()

        with st.spinner("Loading global indices..."):
            indices = st.session_state.market_overview.get_all_indices()
            st.caption(f"Live snapshot refreshed: {datetime.now().strftime('%H:%M:%S')}")

            if indices:
                cols = st.columns(len(indices))
                for i, (exchange, data) in enumerate(indices.items()):
                    with cols[i]:
                        change_color = "normal" if data['change_pct'] >= 0 else "inverse"
                        st.metric(
                            f"{data['name']}",
                            f"{data['currency']}{data['value']:,.2f}",
                            delta=f"{data['change_pct']:.2f}%",
                            delta_color=change_color
                        )

                st.markdown("---")

                # Index comparison chart
                st.markdown("### Index Performance (1 Month)")
                fig_idx = go.Figure()
                for exchange_name, data in indices.items():
                    fig_idx.add_trace(go.Bar(
                        x=[data['name']], y=[data['change_pct']],
                        name=data['name'],
                        marker_color='#2ecc71' if data['change_pct'] >= 0 else '#e74c3c',
                        text=f"{data['change_pct']:.2f}%", textposition='outside'
                    ))
                fig_idx.update_layout(title="Daily Change (%)", yaxis_title="Change %",
                                      height=400, showlegend=False)
                st.plotly_chart(fig_idx, use_container_width=True)
            else:
                st.warning("Could not fetch market indices.")

    with tab2:
        st.markdown("### Sector Performance")

        sp_exchange = st.selectbox("Exchange", ["NSE", "NYSE"], key="mo_sp_ex")

        if st.button("📊 Load Sector Data", type="primary", key="mo_sp_run"):
            with st.spinner(f"Loading sector performance for {sp_exchange}..."):
                sector_data = st.session_state.market_overview.get_sector_performance(sp_exchange)

                if sector_data:
                    # Sector bar chart
                    sectors = list(sector_data.keys())
                    avg_returns = [sector_data[s]['avg_return'] for s in sectors]
                    colors = ['#2ecc71' if r >= 0 else '#e74c3c' for r in avg_returns]

                    fig_sec = go.Figure()
                    fig_sec.add_trace(go.Bar(
                        x=sectors, y=avg_returns, marker_color=colors,
                        text=[f"{r:.2f}%" for r in avg_returns], textposition='outside'
                    ))
                    fig_sec.update_layout(title=f"Sector Performance (1 Month) - {sp_exchange}",
                                          yaxis_title="Avg Return %", height=400)
                    st.plotly_chart(fig_sec, use_container_width=True)

                    # Sector details
                    st.markdown("### Sector Details")
                    for sector, data in sector_data.items():
                        with st.expander(f"📊 {sector}"):
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Avg Return", f"{data['avg_return']:.2f}%")
                            with col2:
                                st.metric("Best Stock", f"{data['best']:.2f}%")
                            with col3:
                                st.metric("Worst Stock", f"{data['worst']:.2f}%")
                            with col4:
                                st.metric("Stocks Analyzed", data['stocks_count'])
                else:
                    st.warning("No sector data available.")

    with tab3:
        st.markdown("### Sector Rotation Analysis")
        st.markdown("Identifies which sectors are gaining/losing momentum across different timeframes")

        sr_exchange = st.selectbox("Exchange", ["NSE", "NYSE"], key="mo_sr_ex")

        if st.button("🔄 Analyze Rotation", type="primary", key="mo_sr_run"):
            with st.spinner(f"Analyzing sector rotation for {sr_exchange}..."):
                rotation = st.session_state.sector_rotation.analyze_rotation(sr_exchange)

                if rotation:
                    # Phase summary
                    st.markdown("### Sector Phases")
                    phase_cols = st.columns(min(len(rotation), 4))
                    for i, (sector, data) in enumerate(rotation.items()):
                        with phase_cols[i % 4]:
                            phase = data['phase']
                            phase_emojis = {
                                "Strong Uptrend": "🟢",
                                "Early Uptrend": "🔵",
                                "Recovery": "🟡",
                                "Topping Out": "🟠",
                                "Downtrend": "🔴",
                                "Consolidation": "⚪"
                            }
                            emoji = phase_emojis.get(phase, "⚪")
                            st.metric(f"{emoji} {sector}", phase,
                                      delta=f"Score: {data['momentum_score']:.2f}")

                    st.markdown("---")

                    # Rotation heatmap
                    st.markdown("### Return Heatmap")
                    sectors_list = list(rotation.keys())
                    periods_list = list(next(iter(rotation.values()))['returns'].keys()) if rotation else []

                    if periods_list:
                        z_data = []
                        for sector in sectors_list:
                            row = [rotation[sector]['returns'].get(p, 0) for p in periods_list]
                            z_data.append(row)

                        fig_hm = go.Figure(data=go.Heatmap(
                            z=z_data, x=periods_list, y=sectors_list,
                            colorscale='RdYlGn', zmid=0,
                            text=[[f"{v:.2f}%" for v in row] for row in z_data],
                            texttemplate="%{text}", textfont={"size": 12}
                        ))
                        fig_hm.update_layout(title=f"Sector Returns Heatmap - {sr_exchange}", height=400)
                        st.plotly_chart(fig_hm, use_container_width=True)

                    # Momentum ranking
                    st.markdown("### Momentum Ranking")
                    ranked = sorted(rotation.items(), key=lambda x: x[1]['momentum_score'], reverse=True)
                    rank_data = []
                    for rank, (sector, data) in enumerate(ranked, 1):
                        rank_data.append({
                            'Rank': rank,
                            'Sector': sector,
                            'Phase': data['phase'],
                            'Momentum Score': f"{data['momentum_score']:.2f}",
                            '1W Return': f"{data['returns'].get('1W', 0):.2f}%",
                            '1M Return': f"{data['returns'].get('1M', 0):.2f}%",
                            '3M Return': f"{data['returns'].get('3M', 0):.2f}%",
                        })
                    st.dataframe(pd.DataFrame(rank_data), use_container_width=True, hide_index=True)
                else:
                    st.warning("No rotation data available.")

    with tab4:
        st.markdown("### 🔗 Correlation Finder")
        st.markdown("Discover correlated and uncorrelated stocks for diversification or pair trading")

        col1, col2, col3 = st.columns(3)
        with col1:
            corr_exchange = st.selectbox("Exchange", st.session_state.exchange_handler.get_supported_exchanges(), key="corr_ex")
            corr_lists = st.session_state.exchange_handler.get_stock_lists(corr_exchange)
            if corr_lists:
                corr_list_name = st.selectbox("Stock List", list(corr_lists.keys()), key="corr_list")
                corr_available = corr_lists[corr_list_name]
            else:
                corr_available = st.session_state.nse_stocks

        with col2:
            corr_stocks = st.multiselect(
                "Select Stocks (3-15)", corr_available,
                default=corr_available[:6] if len(corr_available) >= 6 else corr_available[:3],
                key="corr_stocks"
            )

        with col3:
            corr_period = st.selectbox("Period", ["3mo", "6mo", "1y", "2y"], index=2, key="corr_period")
            corr_method = st.selectbox("Method", ["pearson", "spearman", "kendall"], key="corr_method")

        if st.button("🔍 Find Correlations", type="primary", key="corr_run") and len(corr_stocks) >= 3:
            with st.spinner("Computing correlations..."):
                returns_dict = {}
                for sym in corr_stocks[:15]:
                    try:
                        data = st.session_state.exchange_handler.get_stock_data(sym, corr_exchange, period=corr_period)
                        if data is not None and len(data) > 20:
                            rets = data['Close'].pct_change().dropna()
                            if isinstance(rets, pd.DataFrame):
                                rets = rets.iloc[:, 0]
                            if hasattr(rets.index, 'tz') and rets.index.tz is not None:
                                rets = rets.copy()
                                rets.index = rets.index.tz_localize(None)
                            returns_dict[sym] = rets
                    except:
                        pass

                if len(returns_dict) >= 3:
                    returns_df = pd.DataFrame(returns_dict).dropna()
                    corr_matrix = returns_df.corr(method=corr_method)

                    # Correlation heatmap
                    st.markdown("### Correlation Matrix")
                    fig_corr = go.Figure(data=go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.index,
                        colorscale='RdYlGn', zmin=-1, zmax=1,
                        text=corr_matrix.round(2).values,
                        texttemplate="%{text}",
                        textfont={"size": 11}
                    ))
                    fig_corr.update_layout(title=f"Correlation Matrix ({corr_method.title()})", height=500)
                    st.plotly_chart(fig_corr, use_container_width=True)

                    # Top correlated and uncorrelated pairs
                    st.markdown("### Pair Analysis")
                    pairs = []
                    symbols = list(corr_matrix.columns)
                    for i in range(len(symbols)):
                        for j in range(i + 1, len(symbols)):
                            pairs.append({
                                'Stock A': symbols[i],
                                'Stock B': symbols[j],
                                'Correlation': corr_matrix.iloc[i, j]
                            })

                    pairs_df = pd.DataFrame(pairs).sort_values('Correlation', ascending=False)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### 🟢 Most Correlated (move together)")
                        top_corr = pairs_df.head(5).copy()
                        top_corr['Correlation'] = top_corr['Correlation'].apply(lambda x: f"{x:.3f}")
                        st.dataframe(top_corr, use_container_width=True, hide_index=True)

                    with col2:
                        st.markdown("#### 🔵 Least Correlated (diversification)")
                        low_corr = pairs_df.sort_values('Correlation', key=abs).head(5).copy()
                        low_corr['Correlation'] = low_corr['Correlation'].apply(lambda x: f"{x:.3f}")
                        st.dataframe(low_corr, use_container_width=True, hide_index=True)

                    st.markdown("#### 🔴 Most Negatively Correlated (hedge)")
                    neg_corr = pairs_df.tail(5).copy()
                    neg_corr['Correlation'] = neg_corr['Correlation'].apply(lambda x: f"{x:.3f}")
                    st.dataframe(neg_corr, use_container_width=True, hide_index=True)

                    # Scatter plot for top pair
                    if len(pairs_df) > 0:
                        st.markdown("### Scatter Plot - Top Correlated Pair")
                        top_pair = pairs_df.iloc[0]
                        sym_a, sym_b = top_pair['Stock A'], top_pair['Stock B']
                        fig_scatter = go.Figure()
                        fig_scatter.add_trace(go.Scatter(
                            x=returns_df[sym_a] * 100,
                            y=returns_df[sym_b] * 100,
                            mode='markers',
                            marker=dict(color='#667eea', size=4, opacity=0.5),
                            name=f"{sym_a} vs {sym_b}"
                        ))
                        fig_scatter.update_layout(
                            title=f"{sym_a} vs {sym_b} Daily Returns (Corr: {float(top_pair['Correlation']):.3f})",
                            xaxis_title=f"{sym_a} Return %",
                            yaxis_title=f"{sym_b} Return %",
                            height=400
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)

                    # Export
                    try:
                        corr_csv = corr_matrix.to_csv()
                        st.download_button("📥 Download Correlation Matrix (CSV)", corr_csv,
                                           f"correlation_{corr_exchange}_{corr_period}.csv", "text/csv",
                                           use_container_width=True, key="corr_export")
                    except Exception:
                        pass
                else:
                    st.error("Need at least 3 stocks with valid data.")

# ============================================================================
# PAGE 6: PORTFOLIO
# ============================================================================

elif page == "Portfolio":
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

# ============================================================================
# PAGE 7: STOCK ANALYSIS
# ============================================================================

elif page == "Stock Analysis":
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

# ============================================================================
# PAGE 8: WATCHLIST
# ============================================================================

elif page == "Watchlist":
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

# ============================================================================
# PAGE 9: PRICE ALERTS
# ============================================================================

elif page == "Price Alerts":
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


# ============================================================================
# PAGE 10: TRADE JOURNAL
# ============================================================================

elif page == "Trade Journal":
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


# ============================================================================
# PAGE 11: SETTINGS
# ============================================================================

elif page == "Settings":
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