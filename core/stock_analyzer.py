"""Extracted from app.py"""
import streamlit as st
import yfinance as yf
from core.utils import compute_indicators
from core.data_loader import get_stock_data_cached

import pandas as pd
import numpy as np
import ta


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

