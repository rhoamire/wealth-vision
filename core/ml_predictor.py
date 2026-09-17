"""Extracted from app.py"""
import ta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from core.data_loader import get_stock_data_cached

import pandas as pd
import numpy as np


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

