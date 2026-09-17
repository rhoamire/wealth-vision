"""Extracted from app.py"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
import ta


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

