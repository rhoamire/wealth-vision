import yfinance as yf
import pandas as pd
import numpy as np
import ta
import warnings
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import os
from datetime import datetime, timedelta
import pickle

warnings.filterwarnings("ignore")

# ============================================================================
# ENHANCED ML MODEL WITH CONFIDENCE METRICS
# ============================================================================

class EnhancedMLPredictor:
    """Enhanced ML predictor with ensemble methods and confidence scoring"""
    
    def __init__(self):
        self.models = {
            'rf': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
            'gb': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        }
        self.scaler = StandardScaler()
        self.feature_importance = {}
        
    def create_advanced_features(self, df):
        """Create advanced technical features for ML"""
        features = pd.DataFrame(index=df.index)
        
        # Price-based features
        features['returns_1d'] = df['Close'].pct_change(1)
        features['returns_5d'] = df['Close'].pct_change(5)
        features['returns_20d'] = df['Close'].pct_change(20)
        
        # Volatility features
        features['volatility_10d'] = df['Close'].pct_change().rolling(10).std()
        features['volatility_30d'] = df['Close'].pct_change().rolling(30).std()
        
        # Technical indicators
        features['rsi'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        features['rsi_slope'] = features['rsi'].diff(5)
        
        macd = ta.trend.MACD(df['Close'])
        features['macd'] = macd.macd()
        features['macd_signal'] = macd.macd_signal()
        features['macd_diff'] = macd.macd_diff()
        
        # ADX and DMI
        adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'])
        features['adx'] = adx.adx()
        features['dmi_plus'] = adx.adx_pos()
        features['dmi_minus'] = adx.adx_neg()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['Close'])
        features['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
        features['bb_position'] = (df['Close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
        
        # Volume features
        features['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        features['volume_trend'] = df['Volume'].rolling(5).mean() / df['Volume'].rolling(20).mean()
        
        # OBV
        features['obv'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
        features['obv_slope'] = features['obv'].diff(5)
        
        # Moving averages crossovers
        features['sma_20'] = df['Close'].rolling(20).mean()
        features['sma_50'] = df['Close'].rolling(50).mean()
        features['ma_cross'] = (features['sma_20'] > features['sma_50']).astype(int)
        
        # Price position relative to moving averages
        features['price_to_sma20'] = df['Close'] / features['sma_20']
        features['price_to_sma50'] = df['Close'] / features['sma_50']
        
        return features.fillna(method='ffill').fillna(0)
    
    def create_labels(self, df, forward_days=10, threshold=0.03):
        """Create labels with multiple horizons"""
        df['future_return'] = df['Close'].shift(-forward_days) / df['Close'] - 1
        
        # Multi-class labels: Strong Buy (2), Buy (1), Neutral/Sell (0)
        labels = pd.Series(0, index=df.index)
        labels[df['future_return'] > threshold] = 1
        labels[df['future_return'] > threshold * 2] = 2
        
        return labels[:-forward_days]
    
    def train_ensemble(self, symbol, cv_splits=5):
        """Train ensemble models with cross-validation"""
        try:
            df = self.get_stock_data(symbol)
            if df is None or len(df) < 200:
                return None
            
            # Create features and labels
            features = self.create_advanced_features(df)
            labels = self.create_labels(df)
            
            # Align features and labels
            min_len = min(len(features), len(labels))
            features = features.iloc[:min_len]
            labels = labels.iloc[:min_len]
            
            # Remove rows with NaN
            mask = ~(features.isnull().any(axis=1) | labels.isnull())
            features = features[mask]
            labels = labels[mask]
            
            if len(features) < 100:
                return None
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=cv_splits)
            
            scores = {model_name: [] for model_name in self.models.keys()}
            predictions = {model_name: [] for model_name in self.models.keys()}
            
            for train_idx, test_idx in tscv.split(features_scaled):
                X_train, X_test = features_scaled[train_idx], features_scaled[test_idx]
                y_train, y_test = labels.iloc[train_idx], labels.iloc[test_idx]
                
                for model_name, model in self.models.items():
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    acc = accuracy_score(y_test, y_pred)
                    scores[model_name].append(acc)
                    predictions[model_name].extend(y_pred)
            
            # Calculate ensemble prediction confidence
            avg_scores = {name: np.mean(scores[name]) for name in scores.keys()}
            ensemble_score = np.mean(list(avg_scores.values()))
            
            # Get feature importance from Random Forest
            rf_importance = dict(zip(features.columns, self.models['rf'].feature_importances_))
            top_features = sorted(rf_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Make final prediction
            final_features = features_scaled[-1].reshape(1, -1)
            predictions_final = {}
            probabilities = {}
            
            for model_name, model in self.models.items():
                pred = model.predict(final_features)[0]
                predictions_final[model_name] = pred
                
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(final_features)[0]
                    probabilities[model_name] = proba
            
            # Calculate ensemble confidence
            ensemble_prediction = int(np.mean(list(predictions_final.values())))
            
            # Calculate confidence based on agreement and probability
            agreement = len(set(predictions_final.values())) == 1
            confidence = ensemble_score
            
            if probabilities:
                avg_proba = np.mean([proba[ensemble_prediction] for proba in probabilities.values()])
                confidence = (confidence + avg_proba) / 2
            
            return {
                'ensemble_prediction': ensemble_prediction,
                'confidence': confidence * 100,
                'model_scores': avg_scores,
                'ensemble_score': ensemble_score * 100,
                'agreement': agreement,
                'top_features': top_features,
                'predictions': predictions_final,
                'probabilities': probabilities
            }
            
        except Exception as e:
            print(f"Error training ensemble for {symbol}: {e}")
            return None
    
    def get_stock_data(self, symbol, period="2y"):
        """Get stock data"""
        try:
            ticker = yf.Ticker(symbol + ".NS")
            df = ticker.history(period=period)
            return df if not df.empty and len(df) > 100 else None
        except:
            return None


# ============================================================================
# CONFIDENCE SCORE CALCULATOR
# ============================================================================

class ConfidenceScoreCalculator:
    """Calculate comprehensive confidence scores for stock predictions"""
    
    def __init__(self):
        self.weights = {
            'technical': 0.30,
            'ml_prediction': 0.25,
            'trend_strength': 0.20,
            'volume_confidence': 0.15,
            'consistency': 0.10
        }
    
    def calculate_technical_score(self, conditions_met, total_conditions):
        """Calculate technical analysis confidence"""
        return (conditions_met / total_conditions) * 100
    
    def calculate_trend_strength(self, df):
        """Calculate trend strength score"""
        try:
            # ADX for trend strength
            adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx().iloc[-1]
            
            # Directional movement
            dmi_plus = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx_pos().iloc[-1]
            dmi_minus = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx_neg().iloc[-1]
            
            # Score based on ADX strength and directional bias
            adx_score = min(adx / 40 * 100, 100)  # Normalize ADX
            directional_score = (dmi_plus / (dmi_plus + dmi_minus)) * 100 if (dmi_plus + dmi_minus) > 0 else 50
            
            return (adx_score + directional_score) / 2
        except:
            return 50
    
    def calculate_volume_confidence(self, df):
        """Calculate volume-based confidence"""
        try:
            recent_volume = df['Volume'].iloc[-5:].mean()
            avg_volume = df['Volume'].iloc[-30:].mean()
            
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
            
            # Higher volume = higher confidence
            if volume_ratio > 1.5:
                return 90
            elif volume_ratio > 1.2:
                return 75
            elif volume_ratio > 0.8:
                return 60
            else:
                return 40
        except:
            return 50
    
    def calculate_consistency_score(self, df):
        """Calculate consistency of signals across timeframes"""
        try:
            # Check MACD across timeframes
            daily_macd = ta.trend.MACD(df['Close']).macd().iloc[-1] > 0
            
            df_weekly = df.resample('W').last()
            weekly_macd = False
            if len(df_weekly) > 30:
                weekly_macd = ta.trend.MACD(df_weekly['Close']).macd().iloc[-1] > 0
            
            df_monthly = df.resample('M').last()
            monthly_macd = False
            if len(df_monthly) > 30:
                monthly_macd = ta.trend.MACD(df_monthly['Close']).macd().iloc[-1] > 0
            
            # Count aligned signals
            aligned = sum([daily_macd, weekly_macd, monthly_macd])
            return (aligned / 3) * 100
        except:
            return 50
    
    def calculate_overall_confidence(self, stock_data, ml_results, conditions_met, total_conditions):
        """Calculate overall confidence score"""
        scores = {}
        
        # Technical score
        scores['technical'] = self.calculate_technical_score(conditions_met, total_conditions)
        
        # ML prediction confidence
        if ml_results:
            scores['ml_prediction'] = ml_results.get('confidence', 50)
        else:
            scores['ml_prediction'] = 50
        
        # Trend strength
        scores['trend_strength'] = self.calculate_trend_strength(stock_data)
        
        # Volume confidence
        scores['volume_confidence'] = self.calculate_volume_confidence(stock_data)
        
        # Consistency
        scores['consistency'] = self.calculate_consistency_score(stock_data)
        
        # Calculate weighted overall score
        overall_confidence = sum(scores[key] * self.weights[key] for key in scores.keys())
        
        return {
            'overall_confidence': round(overall_confidence, 2),
            'component_scores': scores,
            'interpretation': self.interpret_confidence(overall_confidence)
        }
    
    def interpret_confidence(self, score):
        """Interpret confidence score"""
        if score >= 80:
            return "Very High - Strong buy signal with high probability"
        elif score >= 70:
            return "High - Good buy opportunity with solid indicators"
        elif score >= 60:
            return "Moderate - Cautiously optimistic, monitor closely"
        elif score >= 50:
            return "Low-Moderate - Weak signals, high uncertainty"
        else:
            return "Low - Insufficient confidence, avoid or wait"


# ============================================================================
# SAVE/LOAD MODEL CACHE
# ============================================================================

def save_model_cache(symbol, ml_results, confidence_results):
    """Save ML model results to cache"""
    cache_dir = "model_cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    cache_data = {
        'ml_results': ml_results,
        'confidence_results': confidence_results,
        'timestamp': datetime.now().isoformat()
    }
    
    cache_file = os.path.join(cache_dir, f"{symbol}_cache.pkl")
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)

def load_model_cache(symbol, max_age_hours=24):
    """Load ML model results from cache if recent"""
    cache_file = os.path.join("model_cache", f"{symbol}_cache.pkl")
    
    if not os.path.exists(cache_file):
        return None
    
    try:
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        # Check if cache is still valid
        cache_time = datetime.fromisoformat(cache_data['timestamp'])
        if datetime.now() - cache_time < timedelta(hours=max_age_hours):
            return cache_data
    except:
        pass
    
    return None


# ============================================================================
# ENHANCED SCREENING FUNCTION
# ============================================================================

def enhanced_check_conditions(symbol, ml_predictor, confidence_calculator):
    """Enhanced condition checking with ML and confidence scores"""
    try:
        # Check cache first
        cached_results = load_model_cache(symbol)
        
        # Get stock data
        df = ml_predictor.get_stock_data(symbol)
        if df is None or len(df) < 100:
            return None
        
        # Add technical indicators
        df = compute_indicators(df)
        
        # Check basic technical conditions (from original code)
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        conditions = {}
        conditions['daily_macd_up'] = latest['MACD'] > 0 and latest['MACD'] > prev['MACD']
        conditions['rsi_above_55'] = latest['RSI'] > 55
        conditions['dmi_positive'] = latest['+DMI'] > latest['-DMI']
        conditions['adx_rising'] = latest['ADX'] > prev['ADX']
        conditions['volume_above_avg'] = latest['Volume'] > latest['Vol_MA'] * 0.9
        
        # Weekly MACD
        df_weekly = df.resample('W').last()
        if len(df_weekly) >= 30:
            df_weekly = compute_indicators(df_weekly)
            w_latest = df_weekly.iloc[-1]
            w_prev = df_weekly.iloc[-2]
            conditions['weekly_macd_up'] = w_latest['MACD'] > 0 and w_latest['MACD'] > w_prev['MACD']
        
        # Monthly MACD
        df_monthly = df.resample('M').last()
        if len(df_monthly) >= 30:
            df_monthly = compute_indicators(df_monthly)
            m_latest = df_monthly.iloc[-1]
            m_prev = df_monthly.iloc[-2]
            conditions['monthly_macd_up'] = m_latest['MACD'] > 0 and m_latest['MACD'] > m_prev['MACD']
        
        conditions_met = sum(conditions.values())
        
        # Minimum filter
        if conditions_met < 4:
            return None
        
        # ML Prediction
        ml_results = None
        if cached_results:
            ml_results = cached_results['ml_results']
        else:
            ml_results = ml_predictor.train_ensemble(symbol)
        
        # Calculate confidence scores
        if cached_results:
            confidence_results = cached_results['confidence_results']
        else:
            confidence_results = confidence_calculator.calculate_overall_confidence(
                df, ml_results, conditions_met, len(conditions)
            )
        
        # Save to cache
        if not cached_results:
            save_model_cache(symbol, ml_results, confidence_results)
        
        # Get market cap
        try:
            ticker = yf.Ticker(symbol + ".NS")
            market_cap = ticker.info.get('marketCap', None)
        except:
            market_cap = None
        
        # Compile results
        result = {
            'Symbol': symbol,
            'Conditions_Met': conditions_met,
            'Total_Conditions': len(conditions),
            'Market_Cap': market_cap,
            **conditions
        }
        
        # Add ML results
        if ml_results:
            result['ML_Prediction'] = ml_results['ensemble_prediction']
            result['ML_Confidence'] = round(ml_results['confidence'], 2)
            result['ML_Score'] = round(ml_results['ensemble_score'], 2)
        
        # Add confidence scores
        result['Overall_Confidence'] = confidence_results['overall_confidence']
        result['Confidence_Interpretation'] = confidence_results['interpretation']
        result['Technical_Score'] = round(confidence_results['component_scores']['technical'], 2)
        result['Trend_Strength_Score'] = round(confidence_results['component_scores']['trend_strength'], 2)
        result['Volume_Confidence_Score'] = round(confidence_results['component_scores']['volume_confidence'], 2)
        
        return result
        
    except Exception as e:
        print(f"Error processing {symbol}: {e}")
        return None


def compute_indicators(df):
    """Compute technical indicators (from original code)"""
    if df is None or len(df) < 30:
        return df
    
    try:
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['Close'], window=20)
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_lower'] = bb.bollinger_lband()
        df['BB_mid'] = bb.bollinger_mavg()
        
        # RSI
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        
        # DMI & ADX
        adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14)
        df['+DMI'] = adx.adx_pos()
        df['-DMI'] = adx.adx_neg()
        df['ADX'] = adx.adx()
        
        # Volume MA
        df['Vol_MA'] = df['Volume'].rolling(20).mean()
        
        # OBV
        df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
        
    except Exception as e:
        print(f"Error computing indicators: {e}")
    
    return df


# Helper function to convert numpy types to Python types
def convert_to_json_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        return obj

# Example usage
if __name__ == "__main__":
    # Initialize
    ml_predictor = EnhancedMLPredictor()
    confidence_calculator = ConfidenceScoreCalculator()
    
    # Test with a symbol
    result = enhanced_check_conditions("RELIANCE", ml_predictor, confidence_calculator)
    if result:
        # Convert to JSON serializable format
        result_serializable = convert_to_json_serializable(result)
        print(json.dumps(result_serializable, indent=2))
    else:
        print("No results found for RELIANCE")