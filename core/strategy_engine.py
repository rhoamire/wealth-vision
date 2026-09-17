"""Extracted from app.py"""
from core.utils import compute_indicators

import pandas as pd
import numpy as np
import ta


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

