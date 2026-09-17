"""Extracted from app.py"""
from datetime import datetime
from core.data_loader import get_stock_data_cached

import pandas as pd


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

