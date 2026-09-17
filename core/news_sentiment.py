"""Extracted from app.py"""
import numpy as np
from config import NEWS_API_URL

import requests
from datetime import datetime, timedelta
from textblob import TextBlob


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

