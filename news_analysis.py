import requests
import pandas as pd
from transformers import pipeline
import yfinance as yf

class SentimentEngine:
    def __init__(self, model_name="ProsusAI/finbert"):
        self.nlp = pipeline(
            "sentiment-analysis", 
            model=model_name, 
            framework="pt"
        )
        self.label_map = {'positive': 1.0, 'negative': -1.0, 'neutral': 0.0}

    def fetch_news(self, ticker, api_key):
        url = f"https://newsapi.org/v2/everything?q={ticker}&language=en&sortBy=publishedAt&apiKey={api_key}"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                return []
            return response.json().get('articles', [])[:20]
        except Exception as e:
            print(f"Error fetching news: {e}")
            return []

    def process_sentiment(self, articles):
        if not articles:
            return pd.DataFrame(), 0
        
        headlines = [a.get('title', '') for a in articles]
        results = self.nlp(headlines) 
        
        processed_data = []
        for i, res in enumerate(results):
            label = res['label'].lower()
            score = self.label_map.get(label, 0.0)
            processed_data.append({
                "Date": pd.to_datetime(articles[i]['publishedAt']).date(),
                "Sentiment": label.upper(),
                "Score": score,
                "Confidence": res['score']
            })
            
        df = pd.DataFrame(processed_data)
        avg_score = df['Score'].mean()
        return df, avg_score

    def get_correlation_analysis(self, sentiment_df, stock_data):
        """
        Calculates the Pearson correlation between daily average sentiment 
        and stock returns. This is a key 'Master's Level' metric.
        """
        if sentiment_df.empty or stock_data.empty:
            return 0.0
        
        # Grouping sentiment by date
        daily_sent = sentiment_df.groupby('Date')['Score'].mean().reset_index()
        daily_sent['Date'] = pd.to_datetime(daily_sent['Date'])
        
        stock_resampled = stock_data[['Close']].copy()
        stock_resampled['Returns'] = stock_resampled['Close'].pct_change()
        stock_resampled = stock_resampled.reset_index()
        stock_resampled['Date'] = pd.to_datetime(stock_resampled['Date']).dt.tz_localize(None)
        
        merged = pd.merge(daily_sent, stock_resampled, on='Date', how='inner')
        
        if len(merged) < 2:
            return 0.0
            
        correlation = merged['Score'].corr(merged['Returns'])
        return correlation