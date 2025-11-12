# src/news/news_sentiment_finnhub.py

import os
import requests
import pandas as pd
import nltk
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer

# 🔹 Ensure VADER is installed
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

sia = SentimentIntensityAnalyzer()

def process_finnhub_news_enhanced(ticker, api_key, start_date="2020-01-01", end_date="2026-12-31"):
    """
    Fetches Finnhub company news + sentiment analysis (TextBlob + VADER)
    Returns: Pandas DataFrame indexed by date with sentiment score
    """
    print(f"🔍 Searching Finnhub news for {ticker}...")

    url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={start_date}&to={end_date}&token={api_key}"

    try:
        response = requests.get(url)
        news_data = response.json()

        if not isinstance(news_data, list) or len(news_data) == 0:
            print("⚠️ No news found for this period.")
            return None
        
        df_news = pd.DataFrame(news_data)

        # Only headlines with valid datetime
        df_news = df_news.dropna(subset=["headline", "datetime"])
        df_news["datetime"] = pd.to_datetime(df_news["datetime"], unit="s")

        # Combined sentiment score
        def compute_sentiment(text):
            text = str(text)
            blob_sent = TextBlob(text).sentiment.polarity
            vader_sent = sia.polarity_scores(text)["compound"]
            return (blob_sent + vader_sent) / 2

        df_news["sentiment_score"] = df_news["headline"].apply(compute_sentiment)

        # Group by date
        df_daily = df_news.groupby(df_news["datetime"].dt.date)["sentiment_score"].mean()
        df_daily.index = pd.to_datetime(df_daily.index)
        df_daily = df_daily.sort_index()

        print(f"✅ {len(df_news)} headlines processed → {len(df_daily)} days consolidated.")
        return df_daily.to_frame(name="sentiment_score")

    except Exception as e:
        print(f"❌ Error while fetching news: {e}")
        return None
