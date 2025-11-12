# news_sentiment_pipeline.py
# ================================================
# Finnhub News Sentiment Pipeline
# ================================================
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()  # para garantir leitura do .env

from .news_sentiment_finnhub import process_finnhub_news_enhanced
from .synthetic_sentiment import create_intelligent_synthetic_sentiment


def apply_sentiment_pipeline(
    df_full: pd.DataFrame,
    ticker: str,
    use_sentiment: bool = True,
    min_news_required: int = 20
) -> pd.DataFrame:

    if not use_sentiment:
        df_full["News_Sentiment"] = 0
        return df_full

    print("\n" + "=" * 60)
    print(f"🚀 STARTING SENTIMENT ANALYSIS FOR {ticker}")
    print("=" * 60)

    start_date = df_full.index.min().strftime("%Y-%m-%d")

    FINNHUB_API_KEY = os.getenv("FINNHUB_KEY", "")  # 🔥 Segurança!

    df_news_sentiment = process_finnhub_news_enhanced(
        ticker=ticker,
        start_date=start_date,
        api_key=FINNHUB_API_KEY
    )

    close_col = f"{ticker}_Close"

    if df_news_sentiment is not None and len(df_news_sentiment) > min_news_required:
        print("✅ Using real news data from Finnhub")

        df_full = df_full.join(df_news_sentiment[['sentiment_score']], how='left')

        missing_mask = df_full['sentiment_score'].isna()

        if missing_mask.any():
            print(f"🔧 Filling {missing_mask.sum()} missing days with synthetic sentiment...")
            synthetic_sentiment = create_intelligent_synthetic_sentiment(df_full, close_col)
            df_full.loc[missing_mask, 'sentiment_score'] = synthetic_sentiment[missing_mask]

        df_full['sentiment_score'] = df_full['sentiment_score'].fillna(0)

    else:
        print("⚠️ Insufficient Finnhub data. Using intelligent synthetic sentiment.")
        df_full['sentiment_score'] = create_intelligent_synthetic_sentiment(df_full, close_col)

    df_full = df_full.rename(columns={'sentiment_score': 'News_Sentiment'})

    # Summary statistics
    print("\n📊 SENTIMENT STATISTICS:")
    print(f"➡ Mean:   {df_full['News_Sentiment'].mean():.4f}")
    print(f"➡ Std:    {df_full['News_Sentiment'].std():.4f}")
    print(f"➡ Min:    {df_full['News_Sentiment'].min():.4f}")
    print(f"➡ Max:    {df_full['News_Sentiment'].max():.4f}")

    correlation = df_full[close_col].corr(df_full['News_Sentiment'])
    print(f"\n🔗 SENTIMENT vs PRICE CORRELATION: {correlation:.4f}")

    return df_full.copy()
