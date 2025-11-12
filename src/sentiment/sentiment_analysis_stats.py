# sentiment_analysis_stats.py
# ===============================================
# Sentiment Statistical Analysis & Visualization
# ===============================================

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def analyze_and_plot_sentiment(df_full: pd.DataFrame, ticker: str):
    """
    Performs statistical analysis and generates 3 visualization charts
    based on News_Sentiment and price data.

    Parameters
    ----------
    df_full : pd.DataFrame
        Main DataFrame with columns:
            - News_Sentiment
            - <ticker>_Close
        Indexed by Date
    ticker : str
        Stock ticker, to locate price column
    """

    TARGET_PRICE_COL = f"{ticker}_Close"

    print(f"✅ Price data: {df_full.shape}")
    print(f"🎯 Target column: {TARGET_PRICE_COL}")

    print(f"\n📊 DETAILED SENTIMENT STATISTICS:")
    print(f"   Mean: {df_full['News_Sentiment'].mean():.4f}")
    print(f"   Median: {df_full['News_Sentiment'].median():.4f}")
    print(f"   Standard Deviation: {df_full['News_Sentiment'].std():.4f}")
    print(f"   Minimum: {df_full['News_Sentiment'].min():.4f}")
    print(f"   Maximum: {df_full['News_Sentiment'].max():.4f}")

    sentiment_bins = {
        "STRONG POSITIVE (≥0.3)": (df_full['News_Sentiment'] >= 0.3).sum(),
        "POSITIVE (0.15-0.3)": ((df_full['News_Sentiment'] >= 0.15) & (df_full['News_Sentiment'] < 0.3)).sum(),
        "LIGHT POSITIVE (0.05-0.15)": ((df_full['News_Sentiment'] >= 0.05) & (df_full['News_Sentiment'] < 0.15)).sum(),
        "NEUTRAL (-0.05-0.05)": ((df_full['News_Sentiment'] >= -0.05) & (df_full['News_Sentiment'] < 0.05)).sum(),
        "LIGHT NEGATIVE (-0.15-0.05)": ((df_full['News_Sentiment'] >= -0.15) & (df_full['News_Sentiment'] < -0.05)).sum(),
        "NEGATIVE (-0.3-0.15)": ((df_full['News_Sentiment'] >= -0.3) & (df_full['News_Sentiment'] < -0.15)).sum(),
        "STRONG NEGATIVE (<-0.3)": (df_full['News_Sentiment'] < -0.3).sum()
    }

    print(f"\n🎭 SENTIMENT DISTRIBUTION:")
    total_days = len(df_full)
    for category, count in sentiment_bins.items():
        percentage = (count / total_days) * 100
        print(f"   {category:25}: {count:3d} dias ({percentage:5.1f}%)")

    correlation = df_full[TARGET_PRICE_COL].corr(df_full['News_Sentiment'])
    print(f"\n🔗 SENTIMENT vs PRICE CORRELATION: {correlation:.4f}")

    # ===============================================
    # 📈 DATA VISUALIZATION
    # ===============================================

    plt.figure(figsize=(15, 12))

    # Chart 1: Sentiment over time
    plt.subplot(3, 1, 1)

    colors = []
    for score in df_full['News_Sentiment']:
        if score >= 0.15:
            colors.append('green')
        elif score >= 0.05:
            colors.append('lightgreen')
        elif score <= -0.15:
            colors.append('red')
        elif score <= -0.05:
            colors.append('lightcoral')
        else:
            colors.append('gray')

    plt.scatter(df_full.index, df_full['News_Sentiment'], c=colors, alpha=0.7, s=30)
    plt.axhline(y=0, color='blue', linestyle='-', alpha=0.3)
    plt.ylabel('Sentiment Score')
    plt.title(f'News Sentiment - {ticker}')
    plt.grid(True, alpha=0.3)

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Positive (≥0.15)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=8, label='Light Positive (0.05-0.15)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Neutral (-0.05-0.05)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', markersize=8, label='Light Negative (-0.15-0.05)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Negative (≤-0.15)')
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    # Chart 2: Price vs Sentiment
    plt.subplot(3, 1, 2)
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    ax1.plot(df_full.index, df_full[TARGET_PRICE_COL], linewidth=2, label=f"{ticker}_Close", alpha=0.8)
    ax2.scatter(df_full.index, df_full['News_Sentiment'], s=20, alpha=0.6, label='Sentimento')

    ax1.set_ylabel('Price (USD)')
    ax2.set_ylabel('Sentiment')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title(f'Relationship: Price vs Sentiment - {ticker}')
    plt.grid(True, alpha=0.3)

    # Chart 3: Correlation Matrix
    plt.subplot(3, 1, 3)
    corr_matrix = df_full[[TARGET_PRICE_COL, 'News_Sentiment']].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, cbar_kws={"shrink": 0.8})
    plt.title(f'Correlation Matrix - {ticker}')

    plt.tight_layout()
    plt.show()

    print(f"\n✅ SENTIMENT ANALYSIS COMPLETED!")
    print(f"   Final shape: {df_full.shape}")
    print(f"   Period: {df_full.index[0].strftime('%Y-%m-%d')} to {df_full.index[-1].strftime('%Y-%m-%d')}")
    print(f"   Columns: {list(df_full.columns)}")

