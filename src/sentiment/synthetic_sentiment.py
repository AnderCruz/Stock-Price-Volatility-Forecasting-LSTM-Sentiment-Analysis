# synthetic_sentiment.py
# ================================================
# Intelligent Synthetic Sentiment Generator
# ================================================

import pandas as pd
import numpy as np


def create_intelligent_synthetic_sentiment(df: pd.DataFrame, close_col_name: str) -> pd.Series:
    """
    Generate synthetic sentiment values from price returns.

    Parameters
    ----------
    df : pd.DataFrame
        Time series price data containing a close column.
    close_col_name : str
        The name of the closing price column in df.

    Returns
    -------
    pd.Series
        Smoothed synthetic sentiment score (-1 to 1).
    """
    
    if close_col_name not in df.columns:
        raise ValueError(f"Column '{close_col_name}' not found in DataFrame")

    # Use daily returns to generate sentiment proxy
    returns = df[close_col_name].pct_change().fillna(0)

    # Normalize returns to -1 to 1 scale
    min_ret = returns.min()
    max_ret = returns.max()

    if max_ret - min_ret == 0:
        normalized_sent = np.zeros_like(returns)
    else:
        normalized_sent = 2 * (returns - min_ret) / (max_ret - min_ret) - 1

    # Apply smoothing to reduce noise
    smooth_sent = pd.Series(normalized_sent).rolling(window=5, min_periods=1).mean()

    # Ensure no NaN or infinities
    smooth_sent = smooth_sent.replace([np.inf, -np.inf], 0).fillna(0)

    return smooth_sent
