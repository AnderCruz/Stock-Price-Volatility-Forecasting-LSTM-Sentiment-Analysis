# Stock Price & Volatility Forecasting with LSTM and Sentiment Analysis

A comprehensive machine learning system for stock price prediction and volatility forecasting using LSTM neural networks with integrated sentiment analysis from financial news.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Model Details](#model-details)
- [Results](#results)
- [Configuration](#configuration)
- [Risk Analysis](#risk-analysis)
- [Disclaimer](#disclaimer)

## 🎯 Overview

This project implements an advanced stock forecasting system that combines:

- **LSTM Neural Networks**: For sequential price pattern recognition
- **Sentiment Analysis**: Real-time news sentiment from Finnhub API
- **Multivariate Analysis**: Correlation with other assets (AAPL, SPY)
- **Volatility Forecasting**: Separate LSTM model for risk assessment
- **Technical Indicators**: Price, volume, and derived features

The system provides both short-term price predictions and volatility forecasts with integrated alert systems.

## ✨ Features

- **Dual LSTM Models**: Separate networks for price and volatility prediction
- **Real-time Sentiment Analysis**: Integration with Finnhub API for news sentiment
- **Multivariate Capability**: Optional inclusion of correlated assets
- **Intelligent Synthetic Sentiment**: Fallback system when news data is limited
- **Comprehensive Metrics**: MAE, RMSE, MAPE, Directional Accuracy
- **Risk Assessment**: Volatility forecasting with confidence intervals
- **Alert System**: Price and volatility threshold monitoring
- **Visual Analytics**: Comparative plots and correlation matrices

## 🏗️ Architecture

### Data Pipeline
```
Yahoo Finance Data → Feature Engineering → Sentiment Analysis → 
LSTM Price Model → LSTM Volatility Model → Ensemble Forecasting → Alert System
```

### Model Stack
- **Price Prediction**: LSTM(72) → Dropout → LSTM(48) → Dropout → Dense(1)
- **Volatility Prediction**: LSTM(50) → Dropout → LSTM(30) → Dropout → Dense(1)
- **Sentiment Integration**: TextBlob + VADER hybrid analysis

## 🚀 Installation

### Prerequisites

```bash
# Core data science libraries
pip install numpy pandas matplotlib seaborn scikit-learn

# Financial data
pip install yfinance finnhub-python

# Deep learning
pip install tensorflow

# Natural Language Processing
pip install nltk textblob

# HTTP requests
pip install requests
```

### Required API Keys

```python
FINNHUB_KEY = "your_finnhub_api_key_here"  # Get from https://finnhub.io
```

## 📖 Usage

### Basic Configuration

```python
# Core parameters
ticker = "TSLA"
data_start = "2023-01-01"
USE_SENTIMENT = True      # Enable/disable sentiment analysis
USE_MULTIVARIATE = False  # Include other assets

# Model parameters
SEQ_LEN = 60              # Historical sequence length
FUTURE_DAYS = 90          # Forecast horizon
TEST_RATIO = 0.2          # Test set size
```

### Running the Pipeline

1. **Data Collection**:
   ```python
   # Automatically downloads historical data from Yahoo Finance
   df_main = yf.download(ticker, start=data_start, progress=False)
   ```

2. **Sentiment Analysis**:
   - Real-time news from Finnhub API
   - Hybrid TextBlob + VADER sentiment scoring
   - Intelligent synthetic sentiment fallback

3. **Feature Engineering**:
   - Price and volume data
   - Log returns and rolling statistics
   - Sentiment scores integration

4. **Model Training**:
   - Automatic LSTM architecture selection
   - Early stopping and validation
   - Separate models for price and volatility

5. **Forecast Generation**:
   - Next-day price prediction
   - 30-day volatility forecast
   - Trend analysis and alerts

## 🔬 Methodology

### Feature Engineering

- **Price Features**: Close prices, log returns, percentage changes
- **Volume Features**: Trading volumes with normalization
- **Technical Indicators**: Rolling means, standard deviations
- **Sentiment Features**: News sentiment scores (-1 to +1 scale)

### Sentiment Analysis

```python
def analyze_sentiment_enhanced(text):
    """Enhanced sentiment analysis with financial terminology"""
    # Combines TextBlob, VADER, and financial keyword analysis
    # Returns normalized score between -1.0 and +1.0
```

### LSTM Architecture

**Price Model**:
```
Input: (60, n_features) → LSTM(72) → Dropout(0.2) → 
LSTM(48) → Dropout(0.2) → Dense(1)
```

**Volatility Model**:
```
Input: (30, 1) → LSTM(50) → Dropout(0.2) → 
LSTM(30) → Dropout(0.2) → Dense(1)
```

## 🤖 Model Details

### Price Prediction LSTM
- **Input Sequence**: 60 days of historical data
- **Features**: Price, volume, sentiment, correlated assets
- **Output**: Next-day price prediction
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam (lr=0.001)

### Volatility Prediction LSTM
- **Input Sequence**: 30 days of volatility data
- **Calculation**: Rolling standard deviation of log returns
- **Output**: Future volatility estimates
- **Application**: Risk assessment and position sizing

### Sentiment Integration
- **Data Source**: Finnhub company news API
- **Methods**: TextBlob (lexicon-based) + VADER (rule-based)
- **Fallback**: Synthetic sentiment from price movements
- **Normalization**: Tanh activation for -1 to +1 range

## 📊 Results

### Performance Metrics
- **Price Prediction**: MAE, RMSE, MAPE, Directional Accuracy
- **Volatility Prediction**: MSE on normalized volatility
- **Sentiment Analysis**: Correlation with price movements

### Output Examples
```
📌 Previsão preço TSLA próximo pregão: $245.32
Preço Atual: $240.50
Variação: $4.82 (+2.00%)
Tendência: 📈 FORTE ALTA

🎯 PREVISÃO DE VOLATILIDADE (Próximos 5 dias):
Volatilidade Atual: 0.023456 (2.3456%)
Volatilidade Prevista: 0.028901 (2.8901%)
Mudança: +0.005445 (+23.22%)
Tendência: 📈 AUMENTO SIGNIFICATIVO DE VOLATILIDADE
```

## ⚙️ Configuration

### Key Parameters

```python
# Data Parameters
ticker = "TSLA"
extra_tickers = ["AAPL", "SPY"]  # Correlated assets
data_start = "2023-01-01"

# Model Parameters
SEQ_LEN = 60           # Price model sequence length
SEQ_LEN_VOL = 30       # Volatility model sequence length
FUTURE_DAYS = 90       # Forecast horizon
TEST_RATIO = 0.2       # Test set percentage

# Feature Flags
USE_SENTIMENT = True    # Enable sentiment analysis
USE_MULTIVARIATE = False # Use multiple assets
```

### Alert Thresholds

```python
# Price Alerts
HIGH_PRICE_THRESHOLD = df_full[price_col].quantile(0.75)
LOW_PRICE_THRESHOLD = df_full[price_col].quantile(0.25)

# Volatility Alerts
HIGH_VOL_THRESHOLD = df_vol['Volatility'].quantile(0.75)
LOW_VOL_THRESHOLD = df_vol['Volatility'].quantile(0.25)
```

## 📈 Risk Analysis

### Volatility Forecasting
- **Input**: 30-day rolling volatility
- **Output**: 30-day future volatility predictions
- **Application**: Risk management and position sizing

### Alert System
- **Price Alerts**: High/Low price levels based on historical percentiles
- **Volatility Alerts**: Significant changes in market volatility
- **Trend Alerts**: Strong upward/downward movements

### Monte Carlo Simulation
(Optional extension) For probabilistic forecasting and confidence intervals.

## ⚠️ Disclaimer

**Important**: This tool is for educational and research purposes only. The forecasts are statistical estimates based on historical data and should not be considered financial advice.

- Always conduct your own research and due diligence
- Consider macroeconomic factors and company fundamentals
- Implement proper risk management strategies
- Past performance does not guarantee future results
- The developers are not responsible for investment decisions made using this tool

## 🔧 Customization

### Adding New Features

```python
# Example: Add technical indicators
def add_technical_indicators(df):
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    df['MACD'] = calculate_macd(df['Close'])
    return df
```

### Modifying Sentiment Analysis

```python
# Add custom financial terminology
custom_positive_terms = {
    'breakout': 0.8, 'momentum': 0.6, 'oversold': 0.3
}
custom_negative_terms = {
    'resistance': -0.5, 'distribution': -0.6
}
```

### Model Architecture Changes

```python
# Deeper LSTM architecture
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(SEQ_LEN, n_features)),
    Dropout(0.3),
    LSTM(64, return_sequences=True),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])
```

## 📞 Support

For issues and questions:

1. **API Keys**: Ensure Finnhub API key is valid
2. **Data Availability**: Check Yahoo Finance data for your ticker
3. **Dependencies**: Verify all packages are correctly installed
4. **Memory Issues**: Reduce sequence length or historical period for large datasets

## 🎯 Future Enhancements

- [ ] Transformer architecture for better sequence modeling
- [ ] Integration with fundamental analysis data
- [ ] Portfolio optimization capabilities
- [ ] Real-time trading interface
- [ ] Advanced risk metrics (VaR, CVaR)
- [ ] Multi-timeframe analysis

---

**Note**: This system is designed for quantitative analysis and requires understanding of financial markets, machine learning, and Python programming. Always test with paper trading before live deployment.

## 🏷️ Tags

```
stock-prediction, lstm, sentiment-analysis, time-series-forecasting, 
financial-machine-learning, python, tensorflow, finnhub-api, 
volatility-forecasting, algorithmic-trading, quantitative-finance,
risk-management, neural-networks, nltk, yfinance
```
