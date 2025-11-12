import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import gdown
import tempfile
import os
import joblib
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from datetime import timedelta

import keras
import tensorflow as tf
from keras import backend as K


@keras.saving.register_keras_serializable(package="Custom")
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


# ----------------------------
# Helper cached function (outside class)
# ----------------------------
@st.cache_data(show_spinner=False)
def cached_yf_download(ticker: str, period: str = "max") -> pd.DataFrame:
    """Download data from yfinance with cache (function outside class to avoid hashing problems)."""
    raw = yf.download(ticker, period=period, progress=False)
    if raw is None:
        return pd.DataFrame()
    return raw


# ----------------------------
# Initial configuration
# ----------------------------
st.set_page_config(
    page_title="Stock Forecast - AI",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🤖 AI Stock Forecast (STABLE VERSION)</h1>', unsafe_allow_html=True)
st.markdown("### 10-day forecast using trained LSTM — for analysis only, not financial advice.")


# ----------------------------
# StockForecaster - manager
# ----------------------------
class StockForecaster:
    def __init__(
        self,
        model_drive_url="https://drive.google.com/uc?id=15rz-a_5ktKYzbD7RMSdc1yYld5hVBZC6",
        scaler_drive_url="https://drive.google.com/uc?id=1jCntAb1ArwVloTgkzIBOxL9-0uEjpU2o",
        info_drive_url="https://drive.google.com/uc?id=1s42zuLjHZaw3Xy7_cnGeqBgyTMFkg9vt",
        seq_len=60,
        tmp_dir=None
    ):
        self.model_url = model_drive_url
        self.scaler_url = scaler_drive_url
        self.info_url = info_drive_url
        self.SEQ_LEN = seq_len

        self.model = None
        self.scaler = None
        self.model_info = None

        temp_dir = tmp_dir or tempfile.gettempdir()
        self.model_file = os.path.join(temp_dir, "forecasting.keras")
        self.scaler_file = os.path.join(temp_dir, "scaler.pkl")
        self.info_file = os.path.join(temp_dir, "model_info.pkl")

        self._download_and_load_assets()

    def _download_and_load_assets(self):
        st.info("📥 Downloading model files (if needed)...")

        try:
            if not os.path.exists(self.model_file):
                gdown.download(self.model_url, self.model_file, quiet=True)
            if not os.path.exists(self.scaler_file):
                gdown.download(self.scaler_url, self.scaler_file, quiet=True)
            if not os.path.exists(self.info_file):
                gdown.download(self.info_url, self.info_file, quiet=True)
        except Exception as e:
            st.warning(f"Unable to download some files automatically: {e}. Trying local load if available.")

        try:
            if not os.path.exists(self.model_file):
                raise FileNotFoundError(f"Model file not found: {self.model_file}")
            self.model = load_model(self.model_file, compile=False)
            st.success("✅ LSTM Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {e}")
            raise

        try:
            if not os.path.exists(self.scaler_file):
                raise FileNotFoundError(f"Scaler file not found: {self.scaler_file}")
            self.scaler = joblib.load(self.scaler_file)
        except Exception as e:
            st.error(f"Error loading scaler: {e}")
            raise

        try:
            if not os.path.exists(self.info_file):
                st.warning("Model info file not found — continuing without it.")
                self.model_info = {}
            else:
                self.model_info = joblib.load(self.info_file)
        except Exception as e:
            st.warning(f"Error loading model info: {e}")
            self.model_info = {}

        st.success("✅ Scaler and Model Info loaded successfully (when available).")

    def download_stock_data(self, ticker):
        """Download data via yfinance and prepare Price/Volume/Sentiment columns."""
        try:
            raw = cached_yf_download(ticker)
        except Exception as e:
            raise RuntimeError(f"Error downloading ticker {ticker}: {e}")

        if raw is None or raw.empty:
            raise RuntimeError(f"No data found for ticker {ticker}.")

        df = raw.copy()

        if 'Adj Close' in df.columns:
            df['Price'] = df['Adj Close']
        else:
            df['Price'] = df['Close']

        df['Volume'] = df['Volume'].fillna(0) if 'Volume' in df.columns else 0
        df['Sentiment'] = 0.0

        df = df[['Price', 'Volume', 'Sentiment']]
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        return df

    def _prepare_sequences(self, df):
        """Generate X and y from DataFrame with ['Price','Volume','Sentiment']."""
        values = df[["Price", "Volume", "Sentiment"]].values.astype(float)
        scaled = self.scaler.transform(values)

        X, y = [], []
        for i in range(self.SEQ_LEN, len(scaled)):
            X.append(scaled[i-self.SEQ_LEN:i])
            y.append(scaled[i, 0])
        if len(X) == 0:
            raise ValueError(f"Insufficient data. Required seq_len={self.SEQ_LEN}, found {len(df)} rows.")
        return np.array(X), np.array(y)

    def predict_future(self, ticker, days=10):
        """Returns (forecast_df, history_df, metrics)."""
        df = self.download_stock_data(ticker)
        X, y = self._prepare_sequences(df)

        split = max(int(len(X) * 0.8), 1)
        X_test = X[split:]
        y_test = y[split:]

        pred_scaled_test = self.model.predict(X_test, verbose=0).reshape(-1, 1)
        other_test = X_test[:, -1, 1:]
        inv_pred_test = self.scaler.inverse_transform(np.hstack([pred_scaled_test, other_test]))[:, 0]
        inv_real_test = self.scaler.inverse_transform(np.hstack([y_test.reshape(-1,1), other_test]))[:, 0]

        mae = mean_absolute_error(inv_real_test, inv_pred_test)
        rmse_val = np.sqrt(mean_squared_error(inv_real_test, inv_pred_test))
        denom = np.where(inv_real_test == 0, 1e-8, inv_real_test)
        mape = np.mean(np.abs((inv_real_test - inv_pred_test) / denom)) * 100

        if len(inv_real_test) >= 2:
            real_diff = np.diff(inv_real_test)
            pred_diff = np.diff(inv_pred_test)
            direction_acc = (np.sum((real_diff * pred_diff) > 0) / len(real_diff)) * 100
        else:
            direction_acc = None

        last_seq = X_test[-1].copy() if len(X_test) else X[-1].copy()
        seq = last_seq.copy()
        future_scaled = []

        for _ in range(days):
            p_scaled = self.model.predict(seq[np.newaxis, :, :], verbose=0)[0, 0]
            future_scaled.append(p_scaled)
            new_row = np.hstack([[p_scaled], seq[-1, 1:]])
            seq = np.vstack([seq[1:], new_row])

        future_other = np.tile(last_seq[-1, 1:], (days, 1))
        inv_future = self.scaler.inverse_transform(np.hstack([np.array(future_scaled).reshape(-1,1), future_other]))[:, 0]

        last_date = df.index[-1]
        future_dates = pd.date_range(last_date + timedelta(days=1), periods=days)

        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted_Price": np.round(inv_future, 2)
        })

        forecast_df['Pct_Change'] = forecast_df['Predicted_Price'].pct_change().fillna(
            (forecast_df['Predicted_Price'].iloc[0] - df['Price'].iloc[-1]) / df['Price'].iloc[-1]
        ) * 100

        metrics = {
            "MAE": float(np.round(mae, 4)),
            "RMSE": float(np.round(rmse_val, 4)),
            "MAPE": float(np.round(mape, 4)),
            "Direction_Accuracy": None if direction_acc is None else float(np.round(direction_acc, 2))
        }

        return forecast_df, df, metrics


# ----------------------------
# UI - Sidebar
# ----------------------------
st.sidebar.header("🎯 Settings")

ticker = st.sidebar.text_input(
    "Enter stock symbol:",
    value="AAPL",
    help="Ex: AAPL, TSLA, PETR4.SA"
).upper().strip()

days = st.sidebar.number_input("Days to forecast", min_value=1, max_value=60, value=10, step=1)
seq_len = st.sidebar.number_input("SEQ_LEN (LSTM sequence length)", min_value=10, max_value=240, value=60, step=1)

predict_button = st.sidebar.button("🎯 Run Forecast", type="primary")

st.sidebar.markdown("---")
st.sidebar.subheader("💡 Suggested Stocks")
st.sidebar.markdown("""
**USA:**
- AAPL, TSLA, GOOGL
- MSFT, AMZN, META

**Brazil:**
- PETR4.SA, VALE3.SA
- ITUB4.SA, BBDC4.SA
""")


# ----------------------------
# Main process
# ----------------------------
if predict_button and ticker:
    main_spinner = st.empty()
    main_spinner.info("🤖 Preparing and generating predictions... (see logs below)")

    progress_bar = st.progress(0)
    for i in range(20):
        time.sleep(0.02)
        progress_bar.progress(int((i+1) * (100/20)))

    try:
        forecaster = StockForecaster(seq_len=int(seq_len))
    except Exception as e:
        st.error(f"Failed to initialize forecaster: {e}")
        st.stop()

    try:
        forecast_df, history_df, metrics = forecaster.predict_future(ticker, days=int(days))
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    main_spinner.empty()
    st.success(f"✅ Forecast completed for {ticker}!")

    st.subheader("📊 Forecast Summary")
    col1, col2, col3 = st.columns(3)

    with col1:
        current_price = history_df['Price'].iloc[-1]
        st.metric("Current Price", f"${current_price:.2f}")

    with col2:
        avg_price = forecast_df["Predicted_Price"].mean()
        st.metric("Avg Predicted Price", f"${avg_price:.2f}")

    with col3:
        total_change = ((forecast_df["Predicted_Price"].iloc[-1] - current_price) / current_price) * 100
        st.metric(f"Total Change ({days} days)", f"{total_change:.2f}%")

    st.subheader("📌 Model Metrics (X_test only)")
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("MAE", f"{metrics.get('MAE', 'N/A')}")
    mcol2.metric("RMSE", f"{metrics.get('RMSE', 'N/A')}")
    mcol3.metric("MAPE", f"{metrics.get('MAPE', 'N/A')}%")
    mcol4.metric("Direction Accuracy", f"{metrics.get('Direction_Accuracy', 'N/A')}%")

    st.subheader("📅 Detailed Forecast")
    display_df = forecast_df.copy()
    display_df["Date"] = display_df["Date"].dt.strftime("%d/%m/%Y")
    display_df = display_df.rename(columns={
        "Date": "Date",
        "Predicted_Price": "Predicted Price (USD)",
        "Pct_Change": "Change (%)"
    })
    st.dataframe(display_df.style.format({
        "Predicted Price (USD)": "{:.2f}",
        "Change (%)": "{:.2f}%"
    }), use_container_width=True)

    st.subheader("📈 Chart View")
    fig = go.Figure()

    hist_plot = history_df['Price'].iloc[-180:]
    fig.add_trace(go.Scatter(
        x=hist_plot.index,
        y=hist_plot.values,
        mode='lines',
        name='Historical Price',
        line=dict(width=2)
    ))

    fig.add_trace(go.Scatter(
        x=pd.to_datetime(forecast_df["Date"]),
        y=forecast_df["Predicted_Price"],
        mode='lines+markers',
        name='Forecasted Price',
        line=dict(dash='dash', width=3),
        marker=dict(size=8)
    ))

    fig.add_hline(
        y=current_price,
        line_dash="dash",
        annotation_text=f"Current Price: ${current_price:.2f}",
        annotation_position="top left"
    )

    fig.update_layout(
        title=f"Price Forecast - {ticker}",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("💡 Insights")
    if total_change > 0:
        st.success(f"**UPWARD TREND** — Forecasted growth of {total_change:.2f}%")
    else:
        st.error(f"**DOWNWARD TREND** — Forecasted decline of {abs(total_change):.2f}%")


st.markdown("---")
st.markdown("🔮 *Predictions generated by AI — For research use only. Not financial advice.*")


