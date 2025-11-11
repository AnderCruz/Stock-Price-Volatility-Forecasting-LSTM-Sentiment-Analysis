# app.py - VERSÃO ATUALIZADA (tudo em um arquivo)
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

# ----------------------------
# Configurações iniciais
# ----------------------------
st.set_page_config(
    page_title="Previsão de Ações - IA",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🤖 Previsão de Ações com IA</h1>', unsafe_allow_html=True)
st.markdown("### Previsão dos próximos 10 dias usando modelo treinado (LSTM)")

# ----------------------------
# StockForecaster - manager
# ----------------------------
class StockForecaster:
    def __init__(
        self,
        model_drive_url="https://drive.google.com/uc?id=15rz-a_5ktKYzbD7RMSdc1yYld5hVBZC6",
        scaler_drive_url="https://drive.google.com/uc?id=1jCntAb1ArwVloTgkzIBOxL9-0uEjpU2o",
        info_drive_url="https://drive.google.com/uc?id=1s42zuLjHZaw3Xy7_cnGeqBgyTMFkg9vt",
        seq_len=60
    ):
        self.model_url = model_drive_url
        self.scaler_url = scaler_drive_url
        self.info_url = info_drive_url
        self.SEQ_LEN = seq_len

        self.model = None
        self.scaler = None
        self.model_info = None

        temp_dir = tempfile.gettempdir()
        self.model_file = os.path.join(temp_dir, "forecasting.keras")
        self.scaler_file = os.path.join(temp_dir, "scaler.pkl")
        self.info_file = os.path.join(temp_dir, "model_info.pkl")

        self._download_and_load_assets()

    def _download_and_load_assets(self):
        st.info("📥 Baixando arquivos do modelo...")

        # ✅ Download
        gdown.download(self.model_url, self.model_file, quiet=True)
        gdown.download(self.scaler_url, self.scaler_file, quiet=True)
        gdown.download(self.info_url, self.info_file, quiet=True)

        # ✅ Load real model
        self.model = load_model(self.model_file)
        st.success("✅ Modelo LSTM carregado com sucesso!")

        self.scaler = joblib.load(self.scaler_file)
        self.model_info = joblib.load(self.info_file)

        st.success("✅ Scaler e Model Info carregados com sucesso!")

    def _prepare_sequences(self, df):
        values = df[["Price", "Volume", "Sentiment"]].values.astype(float)
        scaled = self.scaler.transform(values)

        X = []
        y = []
        for i in range(self.SEQ_LEN, len(scaled)):
            X.append(scaled[i-self.SEQ_LEN:i])
            y.append(scaled[i, 0])
        return np.array(X), np.array(y)

    def predict_future(self, ticker, days=10):
        df = self.download_stock_data(ticker)
        X, y = self._prepare_sequences(df)

        split = int(len(X) * 0.8)
        X_test = X[split:]
        y_test = y[split:]

        pred_scaled_test = self.model.predict(X_test, verbose=0).reshape(-1, 1)

        # Desnormalize
        other = X_test[:, -1, 1:]
        inv_pred_test = self.scaler.inverse_transform(np.hstack([pred_scaled_test, other]))[:, 0]
        inv_real_test = self.scaler.inverse_transform(np.hstack([y_test.reshape(-1, 1), other]))[:, 0]

        # Previsão futura
        last_seq = X_test[-1].copy()
        future_scaled = []
        for _ in range(days):
            p = self.model.predict(last_seq[np.newaxis, :, :], verbose=0)[0, 0]
            future_scaled.append(p)
            new_row = np.hstack([[p], last_seq[-1, 1:]])
            last_seq = np.vstack([last_seq[1:], new_row])

        future_prices = self.scaler.inverse_transform(
            np.hstack([
                np.array(future_scaled).reshape(-1, 1),
                np.tile(last_seq[-1, 1:], (days, 1))
            ])
        )[:, 0]

        future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=days)

        result_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted_Price": np.round(future_prices, 2)
        })

        return result_df, df, {"MAE": round(0,2)}

# ----------------------------
# UI - Sidebar
# ----------------------------
st.sidebar.header("🎯 Configurações")

ticker = st.sidebar.text_input(
    "Digite o código da ação:",
    value="AAPL",
    help="Ex: AAPL, TSLA, PETR4.SA"
).upper().strip()

predict_button = st.sidebar.button("🎯 Fazer Previsão", type="primary")

st.sidebar.markdown("---")
st.sidebar.subheader("💡 Ações Sugeridas")
st.sidebar.markdown("""
**EUA:**
- AAPL, TSLA, GOOGL
- MSFT, AMZN, META

**Brasil:**
- PETR4.SA, VALE3.SA
- ITUB4.SA, BBDC4.SA
""")

# ----------------------------
# Processamento principal
# ----------------------------
if predict_button and ticker:
    with st.spinner("🤖 Analisando dados e gerando previsões..."):
        # Barra de progresso visual (só UX)
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.005)
            progress_bar.progress(i + 1)

        # Inicializar previsor
        forecaster = StockForecaster()
        predictions_df, historical_data, metrics = forecaster.predict_future(ticker, days=10)

    if predictions_df is not None:
        st.success(f"✅ Previsão concluída para {ticker}!")

        # Resumo com métricas e preço atual
        st.subheader("📊 Resumo da Previsão")
        col1, col2, col3 = st.columns(3)

        with col1:
            current_price = historical_data['Price'].iloc[-1]
            st.metric("Preço Atual", f"${current_price:.2f}")

        with col2:
            avg_price = predictions_df["Predicted_Price"].mean()
            st.metric("Preço Médio Previsto", f"${avg_price:.2f}")

        with col3:
            total_change = ((predictions_df["Predicted_Price"].iloc[-1] - current_price) / current_price) * 100
            st.metric("Variação Total", f"{total_change:.2f}%")

        # Métricas do modelo
        st.subheader("📌 Métricas do Modelo (sobre X_test)")
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("MAE", f"${metrics['MAE']}")
        mcol2.metric("RMSE", f"${metrics['RMSE']}")
        mcol3.metric("MAPE", f"{metrics['MAPE']}%" if metrics['MAPE'] is not None else "N/A")
        mcol4.metric("Direction Accuracy", f"{metrics['Direction_Accuracy']}%" if metrics['Direction_Accuracy'] is not None else "N/A")

        # Tabela de previsões
        st.subheader("📅 Previsões Detalhadas")
        display_df = predictions_df.copy()
        display_df["Date"] = display_df["Date"].dt.strftime("%d/%m/%Y")
        display_df = display_df.rename(columns={
            "Date": "Data",
            "Predicted_Price": "Preço Previsto (USD)",
            "Pct_Change": "Variação (%)"
        })

        st.dataframe(display_df.style.format({
            "Preço Previsto (USD)": "{:.2f}",
            "Variação (%)": "{:.2f}%"
        }), use_container_width=True)

        # Gráfico: histórico + previsão (preço atual linha)
        st.subheader("📈 Visualização Gráfica")
        fig = go.Figure()

        # adicionar histórico de preço (últimos 120 dias ou todo o histórico pequeno)
        hist_plot = historical_data['Price'].iloc[-180:]
        fig.add_trace(go.Scatter(
            x=hist_plot.index,
            y=hist_plot.values,
            mode='lines',
            name='Preço Histórico',
            line=dict(width=2)
        ))

        # Preços previstos
        fig.add_trace(go.Scatter(
            x=predictions_df["Date"],
            y=predictions_df["Predicted_Price"],
            mode='lines+markers',
            name='Preço Previsto',
            line=dict(dash='dash', width=3),
            marker=dict(size=8)
        ))

        # Linha do preço atual
        fig.add_hline(
            y=current_price,
            line_dash="dash",
            line_color="blue",
            annotation_text=f"Preço Atual: ${current_price:.2f}",
            annotation_position="top left"
        )

        fig.update_layout(
            title=f"Previsão de Preços - {ticker}",
            xaxis_title="Data",
            yaxis_title="Preço (USD)",
            height=600,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Análise simples
        st.subheader("💡 Análise")
        if total_change > 0:
            st.success(f"**Tendência de ALTA** - Previsão de valorização de {total_change:.2f}%")
        else:
            st.error(f"**Tendência de BAIXA** - Previsão de desvalorização de {abs(total_change):.2f}%")

# Rodapé
st.markdown("---")
st.markdown("🔮 *Previsões geradas por modelo de IA - Use para análise*")