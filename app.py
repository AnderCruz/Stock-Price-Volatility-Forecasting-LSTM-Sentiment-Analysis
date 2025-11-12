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
# Helper cached function (fora da classe)
# ----------------------------
@st.cache_data(show_spinner=False)
def cached_yf_download(ticker: str, period: str = "max") -> pd.DataFrame:
    """Baixa dados via yfinance com cache (função fora da classe para evitar problemas de hashing)."""
    raw = yf.download(ticker, period=period, progress=False)
    if raw is None:
        return pd.DataFrame()
    return raw


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

st.markdown('<h1 class="main-header">🤖 Previsão de Ações com IA (VERSÃO ESTÁVEL)</h1>', unsafe_allow_html=True)
st.markdown("### Previsão dos próximos 10 dias usando modelo treinado (LSTM) — use apenas para análise, não é recomendação financeira.")


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

        # Try to load assets
        self._download_and_load_assets()

    def _download_and_load_assets(self):
        st.info("📥 Baixando arquivos do modelo (se necessário)...")

        # Faz o download somente se não existir localmente
        try:
            if not os.path.exists(self.model_file):
                gdown.download(self.model_url, self.model_file, quiet=True)
            if not os.path.exists(self.scaler_file):
                gdown.download(self.scaler_url, self.scaler_file, quiet=True)
            if not os.path.exists(self.info_file):
                gdown.download(self.info_url, self.info_file, quiet=True)
        except Exception as e:
            st.warning(f"Não foi possível baixar alguns arquivos automaticamente: {e}. Tentando carregar localmente (se existirem).")

        # Carregar modelo
        try:
            if not os.path.exists(self.model_file):
                raise FileNotFoundError(f"Arquivo do modelo não encontrado em: {self.model_file}")
            # tenta carregar respeitando a função rmse registrada
            self.model = load_model(self.model_file, compile=False)
            st.success("✅ Modelo LSTM carregado com sucesso!")
        except Exception as e:
            st.error(f"Erro ao carregar modelo: {e}")
            raise

        # Carregar scaler e model_info (joblib)
        try:
            if not os.path.exists(self.scaler_file):
                raise FileNotFoundError(f"Arquivo scaler não encontrado em: {self.scaler_file}")
            self.scaler = joblib.load(self.scaler_file)
        except Exception as e:
            st.error(f"Erro ao carregar scaler: {e}")
            raise

        try:
            if not os.path.exists(self.info_file):
                # model_info pode ser opcional; permitimos seguir sem ele (mas avisamos)
                st.warning("Arquivo model_info não encontrado — continuando sem ele.")
                self.model_info = {}
            else:
                self.model_info = joblib.load(self.info_file)
        except Exception as e:
            st.warning(f"Erro ao carregar model_info: {e}")
            self.model_info = {}

        st.success("✅ Scaler e Model Info carregados com sucesso (quando disponível).")

    def download_stock_data(self, ticker):
        """
        Baixa dados históricos via yfinance e prepara colunas: Price, Volume, Sentiment.
        Essa função **não** tem cache (o cache foi movido para cached_yf_download fora da classe).
        """
        # Baixa máximo histórico para garantir sequência suficiente
        try:
            raw = cached_yf_download(ticker)
        except Exception as e:
            raise RuntimeError(f"Erro ao baixar dados do ticker {ticker}: {e}")

        if raw is None or raw.empty:
            raise RuntimeError(f"Nenhum dado encontrado para o ticker {ticker}.")

        df = raw.copy()

        # Prepara colunas padronizadas
        # Preferimos 'Adj Close' se disponível, senão 'Close'
        if 'Adj Close' in df.columns:
            df['Price'] = df['Adj Close']
        else:
            df['Price'] = df['Close']

        # Volume
        if 'Volume' not in df.columns:
            df['Volume'] = 0
        else:
            # preencher NA
            df['Volume'] = df['Volume'].fillna(0)

        # Sentiment (se não tivermos uma fonte, assumimos zeros)
        if 'Sentiment' not in df.columns:
            df['Sentiment'] = 0.0

        # Mantemos apenas as colunas necessárias e garantimos índice datetime
        df = df[['Price', 'Volume', 'Sentiment']]
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        return df

    def _prepare_sequences(self, df):
        """
        Gera X e y a partir do dataframe com colunas ['Price','Volume','Sentiment'].
        """
        values = df[["Price", "Volume", "Sentiment"]].values.astype(float)

        # Guardamos uma cópia antes da escala para referência (opcional)
        scaled = self.scaler.transform(values)

        X = []
        y = []
        for i in range(self.SEQ_LEN, len(scaled)):
            X.append(scaled[i-self.SEQ_LEN:i])
            y.append(scaled[i, 0])  # preço (coluna 0) escalado
        if len(X) == 0:
            raise ValueError(f"Dados insuficientes para formar sequências. Requerido seq_len={self.SEQ_LEN} mas temos {len(df)} registros.")
        return np.array(X), np.array(y)

    def predict_future(self, ticker, days=10):
        """
        Retorna (predictions_df, historical_df, metrics)
        """
        # Baixa dados
        df = self.download_stock_data(ticker)

        # Prepara sequências
        X, y = self._prepare_sequences(df)

        # Split simples (80% treino / 20% teste)
        split = int(len(X) * 0.8)
        if split == 0:
            split = 1
        X_test = X[split:]
        y_test = y[split:]

        # Previsão sobre X_test
        pred_scaled_test = self.model.predict(X_test, verbose=0).reshape(-1, 1)  # shape (n,1)

        # Para inverter a escala precisamos juntar o pred (col 0) com as outras features (col 1 e 2)
        other_test = X_test[:, -1, 1:]  # pega Volume e Sentiment do último passo da sequência
        inv_pred_test = self.scaler.inverse_transform(np.hstack([pred_scaled_test, other_test]))[:, 0]
        inv_real_test = self.scaler.inverse_transform(np.hstack([y_test.reshape(-1,1), other_test]))[:, 0]

        # Métricas (baseadas no X_test)
        mae = mean_absolute_error(inv_real_test, inv_pred_test)
        rmse_val = mean_squared_error(inv_real_test, inv_pred_test, squared=False)
        # MAPE: cuidado com divisão por zero
        denom = np.where(inv_real_test == 0, 1e-8, inv_real_test)
        mape = np.mean(np.abs((inv_real_test - inv_pred_test) / denom)) * 100

        # Direction accuracy (previsão do sinal do movimento de um dia para o outro)
        if len(inv_real_test) >= 2:
            real_diff = np.diff(inv_real_test)
            pred_diff = np.diff(inv_pred_test)
            direction_acc = (np.sum((real_diff * pred_diff) > 0) / len(real_diff)) * 100
        else:
            direction_acc = None

        # Previsão futura 'days' usando a última sequência disponível do X_test (ou do X se X_test vazio)
        if len(X_test) == 0:
            last_seq = X[-1].copy()
        else:
            last_seq = X_test[-1].copy()

        future_scaled = []
        seq = last_seq.copy()
        for _ in range(days):
            p_scaled = self.model.predict(seq[np.newaxis, :, :], verbose=0)[0, 0]
            future_scaled.append(p_scaled)
            # Construímos nova linha mantendo Volume/Sentiment do último passo sempre (simplificação)
            new_row = np.hstack([[p_scaled], seq[-1, 1:]])
            seq = np.vstack([seq[1:], new_row])

        # Inverte escala das previsões futuras
        future_other = np.tile(last_seq[-1, 1:], (days, 1))  # mantém Volume/Sentiment do último passo
        inv_future = self.scaler.inverse_transform(np.hstack([np.array(future_scaled).reshape(-1,1), future_other]))[:, 0]

        # Datas futuras
        last_date = df.index[-1]
        future_dates = pd.date_range(last_date + timedelta(days=1), periods=days)

        # DataFrame de previsões
        pred_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted_Price": np.round(inv_future, 2)
        })

        # Coluna de variação percentual (dia-a-dia considerando previsões)
        pred_df['Pct_Change'] = pred_df['Predicted_Price'].pct_change().fillna((pred_df['Predicted_Price'].iloc[0] - df['Price'].iloc[-1]) / df['Price'].iloc[-1]) * 100

        metrics = {
            "MAE": float(np.round(mae, 4)),
            "RMSE": float(np.round(rmse_val, 4)),
            "MAPE": float(np.round(mape, 4)),
            "Direction_Accuracy": None if direction_acc is None else float(np.round(direction_acc, 2))
        }

        return pred_df, df, metrics


# ----------------------------
# UI - Sidebar
# ----------------------------
st.sidebar.header("🎯 Configurações")

ticker = st.sidebar.text_input(
    "Digite o código da ação:",
    value="AAPL",
    help="Ex: AAPL, TSLA, PETR4.SA"
).upper().strip()

days = st.sidebar.number_input("Dias para previsão", min_value=1, max_value=60, value=10, step=1)
seq_len = st.sidebar.number_input("SEQ_LEN (comprimento da sequência LSTM)", min_value=10, max_value=240, value=60, step=1)

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
    # Indicador de processamento
    main_spinner = st.empty()
    main_spinner.info("🤖 Preparando e gerando previsões... (veja mensagens abaixo)")

    # Barra de progresso visual (só UX curto)
    progress_bar = st.progress(0)
    for i in range(20):
        time.sleep(0.02)
        progress_bar.progress(int((i+1) * (100/20)))

    # Inicializar previsor (carrega modelo/scaler)
    try:
        forecaster = StockForecaster(seq_len=int(seq_len))
    except Exception as e:
        st.error(f"Falha ao inicializar o forecaster: {e}")
        st.stop()

    # Gerar previsões
    try:
        predictions_df, historical_data, metrics = forecaster.predict_future(ticker, days=int(days))
    except Exception as e:
        st.error(f"Erro durante a predição: {e}")
        st.stop()

    main_spinner.empty()
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
        st.metric("Variação Total (10 dias)", f"{total_change:.2f}%")

    # Métricas do modelo
    st.subheader("📌 Métricas do Modelo (sobre X_test)")
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("MAE", f"{metrics.get('MAE', 'N/A')}")
    mcol2.metric("RMSE", f"{metrics.get('RMSE', 'N/A')}")
    mcol3.metric("MAPE", f"{metrics.get('MAPE', 'N/A')}%")
    mcol4.metric("Direction Accuracy", f"{metrics.get('Direction_Accuracy', 'N/A')}%")

    # Tabela de previsões
    st.subheader("📅 Previsões Detalhadas")
    display_df = predictions_df.copy()
    display_df["Date"] = display_df["Date"].dt.strftime("%d/%m/%Y")
    display_df = display_df.rename(columns={
        "Date": "Data",
        "Predicted_Price": "Preço Previsto (USD)",
        "Pct_Change": "Variação (%)"
    })
    # Formatação
    st.dataframe(display_df.style.format({
        "Preço Previsto (USD)": "{:.2f}",
        "Variação (%)": "{:.2f}%"
    }), use_container_width=True)

    # Gráfico: histórico + previsão (preço atual linha)
    st.subheader("📈 Visualização Gráfica")
    fig = go.Figure()

    # adicionar histórico de preço (últimos 180 dias ou todo o histórico)
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
        x=pd.to_datetime(predictions_df["Date"]),
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
st.markdown("🔮 *Previsões geradas por modelo de IA - Use para análise apenas. Não constitui aconselhamento financeiro.*")
