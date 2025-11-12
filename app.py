# app_improved.py -- Versão melhorada do seu Streamlit app (único arquivo)
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import gdown
import tempfile
import os
import joblib
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from datetime import timedelta
import keras
from keras import backend as K

# ---------------------------
# Funções auxiliares / custom metrics
# ---------------------------
@keras.saving.register_keras_serializable(package="Custom")
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

# ---------------------------
# Configuração Streamlit
# ---------------------------
st.set_page_config(page_title="Previsão de Ações - IA (Melhorada)", page_icon="🤖", layout="wide")
st.title("🤖 Previsão de Ações com IA — Versão Melhorada")
st.markdown("Use apenas para análise — não é recomendação financeira.")

# ---------------------------
# StockForecaster (refatorado)
# ---------------------------
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
        self.SEQ_LEN = int(seq_len)

        temp_dir = tmp_dir or tempfile.gettempdir()
        self.model_file = os.path.join(temp_dir, "forecasting.keras")
        self.scaler_file = os.path.join(temp_dir, "scaler.pkl")
        self.info_file = os.path.join(temp_dir, "model_info.pkl")

        # Lazy-load: model/scaler carregados via cache_resource
        # (a chamada abaixo apenas garante que os arquivos locais estejam presentes se houver URLs)
        self._ensure_files_from_urls()

    def _ensure_files_from_urls(self):
        # Se URLs forem fornecidas, tenta baixar (apenas se arquivo local não existir)
        try:
            if self.model_url and not os.path.exists(self.model_file):
                gdown.download(self.model_url, self.model_file, quiet=True)
            if self.scaler_url and not os.path.exists(self.scaler_file):
                gdown.download(self.scaler_url, self.scaler_file, quiet=True)
            if self.info_url and not os.path.exists(self.info_file):
                gdown.download(self.info_url, self.info_file, quiet=True)
        except Exception as e:
            st.warning(f"Falha ao baixar arquivos automaticamente: {e}")

    # Carrega modelo (cacheado como recurso - evitar reloads caros)
    @st.cache_resource
    def load_model_resource(self, model_path=None):
        path = model_path or self.model_file
        if not os.path.exists(path):
            raise FileNotFoundError(f"Modelo não encontrado em: {path}")
        model = load_model(path, custom_objects={"rmse": rmse})
        return model

    @st.cache_resource
    def load_scaler_resource(self, scaler_path=None):
        path = scaler_path or self.scaler_file
        if not os.path.exists(path):
            raise FileNotFoundError(f"Scaler não encontrado em: {path}")
        scaler = joblib.load(path)
        return scaler

    # Transformamos o download em staticmethod para evitar hashing de self
    @staticmethod
    @st.cache_data(show_spinner=False)
    def download_stock_data(ticker: str, prefer_adj_close: bool = True):
        """
        Baixa dados históricos via yfinance e prepara colunas: Price, Volume, Sentiment.
        Cacheado por ticker para acelerar múltiplas chamadas.
        """
        try:
            raw = yf.download(ticker, period="max", progress=False)
        except Exception as e:
            raise RuntimeError(f"Erro ao baixar dados do ticker {ticker}: {e}")

        if raw is None or raw.empty:
            raise RuntimeError(f"Nenhum dado encontrado para o ticker {ticker}.")

        df = raw.copy()
        # Escolha de coluna de preço
        if prefer_adj_close and 'Adj Close' in df.columns:
            df['Price'] = df['Adj Close']
        else:
            df['Price'] = df['Close']

        # Volume
        if 'Volume' in df.columns:
            df['Volume'] = df['Volume'].fillna(0)
        else:
            df['Volume'] = 0

        # Sentiment (placeholder - zeros)
        df['Sentiment'] = 0.0

        df = df[['Price', 'Volume', 'Sentiment']]
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df

    def _prepare_sequences(self, df, scaler):
        """
        Gera X e y a partir do dataframe com colunas ['Price','Volume','Sentiment'].
        Retorna X, y, scaled_values (útil para reconstruir / métricas).
        """
        values = df[["Price", "Volume", "Sentiment"]].values.astype(float)
        scaled = scaler.transform(values)

        X = []
        y = []
        for i in range(self.SEQ_LEN, len(scaled)):
            X.append(scaled[i-self.SEQ_LEN:i])
            y.append(scaled[i, 0])
        if len(X) == 0:
            raise ValueError(f"Dados insuficientes para formar sequências. Requer seq_len={self.SEQ_LEN} mas temos {len(df)} registros.")
        return np.array(X), np.array(y), scaled

    def predict_future(self, ticker, days=10, model_path=None, scaler_path=None):
        """
        Retorna:
         - pred_df (datas, preco previsto, pct change, low95, high95)
         - historical_df (df com Price etc)
         - metrics (MAE, RMSE, MAPE, Direction_Accuracy)
        """
        # Baixa dados (cacheado)
        df = StockForecaster.download_stock_data(ticker)

        # Carrega recursos (modelo/scaler) via cache_resource (permite upload override)
        model = None
        scaler = None
        try:
            model = self.load_model_resource(model_path)
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar modelo: {e}")

        try:
            scaler = self.load_scaler_resource(scaler_path)
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar scaler: {e}")

        # Prepara sequências
        X, y, scaled_all = self._prepare_sequences(df, scaler)

        # Split simples (80% / 20%)
        split = int(len(X) * 0.8)
        if split == 0:
            split = 1
        X_test = X[split:]
        y_test = y[split:]

        # Previsões no teste
        pred_scaled_test = model.predict(X_test, verbose=0).reshape(-1, 1)
        other_test = X_test[:, -1, 1:]
        inv_pred_test = scaler.inverse_transform(np.hstack([pred_scaled_test, other_test]))[:, 0]
        inv_real_test = scaler.inverse_transform(np.hstack([y_test.reshape(-1,1), other_test]))[:, 0]

        # Métricas
        mae = mean_absolute_error(inv_real_test, inv_pred_test)
        rmse_val = mean_squared_error(inv_real_test, inv_pred_test, squared=False)
        denom = np.where(inv_real_test == 0, 1e-8, inv_real_test)
        mape = np.mean(np.abs((inv_real_test - inv_pred_test) / denom)) * 100

        # Direction accuracy
        if len(inv_real_test) >= 2:
            real_diff = np.diff(inv_real_test)
            pred_diff = np.diff(inv_pred_test)
            direction_acc = (np.sum((real_diff * pred_diff) > 0) / len(real_diff)) * 100
        else:
            direction_acc = None

        # Erro residual para intervalo de confiança (no espaço de preços)
        residuals = inv_real_test - inv_pred_test
        resid_std = np.std(residuals, ddof=1) if len(residuals) > 1 else 0.0

        # Previsão iterativa futura
        last_seq = X_test[-1].copy() if len(X_test) > 0 else X[-1].copy()
        future_scaled = []
        seq = last_seq.copy()
        for _ in range(days):
            p_scaled = model.predict(seq[np.newaxis, :, :], verbose=0)[0, 0]
            future_scaled.append(p_scaled)
            new_row = np.hstack([[p_scaled], seq[-1, 1:]])
            seq = np.vstack([seq[1:], new_row])

        future_other = np.tile(last_seq[-1, 1:], (days, 1))
        inv_future = scaler.inverse_transform(np.hstack([np.array(future_scaled).reshape(-1,1), future_other]))[:, 0]

        # Intervalo de confiança simples (assume resíduos gaussiano): 95% CI = pred ± 1.96 * resid_std
        low95 = inv_future - 1.96 * resid_std
        high95 = inv_future + 1.96 * resid_std

        # Datas futuras
        last_date = df.index[-1]
        future_dates = pd.date_range(last_date + timedelta(days=1), periods=days)

        pred_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted_Price": np.round(inv_future, 2),
            "Low95": np.round(low95, 2),
            "High95": np.round(high95, 2)
        })

        pred_df['Pct_Change'] = pred_df['Predicted_Price'].pct_change().fillna((pred_df['Predicted_Price'].iloc[0] - df['Price'].iloc[-1]) / df['Price'].iloc[-1]) * 100

        metrics = {
            "MAE": float(np.round(mae, 4)),
            "RMSE": float(np.round(rmse_val, 4)),
            "MAPE": float(np.round(mape, 4)),
            "Direction_Accuracy": None if direction_acc is None else float(np.round(direction_acc, 2)),
            "Resid_STD": float(np.round(resid_std, 4))
        }

        return pred_df, df, metrics

# ---------------------------
# UI - Sidebar (inputs e opções)
# ---------------------------
st.sidebar.header("Configurações")
ticker = st.sidebar.text_input("Código da ação (ticker):", value="AAPL").upper().strip()
days = int(st.sidebar.number_input("Dias para previsão", min_value=1, max_value=60, value=10, step=1))
seq_len = int(st.sidebar.number_input("SEQ_LEN (comprimento da sequência LSTM)", min_value=10, max_value=240, value=60, step=1))

st.sidebar.markdown("---")
st.sidebar.subheader("Modelo / Scaler")
use_drive = st.sidebar.checkbox("Usar URLs do Google Drive para modelo + scaler", value=True)
model_drive_url = None
scaler_drive_url = None
if use_drive:
    model_drive_url = st.sidebar.text_input("URL (Drive) do modelo (.keras / .h5) - leave blank to usar interno", value="")
    scaler_drive_url = st.sidebar.text_input("URL (Drive) do scaler (.pkl)", value="")
else:
    st.sidebar.markdown("Ou faça upload manual abaixo (na área principal).")

st.sidebar.markdown("---")
st.sidebar.subheader("Opções de Preço")
prefer_adj = st.sidebar.checkbox("Preferir 'Adj Close' quando disponível", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Ações sugeridas")
st.sidebar.write("AAPL, MSFT, TSLA, GOOGL  —  PETR4.SA, VALE3.SA, ITUB4.SA")

# Uploader (opcional)
st.sidebar.markdown("---")
st.sidebar.write("Se preferir, faça upload do seu modelo (Keras) e scaler (joblib):")
uploaded_model = st.sidebar.file_uploader("Upload do modelo (.keras/.h5)", type=["keras", "h5", "keras_model"], key="umodel")
uploaded_scaler = st.sidebar.file_uploader("Upload do scaler (.pkl)", type=["pkl"], key="uscaler")

predict_button = st.sidebar.button("Fazer Previsão", type="primary")

# ---------------------------
# Execução principal
# ---------------------------
if predict_button:
    if not ticker:
        st.error("Digite um ticker válido.")
    else:
        processing = st.empty()
        processing.info("Preparando e gerando previsões...")

        # Progress visual curto
        prog = st.progress(0)
        for i in range(15):
            time.sleep(0.02)
            prog.progress(int((i+1) * (100/15)))

        # Inicializa forecaster (irá baixar assets se URLs dadas)
        forecaster = StockForecaster(
            model_drive_url=(model_drive_url if model_drive_url else None),
            scaler_drive_url=(scaler_drive_url if scaler_drive_url else None),
            seq_len=seq_len
        )

        # Se usuário fez upload, gravar temporariamente e passar paths às funções de load
        model_path_override = None
        scaler_path_override = None
        try:
            if uploaded_model is not None:
                tmp_model_path = os.path.join(tempfile.gettempdir(), uploaded_model.name)
                with open(tmp_model_path, "wb") as f:
                    f.write(uploaded_model.getbuffer())
                model_path_override = tmp_model_path

            if uploaded_scaler is not None:
                tmp_scaler_path = os.path.join(tempfile.gettempdir(), uploaded_scaler.name)
                with open(tmp_scaler_path, "wb") as f:
                    f.write(uploaded_scaler.getbuffer())
                scaler_path_override = tmp_scaler_path

            # Gerar previsões
            pred_df, hist_df, metrics = forecaster.predict_future(
                ticker,
                days=days,
                model_path=model_path_override,
                scaler_path=scaler_path_override
            )
        except Exception as e:
            processing.empty()
            st.error(f"Erro durante a predição: {e}")
            st.stop()

        processing.empty()
        st.success(f"Previsão concluída para {ticker}")

        # Resumo
        st.subheader("Resumo da Previsão")
        c1, c2, c3 = st.columns(3)
        current_price = hist_df['Price'].iloc[-1]
        c1.metric("Preço Atual", f"${current_price:.2f}")
        c2.metric("Preço Médio Previsto", f"${pred_df['Predicted_Price'].mean():.2f}")
        total_change = ((pred_df['Predicted_Price'].iloc[-1] - current_price) / current_price) * 100
        c3.metric(f"Variação Total ({days} dias)", f"{total_change:.2f}%")

        # Métricas
        st.subheader("Métricas do Modelo (sobre X_test)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MAE", f"{metrics.get('MAE', 'N/A')}")
        m2.metric("RMSE", f"{metrics.get('RMSE', 'N/A')}")
        m3.metric("MAPE", f"{metrics.get('MAPE', 'N/A')}%")
        m4.metric("Direction Acc.", f"{metrics.get('Direction_Accuracy', 'N/A')}%")

        # Tabela de previsões
        st.subheader("Previsões Detalhadas (com 95% CI)")
        shown = pred_df.copy()
        shown["Date"] = shown["Date"].dt.strftime("%d/%m/%Y")
        shown = shown.rename(columns={
            "Date": "Data",
            "Predicted_Price": "Preço Previsto (USD)",
            "Pct_Change": "Variação (%)",
            "Low95": "Lower 95",
            "High95": "Upper 95"
        })
        st.dataframe(shown.style.format({
            "Preço Previsto (USD)": "{:.2f}",
            "Variação (%)": "{:.2f}%",
            "Lower 95": "{:.2f}",
            "Upper 95": "{:.2f}"
        }), use_container_width=True)

        # Gráfico histórico + previsões + CI
        st.subheader("Visualização Gráfica")
        fig = go.Figure()
        hist_plot = hist_df['Price'].iloc[-180:]
        fig.add_trace(go.Scatter(x=hist_plot.index, y=hist_plot.values, mode='lines', name='Preço Histórico'))

        fig.add_trace(go.Scatter(x=pd.to_datetime(pred_df["Date"]), y=pred_df["Predicted_Price"],
                                 mode='lines+markers', name='Preço Previsto', line=dict(dash='dash')))

        # CI como fill
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(pred_df["Date"]).tolist() + pd.to_datetime(pred_df["Date"])[::-1].tolist(),
            y=pred_df["High95"].tolist() + pred_df["Low95"][::-1].tolist(),
            fill='toself',
            name='Intervalo 95%',
            hoverinfo="skip",
            line=dict(color='rgba(255,255,255,0)')
        ))

        fig.add_hline(y=current_price, line_dash="dash", annotation_text=f"Preço Atual: ${current_price:.2f}", annotation_position="top left")

        fig.update_layout(title=f"Previsão de Preços - {ticker}", xaxis_title="Data", yaxis_title="Preço (USD)", height=600)
        st.plotly_chart(fig, use_container_width=True)

        # Análise simples
        st.subheader("Análise")
        if total_change > 0:
            st.success(f"Tendência de ALTA prevista: {total_change:.2f}% em {days} dias")
        else:
            st.error(f"Tendência de BAIXA prevista: {abs(total_change):.2f}% em {days} dias")

        # Rodapé / disclaimer
        st.markdown("---")
        st.markdown("🔮 *Previsões geradas por modelo de IA - Use para análise apenas. Não constitui aconselhamento financeiro.*")
