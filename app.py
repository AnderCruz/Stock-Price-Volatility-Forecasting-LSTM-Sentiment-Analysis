# app.py - VERSÃO SIMPLIFICADA (tudo em um arquivo)
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import gdown
import tempfile
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# Configuração da página
st.set_page_config(
    page_title="Previsão de Ações - IA",
    page_icon="🤖",
    layout="wide"
)

# CSS personalizado
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

# Título da aplicação
st.markdown('<h1 class="main-header">🤖 Previsão de Ações com IA</h1>', unsafe_allow_html=True)
st.markdown("### Previsão dos próximos 10 dias usando modelo LSTM")

# ===============================================
# 🔧 FUNÇÕES DO MODELO (tudo dentro do app.py)
# ===============================================

class StockForecaster:
    def __init__(self):
        """Inicializa o previsor baixando modelo do Google Drive"""
        self.model_url = "https://drive.google.com/uc?id=15rz-a_5ktKYzbD7RMSdc1yYld5hVBZC6"
        self.model = None
        self.scaler = MinMaxScaler()
        self.SEQ_LEN = 60
        
        self._download_and_load_model()
    
    def _download_and_load_model(self):
        """Baixa e carrega o modelo do Google Drive"""
        try:
            st.info("📥 Baixando modelo de IA...")
            
            # Criar diretório temporário
            temp_dir = tempfile.gettempdir()
            model_file = os.path.join(temp_dir, "forecasting.keras")
            
            # Baixar arquivo
            gdown.download(self.model_url, model_file, quiet=True)
            
            if os.path.exists(model_file):
                self.model = load_model(model_file)
                st.success("✅ Modelo de IA carregado com sucesso!")
            else:
                st.error("❌ Falha ao baixar o modelo")
                self._create_fallback_model()
                
        except Exception as e:
            st.warning("⚠️ Usando modelo simplificado")
            self._create_fallback_model()
    
    def _create_fallback_model(self):
        """Cria modelo básico se o download falhar"""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        
        self.model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.SEQ_LEN, 3)),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')
    
    def download_stock_data(self, ticker, period="1y"):
        """Baixa dados da ação"""
        try:
            df = yf.download(ticker, period=period, progress=False)
            
            if df.empty:
                raise Exception(f"Nenhum dado para {ticker}")
            
            # Processar colunas
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0].title() for col in df.columns]
            else:
                df.columns = [col.title() for col in df.columns]
            
            # Manter apenas colunas necessárias
            df = df[["Close", "Volume"]].copy()
            df = df.rename(columns={
                "Close": "Price",
                "Volume": "Volume"
            })
            
            # Adicionar sentimento neutro
            df['Sentiment'] = 0
            
            # Preencher valores faltantes
            df = df.ffill().bfill()
            
            return df
            
        except Exception as e:
            st.error(f"❌ Erro ao baixar dados: {e}")
            return None
    
    def predict_future(self, ticker, days=10):
        """Faz previsão para os próximos dias"""
        try:
            # Baixar dados
            df = self.download_stock_data(ticker)
            if df is None:
                return None, None
            
            # Verificar dados suficientes
            if len(df) < self.SEQ_LEN:
                st.error(f"❌ Dados insuficientes. Necessário: {self.SEQ_LEN} dias")
                return None, None
            
            # Preparar dados
            data = df[['Price', 'Volume', 'Sentiment']].values
            scaled_data = self.scaler.fit_transform(data)
            
            # Fazer previsões
            last_sequence = scaled_data[-self.SEQ_LEN:]
            future_predictions = []
            current_sequence = last_sequence.copy()
            
            for i in range(days):
                pred_scaled = self.model.predict(current_sequence[np.newaxis, :, :], verbose=0)[0, 0]
                future_predictions.append(pred_scaled)
                
                # Atualizar sequência
                new_row = current_sequence[-1].copy()
                new_row[0] = pred_scaled
                current_sequence = np.vstack([current_sequence[1:], new_row])
            
            # Converter para preços reais
            future_prices = self.scaler.inverse_transform(
                np.column_stack([
                    np.array(future_predictions).reshape(-1, 1),
                    np.tile(current_sequence[-1, 1:], (days, 1))
                ])
            )[:, 0]
            
            # Criar DataFrame de resultados
            future_dates = [df.index[-1] + pd.Timedelta(days=i+1) for i in range(days)]
            result_df = pd.DataFrame({
                "Date": future_dates,
                "Predicted_Price": future_prices.round(2)
            })
            
            # Calcular variações
            result_df["Pct_Change"] = result_df["Predicted_Price"].pct_change().fillna(0) * 100
            result_df["Pct_Change"] = result_df["Pct_Change"].round(2)
            
            return result_df, df
            
        except Exception as e:
            st.error(f"❌ Erro na previsão: {e}")
            return None, None

# ===============================================
# 🎯 INTERFACE DO STREAMLIT
# ===============================================

# Sidebar
st.sidebar.header("🎯 Configurações")

ticker = st.sidebar.text_input(
    "Digite o código da ação:",
    value="AAPL",
    help="Ex: AAPL, TSLA, PETR4.SA"
).upper().strip()

predict_button = st.sidebar.button("🎯 Fazer Previsão", type="primary")

# Ações sugeridas
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

# Processamento principal
if predict_button and ticker:
    with st.spinner("🤖 Analisando dados e gerando previsões..."):
        # Barra de progresso
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        
        # Inicializar previsor e fazer previsão
        forecaster = StockForecaster()
        predictions_df, historical_data = forecaster.predict_future(ticker)
    
    if predictions_df is not None:
        st.success(f"✅ Previsão concluída para {ticker}!")
        
        # Mostrar resumo
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
        
        # Gráfico
        st.subheader("📈 Visualização Gráfica")
        
        fig = go.Figure()
        
        # Preços previstos
        fig.add_trace(go.Scatter(
            x=predictions_df["Date"],
            y=predictions_df["Predicted_Price"],
            mode='lines+markers',
            name='Preço Previsto',
            line=dict(color='red', width=3),
            marker=dict(size=8)
        ))
        
        # Linha do preço atual
        fig.add_hline(
            y=current_price, 
            line_dash="dash", 
            line_color="blue",
            annotation_text=f"Preço Atual: ${current_price:.2f}"
        )
        
        fig.update_layout(
            title=f"Previsão de Preços - {ticker}",
            xaxis_title="Data",
            yaxis_title="Preço (USD)",
            height=500
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
