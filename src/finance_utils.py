from pathlib import Path
import yfinance as yf
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent  # volta um nível
CSV_PATH = BASE_DIR /"data"/"top100_stocks.csv"

def get_top_correlated_stocks(ticker="TSLA", top_n=5, csv_path=CSV_PATH, period="5y"):
    """
    Retorna os top N ativos mais correlacionados com o ativo alvo.
    """
    # Carregar CSV
    tickers_info = pd.read_csv(csv_path)

    # Lista completa de tickers para download
    all_tickers = [ticker] + tickers_info["Ticker"].tolist()

    # Baixar preços ajustados
    data = yf.download(all_tickers, period=period, auto_adjust=True)['Close']

    # Identificar tickers que falharam
    failed_tickers = [col for col in data.columns if data[col].isna().all()]
    if failed_tickers:
        print(f"\n⚠️ Tickers que falharam no download e serão removidos: {failed_tickers}")

    # Remover tickers inválidos
    data = data.drop(columns=failed_tickers)
    tickers_info = tickers_info[~tickers_info["Ticker"].isin(failed_tickers)].reset_index(drop=True)

    # Calcular log-retornos e correlação
    logrets = np.log(data / data.shift(1)).dropna()
    correlations = logrets.corr()[ticker].drop(ticker).sort_values(ascending=False)

    # Selecionar top N ativos
    top_assets = correlations.head(top_n).index.tolist()
    final_df = tickers_info[tickers_info["Ticker"].isin(top_assets)].copy()
    final_df["Correlation"] = final_df["Ticker"].map(correlations)
    final_df = final_df.sort_values("Correlation", ascending=False).reset_index(drop=True)

    additional = top_assets
    all_tickers_final = [ticker] + additional

    return {
        "ticker": ticker,
        "additional": additional,
        "all_tickers": all_tickers_final,
        "final_df": final_df
    }
