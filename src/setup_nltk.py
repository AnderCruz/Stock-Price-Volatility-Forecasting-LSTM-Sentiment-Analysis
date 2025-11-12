# setup_nltk.py

import nltk
from pathlib import Path

def setup_nltk():
    """Configura todos os recursos necessários do NLTK"""
    try:
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('punkt', quiet=True)
        print("✅ Recursos do NLTK baixados com sucesso!")
        return True
    except Exception as e:
        print(f"⚠️ Erro ao baixar recursos do NLTK: {e}")
        return False

# Executa automaticamente se rodar diretamente o arquivo
if __name__ == "__main__":
    setup_nltk()
