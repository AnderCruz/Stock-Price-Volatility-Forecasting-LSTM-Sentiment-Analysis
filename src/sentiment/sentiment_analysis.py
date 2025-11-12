import numpy as np
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer

# ===============================================
# 🎯 MÉTODOS DE ANÁLISE DE SENTIMENTO
# ===============================================

def analyze_sentiment_textblob(text):
    """Analisa sentimento usando TextBlob"""
    if not text or str(text).strip() == "":
        return 0
    try:
        analysis = TextBlob(str(text))
        return analysis.sentiment.polarity
    except:
        return 0

def analyze_sentiment_vader(text):
    """Analisa sentimento usando VADER"""
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        scores = sia.polarity_scores(str(text))
        return scores['compound']
    except:
        return analyze_sentiment_textblob(text)

def analyze_sentiment_enhanced(text):
    """Análise de sentimento melhorada para termos financeiros"""
    text = str(text).lower()
    
    # Palavras-chave positivas
    positive_terms = {
        'bullish': 0.9, 'surge': 0.8, 'rally': 0.8, 'soar': 0.8, 'jump': 0.7,
        'gain': 0.7, 'profit': 0.7, 'beat': 0.8, 'upgrade': 0.7, 'buy': 0.6,
        'outperform': 0.7, 'growth': 0.6, 'record': 0.6, 'high': 0.5, 'all-time high': 0.9,
        'positive': 0.6, 'strong': 0.5, 'opportunity': 0.4, 'rise': 0.6, 'increase': 0.5,
        'success': 0.6, 'win': 0.5, 'boom': 0.7, 'optimistic': 0.7, 'breakthrough': 0.8,
        'leader': 0.5, 'boost': 0.6, 'recovery': 0.5, 'momentum': 0.5, 'dividend': 0.4,
        'earnings beat': 0.9, 'revenue beat': 0.8, 'guidance raise': 0.8, 'target raise': 0.7
    }
    
    # Palavras-chave negativas
    negative_terms = {
        'bearish': -0.9, 'plunge': -0.8, 'crash': -0.9, 'collapse': -0.9, 'drop': -0.7,
        'loss': -0.7, 'miss': -0.8, 'downgrade': -0.7, 'sell': -0.6, 'warning': -0.7,
        'fall': -0.6, 'decline': -0.6, 'low': -0.5, 'negative': -0.6, 'weak': -0.5,
        'risk': -0.4, 'concern': -0.5, 'trouble': -0.6, 'problem': -0.5, 'volatility': -0.4,
        'uncertainty': -0.5, 'challenge': -0.4, 'slide': -0.6, 'slump': -0.7, 'dip': -0.5,
        'pressure': -0.4, 'cut': -0.6, 'layoff': -0.7, 'downside': -0.6, 'earnings miss': -0.9,
        'guidance cut': -0.8, 'lawsuit': -0.7, 'investigation': -0.6, 'target cut': -0.7
    }
    
    # Calcular score base
    base_score = 0
    word_count = 0
    
    for word, weight in positive_terms.items():
        if word in text:
            base_score += weight
            word_count += 1
    
    for word, weight in negative_terms.items():
        if word in text:
            base_score += weight
            word_count += 1
    
    # Normalizar
    if word_count > 0:
        final_score = base_score / min(word_count, 8)
    else:
        final_score = 0
    
    # Aplicar função de ativação
    final_score = np.tanh(final_score * 1.5)
    
    return max(-1.0, min(1.0, final_score))

def get_sentiment_label(score):
    """Converte score numérico em label"""
    if score >= 0.25:
        return "POSITIVO", "🟢"
    elif score >= 0.1:
        return "LEVE POSITIVO", "🟡"
    elif score <= -0.25:
        return "NEGATIVO", "🔴"
    elif score <= -0.1:
        return "LEVE NEGATIVO", "🟠"
    else:
        return "NEUTRO", "⚪"
