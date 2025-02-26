import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from datetime import datetime, timedelta
from scipy.signal import argrelextrema
from PIL import Image
import io
from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
from pytz import timezone

# --- Globale Konfiguration ---
colors = {
    "background": "#1a1a1a",
    "card": "#2c2c2c",
    "text": "#FFFFFF",
    "positive": "#00FF00",
    "negative": "#FF0000",
    "neutral": "#FFA500"
}

st.set_page_config(page_title="Professionelle Marktanalyse", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .main {background-color: #1a1a1a; color: #ffffff;}
    .sidebar .sidebar-content {background-color: #2c2c2c; color: #ffffff;}
    .sidebar .sidebar-content input, .sidebar .sidebar-content button {background-color: #404040; color: #ffffff; border: 1px solid #555;}
    .sidebar .sidebar-content button:hover {background-color: #555;}
    .stButton>button {background-color: #4CAF50; color: white; border-radius: 5px; padding: 10px 20px; font-size: 16px;}
    .stButton>button:hover {background-color: #45a049;}
    .stExpander {background-color: #2c2c2c; border: 1px solid #555; border-radius: 5px;}
    .stExpander > div > div {background-color: #2c2c2c;}
    .stExpander > div > div > p {color: #ffffff;}
    .stCheckbox > div > label {color: #ffffff;}
    h1, h2, h3 {color: #4CAF50 !important;}
    .css-1aumxhk {background-color: #1a1a1a;}
    .legend-toggle {cursor: pointer; padding: 5px 10px; background-color: #2c2c2c; border: 1px solid #555; border-radius: 5px; margin-bottom: 10px;}
    .legend-toggle:hover {background-color: #404040;}
    </style>
""", unsafe_allow_html=True)

# Funktion zur Ermittlung des Live-Preises (inkl. Pre-Market/After-Market)
def get_live_price(symbol):
    try:
        stock = yf.Ticker(symbol)
        # Hole die aktuellsten Daten, inklusive Pre-Market/After-Market
        ticker_info = stock.history(period="1d", interval="1m")
        if not ticker_info.empty:
            current_price = ticker_info['Close'].iloc[-1]
            # Prüfe auf Pre-Market/After-Market-Daten
            pre_market = stock.pre_market_price if hasattr(stock, 'pre_market_price') and stock.pre_market_price else None
            after_market = stock.after_hours_price if hasattr(stock, 'after_hours_price') and stock.after_hours_price else None
            
            price_display = f"{current_price:.2f} USD"
            if pre_market and datetime.now().hour < 9:  # Pre-Market (vor 9:30 EST)
                price_display = f"Pre-Market: {pre_market:.2f} USD"
            elif after_market and datetime.now().hour >= 16:  # After-Market (nach 16:00 EST)
                price_display = f"After-Market: {after_market:.2f} USD"
            return current_price, price_display
        return None, "Keine aktuellen Daten verfügbar"
    except Exception as e:
        return None, f"Fehler beim Abrufen des Preises: {str(e)}"

# Funktion zur Bereinigung von Lücken (nur reguläre Handelstage, 9:30–16:00 EST, aktueller Tag)
def clean_trading_data(df):
    if df.empty:
        return df
    # Entferne Lücken und behalte nur Handelstage (entferne nicht-handelszeitliche Daten)
    df = df.dropna()
    # Sicherstellen, dass nur Daten während der regulären Handelszeiten des aktuellen Datums enthalten sind (9:30–16:00 EST)
    today = datetime.now(timezone('America/New_York')).replace(hour=0, minute=0, second=0, microsecond=0)
    df = df[df.index.date == today.date()].between_time('09:30', '16:00', inclusive='both')
    return df

# Funktion zur Erstellung eines kontinuierlichen Datensatzes für 1-Minuten-Intervalle, nur für den aktuellen Tag und Handelszeiten
def create_continuous_1m_data(df, period_days=1):
    if df.empty:
        return pd.DataFrame()
    
    # Hole das aktuelle Datum mit Zeitzone America/New_York
    current_date = datetime.now(timezone('America/New_York'))
    today = current_date.replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Erstelle einen vollständigen 1-Minuten-Index für den aktuellen Tag, nur für Handelszeiten (9:30–16:00 EST)
    start_time = today.replace(hour=9, minute=30, second=0)
    end_time = today.replace(hour=16, minute=0, second=0)
    date_range = pd.date_range(start=start_time, end=end_time, freq='1min', tz='America/New_York')
    continuous_df = pd.DataFrame(index=date_range)
    
    # Konvertiere den Index von df in die gleiche Zeitzone, falls nicht schon vorhanden
    if df.index.tz is None:
        df.index = df.index.tz_localize('America/New_York')
    else:
        df.index = df.index.tz_convert('America/New_York')
    
    # Filtere df auf den aktuellen Tag
    df_today = df[df.index.date == today.date()]
    
    # Füge die verfügbaren Daten hinzu und fülle fehlende Werte mit den letzten bekannten Werten oder 0
    required_columns = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_columns:
        if col in df_today.columns:
            continuous_df[col] = df_today[col].reindex(date_range, method='ffill').fillna(0)
        else:
            continuous_df[col] = 0  # Standardwert, falls Spalte nicht vorhanden
    
    return continuous_df

# Funktionen für Indikatoren (unverändert)
def fetch_data(symbol, timeframe="1d", period="1y"):
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period=period, interval=timeframe)
        return df if not df.empty else pd.DataFrame()
    except:
        return pd.DataFrame()

def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(data, period=20):
    data = data.copy()
    data['MA20'] = data['Close'].rolling(window=period).mean()
    data['STD20'] = data['Close'].rolling(window=period).std()
    data['Upper'] = data['MA20'] + (2 * data['STD20'])
    data['Lower'] = data['MA20'] - (2 * data['STD20'])
    return data

def calculate_ema(data, period=50):
    data = data.copy()
    data['EMA50'] = data['Close'].ewm(span=period, adjust=False).mean()
    return data

def calculate_macd(data, fast=12, slow=26, signal=9):
    exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_stochastic(data, period=14):
    low_min = data['Low'].rolling(window=period).min()
    high_max = data['High'].rolling(window=period).max()
    k = 100 * (data['Close'] - low_min) / (high_max - low_min)
    d = k.rolling(window=3).mean()
    return k, d

def calculate_vwap(data):
    data = data.copy()
    data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
    return data

def calculate_obv(data):
    data = data.copy()
    obv = [0]
    for i in range(1, len(data)):
        if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
            obv.append(obv[-1] + data['Volume'].iloc[i])
        elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
            obv.append(obv[-1] - data['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    data['OBV'] = pd.Series(obv, index=data.index)
    return data

def calculate_atr(data, period=14):
    data = data.copy()
    high_low = data['High'] - data['Low']
    high_close = abs(data['High'] - data['Close'].shift(1))
    low_close = abs(data['Low'] - data['Close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['ATR'] = tr.rolling(window=period).mean()
    return data

def generate_signals(data, current_price, sentiment=0):
    rsi = calculate_rsi(data)
    data = calculate_bollinger_bands(data)
    data = calculate_ema(data)
    macd, signal_line = calculate_macd(data)
    k, d = calculate_stochastic(data)
    data = calculate_vwap(data)
    data = calculate_obv(data)
    data = calculate_atr(data)

    current_rsi = rsi.iloc[-1]
    upper_band = data['Upper'].iloc[-1]
    lower_band = data['Lower'].iloc[-1]
    ma20 = data['MA20'].iloc[-1]
    ema50 = data['EMA50'].iloc[-1]
    current_k = k.iloc[-1]
    current_d = d.iloc[-1]
    current_vwap = data['VWAP'].iloc[-1]
    current_obv = data['OBV'].iloc[-1]
    prev_obv = data['OBV'].iloc[-2] if len(data) > 2 else current_obv
    current_atr = data['ATR'].iloc[-1]
    current_macd = macd.iloc[-1]
    current_signal = signal_line.iloc[-1]

    entry_range_buy = None
    exit_range_sell = None
    price_increase_forecast = None
    signal_type = None
    stop_loss = None
    profit_take = None
    confidence_score = 0

    # Einfluss des Sentiments auf die Signale
    sentiment_factor = max(-1, min(1, sentiment))  # Begrenze auf -1 bis 1
    long_signals = 0
    short_signals = 0

    # Long-Signale mit Sentiment-Anpassung
    if current_rsi < 45 and current_price < lower_band:
        long_signals += 1 + (sentiment_factor * 0.5)  # Positives Sentiment verstärkt Long-Signale
    if current_k < 45 and current_d < 45:
        long_signals += 1 + (sentiment_factor * 0.5)
    if current_price < ema50 and current_price < current_vwap:
        long_signals += 1 + (sentiment_factor * 0.5)
    if current_obv > prev_obv:
        long_signals += 1 + (sentiment_factor * 0.5)
    if current_macd > current_signal and macd.iloc[-2] <= signal_line.iloc[-2]:
        long_signals += 1 + (sentiment_factor * 0.5)

    # Short-Signale mit Sentiment-Anpassung
    if current_rsi > 55 and current_price > upper_band:
        short_signals += 1 - (sentiment_factor * 0.5)  # Negatives Sentiment verstärkt Short-Signale
    if current_k > 55 and current_d > 55:
        short_signals += 1 - (sentiment_factor * 0.5)
    if current_price > ema50 and current_price > current_vwap:
        short_signals += 1 - (sentiment_factor * 0.5)
    if current_obv < prev_obv:
        short_signals += 1 - (sentiment_factor * 0.5)
    if current_macd < current_signal and macd.iloc[-2] >= signal_line.iloc[-2]:
        short_signals += 1 - (sentiment_factor * 0.5)

    # Entscheidung basierend auf angepassten Signalen
    base_confidence = 20  # Basiskonfidenz pro Signal
    if long_signals >= 3:
        base_score = long_signals * base_confidence
        # Anpassung der Confidence basierend auf Sentiment
        confidence_adjustment = sentiment_factor * 15  # Max. 15% Anpassung (positives Sentiment erhöht, negatives senkt)
        confidence_score = min(100, max(0, base_score + confidence_adjustment))
        entry_range_buy = f"{max(current_price * 0.98, lower_band * 0.98):.2f}–{lower_band:.2f}"
        exit_range_sell = f"{ema50 * 1.10:.2f}–{upper_band * 1.05:.2f}"
        price_increase_forecast = f"{((ema50 * 1.10 / current_price) - 1) * 100:.1f}%"
        stop_loss = max(current_price - (1.5 * current_atr), lower_band * 0.95)
        profit_take = ema50 * 1.15
        signal_type = "Long"
        recommendation = (f"Kaufsignal (Long) im Bereich {entry_range_buy}, Zielzone (Verkauf): {exit_range_sell}, "
                         f"voraussichtlicher Kursanstieg: {price_increase_forecast}, "
                         f"Stop-Loss: {stop_loss:.2f}, Profit-Take: {profit_take:.2f}, Konfidenz: {confidence_score}%")
    elif short_signals >= 3:
        base_score = short_signals * base_confidence
        # Anpassung der Confidence basierend auf Sentiment (negatives Sentiment erhöht, positives senkt)
        confidence_adjustment = -sentiment_factor * 15  # Max. 15% Anpassung (negatives Sentiment erhöht, positives senkt)
        confidence_score = min(100, max(0, base_score + confidence_adjustment))
        entry_range_buy = f"{upper_band:.2f}–{min(current_price * 1.02, upper_band * 1.02):.2f}"
        exit_range_sell = f"{ema50 * 0.90:.2f}–{lower_band * 0.95:.2f}"
        price_increase_forecast = f"{((current_price / (ema50 * 0.90)) - 1) * 100:.1f}%"
        stop_loss = min(current_price + (1.5 * current_atr), upper_band * 1.05)
        profit_take = ema50 * 0.85
        signal_type = "Short"
        recommendation = (f"Verkaufssignal (Short) im Bereich {entry_range_buy}, Zielzone (Rückkauf): {exit_range_sell}, "
                         f"voraussichtlicher Kursrückgang: {price_increase_forecast}, "
                         f"Stop-Loss: {stop_loss:.2f}, Profit-Take: {profit_take:.2f}, Konfidenz: {confidence_score}%")
    else:
        recommendation = "Keine Aktion, keine klare Signale von Indikatoren (weniger als 3 Signale für Long oder Short)"
        signal_type = None
        confidence_score = 0

    return recommendation, signal_type, entry_range_buy, exit_range_sell, price_increase_forecast, confidence_score, stop_loss, profit_take

def backtest_signals(data, ticker, sentiment=0):
    signals = []
    for i in range(20, len(data)):
        temp_data = data.iloc[:i]
        current_price = temp_data['Close'].iloc[-1]
        rec, _, _, _, _, _, stop_loss, profit_take = generate_signals(temp_data, current_price, sentiment)
        if "Kaufsignal" in rec:
            signals.append(('Buy', i, current_price, stop_loss, profit_take))
        elif "Verkaufssignal" in rec:
            signals.append(('Sell', i, current_price, stop_loss, profit_take))
    
    long_returns = 0
    short_returns = 0
    long_trades = 0
    short_trades = 0
    trade_details = []
    
    for i, (signal_type, idx, price, sl, pt) in enumerate(signals):
        if signal_type == 'Buy':
            exit_price = None
            max_holding_days = 30
            for j in range(i + 1, min(i + max_holding_days + 1, len(signals))):
                if signals[j][0] == 'Sell':
                    exit_price = signals[j][2]
                    break
            if exit_price is None:
                future_closes = data['Close'].iloc[idx + 1:idx + max_holding_days + 1].dropna()
                if not future_closes.empty:
                    max_price = future_closes.max()
                    exit_price = min(max(max_price, sl) if sl else max_price, pt) if pt else max_price
                    if exit_price > price and sl:
                        exit_price = min(exit_price, sl)
            if exit_price is not None and exit_price > price:
                trade_return = ((exit_price - price) / price) * 100
                long_returns += trade_return
                long_trades += 1
                trade_details.append(f"Long-Trade: Einstieg {price:.2f}, Ausstieg {exit_price:.2f}, Rendite {trade_return:.2f}%")
        elif signal_type == 'Sell':
            exit_price = None
            max_holding_days = 30
            for j in range(i + 1, min(i + max_holding_days + 1, len(signals))):
                if signals[j][0] == 'Buy':
                    exit_price = signals[j][2]
                    break
            if exit_price is None:
                future_closes = data['Close'].iloc[idx + 1:idx + max_holding_days + 1].dropna()
                if not future_closes.empty:
                    min_price = future_closes.min()
                    exit_price = max(min(min_price, sl) if sl else min_price, pt) if pt else min_price
                    if exit_price < price and sl:
                        exit_price = max(exit_price, sl)
            if exit_price is not None and exit_price < price:
                trade_return = ((price - exit_price) / price) * 100
                short_returns += trade_return
                short_trades += 1
                trade_details.append(f"Short-Trade: Einstieg {price:.2f}, Ausstieg {exit_price:.2f}, Rendite {trade_return:.2f}%")
    
    total_returns = long_returns + short_returns
    total_trades = long_trades + short_trades
    
    if trade_details:
        with st.expander("Trade-Details"):
            for detail in trade_details:
                st.write(detail)
    
    return (f"Backtest-Rendite für {ticker} (Summen-Rendite, realistisch simuliert): "
            f"Long: {long_returns:.2f}% über {long_trades} Signale, "
            f"Short: {short_returns:.2f}% über {short_trades} Signale, "
            f"Gesamt: {total_returns:.2f}% über {total_trades} Signale") if total_trades > 0 else f"Backtest-Rendite für {ticker}: 0.00% über {len(signals)} Signale"

def calculate_correlation(data, ticker):
    sp500 = yf.Ticker("^GSPC").history(start=data.index[0], end=data.index[-1])['Close']
    correlation = data['Close'].corr(sp500)
    volatility = data['Close'].pct_change().std() * np.sqrt(252) * 100
    return f"Korrelation mit S&P 500: {correlation:.2f}, Annualisierte Volatilität: {volatility:.2f}%"

def plot_indicator_heatmap(data):
    indicators = pd.DataFrame({
        'RSI': (calculate_rsi(data) < 45).astype(int) * 2 + (calculate_rsi(data) > 55).astype(int) * -1,
        'Bollinger': (data['Close'] < data['Lower']).astype(int) * 2 + (data['Close'] > data['Upper']).astype(int) * -1,
        'Stochastic': (calculate_stochastic(data)[0] < 45).astype(int) * 2 + (calculate_stochastic(data)[0] > 55).astype(int) * -1,
        'VWAP': (data['Close'] < data['VWAP']).astype(int) * 2 + (data['Close'] > data['VWAP']).astype(int) * -1,
        'OBV': (calculate_obv(data).diff() > 0).astype(int) * 2 + (calculate_obv(data).diff() < 0).astype(int) * -1,
        'MACD': (calculate_macd(data)[0] < calculate_macd(data)[1]).astype(int) * 2 + (calculate_macd(data)[0] > calculate_macd(data)[1]).astype(int) * -1,
        'EMA50': (data['Close'] < data['EMA50']).astype(int) * 2 + (data['Close'] > data['EMA50']).astype(int) * -1
    })
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(indicators.corr(), annot=True, cmap='coolwarm', center=0, fmt='.2f', ax=ax)
    plt.title("**Optimierte Indikator-Korrelations-Heatmap**", fontweight='bold', fontsize=16)
    return fig

# --- Elliot-Wellen-Funktionen ---
def generate_chart_image(df):
    if df.empty:
        return None  # Rückgabe None, wenn DataFrame leer ist
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["Close"], label="Close", color='blue')
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", facecolor='#1a1a1a', edgecolor='none')  # Dunkelgrauer Hintergrund
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    return img

def detect_waves_in_image(img):
    if img is None or isinstance(img, np.ndarray) and (img.size == 0 or img.shape[0] == 1):
        return np.array([], dtype=int)
    # Konvertiere das Bild in ein Array, falls es ein PIL-Bild ist
    if isinstance(img, Image.Image):
        img_array = np.array(img.convert('L'))  # Graustufenkonvertierung
    else:
        img_array = np.array(img)
    # Führe eine einfache Peaks-Erkennung durch (ähnlich wie mit Canny-Edges, aber ohne cv2)
    edges = np.gradient(np.mean(img_array, axis=0))
    peaks = argrelextrema(edges, np.greater, order=10)[0]
    return peaks

def identify_elliot_waves(df, sentiment=0):
    if df.empty:
        return [], 0.5, []
    
    highs = argrelextrema(df["High"].values, np.greater, order=5)[0]
    lows = argrelextrema(df["Low"].values, np.less, order=5)[0]
    
    waves = []
    for i in range(min(len(highs), len(lows)) - 4):
        wave_pattern = {
            "W1": df["Close"].iloc[lows[i]],
            "W2": df["Close"].iloc[highs[i]],
            "W3": df["Close"].iloc[lows[i+1]],
            "W4": df["Close"].iloc[highs[i+1]],
            "W5": df["Close"].iloc[lows[i+2]]
        }
        waves.append(wave_pattern)
    
    features = pd.DataFrame({
        "rsi": calculate_rsi(df),
        "macd": calculate_macd(df)[0],
        "volatility": df["Close"].pct_change().rolling(14).std(),
        "sentiment": sentiment  # Füge Sentiment als Feature hinzu
    }).dropna()
    
    valid_indices = features.index[features.index <= df.index[-1]] if not df.index.empty else features.index[:1] if not features.empty else pd.Index([0])
    if len(valid_indices) == 0 and not features.empty:
        valid_indices = features.index[:1]
    
    shifted_close = df["Close"].shift(-5).dropna()
    current_close = df["Close"].dropna()
    common_indices = shifted_close.index.intersection(current_close.index).intersection(valid_indices)
    
    if len(common_indices) > 0:
        labels = (shifted_close.loc[common_indices] > current_close.loc[common_indices]).astype(int)
    else:
        labels = pd.Series([0], index=[0])
    
    features = features.loc[common_indices]
    
    if not features.empty and not labels.empty:
        rf = RandomForestClassifier(n_estimators=100, max_depth=10)
        rf.fit(features, labels)
        wave_confidence = rf.predict_proba(features)[-1][1] if len(features) > 0 else 0.5
        # Anpassung des Vertrauens basierend auf Sentiment
        if sentiment > 0:
            wave_confidence = min(1.0, wave_confidence + (sentiment * 0.2))  # Positives Sentiment erhöht Vertrauen
        elif sentiment < 0:
            wave_confidence = max(0.0, wave_confidence + (sentiment * 0.2))  # Negatives Sentiment senkt Vertrauen
    else:
        wave_confidence = 0.5
    
    chart_img = generate_chart_image(df)
    waves_ki = detect_waves_in_image(chart_img)
    
    return waves, wave_confidence, waves_ki

def get_sentiment(symbol, manual_sentiment=None):
    if manual_sentiment is not None:
        return manual_sentiment  # Verwende den manuellen Sentiment-Wert, wenn angegeben
    news = [
        f"{symbol} reports strong earnings this quarter.",
        f"Analysts predict a bullish trend for {symbol}.",
        f"Market uncertainty affects {symbol} negatively."
    ]
    sentiment = np.mean([TextBlob(text).sentiment.polarity for text in news])
    return sentiment

def calculate_sma(df, period=50):
    if df.empty:
        return pd.Series([], dtype=float)
    return df["Close"].rolling(window=period).mean()

def calculate_fibonacci_levels(df):
    if df.empty:
        return [], []
    high = df["High"].max()
    low = df["Low"].min()
    levels = [high - (high - low) * level for level in [0, 0.236, 0.382, 0.5, 0.618, 1]]
    zones = [
        {"start": levels[4], "end": levels[3], "type": "Buy"},
        {"start": levels[1], "end": levels[0], "type": "Sell"}
    ]
    return levels, zones

def generate_signals_elliot(df, sentiment=0):
    if df.empty:
        return []
    sma = calculate_sma(df)
    rsi = calculate_rsi(df)
    signals = []
    for i in range(len(df)):
        signal = "Hold"
        # Einfluss des Sentiments auf die Signale
        sentiment_factor = max(-1, min(1, sentiment))  # Begrenze auf -1 bis 1
        if df["Close"].iloc[i] > sma.iloc[i] and rsi.iloc[i] > 70 + (sentiment_factor * 10):  # Anpassung der RSI-Schwelle
            signal = "Sell"
        elif df["Close"].iloc[i] < sma.iloc[i] and rsi.iloc[i] < 30 - (sentiment_factor * 10):  # Anpassung der RSI-Schwelle
            signal = "Buy"
        signals.append(signal)
    return signals

def robustness_check(df, waves):
    if df.empty or len(waves) == 0:
        return False
    z_scores = np.abs((df["Close"] - df["Close"].mean()) / df["Close"].std())
    return not (z_scores > 3).any()

def analyze_wave_details(df, waves, signals_elliot, price_increase_forecast, is_short_term=False, sentiment=0):
    if not waves or not signals_elliot or df.empty or not price_increase_forecast:
        return None
    
    # Finde das letzte Verkauf-/Kaufsignal
    latest_signal_idx = -1
    for i in range(len(signals_elliot) - 1, -1, -1):
        if signals_elliot[i] in ["Buy", "Sell"]:
            latest_signal_idx = i
            break
    
    if latest_signal_idx == -1:
        return None
    
    latest_signal = signals_elliot[latest_signal_idx]
    
    # Signaldatum mit unterschiedlichem Format je nach Short-Term/Long-Term
    if is_short_term:
        signal_date = df.index[latest_signal_idx].strftime("%Y-%m-%d %H:%M:%S")
    else:
        signal_date = df.index[latest_signal_idx].strftime("%Y-%m-%d")
    
    # Erwartete Zeit bis das Signal "losgeht" (nächster Handelstag)
    signal_start = (df.index[latest_signal_idx] + timedelta(days=1)).replace(hour=9, minute=30, second=0).strftime("%Y-%m-%d %H:%M:%S")
    if not df.index[-1].time() < datetime.strptime("16:00", "%H:%M").time():  # Wenn Markt offen ist, am nächsten Tag starten
        signal_start = (df.index[latest_signal_idx] + timedelta(days=1)).replace(hour=9, minute=30, second=0).strftime("%Y-%m-%d %H:%M:%S")
    
    # Historische Wellen analysieren für Dauer
    wave_durations = []
    for wave in waves:
        wave_values = list(wave.values())
        wave_start = df.index[df["Close"] == wave_values[0]][0] if not df[df["Close"] == wave_values[0]].empty else df.index[0]
        wave_end = df.index[df["Close"] == wave_values[-1]][0] if not df[df["Close"] == wave_values[-1]].empty else df.index[-1]
        duration = (wave_end - wave_start).total_seconds() / 60 if is_short_term else (wave_end - wave_start).days  # Umrechnung in Minuten für Short-Term (1m), Tage für Long-Term
        if duration > 0:
            wave_durations.append(duration)
    
    avg_wave_duration = int(np.mean(wave_durations)) if wave_durations else (390 if is_short_term else 30)  # Fallback: 390 Minuten (6.5 Stunden * 60 Minuten für Handelszeiten 9:30–16:00) für Short-Term, 30 Tage für Long-Term
    duration_unit = "Minuten" if is_short_term else "Tage"
    
    expected_return = price_increase_forecast if latest_signal == "Buy" else f"-{price_increase_forecast}"
    
    # Anpassung des erwarteten Returns basierend auf Sentiment
    if sentiment > 0 and latest_signal == "Buy":
        expected_return = f"{float(expected_return.strip('%')) * (1 + sentiment * 0.1):.1f}%"
    elif sentiment < 0 and latest_signal == "Sell":
        expected_return = f"{float(expected_return.strip('%')) * (1 + abs(sentiment) * 0.1):.1f}%"
    
    return {
        "signal_type": latest_signal,
        "signal_date": signal_date,
        "signal_start": signal_start,
        "expected_return": expected_return,
        "duration_to_target": avg_wave_duration,
        "duration_unit": duration_unit
    }

def analyze_stock(ticker, years=5, show_standard=False, show_longterm=False, show_longterm_signals=False, show_longterm_stats=False, show_shortterm=False, show_shortterm_signals=False, show_shortterm_stats=False, manual_sentiment=None):
    current_price, price_display = get_live_price(ticker)
    if current_price is None:
        st.error(price_display)
        return
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    
    data = data.dropna()
    if len(data) < 50:
        st.error(f"Unzureichende Daten für {ticker}. Mindestens 50 Datenpunkte erforderlich.")
        return
    
    # Aktualisiere Daten mit Live-Preis
    if not data.empty:
        data.loc[data.index[-1], 'Close'] = current_price
        data.loc[data.index[-1], 'Open'] = current_price
        data.loc[data.index[-1], 'High'] = current_price
        data.loc[data.index[-1], 'Low'] = current_price
    
    rsi = calculate_rsi(data)
    data = calculate_bollinger_bands(data)
    data = calculate_ema(data)
    macd, signal_line = calculate_macd(data)
    k, d = calculate_stochastic(data)
    data = calculate_vwap(data)
    data = calculate_obv(data)
    data = calculate_atr(data)
    
    sentiment = get_sentiment(ticker, manual_sentiment)
    recommendation, signal_type, entry_range_buy, exit_range_sell, price_increase_forecast, confidence_score, stop_loss, profit_take = generate_signals(data, current_price, sentiment)
    backtest_result = backtest_signals(data, ticker, sentiment)
    
    # Elliot-Wellen-Daten (Longterm: 1d, 1y)
    elliot_df_longterm = fetch_data(ticker, timeframe="1d", period="1y")
    if not elliot_df_longterm.empty:
        elliot_df_longterm.loc[elliot_df_longterm.index[-1], 'Close'] = current_price
        elliot_df_longterm.loc[elliot_df_longterm.index[-1], 'Open'] = current_price
        elliot_df_longterm.loc[elliot_df_longterm.index[-1], 'High'] = current_price
        elliot_df_longterm.loc[elliot_df_longterm.index[-1], 'Low'] = current_price
    waves_longterm, wave_confidence_longterm, waves_ki_longterm = identify_elliot_waves(elliot_df_longterm, sentiment) if not elliot_df_longterm.empty else ([], 0.5, [])
    sma_elliot_longterm = calculate_sma(elliot_df_longterm)
    fib_levels_longterm, fib_zones_longterm = calculate_fibonacci_levels(elliot_df_longterm)
    signals_elliot_longterm = generate_signals_elliot(elliot_df_longterm, sentiment)
    robust_longterm = robustness_check(elliot_df_longterm, waves_longterm)
    wave_details_longterm = analyze_wave_details(elliot_df_longterm, waves_longterm, signals_elliot_longterm, price_increase_forecast, is_short_term=False, sentiment=sentiment)
    
    # Elliot-Wellen-Daten (Shortterm: 1m, 1d für aktuellen Tag und Handelszeiten, nur reguläre Handelszeiten)
    elliot_df_shortterm = fetch_data(ticker, timeframe="1m", period="1d")
    if not elliot_df_shortterm.empty:
        # Erstelle einen kontinuierlichen Datensatz für 1-Minuten-Intervalle, nur für den aktuellen Tag und Handelszeiten
        elliot_df_shortterm = create_continuous_1m_data(elliot_df_shortterm, period_days=1)
        # Bereinige Daten auf reguläre Handelszeiten (9:30–16:00 EST), nur aktueller Tag
        elliot_df_shortterm = clean_trading_data(elliot_df_shortterm)
        # Prüfe, ob die Spalten vorhanden sind, und füge Default-Werte hinzu, falls nicht
        required_columns = ["Open", "High", "Low", "Close"]
        for col in required_columns:
            if col not in elliot_df_shortterm.columns:
                elliot_df_shortterm[col] = elliot_df_shortterm["Close"].fillna(0)
        
        # Aktualisiere Daten mit Live-Preis (nur reguläre Handelszeiten)
        if not elliot_df_shortterm.empty and len(elliot_df_shortterm) > 0:
            elliot_df_shortterm.loc[elliot_df_shortterm.index[-1], 'Close'] = current_price
            elliot_df_shortterm.loc[elliot_df_shortterm.index[-1], 'Open'] = current_price
            elliot_df_shortterm.loc[elliot_df_shortterm.index[-1], 'High'] = current_price
            elliot_df_shortterm.loc[elliot_df_shortterm.index[-1], 'Low'] = current_price
        
        waves_shortterm, wave_confidence_shortterm, waves_ki_shortterm = identify_elliot_waves(elliot_df_shortterm, sentiment) if not elliot_df_shortterm.empty else ([], 0.5, [])
        sma_elliot_shortterm = calculate_sma(elliot_df_shortterm)
        fib_levels_shortterm, fib_zones_shortterm = calculate_fibonacci_levels(elliot_df_shortterm)
        signals_elliot_shortterm = generate_signals_elliot(elliot_df_shortterm, sentiment)
        robust_shortterm = robustness_check(elliot_df_shortterm, waves_shortterm)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("Marktdaten")
        st.write(f"**Aktueller Preis**: {price_display}")
        st.write(f"**RSI**: {rsi.iloc[-1]:.2f}")
        st.write(f"**Bollinger Bands** - Oberes Band: {data['Upper'].iloc[-1]:.2f}, Unteres Band: {data['Lower'].iloc[-1]:.2f}")
        st.write(f"**Stochastic %K**: {k.iloc[-1]:.2f}, %D: {d.iloc[-1]:.2f}")
        st.write(f"**EMA50**: {data['EMA50'].iloc[-1]:.2f}")
        st.write(f"**VWAP**: {data['VWAP'].iloc[-1]:.2f}")
        st.write(f"**OBV**: {data['OBV'].iloc[-1]:.2f}")
        st.write(f"**ATR (Volatilität)**: {data['ATR'].iloc[-1]:.2f}")
        st.write(f"**MACD**: {macd.iloc[-1]:.2f}, Signal Line: {signal_line.iloc[-1]:.2f}")
    
    with col2:
        st.header("Handelsempfehlung")
        st.write(f"**Empfehlung**: {recommendation}")
        if entry_range_buy:
            st.write(f"**Einstiegsbereich**: {entry_range_buy}")
        if exit_range_sell:
            st.write(f"**Verkaufsbereich (oder Rückkauf bei Short)**: {exit_range_sell}")
        if price_increase_forecast:
            st.write(f"**Prognostizierter Kursanstieg/Rückgang**: {price_increase_forecast}")
        if stop_loss:
            st.write(f"**Stop-Loss (Risikomanagement)**: {stop_loss:.2f}")
        if profit_take:
            st.write(f"**Profit-Take (Ziel)**: {profit_take:.2f}")
        st.write(f"**Konfidenz des Signals**: {confidence_score}%")
    
    st.header("Zusammenfassung und Risikoanalyse")
    st.write(f"{calculate_correlation(data, ticker)}")
    st.write(f"{backtest_result}")
    
    if show_standard:
        st.header("Standarddiagramm")
        fig, axes = plt.subplots(5, 1, figsize=(15, 20), facecolor='#1a1a1a')
        plt.style.use('dark_background')
        
        axes[0].plot(data.index, data['Close'], label='Kurs', color='blue', linewidth=1.5)
        axes[0].plot(data.index, data['MA20'], label='MA20', color='orange', linewidth=1.5)
        axes[0].plot(data.index, data['Upper'], label='Oberes Band', color='red', linestyle='--', linewidth=1)
        axes[0].plot(data.index, data['Lower'], label='Unteres Band', color='green', linestyle='--', linewidth=1)
        axes[0].plot(data.index, data['EMA50'], label='EMA50', color='cyan', linestyle='-.', linewidth=1.5)
        axes[0].plot(data.index, data['VWAP'], label='VWAP', color='purple', linestyle='-.', linewidth=1.5)
        if signal_type == "Long":
            axes[0].scatter(data.index[-1], current_price, color='green', marker='^', s=200, label='Kauf-Signal')
        elif signal_type == "Short":
            axes[0].scatter(data.index[-1], current_price, color='red', marker='v', s=200, label='Verkauf-Signal')
        axes[0].set_title(f"**{ticker} - Optimierte Marktanalyse (Kurs & Indikatoren)**", fontweight='bold', fontsize=18, color='white')
        axes[0].set_xlabel("Datum", fontweight='bold', fontsize=12, color='white')
        axes[0].set_ylabel("Preis (USD)", fontweight='bold', fontsize=12, color='white')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., facecolor='#2c2c2c', edgecolor='white', labelcolor='white')
        axes[0].grid(True, linestyle='--', alpha=0.3, color='gray')
        axes[0].set_facecolor('#1a1a1a')
        
        axes[1].plot(data.index, rsi, label='RSI', color='purple', linewidth=1.5)
        axes[1].axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Neutral')
        axes[1].axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Überverkauft (Long)')
        axes[1].axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Überkauft (Short)')
        axes[1].set_title(f"**Relative Strength Index (RSI)**", fontweight='bold', fontsize=16, color='white')
        axes[1].set_xlabel("Datum", fontweight='bold', fontsize=12, color='white')
        axes[1].set_ylabel("RSI-Wert", fontweight='bold', fontsize=12, color='white')
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., facecolor='#2c2c2c', edgecolor='white', labelcolor='white')
        axes[1].grid(True, linestyle='--', alpha=0.3, color='gray')
        axes[1].set_facecolor('#1a1a1a')
        
        axes[2].plot(data.index, k, label='%K', color='red', alpha=0.7, linewidth=1.5)
        axes[2].plot(data.index, d, label='%D', color='green', alpha=0.7, linewidth=1.5)
        axes[2].axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Neutral')
        axes[2].axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Überverkauft (Long)')
        axes[2].axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Überkauft (Short)')
        axes[2].set_title(f"**Stochastic Oscillator**", fontweight='bold', fontsize=16, color='white')
        axes[2].set_xlabel("Datum", fontweight='bold', fontsize=12, color='white')
        axes[2].set_ylabel("Stochastic-Wert", fontweight='bold', fontsize=12, color='white')
        axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., facecolor='#2c2c2c', edgecolor='white', labelcolor='white')
        axes[2].grid(True, linestyle='--', alpha=0.3, color='gray')
        axes[2].set_facecolor('#1a1a1a')
        
        axes[3].plot(data.index, macd, label='MACD', color='blue', linewidth=1.5)
        axes[3].plot(data.index, signal_line, label='Signal Line', color='orange', linewidth=1.5)
        axes[3].bar(data.index, macd - signal_line, label='MACD Histogram', color='gray', alpha=0.5, width=0.8)
        axes[3].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        axes[3].set_title(f"**Moving Average Convergence Divergence (MACD)**", fontweight='bold', fontsize=16, color='white')
        axes[3].set_xlabel("Datum", fontweight='bold', fontsize=12, color='white')
        axes[3].set_ylabel("MACD-Wert", fontweight='bold', fontsize=12, color='white')
        axes[3].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., facecolor='#2c2c2c', edgecolor='white', labelcolor='white')
        axes[3].grid(True, linestyle='--', alpha=0.3, color='gray')
        axes[3].set_facecolor('#1a1a1a')
        
        axes[4].plot(data.index, data['OBV'], label='OBV', color='purple', linewidth=1.5)
        axes[4].plot(data.index, data['VWAP'], label='VWAP', color='blue', linestyle='-.', linewidth=1.5)
        axes[4].set_title(f"**On-Balance Volume (OBV) & Volume Weighted Average Price (VWAP)**", fontweight='bold', fontsize=16, color='white')
        axes[4].set_xlabel("Datum", fontweight='bold', fontsize=12, color='white')
        axes[4].set_ylabel("Wert", fontweight='bold', fontsize=12, color='white')
        axes[4].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., facecolor='#2c2c2c', edgecolor='white', labelcolor='white')
        axes[4].grid(True, linestyle='--', alpha=0.3, color='gray')
        axes[4].set_facecolor('#1a1a1a')
        
        plt.tight_layout()
        st.pyplot(fig)

    if st.checkbox("Indikator-Korrelations-Heatmap anzeigen", key="heatmap_checkbox"):
        heatmap_fig = plot_indicator_heatmap(data)
        st.pyplot(heatmap_fig)
    
    if show_longterm:
        st.header("Elliot Wave (Long-Term) Analyse")
        if st.checkbox("Legende ein-/ausblenden", value=True, key="legend_toggle_longterm"):
            legend_html = f"""
            <div style="background-color: {colors['card']}; padding: 10px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);">
                <ul style="list-style-type: none; padding: 0; margin: 0; font-size: 12px; color: {colors['text']};">
                    <li><span style="color: {colors['positive']};">●</span> Buy Signal / Zone</li>
                    <li><span style="color: {colors['negative']};">●</span> Sell Signal / Zone</li>
                    <li><span style="color: {colors['neutral']};">●</span> SMA</li>
                    <li><span style="color: {colors['positive']};">-</span> Elliot Wave</li>
                </ul>
            </div>
            """
            st.markdown(legend_html, unsafe_allow_html=True)
        
        fig_elliot = go.Figure()
        fig_elliot.add_trace(go.Candlestick(x=elliot_df_longterm.index, open=elliot_df_longterm["Open"], high=elliot_df_longterm["High"], low=elliot_df_longterm["Low"], close=elliot_df_longterm["Close"], name="Preis"))
        
        for i, wave in enumerate(waves_longterm):
            fig_elliot.add_trace(go.Scatter(x=[elliot_df_longterm.index[list(wave.values()).index(v)] for v in wave.values()], 
                                           y=list(wave.values()), mode="lines+markers", name=f"Wave {i+1}", line=dict(color=colors['positive'])))
        
        for i in range(len(elliot_df_longterm)):
            if signals_elliot_longterm[i] == "Buy":
                fig_elliot.add_trace(go.Scatter(x=[elliot_df_longterm.index[i]], y=[elliot_df_longterm["Close"].iloc[i]], mode="markers", 
                                               marker=dict(symbol="triangle-up", size=12, color=colors['positive']), name="Buy Signal"))
            elif signals_elliot_longterm[i] == "Sell":
                fig_elliot.add_trace(go.Scatter(x=[elliot_df_longterm.index[i]], y=[elliot_df_longterm["Close"].iloc[i]], mode="markers", 
                                               marker=dict(symbol="triangle-down", size=12, color=colors['negative']), name="Sell Signal"))
        
        for zone in fib_zones_longterm:
            fig_elliot.add_trace(go.Scatter(x=[elliot_df_longterm.index[0], elliot_df_longterm.index[-1], elliot_df_longterm.index[-1], elliot_df_longterm.index[0], elliot_df_longterm.index[0]], 
                                           y=[zone["start"], zone["start"], zone["end"], zone["end"], zone["start"]], 
                                           fill="toself", fillcolor=colors['positive'] if zone["type"] == "Buy" else colors['negative'], 
                                           opacity=0.2, line=dict(color="rgba(255,255,255,0)"), 
                                           showlegend=False, name=f"{zone['type']} Zone"))
        
        fig_elliot.add_trace(go.Scatter(x=elliot_df_longterm.index, y=sma_elliot_longterm, name="SMA 50", line=dict(color=colors['neutral'])))
        
        fig_elliot.update_layout(
            height=800,
            showlegend=False,
            title_text=f"Elliot Wave (Long-Term) Analyse für {ticker} (1 Jahr)",
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font_color=colors['text'],
            margin=dict(t=80, b=50, l=50, r=50)
        )
        st.plotly_chart(fig_elliot, use_container_width=True)
        
        if wave_details_longterm:
            st.subheader("Details zum aktuellen Elliot-Wellen-Signal (Long-Term)")
            st.write(f"**Signal-Typ**: {wave_details_longterm['signal_type']}")
            st.write(f"**Signaldatum**: {wave_details_longterm['signal_date']}")
            st.write(f"**Geschätzte Dauer bis der prognostizierte Kursgewinn erreicht wird**: {wave_details_longterm['duration_to_target']} {wave_details_longterm['duration_unit']}")
            st.write(f"**Erwarteter Kursgewinn bzw. Kursverlust in %**: {wave_details_longterm['expected_return']}")
        
        if show_longterm_signals:
            st.header("Elliot Wave (Long-Term) Signale & Zonen")
            st.markdown(f'<div class="card">', unsafe_allow_html=True)
            st.write(f"**Aktuelles Signal**: {signals_elliot_longterm[-1] if signals_elliot_longterm else 'Hold'}")
            for zone in fib_zones_longterm:
                zone_color = colors['positive'] if zone["type"] == "Buy" else colors['negative']
                st.markdown(f'<span style="color: {zone_color}">**{zone["type"]} Zone:** {zone["start"]:.2f} - {zone["end"]:.2f}</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        if show_longterm_stats:
            st.header("Elliot Wave (Long-Term) Marktstatistiken")
            st.markdown(f'<div class="card">', unsafe_allow_html=True)
            st.write(f"**Wave Confidence:** {wave_confidence_longterm:.2%}")
            sentiment_text = f"**Sentiment:** {sentiment:.2f} {'(Bullish)' if sentiment > 0 else '(Bearish)' if sentiment < 0 else '(Neutral)'}"
            st.markdown(f'<span style="color: {colors["positive"] if sentiment > 0 else colors["negative"] if sentiment < 0 else colors["neutral"]}">{sentiment_text}</span>', unsafe_allow_html=True)
            st.write(f"**Robust:** {'✅' if robust_longterm else '❌'}")
            st.markdown('</div>', unsafe_allow_html=True)
    
    if show_shortterm:
        st.header("Elliot Wave (Short-Term) Analyse")
        if st.checkbox("Legende ein-/ausblenden", value=True, key="legend_toggle_shortterm"):
            legend_html = f"""
            <div style="background-color: {colors['card']}; padding: 10px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);">
                <ul style="list-style-type: none; padding: 0; margin: 0; font-size: 12px; color: {colors['text']};">
                    <li><span style="color: {colors['positive']};">●</span> Buy Signal / Zone</li>
                    <li><span style="color: {colors['negative']};">●</span> Sell Signal / Zone</li>
                    <li><span style="color: {colors['neutral']};">●</span> SMA</li>
                    <li><span style="color: {colors['positive']};">-</span> Elliot Wave</li>
                </ul>
            </div>
            """
            st.markdown(legend_html, unsafe_allow_html=True)
        
        if not elliot_df_shortterm.empty:
            fig_elliot = go.Figure()
            fig_elliot.add_trace(go.Candlestick(x=elliot_df_shortterm.index, open=elliot_df_shortterm["Open"], high=elliot_df_shortterm["High"], low=elliot_df_shortterm["Low"], close=elliot_df_shortterm["Close"], name="Preis", increasing_line_color=colors['positive'], decreasing_line_color=colors['negative'], line_width=1, opacity=1))
            
            for i, wave in enumerate(waves_shortterm):
                fig_elliot.add_trace(go.Scatter(x=[elliot_df_shortterm.index[list(wave.values()).index(v)] for v in wave.values()], 
                                               y=list(wave.values()), mode="lines", name=f"Wave {i+1}", line=dict(color=colors['positive'], width=1.5)))
            
            for i in range(len(elliot_df_shortterm)):
                if signals_elliot_shortterm[i] == "Buy":
                    fig_elliot.add_trace(go.Scatter(x=[elliot_df_shortterm.index[i]], y=[elliot_df_shortterm["Close"].iloc[i]], mode="markers", 
                                                   marker=dict(symbol="triangle-up", size=8, color=colors['positive']), name="Buy Signal"))
                elif signals_elliot_shortterm[i] == "Sell":
                    fig_elliot.add_trace(go.Scatter(x=[elliot_df_shortterm.index[i]], y=[elliot_df_shortterm["Close"].iloc[i]], mode="markers", 
                                                   marker=dict(symbol="triangle-down", size=8, color=colors['negative']), name="Sell Signal"))
            
            for zone in fib_zones_shortterm:
                zone_color = 'rgba(0, 255, 0, 0.3)' if zone["type"] == "Buy" else 'rgba(165, 42, 42, 0.3)'  # Grün für Buy, Braun für Sell
                fig_elliot.add_trace(go.Scatter(x=[elliot_df_shortterm.index[0], elliot_df_shortterm.index[-1], elliot_df_shortterm.index[-1], elliot_df_shortterm.index[0], elliot_df_shortterm.index[0]], 
                                               y=[zone["start"], zone["start"], zone["end"], zone["end"], zone["start"]], 
                                               fill="toself", fillcolor=zone_color, 
                                               opacity=1, line=dict(color="rgba(255,255,255,0)"), 
                                               showlegend=False, name=f"{zone['type']} Zone"))
            
            fig_elliot.add_trace(go.Scatter(x=elliot_df_shortterm.index, y=sma_elliot_shortterm, name="SMA 50", line=dict(color=colors['neutral'], width=1.5)))
            
            fig_elliot.update_layout(
                height=500,  # Reduzierte Höhe für ein kompakteres Layout
                width=1200,  # Erweiterte Breite für bündigen Look
                showlegend=False,
                title_text=f"Elliot Wave (Short-Term) Analyse für {ticker} (1 Minute, aktueller Tag, nur Handelszeiten)",
                plot_bgcolor=colors['background'],
                paper_bgcolor=colors['background'],
                font_color=colors['text'],
                xaxis=dict(
                    title="Uhrzeit",
                    type="date",
                    tickformat="%H:%M",  # Zeige nur die Uhrzeit (Stunden:Minuten)
                    rangeslider_visible=False,  # Entferne den Rangeslider für einen sauberen Look
                    showgrid=False,  # Keine Gitterlinien auf der X-Achse
                    tickangle=0,  # Horizontale Zeitstempel für bessere Lesbarkeit
                    tickfont=dict(size=10)  # Kleinere Schriftgröße für Zeit
                ),
                yaxis=dict(
                    title="Preis (USD)",
                    showgrid=True,  # Feine Gitterlinien auf der Y-Achse
                    gridcolor='gray',
                    gridwidth=0.5,
                    zeroline=False,
                    tickfont=dict(size=10)  # Kleinere Schriftgröße für Preis
                ),
                margin=dict(t=60, b=50, l=50, r=50)  # Anpassung der Ränder für bündigen Look
            )
            st.plotly_chart(fig_elliot, use_container_width=True)
        
        if show_shortterm_signals:
            st.header("Elliot Wave (Short-Term) Signale & Zonen")
            st.markdown(f'<div class="card">', unsafe_allow_html=True)
            st.write(f"**Aktuelles Signal**: {signals_elliot_shortterm[-1] if signals_elliot_shortterm else 'Hold'}")
            for zone in fib_zones_shortterm:
                zone_color = colors['positive'] if zone["type"] == "Buy" else colors['negative']
                st.markdown(f'<span style="color: {zone_color}">**{zone["type"]} Zone:** {zone["start"]:.2f} - {zone["end"]:.2f}</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        if show_shortterm_stats:
            st.header("Elliot Wave (Short-Term) Marktstatistiken")
            st.markdown(f'<div class="card">', unsafe_allow_html=True)
            st.write(f"**Wave Confidence:** {wave_confidence_shortterm:.2%}")
            sentiment_text = f"**Sentiment:** {sentiment:.2f} {'(Bullish)' if sentiment > 0 else '(Bearish)' if sentiment < 0 else '(Neutral)'}"
            st.markdown(f'<span style="color: {colors["positive"] if sentiment > 0 else colors["negative"] if sentiment < 0 else colors["neutral"]}">{sentiment_text}</span>', unsafe_allow_html=True)
            st.write(f"**Robust:** {'✅' if robust_shortterm else '❌'}")
            st.markdown('</div>', unsafe_allow_html=True)

st.sidebar.header("Eingabe-Parameter")
ticker = st.sidebar.text_input("Aktien-Ticker eingeben (z.B. 'PLTR')", "PLTR", key="ticker_input")
time_period = st.sidebar.selectbox("Zeitraum (Jahre)", [1, 3, 5, 10], index=2, key="time_period_select")

# Sentiment-Schieberegler (Slider) mit kontinuierlichen Werten
st.sidebar.header("Sentiment Anpassung")
manual_sentiment = st.sidebar.slider("Manuelles Sentiment (-1 bis 1)", -1.0, 1.0, 0.0, 0.01)  # Schrittgröße 0.01 für kontinuierliche Werte

st.sidebar.header("Anzeigeoptionen")
show_standard = st.sidebar.checkbox("Standarddiagramm", value=True, key="show_standard")

with st.sidebar.expander("Elliot Wave (Long-Term)"):
    show_longterm = st.checkbox("Elliot Wave (Long-Term) Analyse", value=True, key="show_longterm")
    show_longterm_signals = st.checkbox("Signale & Zonen", value=False, key="show_longterm_signals")
    show_longterm_stats = st.checkbox("Marktstatistik", value=False, key="show_longterm_stats")

with st.sidebar.expander("Elliot Wave (Short-Term)"):
    show_shortterm = st.checkbox("Elliot Wave (Short-Term) Analyse", value=False, key="show_shortterm")
    show_shortterm_signals = st.checkbox("Signale & Zonen", value=False, key="show_shortterm_signals")
    show_shortterm_stats = st.checkbox("Marktstatistik", value=False, key="show_shortterm_stats")

if st.sidebar.button("Analyse starten", key="analyze_button"):
    analyze_stock(ticker, time_period, show_standard, show_longterm, show_longterm_signals, show_longterm_stats, show_shortterm, show_shortterm_signals, show_shortterm_stats, manual_sentiment)

st.markdown("""
    <footer style="text-align: center; padding: 10px; color: #888;">
        © 2025 Professionelle Marktanalyse | Erstellt mit Streamlit
    </footer>
""", unsafe_allow_html=True)
