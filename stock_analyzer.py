import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from datetime import datetime, timedelta
from scipy.signal import argrelextrema
import cv2
import io
from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go

# --- Globale Konfiguration ---
colors = {
    "background": "#1a1a1a",  # An das dunkle Thema des Stock Analyzers angepasst
    "card": "#2c2c2c",
    "text": "#FFFFFF",
    "positive": "#00FF00",
    "negative": "#FF0000",
    "neutral": "#FFA500"
}

# Setze das Streamlit-Theme für ein professionelles Aussehen
st.set_page_config(page_title="Professionelle Marktanalyse", layout="wide", initial_sidebar_state="expanded")

# Benutzerdefinierte CSS für ein modernes Design
st.markdown("""
    <style>
    .main {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-color: #2c2c2c;
        color: #ffffff;
    }
    .sidebar .sidebar-content input, .sidebar .sidebar-content button {
        background-color: #404040;
        color: #ffffff;
        border: 1px solid #555;
    }
    .sidebar .sidebar-content button:hover {
        background-color: #555;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stExpander {
        background-color: #2c2c2c;
        border: 1px solid #555;
        border-radius: 5px;
    }
    .stExpander > div > div {
        background-color: #2c2c2c;
    }
    .stExpander > div > div > p {
        color: #ffffff;
    }
    .stCheckbox > div > label {
        color: #ffffff;
    }
    h1, h2, h3 {
        color: #4CAF50 !important;
    }
    .css-1aumxhk {
        background-color: #1a1a1a;
    }
    .legend-toggle {
        cursor: pointer;
        padding: 5px 10px;
        background-color: #2c2c2c;
        border: 1px solid #555;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .legend-toggle:hover {
        background-color: #404040;
    }
    </style>
""", unsafe_allow_html=True)

# Funktion zur Datenabfrage (aus Elliot-Wellen-Code)
def fetch_data(symbol, timeframe="1d", period="1y"):
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period=period, interval=timeframe)
        return df if not df.empty else pd.DataFrame()
    except:
        return pd.DataFrame()

# Funktion zur Berechnung von RSI (Relative Strength Index)
def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Funktion zur Berechnung von Bollinger Bands
def calculate_bollinger_bands(data, period=20):
    data = data.copy()
    data['MA20'] = data['Close'].rolling(window=period).mean()
    data['STD20'] = data['Close'].rolling(window=period).std()
    data['Upper'] = data['MA20'] + (2 * data['STD20'])
    data['Lower'] = data['MA20'] - (2 * data['STD20'])
    return data

# Funktion zur Berechnung des Exponential Moving Average (EMA)
def calculate_ema(data, period=50):
    data = data.copy()
    data['EMA50'] = data['Close'].ewm(span=period, adjust=False).mean()
    return data

# Funktion zur Berechnung des MACD
def calculate_macd(data, fast=12, slow=26, signal=9):
    exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

# Funktion zur Berechnung des Stochastic Oscillators
def calculate_stochastic(data, period=14):
    low_min = data['Low'].rolling(window=period).min()
    high_max = data['High'].rolling(window=period).max()
    k = 100 * (data['Close'] - low_min) / (high_max - low_min)
    d = k.rolling(window=3).mean()
    return k, d

# Funktion zur Berechnung von VWAP
def calculate_vwap(data):
    data = data.copy()
    data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
    return data

# Funktion zur Berechnung von OBV (On-Balance Volume)
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

# Funktion zur Berechnung von ATR (Average True Range)
def calculate_atr(data, period=14):
    data = data.copy()
    high_low = data['High'] - data['Low']
    high_close = abs(data['High'] - data['Close'].shift(1))
    low_close = abs(data['Low'] - data['Close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['ATR'] = tr.rolling(window=period).mean()
    return data

# Funktion zur Generierung von Signalen mit konservativerer Logik
def generate_signals(data, current_price):
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

    # Long-Signale (Kaufsignale) mit konservativeren Schwellen
    long_signals = 0
    if current_rsi < 45 and current_price < lower_band:  # Strengere RSI-Schwelle für Long
        long_signals += 1
    if current_k < 45 and current_d < 45:  # Strengere Stochastic-Schwelle für Long
        long_signals += 1
    if current_price < ema50 and current_price < current_vwap:  # EMA50 und VWAP für Long-Trend
        long_signals += 1
    if current_obv > prev_obv:  # OBV steigt für Long-Volumen
        long_signals += 1
    if current_macd > current_signal and macd.iloc[-2] <= signal_line.iloc[-2]:  # MACD-Kreuzung für Long
        long_signals += 1

    # Long-Signal, wenn mindestens 3 von 5 Indikatoren erfüllt sind
    if long_signals >= 3:
        confidence_score = min(100, long_signals * 20)  # 20 Punkte pro Indikator (max. 100% für 5/5)
        entry_range_buy = f"{max(current_price * 0.98, lower_band * 0.98):.2f}–{lower_band:.2f}"
        exit_range_sell = f"{ema50 * 1.10:.2f}–{upper_band * 1.05:.2f}"  # Zielzone: 10% über EMA50 bis 5% über oberes Band
        price_increase_forecast = f"{((ema50 * 1.10 / current_price) - 1) * 100:.1f}%"
        stop_loss = max(current_price - (1.5 * current_atr), lower_band * 0.95)  # Konservativer Stop-Loss: 1.5x ATR unter Einstieg
        profit_take = ema50 * 1.15  # Konservativer Profit-Take: 15% über EMA50
        signal_type = "Long"
        recommendation = (f"Kaufsignal (Long) im Bereich {entry_range_buy}, Zielzone (Verkauf): {exit_range_sell}, "
                        f"voraussichtlicher Kursanstieg: {price_increase_forecast}, "
                        f"Stop-Loss: {stop_loss:.2f}, Profit-Take: {profit_take:.2f}, Konfidenz: {confidence_score}%")

    # Short-Signale (Verkaufssignale) mit konservativeren Schwellen
    short_signals = 0
    if current_rsi > 55 and current_price > upper_band:  # Strengere RSI-Schwelle für Short
        short_signals += 1
    if current_k > 55 and current_d > 55:  # Strengere Stochastic-Schwelle für Short
        short_signals += 1
    if current_price > ema50 and current_price > current_vwap:  # EMA50 und VWAP für Short-Trend
        short_signals += 1
    if current_obv < prev_obv:  # OBV fällt für Short-Volumen
        short_signals += 1
    if current_macd < current_signal and macd.iloc[-2] >= signal_line.iloc[-2]:  # MACD-Kreuzung für Short
        short_signals += 1

    # Short-Signal, wenn mindestens 3 von 5 Indikatoren erfüllt sind
    if short_signals >= 3:
        confidence_score = min(100, short_signals * 20)  # 20 Punkte pro Indikator (max. 100% für 5/5)
        entry_range_buy = f"{upper_band:.2f}–{min(current_price * 1.02, upper_band * 1.02):.2f}"
        exit_range_sell = f"{ema50 * 0.90:.2f}–{lower_band * 0.95:.2f}"  # Zielzone: 10% unter EMA50 bis 5% über unteres Band
        price_increase_forecast = f"{((current_price / (ema50 * 0.90)) - 1) * 100:.1f}%"
        stop_loss = min(current_price + (1.5 * current_atr), upper_band * 1.05)  # Konservativer Stop-Loss: 1.5x ATR über Einstieg
        profit_take = ema50 * 0.85  # Konservativer Profit-Take: 15% unter EMA50
        signal_type = "Short"
        recommendation = (f"Verkaufssignal (Short) im Bereich {entry_range_buy}, Zielzone (Rückkauf): {exit_range_sell}, "
                        f"voraussichtlicher Kursrückgang: {price_increase_forecast}, "
                        f"Stop-Loss: {stop_loss:.2f}, Profit-Take: {profit_take:.2f}, Konfidenz: {confidence_score}%")

    if not signal_type:
        recommendation = "Keine Aktion, keine klare Signale von Indikatoren (weniger als 3 Signale für Long oder Short)"
        signal_type = None
        confidence_score = 0

    return recommendation, signal_type, entry_range_buy, exit_range_sell, price_increase_forecast, confidence_score, stop_loss, profit_take

# Funktion für Backtesting mit realistischerer Renditeberechnung für Long und Short
def backtest_signals(data, ticker):
    signals = []
    for i in range(20, len(data)):  # Start nach 20 Tagen für Indikatoren
        temp_data = data.iloc[:i]
        current_price = temp_data['Close'].iloc[-1]
        rec, _, _, _, _, _, stop_loss, profit_take = generate_signals(temp_data, current_price)
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
            max_holding_days = 30  # Maximale Holding-Periode von 30 Tagen
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
            max_holding_days = 30  # Maximale Holding-Periode von 30 Tagen
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

# Funktion für Korrelationsanalyse und Volatilität
def calculate_correlation(data, ticker):
    sp500 = yf.Ticker("^GSPC").history(start=data.index[0], end=data.index[-1])['Close']
    correlation = data['Close'].corr(sp500)
    volatility = data['Close'].pct_change().std() * np.sqrt(252) * 100  # Annualized volatility in %
    return f"Korrelation mit S&P 500: {correlation:.2f}, Annualisierte Volatilität: {volatility:.2f}%"

# Funktion für Heatmap mit allen Indikatoren (optional, für Streamlit angepasst)
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

# --- Elliot-Wellen-Funktionen (unabhängig vom Zeitrahmen, standardmäßig 1 Jahr) ---
def generate_chart_image(df):
    if df.empty:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["Close"], label="Close")
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img = cv2.imdecode(np.frombuffer(buf.read(), np.uint8), cv2.IMREAD_COLOR)
    plt.close()
    return img

def detect_waves_in_image(img):
    if img.size == 0 or img.shape[0] == 1:
        return np.array([], dtype=int)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    peaks = argrelextrema(edges.mean(axis=0), np.greater, order=10)[0]
    return peaks

def identify_elliot_waves(df):
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
        "macd": calculate_macd(df)[0],  # MACD-Linie
        "volatility": df["Close"].pct_change().rolling(14).std()
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
    else:
        wave_confidence = 0.5
    
    chart_img = generate_chart_image(df)
    waves_ki = detect_waves_in_image(chart_img)
    
    return waves, wave_confidence, waves_ki

def get_sentiment(symbol):
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
        {"start": levels[4], "end": levels[3], "type": "Buy"},  # 61.8% - 50% Kaufzone
        {"start": levels[1], "end": levels[0], "type": "Sell"}   # 23.6% - 0% Verkaufszone
    ]
    return levels, zones

def generate_signals_elliot(df):
    if df.empty:
        return []
    sma = calculate_sma(df)
    rsi = calculate_rsi(df)  # RSI bleibt für Signale erhalten
    signals = []
    for i in range(len(df)):
        signal = "Hold"
        if df["Close"].iloc[i] > sma.iloc[i] and rsi.iloc[i] > 70:
            signal = "Sell"
        elif df["Close"].iloc[i] < sma.iloc[i] and rsi.iloc[i] < 30:
            signal = "Buy"
        signals.append(signal)
    return signals

# Funktion zur Robustheitsprüfung (aus Elliot-Wellen-Code)
def robustness_check(df, waves):
    if df.empty or len(waves) == 0:
        return False
    z_scores = np.abs((df["Close"] - df["Close"].mean()) / df["Close"].std())
    return not (z_scores > 3).any()

# Hauptprogramm mit Streamlit-Oberfläche und flexibler Zeitraumwahl
def analyze_stock(ticker, years=5, show_elliot=False, show_stats=False, show_signals=False, show_standard_plots=False):
    # Daten von Yahoo Finance ziehen (flexibler Zeitrahmen für Standardanalyse)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    
    # Prüfe auf fehlende Daten und fülle sie auf
    data = data.dropna()
    if len(data) < 50:  # Mindestens 50 Datenpunkte für sinnvolle Analyse
        st.error(f"Unzureichende Daten für {ticker}. Mindestens 50 Datenpunkte erforderlich.")
        return
    
    # Alle Indikatoren berechnen
    rsi = calculate_rsi(data)
    data = calculate_bollinger_bands(data)
    data = calculate_ema(data)
    macd, signal_line = calculate_macd(data)
    k, d = calculate_stochastic(data)
    data = calculate_vwap(data)
    data = calculate_obv(data)
    data = calculate_atr(data)

    # Aktueller Preis
    current_price = data['Close'].iloc[-1]
    
    # Signale generieren mit konservativerer Strategie
    recommendation, signal_type, entry_range_buy, exit_range_sell, price_increase_forecast, confidence_score, stop_loss, profit_take = generate_signals(data, current_price)
    backtest_result = backtest_signals(data, ticker)
    
    # Elliot-Wellen-Daten (unabhängig, standardmäßig 1 Jahr)
    elliot_df = fetch_data(ticker, timeframe="1d", period="1y")  # Fixer Zeitrahmen von 1 Jahr für Elliot-Wellen
    waves, wave_confidence, waves_ki = identify_elliot_waves(elliot_df) if not elliot_df.empty else ([], 0.5, [])
    sentiment = get_sentiment(ticker)
    sma_elliot = calculate_sma(elliot_df)
    fib_levels, fib_zones = calculate_fibonacci_levels(elliot_df)
    signals_elliot = generate_signals_elliot(elliot_df)
    robust = robustness_check(elliot_df, waves)  # Berechnung der Robustheit für Elliot-Wellen
    
    # Ausgabe in Streamlit mit professionellem Layout
    col1, col2 = st.columns([1, 2])  # Zwei Spalten für eine Dashboard-Struktur
    
    with col1:
        st.header("Marktdaten")
        st.write(f"**Aktueller Preis**: {current_price:.2f} USD")
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
    
    # Zusammenfassung und Risikoanalyse
    st.header("Zusammenfassung und Risikoanalyse")
    st.write(f"{calculate_correlation(data, ticker)}")
    st.write(f"{backtest_result}")
    
    # Elliot-Wellen-Diagramm und Inhalte nur anzeigen, wenn ausgewählt
    if show_elliot:
        st.header("Elliot-Wellen-Analyse")
        # Legende ausklappbar machen
        if st.checkbox("Legende ein-/ausblenden", value=True, key="legend_toggle_elliot"):
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
        fig_elliot.add_trace(go.Candlestick(x=elliot_df.index, open=elliot_df["Open"], high=elliot_df["High"], low=elliot_df["Low"], close=elliot_df["Close"], name="Preis"))
        
        # Elliot-Wellen einzeichnen
        for i, wave in enumerate(waves):
            fig_elliot.add_trace(go.Scatter(x=[elliot_df.index[list(wave.values()).index(v)] for v in wave.values()], 
                                           y=list(wave.values()), mode="lines+markers", name=f"Wave {i+1}", line=dict(color=colors['positive'])))
        
        # Kauf- und Verkaufssignale einzeichnen
        for i in range(len(elliot_df)):
            if signals_elliot[i] == "Buy":
                fig_elliot.add_trace(go.Scatter(x=[elliot_df.index[i]], y=[elliot_df["Close"].iloc[i]], mode="markers", 
                                               marker=dict(symbol="triangle-up", size=12, color=colors['positive']), name="Buy Signal"))
            elif signals_elliot[i] == "Sell":
                fig_elliot.add_trace(go.Scatter(x=[elliot_df.index[i]], y=[elliot_df["Close"].iloc[i]], mode="markers", 
                                               marker=dict(symbol="triangle-down", size=12, color=colors['negative']), name="Sell Signal"))
        
        # Fibonacci-Zonen einzeichnen mit go.Scatter
        for zone in fib_zones:
            fig_elliot.add_trace(go.Scatter(x=[elliot_df.index[0], elliot_df.index[-1], elliot_df.index[-1], elliot_df.index[0], elliot_df.index[0]], 
                                           y=[zone["start"], zone["start"], zone["end"], zone["end"], zone["start"]], 
                                           fill="toself", fillcolor=colors['positive'] if zone["type"] == "Buy" else colors['negative'], 
                                           opacity=0.2, line=dict(color="rgba(255,255,255,0)"), 
                                           showlegend=False, name=f"{zone['type']} Zone"))
        
        # SMA einzeichnen
        fig_elliot.add_trace(go.Scatter(x=elliot_df.index, y=sma_elliot, name="SMA 50", line=dict(color=colors['neutral'])))
        
        fig_elliot.update_layout(
            height=800,
            showlegend=False,  # Legende wird jetzt über Checkbox kontrolliert
            title_text=f"Elliot-Wellen-Analyse für {ticker} (1 Jahr)",
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font_color=colors['text'],
            margin=dict(t=100, b=50, l=50, r=50)  # Mehr Platz für den Chart
        )
        st.plotly_chart(fig_elliot, use_container_width=True)
    
    # Marktstatistiken nur anzeigen, wenn ausgewählt
    if show_stats:
        st.header("Elliot-Wellen-Marktstatistiken")
        st.markdown(f'<div class="card">', unsafe_allow_html=True)
        st.write(f"**Wave Confidence:** {wave_confidence:.2%}")
        sentiment_text = f"**Sentiment:** {sentiment:.2f} {'(Bullish)' if sentiment > 0 else '(Bearish)' if sentiment < 0 else '(Neutral)'}"
        st.markdown(f'<span style="color: {colors["positive"] if sentiment > 0 else colors["negative"] if sentiment < 0 else colors["neutral"]}">{sentiment_text}</span>', unsafe_allow_html=True)
        st.write(f"**Robust:** {'✅' if robust else '❌'}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Signale & Zonen nur anzeigen, wenn ausgewählt
    if show_signals:
        st.header("Elliot-Wellen-Signale & Zonen")
        st.markdown(f'<div class="card">', unsafe_allow_html=True)
        st.write(f"**Aktuelles Signal:** {signals_elliot[-1] if signals_elliot else 'Hold'}")
        for zone in fib_zones:
            zone_color = colors['positive'] if zone["type"] == "Buy" else colors['negative']
            st.markdown(f'<span style="color: {zone_color}">**{zone["type"]} Zone:** {zone["start"]:.2f} - {zone["end"]:.2f}</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Standard-Diagramme nur anzeigen, wenn ausgewählt
    if show_standard_plots:
        # Visualisierung mit professionellem Design
        st.header("Visualisierung")
        fig, axes = plt.subplots(5, 1, figsize=(15, 20), facecolor='#1a1a1a')
        plt.style.use('dark_background')  # Dunkler Hintergrund für Plots
        
        # Kurs, Bollinger Bands, EMA50, VWAP
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
        
        # RSI
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
        
        # Stochastic
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
        
        # MACD
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
        
        # OBV und VWAP
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

    # Optional: Heatmap anzeigen
    if st.checkbox("Indikator-Korrelations-Heatmap anzeigen", key="heatmap_checkbox"):
        heatmap_fig = plot_indicator_heatmap(data)
        st.pyplot(heatmap_fig)

# Streamlit-Interaktion mit flexibler Zeitraumwahl und professionellem Design
st.sidebar.header("Eingabe-Parameter")
ticker = st.sidebar.text_input("Aktien-Ticker eingeben (z.B. 'PLTR')", "PLTR", key="ticker_input")
time_period = st.sidebar.selectbox("Zeitraum (Jahre)", [1, 3, 5, 10], index=2, key="time_period_select")

# Menü für Inhalte vor „Analyse starten“
st.sidebar.header("Anzeigeoptionen")
show_elliot = st.sidebar.checkbox("Elliot-Wellen-Analyse anzeigen", value=False, key="show_elliot")
show_stats = st.sidebar.checkbox("Elliot-Wellen-Marktstatistiken anzeigen", value=False, key="show_stats")
show_signals = st.sidebar.checkbox("Elliot-Wellen-Signale & Zonen anzeigen", value=False, key="show_signals")
show_standard_plots = st.sidebar.checkbox("Standard-Diagramme anzeigen", value=True, key="show_standard_plots")

if st.sidebar.button("Analyse starten", key="analyze_button"):
    analyze_stock(ticker, time_period, show_elliot, show_stats, show_signals, show_standard_plots)

# Füge ein Footer hinzu für ein professionelles Aussehen
st.markdown("""
    <footer style="text-align: center; padding: 10px; color: #888;">
        © 2025 Professionelle Marktanalyse | Erstellt mit Streamlit
    </footer>
""", unsafe_allow_html=True)
