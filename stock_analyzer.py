import streamlit as st
from streamlit_lottie import st_lottie
import requests
import time
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy.signal import argrelextrema
from PIL import Image
import io
from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
from pytz import timezone
import quandl
import fredapi

# API-Schlüssel aus Streamlit Secrets laden
QUANDL_API_KEY = st.secrets["QUANDL_API_KEY"]
FRED_API_KEY = st.secrets["FRED_API_KEY"]
FINNHUB_API_KEY = st.secrets["FINNHUB_API_KEY"]
ALPHA_VANTAGE_API_KEY = st.secrets["ALPHA_VANTAGE_API_KEY"]

# FRED API initialisieren
fred = fredapi.Fred(api_key=FRED_API_KEY)

# --- Globale Konfiguration ---
colors = {
    "background": "#1A2526",  # Dunkles Blau für Finanzthema
    "card": "#2E3839",       # Etwas helleres Blau-Grau
    "text": "#E6ECEC",       # Sanftes Weiß für Lesbarkeit
    "positive": "#00C897",   # Grün für Gewinne
    "negative": "#FF5252",   # Rot für Verluste
    "neutral": "#FFD700",    # Gold für neutrale Akzente
    "header1": "#FFD700",    # Gold für Haupttitel
    "header2": "#00BCD4",    # Cyan für Untertitel
    "header3": "#B0BEC5"     # Leichtes Grau für Subtitel
}

# Definition der Aktien-Clusters (unverändert)
clusters = {
    "Tech 33": [
        "Adobe - ADBE", "AMD - AMD", "Apple - AAPL", "Atlassian - TEAM", "Broadcom - AVGO",
        "Cisco - CSCO", "Cloudflare - NET", "Coinbase - COIN", "eBay - EBAY", "Electronic Arts - EA",
        "Google (Alphabet) - GOOGL", "Meta - META", "Microsoft - MSFT", "MicroStrategy - MSTR", "Netflix - NFLX",
        "NVIDIA - NVDA", "Oracle - ORCL", "Palantir - PLTR", "Palo Alto Networks - PANW", "PayPal - PYPL",
        "Qualcomm - QCOM", "Salesforce - CRM", "Shopify - SHOP", "Snowflake - SNOW", "Tesla - TSLA",
        "Texas Instruments - TXN", "Uber - UBER", "Zscaler - ZS"
    ],
    "DAX 40": [
        "Adidas - ADS.DE", "Airbus - AIR.DE", "Allianz - ALV.DE", "BASF - BAS.DE", "Bayer - BAYN.DE",
        "Beiersdorf - BEI.DE", "BMW - BMW.DE", "Brenntag - BNR.DE", "Commerzbank - CBK.DE", "Continental - CON.DE",
        "Daimler Truck Holding - DTG.DE", "Deutsche Bank - DBK.DE", "Deutsche Börse - DB1.DE", "Deutsche Telekom - DTE.DE",
        "DHL Group - DHL.DE", "E.ON - EOAN.DE", "Fresenius - FRE.DE", "Fresenius Medical Care - FME.DE", "Hannover Rück - HNR1.DE",
        "HeidelbergMaterials - HEI.DE", "Henkel - HEN3.DE", "Infineon Technologies - IFX.DE", "Mercedes Benz Group - MBG.DE",
        "MTU Aero Engines - MTX.DE", "Munchener Re - MUV2.DE", "Porsche AC - P911.DE", "Porsche Automobil Holding - PAH3.DE",
        "Qiagen - QIA.DE", "Rheinmetall - RHM.DE", "RWE - RWE.DE", "SAP - SAP.DE", "Sartorius - SRT3.DE",
        "Siemens - SIE.DE", "Siemens Energy - ENR.DE", "Siemens Healthineers - SHL.DE", "Symrise - SY1.DE",
        "Vonovia - VNA.DE", "Zalando - ZAL.DE", "Volkswagen - VOW3.DE"
    ],
    "US Titans": [
        "3M - MMM", "American Express - AXP", "Amazon - AMZN", "American Water Works - AWK", "Berkshire Hathaway - BRK-B",
        "Black Rock - BLK", "Boeing - BA", "Caterpillar - CAT", "Chevron - CVX", "Cisco - CSCO",
        "Coca-Cola - KO", "Dow Inc - DOW", "Exxon Mobil - XOM", "Goldman Sachs - GS", "Home Depot - HD",
        "Honeywell - HON", "IBM - IBM", "Johnson & Johnson - JNJ", "JPMorgan Chase - JPM", "Mastercard - MA",
        "McDonald’s - MCD", "Merck - MRK", "Microsoft - MSFT", "Nike - NKE", "NVIDIA - NVDA",
        "Occidental Petroleum - OXY", "PepsiCo - PEP", "Procter & Gamble - PG", "Salesforce - CRM", "Starbucks - SBUX",
        "T-Mobile - TMUS", "Travelers - TRV", "Under Armour - UAA", "UnitedHealth - UNH", "Verizon Communications - VZ",
        "Visa - V", "Walgreens Boots Alliance - WBA", "Walmart - WMT", "Waste Management - WM", "Walt Disney - DIS",
        "Warner Bros. - WBD"
    ],
    "China Titans": [
        "Alibaba - BABA", "Baidu - BIDU", "BYD - 1211.HK", "China Life Insurance - 2628.HK", "Geely",
        "JD - JD", "Miniso - MNSO", "NetEase - NTES", "NIO - NIO", "PDD Holdings - PDD",
        "Petrochina", "Sino Biopharmaceutical - 1177.HK", "Tencent Music - TME", "WuXi Biologics - 2269.HK", "Xiaomi - 1810.HK", "XPeng - XPEV"
    ],
    "Eigene Auswahl": []
}

# Neue Funktion zum Abrufen makroökonomischer Daten von FRED
def fetch_macro_data(start_date, end_date):
    try:
        # Inflationsrate (CPIAUCSL - Consumer Price Index)
        inflation = fred.get_series('CPIAUCSL', start_date, end_date)
        inflation = inflation.pct_change(12) * 100  # Jährliche Inflationsrate
        
        # Arbeitslosenrate (UNRATE)
        unemployment = fred.get_series('UNRATE', start_date, end_date)
        
        # Zinssatz (FEDFUNDS - Federal Funds Rate)
        interest_rate = fred.get_series('FEDFUNDS', start_date, end_date)
        
        macro_df = pd.DataFrame({
            'Inflation': inflation,
            'Unemployment': unemployment,
            'InterestRate': interest_rate
        }).dropna()
        
        # Zeitzone entfernen, um mit yfinance-Daten kompatibel zu sein
        macro_df.index = macro_df.index.tz_localize(None)
        
        return macro_df
    except Exception as e:
        st.warning(f"Fehler beim Abrufen makroökonomischer Daten: {str(e)}")
        return pd.DataFrame()

def get_stock_info(symbol):
    try:
        stock = yf.Ticker(symbol)
        ticker_info = stock.history(period="1d", interval="1m")
        info = stock.info
        if not ticker_info.empty:
            current_price = ticker_info['Close'].iloc[-1]
            pre_market = stock.pre_market_price if hasattr(stock, 'pre_market_price') and stock.pre_market_price else None
            after_market = stock.after_hours_price if hasattr(stock, 'after_hours_price') and stock.after_hours_price else None
            price_display = f"{current_price:.2f} USD"
            if pre_market and datetime.now().hour < 9:
                price_display = f"Pre-Market: {pre_market:.2f} USD"
            elif after_market and datetime.now().hour >= 16:
                price_display = f"After-Market: {after_market:.2f} USD"
            return {
                "current_price": current_price, "price_display": price_display,
                "day_range": f"{info.get('dayLow', 0):.2f} – {info.get('dayHigh', 0):.2f} USD",
                "avg_volume": f"{info.get('averageVolume', 0):,.0f}",
                "market_cap": f"{info.get('marketCap', 0):,.0f} USD",
                "pe_ratio": f"{info.get('trailingPE', 0):.2f}" if info.get('trailingPE') else "Nicht verfügbar",
                "beta": f"{info.get('beta', 0):.2f}" if info.get('beta') else "Nicht verfügbar"
            }
        return None, "Keine aktuellen Daten verfügbar"
    except Exception as e:
        return None, f"Fehler beim Abrufen der Daten: {str(e)}"

def clean_trading_data(df):
    if df.empty: return df
    df = df.dropna()
    today = datetime.now(timezone('America/New_York')).replace(hour=0, minute=0, second=0, microsecond=0)
    return df[df.index.date == today.date()].between_time('09:30', '16:00', inclusive='both')

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
    return 100 - (100 / (1 + rs))

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

# Anpassung der Signalgenerierung mit makroökonomischen Faktoren
def generate_signals(data, current_price, sentiment=0, macro_df=None):
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

    sentiment_factor = max(-1, min(1, sentiment))
    long_signals = 0
    short_signals = 0

    # Makroökonomische Faktoren einbeziehen
    macro_factor = 0
    if macro_df is not None and not macro_df.empty:
        latest_macro = macro_df.iloc[-1]
        inflation = latest_macro['Inflation']
        unemployment = latest_macro['Unemployment']
        interest_rate = latest_macro['InterestRate']
        
        # Höhere Inflation und Zinsen könnten Short-Signale verstärken
        if inflation > 3 or interest_rate > 2:
            macro_factor -= 0.5
        # Niedrige Arbeitslosenrate könnte Long-Signale verstärken
        if unemployment < 5:
            macro_factor += 0.5

    if current_rsi < 45 and current_price < lower_band:
        long_signals += 1 + (sentiment_factor * 0.5) + macro_factor
    if current_k < 45 and current_d < 45:
        long_signals += 1 + (sentiment_factor * 0.5) + macro_factor
    if current_price < ema50 and current_price < current_vwap:
        long_signals += 1 + (sentiment_factor * 0.5) + macro_factor
    if current_obv > prev_obv:
        long_signals += 1 + (sentiment_factor * 0.5) + macro_factor
    if current_macd > current_signal and macd.iloc[-2] <= signal_line.iloc[-2]:
        long_signals += 1 + (sentiment_factor * 0.5) + macro_factor

    if current_rsi > 55 and current_price > upper_band:
        short_signals += 1 - (sentiment_factor * 0.5) + macro_factor
    if current_k > 55 and current_d > 55:
        short_signals += 1 - (sentiment_factor * 0.5) + macro_factor
    if current_price > ema50 and current_price > current_vwap:
        short_signals += 1 - (sentiment_factor * 0.5) + macro_factor
    if current_obv < prev_obv:
        short_signals += 1 - (sentiment_factor * 0.5) + macro_factor
    if current_macd < current_signal and macd.iloc[-2] >= signal_line.iloc[-2]:
        short_signals += 1 - (sentiment_factor * 0.5) + macro_factor

    base_confidence = 20
    if long_signals >= 3:
        base_score = long_signals * base_confidence
        confidence_adjustment = (sentiment_factor + macro_factor) * 15
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
        confidence_adjustment = -(sentiment_factor + macro_factor) * 15
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

def backtest_signals(data, ticker, sentiment=0, period="1y", macro_df=None):
    elliot_data = fetch_data(ticker, timeframe="1d", period=period)
    if not elliot_data.empty and len(elliot_data) > 50:
        elliot_data.loc[elliot_data.index[-1], 'Close'] = data['Close'].iloc[-1]
        signals_elliot = generate_signals_elliot(elliot_data, sentiment, macro_df)
    else:
        signals_elliot = ["Hold"] * len(data)

    min_length = min(len(data), len(signals_elliot))
    data = data.iloc[:min_length]
    signals_elliot = signals_elliot[:min_length]

    long_returns = 0
    short_returns = 0
    long_trades = 0
    short_trades = 0
    trade_details = []

    for i in range(20, min_length - 10):
        temp_data = data.iloc[:i]
        current_price = temp_data['Close'].iloc[-1]
        macro_slice = macro_df.loc[:temp_data.index[-1]] if macro_df is not None and not macro_df.empty else None
        rec_standard, signal_type, _, _, _, _, _, _ = generate_signals(temp_data, current_price, sentiment, macro_slice)
        signal_elliot = signals_elliot[i]

        if "Kaufsignal" in rec_standard or signal_elliot == "Buy":
            exit_price = data['Close'].iloc[i + 10] if i + 10 < len(data) else data['Close'].iloc[-1]
            trade_return = ((exit_price - current_price) / current_price) * 100
            long_returns += trade_return
            long_trades += 1
            trade_details.append(f"Long-Trade: Einstieg {current_price:.2f}, Ausstieg {exit_price:.2f}, Rendite {trade_return:.2f}%")
        elif "Verkaufssignal" in rec_standard or signal_elliot == "Sell":
            exit_price = data['Close'].iloc[i + 10] if i + 10 < len(data) else data['Close'].iloc[-1]
            trade_return = ((current_price - exit_price) / current_price) * 100
            short_returns += trade_return
            short_trades += 1
            trade_details.append(f"Short-Trade: Einstieg {current_price:.2f}, Ausstieg {exit_price:.2f}, Rendite {trade_return:.2f}%")

    total_returns = long_returns + short_returns
    total_trades = long_trades + short_trades
    
    if trade_details:
        with st.expander("Trade-Details", expanded=False) as expander:
            st.markdown('<div class="card-3d"><div class="text-3d">', unsafe_allow_html=True)
            for detail in trade_details:
                st.write(detail)
            st.markdown('</div></div>', unsafe_allow_html=True)
    
    return (f"Backtest-Rendite für {ticker} (Elliot & Standard kombiniert): "
            f"Long: {long_returns:.2f}% über {long_trades} Trades, "
            f"Short: {short_returns:.2f}% über {short_trades} Trades, "
            f"Gesamt: {total_returns:.2f}% über {total_trades} Trades") if total_trades > 0 else f"Backtest-Rendite für {ticker}: 0.00% über 0 Trades"

def calculate_correlation(data, ticker):
    sp500 = yf.Ticker("^GSPC").history(start=data.index[0], end=data.index[-1])['Close']
    # Zeitzone von sp500 entfernen, um mit data kompatibel zu sein
    sp500.index = sp500.index.tz_localize(None)
    correlation = data['Close'].corr(sp500)
    return f"Korrelation mit S&P 500: {correlation:.2f}"

def generate_chart_image(df):
    if df.empty: return None
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["Close"], label="Close", color='green')
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", facecolor=colors['background'], edgecolor='none')
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    return img

def detect_waves_in_image(img):
    if img is None or isinstance(img, np.ndarray) and (img.size == 0 or img.shape[0] == 1): return np.array([], dtype=int)
    if isinstance(img, Image.Image):
        img_array = np.array(img.convert('L'))
    else:
        img_array = np.array(img)
    edges = np.gradient(np.mean(img_array, axis=0))
    peaks = argrelextrema(edges, np.greater, order=10)[0]
    return peaks

def identify_elliot_waves(df, sentiment=0, macro_df=None):
    if df.empty: return [], 0.5, []
    highs = argrelextrema(df["High"].values, np.greater, order=5)[0]
    lows = argrelextrema(df["Low"].values, np.less, order=5)[0]
    waves = []
    for i in range(min(len(highs), len(lows)) - 4):
        wave_pattern = {"W1": df["Close"].iloc[lows[i]], "W2": df["Close"].iloc[highs[i]], "W3": df["Close"].iloc[lows[i+1]],
                        "W4": df["Close"].iloc[highs[i+1]], "W5": df["Close"].iloc[lows[i+2]]}
        waves.append(wave_pattern)
    
    features = pd.DataFrame({"rsi": calculate_rsi(df), "macd": calculate_macd(df)[0], "volatility": df["Close"].pct_change().rolling(14).std(), "sentiment": sentiment})
    if macro_df is not None and not macro_df.empty:
        macro_df = macro_df.reindex(features.index, method='ffill')
        features['inflation'] = macro_df['Inflation']
        features['unemployment'] = macro_df['Unemployment']
        features['interest_rate'] = macro_df['InterestRate']
    features = features.dropna()
    
    valid_indices = features.index[features.index <= df.index[-1]] if not df.index.empty else features.index[:1] if not features.empty else pd.Index([0])
    if len(valid_indices) == 0 and not features.empty: valid_indices = features.index[:1]
    
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
        if sentiment > 0: wave_confidence = min(1.0, wave_confidence + (sentiment * 0.2))
        elif sentiment < 0: wave_confidence = max(0.0, wave_confidence + (sentiment * 0.2))
    else:
        wave_confidence = 0.5
    
    chart_img = generate_chart_image(df)
    waves_ki = detect_waves_in_image(chart_img)
    return waves, wave_confidence, waves_ki

def get_sentiment(symbol, manual_sentiment=None):
    if manual_sentiment is not None: return manual_sentiment
    news = [f"{symbol} reports strong earnings this quarter.", f"Analysts predict a bullish trend for {symbol}.", f"Market uncertainty affects {symbol} negatively."]
    return np.mean([TextBlob(text).sentiment.polarity for text in news])

def calculate_sma(df, period=50):
    if df.empty: return pd.Series([], dtype=float)
    return df["Close"].rolling(window=period).mean()

def calculate_fibonacci_levels(df):
    if df.empty: return [], []
    high = df["High"].max()
    low = df["Low"].min()
    levels = [high - (high - low) * level for level in [0, 0.236, 0.382, 0.5, 0.618, 1, 1.618]]
    zones = [{"start": levels[4], "end": levels[3], "type": "Buy"}, {"start": levels[1], "end": levels[0], "type": "Sell"}]
    return levels, zones

def generate_signals_elliot(df, sentiment=0, macro_df=None):
    if df.empty: return []
    sma = calculate_sma(df)
    rsi = calculate_rsi(df)
    signals = []
    macro_factor = 0
    if macro_df is not None and not macro_df.empty:
        latest_macro = macro_df.iloc[-1]
        inflation = latest_macro['Inflation']
        unemployment = latest_macro['Unemployment']
        interest_rate = latest_macro['InterestRate']
        if inflation > 3 or interest_rate > 2:
            macro_factor -= 0.5
        if unemployment < 5:
            macro_factor += 0.5

    for i in range(len(df)):
        signal = "Hold"
        sentiment_factor = max(-1, min(1, sentiment))
        if df["Close"].iloc[i] > sma.iloc[i] and rsi.iloc[i] > (70 + (sentiment_factor * 10) + macro_factor * 10):
            signal = "Sell"
        elif df["Close"].iloc[i] < sma.iloc[i] and rsi.iloc[i] < (30 - (sentiment_factor * 10) - macro_factor * 10):
            signal = "Buy"
        signals.append(signal)
    return signals

def robustness_check(df, waves):
    if df.empty or len(waves) == 0: return False
    z_scores = np.abs((df["Close"] - df["Close"].mean()) / df["Close"].std())
    return not (z_scores > 3).any()

def analyze_stock(ticker, years=5, show_standard=True, show_longterm=True, show_longterm_signals=True, show_longterm_stats=True, manual_sentiment=None):
    stock_info = get_stock_info(ticker)
    if stock_info is None or isinstance(stock_info, str):
        st.error(stock_info[1] if isinstance(stock_info, tuple) else "Keine Daten verfügbar")
        return
    
    current_price = stock_info["current_price"]
    price_display = stock_info["price_display"]
    day_range = stock_info["day_range"]
    avg_volume = stock_info["avg_volume"]
    market_cap = stock_info["market_cap"]
    pe_ratio = stock_info["pe_ratio"]
    beta = stock_info["beta"]

    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    
    # Makroökonomische Daten abrufen
    macro_df = fetch_macro_data(start_date, end_date)
    
    # Zeitzone von yfinance-Daten entfernen
    data.index = data.index.tz_localize(None)
    
    # macro_df an data.index anpassen
    macro_df = macro_df.reindex(data.index, method='ffill')
    
    data = data.dropna()
    if len(data) < 50:
        st.error(f"Unzureichende Daten für {ticker}. Mindestens 50 Datenpunkte erforderlich.")
        return
    
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
    recommendation, signal_type, entry_range_buy, exit_range_sell, price_increase_forecast, confidence_score, stop_loss, profit_take = generate_signals(data, current_price, sentiment, macro_df)
    backtest_result = backtest_signals(data, ticker, sentiment, period=f"{years}y", macro_df=macro_df)
    
    elliot_df_longterm = fetch_data(ticker, timeframe="1d", period="1y")
    if not elliot_df_longterm.empty:
        elliot_df_longterm.index = elliot_df_longterm.index.tz_localize(None)  # Zeitzone entfernen
        elliot_df_longterm.loc[elliot_df_longterm.index[-1], 'Close'] = current_price
        elliot_df_longterm.loc[elliot_df_longterm.index[-1], 'Open'] = current_price
        elliot_df_longterm.loc[elliot_df_longterm.index[-1], 'High'] = current_price
        elliot_df_longterm.loc[elliot_df_longterm.index[-1], 'Low'] = current_price
    macro_df_longterm = fetch_macro_data(elliot_df_longterm.index[0], elliot_df_longterm.index[-1]) if not elliot_df_longterm.empty else pd.DataFrame()
    macro_df_longterm = macro_df_longterm.reindex(elliot_df_longterm.index, method='ffill') if not elliot_df_longterm.empty else pd.DataFrame()
    waves_longterm, wave_confidence_longterm, waves_ki_longterm = identify_elliot_waves(elliot_df_longterm, sentiment, macro_df_longterm) if not elliot_df_longterm.empty else ([], 0.5, [])
    sma_elliot_longterm = calculate_sma(elliot_df_longterm)
    fib_levels_longterm, fib_zones_longterm = calculate_fibonacci_levels(elliot_df_longterm)
    signals_elliot_longterm = generate_signals_elliot(elliot_df_longterm, sentiment, macro_df_longterm)
    robust_longterm = robustness_check(elliot_df_longterm, waves_longterm)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f'<div class="card-3d"><h2 class="header-3d">Marktdaten</h2>'
                    f'<div class="text-3d"><p><b>Aktueller Preis:</b> {price_display}</p>'
                    f'<p><b>Tagesbereich:</b> {day_range}</p>'
                    f'<p><b>Durchschnittliches Volumen:</b> {avg_volume}</p>'
                    f'<p><b>Marktkapitalisierung:</b> {market_cap}</p>'
                    f'<p><b>Kurs-Gewinn-Verhältnis (KGV):</b> {pe_ratio}</p>'
                    f'<p><b>Beta (Marktrisiko):</b> {beta} (<1 stabil, >1 volatil)</p></div></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'<div class="card-3d"><h2 class="header-3d">Handelsempfehlung</h2>', unsafe_allow_html=True)
        if signal_type:
            stop_loss_display = f"{stop_loss:.2f}" if stop_loss is not None else "N/A"
            profit_take_display = f"{profit_take:.2f}" if profit_take is not None else "N/A"
            st.markdown(f'<div class="text-3d"><p><b>Empfehlung:</b> {recommendation}</p>'
                        f'<p><b>Einstiegsbereich:</b> {entry_range_buy if entry_range_buy else "N/A"}</p>'
                        f'<p><b>Verkaufsbereich (oder Rückkauf bei Short):</b> {exit_range_sell if exit_range_sell else "N/A"}</p>'
                        f'<p><b>Prognostizierter Kursanstieg/Rückgang:</b> {price_increase_forecast if price_increase_forecast else "N/A"}</p>'
                        f'<p><b>Stop-Loss (Risikomanagement):</b> {stop_loss_display}</p>'
                        f'<p><b>Profit-Take (Ziel):</b> {profit_take_display}</p>'
                        f'<p><b>Konfidenz des Signals:</b> {confidence_score}%</p></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="text-3d"><p><b>Keine klare Handelsempfehlung vorhanden.</b></p>'
                        f'<p><b>Aktueller RSI:</b> {rsi.iloc[-1]:.2f}</p>'
                        f'<p><b>Bollinger Bands</b> - Oberes Band: {data["Upper"].iloc[-1]:.2f}, Unteres Band: {data["Lower"].iloc[-1]:.2f}</p>'
                        f'<p><b>Stochastic %K:</b> {k.iloc[-1]:.2f}, %D: {d.iloc[-1]:.2f}</p>'
                        f'<p><b>EMA50:</b> {data["EMA50"].iloc[-1]:.2f}</p>'
                        f'<p><b>VWAP:</b> {data["VWAP"].iloc[-1]:.2f}</p></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown(f'<div class="card-3d"><h2 class="header-3d">Zusammenfassung und Risikoanalyse</h2>'
                f'<div class="text-3d"><p>{calculate_correlation(data, ticker)}</p><p>{backtest_result}</p></div></div>', unsafe_allow_html=True)    
    if show_standard:
        st.markdown(f'<div class="card-3d"><h2 class="header-3d">Standardanalyse</h2></div>', unsafe_allow_html=True)
        fig, axes = plt.subplots(4, 1, figsize=(15, 16), facecolor=colors['background'])
        plt.style.use('dark_background')
        
        axes[0].plot(data.index, data['Close'], label='Kurs', color=colors['positive'], linewidth=2.0, alpha=0.8)
        axes[0].plot(data.index, data['MA20'], label='MA20', color=colors['neutral'], linewidth=2.0, alpha=0.8)
        axes[0].plot(data.index, data['Upper'], label='Oberes Band', color=colors['negative'], linestyle='--', linewidth=1.5, alpha=0.7)
        axes[0].plot(data.index, data['Lower'], label='Unteres Band', color=colors['positive'], linestyle='--', linewidth=1.5, alpha=0.7)
        axes[0].plot(data.index, data['EMA50'], label='EMA50', color=colors['neutral'], linestyle='-.', linewidth=2.0, alpha=0.8)
        axes[0].plot(data.index, data['VWAP'], label='VWAP', color=colors['header2'], linestyle='-.', linewidth=2.0, alpha=0.8)
        if signal_type == "Long":
            axes[0].scatter(data.index[-1], current_price, color=colors['positive'], marker='^', s=200, label='Kauf-Signal', edgecolor='white', linewidth=1)
        elif signal_type == "Short":
            axes[0].scatter(data.index[-1], current_price, color=colors['negative'], marker='v', s=200, label='Verkauf-Signal', edgecolor='white', linewidth=1)
        axes[0].set_title("Marktanalyse für " + ticker, fontweight='bold', fontsize=24, color=colors['header1'])
        axes[0].set_xlabel("Datum", fontweight='bold', fontsize=14, color=colors['text'])
        axes[0].set_ylabel("Preis (USD)", fontweight='bold', fontsize=14, color=colors['text'])
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., facecolor=colors['card'], edgecolor='white', labelcolor=colors['text'], shadow=True)
        axes[0].grid(True, linestyle='--', alpha=0.3, color='#4A4A4A')
        axes[0].set_facecolor(colors['background'])
        
        axes[1].plot(data.index, rsi, label='RSI', color=colors['header2'], linewidth=2.0, alpha=0.8)
        axes[1].axhline(y=50, color='#4A4A4A', linestyle='--', alpha=0.5, label='Neutral')
        axes[1].axhline(y=30, color=colors['positive'], linestyle='--', alpha=0.7, label='Überverkauft (Long)')
        axes[1].axhline(y=70, color=colors['negative'], linestyle='--', alpha=0.7, label='Überkauft (Short)')
        axes[1].set_title("Relative Strength Index (RSI)", fontweight='bold', fontsize=20, color=colors['header2'])
        axes[1].set_xlabel("Datum", fontweight='bold', fontsize=14, color=colors['text'])
        axes[1].set_ylabel("RSI-Wert", fontweight='bold', fontsize=14, color=colors['text'])
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., facecolor=colors['card'], edgecolor='white', labelcolor=colors['text'], shadow=True)
        axes[1].grid(True, linestyle='--', alpha=0.3, color='#4A4A4A')
        axes[1].set_facecolor(colors['background'])
        
        axes[2].plot(data.index, k, label='%K', color=colors['negative'], alpha=0.8, linewidth=2.0)
        axes[2].plot(data.index, d, label='%D', color=colors['positive'], alpha=0.8, linewidth=2.0)
        axes[2].axhline(y=50, color='#4A4A4A', linestyle='--', alpha=0.5, label='Neutral')
        axes[2].axhline(y=30, color=colors['positive'], linestyle='--', alpha=0.7, label='Überverkauft (Long)')
        axes[2].axhline(y=70, color=colors['negative'], linestyle='--', alpha=0.7, label='Überkauft (Short)')
        axes[2].set_title("Stochastic Oscillator", fontweight='bold', fontsize=20, color=colors['header2'])
        axes[2].set_xlabel("Datum", fontweight='bold', fontsize=14, color=colors['text'])
        axes[2].set_ylabel("Stochastic-Wert", fontweight='bold', fontsize=14, color=colors['text'])
        axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., facecolor=colors['card'], edgecolor='white', labelcolor=colors['text'], shadow=True)
        axes[2].grid(True, linestyle='--', alpha=0.3, color='#4A4A4A')
        axes[2].set_facecolor(colors['background'])
        
        axes[3].plot(data.index, macd, label='MACD', color=colors['header2'], linewidth=2.0, alpha=0.8)
        axes[3].plot(data.index, signal_line, label='Signal Line', color=colors['neutral'], linewidth=2.0, alpha=0.8)
        axes[3].bar(data.index, macd - signal_line, label='MACD Histogram', color='#4A4A4A', alpha=0.5, width=0.8)
        axes[3].axhline(y=0, color='#4A4A4A', linestyle='-', alpha=0.3)
        axes[3].set_title("Moving Average Convergence Divergence (MACD)", fontweight='bold', fontsize=20, color=colors['header2'])
        axes[3].set_xlabel("Datum", fontweight='bold', fontsize=14, color=colors['text'])
        axes[3].set_ylabel("MACD-Wert", fontweight='bold', fontsize=14, color=colors['text'])
        axes[3].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., facecolor=colors['card'], edgecolor='white', labelcolor=colors['text'], shadow=True)
        axes[3].grid(True, linestyle='--', alpha=0.3, color='#4A4A4A')
        axes[3].set_facecolor(colors['background'])
        
        plt.tight_layout()
        st.pyplot(fig)
    
    if show_longterm:
        st.markdown(f'<div class="card-3d"><h2 class="header-3d">Elliot-Wave-Analyse</h2></div>', unsafe_allow_html=True)
        if st.checkbox("Legende ein-/ausblenden", value=True, key="legend_toggle_longterm"):
            legend_html = f"""
            <div class="card-3d" style="background-color: {colors['card']}; padding: 15px; border-radius: 15px; box-shadow: 0 10px 20px rgba(0, 0, 0, 0.5), 0 0 10px rgba(255, 215, 0, 0.2);">
                <ul style="list-style-type: none; padding: 0; margin: 0; font-size: 14px; color: {colors['text']};">
                    <li><span style="color: {colors['positive']};">●</span> Buy Signal / Zone</li>
                    <li><span style="color: {colors['negative']};">●</span> Sell Signal / Zone</li>
                    <li><span style="color: {colors['neutral']};">●</span> SMA</li>
                    <li><span style="color: {colors['positive']};">-</span> Elliot Wave</li>
                </ul>
            </div>
            """
            st.markdown(legend_html, unsafe_allow_html=True)
        
        fig_elliot = go.Figure()
        fig_elliot.add_trace(go.Candlestick(x=elliot_df_longterm.index, open=elliot_df_longterm["Open"], high=elliot_df_longterm["High"], low=elliot_df_longterm["Low"], close=elliot_df_longterm["Close"], name="Preis",
                                            increasing_line_color=colors['positive'], decreasing_line_color=colors['negative']))
        
        for i, wave in enumerate(waves_longterm):
            fig_elliot.add_trace(go.Scatter(x=[elliot_df_longterm.index[list(wave.values()).index(v)] for v in wave.values()], 
                                            y=list(wave.values()), mode="lines+markers", name=f"Wave {i+1}", line=dict(color=colors['positive'], width=2),
                                            marker=dict(size=8, color=colors['positive'], symbol='triangle-up', line=dict(width=1, color='white'))))
        
        for i in range(len(elliot_df_longterm)):
            if signals_elliot_longterm[i] == "Buy":
                fig_elliot.add_trace(go.Scatter(x=[elliot_df_longterm.index[i]], y=[elliot_df_longterm["Close"].iloc[i]], mode="markers", 
                                                marker=dict(symbol="triangle-up", size=12, color=colors['positive'], line=dict(width=1, color='white')), name="Buy Signal"))
            elif signals_elliot_longterm[i] == "Sell":
                fig_elliot.add_trace(go.Scatter(x=[elliot_df_longterm.index[i]], y=[elliot_df_longterm["Close"].iloc[i]], mode="markers", 
                                                marker=dict(symbol="triangle-down", size=12, color=colors['negative'], line=dict(width=1, color='white')), name="Sell Signal"))
        
        for zone in fib_zones_longterm:
            fig_elliot.add_trace(go.Scatter(x=[elliot_df_longterm.index[0], elliot_df_longterm.index[-1], elliot_df_longterm.index[-1], elliot_df_longterm.index[0], elliot_df_longterm.index[0]], 
                                            y=[zone["start"], zone["start"], zone["end"], zone["end"], zone["start"]], 
                                            fill="toself", fillcolor=colors['positive'] if zone["type"] == "Buy" else colors['negative'], 
                                            opacity=0.2, line=dict(color="rgba(255,255,255,0)"), showlegend=False, name=f"{zone['type']} Zone"))
        
        fig_elliot.add_trace(go.Scatter(x=elliot_df_longterm.index, y=sma_elliot_longterm, name="SMA 50", line=dict(color=colors['neutral'], width=2, dash='dashdot')))
        
        fig_elliot.update_layout(height=900, showlegend=False, title_text=f"Elliot-Wave-Analyse für {ticker} (1 Jahr)",
                                 plot_bgcolor=colors['background'], paper_bgcolor=colors['background'], font_color=colors['text'], 
                                 title_font=dict(size=24, color=colors['header1'], family='Arial Black'),
                                 margin=dict(t=100, b=50, l=50, r=50),
                                 scene=dict(camera=dict(eye=dict(x=1.25, y=1.25, z=0.1))),
                                 hovermode="x unified",
                                 transition={'duration': 300, 'easing': 'cubic-in-out'})
        
        st.plotly_chart(fig_elliot, use_container_width=True)
        
        st.markdown(f'<div class="card-3d"><h2 class="header-3d">Elliot-Wave-Signale & Zonen</h2>', unsafe_allow_html=True)
        st.markdown(f'<div class="text-3d"><p><b>Aktuelles Signal:</b> {signals_elliot_longterm[-1] if signals_elliot_longterm else "Hold"}</p></div>', unsafe_allow_html=True)
        
        latest_signal_idx = -1
        latest_signal_type = signals_elliot_longterm[-1] if signals_elliot_longterm else "Hold"
        for i in range(len(signals_elliot_longterm)-1, -1, -1):
            if signals_elliot_longterm[i] in ["Buy", "Sell"]:
                latest_signal_idx = i
                latest_signal_type = signals_elliot_longterm[i]
                break
        
        if latest_signal_idx != -1 and latest_signal_type != "Hold":
            signal_date = elliot_df_longterm.index[latest_signal_idx].strftime('%Y-%m-%d')
            current_price = elliot_df_longterm["Close"].iloc[latest_signal_idx]
            wave_count = len(waves_longterm)
            current_wave = min(latest_signal_idx // (len(elliot_df_longterm) // wave_count) + 1, wave_count) if wave_count > 0 else 1
            
            forecast = 0.0
            if waves_longterm and wave_count > 0:
                latest_wave = waves_longterm[-1]
                if latest_signal_type == "Buy":
                    target_price = latest_wave.get("W5", latest_wave.get("W3", current_price))
                    forecast = ((target_price - current_price) / current_price) * 100
                elif latest_signal_type == "Sell":
                    target_price = latest_wave.get("W4", latest_wave.get("W2", current_price))
                    forecast = ((current_price - target_price) / current_price) * 100
            
            wave_duration = 0
            if wave_count > 0:
                wave_start_idx = latest_signal_idx - (len(elliot_df_longterm) // wave_count) * (current_wave - 1)
                wave_end_idx = min(wave_start_idx + (len(elliot_df_longterm) // wave_count), len(elliot_df_longterm) - 1)
                if wave_start_idx >= 0 and wave_end_idx < len(elliot_df_longterm):
                    wave_duration = (elliot_df_longterm.index[wave_end_idx] - elliot_df_longterm.index[wave_start_idx]).days
            
            wave_amplitude = 0.0
            if wave_count > 0 and wave_start_idx >= 0 and wave_end_idx < len(elliot_df_longterm):
                wave_start_price = elliot_df_longterm["Close"].iloc[wave_start_idx]
                wave_end_price = elliot_df_longterm["Close"].iloc[wave_end_idx]
                wave_amplitude = ((wave_end_price - wave_start_price) / wave_start_price) * 100
            
            trend_direction = "Neutral"
            if current_wave in [1, 3, 5]: trend_direction = "Bullish"
            elif current_wave in [2, 4]: trend_direction = "Bearish"
            
            st.markdown(f'<div class="text-3d"><p><b>Details zum aktuellsten Signal:</b></p>'
                        f'<p>- <b>Signaldatum:</b> {signal_date}</p>'
                        f'<p>- <b>Prognostizierter Kurs{"anstieg" if latest_signal_type == "Buy" else "verlust"}:</b> {forecast:.1f}%</p>'
                        f'<p>- <b>Wellenlänge:</b> {wave_duration} Tage</p>'
                        f'<p>- <b>Wellenstärke:</b> {wave_amplitude:.1f}%</p>'
                        f'<p>- <b>Trendrichtung:</b> {trend_direction}</p></div>', unsafe_allow_html=True)

        for zone in fib_zones_longterm:
            zone_color = colors['positive'] if zone["type"] == "Buy" else colors['negative']
            st.markdown(f'<div class="text-3d"><span style="color: {zone_color}"><b>{zone["type"]} Zone:</b> {zone["start"]:.2f} - {zone["end"]:.2f}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown(f'<div class="card-3d"><h2 class="header-3d">Elliot-Wave-Marktstatistiken</h2>', unsafe_allow_html=True)
        st.markdown(f'<div class="text-3d"><p><b>Wave Confidence:</b> {wave_confidence_longterm:.2%} (Sicherheit der Wellenstruktur)</p>'
                    f'<p><b>Sentiment:</b> {sentiment:.2f} {"(Bullish)" if sentiment > 0 else "(Bearish)" if sentiment < 0 else "(Neutral)"}</p>'
                    f'<p><b>Robust:</b> {"✅" if robust_longterm else "❌"} (Daten ohne extreme Ausreißer)</p></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def analyze_cluster_for_buy_signals(cluster_name, manual_sentiment=0):
    st.markdown(f'<div class="card-3d"><h2 class="header-3d">Elliot-Wave Kaufsignale für {cluster_name}</h2></div>', unsafe_allow_html=True)
    buy_signals = []
    for stock in sorted(clusters[cluster_name]):
        ticker = stock.split(" - ")[1]
        stock_info = get_stock_info(ticker)
        if isinstance(stock_info, dict):
            current_price = stock_info["current_price"]
            data = fetch_data(ticker, timeframe="1d", period="1y")
            if not data.empty and len(data) >= 50:
                data.loc[data.index[-1], 'Close'] = current_price
                data.loc[data.index[-1], 'Open'] = current_price
                data.loc[data.index[-1], 'High'] = current_price
                data.loc[data.index[-1], 'Low'] = current_price
                macro_df = fetch_macro_data(data.index[0], data.index[-1])
                signals_elliot = generate_signals_elliot(data, manual_sentiment, macro_df)
                if signals_elliot[-1] == "Buy":
                    buy_signals.append(ticker)
        else:
            st.warning(f"Keine Daten für {ticker} verfügbar.")
    
    if buy_signals:
        st.markdown(f'<div class="card-3d"><div class="text-3d"><p><b>Aktien mit Kaufsignal in {cluster_name}:</b></p><ul>{"".join(f"<li>{ticker}</li>" for ticker in buy_signals)}</ul></div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="text-3d"><p><b>Keine Kaufsignale in {cluster_name} gefunden.</b></p></div>', unsafe_allow_html=True)

# Funktion zum Laden der Lottie-Animation
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Lottie-Animation laden
lottie_animation = load_lottie_url("https://lottie.host/f345caf5-6546-46cc-8af2-351bc9988b45/le250UCTFs.json")

# Streamlit-Konfiguration und Design
st.set_page_config(page_title="Aktienanalyse", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.css" />
    <script src="https://unpkg.com/lottie-player@latest/dist/lottie-player.js"></script>
    <style>
        body {font-family: 'Roboto', sans-serif; background: #1A2526; color: #E6ECEC; overflow-x: hidden;}
        .main {background: transparent;}
        .hero {background: linear-gradient(135deg, #1A2526 0%, #2E3839 50%, rgba(0, 200, 151, 0.1) 100%), url('https://www.transparenttextures.com/patterns/asfalt-dark.png'); background-size: cover; padding: 60px 20px; text-align: center; border-radius: 25px; box-shadow: 0 20px 50px rgba(0, 0, 0, 0.7), 0 0 15px rgba(0, 188, 212, 0.3); position: relative; overflow: hidden; margin-bottom: 30px; transition: transform 0.3s ease;}
        .hero h1 {font-size: 4em; font-weight: 900; color: #FFD700; text-shadow: 0 4px 8px rgba(255, 215, 0, 0.4), 0 0 10px rgba(255, 215, 0, 0.2); margin: 0; animation: fadeInDown 1s ease;}
        .hero p {font-size: 1.8em; color: #E6ECEC; text-shadow: 0 2px 4px rgba(230, 236, 236, 0.3); margin: 10px 0 0; animation: fadeInUp 1s ease 0.5s forwards; opacity: 0;}
        .hero:hover {transform: scale(1.02);}
        .hero::before {content: ''; position: absolute; top: -50%; left: -50%; width: 200%; height: 200%; background: radial-gradient(circle, rgba(0, 188, 212, 0.15), transparent); animation: pulse 10s infinite ease-in-out;}
        .sidebar .sidebar-content {background: rgba(46, 56, 57, 0.95); backdrop-filter: blur(10px); border-radius: 20px; padding: 25px; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3), 0 0 10px rgba(255, 215, 0, 0.2); transition: transform 0.3s ease, box-shadow 0.3s ease;}
        .sidebar .sidebar-content:hover {transform: translateY(-10px) scale(1.02); box-shadow: 0 15px 40px rgba(0, 0, 0, 0.4), 0 0 15px rgba(255, 215, 0, 0.3);}
        .sidebar h2 {color: #FFD700; text-shadow: 0 0 5px rgba(255, 215, 0, 0.2); transition: all 0.3s ease;}
        .sidebar h2:hover {text-shadow: 0 0 10px rgba(255, 215, 0, 0.4);}
        .sidebar button, .sidebar select, .sidebar input {background: #2E3839; border: 2px solid #00C897; border-radius: 15px; color: #E6ECEC; transition: all 0.3s ease; box-shadow: 0 0 5px rgba(0, 200, 151, 0.2), inset 0 0 3px rgba(255, 215, 0, 0.1);}
        .sidebar button:hover, .sidebar select:hover, .sidebar input:hover {background: #00C897; box-shadow: 0 0 10px rgba(0, 200, 151, 0.4), 0 0 8px rgba(255, 215, 0, 0.2); transform: scale(1.05); border-color: #FFD700;}
        .card-3d {background: rgba(46, 56, 57, 0.9); border-radius: 20px; padding: 20px; margin: 20px 0; box-shadow: 0 15px 40px rgba(0, 0, 0, 0.5), 0 0 10px rgba(255, 215, 0, 0.2), 0 5px 10px rgba(255, 215, 0, 0.1); transform: perspective(1000px) translateZ(0) rotateX(5deg); transition: transform 0.3s ease, box-shadow 0.3s ease; backdrop-filter: blur(8px); position: relative;}
        .card-3d:hover {transform: perspective(1000px) translateZ(15px) rotateX(10deg) scale(1.02); box-shadow: 0 20px 50px rgba(0, 0, 0, 0.6), 0 0 15px rgba(255, 215, 0, 0.3), 0 8px 15px rgba(255, 215, 0, 0.2);}
        .card-3d::before {content: ''; position: absolute; top: -5%; left: -5%; width: 110%; height: 110%; background: radial-gradient(circle, rgba(255, 215, 0, 0.1), transparent); opacity: 0.3; transition: opacity 0.3s ease;}
        .card-3d:hover::before {opacity: 0.5;}
        .header-3d {color: #FFD700; font-size: 2em; font-weight: 900; text-shadow: 0 0 5px rgba(255, 215, 0, 0.2); background: rgba(255, 215, 0, 0.1); border-radius: 15px; padding: 10px; box-shadow: 0 5px 15px rgba(255, 215, 0, 0.2), inset 0 0 3px rgba(255, 215, 0, 0.1); transform: perspective(1000px) translateZ(10px); transition: all 0.3s ease;}
        .header-3d:hover {text-shadow: 0 0 10px rgba(255, 215, 0, 0.4); transform: perspective(1000px) translateZ(15px) scale(1.05); box-shadow: 0 8px 20px rgba(255, 215, 0, 0.3), inset 0 0 5px rgba(255, 215, 0, 0.2);}
        .text-3d {color: #E6ECEC; font-size: 1em; transition: all 0.3s ease; transform: perspective(1000px) translateZ(5px); background: rgba(46, 56, 57, 0.8); border-radius: 10px; padding: 10px; box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3), inset 0 0 3px rgba(230, 236, 236, 0.1), 0 2px 5px rgba(255, 215, 0, 0.1); margin-top: 10px;}
        .text-3d:hover {transform: perspective(1000px) translateZ(10px) scale(1.02); box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4), inset 0 0 5px rgba(230, 236, 236, 0.2), 0 3px 8px rgba(255, 215, 0, 0.2);}
        @keyframes pulse {0% {transform: scale(1); opacity: 0.15;} 50% {transform: scale(1.1); opacity: 0.25;} 100% {transform: scale(1); opacity: 0.15;}}
        @keyframes fadeInDown {from {opacity: 0; transform: translateY(-20px);} to {opacity: 1; transform: translateY(0);}}
        @keyframes fadeInUp {from {opacity: 0; transform: translateY(20px);} to {opacity: 1; transform: translateY(0);}}
        .footer {background: linear-gradient(45deg, #2E3839, #1A2526); padding: 20px; text-align: center; border-top: 2px solid #00C897; box-shadow: 0 -10px 20px rgba(0, 0, 0, 0.3), 0 -5px 10px rgba(255, 215, 0, 0.1); transition: all 0.3s ease;}
        .footer:hover {box-shadow: 0 -15px 30px rgba(0, 0, 0, 0.4), 0 -8px 15px rgba(255, 215, 0, 0.2);}
        .scroll-to-top {position: fixed; bottom: 20px; right: 20px; background: #00C897; color: #FFFFFF; border: none; border-radius: 50%; width: 50px; height: 50px; font-size: 24px; cursor: pointer; box-shadow: 0 0 10px rgba(0, 200, 151, 0.4), 0 0 5px rgba(255, 215, 0, 0.2); transition: all 0.3s ease; z-index: 9999; opacity: 0; transform: translateY(50px); animation: fadeIn 0.5s ease-in-out 2s forwards;}
        .scroll-to-top:hover {transform: scale(1.2) rotate(360deg); box-shadow: 0 0 15px rgba(0, 200, 151, 0.6), 0 0 8px rgba(255, 215, 0, 0.3); background: #FFD700;}
        @keyframes fadeIn {from {opacity: 0; transform: translateY(50px);} to {opacity: 1; transform: translateY(0);}}
        .stExpander {background-color: rgba(46, 56, 57, 0.9); border: 2px solid #00C897; border-radius: 15px; box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3), 0 0 5px rgba(255, 215, 0, 0.1); transition: all 0.3s ease;}
        .stExpander:hover {box-shadow: 0 10px 25px rgba(0, 0, 0, 0.4), 0 0 8px rgba(255, 215, 0, 0.2);}
        .stExpander > div > div {background-color: transparent;}
        .stExpander > div > div > p {color: #E6ECEC;}
        .stCheckbox > div > label {color: #E6ECEC; transition: all 0.3s ease;}
        .stCheckbox > div > label:hover {color: #FFD700; text-shadow: 0 0 5px rgba(255, 215, 0, 0.2);}}
        .loading-container {position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(26, 37, 38, 0.9); z-index: 9999; display: flex; justify-content: center; align-items: center;}
        .loading-container lottie-player {width: 250px; height: 250px;}
    </style>
""", unsafe_allow_html=True)

# Placeholder für die Ladeanimation
loading_placeholder = st.empty()

# Hero-Bereich mit Abstand
st.markdown("""
    <div style="height: 20px;"></div>  <!-- Zeile Abstand -->
    <div class="hero">
        <h1>Stock - Analyzer</h1>
        <p>Elliot - Wave</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("<h2>📈 Marktanalyse-Konfiguration</h2>", unsafe_allow_html=True)
st.sidebar.subheader("1. Wähle einen Markt-Cluster")
selected_cluster = st.sidebar.radio("Wähle einen Cluster:", ["Tech 33", "DAX 40", "US Titans", "China Titans", "Eigene Auswahl"], index=None, horizontal=True)

ticker = None
if selected_cluster:
    if selected_cluster == "Eigene Auswahl":
        st.sidebar.subheader("2. Gib eine Aktie ein")
        ticker = st.sidebar.text_input("Aktien-Ticker eingeben (z.B. 'AAPL' oder 'ALV.DE')", key="custom_ticker")
    else:
        st.sidebar.subheader("2. Wähle eine Aktie (optional)")
        selected_stock = st.sidebar.selectbox("Aktie auswählen:", ["Keine Auswahl"] + sorted(clusters[selected_cluster]), key="stock_select")
        if selected_stock and selected_stock != "Keine Auswahl":
            ticker = selected_stock.split(" - ")[1]

st.sidebar.subheader("3. Analysenzeitraum")
time_period = st.sidebar.selectbox("Analysezeitraum (Jahre):", [1, 3, 5, 10], index=2)

st.sidebar.subheader("4. Marktsentiment")
manual_sentiment = st.sidebar.slider("Marktsentiment anpassen (-1 = Bearish, 0 = Neutral, 1 = Bullish)", -1.0, 1.0, 0.0, 0.01)

st.sidebar.subheader("5. Analyseart")
show_standard = st.sidebar.checkbox("Standardanalyse anzeigen", value=True)
show_longterm = st.sidebar.checkbox("Elliot-Wave-Analyse anzeigen", value=False)

if st.sidebar.button("🚀 Analyse starten", key="analyze_button"):
    # Ladeanimation anzeigen
    with loading_placeholder.container():
        st_lottie(lottie_animation, height=200, key="loading")
        time.sleep(3)  # Simulierte Verzögerung
    
    # Ladeanimation entfernen
    loading_placeholder.empty()
    
    # Analyse ausführen
    if ticker:
        analyze_stock(ticker, time_period, show_standard, show_longterm, True, True, manual_sentiment)
    elif selected_cluster:
        analyze_cluster_for_buy_signals(selected_cluster, manual_sentiment)

# Footer
st.markdown("""
    <div class="footer">
        © 2025 Aktienanalyse | Erstellt mit Streamlit
    </div>
    <button class="scroll-to-top" onclick="window.scrollTo({top: 0, behavior: 'smooth'})">↑</button>
""", unsafe_allow_html=True)
