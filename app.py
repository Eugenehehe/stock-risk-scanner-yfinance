
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# -----------------------------
# App Config
# -----------------------------
st.set_page_config(
    page_title="Stock Risk Scanner",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Risk Scanner")
st.caption("No-login stock dashboard using yfinance. Educational use only — not financial advice.")


# -----------------------------
# Helper Functions
# -----------------------------
@st.cache_data(ttl=3600)
def load_data(ticker: str, period: str = "3y") -> pd.DataFrame:
    """
    Download historical daily stock data from Yahoo Finance via yfinance.
    """
    data = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        return pd.DataFrame()

    # yfinance sometimes returns multi-index columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.dropna()
    return data


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple technical indicators.
    """
    df = df.copy()

    df["Return_1D"] = df["Close"].pct_change()
    df["Return_5D"] = df["Close"].pct_change(5)
    df["Volatility_10D"] = df["Return_1D"].rolling(10).std()

    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean()

    df["Price_vs_SMA20"] = df["Close"] / df["SMA_20"] - 1
    df["Price_vs_SMA50"] = df["Close"] / df["SMA_50"] - 1
    df["Price_vs_SMA200"] = df["Close"] / df["SMA_200"] - 1

    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # Volume
    df["Volume_SMA20"] = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA20"]

    # Future target:
    # 1 means price is higher after 5 trading days
    # 0 means price is lower or equal after 5 trading days
    df["Future_Return_5D"] = df["Close"].shift(-5) / df["Close"] - 1
    df["Target"] = (df["Future_Return_5D"] > 0).astype(int)

    df = df.dropna()
    return df


def train_model(df: pd.DataFrame):
    """
    Train a basic RandomForest model to classify whether next 5-day return is positive.
    """
    features = [
        "Return_1D",
        "Return_5D",
        "Volatility_10D",
        "Price_vs_SMA20",
        "Price_vs_SMA50",
        "Price_vs_SMA200",
        "RSI_14",
        "MACD",
        "MACD_Signal",
        "MACD_Hist",
        "Volume_Ratio",
    ]

    model_df = df[features + ["Target"]].dropna()

    if len(model_df) < 250:
        return None, None, None, features

    X = model_df[features]
    y = model_df["Target"]

    # Time-aware split: first 80% train, last 20% test
    split_index = int(len(model_df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=10,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    latest_features = X.iloc[[-1]]
    probability_up = model.predict_proba(latest_features)[0][1]

    return model, accuracy, probability_up, features


def classify_signal(probability_up: float, latest: pd.Series) -> tuple[str, str]:
    """
    Convert model probability + indicators into a simple label.
    """
    rsi = latest["RSI_14"]
    price_vs_20 = latest["Price_vs_SMA20"]
    price_vs_50 = latest["Price_vs_SMA50"]
    volume_ratio = latest["Volume_Ratio"]

    if probability_up >= 0.60 and price_vs_20 > 0 and price_vs_50 > 0:
        return "Bullish", "Model leans upward and price is above key moving averages."
    elif probability_up <= 0.40 and price_vs_20 < 0 and price_vs_50 < 0:
        return "Bearish", "Model leans downward and price is below key moving averages."
    elif rsi > 75 and volume_ratio > 1.5:
        return "Overheated", "RSI and volume are elevated. Risk of pullback may be higher."
    elif rsi < 30:
        return "Oversold", "RSI is low. Could be washed out, but falling knives are dangerous."
    else:
        return "Neutral", "No clean directional edge. Better to wait or use smaller size."


def support_resistance(df: pd.DataFrame) -> tuple[float, float]:
    """
    Simple recent support/resistance using last 60 trading days.
    """
    recent = df.tail(60)
    support = recent["Low"].min()
    resistance = recent["High"].max()
    return support, resistance


# -----------------------------
# Sidebar Controls
# -----------------------------
with st.sidebar:
    st.header("Settings")

    ticker = st.text_input("Ticker", value="NVDA").upper().strip()
    period = st.selectbox(
        "History period",
        ["1y", "2y", "3y", "5y", "10y"],
        index=2
    )

    st.markdown("---")
    st.write("Prediction target:")
    st.write("**Will the stock close higher 5 trading days from now?**")

    run_button = st.button("Analyze", type="primary")


# -----------------------------
# Main App
# -----------------------------
if ticker:
    raw = load_data(ticker, period)

    if raw.empty:
        st.error("No data found. Check the ticker symbol.")
        st.stop()

    df = add_indicators(raw)

    if df.empty or len(df) < 250:
        st.error("Not enough historical data to train a model. Try a longer period or a larger stock.")
        st.stop()

    model, accuracy, probability_up, features = train_model(df)

    if model is None:
        st.error("Not enough clean data after indicators. Try a longer history period.")
        st.stop()

    latest = df.iloc[-1]
    signal, explanation = classify_signal(probability_up, latest)
    support, resistance = support_resistance(df)

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Latest Close", f"${latest['Close']:.2f}")
    col2.metric("5D Up Probability", f"{probability_up * 100:.1f}%")
    col3.metric("Backtest Accuracy", f"{accuracy * 100:.1f}%")
    col4.metric("Signal", signal)

    st.info(explanation)

    # Warning when accuracy is weak
    if accuracy < 0.53:
        st.warning(
            "Model accuracy is weak. Treat this as a risk scanner, not a trade signal."
        )

    # Chart
    st.subheader(f"{ticker} Price Chart")

    chart_df = df[["Close", "SMA_20", "SMA_50", "SMA_200"]].tail(365)
    st.line_chart(chart_df)

    # Technical dashboard
    st.subheader("Technical Snapshot")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("RSI 14", f"{latest['RSI_14']:.1f}")
    c2.metric("Volume Ratio", f"{latest['Volume_Ratio']:.2f}x")
    c3.metric("Support 60D", f"${support:.2f}")
    c4.metric("Resistance 60D", f"${resistance:.2f}")

    # Feature importance
    st.subheader("What the Model Cares About")

    importances = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    st.bar_chart(importances.set_index("Feature"))

    # Raw data
    with st.expander("Show recent data"):
        st.dataframe(
            df.tail(30)[
                [
                    "Open", "High", "Low", "Close", "Volume",
                    "RSI_14", "MACD_Hist", "Volume_Ratio",
                    "Future_Return_5D"
                ]
            ],
            use_container_width=True
        )

    with st.expander("Important limitations"):
        st.write(
            """
            This app is not predicting the future with certainty. It uses historical price behavior
            and technical indicators to estimate whether the stock has a higher probability of being
            up over the next 5 trading days.

            Weaknesses:
            - It does not understand earnings, news, dilution, bankruptcy rumors, insider selling, or macro shocks.
            - yfinance data can be delayed, missing, or adjusted.
            - A 55% probability does not mean the trade is good.
            - A high backtest accuracy can still fail in live trading.
            - For real money, risk management matters more than the model.
            """
        )
else:
    st.write("Enter a ticker to begin.")
