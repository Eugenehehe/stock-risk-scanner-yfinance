# Stock Risk Scanner

A no-login stock dashboard using yfinance + Streamlit.

## What it does

- Downloads stock data with yfinance
- Builds indicators: RSI, MACD, moving averages, volatility, volume ratio
- Trains a simple Random Forest model
- Predicts probability that the stock closes higher 5 trading days from now
- Shows chart, signal, support/resistance, and feature importance

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal.

## Important

This is educational only and not financial advice.
