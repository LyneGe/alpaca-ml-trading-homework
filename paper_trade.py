"""
Machine Learning Trading Signal with Alpaca
Paper Trading Demo Only

This script:
1. Fetches 5 years of daily OHLCV data from Alpaca
2. Computes technical indicators
3. Applies PCA and keeps components explaining at least 80% variance
4. Trains a Random Forest classifier
5. Generates a long/flat signal
6. Submits paper trade orders only

IMPORTANT: This is paper trading only. No real money is used.
"""

import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier


load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

TICKER = "AAPL"
QTY = 1
PROBABILITY_THRESHOLD = 0.60


if not API_KEY or not SECRET_KEY:
    raise ValueError("Missing Alpaca keys. Please create a .env file with ALPACA_API_KEY and ALPACA_SECRET_KEY.")


print("This is paper trading only — no real money is used.")
print("Starting paper trading demo...")


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    data["Log_Return"] = np.log(data["Close"] / data["Close"].shift(1))
    data["Rolling_Mean_10"] = data["Close"].rolling(10).mean()
    data["Rolling_Std_10"] = data["Close"].rolling(10).std()

    data["SMA_20"] = data["Close"].rolling(20).mean()
    data["SMA_50"] = data["Close"].rolling(50).mean()
    data["EMA_20"] = data["Close"].ewm(span=20, adjust=False).mean()

    ema_12 = data["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = ema_12 - ema_26
    data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()

    delta = data["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    data["RSI"] = 100 - (100 / (1 + rs))

    low_14 = data["Low"].rolling(14).min()
    high_14 = data["High"].rolling(14).max()
    data["Stochastic"] = 100 * (data["Close"] - low_14) / (high_14 - low_14)
    data["Williams_R"] = -100 * (high_14 - data["Close"]) / (high_14 - low_14)

    rolling_20 = data["Close"].rolling(20)
    data["BB_Middle"] = rolling_20.mean()
    data["BB_Upper"] = data["BB_Middle"] + 2 * rolling_20.std()
    data["BB_Lower"] = data["BB_Middle"] - 2 * rolling_20.std()

    high_low = data["High"] - data["Low"]
    high_close = np.abs(data["High"] - data["Close"].shift())
    low_close = np.abs(data["Low"] - data["Close"].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data["ATR"] = true_range.rolling(14).mean()

    data["OBV"] = (np.sign(data["Close"].diff()) * data["Volume"]).fillna(0).cumsum()

    money_flow_multiplier = ((data["Close"] - data["Low"]) - (data["High"] - data["Close"])) / (data["High"] - data["Low"])
    money_flow_volume = money_flow_multiplier.replace([np.inf, -np.inf], np.nan).fillna(0) * data["Volume"]
    data["CMF"] = money_flow_volume.rolling(20).sum() / data["Volume"].rolling(20).sum()

    up_move = data["High"].diff()
    down_move = -data["Low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_di = 100 * pd.Series(plus_dm, index=data.index).rolling(14).sum() / true_range.rolling(14).sum()
    minus_di = 100 * pd.Series(minus_dm, index=data.index).rolling(14).sum() / true_range.rolling(14).sum()
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    data["ADX"] = dx.rolling(14).mean()

    return data


# Alpaca clients
stock_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

# Fetch 5 years of daily data
end_date = datetime.now(timezone.utc)
start_date = end_date - timedelta(days=365 * 5)

request = StockBarsRequest(
    symbol_or_symbols=TICKER,
    timeframe=TimeFrame.Day,
    start=start_date,
    end=end_date,
    feed=DataFeed.IEX,
)

bars = stock_client.get_stock_bars(request).df

if bars.empty:
    raise ValueError("No data returned from Alpaca. Check ticker, keys, and data permissions.")

if isinstance(bars.index, pd.MultiIndex):
    df = bars.loc[TICKER].copy()
else:
    df = bars.copy()

df = df.sort_index()
df = df.rename(columns={
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume",
})

print(f"Downloaded {len(df)} daily rows for {TICKER}.")

# Feature engineering and target
df = compute_features(df)
df["Next_Return"] = df["Close"].pct_change().shift(-1)
df["Target"] = (df["Next_Return"] > 0).astype(int)
df = df.dropna()

feature_cols = [
    "Log_Return", "Rolling_Mean_10", "Rolling_Std_10",
    "SMA_20", "SMA_50", "EMA_20",
    "MACD", "MACD_Signal", "RSI",
    "Stochastic", "Williams_R",
    "BB_Middle", "BB_Upper", "BB_Lower",
    "ATR", "OBV", "CMF", "ADX",
]

X = df[feature_cols]
y = df["Target"]

# Standardize and PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca_full = PCA()
pca_full.fit(X_scaled)
cum_var = np.cumsum(pca_full.explained_variance_ratio_)
n_components = np.argmax(cum_var >= 0.80) + 1

pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

print(f"PCA components kept: {n_components}")
print(f"Explained variance kept: {cum_var[n_components - 1]:.4f}")

# Train model using all but latest row, then predict latest row
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    random_state=42,
    class_weight="balanced",
)

model.fit(X_pca[:-1], y.iloc[:-1])

latest_scaled = scaler.transform(X.iloc[[-1]])
latest_pca = pca.transform(latest_scaled)
prob_up = model.predict_proba(latest_pca)[0, 1]
signal = "LONG" if prob_up > PROBABILITY_THRESHOLD else "FLAT"

print(f"Ticker: {TICKER}")
print(f"Latest close: {df['Close'].iloc[-1]:.2f}")
print(f"Probability of positive next-day return: {prob_up:.4f}")
print(f"Signal: {signal}")

account = trading_client.get_account()
print(f"Account status: {account.status}")
print(f"Paper buying power: {account.buying_power}")

try:
    position = trading_client.get_open_position(TICKER)
    current_qty = float(position.qty)
except Exception:
    current_qty = 0

print(f"Current paper position in {TICKER}: {current_qty}")

# Submit paper order only
if signal == "LONG" and current_qty == 0:
    order = MarketOrderRequest(
        symbol=TICKER,
        qty=QTY,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
    )
    submitted_order = trading_client.submit_order(order)
    print("Submitted paper BUY order:")
    print(submitted_order)

elif signal == "FLAT" and current_qty > 0:
    order = MarketOrderRequest(
        symbol=TICKER,
        qty=current_qty,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
    )
    submitted_order = trading_client.submit_order(order)
    print("Submitted paper SELL order:")
    print(submitted_order)

else:
    print("No paper order submitted. The signal already matches the current position.")

print("Paper trading demo finished.")
