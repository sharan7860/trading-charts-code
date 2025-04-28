import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates

# ========================
# Indicator Calculations
# ========================

def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    exp1 = data['Close'].ewm(span=short_window, adjust=False).mean()
    exp2 = data['Close'].ewm(span=long_window, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ========================
# Data Preparation
# ========================

ticker = 'APOLLOHOSP.NS'

# Use intraday data for better granularity (last 5 days, 1-minute interval)
data = yf.download(ticker, period='5d', interval='1m')

# Calculate 50-period moving average
data['MA50'] = data['Close'].rolling(window=50).mean()

# Prepare OHLC data for candlestick chart
data_ohlc = data.reset_index()[['Datetime', 'Open', 'High', 'Low', 'Close']]
data_ohlc.rename(columns={'Datetime': 'Date'}, inplace=True)
data_ohlc['Date'] = data_ohlc['Date'].map(mdates.date2num)

# Calculate indicators
macd, signal, histogram = calculate_macd(data)
rsi = calculate_rsi(data)

data['MACD'] = macd
data['Signal'] = signal
data['Histogram'] = histogram
data['RSI'] = rsi

# Generate signals
data['Buy'] = ((data['MACD'] > data['Signal']) & 
               (data['MACD'].shift(1) <= data['Signal'].shift(1)) & 
               (data['RSI'] < 30))
data['Sell'] = ((data['MACD'] < data['Signal']) & 
                (data['MACD'].shift(1) >= data['Signal'].shift(1)) & 
                (data['RSI'] > 70))

# ========================
# Plotting
# ========================

plt.style.use('dark_background')  # Dark style like Yahoo Finance

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
fig.suptitle(f'{ticker} Intraday Technical Analysis', y=1.02)

# 1. Candlestick Chart with 50 MA and Buy/Sell signals
ax1.set_title('Price Chart with 50 MA and Buy/Sell Signals')
candlestick_ohlc(ax1, data_ohlc.values, width=0.0008, colorup='green', colordown='red')

# Plot 50 MA
ax1.plot(data.index, data['MA50'], color='red', label='MA 50')

# Plot Buy/Sell signals
ax1.scatter(data.index[data['Buy']], data['Close'][data['Buy']], marker='^', color='lime', s=100, label='Buy Signal')
ax1.scatter(data.index[data['Sell']], data['Close'][data['Sell']], marker='v', color='red', s=100, label='Sell Signal')

ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# 2. Volume Bars colored by price movement
ax2.set_title('Volume')
colors = ['green' if c >= o else 'red' for c, o in zip(data['Close'], data['Open'])]
ax2.bar(data.index, data['Volume'], color=colors, width=0.0008)
ax2.grid(True, alpha=0.3)

# 3. MACD Panel
ax3.set_title('MACD Indicator')
ax3.plot(data.index, data['MACD'], label='MACD', color='white', linewidth=1.2)
ax3.plot(data.index, data['Signal'], label='Signal', color='red', linewidth=1.2)
ax3.bar(data.index, data['Histogram'], color=['lime' if x >= 0 else 'red' for x in data['Histogram']], alpha=0.7, width=0.0008)
ax3.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax3.legend(loc='upper left')
ax3.grid(True, alpha=0.3)

# Format x-axis for intraday time
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
