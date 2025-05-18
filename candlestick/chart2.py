import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate MACD and its components
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    exp1 = data['Close'].ewm(span=short_window, adjust=False).mean()
    exp2 = data['Close'].ewm(span=long_window, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram

# Function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Download historical data (example: Apple Inc.)
ticker = 'BITX'
data = yf.download(ticker, start='2023-01-01', end='2025-05-11')

# Calculate MACD and RSI
macd, signal, histogram = calculate_macd(data)
rsi = calculate_rsi(data)

data['MACD'] = macd
data['Signal'] = signal
data['Histogram'] = histogram
data['RSI'] = rsi

# Define trading signals
data['Buy'] = ((data['MACD'] > data['Signal']) & (data['MACD'].shift(1) <= data['Signal'].shift(1)) & (data['RSI'] < 30))
data['Sell'] = ((data['MACD'] < data['Signal']) & (data['MACD'].shift(1) >= data['Signal'].shift(1)) & (data['RSI'] > 70))

# Plotting
plt.figure(figsize=(14, 10))

# Price with buy/sell signals
plt.subplot(3, 1, 1)
plt.plot(data['Close'], label='Close Price', color='c')
plt.scatter(data.index[data['Buy']], data['Close'][data['Buy']], marker='^', color='g', label='Buy Signal')
plt.scatter(data.index[data['Sell']], data['Close'][data['Sell']], marker='v', color='r', label='Sell Signal')
plt.title(f'{ticker} Close Price with Buy/Sell Signals')
plt.legend()

# MACD and Signal
plt.subplot(3, 1, 2)
plt.plot(data['MACD'], label='MACD', color='b')
plt.plot(data['Signal'], label='Signal Line', color='r')
plt.bar(data.index, data['Histogram'], label='Histogram', color='gray')
plt.title('MACD Indicator')
plt.legend()

# RSI
plt.subplot(3, 1, 3)
plt.plot(data['RSI'], label='RSI', color='m')
plt.axhline(30, linestyle='--', color='green')
plt.axhline(70, linestyle='--', color='red')
plt.title('RSI Indicator')
plt.legend()

plt.tight_layout()
plt.show()


