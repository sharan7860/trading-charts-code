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

ticker = 'BITX'
data = yf.download(ticker, start='2025-01-01', end='2025-05-10')

# Prepare OHLC data for candlestick chart
data_ohlc = data.reset_index()[['Date', 'Open', 'High', 'Low', 'Close']]
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

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
fig.suptitle(f'{ticker} Technical Analysis', y=1.02)

# 1. Candlestick Chart
ax1.set_title('Price Chart with Buy/Sell Signals of BITX')
candlestick_ohlc(ax1, data_ohlc.values, width=0.6, colorup='g', colordown='r')
ax1.plot(data.index, data['Close'], label='Close Price', color='c', alpha=0.3)
ax1.scatter(data.index[data['Buy']], data['Close'][data['Buy']], 
            marker='^', color='lime', label='Buy Signal', s=100)
ax1.scatter(data.index[data['Sell']], data['Close'][data['Sell']], 
            marker='v', color='red', label='Sell Signal', s=100)
ax1.legend()
ax1.grid(True)

# 2. MACD
ax2.set_title('MACD Indicator')
ax2.plot(data['MACD'], label='MACD', color='b')
ax2.plot(data['Signal'], label='Signal Line', color='r')
ax2.bar(data.index, data['Histogram'], 
        color=np.where(data['Histogram'] > 0, 'g', 'r'), 
        label='Histogram')
ax2.axhline(0, color='gray', linestyle='--')
ax2.legend()
ax2.grid(True)

# 3. RSI
ax3.set_title('RSI Indicator')
ax3.plot(data['RSI'], label='RSI', color='m')
ax3.axhline(30, linestyle='--', color='green')
ax3.axhline(70, linestyle='--', color='red')
ax3.fill_between(data.index, 30, 70, color='gray', alpha=0.1)
ax3.legend()
ax3.grid(True)

# Format x-axis for dates
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

