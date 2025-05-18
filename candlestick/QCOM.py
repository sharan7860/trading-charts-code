import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from yfinance import YFRateLimitError

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

# Function to calculate Stochastic Oscillator
def calculate_stochastic(data, k_period=14, d_period=3):
    lowest_low = data['Low'].rolling(window=k_period).min()
    highest_high = data['High'].rolling(window=k_period).max()
    k = 100 * ((data['Close'] - lowest_low) / (highest_high - lowest_low))
    d = k.rolling(window=d_period).mean()
    return k, d

# Function to calculate Aroon Indicator
def calculate_aroon(data, period=14):
    aroon_up = pd.Series(index=data.index, dtype=float)
    aroon_down = pd.Series(index=data.index, dtype=float)
    data_with_pos = data.copy()
    data_with_pos['Position'] = np.arange(len(data_with_pos))
    
    for i in range(period, len(data)):
        window = data_with_pos.iloc[i-period:i+1]
        highest_pos = window['Position'][window['High'].idxmax()]
        lowest_pos = window['Position'][window['Low'].idxmin()]
        current_pos = window['Position'].iloc[-1]
        periods_since_high = current_pos - highest_pos
        periods_since_low = current_pos - lowest_pos
        aroon_up[i] = ((period - periods_since_high) / period) * 100
        aroon_down[i] = ((period - periods_since_low) / period) * 100
    
    return aroon_up, aroon_down

# Download historical data with retry logic
ticker = 'PANW'
max_retries = 3
retry_delay = 60  # seconds

for attempt in range(max_retries):
    try:
        data = yf.download(ticker, start='2023-01-01', end='2025-05-10')
        if data.empty:
            raise ValueError("No data downloaded")
        break
    except YFRateLimitError:
        if attempt < max_retries - 1:
            print(f"Rate limit exceeded. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            continue
        else:
            raise
    except Exception as e:
        print(f"Error downloading data: {e}")
        exit()

# Add Day column
data['Day'] = data.index.day

# Calculate indicators
macd, signal, histogram = calculate_macd(data)
rsi = calculate_rsi(data)
stoch_k, stoch_d = calculate_stochastic(data)
aroon_up, aroon_down = calculate_aroon(data)

# Add indicators to dataframe
data['MACD'] = macd
data['Signal'] = signal
data['Histogram'] = histogram
data['RSI'] = rsi
data['Stoch_K'] = stoch_k
data['Stoch_D'] = stoch_d
data['Aroon_Up'] = aroon_up
data['Aroon_Down'] = aroon_down

# Define trading signals
data['Buy'] = ((data['MACD'] > data['Signal']) & 
              (data['MACD'].shift(1) <= data['Signal'].shift(1)) & 
              (data['RSI'] < 30))
data['Sell'] = ((data['MACD'] < data['Signal']) & 
               (data['MACD'].shift(1) >= data['Signal'].shift(1)) & 
               (data['RSI'] > 70))

# Plotting
fig, axs = plt.subplots(5, 1, figsize=(14, 20), sharex=True)

# Price with buy/sell signals
axs[0].plot(data['Close'], label='Close Price', color='c')
axs[0].scatter(data.index[data['Buy']], data['Close'][data['Buy']], 
              marker='^', color='g', label='Buy Signal')
axs[0].scatter(data.index[data['Sell']], data['Close'][data['Sell']], 
              marker='v', color='r', label='Sell Signal')
axs[0].set_title(f'{ticker} Close Price with Buy/Sell Signals')
axs[0].legend()
axs[0].grid(True)

# MACD and Signal
axs[1].plot(data['MACD'], label='MACD', color='b')
axs[1].plot(data['Signal'], label='Signal Line', color='r')
axs[1].bar(data.index, data['Histogram'], 
          color=np.where(data['Histogram'] >= 0, 'g', 'r'), 
          alpha=0.5)
axs[1].set_title('MACD Indicator')
axs[1].legend()
axs[1].grid(True)

# RSI
axs[2].plot(data['RSI'], label='RSI', color='m')
axs[2].axhline(30, linestyle='--', color='green')
axs[2].axhline(70, linestyle='--', color='red')
axs[2].set_title('RSI Indicator')
axs[2].legend()
axs[2].grid(True)

# Stochastic Oscillator
axs[3].plot(data['Stoch_K'], label='%K', color='blue')
axs[3].plot(data['Stoch_D'], label='%D', color='red')
axs[3].axhline(80, linestyle='--', color='red')
axs[3].axhline(20, linestyle='--', color='green')
axs[3].set_title('Stochastic Oscillator')
axs[3].set_ylim(0, 100)
axs[3].legend()
axs[3].grid(True)

# Aroon Indicator
axs[4].plot(data['Aroon_Up'], label='Aroon Up', color='green')
axs[4].plot(data['Aroon_Down'], label='Aroon Down', color='red')
axs[4].axhline(70, linestyle='--', color='gray')
axs[4].axhline(30, linestyle='--', color='gray')
axs[4].set_title('Aroon Indicator')
axs[4].set_ylim(0, 100)
axs[4].legend()
axs[4].grid(True)

plt.tight_layout()
plt.show()
