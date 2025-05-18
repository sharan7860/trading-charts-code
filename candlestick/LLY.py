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

# Function to calculate Stochastic Oscillator
def calculate_stochastic(data, k_period=14, d_period=3):
    # Calculate %K - Fast Stochastic
    lowest_low = data['Low'].rolling(window=k_period).min()
    highest_high = data['High'].rolling(window=k_period).max()
    k = 100 * ((data['Close'] - lowest_low) / (highest_high - lowest_low))
    
    # Calculate %D - Slow Stochastic (3-day SMA of %K)
    d = k.rolling(window=d_period).mean()
    
    return k, d

# Function to calculate Aroon Indicator - FIXED VERSION
def calculate_aroon(data, period=14):
    # Initialize Aroon columns
    aroon_up = pd.Series(index=data.index, dtype=float)
    aroon_down = pd.Series(index=data.index, dtype=float)
    
    # Add position column to track index position
    data_with_pos = data.copy()
    data_with_pos['Position'] = np.arange(len(data_with_pos))
    
    for i in range(period, len(data)):
        # Get the window data
        window = data_with_pos.iloc[i-period:i+1]
        
        # Find positions of highest high and lowest low
        highest_pos = window['Position'][window['High'].idxmax()]
        lowest_pos = window['Position'][window['Low'].idxmin()]
        
        # Calculate periods since highest high and lowest low
        current_pos = window['Position'].iloc[-1]
        periods_since_high = current_pos - highest_pos
        periods_since_low = current_pos - lowest_pos
        
        # Calculate Aroon values
        aroon_up[i] = ((period - periods_since_high) / period) * 100
        aroon_down[i] = ((period - periods_since_low) / period) * 100
    
    return aroon_up, aroon_down

# Download historical data
ticker = 'AT&T'
data = yf.download(ticker, start='2023-01-01', end='2025-05-09')

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

# Define trading signals (using original conditions)
data['Buy'] = ((data['MACD'] > data['Signal']) & (data['MACD'].shift(1) <= data['Signal'].shift(1)) & (data['RSI'] < 30))
data['Sell'] = ((data['MACD'] < data['Signal']) & (data['MACD'].shift(1) >= data['Signal'].shift(1)) & (data['RSI'] > 70))

# Plotting
plt.figure(figsize=(14, 15))  # Increased figure height for 5 subplots

# Price with buy/sell signals
plt.subplot(5, 1, 1)
plt.plot(data['Close'], label='Close Price', color='c')
plt.scatter(data.index[data['Buy']], data['Close'][data['Buy']], marker='^', color='g', label='Buy Signal')
plt.scatter(data.index[data['Sell']], data['Close'][data['Sell']], marker='v', color='r', label='Sell Signal')
plt.title(f'{ticker} Close Price with Buy/Sell Signals')
plt.legend()

# MACD and Signal
plt.subplot(5, 1, 2)
plt.plot(data['MACD'], label='MACD', color='b')
plt.plot(data['Signal'], label='Signal Line', color='r')
plt.bar(data.index, data['Histogram'], label='Histogram', color='gray')
plt.title('MACD Indicator')
plt.legend()

# RSI
plt.subplot(5, 1, 3)
plt.plot(data['RSI'], label='RSI', color='m')
plt.axhline(30, linestyle='--', color='green')
plt.axhline(70, linestyle='--', color='red')
plt.title('RSI Indicator')
plt.legend()

# Stochastic Oscillator
plt.subplot(5, 1, 4)
plt.plot(data['Stoch_K'], label='%K', color='blue')
plt.plot(data['Stoch_D'], label='%D', color='red')
plt.axhline(80, linestyle='--', color='red')
plt.axhline(20, linestyle='--', color='green')
plt.title('Stochastic Oscillator')
plt.ylim(0, 100)  # Stochastic ranges from 0-100
plt.legend()

# Aroon Indicator
plt.subplot(5, 1, 5)
plt.plot(data['Aroon_Up'], label='Aroon Up', color='green')
plt.plot(data['Aroon_Down'], label='Aroon Down', color='red')
plt.axhline(70, linestyle='--', color='gray')
plt.axhline(30, linestyle='--', color='gray')
plt.title('Aroon Indicator')
plt.ylim(0, 100)  # Aroon ranges from 0-100
plt.legend()

plt.tight_layout()
plt.show()
