import pandas as pd
import yfinance as yf
import numpy as np

# Function to fetch historical data
def fetch_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    data.reset_index(inplace=True)
    return data

# Example usage
ticker = 'AAPL'
start_date = '2022-01-01'
end_date = '2023-01-01'
stock_data = fetch_stock_data(ticker, start_date, end_date)
print(stock_data.head())

# Calculate Moving Averages
def calculate_moving_averages(data, short_window=20, long_window=50):
    data['Short_MA'] = data['Close'].rolling(window=short_window, min_periods=1).mean()
    data['Long_MA'] = data['Close'].rolling(window=long_window, min_periods=1).mean()
    return data

# Calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data

# Calculate MACD
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    data['MACD'] = data['Close'].ewm(span=short_window, adjust=False).mean() - data['Close'].ewm(span=long_window, adjust=False).mean()
    data['Signal_Line'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()
    return data

# Example usage
stock_data = calculate_moving_averages(stock_data)
stock_data = calculate_rsi(stock_data)
stock_data = calculate_macd(stock_data)
print(stock_data.head())

# Function to identify trends and generate signals
def generate_signals(data):
    data['Signal'] = 0
    data['Signal'][20:] = np.where(data['Short_MA'][20:] > data['Long_MA'][20:], 1, -1)
    data['Position'] = data['Signal'].diff()
    return data

# Example usage
stock_data = generate_signals(stock_data)
buy_signals = stock_data[stock_data['Position'] == 1]
sell_signals = stock_data[stock_data['Position'] == -1]
print("Buy Signals:\n", buy_signals[['Date', 'Close', 'Short_MA', 'Long_MA']])
print("Sell Signals:\n", sell_signals[['Date', 'Close', 'Short_MA', 'Long_MA']])



# Function to provide recommendations
def provide_recommendation(data):
    last_signal = data['Signal'].iloc[-1]
    if last_signal == 1:
        return "Recommendation: Buy"
    elif last_signal == -1:
        return "Recommendation: Sell"
    else:
        return "Recommendation: Hold"

# Example usage
recommendation = provide_recommendation(stock_data)
print(recommendation)