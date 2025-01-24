# from datetime import date, timedelta
# from dateutil.relativedelta import relativedelta
# # Define the start date as four years ago from today, considering leap years
# TODAY = date.today().strftime("%Y-%m-%d")
# START = (date.today() - relativedelta(years=4)).strftime("%Y-%m-%d")
# print(START)
# print(TODAY)



"""#Stock market analysis"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Simulated historical stock data for demonstration purposes
dates = pd.date_range(start='2024-01-01', periods=180)
prices = np.random.normal(loc=100, scale=10, size=len(dates)) + np.cumsum(np.random.randn(len(dates)))

stock_data = pd.DataFrame({'Date': dates, 'Close': prices})
stock_data.set_index('Date', inplace=True)

# Function to calculate moving averages
def calculate_moving_averages(data, short_window=20, long_window=50):
    data['Short_MA'] = data['Close'].rolling(window=short_window, min_periods=1).mean()
    data['Long_MA'] = data['Close'].rolling(window=long_window, min_periods=1).mean()
    return data

# Function to identify trends based on moving averages
def identify_trends(data):
    data['Signal'] = 0
    data['Signal'][short_window:] = np.where(data['Short_MA'][short_window:] > data['Long_MA'][short_window:], 1, 0)
    data['Position'] = data['Signal'].diff()
    return data

# Function to plot stock data with moving averages
def plot_stock_data(data):
    plt.figure(figsize=(14, 7))
    plt.plot(data['Close'], label='Close Price')
    plt.plot(data['Short_MA'], label=f'Short-term MA ({short_window} days)')
    plt.plot(data['Long_MA'], label=f'Long-term MA ({long_window} days)')
    plt.title('Stock Price with Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.show()

# Example usage
short_window = 20
long_window = 50

stock_data = calculate_moving_averages(stock_data, short_window, long_window)
stock_data = identify_trends(stock_data)
plot_stock_data(stock_data)

# Display basic trend signals
buy_signals = stock_data[stock_data['Position'] == 1]
sell_signals = stock_data[stock_data['Position'] == -1]

print("Buy Signals:\n", buy_signals[['Close', 'Short_MA', 'Long_MA']])
print("Sell Signals:\n", sell_signals[['Close', 'Short_MA', 'Long_MA']])
