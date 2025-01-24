'''File Contains Models 
1-FB Prophet 
2-ARIMAX 
3- LSTM
 '''


'''1- FB Prophet'''

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import plotly.express as px
import streamlit as st

def PROPHET_MODEL(data):
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=365)
    forecast = m.predict(future)

    # Plot forecast
    st.subheader("Forecast Plot")
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    # Plot forecast components
    st.subheader("Forecast Components")
    fig2 = m.plot_components(forecast)
    st.write(fig2)


    # Display next 10 d"ays predictions
    next_10_days = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head(10)


    # Plot for next 10 days predictions
    fig_10_days = go.Figure()
    fig_10_days.add_trace(go.Scatter(x=next_10_days['ds'], y=next_10_days['yhat'], mode='lines+markers', name='Forecast'))
    fig_10_days.add_trace(go.Scatter(x=next_10_days['ds'], y=next_10_days['yhat_lower'], mode='lines', name='Lower Bound', line=dict(dash='dash')))
    fig_10_days.add_trace(go.Scatter(x=next_10_days['ds'], y=next_10_days['yhat_upper'], mode='lines', name='Upper Bound', line=dict(dash='dash')))
    fig_10_days.update_layout(title='Next 10 Days Forecast', xaxis_title='Date', yaxis_title='Forecast Value')
    st.plotly_chart(fig_10_days)

    next_10_days['ds'] = next_10_days['ds'].dt.strftime('%Y-%m-%d')
    st.subheader("Next 10 Days Predictions")
    st.write(next_10_days)



'''2- ARIMAX'''

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.graph_objs as go
import streamlit as st

def analyse_ARIMAX_MODEL(df):   
    # Convert 'Date' to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Stationarity test
    stationarity_results = {}
    for i in df.columns[:4]:
        result = sm.tsa.adfuller(df[i])
        stationarity_results[i] = {'ADF Statistic': result[0], 'p-value': result[1], 'Critical Values': result[4]}
        stationarity_results[i]['Stationary'] = result[1] <= 0.05

    # Train/Test split
    train = df[df.index.year < 2023]
    test = df[df.index.year >= 2023]

    # ARIMA model
    exogenous_features = ['Open', 'High', 'Low']
    model = sm.tsa.arima.ARIMA(endog=train['Close'], exog=train[exogenous_features], order=(1, 1, 1))
    model_fit = model.fit()
    train['Predictions'] = model_fit.predict()

    # Predictions on Test Set
    forecast = [model_fit.forecast(exog=test[exogenous_features].iloc[i]).values[0] for i in range(len(test))]
    test['Forecast'] = forecast

    # Predict the next 10 days
    future_dates = pd.date_range(start=test.index[-1] + pd.Timedelta(days=1), periods=10, freq='D')
    future_exog = test[exogenous_features].iloc[-10:].copy()
    future_forecast = model_fit.forecast(steps=10, exog=future_exog).values
    future_df = pd.DataFrame({'Forecast': future_forecast}, index=future_dates)
    
    return df, train, test, future_df, stationarity_results

def plot_forecast_ARIMAX(df, future_df):
    fig = go.Figure()

    # Historical data
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Historical Close'))

    # Future predictions
    fig.add_trace(go.Scatter(x=future_df.index, y=future_df['Forecast'], mode='lines', name='10-Day Forecast', line=dict(dash='dash', color='orange')))

    fig.update_layout(title='10-Day Forecast of Close Prices', xaxis_title='Date', yaxis_title='Close Price', legend=dict(x=0, y=1), template='plotly_white')

    return fig


def ARIMAX_MODEL(df):
    df, train, test, future_df, stationarity_results = analyse_ARIMAX_MODEL(df)

    st.write("Train/Test Split:")
    st.line_chart({'Train': train['Close'], 'Test': test['Close']})

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=train.index, y=train['Close'], mode='lines', name='Close', line=dict(color='blue')))
    fig3.add_trace(go.Scatter(x=train.index, y=train['Predictions'], mode='lines', name='Predictions', line=dict(color='green')))
    fig3.update_layout(
        title='Predictions on Training Set',
        xaxis_title='Date',
        yaxis_title='Price',
        legend_title='Legend',
        showlegend=True
    )
    st.plotly_chart(fig3)
    #st.line_chart({'Close': train['Close'], 'Predictions': train['Predictions']})


    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=test.index, y=test['Close'], mode='lines', name='Close', line=dict(color='blue')))
    fig4.add_trace(go.Scatter(x=test.index, y=test['Forecast'], mode='lines', name='Predictions', line=dict(color='green')))
    fig4.update_layout(
        title='Predictions on Test Set',
        xaxis_title='Date',
        yaxis_title='Price',
        legend_title='Legend',
        showlegend=True
    )
    st.plotly_chart(fig4)
    #st.line_chart({'Close': test['Close'], 'Forecast': test['Forecast']})

    forecast_fig = plot_forecast_ARIMAX(df, future_df)
    st.plotly_chart(forecast_fig)

    st.write("Future Predictions:")
    future_df = future_df.reset_index()
    future_df.rename(columns={'index': 'Date'}, inplace=True)
    future_df['Date'] = future_df['Date'].dt.strftime('%Y-%m-%d')
    st.write(future_df)



'''3- LSTM'''

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta


def analyse_LSTM_MODEL(stock_data):
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data = stock_data.sort_values('Date')

    stock = stock_data[['Date', 'Close', 'High', 'Low', 'Open', 'Volume']]

    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(stock[['Open', 'High', 'Low', 'Volume', 'Close']])

    train_data, test_data = train_test_split(normalized_data, test_size=0.2, shuffle=False)

    train_df = pd.DataFrame(train_data, columns=['Open', 'High', 'Low', 'Volume', 'Close'])
    test_df = pd.DataFrame(test_data, columns=['Open', 'High', 'Low', 'Volume', 'Close'])

    def generate_sequences(df, seq_length=50):
        X = df[['Open', 'High', 'Low', 'Volume', 'Close']].reset_index(drop=True)
        y = df[['Open', 'High', 'Low', 'Volume', 'Close']].reset_index(drop=True)

        sequences = []
        labels = []

        for index in range(len(X) - seq_length + 1):
            sequences.append(X.iloc[index : index + seq_length].values)
            labels.append(y.iloc[index + seq_length - 1].values)

        return np.array(sequences), np.array(labels)

    train_sequences, train_labels = generate_sequences(train_df)
    test_sequences, test_labels = generate_sequences(test_df)

    # Debug: Print the shapes of the sequences and labels
    print(f"Train sequences shape: {train_sequences.shape}")
    print(f"Train labels shape: {train_labels.shape}")
    print(f"Test sequences shape: {test_sequences.shape}")
    print(f"Test labels shape: {test_labels.shape}")

    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(50, 5)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(units=5)
    ])

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
    model.summary()

    # Training the model
    history = model.fit(train_sequences, train_labels, epochs=20, batch_size=32, validation_data=(test_sequences, test_labels), verbose=1)

    # Predictions
    train_predictions = model.predict(train_sequences)
    test_predictions = model.predict(test_sequences)

    # Plotting training data predictions
    fig = make_subplots(rows=1, cols=1, subplot_titles=('Close Predictions'))
    train_close_pred = train_predictions[:, 0]
    train_close_actual = train_labels[:, 0]
    fig.add_trace(go.Scatter(x=np.arange(len(train_close_actual)), y=train_close_actual, mode='lines', name='Actual', opacity=0.9))
    fig.add_trace(go.Scatter(x=np.arange(len(train_close_pred)), y=train_close_pred, mode='lines', name='Predicted', opacity=0.6))
    fig.update_layout(title='Close Price Predictions on Training Data', template='plotly_dark')
    
    # Plotting test data predictions
    fig_test = make_subplots(rows=1, cols=1, subplot_titles=('Close Predictions'))
    test_close_pred = test_predictions[:, 0]
    test_close_actual = test_labels[:, 0]
    fig_test.add_trace(go.Scatter(x=np.arange(len(test_close_actual)), y=test_close_actual, mode='lines', name='Actual', opacity=0.9))
    fig_test.add_trace(go.Scatter(x=np.arange(len(test_close_pred)), y=test_close_pred, mode='lines', name='Predicted', opacity=0.6))
    fig_test.update_layout(title='Close Price Predictions on Test Data', template='plotly_dark')
    # Next 10 days prediction
    latest_prediction = []
    last_seq = test_sequences[-1]

    for _ in range(10):
        prediction = model.predict(last_seq.reshape(1, last_seq.shape[0], last_seq.shape[1]))
        latest_prediction.append(prediction)
        last_seq = np.vstack((last_seq[1:], prediction))

    predicted_data_next = np.array(latest_prediction).squeeze()
    print(f"Predicted data next shape: {predicted_data_next.shape}")

    # Ensure correct shape for inverse transformation
    predicted_data_next = np.reshape(predicted_data_next, (-1, 5))

    # Inverse transform the predictions
    predicted_data_next = scaler.inverse_transform(predicted_data_next)

    # Generate dates for the next 10 days
    last_date = stock['Date'].max()
    next_dates = [last_date + timedelta(days=i) for i in range(1, 11)]

    # Create DataFrame for the next 10 days predictions
    next_10_days_df = pd.DataFrame({
        'Date': next_dates,
        'Predicted_Open': predicted_data_next[:, 0],
        'Predicted_High': predicted_data_next[:, 1],
        'Predicted_Low': predicted_data_next[:, 2],
        'Predicted_Volume': predicted_data_next[:, 3],
        'Predicted_Close': predicted_data_next[:, 4]
    })


    # Plotting the predicted prices for the next 10 days
    fig_future = go.Figure()
    fig_future.add_trace(go.Scatter(x=next_dates, y=predicted_data_next[4], mode='lines+markers', name='Predicted Close'))
    fig_future.update_layout(title="Next 10 Days Predicted Close Price", xaxis_title="Date", yaxis_title="Predicted Close Price", template="plotly_dark")
    
    return fig, fig_test, fig_future, next_10_days_df



def LSTM_MODEL(df):
        fig_train, fig_test, fig_future, predicted_data_next = analyse_LSTM_MODEL(df)

        st.plotly_chart(fig_train)

        st.plotly_chart(fig_test)

        st.plotly_chart(fig_future)

        predicted_data_next['Date'] = predicted_data_next['Date'].dt.strftime('%Y-%m-%d')
        st.write(predicted_data_next)
