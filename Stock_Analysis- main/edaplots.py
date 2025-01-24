# eda1.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def plot_stock_data(stock_data,date_col='Date', price_col='Close', default_year=None):    

    # Plot Closing Prices Over Time
    fig = px.line(stock_data, x='Date', y='Close', title='Closing Prices Over Time')
    fig.update_xaxes(title='Date')
    fig.update_yaxes(title='Closing Price')
    fig.update_layout(template='plotly_dark')
    st.plotly_chart(fig)

    # Plot Stock Price Analysis with additional traces
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Open'], mode='lines+markers', name='Open'))
    fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['High'], mode='lines+markers', name='High'))
    fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Low'], mode='lines+markers', name='Low'))
    fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['Close'], mode='lines+markers', name='Close'))
    fig.update_layout(title='Stock Price Analysis', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)

    
    # Sort the DataFrame by date
    stockdata = stock_data.sort_values(by='Date')  
    # Calculate daily price changes
    stockdata['Daily Change'] = stockdata['Close'].diff()  
    # Drop rows with NaN values in 'Daily Change' (first row will have NaN)
    stockdata = stockdata.dropna(subset=['Daily Change'])  
    # Plot histogram
    fig = px.histogram(stockdata, x='Daily Change', title='Histogram of Daily Price Changes')
    fig.update_layout(xaxis_title='Daily Price Change', yaxis_title='Frequency')
    st.plotly_chart(fig)


    # Plot Candlestick Chart with Moving Average
    stock_data['20-day MA'] = stock_data['Close'].rolling(window=20).mean()
    fig = go.Figure(data=[go.Candlestick(
        x=stock_data['Date'],
        open=stock_data['Open'],
        high=stock_data['High'],
        low=stock_data['Low'],
        close=stock_data['Close'],
        name="Candlesticks",
        increasing_line_color='green',
        decreasing_line_color='red',
        line=dict(width=1),
        showlegend=False
    )])
    fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data['20-day MA'], mode='lines', name='20-day Moving Average', line=dict(color='rgba(255, 255, 0, 0.3)')))
    fig.update_layout(
        title="Stock Price Candlestick Chart with Moving Average",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark",
    )
    st.plotly_chart(fig)

    # Plot Daily Trading Volume
    fig = px.line(stock_data, x='Date', y='Volume', title='Daily Trading Volume')
    fig.update_xaxes(title='Date')
    fig.update_yaxes(title='Volume')
    fig.update_layout(template='plotly_dark')
    st.plotly_chart(fig)

    #mothly sales volume
    data = stock_data.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data.set_index(date_col, inplace=True)
    data.sort_index(inplace=True)
    # Unique years and default selection
    unique_years = data.index.year.unique()
    selected_year = st.session_state.get('selected_year', unique_years[0] if default_year is None else default_year)
    selected_year = st.selectbox("Select Year", options=unique_years, index=list(unique_years).index(selected_year))
    # Filter data for the selected year
    filtered_data = data[data.index.year == selected_year]
    # Resample data to get monthly closing prices
    monthly_sales = filtered_data.resample('M').sum().dropna()
    # All months of the selected year
    all_months = pd.date_range(start=f'{selected_year}-01-01', end=f'{selected_year}-12-31', freq='MS')
    # Plotting
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=monthly_sales.index,
        y=monthly_sales[price_col],
        mode='lines+markers+text',
        name='Monthly Closing Price',
        hoverinfo='x+y+text',
        marker=dict(size=10)
    ))

    for date, amount in zip(monthly_sales.index, monthly_sales[price_col]):
        fig.add_annotation(
            x=date,
            y=amount,
            text=f'{amount:.2f}',  # Formatting amount to 2 decimal places
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40
        )
    fig.update_layout(
        title=f'Monthly {price_col} Prices for {selected_year}',
        xaxis_title='Month',
        yaxis_title='Price',
        xaxis=dict(
            tickmode='array',
            tickvals=all_months,
            tickformat='%b',
            tickangle=-45
        ),
        yaxis=dict(title='Price'))
    st.plotly_chart(fig)