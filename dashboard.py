### NOTE: Set yfinance to the following version to get chart working: "pip install yfinance==0.2.40"

import streamlit as st 
import yfinance as yf
import pandas as pd 
import plotly.graph_objects as go
import ollama
import tempfile
import base64
import os
from datetime import datetime
import tensorflow as tf 
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import ta
from plotly.subplots import make_subplots

# Set up Streamlit app
st.set_page_config(page_title="Stock Dashboard", layout="wide", page_icon=':chart_with_upwards_trend:')
st.title('🤖 AI-Powered Technical Analysis Dashboard')
st.sidebar.header('Configuration')

# Input for stock ticker and date range
ticker = st.sidebar.text_input('Enter Stock or Crypto Ticker, for Stock (e.g. TSLA) and for Crypto (e.g. BTC-USD). You can check in yahoo finance for more options.')
start_date = st.sidebar.date_input('Start Date', value=pd.to_datetime('2023-01-01'))
end_date = st.sidebar.date_input('End Date', value=pd.to_datetime(datetime.now().strftime('%Y-%m-%d')))
forecast_days = st.sidebar.number_input('Forecast Days', min_value=1, max_value=365, value=30)



# Forecasting function
def create_lstm_forecast(data, forecast_days=30):
    # Prepare data for LSTM
    close_prices = data['Close'].values.reshape(-1, 1)
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(close_prices)
    
    # Create sequences
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        return np.array(X), np.array(y)

    seq_length = 60 # Use 60 days of historical data to predict the next day
    X, y = create_sequences(scaled_prices, seq_length)
    
    # Reshape input for LSTM [samples, time steps, features]
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    # Build LSTM Model
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, activation='relu', input_shape=(seq_length, 1), return_sequences=True),
        tf.keras.layers.LSTM(50, activation='relu'),
        tf.keras.layers.Dense(25),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    # Train model
    model.fit(X, y, epochs=50, batch_size=32, verbose=0)
    
    # Prepare input for forecasting
    last_sequence = scaled_prices[-seq_length:]
    
    # Generate forecast
    forecasted_scaled = []
    current_sequence = last_sequence.reshape((1, seq_length, 1))
    
    for _ in range(forecast_days):
        next_pred = model.predict(current_sequence)[0]
        forecasted_scaled.append(next_pred)
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[0, -1, 0] = next_pred
        
    # Inverse transform to get actual prices
    forecasted_prices = scaler.inverse_transform(np.array(forecasted_scaled))
    
    # Generate forecast dates
    last_date = data.index[-1]
    forecast_dates = pd.date_range(start=last_date, periods=forecast_days+1)[1:]
    
    return forecast_dates, forecasted_prices.flatten()



# Technical analysis
def calculate_technical_analysis(data):
    """Calculate multiple technical indicators for a given DataFrame.

    Args:
        data (pd.DataFrame): DataFrame with OHLC data
    """
    # RSI (Relative Strength Index)
    data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
    
    # MACD (Moving Average Convergence Divergence)
    macd = ta.trend.MACD(data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    
    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close'])
    data['Stoch_K'] = stoch.stoch()
    data['Stoch_D'] = stoch.stoch_signal()
    
    # ADX (Average Directional Index)
    adx = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close'])
    data['ADX'] = adx.adx()
    
    # Bollinger Bands with Volatility
    bb = ta.volatility.BollingerBands(data['Close'])
    data['BB_High'] = bb.bollinger_hband()
    data['BB_Low'] = bb.bollinger_lband()
    data['BB_Volatility'] = bb.bollinger_pband()
    
    return data 




# Fetch stock or crypto data using yfinance
if st.sidebar.button('Fetch Data'):
    st.session_state['data'] = yf.download(ticker, start=start_date, end=end_date)
    st.success('Data fetched successfully!')


# Check if data is available
if 'data' in st.session_state:
    data = st.session_state['data']
    data = calculate_technical_analysis(data)
    
    # Plot candlestick chart
    fig = go.Figure(data=[
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Candlestick'
        )
    ])
    
    
    
    # Sidebar: Select technical indicators
    st.sidebar.subheader('Technical Indicators')
    indicators = st.sidebar.multiselect(
        'Select Indicators:',
        ['20-Day SMA', '20-Day EMA', '20-Day Bollinger Bands', 'VWAP'],
        default=['20-Day SMA']
    )
    
    st.sidebar.subheader('Advanced Technical Indicators')
    advanced_indicators = st.sidebar.multiselect(
        'Select Advanced Indicators:',
        ['RSI', 'MACD', 'Stochastic Oscillator', 'ADX', 'Bollinger Bands'],
        default=['RSI', 'MACD']
    ) 
    
    fig_advanced = make_subplots(
        rows=len(advanced_indicators),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=advanced_indicators
    )
    
 
    
    def add_advanced_indicator(enum, indicator):
        if indicator == 'RSI':
            fig_advanced.add_trace(
                go.Scatter(x=data.index, y=data['RSI'], mode='lines', name='RSI'),
                row=enum, col=1
            )
            # Add RSI overbought/oversold lines
            fig_advanced.add_hline(y=70, line_dash='dash', line_color='red', row=enum, col=1)
            fig_advanced.add_hline(y=30, line_dash='dash', line_color='green', row=enum, col=1)
        
        elif indicator == 'MACD':
            fig_advanced.add_trace(
                go.Scatter(x=data.index, y=data['MACD'], mode='lines', name='MACD'),
                row=enum, col=1
            )
            fig_advanced.add_trace(
                go.Scatter(x=data.index, y=data['MACD_Signal'], mode='lines', name='MACD Signal'),
                row=enum, col=1
            )
        
        elif indicator == 'Stochastic Oscillator':
            fig_advanced.add_trace(
                go.Scatter(x=data.index, y=data['Stoch_K'], mode='lines', name='Stochastic %K'),
                row=enum, col=1
            )
            fig_advanced.add_trace(
                go.Scatter(x=data.index, y=data['Stoch_D'], mode='lines', name='Stochastic %D'),
                row=enum, col=1
            )
        
        elif indicator == 'ADX':
            fig_advanced.add_trace(
                go.Scatter(x=data.index, y=data['ADX'], mode='lines', name='ADX'),
                row=enum, col=1
            )
        
        elif indicator == 'Bollinger Bands':
            fig_advanced.add_trace(
                go.Scatter(x=data.index, y=data['BB_High'], mode='lines', name='BB Upper'),
                row=enum, col=1
            )
            fig_advanced.add_trace(
                go.Scatter(x=data.index, y=data['BB_Low'], mode='lines', name='BB Lower'),
                row=enum, col=1
            )
    
    # Helper function to add indicators to the chart
    def add_indicator(indicator):
        if indicator == '20-Day SMA':
            sma = data['Close'].rolling(window=20).mean()
            fig.add_trace(go.Scatter(x=data.index, y=sma, mode='lines', name='SMA (20)'))
        elif indicator == '20-Day EMA':
            ema = data['Close'].ewm(span=20).mean()
            fig.add_trace(go.Scatter(x=data.index, y=ema, mode='lines', name='EMA (20)'))
        elif indicator == '20-Day Bollinger Bands':
            sma = data['Close'].rolling(window=20).mean()
            std = data['Close'].rolling(window=20).std()
            bb_upper = sma + 2 * std 
            bb_lower = sma - 2 * std 
            fig.add_trace(go.Scatter(x=data.index, y=bb_upper, mode='lines', name='BB Upper'))
            fig.add_trace(go.Scatter(x=data.index, y=bb_lower, mode='lines', name='BB Lower'))
        elif indicator == 'VWAP':
            data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
            fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], mode='lines', name='VWAP'))
        
    
    # Add selected indicators to the chart
    for indicator in indicators:
        add_indicator(indicator)
        
    
    for i, indicator in enumerate(advanced_indicators, start=1):
        add_advanced_indicator(i, indicator)
        
    
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig_advanced.update_layout(height=300*len(advanced_indicators), title='Advanced Technical Analysis')
    st.plotly_chart(fig)
    st.plotly_chart(fig_advanced)
    
    
    
  
    
    
    
    # Forecasting with LSTM
    st.subheader('Forecasting with LSTM')
    if st.button('Run Forecasting'):
        with st.spinner('Generating LSTM Forecast...'):
            try:
                # Ensure you have sufficient historical data
                if len(data) < 100: # Minimum recommended data points
                    st.warning('Not enough historical data for reliable forecast. Please select a longer time range.')
                    

                forecast_dates, forecast_prices = create_lstm_forecast(data, forecast_days)
                
                # Create forecast plot
                forecast_fig = go.Figure()
                
                # Original price data
                forecast_fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['Close'],
                    mode='lines',
                    name='Historical Prices'
                ))
                
                # Forecasted price data
                forecast_fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=forecast_prices,
                    mode='lines',
                    name='LSTM Forecast',
                    line=dict(color='red', dash='dot')
                ))
                
                forecast_fig.update_layout(
                    title='Stock Price with LSTM Forecast',
                    xaxis_title='Date',
                    yaxis_title='Price'
                )
                
                st.plotly_chart(forecast_fig)
                
                # Display forecast metrics
                st.subheader('Forecast Summary')
                st.write(f"Forecast for next {len(forecast_dates)} days:")
                forecast_df = pd.DataFrame({
                    'Date': forecast_dates,
                    'Forecasted Price': forecast_prices
                })
                st.dataframe(forecast_df)

            except Exception as e:
                st.error(f"An error occurred while generating the forecast: {e}")

    
    # Analyze chart with LLaMA 3.2 Vision
    st.subheader('AI-Powered Analysis')
    if st.button('Run AI Analysis'):
        with st.spinner('Analyzing the chart, please wait...'):
            # Save chart as a temporary image
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
                fig.write_image(tmpfile.name)
                tmpfile_path = tmpfile.name 
                
            
            # Read image and encode to Base64
            with open(tmpfile_path, 'rb') as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Prepare AI analysis request
            messages = [{
                'role': 'user',
                'content': """You are a financial analyst with expertise in technical analysis. Analyze the given chart and provide a detailed report on the market trends, potential entry and exit points, and any other relevant insights.
                    Analyze the chart's technical indicators and provide a buy/hold/sell recommendation.
                    Base your recommendation only on the candlestick chart and the displayed technical indicators.
                    First, provide the recommendation, then, provide your detailed reasoning.
                """,
                'images': [image_data]
            }]
            response = ollama.chat(model='llama3.2-vision', messages=messages) # Replace 'model' with the appropriate model name for your Ollama instance
            
            # Display AI analysis
            st.write('**AI Analysis Results:**')
            st.write(response['message']['content'])
            
            # Clean up temporary file
            os.remove(tmpfile_path)