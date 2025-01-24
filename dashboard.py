### NOTE: Set yfinance to the following version to get chart working: "pip install yfinance==0.2.40"

import streamlit as st 
import yfinance as yf
import pandas as pd 
import plotly.graph_objects as go
# import ollama
# import tempfile
# import base64
import os
from datetime import datetime
import tensorflow as tf 
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import ta
from plotly.subplots import make_subplots
from dotenv import load_dotenv
import google.generativeai as genai
from phi.agent import RunResponse
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.googlesearch import GoogleSearch 
from phi.tools.yfinance import YFinanceTools
import PIL.Image


load_dotenv()

# Configuration Gemini AI
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")




# Set up Streamlit app
st.set_page_config(page_title="Stock Dashboard", layout="wide", page_icon=':chart_with_upwards_trend:')
st.title('🤖 AI-Powered Technical Analysis Dashboard')
st.sidebar.header('Configuration')

# Input for stock ticker and date range
type_trading=st.sidebar.selectbox('Select type of trading', ('Crypto', 'Stock', 'Forex'))
ticker = st.sidebar.text_input('Enter Stock or Crypto Ticker, for Stock (e.g. TSLA) and for Crypto (e.g. BTC-USD). You can check in yahoo finance for more options and make sure the symbol is right.')
start_date = st.sidebar.date_input('Start Date', value=pd.to_datetime('2023-01-01'))
end_date = st.sidebar.date_input('End Date', value=pd.to_datetime(datetime.now().strftime('%Y-%m-%d')))
forecast_days = st.sidebar.number_input('Forecast Days', min_value=1, max_value=365, value=30)

# Ensure plot directory exists
PLOT_DIR = os.path.join(os.path.dirname(__file__), 'plot')
os.makedirs(PLOT_DIR, exist_ok=True)

def save_plot_to_folder(fig, filename):
    """
    Save a plot to the dedicated plot folder
    
    Args:
        fig (plotly.graph_objs._figure.Figure): Plotly figure to save
        filename (str): Filename for the plot
    
    Returns:
        str: Full path to the saved plot
    """
    # Ensure filename has .png extension
    if not filename.lower().endswith('.png'):
        filename += '.png'
    
    # Generate full path
    plot_path = os.path.join(PLOT_DIR, filename)
    
    # Save the plot
    fig.write_image(plot_path)
    
    return plot_path
    


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
            # If your model can read the image encode base64 you can uncomment the code below. In this case I use Gemini, Gemini can't read image based encode64
            # Save chart as a temporary image
            # with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
            #     fig.write_image(tmpfile.name)
            #     tmpfile_path = tmpfile.name 
                
            
            # # Read image and encode to Base64
            # with open(tmpfile_path, 'rb') as image_file:
            #     image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            
            # # Check if fig_advantages exists and prepare combined analysis
            # combined_images_data = [image_data]
            
            # # If fig_advantages exists, add it to the analysis
            # if 'fig_advanced' in locals():
            #     # Save fig_advantages to a temporary file
            #     with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile_advanced:
            #         fig_advanced.write_image(tmpfile_advanced.name)
            #         tmpfile_advanced_path = tmpfile_advanced.name
                    
            #     # Read advantages image and encode to Base64
            #     with open(tmpfile_advanced_path, 'rb') as advanced_image_file:
            #         advanced_image_data = base64.b64encode(advanced_image_file.read()).decode('utf-8')
                    
            #     # Add advantages image to combined images
            #     combined_images_data.append(advanced_image_data)
                
                    
            
            # Prepare comprehensive AI analysis request
            messages = [{
                'role': 'user',
                'content': f"""Financial Analysis Task: Comprehensive Investment Insights
                        Analyze the following images of {type_trading} {ticker} charts.
                        Use the saved chart images for your technical analysis.
                        
                        Provide a comprehensive analysis including:
                        - Market trend interpretation
                        - Technical indicator analysis
                        - Fundamental analysis
                        - Buy/Hold/Sell recommendation
                        - Risk assessment
                        - Potential entry/exit points
                        """,
                # 'images': [image_data for image_data in combined_images_data]
            }]
            
            # Add all images to the analysis request
            # You can uncomment if you use encode64 based image
            # for idx, img_data in enumerate(combined_images_data, 1):
            #     messages[0]['content'] += f"\n[Image {idx}]\n {img_data}]"
            
            # Uncomment this code below if you want to use ollama
            # response = ollama.chat(model='llama3.2-vision', messages=messages) # Replace 'model' with the appropriate model name for your Ollama instance
            
            plot_path1 = save_plot_to_folder(fig, f'{ticker}_chart.png')
            plot_path2 = save_plot_to_folder(fig_advanced, f'{ticker}_advanced_analysis_chart.png')
            
            chart1 = PIL.Image.open(plot_path1)
            chart2 = PIL.Image.open(plot_path2)
            

            # Add the plot paths to the messages
            
            response = model.generate_content([messages[0]['content'], chart1, chart2])
            
            # Display AI analysis
            st.write('**AI Analysis Results:**')
            st.write(response.text)
            
            # Clean up temporary file
            # os.remove(tmpfile_path)
        
    st.subheader('AI-Powered Sentiment Analysis')
    if st.button('Run Sentiment Analysis'):
        with st.spinner('Generating sentiment analysis...'):
            from sentiment_agent import agent_team
            type_trading_edit = ''
            if type_trading == 'Forex':
                type_trading_edit = 'currencies pair'
            elif type_trading == 'Crypto':
                type_trading_edit = 'cryptocurrency coin'
            elif type_trading == 'Stock':
                type_trading_edit = 'stock'
            
            # Sentiment Agent
            sentiment_agent = Agent(
                name="Sentiment Agent",
                role="Search and interpret news articles.",
                model=Gemini(id="gemini-1.5-flash", api_key=GOOGLE_API_KEY),
                tools=[GoogleSearch()],
                instructions=[
                    f"Find relevant news articles for {ticker} and analyze the sentiment.",
                    "Provide sentiment scores from 1 (negative) to 10 (positive) with reasoning and sources."
                    "Cite your sources. Be specific and provide links."
                ],
                show_tool_calls=True,
                markdown=True,
            )

            # Finance Agent
            finance_agent = Agent(
                name="Finance Agent",
                role="Get financial data and interpret trends.",
                model=Gemini(id="gemini-1.5-flash", api_key=GOOGLE_API_KEY),
                tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True)],
                instructions=[
                    "Retrieve stock prices, analyst recommendations, and key financial data.",
                    "Focus on trends and present the data in tables with key insights."
                ],
                show_tool_calls=True,
                markdown=True,
            )

            # Analyst Agent
            analyst_agent = Agent(
                name="Analyst Agent",
                role="Ensure thoroughness and draw conclusions.",
                model=Gemini(id="gemini-1.5-flash", api_key=GOOGLE_API_KEY),
                instructions=[
                    "Check outputs for accuracy and completeness.",
                    "Synthesize data to provide a final sentiment score (1-10) with justification."
                ],
                show_tool_calls=True,
                markdown=True,
            )

            # Team of Agents
            agent_team = Agent(
                model=Gemini(id="gemini-1.5-flash", api_key=GOOGLE_API_KEY),
                team=[sentiment_agent, finance_agent, analyst_agent],
                instructions=[
                    "Combine the expertise of all agents to provide a cohesive, well-supported response.",
                    "Always include references and dates for all data points and sources.",
                    "Present all data in structured tables for clarity.",
                    "Explain the methodology used to arrive at the sentiment scores."
                ],
                show_tool_calls=True,
                markdown=True,
            )
            
            from datetime import datetime
            date = datetime.now().strftime("%Y-%m-%d")

                
            response: RunResponse = agent_team.run(
                    f"""Analyze the sentiment for the following {type_trading_edit} for one week before the day {date}: {ticker} \n\n"
                    1. **Sentiment Analysis**: Search for relevant news articles and interpret the sentiment for each company. Provide sentiment scores on a scale of 1 to 10, explain your reasoning, and cite your sources.\n\n
                    2. **Financial Data**: Analyze stock price movements, analyst recommendations, and any notable financial data. Highlight key trends or events, and present the data in tables.\n\n
                    3. **Consolidated Analysis**: Combine the insights from sentiment analysis and financial data to assign a final sentiment score (1-10) for each company. Justify the scores and provide a summary of the most important findings.\n\n
                    Ensure your response is accurate, comprehensive, and includes references to sources with publication dates.""",
                    stream=False
                )
            
            st.write('**AI Sentiment Analysis Results:**')
            st.write(response.content)
            