# 🤖 AI-Powered Financial Dashboard

## Overview
This is an advanced financial analysis dashboard built with Streamlit, leveraging AI and machine learning techniques to provide comprehensive stock and cryptocurrency insights.

## 🌟 Features

### Technical Analysis
- Interactive Candlestick Charts
- Multiple Technical Indicators:
  - Simple Moving Average (SMA)
  - Exponential Moving Average (EMA)
  - Bollinger Bands
  - Volume-Weighted Average Price (VWAP)

### Advanced Technical Indicators
- Relative Strength Index (RSI)
- Moving Average Convergence Divergence (MACD)
- Stochastic Oscillator
- Average Directional Index (ADX)

### AI-Powered Forecasting
- LSTM-based Price Forecasting
- Customizable forecast duration (1-365 days)
- Machine Learning Price Prediction

### AI Analysis
- Chart analysis using advanced AI models

## 🛠 Prerequisites

### Software Requirements
- Python 3.8+
- pip
- Virtual Environment (recommended)

### Required Libraries
- streamlit
- yfinance
- plotly
- pandas
- numpy
- tensorflow
- scikit-learn
- ta-lib
- ollama

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/financial-dashboard.git
cd financial-dashboard
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
```

3. Install dependencies:
```bash 
pip install -r requirements.txt
```

## 🖥 Running the Dashboard
```bash
streamlit run dashboard.py
```

## 📊 Usage Guide
1. Enter Stock/Crypto Ticker
    * Supports stocks (e.g., TSLA) and cryptocurrencies (e.g., BTC-USD)
    * Check Yahoo Finance for available tickers
2. Select Date Range
    * Choose start and end dates for historical data
3. Technical Analysis
    * Select from various technical indicators
    * Customize chart view
4. Forecasting
    * Set forecast duration
    * Generate LSTM-based price predictions


## 🤖 AI Features
*   Advanced chart analysis
*   Machine learning price forecasting


## 🔒 Security Notes
* Ensure you have the latest security updates
* Use virtual environments
* Be cautious with API keys and sensitive information

## 🚧 Limitations
* Forecasts are predictive and not guaranteed
* Requires sufficient historical data
* Performance depends on market conditions

## NOTE: You have to install ollama and download the model locally to use this app.