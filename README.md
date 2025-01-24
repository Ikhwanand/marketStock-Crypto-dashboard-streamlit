# 🤖 AI-Powered Financial Dashboard

## Overview
An advanced financial analysis dashboard built with Streamlit and Google AI, providing comprehensive insights for stocks, cryptocurrencies, and forex trading.

## 🌟 Features

### Market Analysis Modes
- Support for multiple trading types:
  - Stocks
  - Cryptocurrencies
  - Forex Trading

### Technical Analysis
- Interactive Candlestick Charts
- Advanced Technical Indicators:
  - Simple Moving Average (SMA)
  - Exponential Moving Average (EMA)
  - Bollinger Bands
  - Volume-Weighted Average Price (VWAP)
  - Relative Strength Index (RSI)
  - Moving Average Convergence Divergence (MACD)
  - Stochastic Oscillator
  - Average Directional Index (ADX)

### AI-Powered Analysis
- Google Gemini AI Integration
- Comprehensive market trend analysis
- Sentiment analysis
- Investment strategy recommendations
- Risk assessment
- Potential entry/exit point identification

### Forecasting
- LSTM-based Price Forecasting
- Customizable forecast duration (1-365 days)
- Machine Learning Price Prediction

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
- google-generativeai
- PIL
- python-dotenv

### AI Services
- Google AI Studio API Key (required)

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

4. Set up environment variables:
- Copy `.env.example` to `.env`
- Add your Google AI Studio API Key

## 🖥 Running the Dashboard
```bash
streamlit run dashboard.py
```

## 📊 Usage Guide
1. Select Trading Type
   - Choose between Stocks, Cryptocurrencies, or Forex
2. Enter Trading Symbol
   - Supports various market symbols
   - Verify symbol accuracy on respective platforms
3. Select Date Range
   - Choose historical data timeframe
4. Technical Analysis
   - Select from multiple technical indicators
   - Customize chart visualization
5. AI-Powered Features
   - Run AI Analysis for comprehensive insights
   - Generate sentiment analysis
   - Get investment recommendations

## 🤖 AI Features
- Advanced chart interpretation
- Market sentiment analysis
- Strategic investment recommendations
- Risk assessment

## 🔒 Security Notes
- Protect your API keys
- Use environment variables
- Keep dependencies updated
- Use virtual environments

## 🚧 Limitations
- AI recommendations are predictive
- Requires reliable market data
- Performance varies with market conditions

## 📝 Notes
- Ensure stable internet connection
- API usage may incur costs
- Recommendations are for informational purposes only

## 🌐 Contributing
Contributions, issues, and feature requests are welcome!

## 📄 License
[Specify your license here]