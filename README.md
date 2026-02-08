# 📈 Multi-Stock Price Prediction Model

> **Improved version that works for ANY stock, not just one!**

An advanced stock price prediction model using Random Forest with comprehensive technical indicators. Inspired by modern ML approaches but enhanced to support multiple stocks simultaneously.

## ⚡ Quick Setup (30 seconds)

```bash
# 1. Get FREE API key: https://www.alphavantage.co/support/#api-key

# 2. Set the API key (Windows PowerShell)
$env:ALPHA_VANTAGE_API_KEY="your_api_key_here"

# 3. Install and run
pip install -r requirements.txt
python main.py
```

**Or use the setup wizard:**
```bash
python setup.py
```

## 🌟 Key Features

- **Universal Stock Support**: Works with ANY stock ticker (AAPL, NVDA, TSLA, etc.)
- **Advanced Technical Indicators**: 50+ features including:
  - Moving Averages (MA5, MA10, MA20, MA50, MA200)
  - Exponential Moving Averages (EMA12, EMA26)
  - MACD, RSI, Bollinger Bands
  - Stochastic Oscillator, ATR, ADX
  - Volume indicators and more
- **Time Series Cross-Validation**: Proper chronological data splitting
- **Interactive Visualizations**: Beautiful charts and comparisons
- **Batch Predictions**: Analyze multiple stocks at once
- **Data Caching**: Faster subsequent runs
- **Model Persistence**: Save and load trained models

## 🚀 Quick Start

### Installation

1. **Clone or download this project**

2. **Get your FREE Alpha Vantage API key:**
   - Visit: https://www.alphavantage.co/support/#api-key
   - Takes less than 20 seconds to get
   - It's completely free!

3. **Set your API key:**

   **Option 1: Environment Variable (Recommended)**
   ```bash
   # Windows PowerShell
   $env:ALPHA_VANTAGE_API_KEY="your_api_key_here"
   
   # Windows CMD
   set ALPHA_VANTAGE_API_KEY=your_api_key_here
   
   # Linux/Mac
   export ALPHA_VANTAGE_API_KEY=your_api_key_here
   ```
   
   **Option 2: .env file**
   ```bash
   # Copy the example file
   copy .env.example .env
   
   # Edit .env and add your key
   ALPHA_VANTAGE_API_KEY=your_api_key_here
   ```

4. **Install dependencies:**
```bash
pip install -r requirements.txt
```

5. **Run the program:**
```bash
python main.py
```

### Usage Examples

#### Option 1: Interactive Mode
```bash
python main.py
```
Follow the prompts to select stocks and time periods.

#### Option 2: Command Line
```bash
# Predict multiple stocks
python main.py AAPL NVDA MSFT

# Single stock
python main.py TSLA
```

#### Option 3: Python Code
```python
from main import predict_single_stock, predict_portfolio

# Single stock prediction
result = predict_single_stock('AAPL', period='1y')

# Multiple stocks
results = predict_portfolio(['AAPL', 'NVDA', 'GOOGL'], period='2y')
```

## 📊 Example Output

```
==================================================================
Multi-Stock Price Prediction
==================================================================

──────────────────────────────────────────────────────────────
Processing AAPL
──────────────────────────────────────────────────────────────
Fetching data for AAPL...
✓ Fetched 252 days of data
Engineering features...
✓ Created 78 features

Training Random Forest model...
Training set: 201 samples
Testing set: 51 samples

==================================================
Model Performance for AAPL
==================================================
Training RMSE: $2.34
Testing RMSE:  $3.67
Training MAE:  $1.82
Testing MAE:   $2.91
Training R²:   0.9876
Testing R²:    0.9654
==================================================

📊 Prediction for AAPL
Current Price: $182.45
Predicted Price (Next Day): $184.23
Expected Change: $1.78 (+0.98%)
```

## 🏗️ Project Structure

```
price prediction/
│
├── main.py                    # Main application entry point
├── stock_predictor.py         # Core prediction model
├── data_manager.py            # Data fetching and caching
├── feature_engineering.py     # Technical indicators
├── visualizer.py              # Visualization tools
├── requirements.txt           # Dependencies
├── README.md                  # This file
│
├── data/                      # Created automatically
│   └── cache/                 # Cached stock data
│
├── models/                    # Saved models (optional)
│   └── AAPL_model.joblib
│
└── plots/                     # Saved plots (optional)
    └── AAPL_20260118.png
```

## 🔧 Advanced Usage

### Custom Stock Predictor

```python
from stock_predictor import StockPredictor

# Create predictor
predictor = StockPredictor('AAPL', period='2y')

# Fetch and prepare data
predictor.fetch_data()
predictor.engineer_features()

# Train model
metrics = predictor.train(n_estimators=200, max_depth=15)

# Predict next day
prediction = predictor.predict_next_day()

# Predict multiple days (experimental)
future_predictions = predictor.predict_future(days=30)

# Save model
predictor.save_model('models/my_aapl_model.joblib')

# Load model later
predictor.load_model('models/my_aapl_model.joblib')
```

### Feature Engineering

```python
from feature_engineering import FeatureEngineer
import yfinance as yf

# Fetch data
data = yf.Ticker('NVDA').history(period='1y')

# Create all features
engineer = FeatureEngineer()
data_with_features = engineer.create_all_features(data)

print(f"Created {len(data_with_features.columns)} features!")
```

### Data Management

```python
from data_manager import DataManager

manager = DataManager()

# Fetch multiple stocks
stocks_data = manager.fetch_multiple_stocks(
    ['AAPL', 'GOOGL', 'MSFT'], 
    period='2y'
)

# Get stock information
info = manager.get_stock_info('AAPL')
print(f"Company: {info['name']}")
print(f"Sector: {info['sector']}")

# Clear cache
manager.clear_cache('AAPL')  # Clear specific stock
manager.clear_cache()         # Clear all
```

### Visualizations

```python
from visualizer import StockVisualizer
import matplotlib.pyplot as plt

viz = StockVisualizer()

# Plot price history
viz.plot_price_history(data, ticker='AAPL')

# Plot predictions
viz.plot_predictions(y_true, y_pred, dates, ticker='AAPL')

# Plot technical indicators
viz.plot_technical_indicators(data, ticker='AAPL')

# Feature importance
viz.plot_feature_importance(feature_names, importances, ticker='AAPL')

plt.show()
```

## 📈 Supported Technical Indicators

### Trend Indicators
- Simple Moving Averages (SMA): 5, 10, 20, 50, 200 days
- Exponential Moving Averages (EMA): 12, 26, 50 days
- MACD (Moving Average Convergence Divergence)

### Momentum Indicators
- RSI (Relative Strength Index)
- Stochastic Oscillator
- Rate of Change (ROC)
- Momentum

### Volatility Indicators
- Bollinger Bands
- ATR (Average True Range)
- Standard Deviation

### Volume Indicators
- Volume Moving Averages
- On-Balance Volume (OBV)
- Volume-Weighted Average Price (VWAP)
- Price-Volume Trend

### Pattern Recognition
- Candlestick patterns (body, shadows, ratios)
- Bullish/Bearish detection
- Doji patterns

## 🎯 Model Performance Tips

1. **More data is better**: Use longer periods (2y or 5y) for more reliable predictions
2. **Cross-validation**: Always check cross-validation scores
3. **Feature importance**: Look at which features matter most
4. **Sector-specific models**: Train separate models for different sectors
5. **Regular retraining**: Retrain models weekly/monthly with new data

## ⚠️ Important Notes

### Limitations
- **Not financial advice**: This is an educational tool
- **Past performance ≠ future results**: Market conditions change
- **Short-term predictions only**: Most reliable for next 1-5 days
- **Market hours**: Works best with intraday or daily data
- **External factors**: Doesn't account for news, events, sentiment

### Best Practices
- Always do your own research
- Use multiple indicators and models
- Consider fundamental analysis alongside technical
- Test on historical data before live use
- Keep models updated with recent data

## 🔄 Improvements Over Basic Models

This model addresses several concerns raised in the Instagram comments:

1. ✅ **Chronological Split**: Uses proper time-series train/test split
2. ✅ **Time Series CV**: Implements `TimeSeriesSplit` for validation
3. ✅ **Feature Diversity**: Goes beyond just moving averages
4. ✅ **Multiple Stocks**: Works for any ticker, not just one
5. ✅ **Collinearity Handling**: Uses momentum and ROC features
6. ✅ **Performance Metrics**: Comprehensive evaluation (RMSE, MAE, R²)

## 🚧 Future Enhancements

Potential improvements you can add:

- [ ] LSTM/Transformer models for better sequence learning
- [ ] Sentiment analysis from news/social media
- [ ] Fundamental indicators (P/E ratio, earnings, etc.)
- [ ] Real-time predictions with live data
- [ ] Web dashboard with Streamlit
- [ ] Portfolio optimization
- [ ] Backtesting framework
- [ ] Multi-timeframe analysis

## 📚 Dependencies

- **pandas**: Data manipulation
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning models
- **alpha-vantage**: Stock data fetching (Free API)
- **requests**: HTTP library
- **matplotlib/seaborn**: Visualizations
- **joblib**: Model persistence

### API Rate Limits (Free Tier)
- Alpha Vantage Free: 25 calls/day, 5 calls/minute
- The code automatically handles rate limiting with delays
- Cached data reused for 24 hours

## 🤝 Contributing

Feel free to:
- Add new technical indicators
- Implement different ML models (LSTM, XGBoost, etc.)
- Improve visualizations
- Add more features
- Fix bugs

## 📝 License

This project is for educational purposes. Use at your own risk.

## 🎓 Learning Resources

- [Technical Analysis Explained](https://www.investopedia.com/technical-analysis-4689657)
- [Machine Learning for Trading](https://www.coursera.org/learn/machine-learning-trading)
- [Time Series Forecasting](https://otexts.com/fpp3/)

## 💡 Example Use Cases

1. **Portfolio Monitoring**: Track predictions for your portfolio stocks
2. **Sector Analysis**: Compare predictions across different sectors
3. **Strategy Development**: Test different technical indicators
4. **Learning Tool**: Understand how ML applies to finance
5. **Research**: Study market patterns and correlations

## 📞 Support

For questions or issues:
1. Check existing documentation
2. Review example code
3. Experiment with different parameters
4. Consult financial ML resources

---

**Happy Predicting! 📊🚀**

*Remember: This is a tool for learning and analysis. Always consult with financial advisors for investment decisions.*
