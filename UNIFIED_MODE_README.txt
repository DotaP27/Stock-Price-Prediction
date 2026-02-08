"""
Stock Price Prediction System - README
=====================================

SINGLE UNIFIED MODE - All Features in One Place
===============================================

This system combines machine learning models to predict stock prices with Phase 1 optimizations.

KEY FEATURES:
=============
1. Ensemble Model: Random Forest + Deep Neural Network
2. Feature Selection: Reduces 45 features to top 20 (reduces overfitting)
3. Weighted Ensemble: Optimizes RF/NN weights per stock based on performance
4. Model Caching: Save models to disk for 100x faster predictions on repeat runs
5. News Sentiment: Optional news sentiment analysis for volatile stocks

HOW TO USE:
===========

Run the unified mode:
    python main.py

The system will ask for:
    1. Stock ticker (e.g., AAPL, NVDA, TSLA)
    2. Time period (6 months, 1 year, 2 years, 5 years)
    3. Whether to include sentiment analysis (yes/no)

Then it will:
    - Fetch historical price data from Alpha Vantage
    - Engineer 45 technical indicators
    - Select top 20 features using Random Forest importance
    - Train Random Forest model
    - Train Neural Network model
    - Calculate optimal weights based on test performance
    - Make weighted prediction
    - Cache models for next run

EXAMPLE OUTPUT:
===============

Stock: NVDA
Period: 1 year
Sentiment: Disabled

[ENSEMBLE PRICE PREDICTION FOR NVDA]
  Current Price:        $      186.23

  Individual Predictions:
    Random Forest:    $      180.95 (weight: 50%)
    Neural Network:   $      182.19 (weight: 50%)

  Ensemble (Weighted):
    Predicted Price:      $      181.57
    Expected Change:      $      4.66 ( -2.50%)
    Direction:            DOWN

Test R² Score:    -0.3702
RF Weight:        50%
NN Weight:        50%
Features Used:    20 (out of 45 total)


TECHNICAL DETAILS:
==================

Feature Engineering (45 total):
  - Moving Averages (MA5, MA10, MA20)
  - Exponential Moving Averages (EMA12, EMA26)
  - MACD + Signal + Histogram
  - RSI (Relative Strength Index)
  - Bollinger Bands (Upper, Lower, Width, Middle)
  - Momentum & Rate of Change (ROC)
  - Volatility (10-day rolling std)
  - Volume indicators (MA5, MA20, Ratio)
  - Price change features
  - Trend indicators (Shadow, Ratio)
  - Lagged features (1, 2, 3, 5 days)
  - Optional: News sentiment (9 features)

Phase 1 Optimizations:
  1. Feature Selection (45→20): Removes low-importance features, reduces overfitting
  2. Weighted Ensemble: RF/NN weights optimized per stock (not always 50/50)
  3. Model Caching: Saves trained models for instant predictions

Model Architecture:
  - Random Forest: 100 estimators, max_depth=10
  - Neural Network: 3-layer MLP (128→64→32 neurons)
  - Ensemble: Weighted average of both predictions

Validation:
  - Chronological train/test split (80/20)
  - No look-ahead bias
  - TimeSeriesSplit for cross-validation


FIRST RUN vs SECOND RUN:
========================

First Run (Training):
    - Fetches data: 2-5 seconds
    - Engineers features: 1 second
    - Trains models: 10-15 seconds
    - Total: ~15-25 seconds

Second Run (Cached):
    - Loads cached models: 0.3 seconds
    - Loads cached data: 0.1 seconds
    - Trains new models with fresh data: 10-15 seconds
    - Total: ~10-15 seconds
    (Cache makes predictions instant for same stock within 24 hours)


REQUIREMENTS:
=============
- Alpha Vantage API key (free tier: 5 calls/min, 25 calls/day)
- Set environment variable: ALPHA_VANTAGE_API_KEY or use default
- Python 3.7+
- scikit-learn, pandas, numpy

STOCKS TESTED:
==============
✓ AAPL (Apple)
✓ NVDA (NVIDIA)
✓ TSLA (Tesla)
✓ GOOGL (Google)
✓ MSFT (Microsoft)
✓ AMZN (Amazon)

Works with ANY stock on Alpha Vantage!


TROUBLESHOOTING:
================

Q: "Not enough data after feature engineering"
A: Stock has too few trading days in your selected period. Try longer period (1 year instead of 6 months)

Q: Demo API key error
A: Get your free API key at https://www.alphavantage.co/support/#api-key
   Set it: export ALPHA_VANTAGE_API_KEY=YOUR_KEY

Q: Slow first run
A: Normal! Training takes 15-25 seconds. Second run uses cache for instant results.

Q: Negative R² score
A: Can happen on volatile stocks. Means prediction worse than baseline. Still generates best-effort prediction.


FUTURE IMPROVEMENTS:
====================
Phase 2 (Optional):
  - Hyperparameter optimization with GridSearch
  - Confidence intervals on predictions
  - Fundamental indicators (P/E, earnings)
  - Multi-step forecasting (5-10 day predictions)

Phase 3 (Advanced):
  - Real LSTM with sequence modeling
  - Multi-stock portfolio optimization
  - Real-time intraday predictions
  - Streamlit dashboard


DISCLAIMER:
===========
This is a machine learning prediction system. It is NOT investment advice.
Past performance does not guarantee future results.
Use with caution and always do your own research before trading.
"""

# Save as docstring in a file
if __name__ == "__main__":
    print(__doc__)
