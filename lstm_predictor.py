"""
Deep Learning Neural Network for Stock Price Prediction
Uses scikit-learn MLPRegressor (avoids TensorFlow DLL issues on Windows)
Implements deep neural network with 3 hidden layers
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings('ignore')

from data_manager import DataManager
from news_features import NewsFeatureBuilder


class LSTMPredictor:
    """Neural Network for stock price prediction (LSTM-inspired)"""
    
    def __init__(self, ticker: str, period: str = "2y", api_key: str = None,
                 use_sentiment: bool = False, sentiment_days: int = 14):
        """
        Initialize Neural Network predictor
        
        Args:
            ticker: Stock symbol (e.g., 'AAPL', 'NVDA')
            period: Historical data period
            api_key: Alpha Vantage API key
        """
        self.ticker = ticker.upper()
        self.period = period
        self.model = None
        self.scaler_X = MinMaxScaler(feature_range=(0, 1))
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))
        self.data = None
        self.trained = False
        self.feature_columns = None  # Initialize for caching
        self.data_manager = DataManager(api_key=api_key)
        self.use_sentiment = use_sentiment
        self.sentiment_days = sentiment_days
        self.news_builder = NewsFeatureBuilder(api_key=api_key)
        
    def fetch_data(self):
        """Fetch stock data from Alpha Vantage"""
        print(f"Fetching data for {self.ticker}...")
        self.data = self.data_manager.fetch_stock_data(self.ticker, self.period)
        
        if self.data.empty:
            raise ValueError(f"No data found for ticker {self.ticker}")
        
        print(f"Fetched {len(self.data)} days of data")
        return self.data
    
    def engineer_features_nn(self):
        """Create features optimized for neural network training
        
        Uses a simpler feature set than Random Forest (less feature bloat)
        to prevent overfitting in neural networks.
        """
        print("Engineering neural network features...")
        df = self.data.copy()
        
        # ════════════════════════════════════════════════════════════════════════════════════════
        # TREND FEATURES - Capture momentum and direction
        # ════════════════════════════════════════════════════════════════════════════════════════
        
        # Moving averages at different time horizons
        df['MA5'] = df['Close'].rolling(window=5).mean()      # Short-term trend
        df['MA10'] = df['Close'].rolling(window=10).mean()    # Medium-term trend
        df['MA20'] = df['Close'].rolling(window=20).mean()    # Long-term trend
        
        # Relative Strength Index (momentum indicator)
        df['RSI'] = self._calculate_rsi(df['Close'])           # 0-100 scale
        
        # Volatility (standard deviation of returns)
        df['Volatility'] = df['Close'].rolling(window=10).std()  # Price variability
        
        # Daily percentage returns
        df['Daily_Return'] = df['Close'].pct_change()         # % change from previous day
        
        # ════════════════════════════════════════════════════════════════════════════════════════
        # SEQUENCE FEATURES - Previous prices for pattern recognition
        # ════════════════════════════════════════════════════════════════════════════════════════
        
        # Lagged features help neural network recognize temporal patterns
        for lag in [1, 2, 5]:
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)  # Prices from lag days ago
        
        df['Target'] = df['Close'].shift(-1)

        if self.use_sentiment:
            df['Date'] = df.index.normalize()
            sentiment_df = self.news_builder.build_daily_features(self.ticker, days=self.sentiment_days)
            if not sentiment_df.empty:
                df = df.merge(sentiment_df, on='Date', how='left')
                sentiment_cols = [c for c in df.columns if c.startswith('Sentiment_')] + ['Headline_Count', 'Relevance_Mean', 'Sentiment_3d_Mean', 'Sentiment_3d_Momentum']
                sentiment_cols = list(dict.fromkeys(sentiment_cols))
                existing = [c for c in sentiment_cols if c in df.columns]
                if existing:
                    df[existing] = df[existing].fillna(0)
            df = df.drop(columns=['Date'])

        df = df.dropna()
        
        if len(df) < 30:
            raise ValueError(f"Not enough data. Got {len(df)} samples, need at least 30.")
        
        self.data = df
        print(f"Created features ({len(df)} samples)")
        return df
    
    @staticmethod
    def _calculate_rsi(prices, period=14):
        """Calculate Relative Strength Index (RSI)
        
        RSI measures the magnitude of recent price changes to evaluate 
        overbought or oversold conditions.
        
        Args:
            prices: Series of price values
            period: Number of periods for RSI calculation (default: 14)
            
        Returns:
            Series of RSI values ranging from 0 to 100
            - RSI > 70: Overbought (potential sell signal)
            - RSI < 30: Oversold (potential buy signal)
        """
        # Calculate price changes from one period to the next
        price_changes = prices.diff()
        
        # Separate gains (positive changes) from losses (negative changes)
        upward_gains = (price_changes.where(price_changes > 0, 0)).rolling(window=period).mean()
        downward_losses = (-price_changes.where(price_changes < 0, 0)).rolling(window=period).mean()
        
        # Calculate RS (Relative Strength) ratio
        strength_ratio = upward_gains / downward_losses
        
        # Convert to RSI (0-100 scale)
        rsi = 100 - (100 / (1 + strength_ratio))
        return rsi
    
    def prepare_train_test(self, test_size: float = 0.2):
        """Prepare training and testing sets"""
        exclude_cols = ['Target', 'Open', 'High', 'Low', 'Close', 'Volume', 
                       'Dividends', 'Stock Splits']
        feature_columns = [col for col in self.data.columns 
                          if col not in exclude_cols]
        
        # Store feature columns for later use in prediction
        self.feature_columns = feature_columns
        
        X = self.data[feature_columns].values
        y = self.data['Target'].values.reshape(-1, 1)
        
        # Scale features
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        # Chronological split
        split_idx = int(len(X_scaled) * (1 - test_size))
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Testing set: {len(X_test)} samples")
        
        return X_train, X_test, y_train.ravel(), y_test.ravel()
    
    def train(self, epochs: int = 100, batch_size: int = 16, verbose: int = 0):
        """Train the neural network"""
        print("\n" + "="*70)
        print("TRAINING DEEP NEURAL NETWORK MODEL")
        print("="*70)
        
        self.engineer_features_nn()
        X_train, X_test, y_train, y_test = self.prepare_train_test()
        
        print("\nTraining started...\n")
        
        # Build multi-layer neural network (deep learning)
        # 3 hidden layers: 128 -> 64 -> 32 neurons
        self.model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),  # 3 hidden layers for deep learning
            activation='relu',
            solver='adam',
            learning_rate='adaptive',
            max_iter=epochs,
            batch_size=batch_size,
            early_stopping=True,
            validation_fraction=0.1,
            verbose=0,
            random_state=42,
            n_iter_no_change=20
        )
        
        self.model.fit(X_train, y_train)
        self.trained = True
        
        print("Training complete\n")
        
        # Evaluate
        print("="*70)
        print("MODEL EVALUATION")
        print("="*70)
        
        # Get predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Inverse transform
        y_train_actual = self.scaler_y.inverse_transform(y_train.reshape(-1, 1))
        y_train_pred_actual = self.scaler_y.inverse_transform(y_train_pred.reshape(-1, 1))
        y_test_actual = self.scaler_y.inverse_transform(y_test.reshape(-1, 1))
        y_test_pred_actual = self.scaler_y.inverse_transform(y_test_pred.reshape(-1, 1))
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train_actual, y_train_pred_actual))
        train_mae = mean_absolute_error(y_train_actual, y_train_pred_actual)
        train_r2 = r2_score(y_train_actual, y_train_pred_actual)
        
        test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred_actual))
        test_mae = mean_absolute_error(y_test_actual, y_test_pred_actual)
        test_r2 = r2_score(y_test_actual, y_test_pred_actual)
        
        # Display
        print("\nDEEP NEURAL NETWORK PERFORMANCE FOR", self.ticker)
        print("-"*60)
        print("Training: RMSE=${:.2f} MAE=${:.2f} R2={:.4f}".format(train_rmse, train_mae, train_r2))
        print("Testing : RMSE=${:.2f} MAE=${:.2f} R2={:.4f}".format(test_rmse, test_mae, test_r2))
        
        return {
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'train_r2': train_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_r2': test_r2
        }
    
    def predict_next_day(self, verbose: bool = True):
        """Predict next day's closing price"""
        if not self.trained:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Get last row of features
        exclude_cols = ['Target', 'Open', 'High', 'Low', 'Close', 'Volume', 
                       'Dividends', 'Stock Splits']
        feature_columns = [col for col in self.data.columns 
                          if col not in exclude_cols]
        
        X_last = self.data[feature_columns].iloc[[-1]].values
        X_last_scaled = self.scaler_X.transform(X_last)
        
        # Predict
        y_pred_scaled = self.model.predict(X_last_scaled)[0]
        predicted_price = self.scaler_y.inverse_transform([[y_pred_scaled]])[0][0]
        
        current_price = self.data['Close'].iloc[-1]
        predicted_change = predicted_price - current_price
        predicted_change_pct = (predicted_change / current_price) * 100
        
        if predicted_change > 0:
            direction = "UP"
            color_change = "+"
        else:
            direction = "DOWN"
            color_change = ""

        if verbose:
            print("\nNEURAL NETWORK PREDICTION FOR", self.ticker)
            print("-"*60)
            print(f"  Current Price:   ${current_price:12.2f}")
            print(f"  Predicted Price: ${predicted_price:12.2f}")
            print(f"  Expected Change: {color_change}${abs(predicted_change):10.2f} ({color_change}{predicted_change_pct:6.2f}%)")
            print(f"  Direction:       {direction}")
        
        return predicted_price
