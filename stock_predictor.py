"""
Multi-Stock Price Prediction Model
Improved version that works for any stock, not just one
Uses Random Forest with multiple technical indicators
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data_manager import DataManager
from datetime import datetime, timedelta
import joblib
import warnings
warnings.filterwarnings('ignore')

from news_features import NewsFeatureBuilder


class StockPredictor:
    """Multi-stock price prediction model using Random Forest"""
    
    def __init__(self, ticker: str, period: str = "2y", api_key: str = None,
                 use_sentiment: bool = False, sentiment_days: int = 14,
                 use_feature_selection: bool = True, top_features: int = 20):
        """
        Initialize the stock predictor
        
        Args:
            ticker: Stock symbol (e.g., 'AAPL', 'NVDA', 'TSLA')
            period: Historical data period ('1y', '2y', '5y', 'max')
            api_key: Alpha Vantage API key (optional, uses env var if not provided)
            use_feature_selection: Enable top-N feature selection (Phase 1 optimization)
            top_features: Number of top features to retain (default: 20)
        """
        self.ticker = ticker.upper()
        self.period = period
        self.model = None
        self.feature_columns = None
        self.data = None
        self.trained = False
        self.data_manager = DataManager(api_key=api_key)
        self.use_sentiment = use_sentiment
        self.sentiment_days = sentiment_days
        self.news_builder = NewsFeatureBuilder(api_key=api_key)
        self.use_feature_selection = use_feature_selection
        self.top_features = top_features
        self.selected_features = None
        self.feature_importance_df = None
        
    def fetch_data(self):
        """Fetch stock data from Alpha Vantage"""
        print(f"Fetching data for {self.ticker}...")
        self.data = self.data_manager.fetch_stock_data(self.ticker, self.period)
        
        if self.data.empty:
            raise ValueError(f"No data found for ticker {self.ticker}")
        
        print(f"Fetched {len(self.data)} days of data")
        return self.data
    
    def engineer_features(self):
        """Create technical indicators and features (optimized for 100-day compact data)"""
        print("Engineering features...")
        df = self.data.copy()
        
        # ════════════════════════════════════════════════════════════════════════════════════════
        # TREND INDICATORS - Capture momentum and direction of price movement
        # ════════════════════════════════════════════════════════════════════════════════════════
        
        # Simple Moving Averages (short, medium, long-term trend indicators)
        df['MA5'] = df['Close'].rolling(window=5).mean()       # 1-week trend
        df['MA10'] = df['Close'].rolling(window=10).mean()     # 2-week trend
        df['MA20'] = df['Close'].rolling(window=20).mean()     # 1-month trend
        
        # Exponential Moving Averages (weight recent prices more heavily)
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()   # Fast EMA
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()   # Slow EMA
        
        # MACD (Moving Average Convergence Divergence) - momentum indicator
        df['MACD'] = df['EMA12'] - df['EMA26']                # Distance between EMAs
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()  # Signal line
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']  # MACD vs Signal
        
        # ════════════════════════════════════════════════════════════════════════════════════════
        # MOMENTUM INDICATORS - Measure strength of price movement
        # ════════════════════════════════════════════════════════════════════════════════════════
        
        # RSI (Relative Strength Index) - measures overbought/oversold conditions
        price_changes = df['Close'].diff()
        upward_gains = (price_changes.where(price_changes > 0, 0)).rolling(window=14).mean()
        downward_losses = (-price_changes.where(price_changes < 0, 0)).rolling(window=14).mean()
        strength_ratio = upward_gains / downward_losses
        df['RSI'] = 100 - (100 / (1 + strength_ratio))  # RSI ranges 0-100
        
        # Rate of Change (ROC) - percentage change over 10 days
        df['Momentum'] = df['Close'] - df['Close'].shift(10)   # Absolute change
        df['ROC'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100  # % change
        
        # ════════════════════════════════════════════════════════════════════════════════════════
        # VOLATILITY INDICATORS - Measure price stability and risk
        # ════════════════════════════════════════════════════════════════════════════════════════
        
        # Bollinger Bands (standard deviation bands around moving average)
        moving_average_20 = df['Close'].rolling(window=20).mean()
        standard_deviation = df['Close'].rolling(window=20).std()
        df['BB_Middle'] = moving_average_20                            # Middle band = MA
        df['BB_Upper'] = moving_average_20 + (standard_deviation * 2)  # Upper = MA + 2*STD
        df['BB_Lower'] = moving_average_20 - (standard_deviation * 2)  # Lower = MA - 2*STD
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']              # Band width = volatility
        
        # Volatility measurement (standard deviation of returns)
        df['Volatility'] = df['Close'].rolling(window=10).std()  # 10-day volatility
        
        # ════════════════════════════════════════════════════════════════════════════════════════
        # VOLUME INDICATORS - Analyze trading volume and liquidity
        # ════════════════════════════════════════════════════════════════════════════════════════
        
        # Volume Moving Averages
        df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()      # Recent volume trend
        df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()    # Long-term volume trend
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA20']         # Current vs average
        
        # ════════════════════════════════════════════════════════════════════════════════════════
        # PRICE CHANGE FEATURES - Capture different aspects of price behavior
        # ════════════════════════════════════════════════════════════════════════════════════════
        
        # Daily returns (percentage change from day to day)
        df['Daily_Return'] = df['Close'].pct_change()
        
        # Intraday price ratios (how price moved within the trading day)
        df['High_Low_Ratio'] = df['High'] / df['Low']              # Intraday range
        df['Close_Open_Ratio'] = df['Close'] / df['Open']          # Direction of day
        
        # ════════════════════════════════════════════════════════════════════════════════════════
        # CANDLESTICK PATTERNS - Capture patterns from candlestick analysis
        # ════════════════════════════════════════════════════════════════════════════════════════
        
        # Upper and lower shadows (wicks) indicate price rejection
        df['Upper_Shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)    # Resistance wick
        df['Lower_Shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']      # Support wick
        
        # ════════════════════════════════════════════════════════════════════════════════════════
        # LAGGED FEATURES - Previous price values for sequential patterns
        # ════════════════════════════════════════════════════════════════════════════════════════
        
        # Previous closing prices (help model recognize recurring patterns)
        for lag in [1, 2, 3, 5]:
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)  # Prices from 1,2,3,5 days ago
        
        # ════════════════════════════════════════════════════════════════════════════════════════
        # TARGET VARIABLE - What we're trying to predict
        # ════════════════════════════════════════════════════════════════════════════════════════
        
        # Next day's closing price (shifted -1 means look ahead)
        df['Target'] = df['Close'].shift(-1)  # Tomorrow's price

        # Optional: add news sentiment features
        if self.use_sentiment:
            # Fetch sentiment scores from recent news headlines
            df['Date'] = df.index.normalize()  # Prepare for merge
            sentiment_df = self.news_builder.build_daily_features(
                self.ticker, 
                days=self.sentiment_days
            )
            
            # Merge sentiment features with price data
            if not sentiment_df.empty:
                df = df.merge(sentiment_df, on='Date', how='left')
                
                # Identify all sentiment-related columns
                sentiment_cols = (
                    [c for c in df.columns if c.startswith('Sentiment_')] + 
                    ['Headline_Count', 'Relevance_Mean', 'Sentiment_3d_Mean', 'Sentiment_3d_Momentum']
                )
                # Remove duplicates while preserving order
                sentiment_cols = list(dict.fromkeys(sentiment_cols))
                # Fill missing sentiment values with neutral (0)
                existing = [c for c in sentiment_cols if c in df.columns]
                if existing:
                    df[existing] = df[existing].fillna(0)
            df = df.drop(columns=['Date'])  # Cleanup
        
        # Remove rows with NaN values (can occur after lagging/shifting)
        initial_rows = len(df)
        df = df.dropna()
        dropped_rows = initial_rows - len(df)
        if dropped_rows > 0:
            print(f"  Dropped {dropped_rows} rows with missing values")
        
        # Ensure we have minimum sample size for training
        minimum_samples = 25
        if len(df) < minimum_samples:
            raise ValueError(
                f"Not enough data after feature engineering. "
                f"Got {len(df)} samples, need at least {minimum_samples}."
            )
        
        self.data = df
        print(f"Created {len(df.columns)} features ({len(df)} samples)")
        return df
    
    def prepare_train_test(self, test_size: float = 0.2):
        """
        Prepare training and testing sets using chronological split
        (Addresses the time series split concern from Instagram comments)
        """
        # Select feature columns (exclude target and original OHLCV)
        exclude_cols = ['Target', 'Open', 'High', 'Low', 'Close', 'Volume', 
                       'Dividends', 'Stock Splits']
        all_features = [col for col in self.data.columns 
                       if col not in exclude_cols]
        
        # Apply feature selection if enabled
        if self.use_feature_selection and self.selected_features is not None:
            self.feature_columns = self.selected_features
            print(f"Using {len(self.feature_columns)} selected features (from {len(all_features)} total)")
        else:
            self.feature_columns = all_features
            print(f"Using all {len(self.feature_columns)} features")
        
        X = self.data[self.feature_columns]
        y = self.data['Target']
        
        # Chronological split (not random!)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Testing set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    def train(self, n_estimators: int = 100, max_depth: int = 10):
        """Train the Random Forest model"""
        print("\nTraining Random Forest model...")
        
        # First pass: train on all features to identify important ones
        if self.use_feature_selection and self.selected_features is None:
            print(f"Phase 1 optimization: Selecting top {self.top_features} features...")
            exclude_cols = ['Target', 'Open', 'High', 'Low', 'Close', 'Volume', 
                           'Dividends', 'Stock Splits']
            all_features = [col for col in self.data.columns if col not in exclude_cols]
            
            X_temp = self.data[all_features]
            y = self.data['Target']
            split_idx = int(len(X_temp) * 0.8)
            X_train_temp = X_temp[:split_idx]
            y_train_temp = y[:split_idx]
            
            # Train quick model to get feature importance
            temp_model = RandomForestRegressor(
                n_estimators=50, max_depth=8, random_state=42, n_jobs=-1
            )
            temp_model.fit(X_train_temp, y_train_temp)
            
            # Get top features
            self.feature_importance_df = pd.DataFrame({
                'Feature': all_features,
                'Importance': temp_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            self.selected_features = self.feature_importance_df.head(self.top_features)['Feature'].tolist()
            print(f"Selected {len(self.selected_features)} features for training")
        
        X_train, X_test, y_train, y_test = self.prepare_train_test()
        
        # Random Forest Regressor
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        self.model.fit(X_train, y_train)
        self.trained = True
        
        # Evaluate
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        print("\nMODEL PERFORMANCE FOR", self.ticker)
        print("-"*60)
        print("Training: RMSE=${:.2f} MAE=${:.2f} R2={:.4f}".format(train_rmse, train_mae, train_r2))
        print("Testing : RMSE=${:.2f} MAE=${:.2f} R2={:.4f}".format(test_rmse, test_mae, test_r2))
        
        if self.use_feature_selection and self.selected_features is not None:
            print(f"Features used: {len(self.selected_features)}/45 (Phase 1 optimization)")
        
        return {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
    
    def cross_validate(self, n_splits: int = 5):
        """
        Time series cross-validation
        (Addresses comment about proper time series validation)
        """
        print(f"\nPerforming {n_splits}-fold time series cross-validation...")
        
        X = self.data[self.feature_columns]
        y = self.data['Target']
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
            y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
            
            model = RandomForestRegressor(n_estimators=100, max_depth=10, 
                                         random_state=42, n_jobs=-1)
            model.fit(X_train_cv, y_train_cv)
            
            val_pred = model.predict(X_val_cv)
            rmse = np.sqrt(mean_squared_error(y_val_cv, val_pred))
            cv_scores.append(rmse)
            print(f"  Fold {fold}: RMSE = ${rmse:.2f}")
        
        print(f"\nAverage CV RMSE: ${np.mean(cv_scores):.2f} ± ${np.std(cv_scores):.2f}")
        return cv_scores
    
    def predict_next_day(self):
        """Predict the next day's closing price"""
        if not self.trained:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Get the latest data point
        latest = self.data[self.feature_columns].iloc[-1:].values
        prediction = self.model.predict(latest)[0]
        
        current_price = self.data['Close'].iloc[-1]
        predicted_change = prediction - current_price
        predicted_change_pct = (predicted_change / current_price) * 100
        
        # Format output
        direction = "📈 UP" if predicted_change > 0 else "📉 DOWN"
        color_change = "+" if predicted_change > 0 else ""
        
        print("\n" + "╔" + "═"*68 + "╗")
        print("║" + f"  🔮 PRICE PREDICTION FOR {self.ticker}".ljust(69) + "║")
        print("╠" + "═"*68 + "╣")
        print("║" + " "*68 + "║")
        print("║" + f"  Current Price:        ${current_price:>12.2f}".ljust(69) + "║")
        print("║" + f"  Predicted Price:      ${prediction:>12.2f}".ljust(69) + "║")
        print("║" + f"  Expected Change:      {color_change}${predicted_change:>10.2f} ({color_change}{predicted_change_pct:>6.2f}%)".ljust(69) + "║")
        print("║" + f"  Direction:            {direction}".ljust(69) + "║")
        print("║" + " "*68 + "║")
        print("╚" + "═"*68 + "╝")
        
        return {
            'current_price': current_price,
            'predicted_price': prediction,
            'predicted_change': predicted_change,
            'predicted_change_pct': predicted_change_pct
        }
    
    def predict_future(self, days: int = 30):
        """
        Predict multiple days into the future
        Note: Predictions become less reliable as we go further
        """
        if not self.trained:
            raise ValueError("Model not trained yet. Call train() first.")
        
        print(f"\nPredicting {days} days into the future...")
        
        predictions = []
        current_data = self.data[self.feature_columns].iloc[-1:].copy()
        last_price = self.data['Close'].iloc[-1]
        
        for day in range(1, days + 1):
            pred = self.model.predict(current_data.values)[0]
            predictions.append(pred)
            
            # Update features for next prediction (simplified)
            # In practice, you'd recalculate all technical indicators
            current_data = current_data.copy()
            # This is a simplified approach - actual implementation would be more complex
        
        return predictions
    
    def get_feature_importance(self, top_n: int = 15):
        """Get the most important features"""
        if not self.trained:
            raise ValueError("Model not trained yet. Call train() first.")
        
        importances = pd.DataFrame({
            'Feature': self.feature_columns,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print(f"\n╔" + "═"*68 + "╗")
        print("║" + f"  ⭐ TOP {top_n} IMPORTANT FEATURES FOR PREDICTION".ljust(69) + "║")
        print("╠" + "═"*68 + "╣")
        print("║" + " "*68 + "║")
        
        for idx, (_, row) in enumerate(importances.head(top_n).iterrows(), 1):
            feature_name = row['Feature'][:28].ljust(28)
            importance_val = row['Importance']
            bar_length = int(importance_val * 200)
            bar = "█" * bar_length
            print("║" + f"  {idx:2d}. {feature_name} {importance_val:7.4f}  {bar}".ljust(69) + "║")
        
        print("║" + " "*68 + "║")
        print("╚" + "═"*68 + "╝")
        
        return importances
    
    def save_model(self, filepath: str = None):
        """Save the trained model"""
        if not self.trained:
            raise ValueError("Model not trained yet. Call train() first.")
        
        if filepath is None:
            filepath = f"models/{self.ticker}_model.joblib"
        
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'ticker': self.ticker
        }
        
        joblib.dump(model_data, filepath)
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']
        self.ticker = model_data['ticker']
        self.trained = True
        print(f"✓ Model loaded from {filepath}")


def predict_multiple_stocks(tickers: list, period: str = "2y"):
    """
    Predict prices for multiple stocks at once
    This is the main improvement - works for any stock!
    """
    results = {}
    
    print(f"\n{'='*60}")
    print(f"Multi-Stock Price Prediction")
    print(f"{'='*60}\n")
    
    for ticker in tickers:
        try:
            print(f"\n{'─'*60}")
            print(f"Processing {ticker}")
            print(f"{'─'*60}")
            
            predictor = StockPredictor(ticker, period)
            predictor.fetch_data()
            predictor.engineer_features()
            metrics = predictor.train()
            prediction = predictor.predict_next_day()
            predictor.get_feature_importance(top_n=10)
            
            results[ticker] = {
                'predictor': predictor,
                'metrics': metrics,
                'prediction': prediction
            }
            
        except Exception as e:
            print(f"❌ Error processing {ticker}: {str(e)}")
            results[ticker] = {'error': str(e)}
    
    return results


if __name__ == "__main__":
    # Example usage - predict multiple stocks
    
    # Popular tech stocks
    tech_stocks = ['AAPL', 'NVDA', 'MSFT', 'GOOGL', 'TSLA']
    
    # You can also try other sectors
    # finance_stocks = ['JPM', 'BAC', 'GS', 'MS', 'WFC']
    # energy_stocks = ['XOM', 'CVX', 'COP', 'SLB', 'MPC']
    
    print("Starting Multi-Stock Price Prediction...")
    print("This model works for ANY stock ticker!\n")
    
    # Train and predict for multiple stocks
    results = predict_multiple_stocks(tech_stocks[:2], period="1y")  # Start with 2 for testing
    
    # Summary
    print("\n" + "="*60)
    print("PREDICTION SUMMARY")
    print("="*60)
    
    for ticker, result in results.items():
        if 'error' not in result:
            pred = result['prediction']
            print(f"\n{ticker}:")
            print(f"  Current: ${pred['current_price']:.2f}")
            print(f"  Predicted: ${pred['predicted_price']:.2f}")
            print(f"  Change: {pred['predicted_change_pct']:+.2f}%")
            print(f"  Model R²: {result['metrics']['test_r2']:.4f}")
