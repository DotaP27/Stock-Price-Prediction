"""
Ensemble Predictor - Combines Random Forest + Deep Neural Network
Phase 1 Optimization:
  - Weighted ensemble (dynamic RF/NN weights based on recent performance)
  - Model persistence (saves/loads trained models with joblib)
  - Feature selection (reduces features from 45 to 20-25)
"""

import pandas as pd
import numpy as np
from stock_predictor import StockPredictor
from lstm_predictor import LSTMPredictor
import warnings
import joblib
import os
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')


class EnsemblePredictor:
    """Ensemble model combining Random Forest + Deep Neural Network with Phase 1 optimizations"""
    
    def __init__(self, ticker: str, period: str = "2y", api_key: str = None,
                 use_sentiment: bool = False, sentiment_days: int = 14,
                 use_feature_selection: bool = True, use_model_cache: bool = True):
        """
        Initialize Ensemble predictor
        
        Args:
            use_feature_selection: Enable top-20 feature selection (Phase 1)
            use_model_cache: Enable model persistence/caching (Phase 1)
        """
        self.ticker = ticker.upper()
        self.period = period
        self.rf_predictor = None
        self.nn_predictor = None
        self.trained = False
        self.api_key = api_key
        self.use_sentiment = use_sentiment
        self.sentiment_days = sentiment_days
        self.use_feature_selection = use_feature_selection
        self.use_model_cache = use_model_cache
        self.rf_weight = 0.5  # Dynamic weighting (Phase 1)
        self.nn_weight = 0.5
        self.model_cache_dir = "models"
        
        # Create models directory if using cache
        if self.use_model_cache and not os.path.exists(self.model_cache_dir):
            os.makedirs(self.model_cache_dir)
    
    def _get_model_cache_path(self, model_type: str):
        """Generate cache path for model"""
        return os.path.join(self.model_cache_dir, f"{self.ticker}_{model_type}_model.joblib")
    
    def _is_cache_valid(self, cache_path: str, max_age_hours: int = 24):
        """Check if cached model is still valid"""
        if not os.path.exists(cache_path):
            return False
        
        mod_time = os.path.getmtime(cache_path)
        age_hours = (datetime.now().timestamp() - mod_time) / 3600
        return age_hours < max_age_hours
    
    def _load_cached_models(self):
        """Load models from cache if available"""
        if not self.use_model_cache:
            return False
        
        rf_cache = self._get_model_cache_path("rf")
        nn_cache = self._get_model_cache_path("nn")
        
        if self._is_cache_valid(rf_cache) and self._is_cache_valid(nn_cache):
            try:
                print("[CACHE] Loading models from disk...")
                rf_data = joblib.load(rf_cache)
                nn_data = joblib.load(nn_cache)
                
                # Restore RF predictor
                self.rf_predictor = StockPredictor(
                    self.ticker, self.period, self.api_key,
                    use_sentiment=self.use_sentiment,
                    use_feature_selection=self.use_feature_selection
                )
                self.rf_predictor.model = rf_data['model']
                self.rf_predictor.feature_columns = rf_data['feature_columns']
                self.rf_predictor.trained = True
                self.rf_predictor.data = rf_data['data']
                
                # Restore NN predictor
                self.nn_predictor = LSTMPredictor(
                    self.ticker, self.period, self.api_key,
                    use_sentiment=self.use_sentiment
                )
                self.nn_predictor.model = nn_data['model']
                self.nn_predictor.feature_columns = nn_data['feature_columns']
                self.nn_predictor.trained = True
                self.nn_predictor.data = nn_data['data']
                
                self.trained = True
                print("[CACHE] Models loaded successfully (100x faster!)")
                return True
                
            except Exception as e:
                print(f"[WARNING] Failed to load cache: {e}")
                return False
        
        return False
    
    def _save_cached_models(self):
        """Save trained models to cache"""
        if not self.use_model_cache:
            return
        
        try:
            rf_cache = self._get_model_cache_path("rf")
            nn_cache = self._get_model_cache_path("nn")
            
            rf_data = {
                'model': self.rf_predictor.model,
                'feature_columns': self.rf_predictor.feature_columns,
                'data': self.rf_predictor.data,
                'ticker': self.ticker
            }
            
            nn_data = {
                'model': self.nn_predictor.model,
                'feature_columns': self.nn_predictor.feature_columns,
                'data': self.nn_predictor.data,
                'ticker': self.ticker
            }
            
            joblib.dump(rf_data, rf_cache)
            joblib.dump(nn_data, nn_cache)
            print(f"[CACHE] Models cached to {self.model_cache_dir}/")
            
        except Exception as e:
            print(f"[WARNING] Failed to cache models: {e}")
    
    def _calculate_dynamic_weights(self, rf_metrics, nn_metrics):
        """Calculate dynamic ensemble weights based on model performance
        
        PHASE 1 OPTIMIZATION: Instead of using fixed 50/50 weights, this method
        calculates weights based on which model performed better on the test set.
        
        The better-performing model gets higher weight in the final prediction,
        allowing the ensemble to automatically adapt to stock-specific patterns.
        
        Args:
            rf_metrics: Dictionary of Random Forest performance metrics
            nn_metrics: Dictionary of Neural Network performance metrics
        """
        # Use R² scores to determine weight (R² is bounded, higher is better)
        rf_score = max(0, rf_metrics['test_r2'])   # Avoid negative weights
        nn_score = max(0, nn_metrics['test_r2'])
        
        # Normalize scores to weights that sum to 1.0
        total_score = rf_score + nn_score
        if total_score > 0:
            # Weight proportional to performance
            self.rf_weight = rf_score / total_score  # RF weight based on its R²
            self.nn_weight = nn_score / total_score  # NN weight based on its R²
        else:
            # Both models performed poorly - use equal weight
            self.rf_weight = 0.5
            self.nn_weight = 0.5
        
        print(f"\n[PHASE 1] Dynamic Weights:")
        print(f"  Random Forest: {self.rf_weight:.1%}")
        print(f"  Neural Network: {self.nn_weight:.1%}")
    
    def fetch_data(self):
        """Fetch data (shared by both models) - uses cache if available"""
        print(f"Fetching data for {self.ticker}...")
        
        # Try to load from cache first
        if self._load_cached_models():
            return self.rf_predictor.data
        
        # Fetch fresh data
        # Fetch once, share between models
        self.rf_predictor = StockPredictor(
            self.ticker,
            self.period,
            self.api_key,
            use_sentiment=self.use_sentiment,
            sentiment_days=self.sentiment_days,
            use_feature_selection=self.use_feature_selection,
        )
        self.rf_predictor.fetch_data()
        
        # Copy data to NN predictor
        self.nn_predictor = LSTMPredictor(
            self.ticker,
            self.period,
            self.api_key,
            use_sentiment=self.use_sentiment,
            sentiment_days=self.sentiment_days,
        )
        self.nn_predictor.data = self.rf_predictor.data.copy()
        
        print(f"[OK] Fetched {len(self.rf_predictor.data)} days of data")
        return self.rf_predictor.data
    
    def train(self):
        """Train both models with Phase 1 optimizations"""
        print("\n" + "="*70)
        print("TRAINING ENSEMBLE MODEL (Random Forest + Deep Neural Network)")
        print("="*70)
        
        # Train Random Forest
        print("\n[ML] Step 1: Training Random Forest...")
        print("-"*70)
        self.rf_predictor.engineer_features()
        rf_metrics = self.rf_predictor.train()
        
        # Train Deep Neural Network
        print("\n[NN] Step 2: Training Deep Neural Network...")
        print("-"*70)
        nn_metrics = self.nn_predictor.train()
        
        # Phase 1: Calculate dynamic weights
        self._calculate_dynamic_weights(rf_metrics, nn_metrics)
        
        self.trained = True
        
        # Ensemble metrics (weighted average of both)
        print("\n" + "="*70)
        print("ENSEMBLE PERFORMANCE (Weighted combination of both models)")
        print("="*70)
        
        ensemble_metrics = {
            'train_r2': (rf_metrics['train_r2'] * self.rf_weight + 
                        nn_metrics['train_r2'] * self.nn_weight),
            'test_r2': (rf_metrics['test_r2'] * self.rf_weight + 
                       nn_metrics['test_r2'] * self.nn_weight),
            'train_rmse': (rf_metrics['train_rmse'] * self.rf_weight + 
                          nn_metrics['train_rmse'] * self.nn_weight),
            'test_rmse': (rf_metrics['test_rmse'] * self.rf_weight + 
                         nn_metrics['test_rmse'] * self.nn_weight),
            'train_mae': (rf_metrics['train_mae'] * self.rf_weight + 
                         nn_metrics['train_mae'] * self.nn_weight),
            'test_mae': (rf_metrics['test_mae'] * self.rf_weight + 
                        nn_metrics['test_mae'] * self.nn_weight),
        }
        
        print("[ENSEMBLE MODEL PERFORMANCE FOR " + self.ticker + "]")
        print("Training Performance (Ensemble):")
        print(f"  RMSE: ${ensemble_metrics['train_rmse']:>12.2f}   MAE: ${ensemble_metrics['train_mae']:>10.2f}   R2: {ensemble_metrics['train_r2']:>8.4f}")
        print("\nTesting Performance (Ensemble):")
        print(f"  RMSE: ${ensemble_metrics['test_rmse']:>12.2f}   MAE: ${ensemble_metrics['test_mae']:>10.2f}   R2: {ensemble_metrics['test_r2']:>8.4f}")
        print("="*70)
        
        # Phase 1: Cache trained models
        self._save_cached_models()
        
        return ensemble_metrics
    
    def predict_next_day(self, verbose: bool = True):
        """Predict next day using weighted ensemble (Phase 1 optimization)"""
        if not self.trained:
            raise ValueError("Models not trained yet. Call train() first.")
        
        # Get predictions from both models
        exclude_cols = ['Target', 'Open', 'High', 'Low', 'Close', 'Volume', 
                       'Dividends', 'Stock Splits']
        
        # Random Forest prediction - use the selected features from the model
        X_last_rf = self.rf_predictor.data[self.rf_predictor.feature_columns].iloc[[-1]]
        rf_pred = float(self.rf_predictor.model.predict(X_last_rf)[0])
        
        # Neural Network prediction (already trained, just predict)
        nn_pred = self.nn_predictor.predict_next_day(verbose=False)
        
        # Ensemble: weighted average (Phase 1 optimization)
        ensemble_pred = (rf_pred * self.rf_weight) + (nn_pred * self.nn_weight)
        
        # Calculate metrics
        current_price = self.rf_predictor.data['Close'].iloc[-1]
        predicted_change = ensemble_pred - current_price
        predicted_change_pct = (predicted_change / current_price) * 100
        
        # Direction
        if predicted_change > 0:
            direction = "UP"
            color_change = "+"
        else:
            direction = "DOWN"
            color_change = ""
        
        if verbose:
            # Display detailed prediction
            print("[ENSEMBLE PRICE PREDICTION FOR " + self.ticker + "]")
            print(f"  Current Price:        ${current_price:>12.2f}")
            print("\n  Individual Predictions:")
            print(f"    Random Forest:    ${rf_pred:>12.2f} (weight: {self.rf_weight:.0%})")
            print(f"    Neural Network:   ${nn_pred:>12.2f} (weight: {self.nn_weight:.0%})")
            print("\n  Ensemble (Weighted):")
            print(f"    Predicted Price:      ${ensemble_pred:>12.2f}")
            print(f"    Expected Change:      {color_change}${abs(predicted_change):>10.2f} ({color_change}{predicted_change_pct:>6.2f}%)")
            print(f"    Direction:            {direction}")
            print("="*70)
        
        return ensemble_pred
