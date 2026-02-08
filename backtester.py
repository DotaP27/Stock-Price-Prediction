"""
Backtesting Module - Test model performance on historical data
Shows how well the model would have performed in the past
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from stock_predictor import StockPredictor


class BacktestEngine:
    """Test model predictions against actual historical prices"""
    
    def __init__(self, ticker: str, period: str = "compact", use_sentiment: bool = False, api_key: str = None):
        """
        Initialize backtester
        
        Args:
            ticker: Stock symbol (e.g., 'AAPL')
            period: Data period ('compact' for 100 days, '1y' for 1 year)
            use_sentiment: Whether to include news sentiment features
            api_key: Alpha Vantage API key
        """
        self.ticker = ticker
        self.period = period
        self.use_sentiment = use_sentiment
        self.predictor = StockPredictor(ticker, period, api_key=api_key, use_sentiment=use_sentiment)
        self.results = []
        self.trades = []
        
    def run_backtest(self, train_window: int = 50, test_window: int = 10):
        """Run a rolling window backtest to evaluate model performance
        
        This simulates real trading by:
        1. Training model on historical data
        2. Making predictions on unseen future data
        3. Comparing predictions to actual prices
        
        This approach is more realistic than training on all data, as it
        avoids "looking into the future" which real trading cannot do.
        
        Args:
            train_window: Number of days to train the model (default: 50)
            test_window: Number of days to test predictions (default: 10)
        
        Returns:
            DataFrame with detailed backtest results including:
            - Predicted vs actual prices
            - Prediction errors
            - Direction accuracy (up/down)
        """
        print(f"\nRunning backtest for {self.ticker}...")
        print(f"   Train window: {train_window} days | Test window: {test_window} days\n")
        
        # Fetch data once
        self.predictor.fetch_data()
        full_data = self.predictor.data.copy()
        
        if len(full_data) < train_window + test_window:
            print(f"❌ Insufficient data: {len(full_data)} days (need {train_window + test_window})")
            return pd.DataFrame()
        
        # Reset results
        self.results = []
        self.trades = []
        
        # Train on first N days
        train_data = full_data.iloc[:train_window].copy()
        test_data = full_data.iloc[train_window:train_window+test_window].copy()
        
        self.predictor.data = train_data
        self.predictor.engineer_features()
        
        try:
            self.predictor.train()
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return pd.DataFrame()
        
        # Make predictions for test window
        print(f"Making {len(test_data)-1} predictions on test data...\n")
        
        for i in range(len(test_data) - 1):
            try:
                last_close = test_data.iloc[i]['Close']
                actual_next = test_data.iloc[i + 1]['Close']
                
                # Get last feature row to make prediction
                if len(self.predictor.data) > 0 and self.predictor.feature_columns:
                    # Get the last row of engineered features
                    X_last = self.predictor.data[self.predictor.feature_columns].iloc[[-1]]
                    predicted_price = float(self.predictor.model.predict(X_last)[0])
                else:
                    continue
                
                # Calculate metrics
                error = abs(predicted_price - actual_next)
                error_pct = (error / actual_next) * 100 if actual_next != 0 else 0
                direction_correct = (predicted_price > last_close) == (actual_next > last_close)
                
                # Store result
                result = {
                    'date': str(test_data.index[i])[:10],
                    'actual_price': actual_next,
                    'predicted_price': predicted_price,
                    'error': error,
                    'error_pct': error_pct,
                    'direction_correct': direction_correct
                }
                self.results.append(result)
                
                # Track trades (BUY when price expected to go up)
                if predicted_price > last_close:
                    profit_loss = actual_next - last_close
                    profit_loss_pct = (profit_loss / last_close) * 100 if last_close != 0 else 0
                    self.trades.append({
                        'action': 'BUY',
                        'entry_price': last_close,
                        'exit_price': actual_next,
                        'profit_loss': profit_loss,
                        'profit_loss_pct': profit_loss_pct,
                        'correct': actual_next > last_close
                    })
                
            except Exception as e:
                continue
        
        return pd.DataFrame(self.results)
    
    def get_accuracy_metrics(self):
        """Calculate comprehensive backtest performance metrics
        
        Evaluates both direction prediction accuracy and trading profitability.
        
        Returns:
            Dictionary containing:
            - total_predictions: Number of predictions made
            - correct_direction: How many correctly predicted up/down
            - accuracy: Percentage of correct directions
            - avg_error: Average price error in dollars
            - avg_error_pct: Average error as percentage
            - win_rate: Percentage of winning trades
            - total_trades: Number of BUY trades executed
            - total_profit: Cumulative profit/loss in dollars
            - total_profit_pct: Cumulative return percentage
        """
        if not self.results:
            return None
        
        results_df = pd.DataFrame(self.results)
        
        # Calculate metrics
        total_predictions = len(results_df)
        correct_direction = results_df['direction_correct'].sum()
        accuracy = (correct_direction / total_predictions * 100) if total_predictions > 0 else 0
        
        avg_error = results_df['error'].mean()
        avg_error_pct = results_df['error_pct'].mean()
        
        # Trading metrics
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            winning_trades = trades_df[trades_df['correct']].shape[0]
            total_trades = len(trades_df)
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            total_profit = trades_df['profit_loss'].sum()
            total_profit_pct = trades_df['profit_loss_pct'].sum()
        else:
            win_rate = 0
            total_profit = 0
            total_profit_pct = 0
        
        metrics = {
            'total_predictions': total_predictions,
            'correct_direction': correct_direction,
            'accuracy': accuracy,
            'avg_error': avg_error,
            'avg_error_pct': avg_error_pct,
            'win_rate': win_rate,
            'total_trades': len(self.trades),
            'total_profit': total_profit,
            'total_profit_pct': total_profit_pct
        }
        
        return metrics
    
    def print_backtest_report(self):
        """Print professional backtest report"""
        metrics = self.get_accuracy_metrics()
        
        if not metrics:
            print("No backtest results available")
            return
        
        print("\n" + "╔" + "═"*78 + "╗")
        print("║" + f"  BACKTEST RESULTS FOR {self.ticker}".ljust(79) + "║")
        print("╠" + "═"*78 + "╣")
        print("║" + " "*78 + "║")
        
        # Prediction accuracy
        print("║" + "  PREDICTION ACCURACY:".ljust(79) + "║")
        print("║" + f"    • Total Predictions:        {metrics['total_predictions']:>5}  predictions".ljust(79) + "║")
        print("║" + f"    • Correct Direction:        {metrics['correct_direction']:>5}  ({metrics['accuracy']:>6.2f}%)".ljust(79) + "║")
        print("║" + f"    • Average Price Error:      ${metrics['avg_error']:>10.2f}  ({metrics['avg_error_pct']:>6.2f}%)".ljust(79) + "║")
        
        print("║" + " "*78 + "║")
        
        # Trading performance
        print("║" + "  TRADING PERFORMANCE:".ljust(79) + "║")
        print("║" + f"    • Total Trades:             {metrics['total_trades']:>5}  trades".ljust(79) + "║")
        print("║" + f"    • Win Rate:                  {metrics['win_rate']:>6.2f}%".ljust(79) + "║")
        print("║" + f"    • Total P&L:                 ${metrics['total_profit']:>10.2f}  ({metrics['total_profit_pct']:>7.2f}%)".ljust(79) + "║")
        
        print("║" + " "*78 + "║")
        print("╚" + "═"*78 + "╝")
        
        # Interpretation
        print("\n📈 INTERPRETATION:")
        if metrics['accuracy'] > 55:
            print(f"  [+] Direction prediction is ABOVE average ({metrics['accuracy']:.1f}% vs 50% random)")
        else:
            print(f"  [!] Direction prediction is BELOW average ({metrics['accuracy']:.1f}% vs 50% random)")
        
        if metrics['win_rate'] > 50:
            print(f"  [+] Trading strategy is PROFITABLE (Win rate: {metrics['win_rate']:.1f}%)")
        else:
            print(f"  [!] Trading strategy shows losses (Win rate: {metrics['win_rate']:.1f}%)")
        
        if metrics['avg_error_pct'] < 3:
            print(f"  [+] Price predictions are ACCURATE (Avg error: {metrics['avg_error_pct']:.2f}%)")
        else:
            print(f"  [!] Price predictions have LARGER errors (Avg error: {metrics['avg_error_pct']:.2f}%)")
        
        print("\nNOTES:")
        print("  • Limited data (100 days) may affect backtest reliability")
        print("  • Past performance does not guarantee future results")
        print("  • Always do your own research before trading")
        
        return metrics


def run_multiple_backtests(tickers: list, train_window: int = 50, test_window: int = 10):
    """
    Run backtest for multiple stocks
    
    Args:
        tickers: List of stock symbols
        train_window: Days to train model
        test_window: Days to test
    
    Returns:
        Dictionary of backtest results
    """
    print("\n" + "="*80)
    print("MULTI-STOCK BACKTEST ANALYSIS")
    print("="*80)
    
    all_results = {}
    
    for ticker in tickers:
        print(f"\n\n{'─'*80}")
        print(f"Testing {ticker}...")
        print(f"{'─'*80}")
        
        backtester = BacktestEngine(ticker)
        backtester.run_backtest(train_window, test_window)
        backtester.print_backtest_report()
        
        all_results[ticker] = backtester.get_accuracy_metrics()
    
    # Summary table
    print("\n" + "="*80)
    print("BACKTEST SUMMARY")
    print("="*80 + "\n")
    
    print(f"{'Stock':<8} {'Accuracy':<12} {'Trades':<10} {'Win Rate':<12} {'Total P&L':<15}")
    print("─"*80)
    
    for ticker, metrics in all_results.items():
        if metrics:
            print(f"{ticker:<8} {metrics['accuracy']:>6.2f}%       {metrics['total_trades']:>6}      "
                  f"{metrics['win_rate']:>6.2f}%      ${metrics['total_profit']:>10.2f}")
    
    print()
    
    return all_results
