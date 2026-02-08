"""
Data Management Module
Handles fetching, caching, and preprocessing of stock data for multiple tickers
"""

import pandas as pd
from alpha_vantage.timeseries import TimeSeries
import requests
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
import time


class DataManager:
    """Manages stock data fetching and caching"""
    
    def __init__(self, cache_dir: str = "data/cache", api_key: str = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Alpha Vantage API key - get free key at https://www.alphavantage.co/support/#api-key
        self.api_key = api_key or os.environ.get('ALPHA_VANTAGE_API_KEY', 'demo')
        if self.api_key == 'demo':
            print("Using demo API key. Get your free key at: https://www.alphavantage.co/support/#api-key")
            print("Set it as: ALPHA_VANTAGE_API_KEY environment variable")
        self.ts = TimeSeries(key=self.api_key, output_format='pandas')
    
    def get_cache_path(self, ticker: str, period: str) -> Path:
        """Get the cache file path for a ticker"""
        filename = f"{ticker}_{period}_{datetime.now().strftime('%Y%m%d')}.csv"
        return self.cache_dir / filename
    
    def is_cache_valid(self, ticker: str, period: str, max_age_hours: int = 24) -> bool:
        """Check if cached data exists and is recent"""
        cache_path = self.get_cache_path(ticker, period)
        
        if not cache_path.exists():
            return False
        
        # Check age
        file_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age = datetime.now() - file_time
        
        return age.total_seconds() / 3600 < max_age_hours
    
    def fetch_stock_data(self, ticker: str, period: str = "2y", 
                        use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch stock data with caching support
        
        Args:
            ticker: Stock symbol
            period: Historical period ('1mo', '3mo', '6mo', '1y', '2y', 'compact', 'full')
            use_cache: Whether to use cached data if available
        """
        ticker = ticker.upper()
        
        # Check cache
        if use_cache and self.is_cache_valid(ticker, period):
            cache_path = self.get_cache_path(ticker, period)
            print(f"Loading {ticker} from cache...")
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            return df
        
        # Map period to Alpha Vantage outputsize
        # Free tier: compact=100 days, full=20+ years (but uses API calls)
        outputsize = 'full' if period in ['2y', '5y', 'full', 'max'] else 'compact'
        # If using demo/free key, avoid premium full outputsize
        if self.api_key == 'demo' and outputsize == 'full':
            outputsize = 'compact'
            print("Using compact outputsize to stay within free tier")
        
        # Fetch from Alpha Vantage (ASCII only for Windows consoles)
        print(f"Fetching {ticker} from Alpha Vantage...")
        try:
            # Get daily data
            data, meta_data = self.ts.get_daily(symbol=ticker, outputsize=outputsize)
            
            if data.empty:
                raise ValueError(f"No data found for {ticker}")
            
            # Rename columns to match yfinance format
            df = data.copy()
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            df = df.sort_index()
            
            # Add missing columns for compatibility
            df['Dividends'] = 0
            df['Stock Splits'] = 0
            
            # Filter by period if needed
            period_days = {'1mo': 30, '3mo': 90, '6mo': 180, '1y': 365, '2y': 730, '5y': 1825}
            if period in period_days:
                cutoff_date = datetime.now() - timedelta(days=period_days[period])
                df = df[df.index >= cutoff_date]
            
            if df.empty:
                raise ValueError(f"No data found for {ticker} in period {period}")
            
            # Save to cache
            cache_path = self.get_cache_path(ticker, period)
            df.to_csv(cache_path)
            print(f"Cached data for {ticker}")
            
            # Alpha Vantage rate limit (5 calls/min for free tier)
            time.sleep(12)  # Wait 12 seconds between calls
            
            return df
            
        except Exception as e:
            raise ValueError(f"Failed to fetch {ticker}: {str(e)}")
    
    def fetch_multiple_stocks(self, tickers: list, period: str = "2y", 
                            use_cache: bool = True) -> dict:
        """
        Fetch data for multiple stocks
        
        Returns:
            Dictionary with ticker as key and DataFrame as value
        """
        results = {}
        
        for ticker in tickers:
            try:
                results[ticker] = self.fetch_stock_data(ticker, period, use_cache)
            except Exception as e:
                print(f"❌ Error fetching {ticker}: {str(e)}")
                results[ticker] = None
        
        return results
    
    def get_stock_info(self, ticker: str) -> dict:
        """Get company information"""
        try:
            # Alpha Vantage company overview
            url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={self.api_key}'
            response = requests.get(url)
            info = response.json()
            
            if info and 'Symbol' in info:
                return {
                    'name': info.get('Name', ticker),
                    'sector': info.get('Sector', 'Unknown'),
                    'industry': info.get('Industry', 'Unknown'),
                    'market_cap': int(info.get('MarketCapitalization', 0)),
                    'currency': info.get('Currency', 'USD')
                }
        except:
            pass
        return {'name': ticker, 'sector': 'Unknown', 'industry': 'Unknown', 'market_cap': 0, 'currency': 'USD'}
    
    def get_popular_stocks_by_sector(self) -> dict:
        """Get lists of popular stocks organized by sector"""
        return {
            'Technology': [
                'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META',
                'TSLA', 'AMD', 'INTC', 'CRM', 'ADBE'
            ],
            'Finance': [
                'JPM', 'BAC', 'WFC', 'GS', 'MS',
                'C', 'BLK', 'SCHW', 'AXP', 'USB'
            ],
            'Healthcare': [
                'JNJ', 'UNH', 'PFE', 'ABBV', 'TMO',
                'MRK', 'LLY', 'DHR', 'ABT', 'BMY'
            ],
            'Energy': [
                'XOM', 'CVX', 'COP', 'SLB', 'EOG',
                'MPC', 'PSX', 'VLO', 'OXY', 'HAL'
            ],
            'Consumer': [
                'AMZN', 'WMT', 'HD', 'MCD', 'NKE',
                'SBUX', 'TGT', 'LOW', 'TJX', 'DG'
            ],
            'Industrial': [
                'BA', 'CAT', 'GE', 'HON', 'UPS',
                'MMM', 'LMT', 'RTX', 'DE', 'UNP'
            ]
        }
    
    def clear_cache(self, ticker: str = None):
        """Clear cached data"""
        if ticker:
            # Clear specific ticker
            for file in self.cache_dir.glob(f"{ticker}_*.csv"):
                file.unlink()
                print(f"Cleared cache for {ticker}")
        else:
            # Clear all cache
            for file in self.cache_dir.glob("*.csv"):
                file.unlink()
            print("Cleared all cache")


class StockDataPreprocessor:
    """Preprocesses stock data for model training"""
    
    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate stock data
        
        Removes incomplete records and duplicates, ensuring data integrity
        for model training.
        
        Args:
            df: Raw stock price DataFrame
            
        Returns:
            Cleaned DataFrame with valid data only
        """
        df = df.copy()
        
        # Remove any rows with missing values in critical columns
        df = df.dropna()
        
        # Remove duplicate dates (keep first occurrence)
        df = df[~df.index.duplicated(keep='first')]
        
        # Ensure chronological order for time series analysis
        df = df.sort_index()
        
        return df
    
    @staticmethod
    def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features based on calendar information
        
        Creates features that capture market-related calendar effects
        (e.g., Monday effects, quarter-end volatility).
        
        Args:
            df: DataFrame with datetime index
            
        Returns:
            DataFrame with additional temporal features
        """
        df = df.copy()
        
        # ════════════════════════════════════════════════════════════════════════════════════════
        # BASIC TIME COMPONENTS - Extract date parts
        # ════════════════════════════════════════════════════════════════════════════════════════
        
        df['DayOfWeek'] = df.index.dayofweek        # 0=Monday, 6=Sunday
        df['Month'] = df.index.month                # 1-12
        df['Quarter'] = df.index.quarter            # 1-4
        df['DayOfMonth'] = df.index.day             # 1-31
        df['WeekOfYear'] = df.index.isocalendar().week  # Week number in year
        
        # ════════════════════════════════════════════════════════════════════════════════════════
        # MARKET-SPECIFIC PATTERNS - Known calendar effects in finance
        # ════════════════════════════════════════════════════════════════════════════════════════
        
        # Monday Effect: Studies show different price behavior on Mondays
        df['IsMonday'] = (df['DayOfWeek'] == 0).astype(int)
        
        # Friday Effect: Week-ending behavior often differs
        df['IsFriday'] = (df['DayOfWeek'] == 4).astype(int)
        
        # Quarter-End Effect: Increased volatility at quarter ends
        df['IsQuarterEnd'] = df.index.is_quarter_end.astype(int)
        
        # Year-End Effect: Portfolio rebalancing and tax considerations
        df['IsYearEnd'] = df.index.is_year_end.astype(int)
        
        return df
    
    @staticmethod
    def normalize_prices(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
        """Normalize price columns"""
        df = df.copy()
        
        if columns is None:
            columns = ['Open', 'High', 'Low', 'Close']
        
        for col in columns:
            if col in df.columns:
                df[f'{col}_Normalized'] = df[col] / df[col].iloc[0]
        
        return df
    
    @staticmethod
    def detect_outliers(df: pd.DataFrame, column: str = 'Close', 
                       threshold: float = 3.0) -> pd.Series:
        """Detect outliers using z-score method"""
        mean = df[column].mean()
        std = df[column].std()
        z_scores = (df[column] - mean) / std
        
        return abs(z_scores) > threshold
    
    @staticmethod
    def get_data_quality_report(df: pd.DataFrame) -> dict:
        """Generate data quality report"""
        return {
            'total_rows': len(df),
            'date_range': f"{df.index.min().date()} to {df.index.max().date()}",
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': df.index.duplicated().sum(),
            'columns': list(df.columns)
        }


if __name__ == "__main__":
    # Example usage
    manager = DataManager()
    
    # Fetch single stock
    print("Fetching Apple stock data...")
    aapl_data = manager.fetch_stock_data('AAPL', period='1y')
    print(f"Fetched {len(aapl_data)} days of data")
    
    # Note: Free tier has rate limits (5 calls/min)
    print("\nFetching single stock (rate limit applies)...")
    stocks = manager.fetch_multiple_stocks(['AAPL'], period='6mo')
    
    for ticker, data in stocks.items():
        if data is not None:
            print(f"{ticker}: {len(data)} days")
    
    # Get stock info
    info = manager.get_stock_info('AAPL')
    print(f"\n{info['name']}")
    print(f"Sector: {info['sector']}")
    print(f"Industry: {info['industry']}")
    
    # Preprocess data
    preprocessor = StockDataPreprocessor()
    clean_data = preprocessor.clean_data(aapl_data)
    data_with_dates = preprocessor.add_date_features(clean_data)
    
    print(f"\nData quality report:")
    report = preprocessor.get_data_quality_report(clean_data)
    print(json.dumps(report, indent=2, default=str))
