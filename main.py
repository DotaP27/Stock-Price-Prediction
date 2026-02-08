"""
Stock Price Prediction - Single Unified Mode
Phase 1 Optimizations: Feature Selection + Weighted Ensemble + Model Caching
Simple, intuitive interface for predicting any stock
"""

from ensemble_predictor import EnsemblePredictor
import warnings
import os
import sys
from datetime import datetime

warnings.filterwarnings('ignore')

# Get API key from environment or use default
API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "4R6LA9IDIKTTY6IT")

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    
    @staticmethod
    def disable():
        Colors.HEADER = Colors.BLUE = Colors.CYAN = Colors.GREEN = Colors.YELLOW = Colors.RED = Colors.BOLD = Colors.UNDERLINE = Colors.END = ''


def main():
    """Single unified mode for interactive stock price prediction
    
    This is the main entry point that guides users through:
    1. Stock selection (any ticker symbol)
    2. Time period selection (6mo to 5y of historical data)
    3. Optional sentiment analysis (from news headlines)
    4. Model training (ensemble of Random Forest + Neural Network)
    5. Price prediction and interpretation
    """
    
    # ════════════════════════════════════════════════════════════════════════════════════════
    # WELCOME BANNER - Display system overview
    # ════════════════════════════════════════════════════════════════════════════════════════
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}")
    print(f"  📈 STOCK PRICE PREDICTION SYSTEM 📈".center(70))
    print(f"  Ensemble: Random Forest + Neural Network".center(70))
    print(f"  Phase 1: Feature Selection + Model Caching".center(70))
    print(f"{'='*70}{Colors.END}\n")
    
    # ════════════════════════════════════════════════════════════════════════════════════════
    # STEP 1: STOCK SELECTION - Any publicly traded ticker works
    # ════════════════════════════════════════════════════════════════════════════════════════
    print(f"{Colors.BOLD}📊 STEP 1: SELECT STOCK{Colors.END}")
    print(f"{Colors.BLUE}Enter the stock ticker you want to predict:{Colors.END}")
    print(f"  Examples: {Colors.GREEN}AAPL{Colors.END} (Apple), {Colors.GREEN}NVDA{Colors.END} (Nvidia), {Colors.GREEN}TSLA{Colors.END} (Tesla),")
    print(f"           {Colors.GREEN}GOOGL{Colors.END} (Google), {Colors.GREEN}MSFT{Colors.END} (Microsoft), {Colors.GREEN}AMZN{Colors.END} (Amazon)")
    ticker = input(f"{Colors.CYAN}\n> Stock ticker (default=AAPL): {Colors.END}").strip().upper() or "AAPL"
    
    # Validate ticker
    if not ticker.isalpha():
        print(f"{Colors.RED}✗ ERROR: Invalid ticker '{ticker}'{Colors.END}")
        return 1
    print(f"{Colors.GREEN}✓ Selected: {ticker}{Colors.END}")
    
    # ════════════════════════════════════════════════════════════════════════════════════════
    # STEP 2: TIME PERIOD SELECTION - More data = better patterns but slower
    # ════════════════════════════════════════════════════════════════════════════════════════
    print(f"\n{Colors.BOLD}⏱️  STEP 2: SELECT TIME PERIOD{Colors.END}")
    print(f"{Colors.BLUE}Select time period for historical data:{Colors.END}")
    print(f"  {Colors.GREEN}1{Colors.END}. 6 months  (recent trends only, faster training)")
    print(f"  {Colors.GREEN}2{Colors.END}. 1 year    (recommended - good balance) ⭐")
    print(f"  {Colors.GREEN}3{Colors.END}. 2 years   (includes multiple market cycles)")
    print(f"  {Colors.GREEN}4{Colors.END}. 5 years   (long-term trend analysis)")
    
    period_choice = input(f"{Colors.CYAN}\n> Choice (default=2): {Colors.END}").strip() or "2"
    period_map = {"1": "6mo", "2": "1y", "3": "2y", "4": "5y"}
    period = period_map.get(period_choice, "1y")
    print(f"{Colors.GREEN}✓ Selected: {period}{Colors.END}")
    
    # ════════════════════════════════════════════════════════════════════════════════════════
    # STEP 3: SENTIMENT ANALYSIS OPTION - Optional market psychology input
    # ════════════════════════════════════════════════════════════════════════════════════════
    print(f"\n{Colors.BOLD}💬 STEP 3: SENTIMENT ANALYSIS (OPTIONAL){Colors.END}")
    print(f"{Colors.BLUE}Include news sentiment analysis in predictions?{Colors.END}")
    print(f"  {Colors.YELLOW}→ Analyzes recent headlines for market sentiment")
    print(f"  → Helpful for volatile stocks but requires more API calls{Colors.END}")
    
    sentiment_choice = input(f"{Colors.CYAN}\n> Use sentiment analysis? (y/n, default=y): {Colors.END}").strip().lower()
    use_sentiment = sentiment_choice != 'n'
    print(f"{Colors.GREEN}✓ Sentiment: {'Enabled' if use_sentiment else 'Disabled'}{Colors.END}")
    
    # ════════════════════════════════════════════════════════════════════════════════════════
    # CONFIGURATION SUMMARY - Show what was selected
    # ════════════════════════════════════════════════════════════════════════════════════════
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}")
    print(f"CONFIGURATION SUMMARY".center(70))
    print(f"{'='*70}{Colors.END}")
    print(f"  {Colors.BOLD}Stock Symbol:{Colors.END}       {Colors.GREEN}{ticker}{Colors.END}")
    print(f"  {Colors.BOLD}Historical Period:{Colors.END}  {Colors.GREEN}{period}{Colors.END}")
    print(f"  {Colors.BOLD}Sentiment Analysis:{Colors.END} {Colors.GREEN}{'Enabled' if use_sentiment else 'Disabled'}{Colors.END}")
    print(f"  {Colors.BOLD}Model Type:{Colors.END}         {Colors.CYAN}Ensemble (Random Forest + Neural Network){Colors.END}")
    print(f"  {Colors.BOLD}Features Used:{Colors.END}      {Colors.CYAN}Top 20 selected from 45+ (Phase 1 optimization){Colors.END}")
    print(f"  {Colors.BOLD}Model Caching:{Colors.END}      {Colors.GREEN}Enabled (100x faster on repeat runs){Colors.END}")
    print(f"{Colors.CYAN}{'='*70}{Colors.END}\n")
    
    # ════════════════════════════════════════════════════════════════════════════════════════
    # MODEL TRAINING PIPELINE - Fetch, train, and predict
    # ════════════════════════════════════════════════════════════════════════════════════════
    try:
        print(f"{Colors.BOLD}🔧 INITIALIZING MODEL...{Colors.END}")
        # Create ensemble predictor with configured settings
        predictor = EnsemblePredictor(
            ticker,
            period=period,
            api_key=API_KEY,
            use_sentiment=use_sentiment,
            use_feature_selection=True,      # Phase 1: Feature selection
            use_model_cache=True             # Phase 1: Model caching
        )
        
        print(f"{Colors.BOLD}📥 FETCHING HISTORICAL DATA...{Colors.END}")
        # Fetch price data from Alpha Vantage (will use cache if available)
        predictor.fetch_data()
        
        print(f"{Colors.BOLD}🤖 TRAINING ENSEMBLE MODEL...{Colors.END}")
        # Train both Random Forest and Neural Network models
        metrics = predictor.train()
        
        print(f"\n{Colors.BOLD}🎯 GENERATING PREDICTION...{Colors.END}")
        prediction = predictor.predict_next_day(verbose=True)
        
        # Summary
        print(f"\n{Colors.BOLD}{Colors.GREEN}{'='*70}")
        print(f"📊 PREDICTION RESULTS".center(70))
        print(f"{'='*70}{Colors.END}\n")
        
        print(f"  {Colors.BOLD}Stock:{Colors.END}                 {Colors.CYAN}{ticker}{Colors.END}")
        print(f"  {Colors.BOLD}Ensemble R² (test):{Colors.END}     {Colors.YELLOW}{metrics['test_r2']:.4f}{Colors.END}")
        print(f"  {Colors.BOLD}Model Weights:{Colors.END}         {Colors.GREEN}RF {predictor.rf_weight:.0%}{Colors.END} | {Colors.BLUE}NN {predictor.nn_weight:.0%}{Colors.END}")
        print(f"  {Colors.BOLD}Features Used:{Colors.END}         {Colors.CYAN}20 (from 45 total){Colors.END}")
        print(f"  {Colors.BOLD}Cache Status:{Colors.END}          {Colors.GREEN}✓ Enabled{Colors.END}")
        print(f"  {Colors.BOLD}Timestamp:{Colors.END}            {Colors.BLUE}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.END}")
        print(f"\n{Colors.BOLD}{Colors.GREEN}{'='*70}{Colors.END}")
        
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ SUCCESS!{Colors.END} Prediction complete!")
        print(f"{Colors.YELLOW}→ Models cached for instant predictions on next run!{Colors.END}\n")
        
        return 0
        
    except Exception as e:
        print(f"\n{Colors.RED}{Colors.BOLD}✗ ERROR: {e}{Colors.END}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
