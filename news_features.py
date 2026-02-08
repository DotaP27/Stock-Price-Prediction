"""
News-based sentiment feature builder using Alpha Vantage NEWS_SENTIMENT
- Fetches recent headlines for a ticker
- Extracts Alpha Vantage overall_sentiment_score (no extra ML dependency)
- Aggregates to daily sentiment metrics for model features

If the API call fails or no headlines are returned, it gracefully returns an
empty DataFrame so upstream code can continue without sentiment features.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests


class NewsFeatureBuilder:
    """Builds sentiment features from Alpha Vantage news feed."""

    def __init__(self, api_key: Optional[str] = None, rate_limit_seconds: int = 12):
        self.api_key = api_key or os.environ.get("ALPHA_VANTAGE_API_KEY", "demo")
        self.rate_limit_seconds = rate_limit_seconds

    def fetch_news(self, ticker: str, limit: int = 60, days: int = 14) -> pd.DataFrame:
        """Fetch raw news items for a ticker from Alpha Vantage."""
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker.upper(),
            "time_from": (datetime.utcnow() - timedelta(days=days)).strftime("%Y%m%dT%H%M"),
            "sort": "LATEST",
            "limit": limit,
            "apikey": self.api_key,
        }

        try:
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json()
            feed = data.get("feed", []) if isinstance(data, dict) else []
            if not feed:
                return pd.DataFrame()

            records = []
            for item in feed:
                published = item.get("time_published")
                date = pd.to_datetime(published[:8]) if published else None
                score = item.get("overall_sentiment_score")
                relevance = item.get("relevance_score")
                if date is None or score is None:
                    continue
                records.append(
                    {
                        "Date": date.normalize(),
                        "Sentiment_Score": float(score),
                        "Relevance": float(relevance) if relevance is not None else None,
                    }
                )

            if not records:
                return pd.DataFrame()

            # Respect API rate limit
            time.sleep(self.rate_limit_seconds)
            return pd.DataFrame(records)

        except Exception:
            return pd.DataFrame()

    def build_daily_features(self, ticker: str, days: int = 14, limit: int = 60) -> pd.DataFrame:
        """Return aggregated daily sentiment features for the ticker."""
        raw = self.fetch_news(ticker, limit=limit, days=days)
        if raw.empty:
            return pd.DataFrame()

        grouped = raw.groupby("Date")
        daily = pd.DataFrame(
            {
                "Sentiment_Mean": grouped["Sentiment_Score"].mean(),
                "Sentiment_Median": grouped["Sentiment_Score"].median(),
                "Sentiment_Min": grouped["Sentiment_Score"].min(),
                "Sentiment_Max": grouped["Sentiment_Score"].max(),
                "Sentiment_Std": grouped["Sentiment_Score"].std().fillna(0),
                "Headline_Count": grouped.size(),
                "Relevance_Mean": grouped["Relevance"].mean(),
            }
        ).reset_index()

        # Momentum: last 3-day average minus previous 3-day average
        daily = daily.sort_values("Date")
        daily["Sentiment_3d_Mean"] = daily["Sentiment_Mean"].rolling(window=3).mean()
        daily["Sentiment_3d_Momentum"] = daily["Sentiment_3d_Mean"] - daily["Sentiment_3d_Mean"].shift(3)

        # Fill small gaps with neutral values
        daily = daily.fillna(0)
        return daily
