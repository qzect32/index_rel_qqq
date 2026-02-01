from __future__ import annotations

from typing import Optional
import pandas as pd
import yfinance as yf

from .base import PriceHistoryResult


class YahooPriceProvider:
    name = "yahoo"

    def fetch_daily_bars(
        self,
        ticker: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> PriceHistoryResult:
        df = yf.download(
            tickers=ticker,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if df is None or df.empty:
            return PriceHistoryResult(prices=pd.DataFrame())

        df = df.reset_index().rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )

        df["ticker"] = str(ticker).upper().strip()
        df["source"] = "yahoo:yfinance"

        # Normalize date to ISO string (safe for sqlite)
        if pd.api.types.is_datetime64_any_dtype(df["date"]):
            df["date"] = df["date"].dt.date.astype(str)
        else:
            df["date"] = df["date"].astype(str)

        keep = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume", "source"]
        for c in keep:
            if c not in df.columns:
                df[c] = None

        return PriceHistoryResult(prices=df[keep])
