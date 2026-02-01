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
        """Fetch daily OHLCV via yfinance.

        Implementation note:
          - Use Ticker().history() to avoid MultiIndex columns that can appear with yf.download.
        """
        t = str(ticker).upper().strip()
        hist = yf.Ticker(t).history(start=start, end=end, interval="1d", auto_adjust=False)

        if hist is None or hist.empty:
            return PriceHistoryResult(prices=pd.DataFrame())

        df = hist.reset_index()
        # yfinance uses 'Date' for daily, 'Datetime' for intraday. Normalize both.
        if "Date" in df.columns:
            df = df.rename(columns={"Date": "date"})
        elif "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "date"})

        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )

        df["ticker"] = t
        df["source"] = "yahoo:yfinance"

        # Normalize date to ISO string (safe for sqlite)
        if "date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["date"]):
            df["date"] = df["date"].dt.date.astype(str)
        elif "date" in df.columns:
            df["date"] = df["date"].astype(str)

        keep = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume", "source"]
        for c in keep:
            if c not in df.columns:
                df[c] = None

        return PriceHistoryResult(prices=df[keep])
