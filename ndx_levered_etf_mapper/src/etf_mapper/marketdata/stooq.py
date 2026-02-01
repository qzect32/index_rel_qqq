from __future__ import annotations

from typing import Optional
import pandas as pd
from pandas_datareader import data as pdr

from .base import PriceHistoryResult


class StooqPriceProvider:
    name = "stooq"

    def fetch_daily_bars(
        self,
        ticker: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> PriceHistoryResult:
        # Stooq uses suffixes for some markets; many US tickers work as-is.
        # If you later need it: US stocks can be like 'AAPL.US'. ETFs sometimes work without suffix.
        sym = str(ticker).upper().strip()

        df = pdr.DataReader(sym, "stooq", start=start, end=end)
        if df is None or df.empty:
            return PriceHistoryResult(prices=pd.DataFrame())

        # Stooq returns descending date order; normalize.
        df = df.sort_index().reset_index().rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )

        df["ticker"] = sym
        df["adj_close"] = None
        df["source"] = "stooq:pandas-datareader"

        if pd.api.types.is_datetime64_any_dtype(df["date"]):
            df["date"] = df["date"].dt.date.astype(str)
        else:
            df["date"] = df["date"].astype(str)

        keep = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume", "source"]
        for c in keep:
            if c not in df.columns:
                df[c] = None

        return PriceHistoryResult(prices=df[keep])
