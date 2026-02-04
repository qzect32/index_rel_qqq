from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .base import PriceHistoryResult
from ..schwab import SchwabAPI, SchwabConfig


@dataclass(frozen=True)
class SchwabPriceProvider:
    """Daily bars provider via Schwab price history.

    Schwab-backed price history provider (replaces prior free bootstrap sources).
    """

    cfg: SchwabConfig

    name: str = "schwab"

    def _api(self) -> SchwabAPI:
        return SchwabAPI(self.cfg)

    def fetch_daily_bars(self, ticker: str, start: Optional[str] = None, end: Optional[str] = None) -> PriceHistoryResult:
        # Schwab API accepts either (periodType/period) or (startDate/endDate). We'll use dates when provided.
        start_ms = None
        end_ms = None
        if start:
            start_ms = int(pd.Timestamp(start).tz_localize("UTC").timestamp() * 1000)
        if end:
            end_ms = int(pd.Timestamp(end).tz_localize("UTC").timestamp() * 1000)

        js = self._api().price_history(
            ticker,
            period_type="year" if not start_ms else "day",
            period=1 if not start_ms else 10,
            frequency_type="daily",
            frequency=1,
            start_date_ms=start_ms,
            end_date_ms=end_ms,
            need_extended_hours_data=False,
        )

        candles = js.get("candles") or js.get("candles", [])
        if not candles:
            return PriceHistoryResult(prices=pd.DataFrame())

        df = pd.DataFrame(candles)
        # candle schema: open/high/low/close/volume/datetime(ms)
        if "datetime" in df.columns:
            df["date"] = pd.to_datetime(df["datetime"], unit="ms", utc=True).dt.tz_convert(None)
        else:
            df["date"] = pd.NaT

        df["ticker"] = ticker.upper().strip()
        df = df.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"})
        if "open" not in df.columns:
            # Some variants use capitalized keys
            df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})

        keep = [c for c in ["date", "ticker", "open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[keep].copy()
        df["adj_close"] = df.get("close")
        df["source"] = "schwab:pricehistory"
        return PriceHistoryResult(prices=df)
