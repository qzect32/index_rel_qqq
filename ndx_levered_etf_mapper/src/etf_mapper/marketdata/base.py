from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Optional
import pandas as pd


@dataclass(frozen=True)
class PriceHistoryResult:
    prices: pd.DataFrame  # columns: date, ticker, open, high, low, close, volume, adj_close(optional), source


class PriceProvider(Protocol):
    name: str

    def fetch_daily_bars(
        self,
        ticker: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> PriceHistoryResult: ...
