from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class FeedStatus:
    ok: bool
    name: str
    detail: str = ""


class Feed:
    """Base class for non-broker feeds (halts/news/calendar/earnings/etc).

    This repo is Schwab-only for broker/marketdata, but we *scaffold* external intel feeds behind
    interfaces so we can wire them later without rewriting the UI.
    """

    name: str = "feed"

    def status(self) -> FeedStatus:
        return FeedStatus(ok=False, name=self.name, detail="not configured")


class HaltsFeed(Feed):
    name = "halts"

    def fetch_halts(self) -> pd.DataFrame:
        """Return a dataframe with at least: symbol, market, reason, halt_time, resume_time."""
        return pd.DataFrame(columns=["symbol", "market", "reason", "halt_time", "resume_time", "source"])


class NewsFeed(Feed):
    name = "news"

    def fetch_headlines(self, symbols: Optional[list[str]] = None) -> pd.DataFrame:
        """Return a dataframe with at least: ts, symbol(optional), headline, url, source."""
        return pd.DataFrame(columns=["ts", "symbol", "headline", "url", "source"])


class CalendarFeed(Feed):
    name = "calendar"

    def fetch_events(self) -> pd.DataFrame:
        """Return a dataframe with at least: ts, event_type, title, url(optional), source."""
        return pd.DataFrame(columns=["ts", "event_type", "title", "url", "source"])


class EarningsFeed(Feed):
    name = "earnings"

    def fetch_earnings(self) -> pd.DataFrame:
        """Return a dataframe with at least: symbol, report_date, call_time(optional), url(optional), source."""
        return pd.DataFrame(columns=["symbol", "report_date", "call_time", "url", "source"])


class FilingsFeed(Feed):
    name = "filings"

    def fetch_filings(self, symbol: str) -> pd.DataFrame:
        """Return a dataframe with at least: form, filed_at, url, source."""
        return pd.DataFrame(columns=["form", "filed_at", "url", "source"])
