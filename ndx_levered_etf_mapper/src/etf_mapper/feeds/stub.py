from __future__ import annotations

import pandas as pd

from .base import HaltsFeed, NewsFeed, CalendarFeed, EarningsFeed, FilingsFeed, FeedStatus


class StubHaltsFeed(HaltsFeed):
    def status(self) -> FeedStatus:
        return FeedStatus(ok=False, name=self.name, detail="stub (paste/manual only)")

    def fetch_halts(self) -> pd.DataFrame:
        return super().fetch_halts()


class StubNewsFeed(NewsFeed):
    def status(self) -> FeedStatus:
        return FeedStatus(ok=False, name=self.name, detail="stub (paste/manual only)")

    def fetch_headlines(self, symbols=None) -> pd.DataFrame:  # type: ignore[override]
        return super().fetch_headlines(symbols=symbols)


class StubCalendarFeed(CalendarFeed):
    def status(self) -> FeedStatus:
        return FeedStatus(ok=False, name=self.name, detail="stub (paste/manual only)")

    def fetch_events(self) -> pd.DataFrame:
        return super().fetch_events()


class StubEarningsFeed(EarningsFeed):
    def status(self) -> FeedStatus:
        return FeedStatus(ok=False, name=self.name, detail="stub (paste/manual only)")

    def fetch_earnings(self) -> pd.DataFrame:
        return super().fetch_earnings()


class StubFilingsFeed(FilingsFeed):
    def status(self) -> FeedStatus:
        return FeedStatus(ok=False, name=self.name, detail="stub (not wired)")

    def fetch_filings(self, symbol: str) -> pd.DataFrame:
        return super().fetch_filings(symbol)
