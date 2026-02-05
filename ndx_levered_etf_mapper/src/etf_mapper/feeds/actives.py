from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .base import Feed, FeedStatus


class ActivesFeed(Feed):
    """Market-wide actives/movers universe.

    Used to power Scanner without manually curated symbol lists.
    """

    name = "actives"

    def fetch_actives(self) -> pd.DataFrame:
        """Return dataframe with at least: symbol, rank, dollar_vol(optional), volume(optional), chg_pct(optional), source."""
        return pd.DataFrame(columns=["symbol", "rank", "dollar_vol", "volume", "chg_pct", "source"])


class StubActivesFeed(ActivesFeed):
    def status(self) -> FeedStatus:
        return FeedStatus(ok=False, name=self.name, detail="stub (no market-wide actives source configured)")

    def fetch_actives(self) -> pd.DataFrame:
        return super().fetch_actives()
