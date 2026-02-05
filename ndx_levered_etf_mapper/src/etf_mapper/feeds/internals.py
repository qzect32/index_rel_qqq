from __future__ import annotations

import pandas as pd

from .base import Feed, FeedStatus


class InternalsFeed(Feed):
    """Market internals / breadth (A/D, ticks, TRIN proxies).

    Typically external. Scaffolded here to keep UI modular.
    """

    name = "internals"

    def fetch_internals(self) -> pd.DataFrame:
        """Return dataframe with at least: ts, name, value, source."""
        return pd.DataFrame(columns=["ts", "name", "value", "source"])


class StubInternalsFeed(InternalsFeed):
    def status(self) -> FeedStatus:
        return FeedStatus(ok=False, name=self.name, detail="stub (no internals feed configured)")

    def fetch_internals(self) -> pd.DataFrame:
        return super().fetch_internals()
